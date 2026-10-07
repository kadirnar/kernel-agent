"""Precision pivots (#96): a plateaued target moves to another precision tier mid-run.

CPU only, no Claude: proposals, checks and the new target's spec (``pivot.py``), the research
session and the round re-plan that propose them, integration and scheduling of the two
arms, and a dry run (``kernel_agent.dryrun``) in which a plateaued exact arm's research plan
proposes ``fp8_weights`` and a new arm in the near-lossless tier appears.
"""

import asyncio
import json

import pytest

from kernel_agent import charts, dryrun, improve, ledger, orchestrator, pivot, research, scheduler
from kernel_agent.agent import prompts
from kernel_agent.agent.runner import AgentResult
from kernel_agent.config import OptimizeConfig
from kernel_agent.dashboard import write_dashboard
from kernel_agent.improve import ImproveConfig, Improver
from kernel_agent.workspace import read_json, write_json


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    """No real toolchain probing and no charts."""
    monkeypatch.setattr(orchestrator.toolchain, "setup", dryrun.SimToolchain)
    monkeypatch.setattr(charts, "available", lambda: False)


def make(tmp_path, seed=0, quality="near-lossless"):
    config = OptimizeConfig(model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path, quality=quality)
    orch = orchestrator.Orchestrator(dryrun.create_run(config, seed), config)
    return orch, dryrun.World(orch)


WHY = "M=1 decode GEMVs stream 2 x 6 MB of bf16 weights: memory bound at 91 % of SOL"


# ------------------------------------------------------------------ pivot.py


def test_ids_and_families():
    assert pivot.pivot_id("dit_layer", "fp8_w8a8") == "dit_layer__fp8_w8a8"
    assert pivot.family("dit_layer__fp8_w8a8") == "dit_layer"
    assert pivot.pivot_id("dit_layer__fp8_weights", "fp8_w8a8") == "dit_layer__fp8_w8a8"
    for plain in ("dit_layer", "a__b", "x__fp9", "__fp8_weights"):
        assert pivot.family(plain) == plain
    assert pivot.label({"precision": "fp8_w8a8", "pivot_of": "dit"}) == "fp8_w8a8 (pivot of dit)"
    assert pivot.label({}) == "exact"


def test_check_and_spec():
    spec = {
        "id": "mlp",
        "module_class": "Qwen3MLP",
        "phase": "decode",
        "why": "w",
        "approach": "GEMV",
        "backends": ["cuda"],
        "capture": {"cases": []},
    }
    ok = {"precision": "fp8_weights", "precision_why": WHY}

    def check(proposal=ok, quality="near-lossless", taken=(), **change):
        return pivot.check({**spec, **change}, proposal, quality=quality, taken=set(taken))

    assert check() is None
    assert "needs --quality near-lossless" in check(quality="exact")
    assert "not one of" in check({"precision": "int4", "precision_why": WHY})
    assert "not one of" in check({"precision": "exact", "precision_why": WHY})
    assert "already" in check(precision="fp8_weights")
    assert "numbers" in check({"precision": "fp8_weights", "precision_why": "it is faster"})
    assert "region" in check(kind="region")
    assert "exists already" in check(taken={"mlp__fp8_weights"})
    assert check(id=None) == "no such target"

    new = pivot.pivot_spec(spec, {**ok, "approach": "FP8 GEMV from the example"})
    assert new == {
        "id": "mlp__fp8_weights",
        "module_class": "Qwen3MLP",
        "phase": "decode",
        "why": "w",
        "approach": "FP8 GEMV from the example",
        "backends": ["cuda"],
        "precision": "fp8_weights",
        "precision_why": WHY,
        "pivot_of": "mlp",
    }  # no capture: the new target is captured afresh
    assert pivot.pivot_spec(spec, ok)["approach"] == "GEMV"


# ------------------------------------------------------------------ orchestrator


def test_orchestrator_pivot_captures_a_new_target_in_its_tier(tmp_path):
    orch, world = make(tmp_path)
    run = orch.run
    research.plan_path(run, "mlp").write_text("# Plan: `mlp`\n\n## Diagnosis\nbound by bytes\n")
    with world.installed():
        done = asyncio.run(
            orch.pivot("mlp", {"precision": "fp8_weights", "precision_why": WHY}, source="t")
        )
    assert done == {"target": "mlp__fp8_weights"}
    spec = read_json(run.target("mlp__fp8_weights") / "spec.json")
    assert spec["precision"] == "fp8_weights" and spec["pivot_of"] == "mlp"
    assert spec["capture"]["tier"] == "near-lossless"  # a fresh capture in the new tier
    assert spec["capture"]["precision"] == "fp8_weights"
    assert "bound by bytes" in (run.target("mlp__fp8_weights") / "plan.md").read_text()
    assert "precision" not in read_json(run.target("mlp") / "spec.json")  # the old one stays
    events = [e for e in ledger.events(run) if e["event"] == "pivot"]
    assert events[-1]["new"] == "mlp__fp8_weights" and events[-1]["source"] == "t"
    # a second proposal of the same pivot, and a pivot in an exact run, are refused
    with world.installed():
        again = asyncio.run(
            orch.pivot("mlp", {"precision": "fp8_weights", "precision_why": WHY}, source="t")
        )
    assert "exists already" in again["refused"]

    exact, world = make(tmp_path / "exact", quality="exact")
    with world.installed():
        refused = asyncio.run(
            exact.pivot("mlp", {"precision": "fp8_weights", "precision_why": WHY}, source="t")
        )
    assert (
        "near-lossless" in refused["refused"] and not exact.run.target("mlp__fp8_weights").exists()
    )
    assert any(e["event"] == "pivot_refused" for e in ledger.events(exact.run))


def test_research_session_may_propose_a_pivot_in_near_lossless_runs(tmp_path):
    for quality in ("near-lossless", "exact"):
        orch, _ = make(tmp_path / quality, quality=quality)
        calls: list[dict] = []

        async def fake(name, calls=calls, **kwargs):
            calls.append(kwargs)
            return AgentResult(name=name)

        orch.agent_runner = fake
        asyncio.run(orch.research("mlp", reason="4 evaluations in a row", label="research-mlp#3"))
        (kw,) = calls
        proposal = pivot.proposal_path(orch.run, "mlp")
        dossier = research.dossier_path(orch.run, "mlp")  # the web tools on (issue #125)
        system = kw["system_append"]
        if quality == "exact":
            assert kw["writable"] == [research.plan_path(orch.run, "mlp"), dossier]
            assert "# Precision pivot" not in system and "pivot.json" not in system
            continue
        assert kw["writable"] == [research.plan_path(orch.run, "mlp"), proposal, dossier]
        assert "# Precision pivot (optional)" in system and str(proposal) in system
        assert "`fp8_w8a8`" in system and "`fp8_weights`" in system  # the other tiers
        assert "mlp__<precision>" in system and "(and `pivot.json` above)" in system


def test_research_evidence_lists_passing_transforms(tmp_path):
    orch, world = make(tmp_path, seed=1)
    with world.installed():
        world._systems()  # simulated transform evaluations
    text = research.transforms_section(orch.run)
    assert "## End-to-end transforms" in text and "| exp | speedup | transforms |" in text
    assert research.transforms_section(make(tmp_path / "none")[0].run) == ""


def test_improver_records_the_pivot_of_a_research_session(tmp_path):
    orch, world = make(tmp_path)
    improver = Improver(orch, ImproveConfig(), require_capture=False, live_charts=False)
    arm = next(a for a in improver.arms() if a.id == "mlp")
    with world.installed():
        rec = asyncio.run(improver._research(arm, "4 evaluations in a row without a new best"))
    assert rec["plan"] and rec["pivot"] == {"target": "mlp__fp8_weights"}
    assert "mlp__fp8_weights" in [a.id for a in improver.arms()]  # a new arm
    state = read_json(orch.run.root / "improve.json")
    assert state["research"][-1]["pivot"] == {"target": "mlp__fp8_weights"}
    other = next(a for a in improver.arms() if a.id == "attn")  # no pivot proposed
    with world.installed():
        assert "pivot" not in asyncio.run(improver._research(other, "4 failed evaluations"))


def test_round_replan_pivots(tmp_path):
    orch, world = make(tmp_path)
    improver = Improver(orch, ImproveConfig(), require_capture=False, live_charts=False)
    round_dir = orch.run.root / "rounds" / "2"
    pivots = [
        {"target": "attn", "precision": "fp8_w8a8", "precision_why": "M=512 prefill, 80 TFLOP"},
        {"target": "nope", "precision": "fp8_weights", "precision_why": WHY},
        {"target": "rmsnorm", "precision": "fp8_weights", "precision_why": "faster"},
    ]
    write_json(round_dir / "plan.json", {"targets": [], "pivots": pivots})
    with world.installed():
        moved = asyncio.run(improver._pivots(round_dir, 2))
    assert moved == ["attn__fp8_w8a8"]  # unknown target ignored, no numbers refused
    spec = read_json(orch.run.target("attn__fp8_w8a8") / "spec.json")
    assert spec["capture"]["tier"] == "near-lossless" and spec["precision"] == "fp8_w8a8"

    arms = improver.arms()
    text = improve.rounds_context(orch.run, improver.state, 2, [], arms, quality="near-lossless")
    assert "## Precision pivots" in text and "`pivots`" in text
    assert "`attn__fp8_w8a8` (`Qwen3Attention`) [fp8_w8a8 (pivot of attn)]: best" in text
    assert "`attn` (`Qwen3Attention`): best" in text
    assert "## Precision pivots" not in improve.rounds_context(
        orch.run, improver.state, 2, [], arms
    )
    schema = prompts.PLAN_SCHEMA["properties"]["pivots"]["items"]
    assert schema["required"] == ["target", "precision", "precision_why"]


def test_integration_and_scheduler_treat_a_pivot_as_another_version():
    key = orchestrator._item_key
    assert key(("kernel", "mlp=/a.py")) == key(("kernel", "mlp__fp8_weights=/b.py"))
    assert key(("kernel", "mlp=/a.py")) != key(("kernel", "attn=/a.py"))
    profiles = [{"shares": {"Qwen3MLP": (0.3, 28)}, "baseline_ms": 100.0, "applied": {}}]
    spec = {"module_class": "Qwen3MLP"}
    assert scheduler._ref_ms("mlp__fp8_weights", spec, profiles) == pytest.approx(30.0)
    profiles[0]["applied"] = {"mlp": 1.5}  # a re-profile with the exact kernel applied
    for target_id in ("mlp", "mlp__fp8_weights"):
        assert scheduler._ref_ms(target_id, spec, profiles) == pytest.approx(45.0)


def test_engineer_prompt_of_a_pivot():
    target = {
        "id": "mlp__fp8_weights",
        "module_class": "Qwen3MLP",
        "why": "w",
        "approach": "a",
        "backends": ["cuda"],
        "precision": "fp8_weights",
        "precision_why": WHY,
        "pivot_of": "mlp",
    }
    near = {"cases": [], "tier": "near-lossless", "precision": "fp8_weights"}
    text = prompts.engineer_prompt(target, near, ["cuda"], "python", "toolchain", 10, None)
    assert "* precision pivot of `mlp`" in text and "`../mlp/`" in text
    assert "# Precision: `fp8_weights`" in text


# ------------------------------------------------------------------ dry run


def test_dry_run_plateaued_exact_arm_pivots_to_fp8_weights(tmp_path):
    orch, world = make(tmp_path, seed=1)
    improver = Improver(
        orch, ImproveConfig(research_every=2), require_capture=False, live_charts=False
    )
    with world.installed():
        reason = asyncio.run(improver.improve())
    run = orch.run
    assert reason.startswith("every arm has stopped")
    state = read_json(run.root / "improve.json")
    (session,) = [r for r in state["research"] if r.get("pivot")]
    # the plateaued exact arm's research session wrote a plan and proposed fp8_weights
    assert session["arm"] == "mlp" and session["plan"] and "without a new best" in session["why"]
    assert session["pivot"] == {"target": "mlp__fp8_weights"}
    proposal = json.loads(pivot.proposal_path(run, "mlp").read_text())
    assert proposal["precision"] == "fp8_weights" and "memory bound" in proposal["precision_why"]

    # a new arm with the reduced tier and a fresh capture
    spec = read_json(run.target("mlp__fp8_weights") / "spec.json")
    assert spec["pivot_of"] == "mlp" and spec["precision"] == "fp8_weights"
    assert spec["capture"]["tier"] == "near-lossless"
    new_rows = [r for r in ledger.rows(run) if r["target"] == "mlp__fp8_weights"]
    assert new_rows and all(r["exp"] > session["exp"] for r in new_rows)
    assert any(s["arm"] == "mlp__fp8_weights" for s in state["slices"])
    assert max(r["speedup"] or 0 for r in new_rows if r["correct"]) > session["best"]

    # the old arm keeps its history: its spec, its rows before the pivot, its own best
    old = read_json(run.target("mlp") / "spec.json")
    assert "precision" not in old and "tier" not in old["capture"]
    old_rows = [r for r in ledger.rows(run) if r["target"] == "mlp"]
    assert sum(r["exp"] <= session["exp"] for r in old_rows) >= 4
    arms = {a.id: a for a in improver.arms()}
    assert arms["mlp"].evals == len(ledger.measured(old_rows)) and arms["mlp__fp8_weights"].evals

    # integration applies one version of the family; ledger, report and dashboard show it
    accepted = [
        a["item"].partition("=")[0] for a in read_json(run.root / "integration.json")["accepted"]
    ]
    assert sum(t in ("mlp", "mlp__fp8_weights") for t in accepted) == 1
    report = run.report.read_text()
    assert (
        "| arm | precision |" in report
        and "| `mlp__fp8_weights` | fp8_weights (pivot of mlp) |" in report
    )
    assert "precision pivot to `mlp__fp8_weights`" in report
    summary = {t["id"]: t for t in ledger.summary(run)["targets"]}
    assert summary["mlp__fp8_weights"]["precision"] == "fp8_weights"
    assert summary["mlp"]["precision"] == "exact"
    page = write_dashboard(run).read_text()
    assert ">precision</th>" in page and "(pivot of <code>mlp</code>)" in page
