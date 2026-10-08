"""Allowed precisions per run (#131): ``--precisions``, no 4-bit unless asked for.

CPU only, no Claude: the option and its default (``precisions.py``, ``run.json``, the CLI),
and every place that enforces it: the planner (policy text, schema enum, dropped targets),
precision pivots, the capture (orchestrator and worker), the library priors, the kernels
phase, the improve scheduler (a stopped arm, the allowed floors of the end-to-end
estimate), the ceilings table's columns, the integration (skipped targets in
``integration.json`` and ``report.md``) and the prompts that mention FP4. The last test
replays the VoxCPM2 latency run whose final integration needed a one-off script to leave
its FP4 targets out: an old run (no ``precisions`` in ``run.json``) now leaves them out by
itself, and ``--precisions`` on an existing run records the new list in ``run.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import dataclasses
from pathlib import Path
from typing import Any

import pytest

from kernel_agent import (
    charts,
    cli,
    dryrun,
    improve,
    ledger,
    library,
    orchestrator,
    pivot,
    precisions,
    scheduler,
    truth,
    worker,
)
from kernel_agent.agent import prompts
from kernel_agent.agent.runner import AgentResult
from kernel_agent.config import OptimizeConfig
from kernel_agent.improve import ImproveConfig, Improver
from kernel_agent.kernels import compare, roofline
from kernel_agent.profiling import ceilings
from kernel_agent.workspace import RunDir, read_json, write_json

NEAR = "near-lossless"
DEFAULT = (  # near-lossless, no 4-bit
    "exact",
    "fp8_weights",
    "reduced",
    "fp8_w8a8",
    "fp8_mx",
    "int8_weights",
    "int8_w8a8",
)
EVERY = tuple(compare.PRECISIONS)  # ... and with fp4_weights asked for
WHY = "M=1 decode GEMVs stream 2 x 6 MB of bf16 weights: memory bound at 91 % of SOL"


@pytest.fixture(autouse=True)
def fast(monkeypatch, tmp_path):
    """No real toolchain probing, no charts, an empty kernel library."""
    monkeypatch.setattr(orchestrator.toolchain, "setup", dryrun.SimToolchain)
    monkeypatch.setattr(charts, "available", lambda: False)
    monkeypatch.setenv(library.ENV, str(tmp_path / "library"))


def make(tmp_path: Path, quality: str = NEAR, allowed: Any = None, seed: int = 0):
    config = OptimizeConfig(
        model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path, quality=quality, precisions=allowed
    )
    orch = orchestrator.Orchestrator(dryrun.create_run(config, seed), config)
    return orch, dryrun.World(orch)


def fp4_target(target_id: str = "lm_fp4", **extra: Any) -> dict[str, Any]:
    return {
        "id": target_id,
        "module_class": "Qwen3MLP",
        "phase": "decode",
        "why": "w",
        "approach": "NVFP4 GEMV",
        "backends": ["cuda"],
        "precision": "fp4_weights",
        "precision_why": WHY,
        **extra,
    }


# ------------------------------------------------------------------ precisions.py


def test_default_parse_check_and_refusal():
    assert precisions.default("exact") == ("exact",) == precisions.default(None)
    assert precisions.default(NEAR) == DEFAULT and "fp4_weights" not in precisions.default(NEAR)
    assert precisions.parse(" FP8_W8A8,fp8_weights ") == ["exact", "fp8_weights", "fp8_w8a8"]
    assert precisions.parse(["fp4_weights"]) == ["exact", "fp4_weights"]
    for bad in ("int4", "", " , "):
        with pytest.raises(ValueError):
            precisions.parse(bad)
    assert (
        precisions.check(NEAR, ["fp4_weights"]) is None and precisions.check("exact", None) is None
    )
    assert "needs --quality near-lossless" in precisions.check("exact", ["fp8_weights"])
    assert precisions.check("exact", ["exact"]) is None
    assert "unknown precision" in precisions.check(NEAR, ["w4"])

    assert precisions.allowed(NEAR) == DEFAULT
    assert precisions.allowed(NEAR, ["fp8_weights", "fp4_weights"]) == (
        "exact",
        "fp8_weights",
        "fp4_weights",
    )
    assert precisions.allowed("exact", ["fp8_weights"]) == ("exact",)  # never reduced
    assert precisions.allowed(NEAR, ["bogus"]) == DEFAULT  # an edited run.json
    # a run.json from before #131 has no precisions: the default, no 4-bit
    assert precisions.of_config({"quality": NEAR}) == DEFAULT
    assert precisions.of_config({"quality": NEAR, "precisions": list(EVERY)}) == EVERY
    assert precisions.of_config(None) == ("exact",)

    why = precisions.refusal("fp4_weights", DEFAULT)
    assert "not allowed" in why and "opt-in" in why and "exact,fp8_weights" in why
    assert precisions.refusal(None, ("exact",)) is None
    assert precisions.refusal("fp4_weights", EVERY) is None
    assert "opt-in" not in precisions.refusal("fp8_weights", ("exact",))
    assert precisions.of_spec({"capture": {"precision": "fp4_weights"}}) == "fp4_weights"
    assert precisions.of_spec({"precision": "fp8_weights"}) == "fp8_weights"
    assert precisions.of_spec({}) == "exact"

    assert precisions.tier_allowed("near-lossless", DEFAULT)
    assert not precisions.tier_allowed("near-lossless-fp4", DEFAULT)
    assert precisions.tier_allowed("near-lossless-fp4", EVERY)
    assert not precisions.tier_allowed("near-lossless", ("exact",))
    assert precisions.tier_allowed("exact", ("exact",)) and precisions.tier_allowed(None, ())
    assert precisions.describe(DEFAULT).endswith("(4-bit not allowed: fp4_weights)")
    assert precisions.describe(("exact",)) == "exact"


# ------------------------------------------------------------------ the CLI and run.json


def test_cli_option_and_the_integrate_command(tmp_path, monkeypatch):
    seen: list[argparse.Namespace] = []
    monkeypatch.setattr(cli, "cmd_optimize", lambda ns: seen.append(ns) or 0)
    argv = ["optimize", "org/m", "--quality", NEAR, "--precisions", "fp8_weights,fp4_weights"]
    assert cli.main(argv) == 0
    assert cli._config(seen[-1]).precisions == ["exact", "fp8_weights", "fp4_weights"]
    assert cli.main(["optimize", "org/m"]) == 0 and cli._config(seen[-1]).precisions is None
    with pytest.raises(SystemExit):  # argparse: unknown precision
        cli.main(["optimize", "org/m", "--precisions", "int4"])

    calls: list[Any] = []

    class Stub:
        simulated = False
        run = RunDir(tmp_path)

        async def integrate(self, reuse: bool = False) -> None:
            calls.append(("integrate", reuse))

        def _mark(self, name: str) -> None:
            calls.append(("mark", name))

    def resume(root: Path, overrides: dict[str, Any]) -> Stub:
        calls.append(("resume", root, overrides))
        return Stub()

    monkeypatch.setattr(orchestrator.Orchestrator, "resume", staticmethod(resume))
    monkeypatch.setattr("kernel_agent.report.write_report", lambda run: run.root / "report.md")
    assert cli.main(["integrate", str(tmp_path), "--precisions", "fp8_weights"]) == 0
    assert calls == [
        ("resume", tmp_path, {"precisions": ["exact", "fp8_weights"]}),
        ("integrate", True),  # the measurements of unchanged content reused
        ("mark", "report"),
    ]
    calls.clear()
    assert cli.main(["integrate", str(tmp_path), "--no-reuse"]) == 0
    assert calls[0] == ("resume", tmp_path, {}) and calls[1] == ("integrate", False)


def test_resume_records_the_precisions_in_run_json(tmp_path):
    orch, _ = make(tmp_path / "near")
    assert orch.allowed_precisions() == DEFAULT  # no --precisions: no 4-bit
    data = orch.run.load()
    data["config"].pop("precisions")  # a run from before #131
    write_json(orch.run.run_json, data)
    old = orchestrator.Orchestrator.resume(orch.run.root)
    assert old.allowed_precisions() == DEFAULT and "precisions" not in old.run.load()["config"]

    asked = orchestrator.Orchestrator.resume(orch.run.root, {"precisions": ["fp4_weights"]})
    assert asked.allowed_precisions() == ("exact", "fp4_weights")
    assert asked.run.load()["config"]["precisions"] == ["exact", "fp4_weights"]
    assert orchestrator.Orchestrator.resume(orch.run.root).allowed_precisions() == (
        "exact",
        "fp4_weights",
    )

    exact, _ = make(tmp_path / "exact", quality="exact")
    with pytest.raises(SystemExit, match="needs --quality near-lossless"):
        orchestrator.Orchestrator.resume(exact.run.root, {"precisions": ["fp8_weights"]})
    assert exact.run.load()["config"].get("precisions") is None  # unchanged


# ------------------------------------------------------------------ planner and prompts


def test_planner_offers_only_the_allowed_precisions(tmp_path):
    plan = {
        "analysis": "decode streams the weights",
        "targets": [
            {**fp4_target("mlp_fp8", module_class="Qwen3MLP"), "precision": "fp8_weights"},
            fp4_target("attn_fp4", module_class="Qwen3Attention"),
        ],
        "transforms": [],
    }
    for allowed, kept in ((None, ["mlp_fp8"]), (list(EVERY), ["mlp_fp8", "attn_fp4"])):
        seen: list[dict[str, Any]] = []

        async def planner(name, *, system_append, result=None, seen=seen, **kwargs):
            seen.append({"system": system_append, **kwargs})
            return AgentResult(name=name, structured=copy.deepcopy(plan))

        orch, _ = make(tmp_path / str(bool(allowed)), allowed=allowed)
        orch.agent_runner = planner
        asyncio.run(orch.plan())
        assert [t["id"] for t in read_json(orch.run.plan_json)["targets"]] == kept
        schema = seen[0]["output_format"]["schema"]["properties"]
        enum = schema["targets"]["items"]["properties"]["precision"]["enum"]
        pivots = schema["pivots"]["items"]["properties"]["precision"]["enum"]
        system = seen[0]["system"]
        if allowed is None:
            assert enum == list(DEFAULT) and pivots == list(DEFAULT[1:])
            assert "`fp4_weights` is not allowed" in system and "No 4-bit weights" in system
            assert '`precision: "fp4_weights"`' not in system
        else:
            assert enum == list(EVERY) and "fp4_weights" in pivots
            assert '`precision: "fp4_weights"`' in system and "No 4-bit weights" not in system
            assert "is not allowed: a target with" not in system
    # the full schema (tests, tools) still names every precision; an exact run's none
    assert "fp4_weights" in str(prompts.PLAN_SCHEMA)
    exact = prompts.plan_schema(("exact",))["properties"]
    assert exact["targets"]["items"]["properties"]["precision"]["enum"] == ["exact"]
    assert "pivots" not in exact


def test_prompts_name_no_4bit_unless_allowed():
    near = prompts.precision_policy(NEAR)
    assert '`precision: "fp8_weights"`' in near and '`precision: "fp8_w8a8"`' in near
    assert '`precision: "fp4_weights"`' not in near and "*FP4 w* / *W4A4* floors" in near
    only = prompts.precision_policy(NEAR, ("exact", "fp8_weights"))
    assert (
        '`precision: "fp8_w8a8"`' not in only
        and "`fp8_w8a8`, `fp8_mx`, `int8_weights`, `int8_w8a8`, `reduced`, `fp4_weights` are"
        in only
    )
    card = {"repo_id": "org/m", "modality": "tts"}
    plan = prompts.planner_prompt(card, {"median_ms": 1.0}, "# P", ["cuda"], 2, "py", "tc", NEAR)
    assert "FP8 / FP4 floors" not in plan and "the floors of the precisions this run" in plan

    target = {"id": "lm", "precision": "fp8_weights", "pivot_of": None}
    block = prompts._pivot_block(target, Path("/r/pivot.json"))
    assert "4-bit precisions (`fp4_weights`) are not allowed" in block  # named as refused
    assert "never propose them" in block
    assert "<one of `reduced`, `fp8_w8a8`, `fp8_mx`, `int8_weights`, `int8_w8a8`>" in block
    assert "streaming weights (`int8_weights`)" in block  # no fp4_weights
    allowed = prompts._pivot_block(target, Path("/r/pivot.json"), EVERY)
    assert "never propose" not in allowed
    assert "streaming weights (`int8_weights`, `fp4_weights`)" in allowed
    assert (
        "<one of `reduced`, `fp4_weights`, `fp8_w8a8`, `fp8_mx`, `fp8_kv`, `int8_weights`, "
        "`int8_w8a8`>" in allowed
    )
    assert prompts._pivot_block(target, Path("/r/p.json"), ("exact", "fp8_weights")) == ""

    spec = {"id": "lm", "module_class": "Linear", "why": "w", "approach": "a", "backends": ["cuda"]}
    capture = {"cases": [], "tier": NEAR, "precision": "fp8_weights"}
    args = (["cuda"], "python", "toolchain", 10, None)
    fp8 = prompts.engineer_prompt({**spec, "precision": "fp8_weights"}, capture, *args)
    assert "No 4-bit weights or activations" in fp8
    fp8_ok = prompts.engineer_prompt(
        {**spec, "precision": "fp8_weights"}, capture, *args, precisions=EVERY
    )
    assert "No 4-bit weights or activations" not in fp8_ok
    fp4 = prompts.engineer_prompt(
        {**spec, "precision": "fp4_weights"},
        {**capture, "tier": "near-lossless-fp4", "precision": "fp4_weights"},
        *args,
    )
    assert "# Precision: `fp4_weights`" in fp4 and "No 4-bit weights" not in fp4

    assert prompts.precision_note("exact", ("exact",)) == ""
    assert "No 4-bit weights" in prompts.precision_note(NEAR, DEFAULT)
    assert "No 4-bit" not in prompts.precision_note(NEAR, EVERY)


def test_systems_session_is_told_the_precisions(tmp_path):
    orch, _ = make(tmp_path)
    seen: list[str] = []

    async def agent(name, *, system_append, result=None, **kwargs):
        seen.append(system_append)
        return AgentResult(name=name)

    orch.agent_runner = agent
    asyncio.run(orch.systems_slice(evaluations=2, digest="", label="systems#1"))
    assert "# Precision" in seen[0] and "No 4-bit weights or activations" in seen[0]


# ------------------------------------------------------------------ pivots


def test_no_pivot_to_a_precision_the_run_does_not_allow(tmp_path):
    spec = {"id": "lm", "module_class": "Qwen3MLP", "precision": "fp8_weights"}
    proposal = {"precision": "fp4_weights", "precision_why": WHY}
    refused = pivot.check(spec, proposal, quality=NEAR, taken=set())
    assert "not allowed" in refused and "opt-in" in refused
    assert pivot.check(spec, proposal, quality=NEAR, taken=set(), allowed=EVERY) is None
    w8a8 = {"precision": "fp8_w8a8", "precision_why": WHY}
    assert "not allowed" in pivot.check(spec, w8a8, quality=NEAR, taken=set(), allowed=("exact",))

    orch, world = make(tmp_path / "default")
    with world.installed():
        done = asyncio.run(orch.pivot("mlp", proposal, source="research-mlp#4"))
    assert "not allowed" in done["refused"] and not orch.run.target("mlp__fp4_weights").exists()
    (event,) = [e for e in ledger.events(orch.run) if e["event"] == "pivot_refused"]
    assert "fp4_weights" in event["why"]

    orch, world = make(tmp_path / "asked", allowed=list(EVERY))
    with world.installed():
        done = asyncio.run(orch.pivot("mlp", proposal, source="research-mlp#4"))
    assert done == {"target": "mlp__fp4_weights"}
    spec = read_json(orch.run.target("mlp__fp4_weights") / "spec.json")
    assert spec["capture"]["tier"] == compare.NEAR_LOSSLESS_FP4_TIER


def test_research_and_replan_offer_only_allowed_pivots(tmp_path):
    for name, allowed, precision, offered in (
        ("default", None, None, True),
        ("exact_only", ["exact"], None, False),  # no reduced precision to move to
        ("only_its_own", ["fp8_w8a8"], "fp8_w8a8", False),  # it is the only reduced one
    ):
        orch, _ = make(tmp_path / name, allowed=allowed)
        if precision:
            data = read_json(orch.run.target("mlp") / "spec.json")
            write_json(orch.run.target("mlp") / "spec.json", {**data, "precision": precision})
        seen: list[dict[str, Any]] = []

        async def agent(name, seen=seen, **kwargs):
            seen.append(kwargs)
            return AgentResult(name=name)

        orch.agent_runner = agent
        asyncio.run(orch.research("mlp", reason="plateau", label="research-mlp#3"))
        proposal = pivot.proposal_path(orch.run, "mlp")
        assert (proposal in seen[0]["writable"]) is offered
        assert ("# Precision pivot" in seen[0]["system_append"]) is offered

    improver = Improver(make(tmp_path / "ctx")[0], ImproveConfig(), require_capture=False)
    arms = improver.arms()
    text = improve.rounds_context(improver.run, improver.state, 2, [], arms, quality=NEAR)
    assert (
        "## Precision pivots" in text
        and "`fp8_w8a8`, `fp8_mx`, `int8_weights`, `int8_w8a8` (the precisions this run allows)"
        in text
    )
    assert "fp4_weights" not in text
    exact_only = improve.rounds_context(
        improver.run, improver.state, 2, [], arms, quality=NEAR, precisions=("exact",)
    )
    assert "## Precision pivots" not in exact_only


# ------------------------------------------------------------------ capture, priors, kernels


def test_capture_refuses_a_disallowed_target(tmp_path, monkeypatch):
    orch, world = make(tmp_path)
    with world.installed():
        kept = orch._capture([fp4_target(), {**fp4_target("mlp8"), "precision": "fp8_weights"}])
    assert kept == ["mlp8"] and not orch.run.target("lm_fp4").exists()
    (event,) = [e for e in ledger.events(orch.run) if e["event"] == "capture_refused"]
    assert event["target"] == "lm_fp4" and "not allowed" in event["why"]

    # the worker refuses it too (run.json's precisions, or the orchestrator's --precisions)
    write_json(orch.run.target("lm_fp4") / "spec.json", fp4_target())
    monkeypatch.setattr(worker, "_workload", lambda run: pytest.fail("no capture"))
    ns = argparse.Namespace(target="lm_fp4", quality=None, precisions=None, parent=False)
    with pytest.raises(ValueError, match="capture of lm_fp4 refused: precision 'fp4_weights'"):
        worker.cmd_capture(orch.run, ns)
    ns.precisions = "exact,fp8_weights"
    with pytest.raises(ValueError, match="--precisions exact,fp8_weights"):
        worker.cmd_capture(orch.run, ns)
    allowed = argparse.Namespace(quality=NEAR, precisions=",".join(EVERY))
    assert worker._allowed(allowed, orch.run) == EVERY


def _entry(root: Path, eid: str, precision: str | None) -> None:
    path = root / "sm_120" / "Qwen3MLP" / eid
    path.mkdir(parents=True)
    (path / "kernel.py").write_text(f"def build(reference):  # {eid}\n    return reference\n")
    sig = library.signature_summary({"cases": [{"signature": "a0[1, 1024]:bfloat16"}]})
    meta = {
        "id": eid,
        "sm_arch": "sm_120",
        "module_class": "Qwen3MLP",
        "signature": sig,
        "speedup": 1.5,
        "files": {"kernel.py": eid},
    }
    write_json(path / "entry.json", meta | ({"precision": precision} if precision else {}))


def test_library_priors_follow_the_runs_precisions(tmp_path):
    root = tmp_path / "library"
    _entry(root, "bf16", "exact")
    _entry(root, "fp4", "fp4_weights")
    cases = [{"signature": "a0[1, 1024]:bfloat16", "count": 28}]
    fp4 = {
        "module_class": "Qwen3MLP",
        "capture": {"cases": cases, "tier": "near-lossless-fp4", "precision": "fp4_weights"},
    }
    assert {e.id for e in library.matches("sm_120", fp4)} == {"bf16", "fp4"}
    assert {e.id for e in library.matches("sm_120", fp4, precisions=EVERY)} == {"bf16", "fp4"}
    assert {e.id for e in library.matches("sm_120", fp4, precisions=DEFAULT)} == {"bf16"}

    orch, _ = make(tmp_path / "run")
    write_json(orch.run.target("lm_fp4") / "spec.json", {**fp4_target(), **fp4})
    capture = orch.run.capture_file("lm_fp4")
    capture.parent.mkdir(parents=True, exist_ok=True)
    capture.write_bytes(b"capture")
    said: list[str] = []
    tried = library.seed_target(
        orch.run, "lm_fp4", arch="sm_120", keeper=truth.of(orch.run), say=said.append
    )
    assert tried == [] and "no library priors: precision 'fp4_weights'" in said[0]
    asyncio.run(orch.seed_library(["lm_fp4"]))  # skipped, not remembered as seeded
    assert library.seeded(orch.run, "lm_fp4") is None


def test_kernels_phase_gives_a_disallowed_target_no_session(tmp_path):
    orch, _ = make(tmp_path)
    write_json(orch.run.target("lm_fp4") / "spec.json", fp4_target())
    data = orch.run.load()
    done = [t for t in orch.run.target_ids() if t != "lm_fp4"]
    data["phases"]["kernels"] = {"finished": done}
    write_json(orch.run.run_json, data)
    names: list[str] = []

    async def agent(name, **kwargs):
        names.append(name)
        return AgentResult(name=name)

    orch.agent_runner = agent
    asyncio.run(orch.kernels())
    assert names == [] and "lm_fp4" not in orch.run.load()["phases"]["kernels"]["finished"]


# ------------------------------------------------------------------ scheduler and ceilings


def test_scheduler_stops_a_disallowed_arm_and_uses_the_allowed_floors(tmp_path):
    from test_scheduler_ceilings import W8A8, arms_of, voxcpm2

    def fp4(data: dict) -> None:
        data["specs"][W8A8]["precision"] = "fp4_weights"  # say, its pivot went to FP4

    run, data = voxcpm2(tmp_path / "fp4", edit=fp4)
    arms = arms_of(run, data)
    arm = arms[W8A8]
    assert arm.stop is not None and arm.stop.startswith("precision 'fp4_weights' is not allowed")
    assert arm.ceiling is None and scheduler.plateau(arm, scheduler.Policy()) is None
    assert scheduler.pick(sorted(arms.values(), key=lambda a: a.stop is not None)) is not arm

    def asked(data: dict) -> None:
        fp4(data)
        data["run"]["config"]["precisions"] = list(EVERY)

    run, data = voxcpm2(tmp_path / "asked", edit=asked)
    arm = arms_of(run, data)[W8A8]
    assert arm.refused is None and arm.ceiling is not None and arm.ceiling.precision == "FP4 w"

    # the systems agent's end-to-end estimate: the lowest floor of the allowed precisions
    table = {
        "baseline_ms": 100.0,
        "per": "per run",
        "precisions": {n: {"label": p.label} for n, p in ceilings.PRECISIONS.items()},
        "e2e": {
            n: {"floor_ms": ms}
            for n, ms in (
                ("exact", 80.0),
                ("fp8_weights", 60.0),
                ("w8a8", 50.0),
                ("fp4_weights", 30.0),
                ("w4a4", 20.0),
            )
        },
    }
    profiles = [{"ceilings": table, "baseline_ms": 100.0}]
    for config, speedup, label in (
        ({"quality": "exact"}, 1.25, "exact"),
        ({"quality": NEAR}, 2.0, "W8A8"),  # an old run.json: no FP4
        ({"quality": NEAR, "precisions": list(EVERY)}, 100 / 30, "FP4 w"),
        ({"quality": NEAR, "precisions": ["fp8_weights"]}, 100 / 60, "FP8 w"),
    ):
        write_json(run.run_json, {**run.load(), "config": config})
        found = scheduler._e2e_estimate(run, profiles, 100.0)
        assert found is not None and found[0] == pytest.approx(speedup), config
        assert found[1].startswith(f": {label} floors")


def test_ceilings_show_and_rank_by_the_allowed_floors_only():
    from test_ceilings import PEAKS, _profile

    peaks = {**PEAKS, "tflops": {**PEAKS["tflops"], roofline.FP4: 640.7}}
    profile = _profile()
    for c in profile["classes"][:2]:  # the CFM and its LocDiT already below the exact floor
        c["work"][0]["inclusive_ms"] = 1200.0
    every = ceilings.build(profile, peaks, 5000.0)
    assert every["columns"] == list(ceilings.PRECISIONS)
    assert every["rows"][0]["saves_best"]["precision"] == "w4a4"  # the 4-bit floor
    text = ceilings.markdown(every)
    assert (
        "| exact | FP8 w | W8A8 | INT8 W8A8 | MXFP8 | FP4 w | W4A4 | saves ms |" in text
        and "not shown" not in text
    )

    table = ceilings.build(profile, peaks, 5000.0, allowed=DEFAULT)
    assert table["columns"] == ["exact", "fp8_weights", "w8a8", "int8_w8a8", "mxfp8"]
    assert table["rows"][0]["saves_best"]["precision"] == "w8a8"  # not W4A4
    assert table["rows"][0]["floors"]["w4a4"] is not None  # ceilings.json keeps every floor
    text = ceilings.markdown(table)
    header = next(line for line in text.splitlines() if line.startswith("| target"))
    assert header.endswith("| bound | exact | FP8 w | W8A8 | INT8 W8A8 | MXFP8 | saves ms |")
    row = next(line for line in text.splitlines() if line.startswith("| `UnifiedCFM`"))
    assert row.count("|") == header.count("|") and "(W8A8 " in row
    assert "not shown: FP4 w, W4A4, precisions this run does not allow" in text
    assert "FP4 w ≥" not in text and "W4A4 ≥" not in text and "W8A8 ≥" in text
    assert "FP4 w (NVFP4" not in text and "W8A8 at FP8 333 TFLOP/s" in text

    exact = ceilings.markdown(ceilings.build(profile, peaks, 5000.0, allowed=("exact",)))
    assert "| bound | exact | saves ms |" in exact
    assert (
        "not shown: FP8 w, W8A8, INT8 W8A8, MXFP8, FP4 w, W4A4," in exact and "W8A8 ≥" not in exact
    )
    assert ceilings.columns(("exact", "fp4_weights")) == ["exact", "fp4_weights", "w4a4"]
    assert ceilings.target_columns(DEFAULT) == [
        "exact",
        "fp8_weights",
        "w8a8",
        "int8_w8a8",
        "mxfp8",
    ]


# ------------------------------------------------------------------ integration and report


def test_an_old_runs_fp4_targets_stay_out_of_its_integration(tmp_path, monkeypatch):
    """The VoxCPM2 latency run (20261005-192504): its planner and a pivot moved the LM to
    FP4 in a near-lossless run. Its run.json has no precisions: a plain integration now
    leaves the FP4 arm out (and says so in integration.json and report.md); `--precisions
    ...,fp4_weights` on the run records the list in run.json and lets it in again."""
    sims = tuple(
        dataclasses.replace(s, pivot="fp4_weights") if s.id == "mlp" else s for s in dryrun.TARGETS
    )
    monkeypatch.setattr(dryrun, "TARGETS", sims)
    orch, world = make(tmp_path, allowed=list(EVERY), seed=1)  # FP4 asked for, back then
    improver = Improver(orch, ImproveConfig(research_every=2), require_capture=False)
    with world.installed():
        asyncio.run(improver.improve())
    run = orch.run
    assert read_json(run.target("mlp__fp4_weights") / "spec.json")["precision"] == "fp4_weights"
    best = orch._kernel_best("mlp__fp4_weights")
    assert best is not None and best["tolerance_tier"] == compare.NEAR_LOSSLESS_FP4_TIER
    tried = [i for h in read_json(run.root / "integration.json")["history"] for i in h["items"]]
    assert any(i.startswith("mlp__fp4_weights=") for i in tried)  # integrated back then

    data = run.load()
    data["config"].pop("precisions")  # a run.json from before #131
    write_json(run.run_json, data)
    old = orchestrator.Orchestrator.resume(run.root)
    with dryrun.World(old).installed():
        asyncio.run(old.integrate(reuse=True))
    integration = read_json(run.root / "integration.json")
    (skipped,) = integration["skipped"]
    assert skipped["target"] == "mlp__fp4_weights" and skipped["precision"] == "fp4_weights"
    assert "not allowed" in skipped["reason"] and "opt-in" in skipped["reason"]
    items = [a["item"] for a in integration["accepted"]] + [
        i for h in integration["history"] for i in h["items"]
    ]
    assert not any(i.startswith("mlp__fp4_weights=") for i in items)
    assert all(t != "mlp__fp4_weights" for t, _ in old._kernel_bests())

    from kernel_agent.report import write_report

    report = write_report(run).read_text()
    assert "* skipped `mlp__fp4_weights`: precision 'fp4_weights' is not allowed" in report
    assert "| `mlp__fp4_weights` (`fp4_weights`: not allowed, skipped) |" in report
    assert (
        "precisions allowed: exact, fp8_weights, reduced, fp8_w8a8, fp8_mx, int8_weights, "
        "int8_w8a8 (4-bit not"
    ) in report

    again = orchestrator.Orchestrator.resume(run.root, {"precisions": list(EVERY)})
    assert run.load()["config"]["precisions"] == list(EVERY)
    with dryrun.World(again).installed():
        asyncio.run(again.integrate(reuse=True))
    integration = read_json(run.root / "integration.json")
    assert "skipped" not in integration
    assert any(
        i.startswith("mlp__fp4_weights=") for h in integration["history"] for i in h["items"]
    )
    assert any(t == "mlp__fp4_weights" for t, _ in again._kernel_bests())


def test_a_dry_run_without_4bit_refuses_the_fp4_pivot(tmp_path, monkeypatch):
    sims = tuple(
        dataclasses.replace(s, pivot="fp4_weights") if s.id == "mlp" else s for s in dryrun.TARGETS
    )
    monkeypatch.setattr(dryrun, "TARGETS", sims)
    orch, world = make(tmp_path, seed=1)  # near-lossless, no --precisions
    improver = Improver(orch, ImproveConfig(research_every=2), require_capture=False)
    with world.installed():
        asyncio.run(improver.improve())
    run = orch.run
    assert not run.target("mlp__fp4_weights").exists()
    state = read_json(run.root / "improve.json")
    (session,) = [r for r in state["research"] if r.get("pivot")]
    assert "not allowed" in session["pivot"]["refused"]
    assert all(r["target"] != "mlp__fp4_weights" for r in ledger.rows(run))
