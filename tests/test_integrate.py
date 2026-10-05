"""Integration candidates from every passing e2e record, and the measured combination as the
seed of the greedy search (issue #43).

CPU only: a simulated run (``dryrun.create_run``), records written through the evaluation
tools' own code, and a fake e2e worker that knows the latency of the combinations of the
VoxCPM2 run in the issue (runs/openbmb--VoxCPM2/20261005-042829).
"""

import argparse
import asyncio
import os
from pathlib import Path

import pytest

from kernel_agent import charts, dryrun, ledger, orchestrator, truth
from kernel_agent.agent.tools import record_candidate, record_e2e_result, snapshot
from kernel_agent.config import OptimizeConfig
from kernel_agent.report import write_report
from kernel_agent.workspace import read_json

BASE = dryrun.BASELINE_MS

# End-to-end speedups by what is applied: transform idea (+ version), kernel target.
ALONE = {
    "graph_cfm_solver": 2.055,
    "graph_lm_step v1": 1.223,
    "graph_lm_step v2": 1.24,
    "compile_dit_estimator": 1.688,
    "autotune_dit_estimator": 1.72,
    "compile_lm_layers": 1.03,
    "attn": 1.679,
    "rmsnorm": 1.18,
    "mlp": 0.99,  # slower than the baseline end to end
}
EXP = ("graph_cfm_solver", "autotune_dit_estimator", "graph_lm_step v2", "compile_lm_layers")
BEST = frozenset({*EXP, "attn"})  # the issue's 847 ms (6.38x) combination
COMBINATIONS = {
    BEST: 6.378,
    BEST | {"compile_dit_estimator"}: 6.31,  # the DiT compiled twice
    BEST | {"rmsnorm"}: 6.36,  # compile_lm_layers fuses the LM's norms already
}


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(orchestrator.toolchain, "setup", dryrun.SimToolchain)
    monkeypatch.setattr(charts, "available", lambda: False)


def make(tmp_path):
    config = OptimizeConfig(model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path)
    return orchestrator.Orchestrator(dryrun.create_run(config), config)


def label(item: str) -> str:
    """What an integration item applies: a kernel's target, a transform's first line."""
    if not item.startswith("/"):
        return item.partition("=")[0]
    return Path(item).read_text().splitlines()[0].removeprefix("# ")


def fake_worker(table: dict[frozenset[str], float], calls: list[list[str]]):
    def worker(run, command, *args):
        assert command == "e2e"
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--kernel", action="append", default=[])
        parser.add_argument("--transform", action="append", default=[])
        ns, _ = parser.parse_known_args(list(args))
        calls.append(list(args))
        key = frozenset(label(i) for i in [*ns.kernel, *ns.transform])
        if len(key) == 1:
            return dryrun._e2e_result(BASE / ALONE[next(iter(key))])
        assert key in table, f"unexpected combination {sorted(key)}"
        return dryrun._e2e_result(BASE / table[key])

    return worker


def kernel(run, target: str, speedup: float) -> str:
    """A correct kernel evaluation; returns ``target=<snapshot>`` as the systems agent's
    prompt lists it (the agent's copy in ``targets/<id>/history/``)."""
    src = run.target(target) / "candidates" / "v4.py"
    src.write_text(f"# {target}\n")
    snap = snapshot(run, src, target)
    result = {"status": "ok", "correct": True, "speedup": speedup, "cases": []}
    record_candidate(run, target, src, snap, result, hypothesis="v4")
    return f"{target}={run.target(target) / 'history' / snap.name}"


def e2e(run, speedup: float | None, files: list[Path], kernels: list[str] | None = None) -> dict:
    """An ``evaluate_e2e`` record of ``files`` (+ ``kernels``); ``None``: failed quality."""
    snaps = [snapshot(run, f) for f in files]
    result = dryrun._e2e_result(BASE / (speedup or 6.5))
    if speedup is None:
        result.update(passed=False, reason="teacher-forced: step 8 cosine 0.68385 < 0.7")
    record, _ = record_e2e_result(run, result, snaps, kernels or [], hypothesis="")
    return record


def write(run, name: str, version: str = "") -> Path:
    path = run.transforms_dir / f"{name}.py"
    path.write_text(f"# {name} {version}".strip() + "\n\ndef apply(workload):\n    pass\n")
    return path


def again(run, record: dict) -> Path:
    """The agent's copy of a record's first snapshot (``history/001_..._<sha>.py``)."""
    return run.transforms_dir / record["transforms"][0]


def voxcpm2(run) -> dict[str, dict]:
    """The issue's transforms/results.jsonl: the improved transforms were only ever
    evaluated on top of the attention kernel; snapshot names as in the real run."""
    attn = kernel(run, "attn", 2.1)
    kernel(run, "rmsnorm", 1.5)
    kernel(run, "mlp", 1.3)
    cfm = e2e(run, 2.0843, [write(run, "graph_cfm_solver")])  # 001
    e2e(run, 1.2366, [write(run, "graph_lm_step", "v1")])  # 002
    dit = e2e(run, 1.375, [write(run, "compile_dit_estimator")])  # 003
    names = ("graph_cfm_solver", "graph_lm_step", "compile_dit_estimator")
    e2e(run, 5.0473, [run.transforms_dir / f"{n}.py" for n in names])  # 004-006, lm v1
    lm = write(run, "graph_lm_step", "v2")
    e2e(run, 5.9027, [again(run, cfm), again(run, dit), lm], [attn])  # 007-009
    layers = write(run, "compile_lm_layers")
    exp6 = e2e(run, 6.1677, [again(run, cfm), again(run, dit), lm, layers], [attn])  # 010-013
    files = [again(run, cfm), write(run, "autotune_dit_estimator"), lm, layers]
    exp7 = e2e(run, 6.3778, files, [attn])  # 014_001_graph_cfm_solver_<sha>_<sha>.py, 015-017
    e2e(run, None, [again(run, cfm), *files[1:], write(run, "compile_feat_encoder")], [attn])
    return {"exp6": exp6, "exp7": exp7}


def test_snapshot_stem_strips_every_history_prefix_and_suffix():
    stem = ledger.snapshot_stem
    assert stem("history/014_001_graph_cfm_solver_300928ff_300928ff.py") == "graph_cfm_solver"
    assert stem("/r/.truth/transforms/history/016_graph_lm_step_4710b7af.py") == "graph_lm_step"
    assert stem("graph_lm_step.py") == "graph_lm_step"
    assert stem("v2_graph.py") == "v2_graph" and stem("2_stage.py") == "2_stage"
    item = "/x/history/010_001_graph_cfm_solver_300928ff_300928ff.py"
    assert ledger.item_label(item) == "graph_cfm_solver"


def test_candidates_come_from_every_passing_record(tmp_path):
    orch = make(tmp_path)
    run = orch.run
    records = voxcpm2(run)
    items, digests = orch._integration_items()
    transforms = {label(a): Path(a) for k, a in items if k == "transform"}
    # one version per idea, from the fastest passing record that has it; the failed
    # record's compile_feat_encoder is none
    assert set(transforms) == {*EXP, "compile_dit_estimator"}
    newest = {Path(t).name for t in records["exp7"]["transforms"]}
    assert {transforms[t].name for t in EXP} == newest
    assert transforms["graph_cfm_solver"].name.startswith("014_001_graph_cfm_solver_")
    assert transforms["compile_dit_estimator"].name.startswith("011_003_compile_dit_estimator_")
    assert all(p.parent == run.history_dir() for p in transforms.values())  # .truth/
    assert all(digests[str(p)] == truth.sha256_file(p) for p in transforms.values())
    assert sorted(label(a) for k, a in items if k == "kernel") == ["attn", "mlp", "rmsnorm"]

    combo, record = orch._integration_composite(digests)
    assert record["exp"] == records["exp7"]["exp"]
    assert [label(a) for _, a in combo] == [*EXP, "attn"]
    attn = Path(combo[-1][1].partition("=")[2])
    assert attn.parent == run.history_dir("attn") and ".truth" in attn.parts
    assert digests[str(attn)] == truth.sha256_file(attn)


def test_integration_seeds_with_the_measured_combination(tmp_path):
    """The issue's run ended at 928.6 ms (5.82x) although 847 ms (6.38x) was measured."""
    orch = make(tmp_path)
    run = orch.run
    exp = voxcpm2(run)["exp7"]["exp"]
    calls: list[list[str]] = []
    orch.worker = fake_worker(COMBINATIONS, calls)
    asyncio.run(orch.integrate())

    data = read_json(run.root / "integration.json")
    accepted = [a["item"] for a in data["accepted"]]
    assert [label(a) for a in accepted] == [*EXP, "attn"]  # graph_lm_step v2 included
    assert data["final"]["passed"] and data["final"]["speedup"] == pytest.approx(6.378, 1e-3)
    assert data["composite"] == {"items": accepted, "exp": exp, "seeded": True}
    # everything measured again: 8 items alone, the combination, then 2 additions
    measured = [frozenset(label(i) for i in h["items"]) for h in data["history"]]
    assert measured[8:] == [BEST, BEST | {"compile_dit_estimator"}, BEST | {"rmsnorm"}]
    assert len(calls) == len(measured) == 11
    assert all("--baseline-ms" in c and "--verify" in c for c in calls)
    assert "graph_lm_step v1" not in {label(i) for h in data["history"] for i in h["items"]}
    rows = [r for r in ledger.rows(run) if r["backend"] == "integrate"]
    assert rows[8]["hypothesis"].endswith(f"(the combination of exp {exp})")

    manifest = read_json(run.optimized_dir / "manifest.json")
    assert [k["target"] for k in manifest["kernels"]] == ["attn"]
    assert len(manifest["transforms"]) == 4

    steps = charts.integration_steps(data)
    assert [s["kind"] for s in steps] == ["total", "keep", "nogain", "nogain", "total"]
    assert steps[1]["label"] == f"exp {exp} combination" and steps[1]["source"] == "5 items"
    assert [steps[2]["label"], steps[3]["label"]] == ["compile_dit_estimator", "rmsnorm"]
    assert steps[-1]["ms"] == pytest.approx(BASE / 6.378, rel=1e-3)
    report = write_report(run).read_text()
    assert f"measured combination of exp {exp} (5 items) seeded the search" in report

    calls.clear()  # a re-integration of the same files measures nothing again
    asyncio.run(orch.integrate(reuse=True))
    assert calls == []
    assert read_json(run.root / "integration.json")["accepted"] == data["accepted"]


def test_a_slower_combination_does_not_seed(tmp_path):
    orch = make(tmp_path)
    run = orch.run
    exp = voxcpm2(run)["exp7"]["exp"]
    seed = frozenset({"graph_cfm_solver", "autotune_dit_estimator"})
    table = {
        BEST: 1.9,  # slower now than graph_cfm_solver alone (2.055x)
        seed: 3.4,
        seed | {"compile_dit_estimator"}: 3.3,
        seed | {"attn"}: 5.0,
        **{seed | {"attn", extra}: 4.9 for extra in (*EXP[2:], "rmsnorm")},
    }
    orch.worker = fake_worker(table, [])
    asyncio.run(orch.integrate())

    data = read_json(run.root / "integration.json")
    assert data["composite"]["seeded"] is False and data["composite"]["exp"] == exp
    accepted = [label(a["item"]) for a in data["accepted"]]
    assert accepted == ["graph_cfm_solver", "autotune_dit_estimator", "attn"]  # greedy as before
    assert data["final"]["speedup"] == pytest.approx(5.0, 1e-3)
    kinds = [s["kind"] for s in charts.integration_steps(data)]
    assert kinds == ["total", "keep", "nogain", "keep", "nogain", "keep", *["nogain"] * 3, "total"]


def test_combination_with_an_unverified_file_is_no_seed(tmp_path):
    orch = make(tmp_path)
    run = orch.run
    attn = kernel(run, "attn", 2.1)
    wip = run.target("attn") / "candidates" / "wip.py"
    wip.write_text("# attn\n")
    graph, lm = write(run, "graph_cfm_solver"), write(run, "graph_lm_step")
    first = e2e(run, 2.5, [graph], [f"attn={wip}"])  # a kernel candidate, not a snapshot
    best = e2e(run, 2.2, [graph, lm], [attn])
    assert orch._integration_composite({})[1]["exp"] == best["exp"]

    snap = run.history_dir() / Path(best["transforms"][1]).name
    os.chmod(snap, 0o644)  # what an agent's Bash can do
    snap.write_text("# graph_lm_step\nCACHED = True\n")
    assert orch._integration_composite({}) is None
    items, _ = orch._integration_items()  # the verified transforms still count
    assert [Path(a).name for k, a in items if k == "transform"] == [
        Path(first["transforms"][0]).name
    ]
