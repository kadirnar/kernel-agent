"""Tamper-proof evaluator state (issue #5): ``.truth/`` layout, digests and their checks.

CPU only, except the last test (``gpu``: the bundled Triton RMSNorm through the
new layout).
"""

import asyncio
import json
import os
import shutil
from pathlib import Path

import pytest
import torch

from kernel_agent import charts, dryrun, ledger, orchestrator, toolchain, truth, worker
from kernel_agent.agent import tools as tools_mod
from kernel_agent.agent.prompts import EXAMPLES_DIR
from kernel_agent.agent.tools import best_for_target, record_candidate, snapshot
from kernel_agent.config import OptimizeConfig
from kernel_agent.improve import ImproveConfig, Improver
from kernel_agent.integrate.export import export_optimized
from kernel_agent.kernels.evaluate import evaluate
from kernel_agent.profiling.capture import load_capture
from kernel_agent.truth import TamperError, Truth
from kernel_agent.workloads.base import WorkloadSpec
from kernel_agent.workspace import RunDir, read_json, read_jsonl, write_json

TOY = Path(__file__).with_name("chaotic_toy.py")
ROOT = os.geteuid() == 0  # root ignores file permissions

SAME_LINEAR = """import torch
from torch import nn


class Same(nn.Linear):
    pass


def build(reference):
    new = Same(reference.in_features, reference.out_features)
    new.load_state_dict(reference.state_dict())
    return new
"""


def sealed_run(tmp_path, **extra) -> RunDir:
    """A run directory with the ``.truth/`` layout (``run.json`` has ``truth``)."""
    run = RunDir.create(tmp_path, "org/m")
    write_json(run.run_json, {"card": {"repo_id": "org/m"}, "truth": truth.new_section(), **extra})
    return run


def tamper_events(run: RunDir) -> list[dict]:
    return [e for e in ledger.events(run) if e["event"] == "tamper"]


def force_write(path: Path, data: bytes, *, append: bool = False) -> None:
    """What an agent's Bash can do: chmod the read-only file and write it."""
    os.chmod(path, 0o644)
    with path.open("ab" if append else "wb") as fh:
        fh.write(data)


def evaluated(run: RunDir, target: str, speedup: float, name: str = "v") -> dict:
    """Snapshot a candidate and record a correct evaluation of it (as evaluate_candidate)."""
    src = run.target(target) / "candidates" / f"{name}.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(f"# {name} {speedup}\nimport triton\n")
    snap = snapshot(run, src, target)
    result = {"status": "ok", "correct": True, "speedup": speedup, "cases": []}
    record, _ = record_candidate(run, target, src, snap, result, hypothesis=name)
    return record


# ------------------------------------------------------------------ layout


def test_layout_of_sealed_and_old_runs(tmp_path):
    run = sealed_run(tmp_path / "new")
    assert run.sealed()
    assert run.capture_file("t") == run.root / ".truth/captures/t.pt"
    assert run.results_file("t") == run.root / ".truth/targets/t/results.jsonl"
    assert run.history_dir("t") == run.root / ".truth/targets/t/history"
    assert run.results_file() == run.root / ".truth/transforms/results.jsonl"
    assert run.baseline_output() == run.root / ".truth/baseline_output.pt"

    old = RunDir.create(tmp_path / "old", "org/m")
    write_json(old.run_json, {"card": {"repo_id": "org/m"}})
    assert not old.sealed()
    assert old.capture_file("t") == old.target("t") / "capture.pt"
    assert old.results_file("t") == old.target("t") / "results.jsonl"
    assert old.history_dir() == old.transforms_dir / "history"
    assert old.baseline_output() == old.root / "baseline_output.pt"
    assert not truth.of(old).enabled and truth.of(old).worker_args() == []


def test_evaluations_go_to_truth_with_agent_copies(tmp_path):
    run = sealed_run(tmp_path)
    record = evaluated(run, "t", 1.5)
    snap = run.history_dir("t") / Path(record["snapshot"]).name
    assert record["snapshot"].startswith("history/001_v_")
    assert record["snapshot_sha256"] == truth.sha256_file(snap)
    assert read_jsonl(run.results_file("t")) == [record]
    # the agent's copies: same snapshot and record in its own directory, never read back
    assert (run.target("t") / record["snapshot"]).read_bytes() == snap.read_bytes()
    assert read_jsonl(run.target("t") / "results.jsonl") == [record]
    if not ROOT:
        assert not os.access(snap, os.W_OK) and not os.access(run.results_file("t"), os.W_OK)
    entry = run.load()["truth"]["files"][".truth/targets/t/results.jsonl"]
    assert entry["bytes"] == run.results_file("t").stat().st_size
    assert best_for_target(run, "t")["snapshot"] == record["snapshot"]

    # numbering continues after the highest snapshot, so a deleted one causes no clash
    second = evaluated(run, "t", 1.6, "w")
    snap.unlink()
    again = evaluated(run, "t", 1.6, "w")
    assert second["snapshot"].startswith("history/002_w_")
    assert again["snapshot"].startswith("history/003_w_")


def test_inputs_only_drops_the_answer_key():
    case = {"method": "forward", "args": (1,), "kwargs": {}, "count": 2}
    case |= {"output": 3, "post_args": (1,), "post_kwargs": {}}
    stripped = truth.inputs_only({"module": "m", "cases": [case]})
    assert stripped["inputs_only"] and stripped["module"] == "m"
    assert stripped["cases"] == [{"method": "forward", "args": (1,), "kwargs": {}, "count": 2}]


# ------------------------------------------------------------------ tampering


def test_tampering_with_each_truth_file_is_detected(tmp_path):
    run = sealed_run(tmp_path)
    keeper = truth.of(run)
    write_json(run.baseline_json, {"median_ms": 10.0})
    out = run.baseline_output()
    out.parent.mkdir(parents=True)
    out.write_bytes(b"baseline output")
    keeper.seal_baseline(10.0)
    capture = run.capture_file("t")
    capture.parent.mkdir(parents=True)
    capture.write_bytes(b"capture with reference outputs")
    keeper.seal(capture)
    integration = run.root / "integration.json"
    write_json(integration, {"history": [{"items": ["x"], "median_ms": 9.0}]})
    keeper.seal(integration)
    evaluated(run, "t", 1.5)

    static = [run.baseline_json, out, capture]
    for path in static:
        assert keeper.verify(path) == truth.sha256_file(path)
        if not ROOT:
            assert not os.access(path, os.W_OK)
    assert keeper.load_json(integration)["history"]
    args = keeper.worker_args()
    assert args[:2] == ["--baseline-ms", "10.0"]
    assert f"baseline.json={keeper.expect(run.baseline_json)}" in args
    assert f".truth/baseline_output.pt={keeper.expect(out)}" in args

    force_write(run.baseline_json, json.dumps({"median_ms": 1.0}).encode())
    force_write(out, b"forged output")
    force_write(capture, b"forged capture")
    force_write(integration, json.dumps({"history": [{"items": ["x"], "median_ms": 0.1}]}).encode())
    results = run.results_file("t")
    lines = results.read_bytes().replace(b'"speedup": 1.5', b'"speedup": 9.5')
    force_write(results, lines)

    for fresh in (keeper, Truth(run)):  # this process's memory, and a resumed process
        for path in static:
            with pytest.raises(TamperError, match="modified"):
                fresh.verify(path)
        assert fresh.load_json(integration) == {}
        with pytest.raises(TamperError, match="were modified"):
            fresh.records(results)
    assert keeper.baseline_ms() == 10.0  # the recorded latency, not baseline.json's 1.0
    assert best_for_target(run, "t", keeper) is None  # refused, not the forged 9.5x
    with pytest.raises(TamperError):
        evaluated(run, "t", 2.0)  # no appending to records that were changed

    flagged = {e["file"] for e in tamper_events(run)}
    expected = {"baseline.json", ".truth/baseline_output.pt", ".truth/captures/t.pt"}
    assert expected | {"integration.json", ".truth/targets/t/results.jsonl"} <= flagged
    with pytest.raises(TamperError, match="no digest"):
        keeper.expect(run.capture_file("unknown"))  # never written by kernel-agent


def test_forged_result_lines_are_ignored(tmp_path):
    run = sealed_run(tmp_path)
    evaluated(run, "t", 1.2, "a")
    best = evaluated(run, "t", 1.5, "b")
    results = run.results_file("t")
    forged = {"correct": True, "speedup": 50.0, "snapshot": "history/999_x.py"}
    force_write(results, (json.dumps(forged) + "\n").encode(), append=True)
    assert best_for_target(run, "t")["snapshot"] == best["snapshot"]
    assert [r["speedup"] for r in truth.of(run).records(results)] == [1.2, 1.5]
    assert [r["speedup"] for r in Truth(run).records(results)] == [1.2, 1.5]  # after a resume
    assert "appended outside kernel-agent" in tamper_events(run)[-1]["problem"]

    evaluated(run, "t", 1.3, "c")  # the next record drops the forged line
    assert [r["speedup"] for r in read_jsonl(results)] == [1.2, 1.5, 1.3]

    # a record whose snapshot changed or disappeared is no winner
    snap = run.history_dir("t") / Path(best["snapshot"]).name
    force_write(snap, b"def build(reference):\n    return cached_output\n")
    assert best_for_target(run, "t")["speedup"] == 1.3
    for rec in read_jsonl(results)[1:]:
        (run.history_dir("t") / Path(rec["snapshot"]).name).unlink()
    assert best_for_target(run, "t")["speedup"] == 1.2
    problems = [e["problem"] for e in tamper_events(run)]
    assert "snapshot changed since it was evaluated" in problems
    assert "snapshot of a record is missing" in problems

    # a whole results file that kernel-agent never wrote is ignored
    other = run.results_file("u")
    other.parent.mkdir(parents=True)
    other.write_text(json.dumps(forged) + "\n")
    assert best_for_target(run, "u") is None


def test_tools_refuse_truth_paths_and_tampered_evaluations(tmp_path, monkeypatch):
    run = sealed_run(tmp_path)
    keeper = truth.of(run)
    capture = run.capture_file("t")
    capture.parent.mkdir(parents=True)
    capture.write_bytes(b"capture")
    keeper.seal(capture)
    write_json(run.target("t") / "spec.json", {"id": "t", "module_class": "M"})
    cand = run.target("t") / "candidates" / "v1.py"
    cand.parent.mkdir(parents=True)
    cand.write_text("def build(r): ...\n")
    seen = []

    def fake_eval(capture_path, snap, *, capture_sha256, **_):
        seen.append((capture_path, snap, capture_sha256))
        if len(seen) == 2:  # something rewrites the snapshot while it is evaluated
            force_write(snap, b"def build(r): return None\n")
        return {"status": "ok", "correct": True, "speedup": 2.0}

    monkeypatch.setattr(tools_mod, "run_evaluation", fake_eval)
    monkeypatch.setattr(tools_mod, "create_sdk_mcp_server", lambda n, version, tools: tools)
    monkeypatch.setattr(tools_mod, "refresh", lambda *a: None)
    server = {t.name: t for t in tools_mod.build_server(run, keeper=keeper)}

    def call(tool, **args):
        out = asyncio.run(server[tool].handler(args))
        return json.loads(out["content"][0]["text"])

    ok = call("evaluate_candidate", target_id="t", candidate="candidates/v1.py", hypothesis="h")
    assert ok["status"] == "ok" and ok["best_so_far"]["speedup"] == 2.0
    assert seen[0][0] == capture and seen[0][2] == keeper.expect(capture)
    assert seen[0][1].parent == run.history_dir("t")

    changed = call(
        "evaluate_candidate", target_id="t", candidate="candidates/v1.py", hypothesis="h"
    )
    assert changed["status"] == "tampered" and changed["ledger"]["status"] == "crash"

    inside = "../../.truth/targets/t/history/" + seen[0][1].name
    refused = call("evaluate_candidate", target_id="t", candidate=inside, hypothesis="h")
    assert refused["status"] == "error" and ".truth/" in refused["error"]
    refused = call("evaluate_e2e", transforms=["../.truth/transforms/x.py"])
    assert refused["status"] == "error" and ".truth/" in refused["error"]
    assert len(seen) == 2
    best = call("best_result", target_id="t")
    assert best["evaluations"] == 2 and best["best"]["speedup"] == 2.0

    run.capture_file("u").write_bytes(b"not written by kernel-agent")
    write_json(run.target("u") / "spec.json", {"id": "u", "module_class": "M"})
    unknown = call(
        "evaluate_candidate", target_id="u", candidate="../t/candidates/v1.py", hypothesis="h"
    )
    assert unknown["status"] == "error" and "no digest" in unknown["error"]
    assert len(seen) == 2


def test_capture_and_e2e_through_the_truth_layout(tmp_path, monkeypatch, capsys):
    """Worker analyze + capture + e2e and the evaluator on the CPU toy, sealed layout."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(toolchain, "setup", lambda *a, **k: None)
    spec = WorkloadSpec(repo_id="toy/chaotic", modality="tts", device="cpu", harness=str(TOY))
    run = sealed_run(tmp_path, workload=spec.to_dict())
    keeper = truth.of(run)

    def call(*argv):
        worker.main([str(a) for a in (argv[0], "--run-dir", run.root, *argv[1:])])
        line = [x for x in capsys.readouterr().out.splitlines() if x.startswith(worker.MARKER)]
        return json.loads(line[-1][len(worker.MARKER) :])

    baseline = call("analyze", "--no-profile", "--iters", 1)
    assert run.baseline_output().is_file() and not (run.root / "baseline_output.pt").exists()
    keeper.seal_baseline(baseline["median_ms"])  # what Orchestrator.analyze does

    write_json(
        run.target("lin") / "spec.json", {"module_class": "Linear", "qualname": "model.backbone"}
    )
    info = call("capture", "--target", "lin")
    assert info["cases"][0]["count"] == 60
    capture = run.capture_file("lin")
    keeper.seal(capture)  # what Orchestrator._capture does
    assert not (run.target("lin") / "capture.pt").exists()
    full = load_capture(capture, sha256=keeper.expect(capture))
    assert all("output" in c and "post_args" in c for c in full["cases"])
    visible = run.target("lin") / "capture_inputs.pt"
    agent = load_capture(visible)
    assert agent["inputs_only"] and agent["cases"]
    for case in agent["cases"]:
        assert {"args", "kwargs"} <= set(case) and not set(truth.ANSWER_KEYS) & set(case)
    assert torch.equal(agent["cases"][0]["args"][0], full["cases"][0]["args"][0])

    cand = run.target("lin") / "candidates" / "same.py"
    cand.parent.mkdir()
    cand.write_text(SAME_LINEAR)
    good = evaluate(capture, cand, device="cpu", capture_sha256=keeper.expect(capture))
    assert good["correct"] and good["status"] == "ok"
    inputs_only = evaluate(visible, cand, device="cpu")
    assert not inputs_only["correct"] and "inputs-only" in inputs_only["error"]

    result = call("e2e", *keeper.worker_args())
    assert result["status"] == "ok" and result["passed"]
    assert result["baseline_ms"] == round(baseline["median_ms"], 3)

    force_write(capture, capture.read_bytes() + b"\0")
    tampered = evaluate(capture, cand, device="cpu", capture_sha256=keeper.expect(capture))
    assert tampered["status"] == "tampered" and not tampered["correct"]
    force_write(run.baseline_output(), b"forged")
    result = call("e2e", *keeper.worker_args())
    assert result["status"] == "tampered" and not result["passed"]
    assert ".truth/baseline_output.pt" in {e["file"] for e in tamper_events(run)}


# ------------------------------------------------------------------ integration + export


def _dry(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator.toolchain, "setup", dryrun.SimToolchain)
    monkeypatch.setattr(charts, "available", lambda: False)
    config = OptimizeConfig(model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path)
    orch = orchestrator.Orchestrator(dryrun.create_run(config), config)
    world = dryrun.World(orch)
    icfg = ImproveConfig(max_slices=4, integrate_every=1000)
    improver = Improver(orch, icfg, require_capture=False, live_charts=False)
    with world.installed():
        asyncio.run(improver.improve())
    return orch, world


def test_integration_and_export_use_verified_snapshots(tmp_path, monkeypatch):
    orch, world = _dry(tmp_path, monkeypatch)
    run = orch.run
    assert run.sealed() and not tamper_events(run)
    items, digests = orch._integration_items()
    kernels = {i.partition("=")[0]: i.partition("=")[2] for k, i in items if k == "kernel"}
    assert kernels and all(Path(p).parent.parent.parent.name == "targets" for p in kernels.values())
    assert all(".truth" in Path(p).parts for p in kernels.values())
    target, path = next(iter(kernels.items()))
    assert digests[path] == truth.sha256_file(Path(path))
    winner = best_for_target(run, target)

    force_write(Path(path), b"def build(reference):\n    return cached\n", append=True)
    with pytest.raises(TamperError):
        export_optimized(run, [("kernel", f"{target}={path}", 0.0)], digests=digests)
    items, _ = orch._integration_items()
    assert f"{target}={path}" not in {i for _, i in items}
    after = best_for_target(run, target)
    assert after is None or after["snapshot"] != winner["snapshot"]

    # a forged integration.json does not feed the re-integration's cache
    integration = run.root / "integration.json"
    data = read_json(integration)
    for h in data["history"]:
        h.update(passed=True, median_ms=1.0, speedup=1500.0)
    force_write(integration, json.dumps(data).encode())
    calls = []
    real = world.worker

    def counting(run_dir, command, *args, **kwargs):
        calls.append(args)
        return real(run_dir, command, *args, **kwargs)

    with world.installed():
        orch.worker = counting
        asyncio.run(orch.integrate(reuse=True))
    assert calls and all("--baseline-ms" in a for a in calls)  # measured again
    assert read_json(integration)["final"]["speedup"] < 100
    flagged = {e["file"] for e in tamper_events(run)}
    assert "integration.json" in flagged and any(f.endswith(Path(path).name) for f in flagged)


def test_old_layout_runs_still_readable(tmp_path):
    from synthetic_run import make_run

    run = make_run(tmp_path / "synthetic")  # written like a run made before .truth/
    assert not run.sealed() and not run.truth_dir.exists()
    keeper = Truth(run)
    assert not keeper.enabled and keeper.baseline_ms() == read_json(run.baseline_json)["median_ms"]
    assert best_for_target(run, "attn", keeper)["correct"]
    rows = ledger.backfill(run)
    records = [*run.targets_dir.glob("*/results.jsonl"), run.transforms_dir / "results.jsonl"]
    assert len(rows) == sum(len(read_jsonl(p)) for p in records)
    assert any(r["backend"] == "triton" for r in rows)  # snapshots found in targets/<id>/history

    # records without digests, capture.pt and baseline_output.pt in the old places
    old = RunDir.create(tmp_path / "old", "org/m")
    write_json(old.run_json, {"card": {"repo_id": "org/m"}})
    write_json(old.target("t") / "spec.json", {"id": "t", "module_class": "M"})
    history = old.target("t") / "history"
    history.mkdir(parents=True)
    (history / "001_v_0a1b2c3d.py").write_text("import triton\n")
    record = {"correct": True, "speedup": 1.4, "snapshot": "history/001_v_0a1b2c3d.py"}
    (old.target("t") / "results.jsonl").write_text(json.dumps(record) + "\n")
    assert best_for_target(old, "t") == record
    assert [r["backend"] for r in ledger.backfill(old)] == ["triton"]
    assert not tamper_events(old)


# ------------------------------------------------------------------ GPU


@pytest.mark.gpu
def test_triton_rmsnorm_through_the_truth_layout(tmp_path, monkeypatch):
    from kernel_agent.selftest import make_rmsnorm_capture

    run = sealed_run(tmp_path)
    keeper = truth.of(run)
    capture = make_rmsnorm_capture(run.capture_file("rms"), hidden=1024)
    keeper.seal(capture)
    truth.write_inputs_capture(capture, run.target("rms") / "capture_inputs.pt")
    write_json(run.target("rms") / "spec.json", {"id": "rms", "module_class": "RMSNorm"})
    cand = run.target("rms") / "candidates" / "triton_v1.py"
    cand.parent.mkdir(parents=True)
    shutil.copy(EXAMPLES_DIR / "triton_rmsnorm.py", cand)
    monkeypatch.setattr(tools_mod, "create_sdk_mcp_server", lambda n, version, tools: tools)
    server = {t.name: t for t in tools_mod.build_server(run, keeper=keeper)}

    def evaluate_candidate() -> dict:
        args = {"target_id": "rms", "candidate": "candidates/triton_v1.py", "hypothesis": "h"}
        out = asyncio.run(server["evaluate_candidate"].handler(args))
        return json.loads(out["content"][0]["text"])

    result = evaluate_candidate()
    assert result["status"] == "ok" and result["correct"], result
    assert result["speedup"] > 0 and result["best_so_far"]["speedup"] == result["speedup"]
    best = best_for_target(run, "rms", keeper)
    snap = run.history_dir("rms") / Path(best["snapshot"]).name
    assert best["snapshot_sha256"] == truth.sha256_file(snap)
    agent = load_capture(run.target("rms") / "capture_inputs.pt")
    assert all("output" not in c for c in agent["cases"])

    force_write(capture, capture.read_bytes() + b"\0")
    refused = evaluate_candidate()
    assert refused["status"] == "tampered" and not refused["correct"]
    assert ".truth/captures/rms.pt" in {e["file"] for e in tamper_events(run)}
