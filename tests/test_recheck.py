"""Independent re-check of winners (issue #21): fresh inputs, separate processes.

CPU except the last test (``gpu``: the bundled Triton RMSNorm example). The integration
tests use a simulated run (``dryrun.create_run``), records written through the
evaluation tools' own code, a fake e2e worker and a fake re-check.
"""

import asyncio
import copy
import statistics
from pathlib import Path

import pytest
import torch

from kernel_agent import charts, dryrun, ledger, orchestrator, truth
from kernel_agent.agent.prompts import EXAMPLES_DIR
from kernel_agent.agent.tools import record_candidate, snapshot
from kernel_agent.config import OptimizeConfig
from kernel_agent.kernels import recheck
from kernel_agent.kernels.evaluate import evaluate
from kernel_agent.profiling.capture import capture_calls
from kernel_agent.report import write_report
from kernel_agent.selftest import RMSNorm, make_rmsnorm_capture
from kernel_agent.workspace import read_json

HONEST = """import torch
from torch import nn


class Norm(nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.weight = reference.weight
        self.eps = reference.variance_epsilon

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * y.to(x.dtype)


def build(reference):
    return Norm(reference)
"""

# Correct on the captured inputs, wrong on any other: the captured outputs, keyed by shape.
SHAPE_KEYED = """import torch
from torch import nn

_CAPTURE = torch.load({capture!r}, weights_only=False)
_OUTPUTS = {{tuple(c["args"][0].shape): c["output"] for c in _CAPTURE["cases"]}}


class Memo(nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.weight = reference.weight

    def forward(self, x):
        return _OUTPUTS[tuple(x.shape)].clone()


def build(reference):
    return Memo(reference)
"""


@pytest.fixture
def cpu_children(monkeypatch):
    """The re-check's subprocesses on the CPU, and no GPU lock to wait for."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("KERNEL_AGENT_LOCK_HELD", "1")


def rmsnorm_capture(path: Path) -> Path:
    torch.manual_seed(0)
    module = RMSNorm(64, eps=1e-5)
    with torch.no_grad():
        module.weight.copy_(torch.randn(64) * 0.1 + 1)
    calls = [((torch.randn(2, 8, 64),), {}, 1), ((torch.randn(1, 1, 64),), {}, 7)]
    capture_calls(module, calls, path)
    return path


def test_fresh_inputs_redraw_floats_and_keep_ids_masks_and_state():
    class Cache:
        def __init__(self) -> None:
            self.k = torch.randn(1, 4, 8)

    base = torch.randn(2, 16)
    x, view = torch.randn(3, 8).t(), base[:, :8]  # a transposed tensor, a view of base
    ids, mask = torch.arange(6), torch.ones(3, 3, dtype=torch.bool).tril()
    additive = torch.zeros(3, 3).masked_fill(mask, -1e9)
    cache = Cache()
    args, kwargs = (x, view, base, ids), {"mask": mask, "bias": additive, "cache": cache}
    gen = torch.Generator().manual_seed(3)
    new_args, new_kwargs, redrawn = recheck.fresh_inputs(args, kwargs, gen, "normal")

    assert redrawn == 3  # x, the view and its base; not ids, the boolean or additive mask
    nx, nview, nbase, nids = new_args
    assert nx.shape == x.shape and nx.stride() == x.stride() and nx.dtype == x.dtype
    assert not torch.equal(nx, x) and not torch.equal(nbase, base)
    assert nview.untyped_storage().data_ptr() == nbase.untyped_storage().data_ptr()  # aliasing
    assert torch.equal(nview, nbase[:, :8])
    assert torch.equal(nids, ids) and torch.equal(new_kwargs["mask"], mask)
    assert torch.equal(new_kwargs["bias"], additive)
    assert new_kwargs["cache"] is not cache and torch.equal(new_kwargs["cache"].k, cache.k)
    assert torch.equal(args[0], x) and torch.equal(base, args[2])  # the originals untouched
    # the statistics of each tensor carry over
    big = torch.randn(256, 256) * 3 + 5
    (drawn,), _, _ = recheck.fresh_inputs((big,), {}, gen, "mix")
    assert abs(float(drawn.mean()) - 5) < 0.2 and abs(float(drawn.std()) - 3) < 0.2


def test_judge_compares_with_the_evaluator():
    def judged(measured, claimed, *, correct=True, spread=0.0):
        result = {"status": "ok" if correct else "incorrect", "correct": correct}
        result.update(speedup=measured, timing_spread=spread)
        return recheck.judge(result, {"correct": True, "speedup": claimed}, 0.25)

    same = judged(1.9, 2.0)
    assert same["passed"] and same["agrees"] == {"correct": True, "speedup": True}
    slower = judged(1.0, 2.0)  # the evaluator's 2x is not reproduced
    assert not slower["passed"] and slower["status"] == "slower"
    assert slower["agrees"]["speedup"] is False and "2.0x claimed" in slower["reason"]
    faster = judged(3.0, 2.0)  # disagrees, but in the kernel's favour: not a failure
    assert faster["passed"] and faster["agrees"]["speedup"] is False
    noisy = judged(1.4, 2.0, spread=0.3)  # noisy rounds widen the tolerance to 60 %
    assert noisy["passed"] and noisy["tolerance"] == 0.6
    wrong = judged(None, 2.0, correct=False)
    assert not wrong["passed"] and wrong["agrees"] == {"correct": False}
    untimed = recheck.judge({"status": "ok", "correct": True}, None, 0.25)
    assert untimed["passed"] and untimed["agrees"] == {}


def test_recheck_flags_outputs_keyed_by_shape_and_passes_an_honest_kernel(tmp_path, cpu_children):
    capture = rmsnorm_capture(tmp_path / "rms.pt")
    cheat = tmp_path / "shape_keyed.py"
    cheat.write_text(SHAPE_KEYED.format(capture=str(capture)))
    honest = tmp_path / "honest.py"
    honest.write_text(HONEST)

    # the cheat matches every captured case (the evaluator's re-verification is what
    # catches it there; the re-check does not rely on it)
    in_process = evaluate(capture, cheat, device="cpu")
    assert in_process["status"] == "incorrect_perturbed", in_process
    assert all(case["ok"] for case in in_process["cases"])

    verdict = {"status": "ok", "correct": True, "speedup": 1.4}
    flagged = recheck.run_recheck(capture, cheat, verdict=verdict, seed=11)
    assert flagged["status"] == "incorrect" and not flagged["passed"], flagged
    assert flagged["correct"] is False and flagged["agrees"] == {"correct": False}
    assert flagged["seed"] == 11 and flagged["seeds"] == 3
    assert "wrong on fresh inputs" in flagged["reason"] and "seed 11" in flagged["reason"]
    assert [c["ok"] for c in flagged["cases"]] == [False, False]
    assert flagged["cases"][0]["failures"][0]["seed"] == 11
    assert "FAILED (incorrect)" in recheck.describe(flagged)

    ok = recheck.run_recheck(capture, honest, verdict=verdict, seed=11)
    assert ok["status"] == "ok" and ok["passed"] and ok["correct"], ok
    assert ok["agrees"] == {"correct": True}  # no timing on the CPU: nothing to compare
    assert ok["timing"] == "skipped: no CUDA device" and ok["speedup"] is None
    assert [c["redrawn"] for c in ok["cases"]] == [1, 1]
    assert [c["calls_per_run"] for c in ok["cases"]] == [1, 7]
    assert "correct on 3 fresh draws of 2 case(s)" in recheck.describe(ok)


def test_recheck_reports_build_and_runtime_errors(tmp_path, cpu_children):
    capture = rmsnorm_capture(tmp_path / "rms.pt")
    broken = tmp_path / "broken.py"
    broken.write_text("def build(reference):\n    raise ValueError('no kernel')\n")
    result = recheck.run_recheck(capture, broken, seed=1)
    assert result["status"] == "build_error" and not result["passed"]
    assert "no kernel" in result["reason"]
    crash = tmp_path / "crash.py"
    crash.write_text(HONEST.replace("return self.weight * y.to(x.dtype)", "raise RuntimeError"))
    result = recheck.run_recheck(capture, crash, seed=1)
    assert result["status"] == "runtime_error" and result["failed_case"]["case"] == 0


# A module that writes into a cache argument in place (its class pickles by value).
CACHE_MODULE = """import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, cache, pos):
        cache[:, int(pos)] = self.proj(x)
        return cache[:, : int(pos) + 1].mean(1)


def get_inputs():
    return []


def get_init_inputs():
    return [16]
"""

CACHE_CANDIDATE = """import torch
from torch import nn


class Fast(nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.proj = reference.proj

    def forward(self, x, cache, pos):
        new = self.proj(x)
        if {write}:
            cache[:, int(pos)] = new
        return torch.cat([cache[:, : int(pos)], new[:, None]], 1).mean(1)


def build(reference):
    return Fast(reference)
"""


def test_recheck_compares_side_effects_on_fresh_inputs(tmp_path, cpu_children):
    from kernel_agent.kernelbench import load_module

    module = load_module(CACHE_MODULE, "ka_test_cache_module", str(tmp_path / "cache.py"))
    torch.manual_seed(0)
    model = module.Model(16)
    cache = torch.zeros(2, 8, 16)
    cache[:, :3] = torch.randn(2, 3, 16)
    calls = [((torch.randn(2, 16), cache, torch.tensor(3)), {}, 5)]
    capture = tmp_path / "cache.pt"
    capture_calls(model, calls, capture)
    for write, passed in ((True, True), (False, False)):
        candidate = tmp_path / f"write_{write}.py"
        candidate.write_text(CACHE_CANDIDATE.format(write=write))
        result = recheck.run_recheck(capture, candidate, seed=2)
        assert result["passed"] is passed, result
        assert result["cases"][0]["redrawn"] == 2  # x and the cache; not the position
    failure = result["cases"][0]["failures"][0]
    assert failure["name"] == "args[1]" and failure["seed"] == 2  # the cache was not written


# ------------------------------------------------------------------ integration


@pytest.fixture
def simulated(monkeypatch):
    monkeypatch.setattr(orchestrator.toolchain, "setup", dryrun.SimToolchain)
    monkeypatch.setattr(charts, "available", lambda: False)


def make(tmp_path, **config):
    cfg = OptimizeConfig(model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path, **config)
    return orchestrator.Orchestrator(dryrun.create_run(cfg), cfg)


def kernel(run, target: str, speedup: float) -> str:
    """A correct kernel evaluation; returns its integration item ``target=<snapshot>``."""
    src = run.target(target) / "candidates" / "v1.py"
    src.write_text(f"# {target}\n")
    snap = snapshot(run, src, target)
    result = {"status": "ok", "correct": True, "speedup": speedup, "cases": []}
    record_candidate(run, target, src, snap, result, hypothesis="v1")
    capture = truth.replace(run.capture_file(target))  # a placeholder: the re-check is fake
    capture.write_bytes(b"capture")
    truth.of(run).seal(capture)
    return f"{target}={run.history_dir(target) / snap.name}"


#: End-to-end latency factor of each kernel target alone.
FACTOR = {"attn": 0.7, "mlp": 0.9}


def worker(calls: list[str]):
    """``e2e`` / ``e2e_ab`` of kernels: the product of their factors (paired rounds)."""

    def ms(items: list[str]) -> float:
        out = dryrun.BASELINE_MS
        for item in items:
            out *= FACTOR[item.partition("=")[0]]
        return out

    def run_worker(run, command, *args):
        args = list(args)
        a = [args[i + 1] for i, x in enumerate(args) if x == "--kernel"]
        b = [args[i + 1] for i, x in enumerate(args) if x == "--b-kernel"]
        calls.extend(x.partition("=")[0] for x in [*a, *b])
        if command == "e2e":
            return dryrun._e2e_result(ms(a))
        wiggle = (1.0004, 0.9996, 1.0002, 0.9998, 1.0001, 0.9999, 1.0003, 0.9997)
        a_ms = [ms(a) * w for w in wiggle]
        b_ms = [ms(b) * w for w in wiggle[::-1]]
        result = dryrun._e2e_result(statistics.median(b_ms))
        result.update(times_ms=b_ms, ab={"mode": "paired", "a_ms": a_ms, "b_ms": b_ms})
        return result

    return run_worker


def test_integration_refuses_a_kernel_that_fails_the_recheck(tmp_path, simulated):
    orch = make(tmp_path)
    run = orch.run
    attn, mlp = kernel(run, "attn", 2.0), kernel(run, "mlp", 1.5)
    rechecked: list[tuple[str, dict]] = []

    def fake_recheck(capture, snap, *, verdict, capture_sha256, timeout):
        target = Path(snap).parent.parent.name
        rechecked.append((target, verdict))
        assert Path(capture) == run.capture_file(target)
        assert capture_sha256 == truth.sha256_file(capture)
        if target == "attn":  # outputs cached by shape: wrong on fresh inputs
            result = {"status": "incorrect", "correct": False, "seeds": 3, "cases": []}
            result["reason"] = "case 0 (a0[1, 1, 1024]), seed 5: output: wrong on fresh inputs"
        else:
            result = {"status": "ok", "correct": True, "speedup": 1.45, "seeds": 3, "cases": []}
        return recheck.judge(result, verdict, recheck.SPEEDUP_TOLERANCE)

    orch.rechecker = fake_recheck
    calls: list[str] = []
    orch.worker = worker(calls)
    asyncio.run(orch.integrate())

    assert [t for t, _ in rechecked] == ["attn", "mlp"]
    assert dict(rechecked)["attn"] == {"correct": True, "speedup": 2.0}  # the verified record
    assert "attn" not in calls  # never measured end to end
    data = read_json(run.root / "integration.json")
    assert [a["item"] for a in data["accepted"]] == [mlp]
    checks = {r["target"]: r for r in data["recheck"]}
    assert not checks["attn"]["passed"] and checks["attn"]["item"] == attn
    assert checks["mlp"]["passed"] and checks["mlp"]["agrees"] == {"correct": True, "speedup": True}
    assert checks["attn"]["sha256"] and checks["attn"]["snapshot"] == Path(attn).name
    events = [e for e in ledger.events(run) if e["event"] == "recheck_failed"]
    assert len(events) == 1 and events[0]["target"] == "attn"
    assert "wrong on fresh inputs" in events[0]["reason"]
    report = write_report(run).read_text()
    assert "recheck `attn`" in report and "— refused" in report
    assert "recheck `mlp`" in report and "1.45x in separate processes vs 1.5x" in report

    rechecked.clear()  # a re-integration of the same snapshots reuses the results
    asyncio.run(orch.integrate(reuse=True))
    assert rechecked == []
    again = read_json(run.root / "integration.json")
    assert [r["passed"] for r in again["recheck"]] == [False, True]
    assert [a["item"] for a in again["accepted"]] == [mlp]


def test_no_recheck_and_simulated_runs(tmp_path, simulated):
    orch = make(tmp_path / "off", recheck=False)
    kernel(orch.run, "attn", 2.0)
    orch.rechecker = lambda *a, **k: pytest.fail("--no-recheck still re-checked")
    orch.worker = worker([])
    asyncio.run(orch.integrate())
    data = read_json(orch.run.root / "integration.json")
    assert "recheck" not in data and len(data["accepted"]) == 1

    orch = make(tmp_path / "sim")  # a simulated run has no captures: skipped, kept
    kernel(orch.run, "attn", 2.0)
    orch.worker = worker([])
    asyncio.run(orch.integrate())
    data = read_json(orch.run.root / "integration.json")
    assert [(r["status"], r["passed"]) for r in data["recheck"]] == [("skipped", True)]
    assert len(data["accepted"]) == 1


def test_a_combination_with_a_refused_kernel_seeds_nothing(tmp_path, simulated):
    orch = make(tmp_path)
    run = orch.run
    attn = kernel(run, "attn", 2.0)
    composite = ([("kernel", attn)], {"exp": 7})
    orch.rechecker = lambda *a, verdict, **k: recheck.judge(
        {"status": "incorrect", "correct": False, "reason": "fresh"}, verdict, 0.25
    )
    items = [("kernel", attn), ("transform", "/x/graph.py")]
    kept, seed, records = orch._recheck_kernels(items, copy.deepcopy(composite), {})
    assert kept == [("transform", "/x/graph.py")] and seed is None
    assert [r["item"] for r in records] == [attn]


# ------------------------------------------------------------------ GPU


@pytest.mark.gpu
def test_recheck_of_the_bundled_triton_rmsnorm(tmp_path):
    from kernel_agent.kernels.evaluate import run_evaluation

    capture = make_rmsnorm_capture(tmp_path / "rmsnorm.pt")
    example = EXAMPLES_DIR / "triton_rmsnorm.py"
    verdict = run_evaluation(capture, example)
    assert verdict["correct"], verdict
    result = recheck.run_recheck(capture, example, verdict=verdict)
    print(recheck.describe(result), result.get("cases"))
    assert result["status"] == "ok" and result["passed"] and result["correct"], result
    assert result["agrees"]["correct"] is True
    assert result["speedup"] and result["speedup"] > 1.0
    assert all(c["ok"] and c["redrawn"] == 1 for c in result["cases"])
    assert [c["calls_per_run"] for c in result["cases"]] == [1, 31]


@pytest.mark.gpu
def test_recheck_runs_the_fresh_inputs_again_after_timing(tmp_path):
    """Correct for its first calls (the correctness pass), then fast and wrong."""
    capture = make_rmsnorm_capture(tmp_path / "rmsnorm.pt")
    switch = tmp_path / "switch.py"
    switch.write_text(
        HONEST.replace(
            "    def forward(self, x):\n",
            "    calls = 0\n\n    def forward(self, x):\n"
            "        Norm.calls += 1\n"
            "        if Norm.calls > 6:  # 3 seeds x 2 cases\n"
            "            return torch.zeros_like(x)\n",
        )
    )
    result = recheck.run_recheck(capture, switch, verdict={"correct": True, "speedup": 1.0})
    assert result["status"] == "incorrect" and not result["passed"], result
    assert "after timing" in result["reason"]
    assert all(f.get("after_timing") for c in result["cases"] for f in c["failures"])
