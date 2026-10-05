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
from kernel_agent.agent.tools import best_for_target, record_candidate, snapshot
from kernel_agent.budget import PLATEAU, Budget, results_streak
from kernel_agent.config import OptimizeConfig
from kernel_agent.kernels import recheck
from kernel_agent.kernels.evaluate import EVALUATOR_SCHEMA, evaluate, evaluator_version
from kernel_agent.profiling.capture import capture_calls
from kernel_agent.report import write_report
from kernel_agent.scheduler import Policy, build_arms, plateau
from kernel_agent.selftest import RMSNorm, make_rmsnorm_capture
from kernel_agent.workspace import read_json, read_jsonl

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
    assert not slower["passed"] and slower["status"] == recheck.DISAGREES
    assert slower["agrees"]["speedup"] is False and "2.0x claimed" in slower["reason"]
    faster = judged(3.0, 2.0)  # the same band in the kernel's favour: disagrees too
    assert not faster["passed"] and faster["status"] == recheck.DISAGREES
    assert "ratio 1.5, outside 0.80-1.25" in faster["reason"]
    noisy = judged(1.5, 2.0, spread=0.2)  # noisy rounds widen the tolerance to 40 %
    assert noisy["passed"] and noisy["tolerance"] == 0.4
    capped = judged(1.4, 2.0, spread=0.6)  # ... at most to 50 %
    assert capped["passed"] and capped["tolerance"] == recheck.SPEEDUP_TOLERANCE_MAX
    assert not judged(1.2, 2.0, spread=0.6)["passed"]
    wrong = judged(None, 2.0, correct=False)
    assert not wrong["passed"] and wrong["agrees"] == {"correct": False}
    untimed = recheck.judge({"status": "ok", "correct": True}, None, 0.25)
    assert untimed["passed"] and untimed["agrees"] == {}
    # judged again against another verdict (a re-evaluation): from the measurement
    again = recheck.judge(slower, {"correct": True, "speedup": 1.05})
    assert again["passed"] and again["status"] == "ok" and "reason" not in again
    assert again["evaluator"]["speedup"] == 1.05 and again["speedup_ratio"] == 0.952
    # the integration keeps a correct kernel whose speedups disagree, as a warning
    warned = recheck.speed_warning(judged(1.0, 2.0))
    assert warned["passed"] and warned["status"] == recheck.SPEED_DISAGREES
    assert warned["conservative_speedup"] == 1.0 and "2.0x claimed" in warned["warning"]
    assert "reason" not in warned and "WARNING: the speedups" in recheck.describe(warned)
    rejudged = recheck.judge(warned, {"correct": True, "speedup": 1.1})
    assert rejudged["status"] == "ok" and "warning" not in rejudged
    assert "conservative_speedup" not in rejudged


def test_speedup_agreement_is_symmetric_with_a_floor_and_a_ceiling():
    tolerance, agree = recheck.speedup_tolerance, recheck.speedups_agree
    assert tolerance() == tolerance(0.0, None) == recheck.SPEEDUP_TOLERANCE == 0.25
    assert tolerance(0.2) == 0.4 and tolerance(0.01, 0.2) == 0.4  # twice the larger spread
    assert tolerance(0.472) == tolerance(3.0) == recheck.SPEEDUP_TOLERANCE_MAX == 0.5
    for a, b in ((1.0, 1.24), (1.0, 1.26), (2.0, 3.0), (18.433, 46.89), (2.921, 5.456)):
        for tol in (0.25, 0.4, 0.5):
            assert agree(a, b, tol) == agree(b, a, tol)  # whichever is larger
    assert agree(1.0, 1.24, 0.25) and not agree(1.0, 1.26, 0.25)
    # The two kernels of the VoxCPM2 re-integration (runs/voxcpm2-reintegrate.log):
    # attn_fused, 18.433x in separate processes (spread 0.052) vs 46.89x recorded by the
    # evaluator before #6/#7 (case spread 0.628), and vs 18.3x by the current one
    assert not agree(18.433, 46.89, tolerance(0.052, 0.628))
    assert agree(18.433, 18.3, tolerance(0.052, 0.01))
    # decoder_layer_fused, 2.921x (spread 0.472) vs 5.456x (0.012): it passed when twice
    # the spread made the band ±94 %; the ceiling flags it like any other disagreement
    assert not agree(2.921, 5.456, tolerance(0.472, 0.012))
    assert not agree(5.456, 2.921, tolerance(0.012, 0.472))


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


CURRENT = evaluator_version()


def kernel(run, target: str, speedup: float, *, name="v1", spread=None, version=CURRENT) -> str:
    """A correct kernel evaluation (by the evaluator ``version``, None: before it was
    recorded); returns its integration item ``target=<snapshot>``."""
    src = run.target(target) / "candidates" / f"{name}.py"
    src.write_text(f"# {target} {name}\n")
    snap = snapshot(run, src, target)
    cases = [] if spread is None else [{"timing_spread": spread}]
    result = {"status": "ok", "correct": True, "speedup": speedup, "cases": cases}
    if version is not None:
        result["evaluator_version"] = version
    record_candidate(run, target, src, snap, result, hypothesis=name)
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


def fakes(orch, measured: dict[str, tuple | list], fresh: dict[str, dict]):
    """A fake re-check (``measured``: snapshot stem -> (speedup, spread), or a list of what
    its re-checks measure in turn, the last one repeated; a string is a failure status)
    and re-evaluation (``fresh``: snapshot stem -> its result); returns the verdicts and
    re-evaluations seen."""
    seen: dict[str, list] = {"verdicts": [], "reevaluated": []}

    def stem(snap) -> str:
        return Path(snap).stem.split("_")[1]  # 001_v1_<sha1> -> v1

    def fake_recheck(capture, snap, *, verdict, capture_sha256, timeout):
        runs = measured[stem(snap)]
        runs = runs if isinstance(runs, list) else [runs]
        done = sum(s == stem(snap) for s, _ in seen["verdicts"])
        seen["verdicts"].append((stem(snap), verdict))
        run = runs[min(done, len(runs) - 1)]
        if isinstance(run, str):
            result = {"status": run, "correct": False, "seeds": 3, "cases": []}
            result["reason"] = f"case 0, seed 9: {run} on fresh inputs"
            return recheck.judge(result, verdict)
        result = {"status": "ok", "correct": True, "seeds": 3, "cases": [], "seed": done}
        result.update(speedup=run[0], timing_spread=run[1])
        return recheck.judge(result, verdict)

    def fake_reevaluate(capture, snap, *, timeout, capture_sha256):
        assert capture == orch.run.capture_file(Path(snap).parent.parent.name)
        assert capture_sha256 == truth.sha256_file(capture)
        seen["reevaluated"].append(stem(snap))
        return {"evaluator_version": CURRENT, **fresh[stem(snap)]}

    orch.rechecker, orch.reevaluator = fake_recheck, fake_reevaluate
    return seen


def ok(speedup: float, spread: float = 0.01, saved: float | None = None) -> dict:
    return {
        "status": "ok",
        "correct": True,
        "speedup": speedup,
        "cases": [{"timing_spread": spread}],
        "est_saved_ms_per_run": saved,
    }


def test_a_stale_record_is_re_evaluated_and_the_kernel_accepted(tmp_path, simulated):
    """attn_fused of the VoxCPM2 run: 46.89x recorded by the evaluator before #6/#7 (no
    evaluator_version), 18.3x by the current one, 18.433x in the re-check."""
    orch = make(tmp_path)
    run = orch.run
    attn = kernel(run, "attn", 46.89, spread=0.628, version=None)
    seen = fakes(orch, {"v1": (18.433, 0.052)}, {"v1": ok(18.31)})
    orch.worker = worker([])
    asyncio.run(orch.integrate())

    assert seen["reevaluated"] == ["v1"]  # up front: the record is stale
    assert seen["verdicts"] == [("v1", {"correct": True, "speedup": 18.31, "timing_spread": 0.01})]
    data = read_json(run.root / "integration.json")
    assert [a["item"] for a in data["accepted"]] == [attn]
    (check,) = data["recheck"]
    assert check["passed"] and check["status"] == "ok" and check["agrees"]["speedup"]
    again = check["reevaluated"]
    assert again["old_speedup"] == 46.89 and again["new_speedup"] == 18.31
    assert "before evaluator_version was recorded" in again["why"]
    # the re-evaluation is appended and stands in for the stale record
    records = truth.of(run).records(run.results_file("attn"))
    assert [r.get("ledger_status") for r in records] == ["keep", ledger.REEVALUATED]
    assert records[1]["reevaluates"] == {
        "exp": records[0]["exp"],
        "speedup": 46.89,
        "evaluator_version": None,
        "why": again["why"],
    }
    assert records[1]["evaluator_version"]["schema"] == EVALUATOR_SCHEMA
    assert records[1]["snapshot"] == records[0]["snapshot"] and records[1]["idea"] is None
    assert best_for_target(run, "attn")["speedup"] == 18.31
    rows = [r for r in ledger.rows(run) if r["target"] == "attn"]
    assert [(r["status"], r["speedup"]) for r in rows] == [("keep", 46.89), ("re-evaluated", 18.31)]
    assert "re-evaluation of exp" in rows[1]["hypothesis"]
    (target,) = [t for t in ledger.summary(run)["targets"] if t["id"] == "attn"]
    assert target["best_speedup"] == 18.31 and target["evals"] == 1  # not an agent's
    report = write_report(run).read_text()
    assert "re-evaluated (measured before evaluator_version" in report
    assert "46.89x recorded, 18.31x by the current evaluator" in report

    seen["reevaluated"].clear()  # the next integration finds a current record
    seen["verdicts"].clear()
    asyncio.run(orch.integrate())
    assert seen["reevaluated"] == [] and [v["speedup"] for _, v in seen["verdicts"]] == [18.31]

    # the keep bar of the target is the re-evaluated 18.31x, not the stale 46.89x
    assert ledger.best_kept([r for r in ledger.rows(run) if r["target"] == "attn"]) == 18.31
    kernel(run, "attn", 20.0, name="v2")
    assert ledger.rows(run)[-1]["status"] == ledger.KEEP


def test_a_re_evaluation_sets_the_bar_of_the_advice_and_the_scheduler(tmp_path, simulated):
    """#64: once attn_fused's stale 46.89x stands as its re-evaluated 18.31x, the evaluation
    advice's streak and the scheduler's arm measure against 18.31x, like the ledger: a later
    20x candidate is a new best (``keep``) and resets the streak."""
    orch = make(tmp_path)
    run = orch.run
    v1 = Path(kernel(run, "attn", 46.89, spread=0.628, version=None).partition("=")[2]).name
    for i, speedup in enumerate((15.0, 14.0), 2):  # not new bests: 46.89x set the bar
        kernel(run, "attn", speedup, name=f"v{i}")
    fakes(orch, {"v1": (18.433, 0.052)}, {"v1": ok(18.31)})
    orch.worker = worker([])
    asyncio.run(orch.integrate())
    results = run.results_file("attn")
    assert read_jsonl(results)[-1]["reevaluates"]["speedup"] == 46.89

    def attn():
        return next(a for a in build_arms(run, Policy(), []) if a.id == "attn")

    budget = Budget(run)
    budget.start_agent("kernel-attn")
    arm = attn()
    assert results_streak(results) == arm.streak == 2  # the re-evaluation is not an evaluation
    assert (arm.best, arm.best_snapshot, arm.evals) == (18.31, v1, 3)
    assert arm.gain_ms == pytest.approx(arm.ref_ms * (1 - 1 / 18.31))
    for i, speedup in enumerate((17.0, 18.4), 4):  # 18.4x: within the noise of 18.31x
        kernel(run, "attn", speedup, name=f"v{i}")
        advice = budget.feedback("kernel-attn", results, None)
    assert ledger.rows(run)[-1]["status"] == ledger.DISCARD
    assert advice["advice"] == "consider_stopping" and advice["budget"]["non_improving"] == PLATEAU
    assert attn().streak == PLATEAU and plateau(attn(), Policy(speedup_goal=None))

    kernel(run, "attn", 20.0, name="v6")
    assert ledger.rows(run)[-1]["status"] == ledger.KEEP
    advice = budget.feedback("kernel-attn", results, None)
    assert advice["advice"] == "continue" and advice["budget"]["non_improving"] == 0
    arm = attn()
    assert (arm.best, arm.streak, arm.evals) == (20.0, 0, 6)
    assert arm.gain_ms == pytest.approx(arm.ref_ms * (1 - 1 / 20.0))


@pytest.mark.parametrize(
    "fresh, retry, status",
    [
        (2.95, None, "ok"),  # the record was off: its re-evaluation agrees with the re-check
        (5.47, (5.484, 0.892), "ok"),  # the second re-check agrees
        (5.47, (2.927, 0.472), recheck.SPEED_DISAGREES),  # both disagree: a warning
    ],
)
def test_a_disagreeing_speedup_is_re_evaluated_and_re_checked(
    tmp_path, simulated, fresh, retry, status
):
    """decoder_layer_fused of the VoxCPM2 run: correct, 2.921x in the re-check (spread
    0.472) vs 5.456x recorded, 5.47x re-evaluated, and 5.484x or 2.927x in a second
    re-check (its forward_step takes 0.126 or 0.239 ms from one process to the next).
    Never refused: its end-to-end A/B decides, on the conservative speedup until then."""
    orch = make(tmp_path)
    item = kernel(orch.run, "attn", 5.456, spread=0.012)
    runs = [(2.921, 0.472)] + ([retry] if retry else [])
    seen = fakes(orch, {"v1": runs}, {"v1": ok(fresh, saved=100.0)})
    kept, _, (check,) = orch._recheck_kernels([("kernel", item)], None, {})

    assert seen["reevaluated"] == ["v1"]  # flagged: not passed by the noise of its rounds
    assert check["reevaluated"]["why"] == "the re-check measured 2.921x"
    assert check["evaluator"]["speedup"] == fresh and check["tolerance"] == 0.5
    assert check["passed"] and check["status"] == status and kept == [("kernel", item)]
    assert best_for_target(orch.run, "attn")["speedup"] == fresh  # the re-evaluation
    assert len(seen["verdicts"]) == (1 if retry is None else 2)
    if retry is not None:
        assert [r["speedup"] for r in check["rechecks"]] == [2.921, retry[0]]
        assert check["speedup"] == retry[0]  # the run that agrees better
    events = [e for e in ledger.events(orch.run) if e["event"] == "recheck_speed_disagrees"]
    if status == "ok":
        assert events == [] and orch.speed_caps == {}
        return
    assert check["conservative_speedup"] == 2.927 and "5.47x claimed" in check["warning"]
    assert [(e["evaluator"], e["recheck"], e["conservative"]) for e in events] == [
        (5.47, 2.927, 2.927)
    ]
    text = recheck.describe(check)
    assert "2.927x in separate processes vs 5.47x in the evaluator" in text
    assert "WARNING: the speedups disagree (ratio 0.535; re-checks: 2.921x, 2.927x)" in text
    assert "ranked by the conservative 2.927x" in text
    # ranked and projected by the conservative speedup
    snap = Path(item.partition("=")[2]).name
    assert orch.speed_caps == {("attn", snap): 2.927}
    best = orch._kernel_best("attn")
    assert best["speedup"] == 2.927 and best["evaluator_speedup"] == 5.47
    expected = 100.0 * (1 - 1 / 2.927) / (1 - 1 / 5.47)
    assert orch._kernel_saving("attn", snap) == pytest.approx(expected)


@pytest.mark.parametrize(
    "version, runs, status",
    [
        (CURRENT, [(2.921, 0.472), "incorrect"], "incorrect"),  # wrong on the second draw
        (CURRENT, ["integrity_violation"], "integrity_violation"),
        (None, [(0.9, 0.01)], recheck.DISAGREES),  # stale, and no speedup at all
        (None, [(1.5, 0.01)], recheck.SPEED_DISAGREES),  # stale, slower but a speedup
    ],
)
def test_what_the_integration_still_refuses(tmp_path, simulated, version, runs, status):
    orch = make(tmp_path)
    item = kernel(orch.run, "attn", 5.0, version=version)
    seen = fakes(orch, {"v1": runs}, {"v1": ok(5.1)})
    kept, _, (check,) = orch._recheck_kernels([("kernel", item)], None, {})
    assert check["status"] == status
    assert len(seen["verdicts"]) == (1 if status == "integrity_violation" else 2)
    refused = status != recheck.SPEED_DISAGREES
    assert check["passed"] is not refused and kept == ([] if refused else [("kernel", item)])
    if status == "incorrect":
        assert check["reason"].startswith("on a second re-check: case 0, seed 9: incorrect")
    if status == recheck.DISAGREES:
        assert check["reason"].endswith("no speedup at all in separate processes")
    failed = [e["status"] for e in ledger.events(orch.run) if e["event"] == "recheck_failed"]
    assert failed == ([status] if refused else [])


def test_a_speed_cap_can_rank_another_snapshot_first(tmp_path, simulated):
    """v1 records 10x but re-checks at 4x twice (re-evaluated 10x): kept with a warning
    at 4x, it ranks behind v2 (8x), which is re-checked and integrated instead."""
    orch = make(tmp_path)
    run = orch.run
    kernel(run, "attn", 8.0, name="v2")
    v1 = kernel(run, "attn", 10.0, name="v1")
    seen = fakes(orch, {"v1": (4.0, 0.0), "v2": (7.9, 0.0)}, {"v1": ok(10.0)})
    kept, _, checks = orch._recheck_kernels([("kernel", v1)], None, {})
    assert [c["status"] for c in checks] == [recheck.SPEED_DISAGREES, "ok"]
    assert [Path(a).stem.split("_")[1] for _, a in kept] == ["v2"]
    assert [s for s, _ in seen["verdicts"]] == ["v1", "v1", "v2"]


def test_a_failed_re_evaluation_refuses_the_kernel(tmp_path, simulated):
    orch = make(tmp_path)
    item = kernel(orch.run, "attn", 3.0, version=None)
    broken = {"status": "incorrect", "correct": False, "error": "case 1: max abs err 0.5"}
    seen = fakes(orch, {"v1": (3.0, 0.0)}, {"v1": broken})
    kept, _, (check,) = orch._recheck_kernels([("kernel", item)], None, {})
    assert kept == [] and not check["passed"] and check["status"] == "reevaluation_failed"
    assert seen["verdicts"] == []  # never re-checked against a stale claim
    assert "re-evaluation: incorrect: case 1" in check["reason"]
    assert "3.0x recorded, incorrect by the current evaluator" in recheck.describe(check)
    assert best_for_target(orch.run, "attn") is None  # its current record is not correct


def test_a_re_evaluation_can_rank_another_snapshot_first(tmp_path, simulated):
    """v1 claimed 46.89x (stale) and ranked first; re-evaluated, it is 18.31x, behind v2
    (20x, current): v2 is re-checked and integrated instead."""
    orch = make(tmp_path)
    run = orch.run
    kernel(run, "attn", 20.0, name="v2")
    v1 = kernel(run, "attn", 46.89, name="v1", version=None)
    items, digests = orch._integration_items()
    assert items == [("kernel", v1)]
    seen = fakes(orch, {"v1": (18.4, 0.0), "v2": (19.8, 0.0)}, {"v1": ok(18.31)})
    kept, _, checks = orch._recheck_kernels(items, None, {}, digests)
    (v2,) = [a for _, a in kept]
    assert Path(v2).stem.split("_")[1] == "v2" and [c["passed"] for c in checks] == [True, True]
    assert seen["reevaluated"] == ["v1"] and [s for s, _ in seen["verdicts"]] == ["v1", "v2"]
    assert digests[v2.partition("=")[2]] == truth.sha256_file(Path(v2.partition("=")[2]))


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
