"""Parameter sweeps (issue #19): configs of ``build(reference, **config)`` under one GPU lock.

CPU: the sweep itself (quick-tier checks, a simulated stand-in for the CUDA timer), the
``sweep_candidate`` tool's records and budget, and the real subprocesses (a config
that kills its process).  GPU (``gpu``): a Triton RMSNorm with ``BLOCK`` / ``num_warps``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.overrides import TorchFunctionMode

from kernel_agent import cli, gpulock, ledger, truth
from kernel_agent.agent import prompts
from kernel_agent.agent import tools as tools_mod
from kernel_agent.budget import Budget
from kernel_agent.kernels import evaluate as evaluate_mod
from kernel_agent.kernels import sweep as sweep_mod
from kernel_agent.profiling.capture import capture_calls
from kernel_agent.selftest import RMSNorm, make_rmsnorm_capture
from kernel_agent.workspace import RunDir, write_json

# `repeat` changes the speed, not the results; `scale` != 1 breaks correctness; `crash`
# kills the process; `poke` changes the reference's weights behind the version counter.
TOY = """
import os

import torch
from torch import nn


class Fast(nn.Module):
    def __init__(self, reference, repeat, scale):
        super().__init__()
        self.weight = reference.weight
        self.eps = reference.variance_epsilon
        self.repeat = repeat
        self.scale = scale

    def forward(self, x):
        for _ in range(self.repeat):
            h = x.float()
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
            y = self.weight * h.to(x.dtype)
        return y * self.scale


def build(reference, repeat=1, scale=1.0, crash=False, poke=False):
    if crash:
        os._exit(17)
    if poke:
        reference.weight.data.mul_(2)
    return Fast(reference, repeat, scale)
"""


def toy_capture(path: Path) -> Path:
    torch.manual_seed(0)
    module = RMSNorm(64, eps=1e-5)
    with torch.no_grad():
        module.weight.copy_(torch.randn(64) * 0.1 + 1)
    calls = [((torch.randn(1, 16, 64),), {}, 1), ((torch.randn(1, 1, 64),), {}, 7)]
    calls.append(((torch.randn(1, 32, 64),), {}, 0))  # correctness only: never timed
    capture_calls(module.eval(), calls, path)
    return path


class TorchCalls(TorchFunctionMode):
    """Counts the torch functions and tensor methods called under it."""

    def __init__(self) -> None:
        super().__init__()
        self.n = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.n += 1
        return func(*args, **(kwargs or {}))


def cpu_timer(fn, args, kwargs, *, l2_flush=False, target_ms=60.0, keep=False):
    """Stand-in for kernels.bench.time_call on the CPU, in simulated time: 1 us per torch
    call. The configs differ in the work they do (``repeat``); wall-clock times of a CPU
    toy on a busy machine could put them in another order."""
    with torch.inference_mode(), TorchCalls() as calls:
        fn(*args, **kwargs)
    return {"median_ms": calls.n * 1e-3}


# ------------------------------------------------------------------ configs


def test_configs_lists_grids_limits_and_binding():
    configs, notes = sweep_mod.configs_from([{"BLOCK": 512}, {"BLOCK": 512}, {}])
    assert configs == [{"BLOCK": 512}, {}] and notes == ["1 duplicate config(s) dropped"]
    grid, _ = sweep_mod.configs_from('{"BLOCK": [256, 512], "num_warps": [4, 8]}')
    assert grid[1] == {"BLOCK": 256, "num_warps": 8} and len(grid) == 4
    many, notes = sweep_mod.configs_from({"B": list(range(100))}, 200)
    assert len(many) == sweep_mod.MAX_CONFIGS and "at most 64" in notes[0]
    assert len(sweep_mod.configs_from({"B": list(range(40))})[0]) == 32  # the default
    assert len(sweep_mod.configs_from({"B": [1, 2, 3]}, 2)[0]) == 2
    for bad in ([], [1], [{"not a name": 1}], [{"class": 1}], [{"B": [1]}], [{"B": 1e999}], "{"):
        with pytest.raises(ValueError):
            sweep_mod.configs_from(bad)
    with pytest.raises(ValueError):
        sweep_mod.configs_from({"B": []})

    source = "def build(reference, BLOCK=64, num_warps=4):\n    return (BLOCK, num_warps)\n"
    assert sweep_mod.bind_config(source, {}) == source  # the default config: the file as is
    scope: dict[str, Any] = {}
    exec(sweep_mod.bind_config(source, {"BLOCK": 256}), scope)
    assert scope["build"](None) == (256, 4)  # bound as the default ...
    assert scope["build"](None, BLOCK=32, num_warps=8) == (32, 8)  # ... still overridable
    assert sweep_mod.label({"BLOCK": 256, "mode": "a"}) == "BLOCK=256, mode='a'"
    assert sweep_mod.label({}) == "default"


# ------------------------------------------------------------------ the sweep


def test_sweep_checks_every_config_and_sorts_the_passing_ones_by_speed(tmp_path):
    capture = toy_capture(tmp_path / "c.pt")
    candidate = tmp_path / "toy.py"
    candidate.write_text(TOY)
    configs = [
        {"repeat": 8},
        {"scale": 1.1},  # wrong results
        {"repeat": 1},
        {"poke": True},  # changes the weights the configs share
        {"nope": 1},  # not a keyword argument of build()
        {"repeat": 32},
    ]
    events: list[dict[str, Any]] = []
    out = sweep_mod.sweep(
        capture, candidate, configs, device="cpu", timer=cpu_timer, emit=events.append
    )
    table = out["table"]
    assert [r["index"] for r in table] == [2, 0, 5, 1, 3, 4]  # fastest first, then rejected
    speedups = [r["speedup"] for r in table[:3]]
    assert speedups == sorted(speedups, reverse=True) and speedups[0] > 3 * speedups[2]
    assert all(r["correct"] and len(r["cases"]) == 2 for r in table[:3])  # the timed cases
    assert table[3]["status"] == "incorrect" and "output" in table[3]["error"]
    assert table[4]["status"] == "integrity_violation" and "weights" in table[4]["error"]
    assert table[5]["status"] == "build_error" and "'nope'" in table[5]["error"]
    # the quick tier: the smallest and the largest case, including the untimed one
    assert out["quick_cases"] == ["a0[1, 1, 64]:float32", "a0[1, 32, 64]:float32"]
    assert [c["calls_per_run"] for c in out["cases"]] == [1, 7] and out["rounds"] == 3
    # the weights were restored after `poke`: the configs after it still pass
    assert next(r for r in table if r["index"] == 5)["correct"]
    rows = [e for e in events if e["event"] == "row"]
    assert [e["index"] for e in rows] == list(range(6))

    compact = sweep_mod.compact_row(table[0])
    assert compact["config"] == {"repeat": 1} and len(compact["speedup_per_case"]) == 2
    assert sweep_mod.compact_row(table[5])["status"] == "build_error"

    late = sweep_mod.sweep(capture, candidate, configs[:2], device="cpu", deadline=0.0)
    assert {r["status"] for r in late["table"]} == {"skipped"}  # out of time: not checked
    untimed = sweep_mod.sweep(capture, candidate, configs[:3], device="cpu")
    assert untimed["timing"] == "skipped: no CUDA device"
    assert [r["index"] for r in untimed["table"]] == [0, 2, 1]  # passing in the given order


# ------------------------------------------------------------------ the tool


@pytest.fixture
def private_lock(tmp_path, monkeypatch):
    """The GPU lock in a temporary directory; counts acquisitions."""
    monkeypatch.setattr(gpulock, "CACHE_DIR", tmp_path / "locks")
    monkeypatch.delenv(gpulock.ENV, raising=False)
    taken = []
    acquire = gpulock._acquire

    def counted(name, gpus, *args):
        taken.append(name)
        return acquire(name, gpus, *args)

    monkeypatch.setattr(gpulock, "_acquire", counted)
    monkeypatch.setattr(sweep_mod, "ensure_peaks", lambda: None)
    return taken


def sealed_target(tmp_path: Path) -> tuple[RunDir, truth.Truth]:
    run = RunDir.create(tmp_path, "org/m")
    write_json(run.run_json, {"card": {"repo_id": "org/m"}, "truth": truth.new_section()})
    keeper = truth.of(run)
    capture = run.capture_file("t")
    capture.parent.mkdir(parents=True, exist_ok=True)
    toy_capture(capture)
    keeper.seal(capture)
    write_json(run.target("t") / "spec.json", {"id": "t", "module_class": "RMSNorm"})
    (run.target("t") / "candidates").mkdir(parents=True)
    (run.target("t") / "candidates" / "toy.py").write_text(TOY)
    return run, keeper


def test_sweep_tool_records_the_best_config_as_one_evaluation(tmp_path, monkeypatch, private_lock):
    run, keeper = sealed_target(tmp_path)
    full = []

    def spawn(capture_path, swept, configs, indices, *, capture_sha256, **_):
        """The sweep subprocess, in this process (CPU, wall-clock timer)."""
        result = sweep_mod.sweep(
            capture_path,
            swept,
            configs,
            indices=indices,
            device="cpu",
            capture_sha256=capture_sha256,
            timer=cpu_timer,
        )
        return {"rows": {}, "result": result, "culprit": None, "timed_out": False, "error": ""}

    def run_evaluation(capture_path, path, *, capture_sha256, timeout, **_):
        """The full evaluator on the CPU, plus the speedup a GPU would measure."""
        assert gpulock._held_gpus() == {"gpu": 0}  # still under the sweep's lock
        full.append(path)
        result = evaluate_mod.evaluate(
            capture_path, path, device="cpu", capture_sha256=capture_sha256
        )
        return result | {"speedup": 2.0} if result["correct"] else result

    monkeypatch.setattr(sweep_mod, "_spawn", spawn)
    monkeypatch.setattr(sweep_mod, "run_evaluation", run_evaluation)
    monkeypatch.setattr(tools_mod, "create_sdk_mcp_server", lambda n, version, tools: tools)
    monkeypatch.setattr(tools_mod, "refresh", lambda *a: None)
    budget = Budget(run, eval_timeout_s=60.0)
    session = tools_mod.SessionBinding(evaluations=5)
    server = {t.name: t for t in tools_mod.build_server(run, budget, keeper, session)}

    def call(tool: str, **args: Any) -> dict[str, Any]:
        out = asyncio.run(server[tool].handler(args))
        return json.loads(out["content"][0]["text"])

    configs = [{"repeat": 16}, {"repeat": 1}, {"scale": 1.1}, {"repeat": 4}]
    out = call(
        "sweep_candidate",
        target_id="t",
        candidate="candidates/toy.py",
        configs=configs,
        hypothesis="fewer passes are faster",
        idea_id="passes",
    )
    assert private_lock == ["gpu"]  # one acquisition for the configs and the full evaluation
    assert out["status"] == "ok" and out["correct"] and out["speedup"] == 2.0
    assert out["config"] == {"repeat": 1} and out["snapshot"].startswith("history/001_toy_")
    table = out["sweep"]["table"]
    assert [r["config"] for r in table] == [
        {"repeat": 1},
        {"repeat": 4},
        {"repeat": 16},
        {"scale": 1.1},
    ]
    speedups = [r["speedup"] for r in table[:3]]
    assert speedups == sorted(speedups, reverse=True)
    assert table[3]["status"] == "incorrect" and "output" in table[3]["error"]
    assert "speedup" not in table[3]
    assert out["sweep"]["passed"] == 3 and out["sweep"]["failed"] == 1

    # recorded once, as a normal evaluation of the bound snapshot
    (record,) = keeper.records(run.results_file("t"))
    snap = run.history_dir("t") / Path(record["snapshot"]).name
    assert full == [snap] and out["snapshot"] == record["snapshot"]
    assert "_KA_SWEEP_CONFIG = {'repeat': 1}" in snap.read_text()
    assert keeper.snapshot_ok(snap, record["snapshot_sha256"])
    assert record["config"] == {"repeat": 1} and len(record["sweep"]["table"]) == 4
    assert record["idea"] == "passes" and record["hypothesis"].startswith("fewer passes")
    (row,) = ledger.rows(run)
    assert row["status"] == "keep" and row["speedup"] == 2.0
    assert row["hypothesis"] == "fewer passes are faster [sweep: repeat=1; best of 3/4 configs]"
    assert tools_mod.best_for_target(run, "t", keeper)["config"] == {"repeat": 1}
    assert out["budget"]["evals_used"] == 1 and out["advice"] == "continue"
    assert out["idea"]["tries"] == 1

    # no config passes: the first failure stands for the sweep, nothing else runs
    out = call(
        "sweep_candidate",
        target_id="t",
        candidate="candidates/toy.py",
        configs={"scale": [1.5, 2.0]},
        hypothesis="h",
    )
    assert out["status"] == "incorrect" and not out["correct"] and len(full) == 1
    assert out["config"] == {"scale": 1.5} and "no config of the sweep passed" in out["error"]
    assert ledger.rows(run)[-1]["status"] == "incorrect"
    assert out["budget"]["evals_used"] == 2  # each sweep counts once
    assert private_lock == ["gpu", "gpu"]

    refused = call("sweep_candidate", target_id="t", candidate="candidates/toy.py",
                   configs=[{"x": [1]}], hypothesis="h")  # fmt: skip
    assert refused["status"] == "error" and "not a number" in refused["error"]
    assert len(keeper.records(run.results_file("t"))) == 2


def test_sweep_subprocesses_restart_without_a_config_that_kills_them(
    tmp_path, monkeypatch, capsys, private_lock
):
    """The real subprocesses on the CPU (children see no GPU), through the CLI."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(evaluate_mod, "ensure_peaks", lambda: None)
    capture = toy_capture(tmp_path / "c.pt")
    candidate = tmp_path / "toy.py"
    candidate.write_text(TOY)
    spec = tmp_path / "configs.json"
    spec.write_text(json.dumps([{"repeat": 2}, {"crash": True}, {"repeat": 1}]))

    code = cli.main(["eval", str(capture), str(candidate), "--sweep", str(spec), "--timeout", "60"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert code == 0 and data["evaluation"]["correct"], data
    info = data["sweep"]
    assert info["runs"] == 2 and info["status"] == "ok"
    assert [r["index"] for r in info["table"]] == [0, 2, 1]  # untimed on the CPU: given order
    crashed = info["table"][2]
    assert crashed["status"] == "crash" and "exit code 17" in crashed["error"]
    assert data["config"] == {"repeat": 2}
    assert data["evaluation"]["timing"] == "skipped: no CUDA device"
    assert data["evaluation"]["parent_check"] == "ok"  # the full evaluator, outside too
    assert "sweep: 3 configs, 2 passed, 1 rejected" in captured.err
    assert private_lock == ["gpu"]


def test_prompt_and_program_describe_sweeps():
    assert "sweep_candidate" in prompts.knowledge("playbook.md")
    assert "One sweep per idea" in (Path(prompts.__file__).parent / "program.md").read_text()


# ------------------------------------------------------------------ GPU

TRITON_LOOPED = '''
"""RMSNorm in Triton, a row per program in BLOCK-wide chunks: BLOCK and num_warps tune."""
import torch
import triton
import triton.language as tl
from torch import nn


@triton.jit
def _rmsnorm(x_ptr, w_ptr, y_ptr, stride, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + row * stride + cols, mask=cols < n_cols, other=0.0).to(tl.float32)
        acc += x * x
    rstd = tl.rsqrt(tl.sum(acc, axis=0) / n_cols + eps)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0)
        tl.store(y_ptr + row * stride + cols, (x * rstd).to(w.dtype) * w, mask=mask)


class LoopedRMSNorm(nn.Module):
    def __init__(self, reference, block, num_warps):
        super().__init__()
        self.weight = reference.weight
        self.eps = float(reference.variance_epsilon)
        self.block, self.num_warps = block, num_warps

    def forward(self, hidden_states):
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1]).contiguous()
        y = torch.empty_like(x)
        _rmsnorm[(x.shape[0],)](
            x, self.weight, y, x.stride(0), x.shape[-1], self.eps,
            BLOCK=self.block, num_warps=self.num_warps,
        )
        return y.view(shape)


def build(reference, BLOCK=1024, num_warps=4):
    return LoopedRMSNorm(reference, BLOCK, num_warps)
'''


@pytest.mark.gpu
def test_triton_rmsnorm_sweep_records_the_best_config(tmp_path, monkeypatch):
    monkeypatch.setenv(gpulock.ENV, "1")  # conftest holds this process's GPU lock
    run = RunDir.create(tmp_path, "org/m")
    write_json(run.run_json, {"card": {"repo_id": "org/m"}, "truth": truth.new_section()})
    keeper = truth.of(run)
    capture = make_rmsnorm_capture(run.capture_file("rms"), hidden=1024)
    keeper.seal(capture)
    write_json(run.target("rms") / "spec.json", {"id": "rms", "module_class": "RMSNorm"})
    (run.target("rms") / "candidates").mkdir(parents=True)
    (run.target("rms") / "candidates" / "looped.py").write_text(TRITON_LOOPED)
    monkeypatch.setattr(tools_mod, "create_sdk_mcp_server", lambda n, version, tools: tools)
    server = {t.name: t for t in tools_mod.build_server(run, Budget(run), keeper)}
    configs = [
        {"BLOCK": 128, "num_warps": 1},
        {"BLOCK": 256, "num_warps": 2},
        {"BLOCK": 512, "num_warps": 4},
        {"BLOCK": 1024, "num_warps": 4},
        {"BLOCK": 1024, "num_warps": 8},
        {"BLOCK": 384, "num_warps": 4},  # not a power of two: Triton refuses it
    ]
    args = {"target_id": "rms", "candidate": "candidates/looped.py", "configs": configs}
    out = asyncio.run(server["sweep_candidate"].handler(args | {"hypothesis": "tile size"}))
    out = json.loads(out["content"][0]["text"])

    table = out["sweep"]["table"]
    assert out["sweep"]["passed"] == 5 and out["sweep"]["rounds"] == sweep_mod.ROUNDS, out
    speedups = [r["speedup"] for r in table[:5]]
    assert speedups == sorted(speedups, reverse=True) and all(s > 0 for s in speedups)
    assert all(len(r["speedup_per_case"]) == 2 for r in table[:5])
    assert all(r.get("pct_of_sol") for r in table[:5]), table  # peaks are measured first
    assert table[5]["config"] == {"BLOCK": 384, "num_warps": 4} and not table[5]["correct"]
    assert table[5]["status"] in ("runtime_error", "build_error") and table[5]["error"]

    assert out["status"] == "ok" and out["correct"] and out["config"] == table[0]["config"]
    assert out["speedup"] > 0 and "integrity" not in str(out.get("stage"))
    (record,) = keeper.records(run.results_file("rms"))
    assert record["config"] == table[0]["config"] and record["speedup"] == out["speedup"]
    snap = run.history_dir("rms") / Path(record["snapshot"]).name
    assert f"_KA_SWEEP_CONFIG = {table[0]['config']!r}" in snap.read_text()
    assert len(ledger.rows(run)) == 1 and out["budget"]["evals_used"] == 1
