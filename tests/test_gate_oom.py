"""Out of GPU memory inside the perceptual gate is no quality verdict (#137).

In the final integration of runs/openbmb--VoxCPM2/20261005-192504 the gate's scoring models
(Whisper-large-v3, WavLM, UTMOS) loaded next to both states of the in-process A/B ran out of
memory, the gate caught the error and the step was recorded as ``perceptual: the perceptual
gate failed: OutOfMemoryError``: a quality failure, so #112's retry in separate processes
never ran and the new FP8 kernel ``enc_dit_stack_fp8`` was dropped without a verdict. Now

* out of GPU memory in a check that catches its own errors (held-out run, natural-length
  run, perceptual gate) makes the step ``oom``, which the integration measures again in
  separate processes and never reuses, also from an ``integration.json`` written before;
* ``e2e_ab`` frees A's state after B's last run (the gate's samples), before the scoring
  models load;
* ``report.md`` shows an ``oom`` step as not measurable, never as a failed gate.

CPU: the worker on the perceptual toy (``perceptual_toy.py``), the A/B session on the toy
decoder, the integration with a fake measurement layer. GPU (last test): a real CUDA
out-of-memory error in the gate and A's CUDA memory freed before it scores.
"""

from __future__ import annotations

import asyncio
import gc
import json
import sys
import weakref

import pytest
import torch
from test_integrate import BASE, WIGGLE, kernel, label, make, paired, parse, write
from test_integrate import e2e as e2e_record
from test_perceptual import _call, _cpu, _sealed, _transform
from test_undo import NORM, _file, _out, toy  # noqa: F401 (fixture)

from kernel_agent import abtest, dryrun, ledger, toolchain, truth, worker
from kernel_agent.integrate import ab
from kernel_agent.integrate.patcher import KernelPatch
from kernel_agent.report import write_report
from kernel_agent.workloads.holdout import outputs_equal
from kernel_agent.workspace import read_json

#: What ``torch`` said in the gate of the issue's run (step "+ enc_dit_stack_fp8").
OOM_TEXT = (
    "CUDA out of memory. Tried to allocate 22.00 MiB. GPU 0 has a total capacity of "
    "15.56 GiB of which 8.56 MiB is free. Including non-PyTorch memory, this process has "
    "14.68 GiB memory in use."
)
OOM_LINE = f"torch.OutOfMemoryError: {OOM_TEXT}"
#: ``metrics.perceptual`` of that step, as ``perceptual.check`` records a scorer's error.
GATE_RECORD = {
    "passed": False,
    "reason": f"the perceptual gate failed: OutOfMemoryError: {OOM_TEXT}"[:500],
    "error": "Traceback (most recent call last):\n"
    '  File "kernel_agent/workloads/perceptual.py", line 247, in score_tts\n'
    "    texts = _transcribe(waves, languages, dev)\n"
    f"{OOM_LINE}\n",
}


# ------------------------------------------------------------------ classification


def test_out_of_memory_in_a_check_record_and_a_step():
    assert abtest.out_of_memory(GATE_RECORD) == OOM_LINE  # the traceback's line first
    assert abtest.out_of_memory({"reason": GATE_RECORD["reason"]}) is not None
    quality = {"passed": False, "reason": "error rate 0.300 vs eager 0.000 (+0.300)"}
    assert abtest.out_of_memory(quality) is None and abtest.out_of_memory(None) is None
    crashed = {
        "passed": False,
        "reason": "the held-out run failed: AcceleratorError: CUDA "
        "error: an illegal memory access was encountered",
    }
    assert abtest.out_of_memory(crashed) is None  # a real failure stays one

    metrics = {"teacher_forced": {"passed": True}, "holdout": crashed, "perceptual": GATE_RECORD}
    assert abtest.checks_out_of_memory(metrics) == {"perceptual": OOM_LINE}
    assert abtest.checks_out_of_memory(None) == {} and abtest.checks_out_of_memory({}) == {}

    # the history entry of the issue's integration.json: status ok, a "failed" gate
    legacy = {
        "status": "ok",
        "passed": False,
        "reason": f"perceptual: {GATE_RECORD['reason']}",
        "metrics": {"teacher_forced": {"passed": True}, "perceptual": GATE_RECORD},
        "ab": {"mode": "paired", "rounds": 8, "accepted": False, "why": "B won 0/8 rounds"},
    }
    gate = f"out of GPU memory in the perceptual gate: {OOM_LINE}"
    assert abtest.step_out_of_memory(legacy) == gate
    assert abtest.step_out_of_memory({"status": abtest.OOM, "reason": gate}) == gate
    a_oom = {"status": "ok", "passed": True, "ab": {"why": f"A measured again: {gate}"}}
    assert abtest.step_out_of_memory(a_oom) == f"A measured again: {gate}"
    rejected = {**legacy, "reason": "perceptual: " + quality["reason"], "metrics": {}}
    assert abtest.step_out_of_memory(rejected) is None


def test_the_worker_makes_a_check_out_of_memory_an_oom_step():
    held = {
        "passed": False,
        "reason": "the held-out run failed: OutOfMemoryError: x",
        "error": f"Traceback\n{OOM_LINE}",
    }
    verdict = {
        "status": "ok",
        "passed": False,
        "reason": "held-out input: the held-out run failed: OutOfMemoryError: x",
        "metrics": {"teacher_forced": {"passed": True}, "holdout": held},
    }
    r = worker._out_of_memory(verdict)
    assert r["status"] == abtest.OOM and r["passed"] is False and r["status"] in abtest.FALLBACK
    assert r["reason"] == f"out of GPU memory in the held-out input: {OOM_LINE}"
    assert r["metrics"]["holdout"]["oom"] and r["metrics"]["holdout"]["passed"] is None
    assert r["metrics"]["holdout"]["reason"] == f"out of GPU memory, not judged: {OOM_LINE}"
    assert verdict["metrics"]["holdout"] is held and "oom" not in held  # not modified
    assert ledger.classify(r, 1.0, e2e=True) == "oom"
    ok = {"status": "ok", "passed": True, "reason": "", "metrics": {"holdout": {"passed": True}}}
    assert worker._out_of_memory(ok) is ok
    failed = {"status": "runtime_error", "passed": False, "reason": "x", "metrics": {}}
    assert worker._out_of_memory(failed) is failed


# ------------------------------------------------------------------ the worker, end to end

#: B's scoring models do not fit (what Whisper-large-v3 next to two VoxCPM2 states did).
SCORERS_OOM = (
    "    def perceptual_quality(samples):\n"
    f"        raise torch.OutOfMemoryError({OOM_TEXT!r})\n\n"
    "    workload.perceptual_quality = perceptual_quality\n"
)
#: A scorer that fails otherwise: still the gate's failure.
SCORERS_FAIL = (
    "    def perceptual_quality(samples):\n"
    "        raise RuntimeError('the scorer exploded')\n\n"
    "    workload.perceptual_quality = perceptual_quality\n"
)


def test_an_oom_in_the_perceptual_gate_is_an_oom_step(tmp_path, monkeypatch, capsys):
    _cpu(monkeypatch)
    run = _sealed(tmp_path, "near-lossless")
    keeper = truth.of(run)
    baseline = _call(capsys, run, "analyze", "--no-profile", "--iters", 1)
    keeper.seal_baseline(baseline["median_ms"])
    oom = _transform(tmp_path, "scorers_oom", SCORERS_OOM)
    gate = f"out of GPU memory in the perceptual gate: {OOM_LINE}"

    r = _call(capsys, run, "e2e", "--transform", oom, "--iters", 1, *keeper.worker_args())
    assert r["status"] == abtest.OOM and r["passed"] is False and r["reason"] == gate, r
    record = r["metrics"]["perceptual"]
    assert record["oom"] and record["passed"] is None and OOM_LINE in record["error"]
    assert "perceptual gate failed" not in json.dumps(r)
    assert r["metrics"]["teacher_forced"]["passed"] and r["metrics"]["holdout"]["passed"]
    assert ledger.classify(r, 1.0, e2e=True) == "oom"
    assert abtest.step_out_of_memory(r) == gate

    # the in-process A/B: the rounds ran, the gate did not fit -> separate processes
    r = _call(capsys, run, "e2e_ab", "--rounds", 2, "--b-transform", oom, *keeper.worker_args())
    assert r["status"] == abtest.OOM and r["status"] in abtest.FALLBACK and r["reason"] == gate
    assert r["ab"]["rounds"] == 2 and r["ab"]["released_gb"] == 0.0  # no CUDA here

    # any other error of the scorers is still the gate's failure
    fail = _transform(tmp_path, "scorers_fail", SCORERS_FAIL)
    r = _call(capsys, run, "e2e", "--transform", fail, "--iters", 1, *keeper.worker_args())
    assert r["status"] == "ok" and not r["passed"]
    assert (
        r["reason"] == "perceptual: the perceptual gate failed: RuntimeError: the scorer exploded"
    )


#: Shared by A's and B's transforms (their directory is on sys.path when they load).
EVENTS = "LOG = []\n"
#: A's state: an object that says when it is freed.
A_STATE = """import gate_events_137 as events


class Held:
    def __del__(self):
        events.LOG.append("A freed")


def apply(workload):
    workload.a_state = Held()
"""
#: B: says when the model runs and when the scoring models score.
B_STATE = """import gate_events_137 as events


def apply(workload):
    run, quality = workload.run, workload.perceptual_quality

    def logged_run(inputs):
        events.LOG.append("B runs")
        return run(inputs)

    def logged_quality(samples):
        events.LOG.append("scored")
        return quality(samples)

    workload.run = logged_run
    workload.perceptual_quality = logged_quality
"""


def test_e2e_ab_frees_a_after_bs_last_run_before_the_scorers(tmp_path, monkeypatch, capsys):
    """A's state is freed after B's checks and the gate's samples (B never runs on memory
    that moved under it, #112) and before the scoring models load."""
    _cpu(monkeypatch)
    run = _sealed(tmp_path, "near-lossless")
    keeper = truth.of(run)
    baseline = _call(capsys, run, "analyze", "--no-profile", "--iters", 1)
    keeper.seal_baseline(baseline["median_ms"])
    _file(tmp_path, "gate_events_137.py", EVENTS)
    a = _file(tmp_path, "a_state.py", A_STATE)
    b = _file(tmp_path, "b_state.py", B_STATE)
    monkeypatch.delitem(sys.modules, "gate_events_137", raising=False)

    r = _call(
        capsys,
        run,
        "e2e_ab",
        "--rounds",
        2,
        "--transform",
        a,
        "--b-transform",
        b,
        *keeper.worker_args(),
    )
    assert r["status"] == "ok" and r["passed"], r
    assert r["ab"]["released_gb"] == 0.0 and r["metrics"]["perceptual"]["passed"]
    log = sys.modules["gate_events_137"].LOG
    assert log.count("A freed") == 1 and log.count("scored") == 1
    assert log[-3:] == ["B runs", "A freed", "scored"], log  # the gate's last sample first

    # exact mode: no gate, nothing to free
    r = _call(
        capsys,
        run,
        "e2e_ab",
        "--rounds",
        2,
        "--transform",
        a,
        "--b-transform",
        b,
        *keeper.worker_args()[:-2],
        "--quality",
        "exact",
    )
    assert r["status"] == "ok" and "released_gb" not in r["ab"]


def test_keep_frees_the_other_state_and_the_original_modules(tmp_path, toy):  # noqa: F811
    """What only the undo handles held goes: A's state (applied afresh for each state, like
    a CUDA graph and its pool), the original modules both states' kernels replaced."""
    norm = _file(tmp_path, "norm.py", NORM)
    wrap = _file(
        tmp_path,
        "wrap.py",
        "from torch import nn\n\n\nclass Wrapped(nn.Module):\n"
        "    def __init__(self, inner):\n        super().__init__()\n        self.inner = inner\n\n"
        "    def forward(self, x):\n        return self.inner(x)\n\n\n"
        "def apply(workload):\n    workload.model.norm = Wrapped(workload.model.norm)\n",
    )
    patches = {"norm": KernelPatch("norm", "ToyRMSNorm", norm)}
    session = ab.Session(toy, lambda items: [patches[i.partition("=")[0]] for i in items])
    original = weakref.ref(toy.model.layers[0].input_layernorm)
    a = session.build("A", [f"norm={norm}"], [str(wrap)])
    a_wrapped = weakref.ref(toy.model.norm)
    b = session.build("B", [f"norm={norm}"], [str(wrap)], on=a)
    assert b.shared == 1 and toy.model.norm is not a_wrapped()
    b_out = _out(toy)
    session.to(a)
    session.to(b)
    assert a_wrapped() is not None and original() is not None  # held by the handles

    session.keep(b, a)
    gc.collect()
    assert a_wrapped() is None and original() is None
    assert a.handles == b.handles == session.applied == [] and not a.items and not b.items
    assert outputs_equal(_out(toy), b_out)  # the model stays in B
    assert b.report.replaced == {"norm": 5} and a.report.transforms == ["wrap"]


# ------------------------------------------------------------------ the integration

#: Latency factor of each item (product over a set).
FACTOR = {"attn": 0.7, "cfm": 0.6}


def _ms(items: list[str]) -> float:
    t = BASE
    for i in items:
        t *= FACTOR[label(i)]
    return t


def _gate_oom(result: dict, *, legacy: bool = False) -> dict:
    """``result`` with the gate out of memory: as the worker reports it now, or (``legacy``)
    as kernel-agent before #137 did, a quality failure."""
    verdict = {
        **result,
        "passed": False,
        "reason": f"perceptual: {GATE_RECORD['reason']}",
        "metrics": {**result["metrics"], "perceptual": dict(GATE_RECORD)},
    }
    return verdict if legacy else worker._out_of_memory(verdict)


def gate_worker(calls: list[tuple[str, list[str]]], *, legacy=False, persist=False, fits=False):
    """A/Bs with an item in A run out of memory in the gate (both states and the scoring
    models in one process) unless ``fits``; with ``persist`` so does B's own process."""

    def worker_(run, command, *args):
        ns = parse(args)
        calls.append((command, list(args)))
        a, b = [*ns.kernel, *ns.transform], [*ns.b_kernel, *ns.b_transform]
        if command == "e2e":
            r = dryrun._e2e_result(_ms(a))
            r["times_ms"] = [round(_ms(a) * w, 3) for w in [*WIGGLE, 1.0, 1.0]]
            if persist and len(a) > 1:
                r = _gate_oom(r)
                r.pop("times_ms")  # what `e2e` returns for a step that is not ok
            return r
        r = paired(_ms(a), _ms(b), ns.rounds)
        return _gate_oom(r, legacy=legacy) if a and not fits else r

    return worker_


def _setup(run) -> None:
    kernel(run, "attn", 3.0)
    e2e_record(run, 1.6, [write(run, "cfm")])  # the faster one alone: the seed


def test_a_gate_oom_is_measured_again_in_separate_processes(tmp_path):
    orch = make(tmp_path)
    _setup(orch.run)
    calls: list[tuple[str, list[str]]] = []
    orch.worker = gate_worker(calls)
    asyncio.run(orch.integrate())

    data = read_json(orch.run.root / "integration.json")
    assert [label(a["item"]) for a in data["accepted"]] == ["cfm", "attn"]
    step = data["history"][-1]
    assert step["status"] == "ok" and step["passed"] and step["ab"]["accepted"]
    assert step["ab"]["mode"] == "separate" and step["ab"]["expandable_segments"]
    assert step["ab"]["fallback"] == f"out of GPU memory in the perceptual gate: {OOM_LINE}"
    assert [c for c, _ in calls] == ["e2e_ab", "e2e_ab", "e2e_ab", "e2e", "e2e"]
    assert all("--expandable-segments" in args for c, args in calls if c == "e2e")
    text = write_report(orch.run).read_text()
    assert "(retried in separate processes: out of GPU memory in one)" in text
    assert "perceptual gate failed" not in text


def test_an_oom_that_persists_is_not_measurable_and_measured_again(tmp_path):
    orch = make(tmp_path)
    _setup(orch.run)
    calls: list[tuple[str, list[str]]] = []
    orch.worker = gate_worker(calls, persist=True)
    asyncio.run(orch.integrate())

    data = read_json(orch.run.root / "integration.json")
    assert [label(a["item"]) for a in data["accepted"]] == ["cfm"]
    step = data["history"][-1]
    gate = f"out of GPU memory in the perceptual gate: {OOM_LINE}"
    assert step["status"] == abtest.OOM and step["reason"] == gate and not step["ab"]["accepted"]
    assert step["metrics"]["perceptual"]["oom"]
    rows = [r for r in ledger.rows(orch.run) if r["backend"] == "integrate"]
    assert [r["status"] for r in rows].count("oom") == 1 and "incorrect" not in str(rows)
    text = write_report(orch.run).read_text()
    assert f"tried 2 item(s): **not measurable**: {gate} (a re-integration measures" in text
    assert "perceptual gate failed" not in text

    calls.clear()  # a re-integration measures it again, reuses the rest
    orch.worker = gate_worker(calls, fits=True)
    asyncio.run(orch.integrate(reuse=True))
    data = read_json(orch.run.root / "integration.json")
    assert data["reuse"] == {"reused": 2, "measured": 1}
    assert [c for c, _ in calls] == ["e2e_ab"]
    assert [label(a["item"]) for a in data["accepted"]] == ["cfm", "attn"]


def test_a_gate_oom_recorded_as_a_failed_gate_is_not_reused(tmp_path):
    """``integration.json`` from before #137: the step says ``status: ok``, ``perceptual:
    the perceptual gate failed: OutOfMemoryError``. The report shows it as not measurable
    and a re-integration measures it again."""
    orch = make(tmp_path)
    _setup(orch.run)
    calls: list[tuple[str, list[str]]] = []
    orch.worker = gate_worker(calls, legacy=True)
    asyncio.run(orch.integrate())

    data = read_json(orch.run.root / "integration.json")
    step = data["history"][-1]
    assert step["status"] == "ok" and step["reason"].startswith("perceptual: the perceptual gate")
    assert [label(a["item"]) for a in data["accepted"]] == ["cfm"]  # dropped, no verdict
    text = write_report(orch.run).read_text()
    gate = f"out of GPU memory in the perceptual gate: {OOM_LINE}"
    assert f"**not measurable**: {gate}" in text and "perceptual gate failed" not in text

    calls.clear()
    orch.worker = gate_worker(calls, fits=True)
    asyncio.run(orch.integrate(reuse=True))
    data = read_json(orch.run.root / "integration.json")
    assert data["reuse"] == {"reused": 2, "measured": 1} and [c for c, _ in calls] == ["e2e_ab"]
    assert [label(a["item"]) for a in data["accepted"]] == ["cfm", "attn"]


# ------------------------------------------------------------------ GPU

#: A's state: 256 MiB of CUDA memory (two states' FP8 weights and graph pools, in small).
A_CUDA = """import torch


def apply(workload):
    workload.a_state = torch.empty(256 * 2**20, dtype=torch.uint8, device="cuda")
"""
#: B's scorers: what they find allocated, then far more than the GPU has.
B_CUDA = """import torch


def apply(workload):
    quality = workload.perceptual_quality

    def scores(samples):
        found = torch.cuda.memory_allocated() / 2**20
        if workload.options.get("huge_scorer"):
            torch.empty(2**50, dtype=torch.uint8, device="cuda")  # 1 PiB
        return [{**s, "cuda_mib": round(found, 1)} for s in quality(samples)]

    workload.perceptual_quality = scores
"""


@pytest.mark.gpu
def test_a_real_cuda_oom_in_the_gate_and_as_memory_freed_before_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(toolchain, "setup", lambda *a, **k: None)
    run = _sealed(tmp_path, "near-lossless")
    keeper = truth.of(run)
    baseline = _call(capsys, run, "analyze", "--no-profile", "--iters", 1)
    keeper.seal_baseline(baseline["median_ms"])
    a = _file(tmp_path, "a_cuda.py", A_CUDA)
    b = _file(tmp_path, "b_cuda.py", B_CUDA)

    torch.cuda.empty_cache()
    held = torch.cuda.memory_allocated() / 2**20
    r = _call(
        capsys,
        run,
        "e2e_ab",
        "--rounds",
        2,
        "--transform",
        a,
        "--b-transform",
        b,
        *keeper.worker_args(),
    )
    assert r["status"] == "ok" and r["passed"], r
    assert r["ab"]["released_gb"] >= 0.24  # at least A's 256 MiB (allocator blocks add to it)
    found = [s["cuda_mib"] for s in r["metrics"]["perceptual"]["per_sample"]]
    assert max(found) < held + 64, (held, found)  # the scorers do not see A's memory

    huge = _transform(tmp_path, "huge", "    workload.options['huge_scorer'] = True\n")
    r = _call(
        capsys,
        run,
        "e2e_ab",
        "--rounds",
        2,
        "--transform",
        a,
        "--b-transform",
        b,
        "--b-transform",
        huge,
        *keeper.worker_args(),
    )
    assert r["status"] == abtest.OOM and r["status"] in abtest.FALLBACK, r
    assert r["reason"].startswith("out of GPU memory in the perceptual gate: ")
    assert "CUDA out of memory" in r["reason"] and r["metrics"]["perceptual"]["oom"]
    gc.collect()
    torch.cuda.empty_cache()
