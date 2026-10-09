"""Evaluate one kernel candidate against a captured module.

Candidate contract (``candidates/<name>.py``)::

    def build(reference: torch.nn.Module) -> torch.nn.Module:
        '''Return a drop-in replacement for ``reference``: same forward
        signature, same outputs (within tolerance), same in-place side effects
        (e.g. KV-cache updates).  Reuse ``reference``'s parameters/buffers.
        Return ``reference`` itself for instances the kernel does not support.'''

``build`` may take keyword arguments with defaults (block sizes, ``num_warps``, ...):
the evaluator calls ``build(reference)``; :mod:`kernels.sweep` tries configs of them.
A multi-file CUDA / C++ project (a directory with ``kernel_project.toml``) is a candidate
too, as a directory or as its bundle (:mod:`kernel_agent.native.project`): its entry
module's ``build`` is the contract.

Each captured case is replayed through the entrypoint it was recorded from:
``candidate(*args, **kwargs)`` for ``forward`` cases and
``candidate.<method>(*args, **kwargs)`` otherwise (e.g. ``forward_step`` of a
custom decode loop), for correctness and timing alike.  A candidate that lacks
a captured method is a ``build_error``.  The cases of a stateful module (a KV-cache
attribute: ``state`` in the capture, :mod:`kernel_agent.profiling.state`) run from the
module state their call saw: written into the candidate (and into the reference copy it was
built from) before every call, outside timed regions, and the state after the call is
checked like in-place argument updates (``state.*`` checks).

Stages and their failure statuses:

1. ``build_error``: import and ``build()``.
2. ``incorrect``: every captured case against the captured outputs and in-place
   side effects (:mod:`kernels.compare`), plus aliasing parity with a live
   reference call (:func:`kernels.verify.alias_errors`).
3. ``incorrect_timed_output``: timing (CUDA only, :func:`kernels.bench.compare_timing`)
   rotates between input copies and checks the output of one random timed call. Every
   round runs at full GPU clocks (:func:`kernels.bench.time_call`; ``clock`` per case:
   the lowest DRAM bandwidth probe around its rounds, about 1 at full clocks), and
   before the profiled activity pass below (its CUPTI subscription stays: every later
   launch of the process is 1.5-3 us slower). With a bar to beat (``early_best``: the
   ``evaluate_candidate`` tool passes the target's best kept speedup) every case first gets
   2 of its 3 rounds, and a candidate that cannot be a new best whatever the third rounds
   measure is timed no further (an early discard, ``early`` in the result:
   :mod:`kernels.early`, issue #190); every check before and after timing still runs.
   Timing runs in the target's context (#226; ``context``, ``l2``, ``context_reason`` in
   the result): eager calls, or calls captured in a CUDA graph (eager when a call of
   either cannot be captured), with a warm or a cold L2. A winner (or ``profile``) is
   timed in the other context too: ``timing`` per case and ``speedup_by_context`` hold
   both, or ``graph: unavailable (<why>)`` for a candidate that would break a graphed
   stage.
4. ``incorrect_perturbed``: re-verification after timing (on CPU right after
   stage 2) at fresh addresses, with redrawn inputs and with the captured inputs
   scaled by 3, 0.01 and −1 (:mod:`kernels.verify`; ``redraws`` in the result: the
   scaled checks that ran and any skipped for a non-finite reference output).

A check on redrawn inputs (stages 3 and 4) that fails is judged again with the reference's
own rounding spread on those inputs (:func:`kernels.verify.against_rerounded`, issue #250):
a deep bf16 chain that rounds differently from eager is not refused for it.

Native candidates (projects, issue #225) also run every case twice from the same inputs and
module state after stage 2: their outputs and in-place side effects must agree bit for bit
(``incorrect`` at stage ``determinism``; ``determinism`` in the result), unless the entry
declares ``ORDER_DEPENDENT_ATOMICS = "<why>"`` (split-K partials summed with float atomics):
races on global memory and counter-ordering bugs that pass one comparison show up here.
Those whose sources synchronise blocks through global memory (counters, atomics, the
megakernel kit) then run :data:`STRESS_CALLS` calls of the main case back to back, on four
inputs in turn and with GPU-side gaps before some, each bit for bit equal to an isolated call
on the same input (``determinism.stress``; issue #248): state one launch leaves for the next
(self-resetting counters, reused pages and buffers) must not be reused early. A
megakernel stopped by its watchdog (:mod:`kernel_agent.native.megakernel.runtime`) turns the
failure it caused into status ``hang``, with the instruction, counter, value and target in
``hang``.

Peak memory (CUDA, not a failure): per timed case the peak GPU memory of one call of
the reference and of the candidate (:func:`kernels.bench.peak_memory`; ``ref_peak_mib``,
``new_peak_mib``, ``peak_delta_mib``) and ``peak_memory`` for the case with the largest
increase, with a ``warning`` above :data:`PEAK_MEMORY_WARN_SHARE` and
:data:`PEAK_MEMORY_WARN_MIB` (KernelBench-Verified: 28 % of correct kernels raised peak
memory; an end-to-end gate can run out of memory, #137).

Anti-gaming guards (:mod:`kernels.integrity`), status ``integrity_violation``
unless noted:

* a snapshot of the timer, comparator, reference, torch functions and backend
  flags, taken before the candidate is imported, must still hold after
  ``build()``, after correctness, after timing and at the end; threads running
  candidate code are not allowed;
* per case, the device-synchronised wall time of a call must not exceed its
  CUDA-event time by more than the reference's does (work hidden on other
  streams or threads);
* one profiled pass over the dominant case: no GPU work launched from other
  threads or left running on other streams; ``fallback`` when the reference's
  entrypoint code runs there with less than half of the GPU time in kernels the
  reference does not launch, or when none of it is (``custom_kernel_share`` 0) and
  it launches every kernel of the reference at least as often (re-running its ops;
  fewer launches is a restructuring, e.g. merged projections);
* :func:`run_evaluation` (outside the candidate's process) compares the
  candidate's saved outputs with the capture and the reference timing with one
  measured in a candidate-free process (10 % margin); results arrive on a line
  tagged with a nonce the candidate cannot read from its environment.

A failed stage is named in ``stage``, with details in ``failed_check``.
:func:`run_evaluation` stamps every result with ``evaluator_version``
(:func:`evaluator_version`), so a record measured by an older evaluator is
recognisable (:func:`stale`).

Run as a subprocess (so compiler crashes and illegal memory accesses cannot
take down the orchestrator)::

    python -m kernel_agent.kernels.evaluate CAPTURE CANDIDATE [--profile] [--json OUT]
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import dataclasses
import functools
import hashlib
import importlib.util
import json
import os
import random
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Generator
from pathlib import Path
from typing import Any

from kernel_agent import gpuqueue
from kernel_agent.gpulock import child_env, gpu_lock, hold_token, holding
from kernel_agent.kernels.roofline import annotate, ensure_peaks
from kernel_agent.truth import TamperError

COMPILE_MARKER = "@@KA_COMPILE_S@@"  # on stderr, so a timed-out run still reports it
RESULT_MARKER = "@@KA_RESULT@@"
# Bound at import (before any candidate): the result line does not depend on module
# attributes a candidate could patch after the last integrity checkpoint.
_dumps = json.dumps
_stdout = sys.stdout
#: Candidate GPU work the timed stream does not see: violation above this many ms
#: and this share of the call's event time (see :func:`kernels.bench.wall_check`).
HIDDEN_WORK_MS = 0.1
HIDDEN_WORK_SHARE = 0.5
#: Candidate calls in the profiled activity pass over the main case.
ACTIVITY_CALLS = 3
#: ``peak_memory`` warns when a candidate's per-call peak exceeds the reference's by more
#: than this share of it and by more than this many MiB (small modules: noise).
PEAK_MEMORY_WARN_SHARE = 0.25
PEAK_MEMORY_WARN_MIB = 16.0
#: Measurement semantics of this evaluator: bump it when a change makes earlier results
#: incomparable (what is timed and how, how cases are weighted, what counts as correct).
#: Records without ``evaluator_version`` predate it and count as schema 0.
#: 2: every timing round at full GPU clocks (#81; earlier records may be measured in an
#: idle performance state, bandwidth-bound kernels up to 27x too slow).
#: 3: timed in the target's context (#226: ``context`` eager or CUDA graph, ``l2`` warm or
#: cold; earlier records are eager with a warm L2, and an eager cold L2 no longer hides the
#: call's host time behind the flush).
#: Not a bump: #250 (a failed redrawn-input check judged again with the reference's own
#: rounding spread) only accepts more, so every earlier verdict of correct and its timing
#: stand; a bump would re-evaluate and re-check every kept kernel and re-run every A/B.
EVALUATOR_SCHEMA = 3
#: ``context_reason`` of an evaluation whose caller chose no context (eager, warm L2 unless
#: ``l2_flush``); the tools choose one per target (:mod:`kernel_agent.kernels.context`).
DEFAULT_CONTEXT_REASON = "no timing context chosen for the target: eager calls"
#: Timing rounds per case in the other context (a winner's, or with ``profile``; #226).
OTHER_ROUNDS = 2


_run = subprocess.run  # bound at import: tests replace subprocess.run for the evaluator
#: The entry-module attribute of a native candidate whose results depend on the order of its
#: atomics (the determinism check is skipped; its value says why).
ORDER_DEPENDENT = "ORDER_DEPENDENT_ATOMICS"
#: Back-to-back calls of the stress check of native candidates that synchronise blocks through
#: global memory (``$KERNEL_AGENT_STRESS_CALLS`` overrides it; 0: off).
STRESS_CALLS = 256
STRESS_ENV = "KERNEL_AGENT_STRESS_CALLS"
#: What makes a native candidate's sources synchronise blocks through global memory.
_COUNTER_SOURCE = re.compile(
    r"ka_mk|ld\.acquire|red\.release|atom\.|atomicAdd|atomicCAS|atomicExch|cuda::atomic"
)
#: Failures a megakernel's watchdog can be behind: its kernel stopped early (wrong outputs) or
#: its runtime refused the next launch.
_HANG_CAN_CAUSE = ("runtime_error", "incorrect", "incorrect_timed_output", "incorrect_perturbed")
_MK_RUNTIME = "kernel_agent.native.megakernel.runtime"


@functools.cache
def _git_sha() -> str | None:
    """The commit of the kernel-agent checkout this module runs from (None: an installed
    package, not a git checkout of kernel-agent, or no git)."""
    here = Path(__file__).resolve()
    try:
        proc = _run(
            ["git", "rev-parse", "--show-toplevel", "HEAD"],
            cwd=here.parent,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = proc.stdout.splitlines()
    if proc.returncode != 0 or len(lines) != 2:
        return None
    top, sha = Path(lines[0]), lines[1]
    ours = (top / "src" / "kernel_agent" / "kernels" / here.name).resolve() == here
    return sha if ours else None  # not the checkout of some other repository around it


def evaluator_version() -> dict[str, Any]:
    """``{"schema": EVALUATOR_SCHEMA, "git": <commit>}`` (``git`` when available)."""
    sha = _git_sha()
    return {"schema": EVALUATOR_SCHEMA, **({"git": sha} if sha else {})}


def record_schema(record: dict[str, Any]) -> int:
    """The evaluator schema a record was measured with (0: before it was recorded)."""
    version = record.get("evaluator_version")
    try:
        return int(version.get("schema") or 0) if isinstance(version, dict) else 0
    except (TypeError, ValueError):
        return 0


def stale(record: dict[str, Any]) -> str | None:
    """Why ``record`` was measured by an older evaluator than this one (None: it was not);
    only the schema counts, not the commit."""
    schema = record_schema(record)
    if schema >= EVALUATOR_SCHEMA:
        return None
    if schema == 0:
        return f"measured before evaluator_version was recorded (now schema {EVALUATOR_SCHEMA})"
    return f"measured by evaluator schema {schema} (now {EVALUATOR_SCHEMA})"


def load_candidate_module(path: Path) -> Any:
    """Import a candidate: a ``.py`` file (a project bundle is one) or a project directory
    (:mod:`kernel_agent.native.project`: its entry module, run from the build cache)."""
    path = path.resolve()
    if path.is_dir():
        from kernel_agent.native import project

        module = project.import_project(path)
        if not callable(getattr(module, "build", None)):
            raise AttributeError(f"{path.name}: the project's entry must define build(reference)")
        return module
    digest = hashlib.sha1(path.read_bytes()).hexdigest()[:10]
    name = f"ka_candidate_{path.stem}_{digest}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    if not callable(getattr(module, "build", None)):
        raise AttributeError(f"{path.name} must define build(reference) -> nn.Module")
    return module


def _short_tb(limit: int = 4000) -> str:
    text = traceback.format_exc()
    return text if len(text) <= limit else "...\n" + text[-limit:]


def capture_precision(capture: dict[str, Any]) -> str | None:
    """The reduced precision of a capture in a near-lossless tier (``fp8_weights``: the
    speed of light counts its weights at 8 bits, :mod:`kernels.roofline`), else None."""
    from kernel_agent.kernels.compare import EXACT_TIER, tier_of

    precision = capture.get("precision")
    return str(precision) if precision and tier_of(capture) != EXACT_TIER else None


def _kernel_table(fn: Any, args: Any, kwargs: Any, top: int = 15) -> list[dict[str, Any]]:
    import torch
    from torch.profiler import ProfilerActivity, profile

    from kernel_agent.kernels.bench import has_mutable_state
    from kernel_agent.profiling.state import split

    a, k = (
        (copy.deepcopy(args), copy.deepcopy(kwargs))
        if has_mutable_state(args, kwargs)
        else (
            args,
            kwargs,
        )
    )
    fn, restore = split(fn)  # the case's module state, restored outside the profile
    if restore is not None:
        restore()
    with torch.inference_mode(), profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(*a, **k)
        torch.cuda.synchronize()
    rows = []
    for evt in prof.key_averages():
        us = getattr(evt, "self_device_time_total", None) or getattr(evt, "self_cuda_time_total", 0)
        if us:
            rows.append({"kernel": evt.key[:140], "calls": int(evt.count), "us": round(us, 2)})
    rows.sort(key=lambda r: -r["us"])
    return rows[:top]


def _numel(value: Any) -> int:
    """Tensor elements in a (nested) structure of arguments."""
    import torch

    if isinstance(value, torch.Tensor):
        return int(value.numel())
    if isinstance(value, dict):
        return sum(_numel(v) for v in value.values())
    if isinstance(value, list | tuple):
        return sum(_numel(v) for v in value)
    return 0


def quick_cases(cases: list[dict[str, Any]]) -> list[int]:
    """Indices of the smallest and the largest case by input elements (``--quick``)."""
    sizes = [_numel((c.get("args"), c.get("kwargs"))) for c in cases]
    order = sorted(range(len(cases)), key=lambda i: (sizes[i], i))
    return sorted({order[0], order[-1]}) if order else []


def _mib(n: int) -> float:
    return round(n / 2**20, 3)


def peak_memory_summary(cases: list[dict[str, Any]]) -> dict[str, Any] | None:
    """``peak_memory`` of a result from its cases' ``ref_peak_mib`` / ``new_peak_mib``: the
    case with the largest increase (``case``, ``ref_mib``, ``new_mib``, ``delta_mib``,
    ``ratio``) and a ``warning`` when the increase is above both :data:`PEAK_MEMORY_WARN_SHARE`
    of the reference's peak and :data:`PEAK_MEMORY_WARN_MIB`; None without measurements."""
    rows = [
        (i, float(c["ref_peak_mib"]), float(c["new_peak_mib"]))
        for i, c in enumerate(cases)
        if c.get("ref_peak_mib") is not None and c.get("new_peak_mib") is not None
    ]
    if not rows:
        return None
    i, ref, new = max(rows, key=lambda r: (r[2] - r[1], r[2]))
    delta = new - ref
    out: dict[str, Any] = {
        "case": i,
        "ref_mib": round(ref, 3),
        "new_mib": round(new, 3),
        "delta_mib": round(delta, 3),
        "ratio": round(new / ref, 3) if ref > 0 else None,
    }
    if delta > PEAK_MEMORY_WARN_MIB and delta > PEAK_MEMORY_WARN_SHARE * ref:
        sig = cases[i].get("signature") or f"case {i}"
        out["warning"] = (
            f"a call of {sig} peaks at {new:.1f} MiB of GPU memory, {delta:.1f} MiB more than "
            f"the reference's {ref:.1f} MiB (warning above +{PEAK_MEMORY_WARN_SHARE:.0%} and "
            f"+{PEAK_MEMORY_WARN_MIB:g} MiB): intermediates, fp32 copies or workspaces the "
            "reference does not allocate; in the whole model they can run it out of memory"
        )
    return out


def _first_error(failures: list[dict[str, Any]]) -> str:
    first = failures[0] if failures else {}
    detail = first.get("error") or (
        f"mismatch {first.get('mismatch_frac')}, max abs err {first.get('max_abs_err')}"
    )
    return f"{first.get('name', '?')}: {detail}"


def _reverify(
    result: dict[str, Any],
    reference: Any,
    holders: tuple[Any, ...],
    cases: list[dict[str, Any]],
    pristine: list[tuple[Any, Any]],
    seed: int,
    device: str,
    replay: Any,
) -> bool:
    """Stage 4 (:func:`kernels.verify.reverify_case` on every case: the candidate is
    ``holders[0]``, every call from the case's module state, ``replay``); False and
    ``incorrect_perturbed`` in ``result`` if a check fails."""
    import torch

    from kernel_agent.kernels.verify import reverify_case
    from kernel_agent.workloads.base import synchronize

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    redraws: dict[str, Any] = {"ran": {}, "skipped": []}
    result["redraws"] = redraws  # the scaled checks (kernels.verify.SCALED) per case
    for i, (case, inputs) in enumerate(zip(cases, pristine, strict=True)):
        record: dict[str, Any] = {}
        try:
            failed = reverify_case(
                replay.call(case, reference),
                replay.call(case, *holders),
                case,
                inputs,
                gen,
                synchronize,
                record,
            )
        except Exception:
            result.update(status="runtime_error", correct=False, error=_short_tb(), failed_case=i)
            return False
        for name, n in (record.get("ran") or {}).items():
            redraws["ran"][name] = redraws["ran"].get(name, 0) + n
        redraws["skipped"] += [{"case": i, **skip} for skip in record.get("skipped") or []]
        if failed:
            check = failed[0]
            result.update(
                status="incorrect_perturbed",
                correct=False,
                stage="perturbed",
                failed_check={"case": i, **check},
                error=f"case {i} ({case['signature']}) passed as captured but failed with "
                f"{check['what']} ({_first_error(check['failures'])}): the candidate must be "
                "correct for any input of these shapes and dtypes, recomputed on every "
                "call (no output caching by address, shape or values; no reading of "
                "unused cache slots; no shortcuts for the captured value range or sign: "
                "scales such as FP8 activation scales follow the input)",
            )
            return False
    result["checks"] = ["captured", "aliasing", "timed_output", "perturbed"]
    if redraws["ran"]:
        result["checks"].append("scaled")
    if "quick" in result or not device.startswith("cuda"):  # untimed
        result["checks"].remove("timed_output")
    return True


def _violation(result: dict[str, Any], stage: str, error: str, **detail: Any) -> None:
    result.update(
        status="integrity_violation",
        correct=False,
        stage=stage,
        error=error,
        failed_check={"check": stage, **detail},
    )


def _hidden_work(wall: dict[str, float]) -> bool:
    return wall["hidden_ms"] > max(HIDDEN_WORK_MS, HIDDEN_WORK_SHARE * wall["new_event_ms"])


def _ms(rounds: list[dict[str, Any]]) -> list[float]:
    """The medians of timing rounds (``bench.time_call`` results)."""
    return [float(r["median_ms"]) for r in rounds]


def _in_context(ref_t: dict[str, Any], new_t: dict[str, Any]) -> dict[str, float]:
    """A case's ``timing`` in one context: the median rounds of reference and candidate."""
    ref_ms, new_ms = float(ref_t["median_ms"]), float(new_t["median_ms"])
    return {
        "ref_ms": round(ref_ms, 5),
        "new_ms": round(new_ms, 5),
        "speedup": round(ref_ms / max(new_ms, 1e-9), 3),
    }


def context_fields(
    reports: list[dict[str, Any]], context: str, l2_flush: bool, reason: str
) -> dict[str, Any]:
    """A timed result's context fields (#226): ``context`` (the one ``speedup`` and
    ``pct_of_sol`` are measured in), ``l2`` (``warm`` or ``cold``), ``context_reason`` and
    ``speedup_by_context``: per context its speedup weighted by calls per run, when every
    timed case was timed in it (``timing`` of the cases), or ``unavailable (<why>)``."""
    from kernel_agent.kernels.bench import CONTEXTS

    timed = [r for r in reports if r.get("timing")]
    by_context: dict[str, Any] = {}
    for name in CONTEXTS:
        found = [r["timing"].get(name) for r in timed]
        why = next((v for v in found if isinstance(v, str)), None)
        if why is not None:
            by_context[name] = why
        elif found and all(isinstance(v, dict) for v in found):
            n = [float(r.get("calls_per_run") or 0) for r in timed]
            ref = sum(k * float(v["ref_ms"]) for k, v in zip(n, found, strict=True))
            new = sum(k * float(v["new_ms"]) for k, v in zip(n, found, strict=True))
            by_context[name] = round(ref / max(new, 1e-9), 3)
    return {
        "context": context,
        "l2": "cold" if l2_flush else "warm",
        "context_reason": reason,
        "speedup_by_context": by_context,
    }


def _graph_unavailable(timed: list[tuple[int, tuple[Any, Any], dict[str, Any]]]) -> str | None:
    """Why these timed cases (index, reference and candidate entrypoints, case) cannot all
    be timed in a CUDA graph: the first call that cannot be captured
    (:func:`kernels.bench.graph_probe`); None when every one can."""
    from kernel_agent.kernels.bench import graph_probe

    for i, fns, case in timed:
        for who, fn in zip(("reference", "candidate"), fns, strict=True):
            if (why := graph_probe(fn, case["args"], case["kwargs"])) is not None:
                return f"{who}, case {i}: {why}"
    return None


@dataclasses.dataclass
class _Rounds:
    """A timed case of :func:`evaluate`: its index, report and case, the reference's and the
    candidate's entrypoints and their timing rounds so far."""

    index: int
    report: dict[str, Any]
    case: dict[str, Any]
    fns: tuple[Any, Any]
    ref: list[dict[str, Any]]
    new: list[dict[str, Any]]


def _other_context(
    timed: list[_Rounds], other: str, l2_flush: bool, graph_state: str | None
) -> str | None:
    """Time the timed cases in the ``other`` context too (:data:`OTHER_ROUNDS` rounds each,
    into their ``timing``); returns why graph timing is unavailable: ``graph_state`` (known
    already), a call that cannot be captured, or a graph replay whose output differs from
    the reference's (correct eagerly, wrong inside a CUDA-graphed stage); None: it is not."""
    from kernel_agent.kernels.bench import GRAPH, GraphUnavailable, median_round, timing_rounds

    if other == GRAPH:
        if graph_state is None:
            graph_state = _graph_unavailable([(t.index, t.fns, t.case) for t in timed])
        if graph_state is not None:
            return graph_state
    measured: dict[int, dict[str, float]] = {}
    for t in timed:
        try:
            ref_r, new_r, checked = timing_rounds(
                *t.fns,
                t.case["args"],
                t.case["kwargs"],
                rounds=OTHER_ROUNDS,
                l2_flush=l2_flush,
                verify=other == GRAPH,
                context=other,
            )
        except GraphUnavailable as exc:
            return f"case {t.index}: {exc}"
        if checked and checked.get("failures"):
            return (
                f"case {t.index}: the output of call #{checked['iteration']} of a graph replay "
                f"differs from the reference ({_first_error(checked['failures'])})"
            )
        measured[t.index] = _in_context(median_round(ref_r), median_round(new_r))
    for t in timed:
        t.report["timing"][other] = measured[t.index]
    return graph_state


@functools.cache
def _code_dirs(candidate_path: Path) -> tuple[Path, ...]:
    """Where the candidate's code lives: its directory, and a project's source directory in
    the build cache (:func:`kernel_agent.native.project.code_dirs`); read before import."""
    from kernel_agent.native.project import code_dirs

    return (candidate_path.parent, *code_dirs(candidate_path))


def _intact(result: dict[str, Any], guard: Any, candidate_path: Path, where: str) -> bool:
    """Integrity checkpoint: False (and ``integrity_violation``) if the candidate changed
    watched state or runs threads."""
    from kernel_agent.kernels.integrity import candidate_threads

    problems = guard.changes()
    problems += [
        f"thread {t} runs candidate code"
        for directory in _code_dirs(candidate_path)
        for t in candidate_threads(directory)
    ]
    if not problems:
        return True
    _violation(
        result,
        "integrity",
        f"{where}: " + "; ".join(problems[:6]) + ". Candidates must not patch torch, the "
        "timer, the comparator or the reference, change global backend flags (TF32, SDPA "
        "backends, ...), or keep threads running.",
        changes=problems[:20],
    )
    return False


def _fallback(
    result: dict[str, Any],
    i: int,
    case: dict[str, Any],
    ran: dict[str, int],
    activity: dict[str, Any] | None = None,
) -> bool:
    """Decide ``fallback`` for the main case ``i`` (:func:`kernels.integrity.fallback_reason`)
    from the reference entrypoints that ran during the candidate's calls (``ran``) and,
    on CUDA, the profiled ``activity`` (custom kernel share, kernel launches)."""
    from kernel_agent.kernels.integrity import fallback_reason

    activity = activity or {}
    share = activity.get("custom_kernel_share")
    why = fallback_reason(
        ran, share, activity.get("reference_kernels"), activity.get("candidate_kernels")
    )
    if why is None:
        return False
    result.update(
        status="fallback",
        correct=False,
        stage="fallback",
        failed_check={"case": i, "check": "fallback", "reference_calls": ran, "share": share},
        error=f"case {i} ({case['signature']}, the dominant case: {case['count']} calls per "
        f"run) {why}: the dominant case must run your own kernels or a genuine "
        "restructuring of the reference's (fewer launches). Do not call or inherit the "
        "reference's entrypoint for it and do not re-launch the reference's ops; fall back "
        "only for shapes you do not support.",
    )
    return True


def evaluate(
    capture_path: Path,
    candidate_path: Path,
    *,
    profile: bool = False,
    l2_flush: bool = False,
    compile_baseline: bool = False,
    device: str | None = None,
    capture_sha256: str | None = None,
    compile_check: bool = False,
    quick: bool = False,
    save_outputs: Path | None = None,
    config: dict[str, Any] | None = None,
    session: dict[str, Any] | None = None,
    early_best: float | None = None,
    context: str = "eager",
    context_reason: str | None = None,
) -> dict[str, Any]:
    """Build, check and time one candidate.  ``device`` defaults to CUDA when
    available; on CPU only correctness is checked (timing needs CUDA events).
    ``capture_sha256``: refuse (status ``tampered``) a capture without this digest.
    ``compile_check`` adds ``result["compile_check"]`` (``compile_check.py``).
    ``quick``: correctness on the smallest and the largest case only, never timed.
    ``save_outputs``: write the candidate's outputs of the correctness stage there
    (:func:`kernels.integrity.flat_outputs`), for :func:`run_evaluation` to check.
    ``config``: keyword arguments of ``build(reference, **config)``. ``session``: what
    the configs of one sweep share (:mod:`kernels.sweep`): the loaded ``capture``, the
    candidate ``module`` and a deepcopy ``memo`` (weights shared, not copied per
    config); a passing quick check leaves its built ``candidate`` there.
    ``early_best``: the speedup a new best must beat (the target's best kept one): a
    correct candidate that cannot reach it stops timing after the first rounds
    (``early`` in the result, :func:`kernels.early.discard`); None: every round.
    ``context``: the target's timing context (``eager`` or ``graph``, :func:`kernels.bench.
    time_call`; with ``l2_flush`` a cold L2), ``context_reason`` why it is the target's
    (:mod:`kernels.context`); graph timing falls back to eager when a call cannot be
    captured, and the result says why (:func:`context_fields`).
    Global state the candidate changed is restored on return."""
    guards: list[Any] = []
    try:
        return _finish(
            _stages(
                capture_path,
                candidate_path,
                guards,
                profile=profile,
                l2_flush=l2_flush,
                compile_baseline=compile_baseline,
                device=device,
                capture_sha256=capture_sha256,
                compile_check=compile_check,
                quick=quick,
                save_outputs=save_outputs,
                config=config or {},
                session=session if session is not None else {},
                early_best=early_best,
                context=context,
                context_reason=context_reason,
            )
        )
    finally:
        _restore(guards)


def _restore(guards: list[Any]) -> None:
    """Global state candidates changed back as it was (their integrity snapshots), and the
    comparator's tolerance tier (:func:`_stages` set it from the capture)."""
    for guard in guards:
        guard.restore()
    from kernel_agent.kernels import compare

    compare.TIER = compare.EXACT_TIER


def _finish(stages: Generator[None, None, dict[str, Any]]) -> dict[str, Any]:
    """The result of a candidate's :func:`_stages`, run to the end."""
    try:
        while True:
            next(stages)
    except StopIteration as done:
        result: dict[str, Any] = done.value
        return _hang(result)


def _hang(result: dict[str, Any]) -> dict[str, Any]:
    """A failed ``result`` whose cause was a megakernel's watchdog: status ``hang`` with where
    it stopped (the instruction id, counter, value and target) in ``hang``."""
    runtime = sys.modules.get(_MK_RUNTIME)  # imported only by candidates that use the kit
    if result.get("status") not in _HANG_CAN_CAUSE or runtime is None:
        return result
    if found := runtime.hangs():
        before = str(result.get("error") or "")
        result.update(status="hang", correct=False, hang=found[0])
        result["error"] = runtime.describe(found[0]) + (f"\n{before[-1500:]}" if before else "")
    return result


def _snapshot(value: Any) -> Any:
    """The tensors of a (nested) output as detached copies (a later call may reuse them)."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _snapshot(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_snapshot(v) for v in value]
    return value


def _bitwise_diff(a: Any, b: Any, name: str) -> str | None:
    """The first place two snapshots differ in a bit (shape, dtype or bytes), or None."""
    import torch

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            return f"{name}: {tuple(a.shape)} {a.dtype} vs {tuple(b.shape)} {b.dtype}"
        bits_a, bits_b = a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)
        if not torch.equal(bits_a, bits_b):
            per_element = (bits_a != bits_b).reshape(a.numel(), -1).any(dim=1)
            return f"{name}: {int(per_element.sum())} of {a.numel()} elements differ"
        return None
    if isinstance(a, dict) and isinstance(b, dict):
        for key in a:
            if (found := _bitwise_diff(a[key], b.get(key), f"{name}.{key}")) is not None:
                return found
        return None
    if isinstance(a, list) and isinstance(b, list):
        for k, (x, y) in enumerate(zip(a, b, strict=False)):
            if (found := _bitwise_diff(x, y, f"{name}[{k}]")) is not None:
                return found
        return None
    return None


def _determinism(
    cases: list[dict[str, Any]], replay: Any, holders: tuple[Any, ...]
) -> dict[str, Any] | None:
    """Every case twice from its captured inputs and module state: the first difference
    between the two calls' outputs and in-place updated arguments (case, where), or None."""
    import torch

    from kernel_agent.workloads.base import synchronize

    for i, case in enumerate(cases):
        runs = []
        for _ in range(2):
            args, kwargs = copy.deepcopy(case["args"]), copy.deepcopy(case["kwargs"])
            with torch.inference_mode():
                out = replay.call(case, *holders)(*args, **kwargs)
            synchronize()
            runs.append(_snapshot({"output": out, "args": args, "kwargs": kwargs}))
        if (where := _bitwise_diff(runs[0], runs[1], "call")) is not None:
            return {"case": i, "where": where}
    return None


def stress_calls() -> int:
    """Calls of the stress check (:data:`STRESS_CALLS`, ``$KERNEL_AGENT_STRESS_CALLS``)."""
    try:
        return max(0, int(os.environ.get(STRESS_ENV, STRESS_CALLS)))
    except ValueError:
        return STRESS_CALLS


def _uses_counters(module: Any) -> bool:
    """Whether a native candidate's project files synchronise blocks through global memory
    (:data:`_COUNTER_SOURCE`): those of its materialised source directory."""
    from kernel_agent.native import project

    digest = (getattr(module, "__ka_project__", None) or {}).get("digest")
    root = project.src_dir(str(digest)) if digest else None
    if root is None or not root.is_dir():
        return False
    sources = (".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc", ".py")
    return any(
        path.is_file()
        and path.suffix in sources
        and _COUNTER_SOURCE.search(path.read_text(errors="replace")) is not None
        for path in sorted(root.rglob("*"))
    )


def _tensors(value: Any) -> list[Any]:
    """The tensors of a (nested) structure, in order."""
    import torch

    if isinstance(value, torch.Tensor):
        return [value.detach()]
    if isinstance(value, dict):
        return [t for v in value.values() for t in _tensors(v)]
    if isinstance(value, list | tuple):
        return [t for v in value for t in _tensors(v)]
    return []


def _variant(args: Any, kwargs: Any, k: int) -> tuple[Any, Any]:
    """A copy of a case's inputs whose floating tensors are the captured ones (k = 0), negated,
    rolled by one along their last dimension or halved: four inputs with exact values."""
    import torch

    a, kw = copy.deepcopy((args, kwargs))
    if k:
        seen: set[int] = set()  # a tensor passed twice is changed once
        for t in _tensors((a, kw)):
            if t.is_floating_point() and t.numel() and t.data_ptr() not in seen:
                seen.add(t.data_ptr())
                with torch.no_grad():
                    if k == 1:
                        t.neg_()
                    elif k == 2:
                        t.copy_(t.roll(1, -1))
                    else:
                        t.mul_(0.5)
    return a, kw


def _differs(got: list[Any], want: list[Any]) -> Any:
    """Whether two tensor lists differ in a bit: a device boolean (no synchronisation), or
    True when their shapes or dtypes differ."""
    import torch

    if len(got) != len(want):
        return True
    flag: Any = False
    for a, b in zip(got, want, strict=True):
        if a.shape != b.shape or a.dtype != b.dtype:
            return True
        d = (a.reshape(-1).view(torch.uint8) != b.reshape(-1).view(torch.uint8)).any()
        flag = d if flag is False else flag | d
    return flag


def _stress(
    case: dict[str, Any], replay: Any, holders: tuple[Any, ...], calls: int
) -> dict[str, Any]:
    """``calls`` calls of ``case`` back to back (no synchronisation between them; on CUDA a
    GPU-side gap of up to ~0.1 ms before a quarter of them), on four inputs in turn
    (:func:`_variant`), each from the case's module state: how many differ, bit for bit, from
    an isolated call on the same input (outputs and in-place updated arguments), the first."""
    import torch

    from kernel_agent.workloads.base import synchronize

    rng = random.Random(0)
    variants = [_variant(case["args"], case["kwargs"], k) for k in range(4)]
    golden = []
    for a, kw in variants:
        a2, kw2 = copy.deepcopy((a, kw))
        synchronize()
        with torch.inference_mode():
            out = replay.call(case, *holders)(*a2, **kw2)
        synchronize()
        golden.append(_tensors(_snapshot({"output": out, "args": a2, "kwargs": kw2})))
    cuda = any(t.is_cuda for g in golden for t in g)
    flags = []
    for i in range(calls):
        k = i % len(variants)
        a2, kw2 = copy.deepcopy(variants[k])
        if cuda and rng.random() < 0.25:
            torch.cuda._sleep(rng.randrange(1_000, 300_000))
        with torch.inference_mode():
            out = replay.call(case, *holders)(*a2, **kw2)
            got = _tensors({"output": out, "args": a2, "kwargs": kw2})
            flags.append(_differs(got, golden[k]))
    synchronize()
    wrong = [i for i, f in enumerate(flags) if bool(f)]
    return {"calls": calls, "wrong": len(wrong), "first": wrong[0] if wrong else None}


def _stages(
    capture_path: Path,
    candidate_path: Path,
    guards: list[Any],
    *,
    profile: bool,
    l2_flush: bool,
    compile_baseline: bool,
    device: str | None,
    capture_sha256: str | None,
    compile_check: bool,
    quick: bool,
    save_outputs: Path | None,
    config: dict[str, Any],
    session: dict[str, Any],
    early_best: float | None = None,
    context: str = "eager",
    context_reason: str | None = None,
) -> Generator[None, None, dict[str, Any]]:
    """The stages of :func:`evaluate`; the result is the generator's return value. It
    pauses once (a ``yield``) after timing, before the profiled activity pass, so a batch
    (:func:`_batch_main`) times every candidate before any profiler's CUPTI subscription
    slows the launches of its process."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda"):
        from kernel_agent import toolchain

        toolchain.setup()

    from kernel_agent import concurrency
    from kernel_agent.kernels import compare as comparator
    from kernel_agent.kernels import early, integrity, weights
    from kernel_agent.kernels.bench import (
        EAGER,
        GRAPH,
        ROUNDS,
        GraphUnavailable,
        median_round,
        peak_memory,
        time_call,
        timing_rounds,
        wall_check,
    )
    from kernel_agent.kernels.compare import compare_side_effects, compare_structures
    from kernel_agent.kernels.verify import alias_errors
    from kernel_agent.profiling.capture import load_capture
    from kernel_agent.profiling.methods import entrypoint
    from kernel_agent.profiling.state import Replay
    from kernel_agent.workloads.base import synchronize

    result: dict[str, Any] = {
        "candidate": str(candidate_path),
        "status": "error",
        "correct": False,
    }
    from kernel_agent import emulate

    if device.startswith("cuda") and (emulated := emulate.record()) is not None:
        result["emulated"] = emulated  # an older GPU's code paths; timings not its (#252)
    t0 = time.perf_counter()
    try:
        capture = session.get("capture") or load_capture(
            capture_path, device=device, sha256=capture_sha256
        )
        session["capture"] = capture  # the next candidates of a batch (evaluate_candidates)
    except TamperError as exc:
        result.update(status="tampered", error=str(exc))
        return result
    if capture.get("inputs_only"):
        result["error"] = (
            f"{capture_path} is an inputs-only capture (no reference outputs); use the "
            "evaluate_candidate tool, or the full capture in the run's .truth/captures/"
        )
        return result
    reference = capture["module"].eval()
    cases = capture["cases"]
    for case in cases:
        case.setdefault("method", "forward")  # captures written before entrypoints existed
    # module state per case (profiling/state.py), from the reference before any call
    replay = session.get("replay") or Replay(capture, reference)
    session["replay"] = replay
    comparator.TIER = comparator.tier_of(capture)  # before the snapshot, which watches it
    if comparator.TIER != comparator.EXACT_TIER:
        result["tolerance_tier"] = comparator.TIER
    guard = integrity.Snapshot(reference, state=replay.keys)  # before the candidate is imported
    guards.append(guard)
    if (mk := sys.modules.get(_MK_RUNTIME)) is not None:
        mk.forget()  # an earlier candidate's megakernels (a batch) are not this one's
    _code_dirs(candidate_path)  # a bundle's code directory, read before it is imported

    # 1. import + build
    t_build = time.perf_counter()
    try:
        module = session.get("module") or load_candidate_module(candidate_path)
        session["module"] = module
        given = copy.deepcopy(reference, dict(session.get("memo") or {}))
        candidate = module.build(given, **config)
        if candidate is None:
            raise TypeError("build() returned None")
        candidate = candidate.eval() if hasattr(candidate, "eval") else candidate
    except Exception:
        result.update(status="build_error", error=_short_tb())
        return result
    if candidate is given or candidate is reference:
        result.update(
            status="build_error",
            error="build() returned the reference module unchanged; nothing to evaluate "
            "(that fallback is only for instances the kernel does not support)",
        )
        return result
    # where a case's module state is restored (and read back): the candidate, and the copy
    # of the reference it was built from (a wrapper keeps the state there)
    holders = (candidate, given)
    calls: collections.Counter[str] = collections.Counter()
    for case in cases:
        calls[case["method"]] += case["count"]
    missing = [m for m in calls if m != "forward" and not callable(getattr(candidate, m, None))]
    if missing:
        result.update(
            status="build_error",
            error=f"build() returned a {type(candidate).__name__} without "
            + ", ".join(f"`{m}()`" for m in missing)
            + "; the model calls this module through "
            + ", ".join(f"`{m}` ({n} calls per run)" for m, n in calls.items())
            + ". Implement every captured entrypoint with the reference signature and side "
            "effects (subclassing the reference class keeps the ones you do not optimise).",
        )
        return result
    if not _intact(result, guard, candidate_path, "after build()"):
        return result

    if quick:  # evaluate_candidate(mode="quick"): two cases, untimed (not a benchmark)
        result["quick"] = {"cases": quick_cases(cases), "of": len(cases)}
        cases = [cases[i] for i in result["quick"]["cases"]]

    # 2. correctness on every captured case (outputs + in-place side effects + aliasing)
    case_reports: list[dict[str, Any]] = []
    all_ok = True
    outputs: list[dict[str, Any]] = []
    for i, case in enumerate(cases):
        args, kwargs = copy.deepcopy(case["args"]), copy.deepcopy(case["kwargs"])
        ref_args, ref_kwargs = copy.deepcopy(case["args"]), copy.deepcopy(case["kwargs"])
        try:
            with torch.inference_mode():
                out = replay.call(case, *holders)(*args, **kwargs)
            synchronize()
        except Exception:
            result.update(status="runtime_error", error=_short_tb(), failed_case=i)
            return result
        if i == 0:
            # import + build + first call: where load_inline / JIT backends compile
            result["compile_s"] = round(time.perf_counter() - t_build, 1)
            print(f"{COMPILE_MARKER}{result['compile_s']}", file=sys.stderr, flush=True)
        inputs = (case["args"], case["kwargs"])  # a cache the call grows: per part
        checks = compare_structures(case["output"], out, "output", inputs=inputs)
        checks += compare_side_effects(case["args"], case["post_args"], args, "args")
        checks += compare_side_effects(case["kwargs"], case["post_kwargs"], kwargs, "kwargs")
        checks += replay.check(case, *holders)  # what the call changed in the module's state
        with torch.inference_mode():  # after the candidate: the aliasing of a live call
            ref_out = replay.call(case, reference)(*ref_args, **ref_kwargs)
        synchronize()
        checks += alias_errors(ref_out, (ref_args, ref_kwargs), out, (args, kwargs))
        del ref_out, ref_args, ref_kwargs
        if save_outputs is not None:  # on the CPU now: a later call may reuse the buffers
            outputs += integrity.flat_outputs([{"output": out, "args": args, "kwargs": kwargs}])
        bad = [c for c in checks if not c.get("ok")]
        ok = not bad
        all_ok &= ok
        worst = max(
            (c for c in checks if "max_abs_err" in c), key=lambda c: c["max_abs_err"], default={}
        )
        case_reports.append(
            {
                "case": i,
                "method": case["method"],
                "signature": case["signature"],
                "calls_per_run": case["count"],
                "ok": ok,
                "tensors_checked": len(checks),
                "max_abs_err": worst.get("max_abs_err"),
                "min_cosine": min((c.get("cosine", 1.0) for c in checks), default=1.0),
                "failures": bad[:5],
            }
        )
        if comparator.TIER != comparator.EXACT_TIER:  # the numerical error to report
            rel_l2 = [c["rel_l2"] for c in checks if "rel_l2" in c]
            case_reports[-1]["max_rel_l2"] = max(rel_l2, default=0.0)
    result["cases"] = case_reports
    # the scale rule of block-scaled activations, MXFP8 and W4A4 (kernels/scale_guard.py)
    if (guarded := capture_precision(capture)) in ("fp8_mx", "fp4_w4a4"):
        from kernel_agent.kernels import scale_guard

        rule = scale_guard.check(module, cases, guarded)
        result["scale_rule"] = rule
        if not rule["ok"]:
            result.update(
                status="incorrect",
                stage="scale_rule",
                failed_check={"check": "scale_rule", "input": rule.get("input")},
                error=rule["error"],
            )
            return result
    if not all_ok:
        failed = next(i for i, r in enumerate(case_reports) if not r["ok"])
        if any(f.get("name", "").startswith("state.") for f in case_reports[failed]["failures"]):
            result["state_note"] = (
                "`state.*` is the module's state outside the call's arguments (e.g. a KV-cache "
                "attribute), set from the capture before every call: the candidate must read "
                "it and update it in place where and as the reference does"
            )
        result.update(  # the failures are in result["cases"]
            status="incorrect",
            stage="correctness",
            failed_check={"case": failed, "check": "captured"},
        )
        return result
    if getattr(module, "__ka_project__", None):  # native: the same call twice, the same bits
        declared = getattr(module, ORDER_DEPENDENT, None)
        if declared:
            result["determinism"] = {"checked": False, "declared": str(declared)[:300]}
        else:
            try:
                diff = _determinism(cases, replay, holders)
            except Exception:
                result.update(status="runtime_error", error=_short_tb(), stage="determinism")
                return result
            result["determinism"] = {"checked": True, "cases": len(cases)}
            if diff is None and (n := stress_calls()) and _uses_counters(module):
                main = integrity.main_case(cases, case_reports)
                try:
                    stress = _stress(cases[main], replay, holders, n)
                except Exception:
                    result.update(status="runtime_error", error=_short_tb(), stage="determinism")
                    return result
                result["determinism"]["stress"] = {"case": main, **stress}
                if stress["wrong"]:
                    result.update(
                        status="incorrect",
                        stage="determinism",
                        failed_check={"case": main, "check": "stress"},
                        error=(
                            f"case {main}: {stress['wrong']} of {n} back-to-back calls differ "
                            "bit for bit from an isolated call on the same input (the first: "
                            f"call {stress['first']}): state one launch leaves for the next "
                            "(counters, flags, pages, buffers) is reused before every block of "
                            "the previous launch is done with it, or blocks race on global "
                            "memory; see the native-engines skill's megakernel.md"
                        ),
                    )
                    return result
            if diff is not None:
                result["determinism"].update(diff)
                result.update(
                    status="incorrect",
                    stage="determinism",
                    failed_check={"case": diff["case"], "check": "determinism"},
                    error=(
                        f"case {diff['case']}: two calls from the same inputs and state gave "
                        f"different results ({diff['where']}): a race on global memory or an "
                        "order-dependent reduction; fix the race, or declare "
                        f'{ORDER_DEPENDENT} = "<why>" in the entry when float atomics sum '
                        "in a varying order on purpose"
                    ),
                )
                return result
    if compile_check:  # optional stage on a fresh build: graph breaks + compiled outputs
        from kernel_agent.kernels.compile_check import check

        result["compile_check"] = check(
            lambda: module.build(copy.deepcopy(reference)),
            cases,
            restore=replay.restore if replay else None,
        )
    if not _intact(result, guard, candidate_path, "after the correctness checks"):
        return result
    if save_outputs is not None:  # with the capture's case indices (quick: a subset)
        indices = result.get("quick", {}).get("cases") or list(range(len(cases)))
        torch.save({"cases": indices, "outputs": outputs}, save_outputs)
    del outputs
    # Inputs as captured, for re-verification (timing writes into them, e.g. KV caches).
    pristine = [(copy.deepcopy(c["args"]), copy.deepcopy(c["kwargs"])) for c in cases]
    seed = int.from_bytes(os.urandom(4), "little")
    result["perturb_seed"] = seed
    codes = integrity.reference_codes(reference, {c["method"] for c in cases})
    if quick or not device.startswith("cuda"):  # re-verified, never timed or profiled
        main = integrity.main_case(cases, case_reports)
        args, kwargs = copy.deepcopy(pristine[main])
        call = replay.call(cases[main], *holders)
        with integrity.count_calls(codes) as counts, torch.inference_mode():
            call(*args, **kwargs)
        ran = {codes[c]: n for c, n in counts.items() if n}
        if _fallback(result, main, cases[main], ran):
            return result
        if not _reverify(result, reference, holders, cases, pristine, seed, device, replay):
            return result
        if not _intact(result, guard, candidate_path, "at the end"):
            return result
        timing = "skipped: quick check" if quick else "skipped: no CUDA device"
        result.update(status="ok", correct=True, timing=timing)
        result["eval_seconds"] = round(time.perf_counter() - t0, 1)
        session["candidate"] = candidate  # a sweep times it next to its other configs
        session["holders"] = holders
        return result

    # 3. performance (reference vs candidate, same inputs, same entrypoint)
    from kernel_agent import hygiene, telemetry

    sched = telemetry.schedstat()  # how long this thread waits for a CPU while it times
    cores = telemetry.cores_busy()  # ... and how much others use its (timing) cores
    hygiene.phase(hygiene.TIMING)  # the agents' builds pause meanwhile (several sessions)
    compiled_ref = copy.deepcopy(reference) if compile_baseline else None
    compiled: dict[str, Any] = {}
    saved = 0.0
    ref_total = 0.0
    new_total = 0.0
    covered = 0.0  # calls per run of the target's instances the timed cases stand for
    users = capture.get("method_instances") or {}
    # with a bar to beat (early discard, kernels/early.py) every case first gets the rounds
    # after which the median of all of them is bounded, the rest only if it can still win
    first = early.min_rounds(ROUNDS) if early_best is not None else ROUNDS
    # the target's timing context (#226): graph timing when every timed case of both can be
    # captured in a CUDA graph, else eager, and the result says why
    reason = context_reason or DEFAULT_CONTEXT_REASON
    graph_state: str | None = None  # why graph timing is unavailable (None: not known to be)
    if context == GRAPH:
        probes = [
            (i, (replay.call(c, reference), replay.call(c, *holders)), c)
            for i, c in enumerate(cases)
            if c["count"]
        ]
        try:
            graph_state = _graph_unavailable(probes)
        except Exception:
            result.update(status="runtime_error", error=_short_tb())
            return result
        if graph_state is not None:
            context, reason = EAGER, f"{reason}; graph timing unavailable: timed eagerly"
    timed: list[_Rounds] = []
    restart = True
    while restart:  # once more, eagerly, when a capture fails after the probe passed
        restart, timed = False, []
        for i, (report, case) in enumerate(zip(case_reports, cases, strict=True)):
            if not case["count"]:  # correctness-only case (another workload setting): not timed
                continue
            try:
                # every call from the case's module state (restored outside the timed region)
                fns = (replay.call(case, reference), replay.call(case, *holders))
                ref_r, new_r, checked = timing_rounds(
                    *fns,
                    case["args"],
                    case["kwargs"],
                    rounds=first,
                    l2_flush=l2_flush,
                    context=context,
                )
                wall = wall_check(*fns, case["args"], case["kwargs"])
                if _hidden_work(wall):  # confirm: other processes can delay one measurement
                    again = wall_check(*fns, case["args"], case["kwargs"])
                    wall = min(wall, again, key=lambda w: w["hidden_ms"])
            except GraphUnavailable as exc:
                graph_state = f"case {i}: {exc}"
                context, reason = EAGER, f"{reason}; graph timing unavailable: timed eagerly"
                restart = True
                break
            except Exception:
                result.update(status="runtime_error", error=_short_tb(), failed_case=i)
                return result
            report["hidden_ms"] = round(wall["hidden_ms"], 4)
            try:  # after timing; a warning, never a failure (peak_memory_summary)
                peak = peak_memory(*fns, case["args"], case["kwargs"])
            except Exception as exc:
                report["peak_memory_error"] = f"{type(exc).__name__}: {exc}"[:200]
            else:
                report["ref_peak_mib"] = _mib(peak["ref_bytes"])
                report["new_peak_mib"] = _mib(peak["new_bytes"])
                report["peak_delta_mib"] = _mib(peak["new_bytes"] - peak["ref_bytes"])
            if _hidden_work(wall):
                _violation(
                    result,
                    "hidden_work",
                    f"case {i} ({case['signature']}): a call takes {wall['new_wall_ms']:.3f} ms "
                    f"between device-wide synchronisations but its timed stream sees only "
                    f"{wall['new_event_ms']:.3f} ms: {wall['hidden_ms']:.3f} ms of GPU work runs "
                    "on streams or threads the timed stream never waits for. Join side streams "
                    "back before returning (`with kernel_agent.concurrency.fork(name):` joins on "
                    "exit) and do not launch work from other threads.",
                    case=i,
                    **{k: round(v, 4) for k, v in wall.items()},
                )
                return result
            checked = checked or {}  # the kept timed call (always in the first rounds)
            if checked.get("failures"):
                result.update(
                    status="incorrect_timed_output",
                    stage="timed_output",
                    failed_check={"case": i, "check": "timed_output", **checked},
                    error=f"case {i} ({case['signature']}): the output of timed call "
                    f"#{checked['iteration']} differs from the reference on the same inputs "
                    f"({_first_error(checked['failures'])}); correctness was checked on the first "
                    "call only, so the candidate must compute every call (no caching by "
                    "address, shape or call count, no skipped work)",
                )
                return result
            timed.append(_Rounds(i, report, case, fns, ref_r, new_r))
    stop = None
    if early_best is not None:  # stop timing a candidate that cannot be a new best
        sofar = [early.Timed(t.case["count"], _ms(t.ref), _ms(t.new)) for t in timed]
        stop = early.discard(sofar, max(float(early_best), 1.0), ROUNDS)
    for t in timed if stop is None and first < ROUNDS else ():
        try:  # the rest of the rounds (the timed output was checked in the first ones)
            more = timing_rounds(
                *t.fns,
                t.case["args"],
                t.case["kwargs"],
                rounds=ROUNDS - first,
                l2_flush=l2_flush,
                verify=False,
                context=context,
            )
        except Exception:
            result.update(status="runtime_error", error=_short_tb(), failed_case=t.index)
            return result
        t.ref += more[0]
        t.new += more[1]
    if stop is not None:
        result["early"] = stop
    for t in timed:
        report, case = t.report, t.case
        method = case["method"]
        ref_t, new_t = median_round(t.ref), median_round(t.new)
        report["ref_ms"] = round(ref_t["median_ms"], 5)
        report["new_ms"] = round(new_t["median_ms"], 5)
        report["speedup"] = round(ref_t["median_ms"] / max(new_t["median_ms"], 1e-9), 3)
        report["timing_spread"] = round(max(ref_t["spread"], new_t["spread"]), 3)
        if clocks := [t["clock"] for t in (ref_t, new_t) if "clock" in t]:
            report["clock"] = round(min(clocks), 2)  # the lowest clock probe (1: full clocks)
        if compiled_ref is not None:
            try:
                if method not in compiled:
                    compiled[method] = torch.compile(
                        entrypoint(compiled_ref, method), mode="max-autotune-no-cudagraphs"
                    )
                comp_t = time_call(
                    replay.wrap(compiled[method], case, compiled_ref),
                    case["args"],
                    case["kwargs"],
                    l2_flush=l2_flush,
                )
                report["torch_compile_ms"] = round(comp_t["median_ms"], 5)
            except Exception as exc:
                report["torch_compile_ms"] = f"failed: {exc}"[:200]
        n = case["count"]
        ref_total += n * ref_t["median_ms"]
        new_total += n * new_t["median_ms"]
        # × the calls of the target's instances it stands for: those with its primary input
        # (instance groups), or an older capture's even split (kernels/weights.py)
        weight = weights.case_weight(case, users, int(capture.get("instances") or 1))
        report["target_calls"] = round(weight, 3)
        covered += weight
        saved += (ref_t["median_ms"] - new_t["median_ms"]) * weight
        report["timing"] = {context: _in_context(ref_t, new_t)}
        if graph_state is not None:
            report["timing"][GRAPH] = f"unavailable ({graph_state})"
    # the other context, for a winner (or with profile): what the verdict would be there,
    # and whether the candidate can run inside a CUDA-graphed stage at all
    if timed and stop is None and (profile or ref_total > new_total):
        other = GRAPH if context == EAGER else EAGER
        try:
            graph_state = _other_context(timed, other, l2_flush, graph_state)
        except Exception:
            result.update(status="runtime_error", error=_short_tb())
            return result
        for t in timed if graph_state is not None else ():
            t.report["timing"][GRAPH] = f"unavailable ({graph_state})"
    cpu_wait = telemetry.cpu_wait_share(sched)
    others = telemetry.others_share(cores)
    hygiene.phase("")

    if not _intact(result, guard, candidate_path, "after timing"):
        return result
    yield  # a batch times its other candidates now (its CUPTI tax comes after them all)
    # One profiled pass over the dominant case: threads, unjoined streams, fallback.
    main = integrity.main_case(cases, case_reports)
    try:
        activity = integrity.activity_check(
            replay.call(cases[main], reference),
            replay.call(cases[main], *holders),
            [copy.deepcopy(pristine[main]) for _ in range(ACTIVITY_CALLS)],
            codes,
        )
    except Exception:
        result.update(status="runtime_error", error=_short_tb(), failed_case=main)
        return result
    share = activity.get("custom_kernel_share")
    result["custom_kernel_share"] = share
    if streams := concurrency.used():  # named side streams (kernel_agent.concurrency)
        result["streams"] = streams
    if activity.get("undeclared_streams"):  # joined, but not named: a note
        result["undeclared_streams"] = activity["undeclared_streams"]
    if "reference_kernels" in activity:  # per call of the dominant case
        result["kernel_launches_reference"] = sum(activity["reference_kernels"].values())
        result["kernel_launches_candidate"] = round(sum(activity["candidate_kernels"].values()), 2)
    if activity.get("note"):
        result["activity_note"] = activity["note"]
    for key, what in (
        ("foreign_threads", "GPU work launched from another thread"),
        ("unjoined", "GPU work on other streams that the timed stream never waits for"),
    ):
        if activity.get(key):
            _violation(
                result,
                key,
                f"case {main} ({cases[main]['signature']}): {what}: "
                + "; ".join(activity[key][:3])
                + ". Launch all work from the calling thread and join side streams back "
                "before returning (`with kernel_agent.concurrency.fork(name):`).",
                case=main,
                details=activity[key],
            )
            return result
    if _fallback(result, main, cases[main], activity["reference_calls"], activity):
        return result

    # 4. re-verification after timing: fresh addresses and redrawn inputs
    if not _reverify(result, reference, holders, cases, pristine, seed, device, replay):
        return result
    result.update(
        status="ok",
        correct=True,
        speedup=round(ref_total / max(new_total, 1e-9), 3),
        est_saved_ms_per_run=round(saved, 3),
        est_saved_calls=weights.summary(capture, covered),  # basis: instance groups / even split
        ref_ms_weighted=round(ref_total, 4),
        new_ms_weighted=round(new_total, 4),
        **context_fields(case_reports, context, l2_flush, reason),
    )
    if cpu_wait is not None:  # a contended CPU delays launches (telemetry.dirty)
        result["cpu_wait_share"] = cpu_wait
    if others is not None:
        result["cpu_others_share"] = others
    if (memory := peak_memory_summary(case_reports)) is not None:
        result["peak_memory"] = memory
    # speed of light per case (after timing, never inside it): sol_ms, pct_of_sol, bound
    annotate(
        result,
        reference,
        cases,
        l2_flush=l2_flush,
        precision=capture_precision(capture),
        restore=(lambda c: replay.restore(c, reference)) if replay else None,
    )
    if profile:
        try:
            first = cases[0]
            a, k = first["args"], first["kwargs"]
            result["kernels_candidate"] = _kernel_table(replay.call(first, *holders), a, k)
            result["kernels_reference"] = _kernel_table(replay.call(first, reference), a, k)
        except Exception as exc:
            result["profile_error"] = str(exc)[:500]
        from kernel_agent.kernels.ncu import compiler_stats  # registers, spills (no GPU work)

        if stats := compiler_stats(module, candidate):
            result["compiler_stats"] = stats
        from kernel_agent.kernels import sass  # the opcodes it issues (no GPU work, #230)

        profiled = [str(r.get("kernel")) for r in result.get("kernels_candidate") or []]
        result["sass"] = sass.census(module, candidate, ran=profiled, gpu=sass.gpu_facts())
    if not _intact(result, guard, candidate_path, "at the end"):
        return result
    result["eval_seconds"] = round(time.perf_counter() - t0, 1)
    return result


def _timeout_result(timeout: float, stderr: str | bytes | None) -> dict[str, Any]:
    """Timeout result; says whether compiling or checking/benchmarking ran out of time."""
    text = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr or ""
    compile_s = None
    for line in text.splitlines():
        if line.startswith(COMPILE_MARKER):
            compile_s = float(line[len(COMPILE_MARKER) :])
    result: dict[str, Any] = {"status": "timeout", "correct": False, "compile_s": compile_s}
    if compile_s is None:
        result["error"] = f"exceeded {timeout:.0f}s before the first call finished (compiling?)"
    else:
        result["error"] = (
            f"exceeded {timeout:.0f}s; compile + first call took {compile_s:.0f}s, "
            "the rest went to correctness checks and benchmarking"
        )
    return result


def run_evaluation(
    capture_path: Path,
    candidate_path: Path,
    *,
    profile: bool = False,
    l2_flush: bool = False,
    compile_baseline: bool = False,
    timeout: float = 300.0,
    capture_sha256: str | None = None,
    compile_check: bool = False,
    quick: bool = False,
    early_best: float | None = None,
    context: str = "eager",
    context_reason: str | None = None,
) -> dict[str, Any]:
    """Evaluate in a fresh subprocess under the GPU lock (``capture_sha256``, ``quick``,
    ``early_best``, ``context``: see :func:`evaluate`; the subprocess checks the bytes it
    loads), then check its result outside the candidate's process
    (:func:`_check_reference_timing`, :func:`_check_outputs`). The result says on which GPU
    of the pool it ran
    (``gpu_index``, :mod:`kernel_agent.gpulock`) and which evaluator measured it
    (``evaluator_version``, set here: the candidate's process cannot choose it).

    With clean timing on (:mod:`kernel_agent.hygiene`, several sessions at once) a timed
    evaluation first builds the candidate's ``load_inline`` extensions without the GPU
    (:mod:`kernel_agent.kernels.prebuild`; ``prebuild`` in the result), and one whose timing
    was dirty (another process on the GPU, a contended CPU, a reference slowdown: see
    :func:`_dirty`) is measured once more, first of its class in the GPU queue; the result
    is the second measurement (``retimed``: why the first did not count; ``timing_dirty``:
    the second was dirty too)."""
    from kernel_agent import hygiene

    timed = hygiene.current() is not None and not quick
    pre: dict[str, Any] | None = None
    if timed and not holding():  # a sweep's evaluation: its configs were built before its lock
        from kernel_agent.kernels import prebuild

        pre = prebuild.prebuild(candidate_path, prebuild.inputs_capture(capture_path))
    data: dict[str, Any] = {}
    first: str | None = None
    for attempt in range(2 if timed else 1):
        data, why = _evaluate_once(
            capture_path,
            candidate_path,
            profile=profile,
            l2_flush=l2_flush,
            compile_baseline=compile_baseline,
            timeout=timeout,
            capture_sha256=capture_sha256,
            compile_check=compile_check,
            quick=quick,
            watch=timed,
            early_best=early_best,
            context=context,
            context_reason=context_reason,
        )
        if why is None:
            break
        if attempt:
            data["timing_dirty"] = why
        else:
            first = why
            if (job := gpuqueue.current()) is not None:
                job.requeue()  # measured again at once: first of its class
    if first is not None:
        data["retimed"] = first
    if pre is not None and pre.get("built"):
        data["prebuild"] = {k: pre[k] for k in ("seconds", "built") if k in pre}
    return data


#: Subprocesses of one :func:`run_evaluations` (a crash or a timeout starts the next one
#: after the candidate it happened in); what is left after them is evaluated one by one.
BATCH_RUNS = 3


def run_evaluations(
    capture_path: Path,
    candidate_paths: list[Path],
    *,
    timeout: float = 300.0,
    capture_sha256: str | None = None,
    compile_check: bool = False,
    early_best: float | None = None,
    l2_flush: bool = False,
    context: str = "eager",
    context_reason: str | None = None,
) -> list[dict[str, Any]]:
    """Evaluate several candidates of one capture (one agent's variants of one idea: the
    ``evaluate_candidates`` tool, issue #190) in one subprocess under one GPU-lock
    acquisition: the capture loads once and the process starts once, and each candidate goes
    through every stage of :func:`evaluate` with its own build, its own integrity snapshot
    and its own timing interleaved with the reference, then the checks outside its process
    (:func:`_check_reference_timing`, :func:`_check_outputs`); one result per candidate, in
    order, each what :func:`run_evaluation` would return. ``timeout`` is per candidate;
    ``l2_flush``, ``context``: their timing context (:func:`evaluate`).

    A candidate that kills the subprocess (or runs out of its time) gets ``crash`` (or
    ``timeout``) and the candidates after it go on in a new subprocess (at most
    :data:`BATCH_RUNS`, then one by one). With clean timing on, the candidates are built
    off the GPU first and those whose timing was dirty are measured once more, as
    :func:`run_evaluation` does."""
    from kernel_agent import hygiene

    paths = list(candidate_paths)
    if len(paths) <= 1:
        return [
            run_evaluation(
                capture_path,
                path,
                timeout=timeout,
                capture_sha256=capture_sha256,
                compile_check=compile_check,
                early_best=early_best,
                l2_flush=l2_flush,
                context=context,
                context_reason=context_reason,
            )
            for path in paths
        ]
    timed = hygiene.current() is not None
    pre: dict[int, dict[str, Any]] = {}
    if timed and not holding():
        from kernel_agent.kernels import prebuild

        inputs = prebuild.inputs_capture(capture_path)
        pre = {i: prebuild.prebuild(path, inputs) for i, path in enumerate(paths)}
    results: dict[int, dict[str, Any]] = {}
    first: dict[int, str] = {}
    todo = list(range(len(paths)))
    for attempt in range(2 if timed else 1):
        measured = _evaluate_batch(
            capture_path,
            paths,
            todo,
            timeout=timeout,
            capture_sha256=capture_sha256,
            compile_check=compile_check,
            watch=timed,
            early_best=early_best,
            timing={"l2_flush": l2_flush, "context": context, "context_reason": context_reason},
        )
        todo = []
        for i, (data, why) in measured.items():
            if why is not None and not attempt:
                first[i] = why
                todo.append(i)
            elif why is not None:
                data["timing_dirty"] = why
            results[i] = data
        if not todo:
            break
        if (job := gpuqueue.current()) is not None:
            job.requeue()  # measured again at once: first of its class
    for i, why in first.items():
        results[i]["retimed"] = why
    for i, built in pre.items():
        if built.get("built"):
            results[i]["prebuild"] = {k: built[k] for k in ("seconds", "built") if k in built}
    return [results[i] for i in range(len(paths))]


def _evaluate_batch(
    capture_path: Path,
    paths: list[Path],
    indices: list[int],
    *,
    timeout: float,
    capture_sha256: str | None,
    compile_check: bool,
    watch: bool,
    early_best: float | None,
    timing: dict[str, Any] | None = None,
) -> dict[int, tuple[dict[str, Any], str | None]]:
    """The candidates ``paths[i]`` of ``indices`` in batch subprocesses under one hold of the
    GPU (:func:`run_evaluations`): ``{i: (result, why its timing was dirty)}``; ``timing``:
    ``l2_flush``, ``context`` and ``context_reason`` (:func:`evaluate`)."""
    version = {"evaluator_version": evaluator_version()}
    workdir = Path(tempfile.mkdtemp(prefix="ka-evals-"))
    out: dict[int, dict[str, Any]] = {}
    whys: dict[int, str | None] = {}
    timing = timing or {}
    flags = _flags(
        capture_sha256=capture_sha256,
        compile_check=compile_check,
        early_best=early_best,
        **timing,
    )
    try:
        with gpu_lock() as gpu:
            ensure_peaks()
            hold = _watch(gpu) if watch else None
            with hold or contextlib.nullcontext():
                left, runs = list(indices), 0
                while left and runs < BATCH_RUNS:
                    runs += 1
                    outputs = workdir / f"run{runs}"
                    outputs.mkdir()
                    batch = [paths[i] for i in left]
                    done, culprit, stopped = _spawn_batch(
                        capture_path, batch, outputs, flags, timeout
                    )
                    for k, data in done.items():
                        data.update(gpu_index=gpu, **version)
                        data["outputs_file"] = str(outputs / f"{k}.pt")
                        _check_reference_timing(
                            data,
                            capture_path,
                            capture_sha256,
                            l2_flush=bool(timing.get("l2_flush")),
                            gpu=gpu,
                        )
                        out[left[k]] = data
                    if culprit is not None:  # it died (or ran out of time) in this one
                        out[left[culprit]] = {**stopped, "gpu_index": gpu, **version}
                    left = [left[k] for k in range(len(left)) if left[k] not in out]
                for i in left:  # after BATCH_RUNS processes: one by one
                    out[i] = run_evaluation(
                        capture_path,
                        paths[i],
                        timeout=timeout,
                        capture_sha256=capture_sha256,
                        compile_check=compile_check,
                        early_best=early_best,
                        **timing,
                    )
                    whys[i] = None  # measured (and re-measured) by run_evaluation already
            for i, data in out.items():
                if i not in whys:
                    whys[i] = _dirty(hold, data) if watch else None
        for data in out.values():
            if saved := data.pop("outputs_file", None):
                _check_outputs(data, capture_path, capture_sha256, Path(saved))  # CPU only
        return {i: (out[i], whys[i]) for i in indices}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _spawn_batch(
    capture_path: Path, paths: list[Path], outputs: Path, flags: list[str], timeout: float
) -> tuple[dict[int, dict[str, Any]], int | None, dict[str, Any]]:
    """One batch subprocess (:func:`main` with ``--also``): the results it reported by their
    place in ``paths``, the place of the candidate it died (or ran out of time) in (None: it
    finished) and why it stopped (crash or timeout)."""
    nonce = secrets.token_hex(16)  # on stdin, read before any candidate is imported
    cmd = [sys.executable, "-m", "kernel_agent.kernels.evaluate", str(capture_path), str(paths[0])]
    for path in paths[1:]:
        cmd += ["--also", str(path)]
    cmd += ["--json", "-", "--outputs-dir", str(outputs), "--nonce-stdin", *flags]
    limit = timeout * len(paths)
    try:
        proc = subprocess.run(
            cmd,
            input=nonce + "\n",
            capture_output=True,
            text=True,
            timeout=limit,
            env=child_env(),
        )
        stdout, stopped = proc.stdout, None
    except subprocess.TimeoutExpired as exc:
        text = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout
        stdout, stopped = text or "", _timeout_result(limit, exc.stderr)
    done, running = _parse_batch(stdout, nonce, len(paths))
    if len(done) == len(paths):
        return done, None, {}
    if stopped is None:
        tail = (proc.stderr or proc.stdout)[-4000:]
        stopped = {"status": "crash", "correct": False, "returncode": proc.returncode}
        stopped["error"] = tail
    culprit = running if running is not None and running not in done else None
    if culprit is None:  # no candidate ran: the first without a result
        culprit = min(k for k in range(len(paths)) if k not in done)
    return done, culprit, stopped


def _parse_batch(stdout: str, nonce: str, n: int) -> tuple[dict[int, dict[str, Any]], int | None]:
    """The ``result`` lines of a batch subprocess of ``n`` candidates by ``index`` (the last
    line of each counts, as :func:`_parse_result` takes the last line of one evaluation)
    and the candidate of its last ``running`` line."""
    marker = f"{RESULT_MARKER}{nonce}@@"
    out: dict[int, dict[str, Any]] = {}
    running: int | None = None
    for line in stdout.splitlines():
        if not line.startswith(marker):
            continue
        with contextlib.suppress(ValueError):
            data = json.loads(line[len(marker) :])
            index = data.pop("index", None) if isinstance(data, dict) else None
            if not isinstance(index, int) or not 0 <= index < n:
                continue
            if data.pop("event", None) == "running":
                running = index
            else:
                out[index] = data
    return out, running


def _dirty(watch: Any, data: dict[str, Any]) -> str | None:
    """Why a timed evaluation is not clean (``telemetry.dirty``: another process on the GPU,
    a contended CPU), or a reference slowdown the candidate-free re-time confirmed: measured
    again once, it is an integrity violation only when it happens again."""
    from kernel_agent import telemetry

    if why := telemetry.dirty(watch, data):
        return why
    if data.get("stage") == "reference_timing":
        return "the reference ran slower next to the candidate than in a candidate-free process"
    return None


def _flags(
    *,
    profile: bool = False,
    l2_flush: bool = False,
    compile_baseline: bool = False,
    capture_sha256: str | None = None,
    compile_check: bool = False,
    quick: bool = False,
    early_best: float | None = None,
    context: str = "eager",
    context_reason: str | None = None,
) -> list[str]:
    """The evaluator subprocess's flags for these options (:func:`main`)."""
    cmd = ["--profile"] if profile else []
    cmd += ["--l2-flush"] if l2_flush else []
    cmd += ["--compile-baseline"] if compile_baseline else []
    cmd += ["--capture-sha256", capture_sha256] if capture_sha256 else []
    cmd += ["--compile-check"] if compile_check else []
    cmd += ["--quick"] if quick else []
    cmd += ["--early-best", repr(float(early_best))] if early_best is not None else []
    cmd += ["--context", context] if context != "eager" else []
    cmd += ["--context-reason", context_reason] if context_reason else []
    return cmd


def _evaluate_once(
    capture_path: Path,
    candidate_path: Path,
    *,
    profile: bool,
    l2_flush: bool,
    compile_baseline: bool,
    timeout: float,
    capture_sha256: str | None,
    compile_check: bool,
    quick: bool,
    watch: bool,
    early_best: float | None = None,
    context: str = "eager",
    context_reason: str | None = None,
) -> tuple[dict[str, Any], str | None]:
    """One evaluation (:func:`run_evaluation`) and, with ``watch``, why its timing was dirty
    (None: clean, or not watched)."""
    version = {"evaluator_version": evaluator_version()}
    workdir = Path(tempfile.mkdtemp(prefix="ka-eval-"))
    outputs = workdir / "outputs.pt"
    nonce = secrets.token_hex(16)  # on stdin, read before the candidate is imported
    cmd = [
        sys.executable,
        "-m",
        "kernel_agent.kernels.evaluate",
        str(capture_path),
        str(candidate_path),
        "--json",
        "-",
        "--outputs",
        str(outputs),
        "--nonce-stdin",
    ]
    cmd += _flags(
        profile=profile,
        l2_flush=l2_flush,
        compile_baseline=compile_baseline,
        capture_sha256=capture_sha256,
        compile_check=compile_check,
        quick=quick,
        early_best=early_best,
        context=context,
        context_reason=context_reason,
    )
    try:
        with gpu_lock() as gpu:  # reference and candidate run on this GPU, in one process
            ensure_peaks()  # measured once per GPU + torch version, outside the evaluation
            hold = _watch(gpu) if watch else None  # other processes on the GPU meanwhile
            with hold or contextlib.nullcontext():
                try:
                    proc = subprocess.run(
                        cmd,
                        input=nonce + "\n",
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        env=child_env(),
                    )
                except subprocess.TimeoutExpired as exc:
                    timed_out = _timeout_result(timeout, exc.stderr) | {"gpu_index": gpu}
                    return timed_out | version, None
                data = _parse_result(proc.stdout, nonce)
                if data is None:
                    tail = (proc.stderr or proc.stdout)[-4000:]
                    crash = {"status": "crash", "correct": False, "returncode": proc.returncode}
                    return {**crash, "error": tail, "gpu_index": gpu, **version}, None
                data.update(gpu_index=gpu, **version)
                _check_reference_timing(
                    data, capture_path, capture_sha256, l2_flush=l2_flush, gpu=gpu
                )  # in a subprocess pinned to the same GPU (child_env)
            why = _dirty(hold, data) if watch else None
        _check_outputs(data, capture_path, capture_sha256, outputs)  # CPU only
        return data, why
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _watch(gpu: int) -> Any:
    """``telemetry.HoldWatch`` of this thread's hold of GPU ``gpu``."""
    from kernel_agent import telemetry

    return telemetry.HoldWatch(gpu, hold_token())


def _parse_result(stdout: str, nonce: str | None) -> dict[str, Any] | None:
    """The result line of the evaluator subprocess (with ``nonce``: only a line tagged
    with it, so a line printed by the candidate cannot stand in for the result)."""
    marker = RESULT_MARKER + (f"{nonce}@@" if nonce else "")
    for line in stdout.splitlines()[::-1]:
        if line.startswith(marker):
            data: dict[str, Any] = json.loads(line[len(marker) :])
            return data
    return None


#: Captures (answer key only) and candidate-free reference timings already loaded
#: by this process, keyed by path + digest (or mtime and size).
_TRUTH: dict[tuple[str, str], dict[str, Any]] = {}
_CLEAN_REF_MS: dict[tuple[str, str, bool, str, str], list[list[float]]] = {}


def _capture_key(path: Path, sha256: str | None) -> tuple[str, str]:
    stat = path.stat()
    return str(path.resolve()), sha256 or f"{stat.st_mtime_ns}:{stat.st_size}"


def _check_outputs(
    result: dict[str, Any], capture_path: Path, capture_sha256: str | None, outputs: Path
) -> None:
    """Outside the candidate's process: the candidate's outputs of the correctness stage
    (saved by the subprocess) against the capture, with this process's comparator
    (``integrity_violation`` if a passing result does not hold up)."""
    import torch

    from kernel_agent.kernels import integrity
    from kernel_agent.profiling.capture import load_capture

    if result.get("status") != "ok" or not result.get("correct"):
        return
    try:
        saved = torch.load(outputs, map_location="cpu", weights_only=True)
    except Exception as exc:
        _violation(
            result,
            "parent_outputs",
            f"the evaluator's record of the candidate's outputs is missing or unreadable "
            f"outside its process ({type(exc).__name__}: {str(exc)[:200]})",
        )
        return
    key = _capture_key(capture_path, capture_sha256)
    truth = _TRUTH.get(key)
    if truth is None:
        try:
            capture = load_capture(capture_path, device="cpu", sha256=capture_sha256)
        except Exception as exc:  # e.g. model code that cannot be imported here
            result["parent_check"] = f"skipped: {type(exc).__name__}: {str(exc)[:200]}"
        else:
            keys = ("method", "signature", "output", "args", "kwargs", "post_args", "post_kwargs")
            truth = {"cases": [{k: c.get(k) for k in keys} for c in capture["cases"]]}
            truth["tier"] = capture.get("tier")  # the tolerance tier (kernels/compare.py)
            _TRUTH.clear()  # one capture at a time: answer keys can be large
            _TRUTH[key] = truth
    if truth is not None:
        failures = integrity.compare_saved_outputs(truth, saved)
        if failures:
            first = failures[0]
            _violation(
                result,
                "parent_outputs",
                f"the candidate passed the checks inside its process, but its outputs do not "
                f"match the capture when compared outside it (case {first.get('case')}, "
                f"{first.get('name')}: {first.get('error') or first.get('mismatch_frac')}): "
                "the comparison inside the candidate's process was tampered with",
                failures=failures[:5],
            )
            return
        result["parent_check"] = "ok"


def _check_reference_timing(
    result: dict[str, Any],
    capture_path: Path,
    capture_sha256: str | None,
    *,
    l2_flush: bool,
    gpu: Any = None,
) -> None:
    """Outside the candidate's process (hold the GPU lock): per case, the reference time
    measured next to the candidate against one measured in a candidate-free subprocess
    (:func:`reference_timing`; cached per capture and process).  A slowdown is only
    reported when a fresh candidate-free measurement confirms it."""
    from kernel_agent.kernels import integrity

    if result.get("status") != "ok" or not result.get("correct"):
        return
    if not any(c.get("ref_ms") for c in result.get("cases") or []):
        return  # nothing timed
    context = str(result.get("context") or "eager")  # the one it was timed in (#226)
    key = (*_capture_key(capture_path, capture_sha256), l2_flush, str(gpu), context)
    clean = _CLEAN_REF_MS.get(key)
    slow: list[str] = []
    for attempt in range(2):
        if clean is None or attempt:
            clean = _clean_reference_timing(
                capture_path, capture_sha256, l2_flush=l2_flush, context=context
            )
            if clean is None:
                result["reference_check"] = "skipped: the candidate-free timing failed"
                return
            _CLEAN_REF_MS[key] = clean
        slow = integrity.reference_slowdown(result, clean)
        if not slow:
            result["reference_check"] = "ok"
            return
    _violation(
        result,
        "reference_timing",
        "the reference runs slower next to the candidate than in a process without it ("
        + "; ".join(slow[:3])
        + "): the candidate slows the reference down (global flags, patched or overridden "
        "torch ops, background load)",
        slow=slow,
        clean_ref_ms=clean,
    )


def _clean_reference_timing(
    capture_path: Path,
    capture_sha256: str | None,
    *,
    l2_flush: bool = False,
    context: str = "eager",
) -> list[list[float]] | None:
    """``[ref_ms, instability]`` per case from :func:`reference_timing` in a subprocess
    that never imports a candidate."""
    cmd = [
        sys.executable,
        "-m",
        "kernel_agent.kernels.evaluate",
        str(capture_path),
        "--reference-timing",
        "--json",
        "-",
    ]
    if l2_flush:
        cmd.append("--l2-flush")
    if context != "eager":
        cmd += ["--context", context]
    if capture_sha256:
        cmd += ["--capture-sha256", capture_sha256]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=child_env())
    except subprocess.TimeoutExpired:
        return None
    data = _parse_result(proc.stdout, None) or {}
    ref_ms, unstable = data.get("ref_ms"), data.get("instability")
    if not isinstance(ref_ms, list) or not isinstance(unstable, list):
        return None
    return [[float(m), float(u)] for m, u in zip(ref_ms, unstable, strict=True)]


def reference_timing(
    capture_path: Path,
    *,
    l2_flush: bool = False,
    capture_sha256: str | None = None,
    context: str = "eager",
) -> dict[str, Any]:
    """Time the reference of every case like :func:`evaluate` does (in ``context``), without
    any candidate in the process: interleaved rounds of the reference against itself give
    two medians per case; ``ref_ms`` is their mean and ``instability`` their relative
    difference (a busy GPU)."""
    from kernel_agent import hygiene, toolchain

    toolchain.setup()
    from kernel_agent.kernels.bench import compare_timing
    from kernel_agent.profiling.capture import load_capture
    from kernel_agent.profiling.state import Replay

    capture = load_capture(capture_path, device="cuda", sha256=capture_sha256)
    reference = capture["module"].eval()
    replay = Replay(capture, reference)  # every call from its case's module state
    ref_ms, unstable = [], []
    with hygiene.timing():  # the agents' builds pause meanwhile (several sessions)
        for case in capture["cases"]:
            if not case.get("count", 1):  # correctness-only case: the evaluator does not time it
                ref_ms.append(0.0)
                unstable.append(0.0)
                continue
            fn = replay.call(case, reference)
            one, two = compare_timing(
                fn,
                fn,
                case["args"],
                case["kwargs"],
                l2_flush=l2_flush,
                verify=False,
                context=context,
            )
            a, b = one["median_ms"], two["median_ms"]
            ref_ms.append(round((a + b) / 2, 5))
            unstable.append(round(abs(a - b) / max(min(a, b), 1e-9), 4))
    return {"status": "ok", "ref_ms": ref_ms, "instability": unstable}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("capture", type=Path)
    parser.add_argument("candidate", type=Path, nargs="?")
    parser.add_argument("--profile", action="store_true", help="add per-kernel tables")
    parser.add_argument("--l2-flush", action="store_true")
    parser.add_argument("--compile-baseline", action="store_true")
    parser.add_argument("--capture-sha256", help="refuse a capture without this digest")
    parser.add_argument("--compile-check", action="store_true", help="torch.compile compat")
    parser.add_argument("--quick", action="store_true", help="smallest + largest case, untimed")
    parser.add_argument(
        "--early-best",
        type=float,
        help="stop timing a correct candidate that cannot beat this speedup (kernels/early.py)",
    )
    parser.add_argument(
        "--context",
        choices=("eager", "graph"),
        default="eager",
        help="timing context: eager calls, or calls captured in a CUDA graph (#226)",
    )
    parser.add_argument("--context-reason", help="why the context is the target's (recorded)")
    parser.add_argument("--json", default=None, help="write result JSON ('-' = stdout marker)")
    parser.add_argument("--outputs", type=Path, help="save the candidate's outputs here")
    parser.add_argument(
        "--nonce-stdin", action="store_true", help="read a nonce for the result line from stdin"
    )
    parser.add_argument(
        "--reference-timing", action="store_true", help="only time the reference (no candidate)"
    )
    parser.add_argument(
        "--ncu-mode", action="store_true", help="run the candidate for ncu only (kernels/ncu.py)"
    )
    parser.add_argument("--ncu-calls", type=int, default=3, help="--ncu-mode: profiled calls")
    parser.add_argument(
        "--also",
        type=Path,
        action="append",
        help="a batch (run_evaluations): another candidate, evaluated after the ones before",
    )
    parser.add_argument(
        "--outputs-dir", type=Path, help="a batch: save candidate i's outputs as DIR/i.pt"
    )
    ns = parser.parse_args(argv)
    if ns.candidate is None and not ns.reference_timing:
        parser.error("a candidate is required")
    if ns.ncu_mode:  # under ncu, no checks or timing (kernels/ncu.py profile_candidate)
        from kernel_agent.kernels.ncu import ncu_entry

        return ncu_entry(
            ns.capture, ns.candidate, capture_sha256=ns.capture_sha256, calls=ns.ncu_calls
        )
    # Read before the candidate is imported; it never sees the nonce in its environment.
    tag = (sys.stdin.readline().strip() + "@@") if ns.nonce_stdin else ""
    if ns.also:
        return _batch_main(ns, tag)
    try:
        if ns.reference_timing:
            result = reference_timing(
                ns.capture,
                l2_flush=ns.l2_flush,
                capture_sha256=ns.capture_sha256,
                context=ns.context,
            )
        else:
            result = evaluate(
                ns.capture,
                ns.candidate,
                profile=ns.profile,
                l2_flush=ns.l2_flush,
                compile_baseline=ns.compile_baseline,
                capture_sha256=ns.capture_sha256,
                compile_check=ns.compile_check,
                quick=ns.quick,
                save_outputs=ns.outputs,
                early_best=ns.early_best,
                context=ns.context,
                context_reason=ns.context_reason,
            )
    except Exception:
        result = {"status": "harness_error", "correct": False, "error": _short_tb()}
    payload = _dumps(result, default=str)
    if ns.json == "-":
        _stdout.write(RESULT_MARKER + tag + payload + "\n")
        _stdout.flush()
    elif ns.json:
        Path(ns.json).write_text(payload)
    else:
        print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("correct") else 1


def _batch_main(ns: argparse.Namespace, tag: str) -> int:
    """``--also``: the candidates of a batch (:func:`run_evaluations`) in this process, the
    capture loaded once, each through every stage of :func:`evaluate` with its own build
    and integrity snapshot, in two passes: build, correctness and timing of each candidate
    in turn (:func:`_stages` up to its pause), then for each the profiled activity pass,
    the re-verification and the rest. So no candidate is timed after a profiler's CUPTI
    subscription made the process's launches slower. Lines (``index``): ``running`` before
    each pass of a candidate, ``result`` when it is done."""
    import gc

    from kernel_agent import concurrency

    def emit(event: dict[str, Any]) -> None:
        payload = _dumps(event, default=str)
        if ns.json == "-":
            _stdout.write(RESULT_MARKER + tag + payload + "\n")
            _stdout.flush()
        else:
            print(payload, flush=True)

    shared: dict[str, Any] = {}
    paused: dict[int, tuple[Generator[None, None, dict[str, Any]], list[Any]]] = {}
    for i, path in enumerate([ns.candidate, *ns.also]):  # 1. build, check and time each
        emit({"index": i, "event": "running", "stage": "timing"})
        concurrency.reset()  # the named streams it uses are its own
        session: dict[str, Any] = {k: shared[k] for k in ("capture", "replay") if k in shared}
        guards: list[Any] = []
        stages = _stages(
            ns.capture,
            path,
            guards,
            profile=False,
            l2_flush=ns.l2_flush,
            compile_baseline=False,
            device=None,
            capture_sha256=ns.capture_sha256,
            compile_check=ns.compile_check,
            quick=False,
            save_outputs=ns.outputs_dir / f"{i}.pt" if ns.outputs_dir else None,
            config={},
            session=session,
            early_best=ns.early_best,
            context=ns.context,
            context_reason=ns.context_reason,
        )
        result: dict[str, Any] | None = None
        try:
            next(stages)
            paused[i] = (stages, guards)  # timed: the rest after the others' timing
        except StopIteration as done:  # it failed or is not timed (CPU): done now
            result = _hang(done.value)
        except Exception:
            result = {"status": "harness_error", "correct": False, "error": _short_tb()}
        shared.update({k: session[k] for k in ("capture", "replay") if k in session})
        if result is not None:
            for guard in guards:  # before the next candidate is imported
                guard.restore()
            emit({"index": i, "event": "result", **result})
    for i in sorted(paused):  # 2. the profiled passes and the re-verification
        emit({"index": i, "event": "running", "stage": "checks"})
        concurrency.reset()
        stages, guards = paused.pop(i)
        try:
            result = _finish(stages)
        except Exception:
            result = {"status": "harness_error", "correct": False, "error": _short_tb()}
        finally:
            for guard in guards:
                guard.restore()
        emit({"index": i, "event": "result", **result})
        del stages
        gc.collect()
    _restore([])  # the comparator's tier
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
