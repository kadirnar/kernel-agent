"""Evaluate one kernel candidate against a captured module.

Candidate contract (``candidates/<name>.py``)::

    def build(reference: torch.nn.Module) -> torch.nn.Module:
        '''Return a drop-in replacement for ``reference``: same forward
        signature, same outputs (within tolerance), same in-place side effects
        (e.g. KV-cache updates).  Reuse ``reference``'s parameters/buffers.
        Return ``reference`` itself for instances the kernel does not support.'''

Each captured case is replayed through the entrypoint it was recorded from:
``candidate(*args, **kwargs)`` for ``forward`` cases and
``candidate.<method>(*args, **kwargs)`` otherwise (e.g. ``forward_step`` of a
custom decode loop), for correctness and timing alike.  A candidate that lacks
a captured method is a ``build_error``.

Stages and their failure statuses:

1. ``build_error``: import and ``build()``.
2. ``incorrect``: every captured case against the captured outputs and in-place
   side effects (:mod:`kernels.compare`), plus aliasing parity with a live
   reference call (:func:`kernels.verify.alias_errors`).
3. ``incorrect_timed_output``: timing (CUDA only, :func:`kernels.bench.compare_timing`)
   rotates between input copies and checks the output of one random timed call.
4. ``incorrect_perturbed``: re-verification after timing (on CPU right after
   stage 2) at fresh addresses and with redrawn inputs (:mod:`kernels.verify`).

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
  reference does not launch, or when none of it is (``custom_kernel_share`` 0);
* :func:`run_evaluation` (outside the candidate's process) compares the
  candidate's saved outputs with the capture and the reference timing with one
  measured in a candidate-free process (10 % margin); results arrive on a line
  tagged with a nonce the candidate cannot read from its environment.

A failed stage is named in ``stage``, with details in ``failed_check``.

Run as a subprocess (so compiler crashes and illegal memory accesses cannot
take down the orchestrator)::

    python -m kernel_agent.kernels.evaluate CAPTURE CANDIDATE [--profile] [--json OUT]
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import importlib.util
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from kernel_agent.gpulock import child_env, gpu_lock
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


def load_candidate_module(path: Path) -> Any:
    path = path.resolve()
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


def _kernel_table(fn: Any, args: Any, kwargs: Any, top: int = 15) -> list[dict[str, Any]]:
    import torch
    from torch.profiler import ProfilerActivity, profile

    from kernel_agent.kernels.bench import has_mutable_state

    a, k = (
        (copy.deepcopy(args), copy.deepcopy(kwargs))
        if has_mutable_state(args, kwargs)
        else (
            args,
            kwargs,
        )
    )
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


def _first_error(failures: list[dict[str, Any]]) -> str:
    first = failures[0] if failures else {}
    detail = first.get("error") or (
        f"mismatch {first.get('mismatch_frac')}, max abs err {first.get('max_abs_err')}"
    )
    return f"{first.get('name', '?')}: {detail}"


def _reverify(
    result: dict[str, Any],
    reference: Any,
    candidate: Any,
    cases: list[dict[str, Any]],
    pristine: list[tuple[Any, Any]],
    seed: int,
    device: str,
) -> bool:
    """Stage 4 (:func:`kernels.verify.reverify_case` on every case); False and
    ``incorrect_perturbed`` in ``result`` if a check fails."""
    import torch

    from kernel_agent.kernels.verify import reverify_case
    from kernel_agent.profiling.methods import entrypoint
    from kernel_agent.workloads.base import synchronize

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    for i, (case, inputs) in enumerate(zip(cases, pristine, strict=True)):
        method = case["method"]
        try:
            failed = reverify_case(
                entrypoint(reference, method),
                entrypoint(candidate, method),
                case,
                inputs,
                gen,
                synchronize,
            )
        except Exception:
            result.update(status="runtime_error", correct=False, error=_short_tb(), failed_case=i)
            return False
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
                "unused cache slots)",
            )
            return False
    result["checks"] = ["captured", "aliasing", "timed_output", "perturbed"]
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


def _intact(result: dict[str, Any], guard: Any, candidate_path: Path, where: str) -> bool:
    """Integrity checkpoint: False (and ``integrity_violation``) if the candidate changed
    watched state or runs threads."""
    from kernel_agent.kernels.integrity import candidate_threads

    problems = guard.changes()
    problems += [
        f"thread {t} runs candidate code" for t in candidate_threads(candidate_path.parent)
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
    result: dict[str, Any], i: int, case: dict[str, Any], ran: dict[str, int], share: Any
) -> bool:
    """Decide ``fallback`` for the main case ``i`` from the reference entrypoints that ran
    during the candidate's calls (``ran``) and its ``custom_kernel_share``."""
    from kernel_agent.kernels.integrity import MIN_CUSTOM_SHARE

    if share != 0 and not (ran and (share is None or share < MIN_CUSTOM_SHARE)):
        return False
    why = []
    if ran:
        why.append("runs the reference's " + ", ".join(f"`{k}`" for k in sorted(ran)))
    if share is not None:
        why.append(f"spends {share:.0%} of its GPU time in kernels the reference does not launch")
    result.update(
        status="fallback",
        correct=False,
        stage="fallback",
        failed_check={"case": i, "check": "fallback", "reference_calls": ran, "share": share},
        error=f"case {i} ({case['signature']}, the dominant case: {case['count']} calls per "
        f"run) {' and '.join(why)}: the dominant case must run your own kernels. Do not call "
        "or inherit the reference's entrypoint for it and do not re-launch the reference's "
        "ops; fall back only for shapes you do not support.",
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
) -> dict[str, Any]:
    """Build, check and time one candidate.  ``device`` defaults to CUDA when
    available; on CPU only correctness is checked (timing needs CUDA events).
    ``capture_sha256``: refuse (status ``tampered``) a capture without this digest.
    ``compile_check`` adds ``result["compile_check"]`` (``compile_check.py``).
    ``quick``: correctness on the smallest and the largest case only, never timed.
    ``save_outputs``: write the candidate's outputs of the correctness stage there
    (:func:`kernels.integrity.flat_outputs`), for :func:`run_evaluation` to check.
    Global state the candidate changed is restored on return."""
    guards: list[Any] = []
    try:
        return _evaluate(
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
        )
    finally:
        for guard in guards:
            guard.restore()


def _evaluate(
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
) -> dict[str, Any]:
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda"):
        from kernel_agent import toolchain

        toolchain.setup()

    from kernel_agent.kernels import integrity
    from kernel_agent.kernels.bench import compare_timing, time_call, wall_check
    from kernel_agent.kernels.compare import compare_side_effects, compare_structures
    from kernel_agent.kernels.verify import alias_errors
    from kernel_agent.profiling.capture import load_capture
    from kernel_agent.profiling.methods import entrypoint
    from kernel_agent.workloads.base import synchronize

    result: dict[str, Any] = {
        "candidate": str(candidate_path),
        "status": "error",
        "correct": False,
    }
    t0 = time.perf_counter()
    try:
        capture = load_capture(capture_path, device=device, sha256=capture_sha256)
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
    guard = integrity.Snapshot(reference)  # before the candidate is imported
    guards.append(guard)

    # 1. import + build
    t_build = time.perf_counter()
    try:
        module = load_candidate_module(candidate_path)
        given = copy.deepcopy(reference)
        candidate = module.build(given)
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
                out = entrypoint(candidate, case["method"])(*args, **kwargs)
            synchronize()
        except Exception:
            result.update(status="runtime_error", error=_short_tb(), failed_case=i)
            return result
        if i == 0:
            # import + build + first call: where load_inline / JIT backends compile
            result["compile_s"] = round(time.perf_counter() - t_build, 1)
            print(f"{COMPILE_MARKER}{result['compile_s']}", file=sys.stderr, flush=True)
        checks = compare_structures(case["output"], out, "output")
        checks += compare_side_effects(case["args"], case["post_args"], args, "args")
        checks += compare_side_effects(case["kwargs"], case["post_kwargs"], kwargs, "kwargs")
        with torch.inference_mode():  # after the candidate: the aliasing of a live call
            ref_out = entrypoint(reference, case["method"])(*ref_args, **ref_kwargs)
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
    result["cases"] = case_reports
    if not all_ok:
        failed = next(i for i, r in enumerate(case_reports) if not r["ok"])
        result.update(  # the failures are in result["cases"]
            status="incorrect",
            stage="correctness",
            failed_check={"case": failed, "check": "captured"},
        )
        return result
    if compile_check:  # optional stage on a fresh build: graph breaks + compiled outputs
        from kernel_agent.kernels.compile_check import check

        result["compile_check"] = check(lambda: module.build(copy.deepcopy(reference)), cases)
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
        with integrity.count_calls(codes) as counts, torch.inference_mode():
            entrypoint(candidate, cases[main]["method"])(*args, **kwargs)
        ran = {codes[c]: n for c, n in counts.items() if n}
        if _fallback(result, main, cases[main], ran, None):
            return result
        if not _reverify(result, reference, candidate, cases, pristine, seed, device):
            return result
        if not _intact(result, guard, candidate_path, "at the end"):
            return result
        timing = "skipped: quick check" if quick else "skipped: no CUDA device"
        result.update(status="ok", correct=True, timing=timing)
        result["eval_seconds"] = round(time.perf_counter() - t0, 1)
        return result

    # 3. performance (reference vs candidate, same inputs, same entrypoint)
    compiled_ref = copy.deepcopy(reference) if compile_baseline else None
    compiled: dict[str, Any] = {}
    saved = 0.0
    ref_total = 0.0
    new_total = 0.0
    for i, (report, case) in enumerate(zip(case_reports, cases, strict=True)):
        method = case["method"]
        if not case["count"]:  # correctness-only case (another workload setting): not timed
            continue
        try:
            ref_t, new_t = compare_timing(
                entrypoint(reference, method),
                entrypoint(candidate, method),
                case["args"],
                case["kwargs"],
                l2_flush=l2_flush,
            )
            fns = (entrypoint(reference, method), entrypoint(candidate, method))
            wall = wall_check(*fns, case["args"], case["kwargs"])
            if _hidden_work(wall):  # confirm: other processes can delay one measurement
                again = wall_check(*fns, case["args"], case["kwargs"])
                wall = min(wall, again, key=lambda w: w["hidden_ms"])
        except Exception:
            result.update(status="runtime_error", error=_short_tb(), failed_case=i)
            return result
        report["hidden_ms"] = round(wall["hidden_ms"], 4)
        if _hidden_work(wall):
            _violation(
                result,
                "hidden_work",
                f"case {i} ({case['signature']}): a call takes {wall['new_wall_ms']:.3f} ms "
                f"between device-wide synchronisations but its timed stream sees only "
                f"{wall['new_event_ms']:.3f} ms: {wall['hidden_ms']:.3f} ms of GPU work runs "
                "on streams or threads the timed stream never waits for. Join side streams "
                "back before returning (`torch.cuda.current_stream().wait_stream(s)`) and do "
                "not launch work from other threads.",
                case=i,
                **{k: round(v, 4) for k, v in wall.items()},
            )
            return result
        timed = new_t.get("timed_output") or {}
        if timed.get("failures"):
            result.update(
                status="incorrect_timed_output",
                stage="timed_output",
                failed_check={"case": i, "check": "timed_output", **timed},
                error=f"case {i} ({case['signature']}): the output of timed call "
                f"#{timed['iteration']} differs from the reference on the same inputs "
                f"({_first_error(timed['failures'])}); correctness was checked on the first "
                "call only, so the candidate must compute every call (no caching by "
                "address, shape or call count, no skipped work)",
            )
            return result
        report["ref_ms"] = round(ref_t["median_ms"], 5)
        report["new_ms"] = round(new_t["median_ms"], 5)
        report["speedup"] = round(ref_t["median_ms"] / max(new_t["median_ms"], 1e-9), 3)
        report["timing_spread"] = round(max(ref_t["spread"], new_t["spread"]), 3)
        if compiled_ref is not None:
            try:
                if method not in compiled:
                    compiled[method] = torch.compile(
                        entrypoint(compiled_ref, method), mode="max-autotune-no-cudagraphs"
                    )
                comp_t = time_call(
                    compiled[method], case["args"], case["kwargs"], l2_flush=l2_flush
                )
                report["torch_compile_ms"] = round(comp_t["median_ms"], 5)
            except Exception as exc:
                report["torch_compile_ms"] = f"failed: {exc}"[:200]
        n = case["count"]
        ref_total += n * ref_t["median_ms"]
        new_total += n * new_t["median_ms"]
        # times the instances that call this entrypoint (all instances for old captures)
        users = capture.get("method_instances", {}).get(method, capture.get("instances", 1))
        saved += n * (ref_t["median_ms"] - new_t["median_ms"]) * users

    if not _intact(result, guard, candidate_path, "after timing"):
        return result
    # One profiled pass over the dominant case: threads, unjoined streams, fallback.
    main = integrity.main_case(cases, case_reports)
    method = cases[main]["method"]
    try:
        activity = integrity.activity_check(
            entrypoint(reference, method),
            entrypoint(candidate, method),
            [copy.deepcopy(pristine[main]) for _ in range(ACTIVITY_CALLS)],
            codes,
        )
    except Exception:
        result.update(status="runtime_error", error=_short_tb(), failed_case=main)
        return result
    share = activity.get("custom_kernel_share")
    result["custom_kernel_share"] = share
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
                "before returning.",
                case=main,
                details=activity[key],
            )
            return result
    if _fallback(result, main, cases[main], activity["reference_calls"], share):
        return result

    # 4. re-verification after timing: fresh addresses and redrawn inputs
    if not _reverify(result, reference, candidate, cases, pristine, seed, device):
        return result
    result.update(
        status="ok",
        correct=True,
        speedup=round(ref_total / max(new_total, 1e-9), 3),
        est_saved_ms_per_run=round(saved, 3),
        ref_ms_weighted=round(ref_total, 4),
        new_ms_weighted=round(new_total, 4),
    )
    # speed of light per case (after timing, never inside it): sol_ms, pct_of_sol, bound
    annotate(result, reference, cases, l2_flush=l2_flush)
    if profile:
        try:
            first = cases[0]
            a, k, m = first["args"], first["kwargs"], first["method"]
            result["kernels_candidate"] = _kernel_table(entrypoint(candidate, m), a, k)
            result["kernels_reference"] = _kernel_table(entrypoint(reference, m), a, k)
        except Exception as exc:
            result["profile_error"] = str(exc)[:500]
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
) -> dict[str, Any]:
    """Evaluate in a fresh subprocess under the GPU lock (``capture_sha256``, ``quick``:
    see :func:`evaluate`; the subprocess checks the bytes it loads), then check its
    result outside the candidate's process (:func:`_check_reference_timing`,
    :func:`_check_outputs`). The result says on which GPU of the pool it ran
    (``gpu_index``, :mod:`kernel_agent.gpulock`)."""
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
    if profile:
        cmd.append("--profile")
    if l2_flush:
        cmd.append("--l2-flush")
    if compile_baseline:
        cmd.append("--compile-baseline")
    if capture_sha256:
        cmd += ["--capture-sha256", capture_sha256]
    if compile_check:
        cmd.append("--compile-check")
    if quick:
        cmd.append("--quick")
    try:
        with gpu_lock() as gpu:  # reference and candidate run on this GPU, in one process
            ensure_peaks()  # measured once per GPU + torch version, outside the evaluation
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
                return _timeout_result(timeout, exc.stderr) | {"gpu_index": gpu}
            data = _parse_result(proc.stdout, nonce)
            if data is None:
                tail = (proc.stderr or proc.stdout)[-4000:]
                return {
                    "status": "crash",
                    "correct": False,
                    "returncode": proc.returncode,
                    "error": tail,
                    "gpu_index": gpu,
                }
            data["gpu_index"] = gpu
            _check_reference_timing(
                data, capture_path, capture_sha256, l2_flush=l2_flush, gpu=gpu
            )  # in a subprocess pinned to the same GPU (child_env)
        _check_outputs(data, capture_path, capture_sha256, outputs)  # CPU only
        return data
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


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
_CLEAN_REF_MS: dict[tuple[str, str, bool, str], list[list[float]]] = {}


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
    key = (*_capture_key(capture_path, capture_sha256), l2_flush, str(gpu))  # per GPU
    clean = _CLEAN_REF_MS.get(key)
    slow: list[str] = []
    for attempt in range(2):
        if clean is None or attempt:
            clean = _clean_reference_timing(capture_path, capture_sha256, l2_flush=l2_flush)
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
    capture_path: Path, capture_sha256: str | None, *, l2_flush: bool = False
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
    capture_path: Path, *, l2_flush: bool = False, capture_sha256: str | None = None
) -> dict[str, Any]:
    """Time the reference of every case like :func:`evaluate` does, without any candidate
    in the process: interleaved rounds of the reference against itself give two
    medians per case; ``ref_ms`` is their mean and ``instability`` their relative
    difference (a busy GPU)."""
    from kernel_agent import toolchain

    toolchain.setup()
    from kernel_agent.kernels.bench import compare_timing
    from kernel_agent.profiling.capture import load_capture
    from kernel_agent.profiling.methods import entrypoint

    capture = load_capture(capture_path, device="cuda", sha256=capture_sha256)
    reference = capture["module"].eval()
    ref_ms, unstable = [], []
    for case in capture["cases"]:
        if not case.get("count", 1):  # correctness-only case: the evaluator does not time it
            ref_ms.append(0.0)
            unstable.append(0.0)
            continue
        fn = entrypoint(reference, case.get("method", "forward"))
        one, two = compare_timing(
            fn, fn, case["args"], case["kwargs"], l2_flush=l2_flush, verify=False
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
    parser.add_argument("--json", default=None, help="write result JSON ('-' = stdout marker)")
    parser.add_argument("--outputs", type=Path, help="save the candidate's outputs here")
    parser.add_argument(
        "--nonce-stdin", action="store_true", help="read a nonce for the result line from stdin"
    )
    parser.add_argument(
        "--reference-timing", action="store_true", help="only time the reference (no candidate)"
    )
    ns = parser.parse_args(argv)
    if ns.candidate is None and not ns.reference_timing:
        parser.error("a candidate is required")
    # Read before the candidate is imported; it never sees the nonce in its environment.
    tag = (sys.stdin.readline().strip() + "@@") if ns.nonce_stdin else ""
    try:
        if ns.reference_timing:
            result = reference_timing(
                ns.capture, l2_flush=ns.l2_flush, capture_sha256=ns.capture_sha256
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


if __name__ == "__main__":
    raise SystemExit(main())
