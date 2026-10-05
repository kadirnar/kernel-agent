"""Held-out end-to-end input and the memoisation probe.

``e2e`` times the workload's main input, the same prompt / text and seed on every
run.  A transform that memoised outputs across runs (keyed on the input, or on
nothing) would therefore pass both the free-running and the teacher-forced
check.  The held-out input closes that gap:

* ``analyze`` runs the baseline on held-out input 1
  (:meth:`~kernel_agent.workloads.base.Workload.holdout_options`: another
  prompt / text / audio / seed, the same shapes where possible) and stores its
  output as ``baseline_output_holdout.pt`` (hashed and read-only in ``.truth/``).
  ``baseline.json`` ``holdout`` records whether it differs from the main
  output (``distinct``) and whether a second held-out input differs from the
  first (``probe_distinct``).
* ``e2e`` judges every candidate on the held-out input too (teacher forced where
  the workload supports it), untimed; only the main input is timed.  The
  candidate fails when the held-out quality check fails, or when its held-out
  output equals its main output although the baseline's differ (replayed
  outputs).
* Memoisation probe: after the main timing has warmed everything up and the
  held-out run has warmed up its shapes, one run on a *fresh* input (held-out
  variant ``>= 2``, random per evaluation, never seen by any process: the
  shapes of input 1, other content) is timed.  Real inference takes about as
  long as the repeated main-input runs; a memoised one is computed for the
  first time.  The candidate fails when that run, and a second fresh one that
  confirms it (one slow run can be a busy GPU), take more than
  :data:`MEMO_RATIO` x the median timed main run (and at least
  :data:`MEMO_MIN_GAP_MS` longer), or when its output equals the held-out
  output although the baseline's differ.
"""

from __future__ import annotations

import math
import secrets
import traceback
from pathlib import Path
from typing import Any

import torch

from kernel_agent.kernels.compare import flatten
from kernel_agent.workloads.base import Workload, timed_run

#: A fresh-input run this many times slower than the repeated main-input runs is memoisation.
MEMO_RATIO = 3.0
#: ... and at least this much slower in absolute terms (timer noise of tiny workloads).
MEMO_MIN_GAP_MS = 10.0


def outputs_equal(a: Any, b: Any) -> bool:
    """Bit-identical outputs: the same tensors (names, shapes, dtypes, values)."""
    ta, tb = flatten(a), flatten(b)
    if not ta and not tb:
        try:
            return bool(a == b)
        except Exception:
            return False
    if ta.keys() != tb.keys():
        return False
    for name, x in ta.items():
        y = tb[name]
        if x.shape != y.shape or x.dtype != y.dtype:
            return False
        if not torch.equal(x.detach().cpu(), y.detach().cpu()):
            return False
    return True


def run_variant(workload: Workload, options: dict[str, Any]) -> tuple[Any, Any, float]:
    """``(inputs, output, ms)`` of one run under the option overrides ``options``; ``ms``
    is the workload's metric (the latency by default, ``objective.py``), as for the
    timed main-input runs it is compared with."""
    with workload.with_options(options):
        inputs = workload.make_inputs()
        output, ms, _ = timed_run(workload, inputs)
    return inputs, output, ms


# ------------------------------------------------------------------ analyze


def record_baseline(workload: Workload, main_output: Any) -> tuple[Any, dict[str, Any]]:
    """Baseline output of held-out input 1 (``None`` when the workload declares
    none) and ``baseline.json`` ``holdout``.  Raises when the held-out run fails."""
    options = workload.holdout_options(1)
    if not options:
        return None, {"status": "none", "reason": "the workload declares no held-out input"}
    inputs, output, ms = run_variant(workload, options)
    info: dict[str, Any] = {
        "status": "ok",
        "options": options,
        "ms": round(ms, 3),
        "distinct": not outputs_equal(main_output, output),
    }
    probe = workload.holdout_options(2)
    if probe:
        _, other, _ = run_variant(workload, probe)
        info["probe_distinct"] = not outputs_equal(output, other)
    if workload.supports_teacher_forcing:
        from kernel_agent.workloads.quality import teacher_forcing_selfcheck

        with workload.with_options(options):
            info["teacher_forcing"] = teacher_forcing_selfcheck(workload, inputs, output)
    return output, info


def save_baseline(workload: Workload, main_output: Any, path: Path) -> dict[str, Any]:
    """:func:`record_baseline`, its output saved to ``path`` (none: no file).  Failures
    are recorded, not raised: ``e2e`` then skips the held-out check."""
    path.unlink(missing_ok=True)  # never leave a stale held-out output behind
    try:
        output, info = record_baseline(workload, main_output)
    except Exception:
        return {"status": "error", "error": traceback.format_exc()[-2000:]}
    if output is not None:
        torch.save(output, path)
    return info


# ------------------------------------------------------------------ e2e


def memo_probe(
    workload: Workload,
    holdout_output: Any,
    *,
    main_ms: float,
    probe_distinct: bool | None,
    variant: int | None = None,
) -> dict[str, Any]:
    """Time one run on a fresh input (held-out ``variant``, random by default).  A slow
    run is confirmed on a second fresh input before it counts: a busy GPU slows one
    run, memoisation every fresh one."""
    first = variant if variant is not None else 2 + secrets.randbelow(2**30)
    second = first + 1 if variant is not None else 2 + secrets.randbelow(2**30)
    if not workload.holdout_options(first):
        return {"flagged": False, "skipped": "the workload declares no fresh held-out input"}

    def slow(ms: float) -> bool:
        return ms > MEMO_RATIO * main_ms and ms - main_ms >= MEMO_MIN_GAP_MS

    times: list[float] = []
    same = False
    for v in (first, second):
        _, output, ms = run_variant(workload, workload.holdout_options(v) or {})
        times.append(ms)
        same = same or (bool(probe_distinct) and outputs_equal(output, holdout_output))
        if same or not slow(ms):
            break
    memoised = all(slow(ms) for ms in times) and len(times) == 2
    ratio = min(times) / main_ms if main_ms > 0 else math.inf
    info: dict[str, Any] = {
        "flagged": memoised or same,
        "variant": first,
        "fresh_ms": round(times[0], 3),
        "main_median_ms": round(main_ms, 3),
        "fresh_over_repeat": round(times[0] / main_ms if main_ms > 0 else math.inf, 3),
        "threshold": MEMO_RATIO,
        "equals_holdout_output": same,
    }
    if len(times) == 2:
        info["confirm_ms"] = round(times[1], 3)
    if memoised:
        info["reason"] = (
            f"memoisation: first runs on two fresh inputs of already warmed-up shapes took "
            f"{times[0]:.1f} and {times[1]:.1f} ms, at least {ratio:.1f}x the median of the "
            f"repeated main-input runs ({main_ms:.1f} ms; limit {MEMO_RATIO:g}x): outputs "
            "are reused across runs"
        )
    elif same:
        info["reason"] = (
            "memoisation: two held-out inputs whose baseline outputs differ gave "
            "identical outputs: outputs are reused across runs"
        )
    return info


def check(
    workload: Workload,
    reference: Any,
    main_reference: Any,
    *,
    main_output: Any,
    main_ms: float,
    baseline: dict[str, Any],
    chaotic: bool,
    variant: int | None = None,
) -> dict[str, Any]:
    """``metrics.holdout`` of an ``e2e`` run: ``{"passed", "reason", ...}``.

    ``reference`` is the stored held-out baseline output (``None``: skipped),
    ``main_reference`` / ``main_output`` the baseline and candidate outputs of
    the main input, ``main_ms`` the candidate's median timed main run."""
    from kernel_agent.workloads.quality import assess

    options = workload.holdout_options(1)
    if not options:
        return {"passed": True, "reason": "", "skipped": "the workload declares no held-out input"}
    if reference is None:
        return {
            "passed": True,
            "reason": "",
            "skipped": "no held-out baseline output (analyze ran before held-out inputs "
            "existed, or the held-out run failed): re-run analyze",
        }
    result: dict[str, Any] = {"options": options}
    problems: list[str] = []
    try:
        inputs, output, ms = run_variant(workload, options)
        result["run_ms"] = round(ms, 3)
        with workload.with_options(options):
            verdict = assess(workload, inputs, reference, output, chaotic=chaotic)
        result["quality"] = verdict
        if not verdict["passed"]:
            problems.append(verdict["reason"] or "quality check failed")
        same = outputs_equal(main_output, output)
        result["equals_main_output"] = same
        if same and not outputs_equal(main_reference, reference):
            problems.append(
                "the held-out output equals the main output although the baseline's differ: "
                "outputs are replayed, not computed"
            )
        memo = memo_probe(
            workload,
            output,
            main_ms=main_ms,
            probe_distinct=(baseline.get("holdout") or {}).get("probe_distinct"),
            variant=variant,
        )
        result["memoisation"] = memo
        if memo.get("flagged"):
            problems.append(memo["reason"])
    except Exception as exc:
        result["error"] = traceback.format_exc()[-3000:]
        problems.append(f"the held-out run failed: {type(exc).__name__}: {exc}"[:500])
    result["passed"] = not problems
    result["reason"] = "; ".join(problems)
    return result


# ------------------------------------------------------------------ reporting


def messages(baseline: dict[str, Any]) -> list[str]:
    """``analyze`` log lines about the held-out input."""
    info = baseline.get("holdout") or {}
    status = info.get("status")
    if status == "error":
        return [
            "WARNING: the held-out run failed; e2e evaluations skip the held-out check "
            f"({str(info.get('error', ''))[-300:]})"
        ]
    if status != "ok":
        return []
    out = [f"analyze: held-out input {_fmt(info.get('options'))} recorded ({info.get('ms')} ms)"]
    if info.get("distinct") is False:
        out.append(
            "WARNING: the held-out output equals the main output; holdout_options() does not "
            "change what the model computes"
        )
    tf = info.get("teacher_forcing")
    if tf is not None and not tf.get("passed"):
        out.append(
            "WARNING: teacher-forcing self-check FAILED on the held-out input "
            f"({tf.get('reason')}); every candidate will fail its held-out check"
        )
    return out


def summary_lines(baseline: dict[str, Any]) -> list[str]:
    """Bullets for the ``End-to-end quality check`` section of ``profile/summary.md``."""
    info = baseline.get("holdout") or {}
    if info.get("status") != "ok":
        return []
    return [
        f"* every candidate is also judged on a **held-out input** ({_fmt(info.get('options'))}), "
        "untimed, with the same quality check; only the main input is timed. A model or "
        "transform that only works for the main input (a CUDA graph with the main input "
        "baked in, outputs cached across runs) fails.",
        f"* memoisation probe: one run on a fresh input is timed after warm-up; more than "
        f"{MEMO_RATIO:g}x the repeated main-input runs, or replayed outputs, fail the candidate.",
    ]


def summary_text(result: dict[str, Any]) -> str:
    """One phrase for reports: the ``metrics.holdout`` of an ``e2e`` run."""
    if result.get("skipped"):
        return f"held-out input skipped ({result['skipped']})"
    if not result.get("passed"):
        return f"held-out input FAILED: {result.get('reason')}"
    ratio = (result.get("memoisation") or {}).get("fresh_over_repeat")
    probe = f" (fresh input {ratio:.2f}x the repeated runs)" if ratio is not None else ""
    return f"held-out input passed{probe}"


def _fmt(options: Any) -> str:
    if not isinstance(options, dict):
        return str(options)
    return ", ".join(f"{k}={str(v)[:40]!s}" for k, v in options.items())
