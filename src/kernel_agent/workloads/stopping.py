"""Stop-condition check: a natural-length run, in which the model decides its own length.

Autoregressive workloads keep their runs comparable with a fixed length (VoxCPM:
``min_len = max_len = patches``), so the main and held-out runs never let the stop
condition decide. A transform that skipped, delayed or reordered the stop decision would
pass every other check and still truncate or run on in real inference. Workloads with a
stop condition opt in with
:meth:`~kernel_agent.workloads.base.Workload.natural_length_run`:

* ``analyze`` runs it once on the baseline, untimed and free running (the model stops
  where it wants, within ``max_steps``), and stores the result as
  ``baseline_output_natural.pt`` (hashed and read-only in ``.truth/``). ``baseline.json``
  ``natural_length`` records the generated steps, the step at which the stop fired and the
  stop logit margin there (and the smallest one before it).
* ``e2e`` runs it once more on every candidate, untimed and teacher forced on the
  baseline's trajectory, so every stop decision sees the baseline history; the candidate
  must stop at the same step
  (:meth:`~kernel_agent.workloads.base.Workload.compare_natural_length`: exact by default,
  ``-o stop_tolerance=1`` accepts ±1 step at a near-tie of the baseline's stop logits).

Workloads without a stop condition (the default hook) are unaffected: no extra run, no
``natural_length`` metrics.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

import torch

from kernel_agent.workloads.base import STOP_NEAR_TIE, Workload, synchronize


def declares(workload: Workload) -> bool:
    """The workload overrides :meth:`Workload.natural_length_run`."""
    return type(workload).natural_length_run is not Workload.natural_length_run


def stop_summary(output: dict[str, Any]) -> dict[str, Any]:
    """``steps``, ``stopped`` and, with recorded margins, ``stop_step`` (0-based), the
    ``stop_margin`` there and the ``closest_margin`` of the steps before it at which the
    stop was consulted (``> min_steps``): how near the run came to stopping early."""
    steps = int(output["steps"])
    min_steps = int(output.get("min_steps", 0))
    max_steps = output.get("max_steps")
    info: dict[str, Any] = {"steps": steps, "min_steps": min_steps, "max_steps": max_steps}
    margins = output.get("stop_margins")
    if not margins or len(margins) != steps:
        info["stopped"] = max_steps is None or steps < int(max_steps)
        return info
    last = steps - 1
    info["stopped"] = last > min_steps and margins[last] > 0
    info["stop_step"] = last if info["stopped"] else None
    info["stop_margin"] = round(float(margins[last]), 4)
    before = [abs(float(m)) for i, m in enumerate(margins[:last]) if i > min_steps]
    info["closest_margin"] = round(min(before), 4) if before else None
    return info


# ------------------------------------------------------------------ analyze


def record_baseline(workload: Workload) -> tuple[Any, dict[str, Any]]:
    """The baseline natural-length result (``None`` when the workload has no stop
    condition) and ``baseline.json`` ``natural_length``. Raises when the run fails."""
    none = {"status": "none", "reason": "the workload has no stop condition"}
    if not declares(workload):
        return None, none
    with torch.inference_mode():
        synchronize()
        start = time.perf_counter()
        output = workload.natural_length_run()
        synchronize()
        ms = (time.perf_counter() - start) * 1000
    if output is None:
        return None, none
    info = {"status": "ok", "ms": round(ms, 3), **stop_summary(output)}
    info["near_tie"] = float(workload.options.get("stop_near_tie", STOP_NEAR_TIE))
    return output, info


def save_baseline(workload: Workload, path: Path) -> dict[str, Any]:
    """:func:`record_baseline`, its result saved to ``path`` (none: no file). Failures are
    recorded, not raised: ``e2e`` then skips the stop-condition check."""
    path.unlink(missing_ok=True)  # never leave a stale natural-length baseline behind
    try:
        output, info = record_baseline(workload)
    except Exception:
        return {"status": "error", "error": traceback.format_exc()[-2000:]}
    if output is not None:
        torch.save(output, path)
    return info


# ------------------------------------------------------------------ e2e


def check(workload: Workload, reference: Any, baseline: dict[str, Any]) -> dict[str, Any] | None:
    """``metrics.natural_length`` of an ``e2e`` run: ``{"passed", "reason", ...}``, or
    ``None`` when the workload has no stop condition. ``reference`` is the stored
    natural-length baseline (``None``: skipped)."""
    info = baseline.get("natural_length") or {}
    if not declares(workload) or info.get("status") == "none":
        return None
    if reference is None:
        why = "its run failed" if info.get("status") == "error" else "analyze predates it"
        return {
            "passed": True,
            "reason": "",
            "skipped": f"no natural-length baseline ({why}): re-run analyze",
        }
    try:
        with torch.inference_mode():
            output = workload.natural_length_run(reference)
        if output is None:
            raise RuntimeError("natural_length_run returned None for a candidate")
        cmp = workload.compare_natural_length(reference, output)
    except Exception as exc:
        return {
            "passed": False,
            "reason": f"the natural-length run failed: {type(exc).__name__}: {exc}"[:500],
            "error": traceback.format_exc()[-3000:],
        }
    return {"passed": cmp.passed, "reason": cmp.reason, **cmp.metrics}


# ------------------------------------------------------------------ reporting


def messages(baseline: dict[str, Any]) -> list[str]:
    """``analyze`` log lines about the natural-length run."""
    info = baseline.get("natural_length") or {}
    status = info.get("status")
    if status == "error":
        return [
            "WARNING: the natural-length run failed; e2e evaluations skip the stop-condition "
            f"check ({str(info.get('error', ''))[-300:]})"
        ]
    if status != "ok":
        return []
    out = [f"analyze: natural-length run: {_describe(info)} ({info.get('ms')} ms)"]
    if not info.get("stopped"):
        out.append(
            "WARNING: the natural-length baseline never stopped; the stop condition is not "
            "exercised (choose another natural input)"
        )
    near = float(info.get("near_tie", STOP_NEAR_TIE))
    margins = [m for m in (info.get("stop_margin"), info.get("closest_margin")) if m is not None]
    if any(abs(m) <= near for m in margins):
        out.append(
            f"WARNING: the natural-length baseline has a stop logit near-tie (|margin| <= "
            f"{near:g}): numerically-correct candidates may stop one step away; "
            "`-o stop_tolerance=1` accepts that"
        )
    return out


def summary_lines(baseline: dict[str, Any]) -> list[str]:
    """Bullets for the ``End-to-end quality check`` section of ``profile/summary.md``."""
    info = baseline.get("natural_length") or {}
    if info.get("status") != "ok":
        return []
    return [
        f"* **stop condition**: the main and held-out runs have a fixed length, so every "
        f"candidate also runs a natural-length input (the model decides its own length; "
        f"baseline: {_describe(info)}), untimed and teacher forced on the baseline's "
        "trajectory. It must stop at the same step: a transform that skips, delays (e.g. "
        "checks the flag one step late) or reorders the stop decision fails.",
    ]


def summary_text(result: dict[str, Any]) -> str:
    """One phrase for reports: the ``metrics.natural_length`` of an ``e2e`` run."""
    if result.get("skipped"):
        return f"natural length skipped ({result['skipped']})"
    if not result.get("passed"):
        return f"natural length FAILED: {result.get('reason')}"
    tolerated = " at a near-tie" if result.get("tolerated") else ""
    return f"natural length passed ({result.get('steps')} steps{tolerated})"


def _describe(info: dict[str, Any]) -> str:
    text = f"{info.get('steps')} steps"
    if info.get("stop_step") is not None:
        text += f", stop at step {info['stop_step']} (margin {info.get('stop_margin'):+.3f}"
        if info.get("closest_margin") is not None:
            text += f", closest before: {info['closest_margin']:.3f}"
        text += ")"
    elif not info.get("stopped"):
        text += f", no stop before max_steps={info.get('max_steps')}"
    return text
