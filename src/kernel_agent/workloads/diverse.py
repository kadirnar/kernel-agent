"""Diverse input set: quality on varied content, and how much a speedup depends on it (#170).

Some optimisations are faster on some inputs than on others of the same shapes:
speculative decoding (prompt-lookup / n-gram drafts, a draft model) is as fast as its
drafts are right, an early exit as early as the input allows, a cache of repeated content
as useful as the content repeats. Timed on the benchmark input alone they can look faster
than a user will see them, and judged on it alone their quality can look better too. So,
besides the benchmark input:

* ``analyze`` runs the baseline on every input of
  :meth:`~kernel_agent.workloads.base.Workload.diverse_inputs` (:data:`WARMUP` untimed run
  and :data:`ITERS` timed runs each), records the times in ``baseline.json`` ``diverse``
  and the outputs as ``baseline_output_diverse.pt`` (hashed and read-only in ``.truth/``).
* ``e2e`` runs every candidate that passed the main checks the same way
  (``metrics.diverse``; ``e2e_ab``: state B of every integration step): each input's output
  is judged against the baseline's with the workload's quality check (teacher forced where
  supported; a failure fails the candidate), and its time gives a per-input speedup with
  the input's decode counters
  (:meth:`~kernel_agent.workloads.base.Workload.report_stats`); the median, min and max
  speedup summarise them.
* A candidate is **data-dependent** when the spread of its per-input speedups exceeds the
  noise, or when its decode steps per token change with the input
  (:mod:`kernel_agent.diversity`). The ledger (``flags``), ``status``, the integration and
  the report say so and show the set's median next to the benchmark speedup. Nothing is
  rejected for it.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import traceback
from pathlib import Path
from typing import Any

import torch

from kernel_agent.diversity import summarize
from kernel_agent.workloads.base import Workload, decode_stats, timed_run

#: Untimed runs per input before the timed ones (compilation, graph capture of new content).
WARMUP = 1
#: Timed runs per input; the median counts.
ITERS = 2


def key_of(options: dict[str, Any]) -> str:
    """A short digest of an input's option overrides: a baseline counts only for the input
    it was recorded on (the workload's set may change between ``analyze`` and ``e2e``)."""
    blob = json.dumps(options, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _spread(times: list[float]) -> float | None:
    median = statistics.median(times) if times else 0.0
    if len(times) < 2 or median <= 0:
        return None
    return round((max(times) - min(times)) / median, 4)


def run_input(
    workload: Workload, options: dict[str, Any], *, warmup: int = WARMUP, iters: int = ITERS
) -> tuple[Any, Any, dict[str, Any]]:
    """``(inputs, last output, timing)`` of one input: ``warmup`` untimed and ``iters`` timed
    runs of the workload's metric under the option overrides; ``timing`` holds ``ms`` (the
    median), ``times_ms``, ``spread`` and the decode counters (``decode_stats``)."""
    with workload.with_options(options):
        inputs = workload.make_inputs()
        with torch.inference_mode():
            for _ in range(warmup):
                workload.run(inputs)
        times, counters, output = [], [], None
        for _ in range(max(iters, 1)):
            output, ms, detail = timed_run(workload, inputs)
            times.append(ms)
            counters.append(detail.get("decode_stats") or {})
    timing: dict[str, Any] = {
        "ms": round(statistics.median(times), 3),
        "times_ms": [round(t, 3) for t in times],
        "spread": _spread(times),
    }
    if (stats := decode_stats(counters)) is not None:
        timing["decode_stats"] = stats
    return inputs, output, timing


def _error_line(text: str) -> str:
    return (text.strip().splitlines() or [""])[-1][:300]


# ------------------------------------------------------------------ analyze


def record_baseline(
    workload: Workload, *, warmup: int = WARMUP, iters: int = ITERS
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The baseline's outputs by label and ``baseline.json`` ``diverse``: the time of every
    input (``status`` ok), or why there is none (``none``). An input that fails is recorded
    with its ``error``; ``e2e`` skips it."""
    declared = workload.diverse_inputs()
    if not declared:
        return {}, {"status": "none", "reason": "the workload declares no diverse input set"}
    if workload.serving() is not None:
        return {}, {"status": "none", "reason": "a serving benchmark times its own requests"}
    outputs: dict[str, Any] = {}
    inputs: list[dict[str, Any]] = []
    for label, options in declared.items():
        entry: dict[str, Any] = {"label": label, "key": key_of(options)}
        try:
            _, outputs[label], timing = run_input(workload, options, warmup=warmup, iters=iters)
            entry.update(timing)
        except Exception:
            entry["error"] = traceback.format_exc()[-1500:]
        inputs.append(entry)
    info = {"status": "ok", "metric": workload.metric, "iters": iters, "inputs": inputs}
    return outputs, info


def save_baseline(workload: Workload, path: Path) -> dict[str, Any]:
    """:func:`record_baseline`, its outputs saved to ``path`` (none: no file). Failures are
    recorded, not raised: ``e2e`` then skips the diverse set."""
    path.unlink(missing_ok=True)  # never leave stale outputs behind
    try:
        outputs, info = record_baseline(workload)
    except Exception:
        return {"status": "error", "error": traceback.format_exc()[-2000:]}
    if outputs:
        torch.save(outputs, path)
    return info


def messages(baseline: dict[str, Any]) -> list[str]:
    """``analyze`` log lines about the diverse input set."""
    info = baseline.get("diverse") or {}
    if info.get("status") == "error":
        return [f"WARNING: the diverse input set failed: {_error_line(info.get('error', ''))}"]
    if info.get("status") != "ok":
        return []
    rows = info.get("inputs") or []
    failed = [r["label"] for r in rows if r.get("error")]
    times = [r["ms"] for r in rows if isinstance(r.get("ms"), int | float)]
    out = [
        f"analyze: diverse input set: {len(times)} of {len(rows)} inputs recorded"
        + (f" ({min(times):,.1f} .. {max(times):,.1f} ms)" if times else "")
    ]
    if failed:
        out.append(f"WARNING: diverse inputs that failed on the baseline: {', '.join(failed)}")
    return out


def summary_lines(baseline: dict[str, Any]) -> list[str]:
    """Bullets for the ``End-to-end quality check`` section of ``profile/summary.md``."""
    info = baseline.get("diverse") or {}
    if info.get("status") != "ok":
        return []
    labels = ", ".join(str(r.get("label")) for r in info.get("inputs") or [])
    return [
        f"* every candidate that passes is also run on a **diverse input set** ({labels}): "
        "its output is judged like the main one's, and its time gives per-input speedups "
        "(median, min, max). A speedup that changes with the content (speculative "
        "decoding, early exit) is welcome: it is labelled *data-dependent* and the report "
        "shows the set's median next to the benchmark number."
    ]


# ------------------------------------------------------------------ e2e


def check(
    workload: Workload,
    reference: dict[str, Any] | None,
    baseline: dict[str, Any],
    *,
    chaotic: bool,
    warmup: int = WARMUP,
    iters: int = ITERS,
) -> dict[str, Any]:
    """``metrics.diverse`` of an ``e2e`` run: every input the baseline recorded, run and
    timed as in ``analyze`` and judged against the baseline's output (``reference``: label
    -> output; None: timed only) by :func:`~kernel_agent.workloads.quality.assess`.
    ``{"passed", "reason", ...}`` plus the speedups of :func:`summarize`; a quality failure
    or a run that raises fails the candidate."""
    from kernel_agent.workloads.quality import assess

    info = baseline.get("diverse") or {}
    if info.get("status") != "ok":
        why = info.get("reason") or "analyze predates the diverse input set: re-run analyze"
        return {"passed": True, "reason": "", "skipped": str(why)}
    declared = workload.diverse_inputs()
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for rec in info.get("inputs") or []:
        label = str(rec.get("label"))
        options = declared.get(label)
        row: dict[str, Any] = {"label": label}
        if options is None or key_of(options) != rec.get("key"):
            rows.append({**row, "skipped": "the workload's input changed since analyze"})
            continue
        if not isinstance(rec.get("ms"), int | float) or rec["ms"] <= 0:
            rows.append({**row, "skipped": "no baseline (it failed in analyze)"})
            continue
        try:
            inputs, output, timing = run_input(workload, options, warmup=warmup, iters=iters)
            ref = (reference or {}).get(label)
            if ref is not None:
                with workload.with_options(options):
                    verdict = assess(workload, inputs, ref, output, chaotic=chaotic)
                row["quality"] = {"passed": verdict["passed"], "reason": verdict["reason"]}
                if not verdict["passed"]:
                    problems.append(f"{label}: {verdict['reason'] or 'quality check failed'}")
        except Exception as exc:
            row["error"] = traceback.format_exc()[-1500:]
            problems.append(f"{label} failed: {type(exc).__name__}: {exc}"[:400])
            rows.append(row)
            continue
        row.update(
            ms=timing["ms"],
            baseline_ms=rec["ms"],
            speedup=round(rec["ms"] / timing["ms"], 4) if timing["ms"] > 0 else None,
            spread=timing["spread"],
            baseline_spread=rec.get("spread"),
        )
        if "decode_stats" in timing:
            row["decode_stats"] = timing["decode_stats"]
        rows.append(row)
    return {"passed": not problems, "reason": "; ".join(problems), **summarize(rows)}
