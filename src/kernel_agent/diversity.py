"""The verdict on a candidate's per-input speedups over the diverse input set, and how
reports show it (#170). Plain data, no torch: the ledger, ``status`` and the report read
it; :mod:`kernel_agent.workloads.diverse` measures it.

A candidate is **data-dependent** when the spread of its per-input speedups,
(max - min) / median, exceeds both :data:`SPREAD_FLOOR` and :data:`NOISE_FACTOR` x the
timing noise (the largest run-to-run spread of an input, the candidate's plus the
baseline's), or when its decode steps per token (the counters a run reports,
``Workload.report_stats``) change with the input: speculative decoding, an early exit, a
cache of repeated content. It is labelled, never rejected.
"""

from __future__ import annotations

import statistics
from typing import Any

#: A spread of the per-input speedups up to this much is never data-dependent.
SPREAD_FLOOR = 0.10
#: ... nor one within this many times the timing noise.
NOISE_FACTOR = 3.0
#: The ledger flag (``results.tsv`` ``flags``) of a data-dependent candidate.
DATA_DEPENDENT = "data_dependent"


def step_rate(stats: dict[str, Any] | None) -> float | None:
    """Decode steps per emitted token of an input's counters (``steps`` alone without
    ``tokens``); None when the run reported no steps."""
    stats = stats or {}
    steps, tokens = stats.get("steps"), stats.get("tokens")
    if not isinstance(steps, int | float) or steps <= 0:
        return None
    if isinstance(tokens, int | float) and tokens > 0:
        return float(steps) / float(tokens)
    return float(steps)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Median, min and max of the per-input ``speedup`` of ``rows`` (with each row's
    ``spread``, ``baseline_spread`` and ``decode_stats``) and whether they make the
    candidate data-dependent, with the ``why``."""
    out: dict[str, Any] = {"inputs": rows}
    measured = [r for r in rows if isinstance(r.get("speedup"), int | float)]
    if len(measured) < 2:
        return {"status": "skipped", "why": f"{len(measured)} of {len(rows)} inputs timed", **out}
    median = statistics.median(float(r["speedup"]) for r in measured)
    lo = min(measured, key=lambda r: r["speedup"])
    hi = max(measured, key=lambda r: r["speedup"])
    spread = (hi["speedup"] - lo["speedup"]) / median if median > 0 else 0.0
    noise = max(
        float(r.get("spread") or 0.0) + float(r.get("baseline_spread") or 0.0) for r in measured
    )
    threshold = max(SPREAD_FLOOR, NOISE_FACTOR * noise)
    rates = [step_rate(r.get("decode_stats")) for r in measured]
    known = [x for x in rates if x is not None]
    steps_vary = len(known) == len(rates) and max(known) - min(known) > 1e-6 * max(known)
    why = []
    if spread > threshold:
        why.append(
            f"speedup {lo['speedup']:.2f}x ({lo['label']}) to {hi['speedup']:.2f}x "
            f"({hi['label']}) over {len(measured)} inputs: spread {spread:.0%} > "
            f"{threshold:.0%} (timing noise {noise:.1%})"
        )
    if steps_vary:
        why.append(
            f"decode steps per token change with the input ({min(known):.3g} .. {max(known):.3g})"
        )
    return {
        "status": "ok",
        "median_speedup": round(median, 4),
        "min_speedup": lo["speedup"],
        "max_speedup": hi["speedup"],
        "min_input": lo["label"],
        "max_input": hi["label"],
        "spread": round(spread, 4),
        "noise": round(noise, 4),
        "threshold": round(threshold, 4),
        "steps_vary": steps_vary,
        DATA_DEPENDENT: bool(why),
        "why": "; ".join(why),
        **out,
    }


# ------------------------------------------------------------------ reading results


def of(result: dict[str, Any] | None) -> dict[str, Any] | None:
    """The timed ``metrics.diverse`` of an ``e2e`` result (None: skipped or absent)."""
    info = ((result or {}).get("metrics") or {}).get("diverse")
    return info if isinstance(info, dict) and info.get("status") == "ok" else None


def is_data_dependent(result: dict[str, Any] | None) -> bool:
    return bool((of(result) or {}).get(DATA_DEPENDENT))


def flags(result: dict[str, Any] | None) -> str:
    """``results.tsv`` ``flags`` of an ``e2e`` result."""
    return DATA_DEPENDENT if is_data_dependent(result) else ""


def median_speedup(result: dict[str, Any] | None) -> float | None:
    value = (of(result) or {}).get("median_speedup")
    return float(value) if isinstance(value, int | float) else None


def compact(result: dict[str, Any] | None) -> dict[str, Any] | None:
    """The speedups of a ``metrics.diverse`` without the per-input rows (integration.json)."""
    info = of(result)
    if info is None:
        return None
    keys = ("median_speedup", "min_speedup", "max_speedup", "min_input", "max_input", "spread")
    return {k: info.get(k) for k in (*keys, DATA_DEPENDENT, "why")}


def headline(result: dict[str, Any] | None) -> str:
    """``diverse-set median 21.30x (15.10x .. 58.70x), data-dependent`` ("" if none)."""
    info = of(result)
    if info is None:
        return ""
    text = (
        f"diverse-set median {info['median_speedup']:.2f}x "
        f"({info['min_speedup']:.2f}x .. {info['max_speedup']:.2f}x)"
    )
    return text + (", data-dependent" if info.get(DATA_DEPENDENT) else "")


def summary_text(result: dict[str, Any]) -> str:
    """One phrase for reports: the ``metrics.diverse`` of an ``e2e`` run."""
    if result.get("skipped"):
        return f"diverse inputs skipped ({result['skipped']})"
    if not result.get("passed"):
        return f"diverse inputs FAILED: {result.get('reason')}"
    judged = sum(isinstance(r.get("quality"), dict) for r in result.get("inputs") or [])
    return f"diverse inputs passed ({judged} judged)"


def stats_text(stats: dict[str, Any] | None) -> str:
    """One phrase on decode counters (``metric_detail.decode_stats``): ``acceptance 62 %
    (310 of 500 drafted), 4.10 tokens per verification, 0.36 steps per token`` ("" if
    none)."""
    if not stats:
        return ""
    parts = []
    if (rate := stats.get("acceptance_rate")) is not None:
        drafted = stats.get("drafted")
        what = (
            f" ({stats.get('accepted') or 0:,.0f} of {drafted:,.0f} drafted)"
            if isinstance(drafted, int | float)
            else ""
        )
        parts.append(f"acceptance {rate:.0%}{what}")
    if (per := stats.get("tokens_per_verify")) is not None:
        parts.append(f"{per:.2f} tokens per verification")
    if (rate := step_rate(stats)) is not None:
        per_token = isinstance(stats.get("tokens"), int | float) and stats["tokens"] > 0
        parts.append(f"{rate:.2f} steps per token" if per_token else f"{rate:,.0f} steps")
    return ", ".join(parts)


def _error_line(text: str) -> str:
    return (text.strip().splitlines() or [""])[-1][:300]


def report_lines(result: dict[str, Any] | None, title: str) -> list[str]:
    """Markdown under ``## title``: the decode counters of the benchmark input
    (``metric_detail.decode_stats``) and the per-input table of ``metrics.diverse``."""
    result = result or {}
    info = (result.get("metrics") or {}).get("diverse")
    bench = (result.get("metric_detail") or {}).get("decode_stats")
    if not isinstance(info, dict) and not bench:
        return []
    lines = ["", f"## {title}", ""]
    if bench:
        lines.append(f"* benchmark input: {stats_text(bench)}")
    if not isinstance(info, dict):
        return lines
    if info.get("status") != "ok":
        why = info.get("skipped") or info.get("why") or info.get("reason")
        return [*lines, f"* diverse input set not timed: {why}"]
    verdict = (
        f"**data-dependent**: {info['why']}"
        if info.get(DATA_DEPENDENT)
        else f"not data-dependent (spread {info['spread']:.0%} <= {info['threshold']:.0%})"
    )
    lines += [
        f"* {headline(result)}, against {result.get('speedup')}x on the benchmark input; {verdict}",
        "",
        "| input | baseline ms | ms | speedup | quality | decode counters |",
        "|---|---|---|---|---|---|",
    ]
    for row in info.get("inputs") or []:
        quality = row.get("quality") or {}
        judged = "—" if not quality else ("passed" if quality.get("passed") else "FAILED")
        if not isinstance(row.get("speedup"), int | float):
            what = row.get("skipped") or _error_line(str(row.get("error") or ""))
            lines.append(f"| {row.get('label')} | | | | {judged} | not timed: {what} |")
            continue
        lines.append(
            f"| {row['label']} | {row['baseline_ms']:,.1f} | {row['ms']:,.1f} | "
            f"{row['speedup']:.2f}x | {judged} | {stats_text(row.get('decode_stats'))} |"
        )
    return lines
