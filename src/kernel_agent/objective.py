"""What a run optimises: the workload metric that ``measure()`` times (``-o metric=``).

* ``latency`` (default): wall-clock time of one end-to-end run.
* ``ttfa``: time to first audio of a streaming TTS run, from the call of
  ``workload.run`` to the first audio chunk (GPU-synchronised). The workload marks
  every chunk the moment it arrives (:meth:`~kernel_agent.workloads.base.Workload.mark_chunk`);
  ``metric_detail`` also reports the steady state: the median latency of the next
  ``steady_chunks`` chunks (default :data:`STEADY_CHUNKS`) and their real-time factor
  (chunk latency / chunk audio duration; below 1 the stream keeps up with playback),
  and the full streamed run. Quality is judged on the full streamed output.
* ``throughput``: reserved for issue #74 (a rate, higher is better); no workload
  implements it yet. A new metric adds an entry here and a branch in
  :meth:`~kernel_agent.workloads.base.Workload.metric_value`.

The value is ``median_ms`` everywhere (``baseline.json``, ``e2e`` results, A/B rounds,
the integration): the optimiser minimises it and every speedup divides it.
``baseline.json`` records the ``metric``; reports, charts, ``status`` and ``watch``
name it. A workload lists the metrics it can time in ``Workload.metrics``.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import Any

LATENCY = "latency"
TTFA = "ttfa"
THROUGHPUT = "throughput"
DEFAULT = LATENCY
#: ``metric=ttfa``: chunks after the first whose latency gives the steady state.
STEADY_CHUNKS = 8


@dataclass(frozen=True)
class Metric:
    name: str
    title: str  # headings: "End-to-end latency"
    label: str  # axes and prose: "end-to-end latency per run"
    short: str  # table columns: "latency"
    per: str  # after a value in ms: "per run"
    implemented: bool = True


METRICS: dict[str, Metric] = {
    LATENCY: Metric(
        LATENCY, "End-to-end latency", "end-to-end latency per run", "latency", "per run"
    ),
    TTFA: Metric(TTFA, "Time to first audio", "time to first audio", "TTFA", "to first audio"),
    THROUGHPUT: Metric(
        THROUGHPUT, "Throughput", "throughput", "throughput", "per item", implemented=False
    ),
}


def get(name: str | None) -> Metric:
    """The metric ``name`` (``None``: the default); unknown names raise ``ValueError``."""
    key = str(name or DEFAULT).lower()
    if key not in METRICS:
        raise ValueError(f"unknown metric {name!r}; choose one of: {', '.join(METRICS)}")
    return METRICS[key]


def of(baseline: dict[str, Any] | None) -> Metric:
    """The metric a run's ``baseline.json`` (or an ``analyze`` result) was measured with;
    runs from before metrics existed time the end-to-end latency."""
    name = str((baseline or {}).get("metric") or DEFAULT)
    return METRICS.get(name.lower()) or Metric(name, name, name, name, name)


def check(name: str, supported: tuple[str, ...], workload: str) -> str:
    """``name`` when ``workload`` (a class name) can time it, else a ``ValueError``
    that says what to do."""
    metric = get(name)
    if not metric.implemented:
        raise ValueError(f"metric={metric.name} is not implemented yet (issue #74)")
    if metric.name not in supported:
        raise ValueError(
            f"{workload} cannot time metric={metric.name} (it supports: {', '.join(supported)}); "
            "a harness declares the metrics it implements in `metrics` (see workloads/base.py)"
        )
    return metric.name


def aggregate(details: list[dict[str, Any]]) -> dict[str, Any]:
    """``metric_detail`` of ``measure()``: the median of every number over the timed runs
    (``None`` where a run had none)."""
    out: dict[str, Any] = {}
    for key in dict.fromkeys(k for d in details for k in d):
        values = [d[key] for d in details if isinstance(d.get(key), int | float)]
        out[key] = round(statistics.median(values), 4) if values else None
    return out


def detail_text(detail: dict[str, Any] | None) -> str:
    """One phrase on a ``metric=ttfa`` ``metric_detail``: the steady state and the full run."""
    if not detail:
        return ""
    parts = []
    if detail.get("chunk_ms") is not None:
        rtf = f", RTF {detail['rtf']:.3f}" if detail.get("rtf") is not None else ""
        parts.append(
            f"steady state {detail['chunk_ms']:,.1f} ms per chunk{rtf} "
            f"(next {detail.get('steady_chunks') or 0:.0f} chunks)"
        )
    if detail.get("run_ms") is not None:
        chunks = f", {detail['chunks']:.0f} chunks" if detail.get("chunks") else ""
        parts.append(f"full streamed run {detail['run_ms']:,.1f} ms{chunks}")
    return "; ".join(parts)


def describe(baseline: dict[str, Any] | None) -> str | None:
    """One line on a non-default metric of ``baseline`` (``None`` for the latency)."""
    metric = of(baseline)
    if metric.name == LATENCY:
        return None
    detail = detail_text((baseline or {}).get("metric_detail"))
    return f"metric: {metric.label} (-o metric={metric.name})" + (f"; {detail}" if detail else "")


def summary_section(baseline: dict[str, Any]) -> str:
    """Markdown for ``profile/summary.md``: what the optimiser minimises (empty for latency)."""
    metric = of(baseline)
    if metric.name == LATENCY:
        return ""
    lines = [
        "",
        f"## Objective: {metric.label}",
        "",
        f"* the run optimises **{metric.label}** (`-o metric={metric.name}`), not the latency "
        "of the whole run: the baseline, every end-to-end evaluation, the A/B rounds of the "
        f"integration and every speedup use it ({_fmt_ms(baseline.get('median_ms'))} at "
        "baseline).",
    ]
    if metric.name == TTFA:
        lines += [
            "* time to first audio = from the request to the first audio chunk of the "
            "streaming path (GPU-synchronised): text tokenisation, the prompt prefill, the "
            "first generated patch and its streaming decode. Work after the first chunk "
            "(later patches, their LM steps) does not count toward it.",
            "* the profile above covers that window only (`run` stops after the first "
            "chunk); kernel captures and calls per run still cover the full streamed run.",
            "* quality is judged on the full streamed output, as for the non-streaming run "
            "(teacher forcing on the latents, the audio of every chunk decoded by the "
            "stateful streaming decoder).",
        ]
    if text := detail_text(baseline.get("metric_detail")):
        lines.append(f"* baseline: {text} (reported, not optimised).")
    return "\n".join(lines) + "\n"


def _fmt_ms(value: Any) -> str:
    return f"{value:,.1f} ms" if isinstance(value, int | float) else "—"


def as_dict(metric: Metric) -> dict[str, Any]:
    """The metric for JSON (``watch``)."""
    return asdict(metric)
