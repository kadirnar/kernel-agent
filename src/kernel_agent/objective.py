"""What a run optimises: the workload metric that ``measure()`` times (``-o metric=``).

* ``latency`` (default): wall-clock time of one end-to-end run.
* ``ttfa``: time to first audio of a streaming TTS run, from the call of
  ``workload.run`` to the first audio chunk (GPU-synchronised). The workload marks
  every chunk the moment it arrives (:meth:`~kernel_agent.workloads.base.Workload.mark_chunk`);
  ``metric_detail`` also reports the steady state: the median latency of the next
  ``steady_chunks`` chunks (default :data:`STEADY_CHUNKS`) and their real-time factor
  (chunk latency / chunk audio duration; below 1 the stream keeps up with playback),
  and the full streamed run. Quality is judged on the full streamed output. The timed
  requests stop at the first chunk (``Workload.metric_window``) but for one whole request
  per measurement, which gives the judged output and those details: a check costs
  requests up to their first chunk, not streamed runs.
* ``throughput``: seconds of audio generated per wall second by a batch of requests
  (VoxCPM: ``-o batch_size=N``, :mod:`kernel_agent.workloads.voxcpm_batch`). A rate is
  higher-is-better, so the value the optimiser sees is its reciprocal, the wall time per
  second of generated audio (``1000 / throughput`` ms): every consumer of ``median_ms``
  (A/B rounds, ledger, charts, reports) keeps "lower is better", and every speedup
  ``base_ms / new_ms`` is exactly ``new_throughput / base_throughput``. ``metric_detail``
  holds the throughput itself and the per-request latency.

A new metric adds an entry here and a branch in
:meth:`~kernel_agent.workloads.base.Workload.metric_value`.

The value is ``median_ms`` everywhere (``baseline.json``, ``e2e`` results, A/B rounds,
the integration): the optimiser minimises it and every speedup divides it.
``baseline.json`` records the ``metric``; reports, charts, ``status`` and ``watch``
name it. A workload lists the metrics it can time in ``Workload.metrics``.

A module-level estimate is in ms per run of the workload (a kernel's ``est_saved_ms``:
its gain per call × its calls per run): :func:`from_run` puts it in the metric's ms
wherever it meets the metric (the projections, ``status``, the charts, the report).
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
        THROUGHPUT,
        "Time per second of audio",
        "wall time per second of generated audio",
        "time per audio s",
        "per second of generated audio",
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
        raise ValueError(f"metric={metric.name} is not implemented yet")
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
    """One phrase on a ``metric_detail``: ``ttfa``, the steady state and the full run;
    ``throughput``, the rate and the per-request latency."""
    if not detail:
        return ""
    parts = []
    if detail.get("throughput") is not None:
        requests = f"{detail['requests']:.0f} requests, " if detail.get("requests") else ""
        parts.append(
            f"throughput {detail['throughput']:,.2f} s of audio per second ({requests}"
            f"{detail.get('audio_s') or 0:,.2f} s of audio in {detail.get('run_ms') or 0:,.0f} ms)"
        )
        if detail.get("request_ms") is not None:
            worst = detail.get("request_ms_max")
            most = f" (median; max {worst:,.0f} ms)" if worst is not None else ""
            parts.append(f"latency per request {detail['request_ms']:,.0f} ms{most}")
        return "; ".join(parts)
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


def profile_window(timing: dict[str, Any]) -> tuple[float, str, str]:
    """``(ms, what, per)`` of the window ``analyze`` profiles, for the profile summary: the
    metric's own (``timing`` is a ``measure()`` result or ``baseline.json``), except for
    ``throughput``, whose value is a rate: there the whole batched run."""
    metric = of(timing)
    run_ms = (timing.get("metric_detail") or {}).get("run_ms")
    if metric.name == THROUGHPUT and isinstance(run_ms, int | float):
        return float(run_ms), "batched run (every request)", "per batched run"
    return float(timing.get("median_ms") or 0.0), metric.title.lower(), metric.per


def run_factor(baseline: dict[str, Any] | None) -> float | None:
    """The metric's ms per ms of one run of the workload, for :func:`from_run`.

    ``latency``: 1. ``throughput``: a batched run makes ``audio_s`` seconds of audio
    (``metric_detail``), so a run's ms is ``1 / audio_s`` ms per second of audio (else
    ``median_ms / run_ms``). None for ``ttfa`` (only a module's calls inside the
    first-audio window count: a share of its own, ``window``) and when ``baseline.json``
    lacks the details."""
    metric = of(baseline)
    if metric.name == LATENCY:
        return 1.0
    if metric.name != THROUGHPUT:
        return None
    detail = (baseline or {}).get("metric_detail") or {}
    audio_s, run_ms = detail.get("audio_s"), detail.get("run_ms")
    if isinstance(audio_s, int | float) and audio_s > 0:
        return 1.0 / float(audio_s)
    value = (baseline or {}).get("median_ms")
    if isinstance(run_ms, int | float) and run_ms > 0 and isinstance(value, int | float):
        return float(value) / float(run_ms)
    return None


def from_run(
    ms: float | None, baseline: dict[str, Any] | None, window: float | None = None
) -> float | None:
    """A module-level estimate, ``ms`` per run of the workload (a kernel's ``est_saved_ms``:
    its gain per call × its calls per run), in the metric's ms: × :func:`run_factor`
    (``throughput``: per second of generated audio). ``ttfa``: the estimate covers the
    calls of a full streamed run, of which only those inside the first-audio window
    count; ``window`` is the estimate's share for them (the gain per call × the calls in
    the window, :func:`kernel_agent.projection.units`). None when unknown (no ``window``,
    no ``audio_s``): such an estimate is not projected."""
    if ms is None:
        return None
    if of(baseline).name == TTFA:
        return None if window is None else ms * window
    factor = run_factor(baseline)
    return None if factor is None else ms * factor


def unknown_why(baseline: dict[str, Any] | None) -> str:
    """Why :func:`from_run` has no value for a module-level estimate of this run."""
    metric = of(baseline)
    if metric.name == TTFA:
        return "its calls inside the first-audio window are unknown"
    if metric.name == THROUGHPUT:
        return "baseline.json has no seconds of audio per run"
    return f"no conversion to {metric.label}"


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
            "* every measurement (`evaluate_e2e`, the A/B rounds, the diverse set) times "
            "several requests that stop at their first chunk (the stream is closed there, as "
            "a client that stops listening closes it) and one whole request whose output is "
            "judged: a transform must leave the model ready for the next request after a "
            "stream closed early (no state that only the end of a run resets).",
        ]
    if metric.name == THROUGHPUT:
        lines += [
            "* throughput = seconds of audio generated per wall second by a batch of "
            "different requests decoded together; the value above (and every ms of this "
            "run) is its reciprocal, wall time per second of generated audio, so lower is "
            "better and a speedup is the throughput ratio.",
            "* the batch shares every weight read: decode steps run at batch N (the LocDiT "
            "at 2N with CFG), so kernels and transforms must handle batch N (or fall back); "
            "quality is judged per request, and one wrong request fails the candidate.",
        ]
    if text := detail_text(baseline.get("metric_detail")):
        lines.append(f"* baseline: {text} (reported, not optimised).")
    return "\n".join(lines) + "\n"


def _fmt_ms(value: Any) -> str:
    return f"{value:,.1f} ms" if isinstance(value, int | float) else "—"


def as_dict(metric: Metric) -> dict[str, Any]:
    """The metric for JSON (``watch``)."""
    return asdict(metric)
