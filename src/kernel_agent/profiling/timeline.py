"""The GPU timeline of one profiled run: streams, stages, idle gaps (issue #146).

The kernel view (:func:`~kernel_agent.profiling.profiler.kernel_profile`) used to sum
kernel times, so two streams that overlap exceed the wall time and the view was marked
unreliable. This module reads the run's Kineto events (``torch.profiler``) instead:

* **Busy**: ``busy_ms`` is the *union* of the GPU intervals (kernels, copies, memsets)
  over every stream; ``kernel_time_ms`` the sum (the old number); ``overlap_ms`` the
  difference; ``streams`` the busy time per stream.
* **Stages**: :class:`StageRanges` puts a ``record_function("ka::<stage>")`` range around
  every entrypoint call of the workload's roots and their direct children (``forward``
  and the entrypoints of :mod:`.methods`: ``forward_step``, ``decode``, ... plus the
  workload's declared ones), whatever model it is. Every GPU event belongs to the
  innermost range around the CPU call that launched it (its correlation id; a CUDA-graph
  replay's kernels carry the replay's id). Per stage: GPU time, busy time, host lead
  (GPU start − launch), graph-launched events, SM fill and achieved occupancy when the
  trace has grid sizes.
* **Gaps**: every idle interval between consecutive GPU work, binned (< 2, 2–10, 10–50,
  50–500, > 500 us); those of at least :data:`MIN_GAP_US` are attributed to a cause and
  to the stages around them. Causes, in order: ``host sync`` (the work before was a
  device-to-host copy, or a synchronising call was waiting when the GPU went idle:
  ``.cpu()``, ``.item()``, ``synchronize()``), ``pageable copy`` (a copy from or to
  pageable host memory, which synchronises), ``host late`` (the next work was launched
  only after the GPU went idle), ``graph launch`` (launched early by a CUDA-graph replay:
  its launch latency), ``launch latency`` (launched early otherwise: kernel boundaries,
  stream waits), ``unattributed`` (no launch call found).
* **Critical path**: :class:`StageRanges` also records each stage call's tensor storages
  (inputs and outputs). A call depends on the latest earlier call that output one of its
  input storages; a floating-point input no earlier call saw (made by glue code between
  stages, e.g. a ``torch.cat`` of stage outputs) is assumed to depend on the call just
  before it. The longest path through these edges, weighted by each call's GPU busy
  time, is the critical path; the work off it could overlap on another stream
  (``overlap_potential_ms``). Approximate: state passed outside the arguments (a KV
  cache held by a module, a buffer written in place) adds no edge.
"""

from __future__ import annotations

import bisect
import collections
import contextlib
import json
import math
import statistics
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

GPU_KINDS = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})
API_KINDS = frozenset({"cuda_runtime", "cuda_driver"})
ANNOTATION = "user_annotation"
#: Prefix of the profiler ranges this module reads; ``ka::run`` brackets the whole run.
PREFIX = "ka::"
RUN = f"{PREFIX}run"
#: Gaps below this are kernel boundaries: binned, not attributed.
MIN_GAP_US = 2.0
BINS = ((2.0, "< 2 us"), (10.0, "2-10 us"), (50.0, "10-50 us"), (500.0, "50-500 us"))
OVER = "> 500 us"
#: Host API calls that block until the GPU (or a stream / event) is done.
SYNC_CALLS = frozenset(
    {
        "cudaDeviceSynchronize",
        "cudaStreamSynchronize",
        "cudaEventSynchronize",
        "cudaMemcpy",
        "cudaMemcpy2D",
        "cuCtxSynchronize",
        "cuStreamSynchronize",
        "cuEventSynchronize",
        "cuMemcpyDtoH",
        "cuMemcpyDtoH_v2",
    }
)
CAUSES = (
    "host sync",
    "pageable copy",
    "host late",
    "graph launch",
    "launch latency",
    "unattributed",
)
OTHER = "(no stage)"  # GPU work launched inside the run but outside every stage range
OUTSIDE = "(outside the run)"
#: Glue inputs (see the module docstring) smaller than this are host-made (positions,
#: masks, step counters), not the output of GPU work between stages.
GLUE_MIN_NUMEL = 64


@dataclass
class Event:
    """One profiler event; times in ns. ``stream``: the stream of a GPU event, the thread
    of a CPU one. ``meta``: kernel launch parameters when the trace has them."""

    kind: str
    name: str
    start: int
    end: int
    correlation: int = -1
    stream: int = -1
    meta: dict[str, Any] | None = None


_KERNEL_META = (
    "grid",
    "block",
    "registers per thread",
    "shared memory",
    "est. achieved occupancy %",
)


def _meta(args: Mapping[str, Any]) -> dict[str, Any] | None:
    meta = {k: args[k] for k in _KERNEL_META if args.get(k) is not None}
    return meta or None


def from_chrome_trace(trace: Mapping[str, Any]) -> list[Event]:
    """The events of a Kineto chrome trace (``prof.export_chrome_trace``, loaded)."""
    keep = GPU_KINDS | API_KINDS | {ANNOTATION}
    out = []
    for ev in trace.get("traceEvents") or []:
        kind = ev.get("cat")
        if kind not in keep or ev.get("ph") != "X":
            continue
        name = str(ev.get("name", ""))
        if kind == ANNOTATION and not name.startswith(PREFIX):
            continue
        args = ev.get("args") or {}
        start = round(float(ev["ts"]) * 1000)
        end = start + round(float(ev.get("dur", 0)) * 1000)
        gpu = kind in GPU_KINDS
        where = args.get("stream", ev.get("tid")) if gpu else ev.get("tid")
        out.append(
            Event(
                kind,
                name,
                start,
                end,
                int(args.get("correlation", -1)),
                int(where) if isinstance(where, int | float) else hash(where),
                _meta(args) if kind == "kernel" else None,
            )
        )
    return out


def _call(obj: Any, name: str, default: Any = None) -> Any:
    method = getattr(obj, name, None)
    if method is None:
        return default
    try:
        return method()
    except Exception:
        return default


def activity_type(e: Any) -> str:
    """The Kineto activity type of a ``_KinetoEvent`` (``kernel``, ``gpu_memcpy``,
    ``cuda_runtime``, ``user_annotation``, ...; the ``cat`` of its chrome-trace event).

    ``_KinetoEvent.activity_type()`` is new in recent torch (2.14 has it, 2.10 does not):
    without it the type is read from what every version has, as the chrome trace of torch
    2.10 labels the events (measured on an NVIDIA A10): a user annotation (a
    ``record_function`` range) is ``user_annotation`` on the CPU and
    ``gpu_user_annotation`` on the GPU; GPU work is ``gpu_memcpy`` (``Memcpy ...``),
    ``gpu_memset`` (``Memset ...``) or else ``kernel``; a CPU event named ``cuda*`` is a
    runtime call (``cudaLaunchKernel``, ``cudaGraphLaunch``), ``cu`` + a capital a driver
    call (``cuLaunchKernelEx``, the launch of Triton and cuda.core), anything else
    ``cpu_op``."""
    method = getattr(e, "activity_type", None)
    if method is not None:
        return str(method())
    name = str(_call(e, "name", ""))
    on_gpu = str(_call(e, "device_type", "")).rsplit(".", 1)[-1] == "CUDA"
    if _call(e, "is_user_annotation", False):
        return "gpu_user_annotation" if on_gpu else ANNOTATION
    if on_gpu and name.startswith("Memcpy"):
        return "gpu_memcpy"
    if on_gpu and name.startswith("Memset"):
        return "gpu_memset"
    if on_gpu:
        return "kernel"
    if name.startswith("cuda"):
        return "cuda_runtime"
    if len(name) > 2 and name.startswith("cu") and name[2].isupper():
        return "cuda_driver"
    return "cpu_op"


def from_profiler(prof: Any) -> list[Event]:
    """The events of a finished ``torch.profiler.profile`` (its Kineto results, without
    exporting a chrome trace: an eager run of a large model has millions of events).
    Kernel grid sizes are only there when Kineto puts them in the event metadata."""
    keep = GPU_KINDS | API_KINDS | {ANNOTATION}
    out = []
    for e in prof.profiler.kineto_results.events():
        kind = activity_type(e)
        if kind not in keep:
            continue
        name = str(_call(e, "name", ""))
        if kind == ANNOTATION and not name.startswith(PREFIX):
            continue
        start = int(_call(e, "start_ns", 0))
        end = _call(e, "end_ns")
        end = int(end) if end is not None else start + int(_call(e, "duration_ns", 0))
        gpu = kind in GPU_KINDS
        where = _call(e, "device_resource_id", -1) if gpu else _call(e, "start_thread_id", -1)
        meta = None
        if kind == "kernel" and (raw := _call(e, "metadata_json", "")):
            with contextlib.suppress(ValueError, TypeError):
                meta = _meta(json.loads("{" + str(raw).strip().strip(",") + "}"))
        out.append(Event(kind, name, start, end, int(_call(e, "correlation_id", -1)), where, meta))
    return out


# ------------------------------------------------------------------ intervals


def union(intervals: Iterable[tuple[int, int]]) -> list[list[int]]:
    """Merged, sorted ``[start, end]`` of possibly overlapping intervals."""
    out: list[list[int]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _total(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(e - s for s, e in union(intervals))


def flatten(intervals: Iterable[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Properly nested ``(start, end, key)`` ranges → non-overlapping segments, each with
    the key of the innermost range that covers it."""
    segs: list[tuple[int, int, int]] = []
    stack: list[tuple[int, int, int]] = []
    cur = -math.inf
    for s, e, key in sorted(intervals, key=lambda x: (x[0], -x[1])):
        while stack and stack[-1][1] <= s:
            top = stack.pop()
            if cur < top[1]:
                segs.append((int(cur), top[1], top[2]))
            cur = max(cur, top[1])
        if stack and cur < s:
            segs.append((int(cur), s, stack[-1][2]))
        cur = max(cur, s)
        stack.append((s, e, key))
    while stack:
        top = stack.pop()
        if cur < top[1]:
            segs.append((int(cur), top[1], top[2]))
        cur = max(cur, top[1])
    return segs


def _bin(gap_us: float) -> str:
    return next((label for limit, label in BINS if gap_us < limit), OVER)


def _ms(ns: float) -> float:
    return round(ns / 1e6, 3)


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


# ------------------------------------------------------------------ analysis


@dataclass
class _Stage:
    calls: int = 0
    events: int = 0
    kernels: int = 0
    gpu_ns: int = 0
    intervals: list[tuple[int, int]] = field(default_factory=list)
    lead: list[float] = field(default_factory=list)  # us
    graph: int = 0
    fill: float = 0.0  # time-weighted
    occupancy: float = 0.0
    weighted_ns: int = 0
    occupancy_ns: int = 0


def _blocks(grid: Any) -> int | None:
    if isinstance(grid, list | tuple) and grid:
        n = 1
        for g in grid:
            n *= int(g)
        return n
    return int(grid) if isinstance(grid, int | float) else None


def _copy_kind(name: str) -> str:
    """``DtoH``, ``HtoD``, ``DtoD`` or ``""`` for a GPU event name."""
    return next((k for k in ("DtoH", "HtoD", "DtoD", "PtoP") if k in name), "")


def analyze(
    events: Iterable[Event],
    *,
    calls: Sequence[Mapping[str, Any]] | None = None,
    sms: int | None = None,
    top: int = 10,
) -> dict[str, Any]:
    """The timeline summary of one profiled run (JSON-able; times in ms unless named
    ``_us``). ``calls``: :attr:`StageRanges.calls` of the run (for the critical path);
    ``sms``: the GPU's SM count (for the SM fill of kernels with a grid size)."""
    events = list(events)
    gpu = sorted((e for e in events if e.kind in GPU_KINDS), key=lambda e: (e.start, e.end))
    if not gpu:
        raise ValueError("no GPU events in the trace")
    api = {e.correlation: e for e in events if e.kind in API_KINDS and e.correlation >= 0}
    syncs = sorted((e.start, e.end) for e in events if e.kind in API_KINDS and e.name in SYNC_CALLS)
    ranges = sorted(
        (e for e in events if e.kind == ANNOTATION and e.name.startswith(PREFIX)),
        key=lambda e: (e.start, -e.end),
    )
    run = next((e for e in ranges if e.name == RUN), None)
    instances = [e for e in ranges if e.name != RUN]
    labels = [e.name[len(PREFIX) :] for e in instances]
    segs = flatten((e.start, e.end, i) for i, e in enumerate(instances))
    seg_starts = [s for s, _, _ in segs]

    def instance_at(t: int) -> int | None:
        j = bisect.bisect_right(seg_starts, t) - 1
        return segs[j][2] if j >= 0 and segs[j][0] <= t < segs[j][1] else None

    window_start = run.start if run else min([gpu[0].start, *(e.start for e in api.values())])
    last_end = max(e.end for e in gpu)
    window_end = max(run.end, last_end) if run else last_end

    def stage_of(t: int) -> str:
        i = instance_at(t)
        if i is not None:
            return labels[i]
        return OTHER if run is None or run.start <= t <= run.end else OUTSIDE

    stages: dict[str, _Stage] = collections.defaultdict(_Stage)
    for label in labels:
        stages[label].calls += 1
    streams: dict[int, dict[str, Any]] = {}
    per_instance: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    lead_all: list[float] = []
    where: list[tuple[str, Event | None]] = []  # per GPU event: stage, launch call
    for e in gpu:
        launch = api.get(e.correlation)
        t = launch.start if launch is not None else e.start
        inst = instance_at(t)
        label = labels[inst] if inst is not None else stage_of(t)
        where.append((label, launch))
        if inst is not None:
            per_instance[inst].append((e.start, e.end))
        st = stages[label]
        st.events += 1
        st.gpu_ns += e.end - e.start
        st.intervals.append((e.start, e.end))
        if launch is not None:
            lead = (e.start - launch.start) / 1e3
            st.lead.append(lead)
            lead_all.append(lead)
            st.graph += "Graph" in launch.name
        s = streams.setdefault(
            e.stream, {"stream": e.stream, "events": 0, "kernels": 0, "iv": [], "ns": 0}
        )
        s["events"] += 1
        s["iv"].append((e.start, e.end))
        s["ns"] += e.end - e.start
        if e.kind == "kernel":
            st.kernels += 1
            s["kernels"] += 1
            meta = e.meta or {}
            blocks = _blocks(meta.get("grid"))
            if blocks and sms:
                st.fill += min(1.0, blocks / sms) * (e.end - e.start)
                st.weighted_ns += e.end - e.start
            occ = meta.get("est. achieved occupancy %")
            if occ is not None:
                with contextlib.suppress(TypeError, ValueError):
                    st.occupancy += float(occ) * (e.end - e.start)
                    st.occupancy_ns += e.end - e.start

    busy = union((e.start, e.end) for e in gpu)
    busy_ns = sum(e - s for s, e in busy)
    kernel_ns = sum(e.end - e.start for e in gpu)
    window_ns = max(window_end - window_start, 1)
    gaps = _gaps(gpu, where, syncs, window_start, top)
    out: dict[str, Any] = {
        "window_ms": _ms(window_ns),
        "busy_ms": _ms(busy_ns),
        "kernel_time_ms": _ms(kernel_ns),
        "overlap_ms": _ms(kernel_ns - busy_ns),
        "idle_ms": _ms(window_ns - busy_ns),
        "busy_fraction": round(busy_ns / window_ns, 4),
        "lead_in_ms": _ms(max(gpu[0].start - window_start, 0)),
        "tail_ms": _ms(max(window_end - last_end, 0)),
        "gpu_events": len(gpu),
        "kernels": sum(e.kind == "kernel" for e in gpu),
        "host_lead_ms": round(m / 1e3, 3) if (m := _median(lead_all)) is not None else None,
        "streams": sorted(
            (
                {
                    "stream": s["stream"],
                    "events": s["events"],
                    "kernels": s["kernels"],
                    "busy_ms": _ms(_total(s["iv"])),
                    "kernel_time_ms": _ms(s["ns"]),
                }
                for s in streams.values()
            ),
            key=lambda s: -s["busy_ms"],
        ),
        "gaps": gaps,
        "stages": sorted(
            (_stage_row(label, st) for label, st in stages.items() if st.events or st.calls),
            key=lambda r: (-r["gpu_ms"], r["stage"]),
        ),
    }
    if calls is not None:
        out["critical_path"] = critical_path(calls, labels, per_instance)
    return out


def _stage_row(label: str, st: _Stage) -> dict[str, Any]:
    lead = _median(st.lead)
    return {
        "stage": label,
        "calls": st.calls,
        "events": st.events,
        "kernels": st.kernels,
        "gpu_ms": _ms(st.gpu_ns),
        "busy_ms": _ms(_total(st.intervals)),
        "host_lead_us": round(lead, 1) if lead is not None else None,
        "graph_events": st.graph,
        "sm_fill": round(st.fill / st.weighted_ns, 3) if st.weighted_ns else None,
        "occupancy_pct": round(st.occupancy / st.occupancy_ns, 1) if st.occupancy_ns else None,
    }


def _cause(prev: Event, nxt: Event, launch: Event | None, gap_start: int, syncing: bool) -> str:
    if syncing or (prev.kind == "gpu_memcpy" and _copy_kind(prev.name) == "DtoH"):
        return "host sync"
    if any(e.kind == "gpu_memcpy" and "Pageable" in e.name for e in (prev, nxt)):
        return "pageable copy"
    if launch is None:
        return "unattributed"
    if launch.start > gap_start:
        return "host late"
    return "graph launch" if "Graph" in launch.name else "launch latency"


def _gaps(
    gpu: list[Event],
    where: list[tuple[str, Event | None]],
    syncs: list[tuple[int, int]],
    window_start: int,
    top: int,
) -> dict[str, Any]:
    """Idle intervals between consecutive GPU work (any stream): bins, causes, places."""
    bins = {label: [0, 0] for _, label in BINS} | {OVER: [0, 0]}
    causes = {c: [0, 0] for c in CAUSES}
    places: dict[tuple[str, str, str], list[Any]] = {}
    largest: list[tuple[int, int, str, str, str]] = []
    sync_i, sync_reach = 0, -math.inf  # sync calls started so far; the latest end among them
    end, prev = gpu[0].end, 0
    for i in range(1, len(gpu)):
        e = gpu[i]
        if e.start > end:
            gap = e.start - end
            entry = bins[_bin(gap / 1e3)]
            entry[0] += 1
            entry[1] += gap
            if gap >= MIN_GAP_US * 1e3:
                while sync_i < len(syncs) and syncs[sync_i][0] <= end:
                    sync_reach = max(sync_reach, syncs[sync_i][1])
                    sync_i += 1
                cause = _cause(gpu[prev], e, where[i][1], end, sync_reach >= end)
                before, after = where[prev][0], where[i][0]
                causes[cause][0] += 1
                causes[cause][1] += gap
                place = places.setdefault((cause, before, after), [0, 0, [], gpu[prev].name])
                place[0] += 1
                place[1] += gap
                place[2].append(gap / 1e3)
                largest.append(
                    (gap, end, cause, f"{before}: {gpu[prev].name}", f"{after}: {e.name}")
                )
        if e.end >= end:
            end, prev = e.end, i
    largest.sort(reverse=True)
    return {
        "min_attributed_us": MIN_GAP_US,
        "bins": [{"bin": k, "count": v[0], "ms": _ms(v[1])} for k, v in bins.items()],
        "causes": {c: {"count": v[0], "ms": _ms(v[1])} for c, v in causes.items() if v[0]},
        "by_place": [
            {
                "cause": cause,
                "before": before,
                "after": after,
                "count": v[0],
                "ms": _ms(v[1]),
                "median_us": round(statistics.median(v[2]), 1),
                "after_event": v[3][:100],
            }
            for (cause, before, after), v in sorted(places.items(), key=lambda kv: -kv[1][1])[:top]
        ],
        "largest": [
            {
                "us": round(gap / 1e3, 1),
                "at_ms": _ms(t - window_start),
                "cause": cause,
                "before": before[:120],
                "after": after[:120],
            }
            for gap, t, cause, before, after in largest[:top]
        ],
    }


# ------------------------------------------------------------------ dependencies


def critical_path(
    calls: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    per_instance: Mapping[int, list[tuple[int, int]]],
    top: int = 8,
) -> dict[str, Any]:
    """Longest producer → consumer chain through the leaf stage calls (the module
    docstring), weighted by each call's GPU busy time. ``calls`` (in issue order) are
    matched to the stage ranges of the trace (``labels``, in start order) per label."""
    by_label: dict[str, list[int]] = collections.defaultdict(list)
    for i, label in enumerate(labels):
        by_label[label].append(i)
    mine: dict[str, list[int]] = collections.defaultdict(list)
    for j, call in enumerate(calls):
        mine[str(call.get("label"))].append(j)
    if {k: len(v) for k, v in by_label.items()} != {k: len(v) for k, v in mine.items()}:
        return {"error": "the recorded stage calls and the trace's stage ranges disagree"}
    instance_of = {j: by_label[label][n] for label, js in mine.items() for n, j in enumerate(js)}
    parents = {int(c.get("parent", -1)) for c in calls}
    leaves = [j for j in range(len(calls)) if j not in parents]
    weight = {j: _total(per_instance.get(instance_of[j], [])) / 1e6 for j in leaves}
    writer: dict[int, int] = {}  # storage -> the latest leaf call that output it
    seen: set[int] = set()
    finish: dict[int, float] = {}
    best: dict[int, int | None] = {}
    edges = assumed = 0
    prev: int | None = None
    for j in leaves:
        call = calls[j]
        preds: set[int] = set()
        for ptr, numel, floating in call.get("inputs") or []:
            if ptr in writer:
                preds.add(writer[ptr])
            elif ptr not in seen and floating and numel >= GLUE_MIN_NUMEL and prev is not None:
                preds.add(prev)  # made between stages from earlier work: assume the last
                assumed += 1
        preds.discard(j)
        edges += len(preds)
        pred = max(preds, key=lambda p: finish[p], default=None)
        finish[j] = weight[j] + (finish[pred] if pred is not None else 0.0)
        best[j] = pred
        for ptr, *_ in call.get("inputs") or []:
            seen.add(ptr)
        for ptr in call.get("outputs") or []:
            writer[ptr] = j
            seen.add(ptr)
        prev = j
    if not finish:
        return {"error": "no stage calls"}
    end: int | None = max(finish, key=lambda j: finish[j])
    path: set[int] = set()
    while end is not None:
        path.add(end)
        end = best[end]
    total = sum(weight.values())
    crit = max(finish.values())
    off: dict[str, list[float]] = collections.defaultdict(lambda: [0, 0.0])
    on: dict[str, float] = collections.defaultdict(float)
    for j in leaves:
        label = str(calls[j].get("label"))
        if j in path:
            on[label] += weight[j]
        else:
            off[label][0] += 1
            off[label][1] += weight[j]
    return {
        "calls": len(leaves),
        "edges": edges,
        "assumed_edges": assumed,
        "stage_busy_ms": round(total, 3),
        "critical_path_ms": round(crit, 3),
        "overlap_potential_ms": round(total - crit, 3),
        "on_path": [
            {"stage": k, "ms": round(v, 3)} for k, v in sorted(on.items(), key=lambda kv: -kv[1])
        ][:top],
        "off_path": [
            {"stage": k, "calls": int(v[0]), "ms": round(v[1], 3)}
            for k, v in sorted(off.items(), key=lambda kv: -kv[1][1])
        ][:top],
    }


# ------------------------------------------------------------------ stage ranges


def _storages(values: Any, depth: int = 2) -> Iterator[tuple[int, int, bool]]:
    """``(storage pointer, numel, floating)`` of the tensors in ``values`` (tuples,
    lists and dict values ``depth`` levels deep)."""
    import torch

    if isinstance(values, torch.Tensor):
        with contextlib.suppress(Exception):
            yield values.untyped_storage().data_ptr(), values.numel(), values.is_floating_point()
        return
    if depth <= 0:
        return
    if isinstance(values, Mapping):
        values = list(values.values())
    if isinstance(values, tuple | list):
        for v in values:
            yield from _storages(v, depth - 1)


_CONTAINERS: tuple[type, ...] = ()


def _containers() -> tuple[type, ...]:
    global _CONTAINERS
    if not _CONTAINERS:
        from torch import nn

        _CONTAINERS = (nn.ModuleList, nn.ModuleDict, nn.ParameterList, nn.ParameterDict)
    return _CONTAINERS


def stage_modules(roots: Mapping[str, Any], depth: int = 1) -> dict[int, tuple[Any, str]]:
    """The modules whose calls are stages: every root and its descendants down to
    ``depth`` levels (a ``ModuleList`` / ``ModuleDict`` is not a level: its items are),
    never inside a ``torch.compile``'d module. ``{id: (module, label)}``, the label being
    the qualname with layer indices folded (``model.layers.*``)."""
    from kernel_agent.projection import fold

    found: dict[int, tuple[Any, str]] = {}

    def visit(module: Any, name: str, level: int) -> None:
        if id(module) in found:
            return
        children = list(module.named_children())
        if isinstance(module, _containers()):  # not a level: its items are
            for child_name, child in children:
                visit(child, f"{name}.{child_name}", level)
            return
        found[id(module)] = (module, fold(name))
        if level >= depth or getattr(module, "_orig_mod", None) is not None:
            return  # deep enough, or compiled: its insides run traced
        for child_name, child in children:
            visit(child, f"{name}.{child_name}", level + 1)

    for root_name, root in roots.items():
        visit(root, root_name, 0)
    return found


class StageRanges:
    """``record_function("ka::<stage>")`` around every entrypoint call of the stage
    modules (:func:`stage_modules`; entrypoints: ``forward`` through hooks, the methods in
    ``methods`` through instance wrappers, as :mod:`.methods`), plus a record of each call
    (``calls``: label, enclosing call, input and output storages) for the critical path.

    Calls traced by Dynamo do nothing (the hooks return at once when
    ``torch.compiler.is_compiling()``, so a compiled region still traces the same graph);
    installed hooks may make compiled code recompile once, which is why the kernel view
    installs them before its warm-up run. Only the outermost entrypoint call of an
    instance is a range (``forward_step`` calling ``self(x)`` is one stage call)."""

    def __init__(
        self,
        roots: Mapping[str, Any],
        methods: Mapping[type, Sequence[str]] | None = None,
        *,
        depth: int = 1,
    ) -> None:
        self.modules = stage_modules(roots, depth)
        self.methods = methods or {}
        self.calls: list[dict[str, Any]] = []
        self._open: list[tuple[int, str, Any, int]] = []  # (module id, method, range, call)
        self._depth: collections.Counter[int] = collections.Counter()
        self._ctx: contextlib.ExitStack | None = None

    @classmethod
    def of(cls, workload: Any) -> StageRanges | None:
        """The stages of ``workload`` (its roots, entrypoints and ``stage_depth`` option,
        default 1); None when it has no roots."""
        from kernel_agent.profiling.methods import discover_entrypoints, workload_entrypoints

        try:
            roots = workload.roots()
            methods = discover_entrypoints(roots, workload_entrypoints(workload))
            options = getattr(workload, "options", None) or {}
            depth = int(options.get("stage_depth", 1))
        except Exception:
            return None
        return cls(roots, methods, depth=depth) if roots else None

    def reset(self) -> None:
        self.calls = []

    def _enter(self, module: Any, method: str, args: Any, kwargs: Any) -> None:
        import torch

        key = id(module)
        self._depth[key] += 1
        if self._depth[key] > 1:  # an entrypoint of the same instance inside its own call
            return
        label = self.modules[key][1] + ("" if method == "forward" else f".{method}")
        rf = torch.autograd.profiler.record_function(PREFIX + label)
        rf.__enter__()
        parent = self._open[-1][3] if self._open else -1
        self.calls.append(
            {
                "label": label,
                "parent": parent,
                "inputs": list(_storages((args, kwargs), 3)),
                "outputs": [],
            }
        )
        self._open.append((key, method, rf, len(self.calls) - 1))

    def _exit(self, module: Any, method: str, output: Any) -> None:
        key = id(module)
        self._depth[key] = max(self._depth[key] - 1, 0)
        if self._depth[key]:
            return
        for k in range(len(self._open) - 1, -1, -1):
            if self._open[k][:2] == (key, method):
                _, _, rf, index = self._open[k]
                del self._open[k:]  # calls opened above it never returned
                self.calls[index]["outputs"] = [p for p, _, _ in _storages(output)]
                rf.__exit__(None, None, None)
                return

    def _hooks(self) -> tuple[Callable[..., None], Callable[..., None]]:
        import torch

        def pre(module: Any, args: Any, kwargs: Any) -> None:
            if torch.compiler.is_compiling():
                return
            self._enter(module, "forward", args, kwargs)

        def post(module: Any, args: Any, kwargs: Any, output: Any) -> None:
            if torch.compiler.is_compiling():
                return
            self._exit(module, "forward", output)

        return pre, post

    def _wrap(self, module: Any, method: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        import functools

        import torch

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if torch.compiler.is_compiling():
                return fn(*args, **kwargs)
            self._enter(module, method, args, kwargs)
            output = fn(*args, **kwargs)
            self._exit(module, method, output)
            return output

        return wrapper

    def __enter__(self) -> StageRanges:
        missing = object()
        with contextlib.ExitStack() as stack:
            pre, post = self._hooks()
            for module, _ in self.modules.values():
                stack.callback(module.register_forward_pre_hook(pre, with_kwargs=True).remove)
                stack.callback(module.register_forward_hook(post, with_kwargs=True).remove)
                for name in self.methods.get(type(module), ()):
                    fn = getattr(module, name, None)
                    if not callable(fn):
                        continue
                    previous = module.__dict__.get(name, missing)

                    def restore(module: Any = module, name: str = name, prev: Any = previous):
                        if prev is missing:
                            module.__dict__.pop(name, None)
                        else:
                            object.__setattr__(module, name, prev)

                    stack.callback(restore)
                    object.__setattr__(module, name, self._wrap(module, name, fn))
            self._ctx = stack.pop_all()
        return self

    def __exit__(self, *exc: object) -> None:
        for _, _, rf, _ in reversed(self._open):  # a call that raised
            with contextlib.suppress(Exception):
                rf.__exit__(None, None, None)
        self._open.clear()
        self._depth.clear()
        if self._ctx is not None:
            self._ctx.close()
            self._ctx = None


# ------------------------------------------------------------------ markdown


def _pct(part: float, whole: float) -> str:
    return f"{part / whole:.1%}" if whole else "?"


def markdown(tl: Mapping[str, Any], *, top: int = 12) -> list[str]:
    """``## Timeline`` lines of the profile summary."""
    window = float(tl["window_ms"])
    lines = [
        "",
        "## Timeline (one profiled run)",
        "",
        f"* GPU busy {tl['busy_ms']:.1f} ms of the {window:.1f} ms profiled window "
        f"({_pct(tl['busy_ms'], window)}; the union over every stream), idle "
        f"{tl['idle_ms']:.1f} ms; {tl['kernels']} kernels, {tl['gpu_events']} GPU events",
    ]
    streams = tl.get("streams") or []
    if len(streams) > 1:
        lines.append(
            f"* {len(streams)} streams: "
            + ", ".join(
                f"stream {s['stream']} {s['busy_ms']:.2f} ms busy ({s['events']} events)"
                for s in streams[:6]
            )
            + f"; they overlap {tl['overlap_ms']:.2f} ms (the summed kernel time, "
            f"{tl['kernel_time_ms']:.2f} ms, counts that twice)"
        )
    lead = tl.get("host_lead_ms")
    if lead is not None:
        lines.append(
            f"* host lead (median time from a launch call to its GPU start): {lead:.3f} ms"
            + (
                " — the host runs ahead: GPU bound"
                if lead >= 1.0
                else " — the GPU waits for launches: host / launch bound"
                if lead < 0.05
                else ""
            )
        )
    gaps = tl.get("gaps") or {}
    causes = gaps.get("causes") or {}
    if causes:
        lines.append(
            f"* idle gaps ≥ {gaps.get('min_attributed_us', MIN_GAP_US):g} us by cause: "
            + ", ".join(
                f"{c} {v['ms']:.2f} ms ({v['count']})"
                for c, v in sorted(causes.items(), key=lambda kv: -kv[1]["ms"])
            )
            + f"; lead-in {tl['lead_in_ms']:.2f} ms before the first GPU work"
            + (f", tail {tl['tail_ms']:.2f} ms after the last" if tl.get("tail_ms") else "")
        )
    bins = gaps.get("bins") or []
    if bins:
        lines.append(
            "* gaps by size: "
            + ", ".join(f"{b['bin']} {b['count']} ({b['ms']:.2f} ms)" for b in bins if b["count"])
        )
    places = gaps.get("by_place") or []
    if places:
        lines += [
            "",
            "Where the GPU idles (gaps ≥ 2 us; *host sync*: the host waited for the GPU, e.g. "
            "`.cpu()` / `.item()` / a pageable copy, then launched the next work late, the "
            "Python lines under *Host synchronisation* when the profile scanned them; *host "
            "late*: launches came after the GPU went idle):",
            "",
            "| cause | idle after stage | until stage | gaps | ms | median us "
            "| last GPU work before |",
            "|---|---|---|---|---|---|---|",
        ]
        for p in places[:8]:
            lines.append(
                f"| {p['cause']} | `{p['before']}` | `{p['after']}` | {p['count']} | "
                f"{p['ms']:.2f} | {p['median_us']} | `{p['after_event'][:60]}` |"
            )
    rows = [s for s in tl.get("stages") or [] if s["events"]]
    if rows:
        lines += [
            "",
            "Stages (calls of the workload's top-level modules and entrypoints; GPU work by "
            "the innermost stage that launched it):",
            "",
            "| stage | calls | kernels | GPU ms | busy ms | host lead us | graph | SM fill |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for s in rows[:top]:
            fill = "" if s.get("sm_fill") is None else f"{s['sm_fill']:.2f}"
            lines.append(
                f"| `{s['stage']}` | {s['calls']} | {s['kernels']} | {s['gpu_ms']:.2f} | "
                f"{s['busy_ms']:.2f} | {s['host_lead_us']} | {s['graph_events']} | {fill} |"
            )
    crit = tl.get("critical_path") or {}
    if crit.get("calls"):
        off = ", ".join(
            f"`{o['stage']}` {o['ms']:.2f} ms ({o['calls']} calls)" for o in crit["off_path"][:5]
        )
        lines += [
            "",
            f"* critical path (producer → consumer through the stage calls' tensors, "
            f"{crit['edges']} edges, {crit['assumed_edges']} assumed through glue ops): "
            f"{crit['critical_path_ms']:.2f} of {crit['stage_busy_ms']:.2f} ms of stage GPU "
            f"time; off the path {crit['overlap_potential_ms']:.2f} ms"
            + (f": {off}" if off else "")
            + " (could overlap on another stream; approximate: state held outside the "
            "arguments, e.g. a KV cache, adds no edge)",
        ]
    return lines
