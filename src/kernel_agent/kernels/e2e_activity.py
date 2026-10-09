"""End-to-end hidden-work check: one profiled run of the workload (#147, docs/PARALLEL.md §6.2).

The end-to-end timing (``workloads.base.timed_run``) brackets ``workload.run`` with
device-wide synchronisations, so every stream of the device is timed. What it cannot see is
work launched *after* ``run()`` returned, from another thread, or on another device. The
rules of the module evaluator (:mod:`kernel_agent.kernels.integrity`), stated for a run:

* all GPU work of ``run()`` is launched from the calling thread (``foreign_threads``);
* every stream is joined before ``run()`` returns: no GPU work launched during the run is
  still running once the caller's stream has drained (``unjoined``), no handle of
  :func:`kernel_agent.concurrency.launch` is left unjoined (``outstanding``), and the
  current stream is the one ``run()`` was called on (``stream_changed``);
* no GPU work is launched after ``run()`` returned (``late_launches``) and no thread runs
  the code of the evaluated transforms or kernels then (``live_threads``);
* no GPU work runs on a device the workload does not use (``other_devices``).

Joined streams, declared (:mod:`kernel_agent.concurrency`) or not, pass. Notes, not
failures: ``streams`` (the declared streams the process used) and ``undeclared_streams``
(streams other than the caller's and the declared ones that ran the run's GPU work: a raw
``torch.cuda.Stream()``, or a graph's internal branch streams).

:func:`analyse` is a pure function of the profiler's events (CPU-testable);
:func:`profiled_run` and :func:`check` make the CUDA calls. The worker runs :func:`check`
once per ``e2e`` / ``e2e_ab`` evaluation, after every timed run (the profiler's CUPTI
subscription slows every later launch of the process).
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import inspect
import secrets
import sys
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol

import torch

from kernel_agent.profiling.timeline import activity_type as _activity_type
from kernel_agent.profiling.timeline import start_warm

#: GPU work of a run may end this long after the caller's stream drained (ns).
JOIN_SLACK_NS = 2_000
#: Wall time after ``run()`` returned in which late launches are looked for (s).
SETTLE_S = 0.05
#: Kineto activity types of GPU work and of the API calls that launch it.
GPU_WORK = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})
API_CALLS = frozenset({"cuda_runtime", "cuda_driver"})
#: Profiler ranges of the check; :func:`profiled_run` appends a random tag to each, so code
#: under test cannot fake them with ``record_function`` ranges of the same name.
RUN, MARK, CALLER, STREAM = "ka::run", "ka::mark", "ka::caller", "ka::stream:"
_LIMIT = 5  # examples kept per finding


class ProfilerUnavailable(RuntimeError):
    """The profiler could not start (the check is skipped, not failed)."""


class _EventLike(Protocol):
    """:class:`Event` or ``kernels.integrity._Event`` (which has no ``device``)."""

    @property
    def kind(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def start(self) -> int: ...
    @property
    def end(self) -> int: ...
    @property
    def corr(self) -> int: ...
    @property
    def resource(self) -> int: ...


@dataclasses.dataclass(frozen=True)
class Event:
    """One profiler event: ``resource`` is the stream of GPU work and the thread of CPU
    events, ``device`` the GPU of GPU work (-1 otherwise)."""

    kind: str
    name: str
    start: int
    end: int
    corr: int
    resource: int
    device: int = -1


def events(prof: Any) -> list[Event]:
    """The events of a finished ``torch.profiler.profile``."""
    out = []
    for e in prof.profiler.kineto_results.events():
        start = int(e.start_ns())
        kind = _activity_type(e)
        device = -1
        if kind in GPU_WORK:
            with contextlib.suppress(Exception):
                device = int(e.device_index())
        out.append(
            Event(
                kind,
                e.name(),
                start,
                start + int(e.duration_ns()),
                int(e.correlation_id()),
                int(e.device_resource_id()),
                device,
            )
        )
    return out


# ------------------------------------------------------------------ pure analysis


def _ranges(evts: Iterable[_EventLike]) -> dict[str, _EventLike]:
    return {e.name: e for e in evts if e.kind == "user_annotation" and e.name.startswith("ka::")}


def _gpu_by_corr(evts: Iterable[_EventLike]) -> dict[int, list[_EventLike]]:
    gpu: dict[int, list[_EventLike]] = collections.defaultdict(list)  # a graph launch: many
    for e in evts:
        if e.kind in GPU_WORK:
            gpu[e.corr].append(e)
    return gpu


def _within(launches: Iterable[_EventLike], span: _EventLike | None) -> list[_EventLike]:
    return [] if span is None else [e for e in launches if span.start <= e.start <= span.end]


def stream_ids(evts: Sequence[_EventLike], tag: str = "") -> dict[str, int]:
    """Profiler stream id of every declared stream, from the marker kernel the evaluator
    enqueued on it inside a ``ka::stream:<label><tag>`` range (:func:`mark_streams`)."""
    gpu = _gpu_by_corr(evts)
    launches = [e for e in evts if e.kind in API_CALLS and e.corr in gpu]
    found = {}
    for name, span in _ranges(evts).items():
        if name.startswith(STREAM) and name.endswith(tag):
            for launch in _within(launches, span):
                label = name.removeprefix(STREAM)
                found[label[: len(label) - len(tag)] if tag else label] = gpu[launch.corr][
                    0
                ].resource
                break
    return found


def caller_stream(mark: Sequence[_EventLike], ident: Sequence[_EventLike]) -> int:
    """The profiler's id of the caller's stream: where the ``ka::mark`` marker kernel ran.
    ``mark``: the GPU records of the marker's correlation id; ``ident``: those of the
    ``ka::caller`` marker, launched on the same stream once the device is idle.

    A correlation id does not always name one launch's work: once CUPTI is initialised before
    a CUDA graph with a conditional node is instantiated (any earlier profiled run in the
    process does it), the profiler records every iteration of the conditional body, on a
    stream id of its own, under correlation ids that are not the graph launch's: 0, ids of no
    recorded call and now and then the id of the API call the host was making when the body
    kernel started, the ``ka::mark`` marker's launch among them (measured on an NVIDIA A10,
    torch 2.10, CUDA 12.9: a WHILE loop's step kernel listed first under the marker's id, so
    the check took its stream for the caller's and its start for the time the caller's stream
    drained, and refused the joined loop's last kernel; #232). Nothing runs on the device
    while the ``ka::caller`` marker is launched, so its stream is the caller's; where it is
    missing, the first record of the marker's id (the earlier rule)."""
    streams = [w.resource for w in mark]
    found = next((w.resource for w in ident if w.resource in streams), None)
    return streams[0] if found is None else found


def undeclared(seen: collections.Counter[int], caller: int, declared: dict[str, int]) -> list[str]:
    """Streams in ``seen`` (stream id -> GPU operations) other than the caller's and the
    declared ones, as readable notes."""
    known = {caller, *declared.values()}
    return [f"stream {s}: {n} GPU operations" for s, n in sorted(seen.items()) if s not in known]


def mark_streams(record_function: Callable[[str], Any], tag: str = "") -> list[str]:
    """Enqueue a marker kernel on every declared stream inside a ``ka::stream:<label><tag>``
    range (for :func:`stream_ids`); returns the labels."""
    from kernel_agent import concurrency

    declared = concurrency.streams()
    for label, s in declared.items():
        with torch.cuda.stream(s), record_function(f"{STREAM}{label}{tag}"):
            torch.cuda._sleep(1)
    return list(declared)


def analyse(
    evts: Sequence[_EventLike],
    *,
    devices: Iterable[int],
    outstanding: Iterable[str] = (),
    stream_changed: bool = False,
    live_threads: Iterable[str] = (),
    tag: str = "",
    slack_ns: int = JOIN_SLACK_NS,
) -> dict[str, Any]:
    """The verdict of one profiled run (:func:`profiled_run`'s events): ``passed``,
    ``reason`` and the findings of the module docstring. ``devices``: the GPUs the workload
    may use; ``outstanding``, ``stream_changed``, ``live_threads``: what the run left behind
    (the concurrency module's unjoined handles, a changed current stream, threads running
    the evaluated code); ``tag``: the suffix of the check's range names."""
    out: dict[str, Any] = {
        "passed": True,
        "reason": "",
        "foreign_threads": [],
        "unjoined": [],
        "late_launches": [],
        "other_devices": [],
        "outstanding": sorted(set(outstanding)),
        "stream_changed": bool(stream_changed),
        "live_threads": sorted(set(live_threads)),
        "undeclared_streams": [],
    }
    ranges = {n: s for n, s in _ranges(evts).items() if n.endswith(tag)}
    run, mark = ranges.get(RUN + tag), ranges.get(MARK + tag)
    gpu = _gpu_by_corr(evts)
    launches = [e for e in evts if e.kind in API_CALLS and e.corr in gpu]
    marks = _within(launches, mark)
    if run is None or not marks:
        out["note"] = "the profiler recorded no GPU activity; GPU checks skipped"
        return _verdict(out)
    main = marks[0].resource  # the thread that called run()
    mark_work = gpu[marks[0].corr]
    idents = [e for e in _within(launches, ranges.get(CALLER + tag)) if e.resource == main]
    caller = caller_stream(mark_work, gpu[idents[0].corr] if idents else [])
    # the caller's stream reached the marker (its own record: the id may name others too)
    drained = min(w.start for w in mark_work if w.resource == caller)
    own = {  # the check's own markers, enqueued after run() returned
        e.corr
        for name, span in ranges.items()
        if name in (MARK + tag, CALLER + tag) or name.startswith(STREAM)
        for e in _within(launches, span)
        if span.start >= run.end
    }
    allowed = set(devices)
    seen: collections.Counter[int] = collections.Counter()
    for launch in launches:
        if launch.corr in own or launch.start < run.start:
            continue
        work = gpu[launch.corr]
        label = work[0].name[:80]
        if launch.resource != main:
            out["foreign_threads"].append(f"{label} launched from thread {launch.resource}")
        if launch.start > run.end:
            out["late_launches"].append(
                f"{label} launched {(launch.start - run.end) / 1e3:.0f} us after run() returned"
            )
            continue
        if launch.resource != main:
            continue
        for w in work:
            seen[w.resource] += 1
            # Work on the caller's own stream precedes the marker by stream order, so it is
            # joined whatever its timestamps say (#268: a WHILE graph's last kernel looked
            # unjoined because a body kernel under the marker's correlation id was taken for
            # the marker; caller_stream).
            if w.resource != caller and w.end > drained + slack_ns:
                out["unjoined"].append(
                    f"{w.name[:80]} (stream {w.resource}) ran {(w.end - drained) / 1e3:.0f} us "
                    "past the caller's stream after run() returned"
                )
    for work in gpu.values():
        for w in work:
            device = getattr(w, "device", -1)
            if device >= 0 and device not in allowed:
                out["other_devices"].append(f"{w.name[:80]} on cuda:{device}")
    out["undeclared_streams"] = undeclared(seen, caller, stream_ids(evts, tag))
    return _verdict(out)


def _verdict(out: dict[str, Any]) -> dict[str, Any]:
    for key in ("foreign_threads", "unjoined", "late_launches", "other_devices"):
        out[key] = sorted(set(out[key]))[:_LIMIT]
    problems = []
    if out["foreign_threads"]:
        problems.append(
            "GPU work launched from another thread: " + "; ".join(out["foreign_threads"][:2])
        )
    if out["late_launches"]:
        problems.append(
            "GPU work launched after run() returned: " + "; ".join(out["late_launches"][:2])
        )
    if out["unjoined"]:
        problems.append(
            "GPU work on a stream the caller's stream never waits for: "
            + "; ".join(out["unjoined"][:2])
        )
    if out["outstanding"]:
        problems.append(
            "kernel_agent.concurrency.launch handles not joined when run() returned: "
            + ", ".join(out["outstanding"])
        )
    if out["stream_changed"]:
        problems.append("run() returned on another current stream than it was called on")
    if out["live_threads"]:
        problems.append(
            "threads still run the evaluated code after run() returned: "
            + "; ".join(out["live_threads"][:2])
        )
    if out["other_devices"]:
        problems.append("GPU work on another device: " + "; ".join(out["other_devices"][:2]))
    if problems:
        out["passed"] = False
        out["reason"] = (
            "hidden work: "
            + ". ".join(problems)
            + ". Launch all GPU work from the calling thread, join every stream before run() "
            "returns (kernel_agent.concurrency: `with cc.fork(name):`, `cc.launch(...)` + "
            "`cc.join_all()`), stay on the workload's device"
        )
    if not out["undeclared_streams"]:
        del out["undeclared_streams"]
    return out


# ------------------------------------------------------------------ threads


def _code_files(fn: Any) -> list[str]:
    fn = getattr(fn, "__func__", fn)
    with contextlib.suppress(Exception):
        fn = inspect.unwrap(fn)
    code = getattr(fn, "__code__", None)
    return [code.co_filename] if code is not None else []


def live_threads(dirs: Iterable[Path]) -> list[str]:
    """Live threads (other than this one) whose target, or any frame running now, is code
    from one of ``dirs`` (the evaluated transforms' and kernels' directories)."""
    roots = [str(Path(d).resolve()) for d in dirs]
    if not roots:
        return []

    def ours(filename: str) -> bool:
        with contextlib.suppress(Exception):
            filename = str(Path(filename).resolve())
        return filename.startswith(tuple(roots))

    frames = sys._current_frames()
    found = []
    for thread in threading.enumerate():
        if thread is threading.current_thread() or not thread.is_alive():
            continue
        files = []
        for attr in ("_target", "function"):  # Thread(target=...), threading.Timer
            files += _code_files(getattr(thread, attr, None))
        if type(thread).run is not threading.Thread.run:
            files += _code_files(type(thread).run)
        frame = frames.get(thread.ident) if thread.ident is not None else None
        while frame is not None:
            files.append(frame.f_code.co_filename)
            frame = frame.f_back
        hits = [f for f in files if ours(f)]
        if hits:
            found.append(f"{thread.name} ({Path(hits[0]).name})")
    return found


# ------------------------------------------------------------------ the profiled run


def _synchronize_all() -> None:
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


def profiled_run(
    run: Callable[[Any], Any],
    inputs: Any,
    *,
    device: int,
    settle_s: float = SETTLE_S,
) -> tuple[list[Event], dict[str, Any]]:
    """One ``run(inputs)`` under the profiler: then a marker kernel on the caller's stream
    (``ka::mark``: it starts once that stream drained), ``settle_s`` of waiting for late
    launches and, once the device is idle, the markers that name the streams: one more on
    the caller's stream (``ka::caller``, :func:`caller_stream`) and one on every declared
    stream (:func:`stream_ids`). Returns the events and what the run left behind
    (``outstanding``, ``stream_changed``, ``tag``: the ranges' suffix). Raises
    :class:`ProfilerUnavailable` when the profiler does not start."""
    from torch.autograd.profiler import record_function
    from torch.profiler import ProfilerActivity, profile

    from kernel_agent import concurrency

    if torch._C._autograd._profiler_enabled():  # started by the code under test
        raise RuntimeError("another torch profiler is active in the process")
    caller = torch.cuda.current_stream(device)
    tag = f"#{secrets.token_hex(4)}"
    state: dict[str, Any] = {"tag": tag}
    _synchronize_all()
    prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
    try:
        start_warm(prof)  # CUPTI may lose a session's first kernel records (timeline.py)
    except Exception as exc:
        raise ProfilerUnavailable(f"{type(exc).__name__}: {exc}") from exc
    try:
        with torch.inference_mode():
            with record_function(RUN + tag):
                run(inputs)
            state["outstanding"] = concurrency.outstanding()
            state["stream_changed"] = torch.cuda.current_stream(device) != caller
            with torch.cuda.stream(caller), record_function(MARK + tag):
                torch.cuda._sleep(1)
            time.sleep(settle_s)  # late launches
            _synchronize_all()
            # the device is idle: no graph body can run under these markers' correlation ids
            with torch.cuda.stream(caller), record_function(CALLER + tag):
                torch.cuda._sleep(1)
            mark_streams(record_function, tag)
            _synchronize_all()
    finally:
        torch.cuda.set_stream(caller)
        prof.stop()
    return events(prof), state


def check(
    run: Callable[[Any], Any],
    inputs: Any,
    *,
    device: torch.device | int | None = None,
    dirs: Iterable[Path] = (),
    settle_s: float = SETTLE_S,
) -> dict[str, Any]:
    """The verdict of one profiled ``run(inputs)`` (:func:`analyse`), plus ``streams``: the
    declared streams used. ``device``: the workload's device (the only GPU it may use);
    ``dirs``: where the evaluated transforms and kernels live. A run that raises under the
    profiler fails; a profiler that cannot start skips."""
    from kernel_agent import concurrency

    if isinstance(device, torch.device) and device.type != "cuda":
        return {"passed": True, "reason": "", "skipped": f"the workload runs on {device}"}
    if not torch.cuda.is_available():
        return {"passed": True, "reason": "", "skipped": "no CUDA device"}
    index = device.index if isinstance(device, torch.device) else device
    index = torch.cuda.current_device() if index is None else int(index)
    allowed = {index}
    try:
        evts, state = profiled_run(run, inputs, device=index, settle_s=settle_s)
    except ProfilerUnavailable as exc:
        concurrency.join_all()
        return {"passed": True, "reason": "", "skipped": f"the profiler did not start: {exc}"}
    except Exception as exc:
        concurrency.join_all()
        return {"passed": False, "reason": f"hidden work: the profiled run failed: {exc}"[:500]}
    verdict = analyse(
        evts,
        devices=allowed,
        outstanding=state.get("outstanding", ()),
        stream_changed=state.get("stream_changed", False),
        live_threads=live_threads(dirs),
        tag=state["tag"],
    )
    concurrency.join_all()  # the next run starts clean
    if streams := concurrency.used():
        verdict["streams"] = streams
    return verdict
