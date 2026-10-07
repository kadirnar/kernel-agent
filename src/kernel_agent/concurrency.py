"""Declared concurrency for kernel candidates and transforms (#147, docs/PARALLEL.md §7.2).

Overlap is legal when it is visible to the evaluator:

1. all GPU work is launched from the calling thread (:func:`launch` enqueues on the
   caller's thread, only on another stream);
2. every stream is joined back before the call (a kernel candidate) or ``run()`` (a
   transform) returns;
3. side streams are declared by name here, so results record them (``streams`` in the
   result rows); a raw ``torch.cuda.Stream()`` stays legal when joined and is reported
   as ``undeclared_streams`` (a note, not a failure).

The evaluator checks 1 and 2 per call (:mod:`kernel_agent.kernels.integrity`) and per run
(:mod:`kernel_agent.kernels.e2e_activity`). Usage::

    from kernel_agent import concurrency as cc

    with cc.fork("aux"):  # the named stream "aux" waits for the caller's stream ...
        y = branch(x)  # ... runs this ...
    # ... and the caller's stream waits for "aux" here. In a CUDA graph capture the fork
    # and the join become graph edges (multi-stream capture).

    handle = cc.launch("aux", branch, x)  # enqueue now on "aux" ...
    y = handle.result()  # ... the current stream waits for it here
    cc.join_all()  # every handle not joined yet, before run() returns

    split = cc.partition(sms=16)  # green contexts: (part, rest), None where unsupported
    if split is None:
        print(cc.partition_error())  # why
    else:
        part, rest = split  # disjoint SMs: 16 (rounded up by the driver) and the others
        with cc.fork("aux", partition=part):
            ...
    blocks = cc.sm_count() * per_sm  # SMs of the current stream's partition

Allocator safety: every entry into a named stream (:func:`fork`, :func:`launch`) first
waits for the caller's stream, so memory the side stream's allocations reuse is free of
the caller's pending work; :func:`launch` (whose work outlives the call) records its tensor
arguments on the side stream and its tensor results on the joining stream
(``record_stream``), outside of graph captures. Use named streams only through
:func:`fork` and :func:`launch`.

CUDA C++ kernels: ``ka_launch.cuh`` in :func:`include_dir` launches with programmatic
dependent launch (PDL) and cooperative grids, partition-aware (``ka_sm_count``).

Measured on one RTX 5070 Ti (docs/PARALLEL.md §4): two already GPU-filling stages on two
eager streams overlap by 1-5 %; two independent branches captured forked into one CUDA
graph run 1.09-1.16x faster than serial; a green-context partition never beat plain
streams for throughput but keeps a DRAM-bound job at full speed beside another job.
"""

from __future__ import annotations

import contextlib
import dataclasses
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch

#: The C++ header for CUDA candidates (``ka_launch``, ``ka_pdl_wait``, ``ka_sm_count``).
HEADER = "ka_launch.cuh"
#: Green contexts need a user-mode driver of at least this version (``cuDriverGetVersion``).
MIN_DRIVER = 12080


def include_dir() -> Path:
    """Directory of :data:`HEADER`, for ``load_inline(extra_include_paths=[...])``."""
    return Path(__file__).parent / "include"


# ------------------------------------------------------------------ CUDA calls


class _Torch:
    """The CUDA calls of this module (the CPU tests substitute a fake)."""

    def current_stream(self, device: int | None = None) -> Any:
        return torch.cuda.current_stream(device)

    def new_stream(self, device: int) -> Any:
        return torch.cuda.Stream(device=device)

    def use(self, stream: Any) -> contextlib.AbstractContextManager[Any]:
        return torch.cuda.stream(stream)

    def event(self) -> Any:
        return torch.cuda.Event()

    def capturing(self) -> bool:
        return bool(torch.cuda.is_current_stream_capturing())

    def device_sms(self, device: int) -> int:
        return int(torch.cuda.get_device_properties(device).multi_processor_count)


_cuda: Any = _Torch()


def _device(stream: Any) -> int:
    index = getattr(getattr(stream, "device", None), "index", None)
    return int(index) if index is not None else 0


def _key(stream: Any) -> tuple[int, int]:
    """Identity of a stream: (device, CUDA stream handle)."""
    return _device(stream), int(stream.cuda_stream)


# ------------------------------------------------------------------ partitions


@dataclasses.dataclass(eq=False)
class Partition:
    """SMs of one green context (:func:`partition`): ``sms`` granted, ``context`` the torch
    ``GreenContext`` its streams come from."""

    sms: int
    device: int
    context: Any
    label: str = ""

    def stream(self) -> Any:
        """A new stream whose kernels run on this partition's SMs only."""
        return self.context.Stream()


def partition_unsupported(device: int | None = None) -> str | None:
    """Why :func:`partition` cannot work here (None: it can)."""
    if not torch.cuda.is_available():
        return "no CUDA device"
    if torch.version.cuda is None or getattr(torch.version, "hip", None):
        return "green contexts need NVIDIA CUDA"
    try:
        from torch.cuda._utils import _check_cuda_bindings, _cuda_bindings_driver
        from torch.cuda.green_contexts import GreenContext
    except Exception as exc:  # older torch
        return f"this torch has no green contexts ({exc})"
    if _cuda_bindings_driver is None:
        return "green contexts need the cuda.bindings package (pip install cuda-python)"
    if not hasattr(GreenContext, "_init_from_cuda_objects"):
        return "this torch's GreenContext cannot wrap a driver partition"
    try:
        version = int(_check_cuda_bindings(_cuda_bindings_driver.cuDriverGetVersion()))
    except Exception as exc:
        return f"cuDriverGetVersion failed: {exc}"
    if version < MIN_DRIVER:
        return f"green contexts need a CUDA driver >= 12.8 (this one: {version})"
    return None


_partitions: list[tuple[Partition, Partition]] = []  # alive as long as the process
_partition_error: dict[int, str] = {}


def partition(sms: int, *, device: int | None = None) -> tuple[Partition, Partition] | None:
    """Two disjoint SM partitions of ``device``: one of at least ``sms`` SMs (the driver
    rounds up to its granularity, 8 SMs on sm_120) and the remaining SMs, or None where
    green contexts are not supported or the split failed (:func:`partition_error` says why;
    :func:`partition_unsupported` checks support without creating anything). Each call
    creates two new green contexts: call it once and keep the result.

    One driver-API split (``cuDevSmResourceSplitByCount``) makes both, so they never share
    SMs; two ``torch.cuda.GreenContext(num_sms=k)`` are each split from the whole device
    and overlap. Kernels on the caller's ordinary streams still run on every SM: put the
    main work on ``rest``'s stream to confine it. A CUDA graph captured outside a partition
    and replayed on its stream is not confined; size cooperative grids by
    :func:`sm_count` on the partition's stream."""
    index = _index(device)
    if (why := partition_unsupported(index)) is not None:
        _partition_error[index] = why
        return None
    try:
        from torch.cuda._utils import _check_cuda_bindings as chk
        from torch.cuda._utils import _cuda_bindings_driver as drv

        dev = chk(drv.cuDeviceGet(index))
        whole = chk(drv.cuDeviceGetDevResource(dev, drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))
        total = int(whole.sm.smCount)
        if not 0 < sms < total:
            raise ValueError(f"asked for {sms} of the device's {total} SMs: nothing would remain")
        groups, count, rest = chk(drv.cuDevSmResourceSplitByCount(1, whole, 0, int(sms)))
        if int(count) != 1 or int(rest.sm.smCount) <= 0:
            raise RuntimeError(f"the split gave {count} groups and {rest.sm.smCount} other SMs")
        part = _wrap(index, dev, groups[0], f"{int(groups[0].sm.smCount)}sm")
        other = _wrap(index, dev, rest, f"rest{int(rest.sm.smCount)}sm")
    except Exception as exc:
        _partition_error[index] = f"{type(exc).__name__}: {exc}"[:300]
        return None
    _partition_error.pop(index, None)
    _partitions.append((part, other))
    return part, other


def _wrap(index: int, dev: Any, resource: Any, label: str) -> Partition:
    """A torch ``GreenContext`` around one driver-API SM resource (docs/PARALLEL.md A.4)."""
    from torch.cuda._utils import _check_cuda_bindings as chk
    from torch.cuda._utils import _cuda_bindings_driver as drv
    from torch.cuda.green_contexts import GreenContext

    desc = chk(drv.cuDevResourceGenerateDesc([resource], 1))
    flags = drv.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM
    green = chk(drv.cuGreenCtxCreate(desc, dev, flags))
    try:
        context = chk(drv.cuCtxFromGreenCtx(green))
    except Exception:
        chk(drv.cuGreenCtxDestroy(green))
        raise
    wrapped = GreenContext.__new__(GreenContext)
    wrapped._init_from_cuda_objects(index, green, context)
    return Partition(int(resource.sm.smCount), index, wrapped, label)


def partition_error(device: int | None = None) -> str | None:
    """Why the last :func:`partition` call on ``device`` returned None."""
    return _partition_error.get(_index(device))


def _index(device: int | None) -> int:
    if device is not None:
        return int(device)
    return torch.cuda.current_device() if torch.cuda.is_available() else 0


# ------------------------------------------------------------------ named streams

_lock = threading.RLock()
_named: dict[tuple[int, str, Partition | None], Any] = {}  # (device, name, partition) -> stream
_labels: dict[tuple[int, int], str] = {}  # stream identity -> label
_in_partition: dict[tuple[int, int], Partition] = {}  # stream identity -> its partition
_used: dict[str, None] = {}  # labels used since reset(), in order
_pending: list[Handle] = []


def _label(name: str, part: Partition | None, device: int) -> str:
    label = f"{name}@{part.label}" if part is not None else name
    return label if device == 0 else f"{label}:cuda{device}"


def stream(name: str, *, partition: Partition | None = None, device: int | None = None) -> Any:
    """The named stream ``name`` (created once per device and partition) and declare it.
    Run work on it through :func:`fork` or :func:`launch`, which order it after the caller's
    stream and join it back."""
    if not isinstance(name, str) or not name or name.startswith("ka::"):
        raise ValueError(f"a stream name is a non-empty string, not {name!r}")
    if partition is not None:
        index = partition.device
    elif device is not None:
        index = int(device)
    else:
        index = _device(_cuda.current_stream())
    key = (index, name, partition)  # a Partition hashes by identity (and stays alive)
    with _lock:
        side = _named.get(key)
        if side is None:
            side = partition.stream() if partition is not None else _cuda.new_stream(index)
            _named[key] = side
            _labels[_key(side)] = _label(name, partition, index)
            if partition is not None:
                _in_partition[_key(side)] = partition
        _used[_labels[_key(side)]] = None
    return side


@contextlib.contextmanager
def fork(name: str, *, partition: Partition | None = None) -> Iterator[Any]:
    """Run the block on the named stream: on entry it waits for the caller's stream, on exit
    (also when the block raises) the caller's stream waits for it. Capture-safe: inside a
    ``torch.cuda.graph`` capture the fork and the join become graph edges. Yields the side
    stream."""
    caller = _cuda.current_stream()
    side = stream(name, partition=partition, device=_device(caller))
    if _key(side) == _key(caller):
        raise ValueError(f"stream {name!r} is the caller's own stream")
    side.wait_stream(caller)  # fork
    try:
        with _cuda.use(side):
            yield side
    finally:
        caller.wait_stream(side)  # join


@dataclasses.dataclass(eq=False)
class Handle:
    """Work :func:`launch` enqueued on a named stream; :meth:`result` joins it."""

    name: str
    stream: Any
    done: Any  # event recorded on ``stream`` right after the work
    value: Any
    joined: bool = False

    def result(self) -> Any:
        """Make the current stream wait for the work (once) and return its result."""
        with _lock:
            if not self.joined:
                current = _cuda.current_stream()
                current.wait_event(self.done)
                if not _cuda.capturing():
                    _record(self.value, current)
                self.joined = True
                with contextlib.suppress(ValueError):
                    _pending.remove(self)
        return self.value


def launch(
    name: str,
    fn: Callable[..., Any],
    /,
    *args: Any,
    partition: Partition | None = None,
    **kwargs: Any,
) -> Handle:
    """Enqueue ``fn(*args, **kwargs)`` on the named stream now, on this thread, after the
    caller's stream's work so far; join it later with :meth:`Handle.result` or
    :func:`join_all` (before the call or ``run()`` returns). For pipelining: the handle
    joins exactly this work (an event), not what is enqueued on the stream after it.
    ``partition`` is this function's own keyword (a :func:`partition` to run in), not
    passed to ``fn``."""
    caller = _cuda.current_stream()
    side = stream(name, partition=partition, device=_device(caller))
    if _key(side) == _key(caller):
        raise ValueError(f"stream {name!r} is the caller's own stream")
    side.wait_stream(caller)
    with _cuda.use(side):
        value = fn(*args, **kwargs)
    done = _cuda.event()
    done.record(side)
    if not _cuda.capturing():  # inputs stay allocated until the side stream is done
        _record((args, kwargs), side)
    handle = Handle(_labels[_key(side)], side, done, value)
    with _lock:
        _pending.append(handle)
    return handle


def join_all() -> int:
    """Join every handle of :func:`launch` not joined yet into the current stream; returns
    how many."""
    with _lock:
        handles = list(_pending)
    for handle in handles:
        handle.result()
    return len(handles)


def outstanding() -> list[str]:
    """Names of the :func:`launch` handles not joined yet."""
    with _lock:
        return [h.name for h in _pending]


def used() -> list[str]:
    """Labels of the named streams used since :func:`reset` (``name``, ``name@16sm`` in a
    partition): what result rows record as ``streams``."""
    with _lock:
        return list(_used)


def streams() -> dict[str, Any]:
    """Every named stream of this process by label (the evaluator marks them)."""
    with _lock:
        return {_labels[_key(s)]: s for s in _named.values()}


def reset() -> None:
    """Forget which streams were used and drop unjoined handles (the streams stay)."""
    with _lock:
        _used.clear()
        _pending.clear()


def sm_count(stream: Any = None) -> int:
    """SMs available to kernels on ``stream`` (default: the current stream): its
    partition's when it is a stream of :func:`partition`, else the device's. Size persistent
    and cooperative grids by this: inside a partition the device attribute
    ``multiProcessorCount`` still reports the whole device."""
    s = _cuda.current_stream() if stream is None else stream
    part = _in_partition.get(_key(s))
    return part.sms if part is not None else _cuda.device_sms(_device(s))


def _record(tree: Any, stream: Any) -> None:
    """``record_stream(stream)`` on every CUDA tensor in ``tree``."""
    if isinstance(tree, torch.Tensor):
        if tree.is_cuda:
            tree.record_stream(stream)
    elif isinstance(tree, dict):
        for value in tree.values():
            _record(value, stream)
    elif isinstance(tree, list | tuple):
        for value in tree:
            _record(value, stream)
