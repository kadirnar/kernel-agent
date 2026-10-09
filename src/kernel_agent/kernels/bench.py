"""GPU timing utilities.

Two timing contexts (:func:`time_call`'s ``context``, issue #226), as a module runs in the
model: ``eager`` calls, each timed with CUDA events (host launch time included when the GPU
waits for it), or ``graph``: :data:`GRAPH_CALLS` calls captured in one CUDA graph and timed
as its replays, as inside a CUDA-graphed stage, where host time is not paid. Either runs
with a warm L2 or a cold one (``l2_flush``): the L2 is evicted before every call, as when
the model's other work between two calls of a module touches more than the L2 holds.
"""

from __future__ import annotations

import contextlib
import copy
import random
import statistics
import time
from collections.abc import Callable
from typing import Any

import torch

from kernel_agent.profiling.state import split as split_state

# Timing primitives, bound when the evaluator imports this module (before any
# candidate): a candidate that later patches ``torch.cuda.Event.elapsed_time``,
# ``torch.cuda.synchronize`` or ``time.perf_counter`` cannot change a measurement
# (kernels/integrity.py reports the attempt).  The C base class of the event
# cannot be patched.
_EventBase: Any = getattr(torch._C, "_CudaEventBase", torch.cuda.Event)
_Event = torch.cuda.Event
_record: Callable[..., Any] = _EventBase.record
_elapsed: Callable[..., float] = _EventBase.elapsed_time
_current_stream = torch.cuda.current_stream
_synchronize: Callable[[], Any] = getattr(torch._C, "_cuda_synchronize", torch.cuda.synchronize)
_perf_counter = time.perf_counter
_empty = torch.empty
_copy: Callable[..., Any] = torch.Tensor.copy_
_allocated: Callable[[], int] = torch.cuda.memory_allocated
_max_allocated: Callable[[], int] = torch.cuda.max_memory_allocated
_reset_peak: Callable[[], None] = torch.cuda.reset_peak_memory_stats
# ... and those of graph timing (the C base class of the graph cannot be patched either)
_GraphBase: Any = getattr(torch._C, "_CUDAGraph", None)
_CUDAGraph: Any = getattr(torch.cuda, "CUDAGraph", None)
_replay: Callable[..., Any] | None = getattr(_GraphBase or _CUDAGraph, "replay", None)
_graph_capture: Any = getattr(torch.cuda, "graph", None)
_Stream = torch.cuda.Stream
_on_stream = torch.cuda.stream
_set_stream = torch.cuda.set_stream
_gpu_sleep: Callable[[int], Any] = torch.cuda._sleep
_mem_get_info: Callable[..., tuple[int, int]] = torch.cuda.mem_get_info

#: Timing contexts of :func:`time_call`.
EAGER = "eager"
GRAPH = "graph"
CONTEXTS = (EAGER, GRAPH)
#: Calls captured in one CUDA graph (``graph`` context): their replay amortises the graph's
#: own launch (12 L2-cold calls in a graph gave the FP8 verdict of VoxCPM2's slice 8).
GRAPH_CALLS = 12
#: Replays per graph measurement: enough for ``target_ms`` of timed calls, within these.
GRAPH_MIN_REPLAYS = 10
GRAPH_MAX_REPLAYS = 100
#: Inputs that hold mutable state get one copy per call of a graph, up to this share of the
#: free GPU memory (fewer calls per graph beyond it).
GRAPH_COPY_SHARE = 0.25

#: Clock guard (#81). After about a second without work the GPU drops to a lower
#: performance state (an RTX 5070 Ti: memory clock 7001 or 405 MHz instead of 13801, DRAM
#: bandwidth 2x to 50x lower), and the driver raises it again only after 0.3 s to several
#: seconds of load. A launch-bound reference hardly notices; a bandwidth-bound kernel timed
#: meanwhile runs 2x to 27x slower (a fused decoder layer: 0.67x in one evaluation, 8x in
#: the next). So :func:`time_call` measures after :func:`ensure_clocks` and before one more
#: DRAM bandwidth probe (:func:`clock_state`), and measures again when either reads below
#: CLOCK_OK of the GPU's bandwidth (at most CLOCK_RETRIES times).
CLOCK_OK = 0.7
CLOCK_RETRIES = 3
#: :func:`ensure_clocks` spins at most this long per call (a GPU busy with other processes,
#: or capped, never gets there), and at most WARM_BUDGET_S in all per process.
WARM_MAX_S = 10.0
WARM_BUDGET_S = 20.0
#: Without the roofline's measured bandwidth to compare probes with, the first
#: :func:`ensure_clocks` of a process spins this long: the slowest clock ramp seen, with margin.
WARM_UNCALIBRATED_S = 2.5
_PROBE_MIN_BYTES = 64 * 1024 * 1024


def _events() -> tuple[Any, Any]:
    return _Event(enable_timing=True), _Event(enable_timing=True)


def _mark(event: Any) -> None:
    _record(event, _current_stream())


def has_mutable_state(args: Any, kwargs: Any) -> bool:
    """True if inputs contain non-tensor objects (e.g. KV caches) that a call may mutate."""

    def walk(v: Any) -> bool:
        if v is None or isinstance(v, torch.Tensor | int | float | bool | str | torch.dtype):
            return False
        if isinstance(v, tuple | list):
            return any(walk(x) for x in v)
        if isinstance(v, dict):
            return any(walk(x) for x in v.values())
        return True

    return walk(args) or walk(kwargs)


_L2_FLUSH: torch.Tensor | None = None
_SLEEP_CYCLES_PER_US = 0.0  # torch.cuda._sleep's cycles per microsecond (0: not measured)
_WARMED = False
_PEAK_GBPS: float | None = None  # the roofline's dram_gbps (0: none cached, or not reached)
_BEST_GBPS = 0.0  # the best clock probe of this process
_CALIBRATED = False  # ensure_clocks ran in this process
_SPENT_S = 0.0  # seconds ensure_clocks spent in this process


def warm_gpu(ms: float = 300.0) -> None:
    """Spin the GPU (and CPU launch path) so clocks leave idle states before timing
    (once per process; :func:`ensure_clocks` checks the clocks)."""
    global _WARMED
    if _WARMED:
        return
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    while True:
        for _ in range(10):
            a = (a @ a).clamp_(-1, 1)
        end.record()
        end.synchronize()
        if start.elapsed_time(end) > ms:
            break
    _WARMED = True


def dram_gbps() -> float | None:
    """DRAM bandwidth (GB/s, bytes read + written) of one copy between two buffers of twice
    the L2 cache (at least 64 MiB each), as the roofline measures ``dram_gbps``; None when
    they cannot be allocated."""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    n = max(_PROBE_MIN_BYTES, 2 * int(getattr(props, "L2_cache_size", 0) or 0))
    try:
        src = _empty(n, dtype=torch.uint8, device="cuda")
        dst = _empty(n, dtype=torch.uint8, device="cuda")
    except RuntimeError:  # out of memory
        return None
    start, end = _events()
    _mark(start)
    _copy(dst, src)
    _mark(end)
    _synchronize()
    return 2 * n / max(_elapsed(start, end), 1e-6) / 1e6


def _peak_gbps() -> float:
    """The roofline's ``dram_gbps`` of this GPU (:func:`kernels.roofline.current_peaks`;
    0 when it is not cached)."""
    global _PEAK_GBPS
    if _PEAK_GBPS is None:
        _PEAK_GBPS = 0.0
        with contextlib.suppress(Exception):
            from kernel_agent.kernels.roofline import current_peaks

            _PEAK_GBPS = float((current_peaks() or {}).get("dram_gbps") or 0.0)
    return _PEAK_GBPS


def clock_state() -> float | None:
    """One DRAM bandwidth probe (:func:`dram_gbps`) as a share of the GPU's bandwidth (the
    roofline's, at least the best probe of this process): about 1 at full clocks, 0.5 at a
    halved memory clock (None: no probe)."""
    global _BEST_GBPS
    gbps = dram_gbps()
    if gbps is None:
        return None
    _BEST_GBPS = max(_BEST_GBPS, gbps)
    return gbps / max(_peak_gbps(), _BEST_GBPS)


def _spin() -> None:
    """A few milliseconds of matmuls (:func:`ensure_clocks` adds a DRAM copy: its probe)."""
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    for _ in range(10):
        a = (a @ a).clamp_(-1, 1)


def ensure_clocks() -> float | None:
    """Spin the GPU until a clock probe reads CLOCK_OK; returns the last :func:`clock_state`.

    At most WARM_MAX_S per call and WARM_BUDGET_S per process; a call that does not get
    there makes the best probe of this process the reference (the GPU is busy or capped;
    a drop below that still counts). The first call without the roofline's bandwidth
    spins WARM_UNCALIBRATED_S (no reference yet: the best probe is the current one)."""
    global _CALIBRATED, _SPENT_S, _PEAK_GBPS
    begin = _perf_counter()
    floor = 0.0
    if not _CALIBRATED:
        _CALIBRATED = True
        floor = 0.0 if _peak_gbps() else WARM_UNCALIBRATED_S
    state = clock_state()
    while True:
        spun = _perf_counter() - begin
        if spun >= floor and (state is None or state >= CLOCK_OK):
            break
        if spun >= max(floor, min(WARM_MAX_S, WARM_BUDGET_S - _SPENT_S)):
            _PEAK_GBPS = 0.0
            state = clock_state()
            break
        _spin()
        state = clock_state()  # a DRAM copy: the memory clock follows DRAM traffic
    _SPENT_S += _perf_counter() - begin
    return state


def flush_l2() -> None:
    """Evict the L2 cache: read a buffer of random values twice its size (at least 64 MiB).

    A read, not a memset: inside a CUDA graph a memset flush left the subtraction of a
    flush-only graph 10-13 us per call short of the time the same call takes after a flush
    outside the timing, a read flush agreed with it within 1 us (an RTX 5070 Ti, a bf16
    GEMM of 352 x 1024 x 2560; the dirty lines a memset leaves are written back on either
    side of the call)."""
    global _L2_FLUSH
    if _L2_FLUSH is None:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        n = max(_PROBE_MIN_BYTES, 2 * int(getattr(props, "L2_cache_size", 0) or 0))
        _L2_FLUSH = torch.rand(n // 4, device="cuda")  # random: no compressible lines
    _L2_FLUSH.sum()


class GraphUnavailable(RuntimeError):
    """A function cannot be timed in a CUDA graph; the message is why (the first line of the
    capture's error: a host sync, a CPU tensor, ...)."""


def first_line(exc: BaseException) -> str:
    """``<type>: <the first line of the message>`` of an exception (at most 200 chars)."""
    text = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {text[0] if text else ''}"[:200]


def per_call_ms(graph_ms: list[float], flush_ms: list[float] | None, calls: int) -> list[float]:
    """Per-call times of graph replays: each replay's time over its ``calls`` calls, less
    the time of the flush-only replay measured next to it when the graph flushed the L2
    before every call (``flush_ms``, same length)."""
    if flush_ms is None:
        return [ms / calls for ms in graph_ms]
    return [(ms - f) / calls for ms, f in zip(graph_ms, flush_ms, strict=True)]


def _sleep_cycles_per_us() -> float:
    """``torch.cuda._sleep`` cycles per microsecond of this GPU (measured once)."""
    global _SLEEP_CYCLES_PER_US
    if not _SLEEP_CYCLES_PER_US:
        cycles = 1_000_000
        start, end = _events()
        _mark(start)
        _gpu_sleep(cycles)
        _mark(end)
        _synchronize()
        _SLEEP_CYCLES_PER_US = cycles / max(_elapsed(start, end) * 1000.0, 1.0)
    return _SLEEP_CYCLES_PER_US


def _ahead(host_us: float) -> None:
    """Keep the GPU busy while the host enqueues the next timed replay (twice the time a
    replay's launch took on the host, at least 50 us): its events then see GPU time only."""
    _gpu_sleep(int(_sleep_cycles_per_us() * max(50.0, 2.0 * host_us)))


def _capture(body: Callable[[], Any], stream: Any) -> tuple[Any, Any]:
    """``body()`` captured in a new CUDA graph (its own private memory pool) on ``stream``:
    the graph and what ``body`` returned. ``GraphUnavailable`` (the first line of ``body``'s
    own error, not the capture's follow-up) when it cannot be captured; the process stays
    usable."""
    if _CUDAGraph is None or _graph_capture is None:
        raise GraphUnavailable("this torch build has no CUDA graphs")
    failed: list[BaseException] = []

    def guarded() -> Any:
        try:
            return body()
        except Exception as exc:
            failed.append(exc)
            raise

    graph = _CUDAGraph()
    current = _current_stream()
    try:
        with _graph_capture(graph, stream=stream):
            out = guarded()
    except Exception as exc:
        # the capture's exit raised before it set the current stream back (torch 2.14)
        _set_stream(current)
        raise GraphUnavailable(first_line(failed[0] if failed else exc)) from exc
    return graph, out


def _warm_up(
    fn: Callable[..., Any],
    restore: Callable[[], None] | None,
    sets: list[tuple[Any, Any]],
    mutable: bool,
    calls: int,
    stream: Any,
) -> None:
    """``calls`` calls on ``stream`` before a capture (lazy initialisation, autotuning and
    workspaces then happen outside the graph): on the input sets in turn, or on fresh deep
    copies of inputs with mutable state, each from the case's module state."""
    stream.wait_stream(_current_stream())
    with _on_stream(stream):
        for i in range(calls):
            if restore is not None:
                restore()
            a, k = copy.deepcopy(sets[0]) if mutable else sets[i % len(sets)]
            fn(*a, **k)
    _current_stream().wait_stream(stream)
    _synchronize()


def _layout(value: Any) -> dict[str, torch.Tensor]:
    """The tensors of a call's ``(args, kwargs)`` by name (:func:`kernels.compare.flatten`:
    ``args[0]``, ``kwargs.past_key_values.layers[0].keys``, ...)."""
    from kernel_agent.kernels.compare import flatten

    return {
        name.replace("in[0]", "args", 1).replace("in[1]", "kwargs", 1): t
        for name, t in flatten(value, "in").items()
    }


def _replaced(before: dict[str, torch.Tensor], value: Any) -> str | None:
    """Why inputs with mutable state cannot be replayed from a graph: a call replaced one of
    their tensors (or added one) instead of writing into it, so a replay would read the
    tensors captured, not the inputs' current ones (None: every tensor is where it was)."""
    for name, t in _layout(value).items():
        if before.get(name) is not t:
            return (
                f"the call replaces `{name}` of its inputs instead of "
                "updating it in place (e.g. a cache grown with torch.cat): a CUDA graph "
                "replays the tensors it captured"
            )
    return None


def _graph_copies(args: Any, kwargs: Any, calls: int) -> int:
    """Calls per graph for inputs with mutable state (one deep copy each): at most
    ``calls``, within :data:`GRAPH_COPY_SHARE` of the free GPU memory."""
    size = sum(t.nbytes for t in _layout((args, kwargs)).values() if t.is_cuda)
    if size <= 0:
        return calls
    free, _ = _mem_get_info()
    return max(1, min(calls, int(GRAPH_COPY_SHARE * free // size)))


def _restorer(
    pristine: tuple[Any, Any], layouts: list[dict[str, torch.Tensor]], stream: Any
) -> Callable[[], None]:
    """Writes the values of ``pristine`` into the tensors of ``layouts`` (:func:`_layout` of
    its deep copies: the inputs of a graph's calls, which write into them): one CUDA graph
    of the copies, replayed outside the timed region (eager copies if it cannot be
    captured)."""
    src = _layout(pristine)
    pairs = [
        (dst, src[name])
        for layout in layouts
        for name, dst in layout.items()
        if name in src and dst.shape == src[name].shape and dst.dtype == src[name].dtype
    ]

    def copy_all() -> None:
        for dst, value in pairs:
            _copy(dst, value)

    if not pairs:
        return lambda: None
    try:
        graph, _ = _capture(copy_all, stream)
    except GraphUnavailable:
        return copy_all
    replay = _replay
    assert replay is not None
    return lambda: replay(graph)


def graph_probe(
    fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> str | None:
    """Why ``fn(*args, **kwargs)`` cannot be timed in a CUDA graph (None: it can): one call
    on a side stream, then one captured (on a copy of inputs with mutable state, which must
    keep their tensors; from the case's module state), never replayed. An error of the
    eager call is raised: the function fails, not its capture."""
    fn, restore = split_state(fn)
    mutable = has_mutable_state(args, kwargs)
    side = _Stream()
    try:
        with torch.inference_mode():
            _warm_up(fn, restore, [(args, kwargs)], mutable, 1, side)
            a, k = copy.deepcopy((args, kwargs)) if mutable else (args, kwargs)
            before = _layout((a, k))
            if restore is not None:
                restore()
            _capture(lambda: fn(*a, **k), side)
            if mutable and (why := _replaced(before, (a, k))) is not None:
                return why
    except GraphUnavailable as exc:
        return str(exc)
    finally:
        _synchronize()
    return None


def _snapshot(value: Any) -> Any:
    """Deep copy of call inputs/outputs (the object itself if it cannot be copied)."""
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _redraw(value: Any) -> None:
    """Redraw the floating-point tensors of call inputs in place (:func:`kernels.verify.
    perturb_`, a random seed)."""
    from kernel_agent.kernels.verify import perturb_

    gen = torch.Generator(device="cuda")
    gen.manual_seed(random.SystemRandom().getrandbits(63))
    perturb_(value, gen, "normal")


def _perturbed_copy(args: Any, kwargs: Any) -> tuple[Any, Any]:
    a, k = copy.deepcopy(args), copy.deepcopy(kwargs)
    _redraw((a, k))
    return a, k


def time_call(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    warmup: int = 5,
    min_iters: int = 20,
    max_iters: int = 300,
    target_ms: float = 150.0,
    l2_flush: bool = False,
    input_sets: int = 3,
    keep: bool = False,
    context: str = EAGER,
) -> dict[str, Any]:
    """Median GPU time of ``fn(*args, **kwargs)`` in milliseconds.

    Mutable inputs (caches) get a fresh deep copy per iteration; the copy is
    made outside the timed region.  Immutable inputs rotate between
    ``input_sets`` sets (``args`` itself and deep copies made before timing), so
    an output cached by input address misses.  With ``keep=True`` one randomly
    chosen timed iteration runs on a copy of its inputs with redrawn values
    (:func:`kernels.verify.perturb_`; an output cached by address, shape or
    call count, or left over in reused memory, no longer matches) and is
    recorded in ``result["kept"]``: its inputs before and after the call and its
    output (copies made outside the timed region).

    The measurement runs after :func:`ensure_clocks` and before one more clock probe, and
    is repeated (at most CLOCK_RETRIES times) when either reads below CLOCK_OK: ``clock``
    is the lower of the two of the measurement returned (about 1: full clocks),
    ``clock_retries`` the repeats.

    An entrypoint bound to a case of a stateful module (:class:`kernel_agent.profiling.
    state.StatefulCall`) gets the case's module state back before every call, outside the
    timed region like the copies of mutable inputs.

    ``context="graph"``: the calls are captured in a CUDA graph and timed as its replays
    (:func:`_measure_graph`); ``GraphUnavailable`` when they cannot be captured.
    """
    if context not in CONTEXTS:
        raise ValueError(f"timing context {context!r} is not one of {CONTEXTS}")
    fn, restore = split_state(fn)
    mutable = has_mutable_state(args, kwargs)
    sets = [(args, kwargs)]
    if not mutable:
        sets += [copy.deepcopy((args, kwargs)) for _ in range(max(input_sets, 1) - 1)]
    turn = 0

    def fresh() -> tuple[tuple[Any, ...], dict[str, Any]]:
        nonlocal turn
        if restore is not None:
            restore()
        if mutable:
            return copy.deepcopy(args), copy.deepcopy(kwargs)
        turn += 1
        return sets[(turn - 1) % len(sets)]

    def measure() -> dict[str, Any]:
        if context == GRAPH:
            return _measure_graph(
                fn,
                restore,
                sets,
                mutable,
                warmup=warmup,
                target_ms=target_ms,
                l2_flush=l2_flush,
                keep=keep,
            )
        return _measure(
            fn,
            fresh,
            warmup=warmup,
            min_iters=min_iters,
            max_iters=max_iters,
            target_ms=target_ms,
            l2_flush=l2_flush,
            keep=keep,
        )

    retries = 0
    while True:
        before = ensure_clocks()
        result = measure()
        probes = [s for s in (before, clock_state()) if s is not None]
        clock = min(probes, default=None)
        if clock is None or clock >= CLOCK_OK or retries == CLOCK_RETRIES:
            break
        retries += 1
    if clock is not None:
        result["clock"] = round(clock, 3)
    if retries:
        result["clock_retries"] = retries
    return result


def _measure(
    fn: Callable[..., Any],
    fresh: Callable[[], tuple[tuple[Any, ...], dict[str, Any]]],
    *,
    warmup: int,
    min_iters: int,
    max_iters: int,
    target_ms: float,
    l2_flush: bool,
    keep: bool,
) -> dict[str, Any]:
    """One measurement of :func:`time_call` (inputs from ``fresh()``)."""
    kept: dict[str, Any] | None = None
    with torch.inference_mode():
        for _ in range(warmup):
            a, k = fresh()
            fn(*a, **k)
        _synchronize()

        # Estimate per-call time to choose the iteration count.
        a, k = fresh()
        start, end = _events()
        _mark(start)
        fn(*a, **k)
        _mark(end)
        _synchronize()
        est = max(_elapsed(start, end), 1e-3)
        iters = int(min(max(target_ms / est, min_iters), max_iters))
        checked = random.SystemRandom().randrange(iters) if keep else -1

        events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        for i in range(iters):
            a, k = fresh()
            if i == checked:
                a, k = _perturbed_copy(a, k)
                pre = _snapshot((a, k))
                _synchronize()
            if l2_flush:
                # waited for: still running, the flush would hide the call's host time,
                # which an eager context pays (#226: an FP8 GEMM of two Triton launches
                # measured 1.97x against cuBLAS so, 0.58x warm and 0.66x waited for)
                flush_l2()
                _synchronize()
            start, end = _events()
            _mark(start)
            out = fn(*a, **k)
            _mark(end)
            events.append((start, end))
            if i == checked:
                kept = {
                    "iteration": i,
                    "pre": pre,
                    "post": _snapshot((a, k)),
                    "output": _snapshot(out),
                }
            del out
        _synchronize()
        ms = [_elapsed(s, e) for s, e in events]
    result = _summary(ms)
    if kept is not None:
        result["kept"] = kept
    return result


def _summary(ms: list[float]) -> dict[str, Any]:
    """Median, trimmed mean (the fastest 90 %), minimum and count of per-call times."""
    ms = sorted(ms)
    trimmed = ms[: max(1, int(len(ms) * 0.9))]
    return {
        "median_ms": statistics.median(ms),
        "mean_ms": statistics.fmean(trimmed),
        "min_ms": ms[0],
        "iters": len(ms),
    }


def _measure_graph(
    fn: Callable[..., Any],
    restore: Callable[[], None] | None,
    sets: list[tuple[Any, Any]],
    mutable: bool,
    *,
    warmup: int,
    target_ms: float,
    l2_flush: bool,
    keep: bool,
) -> dict[str, Any]:
    """One measurement of :func:`time_call` in a CUDA graph (``context="graph"``).

    :data:`GRAPH_CALLS` calls of ``fn`` are captured in one graph (its own private memory
    pool, on a side stream, after a warm-up there: lazy initialisation, autotuning and
    workspaces happen before the capture), rotating the input ``sets``; inputs with mutable
    state get one deep copy per call instead, written back from ``sets[0]`` by a restore
    graph before every replay. The case's module state (``restore``) is written back before
    every replay too, so the calls of one replay share it, as the calls of a graphed stage
    do. Each replay is timed with CUDA events while the GPU is kept busy during its launch
    (:func:`_ahead`): its time is the GPU's, and per call it is the replay's over its calls.

    With ``l2_flush`` the graph evicts the L2 before every call (:func:`flush_l2`) and a
    graph of the flushes alone, replayed after each timed replay, is subtracted
    (:func:`per_call_ms`). With ``keep`` one call of the graph (the first for a stateful
    module: the only one that runs from the case's state) has inputs of its own; before one
    random replay they are redrawn in place (:func:`kernels.verify.perturb_`: the buffers the
    graph reads), and that call's inputs and output are kept as :func:`_measure` keeps them.
    """
    if _replay is None:
        raise GraphUnavailable("this torch build has no CUDA graphs")
    replay = _replay
    args0, kwargs0 = sets[0]
    calls = _graph_copies(args0, kwargs0, GRAPH_CALLS) if mutable else GRAPH_CALLS
    rng = random.SystemRandom()
    checked = (0 if restore is not None else rng.randrange(calls)) if keep else -1
    side = _Stream()
    kept: dict[str, Any] | None = None
    with torch.inference_mode():
        if l2_flush:
            flush_l2()  # its buffer exists before the capture (not in the graph's pool)
        _sleep_cycles_per_us()
        _warm_up(fn, restore, sets, mutable, max(warmup, 1), side)
        if mutable:  # made after the warm-up: as captured, never called
            inputs = [copy.deepcopy((args0, kwargs0)) for _ in range(calls)]
        else:
            inputs = [sets[j % len(sets)] for j in range(calls)]
            if keep:
                inputs[checked] = copy.deepcopy(sets[0])
        layout = [_layout(x) for x in inputs] if mutable else []
        reset = _restorer(sets[0], layout, side) if mutable else None

        def prepare() -> None:
            if reset is not None:
                reset()
            if restore is not None:
                restore()

        def body() -> Any:
            out = None
            for j, (a, k) in enumerate(inputs):
                if l2_flush:
                    flush_l2()
                result = fn(*a, **k)
                if j == checked:  # kept alive: the pool does not reuse it for later calls
                    out = result
                del result
            return out

        prepare()
        graph, output = _capture(body, side)
        for before, now in zip(layout, inputs if mutable else [], strict=True):
            if (why := _replaced(before, now)) is not None:
                raise GraphUnavailable(why)
        flushes = None
        if l2_flush:

            def flush_only() -> None:
                for _ in range(calls):
                    flush_l2()

            flushes, _ = _capture(flush_only, side)

        def timed(g: Any, host_us: float) -> tuple[Any, Any]:
            _ahead(host_us)
            start, end = _events()
            _mark(start)
            replay(g)
            _mark(end)
            return start, end

        # two replays of each first: their host launch time (the second's: the first one
        # uploads the graph) and the GPU time of the timed one
        hosts: list[float] = []
        for g in (graph, flushes):
            if g is None:
                continue
            launch = []
            for _ in range(2):
                prepare()
                t0 = _perf_counter()
                replay(g)
                launch.append((_perf_counter() - t0) * 1e6)
                _synchronize()
            hosts.append(launch[-1])
        prepare()
        start, end = timed(graph, hosts[0])
        _synchronize()
        est = max(_elapsed(start, end), 1e-3)
        replays = int(min(max(target_ms / est, GRAPH_MIN_REPLAYS), GRAPH_MAX_REPLAYS))
        check_at = rng.randrange(replays) if keep else -1
        events: list[tuple[Any, Any]] = []
        flush_events: list[tuple[Any, Any]] = []
        for r in range(replays):
            prepare()
            if r == check_at:
                _redraw(inputs[checked])
                pre = _snapshot(inputs[checked])
            events.append(timed(graph, hosts[0]))
            if r == check_at:
                kept = {
                    "iteration": r * calls + checked,
                    "pre": pre,
                    "post": _snapshot(inputs[checked]),
                    "output": _snapshot(output),
                }
            if flushes is not None:
                flush_events.append(timed(flushes, hosts[1]))
        _synchronize()
        graph_ms = [_elapsed(s, e) for s, e in events]
        flush_ms = [_elapsed(s, e) for s, e in flush_events] if flushes is not None else None
        del graph, flushes, output  # their private pools go back to the allocator
    result = _summary(per_call_ms(graph_ms, flush_ms, calls))
    result.update(context=GRAPH, iters=replays * calls, calls_per_graph=calls, replays=replays)
    if flush_ms:
        result["flush_ms"] = statistics.median(flush_ms) / calls
    if kept is not None:
        result["kept"] = kept
    return result


def wall_check(
    reference: Callable[..., Any],
    candidate: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    iters: int = 10,
) -> dict[str, float]:
    """Per-call CUDA-event time vs device-synchronised wall time (ms).

    The events only see the current stream; the wall clock (device-wide
    ``synchronize()`` before and after each call) sees every stream and thread.
    Each iteration calls the reference and the candidate once, in random order;
    ``hidden_ms`` is the median over iterations of the candidate's wall-minus-event
    gap minus the reference's (the fixed synchronisation overhead cancels): GPU
    work the timer does not see.  Pairing and the median keep a busy GPU (other
    processes) from looking like hidden work; ``*_wall_ms`` / ``*_event_ms`` are
    medians. A case's module state is restored before each call, outside the timing."""
    mutable = has_mutable_state(args, kwargs)
    walls: dict[str, list[float]] = {"ref": [], "new": []}
    evts: dict[str, list[float]] = {"ref": [], "new": []}
    order = [("ref", *split_state(reference)), ("new", *split_state(candidate))]
    rng = random.SystemRandom()
    with torch.inference_mode():
        for _ in range(iters):
            rng.shuffle(order)
            for label, fn, restore in order:
                a, k = (copy.deepcopy(args), copy.deepcopy(kwargs)) if mutable else (args, kwargs)
                if restore is not None:
                    restore()
                start, end = _events()
                _synchronize()
                t0 = _perf_counter()
                _mark(start)
                fn(*a, **k)
                _mark(end)
                _synchronize()
                walls[label].append((_perf_counter() - t0) * 1e3)
                evts[label].append(_elapsed(start, end))
    out: dict[str, float] = {}
    for label in ("ref", "new"):
        out[f"{label}_wall_ms"] = statistics.median(walls[label])
        out[f"{label}_event_ms"] = statistics.median(evts[label])
    gaps = {k: [w - e for w, e in zip(walls[k], evts[k], strict=True)] for k in walls}
    out["hidden_ms"] = statistics.median(
        n - r for n, r in zip(gaps["new"], gaps["ref"], strict=True)
    )
    return out


def peak_memory(
    reference: Callable[..., Any],
    candidate: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    calls: int = 2,
) -> dict[str, int]:
    """Peak GPU memory of one call of each (bytes, ``ref_bytes`` / ``new_bytes``): what the
    caching allocator holds at the call's peak above what it held right before it, the
    output included, the inputs (copied first when they hold mutable state) not. The
    smallest of ``calls`` calls each, after timing (workspaces, compiled kernels and
    tuned configs already in place). Memory the module keeps between calls (weights,
    caches it allocated at build time) is not part of it."""
    mutable = has_mutable_state(args, kwargs)
    peaks: dict[str, list[int]] = {"ref": [], "new": []}
    fns = (("ref", *split_state(reference)), ("new", *split_state(candidate)))
    with torch.inference_mode():
        for _ in range(calls):
            for label, fn, restore in fns:
                a, k = (copy.deepcopy(args), copy.deepcopy(kwargs)) if mutable else (args, kwargs)
                if restore is not None:  # a case's module state, before the measurement
                    restore()
                _synchronize()
                before = _allocated()
                _reset_peak()
                out = fn(*a, **k)
                _synchronize()
                peaks[label].append(max(0, _max_allocated() - before))
                del out, a, k
    return {"ref_bytes": min(peaks["ref"]), "new_bytes": min(peaks["new"])}


def check_timed_output(reference: Callable[..., Any], kept: dict[str, Any]) -> dict[str, Any]:
    """Compare a kept timed call of the candidate with a fresh reference call on the
    same inputs (outputs and in-place side effects; the inputs are redrawn, so a
    reduced-precision tier applies its bounds for redrawn inputs).  The reference runs
    after the candidate, on copies, so it cannot leave the expected output in freed
    memory."""
    from kernel_agent.kernels.compare import compare_side_effects, compare_structures

    pre_args, pre_kwargs = kept["pre"]
    ref_args, ref_kwargs = copy.deepcopy(pre_args), copy.deepcopy(pre_kwargs)
    with torch.inference_mode():
        expected = reference(*ref_args, **ref_kwargs)
    torch.cuda.synchronize()
    new_args, new_kwargs = kept["post"]
    checks = compare_structures(
        expected, kept["output"], "output", inputs=kept["pre"], perturbed=True
    )
    checks += compare_side_effects(pre_args, ref_args, new_args, "args", perturbed=True)
    checks += compare_side_effects(pre_kwargs, ref_kwargs, new_kwargs, "kwargs", perturbed=True)
    return {"iteration": kept["iteration"], "failures": [c for c in checks if not c.get("ok")]}


def median_round(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """The median of timing rounds (:func:`time_call` results), with ``spread``: the
    range of the round medians relative to it, and the lowest ``clock`` and all
    ``clock_retries`` of the rounds."""
    ordered = sorted(rounds, key=lambda r: r["median_ms"])
    best = dict(ordered[len(ordered) // 2])
    best["spread"] = (ordered[-1]["median_ms"] - ordered[0]["median_ms"]) / max(
        best["median_ms"], 1e-9
    )
    if clocks := [r["clock"] for r in rounds if "clock" in r]:
        best["clock"] = min(clocks)
    if retries := sum(r.get("clock_retries", 0) for r in rounds):
        best["clock_retries"] = retries
    return best


#: Interleaved timing rounds of :func:`compare_timing` (the evaluator's)
ROUNDS = 3


def compare_timing(
    reference: Callable[..., Any],
    candidate: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    rounds: int = ROUNDS,
    l2_flush: bool = False,
    verify: bool = True,
    context: str = EAGER,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Interleave reference/candidate timing rounds; report the median round.

    Interleaving cancels slow drifts (clock boost, thermal, background load)
    that would otherwise favour whichever function ran second; every round runs at
    full clocks (:func:`time_call`'s clock guard: a low performance state slows a
    bandwidth-bound function far more than a launch-bound one).  With ``verify``
    one random timed call of the candidate's last round runs on redrawn inputs
    and its output is checked against a fresh reference call on the same inputs
    (``candidate_result["timed_output"]``: the iteration and failing checks).
    ``context``: :func:`time_call`'s."""
    ref_rounds, new_rounds, timed = timing_rounds(
        reference,
        candidate,
        args,
        kwargs,
        rounds=rounds,
        l2_flush=l2_flush,
        verify=verify,
        context=context,
    )
    ref_t, new_t = median_round(ref_rounds), median_round(new_rounds)
    if timed is not None:
        new_t["timed_output"] = timed
    return ref_t, new_t


def timing_rounds(
    reference: Callable[..., Any],
    candidate: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    rounds: int,
    l2_flush: bool = False,
    verify: bool = True,
    context: str = EAGER,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    """The rounds of :func:`compare_timing` (reference's and candidate's :func:`time_call`
    results, in ``context``) and, with ``verify``, the check of the candidate's kept timed
    call of the last of them (None without ``verify``). The evaluator times a case in two
    such calls when it may stop early (``kernels/early.py``): the median of all its rounds
    is the result."""
    warm_gpu()
    ref_rounds: list[dict[str, Any]] = []
    new_rounds: list[dict[str, Any]] = []
    for i in range(rounds):
        ref_rounds.append(
            time_call(reference, args, kwargs, l2_flush=l2_flush, target_ms=60.0, context=context)
        )
        new_rounds.append(
            time_call(
                candidate,
                args,
                kwargs,
                l2_flush=l2_flush,
                target_ms=60.0,
                keep=verify and i == rounds - 1,
                context=context,
            )
        )
    kept = new_rounds[-1].pop("kept", None) if new_rounds else None
    return ref_rounds, new_rounds, check_timed_output(reference, kept) if kept else None
