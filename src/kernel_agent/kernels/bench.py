"""GPU timing utilities."""

from __future__ import annotations

import contextlib
import copy
import random
import statistics
import time
from collections.abc import Callable
from typing import Any

import torch

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
    global _L2_FLUSH
    if _L2_FLUSH is None:
        _L2_FLUSH = torch.empty(64 * 1024 * 1024, dtype=torch.int8, device="cuda")
    _L2_FLUSH.zero_()


def _snapshot(value: Any) -> Any:
    """Deep copy of call inputs/outputs (the object itself if it cannot be copied)."""
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _perturbed_copy(args: Any, kwargs: Any) -> tuple[Any, Any]:
    from kernel_agent.kernels.verify import perturb_

    a, k = copy.deepcopy(args), copy.deepcopy(kwargs)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(random.SystemRandom().getrandbits(63))
    perturb_((a, k), gen, "normal")
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
    """
    mutable = has_mutable_state(args, kwargs)
    sets = [(args, kwargs)]
    if not mutable:
        sets += [copy.deepcopy((args, kwargs)) for _ in range(max(input_sets, 1) - 1)]
    turn = 0

    def fresh() -> tuple[tuple[Any, ...], dict[str, Any]]:
        nonlocal turn
        if mutable:
            return copy.deepcopy(args), copy.deepcopy(kwargs)
        turn += 1
        return sets[(turn - 1) % len(sets)]

    retries = 0
    while True:
        before = ensure_clocks()
        result = _measure(
            fn,
            fresh,
            warmup=warmup,
            min_iters=min_iters,
            max_iters=max_iters,
            target_ms=target_ms,
            l2_flush=l2_flush,
            keep=keep,
        )
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
                flush_l2()
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
    ms.sort()
    trimmed = ms[: max(1, int(len(ms) * 0.9))]
    result: dict[str, Any] = {
        "median_ms": statistics.median(ms),
        "mean_ms": statistics.fmean(trimmed),
        "min_ms": ms[0],
        "iters": len(ms),
    }
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
    medians."""
    mutable = has_mutable_state(args, kwargs)
    walls: dict[str, list[float]] = {"ref": [], "new": []}
    evts: dict[str, list[float]] = {"ref": [], "new": []}
    order = [("ref", reference), ("new", candidate)]
    rng = random.SystemRandom()
    with torch.inference_mode():
        for _ in range(iters):
            rng.shuffle(order)
            for label, fn in order:
                a, k = (copy.deepcopy(args), copy.deepcopy(kwargs)) if mutable else (args, kwargs)
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


def check_timed_output(reference: Callable[..., Any], kept: dict[str, Any]) -> dict[str, Any]:
    """Compare a kept timed call of the candidate with a fresh reference call on the
    same inputs (outputs and in-place side effects).  The reference runs after the
    candidate, on copies, so it cannot leave the expected output in freed memory."""
    from kernel_agent.kernels.compare import compare_side_effects, compare_structures

    pre_args, pre_kwargs = kept["pre"]
    ref_args, ref_kwargs = copy.deepcopy(pre_args), copy.deepcopy(pre_kwargs)
    with torch.inference_mode():
        expected = reference(*ref_args, **ref_kwargs)
    torch.cuda.synchronize()
    new_args, new_kwargs = kept["post"]
    checks = compare_structures(expected, kept["output"], "output")
    checks += compare_side_effects(pre_args, ref_args, new_args, "args")
    checks += compare_side_effects(pre_kwargs, ref_kwargs, new_kwargs, "kwargs")
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


def compare_timing(
    reference: Callable[..., Any],
    candidate: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    rounds: int = 3,
    l2_flush: bool = False,
    verify: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Interleave reference/candidate timing rounds; report the median round.

    Interleaving cancels slow drifts (clock boost, thermal, background load)
    that would otherwise favour whichever function ran second; every round runs at
    full clocks (:func:`time_call`'s clock guard: a low performance state slows a
    bandwidth-bound function far more than a launch-bound one).  With ``verify``
    one random timed call of the candidate's last round runs on redrawn inputs
    and its output is checked against a fresh reference call on the same inputs
    (``candidate_result["timed_output"]``: the iteration and failing checks)."""
    warm_gpu()
    ref_rounds: list[dict[str, Any]] = []
    new_rounds: list[dict[str, Any]] = []
    for i in range(rounds):
        ref_rounds.append(time_call(reference, args, kwargs, l2_flush=l2_flush, target_ms=60.0))
        new_rounds.append(
            time_call(
                candidate,
                args,
                kwargs,
                l2_flush=l2_flush,
                target_ms=60.0,
                keep=verify and i == rounds - 1,
            )
        )
    kept = new_rounds[-1].pop("kept", None)
    ref_t, new_t = median_round(ref_rounds), median_round(new_rounds)
    if kept is not None:
        new_t["timed_output"] = check_timed_output(reference, kept)
    return ref_t, new_t
