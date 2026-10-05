"""GPU timing utilities."""

from __future__ import annotations

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


def warm_gpu(ms: float = 300.0) -> None:
    """Spin the GPU (and CPU launch path) so clocks leave idle states before timing."""
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
    range of the round medians relative to it."""
    ordered = sorted(rounds, key=lambda r: r["median_ms"])
    best = dict(ordered[len(ordered) // 2])
    best["spread"] = (ordered[-1]["median_ms"] - ordered[0]["median_ms"]) / max(
        best["median_ms"], 1e-9
    )
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
    that would otherwise favour whichever function ran second.  With ``verify``
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
