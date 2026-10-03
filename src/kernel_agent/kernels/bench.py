"""GPU timing utilities."""

from __future__ import annotations

import copy
import statistics
from collections.abc import Callable
from typing import Any

import torch


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
) -> dict[str, float]:
    """Median GPU time of ``fn(*args, **kwargs)`` in milliseconds.

    Mutable inputs (caches) get a fresh deep copy per iteration; the copy is
    made outside the timed region.
    """
    mutable = has_mutable_state(args, kwargs)

    def fresh() -> tuple[tuple[Any, ...], dict[str, Any]]:
        if mutable:
            return copy.deepcopy(args), copy.deepcopy(kwargs)
        return args, kwargs

    with torch.inference_mode():
        for _ in range(warmup):
            a, k = fresh()
            fn(*a, **k)
        torch.cuda.synchronize()

        # Estimate per-call time to choose the iteration count.
        a, k = fresh()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*a, **k)
        end.record()
        torch.cuda.synchronize()
        est = max(start.elapsed_time(end), 1e-3)
        iters = int(min(max(target_ms / est, min_iters), max_iters))

        events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        for _ in range(iters):
            a, k = fresh()
            if l2_flush:
                flush_l2()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn(*a, **k)
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
        ms = [s.elapsed_time(e) for s, e in events]
    ms.sort()
    trimmed = ms[: max(1, int(len(ms) * 0.9))]
    return {
        "median_ms": statistics.median(ms),
        "mean_ms": statistics.fmean(trimmed),
        "min_ms": ms[0],
        "iters": len(ms),
    }


def compare_timing(
    reference: Callable[..., Any],
    candidate: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    rounds: int = 3,
    l2_flush: bool = False,
) -> tuple[dict[str, float], dict[str, float]]:
    """Interleave reference/candidate timing rounds; report the median round.

    Interleaving cancels slow drifts (clock boost, thermal, background load)
    that would otherwise favour whichever function ran second."""
    warm_gpu()
    ref_rounds: list[dict[str, float]] = []
    new_rounds: list[dict[str, float]] = []
    for _ in range(rounds):
        ref_rounds.append(time_call(reference, args, kwargs, l2_flush=l2_flush, target_ms=60.0))
        new_rounds.append(time_call(candidate, args, kwargs, l2_flush=l2_flush, target_ms=60.0))

    def pick(rs: list[dict[str, float]]) -> dict[str, float]:
        ordered = sorted(rs, key=lambda r: r["median_ms"])
        best = dict(ordered[len(ordered) // 2])
        best["spread"] = (ordered[-1]["median_ms"] - ordered[0]["median_ms"]) / max(
            best["median_ms"], 1e-9
        )
        return best

    return pick(ref_rounds), pick(new_rounds)
