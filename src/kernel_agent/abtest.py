"""Paired A/B acceptance of integration steps.

A 1 % faster 5-run median from another process is not evidence of a gain:
process-to-process variance on a consumer GPU is often larger than that. The
integration therefore compares the accepted set A with B (A plus one item, or
the systems agent's measured combination) in one process (``worker e2e_ab``,
``integrate/undo.py``): after warm-up A and B alternate for ``rounds`` rounds
(A B, B A, A B, ...), every run timed, and B is accepted when

* B is faster in at least ``min_win_rate`` (80 %) of the rounds, and
* the lower bound of the bootstrap 95 % confidence interval of the relative
  gain ``1 - sum(B) / sum(A)`` (rounds resampled with replacement) is above
  ``min_gain`` (1 %).

When a state cannot be undone in-process, A and B are measured in two
processes back to back with :data:`SEPARATE_ITERS` timed runs each (A measured
again in the same session): the win rate is then the share of (A run, B run)
pairs that B wins, the gain ``1 - median(B) / median(A)``, and its interval
comes from resampling both sets of runs.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Sequence
from typing import Any

ROUNDS = 8
MIN_WIN_RATE = 0.8
MIN_GAIN = 0.01
#: Timed runs per process when A and B are measured in separate processes.
SEPARATE_ITERS = 10
RESAMPLES = 10_000
#: ``e2e_ab`` statuses after which the integration measures in separate processes: a state
#: cannot be undone in-process, a switch did not restore it, or the A/B process itself
#: failed (two states in memory, say); a real failure of B shows again in its own process.
FALLBACK = ("irreversible", "undo_failed", "crash", "error")


def _quantile(values: list[float], q: float) -> float:
    """Linear-interpolated quantile of sorted ``values``."""
    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def _stats(a: Sequence[float], b: Sequence[float], gain: float, boot: list[float]) -> dict:
    boot.sort()
    return {
        "a_median_ms": round(statistics.median(a), 3),
        "b_median_ms": round(statistics.median(b), 3),
        "gain": round(gain, 5),
        "ci95": [round(_quantile(boot, 0.025), 5), round(_quantile(boot, 0.975), 5)],
    }


def paired(
    a_ms: Sequence[float], b_ms: Sequence[float], *, resamples: int = RESAMPLES, seed: int = 0
) -> dict[str, Any]:
    """Statistics of ``rounds`` paired runs (``a_ms[i]`` and ``b_ms[i]`` of round ``i``).

    ``gain`` is ``1 - sum(B) / sum(A)`` (positive: B is faster), ``ci95`` its
    percentile bootstrap interval over resampled rounds, ``wins`` the rounds B won."""
    if len(a_ms) != len(b_ms) or not a_ms:
        raise ValueError("paired timings need the same, non-zero number of A and B runs")
    n = len(a_ms)
    rng = random.Random(seed)
    boot = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        boot.append(1.0 - sum(b_ms[i] for i in idx) / sum(a_ms[i] for i in idx))
    wins = sum(b < a for a, b in zip(a_ms, b_ms, strict=True))
    return {
        "mode": "paired",
        "rounds": n,
        "wins": wins,
        "win_rate": round(wins / n, 4),
        **_stats(a_ms, b_ms, 1.0 - sum(b_ms) / sum(a_ms), boot),
    }


def separate(
    a_ms: Sequence[float], b_ms: Sequence[float], *, resamples: int = RESAMPLES, seed: int = 0
) -> dict[str, Any]:
    """Statistics of unpaired runs of A and B (two processes): ``gain`` is
    ``1 - median(B) / median(A)``, ``ci95`` its bootstrap interval (both sets
    resampled), ``win_rate`` the share of (A run, B run) pairs B wins (ties: half)."""
    if not a_ms or not b_ms:
        raise ValueError("separate timings need runs of A and of B")
    rng = random.Random(seed)
    boot = []
    for _ in range(resamples):
        a = statistics.median(rng.choices(a_ms, k=len(a_ms)))
        b = statistics.median(rng.choices(b_ms, k=len(b_ms)))
        boot.append(1.0 - b / a)
    score = sum((b < a) + 0.5 * (b == a) for a in a_ms for b in b_ms)
    return {
        "mode": "separate",
        "rounds": min(len(a_ms), len(b_ms)),
        "wins": None,
        "win_rate": round(score / (len(a_ms) * len(b_ms)), 4),
        **_stats(a_ms, b_ms, 1.0 - statistics.median(b_ms) / statistics.median(a_ms), boot),
    }


def stats(ab: dict[str, Any]) -> dict[str, Any]:
    """:func:`paired` or :func:`separate` statistics of an ``ab`` record's ``a_ms`` / ``b_ms``."""
    fn = separate if ab.get("mode") == "separate" else paired
    return fn([float(t) for t in ab["a_ms"]], [float(t) for t in ab["b_ms"]])


def _pct(x: float, sign: bool = True) -> str:
    """``+3.1 %`` (the spacing of the rest of kernel-agent)."""
    return f"{100 * x:+.1f} %" if sign else f"{100 * x:.0f} %"


def decide(
    ab: dict[str, Any], *, min_win_rate: float = MIN_WIN_RATE, min_gain: float = MIN_GAIN
) -> tuple[bool, str]:
    """``(accepted, why not)`` for an ``ab`` record with statistics (:func:`stats`)."""
    if not ab.get("rounds"):
        return False, "no timed rounds"
    rate, lo = float(ab["win_rate"]), float(ab["ci95"][0])
    if rate < min_win_rate:
        won = (
            f"{ab['wins']}/{ab['rounds']} rounds"
            if ab.get("wins") is not None
            else _pct(rate, False)
        )
        return False, f"B won {won} (needs {_pct(min_win_rate, False)})"
    if lo <= min_gain:
        return False, f"95 % CI of the gain starts at {_pct(lo)} (needs > {_pct(min_gain, False)})"
    return True, ""


def judge(
    ab: dict[str, Any], *, min_win_rate: float = MIN_WIN_RATE, min_gain: float = MIN_GAIN
) -> dict[str, Any]:
    """``ab`` with fresh statistics and the decision (``accepted``, ``why``, ``rule``)."""
    out = {**ab, **stats(ab)}
    accepted, why = decide(out, min_win_rate=min_win_rate, min_gain=min_gain)
    return {
        **out,
        "accepted": accepted,
        "why": why,
        "rule": {"min_win_rate": min_win_rate, "min_gain": min_gain},
    }


def describe(ab: dict[str, Any]) -> str:
    """One line: ``paired A/B: B won 7/8 rounds, gain +3.1 % (95 % CI +1.9 .. +4.2 %)``."""
    if not ab.get("rounds"):
        return f"{ab.get('mode', 'paired')} A/B: no timed rounds"
    if ab.get("wins") is not None:
        won = f"B won {ab['wins']}/{ab['rounds']} rounds"
    else:
        won = f"B won {_pct(float(ab['win_rate']), False)} of run pairs"
    lo, hi = ab["ci95"]
    return (
        f"{ab['mode']} A/B: {won}, gain {_pct(float(ab['gain']))} "
        f"(95 % CI {_pct(float(lo))} .. {_pct(float(hi))})"
    )
