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

When a state cannot be undone in-process, or A and B do not fit in one process
together (:data:`OOM`: out of GPU memory anywhere in the step, the timed runs, the
quality checks or the perceptual gate), A and B are measured in two processes back
to back with :data:`SEPARATE_ITERS` timed runs each (A measured again in the same
session): the win rate is then the share of (A run, B run) pairs that B wins,
the gain ``1 - median(B) / median(A)``, and its interval comes from resampling
both sets of runs.

**Sequential stop** (:func:`sequential`, ``--early-stop on``, docs/MULTIAGENT.md §3.12.5,
issue #190): from round :data:`SEQ_MIN_ROUNDS` on, the paired A/B stops before its last
round once the verdict of all ``rounds`` is no longer in doubt: B can no longer win
``min_win_rate`` of them (a reject whatever the rest brings), the :data:`SEQ_CONFIDENCE`
interval of the gain ends below ``min_gain`` (a reject), or B has won enough rounds that
the win rate holds whatever the rest brings and the interval starts above ``min_gain`` (an
accept). Every stop is also the verdict of :func:`decide` on the rounds that ran. Only the
timed rounds stop: B's quality checks always run in full. On the 221 paired A/Bs of our
runs (8 rounds each) it ran 33 % fewer rounds with the same verdict on every one
(``tests/fixtures/ab_rounds.json``).
"""

from __future__ import annotations

import math
import random
import re
import statistics
from collections.abc import Sequence
from typing import Any

ROUNDS = 8
MIN_WIN_RATE = 0.8
MIN_GAIN = 0.01
#: Timed runs per process when A and B are measured in separate processes.
SEPARATE_ITERS = 10
RESAMPLES = 10_000
#: Sequential stop (:func:`sequential`): the first round after which an A/B may stop, and
#: the (two-sided) confidence of the interval of the gain at a look before the last round
#: (stricter than the final 95 %: several looks).
SEQ_MIN_ROUNDS = 3
SEQ_CONFIDENCE = 0.99
#: ``e2e`` / ``e2e_ab`` status of a step that ran out of GPU memory: a property of what else
#: the process held (two states of the model in one A/B process), not of the items' code.
OOM = "oom"
#: ``e2e_ab`` statuses after which the integration measures in separate processes: a state
#: cannot be undone in-process, a switch did not restore it, the A/B process itself failed
#: or ran out of memory (two states in memory, say); a real failure of B shows again in its
#: own process.
FALLBACK = ("irreversible", "undo_failed", "crash", "error", OOM)
_OOM = re.compile(
    r"OutOfMemoryError|CUDA out of memory|CUDA error: out of memory|CUBLAS_STATUS_ALLOC_FAILED"
)
#: The checks of an ``e2e`` verdict that catch their own errors, a failed run being their
#: verdict (``metrics.<key>``: workloads/holdout.py, stopping.py, diverse.py, perceptual.py).
#: Out of GPU memory in one is no verdict either: the step is ``oom`` (#137).
CHECKS = {
    "holdout": "the held-out input",
    "natural_length": "the natural-length run",
    "diverse": "the diverse input set",
    "perceptual": "the perceptual gate",
}


def out_of_memory(error: object) -> str | None:
    """The line of ``error`` (an exception, a traceback, or a record whose ``error`` or
    ``reason`` holds one: a check of an ``e2e`` verdict, an integration step) that says the
    GPU ran out of memory; None when it is another error."""
    if isinstance(error, dict):
        return out_of_memory(error.get("error")) or out_of_memory(error.get("reason"))
    if isinstance(error, BaseException):
        error = f"{type(error).__name__}: {error}"
    for line in reversed(str(error or "").splitlines()):
        if _OOM.search(line):
            return line.strip()[:300]
    return None


def checks_out_of_memory(metrics: object) -> dict[str, str]:
    """``{key: line}`` of the :data:`CHECKS` in an ``e2e`` verdict's ``metrics`` that ran
    out of GPU memory (an older kernel-agent recorded that as their failure)."""
    if not isinstance(metrics, dict):
        return {}
    found = {key: out_of_memory(metrics.get(key)) for key in CHECKS}
    return {key: line for key, line in found.items() if line is not None}


def checks_reason(found: dict[str, str]) -> str:
    """The reason of a step whose checks ran out of GPU memory (:func:`checks_out_of_memory`)."""
    return "; ".join(f"out of GPU memory in {CHECKS[key]}: {line}" for key, line in found.items())


def step_out_of_memory(step: dict[str, Any]) -> str | None:
    """Why an integration step (an ``e2e`` / ``e2e_ab`` result, an ``integration.json``
    history entry) has no verdict because it ran out of GPU memory: status :data:`OOM`, a
    check that a kernel-agent before #137 recorded as failed instead, or A of its
    separate-process A/B (``ab.why``); None when it did not."""
    if step.get("status") == OOM:
        return str(step.get("reason") or "out of GPU memory")
    if found := checks_out_of_memory(step.get("metrics")):
        return checks_reason(found)
    why = (step.get("ab") or {}).get("why")
    return str(why) if out_of_memory(why) else None


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
    boot = _paired_boot(a_ms, b_ms, resamples, seed)
    n = len(a_ms)
    wins = sum(b < a for a, b in zip(a_ms, b_ms, strict=True))
    return {
        "mode": "paired",
        "rounds": n,
        "wins": wins,
        "win_rate": round(wins / n, 4),
        **_stats(a_ms, b_ms, 1.0 - sum(b_ms) / sum(a_ms), boot),
    }


def _paired_boot(
    a_ms: Sequence[float], b_ms: Sequence[float], resamples: int, seed: int
) -> list[float]:
    """The bootstrap distribution of the paired gain (rounds resampled with replacement)."""
    if len(a_ms) != len(b_ms) or not a_ms:
        raise ValueError("paired timings need the same, non-zero number of A and B runs")
    n = len(a_ms)
    rng = random.Random(seed)
    boot = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        boot.append(1.0 - sum(b_ms[i] for i in idx) / sum(a_ms[i] for i in idx))
    return boot


def _interval(
    a_ms: Sequence[float], b_ms: Sequence[float], confidence: float, resamples: int, seed: int = 0
) -> tuple[float, float]:
    """The percentile bootstrap interval of the paired gain at ``confidence`` (two-sided),
    rounds resampled with replacement (vectorised: a look of the sequential A/B)."""
    import numpy as np

    a, b = np.asarray(a_ms, dtype=float), np.asarray(b_ms, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(a), size=(resamples, len(a)))
    boot = 1.0 - b[idx].sum(axis=1) / a[idx].sum(axis=1)
    tail = (1.0 - confidence) / 2
    return float(np.quantile(boot, tail)), float(np.quantile(boot, 1.0 - tail))


def stop_rule(min_win_rate: float = MIN_WIN_RATE, min_gain: float = MIN_GAIN) -> dict[str, Any]:
    """What a sequential A/B's stop depends on besides its rounds (part of the integration's
    reuse keys, ``integrate/reuse.py``: a measurement that stopped early is reused only under
    the same rule)."""
    return {
        "sequential": 1,  # bump when the rule changes
        "min_rounds": SEQ_MIN_ROUNDS,
        "confidence": SEQ_CONFIDENCE,
        "min_win_rate": min_win_rate,
        "min_gain": min_gain,
    }


def sequential(
    a_ms: Sequence[float],
    b_ms: Sequence[float],
    rounds: int,
    *,
    min_win_rate: float = MIN_WIN_RATE,
    min_gain: float = MIN_GAIN,
    resamples: int = RESAMPLES,
) -> dict[str, Any] | None:
    """Whether a paired A/B of ``rounds`` rounds stops after the ``len(a_ms)`` it ran
    (``{"verdict": "accept" | "reject", "why", "rounds", "of"}``; None: it goes on).

    From round :data:`SEQ_MIN_ROUNDS` until the one before the last: a reject when B can no
    longer win ``min_win_rate`` of the ``rounds`` (exact) or the :data:`SEQ_CONFIDENCE`
    interval of the gain ends below ``min_gain``; an accept when B has won enough rounds for
    the win rate whatever the rest brings and the interval starts above ``min_gain``. A stop
    is always also the verdict of :func:`decide` on the rounds that ran."""
    n = len(a_ms)
    if n < SEQ_MIN_ROUNDS or n >= rounds:
        return None
    need = math.ceil(min_win_rate * rounds - 1e-9)
    wins = sum(b < a for a, b in zip(a_ms, b_ms, strict=True))
    out = {"rounds": n, "of": rounds}
    if wins + (rounds - n) < need:
        why = f"B won {wins}/{n} rounds: it can no longer win {need} of {rounds}"
        return {"verdict": "reject", "why": why, **out}
    lo, hi = _interval(a_ms, b_ms, SEQ_CONFIDENCE, resamples)
    level = f"{100 * SEQ_CONFIDENCE:.0f} % CI of the gain"
    if hi < min_gain:
        why = f"the {level} ends at {_pct(hi)}, below {_pct(min_gain, False)}"
        return {"verdict": "reject", "why": why, **out}
    if wins >= need and lo > min_gain:
        ab = paired(a_ms, b_ms, resamples=resamples)
        if decide(ab, min_win_rate=min_win_rate, min_gain=min_gain)[0]:
            why = f"B won {wins}/{n} rounds and the {level} starts at {_pct(lo)}"
            return {"verdict": "accept", "why": why, **out}
    return None


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
    stopped = ab.get("stopped") or {}  # the sequential stop (:func:`sequential`)
    early = f", stopped after {stopped['rounds']} of {stopped['of']}" if stopped else ""
    return (
        f"{ab['mode']} A/B: {won}{early}, gain {_pct(float(ab['gain']))} "
        f"(95 % CI {_pct(float(lo))} .. {_pct(float(hi))})"
    )
