"""Early termination of hopeless measurements in the kernel evaluator and the sweeps
(docs/MULTIAGENT.md §3.12.5, issue #190; ``--early-stop off`` turns it off).

A measurement branch may be cut early, but only with evidence, and never its correctness
checks: the cut only skips more timing of a clear loser, and the GPU time goes to the next
job of the queue. Nothing here ends a target or a run.

Both rules rest on one property of the evaluator's timing (``bench.median_round``: the
median of R interleaved rounds): once :func:`min_rounds` of the R rounds ran (2 of 3), the
median of all R is at most the largest and at least the smallest of them, whatever the
remaining rounds measure. So the rounds so far bound the final result.

* **Early discard** (:func:`discard`, ``kernels/evaluate.py``): after :func:`min_rounds` rounds
  of every timed case, the highest speedup the remaining rounds could still give a correct
  candidate is ``Σ n·max(reference) / Σ n·min(candidate)`` (``n``: the case's calls per
  run). When that bound, times ``1 + noise`` (``max(1 %, 2 × timing spread)``, as the keep
  rule), is below the bar (the target's best kept speedup, at least the reference's 1.0:
  what a ``keep`` must beat) the full result could only be a ``discard`` too, so the
  evaluator stops timing it and records ``early`` (the bound, the bar, the noise). An early
  discard therefore never turns away a winner.
* **Racing** (:func:`race`, ``kernels/sweep.py``): successive halving of a sweep's configs.
  After every round but the last, the configs whose fastest possible total time is beyond
  the noise of the leader's slowest possible one drop out, at most half of those still timed
  per round. After the first round (one sample each) "beyond the noise" means more than
  :data:`FIRST_ROUND_NOISE` slower (above the largest timing spread of our recorded
  evaluations); from :func:`min_rounds` on the bounds above hold and the noise is the keep
  rule's. The leader is never dropped; the winner still gets the full evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: The keep rule's floor of the noise (``budget.MIN_GAIN``)
NOISE_FLOOR = 0.01
#: Racing: how much slower than the leader a config must be after its first round to drop out
#: (the largest timing spread of a case in our runs' evaluations with the clock guard: 49.5 %)
FIRST_ROUND_NOISE = 0.5


@dataclass
class Timed:
    """One timed case so far: its calls per run (``count``, the weight of its time) and the
    round medians (ms) of the reference (``ref``) and of the candidate or config (``new``)."""

    count: float
    ref: list[float] = field(default_factory=list)
    new: list[float] = field(default_factory=list)


def min_rounds(rounds: int) -> int:
    """The rounds after which the median of ``rounds`` rounds (the upper median,
    ``bench.median_round``) lies between their smallest and largest value: 2 of 3."""
    return max(rounds // 2 + 1, rounds - rounds // 2)


def spread(values: list[float]) -> float:
    """``bench.median_round``'s ``spread``: range / median of round medians (0: one round)."""
    if len(values) < 2:
        return 0.0
    ordered = sorted(values)
    return (ordered[-1] - ordered[0]) / max(ordered[len(ordered) // 2], 1e-9)


def noise(cases: list[Timed]) -> float:
    """The keep rule's noise of measurements so far: ``max(1 %, 2 × the largest spread)``."""
    worst = max((max(spread(c.ref), spread(c.new)) for c in cases), default=0.0)
    return max(NOISE_FLOOR, 2 * worst)


def _total(cases: list[Timed], side: str, pick: Any) -> float:
    return sum(c.count * pick(getattr(c, side)) for c in cases)


def discard(cases: list[Timed], bar: float, rounds: int) -> dict[str, Any] | None:
    """The early discard of a correct candidate after the rounds of ``cases`` (of ``rounds``
    per case): ``{"rounds", "of", "bound", "bar", "noise", "why"}``, or None when it must be
    timed on (too few rounds yet, none left, or it could still reach ``bar``)."""
    done = min((min(len(c.ref), len(c.new)) for c in cases), default=0)
    if not cases or done < min_rounds(rounds) or done >= rounds:
        return None
    fastest = _total(cases, "new", min)
    if fastest <= 0:
        return None
    bound = _total(cases, "ref", max) / fastest
    margin = noise(cases)
    if bound * (1 + margin) >= bar:
        return None
    return {
        "rounds": done,
        "of": rounds,
        "bound": round(bound, 4),
        "bar": round(bar, 4),
        "noise": round(margin, 4),
        "why": f"after {done} of {rounds} timing rounds its speedup is at most {bound:.3f}x "
        f"whatever the last rounds measure, {bound * (1 + margin):.3f}x with the noise "
        f"({margin:.1%}): below {bar:.3f}x, the best so far (at least the reference), so "
        "it cannot be a new best; its timing stopped there, every correctness check ran",
    }


def race(configs: dict[int, list[Timed]], done: int, rounds: int) -> dict[int, dict[str, Any]]:
    """The configs of a sweep that drop out after ``done`` of ``rounds`` rounds (``configs``:
    the ones still timed, by index, their timed cases so far): ``{index: {"after",
    "leader", "behind", "why"}}``; the slowest first, at most half of ``configs``, never the
    leader (the least slowest possible total time)."""
    if len(configs) < 2 or done >= rounds:
        return {}
    slowest = {i: _total(cases, "new", max) for i, cases in configs.items()}
    leader = min(slowest, key=lambda i: (slowest[i], i))
    bounded = done >= min_rounds(rounds)
    margin = noise(configs[leader]) if bounded else FIRST_ROUND_NOISE
    limit = slowest[leader] * (1 + margin)
    behind = []
    for i, cases in configs.items():
        fastest = _total(cases, "new", min)
        if i != leader and fastest > limit:
            behind.append((fastest, i))
    behind.sort(reverse=True)
    out = {}
    for fastest, i in behind[: len(configs) // 2]:
        ratio = fastest / max(slowest[leader], 1e-12)
        out[i] = {
            "after": done,
            "leader": leader,
            "behind": round(ratio, 3),
            "why": f"after {done} of {rounds} rounds at least {ratio:.2f}x the time of "
            f"config {leader}, beyond the noise ({margin:.0%})",
        }
    return out
