"""Scheduler of ``kernel-agent improve``: which arm gets the next slice, and when an arm stops.

An *arm* is a kernel target (``targets/<id>``) or the systems agent
(:data:`SYSTEMS`: model-level transforms measured end to end). The arms are
rebuilt from the ledger, the profile(s) and the slice log before every
decision, so the scheduler keeps no state of its own and a restarted loop
decides exactly as an uninterrupted one would.

Expected gain, in the metric's ms (``-o metric=``, :mod:`kernel_agent.objective`:
per run for the latency, per second of generated audio for ``throughput``)::

    expected = remaining_ms × headroom × decay ** stale

A kernel target's comes from the ceilings table of the newest profile (issue #122:
an improve round's re-profile of the optimised model, ``profiling/ceilings.py``), whose
ms per profiled window × the round's baseline ÷ that window are the metric's
(:class:`Ceiling`, :func:`arm_ceiling`):

* ``now``: the time of the rows that hold the target's instance groups in that run.
  The module at a group's qualname; when the optimised model hides it inside a
  compiled or graph-replayed parent (the LocDiT layers inside VoxCPM2's CUDA-graphed
  ``UnifiedCFM``), that parent's rows: the part of them the group took in an older
  table that saw both (the analyze profile), else all of them (an upper bound).
* ``remaining_ms``: ``now``, or the target's best kernel per run when that is faster
  (not integrated yet: Σ its new ms × the calls each case stands for).
* ``headroom``: ``1 − floor / remaining_ms``, the floor of those rows at the target's
  precision (``ceilings.TARGET_PRECISIONS``; a precision pivot's arm: its new one).

A region target (and a target that names a ``fusion``) without a ceiling takes its expected
gain from the newest fusion table (issue #231, ``profiling/fusion.py``: the candidate of its
``fusion`` id, else the largest region candidate of its ``parent_class``; :class:`FusionGain`):
``remaining_ms`` as below, ``headroom`` = (its predicted saving − what its best kernel saved
so far) ÷ ``remaining_ms``, so ``expected`` is what the fusion still predicts, and never
below ``1 − 1 / MIN_FURTHER`` of the prediction (a kernel that already saved it all may
still find more).

Without a table, a row or a floor (a region target, unknown work, a peak not measured),
and for the systems agent (Amdahl):

* ``remaining_ms``: what the arm still costs. Kernel targets: their share of the
  profiled time × the baseline ms ÷ their best module speedup so far (a region target:
  its timed reference cases, ms per run converted to the metric's ms,
  ``projection.Units``). Systems: the end-to-end time of its best run so far
  (transforms, on top of kernels or not; a run is credited only for beating the best
  run with the same kernels).
* ``headroom``: the share of ``remaining_ms`` that could still go. ``1 −
  pct_of_sol`` when the best result carries a trustworthy speed-of-light
  estimate (``pct_of_sol``, :mod:`kernel_agent.kernels.roofline`), else ``1 − 1 / further`` where
  ``further = max(estimate / best, MIN_FURTHER)`` is the speedup still expected
  (``estimate``: the module speedup assumed reachable, :attr:`Policy.estimate`;
  for systems the end-to-end speedup the newest ceilings table allows at the run's
  precisions, the GPU-idle share of the profile or :attr:`Policy.systems_estimate`,
  whichever is largest).
* ``stale``: consecutive slices of this arm that found no new best.

:meth:`Arm.why` says how an arm's score came about (the slice log, ``improve.json``):
``share 66.8% of 7.73 ms: now 793 ms per batched run (...), W8A8 floor 258 ms → 3.48 ms``.

A kernel arm's rows are its benchmark evaluations (``ledger.measured``: quick
checks and duplicates count for nothing) of all its workers or islands (``workers.py``:
its move-on rules count across them, the same rules in evaluation units). The
integration's re-evaluations are not evaluations either, but replace a snapshot's
earlier result in the arm's best, as in the ledger's keep bar (``ledger.standing``). The best
is in the target's timing context (#226, ``ledger.in_context``): a result timed in another
one counts at its speedup in this one, or not at all.

UCB on the observed gain per evaluation (as in KernelBand): ``rate`` is the ms
per run an arm's kept results saved, divided by its evaluations. Each arm's
index is ``rate / best rate + explore × sqrt(2 ln(N + 2) / (n + 1))`` (``n``
its evaluations, ``N`` all evaluations), and ``score = expected × index``: the
live arm with the highest score gets the next slice. Untried arms get the
largest exploration bonus; arms that keep paying get more slices.

Move-on rules per arm (AutoKernel's, :func:`stop_reason`): ``patience``
consecutive evaluations without a new best (across slices), ≥ ``sol_stop`` of
its recipe's roofline (a best found in this round), ``target_hours`` spent in its
slices, or the module speedup ``speedup_goal`` reached (counted from its best at the
start of the round). They retire an arm for the current round only (issue #166): its
evaluations, slices and hours count from the round's start (``exp`` of the round
record, ``improve.json``), and when every arm is retired the loop starts a new round
(:mod:`kernel_agent.improve`), where an arm comes back once the round's profile shows it
matters again: its expected gain there is at least ``revive_share`` of the run
(:func:`_revival`). A kernel arm at a precision the run does not allow
(``--precisions``, :mod:`kernel_agent.precisions`: 4-bit unless named) is stopped for
good (``refused``) and has no ceiling; the systems agent's end-to-end estimate takes
only the allowed precisions' floors.

A kernel arm that has plateaued (:func:`plateau`) gets a research session
before the patience rule stops it (:mod:`kernel_agent.research`); a plan it
wrote restarts the arm's count of evaluations without a new best.

The native arm (:data:`NATIVE`, issue #134, :mod:`kernel_agent.native.engine`; only with
:attr:`Policy.native`: ``--native on``, or the plan asks for it) rewrites a stage, a group
of stages or the whole generation loop natively. It waits while any kernel arm is live and
has not plateaued (:func:`native_arm`), then competes like the others: ``remaining_ms`` is
the run at the best module-level result end to end (``native.engine.bar``: integrations and
the systems agent's runs), ``best`` its fastest native run over that bar, ``estimate``
:attr:`Policy.native_estimate`; its stop rules use ``native_patience`` and ``native_hours``.
Every stage of its staged plan beating the bar is a note (``Arm.note``), not a stop (issue
#164): it then works on the stage with the most headroom left in the newest ceilings table
and stops when every stage runs at ``sol_stop`` of its floor (``native.engine.at_floor``).
Native stage targets (``spec.native``) are its own: they get no kernel arm.

Time (issue #100): a slice of an arm needs the agent's warm-up, one evaluation
of the arm and the wrap-up (:func:`slice_seconds`), and the run's time budget
keeps the final integration's expected duration (:func:`integration_estimate`).
A slice that made no evaluation in a session the time budget cut short
(``budget_short`` in its record) counts neither as idle nor as stale.

Concurrent sessions (``improve --agents N``, :mod:`kernel_agent.coordinator`, issue #183):
:func:`assign` gives the free slots to arms one at a time, each by the scores of
:func:`virtual_pulls`: every running session counts as if it had made its expected
evaluations and found nothing (its arm's UCB ``n`` grows, one more step of ``decay``), so the
slots spread over the arms that pay instead of all going to the top one. An arm takes at
most ``max_sessions`` sessions at once (its live islands, ``--islands``: 1 without; which
island gets a slot is the island UCB's, ``workers.rank``), a role at most its
``max_concurrent`` (``roles.REGISTRY``, ``--role-max``), an agent name at most one, and no
new session while a GPU-free session of it (research, a dossier) runs (paused); it needs
time for a slice (:func:`slice_seconds`, the queue's expected wait included). There the
native arm is not held while kernel arms improve
(``build_arms(relax_native=True)``: :attr:`Arm.held`): it takes a slot no other arm can use.
A slice still running counts in no arm's ``stale`` or ``idle`` (its sessions' virtual
pulls do), and a finished one says what its own sessions' ledger rows did
(``improve.Improver._close``).
"""

from __future__ import annotations

import dataclasses
import math
import statistics
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kernel_agent import gpuqueue, ledger, precisions, projection, roles, truth
from kernel_agent.budget import (
    LIBRARY_HYPOTHESIS,
    PLATEAU,
    SOL_STOP_PCT,
    WRAP_UP_SECONDS,
    Standing,
    improves,
    not_agents,
)
from kernel_agent.kernels import context as timing_context
from kernel_agent.kernels import weights
from kernel_agent.kernels.roofline import sol_signal
from kernel_agent.native import engine as native_engine
from kernel_agent.profiling import ceilings
from kernel_agent.workspace import RunDir, read_json, read_jsonl

SYSTEMS = "systems"  # pseudo-target: the systems agent's slices
NATIVE = "native"  # pseudo-target: the systems-native agent's slices (native/engine.py)
KERNEL = "kernel"
MIN_FURTHER = 1.1  # without a SOL estimate, assume at least 10 % more is always possible
IDLE_SLICES = 2  # an arm whose last slices made no evaluation at all is stopped
FAIL_STREAK = 3  # failed evaluations in a row that call for a research session
# An agent reads its digest and NOTES.md and writes a candidate before its first evaluation:
# median 142-328 s from slice start to the first evaluation in three VoxCPM2 runs.
WARMUP_SECONDS = 240.0
# one evaluation of an arm with none timed yet (native: a project build + an end-to-end run)
EVAL_SECONDS = {KERNEL: 60.0, SYSTEMS: 120.0, NATIVE: 300.0}
AB_SECONDS = 2 * EVAL_SECONDS[SYSTEMS]  # one A/B of the integration: A and B end to end
SHORT_SLICE = 2.0  # a session with less than this × slice_seconds was cut short by the budget
# The time budget keeps at most this share of --max-hours for the final integration
# (``--integration-reserve auto``, issue #108): the agents keep two thirds of every
# invocation, and a longer final integration runs past --max-hours (on the GPU alone, no
# agent sessions). A re-integration that reuses (#93, #108) fits: in round 2 of the VoxCPM2
# run 20261006-004718 it was 12 A/B × 3.8 = 45 min of a 3 h invocation's 60, while the
# 113 min of a full one (30 A/B) left the agents 56 min of 180.
INTEGRATION_SHARE = 1 / 3


@dataclass(frozen=True)
class Policy:
    """Scheduler constants and per-arm move-on rules (None / 0 disables a rule); each
    retires an arm for the current round (issue #166)."""

    patience: int = 5  # consecutive evaluations without a new best
    sol_stop: float | None = SOL_STOP_PCT / 100  # fraction of its recipe's roofline
    target_hours: float | None = 2.0  # time in one arm's slices
    speedup_goal: float | None = 2.0  # module speedup of a kernel target
    decay: float = 0.7  # per consecutive slice without a new best
    explore: float = 1.0  # UCB exploration weight
    estimate: float = 2.0  # module speedup assumed reachable without a SOL estimate
    systems_estimate: float = 1.25  # end-to-end speedup assumed reachable by transforms
    systems: bool = True  # schedule systems-agent slices
    native: bool = False  # schedule the native arm (--native on, or the plan asks for it)
    native_estimate: float = 1.3  # speedup over the module-level bar assumed reachable
    native_patience: int = 8  # its native runs in a row without a new best
    native_hours: float | None = 6.0  # time in its slices
    # a retired arm comes back in a new round when its expected gain in the round's profile
    # is at least this share of the run (issue #166)
    revive_share: float = 0.02


@dataclass(frozen=True)
class Ceiling:
    """A kernel arm's place in the newest ceilings table (issue #122): the time of the rows
    that hold its instances and their floor at its precision, ms per profiled window
    (``per``: per batched run for ``throughput``); ``factor`` converts them to the metric's
    ms (the round's baseline ÷ the window)."""

    now: float
    floor: float
    precision: str  # the floor's label: W8A8, FP8 w, ...
    factor: float
    window_ms: float
    per: str
    rows: str  # the rows that hold its instances
    upper: bool = False  # all of a parent's rows (no table split them): an upper bound
    kernel_ms: float | None = None  # its best kernel per run, in the metric's ms

    @property
    def remaining_ms(self) -> float:
        """Its modules now in the metric's ms, or its best kernel when that is faster."""
        now = self.now * self.factor
        return min(now, self.kernel_ms) if self.kernel_ms is not None else now

    @property
    def headroom(self) -> float:
        left = self.remaining_ms
        return min(max(1.0 - self.floor * self.factor / left, 0.0), 1.0) if left > 0 else 0.0

    def describe(self) -> str:
        """``share 66.8% of 7.73 ms: now 793 ms per batched run (…), W8A8 floor 258 ms``."""
        run = self.window_ms * self.factor
        text = (
            f"share {self.now / self.window_ms:.1%} of {run:.3g} ms: now {self.now:,.4g} ms "
            f"{self.per} ({self.rows}{', an upper bound' if self.upper else ''})"
        )
        if self.kernel_ms is not None and self.kernel_ms < self.now * self.factor:
            text += f", its best kernel {self.kernel_ms / self.factor:,.4g} ms"
        return text + f", {self.precision} floor {self.floor:,.4g} ms"


@dataclass(frozen=True)
class FusionGain:
    """A region arm's candidate in the newest fusion table (issue #231): its predicted
    saving in the metric's ms (the table's ms per profiled window × the round's baseline ÷
    that window) and how it was matched."""

    id: str
    saving_ms: float
    how: str
    region: str = ""

    def describe(self) -> str:
        """``fusion f1a2b3c (…): predicts 1.2 ms``."""
        what = f": {self.region[:120]}" if self.region else ""
        return f"{self.how}{what}, predicts {self.saving_ms:,.3g} ms"


@dataclass
class Arm:
    id: str
    kind: str  # KERNEL | SYSTEMS
    # what the arm costs per run at 1.0× (kernels: its ceiling's now × the speedup of its
    # kernel applied in that profile, else its class's share × the baseline)
    ref_ms: float
    module_class: str | None = None
    best: float = 1.0  # best kept speedup (module for kernels, end to end for systems)
    best_snapshot: str | None = None
    sol: float | None = None  # fraction of the speed of light of the best result
    estimate: float = 2.0
    evals: int = 0
    gain_ms: float = 0.0  # ms per run saved by the arm's kept results
    streak: int = 0  # evaluations since the last new best (or the last research plan)
    fails: int = 0  # failed evaluations in a row (since the last research plan)
    stale: int = 0  # slices since the last slice that found a new best
    idle: int = 0  # finished slices in a row without a single evaluation
    short: bool = False  # its last slice made no evaluation in a session the budget cut short
    hours: float = 0.0  # time spent in this arm's improve slices
    expected_ms: float = 0.0
    index: float = 0.0  # UCB index
    score: float = 0.0
    stop: str | None = None
    rows: list[dict[str, Any]] = field(default_factory=list, repr=False)
    ceiling: Ceiling | None = None  # kernels: from the newest ceilings table (issue #122)
    basis: str = ""  # systems: where its estimate comes from
    damp: float = 1.0  # decay ** stale (:func:`rank`)
    # kernels: why its precision is not one the run allows (precisions.py); it never runs
    refused: str | None = None
    base: float = 1.0  # kernels: its best at the start of the round (the goal counts from it)
    fresh: bool = True  # kernels: its best was found in this round (the SOL rule applies)
    # kernels: its best is the library scout's (libscout/, #227): a floor for the engineers,
    # never a stop (the SOL rule and the speedup goal wait for an agent's best)
    scouted: bool = False
    note: str | None = None  # native: its staged plan is done, and what it works on now
    max_sessions: int = 1  # sessions it may run at once (--agents N): its live islands
    # native, gate relaxed (build_arms(relax_native=True)): why it would wait; it takes a
    # slot only when no other arm can use it
    held: str | None = None
    fusion: FusionGain | None = None  # region arms: their fusion candidate (issue #231)

    @property
    def agent(self) -> str:
        return self.kind if self.kind in (SYSTEMS, NATIVE) else f"kernel-{self.id}"

    @property
    def remaining_ms(self) -> float:
        if self.ceiling is not None:
            return self.ceiling.remaining_ms
        return self.ref_ms / max(self.best, 1e-9)

    @property
    def headroom(self) -> float:
        if self.ceiling is not None:
            return self.ceiling.headroom
        if self.fusion is not None and self.remaining_ms > 0:  # what it still predicts
            saving = self.fusion.saving_ms
            left = max(saving - max(self.ref_ms - self.remaining_ms, 0.0), 0.0)
            # never nothing: as MIN_FURTHER, a share of the prediction is always left
            return min(max(left, (1.0 - 1.0 / MIN_FURTHER) * saving) / self.remaining_ms, 1.0)
        if self.sol is not None:
            return min(max(1.0 - self.sol, 0.0), 1.0)
        further = max(self.estimate / max(self.best, 1e-9), MIN_FURTHER)
        return 1.0 - 1.0 / further

    def why(self) -> str:
        """How its score came about: the expected gain's components, the decay and the
        UCB index (the slice log, ``improve.json``)."""
        gain = self.remaining_ms * self.headroom
        if self.ceiling is not None:
            text = self.ceiling.describe()
        elif self.fusion is not None:
            text = (
                f"{self.fusion.describe()}, {max(self.ref_ms - self.remaining_ms, 0.0):.3g} "
                f"of it saved: {self.ref_ms:.3g} ms at 1.0x ÷ its best {self.best:.2f}x = "
                f"{self.remaining_ms:.3g} ms, headroom {self.headroom:.0%}"
            )
        else:
            if self.kind == SYSTEMS:
                text = f"end to end {self.remaining_ms:.3g} ms at its best {self.best:.2f}x"
            elif self.kind == NATIVE:
                text = (
                    f"native: end to end {self.remaining_ms:.3g} ms at {self.best:.2f}x over "
                    "the best module-level result"
                )
            else:
                text = (
                    f"Amdahl: {self.ref_ms:.3g} ms at 1.0x ÷ its best {self.best:.2f}x = "
                    f"{self.remaining_ms:.3g} ms"
                )
            text += f", headroom {self.headroom:.0%} " + (
                f"(at {self.sol:.0%} of the speed of light)"
                if self.sol is not None
                else f"(estimate {self.estimate:.3g}x{self.basis})"
            )
        stale = f" × {self.damp:.2g} ({self.stale} stale)" if self.damp < 1.0 else ""
        return f"{text} → {gain:.3g} ms{stale} × index {self.index:.2f}"

    @property
    def rate(self) -> float:
        """Observed gain per evaluation (ms per run)."""
        return self.gain_ms / self.evals if self.evals else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "arm": self.id,
            "best": round(self.best, 4),
            "remaining_ms": round(self.remaining_ms, 3),
            "headroom": round(self.headroom, 3),
            "sol": None if self.sol is None else round(self.sol, 3),
            "stale": self.stale,
            "streak": self.streak,
            "evals": self.evals,
            "rate_ms": round(self.rate, 3),
            "expected_ms": round(self.expected_ms, 3),
            "index": round(self.index, 3),
            "score": round(self.score, 3),
            "stop": self.stop,
            "why": self.why(),
            **({"note": self.note} if self.note else {}),
        }


# ------------------------------------------------------------------ inputs


def class_shares(profile: dict[str, Any]) -> dict[str, tuple[float, int]]:
    """Module class → (share of the profiled time, instances), as ``charts.target_shares``."""
    classes = profile.get("classes") or []
    roots: dict[str, float] = {}
    for c in classes:
        root = c.get("root", "")
        roots[root] = max(roots.get(root, 0.0), float(c.get("inclusive_ms") or 0.0))
    total = sum(roots.values())
    out: dict[str, tuple[float, int]] = {}
    if total <= 0:
        return out
    for c in classes:
        ms, n = float(c.get("inclusive_ms") or 0.0), int(c.get("instances") or 1)
        share, count = out.get(c["cls"], (0.0, 0))
        out[c["cls"]] = (share + ms / total, count + n)
    return out


def sol_fraction(record: dict[str, Any] | None) -> float | None:
    """Trustworthy ``pct_of_sol`` of a correct result (``roofline.sol_signal``) as a fraction."""
    pct = sol_signal(record) if record else None
    return None if pct is None else pct / 100.0


def _profiles(run: RunDir, rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Profiles newest first: ``{shares, baseline_ms, applied, profile, ceilings, fusions}``
    per round (round 1 = the run's; ``ceilings`` / ``fusions``: the tables next to the
    profile, None without one)."""
    out = []
    for rnd in reversed(rounds):
        path = run.root / str(rnd.get("profile") or "")
        profile = read_json(path, None) if rnd.get("profile") else None
        if profile and rnd.get("baseline_ms"):
            out.append(
                {
                    "shares": class_shares(profile),
                    "baseline_ms": float(rnd["baseline_ms"]),
                    "applied": rnd.get("applied") or {},
                    "profile": profile,
                    "ceilings": read_json(path.parent / "ceilings.json", None),
                    "fusions": read_json(path.parent / "fusions.json", None),
                }
            )
    base = read_json(run.baseline_json, {}) or {}
    profile = read_json(run.profile_dir / "profile.json", {}) or {}
    if base.get("median_ms"):
        out.append(
            {
                "shares": class_shares(profile),
                "baseline_ms": float(base["median_ms"]),
                "applied": {},
                "profile": profile,
                "ceilings": read_json(run.profile_dir / "ceilings.json", None),
                "fusions": read_json(run.profile_dir / "fusions.json", None),
            }
        )
    return out


def _ref_ms(target_id: str, spec: dict[str, Any], profiles: list[dict[str, Any]]) -> float:
    """Time per run of the target's class at 1.0×, from the newest profile that has the class.

    A re-profile (improve rounds) measured the model with the integrated kernels,
    so a class whose kernel was applied there ran ``applied`` × faster (the kernel of
    the target or of another precision of it, ``pivot.py``: one of them is applied).
    """
    cls = spec.get("module_class")
    for prof in profiles:
        if cls in prof["shares"]:
            share, instances = prof["shares"][cls]
            if spec.get("qualname") and instances > 1:
                share /= instances  # restricted to one instance of the class
            return share * prof["baseline_ms"] * _applied(target_id, prof)
    return 0.0


def _applied(target_id: str, prof: dict[str, Any]) -> float:
    """The speedup of the target's kernel (or of another precision of it, ``pivot.py``: one
    of them is applied) in a re-profiled model (1.0: none)."""
    from kernel_agent.pivot import family

    mine = [v for t, v in prof["applied"].items() if family(t) == family(target_id)]
    return float(mine[0] or 1.0) if mine else 1.0


def _per_run(
    run: RunDir, target_id: str, spec: dict[str, Any], record: dict[str, Any] | None, key: str
) -> float | None:
    """Σ ``key`` (``ref_ms`` / ``new_ms``) of an evaluation's timed cases × the calls each
    stands for (as the evaluator, ``kernels/weights.py``): ms per run of the workload,
    converted to the metric's ms (``projection.Units``: per second of audio for
    ``metric=throughput``). None: no timed case, or no value in the metric."""
    users = (spec.get("capture") or {}).get("method_instances") or {}
    timed = [c for c in (record or {}).get("cases") or [] if c.get(key) is not None]
    if not timed:
        return None
    per_run = sum(float(c[key]) * weights.case_weight(c, users) for c in timed)
    return projection.units_of(run, [target_id])(target_id, per_run)


def _region_ref_ms(
    run: RunDir, target_id: str, spec: dict[str, Any], profiles: list[dict[str, Any]]
) -> float:
    """A region target's ``Region_<id>`` class is in no profile (``kernel_agent/region.py``):
    its reference time from the timed cases of its newest timed evaluation (:func:`_per_run`),
    else the time of its parent class (an upper bound) until it has one or when it has no
    value in the metric."""
    if spec.get("kind") != "region":
        return 0.0
    for rec in reversed(read_jsonl(run.results_file(target_id))):
        if any(c.get("ref_ms") is not None for c in rec.get("cases") or []):
            if (ms := _per_run(run, target_id, spec, rec, "ref_ms")) is not None:
                return ms
            break
    return _ref_ms(target_id, {**spec, "module_class": spec.get("parent_class")}, profiles)


def arm_ceiling(
    spec: dict[str, Any], groups: list[projection.Group], profiles: list[dict[str, Any]]
) -> Ceiling | None:
    """A kernel arm's :class:`Ceiling` in the newest profile's ceilings table (issue #122),
    or None (Amdahl, :func:`_ref_ms`): no table there, a region target, no row that holds
    its instance groups, or a floor that is unknown (unknown work, a peak not measured).

    ``groups``: its instance groups (``projection.build``: folded qualnames, its ``qualname``
    / ``qualname_regex`` / ``phase``); a group whose calls no case stands for (the
    ``instance_groups`` of a capture since #119) is left out, a target scoped to one of a
    group's instances (``inside``) takes that share of it. The rows of a group:
    :func:`ceilings.holding`. When the optimised model hides a group inside a parent, the
    group takes the part of the parent's time it took in the oldest table that has both
    (:func:`_split`: the analyze profile), with its own floor there (the work is the math,
    the same in every round); without one, the parent's rows in full (``upper``)."""
    newest = profiles[0] if profiles else {}
    table = newest.get("ceilings") or {}
    window = float(table.get("baseline_ms") or 0.0)
    if spec.get("kind") == "region" or not table.get("rows") or window <= 0:
        return None
    precision = ceilings.target_precision(spec.get("precision"), table.get("peaks"))
    cls, phase = spec.get("module_class"), spec.get("phase") or None
    parts: dict[Any, tuple[float, float, str]] = {}
    upper = False
    for group in groups:
        if not group.pattern or group.share <= 0:
            continue
        rows, inside = ceilings.holding(table, group.pattern, cls, phase)
        if not rows:
            continue
        now = sum(float(r.get("now_ms") or 0.0) for r in rows)
        floor = ceilings.floor_ms(rows, precision, table.get("peaks"))
        key: Any = tuple(sorted((str(r.get("target")), str(r.get("phase"))) for r in rows))
        name = _row_names(rows)
        if inside:
            split = _split(group.pattern, cls, phase, rows, precision, profiles[1:])
            if split is None:
                upper = True
            else:
                part, floor = split
                now, key, name = now * part, group.pattern, f"{part:.0%} of {name}"
            name = f"inside {name}"
        if floor is None:
            return None
        if group.inside < 1.0:
            now, floor, name = now * group.inside, floor * group.inside, f"1 instance: {name}"
        parts[key] = (now, floor, name)
    if not parts:
        return None
    held = sorted(parts.values(), key=lambda p: -p[0])
    return Ceiling(
        now=sum(p[0] for p in held),
        floor=sum(p[1] for p in held),
        precision=precision.label,
        factor=float(newest["baseline_ms"]) / window,
        window_ms=window,
        per=str(table.get("per") or "per run"),
        rows="; ".join(p[2] for p in held),
        upper=upper,
    )


def fusion_gain(spec: dict[str, Any], profiles: list[dict[str, Any]]) -> FusionGain | None:
    """A region target's candidate (or a target's named ``fusion``) in the newest fusion table
    (``fusion.match``), its saving in the metric's ms; None: no table has one."""
    from kernel_agent.profiling import fusion

    if spec.get("kind") != "region" and not spec.get("fusion"):
        return None
    for prof in profiles:  # newest first: each re-profile mines what is left (#231)
        table = prof.get("fusions") or {}
        window = float(table.get("window_ms") or 0.0)
        found = fusion.match(table, spec) if window > 0 else None
        if found is not None:
            hit, how = found
            factor = float(prof["baseline_ms"]) / window
            saving = float(hit.get("saving_ms") or 0.0) * factor
            return FusionGain(str(hit["id"]), saving, how, str(hit.get("region") or ""))
    return None


def _split(
    pattern: str,
    cls: str | None,
    phase: str | None,
    parent: list[dict[str, Any]],
    precision: ceilings.Precision,
    older: list[dict[str, Any]],
) -> tuple[float, float] | None:
    """(the part of the ``parent`` rows' time that the hidden instance group ``pattern``
    took, its own floor at ``precision``) from the oldest of the ``older`` profiles' ceilings
    tables that has its own rows and those of the parent's groups; None: none has."""
    outer = {r.get("group") for r in parent}
    for prof in reversed(older):  # the oldest first: the split of the unmodified model
        table = prof.get("ceilings") or {}
        own, inside = ceilings.holding(table, pattern, cls, phase)
        whole = sum(
            float(r.get("now_ms") or 0.0)
            for r in table.get("rows") or []
            if r.get("group") in outer
        )
        floor = ceilings.floor_ms(own, precision, table.get("peaks"))
        if own and not inside and whole > 0 and floor is not None:
            return min(sum(float(r.get("now_ms") or 0.0) for r in own) / whole, 1.0), floor
    return None


def _row_names(rows: list[dict[str, Any]]) -> str:
    """``UnifiedCFM model.feat_decoder``; the phases of a group with rows in several."""
    phases: dict[str, list[str]] = {}
    for r in rows:
        phases.setdefault(f"{r.get('cls')} {r.get('group')}", []).append(str(r.get("phase")))
    return ", ".join(
        name + (f" ({'+'.join(ps)})" if len(ps) > 1 else "") for name, ps in phases.items()
    )


def _e2e_estimate(
    run: RunDir, profiles: list[dict[str, Any]], base_ms: float
) -> tuple[float, str] | None:
    """Systems: the end-to-end speedup over the baseline (``base_ms``, the metric's ms) the
    newest ceilings table allows (its end-to-end line: every row at its floor, nested rows
    counted once, the time outside them unchanged), at the lowest floor of the precisions
    the run allows (exact; ``near-lossless``: FP8 weights / W8A8 too, FP4 weights only
    with ``--precisions`` naming them: ``precisions.py``), and a note on it."""
    newest = profiles[0] if profiles else {}
    table = newest.get("ceilings") or {}
    e2e, window = table.get("e2e") or {}, float(table.get("baseline_ms") or 0.0)
    names = ceilings.target_columns(precisions.of_run(run))
    floors = {n: float(e2e[n]["floor_ms"]) for n in names if (e2e.get(n) or {}).get("floor_ms")}
    if not floors or window <= 0 or base_ms <= 0:
        return None
    name = min(floors, key=lambda n: floors[n])
    label = ((table.get("precisions") or {}).get(name) or {}).get("label", name)
    speedup = base_ms / (floors[name] * float(newest["baseline_ms"]) / window)
    per = table.get("per") or "per run"
    return speedup, f": {label} floors ≥ {floors[name]:,.4g} of {window:,.4g} ms {per}"


def gpu_busy(run: RunDir) -> float | None:
    """The profile's GPU busy share: kernel time per run without the profiler (``busy_share``;
    a profile from before it: of the profiled run, ``gpu_busy_fraction``). None: none, or a
    kernel view marked unreliable."""
    profile = read_json(run.profile_dir / "profile.json", {}) or {}
    view = profile.get("kernel_view") or {}
    if view.get("reliable") is False:
        return None
    share = ledger._num(view.get("busy_share"))
    busy = min(share, 1.0) if share else ledger._num(view.get("gpu_busy_fraction"))
    return busy if busy and 0.0 < busy <= 1.0 else None


def made_by(row: Mapping[str, Any]) -> str:
    """The agent whose session made a ledger row: its ``session`` label is
    ``<agent>#<slice>`` (``improve.Improver._open_slice``); "" for a row without one."""
    return str(row.get("session") or "").partition("#")[0]


def native_run(row: Mapping[str, Any]) -> bool:
    """An end-to-end evaluation of the native arm: one a native session made, or, for a row
    without a session label, one that ran a native project (``native_engine.is_native``).
    A systems session's stack that runs native projects is the systems arm's evaluation:
    grouping rows by their backend alone credited it to the native arm, so the systems arm's
    slices counted none of their rows and its best and headroom went stale."""
    if row.get("target") != ledger.E2E or row.get("backend") == "integrate":
        return False
    maker = made_by(row)
    if maker in (SYSTEMS, NATIVE):
        return maker == NATIVE
    return native_engine.is_native(row)


def systems_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """End-to-end rows of the systems agent (integration steps and the native arm's runs are
    not its evaluations)."""
    return [
        r
        for r in rows
        if r["target"] == ledger.E2E and r["backend"] != "integrate" and not native_run(r)
    ]


def _improves(row: dict[str, Any], best: float) -> bool:
    rec = {"passed": row["correct"], "speedup": row["speedup"], "timing_spread": row["spread"]}
    return improves(rec, best, ok_key="passed")


def _prior(row: dict[str, Any]) -> bool:
    """A library prior's or the library scout's row: not an agent's evaluation."""
    return not_agents(row["hypothesis"])


def _kernel_history(
    arm: Arm, rows: list[dict[str, Any]], planned: int | None = None, since: int | None = None
) -> None:
    """Best, gain and streak of a kernel target from its ledger rows (``keep`` = new best;
    the library's prior winners do not extend the streak). A ``re-evaluated`` row is not
    an evaluation, but replaces its snapshot's earlier result in the best (and the gain)
    so far (:class:`~kernel_agent.budget.Standing`, the ledger's keep bar).

    ``planned``: the ledger size when the last research plan of the target was
    written; ``since``: when the current round started (issue #166, None: round 1); the
    streak and the failures count only the evaluations after both. ``base`` is the best
    when the round started, ``fresh`` whether the round found a better one.
    """
    stand = Standing()
    crossed = since is None
    for row in rows:
        if since is not None and not crossed and (row["exp"] or 0) > since:
            arm.base, crossed = arm.best, True
        if row["status"] == ledger.REEVALUATED:
            stand.replace(row)
        elif row["status"] in ledger.UNMEASURED:
            continue
        elif row["status"] == ledger.KEEP and row["speedup"]:
            stand.keep(row)
            arm.streak = 0
        elif not _prior(row):
            arm.streak += 1
        if (new := stand.best) != arm.best:
            arm.gain_ms += arm.ref_ms * (1.0 / arm.best - 1.0 / new)
            arm.best = new
    arm.best_snapshot = top["snapshot"] if (top := stand.top) else None
    arm.scouted = top is not None and str(top.get("hypothesis") or "").startswith(
        LIBRARY_HYPOTHESIS
    )
    if not crossed:  # nothing in this round yet
        arm.base = arm.best
    arm.fresh = since is None or arm.best > arm.base
    restart = max((x for x in (planned, since) if x is not None), default=None)
    recent = [
        r
        for r in ledger.measured(rows)
        if not _prior(r) and (restart is None or (r["exp"] or 0) > restart)
    ]
    arm.streak = min(arm.streak, len(recent))
    for row in reversed(recent):
        if row["status"] not in ledger.FAILURES:
            break
        arm.fails += 1


def e2e_kernels(
    rows: list[dict[str, Any]], records: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], frozenset[str]]]:
    """Every ``e2e`` row with the kernels it measured (``target=snapshot name``).

    A systems row takes them from its transforms ``results.jsonl`` record
    (``kernels``: ``TARGET=path`` entries). An integration row only names its
    items: its kernels are the targets it names, each with the kernel an
    integration takes, the fastest correct one so far (``tools.best_for_target``).
    A kernel file that is not a snapshot of its target matches no other row.
    """

    def key(rec: dict[str, Any]) -> str:
        return ledger.e2e_snapshot(rec.get("transforms") or [], rec.get("kernels") or [])

    by_exp = {rec.get("exp"): rec for rec in records}
    by_snapshot = {key(rec): rec for rec in records}
    snapshots: dict[str, set[str]] = {}  # target → its evaluated snapshots so far
    fastest: dict[str, tuple[float, str]] = {}  # target → (speedup, snapshot) of its best
    out: list[tuple[dict[str, Any], frozenset[str]]] = []
    for row in rows:
        target, exp = row["target"], row["exp"]
        if target != ledger.E2E:
            snapshots.setdefault(target, set()).add(row["snapshot"])
            if row["correct"] and (row["speedup"] or 0.0) > fastest.get(target, (0.0, ""))[0]:
                fastest[target] = (float(row["speedup"]), row["snapshot"])
            continue
        rec = None
        if row["backend"] != "integrate":  # integration steps have no transforms record
            rec = by_exp.get(exp)
            if rec is None or key(rec) != row["snapshot"]:
                rec = by_snapshot.get(row["snapshot"])
        if rec is not None:
            pairs = [str(k).partition("=")[::2] for k in rec.get("kernels") or []]
            named = [(t, Path(path).name) for t, path in pairs]
        else:  # the targets named in the snapshot, with the kernel an integration takes
            parts = [t for t in row["snapshot"].split("+") if t in snapshots]
            named = [(t, fastest[t][1] if t in fastest else "") for t in parts]
        kernels = {f"{t}={n}" if n in snapshots.get(t, ()) else f"{t}=?{exp}" for t, n in named}
        out.append((row, frozenset(kernels)))
    return out


def _systems_history(arm: Arm, rows: list[tuple[dict[str, Any], frozenset[str]]]) -> None:
    """Best, gain and streak of the systems agent from the ``e2e`` rows (:func:`e2e_kernels`).

    The gain of the kernels in a run is not the systems agent's. A transform-only
    run is a new best when it beats the best transform-only run so far (1.0: the
    baseline); transforms on top of kernels when they beat the best run measured
    with the same kernels: an integration or a kernels-only run, or the agent's
    previous new best with them (its first run with them sets the bar). Its run on
    top of native projects (:func:`native_run`) must also beat the native arm's fastest
    run so far: their gain is not the systems agent's either. ``gain_ms`` adds the ms
    per run saved over that reference, ``best`` is the fastest new best end to end
    (its kernels included).
    """
    refs: dict[frozenset[str], float] = {frozenset(): 1.0}  # kernels → speedup to beat
    native_best = 0.0  # the native arm's fastest run so far
    for row, kernels in rows:
        measured = float(row["speedup"]) if row["correct"] and row["speedup"] else None
        if native_run(row):  # the native arm's (native_arm)
            native_best = max(native_best, measured or 0.0)
            continue
        ref = refs.get(kernels)
        if native_engine.is_native(row) and native_best:
            ref = max(ref or 0.0, native_best)
        if row["backend"] == "integrate":  # not an evaluation of the systems agent
            if kernels and measured:
                refs[kernels] = max(ref or 0.0, measured)
            continue
        # a native project in a systems stack is one of its transforms (native_run)
        transforms = "transform" in row["backend"] or native_engine.is_native(row)
        if transforms and ref is not None and measured and _improves(row, ref):
            arm.gain_ms += arm.ref_ms * (1.0 / ref - 1.0 / measured)
            refs[kernels], arm.streak = measured, 0
            if measured > arm.best:
                arm.best, arm.best_snapshot = measured, row["snapshot"]
            continue
        arm.streak += 1
        if kernels and measured and (ref is None or not transforms):
            refs[kernels] = max(ref or 0.0, measured)  # the first run with them, or them alone


def snapshot_record(run: RunDir, target_id: str, snapshot: str | None) -> dict[str, Any] | None:
    """The ``results.jsonl`` record of a target's snapshot (file name or path)."""
    if not snapshot:
        return None
    name = Path(snapshot).name
    for rec in reversed(read_jsonl(run.results_file(target_id))):
        if Path(str(rec.get("snapshot", ""))).name == name:
            return rec
    return None


def build_arms(
    run: RunDir,
    policy: Policy,
    slices: list[dict[str, Any]],
    *,
    targets: list[str] | None = None,
    rounds: list[dict[str, Any]] | None = None,
    rows: list[dict[str, Any]] | None = None,
    research: list[dict[str, Any]] | None = None,
    relax_native: bool = False,
    islands: Mapping[str, int] | None = None,
) -> list[Arm]:
    """Every arm with its history, stop reason and score (live arms first, best first).

    ``research``: the research sessions (``improve.json``); one that wrote a plan
    (``plan``) restarts its arm's streak at its ``exp``. ``rounds``: the round records; the
    move-on rules count from the current round's start (its ``exp``, issue #166).
    ``relax_native``: concurrent sessions; the native arm is not stopped while kernel arms
    improve, only marked :attr:`Arm.held` (:func:`assign`). ``islands``: the live islands of
    each kernel target with several (``workers.py``, issue #189), its ``max_sessions``."""
    rows = ledger.rows(run) if rows is None else rows
    profiles = _profiles(run, rounds or [])
    current = (rounds or [{}])[-1]
    since = int(current["exp"]) if current.get("exp") is not None else None
    round_n = int(current.get("n") or 1)
    base_ms = profiles[-1]["baseline_ms"] if profiles else 0.0
    allowed = precisions.of_run(run)  # an arm at another precision is stopped (#131)
    ids = run.target_ids() if targets is None else targets
    specs = {t: read_json(run.target(t) / "spec.json", {}) or {} for t in ids}
    groups: dict[str, list[projection.Group]] = {}  # the instance groups of each target
    for group in projection.build(specs, [p["profile"] for p in reversed(profiles)]).groups:
        groups.setdefault(group.target, []).append(group)
    arms: list[Arm] = []
    for target_id in ids:
        spec = specs[target_id]
        if native_engine.is_stage_target(spec):  # the native arm's own (native_arm)
            continue
        estimate = ledger._num(spec.get("expected_speedup")) or policy.estimate
        target_rows = [r for r in rows if r["target"] == target_id]
        refused = precisions.refusal(precisions.of_spec(spec), allowed)
        ceiling = None if refused else arm_ceiling(spec, groups.get(target_id, []), profiles)
        if ceiling is not None:  # its modules in the re-profiled run, at 1.0x of its kernel
            ref_ms = ceiling.now * ceiling.factor * _applied(target_id, profiles[0])
        else:
            ref_ms = _ref_ms(target_id, spec, profiles)
            ref_ms = ref_ms or _region_ref_ms(run, target_id, spec, profiles)
        arm = Arm(
            target_id,
            KERNEL,
            ref_ms,
            module_class=spec.get("module_class"),
            estimate=estimate,
            rows=ledger.measured(target_rows),
            refused=refused,
            max_sessions=max(int((islands or {}).get(target_id, 1)), 1),
        )
        plans = [int(r["exp"]) for r in research or [] if r["arm"] == target_id and r.get("plan")]
        key = timing_context.current(run, target_id).key  # its bests compare in it (#226)
        compared = ledger.in_context(run, target_id, target_rows, key)
        _kernel_history(arm, compared, max(plans, default=None), since)
        record = snapshot_record(run, target_id, arm.best_snapshot)
        record = timing_context.in_context(record, key) if record is not None else None
        arm.sol = sol_fraction(record)
        if ceiling is not None and record is not None:  # its best kernel, if faster than now
            ceiling = dataclasses.replace(
                ceiling, kernel_ms=_per_run(run, target_id, spec, record, "new_ms")
            )
        arm.ceiling = ceiling
        if ceiling is None and not refused:  # a region's expected gain: its fusion (#231)
            arm.fusion = fusion_gain(spec, profiles)
        arms.append(arm)
    if policy.systems:
        busy = gpu_busy(run)
        estimate = max(policy.systems_estimate, 1.0 / busy if busy else 0.0)
        basis = f": 1 ÷ the GPU busy share {busy:.0%}" if busy and 1.0 / busy == estimate else ""
        if (e2e := _e2e_estimate(run, profiles, base_ms)) is not None and e2e[0] > estimate:
            estimate, basis = e2e
        arm = Arm(
            SYSTEMS, SYSTEMS, base_ms, estimate=estimate, rows=systems_rows(rows), basis=basis
        )
        _systems_history(arm, e2e_kernels(rows, read_jsonl(run.results_file())))
        arm.streak = _in_round(arm.streak, arm.rows, since)
        arms.append(arm)
    if policy.native:
        stage_targets = {t for t in ids if native_engine.is_stage_target(specs[t])}
        arms.append(native_arm(run, rows, policy, stage_targets, since))
    for arm in arms:
        # a slice still running says nothing yet (concurrent sessions: its virtual pulls do)
        mine = [s for s in slices if s.get("arm") == arm.id and s.get("status") != "running"]
        earlier = since is not None and any(int(s.get("round") or 1) < round_n for s in mine)
        if since is not None:  # retired for a round only: its slices in this round count
            mine = [s for s in mine if int(s.get("round") or 1) == round_n]
        arm.hours = sum(float(s.get("seconds") or 0.0) for s in mine) / 3600
        arm.short = bool(mine and mine[-1].get("budget_short"))
        tried = [s for s in mine if not s.get("budget_short")]  # out of time: says nothing
        for s in reversed(tried):
            if s.get("improved"):
                break
            arm.stale += 1
        for s in reversed(tried):
            if s.get("evals") or s.get("status") == "interrupted":
                break
            arm.idle += 1
        arm.evals = len(arm.rows)
        arm.stop = stop_reason(arm, policy)
        if arm.stop is None and earlier and not mine:  # back in a new round only if it matters
            arm.stop = _revival(arm, policy, profiles, round_n)
    _native_gate(run, arms, policy, rows, allowed, relax=relax_native)
    return rank(arms, policy)


def _in_round(streak: int, rows: list[dict[str, Any]], since: int | None) -> int:
    """A streak counted only over the ``rows`` of the current round (issue #166)."""
    if since is None:
        return streak
    return min(streak, sum((r["exp"] or 0) > since for r in rows))


def _revival(arm: Arm, policy: Policy, profiles: list[dict[str, Any]], round_n: int) -> str | None:
    """Why an arm that worked in an earlier round stays retired in round ``round_n``
    (issue #166): its expected gain in the round's profile (``remaining_ms × headroom``) is
    below ``revive_share`` of the run there; None: it comes back."""
    run_ms = float(profiles[0]["baseline_ms"]) if profiles else 0.0
    expected = arm.remaining_ms * arm.headroom
    if not policy.revive_share or run_ms <= 0 or expected >= policy.revive_share * run_ms:
        return None
    return (
        f"retired: expected {expected:.3g} ms in the round-{round_n} profile, below "
        f"{policy.revive_share:.0%} of its {run_ms:.4g} ms"
    )


def native_arm(
    run: RunDir,
    rows: list[dict[str, Any]],
    policy: Policy,
    stage_targets: set[str],
    since: int | None = None,
) -> Arm:
    """The native arm: its evaluations are its end-to-end runs (:func:`native_run`) and
    those of its stage targets; ``best`` is its fastest run over the module-level bar
    (``native.engine.bar``, 1.0 = at the bar), a new best when it beats the previous one
    beyond the noise; ``ref_ms`` is the run at the bar. Its streak counts the runs of the
    current round (``since``: its start, issue #166)."""
    level = native_engine.bar(rows)
    base = float((read_json(run.baseline_json, {}) or {}).get("median_ms") or 0.0)
    arm = Arm(NATIVE, NATIVE, base / level, estimate=policy.native_estimate)
    arm.basis = ": the native estimate (--native)"
    runs = [r for r in rows if native_run(r)]
    stage_rows = ledger.measured(r for r in rows if r["target"] in stage_targets)
    arm.rows = sorted([*runs, *stage_rows], key=lambda r: r["exp"] or 0)
    best = level
    for row in runs:
        measured = float(row["speedup"]) if row["correct"] and row["speedup"] else None
        if measured and _improves(row, best):
            arm.gain_ms += base * (1.0 / best - 1.0 / measured)
            best, arm.streak, arm.best_snapshot = measured, 0, row["snapshot"]
        else:
            arm.streak += 1
        arm.fails = arm.fails + 1 if row["status"] in ledger.FAILURES else 0
    arm.best = best / level
    arm.streak = _in_round(arm.streak, runs, since)
    return arm


def _native_gate(
    run: RunDir,
    arms: list[Arm],
    policy: Policy,
    rows: list[dict[str, Any]],
    allowed: Iterable[str] | None,
    *,
    relax: bool = False,
) -> None:
    """Hold the native arm while a kernel arm is live and has not plateaued (module kernels
    first; ``relax``: only mark it :attr:`Arm.held`, for a free slot no other arm can use).
    Once every stage of its staged plan beat the module-level bar, note it and keep the arm
    live (issue #164): its patience and time cap still apply, and it stops when every stage
    of the newest stage graph runs at ``sol_stop`` of its floor."""
    arm = next((a for a in arms if a.kind == NATIVE), None)
    if arm is None or arm.stop is not None:
        return
    live = [
        a.id for a in arms if a.kind == KERNEL and a.stop is None and plateau(a, policy) is None
    ]
    if live:
        why = f"waiting: module arms still improving ({', '.join(live[:4])})"
        if not relax:
            arm.stop = why
            return
        arm.held = why
    status = native_engine.status(run, rows, ceilings.columns(allowed))
    if status.complete:
        arm.note = status.note()
        arm.stop = native_engine.at_floor(status.graph, policy.sol_stop)


# ------------------------------------------------------------------ decisions


def stop_reason(arm: Arm, policy: Policy) -> str | None:
    """Why an arm gets no more slices in this round (None: it is live). Only ``refused`` is
    for good; the others retire it for the round (issue #166): the SOL rule applies to a best
    found in the round, the speedup goal counts from its best at the round's start."""
    if arm.refused:  # its precision is not allowed: no research session revives it
        return arm.refused
    patience = policy.native_patience if arm.kind == NATIVE else policy.patience
    if patience and arm.streak >= patience:
        return f"plateau: {arm.streak} evaluations in a row without a new best"
    if arm.idle >= IDLE_SLICES:
        return f"no evaluation in its last {arm.idle} slices"
    agents = not arm.scouted  # the library scout's best is a floor to beat, never a stop
    sol = arm.sol is not None and policy.sol_stop and arm.sol >= policy.sol_stop
    if sol and arm.fresh and agents:
        return f"at {arm.sol:.0%} of its recipe's roofline (move on at {policy.sol_stop:.0%})"
    cap = policy.native_hours if arm.kind == NATIVE else policy.target_hours
    if cap and arm.hours >= cap:
        return f"time cap: {arm.hours:.1f} h in its slices (cap {cap:g} h)"
    goal = policy.speedup_goal
    if arm.kind == KERNEL and goal and agents and arm.best >= goal * arm.base:
        since = f" ({arm.best / arm.base:.2f}x this round)" if arm.base > 1.0 else ""
        return f"speedup goal reached: {arm.best:.2f}x{since} (goal {goal:g}x)"
    return None


def plateau(arm: Arm, policy: Policy) -> str | None:
    """Why a kernel arm needs a research session, or None.

    It has plateaued: :data:`~kernel_agent.budget.PLATEAU` evaluations in a row
    without a new best (the evaluation advice is ``consider_stopping`` then; fewer
    with a smaller ``patience``), or :data:`FAIL_STREAK` failed ones. And nothing
    but the plateau stops it (the speed of light, the time cap or the goal do not).
    """
    if arm.kind != KERNEL or stop_reason(dataclasses.replace(arm, streak=0), policy):
        return None
    limit = min(PLATEAU, policy.patience) if policy.patience else PLATEAU
    if arm.fails >= FAIL_STREAK:
        return f"{arm.fails} failed evaluations in a row"
    if arm.streak >= limit:
        return f"{arm.streak} evaluations in a row without a new best"
    return None


def rank(arms: list[Arm], policy: Policy) -> list[Arm]:
    """Fill in expected gain, UCB index and score; live arms first, highest score first."""
    total = sum(a.evals for a in arms)
    top = max((a.rate for a in arms), default=0.0)
    for arm in arms:
        arm.damp = policy.decay**arm.stale
        arm.expected_ms = arm.remaining_ms * arm.headroom * arm.damp
        exploit = arm.rate / top if top > 0 else 0.0
        arm.index = exploit + policy.explore * math.sqrt(2 * math.log(total + 2) / (arm.evals + 1))
        arm.score = arm.expected_ms * arm.index
    return sorted(arms, key=lambda a: (a.stop is not None, -a.score, a.id))


def pick(arms: list[Arm]) -> Arm | None:
    """The live arm with the highest score (arms as returned by :func:`build_arms`)."""
    return next((a for a in arms if a.stop is None), None)


# ------------------------------------------------------------------ concurrent sessions


def role_caps(overrides: Mapping[str, int] | None = None) -> dict[str, int]:
    """Sessions of each role at once: the role registry's ``max_concurrent``
    (``roles.REGISTRY``: kernel 4, systems 1, native 1) with ``overrides`` (``--role-max``)."""
    caps = {r.name: r.max_concurrent for r in roles.REGISTRY.values() if r.max_concurrent}
    return {**caps, **(overrides or {})}


@dataclass(frozen=True)
class Running:
    """A running session as the scheduler sees it (:func:`assign`): its arm, its role
    (``kernel``, ``systems``, ``native``, ``research``, ``dossier``), its agent name and the
    evaluations it is expected to make (its virtual pull)."""

    arm: str
    role: str
    agent: str
    evaluations: int = 0


def virtual_pulls(arms: list[Arm], running: list[Running], policy: Policy) -> list[Arm]:
    """Copies of ``arms`` ranked as if every running session had made its expected
    evaluations without a new best: its arm's evaluations (the UCB ``n``, and ``N``) grow
    by them and it is one more slice stale (``decay``). Without this every free slot would
    go to the top arm, whose running session has no result in the ledger yet."""
    pulled = []
    for arm in arms:
        mine = [r for r in running if r.arm == arm.id and r.evaluations]
        evals = arm.evals + sum(r.evaluations for r in mine)
        pulled.append(dataclasses.replace(arm, evals=evals, stale=arm.stale + len(mine)))
    return rank(pulled, policy)


def _eligible(
    arm: Arm,
    running: list[Running],
    role_max: Mapping[str, int],
    time_left: float | None,
) -> bool:
    """Whether ``arm`` may take a slot now (:func:`assign`)."""
    mine = [r for r in running if r.arm == arm.id]
    if arm.stop is not None or any(not roles.get(r.role).needs_gpu for r in mine):
        return False  # stopped, or paused while its research (or dossier) session runs
    if len(mine) >= arm.max_sessions or any(r.agent == arm.agent for r in running):
        return False
    cap = role_max.get(arm.kind)
    if cap is not None and sum(r.role == arm.kind for r in running) >= cap:
        return False
    need = slice_seconds(arm) * (SHORT_SLICE if arm.short else 1.0)
    return time_left is None or need <= time_left


def assign(
    arms: list[Arm],
    slots: int,
    running: list[Running],
    policy: Policy,
    *,
    role_max: Mapping[str, int] | None = None,
    time_left: float | None = None,
    evaluations: Mapping[str, int] | None = None,
    ok: Callable[[Arm], bool] | None = None,
) -> list[Arm]:
    """Arms for up to ``slots`` free slots, one at a time and best first, each ranked by
    :func:`virtual_pulls` of the sessions running then (those chosen before it included).

    An arm is eligible when it is live, not paused (a GPU-free session of it runs: research,
    a dossier), below ``max_sessions`` and its role's cap (``role_max``, default
    :func:`role_caps`), its agent name not running, with
    ``time_left`` for a slice (:func:`slice_seconds`; None: no time limit) and ``ok`` (the
    coordinator's own conditions). A held native arm (:attr:`Arm.held`) takes a slot only
    when no other arm can. ``evaluations``: a session's expected evaluations by arm kind
    (default 4)."""
    caps = role_caps() if role_max is None else role_max
    per = evaluations or {}
    chosen: list[Arm] = []
    busy = list(running)
    for _ in range(slots):
        ranked = [
            a
            for a in virtual_pulls(arms, busy, policy)
            if _eligible(a, busy, caps, time_left) and (ok is None or ok(a))
        ]
        arm = next((a for a in ranked if a.held is None), ranked[0] if ranked else None)
        if arm is None:
            break
        chosen.append(arm)
        busy.append(Running(arm.id, arm.kind, arm.agent, per.get(arm.kind, 4)))
    return chosen


# ------------------------------------------------------------------ time


def median_eval_s(rows: Iterable[dict[str, Any]]) -> float | None:
    """Median ``eval_s`` of ledger rows (None: none was timed)."""
    times = [float(r["eval_s"]) for r in rows if r.get("eval_s")]
    return statistics.median(times) if times else None


def slice_seconds(arm: Arm) -> float:
    """Time one slice of ``arm`` needs: the agent's warm-up, one evaluation of the arm (the
    median ``eval_s`` of its evaluations, else :data:`EVAL_SECONDS`), the time it is
    expected to wait for the GPU behind the jobs there now (``gpuqueue.expected_wait``; 0
    when nothing runs) and the wrap-up after it (the evaluation advice says ``stop`` with
    less than ``WRAP_UP_SECONDS`` left)."""
    wait = gpuqueue.expected_wait(gpuqueue.EVAL if arm.kind == KERNEL else gpuqueue.E2E)
    each = median_eval_s(arm.rows) or EVAL_SECONDS[arm.kind]
    return WARMUP_SECONDS + each + wait + WRAP_UP_SECONDS


def integration_estimate(
    run: RunDir,
    rows: list[dict[str, Any]],
    kernels: dict[str, str],
    reusable: list[dict[str, Any]],
) -> tuple[int, float]:
    """(A/B measurements, seconds per measurement) of an integration of everything so far.

    Its items (as ``Orchestrator._integration_items`` takes them, without the digest
    checks of the transforms): the ``kernels`` (target → ``target=path`` of the kernel the
    integration takes, ``Orchestrator._kernel_best``) and the version of the fastest
    passing end-to-end run of every transform idea faster than the baseline. It measures
    each alone, then the systems agent's fastest combination of them (the seed; none: the
    fastest item alone), then adds each item outside the seed in turn. A re-integration
    reuses what the last one measured with the same content
    (:mod:`kernel_agent.integrate.reuse`; here: the sha256 of each item's file, against
    the ``reusable`` history entries of the last integration, ``Orchestrator.reusable``:
    those with a ``reuse_key``, or those a migration keyed in a file from before content
    keys, #108): it measures only the new and changed items alone, but the combination
    steps again (they follow the order of the gains alone), and nothing when no item
    changed. One measurement takes the median ``eval_s`` of the run's integration rows,
    else twice that of an end-to-end evaluation (an A/B runs A and B), else
    :data:`AB_SECONDS`."""
    items = dict(kernels)  # label -> item
    passing = [
        rec
        for rec in read_jsonl(run.results_file())
        if rec.get("passed") and float(rec.get("speedup") or 0.0) > 1.0
    ]
    for rec in sorted(passing, key=lambda r: float(r["speedup"])):  # the fastest one last
        for name in rec.get("transforms") or []:
            items[ledger.snapshot_stem(name)] = str(run.history_dir() / Path(name).name)
    combos = [
        (float(rec["speedup"]), parts)
        for rec in passing
        if (parts := len(rec.get("transforms") or []) + len(rec.get("kernels") or [])) > 1
    ]
    seed = max(combos)[1] if combos else 1
    alone = _changed(items, reusable) if reusable else len(items)
    steps = alone + bool(combos) + max(len(items) - seed, 0) if alone else 0
    measured = [r for r in rows if r["target"] == ledger.E2E and r["backend"] == "integrate"]
    each = median_eval_s(measured)
    if each is None and (one := median_eval_s(systems_rows(rows))) is not None:
        each = 2 * one
    return steps, each or AB_SECONDS


def _changed(items: dict[str, str], history: list[dict[str, Any]]) -> int:
    """How many of ``items`` (label → item) no step of an integration's ``history`` holds
    with the same content (the sha256 of the item's file: a kernel's ``target=path``)."""
    hashes: dict[str, str | None] = {}

    def content(item: str) -> tuple[str, str | None]:
        if item not in hashes:
            target, sep, rest = item.partition("=")
            path = Path(rest if sep and "/" not in target else item)
            hashes[item] = truth.sha256_file(path) if path.is_file() else None
        return ledger.item_label(item), hashes[item]

    before = {
        content(item)
        for h in history
        for item in [*h.get("items", []), *(h.get("ab") or {}).get("a_items", [])]
    }
    return sum(content(item) not in before for item in items.values())
