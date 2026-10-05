"""Scheduler of ``kernel-agent improve``: which arm gets the next slice, and when an arm stops.

An *arm* is a kernel target (``targets/<id>``) or the systems agent
(:data:`SYSTEMS`: model-level transforms measured end to end). The arms are
rebuilt from the ledger, the profile(s) and the slice log before every
decision, so the scheduler keeps no state of its own and a restarted loop
decides exactly as an uninterrupted one would.

Expected gain (Amdahl), in ms per run of the whole model::

    expected = remaining_ms × headroom × decay ** stale

* ``remaining_ms``: what the arm still costs. Kernel targets: their share of the
  profiled time × the baseline ms ÷ their best module speedup so far. Systems:
  the end-to-end time with the best transform so far.
* ``headroom``: the share of ``remaining_ms`` that could still go. ``1 −
  pct_of_sol`` when the best result carries a trustworthy speed-of-light
  estimate (``pct_of_sol``, :mod:`kernel_agent.kernels.roofline`), else ``1 − 1 / further`` where
  ``further = max(estimate / best, MIN_FURTHER)`` is the speedup still expected
  (``estimate``: the module speedup assumed reachable, :attr:`Policy.estimate`;
  for systems the GPU-idle share of the profile or :attr:`Policy.systems_estimate`).
* ``stale``: consecutive slices of this arm that found no new best.

UCB on the observed gain per evaluation (as in KernelBand): ``rate`` is the ms
per run an arm's kept results saved, divided by its evaluations. Each arm's
index is ``rate / best rate + explore × sqrt(2 ln(N + 2) / (n + 1))`` (``n``
its evaluations, ``N`` all evaluations), and ``score = expected × index``: the
live arm with the highest score gets the next slice. Untried arms get the
largest exploration bonus; arms that keep paying get more slices.

Stop rules per arm (AutoKernel's move-on rules, :func:`stop_reason`): ``patience``
consecutive evaluations without a new best (across slices), ≥ ``sol_stop`` of
the speed of light, ``target_hours`` spent in its slices, or the module speedup
``speedup_goal`` reached. The loop stops when the budget is spent or every arm
has stopped (:mod:`kernel_agent.improve`).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kernel_agent import ledger
from kernel_agent.budget import PRIOR_HYPOTHESIS, SOL_STOP_PCT, improves
from kernel_agent.kernels.roofline import sol_signal
from kernel_agent.workspace import RunDir, read_json, read_jsonl

SYSTEMS = "systems"  # pseudo-target: the systems agent's slices
KERNEL = "kernel"
MIN_FURTHER = 1.1  # without a SOL estimate, assume at least 10 % more is always possible
IDLE_SLICES = 2  # an arm whose last slices made no evaluation at all is stopped


@dataclass(frozen=True)
class Policy:
    """Scheduler constants and per-arm stop rules (None / 0 disables a rule)."""

    patience: int = 5  # consecutive evaluations without a new best
    sol_stop: float | None = SOL_STOP_PCT / 100  # fraction of the speed of light
    target_hours: float | None = 2.0  # time in one arm's slices
    speedup_goal: float | None = 2.0  # module speedup of a kernel target
    decay: float = 0.7  # per consecutive slice without a new best
    explore: float = 1.0  # UCB exploration weight
    estimate: float = 2.0  # module speedup assumed reachable without a SOL estimate
    systems_estimate: float = 1.25  # end-to-end speedup assumed reachable by transforms
    systems: bool = True  # schedule systems-agent slices


@dataclass
class Arm:
    id: str
    kind: str  # KERNEL | SYSTEMS
    ref_ms: float  # what the arm costs per run at 1.0× (kernels: share × baseline)
    module_class: str | None = None
    best: float = 1.0  # best kept speedup (module for kernels, end to end for systems)
    best_snapshot: str | None = None
    sol: float | None = None  # fraction of the speed of light of the best result
    estimate: float = 2.0
    evals: int = 0
    gain_ms: float = 0.0  # ms per run saved by the arm's kept results
    streak: int = 0  # evaluations since the last new best
    stale: int = 0  # slices since the last slice that found a new best
    idle: int = 0  # finished slices in a row without a single evaluation
    hours: float = 0.0  # time spent in this arm's improve slices
    expected_ms: float = 0.0
    index: float = 0.0  # UCB index
    score: float = 0.0
    stop: str | None = None
    rows: list[dict[str, Any]] = field(default_factory=list, repr=False)

    @property
    def agent(self) -> str:
        return SYSTEMS if self.kind == SYSTEMS else f"kernel-{self.id}"

    @property
    def remaining_ms(self) -> float:
        return self.ref_ms / max(self.best, 1e-9)

    @property
    def headroom(self) -> float:
        if self.sol is not None:
            return min(max(1.0 - self.sol, 0.0), 1.0)
        further = max(self.estimate / max(self.best, 1e-9), MIN_FURTHER)
        return 1.0 - 1.0 / further

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
    """Profiles newest first: ``{shares, baseline_ms, applied}`` per round (round 1 = the run's)."""
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
            }
        )
    return out


def _ref_ms(target_id: str, spec: dict[str, Any], profiles: list[dict[str, Any]]) -> float:
    """Time per run of the target's class at 1.0×, from the newest profile that has the class.

    A re-profile (improve rounds) measured the model with the integrated kernels,
    so a class whose kernel was applied there ran ``applied`` × faster.
    """
    cls = spec.get("module_class")
    for prof in profiles:
        if cls in prof["shares"]:
            share, instances = prof["shares"][cls]
            if spec.get("qualname") and instances > 1:
                share /= instances  # restricted to one instance of the class
            applied = float(prof["applied"].get(target_id) or 1.0)
            return share * prof["baseline_ms"] * applied
    return 0.0


def gpu_busy(run: RunDir) -> float | None:
    profile = read_json(run.profile_dir / "profile.json", {}) or {}
    busy = ledger._num((profile.get("kernel_view") or {}).get("gpu_busy_fraction"))
    return busy if busy and 0.0 < busy <= 1.0 else None


def systems_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """End-to-end rows of the systems agent (integration steps are not its evaluations)."""
    return [r for r in rows if r["target"] == ledger.E2E and r["backend"] != "integrate"]


def _improves(row: dict[str, Any], best: float) -> bool:
    rec = {"passed": row["correct"], "speedup": row["speedup"], "timing_spread": row["spread"]}
    return improves(rec, best, ok_key="passed")


def _kernel_history(arm: Arm, rows: list[dict[str, Any]]) -> None:
    """Best, gain and streak of a kernel target from its ledger rows (``keep`` = new best;
    the library's prior winners do not extend the streak)."""
    for row in rows:
        if row["status"] == ledger.KEEP and row["speedup"]:
            new = float(row["speedup"])
            arm.gain_ms += arm.ref_ms * (1.0 / arm.best - 1.0 / new)
            arm.best, arm.best_snapshot, arm.streak = new, row["snapshot"], 0
        elif not str(row["hypothesis"] or "").startswith(PRIOR_HYPOTHESIS):
            arm.streak += 1


def _systems_history(arm: Arm, rows: list[dict[str, Any]]) -> None:
    """Like :func:`_kernel_history`; only transform-only runs can set a new best.

    The ledger's ``keep`` of an ``e2e`` row compares with integration results too,
    and runs that include kernels measure the kernels' gain as well.
    """
    for row in rows:
        if row["backend"] == "transform" and row["speedup"] and _improves(row, arm.best):
            new = float(row["speedup"])
            arm.gain_ms += arm.ref_ms * (1.0 / arm.best - 1.0 / new)
            arm.best, arm.best_snapshot, arm.streak = new, row["snapshot"], 0
        else:
            arm.streak += 1


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
) -> list[Arm]:
    """Every arm with its history, stop reason and score (live arms first, best first)."""
    rows = ledger.rows(run) if rows is None else rows
    profiles = _profiles(run, rounds or [])
    base_ms = profiles[-1]["baseline_ms"] if profiles else 0.0
    arms: list[Arm] = []
    for target_id in run.target_ids() if targets is None else targets:
        spec = read_json(run.target(target_id) / "spec.json", {}) or {}
        estimate = ledger._num(spec.get("expected_speedup")) or policy.estimate
        arm = Arm(
            target_id,
            KERNEL,
            _ref_ms(target_id, spec, profiles),
            module_class=spec.get("module_class"),
            estimate=estimate,
            rows=[r for r in rows if r["target"] == target_id],
        )
        _kernel_history(arm, arm.rows)
        arm.sol = sol_fraction(snapshot_record(run, target_id, arm.best_snapshot))
        arms.append(arm)
    if policy.systems:
        busy = gpu_busy(run)
        estimate = max(policy.systems_estimate, 1.0 / busy if busy else 0.0)
        arm = Arm(SYSTEMS, SYSTEMS, base_ms, estimate=estimate, rows=systems_rows(rows))
        _systems_history(arm, arm.rows)
        arms.append(arm)
    for arm in arms:
        mine = [s for s in slices if s.get("arm") == arm.id]
        arm.hours = sum(float(s.get("seconds") or 0.0) for s in mine) / 3600
        for s in reversed(mine):
            if s.get("improved"):
                break
            arm.stale += 1
        for s in reversed(mine):
            if s.get("evals") or s.get("status") == "interrupted":
                break
            arm.idle += 1
        arm.evals = len(arm.rows)
        arm.stop = stop_reason(arm, policy)
    return rank(arms, policy)


# ------------------------------------------------------------------ decisions


def stop_reason(arm: Arm, policy: Policy) -> str | None:
    """Why an arm gets no more slices (None: it is live)."""
    if policy.patience and arm.streak >= policy.patience:
        return f"plateau: {arm.streak} evaluations in a row without a new best"
    if arm.idle >= IDLE_SLICES:
        return f"no evaluation in its last {arm.idle} slices"
    if policy.sol_stop and arm.sol is not None and arm.sol >= policy.sol_stop:
        return f"at {arm.sol:.0%} of the speed of light (stop at {policy.sol_stop:.0%})"
    if policy.target_hours and arm.hours >= policy.target_hours:
        return f"time cap: {arm.hours:.1f} h in its slices (cap {policy.target_hours:g} h)"
    if arm.kind == KERNEL and policy.speedup_goal and arm.best >= policy.speedup_goal:
        return f"speedup goal reached: {arm.best:.2f}x (goal {policy.speedup_goal:g}x)"
    return None


def rank(arms: list[Arm], policy: Policy) -> list[Arm]:
    """Fill in expected gain, UCB index and score; live arms first, highest score first."""
    total = sum(a.evals for a in arms)
    top = max((a.rate for a in arms), default=0.0)
    for arm in arms:
        arm.expected_ms = arm.remaining_ms * arm.headroom * policy.decay**arm.stale
        exploit = arm.rate / top if top > 0 else 0.0
        arm.index = exploit + policy.explore * math.sqrt(2 * math.log(total + 2) / (arm.evals + 1))
        arm.score = arm.expected_ms * arm.index
    return sorted(arms, key=lambda a: (a.stop is not None, -a.score, a.id))


def pick(arms: list[Arm]) -> Arm | None:
    """The live arm with the highest score (arms as returned by :func:`build_arms`)."""
    return next((a for a in arms if a.stop is None), None)
