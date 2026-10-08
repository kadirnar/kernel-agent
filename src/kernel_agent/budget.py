"""Run budgets: wall-clock and USD limits, per-agent timeouts and advice for agents.

* ``max_hours`` / ``max_usd`` limit the whole run. Before a kernel or transform
  agent starts, :meth:`Budget.exhausted` checks the elapsed time and the sum of
  ``costs.json``; once the budget is spent the remaining agents are skipped, but
  integrate + report always run on whatever exists (``reserve`` of the time
  budget is kept for them, or the improve loop's estimate of its final
  integration, ``final_reserve_s``, when that is longer). ``max_sessions``
  limits the agent sessions this process starts the same way, and a usage limit
  that resets only after the time budget ends (``blocked``,
  :mod:`kernel_agent.agent.auth`) stops new agents too.
* ``agent_minutes`` limits one agent session: the orchestrator wraps
  ``run_agent`` in ``asyncio.timeout`` (see :meth:`Budget.start_agent`), and
  each agent's ``max_budget_usd`` is lowered to the USD that is left.
* With concurrent sessions (``improve --agents N``, ``coordinator.py``) the USD of the
  running sessions is reserved (:meth:`Budget.reserve_usd`: the role's median cost so far,
  :meth:`Budget.expected_usd`): a session's cap is what is left minus the others'
  reservations (:meth:`Budget.free_usd`), and while they leave less than
  ``MIN_AGENT_USD``, or while the shared rate gate is closed (a usage limit), no session
  starts (:meth:`Budget.waiting`, part of :meth:`Budget.exhausted`). With one session
  there is no reservation and no gate.
* The evaluation tools append :meth:`Budget.feedback` to every result: the
  budget left and an ``advice`` (``continue`` / ``consider_stopping`` / ``stop``).
  The evaluation budget is the session's own (``agent.tools.SessionBinding``: each
  session's tools are bound to it), never a run-wide setting another session changes.
  A kernel evaluation within ``100 - SOL_STOP_PCT`` % of its recipe's roofline
  (weighted ``pct_of_sol``, :mod:`kernel_agent.kernels.roofline`) also means ``stop``:
  move on to another recipe (issue #166: a bound of the recipe, not of the model).
  A ``sweep_candidate`` call is one evaluation (one :meth:`Budget.feedback`), however
  many configs it times (:mod:`kernel_agent.kernels.sweep`); an ``evaluate_candidates`` or
  ``evaluate_e2e_batch`` call is one evaluation per candidate or set it ran (#190).

Time is measured from the start of this process (``optimize`` or ``resume``);
USD is the sum of ``costs.json``, so it covers the whole run.
"""

from __future__ import annotations

import contextlib
import dataclasses
import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kernel_agent import roles
from kernel_agent.config import OptimizeConfig
from kernel_agent.workspace import RunDir, append_jsonl, read_json, read_jsonl

if TYPE_CHECKING:
    from kernel_agent.coordinator import RateGate

PLATEAU = 4  # this many non-improving evaluations in a row -> "consider_stopping"
MIN_GAIN = 0.01  # an improvement beats the best by more than max(1 %, 2 x timing spread)
MIN_AGENT_SECONDS = 120.0  # do not start a kernel/transform agent with less time left
MIN_AGENT_USD = 0.25  # ... or with less money left
WRAP_UP_SECONDS = 120.0  # advice is "stop" when an agent has less time than this left
SOL_STOP_PCT = 90.0  # advice is "stop" once a kernel reaches this % of its recipe's roofline

EVAL_TOOLS = (
    "evaluate_candidate",
    "evaluate_candidates",
    "sweep_candidate",
    "evaluate_e2e",
    "evaluate_e2e_batch",
)
#: USD of one session of a role before the run has finished one (the median API-equivalent
#: cost per session of the runs studied for #174, docs/MULTIAGENT-DATA.md T2)
ROLE_USD = {
    "kernel": 1.77,
    "systems": 1.68,
    "native": 4.04,
    "research": 1.17,
    "dossier": 0.19,
    "planner": 0.78,
    "librarian": 0.31,
}
#: Hypothesis of the library's prior winners (library.py), evaluated before a target's agent
#: starts: they may set the best result, but are not the agent's attempts without a gain.
PRIOR_HYPOTHESIS = "prior winner from "


# ------------------------------------------------------------ advice signals


def _spread(rec: dict[str, Any]) -> float:
    """Worst relative timing spread of an evaluation (0 when not reported)."""
    spreads = [float(c.get("timing_spread") or 0.0) for c in rec.get("cases") or []]
    return max(spreads, default=float(rec.get("timing_spread") or 0.0))


def improves(rec: dict[str, Any], best: float, *, ok_key: str = "correct") -> bool:
    """Whether an evaluation record beats ``best`` by more than the noise.

    It must pass (``ok_key``: ``correct`` for kernels, ``passed`` for e2e) and
    its speedup must exceed ``best * (1 + max(MIN_GAIN, 2 * timing_spread))``.
    """
    speedup = rec.get("speedup")
    if not rec.get(ok_key) or speedup is None:
        return False
    return float(speedup) > best * (1 + max(MIN_GAIN, 2 * _spread(rec)))


def _snapshot_name(row: dict[str, Any]) -> str:
    return Path(str(row.get("snapshot") or "")).name


@dataclass
class Standing:
    """The results of one target that stand, collected in evaluation order: those that set
    a new best (``keep``) and the integration's re-evaluations of earlier snapshots
    (:mod:`kernel_agent.kernels.recheck`), each of which replaces its snapshot's earlier
    results. :attr:`best` is the speedup the next evaluation has to beat
    (:func:`improves`), so a stale record that a re-evaluation replaced never sets it.

    The one definition of the bar for ledger rows (:func:`kernel_agent.ledger.standing`,
    :func:`kernel_agent.ledger.best_kept`, the scheduler's arms) and ``results.jsonl``
    records (:func:`non_improving_streak`, the evaluation advice).
    """

    start: float = 1.0  # the speedup to beat before any result (1.0 = the reference)
    rows: list[dict[str, Any]] = field(default_factory=list)

    def keep(self, row: dict[str, Any]) -> None:
        """A result that set a new best."""
        self.rows.append(row)

    def replace(self, row: dict[str, Any]) -> None:
        """A re-evaluation: it stands (when correct) instead of its snapshot's earlier results."""
        name = _snapshot_name(row)
        self.rows = [r for r in self.rows if _snapshot_name(r) != name]
        if row.get("correct"):
            self.rows.append(row)

    @property
    def top(self) -> dict[str, Any] | None:
        """The fastest standing result faster than ``start`` (None: there is none)."""
        faster = [r for r in self.rows if float(r.get("speedup") or 0.0) > self.start]
        return max(faster, key=lambda r: float(r["speedup"]), default=None)

    @property
    def best(self) -> float:
        top = self.top
        return float(top["speedup"]) if top else self.start


def non_improving_streak(
    records: Iterable[dict[str, Any]], *, ok_key: str = "correct", start: float = 1.0
) -> int:
    """Evaluations since the last one that improved on the best result so far.

    ``start`` is the speedup to beat before any record (1.0 = the reference).
    Failed, incorrect and not-faster evaluations all extend the streak, except
    the library's prior winners (``PRIOR_HYPOTHESIS``). The integration's
    re-evaluations of earlier snapshots (``reevaluates``) are not the agent's: they
    extend nothing, but replace the snapshot's earlier record in the best so far
    (:class:`Standing`, like the ledger's keep bar).
    """
    stand, streak = Standing(start), 0
    for rec in records:
        if rec.get("reevaluates"):
            stand.replace(rec)
        elif improves(rec, stand.best, ok_key=ok_key):
            stand.keep(rec)
            streak = 0
        elif not str(rec.get("hypothesis") or "").startswith(PRIOR_HYPOTHESIS):
            streak += 1
    return streak


def results_streak(results: Path, *, ok_key: str = "correct") -> int:
    """:func:`non_improving_streak` of a ``results.jsonl`` file."""
    return non_improving_streak(read_jsonl(results), ok_key=ok_key)


# ------------------------------------------------------------ run.json / events


def note(run: RunDir, phase: str, key: str, item: dict[str, Any]) -> None:
    """Append ``item`` to ``run.json`` ``phases[phase][key]`` (and ``events.jsonl`` if present)."""

    def add(data: dict[str, Any]) -> None:
        data.setdefault("phases", {}).setdefault(phase, {}).setdefault(key, []).append(item)

    run.update(add)  # under run.json's lock: threads note and seal truth files meanwhile
    events = run.root / "events.jsonl"
    if events.exists():
        append_jsonl(
            events, {"time": time.strftime("%H:%M:%S"), "event": key, "phase": phase, **item}
        )


# ------------------------------------------------------------ budget


@dataclass
class Budget:
    run: RunDir
    max_hours: float | None = None
    max_usd: float | None = None
    agent_minutes: float | None = None
    reserve: float = 0.15
    eval_timeout_s: float = 300.0
    max_sessions: int | None = None
    started: float = field(default_factory=time.monotonic)
    deadlines: dict[str, float] = field(default_factory=dict)
    # agent -> its own session length (the native agent's longer one), else agent_minutes
    minutes_by_agent: dict[str, float] = field(default_factory=dict)
    evals: dict[str, int] = field(default_factory=dict)
    # agent -> evaluations in its results file when its plateau count restarted (a research plan)
    restarted: dict[str, int] = field(default_factory=dict)
    sessions: int = 0  # agent sessions started by this process (a resumed one counts once)
    blocked: str | None = None  # why no agent can run any more (a usage limit, auth.py)
    final_reserve_s: float = 0.0  # time the improve loop keeps for its final integration
    # re-estimates final_reserve_s after every evaluation (one may add an integration item)
    estimate_reserve: Callable[[], float] | None = None
    # concurrent sessions (coordinator.py): the USD each running session is expected to spend,
    # by its label (:meth:`reserve_usd`), the label of each running agent, the shared rate gate
    reservations: dict[str, float] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    gate: RateGate | None = None
    # the clock of the session deadlines (a virtual-time dry run's simulated one)
    monotonic: Callable[[], float] = field(default=time.monotonic, repr=False)
    # the evaluation tools' early termination (--early-stop, issue #190): the evaluator's
    # early discard and sweep racing
    early_stop: bool = True

    @classmethod
    def from_config(cls, run: RunDir, cfg: OptimizeConfig) -> Budget:
        return cls(
            run,
            max_hours=cfg.max_hours,
            max_usd=cfg.max_usd,
            max_sessions=cfg.max_sessions,
            agent_minutes=cfg.agent_minutes,
            reserve=cfg.budget_reserve,
            eval_timeout_s=cfg.eval_timeout_s,
            early_stop=cfg.early_stop,
        )

    # -------------------------------------------------------- run level

    def elapsed_s(self) -> float:
        return time.monotonic() - self.started

    def spent_usd(self) -> float:
        costs = read_json(self.run.root / "costs.json", {})
        return float(sum(c.get("usd") or 0.0 for c in costs.values()))

    def usd_left(self) -> float | None:
        return None if self.max_usd is None else self.max_usd - self.spent_usd()

    def reserve_usd(self, label: str, usd: float) -> None:
        """Session ``label`` starts and is expected to spend ``usd`` (until :meth:`release_usd`)."""
        self.reservations[label] = max(float(usd), 0.0)

    def release_usd(self, label: str) -> None:
        """Session ``label`` ended: its actual cost is in ``costs.json`` now."""
        self.reservations.pop(label, None)

    def reserved_usd(self, exclude: str | None = None) -> float:
        """USD the running sessions (but ``exclude``) are expected to spend still."""
        return sum(usd for label, usd in self.reservations.items() if label != exclude)

    def free_usd(self, label: str | None = None) -> float | None:
        """:meth:`usd_left` minus what the running sessions other than ``label`` reserved
        (None: no USD limit)."""
        left = self.usd_left()
        return None if left is None else left - self.reserved_usd(exclude=label)

    def expected_usd(self, role: str) -> float:
        """USD a session of ``role`` is expected to cost: the median of the run's sessions of
        that role (``costs.json``), else :data:`ROLE_USD`."""
        costs = read_json(self.run.root / "costs.json", {}) or {}
        done = [
            float(c.get("usd") or 0.0)
            for label, c in costs.items()
            if isinstance(c, dict) and (c.get("role") or roles.role_of(label)) == role
        ]
        return statistics.median(done) if done else ROLE_USD.get(role, 1.0)

    def seconds_left(self) -> float | None:
        """Time left of ``max_hours`` (None: no time limit)."""
        return None if self.max_hours is None else self.max_hours * 3600 - self.elapsed_s()

    def reserve_s(self) -> float:
        """Seconds of ``max_hours`` kept for integrate + report: ``reserve`` of it, or the
        improve loop's estimate of its final integration (``final_reserve_s``) if longer."""
        return max((self.max_hours or 0.0) * 3600 * self.reserve, self.final_reserve_s)

    def agent_seconds_left(self) -> float | None:
        """Time left for agents; :meth:`reserve_s` is kept for integrate + report."""
        left = self.seconds_left()
        return None if left is None else left - self.reserve_s()

    def exhausted(self, label: str | None = None) -> str | None:
        """Why no further kernel/transform agent may start, or None; ``label``: of a session
        that starts now (its own reservation does not count, :meth:`waiting`)."""
        if self.blocked:
            return self.blocked
        if self.max_sessions is not None and self.sessions >= self.max_sessions:
            return f"session budget spent ({self.sessions} of {self.max_sessions} agent sessions)"
        left = self.agent_seconds_left()
        if self.max_hours is not None and left is not None and left < MIN_AGENT_SECONDS:
            kept = f"{self.reserve:.0%}"
            if self.final_reserve_s > self.max_hours * 3600 * self.reserve:
                kept = f"{self.final_reserve_s / 60:.0f} min"
            return (
                f"time budget spent ({self.elapsed_s() / 60:.1f} of "
                f"{self.max_hours * 60:.0f} min used, {kept} kept for "
                "integrate + report)"
            )
        usd = self.usd_left()
        if self.max_usd is not None and usd is not None and usd < MIN_AGENT_USD:
            return f"USD budget spent (${self.spent_usd():.2f} of ${self.max_usd:.2f})"
        return self.waiting(label)

    def waiting(self, label: str | None = None) -> str | None:
        """Why no agent may start *for now* (None: it may): the shared rate gate is closed
        (a usage limit, until it resets), or the USD the running sessions other than
        ``label`` reserved leaves less than ``MIN_AGENT_USD``. Never with one session."""
        if self.gate is not None and (closed := self.gate.closed()):
            return closed
        reserved, free = self.reserved_usd(exclude=label), self.free_usd(label)
        if reserved and free is not None and free < MIN_AGENT_USD:
            left = self.usd_left() or 0.0
            return f"USD budget reserved (${reserved:.2f} of ${left:.2f} left, by running sessions)"
        return None

    def note(self, phase: str, key: str, item: dict[str, Any]) -> None:
        note(self.run, phase, key, item)

    # -------------------------------------------------------- agent level

    def start_agent(
        self, name: str, worked_s: float | None = None, label: str | None = None
    ) -> float | None:
        """Register an agent session; returns its timeout in seconds (None = no limit).

        ``worked_s``: the session is resumed after a usage limit and has run that long:
        it is not a new session, that time counts against ``agent_minutes`` and its
        evaluations so far still count. ``label``: the session's (its USD reservation)."""
        if label:
            self.labels[name] = label
        limits = []
        minutes = self.minutes_by_agent.get(name, self.agent_minutes)
        if minutes is not None:
            limits.append(max(minutes * 60 - (worked_s or 0.0), 0.0))
        left = self.agent_seconds_left()
        if left is not None:
            limits.append(max(left, MIN_AGENT_SECONDS))
        timeout = min(limits) if limits else None
        if worked_s is None:
            self.sessions += 1
            self.evals[name] = 0
        if timeout is None:
            self.deadlines.pop(name, None)
        else:
            self.deadlines[name] = self.monotonic() + timeout
        return timeout

    def end_agent(self, name: str) -> None:
        self.deadlines.pop(name, None)
        self.labels.pop(name, None)

    def extend_deadline(self, name: str, seconds: float) -> None:
        """Push agent ``name``'s deadline back by ``seconds`` its GPU jobs waited behind other
        jobs (:mod:`kernel_agent.gpuqueue`), so ``minutes_left`` and the ``stop`` advice leave
        that time out; never past the run's agent time (nor earlier than it was)."""
        if name not in self.deadlines or seconds <= 0:
            return
        when = self.deadlines[name] + seconds
        if (left := self.agent_seconds_left()) is not None:
            when = min(when, self.monotonic() + left)
        self.deadlines[name] = max(self.deadlines[name], when)

    def agent_config(self, cfg: OptimizeConfig, label: str | None = None) -> OptimizeConfig:
        """``cfg`` with the per-agent USD cap lowered to what is left of ``max_usd`` (minus
        what the running sessions other than ``label`` reserved)."""
        left = self.free_usd(label)
        if left is None:
            return cfg
        cap = max(left, 0.01)
        if cfg.budget_usd_per_agent is not None:
            cap = min(cap, cfg.budget_usd_per_agent)
        return dataclasses.replace(cfg, budget_usd_per_agent=round(cap, 2))

    def minutes_left(self, agent: str | None = None) -> float | None:
        lefts = [self.agent_seconds_left()]
        if agent in self.deadlines:
            lefts.append(self.deadlines[agent] - self.monotonic())
        known = [s for s in lefts if s is not None]
        return min(known) / 60 if known else None

    def prompt_note(self, name: str, cfg: OptimizeConfig, mcp_tools: list[str]) -> str:
        """``# Budget`` section appended to an agent's system prompt ("" when unlimited)."""
        lines = []
        minutes = self.minutes_left(name)
        if minutes is not None:
            lines.append(
                f"* Time: you have about {max(1, round(minutes))} min for this session; it is "
                "stopped after that (every evaluated snapshot is kept, so evaluate "
                "your best version before time runs out)."
            )
        if cfg.budget_usd_per_agent is not None:
            lines.append(f"* Cost: this session may spend at most ${cfg.budget_usd_per_agent:.2f}.")
        if any(t.endswith(EVAL_TOOLS) for t in mcp_tools):
            lines.append(
                "* Every evaluation result has `budget` (evals_used, evals_budget, "
                "minutes_left) and `advice`: `continue`; `consider_stopping` after "
                f"{PLATEAU} evaluations in a row that did not beat the best result (try a "
                "fundamentally different idea or finish); `stop` when the budget is spent "
                f"or the candidate reaches {SOL_STOP_PCT:.0f} % of its recipe's roofline "
                "(SOL; the next gain needs another recipe). Then write your summary and your "
                "next ideas and end the session."
            )
        return "\n\n# Budget\n" + "\n".join(lines) if lines else ""

    # -------------------------------------------------------- evaluation level

    def feedback(
        self,
        agent: str,
        results: Path,
        evals_budget: int | None,
        *,
        ok_key: str = "correct",
        pct_of_sol: float | None = None,
        evaluations: int = 1,
    ) -> dict[str, Any]:
        """``budget`` + ``advice`` for an evaluation tool result.

        Call once per evaluation, after it was appended to ``results``;
        ``evals_used`` counts the evaluations of the current agent session; the
        non-improving streak starts again after ``restarted[agent]`` evaluations.
        ``pct_of_sol`` is the evaluation's weighted share of its roofline
        (:func:`kernel_agent.kernels.roofline.sol_signal`); at ``SOL_STOP_PCT`` or
        more the recipe is at its bound, so the advice is ``stop`` (move on to another).
        ``evaluations``: the evaluations one tool call made (``evaluate_candidates``,
        ``evaluate_e2e_batch``: one per candidate or set).
        """
        used = self.evals[agent] = self.evals.get(agent, 0) + max(evaluations, 1)
        if self.estimate_reserve is not None:
            with contextlib.suppress(Exception):  # advice only: never fails an evaluation
                self.final_reserve_s = self.estimate_reserve()
        minutes = self.minutes_left(agent)
        records = read_jsonl(results)
        streak = non_improving_streak(records, ok_key=ok_key)
        if (since := self.restarted.get(agent)) is not None:  # re-evaluations are not the agent's
            done = sum(not rec.get("reevaluates") for rec in records)
            streak = min(streak, max(done - since, 0))
        usd = self.free_usd(self.labels.get(agent))  # the others' reservations are theirs
        if evals_budget is not None and used >= evals_budget:
            advice, why = "stop", f"evaluation budget used ({used} of {evals_budget})"
        elif minutes is not None and minutes * 60 < WRAP_UP_SECONDS:
            advice, why = "stop", f"time is up ({max(minutes, 0):.1f} min left)"
        elif usd is not None and usd < MIN_AGENT_USD:
            advice, why = "stop", "run USD budget spent"
        elif pct_of_sol is not None and pct_of_sol >= SOL_STOP_PCT:
            advice = "stop"
            why = (
                f"at {pct_of_sol:.0f} % of this recipe's roofline (SOL): the next gain needs "
                "another recipe; write it into your open ideas"
            )
        elif streak >= PLATEAU:
            advice, why = "consider_stopping", f"{streak} evaluations without a new best"
        else:
            advice, why = "continue", ""
        out: dict[str, Any] = {
            "budget": {
                "evals_used": used,
                "evals_budget": evals_budget,
                "minutes_left": None if minutes is None else round(max(minutes, 0.0), 1),
                "non_improving": streak,
            },
            "advice": advice,
        }
        if why:
            out["advice_reason"] = why
        return out
