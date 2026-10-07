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
* The evaluation tools append :meth:`Budget.feedback` to every result: the
  budget left and an ``advice`` (``continue`` / ``consider_stopping`` / ``stop``).
  A kernel evaluation within ``100 - SOL_STOP_PCT`` % of its speed of light
  (weighted ``pct_of_sol``, :mod:`kernel_agent.kernels.roofline`) also means ``stop``.
  A ``sweep_candidate`` call is one evaluation (one :meth:`Budget.feedback`), however
  many configs it times (:mod:`kernel_agent.kernels.sweep`).

Time is measured from the start of this process (``optimize`` or ``resume``);
USD is the sum of ``costs.json``, so it covers the whole run.
"""

from __future__ import annotations

import contextlib
import dataclasses
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kernel_agent.config import OptimizeConfig
from kernel_agent.workspace import RunDir, append_jsonl, read_json, read_jsonl, write_json

PLATEAU = 4  # this many non-improving evaluations in a row -> "consider_stopping"
MIN_GAIN = 0.01  # an improvement beats the best by more than max(1 %, 2 x timing spread)
MIN_AGENT_SECONDS = 120.0  # do not start a kernel/transform agent with less time left
MIN_AGENT_USD = 0.25  # ... or with less money left
WRAP_UP_SECONDS = 120.0  # advice is "stop" when an agent has less time than this left
SOL_STOP_PCT = 90.0  # advice is "stop" once a kernel reaches this % of its speed of light

EVAL_TOOLS = ("evaluate_candidate", "sweep_candidate", "evaluate_e2e")
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
    data = run.load()
    data.setdefault("phases", {}).setdefault(phase, {}).setdefault(key, []).append(item)
    write_json(run.run_json, data)
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
    kernel_evals: int | None = None
    transform_evals: int | None = None
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
            kernel_evals=cfg.evaluations_per_target,
            transform_evals=cfg.transform_evaluations,
        )

    # -------------------------------------------------------- run level

    def elapsed_s(self) -> float:
        return time.monotonic() - self.started

    def spent_usd(self) -> float:
        costs = read_json(self.run.root / "costs.json", {})
        return float(sum(c.get("usd") or 0.0 for c in costs.values()))

    def usd_left(self) -> float | None:
        return None if self.max_usd is None else self.max_usd - self.spent_usd()

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

    def exhausted(self) -> str | None:
        """Why no further kernel/transform agent may start, or None."""
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
        return None

    def note(self, phase: str, key: str, item: dict[str, Any]) -> None:
        note(self.run, phase, key, item)

    # -------------------------------------------------------- agent level

    def start_agent(self, name: str, worked_s: float | None = None) -> float | None:
        """Register an agent session; returns its timeout in seconds (None = no limit).

        ``worked_s``: the session is resumed after a usage limit and has run that long:
        it is not a new session, that time counts against ``agent_minutes`` and its
        evaluations so far still count."""
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
            self.deadlines[name] = time.monotonic() + timeout
        return timeout

    def end_agent(self, name: str) -> None:
        self.deadlines.pop(name, None)

    def agent_config(self, cfg: OptimizeConfig) -> OptimizeConfig:
        """``cfg`` with the per-agent USD cap lowered to what is left of ``max_usd``."""
        left = self.usd_left()
        if left is None:
            return cfg
        cap = max(left, 0.01)
        if cfg.budget_usd_per_agent is not None:
            cap = min(cap, cfg.budget_usd_per_agent)
        return dataclasses.replace(cfg, budget_usd_per_agent=round(cap, 2))

    def minutes_left(self, agent: str | None = None) -> float | None:
        lefts = [self.agent_seconds_left()]
        if agent in self.deadlines:
            lefts.append(self.deadlines[agent] - time.monotonic())
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
                f"or the candidate reaches {SOL_STOP_PCT:.0f} % of its speed of light "
                "(write your summary and end the session now)."
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
    ) -> dict[str, Any]:
        """``budget`` + ``advice`` for an evaluation tool result.

        Call once per evaluation, after it was appended to ``results``;
        ``evals_used`` counts the evaluations of the current agent session; the
        non-improving streak starts again after ``restarted[agent]`` evaluations.
        ``pct_of_sol`` is the evaluation's weighted share of its speed of light
        (:func:`kernel_agent.kernels.roofline.sol_signal`); at ``SOL_STOP_PCT`` or
        more further work cannot pay off, so the advice is ``stop``.
        """
        used = self.evals[agent] = self.evals.get(agent, 0) + 1
        if self.estimate_reserve is not None:
            with contextlib.suppress(Exception):  # advice only: never fails an evaluation
                self.final_reserve_s = self.estimate_reserve()
        minutes = self.minutes_left(agent)
        records = read_jsonl(results)
        streak = non_improving_streak(records, ok_key=ok_key)
        if (since := self.restarted.get(agent)) is not None:  # re-evaluations are not the agent's
            done = sum(not rec.get("reevaluates") for rec in records)
            streak = min(streak, max(done - since, 0))
        usd = self.usd_left()
        if evals_budget is not None and used >= evals_budget:
            advice, why = "stop", f"evaluation budget used ({used} of {evals_budget})"
        elif minutes is not None and minutes * 60 < WRAP_UP_SECONDS:
            advice, why = "stop", f"time is up ({max(minutes, 0):.1f} min left)"
        elif usd is not None and usd < MIN_AGENT_USD:
            advice, why = "stop", "run USD budget spent"
        elif pct_of_sol is not None and pct_of_sol >= SOL_STOP_PCT:
            advice = "stop"
            why = (
                f"within {max(100 - pct_of_sol, 0):.0f} % of speed of light "
                f"({pct_of_sol:.0f} % of SOL)"
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
