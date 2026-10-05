"""Run budgets: wall-clock and USD limits, per-agent timeouts and advice for agents.

* ``max_hours`` / ``max_usd`` limit the whole run. Before a kernel or transform
  agent starts, :meth:`Budget.exhausted` checks the elapsed time and the sum of
  ``costs.json``; once the budget is spent the remaining agents are skipped, but
  integrate + report always run on whatever exists (``reserve`` of the time
  budget is kept for them).
* ``agent_minutes`` limits one agent session: the orchestrator wraps
  ``run_agent`` in ``asyncio.timeout`` (see :meth:`Budget.start_agent`), and
  each agent's ``max_budget_usd`` is lowered to the USD that is left.
* The evaluation tools append :meth:`Budget.feedback` to every result: the
  budget left and an ``advice`` (``continue`` / ``consider_stopping`` / ``stop``).
  A kernel evaluation within ``100 - SOL_STOP_PCT`` % of its speed of light
  (weighted ``pct_of_sol``, :mod:`kernel_agent.kernels.roofline`) also means ``stop``.

Time is measured from the start of this process (``optimize`` or ``resume``);
USD is the sum of ``costs.json``, so it covers the whole run.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Iterable
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

EVAL_TOOLS = ("evaluate_candidate", "evaluate_e2e")
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


def non_improving_streak(
    records: Iterable[dict[str, Any]], *, ok_key: str = "correct", start: float = 1.0
) -> int:
    """Evaluations since the last one that improved on the best result so far.

    ``start`` is the speedup to beat before any record (1.0 = the reference).
    Failed, incorrect and not-faster evaluations all extend the streak, except
    the library's prior winners (``PRIOR_HYPOTHESIS``).
    """
    best, streak = start, 0
    for rec in records:
        if improves(rec, best, ok_key=ok_key):
            best, streak = float(rec["speedup"]), 0
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
    started: float = field(default_factory=time.monotonic)
    deadlines: dict[str, float] = field(default_factory=dict)
    evals: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_config(cls, run: RunDir, cfg: OptimizeConfig) -> Budget:
        return cls(
            run,
            max_hours=cfg.max_hours,
            max_usd=cfg.max_usd,
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

    def agent_seconds_left(self) -> float | None:
        """Time left for agents; ``reserve`` of ``max_hours`` is kept for integrate + report."""
        if self.max_hours is None:
            return None
        return self.max_hours * 3600 * (1 - self.reserve) - self.elapsed_s()

    def exhausted(self) -> str | None:
        """Why no further kernel/transform agent may start, or None."""
        left = self.agent_seconds_left()
        if self.max_hours is not None and left is not None and left < MIN_AGENT_SECONDS:
            return (
                f"time budget spent ({self.elapsed_s() / 60:.1f} of "
                f"{self.max_hours * 60:.0f} min used, {self.reserve:.0%} kept for "
                "integrate + report)"
            )
        usd = self.usd_left()
        if self.max_usd is not None and usd is not None and usd < MIN_AGENT_USD:
            return f"USD budget spent (${self.spent_usd():.2f} of ${self.max_usd:.2f})"
        return None

    def note(self, phase: str, key: str, item: dict[str, Any]) -> None:
        note(self.run, phase, key, item)

    # -------------------------------------------------------- agent level

    def start_agent(self, name: str) -> float | None:
        """Register an agent session; returns its timeout in seconds (None = no limit)."""
        limits = []
        if self.agent_minutes is not None:
            limits.append(self.agent_minutes * 60)
        left = self.agent_seconds_left()
        if left is not None:
            limits.append(max(left, MIN_AGENT_SECONDS))
        timeout = min(limits) if limits else None
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
        ``evals_used`` counts the evaluations of the current agent session.
        ``pct_of_sol`` is the evaluation's weighted share of its speed of light
        (:func:`kernel_agent.kernels.roofline.sol_signal`); at ``SOL_STOP_PCT`` or
        more further work cannot pay off, so the advice is ``stop``.
        """
        used = self.evals[agent] = self.evals.get(agent, 0) + 1
        minutes = self.minutes_left(agent)
        streak = results_streak(results, ok_key=ok_key)
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
