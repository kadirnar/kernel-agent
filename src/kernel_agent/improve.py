"""``kernel-agent improve``: the continuous optimisation loop (autoresearch style).

::

    until the budget is spent or every arm has stopped:
        pick the arm with the best score            (scheduler.py: Amdahl × UCB, stop rules)
            that has time for a slice, the final integration's time kept   (issue #100)
        it has plateaued: first a clean-context research session that writes its
            plan.md (research.py; at most one per --research-every slices of the arm)
        a kernel arm's first slice: first a short dossier session that writes its
            research.md from the documentation (web on, not --no-dossier; issue #125)
        run one slice: a fresh agent session with --slice evaluations, seeded with
            a digest (last ledger rows, ideas, best snapshot, plan.md, NOTES.md); a
            target with workers (--seeds-per-target, workers.py) gets one session per
            worker, concurrently up to --parallel, that share the slice's evaluations
        every --integrate-every kept results: measured re-integration
    every arm stopped: with --rounds > 1 and a real end-to-end gain in this round,
        re-profile the optimised model, re-plan with the prior rounds as context
        (rounds/<n>/) and continue with the new targets
    final integration + report

Slices go through ``Orchestrator.kernel_slice`` / ``systems_slice`` and so through
``Orchestrator._agent``: budgets, per-agent timeouts, ``program.md`` and the event
log apply to every session. Each slice is a new session, so the context of an
agent never grows beyond its digest.

State: ``improve.json`` in the run directory (slices, research sessions,
integrations, rounds and why the loop stopped). Everything else is read from the
ledger, so a restart after Ctrl-C or a crash continues where the loop stopped; a
slice that was running is recorded as ``interrupted`` together with the
evaluations it made. Ctrl-C or SIGTERM (``kernel_agent.interrupt``) also records
what the loop was doing in ``interrupted`` (moved to ``interruptions`` by the
restart).
"""

from __future__ import annotations

import asyncio
import dataclasses
import math
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kernel_agent import interrupt, ledger, objective, pivot, research, workers
from kernel_agent.budget import improves
from kernel_agent.config import OptimizeConfig
from kernel_agent.dashboard import refresh
from kernel_agent.integrate.owners import context_line
from kernel_agent.research import rows_table
from kernel_agent.scheduler import (
    INTEGRATION_SHARE,
    KERNEL,
    SHORT_SLICE,
    SYSTEMS,
    Arm,
    Policy,
    build_arms,
    integration_estimate,
    pick,
    plateau,
    rank,
    slice_seconds,
    snapshot_record,
)
from kernel_agent.workspace import RunDir, read_json, write_json

if TYPE_CHECKING:
    from kernel_agent.agent.runner import AgentResult
    from kernel_agent.orchestrator import Orchestrator

STATE = "improve.json"
LAST_ROWS = 15  # ledger rows of the arm in a slice digest
NOTES_CHARS = 4000  # tail of NOTES.md in a slice digest
IDEAS_CHARS = 2000
MAX_FAILED_SLICES = 3  # agent sessions in a row that raised: something is broken, stop
_IDEAS = re.compile(
    r"^#+[ \t]+[^\n]*\bideas?\b[^\n]*\n(.*?)(?=^#{1,6}[ \t]|\Z)", re.M | re.S | re.I
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] improve: {msg}", flush=True)


@dataclass
class ImproveConfig:
    slice: int = 4  # evaluations per slice (one fresh agent session)
    rounds: int = 1  # 1 = never re-profile / re-plan
    integrate_every: int = 4  # kept results between measured re-integrations (0: final only)
    max_slices: int | None = None  # slices in this invocation (None: until budget / plateau)
    research_every: int = 3  # slices of a target between its research sessions (0: none)
    policy: Policy = field(default_factory=Policy)
    # minutes the time budget keeps for the final integration (None: its estimate, at most
    # INTEGRATION_SHARE of --max-hours; --integration-reserve)
    integration_reserve: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------ state


def load_state(run: RunDir) -> dict[str, Any]:
    state = read_json(run.root / STATE, None)
    if isinstance(state, dict):
        return state
    return {
        "start_exp": len(ledger.rows(run)),  # rows before the first improve slice
        "slices": [],
        "research": [],
        "integrations": [],
        "rounds": [{"n": 1, "speedup": 1.0}],
    }


def _ts() -> float:
    return round(ledger.clock(), 3)


# ------------------------------------------------------------------ digests


def open_ideas(notes: str) -> str:
    """The ``## Open ideas`` section of a NOTES.md (any heading level containing "idea")."""
    match = _IDEAS.search(notes)
    return match.group(1).strip()[:IDEAS_CHARS] if match else ""


def _tail(text: str, limit: int = NOTES_CHARS) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[-limit:]
    return "…\n" + cut[cut.find("\n") + 1 :] if "\n" in cut else "…" + cut


def _notes(path: Path, what: str) -> list[str]:
    text = path.read_text() if path.exists() else ""
    ideas = open_ideas(text)
    return [
        "",
        f"## Open ideas (the `## Open ideas` section of {what})",
        ideas or "(none recorded yet)",
        "",
        f"## {what} (most recent part)",
        "```",
        _tail(text) or "(empty)",
        "```",
    ]


def _header(n: int, evaluations: int, history: str) -> list[str]:
    return [
        "",
        "",
        f"# Improve slice {n}",
        "`kernel-agent improve` works in short, fresh sessions (slices) of about "
        f"{evaluations} evaluations and gives the next slice to whichever target pays most. "
        f"This section is all you get from the earlier sessions; the full history is in "
        f"{history}.",
    ]


def _footer(notes: str) -> list[str]:
    return [
        "",
        "## Before you finish",
        f"Update `{notes}`: one line per evaluation (hypothesis → result) and a "
        "`## Open ideas` section with the untried ideas worth testing next, most promising "
        "first (with `idea_id`, expected gain and ceiling where you have them). The next "
        "session sees only that file, the ledger and a digest like this one.",
    ]


def refusals(run: RunDir, target_id: str) -> list[str]:
    """``## Refused by the integration``: this target's snapshots the last integration's
    re-check refused (``integration.json``), with the reason and, for memory errors under
    memcheck (kernels/memcheck.py), the sanitizer's first report."""
    data = read_json(run.root / "integration.json", {}) or {}
    refused = [r for r in data.get("recheck") or [] if r.get("target") == target_id]
    refused = [r for r in refused if not r.get("passed")]
    if not refused:
        return []
    lines = [
        "",
        "## Refused by the integration",
        "These snapshots passed the evaluator but not the integration's re-check; fix the "
        "cause before you build on one of them.",
    ]
    for r in refused:
        why = str(r.get("reason") or "")[:600]
        lines.append(f"* `history/{r.get('snapshot')}` ({r.get('status')}): {why}")
        if report := (r.get("memcheck") or {}).get("report"):
            lines += ["  ```", *[f"  {x}" for x in report.splitlines()], "  ```"]
    return lines


def kernel_digest(
    run: RunDir, arm: Arm, n: int, evaluations: int, policy: Policy, worker: int | None = None
) -> str:
    """Context of a fresh kernel-engineer session (bounded: no growth with the slice count);
    ``worker``: of that worker's session (its own NOTES.md, the target's shared ledger)."""
    lines = _header(n, evaluations, "`results.jsonl`, `NOTES.md` and `history/`")
    lines += ["", "## Best so far"]
    if arm.best_snapshot:
        sol = "" if arm.sol is None else f", {arm.sol:.0%} of the speed of light"
        lines.append(
            f"* `history/{arm.best_snapshot}`: {arm.best:.3f}x module speedup{sol}. Build on "
            f'it (`parent="history/{arm.best_snapshot}"`) unless you test a different approach.'
        )
    else:
        lines.append("* no correct candidate faster than the reference yet")
    lines += refusals(run, arm.id)
    rows = arm.rows[-LAST_ROWS:]
    if rows:
        lines += ["", f"## Last {len(rows)} evaluations (oldest first; `keep` = new best)", ""]
        lines += rows_table(rows)
    if stats := research.target_ideas(run, arm.id)[-LAST_ROWS:]:
        lines += ["", "## Ideas so far (`idea_id`; buggy = never correct: retry, not refuted)", ""]
        lines += research.ideas_table(stats)
    lines += research.plan_section(run, arm.id)
    lines += _notes(workers.notes_file(run, arm.id, worker), "NOTES.md")
    lines += [
        "",
        "## Where this target stands",
        # the scheduler's ms are the metric's (#114): per second of audio for throughput
        f"* It costs about {arm.remaining_ms:.1f} ms {_per(run)} now; the scheduler expects "
        f"{arm.headroom:.0%} of that can still go"
        + (f" ({arm.ceiling.describe()})." if arm.ceiling else "."),
        f"* {arm.streak} evaluations in a row without a new best; after {policy.patience} the "
        "target is stopped, so prefer a fundamentally different idea over small variations.",
    ]
    return "\n".join(lines + _footer("NOTES.md"))


def _per(run: RunDir) -> str:
    """What a ms of the run's metric is per: ``per model run`` (the latency), ``per second
    of generated audio`` (throughput), ``to first audio`` (ttfa)."""
    metric = objective.of(read_json(run.baseline_json, {}) or {})
    return "per model run" if metric.name == objective.LATENCY else metric.per


def systems_digest(run: RunDir, arm: Arm, n: int, evaluations: int, policy: Policy) -> str:
    lines = _header(n, evaluations, "`results.jsonl`, `NOTES.md` and `history/`")
    base = (read_json(run.baseline_json, {}) or {}).get("median_ms")
    lines += ["", "## Best end-to-end configuration so far (transforms, plus kernels if listed)"]
    if arm.best_snapshot and base:
        lines.append(
            f"* `{arm.best_snapshot}`: {float(base) / arm.best:.1f} ms vs the {float(base):.1f} "
            f"ms baseline ({arm.best:.3f}x); `+<target>` names a kernel it ran on top of"
        )
    else:
        lines.append("* no transform has beaten the baseline yet")
    integration = read_json(run.root / "integration.json", {}) or {}
    final = integration.get("final") or {}
    if final.get("passed"):
        accepted = ", ".join(
            f"`{ledger.item_label(i['item'])}`" for i in integration.get("accepted", [])
        )
        lines += [
            "",
            "## Current integration (everything accepted, measured together)",
            f"* {final.get('median_ms')} ms ({final.get('speedup')}x): {accepted}",
        ]
    ideas = [
        f"* `{t.get('id')}` (round {path.parent.name}): {t.get('idea')}"
        for path in sorted((run.root / "rounds").glob("*/plan.json"))
        for t in (read_json(path, {}) or {}).get("transforms", [])
    ]
    if ideas:
        lines += ["", "## Transform ideas of later planner rounds", *ideas[-10:]]
    rows = arm.rows[-LAST_ROWS:]
    if rows:
        lines += ["", f"## Last {len(rows)} evaluations (oldest first)", ""]
        lines += rows_table(rows)
    lines += _notes(run.transforms_dir / "NOTES.md", "NOTES.md")
    lines += [
        "",
        f"* {arm.streak} evaluations in a row without a new best; after {policy.patience} the "
        "systems agent is stopped.",
    ]
    return "\n".join(lines + _footer("NOTES.md"))


def rounds_context(
    run: RunDir,
    state: dict[str, Any],
    n: int,
    accepted: list[dict[str, Any]],
    arms: list[Arm],
    *,
    quality: str = "exact",
    owners: dict[str, dict[str, list[str]]] | None = None,
) -> str:
    """Planner context for round ``n``: what earlier rounds did and which targets exist (and,
    in a near-lossless run, how to move one of them to another precision: ``pivot.py``);
    ``owners``: the modules the accepted items change (``integration.json``)."""
    base = (read_json(run.baseline_json, {}) or {}).get("median_ms")
    lines = [
        "",
        "",
        f"# Round {n}: re-plan of an optimised model",
        f"`kernel-agent improve` has optimised this model for {n - 1} round(s). The baseline "
        "and the profile above were measured WITH the accepted optimisations applied, so the "
        f"hot spots have moved. The original baseline is {base} ms.",
        "",
        "## Rounds so far",
    ]
    for rnd in state["rounds"]:
        done = [i for i in state["integrations"] if i.get("round") == rnd["n"]]
        result = f"{done[-1]['speedup']:.3f}x" if done else "not integrated"
        lines.append(f"* round {rnd['n']}: started at {rnd.get('speedup', 1.0):.3f}x → {result}")
    lines += ["", "## Applied in this profile"]
    lines += [f"* {i['kind']} `{ledger.item_label(i['item'])}`" for i in accepted] or ["* none"]
    if owned := context_line(owners):  # integrate/owners.py
        lines += ["", owned]
    lines += ["", "## Existing targets (do not propose these module classes again)"]
    for arm in arms:
        if arm.kind == KERNEL:
            spec = read_json(run.target(arm.id) / "spec.json", {}) or {}
            tier = f" [{pivot.label(spec)}]" if spec.get("precision") else ""
            lines.append(
                f"* `{arm.id}` (`{arm.module_class}`){tier}: best {arm.best:.2f}x after "
                f"{arm.evals} evaluations; {arm.stop or 'still open'}"
            )
    lines += [
        "",
        "Propose only NEW targets: module classes that are hot in this profile and not listed "
        "above. An empty `targets` list is a valid answer when nothing new is worth a kernel. "
        "New transform ideas are passed on to the systems agent.",
    ]
    if quality == "near-lossless":
        lines += [
            "",
            "## Precision pivots",
            "To move an existing target to another precision tier (its precision was fixed "
            'when it was planned), list it under `pivots` instead: `{"target": id, '
            '"precision": ..., "precision_why": ...}`, the `precision_why` with the numbers '
            "behind it (its bound and ceiling, a passing transform that already uses that "
            "precision on its modules). kernel-agent captures it again in that tier as a new "
            "target `<id>__<precision>`; the old one keeps its results.",
        ]
    return "\n".join(lines)


# ------------------------------------------------------------------ the loop


class Improver:
    def __init__(
        self,
        orch: Orchestrator,
        icfg: ImproveConfig,
        *,
        require_capture: bool = True,
        live_charts: bool = True,
    ) -> None:
        self.orch = orch
        self.run = orch.run
        self.icfg = icfg
        self.policy = dataclasses.replace(
            icfg.policy, systems=icfg.policy.systems and orch.cfg.do_transforms
        )
        self.state = load_state(self.run)
        self.state.setdefault("research", [])  # improve.json from before research sessions
        self.require_capture = require_capture  # dry runs have no captures
        self.live_charts = live_charts
        self.kept = ""  # what the time budget keeps for the final integration (last logged)
        self.migrated = ""  # the migration of a pre-#93 integration.json (last logged)
        self.doing: str | None = None  # what the loop is busy with (``_doing``)

    # -------------------------------------------------------- helpers

    def save(self) -> None:
        write_json(self.run.root / STATE, self.state)

    @property
    def round(self) -> int:
        return int(self.state["rounds"][-1]["n"])

    def targets(self) -> list[str]:
        ids = self.run.target_ids()
        if self.require_capture:
            ids = [t for t in ids if self.run.capture_file(t).exists()]
        return ids

    def arms(self, rows: list[dict[str, Any]] | None = None) -> list[Arm]:
        return build_arms(
            self.run,
            self.policy,
            self.state["slices"],
            targets=self.targets(),
            rounds=self.state["rounds"],
            rows=rows,
            research=self.state["research"],
        )

    def research_due(self, arm: Arm) -> str | None:
        """Why ``arm`` gets a research session before its next slice, or None.

        It has plateaued (:func:`~kernel_agent.scheduler.plateau`), and its last
        research session, if any, was at least ``research_every`` of its slices ago
        and, if it wrote a plan, the plan led to a new best (a plan that did not
        leaves the plateau standing: the arm stops).
        """
        every = self.icfg.research_every
        why = plateau(arm, self.policy) if every > 0 else None
        done = [r for r in self.state["research"] if r["arm"] == arm.id]
        if why is None or not done:
            return why
        last = done[-1]
        if last.get("plan") and arm.best <= float(last.get("best") or 1.0):
            return None
        since = sum(
            s["arm"] == arm.id and s["n"] > last["after_slice"] for s in self.state["slices"]
        )
        return why if since >= every else None

    def dossier_due(self, arm: Arm) -> bool:
        """Whether ``arm`` gets a dossier session before its first slice (issue #125): a
        kernel arm with no slice, no ``research.md`` and no dossier session yet, in a run
        with the web tools and dossiers on (``Orchestrator.dossier``)."""
        cfg = self.orch.cfg
        if arm.kind != KERNEL or not (cfg.allow_web and cfg.dossier):
            return False
        if research.dossier_path(self.run, arm.id).is_file():
            return False
        earlier = [*self.state["slices"], *self.state.get("dossiers", [])]
        return not any(s["arm"] == arm.id for s in earlier)

    def _pickable(self) -> list[Arm]:
        """The arms, ranked; an arm stopped by its plateau waits for its research session."""
        arms = self.arms()
        for arm in arms:
            if arm.stop and self.research_due(arm):
                arm.stop = None
        return rank(arms, self.policy)

    def keeps_since_integration(self) -> int:
        done = self.state["integrations"]
        start = int(done[-1]["exp_after"] if done else self.state.get("start_exp", 0))
        return sum(
            r["status"] == ledger.KEEP and r["backend"] != "integrate"
            for r in ledger.rows(self.run)
            if (r["exp"] or 0) > start
        )

    def _charts(self) -> None:
        refresh(self.run)
        slices_chart(self.run)

    @contextmanager
    def _doing(self, what: str) -> Iterator[None]:
        """``what`` the loop is busy with, for the ``interrupted`` record (kept when the
        body raises: that is what was interrupted)."""
        outer, self.doing = self.doing, what
        yield
        self.doing = outer

    def _interrupted(self) -> None:
        """Ctrl-C, SIGTERM or a cancellation: ``interrupted`` in ``improve.json`` says what
        was running (the slice / research session records say it too)."""
        rec: dict[str, Any] = {
            "at": _ts(),
            "during": self.doing or "scheduling",
            "signal": interrupt.signal_name(),
        }
        self.state["interrupted"] = rec
        self.save()
        ledger.event(self.run, "interrupted", **rec)
        log(f"interrupted during {rec['during']}")

    # -------------------------------------------------------- time (issue #100)

    def _integration_estimate(self) -> tuple[int, float]:
        rows = ledger.rows(self.run)
        reusable, migrated = self.orch.reusable()  # a pre-#93 integration.json: migrated
        if migrated and migrated != self.migrated:
            log(migrated)
            self.migrated = migrated
        kernels = {  # the snapshot the integration takes (not the arm's keep bar)
            t: f"{t}={self.run.history_dir(t) / Path(best['snapshot']).name}"
            for t, best in self.orch._kernel_bests()
        }
        return integration_estimate(self.run, rows, kernels, reusable)

    def _reserve(self, estimate_s: float) -> float:
        """Seconds the time budget keeps for a final integration expected to take
        ``estimate_s``: ``--integration-reserve`` minutes, else (``auto``) the estimate, at
        most ``INTEGRATION_SHARE`` of ``--max-hours``."""
        if (minutes := self.icfg.integration_reserve) is not None:
            return minutes * 60
        return min(estimate_s, INTEGRATION_SHARE * (self.orch.budget.max_hours or 0.0) * 3600)

    def _keep_time(self) -> None:
        """Keep the final integration's time out of the agents' time
        (``Budget.final_reserve_s``, :meth:`_reserve` of its expected duration,
        :func:`~kernel_agent.scheduler.integration_estimate`), estimated again after every
        evaluation (an evaluation may add an item): no session runs into it (the evaluation
        advice says ``stop``), and no slice starts in it. A final integration longer than
        that runs past ``--max-hours`` (:meth:`_past_budget`)."""
        budget = self.orch.budget
        if budget.max_hours is None:
            return
        steps, each = self._integration_estimate()
        budget.final_reserve_s = self._reserve(steps * each)
        budget.estimate_reserve = lambda: self._reserve(math.prod(self._integration_estimate()))
        what = f"integrate + report ({budget.reserve:.0%} of --max-hours)"
        estimate = f"{steps} A/B measurements × {each / 60:.1f} min"
        if budget.final_reserve_s > budget.max_hours * 3600 * budget.reserve:
            what = f"the final integration ({estimate})"
            over = (steps * each - budget.final_reserve_s) / 60
            if self.icfg.integration_reserve is not None:
                what = (
                    f"the final integration (--integration-reserve; estimated "
                    f"{steps * each / 60:.0f} min: {estimate})"
                )
            elif over >= 0.5:
                what = (
                    f"the final integration ({INTEGRATION_SHARE:.0%} of --max-hours; estimated "
                    f"{steps * each / 60:.0f} min: {estimate}, so it may run {over:.0f} min "
                    "past --max-hours)"
                )
        if (kept := f"{budget.reserve_s() / 60:.0f} min kept for {what}") != self.kept:
            log(kept)
            self.kept = kept

    def _past_budget(self) -> None:
        """Log it when the final integration is expected to end after ``--max-hours`` (its
        reserve was capped, or the loop stopped late): it runs to the end all the same, on
        the GPU alone (no agent sessions)."""
        left = self.orch.budget.seconds_left()
        if left is None:
            return
        with suppress(Exception):  # a log line only: it never keeps the integration from running
            steps, each = self._integration_estimate()
            if steps * each > max(left, 0.0):
                log(
                    f"the final integration ({steps} A/B measurements × {each / 60:.1f} min ≈ "
                    f"{steps * each / 60:.0f} min) runs past --max-hours "
                    f"({max(left, 0.0) / 60:.0f} min left): it uses the GPU only, no agent sessions"
                )

    def _in_time(self, arms: list[Arm]) -> tuple[Arm | None, str]:
        """The best live arm with time left for one slice
        (:func:`~kernel_agent.scheduler.slice_seconds`, ``SHORT_SLICE`` × that when its last
        slice ran out of time without an evaluation), or None and why there is none."""
        live = [a for a in arms if a.stop is None]
        left = self.orch.budget.agent_seconds_left()
        if left is None:
            return live[0], ""
        need = {a.id: slice_seconds(a) * (SHORT_SLICE if a.short else 1.0) for a in live}
        arm = next((a for a in live if need[a.id] <= left), None)
        if arm is not None and arm is not live[0]:
            log(
                f"{live[0].id}: a slice needs {need[live[0].id] / 60:.1f} min, "
                f"{max(left, 0) / 60:.1f} min left; {arm.id} instead"
            )
        if arm is not None:
            return arm, ""
        least = min(live, key=lambda a: need[a.id])
        total = max(self.orch.budget.seconds_left() or 0.0, 0.0)
        return None, (
            f"time left {total / 60:.1f} min < one slice of {least.id} "
            f"({need[least.id] / 60:.1f} min: warm-up, one evaluation, wrap-up) + {self.kept}"
        )

    def _session_time(self, arm: Arm) -> dict[str, Any]:
        """``limit_s`` (the time the run's budget leaves a slice's session, when less than
        ``--agent-minutes``) and ``need_s`` (:func:`~kernel_agent.scheduler.slice_seconds`)
        for its record: with less than ``SHORT_SLICE`` × ``need_s`` and no evaluation the
        slice ran out of time (``budget_short``) and is not counted against the arm."""
        budget = self.orch.budget
        left = budget.agent_seconds_left()
        if left is None or (budget.agent_minutes is not None and budget.agent_minutes * 60 <= left):
            return {}
        return {"limit_s": round(max(left, 0.0)), "need_s": round(slice_seconds(arm))}

    # -------------------------------------------------------- main

    async def improve(self) -> str:
        """Loop until the budget is spent or every arm has stopped; returns why it stopped."""
        self._recover()
        self.orch.phase = "improve"
        self.orch.budget.kernel_evals = self.orch.budget.transform_evals = self.icfg.slice
        self.state["config"] = self.icfg.to_dict()
        self.state.pop("finished", None)
        self.save()
        ledger.event(self.run, "phase_start", phase="improve")
        try:
            reason = await self._loop()
            log(f"stopping: {reason}")
            await self._finish(reason)
        except BaseException as exc:  # Ctrl-C too: the state and the event log stay consistent
            if isinstance(exc, KeyboardInterrupt | asyncio.CancelledError):
                self._interrupted()
            ledger.event(self.run, "phase_failed", phase="improve", error=repr(exc)[:300])
            raise
        ledger.event(self.run, "phase_done", phase="improve")
        return reason

    async def _loop(self) -> str:
        done = failed = 0
        while True:
            interrupt.check()
            self._keep_time()
            if reason := self.orch.budget.exhausted():
                return reason
            if self.icfg.max_slices is not None and done >= self.icfg.max_slices:
                return f"--max-slices {self.icfg.max_slices} reached"
            await self.orch.seed_library(self.targets())  # library priors before any slice
            arms = self._pickable()
            arm = pick(arms)
            if arm is None:
                if await self._next_round(arms):
                    continue
                return (
                    "every arm has stopped (" + "; ".join(f"{a.id}: {a.stop}" for a in arms) + ")"
                )
            arm, short = self._in_time(arms)  # no slice without time for one evaluation
            if arm is None:
                return short
            if (why := self.research_due(arm)) is not None:
                with self._doing(f"research session of {arm.id}"):
                    await self._research(arm, why)
                continue  # its plan restarts the arm's patience; the next slice reads it
            if self.dossier_due(arm):
                with self._doing(f"dossier of {arm.id}"):
                    await self._dossier(arm)
                continue  # the time it took counts: pick again
            with self._doing(f"slice {len(self.state['slices']) + 1} ({arm.id})"):
                rec = await self._slice(arm, arms)
            done += 1
            failed = failed + 1 if rec["status"] == "failed" else 0
            if failed >= MAX_FAILED_SLICES:
                return f"{failed} agent sessions in a row failed (last: {rec.get('error')})"
            every = self.icfg.integrate_every  # 0: only the final integration
            if every > 0 and (kept := self.keeps_since_integration()) >= every:
                await self.reintegrate(f"{kept} kept results since the last integration")
            elif self.live_charts:
                slices_chart(self.run)

    async def _finish(self, reason: str) -> None:
        integrated = (self.run.root / "integration.json").exists()
        if self.keeps_since_integration() or not integrated:
            self._past_budget()
            await self.reintegrate("final integration")
        self.state["finished"] = {"reason": reason, "at": _ts()}
        self.save()
        self.orch._mark("improve", reason=reason, slices=len(self.state["slices"]))
        slices_chart(self.run)
        await self.orch.report()  # report.md, charts and dashboard

    # -------------------------------------------------------- slices

    async def _slice(self, arm: Arm, arms: list[Arm]) -> dict[str, Any]:
        n = len(self.state["slices"]) + 1
        label = f"{arm.agent}#{n}"
        info = arm.summary()
        rec: dict[str, Any] = {
            "n": n,
            "arm": arm.id,
            "round": self.round,
            "agent": arm.agent,
            "label": label,
            "status": "running",
            "started": _ts(),
            "exp_before": len(ledger.rows(self.run)),
            "best_before": arm.best,
            **{k: info[k] for k in ("remaining_ms", "headroom", "expected_ms", "index", "score")},
            "why": info["why"],  # the score's components (scheduler.Arm.why, issue #122)
            **self._session_time(arm),
        }
        self.state["slices"].append(rec)
        self.save()
        others = ", ".join(f"{a.id} {a.score:.3g}" for a in arms if a is not arm and not a.stop)
        log(
            f"slice {n}: {arm.id} (best {arm.best:.2f}x, expected gain {arm.expected_ms:.3g} ms, "
            f"score {arm.score:.3g}: {info['why']}; others: {others or 'none'})"
        )
        ledger.event(
            self.run,
            "slice_start",
            slice=n,
            arm=arm.id,
            score=info["score"],
            expected_ms=info["expected_ms"],
            why=info["why"],
        )
        evaluations = self.icfg.slice
        try:
            if arm.kind == SYSTEMS:
                digest = systems_digest(self.run, arm, n, evaluations, self.policy)
                results = [
                    await self.orch.systems_slice(
                        evaluations=evaluations, digest=digest, label=label
                    )
                ]
            elif team := self._team(arm, evaluations):
                rec["workers"] = [s.worker for s in team]
                self.save()
                results = await self._workers(arm, team, n, rec)
            else:
                self._restart_advice(arm)
                digest = kernel_digest(self.run, arm, n, evaluations, self.policy)
                results = [
                    await self.orch.kernel_slice(
                        arm.id, evaluations=evaluations, digest=digest, label=label
                    )
                ]
        except Exception as exc:  # an SDK / CLI failure must not end an unattended loop
            if interrupt.requested():  # it failed because the stop ended its processes
                self._close(rec, "interrupted")
                raise interrupt.Interrupted from exc
            log(f"slice {n}: agent session failed: {exc!r}")
            rec["error"] = repr(exc)[:500]
            self._close(rec, "failed")
            return rec
        except BaseException:  # Ctrl-C, cancellation
            self._close(rec, "interrupted")
            raise
        if interrupt.requested():  # sessions ended by the stop: not a finished slice
            self._close(rec, "interrupted")
            raise interrupt.Interrupted
        status = "done"
        if any(r.is_error for r in results) or rec.get("error"):
            status = "error"
        if any(r.timed_out for r in results):
            status = "timed_out"
        self._close(rec, status, usd=sum(r.cost_usd for r in results))
        return rec

    def _team(self, arm: Arm, evaluations: int) -> list[workers.Seed]:
        """The worker sessions of a kernel slice ([]: one classic session). With
        ``--reseed-workers`` the slices after the arm's first worker slice start from its
        two best snapshots (``workers.reseeds``)."""
        team = self.orch.kernel_seeds(arm.id, evaluations)
        earlier = [s for s in self.state["slices"] if s["arm"] == arm.id and s.get("workers")]
        if team and earlier and self.orch.cfg.reseed_workers:
            team = workers.reseeds(self.run, arm.id, team, evaluations, self.orch.truth) or team
        return team

    async def _workers(
        self, arm: Arm, team: list[workers.Seed], n: int, rec: dict[str, Any]
    ) -> list[AgentResult]:
        """One session per worker, concurrently up to ``--parallel``, each with its share of
        the slice's evaluations and a digest made when it starts (so a session that waited
        for its slot sees the results of the ones before it). A session that raised is
        logged in ``rec``; the slice fails only when every session raised."""
        sem = asyncio.Semaphore(max(1, self.orch.cfg.parallel))
        self._restart_advice(arm, [workers.agent_name(arm.id, s.worker) for s in team])
        log(
            f"slice {n}: {arm.id}: {len(team)} workers, "
            + ", ".join(f"w{s.worker} {s.evaluations} evals ({s.origin})" for s in team)
        )

        async def one(seed: workers.Seed) -> AgentResult:
            async with sem:
                now = next((a for a in self.arms() if a.id == arm.id), arm)
                digest = kernel_digest(
                    self.run, now, n, seed.evaluations, self.policy, worker=seed.worker
                )
                return await self.orch.worker_session(
                    arm.id,
                    seed,
                    team,
                    prompt=f"Continue optimising target `{arm.id}` as worker {seed.worker} "
                    "(see the `# Worker` section). Read the `# Improve slice` section first: "
                    "it says where the previous sessions left off.",
                    digest=digest,
                    label=f"{workers.agent_name(arm.id, seed.worker)}#{n}",
                )

        out = await asyncio.gather(*(one(s) for s in team), return_exceptions=True)
        failed = [r for r in out if isinstance(r, BaseException)]
        for exc in failed:
            if not isinstance(exc, Exception):  # cancellation: the slice is interrupted
                raise exc
        if failed and len(failed) == len(out):
            raise failed[0]
        if failed:
            log(f"slice {n}: {len(failed)} of {len(out)} worker sessions failed: {failed[0]!r}")
            rec["error"] = repr(failed[0])[:500]
        return [r for r in out if not isinstance(r, BaseException)]

    def _restart_advice(self, arm: Arm, agents: list[str] | None = None) -> None:
        """Let the evaluation advice count the arm's plateau from its last research plan,
        as the scheduler does (else the first evaluation after a plan says stop);
        ``agents``: the sessions it applies to (default: the arm's classic session)."""
        plans = [r["exp"] for r in self.state["research"] if r["arm"] == arm.id and r.get("plan")]
        done = sum((r["exp"] or 0) <= max(plans) for r in arm.rows) if plans else None
        for agent in agents or [arm.agent]:
            if done is None:
                self.orch.budget.restarted.pop(agent, None)
            else:
                self.orch.budget.restarted[agent] = done

    def _close(
        self,
        rec: dict[str, Any],
        status: str,
        *,
        usd: float | None = None,
        ended: float | None = None,
    ) -> None:
        arm = next((a for a in self.arms() if a.id == rec["arm"]), None)
        new = [r for r in (arm.rows if arm else []) if (r["exp"] or 0) > rec["exp_before"]]
        ended = _ts() if ended is None else ended
        best = arm.best if arm else rec["best_before"]
        rec.update(
            status=status,
            ended=ended,
            seconds=round(max(ended - rec["started"], 0.0), 1),
            evals=len(new),
            keeps=sum(r["status"] == ledger.KEEP for r in new),
            failures=sum(r["status"] in ledger.FAILURES for r in new),
            best_after=best,
            improved=best > rec["best_before"],
        )
        if usd is not None:
            rec["usd"] = round(usd, 4)
        short = "limit_s" in rec and rec["limit_s"] < SHORT_SLICE * rec["need_s"]
        if short and not new and status in ("done", "timed_out"):
            rec["budget_short"] = True  # out of time: not counted against the arm (scheduler)
            log(
                f"slice {rec['n']}: no evaluation with {rec['limit_s'] / 60:.1f} min of the time "
                f"budget left: not counted against {rec['arm']}"
            )
        self.save()
        ledger.event(
            self.run,
            "slice_done",
            slice=rec["n"],
            arm=rec["arm"],
            status=status,
            evals=len(new),
            improved=rec["improved"],
            best=round(best, 4),
        )
        log(
            f"slice {rec['n']}: {rec['arm']} {status}, {len(new)} evaluations, best "
            f"{rec['best_before']:.2f}x → {best:.2f}x"
        )

    async def _research(self, arm: Arm, why: str) -> dict[str, Any]:
        """A research session for a plateaued arm (``Orchestrator.research``) and its record.

        ``plan`` in the record: whether the session wrote a new ``plan.md``; only
        then does the arm's count of evaluations without a new best restart. ``pivot``:
        what became of a precision pivot it proposed (``pivot.json``, ``pivot.py``): the
        new target, or why it was refused."""
        n = len(self.state["slices"])
        plan = research.plan_path(self.run, arm.id)
        before = plan.read_bytes() if plan.is_file() else None
        proposal = pivot.proposal_path(self.run, arm.id)
        proposed = proposal.read_bytes() if proposal.is_file() else None
        rec: dict[str, Any] = {
            "n": len(self.state["research"]) + 1,
            "arm": arm.id,
            "round": self.round,
            "after_slice": n,
            "label": f"research-{arm.id}#{n}",
            "why": why,
            "status": "running",
            "started": _ts(),
            "exp": len(ledger.rows(self.run)),
            "best": arm.best,
        }
        self.state["research"].append(rec)
        self.save()
        log(f"research: {arm.id} has plateaued ({why}); clean-context review of the target")
        ledger.event(self.run, "research_start", arm=arm.id, why=why, label=rec["label"])
        usd = None
        try:
            result = await self.orch.research(arm.id, reason=why, label=rec["label"])
            usd = result.cost_usd
            status = "timed_out" if result.timed_out else "error" if result.is_error else "done"
        except Exception as exc:  # like a failed slice: the loop goes on without a plan
            if interrupt.requested():
                self._close_research(rec, "interrupted", plan=False)
                raise interrupt.Interrupted from exc
            log(f"research: {arm.id}: agent session failed: {exc!r}")
            rec["error"] = repr(exc)[:500]
            status = "failed"
        except BaseException:  # Ctrl-C, cancellation
            self._close_research(rec, "interrupted", plan=False)
            raise
        wrote = plan.is_file() and plan.read_bytes() != before
        self._close_research(rec, status, plan=wrote, usd=usd)
        if proposal.is_file() and proposal.read_bytes() != proposed:  # a new arm in its tier
            found = pivot.read_proposal(proposal) or {}
            rec["pivot"] = await self.orch.pivot(arm.id, found, source=rec["label"])
            self.save()
        return rec

    async def _dossier(self, arm: Arm) -> dict[str, Any]:
        """The dossier session of a kernel arm before its first slice and its record
        (``improve.json`` → ``dossiers``; ``file``: it wrote ``research.md``)."""
        rec: dict[str, Any] = {
            "arm": arm.id,
            "label": f"dossier-{arm.id}",
            "status": "running",
            "started": _ts(),
        }
        self.state.setdefault("dossiers", []).append(rec)
        self.save()
        try:
            result = await self.orch.dossier(arm.id, label=rec["label"])
        except BaseException:  # Ctrl-C, cancellation (Orchestrator.dossier keeps the rest)
            rec.update(status="interrupted", ended=_ts())
            self.save()
            raise
        status = "failed" if result is None else "done"
        if result is not None and (result.timed_out or result.is_error):
            status = "timed_out" if result.timed_out else "error"
        ended = _ts()
        rec.update(status=status, ended=ended, seconds=round(ended - rec["started"], 1))
        rec["file"] = research.dossier_path(self.run, arm.id).is_file()
        if result is not None:
            rec["usd"] = round(result.cost_usd, 4)
        self.save()
        ledger.event(self.run, "dossier_done", arm=arm.id, status=status, file=rec["file"])
        where = f"targets/{arm.id}/{research.DOSSIER_FILE}" if rec["file"] else "no dossier"
        log(f"dossier: {arm.id} {status} in {rec['seconds'] / 60:.1f} min; {where}")
        return rec

    def _close_research(
        self,
        rec: dict[str, Any],
        status: str,
        *,
        plan: bool,
        usd: float | None = None,
        ended: float | None = None,
    ) -> None:
        ended = _ts() if ended is None else ended
        rec.update(
            status=status,
            ended=ended,
            seconds=round(max(ended - rec["started"], 0.0), 1),
            plan=plan,
        )
        if usd is not None:
            rec["usd"] = round(usd, 4)
        self.save()
        ledger.event(self.run, "research_done", arm=rec["arm"], status=status, plan=plan)
        where = f"targets/{rec['arm']}/{research.PLAN_FILE}"
        log(f"research: {rec['arm']} {status}; " + (f"plan in {where}" if plan else "no new plan"))

    def _recover(self) -> None:
        """Close slices and research sessions left ``running`` by a process that did not
        exit cleanly (an interrupted research session counts as one without a plan); the
        last invocation's ``interrupted`` record goes to ``interruptions``."""
        if last := self.state.pop("interrupted", None):
            log(f"the last invocation was interrupted during {last['during']}; continuing")
            self.state.setdefault("interruptions", []).append(last)
            self.save()
        for res in self.state["research"]:
            if res.get("status") == "running":
                self._close_research(res, "interrupted", plan=False, ended=res["started"])
        for rec in self.state.get("dossiers", []):  # not run again (Improver.dossier_due)
            if rec.get("status") == "running":
                rec.update(status="interrupted", ended=rec["started"])
                self.save()
        for rec in self.state["slices"]:
            if rec.get("status") != "running":
                continue
            arm = next((a for a in self.arms() if a.id == rec["arm"]), None)
            times = [
                t
                for r in (arm.rows if arm else [])
                if (r["exp"] or 0) > rec["exp_before"] and (t := ledger.epoch(r["time"]))
            ]
            log(f"slice {rec['n']} ({rec['arm']}) was interrupted; recording what it did")
            self._close(rec, "interrupted", ended=max([rec["started"], *times]))

    # -------------------------------------------------------- integration and rounds

    async def reintegrate(self, why: str) -> dict[str, Any]:
        """Measured end-to-end integration (``Orchestrator.integrate``) and its record."""
        done = self.state["integrations"]
        before = max([1.0, *(float(i["speedup"]) for i in done)])
        exp_before = len(ledger.rows(self.run))
        log(f"re-integrating: {why}")
        ledger.event(self.run, "reintegrate", why=why)
        with self._doing(f"integration ({why})"):
            await self.orch.integrate(reuse=True)
        data = self.orch.truth.load_json(self.run.root / "integration.json") or {}
        final = data.get("final") or {}
        speedup = float(final["speedup"]) if final.get("passed") and final.get("speedup") else 1.0
        spread = ledger.e2e_spread(final) or 0.0
        rec = {
            "n": len(done) + 1,
            "at": _ts(),
            "round": self.round,
            "why": why,
            "exp_before": exp_before,
            "exp_after": len(ledger.rows(self.run)),
            "speedup": speedup,
            "spread": spread,
            "median_ms": final.get("median_ms"),
            "accepted": [ledger.item_label(i["item"]) for i in data.get("accepted", [])],
            "reused": (data.get("reuse") or {}).get("reused"),  # measurements not repeated
            "gain": improves(
                {"passed": True, "speedup": speedup, "timing_spread": spread},
                before,
                ok_key="passed",
            ),
        }
        done.append(rec)
        self.save()
        ledger.event(self.run, "integrated", speedup=speedup, median_ms=final.get("median_ms"))
        self._charts()
        return rec

    async def _next_round(self, arms: list[Arm]) -> bool:
        """Every arm has stopped: start round n+1 if allowed and this round paid off."""
        if self.round >= self.icfg.rounds:
            return False
        if self.keeps_since_integration() or not self.state["integrations"]:
            await self.reintegrate(f"every arm of round {self.round} has stopped")
        last = self.state["integrations"][-1]
        start = float(self.state["rounds"][-1].get("speedup") or 1.0)
        measured = {"passed": True, "speedup": last["speedup"], "timing_spread": last.get("spread")}
        if not improves(measured, start, ok_key="passed"):
            log(f"round {self.round} brought no real end-to-end gain; no further round")
            return False
        return await self._start_round(last, arms)

    async def _start_round(self, integration: dict[str, Any], arms: list[Arm]) -> bool:
        n = self.round + 1
        round_dir = self.run.root / "rounds" / str(n)
        integrated = self.orch.truth.load_json(self.run.root / "integration.json") or {}
        accepted = integrated.get("accepted", [])
        names = ", ".join(ledger.item_label(i["item"]) for i in accepted)
        log(f"round {n}: re-profiling the optimised model ({names or 'nothing applied'})")
        ledger.event(self.run, "reprofile", round=n, items=[i["item"] for i in accepted])
        with self._doing(f"re-profile for round {n}"):
            info = await asyncio.to_thread(self.orch.reprofile, round_dir, accepted)
        if "error" in info or not info.get("median_ms"):
            log(f"round {n}: re-profile failed:\n{str(info.get('error'))[-800:]}")
            ledger.event(self.run, "round_failed", round=n, error=str(info.get("error"))[:300])
            return False
        applied = {}
        for item in accepted:
            if item["kind"] == "kernel":
                target_id, _, path = item["item"].partition("=")
                rec = snapshot_record(self.run, target_id, path) or {}
                applied[target_id] = float(rec.get("speedup") or 1.0)
        quality = self.orch.cfg.quality
        owners = integrated.get("owners")
        context = rounds_context(
            self.run, self.state, n, accepted, arms, quality=quality, owners=owners
        )
        with self._doing(f"re-plan for round {n}"):
            new = await self.orch.replan(round_dir, context, label=f"planner#round{n}")
            captured = await self.orch.capture_targets(new) if new else []
            captured += await self._pivots(round_dir, n)
        self.state["rounds"].append(
            {
                "n": n,
                "started": _ts(),
                "dir": f"rounds/{n}",
                "profile": f"rounds/{n}/profile/profile.json",
                "baseline_ms": float(info["median_ms"]),
                "speedup": integration["speedup"],
                "applied": applied,
                "targets": captured,
            }
        )
        self.save()
        ledger.event(self.run, "round_start", round=n, targets=captured)
        log(f"round {n}: {info['median_ms']:.1f} ms; new targets: {captured or 'none'}")
        return bool(captured)

    async def _pivots(self, round_dir: Path, n: int) -> list[str]:
        """The precision pivots the round's re-plan proposed (``pivots`` in its
        ``plan.json``, ``pivot.py``): the ids of the new targets captured."""
        plan = read_json(round_dir / "plan.json", {}) or {}
        moved = []
        for proposal in plan.get("pivots") or []:
            if not isinstance(proposal, dict) or not proposal.get("target"):
                continue
            target_id = str(proposal["target"])
            if target_id not in self.run.target_ids():
                log(f"round {n}: pivot of unknown target {target_id!r} ignored")
                continue
            done = await self.orch.pivot(target_id, proposal, source=f"planner#round{n}")
            if done.get("target"):
                moved.append(str(done["target"]))
        return moved


# ------------------------------------------------------------------ report + chart


def report_lines(run: RunDir) -> list[str]:
    """``## Improve loop`` section of report.md (empty when ``improve`` never ran)."""
    state = read_json(run.root / STATE, None)
    if not isinstance(state, dict) or not state.get("slices"):
        return []
    slices = state["slices"]
    lines = ["", "## Improve loop", ""]
    if (run.root / "improve.png").exists():
        lines += ["![improve slices](improve.png)", ""]
    finished = state.get("finished") or {}
    lines += [
        f"* {len(slices)} slices, {sum(s.get('evals') or 0 for s in slices)} evaluations, "
        f"{len(state.get('integrations', []))} re-integrations, {len(state['rounds'])} round(s)",
        f"* stopped: {finished.get('reason', 'not finished (interrupted or running)')}",
        "",
        "| arm | precision | slices | evaluations | slices with a new best | best |",
        "|---|---|---|---|---|---|",
    ]
    for arm in dict.fromkeys(s["arm"] for s in slices):
        mine = [s for s in slices if s["arm"] == arm]
        best = max(float(s.get("best_after") or s.get("best_before") or 1.0) for s in mine)
        spec = read_json(run.target(arm) / "spec.json", {}) or {}
        tier = pivot.label(spec) if arm != SYSTEMS else "—"
        lines.append(
            f"| `{arm}` | {tier} | {len(mine)} | {sum(s.get('evals') or 0 for s in mine)} | "
            f"{sum(bool(s.get('improved')) for s in mine)} | {best:.3f}x |"
        )
    integrations = state.get("integrations", [])
    if integrations:
        lines += ["", "Re-integrations: " + ", ".join(f"{i['speedup']:.3f}x" for i in integrations)]
    if dossiers := state.get("dossiers") or []:
        done = ", ".join(
            f"`{d['arm']}` ("
            + ("wrote it" if d.get("file") else f"none: {d.get('status')}")
            + (f", {d['seconds'] / 60:.1f} min" if d.get("seconds") is not None else "")
            + ")"
            for d in dossiers
        )
        lines += ["", f"Research dossiers (`targets/<id>/research.md`, issue #125): {done}"]
    sessions = state.get("research") or []
    if sessions:
        lines += ["", "Research sessions on plateaued targets (`targets/<id>/plan.md`):", ""]
        for r in sessions:
            outcome = "wrote a plan" if r.get("plan") else f"no plan ({r.get('status')})"
            if moved := (r.get("pivot") or {}).get("target"):
                outcome += f", precision pivot to `{moved}`"
            elif refused := (r.get("pivot") or {}).get("refused"):
                outcome += f", precision pivot refused ({refused})"
            later = [s for s in slices if s["arm"] == r["arm"] and s["n"] > r["after_slice"]]
            best = max([float(r.get("best") or 1.0), *(s.get("best_after") or 0 for s in later)])
            lines.append(
                f"* `{r['arm']}` after slice {r['after_slice']} ({r['why']}): {outcome}; "
                f"best {float(r.get('best') or 1.0):.3f}x → {best:.3f}x since"
            )
    return [*lines, ""]


def slices_chart(run: RunDir) -> Path | None:
    """``improve.png``: one lane per arm, a bar per slice, re-integrations and rounds."""
    from kernel_agent import charts

    state = read_json(run.root / STATE, None)
    if not isinstance(state, dict) or not state.get("slices") or not charts.available():
        return None
    start = ledger.start_time(run)
    if start is None:
        return None
    return charts._render(
        run.root / "improve.png", (10.0, 4.6), lambda fig, ax: _draw_slices(ax, state, start)
    )


def _draw_slices(ax: Any, state: dict[str, Any], start: float) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from kernel_agent import charts

    slices = state["slices"]
    arms = list(dict.fromkeys(s["arm"] for s in slices))
    lane = {a: i for i, a in enumerate(arms)}

    def minutes(ts: float | None) -> float:
        return max((ts or start) - start, 0.0) / 60

    end = max(
        [minutes(s.get("ended") or s.get("started")) for s in slices]
        + [minutes(i.get("at")) for i in state.get("integrations", [])],
        default=1.0,
    )
    ax.set_xlim(0, max(end, 1.0) * 1.03)
    ax.set_ylim(len(arms) + 0.05, -0.6)  # room below the last lane for the integration labels
    ax.set_yticks(range(len(arms)), arms)
    ax.grid(axis="y", visible=False)
    for s in slices:
        a, b = minutes(s["started"]), minutes(s.get("ended") or s["started"])
        failed = s.get("status") in ("interrupted", "timed_out", "error", "failed")
        color = charts.KEEP_COLOR if s.get("improved") else charts.DISCARD_COLOR
        ax.barh(
            lane[s["arm"]],
            max(b - a, 0.3),
            left=a,
            height=0.56,
            color=color,
            edgecolor=charts.FAIL_COLOR if failed else charts.SURFACE,
            linewidth=1.4 if failed else 0.8,
            hatch="///" if failed else None,
            zorder=3,
        )
        if s.get("improved") and charts._fits(ax, f"{s['best_after']:.2f}x", 7.5, a, b, 2):
            ax.text(
                (a + b) / 2,
                lane[s["arm"]],
                f"{s['best_after']:.2f}x",
                ha="center",
                va="center",
                fontsize=7.5,
                color=charts._ink_on(charts.KEEP_COLOR),
                zorder=4,
            )
    sessions = [r for r in state.get("research") or [] if r["arm"] in lane]
    for res in sessions:  # research sessions sit between the arm's slices
        ax.plot(
            (minutes(res["started"]) + minutes(res.get("ended") or res["started"])) / 2,
            lane[res["arm"]],
            marker="D",
            markersize=6.5,
            color=charts.INK_2 if res.get("plan") else charts.FAIL_COLOR,
            markeredgecolor=charts.SURFACE,
            linestyle="none",
            zorder=5,
        )
    for rnd in state.get("rounds", [])[1:]:
        x = minutes(rnd.get("started"))
        ax.axvline(x, color=charts.INK_2, lw=1.2, zorder=2)
        ax.annotate(
            f"round {rnd['n']}",
            xy=(x, 1),
            xycoords=("data", "axes fraction"),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8.5,
            color=charts.INK_2,
        )
    for integ in state.get("integrations", []):
        x = minutes(integ.get("at"))
        ax.axvline(x, color=charts.PROJECTED_COLOR, lw=1.0, ls=(0, (4, 3)), zorder=2)
        ax.annotate(
            f"{integ['speedup']:.2f}x",
            xy=(x, 0),
            xycoords=("data", "axes fraction"),
            xytext=(-3, 4),
            textcoords="offset points",
            rotation=90,
            ha="right",
            va="bottom",
            fontsize=8,
            color=charts.PROJECTED_COLOR,
        )
    ax.set_xlabel("wall-clock time since the run started (min)")
    integrations = state.get("integrations", [])
    best = max([1.0, *(float(i["speedup"]) for i in integrations)])
    finished = (state.get("finished") or {}).get("reason", "running").split(" (")[0]
    charts._header(
        ax,
        f"improve: {len(slices)} slices, measured end to end 1.00 → {best:.2f}x",
        charts._short(
            f"{sum(s.get('evals') or 0 for s in slices)} evaluations, "
            f"{len(integrations)} re-integrations, {len(state.get('rounds', []))} round(s) · "
            f"stopped: {finished}",
            120,
        ),
    )
    charts._legend(
        ax,
        [
            Patch(color=charts.KEEP_COLOR, label="slice found a new best"),
            Patch(color=charts.DISCARD_COLOR, label="no new best"),
            Patch(
                facecolor=charts.DISCARD_COLOR,
                edgecolor=charts.FAIL_COLOR,
                hatch="///",
                label="interrupted, failed or timed out",
            ),
            Line2D([], [], color=charts.PROJECTED_COLOR, ls=(0, (4, 3)), label="re-integration"),
            *(
                [Line2D([], [], color=charts.INK_2, marker="D", ls="none", label="research plan")]
                if sessions
                else []
            ),
        ],
    )


# ------------------------------------------------------------------ entry point


@contextmanager
def _interrupt_note(run: RunDir) -> Iterator[None]:
    try:
        yield
    except (KeyboardInterrupt, asyncio.CancelledError):
        log(f"interrupted; `kernel-agent improve {run.root}` continues this run")
        raise


async def improve(
    ref: str,
    cfg: OptimizeConfig,
    icfg: ImproveConfig,
    *,
    dry_run: bool = False,
    seed: int = 0,
) -> RunDir:
    """``kernel-agent improve``: continue the run at ``ref`` or start one for a model.

    ``cfg.max_hours`` / ``cfg.max_usd`` / ``cfg.max_sessions`` are the budget of this
    invocation: hours from now, USD on top of what the run has already spent, and
    agent sessions started from now. A usage limit is waited out inside the slice
    (``Orchestrator._wait_for_limit``), so the loop survives it.
    """
    from kernel_agent import dryrun
    from kernel_agent.orchestrator import Orchestrator

    path = Path(ref).expanduser()
    if (path / "run.json").exists():
        marked = bool(RunDir(path.resolve()).load().get("dry_run"))
        if marked != dry_run:
            raise SystemExit(
                f"{path} is {'a' if marked else 'not a'} dry-run run; "
                + ("pass --dry-run" if marked else "--dry-run only continues dry-run runs")
            )
        overrides: dict[str, Any] = {
            "max_hours": cfg.max_hours,
            "max_usd": cfg.max_usd,
            "max_sessions": cfg.max_sessions,
            **({"auth": cfg.auth} if cfg.auth != "auto" else {}),  # else the run's
            **{
                k: getattr(cfg, k)
                for k in ("agent_minutes", "program", "budget_usd_per_agent")
                if getattr(cfg, k) is not None
            },
            **{
                k: False
                for k in ("use_library", "librarian", "allow_web", "dossier")
                if not getattr(cfg, k)
            },
            **({"web_domains": cfg.web_domains} if cfg.web_domains else {}),
            **({"seeds_per_target": cfg.seeds_per_target} if cfg.seeds_per_target else {}),
            **({"reseed_workers": True} if cfg.reseed_workers else {}),
            **({"parallel": cfg.parallel} if cfg.parallel > 1 else {}),
        }
        orch = Orchestrator.resume(path, overrides)
    elif dry_run:
        orch = Orchestrator(dryrun.create_run(cfg, seed), cfg)
    else:
        orch = Orchestrator.create(cfg)
    if cfg.max_usd is not None:
        orch.budget.max_usd = orch.budget.spent_usd() + cfg.max_usd
    log(f"run directory: {orch.run.root}")
    world = dryrun.World(orch) if dry_run else None
    with _interrupt_note(orch.run), world.installed() if world else nullcontext():
        if not all(orch._phase_done(p) for p in ("analyze", "plan", "capture")):
            await orch.run_all(until="capture")
        await Improver(orch, icfg, require_capture=not dry_run, live_charts=not dry_run).improve()
    return orch.run
