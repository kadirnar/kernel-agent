"""``kernel-agent improve``: the continuous optimisation loop (autoresearch style).

::

    until the budget is spent or every arm has stopped:
        pick the arm with the best score            (scheduler.py: Amdahl × UCB, stop rules)
        it has plateaued: first a clean-context research session that writes its
            plan.md (research.py; at most one per --research-every slices of the arm)
        run one slice: a fresh agent session with --slice evaluations, seeded with
            a digest (last ledger rows, ideas, best snapshot, plan.md, NOTES.md)
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
evaluations it made.
"""

from __future__ import annotations

import asyncio
import dataclasses
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kernel_agent import ledger, research
from kernel_agent.budget import improves
from kernel_agent.config import OptimizeConfig
from kernel_agent.dashboard import refresh
from kernel_agent.research import rows_table
from kernel_agent.scheduler import (
    KERNEL,
    SYSTEMS,
    Arm,
    Policy,
    build_arms,
    pick,
    plateau,
    rank,
    snapshot_record,
)
from kernel_agent.workspace import RunDir, read_json, write_json

if TYPE_CHECKING:
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
    integrate_every: int = 4  # kept results between measured re-integrations
    max_slices: int | None = None  # slices in this invocation (None: until budget / plateau)
    research_every: int = 3  # slices of a target between its research sessions (0: none)
    policy: Policy = field(default_factory=Policy)

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


def kernel_digest(run: RunDir, arm: Arm, n: int, evaluations: int, policy: Policy) -> str:
    """Context of a fresh kernel-engineer session (bounded: no growth with the slice count)."""
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
    rows = arm.rows[-LAST_ROWS:]
    if rows:
        lines += ["", f"## Last {len(rows)} evaluations (oldest first; `keep` = new best)", ""]
        lines += rows_table(rows)
    if stats := research.target_ideas(run, arm.id)[-LAST_ROWS:]:
        lines += ["", "## Ideas so far (`idea_id`; buggy = never correct: retry, not refuted)", ""]
        lines += research.ideas_table(stats)
    lines += research.plan_section(run, arm.id)
    lines += _notes(run.target(arm.id) / "NOTES.md", "NOTES.md")
    lines += [
        "",
        "## Where this target stands",
        f"* It costs about {arm.remaining_ms:.1f} ms per model run now; the scheduler expects "
        f"{arm.headroom:.0%} of that can still go.",
        f"* {arm.streak} evaluations in a row without a new best; after {policy.patience} the "
        "target is stopped, so prefer a fundamentally different idea over small variations.",
    ]
    return "\n".join(lines + _footer("NOTES.md"))


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
    run: RunDir, state: dict[str, Any], n: int, accepted: list[dict[str, Any]], arms: list[Arm]
) -> str:
    """Planner context for round ``n``: what earlier rounds did and which targets exist."""
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
    lines += ["", "## Existing targets (do not propose these module classes again)"]
    for arm in arms:
        if arm.kind == KERNEL:
            lines.append(
                f"* `{arm.id}` (`{arm.module_class}`): best {arm.best:.2f}x after {arm.evals} "
                f"evaluations; {arm.stop or 'still open'}"
            )
    lines += [
        "",
        "Propose only NEW targets: module classes that are hot in this profile and not listed "
        "above. An empty `targets` list is a valid answer when nothing new is worth a kernel. "
        "New transform ideas are passed on to the systems agent.",
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
            ledger.event(self.run, "phase_failed", phase="improve", error=repr(exc)[:300])
            raise
        ledger.event(self.run, "phase_done", phase="improve")
        return reason

    async def _loop(self) -> str:
        done = failed = 0
        while True:
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
            if (why := self.research_due(arm)) is not None:
                await self._research(arm, why)
                continue  # its plan restarts the arm's patience; the next slice reads it
            rec = await self._slice(arm, arms)
            done += 1
            failed = failed + 1 if rec["status"] == "failed" else 0
            if failed >= MAX_FAILED_SLICES:
                return f"{failed} agent sessions in a row failed (last: {rec.get('error')})"
            if (kept := self.keeps_since_integration()) >= self.icfg.integrate_every:
                await self.reintegrate(f"{kept} kept results since the last integration")
            elif self.live_charts:
                slices_chart(self.run)

    async def _finish(self, reason: str) -> None:
        integrated = (self.run.root / "integration.json").exists()
        if self.keeps_since_integration() or not integrated:
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
        }
        self.state["slices"].append(rec)
        self.save()
        others = ", ".join(f"{a.id} {a.score:.0f}" for a in arms if a is not arm and not a.stop)
        log(
            f"slice {n}: {arm.id} (best {arm.best:.2f}x, expected gain {arm.expected_ms:.1f} ms, "
            f"score {arm.score:.0f}; others: {others or 'none'})"
        )
        ledger.event(
            self.run,
            "slice_start",
            slice=n,
            arm=arm.id,
            score=info["score"],
            expected_ms=info["expected_ms"],
        )
        evaluations = self.icfg.slice
        try:
            if arm.kind == SYSTEMS:
                digest = systems_digest(self.run, arm, n, evaluations, self.policy)
                result = await self.orch.systems_slice(
                    evaluations=evaluations, digest=digest, label=label
                )
            else:
                self._restart_advice(arm)
                digest = kernel_digest(self.run, arm, n, evaluations, self.policy)
                result = await self.orch.kernel_slice(
                    arm.id, evaluations=evaluations, digest=digest, label=label
                )
        except Exception as exc:  # an SDK / CLI failure must not end an unattended loop
            log(f"slice {n}: agent session failed: {exc!r}")
            rec["error"] = repr(exc)[:500]
            self._close(rec, "failed")
            return rec
        except BaseException:  # Ctrl-C, cancellation
            self._close(rec, "interrupted")
            raise
        status = "timed_out" if result.timed_out else "error" if result.is_error else "done"
        self._close(rec, status, usd=result.cost_usd)
        return rec

    def _restart_advice(self, arm: Arm) -> None:
        """Let the evaluation advice count the arm's plateau from its last research plan,
        as the scheduler does (else the first evaluation after a plan says stop)."""
        plans = [r["exp"] for r in self.state["research"] if r["arm"] == arm.id and r.get("plan")]
        if plans:
            done = sum((r["exp"] or 0) <= max(plans) for r in arm.rows)
            self.orch.budget.restarted[arm.agent] = done
        else:
            self.orch.budget.restarted.pop(arm.agent, None)

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
        then does the arm's count of evaluations without a new best restart."""
        n = len(self.state["slices"])
        plan = research.plan_path(self.run, arm.id)
        before = plan.read_bytes() if plan.is_file() else None
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
            log(f"research: {arm.id}: agent session failed: {exc!r}")
            rec["error"] = repr(exc)[:500]
            status = "failed"
        except BaseException:  # Ctrl-C, cancellation
            self._close_research(rec, "interrupted", plan=False)
            raise
        wrote = plan.is_file() and plan.read_bytes() != before
        self._close_research(rec, status, plan=wrote, usd=usd)
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
        exit cleanly (an interrupted research session counts as one without a plan)."""
        for res in self.state["research"]:
            if res.get("status") == "running":
                self._close_research(res, "interrupted", plan=False, ended=res["started"])
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
        context = rounds_context(self.run, self.state, n, accepted, arms)
        new = await self.orch.replan(round_dir, context, label=f"planner#round{n}")
        captured = self.orch._capture(new) if new else []
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
        "| arm | slices | evaluations | slices with a new best | best |",
        "|---|---|---|---|---|",
    ]
    for arm in dict.fromkeys(s["arm"] for s in slices):
        mine = [s for s in slices if s["arm"] == arm]
        best = max(float(s.get("best_after") or s.get("best_before") or 1.0) for s in mine)
        lines.append(
            f"| `{arm}` | {len(mine)} | {sum(s.get('evals') or 0 for s in mine)} | "
            f"{sum(bool(s.get('improved')) for s in mine)} | {best:.3f}x |"
        )
    integrations = state.get("integrations", [])
    if integrations:
        lines += ["", "Re-integrations: " + ", ".join(f"{i['speedup']:.3f}x" for i in integrations)]
    sessions = state.get("research") or []
    if sessions:
        lines += ["", "Research sessions on plateaued targets (`targets/<id>/plan.md`):", ""]
        for r in sessions:
            outcome = "wrote a plan" if r.get("plan") else f"no plan ({r.get('status')})"
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
    except KeyboardInterrupt:
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

    ``cfg.max_hours`` / ``cfg.max_usd`` are the budget of this invocation: hours
    from now, and USD on top of what the run has already spent.
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
            **{
                k: getattr(cfg, k)
                for k in ("agent_minutes", "program", "budget_usd_per_agent")
                if getattr(cfg, k) is not None
            },
            **{k: False for k in ("use_library", "librarian") if not getattr(cfg, k)},
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
