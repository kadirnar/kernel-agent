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
            target with islands (--islands, workers.py, issue #189) gives the slice to
            the island with the best island UCB (its own lineage and NOTES.md; the
            best results of the other islands as inspirations every --migrate-every
            evaluations; reseeded from the target's best once it falls --cull-gap behind)
        the native arm (--native, native/engine.py, issue #134), once every module arm
            has plateaued: first the capture of its current stage (teacher-forced
            checks), then a longer systems-native session with --native-evaluations;
            once every stage of its plan beat the bar it keeps going (issue #164): a
            re-profile of its best run says which stage has the most time left
        every --integrate-every kept results: measured re-integration
    every arm retired for this round (issue #166: the move-on rules retire an arm for
        a round, not for good): while --rounds allows (default: as many as the budget
        allows) and this round brought a real end-to-end gain or a budget bounds the
        run, re-profile the optimised model, re-plan with the prior rounds as context
        (rounds/<n>/) and continue with the new targets and the arms the round's
        profile shows still matter
    final integration + report (with the budget left unused and why)

With ``--agents N`` (N > 1) :mod:`kernel_agent.coordinator` runs this loop with up to N
sessions at once (slices, research sessions and dossiers of different arms; the
re-integration in the background; a new round only once nothing runs); ``--agents 1`` is
the sequential loop above.

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

from kernel_agent import (
    board,
    demotion,
    governor,
    interrupt,
    ledger,
    objective,
    pivot,
    projection,
    research,
    workers,
)
from kernel_agent.budget import improves
from kernel_agent.config import OptimizeConfig
from kernel_agent.dashboard import refresh
from kernel_agent.integrate.owners import context_line
from kernel_agent.libscout import scout as libscout
from kernel_agent.native import engine as native_engine
from kernel_agent.profiling import ceilings
from kernel_agent.research import rows_table
from kernel_agent.scheduler import (
    INTEGRATION_SHARE,
    KERNEL,
    NATIVE,
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
from kernel_agent.workspace import RunDir, coordinator_lock, read_json, write_json

if TYPE_CHECKING:
    from kernel_agent.agent.runner import AgentResult
    from kernel_agent.orchestrator import Orchestrator

STATE = "improve.json"
LAST_ROWS = 15  # ledger rows of the arm in a slice digest
NOTES_CHARS = 4000  # tail of NOTES.md in a slice digest
IDEAS_CHARS = 2000
MAX_FAILED_SLICES = 3  # agent sessions in a row that raised: something is broken, stop
# a re-profile of the native arm's best run once its plan is done (issue #164): VoxCPM2's
# round-2 re-profile took 75 s; the margin covers a model that loads slower
NATIVE_REPROFILE_SECONDS = 300.0
# a new round (issue #166) needs a re-profile (~5 min), a re-plan (~5 min) and one slice
# (warm-up, one kernel evaluation, wrap-up: 7 min); with less agent time left it is not started
NEW_ROUND_SECONDS = 1020.0
_IDEAS = re.compile(
    r"^#+[ \t]+[^\n]*\bideas?\b[^\n]*\n(.*?)(?=^#{1,6}[ \t]|\Z)", re.M | re.S | re.I
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] improve: {msg}", flush=True)


@dataclass
class ImproveConfig:
    slice: int = 4  # evaluations per slice (one fresh agent session)
    # re-profile + re-plan rounds: None = as many as the budget allows (issue #166; without
    # --max-hours / --max-usd / --max-sessions: while each round brings a real gain), 1 = none
    rounds: int | None = None
    integrate_every: int = 4  # kept results between measured re-integrations (0: final only)
    max_slices: int | None = None  # slices in this invocation (None: until budget / plateau)
    research_every: int = 3  # slices of a target between its research sessions (0: none)
    policy: Policy = field(default_factory=Policy)
    # minutes the time budget keeps for the final integration (None: its estimate, at most
    # INTEGRATION_SHARE of --max-hours; --integration-reserve)
    integration_reserve: float | None = None
    # agent sessions at once (--agents, coordinator.py, issue #183; 1: the sequential loop)
    agents: int = 1
    # --agents auto (governor.py, issue #191): up to ``agents`` sessions, as many as make the
    # run fastest (the GPU's knee; the usage windows only so they do not run out; the host's
    # memory; USD only with --max-usd)
    governor: bool = False
    # --async-evals (issue #191): submit_evaluation / evaluation_result beside
    # evaluate_candidate, so a session writes its next candidate while one is evaluated
    async_evals: bool = False
    # sessions of a role at once (--role-max) over the role registry's max_concurrent
    # (roles.REGISTRY: kernel 4, systems 1, native 1; scheduler.role_caps)
    role_max: dict[str, int] = field(default_factory=dict)
    # sessions whose modules overlap (--overlap): allow, warn (a digest note) or avoid
    overlap: str = "warn"
    # a role's first session starts alone until it streams (its prompt cache), then the rest
    stagger: bool = True
    # the blackboard (--board, board.py, issue #187): auto = with --agents N > 1, on, off
    board: str = "auto"
    # the critic of every full evaluation (--critic, critic.py, issue #188): static checks
    # (static), plus a cheap model's triage while a job waits --critic-wait s (model), off
    critic: str = "static"
    critic_wait: float = 30.0  # critic.MIN_WAIT_S
    # islands (--islands, workers.py, issue #189): the target's evaluations between two offers
    # of the other islands' best results to an island (0: no migration), and how far below
    # the target's best a stagnant island is reseeded from it (0: never)
    migrate_every: int = workers.MIGRATE_EVERY
    cull_gap: float = workers.CULL_GAP
    # how the agents' commands reach the GPU (--agent-gpu, #185): "tool" (their Bash sees no
    # GPU: run_on_gpu) or "bash"; None: tool with agents > 1, else bash
    agent_gpu: str | None = None
    # physical cores the timed jobs get to themselves (--timing-cores, hygiene.py; None: 2
    # with agents > 1, 0: no CPU isolation)
    timing_cores: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def concurrent(self) -> bool:
        """The coordinator runs the loop (``--agents`` above 1, or ``auto``)."""
        return self.agents > 1 or self.governor


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
        "session sees only that file, the ledger and a digest like this one. There is always "
        "a next idea: a ceiling bounds a recipe, not the model (another precision, fusion "
        "across modules, another algorithm, a native rewrite).",
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
    run: RunDir,
    arm: Arm,
    n: int,
    evaluations: int,
    policy: Policy,
    worker: int | None = None,
    *,
    label: str = "",
    island: list[str] | None = None,
) -> str:
    """Context of a fresh kernel-engineer session (bounded: no growth with the slice count);
    ``worker``: of that worker's session (its own NOTES.md, the target's shared ledger);
    ``label``: the session's (its part of the run's board, :func:`board_section`);
    ``island``: an island session's ``## Your island`` (``workers.island_lines``) in place of
    ``## Best so far``."""
    lines = _header(n, evaluations, "`results.jsonl`, `NOTES.md` and `history/`")
    if island is not None:
        lines += island
    elif arm.best_snapshot:
        sol = "" if arm.sol is None else f", {arm.sol:.0%} of its recipe's roofline"
        lines += [
            "",
            "## Best so far",
            f"* `history/{arm.best_snapshot}`: {arm.best:.3f}x module speedup{sol}. Build on "
            f'it (`parent="history/{arm.best_snapshot}"`) unless you test a different approach.',
        ]
    else:
        lines += ["", "## Best so far", "* no correct candidate faster than the reference yet"]
    lines += libscout.bar_lines(run, arm.id)  # library kernels with no agent: a floor (#227)
    lines += refusals(run, arm.id)
    rows = arm.rows[-LAST_ROWS:]
    if rows:
        lines += ["", f"## Last {len(rows)} evaluations (oldest first; `keep` = new best)", ""]
        lines += rows_table(rows)
    if stats := research.target_ideas(run, arm.id)[-LAST_ROWS:]:
        lines += ["", "## Ideas so far (`idea_id`; buggy = never correct: retry, not refuted)", ""]
        lines += research.ideas_table(stats)
        if refuted := [f"`{s['idea']}`" for s in stats if s["verdict"] == "refuted"]:
            lines += [
                "",
                f"Refuted: {', '.join(refuted)} ({ledger.REFUTED_TRIES} or more correct tries, "
                "none within the noise of the best): stop variations of them and take the next "
                "open idea; a variant needs force=true past the critic",
            ]
    lines += research.plan_section(run, arm.id)
    lines += _notes(workers.notes_file(run, arm.id, worker), "NOTES.md")
    lines += [
        "",
        "## Where this target stands",
        # the scheduler's ms are the metric's (#114): per second of audio for throughput
        f"* It costs about {arm.remaining_ms:.1f} ms {_per(run)} now; the scheduler expects "
        f"{arm.headroom:.0%} of that can still go"
        + (f" ({arm.ceiling.describe()})." if arm.ceiling else "")
        + (f" ({arm.fusion.describe()}, `profile/fusions.md`)." if arm.fusion else "")
        + ("" if arm.ceiling or arm.fusion else "."),
        f"* {arm.streak} evaluations in a row without a new best; after {policy.patience} the "
        "target's time goes to the other arms for this round, so prefer a fundamentally "
        "different idea over small variations.",
    ]
    spec = read_json(run.target(arm.id) / "spec.json", {}) or {}
    lines += demotion.digest_lines(spec)  # a W4A4 target's precision mix (#233)
    lines += docs_section(run, spec)
    lines += board_section(run, arm, label)
    return "\n".join(lines + _footer("NOTES.md"))


def _per(run: RunDir) -> str:
    """What a ms of the run's metric is per: ``per model run`` (the latency), ``per second
    of generated audio`` (throughput), ``to first audio`` (ttfa)."""
    metric = objective.of(read_json(run.baseline_json, {}) or {})
    return "per model run" if metric.name == objective.LATENCY else metric.per


def docs_section(
    run: RunDir, spec: dict[str, Any], *, native: bool = False, pattern: str | None = None
) -> list[str]:
    """``## Docs to read first`` of a digest (``doclib/reading.py``, #186; [] before the doc
    library is built): the library's best sections for a target's ``spec`` (its precision,
    backends, module class, approach and why) on the run's GPU, found without a model;
    ``native``: in the native engine's libraries, ``pattern``: its stage's."""
    from kernel_agent.doclib import reading

    gpu = (read_json(run.toolchain_json, {}) or {}).get("gpu") or {}
    found = reading.queries(
        module_class=spec.get("module_class"),
        text=f"{spec.get('approach') or ''} {spec.get('why') or ''}",
        precision=spec.get("precision"),
        arch=gpu.get("arch"),
        pattern=pattern,
    )
    libs = reading.NATIVE_LIBRARIES if native else reading.libraries(spec.get("backends") or [])
    try:
        return reading.section(reading.first_reads(found, libs))
    except Exception as exc:  # documentation is a bonus: never fail a slice for it
        log(f"docs to read first: {exc!r}")
        return []


def board_section(run: RunDir, arm: Arm, label: str) -> list[str]:
    """``## Board`` of the digest of the session ``label`` of ``arm`` ([] without a board or
    a label; ``board.py``, issue #187): the newest entries for it (its subscription). It
    sets the session's cursor: what is posted after rides on its evaluation results."""
    found = board.active(run)
    if found is None or not label:
        return []
    return found.section(board.Reader.of(run, label, arm.kind, arm.id))


def systems_digest(
    run: RunDir, arm: Arm, n: int, evaluations: int, policy: Policy, *, label: str = ""
) -> str:
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
        "systems agent's time goes to the other arms for this round.",
    ]
    lines += board_section(run, arm, label)
    return "\n".join(lines + _footer("NOTES.md"))


def native_digest(
    run: RunDir,
    arm: Arm,
    n: int,
    evaluations: int,
    policy: Policy,
    status: native_engine.Status,
    arms: list[Arm],
    *,
    label: str = "",
) -> str:
    """Context of a fresh systems-native session: the module-level bar, the staged plan
    with each stage's state, why the module arms stopped, the last native evaluations; the
    board's newest entries for the session ``label`` (the live winners, :func:`board_section`)."""
    lines = _header(
        n, evaluations, "the transforms' `results.jsonl`, your `NOTES.md` and the ledger"
    )
    lines += [
        "",
        "## Where the native engine stands",
        f"* Bar: the best module-level result end to end is {status.bar:.3f}x the baseline "
        "(integrations, transforms); a native run counts once it beats it.",
    ]
    if arm.best_snapshot:
        lines.append(f"* Best native run: `{arm.best_snapshot}`, {arm.best:.3f}x the bar.")
    else:
        lines.append("* No native run has beaten the bar yet.")
    lines += ["", "## Staged plan (a stage must beat the bar end to end before the next)"]
    for i, stage in enumerate(status.plan, 1):
        state = "later"
        if stage.id in status.finished:
            state = f"done, {status.finished[stage.id]:.3f}x"
        elif status.stage is not None and stage.id == status.stage.id:
            state = "**current**"
        target = stage.target_id
        teacher = (
            f"; teacher-forced target `{target}` (evaluate_candidate)"
            if target and run.capture_file(target).exists()
            else ""
        )
        if target and not teacher and (refused := native_engine.capture_refusal(run, target)):
            teacher = f"; no teacher-forced target (capture refused: {refused}): end to end only"
        lines.append(f"{i}. [{state}] {stage.describe()}{teacher}")
    if not status.plan:
        lines.append("* (no stage graph: no ceilings table; derive the stages from the profile)")
    if status.complete:
        lines += _after_plan(run, status, policy)
    lines += native_engine.megakernel_hint(run, arm.rows)  # grid syncs, many launches (#225)
    module = [a for a in arms if a.kind == KERNEL]
    if module:
        lines += ["", "## Module arms (their kernels are your building blocks)"]
        lines += [
            f"* `{a.id}` ({a.module_class}): best {a.best:.2f}x"
            + (f", {a.stop}" if a.stop else ", plateaued")
            for a in module
        ]
    rows = arm.rows[-LAST_ROWS:]
    if rows:
        lines += ["", f"## Last {len(rows)} evaluations (oldest first; `keep` = new best)", ""]
        lines += rows_table(rows)
    lines += _notes(native_engine.native_dir(run) / "NOTES.md", "NOTES.md")
    lines += [
        "",
        f"* {arm.streak} native runs in a row without a new best; after "
        f"{policy.native_patience} the native arm's time goes to the other arms for this round.",
    ]
    if (current := status.stage) is not None:  # the docs of the stage it works on
        spec = {"module_class": current.module_class, "precision": current.precision}
        spec |= {"approach": current.idea, "why": current.why}
        lines += docs_section(run, spec, native=True, pattern=current.pattern)
    elif module:  # no stage graph: the docs of the modules it fuses
        spec = {"module_class": " ".join(a.module_class or "" for a in module)}
        lines += docs_section(run, spec, native=True)
    lines += board_section(run, arm, label)
    return "\n".join(lines + _footer("NOTES.md"))


def _after_plan(run: RunDir, status: native_engine.Status, policy: Policy) -> list[str]:
    """The native digest once every stage of the plan beat the bar (issue #164): what is slow
    now (the stage graph of the newest ceilings table), the focus, the stop rules."""
    stop = f"{policy.sol_stop:.0%}" if policy.sol_stop else "100%"
    lines = [
        "",
        "## After the staged plan",
        "* Every stage of the plan beat the bar once: a note, not a stop. Keep improving your "
        f"best run until {policy.native_patience} native runs in a row find no new best, "
        f"{policy.native_hours or 0:g} h in native slices this round, or every stage runs at "
        f"{stop} of its floor.",
    ]
    if not status.graph:
        lines.append(
            "* (no stage graph: no ceilings table; find the slowest part with your own profile)"
        )
        return lines
    lines.append(
        f"* What is slow now, by time above the floor (`{status.table}`, the stage graph "
        "derived again from the newest profile):"
    )
    for i, stage in enumerate(status.graph, 1):
        mark = "**focus**" if status.stage is not None and stage.id == status.stage.id else ""
        prefix = f"[{mark}] " if mark else ""
        target = stage.target_id
        check = (
            f"; teacher-forced target `{target}`"
            if target and run.capture_file(target).exists()
            else ""
        )
        lines.append(f"  {i}. {prefix}{stage.describe()}{check}")
    lines.append(
        "* Work on the focus (name its projects after it), or on a group of these stages "
        "fused across their boundaries (end to end) when that is where the time is."
    )
    return lines


def rounds_context(
    run: RunDir,
    state: dict[str, Any],
    n: int,
    accepted: list[dict[str, Any]],
    arms: list[Arm],
    *,
    quality: str = "exact",
    owners: dict[str, dict[str, list[str]]] | None = None,
    precisions: tuple[str, ...] | None = None,
) -> str:
    """Planner context for round ``n``: what earlier rounds did and which targets exist (and,
    in a near-lossless or relaxed run, how to move one of them to another of the ``precisions`` it
    allows, ``pivot.py``; None: the quality mode's default, no 4-bit); ``owners``: the
    modules the accepted items change (``integration.json``)."""
    from kernel_agent import precisions as allowed_precisions
    from kernel_agent.kernels.compare import allows_reduced

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
            mix = demotion.label(spec)  # a W4A4 target's layers at 8 bits (#233)
            tier = f" [{pivot.label(spec)}{mix}]" if spec.get("precision") else ""
            lines.append(
                f"* `{arm.id}` (`{arm.module_class}`){tier}: best {arm.best:.2f}x after "
                f"{arm.evals} evaluations; {arm.stop or 'still open'}"
            )
    kernels = [a.id for a in arms if a.kind == KERNEL]
    if bars := libscout.planner_lines(run, kernels):  # what libraries reach there (#227)
        lines += ["", "## Library bars (the library scout, no agent: floors to beat)", *bars]
    lines += [
        "",
        "Propose only NEW targets: module classes that are hot in this profile and not listed "
        "above. An empty `targets` list is a valid answer when nothing new is worth a kernel. "
        "New transform ideas are passed on to the systems agent.",
    ]
    allowed = allowed_precisions.default(quality) if precisions is None else precisions
    if allows_reduced(quality) and (reduced := allowed_precisions.reduced(allowed)):
        lines += [
            "",
            "## Precision pivots",
            "To move an existing target to another precision tier (its precision was fixed "
            'when it was planned), list it under `pivots` instead: `{"target": id, '
            '"precision": ..., "precision_why": ...}`, the `precision` one of '
            + ", ".join(f"`{p}`" for p in reduced)
            + " (the precisions this run allows), the `precision_why` with the numbers "
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
        # what the loop is busy with (``_doing``): the activities of each of its tasks
        self.activities: dict[object, list[str]] = {}
        self.no_round: str | None = None  # why the loop started no new round (_next_round)
        # concurrent sessions (--agents > 1, coordinator.py): slices are attributed by the
        # ledger rows of their own sessions (their labels, ``sessions``)
        self.concurrent = icfg.concurrent

    # -------------------------------------------------------- helpers

    def save(self) -> None:
        write_json(self.run.root / STATE, self.state)

    def _live(self, snapshot: dict[str, Any]) -> None:
        """``sessions`` (the agent sessions running now, their states and time split) and
        ``gpu`` (the GPU job queue) of ``improve.json`` (``sessions.Observer.attach``)."""
        self.state.update(snapshot)
        self.save()

    @property
    def round(self) -> int:
        return int(self.state["rounds"][-1]["n"])

    def targets(self) -> list[str]:
        ids = self.run.target_ids()
        if self.require_capture:
            ids = [t for t in ids if self.run.capture_file(t).exists()]
        return ids

    def native_why(self) -> str | None:
        """Why the run has a native arm (``--native``, the plan's ``native`` entry), or None."""
        return native_engine.enabled(self.orch.cfg.native, native_engine.plan_entry(self.run))

    def arms(self, rows: list[dict[str, Any]] | None = None, *, relax: bool = False) -> list[Arm]:
        """The arms (``relax``: the native arm held, not stopped, while kernel arms improve:
        concurrent sessions, ``scheduler.assign``)."""
        return build_arms(
            self.run,
            dataclasses.replace(self.policy, native=self.native_why() is not None),
            self.state["slices"],
            targets=self.targets(),
            rounds=self.state["rounds"],
            rows=rows,
            research=self.state["research"],
            relax_native=relax,
            islands=self._island_caps(),
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
        done = [  # of this round: a new round's profile may change what pays (issue #166)
            r
            for r in self.state["research"]
            if r["arm"] == arm.id and int(r.get("round") or 1) == self.round
        ]
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

    def _pickable(self, *, relax: bool = False) -> list[Arm]:
        """The arms, ranked; an arm stopped by its plateau waits for its research session."""
        arms = self.arms(relax=relax)
        for arm in arms:
            if arm.stop and self.research_due(arm):
                arm.stop = None
        return rank(arms, self.policy)

    def keeps_since_integration(self) -> int:
        """Kept results the last integration did not take: since its end, or since its start
        for one that ran in the background (sessions kept results meanwhile)."""
        done = self.state["integrations"]
        last = done[-1] if done else {}
        start = int(
            last["exp_before" if last.get("background") else "exp_after"]
            if done
            else self.state.get("start_exp", 0)
        )
        return sum(
            r["status"] == ledger.KEEP and r["backend"] != "integrate"
            for r in ledger.rows(self.run)
            if (r["exp"] or 0) > start
        )

    def _charts(self) -> None:
        refresh(self.run)
        slices_chart(self.run)

    @property
    def doing(self) -> str | None:
        """What the loop is busy with: the innermost activity of each of its tasks (one with
        one session; with concurrent sessions, each running one's), or None."""
        now = [stack[-1] for stack in self.activities.values() if stack]
        return "; ".join(now) or None

    @contextmanager
    def _doing(self, what: str) -> Iterator[None]:
        """``what`` the loop (this task of it) is busy with, for the ``interrupted`` record
        (kept when the body raises: that is what was interrupted)."""
        key: object = asyncio.current_task()
        stack = self.activities.setdefault(key, [])
        stack.append(what)
        yield
        stack.pop()
        if not stack:
            self.activities.pop(key, None)

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
        if hint := projection.recapture_hint(self.run):  # estimates of a capture before #119
            log(f"improve: {hint}")
        self.orch.phase = "improve"
        self.state["config"] = self.icfg.to_dict()
        self.state.pop("finished", None)
        self.state.pop("coordinator", None)  # of an earlier invocation with --agents N
        self.state.pop("governor", None)  # ... with --agents auto
        self.orch.async_evals = self.icfg.async_evals  # --async-evals (agent/tools.py)
        self.save()
        ledger.event(self.run, "phase_start", phase="improve")
        if board.enabled(self.icfg.board, self.icfg.agents):  # the blackboard (board.py, #187)
            board.open_board(self.run)
        self.orch.observer.attach(self._live)  # improve.json: sessions and GPU, live (#184)
        from kernel_agent import critic  # the critic of every full evaluation (#188)

        critic.open_critic(
            self.run,
            self.icfg.critic,
            agents=self.icfg.agents,
            wait_s=self.icfg.critic_wait,
            cfg=self.orch.cfg,
            env=self.orch.env,
            simulated=self.orch.simulated,
        )
        try:
            reason = await self._loop()
            log(f"stopping: {reason}")
            await self._finish(reason)
        except BaseException as exc:  # Ctrl-C too: the state and the event log stay consistent
            if isinstance(exc, KeyboardInterrupt | asyncio.CancelledError):
                self._interrupted()
            ledger.event(self.run, "phase_failed", phase="improve", error=repr(exc)[:300])
            raise
        finally:
            board.close(self.run)
            self.orch.observer.detach()
            critic.close(self.run)
            self.orch.async_evals = False
        ledger.event(self.run, "phase_done", phase="improve")
        return reason

    async def _loop(self) -> str:
        if self.concurrent:  # --agents N: up to N sessions at once (coordinator.py)
            from kernel_agent.coordinator import Coordinator

            return await Coordinator(self).run()
        done = failed = 0
        while True:
            interrupt.check()
            self._keep_time()
            if reason := self.orch.budget.exhausted():
                return reason
            if self.icfg.max_slices is not None and done >= self.icfg.max_slices:
                return f"--max-slices {self.icfg.max_slices} reached"
            await self.orch.seed_library(self.targets())  # library priors before any slice
            await self.orch.scout_libraries(self.targets())  # the library bar (#227)
            arms = self._pickable()
            arm = pick(arms)
            if arm is None:
                if await self._next_round(arms):
                    continue
                arms = self._pickable()  # a round that started with nothing to work on
                stops = "; ".join(f"{a.id}: {a.stop}" for a in arms)
                return f"every arm has stopped ({stops})" + (
                    f"; no new round: {self.no_round}" if self.no_round else ""
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

    def _budget_used(self) -> dict[str, Any]:
        """What the loop used of this invocation's budget when it stopped, and what it left
        unused (the report says why, issue #166): agent time beyond what is kept for the
        final integration, USD and agent sessions."""
        budget = self.orch.budget
        out: dict[str, Any] = {"hours": round(budget.elapsed_s() / 3600, 3)}
        if budget.max_hours is not None:
            left = max(budget.agent_seconds_left() or 0.0, 0.0)
            out.update(max_hours=budget.max_hours, unused_hours=round(left / 3600, 3))
        if budget.max_usd is not None:
            out.update(usd=round(budget.spent_usd(), 2), max_usd=round(budget.max_usd, 2))
        if budget.max_sessions is not None:
            out.update(sessions=budget.sessions, max_sessions=budget.max_sessions)
        return out

    async def _finish(self, reason: str) -> None:
        used = self._budget_used()  # before the final integration, which has its own reserve
        integrated = (self.run.root / "integration.json").exists()
        if self.keeps_since_integration() or not integrated:
            self._past_budget()
            await self.reintegrate("final integration")
        self.state["finished"] = {"reason": reason, "at": _ts(), "budget": used}
        self.save()
        self.orch._mark("improve", reason=reason, slices=len(self.state["slices"]))
        slices_chart(self.run)
        await self.orch.report()  # report.md, charts and dashboard

    # -------------------------------------------------------- slices

    async def _slice(self, arm: Arm, arms: list[Arm]) -> dict[str, Any]:
        rec = self._open_slice(arm, arms)
        return await self._run_slice(rec, arm, arms)

    def _open_slice(self, arm: Arm, arms: list[Arm]) -> dict[str, Any]:
        """The record of a new slice of ``arm`` (``improve.json``, ``slice_start``); with
        concurrent sessions it names its sessions (``sessions``: their labels, whose ledger
        rows are its own)."""
        n = len(self.state["slices"]) + 1
        island = self._island(arm)  # with --islands: the island it goes to (workers.py)
        agent = island.agent if island else arm.agent
        label = f"{agent}#{n}"
        info = arm.summary()
        rec: dict[str, Any] = {
            "n": n,
            "arm": arm.id,
            "round": self.round,
            "agent": agent,
            "label": label,
            "status": "running",
            "started": _ts(),
            "exp_before": len(ledger.rows(self.run)),
            "best_before": arm.best,
            **{k: info[k] for k in ("remaining_ms", "headroom", "expected_ms", "index", "score")},
            "why": info["why"],  # the score's components (scheduler.Arm.why, issue #122)
            **self._session_time(arm),
            **({"sessions": [label]} if self.concurrent else {}),
            **({"island": island.k, "generation": island.generation} if island else {}),
        }
        self.state["slices"].append(rec)
        self.save()
        others = ", ".join(f"{a.id} {a.score:.3g}" for a in arms if a is not arm and not a.stop)
        on = f" island {island.k} (island score {island.score:.3g})" if island else ""
        log(
            f"slice {n}: {arm.id}{on} (best {arm.best:.2f}x, expected gain "
            f"{arm.expected_ms:.3g} ms, score {arm.score:.3g}: {info['why']}; others: "
            f"{others or 'none'})"
        )
        ledger.event(
            self.run,
            "slice_start",
            slice=n,
            arm=arm.id,
            score=info["score"],
            expected_ms=info["expected_ms"],
            why=info["why"],
            **({"island": island.k} if island else {}),
        )
        return rec

    async def _run_slice(
        self, rec: dict[str, Any], arm: Arm, arms: list[Arm], beside: str = ""
    ) -> dict[str, Any]:
        """The sessions of the slice ``rec`` opened (:meth:`_open_slice`), then its record
        closed; ``beside``: appended to its digest (the sessions running beside it)."""
        n, label = rec["n"], rec["label"]
        evaluations = self.icfg.slice
        try:
            if arm.kind == KERNEL:  # a W4A4 target's mix moves on a failed gate (#233)
                await self._w4a4_mix(arm.id)
            if arm.kind == NATIVE:
                evaluations = self.orch.cfg.native_evaluations or evaluations
                stand = await self._native_stage(arm)
                if note := stand.note():
                    rec["note"] = note
                    log(f"slice {n}: native: {note}")
                digest = native_digest(
                    self.run, arm, n, evaluations, self.policy, stand, arms, label=label
                )
                digest += beside
                results = [
                    await self.orch.native_slice(
                        evaluations=evaluations, digest=digest, label=label
                    )
                ]
            elif arm.kind == SYSTEMS:
                digest = systems_digest(self.run, arm, n, evaluations, self.policy, label=label)
                digest += beside
                results = [
                    await self.orch.systems_slice(
                        evaluations=evaluations, digest=digest, label=label
                    )
                ]
            elif rec.get("island") is not None:  # one island's session (--islands, #189)
                results = [await self._island_session(rec, arm, evaluations, beside)]
            else:
                self._restart_advice(arm)
                digest = kernel_digest(self.run, arm, n, evaluations, self.policy, label=label)
                digest += beside
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

    async def _native_stage(self, arm: Arm) -> native_engine.Status:
        """The native arm's staged plan and current stage; a stage with a module group is
        captured as its kernel target first (once), so the native session can check it
        teacher forced (``evaluate_candidate``) before the end-to-end run. Once the plan is
        done, the stage comes from a re-profile of the arm's best run
        (:meth:`_native_reprofile`)."""
        columns = ceilings.columns(self.orch.allowed_precisions())
        status = native_engine.status(self.run, ledger.rows(self.run), columns)
        if status.complete and await self._native_reprofile(arm):
            status = native_engine.status(self.run, ledger.rows(self.run), columns)
        stage = status.stage
        if stage is None or stage.target_id is None:
            return status
        target_dir = self.run.target(stage.target_id)
        if (
            self.run.capture_file(stage.target_id).exists()
            or (target_dir / "spec.failed.json").exists()
        ):
            return status
        specs = [
            read_json(self.run.target(t) / "spec.json", {}) or {} for t in self.run.target_ids()
        ]
        backends = [b for b in self.orch._available_backends() if b == "cuda"]
        spec = native_engine.target_spec(
            stage, native_engine.module_precision(specs), backends or ["cuda"]
        )
        if spec is not None:
            log(f"native: capturing stage {stage.id} as target {spec['id']}")
            with self._doing(f"capture of native stage {stage.id}"):
                await self.orch.capture_targets([spec])
        return status

    async def _native_reprofile(self, arm: Arm) -> bool:
        """Once every stage of the native plan beat the bar (issue #164): profile the model as
        the arm's best run has it (its transforms and kernels, ``Orchestrator.e2e_items``)
        into ``rounds/<n>/native/<k>/`` (``native.engine.reprofile_dir``), whose ceilings
        table the stage graph is derived from again: the stage times moved. Once per best
        run (``native_profiles`` in ``improve.json``, failures too); whether it wrote one."""
        done = self.state.setdefault("native_profiles", [])
        snapshot = arm.best_snapshot
        if not snapshot or any(p.get("snapshot") == snapshot for p in done):
            return False
        left = self.orch.budget.agent_seconds_left()
        if left is not None and left < NATIVE_REPROFILE_SECONDS + slice_seconds(arm):
            return False  # no time for it and a slice: the newest table stands
        k = len(done) + 1
        out = native_engine.reprofile_dir(self.run, self.round, k)
        rec: dict[str, Any] = {
            "n": k,
            "round": self.round,
            "snapshot": snapshot,
            "dir": str(out.relative_to(self.run.root)),
            "started": _ts(),
        }
        items = self.orch.e2e_items(snapshot)
        if not items:
            rec["error"] = "its transforms or kernels are not verified snapshots"
        else:
            rec["items"] = [i["item"] for i in items]
            log(f"native: the staged plan is done; re-profiling its best run {snapshot}")
            ledger.event(self.run, "native_reprofile", n=k, snapshot=snapshot, items=rec["items"])
            with self._doing(f"re-profile of the native best run ({snapshot})"):
                info = await asyncio.to_thread(self.orch.reprofile, out, items)
            if "error" in info or not info.get("median_ms"):
                rec["error"] = str(info.get("error") or "no median_ms")[-500:]
            else:
                rec["median_ms"] = float(info["median_ms"])
        rec["ended"] = _ts()
        done.append(rec)
        self.save()
        if "error" in rec:
            log(f"native: no re-profile of {snapshot}: {rec['error'][-300:]}")
            return False
        log(f"native: re-profiled {snapshot}: {rec['median_ms']:.1f} ms ({rec['dir']})")
        return True

    # -------------------------------------------------------- islands (workers.py, #189)

    def _island_caps(self) -> dict[str, int] | None:
        """The live islands of every kernel target with several (``--islands``): the most
        sessions its arm runs at once (None without the setting: one per arm)."""
        if self.orch.cfg.seeds_per_target is None:
            return None
        counts = {t: self.orch.island_count(t) for t in self.targets()}
        return {t: k for t, k in counts.items() if k > 1}

    def islands(self, arm: Arm) -> list[workers.Island]:
        """The islands of the kernel arm ``arm`` (``improve.json`` → ``islands``, new ones
        saved), measured from the ledger and ranked by the island UCB (``workers.rank``);
        [] with one island (its classic session)."""
        if arm.kind != KERNEL or (k := self.orch.island_count(arm.id)) < 2:
            return []
        saved = self.state.setdefault("islands", {}).setdefault(arm.id, {})
        spec = read_json(self.run.target(arm.id) / "spec.json", {}) or {}
        elite = (arm.best_snapshot, arm.best) if arm.best_snapshot else None
        available = self.orch._available_backends()
        found = workers.new_islands(arm.id, k, spec, available, saved, elite)
        if new := [i for i in found if str(i.k) not in saved]:
            saved.update({str(i.k): i.state() for i in new})
            self.save()
        rows = [r for r in ledger.rows(self.run) if r["target"] == arm.id]
        busy = {
            int(s["island"])
            for s in self.state["slices"]
            if s["arm"] == arm.id and s.get("status") == "running" and s.get("island")
        }
        for island in found:
            workers.measure(island, rows, island.k in busy)
        return workers.rank(found, self.policy.decay, self.policy.explore)

    def _island(self, arm: Arm) -> workers.Island | None:
        """The island the next slice of ``arm`` goes to (None: its classic session): the
        stagnant ones reseeded first (:meth:`_cull`), then the best by the island UCB without
        a running session (``scheduler.assign`` keeps one free: the arm's ``max_sessions``)."""
        islands = self.islands(arm)
        if islands and self._cull(arm, islands):
            islands = self.islands(arm)
        return workers.choose(islands) or (islands[0] if islands else None)

    def _cull(self, arm: Arm, islands: list[workers.Island]) -> bool:
        """Reseed every island of ``arm`` that fell ``--cull-gap`` behind the target's best
        (``workers.cull_reason``) from it, in the target's next unused direction, its
        ``NOTES.md`` archived (``improve.json`` → ``culls``); whether one was."""
        if not arm.best_snapshot:
            return False
        spec = read_json(self.run.target(arm.id) / "spec.json", {}) or {}
        rows = ledger.rows(self.run)
        evaluations = len(ledger.measured(r for r in rows if r["target"] == arm.id))
        elite = (arm.best_snapshot, arm.best)
        culled = False
        for island in islands:
            if (why := workers.cull_reason(island, arm.best, self.icfg.cull_gap)) is None:
                continue
            done = sum(c["target"] == arm.id for c in self.state.get("culls", []))
            new = workers.reseed(
                island,
                spec,
                self.orch._available_backends(),
                len(islands) + done + 1,  # the directions so far: one per island and reseed
                elite,
                len(rows),
                evaluations,
            )
            archived = workers.archive_notes(self.run, arm.id, island)
            self.state["islands"][arm.id][str(island.k)] = new.state()
            self.state.setdefault("culls", []).append(
                {
                    "target": arm.id,
                    "island": island.k,
                    "generation": new.generation,
                    "at": _ts(),
                    "exp": len(rows),
                    "why": why,
                    "best": round(island.best, 4),
                    "target_best": round(arm.best, 4),
                    "parent": new.parent,
                    "approach": new.approach,
                    **({"archived": str(archived.relative_to(self.run.root))} if archived else {}),
                }
            )
            log(f"{arm.id}: {why}: reseeded from {new.parent} in a new direction")
            ledger.event(
                self.run,
                "island_reseeded",
                target=arm.id,
                island=island.k,
                generation=new.generation,
                parent=new.parent,
            )
            culled = True
        if culled:
            self.save()
        return culled

    def _migrate(
        self, rec: dict[str, Any], arm: Arm, island: workers.Island
    ) -> list[dict[str, Any]]:
        """The inspirations of the session of slice ``rec`` on ``island`` when a migration is
        due (``--migrate-every``, ``workers.inspirations``): recorded in ``improve.json`` →
        ``migrations`` and the slice's ``inspirations`` ([]: none due or none to offer)."""
        rows = [r for r in ledger.rows(self.run) if r["target"] == arm.id]
        evaluations = len(ledger.measured(rows))
        if not workers.migration_due(island, evaluations, self.icfg.migrate_every):
            return []
        offered = workers.inspirations(island, rows)
        if not offered:
            return []
        island.inspired = evaluations
        self.state["islands"][arm.id][str(island.k)] = island.state()
        self.state.setdefault("migrations", []).append(
            {
                "target": arm.id,
                "island": island.k,
                "slice": rec["n"],
                "label": rec["label"],
                "at": _ts(),
                "evaluations": evaluations,
                "inspirations": [
                    {k: e[k] for k in ("snapshot", "speedup", "backend", "island")} for e in offered
                ],
            }
        )
        rec["inspirations"] = [e["snapshot"] for e in offered]
        self.save()
        log(
            f"slice {rec['n']}: {arm.id} island {island.k}: inspirations "
            + ", ".join(f"{e['snapshot']} ({e['speedup']:.3f}x, {e['backend']})" for e in offered)
        )
        return offered

    async def _island_session(
        self, rec: dict[str, Any], arm: Arm, evaluations: int, beside: str = ""
    ) -> AgentResult:
        """The session of slice ``rec`` on its island: the island's own directory, notes and
        lineage, the other islands' best results when a migration is due (:meth:`_migrate`)
        and a whole slice of evaluations (no split: the island UCB and the free slots spread
        the target's evaluations over its islands)."""
        islands = self.islands(arm)
        island = next(i for i in islands if i.k == int(rec["island"]))
        offered = self._migrate(rec, arm, island)
        self._restart_advice(arm, [island.agent])
        lines = workers.island_lines(island, islands, arm.best, arm.best_snapshot, offered)
        digest = kernel_digest(
            self.run,
            arm,
            rec["n"],
            evaluations,
            self.policy,
            island.k,
            label=rec["label"],
            island=lines,
        )
        team = sorted(islands, key=lambda i: i.k)
        return await self.orch.worker_session(
            arm.id,
            island.seed(evaluations),
            [i.seed(evaluations) for i in team],
            prompt=f"Continue optimising target `{arm.id}` as island {island.k} (see the "
            "`# Island` section). Read the `# Improve slice` section first: it says where the "
            "previous sessions left off.",
            digest=digest + beside,
            label=rec["label"],
            note=workers.island_note(arm.id, island, team, evaluations),
        )

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

    def _own(self, rec: dict[str, Any]) -> tuple[Arm | None, list[dict[str, Any]], bool]:
        """(the arm of slice ``rec``, its ledger rows, whether it found a new best). With
        concurrent sessions (``sessions``) the rows its own sessions made and whether they
        raised the arm's best beyond what the rows of everyone else reach (a re-integration
        and the other sessions run meanwhile); else the arm's rows since the slice started
        and whether its best rose."""
        if "sessions" not in rec:
            arm = next((a for a in self.arms() if a.id == rec["arm"]), None)
            new = [r for r in (arm.rows if arm else []) if (r["exp"] or 0) > rec["exp_before"]]
            return arm, new, (arm.best if arm else rec["best_before"]) > rec["best_before"]
        mine = set(rec["sessions"])
        rows = ledger.rows(self.run)
        arm = next((a for a in self.arms(rows) if a.id == rec["arm"]), None)
        new = [r for r in (arm.rows if arm else []) if r.get("session") in mine]
        if not arm or not new:
            return arm, new, False
        others = [r for r in rows if r.get("session") not in mine]
        rest = next((a for a in self.arms(others) if a.id == rec["arm"]), None)
        return arm, new, arm.best > (rest.best if rest else rec["best_before"])

    def _close(
        self,
        rec: dict[str, Any],
        status: str,
        *,
        usd: float | None = None,
        ended: float | None = None,
    ) -> None:
        arm, new, improved = self._own(rec)
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
            improved=improved,
        )
        if "sessions" in rec:  # the time its evaluations waited for the GPU (gpuqueue.py)
            rec["queue_s"] = round(sum(float(r.get("queue_s") or 0.0) for r in new), 1)
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

    async def _research(
        self, arm: Arm, why: str, rec: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """A research session for a plateaued arm (``Orchestrator.research``) and its record
        (``rec``: opened already, :meth:`_open_research`).

        ``plan`` in the record: whether the session wrote a new ``plan.md``; only
        then does the arm's count of evaluations without a new best restart. ``pivot``:
        what became of a precision pivot it proposed (``pivot.json``, ``pivot.py``): the
        new target, or why it was refused."""
        plan = research.plan_path(self.run, arm.id)
        before = plan.read_bytes() if plan.is_file() else None
        proposal = pivot.proposal_path(self.run, arm.id)
        proposed = proposal.read_bytes() if proposal.is_file() else None
        rec = self._open_research(arm, why) if rec is None else rec
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

    async def _w4a4_mix(self, target_id: str) -> dict[str, Any]:
        """The per-layer demotion policy of a W4A4 target (``demotion.py``) on its records and
        the last integration: a failed gate moves its next group of layers to 8 bits (the
        slice's engineer reads the mix in ``spec.json``); with every group there, the pivot
        to its 8-bit class it proposes is made (``Orchestrator.pivot``). What it did, or {}."""
        from kernel_agent.truth import TamperError

        spec = read_json(self.run.target(target_id) / "spec.json", {}) or {}
        if not demotion.applies(spec):
            return {}
        try:
            records = self.orch.truth.records(self.run.results_file(target_id))
        except TamperError:
            records = []
        integration = self.orch.truth.load_json(self.run.root / "integration.json") or {}
        try:
            done = demotion.step(self.run, target_id, records=records, integration=integration)
        except Exception as exc:  # bookkeeping: never a reason to fail the arm's slice
            log(f"w4a4 mix: {target_id}: not updated: {exc!r}")
            return {}
        if done:
            mix = done["mix"]
            log(
                f"w4a4 mix: {target_id}: {mix['demoted']} of {mix['groups']} groups in "
                f"{mix['eight_bit']} ({done['trigger']}: {str(done['reason'])[:200]})"
            )
        if proposal := done.get("pivot"):
            done["pivot_result"] = await self.orch.pivot(
                target_id, proposal, source=demotion.SOURCE
            )
            demotion.record_pivot(self.run, target_id, done["pivot_result"])
        return done

    def _open_research(self, arm: Arm, why: str) -> dict[str, Any]:
        """The record of a new research session of ``arm`` (``improve.json``)."""
        n = len(self.state["slices"])
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
        return rec

    async def _dossier(self, arm: Arm, rec: dict[str, Any] | None = None) -> dict[str, Any]:
        """The dossier session of a kernel arm before its first slice and its record
        (``improve.json`` → ``dossiers``; ``file``: it wrote ``research.md``; ``rec``: opened
        already, :meth:`_open_dossier`)."""
        rec = self._open_dossier(arm) if rec is None else rec
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

    def _open_dossier(self, arm: Arm) -> dict[str, Any]:
        """The record of the dossier session of ``arm`` (``improve.json`` → ``dossiers``)."""
        rec: dict[str, Any] = {
            "arm": arm.id,
            "label": f"dossier-{arm.id}",
            "status": "running",
            "started": _ts(),
        }
        self.state.setdefault("dossiers", []).append(rec)
        self.save()
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
            _, new, _ = self._own(rec)  # its rows (of its own sessions, with several at once)
            times = [t for r in new if (t := ledger.epoch(r["time"]))]
            log(f"slice {rec['n']} ({rec['arm']}) was interrupted; recording what it did")
            self._close(rec, "interrupted", ended=max([rec["started"], *times]))

    # -------------------------------------------------------- integration and rounds

    async def reintegrate(self, why: str, *, background: bool = False) -> dict[str, Any]:
        """Measured end-to-end integration (``Orchestrator.integrate``) and its record;
        ``background``: while agent sessions run (``coordinator.py``), whose kept results
        after its start it did not take (:meth:`keeps_since_integration`)."""
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
            **({"background": True} if background else {}),
        }
        done.append(rec)
        self.save()
        ledger.event(self.run, "integrated", speedup=speedup, median_ms=final.get("median_ms"))
        board.integration(self.run, rec)  # every session's baseline moved (#187)
        self._charts()
        return rec

    def _bounded(self) -> bool:
        """Whether a budget bounds this invocation (--max-hours, --max-usd, --max-sessions)."""
        budget = self.orch.budget
        return any(x is not None for x in (budget.max_hours, budget.max_usd, budget.max_sessions))

    async def _next_round(self, arms: list[Arm]) -> bool:
        """Every arm has retired for this round: start round n+1 (issue #166) while
        ``--rounds`` allows (default: no cap) and either this round paid off or a budget
        bounds the run (then the rest of it goes to a new round, not unused); else why not
        is ``no_round``."""
        self.no_round = None
        if self.icfg.rounds is not None and self.round >= self.icfg.rounds:
            if self.icfg.rounds > 1:
                self.no_round = f"--rounds {self.icfg.rounds} reached"
            return False
        left = self.orch.budget.agent_seconds_left()
        if left is not None and left < NEW_ROUND_SECONDS:
            self.no_round = (
                f"{max(left, 0) / 60:.0f} min left for agents, less than a new round needs "
                f"({NEW_ROUND_SECONDS / 60:.0f} min: re-profile, re-plan, one slice)"
            )
            return False
        if self.keeps_since_integration() or not self.state["integrations"]:
            await self.reintegrate(f"every arm of round {self.round} has stopped")
        last = self.state["integrations"][-1]
        start = float(self.state["rounds"][-1].get("speedup") or 1.0)
        measured = {"passed": True, "speedup": last["speedup"], "timing_spread": last.get("spread")}
        if not improves(measured, start, ok_key="passed"):
            if self.icfg.rounds is not None or not self._bounded():
                self.no_round = f"round {self.round} brought no real end-to-end gain"
                log(f"{self.no_round}; no further round")
                return False
            log(
                f"round {self.round} brought no real end-to-end gain; budget is left, so "
                f"round {self.round + 1} re-plans and revives what still matters"
            )
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
            self.no_round = f"the re-profile for round {n} failed"
            return False
        libscout.write_ceilings(self.run)  # the re-planner's ceilings.md: the library bar (#227)
        applied = {}
        for item in accepted:
            if item["kind"] == "kernel":
                target_id, _, path = item["item"].partition("=")
                rec = snapshot_record(self.run, target_id, path) or {}
                applied[target_id] = float(rec.get("speedup") or 1.0)
        quality = self.orch.cfg.quality
        owners = integrated.get("owners")
        context = rounds_context(
            self.run,
            self.state,
            n,
            accepted,
            arms,
            quality=quality,
            owners=owners,
            precisions=self.orch.allowed_precisions(),
        )
        with self._doing(f"re-plan for round {n}"):
            new = await self.orch.replan(round_dir, context, label=f"planner#round{n}")
            captured = await self.orch.capture_targets(new) if new else []
            captured += await self._pivots(round_dir, n)
        self.state["rounds"].append(
            {
                "n": n,
                "started": _ts(),
                "exp": len(ledger.rows(self.run)),  # the move-on rules count from here (#166)
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
        live = [a.id for a in self._pickable() if a.stop is None]
        log(
            f"round {n}: {info['median_ms']:.1f} ms; new targets: {captured or 'none'}; "
            f"live arms: {', '.join(live) or 'none'}"
        )
        board.round_started(self.run, n, float(info["median_ms"]), captured, live)  # (#187)
        if not live:
            self.no_round = f"round {n} has no new target and no arm that still matters"
        return bool(live)

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
        *_budget_lines(finished),
        *_concurrency_lines(state),
        *_island_lines(run, state),
        *board.report_lines(run),  # board.jsonl (#187)
        *_critic_lines(run),  # critic.jsonl (#188)
        "",
        "| arm | precision | slices | evaluations | slices with a new best | best |",
        "|---|---|---|---|---|---|",
    ]
    for arm in dict.fromkeys(s["arm"] for s in slices):
        mine = [s for s in slices if s["arm"] == arm]
        best = max(float(s.get("best_after") or s.get("best_before") or 1.0) for s in mine)
        spec = read_json(run.target(arm) / "spec.json", {}) or {}
        tier = pivot.label(spec) + demotion.label(spec) if arm not in (SYSTEMS, NATIVE) else "—"
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


def _budget_lines(finished: dict[str, Any]) -> list[str]:
    """The budget the loop used and left unused when it stopped, and why (issue #166)."""
    used = finished.get("budget") or {}
    if not used:
        return []
    parts = []
    if used.get("max_hours") is not None:
        hours = float(used.get("unused_hours") or 0.0)
        share = hours / float(used["max_hours"]) if used["max_hours"] else 0.0
        parts.append(
            f"{used['hours']:.2f} h of {used['max_hours']:g} h used, "
            f"{hours * 60:.0f} min ({share:.0%}) unused"
        )
    else:
        parts.append(f"{used['hours']:.2f} h (no --max-hours)")
    if used.get("max_usd") is not None:
        parts.append(f"${used['usd']:.2f} of ${used['max_usd']:.2f}")
    if used.get("max_sessions") is not None:
        parts.append(f"{used['sessions']} of {used['max_sessions']} agent sessions")
    left = float(used.get("unused_hours") or 0.0) * 60 >= 5.0  # more than a rounding error
    why = f"; unused because: {finished.get('reason')}" if left else ""
    return [f"* budget: {', '.join(parts)}{why}"]


def _critic_lines(run: RunDir) -> list[str]:
    from kernel_agent import critic  # the critic's verdicts and calibration (#188)

    return critic.report_lines(run)


def _concurrency_lines(state: dict[str, Any]) -> list[str]:
    """``--agents N`` (``coordinator.py``): the most sessions at once, the GPU waits of the
    slices' evaluations and the shared usage-limit waits ([] with one session at a time)."""
    done = state.get("coordinator")
    if not done:
        return []
    waited = sum(float(s.get("queue_s") or 0.0) for s in state["slices"])
    gov = done.get("governor")  # --agents auto (governor.py, #191)
    agents = f"auto (up to {done['agents']})" if gov else done["agents"]
    lines = [
        f"* --agents {agents}: at most {done['peak']} sessions at once; their "
        f"evaluations waited {waited / 60:.0f} min for the GPU in all; "
        f"{done['rate_limit_waits']} shared usage-limit wait(s)"
    ]
    if gov:
        lines.append(
            f"* governor: k between {gov['low']} and {gov['high']} (mean {gov['mean']:.1f}), "
            f"changed {gov['changes']} times (`governor` events)"
            + (f"; last: {line}" if (line := governor.status_line(state.get("governor"))) else "")
        )
    return lines


def _island_lines(run: RunDir, state: dict[str, Any]) -> list[str]:
    """``--islands`` (workers.py, issue #189): per target with islands, each island's best and
    backend, the migrations offered and used and the islands reseeded ([] without islands)."""
    saved = state.get("islands") or {}
    if not saved:
        return []
    rows = ledger.rows(run)
    out = []
    for target, islands in saved.items():
        mine = [r for r in rows if r["target"] == target]
        bests = []
        for k in sorted(islands, key=int):
            island = workers.measure(workers.Island.of(target, int(k), islands[k]), mine)
            bests.append(
                f"{k}: {island.best:.3f}x `{island.backend or '?'}` gen {island.generation}"
            )
        offered = sum(m["target"] == target for m in state.get("migrations") or [])
        culled = sum(c["target"] == target for c in state.get("culls") or [])
        out.append(
            f"* islands of `{target}` (`--islands`): {'; '.join(bests)}; {offered} migrations "
            f"offered, {workers.adoptions(mine)} times an island built on another's result, "
            f"{culled} reseeded"
        )
    return out


def slices_chart(run: RunDir) -> Path | None:
    """``improve.png``: one lane per arm, a bar per slice (sub-lanes for its sessions that
    ran at once), re-integrations, rounds and the GPU's busy strip."""
    from kernel_agent import charts, sessions

    state = read_json(run.root / STATE, None)
    if not isinstance(state, dict) or not state.get("slices") or not charts.available():
        return None
    start = ledger.start_time(run)
    if start is None:
        return None
    spans = sessions.spans(run)  # every session's start and end (sessions.jsonl, #184)
    holds = [(h.start, h.end, h.session) for h in sessions.gpu_holds(sessions.read_gpu(run))]
    return charts._render(
        run.root / "improve.png",
        (10.0, 4.6),
        lambda fig, ax: _draw_slices(ax, state, start, spans, holds),
    )


def _bars(
    slices: list[dict[str, Any]], spans: dict[str, tuple[float, float | None]]
) -> list[tuple[dict[str, Any], float, float, int, int]]:
    """(slice, start, end, sub-lane, sub-lanes of its arm) per bar: one per slice, one per
    session of a slice whose sessions are known (``sessions.jsonl``) and more than one (its
    workers). Bars of an arm that overlap (concurrent sessions) go to different sub-lanes."""
    bars: list[tuple[dict[str, Any], float, float]] = []
    for s in slices:
        known = [spans[label] for label in s.get("sessions") or [] if label in spans]
        if len(known) > 1:
            end = s.get("ended") or s["started"]
            bars += [(s, a, b if b is not None else end) for a, b in known]
        else:
            bars.append((s, s["started"], s.get("ended") or s["started"]))
    free: dict[str, list[float]] = {}  # arm -> the end of each of its sub-lanes so far
    placed = []
    for s, a, b in sorted(bars, key=lambda bar: bar[1]):
        ends = free.setdefault(s["arm"], [])
        k = next((i for i, e in enumerate(ends) if e <= a + 1e-6), len(ends))
        ends[k : k + 1] = [b]
        placed.append((s, a, b, k))
    return [(s, a, b, k, len(free[s["arm"]])) for s, a, b, k in placed]


def _draw_slices(
    ax: Any,
    state: dict[str, Any],
    start: float,
    spans: dict[str, tuple[float, float | None]] | None = None,
    holds: list[tuple[float, float | None, str | None]] | None = None,
) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from kernel_agent import charts

    slices = state["slices"]
    arms = list(dict.fromkeys(s["arm"] for s in slices))
    lane = {a: i for i, a in enumerate(arms)}
    holds = holds or []

    def minutes(ts: float | None) -> float:
        return max((ts or start) - start, 0.0) / 60

    end = max(
        [minutes(s.get("ended") or s.get("started")) for s in slices]
        + [minutes(i.get("at")) for i in state.get("integrations", [])],
        default=1.0,
    )
    rows = [*arms, "GPU"] if holds else arms  # the GPU's busy strip below the arms
    ax.set_xlim(0, max(end, 1.0) * 1.03)
    # room below the last lane for the integration labels
    ax.set_ylim(len(rows) + (0.45 if holds else 0.05), -0.6)
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(axis="y", visible=False)
    for s, t0, t1, k, n in _bars(slices, spans or {}):
        a, b = minutes(t0), minutes(t1)
        failed = s.get("status") in ("interrupted", "timed_out", "error", "failed")
        color = charts.KEEP_COLOR if s.get("improved") else charts.DISCARD_COLOR
        y = lane[s["arm"]] + (k - (n - 1) / 2) * 0.62 / n  # sub-lane k of n
        ax.barh(
            y,
            max(b - a, 0.3),
            left=a,
            height=0.56 / n,
            color=color,
            edgecolor=charts.FAIL_COLOR if failed else charts.SURFACE,
            linewidth=1.4 if failed else 0.8,
            hatch="///" if failed else None,
            zorder=3,
        )
        label = f"{s['best_after']:.2f}x" if s.get("improved") else ""
        if label and n == 1 and charts._fits(ax, label, 7.5, a, b, 2):
            ax.text(
                (a + b) / 2,
                y,
                label,
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
    busy = 0.0
    for a0, b0, session in holds:  # who held the GPU: the agents' jobs, background work
        a, b = minutes(a0), minutes(b0 if b0 is not None else a0)
        busy += b - a
        ax.barh(
            len(arms),
            max(b - a, 0.05),
            left=a,
            height=0.4,
            color=charts.TARGET_COLORS[2] if session else charts.MUTED,
            linewidth=0,
            zorder=3,
        )
    ax.set_xlabel("wall-clock time since the run started (min)")
    integrations = state.get("integrations", [])
    best = max([1.0, *(float(i["speedup"]) for i in integrations)])
    finished = (state.get("finished") or {}).get("reason", "running").split(" (")[0]
    first = min((minutes(a) for a, _, _ in holds), default=0.0)
    gpu = f" · GPU busy {busy / (end - first):.0%}" if holds and end > first else ""
    charts._header(
        ax,
        f"improve: {len(slices)} slices, measured end to end 1.00 → {best:.2f}x",
        charts._short(
            f"{sum(s.get('evals') or 0 for s in slices)} evaluations, "
            f"{len(integrations)} re-integrations, {len(state.get('rounds', []))} round(s) · "
            f"stopped: {finished}{gpu}",
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
            *(
                [
                    Patch(color=charts.TARGET_COLORS[2], label="GPU: agents' jobs"),
                    Patch(color=charts.MUTED, label="GPU: integration, captures"),
                ]
                if holds
                else []
            ),
        ],
        ncol=4 if holds else None,
    )


# ------------------------------------------------------------------ entry point


@contextmanager
def _interrupt_note(run: RunDir) -> Iterator[None]:
    try:
        yield
    except (KeyboardInterrupt, asyncio.CancelledError):
        log(f"interrupted; `kernel-agent improve {run.root}` continues this run")
        raise


@contextmanager
def _clean_timing(orch: Orchestrator, icfg: ImproveConfig, *, simulated: bool) -> Iterator[None]:
    """Clean timing under concurrent load (#185, ``hygiene.py``): with ``--agents`` above 1 (or
    ``--timing-cores N``) CPU isolation, builds off the GPU lock and dirty-timing re-runs (not
    in a dry run: nothing is timed); and how the agents reach the GPU (``--agent-gpu``, by
    default ``tool`` with several sessions: their Bash commands see no GPU, ``run_on_gpu``)."""
    from kernel_agent import hygiene
    from kernel_agent.agent.runner import GPU_BASH, GPU_TOOL

    mode = icfg.agent_gpu or (GPU_TOOL if icfg.agents > 1 else GPU_BASH)
    on = not simulated and (icfg.agents > 1 or bool(icfg.timing_cores))
    with hygiene.active(icfg.timing_cores, icfg.agents) if on else nullcontext() as clean:
        if clean is not None:
            plan = clean.plan.describe() if clean.plan else "no CPU isolation (--timing-cores 0)"
            log(f"clean timing: {plan}; builds before the GPU lock; dirty timings re-run")
        if mode != GPU_BASH or icfg.agent_gpu:
            log(f"agents' GPU access: {mode} (--agent-gpu)")
        with orch.gpu_access(mode):  # after hygiene.active: the agents' MAX_JOBS
            yield


def _role_overrides(cfg: OptimizeConfig) -> dict[str, Any]:
    """The per-role models and efforts a resumed run takes over its own: those that differ
    from the defaults (``--role-model`` / ``--role-effort``; ``roles.changed``)."""
    from kernel_agent import roles
    from kernel_agent.config import ROLE_EFFORTS, ROLE_MODELS

    out: dict[str, Any] = {}
    for key, defaults in (("role_models", ROLE_MODELS), ("role_efforts", ROLE_EFFORTS)):
        if given := roles.changed(getattr(cfg, key), defaults):
            out[key] = given
    return out


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
        with coordinator_lock(RunDir(path.resolve())):  # another process on it: refused now
            pass
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
                for k in ("use_library", "librarian", "allow_web", "dossier", "early_stop")
                if not getattr(cfg, k)
            },
            **({"web_domains": cfg.web_domains} if cfg.web_domains else {}),
            **({"seeds_per_target": cfg.seeds_per_target} if cfg.seeds_per_target else {}),
            **({"reseed_workers": True} if cfg.reseed_workers else {}),
            **({"parallel": cfg.parallel} if cfg.parallel > 1 else {}),
            **({"precisions": cfg.precisions} if cfg.precisions else {}),  # into run.json
            **({"native": cfg.native} if cfg.native != "plan" else {}),  # else the run's
            **({"native_minutes": cfg.native_minutes} if cfg.native_minutes else {}),
            **_role_overrides(cfg),  # --role-model / --role-effort over the run's (#181)
        }
        orch = Orchestrator.resume(path, overrides)
    elif dry_run:
        orch = Orchestrator(dryrun.create_run(cfg, seed), cfg)
    else:
        orch = Orchestrator.create(cfg)
    if cfg.max_usd is not None:
        orch.budget.max_usd = orch.budget.spent_usd() + cfg.max_usd
    log(f"run directory: {orch.run.root}")
    # concurrent simulated sessions run in virtual time (dryrun.VirtualClock)
    world = dryrun.World(orch, virtual=icfg.concurrent) if dry_run else None
    with (
        coordinator_lock(orch.run),  # one process per run: its truth is in our memory
        _interrupt_note(orch.run),
        world.installed() if world else nullcontext(),
        world.driving() if world and world.virtual else nullcontext(),
        _clean_timing(orch, icfg, simulated=dry_run),
    ):
        if not all(orch._phase_done(p) for p in ("analyze", "plan", "capture")):
            await orch.run_all(until="capture")
        await Improver(orch, icfg, require_capture=not dry_run, live_charts=not dry_run).improve()
    return orch.run
