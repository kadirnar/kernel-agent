"""Clean-context research sessions for kernel targets that have plateaued.

When a target plateaus (:func:`kernel_agent.scheduler.plateau`: the evaluation
advice turned ``consider_stopping``, ``--patience`` is reached, or three
evaluations in a row failed), ``kernel-agent improve`` runs a
``research-<target>`` session before the target is stopped
(:meth:`kernel_agent.orchestrator.Orchestrator.research`). The session starts
with no history of the engineer's reasoning, has read-only tools and may write
one file, ``targets/<id>/plan.md``: a diagnosis along a pathology checklist,
directions ranked by their ceiling, ideas to retry and a do-not-try list (the
research subagent of auto-gpu-kernel; K-Search's split of strategy from
implementation). The next engineer slice of the target is a fresh session whose
digest includes the plan, and the plan restarts the target's patience count.

This module builds what the research session is shown (:func:`evidence`) and the
ledger and idea tables it shares with the slice digests (:mod:`kernel_agent.improve`).

With the web tools on, a short ``dossier-<target>`` session runs before a target's first
engineer session (``Orchestrator.dossier``, issue #125): it looks up the documentation,
reference code and papers for the target and writes ``targets/<id>/research.md``
(findings with their sources, ideas); the research sessions may update it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kernel_agent import board, ledger, truth
from kernel_agent.agent.prompts import DOSSIER_FILE
from kernel_agent.agent.tools import best_for_target, idea_rows
from kernel_agent.truth import TamperError, Truth
from kernel_agent.workers import all_notes
from kernel_agent.workspace import RunDir

PLAN_FILE = "plan.md"
PLAN_CHARS = 6000  # of plan.md in a slice digest
LEDGER_ROWS = 40  # ledger rows of the target shown to the research session
TRANSFORM_ROWS = 5  # passing end-to-end transform evaluations shown (near-lossless runs)


def plan_path(run: RunDir, target_id: str) -> Path:
    return run.target(target_id) / PLAN_FILE


def dossier_path(run: RunDir, target_id: str) -> Path:
    """``targets/<id>/research.md``: the target's research dossier, written before its
    first engineer session and updated by its research sessions (issue #125)."""
    return run.target(target_id) / DOSSIER_FILE


def _cell(text: Any, limit: int = 200) -> str:
    return " ".join(str(text or "").split())[:limit].replace("|", "/")


def rows_table(rows: list[dict[str, Any]], *, sol: bool = False) -> list[str]:
    """Ledger rows as a Markdown table; ``idea`` / ``worker`` only when a row has one,
    ``sol``: % of SOL."""
    ideas = any(r.get("idea") for r in rows)
    team = any(r.get("worker") for r in rows)
    head = ["exp", "status", "speedup"] + (["% SOL"] if sol else []) + (["idea"] if ideas else [])
    head += (["worker"] if team else []) + ["backend", "snapshot", "hypothesis"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [str(r["exp"]), r["status"], "" if r["speedup"] is None else f"{r['speedup']:.3f}x"]
        if sol:
            pct = r.get("pct_of_sol")
            cells.append("" if pct is None else f"{pct:.0f}")
        if ideas:
            cells.append(_cell(r.get("idea"), 40))
        if team:
            cells.append(f"w{r['worker']}" if r.get("worker") else "")
        cells += [r["backend"], r["snapshot"], _cell(r["hypothesis"])]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def ideas_table(stats: list[dict[str, Any]]) -> list[str]:
    """:func:`kernel_agent.ledger.ideas` as a Markdown table."""
    lines = [
        "| idea | tries | best | expected | kept | slow | bugs | verdict | last hypothesis |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in stats:
        bugs = str(s["bugs"])
        failed = {k: v for k, v in s["statuses"].items() if k in ledger.FAILURES}
        if failed:
            bugs += " (" + ", ".join(f"{k} {v}" for k, v in failed.items()) + ")"
        verdict = {"buggy": "buggy: retry", "slow": "slow"}.get(s["verdict"], s["verdict"])
        lines.append(
            f"| `{s['idea']}` | {s['tries']} | "
            + ("" if s["best"] is None else f"{s['best']:.3f}x")
            + " | "
            + ("" if s["expected"] is None else f"{s['expected']:.2f}x")
            + f" | {s['kept']} | {s['slow']} | {bugs} | {verdict} | "
            f"{_cell(s['last_hypothesis'], 120)} |"
        )
    return lines


def target_ideas(run: RunDir, target_id: str, keeper: Truth | None = None) -> list[dict[str, Any]]:
    """:func:`kernel_agent.ledger.ideas` of a target from its records (they carry the
    expected speedups); from its ledger rows when the records cannot be verified."""
    try:
        records = (keeper or truth.of(run)).records(run.results_file(target_id))
    except TamperError:
        return ledger.ideas(r for r in ledger.rows(run) if r["target"] == target_id)
    return ledger.ideas(idea_rows(records))


def plan_section(run: RunDir, target_id: str) -> list[str]:
    """``## Research plan`` of a slice digest ([] when the target has no ``plan.md``)."""
    path = plan_path(run, target_id)
    text = path.read_text().strip() if path.is_file() else ""
    if not text:
        return []
    if len(text) > PLAN_CHARS:
        text = text[:PLAN_CHARS].rsplit("\n", 1)[0] + "\n… (cut: read plan.md for the rest)"
    return [
        "",
        f"## Research plan (`{PLAN_FILE}`, a clean-context review of this target)",
        "Start from its ranked directions and retry list. Do not repeat what it lists under "
        "*Do not try* unless you have new evidence; tag your evaluations with its `idea_id`s.",
        "",
        text,
    ]


def _best(run: RunDir, target_id: str, keeper: Truth | None) -> list[str]:
    best = best_for_target(run, target_id, keeper)
    if not best:
        return ["* no correct candidate faster than the reference yet"]
    sol = ""
    if best.get("pct_of_sol") is not None:
        sol = f", {best['pct_of_sol']:.0f} % of its recipe's roofline ({best.get('bound')} bound)"
    lines = [
        f"* `{best['snapshot']}` (exp {best.get('exp')}): {float(best['speedup']):.3f}x module "
        f"speedup{sol}"
        + (f"; idea `{best['idea']}`" if best.get("idea") else "")
        + f". Hypothesis: {_cell(best.get('hypothesis'), 300)}",
        "",
        "| case | calls/run | ref ms | new ms | speedup | SOL ms | % SOL | bound |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in best.get("cases") or []:
        cells = [f"`{_cell(c.get('signature'), 80)}`", str(c.get("calls_per_run", ""))]
        for key in ("ref_ms", "new_ms", "speedup", "sol_ms", "pct_of_sol"):
            value = c.get(key)
            cells.append("" if value is None else f"{value:.4g}")
        cells.append(str(c.get("bound") or ""))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def transforms_section(run: RunDir, top: int = TRANSFORM_ROWS) -> str:
    """``## End-to-end transforms`` of a near-lossless or relaxed run's research evidence: the
    fastest passing transform evaluations (a transform that already uses another precision on
    the target's modules is evidence for a precision pivot, ``pivot.py``)."""
    rows = [
        r
        for r in ledger.rows(run)
        if r["target"] == ledger.E2E
        and "transform" in r["backend"]
        and r["correct"]
        and r["speedup"] is not None
    ]
    rows.sort(key=lambda r: -float(r["speedup"]))
    if not rows:
        return ""
    lines = [
        "",
        "",
        f"## End-to-end transforms: the {min(top, len(rows))} fastest passing evaluations",
        "(model-level changes the systems agent measured; `transforms/` holds the files)",
        "",
        "| exp | speedup | transforms | hypothesis |",
        "|---|---|---|---|",
    ]
    for r in rows[:top]:
        names = ", ".join(ledger.snapshot_stem(s) for s in str(r["snapshot"]).split("+"))
        lines.append(
            f"| {r['exp']} | {float(r['speedup']):.3f}x | {_cell(names, 300)} | "
            f"{_cell(r['hypothesis'], 300)} |"
        )
    return "\n".join(lines)


def evidence(run: RunDir, target_id: str, reason: str, keeper: Truth | None = None) -> str:
    """The ``# Evidence`` of a research session: why it runs, the best result with its
    speed of light per case, the per-idea aggregates and the target's ledger rows (and its
    board history: the conclusions the sessions posted about it, ``board.py``)."""
    rows = ledger.measured(r for r in ledger.rows(run) if r["target"] == target_id)
    stats = target_ideas(run, target_id, keeper)
    kept = sum(r["status"] == ledger.KEEP for r in rows)
    failed = sum(r["status"] in ledger.FAILURES for r in rows)
    lines = [
        "## Why you were called",
        f"* {reason}. {len(rows)} evaluations so far: {kept} kept, {failed} failed.",
        "",
        "## Best so far",
        *_best(run, target_id, keeper),
        "",
        "## Ideas (the ledger's `idea` column)",
    ]
    if stats:
        lines += ideas_table(stats)
        untagged = sum(not r.get("idea") for r in rows)
        if untagged:
            lines.append(f"\n{untagged} evaluations have no `idea_id`: group them by hypothesis.")
    else:
        lines.append("* no evaluation was tagged with an `idea_id`: group them by hypothesis")
    shown = rows[-LEDGER_ROWS:]
    if shown:
        lines += [
            "",
            f"## Ledger: the last {len(shown)} of {len(rows)} evaluations (oldest first; "
            "`keep` = new best)",
            "",
            *rows_table(shown, sol=True),
        ]
    notes = [f"`{p.relative_to(run.target(target_id))}`" for _, p in all_notes(run, target_id)]
    if len(notes) > 1:
        lines += [
            "",
            f"The target has parallel workers; read every notes file: {', '.join(notes)}.",
        ]
    if plan_path(run, target_id).is_file():
        lines += [
            "",
            f"`{PLAN_FILE}` already holds an earlier research plan: check which of its "
            "directions were tried (checklist item 7) before you replace it.",
        ]
    lines += board.history_lines(run, target_id)  # the sessions' conclusions on it (#187)
    return "\n".join(lines)
