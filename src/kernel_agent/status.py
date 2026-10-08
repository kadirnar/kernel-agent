"""``kernel-agent status``: terminal view of a run, from the ledger and the event log."""

from __future__ import annotations

import shutil
import textwrap
import time
from typing import Any

from kernel_agent import (
    backends,
    diversity,
    governor,
    gpuqueue,
    ledger,
    objective,
    projection,
    sessions,
    strong_baseline,
)
from kernel_agent.agent import auth
from kernel_agent.config import QUALITY_NOTES
from kernel_agent.workspace import RunDir, read_json


def _ms(value: float | None, nd: int = 1) -> str:
    return "—" if value is None else f"{value:,.{nd}f} ms"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f} %"


def _x(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}x"


def _table(headers: list[str], rows: list[list[str]], right: set[int], width: int) -> list[str]:
    """Aligned columns; the last column is cut to fit ``width``."""
    widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]
    fixed = sum(widths[:-1]) + 2 * (len(headers) - 1)
    widths[-1] = max(min(widths[-1], width - fixed), 12)

    def line(cells: list[str]) -> str:
        out = []
        for i, cell in enumerate(cells):
            if i == len(cells) - 1:
                cell = cell if len(cell) <= widths[i] else cell[: widths[i] - 1] + "…"
                out.append(cell)
            else:
                out.append(cell.rjust(widths[i]) if i in right else cell.ljust(widths[i]))
        return "  ".join(out).rstrip()

    return [line(headers), *(line(r) for r in rows)]


def _agents(run: RunDir, width: int, running: bool) -> list[str]:
    """The Agents table: the agent sessions open now (role, arm, state and for how long,
    evaluations, the USD reserved for them), and the time split of every session so far
    (``sessions.py``; [] before the first session)."""
    live, ended = sessions.agents(run)
    if not live and not ended:
        return []
    lines = []
    now = time.time()
    if live:
        state = read_json(run.root / "improve.json", {}) or {}
        config = state.get("config") or {}
        n = config.get("agents") if isinstance(config, dict) else None
        if isinstance(config, dict) and config.get("governor"):
            n = f"auto, up to {n}"
        lines.append(
            f"agents: {len(live)} running"
            + (f" (--agents {n})" if isinstance(n, str) or (isinstance(n, int) and n > 1) else "")
            + ("" if running else " (the run is not running: left open)")
        )
        if line := governor.status_line(state.get("governor")):  # --agents auto (#191)
            lines.append(line[:width])
        rows = [
            [
                a.label,
                a.role,
                a.arm or "—",
                sessions.duration(max(now - a.since, 0.0)),
                str(a.evaluations),
                f"≤{a.reserved:.2f}" if a.reserved else "—",
                sessions.describe(a.state, a.detail),
            ]
            for a in live
        ]
        headers = ["session", "role", "arm", "for", "evals", "$", "state"]
        lines += _table(headers, rows, {3, 4, 5}, width)
    every = [*ended, *live]
    split = sessions.total_split(every, max(now, *(a.since for a in every)))
    if text := sessions.split_text(split):
        hours = sum(split.values()) / 3600
        lines.append(f"agent time ({len(every)} sessions, {hours:.1f} h): {text}"[:width])
    return lines


def render(run: RunDir, width: int | None = None, last: int = 10) -> str:
    s: dict[str, Any] = ledger.summary(run)
    width = width or shutil.get_terminal_size((120, 40)).columns
    base = s["baseline_ms"]
    phase = ledger.phase_label(s)
    elapsed = f"  ·  {s['elapsed_min']:.0f} min" if s["elapsed_min"] is not None else ""
    quality = str((run.load().get("config") or {}).get("quality") or "exact")
    lines = [
        f"{s['repo_id']} ({s['modality'] or '?'})  ·  phase: {phase}{elapsed}",
        f"run: {s['root']}",
        f"quality: {quality} ({QUALITY_NOTES.get(quality, '?')})"[:width],
        *([line[:width]] if (line := objective.describe(s["baseline"])) else []),  # metric=ttfa
        "",
    ]

    comp = s["compiled_ms"]

    def vs(ms: float) -> str:  # speedup vs eager (and vs the compiled baseline when measured)
        return f"{_x(base / ms)} vs eager, {_x(comp / ms)} vs compiled" if comp else _x(base / ms)

    parts = [f"baseline {_ms(base)}"]
    if comp:
        parts.append(f"compiled {_ms(comp)} ({_x(base / comp) if base else '—'})")
    shown = s["shown"]  # projection.Shown: the integration's last accepted set, or the kernels'
    if base and s["projected_ms"]:
        parts.append(f"projected {_ms(s['projected_ms'])} ({_x(base / s['projected_ms'])})")
    elif shown.why:  # estimates that exceed the run: the reason below, never a ratio (#128)
        parts.append("projected: not projectable")
    best = s["best_e2e"]
    final = s["final"] or {}
    if base and final.get("passed") and final.get("median_ms"):
        parts.append(f"measured {_ms(final['median_ms'])} ({vs(final['median_ms'])}, integrated)")
        if spread := diversity.headline(final):  # the diverse set's speedups (#170)
            parts.append(spread)
    elif best and base and best["new_ms"]:
        parts.append(f"measured {_ms(best['new_ms'])} ({vs(best['new_ms'])}, {best['snapshot']})")
        if best.get("diverse_speedup"):
            dependent = diversity.DATA_DEPENDENT in str(best.get("flags"))
            parts.append(
                f"diverse-set median {_x(best['diverse_speedup'])}"
                + (", data-dependent" if dependent else "")
            )
    else:
        parts.append("measured —")
    notional = auth.usd_note(s["costs"])
    parts.append(f"cost ${s['cost_usd']:.2f}" + (f" {notional}" if notional else ""))
    header = [parts[0]]
    for part in parts[1:]:  # wrap between parts at the terminal width
        if len(header[-1]) + 5 + len(part) > width:
            header.append(part)
        else:
            header[-1] += "  |  " + part
    lines += header
    # what the projection counts (projection.py: nested targets once, accepted sets #121) or
    # why it is not projectable, and what a capture from before #119 means: wrapped, not cut
    for text in (shown.note(), projection.recapture_hint(run)):
        lines += textwrap.wrap(
            text, width, subsequent_indent="  ", break_long_words=False, break_on_hyphens=False
        )
    if s["reference"]:
        lines.append(
            "compiled baseline + accepted kernels: "
            + strong_baseline.combination_text(s["reference"])
        )
    elif s["compiled_ms"] is None and (line := strong_baseline.describe(s["baseline"])):
        lines.append(line.replace("**", ""))  # the compiled baseline failed: say why
    lines.append("")

    rows = [
        [
            t["id"],
            str(t["module_class"] or "")
            + (f" [{t['precision']}]" if t.get("precision", "exact") != "exact" else "")
            + (f" ({len(t['workers'])} workers)" if len(t.get("workers") or []) > 1 else ""),
            str(t["evals"]),
            str(t["keeps"]),
            str(t["failures"]),
            _x(t["best_speedup"]),
            _pct(t.get("best_pct_of_sol")),
            _ms(t["saved_ms"]),  # in the metric's ms (per audio s for metric=throughput)
            t["last_hypothesis"] or "",
        ]
        for t in s["targets"]
    ]
    e2e = [r for r in s["rows"] if r["target"] == ledger.E2E]
    if e2e:
        rows.append(
            [
                ledger.E2E,
                "(transforms, integration)",
                str(len(e2e)),
                str(sum(r["status"] == ledger.KEEP for r in e2e)),
                str(sum(r["status"] in ledger.FAILURES for r in e2e)),
                _x(
                    max(
                        (r["speedup"] or 0.0 for r in e2e if r["status"] == ledger.KEEP),
                        default=None,
                    )
                ),
                "—",
                _ms(best["est_saved_ms"]) if best else "—",
                e2e[-1]["hypothesis"] or "",
            ]
        )
    if rows:
        headers = [
            "target",
            "class",
            "evals",
            "kept",
            "failed",
            "best",
            "% SOL",
            "est. saved",
            "last hypothesis",
        ]
        lines += _table(headers, rows, {2, 3, 4, 5, 6, 7}, width)
    else:
        lines.append("no evaluations yet")
    if per_backend := backends.status_lines(run, s["rows"]):  # by source, priors left out
        lines += ["", *(line[:width] for line in per_backend)]
    if agents := _agents(run, width, bool(s["phase_running"])):  # sessions.jsonl (#184)
        lines += ["", *agents]
    if gpu := gpuqueue.status_lines(run, width):  # the GPU job queue (gpu_queue.jsonl)
        lines += ["", *gpu, *sessions.gpu_lines(run, width)]

    recent = s["rows"][-last:]
    if recent:
        lines += ["", f"last {len(recent)} evaluations (hypotheses: kernel-agent exp show)"]
        lines += _table(
            ["#", "time", "target", "backend", "status", "speedup", "title"],
            [
                [
                    str(r["exp"]),
                    str(r["time"])[11:] or str(r["time"]),
                    r["target"] + (f"/w{r['worker']}" if r.get("worker") else ""),
                    r["backend"],
                    r["status"]
                    + (
                        " (data-dependent)"
                        if diversity.DATA_DEPENDENT in str(r.get("flags"))
                        else ""
                    ),
                    _x(r["speedup"]),
                    ledger.title(r),  # the hypothesis: `kernel-agent exp show` (#222)
                ]
                for r in recent
            ],
            {0, 5},
            width,
        )
    if run.dashboard.exists():
        lines += ["", f"dashboard: {run.dashboard}"]
    return "\n".join(lines)
