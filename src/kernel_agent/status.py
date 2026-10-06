"""``kernel-agent status``: terminal view of a run, from the ledger and the event log."""

from __future__ import annotations

import shutil
from typing import Any

from kernel_agent import ledger, objective, strong_baseline
from kernel_agent.agent import auth
from kernel_agent.workspace import RunDir


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


def render(run: RunDir, width: int | None = None, last: int = 10) -> str:
    s: dict[str, Any] = ledger.summary(run)
    width = width or shutil.get_terminal_size((120, 40)).columns
    base = s["baseline_ms"]
    phase = ledger.phase_label(s)
    elapsed = f"  ·  {s['elapsed_min']:.0f} min" if s["elapsed_min"] is not None else ""
    lines = [
        f"{s['repo_id']} ({s['modality'] or '?'})  ·  phase: {phase}{elapsed}",
        f"run: {s['root']}",
        *([line[:width]] if (line := objective.describe(s["baseline"])) else []),  # metric=ttfa
        "",
    ]

    comp = s["compiled_ms"]

    def vs(ms: float) -> str:  # speedup vs eager (and vs the compiled baseline when measured)
        return f"{_x(base / ms)} vs eager, {_x(comp / ms)} vs compiled" if comp else _x(base / ms)

    parts = [f"baseline {_ms(base)}"]
    if comp:
        parts.append(f"compiled {_ms(comp)} ({_x(base / comp) if base else '—'})")
    if base and s["projected_ms"]:
        parts.append(f"projected {_ms(s['projected_ms'])} ({_x(base / s['projected_ms'])})")
    best = s["best_e2e"]
    final = s["final"] or {}
    if base and final.get("passed") and final.get("median_ms"):
        parts.append(f"measured {_ms(final['median_ms'])} ({vs(final['median_ms'])}, integrated)")
    elif best and base and best["new_ms"]:
        parts.append(f"measured {_ms(best['new_ms'])} ({vs(best['new_ms'])}, {best['snapshot']})")
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
    if s["projection"] and s["projection"].shown:  # projection.py: nested targets counted once
        text = s["projection"].headline()
        lines.append(text if len(text) <= width else text[: width - 1] + "…")
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

    recent = s["rows"][-last:]
    if recent:
        lines += ["", f"last {len(recent)} evaluations"]
        lines += _table(
            ["#", "time", "target", "backend", "status", "speedup", "hypothesis"],
            [
                [
                    str(r["exp"]),
                    str(r["time"])[11:] or str(r["time"]),
                    r["target"] + (f"/w{r['worker']}" if r.get("worker") else ""),
                    r["backend"],
                    r["status"],
                    _x(r["speedup"]),
                    ledger.labelled(r) or r["snapshot"] or "",
                ]
                for r in recent
            ],
            {0, 5},
            width,
        )
    if run.dashboard.exists():
        lines += ["", f"dashboard: {run.dashboard}"]
    return "\n".join(lines)
