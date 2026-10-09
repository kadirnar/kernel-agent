"""Self-contained ``dashboard.html`` of a run (charts inlined as base64 PNG + tables).

Regenerated together with the charts after every phase (:func:`refresh`) and, through the
run's :class:`Refresher` (one thread, debounced), after evaluations, so it can be left open
in a browser while a run is going; it reloads itself every 30 s until the report phase is
done. Every rewrite first commits the new ledger rows to ``experiments.git``
(:func:`kernel_agent.expgit.sync`), so no evaluation waits for git either.
"""

from __future__ import annotations

import atexit
import base64
import html
import math
import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kernel_agent import charts, expgit, ledger
from kernel_agent.agent import auth
from kernel_agent.workspace import RunDir

#: After evaluations, a run's charts and dashboard are rewritten at most this often.
INTERVAL_S = 5.0
_warned = False
_lock = threading.Lock()  # one rewrite at a time: the refreshers' threads, phases, the report


def refresh(run: RunDir, target_id: str | None = None) -> None:
    """Rewrite the charts (only ``target_id``'s progress chart) and the dashboard.

    Never raises: monitoring must not break an optimisation run.
    """
    _refresh(run, None if target_id is None else [target_id])


def _refresh(run: RunDir, targets: list[str] | None) -> None:
    """:func:`refresh` of the progress charts of ``targets`` (None: of every target)."""
    global _warned
    expgit.sync(run)  # the experiments' git history (#224) first; it never raises
    try:
        with _lock:
            charts.write_charts(run, targets=targets)
            write_dashboard(run)
    except Exception as exc:
        if not _warned:
            print(f"[kernel-agent] chart/dashboard refresh failed: {exc!r}", file=sys.stderr)
            _warned = True


class Refresher:
    """Rewrites one run's charts and dashboard in a background thread, at most once per
    ``interval`` seconds, so evaluations (of any number of sessions) neither wait for the
    charts nor race on their files.

    Requests (:meth:`request`) coalesce until the thread gets to them: the progress charts
    of every target asked for, of all targets once one request asks for all. The first
    request after a quiet ``interval`` is served at once. The thread ends when nothing is
    pending; the next request starts another. ``render`` (tests): what a rewrite does."""

    def __init__(
        self,
        run: RunDir,
        interval: float = INTERVAL_S,
        render: Callable[[RunDir, list[str] | None], None] | None = None,
    ) -> None:
        self.run = run
        self.interval = interval
        self._render = render or _refresh
        self._cond = threading.Condition()
        self._targets: set[str] = set()
        self._all = False  # a pending request for every target
        self._pending = False
        self._busy = False
        self._now = False  # flush(): no waiting for the interval
        self._closed = False
        self._last = -math.inf  # time.monotonic() of the last rewrite
        self._thread: threading.Thread | None = None

    def request(self, target_id: str | None = None) -> None:
        """Ask for a rewrite (only ``target_id``'s progress chart; None: every target's)."""
        with self._cond:
            if self._closed:
                return
            if target_id is None:
                self._all = True
            else:
                self._targets.add(target_id)
            self._pending = True
            if self._thread is None:
                self._thread = threading.Thread(target=self._work, name="dashboard", daemon=True)
                self._thread.start()

    def _work(self) -> None:
        while True:
            with self._cond:
                while True:
                    if not self._pending or self._closed:
                        self._thread = None
                        self._cond.notify_all()
                        return
                    wait = self._last + self.interval - time.monotonic()
                    if wait <= 0 or self._now:
                        break
                    self._cond.wait(wait)
                targets = None if self._all else sorted(self._targets)
                self._targets, self._all, self._pending, self._now = set(), False, False, False
                self._busy = True
            try:
                self._render(self.run, targets)
            finally:
                with self._cond:
                    self._busy, self._last = False, time.monotonic()
                    self._cond.notify_all()

    def flush(self, timeout: float | None = None) -> bool:
        """Serve the pending requests now and wait for them (False: ``timeout`` passed)."""
        with self._cond:
            self._now = self._pending
            self._cond.notify_all()
            return self._cond.wait_for(lambda: not self._pending and not self._busy, timeout)

    def close(self, timeout: float | None = None) -> None:
        """Drop what is pending and wait (up to ``timeout``) for a rewrite in progress."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout)


_refreshers: dict[Path, Refresher] = {}
_refreshers_lock = threading.Lock()


def refresher(run: RunDir) -> Refresher:
    """The :class:`Refresher` of ``run`` in this process (one per run)."""
    key = run.root.resolve()
    with _refreshers_lock:
        found = _refreshers.get(key)
        if found is None:
            found = _refreshers[key] = Refresher(run)
        return found


@atexit.register
def _close_refreshers() -> None:  # a rewrite in progress finishes, nothing new starts
    with _refreshers_lock:
        found = list(_refreshers.values())
    for item in found:
        item.close(timeout=30.0)


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _ms(value: float | None, nd: int = 1) -> str:
    return "—" if value is None else f"{value:,.{nd}f} ms"


def _x(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}×"


def _img(path: Path, alt: str) -> str:
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode()
    return (
        f'<figure class="chart"><img alt="{_e(alt)}" src="data:image/png;base64,{data}"></figure>'
    )


def _status(status: str) -> str:
    kind = "keep" if status == ledger.KEEP else "discard" if status == ledger.DISCARD else "fail"
    return f'<span class="pill {kind}"><i></i>{_e(status.replace("_", " "))}</span>'


def _table(headers: list[str], rows: list[list[str]], numeric: set[int]) -> str:
    head = "".join(
        f'<th class="{"num" if i in numeric else ""}">{_e(h)}</th>' for i, h in enumerate(headers)
    )
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="{"num" if i in numeric else ""}">{cell}</td>' for i, cell in enumerate(r)
        )
        + "</tr>"
        for r in rows
    )
    table = f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    return f'<div class="scroll">{table}</div>'


def write_dashboard(run: RunDir) -> Path:
    s = ledger.summary(run)
    base = s["baseline_ms"]
    final = s["final"] or {}
    best = s["best_e2e"]
    measured = final.get("median_ms") if final.get("passed") else (best or {}).get("new_ms")
    running = s["phase_running"] or (s["phase"] != "report" and not s["phase_failed"])
    phase = ledger.phase_label(s)

    comp = s["compiled_ms"]  # strong baseline (strong_baseline.py)
    projected = s["projected_ms"]  # projection.Shown: the last accepted set's, or the kernels'
    # "projected from a + b; not counted (nested): c", "not projected (why): d", or why it
    # is not projectable (#128): never a ratio then
    counted = s["shown"].note()
    compiled = f"torch.compile {_ms(comp)} ({base / comp:.2f}×)" if comp and base else ""
    vs_comp = f" · {comp / measured:.2f}× vs. compiled" if comp and measured else ""
    tiles = [
        ("baseline (eager)", _ms(base), _e(compiled or s["workload"] or "")),
        (
            "measured end-to-end",
            _ms(measured),
            f"{base / measured:.2f}× vs. eager{vs_comp}"
            if base and measured
            else "no passing run yet",
        ),
        (
            "projected",
            _ms(projected),
            " · ".join(
                filter(
                    None, [f"{base / projected:.2f}×" if base and projected else "", _e(counted)]
                )
            ),
        ),
        (
            "evaluations",
            str(s["evaluations"]),
            f"{s['keeps']} kept · {s['failures']} failed",
        ),
        (
            "agent cost",
            f"${s['cost_usd']:.2f}",
            f"{len(s['costs'])} agents" + (f" · {n}" if (n := auth.usd_note(s["costs"])) else ""),
        ),
    ]
    tile_html = "".join(
        f'<div class="tile"><div class="label">{_e(label)}</div>'
        f'<div class="value">{_e(value)}</div><div class="sub">{sub}</div></div>'
        for label, value, sub in tiles
    )

    colors = {
        t: charts.TARGET_COLORS[i % len(charts.TARGET_COLORS)]
        for i, (t, _, _) in enumerate(charts.target_shares(run))
    }
    target_rows = [
        [
            f'<span class="swatch" style="background:{colors.get(t["id"], charts.OTHER_COLOR)}">'
            f"</span><code>{_e(t['id'])}</code>",
            f"<code>{_e(t['module_class'])}</code>",
            _e(t["precision"])
            + (f" (pivot of <code>{_e(t['pivot_of'])}</code>)" if t.get("pivot_of") else ""),
            _e(", ".join(t["backends"])),
            str(t["evals"]),
            str(t["keeps"]),
            str(t["failures"]),
            _x(t["best_speedup"]),
            _ms(t["saved_ms"]),  # in the metric's ms (per audio s for metric=throughput)
            _e(t["best_backend"] or ""),
            _e(t["last_hypothesis"]),
        ]
        for t in s["targets"]
    ]
    ledger_rows = [
        [
            str(r["exp"]),
            _e(str(r["time"])[11:] or r["time"]),
            f"<code>{_e(r['target'])}</code>",
            _e(r["backend"]),
            _status(r["status"]),
            _x(r["speedup"]),
            _ms(r["new_ms"], 3) if r["target"] != ledger.E2E else _ms(r["new_ms"]),
            _ms(s["units"].of_row(r)),  # a kernel's estimate in the metric's ms, as e2e rows
            f'<span title="{_e(ledger.labelled(r))}">{_e(ledger.title(r))}</span>',  # hover
        ]
        for r in reversed(s["rows"][-30:])
    ]
    cost_rows = [
        [
            _e(name),
            f"${float(c.get('usd') or 0):.2f}",
            _e(c.get("turns")),
            _e(c.get("minutes")),
            _e(", ".join(f"{k}×{v}" for k, v in (c.get("tools") or {}).items())),
        ]
        for name, c in s["costs"].items()
    ]

    target_charts = "".join(
        _img(run.target(t["id"]) / "progress.png", f"{t['id']} progress") for t in s["targets"]
    )
    sections = [
        '<section><h2>Progress</h2><div class="stack">'
        + _img(run.root / "progress.png", "progress over experiment number")
        + _img(run.root / "timeline.png", "end-to-end latency over wall-clock time")
        + _img(run.root / "amdahl.png", "time split before and after")
        + _img(run.root / "integration.png", "integration waterfall")
        + "</div></section>",
    ]
    if target_charts:
        sections.append(
            f'<section><h2>Targets</h2><div class="grid">{target_charts}</div></section>'
        )
    sections.append(
        "<section><h2>Target summary</h2>"
        + _table(
            [
                "target",
                "class",
                "precision",
                "backends",
                "evals",
                "kept",
                "failed",
                "best",
                "est. saved",
                "best backend",
                "last hypothesis",
            ],
            target_rows,
            {4, 5, 6, 7, 8},
        )
        + "</section>"
    )
    sections.append(
        "<section><h2>Latest evaluations</h2>"
        + _table(
            ["#", "time", "target", "backend", "status", "speedup", "new", "saved", "title"],
            ledger_rows,
            {0, 5, 6, 7},
        )
        + '<p class="note">Full ledger: <code>results.tsv</code> · events: '
        "<code>events.jsonl</code></p></section>"
    )
    if cost_rows:
        sections.append(
            "<section><h2>Agents</h2>"
            + _table(["agent", "cost", "turns", "minutes", "tools"], cost_rows, {1, 2, 3})
            + "</section>"
        )

    elapsed = f" · {s['elapsed_min']:.0f} min" if s["elapsed_min"] is not None else ""
    refresh_meta = '<meta http-equiv="refresh" content="30">' if running else ""
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{refresh_meta}
<title>{_e(s["repo_id"])} · kernel-agent</title>
<style>
:root {{
  color-scheme: light dark;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --line: #e1e0d9; --keep: {charts.KEEP_COLOR}; --discard: {charts.DISCARD_COLOR};
  --fail: {charts.FAIL_COLOR}; --code: #f0efec;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --line: #2c2c2a; --code: #2c2c2a;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
}}
main {{ max-width: 1320px; margin: 0 auto; padding: 24px 16px 48px; }}
header h1 {{ font-size: 22px; margin: 0 0 4px; }}
header p {{ margin: 0; color: var(--ink-2); }}
h2 {{ font-size: 15px; margin: 0 0 10px; }}
section {{ margin-top: 28px; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px;
  margin-top: 20px; }}
.tile {{ background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
  padding: 12px 14px; }}
.tile .label {{ color: var(--ink-2); font-size: 12px; }}
.tile .value {{ font-size: 24px; font-weight: 600; margin-top: 2px; }}
.tile .sub {{ color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }}
.chart {{ margin: 0; background: #fcfcfb; border: 1px solid var(--line); border-radius: 10px;
  padding: 10px; }}
.chart img {{ display: block; width: 100%; height: auto; }}
.stack {{ display: grid; gap: 16px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 600px), 1fr));
  gap: 16px; align-items: start; }}
.scroll {{ overflow-x: auto; background: var(--surface); border: 1px solid var(--line);
  border-radius: 10px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line);
  vertical-align: top; }}
th {{ color: var(--ink-2); font-weight: 600; white-space: nowrap; }}
tr:last-child td {{ border-bottom: 0; }}
td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
code {{ background: var(--code); border-radius: 4px; padding: 1px 5px; font-size: 12px; }}
.pill {{ display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }}
.pill i {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; }}
.pill.keep i {{ background: var(--keep); box-shadow: 0 0 0 1px var(--ink); }}
.pill.keep {{ font-weight: 600; }}
.pill.discard i {{ background: var(--discard); }}
.pill.fail i {{ background: var(--fail); border-radius: 2px; transform: rotate(45deg); }}
.swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px;
  margin-right: 6px; vertical-align: -1px; }}
.note {{ color: var(--muted); font-size: 12px; margin: 8px 0 0; }}
</style>
</head>
<body>
<main>
<header>
<h1>{_e(s["repo_id"])}</h1>
<p>{_e(s["modality"] or "")} · phase: <strong>{_e(phase)}</strong>{_e(elapsed)} ·
<code>{_e(s["root"])}</code></p>
<p class="note">updated {_e(time.strftime("%Y-%m-%d %H:%M:%S"))}</p>
</header>
<div class="tiles">{tile_html}</div>
{"".join(sections)}
</main>
</body>
</html>
"""
    tmp = run.dashboard.with_name(f".dashboard.tmp.{os.getpid()}.{threading.get_native_id()}.html")
    tmp.write_text(page)
    tmp.replace(run.dashboard)
    return run.dashboard
