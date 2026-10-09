"""A run's experiments: its ledger rows seen as autoresearch-style experiments (issue #222).

Every measured row of ``results.tsv`` is one experiment (:func:`experiments`): what it
measured (``kind``: ``kernel``, ``e2e``, an ``integration`` step or an integration
``probe``, :func:`kernel_agent.ledger.kind`), its ``lineage`` (the kernel target, or
:data:`MODEL` for everything measured end to end), its ``value`` (a kernel's module speedup
in ×, the model's metric in ms), the lineage's best before it and the ``gain`` over that,
its ``title`` (:func:`kernel_agent.ledger.title`), the ``files`` it measured (its
``evaluation`` event) and its parent experiment. Rows that measured nothing (quick checks,
duplicates, re-evaluations: :data:`kernel_agent.ledger.UNMEASURED`) are counted, not
listed; a re-evaluation still moves its lineage's best, as in the ledger's keep rule.

``exp`` is the order rows were *recorded* in: every row is numbered and classified under
the ledger's lock, in the one process that holds the run's coordinator lock, so a row's
status is relative to every row with a smaller number and a lineage's kept values improve
along ``exp`` (a re-evaluation can lower the bar in between). With asynchronous
evaluations (#191) an evaluation that started earlier can get a later number.

``kernel-agent exp`` (:func:`add_parser`, :func:`main`) prints them: ``exp RUN`` the
headline "N experiments, K kept improvements", a table per lineage and the kept
improvements ranked by gain; ``exp list``, ``exp show`` and ``exp diff``. Every command
only reads the run and takes no file lock: ``ledger.read_tsv`` leaves out a row still being
written and partly written JSON lines are skipped, so they are safe on a live run.
"""

from __future__ import annotations

import argparse
import difflib
import json
import shutil
import sys
import textwrap
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kernel_agent import dedup, ledger, objective
from kernel_agent.budget import Standing
from kernel_agent.workspace import RunDir

MODEL = "model"  # the lineage of every end-to-end measurement (e2e, integration, probe)
X, MS = "×", "ms"  # units of a kernel lineage (module speedup) and of the model's metric
ERROR_TAIL = 800  # characters of an error that ``exp show`` prints


@dataclass(frozen=True)
class Experiment:
    """One ledger row seen as an experiment. ``value`` is in ``unit`` (× for a kernel
    lineage, the metric's ms for :data:`MODEL`; None when it failed), ``best_before`` the
    lineage's standing best before it in the same unit (the reference's 1.0×, the
    baseline's ms), ``gain`` its speedup over that best (0.05 = 5 % faster; None when it
    failed); ``parent_exp`` the experiment it built on: its ``parent`` snapshot's, else the
    lineage's standing best at that time (None: the reference or the baseline)."""

    exp: int
    time: str
    kind: str
    lineage: str
    status: str
    value: float | None
    unit: str
    best_before: float | None
    gain: float | None
    title: str
    hypothesis: str
    files: tuple[str, ...]
    parent_exp: int | None
    session: str
    worker: str
    idea: str
    speedup: float | None = None  # the row's (module or end-to-end) speedup when correct
    row: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def measured(self) -> bool:
        return self.status not in ledger.UNMEASURED

    @property
    def failed(self) -> bool:
        return self.status in ledger.FAILURES


# ------------------------------------------------------------------ reading a run


def _jsonl(path: Path) -> list[dict[str, Any]]:
    """The records of a JSON-lines file; a line that does not parse (one still being
    written by the run) is skipped."""
    out = []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            out.append(record)
    return out


def _json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _event_files(run: RunDir) -> dict[int, tuple[str, ...]]:
    """``exp`` → the run-relative files its ``evaluation`` event lists."""
    out: dict[int, tuple[str, ...]] = {}
    for ev in _jsonl(run.events):
        if ev.get("event") == "evaluation" and isinstance(ev.get("exp"), int) and ev.get("files"):
            out[ev["exp"]] = tuple(str(f) for f in ev["files"])
    return out


def _e2e_files(run: RunDir) -> dict[int, tuple[str, ...]]:
    """``exp`` → the files of the transforms' records (``evaluate_e2e`` rows recorded
    before their events listed files)."""
    out: dict[int, tuple[str, ...]] = {}
    history = run.history_dir()
    for rec in _jsonl(run.results_file()):
        if isinstance(rec.get("exp"), int):
            items = [str(history / Path(str(t)).name) for t in rec.get("transforms") or []]
            items += [str(k) for k in rec.get("kernels") or []]
            out[rec["exp"]] = tuple(ledger.item_files(run, items))
    return out


def _parent(row: dict[str, Any], earlier: list[dict[str, Any]]) -> int | None:
    """The experiment a row's ``parent`` names: the latest earlier measured row of its
    lineage with that snapshot, else with a snapshot of the same file stem (a parent given
    as ``candidates/<stem>.py``)."""
    parent = Path(str(row.get("parent") or ""))
    if not parent.name:
        return None
    for match in (
        lambda r: Path(str(r.get("snapshot") or "")).name == parent.name,
        lambda r: ledger.snapshot_stem(str(r.get("snapshot") or "")) == parent.stem,
    ):
        found = next((r for r in reversed(earlier) if match(r)), None)
        if found is not None:
            return int(found["exp"])
    return None


def _ms(top: dict[str, Any] | None, base: float | None, speedup: float) -> float | None:
    """The model's standing best in ms: its row's measured ms, else ``base`` / speedup."""
    if top is not None and top.get("new_ms"):
        return float(top["new_ms"])
    return base / speedup if base else None


def baseline_ms(run: RunDir) -> float | None:
    value = _json(run.baseline_json).get("median_ms")
    return float(value) if isinstance(value, int | float) and value > 0 else None


def all_rows(run: RunDir, rows: list[dict[str, Any]] | None = None) -> list[Experiment]:
    """Every ledger row (``rows``: these instead of the run's) as an :class:`Experiment`,
    unmeasured ones included, in ``exp`` order."""
    rows = ledger.rows(run) if rows is None else rows
    files = _event_files(run)
    e2e_files: dict[int, tuple[str, ...]] | None = None  # read when an older row needs it
    base = baseline_ms(run)
    stands: dict[str, Standing] = {}
    earlier: dict[str, list[dict[str, Any]]] = {}  # measured rows per lineage so far
    out: list[Experiment] = []
    for row in sorted((r for r in rows if isinstance(r.get("exp"), int)), key=lambda r: r["exp"]):
        kind = ledger.kind(row)
        lineage = MODEL if kind != ledger.KERNEL else str(row.get("target"))
        stand = stands.setdefault(lineage, Standing())
        before, top = stand.best, stand.top  # the bar this row was classified against
        status = str(row.get("status") or "")
        correct = status in (ledger.KEEP, ledger.DISCARD) or (
            status == ledger.REEVALUATED and bool(row.get("correct"))
        )
        speedup = row.get("speedup") if correct else None
        ref = row.get("ref_ms") or base
        if lineage == MODEL:  # the best's own measured ms (the baseline's before any)
            unit = MS
            value = row.get("new_ms") if correct else None
            best_before = _ms(top, ref, before)
        else:
            unit, value, best_before = X, speedup, before
        exp_files = files.get(row["exp"])
        if exp_files is None and kind == ledger.KERNEL and row.get("snapshot"):
            exp_files = tuple(ledger.kernel_files(run, lineage, str(row["snapshot"])))
        elif exp_files is None and kind == ledger.E2E:
            e2e_files = _e2e_files(run) if e2e_files is None else e2e_files
            exp_files = e2e_files.get(row["exp"])
        parent = _parent(row, earlier.get(lineage, []))
        if parent is None and top is not None:  # it started from the standing best
            parent = top.get("exp")
        out.append(
            Experiment(
                exp=row["exp"],
                time=str(row.get("time") or ""),
                kind=kind,
                lineage=lineage,
                status=status,
                value=value,
                unit=unit,
                best_before=best_before,
                gain=speedup / before - 1 if speedup and before else None,
                title=ledger.title(row),
                hypothesis=str(row.get("hypothesis") or ""),
                files=exp_files or (),
                parent_exp=parent,
                session=str(row.get("session") or ""),
                worker=str(row.get("worker") or ""),
                idea=str(row.get("idea") or ""),
                speedup=speedup,
                row=row,
            )
        )
        if status == ledger.REEVALUATED:
            stand.replace(row)
        elif status == ledger.KEEP:
            stand.keep(row)
        if status not in ledger.UNMEASURED:
            earlier.setdefault(lineage, []).append(row)
    return out


def experiments(run: RunDir, rows: list[dict[str, Any]] | None = None) -> list[Experiment]:
    """The run's experiments: its measured ledger rows (probes included, ``kind`` says)."""
    return [e for e in all_rows(run, rows) if e.measured]


def kept(run: RunDir, items: list[Experiment] | None = None) -> list[Experiment]:
    """The kept improvements ("top hits"): every new best of a lineage, ranked by its gain
    over the best before it (the larger first; equal gains in ``exp`` order)."""
    items = experiments(run) if items is None else [e for e in items if e.measured]
    return sorted((e for e in items if e.status == ledger.KEEP), key=lambda e: -(e.gain or 0.0))


def summary(run: RunDir, items: list[Experiment] | None = None) -> dict[str, Any]:
    """The run's experiments in numbers: the headline (``experiments`` N: measured rows
    without the integration's probes; ``kept`` K: the new bests), the ``probes`` and the
    ``unmeasured`` rows next to it, counts by kind and status, keep and failure rates (of
    the N), and per lineage its start → best, experiments, kept, failed and the experiments
    since its last keep (``since_keep``: the plateau streak)."""
    everything = all_rows(run) if items is None else items
    measured = [e for e in everything if e.measured]
    probes = [e for e in measured if e.kind == ledger.PROBE]
    exps = [e for e in measured if e.kind != ledger.PROBE]
    by_status = Counter(e.status for e in exps)
    failures = sum(e.failed for e in exps)
    base = baseline_ms(run)
    lineages = []
    for name in dict.fromkeys([MODEL, *(e.lineage for e in everything)]):
        mine = [e for e in everything if e.lineage == name]
        steps = [e for e in mine if e.measured and e.kind != ledger.PROBE]
        if not steps:
            continue
        stand = Standing()
        for e in mine:  # the standing best at the end, re-evaluations included
            if e.status == ledger.REEVALUATED:
                stand.replace(e.row)
            elif e.status == ledger.KEEP:
                stand.keep(e.row)
        keeps = [i for i, e in enumerate(steps) if e.status == ledger.KEEP]
        top = stand.top
        start: float | None = 1.0  # the reference module
        best: float | None = stand.best
        if name == MODEL:  # in the metric's ms: the baseline's, and the best's own
            start = base or next((e.row["ref_ms"] for e in mine if e.row.get("ref_ms")), None)
            best = _ms(top, start, stand.best)
        lineages.append(
            {
                "lineage": name,
                "unit": MS if name == MODEL else X,
                "start": start,
                "best": best,
                "best_exp": top.get("exp") if top else None,
                "experiments": len(steps),
                "kept": len(keeps),
                "failed": sum(e.failed for e in steps),
                "since_keep": len(steps) - 1 - keeps[-1] if keeps else len(steps),
            }
        )
    data = _json(run.run_json)
    return {
        "repo_id": (data.get("card") or {}).get("repo_id") or run.root.name,
        "metric": objective.of(_json(run.baseline_json)).label,
        "baseline_ms": base,
        "experiments": len(exps),
        "kept": sum(e.status == ledger.KEEP for e in measured),
        "failed": sum(e.failed for e in measured),
        "probes": len(probes),
        "unmeasured": dict(Counter(e.status for e in everything if not e.measured)),
        "by_kind": {k: n for k in ledger.KINDS if (n := sum(e.kind == k for e in measured))},
        "by_status": dict(by_status),
        "keep_rate": by_status[ledger.KEEP] / len(exps) if exps else None,
        "failure_rate": failures / len(exps) if exps else None,
        "lineages": lineages,
    }


# ------------------------------------------------------------------ code and diffs


def _key(file: str) -> tuple[str, str]:
    """(what a measured file is, its path): ``target`` for a kernel's snapshot (a kernel
    row's or an item ``target=path``), ``target/rewrite.py``, ``transform <stem>``."""
    target, sep, path = file.partition("=")
    if sep and "/" not in target:
        return target, path
    parts = Path(file).parts
    if "targets" in parts[:-1]:
        target = parts[parts.index("targets") + 1]
        return (f"{target}/rewrite.py" if parts[-1] == "rewrite.py" else target), file
    return f"transform {ledger.snapshot_stem(file)}", file


def code(run: RunDir, e: Experiment) -> dict[str, tuple[str, str | None]]:
    """What an experiment measured: per file (:func:`_key`) its path and text (None:
    missing from the run directory)."""
    out: dict[str, tuple[str, str | None]] = {}
    for file in e.files:
        key, path = _key(file)
        where = Path(path) if Path(path).is_absolute() else run.root / path
        try:
            text: str | None = where.read_text(errors="replace")
        except OSError:
            text = None
        out[key] = (path, text)
    return out


def diff(run: RunDir, a: Experiment, b: Experiment) -> str:
    """A unified diff of what ``b`` measured → what ``a`` measured, file by file (files
    matched by target, rewrite and transform stem); "" when they measured the same code."""
    new, old = code(run, a), code(run, b)
    chunks = []
    for key in dict.fromkeys([*old, *new]):
        (path_b, text_b), (path_a, text_a) = old.get(key, ("", "")), new.get(key, ("", ""))
        if text_a is None or text_b is None:
            missing = path_a if text_a is None else path_b
            chunks.append(f"# {key}: {missing} is not in the run directory\n")
            continue
        lines = difflib.unified_diff(
            text_b.splitlines(keepends=True),
            text_a.splitlines(keepends=True),
            fromfile=f"exp {b.exp}: {path_b}" if path_b else "/dev/null",
            tofile=f"exp {a.exp}: {path_a}" if path_a else "/dev/null",
        )
        chunks.append("".join(line if line.endswith("\n") else line + "\n" for line in lines))
    return "".join(chunks)


def record(run: RunDir, e: Experiment) -> dict[str, Any] | None:
    """The ``results.jsonl`` record of an experiment (a kernel's, a quick check's or an
    ``evaluate_e2e``'s; None for the integration's, whose steps ``integration.json`` keeps)."""
    if e.kind == ledger.KERNEL:
        paths = [run.results_file(e.lineage), dedup.quick_file(run, e.lineage)]
    elif e.kind == ledger.E2E:
        paths = [run.results_file()]
    else:
        return None
    for path in paths:
        for rec in _jsonl(path):
            if rec.get("exp") == e.exp:
                return rec
    return None


def digest(e: Experiment, rec: dict[str, Any]) -> list[tuple[str, str]]:
    """What ``exp show`` prints of a record: the evaluator's status, the first correctness
    stage that failed (and its check) or the failed quality checks, the timing spread, an
    early stop, the error's tail."""
    out = [("evaluator", str(rec.get("status") or "—"))]  # its status, not the ledger's
    if rec.get("stage"):
        check = json.dumps(rec.get("failed_check"), default=str) if rec.get("failed_check") else ""
        out.append(("stage", f"{rec['stage']}" + (f": {check[:300]}" if check else "")))
    if rec.get("reason"):
        out.append(("checks", str(rec["reason"])[:500]))
    spread = ledger.kernel_spread(rec) if e.kind == ledger.KERNEL else ledger.e2e_spread(rec)
    if spread is not None:
        out.append(("spread", f"{spread:.1%}"))
    early = rec.get("early") or (rec.get("ab") or {}).get("stopped")
    if early:
        out.append(("early", json.dumps(early, default=str)[:300]))
    if rec.get("reevaluates"):
        out.append(("re-evaluates", json.dumps(rec["reevaluates"], default=str)[:300]))
    if error := str(rec.get("error") or "").strip():
        out.append(("error", error[-ERROR_TAIL:]))
    return out


# ------------------------------------------------------------------ formatting


def _cell(value: Any) -> str:
    """A TSV cell: one line, no tabs; numbers as the ledger writes them."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return " ".join(str(value).split())


def value_text(value: float | None, unit: str) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}{X}" if unit == X else f"{value:,.1f} ms"


def gain_text(gain: float | None) -> str:
    return "—" if gain is None else f"{gain:+.1%}"


def _rate(rate: float | None) -> str:
    return "" if rate is None else f" ({rate:.0%})"


def _width() -> int | None:
    """Columns to fit a table into: the terminal's; None (no cut) when not a terminal."""
    return shutil.get_terminal_size().columns if sys.stdout.isatty() else None


def table(headers: list[str], rows: list[list[str]], right: Iterable[int] = ()) -> list[str]:
    """Aligned columns; the last one is cut to fit the terminal (:func:`_width`)."""
    right = set(right)
    widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]
    room = None if (width := _width()) is None else width - sum(widths[:-1]) - 2 * len(widths[:-1])

    def line(cells: list[str]) -> str:
        out = [
            c.rjust(widths[i]) if i in right else c.ljust(widths[i]) for i, c in enumerate(cells)
        ]
        last = cells[-1]
        out[-1] = ledger.cut(last, max(room, 12)) if room is not None else last
        return "  ".join(out).rstrip()

    return [line(headers), *(line(r) for r in rows)]


def headline(s: dict[str, Any]) -> str:
    return f"{s['repo_id']}: {s['experiments']} experiments, {s['kept']} kept improvements"


def render_summary(run: RunDir) -> str:
    """``kernel-agent exp RUN``: the headline, the counts, a table per lineage and the kept
    improvements ranked by gain."""
    everything = all_rows(run)
    s = summary(run, everything)
    lines = [headline(s)]
    beside = [f"+{s['probes']} integration probes"] if s["probes"] else []
    if unmeasured := s["unmeasured"]:
        counts = ", ".join(f"{n} {k}" for k, n in sorted(unmeasured.items(), key=lambda x: -x[1]))
        beside.append(f"{sum(unmeasured.values())} rows not measured: {counts}")
    if beside:
        lines.append("(" + "; ".join(beside) + ")")
    if s["by_kind"]:
        lines.append("kinds: " + ", ".join(f"{n} {k}" for k, n in s["by_kind"].items()))
    by = s["by_status"]
    fails = sorted(((k, n) for k, n in by.items() if k in ledger.FAILURES), key=lambda x: -x[1])
    nfail = sum(n for _, n in fails)
    detail = ": " + ", ".join(f"{n} {k}" for k, n in fails) if fails else ""
    lines.append(
        f"status: {by.get(ledger.KEEP, 0)} keep{_rate(s['keep_rate'])}, "
        f"{by.get(ledger.DISCARD, 0)} discard, {nfail} failed{_rate(s['failure_rate'])}{detail}"
    )
    lines.append(f"model: {s['metric']}, in ms; kernels: module speedup")
    if s["lineages"]:
        lines.append("")
        rows = [
            [
                ln["lineage"],
                f"{value_text(ln['start'], ln['unit'])} → {value_text(ln['best'], ln['unit'])}",
                str(ln["experiments"]),
                str(ln["kept"]),
                str(ln["failed"]),
                str(ln["since_keep"]),
                f"exp {ln['best_exp']}" if ln["best_exp"] else "—",
            ]
            for ln in s["lineages"]
        ]
        headers = ["lineage", "start → best", "exps", "kept", "failed", "since keep", "best"]
        lines += table(headers, rows, {2, 3, 4, 5})
    top = kept(run, everything)
    if top:
        lines += ["", "kept improvements, ranked by gain"]
        rows = [
            [
                str(e.exp),
                e.lineage,
                f"{value_text(e.best_before, e.unit)} → {value_text(e.value, e.unit)}",
                gain_text(e.gain),
                e.title,
            ]
            for e in top
        ]
        lines += table(["exp", "lineage", "before → after", "gain", "title"], rows, {0, 3})
    return "\n".join(lines)


def _find(items: list[Experiment], exp: int) -> Experiment:
    found = next((e for e in items if e.exp == exp), None)
    if found is None:
        raise SystemExit(f"no exp {exp} in the ledger")
    return found


def render_list(run: RunDir, ns: argparse.Namespace) -> str:
    items = experiments(run)
    if ns.lineage:
        items = [e for e in items if e.lineage == ns.lineage]
    if ns.status:
        failed = ns.status == "failed"
        items = [e for e in items if (e.failed if failed else e.status == ns.status)]
    if ns.kind:
        items = [e for e in items if e.kind == ns.kind]
    if ns.session:
        items = [e for e in items if e.session == ns.session]
    if ns.last:
        items = items[-ns.last :]
    if ns.tsv:
        columns = ["exp", "time", "kind", "lineage", "status", "value", "unit", "best_before"]
        columns += ["gain", "parent_exp", "session", "worker", "idea", "title", "files"]
        out = ["\t".join(columns)]
        for e in items:
            cells = {**e.__dict__, "files": " ".join(e.files)}
            out.append("\t".join(_cell(cells[c]) for c in columns))
        return "\n".join(out)
    if not items:
        return "no experiment matches"
    rows = [
        [
            str(e.exp),
            e.time[5:16],
            e.kind,
            e.lineage,
            e.status,
            value_text(e.value, e.unit),
            gain_text(e.gain),
            e.title,
        ]
        for e in items
    ]
    headers = ["exp", "time", "kind", "lineage", "status", "value", "gain", "title"]
    return "\n".join(table(headers, rows, {0, 5, 6}))


def render_show(run: RunDir, exp: int) -> str:
    everything = all_rows(run)
    e = _find(everything, exp)
    against = f" over {value_text(e.best_before, e.unit)}" if e.gain is not None else ""
    head = f"exp {e.exp} · {e.kind} · {e.lineage} · {e.status}"
    if e.value is not None:
        head += f" · {value_text(e.value, e.unit)} ({gain_text(e.gain)}{against})"
    lines = [head, ""]

    def put(name: str, text: str) -> None:
        if text:
            wrapped = textwrap.wrap(text, 88) or [""]
            lines.append(f"{name:<12}{wrapped[0]}")
            lines.extend(f"{'':<12}{w}" for w in wrapped[1:])

    put("title", e.title)
    put("time", e.time)
    for key in ("session", "worker", "idea", "review", "backend", "snapshot", "parent"):
        put(key, str(e.row.get(key) or ""))
    if e.row.get("early"):
        put("early", "stopped once its verdict was decided")
    put("hypothesis", e.hypothesis)
    rec = record(run, e)
    if rec is None:
        why = "the integration keeps its steps in integration.json"
        put("record", "none" + (f" ({why})" if e.kind != ledger.KERNEL else ""))
    else:
        for name, text in digest(e, rec):
            if name == "error":
                lines.append("error tail")
                lines += [f"    {line}" for line in text.splitlines()]
            else:
                put(name, text)
    lines.append(f"{'files':<12}{e.files[0] if e.files else 'not recorded'}")
    lines += [f"{'':<12}{f}" for f in e.files[1:]]
    parent = next((p for p in everything if p.exp == e.parent_exp), None)
    if parent is None:
        start = "the baseline" if e.lineage == MODEL else "the reference"
        lines.append(f"{'parent':<12}none: {start}")
        return "\n".join(lines)
    lines.append(f"{'parent exp':<12}exp {parent.exp}: {parent.title}")
    if not (e.files and parent.files):
        lines += ["", "no diff: the files of one of them are not recorded"]
    else:
        text = diff(run, e, parent).rstrip("\n")
        lines += ["", f"diff against exp {parent.exp}", text or "(the same code)"]
    return "\n".join(lines)


def render_diff(run: RunDir, a_exp: int, b_exp: int | None) -> tuple[str, int]:
    everything = all_rows(run)
    a = _find(everything, a_exp)
    if b_exp is None:
        b_exp = a.parent_exp
        if b_exp is None:
            start = "the baseline" if a.lineage == MODEL else "the reference"
            return f"exp {a.exp} has no parent experiment (it starts from {start}): give B", 1
    b = _find(everything, b_exp)
    for e in (a, b):
        if not e.files:
            return f"exp {e.exp}: its files are not recorded", 1
    from kernel_agent import expgit  # it imports this module

    text = expgit.diff(run, a.exp, b.exp)  # git diff exp/B exp/A once both are committed
    text = diff(run, a, b) if text is None else text
    return text.rstrip("\n") or f"exp {a.exp} and exp {b.exp}: the same code", 0


# ------------------------------------------------------------------ the CLI


def _run(path: str) -> RunDir:
    run = RunDir(Path(path).resolve())
    if not (run.run_json.exists() or run.ledger.exists()):
        raise SystemExit(f"{run.root} is not a run directory (no run.json, no results.tsv)")
    return run


def _parsers() -> dict[str, argparse.ArgumentParser]:
    """The parsers of ``kernel-agent exp list | show | diff``."""
    out = {}
    p = argparse.ArgumentParser(prog="kernel-agent exp list", description="list experiments")
    p.add_argument("run_dir")
    p.add_argument("--lineage", metavar="ID", help=f"a target id, or {MODEL}")
    p.add_argument("--status", choices=[ledger.KEEP, ledger.DISCARD, "failed", *ledger.FAILURES])
    p.add_argument("--kind", choices=ledger.KINDS)
    p.add_argument("--session", metavar="LABEL", help="an agent session's label")
    p.add_argument("--last", type=int, metavar="N", help="only the last N")
    p.add_argument("--tsv", action="store_true", help="tab-separated, every field")
    out["list"] = p
    p = argparse.ArgumentParser(
        prog="kernel-agent exp show",
        description="one experiment: its row, hypothesis, record, files and diff to its parent",
    )
    p.add_argument("run_dir")
    p.add_argument("exp", type=int)
    out["show"] = p
    p = argparse.ArgumentParser(
        prog="kernel-agent exp diff",
        description="unified diff of what exp B measured to what exp A measured",
    )
    p.add_argument("run_dir")
    p.add_argument("a", type=int, metavar="A")
    p.add_argument("b", type=int, metavar="B", nargs="?", help="default: A's parent experiment")
    out["diff"] = p
    return out


COMMANDS = ("list", "show", "diff")


def add_parser(sub: Any) -> argparse.ArgumentParser:
    """``kernel-agent exp`` (``sub``: the CLI's subparsers)."""
    p: argparse.ArgumentParser = sub.add_parser(
        "exp",
        help="the run's experiments: summary, list, show, diff, git",
        description="The run's experiments (results.tsv rows; read only, safe on a live run):\n"
        "  kernel-agent exp RUN                 headline, lineages, kept improvements by gain\n"
        "  kernel-agent exp list RUN [--lineage ID|model] [--status keep|discard|failed]\n"
        "                       [--kind K] [--session S] [--last N] [--tsv]\n"
        "  kernel-agent exp show RUN N          row, hypothesis, record, files, parent diff\n"
        "  kernel-agent exp diff RUN A [B]      A's code against B (default: A's parent)\n"
        "  kernel-agent exp git RUN [--sync | --rebuild [--out DIR]] [-- GIT ARGS]\n"
        "                                       the experiments.git history (expgit.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("command_or_run", metavar="RUN | list | show | diff | git")
    p.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return p


def main(ns: argparse.Namespace) -> int:
    word, rest = ns.command_or_run, list(ns.args)
    if word == "git":  # the git history of the experiments (#224)
        from kernel_agent import expgit

        return expgit.main(rest)
    if word not in COMMANDS:
        if rest:
            raise SystemExit(f"kernel-agent exp: unexpected {' '.join(rest)!r}")
        print(render_summary(_run(word)))
        return 0
    args = _parsers()[word].parse_args(rest)
    run = _run(args.run_dir)
    if word == "list":
        print(render_list(run, args))
        return 0
    if word == "show":
        print(render_show(run, args.exp))
        return 0
    text, rc = render_diff(run, args.a, args.b)
    print(text)
    return rc
