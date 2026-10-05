"""Run-wide experiment ledger (``results.tsv``) and event log (``events.jsonl``).

Every evaluation of a run (kernel candidates, model-level transforms and the
integration steps) becomes one ledger row, classified like an autoresearch
experiment:

* ``keep``    correct and faster than the best kept result by more than the noise
              margin: ``speedup > best × (1 + max(1 %, 2 × timing_spread))``, the
              same rule as the budget advice (:func:`kernel_agent.budget.improves`);
              kernel targets start from the reference module (1.0×), ``e2e`` rows
              from the baseline run
* ``discard`` correct, but not better
* ``incorrect``, ``build_error``, ``runtime_error``, ``crash``, ``timeout``: failures

``targets/<id>/results.jsonl`` keeps the full records; the TSV is the readable
summary that the charts, ``kernel-agent status`` and ``dashboard.html`` read.
"""

from __future__ import annotations

import math
import re
import statistics
import threading
import time
from pathlib import Path
from typing import Any

from kernel_agent.budget import improves
from kernel_agent.workspace import RunDir, append_jsonl, read_json, read_jsonl

COLUMNS = (
    "exp",
    "time",
    "target",
    "backend",
    "snapshot",
    "parent",
    "status",
    "correct",
    "speedup",
    "ref_ms",
    "new_ms",
    "est_saved_ms",
    "spread",
    "pct_of_sol",
    "eval_s",
    "hypothesis",
)
KEEP = "keep"
DISCARD = "discard"
FAILURES = ("incorrect", "build_error", "runtime_error", "crash", "timeout")
STATUSES = (KEEP, DISCARD, *FAILURES)
E2E = "e2e"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

_FLOATS = {"speedup", "ref_ms", "new_ms", "est_saved_ms", "spread", "pct_of_sol", "eval_s"}
# Statuses of the evaluator / e2e worker that are not ledger statuses.
_STATUS_MAP = {"patch_error": "build_error", "harness_error": "crash", "error": "crash"}
_BACKENDS = (
    ("cute", re.compile(r"^\s*(?:import|from)\s+cutlass\b", re.M)),
    ("tilelang", re.compile(r"^\s*(?:import|from)\s+tilelang\b", re.M)),
    (
        "nvrtc",
        re.compile(
            r"^\s*(?:import|from)\s+cuda\.(?:core|bindings)\b|^\s*from\s+cuda\s+import\s.*\bnvrtc\b",
            re.M,
        ),
    ),
    (
        "cuda",
        re.compile(r"\bload_inline\b|^\s*(?:import|from)\s+torch\.utils\.cpp_extension\b", re.M),
    ),
    ("triton", re.compile(r"^\s*(?:import|from)\s+triton\b", re.M)),
)
_lock = threading.RLock()


# ------------------------------------------------------------------ classification


def detect_backend(source: str) -> str:
    """Kernel backend(s) a candidate uses, from its imports (``torch`` if none)."""
    return "+".join(name for name, pattern in _BACKENDS if pattern.search(source)) or "torch"


def classify(result: dict[str, Any], best: float, *, e2e: bool = False) -> str:
    """Ledger status of one evaluation result, given the best kept speedup so far."""
    status = str(result.get("status") or "crash")
    ok_key = "passed" if e2e else "correct"
    if status != "ok" or not result.get(ok_key):
        if status == "ok":  # measured, but the output does not match
            return "incorrect"
        return status if status in FAILURES else _STATUS_MAP.get(status, "crash")
    if _num(result.get("speedup")) is None:
        return "crash"
    if e2e:  # run-to-run spread of the end-to-end timing
        result = {**result, "timing_spread": e2e_spread(result) or 0.0}
    return KEEP if improves(result, best, ok_key=ok_key) else DISCARD


def kernel_spread(result: dict[str, Any]) -> float | None:
    """Worst timing spread over the captured cases (what the keep rule uses)."""
    spreads = [_num(c.get("timing_spread")) for c in result.get("cases") or []]
    return max((v for v in spreads if v is not None), default=None)


def e2e_spread(result: dict[str, Any]) -> float | None:
    """Relative spread (max − min) / median of the end-to-end timing runs."""
    times = [t for t in (_num(t) for t in result.get("times_ms") or []) if t is not None]
    if len(times) < 2:
        return None
    median = statistics.median(times)
    return round((max(times) - min(times)) / median, 4) if median > 0 else None


def best_kept(rows: list[dict[str, Any]]) -> float:
    """Running best of a target's (or ``e2e``'s) kept rows; 1.0 = reference / baseline."""
    return max((r["speedup"] for r in rows if r["status"] == KEEP and r["speedup"]), default=1.0)


def e2e_backend(transforms: list[Any], kernels: list[str]) -> str:
    return "+".join(k for k, v in (("transform", transforms), ("kernels", kernels)) if v) or "none"


def e2e_snapshot(transforms: list[Any], kernels: list[str]) -> str:
    """``<transform snapshot>+...+<target>...`` of an ``evaluate_e2e`` call."""
    return "+".join([Path(t).name for t in transforms] + [k.partition("=")[0] for k in kernels])


def item_label(item: str) -> str:
    """Short name of an integration item (``target=path`` or a transform path)."""
    target, sep, _ = item.partition("=")
    if sep and "/" not in target:
        return target
    return re.sub(r"^\d+_|_[0-9a-f]{8}$", "", Path(item).stem)


# ------------------------------------------------------------------ the TSV


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}" if math.isfinite(value) else ""
    return " ".join(str(value).split())  # no tabs / newlines inside a cell


def _parse(key: str, cell: str) -> Any:
    if key == "exp":
        return int(cell) if cell.isdigit() else None
    if key == "correct":
        return cell == "true"
    if key in _FLOATS:
        try:
            return float(cell) if cell else None
        except ValueError:
            return None
    return cell


def stamp(when: float | None = None) -> str:
    return time.strftime(TIME_FORMAT, time.localtime(time.time() if when is None else when))


def epoch(text: str | None) -> float | None:
    """Seconds since the epoch of a ``TIME_FORMAT`` local timestamp."""
    if not text:
        return None
    try:
        return time.mktime(time.strptime(text, TIME_FORMAT))
    except ValueError:
        return None


def parse_row(header: list[str], line: str) -> dict[str, Any]:
    """One TSV line as a row dict (missing trailing cells are empty)."""
    cells = line.split("\t")
    cells += [""] * (len(header) - len(cells))
    return {key: _parse(key, cell) for key, cell in zip(header, cells, strict=False)}


def read_tsv(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text().splitlines() if path.exists() else []
    if not lines:
        return []
    header = lines[0].split("\t")
    return [parse_row(header, line) for line in lines[1:] if line.strip()]


def append(run: RunDir, row: dict[str, Any]) -> dict[str, Any]:
    """Append a row (``exp`` is assigned here) and return it."""
    with _lock:
        path = run.ledger
        exists = path.exists() and path.stat().st_size > 0
        count = 0
        columns: tuple[str, ...] = COLUMNS
        if exists:
            with path.open() as fh:
                columns = tuple(fh.readline().rstrip("\n").split("\t"))  # older runs: fewer
                count = sum(1 for line in fh if line.strip())
        row = {**row, "exp": count + 1}
        with path.open("a") as fh:
            if not exists:
                fh.write("\t".join(columns) + "\n")
            fh.write("\t".join(_cell(row.get(c)) for c in columns) + "\n")
    return row


def record_kernel(
    run: RunDir,
    target_id: str,
    result: dict[str, Any],
    *,
    snapshot: str,
    hypothesis: str,
    parent: str | None = None,
    source: str = "",
    eval_s: float | None = None,
    when: float | None = None,
) -> dict[str, Any]:
    """Classify a kernel evaluation against the target's running best and append it."""
    with _lock:
        best = best_kept([r for r in rows(run) if r["target"] == target_id])
        row = append(
            run,
            {
                "time": stamp(when),
                "target": target_id,
                "backend": detect_backend(source),
                "snapshot": Path(snapshot).name,
                "parent": parent or "",
                "status": classify(result, best),
                "correct": bool(result.get("correct")),
                "speedup": _num(result.get("speedup")),
                "ref_ms": _num(result.get("ref_ms_weighted")),
                "new_ms": _num(result.get("new_ms_weighted")),
                "est_saved_ms": _num(result.get("est_saved_ms_per_run")),
                "spread": kernel_spread(result),
                "pct_of_sol": _num(result.get("pct_of_sol")),
                "eval_s": eval_s if eval_s is not None else _num(result.get("eval_seconds")),
                "hypothesis": hypothesis,
            },
        )
    event(run, "evaluation", when=when, target=target_id, exp=row["exp"], status=row["status"])
    return row


def record_e2e(
    run: RunDir,
    result: dict[str, Any],
    *,
    backend: str,
    snapshot: str,
    hypothesis: str,
    parent: str | None = None,
    eval_s: float | None = None,
    when: float | None = None,
) -> dict[str, Any]:
    """Classify an end-to-end measurement (transform or integration step) and append it."""
    with _lock:
        best = best_kept([r for r in rows(run) if r["target"] == E2E])
        status = classify(result, best, e2e=True)
        base, new = _num(result.get("baseline_ms")), _num(result.get("median_ms"))
        row = append(
            run,
            {
                "time": stamp(when),
                "target": E2E,
                "backend": backend,
                "snapshot": snapshot,
                "parent": parent or "",
                "status": status,
                "correct": bool(result.get("passed")),
                "speedup": _num(result.get("speedup")),
                "ref_ms": base,
                "new_ms": new,
                "est_saved_ms": round(base - new, 3)
                if base is not None and new is not None
                else None,
                "spread": e2e_spread(result),
                "eval_s": eval_s,
                "hypothesis": hypothesis,
            },
        )
    event(run, "evaluation", when=when, target=E2E, exp=row["exp"], status=row["status"])
    return row


def rows(run: RunDir) -> list[dict[str, Any]]:
    """All ledger rows of a run; reconstructed from ``results.jsonl`` for older runs."""
    if run.ledger.exists():
        return read_tsv(run.ledger)
    return backfill(run)


def backfill(run: RunDir) -> list[dict[str, Any]]:
    """Ledger rows rebuilt from the per-target / transform ``results.jsonl`` files.

    Older records carry only ``HH:MM:SS``; they are dated from ``run.json``'s
    ``created`` (rolling over midnight), and have no hypothesis.
    """
    created = epoch(run.load().get("created"))
    out: list[dict[str, Any]] = []

    def when(record: dict[str, Any]) -> str:
        if created is None or not re.fullmatch(r"\d\d:\d\d:\d\d", str(record.get("time"))):
            return str(record.get("time") or "")
        day = time.strftime("%Y-%m-%d", time.localtime(created))
        t = epoch(f"{day} {record['time']}") or created
        return stamp(t + 86400 if t < created else t)

    for target_id in run.target_ids():
        kept: list[dict[str, Any]] = []
        for rec in read_jsonl(run.target(target_id) / "results.jsonl"):
            snap = run.target(target_id) / str(rec.get("snapshot", ""))
            row = {
                "time": when(rec),
                "target": target_id,
                "backend": detect_backend(snap.read_text()) if snap.is_file() else "",
                "snapshot": snap.name,
                "parent": rec.get("parent") or "",
                "status": rec.get("ledger_status") or classify(rec, best_kept(kept)),
                "correct": bool(rec.get("correct")),
                "speedup": _num(rec.get("speedup")),
                "ref_ms": _num(rec.get("ref_ms_weighted")),
                "new_ms": _num(rec.get("new_ms_weighted")),
                "est_saved_ms": _num(rec.get("est_saved_ms_per_run")),
                "spread": kernel_spread(rec),
                "pct_of_sol": _num(rec.get("pct_of_sol")),
                "eval_s": _num(rec.get("eval_seconds")),
                "hypothesis": rec.get("hypothesis") or "",
            }
            kept.append(row)
            out.append(row)
    kept = []
    for rec in read_jsonl(run.transforms_dir / "results.jsonl"):
        base, new = _num(rec.get("baseline_ms")), _num(rec.get("median_ms"))
        row = {
            "time": when(rec),
            "target": E2E,
            "backend": e2e_backend(rec.get("transforms") or [], rec.get("kernels") or []),
            "snapshot": e2e_snapshot(rec.get("transforms") or [], rec.get("kernels") or []),
            "parent": "",
            "status": rec.get("ledger_status") or classify(rec, best_kept(kept), e2e=True),
            "correct": bool(rec.get("passed")),
            "speedup": _num(rec.get("speedup")),
            "ref_ms": base,
            "new_ms": new,
            "est_saved_ms": round(base - new, 3) if base is not None and new is not None else None,
            "spread": e2e_spread(rec),
            "eval_s": None,
            "hypothesis": rec.get("hypothesis") or "",
        }
        kept.append(row)
        out.append(row)
    out.sort(key=lambda r: r["time"])
    return [{**r, "exp": i} for i, r in enumerate(out, 1)]


# ------------------------------------------------------------------ events


def event(run: RunDir, kind: str, *, when: float | None = None, **data: Any) -> None:
    """Append to ``events.jsonl`` (phase changes, agent start/stop, evaluations)."""
    ts = time.time() if when is None else when
    append_jsonl(run.events, {"ts": round(ts, 3), "time": stamp(ts), "event": kind, **data})


def events(run: RunDir) -> list[dict[str, Any]]:
    return read_jsonl(run.events)


def phase_spans(
    run: RunDir, log: list[dict[str, Any]] | None = None
) -> list[tuple[str, float, float | None]]:
    """``(phase, start_ts, end_ts or None while running)`` from the event log (or ``log``)."""
    spans: list[tuple[str, float, float | None]] = []
    for ev in events(run) if log is None else log:
        if "ts" not in ev:
            continue
        if ev.get("event") == "phase_start":
            spans.append((str(ev.get("phase")), float(ev["ts"]), None))
        elif ev.get("event") in ("phase_done", "phase_failed"):
            for i in range(len(spans) - 1, -1, -1):
                if spans[i][0] == ev.get("phase") and spans[i][2] is None:
                    spans[i] = (spans[i][0], spans[i][1], float(ev["ts"]))
                    break
    return spans


# ------------------------------------------------------------------ run summary


def phase_label(summary: dict[str, Any]) -> str:
    if summary.get("phase_failed"):
        return f"{summary['phase_failed']} (failed)"
    phase = summary.get("phase") or "not started"
    return phase + (" (running)" if summary.get("phase_running") else "")


def start_time(run: RunDir, ledger_rows: list[dict[str, Any]] | None = None) -> float | None:
    """When the run started: ``run.json`` ``created``, else the first event / row."""
    created = epoch(run.load().get("created")) if run.run_json.exists() else None
    if created is not None:
        return created
    candidates = [float(e["ts"]) for e in events(run) if "ts" in e][:1]
    candidates += [t for t in (epoch(r["time"]) for r in (ledger_rows or [])[:1]) if t is not None]
    return min(candidates, default=None)


def summary(run: RunDir) -> dict[str, Any]:
    """Everything ``status``, the dashboard and the report show about a run."""
    data = run.load() if run.run_json.exists() else {}
    ledger_rows = rows(run)
    baseline = read_json(run.baseline_json, {}) or {}
    base_ms = _num(baseline.get("median_ms"))
    targets = []
    ids = list(run.target_ids())
    ids += sorted({r["target"] for r in ledger_rows} - {E2E} - set(ids))
    for target_id in ids:
        spec = read_json(run.target(target_id) / "spec.json", {}) or {}
        trows = [r for r in ledger_rows if r["target"] == target_id]
        kept = [r for r in trows if r["status"] == KEEP]
        best = max(kept, key=lambda r: r["speedup"] or 0.0, default=None)
        targets.append(
            {
                "id": target_id,
                "module_class": spec.get("module_class"),
                "backends": spec.get("backends", []),
                "evals": len(trows),
                "keeps": len(kept),
                "failures": sum(r["status"] in FAILURES for r in trows),
                "best_speedup": best["speedup"] if best else None,
                "best_pct_of_sol": best.get("pct_of_sol") if best else None,
                "best_snapshot": best["snapshot"] if best else None,
                "best_backend": best["backend"] if best else None,
                "est_saved_ms": best["est_saved_ms"] if best else None,
                "last_hypothesis": trows[-1]["hypothesis"] if trows else "",
            }
        )
    saved = sum(t["est_saved_ms"] or 0.0 for t in targets)
    kept_e2e = [r for r in ledger_rows if r["target"] == E2E and r["status"] == KEEP]
    best_e2e = kept_e2e[-1] if kept_e2e else None  # running best end-to-end measurement
    final = (read_json(run.root / "integration.json", {}) or {}).get("final") or {}
    costs = read_json(run.root / "costs.json", {}) or {}
    phases = data.get("phases", {})
    spans = phase_spans(run)
    running = next((p for p, _, end in reversed(spans) if end is None), None)
    done = [p for p, info in phases.items() if isinstance(info, dict) and info.get("done")]
    log = events(run)
    failed: dict[str, Any] = next(
        (e for e in reversed(log) if e.get("event") in ("phase_failed", "phase_start")), {}
    )
    start = start_time(run, ledger_rows)
    last = max(
        [float(e["ts"]) for e in log if "ts" in e][-1:]
        + [t for t in (epoch(r["time"]) for r in ledger_rows[-1:]) if t is not None],
        default=None,
    )
    return {
        "repo_id": (data.get("card") or {}).get("repo_id", run.root.name),
        "modality": (data.get("card") or {}).get("modality"),
        "root": str(run.root),
        "phase": running or (done[-1] if done else None),
        "phase_running": running is not None,
        "phase_failed": failed.get("phase") if failed.get("event") == "phase_failed" else None,
        "elapsed_min": round((last - start) / 60, 1) if start and last else None,
        "baseline_ms": base_ms,
        "workload": baseline.get("workload"),
        "projected_ms": round(max(base_ms - saved, 0.0), 3) if base_ms else None,
        "best_e2e": best_e2e,
        "final": final or None,
        "evaluations": len(ledger_rows),
        "keeps": sum(r["status"] == KEEP for r in ledger_rows),
        "failures": sum(r["status"] in FAILURES for r in ledger_rows),
        "targets": targets,
        "e2e_evals": sum(r["target"] == E2E for r in ledger_rows),
        "cost_usd": round(sum(float(c.get("usd") or 0.0) for c in costs.values()), 2),
        "costs": costs,
        "rows": ledger_rows,
    }
