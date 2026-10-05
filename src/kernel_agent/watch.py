"""``kernel-agent watch``: live dashboard of a run (stdlib HTTP + Server-Sent Events).

``kernel-agent watch <run_dir>`` serves one page on ``http://127.0.0.1:8765``.
Start it before, during or after a run: it only reads files, so the
optimisation process never notices it.

* ``results.tsv`` (ledger rows), ``events.jsonl`` and ``logs/agent-*.jsonl`` are
  tailed by byte offset; a partly written last line is left for the next poll.
* ``run.json``, ``baseline.json``, ``toolchain.json``, ``costs.json``,
  ``integration.json``, the profile and the target list are re-read when their
  size or mtime changes.

Endpoints:

* ``/``            the page (``watch.html``: inline JS, CSS and SVG, no CDN) with
                   the current state embedded, so it renders before the stream connects
* ``/api/state``   JSON snapshot: run header, baselines (eager + compiled),
                   per-target summary, ledger rows, events and agent-log tails,
                   costs, integration
* ``/api/files``   the run's text files, for the file browser
* ``/events``      SSE: ``state`` on connect, then ``delta`` (new rows, events,
                   agent-log lines and the recomputed summary) after a poll that
                   found something new; ``: ping`` keep-alives in between
* ``/file?path=``  read-only text viewer, confined to the run directory

Only loopback ``Host`` headers are accepted unless ``--host`` binds a wildcard
address, so a web page cannot read the run through DNS rebinding.
"""

from __future__ import annotations

import json
import math
import secrets
import socket
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from kernel_agent import charts, ledger, projection
from kernel_agent.ledger import E2E, FAILURES, KEEP
from kernel_agent.workspace import TRUTH_DIR, RunDir

PAGE = Path(__file__).with_name("watch.html")
POLL_S = 1.0  # how often an SSE connection looks for new data
PING_S = 15.0  # keep-alive comment when nothing changed for this long
EVENTS_TAIL = 300  # events sent with a snapshot
LOG_TAIL = 200  # agent-log lines sent with a snapshot
LOG_BYTES = 256 * 1024  # read this much from the end of each agent log on connect
MAX_FILE_BYTES = 512 * 1024  # /file returns at most this much of a file
MAX_FILES = 4000  # /api/files lists at most this many files
TEXT_SUFFIXES = frozenset(
    {".py", ".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp", ".diff", ".patch"}
    | {".md", ".txt", ".log", ".json", ".jsonl", ".tsv", ".csv", ".toml", ".yaml", ".yml"}
)
LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})
WILDCARD = frozenset({"", "0.0.0.0", "::"})


# ------------------------------------------------------------------ tailing


class Tail:
    """Complete lines appended to a file since the previous :meth:`read`.

    Keeps a byte offset; an unfinished last line (no newline yet) stays for the
    next read. ``from_end`` starts that many bytes before the end (skipping the
    first, partial line). A file that shrank or was replaced is read again from
    the start and reported as reset.
    """

    def __init__(self, path: Path, from_end: int | None = None) -> None:
        self.path = path
        self.from_end = from_end
        self.pos = 0
        self.inode: int | None = None
        self.skip = False  # drop the line the offset points into

    def read(self) -> tuple[list[str], bool]:
        """``(new complete lines, reset)``."""
        try:
            st = self.path.stat()
        except OSError:
            reset = self.inode is not None
            self.pos, self.inode, self.skip = 0, None, False
            return [], reset
        reset = self.inode is not None and (st.st_ino != self.inode or st.st_size < self.pos)
        if reset or self.inode is None:
            self.pos = max(st.st_size - self.from_end, 0) if self.from_end else 0
            self.skip = self.pos > 0
        self.inode = st.st_ino
        if st.st_size <= self.pos:
            return [], reset
        with self.path.open("rb") as fh:
            fh.seek(self.pos)
            data = fh.read(st.st_size - self.pos)
        end = data.rfind(b"\n")
        if end < 0:
            return [], reset
        self.pos += end + 1
        lines = data[:end].split(b"\n")
        if self.skip:
            lines, self.skip = lines[1:], False
        return [line.decode("utf-8", "replace") for line in lines], reset


def _json(path: Path) -> dict[str, Any]:
    """The JSON object in ``path``; {} when it is missing, being written or not an object."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _obj(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _warn(exc: BaseException) -> None:
    print(f"[kernel-agent watch] {exc!r}", file=sys.stderr, flush=True)


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _clean(value: Any) -> Any:
    """``value`` with non-finite floats as None (``JSON.parse`` rejects NaN)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    return value


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _brief(data: Any, limit: int = 140) -> str:
    if isinstance(data, dict):
        for key in ("candidate", "file_path", "command", "pattern", "url", "transforms", "path"):
            if key in data:
                return f"{key}={' '.join(str(data[key]).split())[:limit]}"
    return " ".join(str(data).split())[:limit]


def agent_entries(agent: str, line: str) -> list[dict[str, Any]]:
    """Short activity lines (tool calls, text, the result) of one agent-log message."""
    if '"AssistantMessage"' not in line[:48] and '"ResultMessage"' not in line[:48]:
        return []  # tool results, system messages: large and not shown
    try:
        message = json.loads(line)
    except ValueError:
        return []
    data = message.get("data") if isinstance(message, dict) else None
    if not isinstance(data, dict):
        return []
    out: list[dict[str, Any]] = []
    if message.get("type") == "AssistantMessage":
        for block in data.get("content") or []:
            if not isinstance(block, dict):
                continue
            if isinstance(block.get("name"), str) and "input" in block:
                name = block["name"].removeprefix("mcp__ka__")
                out.append(
                    {"agent": agent, "kind": "tool", "text": f"{name} {_brief(block['input'])}"}
                )
            elif isinstance(block.get("text"), str) and block["text"].strip():
                text = " ".join(block["text"].split())
                text = text if len(text) <= 300 else text[:299] + "…"
                out.append({"agent": agent, "kind": "text", "text": text})
    else:
        text = "finished" + (" with an error" if data.get("is_error") else "")
        if isinstance(data.get("num_turns"), int):
            text += f", {data['num_turns']} turns"
        if _num(data.get("total_cost_usd")) is not None:
            text += f", ${data['total_cost_usd']:.2f}"
        out.append({"agent": agent, "kind": "result", "text": text})
    return out


# ------------------------------------------------------------------ state


class Watcher:
    """Follows one run directory: :meth:`snapshot` once, then :meth:`poll` for deltas."""

    def __init__(self, run: RunDir) -> None:
        self.run = run
        self._reset()

    def _reset(self) -> None:
        self.ledger = Tail(self.run.ledger)
        self.events_tail = Tail(self.run.events)
        self.logs: dict[Path, Tail] = {}
        self.header: list[str] | None = None
        self.rows: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.log_lines: list[dict[str, Any]] = []
        self.backfilled = False
        self.signature: Any = None
        self.files: dict[str, Any] = {}
        self.activity: float | None = None

    # -------------------------------------------------------------- reading

    def _read_rows(self) -> tuple[list[dict[str, Any]], bool]:
        lines, reset = self.ledger.read()
        if reset:
            self.header = None
        out = []
        for line in lines:
            if self.header is None:
                self.header = line.split("\t")
            elif line.strip():
                out.append(self._row(ledger.parse_row(self.header, line)))
        return out, reset

    def _row(self, row: dict[str, Any]) -> dict[str, Any]:
        """A ledger row for the page: epoch ``ts`` and the snapshot ``file`` if it exists."""
        row = _clean(row)
        row["ts"] = ledger.epoch(row.get("time"))
        snapshot = str(row.get("snapshot") or "")
        rel = None
        if row.get("target") != E2E and snapshot:
            rel = f"targets/{row['target']}/history/{snapshot}"
        elif snapshot.split("+")[0].endswith(".py"):
            rel = f"transforms/history/{snapshot.split('+')[0]}"
        row["file"] = None
        # the evaluated snapshot in .truth/ when the run has one, else the run's own copy
        for path in (f"{TRUTH_DIR}/{rel}", rel) if rel else ():
            try:
                viewable(self.run.root, path)
            except FileError:
                continue
            row["file"] = path
            break
        return row

    def _read_events(self) -> tuple[list[dict[str, Any]], bool]:
        lines, reset = self.events_tail.read()
        out = []
        for line in lines:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                out.append(_clean(record))
        return out, reset

    def _read_logs(self, live: bool) -> list[dict[str, Any]]:
        log_dir = self.run.root / "logs"
        paths = list(log_dir.glob("agent-*.jsonl")) if log_dir.is_dir() else []
        paths.sort(key=lambda p: (_mtime(p) or 0.0, p.name))  # oldest first
        now = round(time.time(), 3)
        out = []
        for path in paths:
            tail = self.logs.setdefault(path, Tail(path, from_end=LOG_BYTES))
            lines, _ = tail.read()
            agent = path.stem.removeprefix("agent-")
            for line in lines:
                for entry in agent_entries(agent, line):
                    out.append({**entry, "ts": now} if live else entry)
        return out

    def _refresh_files(self) -> bool:
        """Re-read the small JSON files when one changed; True if something did."""
        run = self.run
        paths = [
            run.run_json,
            run.baseline_json,
            run.toolchain_json,
            run.root / "costs.json",
            run.root / "integration.json",
            run.profile_dir / "profile.json",
        ]
        ids = run.target_ids()
        specs = [run.target(t) / "spec.json" for t in ids]
        signature = (tuple(ids), tuple(_stat(p) for p in [*paths, *specs]))
        if signature == self.signature:
            return False
        self.signature = signature
        try:
            shares = charts.target_shares(run)
        except Exception:  # a profile or spec in an unexpected shape: no shares
            shares = []
        try:
            tree = projection.tree(run, ids)
        except Exception:  # the same: every target counts in full
            tree = projection.Tree()
        self.files = {
            "run": _json(run.run_json),
            "baseline": _json(run.baseline_json),
            "toolchain": _json(run.toolchain_json),
            "costs": _json(run.root / "costs.json"),
            "integration": _json(run.root / "integration.json"),
            "targets": ids,
            "specs": {t: _json(run.target(t) / "spec.json") for t in ids},
            "shares": shares,
            "tree": tree,
        }
        return True

    # -------------------------------------------------------------- public

    def snapshot(self) -> dict[str, Any]:
        """Everything the page shows, read from scratch."""
        self._reset()
        self.rows, _ = self._read_rows()
        if not self.run.ledger.exists():  # a run that predates the ledger
            try:
                self.rows = [self._row(r) for r in ledger.backfill(self.run)]
            except (OSError, ValueError, KeyError, TypeError):
                self.rows = []
            self.backfilled = True
        self.events, _ = self._read_events()
        self.log_lines = self._read_logs(live=False)[-LOG_TAIL:]
        self._refresh_files()
        log_dir = self.run.root / "logs"
        watched = [self.run.ledger, self.run.events, self.run.root / "costs.json"]
        watched += list(log_dir.glob("agent-*.jsonl")) if log_dir.is_dir() else []
        self.activity = max((t for t in map(_mtime, watched) if t is not None), default=None)
        return {
            "summary": self.summary(),
            "rows": self.rows,
            "events": self.events[-EVENTS_TAIL:],
            "logs": self.log_lines,
        }

    def poll(self) -> tuple[str, dict[str, Any]] | None:
        """``("delta", new data + summary)``, ``("state", snapshot)`` after a reset, or None."""
        rows, rows_reset = self._read_rows()
        events, events_reset = self._read_events()
        if rows_reset or events_reset or (rows and self.backfilled):
            return "state", self.snapshot()
        logs = self._read_logs(live=True)
        changed = self._refresh_files()
        if not (rows or events or logs or changed):
            return None
        self.rows += rows
        self.events += events
        self.log_lines = (self.log_lines + logs)[-LOG_TAIL:]
        if rows or events or logs:
            self.activity = time.time()
        delta: dict[str, Any] = {"summary": self.summary()}
        if rows:
            delta["rows"] = rows
        if events:
            delta["events"] = events[-EVENTS_TAIL:]
        if logs:
            delta["logs"] = logs[-LOG_TAIL:]
        return "delta", delta

    def summary(self) -> dict[str, Any]:
        """Run header, baselines, targets, costs and integration (everything but the lists)."""
        now = time.time()
        files, rows, log = self.files, self.rows, self.events
        data = files.get("run") or {}
        card, config = _obj(data.get("card")), _obj(data.get("config"))
        baseline = files.get("baseline") or {}
        gpu = _obj(_obj(files.get("toolchain")).get("gpu"))
        base_ms = _num(baseline.get("median_ms"))

        spans = ledger.phase_spans(self.run, log)
        running = next((p for p, _, end in reversed(spans) if end is None), None)
        done = [
            p for p, i in _obj(data.get("phases")).items() if isinstance(i, dict) and i.get("done")
        ]
        last_phase: dict[str, Any] = next(
            (e for e in reversed(log) if e.get("event") in ("phase_failed", "phase_start")), {}
        )
        failed = last_phase.get("phase") if last_phase.get("event") == "phase_failed" else None
        active: dict[str, float | None] = {}
        for e in log:
            if e.get("event") == "agent_start":
                active[str(e.get("agent"))] = _num(e.get("ts"))
            elif e.get("event") == "agent_done":
                active.pop(str(e.get("agent")), None)
        stamps = [t for t in (_num(e.get("ts")) for e in log) if t is not None]
        stamps += [r["ts"] for r in rows if r.get("ts")]
        start = ledger.epoch(data.get("created")) or min(stamps, default=None)
        last = max(stamps, default=None)
        end = now if running else last

        targets = self._targets()
        # nested targets counted once (projection.py), after every kept kernel for the chart
        tree = files.get("tree") or projection.Tree()
        proj = projection.project(tree, {t["id"]: t["est_saved_ms"] for t in targets}, base_ms or 0)
        projected = projection.series(tree, base_ms, rows) if base_ms else []
        e2e = [r for r in rows if r["target"] == E2E]
        kept_e2e = [r for r in e2e if r["status"] == KEEP]
        integration = files.get("integration") or {}
        final = _obj(integration.get("final"))
        measured = None
        if base_ms and final.get("passed") and _num(final.get("median_ms")):
            ms = float(final["median_ms"])
            measured = {"ms": ms, "speedup": base_ms / ms, "label": "integrated", "exp": None}
        elif base_ms and kept_e2e and kept_e2e[-1]["new_ms"]:
            best = kept_e2e[-1]
            measured = {
                "ms": best["new_ms"],
                "speedup": base_ms / best["new_ms"],
                "label": best["snapshot"] or best["hypothesis"],
                "exp": best["exp"],
            }

        costs = {k: v for k, v in (files.get("costs") or {}).items() if isinstance(v, dict)}
        burn, total = [], 0.0
        for e in log:
            if e.get("event") == "agent_done" and _num(e.get("ts")) is not None:
                usd = _num(e.get("usd")) or 0.0
                total += usd
                burn.append(
                    {"ts": e["ts"], "usd": round(total, 4), "add": usd, "agent": e.get("agent")}
                )

        try:
            steps = charts.integration_steps(integration) if integration else []
        except Exception:  # an integration.json in an unexpected shape
            steps = []
        return _clean(
            {
                "now": round(now, 3),
                "run": {
                    "repo_id": card.get("repo_id", self.run.root.name),
                    "modality": card.get("modality"),
                    "root": str(self.run.root),
                    "path": _shown(self.run.root),
                    "name": self.run.root.name,
                    "gpu": gpu.get("name"),
                    "arch": gpu.get("arch"),
                    "workload": baseline.get("workload"),
                    "claude_model": config.get("claude_model"),
                    "parallel": config.get("parallel"),
                    "max_hours": _num(config.get("max_hours")),
                    "max_usd": _num(config.get("max_usd")),
                    "start_ts": start,
                    "last_ts": last,
                    "elapsed_s": round(end - start, 1) if start and end else None,
                    "idle_s": round(now - self.activity, 1) if self.activity else None,
                    "running": running is not None,
                    "phase": running or (done[-1] if done else None),
                    "phase_failed": failed,
                    "active_agents": [
                        {"agent": a, "since": ts} for a, ts in active.items() if running
                    ],
                },
                "phases": [{"phase": p, "start": a, "end": b} for p, a, b in spans],
                "baseline": {
                    "eager_ms": base_ms,
                    "compiled_ms": charts._compiled_ms(baseline),
                    "times_ms": baseline.get("times_ms"),
                },
                "projected_ms": proj.projected_ms if base_ms else None,
                "projection": {
                    **proj.as_dict(),
                    "steps": {str(r["exp"]): p.projected_ms for r, p in projected},
                }
                if base_ms
                else None,
                "measured": measured,
                "counts": {
                    "evaluations": len(ledger.measured(rows)),
                    "keeps": sum(r["status"] == KEEP for r in rows),
                    "failures": sum(r["status"] in FAILURES for r in rows),
                    "e2e": len(e2e),
                    "e2e_keeps": len(kept_e2e),
                    "e2e_failures": sum(r["status"] in FAILURES for r in e2e),
                },
                "targets": targets,
                "costs": {
                    "total_usd": round(sum(_num(c.get("usd")) or 0.0 for c in costs.values()), 4),
                    "agents": costs,
                    "burn": burn,
                },
                "integration": {
                    "steps": steps,
                    "final": final or None,
                    "accepted": [
                        ledger.item_label(str(a.get("item", "")))
                        for a in integration.get("accepted") or []
                        if isinstance(a, dict)
                    ],
                }
                if steps
                else None,
            }
        )

    def _targets(self) -> list[dict[str, Any]]:
        files, rows = self.files, self.rows
        ids = list(files.get("targets") or [])
        ids += sorted({r["target"] for r in rows} - {E2E} - set(ids))
        order = [t for t, _, _ in files.get("shares") or []]
        shares = {t: share for t, _, share in files.get("shares") or []}
        out = []
        for target_id in ids:
            spec = (files.get("specs") or {}).get(target_id) or {}
            trows = ledger.measured(r for r in rows if r["target"] == target_id)
            kept = [r for r in trows if r["status"] == KEEP]
            best = max(kept, key=lambda r: r["speedup"] or 0.0, default=None)
            out.append(
                {
                    "id": target_id,
                    "module_class": spec.get("module_class"),
                    "backends": spec.get("backends") or [],
                    "share": shares.get(target_id),
                    "color": order.index(target_id) if target_id in order else None,
                    "evals": len(trows),
                    "keeps": len(kept),
                    "failures": sum(r["status"] in FAILURES for r in trows),
                    "best_speedup": best["speedup"] if best else None,
                    "best_snapshot": best["snapshot"] if best else None,
                    "best_backend": best["backend"] if best else None,
                    "est_saved_ms": best["est_saved_ms"] if best else None,
                    "last_hypothesis": trows[-1]["hypothesis"] if trows else "",
                }
            )
        return out


def _shown(path: Path) -> str:
    """``path`` relative to the working directory when it is inside it (as typed), else absolute."""
    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _stat(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def state(run: RunDir) -> dict[str, Any]:
    """``/api/state``: a snapshot of the run."""
    return Watcher(run).snapshot()


def sse(event: str, data: Any) -> bytes:
    """One Server-Sent Events message (``data`` as single-line JSON)."""
    body = json.dumps(data, separators=(",", ":"), default=str)
    return f"event: {event}\ndata: {body}\n\n".encode()


# ------------------------------------------------------------------ files


class FileError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


def viewable(root: Path, rel: str) -> Path:
    """The text file ``rel`` names inside the run directory ``root`` (resolved).

    Raises :class:`FileError` for absolute paths, ``..``, symlinks that leave
    ``root``, directories, missing files and non-text suffixes.
    """
    if not rel or "\x00" in rel or "\\" in rel:
        raise FileError(HTTPStatus.BAD_REQUEST, "bad path")
    path = Path(rel)
    if path.is_absolute() or ".." in path.parts:
        raise FileError(HTTPStatus.FORBIDDEN, "outside the run directory")
    root = root.resolve()
    try:
        target = (root / path).resolve(strict=True)
    except (OSError, RuntimeError):
        raise FileError(HTTPStatus.NOT_FOUND, "no such file") from None
    if not target.is_relative_to(root):
        raise FileError(HTTPStatus.FORBIDDEN, "outside the run directory")
    if not target.is_file():
        raise FileError(HTTPStatus.NOT_FOUND, "not a file")
    if target.suffix.lower() not in TEXT_SUFFIXES:
        raise FileError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "not a text file")
    return target


def read_text(path: Path, limit: int = MAX_FILE_BYTES) -> tuple[str, int, bool]:
    """``(text, size in bytes, truncated)``; raises :class:`FileError` for binary data."""
    with path.open("rb") as fh:
        data = fh.read(limit + 1)
        size = max(fh.seek(0, 2), len(data))
    if b"\x00" in data[:8192]:
        raise FileError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "binary file")
    return data[:limit].decode("utf-8", "replace"), size, len(data) > limit


def list_files(root: Path) -> list[dict[str, Any]]:
    """Text files of the run (no hidden files), for the file browser."""
    root = root.resolve()
    out: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if path.suffix.lower() not in TEXT_SUFFIXES or any(p.startswith(".") for p in rel.parts):
            continue
        try:
            if not path.is_file() or not path.resolve().is_relative_to(root):
                continue
            st = path.stat()
        except OSError:
            continue
        out.append({"path": rel.as_posix(), "size": st.st_size, "mtime": round(st.st_mtime, 3)})
        if len(out) >= MAX_FILES:
            break
    return out


# ------------------------------------------------------------------ server


def page(run: RunDir, nonce: str) -> bytes:
    """``watch.html`` with the current state embedded (``<`` escaped for the script tag)."""
    blob = json.dumps(state(run), separators=(",", ":"), default=str)
    blob = blob.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    html = PAGE.read_text().replace("__NONCE__", nonce).replace("__STATE__", blob)
    return html.encode()


class Handler(BaseHTTPRequestHandler):
    server: WatchServer
    server_version = "kernel-agent-watch"

    def log_message(self, format: str, *args: Any) -> None:
        pass  # quiet: the terminal shows the run, not the requests

    def _send(
        self,
        status: HTTPStatus,
        body: bytes,
        ctype: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: Any) -> None:
        body = json.dumps(data, separators=(",", ":"), default=str).encode()
        self._send(HTTPStatus.OK, body, "application/json")

    def _text(self, status: HTTPStatus, message: str) -> None:
        self._send(status, message.encode(), "text/plain; charset=utf-8")

    def _host_ok(self) -> bool:
        allowed = self.server.allowed_hosts
        host = (self.headers.get("Host") or "").strip().lower()
        if allowed is None or not host:
            return True
        if host.startswith("["):  # [::1]:8765
            name = host[1 : host.find("]")]
        else:
            name = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
        return name in allowed

    def do_GET(self) -> None:
        if not self._host_ok():
            return self._text(HTTPStatus.FORBIDDEN, "unexpected Host header")
        url = urlsplit(self.path)
        run = self.server.run
        try:
            if url.path in ("/", "/index.html"):
                nonce = secrets.token_urlsafe(16)
                csp = (
                    f"default-src 'none'; script-src 'nonce-{nonce}'; "
                    "style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; "
                    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
                )
                self._send(
                    HTTPStatus.OK,
                    page(run, nonce),
                    "text/html; charset=utf-8",
                    {"Content-Security-Policy": csp},
                )
            elif url.path == "/api/state":
                self._json(state(run))
            elif url.path == "/api/files":
                self._json(list_files(run.root))
            elif url.path == "/file":
                self._file((parse_qs(url.query).get("path") or [""])[0])
            elif url.path == "/events":
                self._stream()
            else:
                self._text(HTTPStatus.NOT_FOUND, "not found")
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser went away; EventSource reconnects on its own
        except Exception as exc:  # before any response was sent (streams handle their own)
            _warn(exc)
            self._text(HTTPStatus.INTERNAL_SERVER_ERROR, repr(exc))

    def _file(self, rel: str) -> None:
        try:
            path = viewable(self.server.run.root, rel)
            text, size, truncated = read_text(path)
        except FileError as exc:
            return self._text(exc.status, str(exc))
        self._send(
            HTTPStatus.OK,
            text.encode(),
            "text/plain; charset=utf-8",
            {"X-File-Size": str(size), "X-Truncated": "1" if truncated else "0"},
        )

    def _stream(self) -> None:
        watcher = Watcher(self.server.run)
        first = sse("state", watcher.snapshot())
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.wfile.write(b"retry: 2000\n" + first)
        self.wfile.flush()
        quiet, resync = 0.0, False
        stopping, interval = self.server.stopping, self.server.interval
        while not stopping.wait(interval):
            try:
                message = ("state", watcher.snapshot()) if resync else watcher.poll()
                resync = False
            except Exception as exc:  # e.g. a file replaced mid-read: start over next time
                _warn(exc)
                message, resync = None, True
            if message is not None:
                self.wfile.write(sse(*message))
                quiet = 0.0
            else:
                quiet += interval
                if quiet < PING_S:
                    continue
                self.wfile.write(b": ping\n\n")
                quiet = 0.0
            self.wfile.flush()


class WatchServer(ThreadingHTTPServer):
    """``ThreadingHTTPServer`` for one run; :meth:`shutdown` also ends the SSE streams."""

    daemon_threads = True

    def __init__(
        self,
        run: RunDir,
        host: str = "127.0.0.1",
        port: int = 8765,
        interval: float = POLL_S,
    ) -> None:
        self.run = RunDir(run.root.resolve())
        self.interval = interval
        self.stopping = threading.Event()
        self.allowed_hosts = None if host in WILDCARD else LOOPBACK | {host.lower()}
        if ":" in host:
            self.address_family = socket.AF_INET6
        super().__init__((host, port), Handler)

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        host = str(host)
        if host in WILDCARD:
            host = "127.0.0.1"
        return f"http://[{host}]:{port}/" if ":" in host else f"http://{host}:{port}/"

    def shutdown(self) -> None:
        self.stopping.set()
        super().shutdown()


def serve(run_dir: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve the dashboard of ``run_dir`` until Ctrl-C."""
    server = WatchServer(RunDir(Path(run_dir).resolve()), host, port)
    print(f"kernel-agent watch: {server.url}  ({server.run.root})", flush=True)
    if host not in LOOPBACK:
        print(
            f"warning: listening on {host}; anyone who can reach this port can read the run",
            file=sys.stderr,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.stopping.set()
        server.server_close()
