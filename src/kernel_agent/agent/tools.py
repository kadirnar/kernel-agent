"""In-process MCP tools exposed to the agents (Claude Agent SDK)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from kernel_agent import board, dashboard, dedup, gpuqueue, ledger, region, truth, workers
from kernel_agent.budget import Budget
from kernel_agent.kernels import sweep as sweep_mod
from kernel_agent.kernels.evaluate import run_evaluation
from kernel_agent.kernels.roofline import sol_signal
from kernel_agent.native import engine as native_engine
from kernel_agent.native import project as native_project
from kernel_agent.truth import TamperError, Truth, sha256_file
from kernel_agent.worker import call_worker
from kernel_agent.workspace import RunDir, append_jsonl, read_json, write_json

SERVER_NAME = "ka"
QUICK_NOTE = (
    "quick check: correctness only, on the smallest and the largest captured case. NOT a "
    "benchmark: no timing, no speedup, never a new best, not counted against your evaluation "
    'budget. Evaluate with mode="full" to measure it.'
)
# (run, target, source key) of evaluations in flight: the same source waits for the first one
_inflight: dict[tuple[str, str, str], asyncio.Event] = {}


@dataclass(frozen=True)
class SessionBinding:
    """What the tools of one agent session are bound to: each session gets its own MCP
    server (:func:`build_server`), so nothing a tool decides depends on which session ran
    last (the run-wide ``Budget`` holds no session's evaluation budget).

    ``label``: the session (its ``costs.json`` key), stamped on its ledger rows and records
    (``session``); ``role``: kernel, systems, native, ...; ``target_id`` / ``worker``: the
    target and worker (island) of a kernel session (``workers.py``: a worker's directory and
    the ``worker`` of its rows, for its own target); ``agent``: the name every evaluation's
    budget advice is kept under (the native session's; a worker's, for its own target;
    None: ``kernel-<target>`` for kernel evaluations, ``systems`` for ``evaluate_e2e``);
    ``evaluations``: its evaluation budget (None: none); ``cwd``: its working directory,
    where relative candidate, transform and project paths are looked up first."""

    label: str = ""
    role: str = ""
    target_id: str | None = None
    worker: int | None = None
    agent: str | None = None
    evaluations: int | None = None
    cwd: Path | None = None

    def mine(self, target_id: str) -> bool:
        """Whether ``target_id`` is this worker session's own target."""
        return self.worker is not None and self.target_id == target_id

    def kernel_agent(self, target_id: str) -> str:
        """The name a kernel evaluation of ``target_id``'s budget advice is kept under."""
        if self.agent and (self.worker is None or self.mine(target_id)):
            return self.agent
        return f"kernel-{target_id}"

    def e2e_agent(self) -> str:
        """The name an ``evaluate_e2e`` result's budget advice is kept under."""
        return self.agent if self.agent and self.worker is None else "systems"


def refresh(run: RunDir, target_id: str | None = None) -> None:
    """Charts and ``dashboard.html`` after an evaluation: asked of the run's refresher thread
    (``dashboard.Refresher``: debounced, never waited for)."""
    dashboard.refresher(run).request(target_id)


def _text(data: Any) -> dict[str, Any]:
    body = data if isinstance(data, str) else json.dumps(data, indent=1, default=str)
    if len(body) > 24000:
        body = body[:24000] + "\n... (truncated)"
    return {"content": [{"type": "text", "text": body}]}


def _resolve(base: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base / p)


_history_locks: dict[Path, threading.Lock] = {}
_history_locks_lock = threading.Lock()


def _history_lock(history: Path) -> threading.Lock:
    with _history_locks_lock:
        return _history_locks.setdefault(history.resolve(), threading.Lock())


def _snapshot(src: Path, history: Path) -> Path:
    """Copy ``src`` into ``history`` as ``NNN_<stem>_<sha1:8>.py``; a project directory
    (``native/project.py``) as its bundle, named after the project. The number is taken
    under the directory's lock and the file created exclusively (``O_EXCL``), so snapshots
    taken at once (sessions, a sweep's thread) never share a number or overwrite a file."""
    history.mkdir(parents=True, exist_ok=True)
    project_dir = src.is_dir()
    if project_dir:
        text, project = native_project.pack(src)
        data, stem = text.encode(), project.manifest.name
    else:
        data, stem = src.read_bytes(), src.stem
    digest = hashlib.sha1(data).hexdigest()[:8]
    with _history_lock(history):
        # after the highest number, not the count: a deleted snapshot must not cause a clash
        names = (p.name for p in history.glob("*.py"))
        numbers = [int(m[1]) for name in names if (m := re.match(r"(\d+)_", name))]
        seq = 1 + max(numbers, default=0)
        while True:
            dst = history / f"{seq:03d}_{stem}_{digest}.py"
            try:
                fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
                break
            except FileExistsError:  # another process took the number
                seq += 1
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)  # the bytes the digest is of
    except BaseException:
        dst.unlink(missing_ok=True)
        raise
    if not project_dir:
        shutil.copystat(src, dst)  # as shutil.copy2: mode and times of the candidate
    return dst


def _agent_dir(run: RunDir, target_id: str | None) -> Path:
    """The agent's working directory of a target (``None``: the transforms)."""
    return run.target(target_id) if target_id else run.transforms_dir


def snapshot(run: RunDir, src: Path, target_id: str | None = None) -> Path:
    """Snapshot ``src`` into the history of a target (``None``: of the transforms).

    The snapshot is what gets evaluated and integrated; in a run with ``.truth/``
    it is read-only there and the agent gets a copy in its own ``history/``."""
    snap = _snapshot(src, run.history_dir(target_id))
    if run.sealed():
        truth.read_only(snap)
        copy = truth.replace(_agent_dir(run, target_id) / "history" / snap.name)
        shutil.copyfile(snap, copy)
    return snap


def _append(
    run: RunDir,
    target_id: str | None,
    record: dict[str, Any],
    keeper: Truth | None,
    *,
    quick: bool = False,
) -> None:
    """Append an evaluation record (``.truth/`` + the agent's copy, or the old place);
    ``quick``: to the quick checks' file of the target (``dedup.quick_file``)."""
    path = dedup.quick_file(run, target_id) if quick and target_id else run.results_file(target_id)
    (keeper or truth.of(run)).append(path, record)
    if run.sealed():
        append_jsonl(_agent_dir(run, target_id) / path.name, record)


def _in_truth(run: RunDir, path: Path) -> dict[str, Any] | None:
    """Error result for a path inside ``.truth/`` (agents work on their own copies)."""
    if not path.resolve().is_relative_to(run.truth_dir.resolve()):
        return None
    return {
        "status": "error",
        "error": f"{path} is inside {truth.TRUTH_DIR}/, the evaluator's ground truth; use the "
        "files in your working directory (candidates/, history/)",
    }


def compact(result: dict[str, Any]) -> dict[str, Any]:
    """Keep the evaluation result readable for the model."""
    keep: dict[str, Any] = {
        k: result[k]
        for k in (
            "status",
            "correct",
            "speedup",
            "est_saved_ms_per_run",
            "est_saved_calls",  # the calls it stands for (kernels/weights.py)
            "ref_ms_weighted",
            "new_ms_weighted",
            "error",
            "failed_case",
            # which correctness stage failed (kernels/evaluate.py)
            "stage",
            "failed_check",
            "custom_kernel_share",
            "kernel_launches_reference",
            "kernel_launches_candidate",
            "kernels_candidate",
            "kernels_reference",
            "streams",  # declared side streams (kernel_agent.concurrency)
            "undeclared_streams",
            "compiler_stats",  # registers, spills (kernels/ncu.py)
            "eval_seconds",
            "compile_s",
            "compile_check",  # torch.compile compatibility (kernels/compile_check.py)
            "scale_rule",  # fp8_mx: the MXFP8 scale-rule guard (kernels/scale_guard.py)
            # speed of light (kernels/roofline.py)
            "pct_of_sol",
            "sol_ms_weighted",
            "bound",
            "launch_floor_ms",
            "suspicious_faster_than_sol",
            "sol_unreliable",
            "sol_note",
            "sol_error",
            "peak_memory",  # the per-call peak vs the reference's (kernels/evaluate.py)
        )
        if k in result
    }
    cases = []
    for c in result.get("cases", []):
        item = {
            k: c.get(k)
            for k in (
                "signature",
                "calls_per_run",
                "target_calls",
                "ok",
                "max_abs_err",
                "min_cosine",
                "ref_ms",
                "new_ms",
                "speedup",
                "timing_spread",
                "torch_compile_ms",
                "flops",
                "min_bytes",
                "sol_ms",
                "pct_of_sol",
                "bound",
                "l2_resident",
                "suspicious_faster_than_sol",
                "sol_unreliable",
                "peak_delta_mib",
            )
            if c.get(k) is not None
        }
        if c.get("failures"):
            item["failures"] = c["failures"]
        cases.append(item)
    if cases:
        keep["cases"] = cases
    return keep


#: An evaluation's profile tables (``profile=true`` / ``"ncu"``): they go to the session's
#: ``profiles/<snapshot>.json`` and the result keeps a summary (:func:`profile_file`, #186).
PROFILE_KEYS = ("kernels_candidate", "kernels_reference", "compiler_stats", "profile_error", "ncu")
PROFILES_DIR = "profiles"
PROFILE_TOP = 3  # kernels per side, spill warnings and ncu kernels in the summary


def _share_rows(rows: Any) -> dict[str, Any]:
    """A per-kernel GPU time table (``kernels_candidate``) in short: the kernels it lists,
    their GPU time and the top ones with their share."""
    rows = [r for r in rows or [] if isinstance(r, dict)]
    total = sum(float(r.get("us") or 0.0) for r in rows)
    top = [
        f"{str(r.get('kernel'))[:70]}: {r.get('us')} us, {r.get('calls')} calls"
        + (f" ({float(r.get('us') or 0.0) / total:.0%})" if total else "")
        for r in rows[:PROFILE_TOP]
    ]
    return {"kernels": len(rows), "gpu_us": round(total, 1), "top": top}


def profile_summary(tables: dict[str, Any]) -> dict[str, Any]:
    """The few lines of an evaluation's profile tables (:data:`PROFILE_KEYS`) that stay in
    its result: the top kernels of candidate and reference with their share of GPU time,
    the compiler's spill warnings, each top ncu kernel's bound."""
    out: dict[str, Any] = {}
    for side in ("candidate", "reference"):
        if f"kernels_{side}" in tables:
            out[side] = _share_rows(tables[f"kernels_{side}"])
    stats = tables.get("compiler_stats") or {}
    if isinstance(stats, dict) and stats.get("warnings"):
        out["warnings"] = stats["warnings"][:PROFILE_TOP]
    if error := tables.get("profile_error"):
        out["error"] = str(error)[:300]
    ncu = tables.get("ncu")
    if isinstance(ncu, dict) and ncu.get("status") == "ok":
        out["ncu"] = [
            f"{str(k.get('kernel'))[:70]}: {k.get('bound')} ({str(k.get('why') or '')[:160]})"
            for k in (ncu.get("kernels") or [])[:PROFILE_TOP]
            if isinstance(k, dict)
        ]
    elif isinstance(ncu, dict):
        out["ncu"] = {"status": ncu.get("status"), "reason": ncu.get("reason")}
    return out


def profile_file(directory: Path, snap: Path, out: dict[str, Any], result: dict[str, Any]) -> None:
    """Move an evaluation's profile tables (:data:`PROFILE_KEYS`, from ``out`` and the full
    ``result``) into ``<directory>/profiles/<snapshot>.json`` (``directory``: the session's
    working directory) and leave in ``out`` its ``profile``: :func:`profile_summary` and the
    file's path. Up to 24 KB of tables then stay out of the engineer's context; it reads
    the file when it needs more, or hands it to the ``profile-analyst`` helper (#186)."""
    tables = {k: out.pop(k, None) or result.get(k) for k in PROFILE_KEYS}
    tables = {k: v for k, v in tables.items() if v}
    if not tables:
        return
    path = directory / PROFILES_DIR / f"{Path(snap.name).stem}.json"
    try:
        write_json(path, {"snapshot": snap.name, **tables})
    except OSError as exc:  # the tables then stay inline
        out |= tables
        out["profile_note"] = f"profile tables not written to {path}: {exc}"
        return
    out["profile"] = profile_summary(tables) | {
        "file": str(path),
        "note": "the full tables (per-kernel GPU time of candidate and reference, compiler "
        "stats, ncu metrics) are in the file: Read it, or give the path to profile-analyst",
    }


def record_candidate(
    run: RunDir,
    target_id: str,
    src: Path,
    snap: Path,
    result: dict[str, Any],
    *,
    hypothesis: str,
    parent: str | None = None,
    eval_s: float | None = None,
    when: float | None = None,
    snapshot_sha256: str | None = None,
    keeper: Truth | None = None,
    idea: str = "",
    expected_speedup: float | None = None,
    worker: int | None = None,
    mode: str = dedup.FULL,
    reevaluates: dict[str, Any] | None = None,
    queue_s: float | None = None,
    session: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Append a kernel evaluation to ``results.jsonl`` and the run ledger.

    The record carries the sha256 of the snapshot it measured (``snapshot_sha256``,
    computed from ``snap`` when not given); winners are picked only from records
    whose snapshot still has it. ``keeper``: this process's :class:`Truth` of the run.
    ``idea`` / ``expected_speedup``: the agent's ``idea_id`` and the speedup it expected;
    ``worker``: the target's worker that evaluated it (``workers.py``). A ``quick``
    ``mode`` check goes to ``quick.jsonl`` instead, as a ``quick_ok`` / ``quick_fail`` row.
    ``reevaluates``: the earlier record of the same snapshot this evaluation replaces
    (its ``exp``, ``speedup``, why; a ``re-evaluated`` row, :func:`current_records`).
    ``queue_s``: the time it waited for the GPU (:mod:`kernel_agent.gpuqueue`; ``eval_s``
    is the evaluation's own time).
    ``session``: the agent session that evaluated it (:class:`SessionBinding` ``label``).
    """
    target_dir = run.target(target_id)
    quick = mode == dedup.QUICK
    source = snap.read_text()
    status = (ledger.QUICK_OK if result.get("correct") else ledger.QUICK_FAIL) if quick else None
    row = ledger.record_kernel(
        run,
        target_id,
        result,
        snapshot=snap.name,
        hypothesis=hypothesis,
        parent=parent,
        source=source,
        eval_s=eval_s,
        when=when,
        idea=idea,
        worker=worker,
        status=ledger.REEVALUATED if reevaluates else status,
        queue_s=queue_s,
        session=session,
    )
    record = {
        "time": time.strftime("%H:%M:%S", time.localtime(when)),
        "candidate": str(src.relative_to(target_dir))
        if src.is_relative_to(target_dir)
        else str(src),
        "snapshot": f"history/{snap.name}",
        "snapshot_sha256": snapshot_sha256 or sha256_file(snap),
        **{k: v for k, v in result.items() if k not in ("kernels_candidate", "kernels_reference")},
        "exp": row["exp"],
        "ledger_status": row["status"],
        "backend": row["backend"],
        "hypothesis": hypothesis,
        "parent": parent,
        "idea": idea or None,
        "expected_speedup": expected_speedup,
        "source_key": dedup.source_key(source),
        **({"worker": worker} if worker else {}),
        **({"session": session} if session else {}),
        **({"mode": dedup.QUICK} if quick else {}),
        **({"reevaluates": reevaluates} if reevaluates else {}),
        **({"queue_s": queue_s} if queue_s is not None else {}),
    }
    if isinstance(record.get("error"), str):
        record["error"] = record["error"][-1500:]
    _append(run, target_id, record, keeper, quick=quick)
    if row["status"] == ledger.KEEP:  # a new best: on the board for every session (#187)
        board.kernel_winner(run, target_id, row, hypothesis=hypothesis, session=session)
    return record, row


def _expected(value: Any) -> float | None:
    """``expected_speedup`` as a positive float (None when missing or not a number)."""
    try:
        number = float(str(value).strip().rstrip("xX"))  # "1.5x" too
    except ValueError:
        return None
    return number if 0 < number < 1e6 else None


def idea_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``results.jsonl`` records as rows for :func:`kernel_agent.ledger.ideas`."""
    return [
        {
            "idea": r.get("idea"),
            "status": r.get("ledger_status") or r.get("status"),
            "correct": bool(r.get("correct")),
            "speedup": r.get("speedup"),
            "expected_speedup": r.get("expected_speedup"),
            "hypothesis": r.get("hypothesis"),
            "exp": r.get("exp"),
        }
        for r in records
    ]


def idea_feedback(
    records: list[dict[str, Any]], idea: str, expected: float | None, row: dict[str, Any]
) -> dict[str, Any]:
    """``idea`` part of an ``evaluate_candidate`` result: expected vs measured, the idea so far."""
    out: dict[str, Any] = {"id": idea or None, "expected_speedup": expected}
    out["speedup"] = row["speedup"] if row["correct"] else None
    if row["correct"] and expected and row["speedup"]:
        out["vs_expected"] = f"{row['speedup']:.3f}x measured vs {expected:.3f}x expected"
    stats = next((s for s in ledger.ideas(idea_rows(records)) if s["idea"] == idea), None)
    if stats:
        out |= {k: stats[k] for k in ("tries", "best", "kept", "slow", "bugs", "verdict")}
    if not row["correct"] and idea:
        out["note"] = (
            f"a {row['status']} is a bug in this attempt, not evidence against the idea: "
            f"fix it and evaluate again with idea_id={idea!r} before you drop the idea"
        )
    return out


def record_e2e_result(
    run: RunDir,
    result: dict[str, Any],
    snaps: list[Path],
    kernels: list[str],
    *,
    hypothesis: str = "",
    eval_s: float | None = None,
    when: float | None = None,
    keeper: Truth | None = None,
    queue_s: float | None = None,
    session: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Append an ``evaluate_e2e`` measurement to the transforms' ``results.jsonl`` and the
    ledger; ``transforms_sha256`` holds the digests of the snapshots it measured; a run
    with a project bundle or a native stage target is the native arm's (``native/engine.py``);
    ``session``: the agent session that ran it (:class:`SessionBinding` ``label``)."""
    row = ledger.record_e2e(
        run,
        result,
        backend=native_engine.e2e_backend(snaps, kernels),
        snapshot=ledger.e2e_snapshot(snaps, kernels),
        hypothesis=hypothesis,
        eval_s=eval_s,
        when=when,
        queue_s=queue_s,
        session=session,
    )
    record = {
        "time": time.strftime("%H:%M:%S", time.localtime(when)),
        "transforms": [f"history/{s.name}" for s in snaps],
        "transforms_sha256": [sha256_file(s) for s in snaps],
        "kernels": kernels,
        **result,
        "exp": row["exp"],
        "ledger_status": row["status"],
        "hypothesis": hypothesis,
        **({"queue_s": queue_s} if queue_s is not None else {}),
        **({"session": session} if session else {}),
    }
    _append(run, None, record, keeper)
    if row["status"] == ledger.KEEP:  # a new end-to-end best: on the board (#187)
        board.e2e_winner(run, row, snaps, kernels, hypothesis=hypothesis, session=session)
    return record, row


def current_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``records`` without those a later re-evaluation of the same snapshot replaced
    (``reevaluates``): the current evaluator's verdict on a snapshot counts."""

    def name(rec: dict[str, Any]) -> str:
        return Path(str(rec.get("snapshot", ""))).name

    last = {name(r): i for i, r in enumerate(records) if r.get("reevaluates")}
    return [r for i, r in enumerate(records) if last.get(name(r), i) <= i]


def ranked_for_target(
    run: RunDir, target_id: str, keeper: Truth | None = None
) -> Iterator[dict[str, Any]]:
    """The correct records of a target, fastest first, whose snapshot is still the
    evaluated file (checked lazily, as the caller asks for the next one).

    Only records kernel-agent wrote count (lines appended by anyone else are
    ignored); a target whose records were modified has none. Every worker of a
    target records here (``workers.py``), so this ranks across workers. A re-evaluated
    snapshot ranks by its re-evaluation (:func:`current_records`)."""
    keeper = keeper or truth.of(run)
    try:
        records = current_records(keeper.records(run.results_file(target_id)))
    except TamperError:
        return
    ranked = [r for r in records if r.get("correct") and r.get("speedup") is not None]
    ranked.sort(key=lambda r: -float(r["speedup"]))  # stable: the first of equals wins
    history = run.history_dir(target_id)
    for rec in ranked:
        name = Path(str(rec.get("snapshot", ""))).name
        if keeper.snapshot_ok(history / name, rec.get("snapshot_sha256")):
            yield rec


def best_for_target(
    run: RunDir, target_id: str, keeper: Truth | None = None
) -> dict[str, Any] | None:
    """The fastest correct record of a target (of all its workers) whose snapshot is still
    the evaluated file (:func:`ranked_for_target`)."""
    return next(ranked_for_target(run, target_id, keeper), None)


def _uncounted(budget: Budget, agent: str, evals_budget: int | None) -> dict[str, Any]:
    """``budget`` of a quick check or a duplicate: shown, not counted."""
    used = budget.evals.get(agent, 0)
    return {"budget": {"evals_used": used, "evals_budget": evals_budget, "counted": False}}


def _prebuilt(path: Path) -> dict[str, Any] | None:
    """Compile a project candidate (directory or bundle) outside the GPU lock
    (``native_project.prebuild``): None when it built (or cannot be built without the
    evaluator), else the ``build_error`` / ``timeout`` result to record instead of an
    evaluation."""
    if not (path.is_dir() or native_project.read_bundle(path) is not None):
        return None
    pre = native_project.prebuild(path)
    if pre.get("status") in ("ok", "skipped"):
        return None
    return {
        "status": pre.get("status") or "build_error",
        "correct": False,
        "passed": False,
        "error": f"the project did not build (outside the GPU, before the evaluation):\n"
        f"{pre.get('error', '')}"[-4000:],
        "compile_s": pre.get("seconds"),
    }


def build_server(
    run: RunDir,
    budget: Budget | None = None,
    keeper: Truth | None = None,
    binding: SessionBinding | None = None,
) -> Any:
    """Tools bound to one run directory (its budget: eval timeout + advice; its truth) and
    to one agent session (:class:`SessionBinding`: its label, worker directory, the name its
    budget advice is kept under, its evaluation budget and working directory). Build one
    server per session (``Orchestrator._agent``)."""
    budget = budget or Budget(run)
    keeper = keeper or truth.of(run)
    bound = binding or SessionBinding()
    cwd, session = bound.cwd, bound.label or None
    # the run's blackboard (board.py, #187): the subscription of a session of a role with the
    # board; what is on it now is in its digest, what is posted later rides on its results
    posts = board.active(run)
    reader = board.Reader.of_session(run, bound) if posts is not None else None
    if posts is None or reader is None or reader.role not in board.ROLES:
        reader = None
    else:
        posts.join(reader.label)

    def _news() -> dict[str, Any]:
        """``board``: the first lines of the board's new entries for this session ({}: none)."""
        found = board.active(run)
        return found.piggyback(reader) if found is not None and reader is not None else {}

    def _path(base: Path, path: str) -> Path:
        """``path`` relative to the session's ``cwd`` when it exists there, else to ``base``."""
        if cwd is not None and not Path(path).is_absolute() and (cwd / path).exists():
            return cwd / path
        return _resolve(base, path)

    def _best_so_far(target_id: str) -> dict[str, Any]:
        best = best_for_target(run, target_id, keeper)
        return {
            "best_so_far": {"snapshot": best["snapshot"], "speedup": best["speedup"]}
            if best
            else None
        }

    async def _duplicate(
        cached: dict[str, Any],
        args: dict[str, Any],
        hypothesis: str,
        source: str,
        idea: str,
        worker: int | None,
    ) -> dict[str, Any]:
        """The earlier result of a candidate evaluated before (``dedup.py``) + its ledger row."""
        target_id = args["target_id"]
        row = ledger.record_kernel(
            run,
            target_id,
            {"correct": cached.get("correct")},
            snapshot=Path(str(cached.get("snapshot"))).name,
            hypothesis=hypothesis,
            parent=args.get("parent"),
            source=source,
            idea=idea,
            worker=worker,
            status=ledger.DUPLICATE,
            session=session,
        )
        refresh(run, target_id)
        out = compact(cached)
        out["duplicate"] = (
            f"{dedup.label(cached)}: the same source up to comments and formatting, so it was "
            "not evaluated again and this is that result. No evaluation, budget or streak was "
            "used; change the kernel to measure something new."
        )
        out["duplicate_of"] = cached.get("snapshot")
        if cached.get("mode") == dedup.QUICK:
            out["mode"], out["not_a_benchmark"] = dedup.QUICK, QUICK_NOTE
        out["ledger"] = {"exp": row["exp"], "status": row["status"]}
        return out | _best_so_far(target_id)

    @tool(
        "evaluate_candidate",
        "Compile a kernel candidate, verify it on every captured case of the target and "
        "benchmark it against the reference module. Records the result and snapshots the file. "
        "mode=quick only checks correctness on two cases (no timing, free); a candidate that "
        "was evaluated before (same code up to comments/formatting) returns that result.",
        {
            "type": "object",
            "properties": {
                "target_id": {"type": "string"},
                "candidate": {
                    "type": "string",
                    "description": "path to the candidate .py (relative to your working dir)",
                },
                "hypothesis": {
                    "type": "string",
                    "description": "one sentence: what changed and why it should be faster",
                },
                "parent": {
                    "type": "string",
                    "description": "snapshot (history/...) or candidate this one builds on",
                },
                "idea_id": {
                    "type": "string",
                    "description": "short slug of the idea this candidate implements, e.g. "
                    "`splitk_gemv`: the same id for every attempt and fix of one idea",
                },
                "expected_speedup": {
                    "type": "number",
                    "description": "module speedup you expect if the idea works (the result "
                    "puts it next to the measured one)",
                },
                "profile": {
                    "type": ["boolean", "string"],
                    "enum": [False, True, "ncu"],
                    "description": "true: per-kernel GPU time tables and compiler stats "
                    '(registers, spills); "ncu": also Nsight Compute metrics per candidate '
                    "kernel (SM / memory throughput, occupancy, cache hit rates, warp stalls, "
                    "memory / compute / under-utilised), when ncu can profile on this machine. "
                    "The tables go to profiles/<snapshot>.json in your working directory; the "
                    "result has a summary and the file's path (`profile`)",
                    "default": False,
                },
                "compile_check": {
                    "type": "boolean",
                    "description": "also torch.compile the candidate: graph breaks and "
                    "compiled outputs (does it survive the model's torch.compile?)",
                    "default": False,
                },
                "mode": {
                    "type": "string",
                    "enum": [dedup.FULL, dedup.QUICK],
                    "description": "quick: correctness on the smallest and the largest "
                    "captured case only, no timing (not a benchmark, not counted against "
                    "the evaluation budget); full (default): every case, timed",
                    "default": dedup.FULL,
                },
            },
            "required": ["target_id", "candidate", "hypothesis"],
        },
    )
    async def evaluate_candidate(args: dict[str, Any]) -> dict[str, Any]:
        target_id = args["target_id"]
        target_dir = run.target(target_id)
        capture = run.capture_file(target_id)
        if not capture.exists():
            return _text({"status": "error", "error": f"unknown target {target_id}"})
        worker = bound.worker if bound.mine(target_id) else None
        if worker is not None:  # a worker session: its own directory
            target_dir = workers.directory(run, target_id, worker)
        src = _path(target_dir, args["candidate"])
        if (refused := _in_truth(run, src)) is not None:
            return _text(refused)
        if not src.exists():
            return _text({"status": "error", "error": f"{src} does not exist"})
        try:  # a project directory evaluates as its bundle (native/project.py)
            source = native_project.source_of(src)
        except (native_project.ProjectError, OSError) as exc:
            return _text({"status": "error", "error": f"{src}: {exc}"})
        hypothesis = str(args.get("hypothesis") or "").strip()
        if not hypothesis:
            return _text(
                {
                    "status": "error",
                    "error": "hypothesis is required: one sentence on what changed and why "
                    "it should be faster",
                }
            )
        mode = str(args.get("mode") or dedup.FULL).strip().lower()
        if mode not in (dedup.FULL, dedup.QUICK):
            return _text({"status": "error", "error": f'mode is "full" or "quick", not {mode!r}'})
        quick = mode == dedup.QUICK
        idea = ledger.idea_slug(args.get("idea_id"))
        expected = _expected(args.get("expected_speedup"))
        name, evals_budget = bound.kernel_agent(target_id), bound.evaluations
        try:
            capture_sha256 = keeper.expect(capture)
        except TamperError as exc:
            return _text({"status": "error", "error": str(exc)})
        slot = (str(run.root), target_id, dedup.source_key(source))
        while (busy := _inflight.get(slot)) is not None:  # another worker evaluates this source
            await busy.wait()
        if quick or not args.get("profile"):  # profile tables are never stored: run it again
            cached = dedup.find(
                run,
                target_id,
                slot[2],
                keeper,
                mode=mode,
                compile_check=bool(args.get("compile_check")),
            )
            if cached is not None:
                try:  # a duplicate skips the evaluator, which is where the capture is checked
                    keeper.verify(capture)
                except TamperError as exc:
                    return _text({"status": "tampered", "correct": False, "error": str(exc)})
                return _text(
                    await _duplicate(cached, args, hypothesis, source, idea, worker)
                    | _uncounted(budget, name, evals_budget)
                    | _news()
                )
        _inflight[slot] = done = asyncio.Event()
        try:
            snap = snapshot(run, src, target_id)
            snap_sha256 = sha256_file(snap)
            start = time.perf_counter()
            job = gpuqueue.Job.of(run, "quick" if quick else "eval", target_id)
            # a project compiles outside the GPU lock first: a compiler error is its result
            result = await asyncio.to_thread(_prebuilt, snap)
            if result is None:
                result = await gpuqueue.run(
                    job,
                    run_evaluation,
                    capture,
                    snap,
                    profile=bool(args.get("profile")) and not quick,
                    timeout=budget.eval_timeout_s,
                    capture_sha256=capture_sha256,
                    **({"compile_check": True} if args.get("compile_check") else {}),
                    **({"quick": True} if quick else {}),
                )
            if result.get("status") == "tampered":  # the evaluator refused the capture
                keeper.alarm(capture, str(result.get("error")))
            elif sha256_file(snap) != snap_sha256:
                keeper.alarm(snap, "snapshot changed during its evaluation")
                result = {
                    "status": "tampered",
                    "correct": False,
                    "error": f"{snap.name} changed while it was evaluated; result discarded",
                }
            _, row = record_candidate(
                run,
                target_id,
                src,
                snap,
                result,
                hypothesis=hypothesis,
                parent=args.get("parent"),
                eval_s=round(time.perf_counter() - start - job.wait_s, 1),
                snapshot_sha256=snap_sha256,
                keeper=keeper,
                idea=idea,
                expected_speedup=expected,
                worker=worker,
                mode=mode,
                queue_s=job.queue_s,
                session=session,
            )
        finally:
            _inflight.pop(slot, None)
            done.set()
        refresh(run, target_id)
        out = compact(result)
        out["ledger"] = {"exp": row["exp"], "status": row["status"]}
        if str(args.get("profile")).lower() == "ncu" and not quick and result.get("correct"):
            from kernel_agent.kernels import ncu  # Nsight Compute (#10): not stored

            out["ncu"] = await gpuqueue.run(
                gpuqueue.Job.of(run, "ncu", target_id),
                ncu.profile_candidate,
                capture,
                snap,
                capture_sha256=capture_sha256,
            )
        if args.get("profile") and not quick:  # the tables to a file, a summary here (#186)
            profile_file(cwd or target_dir, snap, out, result)
        if quick:
            out["mode"], out["not_a_benchmark"] = dedup.QUICK, QUICK_NOTE
            out |= _best_so_far(target_id) | _uncounted(budget, name, evals_budget)
            return _text(out | _news())
        if idea or expected is not None:
            try:
                records = keeper.records(run.results_file(target_id))
            except TamperError:
                records = []
            out["idea"] = idea_feedback(records, idea, expected, row)
        out |= _best_so_far(target_id)
        out |= budget.feedback(
            name,
            run.results_file(target_id),
            evals_budget,
            pct_of_sol=sol_signal(result),
        )
        return _text(out | _news())

    @tool(
        "sweep_candidate",
        "Tune the keyword arguments of a candidate's build(reference, **config) (block "
        "sizes, num_warps, num_stages, vector widths) in one GPU session: every config is "
        "built and checked on two cases (failing ones are listed with their error), the "
        "passing ones are timed interleaved against the reference, and the fastest is fully "
        "evaluated and recorded like evaluate_candidate (with its config bound into the "
        "snapshot). Returns the table sorted by weighted speedup. Counts as ONE evaluation.",
        {
            "type": "object",
            "properties": {
                "target_id": {"type": "string"},
                "candidate": {
                    "type": "string",
                    "description": "path to the candidate .py (relative to your working dir) "
                    "whose build(reference, **config) takes the swept keyword arguments",
                },
                "configs": {
                    "type": ["array", "object"],
                    "description": 'build() keyword arguments per config, e.g. [{"BLOCK": '
                    '512, "num_warps": 4}, {"BLOCK": 1024, "num_warps": 8}], or a dict of '
                    'lists for every combination ({"BLOCK": [512, 1024], "num_warps": [4, 8]})',
                },
                "max_configs": {
                    "type": "integer",
                    "description": f"sweep at most this many configs (default "
                    f"{sweep_mod.DEFAULT_MAX_CONFIGS}, at most {sweep_mod.MAX_CONFIGS})",
                    "default": sweep_mod.DEFAULT_MAX_CONFIGS,
                },
                "hypothesis": {
                    "type": "string",
                    "description": "one sentence: which parameters you sweep and why they matter",
                },
                "idea_id": {"type": "string", "description": "the idea these configs tune"},
                "expected_speedup": {"type": "number"},
                "parent": {"type": "string"},
                "compile_check": {"type": "boolean", "default": False},
            },
            "required": ["target_id", "candidate", "configs", "hypothesis"],
        },
    )
    async def sweep_candidate(args: dict[str, Any]) -> dict[str, Any]:
        target_id = args["target_id"]
        target_dir = run.target(target_id)
        capture = run.capture_file(target_id)
        if not capture.exists():
            return _text({"status": "error", "error": f"unknown target {target_id}"})
        worker = bound.worker if bound.mine(target_id) else None
        if worker is not None:
            target_dir = workers.directory(run, target_id, worker)
        src = _path(target_dir, args["candidate"])
        if (refused := _in_truth(run, src)) is not None:
            return _text(refused)
        if not src.exists():
            return _text({"status": "error", "error": f"{src} does not exist"})
        if src.is_dir():  # a project: valid, and compiled outside the GPU lock first
            try:
                native_project.Project.from_dir(src)
            except native_project.ProjectError as exc:
                return _text({"status": "error", "error": f"{src}: {exc}"})
            if (failed := await asyncio.to_thread(_prebuilt, src)) is not None:
                return _text(failed)
        hypothesis = str(args.get("hypothesis") or "").strip()
        if not hypothesis:
            return _text({"status": "error", "error": "hypothesis is required"})
        try:
            configs, notes = sweep_mod.configs_from(args.get("configs"), args.get("max_configs"))
        except ValueError as exc:
            return _text({"status": "error", "error": str(exc)})
        idea = ledger.idea_slug(args.get("idea_id"))
        expected = _expected(args.get("expected_speedup"))
        name, evals_budget = bound.kernel_agent(target_id), bound.evaluations
        try:
            capture_sha256 = keeper.expect(capture)
        except TamperError as exc:
            return _text({"status": "error", "error": str(exc)})
        snaps: list[tuple[Path, str]] = []

        def prepare(bound: Path) -> Path:  # the best config, bound into the source
            snap = snapshot(run, bound, target_id)
            snaps.append((snap, sha256_file(snap)))
            return snap

        start = time.perf_counter()
        job = gpuqueue.Job.of(run, "sweep", target_id)
        data = await gpuqueue.run(
            job,
            sweep_mod.run_sweep,
            capture,
            src,
            configs,
            timeout=budget.eval_timeout_s,
            capture_sha256=capture_sha256,
            compile_check=bool(args.get("compile_check")),
            prepare=prepare,
        )
        snap, snap_sha256 = snaps[0]
        result = data["evaluation"]
        if result.get("status") == "tampered":
            keeper.alarm(capture, str(result.get("error")))
        elif sha256_file(snap) != snap_sha256:
            keeper.alarm(snap, "snapshot changed during its evaluation")
            result = {
                "status": "tampered",
                "correct": False,
                "error": f"{snap.name} changed while it was evaluated; result discarded",
            }
        info = data["sweep"]
        config = data["config"]
        result = {**result, "config": config, "sweep": {**info, "notes": notes}}
        tag = f"{sweep_mod.label(config)}; best of {info['passed']}/{info['configs']} configs"
        _, row = record_candidate(
            run,
            target_id,
            src,
            snap,
            result,
            hypothesis=f"{hypothesis} [sweep: {tag}]",
            parent=args.get("parent"),
            eval_s=round(time.perf_counter() - start - job.wait_s, 1),
            snapshot_sha256=snap_sha256,
            keeper=keeper,
            idea=idea,
            expected_speedup=expected,
            worker=worker,
            queue_s=job.queue_s,
            session=session,
        )
        refresh(run, target_id)
        out = compact(result)
        out["config"], out["snapshot"] = config, f"history/{snap.name}"
        out["sweep"] = {
            k: info[k] for k in ("configs", "passed", "failed", "skipped", "seconds") if k in info
        }
        for key in ("rounds", "cases", "timing", "note", "sol_note"):
            if info.get(key):
                out["sweep"][key] = info[key]
        if notes:
            out["sweep"]["notes"] = notes
        out["sweep"]["table"] = [sweep_mod.compact_row(r) for r in info["table"]]
        out["ledger"] = {"exp": row["exp"], "status": row["status"]}
        if idea or expected is not None:
            try:
                records = keeper.records(run.results_file(target_id))
            except TamperError:
                records = []
            out["idea"] = idea_feedback(records, idea, expected, row)
        out |= _best_so_far(target_id)
        out |= budget.feedback(  # a sweep is one evaluation, however many configs it timed
            name,
            run.results_file(target_id),
            evals_budget,
            pct_of_sol=sol_signal(result),
        )
        return _text(out | _news())

    @tool(
        "best_result",
        "Best correct evaluation so far for a target, the number of evaluations, the last "
        "15 and per idea (idea_id): tries, best speedup, bugs (failed) vs slow (correct, "
        "not faster), verdict and last hypothesis.",
        {"target_id": str},
    )
    async def best_result(args: dict[str, Any]) -> dict[str, Any]:
        try:
            records = keeper.records(run.results_file(args["target_id"]))
        except TamperError as exc:
            return _text({"status": "error", "error": str(exc)})
        best = best_for_target(run, args["target_id"], keeper)
        return _text(
            {
                "evaluations": len(records),
                "best": compact(best) | {"snapshot": best["snapshot"]} if best else None,
                "history": [
                    {
                        "snapshot": r.get("snapshot"),
                        "status": r.get("ledger_status") or r.get("status"),
                        "speedup": r.get("speedup"),
                        "idea": r.get("idea"),
                        "hypothesis": r.get("hypothesis"),
                        **({"worker": r["worker"]} if r.get("worker") else {}),
                    }
                    for r in records[-15:]
                ],
                "ideas": ledger.ideas(idea_rows(records)),
                "untagged": sum(not r.get("idea") for r in records),
            }
        )

    @tool(
        "evaluate_e2e",
        "Load the full model in a fresh process, apply kernel replacements and/or model "
        "transforms, run the workload, compare with the baseline output and time it.",
        {
            "type": "object",
            "properties": {
                "transforms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "transform .py paths (relative to the run's transforms dir)",
                },
                "kernels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "TARGET_ID=path/to/candidate.py entries",
                },
                "hypothesis": {
                    "type": "string",
                    "description": "one sentence: what this combination tests",
                },
            },
        },
    )
    async def evaluate_e2e(args: dict[str, Any]) -> dict[str, Any]:
        cli: list[str] = ["--warmup", "2"]
        snaps = []
        for t in args.get("transforms") or []:
            src = _path(run.transforms_dir, t)
            if (refused := _in_truth(run, src)) is not None:
                return _text(refused)
            if not src.exists():
                return _text({"status": "error", "error": f"{src} does not exist"})
            try:  # a project directory: its bundle (native/project.py)
                snap = snapshot(run, src)
            except native_project.ProjectError as exc:
                return _text({"status": "error", "error": f"{src}: {exc}"})
            snaps.append(snap)
            cli += ["--transform", str(snap)]
        kernels: list[Path] = []
        for k in args.get("kernels") or []:
            target_id, _, path = k.partition("=")
            kernel = _path(run.target(target_id), path)
            if (refused := _in_truth(run, kernel)) is not None:
                return _text(refused)
            kernels.append(kernel)
            cli += ["--kernel", f"{target_id}={kernel}"]
        start = time.perf_counter()
        job = gpuqueue.Job.of(run, "e2e")
        result: dict[str, Any] | None = None
        for file in [*snaps, *kernels]:  # projects compile outside the GPU lock, first
            if result is None:
                result = await asyncio.to_thread(_prebuilt, file)
        if result is None:
            result = await gpuqueue.run(job, call_worker, run, "e2e", *cli, *keeper.worker_args())
        _, row = record_e2e_result(
            run,
            result,
            snaps,
            args.get("kernels") or [],
            hypothesis=str(args.get("hypothesis") or ""),
            eval_s=round(time.perf_counter() - start - job.wait_s, 1),
            keeper=keeper,
            queue_s=job.queue_s,
            session=session,
        )
        refresh(run)
        result["ledger"] = {"exp": row["exp"], "status": row["status"]}
        if isinstance(result.get("error"), str):
            result["error"] = result["error"][-3000:]
        result |= budget.feedback(
            bound.e2e_agent(), run.results_file(), bound.evaluations, ok_key="passed"
        )
        return _text(result | _news())

    @tool(
        "check_harness",
        "Load harness.py from the run directory, run it twice and report latency, output "
        "summary and determinism.",
        {"type": "object", "properties": {}},
    )
    async def check_harness(args: dict[str, Any]) -> dict[str, Any]:
        harness = str(run.harness)
        run.update(lambda cfg: cfg["workload"].__setitem__("harness", harness))
        job = gpuqueue.Job.of(run, "harness")
        result = await gpuqueue.run(job, call_worker, run, "analyze", "--no-profile")
        if "error" not in result:
            result["status"] = "ok"
        return _text(result)

    @tool(
        "run_info",
        "Run-level facts: baseline latency, plan, targets and their best results.",
        {"type": "object", "properties": {}},
    )
    async def run_info(args: dict[str, Any]) -> dict[str, Any]:
        baseline = read_json(run.baseline_json, {})
        info = {
            "baseline_ms": baseline.get("median_ms"),
            "workload": baseline.get("workload"),
            "plan": read_json(run.plan_json),
            "targets": {
                t: (best_for_target(run, t, keeper) or {}).get("speedup") for t in run.target_ids()
            },
        }
        return _text(info)

    @tool(
        "verify_rewrite",
        "Region targets: apply rewrite.py to a copy of the captured parent module and replay "
        "every captured call. Outputs and in-place side effects must equal the reference "
        "(bitwise, or within about one unit in the last place) and the parent must call its "
        "Region_<id> module. Reports per case: ok, bitwise, max_abs_err, region_calls.",
        {"target_id": str},
    )
    async def verify_rewrite(args: dict[str, Any]) -> dict[str, Any]:
        timeout = budget.eval_timeout_s
        target_id = str(args["target_id"])
        job = gpuqueue.Job.of(run, "verify_rewrite", target_id)
        result = await gpuqueue.run(job, region.check, run, target_id, keeper, timeout=timeout)
        return _text(result)

    @tool(
        "post_note",
        "Post a conclusion to the run's board, which the other agent sessions read (advice, "
        "never scored): why a kept result of yours wins (winner: its exp:N or snapshot in "
        "refs), a dead end and the error or measurement that proves it (trap), a fact that "
        "holds beyond your kernel (insight), a module you are about to change (claim), a "
        "question for another arm (question, with `to`). Conclusions only, never progress.",
        {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": list(board.AGENT_KINDS)},
                "text": {
                    "type": "string",
                    "description": "the conclusion with its numbers; the first line is what the "
                    f"others see first (at most {board.TEXT_CHARS} characters)",
                },
                "target": {
                    "type": "string",
                    "description": "the target id (or systems, native, a module class) it is "
                    "about; omit it for a note that holds for every target",
                },
                "refs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "what it rests on: history/<snapshot>, exp:<N>, a file path",
                },
                "to": {
                    "type": "string",
                    "description": 'only for "arm:<target, systems or native>", "role:<role>" '
                    'or "session:<label>"',
                },
                "reply_to": {"type": "integer", "description": "the id of the note it answers"},
            },
            "required": ["kind", "text"],
        },
    )
    async def post_note(args: dict[str, Any]) -> dict[str, Any]:
        found = board.active(run)
        if found is None or reader is None:
            return _text({"status": "error", "error": "this session has no board"})
        try:
            out = found.note(run, reader, args)
        except board.Refused as exc:
            return _text({"status": "refused", "error": str(exc)})
        return _text({"status": "posted", **out})

    @tool(
        "read_board",
        "Read the run's board with the full text of its entries: by default those for this "
        "session (about your target or module class, notes for every target, what is "
        "addressed to you; every winner for the systems and native agents) after `since` or "
        "of the last `minutes`; or filtered by kinds and target, or all of them. Advice, "
        "never scored.",
        {
            "type": "object",
            "properties": {
                "since": {"type": "integer", "description": "entries after this id"},
                "minutes": {"type": "integer", "description": "entries of the last N minutes"},
                "kinds": {"type": "array", "items": {"type": "string", "enum": list(board.KINDS)}},
                "target": {"type": "string", "description": "a target id or module class"},
                "all": {"type": "boolean", "description": "every entry, not only yours"},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": board.READ_MAX,
                    "default": board.READ_ITEMS,
                },
            },
        },
    )
    async def read_board(args: dict[str, Any]) -> dict[str, Any]:
        found = board.active(run)
        if found is None or reader is None:
            return _text({"status": "error", "error": "this session has no board"})
        try:
            return _text(found.read(reader, args))
        except board.Refused as exc:
            return _text({"status": "error", "error": str(exc)})

    doc_search, doc_read = doc_tools()
    tools = [
        evaluate_candidate,
        sweep_candidate,
        best_result,
        evaluate_e2e,
        check_harness,
        run_info,
        verify_rewrite,
        doc_search,
        doc_read,
    ]
    if reader is not None:  # a session of a role with the board (board.ROLES)
        tools += [post_note, read_board]
    return create_sdk_mcp_server(SERVER_NAME, version="0.1.0", tools=tools)


def doc_tools() -> tuple[Any, Any]:
    """``doc_search`` / ``doc_read``: the local doc library (``doclib``, issue #177) of
    every session. No network; the first call builds what is missing (seconds)."""
    from kernel_agent import doclib

    @tool(
        "doc_search",
        "Search the local documentation library (no network) of the tools installed here, "
        "at their installed versions: Triton (triton.language, Gluon), CuTe DSL / CUTLASS, "
        "TileLang, PyTorch (cpp_extension, torch.cuda), the CUDA headers (runtime and "
        "driver API, launch attributes, FP8 / FP4 conversions) and cuBLASLt; plus the CUDA "
        "Programming Guide, PTX ISA, cuBLAS, CUTLASS, Triton and TileLang web docs. Query "
        "with the identifier and a few words (`tl.dot_scaled scale layout`, "
        "`cp.async.bulk.tensor`, `cudaLaunchAttributeProgrammaticStreamSerialization`). "
        "Returns the k best chunks: id, title, library, version, origin (installed / web), "
        "source and a snippet; read one with doc_read.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "identifiers and a few words"},
                "library": {
                    "type": "string",
                    "description": "only these libraries (comma-separated): cuda, ptx, "
                    "cublas, cutlass, triton, tilelang, torch",
                },
                "k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 8},
            },
            "required": ["query"],
        },
    )
    async def doc_search(args: dict[str, Any]) -> dict[str, Any]:
        query, library = str(args.get("query") or ""), args.get("library")
        try:
            out = await asyncio.to_thread(doclib.search, query, library, args.get("k") or 8)
        except Exception as exc:  # a broken library never fails the session
            out = {"error": f"the doc library is not available: {exc!r}"}
        return _text(out)

    @tool(
        "doc_read",
        "Read a chunk of the documentation library by its id (from doc_search) and the "
        "chunks after it in the same page or file, up to max_chars. Returns the text with "
        "its title, library, version and source (an installed file:line or a URL): cite "
        "that source. `next`: the id to read on from.",
        {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "max_chars": {"type": "integer", "minimum": 500, "maximum": 20000},
            },
            "required": ["id"],
        },
    )
    async def doc_read(args: dict[str, Any]) -> dict[str, Any]:
        chars = args.get("max_chars") or doclib.store.READ_CHARS
        try:
            out = await asyncio.to_thread(doclib.read, str(args.get("id") or ""), chars)
        except Exception as exc:
            out = {"error": f"the doc library is not available: {exc!r}"}
        return _text(out)

    return doc_search, doc_read


def tool_names(*names: str) -> list[str]:
    return [f"mcp__{SERVER_NAME}__{n}" for n in names]
