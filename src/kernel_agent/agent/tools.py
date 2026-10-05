"""In-process MCP tools exposed to the agents (Claude Agent SDK)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from kernel_agent import dedup, ledger, region, truth, workers
from kernel_agent.budget import Budget
from kernel_agent.dashboard import refresh
from kernel_agent.kernels import sweep as sweep_mod
from kernel_agent.kernels.evaluate import run_evaluation
from kernel_agent.kernels.roofline import sol_signal
from kernel_agent.truth import TamperError, Truth, sha256_file
from kernel_agent.worker import call_worker
from kernel_agent.workspace import RunDir, append_jsonl, read_json

SERVER_NAME = "ka"
QUICK_NOTE = (
    "quick check: correctness only, on the smallest and the largest captured case. NOT a "
    "benchmark: no timing, no speedup, never a new best, not counted against your evaluation "
    'budget. Evaluate with mode="full" to measure it.'
)
# (run, target, source key) of evaluations in flight: the same source waits for the first one
_inflight: dict[tuple[str, str, str], asyncio.Event] = {}


def _text(data: Any) -> dict[str, Any]:
    body = data if isinstance(data, str) else json.dumps(data, indent=1, default=str)
    if len(body) > 24000:
        body = body[:24000] + "\n... (truncated)"
    return {"content": [{"type": "text", "text": body}]}


def _resolve(base: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base / p)


def _snapshot(src: Path, history: Path) -> Path:
    history.mkdir(parents=True, exist_ok=True)
    # after the highest number, not the count: a deleted snapshot must not cause a clash
    numbers = [int(m[1]) for p in history.glob("*.py") if (m := re.match(r"(\d+)_", p.name))]
    seq = 1 + max(numbers, default=0)
    digest = hashlib.sha1(src.read_bytes()).hexdigest()[:8]
    dst = history / f"{seq:03d}_{src.stem}_{digest}.py"
    shutil.copy2(src, dst)
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
            "eval_seconds",
            "compile_s",
            "compile_check",  # torch.compile compatibility (kernels/compile_check.py)
            # speed of light (kernels/roofline.py)
            "pct_of_sol",
            "sol_ms_weighted",
            "bound",
            "launch_floor_ms",
            "suspicious_faster_than_sol",
            "sol_unreliable",
            "sol_note",
            "sol_error",
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
            )
            if c.get(k) is not None
        }
        if c.get("failures"):
            item["failures"] = c["failures"]
        cases.append(item)
    if cases:
        keep["cases"] = cases
    return keep


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
        **({"mode": dedup.QUICK} if quick else {}),
        **({"reevaluates": reevaluates} if reevaluates else {}),
    }
    if isinstance(record.get("error"), str):
        record["error"] = record["error"][-1500:]
    _append(run, target_id, record, keeper, quick=quick)
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Append an ``evaluate_e2e`` measurement to the transforms' ``results.jsonl`` and the
    ledger; ``transforms_sha256`` holds the digests of the snapshots it measured."""
    row = ledger.record_e2e(
        run,
        result,
        backend=ledger.e2e_backend(snaps, kernels),
        snapshot=ledger.e2e_snapshot(snaps, kernels),
        hypothesis=hypothesis,
        eval_s=eval_s,
        when=when,
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
    }
    _append(run, None, record, keeper)
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


def build_server(
    run: RunDir,
    budget: Budget | None = None,
    keeper: Truth | None = None,
    worker: workers.Binding | None = None,
) -> Any:
    """Tools bound to one run directory (its budget: eval timeout + advice; its truth) and,
    for a worker session, to that worker (its directory, agent name and evaluation budget)."""
    budget = budget or Budget(run)
    keeper = keeper or truth.of(run)

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
        mine: workers.Binding | None,
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
            worker=mine.worker if mine else None,
            status=ledger.DUPLICATE,
        )
        await asyncio.to_thread(refresh, run, target_id)
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
                    "type": "boolean",
                    "description": "include per-kernel GPU time tables",
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
        mine = worker if worker is not None and worker.target_id == target_id else None
        if mine is not None:  # a worker session: its own directory, name and budget
            target_dir = workers.directory(run, target_id, mine.worker)
        src = _resolve(target_dir, args["candidate"])
        if (refused := _in_truth(run, src)) is not None:
            return _text(refused)
        if not src.exists():
            return _text({"status": "error", "error": f"{src} does not exist"})
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
        agent = mine.agent if mine else f"kernel-{target_id}"
        evals_budget = budget.kernel_evals
        if mine is not None and mine.evaluations is not None:
            evals_budget = mine.evaluations
        try:
            capture_sha256 = keeper.expect(capture)
        except TamperError as exc:
            return _text({"status": "error", "error": str(exc)})
        source = src.read_text(errors="replace")
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
                    await _duplicate(cached, args, hypothesis, source, idea, mine)
                    | _uncounted(budget, agent, evals_budget)
                )
        _inflight[slot] = done = asyncio.Event()
        try:
            snap = snapshot(run, src, target_id)
            snap_sha256 = sha256_file(snap)
            start = time.perf_counter()
            result = await asyncio.to_thread(
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
                eval_s=round(time.perf_counter() - start, 1),
                snapshot_sha256=snap_sha256,
                keeper=keeper,
                idea=idea,
                expected_speedup=expected,
                worker=mine.worker if mine else None,
                mode=mode,
            )
        finally:
            _inflight.pop(slot, None)
            done.set()
        await asyncio.to_thread(refresh, run, target_id)
        out = compact(result)
        out["ledger"] = {"exp": row["exp"], "status": row["status"]}
        if quick:
            out["mode"], out["not_a_benchmark"] = dedup.QUICK, QUICK_NOTE
            return _text(out | _best_so_far(target_id) | _uncounted(budget, agent, evals_budget))
        if idea or expected is not None:
            try:
                records = keeper.records(run.results_file(target_id))
            except TamperError:
                records = []
            out["idea"] = idea_feedback(records, idea, expected, row)
        out |= _best_so_far(target_id)
        out |= budget.feedback(
            agent,
            run.results_file(target_id),
            evals_budget,
            pct_of_sol=sol_signal(result),
        )
        return _text(out)

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
        mine = worker if worker is not None and worker.target_id == target_id else None
        if mine is not None:
            target_dir = workers.directory(run, target_id, mine.worker)
        src = _resolve(target_dir, args["candidate"])
        if (refused := _in_truth(run, src)) is not None:
            return _text(refused)
        if not src.exists():
            return _text({"status": "error", "error": f"{src} does not exist"})
        hypothesis = str(args.get("hypothesis") or "").strip()
        if not hypothesis:
            return _text({"status": "error", "error": "hypothesis is required"})
        try:
            configs, notes = sweep_mod.configs_from(args.get("configs"), args.get("max_configs"))
        except ValueError as exc:
            return _text({"status": "error", "error": str(exc)})
        idea = ledger.idea_slug(args.get("idea_id"))
        expected = _expected(args.get("expected_speedup"))
        agent = mine.agent if mine else f"kernel-{target_id}"
        evals_budget = budget.kernel_evals
        if mine is not None and mine.evaluations is not None:
            evals_budget = mine.evaluations
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
        data = await asyncio.to_thread(
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
            eval_s=round(time.perf_counter() - start, 1),
            snapshot_sha256=snap_sha256,
            keeper=keeper,
            idea=idea,
            expected_speedup=expected,
            worker=mine.worker if mine else None,
        )
        await asyncio.to_thread(refresh, run, target_id)
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
            agent,
            run.results_file(target_id),
            evals_budget,
            pct_of_sol=sol_signal(result),
        )
        return _text(out)

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
            src = _resolve(run.transforms_dir, t)
            if (refused := _in_truth(run, src)) is not None:
                return _text(refused)
            if not src.exists():
                return _text({"status": "error", "error": f"{src} does not exist"})
            snap = snapshot(run, src)
            snaps.append(snap)
            cli += ["--transform", str(snap)]
        for k in args.get("kernels") or []:
            target_id, _, path = k.partition("=")
            kernel = _resolve(run.target(target_id), path)
            if (refused := _in_truth(run, kernel)) is not None:
                return _text(refused)
            cli += ["--kernel", f"{target_id}={kernel}"]
        start = time.perf_counter()
        result = await asyncio.to_thread(call_worker, run, "e2e", *cli, *keeper.worker_args())
        _, row = record_e2e_result(
            run,
            result,
            snaps,
            args.get("kernels") or [],
            hypothesis=str(args.get("hypothesis") or ""),
            eval_s=round(time.perf_counter() - start, 1),
            keeper=keeper,
        )
        await asyncio.to_thread(refresh, run)
        result["ledger"] = {"exp": row["exp"], "status": row["status"]}
        if isinstance(result.get("error"), str):
            result["error"] = result["error"][-3000:]
        result |= budget.feedback(
            "systems", run.results_file(), budget.transform_evals, ok_key="passed"
        )
        return _text(result)

    @tool(
        "check_harness",
        "Load harness.py from the run directory, run it twice and report latency, output "
        "summary and determinism.",
        {"type": "object", "properties": {}},
    )
    async def check_harness(args: dict[str, Any]) -> dict[str, Any]:
        cfg = run.load()
        cfg["workload"]["harness"] = str(run.harness)
        from kernel_agent.workspace import write_json

        write_json(run.run_json, cfg)
        result = await asyncio.to_thread(call_worker, run, "analyze", "--no-profile")
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
        result = await asyncio.to_thread(region.check, run, target_id, keeper, timeout=timeout)
        return _text(result)

    return create_sdk_mcp_server(
        SERVER_NAME,
        version="0.1.0",
        tools=[
            evaluate_candidate,
            sweep_candidate,
            best_result,
            evaluate_e2e,
            check_harness,
            run_info,
            verify_rewrite,
        ],
    )


def tool_names(*names: str) -> list[str]:
    return [f"mcp__{SERVER_NAME}__{n}" for n in names]
