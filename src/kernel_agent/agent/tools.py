"""In-process MCP tools exposed to the agents (Claude Agent SDK)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from kernel_agent import ledger, truth
from kernel_agent.budget import Budget
from kernel_agent.dashboard import refresh
from kernel_agent.kernels.evaluate import run_evaluation
from kernel_agent.kernels.roofline import sol_signal
from kernel_agent.truth import TamperError, Truth, sha256_file
from kernel_agent.worker import call_worker
from kernel_agent.workspace import RunDir, append_jsonl, read_json

SERVER_NAME = "ka"


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
    run: RunDir, target_id: str | None, record: dict[str, Any], keeper: Truth | None
) -> None:
    """Append an evaluation record (``.truth/`` + the agent's copy, or the old place)."""
    (keeper or truth.of(run)).append(run.results_file(target_id), record)
    if run.sealed():
        append_jsonl(_agent_dir(run, target_id) / "results.jsonl", record)


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
            "kernels_candidate",
            "kernels_reference",
            "eval_seconds",
            "compile_s",
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Append a kernel evaluation to ``results.jsonl`` and the run ledger.

    The record carries the sha256 of the snapshot it measured (``snapshot_sha256``,
    computed from ``snap`` when not given); winners are picked only from records
    whose snapshot still has it. ``keeper``: this process's :class:`Truth` of the run.
    """
    target_dir = run.target(target_id)
    row = ledger.record_kernel(
        run,
        target_id,
        result,
        snapshot=snap.name,
        hypothesis=hypothesis,
        parent=parent,
        source=snap.read_text(),
        eval_s=eval_s,
        when=when,
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
    }
    if isinstance(record.get("error"), str):
        record["error"] = record["error"][-1500:]
    _append(run, target_id, record, keeper)
    return record, row


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


def best_for_target(
    run: RunDir, target_id: str, keeper: Truth | None = None
) -> dict[str, Any] | None:
    """The fastest correct record of a target whose snapshot is still the evaluated file.

    Only records kernel-agent wrote count (lines appended by anyone else are
    ignored); a target whose records were modified has no winner."""
    keeper = keeper or truth.of(run)
    try:
        records = keeper.records(run.results_file(target_id))
    except TamperError:
        return None
    ranked = [r for r in records if r.get("correct") and r.get("speedup") is not None]
    ranked.sort(key=lambda r: -float(r["speedup"]))  # stable: the first of equals wins
    history = run.history_dir(target_id)
    for rec in ranked:
        name = Path(str(rec.get("snapshot", ""))).name
        if keeper.snapshot_ok(history / name, rec.get("snapshot_sha256")):
            return rec
    return None


def build_server(run: RunDir, budget: Budget | None = None, keeper: Truth | None = None) -> Any:
    """Tools bound to one run directory (its budget: eval timeout + advice; its truth)."""
    budget = budget or Budget(run)
    keeper = keeper or truth.of(run)

    @tool(
        "evaluate_candidate",
        "Compile a kernel candidate, verify it on every captured case of the target and "
        "benchmark it against the reference module. Records the result and snapshots the file.",
        {
            "type": "object",
            "properties": {
                "target_id": {"type": "string"},
                "candidate": {
                    "type": "string",
                    "description": "path to the candidate .py (relative to the target dir)",
                },
                "hypothesis": {
                    "type": "string",
                    "description": "one sentence: what changed and why it should be faster",
                },
                "parent": {
                    "type": "string",
                    "description": "snapshot (history/...) or candidate this one builds on",
                },
                "profile": {
                    "type": "boolean",
                    "description": "include per-kernel GPU time tables",
                    "default": False,
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
        try:
            capture_sha256 = keeper.expect(capture)
        except TamperError as exc:
            return _text({"status": "error", "error": str(exc)})
        snap = snapshot(run, src, target_id)
        snap_sha256 = sha256_file(snap)
        start = time.perf_counter()
        result = await asyncio.to_thread(
            run_evaluation,
            capture,
            snap,
            profile=bool(args.get("profile")),
            timeout=budget.eval_timeout_s,
            capture_sha256=capture_sha256,
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
        )
        await asyncio.to_thread(refresh, run, target_id)
        out = compact(result)
        out["ledger"] = {"exp": row["exp"], "status": row["status"]}
        best = best_for_target(run, target_id, keeper)
        out["best_so_far"] = (
            {"snapshot": best["snapshot"], "speedup": best["speedup"]} if best else None
        )
        out |= budget.feedback(
            f"kernel-{target_id}",
            run.results_file(target_id),
            budget.kernel_evals,
            pct_of_sol=sol_signal(result),
        )
        return _text(out)

    @tool(
        "best_result",
        "Best correct evaluation so far for a target, plus the number of evaluations.",
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
                        "hypothesis": r.get("hypothesis"),
                    }
                    for r in records[-15:]
                ],
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

    return create_sdk_mcp_server(
        SERVER_NAME,
        version="0.1.0",
        tools=[evaluate_candidate, best_result, evaluate_e2e, check_harness, run_info],
    )


def tool_names(*names: str) -> list[str]:
    return [f"mcp__{SERVER_NAME}__{n}" for n in names]
