"""In-process MCP tools exposed to the agents (Claude Agent SDK)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from kernel_agent.kernels.evaluate import run_evaluation
from kernel_agent.worker import call_worker
from kernel_agent.workspace import RunDir, append_jsonl, read_json, read_jsonl

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
    seq = len(list(history.glob("*.py"))) + 1
    digest = hashlib.sha1(src.read_bytes()).hexdigest()[:8]
    dst = history / f"{seq:03d}_{src.stem}_{digest}.py"
    shutil.copy2(src, dst)
    return dst


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
            )
            if c.get(k) is not None
        }
        if c.get("failures"):
            item["failures"] = c["failures"]
        cases.append(item)
    if cases:
        keep["cases"] = cases
    return keep


def best_for_target(run: RunDir, target_id: str) -> dict[str, Any] | None:
    best = None
    for rec in read_jsonl(run.target(target_id) / "results.jsonl"):
        if (
            rec.get("correct")
            and rec.get("speedup") is not None
            and (best is None or rec["speedup"] > best["speedup"])
        ):
            best = rec
    return best


def build_server(run: RunDir) -> Any:
    """Tools bound to one run directory."""

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
                "profile": {
                    "type": "boolean",
                    "description": "include per-kernel GPU time tables",
                    "default": False,
                },
            },
            "required": ["target_id", "candidate"],
        },
    )
    async def evaluate_candidate(args: dict[str, Any]) -> dict[str, Any]:
        target_dir = run.target(args["target_id"])
        if not (target_dir / "capture.pt").exists():
            return _text({"status": "error", "error": f"unknown target {args['target_id']}"})
        src = _resolve(target_dir, args["candidate"])
        if not src.exists():
            return _text({"status": "error", "error": f"{src} does not exist"})
        snap = _snapshot(src, target_dir / "history")
        result = await asyncio.to_thread(
            run_evaluation, target_dir / "capture.pt", snap, profile=bool(args.get("profile"))
        )
        record = {
            "time": time.strftime("%H:%M:%S"),
            "candidate": str(src.relative_to(target_dir))
            if src.is_relative_to(target_dir)
            else str(src),
            "snapshot": str(snap.relative_to(target_dir)),
            **{
                k: v
                for k, v in result.items()
                if k not in ("kernels_candidate", "kernels_reference")
            },
        }
        if isinstance(record.get("error"), str):
            record["error"] = record["error"][-1500:]
        append_jsonl(target_dir / "results.jsonl", record)
        out = compact(result)
        best = best_for_target(run, args["target_id"])
        out["best_so_far"] = (
            {"snapshot": best["snapshot"], "speedup": best["speedup"]} if best else None
        )
        return _text(out)

    @tool(
        "best_result",
        "Best correct evaluation so far for a target, plus the number of evaluations.",
        {"target_id": str},
    )
    async def best_result(args: dict[str, Any]) -> dict[str, Any]:
        records = read_jsonl(run.target(args["target_id"]) / "results.jsonl")
        best = best_for_target(run, args["target_id"])
        return _text(
            {
                "evaluations": len(records),
                "best": compact(best) | {"snapshot": best["snapshot"]} if best else None,
                "history": [
                    {
                        "snapshot": r.get("snapshot"),
                        "status": r.get("status"),
                        "speedup": r.get("speedup"),
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
            },
        },
    )
    async def evaluate_e2e(args: dict[str, Any]) -> dict[str, Any]:
        cli: list[str] = ["--warmup", "2"]
        snaps = []
        for t in args.get("transforms") or []:
            src = _resolve(run.transforms_dir, t)
            if not src.exists():
                return _text({"status": "error", "error": f"{src} does not exist"})
            snap = _snapshot(src, run.transforms_dir / "history")
            snaps.append(snap)
            cli += ["--transform", str(snap)]
        for k in args.get("kernels") or []:
            target_id, _, path = k.partition("=")
            cli += ["--kernel", f"{target_id}={_resolve(run.target(target_id), path)}"]
        result = await asyncio.to_thread(call_worker, run, "e2e", *cli)
        record = {
            "time": time.strftime("%H:%M:%S"),
            "transforms": [str(s.relative_to(run.transforms_dir)) for s in snaps],
            "kernels": args.get("kernels") or [],
            **result,
        }
        append_jsonl(run.transforms_dir / "results.jsonl", record)
        if isinstance(result.get("error"), str):
            result["error"] = result["error"][-3000:]
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
                t: (best_for_target(run, t) or {}).get("speedup") for t in run.target_ids()
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
