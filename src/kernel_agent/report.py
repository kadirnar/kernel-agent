"""Markdown report for a finished (or partially finished) run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kernel_agent.agent.tools import best_for_target
from kernel_agent.workspace import RunDir, read_json, read_jsonl


def _fmt(v: Any, nd: int = 3) -> str:
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return "—" if v is None else str(v)


def write_report(run: RunDir) -> Path:
    data = run.load()
    card = data["card"]
    tc = read_json(run.toolchain_json, {})
    baseline = read_json(run.baseline_json, {})
    plan = read_json(run.plan_json, {})
    integration = read_json(run.root / "integration.json", {})
    costs = read_json(run.root / "costs.json", {})
    gpu = tc.get("gpu") or {}
    final = integration.get("final") or {}

    lines = [
        f"# kernel-agent report: `{card['repo_id']}`",
        "",
        f"* modality: **{card['modality']}**, architectures: {card.get('architectures')}, "
        f"params: {card.get('params')}",
        f"* GPU: {gpu.get('name')} ({gpu.get('arch')}), torch {tc.get('torch_version')}",
        f"* workload: `{baseline.get('workload')}`",
        "",
        "## Result",
        "",
        "| | latency (ms) | speedup | quality |",
        "|---|---|---|---|",
        f"| baseline | {_fmt(baseline.get('median_ms'), 1)} | 1.00x | reference |",
    ]
    if final:
        metrics = ", ".join(f"{k}={v}" for k, v in (final.get("metrics") or {}).items())
        lines.append(
            f"| optimised | {_fmt(final.get('median_ms'), 1)} | **{final.get('speedup')}x** | "
            f"{metrics} |"
        )
    else:
        lines.append("| optimised | — | — | no optimisation passed end-to-end validation |")
    lines += ["", "## Planner analysis", "", plan.get("analysis", "(no plan)"), ""]

    lines += [
        "## Kernel targets",
        "",
        "| target | class | backends | evaluations | best speedup (module) "
        "| est. saved ms/run | best file |",
        "|---|---|---|---|---|---|---|",
    ]
    for target_id in run.target_ids():
        spec = read_json(run.target(target_id) / "spec.json", {})
        records = read_jsonl(run.target(target_id) / "results.jsonl")
        best = best_for_target(run, target_id)
        lines.append(
            f"| `{target_id}` | `{spec.get('module_class')}` | "
            f"{', '.join(spec.get('backends', []))} | "
            f"{len(records)} | {_fmt(best and best.get('speedup'))} | "
            f"{_fmt(best and best.get('est_saved_ms_per_run'))} | "
            f"{best['snapshot'] if best else '—'} |"
        )
        if best:
            for case in best.get("cases", []):
                lines.append(
                    f"|  ↳ `{case.get('signature', '')[:60]}` ×{case.get('calls_per_run')} | | | | "
                    f"{_fmt(case.get('speedup'))} ({_fmt(case.get('ref_ms'), 4)} → "
                    f"{_fmt(case.get('new_ms'), 4)} ms) | | |"
                )
    transforms = read_jsonl(run.transforms_dir / "results.jsonl")
    if transforms:
        lines += [
            "",
            "## Model-level transforms",
            "",
            "| transforms | passed | ms | speedup | note |",
            "|---|---|---|---|---|",
        ]
        for rec in transforms:
            lines.append(
                f"| {', '.join(rec.get('transforms', []))} | {rec.get('passed')} | "
                f"{_fmt(rec.get('median_ms'), 1)} | {_fmt(rec.get('speedup'))} | "
                f"{(rec.get('reason') or rec.get('status') or '')[:80]} |"
            )
    if integration:
        lines += ["", "## Integration", ""]
        for item in integration.get("accepted", []):
            lines.append(f"* accepted {item['kind']}: `{item['item']}`")
        for h in integration.get("history", []):
            lines.append(
                f"* tried {len(h['items'])} item(s): passed={h.get('passed')} "
                f"speedup={h.get('speedup')} {h.get('reason') or ''}"
            )
    if costs:
        total = sum(c.get("usd", 0) for c in costs.values())
        lines += [
            "",
            f"## Agent usage (total ${total:.2f})",
            "",
            "| agent | $ | turns | min | tools |",
            "|---|---|---|---|---|",
        ]
        for name, c in costs.items():
            tools = ", ".join(f"{k}×{v}" for k, v in c.get("tools", {}).items())
            lines.append(
                f"| {name} | {c.get('usd')} | {c.get('turns')} | {c.get('minutes')} | {tools} |"
            )
    lines += [
        "",
        "## Use the optimised model",
        "",
        "```python",
        f'import sys; sys.path.insert(0, "{run.optimized_dir}")',
        "from apply import apply_kernels",
        "apply_kernels(model)",
        "```",
        "",
    ]
    run.report.write_text("\n".join(lines))
    return run.report
