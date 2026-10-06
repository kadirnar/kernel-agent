"""Markdown report for a finished (or partially finished) run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kernel_agent import abtest, ledger, library, objective, projection, strong_baseline
from kernel_agent.agent import auth
from kernel_agent.agent.tools import best_for_target
from kernel_agent.dashboard import refresh
from kernel_agent.improve import report_lines
from kernel_agent.kernels import recheck
from kernel_agent.workspace import RunDir, read_json, read_jsonl


def _fmt(v: Any, nd: int = 3) -> str:
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return "—" if v is None else str(v)


def _x(v: float | None) -> str:
    return "—" if v is None else f"{v:.2f}x"


def _chart(run: RunDir, path: Path, alt: str) -> list[str]:
    """Markdown image line for a chart that exists (charts need the ``viz`` extra)."""
    return [f"![{alt}]({path.relative_to(run.root).as_posix()})", ""] if path.exists() else []


def _flat_metrics(metrics: Any, prefix: str = "") -> dict[str, Any]:
    """One level of ``{"teacher_forced": {...}, "free_running": {...}}`` -> dotted keys."""
    flat: dict[str, Any] = {}
    for key, value in (metrics or {}).items():
        if isinstance(value, dict):
            flat.update(_flat_metrics(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _projection_lines(
    entries: list[dict[str, Any]], run: RunDir, baseline: dict[str, Any], base_ms: Any
) -> list[str]:
    """Projected (baseline − Σ est. saved ms) vs measured latency of each accepted set; an
    ``integration.json`` from before #114 has its kernels' savings converted to the metric's
    ms here (:func:`projection.in_metric`)."""
    if not entries:
        return []
    metric = objective.of(baseline)
    base = ledger._num(base_ms) or ledger._num(baseline.get("median_ms"))
    if base is not None and any("est_saved_unit" not in p for p in entries):
        tree, units = projection.tree(run), projection.units_of(run)
        entries = [projection.in_metric(p, tree, units, base) for p in entries]
    lines = [
        "",
        f"Projected (baseline − Σ est. saved ms {metric.per}: a kernel's module-level "
        "estimate, nested kernels counted once, a transform's gain alone) vs measured "
        f"{metric.label} of every accepted set:",
        "",
        "| accepted set | projected ms | measured ms | measured / projected |",
        "|---|---|---|---|",
    ]
    for p in entries:
        names = " + ".join(f"`{ledger.item_label(i)}`" for i in p.get("items", []))
        saved = p.get("est_saved_ms") or {}
        unknown = [ledger.item_label(i) for i, v in saved.items() if v is None]
        nested = [  # the kernels inside (or around) a kernel that counts instead
            ledger.item_label(i) + (f" ({ms / saved[i]:.0%} counted)" if ms > 0 else "")
            for i, ms in (p.get("counted_ms") or {}).items()
            if saved.get(i) and ms < 0.995 * saved[i]
        ]
        projected, measured = p.get("projected_ms"), p.get("measured_ms")
        ratio = measured / projected if measured and projected else None
        note = f" (no estimate: {', '.join(unknown)})" if unknown else ""
        note += f" (not counted, nested: {', '.join(nested)})" if nested else ""
        lines.append(
            f"| {names}{note} | {_fmt(projected, 1)} | {_fmt(measured, 1)} | {_fmt(ratio, 2)} |"
        )
    return lines


def _quality_lines(data: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """The quality mode, and eager's perceptual scores next to it (``near-lossless``)."""
    mode = (data.get("config") or {}).get("quality") or "exact"
    if mode == "exact":
        return []
    info = baseline.get("perceptual") or {}
    if info.get("status") != "ok":
        why = info.get("reason") or info.get("status") or "analyze predates it"
        return [f"* quality: **{mode}**, no perceptual baseline ({why}): exact checks"]
    mean = ", ".join(f"{k}={v}" for k, v in (info.get("mean") or {}).items())
    return [
        f"* quality: **{mode}**: perceptual gate on {info.get('samples')} samples (eager: {mean})"
    ]


def write_report(run: RunDir) -> Path:
    refresh(run)  # charts + dashboard.html, so the report embeds current images
    data = run.load()
    card = data["card"]
    tc = read_json(run.toolchain_json, {})
    baseline = read_json(run.baseline_json, {})
    plan = read_json(run.plan_json, {})
    integration = read_json(run.root / "integration.json", {})
    costs = read_json(run.root / "costs.json", {})
    gpu = tc.get("gpu") or {}
    final = integration.get("final") or {}
    compiled = strong_baseline.compiled_ms(baseline)
    reference = integration.get("reference") or {}
    metric = objective.of(baseline)  # what every ms below measures (-o metric=)

    def row(label: str, ms: Any, quality: str, eager_x: str | None = None) -> str:
        vs_eager, vs_compiled = strong_baseline.speedups(baseline, ms)
        return (
            f"| {label} | {_fmt(ms, 1)} | {eager_x or _x(vs_eager)} | {_x(vs_compiled)} "
            f"| {quality} |"
        )

    lines = [
        f"# kernel-agent report: `{card['repo_id']}`",
        "",
        f"* modality: **{card['modality']}**, architectures: {card.get('architectures')}, "
        f"params: {card.get('params')}",
        f"* GPU: {gpu.get('name')} ({gpu.get('arch')}), torch {tc.get('torch_version')}",
        f"* workload: `{baseline.get('workload')}`",
        *([f"* {line}"] if (line := objective.describe(baseline)) else []),
        *_quality_lines(data, baseline),
        "",
        "## Result",
        "",
        f"| | {metric.short} (ms) | vs eager | vs compiled | quality |",
        "|---|---|---|---|---|",
        row("baseline (eager)", baseline.get("median_ms"), "reference"),
    ]
    if compiled is not None:
        quality = strong_baseline.quality_text(
            (baseline.get("compiled_detail") or {}).get("quality")
        )
        lines.append(row("compiled baseline", compiled, quality))
    if final:
        found = dict(final.get("metrics") or {})
        held = found.pop("holdout", None)  # workloads/holdout.py: one phrase, not every metric
        natural = found.pop("natural_length", None)  # workloads/stopping.py: one phrase too
        perceived = found.pop("perceptual", None)  # workloads/perceptual.py: near-lossless
        metrics = ", ".join(f"{k}={v}" for k, v in _flat_metrics(found).items())
        if isinstance(perceived, dict):
            from kernel_agent.workloads import perceptual

            metrics = f"{perceptual.summary_text(perceived)}; {metrics}"
        if isinstance(held, dict):
            from kernel_agent.workloads.holdout import summary_text

            metrics += f"; {summary_text(held)}"
        if isinstance(natural, dict):
            from kernel_agent.workloads import stopping

            metrics += f"; {stopping.summary_text(natural)}"
        lines.append(
            row("optimised", final.get("median_ms"), metrics, f"**{final.get('speedup')}x**")
        )
    else:
        lines.append("| optimised | — | — | — | no optimisation passed end-to-end validation |")
    if reference:
        text = strong_baseline.combination_text(reference)
        lines.append(row("compiled baseline + accepted kernels", reference.get("median_ms"), text))
    lines.append("")
    if (line := strong_baseline.describe(baseline)) is not None:
        lines += [f"* {line}. *vs compiled* = against what users get without custom kernels.", ""]
    lines += _chart(run, run.root / "progress.png", "end-to-end progress")
    lines += _chart(run, run.root / "amdahl.png", "time split before and after the best kernels")
    lines += [
        "Every evaluation is a row of `results.tsv` (keep / discard / failure); "
        "`dashboard.html` shows the charts and tables, `kernel-agent status <run_dir>` a "
        "terminal summary.",
        "",
        "## Planner analysis",
        "",
        plan.get("analysis", "(no plan)"),
        "",
    ]

    units = projection.units_of(run)  # est. saved ms per run → the metric's ms
    # another metric: the estimate in its ms too (throughput: per second of audio)
    other = f" est. saved ms {metric.per} |" if metric.name != objective.LATENCY else ""
    lines += [
        "## Kernel targets",
        "",
        "| target | class | backends | evaluations | best speedup (module) "
        f"| est. saved ms/run |{other} best file |",
        "|---|---|---|---|---|---|---|" + ("---|" if other else ""),
    ]
    saved: dict[str, float | None] = {}
    for target_id in run.target_ids():
        spec = read_json(run.target(target_id) / "spec.json", {})
        records = read_jsonl(run.results_file(target_id))
        best = best_for_target(run, target_id)
        saved[target_id] = ledger._num((best or {}).get("est_saved_ms_per_run"))
        mine = f" {_fmt(units(target_id, saved[target_id]))} |" if other else ""
        lines.append(
            f"| `{target_id}` | `{spec.get('module_class')}` | "
            f"{', '.join(spec.get('backends', []))} | "
            f"{len(records)} | {_fmt(best and best.get('speedup'))} | "
            f"{_fmt(best and best.get('est_saved_ms_per_run'))} |{mine} "
            f"{best['snapshot'] if best else '—'} |"
        )
        if best:
            for case in best.get("cases", []):
                sol = (
                    f", {case['pct_of_sol']} % of SOL, {case.get('bound')}"
                    if case.get("pct_of_sol") is not None
                    else ""
                )
                lines.append(
                    f"|  ↳ `{case.get('signature', '')[:60]}` ×{case.get('calls_per_run')} | | | | "
                    f"{_fmt(case.get('speedup'))} ({_fmt(case.get('ref_ms'), 4)} → "
                    f"{_fmt(case.get('new_ms'), 4)} ms{sol}) | | |" + (" |" if other else "")
                )
    lines.append("")
    proj = projection.of_run(run, saved, ledger._num(baseline.get("median_ms")), units)
    if proj and proj.used:
        lines += [
            f"Projected from the best kernels: **{proj.projected_ms:.1f} ms** "
            f"({_x(proj.baseline_ms / max(proj.projected_ms, 1e-9))} vs eager) = baseline − "
            f"est. saved ms {metric.per} of {proj.describe()}. Nested targets count once: per "
            "instance the better of the parent's kernel and the sum of its children's.",
            "",
        ]
    elif proj and proj.unknown:
        lines += [f"Kernels {proj.describe()}.", ""]  # "not projected (why): a, b"
    for target_id in run.target_ids():
        lines += _chart(run, run.target(target_id) / "progress.png", f"{target_id} progress")
    transforms = read_jsonl(run.results_file())
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
        lines += _chart(run, run.root / "integration.png", "integration waterfall")
        if composite := integration.get("composite"):
            lines.append(
                f"* the measured combination of exp {composite.get('exp')} "
                f"({len(composite.get('items') or [])} items) "
                + ("seeded the search" if composite.get("seeded") else "was not faster")
            )
        for item in integration.get("accepted", []):
            lines.append(f"* accepted {item['kind']}: `{item['item']}`")
        for r in integration.get("recheck") or []:  # kernels/recheck.py
            lines.append(
                f"* recheck `{r.get('target')}` (`{r.get('snapshot')}`): "
                + recheck.describe(r)
                + ("" if r.get("passed") else " — refused")
            )
        for h in integration.get("history", []):
            ab = h.get("ab") or {}
            verdict = ""
            if ab.get("rounds"):  # abtest.py: paired (or separate-process) A/B against A
                verdict = f" {abtest.describe(ab)}"
                if ab.get("a_items") or len(h["items"]) > 1:  # a step of the greedy search
                    verdict += ": " + (
                        "accepted" if ab.get("accepted") else f"rejected ({ab.get('why')})"
                    )
                else:  # an item alone, against the unmodified model
                    verdict += " vs the unmodified model"
            tried = f"tried {len(h['items'])} item(s)"
            if h.get("kind") == "swap":  # another version of an accepted item
                old, new = str(h.get("old")), str(h.get("new"))
                tried = f"swap `{ledger.item_label(new)}` `{Path(old).name}` → `{Path(new).name}`"
            elif h.get("kind") == "replace":  # in the place of the items it overlaps
                olds = ", ".join(f"`{ledger.item_label(o)}`" for o in h.get("old") or [])
                tried = f"`{ledger.item_label(str(h.get('new')))}` instead of {olds}"
            lines.append(
                f"* {tried}: passed={h.get('passed')} "
                f"speedup={h.get('speedup')} {h.get('reason') or ''}{verdict}"
            )
        lines += _projection_lines(
            integration.get("projection") or [], run, baseline, integration.get("baseline_ms")
        )
        if reference:
            items = ", ".join(f"`{ledger.item_label(i)}`" for i in reference.get("items", []))
            lines.append(f"* {items}: " + strong_baseline.combination_text(reference))
    lines += report_lines(run)  # `kernel-agent improve` slices, re-integrations, rounds
    lines += library.report_lines(run)  # prior winners reused, entries stored, lessons
    if costs:
        total = sum(c.get("usd", 0) for c in costs.values())
        notional = auth.usd_note(costs)  # sessions on the Claude subscription: an estimate
        lines += [
            "",
            f"## Agent usage (total ${total:.2f}{', ' + notional if notional else ''})",
            "",
            "| agent | $ | turns | min | tools |",
            "|---|---|---|---|---|",
        ]
        for name, c in costs.items():
            tools = ", ".join(f"{k}×{v}" for k, v in c.get("tools", {}).items())
            if c.get("timed_out"):
                name += " (timed out; $ not reported)"
            lines.append(
                f"| {name} | {c.get('usd')} | {c.get('turns')} | {c.get('minutes')} | {tools} |"
            )
    stops = [
        f"* {phase}: {key.replace('_', ' ')} `{item.get('agent')}` {item.get('reason') or ''}"
        for phase, info in data.get("phases", {}).items()
        for key in ("timed_out", "budget_skipped", "usage_limit_stop")
        for item in info.get(key, [])
    ]
    if stops:
        lines += ["", "## Budget stops", "", *stops]
    lines += [
        "",
        "## Use the optimised model",
        "",
        "```python",
        f'import sys; sys.path.insert(0, "{run.optimized_dir}")',
        "from apply import apply_kernels",
        "apply_kernels(model)",
        *(
            [
                "# then the reference optimisations (they compose with the kernels): "
                + str((baseline.get("compiled_detail") or {}).get("description"))
            ]
            if reference.get("verdict") == "composes"
            else []
        ),
        "```",
        "",
    ]
    run.report.write_text("\n".join(lines))
    return run.report
