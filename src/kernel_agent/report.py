"""Markdown report for a finished (or partially finished) run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kernel_agent import (
    abtest,
    ledger,
    library,
    objective,
    precisions,
    projection,
    strong_baseline,
)
from kernel_agent.agent import auth, web
from kernel_agent.agent.tools import best_for_target
from kernel_agent.dashboard import refresh
from kernel_agent.improve import report_lines
from kernel_agent.kernels import recheck, weights
from kernel_agent.workspace import RunDir, read_json, read_jsonl


def _fmt(v: Any, nd: int = 3) -> str:
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return "—" if v is None else str(v)


def _x(v: float | None) -> str:
    return "—" if v is None else f"{v:.2f}x"


def _calls(target_id: str, spec: dict[str, Any], rec: dict[str, Any]) -> str:
    """The calls a kernel's estimate counts (``est_saved_calls``, :mod:`kernel_agent.kernels.
    weights`): those of the target's instances its cases stand for, and the calls with a
    primary input no case has; an even split over several instances is named ("": one
    instance, nothing to say)."""
    calls = rec.get("est_saved_calls") or {}
    if calls.get("basis") == weights.INSTANCE_GROUPS:
        rest, n = float(calls.get("uncovered") or 0), float(calls.get("calls") or 0)
        tail = f" (not {rest:,.0f} with a primary input no case has)" if rest else ""
        return f"`{target_id}` {n:,.0f} call{'' if n == 1 else 's'}{tail}"
    users = (spec.get("capture") or {}).get("method_instances") or {}
    if max((int(n or 0) for n in users.values()), default=1) <= 1:
        return ""
    return (
        f"`{target_id}` even split (the captured instance's calls × the instances calling "
        "its entrypoint: a capture without per-instance counts, before #119)"
    )


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


def _projected_lines(
    proj: projection.Projection | None, last: dict[str, Any] | None, metric: objective.Metric
) -> list[str]:
    """The projection of the integration's last accepted set where the run has one (#121),
    then the best kernels' alone. One that is not projectable says why, never a ratio
    (#128)."""
    if proj is None:
        return []
    lines = []
    shown = projection.shown(proj, last)
    if shown.source == projection.INTEGRATION and last is not None:
        what = "Projected for the integration's last accepted set"
        if shown.ms is None:
            lines += [f"{what}: not projectable: {shown.why}.", ""]
        else:
            how = (
                "the set before it as measured − the estimated gain of its step"
                if last.get("step")
                else "baseline − Σ est. saved ms of its items, overlaps counted once"
            )
            lines += [
                f"{what}: **{shown.ms:.1f} ms** ({_x(proj.baseline_ms / shown.ms)} vs eager, "
                f"measured {_fmt(last.get('measured_ms'), 1)} ms): {how} (see Integration).",
                "",
            ]
    nested = (
        "Nested targets count once: per instance the better of the parent's kernel and the "
        "sum of its children's."
    )
    if proj.used and proj.projected_ms is not None:
        lines += [
            f"Projected from the best kernels: **{proj.projected_ms:.1f} ms** "
            f"({_x(proj.baseline_ms / proj.projected_ms)} vs eager) = baseline − "
            f"est. saved ms {metric.per} of {proj.describe()}. {nested}",
            "",
        ]
    elif proj.used:  # module-level estimates against the eager model that exceed the run
        lines += [
            f"Projected from the best kernels (baseline − est. saved ms {metric.per}): "
            f"{proj.headline()}. Module-level estimates are taken against the eager model: "
            "they exceed the run where transforms already took their modules' time or a "
            f"capture from before #119 split the calls evenly over the instances. {nested}",
            "",
        ]
    elif proj.unknown:
        lines += [f"Kernels {proj.describe()}.", ""]  # "not projected (why): a, b"
    return lines


def _projection_lines(entries: list[dict[str, Any]], baseline: dict[str, Any]) -> list[str]:
    """Projected vs measured latency of each accepted set: the first from the baseline − Σ
    est. saved ms, every later one from the set before it − its step's estimated gain. An
    ``integration.json`` from before #121 (or #114) is projected again from its savings and
    history (:func:`projection.accepted_sets`)."""
    if not entries:
        return []
    metric = objective.of(baseline)
    lines = [
        "",
        f"Projected vs measured {metric.label} of every accepted set. The first: baseline − "
        f"Σ est. saved ms {metric.per} (a kernel's module-level estimate, a transform's gain "
        "alone; nested kernels and items that change the same modules counted once). Every "
        "later one: the set before it, as measured in its step's A/B, − the estimated gain "
        "of the step (what it adds minus what it removes):",
        "",
        "| accepted set | projected ms | measured ms | measured / projected |",
        "|---|---|---|---|",
    ]
    before: dict[str, Any] | None = None
    for p in entries:
        lines.append(_set_row(p, before))
        before = p
    return lines


def _set_row(p: dict[str, Any], before: dict[str, Any] | None) -> str:
    """The projection table's row of an accepted set. The first lists its items; a later
    one its step (``↳ + `a` instead of `b```), the estimated gain of the step next to the
    measured one, and only what changed in what is counted: the new items not counted
    (nested in or overlapping a counted item), the items they are counted instead of."""
    saved, counted = p.get("est_saved_ms") or {}, p.get("counted_ms") or {}
    items, step = p.get("items") or [], p.get("step")
    mine = set(step.get("new") or []) if step else set(items)
    once = {i for g in p.get("overlaps") or [] for i in g["items"] if i not in g["counted"]}
    unknown = [ledger.item_label(i) for i in items if i in mine and i in saved and saved[i] is None]
    nested = [  # the kernels inside (or around) a kernel that counts instead
        ledger.item_label(i) + (f" ({ms / saved[i]:.0%} counted)" if ms > 0 else "")
        for i, ms in counted.items()
        if i in mine and saved.get(i) and ms < 0.995 * saved[i] and i not in once
    ]
    overlapping = [ledger.item_label(i) for i in items if i in mine and i in once]
    note = f" (no estimate: {', '.join(unknown)})" if unknown else ""
    note += f" (not counted, nested: {', '.join(nested)})" if nested else ""
    if overlapping:
        note += f" (not counted, overlapping a counted item: {', '.join(overlapping)})"
    if step:
        was = (before or {}).get("counted_ms") or {}
        instead = [
            ledger.item_label(i)
            for i in items
            if i not in mine and (was.get(i) or 0.0) > 0 and (counted.get(i) or 0.0) <= 0
        ]
        note += f" (counted instead of {', '.join(instead)})" if instead else ""
        note += (
            f" (est. gain {_fmt(step.get('est_gain_ms'), 2)} ms, "
            f"measured {_fmt(step.get('measured_gain_ms'), 2)} ms)"
        )
    if p.get("not_additive"):
        note += f" (not additive: {p['not_additive']})"
    if step:
        names = f"↳ {_step_text(step)}"
    else:
        names = " + ".join(f"`{ledger.item_label(i)}`" for i in items)
    projected, measured = p.get("projected_ms"), p.get("measured_ms")
    ratio = measured / projected if measured and projected else None
    shown = "not additive" if p.get("not_additive") else _fmt(projected, 1)
    return f"| {names}{note} | {shown} | {_fmt(measured, 1)} | {_fmt(ratio, 2)} |"


def _step_text(step: dict[str, Any]) -> str:
    """``+ `a` instead of `b`, `c``` (a version swap: ``+ `a` #014 instead of `a` #013``)."""
    added, removed = step.get("new") or [], step.get("old") or []
    swap = {ledger.item_label(i) for i in added} & {ledger.item_label(i) for i in removed}

    def names(items: list[str]) -> str:
        return ", ".join(
            f"`{ledger.item_label(i)}`" + (f" {ledger.item_version(i)}" if swap else "")
            for i in items
        )

    return f"+ {names(added) or 'nothing'}" + (f" instead of {names(removed)}" if removed else "")


def _quality_lines(data: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """The quality mode, eager's perceptual scores next to it and the precisions the run
    allows (``near-lossless``)."""
    mode = (data.get("config") or {}).get("quality") or "exact"
    if mode == "exact":
        return []
    allowed = precisions.of_config(data.get("config"))
    line = f"* precisions allowed: {precisions.describe(allowed)} (`--precisions`)"
    info = baseline.get("perceptual") or {}
    if info.get("status") != "ok":
        why = info.get("reason") or info.get("status") or "analyze predates it"
        return [f"* quality: **{mode}**, no perceptual baseline ({why}): exact checks", line]
    mean = ", ".join(f"{k}={v}" for k, v in (info.get("mean") or {}).items())
    return [
        f"* quality: **{mode}**: perceptual gate on {info.get('samples')} samples (eager: {mean})",
        line,
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
    counted: list[str] = []  # the calls behind each estimate (kernels/weights.py)
    allowed = precisions.of_config(data.get("config"))
    refused: list[str] = []  # at a precision the run does not allow: not integrated
    for target_id in run.target_ids():
        spec = read_json(run.target(target_id) / "spec.json", {})
        records = read_jsonl(run.results_file(target_id))
        best = best_for_target(run, target_id)
        est = ledger._num((best or {}).get("est_saved_ms_per_run"))
        mark = ""
        if precisions.refusal(precisions.of_spec(spec), allowed):
            refused.append(target_id)
            mark = f" (`{precisions.of_spec(spec)}`: not allowed, skipped)"
        else:
            saved[target_id] = est
        if best and est is not None and not mark and (note := _calls(target_id, spec, best)):
            counted.append(note)
        mine = f" {_fmt(units(target_id, est))} |" if other else ""
        lines.append(
            f"| `{target_id}`{mark} | `{spec.get('module_class')}` | "
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
                calls, weight = case.get("calls_per_run"), case.get("target_calls")
                if weight is not None and calls is not None and weight != calls:
                    calls = f"{calls} (×{weight:,.0f} over the target's instances)"
                lines.append(
                    f"|  ↳ `{case.get('signature', '')[:60]}` ×{calls} | | | | "
                    f"{_fmt(case.get('speedup'))} ({_fmt(case.get('ref_ms'), 4)} → "
                    f"{_fmt(case.get('new_ms'), 4)} ms{sol}) | | |" + (" |" if other else "")
                )
    lines.append("")
    if refused:
        lines += [
            f"Not allowed in this run (precisions: {precisions.describe(allowed)}; "
            f"`--precisions`): {', '.join(f'`{t}`' for t in refused)}. Their kernels are "
            "neither projected nor integrated.",
            "",
        ]
    if counted:
        lines += [
            "Calls behind the estimates (a case's gain per call × the calls of the target's "
            "instances it stands for): " + "; ".join(counted) + ".",
            "",
        ]
        if hint := projection.recapture_hint(run):
            lines += [hint[0].upper() + hint[1:] + ".", ""]
    base_ms = ledger._num(baseline.get("median_ms"))
    proj = projection.of_run(run, saved, base_ms, units)
    sets = projection.accepted_sets(run, integration, base_ms)  # an old one projected again
    lines += _projected_lines(proj, sets[-1] if sets else None, metric)
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
            note = rec.get("reason") or rec.get("status") or ""
            if (oom := abtest.step_out_of_memory(rec)) is not None:  # no quality verdict (#137)
                note = f"not measurable: {oom}"
            lines.append(
                f"| {', '.join(rec.get('transforms', []))} | {rec.get('passed')} | "
                f"{_fmt(rec.get('median_ms'), 1)} | {_fmt(rec.get('speedup'))} | {note[:80]} |"
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
        for s in integration.get("skipped") or []:  # precisions.py: not allowed in this run
            lines.append(f"* skipped `{s.get('target')}`: {s.get('reason')}")
        for item in integration.get("accepted", []):
            lines.append(f"* accepted {item['kind']}: `{item['item']}`")
        for r in integration.get("recheck") or []:  # kernels/recheck.py
            lines.append(
                f"* recheck `{r.get('target')}` (`{r.get('snapshot')}`): "
                + recheck.describe(r)
                + ("" if r.get("passed") else " — refused")
            )
            if r.get("status") == "memcheck" and (r.get("memcheck") or {}).get("report"):
                lines += ["", "  ```", *[f"  {x}" for x in r["memcheck"]["report"].splitlines()]]
                lines += ["  ```"]
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
            if abtest.out_of_memory(ab.get("fallback")):  # the in-process A/B did not fit
                verdict += " (retried in separate processes: out of GPU memory in one)"
            outcome = f"passed={h.get('passed')} speedup={h.get('speedup')} {h.get('reason') or ''}"
            if (oom := abtest.step_out_of_memory(h)) is not None:  # no quality verdict (#137)
                outcome = f"**not measurable**: {oom} (a re-integration measures it again)"
            tried = f"tried {len(h['items'])} item(s)"
            if h.get("kind") == "swap":  # another version of an accepted item
                old, new = str(h.get("old")), str(h.get("new"))
                tried = f"swap `{ledger.item_label(new)}` `{Path(old).name}` → `{Path(new).name}`"
            elif h.get("kind") == "replace":  # in the place of the items it overlaps
                olds = ", ".join(f"`{ledger.item_label(o)}`" for o in h.get("old") or [])
                tried = f"`{ledger.item_label(str(h.get('new')))}` instead of {olds}"
            lines.append(f"* {tried}: {outcome}{verdict}")
        lines += _projection_lines(sets, baseline)
        if reference:
            items = ", ".join(f"`{ledger.item_label(i)}`" for i in reference.get("items", []))
            lines.append(f"* {items}: " + strong_baseline.combination_text(reference))
    lines += report_lines(run)  # `kernel-agent improve` slices, re-integrations, rounds
    lines += library.report_lines(run)  # prior winners reused, entries stored, lessons
    lines += web.report_lines(run.root)  # pages the agents fetched (issue #125)
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
