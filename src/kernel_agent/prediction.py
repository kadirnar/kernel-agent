"""Prediction error: what each accepted item was predicted to save against what the
integration measured when it added the item (issue #226).

The projection of the integration's accepted sets (:func:`kernel_agent.projection.of_sets`)
predicts every step of the greedy search: what the step adds minus what it removes, a
kernel's module-level estimate (``est_saved_ms_per_run`` in the metric's ms,
:func:`kernel_agent.objective.from_run`), a transform's gain measured alone, nested kernels
and items that change the same modules counted once. The step's paired A/B
(``integration.json`` ``history[].ab``, :mod:`kernel_agent.abtest`) measures it.
:func:`items` puts the two side by side, one record per accepted item, in the order the
integration accepted them:

* ``set``: the accepted set that added it (1 = the first); ``kind``: ``kernel``,
  ``transform`` or ``region`` (the kernel of a region target, :mod:`kernel_agent.region`);
* ``predicted_ms``: its own est. saved ms in the metric's ms (None: no estimate);
  ``counted_ms``: the part of it the projection of its set counts;
* ``step``: the step that added it: the items it added (``items``) and removed (``old``),
  its estimated gain (``predicted_ms``: the difference of the two sets projected from the
  baseline, ``step.est_gain_ms``; of the first set, baseline − its projection), its
  measured gain (``measured_ms``: A − B, the medians of its A/B, ``from_ms`` = A), the 95 %
  interval of that gain in ms (``ci95_ms``: the A/B's interval of its relative gain × A),
  whether an A/B record measured it against the set before it (``paired``) and the items
  of that set that counted there and no longer do (``instead``: it counts instead);
* ``error_ms`` = the step's measured − estimated gain, ``ratio`` = measured ÷ estimated
  (None when the step was estimated to gain nothing or to lose);
* ``shared``: the other items the step added at once (the error is the step's, shared);
  ``overlaps``: the items of its set whose modules it shares (``integrate/owners.py``:
  of such a group only the items that do not overlap count, so its ``counted_ms`` can be
  0); ``not_additive``: why its set is not projectable (:func:`projection.projectable`);
* a kernel: the ``bound`` holding most of its time and its timing ``context`` / ``l2``
  (#226) from its evaluation record, where the record has them; a region target (or a
  target that names a ``fusion``): ``fusion``, its fusion candidate's predicted saving in
  the metric's ms (``scheduler.fusion_gain``: ``profile/fusions.json`` through
  ``profiling.fusion.match``, issue #231) next to the measured gain of a step that added
  only it.

The first set is a step from the unmodified model. Its measured gain is from its A/B when
that A was the unmodified model (an item alone), else (the systems agent's combination,
measured against the best single item) from the baseline of ``analyze``: not paired, no
interval.

``report.md`` shows the table under *Prediction error* (:func:`report_lines`), the
integration rows of ``results.tsv`` the predicted saving next to the measured one
(``pred_saved_ms``, :mod:`kernel_agent.ledger`), and the kernel library keeps the errors
of every run as a lesson (``library.store_predictions``)::

    python -m kernel_agent.prediction RUN_DIR      # the section of a run's report.md
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from kernel_agent import ledger, objective, projection
from kernel_agent.workspace import RunDir, read_json, read_jsonl

KERNEL, TRANSFORM, REGION = "kernel", "transform", "region"


_num = ledger._num  # a finite number, else None


def _accepted_ab(items: Sequence[str], history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The ``ab`` record of the A/B that accepted the set ``items`` ({} when the history has
    none: an ``integration.json`` written without it)."""
    for h in reversed(history):
        ab = h.get("ab") or {}
        if ab.get("accepted") and list(h.get("items") or []) == list(items):
            return dict(ab)
    return {}


def _a_ms(ab: Mapping[str, Any]) -> float | None:
    """A's median of an A/B: its ``a_median_ms``, else the median of its ``a_ms``."""
    if (ms := _num(ab.get("a_median_ms"))) is not None:
        return ms
    times = [t for t in (_num(x) for x in ab.get("a_ms") or []) if t is not None]
    return round(statistics.median(times), 3) if times else None


def _against_model(ab: Mapping[str, Any], items: Sequence[str]) -> bool:
    """Whether an A/B measured B against the unmodified model: no ``a_items``; a record
    from before ``a_items`` existed only when B is one item (a combination of several was
    measured against the best single item)."""
    if not ab:
        return False
    return not ab.get("a_items") if "a_items" in ab else len(items) == 1


def step_of(
    entry: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    baseline_ms: float | None,
    before: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The step that accepted the set ``entry`` (an ``integration.json`` projection entry as
    :func:`projection.of_sets` writes it) from the set ``before`` it: its items, estimated
    and measured gain, the interval of the measured one (:func:`items`); ``instead``: the
    items of the set before it that counted there and no longer do (the step's item counts
    instead: the estimate of the step is less than the item's own)."""
    items = list(entry.get("items") or [])
    ab = _accepted_ab(items, history)
    s = entry.get("step")
    if s is not None:  # a later set: the set before it as measured in the same A/B
        new, old = list(s.get("new") or []), list(s.get("old") or [])
        predicted, start = _num(s.get("est_gain_ms")), _num(s.get("from_ms"))
        measured, paired = _num(s.get("measured_gain_ms")), bool(ab)
        against = list(ab.get("a_items") or [])
    else:  # the first set: from the unmodified model
        new, old = items, []
        summed = _num(entry.get("summed_ms"))
        predicted = (
            round(baseline_ms - summed, 3)
            if baseline_ms is not None and summed is not None
            else None
        )
        paired = _against_model(ab, items) and _a_ms(ab) is not None
        start = _a_ms(ab) if paired else baseline_ms
        against = [] if paired else list(ab.get("a_items") or [])
        b_ms = _num(entry.get("measured_ms"))
        measured = round(start - b_ms, 3) if start is not None and b_ms is not None else None
    ci = ab.get("ci95") if paired else None
    ci95 = None
    if start is not None and isinstance(ci, list | tuple) and len(ci) == 2:
        lo, hi = _num(ci[0]), _num(ci[1])
        if lo is not None and hi is not None:
            ci95 = [round(lo * start, 3), round(hi * start, 3)]
    was, now = (before or {}).get("counted_ms") or {}, entry.get("counted_ms") or {}
    instead = [x for x in items if x not in new and (was.get(x) or 0.0) > 0 >= (now.get(x) or 0.0)]
    return {
        "first": s is None,
        "items": new,
        "old": old,
        "instead": instead,
        "predicted_ms": predicted,
        "measured_ms": measured,
        "from_ms": start,
        "ci95_ms": ci95,
        "paired": paired,
        "against": against,  # the first set: its A/B's A when that was not the model
    }


def _error(step: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """(measured − estimated gain, measured ÷ estimated) of a step; the ratio only of a
    step estimated to gain something."""
    predicted, measured = step.get("predicted_ms"), step.get("measured_ms")
    if predicted is None or measured is None:
        return None, None
    ratio = round(measured / predicted, 3) if predicted > 0 else None
    return round(measured - predicted, 3), ratio


def items(
    entries: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
    baseline_ms: float | None,
    about: Callable[[str], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """One record per accepted item (see the module docstring), in the order the integration
    accepted them. ``entries``: the integration's ``projection`` as :func:`projection.of_sets`
    writes it (:func:`projection.accepted_sets` projects an older file again); ``history``:
    its A/Bs; ``about``: what else is known of an item (``kind``, a kernel's ``bound``,
    ``context``, ``l2``, ``fusion``; :func:`of_run` reads them from the run)."""
    out: list[dict[str, Any]] = []
    for n, entry in enumerate(entries, 1):
        before = entries[n - 2] if n > 1 else None
        step = {"set": n, **step_of(entry, history, baseline_ms, before)}
        error, ratio = _error(step)
        saved, counted = entry.get("est_saved_ms") or {}, entry.get("counted_ms") or {}
        groups = entry.get("overlaps") or []
        for item in step["items"]:
            info = dict(about(item)) if about is not None else {}
            kind = info.pop("kind", None) or (
                KERNEL if projection.kernel_target(item) else TRANSFORM
            )
            row: dict[str, Any] = {
                "set": n,
                "item": item,
                "label": ledger.item_label(item),
                "kind": kind,
                "predicted_ms": _num(saved.get(item)),
                "counted_ms": _num(counted.get(item)),
                "step": step,
                "error_ms": error,
                "ratio": ratio,
                "shared": [x for x in step["items"] if x != item],
                "overlaps": sorted(
                    {x for g in groups if item in g.get("items", []) for x in g["items"]} - {item}
                ),
            }
            if entry.get("not_additive"):
                row["not_additive"] = entry["not_additive"]
            fusion = info.pop("fusion", None)
            if fusion:  # a region's fusion candidate: its own prediction of the same step
                fusion = dict(fusion)
                alone = len(step["items"]) == 1 and not step["old"]
                f_error, f_ratio = _error({**step, "predicted_ms": fusion.get("predicted_ms")})
                fusion.update(error_ms=f_error if alone else None, ratio=f_ratio if alone else None)
                row["fusion"] = fusion
            row.update({k: v for k, v in info.items() if v is not None})
            out.append(row)
    return out


def steps(found: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The steps of :func:`items` (each once), with their error and ratio."""
    out: dict[int, dict[str, Any]] = {}
    for row in found:
        out.setdefault(
            row["set"], {**row["step"], "error_ms": row["error_ms"], "ratio": row["ratio"]}
        )
    return list(out.values())


def summary(found: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Over the steps (each once: an error shared by several items counts once): how many,
    the median and range of measured ÷ predicted, and the worst miss (the step with the
    largest absolute error in ms; the first of equal ones)."""
    every = steps(found)
    ratios = sorted(s["ratio"] for s in every if s["ratio"] is not None)
    missed = [s for s in every if s["error_ms"] is not None]
    worst = max(missed, key=lambda s: abs(s["error_ms"]), default=None)
    return {
        "steps": len(every),
        "with_ratio": len(ratios),
        "median_ratio": round(statistics.median(ratios), 3) if ratios else None,
        "min_ratio": ratios[0] if ratios else None,
        "max_ratio": ratios[-1] if ratios else None,
        "worst": worst,
    }


def _names(names: Sequence[str]) -> str:
    return ", ".join(f"`{ledger.item_label(x)}`" for x in names)


def headline(found: Sequence[Mapping[str, Any]]) -> str:
    """One line: the median ratio over the steps and the worst miss ("" without a step)."""
    s = summary(found)
    if not s["steps"]:
        return ""
    if s["median_ratio"] is None:
        text = f"Measured ÷ predicted gain: no step of {s['steps']} was predicted to gain"
    else:
        text = (
            f"Measured ÷ predicted gain: median {s['median_ratio']:.2f}x over "
            f"{s['with_ratio']} step{'' if s['with_ratio'] == 1 else 's'}"
        )
        if s["with_ratio"] > 1:
            text += f" ({s['min_ratio']:.2f}x .. {s['max_ratio']:.2f}x)"
    if (w := s["worst"]) is not None:
        ratio = f", {w['ratio']:.2f}x" if w["ratio"] is not None else ""
        text += (
            f"; the worst miss: {_names(w['items'])} (set {w['set']}): predicted "
            f"{w['predicted_ms']:.2f} ms, measured {w['measured_ms']:.2f} ms "
            f"({w['error_ms']:+.2f} ms{ratio})"
        )
    return text + "."


def _ms(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _measured(step: Mapping[str, Any]) -> str:
    text = _ms(step.get("measured_ms"))
    if ci := step.get("ci95_ms"):
        text += f" ({ci[0]:.2f} .. {ci[1]:.2f})"
    return text


def _timing(row: Mapping[str, Any]) -> str:
    """``memory-bound, timed graph / cold L2`` of a kernel row ("" when unknown)."""
    parts = [f"{row['bound']}-bound"] if row.get("bound") else []
    if row.get("context"):
        parts.append(f"timed {row['context']}" + (f" / {row['l2']} L2" if row.get("l2") else ""))
    return ", ".join(parts)


def note(row: Mapping[str, Any]) -> str:
    """The *note* cell of an item's row: the step it shares, what it replaced, its own
    estimate where the step's is another (shared, removals, counted once), a region's fusion
    prediction, the baseline a first set is measured against, its timing context."""
    step = row["step"]
    first = row["item"] == step["items"][0]
    parts = []
    if row["shared"] and first:
        parts.append(f"the step added {len(step['items'])} items: its error, shared")
    elif row["shared"]:
        parts.append(f"the same step as `{ledger.item_label(step['items'][0])}`")
    if step["old"] and first:
        parts.append(f"instead of {_names(step['old'])}")
    own, counted = row.get("predicted_ms"), row.get("counted_ms") or 0.0
    whole = step.get("predicted_ms")
    if own is None:
        parts.append("no estimate")
    elif own < 0:
        parts.append(f"est. {own:.2f} ms (slower alone)")
    elif row["shared"] or abs(counted - own) > 5e-4 or abs((whole or 0.0) - own) > 5e-4:
        part = f"est. saved {own:.2f} ms"
        if abs(counted - own) > 5e-4:
            part += f", {counted:.2f} counted"
            if row["overlaps"]:
                part += f": overlaps {_names(row['overlaps'])}"
            elif counted < own:
                part += " (nested)"
        parts.append(part)
    if first and step.get("instead"):  # they counted in the set before it, it counts now
        parts.append(f"counted instead of {_names(step['instead'])}")
    if fusion := row.get("fusion"):
        text = f"fusion `{fusion.get('id')}`: predicted {_ms(fusion.get('predicted_ms'))} ms"
        if fusion.get("ratio") is not None:
            text += f", measured ÷ predicted {fusion['ratio']:.2f}x"
        parts.append(text)
    if first and step["first"] and not step["paired"] and step.get("measured_ms") is not None:
        against = f" (its A/B's A: {_names(step['against'])})" if step.get("against") else ""
        parts.append(f"measured against the baseline of analyze{against}")
    if first and row.get("not_additive"):
        parts.append("its set is not projectable (see Integration)")
    if timing := _timing(row):
        parts.append(timing)
    return "; ".join(parts)


def table(found: Sequence[Mapping[str, Any]]) -> list[str]:
    """The markdown table of :func:`items`: a step's numbers on the row of its first item,
    ``″`` on the rows of the others it added at once."""
    lines = [
        "| set | item | kind | predicted ms | measured ms (95 % CI) | error ms "
        "| measured ÷ predicted | note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in found:
        step = row["step"]
        if row["item"] == step["items"][0]:
            ratio = "—" if row["ratio"] is None else f"{row['ratio']:.2f}x"
            error = "—" if row["error_ms"] is None else f"{row['error_ms']:+.2f}"
            cells = [_ms(step["predicted_ms"]), _measured(step), error, ratio]
        else:
            cells = ["″"] * 4
        name = f"`{row['label']}`"
        if any(ledger.item_label(x) == row["label"] for x in step["old"]):  # a version swap
            name += f" {ledger.item_version(row['item'])}"
        text = note(row).replace("|", "\\|")
        lines.append(f"| {row['set']} | {name} | {row['kind']} | {' | '.join(cells)} | {text} |")
    return lines


# ------------------------------------------------------------------ a run


def _rounds(run: RunDir) -> list[dict[str, Any]]:
    """The improve loop's rounds (``improve.json``; [] without one)."""
    state = read_json(run.root / "improve.json", {}) or {}
    rounds = state.get("rounds") if isinstance(state, dict) else None
    return [r for r in rounds or [] if isinstance(r, dict)]


def about(run: RunDir) -> Callable[[str], dict[str, Any]]:
    """What :func:`items` takes of an item besides its numbers, from the run: a kernel's
    ``kind`` (``region`` for a region target), ``target``, ``module_class``, the ``bound``
    and timing ``context`` / ``l2`` of the evaluation record of its snapshot (its last one:
    a re-evaluation stands for it); a region's (or a target naming a ``fusion``) fusion
    candidate (``scheduler.fusion_gain`` on the run's fusion tables, newest first)."""
    specs: dict[str, dict[str, Any]] = {}
    profiles: list[list[dict[str, Any]]] = []  # read once, when a region asks for them

    def info(item: str) -> dict[str, Any]:
        target = projection.kernel_target(item)
        if target is None:
            return {"kind": TRANSFORM}
        if target not in specs:
            specs[target] = read_json(run.target(target) / "spec.json", {}) or {}
        spec = specs[target]
        name = Path(item.partition("=")[2]).name
        rec: dict[str, Any] = next(
            (
                r
                for r in reversed(read_jsonl(run.results_file(target)))
                if Path(str(r.get("snapshot") or "")).name == name
            ),
            {},
        )
        out: dict[str, Any] = {
            "kind": REGION if spec.get("kind") == REGION else KERNEL,
            "target": target,
            "module_class": spec.get("module_class"),
            "bound": rec.get("bound"),
            "context": rec.get("context"),
            "l2": rec.get("l2"),
        }
        if spec.get("kind") == REGION or spec.get("fusion"):
            from kernel_agent import scheduler

            if not profiles:
                profiles.append(scheduler._profiles(run, _rounds(run)))
            gain = scheduler.fusion_gain(spec, profiles[0])
            if gain is not None:
                out["fusion"] = {
                    "id": gain.id,
                    "how": gain.how,
                    "predicted_ms": round(gain.saving_ms, 3),
                }
        return out

    return info


def of_run(run: RunDir, integration: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """:func:`items` of a run's ``integration.json`` (``integration``: already read); an
    ``integration.json`` from before #121 (or #114) is projected again first
    (:func:`projection.accepted_sets`). [] without an integration."""
    if integration is None:
        integration = read_json(run.root / "integration.json", {}) or {}
    baseline = read_json(run.baseline_json, {}) or {}
    base = _num(integration.get("baseline_ms")) or _num(baseline.get("median_ms"))
    entries = projection.accepted_sets(run, integration, base)
    return items(entries, list(integration.get("history") or []), base, about(run))


def report_lines(
    run: RunDir,
    integration: Mapping[str, Any] | None = None,
    baseline: Mapping[str, Any] | None = None,
) -> list[str]:
    """``## Prediction error`` of ``report.md`` (empty without an accepted set)."""
    found = of_run(run, integration)
    if not found:
        return []
    if baseline is None:
        baseline = read_json(run.baseline_json, {}) or {}
    metric = objective.of(dict(baseline))
    return [
        "",
        "## Prediction error",
        "",
        f"What the integration measured against what was predicted, per accepted item, in ms "
        f"{metric.per}. *predicted*: the estimated gain of the step that added the item, as "
        "the projection counts it (a kernel's module-level estimate in the metric's ms, a "
        "transform's gain measured alone; nested kernels and items that change the same "
        "modules counted once; what the step removes subtracted). *measured*: the step's A/B, "
        "A − B medians, with the 95 % interval of its gain × A. *error* = measured − "
        "predicted. A step that added several items has one error, shared.",
        "",
        headline(found),
        "",
        *table(found),
        "",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="The prediction error of a run's integration (report.md's section)."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--json", action="store_true", help="the per-item records as JSON")
    ns = parser.parse_args(argv)
    run = RunDir(ns.run_dir)
    if ns.json:
        import json

        print(json.dumps(of_run(run), indent=1, default=str))
        return 0
    lines = report_lines(run)
    print("\n".join(lines).strip() if lines else f"{run.root}: no accepted set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
