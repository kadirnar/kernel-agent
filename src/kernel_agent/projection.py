"""Projected end-to-end latency from the best kernels, nested targets counted once.

``projected = baseline − Σ est. saved ms`` over targets that do not overlap.
Targets nest: VoxCPM's ``decoder_layer_fused`` (``MiniCPMDecoderLayer``) contains
``attn_fused``, ``mlp_fused`` and ``rmsnorm_fused``, and the layer's kernel
replaces theirs, so a plain sum counts the attention twice.

The module tree comes from instance qualnames (:func:`tree`). A target's
instances form *groups*, one per qualname with the layer indices folded
(``model.base_lm.layers.*.self_attn``, :func:`fold`). A group sits inside another
target's group when that group's pattern is a prefix of its own. Each group gets
its target's saving × its share of the target's instances. Then, from the
innermost groups outwards, a group counts ``max(its own saving, Σ of what the
groups inside it count)`` (:func:`project`): per parent instance the better of
{the parent kernel alone, its children}, the best non-overlapping choice. A
parent whose instances hold only some of a child's instances (``VoxCPMLocEnc``
holds 12 of the 60 decoder layers) replaces only that share of the child.

Instance qualnames, in order of preference:

* ``profile.json`` ``classes[].groups`` (pattern → instances);
* older profiles: the class's per-phase ``groups``, its ``example_qualname`` and
  the instance each of its targets captured (``spec.json`` ``capture.qualname``),
  with the class's instances split evenly over the patterns without a count;
* a class that is in no profile (one only an improve round re-profiled is looked
  up in ``rounds/*/profile``): its captured instance, else a group that contains
  nothing and lies in nothing.

A target's scope narrows its instances: ``qualname`` (one instance: it holds that
share of every group inside its pattern, ``inside``), ``qualname_regex`` (searched
in the folded pattern, with ``*`` also read as ``0``) and ``phase`` (that phase's
groups, when the profile has them). A region target (``kind: region``,
:mod:`kernel_agent.region`) lies inside the instances of its ``parent_class``.

A target's saving is spread over its groups by the calls its cases stand for in each
(the capture's ``instance_groups``, :mod:`kernel_agent.kernels.weights`): VoxCPM2's
LocEnc layers run shapes no case of a LocDiT layer's capture has, so the LocDiT group
holds all of that saving. Captures from before #119: evenly over the instances.

Approximations:

* a capture from before #119 spreads a target's saving evenly over its instances.
  Its estimate scaled the captured instance's gain per call by the instances that call
  each entrypoint (the even split), so instances that only run other shapes get a share;
* two targets on the same instances with different ``phase`` add up; a
  phase-specific parent still replaces all of a child's saving inside it;
* a region overlaps none of its parent's other children;
* every saving is still a module-level estimate: what shows up end to end is
  what the integration measures.

Units (:class:`Units`): a kernel's est. saved ms is per run of the workload, the baseline
in the run's metric (:mod:`kernel_agent.objective`). Every projection of kernel savings
converts them first (:func:`kernel_agent.objective.from_run`): as they are for
``latency``, ÷ the seconds of audio of a batched run for ``throughput`` (ms per second of
generated audio), their share inside the first-audio window for ``ttfa``
(:func:`window`). A saving with no value in the metric is not projected (``unknown``).

Accepted sets of the integration (:func:`of_sets`, ``integration.json`` ``projection``)
also hold transforms, whose saving is their gain measured alone. Items whose modules
overlap count once (#121): the modules each item changed (``integrate/owners.py``) put a
kernel and the transforms of its modules, or two transforms of one module, in one group,
and each group counts its best items that do not overlap. Every set after the first is
projected from the set before it, as measured, minus the estimated gain of its step.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kernel_agent import ledger, objective
from kernel_agent.integrate import owners as owners_mod
from kernel_agent.kernels import weights
from kernel_agent.workspace import RunDir, read_json

_INDEX = re.compile(r"\.\d+(?=\.|$)")


def fold(qualname: str) -> str:
    """``qualname`` with layer indices folded: ``model.layers.3.mlp`` → ``model.layers.*.mlp``."""
    return _INDEX.sub(".*", qualname)


@dataclass(frozen=True)
class Group:
    """Instances of one target that share a folded qualname."""

    target: str
    pattern: str  # "" when unknown: it contains nothing and lies in nothing
    share: float  # of the target's instances
    phase: str | None = None
    inside: float = 1.0  # of the instances with this pattern (one instance of four: 0.25)


@dataclass(frozen=True)
class Tree:
    groups: tuple[Group, ...] = ()
    parents: tuple[int, ...] = ()  # index of the innermost enclosing group, -1 at the top

    def nested(self) -> bool:
        return any(p >= 0 for p in self.parents)

    def holders(self) -> dict[str, set[str]]:
        """Target → the other targets whose groups hold one of its groups, at any depth."""
        out: dict[str, set[str]] = {g.target: set() for g in self.groups}
        for i, g in enumerate(self.groups):
            j = self.parents[i]
            while j >= 0:
                if self.groups[j].target != g.target:
                    out[g.target].add(self.groups[j].target)
                j = self.parents[j]
        return out


@dataclass(frozen=True)
class Projection:
    baseline_ms: float
    savings: dict[str, float]  # each target's own est. saved ms (> 0), in the metric's ms
    counted: dict[str, float]  # the part of it the projection counts
    unknown: tuple[str, ...] = ()  # a saving per run with no value in the metric: left out
    unknown_why: str = ""  # why (objective.unknown_why)

    @property
    def saved_ms(self) -> float:
        return sum(self.counted.values())

    @property
    def projected_ms(self) -> float:
        return round(max(self.baseline_ms - self.saved_ms, 0.0), 3)

    @property
    def used(self) -> list[str]:
        """Targets counted (in full or in part), biggest saving first."""
        return sorted((t for t, ms in self.counted.items() if ms > 0), key=self._order)

    @property
    def left_out(self) -> list[str]:
        """Targets with a saving that is not counted at all (nested in a better choice)."""
        return sorted((t for t in self.savings if self.counted.get(t, 0.0) <= 0), key=self._order)

    def part(self, target: str) -> float:
        """Share of a target's saving that is counted (1.0: all of it)."""
        own = self.savings.get(target, 0.0)
        return min(self.counted.get(target, 0.0) / own, 1.0) if own > 0 else 0.0

    @property
    def shown(self) -> bool:
        """Whether there is anything to say: a counted saving or one left out as unknown."""
        return bool(self.used or self.unknown)

    def describe(self) -> str:
        """``a + b (40 %) + c; not counted (nested): d, e; not projected (why): f`` ("" without
        a saving)."""
        used = [t if self.part(t) > 0.995 else f"{t} ({self.part(t):.0%})" for t in self.used]
        parts = [" + ".join(used)] if used else []
        if self.left_out:
            parts.append("not counted (nested): " + ", ".join(self.left_out))
        if self.unknown:
            parts.append(f"not projected ({self.unknown_why}): " + ", ".join(self.unknown))
        return "; ".join(parts)

    def headline(self) -> str:
        """``projected from`` :meth:`describe`; only what is not projected without a saving."""
        return f"projected from {self.describe()}" if self.used else self.describe()

    def as_dict(self) -> dict[str, Any]:
        return {
            "projected_ms": self.projected_ms,
            "saved_ms": round(self.saved_ms, 3),
            "counted": {t: round(ms, 3) for t, ms in self.counted.items()},
            "left_out": self.left_out,
            "unknown": list(self.unknown),
            "label": self.describe(),
        }

    def _order(self, target: str) -> tuple[float, str]:
        return (-self.counted.get(target, 0.0), target)


# ------------------------------------------------------------------ projection


def project(
    tree: Tree,
    saved: Mapping[str, float | None],
    baseline_ms: float,
    units: Units | None = None,
) -> Projection:
    """``baseline − Σ saved`` over the best set of targets that do not overlap.

    ``saved``: est. saved ms of each target (its best kept result) in the metric's ms, or
    per run of the workload with ``units``, which converts them (a saving it cannot
    convert is left out: ``unknown``). A target that is not in the tree counts in full."""
    unknown: list[str] = []
    if units is not None:
        per_run, saved = saved, units.convert(saved)
        unknown = sorted(t for t, ms in per_run.items() if ms and ms > 0 and saved[t] is None)
    savings = {t: float(ms) for t, ms in saved.items() if ms is not None and ms > 0}
    groups, parents = tree.groups, tree.parents
    own = [savings.get(g.target, 0.0) * g.share for g in groups]
    children: list[list[int]] = [[] for _ in groups]
    roots: list[int] = []
    for i, parent in enumerate(parents):
        (children[parent] if parent >= 0 else roots).append(i)
    # What each group's subtree saves at best, innermost first. A group that holds only
    # some of the instances with its pattern holds that share of the groups inside it.
    value = list(own)
    for i in sorted(range(len(groups)), key=lambda i: -_depth(parents, i)):
        inner, part = sum(value[c] for c in children[i]), groups[i].inside
        value[i] = max(own[i], part * inner) + (1.0 - part) * inner
    counted: dict[str, float] = {}
    todo = [(i, 1.0) for i in roots]  # (group, share of its instances not taken yet)
    while todo:
        i, free = todo.pop()
        inner, part = sum(value[c] for c in children[i]), groups[i].inside
        if own[i] > 0 and own[i] >= part * inner:  # the parent (on a tie: one kernel, not many)
            counted[groups[i].target] = counted.get(groups[i].target, 0.0) + free * own[i]
            free *= 1.0 - part
        if free > 0:
            todo += [(c, free) for c in children[i]]
    in_tree = {g.target for g in groups}
    for target, ms in savings.items():
        if target not in in_tree:
            counted[target] = ms
    why = units.why if units is not None and unknown else ""
    return Projection(baseline_ms, savings, counted, tuple(unknown), why)


def series(
    tree: Tree, baseline_ms: float, rows: Iterable[dict[str, Any]], units: Units | None = None
) -> list[tuple[dict[str, Any], Projection]]:
    """The projection after every kept or re-evaluated kernel row, each target at its best
    result that stands (:func:`ledger.standing`, as in :func:`ledger.summary`); ``units``
    converts the rows' est. saved ms per run to the metric (:func:`project`)."""
    mine: dict[str, list[dict[str, Any]]] = {}
    saved: dict[str, float | None] = {}
    out = []
    for row in rows:
        if row["target"] == ledger.E2E:
            continue
        mine.setdefault(row["target"], []).append(row)
        if row["status"] in (ledger.KEEP, ledger.REEVALUATED):
            stand = ledger.standing(mine[row["target"]])
            best = max(stand, key=lambda r: r["speedup"] or 0.0, default=None)
            saved[row["target"]] = best.get("est_saved_ms") if best else None
            out.append((row, project(tree, saved, baseline_ms, units)))
    return out


def of_run(
    run: RunDir,
    saved: Mapping[str, float | None],
    baseline_ms: float | None,
    units: Units | None = None,
) -> Projection | None:
    """The run's projection of ``saved`` (est. saved ms per run of each target), converted
    with ``units`` (default: the run's, :func:`units_of`)."""
    if baseline_ms is None:
        return None
    return project(tree(run), saved, baseline_ms, units or units_of(run))


def kernel_target(item: str) -> str | None:
    """The target of an integration item that is a kernel (``target=path``), else None (a
    transform's path)."""
    target, sep, _ = item.partition("=")
    return target if sep and "/" not in target else None


#: ``integration.json`` ``projection[].est_saved_unit``: every saving is in the metric's ms
#: (since #114; an entry without it holds a kernel's ms per run: :func:`in_metric`).
SAVED_UNIT = "metric"


def in_metric(
    entry: Mapping[str, Any], tree: Tree, units: Units, baseline_ms: float
) -> dict[str, Any]:
    """An ``integration.json`` projection entry with every saving in the metric's ms. One
    written before #114 (no ``est_saved_unit``) holds its kernels' est. saved ms per run
    of the workload: :func:`of_set` projects it again with them converted (``units``)."""
    if entry.get("est_saved_unit") == SAVED_UNIT:
        return dict(entry)
    saved: dict[str, float | None] = {}
    for item, ms in (entry.get("est_saved_ms") or {}).items():
        target = kernel_target(item)
        value = units(target, ms) if target is not None else ms
        saved[item] = None if value is None else round(value, 3)
    return {**entry, **of_set(tree, saved, baseline_ms), "est_saved_unit": SAVED_UNIT}


def of_set(
    tree: Tree,
    saved: Mapping[str, float | None],
    baseline_ms: float,
    changes: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, Any]:
    """Projected ms of an integration's accepted set from the baseline: ``saved`` maps each
    item (a kernel's ``target=path``, a transform's path) to its est. saved ms in the
    metric's ms (a kernel's module-level estimate, :class:`Units`; a transform's measured
    gain alone). Kernels nest by target (the tree). Items whose modules overlap count once
    (``changes``: the modules each item touched or owns, ``integrate/owners.py``;
    :func:`owners.common <kernel_agent.integrate.owners.common>`, #121): of each group of
    them, the items that do not overlap with the largest saving (of two, the larger).
    A slower item (a negative saving) is added back, unless it overlaps another: its time
    alone says nothing of what it does with them. ``counted_ms``: the part of each saving
    counted; ``overlaps``: each group, the items counted and where they overlap."""
    key = {a: kernel_target(a) or a for a in saved}
    proj = project(tree, {key[a]: v for a, v in saved.items()}, baseline_ms)
    counted = {
        a: proj.counted.get(key[a], 0.0) for a, v in saved.items() if v is not None and v > 0
    }
    pairs = _overlaps(list(saved), changes or {})
    groups: list[dict[str, Any]] = []
    for members in _components(list(saved), pairs):
        inner = {p: w for p, w in pairs.items() if p[0] in members}
        keep = _once({a: counted.get(a, 0.0) for a in members}, inner)
        for a in members:
            if a in counted and a not in keep:
                counted[a] = 0.0
        order = sorted(members, key=lambda a: -(saved[a] or 0.0))
        where = owners_mod.topmost(m for w in inner.values() for m in w)
        groups.append({"items": order, "counted": [a for a in order if a in keep], "where": where})
    grouped = {a for g in groups for a in g["items"]}
    slower = sum(v for a, v in saved.items() if v is not None and v < 0 and a not in grouped)
    out: dict[str, Any] = {
        "projected_ms": round(baseline_ms - sum(counted.values()) - slower, 3),
        "est_saved_ms": dict(saved),
        "counted_ms": {a: round(ms, 3) for a, ms in counted.items()},
    }
    if groups:
        out["overlaps"] = groups
    return out


def of_sets(
    tree: Tree,
    sets: Sequence[Mapping[str, Any]],
    baseline_ms: float,
    changes: Mapping[str, Iterable[str]] | None = None,
) -> list[dict[str, Any]]:
    """``integration.json`` ``projection``: projected vs measured ms of every accepted set,
    in the order the integration accepted them (each with ``items``, ``est_saved_ms`` in the
    metric's ms, ``measured_ms`` and ``from_ms``: the set before it, measured in the A/B of
    its step).

    ``summed_ms`` projects a set from the baseline (:func:`of_set`, overlaps counted once).
    ``projected_ms`` of the first set is the same; of every later one it is the set before
    it as measured − the estimated gain of its step (``step``: the items it adds and
    removes, ``est_gain_ms`` = the difference of the two sets' ``summed_ms``, next to its
    ``measured_gain_ms``). So the estimate of the new item meets what it gained on top of
    the others, instead of the others' gains alone, which do not add up (#121). A
    projection at or below 0 ms counts more saving than there is time to save:
    ``projected_ms`` is None and ``not_additive`` says why."""
    out: list[dict[str, Any]] = []
    for s in sets:
        saved = s.get("est_saved_ms") or {}
        entry: dict[str, Any] = {
            "items": list(s.get("items") or []),
            **of_set(tree, saved, baseline_ms, changes),
        }
        entry.update(summed_ms=entry["projected_ms"], measured_ms=s.get("measured_ms"))
        entry["est_saved_unit"] = SAVED_UNIT
        if out:
            _step(entry, out[-1], s.get("from_ms"), changes or {})
        elif entry["summed_ms"] <= 0:
            counted = entry["counted_ms"]
            big = sorted((a for a, ms in counted.items() if ms > 0), key=lambda a: -counted[a])
            entry["projected_ms"] = None
            entry["not_additive"] = (
                f"the savings counted ({sum(counted.values()):.1f} ms) are more than the "
                f"baseline ({baseline_ms:.1f} ms); the largest: "
                + _few([f"{ledger.item_label(a)} {counted[a]:.1f} ms" for a in big])
            )
        out.append(entry)
    return out


def of_integration(
    data: Mapping[str, Any], tree: Tree, units: Units, baseline_ms: float
) -> list[dict[str, Any]]:
    """``integration.json`` ``projection`` as :func:`of_sets` writes it. An entry written
    before #121 (no ``summed_ms``) is projected again: its savings in the metric's ms
    (:func:`in_metric`), the modules each item changed and the A of each step from the
    file's ``history``."""
    entries = [dict(p) for p in data.get("projection") or []]
    if all(current(p) for p in entries):
        return entries
    history = list(data.get("history") or [])
    starts = {  # the A/B that accepted each set: its A is the set before it
        tuple(h.get("items") or []): (h.get("ab") or {}).get("a_median_ms")
        for h in history
        if (h.get("ab") or {}).get("accepted")
    }
    sets = [
        {**p, "from_ms": starts.get(tuple(p.get("items") or []))}
        for p in (in_metric(p, tree, units, baseline_ms) for p in entries)
    ]
    return of_sets(tree, sets, baseline_ms, owners_mod.Owners.of_history(history).changes())


def current(entry: Mapping[str, Any]) -> bool:
    """Whether an ``integration.json`` projection entry is as :func:`of_sets` writes it
    (savings in the metric's ms, #114; overlaps counted once, ``summed_ms``, #121)."""
    return entry.get("est_saved_unit") == SAVED_UNIT and "summed_ms" in entry


def _step(
    entry: dict[str, Any],
    before: Mapping[str, Any],
    from_ms: float | None,
    changes: Mapping[str, Iterable[str]],
) -> None:
    """``entry``'s ``step`` from the set ``before`` it, and its ``projected_ms`` from there
    (:func:`of_sets`)."""
    new = [a for a in entry["items"] if a not in before["items"]]
    old = [a for a in before["items"] if a not in entry["items"]]
    start = from_ms if from_ms is not None else before.get("measured_ms")
    est = round(before["summed_ms"] - entry["summed_ms"], 3)
    measured = entry.get("measured_ms")
    gain = round(start - measured, 3) if start is not None and measured is not None else None
    entry["step"] = {
        "new": new,
        "old": old,
        "from_ms": start,
        "est_gain_ms": est,
        "measured_gain_ms": gain,
    }
    if start is None:
        entry["projected_ms"] = None
    elif start - est > 0:
        entry["projected_ms"] = round(start - est, 3)
    else:
        entry["projected_ms"] = None
        why = f"the estimated gain of {_few([ledger.item_label(a) for a in new])} ({est:.1f} ms)"
        why += f" is more than the {start:.1f} ms of the set before it"
        rest = [a for a in entry["items"] if a not in new]
        pairs = _overlaps([*new, *rest], changes)
        saved = entry.get("est_saved_ms") or {}
        others = sorted(
            {b for a, b in pairs if a in new and b not in new}, key=lambda b: -(saved[b] or 0.0)
        )
        if others:
            where = owners_mod.topmost(m for (a, b), w in pairs.items() if a in new for m in w)
            why += f", where it overlaps {_few([ledger.item_label(b) for b in others])}"
            why += f" (in {_few(where, 2)})"
        entry["not_additive"] = why


def _overlaps(
    items: Sequence[str], changes: Mapping[str, Iterable[str]]
) -> dict[tuple[str, str], list[str]]:
    """The pairs of ``items`` whose savings overlap, with where (:func:`owners.common
    <kernel_agent.integrate.owners.common>`): a kernel and a transform, or two transforms.
    Two kernels nest in the tree instead, which counts the instances they share."""
    out: dict[tuple[str, str], list[str]] = {}
    for i, a in enumerate(items):
        for b in items[i + 1 :]:
            if kernel_target(a) and kernel_target(b):
                continue
            if where := owners_mod.common(changes.get(a, ()), changes.get(b, ())):
                out[(a, b)] = where
    return out


def _components(items: Sequence[str], pairs: Iterable[tuple[str, str]]) -> list[list[str]]:
    """The groups of ``items`` that overlap, directly or through others (two or more)."""
    group = {a: {a} for a in items}
    for a, b in pairs:
        if group[a] is not group[b]:
            merged = group[a] | group[b]
            for x in merged:
                group[x] = merged
    out: list[list[str]] = []
    for a in items:
        if len(group[a]) > 1 and not any(a in g for g in out):
            out.append([x for x in items if x in group[a]])
    return out


def _once(value: Mapping[str, float], pairs: Iterable[tuple[str, str]]) -> set[str]:
    """Of a group of overlapping items, those counted: the items that do not overlap with
    the largest total saving (on a tie, the fewer), as the tree takes the better of a
    parent and what lies inside it. Groups are small (one per module of the model)."""
    near: dict[str, set[str]] = {a: set() for a in value}
    for a, b in pairs:
        near[a].add(b)
        near[b].add(a)

    @functools.cache
    def best(free: frozenset[str]) -> tuple[float, int, tuple[str, ...]]:
        if not free:
            return (0.0, 0, ())
        v = max(free, key=lambda a: (len(near[a] & free), value[a], a))
        if not near[v] & free:  # none of them overlaps another one any more: all count
            return (round(sum(value[a] for a in free), 9), -len(free), tuple(sorted(free)))
        rest = best(free - {v} - near[v])
        take = (round(rest[0] + value[v], 9), rest[1] - 1, tuple(sorted((*rest[2], v))))
        return max(best(free - {v}), take)

    return set(best(frozenset(a for a, ms in value.items() if ms > 0))[2])


def _few(names: Sequence[str], n: int = 3) -> str:
    """The first ``n`` of ``names``, and how many more."""
    return ", ".join(names[:n]) + (f" (+{len(names) - n} more)" if len(names) > n else "")


# ------------------------------------------------------------------ units


@dataclass(frozen=True)
class Units:
    """A target's module-level estimate, ms per run of the workload, in the run's metric
    (:func:`kernel_agent.objective.from_run`): ``baseline`` is its ``baseline.json``,
    ``windows`` (``metric=ttfa``) each target's share inside the first-audio window
    (:func:`window`). ``Units()`` keeps ms per run (the latency)."""

    baseline: dict[str, Any] = field(default_factory=dict)
    windows: dict[str, float] = field(default_factory=dict)

    def __call__(self, target: str, ms: float | None) -> float | None:
        return objective.from_run(ms, self.baseline, self.windows.get(target))

    def convert(self, saved: Mapping[str, float | None]) -> dict[str, float | None]:
        return {t: self(t, ms) for t, ms in saved.items()}

    def factor(self, target: str) -> float | None:
        """The metric's ms per ms per run of ``target`` (None: unknown)."""
        return self(target, 1.0)

    def of_row(self, row: Mapping[str, Any]) -> float | None:
        """A ledger row's est. saved ms in the metric: a kernel row's estimate converted, an
        ``e2e`` row's measured gain (baseline − new) as it is."""
        ms = row.get("est_saved_ms")
        return ms if row.get("target") == ledger.E2E else self(str(row.get("target")), ms)

    @property
    def why(self) -> str:
        return objective.unknown_why(self.baseline)


def units_of(
    run: RunDir, targets: Iterable[str] | None = None, baseline: dict[str, Any] | None = None
) -> Units:
    """The :class:`Units` of a run (``metric=ttfa``: with the :func:`window` of each of
    ``targets``, default every target); ``baseline``: its ``baseline.json`` when already
    read (the orchestrator's, verified)."""
    if baseline is None:
        baseline = read_json(run.baseline_json, {}) or {}
    if objective.of(baseline).name != objective.TTFA:
        return Units(baseline)
    ids = run.target_ids() if targets is None else list(targets)
    profiles = [read_json(p, {}) or {} for p in _profile_paths(run)]
    shares = {t: window(read_json(run.target(t) / "spec.json", {}) or {}, profiles) for t in ids}
    return Units(baseline, {t: s for t, s in shares.items() if s is not None})


def window(spec: Mapping[str, Any], profiles: Sequence[Mapping[str, Any]]) -> float | None:
    """``metric=ttfa``: the share of a target's est. saved ms per run inside the first-audio
    window, or None when the profiles cannot tell.

    The estimate covers the calls of a full streamed run (the capture runs it whole): the
    calls its cases stand for, as the evaluator weights them (:mod:`kernel_agent.kernels.
    weights`). The profiles are taken inside the window (``Workload.metric_window``):
    the calls of the target's class there (a region: of its parent class), in its
    ``phase``; with a ``qualname_regex``, of the instance groups it matches
    (``classes[].work``; None without them). A class that no profile saw makes no call
    before the first audio. Of these, the share of the run's calls that a case stands for
    counts (``coverage``: the instance groups of a capture since #119). The share = calls
    in the window ÷ calls covered (at most 1): the estimate's gain per call × the calls
    inside the window."""
    capture = spec.get("capture") or {}
    covered = sum(weights.weights(capture))
    calls = _window_calls(spec, profiles)
    if covered <= 0 or calls is None:
        return None
    coverage = weights.coverage(capture)
    return min(calls * (1.0 if coverage is None else coverage) / covered, 1.0)


def _window_calls(spec: Mapping[str, Any], profiles: Sequence[Mapping[str, Any]]) -> float | None:
    """Calls of a target's instances in the profiled window (:func:`window`)."""
    region = spec.get("kind") == "region"
    cls = str(spec.get("parent_class") if region else spec.get("module_class"))
    phase, regex = spec.get("phase") or None, spec.get("qualname_regex")
    for profile in profiles:
        entries = [c for c in profile.get("classes") or [] if c.get("cls") == cls]
        if not entries:
            continue
        if regex:
            try:
                matcher = re.compile(str(regex))
            except re.error:
                return None
            work = [w for c in entries for w in c.get("work") or []]
            if not work:  # calls per instance group unknown (a profile from before #90)
                return None
            mine = [w for w in work if _matches(matcher, str(w.get("group") or ""))]
            return _calls(w for w in mine if phase is None or w.get("phase") == phase)
        if phase:
            return _calls((c.get("phases") or {}).get(phase) or {} for c in entries)
        return _calls(entries)
    return 0.0 if any(p.get("classes") for p in profiles) else None


def _calls(stats: Iterable[Mapping[str, Any]]) -> float:
    return float(sum(int(s.get("calls") or 0) for s in stats))


def _matches(matcher: re.Pattern[str], pattern: str) -> bool:
    """A ``qualname_regex`` on a folded qualname (``*`` also read as ``0``)."""
    return bool(matcher.search(pattern) or matcher.search(pattern.replace("*", "0")))


def _depth(parents: Sequence[int], i: int) -> int:
    depth = 0
    while parents[i] >= 0:
        i, depth = parents[i], depth + 1
    return depth


# ------------------------------------------------------------------ the module tree


def tree(run: RunDir, targets: Iterable[str] | None = None) -> Tree:
    """The tree of the targets' instances, from their specs and the run's profile(s)."""
    ids = run.target_ids() if targets is None else list(targets)
    specs = {t: read_json(run.target(t) / "spec.json", {}) or {} for t in ids}
    return build(specs, [read_json(p, {}) or {} for p in _profile_paths(run)])


def _profile_paths(run: RunDir) -> list[Path]:
    """The run's profile, then those of its improve rounds."""
    paths = [run.profile_dir / "profile.json"]
    return paths + sorted((run.root / "rounds").glob("*/profile/profile.json"))


def build(specs: Mapping[str, Mapping[str, Any]], profiles: Sequence[Mapping[str, Any]]) -> Tree:
    """:func:`tree` from target specs and profiles (the first profile with a class wins)."""
    classes: dict[str, list[Mapping[str, Any]]] = {}
    for profile in profiles:
        found: dict[str, list[Mapping[str, Any]]] = {}
        for c in profile.get("classes") or []:
            found.setdefault(str(c.get("cls")), []).append(c)
        for cls, entries in found.items():
            classes.setdefault(cls, entries)
    captured: dict[str, set[str]] = {}  # class → instances its targets captured
    for spec in specs.values():
        qualname = (spec.get("capture") or {}).get("qualname") or spec.get("qualname")
        if qualname and spec.get("module_class"):
            captured.setdefault(str(spec["module_class"]), set()).add(str(qualname))

    groups: list[Group] = []
    for target_id, spec in specs.items():
        phase = spec.get("phase") or None
        region = spec.get("kind") == "region" and spec.get("parent_class")
        cls = str(spec.get("parent_class") if region else spec.get("module_class"))
        instances = _scoped(spec, _instances(classes.get(cls, []), captured.get(cls, set()), phase))
        for pattern, (share, inside) in _shares(spec, instances, bool(region)).items():
            if region and pattern:
                pattern = f"{pattern}.[{target_id}]"  # inside each parent instance
            groups.append(Group(target_id, pattern, share, phase, inside))
    return of_groups(groups)


def _shares(
    spec: Mapping[str, Any], instances: dict[str, tuple[float, float]], region: bool
) -> dict[str, tuple[float, float]]:
    """Pattern → (share of the target's saving, ``inside``): by the calls its cases stand
    for in each instance group (the capture's ``instance_groups``, #119), else by
    instances."""
    calls = {} if region else weights.group_calls(spec.get("capture") or {})
    total = sum(calls.get(p, 0.0) for p in instances)
    if total > 0:
        return {p: (calls.get(p, 0.0) / total, inside) for p, (_, inside) in instances.items()}
    total = sum(n for n, _ in instances.values())
    return {p: (n / total, inside) for p, (n, inside) in instances.items()}


def of_groups(groups: Iterable[Group]) -> Tree:
    """The tree of explicit groups (:func:`build` without the specs; e.g. the ceilings
    table's rows, ``profiling/ceilings.py``)."""
    ordered = sorted(groups, key=lambda g: g.pattern.count("."))  # outer groups first
    return Tree(tuple(ordered), tuple(_parent(ordered, i) for i in range(len(ordered))))


def _instances(
    entries: list[Mapping[str, Any]], known: set[str], phase: str | None
) -> dict[str, float]:
    """Folded qualname → instances of a class (the profile entries of one class name)."""
    total = sum(int(e.get("instances") or 0) for e in entries)
    by_phase = [((e.get("phases") or {}).get(phase) or {}) for e in entries] if phase else []
    counts: dict[str, float] = {}
    if by_phase and all(p.get("groups") for p in by_phase):
        total = sum(int(p.get("instances") or 0) for p in by_phase)
        for p in by_phase:
            for pattern, n in p["groups"].items():
                counts[pattern] = counts.get(pattern, 0) + n
    elif entries and all(e.get("groups") for e in entries):
        for e in entries:
            for pattern, n in e["groups"].items():
                counts[pattern] = counts.get(pattern, 0) + n
    else:  # an older profile: what is known of where the instances are
        for e in entries:
            for info in (e.get("phases") or {}).values():
                for pattern, n in (info.get("groups") or {}).items():
                    counts[pattern] = max(counts.get(pattern, 0), n)
        for qualname in [*(e.get("example_qualname") for e in entries), *sorted(known)]:
            if qualname:
                counts.setdefault(fold(str(qualname)), 0)
    missing = total - sum(counts.values())
    if missing > 0 and counts:
        spread = [p for p, n in counts.items() if n == 0] or list(counts)
        for pattern in spread:
            counts[pattern] += missing / len(spread)
    return {p: float(n) for p, n in counts.items() if n > 0}


def _scoped(spec: Mapping[str, Any], instances: dict[str, float]) -> dict[str, tuple[float, float]]:
    """The target's instances, pattern → (instances, its share of the class's instances with
    that pattern): its ``qualname``, those its ``qualname_regex`` matches, else all."""
    captured = (spec.get("capture") or {}).get("qualname")
    if spec.get("qualname") and spec.get("kind") != "region":
        one = fold(str(spec["qualname"]))
        return {one: (1.0, 1.0 / max(instances.get(one, 1.0), 1.0))}
    if regex := spec.get("qualname_regex"):
        try:
            matcher = re.compile(str(regex))
        except re.error:
            matcher = None
        if matcher is not None:
            instances = {p: n for p, n in instances.items() if _matches(matcher, p)}
    if not instances:
        return {fold(str(captured)) if captured else "": (1.0, 1.0)}
    return {p: (n, 1.0) for p, n in instances.items()}


def _parent(groups: list[Group], i: int) -> int:
    """The innermost group of another target that holds group ``i`` (-1: none).

    Groups are sorted outer first; of two groups with the same pattern the
    earlier holds the later (the projection then takes the better of them)."""
    g = groups[i]
    if not g.pattern:
        return -1
    best, depth = -1, -1
    for j, h in enumerate(groups):
        if j == i or h.target == g.target or not h.pattern:
            continue
        if h.phase and g.phase and h.phase != g.phase:
            continue  # one phase each: different calls, they add up
        inside = g.pattern.startswith(h.pattern + ".") or (g.pattern == h.pattern and j < i)
        if inside and len(h.pattern) >= depth:
            best, depth = j, len(h.pattern)
    return best
