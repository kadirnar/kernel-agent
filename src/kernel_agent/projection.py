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

Approximations:

* a target's saving is spread evenly over its instances. The evaluator scales the
  captured instance's gain per call by the instances that call each entrypoint,
  so instances that only run other shapes get a share too;
* two targets on the same instances with different ``phase`` add up; a
  phase-specific parent still replaces all of a child's saving inside it;
* a region overlaps none of its parent's other children;
* every saving is still a module-level estimate: what shows up end to end is
  what the integration measures.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from kernel_agent import ledger
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
    savings: dict[str, float]  # each target's own est. saved ms (> 0)
    counted: dict[str, float]  # the part of it the projection counts

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

    def describe(self) -> str:
        """``a + b (40 %) + c; not counted (nested): d, e`` ("" without a saving)."""
        used = [t if self.part(t) > 0.995 else f"{t} ({self.part(t):.0%})" for t in self.used]
        text = " + ".join(used)
        if self.left_out:
            text += "; not counted (nested): " + ", ".join(self.left_out)
        return text

    def as_dict(self) -> dict[str, Any]:
        return {
            "projected_ms": self.projected_ms,
            "saved_ms": round(self.saved_ms, 3),
            "counted": {t: round(ms, 3) for t, ms in self.counted.items()},
            "left_out": self.left_out,
            "label": self.describe(),
        }

    def _order(self, target: str) -> tuple[float, str]:
        return (-self.counted.get(target, 0.0), target)


# ------------------------------------------------------------------ projection


def project(tree: Tree, saved: Mapping[str, float | None], baseline_ms: float) -> Projection:
    """``baseline − Σ saved`` over the best set of targets that do not overlap.

    ``saved``: est. saved ms per run of each target (its best kept result). A
    target that is not in the tree counts in full."""
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
    return Projection(baseline_ms, savings, counted)


def series(
    tree: Tree, baseline_ms: float, rows: Iterable[dict[str, Any]]
) -> list[tuple[dict[str, Any], Projection]]:
    """The projection after every kept or re-evaluated kernel row, each target at its best
    result that stands (:func:`ledger.standing`, as in :func:`ledger.summary`)."""
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
            out.append((row, project(tree, saved, baseline_ms)))
    return out


def of_run(
    run: RunDir, saved: Mapping[str, float | None], baseline_ms: float | None
) -> Projection | None:
    return None if baseline_ms is None else project(tree(run), saved, baseline_ms)


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
    paths = [run.profile_dir / "profile.json"]
    paths += sorted((run.root / "rounds").glob("*/profile/profile.json"))
    return build(specs, [read_json(p, {}) or {} for p in paths])


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
        total = sum(n for n, _ in instances.values())
        for pattern, (n, inside) in instances.items():
            if region and pattern:
                pattern = f"{pattern}.[{target_id}]"  # inside each parent instance
            groups.append(Group(target_id, pattern, n / total, phase, inside))
    return of_groups(groups)


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
            instances = {
                p: n
                for p, n in instances.items()
                if matcher.search(p) or matcher.search(p.replace("*", "0"))
            }
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
