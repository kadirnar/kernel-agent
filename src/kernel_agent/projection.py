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

Units (:class:`Units`): a kernel's est. saved ms is per run of the workload, the baseline
in the run's metric (:mod:`kernel_agent.objective`). Every projection of kernel savings
converts them first (:func:`kernel_agent.objective.from_run`): as they are for
``latency``, ÷ the seconds of audio of a batched run for ``throughput`` (ms per second of
generated audio), their share inside the first-audio window for ``ttfa``
(:func:`window`). A saving with no value in the metric is not projected (``unknown``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kernel_agent import ledger, objective
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


def of_set(tree: Tree, saved: Mapping[str, float | None], baseline_ms: float) -> dict[str, Any]:
    """Projected ms of an integration's accepted set (``integration.json`` ``projection``):
    ``saved`` maps each item (a kernel's ``target=path``, a transform's path) to its est.
    saved ms in the metric's ms (a kernel's module-level estimate, :class:`Units`; a
    transform's measured gain alone). Kernels nest by target; transforms are in no tree
    and count in full, a slower one too. ``counted_ms``: the part of each saving counted."""
    key = {a: kernel_target(a) or a for a in saved}
    proj = project(tree, {key[a]: v for a, v in saved.items()}, baseline_ms)
    slower = sum(v for v in saved.values() if v is not None and v < 0)
    return {
        "projected_ms": round(baseline_ms - proj.saved_ms - slower, 3),
        "est_saved_ms": dict(saved),
        "counted_ms": {
            a: round(proj.counted.get(key[a], 0.0), 3)
            for a, v in saved.items()
            if v is not None and v > 0
        },
    }


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

    The estimate covers the calls of a full streamed run (the capture runs it whole): its
    cases' calls per run × the instances that call their entrypoint, as the evaluator
    weights them. The profiles are taken inside the window (``Workload.metric_window``):
    the calls of the target's class there (a region: of its parent class), in its
    ``phase``; with a ``qualname_regex``, of the instance groups it matches
    (``classes[].work``; None without them). A class that no profile saw makes no call
    before the first audio. The share = calls in the window ÷ calls covered (at most 1):
    the estimate's gain per call × the calls inside the window."""
    capture = spec.get("capture") or {}
    users = capture.get("method_instances") or {}
    covered = sum(
        float(c.get("count") or 0) * float(users.get(str(c.get("method") or "forward")) or 1)
        for c in capture.get("cases") or []
    )
    calls = _window_calls(spec, profiles)
    return None if covered <= 0 or calls is None else min(calls / covered, 1.0)


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
