"""The timing context of a target: how its module runs in the model (issue #226).

The evaluator times a candidate in one context (:func:`kernel_agent.kernels.bench.time_call`):
eager calls, or calls captured in a CUDA graph, with a warm or a cold L2. Eager timing of a
module that runs inside a CUDA-graphed stage rewards host time the model never pays (an FP8
GEMM of two Triton launches at M = 352 measured 0.58x eager and 2.0x graph-timed with a cold
L2 on an RTX 5070 Ti); warm-L2 timing of a module whose weights the rest of the model evicts
between its calls times them from L2, not DRAM. Which context is the target's comes from the
run's newest profile, never from names:

* **graph** when at least :data:`GRAPH_SHARE` of the GPU events of the timeline stage that
  holds the target's instance are launched by CUDA-graph replays (``kernel_view.timeline``
  of the newest profile with a timeline, :mod:`kernel_agent.profiling.timeline`: the
  innermost stage whose qualname is the instance's or a prefix of it); else eager.
* **cold** L2 when the bytes the rest of the model touches between two consecutive calls of
  one instance exceed the GPU's L2 (``GPUInfo.l2_cache_mb`` in ``toolchain.json``): the work
  bytes of the profile's leaf classes per run (``classes[].work``: the weights each call
  reads, its first input and its output) less the instance's own, over the instance's calls
  per run. An estimate: an average over the run (a module called in a tight loop is warm
  between most of its calls), without attention's KV reads; else warm.

No profile, timeline, work or L2 size: eager or warm, and the reason says which fact is
missing. The roofline (``roofline.apply_sol(hot_l2=...)``) follows the L2 choice through the
evaluator's ``l2_flush``.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernel_agent.workspace import RunDir, read_json

#: A stage is graph-launched from this share of its GPU events launched by graph replays.
GRAPH_SHARE = 0.5
_MB = 1024 * 1024


@dataclass(frozen=True)
class TimingContext:
    """A target's timing context: ``context`` (``eager`` or ``graph``), ``l2`` (``warm`` or
    ``cold``) and why (``reason``)."""

    context: str = "eager"
    l2: str = "warm"
    reason: str = ""

    def kwargs(self) -> dict[str, Any]:
        """The evaluator's keyword arguments (``run_evaluation``, ``run_sweep``, ...)."""
        return {
            "context": self.context,
            "l2_flush": self.l2 == "cold",
            "context_reason": self.reason,
        }


def _qualname(spec: dict[str, Any]) -> str | None:
    """The instance a target's capture recorded (its spec's ``capture.qualname``)."""
    name = (spec.get("capture") or {}).get("qualname") or spec.get("qualname")
    return str(name) if name else None


def stage_of(
    timeline: dict[str, Any], qualname: str, methods: set[str] | frozenset[str] = frozenset()
) -> tuple[str, int, int] | None:
    """The innermost timeline stage that holds ``qualname``: its module's label, GPU events
    and graph-launched events (summed over its entrypoints); None: no stage holds it. A
    stage's label is its module's qualname with layer indices folded (``layers.*``) and,
    for an entrypoint other than ``forward`` (one of ``methods``), ``.<method>``."""
    found: dict[str, list[int]] = {}
    for stage in timeline.get("stages") or []:
        label = str(stage.get("stage") or "")
        if not label or label.startswith("("):  # (no stage), (outside the run)
            continue
        module, _, method = label.rpartition(".")
        if not module or method not in methods:
            module = label
        if fnmatch.fnmatchcase(qualname, module) or fnmatch.fnmatchcase(qualname, module + ".*"):
            acc = found.setdefault(module, [0, 0])
            acc[0] += int(stage.get("events") or 0)
            acc[1] += int(stage.get("graph_events") or 0)
    if not found:
        return None
    module = max(found, key=lambda m: (m.count("."), len(m)))
    return module, found[module][0], found[module][1]


def graph_choice(profile: dict[str, Any], qualname: str, where: str) -> tuple[str, str] | None:
    """``(context, reason)`` from ``profile``'s timeline; None: it has no timeline."""
    timeline = (profile.get("kernel_view") or {}).get("timeline") or {}
    if not timeline.get("stages"):
        return None
    methods = {str(e).rsplit(".", 1)[-1] for e in profile.get("entrypoints") or []}
    stage = stage_of(timeline, qualname, methods)
    if stage is None:
        return "eager", f"eager: no timeline stage of {where} holds {qualname}"
    label, events, graph_events = stage
    if events <= 0:
        return "eager", f"eager: stage {label} launched no GPU work in {where}"
    share = graph_events / events
    context = "graph" if share >= GRAPH_SHARE else "eager"
    return context, (
        f"{context}: {share:.0%} of the {events} GPU events of stage {label} (which holds "
        f"{qualname}) were launched by CUDA-graph replays in {where} "
        f"({'at least' if context == 'graph' else 'below'} {GRAPH_SHARE:.0%})"
    )


def _row_bytes(row: dict[str, Any]) -> float:
    return float(row.get("weight_bytes") or 0) + float(row.get("io_bytes") or 0)


def l2_choice(
    profile: dict[str, Any], spec: dict[str, Any], l2_mb: float | None, where: str
) -> tuple[str, str] | None:
    """``(l2, reason)`` from ``profile``'s work per class; None: it has none for the target's
    instances, or it does not see the whole model's work (``module_gaps``: compiled or
    graph-replayed regions, whose calls the hooks never see)."""
    cls = str(
        spec.get("parent_class") if spec.get("kind") == "region" else spec.get("module_class")
    )
    if any((profile.get("module_gaps") or {}).values()):
        return None  # compiled or graph-replayed regions: their work is not in the classes
    classes = profile.get("classes") or []
    mine = [r for c in classes if c.get("cls") == cls for r in c.get("work") or []]
    total = sum(_row_bytes(r) for c in classes if c.get("is_leaf") for r in c.get("work") or [])
    if not mine or total <= 0:
        return None
    # the rows of the target's instances: its capture's instance groups, else the group of
    # its captured instance, else (a target of every instance) the whole class
    qualname = _qualname(spec)
    patterns = set(((spec.get("capture") or {}).get("instance_groups") or {}).keys())
    if patterns:
        rows = [r for r in mine if r.get("group") in patterns]
    elif qualname:
        rows = [r for r in mine if fnmatch.fnmatchcase(qualname, str(r.get("group") or ""))]
    else:
        rows = mine
    phase = spec.get("phase")
    rows = [r for r in rows if not phase or r.get("phase") == phase] or rows
    if not rows:
        return None
    groups: dict[str, list[float]] = {}  # group -> [bytes, calls] of one of its instances
    for r in rows:
        n = max(int(r.get("instances") or 1), 1)
        acc = groups.setdefault(str(r.get("group")), [0.0, 0.0])
        acc[0] += _row_bytes(r) / n
        acc[1] += float(r.get("calls") or 0) / n
    group, (own, calls) = max(groups.items(), key=lambda kv: kv[1][1])
    if calls <= 0:
        return None
    if not l2_mb:
        return "warm", "warm L2: the GPU's L2 size is unknown (toolchain.json)"
    between = max(total - own, 0.0) / calls
    cold = between > l2_mb * _MB
    return ("cold" if cold else "warm"), (
        f"{'cold' if cold else 'warm'} L2: about {between / _MB:,.0f} MB of other work between "
        f"two calls of one instance of {group} ({calls:,.0f} calls per run, {where}) "
        f"{'exceeds' if cold else 'fits in'} the {l2_mb:g} MB L2"
    )


def select(
    profiles: list[tuple[str, dict[str, Any]]], spec: dict[str, Any], l2_mb: float | None
) -> TimingContext:
    """The timing context of the target ``spec`` from ``profiles`` (``(where, profile)``,
    newest first) and the GPU's L2 size in MB (see the module docstring): the context from
    the newest profile with a timeline (an improve round's re-profile has the optimised
    model's CUDA graphs), the L2 from the oldest with the target's work (the work is the
    math, the same in every round, and a re-profile does not see inside graphed regions)."""
    qualname = _qualname(spec)
    graph: tuple[str, str] | None = None
    l2: tuple[str, str] | None = None
    for where, profile in profiles:
        if graph is None and qualname:
            graph = graph_choice(profile, qualname, where)
    for where, profile in reversed(profiles):
        if l2 is None:
            l2 = l2_choice(profile, spec, l2_mb, where)
    if graph is None:
        why = "no timeline in the run's profiles" if qualname else "no captured instance"
        graph = ("eager", f"eager: {why}")
    if l2 is None:
        l2 = (
            "warm",
            f"warm L2: no profile of the whole model with the per-call work of "
            f"{spec.get('module_class')}",
        )
    return TimingContext(graph[0], l2[0], f"{graph[1]}; {l2[1]}")


def _round(path: Path) -> int:
    try:
        return int(path.parent.parent.name)
    except ValueError:
        return -1


def profiles_of(run: RunDir) -> list[tuple[str, dict[str, Any]]]:
    """The run's profiles newest first: its improve rounds' re-profiles, then its own, as
    ``(path relative to the run, profile)``."""
    paths = sorted(run.root.glob("rounds/*/profile/profile.json"), key=_round, reverse=True)
    out = []
    for path in [*paths, run.profile_dir / "profile.json"]:
        data = read_json(path, None)
        if isinstance(data, dict):
            out.append((str(path.relative_to(run.root)), data))
    return out


def for_target(run: RunDir, target_id: str) -> TimingContext:
    """The timing context of a run's target (never raises: eager, warm with the reason)."""
    try:
        spec = read_json(run.target(target_id) / "spec.json", {}) or {}
        gpu = (read_json(run.toolchain_json, {}) or {}).get("gpu") or {}
        return select(profiles_of(run), spec, float(gpu.get("l2_cache_mb") or 0) or None)
    except Exception as exc:  # a broken profile must not stop an evaluation
        return TimingContext(reason=f"eager, warm L2: no context ({type(exc).__name__}: {exc})")
