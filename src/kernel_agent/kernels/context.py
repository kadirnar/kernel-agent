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
missing. The roofline (``roofline.apply_sol(hot_l2=..., context=...)``) follows the L2 choice
through the evaluator's ``l2_flush`` and the launch floor through ``context``.

**Records across contexts.** A target's context can change between improve rounds (a
re-profile after the systems agent graphed a stage: eager -> graph), so its records may be
timed in another context than its current one (an evaluation's ``context`` and ``l2``; one
from before #226 has neither: eager, warm). :func:`comparable` gives a record's speedup in a
context: its ``speedup`` when it was timed there, ``speedup_by_context[<context>]`` when it
was timed with the same L2 in the other context and timed in this one too (a winner's), else
none, and why (the context changed). Whatever compares an evaluation with older records of
its target goes through it: the ledger's keep bar and the early-discard bar
(``ledger.bar``), the budget's non-improving streak, an idea's ``refuted`` verdict, the
duplicate cache (``dedup.find``), the integration's ``stale`` check (``evaluate.stale``) and
the best-record ranking (``tools.ranked_for_target``, the scheduler's arms); a record with no
speedup there sets no bar, is not served from the cache, ranks after those that have one
and is re-evaluated before the integration takes it.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Mapping
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

    @property
    def key(self) -> tuple[str, str]:
        """``(context, l2)``: what a record compares in (:func:`comparable`)."""
        return self.context, self.l2


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


_CURRENT: dict[tuple[str, str], tuple[tuple[Any, ...], TimingContext]] = {}


def _stamp(run: RunDir, target_id: str) -> tuple[Any, ...]:
    """``(path, inode, mtime, size)`` of every file :func:`for_target` reads."""
    paths = [
        run.target(target_id) / "spec.json",
        run.toolchain_json,
        run.profile_dir / "profile.json",
        *sorted(run.root.glob("rounds/*/profile/profile.json")),
    ]
    out: list[tuple[Any, ...]] = []
    for path in paths:
        try:
            st = path.stat()
        except OSError:
            out.append((str(path),))
        else:
            out.append((str(path), st.st_ino, st.st_mtime_ns, st.st_size))
    return tuple(out)


def current(run: RunDir, target_id: str) -> TimingContext:
    """:func:`for_target`, read again only when a file it reads changed: the rankings of a
    target's records ask for it on every call, and a profile is large."""
    key, stamp = (str(run.root), target_id), _stamp(run, target_id)
    found = _CURRENT.get(key)
    if found is not None and found[0] == stamp:
        return found[1]
    context = for_target(run, target_id)
    _CURRENT[key] = (stamp, context)
    return context


# ------------------------------------------------------------------ records across contexts


#: A record's roofline fields that depend on the times and the launch floor of the context it
#: was timed in: left out of its view from another context (:func:`in_context`).
_TIMED_SOL = (
    "pct_of_sol",
    "bound",
    "launch_floor_ms",
    "launch_floor_note",
    "suspicious_faster_than_sol",
    "sol_unreliable",
)


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def timed_in(record: Mapping[str, Any]) -> tuple[str, str]:
    """``(context, l2)`` an evaluation result or record was timed in: eager calls with a warm
    L2 for one from before #226 (no ``context``)."""
    return str(record.get("context") or "eager"), str(record.get("l2") or "warm")


def _how(key: tuple[str, str]) -> str:
    return f"{'in a CUDA graph' if key[0] == 'graph' else 'eagerly'} with a {key[1]} L2"


def comparable(record: Mapping[str, Any], key: tuple[str, str]) -> tuple[float | None, str | None]:
    """``(speedup, why not)`` of an evaluation record in the timing context ``key``
    (``(context, l2)``, :attr:`TimingContext.key`): its ``speedup`` when it was timed there,
    its ``speedup_by_context[context]`` when it was timed with the same L2 in the other
    context and timed in this one too (a winner, or a ``profile`` evaluation); else ``(None,
    why)``: the context changed and the record has no speedup in this one. ``(None, None)``
    for a record without a speedup (a failure, a quick check)."""
    speedup = _number(record.get("speedup"))
    if speedup is None:
        return None, None
    mine = timed_in(record)
    if mine == key:
        return speedup, None
    other = (record.get("speedup_by_context") or {}).get(key[0])
    if mine[1] == key[1] and (found := _number(other)) is not None:
        return found, None
    if mine[1] != key[1]:
        detail = f"its speedup with a {key[1]} L2 was never measured"
    elif isinstance(other, str):
        detail = f"{key[0]}: {other}"
    else:
        detail = f"no {key[0]} timing in it: only a winner is timed in the other context too"
    return None, (
        f"the timing context changed: timed {_how(mine)}, the target is now timed "
        f"{_how(key)} ({detail})"
    )


def in_context(record: dict[str, Any], key: tuple[str, str]) -> dict[str, Any] | None:
    """``record`` seen from the timing context ``key``, or None when it has no speedup there
    (:func:`comparable`). A record timed there, or without a speedup, is returned as is. One
    timed in the other context becomes a copy with that context's ``speedup`` and
    ``context``, per-case times from the cases' ``timing`` (``ref_ms``, ``new_ms``,
    ``speedup``), weighted times and ``est_saved_ms_per_run``; ``measured_in`` says where it
    was timed and at which speedup, ``context_note`` how to read it. Its roofline fields
    (``pct_of_sol``, ``bound``, ...: the other context's times and launch floor) are left
    out."""
    speedup, why = comparable(record, key)
    if why is not None:
        return None
    mine = timed_in(record)
    if speedup is None or mine == key:
        return record
    view = {k: v for k, v in record.items() if k not in _TIMED_SOL}
    cases: list[dict[str, Any]] = []
    ref_total = new_total = saved = 0.0
    weighted = True  # every timed case says how many calls of the target it stands for
    for timed in record.get("cases") or []:
        timing = (timed.get("timing") or {}).get(key[0])
        case = {k: v for k, v in timed.items() if k not in _TIMED_SOL}
        if isinstance(timing, Mapping) and "ref_ms" in timing and "new_ms" in timing:
            ref, new = float(timing["ref_ms"]), float(timing["new_ms"])
            case.update(ref_ms=ref, new_ms=new, speedup=timing.get("speedup"))
            n = float(case.get("calls_per_run") or 0)
            ref_total, new_total = ref_total + n * ref, new_total + n * new
            if (calls := _number(case.get("target_calls"))) is None:
                weighted = False
            else:
                saved += calls * (ref - new)
        cases.append(case)
    view.update(
        cases=cases,
        speedup=speedup,
        context=key[0],
        measured_in={"context": mine[0], "l2": mine[1], "speedup": record.get("speedup")},
        context_note=f"timed {_how(mine)} at {record.get('speedup')}x; speedup and the "
        f"per-case times are its timing {_how(key)} (speedup_by_context, fewer rounds); its "
        "roofline (pct_of_sol, bound) was measured with the other context's times: evaluate "
        "it again for this context's",
    )
    if new_total > 0:
        view.update(ref_ms_weighted=round(ref_total, 4), new_ms_weighted=round(new_total, 4))
    else:  # no case says its times there: none from the other context either
        view.pop("ref_ms_weighted", None)
        view.pop("new_ms_weighted", None)
    if weighted and new_total > 0:
        view["est_saved_ms_per_run"] = round(saved, 3)
    else:
        view.pop("est_saved_ms_per_run", None)
    return view
