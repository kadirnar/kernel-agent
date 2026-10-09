"""Native engine targets and the native arm of the improve loop (issue #134).

Module kernels plateau where the remaining time is not inside one module but between
them: many short launches per iteration, weights re-streamed by every call, glue kernels,
grid syncs that drain the weight stream. A *native engine target* hands such a part of the
inference path to the systems-native agent (``native`` sessions), which rewrites it as
native code (CUDA C++ / CuTe, a multi-file project: :mod:`kernel_agent.native.project`)
that keeps weights streaming, fuses across modules and runs the part in one persistent
kernel or a few. Scopes:

* ``stage``: one module group of the profile (an iterative solver module, a layer stack, a
  decode step, a one-shot decoder): replaced as a kernel target (``native_<id>``, captured
  like any target, so its correctness is checked teacher forced on the stage's recorded
  inputs, each case from the module state its call saw, e.g. a KV-cache attribute; a
  capture the unmodified reference fails is refused, and the digest says why:
  :func:`capture_refusal`) or as a transform;
* ``group``: the stages that run once per iteration of the generation loop (the same calls
  per run), fused across their boundaries: a transform, checked end to end;
* ``loop``: the whole generation loop: a transform that keeps the per-iteration seams the
  workload's quality checks need (a teacher-forced workload wraps one module call per
  iteration, which the engine must keep calling from Python).

**Stage graph** (:func:`stage_graph`), from the newest ceilings table (``profiling/
ceilings.py``: module class at one instance group and phase, ms now, floors per precision,
calls per run; :func:`newest_table_path`): the stages are the topmost non-leaf rows with at
least :data:`MIN_SHARE` of the run, each with its pattern (:func:`pattern`):

* ``solver``: a non-leaf row inside it is called k ≥ 2 times per stage call and holds most of
  its time (diffusion / flow-matching samplers, iterative refinement);
* ``stack``: most of its time is in a repeated layer group (``….layers.*``: encoders, LM
  bodies, vision backbones, decoder stacks);
* ``other`` (vocoders, VAEs and other convolution stacks without a folded layer index).

``calls`` says how often it runs (1: one-shot; many: a decode / per-iteration step). The
stages called equally often (≥ 2 per run) form the loop body (``group`` scope, with two or
more of them); with :data:`LOOP_SHARE` or more of the run, the whole loop is a target too.
Fusion chains that span stages (``profile/fusions.json``, issue #231: the ops of one
fusible chain in the modules of two or more stages) are evidence for a group: the loop
body's when its stages hold the chain, else a ``group`` of the stages it spans
(``fuse_<a>_<b>``), each with the chains and their predicted saving (``evidence``).
Order (the staged plan): stages by the time above their floor (``now − floor`` at the best
precision the run allows), then the group, then the loop. The planner can replace this with
its own ``native`` plan entry (``plan.json``: ``why`` and ``stages``).

**Native arm** (``scheduler.NATIVE``, opt-in: ``--native on``, or ``plan`` (the default)
when the plan asks for it): it waits while any kernel arm is live and has not plateaued, then
gets slices of its own (a longer session, ``--native-minutes``). Its bar is the best
module-level result end to end (:func:`bar`: integrations and the systems agent's runs); a
stage is done when a native end-to-end run of it beats that bar (:func:`current`), and only
then does the next stage start. Its end-to-end runs are recorded with backend ``native``
(:func:`e2e_backend`).

**After the plan** (issue #164): every stage beating the bar once is a note, not a stop. The
improve loop re-profiles the model of the arm's best run (:func:`reprofile_dir`, the newest
table from then on), the stage graph is derived again from it, and the arm works on the
stage with the most time left above its floor (:func:`focus`; a group scope is the agent's
alternative) under its stop rules: its patience, its time cap, and every stage at the floor
(:func:`at_floor`).

Design: ``docs/NATIVE.md``.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kernel_agent import ledger
from kernel_agent.budget import improves
from kernel_agent.native import project
from kernel_agent.workspace import RunDir, read_json, read_jsonl

MODES = ("off", "plan", "on")
DEFAULT_MODE = "plan"  # the native arm exists only when the plan asks for it
DIR = "native"  # the native agent's working directory, inside the run's transforms/
TARGET_PREFIX = "native_"  # kernel targets of native stages (captured for teacher forcing)
MIN_SHARE = 0.05  # a stage holds at least this share of the run
LOOP_SHARE = 0.6  # the loop body covers this much of the run: the whole loop is a target
STEP_CALLS = 2  # stages called at least this often per run can form the loop body
DOMINANT = 0.5  # share of a stage's time that makes an inner row its pattern
MAX_STAGES = 6
SCOPES = ("stage", "group", "loop")
PATTERNS = ("solver", "stack", "other")
#: Words for the agents: what each pattern usually is, per model family.
PATTERN_HINTS = {
    "solver": "an iterative solver (diffusion / flow-matching sampler, refinement loop): "
    "fuse the steps, keep the inner network's weights streaming across them, fold the "
    "guidance branches and the update math into the network's kernels",
    "stack": "a stack of repeated layers (LM body, encoder, vision backbone, decoder "
    "stack): one persistent kernel per call or per few layers, weight streaming with "
    "warp-specialised producers and cross-layer prefetch, PDL between launches",
    "other": "a one-off network (vocoder, VAE decoder, convolution stack): fused "
    "convolutions in a layout of your own, activations kept on chip between ops",
}
_INDEX = re.compile(r"\.\*(?=\.|$)")


@dataclass(frozen=True)
class Stage:
    """One native engine target of the staged plan."""

    id: str
    scope: str  # stage | group | loop
    group: str = ""  # instance group (qualname, layer indices as *) of a stage
    module_class: str | None = None
    members: tuple[str, ...] = ()  # the stage ids of a group / loop
    calls: float = 0.0  # per run
    now_ms: float = 0.0  # in the ceilings table's window
    share: float = 0.0
    floor_ms: float | None = None  # at the best precision the run allows
    floor_label: str = ""
    pattern: str = "other"
    inner: str = ""  # the row that sets its pattern, e.g. "Net@m.solver.net × 9 per call"
    phase: str | None = None
    idea: str = ""
    why: str = ""
    precision: str | None = None  # the plan's precision for its capture
    expected_speedup: float | None = None
    evidence: str = ""  # a group: the fusion chains across its stages' boundaries (#231)

    @property
    def target_id(self) -> str | None:
        """The kernel target of a stage (None for a group or the loop: end to end only)."""
        return f"{TARGET_PREFIX}{self.id}"[:40] if self.scope == "stage" else None

    @property
    def headroom_ms(self) -> float:
        return max(self.now_ms - self.floor_ms, 0.0) if self.floor_ms is not None else 0.0

    @property
    def of_floor(self) -> float | None:
        """Floor / now: how close the stage already runs to its floor (1.0: at it)."""
        if self.floor_ms is None or self.now_ms <= 0:
            return None
        return min(self.floor_ms / self.now_ms, 1.0)

    def describe(self) -> str:
        """One line for a digest or a prompt."""
        where = f"`{self.module_class}@{self.group}`" if self.group else ""
        if self.members:
            where = "stages " + ", ".join(f"`{m}`" for m in self.members)
        parts = [f"**{self.id}** ({self.scope}, {self.pattern}): {where}"]
        if self.now_ms:
            parts.append(f"{self.now_ms:,.4g} ms now ({self.share:.0%} of the run)")
        if self.calls:
            parts.append(f"{self.calls:,.0f} calls per run")
        if self.floor_ms is not None and self.of_floor is not None:
            parts.append(
                f"{self.floor_label} floor {self.floor_ms:,.4g} ms (runs at {self.of_floor:.0%} "
                f"of it, {self.headroom_ms:,.4g} ms above)"
            )
        if self.inner:
            parts.append(self.inner)
        if self.evidence:
            parts.append(self.evidence)
        text = "; ".join(parts)
        return text + (f". Plan: {self.idea}" if self.idea else "")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            k: getattr(self, k)
            for k in (
                "id",
                "scope",
                "group",
                "module_class",
                "calls",
                "now_ms",
                "share",
                "floor_ms",
                "pattern",
                "phase",
                "idea",
                "precision",
                "evidence",
            )
            if getattr(self, k) not in (None, "", 0.0)
        }
        if self.members:
            out["members"] = list(self.members)
        return out


# ------------------------------------------------------------------ the stage graph


def slug(group: str, cls: str | None = None) -> str:
    """A stage id from its instance group: ``model.encoder.layers.*`` → ``encoder_layers``
    (the root and layer indices dropped), else the class name."""
    parts = [p for p in group.split(".")[1:] if p and p != "*"]
    text = "_".join(parts) or (cls or "stage")
    return project.safe_name(text)[:32]


def inside(name: str, group: str) -> bool:
    return name != group and name.startswith(group + ".")


def _floor(row: Mapping[str, Any], columns: Sequence[str]) -> tuple[float | None, str]:
    """The lowest floor of a row among the precisions the run allows (``columns``)."""
    floors = row.get("floors") or {}
    known = [(float(floors[c]), c) for c in columns if floors.get(c) is not None]
    if not known:
        return None, ""
    value, column = min(known)
    return value, column


def pattern(row: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    """``solver`` / ``stack`` / ``other`` of a stage row, and the inner row that decides it."""
    group, calls, now = str(row["group"]), float(row.get("calls") or 0), float(row["now_ms"])
    nested = [r for r in rows if inside(str(r["group"]), group) and r.get("now_ms")]
    for r in sorted(nested, key=lambda r: -float(r["now_ms"])):
        k = float(r.get("calls") or 0) / calls if calls else 0.0
        heavy = float(r["now_ms"]) >= DOMINANT * now
        if heavy and k >= 2 and "*" not in str(r["group"])[len(group) :]:
            return "solver", f"`{r['cls']}@{r['group']}` runs {k:.3g}× per call"
    for r in sorted(nested, key=lambda r: -float(r["now_ms"])):
        if float(r["now_ms"]) >= DOMINANT * now and _INDEX.search(str(r["group"])[len(group) :]):
            n = int(r.get("instances") or 0)
            return "stack", f"`{r['cls']}@{r['group']}` × {n} layers" if n else "layers"
    return "other", ""


def stage_graph(
    table: Mapping[str, Any] | None,
    *,
    columns: Sequence[str] | None = None,
    min_share: float = MIN_SHARE,
    fusions: Sequence[Mapping[str, Any]] | None = None,
) -> list[Stage]:
    """The native engine targets of a ceilings table, in staged-plan order: the stages
    (topmost non-leaf rows with ``min_share`` of the run) by their time above the floor,
    then the loop body (stages called equally often, ≥ 2 of them), the groups of stages
    that fusion chains span (``fusions``: a fusion table's candidates,
    :func:`fusion_groups`) and the whole loop (when that body covers :data:`LOOP_SHARE` of
    the run). ``columns``: the precisions whose floors count (default: the table's
    ``columns``)."""
    if not table:
        return []
    rows = [r for r in table.get("rows") or [] if r.get("group") and r.get("now_ms")]
    cols = list(columns if columns is not None else table.get("columns") or ["exact"])
    # non-leaf: something is nested inside it
    candidates = [
        r
        for r in rows
        if float(r.get("share") or 0.0) >= min_share
        and any(inside(str(o["group"]), str(r["group"])) for o in rows)
    ]
    candidates.sort(key=lambda r: (-float(r["now_ms"]), str(r["group"])))
    top: list[Mapping[str, Any]] = []
    for r in candidates:
        if not any(
            inside(str(r["group"]), str(t["group"])) or r["group"] == t["group"] for t in top
        ):
            top.append(r)
    stages: list[Stage] = []
    used: set[str] = set()
    for r in top[:MAX_STAGES]:
        sid = slug(str(r["group"]), str(r.get("cls")))
        while sid in used:
            sid = f"{sid}_2"
        used.add(sid)
        floor, column = _floor(r, cols)
        kind, inner = pattern(r, rows)
        label = str(((table.get("precisions") or {}).get(column) or {}).get("label") or column)
        stages.append(
            Stage(
                id=sid,
                scope="stage",
                group=str(r["group"]),
                module_class=str(r.get("cls")),
                calls=float(r.get("calls") or 0.0),
                now_ms=float(r["now_ms"]),
                share=float(r.get("share") or 0.0),
                floor_ms=floor,
                floor_label=label,
                pattern=kind,
                inner=inner,
                phase=r.get("phase"),
            )
        )
    stages.sort(key=lambda s: (-s.headroom_ms, -s.now_ms, s.id))
    loops = loop_targets(stages)
    body = next((s for s in loops if s.scope == "group"), None)
    body, spans = fusion_groups(stages, fusions or [], body)
    groups = [body] if body is not None else []
    return stages + groups + spans + [s for s in loops if s.scope != "group"]


def fusion_groups(
    stages: Sequence[Stage], fusions: Sequence[Mapping[str, Any]], body: Stage | None = None
) -> tuple[Stage | None, list[Stage]]:
    """The loop ``body`` with the fusion chains inside its stages as evidence (issue #231),
    and a ``group`` (``fuse_<a>_<b>``) per other set of stages that chains span (their ops in
    the modules of two or more stages: ``modules``, folded qualnames), by predicted saving
    (of the chains whose savings add up: :func:`_additive`)."""
    spans: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for chain in fusions:
        modules = [str(m) for m in chain.get("modules") or [] if m]
        spanned = {
            s.id
            for s in stages
            if s.scope == "stage" and any(m == s.group or inside(m, s.group) for m in modules)
        }
        if len(spanned) >= 2:
            spans.setdefault(tuple(sorted(spanned)), []).append(chain)
    members = set(body.members) if body is not None else set()
    if body is not None:
        mine = [c for ids, chains in spans.items() if set(ids) <= members for c in chains]
        if mine:
            body = dataclasses.replace(body, evidence=_evidence(mine))
    rest = [(ids, chains) for ids, chains in spans.items() if not set(ids) <= members]
    rest.sort(key=lambda kv: (-_saving(_additive(kv[1])), kv[0]))
    by_id = {s.id: s for s in stages}
    groups = []
    for ids, chains in rest:
        parts = [by_id[i] for i in ids]
        floors = [s.floor_ms for s in parts]
        floor = sum(f for f in floors if f is not None) if None not in floors else None
        groups.append(
            Stage(
                id=project.safe_name("fuse_" + "_".join(ids))[:32],
                scope="group",
                members=ids,
                calls=min(s.calls for s in parts),
                now_ms=sum(s.now_ms for s in parts),
                share=sum(s.share for s in parts),
                floor_ms=floor,
                floor_label=parts[0].floor_label if floor is not None else "",
                evidence=_evidence(chains),
            )
        )
    return body, groups


def _additive(chains: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The chains whose savings add up (``fusion.additive``: rows that share an op, a GEMM
    in one's prologue and another's epilogue, count once)."""
    from kernel_agent.profiling.fusion import additive

    return additive(chains)


def _saving(chains: Iterable[Mapping[str, Any]]) -> float:
    return sum(float(c.get("saving_ms") or 0.0) for c in chains)


def _evidence(chains: Sequence[Mapping[str, Any]], shown: int = 3) -> str:
    """``2 fusion chains across its stages' boundaries predict 0.5 ms (…): `f1a2b3c` 0.42 ms
    (`add` → `mul`), …``: the total of those that add up, alternatives (they share an op
    with one of them) listed after them and not added."""
    kept = {id(c) for c in _additive(chains)}
    ranked = sorted(
        chains,
        key=lambda c: (id(c) not in kept, -float(c.get("saving_ms") or 0.0), str(c.get("id"))),
    )
    total = _saving(c for c in ranked if id(c) in kept)
    items = []
    for c in ranked[:shown]:
        ops = " → ".join(f"`{o.get('op')}`" for o in (c.get("ops") or [])[:6])
        alt = "" if id(c) in kept else ", an alternative"
        items.append(f"`{c.get('id')}` {float(c.get('saving_ms') or 0.0):,.3g} ms ({ops}{alt})")
    more = f" and {len(ranked) - shown} more" if len(ranked) > shown else ""
    others = len(ranked) - len(kept)
    alternatives = f"; {others} of them share an op with one counted, not added" if others else ""
    return (
        f"{len(ranked)} fusion chains across its stages' boundaries predict {total:,.3g} ms "
        f"(`profile/fusions.md`{alternatives}): " + ", ".join(items) + more
    )


def fusion_candidates(run: RunDir) -> list[dict[str, Any]]:
    """The candidates of the run's fusion table (``profile/fusions.json``: the analyze
    profile of the unmodified model; [] without one)."""
    table = read_json(run.profile_dir / "fusions.json", None)
    found = table.get("candidates") if isinstance(table, dict) else None
    return [c for c in found or [] if isinstance(c, dict)]


def loop_targets(stages: Sequence[Stage]) -> list[Stage]:
    """The loop body (``group``) and the whole loop (``loop``) of a list of stages: the
    largest set of stages with equal calls per run (≥ :data:`STEP_CALLS`)."""
    by_calls: dict[float, list[Stage]] = {}
    for s in stages:
        if s.scope == "stage" and s.calls >= STEP_CALLS:
            by_calls.setdefault(round(s.calls), []).append(s)
    if not by_calls:
        return []
    calls, body = max(by_calls.items(), key=lambda kv: (sum(s.now_ms for s in kv[1]), kv[0]))
    floors = [s.floor_ms for s in body]
    known = [f for f in floors if f is not None]
    floor = sum(known) if len(known) == len(floors) else None
    loop = Stage(
        id="loop",
        scope="loop",
        members=tuple(s.id for s in body),
        calls=float(calls),
        now_ms=sum(s.now_ms for s in body),
        share=sum(s.share for s in body),
        floor_ms=floor,
        floor_label=body[0].floor_label if floor is not None else "",
    )
    out: list[Stage] = []
    if len(body) >= 2:
        out.append(dataclasses.replace(loop, id="loop_body", scope="group"))
    if loop.share >= LOOP_SHARE:
        out.append(loop)
    return out


# ------------------------------------------------------------------ the plan


def plan_entry(run: RunDir) -> dict[str, Any] | None:
    """The newest plan's ``native`` entry (a round's re-plan before the first plan)."""
    plans = sorted(
        (run.root / "rounds").glob("*/plan.json"),
        key=lambda p: int(p.parent.name) if p.parent.name.isdigit() else 0,
        reverse=True,
    )
    for path in [*plans, run.plan_json]:
        entry = (read_json(path, {}) or {}).get("native")
        if isinstance(entry, dict) and entry:
            return entry
    return None


def reprofile_dir(run: RunDir, round_n: int, k: int) -> Path:
    """Where the improve loop re-profiles the model of the native arm's best run once its
    staged plan is done (``rounds/<n>/native/<k>/``: ``baseline.json``, ``profile/``)."""
    return run.root / "rounds" / str(round_n) / DIR / str(k)


def _int(text: str) -> int:
    return int(text) if text.isdigit() else 0


def newest_table_path(run: RunDir) -> Path | None:
    """The newest ceilings table with rows: the native arm's newest re-profile of a round
    (:func:`reprofile_dir`), the round's own re-profile, else the analyze profile's; a
    round's native re-profiles are newer than its own."""
    found = [((1, 0), run.profile_dir / "ceilings.json")]
    for path in (run.root / "rounds").glob("*/profile/ceilings.json"):
        found.append(((_int(path.parent.parent.name), 0), path))
    for path in (run.root / "rounds").glob(f"*/{DIR}/*/profile/ceilings.json"):
        k, round_dir = path.parent.parent, path.parent.parent.parent.parent
        found.append(((_int(round_dir.name), _int(k.name)), path))
    for _, path in sorted(found, key=lambda kp: kp[0], reverse=True):
        table = read_json(path, None)
        if isinstance(table, dict) and table.get("rows"):
            return path
    return None


def newest_table(run: RunDir) -> dict[str, Any] | None:
    """The newest ceilings table (:func:`newest_table_path`)."""
    path = newest_table_path(run)
    return read_json(path, None) if path is not None else None


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def planned(entry: Mapping[str, Any], derived: Sequence[Stage]) -> list[Stage]:
    """The stages of a plan's ``native`` entry, in its order, with the numbers of the
    derived stage of the same group (or id) when there is one."""
    by_group = {s.group: s for s in derived if s.group}
    by_id = {s.id: s for s in derived}
    out: list[Stage] = []
    for item in entry.get("stages") or []:
        if not isinstance(item, dict):
            continue
        scope = str(item.get("scope") or "stage")
        scope = scope if scope in SCOPES else "stage"
        group = str(item.get("group") or "")
        base = by_group.get(group) or by_id.get(str(item.get("id") or ""))
        sid = project.safe_name(str(item.get("id") or (base.id if base else slug(group))))[:32]
        members = tuple(str(m) for m in item.get("members") or ())
        stage = Stage(
            id=sid,
            scope=scope,
            group=group or (base.group if base else ""),
            module_class=item.get("module_class") or (base.module_class if base else None),
            members=members or (base.members if base else ()),
            calls=base.calls if base else 0.0,
            now_ms=base.now_ms if base else 0.0,
            share=base.share if base else 0.0,
            floor_ms=base.floor_ms if base else None,
            floor_label=base.floor_label if base else "",
            pattern=str(item.get("pattern") or (base.pattern if base else "other")),
            inner=base.inner if base else "",
            phase=item.get("phase") or (base.phase if base else None),
            idea=str(item.get("idea") or ""),
            why=str(item.get("why") or ""),
            precision=item.get("precision"),
            expected_speedup=_num(item.get("expected_speedup")),
        )
        if stage.scope == "stage" and not (stage.module_class and stage.group):
            continue  # a stage target needs its module group (else: plan it as a group)
        out.append(stage)
    return out


def stages(run: RunDir, columns: Sequence[str] | None = None) -> list[Stage]:
    """The staged plan of a run: the plan's ``native`` stages, else the derived graph."""
    derived = stage_graph(newest_table(run), columns=columns, fusions=fusion_candidates(run))
    entry = plan_entry(run)
    return (planned(entry, derived) if entry else []) or derived


def mode(value: Any) -> str:
    """``--native``: off / plan / on (anything else: the default)."""
    text = str(value or DEFAULT_MODE).strip().lower()
    return text if text in MODES else DEFAULT_MODE


def enabled(value: Any, entry: Mapping[str, Any] | None) -> str | None:
    """Why the run has a native arm (None: it has none)."""
    chosen = mode(value)
    if chosen == "on":
        return "--native on"
    if chosen == "plan" and entry:
        return "the plan asks for it" + (f": {entry['why']}" if entry.get("why") else "")
    return None


# ------------------------------------------------------------------ targets


def qualname_regex(group: str) -> str:
    """The ``qualname_regex`` of an instance group (``*``: any layer index)."""
    return "^" + re.escape(group).replace(r"\*", r"\d+") + "$"


def target_spec(
    stage: Stage, precision: str | None = None, backends: Sequence[str] = ("cuda",)
) -> dict[str, Any] | None:
    """The kernel target of a stage (``orchestrator._capture``): its module group captured
    from the real run, so a native stage is checked teacher forced on its recorded inputs
    (None: a group or the loop, checked end to end only)."""
    if stage.target_id is None or not stage.module_class or not stage.group:
        return None
    spec: dict[str, Any] = {
        "id": stage.target_id,
        "module_class": stage.module_class,
        "qualname_regex": qualname_regex(stage.group),
        "why": stage.why or f"native engine stage {stage.id} ({stage.scope}, {stage.pattern})",
        "approach": stage.idea or PATTERN_HINTS.get(stage.pattern, ""),
        "backends": list(backends),
        "native": True,
    }
    chosen = stage.precision or precision
    if chosen and chosen != "exact":
        spec["precision"] = chosen
    return spec


def module_precision(specs: Iterable[Mapping[str, Any]]) -> str | None:
    """The reduced precision most of the run's kernel targets use (the module kernels a
    native stage replaces already run at it, so its capture's tier allows it), else None
    (exact)."""
    counts: dict[str, int] = {}
    for spec in specs:
        precision = spec.get("precision")
        if not is_stage_target(spec) and precision and precision != "exact":
            counts[str(precision)] = counts.get(str(precision), 0) + 1
    return max(sorted(counts), key=lambda p: counts[p]) if counts else None


def is_stage_target(spec: Mapping[str, Any]) -> bool:
    """A target the native arm owns (no kernel arm of its own)."""
    return bool(spec.get("native"))


def capture_refusal(run: RunDir, target_id: str) -> str | None:
    """Why the capture of a (stage) target failed (``capture_error`` of its
    ``spec.failed.json``: the last line of the error, e.g. the reference failing its own
    capture, :class:`kernel_agent.profiling.capture.UnverifiableCapture`); None when it
    did not fail."""
    spec = read_json(run.target(target_id) / "spec.failed.json", {}) or {}
    lines = [line.strip() for line in str(spec.get("capture_error") or "").splitlines()]
    last = next((line for line in reversed(lines) if line), "")
    if not last:
        return None
    raised = re.match(r"[A-Za-z_][\w.]*: (.+)", last)  # "<module>.<Exception>: <why>"
    return (raised.group(1) if raised else last)[:600]


# ------------------------------------------------------------------ rows


def e2e_backend(transforms: Sequence[Path], kernels: Sequence[str]) -> str:
    """The ledger backend of an ``evaluate_e2e`` run: ``native`` (``+kernels``) when it ran
    a project bundle or a native stage's kernel target, else :func:`ledger.e2e_backend`."""
    native = any(project.read_bundle(Path(t)) is not None for t in transforms) or any(
        k.partition("=")[0].startswith(TARGET_PREFIX) for k in kernels
    )
    if not native:
        return ledger.e2e_backend(list(transforms), list(kernels))
    others = [k for k in kernels if not k.partition("=")[0].startswith(TARGET_PREFIX)]
    return "native+kernels" if others else "native"


def is_native(row: Mapping[str, Any]) -> bool:
    """An end-to-end row of the native arm."""
    return row.get("target") == ledger.E2E and str(row.get("backend") or "").startswith("native")


def _improves(row: Mapping[str, Any], best: float) -> bool:
    rec = {"passed": row.get("correct"), "speedup": row.get("speedup")}
    rec["timing_spread"] = row.get("spread")
    return improves(rec, best, ok_key="passed")


def _parts(row: Mapping[str, Any]) -> list[str]:
    return [p for p in str(row.get("snapshot") or "").split("+") if p]


def own_items(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    """The labels (``ledger.item_label``, as an integration row names its items) of the
    native arm's own items: its stage targets and the transforms that only its runs measured
    (its projects; not the module kernels and the systems agent's transforms it runs on)."""
    native: set[str] = set()
    other: set[str] = set()
    for row in rows:
        if row.get("target") != ledger.E2E or row.get("backend") == "integrate":
            continue
        for part in _parts(row):
            if part.endswith(".py"):  # a transform snapshot
                (native if is_native(row) else other).add(ledger.snapshot_stem(part))
            elif part.startswith(TARGET_PREFIX):  # a stage target
                native.add(part)
    return native - other


def bar(rows: Iterable[Mapping[str, Any]]) -> float:
    """The best module-level result end to end: the fastest passing end-to-end run that is
    not the native arm's (integrations, the systems agent's), at least 1.0. An integration
    measurement with one of the arm's own items (:func:`own_items`) is the arm's too, so an
    integration that accepted them does not raise the bar its runs are measured against."""
    rows = list(rows)
    own = own_items(rows)
    speeds = [
        float(r["speedup"])
        for r in rows
        if r.get("target") == ledger.E2E and r.get("correct") and r.get("speedup")
        if not is_native(r) and not (r.get("backend") == "integrate" and own & set(_parts(r)))
    ]
    return max([1.0, *speeds])


def stage_of(row: Mapping[str, Any], plan: Sequence[Stage]) -> str | None:
    """The stage a native end-to-end row measured: a transform snapshot named after the
    stage (its project's name starts with the stage id) or the stage's kernel target."""
    parts = [p for p in str(row.get("snapshot") or "").split("+") if p]
    for stage in plan:
        for part in parts:
            stem = ledger.snapshot_stem(part)
            if part == stage.target_id or stem == stage.id or stem.startswith(stage.id + "_"):
                return stage.id
    return None


def done(plan: Sequence[Stage], rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Stage id → the best end-to-end speedup of a native run of it that beat the bar of
    the module-level results."""
    rows = list(rows)
    level = bar(rows)
    out: dict[str, float] = {}
    for row in rows:
        if not is_native(row) or not _improves(row, level):
            continue
        sid = stage_of(row, plan)
        if sid is not None:
            out[sid] = max(out.get(sid, 0.0), float(row["speedup"]))
    return out


def current(plan: Sequence[Stage], rows: Iterable[Mapping[str, Any]]) -> Stage | None:
    """The stage the native arm works on: the first of the plan that no native run has yet
    taken past the module-level bar (None: every stage is done)."""
    finished = done(plan, rows)
    return next((s for s in plan if s.id not in finished), None)


def focus(graph: Sequence[Stage]) -> Stage | None:
    """After the staged plan (issue #164): the stage of a stage graph (the newest ceilings
    table's, :func:`stage_graph`) with the most time left above its floor, else its group or
    loop with the most; None: no target has measured headroom (no floor, or at it)."""
    for scopes in (("stage",), SCOPES):
        left = [s for s in graph if s.scope in scopes and s.headroom_ms > 0]
        if left:
            return max(left, key=lambda s: s.headroom_ms)
    return None


def at_floor(graph: Sequence[Stage], stop: float | None) -> str | None:
    """Why the native arm stops after its staged plan: every stage of the stage graph runs at
    ``stop`` (``Policy.sol_stop``) or more of its floor (None: one has headroom left, a floor
    is unknown, or there is no stage)."""
    stages = [s for s in graph if s.scope == "stage"]
    fractions = [s.of_floor for s in stages]
    if not stop or not stages or any(f is None for f in fractions):
        return None
    lowest = min(f for f in fractions if f is not None)
    if lowest < stop:
        return None
    return (
        f"at the floor: every stage of the newest profile runs at {lowest:.0%} or more of its "
        f"floor (stop at {stop:.0%})"
    )


# ------------------------------------------------------------------ prompts


def building_blocks(
    winners: Sequence[tuple[str, str, float]], arch: str | None, *, limit: int = 12
) -> list[str]:
    """Markdown lines: this run's best kernels and the kernel library's entries for this
    GPU architecture (earlier runs' winners), fastest first, as building blocks."""
    lines = [f"* this run: `{t}` {s:.2f}x: `{p}`" for t, p, s in winners]
    if arch:
        from kernel_agent import library

        entries = [e for e in library.entries(arch) if e.problem(arch) is None]
        entries.sort(key=lambda e: -e.speedup)
        for e in entries[:limit]:
            repo = e.meta.get("repo_id") or "?"
            lines.append(
                f"* library ({repo}): `{e.path.parent.name}` {e.speedup:.2f}x: "
                f"`{e.path / 'kernel.py'}`"
            )
    return lines or ["* (none yet)"]


@dataclass
class Status:
    """Where the native arm stands (for a slice digest and the scheduler)."""

    plan: list[Stage] = field(default_factory=list)
    bar: float = 1.0
    finished: dict[str, float] = field(default_factory=dict)
    # the current stage of the plan; once the plan is done, the focus (:func:`focus`)
    stage: Stage | None = None
    # the stage graph of the newest ceilings table (what is slow now), and where it is from
    graph: list[Stage] = field(default_factory=list)
    table: str = ""

    @property
    def complete(self) -> bool:
        """Every stage of the plan beat the bar once: a note, not a stop (issue #164)."""
        return bool(self.plan) and all(s.id in self.finished for s in self.plan)

    def note(self) -> str | None:
        """The note of a done plan: what the arm works on now (None: the plan is not done)."""
        if not self.complete:
            return None
        text = "the staged plan is done (every stage beat the best module-level result once)"
        where = f" ({self.table})" if self.table else ""
        if not self.graph:
            return text + "; no ceilings table: no measured headroom per stage"
        if self.stage is None:
            return text + f"; no stage has measured headroom left{where}"
        return text + (
            f"; now {self.stage.id} ({self.stage.scope}), the most time left above its floor: "
            f"{self.stage.headroom_ms:,.4g} ms{where}"
        )


def status(
    run: RunDir, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None
) -> Status:
    """The staged plan (:func:`stages`), its finished stages and its current stage; once
    every stage is done, the stage graph of the newest ceilings table (a re-profile of the
    native arm's best run, :func:`reprofile_dir`) says what to work on next (:func:`focus`)."""
    path = newest_table_path(run)
    table = read_json(path, None) if path else None
    graph = stage_graph(table, columns=columns, fusions=fusion_candidates(run))
    entry = plan_entry(run)
    plan = (planned(entry, graph) if entry else []) or graph
    finished = done(plan, rows)
    stage = next((s for s in plan if s.id not in finished), None)
    table = str(path.relative_to(run.root)) if path is not None else ""
    out = Status(plan, bar(rows), finished, stage, graph, table)
    if out.complete:
        out.stage = focus(graph)
    return out


def native_dir(run: RunDir) -> Path:
    return run.transforms_dir / DIR


# ------------------------------------------------------------------ the megakernel kit

#: Grid-wide synchronisation in a native kernel's sources: cooperative-groups grid syncs,
#: cooperative launches (hand-rolled grid barriers ride on them).
GRID_SYNC = re.compile(
    r"\bthis_grid\s*\(|\bgrid_group\b|\bgrid\.sync\s*\(|cudaLaunchCooperativeKernel"
    r"|\bcooperative\s*=\s*true|\bgrid_barrier\b|\bgrid_sync\b"
)
#: More kernel launches per call than this: one megakernel launch may pay (issue #225).
MEGAKERNEL_LAUNCHES = 3


def megakernel_hint(run: RunDir, rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """Digest lines pointing to the megakernel kit (issue #225) for every native stage target
    whose best kernel so far synchronises its whole grid (:data:`GRID_SYNC` in its project's
    sources) or launches more than :data:`MEGAKERNEL_LAUNCHES` kernels per call (the
    evaluator's ``kernel_launches_candidate``). Empty when none does."""
    best: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        target, speedup = str(row.get("target") or ""), row.get("speedup")
        if not target.startswith(TARGET_PREFIX) or not row.get("correct") or not speedup:
            continue
        if target not in best or float(speedup) > float(best[target]["speedup"]):
            best[target] = row
    hints = []
    for target, row in sorted(best.items()):
        why = []
        snap = run.history_dir(target) / str(row.get("snapshot") or "")
        payload = project.read_bundle(snap) if snap.is_file() else None
        files = (payload or {}).get("files") or {}
        if any(GRID_SYNC.search(str(text)) for text in files.values()):
            why.append("synchronises its whole grid")
        records = read_jsonl(run.results_file(target))
        rec = next((r for r in records if Path(str(r.get("snapshot"))).name == snap.name), {})
        launches = rec.get("kernel_launches_candidate")
        if isinstance(launches, int | float) and launches > MEGAKERNEL_LAUNCHES:
            why.append(f"launches {launches:g} kernels per call")
        if why:
            hints.append(f"* `{target}`: its best kernel `{snap.name}` {' and '.join(why)}.")
    if not hints:
        return []
    return [
        "",
        "## Megakernel kit",
        *hints,
        "* Grid barriers and launch boundaries drain the weight stream (~2.2-2.4 us per hand-"
        "rolled grid barrier, ~0.9 us per graph kernel boundary, measured on an RTX 5070 Ti). "
        "The megakernel kit (`kernel_agent.native.megakernel`: `ka_mk.cuh` on every project's "
        "include path, the scheduler and its simulator) replaces them by counter dependencies "
        "and loads the next instructions' weights while a block waits: the `native-engines` "
        "skill's `megakernel.md` and `examples/native_megakernel`.",
    ]
