"""Native engine targets and the native arm (kernel_agent/native/engine.py, issue #134):
the stage graph from a ceilings table (any model family), the staged plan, the scheduler's
gate and the improve loop's native slices, the planner's ``native`` entry and the prompts."""

from __future__ import annotations

import asyncio
import dataclasses
import re
from pathlib import Path

import pytest

from kernel_agent import cli, dryrun, improve, ledger, orchestrator, program, scheduler
from kernel_agent.agent import prompts
from kernel_agent.agent.tools import record_e2e_result, snapshot
from kernel_agent.budget import Budget
from kernel_agent.config import OptimizeConfig
from kernel_agent.improve import ImproveConfig, Improver, native_digest
from kernel_agent.native import engine
from kernel_agent.native import project as proj
from kernel_agent.scheduler import KERNEL, NATIVE, SYSTEMS, Policy, build_arms, systems_rows
from kernel_agent.workspace import RunDir, read_json, write_json

BASE = dryrun.BASELINE_MS


@pytest.fixture(autouse=True)
def fast(monkeypatch, tmp_path):
    monkeypatch.setattr(orchestrator.toolchain, "setup", dryrun.SimToolchain)
    monkeypatch.setattr(improve, "slices_chart", lambda run: None)
    monkeypatch.setenv(proj.CACHE_ENV, str(tmp_path / "native-cache"))


def row(cls, group, now, *, calls=1, share=None, floor=None, instances=1, phase="prefill"):
    return {
        "cls": cls,
        "group": group,
        "now_ms": now,
        "share": now / 1000.0 if share is None else share,
        "calls": calls,
        "instances": instances,
        "phase": phase,
        "floors": {"exact": floor, "fp8_weights": None if floor is None else floor / 2},
    }


#: A diffusion pipeline (sampler loop + VAE + text encoder) and an LLM (decode steps): the
#: stage graph comes from the table's structure, never from class or module names.
DIFFUSION = {
    "columns": ["exact"],
    "precisions": {"exact": {"label": "bf16"}},
    "rows": [
        row("Sampler", "pipe.sampler", 700, floor=200),
        row("Net", "pipe.sampler.net", 690, calls=30, floor=200),
        row("Block", "pipe.sampler.net.blocks.*", 600, calls=30 * 12, instances=12),
        row("Linear", "pipe.sampler.net.blocks.*.proj", 300, calls=30 * 12 * 4),
        row("Decoder", "pipe.vae", 200, floor=150),
        row("Conv2d", "pipe.vae.conv_in", 150, calls=1),
        row("Encoder", "pipe.text_encoder", 40, floor=10),
        row("Linear", "pipe.text_encoder.fc", 39),
    ],
}
LLM = {
    "columns": ["exact", "fp8_weights"],
    "precisions": {"exact": {"label": "bf16"}, "fp8_weights": {"label": "FP8 w"}},
    "rows": [
        row("Model", "model.model", 900, calls=64, floor=400, phase="decode"),
        row("Layer", "model.model.layers.*", 880, calls=64 * 16, instances=16, phase="decode"),
        row("Linear", "model.model.layers.*.mlp", 500, calls=64 * 16, phase="decode"),
        row("Linear", "model.lm_head", 60, calls=64, phase="decode"),
    ],
}


def test_stage_graph_of_a_diffusion_pipeline():
    stages = engine.stage_graph(DIFFUSION)
    assert [s.id for s in stages] == ["sampler", "vae"]  # by time above the floor
    sampler, vae = stages
    assert sampler.scope == "stage" and sampler.pattern == "solver"
    assert "30× per call" in sampler.inner and sampler.module_class == "Sampler"
    assert sampler.headroom_ms == 500 and sampler.of_floor == pytest.approx(200 / 700)
    assert vae.pattern == "other" and vae.calls == 1
    assert sampler.target_id == "native_sampler"
    assert "text_encoder" not in {s.id for s in stages}  # below MIN_SHARE
    assert "bf16 floor 200 ms" in sampler.describe()


def test_stage_graph_of_an_llm_decode_loop():
    stages = engine.stage_graph(LLM)
    model, loop = stages
    assert model.id == "model" and model.pattern == "stack" and "× 16 layers" in model.inner
    assert model.floor_ms == 200 and model.floor_label == "FP8 w"  # the best allowed floor
    assert loop.scope == "loop" and loop.members == ("model",) and loop.target_id is None
    exact = engine.stage_graph(LLM, columns=["exact"])[0]
    assert exact.floor_ms == 400 and exact.floor_label == "bf16"


def test_loop_body_groups_the_stages_of_one_iteration():
    table = {
        "columns": ["exact"],
        "rows": [
            row("Solver", "m.solver", 500, calls=60, floor=100),
            row("Net", "m.solver.net", 480, calls=540, floor=100),
            row("LM", "m.lm", 250, calls=60, floor=150),
            row("Layer", "m.lm.layers.*", 240, calls=60 * 28, instances=28),
            row("Enc", "m.enc", 80, calls=60, floor=40),
            row("Layer", "m.enc.layers.*", 70, calls=60 * 4, instances=4),
            row("VAE", "m.vae", 70, calls=1, floor=60),
            row("Conv", "m.vae.conv", 60),
        ],
    }
    stages = engine.stage_graph(table)
    assert [s.id for s in stages] == ["solver", "lm", "enc", "vae", "loop_body", "loop"]
    body = stages[-2]
    assert body.scope == "group" and body.members == ("solver", "lm", "enc")
    assert body.calls == 60 and body.now_ms == 830 and body.floor_ms == 290
    assert stages[0].pattern == "solver" and stages[1].pattern == "stack"


def test_plan_entry_orders_and_names_the_stages(tmp_path):
    derived = engine.stage_graph(DIFFUSION)
    entry = {
        "why": "the sampler re-streams its net 30 times",
        "stages": [
            {"id": "vae_fused", "scope": "stage", "group": "pipe.vae", "idea": "one kernel"},
            {"id": "solver", "scope": "stage", "group": "pipe.sampler", "idea": "fuse steps"},
            {"id": "nothing", "scope": "stage", "idea": "no group: dropped"},
            {"id": "whole", "scope": "loop", "members": ["solver"], "idea": "persistent"},
        ],
    }
    plan = engine.planned(entry, derived)
    assert [s.id for s in plan] == ["vae_fused", "solver", "whole"]
    assert plan[0].module_class == "Decoder" and plan[0].now_ms == 200
    assert plan[1].pattern == "solver" and plan[1].idea == "fuse steps"
    assert plan[2].target_id is None


def test_modes_and_target_specs():
    assert engine.enabled("off", {"why": "x"}) is None
    assert engine.enabled(None, None) is None  # the default asks the plan
    assert engine.enabled("plan", {"why": "solver loop"}) == "the plan asks for it: solver loop"
    assert engine.enabled("on", None) == "--native on"
    assert engine.mode("bogus") == engine.DEFAULT_MODE
    stage = engine.stage_graph(DIFFUSION)[0]
    spec = engine.target_spec(stage, "fp8_weights")
    assert spec["id"] == "native_sampler" and spec["native"] and spec["module_class"] == "Sampler"
    assert spec["precision"] == "fp8_weights" and engine.is_stage_target(spec)
    regex = engine.qualname_regex("pipe.sampler.net.blocks.*")
    assert re.search(regex, "pipe.sampler.net.blocks.11")
    assert not re.search(regex, "pipe.sampler.net.blocks.11.proj")
    assert "precision" not in engine.target_spec(stage, "exact")
    specs = [{"precision": "fp8_weights"}, {"precision": "fp8_w8a8"}, {"precision": "fp8_weights"}]
    assert engine.module_precision([*specs, {"native": True, "precision": "x"}]) == "fp8_weights"
    assert engine.module_precision([{"precision": "exact"}]) is None


def e2e(snapshot_name, speedup, *, backend="native", correct=True, exp=1):
    return {
        "exp": exp,
        "target": ledger.E2E,
        "backend": backend,
        "snapshot": snapshot_name,
        "correct": correct,
        "speedup": speedup,
        "spread": 0.002,
        "status": "keep",
    }


def test_each_stage_must_beat_the_module_level_bar_before_the_next():
    plan = engine.stage_graph(DIFFUSION)
    rows = [
        e2e("x", 3.0, backend="integrate"),
        e2e("007_tweak_ab12cd34.py", 3.2, backend="transform"),
    ]
    assert engine.bar(rows) == 3.2
    assert engine.current(plan, rows).id == "sampler"
    rows.append(e2e("009_sampler_0011aabb.py", 3.1))  # slower than the bar: not done
    assert engine.current(plan, rows).id == "sampler"
    rows.append(e2e("010_sampler_v2_0011aabc.py+attn", 3.6, backend="native+kernels"))
    assert engine.stage_of(rows[-1], plan) == "sampler"
    assert engine.done(plan, rows) == {"sampler": 3.6}
    assert engine.current(plan, rows).id == "vae"
    rows.append(e2e("native_vae", 3.8))  # the stage's kernel target in a native run
    assert engine.current(plan, rows) is None
    assert engine.bar(rows) == 3.2  # native runs never raise the bar
    # ... nor an integration that measured the arm's own items (its projects, its stage
    # targets); the module kernels it ran on top of are not its own
    assert engine.own_items(rows) == {"sampler", "sampler_v2", "native_vae"}
    rows.append(e2e("tweak+sampler_v2", 3.9, backend="integrate"))
    rows.append(e2e("native_vae+attn", 3.9, backend="integrate"))
    assert engine.bar(rows) == 3.2 and engine.current(plan, rows) is None
    rows.append(e2e("tweak+attn", 3.4, backend="integrate"))
    assert engine.bar(rows) == 3.4


#: The diffusion pipeline re-profiled after native engines of both stages: the sampler is
#: near its floor now, the vae has the most time left above its own.
MOVED = {
    **DIFFUSION,
    "rows": [
        row("Sampler", "pipe.sampler", 230, floor=200),
        row("Net", "pipe.sampler.net", 220, calls=30, floor=200),
        row("Decoder", "pipe.vae", 200, floor=150),
        row("Conv2d", "pipe.vae.conv_in", 150),
        row("Encoder", "pipe.text_encoder", 40, floor=10),
        row("Linear", "pipe.text_encoder.fc", 39),
    ],
}
#: ... and every stage within 10 % of its floor.
AT_FLOOR = {
    **DIFFUSION,
    "rows": [
        row("Sampler", "pipe.sampler", 210, floor=200),
        row("Net", "pipe.sampler.net", 205, calls=30, floor=200),
        row("Decoder", "pipe.vae", 160, floor=150),
        row("Conv2d", "pipe.vae.conv_in", 150),
    ],
}


def test_after_the_plan_the_stage_with_the_most_headroom_left():
    """Issue #164: once every stage beat the bar, the arm works on the stage of the newest
    stage graph with the most time above its floor, and stops only when every stage is at
    its floor."""
    assert engine.focus(engine.stage_graph(DIFFUSION)).id == "sampler"  # 500 ms; the vae 50
    moved = engine.stage_graph(MOVED)
    assert [s.id for s in moved] == ["vae", "sampler"]
    assert engine.focus(moved).id == "vae" and engine.at_floor(moved, 0.9) is None
    floor = engine.stage_graph(AT_FLOOR)
    assert engine.focus(floor).id == "sampler"  # still 10 ms above it
    assert engine.at_floor(floor, 0.9) == (
        "at the floor: every stage of the newest profile runs at 94% or more of its floor "
        "(stop at 90%)"
    )
    assert engine.at_floor(floor, 0.95) is None and engine.at_floor(floor, None) is None
    unknown = [dataclasses.replace(floor[0], floor_ms=None), floor[1]]
    assert engine.at_floor(unknown, 0.9) is None  # a floor not known: no floor stop
    assert engine.focus([]) is None and engine.at_floor([], 0.9) is None
    # the stages of an LLM decode loop at their floors, the loop above it: the loop
    loop = engine.Stage("loop", "loop", members=("model",), now_ms=500, floor_ms=300)
    model = engine.Stage("model", "stage", group="m.model", now_ms=300, floor_ms=300)
    assert engine.focus([model, loop]) is loop


def test_the_newest_table_is_the_newest_reprofile(tmp_path):
    run = RunDir(tmp_path)
    assert engine.newest_table_path(run) is None and engine.newest_table(run) is None
    write_json(run.profile_dir / "ceilings.json", DIFFUSION)
    assert engine.newest_table_path(run) == run.profile_dir / "ceilings.json"
    native = engine.reprofile_dir(run, 1, 1) / "profile" / "ceilings.json"
    assert native == tmp_path / "rounds/1/native/1/profile/ceilings.json"
    write_json(native, MOVED)  # the native arm's re-profile in round 1: newer than analyze's
    assert engine.newest_table_path(run) == native and engine.newest_table(run) == MOVED
    round2 = tmp_path / "rounds/2/profile/ceilings.json"
    write_json(round2, LLM)  # a later round's re-profile is newer still
    assert engine.newest_table_path(run) == round2
    write_json(engine.reprofile_dir(run, 2, 2) / "profile" / "ceilings.json", MOVED)
    write_json(engine.reprofile_dir(run, 2, 10) / "profile" / "ceilings.json", AT_FLOOR)
    write_json(engine.reprofile_dir(run, 2, 11) / "profile" / "ceilings.json", {"rows": []})
    assert engine.newest_table(run) == AT_FLOOR  # by number; a table without rows is skipped


def test_status_after_the_plan_works_on_what_is_slow_now(tmp_path):
    orch, _ = make(tmp_path)
    run = orch.run
    write_json(run.profile_dir / "ceilings.json", DIFFUSION)
    ledger.record_e2e(
        run, dryrun._e2e_result(BASE / 2.0), backend="integrate", snapshot="x", hypothesis=""
    )
    status = engine.status(run, ledger.rows(run))
    assert not status.complete and status.stage.id == "sampler" and status.note() is None
    assert status.table == "profile/ceilings.json"
    assert [s.id for s in status.graph] == ["sampler", "vae"]
    native_run(run, "sampler", 2.4)
    native_run(run, "vae", 2.5)
    status = engine.status(run, ledger.rows(run))
    assert status.complete and status.finished == {"sampler": 2.4, "vae": 2.5}
    assert status.stage.id == "sampler"  # the most headroom left in the newest table
    write_json(engine.reprofile_dir(run, 1, 1) / "profile" / "ceilings.json", MOVED)
    status = engine.status(run, ledger.rows(run))
    assert status.complete and status.stage.id == "vae"  # derived again from the re-profile
    assert status.table == "rounds/1/native/1/profile/ceilings.json"
    assert status.note() == (
        "the staged plan is done (every stage beat the best module-level result once); now "
        "vae (stage), the most time left above its floor: 50 ms "
        "(rounds/1/native/1/profile/ceilings.json)"
    )


def test_building_blocks_list_this_run_and_the_library(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNEL_AGENT_LIBRARY", str(tmp_path / "library"))  # an empty one
    lines = engine.building_blocks([("attn", "/r/targets/attn/history/003_a.py", 4.2)], None)
    assert lines == ["* this run: `attn` 4.20x: `/r/targets/attn/history/003_a.py`"]
    assert engine.building_blocks([], "sm_120") == ["* (none yet)"]  # empty library


# ------------------------------------------------------------------ scheduler


def make(tmp_path, **cfg):
    cfg.setdefault("quality", "exact")  # an exact run (new runs default to relaxed, #175)
    config = OptimizeConfig(model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path, dossier=False, **cfg)
    orch = orchestrator.Orchestrator(dryrun.create_run(config), config)
    return orch, dryrun.World(orch)


def native_run(run: RunDir, name: str, speedup: float, kernels=(), session=None):
    root = engine.native_dir(run) / name
    (root / "csrc").mkdir(parents=True, exist_ok=True)
    (root / proj.MANIFEST).write_text(
        f'[project]\nname = "{name}"\nkind = "transform"\n[build]\nsources = ["csrc/*.cu"]\n'
    )
    (root / "candidate.py").write_text(f"def apply(workload):\n    pass  # {speedup}\n")
    (root / "csrc" / "e.cu").write_text(f"// {speedup}\n")
    snap = snapshot(run, root)
    result = dryrun._e2e_result(BASE / speedup)
    record_e2e_result(
        run, result, [snap], list(kernels), hypothesis=f"native {speedup}", session=session
    )
    return snap


def test_native_arm_waits_for_the_module_arms_and_tracks_its_runs(tmp_path):
    orch, _ = make(tmp_path)
    run = orch.run
    assert NATIVE not in {a.id for a in build_arms(run, Policy(), [])}  # off unless asked
    arms = {a.id: a for a in build_arms(run, Policy(native=True), [])}
    native = arms[NATIVE]
    assert native.kind == NATIVE and native.agent == "native"
    assert native.stop.startswith("waiting: module arms still improving")
    arms = {a.id: a for a in build_arms(run, Policy(native=True), [], targets=[])}
    assert arms[NATIVE].stop is None  # no module arm left: it opens

    integration = dryrun._e2e_result(BASE / 2.0)
    ledger.record_e2e(run, integration, backend="integrate", snapshot="x", hypothesis="all")
    snap = native_run(run, "loop", 2.4)
    native_run(run, "loop", 2.3)
    rows = ledger.rows(run)
    assert [r["backend"] for r in rows if r["target"] == ledger.E2E][-2:] == ["native"] * 2
    assert not systems_rows(rows)  # not the systems agent's evaluations
    arm = next(a for a in build_arms(run, Policy(native=True), [], targets=[]) if a.id == NATIVE)
    assert arm.best == pytest.approx(1.2, rel=1e-3) and arm.best_snapshot == snap.name
    assert arm.streak == 1 and arm.evals == 2
    assert arm.ref_ms == pytest.approx(BASE / 2.0, rel=1e-3)
    assert arm.remaining_ms == pytest.approx(BASE / 2.4, rel=1e-3)
    assert arm.gain_ms == pytest.approx(BASE * (1 / 2.0 - 1 / 2.4), rel=1e-3)
    assert "over the best module-level result" in arm.why()
    stopped = build_arms(run, Policy(native=True, native_patience=1), [], targets=[])
    assert "plateau" in next(a for a in stopped if a.id == NATIVE).stop


def test_a_systems_stack_on_native_projects_is_the_systems_arms(tmp_path):
    """A row is the arm's whose session made it. Measured on a VoxCPM2 run: a systems
    session's stack on the native projects (exp 146, ledger backend ``native+kernels``) went
    to the native arm by its backend, so its slice counted 0 evaluations and the systems
    best stayed at its last run without a native project (7.37x for 9.51x)."""
    orch, _ = make(tmp_path)
    run = orch.run
    integration = dryrun._e2e_result(BASE / 2.0)
    ledger.record_e2e(run, integration, backend="integrate", snapshot="x", hypothesis="all")
    loop = native_run(run, "loop", 2.4, session="native#1")
    graph = run.transforms_dir / "graph_prefill.py"
    graph.write_text("def apply(workload): pass\n")
    snap = snapshot(run, graph)
    alone = dryrun._e2e_result(BASE / 2.2)
    record_e2e_result(run, alone, [snap], [], hypothesis="graphs", session="systems#2")
    stacked = dryrun._e2e_result(BASE / 2.6)
    record_e2e_result(run, stacked, [loop, snap], [], hypothesis="on the loop", session="systems#3")
    rows = ledger.rows(run)
    stack = rows[-1]
    assert stack["backend"] == "native" and engine.is_native(stack)  # what it ran
    assert not scheduler.native_run(stack) and stack in systems_rows(rows)  # whose it is
    assert scheduler.native_run(rows[1]) and scheduler.made_by(rows[1]) == NATIVE

    arms = {a.id: a for a in build_arms(run, Policy(native=True), [], targets=[])}
    systems, native = arms[SYSTEMS], arms[NATIVE]
    assert systems.evals == 2 and systems.best == pytest.approx(2.6, rel=1e-3)
    assert systems.best_snapshot == stack["snapshot"] and systems.streak == 0
    # the stack had to beat the native arm's 2.4x run too: only that gain is the agent's
    gain = (1.0 - 1.0 / 2.2) + (1.0 / 2.4 - 1.0 / 2.6)
    assert systems.gain_ms == pytest.approx(systems.ref_ms * gain, rel=1e-3)
    assert native.evals == 1 and native.best == pytest.approx(2.4 / 2.2, rel=1e-3)

    improver = Improver(orch, ImproveConfig(agents=2), require_capture=False, live_charts=False)
    arm, new, improved = improver._own({"arm": SYSTEMS, "sessions": ["systems#3"]})
    assert arm is not None and arm.id == SYSTEMS
    assert [r["exp"] for r in new] == [stack["exp"]] and improved  # its slice's own row


def test_stage_targets_belong_to_the_native_arm(tmp_path):
    orch, _ = make(tmp_path)
    run = orch.run
    spec = read_json(run.target("attn") / "spec.json")
    write_json(run.target("attn") / "spec.json", {**spec, "native": True})
    ids = {a.id for a in build_arms(run, Policy(native=True), [])}
    assert "attn" not in ids and NATIVE in ids


def test_the_plan_opens_the_native_arm_and_a_done_plan_is_a_note(tmp_path):
    orch, _ = make(tmp_path)
    run = orch.run
    improver = Improver(orch, ImproveConfig(), require_capture=False, live_charts=False)
    assert improver.native_why() is None and NATIVE not in {a.id for a in improver.arms()}
    plan = read_json(run.plan_json)
    entry = {"why": "glue between stages", "stages": [{"id": "loop", "scope": "loop", "idea": "x"}]}
    write_json(run.plan_json, {**plan, "native": entry})
    assert improver.native_why() == "the plan asks for it: glue between stages"
    assert NATIVE in {a.id for a in improver.arms()}
    ledger.record_e2e(
        run, dryrun._e2e_result(BASE / 2.0), backend="integrate", snapshot="x", hypothesis=""
    )
    native_run(run, "loop", 2.5)
    arm = next(a for a in build_arms(run, Policy(native=True), [], targets=[]) if a.id == NATIVE)
    assert arm.stop is None  # every stage beat the bar once: a note, not a stop (#164)
    assert arm.note.startswith("the staged plan is done") and "no ceilings table" in arm.note
    assert arm.summary()["note"] == arm.note


def native_arm(run: RunDir, policy: Policy | None = None, slices=()):
    arms = build_arms(run, policy or Policy(native=True), list(slices), targets=[])
    return next(a for a in arms if a.id == NATIVE)


def test_a_done_plan_keeps_the_native_arm_live_under_its_stop_rules(tmp_path):
    """Issue #164: every stage beating the bar once is a note; the arm stops by its patience,
    its time cap, or every stage at its floor in the newest ceilings table."""
    orch, _ = make(tmp_path)
    run = orch.run
    write_json(run.profile_dir / "ceilings.json", DIFFUSION)
    ledger.record_e2e(
        run, dryrun._e2e_result(BASE / 2.0), backend="integrate", snapshot="x", hypothesis=""
    )
    native_run(run, "sampler", 2.4)
    assert native_arm(run).note is None  # the vae is next
    native_run(run, "vae", 2.5)
    arm = native_arm(run)
    assert arm.stop is None and "now sampler (stage)" in arm.note
    native_run(run, "sampler_v2", 2.45)  # no new best
    assert native_arm(run).stop is None
    patience = native_arm(run, Policy(native=True, native_patience=1))
    assert patience.stop == "plateau: 1 evaluations in a row without a new best"
    cap = native_arm(run, Policy(native=True, native_hours=1.0), [{"arm": NATIVE, "seconds": 3600}])
    assert cap.stop.startswith("time cap: 1.0 h")
    write_json(engine.reprofile_dir(run, 1, 1) / "profile" / "ceilings.json", AT_FLOOR)
    floor = native_arm(run)
    assert floor.stop.startswith("at the floor: every stage of the newest profile runs at 94%")
    assert native_arm(run, Policy(native=True, sol_stop=None)).stop is None


def test_the_native_best_run_is_reprofiled_once_the_plan_is_done(tmp_path):
    orch, world = make(tmp_path)
    run = orch.run
    write_json(run.profile_dir / "ceilings.json", DIFFUSION)
    ledger.record_e2e(
        run, dryrun._e2e_result(BASE / 2.0), backend="integrate", snapshot="x", hypothesis=""
    )
    improver = Improver(orch, ImproveConfig(), require_capture=False, live_charts=False)
    native_run(run, "sampler", 2.4)
    with world.installed():
        status = asyncio.run(improver._native_stage(native_arm(run)))
        assert not status.complete and status.stage.id == "vae"
        assert "native_profiles" not in read_json(run.root / "improve.json", {})
        best = native_run(run, "vae", 2.5)
        arm = native_arm(run)
        assert arm.best_snapshot == best.name
        assert orch.e2e_items(best.name) == [
            {"kind": "transform", "item": str(run.history_dir() / best.name)}
        ]
        status = asyncio.run(improver._native_stage(arm))
        assert asyncio.run(improver._native_reprofile(arm)) is False  # once per best run
    profiles = read_json(run.root / "improve.json")["native_profiles"]
    assert len(profiles) == 1 and profiles[0]["snapshot"] == best.name
    assert profiles[0]["dir"] == "rounds/1/native/1" and "error" not in profiles[0]
    assert profiles[0]["median_ms"] == pytest.approx(BASE / 2.5, rel=0.01)
    assert status.table == "rounds/1/native/1/profile/ceilings.json"  # the stage graph from it
    table = read_json(run.root / status.table)
    assert table["rows"][0]["now_ms"] == pytest.approx(700 / 2.5, rel=0.01)
    assert status.complete and status.stage.id == "sampler"
    assert orch.e2e_items("nothing.py") is None


def test_dry_run_reaches_the_native_arm_after_the_module_arms(tmp_path):
    orch, world = make(tmp_path, native="on", native_evaluations=2)
    policy = Policy(patience=2, native_patience=2, target_hours=0.5)
    improver = Improver(
        orch, ImproveConfig(policy=policy, rounds=1), require_capture=False, live_charts=False
    )
    with world.installed():
        reason = asyncio.run(improver.improve())
    assert reason.startswith("every arm has stopped")
    slices = read_json(orch.run.root / "improve.json")["slices"]
    arms = [s["arm"] for s in slices]
    assert NATIVE in arms
    first = arms.index(NATIVE)
    kernels = {a.id for a in improver.arms() if a.kind == KERNEL}
    assert kernels <= set(arms[:first])  # every module arm had its slices first
    sessions = [s for s in world.sessions if s["name"] == "native"]
    assert sessions and all("# Improve slice" in s["prompt"] for s in sessions)
    assert "## Where the native engine stands" in sessions[0]["prompt"]
    assert "kernel-agent:native-engines" in sessions[0]["system"]
    rows = [r for r in ledger.rows(orch.run) if engine.is_native(r)]
    assert rows and all(r["snapshot"].split("_")[1] == "loop" for r in rows)
    history = orch.run.history_dir()
    assert all(proj.read_bundle(history / r["snapshot"]) for r in rows)  # bundles snapshotted
    costs = read_json(orch.run.root / "costs.json")
    assert any(label.startswith("native#") for label in costs)


#: A layer stack called once per token that holds most of the run: the staged plan is the
#: stage, then the loop (two native runs over the bar).
BODY = {
    "columns": ["exact"],
    "rows": [
        row("Body", "m.body", 700, calls=128, share=0.7, floor=150, phase="decode"),
        row("Layer", "m.body.layers.*", 680, calls=128 * 28, instances=28, phase="decode"),
    ],
}


def test_dry_run_keeps_improving_after_the_plan_until_patience(tmp_path):
    """Issue #164: a plan done early does not stop the native arm: it works on the stage with
    the most headroom left in a re-profile of its best run until its patience stops it."""
    orch, world = make(tmp_path, native="on", native_evaluations=2)
    run = orch.run
    write_json(run.profile_dir / "ceilings.json", BODY)
    assert [s.id for s in engine.stages(run)] == ["body", "loop"]
    policy = Policy(patience=2, native_patience=3, target_hours=0.5)
    improver = Improver(
        orch, ImproveConfig(policy=policy, rounds=1), require_capture=False, live_charts=False
    )
    with world.installed():
        reason = asyncio.run(improver.improve())
    assert reason.startswith("every arm has stopped")
    assert re.search(r"native: plateau: \d+ evaluations in a row without a new best", reason)
    rows = ledger.rows(run)
    native = [r for r in rows if engine.is_native(r)]
    plan = engine.stages(run)

    def complete(exp: int) -> bool:
        return engine.current(plan, [r for r in rows if r["exp"] <= exp]) is None

    done_at = next(r["exp"] for r in native if complete(r["exp"]))
    after = [r for r in native if r["exp"] > done_at]
    assert after and all(r["snapshot"].split("_")[1] == "body" for r in after)  # the focus
    state = read_json(run.root / "improve.json")
    profiles = state["native_profiles"]
    assert profiles and "error" not in profiles[0] and profiles[0]["dir"] == "rounds/1/native/1"
    assert engine.newest_table_path(run).is_relative_to(run.root / profiles[-1]["dir"])
    notes = [s for s in state["slices"] if s.get("note")]
    assert notes and all(s["arm"] == NATIVE for s in notes)
    assert "now body (stage)" in notes[-1]["note"]
    sessions = [s["prompt"] for s in world.sessions if s["name"] == "native"]
    later = [text for text in sessions if "## After the staged plan" in text]
    assert later and "[**focus**] **body**" in later[-1]
    assert "rounds/1/native/" in later[-1]


def test_native_digest(tmp_path):
    orch, _ = make(tmp_path)
    run = orch.run
    write_json(run.profile_dir / "ceilings.json", DIFFUSION)
    status = engine.status(run, ledger.rows(run), ["exact"])
    arms = build_arms(run, Policy(native=True), [])
    arm = next(a for a in arms if a.id == NATIVE)
    text = native_digest(run, arm, 7, 6, Policy(), status, arms)
    assert "# Improve slice 7" in text and "[**current**] **sampler**" in text
    assert "2. [later] **vae**" in text and "## Module arms" in text
    assert "## After the staged plan" not in text
    ledger.record_e2e(
        run, dryrun._e2e_result(BASE / 2.0), backend="integrate", snapshot="x", hypothesis=""
    )
    native_run(run, "sampler", 2.4)
    native_run(run, "vae", 2.5)
    write_json(engine.reprofile_dir(run, 1, 1) / "profile" / "ceilings.json", MOVED)
    status = engine.status(run, ledger.rows(run), ["exact"])
    text = native_digest(run, arm, 8, 6, Policy(), status, arms)
    # the plan, derived again from the re-profile: the vae has the most time above its floor
    assert "1. [done, 2.500x] **vae**" in text and "2. [done, 2.400x] **sampler**" in text
    assert "## After the staged plan" in text and "a note, not a stop" in text
    assert "`rounds/1/native/1/profile/ceilings.json`" in text
    assert "  1. [**focus**] **vae**" in text and "  2. **sampler**" in text
    assert "8 native runs in a row find no new best, 6 h in native slices" in text


# ------------------------------------------------------------------ config, prompts, budget


def test_cli_native_flags(monkeypatch, tmp_path):
    seen = []

    async def fake_improve(ref, cfg, icfg, *, dry_run=False, seed=0):
        seen.append(cfg)
        return RunDir(tmp_path)

    monkeypatch.setattr(improve, "improve", fake_improve)
    argv = ["improve", "runs/x", "--native", "on", "--native-minutes", "90", "--dry-run"]
    assert cli.main(argv) == 0
    assert seen[0].native == "on" and seen[0].native_minutes == 90.0
    assert seen[0].native_evaluations == 6
    assert cli.main(["improve", "runs/x", "--dry-run"]) == 0
    assert seen[1].native == "plan" and OptimizeConfig("m").native == "plan"


def test_native_sessions_are_longer(tmp_path):
    orch, _ = make(tmp_path, agent_minutes=20.0)
    assert orch.native_minutes() == 60.0
    orch.cfg.native_minutes = 45.0
    assert orch.native_minutes() == 45.0
    budget = Budget(orch.run, agent_minutes=20.0)
    budget.minutes_by_agent["native"] = 45.0
    assert budget.start_agent("native") == pytest.approx(45 * 60)
    assert budget.start_agent("systems") == pytest.approx(20 * 60)


def test_plan_schema_and_prompts_know_the_native_engine():
    schema = prompts.plan_schema()
    native = schema["properties"]["native"]
    assert native["properties"]["stages"]["items"]["properties"]["scope"]["enum"] == list(
        engine.SCOPES
    )
    assert "native" not in schema["required"]
    text = prompts.native_prompt(
        {"repo_id": "org/m", "modality": "llm"},
        {"median_ms": 100.0, "workload": "w"},
        "profile",
        "python",
        "toolchain",
        5,
        why="--native on",
        blocks=["* this run: `attn` 2.00x: `x.py`"],
    )
    assert "kernel-agent:native-engines" in text and "native_project" in text
    assert "`attn` 2.00x" in text and "about 5 evaluations" in text
    assert (prompts.KNOWLEDGE_DIR / "native-engines" / "SKILL.md").is_file()
    assert "6 lookups" in prompts.web_note("native", ["docs.nvidia.com"])
    assert program.role_of("native") == "native"


def test_library_code_names_no_model():
    """The native engine is model-agnostic: no model-specific names in its code or prompts."""
    sources = [engine.__file__, proj.__file__]
    text = "\n".join(Path(p).read_text() for p in sources).lower()
    text += prompts.knowledge("native-engines").lower()
    for name in ("voxcpm", "locdit", "minicpm", "locenc", "audiovae", "llama", "qwen"):
        assert name not in text, name
