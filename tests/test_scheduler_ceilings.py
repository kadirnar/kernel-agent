"""The improve scheduler's expected gains from the ceilings table (issue #122).

``fixtures/voxcpm2_round2.json`` is the state of the VoxCPM2 run
``runs/openbmb--VoxCPM2/20261006-004718-retest2`` before its improve slice 28 (round 2),
trimmed to what the scheduler reads: ``improve.json`` (slices, research, rounds), the
ledger rows of its kernel targets, their specs and kept records, the run's and the round-2
re-profile's ``profile.json`` and the round-2 ``ceilings.json``. The run logged
``slice 28: loc_enc_decode (best 7.64x, expected gain 0.1 ms, score 0; others:
dit_layer__fp8_w8a8 0, vae_decoder__reduced 0)``; the LocDiT, the W8A8 arm's modules, was
665 ms per batched run against a 240 ms W8A8 floor.
"""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from kernel_agent import charts, dryrun, orchestrator, scheduler
from kernel_agent.config import OptimizeConfig
from kernel_agent.improve import ImproveConfig, Improver
from kernel_agent.profiling import ceilings
from kernel_agent.scheduler import SYSTEMS, Policy, build_arms, pick
from kernel_agent.workspace import RunDir, read_json, write_json

FIXTURE = Path(__file__).parent / "fixtures" / "voxcpm2_round2.json"
W8A8 = "dit_layer__fp8_w8a8"


def voxcpm2(tmp_path: Path, *, table: bool = True, edit: Any = None) -> tuple[RunDir, dict]:
    """The fixture's run directory (``table``: with the round-2 ceilings table; ``edit``:
    changes the fixture's data first)."""
    data = json.loads(FIXTURE.read_text())
    if edit is not None:
        edit(data)
    root = tmp_path / "run"
    write_json(root / "run.json", data["run"])
    write_json(root / "baseline.json", data["baseline"])
    for rel, profile in data["profiles"].items():
        write_json(root / rel, profile)
    if table:
        for rel, found in data["ceilings"].items():
            write_json(root / rel, found)
    for target_id, spec in data["specs"].items():
        write_json(root / "targets" / target_id / "spec.json", spec)
        records = "".join(json.dumps(r) + "\n" for r in data["records"][target_id])
        (root / "targets" / target_id / "results.jsonl").write_text(records)
    return RunDir(root), data


def arms_of(run: RunDir, data: dict) -> dict[str, scheduler.Arm]:
    state = data["improve"]
    arms = build_arms(
        run,
        Policy(**state["policy"]),
        state["slices"],
        rounds=state["rounds"],
        rows=data["ledger"],
        research=state["research"],
    )
    return {a.id: a for a in arms}


# ------------------------------------------------------------------ the VoxCPM2 replay


def test_the_round_two_replay_ranks_the_w8a8_locdit_arm_first(tmp_path):
    run, data = voxcpm2(tmp_path)
    arms = arms_of(run, data)
    ranked = sorted(arms.values(), key=lambda a: (a.stop is not None, -a.score))
    assert [a.id for a in ranked if not a.stop] == [W8A8, "vae_decoder__reduced", "loc_enc_decode"]
    assert pick(ranked) is arms[W8A8]

    w8a8 = arms[W8A8]
    c = w8a8.ceiling
    assert c is not None and c.precision == "W8A8" and c.per == "per batched run"
    # UnifiedCFM (the CUDA-graphed LocDiT) 665 + the graphed LocEnc 115 + 12.8 per batched run
    assert c.now == pytest.approx(792.7, abs=0.1)
    assert c.floor == pytest.approx(257.55, abs=0.1)  # 240 (LocDiT) + 16.2 + 1.71 at W8A8
    assert c.upper  # no older table split the parents: all of them, an upper bound
    assert c.factor == pytest.approx(1 / 153.6, rel=1e-4)  # ms per second of audio
    # its best kernel (3.05x of the eager layer, 2,224 ms per run) is slower than now
    assert c.kernel_ms is not None and c.kernel_ms > c.now * c.factor
    assert w8a8.remaining_ms == pytest.approx(792.7 / 153.6, rel=1e-3)
    assert w8a8.headroom == pytest.approx(1 - 257.55 / 792.7, rel=1e-3)
    assert w8a8.stale == 1 and w8a8.expected_ms == pytest.approx(3.484 * 0.7, rel=1e-3)
    why = w8a8.why()
    assert why.startswith("share 66.7% of 7.73 ms: now 792.7 ms per batched run (")
    assert "inside UnifiedCFM model.feat_decoder" in why
    assert "W8A8 floor 257.6 ms → 3.48 ms × 0.7 (1 stale) × index" in why
    assert w8a8.summary()["why"] == why

    loc = arms["loc_enc_decode"].ceiling  # its best kernel (7.64x) is faster than now
    assert (
        loc is not None
        and loc.precision == "FP8 w"
        and loc.rows == "VoxCPMLocEnc model.feat_encoder"
    )
    assert loc.now == pytest.approx(114.6) and loc.kernel_ms is not None
    assert loc.kernel_ms / loc.factor == pytest.approx(48.6, abs=0.1)
    assert "its best kernel 48.63 ms, FP8 w floor 19.96 ms" in arms["loc_enc_decode"].why()
    vae = arms["vae_decoder__reduced"].ceiling  # a transform wrapped the decoder: same qualname
    assert vae is not None and vae.precision == "bf16"
    assert vae.rows == "_TF32Scope model.audio_vae.decoder" and not vae.upper
    assert vae.floor == pytest.approx(2.762e12 / 99.9e12 * 1e3, rel=0.01)  # its FLOPs at bf16
    # the arms stopped in the run stay stopped (no evaluation in their last 2 slices)
    assert {a for a, arm in arms.items() if arm.stop} == {"dit_layer", "vae_decoder", SYSTEMS}


def test_without_a_table_the_replay_is_the_runs_amdahl_pick(tmp_path):
    """The round-2 profile without its ceilings: the scheduler's numbers before #122, as
    the run logged them (its class shares × the baseline ÷ the best speedup). The scores
    differ a little: the systems agent's end-to-end rows, which enter every arm's UCB
    index, are not in the fixture."""
    run, data = voxcpm2(tmp_path, table=False)
    arms = arms_of(run, data)
    assert all(a.ceiling is None for a in arms.values())
    ranked = sorted((a for a in arms.values() if not a.stop), key=lambda a: -a.score)
    assert [a.id for a in ranked] == ["loc_enc_decode", W8A8, "vae_decoder__reduced"]
    assert data["slice"]["arm"] == "loc_enc_decode"
    assert ranked[0].expected_ms == pytest.approx(data["slice"]["expected_ms"], abs=5e-4)
    assert arms[W8A8].expected_ms == pytest.approx(0.0563, abs=1e-4)  # 0.1 ms, "score 0"
    assert arms[W8A8].remaining_ms == pytest.approx(0.1026, abs=1e-4)  # the LM's layers
    assert arms[W8A8].why().startswith("Amdahl: 0.313 ms at 1.0x ÷ its best 3.05x = 0.103 ms")


def test_an_arm_takes_the_floor_of_its_own_precision(tmp_path):
    """The exact ``dit_layer`` and its W8A8 pivot hold the same modules; the LocDiT already
    runs below its exact floor (the run's W8A8 transforms), so only the pivot has headroom."""
    run, data = voxcpm2(tmp_path)
    arms = arms_of(run, data)
    exact, pivot = arms["dit_layer"].ceiling, arms[W8A8].ceiling
    assert exact is not None and pivot is not None
    assert exact.now == pivot.now and exact.precision == "exact"
    assert exact.floor > exact.now and arms["dit_layer"].headroom == 0.0
    assert pivot.floor < pivot.now / 3


def test_a_group_no_case_stands_for_is_left_out(tmp_path):
    """A capture since #119 knows the calls its cases cover per instance group: the LocDiT
    layer's case covers none of the LocEnc layers' calls, so the LocEnc rows are not its."""

    def covered(data: dict) -> None:
        data["specs"][W8A8]["capture"].update(
            cases=[{"method": "forward", "count": 540, "target_calls": 6480}],
            instance_groups={
                "model.feat_decoder.estimator.decoder.layers.*": {
                    "instances": 12,
                    "calls": 6480,
                    "covered": 6480,
                },
                "model.feat_encoder.encoder.layers.*": {
                    "instances": 12,
                    "calls": 732,
                    "covered": 0,
                },
            },
        )

    run, data = voxcpm2(tmp_path, edit=covered)
    c = arms_of(run, data)[W8A8].ceiling
    assert c is not None and c.rows == "inside UnifiedCFM model.feat_decoder"
    assert c.now == pytest.approx(665.0, abs=0.5) and c.floor == pytest.approx(239.6, abs=0.1)


def test_a_hidden_group_takes_its_share_of_the_parent_from_an_older_table(tmp_path):
    """The analyze profile's table saw the LocDiT layers themselves: they took 90 % of
    ``UnifiedCFM``'s time there, so they take 90 % of it in the round's table, with their own
    floor (their work is the same math)."""
    run, data = voxcpm2(tmp_path)
    table = copy.deepcopy(data["ceilings"]["rounds/2/profile/ceilings.json"])
    cfm = next(r for r in table["rows"] if r["cls"] == "UnifiedCFM")
    layers = {
        **cfm,
        "cls": "MiniCPMDecoderLayer",
        "group": "model.feat_decoder.estimator.decoder.layers.*",
        "target": "MiniCPMDecoderLayer@model.feat_decoder.estimator.decoder.layers.*",
        "flops": {k: v // 2 for k, v in cfm["flops"].items()},
    }
    eager = {**table, "rows": [{**cfm, "now_ms": 3939.0}, {**layers, "now_ms": 3545.1}]}
    write_json(run.profile_dir / "ceilings.json", eager)
    c = arms_of(run, data)[W8A8].ceiling
    assert c is not None
    part = c.rows.split("; ")[0]
    assert part == "inside 90% of UnifiedCFM model.feat_decoder"
    assert c.upper  # the LocEnc layers: no table saw them
    own = ceilings.floor_ms([layers], ceilings.TARGET_PRECISIONS["fp8_w8a8"], table["peaks"])
    assert own is not None and own < 239.0
    assert c.now == pytest.approx(0.9 * 665.0 + 114.6 + 12.8, abs=0.5)
    assert c.floor == pytest.approx(own + 16.2 + 1.71, abs=0.1)


def test_no_floor_falls_back_to_amdahl(tmp_path):
    """A peak the arm's precision needs was not measured: no ratio is assumed."""

    def no_fp8(data: dict) -> None:
        peaks = data["ceilings"]["rounds/2/profile/ceilings.json"]["peaks"]
        peaks["tflops"].pop("float8_e4m3fn")

    run, data = voxcpm2(tmp_path, edit=no_fp8)
    arms = arms_of(run, data)
    assert arms[W8A8].ceiling is None and arms[W8A8].why().startswith("Amdahl:")
    assert arms["loc_enc_decode"].ceiling is not None  # FP8 weights: bf16 math, no FP8 peak


def test_systems_estimate_from_the_end_to_end_line(tmp_path):
    run, data = voxcpm2(tmp_path)
    systems = arms_of(run, data)[SYSTEMS]
    # near-lossless: W8A8 ≥ 502 of 1,188 ms per batched run = 3.27 ms per second of audio
    assert systems.estimate == pytest.approx(37.659 / (501.7 / 153.6), rel=1e-3)
    assert "(estimate 11.5x: W8A8 floors ≥ 501.7 of 1,188 ms per batched run)" in systems.why()

    exact, data = voxcpm2(
        tmp_path / "exact", edit=lambda d: d["run"]["config"].update(quality="exact")
    )
    assert arms_of(exact, data)[SYSTEMS].estimate == pytest.approx(
        37.659 / (1027.0 / 153.6), rel=1e-3
    )


# ------------------------------------------------------------------ the table's rows


def _row(cls: str, group: str, phase: str, now: float, **kw: Any) -> dict[str, Any]:
    return {
        "cls": cls,
        "group": group,
        "phase": phase,
        "now_ms": now,
        "target": f"{cls}@{group}",
    } | kw


def test_holding_finds_own_rows_wrappers_and_parents():
    table = {
        "rows": [
            _row("Model", "model", "prefill", 100.0),
            _row("Block", "model.layers.*", "prefill", 60.0),
            _row("Block", "model.layers.*", "decode", 30.0),
            _row("Linear", "model.layers.*.attn.{k_proj,q_proj}", "prefill", 9.0),
            _row("Wrapper", "model.vae", "prefill", 7.0),
            _row("Graphed", "model.dit", "prefill", 50.0),
            _row("Graphed", "model.dit", "decode", 5.0),
        ]
    }
    names = lambda rows: [(r["cls"], r["phase"]) for r in rows]  # noqa: E731
    rows, inside = ceilings.holding(table, "model.layers.*", "Block", "decode")
    assert names(rows) == [("Block", "decode")] and not inside
    rows, _ = ceilings.holding(table, "model.layers.*", "Block")
    assert names(rows) == [("Block", "prefill"), ("Block", "decode")]
    # its rows are in other phases only: no row says what its calls in this one take
    assert ceilings.holding(table, "model.layers.*", "Block", "verify") == ([], False)
    rows, inside = ceilings.holding(table, "model.layers.*.attn.q_proj", "Linear")
    assert names(rows) == [("Linear", "prefill")] and not inside  # siblings share a row
    rows, inside = ceilings.holding(table, "model.vae", "CausalDecoder")  # wrapped in place
    assert names(rows) == [("Wrapper", "prefill")] and not inside
    rows, inside = ceilings.holding(table, "model.dit.blocks.*", "DitBlock", "prefill")
    assert names(rows) == [("Graphed", "prefill"), ("Graphed", "decode")] and inside
    rows, inside = ceilings.holding(table, "model.layers.*.mlp", "MLP")  # the innermost parent
    assert names(rows) == [("Block", "prefill"), ("Block", "decode")] and inside
    assert ceilings.holding(table, "other.thing", "X") == ([], False)
    assert ceilings.patterns({"group": "a.*.{k,q}"}) == ["a.*.k", "a.*.q"]


def test_target_precisions():
    assert ceilings.target_precision(None).label == "exact"
    assert ceilings.target_precision("fp8_w8a8") is ceilings.PRECISIONS["w8a8"]
    assert ceilings.target_precision("fp8_weights") is ceilings.PRECISIONS["fp8_weights"]
    assert ceilings.target_precision("fp4_weights") is ceilings.PRECISIONS["fp4_weights"]
    reduced = ceilings.target_precision("reduced")  # fp32 work at the bf16 peak, 2 B weights
    row = {"flops": {"float32": int(4e12)}, "weight_elems": 10**9, "weight_bytes": 4 * 10**9}
    row |= {"io_bytes": 0, "calls": 1}
    peaks = {"tflops": {"float32": 20.0, "bfloat16": 100.0}, "dram_gbps": 1000.0}
    assert ceilings.floor_ms([row], reduced, peaks) == pytest.approx(40.0)
    assert ceilings.floor_ms([row], ceilings.PRECISIONS["exact"], peaks) == pytest.approx(200.0)
    assert ceilings.floor_ms([{**row, "work_known": False}], reduced, peaks) is None
    assert ceilings.floor_ms([row], reduced, None) is None
    # #257: on a GPU without bf16 tensor cores (Turing) `reduced` runs at the fp16 peak
    turing = {"arch": "sm_75", "tflops": {"float32": 8.0, "bfloat16": 8.0, "float16": 65.0}}
    fp16 = ceilings.target_precision("reduced", turing | {"dram_gbps": 320.0})
    assert fp16 is ceilings.REDUCED_FP16 and fp16.label == "fp16"
    assert ceilings.floor_ms([row], fp16, turing | {"dram_gbps": 1e6}) == pytest.approx(4e3 / 65)
    assert ceilings.target_precision("reduced", {"arch": "sm_86"}) is reduced
    assert ceilings.target_precision("exact", turing).label == "exact"


def test_a_reduced_arm_on_turing_takes_the_fp16_floor():
    spec = {"module_class": "Block", "qualname": "model.layers.3", "precision": "reduced"}
    row = _row("Block", "model.layers.*", "prefill", 80.0, flops={"float32": int(65e12)})
    row |= {"weight_elems": 10**6, "weight_bytes": 4 * 10**6, "io_bytes": 0, "calls": 4}
    table = {"baseline_ms": 2000.0, "per": "per run", "rows": [row]}
    table["peaks"] = {"arch": "sm_75", "dram_gbps": 320.0}
    table["peaks"]["tflops"] = {"float32": 8.0, "bfloat16": 8.0, "float16": 65.0}
    groups = [scheduler.projection.Group("t", "model.layers.*", 1.0)]
    profiles = [{"baseline_ms": 2000.0, "ceilings": table}]
    ceiling = scheduler.arm_ceiling(spec, groups, profiles)
    assert ceiling is not None and ceiling.precision == "fp16"
    assert ceiling.floor == pytest.approx(1000.0)  # 65 TFLOP at 65 TFLOP/s, not at 8
    table["peaks"]["arch"] = "sm_86"
    assert scheduler.arm_ceiling(spec, groups, profiles).floor == pytest.approx(65e3 / 8)


def test_one_instance_of_a_group_takes_its_share():
    spec = {"module_class": "Block", "qualname": "model.layers.3", "precision": None}
    row = _row("Block", "model.layers.*", "prefill", 80.0, flops={"bfloat16": int(1e12)})
    row |= {"weight_elems": 10**8, "weight_bytes": 2 * 10**8, "io_bytes": 0, "calls": 4}
    table = {"baseline_ms": 200.0, "per": "per run", "rows": [row]}
    table["peaks"] = {"tflops": {"bfloat16": 100.0}, "dram_gbps": 1000.0}
    groups = [scheduler.projection.Group("t", "model.layers.*", 1.0, None, 0.25)]
    c = scheduler.arm_ceiling(spec, groups, [{"baseline_ms": 20.0, "ceilings": table}])
    assert c is not None and c.now == pytest.approx(20.0) and c.floor == pytest.approx(2.5)
    assert c.rows == "1 instance: Block model.layers.*" and c.factor == pytest.approx(0.1)
    assert scheduler.arm_ceiling({**spec, "kind": "region"}, groups, [{"ceilings": table}]) is None


# ------------------------------------------------------------------ the slice log


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr(orchestrator.toolchain, "setup", dryrun.SimToolchain)
    monkeypatch.setattr(charts, "available", lambda: False)


def test_the_slice_log_names_the_score_components(tmp_path, capsys, quiet):
    config = OptimizeConfig(model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path)
    orch = orchestrator.Orchestrator(dryrun.create_run(config), config)
    world = dryrun.World(orch)
    improver = Improver(orch, ImproveConfig(max_slices=2), require_capture=False, live_charts=False)
    with world.installed():
        asyncio.run(improver.improve())
    out = capsys.readouterr().out
    slices = read_json(orch.run.root / "improve.json")["slices"]
    assert all(s["why"] and " × index " in s["why"] for s in slices)
    for s in slices:
        assert f"score {s['score']:.3g}: {s['why']}; others:" in out
