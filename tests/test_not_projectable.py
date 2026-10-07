"""A projection the estimates take past the run is "not projectable", never a ratio (#128).

``runs/openbmb--VoxCPM2/20261006-004718-retest2`` printed "Projected from the best kernels:
**0.0 ms** (37659015247.39x vs eager)": the W8A8 DiT layer kernel (captured before #119, its
estimate an even split over the 24 LocDiT + LocEnc layers), the reduced VAE decoder and the
LocEnc kernel saved 42.6 ms per second of audio by their module-level estimates against a
37.7 ms baseline, most of which the integration's transforms had already taken. The
projection was clamped at 0 and the ratio divided by ~1e-9. Now such a projection says why
(the items and their savings, ``not_additive`` as an accepted set of the integration, #121),
the run shows the projection of the integration's last accepted set where it has one, and
the targets captured before #119 come with a hint how to capture them again.

CPU only: a run directory of the retest2 run's shape (its baseline, profile classes, target
specs and captures, best kernels and accepted sets), rendered by report, status, the
dashboard, ``watch`` and ``progress.png``.
"""

from __future__ import annotations

import re

import pytest
from test_projection import _texts

from kernel_agent import charts, ledger, projection, watch
from kernel_agent.agent.tools import record_candidate, snapshot
from kernel_agent.dashboard import write_dashboard
from kernel_agent.kernels import weights
from kernel_agent.report import write_report
from kernel_agent.status import render
from kernel_agent.workspace import RunDir, read_json, write_json

BASE = 37.65901524739472  # ms per second of generated audio (metric=throughput)
AUDIO_S = 153.6
BASELINE = {
    "workload": "VoxCPMBatchWorkload(openbmb/VoxCPM2, bfloat16; batch_size=16, metric=throughput)",
    "median_ms": BASE,
    "metric": "throughput",
    "metric_detail": {"throughput": 26.5541, "audio_s": AUDIO_S, "run_ms": 5784.4247},
}
LAYERS = {
    "model.base_lm.layers.*": 28,
    "model.feat_decoder.estimator.decoder.layers.*": 12,
    "model.feat_encoder.encoder.layers.*": 12,
    "model.residual_lm.layers.*": 8,
}


def _cls(name: str, groups: dict[str, int], phases: dict[str, dict[str, int]]) -> dict:
    """A ``profile.json`` class entry as the run's: instances by folded qualname, per phase."""
    return {
        "root": "model",
        "cls": name,
        "instances": sum(groups.values()),
        "inclusive_ms": 1.0,
        "example_qualname": next(iter(groups)).replace("*", "0"),
        "groups": groups,
        "phases": {p: {"instances": sum(g.values()), "groups": g} for p, g in phases.items()},
    }


PROFILE = [
    _cls("MiniCPMDecoderLayer", LAYERS, {"prefill": LAYERS}),
    _cls(
        "VoxCPMLocEnc",
        {"model.feat_encoder": 1},
        {"decode": {"model.feat_encoder": 1}, "prefill": {"model.feat_encoder": 1}},
    ),
    _cls(
        "CausalDecoder", {"model.audio_vae.decoder": 1}, {"prefill": {"model.audio_vae.decoder": 1}}
    ),
]


def _capture(instances: int, *counts: int) -> dict:
    """A capture from before #119: calls per case of the captured instance, the instances
    calling ``forward``; no ``target_calls``, no ``instance_groups``."""
    cases = [{"method": "forward", "count": n} for n in counts]
    return {"cases": cases, "method_instances": {"forward": instances}}


DIT = {
    "module_class": "MiniCPMDecoderLayer",
    "qualname_regex": r"feat_decoder\.|feat_encoder\.",
    "phase": "prefill",
    "capture": _capture(24, 540, 0),  # the LocDiT layer's 540 calls × 24: an even split
}
SPECS = {
    "dit_layer": DIT,
    "dit_layer__fp8_w8a8": {**DIT, "precision": "fp8_w8a8", "pivot_of": "dit_layer"},
    "loc_enc_decode": {
        "module_class": "VoxCPMLocEnc",
        "qualname": "model.feat_encoder",
        "phase": "decode",
        "precision": "fp8_weights",
        "capture": _capture(1, 20, 20, 20, 0),
    },
    "vae_decoder": {"module_class": "CausalDecoder", "capture": _capture(1, 1, 0, 0)},
    "vae_decoder__reduced": {
        "module_class": "CausalDecoder",
        "precision": "reduced",
        "pivot_of": "vae_decoder",
        "capture": _capture(1, 1, 0, 0),
    },
}
#: The best kernel of each target: (module speedup, est. saved ms per run of the workload).
KERNELS = {
    "dit_layer": (3.090, 4650.670),
    "dit_layer__fp8_w8a8": (6.235, 5770.829),
    "loc_enc_decode": (8.127, 322.004),
    "vae_decoder__reduced": (9.642, 453.126),
}

# The integration's items: est. saved ms per second of audio (a kernel's module-level
# estimate, a transform's gain alone) and the modules the patcher recorded for each.
HISTORY = "/r/.truth/transforms/history"
MERGE, CFM = f"{HISTORY}/153_merge_locdit_projections.py", f"{HISTORY}/155_graph_cfm_solver.py"
LM, LOCENC = f"{HISTORY}/156_graph_lm_step.py", f"{HISTORY}/157_graph_loc_enc.py"
VAE_CL, FUSED = f"{HISTORY}/158_vae_channels_last.py", f"{HISTORY}/159_fused_lm_step.py"
VAE_BF16, ATTN = f"{HISTORY}/162_vae_bf16.py", f"{HISTORY}/166_fp8_lm_attn_proj.py"
W8A8 = "dit_layer__fp8_w8a8=/r/.truth/targets/dit_layer__fp8_w8a8/history/012_cuda_w8a8.py"
VAE = "vae_decoder__reduced=/r/.truth/targets/vae_decoder__reduced/history/008_cuda_ru.py"
SAVED = {
    MERGE: 1.434,
    CFM: 16.261,
    LM: 7.083,
    LOCENC: 1.842,
    VAE_CL: 1.668,
    FUSED: 6.986,
    VAE_BF16: 1.085,
    ATTN: 6.965,
    W8A8: round(5770.829 / AUDIO_S, 3),
    VAE: round(453.126 / AUDIO_S, 3),
}
CHANGES = {
    MERGE: ["model.feat_decoder.estimator.decoder.layers.*.mlp"],
    CFM: ["model.feat_decoder"],
    LM: ["workload.lm_step"],
    LOCENC: ["model.feat_encoder"],
    VAE_CL: ["model.audio_vae.decoder"],
    FUSED: ["workload.lm_step"],
    VAE_BF16: ["model.audio_vae"],
    ATTN: ["workload.lm_step"],
    W8A8: ["model.feat_decoder.estimator.decoder.layers.*", "model.feat_encoder.encoder.layers.*"],
    VAE: ["model.audio_vae.decoder"],
}
FIRST = [MERGE, CFM, LM, LOCENC, VAE_CL, FUSED, VAE_BF16, ATTN]
#: The accepted sets: (items, measured ms, the set before it measured in the step's A/B).
SETS = [
    (FIRST, 7.664, None),
    ([*FIRST, W8A8], 6.587, 7.70),  # its estimated gain is more than the set before it
    ([MERGE, CFM, LM, LOCENC, VAE, FUSED, ATTN, W8A8], 6.032, 6.594),
]


def make_run(tmp_path, *, integration: bool = True) -> RunDir:
    """A run of the retest2 run's shape, finished, with its best kernels kept and (with
    ``integration``) its three accepted sets."""
    run = RunDir(tmp_path / "openbmb--VoxCPM2" / "20261006-004718-retest2")
    run.root.mkdir(parents=True)
    card = {"repo_id": "openbmb/VoxCPM2", "modality": "tts", "architectures": ["voxcpm2"]}
    data = {
        "card": card,
        "created": "2026-10-06 00:47:18",
        "config": {"quality": "near-lossless"},  # its targets' FP8 / reduced precisions (#131)
        "phases": {"report": {"done": True}},
    }
    write_json(run.run_json, data)
    write_json(run.toolchain_json, {"gpu": {"name": "NVIDIA GeForce RTX 5070 Ti"}})
    write_json(run.baseline_json, BASELINE)
    write_json(run.profile_dir / "profile.json", {"classes": PROFILE})
    plan = [
        {"id": t, "module_class": SPECS[t]["module_class"]} for t in ("dit_layer", "vae_decoder")
    ]
    write_json(run.plan_json, {"analysis": "(retest2)", "targets": plan})
    for target, spec in SPECS.items():
        write_json(run.target(target) / "spec.json", {"id": target, **spec})
    for target, (speedup, saved) in KERNELS.items():
        src = run.target(target) / "candidates" / "best.py"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text(f"# {target}\n")
        result = {"status": "ok", "correct": True, "speedup": speedup, "cases": []}
        result["est_saved_ms_per_run"] = saved
        record_candidate(run, target, src, snapshot(run, src, target), result, hypothesis="best")
    if integration:
        sets = [
            {
                "items": items,
                "est_saved_ms": {a: SAVED[a] for a in items},
                "measured_ms": measured,
                "from_ms": a_ms,
            }
            for items, measured, a_ms in SETS
        ]
        entries = projection.of_sets(projection.tree(run), sets, BASE, CHANGES)
        final = {"passed": True, "median_ms": 6.032, "baseline_ms": BASE, "speedup": 6.2431}
        write_json(
            run.root / "integration.json",
            {"baseline_ms": BASE, "final": final, "history": [], "projection": entries},
        )
    return run


#: What retest2 printed (current main before #128) and must never print again.
ABSURD = re.compile(r"37659015247|\*\*0\.0 ms\*\*|projected 0\.0 ms|1e-9")
REASON = (
    "the savings counted (42.6 ms) are more than the baseline (37.7 ms); the largest: "
    "dit_layer__fp8_w8a8 37.6 ms, vae_decoder__reduced 3.0 ms, loc_enc_decode 2.1 ms"
)


def test_the_best_kernels_of_retest2_are_not_projectable(tmp_path):
    run = make_run(tmp_path, integration=False)
    proj = ledger.summary(run)["projection"]
    assert proj.saved_ms == pytest.approx(42.617, abs=1e-3)  # 37.571 + 2.950 + 2.096
    assert proj.used == ["dit_layer__fp8_w8a8", "vae_decoder__reduced", "loc_enc_decode"]
    assert proj.left_out == ["dit_layer"]
    assert proj.projected_ms is None and proj.not_additive == REASON
    assert proj.headline() == f"not projectable: {REASON}; not counted (nested): dit_layer"
    assert proj.as_dict()["not_additive"] == REASON and proj.as_dict()["projected_ms"] is None

    text = write_report(run).read_text()
    assert not ABSURD.search(text)
    assert (
        "Projected from the best kernels (baseline − est. saved ms per second of generated "
        f"audio): not projectable: {REASON}; not counted (nested): dit_layer. Module-level"
    ) in text
    assert "Projected for the integration's last accepted set" not in text  # none

    status = render(run, width=400)
    assert not ABSURD.search(status)
    assert "baseline 37.7 ms  |  projected: not projectable  |  measured —" in status
    assert f"not projectable: {REASON}; not counted (nested): dit_layer" in status

    html = write_dashboard(run).read_text()
    assert not ABSURD.search(html)
    tile = re.search(r'<div class="label">projected</div>(.*?)</div></div>', html, re.S)
    assert tile is not None
    assert '<div class="value">—</div>' in tile.group(1) and "×" not in tile.group(1)
    assert "not projectable: the savings counted (42.6 ms)" in tile.group(1)

    state = watch.state(run)["summary"]
    assert state["projected_ms"] is None
    assert state["projected_note"].startswith(f"not projectable: {REASON}")
    assert state["projection"]["not_additive"] == REASON
    steps = state["projection"]["steps"]
    *_, end = steps.values()
    assert next(iter(steps.values())) > 0 and end is None  # the line stops

    if charts.available():
        texts = _texts(run)
        assert not any(ABSURD.search(t) for t in texts)
        subtitle = next(t for t in texts if "kernel evaluations" in t)
        assert "projected from the best kernels: not projectable\nnot projectable: the" in subtitle
        assert any(t.endswith(", then not projectable") for t in texts)  # where the line stops


def test_the_run_shows_the_integrations_last_accepted_set_instead(tmp_path):
    run = make_run(tmp_path)
    s = ledger.summary(run)
    integration = read_json(run.root / "integration.json")
    first, second, last = projection.accepted_sets(run, integration, BASE)
    assert first["projected_ms"] == pytest.approx(first["summed_ms"])
    assert second["projected_ms"] is None and second["not_additive"].startswith(
        "the estimated gain of dit_layer__fp8_w8a8"
    )
    projected = last["projected_ms"]
    assert projected == pytest.approx(6.594 - last["step"]["est_gain_ms"])
    assert 0 < projected < BASE
    assert s["projected_ms"] == projected and s["shown"].source == projection.INTEGRATION
    note = s["shown"].note()
    assert note == (
        "projected for the integration's last accepted set; the best kernels alone: "
        f"not projectable: {REASON}; not counted (nested): dit_layer"
    )

    text = write_report(run).read_text()
    assert not ABSURD.search(text)
    assert (
        f"Projected for the integration's last accepted set: **{projected:.1f} ms** "
        f"({BASE / projected:.2f}x vs eager, measured 6.0 ms)"
    ) in text
    assert f"not projectable: {REASON}" in text  # the kernels' alone, as before

    status = render(run, width=400)
    assert f"projected {projected:.1f} ms ({BASE / projected:.2f}x)" in status
    assert note in status and not ABSURD.search(status)

    html = write_dashboard(run).read_text()
    assert f"{projected:.1f} ms" in html and f"{BASE / projected:.2f}×" in html
    state = watch.state(run)["summary"]
    assert state["projected_ms"] == projected and state["projected_note"] == note

    if charts.available():
        subtitle = next(t for t in _texts(run) if "kernel evaluations" in t)
        assert (
            "projected for the integration's last accepted set: "
            f"{projected:.1f} ms ({BASE / projected:.2f}×)\nthe best kernels (the step line): "
            "not projectable: the savings counted (42.6 ms)"
        ) in subtitle


def test_a_hint_names_the_targets_captured_before_119_and_how_to_capture_them(tmp_path):
    run = make_run(tmp_path)
    assert projection.even_split(run) == ["dit_layer", "dit_layer__fp8_w8a8"]  # 24 instances
    hint = projection.recapture_hint(run)
    assert hint.startswith("captured before #119: dit_layer, dit_layer__fp8_w8a8 (")
    assert (
        f"to capture dit_layer again: `kernel-agent resume {run.root} --redo capture "
        "--until capture` (every target of plan.json)"
    ) in hint
    assert hint.endswith("dit_layer__fp8_w8a8: not in plan.json, only a new run captures it again")
    status = " ".join(render(run, width=120).split())  # wrapped, not cut
    assert " ".join(hint.split()) in status
    assert hint[0].upper() + hint[1:] + "." in write_report(run).read_text()

    # a capture since #119 (calls per instance group) or of one instance: nothing to say
    groups = {"model.feat_decoder.estimator.decoder.layers.*": {"calls": 6480, "covered": 6480}}
    cases = [{"method": "forward", "count": 540, "target_calls": 6480}]
    assert not weights.even_split({**DIT["capture"], "cases": cases, "instance_groups": groups})
    assert not weights.even_split(SPECS["loc_enc_decode"]["capture"])
    assert not weights.even_split({})  # no capture (a failed one)
    for target in ("dit_layer", "dit_layer__fp8_w8a8"):
        write_json(run.target(target) / "spec.json", {**DIT, "capture": {"cases": cases}})
    assert projection.recapture_hint(run) == ""


def test_projectable():
    assert projection.projectable(5.3, BASE) and projection.projectable(BASE, BASE)
    assert projection.projectable(round(BASE, 3), BASE)  # rounded up by less than 0.001 ms
    assert not projection.projectable(0.0, BASE) and not projection.projectable(-4.9, BASE)
    assert not projection.projectable(BASE + 0.01, BASE)
    assert not projection.projectable(None, BASE) and not projection.projectable(5.3, None)


def test_a_set_projected_above_the_baseline_is_not_projectable():
    tree = projection.Tree()
    slow, fast = "/t/history/001_slow.py", "/t/history/002_fast.py"
    # the first set: an item slower alone by more than the other saves
    (alone,) = projection.of_sets(
        tree, [{"items": [slow, fast], "est_saved_ms": {slow: -12.0, fast: 4.0}}], 100.0
    )
    assert alone["summed_ms"] == 108.0 and alone["projected_ms"] is None
    assert alone["not_additive"] == (
        "the items slower alone (12.0 ms) are more than the savings counted (4.0 ms); the "
        "slowest: slow 12.0 ms"
    )
    # a later set: its step's estimated loss takes the set before it past the baseline
    sets = [
        {"items": [fast], "est_saved_ms": {fast: 4.0}, "measured_ms": 90.0},
        {
            "items": [fast, slow],
            "est_saved_ms": {fast: 4.0, slow: -12.0},
            "measured_ms": 89.0,
            "from_ms": 95.0,
        },
    ]
    first, second = projection.of_sets(tree, sets, 100.0)
    assert first["projected_ms"] == 96.0 and second["step"]["est_gain_ms"] == -12.0
    assert second["projected_ms"] is None
    assert second["not_additive"] == (
        "the estimated gain of slow (-12.0 ms) takes the 95.0 ms of the set before it "
        "above the baseline (100.0 ms)"
    )
