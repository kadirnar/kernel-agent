"""``--quality relaxed`` (#175): about twice near-lossless's error budgets, the default of
new runs.

CPU only, no Claude: the mode and its tiers (``kernels/compare.py``), the default of new
runs and the recorded mode of old ones (``config.py``, the CLI, the KernelBench suite), the
precisions it allows, the planner / engineer / research / systems prompts, the perceptual
gate's and the sanity floor's thresholds (``perceptual.RELAXED_GATE``,
``Workload.relaxed_options``; the analyze + e2e flow on the toy workload), and the report
and status lines. The calibration of the tiers on redrawn inputs, and that blatant bugs
still fail, is in ``test_perturbed_calibration.py`` and ``test_fp8_toolkit.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch

from kernel_agent import cli, config, dryrun, orchestrator, pivot, precisions, status, truth
from kernel_agent.agent import prompts
from kernel_agent.config import OptimizeConfig
from kernel_agent.kernels import compare
from kernel_agent.report import _quality_lines, quality_bounds
from kernel_agent.suite import SuiteConfig
from kernel_agent.workloads import perceptual
from kernel_agent.workloads.base import WorkloadSpec
from kernel_agent.workloads.llm import LLMWorkload
from kernel_agent.workloads.voxcpm import VoxCPMWorkload
from kernel_agent.workspace import RunDir, read_json, write_json

RELAXED = "relaxed"
TOY = Path(__file__).with_name("perceptual_toy.py")
WHY = "M=1 decode GEMVs stream 2 x 6 MB of bf16 weights: memory bound at 91 % of SOL"


def test_the_mode_its_tiers_and_the_default_of_new_runs():
    assert compare.QUALITIES == config.QUALITIES and compare.DEFAULT_QUALITY == RELAXED
    assert config.DEFAULT_QUALITY == RELAXED and OptimizeConfig("org/m").quality == RELAXED
    assert compare.REDUCED_QUALITIES == ("near-lossless", RELAXED)
    assert compare.allows_reduced(RELAXED) and not compare.allows_reduced("exact")
    assert compare.tier_for(RELAXED, "fp8_weights") == compare.RELAXED_TIER == RELAXED
    for precision in ("fp8_w8a8", "fp8_mx", "reduced"):
        assert compare.tier_for(RELAXED, precision) == RELAXED
    assert compare.tier_for(RELAXED, "fp4_weights") == compare.RELAXED_FP4_TIER
    assert compare.tier_for(RELAXED, "fp8_kv") == compare.RELAXED_KV_TIER
    assert compare.tier_for(RELAXED, None) == compare.tier_for(RELAXED, "exact") == "exact"
    assert compare.tier_of({"tier": compare.RELAXED_FP4_TIER}) == compare.RELAXED_FP4_TIER
    # the status line's numbers are the tiers'
    for mode in ("near-lossless", RELAXED):
        cosine, rel_l2, norm, _ = compare.NEAR_LOSSLESS_BOUNDS[mode]
        note = config.QUALITY_NOTES[mode]
        assert f"cosine >= {cosine:g}" in note and f"L2 <= {rel_l2:g}" in note
        assert f"±{norm * 100:g} %" in note


def test_an_old_run_keeps_its_recorded_mode():
    """``run.json`` records the mode; one written before ``--quality`` existed was exact."""
    assert OptimizeConfig.from_dict({"model_ref": "m"}).quality == "exact"
    assert OptimizeConfig.from_dict({"model_ref": "m", "quality": "near-lossless"}).quality == (
        "near-lossless"
    )
    assert OptimizeConfig.from_dict(OptimizeConfig("m").to_dict()).quality == RELAXED
    assert perceptual.mode_of(None) == "exact" and perceptual.mode_of(RELAXED) == RELAXED


def test_cli_and_the_kernelbench_suite(tmp_path):
    parser = cli.argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    cli._add_run_args(sub.add_parser("optimize"))
    assert cli._config(parser.parse_args(["optimize", "org/m"])).quality == RELAXED
    for mode in compare.QUALITIES:
        assert cli._config(parser.parse_args(["optimize", "org/m", "--quality", mode])).quality == (
            mode
        )
    with pytest.raises(SystemExit):
        parser.parse_args(["optimize", "org/m", "--quality", "lossy"])
    # KernelBench's correctness is the benchmark's own: numerics within rounding noise
    assert SuiteConfig(runs_dir=tmp_path).optimize_config().quality == "exact"


def test_relaxed_allows_the_precisions_of_near_lossless():
    assert precisions.default(RELAXED) == precisions.default("near-lossless")
    assert "fp4_weights" not in precisions.default(RELAXED)  # 4-bit stays opt-in
    assert "fp8_kv" not in precisions.default(RELAXED)
    assert precisions.check(RELAXED, ["fp8_weights", "fp4_weights"]) is None
    assert "needs --quality near-lossless or relaxed" in precisions.check("exact", ["reduced"])
    every = tuple(compare.PRECISIONS)
    assert precisions.allowed(RELAXED, every) == every
    assert precisions.allowed("exact", every) == ("exact",)
    default = precisions.default(RELAXED)
    assert precisions.tier_allowed(RELAXED, default)
    assert not precisions.tier_allowed(compare.RELAXED_FP4_TIER, default)
    assert precisions.tier_allowed(compare.RELAXED_FP4_TIER, every)
    assert precisions.tier_allowed(compare.NEAR_LOSSLESS_TIER, default)


def test_the_planner_pivots_and_capture_in_a_relaxed_run(tmp_path, monkeypatch):
    target = {"id": "mlp", "precision": "fp8_weights", "precision_why": WHY}
    assert orchestrator._precision(dict(target), RELAXED) is None
    assert "not allowed" in orchestrator._precision({**target, "precision": "fp4_weights"}, RELAXED)
    spec = {"id": "mlp", "module_class": "Qwen3MLP", "precision": "fp8_weights"}
    proposal = {"precision": "fp8_w8a8", "precision_why": "M=352 rows: compute bound at 92 %"}
    assert pivot.check(spec, proposal, quality=RELAXED, taken=set()) is None
    assert "needs --quality" in pivot.check(spec, proposal, quality="exact", taken=set())

    # the dry run's capture records the relaxed tier, as the worker's does
    monkeypatch.setattr(orchestrator.toolchain, "setup", dryrun.SimToolchain)
    cfg = OptimizeConfig(model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path)  # the default mode
    orch = orchestrator.Orchestrator(dryrun.create_run(cfg, 0), cfg)
    assert orch.run.load()["config"]["quality"] == RELAXED
    targets = [
        {
            "id": "mlp8",
            "module_class": "Qwen3MLP",
            "phase": "decode",
            "why": "w",
            "approach": "a",
            "backends": ["cuda"],
            "precision": "fp8_weights",
            "precision_why": WHY,
        },
    ]
    world = dryrun.World(orch)
    with world.installed():
        assert orch._capture(targets) == ["mlp8"]
        # its simulated e2e records carry the perceptual gate, as a relaxed run's do
        assert world.gated and world._e2e([], [])["metrics"]["perceptual"]["passed"]
    capture = read_json(orch.run.target("mlp8") / "spec.json")["capture"]
    assert capture["tier"] == RELAXED and capture["precision"] == "fp8_weights"


def test_prompts_of_a_relaxed_run():
    policy = prompts.precision_policy(RELAXED)
    assert "# Precision (`--quality relaxed`)" in policy and prompts.RELAXED_POLICY in policy
    assert "fusions whose rounding differs" in policy and "4-bit" in policy
    assert prompts.RELAXED_POLICY not in prompts.precision_policy("near-lossless")
    cosine, rel_l2, norm, _ = compare.NEAR_LOSSLESS_BOUNDS[RELAXED]  # the policy's numbers
    gate = perceptual.RELAXED_GATE
    for number in (
        f"cosine >= {cosine:g}",
        f"relative L2 error\n<= {rel_l2:g}",
        f"±{norm * 100:g} %",
        f"error rate +{gate['max_error_increase']:.2f}",
        f"speaker similarity >= {gate['min_speaker_similarity']:.2f}",
        f"mean KL <= {gate['max_kl']:.2f}",
        f"top-1 >= {gate['min_top1']:.2f}",
    ):
        assert number in prompts.RELAXED_POLICY, number
    note = prompts.precision_note(RELAXED, precisions.default(RELAXED))
    assert "relaxed run" in note and "No 4-bit" in note
    assert "relaxed run" not in prompts.precision_note("near-lossless", ("exact", "reduced"))

    target = {"id": "mlp", "module_class": "Qwen3MLP", "precision": "fp8_weights"}
    capture = {"cases": [], "tier": RELAXED, "precision": "fp8_weights"}
    text = prompts.engineer_prompt(target, capture, ["triton"], "python", "toolchain", 10, None)
    cosine, rel_l2, norm, (a, r) = compare.NEAR_LOSSLESS_BOUNDS[RELAXED]
    assert "The evaluator checks it in the relaxed tolerance tier" in text
    assert f"cosine >= {cosine:g}, relative L2 error <= {rel_l2:g}" in text
    assert f"±{norm * 100:g} %" in text and f"{a:g} x RMS + {r:g} x |reference|" in text
    research = prompts.research_prompt(target, capture, "evidence", Path("/r/plan.md"), "tc")
    assert "(relaxed tolerance tier; low_precision.md)" in research
    near = prompts.engineer_prompt(
        target, {**capture, "tier": "near-lossless"}, ["triton"], "python", "toolchain", 10, None
    )
    assert "The evaluator checks it in the near-lossless tolerance tier" in near


def test_the_gate_and_floor_thresholds_of_a_relaxed_run():
    voxcpm = VoxCPMWorkload(WorkloadSpec(repo_id="openbmb/VoxCPM2", modality="tts"))
    floor = perceptual.floor_options(voxcpm, RELAXED)
    assert floor["min_mean_step_cosine"] == 0.90 and floor["min_step_cosine"] == 0.2
    assert floor["stop_tolerance"] == 2 and floor["max_error_increase"] == 0.10
    assert perceptual.floor_options(voxcpm, "near-lossless") == voxcpm.near_lossless_options
    user = VoxCPMWorkload(
        WorkloadSpec(repo_id="openbmb/VoxCPM2", modality="tts", options={"max_mos_drop": 0.2})
    )
    assert "max_mos_drop" not in perceptual.floor_options(user, RELAXED)  # -o wins
    llm = LLMWorkload(WorkloadSpec(repo_id="Qwen/Qwen3-0.6B", modality="llm"))
    floor = perceptual.floor_options(llm, RELAXED)
    assert floor["min_cosine"] == 0.96 and floor["min_prefix"] == 0 and floor["max_kl"] == 0.10
    # every relaxed gate threshold loosens its near-lossless one
    near = {
        "max_error_increase": perceptual.MAX_ERROR_INCREASE,
        "min_speaker_similarity": perceptual.MIN_SPEAKER_SIMILARITY,
        "min_speaker_similarity_worst": perceptual.MIN_SPEAKER_SIMILARITY_WORST,
        "max_mos_drop": perceptual.MAX_MOS_DROP,
        "max_kl": perceptual.LLM_MAX_KL,
        "max_kl_worst": perceptual.LLM_MAX_KL_WORST,
        "min_top1": perceptual.LLM_MIN_TOP1,
        "max_nll_increase": perceptual.LLM_MAX_NLL_INCREASE,
    }
    assert set(perceptual.RELAXED_GATE) == set(near)
    for key, value in perceptual.RELAXED_GATE.items():
        assert (value < near[key]) if key.startswith("min_") else (value > near[key]), key


def _llm(kl: float, top1: float, nll: float) -> list[dict]:
    return [{"kl": kl, "top1": top1, "nll": nll, "tokens": [1]} for _ in range(4)]


@pytest.mark.parametrize(
    ("variant", "kl", "top1", "nll", "near_lossless", "relaxed"),
    [  # the README's Qwen3-0.6B calibration (mean KL, top-1 agreement, NLL change)
        ("fp8 per output channel", 0.0111, 0.951, -0.068, True, True),
        ("fp8 per channel, scales x1.05", 0.0307, 0.930, 0.106, True, True),
        ("a change twice FP8 x1.05's", 0.07, 0.86, 0.2, False, True),
        ("int4 weights, group 128", 0.350, 0.725, 0.104, False, False),
        ("fp8 per channel, scales x1.2", 0.337, 0.756, 0.248, False, False),
        ("one KV head dropped", 0.507, 0.698, 0.284, False, False),
        ("decode loop one token late", 0.0, 1.0, 0.384, False, False),
    ],
)
def test_the_relaxed_llm_gate_still_fails_the_broken_variants(
    variant, kl, top1, nll, near_lossless, relaxed
):
    eager = _llm(0.0, 1.0, 1.0)
    candidate = _llm(kl, top1, 1.0 + nll)
    assert perceptual.compare_llm(eager, candidate).passed == near_lossless, variant
    loose = {k: v for k, v in perceptual.RELAXED_GATE.items() if k in ("max_kl", "max_kl_worst")}
    gate = perceptual.compare_llm(
        eager,
        candidate,
        **loose,
        min_top1=perceptual.RELAXED_GATE["min_top1"],
        max_nll_increase=perceptual.RELAXED_GATE["max_nll_increase"],
    )
    assert gate.passed == relaxed, (variant, gate.reason)


def _tts(errors: list[float], similarity: float, mos: float) -> list[dict]:
    angle = float(torch.arccos(torch.tensor(similarity)))
    emb = [float(torch.cos(torch.tensor(angle))), float(torch.sin(torch.tensor(angle))), 0.0]
    return [{"error_rate": e, "embedding": emb, "mos": mos} for e in errors]


@pytest.mark.parametrize(
    ("variant", "error", "similarity", "mos", "near_lossless", "relaxed"),
    [  # the README's VoxCPM2 calibration (error-rate increase, speaker similarity, MOS change)
        ("FP8 weight-only", 0.0, 0.988, 0.04, True, True),
        ("MXFP4 weights", 0.0, 0.973, -0.27, True, True),
        ("a small measured drop", 0.08, 0.91, -0.4, False, True),
        ("RMSNorm eps 1e-2", 0.742, 0.912, -0.42, False, False),
        ("one KV head dropped", 1.0, 0.724, -2.03, False, False),
        ("int4 per tensor", 1.0, 0.620, -2.02, False, False),
    ],
)
def test_the_relaxed_tts_gate_still_fails_the_broken_variants(
    variant, error, similarity, mos, near_lossless, relaxed
):
    eager = _tts([0.0] * 4, 1.0, 3.26)
    candidate = _tts([error] * 4, similarity, 3.26 + mos)
    assert perceptual.compare_tts(eager, candidate).passed == near_lossless, variant
    thresholds = {
        k: v
        for k, v in perceptual.RELAXED_GATE.items()
        if k in ("max_error_increase", "min_speaker_similarity", "max_mos_drop")
    }
    worst = perceptual.RELAXED_GATE["min_speaker_similarity_worst"]
    gate = perceptual.compare_tts(
        eager, candidate, **thresholds, min_speaker_similarity_worst=worst
    )
    assert gate.passed == relaxed, (variant, gate.reason)


# ---------------------------------------------------------------- analyze + e2e, relaxed


def _sealed(tmp_path: Path, quality: str) -> RunDir:
    spec = WorkloadSpec(repo_id="toy/perceptual", modality="tts", device="cpu", harness=str(TOY))
    run = RunDir.create(tmp_path, "toy/perceptual")
    write_json(
        run.run_json,
        {
            "card": {"repo_id": "toy/perceptual"},
            "workload": spec.to_dict(),
            "config": {"quality": quality},
            "truth": truth.new_section(),
        },
    )
    return run


def _call(capsys, run: RunDir, *argv):
    from kernel_agent import worker

    worker.main([str(a) for a in (argv[0], "--run-dir", run.root, *argv[1:])])
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith(worker.MARKER)]
    return json.loads(line[-1][len(worker.MARKER) :])


def _quieter(tmp_path: Path, factor: float) -> Path:
    """A transform whose perceptual samples (40 steps) come out ``factor`` times as loud."""
    path = tmp_path / f"quieter_{factor}.py"
    path.write_text(
        "def apply(workload):\n"
        "    run = workload.run\n\n"
        "    def patched(inputs):\n"
        "        out = run(inputs)\n"
        "        if int(workload.options['steps']) == 40:\n"
        f"            out['states'] = out['states'] * {factor}\n"
        "        return out\n\n"
        "    workload.run = patched\n"
    )
    return path


def test_relaxed_e2e_flow_passes_a_small_measured_drop(tmp_path, monkeypatch, capsys):
    from kernel_agent import toolchain

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(toolchain, "setup", lambda *a, **k: None)
    run = _sealed(tmp_path, RELAXED)
    keeper = truth.of(run)
    baseline = _call(capsys, run, "analyze", "--no-profile", "--iters", 1)
    info = baseline["perceptual"]
    assert info["status"] == "ok" and info["quality"] == RELAXED, info
    keeper.seal_baseline(baseline["median_ms"])
    assert keeper.worker_args()[-2:] == ["--quality", RELAXED]
    assert any("quality mode: relaxed" in line for line in perceptual.summary_lines(baseline))

    # samples 15 % quieter: beyond near-lossless's gate (±10 %), within relaxed's (±20 %)
    quieter = _quieter(tmp_path, 0.85)
    args = ("e2e", "--transform", quieter, "--iters", 1, *keeper.worker_args()[:-2])
    result = _call(capsys, run, *args, "--quality", RELAXED)
    assert result["status"] == "ok" and result["passed"], result
    gate = result["metrics"]["perceptual"]
    assert gate["quality"] == RELAXED and gate["rms_ratio"] == pytest.approx(0.85, abs=0.01)
    near = _call(capsys, run, *args, "--quality", "near-lossless")
    assert not near["passed"] and near["reason"] == "perceptual: sample energy x0.850", near

    # broken: half as loud fails the relaxed gate too
    result = _call(capsys, run, *args[:2], _quieter(tmp_path, 0.5), *args[3:], "--quality", RELAXED)
    assert not result["passed"] and result["reason"] == "perceptual: sample energy x0.500"


# ---------------------------------------------------------------- report + status


def test_report_and_status_show_the_mode_and_its_bounds(tmp_path):
    bounds = quality_bounds(RELAXED)
    assert "cosine >= 0.99, relative L2 <= 0.16, norm ±4 %" in bounds
    assert "mean KL <= 0.1" in bounds and "error rate +0.1" in bounds
    assert "cosine >= 0.996" in quality_bounds("near-lossless")
    base = {"perceptual": {"status": "ok", "samples": 8, "mean": {"error_rate": 0.01}}}
    lines = _quality_lines({"config": {"quality": RELAXED}}, base)
    assert lines[0].startswith("* quality: **relaxed**: perceptual gate on 8 samples")
    assert lines[1] == f"* relaxed bounds: {bounds}" and "precisions allowed" in lines[2]
    assert _quality_lines({"config": {}}, {}) == [
        "* quality: **exact**: numerics within rounding noise of eager"
    ]

    run = RunDir.create(tmp_path, "org/m")
    write_json(run.run_json, {"card": {"repo_id": "org/m"}, "config": {"quality": RELAXED}})
    text = status.render(run, width=200)
    assert re.search(r"^quality: relaxed \(about twice near-lossless", text, re.M), text
