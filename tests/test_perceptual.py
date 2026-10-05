"""``--quality near-lossless`` (issue #71): the perceptual gate, the sanity floor and the
near-lossless module tolerance tier.

CPU: text metrics, the paired TTS comparison, the tier, and the whole analyze + e2e flow on
a toy workload with a cheap scorer (``perceptual_toy.py``), sealed in ``.truth/``. GPU
(last test): the VoxCPM2 gate with the real scorers (Whisper-large-v3, WavLM-SV, UTMOS22)
on eager, an FP8 weight-only fake quantisation (passes) and a broken RMSNorm (fails).
"""

import json
import os
from pathlib import Path

import pytest
import torch
from torch import nn

from kernel_agent import cli, toolchain, truth, worker
from kernel_agent.kernels import compare
from kernel_agent.kernels.evaluate import evaluate
from kernel_agent.profiling.capture import load_capture
from kernel_agent.workloads import perceptual
from kernel_agent.workloads.base import WorkloadSpec
from kernel_agent.workloads.harness import load_harness
from kernel_agent.workloads.quality import summary_section
from kernel_agent.workspace import RunDir, read_json, write_json

TOY = Path(__file__).with_name("perceptual_toy.py")


# ---------------------------------------------------------------- text + TTS comparison


def test_error_rate_by_words_and_characters():
    text = "Please remember: the charger, and the laptop!"
    assert perceptual.error_rate(text, "please remember the charger and the laptop") == 0.0
    assert perceptual.error_rate(text, "please remember the charger") == pytest.approx(3 / 7)
    assert perceptual.error_rate("a b", "") == 1.0 and perceptual.error_rate("", "x") == 1.0
    # character error rate for languages without spaces, punctuation ignored
    assert perceptual.error_rate("今天天气很好。", "今天天气很好", language="zh") == 0.0
    assert perceptual.error_rate("今天天气很好", "今天天氣很好", language="zh") == pytest.approx(
        1 / 6
    )
    assert perceptual.tokens("Der Zug fährt ab.") == ["der", "zug", "fährt", "ab"]
    assert perceptual.edit_distance(list("kitten"), list("sitting")) == 3


def _scores(errors, mos=None, turn=0.0):
    """TTS scores with unit speaker embeddings rotated by ``turn`` radians per sample."""
    out = []
    for i, err in enumerate(errors):
        angle = turn if isinstance(turn, float) else turn[i]
        emb = [float(torch.cos(torch.tensor(angle))), float(torch.sin(torch.tensor(angle))), 0.0]
        out.append({"error_rate": err, "embedding": emb, "mos": None if mos is None else mos[i]})
    return out


def test_compare_tts_paired_thresholds():
    ref = _scores([0.0, 0.1, 0.0, 0.05], mos=[3.0, 3.2, 3.1, 2.9])
    same = perceptual.compare_tts(ref, _scores([0.05, 0.05, 0.0, 0.1], mos=[3.1, 3.0, 3.0, 3.0]))
    assert same.passed, same.reason
    assert same.metrics["error_increase"] == pytest.approx(0.0125)
    assert same.metrics["speaker_similarity"] == 1.0 and same.metrics["mos"] == 3.025

    worse = perceptual.compare_tts(ref, _scores([0.3, 0.1, 0.0, 0.05], mos=[3.0, 3.2, 3.1, 2.9]))
    assert not worse.passed and "error rate" in worse.reason
    other_voice = perceptual.compare_tts(ref, _scores([0.0, 0.1, 0.0, 0.05], turn=[0, 0, 1.2, 0]))
    assert not other_voice.passed and "sample 2" in other_voice.reason
    assert other_voice.metrics["worst_sample"] == 2
    flat = perceptual.compare_tts(ref, _scores([0.0, 0.1, 0.0, 0.05], mos=[2.5, 2.6, 2.7, 2.5]))
    assert not flat.passed and "MOS" in flat.reason
    no_mos = perceptual.compare_tts(_scores([0.0]), _scores([0.0]))
    assert no_mos.passed and no_mos.metrics["mos"] == "unavailable"
    assert not perceptual.compare_tts(ref, ref[:2]).passed


def test_modes_and_floor_options():
    assert perceptual.mode_of(None) == "exact" and perceptual.mode_of("near-lossless")
    with pytest.raises(ValueError, match="unknown quality mode"):
        perceptual.mode_of("lossy")
    spec = WorkloadSpec(
        repo_id="toy/p", modality="tts", device="cpu", options={"min_step_cosine": 0.99}
    )
    wl = load_harness(TOY, spec)
    floor = perceptual.floor_options(wl)  # the user's -o min_step_cosine wins
    assert floor == {"min_mean_step_cosine": 0.95}
    with perceptual.judging(wl, "near-lossless", {"samples": []}) as gate:
        assert gate and wl.options["min_mean_step_cosine"] == 0.95
    assert wl.options["min_mean_step_cosine"] == 0.9999
    with perceptual.judging(wl, "near-lossless", None) as gate:  # no perceptual baseline
        assert not gate and wl.options["min_mean_step_cosine"] == 0.9999
    assert perceptual.skipped("exact", {}) is None
    why = perceptual.skipped("near-lossless", {"perceptual": {"status": "none", "reason": "x"}})
    assert why and why["passed"] and "exact checks apply" in why["skipped"]


def test_quality_flag_reaches_config_and_worker_args(tmp_path):
    ns = _parse(["analyze", "org/m", "--quality", "near-lossless"])
    assert cli._config(ns).quality == "near-lossless"
    assert cli._config(_parse(["analyze", "org/m"])).quality == "exact"
    run = RunDir.create(tmp_path, "org/m")
    write_json(run.run_json, {"truth": truth.new_section(), "config": {"quality": "near-lossless"}})
    keeper = truth.Truth(run)
    assert keeper.worker_args()[-2:] == ["--quality", "near-lossless"]
    data = run.load()
    data["config"]["quality"] = "exact"  # an agent edits run.json: this process keeps its mode
    write_json(run.run_json, data)
    assert keeper.worker_args()[-2:] == ["--quality", "near-lossless"]


def test_orchestrator_passes_the_quality_mode_to_capture():
    from kernel_agent import orchestrator
    from kernel_agent.config import OptimizeConfig

    orch = orchestrator.Orchestrator.__new__(orchestrator.Orchestrator)
    calls: list[tuple] = []
    orch.worker = lambda run, command, *args: calls.append((command, args)) or {}
    orch.run = None
    orch.cfg = OptimizeConfig(model_ref="org/m", quality="near-lossless")
    orch._worker("capture", "--target", "t")
    orch._worker("e2e", "--iters", "1")  # e2e / e2e_ab: Truth.worker_args() carries it
    assert calls == [
        ("capture", ("--target", "t", "--quality", "near-lossless")),
        ("e2e", ("--iters", "1")),
    ]


def _parse(argv):
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    cli._add_run_args(sub.add_parser("analyze"))
    return parser.parse_args(argv)


# ---------------------------------------------------------------- module tolerance tier


def _fp8(w: torch.Tensor) -> torch.Tensor:
    """e4m3 per output channel, back to the weight's dtype (FP8 weight storage)."""
    scale = w.float().abs().amax(dim=1, keepdim=True) / 448.0
    return ((w.float() / scale).to(torch.float8_e4m3fn).float() * scale).to(w.dtype)


def test_near_lossless_tier_accepts_fp8_weights_and_catches_bugs():
    gen = torch.Generator().manual_seed(0)
    w = torch.randn(512, 1024, generator=gen).to(torch.bfloat16) / 32
    x = torch.randn(4, 1024, generator=gen).to(torch.bfloat16)
    ref, fp8 = x @ w.T, x @ _fp8(w).T
    exact = compare.compare_tensors("y", ref, fp8)
    near = compare.compare_tensors("y", ref, fp8, tier="near-lossless")
    assert not exact["ok"] and near["ok"], (exact, near)
    assert near["tier"] == "near-lossless" and near["rel_l2"] < 0.06

    row = fp8.clone()
    row[:, 7] = 0  # one output channel dropped
    error = compare.compare_tensors("y", ref, row, tier="near-lossless")["error"]
    assert "near-lossless tolerance away" in error
    scaled = compare.compare_tensors("y", ref, fp8 * 1.05, tier="near-lossless")
    assert "norm x1.04" in scaled["error"] and scaled["rel_l2"] < 0.08  # systematic, not noise
    scale = w.float().abs().amax() / 7  # int4, one scale per tensor
    int4 = x @ (torch.round(w.float() / scale).clamp(-8, 7) * scale).to(torch.bfloat16).T
    assert not compare.compare_tensors("y", ref, int4, tier="near-lossless")["ok"]
    zeros = torch.zeros(64)  # no signal: the exact checks
    assert not compare.compare_tensors("z", zeros, zeros + 0.01, tier="near-lossless")["ok"]

    # side effects (a KV-cache write) and nested outputs take the tier too
    cache, ref_cache, new_cache = torch.zeros(8, 512), torch.zeros(8, 512), torch.zeros(8, 512)
    ref_cache[3], new_cache[3] = ref[0].float(), fp8[0].float()
    args = ((cache,), (ref_cache,), (new_cache,))
    assert not all(c["ok"] for c in compare.compare_side_effects(*args))
    assert all(c["ok"] for c in compare.compare_side_effects(*args, tier="near-lossless"))
    assert all(c["ok"] for c in compare.compare_structures([ref], [fp8], tier="near-lossless"))

    assert compare.tier_for("near-lossless", "reduced") == "near-lossless"
    assert compare.tier_for("near-lossless", None) == compare.tier_for("exact", "reduced")
    assert compare.tier_of({"tier": "near-lossless"}) == "near-lossless"
    assert compare.tier_of({"tier": "anything"}) == compare.tier_of(None) == "exact"


FP8_LINEAR = """import torch
from torch import nn


class Fp8(nn.Linear):
    pass


def build(reference):
    new = Fp8(reference.in_features, reference.out_features)
    new.load_state_dict(reference.state_dict())
    with torch.no_grad():
        w = new.weight
        scale = w.abs().amax(dim=1, keepdim=True) / 448.0
        w.copy_((w / scale).to(torch.float8_e4m3fn).float() * scale)
    return new
"""


def _cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(toolchain, "setup", lambda *a, **k: None)


def _sealed(tmp_path, quality: str, harness: Path = TOY) -> RunDir:
    spec = WorkloadSpec(
        repo_id="toy/perceptual", modality="tts", device="cpu", harness=str(harness)
    )
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
    worker.main([str(a) for a in (argv[0], "--run-dir", run.root, *argv[1:])])
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith(worker.MARKER)]
    return json.loads(line[-1][len(worker.MARKER) :])


@pytest.mark.parametrize(
    ("precision", "quality"),
    [
        (None, "near-lossless"),
        ("reduced", "near-lossless"),
        ("fp8_weights", "near-lossless"),
        ("fp8_weights", "exact"),  # a spec edited after the plan: still the exact tier
    ],
)
def test_capture_records_the_tier_of_reduced_precision_targets(
    precision, quality, tmp_path, monkeypatch, capsys
):
    _cpu(monkeypatch)
    run = _sealed(tmp_path, quality)
    spec = {"module_class": "Linear", "qualname": "model.backbone", "precision": precision}
    write_json(run.target("lin") / "spec.json", spec)
    info = _call(capsys, run, "capture", "--target", "lin", "--quality", quality)
    capture = run.capture_file("lin")
    keeper = truth.of(run)
    keeper.seal(capture)
    assert compare.tier_of(load_capture(capture)) == (info.get("tier") or "exact")
    assert load_capture(capture).get("precision") == info.get("precision")
    cand = tmp_path / "fp8.py"
    cand.write_text(FP8_LINEAR)
    result = evaluate(capture, cand, device="cpu", capture_sha256=keeper.expect(capture))
    assert compare.TIER == "exact"  # reset after the evaluation
    if precision in compare.REDUCED_PRECISIONS and quality == "near-lossless":
        assert info["tier"] == "near-lossless" and info["precision"] == precision
        assert result["correct"] and result["tolerance_tier"] == "near-lossless", result
    else:
        assert "tier" not in info and "precision" not in info
        assert not result["correct"], result


# ---------------------------------------------------------------- analyze + e2e, sealed


def _transform(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / f"{name}.py"
    path.write_text("import torch\n\n\ndef apply(workload):\n" + body)
    return path


def test_near_lossless_e2e_flow(tmp_path, monkeypatch, capsys):
    _cpu(monkeypatch)
    run = _sealed(tmp_path, "near-lossless")
    keeper = truth.of(run)
    baseline = _call(capsys, run, "analyze", "--no-profile", "--iters", 1)
    info = baseline["perceptual"]
    assert info["status"] == "ok" and info["samples"] == 2, info
    assert info["mean"]["rms"] > 0 and [s["sample"] for s in info["per_sample"]] == ["10", "11"]
    assert run.baseline_output_perceptual().is_file()
    keeper.seal_baseline(baseline["median_ms"])  # what Orchestrator.analyze does
    assert any("perceptual" in a for a in keeper.worker_args())
    assert "near-lossless" in summary_section(read_json(run.baseline_json))
    assert any("perceptual baseline: 2 samples" in m for m in perceptual.messages(baseline))

    # weights x 1.1: outside the exact teacher-forcing thresholds, inside the sanity floor
    moderate = _transform(
        tmp_path,
        "moderate",
        "    with torch.no_grad():\n        workload.model.backbone.weight.mul_(1.1)\n",
    )
    exact = _call(
        capsys,
        run,
        "e2e",
        "--transform",
        moderate,
        "--iters",
        1,
        *keeper.worker_args()[:-2],
        "--quality",
        "exact",
    )
    assert exact["status"] == "ok" and not exact["passed"], exact
    assert exact["reason"].startswith("teacher-forced") and "perceptual" not in exact["metrics"]

    result = _call(capsys, run, "e2e", "--transform", moderate, "--iters", 1, *keeper.worker_args())
    assert result["status"] == "ok" and result["passed"], result
    gate = result["metrics"]["perceptual"]
    assert gate["passed"] and gate["rms_ratio"] == pytest.approx(1.0, abs=0.1)
    assert len(gate["per_sample"]) == 2 and gate["generate_s"] >= 0
    assert perceptual.summary_text(gate).startswith("perceptual gate passed")

    # correct on the timed and held-out inputs, broken on other ones: only the gate sees it
    sneaky = _transform(
        tmp_path,
        "sneaky",
        "    run = workload.run\n\n"
        "    def patched(inputs):\n"
        "        out = run(inputs)\n"
        "        if int(workload.options['steps']) == 40:\n"
        "            out['states'] = out['states'] * 0.5\n"
        "        return out\n\n"
        "    workload.run = patched\n",
    )
    result = _call(capsys, run, "e2e", "--transform", sneaky, "--iters", 1, *keeper.worker_args())
    assert result["status"] == "ok" and not result["passed"], result
    assert result["reason"] == "perceptual: sample energy x0.500"
    assert result["metrics"]["teacher_forced"]["passed"]

    # broken beyond the sanity floor: rejected without running the gate
    broken = _transform(
        tmp_path,
        "broken",
        "    with torch.no_grad():\n        workload.model.backbone.weight.neg_()\n",
    )
    result = _call(capsys, run, "e2e", "--transform", broken, "--iters", 1, *keeper.worker_args())
    assert not result["passed"] and result["reason"].startswith("teacher-forced")
    assert result["metrics"]["perceptual"]["skipped"] == "the other checks failed"

    # the perceptual baseline is a truth file
    path = run.baseline_output_perceptual()
    os.chmod(path, 0o644)
    path.write_bytes(path.read_bytes() + b"\0")
    result = _call(capsys, run, "e2e", "--transform", moderate, "--iters", 1, *keeper.worker_args())
    assert result["status"] == "tampered" and not result["passed"]


def test_near_lossless_without_a_perceptual_baseline_keeps_the_exact_checks(
    tmp_path, monkeypatch, capsys
):
    _cpu(monkeypatch)
    run = _sealed(tmp_path, "exact")
    baseline = _call(capsys, run, "analyze", "--no-profile", "--iters", 1)
    assert baseline["perceptual"] == {"status": "none", "reason": "--quality exact"}
    assert not run.baseline_output_perceptual().exists()
    moderate = _transform(
        tmp_path,
        "moderate",
        "    with torch.no_grad():\n        workload.model.backbone.weight.mul_(1.1)\n",
    )
    result = _call(
        capsys, run, "e2e", "--transform", moderate, "--iters", 1, "--quality", "near-lossless"
    )
    assert not result["passed"] and result["reason"].startswith("teacher-forced"), result
    assert "exact checks apply" in result["metrics"]["perceptual"]["skipped"]


def test_report_shows_the_perceptual_metrics():
    from kernel_agent.report import _quality_lines

    gate = {"passed": False, "reason": "speaker similarity 0.7 < 0.9", "error_rate": 0.02}
    assert perceptual.summary_text(gate).startswith("perceptual gate FAILED: speaker")
    assert _quality_lines({"config": {}}, {}) == []
    base = {"perceptual": {"status": "ok", "samples": 8, "mean": {"error_rate": 0.01}}}
    line = _quality_lines({"config": {"quality": "near-lossless"}}, base)[0]
    assert "perceptual gate on 8 samples" in line and "error_rate=0.01" in line
    line = _quality_lines({"config": {"quality": "near-lossless"}}, {})[0]
    assert "no perceptual baseline" in line


# ---------------------------------------------------------------- GPU: VoxCPM2


def _voxcpm2_cached() -> bool:
    import importlib.util

    if (
        importlib.util.find_spec("voxcpm") is None
        or importlib.util.find_spec("transformers") is None
    ):
        return False
    from huggingface_hub import try_to_load_from_cache

    repos = ("openbmb/VoxCPM2", perceptual.ASR_MODEL, perceptual.SPEAKER_MODEL)
    return all(isinstance(try_to_load_from_cache(r, "config.json"), str) for r in repos)


def _fake_quant_fp8(model: nn.Module) -> int:
    """Every nn.Linear weight of base_lm, residual_lm and the LocDiT to e4m3 per output
    channel and back (simulated FP8 weight storage)."""
    linears = {
        id(m): m
        for root in (model.base_lm, model.residual_lm, model.feat_decoder)
        for m in root.modules()
        if isinstance(m, nn.Linear)
    }
    with torch.no_grad():
        for m in linears.values():
            m.weight.copy_(_fp8(m.weight))
    return len(linears)


@pytest.mark.gpu
@pytest.mark.skipif(not _voxcpm2_cached(), reason="needs voxcpm, VoxCPM2 and the scorers cached")
@pytest.mark.parametrize("metric", ["latency", "ttfa"])
def test_voxcpm2_perceptual_gate(metric):
    """Eager's samples, then the gate on a broken RMSNorm (eps 1e-2: it mumbles) and on FP8
    weight-only fake quantisation (passes; the exact teacher forcing rejects it). With
    ``metric=ttfa`` every sample is the whole output of the streaming path."""
    from kernel_agent.workloads import create_workload

    spec = WorkloadSpec(
        repo_id="openbmb/VoxCPM2", modality="tts", family="voxcpm", options={"metric": metric}
    )
    wl = create_workload(spec)
    wl.load()
    reference, info = perceptual.record_baseline(wl)
    assert reference is not None and info["samples"] == 8 and info["metric"] == metric, info
    assert info["mean"]["error_rate"] < 0.05, info  # eager says the sentences
    if metric == "ttfa":  # streamed: one chunk per patch, the last run's marks
        assert len(wl.chunk_marks) == reference["samples"][-1]["output"]["latents"].shape[0]

    norms = [m for m in wl.model.modules() if type(m).__name__ == "MiniCPMRMSNorm"]
    eps = {id(m): m.variance_epsilon for m in norms}
    for m in norms:
        m.variance_epsilon = 1e-2
    broken = perceptual.check(wl, reference)
    for m in norms:
        m.variance_epsilon = eps[id(m)]
    assert not broken["passed"] and "error rate" in broken["reason"], broken

    assert _fake_quant_fp8(wl.model) == 343
    fp8 = perceptual.check(wl, reference)
    assert fp8["passed"], fp8
    assert fp8["error_increase"] <= 0.02 and fp8["speaker_similarity"] > 0.95, fp8
