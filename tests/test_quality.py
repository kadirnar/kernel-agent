"""CPU tests: teacher-forced e2e decision logic, sanity check, sensitivity probe."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from kernel_agent import toolchain, worker
from kernel_agent.workloads.base import Comparison, WorkloadSpec, compare_steps
from kernel_agent.workloads.harness import load_harness
from kernel_agent.workloads.quality import (
    assess,
    free_running_sanity,
    is_chaotic,
    judge,
    perturb_linears,
    probe,
    probe_messages,
    summary_section,
)
from kernel_agent.workspace import write_json

TOY = Path(__file__).with_name("chaotic_toy.py")


@pytest.fixture
def toy():
    wl = load_harness(TOY, WorkloadSpec(repo_id="toy/chaotic", modality="tts", device="cpu"))
    wl.load()
    inputs = wl.make_inputs()
    return wl, inputs, wl.run(inputs)


def _ok(**metrics):
    return Comparison(True, metrics)


def _bad(reason, **metrics):
    return Comparison(False, metrics, reason)


# ---------------------------------------------------------------- pure logic


def test_judge_without_teacher_forcing_uses_free_running():
    assert judge(_ok()) == (True, "")
    assert judge(_bad("diverged")) == (False, "diverged")
    # chaotic does not relax anything when there is no teacher forcing
    assert judge(_bad("diverged"), chaotic=True) == (False, "diverged")


def test_judge_with_teacher_forcing():
    assert judge(_bad("diverged"), _ok(), _ok(), chaotic=True) == (True, "")
    passed, reason = judge(_bad("diverged"), _ok(), _ok(), chaotic=False)
    assert not passed and reason == "free-running: diverged"
    passed, reason = judge(_ok(), _bad("step 3"), _ok(), chaotic=True)
    assert not passed and reason == "teacher-forced: step 3"
    passed, reason = judge(_bad("x"), _ok(), _bad("audio has non-finite values"), chaotic=True)
    assert not passed and reason == "free-running sanity: audio has non-finite values"


def test_is_chaotic():
    plain = SimpleNamespace(chaotic=False)
    assert not is_chaotic(plain, {})
    assert not is_chaotic(plain, {"sensitivity": {"free_running_passed": True}})
    assert not is_chaotic(plain, {"sensitivity": {"free_running_passed": None}})
    assert is_chaotic(plain, {"sensitivity": {"free_running_passed": False}})
    assert is_chaotic(SimpleNamespace(chaotic=True), {})


def test_free_running_sanity():
    ref = {"audio": torch.ones(1000), "latents": torch.ones(10, 4), "tokens": torch.ones(3).long()}
    same = {"audio": -torch.ones(1000), "latents": torch.full((10, 4), 1.2), "tokens": None}
    cmp = free_running_sanity(ref, same)
    assert cmp.passed, cmp.reason
    assert cmp.metrics == {"audio_rms_ratio": 1.0, "latents_rms_ratio": 1.2}

    loud = {**same, "latents": torch.full((10, 4), 1.3)}
    cmp = free_running_sanity(ref, loud)
    assert not cmp.passed and "latents RMS energy x1.300" in cmp.reason
    assert free_running_sanity(ref, loud, tolerances={"latents": 0.5}).passed

    nan = {**same, "audio": torch.full((1000,), float("nan"))}
    assert "non-finite" in free_running_sanity(ref, nan).reason
    short = {**same, "audio": torch.ones(900)}
    assert "shape [900] != [1000]" in free_running_sanity(ref, short).reason
    assert "latents missing" in free_running_sanity(ref, {"audio": ref["audio"]}).reason
    silent = {**same, "audio": torch.zeros(1000)}
    assert "audio RMS energy x0.000" in free_running_sanity(ref, silent).reason


def test_compare_steps():
    ref = torch.randn(8, 1, 16)
    cmp = compare_steps(ref, ref.clone(), min_step_cosine=0.99, min_mean_step_cosine=0.999)
    assert cmp.passed and cmp.metrics["min_step_cosine"] == 1.0
    assert cmp.metrics["steps"] == 8 and cmp.metrics["mean_rel_error"] == 0.0

    one_bad = ref.clone()
    one_bad[5] = -one_bad[5]
    cmp = compare_steps(ref, one_bad, min_step_cosine=0.5, min_mean_step_cosine=0.5)
    assert not cmp.passed and cmp.metrics["worst_step"] == 5 and "step 5" in cmp.reason
    cmp = compare_steps(ref, one_bad, min_step_cosine=-2, min_mean_step_cosine=0.9)
    assert not cmp.passed and "mean step cosine" in cmp.reason

    # cosines cannot see a magnitude error; the RMS ratio does
    scaled = compare_steps(ref, ref * 1.5, **_th())
    assert scaled.metrics["mean_step_cosine"] > 0.9999 and "RMS x1.5000" in scaled.reason
    assert compare_steps(ref, ref * 1.5, **_th(), max_rms_change=0.6).passed

    assert "7 teacher-forced steps != 8" in compare_steps(ref, ref[:7], **_th()).reason
    assert "step shape" in compare_steps(ref, ref.view(8, 16, 1), **_th()).reason
    nan = ref.clone()
    nan[0, 0, 0] = float("nan")
    assert "non-finite" in compare_steps(ref, nan, **_th()).reason
    assert not compare_steps(ref[:0], ref[:0], **_th()).passed


def _th():
    return {"min_step_cosine": 0.9, "min_mean_step_cosine": 0.9}


def test_perturb_linears_is_benign_and_leaves_rng_alone():
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64))
    x = torch.randn(4, 64)
    with torch.no_grad():
        ref = model[0](x)
        state = torch.get_rng_state()
        with perturb_linears({"m": model, "again": model}) as hooked:
            out = model[0](x)
        assert hooked == 2  # shared modules are hooked once
        assert torch.equal(torch.get_rng_state(), state)
        rel = (out - ref).abs() / ref.abs().clamp_min(1e-12)
        assert torch.allclose(rel, torch.full_like(rel, 2.0**-8), rtol=1e-3)
        assert torch.equal(model[0](x), ref)  # hooks removed


def test_summary_and_messages():
    chaotic = {
        "sensitivity": {"free_running_passed": False, "reason": "spectral cosine 0.6"},
        "chaotic": True,
    }
    assert "no teacher forcing" in summary_section(chaotic)
    assert "WARNING" in probe_messages(chaotic)[0]
    with_tf = {**chaotic, "teacher_forcing": {"passed": True, "metrics": {"steps": 3}}}
    text = summary_section(with_tf)
    assert "CHAOTIC" in text and "teacher forcing" in text and "informational" in text
    assert not any("WARNING" in m for m in probe_messages(with_tf))
    broken_tf = {**with_tf, "teacher_forcing": {"passed": False, "error": "boom"}}
    assert any("self-check FAILED (boom)" in m for m in probe_messages(broken_tf))
    stable = {"sensitivity": {"free_running_passed": True, "metrics": {"x": 1}}}
    assert "Standard end-to-end validation applies" in summary_section(stable)
    assert probe_messages(stable) == []
    assert summary_section({}) == ""


# ---------------------------------------------------------------- toy chaotic workload


def test_probe_on_chaotic_toy(toy):
    wl, inputs, ref = toy
    result = probe(wl, inputs, ref)
    assert result["sensitivity"]["free_running_passed"] is False
    assert "2 modules" in result["sensitivity"]["perturbation"]
    assert result["teacher_forcing"]["passed"]
    assert result["teacher_forcing"]["metrics"]["min_step_cosine"] == 1.0  # exact replay
    assert result["chaotic"] is True


def test_assess_benign_change_passes_only_with_teacher_forcing(toy):
    wl, inputs, ref = toy
    with perturb_linears(wl.roots()):
        out = wl.run(inputs)
        verdict = assess(wl, inputs, ref, out, chaotic=True)
        strict = assess(wl, inputs, ref, out, chaotic=False)
    assert verdict["passed"], verdict
    free = verdict["metrics"]["free_running"]
    assert free["passed"] is False and free["gating"] is False and free["sanity_passed"]
    assert verdict["metrics"]["teacher_forced"]["passed"]
    assert not strict["passed"] and strict["reason"].startswith("free-running: ")


def test_assess_rejects_broken_change(toy):
    wl, inputs, ref = toy
    with torch.no_grad():
        wl.model.backbone.weight.mul_(1.3)
    verdict = assess(wl, inputs, ref, wl.run(inputs), chaotic=True)
    assert not verdict["passed"] and verdict["reason"].startswith("teacher-forced: ")


def test_teacher_forcing_needs_the_same_noise(toy):
    wl, inputs, ref = toy
    wl.options["seed"] = 1
    forced = wl.run_teacher_forced(inputs, ref)
    assert not wl.compare_teacher_forced(ref, forced).passed


def test_assess_without_teacher_forcing_keeps_flat_metrics(toy):
    wl, inputs, ref = toy
    wl.supports_teacher_forcing = False
    verdict = assess(wl, inputs, ref, ref, chaotic=True)
    assert verdict == {"passed": True, "reason": "", "metrics": {"tail_cosine": 1.0}}


# ---------------------------------------------------------------- worker, end to end


def _worker(capsys, *argv):
    worker.main([str(a) for a in argv])
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith(worker.MARKER)][-1]
    return json.loads(line[len(worker.MARKER) :])


def test_worker_analyze_and_e2e_use_teacher_forcing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(toolchain, "setup", lambda *a, **k: None)
    root = tmp_path / "run"
    spec = WorkloadSpec(repo_id="toy/chaotic", modality="tts", device="cpu", harness=str(TOY))
    write_json(root / "run.json", {"workload": spec.to_dict()})

    baseline = _worker(capsys, "analyze", "--run-dir", root, "--no-profile", "--iters", 1)
    assert baseline["deterministic"] and baseline["chaotic"]
    assert baseline["sensitivity"]["free_running_passed"] is False
    assert baseline["teacher_forcing"]["passed"]

    benign = tmp_path / "benign.py"
    benign.write_text(
        "import torch\n\n\ndef apply(workload):\n"
        "    with torch.no_grad():\n"
        "        workload.model.backbone.weight.mul_(1 + 2**-10)\n"
    )
    broken = tmp_path / "broken.py"
    broken.write_text(benign.read_text().replace("1 + 2**-10", "1.3"))

    result = _worker(capsys, "e2e", "--run-dir", root, "--transform", benign, "--iters", 1)
    assert result["status"] == "ok" and result["passed"], result
    assert result["metrics"]["teacher_forced"]["passed"]
    assert result["metrics"]["free_running"]["passed"] is False
    assert result["metrics"]["free_running"]["gating"] is False

    result = _worker(capsys, "e2e", "--run-dir", root, "--transform", broken, "--iters", 1)
    assert result["status"] == "ok" and not result["passed"]
    assert result["reason"].startswith("teacher-forced: ")
