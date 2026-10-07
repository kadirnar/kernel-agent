"""The evaluator's scaled and sign-flipped checks and its peak-memory delta (issue #148,
docs/RESEARCH-TRITON.md §3.3): the captured floating-point inputs x 3, x 0.01 and x -1
catch value-conditional shortcuts and constants calibrated on the captured values (an FP8
activation scale: test_perturbed_calibration.py), with tolerances that follow the scale;
the per-call peak memory of the candidate vs the reference is reported with a warning."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from kernel_agent import report
from kernel_agent.agent import tools
from kernel_agent.kernels import compare, evaluate, verify
from kernel_agent.profiling.capture import capture_calls


class ReluScale(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(0.5, 1.5, 64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) * self.weight


class Amplify(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 20000.0


# Correct on the captured inputs, all positive (2..3), and on inputs redrawn from their mean
# and std, which stay positive: the ReLU "cannot" matter (KernelBench-Verified's 374x).
POSITIVE_ONLY = """import torch


class M(torch.nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.weight = reference.weight

    def forward(self, x):
        return x * self.weight


def build(reference):
    return M(reference)
"""
HONEST = POSITIVE_ONLY.replace("return x * self.weight", "return x.clamp_min(0) * self.weight")
AMPLIFY = """import torch


class M(torch.nn.Module):
    def forward(self, x):
        return x * 20000.0


def build(reference):
    return M()
"""


def _capture(tmp_path: Path, module: nn.Module, x: torch.Tensor) -> Path:
    path = tmp_path / "capture.pt"
    capture_calls(module.eval(), [((x,), {}, 10)], path)
    return path


def _candidate(tmp_path: Path, source: str, name: str = "cand.py") -> Path:
    path = tmp_path / name
    path.write_text(source)
    return path


def _positive_input() -> torch.Tensor:
    return torch.rand(16, 64, generator=torch.Generator().manual_seed(0)) + 2.0


def test_scale_multiplies_floats_in_place_and_keeps_ids_masks_and_non_finite():
    x = torch.randn(4, 8)
    ids = torch.arange(4)
    flags = torch.ones(4, dtype=torch.bool)
    mask = torch.zeros(4, 4)
    mask[0, 1:] = -1e9  # an additive mask: never redrawn, never scaled
    inf = torch.tensor([1.0, float("inf")])
    before = [t.clone() for t in (x, ids, flags, mask, inf)]
    n = verify.scale_(((x, ids), {"flags": flags, "mask": mask, "inf": inf}), -1.0)
    assert n == 1
    assert torch.equal(x, -before[0]) and x.data_ptr() == x.data_ptr()
    for t, old in zip((ids, flags, mask, inf), before[1:], strict=True):
        assert torch.equal(t, old)
    assert dict(verify.SCALED) == {"scaled_x3": 3.0, "scaled_x0.01": 0.01, "sign_flipped": -1.0}
    assert all(name in verify.CHECKS for name, _ in verify.SCALED)


def test_input_scale_sets_the_signal_threshold_and_the_absolute_tolerance():
    gen = torch.Generator().manual_seed(0)
    small = (torch.randn(4096, generator=gen) * 0.004).to(torch.bfloat16)  # RMS < atol
    zeros = torch.zeros_like(small)
    # x 1: no signal, every element within the absolute tolerance: zeros pass ...
    assert compare.compare_tensors("y", small, zeros)["ok"]
    # ... x 0.01: an output 100x smaller has signal from 0.01 x atol on
    shrunk = compare.compare_tensors("y", small, zeros, input_scale=0.01)
    assert not shrunk["ok"] and "relative L2" in shrunk["error"]
    assert compare.compare_tensors("y", small, small.clone(), input_scale=0.01)["ok"]

    big = torch.randn(4096, generator=gen).to(torch.bfloat16)
    noisy = big.clone()
    near_zero = big.float().abs().argsort()[:60]
    noisy[near_zero] += 0.03  # 1.5x the bf16 atol on 1.5 % of the elements
    assert "outside tolerance" in compare.compare_tensors("y", big, noisy)["error"]
    grown = compare.compare_tensors("y", big, noisy, input_scale=3.0)  # x 3: atol x 3
    assert grown["ok"] and grown["atol"] == pytest.approx(0.06)
    # the near-lossless tiers' bounds are relative: the same at any scale with signal
    tier = compare.NEAR_LOSSLESS_TIER
    for scale in (3.0, 0.01, -1.0):
        assert compare.compare_tensors("y", big, big, tier=tier, perturbed=True, input_scale=scale)[
            "ok"
        ]


def test_sign_flip_catches_a_shortcut_for_positive_inputs(tmp_path):
    capture = _capture(tmp_path, ReluScale(), _positive_input())
    result = evaluate.evaluate(capture, _candidate(tmp_path, POSITIVE_ONLY), device="cpu")
    assert result["status"] == "incorrect_perturbed", result
    assert result["failed_check"]["check"] == "sign_flipped"
    assert "x -1" in result["error"] and "scales such as FP8" in result["error"]

    honest = evaluate.evaluate(capture, _candidate(tmp_path, HONEST, "honest.py"), device="cpu")
    assert honest["status"] == "ok", honest
    assert "scaled" in honest["checks"]
    assert honest["redraws"] == {
        "ran": {"scaled_x3": 1, "scaled_x0.01": 1, "sign_flipped": 1},
        "skipped": [],
    }


def test_a_reference_that_overflows_skips_the_scaled_check_and_records_it(tmp_path):
    x = (torch.rand(8, 64, generator=torch.Generator().manual_seed(0)) + 0.5).half()
    capture = _capture(tmp_path, Amplify(), x)  # x 20000 <= 30000: finite in fp16
    result = evaluate.evaluate(capture, _candidate(tmp_path, AMPLIFY), device="cpu")
    assert result["status"] == "ok", result
    redraws = result["redraws"]
    assert redraws["skipped"] == [
        {"case": 0, "check": "scaled_x3", "why": "the reference's output is not finite"}
    ]
    assert redraws["ran"] == {"scaled_x0.01": 1, "sign_flipped": 1}


def test_reverify_case_reports_every_failed_check_and_what_ran():
    module = ReluScale().eval()
    x = _positive_input()
    with torch.inference_mode():
        out = module(x)
    case = {"output": out, "post_args": (x.clone(),), "post_kwargs": {}}
    namespace: dict = {}
    exec(POSITIVE_ONLY, namespace)
    candidate = namespace["build"](module)
    record: dict = {}
    failed = verify.reverify_case(
        module, candidate, case, ((x,), {}), torch.Generator().manual_seed(0), lambda: None, record
    )
    assert [f["check"] for f in failed] == ["sign_flipped"]
    assert record["ran"] == {"scaled_x3": 1, "scaled_x0.01": 1, "sign_flipped": 1}


def test_changing_the_scaled_factors_is_an_integrity_violation(tmp_path):
    capture = _capture(tmp_path, ReluScale(), _positive_input())
    source = POSITIVE_ONLY.replace(
        "def build(reference):\n",
        "def build(reference):\n"
        "    from kernel_agent.kernels import verify\n"
        "    verify.SCALED = ()\n",
    )
    saved = verify.SCALED
    try:
        result = evaluate.evaluate(capture, _candidate(tmp_path, source), device="cpu")
    finally:
        verify.SCALED = saved
    assert result["status"] == "integrity_violation", result
    assert "SCALED" in result["error"]


# ------------------------------------------------------------------ peak memory


def _cases(*peaks: tuple[float, float]) -> list[dict]:
    return [
        {"signature": f"x[{i}]", "ref_peak_mib": ref, "new_peak_mib": new}
        for i, (ref, new) in enumerate(peaks)
    ]


def test_peak_memory_summary_takes_the_largest_increase_and_warns_above_both_thresholds():
    summary = evaluate.peak_memory_summary(_cases((100.0, 110.0), (10.0, 40.0)))
    assert summary is not None
    assert summary["case"] == 1 and summary["delta_mib"] == 30.0 and summary["ratio"] == 4.0
    assert "x[1]" in summary["warning"] and "30.0 MiB more" in summary["warning"]

    assert "warning" not in evaluate.peak_memory_summary(_cases((10.0, 20.0)))  # < 16 MiB
    assert "warning" not in evaluate.peak_memory_summary(_cases((1000.0, 1100.0)))  # < +25 %
    less = evaluate.peak_memory_summary(_cases((50.0, 20.0)))
    assert less["delta_mib"] == -30.0 and "warning" not in less
    assert evaluate.peak_memory_summary([{"signature": "x", "ref_ms": 1.0}]) is None
    assert evaluate.PEAK_MEMORY_WARN_SHARE == 0.25 and evaluate.PEAK_MEMORY_WARN_MIB == 16.0


def test_compact_results_keep_the_peak_memory():
    summary = evaluate.peak_memory_summary(_cases((10.0, 40.0)))
    result = {
        "status": "ok",
        "correct": True,
        "peak_memory": summary,
        "cases": [{"signature": "x[0]", "ok": True, "peak_delta_mib": 30.0, "new_peak_mib": 40}],
    }
    kept = tools.compact(result)
    assert kept["peak_memory"] == summary
    assert kept["cases"][0]["peak_delta_mib"] == 30.0


def test_report_lists_the_peak_memory_warnings_of_the_best_kernels(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from synthetic_run import make_run

    run = make_run(tmp_path)
    first = run.target_ids()[0]
    summary = evaluate.peak_memory_summary(_cases((10.0, 40.0)))
    rec = {"snapshot": "001_triton_v1.py", "speedup": 1.5, "peak_memory": summary}
    monkeypatch.setattr(report, "best_for_target", lambda run, t, keeper=None: rec)
    text = report.write_report(run).read_text()
    assert "Peak GPU memory above the reference's" in text
    assert f"`{first}`: {summary['warning']}" in text
