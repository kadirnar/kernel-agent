"""Ceilings table (#90): the profiler's per-call work and the floors per precision."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from kernel_agent import toolchain
from kernel_agent.agent import prompts
from kernel_agent.kernels import roofline
from kernel_agent.profiling import ceilings
from kernel_agent.profiling.profiler import ModuleTimer

# The RTX 5070 Ti's peaks (measured 2026-10-06; FP8 / NVFP4 via torch scaled_mm).
PEAKS = {
    "version": 2,
    "dram_gbps": 767.1,
    "launch_floor_us": 15.78,
    "tflops": {"bfloat16": 99.4, "float32": 34.4, "float8_e4m3fn": 332.6},
    "tflops_unavailable": {"float4_e2m1fn_x2": "RuntimeError: no NVFP4 kernel"},
}


class Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(16, 64, bias=False)
        self.down = nn.Linear(64, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.relu(self.up(x)))


class Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(16)
        self.mlp = Mlp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


class Fused(nn.Module):
    """Holds a Linear but never calls it (a replaced kernel): estimated from its weights."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(16, 32, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.proj.weight)


class Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Layer(), Layer()])
        self.fused = Fused()
        self.conv = nn.Conv1d(4, 8, 3)
        self.deconv = nn.ConvTranspose1d(8, 4, 3)

    def forward(self, x: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        self.fused(x)
        self.deconv(self.conv(audio))
        return x


def _work(stats, cls):
    (stat,) = [s for s in stats if s.cls == cls]
    return {(w["group"], w["phase"]): w for w in stat.work}


def test_profiler_records_linear_and_conv_work_per_group_and_phase():
    torch.manual_seed(0)
    model = Toy()
    x, audio = torch.randn(2, 5, 16), torch.randn(2, 4, 9)
    with torch.inference_mode(), ModuleTimer({"m": model}, {}, cuda=False) as timer:
        model(x, audio)
        model(x[:, :1], audio)  # one position: the decode phase
    stats = timer.class_stats()

    mlp_elems = 16 * 64 + 64 * 16 + 16  # up (no bias) + down + its bias
    layer = _work(stats, "Layer")[("m.layers.*", "prefill")]
    assert layer["calls"] == 2 and layer["instances"] == 2
    assert layer["flops"] == {"float32": 2 * 2 * 10 * (16 * 64 + 64 * 16)}  # 2 calls x 10 rows
    assert layer["weight_elems"] == 2 * mlp_elems and layer["weight_bytes"] == 4 * 2 * mlp_elems
    assert layer["io_bytes"] == 2 * 2 * (2 * 5 * 16 * 4)  # input + output per call
    assert layer["signature"] == "a0[2, 5, 16]:float32" and "estimated_calls" not in layer
    decode = _work(stats, "Layer")[("m.layers.*", "decode")]
    assert decode["flops"] == {"float32": 2 * 2 * 2 * (16 * 64 + 64 * 16)}  # 2 rows per call

    linear = _work(stats, "Linear")
    up = linear[("m.layers.*.mlp.up", "prefill")]
    assert up["calls"] == 2 and up["weight_elems"] == 2 * 16 * 64
    assert linear[("m.layers.*.mlp.down", "prefill")]["weight_elems"] == 2 * (64 * 16 + 16)

    # convolutions: every output (conv) / input (transposed) element x C/groups x kernel
    conv = _work(stats, "Conv1d")[("m.conv", "prefill")]
    assert conv["flops"] == {"float32": 2 * 2 * (2 * 8 * 7) * (4 * 3)}  # two runs
    deconv = _work(stats, "ConvTranspose1d")[("m.deconv", "prefill")]
    assert deconv["flops"] == {"float32": 2 * 2 * (2 * 8 * 7) * (4 * 3)}
    assert deconv["weight_elems"] == 2 * (8 * 4 * 3 + 4)

    # a module that never calls its Linear: estimated from the weights at its input's rows
    fused = _work(stats, "Fused")[("m.fused", "prefill")]
    assert fused["flops"] == {"float32": 2 * 10 * 16 * 32} and fused["estimated_calls"] == 1
    toy = _work(stats, "Toy")[("m", "prefill")]  # the root sums everything inside it once
    assert toy["calls"] == 1 and "estimated_calls" in toy


def _profile() -> dict:
    """A VoxCPM2-like batch-16 profile: the LocDiT (inside the CFM solver) at 352 rows and
    the batched LM decode (Linear / MLP at 16 rows, 60 steps)."""
    dit_params, steps = 212_000_000, 540
    lm_attn, lm_mlp = 2048 * (2048 + 256 + 256) + 2048 * 2048, 3 * 2048 * 6144

    def work(group, calls, rows, params, ms, phase="prefill", io=0):
        return {
            "group": group,
            "phase": phase,
            "instances": 1,
            "calls": calls,
            "inclusive_ms": ms,
            "flops": {"bfloat16": 2 * rows * params * calls},
            "weight_elems": params * calls,
            "weight_bytes": 2 * params * calls,
            "io_bytes": io,
            "signature": f"a0[{rows}, 1024]:bfloat16",
        }

    dit = work("model.feat_decoder.estimator", steps, 352, dit_params, 7000.0, io=10**9)
    cfm = dict(work("model.feat_decoder", 60, 352, dit_params, 7100.0), calls=60)
    cfm["flops"], cfm["weight_elems"] = dit["flops"], dit["weight_elems"]  # the same work
    cfm["weight_bytes"] = dit["weight_bytes"]
    linears = [
        dict(work(f"model.base_lm.layers.*.self_attn.{name}", 28 * 60, 16, lm_attn // 4, 80.0))
        for name in ("q_proj", "k_proj", "v_proj", "o_proj")
    ]
    return {
        "hooked_wall_ms": 10_000.0,
        "classes": [
            {"cls": "UnifiedCFM", "is_leaf": False, "work": [cfm]},
            {"cls": "VoxCPMLocDiT", "is_leaf": False, "work": [dit]},
            {"cls": "Linear", "is_leaf": True, "work": linears},
            {
                "cls": "MiniCPMMLP",
                "is_leaf": False,
                "work": [work("model.base_lm.layers.*.mlp", 28 * 60, 16, lm_mlp, 600.0)],
            },
            {"cls": "RMSNorm", "is_leaf": True, "work": [work("model.norm", 20_000, 16, 0, 50.0)]},
        ],
    }


def test_ceilings_reproduce_the_hand_numbers():
    table = ceilings.build(_profile(), PEAKS, 5000.0, per="per batched run")
    rows = {r["target"]: r for r in table["rows"]}
    dit = rows["VoxCPMLocDiT@model.feat_decoder.estimator"]
    # 2 x 212 M x 352 rows x 540 calls = 80.6 TFLOP = 811 ms at 99.4 TFLOP/s: compute bound
    assert dit["flops_total"] == pytest.approx(80.6e12, rel=1e-3) and dit["m"] == 352
    assert dit["floors"]["exact"] == pytest.approx(811, rel=1e-3) and dit["bound"] == "compute"
    assert dit["floors"]["fp8_weights"] == dit["floors"]["exact"]  # bf16 math: no gain
    assert dit["floors"]["w8a8"] == pytest.approx(80.6e12 / 332.6e12 * 1e3, rel=1e-3)
    assert dit["floors"]["w4a4"] is None  # no NVFP4 peak: unknown, never a ratio
    assert dit["now_ms"] == pytest.approx(3500.0) and dit["share"] == pytest.approx(0.7)
    assert dit["saves_ms"]["exact"] == pytest.approx(3500 - 811, rel=1e-3)

    # The LM decode at 16 rows: the MLP streams its weights (memory bound); q/k/v/o share
    # one row, launch bound (four GEMV launches per layer cost more than their bytes).
    attn = rows["Linear@model.base_lm.layers.*.self_attn.{k_proj,o_proj,q_proj,v_proj}"]
    assert attn["instances"] == 4 and attn["calls"] == 4 * 28 * 60 and attn["bound"] == "launch"
    mlp = rows["MiniCPMMLP@model.base_lm.layers.*.mlp"]
    assert mlp["bound"] == "memory" and mlp["m"] == 16
    per_step = (attn["weight_bytes"] + mlp["weight_bytes"]) / 60
    assert per_step == pytest.approx(2 * 28 * (2048 * 2560 + 2048**2 + 3 * 2048 * 6144))
    assert mlp["floors"]["exact"] == pytest.approx(mlp["weight_bytes"] / 767.1e6, rel=1e-3)
    assert mlp["floors"]["fp8_weights"] == pytest.approx(mlp["floors"]["exact"] / 2, rel=1e-3)
    assert mlp["floors"]["fp4_weights"] == pytest.approx(mlp["floors"]["exact"] * 4.5 / 16, 1e-3)
    assert rows["RMSNorm@model.norm"]["bound"] == "launch"  # 20k tiny calls

    # End to end: the CFM holds the LocDiT, so the LocDiT's saving counts once.
    e2e = table["e2e"]
    assert table["e2e"]["w4a4"] is None
    saved = sum(r["saves_ms"]["exact"] for t, r in rows.items() if not t.startswith("VoxCPMLocDiT"))
    assert e2e["exact"]["floor_ms"] == pytest.approx(5000 - saved, rel=1e-3)
    assert e2e["exact"]["counted"][0] == "UnifiedCFM@model.feat_decoder"
    assert e2e["w8a8"]["floor_ms"] < e2e["fp8_weights"]["floor_ms"] < e2e["exact"]["floor_ms"]
    json.dumps(table)

    text = ceilings.markdown(table)
    assert "## Ceilings" in text and "| `VoxCPMLocDiT` `model.feat_decoder.estimator` |" in text
    assert "compute bound near M ≈ 130" in text and "W4A4 at NVFP4 unknown" in text
    assert "W4A4 unknown (RuntimeError: no NVFP4 kernel)" in text
    first = next(line for line in text.splitlines() if line.startswith("| `"))
    assert first.startswith("| `UnifiedCFM`")  # ranked by what reaching the floor saves


def test_ceilings_without_peaks_or_work(tmp_path):
    table = ceilings.build(_profile(), None, 5000.0)
    assert all(v is None for r in table["rows"] for v in r["floors"].values())
    assert table["e2e"] == {} and "peaks not measured" in ceilings.markdown(table)
    old = {"hooked_wall_ms": 1.0, "classes": [{"cls": "A", "is_leaf": True, "params": 3}]}
    assert ceilings.markdown(ceilings.build(old, PEAKS, 1.0)) == ""  # a profile made before #90
    table = ceilings.write(tmp_path, {"classes": [{"work": [{}]}]}, PEAKS, 1.0)  # malformed
    assert "error" in table and (tmp_path / "ceilings.json").exists()
    assert (tmp_path / "ceilings.md").read_text() == ""
    ceilings.write(tmp_path, _profile(), PEAKS, 5000.0)
    assert (tmp_path / "ceilings.md").read_text().startswith("## Ceilings")
    assert json.loads((tmp_path / "ceilings.json").read_text())["rows"]


def test_planner_prompt_ranks_by_ceiling():
    card = {"repo_id": "org/m", "modality": "text"}
    plan = prompts.planner_prompt(card, {"median_ms": 1.0}, "# Profile", ["triton"], 3, "py", "tc")
    assert "*Ceilings* table" in plan and "ceiling × share" in plan


def test_peaks_cache_upgrade_and_low_precision_peaks(tmp_path, monkeypatch):
    monkeypatch.setattr(toolchain, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("KERNEL_AGENT_LOCK_HELD", "1")
    monkeypatch.setattr(roofline, "_MEASURE_FAILED", False)
    gpu = toolchain.GPUInfo("Fake GPU", (12, 0), 16.0, 70, 48.0)
    tc = toolchain.Toolchain(
        gpu=gpu, torch_version="2.0", torch_cuda=None, cuda_home=None, nvcc_version=None
    )
    monkeypatch.setattr(toolchain, "setup", lambda: tc)
    path = toolchain.peaks_path("Fake GPU", "2.0")
    old = {"dram_gbps": 500.0, "tflops": {"bfloat16": 90.0}}  # before FP8 (no version)
    path.write_text(json.dumps(old))
    runs = []

    def failing_run(cmd, **kwargs):
        runs.append(cmd)
        raise OSError("GPU busy")

    monkeypatch.setattr(roofline.subprocess, "run", failing_run)
    assert roofline.ensure_peaks() == old and len(runs) == 1  # the old cache stays in use
    assert roofline.ensure_peaks() == old and len(runs) == 1  # not retried in this process

    monkeypatch.setattr(roofline, "_MEASURE_FAILED", False)
    new = {**old, "version": roofline.PEAKS_VERSION, "tflops": {**old["tflops"], "fp": 1.0}}

    def measuring_run(cmd, **kwargs):
        runs.append(cmd)
        path.write_text(json.dumps(new))
        return type("Proc", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(roofline.subprocess, "run", measuring_run)
    assert roofline.ensure_peaks() == new and len(runs) == 2  # measured once more
    assert roofline.ensure_peaks() == new and len(runs) == 2

    line = toolchain.format_peaks(
        {
            "dram_gbps": 767.1,
            "tflops": {"bfloat16": 99.4, roofline.FP8: 332.6, roofline.FP4: 640.7},
            "tflops_unavailable": {"x": "y"},
        }
    )
    assert line == "copy DRAM 767 GB/s, matmul bf16/fp8/nvfp4 99 / 333 / 641 TFLOP/s, no x matmul"
