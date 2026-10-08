"""INT8 precision classes (#178): ``int8_weights`` (weight-only) and ``int8_w8a8`` (IMMA,
int32 accumulation) for GPUs without FP8 tensor cores and as an option everywhere. The
reference math (quantisers, exact int32 products, SmoothQuant), the classes through the
planner schema / policy, engineer contracts, GPU gating per faked arch, the ceilings column,
the speed of light, the instruction rate and the backend policy; ``gpu``: ``torch._int_mm``
against the CPU path and the bundled examples on the evaluator. The tier on captured,
redrawn and scaled inputs: tests/test_perturbed_calibration.py."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch import nn

from kernel_agent import backends, gpu_arch, pivot, precisions, selftest
from kernel_agent.agent import prompts
from kernel_agent.kernels import compare, mma_peaks, quant, roofline
from kernel_agent.kernels.evaluate import evaluate, run_evaluation
from kernel_agent.profiling import ceilings

# ---------------------------------------------------------------- reference math


def test_quantize_int8_per_channel():
    gen = torch.Generator().manual_seed(0)
    w = torch.randn(64, 256, generator=gen)
    w[3] = 0  # a row of zeros
    w[5, 7] = 50.0  # a row with an outlier
    q, scale = quant.quantize_int8(w.to(torch.bfloat16))
    assert q.shape == (64, 256) and q.dtype == torch.int8 and q.is_contiguous()
    assert scale.shape == (64,) and scale.dtype == torch.float32
    assert int(q.abs().max()) == 127 and int(q.min()) >= -127  # -128 never used
    assert scale[3] == 1.0 and not q[3].any()
    amax = w.to(torch.bfloat16).float().abs().amax(dim=1)
    assert torch.equal(scale[amax > 0], amax[amax > 0] * quant.INT8_STEP)
    assert pytest.approx(1 / 127, rel=1e-7) == quant.INT8_STEP
    report = quant.int8_error(w, q, scale)
    assert report["format"] == "int8" and report["bytes"]["after"] == 64 * 256 + 4 * 64
    assert report["crest"] > 15 and report["underflow"] > 0.001  # the outlier row
    gauss = quant.int8_error(w[:3], *quant.quantize_int8(w[:3]))
    assert 0.004 < gauss["rel_l2"] < 0.012  # int8 per channel on Gaussian rows (e4m3: ~0.026)
    assert gauss["rel_l2"] < quant.fp8_error(w[:3], *quant.quantize_fp8(w[:3]))["rel_l2"] / 2
    # codes round to nearest even: 2.5 and 3.5 steps -> 2 and 4
    row = torch.tensor([[127.0, 2.5, 3.5, -2.5]])
    assert quant.quantize_int8(row)[0].tolist() == [[127, 2, 4, -2]]
    with pytest.raises(ValueError, match="2-D"):
        quant.quantize_int8(w[0])
    with pytest.raises(ValueError, match="non-finite"):
        quant.quantize_int8(torch.full((2, 4), math.inf))


def test_quantize_int8_activations_per_token():
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(2, 5, 256, generator=gen).to(torch.bfloat16)
    x[0, 1] = 0  # a token of zeros
    x[1, 2, 7] = 300.0  # an outlier token
    q, scale = quant.quantize_int8_activations(x)
    assert q.shape == (10, 256) and q.dtype == torch.int8 and scale.shape == (10,)
    rows = x.float().reshape(10, 256)
    assert scale[1] == 1.0 and not q[1].any()
    nonzero = rows.abs().amax(dim=1) > 0
    assert torch.equal(q.abs().amax(dim=1)[nonzero], torch.full((9,), 127, dtype=torch.int8))
    deq = q.float() * scale[:, None]
    rel = (deq - rows).norm(dim=1) / rows.norm(dim=1).clamp_min(1e-30)
    assert float(rel[nonzero].max()) < 0.3 and float(rel[[0, 2, 3, 4]].max()) < 0.012
    # SmoothQuant factors divide the activations before quantising
    s = torch.rand(256, generator=gen) + 0.5
    qs, ss = quant.quantize_int8_activations(x, s)
    assert torch.equal(qs, quant.quantize_int8_activations(x.float() / s)[0])
    assert torch.equal(ss, quant.quantize_int8_activations(x.float() / s)[1])
    with pytest.raises(ValueError, match="smoothing factors"):
        quant.quantize_int8_activations(x, torch.ones(3))


def test_int8_matmul_is_exact():
    gen = torch.Generator().manual_seed(0)
    for m, k, n in ((1, 64, 8), (5, 3000, 24), (17, 4096, 40)):  # K past 1024: chunked
        a = torch.randint(-127, 128, (m, k), generator=gen, dtype=torch.int8)
        b = torch.randint(-127, 128, (n, k), generator=gen, dtype=torch.int8)
        exact = a.long() @ b.long().T
        got = quant.int8_matmul(a, b)
        assert got.dtype == torch.int32 and torch.equal(got.long(), exact)
    worst = torch.full((2, 4096), 127, dtype=torch.int8)  # 4096 x 127² > 2^24: still exact
    assert int(quant.int8_matmul(worst, -worst)[0, 0]) == -4096 * 127 * 127
    assert quant.int8_matmul(worst[:0], worst).shape == (0, 2)
    with pytest.raises(ValueError, match="int8"):
        quant.int8_matmul(worst.float(), worst)


def test_int8_w8a8_linear_and_error_report():
    gen = torch.Generator().manual_seed(1)
    w = torch.randn(256, 512, generator=gen) * 512**-0.5
    bias = torch.randn(256, generator=gen)
    x = torch.randn(3, 7, 512, generator=gen)
    q, scale = quant.quantize_int8(w)
    y = quant.int8_w8a8_linear(x, q, scale, bias)
    xq, xs = quant.quantize_int8_activations(x)
    acc = (xq.long() @ q.long().T).float()  # exact integers in fp32 here
    expected = acc * xs[:, None] * scale[None, :] + bias
    assert y.shape == (3, 7, 256) and y.dtype == x.dtype
    assert torch.equal(y.reshape(21, 256), expected)  # the epilogue's roundings, bit for bit
    ref = x @ w.T + bias
    rel = float((y - ref).norm() / ref.norm())
    fp8 = quant.fp8_w8a8_linear(x, *quant.quantize_fp8(w), bias)
    assert 0.004 < rel < float((fp8 - ref).norm() / ref.norm())  # Gaussian: int8 beats e4m3

    report = quant.int8_w8a8_error(w, q, scale, x)
    assert report["activations"] == "int8 per token"
    assert report["rel_l2"] == quant.int8_error(w, q, scale)["rel_l2"]
    assert 0.005 < report["activation_rel_l2"] < 0.02 and report["activation_underflow"] < 0.02
    assert 3 < report["activation_crest"] < 6 and report["output_cosine"] > 0.9995
    assert 0.005 < report["output_rel_l2"] < 0.02 and abs(report["output_norm_ratio"] - 1) < 0.01

    w8 = quant.int8_weights_linear(x, q, scale, bias)
    assert float((w8 - ref).norm() / ref.norm()) < rel  # weight-only: no activation error


def _outliers(k: int = 1024, rows: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Activations with a massive input channel (crest ~29, as the VoxCPM2 LocDiT MLP's
    inputs) and a Gaussian weight."""
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(rows, k, generator=gen)
    x[:, 0] = 64.0 * torch.sign(torch.randn(rows, generator=gen))
    w = torch.randn(512, k, generator=gen) * k**-0.5
    return x, w


def test_outlier_channels_cost_int8_more_than_fp8_and_smoothquant_recovers():
    x, w = _outliers()
    q, scale = quant.quantize_int8(w)
    plain = quant.int8_w8a8_error(w, q, scale, x)
    assert plain["activation_crest"] > 25 and plain["activation_underflow"] > 0.1
    fp8 = quant.fp8_w8a8_error(w, *quant.quantize_fp8(w), x)
    assert plain["output_rel_l2"] > 1.5 * fp8["output_rel_l2"]
    s = quant.smoothquant_factors(x.abs().amax(dim=0), w, 0.4)
    smooth = quant.int8_w8a8_error(w, *quant.quantize_int8(w, s), x, smooth=s)
    assert smooth["activations"] == "int8 per token (SmoothQuant)"
    assert smooth["output_rel_l2"] < plain["output_rel_l2"] / 2
    y = quant.int8_w8a8_linear(x, *quant.quantize_int8(w, s), smooth=s)
    ref = x @ w.T
    assert float((y - ref).norm() / ref.norm()) == pytest.approx(smooth["output_rel_l2"], rel=1e-3)


def test_smoothquant_factors():
    x, w = _outliers(k=64, rows=8)
    amax = x.abs().amax(dim=0)
    amax[5] = 0  # a dead channel
    s = quant.smoothquant_factors(amax, w, 0.5)
    expected = amax.sqrt() / w.abs().amax(dim=0).sqrt()
    assert s.shape == (64,) and s[5] == 1.0
    keep = torch.arange(64) != 5
    assert torch.allclose(s[keep], expected[keep])
    assert float(s[0]) > 3 * float(s[keep][1:].median())  # the outlier channel moves most
    # exact in real arithmetic: (x / s) @ (W * s)ᵀ == x @ Wᵀ
    xd, wd, sd = x.double(), w.double(), s.double()
    assert torch.allclose((xd / sd) @ (wd * sd).T, xd @ wd.T)
    assert torch.equal(quant.smoothquant_factors(amax, w, 0.0)[keep], 1 / w.abs().amax(0)[keep])
    with pytest.raises(ValueError, match="alpha"):
        quant.smoothquant_factors(amax, w, 1.5)
    with pytest.raises(ValueError, match="inputs"):
        quant.smoothquant_factors(amax[:10], w)


# ---------------------------------------------------------------- precision classes


INT8 = ("int8_weights", "int8_w8a8")


def test_precision_classes_tier_and_defaults():
    for name in INT8:
        assert name in compare.REDUCED_PRECISIONS and name in compare.PRECISIONS
        assert compare.tier_for("near-lossless", name) == compare.NEAR_LOSSLESS_TIER
        assert compare.tier_for("relaxed", name) == compare.RELAXED_TIER  # #175: 8-bit tier
        assert compare.tier_for("exact", name) == compare.EXACT_TIER
        assert name in precisions.default("near-lossless") and name not in precisions.OPT_IN
        assert name in precisions.default("relaxed")  # the default mode of new runs
        enum = prompts.PLAN_SCHEMA["properties"]["targets"]["items"]["properties"]["precision"]
        assert name in enum["enum"]
        assert precisions.parse(f"{name},exact") == ["exact", name]
    assert set(INT8) <= set(roofline.WEIGHT_BITS) and roofline.MATH_DTYPE["int8_w8a8"] == "int8"


@pytest.mark.parametrize(
    ("cap", "w8a8", "fp8"),
    [((7, 0), False, False), ((7, 5), True, False), ((8, 6), True, False), ((8, 9), True, True)],
)
def test_int8_w8a8_follows_the_gpu(cap, w8a8, fp8):
    allowed = precisions.allowed("near-lossless", None, cap)
    assert "int8_weights" in allowed  # weight-only: dequantised in registers, every GPU
    assert ("int8_w8a8" in allowed) is w8a8 and ("fp8_w8a8" in allowed) is fp8
    refused = precisions.gpu_refused("near-lossless", None, cap)
    assert ("int8_w8a8" in refused) is not w8a8
    if not w8a8:
        why = precisions.refusal("int8_w8a8", precisions.allowed("near-lossless"), cap)
        assert why and "IMMA" in why and "sm_75+" in why
        assert "int8_w8a8" in precisions.describe(allowed, cap).split("not on sm_70: ")[1]
    policy = prompts.precision_policy("near-lossless", None, cap)
    assert ('`precision: "int8_w8a8"`' in policy) is w8a8
    assert ('`precision: "fp8_w8a8"`' in policy) is fp8
    assert '`precision: "int8_weights"`' in policy
    if w8a8 and not fp8:  # Turing / Ampere: INT8 is the 8-bit compute class
        assert "INT8 activation scales stay dynamic" in policy
        summary = gpu_arch.summary_lines(_Gpu(cap), None)
        assert any("8-bit tensor-core math here is INT8" in line for line in summary)


class _Gpu:
    def __init__(self, cap: tuple[int, int]) -> None:
        self.capability, self.smem_per_block_kb = cap, 99.0


def test_precision_notes_per_family():
    assert "m8n8k16" in str(gpu_arch.precision_note("int8_w8a8", (7, 5)))
    assert "wgmma" in str(gpu_arch.precision_note("int8_w8a8", (9, 0)))
    assert "kind::i8" in str(gpu_arch.precision_note("int8_w8a8", (10, 0)))
    assert "1/30" in str(gpu_arch.precision_note("int8_w8a8", (10, 3)))  # Blackwell Ultra
    assert gpu_arch.precision_note("int8_w8a8", (8, 6)) is None  # IMMA m16n8k32 is the path
    assert gpu_arch.precision_note("int8_w8a8", (12, 0)) is None
    assert gpu_arch.precision_note("int8_weights", (8, 0)) is None  # no e4m3 emulation
    turing = gpu_arch.summary_lines(_Gpu((7, 5)), None)
    assert any(line.startswith("INT8 W8A8: Turing's IMMA") for line in turing)
    assert not any("INT8 W8A8:" in line for line in gpu_arch.summary_lines(_Gpu((9, 0)), None))


def test_planner_policy_and_exact_runs():
    exact = prompts.precision_policy("exact")
    assert "`int8_weights`, `int8_w8a8`" in exact and "is refused" in exact
    near = prompts.precision_policy("near-lossless")
    assert near.index('"fp8_weights"') < near.index('"int8_weights"') < near.index('"fp8_w8a8"')
    assert "SmoothQuant" in near and "crest" in near and "sm_103" in near
    only = prompts.precision_policy("near-lossless", ["exact", "fp8_w8a8"])
    assert "`int8_weights`, `int8_w8a8`" in only.split("not allowed")[0]  # named as refused
    assert '`precision: "int8_w8a8"`' not in only
    relaxed = prompts.precision_policy("relaxed", None, (8, 6))  # #175, the default mode
    assert '`precision: "int8_w8a8"`' in relaxed and "INT8 activation scales" in relaxed
    assert "**relaxed** run" in relaxed and '`precision: "fp8_w8a8"`' not in relaxed


def _target(precision: str) -> dict:
    return {
        "id": "dit",
        "module_class": "Linear",
        "why": "w",
        "approach": "a",
        "backends": ["triton", "cuda"],
        "precision": precision,
        "precision_why": "M=352, compute bound",
    }


@pytest.mark.parametrize(
    ("precision", "needles"),
    [
        (
            "int8_w8a8",
            (
                "INT8 W8A8 (INT8 tensor-core math, IMMA)",
                "quantize_int8, quantize_int8_activations, int8_matmul",
                "out_dtype=tl.int32",
                "triton_int8_w8a8_gemm.py",
                "cuda_int8_skinny_gemm.py",
                "smoothquant_factors",
                "never one static or per-tensor scale",
                "int8_w8a8_error(weight, q, scale, x)",
            ),
        ),
        (
            "int8_weights",
            (
                "INT8 weight-only",
                "quantize_int8, int8_weights_linear, int8_error",
                "cuda_int8_gemv",
            ),
        ),
    ],
)
def test_engineer_prompt_states_the_int8_contracts(precision, needles):
    cases = [{"signature": "a0[352, 1024]:bfloat16", "count": 540}]
    near = {"cases": cases, "tier": "near-lossless", "precision": precision}
    target = _target(precision)
    text = prompts.engineer_prompt(target, near, ["triton"], "python", "toolchain", 10, None)
    assert prompts.reduced_precision(target, near) == precision
    assert f"# Precision: `{precision}`" in text and "near-lossless tolerance tier" in text
    for needle in needles:
        assert needle in text, needle
    assert "FP8 W8A8 (FP8 tensor-core math)" not in text


def test_pivot_to_int8_and_its_research_offer():
    spec = {"id": "dit_layer", "module_class": "Layer", "precision": "fp8_w8a8"}
    proposal = {"precision": "int8_w8a8", "precision_why": "INT8 W8A8 floor 0.4 ms < W8A8 0.6"}
    assert pivot.check(spec, proposal, quality="near-lossless", taken=set()) is None
    assert pivot.pivot_spec(spec, proposal)["id"] == "dit_layer__int8_w8a8"
    assert pivot.family("dit_layer__int8_weights") == "dit_layer"
    block = prompts._pivot_block(spec, Path("targets/dit_layer/pivot.json"))
    assert "`int8_w8a8`" in block and "`int8_weights`" in block and "INT8 W8A8" in block


# ---------------------------------------------------------------- ceilings and speed of light

PEAKS = {
    "version": 5,
    "arch": "sm_86",
    "dram_gbps": 900.0,
    "launch_floor_us": 6.0,
    "tflops": {"bfloat16": 70.0, "float32": 35.0, roofline.INT8: 280.0},
}


def _profile() -> dict:
    work = {
        "group": "model.dit.mlp",
        "phase": "prefill",
        "instances": 1,
        "calls": 540,
        "inclusive_ms": 600.0,
        "flops": {"bfloat16": 2 * 352 * 8192 * 1024 * 540},
        "weight_elems": 8192 * 1024,
        "weight_bytes": 2 * 8192 * 1024,
        "io_bytes": 0,
        "signature": "",
    }
    return {"hooked_wall_ms": 1000.0, "classes": [{"cls": "Mlp", "is_leaf": False, "work": [work]}]}


def test_ceilings_have_an_int8_column_where_the_gpu_has_imma():
    table = ceilings.build(_profile(), PEAKS, 1000.0)
    assert "int8_w8a8" in table["columns"] and {"w8a8", "mxfp8"} <= set(table["gpu_hidden"])
    row = table["rows"][0]
    flops = 2 * 352 * 8192 * 1024 * 540
    assert row["floors"]["int8_w8a8"] == pytest.approx(flops / 280e12 * 1e3, rel=1e-3)
    assert row["floors"]["int8_w8a8"] < row["floors"]["exact"]
    md = ceilings.markdown(table)
    assert "| INT8 W8A8 |" in md and "INT8 W8A8 at INT8 280 TOPS" in md
    assert "also INT8 weight-only's floor" in md
    old = {**PEAKS, "arch": "sm_70"}  # no IMMA: the column is not shown
    assert "int8_w8a8" in ceilings.build(_profile(), old, 1000.0)["gpu_hidden"]
    # the scheduler's floor of a target per precision
    assert ceilings.target_precision("int8_w8a8") is ceilings.PRECISIONS["int8_w8a8"]
    weights = ceilings.target_precision("int8_weights")
    assert weights.label == "INT8 w" and weights.weight_bytes == 1.0 and weights.peak is None
    assert ceilings.target_columns(["exact", "int8_weights"]) == ["exact", "fp8_weights"]
    assert "int8_w8a8" not in ceilings.columns(["exact", "fp8_w8a8"])


class _Gemm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(256, 512, bias=False).to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj(x)
        return y @ y.transpose(-1, -2)


def test_speed_of_light_counts_int8_gemms_at_the_int8_peak():
    module, x = _Gemm(), torch.randn(352, 256, dtype=torch.bfloat16)
    exact = roofline.count_case(module, (x,), {})
    w8a8 = roofline.count_case(module, (x,), {}, precision="int8_w8a8")
    weights = roofline.count_case(module, (x,), {}, precision="int8_weights")
    gemm, attn = 2 * 352 * 256 * 512, 2 * 352 * 352 * 512
    assert w8a8.flops == {roofline.INT8: gemm, "bfloat16": attn}
    assert weights.flops == exact.flops and w8a8.min_bytes == weights.min_bytes < exact.min_bytes
    peaks = {"dram_gbps": 1e9, "tflops": {"bfloat16": 100.0, roofline.INT8: 200.0}}
    sol = roofline.sol_time(w8a8, peaks)
    assert sol["sol_ms"] == pytest.approx((gemm / 200 + attn / 100) / 1e9)
    assert "peak_missing" not in sol
    assert roofline.sol_time(w8a8, {**peaks, "tflops": {"bfloat16": 100.0}})["peak_missing"] == (
        roofline.INT8
    )
    assert roofline.PEAKS_VERSION >= 5  # caches without the INT8 peak are measured again


def test_the_imma_instruction_rate():
    s8 = mma_peaks.BY_KEY[mma_peaks.S8_S32]
    src = mma_peaks.source(s8)
    assert "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32" in src and '"+r"(n[c][0])' in src
    assert mma_peaks.flops(s8, 70) == mma_peaks.flops(mma_peaks.BY_KEY["e4m3_f32"], 70)
    assert mma_peaks.available(s8, (8, 0)) is None and mma_peaks.available(s8, (9, 0)) is None
    assert "needs sm_80+" in str(mma_peaks.available(s8, (7, 5)))
    rates = {"bf16_f32": 104.0, "e4m3_f32": 206.0, mma_peaks.S8_S32: 410.0}
    line = mma_peaks.describe(rates)
    assert line == "mma.sync bf16 HMMA.F32 104 · e4m3 QMMA.F32 206 TFLOP/s · s8 IMMA.S32 410 TOPS"
    assert mma_peaks.describe({mma_peaks.S8_S32: 300.0}) == "mma.sync s8 IMMA.S32 300 TOPS"
    summary = "GPU: NVIDIA A100 (sm_80, 108 SMs)\npeaks: " + line
    assert gpu_arch.from_summary(summary).mma == rates


# ---------------------------------------------------------------- backend policy and examples


def test_backend_policy_has_an_int8_gemm_class():
    spec = {
        "module_class": "Qwen3MLP",
        "precision": "int8_w8a8",
        "capture": {"cases": [{"signature": "a0[32, 11, 1024]:bfloat16", "count": 540}]},
    }
    assert backends.target_class(spec) == "int8_gemm"
    decode = {**spec, "capture": {"cases": [{"signature": "a0[1, 1, 1024]", "count": 9}]}}
    assert backends.target_class(decode) == "small_m_gemm"
    ampere = gpu_arch.Facts("NVIDIA A100", (8, 0))
    text = backends.policy_text(["triton", "cuda"], ampere)
    assert "Compute-bound INT8 GEMM" in text and "Compute-bound FP8 GEMM" not in text
    assert "The 8-bit compute class is INT8 W8A8" in text
    volta = gpu_arch.Facts("Tesla V100", (7, 0))
    assert "INT8 GEMM" not in backends.policy_text(["triton", "cuda"], volta)
    turing = backends.policy("int8_gemm", gpu_arch.Facts("Tesla T4", (7, 5)))
    assert "m8n8k16" in turing.first and turing.order[0] == "cuda"
    assert "wgmma" in backends.policy("int8_gemm", gpu_arch.Facts("H100", (9, 0))).first
    note = backends.engineer_note({**spec, "module_class": "Qwen3DecoderLayer"}, ["cuda"], ampere)
    assert "The GEMMs inside (M = 352)" in note and "tl.dot` on int8" in note


def test_examples_and_selftest_tables():
    for table in (selftest.INT8_W8A8_EXAMPLES, selftest.INT8_WEIGHT_EXAMPLES):
        for name in table:
            path = prompts.EXAMPLES_DIR / name
            source = path.read_text()
            assert gpu_arch.example_requirement(path)[0] == "sm_80+", name
            for needle in ("def build(", "quantize_int8", "int8_error", "ARCHS_WHY"):
                assert needle in source, (name, needle)
    triton = (prompts.EXAMPLES_DIR / "triton_int8_w8a8_gemm.py").read_text()
    for needle in ("out_dtype=tl.int32", "tl.div_rn", "custom_op", "register_fake", "INT8_STEP"):
        assert needle in triton, needle
    skinny = (prompts.EXAMPLES_DIR / "cuda_int8_skinny_gemm.py").read_text()
    assert "m16n8k32.row.col.s32.s8.s8.s32" in skinny and "__fdiv_rn" in skinny
    assert "__byte_perm" in (prompts.EXAMPLES_DIR / "cuda_int8_gemv.py").read_text()
    assert selftest._EXAMPLES["int8_w8a8"] is selftest.INT8_W8A8_EXAMPLES
    text = prompts.knowledge("low_precision.md")
    for needle in ("## INT8", "int8_w8a8", "int8_weights", "IMMA", "SmoothQuant", "410 TOPS"):
        assert needle in text, needle


# ---------------------------------------------------------------- the tier on an INT8 candidate

CANDIDATE = """import torch
from kernel_agent.kernels import quant


class W8A8(torch.nn.Module):
    def __init__(self, ref):
        super().__init__()
        self.q1, self.s1 = quant.quantize_int8(ref.up.weight)
        self.q2, self.s2 = quant.quantize_int8(ref.down.weight)
        self.s1 = self.s1 * {weight_scale}

    def forward(self, x):
        h = torch.nn.functional.silu(quant.int8_w8a8_linear(x, self.q1, self.s1))
        return quant.int8_w8a8_linear(h, self.q2, self.s2)


def build(reference):
    return W8A8(reference)
"""


class Mlp(nn.Module):
    def __init__(self, hidden: int = 256, inter: int = 1024) -> None:
        super().__init__()
        self.up = nn.Linear(hidden, inter, bias=False)
        self.down = nn.Linear(inter, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(nn.functional.silu(self.up(x)))


def test_an_int8_w8a8_candidate_passes_near_lossless_and_fails_exact(tmp_path, monkeypatch):
    from kernel_agent.profiling.capture import capture_calls

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(0)
    mlp = Mlp().to(torch.bfloat16)
    with torch.no_grad():
        mlp.up.weight.normal_(0, 256**-0.5)
        mlp.down.weight.normal_(0, 1024**-0.5)
    calls = [((torch.randn(32, 11, 256, dtype=torch.bfloat16),), {}, 540)]
    near, exact = tmp_path / "near.pt", tmp_path / "exact.pt"
    capture_calls(mlp, calls, near, tier="near-lossless", precision="int8_w8a8")
    capture_calls(mlp, calls, exact)
    good = tmp_path / "good.py"
    good.write_text(CANDIDATE.format(weight_scale=1.0))
    result = evaluate(near, good, device="cpu")
    assert result["correct"] and result["tolerance_tier"] == "near-lossless", result
    assert 0.005 < result["cases"][0]["max_rel_l2"] < 0.03  # e4m3 W8A8 here: ~0.03
    assert evaluate(exact, good, device="cpu")["status"] == "incorrect"
    bad = tmp_path / "bad.py"
    bad.write_text(CANDIDATE.format(weight_scale=1.05))
    assert evaluate(near, bad, device="cpu")["status"] == "incorrect"


# ---------------------------------------------------------------- GPU


@pytest.mark.gpu
def test_int_mm_matches_the_exact_cpu_path():
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    gen = torch.Generator().manual_seed(0)
    for m, k, n in ((1, 2048, 1024), (5, 1024, 6144), (352, 4096, 1024), (33, 3000, 40)):
        a = torch.randint(-127, 128, (m, k), generator=gen, dtype=torch.int8)
        b = torch.randint(-127, 128, (n, k), generator=gen, dtype=torch.int8)
        assert torch.equal(quant.int8_matmul(a.cuda(), b.cuda()).cpu(), quant.int8_matmul(a, b))
        x = torch.randn(m, k, generator=gen).to(torch.bfloat16)
        w = (torch.randn(n, k, generator=gen) * k**-0.5).to(torch.bfloat16)
        q, s = quant.quantize_int8(w)
        cpu = quant.int8_w8a8_linear(x, q, s)
        gpu = quant.int8_w8a8_linear(x.cuda(), q.cuda(), s.cuda())
        assert torch.equal(gpu.cpu(), cpu)  # the same codes, integers and epilogue roundings


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("name", "precision"),
    [
        *((n, "int8_w8a8") for n in sorted(selftest.INT8_W8A8_EXAMPLES)),
        *((n, "int8_weights") for n in sorted(selftest.INT8_WEIGHT_EXAMPLES)),
    ],
)
def test_int8_examples_pass_the_reduced_tier_and_fail_exact(name, precision, tmp_path):
    from kernel_agent import toolchain

    tc = toolchain.setup()
    if not selftest.int8_supported(tc):
        pytest.skip("needs the triton and cuda backends on sm_80+")
    tables = {
        "int8_w8a8": selftest.INT8_W8A8_EXAMPLES,
        "int8_weights": selftest.INT8_WEIGHT_EXAMPLES,
    }
    k, n, calls = tables[precision][name]
    near = selftest.make_linear_capture(
        tmp_path / "near.pt", k, n, calls, tier="near-lossless", precision=precision
    )
    exact = selftest.make_linear_capture(tmp_path / "exact.pt", k, n, calls)
    example = prompts.EXAMPLES_DIR / name
    result = run_evaluation(near, example)
    assert result["status"] == "ok" and result["correct"], result
    assert result["tolerance_tier"] == "near-lossless" and result["parent_check"] == "ok"
    for case in result["cases"]:
        assert case["max_rel_l2"] < 0.02 and case["min_cosine"] > 0.9998  # int8: < e4m3's 0.026
    assert run_evaluation(exact, example, quick=True)["status"] == "incorrect"


@pytest.mark.gpu
def test_the_examples_equal_the_reference_bit_for_bit():
    from kernel_agent import toolchain
    from kernel_agent.kernels.evaluate import load_candidate_module

    if not selftest.int8_supported(toolchain.setup()):
        pytest.skip("needs the triton and cuda backends on sm_80+")
    torch.manual_seed(0)
    lin = nn.Linear(1024, 2560, bias=True).cuda().to(torch.bfloat16)
    for name, rows in (("triton_int8_w8a8_gemm.py", 300), ("cuda_int8_skinny_gemm.py", 40)):
        module = load_candidate_module(prompts.EXAMPLES_DIR / name).build(lin)
        x = torch.randn(rows, 1024, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            ref = quant.int8_w8a8_linear(x, module.weight_int8, module.weight_scale, lin.bias)
            assert torch.equal(module(x), ref), name


@pytest.mark.gpu
def test_the_imma_rate_is_measured():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("needs sm_80+")
    rates, missing = mma_peaks.measure()
    assert rates.get(mma_peaks.S8_S32, 0) > 1.5 * rates["bf16_f32"], missing  # IMMA: 2x HMMA
