"""W4A4 (#233): the opt-in ``fp4_w4a4`` precision class on block-scaled FP4 tensor cores.

CPU, deterministic: the activation and weight quantisers against a pure-Python e2m1 / e4m3 /
fp32 reference, bit for bit (scale saturation, subnormal and zero block scales included);
the reference GEMM, the rotation, the error report and the per-layer sensitivity probe; the
``near-lossless-fp4a`` / ``relaxed-fp4a`` tiers; the class refused without ``--precisions``
in every quality mode and on GPUs without block-scaled FP4 tensor cores (sm_89, sm_90, ...),
with the reason; the ceilings' W4A4 column, the plan schema and the prompts only when it is
allowed; the speed of light at the NVFP4 peak; the CuTe example compiled for sm_120a with no
GPU. GPU (``-m gpu``): the examples bit for bit against the reference and through the
evaluator in their tier (rejected by the 8-bit one)."""

import importlib.util
import math
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from kernel_agent import backends, gpu_arch, precisions, selftest, skills, toolchain
from kernel_agent.agent import prompts
from kernel_agent.kernels import compare, quant, roofline
from kernel_agent.profiling import ceilings

W4A4 = "fp4_w4a4"
NEAR, RELAXED = "near-lossless", "relaxed"
EXAMPLES = prompts.EXAMPLES_DIR
TRITON = EXAMPLES / "triton_nvfp4_w4a4_gemm.py"
CUTE = EXAMPLES / "cute_nvfp4_w4a4_gemm.py"
HAS_CUTE = importlib.util.find_spec("cutlass") is not None


# ------------------------------------------------------------------ a pure-Python reference


def f32(v: float) -> float:
    """``v`` rounded to the nearest fp32 (ties to even)."""
    return struct.unpack("f", struct.pack("f", v))[0]


def e2m1_code(v: float) -> int:
    """The e2m1 code of ``v`` (nearest magnitude of quant.E2M1_VALUES, ties to the even code,
    saturated at 6; sign bit 3, no negative zero)."""
    mag = min(abs(v), 6.0)
    best = min(range(8), key=lambda i: (abs(quant.E2M1_VALUES[i] - mag), i % 2))
    return best | (8 if v < 0 and best else 0)


def e4m3(v: float) -> float:
    """``v`` (0 <= v <= 448) rounded to e4m3, to nearest even: 3 mantissa bits from 2^-6,
    steps of 2^-9 below."""
    if v == 0:
        return 0.0
    _, e = math.frexp(v)  # v = m * 2^e, m in [0.5, 1)
    step = 2.0**-9 if v < 2.0**-6 else 2.0 ** (e - 4)
    return round(v / step) * step  # exact quotient; round() is half to even


def ref_nvfp4(rows: list[list[float]], outer: list[float], scale_of) -> tuple:
    """Codes, block scales and saturation counts of NVFP4 rows with the given outer scale per
    row and ``scale_of(bmax, outer)`` -> (unrounded ratio, e4m3 scale)."""
    codes, scales, stats = [], [], {"clamped": 0, "saturated": 0, "subnormal": 0, "zero": 0}
    for row, t in zip(rows, outer, strict=True):
        row_codes, row_scales = [], []
        for b0 in range(0, len(row), 16):
            block = row[b0 : b0 + 16]
            bmax = max(abs(v) for v in block)
            ratio, s = scale_of(bmax, t)
            stats["clamped"] += ratio > 448.0
            stats["subnormal"] += 0 < s < 2.0**-6
            stats["zero"] += s == 0 and bmax > 0
            step = f32(s * t)
            row_scales.append(s)
            for v in block:
                q = f32(v / step) if step > 0 else 0.0
                stats["saturated"] += abs(q) > 6.0
                row_codes.append(e2m1_code(q))
        codes.append(row_codes)
        scales.append(row_scales)
    return codes, scales, stats


def ref_activations(x: torch.Tensor, granularity: str) -> tuple:
    rows = x.float().tolist()
    amax = [max(abs(v) for v in row) for row in rows]
    if granularity == "tensor":
        amax = [max(amax)] * len(rows)
    outer = [f32(a * quant.NVFP4_OUTER_STEP) if a > 0 else 1.0 for a in amax]

    def scale_of(bmax: float, t: float) -> tuple[float, float]:
        ratio = f32(bmax / f32(t * 6.0))
        return ratio, e4m3(min(ratio, 448.0))

    return (*ref_nvfp4(rows, outer, scale_of), outer)


def unpack(codes: torch.Tensor) -> list[list[int]]:
    return torch.stack([codes & 0xF, codes >> 4], dim=-1).reshape(codes.shape[0], -1).tolist()


def _activations(rows: int = 256, k: int = 64, seed: int = 0) -> torch.Tensor:
    """Rows at scales from 1e-4 to 1e4, an outlier row (one element 1e6 x the rest: zero and
    subnormal block scales), a row with a zero block and a zero row, as fp32 values of bf16."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, k, generator=gen) * torch.logspace(-4, 4, rows)[:, None]
    x[1] = torch.randn(k, generator=gen) * 4e-3  # blocks 1 and 2: subnormal e4m3 scales
    x[1, 5] = 1e3
    x[1, 48:64] = torch.randn(16, generator=gen) * 1e-5  # block 3: below e4m3, scale 0
    x[2, 16:32] = 0
    x[3] = 0
    return x.to(torch.bfloat16).float()


# ------------------------------------------------------------------ quantisers (CPU)


@pytest.mark.parametrize("granularity", ["token", "tensor"])
def test_nvfp4_activation_quantiser_matches_a_python_reference_bit_for_bit(granularity):
    x = _activations()
    codes, scales, outer = quant.quantize_fp4_activations(x, "nvfp4", granularity)
    ref_codes, ref_scales, stats, ref_outer = ref_activations(x, granularity)
    assert codes.dtype == torch.uint8 and codes.shape == (256, 32)
    assert scales.dtype == torch.float8_e4m3fn and scales.shape == (256, 4)
    assert outer.dtype == torch.float32 and outer.tolist() == ref_outer
    assert scales.float().tolist() == ref_scales
    assert unpack(codes) == ref_codes
    # every corner of the format was exercised: block scales clamped at 448 (outer rounded
    # down), block maxima saturated at ±6 (scale rounded down), subnormal and zero scales
    assert stats["saturated"] > 0 and stats["subnormal"] > 0 and stats["zero"] > 0
    if granularity == "token":
        assert stats["clamped"] > 0
    assert not codes[3].any() and (granularity == "tensor" or outer[3] == 1.0)  # a zero row
    report = quant.fp4_saturation(x, scales, outer)
    assert report["saturated"] > 0 and report["worst_ratio"] > 6.0


def test_mxfp4_activations_and_the_weight_quantisers_match_python_references():
    x = _activations(64, 128)
    codes, scales, outer = quant.quantize_fp4_activations(x, "mxfp4")
    w_codes, w_scales, ts = quant.quantize_fp4(x, "mxfp4")  # the same rule on rows
    assert torch.equal(codes, w_codes) and torch.equal(
        scales.view(torch.uint8), w_scales.view(torch.uint8)
    )
    assert float(ts) == 1.0 and outer.tolist() == [1.0] * 64
    rows = x.tolist()
    for r, row in enumerate(rows):
        for b in range(4):
            bmax = max(abs(v) for v in row[32 * b : 32 * (b + 1)])
            step = 2.0 ** (math.ceil(math.log2(bmax / 6.0)) if bmax > 0 else -127)
            assert scales[r, b].float().item() == step  # the smallest power of two within ±6
            assert unpack(codes[r : r + 1])[0][32 * b : 32 * (b + 1)] == [
                e2m1_code(v / step) for v in row[32 * b : 32 * (b + 1)]
            ]

    # NVFP4 weights: one fp32 tensor scale, e4m3 block scales (quant.quantize_fp4 on the CPU)
    w = _activations(32, 64, seed=1)
    w[3] = 1.0  # quantize_fp4 refuses nothing finite; a zero row is in _activations
    codes, scales, ts = quant.quantize_fp4(w)
    amax = float(w.abs().max())
    assert float(ts) == f32(amax / 2688.0)

    def scale_of(bmax: float, t: float) -> tuple[float, float]:
        ratio = f32(bmax / f32(6.0 * t))
        return ratio, e4m3(min(ratio, 448.0))

    ref_codes, ref_scales, stats = ref_nvfp4(w.tolist(), [float(ts)] * 32, scale_of)
    assert scales.float().tolist() == ref_scales and unpack(codes) == ref_codes
    assert stats["saturated"] > 0


def test_quantiser_shapes_and_refusals():
    x = torch.randn(2, 3, 32)
    codes, scales, outer = quant.quantize_fp4_activations(x)
    assert codes.shape == (6, 16) and scales.shape == (6, 2) and outer.shape == (6,)
    empty = quant.quantize_fp4_activations(torch.zeros(0, 32))
    assert [tuple(t.shape) for t in empty] == [(0, 16), (0, 2), (0,)]
    with pytest.raises(ValueError, match="multiple"):
        quant.quantize_fp4_activations(torch.randn(4, 40))
    with pytest.raises(ValueError, match="granularity"):
        quant.quantize_fp4_activations(x, granularity="channel")
    with pytest.raises(ValueError, match="unknown FP4 format"):
        quant.quantize_fp4_activations(x, "int4")
    # the hardware's tie directions: e2m1 midpoints 0.25 / 1.25 / 2.5 / 5 down, 0.75 / 1.75 / 3.5 up
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -5.0])
    assert quant._round_e2m1(ties).tolist() == [0, 2, 2, 4, 4, 6, 6, 0, 14]


# ------------------------------------------------------------------ math, rotation, reports (CPU)


def _layer(k: int = 256, n: int = 128, m: int = 48, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    w = (torch.randn(n, k, generator=gen) * k**-0.5).to(torch.bfloat16)
    x = torch.randn(m, k, generator=gen).to(torch.bfloat16)
    bias = torch.randn(n, generator=gen).to(torch.bfloat16)
    return w, x, bias


def test_w4a4_linear_is_the_block_scaled_math():
    w, x, bias = _layer()
    codes, scales, ts = quant.quantize_fp4(w)
    y = quant.fp4_w4a4_linear(x.reshape(4, 12, -1), codes, scales, ts, bias)
    assert y.shape == (4, 12, 128) and y.dtype == torch.bfloat16
    xq, xs, outer = quant.quantize_fp4_activations(x)
    acc = quant.fp4_values(xq, xs) @ quant.fp4_values(codes, scales).T
    expected = (acc * (outer[:, None] * float(ts)) + bias.float()).to(torch.bfloat16)
    assert torch.equal(y.reshape(48, 128), expected)
    assert torch.equal(quant.fp4_w4a4_linear(x, codes, scales, float(ts), bias), expected)
    exact = x.float() @ w.float().T + bias.float()
    rel = float((y.reshape(48, 128).float() - exact).norm() / exact.norm())
    assert 0.03 < rel < 0.25  # W4A4's noise on Gaussian data, not a bug
    tensor = quant.fp4_w4a4_linear(x, codes, scales, ts, granularity="tensor")
    assert float((tensor.float() - (exact - bias.float())).norm() / exact.norm()) < 0.25


def test_the_hadamard_rotation_is_orthonormal_and_leaves_the_product():
    h = quant.hadamard(16)
    assert torch.allclose(h @ h, torch.eye(16), atol=1e-6) and torch.equal(h, h.T)
    with pytest.raises(ValueError, match="power of two"):
        quant.hadamard(12)
    w, x, _ = _layer()
    xr, wr = quant.hadamard_rotate(x, 32), quant.hadamard_rotate(w, 32)
    assert torch.allclose(xr @ wr.T, x.float() @ w.float().T, atol=1e-3, rtol=1e-4)
    codes, scales, ts = quant.quantize_fp4(wr)
    y = quant.fp4_w4a4_linear(x, codes, scales, ts, rotate=32)
    exact = x.float() @ w.float().T
    assert float((y.float() - exact).norm() / exact.norm()) < 0.25
    error = quant.fp4_w4a4_error(w, codes, scales, ts, x, rotate=32)
    assert "Hadamard 32" in error["activations"] and error["output_rel_l2"] < 0.25


def test_the_error_report():
    w, x, _ = _layer()
    codes, scales, ts = quant.quantize_fp4(w)
    report = quant.fp4_w4a4_error(w, codes, scales, ts, x)
    assert report["format"] == "nvfp4" and report["activations"].startswith("nvfp4: e2m1 + e4m3")
    assert 0.05 < report["activation_rel_l2"] < 0.15 and report["output_cosine"] > 0.97
    assert 0.9 < report["output_norm_ratio"] < 1.1 and report["activation_crest"] > 2
    assert 0 < report["activation_saturation"]["share"] < 1
    plain = quant.fp4_w4a4_linear(x, codes, scales, ts).float()
    exact = x.float() @ w.float().T
    rel = float((plain - exact).norm() / exact.norm())
    assert report["output_rel_l2"] == pytest.approx(rel, rel=2e-2)  # the report's math


class _Two(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(64, 96, bias=False)
        self.b = nn.Linear(96, 32, bias=False)
        self.odd = nn.Linear(32, 8, bias=False)  # K = 32: a whole number of blocks
        self.skip = nn.Linear(8, 8, bias=False)  # K = 8: not a multiple of 16, skipped

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(self.odd(self.b(torch.relu(self.a(x)))))


def test_the_sensitivity_probe_ranks_every_linear():
    torch.manual_seed(0)
    module = _Two()
    xs = [torch.randn(20, 64) for _ in range(6)]
    rows = quant.fp4_w4a4_sensitivity(module, lambda: [module(x) for x in xs], calls=2)
    assert [r["name"] for r in rows] == sorted(
        ("a", "b", "odd"), key=lambda n: -next(r["w4a4_rel_l2"] for r in rows if r["name"] == n)
    )
    assert all(r["rows"] == 40 for r in rows)  # the first two calls of each
    assert sum(r["flop_share"] for r in rows) == pytest.approx(1.0, abs=1e-3)
    for r in rows:
        assert r["w4a4_rel_l2"] > r["fp8_rel_l2"] > 0 and abs(r["w4a4_norm_change"]) < 0.1
    a = next(r for r in rows if r["name"] == "a")
    x = torch.cat(xs[:2])
    direct = quant.fp4_w4a4_error(module.a.weight, *quant.quantize_fp4(module.a.weight), x)
    assert a["w4a4_rel_l2"] == direct["output_rel_l2"]
    assert not module.a._forward_pre_hooks  # the probe's hooks are gone


# ------------------------------------------------------------------ tiers (CPU)


def test_w4a4_has_its_own_tiers_looser_than_fp4_weights():
    assert W4A4 in compare.REDUCED_PRECISIONS and W4A4 in compare.PRECISIONS
    assert compare.tier_for(NEAR, W4A4) == compare.NEAR_LOSSLESS_FP4A_TIER == "near-lossless-fp4a"
    assert compare.tier_for(RELAXED, W4A4) == compare.RELAXED_FP4A_TIER == "relaxed-fp4a"
    assert compare.tier_for("exact", W4A4) == compare.EXACT_TIER
    fp4a, fp4 = (
        compare.NEAR_LOSSLESS_BOUNDS[t] for t in ("near-lossless-fp4a", "near-lossless-fp4")
    )
    assert fp4a[0] <= fp4[0] and fp4a[1] >= fp4[1] and fp4a[2] >= fp4[2] and fp4a[3] >= fp4[3]
    near = compare.NEAR_LOSSLESS_BOUNDS[compare.NEAR_LOSSLESS_TIER]
    assert fp4a[1] > near[1] and fp4a[2] > near[2]


def test_the_tier_passes_w4a4_rounding_and_rejects_broken_scales():
    w, x, bias = _layer(512, 256, 64)
    ref = (x.float() @ w.float().T + bias.float()).to(torch.bfloat16)
    codes, scales, ts = quant.quantize_fp4(w)
    good = quant.fp4_w4a4_linear(x, codes, scales, ts, bias)
    for tier in ("near-lossless-fp4a", "relaxed-fp4a"):
        assert compare.compare_tensors("y", ref, good, tier=tier)["ok"], tier
    assert not compare.compare_tensors("y", ref, good, tier=NEAR)["ok"]  # the 8-bit tier
    wrong = quant.fp4_w4a4_linear(x, codes, scales, ts * 1.2, bias)
    assert not compare.compare_tensors("y", ref, wrong, tier="relaxed-fp4a")["ok"]
    swapped = (codes >> 4) | ((codes & 0xF) << 4)  # nibbles in the wrong order
    nibbles = quant.fp4_w4a4_linear(x, swapped, scales, ts, bias)
    assert not compare.compare_tensors("y", ref, nibbles, tier="relaxed-fp4a")["ok"]


# ------------------------------------------------------------------ opt-in and the GPU (CPU)


def test_w4a4_is_opt_in_in_every_quality_mode():
    for quality in (NEAR, RELAXED):
        assert W4A4 not in precisions.default(quality) and W4A4 in precisions.FOUR_BIT
        assert W4A4 in precisions.OPT_IN
        why = precisions.refusal(W4A4, precisions.default(quality))
        assert "not allowed" in why and "4-bit precisions are opt-in" in why
        asked = precisions.allowed(quality, ["fp8_weights", W4A4])
        assert asked == ("exact", "fp8_weights", W4A4) and precisions.refusal(W4A4, asked) is None
    assert precisions.allowed("exact", [W4A4]) == ("exact",)
    assert "needs --quality" in precisions.check("exact", [W4A4])
    assert precisions.tier_allowed("near-lossless-fp4a", ("exact", W4A4))
    assert not precisions.tier_allowed("near-lossless-fp4a", precisions.default(NEAR))
    assert not precisions.tier_allowed("relaxed-fp4a", ("exact", "fp4_weights"))
    assert precisions.describe(precisions.default(NEAR)).endswith(
        "(4-bit not allowed: fp4_weights, fp4_w4a4)"
    )


@pytest.mark.parametrize(
    ("arch", "runs"),
    [
        ((8, 0), False),
        ((8, 6), False),
        ((8, 9), False),
        ((9, 0), False),
        ((10, 0), True),
        ((10, 3), True),
        ((12, 0), True),
        ((12, 1), True),
    ],
)
def test_the_gpu_refuses_w4a4_without_block_scaled_fp4_tensor_cores(arch, runs):
    asked = precisions.allowed(NEAR, [W4A4], arch)
    assert (W4A4 in asked) is runs
    why = precisions.refusal(W4A4, ("exact", W4A4), arch)
    unsupported = gpu_arch.precision_unsupported(W4A4, arch)
    if runs:
        assert why is None and unsupported is None
        assert gpu_arch.column_unsupported("w4a4", arch) is None
    else:
        assert "cannot run on this GPU" in why and "block-scaled FP4 tensor cores" in why
        assert gpu_arch.arch_of(arch) in why and "sm_100+" in unsupported
        assert gpu_arch.column_unsupported("w4a4", arch) is not None
        assert W4A4 in precisions.gpu_refused(NEAR, [W4A4], arch)
    note = gpu_arch.precision_note(W4A4, arch)
    if arch[0] == 12:
        assert "MmaMXF4NVF4Op" in note and "mxf4nvf4.block_scale" in note
    elif arch[0] == 10:
        assert "tcgen05.mma kind::mxf4nvf4" in note
    else:
        assert note is None


def test_the_examples_declare_their_gpus():
    triton, cute = (gpu_arch.example_requirement(p)[0] for p in (TRITON, CUTE))
    assert triton == "sm_100+" and cute == "sm_12x"
    for arch in ((8, 9), (9, 0)):
        assert gpu_arch.example_skip(TRITON, arch) and gpu_arch.example_skip(CUTE, arch)
    assert gpu_arch.example_skip(TRITON, (12, 0)) is None
    assert gpu_arch.example_skip(CUTE, (12, 0)) is None and gpu_arch.example_skip(CUTE, (10, 0))
    assert selftest.W4A4_EXAMPLES and selftest.CUTE_W4A4_EXAMPLES


# ------------------------------------------------------------------ ceilings, plan, prompts (CPU)


def test_the_ceilings_column_only_when_allowed():
    default = precisions.default(NEAR)
    assert "w4a4" not in ceilings.columns(default)
    assert "w4a4" not in ceilings.columns(("exact", "fp4_weights"))  # 4-bit weights only
    assert ceilings.columns(("exact", W4A4)) == ["exact", "w4a4"]
    assert ceilings.target_columns((*default, W4A4))[-1] == "w4a4"
    assert ceilings.target_precision(W4A4) is ceilings.PRECISIONS["w4a4"]
    assert ceilings.PRECISIONS["w4a4"].peak == roofline.FP4


def test_the_plan_schema_and_the_policy_offer_w4a4_only_when_allowed():
    default = precisions.default(NEAR)
    enum = prompts.plan_schema(default)["properties"]["targets"]["items"]["properties"]
    assert W4A4 not in enum["precision"]["enum"]
    asked = (*default, W4A4)
    enum = prompts.plan_schema(asked)["properties"]["targets"]["items"]["properties"]
    assert W4A4 in enum["precision"]["enum"]
    assert (
        W4A4
        in prompts.PLAN_SCHEMA["properties"]["targets"]["items"]["properties"]["precision"]["enum"]
    )
    off = prompts.precision_policy(NEAR, default)
    assert '`precision: "fp4_w4a4"`' not in off and "No 4-bit weights or activations" in off
    on = prompts.precision_policy(RELAXED, asked)
    assert '`precision: "fp4_w4a4"`' in on and "*W4A4*" in on and "No 4-bit" not in on
    weights_only = prompts.precision_policy(NEAR, (*default, "fp4_weights"))
    assert "No 4-bit activations (W4A4" in weights_only
    assert '`precision: "fp4_weights"`' in weights_only
    assert "No 4-bit activations (W4A4)" in prompts.precision_note(NEAR, (*default, "fp4_weights"))
    assert "No 4-bit" not in prompts.precision_note(NEAR, asked)
    target = {"id": "dit", "precision": "fp8_w8a8", "pivot_of": None}
    pivot = prompts._pivot_block(target, Path("/r/pivot.json"), asked)
    assert "`fp4_w4a4`" in pivot and "block-scaled FP4 tensor cores" in pivot


def test_the_engineer_gets_the_w4a4_contract():
    spec = {
        "id": "dit",
        "module_class": "Linear",
        "why": "w",
        "approach": "a",
        "backends": ["cute"],
        "precision": W4A4,
        "precision_why": "M=352 GEMMs, compute bound",
    }
    capture = {"cases": [], "tier": "near-lossless-fp4a", "precision": W4A4}
    toolchain = "GPU: NVIDIA GeForce RTX 5070 Ti (sm_120, 16 GB)"
    text = prompts.engineer_prompt(
        spec,
        capture,
        ["cute"],
        "python",
        toolchain,
        10,
        None,
        precisions=(*precisions.default(NEAR), W4A4),
    )
    assert "# Precision: `fp4_w4a4`" in text and "near-lossless-fp4a tolerance tier" in text
    assert "quantize_fp4_activations" in text and "fp4_w4a4_sensitivity" in text
    assert "cute_nvfp4_w4a4_gemm.py" in text and "triton_nvfp4_w4a4_gemm.py" in text
    assert "MmaMXF4NVF4Op" in text  # this GPU's note
    assert "No 4-bit" not in text
    fp8 = prompts.engineer_prompt(
        {**spec, "precision": "fp8_w8a8"},
        {**capture, "tier": NEAR, "precision": "fp8_w8a8"},
        ["cute"],
        "python",
        toolchain,
        10,
        None,
        precisions=(*precisions.default(NEAR), "fp4_weights"),
    )
    assert "No 4-bit activations (W4A4)" in fp8
    assert skills.PRECISION_SKILLS[W4A4] == "fp4-w4a4"


def _facts(name: str, cap: tuple[int, int]) -> gpu_arch.Facts:
    """The facts of a fake GPU, read back from its toolchain summary (no GPU needed)."""
    gpu = toolchain.GPUInfo(name, cap, 32.0, 100, 64.0, 99.0, 100.0)
    found = {"cuda": True, "triton": True, "cute": True, "nvrtc": True, "tilelang": False}
    tc = toolchain.Toolchain(gpu, "2.14", "13.0", None, "13.0", found, [], {}, None)
    return gpu_arch.from_summary(tc.summary())


_MLP = {
    "module_class": "Qwen3MLP",
    "precision": W4A4,
    "capture": {"cases": [{"signature": "a0[2, 176, 1024]:bfloat16", "count": 10}]},
}


@pytest.mark.parametrize(
    ("name", "cap"),
    [
        ("NVIDIA B200", (10, 0)),
        ("NVIDIA B300", (10, 3)),
        ("NVIDIA GeForce RTX 5070 Ti", (12, 0)),
        ("NVIDIA GB10", (12, 1)),
    ],
)
def test_the_backend_policy_has_a_w4a4_row_on_fp4_tensor_cores(name, cap):
    facts = _facts(name, cap)
    assert facts.capability == cap and facts.has("fp4_tc")
    assert backends.target_class(_MLP) == "fp4_gemm"
    assert backends.target_class({**_MLP, "precision": "fp8_w8a8"}) == "fp8_gemm"
    decode = {**_MLP, "capture": {"cases": [{"signature": "a0[1, 1, 1024]", "count": 9}]}}
    assert backends.target_class(decode) == "small_m_gemm"  # memory bound: not W4A4's job
    row = backends.policy("fp4_gemm", facts)
    every = ["cuda", "triton", "cute"]
    text = backends.policy_text(every, facts, (*precisions.default(NEAR), W4A4))
    assert row.label in text and "* Compute-bound W4A4 GEMMs (`fp4_w4a4`, M ≳ 128)" in text
    for off in (None, precisions.default(NEAR), (*precisions.default(NEAR), "fp4_weights")):
        assert "W4A4" not in backends.policy_text(every, facts, off)  # opt-in
    note = backends.engineer_note(_MLP, ["cute", "triton"], facts)
    assert f"Target class: **{row.label}**" in note and row.first in note
    layer = backends.engineer_note({**_MLP, "module_class": "LlamaDecoderLayer"}, every, facts)
    assert f"The GEMMs inside (M = 352): {row.first}" in layer
    if cap[0] == 10:  # datacenter Blackwell: tcgen05, the sm_12x CuTe example not named
        assert "`tcgen05.mma kind::mxf4nvf4`" in row.first and row.order[0] == "triton"
        assert "triton_nvfp4_w4a4_gemm.py" in row.first
        assert "cute_sm100_gemm_tcgen05.py" in row.first
        assert "cute_nvfp4_w4a4_gemm.py" not in text + note
        assert "never a `mma.sync` kernel for a compute-bound W4A4 GEMM on sm_100" in text
        assert "not measured by kernel-agent yet" in row.why
    else:  # GeForce Blackwell: block-scaled FP4 mma.sync, the measured CuTe example first
        assert "MmaMXF4NVF4Op" in row.first and "cute_nvfp4_w4a4_gemm.py" in row.first
        assert row.order[0] == "cute" and "650 TFLOP/s" in row.why and "RTX 5070 Ti" in row.why
        assert "triton_nvfp4_w4a4_gemm.py" in row.second
    assert backends.unrunnable(text + note + layer, cap) == []


@pytest.mark.parametrize("cap", [(7, 5), (8, 6), (8, 9), (9, 0)])
def test_no_w4a4_backend_row_without_fp4_tensor_cores(cap):
    facts = _facts("an older GPU", cap)
    assert not facts.has("fp4_tc")
    assert "W4A4" not in backends.policy_text(["cuda", "triton", "cute"], facts, ("exact", W4A4))


class _Gemm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(256, 512, bias=False).to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def test_speed_of_light_counts_w4a4_gemms_at_the_nvfp4_peak():
    module, x = _Gemm(), torch.randn(352, 256, dtype=torch.bfloat16)
    exact = roofline.count_case(module, (x,), {})
    w4a4 = roofline.count_case(module, (x,), {}, precision=W4A4)
    weights = roofline.count_case(module, (x,), {}, precision="fp4_weights")
    assert w4a4.flops == {roofline.FP4: 2 * 352 * 256 * 512}
    assert w4a4.min_bytes == weights.min_bytes < exact.min_bytes
    weight_bytes = 256 * 512 * 2
    assert exact.min_bytes - w4a4.min_bytes == weight_bytes - (
        weight_bytes // 4 + 256 * 512 // 16 + 4
    )
    peaks = {"dram_gbps": 1e9, "tflops": {"bfloat16": 100.0, roofline.FP4: 600.0}}
    assert roofline.sol_time(w4a4, peaks)["sol_ms"] == pytest.approx(
        2 * 352 * 256 * 512 / 600 / 1e9
    )
    assert roofline.sol_time(w4a4, {**peaks, "tflops": {"bfloat16": 100.0}})["peak_missing"] == (
        roofline.FP4
    )


COMPILE = r"""
import importlib.util, sys
spec = importlib.util.spec_from_file_location("cute_nvfp4_w4a4_gemm", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.compile_quant(1024)
mod.compile_gemm(1024, 1024)
mod.compile_gemm(4096, 1024, has_bias=True)
print("COMPILED")
"""


@pytest.mark.skipif(not HAS_CUTE, reason="no CuTe DSL")
def test_the_cute_example_compiles_for_sm120_without_a_gpu(tmp_path):
    from test_cute_examples import _cpu_env

    (tmp_path / "dump").mkdir()
    out = subprocess.run(
        [sys.executable, "-c", COMPILE, str(CUTE)],
        env=_cpu_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=600,
        cwd=tmp_path,
    )
    assert out.returncode == 0 and "COMPILED" in out.stdout, out.stderr[-3000:]
    ptx = [p.read_text() for p in (tmp_path / "dump").glob("*.ptx")]
    gemms = [t for t in ptx if "kind::mxf4nvf4.block_scale" in t]
    assert len(gemms) == 2
    for text in gemms:
        assert "scale_vec::4X.m16n8k64" in text and ".e2m1.e2m1.f32.ue4m3" in text
        plain = [ln for ln in text.splitlines() if "mma.sync" in ln and "block_scale" not in ln]
        assert plain == [] and "cp.async.bulk.tensor" in text
    assert any("div.rn.f32" in t for t in ptx if t not in gemms)  # the quantiser's IEEE divisions


# ------------------------------------------------------------------ GPU


def _capability() -> tuple[int, int]:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    return torch.cuda.get_device_capability()


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.gpu
def test_scaled_mm_is_bit_for_bit_the_fp32_math(monkeypatch):
    if _capability() < (10, 0):
        pytest.skip("block-scaled FP4 tensor cores: sm_100+")
    for m, k, n in ((1, 256, 256), (17, 96, 48), (352, 1024, 4096), (352, 4096, 1024)):
        w, x, bias = (t.cuda() for t in _layer(k, n, m))
        codes, scales, ts = quant.quantize_fp4(w)
        assert quant._fp4_scaled_mm_ok(x, codes, "nvfp4")
        fast = quant.fp4_w4a4_linear(x, codes, scales, ts, bias)
        with monkeypatch.context() as patch:
            patch.setattr(quant, "_fp4_scaled_mm_ok", lambda *a: False)
            slow = quant.fp4_w4a4_linear(x, codes, scales, ts, bias)
        assert torch.equal(fast, slow), (m, k, n)
        on_cpu = quant.quantize_fp4_activations(x.cpu())
        assert all(
            torch.equal(
                a.cpu().view(torch.uint8) if a.dtype != torch.float32 else a.cpu(),
                b.view(torch.uint8) if b.dtype != torch.float32 else b,
            )
            for a, b in zip(quant.quantize_fp4_activations(x), on_cpu, strict=True)
        )


@pytest.mark.gpu
def test_the_triton_example_is_the_reference_bit_for_bit():
    if _capability() < (10, 0):
        pytest.skip("block-scaled FP4 tensor cores: sm_100+")
    example = _load(TRITON)
    torch.manual_seed(0)
    for m, k, n in ((1, 64, 48), (17, 96, 64), (300, 1024, 4096), (352, 4096, 1024)):
        x = torch.randn(m, k, device="cuda") * torch.logspace(-2, 2, k, device="cuda")
        x[:, 5] *= 100
        x = x.to(torch.bfloat16)
        got = example.quantize_activations(x)
        want = quant.quantize_fp4_activations(x, "nvfp4", "tensor")
        assert torch.equal(got[0], want[0]) and torch.equal(got[2], want[2])
        assert torch.equal(got[1].view(torch.uint8), want[1].view(torch.uint8))
        lin = nn.Linear(k, n, bias=False, device="cuda", dtype=torch.bfloat16)
        module = example.build(lin)
        assert type(module).__name__ == "NVFP4W4A4Linear"
        with torch.no_grad():
            y = module(x)
        ref = quant.fp4_w4a4_linear(
            x, module.weight_fp4, module.weight_scales, module.tensor_scale, granularity="tensor"
        )
        assert torch.equal(y, ref), (m, k, n)


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_CUTE, reason="no CuTe DSL")
def test_the_cute_example_selftest():
    if _capability()[0] != 12:
        pytest.skip("the block-scaled FP4 mma.sync needs sm_120 / sm_121")
    errors = _load(CUTE).selftest(m=352, n=1024, k=1024)
    assert errors["no_bias"] == 0.0 and errors["bias"] < 1e-4


@pytest.mark.gpu
def test_the_w4a4_examples_pass_their_tier_and_fail_the_8bit_one(tmp_path):
    cap = _capability()
    if cap < (10, 0):
        pytest.skip("block-scaled FP4 tensor cores: sm_100+")
    assert selftest.smoke_fp4(tmp_path, True, precision=W4A4)
    if cap[0] == 12 and HAS_CUTE:
        assert selftest.smoke_fp4(
            tmp_path, True, precision=W4A4, examples=selftest.CUTE_W4A4_EXAMPLES
        )
