"""FP4 block-scaled weights (#92): the NVFP4 / MXFP4 helpers, the ``fp4_weights`` precision
and its near-lossless-fp4 tolerance tier (planner, orchestrator, engineer prompt, capture,
speed of light, library, evaluator), and (``gpu``) the bundled FP4 GEMV on the evaluator:
correct in its tier, rejected by the FP8 tier."""

import math

import pytest
import torch
from torch import nn

from kernel_agent import library, orchestrator, selftest
from kernel_agent.agent import prompts
from kernel_agent.kernels import compare, quant, roofline
from kernel_agent.kernels.evaluate import capture_precision, evaluate, run_evaluation
from kernel_agent.profiling.capture import capture_calls, load_capture

FP4 = compare.NEAR_LOSSLESS_FP4_TIER


def _gaussian(rows: int = 256, cols: int = 1024, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, cols, generator=gen) * cols**-0.5).to(torch.bfloat16)


# ---------------------------------------------------------------- quantisation helpers


def test_e2m1_rounds_to_nearest_even_and_saturates():
    values = torch.tensor([0.0, 0.25, 0.26, 0.74, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.01, 6.0])
    codes = quant._round_e2m1(values)
    assert codes.tolist() == [0, 0, 1, 1, 2, 2, 4, 4, 6, 6, 7, 7]
    lut = torch.tensor(quant.E2M1_VALUES)
    assert lut[codes.long()].tolist() == [0, 0, 0.5, 0.5, 1, 1, 2, 2, 4, 4, 6, 6]
    negative = quant._round_e2m1(-values)
    assert negative[0] == 0 and negative[1] == 0  # no negative zero
    assert (negative[2:] == codes[2:] + 8).all()  # sign bit 3
    every = torch.tensor([*quant.E2M1_VALUES, *[-v for v in quant.E2M1_VALUES[1:]]])
    assert torch.equal(quant._round_e2m1(every).long(), torch.tensor([*range(8), *range(9, 16)]))


def test_quantize_nvfp4_layout_and_exact_round_trip():
    gen = torch.Generator().manual_seed(1)
    rows, cols = 8, 64
    # weights that NVFP4 holds exactly: e2m1 values x an e4m3 block scale x a tensor scale
    codes = torch.randint(0, 16, (rows, cols), generator=gen)
    codes[:, ::16] = 7  # every block reaches 6: its scale is block amax / 6
    lut = torch.tensor(quant.E2M1_VALUES)
    values = torch.cat([lut, -lut])[codes]
    block_scale = torch.tensor([0.5, 1.0, 2.0, 0.25]).repeat(rows, 1)
    weight = (values.view(rows, 4, 16) * block_scale[..., None] * 0.01).view(rows, cols)
    weight[3] = 0  # a zero row
    packed, scales, tensor_scale = quant.quantize_fp4(weight)
    assert packed.dtype == torch.uint8 and packed.shape == (rows, cols // 2)
    assert packed.is_contiguous() and scales.is_contiguous()
    assert scales.dtype == torch.float8_e4m3fn and scales.shape == (rows, cols // 16)
    assert tensor_scale.dtype == torch.float32 and tensor_scale.dim() == 0
    assert float(tensor_scale) == pytest.approx(6 * 2.0 * 0.01 / (6 * 448))
    assert not scales[3].float().any() and not packed[3].any()
    deq = quant.dequantize_fp4(packed, scales, tensor_scale, torch.float32)
    torch.testing.assert_close(deq, weight, rtol=1e-6, atol=0)
    assert quant.dequantize_fp4(packed, scales, tensor_scale).dtype == torch.bfloat16
    # packing: element 2j in the low nibble of byte j, 2j + 1 in the high nibble
    nibbles = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(rows, cols)
    expected = torch.where(codes == 8, 0, codes)  # -0 is stored as 0
    expected[3] = 0
    assert torch.equal(nibbles.long(), expected)
    assert packed.view(torch.float4_e2m1fn_x2).shape == (rows, cols // 2)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
def test_quantize_fp4_is_idempotent_and_refuses_bad_input(fmt):
    w = _gaussian()
    codes, scales, ts = quant.quantize_fp4(w, fmt)
    deq = quant.dequantize_fp4(codes, scales, ts, torch.float32)
    again = quant.quantize_fp4(deq, fmt)  # FP4 values are kept exactly
    assert torch.equal(quant.dequantize_fp4(*again, torch.float32), deq)
    if fmt == "nvfp4":  # ... with the same codes and scales (MXFP4 may halve a scale)
        assert torch.equal(again[0], codes) and float(again[2]) == float(ts)
        assert torch.equal(again[1].view(torch.uint8), scales.view(torch.uint8))
    with pytest.raises(ValueError, match="2-D"):
        quant.quantize_fp4(w[0], fmt)
    with pytest.raises(ValueError, match="multiple"):
        quant.quantize_fp4(w[:, :40], fmt)
    bad = w.clone()
    bad[1, 2] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        quant.quantize_fp4(bad, fmt)
    with pytest.raises(ValueError, match="unknown FP4 format"):
        quant.quantize_fp4(w, "int4")


def test_mxfp4_power_of_two_scales():
    w = _gaussian(64, 256)
    codes, scales, ts = quant.quantize_fp4(w, "mxfp4")
    assert scales.dtype == torch.float8_e8m0fnu and scales.shape == (64, 256 // 32)
    assert float(ts) == 1.0
    step = scales.float()
    amax = w.float().view(64, 8, 32).abs().amax(-1)
    # the smallest power of two that keeps the block within ±6: amax / scale in (3, 6]
    assert torch.equal(step, torch.exp2(torch.ceil(torch.log2(amax / 6))))
    assert bool(((amax / step > 3) & (amax / step <= 6)).all())
    deq = quant.dequantize_fp4(codes, scales, ts, torch.float32).view(64, 8, 32)
    top = deq.abs().amax(-1) / step  # the block maximum, rounded, never saturated
    lut = torch.tensor(quant.E2M1_VALUES)
    assert torch.equal(top, lut[quant._round_e2m1(amax / step).long()])
    exact = quant.quantize_fp4(torch.tensor([[6.0] + [0.0] * 31, [1.5] + [0.0] * 31]), "mxfp4")
    assert exact[1].float().tolist() == [[1.0], [0.25]]  # a power-of-two amax / 6: no rounding


def test_fp4_error_report_and_accuracy():
    w = _gaussian(512, 2048)
    x = torch.randn(4, 2048, generator=torch.Generator().manual_seed(2))
    fp8 = quant.fp8_error(w, *quant.quantize_fp8(w), x)
    nv_q = quant.quantize_fp4(w)
    nv = quant.fp4_error(w, *nv_q, x)
    mx = quant.fp4_error(w, *quant.quantize_fp4(w, "mxfp4"), x)
    assert nv["format"] == "nvfp4" and mx["format"] == "mxfp4"
    assert nv["granularity"] == "e4m3 scale per 16 + fp32 tensor scale"
    assert mx["granularity"] == "e8m0 scale per 32"
    deq = quant.dequantize_fp4(*nv_q, torch.float32)
    assert nv["rel_l2"] == pytest.approx(float((deq - w.float()).norm() / w.float().norm()), 1e-3)
    # Gaussian rows: NVFP4 ~0.095, MXFP4 ~0.116, FP8 ~0.026
    assert 0.08 < nv["rel_l2"] < 0.11 < mx["rel_l2"] < 0.13 and fp8["rel_l2"] < 0.03
    assert nv["worst_channel_rel_l2"] >= nv["rel_l2"]
    assert nv["output_rel_l2"] == pytest.approx(nv["rel_l2"], rel=0.3)
    assert 0.99 < nv["output_cosine"] < 0.997
    assert 0.03 < nv["underflow"] < 0.12  # |w| below a quarter of a block's step: 0
    n, k = w.shape
    assert nv["bytes"] == {"before": n * k * 2, "after": n * k // 2 + n * k // 16 + 4}
    assert mx["bytes"]["after"] == n * k // 2 + n * k // 32 + 4
    assert "output_rel_l2" not in quant.fp4_error(w, *nv_q)


# ---------------------------------------------------------------- the tier


def _nvfp4(w: torch.Tensor) -> torch.Tensor:
    return quant.dequantize_fp4(*quant.quantize_fp4(w), w.dtype)


def test_fp4_tier_accepts_nvfp4_weights_and_catches_bugs():
    w = _gaussian(512, 1024, seed=3)
    x = torch.randn(4, 1024, generator=torch.Generator().manual_seed(4)).to(torch.bfloat16)
    ref, fp4 = x @ w.T, x @ _nvfp4(w).T
    result = compare.compare_tensors("y", ref, fp4, tier=FP4)
    assert result["ok"] and result["tier"] == FP4, result
    assert 0.08 < result["rel_l2"] < 0.12
    mx = x @ quant.dequantize_fp4(*quant.quantize_fp4(w, "mxfp4"), w.dtype).T
    assert compare.compare_tensors("y", ref, mx, tier=FP4)["ok"]
    for tier in ("near-lossless", "exact"):  # FP4's error is above the FP8 tier's bounds
        assert not compare.compare_tensors("y", ref, fp4, tier=tier)["ok"]

    codes, scales, ts = quant.quantize_fp4(w)
    swapped = (codes >> 4) | ((codes & 0xF) << 4)  # the two nibbles of a byte swapped
    bugs = {
        "nibbles swapped": quant.dequantize_fp4(swapped, scales, ts, w.dtype),
        "block scales shifted": quant.dequantize_fp4(
            codes, scales.view(torch.uint8).roll(1, 1).view(scales.dtype), ts, w.dtype
        ),
        "tensor scale x2": quant.dequantize_fp4(codes, scales, 2 * ts, w.dtype),
    }
    for label, bad in bugs.items():
        assert not compare.compare_tensors("y", ref, x @ bad.T, tier=FP4)["ok"], label
    row = fp4.clone()
    row[:, ref.abs().amax(0).argmax()] = 0  # the output channel with the largest output dropped
    assert "tolerance away" in compare.compare_tensors("y", ref, row, tier=FP4)["error"]
    scaled = compare.compare_tensors("y", ref, fp4 * 1.05, tier=FP4)
    assert "norm x1.0" in scaled["error"]  # systematic, not noise
    zeros = torch.zeros(64)  # no signal: the exact checks
    assert not compare.compare_tensors("z", zeros, zeros + 0.01, tier=FP4)["ok"]
    assert all(c["ok"] for c in compare.compare_structures([ref], [fp4], tier=FP4))


def test_fp4_precision_maps_to_its_tier():
    assert "fp4_weights" in compare.REDUCED_PRECISIONS and FP4 in compare.TIERS
    assert compare.tier_for("near-lossless", "fp4_weights") == FP4
    assert compare.tier_for("near-lossless", "fp8_weights") == compare.NEAR_LOSSLESS_TIER
    assert compare.tier_for("exact", "fp4_weights") == compare.EXACT_TIER
    assert compare.tier_of({"tier": FP4}) == FP4
    assert set(compare.NEAR_LOSSLESS_BOUNDS) == {compare.NEAR_LOSSLESS_TIER, FP4}
    fp8_bounds, fp4_bounds = (compare.NEAR_LOSSLESS_BOUNDS[t] for t in ("near-lossless", FP4))
    assert fp4_bounds[0] < fp8_bounds[0] and fp4_bounds[1] > fp8_bounds[1]  # looser


def test_fp4_policy_planner_orchestrator_and_engineer_prompt():
    enum = prompts.PLAN_SCHEMA["properties"]["targets"]["items"]["properties"]["precision"]
    assert "fp4_weights" in enum["enum"]
    every = compare.PRECISIONS  # 4-bit is opt-in (--precisions ...,fp4_weights; #131)
    policy = prompts.precision_policy("near-lossless", every)
    assert '`precision: "fp4_weights"`' in policy and "`fp8_weights` is already in use" in policy
    assert '`precision: "fp4_weights"`' not in prompts.precision_policy("near-lossless")
    target = {"id": "mlp", "precision": "fp4_weights", "precision_why": "decode GEMVs"}
    assert orchestrator._precision(dict(target), "near-lossless", every) is None
    assert "not allowed" in orchestrator._precision(dict(target), "near-lossless")
    assert "needs --quality near-lossless" in orchestrator._precision(dict(target), "exact")

    spec = {
        "id": "mlp",
        "module_class": "Linear",
        "why": "w",
        "approach": "a",
        "backends": ["cuda"],
        "precision": "fp4_weights",
        "precision_why": "decode GEMVs at 90 % of the FP8 speed of light",
    }
    cases = [{"signature": "a0[1, 2048]:bfloat16", "count": 28}]
    capture = {"cases": cases, "tier": FP4, "precision": "fp4_weights"}
    args = (["cuda"], "python", "toolchain", 10, None)
    text = prompts.engineer_prompt(spec, capture, *args)
    assert prompts.reduced_precision(spec, capture) == "fp4_weights"
    cosine, rel_l2, _, _ = compare.NEAR_LOSSLESS_BOUNDS[FP4]
    for needle in (
        "# Precision: `fp4_weights`",
        f"the {FP4} tolerance tier",
        f"cosine >= {cosine:g}",
        f"relative L2 error <= {rel_l2:g}",
        "quantize_fp4",
        "fp4_error",
        "one e4m3 scale per 16",
        "cuda_fp4_gemv.py",
        "# Low-precision weights",
    ):
        assert needle in text, needle
    fp8 = prompts.engineer_prompt(
        {**spec, "precision": "fp8_weights"}, {**capture, "tier": "near-lossless"}, *args
    )
    assert "the near-lossless tolerance tier" in fp8 and "cosine >= 0.996" in fp8
    exact = prompts.engineer_prompt(spec, {"cases": cases}, *args)
    assert "# Precision" not in exact


def test_knowledge_and_example_exist():
    text = prompts.knowledge("low_precision.md")
    for needle in ("fp4_weights", "quantize_fp4", "near-lossless-fp4", "cuda_fp4_gemv.py"):
        assert needle in text, needle
    for name in selftest.FP4_EXAMPLES:
        source = (prompts.EXAMPLES_DIR / name).read_text()
        assert "def build(" in source and "quantize_fp4" in source and "load_inline" in source


# ---------------------------------------------------------------- capture, roofline, library


def test_capture_and_speed_of_light_count_fp4_weight_bytes(tmp_path):
    torch.manual_seed(0)
    lin = nn.Linear(256, 512, bias=False).to(torch.bfloat16)
    x = torch.randn(1, 256, dtype=torch.bfloat16)
    path = tmp_path / "lin.pt"
    capture_calls(lin, [((x,), {}, 28)], path, tier=FP4, precision="fp4_weights")
    cap = load_capture(path)
    assert cap["tier"] == FP4 and capture_precision(cap) == "fp4_weights"
    assert library.precision_of(cap) == "fp4_weights"

    io = 256 * 2 + 512 * 2  # x read, y written
    exact = roofline.count_case(lin, (x,), {})
    fp4 = roofline.count_case(lin, (x,), {}, precision="fp4_weights")
    # codes (half a byte each) + one e4m3 scale per 16 + the fp32 tensor scale
    assert fp4.min_bytes == 512 * 256 // 2 + 512 * 256 // 16 + 4 + io
    assert fp4.flops == exact.flops
    assert fp4.min_bytes / exact.min_bytes < 0.3


def test_evaluator_checks_fp4_in_its_tier(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(0)
    lin = nn.Linear(512, 256, bias=False)
    with torch.no_grad():
        lin.weight.normal_(0.0, 512**-0.5)
    cand = tmp_path / "fp4.py"
    cand.write_text(
        "import torch\n"
        "from kernel_agent.kernels.quant import dequantize_fp4, quantize_fp4\n\n\n"
        "class Fp4(torch.nn.Module):\n"
        "    def __init__(self, ref):\n"
        "        super().__init__()\n"
        "        self.w = dequantize_fp4(*quantize_fp4(ref.weight), torch.float32)\n\n"
        "    def forward(self, x):\n"
        "        return x @ self.w.T\n\n\n"
        "def build(reference):\n"
        "    return Fp4(reference)\n"
    )
    calls = [((torch.randn(4, 512),), {}, 1), ((torch.randn(2, 3, 512),), {}, 1)]
    fp4, fp8 = tmp_path / "fp4.pt", tmp_path / "fp8.pt"
    capture_calls(lin, calls, fp4, tier=FP4, precision="fp4_weights")
    capture_calls(lin, calls, fp8, tier="near-lossless", precision="fp8_weights")
    result = evaluate(fp4, cand, device="cpu")
    assert result["correct"] and result["tolerance_tier"] == FP4, result
    assert all(0.07 < c["max_rel_l2"] < 0.13 for c in result["cases"])
    rejected = evaluate(fp8, cand, device="cpu")
    assert rejected["status"] == "incorrect"


def test_loosening_the_fp4_bounds_is_an_integrity_violation(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(0)
    lin = nn.Linear(256, 128, bias=False)
    cand = tmp_path / "loose.py"
    cand.write_text(
        "import torch\n"
        "from kernel_agent.kernels import compare\n\n\n"
        "class Wrong(torch.nn.Module):\n"
        "    def __init__(self, ref):\n"
        "        super().__init__()\n"
        "        self.w = ref.weight.detach() * 1.5\n\n"
        "    def forward(self, x):\n"
        "        return x @ self.w.T\n\n\n"
        "def build(reference):\n"
        "    loose = (-1.0, 9.0, 9.0, (9.0, 9.0))\n"
        "    compare.NEAR_LOSSLESS_BOUNDS[compare.NEAR_LOSSLESS_FP4_TIER] = loose\n"
        "    return Wrong(reference)\n"
    )
    capture = tmp_path / "fp4.pt"
    calls = [((torch.randn(4, 256),), {}, 1)]
    capture_calls(lin, calls, capture, tier=FP4, precision="fp4_weights")
    saved = dict(compare.NEAR_LOSSLESS_BOUNDS)
    try:
        result = evaluate(capture, cand, device="cpu")
    finally:
        compare.NEAR_LOSSLESS_BOUNDS.clear()
        compare.NEAR_LOSSLESS_BOUNDS.update(saved)
        compare.TIER = compare.EXACT_TIER
    assert not result["correct"] and result["status"] == "integrity_violation", result


# ---------------------------------------------------------------- GPU: the example


def _gpu_ready():
    from kernel_agent import toolchain

    tc = toolchain.setup()
    if not selftest.fp8_supported(tc):
        pytest.skip("needs the cuda backend on sm_89+")


@pytest.mark.gpu
def test_fp4_gemv_matches_its_dequantised_weight():
    _gpu_ready()
    from importlib import util

    spec = util.spec_from_file_location("fp4_gemv", prompts.EXAMPLES_DIR / "cuda_fp4_gemv.py")
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(0)
    lin = nn.Linear(1024, 768, bias=True).cuda().to(torch.bfloat16)
    fp4 = module.build(lin)
    assert type(fp4).__name__ == "Fp4Linear" and fp4.quant_error["format"] == "nvfp4"
    w = quant.dequantize_fp4(fp4.weight_codes, fp4.weight_scales, fp4.weight_tensor_scale)
    for shape in ((1, 1024), (3, 1024), (2, 11, 1024)):  # GEMV, 3 rows, the fallback
        x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        want = (x.float() @ w.float().T + lin.bias.detach().float()).to(torch.bfloat16)
        with torch.no_grad():
            got = fp4(x)
        assert got.shape == want.shape and got.dtype == torch.bfloat16
        # the same weights: only the summation order and bf16 rounding differ (1 ulp)
        err = float((got.float() - want.float()).norm() / want.float().norm())
        assert err < 4e-3, err
        torch.testing.assert_close(got.float(), want.float(), rtol=1.6e-2, atol=1e-2)


@pytest.mark.gpu
@pytest.mark.parametrize("name", sorted(selftest.FP4_EXAMPLES))
def test_fp4_example_passes_its_tier_and_fails_the_fp8_tier(name, tmp_path):
    _gpu_ready()
    k, n, calls = selftest.FP4_EXAMPLES[name]
    fp4 = selftest.make_linear_capture(
        tmp_path / "fp4.pt", k, n, calls, tier=FP4, precision="fp4_weights"
    )
    fp8 = selftest.make_linear_capture(
        tmp_path / "fp8.pt", k, n, calls, tier="near-lossless", precision="fp8_weights"
    )
    example = prompts.EXAMPLES_DIR / name

    result = run_evaluation(fp4, example)  # subprocess + the checks outside it
    assert result["status"] == "ok" and result["correct"], result
    assert result["tolerance_tier"] == FP4 and result["parent_check"] == "ok"
    assert result["speedup"] > 1.0
    for case, (shape, _) in zip(result["cases"], calls, strict=True):
        assert 0.08 < case["max_rel_l2"] < 0.12 and case["min_cosine"] > 0.99
        rows = math.prod(shape)
        if "min_bytes" in case:  # with measured peaks: the weights count at 4 bits
            assert case["min_bytes"] == n * k // 2 + n * k // 16 + 4 + rows * (k + n) * 2

    rejected = evaluate(fp8, example)
    assert rejected["status"] == "incorrect", rejected
