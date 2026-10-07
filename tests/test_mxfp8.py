"""MXFP8 W8A8 precision class (#144): the ceil-rule reference quantiser, the swizzled scale
layout, the saturation guard (``kernels/scale_guard.py``) in the evaluator, the ``fp8_mx``
precision through compare / precisions / planner / engineer prompt / pivots / ceilings /
speed of light, and the bundled example on the CPU (its GPU path: ``gpu``, not run when
written)."""

import math
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from kernel_agent import pivot, precisions, selftest
from kernel_agent.agent import prompts
from kernel_agent.kernels import compare, quant, roofline, scale_guard
from kernel_agent.kernels.evaluate import evaluate, run_evaluation
from kernel_agent.profiling import ceilings
from kernel_agent.profiling.capture import capture_calls

NEAR = "near-lossless"
EXAMPLE = "triton_mxfp8_gemm.py"


# ---------------------------------------------------------------- the reference quantiser


def test_scale_rules_are_exact():
    amax = torch.tensor([0.0, 1e-40, 1.0, 224.0, 448.0, 449.0, 15.2, 3.0e38])
    ceil = quant.mx_scale_exponents(amax, "ceil")
    floor = quant.mx_scale_exponents(amax, "floor")
    assert ceil.dtype == torch.int32
    assert ceil.tolist() == [-127, -127, -8, -1, 0, 1, -4, 120]
    assert floor.tolist() == [-127, -127, -8, -1, 0, 0, -5, 119]  # 449 and 15.2 saturate
    # the ceil rule is the smallest power of two that keeps every block within 448
    gen = torch.Generator().manual_seed(0)
    a = torch.exp(torch.randn(4096, generator=gen) * 20).clamp(1e-30, 1e30)
    e = quant.mx_scale_exponents(a, "ceil").double()
    assert bool((a.double() <= 448 * torch.pow(2.0, e)).all())
    assert bool((a.double() > 448 * torch.pow(2.0, e - 1)).all())
    assert torch.equal(
        quant.mx_scale_exponents(a, "floor"), torch.floor(torch.log2(a.double())).int() - 8
    )
    with pytest.raises(ValueError, match="unknown MXFP8 scale rule"):
        quant.mx_scale_exponents(a, "round")


def test_quantize_and_dequantize():
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(3, 7, 256, generator=gen).to(torch.bfloat16)
    x[0, 0, :32] = 0  # a block of zeros
    x[1, 2, 5] = 3000.0  # an outlier spoils its block only
    codes, scales = quant.quantize_mxfp8(x)
    assert codes.shape == (21, 256) and codes.dtype == torch.float8_e4m3fn
    assert scales.shape == (21, 8) and scales.dtype == torch.float8_e8m0fnu
    assert scales.view(torch.uint8)[0, 0] == 0 and not codes[0, :32].float().any()
    deq = quant.dequantize_mxfp8(codes, scales, torch.float32)
    rows = x.float().reshape(21, 256)
    rel = (deq - rows).norm(dim=1) / rows.norm(dim=1)
    assert float(rel.max()) < 0.05  # e4m3 per block of 32
    assert codes.float().abs().max() <= 448
    assert quant.mxfp8_saturation(x, scales)["saturated"] == 0
    other = rows.clone()
    other[7, 5] = 0  # the outlier's row without it: its other blocks are as before
    assert float(((deq - rows).abs())[7, 32:].max()) < 0.2
    with pytest.raises(ValueError, match="multiple of the MXFP8 block"):
        quant.quantize_mxfp8(torch.zeros(2, 48))


def _to_blocked(m: torch.Tensor) -> torch.Tensor:
    """The research scripts' layout (docs/research-scripts/fp8-sm120/probe_support.py), on
    which F.scaled_mm's MXFP8 result equalled the fp32 math of the codes."""
    rows, cols = m.shape
    nrb, ncb = -(-rows // 128), -(-cols // 4)
    p = torch.zeros(nrb * 128, ncb * 4, dtype=m.dtype)
    p[:rows, :cols] = m
    blocks = p.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


@pytest.mark.parametrize(("rows", "cols"), [(352, 32), (22, 8), (200, 12), (128, 4)])
def test_swizzled_scale_layout(rows, cols):
    gen = torch.Generator().manual_seed(rows)
    s = torch.randint(1, 255, (rows, cols), generator=gen, dtype=torch.uint8)
    flat = quant.swizzle_mx_scales(s.view(torch.float8_e8m0fnu))
    assert flat.dtype == torch.float8_e8m0fnu and flat.dim() == 1
    assert flat.numel() == math.ceil(rows / 128) * 128 * math.ceil(cols / 4) * 4
    u = flat.view(torch.uint8)
    assert torch.equal(u, _to_blocked(s))
    r, c = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
    assert torch.equal(u[quant.mx_scale_offset(r, c, cols)], s)
    assert int((u != 0).sum()) == rows * cols  # padding: code 0
    assert quant.mx_scale_offset(33, 5, 8) == 512 + 1 * 16 + 1 * 4 + 1
    with pytest.raises(ValueError, match="e8m0 / uint8"):
        quant.swizzle_mx_scales(torch.zeros(4, 4))


def test_linear_reference_and_error_report():
    gen = torch.Generator().manual_seed(2)
    w = torch.randn(64, 256, generator=gen) * 256**-0.5
    bias = torch.randn(64, generator=gen)
    x = torch.randn(4, 11, 256, generator=gen)
    q, s = quant.quantize_mxfp8(w)
    y = quant.mxfp8_linear(x, q, s, bias)
    xq, xs = quant.quantize_mxfp8(x)
    expected = (
        quant.dequantize_mxfp8(xq, xs, torch.float32)
        @ quant.dequantize_mxfp8(q, s, torch.float32).T
        + bias
    )
    assert y.shape == (4, 11, 64) and torch.allclose(y.reshape(44, 64), expected)
    ref = x @ w.T + bias
    assert 0.01 < float((y - ref).norm() / ref.norm()) < 0.05  # like per-token W8A8

    report = quant.mxfp8_error(w, q, s, x)
    assert report["format"] == "mxfp8" and "ceil rule" in report["granularity"]
    assert report["bytes"]["after"] == 64 * 256 + 64 * 8
    assert 0.015 < report["activation_rel_l2"] < 0.04
    assert report["activation_saturation"]["share"] == 0
    assert report["activation_saturation"]["worst_ratio"] <= 448
    assert report["output_cosine"] > 0.998 and abs(report["output_norm_ratio"] - 1) < 0.01
    weight_only = quant.mxfp8_error(w, q, s)
    assert "output_rel_l2" not in weight_only and weight_only["rel_l2"] == report["rel_l2"]
    floor = quant.mxfp8_error(w, q, s, quant.mxfp8_stress_input((44, 256)), rule="floor")
    assert floor["activation_saturation"]["share"] == 1.0


def test_saturation_check_and_stress_input():
    x = quant.mxfp8_stress_input((2, 33, 128), dtype=torch.float32, seed=3)
    assert x.shape == (2, 33, 128) and x.dtype == torch.float32
    _, ceil = quant.quantize_mxfp8(x, "ceil")
    _, floor = quant.quantize_mxfp8(x, "floor")
    assert quant.mxfp8_scale_problem(x, ceil) is None
    good = quant.mxfp8_saturation(x, ceil)
    assert good["saturated"] == 0 and good["worst_ratio"] == pytest.approx(243.2)
    bad = quant.mxfp8_saturation(x, floor)
    assert bad["saturated"] == bad["blocks"] == 66 * 4 and bad["worst_ratio"] > 480
    problem = quant.mxfp8_scale_problem(x, floor)
    assert problem is not None and "264 of 264" in problem and "2^ceil" in problem
    # uint8 biased exponents and fp32 values are scales too; twice the scale is coarser
    assert quant.mxfp8_saturation(x, ceil.view(torch.uint8))["saturated"] == 0
    doubled = quant.mxfp8_saturation(x, ceil.float() * 2)
    assert doubled["saturated"] == 0 and doubled["coarser"] == 264
    assert "not powers of two" in quant.mxfp8_scale_problem(x, ceil.float() * 1.5)
    with pytest.raises(ValueError, match="unswizzled"):
        quant.mxfp8_saturation(x, quant.swizzle_mx_scales(ceil))


# ---------------------------------------------------------------- the guard


def _cases(x: torch.Tensor) -> list[dict]:
    return [
        {"args": (torch.ones(3),), "kwargs": {}, "count": 0},
        {"args": (x,), "kwargs": {}, "count": 40},
    ]


def test_scale_guard_checks_the_candidates_quantiser():
    gen = torch.Generator().manual_seed(4)
    x = torch.randn(2, 11, 256, generator=gen)
    x[..., 7] *= 300  # a massive-activation channel
    cases = _cases(x)
    assert scale_guard.activation(cases) is x

    def module(fn):
        return types.SimpleNamespace(quantize_activations=fn)

    ceil = scale_guard.check(module(lambda t: quant.quantize_mxfp8(t, "ceil")), cases)
    assert ceil["ok"] and ceil["checked"] and ceil["stress"]["saturated"] == 0
    assert ceil["captured"]["blocks"] == 22 * 8 and ceil["outliers"]["channel_ratio"] > 50

    floor = scale_guard.check(module(lambda t: quant.quantize_mxfp8(t, "floor")), cases)
    assert not floor["ok"] and "saturate" in floor["error"]
    assert floor["error"].startswith("quantize_activations on the")

    def broken(t):
        raise RuntimeError("no kernel for this shape")

    raised = scale_guard.check(module(broken), cases)
    assert not raised["ok"] and "RuntimeError: no kernel" in raised["error"]
    swizzled = scale_guard.check(
        module(lambda t: (lambda c, s: (c, quant.swizzle_mx_scales(s)))(*quant.quantize_mxfp8(t))),
        cases,
    )
    assert not swizzled["ok"] and "unswizzled" in swizzled["error"]

    missing = scale_guard.check(types.SimpleNamespace(), cases)
    assert missing["ok"] and not missing["checked"] and "not checked" in missing["note"]
    assert "outliers" in missing
    nothing = scale_guard.check(module(quant.quantize_mxfp8), [{"args": (torch.ones(5),)}])
    assert nothing["ok"] and not nothing["checked"]


def test_patching_the_guard_is_an_integrity_violation(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    near = _linear_capture(tmp_path)
    path = tmp_path / "loosened.py"
    path.write_text(
        "from kernel_agent.kernels import quant\n"
        "from kernel_agent.kernels.quant import mxfp8_linear, quantize_mxfp8\n"
        "import torch\n\n"
        "def quantize_activations(x):\n"
        "    return quantize_mxfp8(x, 'floor')\n\n"
        "class M(torch.nn.Module):\n"
        "    def __init__(self, ref):\n"
        "        super().__init__()\n"
        "        self.q, self.s = quantize_mxfp8(ref.weight)\n"
        "    def forward(self, x):\n"
        "        return mxfp8_linear(x, self.q, self.s, rule='floor')\n\n"
        "def build(reference):\n"
        "    quant.MX_SATURATION_SLACK = 1.0\n"
        "    return M(reference)\n"
    )
    saved = quant.MX_SATURATION_SLACK
    try:
        result = evaluate(near, path, device="cpu")
    finally:
        quant.MX_SATURATION_SLACK = saved
        compare.TIER = compare.EXACT_TIER
    assert result["status"] == "integrity_violation", result


# ---------------------------------------------------------------- the precision class


def test_precision_class_tier_and_allowed_by_default():
    assert "fp8_mx" in compare.REDUCED_PRECISIONS and "fp8_mx" in compare.PRECISIONS
    assert compare.tier_for(NEAR, "fp8_mx") == compare.NEAR_LOSSLESS_TIER  # W8A8's bounds
    assert compare.tier_for("exact", "fp8_mx") == compare.EXACT_TIER
    assert "fp8_mx" in precisions.default(NEAR) and "fp8_mx" not in precisions.FOUR_BIT
    assert precisions.parse("fp8_mx") == ["exact", "fp8_mx"]
    assert precisions.refusal("fp8_mx", ("exact", "fp8_w8a8")) is not None
    enum = prompts.PLAN_SCHEMA["properties"]["targets"]["items"]["properties"]["precision"]
    assert "fp8_mx" in enum["enum"]
    assert (
        "fp8_mx"
        not in prompts.plan_schema(("exact", "fp8_w8a8"))["properties"]["targets"]["items"][
            "properties"
        ]["precision"]["enum"]
    )


def test_planner_policy_and_engineer_contract():
    near = prompts.precision_policy(NEAR)
    assert '`precision: "fp8_mx"`' in near and "N >= ~2560" in near and "M >= ~64" in near
    assert "no split-K" in near and "tensor-wise W8A8 with split-K" in near
    assert "static (offline-calibrated) activation scale is allowed" in near
    assert "redrawn-input check" in near and "outlier channels" in near
    without = prompts.precision_policy(NEAR, ("exact", "fp8_weights"))
    assert '`precision: "fp8_mx"`' not in without and "offline-calibrated" not in without
    assert "`fp8_mx`" in prompts.precision_policy("exact")  # refused

    target = {
        "id": "dit",
        "module_class": "Linear",
        "why": "w",
        "approach": "a",
        "backends": ["triton"],
        "precision": "fp8_mx",
        "precision_why": "M=352, N=8192, compute bound",
    }
    capture = {"cases": [], "tier": NEAR, "precision": "fp8_mx"}
    text = prompts.engineer_prompt(target, capture, ["triton"], "python", "toolchain", 10, None)
    assert prompts.reduced_precision(target, capture) == "fp8_mx"
    for needle in (
        "# Precision: `fp8_mx`",
        "MXFP8 W8A8",
        "2^ceil(log2(amax / 448))",
        "quantize_activations(x) -> (codes, scales)",
        "BlockWise1x32",
        "SWIZZLE_32_4_4",
        "triton_mxfp8_gemm.py",
        "mxfp8_error(weight, q, scales, x)",
        "near-lossless tolerance tier",
        "## MXFP8 W8A8",
    ):
        assert needle in text, needle

    block = prompts._pivot_block({"id": "dit", "precision": "fp8_w8a8"}, Path("/r/p.json"))
    assert "`fp8_mx`" in block and "wide outputs" in block


def test_pivot_to_fp8_mx():
    spec = {"id": "dit", "module_class": "Layer", "why": "w", "approach": "a", "backends": []}
    proposal = {"precision": "fp8_mx", "precision_why": "M=352, N=8192: 239 vs 200 TFLOP/s"}
    assert pivot.check(spec, proposal, quality=NEAR, taken=set()) is None
    assert pivot.pivot_spec(spec, proposal)["id"] == "dit__fp8_mx"
    assert pivot.family("dit__fp8_mx") == "dit"
    no_mx = ("exact", "fp8_w8a8")
    assert "not allowed" in pivot.check(spec, proposal, quality=NEAR, taken=set(), allowed=no_mx)


def test_ceilings_column_and_speed_of_light():
    assert ceilings.target_precision("fp8_mx") is ceilings.PRECISIONS["mxfp8"]
    assert ceilings.target_columns(("exact", "fp8_mx")) == ["exact", "mxfp8"]
    row = {"flops": {"bfloat16": int(80e12)}, "weight_elems": 10**8, "weight_bytes": 2 * 10**8}
    row |= {"io_bytes": 0, "calls": 1}
    peaks = {"tflops": {"bfloat16": 100.0, roofline.FP8: 330.0, roofline.MXFP8: 320.0}}
    peaks["dram_gbps"] = 1000.0
    mx = ceilings.floor(row, ceilings.PRECISIONS["mxfp8"], peaks)
    assert mx is not None and mx["ms"] == pytest.approx(250.0) and mx["bound"] == "compute"
    assert mx["memory_ms"] == pytest.approx(10**8 * (1 + 1 / 32) / 1e9 * 1000 / 1000)
    old = {"tflops": {"bfloat16": 100.0, roofline.FP8: 330.0}, "dram_gbps": 1000.0}
    assert ceilings.floor(row, ceilings.PRECISIONS["mxfp8"], old) is None  # unknown, no ratio

    module = nn.Linear(256, 512, bias=False).to(torch.bfloat16)
    x = torch.randn(352, 256, dtype=torch.bfloat16)
    cost = roofline.count_case(module, (x,), {}, precision="fp8_mx")
    w8a8 = roofline.count_case(module, (x,), {}, precision="fp8_w8a8")
    assert cost.flops == {roofline.MXFP8: 2 * 352 * 256 * 512}
    weights = 256 * 512
    assert cost.min_bytes - w8a8.min_bytes == (weights // 32 + 4) - 4 * 512  # e8m0 per 32
    assert roofline.sol_time(cost, old)["peak_missing"] == roofline.MXFP8
    assert "peak_missing" not in roofline.sol_time(cost, peaks)
    assert roofline.PEAKS_VERSION >= 3


# ---------------------------------------------------------------- the example on the CPU


def _linear_capture(tmp_path, tier=NEAR, precision="fp8_mx"):
    torch.manual_seed(0)
    module = nn.Linear(256, 512, bias=False).to(torch.bfloat16)
    with torch.no_grad():
        module.weight.normal_(0, 256**-0.5)
    calls = [
        ((torch.randn(16, 11, 256, dtype=torch.bfloat16),), {}, 540),
        ((torch.randn(4, 11, 256, dtype=torch.bfloat16),), {}, 0),
    ]
    path = tmp_path / f"{precision}_{tier}.pt"
    capture_calls(module, calls, path, tier=tier, precision=precision)
    return path


def test_example_passes_the_tier_and_its_floor_rule_fails_the_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    example = prompts.EXAMPLES_DIR / EXAMPLE
    source = example.read_text()
    for needle in ("def build(", "BlockWise1x32", "SWIZZLE_32_4_4", "def quantize_activations"):
        assert needle in source, needle
    assert EXAMPLE in selftest.MX_EXAMPLES

    near = _linear_capture(tmp_path)
    result = evaluate(near, example, device="cpu")  # the reference math on the CPU
    assert result["correct"] and result["tolerance_tier"] == NEAR, result
    assert result["scale_rule"]["ok"] and result["scale_rule"]["checked"]
    for case in result["cases"]:
        assert 0.01 < case["max_rel_l2"] < 0.06 and case["min_cosine"] > 0.998
    exact = evaluate(_linear_capture(tmp_path, tier=None, precision=None), example, device="cpu")
    assert exact["status"] == "incorrect" and "scale_rule" not in exact

    floor = evaluate(near, selftest.floor_rule_variant(tmp_path, EXAMPLE), device="cpu")
    assert floor["status"] == "incorrect" and floor["stage"] == "scale_rule", floor
    assert "saturate" in floor["error"]


def test_knowledge_and_readme():
    text = prompts.knowledge("low_precision.md")
    for needle in (
        "## MXFP8 W8A8 (`precision: fp8_mx`)",
        "2^ceil(log2(amax / 448))",
        "quantize_activations",
        "split-K",
        "redrawn-input check",
        "triton_mxfp8_gemm.py",
    ):
        assert needle in text, needle
    readme = (prompts.AGENT_DIR.parents[2] / "README.md").read_text()
    assert "fp8_mx" in readme and "scale_rule" in readme


# ---------------------------------------------------------------- GPU (not run when written)


@pytest.mark.gpu
@pytest.mark.parametrize("name", sorted(selftest.MX_EXAMPLES))
def test_mxfp8_example_on_the_gpu(name, tmp_path):
    from kernel_agent import toolchain

    if not selftest.mxfp8_supported(toolchain.setup()):
        pytest.skip("needs the triton backend on sm_100+ and F.scaled_mm")
    k, n, calls = selftest.MX_EXAMPLES[name]
    near = selftest.make_linear_capture(
        tmp_path / "near.pt", k, n, calls, tier=NEAR, precision="fp8_mx"
    )
    result = run_evaluation(near, prompts.EXAMPLES_DIR / name)
    assert result["status"] == "ok" and result["correct"], result
    assert result["scale_rule"]["checked"] and result["parent_check"] == "ok"
    for case in result["cases"]:
        assert 0.02 < case["max_rel_l2"] < 0.05 and case["min_cosine"] > 0.998
    floor = run_evaluation(near, selftest.floor_rule_variant(tmp_path, name), quick=True)
    assert floor["stage"] == "scale_rule", floor

    # the Triton quantiser equals the reference (codes and scales), swizzled in place
    from kernel_agent.kernels.evaluate import load_candidate_module
    from kernel_agent.profiling.capture import load_capture

    x = load_capture(near)["cases"][0]["args"][0]
    example = load_candidate_module(prompts.EXAMPLES_DIR / name)
    codes, scales = example.quantize_activations(x)
    ref_codes, ref_scales = quant.quantize_mxfp8(x)
    assert torch.equal(scales.view(torch.uint8), ref_scales.view(torch.uint8))
    assert torch.equal(codes.view(torch.uint8), ref_codes.view(torch.uint8))
    _, flat = example._quantize(x.reshape(-1, k), swizzle=True)
    assert torch.equal(
        flat.view(torch.uint8), quant.swizzle_mx_scales(ref_scales).view(torch.uint8)
    )
