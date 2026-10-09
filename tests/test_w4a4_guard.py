"""The scale-rule guard of ``fp4_w4a4`` candidates (#233, like ``fp8_mx``'s, #144).

CPU, deterministic: :func:`quant.fp4_scale_check` accepts the reference NVFP4 quantiser (per
token and per call) and MXFP4's ceil rule, bit patterns and values alike, and rejects block
scales rounded down, subnormal scales flushed to zero, an outer scale too small for the
row's blocks and MXFP4's OCP floor rule, each with its reason; the stress input puts block
maxima where those rules show; ``scale_guard.check`` runs a candidate's
``quantize_activations`` on the captured and the stress input (missing: unchecked, as for
``fp8_mx``; swizzled scales: refused with the unswizzled form); the evaluator rejects a
saturating W4A4 candidate at ``stage: scale_rule`` and passes the reference math and the
bundled Triton example (its CPU path) in ``near-lossless-fp4a``."""

import types

import pytest
import torch
from torch import nn

from kernel_agent.agent import prompts
from kernel_agent.kernels import compare, quant, scale_guard
from kernel_agent.kernels.evaluate import evaluate
from kernel_agent.profiling.capture import capture_calls

W4A4 = "fp4_w4a4"
FP4A = "near-lossless-fp4a"


def _activations(rows: int = 64, k: int = 256, seed: int = 0) -> torch.Tensor:
    """Rows from 1e-3 to 1e3 (subnormal and zero block scales in the rows with an outlier)
    and one massive activation, as bf16."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, k, generator=gen) * torch.logspace(-3, 3, rows)[:, None]
    x[3, 7] = 500.0
    return x.to(torch.bfloat16)


def _round_down(v: torch.Tensor) -> torch.Tensor:
    """e4m3 of ``v`` (0 <= v) rounded toward zero (truncated), clamped at 448."""
    rn = v.clamp(max=448.0).to(torch.float8_e4m3fn)
    bits = rn.view(torch.uint8)
    down = (rn.float() > v) & (bits > 0)
    return torch.where(down, (bits - 1).view(torch.float8_e4m3fn).float(), rn.float()).to(
        torch.float8_e4m3fn
    )


def _nvfp4(x: torch.Tensor, rule: str = "nearest", granularity: str = "token", small=1.0):
    """``(codes, scales, outer)`` of an NVFP4 quantiser with block scales rounded ``nearest``
    (the reference), ``down`` or with subnormal ones flushed to zero (``ftz``); ``small``
    scales the outer scale (below 1: block scales clamp at 448)."""
    a = x.reshape(-1, x.shape[-1]).float()
    rows, k = a.shape
    blocks = a.reshape(rows, k // 16, 16)
    amax = a.abs().amax(dim=1)
    if granularity == "tensor":
        amax = amax.amax().expand(rows)
    outer = torch.where(amax > 0, amax * quant.NVFP4_OUTER_STEP * small, torch.ones_like(amax))
    ratio = blocks.abs().amax(dim=-1) / (outer * quant.FP4_MAX)[:, None]
    if rule == "down":
        scales = _round_down(ratio)
    else:
        scales = ratio.clamp(max=448.0).to(torch.float8_e4m3fn)
        if rule == "ftz":
            flushed = scales.float() < quant.E4M3_MIN_NORMAL
            scales = torch.where(flushed, 0.0, scales.float()).to(torch.float8_e4m3fn)
    return quant._fp4_codes(blocks, scales.float() * outer[:, None]), scales, outer


def _mx_floor(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP4 scales of the OCP reference rule ``2^(floor(log2 bmax) - 2)`` (e8m0)."""
    a = x.reshape(-1, x.shape[-1]).float()
    bmax = a.reshape(a.shape[0], -1, 32).abs().amax(dim=-1)
    e = torch.frexp(bmax)[1] - 1 - 2
    e = torch.where(bmax > 0, e, torch.full_like(e, -127)).clamp(-127, 127)
    return torch.zeros(1), (e + 127).to(torch.uint8).view(torch.float8_e8m0fnu)


# ------------------------------------------------------------------ the rule (quant.py)


@pytest.mark.parametrize("granularity", ["token", "tensor"])
def test_the_reference_passes_and_saturating_nvfp4_rules_fail(granularity):
    stress = quant.fp4_stress_input((4, 16, 256))
    for x in (_activations(), stress):
        _, scales, outer = quant.quantize_fp4_activations(x, "nvfp4", granularity)
        found = quant.fp4_scale_check(x, scales, outer)
        assert quant.fp4_scale_problem(x, scales, outer) is None, found
        assert found["format"] == "nvfp4" and found["saturated"] == found["underflow"] == 0
        assert found["coarser"] == 0 and found["blocks"] == x.numel() // 16
        # e4m3 bits and values are scales too; a plain per-call outer scale as one value
        assert quant.fp4_scale_problem(x, scales.view(torch.uint8), outer, "nvfp4") is None
        assert quant.fp4_scale_problem(x, scales.float(), outer, "nvfp4") is None
        assert (
            _nvfp4(x, granularity=granularity)[1].view(torch.uint8).equal(scales.view(torch.uint8))
        )  # the helper's nearest rule is the reference
        down = quant.fp4_scale_problem(x, *_nvfp4(x, "down", granularity)[1:])
        assert down is not None and "saturate beyond the rounding" in down and "never down" in down
        clamped = quant.fp4_scale_problem(x, *_nvfp4(x, "nearest", granularity, small=0.8)[1:])
        assert clamped is not None and "clamps at 448" in clamped
    # the captured-like input has subnormal scales only in its small rows (per token: none)
    ftz = quant.fp4_scale_check(stress, *_nvfp4(stress, "ftz", granularity)[1:])
    assert ftz["underflow"] > 0 and ftz["saturated"] == 0
    assert "scale 0" in quant.fp4_scale_problem(stress, *_nvfp4(stress, "ftz", granularity)[1:])
    # the reference's own saturation (rounded to nearest, subnormal scales) is not a problem
    assert (
        quant.fp4_saturation(_activations(), *quant.quantize_fp4_activations(_activations())[1:])[
            "saturated"
        ]
        > 0
    )


def test_mxfp4_ceil_rule_passes_and_the_ocp_floor_rule_fails():
    stress = quant.fp4_stress_input((4, 16, 256), "mxfp4")
    for x in (_activations(), stress):
        _, scales, outer = quant.quantize_fp4_activations(x, "mxfp4")
        assert quant.fp4_scale_problem(x, scales, outer) is None
        assert quant.fp4_scale_problem(x, scales.view(torch.uint8), None, "mxfp4") is None
        problem = quant.fp4_scale_problem(x, *_mx_floor(x)[1:])
        assert problem is not None and "2^ceil(log2(amax / 6))" in problem
    assert quant.fp4_scale_check(stress, _mx_floor(stress)[1])["share"] == 1.0  # every block
    found = quant.fp4_scale_check(stress, quant.quantize_fp4_activations(stress, "mxfp4")[1])
    assert found["worst_ratio"] == pytest.approx(3.8, rel=1e-2)  # 1.9 x 2^e / 2^(e - 1)
    _, scales, _ = quant.quantize_fp4_activations(stress, "mxfp4")
    assert "not powers of two" in quant.fp4_scale_problem(stress, scales.float() * 1.5)


def test_scale_shapes_and_non_finite_scales_are_named():
    x = _activations()
    _, scales, outer = quant.quantize_fp4_activations(x)
    with pytest.raises(ValueError, match="unswizzled"):
        quant.fp4_scale_check(x, quant.swizzle_fp4_scales(scales), outer)
    with pytest.raises(ValueError, match="outer scale per row"):
        quant.fp4_scale_check(x, scales, outer[:5])
    assert quant.fp4_format_of(256, scales.view(torch.uint8)) == "nvfp4"
    assert quant.fp4_format_of(256, torch.zeros(3, 8)) == "mxfp4"
    nan = scales.clone()
    nan.view(torch.uint8)[2, 3] = 0x7F  # e4m3 NaN
    assert "not finite" in quant.fp4_scale_problem(x, nan, outer)
    assert "not finite" in quant.fp4_scale_problem(x, scales, torch.zeros_like(outer))
    assert "not e4m3 values" in quant.fp4_scale_problem(x, scales.float() * 1.03, outer, "nvfp4")


def test_the_stress_input_puts_block_scales_where_rounding_decides():
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        x = quant.fp4_stress_input((3, 5, 512), dtype=dtype, seed=2)
        assert x.dtype == dtype and x.shape == (3, 5, 512)
        a = x.reshape(-1, 512).float()
        assert torch.equal(a.abs().amax(dim=1), torch.full((15,), 64.0))  # per token = per call
        ratio = a.reshape(15, 32, 16).abs().amax(dim=-1) * 448.0 / 64.0  # bmax / (outer x 6)
        targets = torch.tensor(quant.NVFP4_STRESS_RATIOS)
        nearest = (ratio[:, 1:, None] / targets - 1).abs().amin(dim=-1)
        assert float(nearest.max()) < 5e-3  # every block within 0.5 % of its target
        _, scales, outer = quant.quantize_fp4_activations(x)
        assert quant.fp4_scale_problem(x, scales, outer) is None
        assert quant.fp4_scale_check(x, *_nvfp4(x, "down")[1:])["share"] > 0.9
    with pytest.raises(ValueError, match="multiple"):
        quant.fp4_stress_input((4, 40))
    with pytest.raises(ValueError, match="unknown FP4 format"):
        quant.fp4_stress_input((4, 64), "int4")


# ------------------------------------------------------------------ the guard (scale_guard.py)


def _module(fn) -> types.SimpleNamespace:
    return types.SimpleNamespace(quantize_activations=fn)


def test_the_guard_checks_a_w4a4_candidates_quantiser():
    x = _activations(2 * 11, 256).reshape(2, 11, 256)
    cases = [
        {"args": (torch.ones(3),), "kwargs": {}, "count": 0},
        {"args": (x,), "kwargs": {}, "count": 40},
    ]
    assert scale_guard.activation(cases, 16) is x

    good = scale_guard.check(_module(quant.quantize_fp4_activations), cases, W4A4)
    assert good["ok"] and good["checked"] and good["stress"]["format"] == "nvfp4"
    assert good["captured"]["blocks"] == 22 * 16 and good["stress"]["saturated"] == 0
    assert good["outliers"]["crest"] > 10
    per_call = scale_guard.check(
        _module(lambda t: quant.quantize_fp4_activations(t, granularity="tensor")), cases, W4A4
    )
    assert per_call["ok"] and per_call["checked"]
    mx = scale_guard.check(
        _module(lambda t: quant.quantize_fp4_activations(t, "mxfp4")[:2]), cases, W4A4
    )
    assert mx["ok"] and mx["stress"]["format"] == "mxfp4"  # outer left out: 1

    down = scale_guard.check(_module(lambda t: _nvfp4(t, "down")), cases, W4A4)
    assert not down["ok"] and down["error"].startswith("quantize_activations on the captured")
    assert "saturate beyond the rounding" in down["error"]
    floor = scale_guard.check(_module(lambda t: (*_mx_floor(t), None)), cases, W4A4)
    assert not floor["ok"] and "OCP rule" in floor["error"]

    # a flush to zero that the captured rows never show: the stress input names it
    plain = torch.randn(2, 11, 256, generator=torch.Generator().manual_seed(1))
    plain_cases = [{"args": (plain.to(torch.bfloat16),), "kwargs": {}, "count": 5}]
    ftz = scale_guard.check(_module(lambda t: _nvfp4(t, "ftz")), plain_cases, W4A4)
    assert not ftz["ok"] and ftz["input"] == "stress" and ftz["captured"]["underflow"] == 0
    assert "scale 0" in ftz["error"]

    swizzled = scale_guard.check(
        _module(
            lambda t: (lambda c, s, o: (c, quant.swizzle_fp4_scales(s), o))(
                *quant.quantize_fp4_activations(t)
            )
        ),
        cases,
        W4A4,
    )
    assert not swizzled["ok"] and "unswizzled" in swizzled["error"]

    # a quantiser that rotates first names what it quantised (else: its scales do not fit x)
    def rotated(t, named=True):
        r = quant.hadamard_rotate(t, 16)
        return (*quant.quantize_fp4_activations(r), *([r] if named else []))

    assert scale_guard.check(_module(rotated), cases, W4A4)["ok"]
    unnamed = scale_guard.check(_module(lambda t: rotated(t, named=False)), cases, W4A4)
    assert not unnamed["ok"] and "saturate" in unnamed["error"]

    def broken(t):
        raise RuntimeError("no kernel for this shape")

    assert "RuntimeError: no kernel" in scale_guard.check(_module(broken), cases, W4A4)["error"]
    missing = scale_guard.check(types.SimpleNamespace(), cases, W4A4)
    assert missing["ok"] and not missing["checked"] and "(codes, scales, outer)" in missing["note"]
    assert "W4A4 scale rule is not checked" in missing["note"]
    nothing = scale_guard.check(
        _module(quant.quantize_fp4_activations), [{"args": (torch.ones(4, 8),)}], W4A4
    )
    assert nothing["ok"] and not nothing["checked"] and "multiple of 16" in nothing["note"]


# ------------------------------------------------------------------ through the evaluator (CPU)

CANDIDATE = """
import torch
from kernel_agent.kernels import quant

RULE = "nearest"


def _e4m3(v):
    rn = v.clamp(max=448.0).to(torch.float8_e4m3fn)
    if RULE != "down":
        return rn
    bits = rn.view(torch.uint8)
    down = (rn.float() > v) & (bits > 0)
    return torch.where(down, (bits - 1).view(torch.float8_e4m3fn).float(), rn.float()).to(
        torch.float8_e4m3fn
    )


def quantize_activations(x):
    a = x.reshape(-1, x.shape[-1]).float()
    rows, k = a.shape
    blocks = a.reshape(rows, k // 16, 16)
    amax = a.abs().amax(dim=1)
    outer = torch.where(amax > 0, amax * quant.NVFP4_OUTER_STEP, torch.ones_like(amax))
    scales = _e4m3(blocks.abs().amax(dim=-1) / (outer * quant.FP4_MAX)[:, None])
    return quant._fp4_codes(blocks, scales.float() * outer[:, None]), scales, outer


class W4A4(torch.nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.codes, self.scales, self.ts = quant.quantize_fp4(reference.weight)

    def forward(self, x):
        xq, xs, outer = quantize_activations(x)
        acc = quant.fp4_values(xq, xs) @ quant.fp4_values(self.codes, self.scales).T
        y = acc * (outer[:, None] * float(self.ts))
        return y.to(x.dtype).reshape(*x.shape[:-1], -1)


def build(reference):
    return W4A4(reference)
"""


def _capture(tmp_path, tier=FP4A, precision=W4A4):
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


def test_the_evaluator_rejects_a_w4a4_scale_rounded_down(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    capture = _capture(tmp_path)
    good, down, bare = (tmp_path / f"{n}.py" for n in ("nearest", "down", "bare"))
    good.write_text(CANDIDATE)
    down.write_text(CANDIDATE.replace('RULE = "nearest"', 'RULE = "down"'))
    bare.write_text(CANDIDATE.replace("def quantize_activations(", "def _quantize_activations("))
    bare.write_text(
        bare.read_text().replace("= quantize_activations(x)", "= _quantize_activations(x)")
    )
    try:
        result = evaluate(capture, good, device="cpu")
        assert result["correct"] and result["tolerance_tier"] == FP4A, result
        assert result["scale_rule"]["ok"] and result["scale_rule"]["checked"]
        assert result["scale_rule"]["captured"]["format"] == "nvfp4"
        rejected = evaluate(capture, down, device="cpu")
        assert rejected["status"] == "incorrect" and rejected["stage"] == "scale_rule", rejected
        assert rejected["failed_check"] == {"check": "scale_rule", "input": "captured"}
        assert "saturate beyond the rounding" in rejected["error"]
        unchecked = evaluate(capture, bare, device="cpu")
        assert unchecked["correct"] and not unchecked["scale_rule"]["checked"]
        # the bundled Triton example (its reference path on the CPU) passes the guard
        example = evaluate(
            capture, prompts.EXAMPLES_DIR / "triton_nvfp4_w4a4_gemm.py", device="cpu"
        )
        assert example["correct"] and example["scale_rule"]["checked"], example
        # an 8-bit capture has no W4A4 guard
        exact = evaluate(_capture(tmp_path, compare.NEAR_LOSSLESS_TIER, "fp8_w8a8"), good)
        assert "scale_rule" not in exact
    finally:
        compare.TIER = compare.EXACT_TIER
