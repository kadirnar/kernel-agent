"""W4A4's norm bias (#233 follow-up 8): e2m1's grid shrinks a GEMM's output along the exact
output by ~1 %; the opt-in ``unbiased`` correction (``quant.fp4_bias_correction``: ``sum x^2 /
sum x x^`` per token in the epilogue, per output channel in the weight's tensor scale) removes
it without touching codes or block scales.

CPU, deterministic: the factor against a float64 Python reference (zero rows: 1), the weight
quantiser's per-channel tensor scale and its dequantisation, the reference GEMM's epilogue
bit for bit, the error report's in-phase ``output_gain``, and the shrink removed on Gaussian
activations (measured on Qwen3-0.6B: docs/research-scripts/w4a4-233/norm_bias.py)."""

import pytest
import torch

from kernel_agent.kernels import quant


def _data(rows: int = 64, k: int = 256, n: int = 128, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, k, generator=gen).to(torch.bfloat16)
    w = (torch.randn(n, k, generator=gen) * k**-0.5).to(torch.bfloat16)
    return x, w


def _python_factor(x: torch.Tensor, xh: torch.Tensor) -> list[float]:
    out = []
    for row, qrow in zip(x.double().tolist(), xh.double().tolist(), strict=True):
        num = sum(v * v for v in row)
        den = sum(v * q for v, q in zip(row, qrow, strict=True))
        out.append(num / den if den > 0 else 1.0)
    return out


def test_the_bias_factor_is_sum_x2_over_sum_x_xhat():
    x, w = _data(16, 64)
    x[3] = 0  # a row of zeros: 1
    codes, scales, outer = quant.quantize_fp4_activations(x)
    c = quant.fp4_bias_correction(x, codes, scales, outer)
    xh = quant.fp4_values(codes, scales) * outer[:, None]
    assert c.dtype == torch.float32 and c.shape == (16,) and c[3] == 1.0
    assert c.tolist() == pytest.approx(_python_factor(x.float(), xh), rel=1e-6)
    assert float(c[c != 1].min()) > 0.9 and float(c.max()) < 1.1  # 4 blocks: a few %
    # one outer scale for every row (a weight's tensor scale), or none
    wc, ws, ts = quant.quantize_fp4(w)
    cw = quant.fp4_bias_correction(w, wc, ws, ts)
    want = _python_factor(w.float(), quant.dequantize_fp4(wc, ws, ts, torch.float32))
    assert cw.tolist() == pytest.approx(want, rel=1e-6)
    plain = quant.fp4_bias_correction(w, wc, ws)  # outer None: 1 (x^ without the tensor scale)
    assert plain.tolist() == pytest.approx((cw * float(ts)).tolist(), rel=1e-6)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
def test_the_unbiased_weight_quantiser_moves_only_the_tensor_scale(fmt):
    _, w = _data(1, 256, 96)
    codes, scales, ts = quant.quantize_fp4(w, fmt)
    ucodes, uscales, uts = quant.quantize_fp4(w, fmt, unbiased=True)
    assert torch.equal(codes, ucodes) and torch.equal(
        scales.view(torch.uint8), uscales.view(torch.uint8)
    )
    assert ts.numel() == 1 and uts.shape == (96,) and uts.dtype == torch.float32
    c = quant.fp4_bias_correction(w, codes, scales, ts)
    assert torch.equal(uts, ts * c)  # the tensor scale times each channel's factor
    deq = quant.dequantize_fp4(ucodes, uscales, uts, torch.float32)
    want = quant.dequantize_fp4(codes, scales, 1.0, torch.float32) * uts[:, None]
    assert torch.equal(deq, want)
    report = quant.fp4_error(w, ucodes, uscales, uts)
    assert "per channel (unbiased)" in report["granularity"]
    assert report["bytes"]["after"] == codes.numel() + scales.numel() + 4 * 96
    assert (
        quant.fp4_error(w, codes, scales, ts)["bytes"]["after"]
        == codes.numel() + scales.numel() + 4
    )


@pytest.mark.parametrize("granularity", ["token", "tensor"])
def test_the_unbiased_gemm_is_the_corrected_epilogue_bit_for_bit(granularity):
    x, w = _data()
    codes, scales, ts = quant.quantize_fp4(w, unbiased=True)
    bias = torch.linspace(-1, 1, 128).to(torch.bfloat16)
    y = quant.fp4_w4a4_linear(x, codes, scales, ts, bias, granularity=granularity, unbiased=True)
    xq, xs, outer = quant.quantize_fp4_activations(x, granularity=granularity)
    c = quant.fp4_bias_correction(x, xq, xs, outer)
    acc = quant.fp4_values(xq, xs) @ quant.fp4_values(codes, scales).T
    want = (acc * ((outer * c)[:, None] * ts) + bias.float()).to(torch.bfloat16)
    assert torch.equal(y, want)
    # the plain path is unchanged by the option's existence
    plain = quant.fp4_w4a4_linear(x, *quant.quantize_fp4(w), bias, granularity=granularity)
    assert not torch.equal(plain, y)


def test_unbiased_removes_the_in_phase_shrink():
    x, w = _data(256, 1024, 512, seed=3)
    base = quant.fp4_w4a4_error(w, *quant.quantize_fp4(w), x)
    act = quant.fp4_w4a4_error(w, *quant.quantize_fp4(w), x, unbiased=True)
    both = quant.fp4_w4a4_error(w, *quant.quantize_fp4(w, unbiased=True), x, unbiased=True)
    assert "unbiased" in both["activations"] and "unbiased" not in base["activations"]
    # e2m1 shrinks the output's in-phase component ~1 % (Gaussian rows: 0.990); the factors
    # take it back while the error and the direction stay as they were
    assert base["output_gain"] < 0.995
    assert abs(act["output_gain"] - 1) < abs(base["output_gain"] - 1)
    assert abs(both["output_gain"] - 1) < 0.2 * abs(base["output_gain"] - 1)
    assert both["output_rel_l2"] < 1.01 * base["output_rel_l2"]
    assert both["output_cosine"] > base["output_cosine"] - 1e-4
    # the norm: the noise is no longer hidden by the shrink
    assert both["output_norm_ratio"] > base["output_norm_ratio"]
