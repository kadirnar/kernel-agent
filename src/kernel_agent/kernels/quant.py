"""Weight quantisation helpers for reduced-precision kernels.

A target whose spec says ``"precision": "fp8_weights"`` (planned in a ``--quality
near-lossless`` run, evaluated in the near-lossless tolerance tier of
:mod:`kernel_agent.kernels.compare`) may store its weights in FP8. The contract
(``agent/knowledge/low_precision.md``): quantise once in ``build()``, one scale per
output channel, activations stay bf16, accumulate in fp32, report the numerical error.

* :func:`quantize_fp8`: symmetric FP8 (e4m3 by default) with one fp32 scale per row of an
  ``nn.Linear`` weight ``[out_features, in_features]``: ``scale = amax(|row|) / 448``, so
  the largest element of every row maps to the largest finite e4m3 value.
* :func:`dequantize_fp8`: back to a floating dtype (the math of a fallback path, and the
  reference a kernel's output is checked against while debugging).
* :func:`fp8_error`: the error report of a quantised weight and, given activations, of the
  layer output; record it in ``NOTES.md`` next to the evaluator's per-case numbers.

The bundled examples (``agent/examples/cuda_fp8_gemv.py``, ``cuda_fp8_skinny_gemm.py``)
import these; a kernel that must load without kernel-agent can copy the few lines of
:func:`quantize_fp8`.

``"precision": "fp4_weights"`` (block-scaled FP4 weights, bf16 activations; its own
near-lossless tier, :data:`kernel_agent.kernels.compare.NEAR_LOSSLESS_FP4_TIER`):

* :func:`quantize_fp4`: e2m1 codes (two per byte) with a scale per block of consecutive
  elements of a row: ``nvfp4`` (default) one e4m3 scale per 16 and one fp32 scale per
  tensor, ``mxfp4`` (OCP MX) one power-of-two e8m0 scale per 32.
* :func:`dequantize_fp4`, :func:`fp4_error`: as for FP8 (``agent/examples/cuda_fp4_gemv.py``).

A target whose spec says ``"precision": "fp8_w8a8"`` (FP8 tensor-core math, for
compute-bound GEMMs) also quantises its activations, per token and per call:

* :func:`quantize_fp8_activations`: e4m3 codes and one fp32 scale per token (row of
  ``x.reshape(-1, in_features)``), the math a kernel computes on the fly.
* :func:`fp8_w8a8_linear`: ``x @ Wᵀ (+ bias)`` with W8A8 numerics, through
  ``torch._scaled_mm`` (cuBLASLt FP8 tensor cores) where it applies: the fallback path of
  a W8A8 kernel and its reference while debugging.
* :func:`fp8_w8a8_error`: :func:`fp8_error` plus the activations' quantisation error and
  the W8A8 layer output's error (``agent/examples/triton_fp8_w8a8_gemm.py``).

``"precision": "fp8_mx"`` (MXFP8 W8A8: e4m3 with one power-of-two ue8m0 scale per 32
elements along K on both operands, the block-scaled tensor cores of sm_100 / sm_120;
docs/FP8.md §3):

* :func:`quantize_mxfp8`: codes and e8m0 scales of activations or a weight, with the
  non-saturating scale rule ``2^ceil(log2(amax / 448))`` (:data:`MX_RULES`: the OCP
  reference rule ``2^(floor(log2 amax) - 8)`` saturates block maxima above 448 x scale);
  :func:`dequantize_mxfp8`; :func:`swizzle_mx_scales` / :func:`mx_scale_offset`: the 128 x
  4 blocked scale layout of cuBLASLt (``F.scaled_mm`` with ``SWIZZLE_32_4_4``).
* :func:`mxfp8_linear`: ``x @ Wᵀ (+ bias)`` with MXFP8 numerics, through ``F.scaled_mm``
  ``BlockWise1x32`` where it applies (the reference); :func:`mxfp8_error`: the error report.
* :func:`mxfp8_saturation`, :func:`mxfp8_scale_problem`: a quantiser's scales against the
  block maxima (the evaluator's scale-rule guard, :mod:`kernel_agent.kernels.scale_guard`);
  :func:`mxfp8_stress_input`: blocks whose maxima sit where the OCP rule saturates.
"""

from __future__ import annotations

import math
from typing import Any

import torch

#: Largest finite value of each FP8 format (OCP FP8: e4m3 "fn" has no infinities).
FP8_MAX = {torch.float8_e4m3fn: 448.0, torch.float8_e5m2: 57344.0}


def quantize_fp8(
    weight: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(q, scale)`` of a 2-D weight ``[out, in]``: ``q`` in ``dtype`` (same shape,
    contiguous) and ``scale`` (fp32 ``[out]``) with ``weight ≈ q * scale[:, None]``.

    Symmetric, per output channel (row), computed in fp32 with round-to-nearest-even.
    A row of zeros gets scale 1 (its codes are 0). Non-finite weights are refused."""
    if weight.dim() != 2:
        raise ValueError(f"expected a 2-D weight [out, in], got shape {tuple(weight.shape)}")
    if dtype not in FP8_MAX:
        raise ValueError(f"unsupported FP8 dtype {dtype}")
    w = weight.detach().float()
    if not bool(torch.isfinite(w).all()):
        raise ValueError("the weight has non-finite values")
    amax = w.abs().amax(dim=1)
    scale = torch.where(amax > 0, amax / FP8_MAX[dtype], torch.ones_like(amax))
    q = (w / scale[:, None]).clamp(-FP8_MAX[dtype], FP8_MAX[dtype]).to(dtype)
    return q.contiguous(), scale.contiguous()


def dequantize_fp8(
    q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """``q * scale[:, None]`` in ``dtype`` (product in fp32, one rounding to ``dtype``)."""
    return (q.float() * scale.float()[:, None]).to(dtype)


def _sig(value: float, digits: int = 4) -> float:
    return float(f"{value:.{digits}g}") if math.isfinite(value) else value


def fp8_error(
    weight: torch.Tensor,
    q: torch.Tensor,
    scale: torch.Tensor,
    x: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Numerical error of ``weight`` stored as ``(q, scale)`` (:func:`quantize_fp8`).

    * ``rel_l2``: ``‖W − Ŵ‖ / ‖W‖`` of the whole weight; ``worst_channel_rel_l2``: the
      same per output channel, the worst one (FP8 e4m3 per channel: about 0.02-0.03 on
      Gaussian-like rows).
    * ``underflow``: share of the non-zero weights that became 0 (a channel with an
      outlier gets a large scale, and its small weights flush to zero).
    * ``crest``: the largest ``amax / RMS`` of a channel (outliers: a high crest factor
      leaves few FP8 steps for the bulk of the row).
    * ``bytes``: weight bytes before (in ``weight``'s dtype) and after (codes + fp32
      scales).
    * with ``x`` (activations ``[..., in]``): ``output_rel_l2`` and ``output_cosine`` of
      ``x @ Ŵᵀ`` against ``x @ Wᵀ`` (both in fp32), the module-level error the evaluator's
      near-lossless tier bounds (cosine >= 0.996, relative L2 <= 0.08)."""
    w = weight.detach().float()
    deq = dequantize_fp8(q, scale, torch.float32)
    diff = w - deq
    norm = float(w.norm())
    row_norm = w.norm(dim=1)
    row_err = diff.norm(dim=1) / row_norm.clamp_min(1e-30)
    nonzero = w != 0
    flushed = nonzero & (deq == 0)
    rms = row_norm / math.sqrt(max(w.shape[1], 1))
    crest = w.abs().amax(dim=1) / rms.clamp_min(1e-30)
    report: dict[str, Any] = {
        "format": str(q.dtype).removeprefix("torch."),
        "granularity": "per output channel",
        "rel_l2": _sig(float(diff.norm()) / norm if norm > 0 else 0.0),
        "worst_channel_rel_l2": _sig(float(row_err[row_norm > 0].max()) if norm > 0 else 0.0),
        "underflow": _sig(float(flushed.sum()) / max(int(nonzero.sum()), 1)),
        "crest": _sig(float(crest[row_norm > 0].max()) if norm > 0 else 0.0),
        "bytes": {
            "before": weight.numel() * weight.element_size(),
            "after": q.numel() * q.element_size() + scale.numel() * scale.element_size(),
        },
    }
    if x is not None:
        a = x.detach().float().reshape(-1, w.shape[1]).to(w.device)
        ref, new = a @ w.T, a @ deq.T
        ref_norm, new_norm = float(ref.norm()), float(new.norm())
        cos = float((ref.flatten() @ new.flatten()) / (ref_norm * new_norm)) if ref_norm else 1.0
        report["output_rel_l2"] = _sig(float((ref - new).norm()) / ref_norm if ref_norm else 0.0)
        report["output_cosine"] = round(cos, 6)
    return report


# ------------------------------------------------------------------ FP4 (block-scaled e2m1)

#: Magnitudes of the FP4 e2m1 codes 0..7 (1 sign, 2 exponent and 1 mantissa bit; code
#: ``c | 8`` is ``-value[c]``). No infinities or NaN; the largest value is 6.
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FP4_MAX = 6.0
#: Block-scaled FP4 formats: elements per scale and the scale's dtype. ``nvfp4``: an e4m3
#: scale per 16 elements, times one fp32 scale per tensor; ``mxfp4`` (OCP MX): an e8m0
#: (power of two) scale per 32 elements.
FP4_FORMATS: dict[str, tuple[int, torch.dtype]] = {
    "nvfp4": (16, torch.float8_e4m3fn),
    "mxfp4": (32, torch.float8_e8m0fnu),
}
#: Midpoints between consecutive e2m1 magnitudes; a value exactly on one of the ``_TIE_UP``
#: midpoints rounds up to the even code (round to nearest even), on the others down.
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_TIE_UP = (0.75, 1.75, 3.5)


def _round_e2m1(v: torch.Tensor) -> torch.Tensor:
    """e2m1 codes (uint8 0..15, one per element) of fp32 values within ±6, rounded to
    nearest even."""
    mag = v.abs()
    idx = torch.bucketize(mag, torch.tensor(_E2M1_MIDPOINTS, device=v.device))
    for tie in _TIE_UP:
        idx += mag == tie
    negative = (v < 0) & (idx > 0)  # no negative zero
    return (idx + 8 * negative).to(torch.uint8)


def quantize_fp4(
    weight: torch.Tensor, fmt: str = "nvfp4"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(codes, scales, tensor_scale)`` of a 2-D weight ``[out, in]`` in block-scaled FP4
    (:data:`FP4_FORMATS`): ``codes`` uint8 ``[out, in / 2]`` (two e2m1 codes per byte, the
    even element in the low nibble; ``codes.view(torch.float4_e2m1fn_x2)`` for torch's FP4
    ops), ``scales`` ``[out, in / block]`` in the format's scale dtype and ``tensor_scale``
    a 0-d fp32 tensor (1 for ``mxfp4``), with ``weight[n, k] ≈ e2m1(code[n, k]) *
    scales[n, k // block] * tensor_scale``.

    ``nvfp4``: ``tensor_scale = amax(|W|) / (6 * 448)``, block scale ``amax(|block|) / 6 /
    tensor_scale`` rounded to e4m3, so the largest element of a block maps to about ±6.
    ``mxfp4``: block scale ``2^ceil(log2(amax(|block|) / 6))``, the smallest power of two
    that keeps the block within ±6 (the OCP MX reference conversion,
    ``2^(floor(log2 amax) - 2)``, saturates the largest elements of most blocks: on VoxCPM2
    that shrinks outputs by up to 13 %, where real activations meet the largest weights).
    Codes round to nearest even, saturated to ±6, computed in fp32. ``in`` must be a
    multiple of the block size. Non-finite weights are refused."""
    if weight.dim() != 2:
        raise ValueError(f"expected a 2-D weight [out, in], got shape {tuple(weight.shape)}")
    if fmt not in FP4_FORMATS:
        raise ValueError(f"unknown FP4 format {fmt!r} (one of {', '.join(FP4_FORMATS)})")
    block, scale_dtype = FP4_FORMATS[fmt]
    rows, cols = weight.shape
    if cols % block:
        raise ValueError(f"in_features {cols} is not a multiple of the {fmt} block ({block})")
    w = weight.detach().float()
    if not bool(torch.isfinite(w).all()):
        raise ValueError("the weight has non-finite values")
    blocks = w.reshape(rows, cols // block, block)
    bmax = blocks.abs().amax(dim=-1)
    if fmt == "nvfp4":
        amax = float(bmax.max()) if bmax.numel() else 0.0
        tensor_scale = torch.tensor(amax / (FP4_MAX * 448.0) if amax > 0 else 1.0)
        scales = (bmax / (FP4_MAX * float(tensor_scale))).clamp(max=448.0).to(scale_dtype)
        step = scales.float() * float(tensor_scale)  # 0: a zero block (or below e4m3's range)
    else:
        mantissa, exponent = torch.frexp(bmax / FP4_MAX)  # m * 2^e, m in [0.5, 1)
        ceil_log2 = exponent - (mantissa == 0.5).to(exponent.dtype)
        unbiased = torch.where(bmax > 0, ceil_log2, torch.full_like(exponent, -127))
        scales = (unbiased.clamp(-127, 127) + 127).to(torch.uint8).view(scale_dtype)
        tensor_scale = torch.tensor(1.0)
        step = scales.float()
    values = torch.where(
        step[..., None] > 0, blocks / step.clamp_min(1e-38)[..., None], torch.zeros_like(blocks)
    )
    codes = _round_e2m1(values.clamp(-FP4_MAX, FP4_MAX)).reshape(rows, cols)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.contiguous(), scales.contiguous(), tensor_scale.to(w.device)


def dequantize_fp4(
    codes: torch.Tensor,
    scales: torch.Tensor,
    tensor_scale: torch.Tensor | float,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """The weight of :func:`quantize_fp4`'s ``(codes, scales, tensor_scale)`` in ``dtype``
    (``e2m1 * scale * tensor_scale`` in fp32, one rounding to ``dtype``)."""
    rows = codes.shape[0]
    lut = torch.tensor(E2M1_VALUES, device=codes.device)
    lut = torch.cat([lut, -lut])
    nibbles = torch.stack([codes & 0xF, codes >> 4], dim=-1).reshape(rows, -1).long()
    values = lut[nibbles].reshape(rows, scales.shape[1], -1)
    weight = values * scales.float()[..., None] * float(tensor_scale)
    return weight.reshape(rows, -1).to(dtype)


def _error_metrics(w: torch.Tensor, deq: torch.Tensor, x: torch.Tensor | None) -> dict[str, Any]:
    """Weight and (given activations ``x``) output error of ``deq`` against ``w`` (both
    fp32 ``[out, in]``): the fields of :func:`fp8_error` without ``format`` and ``bytes``."""
    diff = w - deq
    norm = float(w.norm())
    row_norm = w.norm(dim=1)
    row_err = diff.norm(dim=1) / row_norm.clamp_min(1e-30)
    nonzero = w != 0
    rms = row_norm / math.sqrt(max(w.shape[1], 1))
    crest = w.abs().amax(dim=1) / rms.clamp_min(1e-30)
    report: dict[str, Any] = {
        "rel_l2": _sig(float(diff.norm()) / norm if norm > 0 else 0.0),
        "worst_channel_rel_l2": _sig(float(row_err[row_norm > 0].max()) if norm > 0 else 0.0),
        "underflow": _sig(float((nonzero & (deq == 0)).sum()) / max(int(nonzero.sum()), 1)),
        "crest": _sig(float(crest[row_norm > 0].max()) if norm > 0 else 0.0),
    }
    if x is not None:
        a = x.detach().float().reshape(-1, w.shape[1]).to(w.device)
        ref, new = a @ w.T, a @ deq.T
        ref_norm, new_norm = float(ref.norm()), float(new.norm())
        cos = float((ref.flatten() @ new.flatten()) / (ref_norm * new_norm)) if ref_norm else 1.0
        report["output_rel_l2"] = _sig(float((ref - new).norm()) / ref_norm if ref_norm else 0.0)
        report["output_cosine"] = round(cos, 6)
    return report


def fp4_error(
    weight: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    tensor_scale: torch.Tensor | float,
    x: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Numerical error of ``weight`` stored as :func:`quantize_fp4`'s ``(codes, scales,
    tensor_scale)``: the report of :func:`fp8_error` (``rel_l2``, ``worst_channel_rel_l2``,
    ``underflow``, ``crest``; ``bytes``: codes + block scales + the tensor scale; with
    ``x``: ``output_rel_l2`` and ``output_cosine``). On VoxCPM2's weights NVFP4 reaches a
    relative L2 error of 0.095 and MXFP4 0.116 (FP8 0.026), and ``underflow`` ~7 % (FP4
    flushes weights below a quarter of their block's step: normal, not a bug), hence the
    target's own tier (:data:`kernel_agent.kernels.compare.NEAR_LOSSLESS_FP4_TIER`)."""
    w = weight.detach().float()
    deq = dequantize_fp4(codes, scales, tensor_scale, torch.float32).to(w.device)
    fmt = next((f for f, (_, dtype) in FP4_FORMATS.items() if dtype == scales.dtype), "fp4")
    block = w.shape[1] // max(scales.shape[1], 1)
    kind = "e4m3 scale per {} + fp32 tensor scale" if fmt == "nvfp4" else "e8m0 scale per {}"
    return {
        "format": fmt,
        "granularity": kind.format(block),
        **_error_metrics(w, deq, x),
        "bytes": {
            "before": weight.numel() * weight.element_size(),
            "after": codes.numel() + scales.numel() * scales.element_size() + 4,
        },
    }


def quantize_fp8_activations(
    x: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(q, scale)`` of activations ``x [..., in]``, per token: ``q`` in ``dtype``
    (``[tokens, in]``, contiguous) and ``scale`` (fp32 ``[tokens]``) with
    ``x.reshape(-1, in) ≈ q * scale[:, None]``.

    Dynamic: computed for every call from the row's ``amax / 448`` (e4m3), in fp32, round to
    nearest even; a row of zeros gets scale 1. The reference math of the per-token
    quantisation a W8A8 kernel does in its prologue (or fuses into the producer of ``x``)."""
    if dtype not in FP8_MAX:
        raise ValueError(f"unsupported FP8 dtype {dtype}")
    a = x.detach().reshape(-1, x.shape[-1]).float()
    amax = a.abs().amax(dim=1)
    scale = torch.where(amax > 0, amax / FP8_MAX[dtype], torch.ones_like(amax))
    q = (a / scale[:, None]).clamp(-FP8_MAX[dtype], FP8_MAX[dtype]).to(dtype)
    return q.contiguous(), scale.contiguous()


def _scaled_mm_ok(x: torch.Tensor, q: torch.Tensor) -> bool:
    """Whether ``torch._scaled_mm`` takes these operands (FP8 row-wise scales: CUDA, sm_89+,
    bf16 out, ``in`` and ``out`` features multiples of 16)."""
    if not (x.is_cuda and q.is_cuda and x.dtype == torch.bfloat16):
        return False
    if q.dtype != torch.float8_e4m3fn or q.shape[0] % 16 or q.shape[1] % 16:
        return False
    return torch.cuda.get_device_capability(x.device) >= (8, 9)


def fp8_w8a8_linear(
    x: torch.Tensor,
    q: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x @ Wᵀ + bias`` with FP8 W8A8 numerics, in ``x``'s dtype and shape ``[..., out]``.

    ``(q, scale)``: the weight from :func:`quantize_fp8` (e4m3, one scale per output
    channel); ``x`` is quantised per token (:func:`quantize_fp8_activations`), the products
    accumulate in fp32 and both scales apply once per output. Through ``torch._scaled_mm``
    (cuBLASLt FP8 tensor cores) for bf16 CUDA inputs whose feature counts are multiples of
    16; otherwise the same math in fp32 (slow: a fallback and a reference)."""
    xq, xs = quantize_fp8_activations(x, q.dtype)
    if _scaled_mm_ok(x, q):
        y = torch._scaled_mm(
            xq,
            q.t(),  # column-major [in, out]: cuBLASLt's layout for the second operand
            scale_a=xs[:, None],
            scale_b=scale.float()[None, :],
            bias=None if bias is None else bias.to(torch.bfloat16),
            out_dtype=torch.bfloat16,
        )
    else:
        y = (xq.float() * xs[:, None]) @ (q.float() * scale.float()[:, None]).T
        if bias is not None:
            y = y + bias.float()
    return y.to(x.dtype).reshape(*x.shape[:-1], q.shape[0])


def fp8_w8a8_error(
    weight: torch.Tensor, q: torch.Tensor, scale: torch.Tensor, x: torch.Tensor
) -> dict[str, Any]:
    """Numerical error of a W8A8 layer: :func:`fp8_error` of the weight, plus

    * ``activation_rel_l2``: ``‖x − x̂‖ / ‖x‖`` of the per-token quantised activations;
      ``activation_crest``: the largest ``amax / RMS`` of a token (outlier channels set the
      token's scale and squeeze the rest of the row: ~30 on VoxCPM2's LM attention input);
      ``activation_underflow``: share of the non-zero activations that became 0;
    * ``output_rel_l2``, ``output_cosine`` and ``output_norm_ratio`` of the W8A8 output
      against ``x @ Wᵀ`` in fp32: the module-level error the near-lossless tier bounds
      (cosine >= 0.996, relative L2 <= 0.08, norm within ±2 %)."""
    report = fp8_error(weight, q, scale)
    w = weight.detach().float()
    a = x.detach().float().reshape(-1, w.shape[1]).to(w.device)
    xq, xs = quantize_fp8_activations(a, q.dtype)
    a_hat = xq.float() * xs[:, None]
    a_norm = float(a.norm())
    rms = a.pow(2).mean(dim=1).sqrt()
    crest = a.abs().amax(dim=1) / rms.clamp_min(1e-30)
    nonzero = a != 0
    ref = a @ w.T
    new = a_hat @ dequantize_fp8(q, scale, torch.float32).to(w.device).T
    ref_norm, new_norm = float(ref.norm()), float(new.norm())
    cos = float((ref.flatten() @ new.flatten()) / (ref_norm * new_norm)) if ref_norm else 1.0
    report.update(
        activations="e4m3 per token",
        activation_rel_l2=_sig(float((a - a_hat).norm()) / a_norm if a_norm > 0 else 0.0),
        activation_crest=_sig(float(crest[rms > 0].max()) if bool((rms > 0).any()) else 0.0),
        activation_underflow=_sig(
            float((nonzero & (a_hat == 0)).sum()) / max(int(nonzero.sum()), 1)
        ),
        output_rel_l2=_sig(float((ref - new).norm()) / ref_norm if ref_norm else 0.0),
        output_cosine=round(cos, 6),
        output_norm_ratio=round(new_norm / ref_norm, 5) if ref_norm else 1.0,
    )
    return report


# ------------------------------------------------------------------ MXFP8 (fp8_mx)

#: Elements per MXFP8 scale: one power-of-two (ue8m0) scale per 32 consecutive elements along
#: K (OCP MX v1.0), on both GEMM operands; the tensor core applies them (``mma.sync
#: kind::mxf8f6f4.block_scale``, cuBLASLt ``VEC32_UE8M0``).
MX_BLOCK = 32
#: The MXFP8 scale rules (:func:`mx_scale_exponents`). ``ceil``: ``2^ceil(log2(amax / 448))``,
#: the smallest power of two that keeps the block within ±448 (use it). ``floor``: the OCP
#: reference conversion ``2^(floor(log2 amax) - 8)``, which maps block maxima in [256, 512) x
#: scale onto e4m3 and saturates those above 448 x scale: on VoxCPM2's massive activations it
#: fails the near-lossless tier (LocDiT o_proj / down_proj norm off by 4.0 % / 2.0 %,
#: docs/FP8.md §3.2). Kept to show what the guard rejects.
MX_RULES = ("ceil", "floor")
#: A block saturates when its maximum exceeds ``448 x scale`` by more than this share (the
#: fp32 rounding of a kernel's scale computation; e4m3's own step at 448 is 32).
MX_SATURATION_SLACK = 2.0**-10
_E8M0_BIAS = 127
_E4M3 = torch.float8_e4m3fn


def mx_scale_exponents(amax: torch.Tensor, rule: str = "ceil") -> torch.Tensor:
    """Unbiased exponents (int32, clamped to e8m0's [-127, 127]) of the power-of-two scales of
    blocks with maxima ``amax`` (>= 0) under ``rule`` (:data:`MX_RULES`); a block of zeros
    gets -127. Exact: computed from the fp32 exponent bits, not ``log2``."""
    if rule not in MX_RULES:
        raise ValueError(f"unknown MXFP8 scale rule {rule!r} (one of {', '.join(MX_RULES)})")
    a = amax.float()
    if rule == "floor":
        _, exponent = torch.frexp(a)  # a = m * 2^e, m in [0.5, 1): floor(log2 a) = e - 1
        e = exponent - 1 - 8  # e4m3's largest power of two is 2^8
    else:
        mantissa, exponent = torch.frexp(a / FP8_MAX[_E4M3])
        e = exponent - (mantissa == 0.5).to(exponent.dtype)  # ceil(log2(a / 448))
        # a / 448 rounds in fp32: the smallest e with a <= 448 * 2^e (exact in fp32)
        e = e + (a > FP8_MAX[_E4M3] * torch.pow(2.0, e.double()).float()).to(e.dtype)
    e = torch.where(a > 0, e, torch.full_like(e, -_E8M0_BIAS))
    return e.clamp(-_E8M0_BIAS, _E8M0_BIAS).to(torch.int32)


def mx_scale_values(scales: torch.Tensor) -> torch.Tensor:
    """fp32 values of MXFP8 scales: ``float8_e8m0fnu``, ``uint8`` biased exponents (127 =
    2^0) or floating-point values (as they are)."""
    if scales.dtype == torch.uint8:
        scales = scales.view(torch.float8_e8m0fnu)
    return scales.float()


def quantize_mxfp8(x: torch.Tensor, rule: str = "ceil") -> tuple[torch.Tensor, torch.Tensor]:
    """``(codes, scales)`` of ``x [..., K]`` in MXFP8: ``codes`` e4m3 ``[rows, K]`` (the rows of
    ``x.reshape(-1, K)``: activations per call, or a weight ``[N, K]`` once in ``build()``)
    and ``scales`` ``float8_e8m0fnu`` ``[rows, K / 32]`` (unswizzled; :func:`swizzle_mx_scales`
    for ``F.scaled_mm``), with ``x[r, k] ≈ codes[r, k] * scales[r, k // 32]``.

    The scale of each block of 32 consecutive elements follows ``rule`` (:data:`MX_RULES`;
    default ``ceil``, ``2^ceil(log2(amax / 448))``: the block maximum lands in (224, 448],
    nothing saturates). Dynamic (every call), computed in fp32, codes round to nearest
    even. ``K`` must be a multiple of 32."""
    k = x.shape[-1]
    if k % MX_BLOCK:
        raise ValueError(f"the last dimension {k} is not a multiple of the MXFP8 block (32)")
    a = x.detach().reshape(-1, k).float()
    blocks = a.reshape(a.shape[0], k // MX_BLOCK, MX_BLOCK)
    e = mx_scale_exponents(blocks.abs().amax(dim=-1), rule)
    scales = (e + _E8M0_BIAS).to(torch.uint8).view(torch.float8_e8m0fnu)
    step = scales.float()  # exact powers of two (2^-127 for a block of zeros)
    limit = FP8_MAX[_E4M3]
    codes = (blocks / step[..., None]).clamp(-limit, limit).to(_E4M3)
    return codes.reshape(a.shape[0], k).contiguous(), scales.contiguous()


def dequantize_mxfp8(
    codes: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """``codes * scales`` per block of 32 (:func:`quantize_mxfp8`'s outputs) in ``dtype``,
    ``[rows, K]`` (product in fp32, one rounding to ``dtype``)."""
    rows, k = codes.shape
    values = codes.float().reshape(rows, k // MX_BLOCK, MX_BLOCK)
    step = mx_scale_values(scales).to(codes.device).reshape(rows, k // MX_BLOCK, 1)
    return (values * step).reshape(rows, k).to(dtype)


def mx_scale_offset(row: Any, col: Any, cols: int) -> Any:
    """Offset of scale ``[row, col]`` (of a ``[rows, cols]`` scale matrix, ``cols = K / 32``)
    in the swizzled layout of :func:`swizzle_mx_scales`: 128 x 4 blocks of 512 bytes, row
    major over the blocks, inside a block row ``r`` at ``(r % 32) * 16 + (r // 32) * 4``
    plus the column. Ints or integer tensors: a kernel that writes its scales in place."""
    blocks_per_row = -(-cols // 4)
    block = (row // 128) * blocks_per_row + col // 4
    return block * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + col % 4


def swizzle_mx_scales(scales: torch.Tensor) -> torch.Tensor:
    """``[rows, K / 32]`` MXFP8 scales (e8m0 or uint8) → the flat ``float8_e8m0fnu`` layout
    cuBLASLt reads (``F.scaled_mm(..., swizzle_a=SwizzleType.SWIZZLE_32_4_4)``): rows padded to
    a multiple of 128 and columns to 4 (padding: code 0), blocks of 128 x 4 one after the
    other (:func:`mx_scale_offset`)."""
    s = scales.view(torch.uint8) if scales.dtype == torch.float8_e8m0fnu else scales
    if s.dtype != torch.uint8 or s.dim() != 2:
        raise ValueError(f"expected [rows, K / 32] e8m0 / uint8 scales, got {s.dtype} {s.shape}")
    rows, cols = s.shape
    nrb, ncb = -(-rows // 128), -(-cols // 4)
    padded = s.new_zeros(nrb * 128, ncb * 4)
    padded[:rows, :cols] = s
    blocks = padded.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)  # [nrb, ncb, 128, 4]
    out = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1)  # row 32 * i + j -> (j, i)
    return out.contiguous().view(torch.float8_e8m0fnu)


def _mx_scaled_mm_ok(x: torch.Tensor, q: torch.Tensor) -> bool:
    """Whether ``F.scaled_mm`` runs these operands as MXFP8 (``BlockWise1x32``): CUDA bf16
    activations, an e4m3 weight with K a multiple of 128 (whole 128 x 4 scale blocks) and N of
    16, a GPU with block-scaled tensor cores (sm_100+: sm_120 measured) and a torch with
    ``F.scaled_mm``."""
    import torch.nn.functional as F

    if not (x.is_cuda and q.is_cuda and x.dtype == torch.bfloat16 and q.dtype == _E4M3):
        return False
    if q.shape[1] % 128 or q.shape[0] % 16 or not hasattr(F, "scaled_mm"):
        return False
    return torch.cuda.get_device_capability(x.device) >= (10, 0)


def mxfp8_linear(
    x: torch.Tensor,
    q: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    rule: str = "ceil",
) -> torch.Tensor:
    """``x @ Wᵀ + bias`` with MXFP8 numerics, in ``x``'s dtype and shape ``[..., out]``.

    ``(q, scales)``: the weight from :func:`quantize_mxfp8` (e4m3 ``[N, K]``, e8m0 ``[N,
    K / 32]``, unswizzled); ``x`` is quantised per block of 32 with ``rule`` on every call,
    the products accumulate in fp32 with both block scales applied by the tensor core.
    Through ``F.scaled_mm`` (``ScalingType.BlockWise1x32``, ``SwizzleType.SWIZZLE_32_4_4``:
    cuBLASLt ``VEC32_UE8M0``) where it applies (bias added to its bf16 output); otherwise the
    same math in fp32 (slow: a fallback and a reference)."""
    xq, xs = quantize_mxfp8(x, rule)
    if _mx_scaled_mm_ok(x, q):
        import torch.nn.functional as F

        blockwise, swizzle = F.ScalingType.BlockWise1x32, F.SwizzleType.SWIZZLE_32_4_4
        y = F.scaled_mm(
            xq,
            q.t(),  # column-major [K, N]: cuBLASLt's layout for the second operand
            scale_a=swizzle_mx_scales(xs),
            scale_recipe_a=blockwise,
            scale_b=swizzle_mx_scales(scales),
            scale_recipe_b=blockwise,
            swizzle_a=swizzle,
            swizzle_b=swizzle,
            output_dtype=torch.bfloat16,
        )
        if bias is not None:
            y = y + bias.to(y.dtype)
    else:
        w = dequantize_mxfp8(q, scales, torch.float32).to(xq.device)
        y = dequantize_mxfp8(xq, xs, torch.float32) @ w.T
        if bias is not None:
            y = y + bias.float()
    return y.to(x.dtype).reshape(*x.shape[:-1], q.shape[0])


def mxfp8_saturation(x: torch.Tensor, scales: torch.Tensor) -> dict[str, Any]:
    """How MXFP8 ``scales`` fit the block maxima of ``x [..., K]``: ``scales`` ``[rows, K /
    32]`` for the rows of ``x.reshape(-1, K)`` (e8m0, uint8 biased exponents or values).

    * ``blocks``; ``saturated``: blocks whose maximum exceeds ``448 x scale`` (by more than
      :data:`MX_SATURATION_SLACK`): the quantiser clamps their largest elements; ``share``;
    * ``worst_ratio``: the largest ``block max / scale`` (<= 448 without saturation);
    * ``coarser``: blocks whose scale is above the ceil rule's (no saturation, one bit of
      precision or more given away);
    * ``not_pow2``: scales that are not powers of two (not ue8m0).

    ValueError: ``scales`` of another shape."""
    k = x.shape[-1]
    a = x.detach().reshape(-1, k).float()
    rows, nblocks = a.shape[0], k // MX_BLOCK
    if k % MX_BLOCK or tuple(scales.shape) != (rows, nblocks):
        raise ValueError(
            f"expected scales [{rows}, {nblocks}] (rows of x, K / 32, unswizzled) for x "
            f"{tuple(x.shape)}, got {tuple(scales.shape)}"
        )
    amax = a.reshape(rows, nblocks, MX_BLOCK).abs().amax(dim=-1)
    step = mx_scale_values(scales).to(a.device).reshape(rows, nblocks)
    mantissa, _ = torch.frexp(step)
    not_pow2 = int(((mantissa != 0.5) & (step > 0)).sum())
    ratio = torch.where(amax > 0, amax / step.clamp_min(1e-38), torch.zeros_like(amax))
    limit = FP8_MAX[_E4M3] * (1 + MX_SATURATION_SLACK)
    saturated = int(((ratio > limit) | ((step <= 0) & (amax > 0))).sum())
    ceil = mx_scale_exponents(amax, "ceil").double()
    coarser = int(((step.double() > torch.pow(2.0, ceil)) & (amax > 0)).sum())
    blocks = rows * nblocks
    return {
        "blocks": blocks,
        "saturated": saturated,
        "share": _sig(saturated / blocks) if blocks else 0.0,
        "worst_ratio": _sig(float(ratio.max())) if blocks else 0.0,
        "coarser": coarser,
        "not_pow2": not_pow2,
    }


def mxfp8_scale_problem(x: torch.Tensor, scales: torch.Tensor) -> str | None:
    """Why ``scales`` are not a valid MXFP8 quantisation of ``x`` (:func:`mxfp8_saturation`):
    a block maximum above ``448 x scale`` (a saturating scale rule) or scales that are not
    powers of two; None when they are."""
    found = mxfp8_saturation(x, scales)
    if found["saturated"]:
        return (
            f"{found['saturated']} of {found['blocks']} MXFP8 blocks ({found['share']:.1%}) "
            f"saturate: block maximum up to {found['worst_ratio']:.4g} x scale, above e4m3's "
            "448 (their largest elements are clamped). Use the scale 2^ceil(log2(amax / 448)); "
            "the OCP rule 2^(floor(log2 amax) - 8) saturates maxima in (448, 512) x scale"
        )
    if found["not_pow2"]:
        return (
            f"{found['not_pow2']} of {found['blocks']} MXFP8 scales are not powers of two "
            "(ue8m0 holds an exponent only)"
        )
    return None


def mxfp8_stress_input(
    shape: tuple[int, ...] | torch.Size,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> torch.Tensor:
    """Activations of ``shape`` (``[..., K]``, K a multiple of 32) whose blocks of 32 hold
    Gaussian values with their maximum at ±1.9 x 2^e (e from -8 to 8 across blocks): where
    the OCP floor rule saturates every block (maximum 486 x scale) and the ceil rule none (243
    x scale). The scale-rule guard runs a candidate's quantiser on it."""
    k = int(shape[-1])
    if k % MX_BLOCK:
        raise ValueError(f"the last dimension {k} is not a multiple of the MXFP8 block (32)")
    rows = math.prod(int(n) for n in shape[:-1])
    gen = torch.Generator().manual_seed(seed)
    v = torch.randn(rows, k // MX_BLOCK, MX_BLOCK, generator=gen)
    peak = v.abs().argmax(dim=-1, keepdim=True)
    v = v / v.abs().gather(-1, peak)  # block maximum ±1, at the largest Gaussian value
    e = torch.arange(rows * (k // MX_BLOCK)).reshape(rows, k // MX_BLOCK, 1) % 17 - 8
    v = v * 1.9 * torch.pow(2.0, e.float())
    return v.reshape(*shape).to(dtype=dtype, device=device)


def mxfp8_error(
    weight: torch.Tensor,
    q: torch.Tensor,
    scales: torch.Tensor,
    x: torch.Tensor | None = None,
    rule: str = "ceil",
) -> dict[str, Any]:
    """Numerical error of an MXFP8 layer (weight ``(q, scales)`` from :func:`quantize_mxfp8`,
    activations ``x`` quantised with ``rule``): the weight report of :func:`fp8_error`
    (``rel_l2``, ``worst_channel_rel_l2``, ``underflow``, ``crest``), plus with ``x``

    * ``activation_rel_l2`` and ``activation_saturation`` (:func:`mxfp8_saturation` of the
      activation scales: ``share`` of saturated blocks, ``worst_ratio``);
    * ``output_rel_l2``, ``output_cosine`` and ``output_norm_ratio`` of the MXFP8 output
      against ``x @ Wᵀ`` in fp32: the module-level error the near-lossless tier bounds
      (cosine >= 0.996, relative L2 <= 0.08, norm within ±2 %). With the ceil rule MXFP8
      has per-token W8A8's error (docs/FP8.md §3.2: 0.0205 vs 0.0204 on a DiT layer)."""
    w = weight.detach().float()
    deq = dequantize_mxfp8(q, scales, torch.float32).to(w.device)
    report: dict[str, Any] = {
        "format": "mxfp8",
        "granularity": f"e4m3 + e8m0 scale per {MX_BLOCK} along K, both operands ({rule} rule)",
        **_error_metrics(w, deq, None),
        "bytes": {
            "before": weight.numel() * weight.element_size(),
            "after": q.numel() + scales.numel(),
        },
    }
    if x is None:
        return report
    a = x.detach().float().reshape(-1, w.shape[1]).to(w.device)
    xq, xs = quantize_mxfp8(a, rule)
    a_hat = dequantize_mxfp8(xq, xs, torch.float32)
    a_norm = float(a.norm())
    ref, new = a @ w.T, a_hat @ deq.T
    ref_norm, new_norm = float(ref.norm()), float(new.norm())
    cos = float((ref.flatten() @ new.flatten()) / (ref_norm * new_norm)) if ref_norm else 1.0
    saturation = mxfp8_saturation(a, xs)
    return {
        **report,
        "activations": f"e4m3 + e8m0 per {MX_BLOCK} ({rule} rule)",
        "activation_rel_l2": _sig(float((a - a_hat).norm()) / a_norm if a_norm > 0 else 0.0),
        "activation_saturation": {k: saturation[k] for k in ("share", "worst_ratio")},
        "output_rel_l2": _sig(float((ref - new).norm()) / ref_norm if ref_norm else 0.0),
        "output_cosine": round(cos, 6),
        "output_norm_ratio": round(new_norm / ref_norm, 5) if ref_norm else 1.0,
    }
