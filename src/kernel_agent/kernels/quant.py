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
