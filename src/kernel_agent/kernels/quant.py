"""Weight quantisation helpers for reduced-precision kernels.

A target whose spec says ``"precision": "fp8_weights"`` (planned in a ``--quality
near-lossless`` run, evaluated in the near-lossless tolerance tier of
:mod:`kernel_agent.kernels.compare`) may store its weights in FP8. The contract
(skill ``fp8-weights``): quantise once in ``build()``, one scale per
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

INT8 (issue #178: the 8-bit classes of GPUs without FP8 tensor cores, sm_75 / sm_80 / sm_86,
and an option everywhere else): symmetric int8 codes in [-127, 127] (-128 unused), one fp32
scale ``amax / 127`` (computed as ``amax * INT8_STEP``) per output channel of a weight and
per token of the activations, codes ``round(x / scale)`` to nearest even.

* ``"precision": "int8_weights"`` (weight-only, bf16 activations, dequantised in the kernel):
  :func:`quantize_int8`, :func:`dequantize_int8`, :func:`int8_weights_linear` (the reference),
  :func:`int8_error`.
* ``"precision": "int8_w8a8"`` (INT8 tensor-core math: IMMA, int32 accumulation, both scales
  in the epilogue): :func:`quantize_int8_activations` (dynamic, per token, every call),
  :func:`int8_matmul` (the exact int32 products: ``torch._int_mm``, cuBLASLt's IMMA kernels,
  where it applies), :func:`int8_w8a8_linear` (the reference and fallback),
  :func:`int8_w8a8_error`, and SmoothQuant-style migration of activation outliers into the
  weights (:func:`smoothquant_factors`: per input channel ``s``, activations ``x / s``,
  weight columns ``W * s``; the same product in exact arithmetic).

``"precision": "fp4_w4a4"`` (W4A4, issue #233; opt-in: block-scaled FP4 weights and
activations on the FP4 tensor cores of sm_100 / sm_120, fp32 accumulation; the
``near-lossless-fp4a`` tier): weights from :func:`quantize_fp4`, activations per call
:func:`quantize_fp4_activations` (NVFP4: an e4m3 scale per 16 and an fp32 outer scale per
token; MXFP4: e8m0 per 32), :func:`fp4_w4a4_linear` (``F.scaled_mm`` NVFP4 where it applies:
the reference and fallback), :func:`swizzle_fp4_scales`, :func:`fp4_values`,
:func:`fp4_saturation`, :func:`fp4_w4a4_error`, an optional block Hadamard rotation
(:func:`hadamard_rotate`) and the per-layer sensitivity probe (:func:`fp4_w4a4_sensitivity`:
which ``nn.Linear`` to keep in FP8).
"""

from __future__ import annotations

import math
from collections.abc import Callable
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
    # comparisons with the midpoints, not a bucketize table: no host-to-device copy, so the
    # reference runs inside a CUDA graph (a W4A4 fallback path quantises on every call)
    idx = torch.zeros(mag.shape, dtype=torch.int64, device=mag.device)
    for mid in _E2M1_MIDPOINTS:
        idx += (mag >= mid) if mid in _TIE_UP else (mag > mid)
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


# ------------------------------------------------------------------ INT8 (int8_weights, int8_w8a8)

#: Largest int8 code of the symmetric scheme: codes in [-127, 127], so -128 (no positive
#: twin) is never produced and negation stays exact.
INT8_MAX = 127.0
#: ``1 / 127`` in fp32: a scale is ``amax * INT8_STEP`` (a product, the same on every device
#: and in a kernel's ``amax * (1.0f / 127.0f)``; torch divides by a Python scalar through its
#: reciprocal on the GPU but not on the CPU).
INT8_STEP = float(torch.tensor(1.0 / 127.0, dtype=torch.float32))
#: K per exact fp32 partial product of :func:`int8_matmul`'s fallback: 1024 x 127 x 127 is
#: below 2^24, so every partial sum of int8 products is an exact fp32 integer.
_INT8_EXACT_K = 1024


def _int8_codes(v: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """int8 codes of fp32 ``v`` [rows, K] with one scale per row: ``round(v / scale)`` (an
    IEEE division: ``__fdiv_rn`` / ``tl.div_rn`` in a kernel) to nearest even, clamped to
    ±127."""
    return torch.round(v / scale[:, None]).clamp(-INT8_MAX, INT8_MAX).to(torch.int8)


def _row_scales(v: torch.Tensor) -> torch.Tensor:
    """``amax(|row|) * (1 / 127)`` per row of fp32 ``v`` (1 for a row of zeros)."""
    amax = v.abs().amax(dim=1)
    return torch.where(amax > 0, amax * INT8_STEP, torch.ones_like(amax))


def _smooth_of(smooth: torch.Tensor | None, k: int, device: torch.device) -> torch.Tensor | None:
    if smooth is None:
        return None
    s = smooth.detach().float().to(device).reshape(-1)
    if s.numel() != k:
        raise ValueError(f"expected {k} smoothing factors (one per input channel), got {s.numel()}")
    if not bool(torch.isfinite(s).all()) or not bool((s > 0).all()):
        raise ValueError("smoothing factors must be finite and positive")
    return s


def quantize_int8(
    weight: torch.Tensor, smooth: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(q, scale)`` of a 2-D weight ``[out, in]``: ``q`` int8 (same shape, contiguous) and
    ``scale`` (fp32 ``[out]``) with ``weight ≈ q * scale[:, None]``.

    Symmetric, per output channel (row), computed in fp32: ``scale = amax(|row|) / 127``,
    codes rounded to nearest even and clamped to ±127; a row of zeros gets scale 1.
    ``smooth`` (SmoothQuant, :func:`smoothquant_factors`): quantise ``weight * smooth[None,
    :]`` instead, for activations divided by ``smooth``. Non-finite weights are refused."""
    if weight.dim() != 2:
        raise ValueError(f"expected a 2-D weight [out, in], got shape {tuple(weight.shape)}")
    w = weight.detach().float()
    if not bool(torch.isfinite(w).all()):
        raise ValueError("the weight has non-finite values")
    s = _smooth_of(smooth, w.shape[1], w.device)
    if s is not None:
        w = w * s[None, :]
    scale = _row_scales(w)
    return _int8_codes(w, scale).contiguous(), scale.contiguous()


def dequantize_int8(
    q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """``q * scale[:, None]`` in ``dtype`` (product in fp32, one rounding to ``dtype``)."""
    return (q.float() * scale.float().to(q.device)[:, None]).to(dtype)


def quantize_int8_activations(
    x: torch.Tensor, smooth: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(q, scale)`` of activations ``x [..., in]``, per token: ``q`` int8 ``[tokens, in]``
    (contiguous) and ``scale`` (fp32 ``[tokens]``) with ``x.reshape(-1, in) ≈ q *
    scale[:, None]`` (with ``smooth``: ``≈ (q * scale[:, None]) * smooth``).

    Dynamic: computed for every call from the row's ``amax / 127`` in fp32 (of ``x /
    smooth`` with SmoothQuant factors), codes rounded to nearest even; a row of zeros gets
    scale 1. The reference math of the per-token quantisation a W8A8 kernel does in its
    prologue (or fuses into the producer of ``x``)."""
    a = x.detach().reshape(-1, x.shape[-1]).float()
    s = _smooth_of(smooth, a.shape[1], a.device)
    if s is not None:
        a = a / s[None, :]
    scale = _row_scales(a)
    return _int8_codes(a, scale).contiguous(), scale.contiguous()


def _int_mm_ok(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether ``torch._int_mm`` (cuBLASLt int8 x int8 -> int32) takes ``a [M, K] @ b [N,
    K]ᵀ`` on the GPU: CUDA tensors, K and N multiples of 8 (M is padded past 16)."""
    return a.is_cuda and b.is_cuda and a.shape[1] % 8 == 0 and b.shape[0] % 8 == 0


def int8_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ bᵀ`` of int8 ``a [M, K]`` and ``b [N, K]`` (an ``nn.Linear`` weight's layout) as
    exact int32 ``[M, N]``: what the IMMA tensor cores accumulate.

    Through ``torch._int_mm`` (cuBLASLt's int8 kernels; rows padded with zeros past its
    M > 16 minimum) on the GPU where K and N are multiples of 8; otherwise in fp32 over
    chunks of :data:`_INT8_EXACT_K` along K (every partial sum an exact integer) summed in
    int32: the same integers on any device."""
    if a.dtype != torch.int8 or b.dtype != torch.int8 or a.dim() != 2 or b.dim() != 2:
        raise ValueError("int8_matmul takes 2-D int8 a [M, K] and b [N, K]")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"K differs: a {tuple(a.shape)}, b {tuple(b.shape)}")
    m, k = a.shape
    b = b.to(a.device)
    if m == 0:
        return torch.zeros(0, b.shape[0], dtype=torch.int32, device=a.device)
    if _int_mm_ok(a, b):
        rows = max(m, 17)
        rows += -rows % 8
        padded = a if rows == m else torch.cat([a, a.new_zeros(rows - m, k)])
        return torch._int_mm(padded.contiguous(), b.t())[:m]
    acc = torch.zeros(m, b.shape[0], dtype=torch.int32, device=a.device)
    for k0 in range(0, k, _INT8_EXACT_K):
        part = a[:, k0 : k0 + _INT8_EXACT_K].float() @ b[:, k0 : k0 + _INT8_EXACT_K].float().T
        acc += part.to(torch.int32)
    return acc


def int8_weights_linear(
    x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """``x @ Wᵀ + bias`` with INT8 weight-only numerics (``int8_weights``), in ``x``'s dtype
    and shape ``[..., out]``: the activations as they are, the int8 codes as values, fp32
    accumulation, the per-channel scale (and the bias) once per output, one rounding."""
    a = x.reshape(-1, x.shape[-1]).float()
    y = (a @ q.to(a.device).float().T) * scale.float().to(a.device)[None, :]
    if bias is not None:
        y = y + bias.float().to(a.device)
    return y.to(x.dtype).reshape(*x.shape[:-1], q.shape[0])


def int8_w8a8_linear(
    x: torch.Tensor,
    q: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    smooth: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x @ Wᵀ + bias`` with INT8 W8A8 numerics, in ``x``'s dtype and shape ``[..., out]``.

    ``(q, scale)``: the weight from :func:`quantize_int8` (one scale per output channel; with
    ``smooth``, quantised with the same factors); ``x`` is quantised per token on every call
    (:func:`quantize_int8_activations`), the int8 products accumulate exactly in int32
    (:func:`int8_matmul`: ``torch._int_mm`` on the GPU), and ``acc * x_scale[m] *
    w_scale[n] (+ bias[n])`` is computed once per output in fp32, one rounding to ``x``'s
    dtype: the epilogue of an IMMA kernel. The fallback path of a W8A8 kernel and its
    reference while debugging."""
    xq, xs = quantize_int8_activations(x, smooth)
    acc = int8_matmul(xq, q.to(xq.device))
    y = acc.float() * xs[:, None] * scale.float().to(xq.device)[None, :]
    if bias is not None:
        y = y + bias.float().to(xq.device)
    return y.to(x.dtype).reshape(*x.shape[:-1], q.shape[0])


def int8_error(
    weight: torch.Tensor,
    q: torch.Tensor,
    scale: torch.Tensor,
    x: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Numerical error of ``weight`` stored as :func:`quantize_int8`'s ``(q, scale)``: the
    report of :func:`fp8_error` (``rel_l2``, ``worst_channel_rel_l2``, ``underflow``,
    ``crest``, ``bytes``; with ``x``: ``output_rel_l2`` and ``output_cosine``). int8's step
    is uniform (``amax / 127``): about 0.01 relative L2 on Gaussian-like rows (e4m3: 0.02-0.03),
    but a row with an outlier (high ``crest``) loses its small weights (``underflow``), where
    e4m3's relative step does not."""
    w = weight.detach().float()
    deq = dequantize_int8(q, scale, torch.float32).to(w.device)
    return {
        "format": "int8",
        "granularity": "per output channel, symmetric (amax / 127)",
        **_error_metrics(w, deq, x),
        "bytes": {
            "before": weight.numel() * weight.element_size(),
            "after": q.numel() + scale.numel() * scale.element_size(),
        },
    }


def int8_w8a8_error(
    weight: torch.Tensor,
    q: torch.Tensor,
    scale: torch.Tensor,
    x: torch.Tensor,
    *,
    smooth: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Numerical error of an INT8 W8A8 layer (``(q, scale)`` from :func:`quantize_int8` with
    the same ``smooth``): the weight report of :func:`int8_error` (of ``weight * smooth`` with
    SmoothQuant), plus

    * ``activation_rel_l2``: ``‖x − x̂‖ / ‖x‖`` of the per-token quantised activations;
      ``activation_crest``: the largest ``amax / RMS`` of a token (an outlier channel sets the
      token's step: at crest 30 the other values keep ~4 int8 steps per RMS, where e4m3 keeps
      its relative step, so activation outliers cost INT8 more than FP8);
      ``activation_underflow``: share of the non-zero activations that became 0;
    * ``output_rel_l2``, ``output_cosine`` and ``output_norm_ratio`` of the W8A8 output against
      ``x @ Wᵀ`` in fp32: the module-level error the near-lossless tier bounds (cosine >=
      0.996, relative L2 <= 0.08, norm within ±2 %)."""
    w = weight.detach().float()
    a = x.detach().float().reshape(-1, w.shape[1]).to(w.device)
    s = _smooth_of(smooth, w.shape[1], w.device)
    report = int8_error(w if s is None else w * s[None, :], q, scale)
    xq, xs = quantize_int8_activations(a, s)
    a_hat = xq.float() * xs[:, None]
    if s is not None:
        a_hat = a_hat * s[None, :]
    a_norm = float(a.norm())
    rms = a.pow(2).mean(dim=1).sqrt()
    crest = a.abs().amax(dim=1) / rms.clamp_min(1e-30)
    nonzero = a != 0
    ref = a @ w.T
    acc = int8_matmul(xq, q.to(xq.device)).double()
    new = (acc * xs.double()[:, None] * scale.double().to(xq.device)[None, :]).float()
    ref_norm, new_norm = float(ref.norm()), float(new.norm())
    cos = float((ref.flatten() @ new.flatten()) / (ref_norm * new_norm)) if ref_norm else 1.0
    report.update(
        activations="int8 per token" + (" (SmoothQuant)" if s is not None else ""),
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


def smoothquant_factors(
    act_amax: torch.Tensor, weight: torch.Tensor, alpha: float = 0.5
) -> torch.Tensor:
    """SmoothQuant's per-input-channel factors ``s_j = amax(|X_j|)^alpha / amax(|W_j|)^(1 -
    alpha)`` (fp32 ``[in]``; Xiao et al. 2022): divide the activations by ``s`` and multiply
    the weight's columns by it (:func:`quantize_int8` / :func:`quantize_int8_activations`
    with ``smooth=s``), so an outlier channel's range moves into the weight, whose per-channel
    scales absorb it. Exact in real arithmetic: only the rounding changes.

    ``act_amax``: the largest ``|x|`` per input channel over calibration activations
    (``x.reshape(-1, in).abs().amax(0)``: the captured inputs); ``alpha``: the share of the
    range that moves (0.5 balances both operands; up to ~0.8 for strong outliers). A channel
    that is zero in either operand gets 1. The factors are static: a target that uses them
    must still pass the evaluator's redrawn-input check."""
    a = act_amax.detach().float().reshape(-1)
    w = weight.detach().float().abs().amax(dim=0).to(a.device)
    if a.numel() != w.numel():
        raise ValueError(f"{a.numel()} activation maxima for a weight with {w.numel()} inputs")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    ok = (a > 0) & (w > 0) & torch.isfinite(a)
    s = a.clamp_min(1e-30).pow(alpha) / w.clamp_min(1e-30).pow(1.0 - alpha)
    return torch.where(ok, s.clamp(1e-5, 1e5), torch.ones_like(s)).contiguous()


# ------------------------------------------------------------------ W4A4 (fp4_w4a4)

#: ``1 / (6 * 448)`` in fp32: the outer (second-level) scale of an NVFP4 row is ``amax *
#: NVFP4_OUTER_STEP``, so the row's largest block scale lands at e4m3's 448 (a product with
#: an fp32 constant: the same on every device and in a kernel, where a division by a Python
#: scalar is a multiplication by its reciprocal on the GPU only).
NVFP4_OUTER_STEP = float(torch.tensor(1.0 / (FP4_MAX * 448.0), dtype=torch.float32))
#: Where the outer NVFP4 scale of the activations comes from (:func:`quantize_fp4_activations`):
#: ``token`` (default) one per row of ``x.reshape(-1, K)``, from that row alone (a producer
#: quantises a row without a grid-wide reduction; the epilogue multiplies by it); ``tensor``
#: one per call (two-level ``F.scaled_mm`` applies it as its tensor-wise scale and writes
#: bf16 itself, but every row's amax is needed first).
FP4_GRANULARITIES = ("token", "tensor")
_FP4_DTYPE = getattr(torch, "float4_e2m1fn_x2", None)


def _fp4_codes(blocks: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
    """e2m1 codes (two per byte, the even element in the low nibble) of fp32 ``blocks [rows,
    nb, block]`` with one fp32 ``step`` per block: ``e2m1(x / step)`` (an IEEE division) to
    nearest even, saturated to ±6; a block whose step is 0 gets codes 0."""
    rows = blocks.shape[0]
    safe = torch.where(step > 0, step, torch.ones_like(step))  # any positive step, exactly
    values = torch.where(step[..., None] > 0, blocks / safe[..., None], torch.zeros_like(blocks))
    codes = _round_e2m1(values.clamp(-FP4_MAX, FP4_MAX)).reshape(
        rows, blocks.shape[1] * blocks.shape[2]
    )
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()


def quantize_fp4_activations(
    x: torch.Tensor, fmt: str = "nvfp4", granularity: str = "token"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(codes, scales, outer)`` of activations ``x [..., K]`` in block-scaled FP4 (W4A4,
    ``fp4_w4a4``): ``codes`` uint8 ``[rows, K / 2]`` (two e2m1 codes per byte, the even element
    in the low nibble, as :func:`quantize_fp4`), ``scales`` ``[rows, K / block]`` in the
    format's scale dtype and ``outer`` fp32 ``[rows]``, with ``x[r, k] ≈ e2m1(code[r, k]) *
    scales[r, k // block] * outer[r]``.

    ``nvfp4`` (an e4m3 scale per 16), in fp32, per row ``r`` of ``x.reshape(-1, K)``
    (``granularity="tensor"``: the amax of the whole call for every row):

    * ``outer = amax(|x_r|) * NVFP4_OUTER_STEP`` (1 for a row of zeros);
    * block scale ``e4m3(min(bmax / (outer * 6), 448))`` (``outer * 6`` rounded to fp32, an
      IEEE division, e4m3 to nearest even): the row's largest block lands at 448;
    * ``step = scale * outer`` (fp32), codes ``e2m1(x / step)`` (an IEEE division) to nearest
      even, saturated to ±6: a block scale that rounds down clamps the block's largest
      elements (by up to e4m3's half step, 6.25 %), as cuBLASLt's and TensorRT's NVFP4
      quantisers do; a block whose scale is 0 (below e4m3's range) gets codes 0.

    ``mxfp4`` (an e8m0 scale per 32): the scale ``2^ceil(log2(bmax / 6))`` of
    :func:`quantize_fp4` (exact, from the exponent bits of ``bmax``; never saturates),
    ``outer`` 1. Dynamic (every call), never a calibrated scale. ``K`` must be a multiple of
    the block."""
    if fmt not in FP4_FORMATS:
        raise ValueError(f"unknown FP4 format {fmt!r} (one of {', '.join(FP4_FORMATS)})")
    if granularity not in FP4_GRANULARITIES:
        raise ValueError(f"unknown granularity {granularity!r} ({', '.join(FP4_GRANULARITIES)})")
    block, scale_dtype = FP4_FORMATS[fmt]
    k = x.shape[-1]
    if k % block:
        raise ValueError(f"the last dimension {k} is not a multiple of the {fmt} block ({block})")
    a = x.detach().reshape(-1, k).float()
    rows = a.shape[0]
    blocks = a.reshape(rows, k // block, block)
    bmax = blocks.abs().amax(dim=-1)
    if fmt == "mxfp4":
        # bmax = m * 2^e, m in [0.5, 1): 6 * 2^(e - 3) = 0.75 * 2^e covers it when m <= 0.75
        mantissa, exponent = torch.frexp(bmax)
        e = exponent - 3 + (mantissa > 0.75).to(exponent.dtype)
        e = torch.where(bmax > 0, e, torch.full_like(e, -127)).clamp(-127, 127)
        scales = (e + 127).to(torch.uint8).view(scale_dtype)
        outer = torch.ones(rows, device=a.device)
        step = scales.float()
    else:
        amax = a.abs().amax(dim=1) if k else torch.zeros(rows, device=a.device)
        if granularity == "tensor" and rows:
            amax = amax.amax().expand(rows)
        outer = torch.where(amax > 0, amax * NVFP4_OUTER_STEP, torch.ones_like(amax))
        scales = (bmax / (outer * FP4_MAX)[:, None]).clamp(max=448.0).to(scale_dtype)
        step = scales.float() * outer[:, None]
    return _fp4_codes(blocks, step), scales.contiguous(), outer.contiguous()


def fp4_values(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """fp32 ``e2m1(code) * block scale`` ``[rows, K]`` of FP4 ``codes [rows, K / 2]`` and
    ``scales [rows, K / block]`` (no outer or tensor scale): exact in fp32 (an e2m1 value has
    2 significant bits, an e4m3 scale 4), what the block-scaled tensor cores multiply."""
    return dequantize_fp4(codes, scales, 1.0, torch.float32)


def hadamard(n: int, device: torch.device | str | None = None) -> torch.Tensor:
    """The orthonormal Sylvester-Hadamard matrix of order ``n`` (a power of two), fp32,
    symmetric: ``H @ H = I``."""
    if n < 1 or n & (n - 1):
        raise ValueError(f"the Hadamard order must be a power of two, got {n}")
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def hadamard_rotate(x: torch.Tensor, block: int) -> torch.Tensor:
    """``x [..., K]`` with every ``block`` consecutive elements along K multiplied by
    :func:`hadamard` (``block``), in fp32: a block-diagonal rotation (QuaRot / QuTLASS style)
    that spreads an outlier over its block before FP4 quantisation. Applied to the
    activations and to the weight's rows alike it leaves ``x @ Wᵀ`` unchanged in exact
    arithmetic: ``(x H)(W H)ᵀ = x Wᵀ``. ``K`` must be a multiple of ``block``."""
    k = x.shape[-1]
    if k % block:
        raise ValueError(f"the last dimension {k} is not a multiple of the rotation ({block})")
    h = hadamard(block, x.device)
    v = x.detach().float().reshape(-1, k // block, block) @ h
    return v.reshape(*x.shape[:-1], k)


def swizzle_fp4_scales(scales: torch.Tensor) -> torch.Tensor:
    """FP4 block scales ``[rows, K / block]`` (NVFP4 e4m3 or MXFP4 e8m0) → cuBLASLt's flat 128
    x 4 blocked layout (:func:`swizzle_mx_scales`: rows padded to 128, columns to 4, padding
    code 0), in their dtype: ``F.scaled_mm``'s ``SWIZZLE_32_4_4``, CuTe's
    ``tile_atom_to_shape_SF``."""
    swizzled = swizzle_mx_scales(scales.view(torch.uint8))
    return swizzled.view(torch.uint8).view(scales.dtype)


def _fp4_scaled_mm_ok(x: torch.Tensor, codes: torch.Tensor, fmt: str) -> bool:
    """Whether ``F.scaled_mm`` runs these operands as NVFP4 (``BlockWise1x16`` e4m3 scales):
    CUDA bf16 activations, K a multiple of 16 and N of 8, block-scaled FP4 tensor cores
    (sm_100+; measured on sm_120) and a torch with ``F.scaled_mm`` and an FP4 dtype. torch
    2.14 refuses MXFP4 (``BlockWise1x32`` on e2m1) outside B200 / B300: the fallback."""
    import torch.nn.functional as F

    if fmt != "nvfp4" or _FP4_DTYPE is None or not hasattr(F, "scaled_mm"):
        return False
    if not (x.is_cuda and codes.is_cuda and x.dtype == torch.bfloat16):
        return False
    if (2 * codes.shape[1]) % 16 or codes.shape[0] % 8:
        return False
    return torch.cuda.get_device_capability(x.device) >= (10, 0)


def fp4_w4a4_linear(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    tensor_scale: torch.Tensor | float,
    bias: torch.Tensor | None = None,
    *,
    fmt: str = "nvfp4",
    granularity: str = "token",
    rotate: int | None = None,
) -> torch.Tensor:
    """``x @ Wᵀ + bias`` with W4A4 numerics (``fp4_w4a4``), in ``x``'s dtype and shape
    ``[..., out]``.

    ``(codes, scales, tensor_scale)``: the weight from :func:`quantize_fp4` in ``fmt`` (of
    ``hadamard_rotate(W, rotate)`` with ``rotate``). ``x`` is rotated (``rotate``) and
    quantised on every call (:func:`quantize_fp4_activations`), the block-scaled products
    ``Σ_k (e2m1 · s_x)(e2m1 · s_w)`` accumulate in fp32, and ``acc * (outer[m] *
    tensor_scale) (+ bias[n])`` is computed once per output in fp32, one rounding to ``x``'s
    dtype: the epilogue of a block-scaled tensor-core kernel. NVFP4 runs through
    ``F.scaled_mm`` (``BlockWise1x16``, fp32 out: bit for bit the fp32 math on sm_120) where
    it applies; otherwise the same math in fp32 (slow: a fallback and a reference)."""
    xq, xs, outer = quantize_fp4_activations(
        hadamard_rotate(x, rotate) if rotate else x, fmt, granularity
    )
    if _fp4_scaled_mm_ok(x, codes, fmt):
        import torch.nn.functional as F

        blockwise, swizzle = F.ScalingType.BlockWise1x16, F.SwizzleType.SWIZZLE_32_4_4
        acc = F.scaled_mm(
            xq.view(torch.float4_e2m1fn_x2),
            codes.view(torch.float4_e2m1fn_x2).t(),  # column-major [K / 2, N]: cuBLASLt's B layout
            scale_a=swizzle_fp4_scales(xs),
            scale_recipe_a=blockwise,
            scale_b=swizzle_fp4_scales(scales),
            scale_recipe_b=blockwise,
            swizzle_a=swizzle,
            swizzle_b=swizzle,
            output_dtype=torch.float32,
        )
    else:
        acc = fp4_values(xq, xs) @ fp4_values(codes, scales).to(xq.device).T
    if isinstance(tensor_scale, torch.Tensor):  # a 0-d tensor: no host sync (CUDA graphs)
        ts: torch.Tensor | float = tensor_scale.float().to(acc.device)
    else:
        ts = float(tensor_scale)
    y = acc * (outer[:, None] * ts)
    if bias is not None:
        y = y + bias.float().to(y.device)
    return y.to(x.dtype).reshape(*x.shape[:-1], codes.shape[0])


def fp4_saturation(
    x: torch.Tensor, scales: torch.Tensor, outer: torch.Tensor, fmt: str = "nvfp4"
) -> dict[str, Any]:
    """How FP4 activation scales fit the block maxima of ``x [..., K]`` (``scales [rows, K /
    block]`` and ``outer [rows]`` of :func:`quantize_fp4_activations`): ``blocks``,
    ``saturated`` (blocks whose maximum exceeds ``6 x step``: their largest elements are
    clamped; NVFP4 rounds its block scales to nearest, so some blocks saturate a little),
    ``share`` and ``worst_ratio`` (the largest ``bmax / step``: up to 6.375 for NVFP4 with a
    normal e4m3 block scale, more with a subnormal one, below 2^-6)."""
    block = FP4_FORMATS[fmt][0]
    k = x.shape[-1]
    a = x.detach().reshape(-1, k).float()
    rows = a.shape[0]
    bmax = a.reshape(rows, k // block, block).abs().amax(dim=-1)
    step = scales.float().to(a.device) * outer.float().to(a.device)[:, None]
    ratio = torch.where(step > 0, bmax / step.clamp_min(1e-38), torch.zeros_like(bmax))
    saturated = int((ratio > FP4_MAX * (1 + 2.0**-20)).sum())
    blocks = rows * (k // block)
    return {
        "blocks": blocks,
        "saturated": saturated,
        "share": _sig(saturated / blocks) if blocks else 0.0,
        "worst_ratio": _sig(float(ratio.max())) if blocks else 0.0,
    }


def fp4_w4a4_error(
    weight: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    tensor_scale: torch.Tensor | float,
    x: torch.Tensor,
    *,
    fmt: str = "nvfp4",
    granularity: str = "token",
    rotate: int | None = None,
) -> dict[str, Any]:
    """Numerical error of a W4A4 layer (``(codes, scales, tensor_scale)``: :func:`quantize_fp4`
    of ``weight``, rotated with ``rotate``): the weight report of :func:`fp4_error` (of the
    rotated weight), plus

    * ``activation_rel_l2``: ``‖x − x̂‖ / ‖x‖`` of the quantised (rotated) activations;
      ``activation_crest``: the largest ``amax / RMS`` of a token (before the rotation);
      ``activation_saturation``: :func:`fp4_saturation` (``share``, ``worst_ratio``);
    * ``output_rel_l2``, ``output_cosine`` and ``output_norm_ratio`` of the W4A4 output
      against ``x @ Wᵀ`` in fp32: the module-level error the ``near-lossless-fp4a`` tier
      bounds (:data:`kernel_agent.kernels.compare.NEAR_LOSSLESS_BOUNDS`)."""
    w = weight.detach().float()
    a = x.detach().float().reshape(-1, w.shape[1]).to(w.device)
    report = fp4_error(hadamard_rotate(w, rotate) if rotate else w, codes, scales, tensor_scale)
    a_rot = hadamard_rotate(a, rotate) if rotate else a
    xq, xs, outer = quantize_fp4_activations(a_rot, fmt, granularity)
    a_hat = fp4_values(xq, xs) * outer[:, None]
    rms = a.pow(2).mean(dim=1).sqrt()
    crest = a.abs().amax(dim=1) / rms.clamp_min(1e-30)
    ref = a @ w.T
    acc = fp4_values(xq, xs).double() @ fp4_values(codes, scales).to(a.device).double().T
    new = (acc * (outer.double()[:, None] * float(tensor_scale))).float()
    ref_norm, new_norm = float(ref.norm()), float(new.norm())
    cos = float((ref.flatten() @ new.flatten()) / (ref_norm * new_norm)) if ref_norm else 1.0
    a_norm = float(a_rot.norm())
    saturation = fp4_saturation(a_rot, xs, outer, fmt)
    block, kind = FP4_FORMATS[fmt][0], "e4m3" if fmt == "nvfp4" else "e8m0"
    report.update(
        activations=f"{fmt}: e2m1 + {kind} scale per {block}"
        + (f" x fp32 per {granularity}" if fmt == "nvfp4" else "")
        + (f", Hadamard {rotate}" if rotate else ""),
        activation_rel_l2=_sig(float((a_rot - a_hat).norm()) / a_norm if a_norm > 0 else 0.0),
        activation_crest=_sig(float(crest[rms > 0].max()) if bool((rms > 0).any()) else 0.0),
        activation_saturation={k: saturation[k] for k in ("share", "worst_ratio")},
        output_rel_l2=_sig(float((ref - new).norm()) / ref_norm if ref_norm else 0.0),
        output_cosine=round(cos, 6),
        output_norm_ratio=round(new_norm / ref_norm, 5) if ref_norm else 1.0,
    )
    return report


def fp4_w4a4_sensitivity(
    module: torch.nn.Module,
    run: Callable[[], Any],
    *,
    fmt: str = "nvfp4",
    granularity: str = "token",
    rotate: int | None = None,
    calls: int = 4,
) -> list[dict[str, Any]]:
    """The sensitivity probe of a W4A4 target: which of ``module``'s ``nn.Linear`` layers to
    keep in FP8 W8A8. ``run()`` calls the module (e.g. on its captured inputs); the first
    ``calls`` inputs of every ``nn.Linear`` are kept (their rows together), and each layer is
    quantised alone to W4A4 (:func:`fp4_w4a4_error` in ``fmt`` / ``granularity`` / ``rotate``)
    and to FP8 W8A8 (:func:`fp8_w8a8_error`) on them.

    One row per layer, the most sensitive first (largest W4A4 output relative L2 error):
    ``name``, ``rows``, ``in_features``, ``out_features``, ``flop_share`` (of the layers'
    GEMM FLOPs over those calls), ``activation_crest``, ``w4a4_rel_l2``,
    ``w4a4_norm_change`` (``‖new‖ / ‖ref‖ - 1``: quantised activations can shrink a GEMM's
    output by up to ~1 %, compounding through an MLP), ``fp8_rel_l2``. On the VoxCPM2 LocDiT
    layer (M = 352) gate / up rank first (0.088 vs FP8's 0.019; 49 % of the FLOPs) and FP8 for
    them alone moves the layer's hidden output from relative L2 0.036 / norm -3.4 % to 0.014 /
    -0.9 %. Move layers to FP8 from the top until the evaluator and the perceptual gate pass;
    a layer the module calls with other inputs later is judged on these only."""
    seen: dict[str, list[torch.Tensor]] = {}
    layers = {n: m for n, m in module.named_modules() if isinstance(m, torch.nn.Linear)}

    def keep(name: str) -> Callable[..., None]:
        def hook(_: torch.nn.Module, args: tuple[Any, ...]) -> None:
            if len(seen.setdefault(name, [])) < calls and isinstance(args[0], torch.Tensor):
                seen[name].append(args[0].detach().reshape(-1, args[0].shape[-1]))

        return hook

    hooks = [m.register_forward_pre_hook(keep(n)) for n, m in layers.items()]
    try:
        with torch.no_grad():
            run()
    finally:
        for h in hooks:
            h.remove()
    found = []
    for name, inputs in seen.items():
        w = layers[name].weight.detach()
        x = torch.cat(inputs).to(w.device)
        if w.shape[1] % (rotate or FP4_FORMATS[fmt][0]) or not x.numel():
            continue  # K not a multiple of the block (or the rotation): W4A4 does not apply
        wq = hadamard_rotate(w, rotate) if rotate else w
        e4 = fp4_w4a4_error(
            w, *quantize_fp4(wq, fmt), x, fmt=fmt, granularity=granularity, rotate=rotate
        )
        e8 = fp8_w8a8_error(w, *quantize_fp8(w), x)
        found.append(
            {
                "name": name,
                "rows": int(x.shape[0]),
                "in_features": int(w.shape[1]),
                "out_features": int(w.shape[0]),
                "flops": 2 * int(x.shape[0]) * int(w.shape[1]) * int(w.shape[0]),
                "activation_crest": e4["activation_crest"],
                "w4a4_rel_l2": e4["output_rel_l2"],
                "w4a4_norm_change": _sig(e4["output_norm_ratio"] - 1.0),
                "fp8_rel_l2": e8["output_rel_l2"],
            }
        )
    total = sum(r["flops"] for r in found) or 1
    for row in found:
        row["flop_share"] = _sig(row.pop("flops") / total)
    return sorted(found, key=lambda r: -r["w4a4_rel_l2"])
