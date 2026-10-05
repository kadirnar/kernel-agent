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
