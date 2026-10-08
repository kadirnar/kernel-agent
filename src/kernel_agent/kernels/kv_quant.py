"""FP8 KV cache helpers (``"precision": "fp8_kv"``): reference math, a quantise-on-append
cache, and the byte share that decides whether the class can pay.

A target whose spec says ``"precision": "fp8_kv"`` (opt-in: ``--precisions ...,fp8_kv`` in a
``--quality near-lossless`` run, ``kernel_agent/precisions.py``; near-lossless tier) may keep
its attention's KV cache in e4m3. The contract (skill ``fp8-kv-cache``, "FP8 KV
cache"): K and V get one fp32 scale per (token, KV head) (``amax`` over the head dimension /
448, dynamic), written once when tokens are appended and never re-quantised per step; the
attention reads codes + scales (dequantised in registers, or the K scale folded into the
scores and the V scale into P); Q, the softmax and the output stay as in eager; weights are
unchanged. Per-token scales need no calibration: a static per-tensor K/V scale taken from one
text was exceeded by another in ~50 % of a model's layers (docs/FP8.md §5).

* :func:`quantize_fp8_kv` / :func:`dequantize_fp8_kv`: the codes and scales of K or V
  ``[..., tokens, head_dim]``.
* :func:`fp8_kv_attention`: softmax(Q Kᵀ · scale) V in fp32 over the dequantised cache
  (grouped-query heads, optional per-sequence lengths): the reference a kernel is checked
  against while debugging, and the fallback path.
* :class:`Fp8KVCache`: a preallocated e4m3 cache that quantises only the appended tokens.
* :func:`kv_cache_share`: the KV cache's share of the bytes a decode step reads, from the
  model's own shapes. FP8 halves the KV bytes only: it pays when the cache is a large share
  (long contexts, large batches). Measured (RTX 5070 Ti, a Triton decode kernel, batch 16,
  GQA 8, docs/FP8.md §5): 0.90x at 77 cached tokens (latency bound, the conversion costs
  more than the bytes save), 1.21x at 512, 1.29x at 2048, 1.31x at 8192 (one program per
  sequence and KV head; split-KV is the way to more).
"""

from __future__ import annotations

from typing import Any

import torch

#: Largest finite e4m3 value.
E4M3_MAX = 448.0


def quantize_fp8_kv(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(codes, scales)`` of K or V ``t [..., tokens, head_dim]``: e4m3 codes of ``t``'s shape
    and fp32 scales ``[..., tokens]``, one per (token, head): ``amax(|row|) / 448`` (1 for a
    row of zeros), codes ``round(t / scale)`` to nearest even, clamped to ±448."""
    a = t.detach().float()
    amax = a.abs().amax(dim=-1)
    scale = torch.where(amax > 0, amax / E4M3_MAX, torch.ones_like(amax))
    codes = (a / scale[..., None]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return codes.contiguous(), scale.contiguous()


def dequantize_fp8_kv(
    codes: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """The values of :func:`quantize_fp8_kv`'s ``(codes, scales)`` in ``dtype``."""
    return (codes.float() * scales.float()[..., None]).to(dtype)


def fp8_kv_attention(
    q: torch.Tensor,
    k_codes: torch.Tensor,
    k_scales: torch.Tensor,
    v_codes: torch.Tensor,
    v_scales: torch.Tensor,
    scale: float | None = None,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """softmax(Q Kᵀ · scale) V over an e4m3 cache, in fp32, returned in ``q``'s dtype.

    ``q [B, Hq, Lq, D]``; codes ``[B, Hkv, L, D]`` and scales ``[B, Hkv, L]`` (Hq a multiple
    of Hkv: grouped-query heads); ``scale`` default ``D ** -0.5``; ``lengths [B]``: the
    valid cached tokens per sequence (default all ``L``; at least one). Every query sees
    every valid cached token (decode: the new token's K / V are already appended)."""
    qf = q.float()
    k = dequantize_fp8_kv(k_codes, k_scales).to(qf.device)
    v = dequantize_fp8_kv(v_codes, v_scales).to(qf.device)
    rep = qf.shape[1] // k.shape[1]
    if rep > 1:
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    s = qf @ k.transpose(-1, -2) * (qf.shape[-1] ** -0.5 if scale is None else scale)
    if lengths is not None:
        valid = torch.arange(k.shape[2], device=s.device)[None, :] < lengths.to(s.device)[:, None]
        s = s.masked_fill(~valid[:, None, None, :], float("-inf"))
    return (torch.softmax(s, dim=-1) @ v).to(q.dtype)


def fp8_kv_error(k: torch.Tensor, v: torch.Tensor) -> dict[str, Any]:
    """Error report of an e4m3 cache for ``NOTES.md``: relative L2 error of K and V after
    :func:`quantize_fp8_kv`, their worst per-(token, head) error, and the bytes before /
    after (codes + one fp32 scale per token and head)."""
    report: dict[str, Any] = {"format": "float8_e4m3fn", "granularity": "per token and head"}
    for name, t in (("k", k), ("v", v)):
        a = t.detach().float()
        deq = dequantize_fp8_kv(*quantize_fp8_kv(a))
        norm = float(a.norm())
        rows = (deq - a).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-30)
        report[f"{name}_rel_l2"] = round(float((deq - a).norm()) / norm, 5) if norm else 0.0
        report[f"{name}_worst_row_rel_l2"] = round(float(rows.max()), 5) if rows.numel() else 0.0
    elems = k.numel() + v.numel()
    rows = elems // max(k.shape[-1], 1)
    report["bytes"] = {"before": elems * k.element_size(), "after": elems + 4 * rows}
    return report


class Fp8KVCache:
    """A preallocated e4m3 KV cache ``[batch, kv_heads, max_tokens, head_dim]`` with one fp32
    scale per (token, head). :meth:`append` quantises only the new tokens (per-token scales:
    the same codes as quantising the whole cache at once) and returns the valid views."""

    def __init__(
        self,
        batch: int,
        kv_heads: int,
        max_tokens: int,
        head_dim: int,
        device: torch.device | str | None = None,
    ) -> None:
        shape = (batch, kv_heads, max_tokens, head_dim)
        self.k = torch.zeros(shape, dtype=torch.float8_e4m3fn, device=device)
        self.v = torch.zeros(shape, dtype=torch.float8_e4m3fn, device=device)
        self.k_scale = torch.ones(shape[:3], dtype=torch.float32, device=device)
        self.v_scale = torch.ones(shape[:3], dtype=torch.float32, device=device)
        self.length = 0

    def append(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Write ``k``, ``v`` ``[batch, kv_heads, new, head_dim]`` after the cached tokens;
        returns ``(k_codes, k_scales, v_codes, v_scales)`` of every valid token."""
        start, end = self.length, self.length + k.shape[2]
        if end > self.k.shape[2]:
            raise ValueError(f"FP8 KV cache full: {end} > {self.k.shape[2]} tokens")
        for codes, scales, t in ((self.k, self.k_scale, k), (self.v, self.v_scale, v)):
            c, s = quantize_fp8_kv(t)
            codes[:, :, start:end] = c.to(codes.device)
            scales[:, :, start:end] = s.to(scales.device)
        self.length = end
        return (
            self.k[:, :, :end],
            self.k_scale[:, :, :end],
            self.v[:, :, :end],
            self.v_scale[:, :, :end],
        )


def kv_cache_share(
    *,
    batch: int,
    tokens: int,
    layers: int,
    kv_heads: int,
    head_dim: int,
    other_bytes: float,
    elem_bytes: float = 2.0,
) -> dict[str, float]:
    """The KV cache's share of what one decode step reads: ``kv_bytes`` (K and V of every
    layer at ``tokens`` cached tokens, ``elem_bytes`` per element: 2 for bf16), ``share`` of
    ``kv_bytes + other_bytes`` (``other_bytes``: the weights and activations the step streams,
    e.g. the profile's bytes per step), and the step's bytes saved by e4m3 K / V
    (``saved_share``; one fp32 scale per token and head included). Measure on the model at
    hand: FP8 KV pays only when the share is large (long contexts, large batches)."""
    rows = 2 * batch * tokens * layers * kv_heads
    kv = rows * head_dim * elem_bytes
    fp8 = rows * (head_dim + 4)
    total = kv + other_bytes
    return {
        "kv_bytes": kv,
        "share": kv / total if total else 0.0,
        "saved_share": max(kv - fp8, 0.0) / total if total else 0.0,
    }
