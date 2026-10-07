"""Example candidate (Triton): split-KV (flash-decoding) attention over an FP8 e4m3 KV cache,
for decode steps over long contexts.

Reduced precision: ``"precision": "fp8_kv"`` (opt-in: ``--precisions ...,fp8_kv`` in a
``--quality near-lossless`` run; knowledge/low_precision.md, "FP8 KV cache"). Reference math
and helpers: ``kernel_agent.kernels.kv_quant`` (``quantize_fp8_kv``, ``fp8_kv_attention``,
``Fp8KVCache``, ``kv_cache_share``).

When it pays: only where the KV cache is a large share of the bytes a decode step reads
(``kv_cache_share`` on the model's own shapes: thousands of cached tokens, large batches).
Measured with a simpler kernel (one program per sequence and KV head; RTX 5070 Ti, batch 16,
16 query / 2 KV heads, head dim 128, docs/FP8.md §5): e4m3 K / V 0.90x of bf16 at 77 cached
tokens, 1.21x at 512, 1.29x at 2048, 1.31x at 8192: 32 programs on 70 SMs and a conversion
per element left it at ~550 GB/s of codes. Splitting the cache across programs (this kernel)
is what should approach the 2x of half the bytes; not measured yet.

* ``fp8_kv_decode(q, k_codes, k_scales, v_codes, v_scales, scale, lengths)``: q ``[B, Hq, D]``
  (or ``[B, Hq, 1, D]``) bf16, codes ``[B, Hkv, L, D]`` e4m3 and scales ``[B, Hkv, L]`` fp32
  from ``quantize_fp8_kv`` / ``Fp8KVCache`` (any strides with a unit last stride), optional
  ``lengths [B]`` (valid tokens per sequence, >= 1). Grouped-query heads: one program serves
  the ``Hq / Hkv`` query heads of a KV head (padded to 16 rows for the MMA) over one chunk of
  the cache; the chunks (``splits``, chosen to give ~4 programs per SM) write fp32 partial
  (max, sum, output) and ``_combine_kernel`` merges them.
* Math: scores ``(q · k_codes) * k_scale * scale`` (bf16 x e4m3-as-bf16 products are exact in
  fp32; the K scale folds into the scores), online softmax in fp32, ``P * v_scale`` rounded
  to bf16 for the P·V MMA on the V codes (as FlashAttention rounds P), fp32 accumulation,
  one bf16 rounding of the output.
* ``build()`` wraps a decode-attention module ``forward(q, k, v)`` with SDPA semantics
  (``q [B, Hq, 1, D]``, ``k / v [B, Hkv, L, D]``, scale ``D ** -0.5`` or the module's
  ``scale``, no mask): it quantises K and V on every call so the evaluator can check it. In a
  model that is the wrong place: write the cache in e4m3 when tokens are appended
  (``Fp8KVCache.append``, or the model's cache update) and call :func:`fp8_kv_decode` on it;
  the per-call quantisation reads the bf16 cache once more, so this module is not faster
  than SDPA. D a power of two in 16..256; other shapes fall back to the reference.

Not run on a GPU yet; verify with ``kernel-agent doctor --smoke`` and ``pytest -m gpu
tests/test_fp8_toolkit.py``.
"""

import torch
import triton
import triton.language as tl
from torch import nn

from kernel_agent.kernels.kv_quant import fp8_kv_attention, quantize_fp8_kv

BL = 64  # cached tokens per inner tile


@triton.jit
def _quant_kv_kernel(x, q, s, R, D: tl.constexpr, BR: tl.constexpr):
    # rows of D elements (one token of one head): scale = amax / 448, codes x / scale
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    rd = tl.arange(0, D)
    ok = r < R
    v = tl.load(x + r[:, None] * D + rd[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(v), 1)
    sc = tl.math.div_rn(amax, tl.full(amax.shape, 448.0, tl.float32))
    sc = tl.where(amax > 0, sc, 1.0)
    codes = tl.clamp(tl.math.div_rn(v, tl.zeros_like(v) + sc[:, None]), -448.0, 448.0)
    tl.store(q + r[:, None] * D + rd[None, :], codes.to(tl.float8e4nv), mask=ok[:, None])
    tl.store(s + r, sc, mask=ok)


@triton.jit
def _split_kernel(
    q,
    kc,
    ks,
    vc,
    vs,
    lens,
    po,
    pm,
    pl,
    Hkv,
    L,
    chunk,
    sm_scale,
    sqb,
    sqh,
    skb,
    skh,
    skl,
    ssb,
    ssh,
    svb,
    svh,
    svl,
    stb,
    sth,
    G: tl.constexpr,
    GP: tl.constexpr,
    D: tl.constexpr,
    BL: tl.constexpr,
    HAS_LENS: tl.constexpr,
):
    # program (b * Hkv + h, split): the G query heads of KV head h over tokens [start, end)
    bh = tl.program_id(0)
    sp = tl.program_id(1)
    S = tl.num_programs(1)
    b = bh // Hkv
    h = bh % Hkv
    n = L
    if HAS_LENS:
        n = tl.minimum(tl.load(lens + b), L)
    rg = tl.arange(0, GP)
    rd = tl.arange(0, D)
    q_ptrs = q + b * sqb + (h * G + rg)[:, None] * sqh + rd[None, :]
    qv = tl.load(q_ptrs, mask=(rg < G)[:, None], other=0.0).to(tl.bfloat16)
    m = tl.full((GP,), float("-inf"), tl.float32)
    lsum = tl.zeros((GP,), tl.float32)
    acc = tl.zeros((GP, D), tl.float32)
    start = sp * chunk
    end = tl.minimum(start + chunk, n)
    for t0 in range(start, end, BL):
        rl = t0 + tl.arange(0, BL)
        ok = rl < end
        k_ptrs = kc + b * skb + h * skh + rl[:, None] * skl + rd[None, :]
        kt = tl.load(k_ptrs, mask=ok[:, None], other=0.0)
        k_s = tl.load(ks + b * ssb + h * ssh + rl, mask=ok, other=0.0)
        sc = tl.dot(qv, tl.trans(kt.to(tl.bfloat16)))  # q . codes, fp32 [GP, BL]
        sc = sc * (k_s * sm_scale)[None, :]
        sc = tl.where(ok[None, :], sc, float("-inf"))
        mn = tl.maximum(m, tl.max(sc, 1))
        p = tl.exp(sc - mn[:, None])
        alpha = tl.exp(m - mn)
        lsum = lsum * alpha + tl.sum(p, 1)
        v_ptrs = vc + b * svb + h * svh + rl[:, None] * svl + rd[None, :]
        vt = tl.load(v_ptrs, mask=ok[:, None], other=0.0)  # 0, never a NaN code, past the end
        v_s = tl.load(vs + b * stb + h * sth + rl, mask=ok, other=0.0)
        pv = (p * v_s[None, :]).to(tl.bfloat16)  # the V scale folds into P
        acc = acc * alpha[:, None] + tl.dot(pv, vt.to(tl.bfloat16))
        m = mn
    base = bh * S + sp
    tl.store(po + (base * GP + rg)[:, None] * D + rd[None, :], acc)
    tl.store(pm + base * GP + rg, m)
    tl.store(pl + base * GP + rg, lsum)


@triton.jit
def _combine_kernel(
    po, pm, pl, out, S, Hkv, sob, soh, G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr
):
    # out[b, h * G + g] = sum_s e^(m_s - M) o_s / sum_s e^(m_s - M) l_s (empty splits: m = -inf)
    bh = tl.program_id(0)
    b = bh // Hkv
    h = bh % Hkv
    rg = tl.arange(0, GP)
    rd = tl.arange(0, D)
    big = tl.full((GP,), float("-inf"), tl.float32)
    for sp in range(0, S):
        big = tl.maximum(big, tl.load(pm + (bh * S + sp) * GP + rg))
    acc = tl.zeros((GP, D), tl.float32)
    den = tl.zeros((GP,), tl.float32)
    for sp in range(0, S):
        base = bh * S + sp
        w = tl.exp(tl.load(pm + base * GP + rg) - big)
        den += w * tl.load(pl + base * GP + rg)
        acc += w[:, None] * tl.load(po + (base * GP + rg)[:, None] * D + rd[None, :])
    o = acc / den[:, None]
    ptrs = out + b * sob + (h * G + rg)[:, None] * soh + rd[None, :]
    tl.store(ptrs, o.to(tl.bfloat16), mask=(rg < G)[:, None])


_SMS: dict[int, int] = {}


def _sms(device: torch.device) -> int:
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _SMS:
        _SMS[index] = torch.cuda.get_device_properties(index).multi_processor_count
    return _SMS[index]


def quantize_kv(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(codes, scales)`` of K or V ``[..., L, D]`` on the GPU: ``quantize_fp8_kv``'s math
    (one scale per token and head) in one Triton pass."""
    d = t.shape[-1]
    x = t.contiguous()
    codes = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(x.shape[:-1], device=x.device, dtype=torch.float32)
    rows = x.numel() // d
    if rows:
        br = 16
        _quant_kv_kernel[(triton.cdiv(rows, br),)](x, codes, scales, rows, D=d, BR=br)
    return codes, scales


def fp8_kv_decode(
    q: torch.Tensor,
    k_codes: torch.Tensor,
    k_scales: torch.Tensor,
    v_codes: torch.Tensor,
    v_scales: torch.Tensor,
    scale: float | None = None,
    lengths: torch.Tensor | None = None,
    splits: int = 0,
) -> torch.Tensor:
    """Decode attention of one query token per sequence over an e4m3 cache; returns ``q``'s
    shape in bf16. ``splits``: chunks of the cache per (sequence, KV head) (0: ~4 programs
    per SM)."""
    b, hkv, length, d = k_codes.shape
    k_codes, v_codes, k_scales, v_scales = (
        t if t.stride(-1) == 1 else t.contiguous() for t in (k_codes, v_codes, k_scales, v_scales)
    )
    q3 = q.reshape(b, -1, d)
    hq = q3.shape[1]
    g = hq // hkv
    gp = max(16, triton.next_power_of_2(g))
    if splits <= 0:
        splits = max(1, min(triton.cdiv(length, BL), triton.cdiv(4 * _sms(q.device), b * hkv)))
    chunk = triton.cdiv(triton.cdiv(length, splits), BL) * BL
    splits = triton.cdiv(length, chunk)
    f32 = dict(device=q.device, dtype=torch.float32)
    po = torch.empty((b * hkv, splits, gp, d), **f32)
    pm = torch.empty((b * hkv, splits, gp), **f32)
    pl = torch.empty((b * hkv, splits, gp), **f32)
    out = torch.empty((b, hq, d), device=q.device, dtype=torch.bfloat16)
    sm_scale = d**-0.5 if scale is None else float(scale)
    lens = lengths.to(device=q.device, dtype=torch.int32) if lengths is not None else pm
    _split_kernel[(b * hkv, splits)](
        q3,
        k_codes,
        k_scales,
        v_codes,
        v_scales,
        lens,
        po,
        pm,
        pl,
        hkv,
        length,
        chunk,
        sm_scale,
        q3.stride(0),
        q3.stride(1),
        *k_codes.stride()[:3],
        *k_scales.stride()[:2],
        *v_codes.stride()[:3],
        *v_scales.stride()[:2],
        G=g,
        GP=gp,
        D=d,
        BL=BL,
        HAS_LENS=lengths is not None,
        num_warps=4,
    )
    _combine_kernel[(b * hkv,)](
        po, pm, pl, out, splits, hkv, out.stride(0), out.stride(1), G=g, GP=gp, D=d
    )
    return out.reshape(q.shape)


def _ok(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    if not (q.is_cuda and q.dtype == k.dtype == v.dtype == torch.bfloat16):
        return False
    if q.dim() != 4 or k.dim() != 4 or q.shape[2] != 1 or k.shape != v.shape:
        return False
    b, hq, _, d = q.shape
    return (
        k.shape[0] == b
        and k.shape[3] == d
        and k.shape[2] >= 1
        and hq % k.shape[1] == 0
        and 16 <= d <= 256
        and d & (d - 1) == 0
        and q.stride(-1) == 1
    )


class Fp8KVDecodeAttention(nn.Module):
    """SDPA decode attention (``q [B, Hq, 1, D]``, ``k / v [B, Hkv, L, D]``) reading K / V in
    e4m3 with per-(token, head) scales."""

    def __init__(self, reference: nn.Module) -> None:
        super().__init__()
        self.reference = reference
        self.scale = getattr(reference, "scale", None)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if not _ok(q, k, v):
            if q.dim() == 4 and k.dim() == 4 and q.shape[-1] == k.shape[-1]:
                kc, ks = quantize_fp8_kv(k)  # the same numerics in torch
                vc, vs = quantize_fp8_kv(v)
                return fp8_kv_attention(q, kc, ks, vc, vs, self.scale)
            return self.reference(q, k, v)
        # in a model: the cache is already e4m3 (written on append); here per call (checkable)
        kc, ks = quantize_kv(k)
        vc, vs = quantize_kv(v)
        return fp8_kv_decode(q, kc, ks, vc, vs, self.scale)


def build(reference: nn.Module) -> nn.Module:
    return Fp8KVDecodeAttention(reference)
