"""Example candidate (Triton): attention over short sequences (query and key length <= 32)
as one tile per (sequence, query head), for any head count, GQA ratio and head dimension.

FlashAttention-style kernels tile 64-128 query rows per program; over a handful of tokens
those tiles are mostly padding. Here each program loads its whole query block and the
shared key / value head into registers (rows padded to 16 or 32, the head dimension to a
power of two, both masked), computes S = Q K^T (``tl.dot``, fp32 accumulation), an fp32
softmax with the causal / boolean / additive mask, P (in the input dtype, as SDPA's fused
kernels do) @ V, and writes ``[B, Sq, Hq, D]``, so ``out.transpose(1, 2)`` is SDPA's
``[B, Hq, Sq, D]`` and a caller's ``transpose(1, 2).reshape(B, Sq, Hq * D)`` is a view.
Measured in the systems run of a TTS model (11 tokens, 32 sequences x 16 query heads,
2 KV heads, head dim 128, bf16, in a CUDA graph): 5.7 us per call vs 15.9 us for cuDNN's
SDPA and 35 us for FlashAttention (docs/RESEARCH-TRITON.md §4.2-4.3).

Semantics are ``F.scaled_dot_product_attention``'s: ``scale`` defaults to D^-0.5;
``is_causal`` (query length = key length) masks key j > query i; a boolean ``attn_mask`` keeps True,
a floating one is added to the scores, either broadcast to [B, Hq, Sq, Sk]; GQA maps query
head h to key head h // (Hq / Hk) (``enable_gqa``); a fully masked row gives zeros.
Anything else (dropout, fp32, longer sequences, 3-D inputs) goes to SDPA unchanged.

Three ways in, one kernel:

* ``sdpa(q, k, v, ...)``: a drop-in for ``F.scaled_dot_product_attention`` in code you own.
* ``ShortAttentionMode``: a ``torch.overrides.TorchFunctionMode`` that sends every
  supported ``F.scaled_dot_product_attention`` call made inside ``with
  ShortAttentionMode():`` to the kernel, for forward code you do not want to edit (a
  model's decoder in a model-level transform, a layer forward copied into your module).
  Enter it inside the forward: it traces under ``torch.compile(fullgraph=True)`` and runs
  eagerly.
* ``short_attention``: a ``torch.library.custom_op`` (+ ``register_fake``), so compiled
  graphs and CUDA graphs see one opaque op. Eager calls skip the custom-op dispatcher
  (~40 us of host time) and launch through ``CachedLaunch``
  (``kernel_agent.kernels.triton_launch``: the cached ``CompiledKernel``, no JIT binding
  per call), which matters when the module evaluator times eager calls.

``num_warps`` per tile shape and grid size comes from the persistent tuned-config cache
(``kernel_agent.kernels.tuned``): timed once per GPU, library versions and shape class,
then reused by every later evaluation and run (not while a CUDA graph is captured).

The candidate below replaces an attention-core module, ``forward(q, k, v,
attn_mask=None, is_causal=False)`` -> SDPA's output as ``[B, Sq, Hq * D]``. For a whole
attention layer, copy the layer's forward into your module (projections, RoPE, the cache
write) and keep its SDPA call inside ``ShortAttentionMode()`` or call ``sdpa`` there:
the evaluator counts a call of the reference's own forward as a fallback.

Contract: ``build(reference) -> nn.Module`` returning a drop-in replacement.
"""

import hashlib
import inspect
import re

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn
from torch.overrides import TorchFunctionMode

from kernel_agent.kernels import tuned
from kernel_agent.kernels.triton_launch import CachedLaunch

#: GPUs this example runs on (``kernel_agent.gpu_arch.supports``: ``doctor --smoke`` skips
#: it elsewhere and says why).
ARCHS = "sm_80+"
ARCHS_WHY = "bf16 tl.dot (Triton has no MMA below sm_80)"

# One namespace per candidate file (the evaluator names the module after the file's hash).
_NS = re.sub(r"\W", "_", __name__)
#: Longest query / key sequence the single tile takes (registers: 32 x 256 per tile).
MAX_SEQ = 32
MAX_HEAD_DIM = 256
DTYPES = (torch.bfloat16, torch.float16)
#: num_warps candidates timed per (grid size, tile, dtype, mask kind) class.
WARPS = ({"num_warps": 2}, {"num_warps": 4}, {"num_warps": 8})
DEFAULT_WARPS = {"num_warps": 4}


@triton.jit
def _attn_kernel(
    q, k, v, o, m,
    Hq, G, Sq, Sk, D,
    sqb, sqh, sqs, sqd,
    skb, skh, sks, skd,
    svb, svh, svs, svd,
    sob, soh, sos,
    smb, smh, smq, smk,
    scale,
    CAUSAL: tl.constexpr,
    MASK: tl.constexpr,  # 0: none, 1: boolean (True attends), 2: additive
    S: tl.constexpr,
    BD: tl.constexpr,
):  # fmt: skip
    pid = tl.program_id(0)
    b = pid // Hq
    h = pid % Hq
    hk = h // G  # GQA: query heads [hk * G, (hk + 1) * G) share key / value head hk
    rs = tl.arange(0, S)
    rd = tl.arange(0, BD)
    d_ok = rd[None, :] < D
    q_ok = (rs[:, None] < Sq) & d_ok
    k_ok = (rs[:, None] < Sk) & d_ok
    qt = tl.load(
        q + b * sqb + h * sqh + rs[:, None] * sqs + rd[None, :] * sqd, mask=q_ok, other=0.0
    )
    kt = tl.load(
        k + b * skb + hk * skh + rs[:, None] * sks + rd[None, :] * skd, mask=k_ok, other=0.0
    )
    vt = tl.load(
        v + b * svb + hk * svh + rs[:, None] * svs + rd[None, :] * svd, mask=k_ok, other=0.0
    )
    s = tl.dot(qt, tl.trans(kt)) * scale  # [S, S] fp32
    ok = rs[None, :] < Sk
    if CAUSAL:  # SDPA's is_causal: top-left aligned
        ok = ok & (rs[None, :] <= rs[:, None])
    if MASK != 0:
        m_ok = (rs[:, None] < Sq) & (rs[None, :] < Sk)
        m_ptr = m + b * smb + h * smh + rs[:, None] * smq + rs[None, :] * smk
        if MASK == 1:
            ok = ok & (tl.load(m_ptr, mask=m_ok, other=0) != 0)
        else:
            s = s + tl.load(m_ptr, mask=m_ok, other=0.0).to(tl.float32)
    s = tl.where(ok, s, float("-inf"))
    row_max = tl.max(s, axis=1)
    row_max = tl.where(row_max == float("-inf"), 0.0, row_max)  # fully masked row
    p = tl.exp(s - row_max[:, None])
    total = tl.sum(p, axis=1)
    p = p / tl.where(total == 0.0, 1.0, total)[:, None]  # ... gives zeros, not NaN
    out = tl.dot(p.to(vt.dtype), vt)
    tl.store(
        o + b * sob + h * soh + rs[:, None] * sos + rd[None, :],
        out.to(o.dtype.element_ty),
        mask=q_ok,
    )


_launch = CachedLaunch(_attn_kernel)
# Tuned configs belong to this kernel's source: an edited kernel tunes again.
_OP = (
    "short_attention/" + hashlib.sha1(inspect.getsource(_attn_kernel.fn).encode()).hexdigest()[:10]
)
_WARPS_MEMO: dict[tuple, int] = {}  # per process: no cache lookup per call


def _num_warps(key: tuple, run) -> int:
    warps = _WARPS_MEMO.get(key)
    if warps is None:
        programs, s, bd, dtype, kind = key
        shape = {"programs": programs, "S": s, "D": bd, "dtype": dtype, "mask": kind}
        config = tuned.best_config(
            _OP,
            shape,
            WARPS,
            lambda c: tuned.time_ms(lambda: run(c["num_warps"]), warmup=2, reps=10),
            exact=("S", "D", "mask"),
            default=DEFAULT_WARPS,
            source=__name__,
        )
        warps = int(config["num_warps"])
        if not torch.cuda.is_current_stream_capturing():  # captured: the default, untuned
            _WARPS_MEMO[key] = warps
    return warps


def _short_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    causal: bool,
    scale: float,
) -> torch.Tensor:
    """q [B, Hq, Sq, D], k / v [B, Hk, Sk, D] (any strides) -> [B, Sq, Hq, D]."""
    B, Hq, Sq, D = q.shape
    Hk, Sk = k.shape[1], k.shape[2]
    o = torch.empty((B, Sq, Hq, D), device=q.device, dtype=q.dtype)  # fresh, never a view
    if B * Hq == 0:
        return o
    tile = max(16, triton.next_power_of_2(max(Sq, Sk)))
    bd = max(16, triton.next_power_of_2(D))
    if mask is None:
        kind, m, m_strides = 0, q, (0, 0, 0, 0)  # q: an unused pointer
    else:
        m = mask.expand(B, Hq, Sq, Sk)
        kind = 1 if m.dtype == torch.bool else 2
        m_strides = m.stride()
    args = (
        q, k, v, o, m,
        Hq, Hq // Hk, Sq, Sk, D,
        *q.stride(), *k.stride(), *v.stride(),
        o.stride(0), o.stride(2), o.stride(1),
        *m_strides,
        float(scale),
    )  # fmt: skip
    meta = {"CAUSAL": bool(causal), "MASK": kind, "S": tile, "BD": bd}
    grid = (B * Hq,)

    def run(num_warps: int) -> None:
        _launch[grid](*args, **meta, num_warps=num_warps)

    warps = _num_warps((B * Hq, tile, bd, str(q.dtype).removeprefix("torch."), kind), run)
    run(warps)
    return o


# The same launcher as one opaque op for torch.compile / CUDA graphs; the fake only allocates.
short_attention = torch.library.custom_op(
    f"{_NS}::short_attention", _short_attention, mutates_args=()
)


@short_attention.register_fake
def _(q, k, v, mask, causal, scale):
    return q.new_empty((q.shape[0], q.shape[2], q.shape[1], q.shape[3]))


def supported(
    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
):
    """Whether the kernel takes this ``F.scaled_dot_product_attention`` call."""
    if dropout_p != 0.0 or not all(isinstance(t, torch.Tensor) for t in (q, k, v)):
        return False
    if not all(t.dim() == 4 and t.is_cuda and t.dtype == q.dtype for t in (q, k, v)):
        return False
    if q.dtype not in DTYPES:
        return False
    B, Hq, Sq, D = q.shape
    Hk, Sk = k.shape[1], k.shape[2]
    if k.shape != v.shape or k.shape[0] != B or k.shape[3] != D or not 0 < D <= MAX_HEAD_DIM:
        return False
    if Hk == 0 or Hq % Hk or (Hq != Hk and not enable_gqa):
        return False
    if not (0 < Sq <= MAX_SEQ and 0 < Sk <= MAX_SEQ):
        return False
    if is_causal and Sq != Sk:  # top-left aligned in SDPA's docs; not checked here
        return False
    if attn_mask is not None:
        if is_causal or not isinstance(attn_mask, torch.Tensor) or attn_mask.dim() > 4:
            return False  # SDPA rejects a mask with is_causal
        if attn_mask.device != q.device or not (
            attn_mask.dtype == torch.bool or attn_mask.dtype.is_floating_point
        ):
            return False
        full = (B, Hq, Sq, Sk)[4 - attn_mask.dim() :]
        if any(n not in (1, f) for n, f in zip(attn_mask.shape, full, strict=True)):
            return False
    return True


def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    """``F.scaled_dot_product_attention`` with the short-sequence kernel where it applies."""
    if not supported(q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
            scale=scale, enable_gqa=enable_gqa,
        )  # fmt: skip
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    # compiled: one opaque op; eager: the launcher itself (no custom-op dispatch cost)
    launch = short_attention if torch.compiler.is_compiling() else _short_attention
    return launch(q, k, v, attn_mask, bool(is_causal), scale).transpose(1, 2)


_SDPA_ARGS = ("query", "key", "value", "attn_mask", "dropout_p", "is_causal", "scale")


class ShortAttentionMode(TorchFunctionMode):
    """Routes the ``F.scaled_dot_product_attention`` calls made inside it through
    :func:`sdpa`; every other torch call runs unchanged."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is F.scaled_dot_product_attention and len(args) <= len(_SDPA_ARGS):
            bound = dict(zip(_SDPA_ARGS, args, strict=False)) | kwargs
            q, k, v = (bound.pop(name, None) for name in _SDPA_ARGS[:3])
            if q is not None and k is not None and v is not None:
                return sdpa(q, k, v, **bound)
        return func(*args, **kwargs)


class ShortAttentionCore(nn.Module):
    """An attention core: ``forward(q, k, v, attn_mask=None, is_causal=False)`` with
    q [B, Hq, S, D] and k / v [B, Hk, S, D] -> SDPA's output as [B, S, Hq * D]."""

    def forward(self, q, k, v, attn_mask=None, is_causal=False):
        with ShortAttentionMode():  # the same code as the reference, routed
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                is_causal=is_causal,
                enable_gqa=q.shape[1] != k.shape[1],
            )
        return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


def build(reference: nn.Module) -> nn.Module:
    if any(True for _ in reference.parameters()):
        return reference  # not an attention core (it has projections): see the docstring
    return ShortAttentionCore()
