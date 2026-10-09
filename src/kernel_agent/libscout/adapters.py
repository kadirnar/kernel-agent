"""The library scout's adapters (issue #227): one per library and op family.

Available in every environment (torch itself, the CUDA toolkit kernel-agent needs anyway),
run on a GPU by the tests:

* ``torch-sdpa``: ``F.scaled_dot_product_attention`` pinned to one backend (flash,
  mem-efficient, cuDNN, math) with ``torch.nn.attention.sdpa_kernel``;
* ``torch-rms-norm``: the manual RMSNorm pattern (``pow`` / ``mean`` / ``rsqrt``, about six
  kernels) folded into ``F.rms_norm``, torch's fused RMSNorm kernel;
* ``cublaslt``: ``F.linear`` through cuBLASLt with the bias, a residual add or a tanh GELU
  in the GEMM's epilogue and the algorithm chosen per shape (the heuristic's top 8 timed, or
  its i-th), generalised from ``agent/examples/cuda_cublaslt_fp8.py`` to bf16 / fp16 (fp16
  call sites from sm_75, bf16 ones from sm_80: the tensor cores of each dtype);
* ``torch-scaled-mm``: ``F.linear`` as an FP8 W8A8 GEMM (``torch._scaled_mm``; e4m3 weights
  quantised once, activations per call), only on targets planned at ``fp8_w8a8``.

Probed and used when installed (``kernel-agent[libs]``); run on a GPU by the tests when their
library is installed (an NVIDIA A10, sm_86: flash-attn 2.8.1, FlashInfer 0.6.17,
Liger-Kernel 0.8.4):

* ``flash-attn`` (FlashAttention 2, sm_80+);
* ``flashinfer-attention``: one request through FlashInfer's single decode / prefill
  kernels, a batch through its paged wrappers with one page per request (K / V in place,
  each wrapper planned once per call shape); ``flashinfer-norm`` (RMSNorm). FlashInfer
  sampling is listed and skipped: a random draw cannot be compared with the reference's
  (``kernels/distribution.py``: the comparison it needs);
* ``liger-rmsnorm``, ``liger-swiglu``, ``liger-rope`` (Liger-Kernel, Triton; RoPE: the
  rotate-half pattern of q and k with one cos and sin folded into one call,
  ``fx_rewrites.fold_rope``);
* ``gemlite`` (low-bit weight-only GEMV / GEMM, Triton; gemlite 0.6.0.post2): only on
  targets planned at ``int8_weights``, ``fp8_weights`` or (opt-in) ``fp4_weights``, each
  format on the GPUs its kernels compile for (:attr:`Gemlite.FORMATS`).

Written from each library's documentation (``verified`` False: their templates are checked
statically until a GPU run with the library installed; their GPUs were not at hand):

* ``flash-attn-3`` (FlashAttention 3, sm_90);
* ``quack-rmsnorm``, ``quack-softmax`` (QuACK, CuTe DSL; H100, B200, RTX 50).
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, ClassVar

from kernel_agent import gpu_arch
from kernel_agent.libscout.registry import Adapter, Site

_HALF = ("bfloat16", "float16")
_FLOAT = ("bfloat16", "float16", "float32")
#: The wheels torch loads cuBLAS(Lt) and cuDNN from (the CUDA 12 and CUDA 13 names): their
#: versions key a remembered scout with torch's (``Adapter.runtime``)
_CUBLAS = ("nvidia-cublas-cu12", "nvidia-cublas")
_CUDNN = ("nvidia-cudnn-cu12", "nvidia-cudnn-cu13")


def _cap(gpu: Mapping[str, Any]) -> tuple[int, ...] | None:
    cap = gpu.get("capability")
    return tuple(int(c) for c in cap) if cap else None


def _resolve(module: Any, dotted: str) -> Any:
    """``module.a.b.c`` with each submodule imported on the way (None: missing): a package
    does not import its submodules by itself (``liger_kernel`` leaves ``transformers`` to
    an import, measured: the attribute walk alone found no ``liger_rms_norm``)."""
    obj = module
    for part in dotted.split("."):
        found = getattr(obj, part, None)
        name = getattr(obj, "__name__", None)
        if found is None and isinstance(name, str) and hasattr(obj, "__path__"):
            try:
                found = importlib.import_module(f"{name}.{part}")
            except ImportError:
                found = None
        if found is None:
            return None
        obj = found
    return obj


# ------------------------------------------------------------------ attention


class TorchSdpa(Adapter):
    """SDPA backend pinning: the backend torch picks is not always the fastest (cuDNN beat
    flash 16 vs 35 us per call on an 11-token GQA attention, RTX 5070 Ti)."""

    BACKENDS = ("flash", "efficient", "cudnn", "math")

    def configs(self, found: Mapping[str, Site], gpu: Mapping[str, Any]) -> list[dict[str, Any]]:
        cap = _cap(gpu)
        sites = found["sdpa"].sites if "sdpa" in found else []
        half = all(s.get("dtype") in _HALF for s in sites)
        dims = [int(s.get("head_dim") or 0) for s in sites]
        out = []
        for name in self.BACKENDS:
            ampere = half and gpu_arch.supports("sm_80+", cap)
            if name in ("flash", "cudnn") and not ampere:  # half precision, Ampere and newer
                continue
            if name == "flash" and any(d > 256 or d % 8 for d in dims):
                continue
            out.append({"BACKEND": name})
        return out

    CODE = r'''
from torch.nn.attention import SDPBackend, sdpa_kernel

_SDPA = torch.nn.functional.scaled_dot_product_attention
_BACKENDS = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "math": SDPBackend.MATH,
}


def ops(reference=None, BACKEND="cudnn"):
    """F.scaled_dot_product_attention -> the same call with only BACKEND enabled."""
    backends = [_BACKENDS[BACKEND]]

    def sdpa_pinned(*args, **kwargs):
        with sdpa_kernel(backends):
            return _SDPA(*args, **kwargs)

    return {_SDPA: sdpa_pinned}


def rewrite(graph, reference=None, **config):
    return swap_calls(graph, ops(reference, **config))
'''


def _attention_site(site: Mapping[str, Any], *, max_dim: int = 256) -> str | None:
    if site.get("mask"):
        return "attention with an explicit mask"
    dim = int(site.get("head_dim") or 0)
    if not dim or dim > max_dim or dim % 8:
        return f"head size {dim}"
    if site.get("causal") and site.get("seq_q") != site.get("seq_kv"):
        return "causal attention with fewer queries than keys (top-left vs bottom-right mask)"
    return None


_FLASH_TAKES = r'''
_SDPA = torch.nn.functional.scaled_dot_product_attention


def _takes(q, k, v, attn_mask, dropout_p, is_causal, enable_gqa):
    """What the library takes; every other call stays torch's SDPA."""
    return (
        q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.dtype == k.dtype == v.dtype
        and attn_mask is None
        and not dropout_p
        and q.dim() == 4
        and k.dim() == 4
        and v.shape[-1] == q.shape[-1] == k.shape[-1]
        and q.shape[-1] <= 256
        and q.shape[-1] % 8 == 0
        and q.shape[1] % k.shape[1] == 0
        and (k.shape[1] == q.shape[1] or enable_gqa)
        and (not is_causal or q.shape[2] == k.shape[2])
    )
'''


class FlashAttn(Adapter):
    """FlashAttention 2 / 3 for SDPA calls without a mask."""

    def check(self, module: Any) -> str | None:
        return None if hasattr(module, "flash_attn_func") else "no flash_attn_func"

    def sites_ok(self, family: str, site: Mapping[str, Any]) -> str | None:
        return super().sites_ok(family, site) or _attention_site(site)

    CODE = (
        r"""
from flash_attn import flash_attn_func
"""
        + _FLASH_TAKES
        + r'''

def ops(reference=None):
    """F.scaled_dot_product_attention -> flash_attn_func ([B, S, H, D] views)."""

    def sdpa_flash_attn(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
        enable_gqa=False,
    ):
        if not _takes(query, key, value, attn_mask, dropout_p, is_causal, enable_gqa):
            return _SDPA(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                is_causal=is_causal, scale=scale, enable_gqa=enable_gqa,
            )
        out = flash_attn_func(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            softmax_scale=scale,
            causal=is_causal,
        )
        return out.transpose(1, 2)

    return {_SDPA: sdpa_flash_attn}


def rewrite(graph, reference=None, **config):
    return swap_calls(graph, ops(reference, **config))
'''
    )


class FlashAttn3(FlashAttn):
    CODE = (
        r"""
from flash_attn_interface import flash_attn_func
"""
        + _FLASH_TAKES
        + r'''

def ops(reference=None):
    """F.scaled_dot_product_attention -> FlashAttention 3's flash_attn_func."""

    def sdpa_flash_attn3(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
        enable_gqa=False,
    ):
        if not _takes(query, key, value, attn_mask, dropout_p, is_causal, enable_gqa):
            return _SDPA(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                is_causal=is_causal, scale=scale, enable_gqa=enable_gqa,
            )
        out = flash_attn_func(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            softmax_scale=scale,
            causal=is_causal,
        )
        out = out[0] if isinstance(out, tuple) else out  # (out, lse) in some versions
        return out.transpose(1, 2)

    return {_SDPA: sdpa_flash_attn3}


def rewrite(graph, reference=None, **config):
    return swap_calls(graph, ops(reference, **config))
'''
    )


#: FlashInfer's attention kernels are instantiated for these head sizes
FLASHINFER_HEAD_DIMS = (64, 128, 256)


def flashinfer_path(site: Mapping[str, Any]) -> str:
    """The FlashInfer kernel a detected SDPA call site maps onto, from its shapes: one
    request (batch 1) takes the single-request kernels (``single decode``: one query;
    ``single prefill``), a batch the paged wrappers with one page per request (``batch
    decode``: ``BatchDecodeWithPagedKVCacheWrapper``, one query per request; ``batch
    prefill``: ``BatchPrefillWithPagedKVCacheWrapper``), whose page table holds each
    request's KV length (all ``seq_kv`` in a dense SDPA call)."""
    one = int(site.get("batch") or 1) == 1
    decode = int(site.get("seq_q") or 0) == 1
    return f"{'single' if one else 'batch'} {'decode' if decode else 'prefill'}"


class FlashInferAttention(Adapter):
    """FlashInfer's decode / prefill attention: one request through its single-request
    kernels, a batch through the paged wrappers (one page per request: K / V used in place,
    no copy into a page pool), each wrapper planned once per call shape."""

    def check(self, module: Any) -> str | None:
        missing = [
            f
            for f in (
                "single_decode_with_kv_cache",
                "single_prefill_with_kv_cache",
                "BatchDecodeWithPagedKVCacheWrapper",
                "BatchPrefillWithPagedKVCacheWrapper",
            )
            if not hasattr(module, f)
        ]
        return f"flashinfer has no {', '.join(missing)}" if missing else None

    def sites_ok(self, family: str, site: Mapping[str, Any]) -> str | None:
        if int(site.get("head_dim") or 0) not in FLASHINFER_HEAD_DIMS:
            dims = ", ".join(map(str, FLASHINFER_HEAD_DIMS))
            return f"head size {site.get('head_dim')} (FlashInfer: {dims})"
        return super().sites_ok(family, site) or _attention_site(site)

    def configs(self, found: Mapping[str, Site], gpu: Mapping[str, Any]) -> list[dict[str, Any]]:
        # decode: CUDA-core kernels or tensor cores (the prefill kernels over the query heads
        # of each KV head: a large GQA group fills an MMA tile); prefill has one form
        sites = found["sdpa"].sites if "sdpa" in found else []
        taken = [s for s in sites if self.sites_ok("sdpa", s) is None]
        if any(flashinfer_path(s).endswith("decode") for s in taken):
            return [{"TENSOR_CORES": 0}, {"TENSOR_CORES": 1}]
        return [{}]

    CODE = r'''
import flashinfer

_SDPA = torch.nn.functional.scaled_dot_product_attention
_WORKSPACE_MB = 128  # FlashInfer's recommended float workspace (split-KV partial outputs)
#: per device: the float workspace every wrapper shares (they run one after another on the
#: caller's stream) and the batched wrappers, planned once per call shape
_STATE = {}


def _takes(q, k, v, attn_mask, dropout_p, is_causal, enable_gqa):
    """What FlashInfer takes; every other call stays torch's SDPA."""
    return (
        q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.dtype == k.dtype == v.dtype
        and attn_mask is None
        and not dropout_p
        and q.dim() == k.dim() == v.dim() == 4
        and q.shape[0] == k.shape[0] == v.shape[0]
        and q.shape[-1] in (64, 128, 256)
        and k.shape[-1] == v.shape[-1] == q.shape[-1]
        and k.shape[1:3] == v.shape[1:3]
        and q.shape[1] % k.shape[1] == 0
        and (k.shape[1] == q.shape[1] or enable_gqa)
        and (not is_causal or q.shape[2] == k.shape[2])
        and q.shape[2] > 0
        and k.shape[2] > 0
    )


def _pages(k, v):
    """K and V [B, H_kv, S, D] as FlashInfer pages of one request each (page size S), in
    place where the layout allows: ``NHD`` when they are [B, S, H_kv, D] tensors viewed as
    heads (projections, the usual prefill), ``HND`` when contiguous (a KV cache); other
    strides: contiguous copies, HND."""
    kt, vt = k.transpose(1, 2), v.transpose(1, 2)
    if kt.is_contiguous() and vt.is_contiguous():
        return kt, vt, "NHD"
    return k.contiguous(), v.contiguous(), "HND"


def _rows(q):
    """q [B, H, S, D] as FlashInfer's ragged queries [B * S, H, D] (a view when q is a
    [B, S, H, D] projection viewed as heads)."""
    return q.transpose(1, 2).reshape(-1, q.shape[1], q.shape[-1]).contiguous()


def _capturing(t):
    return t.is_cuda and torch.cuda.is_current_stream_capturing()


def _wrapper(kind, q, k, layout, causal, scale, tensor_cores):
    """The batched wrapper of this call shape, planned on its first call: plan() reads the
    page table on the host (a sync), so it runs once per shape (the evaluator's warm-up),
    never in a timed call; None inside a CUDA graph capture before it was planned."""
    b, hq, sq, d = q.shape
    hkv, skv = (k.shape[2], k.shape[1]) if layout == "NHD" else (k.shape[1], k.shape[2])
    state = _STATE.setdefault(q.device, {"plans": {}})
    key = (kind, b, hq, hkv, sq, skv, d, q.dtype, layout, bool(causal), scale, tensor_cores)
    found = state["plans"].get(key)
    if found is not None or _capturing(q):
        return found
    if "workspace" not in state:
        state["workspace"] = torch.empty(_WORKSPACE_MB << 20, dtype=torch.uint8, device=q.device)
    # one page per request: page i is request i's whole KV (page size = its length)
    indptr = torch.arange(b + 1, dtype=torch.int32)
    last = torch.full((b,), skv, dtype=torch.int32)
    types = {"q_data_type": q.dtype, "kv_data_type": q.dtype, "sm_scale": scale}
    if kind == "decode":
        found = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            state["workspace"], layout, use_tensor_cores=tensor_cores
        )
        found.plan(indptr, indptr[:-1], last, hq, hkv, d, skv, **types)
    else:
        found = flashinfer.BatchPrefillWithPagedKVCacheWrapper(state["workspace"], layout)
        found.plan(indptr * sq, indptr, indptr[:-1], last, hq, hkv, d, skv, causal=causal, **types)
    state["plans"][key] = found
    return found


def ops(reference=None, TENSOR_CORES=0):
    """F.scaled_dot_product_attention -> FlashInfer: one request through its single decode
    (one query) / prefill kernels, a batch through BatchDecode / BatchPrefill
    WithPagedKVCacheWrapper (one page per request); TENSOR_CORES: decode on tensor cores."""
    cores = bool(TENSOR_CORES)

    def sdpa_flashinfer(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
        enable_gqa=False,
    ):
        if not _takes(query, key, value, attn_mask, dropout_p, is_causal, enable_gqa):
            return _SDPA(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                is_causal=is_causal, scale=scale, enable_gqa=enable_gqa,
            )
        b, hq, sq, d = query.shape
        k, v, layout = _pages(key, value)
        if b == 1 and sq == 1:
            out = flashinfer.single_decode_with_kv_cache(
                query[0, :, 0].contiguous(), k[0], v[0], kv_layout=layout,
                use_tensor_cores=cores, sm_scale=scale,
            )
            return out.view(1, hq, 1, d)
        if b == 1:
            out = flashinfer.single_prefill_with_kv_cache(
                _rows(query), k[0], v[0], causal=is_causal, kv_layout=layout, sm_scale=scale
            )
            return out.view(1, sq, hq, d).transpose(1, 2)
        kind = "decode" if sq == 1 else "prefill"
        wrapper = _wrapper(kind, query, k, layout, bool(is_causal), scale, cores)
        if wrapper is None:  # first seen inside a CUDA graph capture: plan() cannot run
            return _SDPA(
                query, key, value, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa
            )
        if kind == "decode":
            return wrapper.run(query[:, :, 0].contiguous(), (k, v)).unsqueeze(2)
        return wrapper.run(_rows(query), (k, v)).view(b, sq, hq, d).transpose(1, 2)

    return {_SDPA: sdpa_flashinfer}


def rewrite(graph, reference=None, **config):
    return swap_calls(graph, ops(reference, **config))
'''


# ------------------------------------------------------------------ normalisation


def _norm_site(site: Mapping[str, Any]) -> str | None:
    if not site.get("hidden"):
        return "unknown hidden size"
    return None


#: torch's RMSNorm for a folded pattern (``fx_rewrites.fold_rms_norm``): the reference's
#: roundings (normalised and rounded to ``mid``, the weight multiplied after: one fused
#: kernel and a multiply), or with the weight inside (one kernel, one rounding)
_RMS_TORCH = r'''
def _rms_norm_split(x, weight, eps, dtype, mid=None):
    """F.rms_norm without the weight, rounded where the reference rounds (``mid``), then the
    weight multiply as the reference does it; ``dtype``: the pattern's output dtype."""
    if mid is None:
        return _rms_norm_fused(x, weight, eps, dtype)
    x = x if x.dtype == mid else x.to(mid)
    normed = torch.nn.functional.rms_norm(x, (x.shape[-1],), None, eps)
    out = normed if weight is None else normed * weight
    return out if out.dtype == dtype else out.to(dtype)


def _rms_norm_fused(x, weight, eps, dtype, mid=None):
    """F.rms_norm with the weight inside: one kernel, one rounding."""
    if weight is not None and weight.dtype != x.dtype:
        x = x.to(weight.dtype)
    out = torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, eps)
    return out if out.dtype == dtype else out.to(dtype)
'''


class TorchRmsNorm(Adapter):
    """The manual RMSNorm pattern as ``F.rms_norm`` (one fused kernel instead of ~6)."""

    def sites_ok(self, family: str, site: Mapping[str, Any]) -> str | None:
        if site.get("form") != "manual":
            return "already F.rms_norm (torch's fused kernel)"
        return super().sites_ok(family, site) or _norm_site(site)

    def configs(self, found: Mapping[str, Site], gpu: Mapping[str, Any]) -> list[dict[str, Any]]:
        # FUSE 0: the reference's roundings (a decoder layer whose residual stream cancels
        # fails the tolerance with one rounding fewer, measured); 1: the weight inside too
        return [{"FUSE": 0}, {"FUSE": 1}]

    CODE = (
        _RMS_TORCH
        + r"""

def ops(reference=None, FUSE=0):
    return {}  # a pattern of several ops: nothing to swap one for one


def patterns(reference=None, FUSE=0):
    # what rewrite() puts in a written-out RMSNorm's place (the probe's op bar)
    return {"rms_norm": _rms_norm_fused if FUSE else _rms_norm_split}


def rewrite(graph, reference=None, FUSE=0):
    return fold_rms_norm(graph, _rms_norm_fused if FUSE else _rms_norm_split)
"""
    )


class LibraryRmsNorm(Adapter):
    """RMSNorm through a library kernel (FlashInfer, QuACK, Liger): the manual pattern and
    ``F.rms_norm`` both become one call of it."""

    CALL = ""  # the library call on a 2-D x [rows, hidden] and weight [hidden]
    IMPORT = ""
    ATTR = ""

    def check(self, module: Any) -> str | None:
        if _resolve(module, self.ATTR) is None:
            return f"{self.module} has no {self.ATTR}"
        return None

    def sites_ok(self, family: str, site: Mapping[str, Any]) -> str | None:
        if not site.get("weight"):
            return "RMSNorm without a weight"
        return super().sites_ok(family, site) or _norm_site(site)

    def code(self) -> str:
        return (
            self.IMPORT
            + _RMS_TORCH
            + f'''

def _rms_norm_lib(x, weight, eps, dtype, mid=None):
    """{self.package}'s RMSNorm on x [..., hidden] (2-D rows; the weight inside, one
    rounding); other calls: F.rms_norm."""
    takes = (
        x.is_cuda
        and weight is not None
        and x.dtype in (torch.float16, torch.bfloat16)
        and weight.dtype == x.dtype
        and dtype == x.dtype
        and x.shape[-1] == weight.shape[0]
        and x.numel() > 0
    )
    if not takes:
        return _rms_norm_fused(x, weight, eps, dtype)
    rows = x.reshape(-1, x.shape[-1]).contiguous()
    return ({self.CALL}).reshape(x.shape)


def ops(reference=None):
    return {{}}  # folded from patterns and F.rms_norm by rewrite()


def patterns(reference=None, **config):
    """What rewrite() puts in a written-out RMSNorm's place (the probe's op bar)."""
    return {{"rms_norm": _rms_norm_lib}}


def rewrite(graph, reference=None, **config):
    return fold_rms_norm(graph, _rms_norm_lib)
'''
        )


class FlashInferNorm(LibraryRmsNorm):
    IMPORT = "\nimport flashinfer\n"
    ATTR = "norm.rmsnorm"
    CALL = "flashinfer.norm.rmsnorm(rows, weight, eps)"


class QuackRmsNorm(LibraryRmsNorm):
    IMPORT = "\nimport quack\n"
    ATTR = "rmsnorm"
    CALL = "quack.rmsnorm(rows, weight, eps=eps)"


class LigerRmsNorm(LibraryRmsNorm):
    IMPORT = "\nfrom liger_kernel.transformers.functional import liger_rms_norm\n"
    ATTR = "transformers.functional.liger_rms_norm"
    CALL = 'liger_rms_norm(rows, weight, eps, 0.0, "llama", False)'


class QuackSoftmax(Adapter):
    """Softmax over the last dimension through QuACK (2-D rows)."""

    def check(self, module: Any) -> str | None:
        return None if hasattr(module, "softmax") else "quack has no softmax"

    CODE = r'''
import quack

_SOFTMAXES = (torch.nn.functional.softmax, torch.softmax)


def ops(reference=None):
    """softmax(x, dim=-1) -> quack.softmax on [rows, n]; other calls unchanged."""

    def softmax_quack(input, dim=None, dtype=None, *args, **kwargs):
        last = dim in (-1, input.dim() - 1)
        takes = (
            input.is_cuda
            and last
            and dtype is None
            and not args
            and not kwargs
            and input.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and input.numel() > 0
        )
        if not takes:
            return torch.softmax(input, dim if dim is not None else -1, dtype=dtype)
        rows = input.reshape(-1, input.shape[-1]).contiguous()
        return quack.softmax(rows).reshape(input.shape)

    table = {f: softmax_quack for f in _SOFTMAXES}
    table["softmax"] = softmax_quack  # x.softmax(dim) in a graph
    return table


def rewrite(graph, reference=None, **config):
    return swap_calls(graph, ops(reference, **config))
'''


class LigerSwiGLU(Adapter):
    """``silu(gate) * up`` as Liger's fused SwiGLU kernel."""

    def check(self, module: Any) -> str | None:
        import importlib

        functional = importlib.import_module("liger_kernel.transformers.functional")
        return None if hasattr(functional, "liger_swiglu") else "no liger_swiglu"

    def sites_ok(self, family: str, site: Mapping[str, Any]) -> str | None:
        if not str(site.get("act", "")).startswith("silu"):
            return f"gated MLP with {site.get('act')} (SwiGLU is SiLU)"
        return super().sites_ok(family, site)

    CODE = r'''
from liger_kernel.transformers.functional import liger_swiglu


def _silu_mul(gate, up):
    """Liger's SwiGLU (silu(gate) * up in one kernel) where shapes and dtypes match."""
    takes = (
        gate.is_cuda
        and gate.shape == up.shape
        and gate.dtype == up.dtype
        and gate.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )
    if not takes:
        return torch.nn.functional.silu(gate) * up
    return liger_swiglu(gate.contiguous(), up.contiguous())


def ops(reference=None):
    return {}  # silu(gate) * up: a pattern, folded by rewrite()


def rewrite(graph, reference=None, **config):
    return fold_silu_mul(graph, _silu_mul)
'''


class LigerRope(Adapter):
    """Rotate-half RoPE of q and k as Liger's fused RoPE kernel: one launch for both
    tensors instead of about ten elementwise kernels (``fx_rewrites.fold_rope``)."""

    def check(self, module: Any) -> str | None:
        import importlib

        functional = importlib.import_module("liger_kernel.transformers.functional")
        return None if hasattr(functional, "liger_rope") else "no liger_rope"

    def sites_ok(self, family: str, site: Mapping[str, Any]) -> str | None:
        if not site.get("paired"):
            return (
                "rotary embedding of one tensor (Liger's RoPE rotates q and k with one cos "
                "and sin together)"
            )
        if len(site.get("shape") or []) != 4:
            return f"rotary embedding of a {len(site.get('shape') or [])}-D tensor (takes 4-D)"
        if int(site.get("head_dim") or 0) % 2:
            return f"odd head size {site.get('head_dim')}"
        return super().sites_ok(family, site)

    CODE = r'''
from liger_kernel.transformers.functional import liger_rope


def _rotate(x, cos, sin):
    """The reference's rotate-half RoPE of x."""
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sin


def _table(t, q):
    """cos / sin as Liger takes them: [1 or batch, seq, head_dim] (None: a layout it does
    not take, such as one per head)."""
    if t.dim() == 2:
        t = t.unsqueeze(0)
    elif t.dim() == 4 and t.shape[1] == 1:
        t = t.squeeze(1)
    elif t.dim() != 3 or t.shape[0] != 1:
        return None
    if t.shape[0] not in (1, q.shape[0]) or tuple(t.shape[1:]) != tuple(q.shape[2:]):
        return None
    return t


def _rope(q, k, cos, sin, q_free=False, k_free=False):
    """Liger's RoPE of q [batch, heads, seq, d] and k [batch, kv_heads, seq, d] in one
    kernel; the reference's math where it does not take them. Liger writes its results into
    q's and k's storage: one something else still reads (``q_free`` / ``k_free`` False, from
    the graph) is copied first. It reads the first half of cos and sin only (rotate-half
    tables repeat their frequencies; the evaluator rejects a model whose do not)."""
    c, s = _table(cos, q), _table(sin, q)
    takes = (
        q.is_cuda
        and q.dim() == 4
        and k.dim() == 4
        and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and q.dtype == k.dtype == cos.dtype == sin.dtype
        and q.shape[0] == k.shape[0]
        and q.shape[2:] == k.shape[2:]
        and q.shape[-1] % 2 == 0
        and c is not None
        and s is not None
        and c.shape == s.shape
    )
    if not takes:
        return _rotate(q, cos, sin), _rotate(k, cos, sin)
    q = q if q_free else q.clone()
    k = k if k_free else k.clone()
    return liger_rope(q, k, c, s)


def ops(reference=None):
    return {}  # x * cos + rotate_half(x) * sin of q and k: a pattern, folded by rewrite()


def rewrite(graph, reference=None, **config):
    return fold_rope(graph, _rope)
'''


# ------------------------------------------------------------------ GEMMs


def _gemm_site(site: Mapping[str, Any], multiple: int) -> str | None:
    n, k = int(site.get("n") or 0), int(site.get("k") or 0)
    if n % multiple or k % multiple:
        return f"linear N={n} K={k} (needs multiples of {multiple})"
    return None


class CublasLt(Adapter):
    """``F.linear`` through cuBLASLt: epilogues and a per-shape algorithm choice."""

    #: ``ALGO``: -1 times the heuristic's algorithms per shape and keeps the fastest (the
    #: recipe that took a W8A8 layer from 3.46x to 5.42x); i >= 0 takes the heuristic's i-th.
    #: ``FUSE``: 1 adds a residual and a tanh GELU in the epilogue, 0 only the bias (a fused
    #: residual rounds once where the reference rounds twice: the evaluator decides).
    #: (the heuristic's 2nd and 3rd alone never beat ALGO=-1, which times them per shape, on
    #: the Qwen3-0.6B and VoxCPM2 captures: three configs keep a sweep near a minute)
    CONFIGS = (
        {"ALGO": -1, "FUSE": 1},
        {"ALGO": -1, "FUSE": 0},
        {"ALGO": 0, "FUSE": 0},
    )

    def sites_ok(self, family: str, site: Mapping[str, Any]) -> str | None:
        return super().sites_ok(family, site) or _gemm_site(site, 8)

    def configs(self, found: Mapping[str, Site], gpu: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [dict(c) for c in self.CONFIGS]

    CODE = r'''
import hashlib

from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <algorithm>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

#define LT_CHECK(x)                                                                     \
  do {                                                                                  \
    const cublasStatus_t st_ = (x);                                                     \
    TORCH_CHECK(st_ == CUBLAS_STATUS_SUCCESS, "cublasLt: " #x " returned ", (int)st_); \
  } while (0)
#define DESC_SET(desc, attr, v) \
  LT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_##attr, &(v), sizeof(v)))
#define PREF_SET(pref, attr, v) \
  LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_##attr, &(v), sizeof(v)))

// y [M, N] (row-major) = x [M, K] . w [N, K]^T (+ bias [N]) (+ residual [M, N]) (GELU tanh).
// cuBLASLt's column-major D [N x M] = op(A) . B with A = w (K x N, ld K, transposed) and
// B = x (K x M, ld K); C = the residual (beta 1) or D (beta 0).
struct Plan {
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  std::vector<cublasLtMatmulHeuristicResult_t> algos;
  int pick = 0;
  float us = -1.f;
};

static constexpr size_t kWs = 32u << 20;
static std::mutex g_mu;
static std::map<int, void*> g_ws;  // one fixed workspace per device: graph-capturable
using Key = std::tuple<int, int64_t, int64_t, int64_t, int, int, int, int, int64_t>;
static std::map<Key, Plan> g_plans;

static cublasLtHandle_t handle() {
  static cublasLtHandle_t h = nullptr;
  if (!h) LT_CHECK(cublasLtCreate(&h));
  return h;
}

static void* workspace(int dev) {
  auto it = g_ws.find(dev);
  if (it != g_ws.end()) return it->second;
  void* p = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&p, kWs));
  g_ws[dev] = p;
  return p;
}

static cublasLtEpilogue_t epilogue_of(bool bias, bool gelu) {
  if (bias) return gelu ? CUBLASLT_EPILOGUE_GELU_BIAS : CUBLASLT_EPILOGUE_BIAS;
  return gelu ? CUBLASLT_EPILOGUE_GELU : CUBLASLT_EPILOGUE_DEFAULT;
}

static Plan make_plan(int dev, int64_t M, int64_t N, int64_t K, cudaDataType_t dt, bool bias,
                      bool gelu, uint32_t align) {
  Plan p;
  LT_CHECK(cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  DESC_SET(p.desc, TRANSA, ta);
  DESC_SET(p.desc, TRANSB, tb);
  const cublasLtEpilogue_t epi = epilogue_of(bias, gelu);
  DESC_SET(p.desc, EPILOGUE, epi);
  if (bias) {  // a valid pointer for the heuristic; each call sets its own
    const void* dummy = workspace(dev);
    DESC_SET(p.desc, BIAS_POINTER, dummy);
  }
  LT_CHECK(cublasLtMatrixLayoutCreate(&p.la, dt, K, N, K));
  LT_CHECK(cublasLtMatrixLayoutCreate(&p.lb, dt, K, M, K));
  LT_CHECK(cublasLtMatrixLayoutCreate(&p.lc, dt, N, M, N));
  cublasLtMatmulPreference_t pref;
  LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  const size_t ws = kWs;
  PREF_SET(pref, MAX_WORKSPACE_BYTES, ws);
  PREF_SET(pref, MIN_ALIGNMENT_A_BYTES, align);
  PREF_SET(pref, MIN_ALIGNMENT_B_BYTES, align);
  PREF_SET(pref, MIN_ALIGNMENT_C_BYTES, align);
  PREF_SET(pref, MIN_ALIGNMENT_D_BYTES, align);
  p.algos.resize(8);
  int got = 0;
  const cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(handle(), p.desc, p.la, p.lb, p.lc,
                                                           p.lc, pref, 8, p.algos.data(), &got);
  cublasLtMatmulPreferenceDestroy(pref);
  p.algos.resize(st == CUBLAS_STATUS_SUCCESS ? got : 0);
  p.algos.erase(std::remove_if(p.algos.begin(), p.algos.end(),
                               [](const cublasLtMatmulHeuristicResult_t& r) {
                                 return r.state != CUBLAS_STATUS_SUCCESS || r.workspaceSize > kWs;
                               }),
                p.algos.end());
  return p;
}

static cublasStatus_t matmul(Plan& p, int a, const void* w, const void* x, const void* bias,
                             const void* res, void* d, void* ws, cudaStream_t stream) {
  if (bias) {  // read at launch: a graph capture records this call's pointer
    const cublasLtMatmulDescAttributes_t attr = CUBLASLT_MATMUL_DESC_BIAS_POINTER;
    cublasLtMatmulDescSetAttribute(p.desc, attr, &bias, sizeof(bias));
  }
  const float alpha = 1.f, beta = res ? 1.f : 0.f;
  return cublasLtMatmul(handle(), p.desc, &alpha, w, p.la, x, p.lb, &beta, res ? res : d, p.lc,
                        d, p.lc, &p.algos[a].algo, ws, kWs, stream);
}

// The plan's algorithm: the heuristic's i-th (algo >= 0, clamped), or every one timed
// interleaved (3 rounds of 10 calls on scratch operands, the minimum per algorithm); inside a
// graph capture the heuristic's first, untimed.
static void choose(Plan& p, int dev, int64_t M, int64_t N, int64_t K, at::ScalarType st_type,
                   bool bias, bool res, int64_t algo, cudaStream_t st, bool capturing) {
  const int n = (int)p.algos.size();
  if (algo >= 0 || capturing || n == 1) {
    p.pick = algo >= 0 ? std::min<int>((int)algo, n - 1) : 0;
    return;
  }
  auto opts = at::TensorOptions().device(at::kCUDA, dev).dtype(st_type);
  at::Tensor w = at::full({N, K}, 0.01, opts), x = at::full({M, K}, 0.01, opts);
  at::Tensor d = at::empty({M, N}, opts), r = at::zeros({M, N}, opts), b = at::zeros({N}, opts);
  void* ws = workspace(dev);
  const void* pb = bias ? b.data_ptr() : nullptr;
  const void* pr = res ? r.data_ptr() : nullptr;
  std::vector<float> best(n, 1e30f);
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < 2; ++k)
      if (matmul(p, i, w.data_ptr(), x.data_ptr(), pb, pr, d.data_ptr(), ws, st) !=
          CUBLAS_STATUS_SUCCESS)
        best[i] = -1.f;
  cudaEvent_t e0, e1;
  C10_CUDA_CHECK(cudaEventCreate(&e0));
  C10_CUDA_CHECK(cudaEventCreate(&e1));
  for (int round = 0; round < 3; ++round)
    for (int i = 0; i < n; ++i) {
      if (best[i] < 0.f) continue;
      C10_CUDA_CHECK(cudaEventRecord(e0, st));
      for (int t = 0; t < 10; ++t)
        matmul(p, i, w.data_ptr(), x.data_ptr(), pb, pr, d.data_ptr(), ws, st);
      C10_CUDA_CHECK(cudaEventRecord(e1, st));
      C10_CUDA_CHECK(cudaEventSynchronize(e1));
      float ms = 0.f;
      C10_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
      best[i] = std::min(best[i], 100.f * ms);  // us per call
    }
  C10_CUDA_CHECK(cudaEventDestroy(e0));
  C10_CUDA_CHECK(cudaEventDestroy(e1));
  int arg = -1;
  for (int i = 0; i < n; ++i)
    if (best[i] >= 0.f && (arg < 0 || best[i] < best[arg])) arg = i;
  TORCH_CHECK(arg >= 0, "cublasLt: no algorithm ran for M=", M, " N=", N, " K=", K);
  p.pick = arg;
  p.us = best[arg];
}

static uint32_t alignment(std::initializer_list<const void*> ptrs) {
  uint32_t a = 16;
  for (const void* p : ptrs)
    while (p && a > 2 && (reinterpret_cast<uintptr_t>(p) % a)) a /= 2;
  return a;
}

torch::Tensor lt_linear(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> bias,
                        c10::optional<torch::Tensor> residual, int64_t gelu, int64_t algo) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() && w.is_contiguous(),
              "lt_linear: x [M, K] and w [N, K] contiguous on the GPU");
  TORCH_CHECK(x.scalar_type() == w.scalar_type(), "lt_linear: x and w of one dtype");
  const at::cuda::CUDAGuard guard(x.device());
  const int dev = x.get_device();
  const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(w.size(1) == K, "lt_linear: K mismatch");
  at::Tensor y = at::empty({M, N}, x.options());
  const bool has_bias = bias.has_value() && bias->defined();
  const bool has_res = residual.has_value() && residual->defined();
  const void* pb = has_bias ? bias->data_ptr() : nullptr;
  const void* pr = has_res ? residual->data_ptr() : nullptr;
  if (has_res) TORCH_CHECK(residual->is_contiguous() && residual->numel() == M * N, "residual");
  const cudaDataType_t dt = x.scalar_type() == at::kBFloat16 ? CUDA_R_16BF : CUDA_R_16F;
  const uint32_t align = alignment({x.data_ptr(), w.data_ptr(), y.data_ptr(), pb, pr});
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
  C10_CUDA_CHECK(cudaStreamIsCapturing(st, &cs));
  const bool capturing = cs != cudaStreamCaptureStatusNone;
  const Key key = std::make_tuple(dev, M, N, K, (int)dt, (int)has_bias * 2 + (int)(gelu != 0),
                                  (int)has_res, (int)align, algo);
  std::lock_guard<std::mutex> lock(g_mu);
  TORCH_CHECK(!capturing || g_ws.count(dev),
              "lt_linear: call it once outside CUDA graph capture first (workspace)");
  auto it = g_plans.find(key);
  if (it == g_plans.end()) {
    Plan p = make_plan(dev, M, N, K, dt, has_bias, gelu != 0, align);
    TORCH_CHECK(!p.algos.empty(), "cublasLt has no algorithm for M=", M, " N=", N, " K=", K);
    choose(p, dev, M, N, K, x.scalar_type(), has_bias, has_res, algo, st, capturing);
    it = g_plans.emplace(key, p).first;
  }
  const cublasStatus_t status = matmul(it->second, it->second.pick, w.data_ptr(), x.data_ptr(),
                                       pb, pr, y.data_ptr(), workspace(dev), st);
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasLtMatmul returned ", (int)status);
  return y;
}

// The plans made so far, one row each: M, N, K, epilogue (2 bias + 1 GELU), residual, the
// algo argument, the algorithm run (the heuristic's index), how many it offered, and the
// pick's time in us when they were timed (-1: not timed).
std::vector<std::vector<double>> lt_plans() {
  std::lock_guard<std::mutex> lock(g_mu);
  std::vector<std::vector<double>> rows;
  for (const auto& kv : g_plans) {
    const Key& k = kv.first;
    const Plan& p = kv.second;
    rows.push_back({(double)std::get<1>(k), (double)std::get<2>(k), (double)std::get<3>(k),
                    (double)std::get<5>(k), (double)std::get<6>(k), (double)std::get<8>(k),
                    (double)p.pick, (double)p.algos.size(), (double)p.us});
  }
  return rows;
}
"""

CPP_SRC = (
    "torch::Tensor lt_linear(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> "
    "bias, c10::optional<torch::Tensor> residual, int64_t gelu, int64_t algo);\n"
    "std::vector<std::vector<double>> lt_plans();"
)
_EXT = []


def _ext():
    """The cuBLASLt extension (built once per source: torch's extension cache)."""
    if not _EXT:
        tag = hashlib.sha1(CUDA_SRC.encode()).hexdigest()[:10]
        _EXT.append(
            load_inline(
                name=f"ka_libscout_cublaslt_{tag}",
                cpp_sources=CPP_SRC,
                cuda_sources=CUDA_SRC,
                functions=["lt_linear", "lt_plans"],
                extra_cuda_cflags=["-O3"],
                extra_ldflags=["-lcublasLt"],
            )
        )
    return _EXT[0]


def _lt_takes(x, weight, bias, residual):
    return (
        x.is_cuda
        and x.dtype in (torch.bfloat16, torch.float16)
        and weight.dtype == x.dtype
        and weight.dim() == 2
        and weight.is_contiguous()
        and x.shape[-1] == weight.shape[1]
        and x.shape[-1] % 8 == 0
        and weight.shape[0] % 8 == 0
        and x.numel() > 0
        and (bias is None or (bias.dtype == x.dtype and bias.is_contiguous()))
        and (residual is None or residual.dtype == x.dtype)
    )


def prepare(reference):
    """Build the extension in build(), before anything is timed."""
    if torch.cuda.is_available():
        _ext()


def ops(reference=None, ALGO=-1, FUSE=1):
    """F.linear -> cuBLASLt (with FUSE, ``rewrite`` also folds a residual add or a tanh GELU
    into the epilogue)."""

    def linear_cublaslt(x, weight, bias=None, residual=None, act=None):
        if not _lt_takes(x, weight, bias, residual):
            out = torch.nn.functional.linear(x, weight, bias)
            if act == "gelu":
                out = torch.nn.functional.gelu(out, approximate="tanh")
            return out if residual is None else out + residual
        rows = x.reshape(-1, x.shape[-1]).contiguous()
        res = None
        if residual is not None:
            res = residual.reshape(-1, weight.shape[0]).contiguous()
        out = _ext().lt_linear(rows, weight, bias, res, 1 if act == "gelu" else 0, int(ALGO))
        return out.view(*x.shape[:-1], weight.shape[0])

    if FUSE:  # what an op bar of the GEMM alone leaves out (the probe does not prune on it)
        linear_cublaslt.folds = "a residual add or a tanh GELU into the epilogue"
    return {torch._C._nn.linear: linear_cublaslt}


def bar_details(func, args, kwargs, ALGO=-1, FUSE=1):
    """The algorithm cuBLASLt runs for an op bar's linear call (the probe's): the
    heuristic's i-th of n, and with ALGO=-1 (all n timed per shape) the pick's time."""
    if not _EXT or len(args) < 2:
        return {}
    x, weight = args[0], args[1]
    bias = args[2] if len(args) > 2 else (kwargs or {}).get("bias")
    want = (x.numel() // x.shape[-1], weight.shape[0], weight.shape[1], 2 * (bias is not None))
    for m, n, k, epilogue, res, algo, pick, count, us in _EXT[0].lt_plans():
        if (m, n, k, epilogue, res, algo) == (*want, 0, int(ALGO)):
            found = {"algorithm": int(pick), "algorithms": int(count)}
            if us >= 0:
                found["algorithm_us"] = round(us, 2)
            return found
    return {}


def rewrite(graph, reference=None, **config):
    (linear,) = ops(reference, **config).values()
    fuse = bool(config.get("FUSE", 1))
    return fold_linear(graph, linear, residual=fuse, gelu=fuse)
'''


class TorchScaledMm(Adapter):
    """FP8 W8A8 linears through ``torch._scaled_mm`` (targets planned at ``fp8_w8a8``)."""

    def sites_ok(self, family: str, site: Mapping[str, Any]) -> str | None:
        return super().sites_ok(family, site) or _gemm_site(site, 16)

    def configs(self, found: Mapping[str, Site], gpu: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [{"SCALING": "rowwise"}, {"SCALING": "tensorwise"}]

    CODE = r'''
import weakref

_E4M3 = torch.float8_e4m3fn
_E4M3_MAX = 448.0


def _quantize(t, rowwise):
    """e4m3 codes of t [rows, k] and its fp32 scales: one per row, or one in all."""
    t32 = t.float()
    amax = t32.abs().amax(dim=1, keepdim=True) if rowwise else t32.abs().amax()
    scale = amax.clamp(min=1e-12) / _E4M3_MAX
    return (t32 / scale).clamp(-_E4M3_MAX, _E4M3_MAX).to(_E4M3), scale


_QUANTISED = weakref.WeakKeyDictionary()  # reference -> {rowwise: its table}


def _weights(reference, rowwise):
    """Every 2-D bf16 / fp16 weight of the reference quantised once (per output channel or
    per tensor), keyed by the parameter (what the graph passes in). Once per reference:
    Dynamo recompiles (another input's guards fail) call ``rewrite`` again, inside a CUDA
    graph capture too, where quantising cannot run."""
    if reference is None:
        return {}
    done = _QUANTISED.setdefault(reference, {})
    if rowwise in done:
        return done[rowwise]
    table = done[rowwise] = {}
    for p in reference.parameters():
        if p.dim() == 2 and p.dtype in (torch.bfloat16, torch.float16) and p.is_cuda:
            if p.shape[0] % 16 == 0 and p.shape[1] % 16 == 0:
                codes, scale = _quantize(p.detach(), rowwise)
                table[p] = (codes, scale.reshape(1, -1) if rowwise else scale)
    return table


def ops(reference=None, SCALING="rowwise"):
    """F.linear -> FP8 (e4m3) W8A8 ``torch._scaled_mm``, fp32 accumulation, scales per
    token and output channel (``rowwise``) or per tensor."""
    rowwise = SCALING == "rowwise"
    table = _weights(reference, rowwise)

    def linear_fp8(x, weight, bias=None, residual=None, act=None):
        takes = (
            x.is_cuda
            and x.dtype in (torch.bfloat16, torch.float16)
            and weight in table
            and x.shape[-1] == weight.shape[1]
            and x.numel() > 0
        )
        if not takes:
            out = torch.nn.functional.linear(x, weight, bias)
        else:
            codes, w_scale = table[weight]
            rows = x.reshape(-1, x.shape[-1])
            xq, x_scale = _quantize(rows, rowwise)
            out = torch._scaled_mm(
                xq,
                codes.t(),
                scale_a=x_scale,
                scale_b=w_scale,
                out_dtype=x.dtype,
            )
            out = out.view(*x.shape[:-1], weight.shape[0])
            if bias is not None:
                out = out + bias
        if act == "gelu":
            out = torch.nn.functional.gelu(out, approximate="tanh")
        return out if residual is None else out + residual

    return {torch._C._nn.linear: linear_fp8}


def rewrite(graph, reference=None, **config):
    (linear,) = ops(reference, **config).values()
    return fold_linear(graph, linear)
'''


class Gemlite(Adapter):
    """Weight-only low-bit linears through gemlite's Triton GEMV / GEMM kernels (bf16 /
    fp16 activations; the weights quantised once at build): INT8 or FP8 per output channel
    at ``int8_weights`` / ``fp8_weights`` targets, MXFP4 or NVFP4 at ``fp4_weights`` ones
    (opt-in: ``--precisions ...,fp4_weights``). Only on targets planned at one of these
    precisions, which the run allows."""

    #: target precision -> its formats: (FORMAT, the GPUs gemlite runs it on, why)
    FORMATS: ClassVar[dict[str, tuple[tuple[str, str, str], ...]]] = {
        "int8_weights": (("int8", "sm_80+", "INT8 weights dequantised in registers"),),
        "fp8_weights": (
            ("fp8", "sm_89+", "Triton converts e4m3 from sm_89; sm_86: a compile error"),
        ),
        "fp4_weights": (
            ("mxfp4", "sm_80+", "MXFP4 weights (e8m0 scales) dequantised in registers"),
            ("nvfp4", "sm_89+", "NVFP4's e4m3 scales: from sm_89; sm_86: an error"),
        ),
    }

    def configs(self, found: Mapping[str, Site], gpu: Mapping[str, Any]) -> list[dict[str, Any]]:
        cap = _cap(gpu)
        return [
            {"FORMAT": name}
            for name, spec, _ in self.FORMATS.get(str(gpu.get("precision")), ())
            if gpu_arch.supports(spec, cap)
        ]

    def no_config(self, found: Mapping[str, Site], gpu: Mapping[str, Any]) -> str:
        cap = _cap(gpu)
        here = gpu_arch.arch_of(cap) if cap else "none"
        why = [
            f"{name} weights need {spec} ({reason}); this GPU is {here}"
            for name, spec, reason in self.FORMATS.get(str(gpu.get("precision")), ())
        ]
        return "; ".join(why) or super().no_config(found, gpu)

    def sites_ok(self, family: str, site: Mapping[str, Any]) -> str | None:
        return super().sites_ok(family, site) or _gemm_site(site, 32)

    CODE = r'''
import types
import weakref

from gemlite.helper import A16W4_MXFP, A16W4_NVFP, A16W8_FP8, A16W8_INT8

_FORMATS = {"int8": A16W8_INT8, "fp8": A16W8_FP8, "mxfp4": A16W4_MXFP, "nvfp4": A16W4_NVFP}
_GROUP = {"int8": 1, "fp8": 1, "mxfp4": 32, "nvfp4": 16}  # gemlite's scale groups along K


_QUANTISED = weakref.WeakKeyDictionary()  # reference -> {FORMAT: its layers}


def _layers(reference, FORMAT):
    """Every 2-D bf16 / fp16 CUDA weight of the reference quantised once into a gemlite
    layer, keyed by the parameter (what the graph passes in); gemlite needs K a multiple
    of 32 and of its group, N of 32. Once per reference: Dynamo recompiles call
    ``rewrite`` again, inside a CUDA graph capture too, where packing cannot run."""
    if reference is None:
        return {}
    done = _QUANTISED.setdefault(reference, {})
    if FORMAT in done:
        return done[FORMAT]
    table = done[FORMAT] = {}
    make = _FORMATS[FORMAT]
    for p in reference.parameters():
        if p.dim() != 2 or p.dtype not in (torch.bfloat16, torch.float16) or not p.is_cuda:
            continue
        n, k = p.shape
        if k % 32 or k % _GROUP[FORMAT] or n % 32:
            continue
        stand_in = types.SimpleNamespace(weight=p.detach(), bias=None)  # gemlite's Linear
        quantiser = make(device=str(p.device), dtype=p.dtype)
        table[p] = quantiser.from_linear(stand_in, del_orig=False)
    return table


def ops(reference=None, FORMAT="int8"):
    """F.linear -> gemlite's low-bit weight GEMV / GEMM (the reference's bias, activation
    and residual stay its own ops)."""
    table = _layers(reference, FORMAT)

    def linear_gemlite(x, weight, bias=None, residual=None, act=None):
        layer = table.get(weight)
        takes = (
            layer is not None
            and x.is_cuda
            and x.dtype == weight.dtype
            and x.shape[-1] == weight.shape[1]
            and x.numel() > 0
        )
        out = layer(x) if takes else torch.nn.functional.linear(x, weight)
        if bias is not None:
            out = out + bias
        if act == "gelu":
            out = torch.nn.functional.gelu(out, approximate="tanh")
        return out if residual is None else out + residual

    return {torch._C._nn.linear: linear_gemlite}


def rewrite(graph, reference=None, **config):
    (linear,) = ops(reference, **config).values()
    return fold_linear(graph, linear, residual=False, gelu=False)
'''


# ------------------------------------------------------------------ the registry


ADAPTERS: tuple[Adapter, ...] = (
    TorchSdpa(
        name="torch-sdpa",
        title="SDPA pinned to one backend (torch.nn.attention.sdpa_kernel)",
        package="torch",
        module="torch",
        licence="BSD-3-Clause",
        families=("sdpa",),
        dtypes=_FLOAT,
        runtime=_CUDNN,
    ),
    TorchRmsNorm(
        name="torch-rms-norm",
        title="manual RMSNorm pattern as F.rms_norm (torch's fused kernel)",
        package="torch",
        module="torch",
        licence="BSD-3-Clause",
        families=("rms_norm",),
        dtypes=_FLOAT,
        helpers=("fold_rms_norm",),
    ),
    CublasLt(
        name="cublaslt",
        title="F.linear through cuBLASLt (epilogue fusion, per-shape algorithm choice)",
        package="torch",
        module="torch",
        licence="BSD-3-Clause (torch); cuBLASLt: NVIDIA CUDA EULA",
        families=("linear",),
        archs="sm_75+",
        archs_why="fp16 tensor cores",
        dtype_archs={"bfloat16": ("sm_80+", "bf16 tensor cores")},
        compiles=True,
        helpers=("fold_linear",),
        runtime=_CUBLAS,
    ),
    TorchScaledMm(
        name="torch-scaled-mm",
        title="F.linear as FP8 W8A8 torch._scaled_mm",
        package="torch",
        module="torch",
        licence="BSD-3-Clause",
        families=("linear",),
        archs="sm_89+",
        archs_why="FP8 (e4m3) tensor cores",
        precisions=("fp8_w8a8",),
        helpers=("fold_linear",),
        runtime=_CUBLAS,
    ),
    FlashAttn(
        name="flash-attn",
        title="SDPA as FlashAttention 2",
        package="flash-attn",
        module="flash_attn",
        licence="BSD-3-Clause",
        families=("sdpa",),
        archs="sm_80+",
        archs_why="FlashAttention 2: Ampere and newer",
        min_version="2.0",
    ),
    FlashAttn3(
        name="flash-attn-3",
        title="SDPA as FlashAttention 3",
        package="flash-attn-3",
        module="flash_attn_interface",
        licence="BSD-3-Clause",
        families=("sdpa",),
        archs="sm_90",
        archs_why="FlashAttention 3: Hopper (wgmma, TMA)",
        verified=False,
    ),
    FlashInferAttention(
        name="flashinfer-attention",
        title="SDPA as FlashInfer's decode / prefill kernels (one request, or a batch through "
        "the paged wrappers)",
        package="flashinfer-python",
        module="flashinfer",
        licence="Apache-2.0",
        families=("sdpa",),
        archs="sm_75+",
        archs_why="FlashInfer: Turing and newer",
    ),
    FlashInferNorm(
        name="flashinfer-norm",
        title="RMSNorm as FlashInfer's rmsnorm",
        package="flashinfer-python",
        module="flashinfer",
        licence="Apache-2.0",
        families=("rms_norm",),
        archs="sm_75+",
        archs_why="FlashInfer: Turing and newer",
        helpers=("fold_rms_norm",),
    ),
    Adapter(
        name="flashinfer-sampling",
        title="top-k / top-p sampling as FlashInfer's sampling kernels",
        package="flashinfer-python",
        module="flashinfer",
        licence="Apache-2.0",
        families=("sampling",),
        verified=False,
        no_template=(
            "sampling draws random numbers: the evaluator compares outputs with the "
            "reference's own draws (kernels/distribution.py's distributional comparison is "
            "not wired into it yet)"
        ),
    ),
    QuackRmsNorm(
        name="quack-rmsnorm",
        title="RMSNorm as QuACK's rmsnorm (CuTe DSL)",
        package="quack-kernels",
        module="quack",
        licence="Apache-2.0",
        families=("rms_norm",),
        archs="sm_90,sm_10x,sm_12x",
        archs_why="QuACK lists H100, B200 / B300 and RTX 50",
        verified=False,
        helpers=("fold_rms_norm",),
        runtime=("nvidia-cutlass-dsl",),
    ),
    QuackSoftmax(
        name="quack-softmax",
        title="softmax as QuACK's softmax (CuTe DSL)",
        package="quack-kernels",
        module="quack",
        licence="Apache-2.0",
        families=("softmax",),
        archs="sm_90,sm_10x,sm_12x",
        archs_why="QuACK lists H100, B200 / B300 and RTX 50",
        dtypes=_FLOAT,
        verified=False,
        runtime=("nvidia-cutlass-dsl",),
    ),
    LigerRmsNorm(
        name="liger-rmsnorm",
        title="RMSNorm as Liger-Kernel's RMSNorm (Triton)",
        package="liger-kernel",
        module="liger_kernel",
        licence="BSD-2-Clause",
        families=("rms_norm",),
        archs="sm_80+",
        archs_why="Liger-Kernel's Triton kernels: Ampere and newer",
        helpers=("fold_rms_norm",),
        runtime=("triton",),
    ),
    LigerSwiGLU(
        name="liger-swiglu",
        title="silu(gate) * up as Liger-Kernel's SwiGLU (Triton)",
        package="liger-kernel",
        module="liger_kernel",
        licence="BSD-2-Clause",
        families=("gated_mlp",),
        archs="sm_80+",
        archs_why="Liger-Kernel's Triton kernels: Ampere and newer",
        dtypes=_FLOAT,
        helpers=("fold_silu_mul",),
        runtime=("triton",),
    ),
    LigerRope(
        name="liger-rope",
        title="rotary embedding of q and k as Liger-Kernel's RoPE (Triton)",
        package="liger-kernel",
        module="liger_kernel",
        licence="BSD-2-Clause",
        families=("rotary",),
        archs="sm_80+",
        archs_why="Liger-Kernel's Triton kernels: Ampere and newer",
        dtypes=_FLOAT,
        verified=True,  # tests: the gpu tests (an A10, liger-kernel 0.8.4)
        helpers=("fold_rope",),
        runtime=("triton",),
    ),
    Gemlite(
        name="gemlite",
        title="low-bit weight GEMV / GEMM (gemlite)",
        package="gemlite",
        module="gemlite",
        licence="Apache-2.0",
        families=("linear",),
        archs="sm_80+",
        archs_why="gemlite's Triton kernels: Ampere and newer (per format: Gemlite.FORMATS)",
        precisions=("fp8_weights", "int8_weights", "fp4_weights"),
        verified=True,  # tests: the gpu tests (an A10, gemlite 0.6.0.post2)
        helpers=("fold_linear",),
        runtime=("triton",),
    ),
)

BY_NAME = {a.name: a for a in ADAPTERS}


def packages() -> dict[str, tuple[str, str]]:
    """``{package: (import module, licence)}`` of every adapter's library."""
    out: dict[str, tuple[str, str]] = {}
    for a in ADAPTERS:
        out.setdefault(a.package, (a.module.split(".")[0], a.licence))
    return out
