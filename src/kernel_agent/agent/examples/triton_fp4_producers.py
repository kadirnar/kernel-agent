"""Example candidate (Triton): producers that emit NVFP4 (W4A4). RMSNorm and ``silu(gate) * up``
write packed e2m1 codes, e4m3 block scales (straight into the swizzled layout the W4A4 GEMMs
read) and the per-token fp32 outer scale in their epilogue, so the next W4A4 GEMM reads half a
byte per activation and no separate quantisation pass runs: the FP4 analogue of
``triton_fp8_producers.py`` (issue #233).

Reduced precision, opt-in: only for a target whose spec says ``"precision": "fp4_w4a4"`` (a
run whose ``--precisions`` names it; skill fp4-w4a4), checked in the ``near-lossless-fp4a``
tier. The producer computes its output as eager does (bf16 roundings included), then
quantises it per token, bit for bit ``quant.quantize_fp4_activations(h, "nvfp4")`` of that
output (``h``):

* ``outer = amax(|h_row|) * NVFP4_OUTER_STEP`` (1 for a row of zeros), the block scale of each
  16 elements ``e4m3(min(bmax / (outer * 6), 448))`` (an IEEE division, round to nearest
  even), codes ``e2m1(h / (scale * outer))`` to nearest even, saturated at ±6, two per byte
  (the even element in the low nibble); a block whose scale is 0 gets codes 0;
* ``mx=True``: MXFP4 (``quantize_fp4_activations(h, "mxfp4")``): an e8m0 scale per 32,
  ``2^ceil(log2(bmax / 6))`` from the exponent bits, outer 1 (for an MXFP4 consumer:
  ``tl.dot_scaled`` on e2m1, CuTe ``MmaMXF4Op``);
* scales row-major ``[rows, K / block]`` or ``swizzle=True``: cuBLASLt's 128 x 4 blocks
  (``quant.swizzle_fp4_scales`` / ``mx_scale_offset``: ``F.scaled_mm``'s ``SWIZZLE_32_4_4``,
  CuTe's ``tile_atom_to_shape_SF``), padding rows and columns written as 0 by the kernel.

No FP8 type is needed: the e4m3 scale bits come from integer arithmetic on the fp32 bits
(Triton has no ``fp8e4nv`` before sm_89: "type fp8e4nv not supported in this architecture"
on an A10), the e8m0 ones from the exponent field. So the producers run on any GPU Triton
supports (``ARCHS_COMPILES``), checked bit for bit against the reference on an NVIDIA A10
(sm_86): codes, scales and outer scales of real Qwen3-0.6B activations and of rows at
scales from 1e-30 to 1e30 (subnormal block scales, zero blocks and rows); on the CPU, Triton's
interpreter runs the same kernels (``tests/test_fp4_producers.py``).

A per-token outer scale needs the whole row (its amax), which a row-wise producer has; an
epilogue tile of a GEMM does not: there emit MXFP4 (no outer scale; blocks of 32 inside the
tile) or write bf16 and let the next producer quantise (as below: the gate|up GEMM writes
bf16, the SiLU-mul quantises for ``down_proj``). The optional Hadamard rotation is not fused
here: ``quant.hadamard_rotate`` is an fp32 matmul (BLAS order), so a fused rotation could
match it only to rounding, not bit for bit, and it made NVFP4 worse on the captures measured
(skill fp4-w4a4).

* :func:`rmsnorm_fp4` (``x * rsqrt(mean(x^2) + eps)`` in fp32, cast to the input dtype,
  times the weight in that dtype: Llama / Qwen / Mistral RMSNorm), :func:`silu_mul_fp4`
  (``F.silu(gate) * up`` from separate or merged ``[..., 2F]`` gate|up), :func:`quantize_rows`
  (the plain pass, for inputs that arrive in bf16; :func:`quantize_activations` is it for
  the scale-rule guard). One program per row, the row in registers (K <= :data:`MAX_K`).
  ``out``: also write the producer's bf16 output (a consumer kept in 8 bits by the W4A4 mix,
  or a check of the quantisation on the kernel's own values).
* ``build()`` takes a gated MLP with ``gate_proj`` / ``up_proj`` / ``down_proj`` Linears (no
  bias) and a SiLU ``act_fn``: gate|up merged into one W4A4 GEMM, the SiLU-mul producer emits
  NVFP4 for ``down_proj``. The GEMMs on block-scaled FP4 tensor cores (sm_100+) are
  ``F.scaled_mm`` NVFP4 (``BlockWise1x16`` with fp32 output, then the per-token outer scale
  and the weight's tensor scale once per output: ``quant.fp4_w4a4_linear``'s fast path; on
  sm_120 swap in ``cute_nvfp4_w4a4_gemm.py``'s GEMM, which applies them in its epilogue);
  elsewhere the same math with the GEMMs emulated in fp32 from the codes the producers wrote
  (a check of the producers on any GPU, not a speed-up). The MLP's own input arrives in bf16
  (its RMSNorm sits outside the module), so it takes :func:`quantize_rows`; in a decoder
  layer target call :func:`rmsnorm_fp4` instead.

``unbiased=True`` (opt-in, the producers and ``build()``): the norm-bias correction of
``quant.fp4_bias_correction``: the producers also write the epilogue's per-token scale ``outer
* sum h^2 / sum h h^`` (fourth output; within 4e-7 of the reference on Qwen3-0.6B, not bit for
bit: fp32 sums), the weights get a factor per output channel in their tensor scale. It takes
back the ~1 % e2m1 shrinks a GEMM's output by (skill fp4-w4a4: when to use it).

Measured on an NVIDIA A10 (sm_86, CUDA graphs, us per call; ``docs/research-scripts/w4a4-233/
bench_producers.py``), the producer writing bf16 / fused NVFP4 (swizzled scales) / with the bias
factor / bf16 then a separate ``quantize_rows`` pass: RMSNorm 352 x 1024 2.7 / 5.9 / 6.9 / 7.6,
4096 x 1024 35.4 / 32.9 / 39.6 / 60.3; ``silu(gate) * up`` 352 x 3072 7.7 / 18.2 / 20.0 / 27.6,
4096 x 3072 156 / 128 / 173 / 255. Fusing beats the separate pass 1.3-2.0x everywhere; at M =
352 the quantiser's per-element IEEE division and e2m1 rounding are what cost (a single wave of
programs: latency), at 4096 rows the fused producer beats even the bf16 one (half a byte
written per element). The FP4 GEMMs did not run there (no FP4 tensor cores): the
``F.scaled_mm`` path is the reference's own (``fp4_w4a4_linear``, verified on an RTX 5070 Ti);
run ``pytest -m gpu tests/test_fp4_producers.py`` on sm_100 / sm_120.
"""

import re

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn

from kernel_agent.kernels.quant import (
    NVFP4_OUTER_STEP,
    fp4_error,
    fp4_values,
    fp4_w4a4_linear,
    quantize_fp4,
    quantize_fp4_activations,
    swizzle_fp4_scales,
)

#: GPUs this example runs on as a W4A4 candidate (``kernel_agent.gpu_arch.supports``:
#: ``doctor --smoke`` skips it elsewhere and says why): its consumer GEMMs.
ARCHS = "sm_100+"
ARCHS_WHY = (
    "its W4A4 GEMMs need block-scaled FP4 tensor cores (F.scaled_mm BlockWise1x16); the "
    "producers themselves run on every GPU of ARCHS_COMPILES (A10, sm_86: bit for bit)"
)
#: GPUs its Triton producers compile and run on (no FP8 type: e4m3 bits by integer math).
ARCHS_COMPILES = "sm_75+"

# One namespace per candidate file (the evaluator names the module after the file's hash).
_NS = re.sub(r"\W", "_", __name__)

#: Longest row one program keeps in registers (a longer row needs a two-pass kernel).
MAX_K = 16384
# constants the kernels read (Triton reads globals only as tl.constexpr)
_E4M3_MIN_NORMAL = tl.constexpr(0.015625)  # 2^-6
_E4M3_SUBNORMAL_STEP = tl.constexpr(0.001953125)  # 2^-9
_E4M3_REBIAS = tl.constexpr((127 - 7) << 3)  # fp32 exponent bias to e4m3's, past 3 mantissa bits
_FP4 = "float4_e2m1fn_x2"  # torch's packed e2m1 dtype (torch 2.8+)


@triton.jit
def _e4m3_rn(r):
    # e4m3 bits (int32, 0..126) of fp32 0 <= r <= 448, rounded to nearest even: what
    # .to(tl.float8e4nv) (cvt.rn.satfinite.e4m3x2.f32) gives on sm_89+, by integer math so
    # that it runs on every GPU. Normal (r >= 2^-6): keep 3 of fp32's 23 mantissa bits to
    # nearest even (a carry moves into the exponent) and rebias the exponent 127 -> 7.
    bits = r.to(tl.int32, bitcast=True)
    normal = ((bits + 0x7FFFF + ((bits >> 20) & 1)) >> 20) - _E4M3_REBIAS
    # subnormal (r < 2^-6): steps of 2^-9, r * 512 (exact) to the nearest even integer
    y = r * 512.0
    whole = y.to(tl.int32)  # floor: y >= 0
    frac = y - whole.to(tl.float32)  # exact
    up = (frac > 0.5) | ((frac == 0.5) & ((whole & 1) == 1))
    return tl.where(r >= _E4M3_MIN_NORMAL, normal, whole + up.to(tl.int32))


@triton.jit
def _e4m3_value(code):
    # the fp32 value of e4m3 bits 0..126 (exact)
    normal = ((code + _E4M3_REBIAS) << 20).to(tl.float32, bitcast=True)
    return tl.where(code >= 8, normal, code.to(tl.float32) * _E4M3_SUBNORMAL_STEP)


@triton.jit
def _e2m1(g, step):
    # (codes, quotients): e2m1 codes (int32, sign in bit 3) of g / step (an IEEE division)
    # to nearest even, saturated at ±6, and the quotients themselves; 0 where step is 0. The
    # midpoints 0.25 / 1.25 / 2.5 / 5 round down, 0.75 / 1.75 / 3.5 up (to the even code).
    # The division is the reference's, div_rn: about a third of the kernel at M = 352 on an
    # A10 (g * (1 / step) is faster and gives other codes).
    safe = tl.where(step > 0, step, 1.0)[:, None] + tl.zeros_like(g)
    q = tl.where(step[:, None] > 0, tl.math.div_rn(g, safe), 0.0)
    v = tl.minimum(tl.maximum(q, -6.0), 6.0)
    mag = tl.abs(v)
    idx = (mag > 0.25).to(tl.int32) + (mag >= 0.75).to(tl.int32) + (mag > 1.25).to(tl.int32)
    idx += (mag >= 1.75).to(tl.int32) + (mag > 2.5).to(tl.int32) + (mag >= 3.5).to(tl.int32)
    idx += (mag > 5.0).to(tl.int32)
    return idx + tl.where((v < 0) & (idx > 0), 8, 0), q  # no negative zero


@triton.jit
def _tile(MX: tl.constexpr, BLOCK: tl.constexpr):
    # a row's columns as [BLOCK / SF, SF]: blocks of SF consecutive elements (NVFP4 16, MXFP4
    # 32), 2-D from the load on (vector loads along a block; its maximum a short reduction)
    SF: tl.constexpr = 32 if MX else 16
    return tl.arange(0, BLOCK // SF)[:, None] * SF + tl.arange(0, SF)[None, :]


@triton.jit
def _emit_fp4(
    g,
    row,
    live,
    K,
    q,
    s,
    o,
    e,
    outer_step,
    MX: tl.constexpr,
    SWIZZLE: tl.constexpr,
    UNBIASED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # g: the producer's fp32 row as [BLOCK / SF, SF] (_tile; values already rounded as eager
    # rounds them), 0 past K. Writes packed codes q[row, :K / 2], the row's block scales into
    # s, its outer scale o[row] and (UNBIASED) the epilogue's scale e[row]. Rows past M (live
    # false: the 128-row padding of the swizzled layout) write zero scales only.
    SF: tl.constexpr = 32 if MX else 16
    NB: tl.constexpr = BLOCK // SF
    bmax = tl.max(tl.abs(g), 1)
    if MX:
        # 2^ceil(log2(bmax / 6)) from the bits of bmax = 1.f * 2^(E - 127): E - 129, plus one
        # when 1.f > 1.5; zero / subnormal maxima -127 (quant.quantize_fp4_activations)
        bits = bmax.to(tl.int32, bitcast=True)
        field = (bits >> 23) & 0xFF
        e = field - 129 + ((bits & 0x7FFFFF) > 0x400000).to(tl.int32)
        e = tl.where(field == 0, -127, e)
        tiny = tl.full(e.shape, 0x400000, tl.int32).to(tl.float32, bitcast=True)  # 2^-127
        step = tl.where(e >= -126, ((e + 127) << 23).to(tl.float32, bitcast=True), tiny)
        byte = e + 127
        outer = 1.0
    else:
        amax = tl.max(bmax, 0)
        outer = tl.where(amax > 0, amax * outer_step, 1.0)
        ratio = tl.math.div_rn(bmax, tl.zeros_like(bmax) + outer * 6.0)
        byte = _e4m3_rn(tl.minimum(ratio, 448.0))
        step = _e4m3_value(byte) * outer
    code, v = _e2m1(g, step)
    lo, hi = tl.split(tl.reshape(code, (NB, SF // 2, 2)))
    pcols = tl.arange(0, NB)[:, None] * (SF // 2) + tl.arange(0, SF // 2)[None, :]
    packed = (lo | (hi << 4)).to(tl.uint8)  # the even element in the low nibble
    tl.store(q + row * (K // 2) + pcols, packed, mask=(pcols < K // 2) & live)
    blk = tl.arange(0, NB)
    cols = K // SF
    if SWIZZLE:  # cuBLASLt's 128 x 4 blocks (quant.mx_scale_offset), padding written as 0
        ncb = (cols + 3) // 4
        r = row % 128
        off = ((row // 128) * ncb + blk // 4) * 512 + (r % 32) * 16 + (r // 32) * 4 + blk % 4
        byte = tl.where(live & (blk < cols), byte, 0)
        tl.store(s + off, byte.to(tl.uint8), mask=blk < ncb * 4)
    else:  # row-major [rows, K / SF]
        tl.store(s + row * cols + blk, byte.to(tl.uint8), mask=(blk < cols) & live)
    tl.store(o + row, outer, mask=live)
    if UNBIASED:
        # outer * sum h^2 / sum h h^ (quant.fp4_bias_correction; 1 x outer for a row of
        # zeros), from the quotients v = h / step and their e2m1 values, each block weighted
        # by (step / the row's largest step)^2: the same ratio, no overflow at any scale
        # (blocks whose step is 0 drop out: below 2^-20 of the row's energy)
        idx = code & 7  # e2m1 magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6
        mag = tl.where(idx <= 4, idx.to(tl.float32) * 0.5, (idx - 2).to(tl.float32))
        val = tl.where(code >= 8, -1.0, 1.0) * tl.where(idx == 7, 6.0, mag)
        top = tl.max(step, 0)
        wb = step / tl.where(top > 0, top, 1.0)
        wb = wb * wb
        num = tl.sum(wb * tl.sum(v * v, 1), 0)
        den = tl.sum(wb * tl.sum(v * val, 1), 0)
        c = tl.where(den > 0, num / tl.where(den > 0, den, 1.0), 1.0)
        tl.store(e + row, outer * c, mask=live)


@triton.jit
def _quant_rows_kernel(
    x,
    q,
    s,
    o,
    e,
    M,
    K,
    ldx,
    outer_step,
    MX: tl.constexpr,
    SWIZZLE: tl.constexpr,
    UNBIASED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    live = row < M
    cols = _tile(MX, BLOCK)
    h = tl.load(x + row * ldx + cols, mask=(cols < K) & live, other=0.0).to(tl.float32)
    _emit_fp4(h, row, live, K, q, s, o, e, outer_step, MX, SWIZZLE, UNBIASED, BLOCK)


@triton.jit
def _rmsnorm_fp4_kernel(
    x,
    w,
    q,
    s,
    o,
    e,
    out,
    M,
    K,
    ldx,
    eps,
    outer_step,
    MX: tl.constexpr,
    SWIZZLE: tl.constexpr,
    UNBIASED: tl.constexpr,
    WRITE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    live = row < M
    cols = _tile(MX, BLOCK)
    mask = (cols < K) & live
    dt = x.dtype.element_ty
    v = tl.load(x + row * ldx + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.rsqrt(tl.sum(tl.sum(v * v, 1), 0) / K + eps)
    hn = (v * r).to(dt).to(tl.float32)  # .to(input_dtype), as eager
    wv = tl.load(w + cols, mask=cols < K, other=0.0).to(tl.float32)
    h = (hn * wv).to(dt)  # weight * h in the input dtype
    if WRITE:
        tl.store(out + row * K + cols, h, mask=mask)
    hf = h.to(tl.float32)
    _emit_fp4(hf, row, live, K, q, s, o, e, outer_step, MX, SWIZZLE, UNBIASED, BLOCK)


@triton.jit
def _silu_mul_fp4_kernel(
    g,
    u,
    q,
    s,
    o,
    e,
    out,
    M,
    K,
    ldg,
    ldu,
    outer_step,
    MX: tl.constexpr,
    SWIZZLE: tl.constexpr,
    UNBIASED: tl.constexpr,
    WRITE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    live = row < M
    cols = _tile(MX, BLOCK)
    mask = (cols < K) & live
    dt = g.dtype.element_ty
    gv = tl.load(g + row * ldg + cols, mask=mask, other=0.0).to(tl.float32)
    uv = tl.load(u + row * ldu + cols, mask=mask, other=0.0).to(tl.float32)
    a = (gv / (1.0 + tl.exp(-gv))).to(dt).to(tl.float32)  # F.silu in the input dtype
    h = (a * uv).to(dt)  # * up in the input dtype
    if WRITE:
        tl.store(out + row * K + cols, h, mask=mask)
    hf = h.to(tl.float32)
    _emit_fp4(hf, row, live, K, q, s, o, e, outer_step, MX, SWIZZLE, UNBIASED, BLOCK)


def _rows(t: torch.Tensor) -> torch.Tensor:
    t2 = t.reshape(-1, t.shape[-1])
    return t2 if t2.stride(-1) == 1 else t2.contiguous()


def _outputs(
    m: int, k: int, device: torch.device, mx: bool, swizzle: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """(codes uint8 [m, k / 2], scales uint8 [m, k / block] or flat swizzled (rows padded to
    128, columns to 4), outer fp32 [m], programs)."""
    block = 32 if mx else 16
    if k % block or k > MAX_K:
        raise ValueError(f"{'MXFP4' if mx else 'NVFP4'} needs K % {block} == 0, K <= {MAX_K}")
    q = torch.empty((m, k // 2), device=device, dtype=torch.uint8)
    o = torch.empty((m,), device=device, dtype=torch.float32)
    if not swizzle:
        return q, torch.empty((m, k // block), device=device, dtype=torch.uint8), o, m
    rows = triton.cdiv(m, 128) * 128
    size = rows * triton.cdiv(k // block, 4) * 4
    return q, torch.empty((size,), device=device, dtype=torch.uint8), o, rows


def _launch(kernel, args, k: int, programs: int, mx: bool, swizzle: bool, **consts) -> None:
    # at least 4 scale columns per row: the swizzled layout's column padding
    block = max(triton.next_power_of_2(k), 128 if mx else 64)
    # the bias factor keeps every quotient alive until its sums (128 registers for a row of
    # 4096 on 4 warps): 8 warps from there (an A10: 21 instead of 27 us at 352 x 4096)
    warps = 4 if block <= (2048 if consts.get("UNBIASED") else 4096) else 8
    kernel[(programs,)](
        *args, NVFP4_OUTER_STEP, MX=mx, SWIZZLE=swizzle, BLOCK=block, num_warps=warps, **consts
    )


def _result(q, s, o, e, mx: bool, unbiased: bool) -> tuple[torch.Tensor, ...]:
    out = (q, s.view(torch.float8_e8m0fnu if mx else torch.float8_e4m3fn), o)
    return (*out, e) if unbiased else out


def quantize_rows(
    x: torch.Tensor, mx: bool = False, swizzle: bool = False, unbiased: bool = False
) -> tuple[torch.Tensor, ...]:
    """``(codes, scales, outer)`` of ``x [..., K]`` (CUDA): the plain quantisation pass;
    ``unbiased``: and the epilogue's per-token scale ``outer * quant.fp4_bias_correction``."""
    x2 = _rows(x)
    m, k = x2.shape
    q, s, o, programs = _outputs(m, k, x.device, mx, swizzle)
    e = torch.empty_like(o) if unbiased else o
    if m:
        args = (x2, q, s, o, e, m, k, x2.stride(0))
        _launch(_quant_rows_kernel, args, k, programs, mx, swizzle, UNBIASED=unbiased)
    return _result(q, s, o, e, mx, unbiased)


def rmsnorm_fp4(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    mx: bool = False,
    swizzle: bool = False,
    out: torch.Tensor | None = None,
    unbiased: bool = False,
) -> tuple[torch.Tensor, ...]:
    """``(codes, scales, outer)`` of ``RMSNorm(x) * weight`` (``x [..., K]`` bf16 / fp16),
    quantised in the norm's epilogue; ``out`` (``[rows, K]``, x's dtype): also write the
    norm's output there; ``unbiased``: also the epilogue's per-token scale (fourth)."""
    x2 = _rows(x)
    m, k = x2.shape
    q, s, o, programs = _outputs(m, k, x.device, mx, swizzle)
    e = torch.empty_like(o) if unbiased else o
    if m:
        h = x2 if out is None else out
        args = (x2, weight, q, s, o, e, h, m, k, x2.stride(0), float(eps))
        consts = {"UNBIASED": unbiased, "WRITE": out is not None}
        _launch(_rmsnorm_fp4_kernel, args, k, programs, mx, swizzle, **consts)
    return _result(q, s, o, e, mx, unbiased)


def silu_mul_fp4(
    gate: torch.Tensor,
    up: torch.Tensor | None = None,
    mx: bool = False,
    swizzle: bool = False,
    out: torch.Tensor | None = None,
    unbiased: bool = False,
) -> tuple[torch.Tensor, ...]:
    """``(codes, scales, outer)`` of ``F.silu(gate) * up`` (bf16 / fp16), quantised in the
    epilogue. Without ``up``, ``gate`` is the merged ``[..., 2F]`` output of a gate|up GEMM
    (gate first). ``out``: also write ``F.silu(gate) * up`` there; ``unbiased``: also the
    epilogue's per-token scale (fourth)."""
    g2 = _rows(gate)
    if up is None:
        f = g2.shape[1] // 2
        g2, u2 = g2[:, :f], g2[:, f:]
    else:
        u2 = _rows(up)
    m, k = g2.shape
    q, s, o, programs = _outputs(m, k, gate.device, mx, swizzle)
    e = torch.empty_like(o) if unbiased else o
    if m:
        h = g2 if out is None else out
        args = (g2, u2, q, s, o, e, h, m, k, g2.stride(0), u2.stride(0))
        consts = {"UNBIASED": unbiased, "WRITE": out is not None}
        _launch(_silu_mul_fp4_kernel, args, k, programs, mx, swizzle, **consts)
    return _result(q, s, o, e, mx, unbiased)


def quantize_activations(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """This example's activation quantisation of ``x [..., K]`` (the module's input; the
    producers share its code): packed codes ``[rows, K / 2]``, unswizzled e4m3 scales
    ``[rows, K / 16]`` and the outer scale per row ``[rows]``. CUDA (any GPU): the Triton
    kernel; else the reference math."""
    x2 = x.reshape(-1, x.shape[-1])
    k = x2.shape[-1]
    if x2.is_cuda and x2.shape[0] and k % 16 == 0 and k <= MAX_K:
        codes, scales, outer = quantize_rows(x2)
        return codes, scales, outer
    return quantize_fp4_activations(x2, "nvfp4")


# ------------------------------------------------------------------ the W4A4 MLP


def block_scaled_fp4(device: torch.device) -> bool:
    """Whether ``F.scaled_mm`` NVFP4 runs on ``device``: block-scaled FP4 tensor cores (sm_100+)
    and a torch with ``F.scaled_mm`` and the FP4 dtype."""
    if device.type != "cuda" or not hasattr(F, "scaled_mm") or not hasattr(torch, _FP4):
        return False
    return torch.cuda.get_device_capability(device) >= (10, 0)


def _gemm(
    xq: torch.Tensor,
    xs: torch.Tensor,
    xo: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    ts: torch.Tensor,
    emulated: torch.Tensor | None,
) -> torch.Tensor:
    """``acc * (outer[m] * tensor_scale)`` in bf16 of NVFP4 activations ``(xq, xs, xo)`` and
    weight ``(codes, scales, ts)``: ``F.scaled_mm`` with swizzled scales (``emulated`` None), or
    the fp32 math with the weight's values ``emulated`` [N, K] and unswizzled ``xs``."""
    if emulated is None:
        fp4, blockwise = torch.float4_e2m1fn_x2, F.ScalingType.BlockWise1x16
        acc = F.scaled_mm(
            xq.view(fp4),
            codes.view(fp4).t(),  # column-major [K / 2, N]: cuBLASLt's layout for B
            scale_a=xs,
            scale_recipe_a=blockwise,
            scale_b=scales,
            scale_recipe_b=blockwise,
            swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
            swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
            output_dtype=torch.float32,
        )
    else:
        acc = fp4_values(xq, xs) @ emulated.T
    return (acc * (xo[:, None] * ts)).to(torch.bfloat16)


def _w4a4_mlp(
    x: torch.Tensor,
    gu_codes: torch.Tensor,
    gu_scales: torch.Tensor,
    gu_ts: torch.Tensor,
    d_codes: torch.Tensor,
    d_scales: torch.Tensor,
    d_ts: torch.Tensor,
    gu_emu: torch.Tensor | None,
    d_emu: torch.Tensor | None,
    unbiased: bool,
) -> torch.Tensor:
    """down(silu(gate(x)) * up(x)), every GEMM W4A4; the activations quantised by the
    producers (swizzled for ``F.scaled_mm``, row-major for the emulation); ``unbiased``: the
    epilogues take the producers' per-token scales (``outer * c``)."""
    swizzle = gu_emu is None
    xq, xs, *xo = quantize_rows(x, swizzle=swizzle, unbiased=unbiased)  # or rmsnorm_fp4
    gu = _gemm(xq, xs, xo[-1], gu_codes, gu_scales, gu_ts, gu_emu)  # [M, 2F] bf16
    hq, hs, *ho = silu_mul_fp4(gu, swizzle=swizzle, unbiased=unbiased)  # NVFP4, no bf16 h
    y = _gemm(hq, hs, ho[-1], d_codes, d_scales, d_ts, d_emu)
    return y.reshape(*x.shape[:-1], d_codes.shape[0])


class Fp4ProducerMLP(nn.Module):
    """A gated SiLU MLP with W4A4 GEMMs and the SiLU-mul emitting NVFP4 for ``down_proj``;
    ``unbiased``: the opt-in bias correction (weights ``quantize_fp4(..., unbiased=True)``, a
    factor per output channel in the tensor scale; the producers' per-token factor)."""

    def __init__(self, reference: nn.Module, unbiased: bool = False) -> None:
        super().__init__()
        gate, up, down = reference.gate_proj, reference.up_proj, reference.down_proj
        w_gu = torch.cat([gate.weight.detach(), up.weight.detach()])  # one GEMM for both
        gu = quantize_fp4(w_gu, unbiased=unbiased)  # once, here: never per call
        d = quantize_fp4(down.weight.detach(), unbiased=unbiased)
        self.quant_error = {"gate_up": fp4_error(w_gu, *gu), "down": fp4_error(down.weight, *d)}
        self.unbiased = unbiased
        self.reference_args = (*gu, *d)  # unswizzled: the reference math (fp4_w4a4_linear)
        self._args: tuple = ()
        if block_scaled_fp4(w_gu.device):  # F.scaled_mm's layout, swizzled once
            self._args = (gu[0], swizzle_fp4_scales(gu[1]), gu[2], d[0], swizzle_fp4_scales(d[1]))
            self._args += (d[2], None, None, unbiased)
        elif w_gu.is_cuda:  # the same math with the GEMMs in fp32: the weights' values, once
            self._args = (*gu, *d, fp4_values(gu[0], gu[1]), fp4_values(d[0], d[1]), unbiased)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.bfloat16 or not x.is_cuda or not self._args:  # the same numerics
            gu_codes, gu_scales, gu_ts, d_codes, d_scales, d_ts = self.reference_args
            gu = fp4_w4a4_linear(x, gu_codes, gu_scales, gu_ts, unbiased=self.unbiased)
            g, u = gu.chunk(2, dim=-1)
            return fp4_w4a4_linear(F.silu(g) * u, d_codes, d_scales, d_ts, unbiased=self.unbiased)
        launch = w4a4_mlp if torch.compiler.is_compiling() else _w4a4_mlp
        return launch(x, *self._args)


w4a4_mlp = torch.library.custom_op(f"{_NS}::w4a4_mlp", _w4a4_mlp, mutates_args=())


@w4a4_mlp.register_fake
def _(x, gu_codes, gu_scales, gu_ts, d_codes, d_scales, d_ts, gu_emu, d_emu, unbiased):
    return x.new_empty((*x.shape[:-1], d_codes.shape[0]))


def _is_silu(act: object) -> bool:
    return act is F.silu or type(act).__name__ in ("SiLU", "SiLUActivation")


def build(reference: nn.Module, unbiased: bool = False) -> nn.Module:
    """``unbiased``: the opt-in bias correction of e2m1's in-phase shrink (skill fp4-w4a4;
    a tuning keyword for ``sweep_candidate``)."""
    linears = [getattr(reference, n, None) for n in ("gate_proj", "up_proj", "down_proj")]
    if not all(isinstance(lin, nn.Linear) for lin in linears):
        return reference
    gate, up, down = linears
    hidden, inter = gate.in_features, gate.out_features
    ok = (
        _is_silu(getattr(reference, "act_fn", None))
        and all(lin.bias is None and lin.weight.dtype == torch.bfloat16 for lin in linears)
        and (up.in_features, up.out_features) == (hidden, inter)
        and (down.in_features, down.out_features) == (inter, hidden)
        and hidden % 16 == 0
        and inter % 16 == 0  # NVFP4 blocks along K; F.scaled_mm's N % 8
        and max(hidden, inter) <= MAX_K
    )
    return Fp4ProducerMLP(reference, unbiased) if ok else reference
