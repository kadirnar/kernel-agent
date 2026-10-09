"""Example candidate (Triton quantiser + cuBLASLt NVFP4 GEMM): W4A4 ``nn.Linear`` on the
block-scaled FP4 tensor cores (sm_100 / sm_120), for compute-bound GEMMs.

Reduced precision, opt-in: only for a target whose spec says ``"precision": "fp4_w4a4"`` (a
run whose ``--precisions`` names it; skill fp4-w4a4), checked in the ``near-lossless-fp4a``
tier; the exact and the 8-bit tiers reject it.

* ``build()`` quantises the weight once (``kernel_agent.kernels.quant.quantize_fp4``: NVFP4
  e2m1 codes, two per byte, an e4m3 scale per 16 along K, an fp32 tensor scale) and swizzles
  its block scales once (``swizzle_fp4_scales``: cuBLASLt's 128 x 4 blocked layout); no bf16
  copy is kept.
* Every call quantises the activations in two Triton launches: ``_amax_kernel`` (the call's
  amax, an atomic max) and ``_nvfp4_quant_kernel`` (one program per row and ``BLOCKS`` blocks
  of 16), bit for bit ``quant.quantize_fp4_activations(x, "nvfp4", "tensor")``: ``outer =
  amax * NVFP4_OUTER_STEP``, block scale ``e4m3(min(bmax / (outer * 6), 448))`` (``tl.div_rn``,
  the hardware's round-to-nearest e4m3 conversion), codes ``e2m1(x / (scale * outer))`` to
  nearest even (the comparisons below are the e2m1 midpoints with their tie directions),
  saturated at ±6, packed two per byte; the scales go straight into the swizzled layout
  (``quant.mx_scale_offset``).
* The GEMM: two-level ``F.scaled_mm`` (``[BlockWise1x16, TensorWise]`` on both operands:
  the e4m3 block scales applied by the tensor core, the activations' outer scale and the
  weight's tensor scale and the bias by cuBLASLt's epilogue in fp32), bf16 out: without a bias
  bit for bit ``fp4_w4a4_linear(..., granularity="tensor")`` (measured on sm_120; with one,
  1-ulp differences in 3 of 100 000 outputs).
  One outer scale per call (``granularity="tensor"``) because ``F.scaled_mm`` takes no
  per-row fp32 scale: per-token outer scales (the reference default, measured as accurate)
  need fp32 output and an epilogue pass that costs more than the NVFP4 GEMM at M = 352
  (``cute_nvfp4_w4a4_gemm.py`` applies them in its own epilogue).
* Shapes it does not cover (K not a multiple of 16, N of 8, CPU, non-bf16, a GPU without
  block-scaled FP4 MMA): ``kernel_agent.kernels.quant.fp4_w4a4_linear``, the same numerics.

``quantize_activations`` (module level) returns this example's codes, unswizzled scales and
outer scales (checked against the reference). Report ``fp4_w4a4_error(reference.weight,
codes, scales, tensor_scale, x, granularity="tensor")`` on captured activations and the
evaluator's per-case ``max_rel_l2`` in ``NOTES.md``.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn

from kernel_agent.kernels.quant import (
    NVFP4_OUTER_STEP,
    fp4_error,
    fp4_w4a4_linear,
    quantize_fp4,
    quantize_fp4_activations,
    swizzle_fp4_scales,
)

#: GPUs this example runs on (``kernel_agent.gpu_arch.supports``: ``doctor --smoke`` skips
#: it elsewhere and says why).
ARCHS = "sm_100+"
ARCHS_WHY = "block-scaled FP4 tensor cores (F.scaled_mm BlockWise1x16, cuBLASLt NVFP4)"

#: 16-element blocks per program of the quantiser, and its warps (RTX 5070 Ti, CUDA graph:
#: 4096 x 4096 bf16 in 40 us with 32 / 2, 62 us with 8 / 1; 352 x 4096 in 4.4 us).
BLOCKS = 32
QUANT_WARPS = 2
#: The outer scale of the activations: one per call (``F.scaled_mm``'s tensor-wise scale).
GRANULARITY = "tensor"


@triton.jit
def _amax_kernel(x_ptr, amax_ptr, K, stride_x, CHUNK: tl.constexpr):
    # the call's amax(|x|) into amax_ptr (zeroed before): one program per row
    row = tl.program_id(0)
    acc = tl.zeros((CHUNK,), dtype=tl.float32)
    for k0 in range(0, K, CHUNK):
        cols = k0 + tl.arange(0, CHUNK)
        x = tl.load(x_ptr + row * stride_x + cols, mask=cols < K, other=0.0).to(tl.float32)
        acc = tl.maximum(acc, tl.abs(x))
    tl.atomic_max(amax_ptr, tl.max(acc, axis=0))


@triton.jit
def _nvfp4_quant_kernel(
    x_ptr,
    amax_ptr,
    q_ptr,
    s_ptr,
    K,
    stride_x,
    scale_cols,
    outer_step,
    SWIZZLE: tl.constexpr,
    BLOCKS: tl.constexpr,
):
    # packed e2m1 codes q [M, K / 2] (uint8) and e4m3 block scales s of x [M, K]: one program
    # per row and BLOCKS blocks of 16 along K; amax_ptr[1] receives the outer scale
    row = tl.program_id(0)
    blk = tl.program_id(1) * BLOCKS + tl.arange(0, BLOCKS)
    cols = blk[:, None] * 16 + tl.arange(0, 16)[None, :]
    mask = cols < K
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    amax = tl.load(amax_ptr)
    outer = tl.where(amax > 0, amax * outer_step, 1.0)
    if (row == 0) & (tl.program_id(1) == 0):
        tl.store(amax_ptr + 1, outer)
    bmax = tl.max(tl.abs(x), axis=1)
    ratio = tl.div_rn(bmax, tl.zeros_like(bmax) + outer * 6.0)
    scale = tl.minimum(ratio, 448.0).to(tl.float8e4nv)  # round to nearest even
    step = scale.to(tl.float32) * outer
    safe = tl.where(step > 0, step, 1.0)[:, None] + tl.zeros_like(x)
    v = tl.where(step[:, None] > 0, tl.div_rn(x, safe), 0.0)  # a zero scale: codes 0
    v = tl.minimum(tl.maximum(v, -6.0), 6.0)
    mag = tl.abs(v)
    # e2m1 to nearest even: the midpoints 0.25 / 1.25 / 2.5 / 5 round down, 0.75 / 1.75 / 3.5 up
    idx = (mag > 0.25).to(tl.int32) + (mag >= 0.75).to(tl.int32) + (mag > 1.25).to(tl.int32)
    idx += (mag >= 1.75).to(tl.int32) + (mag > 2.5).to(tl.int32) + (mag >= 3.5).to(tl.int32)
    idx += (mag > 5.0).to(tl.int32)
    code = idx + tl.where((v < 0) & (idx > 0), 8, 0)  # no negative zero
    lo, hi = tl.split(tl.reshape(code, (BLOCKS, 8, 2)))
    packed = (lo | (hi << 4)).to(tl.uint8)
    pcols = blk[:, None] * 8 + tl.arange(0, 8)[None, :]
    tl.store(q_ptr + row * (K // 2) + pcols, packed, mask=pcols < K // 2)
    if SWIZZLE:  # cuBLASLt's 128 x 4 blocks (kernel_agent.kernels.quant.mx_scale_offset)
        off = (row // 128) * ((scale_cols + 3) // 4) + blk // 4
        off = off * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + blk % 4
    else:
        off = row * scale_cols + blk
    tl.store(s_ptr + off, scale.to(tl.uint8, bitcast=True), mask=blk < scale_cols)


def _fast(x: torch.Tensor, codes: torch.Tensor) -> bool:
    """bf16 CUDA activations, K a multiple of 16 and N of 8, block-scaled FP4 MMA (sm_100+)."""
    if not (x.is_cuda and x.dtype == torch.bfloat16 and hasattr(F, "scaled_mm")):
        return False
    if (2 * codes.shape[1]) % 16 or codes.shape[0] % 8:
        return False
    return torch.cuda.get_device_capability(x.device) >= (10, 0)


def _quantize(x2: torch.Tensor, swizzle: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The Triton quantiser on ``x2 [M, K]`` (CUDA, K a multiple of 16): packed codes
    [M, K / 2], e4m3 scales (swizzled, flat; or [M, K / 16]) and the outer scale (0-d)."""
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m, k = x2.shape
    cols = k // 16
    found = torch.zeros(2, device=x2.device, dtype=torch.float32)  # amax, outer (the kernel)
    codes = torch.empty((m, k // 2), device=x2.device, dtype=torch.uint8)
    if swizzle:  # padding (rows to 128, columns to 4) must be code 0: never in a stored output
        size = triton.cdiv(m, 128) * 128 * triton.cdiv(cols, 4) * 4
        padded = m % 128 or cols % 4
        scales = (torch.zeros if padded else torch.empty)(size, device=x2.device, dtype=torch.uint8)
    else:
        scales = torch.empty((m, cols), device=x2.device, dtype=torch.uint8)
    if m:
        _amax_kernel[(m,)](x2, found, k, x2.stride(0), CHUNK=512, num_warps=4)
        grid = (m, triton.cdiv(cols, BLOCKS))
        _nvfp4_quant_kernel[grid](
            x2,
            found,
            codes,
            scales,
            k,
            x2.stride(0),
            cols,
            NVFP4_OUTER_STEP,
            SWIZZLE=swizzle,
            BLOCKS=BLOCKS,
            num_warps=QUANT_WARPS,
        )
    return codes, scales.view(torch.float8_e4m3fn), found[1]


def quantize_activations(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """This example's activation quantisation of ``x [..., K]``: packed codes ``[rows, K /
    2]``, unswizzled e4m3 scales ``[rows, K / 16]`` and the outer scale per row ``[rows]``
    (one value per call). CUDA: the Triton kernels; else the reference math."""
    x2 = x.reshape(-1, x.shape[-1])
    if x2.is_cuda and x2.shape[-1] % 16 == 0:
        codes, scales, outer = _quantize(x2, swizzle=False)
        return codes, scales.view(x2.shape[0], -1), outer.expand(x2.shape[0]).contiguous()
    return quantize_fp4_activations(x2, "nvfp4", GRANULARITY)


class NVFP4W4A4Linear(nn.Module):
    """``nn.Linear`` with NVFP4 weights and activations (e2m1 + an e4m3 scale per 16 along K
    on both operands, fp32 outer scales)."""

    def __init__(self, reference: nn.Linear) -> None:
        super().__init__()
        self.in_features, self.out_features = reference.in_features, reference.out_features
        codes, scales, tensor_scale = quantize_fp4(reference.weight)  # once, here
        self.register_buffer("weight_fp4", codes)
        self.register_buffer("weight_scales", scales)
        self.register_buffer("weight_scales_swizzled", swizzle_fp4_scales(scales))
        self.register_buffer("tensor_scale", tensor_scale.float().reshape(()))
        self.register_parameter("bias", reference.bias)
        # the weight's error, for NOTES.md (with captured activations: fp4_w4a4_error)
        self.quant_error = fp4_error(reference.weight, codes, scales, tensor_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        codes, n = self.weight_fp4, self.out_features
        if not _fast(x, codes):
            return fp4_w4a4_linear(
                x,
                codes,
                self.weight_scales,
                self.tensor_scale,
                self.bias,
                granularity=GRANULARITY,
            )
        xq, xs, outer = _quantize(x.reshape(-1, self.in_features), swizzle=True)
        fp4 = torch.float4_e2m1fn_x2
        blockwise, tensorwise = F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise
        swizzle, plain = F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE
        y = F.scaled_mm(
            xq.view(fp4),
            codes.view(fp4).t(),  # column-major [K / 2, N]: cuBLASLt's layout for B
            scale_a=[xs, outer],
            scale_recipe_a=[blockwise, tensorwise],
            scale_b=[self.weight_scales_swizzled, self.tensor_scale],
            scale_recipe_b=[blockwise, tensorwise],
            swizzle_a=[swizzle, plain],
            swizzle_b=[swizzle, plain],
            bias=None if self.bias is None else self.bias.to(torch.bfloat16),  # fp32 epilogue
            output_dtype=torch.bfloat16,
        )
        return y.reshape(*x.shape[:-1], n)


def build(reference: nn.Module) -> nn.Module:
    ok = (
        isinstance(reference, nn.Linear)
        and reference.weight.is_floating_point()
        and reference.in_features % 16 == 0
    )
    return NVFP4W4A4Linear(reference) if ok else reference
