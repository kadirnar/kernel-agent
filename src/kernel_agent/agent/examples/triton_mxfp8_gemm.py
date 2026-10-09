"""Example candidate (Triton quantiser + cuBLASLt MXFP8 GEMM): MXFP8 W8A8 ``nn.Linear`` on
the block-scaled tensor cores (sm_100 / sm_120), for compute-bound GEMMs with wide outputs.

Reduced precision: only for a target whose spec says ``"precision": "fp8_mx"`` (a
``--quality near-lossless`` run, skill mxfp8); the exact tier
rejects it. Written for issue #144 from the measurements of docs/FP8.md (RTX 5070 Ti,
torch 2.14, cuBLASLt 13.1) and NOT run on a GPU when written: verify with
``kernel-agent doctor --smoke`` (selftest ``MX_EXAMPLES``) or ``pytest -m gpu
tests/test_mxfp8.py`` before relying on its numbers.

* ``build()`` quantises the weight once (``kernel_agent.kernels.quant.quantize_mxfp8``: e4m3
  codes [N, K], one e8m0 (power-of-two) scale per 32 consecutive K elements) and swizzles
  its scales once (``swizzle_mx_scales``: cuBLASLt's 128 x 4 blocked layout); no bf16 copy.
* Every call quantises the activations per block of 32 (``_mx_quant_kernel``: one program
  per row and 4 blocks; dynamic, never calibrated offline) with the non-saturating scale
  ``2^ceil(log2(amax / 448))``, computed exactly from the exponent bits of the block
  maximum ``amax = m * 2^E`` (m in [1, 2)): ``e = E - 8``, plus one when ``m > 1.75``
  (448 = 1.75 * 2^8). The OCP reference rule ``2^(floor(log2 amax) - 8)`` (``RULE =
  "floor"``) saturates block maxima above 448 x scale; the evaluator's scale-rule guard
  rejects it (the selftest checks that). The kernel writes the scales straight into the
  swizzled layout (``kernel_agent.kernels.quant.mx_scale_offset``).
* The GEMM: ``F.scaled_mm`` with ``ScalingType.BlockWise1x32`` on both operands and
  ``SwizzleType.SWIZZLE_32_4_4`` (cuBLASLt ``VEC32_UE8M0``): e4m3 x e4m3 with both block
  scales applied by the tensor core, fp32 accumulation, bf16 out (its result equals the
  fp32 math of the codes to bf16 rounding, docs/FP8.md §2). Measured there at M = 352 (CUDA
  graph, L2-cold weights, GEMM alone): gate|up [1024 -> 8192] 24.7 us vs 29.5 tensor-wise
  FP8 and 69.0 bf16; q|k|v [1024 -> 2560] 9.3 vs 10.6; at N = 1024 MXFP8 loses (one
  algorithm, no split-K: o_proj 14.8 vs 8.8 tensor-wise with batched split-K).
* Host time: ``F.scaled_mm`` costs ~27 us per eager call (Python recipe dispatch), the
  quantiser one Triton launch and two allocations; inside CUDA graphs none of it counts. An
  eager-timed target with several GEMMs per call gains from a direct cuBLASLt call.
* Shapes it does not cover (K not a multiple of 128, N of 16, CPU, non-bf16, a GPU without
  block-scaled MMA): ``kernel_agent.kernels.quant.mxfp8_linear``, the same numerics.

``quantize_activations`` (module level) is the hook of the evaluator's scale-rule guard
(``kernel_agent.kernels.scale_guard``): it returns this example's codes and unswizzled
scales. Report ``mxfp8_error(reference.weight, q, scales, x)`` on captured activations and
the evaluator's per-case ``max_rel_l2`` in ``NOTES.md``.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn

from kernel_agent.kernels.quant import (
    mxfp8_error,
    mxfp8_linear,
    quantize_mxfp8,
    swizzle_mx_scales,
)

#: GPUs this example runs on (``kernel_agent.gpu_arch.supports``: ``doctor --smoke`` skips
#: it elsewhere and says why).
ARCHS = "sm_100+"
ARCHS_WHY = "block-scaled FP8 tensor cores (F.scaled_mm BlockWise1x32, cuBLASLt VEC32_UE8M0)"
#: GPUs its quantiser compiles for (Triton's e4m3 type: sm_89+; the GEMM is cuBLASLt's).
ARCHS_COMPILES = "sm_89+"

#: The activation scale rule: ``ceil`` (2^ceil(log2(amax / 448))) or ``floor`` (the OCP rule,
#: saturating: rejected by the evaluator's scale-rule guard).
RULE = "ceil"
#: 32-element blocks per program of the quantiser.
BLOCKS = 4


@triton.jit
def _mx_quant_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    K,
    stride_x,
    scale_cols,
    CEIL: tl.constexpr,
    SWIZZLE: tl.constexpr,
    BLOCKS: tl.constexpr,
):
    # codes q [M, K] e4m3 and scales s (uint8 biased exponents: 127 = 2^0) of x [M, K]: one
    # program per row and BLOCKS blocks of 32 along K
    row = tl.program_id(0)
    blk = tl.program_id(1) * BLOCKS + tl.arange(0, BLOCKS)
    cols = blk[:, None] * 32 + tl.arange(0, 32)[None, :]
    mask = cols < K
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    bits = amax.to(tl.int32, bitcast=True)  # amax >= 0: no sign bit
    field = (bits >> 23) & 0xFF
    e = field - 127 - 8  # floor(log2 amax) - 8
    if CEIL:
        e += ((bits & 0x7FFFFF) > 0x600000).to(tl.int32)  # mantissa above 1.75: one more
    e = tl.where(field == 0, -127, e)  # zero or subnormal block maximum
    e = tl.minimum(tl.maximum(e, -127), 126)
    inv = ((127 - e) << 23).to(tl.float32, bitcast=True)  # 2^-e, exact
    q = tl.clamp(x * inv[:, None], -448.0, 448.0).to(tl.float8e4nv)  # round to nearest even
    tl.store(q_ptr + row * K + cols, q, mask=mask)
    if SWIZZLE:  # cuBLASLt's 128 x 4 blocks (kernel_agent.kernels.quant.mx_scale_offset)
        off = (row // 128) * ((scale_cols + 3) // 4) + blk // 4
        off = off * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + blk % 4
    else:
        off = row * scale_cols + blk
    tl.store(s_ptr + off, (e + 127).to(tl.uint8), mask=blk < scale_cols)


def _fast(x: torch.Tensor, q: torch.Tensor) -> bool:
    """bf16 CUDA activations, K a multiple of 128 and N of 16, block-scaled MMA (sm_100+)."""
    if not (x.is_cuda and x.dtype == torch.bfloat16 and hasattr(F, "scaled_mm")):
        return False
    if q.shape[1] % 128 or q.shape[0] % 16:
        return False
    return torch.cuda.get_device_capability(x.device) >= (10, 0)


def _quantize(x2: torch.Tensor, swizzle: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """The Triton quantiser on ``x2 [M, K]`` (CUDA, K a multiple of 32): codes e4m3 [M, K]
    and e8m0 scales, swizzled (flat) or ``[M, K / 32]``."""
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m, k = x2.shape
    cols = k // 32
    codes = torch.empty((m, k), device=x2.device, dtype=torch.float8_e4m3fn)
    if swizzle:  # padding (rows to 128, columns to 4) stays code 0: never in a stored output
        size = triton.cdiv(m, 128) * 128 * triton.cdiv(cols, 4) * 4
        scales = torch.zeros(size, device=x2.device, dtype=torch.uint8)
    else:
        scales = torch.empty((m, cols), device=x2.device, dtype=torch.uint8)
    if m:
        grid = (m, triton.cdiv(cols, BLOCKS))
        _mx_quant_kernel[grid](
            x2,
            codes,
            scales,
            k,
            x2.stride(0),
            cols,
            CEIL=RULE == "ceil",
            SWIZZLE=swizzle,
            BLOCKS=BLOCKS,
            num_warps=1,
        )
    return codes, scales.view(torch.float8_e8m0fnu)


def quantize_activations(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """This example's activation quantisation of ``x [..., K]``: codes e4m3 ``[rows, K]`` and
    unswizzled e8m0 scales ``[rows, K / 32]`` (the evaluator's scale-rule guard checks them
    against the block maxima). CUDA: the Triton kernel; else the reference math."""
    x2 = x.reshape(-1, x.shape[-1])
    if x2.is_cuda and x2.shape[-1] % 32 == 0:
        return _quantize(x2, swizzle=False)
    return quantize_mxfp8(x2, RULE)


class MXFP8Linear(nn.Module):
    """``nn.Linear`` with MXFP8 weights and activations (e4m3 + an e8m0 scale per 32 along
    K on both operands)."""

    def __init__(self, reference: nn.Linear) -> None:
        super().__init__()
        self.in_features, self.out_features = reference.in_features, reference.out_features
        q, scales = quantize_mxfp8(reference.weight)  # once, here: never per call
        self.register_buffer("weight_fp8", q)
        self.register_buffer("weight_scales", scales)
        self.register_buffer("weight_scales_swizzled", swizzle_mx_scales(scales))
        self.register_parameter("bias", reference.bias)
        self.quant_error = mxfp8_error(reference.weight, q, scales)  # for NOTES.md

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, n = self.weight_fp8, self.out_features
        if not _fast(x, q):
            return mxfp8_linear(x, q, self.weight_scales, self.bias, rule=RULE)
        xq, xs = _quantize(x.reshape(-1, self.in_features), swizzle=True)
        blockwise, swizzle = F.ScalingType.BlockWise1x32, F.SwizzleType.SWIZZLE_32_4_4
        y = F.scaled_mm(
            xq,
            q.t(),  # column-major [K, N]: cuBLASLt's layout for the second operand
            scale_a=xs,
            scale_recipe_a=blockwise,
            scale_b=self.weight_scales_swizzled,
            scale_recipe_b=blockwise,
            swizzle_a=swizzle,
            swizzle_b=swizzle,
            output_dtype=torch.bfloat16,
        )
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y.reshape(*x.shape[:-1], n)


def build(reference: nn.Module) -> nn.Module:
    ok = (
        isinstance(reference, nn.Linear)
        and reference.weight.is_floating_point()
        and reference.in_features % 32 == 0
    )
    return MXFP8Linear(reference) if ok else reference
