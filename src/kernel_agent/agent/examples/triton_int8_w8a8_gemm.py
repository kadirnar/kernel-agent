"""Example candidate (Triton): INT8 W8A8 GEMM for ``nn.Linear`` on the IMMA tensor cores
(s8 x s8 -> int32), for compute-bound GEMMs (hundreds of rows per call). The 8-bit compute
path of GPUs without FP8 tensor cores (Ampere sm_80 / sm_86), and an option on every GPU
from sm_80 on.

Reduced precision: only for a target whose spec says ``"precision": "int8_w8a8"`` (a
``--quality near-lossless`` or ``relaxed`` run, skill int8-w8a8); the exact tier
rejects it.

* ``build()`` quantises the weight once (``kernel_agent.kernels.quant.quantize_int8``:
  symmetric int8 codes in [-127, 127], one fp32 scale ``amax / 127`` per output channel); no
  bf16 copy is kept.
* Every call quantises the activations per token (``_quant_kernel``, one program per row, one
  pass: ``scale = amax(|row|) * (1 / 127)``, codes ``round(x / scale)`` to nearest even, IEEE
  division as in ``quant.quantize_int8_activations``: the same codes; dynamic, never
  calibrated offline). In a model, fuse it into the producer of ``x`` (the RMSNorm or
  ``silu(gate) * up`` before the GEMM: :func:`quantize_rows` is the device code to copy).
* ``_gemm_kernel``: ``tl.dot`` on int8 tiles with an int32 accumulator (``mma.sync
  m16n8k32.s32.s8.s8.s32``, SASS ``IMMA.16832.S8.S8``: exact integer sums), both scales and
  the bias applied once per output in the epilogue (``acc * x_scale[m] * w_scale[n] +
  bias[n]`` in fp32, one rounding to the activations' dtype, bf16 or fp16: :data:`DTYPES`,
  taken from the input): the output equals ``quant.int8_w8a8_linear``
  (``torch._int_mm``) bit for bit. A plain tile loop, grouped along M for L2 reuse of the
  weight tiles. On an RTX 5070 Ti (sm_120) the s8 ``mma.sync`` runs at 410 TOPS, twice the
  plain e4m3 ``QMMA.F32`` (206) of Triton's ``tl.dot`` on FP8 and as fast as the
  block-scaled ``QMMA.SF`` (412): no block-scale trick is needed for the full rate.
* One tile config per weight shape ``(N, K)`` (:data:`CONFIGS`, swept on the RTX 5070 Ti at
  M = 352 and 704, CUDA graph); other shapes :data:`DEFAULT`; tune with ``sweep_candidate``
  (``bm``, ``bn``, ``bk``, ``warps``, ``stages``).
* Shapes the kernel does not tile (N or K not a multiple of the tile), other dtypes or CPU
  inputs: ``kernel_agent.kernels.quant.int8_w8a8_linear`` (``torch._int_mm`` where it
  applies), the same numerics.
* The launcher is also a ``torch.library.custom_op`` with a fake implementation, so the GEMM
  stays one opaque op under ``torch.compile`` (``fullgraph=True``) and CUDA graphs. Eager
  calls skip the op and call the launcher directly (the custom-op dispatcher roughly doubles
  the host time of a call).

Measured (RTX 5070 Ti, CUDA graph, quantisation included; docs/research-scripts/int8-178):
gate|up [352, 1024] -> 8192 in 25.1 us (cuBLAS bf16 68.4, the FP8 example 32.0), at M = 704
45.8 (126.6 / 57.7); down [352, 4096] -> 1024 20.0 (39.0 / 18.8), q|k|v -> 2560 10.6 (25.5 /
10.8), o_proj [352, 2048] -> 1024 11.7 (22.6 / 10.7). Timed eagerly by the module evaluator
(gate|up at M = 704, ``doctor --smoke``): 1.70x. Report ``int8_w8a8_error(reference.weight,
q, scale, x)`` on captured activations (weight, activation and output error;
``activation_crest`` above ~20 means
outlier channels: SmoothQuant or FP8 / bf16 there) and the evaluator's per-case
``max_rel_l2`` in ``NOTES.md``.
"""

import re

import torch
import triton
import triton.language as tl
from torch import nn
from triton.language.extra import libdevice

from kernel_agent.kernels.quant import INT8_STEP as INT8_STEP_VALUE
from kernel_agent.kernels.quant import int8_error, int8_w8a8_linear, quantize_int8

#: GPUs this example runs on (``kernel_agent.gpu_arch.supports``: ``doctor --smoke`` skips
#: it elsewhere and says why).
ARCHS = "sm_80+"
ARCHS_WHY = (
    "Triton's int8 tl.dot on the IMMA tensor cores (mma.sync m16n8k32 s8: sm_80+; for sm_75 "
    "Triton 3.8 fails to compile it: cuda_int8_skinny_gemm.py has Turing's m8n8k16)"
)
#: Activation dtypes the kernels take (from the input; the output keeps it).
DTYPES = (torch.bfloat16, torch.float16)

# One namespace per candidate file (the evaluator names the module after the file's hash).
_NS = re.sub(r"\W", "_", __name__)

#: (N, K) -> (BM, BN, BK, warps, stages): VoxCPM2 LocDiT GEMM shapes at M = 352 / 704.
CONFIGS = {
    (8192, 1024): (128, 128, 64, 8, 3),  # gate|up merged
    (1024, 4096): (128, 64, 128, 4, 3),  # down
    (2560, 1024): (128, 64, 64, 4, 3),  # q|k|v merged
    (1024, 2048): (128, 64, 64, 4, 3),  # o_proj
}
DEFAULT = (128, 128, 64, 8, 3)
#: ``1 / 127`` in fp32 (``quant.INT8_STEP``): the scale is a product, as in the reference.
INT8_STEP: tl.constexpr = tl.constexpr(INT8_STEP_VALUE)


@triton.jit
def quantize_rows(x):
    """int8 codes and scale of one row ``x`` (fp32, padded with zeros to a power of two):
    ``scale = amax * (1 / 127)`` (1 for a row of zeros), ``round(x / scale)`` to nearest
    even with an IEEE division: ``quant.quantize_int8_activations``' codes, bit for bit."""
    amax = tl.max(tl.abs(x), axis=0)
    scale = tl.where(amax > 0, amax * INT8_STEP, 1.0)
    q = libdevice.rint(tl.div_rn(x, scale))
    return tl.clamp(q, -127.0, 127.0).to(tl.int8), scale


@triton.jit
def _quant_kernel(x_ptr, q_ptr, s_ptr, K, stride_x, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < K
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    q, scale = quantize_rows(x)
    tl.store(q_ptr + row * K + cols, q, mask=mask)
    tl.store(s_ptr + row, scale)


@triton.jit
def _gemm_kernel(
    a,
    b,
    sa,
    sb,
    bias,
    c,
    M,
    N,
    K,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GM: tl.constexpr,
):
    # c[m, n] = sa[m] * sb[n] * sum_k a[m, k] b[n, k] (+ bias[n]); a: [M, K] int8 tokens,
    # b: [N, K] int8 weight (nn.Linear layout), exact int32 accumulation, bf16 out
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM)
    npn = tl.cdiv(N, BN)
    group = GM * npn
    first_m = (pid // group) * GM
    group_m = min(npm - first_m, GM)
    pm = first_m + (pid % group) % group_m
    pn = (pid % group) // group_m
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptr = a + rm[:, None] * K + rk[None, :]
    b_ptr = b + rn[None, :] * K + rk[:, None]
    row_ok = rm[:, None] < M
    acc = tl.zeros((BM, BN), tl.int32)
    for _ in range(0, K, BK):
        acc = tl.dot(tl.load(a_ptr, mask=row_ok, other=0), tl.load(b_ptr), acc, out_dtype=tl.int32)
        a_ptr += BK
        b_ptr += BK
    x_scale = tl.load(sa + rm, mask=rm < M, other=0.0)  # tail rows: masked, never read past M
    out = acc.to(tl.float32) * x_scale[:, None] * tl.load(sb + rn)[None, :]
    if HAS_BIAS:
        out += tl.load(bias + rn).to(tl.float32)[None, :]
    # one rounding to the output's dtype (the activations': bf16 or fp16)
    tl.store(c + rm[:, None] * N + rn[None, :], out.to(c.dtype.element_ty), mask=row_ok)


def _w8a8_linear(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    bias: torch.Tensor | None,
    config: list[int],
) -> torch.Tensor:
    """x [..., K] bf16 / fp16, w_q [N, K] int8, w_s [N] fp32 -> [..., N] in x's dtype (W8A8,
    int32 acc). ``config``: BM, BN, BK, warps, stages."""
    N, K = w_q.shape
    bm, bn, bk, warps, stages = config
    if N % bn or K % bk:
        return int8_w8a8_linear(x, w_q, w_s, bias)  # torch._int_mm where it applies
    x2 = x.reshape(-1, K)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    M = x2.shape[0]
    out = torch.empty((*x.shape[:-1], N), device=x.device, dtype=x.dtype)
    if M == 0:
        return out
    x_q = torch.empty((M, K), device=x.device, dtype=torch.int8)
    x_s = torch.empty((M,), device=x.device, dtype=torch.float32)
    block = triton.next_power_of_2(K)
    _quant_kernel[(M,)](x2, x_q, x_s, K, x2.stride(0), BLOCK=block, num_warps=4)
    grid = (triton.cdiv(M, bm) * (N // bn),)
    _gemm_kernel[grid](
        x_q,
        w_q,
        x_s,
        w_s,
        x_s if bias is None else bias,  # unused without a bias
        out,
        M,
        N,
        K,
        HAS_BIAS=bias is not None,
        BM=bm,
        BN=bn,
        BK=bk,
        GM=8,
        num_warps=warps,
        num_stages=stages,
        # no FMA contraction of `acc * x_scale * w_scale + bias`: the reference's roundings
        # (contracted, a few outputs in a million differ; the products are exact integers)
        enable_fp_fusion=False,
    )
    return out


# The same launcher as one opaque op for torch.compile / CUDA graphs (fresh output, no
# in-place writes); the fake only allocates.
w8a8_linear = torch.library.custom_op(f"{_NS}::int8_w8a8_linear", _w8a8_linear, mutates_args=())


@w8a8_linear.register_fake
def _(x, w_q, w_s, bias, config):
    return x.new_empty((*x.shape[:-1], w_q.shape[0]))


class Int8W8A8Linear(nn.Module):
    """``nn.Linear`` with int8 weights (one fp32 scale per output channel) and int8
    activations (one fp32 scale per token, every call), int32 accumulation."""

    def __init__(self, reference: nn.Linear, config: tuple[int, ...]) -> None:
        super().__init__()
        self.in_features, self.out_features = reference.in_features, reference.out_features
        q, scale = quantize_int8(reference.weight)  # once, here: never per call
        self.register_buffer("weight_int8", q)
        self.register_buffer("weight_scale", scale)
        self.register_parameter("bias", reference.bias)
        self.quant_error = int8_error(reference.weight, q, scale)  # for NOTES.md
        # plain attributes: the forward does no nn.Module lookups
        self._args = (q, scale, reference.bias, list(config))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype not in DTYPES or not x.is_cuda:
            return int8_w8a8_linear(x, *self._args[:3])
        # compiled: one opaque op; eager: the launcher itself (no custom-op dispatch cost)
        launch = w8a8_linear if torch.compiler.is_compiling() else _w8a8_linear
        return launch(x, *self._args)


def build(
    reference: nn.Module,
    bm: int = 0,
    bn: int = 0,
    bk: int = 0,
    warps: int = 0,
    stages: int = 0,
) -> nn.Module:
    """Tile sizes ``bm`` / ``bn`` / ``bk``, ``warps`` and ``stages`` are tuning keywords
    for ``sweep_candidate``; 0 takes the value of :data:`CONFIGS` for the weight's shape
    (else :data:`DEFAULT`)."""
    ok = (
        isinstance(reference, nn.Linear)
        and reference.weight.dtype in DTYPES
        and reference.weight.is_cuda
        and torch.cuda.get_device_capability(reference.weight.device) >= (8, 0)  # int8 MMA
        and (reference.bias is None or reference.bias.dtype == reference.weight.dtype)
    )
    if not ok:
        return reference
    shape = (reference.out_features, reference.in_features)
    picked = CONFIGS.get(shape, DEFAULT)
    config = tuple(v or d for v, d in zip((bm, bn, bk, warps, stages), picked, strict=True))
    return Int8W8A8Linear(reference, config)
