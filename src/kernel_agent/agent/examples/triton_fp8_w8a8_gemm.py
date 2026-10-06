"""Example candidate (Triton): FP8 W8A8 GEMM for ``nn.Linear`` on e4m3 tensor cores, for
compute-bound GEMMs (hundreds of rows per call).

Reduced precision: only for a target whose spec says ``"precision": "fp8_w8a8"`` (a
``--quality near-lossless`` run, knowledge/low_precision.md); the exact tier rejects it.
The recipe is the one the systems agent of the VoxCPM2 throughput run found
(``runs/openbmb--VoxCPM2/20261006-004718``, transforms ``fp8_locdit_mlp`` and
``triton_fp8_locdit_gemm``, ledger exp 40-47: the LocDiT GEMMs at M = 352, end to end
3.32x -> 4.71x within the perceptual gate):

* ``build()`` quantises the weight once (``kernel_agent.kernels.quant.quantize_fp8``: e4m3,
  one fp32 scale per output channel); no bf16 copy is kept.
* Every call quantises the activations per token (``_quant_kernel``, one program per row:
  ``scale = amax(|row|) / 448``, codes ``round(x / scale)`` in e4m3; dynamic, never
  calibrated offline). In a model, fuse this into the producer of ``x`` (the RMSNorm or
  ``silu(gate) * up`` before the GEMM): the run let Inductor fuse it there.
* ``_gemm_kernel``: ``tl.dot`` on e4m3 x e4m3 with fp32 accumulation, both scales (and the
  bias) applied once per output in the epilogue, one rounding to bf16. A plain tile loop,
  grouped along M for L2 reuse of the weight tiles.
* One tile config per weight shape ``(N, K)``, picked by the run at M = 352 on the RTX
  5070 Ti inside a CUDA graph with 12 L2-cold weights: cuBLASLt's sm_120 FP8 kernels
  (``torch._scaled_mm``) use large tiles that leave SMs idle at that M (24 output tiles for
  N = 1024 on 70 SMs). gate|up [1024 -> 8192] 36.3 vs 41.2 us, QKV [1024 -> 2560] 14.4 vs
  15.6, o_proj [2048 -> 1024] 14.0 vs 15.1, down [4096 -> 1024] 26.5 vs 27.0 (cuBLAS bf16:
  68 and 41 us for gate|up and down). Other shapes: :data:`DEFAULT`; tune with
  ``sweep_candidate`` (``bm``, ``bn``, ``bk``, ``warps``, ``stages``).
* Shapes the kernel does not tile (N or K not a multiple of the tile) and non-bf16 or CPU
  inputs: ``kernel_agent.kernels.quant.fp8_w8a8_linear`` (``torch._scaled_mm`` where it
  applies), the same numerics.
* The launcher is also a ``torch.library.custom_op`` with a fake implementation, so the
  GEMM stays one opaque op under ``torch.compile`` (``fullgraph=True``) and CUDA graphs
  (both bit-identical to eager). Eager calls skip the op and call the launcher directly:
  through the custom-op dispatcher a call costs ~89 us of host time instead of ~47 us
  (two Triton launches and three allocations), against ~38 us on the GPU at M = 352.

Measured (RTX 5070 Ti, M = 352, CUDA graph, warm L2), quantisation included: gate|up 36.1
us (cuBLAS bf16 68.2, ``torch._scaled_mm`` alone 41.3), q|k|v 15.1 (25.5 / 15.5), o_proj
15.5 (22.5 / 15.1), down 28.0 (38.9 / 26.6); the output is bit-identical to
``_scaled_mm``'s. Timed eagerly (the module evaluator), the host time hides the gain at
M = 352 (0.95x on gate|up); at M = 704 it measures 1.79x. ``tl.dot`` on e4m3 tops out
at ~195 TFLOP/s on sm_120, cuBLASLt's FP8 kernel for scalar scales at 338 (gate|up in
28.9 us, the scales then applied by the consumer): knowledge/low_precision.md, "The FP8
peak on sm_120".

Report ``fp8_w8a8_error(reference.weight, q, scale, x)`` on captured activations (weight,
activation and output error) and the evaluator's per-case ``max_rel_l2`` in ``NOTES.md``.
"""

import re

import torch
import triton
import triton.language as tl
from torch import nn

from kernel_agent.kernels.quant import fp8_error, fp8_w8a8_linear, quantize_fp8

# One namespace per candidate file (the evaluator names the module after the file's hash).
_NS = re.sub(r"\W", "_", __name__)

#: (N, K) -> (BM, BN, BK, warps, stages), measured at M = 352 (VoxCPM2 LocDiT, batch 16).
CONFIGS = {
    (8192, 1024): (64, 64, 64, 4, 3),  # gate|up merged
    (1024, 4096): (128, 64, 128, 8, 3),  # down
    (2560, 1024): (128, 64, 64, 4, 4),  # q|k|v merged
    (1024, 2048): (128, 64, 128, 8, 4),  # o_proj
}
DEFAULT = (128, 64, 128, 8, 3)


@triton.jit
def _quant_kernel(x_ptr, q_ptr, s_ptr, K, stride_x, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < K
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    scale = tl.where(amax > 0, amax / 448.0, 1.0)  # a row of zeros: scale 1, codes 0
    q = tl.clamp(x / scale, -448.0, 448.0).to(tl.float8e4nv)  # round to nearest even
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
    # c[m, n] = sa[m] * sb[n] * sum_k a[m, k] b[n, k] (+ bias[n]); a: [M, K] e4m3 tokens,
    # b: [N, K] e4m3 weight (nn.Linear layout), fp32 accumulation, bf16 out
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
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(0, K, BK):
        acc = tl.dot(tl.load(a_ptr, mask=row_ok, other=0.0), tl.load(b_ptr), acc)
        a_ptr += BK
        b_ptr += BK
    acc = acc * tl.load(sa + rm, mask=rm < M, other=0.0)[:, None] * tl.load(sb + rn)[None, :]
    if HAS_BIAS:
        acc += tl.load(bias + rn).to(tl.float32)[None, :]
    tl.store(c + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=row_ok)


def _w8a8_linear(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    bias: torch.Tensor | None,
    config: list[int],
) -> torch.Tensor:
    """x [..., K] bf16, w_q [N, K] e4m3, w_s [N] fp32 -> [..., N] bf16 (W8A8, fp32 acc)."""
    N, K = w_q.shape
    bm, bn, bk, warps, stages = config
    if N % bn or K % bk:
        return fp8_w8a8_linear(x, w_q, w_s, bias)  # torch._scaled_mm where it applies
    x2 = x.reshape(-1, K)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    M = x2.shape[0]
    out = torch.empty((*x.shape[:-1], N), device=x.device, dtype=torch.bfloat16)
    if M == 0:
        return out
    x_q = torch.empty((M, K), device=x.device, dtype=torch.float8_e4m3fn)
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
    )
    return out


# The same launcher as one opaque op for torch.compile / CUDA graphs (fresh output, no
# in-place writes); the fake only allocates.
w8a8_linear = torch.library.custom_op(f"{_NS}::w8a8_linear", _w8a8_linear, mutates_args=())


@w8a8_linear.register_fake
def _(x, w_q, w_s, bias, config):
    return x.new_empty((*x.shape[:-1], w_q.shape[0]), dtype=torch.bfloat16)


class Fp8W8A8Linear(nn.Module):
    """``nn.Linear`` with e4m3 weights (one fp32 scale per output channel) and e4m3
    activations (one fp32 scale per token, every call)."""

    def __init__(self, reference: nn.Linear, config: tuple[int, ...]) -> None:
        super().__init__()
        self.in_features, self.out_features = reference.in_features, reference.out_features
        q, scale = quantize_fp8(reference.weight)  # once, here: never per call
        self.register_buffer("weight_fp8", q)
        self.register_buffer("weight_scale", scale)
        self.register_parameter("bias", reference.bias)
        self.quant_error = fp8_error(reference.weight, q, scale)  # for NOTES.md
        # plain attributes: the forward does no nn.Module lookups
        self._args = (q, scale, reference.bias, list(config))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.bfloat16 or not x.is_cuda:
            return fp8_w8a8_linear(x, *self._args[:3])
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
        and reference.weight.dtype == torch.bfloat16
        and reference.weight.is_cuda
        and torch.cuda.get_device_capability(reference.weight.device) >= (8, 9)  # e4m3 MMA
        and (reference.bias is None or reference.bias.dtype == torch.bfloat16)
    )
    if not ok:
        return reference
    shape = (reference.out_features, reference.in_features)
    picked = CONFIGS.get(shape, DEFAULT)
    config = tuple(v or d for v, d in zip((bm, bn, bk, warps, stages), picked, strict=True))
    return Fp8W8A8Linear(reference, config)
