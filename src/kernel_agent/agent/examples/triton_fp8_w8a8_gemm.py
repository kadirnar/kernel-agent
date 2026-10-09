"""Example candidate (Triton): FP8 W8A8 GEMM for ``nn.Linear`` on e4m3 tensor cores, for
compute-bound GEMMs (hundreds of rows per call).

Reduced precision: only for a target whose spec says ``"precision": "fp8_w8a8"`` (a
``--quality near-lossless`` run, skill fp8-w8a8); the exact tier rejects it.
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
* ``_gemm_kernel``: e4m3 x e4m3 with fp32 accumulation, both scales (and the bias)
  applied once per output in the epilogue, one rounding to bf16. A plain tile loop,
  grouped along M for L2 reuse of the weight tiles. On sm_120 / sm_121 the product is
  ``tl.dot_scaled`` with constant unit ue8m0 scales (127 = 2^0, one per 32 along K): it
  lowers to the block-scaled MMA (PTX ``mma.sync ... kind::mxf8f6f4.block_scale``, SASS
  ``QMMA.SF``, 416 TFLOP/s on sm_120 against 208 for the plain e4m3 ``QMMA.F32`` of
  ``tl.dot``) and its result is bit-identical to ``tl.dot`` (docs/RESEARCH-TRITON.md §1.2:
  gate|up 33.3 vs 36.4 us at M = 352, output equal to row-wise ``torch._scaled_mm``'s). On
  sm_89 / sm_90 (no block-scaled MMA: Triton would emulate it through bf16) the kernel
  keeps ``tl.dot`` (QMMA on sm_89, ``wgmma`` on sm_90), and on sm_100 too (``tcgen05.mma
  kind::f8f6f4``, full rate there: :func:`block_scale_capable`). :func:`gemm_ptx` compiles
  the kernel for a GPU (no launch, no GPU needed) and :func:`block_scale_mma` checks its
  PTX; ``doctor --smoke`` requires the block-scaled MMA on sm_12x.
* One tile config per weight shape ``(N, K)`` at M = 352 (RTX 5070 Ti, CUDA graph, L2-cold
  weights): the best ``tl.dot_scaled`` tiles measured in docs/RESEARCH-TRITON.md §1.2 and
  docs/FP8.md (``gemm_dit352*.out``); cuBLASLt's sm_120 FP8 kernels (``torch._scaled_mm``)
  use large tiles that leave SMs idle at that M (24 output tiles for N = 1024 on 70 SMs).
  The ``tl.dot`` version of this example ran gate|up [1024 -> 8192] in 36.3 vs 41.2 us,
  QKV [1024 -> 2560] 14.4 vs 15.6, o_proj [2048 -> 1024] 14.0 vs 15.1, down [4096 -> 1024]
  26.5 vs 27.0 (cuBLAS bf16: 68 and 41 us for gate|up and down); ``tl.dot_scaled`` with
  loaded unit scales 34.3 / 12.4 / 11.9 / 25.4 us. Other shapes: :data:`DEFAULT`; tune with
  ``sweep_candidate`` (``bm``, ``bn``, ``bk``, ``warps``, ``stages``; ``scaled=0`` for the
  ``tl.dot`` kernel).
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
at ~195 TFLOP/s on sm_120 (``tl.dot_scaled`` 214-227 at 4096^3), cuBLASLt's FP8 kernel for
scalar scales at 338 (gate|up in 28.9 us, the scales then applied by the consumer;
``cuda_cublaslt_fp8.py`` calls it directly): skill fp8-w8a8 (gemm-paths.md), "The FP8 peak on
sm_120". These figures were measured on the ``tl.dot`` version and on the research kernels;
the ``tl.dot_scaled`` switch and its :data:`CONFIGS` are not re-measured on this file yet:
re-sweep them on a new GPU.

Report ``fp8_w8a8_error(reference.weight, q, scale, x)`` on captured activations (weight,
activation and output error) and the evaluator's per-case ``max_rel_l2`` in ``NOTES.md``.
"""

import re

import torch
import triton
import triton.language as tl
from torch import nn

from kernel_agent.kernels.quant import fp8_error, fp8_w8a8_linear, quantize_fp8

#: GPUs this example runs on (``kernel_agent.gpu_arch.supports``: ``doctor --smoke`` skips
#: it elsewhere and says why).
ARCHS = "sm_89+"
ARCHS_WHY = "e4m3 tensor cores (tl.dot on e4m3; tl.dot_scaled on sm_12x)"

# One namespace per candidate file (the evaluator names the module after the file's hash).
_NS = re.sub(r"\W", "_", __name__)

#: (N, K) -> (BM, BN, BK, warps, stages): the best ``tl.dot_scaled`` tiles at M = 352 (VoxCPM2
#: LocDiT, batch 16; BK a multiple of 32, one ue8m0 scale per 32 along K).
CONFIGS = {
    (8192, 1024): (64, 64, 128, 4, 3),  # gate|up merged (constant unit scales: 33.3 us)
    (1024, 4096): (128, 64, 128, 8, 3),  # down
    (2560, 1024): (64, 128, 128, 4, 3),  # q|k|v merged
    (1024, 2048): (64, 128, 128, 4, 3),  # o_proj
}
DEFAULT = (64, 64, 128, 4, 3)


def block_scale_capable(capability: tuple[int, int]) -> bool:
    """Whether the GEMM uses ``tl.dot_scaled`` on a GPU of ``capability``: GeForce Blackwell
    (sm_12x), where it lowers to the full-rate block-scaled ``mma.sync`` and plain e4m3 runs
    at half rate. Hopper and Ada emulate it through bf16; on sm_100 plain ``tl.dot`` is
    already full-rate ``tcgen05.mma kind::f8f6f4``, and ``tl.dot_scaled`` at this example's
    64-row tiles falls back to ``kind::f16`` (Triton 3.8, compiled for sm_100 on the CPU;
    128-row tiles reach ``kind::mxf8f6f4.block_scale``)."""
    return int(capability[0]) == 12


def block_scale_mma(ptx: str) -> bool:
    """Whether ``ptx`` runs its products on the block-scaled MMA (``mma.sync ...
    kind::mxf8f6f4.block_scale`` on sm_120, ``tcgen05.mma ... block_scale`` on sm_100)."""
    return any("mma" in line and "block_scale" in line for line in ptx.splitlines())


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
    SCALED: tl.constexpr = False,
):
    # c[m, n] = sa[m] * sb[n] * sum_k a[m, k] b[n, k] (+ bias[n]); a: [M, K] e4m3 tokens,
    # b: [N, K] e4m3 weight (nn.Linear layout), fp32 accumulation, bf16 out. SCALED: the
    # product on the block-scaled MMA with unit ue8m0 scales (bit-identical to tl.dot)
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
    if SCALED:
        # one ue8m0 scale per 32 along K, all 2^0: constants, never loaded
        ones_a = tl.full((BM, BK // 32), 127, tl.uint8)
        ones_b = tl.full((BN, BK // 32), 127, tl.uint8)
        for _ in range(0, K, BK):
            a_t = tl.load(a_ptr, mask=row_ok, other=0.0)
            acc = tl.dot_scaled(a_t, ones_a, "e4m3", tl.load(b_ptr), ones_b, "e4m3", acc)
            a_ptr += BK
            b_ptr += BK
    else:
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
    """x [..., K] bf16, w_q [N, K] e4m3, w_s [N] fp32 -> [..., N] bf16 (W8A8, fp32 acc).
    ``config``: BM, BN, BK, warps, stages, scaled (1: ``tl.dot_scaled``, 0: ``tl.dot``)."""
    N, K = w_q.shape
    bm, bn, bk, warps, stages, scaled = config
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
        SCALED=bool(scaled),
        num_warps=warps,
        num_stages=stages,
    )
    return out


def gemm_ptx(
    capability: tuple[int, int], config: tuple[int, ...] = DEFAULT, scaled: bool | None = None
) -> str:
    """PTX of :func:`_gemm_kernel` compiled for a GPU of ``capability`` with tile ``config``
    (BM, BN, BK, warps, stages) and ``scaled`` (None: :func:`block_scale_capable`). Compiles
    only (Triton's own ptxas): no launch, no GPU needed."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    bm, bn, bk, warps, stages = config[:5]
    scaled = block_scale_capable(capability) if scaled is None else scaled
    ptr = {"a": "*fp8e4nv", "b": "*fp8e4nv", "sa": "*fp32", "sb": "*fp32", "bias": "*bf16"}
    consts = {"HAS_BIAS": False, "BM": bm, "BN": bn, "BK": bk, "GM": 8, "SCALED": scaled}
    signature = {**ptr, "c": "*bf16", "M": "i32", "N": "i32", "K": "i32"}
    signature.update(dict.fromkeys(consts, "constexpr"))
    # specialised as the JIT specialises a launch: 16-byte aligned pointers, sizes divisible
    # by 16 (without it the tile loads are not vectorised and the K loop gets no cp.async)
    aligned = {(i,): [["tt.divisibility", 16]] for i in range(len(ptr) + 4)}
    source = ASTSource(fn=_gemm_kernel, signature=signature, constexprs=consts, attrs=aligned)
    target = GPUTarget("cuda", capability[0] * 10 + capability[1], 32)
    compiled = triton.compile(
        source, target=target, options={"num_warps": warps, "num_stages": stages}
    )
    return compiled.asm["ptx"]


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
    scaled: int = -1,
) -> nn.Module:
    """Tile sizes ``bm`` / ``bn`` / ``bk``, ``warps`` and ``stages`` are tuning keywords
    for ``sweep_candidate``; 0 takes the value of :data:`CONFIGS` for the weight's shape
    (else :data:`DEFAULT`). ``scaled``: 1 ``tl.dot_scaled`` (block-scaled MMA), 0 ``tl.dot``,
    -1 by the GPU (:func:`block_scale_capable`)."""
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
    if scaled < 0:
        scaled = int(block_scale_capable(torch.cuda.get_device_capability(reference.weight.device)))
    if scaled and config[2] % 32:
        scaled = 0  # one ue8m0 scale per 32 along K: BK must be a multiple of 32
    return Fp8W8A8Linear(reference, (*config, int(scaled)))
