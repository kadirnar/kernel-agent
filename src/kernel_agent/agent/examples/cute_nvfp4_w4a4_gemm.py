"""Example candidate (CuTe DSL): NVFP4 W4A4 GEMM for ``nn.Linear`` on sm_120's block-scaled
FP4 tensor-core MMA (``precision: fp4_w4a4``, issue #233), with the per-token activation
scale, the weight's tensor scale and the bias fused into the epilogue. Persistent and
warp-specialised.

Reduced precision, opt-in: only for a target whose spec says ``"precision": "fp4_w4a4"``
(a run whose ``--precisions`` names it; its tier is ``near-lossless-fp4a``); every other
tier rejects it. Written from this toolkit's FP8 example (``cute_fp8_blockscaled_gemm.py``,
itself from NVIDIA's CuTe DSL GeForce block-scaled GEMM, CUTLASS 4.6, BSD-3-Clause) with
the FP4 operand, scale-factor and MMA types swapped in; run on an RTX 5070 Ti (sm_120,
CUTLASS DSL 4.8, torch 2.14): ``selftest()`` and ``docs/research-scripts/w4a4-233/
bench_cute.py``.

Why this instruction. ``mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64
... .e2m1.e2m1.f32.ue4m3`` (CuTe DSL ``cute.nvgpu.warp.MmaMXF4NVF4Op``, sm_120a / sm_121a
only; sm_100 has the ``tcgen05`` form) multiplies packed e2m1 operands (two per byte) with
one e4m3 scale per 16 K-elements of each operand applied by the tensor core, fp32
accumulation: twice the K per instruction of the FP8 ``QMMA.SF``. cuBLASLt reaches 661
TFLOP/s with it on an RTX 5070 Ti (``F.scaled_mm`` NVFP4, the toolkit's peaks), FP8
tensor-wise 332.

Numerics (``kernel_agent.kernels.quant``, bit for bit the reference's quantisation):

* the weight once in ``build()``: ``quantize_fp4`` (NVFP4: e2m1 codes, an e4m3 scale per
  16 consecutive weights of a row, one fp32 tensor scale), scales swizzled once into
  cuBLASLt's 128 x 4 blocks (``swizzle_fp4_scales``: the MMA's ``tile_atom_to_shape_SF``
  layout); no bf16 copy;
* the activations on every call, per token (``_quant_rows_kernel``, one CTA per row, two
  passes): ``outer = amax(|row|) * NVFP4_OUTER_STEP``, block scale ``e4m3(min(bmax /
  (outer * 6), 448))``, codes ``e2m1(x / (scale * outer))`` (IEEE divisions, round to
  nearest even, saturated at ±6): ``quantize_fp4_activations(x, "nvfp4", "token")`` bit for
  bit; the scales go straight into the swizzled layout (``quant.mx_scale_offset``). In a
  fused layer, quantise in the producer (RMSNorm, ``silu(gate) * up``) instead;
* the GEMM: ``acc[m, n] = sum_k (e2m1 * s_x)(e2m1 * s_w)`` in fp32 on the tensor cores,
  then ``acc * (outer[m] * tensor_scale) (+ bias[n])`` once per output (one fma with a
  bias), one rounding to bf16. Measured: without a bias bit for bit ``quant.fp4_w4a4_linear``
  (its ``F.scaled_mm`` path) at M = 1 to 4096; with one, bit for bit ``bf16(fma(acc, s,
  bias))``, which differs from the reference's rounded product plus bias by one bf16 step
  on outputs at a rounding tie (151 of 16.8 M at 4096 x 4096 x 1024).

Kernel (``_BlockScaledGemm``): one TMA producer warp (A, B and their scale factors into a
``PipelineTmaAsync`` ring of shared-memory stages; no multicast, a 1x1x1 cluster on
GeForce) and two consumer warpgroups (8 MMA warps, ``ldmatrix`` + ``MmaMXF4NVF4Op``) on a
128x128 output tile per K step of :data:`TILE`; persistent (a static tile scheduler over
``min(tiles, SMs)`` CTAs) or one CTA per tile (``persistent=False``); epilogue through
shared memory and a TMA store. Rows ``M`` are dynamic (one compilation per weight shape);
TMA zero-fills rows past ``M`` and clips the store. ``N`` must be a multiple of 128 and
``K`` of :data:`TILE`'s K, else the module falls back to ``quant.fp4_w4a4_linear`` (same
numerics: ``F.scaled_mm`` NVFP4 there, or fp32 math).

Measured on an RTX 5070 Ti (``docs/research-scripts/w4a4-233/bench_cute.py``, CUDA graphs,
weights streamed from DRAM at M = 352; us, and the speed-up over cuBLASLt FP8 tensor-wise):

| M x K -> N          | bf16 | FP8 tensor-wise | NVFP4 F.scaled_mm | CuTe GEMM    | quant + GEMM |
|---------------------|------|-----------------|-------------------|--------------|--------------|
| 4096 x 4096 -> 4096 | 1397 | 416             | 210 (1.98x)       | 212 (1.96x)  | 262 (1.58x)  |
| 352 x 1024 -> 8192  | 69.2 | 29.6            | 14.7 (2.01x)      | 14.1 (2.10x) | 16.8 (1.76x) |
| 352 x 1024 -> 4096  | 38.2 | 18.4            | 10.2 (1.80x)      | 9.5 (1.93x)  | 11.7 (1.57x) |
| 352 x 1024 -> 2048  | 20.2 | 11.4            | 5.95 (1.92x)      | 5.39 (2.12x) | 7.82 (1.46x) |
| 352 x 2048 -> 1024  | 22.6 | 13.2            | 8.76 (1.51x)      | 8.19 (1.61x) | 12.8 (1.03x) |
| 352 x 4096 -> 1024  | 39.0 | 18.4            | 14.8 (1.25x)      | 14.5 (1.28x) | 22.2 (0.83x) |

The GEMM alone runs at the cuBLASLt NVFP4 rate (650 vs 661 TFLOP/s measured peak; 1.96x FP8
at 4096^3). The quantiser is a separate launch (2.5 us at 352 x 1024, 5.5 at 352 x 4096):
at N = 1024 with K = 4096 it eats the gain (24 output tiles on 70 SMs; split-K and a
quantiser fused into the producer of ``x`` are the next steps). Small M (memory bound:
decode) is not this kernel's job: FP4 weights (``fp4_weights``, ``cuda_fp4_gemv.py``)
stream the same bytes there.
"""

import re

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm120_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.memory import SmemAllocator, get_smem_capacity_in_bytes
from cutlass.tensor_utils import LayoutEnum
from torch import nn

from kernel_agent.kernels.quant import (
    NVFP4_OUTER_STEP,
    fp4_error,
    fp4_w4a4_linear,
    quantize_fp4,
    quantize_fp4_activations,
    swizzle_fp4_scales,
)

try:  # an on-disk cache of the compiled kernels across evaluations (kernel_agent/cute_dsl.py)
    from kernel_agent.cute_dsl import compile_cached
except ImportError:  # pragma: no cover - outside kernel-agent
    compile_cached = None

#: GPUs this example runs on (``kernel_agent.gpu_arch.supports``: ``doctor --smoke`` skips
#: it elsewhere and says why).
ARCHS = "sm_12x"
ARCHS_WHY = (
    "block-scaled FP4 mma.sync (MmaMXF4NVF4Op, kind::mxf4nvf4.block_scale) exists on "
    "sm_120a / sm_121a only; sm_100 has the tcgen05 form"
)

_NS = re.sub(r"\W", "_", __name__)
TILE = (128, 128, 128)  # (M, N, K) per CTA and K step; K in e2m1 elements
EPI_TILE = (64, 32)  # bf16 epilogue sub-tile
SF_VEC = 16  # one e4m3 scale per 16 K-elements (NVFP4)
QUANT_MAX_THREADS = 256  # one thread per 16-element block, up to this many per row
#: e2m1 rounding: the midpoints between consecutive magnitudes; on the ``_TIE_UP`` ones a
#: tie rounds up to the even code (round to nearest even), on the others down
_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_TIE_UP = (0.75, 1.75, 3.5)


# ------------------------------------------------------------------ per-token quantiser


@cute.jit
def _e2m1(x: cutlass.Float32, step: cutlass.Float32) -> cutlass.Int32:
    """e2m1 code (0..15, sign in bit 3) of ``x / step`` (an IEEE division), to nearest even,
    saturated at ±6; 0 when ``step`` is 0 (a block below e4m3's range)."""
    v = cutlass.Float32(0.0)
    if step > cutlass.Float32(0.0):
        v = x / step
    mag = cutlass.min(cute.math.absf(v), cutlass.Float32(6.0))
    idx = cutlass.Int32(0)
    for i in cutlass.range_constexpr(len(_MIDPOINTS)):
        mid = cutlass.Float32(_MIDPOINTS[i])
        up = mag >= mid if cutlass.const_expr(_MIDPOINTS[i] in _TIE_UP) else mag > mid
        idx = idx + cutlass.Int32(cutlass.select_(up, cutlass.Int32(1), cutlass.Int32(0)))
    negative = (v < cutlass.Float32(0.0)) & (idx > cutlass.Int32(0))  # no negative zero
    return idx + cutlass.Int32(cutlass.select_(negative, cutlass.Int32(8), cutlass.Int32(0)))


@cute.kernel
def _quant_rows_kernel(
    gX: cute.Tensor,
    gQ: cute.Tensor,
    gSF: cute.Tensor,
    gO: cute.Tensor,
    swizzle: cutlass.Constexpr,
    threads: cutlass.Constexpr,
):
    """One CTA of ``threads`` per row of ``gX [M, K]`` (bf16): packed e2m1 codes ``gQ [M, K /
    2]``, e4m3 block scales ``gSF`` (flat, swizzled; or ``[M, K / 16]``) and the outer scale
    ``gO[M]``. Two passes over the row (the second one hits L2), a 16-element block per
    thread and step."""
    tidx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    nblk = gX.shape[1] // SF_VEC
    smem = SmemAllocator()
    red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(threads // 32))
    xr = cute.make_rmem_tensor((1, SF_VEC), gX.element_type)
    xf = cute.make_rmem_tensor((1, SF_VEC), cutlass.Float32)

    amax = cutlass.Float32(0.0)
    for blk in range(tidx, nblk, threads):
        cute.autovec_copy(cute.local_tile(gX, (1, SF_VEC), (row, blk)), xr)
        xf.store(xr.load().to(cutlass.Float32))
        for i in cutlass.range_constexpr(SF_VEC):
            amax = cutlass.max(amax, cute.math.absf(xf[i]))
    amax = cute.arch.warp_reduction_max(amax)
    if tidx % 32 == 0:
        red[tidx // 32] = amax
    cute.arch.barrier()
    total = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(threads // 32):
        total = cutlass.max(total, red[i])
    outer = total * cutlass.Float32(NVFP4_OUTER_STEP)
    if total == cutlass.Float32(0.0):
        outer = cutlass.Float32(1.0)
    d = outer * cutlass.Float32(6.0)

    sv = cute.make_rmem_tensor((2,), cutlass.Float32)
    s8 = cute.make_rmem_tensor((2,), cutlass.Float8E4M3FN)
    qb = cute.make_rmem_tensor((1, SF_VEC // 2), cutlass.Uint8)
    for blk in range(tidx, nblk, threads):
        cute.autovec_copy(cute.local_tile(gX, (1, SF_VEC), (row, blk)), xr)
        xf.store(xr.load().to(cutlass.Float32))
        bmax = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(SF_VEC):
            bmax = cutlass.max(bmax, cute.math.absf(xf[i]))
        # e4m3 block scale (round to nearest even), then its exact fp32 value
        sv[0] = cutlass.min(bmax / d, cutlass.Float32(448.0))
        sv[1] = cutlass.Float32(0.0)
        s8.store(sv.load().to(cutlass.Float8E4M3FN))
        sv.store(s8.load().to(cutlass.Float32))
        step = sv[0] * outer
        for j in cutlass.range_constexpr(SF_VEC // 2):
            lo = _e2m1(xf[2 * j], step)
            hi = _e2m1(xf[2 * j + 1], step)
            qb[j] = (lo | (hi << 4)).to(cutlass.Uint8)  # even element in the low nibble
        cute.autovec_copy(qb, cute.local_tile(gQ, (1, SF_VEC // 2), (row, blk)))
        if cutlass.const_expr(swizzle):
            # cuBLASLt's 128 x 4 blocks (kernel_agent.kernels.quant.mx_scale_offset)
            per_row = (nblk + 3) // 4
            off = ((row // 128) * per_row + blk // 4) * 512
            off = off + (row % 32) * 16 + ((row % 128) // 32) * 4 + blk % 4
            gSF[off] = s8[0]
        else:
            gSF[row, blk] = s8[0]
    if tidx == 0:
        gO[row] = outer


@cute.jit
def _quant_rows(
    mX: cute.Tensor,
    mQ: cute.Tensor,
    mSF: cute.Tensor,
    mO: cute.Tensor,
    swizzle: cutlass.Constexpr,
    stream,
):
    sf = cute.make_tensor(cute.recast_ptr(mSF.iterator, dtype=cutlass.Float8E4M3FN), mSF.layout)
    blocks = mX.shape[1] // SF_VEC  # static: one compilation per K
    threads = min(QUANT_MAX_THREADS, max(32, -(-blocks // 32) * 32))
    _quant_rows_kernel(mX, mQ, sf, mO, swizzle, threads).launch(
        grid=(mX.shape[0], 1, 1), block=(threads, 1, 1), stream=stream
    )


# ------------------------------------------------------------------ the GEMM


class _BlockScaledGemm:
    """``C[m, n] = sum_k A[m, k] SFA[m, k/16] B[n, k] SFB[n, k/16] * sx[m] * sw[n] (+
    bias[n])``, A [M, K] and B [N, K] packed e2m1 (K-major, two per byte), SFA / SFB e4m3 in
    the MMA's swizzled layout, C [M, N] row-major bf16."""

    def __init__(self, *, has_bias: bool):
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = SF_VEC
        self.tile_shape_mnk = TILE
        self.epi_tile = EPI_TILE
        self.cluster_shape_mnk = (1, 1, 1)  # GeForce: no clusters, no TMA multicast
        self.has_bias = has_bias
        self.occupancy = 1
        self.num_mma_warps = 8
        self.tma_load_warp_id = self.num_mma_warps
        self.num_threads_per_warp = 32
        self.threads_per_cta = (self.num_mma_warps + 1) * self.num_threads_per_warp
        self.smem_capacity = get_smem_capacity_in_bytes("sm_120")
        self.buffer_align_bytes = 1024
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=self.num_mma_warps * self.num_threads_per_warp
        )
        self.load_register_requirement = 40
        self.mma_register_requirement = 232

    def _setup_attributes(self):
        mma_op = cute.nvgpu.warp.MmaMXF4NVF4Op(self.a_dtype, self.acc_dtype, self.sf_dtype)
        permutation_mnk = sm120_utils.get_permutation_mnk(
            self.tile_shape_mnk, self.sf_vec_size, False
        )
        self.tiled_mma = cute.make_tiled_mma(
            mma_op, cute.make_layout((4, 2, 1)), permutation_mnk=permutation_mnk
        )
        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
        sfa_one = blockscaled_utils.sm120_make_smem_layout_sfa(
            self.tiled_mma, self.tile_shape_mnk, self.sf_vec_size, 1
        )
        sfb_one = blockscaled_utils.sm120_make_smem_layout_sfb(
            self.tiled_mma, self.tile_shape_mnk, self.sf_vec_size, 1
        )
        self.ab_stage, self.epi_stage = self._compute_stages(sfa_one, sfb_one)
        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts()

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,  # (1, M, K / 2) uint8: packed e2m1 activations
        b: cute.Tensor,  # (1, N, K / 2) uint8: packed e2m1 weight (nn.Linear layout)
        sfa: cute.Tensor,  # uint8 e4m3, swizzled for (M, K)
        sfb: cute.Tensor,  # uint8 e4m3, swizzled for (N, K)
        c: cute.Tensor,  # (1, M, N) bf16
        sx: cute.Tensor,  # (M,) fp32 per-token outer scales
        sw: cute.Tensor,  # (N,) fp32 weight scales (the tensor scale, broadcast)
        bias: cute.Tensor,  # (N,) bf16 (unused without has_bias)
        max_active_clusters: cutlass.Constexpr,
        stream,
    ):
        # (1, M, K / 2) uint8 -> (M, K, 1) e2m1: CuTe's (mode0, mode1, batch) convention
        fp4 = cutlass.Float4E2M1FN
        a = cute.recast_tensor(a, fp4)
        b = cute.recast_tensor(b, fp4)
        a = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))
        b = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[1, 2, 0]))
        c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.sf_dtype = cutlass.Float8E4M3FN
        self.a_layout = LayoutEnum.from_tensor(a)
        self.b_layout = LayoutEnum.from_tensor(b)
        self.c_layout = LayoutEnum.from_tensor(c)
        self._setup_attributes()

        # ((Atom_M, Rest_M), (Atom_K, Rest_K), RestL): the MMA's scale-factor layout
        sfa_tensor = cute.make_tensor(
            cute.recast_ptr(sfa.iterator, dtype=self.sf_dtype),
            blockscaled_utils.tile_atom_to_shape_SF(a.shape, self.sf_vec_size),
        )
        sfb_tensor = cute.make_tensor(
            cute.recast_ptr(sfb.iterator, dtype=self.sf_dtype),
            blockscaled_utils.tile_atom_to_shape_SF(b.shape, self.sf_vec_size),
        )
        tm, tn, tk = self.tile_shape_mnk
        tma_atom_a, tma_tensor_a = self._tma_load(a, self.a_smem_layout_staged, (tm, tk))
        tma_atom_b, tma_tensor_b = self._tma_load(b, self.b_smem_layout_staged, (tn, tk))
        tma_atom_sfa, tma_tensor_sfa = self._tma_load(
            sfa_tensor, self.sfa_smem_layout_staged, (tm, tk), internal_type=cutlass.Int16
        )
        tma_atom_sfb, tma_tensor_sfb = self._tma_load(
            sfb_tensor, self.sfb_smem_layout_staged, (tn, tk), internal_type=cutlass.Int16
        )
        epi_smem_layout = cute.slice_(self.epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, self.epi_tile
        )
        tile_sched_params, grid = self._compute_grid(c, max_active_clusters)

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[self.c_dtype, cute.cosize(self.epi_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            sx,
            sw,
            bias,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
            tile_sched_params,
        ).launch(grid=grid, block=[self.threads_per_cta, 1, 1], cluster=[1, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        mSX: cute.Tensor,
        mSW: cute.Tensor,
        mBias: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        sfa_smem_layout = cute.slice_(sfa_smem_layout_staged, (None, None, 0))
        sfb_smem_layout = cute.slice_(sfb_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = (
            cute.size_in_bytes(self.a_dtype, a_smem_layout)
            + cute.size_in_bytes(self.b_dtype, b_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        )

        smem = SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_warps),
            tx_count=tma_copy_bytes,
            barrier_storage=storage.mainloop_pipeline_array_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
        )

        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        # (bM, bK, loopM, loopK, loopL) etc.
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.tile_shape_mnk, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.tile_shape_mnk, (0, None, None)), (None, None, None)
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.tile_shape_mnk, (None, 0, None)), (None, None, None)
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl, cute.slice_(self.tile_shape_mnk, (0, None, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.tile_shape_mnk, (None, None, 0)), (None, None, None)
        )
        thr_mma = tiled_mma.get_slice(tidx)

        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sSFA, 0, 2),
            cute.group_modes(gSFA_mkl, 0, 2),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sSFB, 0, 2),
            cute.group_modes(gSFB_nkl, 0, 2),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrSFA = sm120_utils.partition_fragment_SFA(sSFA[None, None, 0], thr_mma, tidx)
        tCrSFB = sm120_utils.partition_fragment_SFB(sSFB[None, None, 0], thr_mma, tidx)
        tCrSFA = cute.group_modes(tCrSFA, 2, cute.rank(tCrSFA))
        tCrSFB = cute.group_modes(tCrSFB, 2, cute.rank(tCrSFB))

        tCgC = thr_mma.partition_C(gC_mnl)
        accumulators = cute.make_rmem_tensor(tCgC.shape[:3], self.acc_dtype)
        # (m, n) inside the tile of every accumulator element, for the fused epilogue
        tCcC = thr_mma.partition_C(
            cute.make_identity_tensor(cute.slice_(self.tile_shape_mnk, (None, None, 0)))
        )
        cute.arch.sync_threads()

        k_tile_cnt = cute.size(gA_mkl, mode=[3])
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()
        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        if warp_idx < self.num_mma_warps:
            # ---------------------------------------------------------- MMA warps
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
            num_k_blocks = cute.size(tCrA, mode=[2])
            ldsm_a = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    transpose=self.a_layout.is_m_major_a(), num_matrices=4
                ),
                self.a_dtype,
            )
            ldsm_b = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    transpose=self.b_layout.is_n_major_b(), num_matrices=4
                ),
                self.b_dtype,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(ldsm_a, tiled_mma)
            smem_tiled_copy_B = cute.make_tiled_copy_B(ldsm_b, tiled_mma)
            sf_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.sf_dtype)
            smem_tiled_copy_SFA = cute.make_tiled_copy(
                sf_atom,
                sm120_utils.get_layoutSFA_TV(tiled_mma),
                (
                    cute.size(tiled_mma.permutation_mnk[0]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )
            smem_tiled_copy_SFB = cute.make_tiled_copy(
                sf_atom,
                sm120_utils.get_layoutSFB_TV(tiled_mma),
                (
                    cute.size(tiled_mma.permutation_mnk[1]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )
            thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
            thr_copy_B = smem_tiled_copy_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_A.partition_S(sA)
            tCrA_copy_view = thr_copy_A.retile(tCrA)
            tCsB_copy_view = thr_copy_B.partition_S(sB)
            tCrB_copy_view = thr_copy_B.retile(tCrB)
            thr_copy_SFA = smem_tiled_copy_SFA.get_slice(tidx)
            thr_copy_SFB = smem_tiled_copy_SFB.get_slice(tidx)
            tCsSFA_copy_view = thr_copy_SFA.partition_S(sSFA)
            tCrSFA_copy_view = cute.filter_zeros(thr_copy_SFA.retile(tCrSFA))
            tCsSFB_copy_view = thr_copy_SFB.partition_S(sSFB)
            tCrSFB_copy_view = cute.filter_zeros(thr_copy_SFB.retile(tCrSFB))

            # epilogue copies: registers -> shared memory (stmatrix layout) -> TMA store
            copy_atom_r2s = sm120_utils.sm120_get_smem_store_op(
                self.c_layout, elem_ty_d=self.c_dtype, elem_ty_acc=self.acc_dtype
            )
            copy_atom_C = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(self.c_layout.is_m_major_c(), 2),
                self.c_dtype,
            )
            tiled_copy_r2s = cute.make_tiled_copy_S(
                copy_atom_r2s, cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
            )
            thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
            tRS_sD = thr_copy_r2s.partition_D(sC)  # (R2S, R2S_M, R2S_N, PIPE_D)
            tRS_rAcc = tiled_copy_r2s.retile(accumulators)
            tRS_rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))[:3]
            tRS_rD = cute.make_rmem_tensor(tRS_rD_shape, self.acc_dtype)
            tRS_rD_out = cute.make_rmem_tensor(tRS_rD_shape, self.c_dtype)
            tma_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.epi_stage,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, self.num_mma_warps * self.num_threads_per_warp
                ),
            )
            m_last = mSX.shape[0] - 1
            epi_buffer = cutlass.Int32(0)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                gC_mnl_slice = gC_mnl[(None, None, *tile_coord_mnl)]
                accumulators.fill(0.0)

                # ------------------------------------------------ pipelined mainloop
                mainloop_consumer_state.reset_count()
                peek = cutlass.Boolean(1)
                if mainloop_consumer_state.count < k_tile_cnt:
                    peek = mainloop_pipeline.consumer_try_wait(mainloop_consumer_state)
                mainloop_pipeline.consumer_wait(mainloop_consumer_state, peek)
                tCsA_p = tCsA_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsB_p = tCsB_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsSFA_p = cute.filter_zeros(
                    tCsSFA_copy_view[None, None, None, mainloop_consumer_state.index]
                )
                tCsSFB_p = cute.filter_zeros(
                    tCsSFB_copy_view[None, None, None, mainloop_consumer_state.index]
                )
                cute.copy(smem_tiled_copy_A, tCsA_p[None, None, 0], tCrA_copy_view[None, None, 0])
                cute.copy(smem_tiled_copy_B, tCsB_p[None, None, 0], tCrB_copy_view[None, None, 0])
                cute.copy(
                    smem_tiled_copy_SFA, tCsSFA_p[None, None, 0], tCrSFA_copy_view[None, None, 0]
                )
                cute.copy(
                    smem_tiled_copy_SFB, tCsSFB_p[None, None, 0], tCrSFB_copy_view[None, None, 0]
                )

                for _k_tile in range(0, k_tile_cnt - 1, 1, unroll=1):
                    for k_block in cutlass.range_constexpr(num_k_blocks):
                        k_next = 0 if k_block + 1 == num_k_blocks else k_block + 1
                        if k_block == num_k_blocks - 1:
                            mainloop_pipeline.consumer_release(mainloop_consumer_state)
                            mainloop_consumer_state.advance()
                            peek = mainloop_pipeline.consumer_try_wait(mainloop_consumer_state)
                            idx = mainloop_consumer_state.index
                            tCsA_p = tCsA_copy_view[None, None, None, idx]
                            tCsB_p = tCsB_copy_view[None, None, None, idx]
                            tCsSFA_p = cute.filter_zeros(tCsSFA_copy_view[None, None, None, idx])
                            tCsSFB_p = cute.filter_zeros(tCsSFB_copy_view[None, None, None, idx])
                            mainloop_pipeline.consumer_wait(mainloop_consumer_state, peek)
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            [tCrA[None, None, k_block], tCrSFA[None, None, k_block]],
                            [tCrB[None, None, k_block], tCrSFB[None, None, k_block]],
                            accumulators,
                        )
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, k_next],
                            tCrA_copy_view[None, None, k_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[None, None, k_next],
                            tCrB_copy_view[None, None, k_next],
                        )
                        cute.copy(
                            smem_tiled_copy_SFA,
                            tCsSFA_p[None, None, k_next],
                            tCrSFA_copy_view[None, None, k_next],
                        )
                        cute.copy(
                            smem_tiled_copy_SFB,
                            tCsSFB_p[None, None, k_next],
                            tCrSFB_copy_view[None, None, k_next],
                        )
                # the last k tile, hoisted: release the stage before the epilogue
                for k_block in cutlass.range_constexpr(num_k_blocks):
                    k_next = 0 if k_block + 1 == num_k_blocks else k_block + 1
                    if k_block == num_k_blocks - 1:
                        cute.arch.fence_proxy("async.shared", space="cta")
                        mainloop_pipeline.consumer_release(mainloop_consumer_state)
                        mainloop_consumer_state.advance()
                    if k_next > 0:
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, k_next],
                            tCrA_copy_view[None, None, k_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[None, None, k_next],
                            tCrB_copy_view[None, None, k_next],
                        )
                        cute.copy(
                            smem_tiled_copy_SFA,
                            tCsSFA_p[None, None, k_next],
                            tCrSFA_copy_view[None, None, k_next],
                        )
                        cute.copy(
                            smem_tiled_copy_SFB,
                            tCsSFB_p[None, None, k_next],
                            tCrSFB_copy_view[None, None, k_next],
                        )
                    cute.gemm(
                        tiled_mma,
                        accumulators,
                        [tCrA[None, None, k_block], tCrSFA[None, None, k_block]],
                        [tCrB[None, None, k_block], tCrSFB[None, None, k_block]],
                        accumulators,
                    )

                # ------------------------------------------------ fused epilogue math
                # acc * sx[m] * sw[n] (+ bias[n]) in fp32; rows past M (zero-filled A) are
                # clamped for the loads and clipped by the TMA store
                tile_m0 = tile_coord_mnl[0] * self.tile_shape_mnk[0]
                tile_n0 = tile_coord_mnl[1] * self.tile_shape_mnk[1]
                for i in cutlass.range_constexpr(cute.size(accumulators)):
                    coord = tCcC[i]
                    m = cutlass.min(tile_m0 + coord[0], m_last)
                    n = tile_n0 + coord[1]
                    v = accumulators[i] * (mSX[m] * mSW[n])
                    if cutlass.const_expr(self.has_bias):
                        v = v + mBias[n].to(cutlass.Float32)
                    accumulators[i] = v

                # ------------------------------------------------ store: smem + TMA
                bSG_sD, bSG_gD = cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sC, 0, 2),
                    cute.zipped_divide(gC_mnl_slice, self.epi_tile),
                )
                epi_rest_m = bSG_gD.shape[1][0]
                epi_rest_n = bSG_gD.shape[1][1]
                mma_tile_m = self.tile_shape_mnk[0] // cute.size(tRS_rAcc, mode=[1])
                mma_tile_n = self.tile_shape_mnk[1] // cute.size(tRS_rAcc, mode=[2])
                mma_m_per_epi = self.epi_tile[0] // mma_tile_m
                mma_n_per_epi = self.epi_tile[1] // mma_tile_n
                for epi_m in cutlass.range_constexpr(epi_rest_m):
                    for epi_n in cutlass.range_constexpr(epi_rest_n):
                        for mma_n_in_epi in cutlass.range_constexpr(mma_n_per_epi):
                            for mma_m_in_epi in cutlass.range_constexpr(mma_m_per_epi):
                                mma_n = epi_n * mma_n_per_epi + mma_n_in_epi
                                mma_m = epi_m * mma_m_per_epi + mma_m_in_epi
                                dst = tRS_rD[(None, mma_m_in_epi, mma_n_in_epi)]
                                src = tRS_rAcc[(None, mma_m, mma_n)]
                                for e in cutlass.range_constexpr(cute.size(dst)):
                                    dst[e] = src[e]
                        tRS_rD_out.store(tRS_rD.load().to(self.c_dtype))
                        epi_buffer = (epi_buffer + 1) % cute.size(tRS_sD, mode=[3])
                        self.epilog_sync_barrier.arrive_and_wait()
                        cute.copy(
                            tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, epi_buffer)]
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()
                        if warp_idx == 0:
                            cute.copy(
                                tma_atom_c,
                                bSG_sD[(None, epi_buffer)],
                                bSG_gD[(None, (epi_m, epi_n))],
                            )
                            tma_store_pipeline.producer_commit()
                            tma_store_pipeline.producer_acquire()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            tma_store_pipeline.producer_tail()  # the last TMA stores read shared memory
        elif warp_idx == self.tma_load_warp_id:
            # ---------------------------------------------------------- TMA producer warp
            cute.arch.setmaxregister_decrease(self.load_register_requirement)
            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                tBgB_nkl = tBgB[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]
                tAgSFA_mkl = tAgSFA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                tBgSFB_nkl = tBgSFB[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]
                mainloop_producer_state.reset_count()
                for _k_tile in range(0, k_tile_cnt, 1, unroll=1):
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)
                    count = mainloop_producer_state.count
                    stage = mainloop_producer_state.index
                    bar = mainloop_pipeline.producer_get_barrier(mainloop_producer_state)
                    cute.copy(
                        tma_atom_a, tAgA_mkl[(None, count)], tAsA[(None, stage)], tma_bar_ptr=bar
                    )
                    cute.copy(
                        tma_atom_b, tBgB_nkl[(None, count)], tBsB[(None, stage)], tma_bar_ptr=bar
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_mkl[(None, count)],
                        tAsSFA[(None, stage)],
                        tma_bar_ptr=bar,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_nkl[(None, count)],
                        tBsSFB[(None, stage)],
                        tma_bar_ptr=bar,
                    )
                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            mainloop_pipeline.producer_tail(mainloop_producer_state)

    # -------------------------------------------------------------- host-side helpers

    def _compute_stages(self, sfa_layout: cute.Layout, sfb_layout: cute.Layout):
        tm, tn, tk = self.tile_shape_mnk
        epi_stage = min((tn // self.epi_tile[1]) * (tm // self.epi_tile[0]), 4)
        epi_bytes = cute.size(self.epi_tile) * self.c_dtype.width // 8 * epi_stage
        ab_bytes = (tm * tk * self.a_dtype.width + tn * tk * self.b_dtype.width) // 8
        sf_bytes = (
            cute.size(cute.filter_zeros(sfa_layout).shape) * self.sf_dtype.width // 8
            + cute.size(cute.filter_zeros(sfb_layout).shape) * self.sf_dtype.width // 8
        )
        free = (self.smem_capacity - self.occupancy * 1024) // self.occupancy - 1024 - epi_bytes
        return free // (ab_bytes + sf_bytes), epi_stage

    def _make_smem_layouts(self):
        tm, tn, tk = self.tile_shape_mnk
        a_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(self.a_layout, self.a_dtype, tk), self.a_dtype
        )
        a_staged = cute.tile_to_shape(a_atom, (tm, tk, self.ab_stage), order=(0, 1, 2))
        b_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(self.b_layout, self.b_dtype, tk), self.b_dtype
        )
        b_staged = cute.tile_to_shape(b_atom, (tn, tk, self.ab_stage), order=(0, 1, 2))
        sfa_staged = blockscaled_utils.sm120_make_smem_layout_sfa(
            self.tiled_mma, self.tile_shape_mnk, self.sf_vec_size, self.ab_stage
        )
        sfb_staged = blockscaled_utils.sm120_make_smem_layout_sfb(
            self.tiled_mma, self.tile_shape_mnk, self.sf_vec_size, self.ab_stage
        )
        c_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(self.c_layout, self.c_dtype, self.epi_tile[1]),
            self.c_dtype,
        )
        epi_staged = cute.tile_to_shape(c_atom, (*self.epi_tile, self.epi_stage), order=(0, 1, 2))
        return a_staged, b_staged, sfa_staged, sfb_staged, epi_staged

    def _compute_grid(self, c: cute.Tensor, max_active_clusters: cutlass.Constexpr):
        gc = cute.zipped_divide(c, tiler=cute.slice_(self.tile_shape_mnk, (None, None, 0)))
        params = utils.PersistentTileSchedulerParams(gc[(0, (None, None, None))].shape, (1, 1, 1))
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(params, max_active_clusters)
        return params, grid

    @staticmethod
    def _tma_load(tensor, smem_layout_staged, tile, internal_type=None):
        return cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            tensor,
            cute.slice_(smem_layout_staged, (None, None, 0)),
            tile,
            num_multicast=1,
            internal_type=internal_type,
        )


# ------------------------------------------------------------------ compilation


def _compile(fn, *args, key):
    options = "--enable-tvm-ffi"
    if compile_cached is not None:
        return compile_cached(
            fn, *args, key=key, options=options, name=getattr(fn, "__name__", None)
        )
    return cute.compile(fn, *args, options=options)


def compile_quant(k: int, swizzle: bool = True):
    """The per-token NVFP4 quantiser for rows of ``k`` bf16 elements (``k % 16 == 0``):
    scales into the swizzled layout (``swizzle``) or ``[rows, k / 16]``."""
    m = cute.sym_int()
    x = make_fake_compact_tensor(cutlass.BFloat16, (m, k), stride_order=(1, 0), assumed_align=16)
    q = make_fake_compact_tensor(cutlass.Uint8, (m, k // 2), stride_order=(1, 0), assumed_align=8)
    if swizzle:
        sf = make_fake_compact_tensor(cutlass.Uint8, (cute.sym_int(),), assumed_align=16)
    else:
        sf = make_fake_compact_tensor(cutlass.Uint8, (m, k // SF_VEC), stride_order=(1, 0))
    o = make_fake_compact_tensor(cutlass.Float32, (m,), assumed_align=4)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    return _compile(_quant_rows, x, q, sf, o, swizzle, stream, key=("quant", k, swizzle))


def compile_gemm(n: int, k: int, *, has_bias: bool = False, max_ctas: int = 1 << 20):
    """The GEMM for an [n, k] e2m1 weight (``n % 128 == 0``, ``k`` a multiple of
    :data:`TILE`'s K), any number of rows. ``max_ctas``: the SM count for the persistent
    schedule, huge for one CTA per tile."""
    m = cute.sym_int()
    order3 = (2, 1, 0)  # (1, rows, cols), cols innermost
    a = make_fake_compact_tensor(
        cutlass.Uint8, (1, m, k // 2), stride_order=order3, assumed_align=16
    )
    b = make_fake_compact_tensor(
        cutlass.Uint8, (1, n, k // 2), stride_order=order3, assumed_align=16
    )
    sfa = make_fake_compact_tensor(cutlass.Uint8, (cute.sym_int(),), assumed_align=16)
    sfb = make_fake_compact_tensor(cutlass.Uint8, (n * k // SF_VEC,), assumed_align=16)
    c = make_fake_compact_tensor(cutlass.BFloat16, (1, m, n), stride_order=order3, assumed_align=16)
    sx = make_fake_compact_tensor(cutlass.Float32, (m,), assumed_align=4)
    sw = make_fake_compact_tensor(cutlass.Float32, (n,), assumed_align=4)
    bias = make_fake_compact_tensor(cutlass.BFloat16, (n,), assumed_align=2)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    gemm = _BlockScaledGemm(has_bias=has_bias)
    key = ("gemm", n, k, has_bias, max_ctas, TILE, EPI_TILE)
    return _compile(gemm, a, b, sfa, sfb, c, sx, sw, bias, max_ctas, stream, key=key)


def swizzled_scale_bytes(rows: int, k: int) -> int:
    """Bytes of the swizzled e4m3 scales of a [rows, k] operand (rows padded to 128, k / 16
    scales to 4)."""
    return -(-rows // 128) * 128 * (-(-(k // SF_VEC) // 4) * 4)


# ------------------------------------------------------------------ the module


def _launch(mod: "CuteNvfp4Linear", x: torch.Tensor) -> torch.Tensor:
    n, k = mod.out_features, mod.in_features
    x2 = x.reshape(-1, k)
    if x2.stride(-1) != 1 or x2.data_ptr() % 16:
        x2 = x2.contiguous()
    rows = x2.shape[0]
    out = torch.empty((rows, n), device=x.device, dtype=torch.bfloat16)
    if rows == 0:
        return out.view(*x.shape[:-1], n)
    xq = torch.empty((rows, k // 2), device=x.device, dtype=torch.uint8)
    # padding rows (past ``rows`` in the last 128-row block) meet TMA's zero-filled A rows
    # and are clipped by the store: their scale bytes need no initialisation
    sfa = torch.empty((swizzled_scale_bytes(rows, k),), device=x.device, dtype=torch.uint8)
    outer = torch.empty((rows,), device=x.device, dtype=torch.float32)
    mod.quant(x2, xq, sfa, outer)
    mod.gemm(
        xq.unsqueeze(0),
        mod.weight_codes.unsqueeze(0),
        sfa,
        mod.sfb,
        out.unsqueeze(0),
        outer,
        mod.weight_scale,
        mod.bias_or_zero,
    )
    return out.view(*x.shape[:-1], n)


class CuteNvfp4Linear(nn.Module):
    """``nn.Linear`` with NVFP4 weights (e2m1 + an e4m3 scale per 16 + an fp32 tensor scale)
    and NVFP4 activations (per token, every call) on the block-scaled FP4 MMA."""

    def __init__(self, reference: nn.Linear, persistent: bool = True) -> None:
        super().__init__()
        self.in_features, self.out_features = reference.in_features, reference.out_features
        dev = reference.weight.device
        codes, scales, tensor_scale = quantize_fp4(reference.weight)  # once, here
        self.weight_codes = codes
        self.weight_scales, self.tensor_scale = scales, tensor_scale
        self.bias = reference.bias
        self.quant_error = fp4_error(reference.weight, codes, scales, tensor_scale)  # NOTES.md
        n, k = self.out_features, self.in_features
        self.sfb = swizzle_fp4_scales(scales).view(torch.uint8)
        self.weight_scale = torch.full((n,), float(tensor_scale), device=dev, dtype=torch.float32)
        self.bias_or_zero = (
            reference.bias.detach().to(torch.bfloat16)
            if reference.bias is not None
            else torch.zeros(n, device=dev, dtype=torch.bfloat16)
        )
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        self.max_ctas = sms if persistent else 1 << 20
        self.quant = compile_quant(k)
        self.gemm = compile_gemm(n, k, has_bias=reference.bias is not None, max_ctas=self.max_ctas)
        self._key = len(_MODULES)  # the custom op's handle on this module
        _MODULES.append(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.bfloat16 or not x.is_cuda:
            return fp4_w4a4_linear(
                x, self.weight_codes, self.weight_scales, self.tensor_scale, self.bias
            )
        # compiled: one opaque op; eager: the launcher itself (no custom-op dispatch cost)
        launch = cute_nvfp4_linear if torch.compiler.is_compiling() else _eager
        return launch(x, self._key)


_MODULES: list[CuteNvfp4Linear] = []  # compiled kernels are not tensors: ops get an index


def _eager(x: torch.Tensor, key: int) -> torch.Tensor:
    return _launch(_MODULES[key], x)


# One opaque op under torch.compile / CUDA graphs (fresh output, no in-place writes).
cute_nvfp4_linear = torch.library.custom_op(f"{_NS}::cute_nvfp4_linear", _eager, mutates_args=())


@cute_nvfp4_linear.register_fake
def _(x, key):
    return x.new_empty((*x.shape[:-1], _MODULES[key].out_features), dtype=torch.bfloat16)


def build(reference: nn.Module, persistent: bool = True) -> nn.Module:
    """``persistent``: a static persistent schedule over the SMs (True) or one CTA per
    output tile (False), a tuning keyword for ``sweep_candidate``."""
    ok = (
        isinstance(reference, nn.Linear)
        and reference.weight.dtype == torch.bfloat16
        and reference.weight.is_cuda
        and torch.cuda.get_device_capability(reference.weight.device)[0] == 12  # sm_120/121
        and reference.out_features % TILE[1] == 0
        and reference.in_features % TILE[2] == 0
        and (reference.bias is None or reference.bias.dtype == torch.bfloat16)
    )
    if not ok:
        return reference
    return CuteNvfp4Linear(reference, persistent=persistent)


_QUANT_PLAIN: dict[int, object] = {}


def quantize_activations(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """This example's activation quantisation of ``x [..., K]``: packed e2m1 codes ``[rows,
    K / 2]``, unswizzled e4m3 scales ``[rows, K / 16]`` and the outer fp32 scales ``[rows]``
    (``quant.quantize_fp4_activations(x, "nvfp4")`` bit for bit). CUDA bf16 on sm_120: the
    CuTe kernel; else the reference math."""
    x2 = x.reshape(-1, x.shape[-1])
    k = x2.shape[-1]
    ok = x2.is_cuda and x2.dtype == torch.bfloat16 and k % SF_VEC == 0 and x2.shape[0] > 0
    if not ok or torch.cuda.get_device_capability(x2.device)[0] != 12:
        return quantize_fp4_activations(x2, "nvfp4")
    x2 = x2.contiguous()
    rows = x2.shape[0]
    if k not in _QUANT_PLAIN:
        _QUANT_PLAIN[k] = compile_quant(k, swizzle=False)
    q = torch.empty((rows, k // 2), device=x2.device, dtype=torch.uint8)
    sf = torch.empty((rows, k // SF_VEC), device=x2.device, dtype=torch.uint8)
    outer = torch.empty((rows,), device=x2.device, dtype=torch.float32)
    _QUANT_PLAIN[k](x2, q, sf, outer)
    return q, sf.view(torch.float8_e4m3fn), outer


# ------------------------------------------------------------------ selftest (GPU)


def selftest(m: int = 352, n: int = 1024, k: int = 1024, seed: int = 0) -> dict:
    """Run on an sm_120 GPU: the quantiser against ``quantize_fp4_activations`` (bit for
    bit: codes, scales, outer scales) and the module against ``fp4_w4a4_linear`` (the same
    codes; the outputs differ only by the summation order and one bf16 rounding), with and
    without a bias. Returns the relative errors; raises AssertionError past 1e-2."""
    torch.manual_seed(seed)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    x[:, 3] *= 50  # an outlier channel
    q, sf, outer = quantize_activations(x)
    rq, rsf, router = quantize_fp4_activations(x, "nvfp4")
    assert torch.equal(q, rq), "codes differ from quantize_fp4_activations"
    assert torch.equal(sf.view(torch.uint8), rsf.view(torch.uint8)), "scales differ"
    assert torch.equal(outer, router), "outer scales differ"

    def rel(a, b):
        return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))

    out = {}
    for bias in (True, False):
        lin = nn.Linear(k, n, bias=bias, device="cuda", dtype=torch.bfloat16)
        mod = build(lin)
        assert isinstance(mod, CuteNvfp4Linear), "build() fell back to the reference"
        ref = fp4_w4a4_linear(x, mod.weight_codes, mod.weight_scales, mod.tensor_scale, lin.bias)
        out["bias" if bias else "no_bias"] = rel(mod(x), ref)
    torch.cuda.synchronize()
    for name, value in out.items():
        assert value < 1e-2, f"{name}: relative L2 {value:.3g} >= 1e-2"
    return out
