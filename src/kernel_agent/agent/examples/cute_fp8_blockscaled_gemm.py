"""Example candidate (CuTe DSL): FP8 W8A8 GEMM for ``nn.Linear`` on sm_120's block-scaled
tensor-core MMA, with the scales, bias, a residual and an optional e4m3 output fused into
the epilogue. Persistent and warp-specialised.

Status: written from NVIDIA's CuTe DSL GeForce example
(``examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/
dense_blockscaled_gemm_persistent_cooperative.py``, CUTLASS 4.6, BSD-3-Clause) and traced
on the CPU; **not yet run on a GPU**. ``kernel-agent doctor --smoke`` (sm_120) and
``pytest -m gpu tests/test_cute_examples.py`` run its selftest. Keep that in mind before
copying it: evaluate a copy with ``mode="quick"`` first.

Why this instruction. On sm_120 (GeForce Blackwell) e4m3 ``mma.sync`` with fp32
accumulation (SASS ``QMMA.F32``: Triton ``tl.dot``, row-wise ``_scaled_mm``, CUTLASS
``SM120_16x8x32_TN``) runs at half rate, 208 TFLOP/s on an RTX 5070 Ti; the block-scaled
form ``mma.sync.kind::mxf8f6f4.block_scale.scale_vec::1X ... .ue8m0`` (``QMMA.SF``) runs at
416 with the same fp32 accumulation (docs/RESEARCH-TRITON.md §1.1). CuTe DSL exposes it as
``cute.nvgpu.warp.MmaMXF8Op``. The per-token / per-channel scales of a W8A8 layer are not
32-element blocks, so the MMA gets **unit ue8m0 scales** (byte 127 = 2^0: the product is
exactly the fp32-accumulated e4m3 GEMM) and the real scales are applied once per output in
the epilogue. Real MXFP8 scales (an ``fp8_mx`` target: one ue8m0 per 32 K-elements) can
be passed instead: ``kernel_agent.kernels.quant.quantize_mxfp8`` codes and
``swizzle_mx_scales(scales).view(torch.uint8)`` bytes as ``sfa`` / ``sfb`` (cuBLASLt's
128x4 ``SWIZZLE_32_4_4`` blocks are the MMA's ``tile_atom_to_shape_SF`` layout), with
``sx`` / ``sw`` all ones.

Kernel (``_BlockScaledGemm``):

* one TMA producer warp (``cp.async.bulk.tensor``: A, B and their scale factors into a
  ``PipelineTmaAsync`` ring of shared-memory stages; no multicast and a 1x1x1 cluster on
  GeForce) and two consumer warpgroups (8 MMA warps, ``ldmatrix`` + ``MmaMXF8Op``) working
  on the same 128x128x128 output tile (the "cooperative" schedule), registers rebalanced
  with ``setmaxnreg`` (40 for the producer, 232 for the MMA warps);
* persistent (``persistent=True``): a static tile scheduler over ``min(tiles, SMs)`` CTAs,
  so the producer prefetches the next tile's operands while the MMA warps run the
  epilogue; ``persistent=False`` launches one CTA per tile (the same code, one tile each);
* epilogue on the fp32 accumulators: ``acc * sx[m] * sw[n] (+ bias[n]) (+ residual[m, n])``,
  then bf16, or e4m3 after ``/ out_scale`` (a per-tensor scale the caller supplies, e.g.
  the next GEMM's input scale), through shared memory and a TMA store;
* rows ``M`` are dynamic (``cute.sym_int``: one compilation per weight shape); TMA
  zero-fills the rows past ``M`` and clips the store; ``N`` and ``K`` must be multiples of
  128 (the only FP8 tile shape of the upstream example), else the module falls back to
  ``kernel_agent.kernels.quant.fp8_w8a8_linear`` (same numerics).

Activations are quantised per token (``_quant_rows_kernel``: amax / 448, e4m3 codes) every
call; in a fused layer, quantise in the producer (RMSNorm, silu·up) instead (FP8.md §4).
Compiled once per (N, K, flags) with fake tensors and TVM-FFI, through
``kernel_agent.cute_dsl.compile_cached`` (an on-disk cache across evaluations; plain
``cute.compile`` when kernel_agent is not importable).

Baseline to beat at M = 352 (docs/FP8.md §3.1, L2-cold, CUDA graph): cuBLASLt tensor-wise
(nvjet) gate|up 352x8192x1024 29.5 us, cuBLASLt MXFP8 24.7 us, Triton ``tl.dot`` 35.3 us.
Small M (< 128 rows, memory bound) belongs to the CUDA C++ GEMV / skinny examples.
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
from cutlass.memory import get_smem_capacity_in_bytes
from cutlass.tensor_utils import LayoutEnum
from torch import nn

from kernel_agent.kernels.quant import fp8_error, fp8_w8a8_linear, quantize_fp8

try:  # an on-disk cache of the compiled kernels across evaluations (kernel_agent/cute_dsl.py)
    from kernel_agent.cute_dsl import compile_cached
except ImportError:  # pragma: no cover - outside kernel-agent
    compile_cached = None

#: GPUs this example runs on (``kernel_agent.gpu_arch.supports``: ``doctor --smoke`` skips
#: it elsewhere and says why).
ARCHS = "sm_12x"
ARCHS_WHY = (
    "block-scaled mma.sync (MmaMXF8Op, kind::mxf8f6f4.block_scale) exists on sm_120a / "
    "sm_121a only; sm_100 has the tcgen05 form"
)

_NS = re.sub(r"\W", "_", __name__)
TILE = (128, 128, 128)  # the FP8 tile of the upstream example (M, N, K)
EPI_TILE = (64, 32)  # bf16 epilogue sub-tile: 4 smem stages of 64x32 leave room for 2 AB stages
#: e4m3 epilogue sub-tile: the 8-bit register -> smem copy covers the tile's 128 rows at once
#: (with 64-row sub-tiles nothing would be copied and the output would be garbage)
EPI_TILE_FP8 = (128, 32)
SF_VEC = 32  # one ue8m0 scale per 32 K-elements (MXFP8)
SF_UNIT = 127  # ue8m0 code of 2^0
FP8_MAX = 448.0
QUANT_THREADS = 256
VEC = 8  # elements per vector load in the quantiser


# ------------------------------------------------------------------ per-token quantiser


@cute.kernel
def _quant_rows_kernel(gX: cute.Tensor, gQ: cute.Tensor, gS: cute.Tensor):
    """One CTA per row: scale = amax / 448, codes = x / scale in e4m3 (round to nearest,
    saturating). Two passes over the row (the second one hits L2)."""
    tidx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    chunks = gX.shape[1] // VEC
    smem = cutlass.utils.SmemAllocator()
    red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(QUANT_THREADS // 32))
    xr = cute.make_rmem_tensor((1, VEC), gX.element_type)
    xf = cute.make_rmem_tensor((1, VEC), cutlass.Float32)
    qr = cute.make_rmem_tensor((1, VEC), gQ.element_type)

    amax = cutlass.Float32(0.0)
    for chunk in range(tidx, chunks, QUANT_THREADS):
        cute.autovec_copy(cute.local_tile(gX, (1, VEC), (row, chunk)), xr)
        xf.store(xr.load().to(cutlass.Float32))
        for i in cutlass.range_constexpr(VEC):
            amax = cutlass.max(amax, cute.math.absf(xf[i]))
    amax = cute.arch.warp_reduction_max(amax)
    if tidx % 32 == 0:
        red[tidx // 32] = amax
    cute.arch.barrier()
    total = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(QUANT_THREADS // 32):
        total = cutlass.max(total, red[i])
    # a row of zeros: any scale gives codes 0 and an output of 0
    scale = cutlass.max(total, cutlass.Float32(1e-12)) / FP8_MAX
    inv = cutlass.Float32(1.0) / scale
    if tidx == 0:
        gS[row] = scale
    for chunk in range(tidx, chunks, QUANT_THREADS):
        cute.autovec_copy(cute.local_tile(gX, (1, VEC), (row, chunk)), xr)
        qr.store((xr.load().to(cutlass.Float32) * inv).to(gQ.element_type))
        cute.autovec_copy(qr, cute.local_tile(gQ, (1, VEC), (row, chunk)))


@cute.jit
def _quant_rows(mX: cute.Tensor, mQ: cute.Tensor, mS: cute.Tensor, stream):
    q = cute.make_tensor(cute.recast_ptr(mQ.iterator, dtype=cutlass.Float8E4M3FN), mQ.layout)
    _quant_rows_kernel(mX, q, mS).launch(
        grid=(mX.shape[0], 1, 1), block=(QUANT_THREADS, 1, 1), stream=stream
    )


# ------------------------------------------------------------------ the GEMM


class _BlockScaledGemm:
    """``C[m, n] = epi(sum_k A[m, k] SFA[m, k/32] B[n, k] SFB[n, k/32])``, A [M, K] and
    B [N, K] e4m3 (K-major), SFA / SFB ue8m0 in the MMA's swizzled layout, C [M, N]
    row-major bf16 or e4m3. ``epi``: ``* sx[m] * sw[n] (+ bias[n]) (+ res[m, n])`` and, for an
    e4m3 C, ``/ out_scale[0]``."""

    def __init__(self, *, has_bias: bool, has_residual: bool, out_fp8: bool):
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = SF_VEC
        self.tile_shape_mnk = TILE
        self.epi_tile = EPI_TILE_FP8 if out_fp8 else EPI_TILE
        self.cluster_shape_mnk = (1, 1, 1)  # GeForce: no clusters, no TMA multicast
        self.has_bias = has_bias
        self.has_residual = has_residual
        self.out_fp8 = out_fp8
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
        mma_op = cute.nvgpu.warp.MmaMXF8Op(self.a_dtype, self.acc_dtype, self.sf_dtype)
        permutation_mnk = sm120_utils.get_permutation_mnk(
            self.tile_shape_mnk, self.sf_vec_size, True
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
        a: cute.Tensor,  # (1, M, K) uint8: e4m3 activations
        b: cute.Tensor,  # (1, N, K) uint8: e4m3 weight (nn.Linear layout)
        sfa: cute.Tensor,  # uint8 ue8m0, swizzled for (M, K)
        sfb: cute.Tensor,  # uint8 ue8m0, swizzled for (N, K)
        c: cute.Tensor,  # (1, M, N) bf16, or uint8 for e4m3
        sx: cute.Tensor,  # (M,) fp32 per-token scales
        sw: cute.Tensor,  # (N,) fp32 per-channel scales
        bias: cute.Tensor,  # (N,) bf16 (unused without has_bias)
        residual: cute.Tensor,  # (R, N) bf16 (unused without has_residual)
        out_scale: cute.Tensor,  # (1,) fp32 (unused without out_fp8)
        max_active_clusters: cutlass.Constexpr,
        stream,
    ):
        # (1, M, K) -> (M, K, 1): CuTe's (mode0, mode1, batch) convention; uint8 -> e4m3
        fp8 = cutlass.Float8E4M3FN
        a = cute.make_tensor(
            cute.recast_ptr(a.iterator, dtype=fp8), cute.select(a.layout, mode=[1, 2, 0])
        )
        b = cute.make_tensor(
            cute.recast_ptr(b.iterator, dtype=fp8), cute.select(b.layout, mode=[1, 2, 0])
        )
        c_ptr = c.iterator
        if cutlass.const_expr(self.out_fp8):
            c_ptr = cute.recast_ptr(c.iterator, dtype=fp8)
        c = cute.make_tensor(c_ptr, cute.select(c.layout, mode=[1, 2, 0]))
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.sf_dtype = cutlass.Float8E8M0FNU
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
            residual,
            out_scale,
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
        mRes: cute.Tensor,
        mOutScale: cute.Tensor,
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

        smem = cutlass.utils.SmemAllocator()
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
                # acc * sx[m] * sw[n] (+ bias[n]) (+ res[m, n]) (/ out_scale), in fp32;
                # rows past M (zero-filled A) are clamped for the loads and clipped by the
                # TMA store
                tile_m0 = tile_coord_mnl[0] * self.tile_shape_mnk[0]
                tile_n0 = tile_coord_mnl[1] * self.tile_shape_mnk[1]
                inv_out = cutlass.Float32(1.0)
                if cutlass.const_expr(self.out_fp8):
                    inv_out = cutlass.Float32(1.0) / mOutScale[0]
                for i in cutlass.range_constexpr(cute.size(accumulators)):
                    coord = tCcC[i]
                    m = cutlass.min(tile_m0 + coord[0], m_last)
                    n = tile_n0 + coord[1]
                    v = accumulators[i] * mSX[m] * mSW[n]
                    if cutlass.const_expr(self.has_bias):
                        v = v + mBias[n].to(cutlass.Float32)
                    if cutlass.const_expr(self.has_residual):
                        v = v + mRes[m, n].to(cutlass.Float32)
                    if cutlass.const_expr(self.out_fp8):
                        v = cutlass.max(
                            cutlass.min(v * inv_out, cutlass.Float32(FP8_MAX)),
                            cutlass.Float32(-FP8_MAX),
                        )
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
                assert mma_m_per_epi >= 1 and mma_n_per_epi >= 1, (
                    f"epilogue tile {self.epi_tile} is smaller than the copy's "
                    f"{mma_tile_m}x{mma_tile_n} sub-tile for {self.c_dtype}"
                )
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


def compile_quant(k: int):
    """The per-token quantiser for rows of ``k`` bf16 elements (``k % 8 == 0``)."""
    m = cute.sym_int()
    x = make_fake_compact_tensor(cutlass.BFloat16, (m, k), stride_order=(1, 0), assumed_align=16)
    q = make_fake_compact_tensor(cutlass.Uint8, (m, k), stride_order=(1, 0), assumed_align=16)
    s = make_fake_compact_tensor(cutlass.Float32, (m,), assumed_align=4)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    return _compile(_quant_rows, x, q, s, stream, key=("quant", k))


def compile_gemm(
    n: int,
    k: int,
    *,
    has_bias: bool = False,
    has_residual: bool = False,
    out_fp8: bool = False,
    max_ctas: int = 1 << 20,
):
    """The GEMM for an [n, k] e4m3 weight (``n % 128 == k % 128 == 0``), any number of rows.
    ``max_ctas``: the SM count for the persistent schedule, huge for one CTA per tile."""
    m = cute.sym_int()
    order3 = (2, 1, 0)  # (1, rows, cols), cols innermost
    a = make_fake_compact_tensor(cutlass.Uint8, (1, m, k), stride_order=order3, assumed_align=16)
    b = make_fake_compact_tensor(cutlass.Uint8, (1, n, k), stride_order=order3, assumed_align=16)
    sfa = make_fake_compact_tensor(cutlass.Uint8, (cute.sym_int(),), assumed_align=16)
    sfb = make_fake_compact_tensor(cutlass.Uint8, (n * k // SF_VEC,), assumed_align=16)
    c_type = cutlass.Uint8 if out_fp8 else cutlass.BFloat16
    c = make_fake_compact_tensor(c_type, (1, m, n), stride_order=order3, assumed_align=16)
    sx = make_fake_compact_tensor(cutlass.Float32, (m,), assumed_align=4)
    sw = make_fake_compact_tensor(cutlass.Float32, (n,), assumed_align=4)
    bias = make_fake_compact_tensor(cutlass.BFloat16, (n,), assumed_align=2)
    res = make_fake_compact_tensor(
        cutlass.BFloat16, (cute.sym_int(), n), stride_order=(1, 0), assumed_align=16
    )
    out_scale = make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    gemm = _BlockScaledGemm(has_bias=has_bias, has_residual=has_residual, out_fp8=out_fp8)
    key = ("gemm", n, k, has_bias, has_residual, out_fp8, max_ctas, TILE, EPI_TILE, EPI_TILE_FP8)
    return _compile(
        gemm, a, b, sfa, sfb, c, sx, sw, bias, res, out_scale, max_ctas, stream, key=key
    )


def unit_scales(rows: int, k: int, device) -> torch.Tensor:
    """ue8m0 bytes of 2^0 for a [rows, k] operand, sized for the MMA's swizzled layout
    (rows padded to 128, k / 32 scales padded to 4)."""
    padded_rows = -(-rows // 128) * 128
    sf_k = -(-(k // SF_VEC) // 4) * 4
    return torch.full((padded_rows * sf_k,), SF_UNIT, dtype=torch.uint8, device=device)


# ------------------------------------------------------------------ the module


def _launch(mod: "CuteFp8Linear", x: torch.Tensor, residual: torch.Tensor | None) -> torch.Tensor:
    n, k = mod.out_features, mod.in_features
    x2 = x.reshape(-1, k)
    if x2.stride(-1) != 1 or x2.data_ptr() % 16:
        x2 = x2.contiguous()
    rows = x2.shape[0]
    out = torch.empty((rows, n), device=x.device, dtype=torch.bfloat16)
    if rows == 0:
        return out.view(*x.shape[:-1], n)
    xq = torch.empty((rows, k), device=x.device, dtype=torch.uint8)
    xs = torch.empty((rows,), device=x.device, dtype=torch.float32)
    mod.quant(x2, xq, xs)
    sfa = mod.unit_sfa(rows)
    res = mod.no_residual if residual is None else residual.reshape(-1, n).contiguous()
    gemm = mod.gemm if residual is None else mod.gemm_residual()
    gemm(
        xq.unsqueeze(0),
        mod.weight_q.unsqueeze(0),
        sfa,
        mod.sfb,
        out.unsqueeze(0),
        xs,
        mod.weight_scale,
        mod.bias_or_zero,
        res,
        mod.one,
    )
    return out.view(*x.shape[:-1], n)


class CuteFp8Linear(nn.Module):
    """``nn.Linear`` with e4m3 weights (one fp32 scale per output channel) and e4m3
    activations (one fp32 scale per token, every call) on the block-scaled MMA."""

    def __init__(self, reference: nn.Linear, persistent: bool = True) -> None:
        super().__init__()
        self.in_features, self.out_features = reference.in_features, reference.out_features
        dev = reference.weight.device
        q, scale = quantize_fp8(reference.weight)  # once, here: never per call
        self.weight_q = q.view(torch.uint8)
        self.weight_fp8, self.weight_scale = q, scale.contiguous()
        self.bias = reference.bias
        self.quant_error = fp8_error(reference.weight, q, scale)  # for NOTES.md
        n, k = self.out_features, self.in_features
        self.bias_or_zero = (
            reference.bias.detach().to(torch.bfloat16)
            if reference.bias is not None
            else torch.zeros(n, device=dev, dtype=torch.bfloat16)
        )
        self.no_residual = torch.zeros((1, n), device=dev, dtype=torch.bfloat16)
        self.one = torch.ones(1, device=dev, dtype=torch.float32)
        self.sfb = unit_scales(n, k, dev)
        self._sfa = unit_scales(128, k, dev)
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        self.max_ctas = sms if persistent else 1 << 20
        self.has_bias = reference.bias is not None
        self.quant = compile_quant(k)
        self.gemm = compile_gemm(n, k, has_bias=self.has_bias, max_ctas=self.max_ctas)
        self._gemm_res = None
        self._key = len(_MODULES)  # the custom op's handle on this module
        _MODULES.append(self)

    def unit_sfa(self, rows: int) -> torch.Tensor:
        need = -(-rows // 128) * 128 * (self.in_features // SF_VEC)
        if self._sfa.numel() < need:  # grown, never per call
            self._sfa = unit_scales(rows, self.in_features, self._sfa.device)
        return self._sfa

    def gemm_residual(self):
        if self._gemm_res is None:
            self._gemm_res = compile_gemm(
                self.out_features,
                self.in_features,
                has_bias=self.has_bias,
                has_residual=True,
                max_ctas=self.max_ctas,
            )
        return self._gemm_res

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        if x.dtype != torch.bfloat16 or not x.is_cuda:
            y = fp8_w8a8_linear(x, self.weight_fp8, self.weight_scale, self.bias)
            return y if residual is None else y + residual
        # compiled: one opaque op; eager: the launcher itself (no custom-op dispatch cost)
        launch = cute_fp8_linear if torch.compiler.is_compiling() else _eager
        return launch(x, residual, self._key)


_MODULES: list[CuteFp8Linear] = []  # compiled kernels are not tensors: ops get an index


def _eager(x: torch.Tensor, residual: torch.Tensor | None, key: int) -> torch.Tensor:
    return _launch(_MODULES[key], x, residual)


# One opaque op under torch.compile / CUDA graphs (fresh output, no in-place writes).
cute_fp8_linear = torch.library.custom_op(f"{_NS}::cute_fp8_linear", _eager, mutates_args=())


@cute_fp8_linear.register_fake
def _(x, residual, key):
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
    return CuteFp8Linear(reference, persistent=persistent)


# ------------------------------------------------------------------ selftest (GPU)


def selftest(m: int = 352, n: int = 1024, k: int = 1024, seed: int = 0) -> dict:
    """Run on an sm_120 GPU: the module against ``fp8_w8a8_linear`` (same quantisation;
    the outputs differ only by the summation order and one bf16 rounding), with a bias,
    with a residual, and the e4m3 output against the bf16 one. Returns the relative
    errors; raises AssertionError past 1e-2."""
    torch.manual_seed(seed)
    lin = nn.Linear(k, n, bias=True, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    mod = build(lin)
    assert isinstance(mod, CuteFp8Linear), "build() fell back to the reference"
    ref = fp8_w8a8_linear(x, mod.weight_fp8, mod.weight_scale, lin.bias).float()

    def rel(a, b):
        return float((a.float() - b).norm() / b.norm().clamp_min(1e-30))

    out = {"bias": rel(mod(x), ref)}
    r = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    out["residual"] = rel(mod(x, r), ref + r.float())
    fp8 = compile_gemm(n, k, has_bias=True, out_fp8=True, max_ctas=mod.max_ctas)
    xq = torch.empty((m, k), device="cuda", dtype=torch.uint8)
    xs = torch.empty((m,), device="cuda", dtype=torch.float32)
    mod.quant(x, xq, xs)
    c8 = torch.empty((m, n), device="cuda", dtype=torch.uint8)
    out_scale = (ref.abs().amax() / FP8_MAX).reshape(1).float()
    fp8(
        xq.unsqueeze(0),
        mod.weight_q.unsqueeze(0),
        mod.unit_sfa(m),
        mod.sfb,
        c8.unsqueeze(0),
        xs,
        mod.weight_scale,
        mod.bias_or_zero,
        mod.no_residual,
        out_scale,
    )
    out["e4m3_out"] = rel(c8.view(torch.float8_e4m3fn).float() * out_scale, ref)
    torch.cuda.synchronize()
    # e4m3 output: 3 mantissa bits, ~3 % RMS rounding error per element
    bounds = {"bias": 1e-2, "residual": 1e-2, "e4m3_out": 8e-2}
    for name, value in out.items():
        assert value < bounds[name], f"{name}: relative L2 {value:.3g} >= {bounds[name]}"
    return out
