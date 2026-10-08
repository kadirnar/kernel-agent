"""Example candidate (CuTe DSL): bf16 or FP8 W8A8 GEMM for ``nn.Linear`` on datacenter
Blackwell's tensor-memory MMA (``tcgen05.mma``, accumulators in TMEM), fed by TMA, warp-
specialised and persistent, optionally on 2-CTA pairs, with the scales, bias, a residual
and an optional e4m3 output fused into the epilogue.

Status: ported from NVIDIA's CuTe DSL Blackwell examples (CUTLASS 4.8, BSD-3-Clause, notice
at the end of this file): ``examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/
dense_gemm_alpha_beta_persistent.py`` (warp roles, pipelines, TMEM epilogue) and
``dense_gemm_persistent.py`` (persistent schedule, 2-CTA), written for
``nvidia-cutlass-dsl`` 4.8. Compiled for ``sm_100a`` on the CPU
(``tests/test_cute_examples.py`` checks the PTX for ``tcgen05.mma``, ``tcgen05.alloc`` and
``tcgen05.ld``); **not run on a B200 yet**: ``kernel-agent doctor --smoke`` and
``pytest -m gpu tests/test_cute_examples.py`` run its selftest there. Evaluate a copy with
``mode="quick"`` first and report what fails.

Why this instruction. Datacenter Blackwell reaches its tensor-core peak (FP8 at twice
bf16, block-scaled formats at the same rate) through ``tcgen05.mma`` only: ``mma.sync``
saturates near a quarter of the B200 peak, FP8 ``mma.sync`` is emulated through fp16, and
``wgmma`` does not exist on sm_100 (the ``gpu-architectures`` skill). One thread issues the
MMA for the whole CTA (or CTA pair) from shared-memory descriptors into tensor memory, so
registers stay free for the epilogue. Nothing here runs on sm_120 (no ``tcgen05`` / TMEM
there): ``ARCHS`` keeps it on sm_100 / sm_103.

Kernel (``_Tcgen05Gemm``, 6 warps):

* warp 5, TMA producer: ``cp.async.bulk.tensor`` loads of A and B into a
  ``PipelineTmaUmma`` ring of shared-memory stages (as many as fit in ~227 KB after the
  epilogue buffers); with ``two_cta`` each CTA of the pair loads half of the 256-row A tile
  and the B tile is multicast to both;
* warp 4, MMA: waits for a stage, issues ``tcgen05.mma`` (bf16: ``kind::f16``, e4m3:
  ``kind::f8f6f4``; ``cta_group::2`` on a pair, issued by the even CTA only) into one of two
  TMEM accumulator buffers and releases the stage with ``tcgen05.commit``; it moves on to
  the next tile's buffer while the epilogue drains the other one;
* warps 0-3, epilogue: allocate TMEM (``tcgen05.alloc``: 2 buffers x ``tile_n`` fp32
  columns, at most 512), wait for a full accumulator, copy it to registers sub-tile by
  sub-tile (``tcgen05.ld``), apply ``acc * sx[m] * sw[n]`` (FP8) ``+ bias[n]``
  ``+ residual[m, n]`` (``/ out_scale``, saturated to ±448, for an e4m3 output) with each
  register's ``(m, n)`` from an identity tensor partitioned like the output, convert, store
  to shared memory and TMA-store; ``tcgen05.dealloc`` at the end;
* persistent: a static scheduler over ``min(tiles, SMs / cluster)`` clusters
  (``persistent=False``: one cluster per tile, the same code);
* rows ``M`` are dynamic (``cute.sym_int``); ``N`` must be a multiple of ``tile_n`` and
  ``K`` of 128, else ``build`` keeps the reference.

Precision: ``build(reference, fp8=True)`` is W8A8 (``precision: fp8_w8a8``): e4m3 weights
with one fp32 scale per output channel (quantised once) and e4m3 activations with one
scale per token (``_quant_rows_kernel``, every call). The MMA accumulates in fp32 in TMEM.
``fp8=False`` is a bf16 GEMM (another summation order than cuBLAS: the near-lossless tier,
or bf16 tolerances). MXFP8 (``fp8_mx``, block-scaled ``tcgen05.mma``) is CUTLASS's
``blockscaled_gemm/dense_blockscaled_gemm_persistent.py``: not ported yet.

Baseline to beat (measure it here): cuBLASLt tensor-wise / MXFP8 (``torch._scaled_mm``,
``examples/cuda_cublaslt_fp8.py``) and Triton ``tl.dot`` on e4m3 (``tcgen05.mma
kind::f8f6f4`` on sm_100: ``examples/triton_fp8_w8a8_gemm.py``). Triton stays first for
GEMM-shaped glue here; this template is for epilogues and pipelines Triton cannot express.
Compiled once per (N, K, flags) with fake tensors and TVM-FFI through
``kernel_agent.cute_dsl.compile_cached`` (plain ``cute.compile`` outside kernel-agent).
"""

import re

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
import torch.nn.functional as F
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.memory import get_smem_capacity_in_bytes
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.tensor_utils import LayoutEnum
from torch import nn

from kernel_agent.kernels.quant import fp8_error, fp8_w8a8_linear, quantize_fp8

try:  # an on-disk cache of the compiled kernels across evaluations (kernel_agent/cute_dsl.py)
    from kernel_agent.cute_dsl import compile_cached
except ImportError:  # pragma: no cover - outside kernel-agent
    compile_cached = None

#: GPUs this example runs on (``kernel_agent.gpu_arch.supports``: ``doctor --smoke`` skips
#: it elsewhere and says why).
ARCHS = "sm_10x"
ARCHS_WHY = (
    "tcgen05.mma and tensor memory exist on sm_100a / sm_103a only: sm_90 has wgmma "
    "instead, sm_120 neither (mma.sync only)"
)

_NS = re.sub(r"\W", "_", __name__)
K_ALIGN = 128  # K tile of the e4m3 MMA (4 x k32); bf16 uses 64
TMEM_COLUMNS = 512  # tensor memory per SM: 128 lanes x 512 32-bit columns
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


class _Tcgen05Gemm:
    """``C[m, n] = epi(sum_k X[m, k] W[n, k])``, X [M, K] and W [N, K] K-major (bf16, or
    e4m3 passed as uint8), C [M, N] row-major bf16 or e4m3. ``epi``: ``* sx[m] * sw[n]``
    (``fp8``), ``+ bias[n]``, ``+ res[m, n]`` and, for an e4m3 C, ``/ out_scale[0]``."""

    def __init__(
        self,
        *,
        fp8: bool,
        has_bias: bool,
        has_residual: bool,
        out_fp8: bool,
        two_cta: bool,
        tile_n: int,
    ):
        self.acc_dtype = cutlass.Float32
        self.fp8 = fp8
        self.has_bias = has_bias
        self.has_residual = has_residual
        self.out_fp8 = out_fp8
        self.fused = fp8 or has_bias or has_residual or out_fp8
        self.use_2cta_instrs = two_cta
        # a CTA pair shares one 256-row MMA (128 accumulator rows each)
        self.mma_tiler = (256 if two_cta else 128, tile_n, 1)  # K is set in _setup_attributes
        self.cluster_shape_mn = (2, 1) if two_cta else (1, 1)
        self.cta_group = tcgen05.CtaGroup.TWO if two_cta else tcgen05.CtaGroup.ONE
        self.occupancy = 1
        self.epilog_warp_ids = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * (len(self.epilog_warp_ids) + 2)
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.smem_capacity = get_smem_capacity_in_bytes("sm_100")
        self.buffer_align_bytes = 1024

    def _make_tiled_mma(self):
        return sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

    def _setup_attributes(self):
        if self.mma_tiler[1] % 32 or not 32 <= self.mma_tiler[1] <= 256:
            raise ValueError("MMA tile N must be 32-256 in steps of 32")
        tiled_mma = self._make_tiled_mma()
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        self.mma_tiler = (self.mma_tiler[0], self.mma_tiler[1], mma_inst_shape_k * 4)
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk, self.use_2cta_instrs, self.c_layout, self.c_dtype
        )
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(tiled_mma)
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)
        if self.num_tmem_alloc_cols > TMEM_COLUMNS:
            raise ValueError(f"{self.num_tmem_alloc_cols} TMEM columns > {TMEM_COLUMNS}")

    def _operand(self, t: cute.Tensor) -> cute.Tensor:
        """(1, rows, cols) -> (rows, cols, 1), CuTe's (mode0, mode1, batch); uint8 -> e4m3."""
        ptr = t.iterator
        if cutlass.const_expr(self.fp8):
            ptr = cute.recast_ptr(t.iterator, dtype=cutlass.Float8E4M3FN)
        return cute.make_tensor(ptr, cute.select(t.layout, mode=[1, 2, 0]))

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,  # (1, M, K) bf16, or uint8 e4m3 codes (fp8)
        w: cute.Tensor,  # (1, N, K) like x: the nn.Linear weight
        c: cute.Tensor,  # (1, M, N) bf16, or uint8 for an e4m3 output
        sx: cute.Tensor,  # (M,) fp32 per-token scales (fp8)
        sw: cute.Tensor,  # (N,) fp32 per-channel scales (fp8)
        bias: cute.Tensor,  # (N,) bf16 (has_bias)
        residual: cute.Tensor,  # (R, N) bf16 (has_residual)
        out_scale: cute.Tensor,  # (1,) fp32 (out_fp8)
        max_active_clusters: cutlass.Constexpr,
        stream,
    ):
        a, b = self._operand(x), self._operand(w)
        c_ptr = c.iterator
        if cutlass.const_expr(self.out_fp8):
            c_ptr = cute.recast_ptr(c.iterator, dtype=cutlass.Float8E4M3FN)
        out = cute.make_tensor(c_ptr, cute.select(c.layout, mode=[1, 2, 0]))
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = out.element_type
        self.a_major_mode = LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = LayoutEnum.from_tensor(out)
        self._setup_attributes()
        tiled_mma = self._make_tiled_mma()
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a, a_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        # the even CTA's barrier counts the bytes both CTAs of a pair load
        self.num_tma_load_bytes = (
            cute.size_in_bytes(self.a_dtype, a_smem_layout)
            + cute.size_in_bytes(self.b_dtype, b_smem_layout)
        ) * atom_thr_size
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            out,
            cute.slice_(self.c_smem_layout_staged, (None, None, 0)),
            self.epi_tile,
        )
        tile_sched_params, grid = self._compute_grid(out, max_active_clusters)

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sC: cute.struct.Align[
                cute.struct.MemRange[self.c_dtype, cute.cosize(self.c_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            sx,
            sw,
            bias,
            residual,
            out_scale,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        mSX: cute.Tensor,
        mSW: cute.Tensor,
        mBias: cute.Tensor,
        mRes: cute.Tensor,
        mOutScale: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0  # the even CTA issues a pair's MMAs
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producer),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        # the epilogue warps of both CTAs of a pair drain the pair's accumulator
        num_acc_consumer_threads = len(self.epilog_warp_ids) * (2 if use_2cta_instrs else 1)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_acc_consumer_threads
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_ids)),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_ids[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        # the cluster's barriers are initialised before any CTA signals into them
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        # (bM, bK, RestM, RestK, RestL) etc.
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        # the (m, n, l) of every output element, partitioned like the output
        cC_mnl = cute.local_tile(
            cute.make_identity_tensor(mC_mnl.shape),
            cute.slice_(self.mma_tiler, (None, None, 0)),
            (None, None, None),
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)  # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)  # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCcC = thr_mma.partition_C(cC_mnl)

        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        tCrA = tiled_mma.make_fragment_A(sA)  # (MMA, MMA_M, MMA_K, STAGE): smem descriptors
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))
        epilog_sync_barrier = pipeline.NamedBarrier(
            self.epilog_sync_bar_id, 32 * len(self.epilog_warp_ids)
        )
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        if warp_idx == self.tma_warp_id:
            # ---------------------------------------------------------- TMA producer warp
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            while work_tile.is_valid_tile:
                cur = work_tile.tile_idx
                mma_tile_coord_mnl = (cur[0] // cute.size(tiled_mma.thr_id.shape), cur[1], cur[2])
                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
                ab_producer_state.reset_count()
                peek_ab_empty = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty = ab_pipeline.producer_try_acquire(ab_producer_state)
                for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty)
                    bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, ab_producer_state.count)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=bar,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, ab_producer_state.count)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=bar,
                        mcast_mask=b_full_mcast_mask,
                    )
                    ab_producer_state.advance()
                    peek_ab_empty = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty = ab_pipeline.producer_try_acquire(ab_producer_state)
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            ab_pipeline.producer_tail(ab_producer_state)

        if warp_idx == self.mma_warp_id:
            # ---------------------------------------------------------- MMA warp
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            num_k_blocks = cute.size(tCrA, mode=[2])
            while work_tile.is_valid_tile:
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]
                ab_consumer_state.reset_count()
                peek_ab_full = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full = ab_pipeline.consumer_try_wait(ab_consumer_state)
                if is_leader_cta:  # the epilogue has drained this TMEM buffer
                    acc_pipeline.producer_acquire(acc_producer_state)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)  # a fresh tile
                for _k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full)
                        for k_block in cutlass.range(num_k_blocks, unroll_full=True):
                            crd = (None, None, k_block, ab_consumer_state.index)
                            cute.gemm(tiled_mma, tCtAcc, tCrA[crd], tCrB[crd], tCtAcc)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        # tcgen05.commit: the stage is free once these MMAs have read it
                        ab_pipeline.consumer_release(ab_consumer_state)
                    ab_consumer_state.advance()
                    peek_ab_full = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                        peek_ab_full = ab_pipeline.consumer_try_wait(ab_consumer_state)
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            acc_pipeline.producer_tail(acc_producer_state)

        if warp_idx < self.mma_warp_id:
            # ---------------------------------------------------------- epilogue warps
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            # TMEM -> registers (tcgen05.ld), sub-tile by sub-tile
            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.cta_tile_shape_mnk,
                self.c_layout,
                self.c_dtype,
                self.acc_dtype,
                epi_tile,
                use_2cta_instrs,
            )
            tAcc_epi = cute.flat_divide(tCtAcc_base[((None, None), 0, 0, None)], epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc_base = thr_copy_t2r.partition_S(tAcc_epi)  # (T2R, M, N, EPI_M, EPI_N, STAGE)
            gC_epi_one = cute.flat_divide(tCgC[((None, None), 0, 0, 0, 0, 0)], epi_tile)
            tTR_rAcc = cute.make_rmem_tensor(
                thr_copy_t2r.partition_D(gC_epi_one)[(None, None, None, 0, 0)].shape,
                self.acc_dtype,
            )
            # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL): each register's (m, n, l)
            tTR_cC_all = thr_copy_t2r.partition_D(
                cute.flat_divide(tCcC[((None, None), 0, 0, None, None, None)], epi_tile)
            )
            # registers -> shared memory
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            copy_atom_r2s = sm100_utils.get_smem_store_op(
                self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
            )
            tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
            thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
            tRS_sC = thr_copy_r2s.partition_D(sC)
            tRS_rC = tiled_copy_r2s.retile(tTR_rC)
            # shared memory -> global (TMA store)
            bSG_sC, bSG_gC_all = cpasync.tma_partition(
                tma_atom_c,
                0,
                cute.make_layout(1),
                cute.group_modes(sC, 0, 2),
                cute.group_modes(
                    cute.flat_divide(tCgC[((None, None), 0, 0, None, None, None)], epi_tile), 0, 2
                ),
            )
            row_last = cute.size(mC_mnl, mode=[0]) - 1

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, 32 * len(self.epilog_warp_ids)
                ),
            )
            inv_out = cutlass.Float32(1.0)
            if cutlass.const_expr(self.out_fp8):
                inv_out = cutlass.Float32(1.0) / mOutScale[0]
            while work_tile.is_valid_tile:
                cur = work_tile.tile_idx
                mma_tile_coord_mnl = (cur[0] // cute.size(tiled_mma.thr_id.shape), cur[1], cur[2])
                bSG_gC = bSG_gC_all[(None, None, None, *mma_tile_coord_mnl)]
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
                tTR_cC = tTR_cC_all[(None, None, None, None, None, *mma_tile_coord_mnl)]
                tTR_cC = cute.group_modes(tTR_cC, 3, cute.rank(tTR_cC))
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                acc_pipeline.consumer_wait(acc_consumer_state)

                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
                for subtile_idx in cutlass.range(subtile_cnt):
                    cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, subtile_idx)], tTR_rAcc)
                    if cutlass.const_expr(self.fused):
                        tTR_cC_mn = tTR_cC[(None, None, None, subtile_idx)]
                        for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                            coord = tTR_cC_mn[i]
                            m = cutlass.min(coord[0], row_last)  # rows past M: never stored
                            n = coord[1]
                            v = tTR_rAcc[i]
                            if cutlass.const_expr(self.fp8):
                                v = v * mSX[m] * mSW[n]
                            if cutlass.const_expr(self.has_bias):
                                v = v + mBias[n].to(cutlass.Float32)
                            if cutlass.const_expr(self.has_residual):
                                v = v + mRes[m, n].to(cutlass.Float32)
                            if cutlass.const_expr(self.out_fp8):
                                v = cutlass.max(
                                    cutlass.min(v * inv_out, cutlass.Float32(FP8_MAX)),
                                    cutlass.Float32(-FP8_MAX),
                                )
                            tTR_rAcc[i] = v
                    tRS_rC.store(tiled_copy_r2s.retile(tTR_rAcc).load().to(self.c_dtype))
                    c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                    cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
                    # the TMA store (async proxy) must see the generic-proxy smem writes
                    cute.arch.fence_proxy("async.shared", space="cta")
                    epilog_sync_barrier.arrive_and_wait()
                    if warp_idx == self.epilog_warp_ids[0]:
                        cute.copy(tma_atom_c, bSG_sC[(None, c_buffer)], bSG_gC[(None, subtile_idx)])
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    epilog_sync_barrier.arrive_and_wait()
                # tcgen05.wait::ld: every load from this TMEM buffer has completed before the
                # MMA warp may write the next tile into it
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            tmem.relinquish_alloc_permit()
            epilog_sync_barrier.arrive_and_wait()  # every warp's tcgen05.ld has completed
            tmem.free(tmem_ptr)
            c_pipeline.producer_tail()  # the last TMA stores read shared memory

    # -------------------------------------------------------------- host-side helpers

    def _compute_stages(self, tiled_mma: cute.TiledMma):
        """(accumulator, A/B, C) stages: 2 TMEM accumulators, the A/B stages that fit next to
        2 C buffers, and the shared memory left over as more C buffers."""
        num_acc_stage, num_c_stage = 2, 2
        a_one = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, self.a_dtype, 1)
        b_one = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, self.b_dtype, 1)
        c_one = sm100_utils.make_smem_layout_epi(self.c_dtype, self.c_layout, self.epi_tile, 1)
        ab_bytes = cute.size_in_bytes(self.a_dtype, a_one) + cute.size_in_bytes(self.b_dtype, b_one)
        c_bytes_one = cute.size_in_bytes(self.c_dtype, c_one)
        mbar_bytes = 1024
        free = self.smem_capacity // self.occupancy - (mbar_bytes + c_bytes_one * num_c_stage)
        num_ab_stage = free // ab_bytes
        num_c_stage += (free - num_ab_stage * ab_bytes) // c_bytes_one
        return num_acc_stage, num_ab_stage, num_c_stage

    def _compute_grid(self, c: cute.Tensor, max_active_clusters: cutlass.Constexpr):
        gc = cute.zipped_divide(c, tiler=cute.slice_(self.cta_tile_shape_mnk, (None, None, 0)))
        params = utils.PersistentTileSchedulerParams(
            gc[(0, (None, None, None))].shape, (*self.cluster_shape_mn, 1)
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(params, max_active_clusters)
        return params, grid


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
    fp8: bool = True,
    has_bias: bool = False,
    has_residual: bool = False,
    out_fp8: bool = False,
    two_cta: bool = False,
    tile_n: int = 128,
    max_ctas: int = 1 << 20,
):
    """The GEMM for an [n, k] weight (``n % tile_n == k % 128 == 0``), any number of rows.
    ``max_ctas``: clusters of the persistent schedule (SMs, or SMs / 2 with ``two_cta``),
    huge for one cluster per tile."""
    m = cute.sym_int()
    order3 = (2, 1, 0)  # (1, rows, cols), cols innermost
    ab = cutlass.Uint8 if fp8 else cutlass.BFloat16
    x = make_fake_compact_tensor(ab, (1, m, k), stride_order=order3, assumed_align=16)
    w = make_fake_compact_tensor(ab, (1, n, k), stride_order=order3, assumed_align=16)
    c_type = cutlass.Uint8 if out_fp8 else cutlass.BFloat16
    c = make_fake_compact_tensor(c_type, (1, m, n), stride_order=order3, assumed_align=16)
    sx = make_fake_compact_tensor(cutlass.Float32, (cute.sym_int(),), assumed_align=4)
    sw = make_fake_compact_tensor(cutlass.Float32, (n,), assumed_align=4)
    bias = make_fake_compact_tensor(cutlass.BFloat16, (n,), assumed_align=2)
    res = make_fake_compact_tensor(
        cutlass.BFloat16, (cute.sym_int(), n), stride_order=(1, 0), assumed_align=16
    )
    out_scale = make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    gemm = _Tcgen05Gemm(
        fp8=fp8,
        has_bias=has_bias,
        has_residual=has_residual,
        out_fp8=out_fp8,
        two_cta=two_cta,
        tile_n=tile_n,
    )
    key = ("sm100", n, k, fp8, has_bias, has_residual, out_fp8, two_cta, tile_n, max_ctas)
    return _compile(gemm, x, w, c, sx, sw, bias, res, out_scale, max_ctas, stream, key=key)


# ------------------------------------------------------------------ the module


def _launch(mod: "CuteSm100Linear", x: torch.Tensor, residual: torch.Tensor | None):
    n, k = mod.out_features, mod.in_features
    x2 = x.reshape(-1, k)
    if not x2.is_contiguous() or x2.data_ptr() % 16:  # the kernels take compact rows
        x2 = x2.contiguous()
    rows = x2.shape[0]
    out = torch.empty((rows, n), device=x.device, dtype=torch.bfloat16)
    if rows == 0:
        return out.view(*x.shape[:-1], n)
    if mod.fp8:
        a = torch.empty((rows, k), device=x.device, dtype=torch.uint8)
        sx = torch.empty((rows,), device=x.device, dtype=torch.float32)
        mod.quant(x2, a, sx)
    else:
        a, sx = x2, mod.one
    res = mod.no_residual if residual is None else residual.reshape(-1, n).contiguous()
    mod.kernel(residual=residual is not None)(
        a.unsqueeze(0),
        mod.weight_op.unsqueeze(0),
        out.unsqueeze(0),
        sx,
        mod.weight_scale,
        mod.bias_or_zero,
        res,
        mod.one,
    )
    return out.view(*x.shape[:-1], n)


class CuteSm100Linear(nn.Module):
    """``nn.Linear`` on ``tcgen05.mma``: e4m3 weights (one fp32 scale per output channel) and
    e4m3 activations (one fp32 scale per token, every call) with ``fp8``, else bf16."""

    def __init__(
        self,
        reference: nn.Linear,
        *,
        fp8: bool = True,
        persistent: bool = True,
        two_cta: bool = False,
        tile_n: int = 128,
    ) -> None:
        super().__init__()
        self.in_features, self.out_features = reference.in_features, reference.out_features
        n, k = self.out_features, self.in_features
        dev = reference.weight.device
        self.fp8 = fp8
        self.bias = reference.bias
        if fp8:
            q, scale = quantize_fp8(reference.weight)  # once, here: never per call
            self.weight_fp8, self.weight_scale = q, scale.contiguous()
            self.weight_op = q.view(torch.uint8)
            self.quant_error = fp8_error(reference.weight, q, scale)  # for NOTES.md
            self.quant = compile_quant(k)
        else:
            self.weight_op = reference.weight.detach().contiguous()
            self.weight_scale = torch.ones(n, device=dev, dtype=torch.float32)
        self.bias_or_zero = (
            reference.bias.detach().to(torch.bfloat16)
            if reference.bias is not None
            else torch.zeros(n, device=dev, dtype=torch.bfloat16)
        )
        self.no_residual = torch.zeros((1, n), device=dev, dtype=torch.bfloat16)
        self.one = torch.ones(1, device=dev, dtype=torch.float32)
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        self.config = {
            "fp8": fp8,
            "has_bias": reference.bias is not None,
            "two_cta": two_cta,
            "tile_n": tile_n,
            "max_ctas": sms // (2 if two_cta else 1) if persistent else 1 << 20,
        }
        self._kernels: dict[bool, object] = {}
        self.kernel(residual=False)  # compiled here, never inside a capture
        self._key = len(_MODULES)  # the custom op's handle on this module
        _MODULES.append(self)

    def kernel(self, *, residual: bool):
        if residual not in self._kernels:
            self._kernels[residual] = compile_gemm(
                self.out_features, self.in_features, has_residual=residual, **self.config
            )
        return self._kernels[residual]

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        if x.dtype != torch.bfloat16 or not x.is_cuda:
            if self.fp8:
                y = fp8_w8a8_linear(x, self.weight_fp8, self.weight_scale, self.bias)
            else:
                y = F.linear(x, self.weight_op.to(x.dtype), self.bias)
            return y if residual is None else y + residual
        # compiled: one opaque op; eager: the launcher itself (no custom-op dispatch cost)
        launch = cute_sm100_linear if torch.compiler.is_compiling() else _eager
        return launch(x, residual, self._key)


_MODULES: list[CuteSm100Linear] = []  # compiled kernels are not tensors: ops get an index


def _eager(x: torch.Tensor, residual: torch.Tensor | None, key: int) -> torch.Tensor:
    return _launch(_MODULES[key], x, residual)


# One opaque op under torch.compile / CUDA graphs (fresh output, no in-place writes).
cute_sm100_linear = torch.library.custom_op(f"{_NS}::cute_sm100_linear", _eager, mutates_args=())


@cute_sm100_linear.register_fake
def _(x, residual, key):
    return x.new_empty((*x.shape[:-1], _MODULES[key].out_features), dtype=torch.bfloat16)


def build(
    reference: nn.Module,
    fp8: bool = True,
    persistent: bool = True,
    two_cta: bool = False,
    tile_n: int = 128,
) -> nn.Module:
    """``fp8``: W8A8 (True) or bf16; ``persistent``: a static persistent schedule over the
    SMs or one cluster per tile; ``two_cta``: 256-row MMAs on CTA pairs (B tile multicast,
    half the B traffic per CTA; CUTLASS measured about +8 % at 4096³ on a B200);
    ``tile_n``: 64-256. Tuning keywords for ``sweep_candidate``."""
    ok = (
        isinstance(reference, nn.Linear)
        and reference.weight.dtype == torch.bfloat16
        and reference.weight.is_cuda
        and torch.cuda.get_device_capability(reference.weight.device)[0] == 10
        and reference.out_features % tile_n == 0
        and reference.in_features % K_ALIGN == 0
        and (reference.bias is None or reference.bias.dtype == torch.bfloat16)
    )
    if not ok:
        return reference
    return CuteSm100Linear(
        reference, fp8=fp8, persistent=persistent, two_cta=two_cta, tile_n=tile_n
    )


# ------------------------------------------------------------------ selftest (GPU)


@torch.no_grad()
def selftest(m: int = 352, n: int = 1024, k: int = 1024, seed: int = 0) -> dict:
    """Run on an sm_100 / sm_103 GPU: the FP8 module against ``fp8_w8a8_linear`` (same
    quantisation; the outputs differ only by the summation order and one bf16 rounding)
    with a bias, with a residual, on 2-CTA pairs and with an e4m3 output; the bf16 module
    against ``F.linear``. Returns the relative errors; raises AssertionError past the
    bounds."""
    torch.manual_seed(seed)
    lin = nn.Linear(k, n, bias=True, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    mod = build(lin)
    assert isinstance(mod, CuteSm100Linear), "build() fell back to the reference"
    ref = fp8_w8a8_linear(x, mod.weight_fp8, mod.weight_scale, lin.bias).float()

    def rel(a, b):
        a, b = a.detach().float(), b.detach().float()
        return float((a - b).norm() / b.norm().clamp_min(1e-30))

    out = {"bias": rel(mod(x), ref)}
    r = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    out["residual"] = rel(mod(x, r), ref + r.float())
    out["two_cta"] = rel(build(lin, two_cta=True)(x), ref)
    fp8 = compile_gemm(n, k, has_bias=True, out_fp8=True, max_ctas=mod.config["max_ctas"])
    xq = torch.empty((m, k), device="cuda", dtype=torch.uint8)
    sx = torch.empty((m,), device="cuda", dtype=torch.float32)
    mod.quant(x, xq, sx)
    c8 = torch.empty((m, n), device="cuda", dtype=torch.uint8)
    out_scale = (ref.abs().amax() / FP8_MAX).reshape(1).float()
    fp8(
        xq.unsqueeze(0),
        mod.weight_op.unsqueeze(0),
        c8.unsqueeze(0),
        sx,
        mod.weight_scale,
        mod.bias_or_zero,
        mod.no_residual,
        out_scale,
    )
    out["e4m3_out"] = rel(c8.view(torch.float8_e4m3fn).float() * out_scale, ref)
    bf16 = build(lin, fp8=False)
    out["bf16"] = rel(bf16(x), F.linear(x, lin.weight, lin.bias).float())
    torch.cuda.synchronize()
    # e4m3 output: 3 mantissa bits, ~3 % RMS rounding error per element
    bounds = {"bias": 1e-2, "residual": 1e-2, "two_cta": 1e-2, "e4m3_out": 8e-2, "bf16": 1e-2}
    for name, value in out.items():
        assert value < bounds[name], f"{name}: relative L2 {value:.3g} >= {bounds[name]}"
    return out


# ------------------------------------------------------------------ license of the port
# The kernel above is a port of NVIDIA's CuTe DSL examples (CUTLASS 4.8):
#
# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
