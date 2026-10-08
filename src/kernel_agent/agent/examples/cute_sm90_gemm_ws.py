"""Example candidate (CuTe DSL): bf16 or FP8 W8A8 GEMM for ``nn.Linear`` on Hopper's
warpgroup MMA (``wgmma``), fed by TMA, warp-specialised and persistent, with the scales,
bias, a residual and an optional e4m3 output fused into the epilogue; a swap-AB form for
decode rows (M ≤ 64).

Status: ported from NVIDIA's CuTe DSL Hopper examples (CUTLASS 4.8, BSD-3-Clause, notice at
the end of this file): ``examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/
dense_gemm_persistent.py`` for the kernel and ``dense_gemm_fp8_gelu_persistent.py`` for the
FP8 scale epilogue, written for ``nvidia-cutlass-dsl`` 4.8. Compiled for ``sm_90a`` on the
CPU (``tests/test_cute_examples.py`` checks the PTX for ``wgmma.mma_async`` and
``cp.async.bulk.tensor``); **not run on an H100 yet**: ``kernel-agent doctor --smoke`` and
``pytest -m gpu tests/test_cute_examples.py`` run its selftest there. Evaluate a copy with
``mode="quick"`` first and report what fails.

Why this instruction. Hopper's tensor-core peak (FP8 at twice bf16) is reached by ``wgmma``
only: ``mma.sync`` stops near 2/3 of it, FP8 ``mma.sync`` is emulated through fp16 on
sm_90 and there is no block-scaled MMA (the ``gpu-architectures`` skill, Hopper). ``wgmma``
reads both operands from shared memory (descriptors), so TMA writes the tiles there while a
warpgroup runs asynchronous MMAs on the previous stage. None of this exists on sm_120
(``mma.sync`` only) or sm_100 (``tcgen05.mma``): ``ARCHS`` keeps it on sm_90.

Kernel (``_WgmmaGemm``):

* warpgroup 0 is the producer: one warp issues the TMA loads (``cp.async.bulk.tensor``,
  multicast over a cluster with ``cluster_m=2``) into a ``PipelineTmaAsync`` ring of
  shared-memory stages, and the warpgroup gives its registers away (``setmaxnreg`` 40);
* the consumer warpgroups take 232 registers each and issue ``wgmma.mma_async`` per K
  block (bf16: k16 atoms, e4m3: k32; K tile 64 / 128), keep one MMA group in flight
  (``wait_group(1)``) and release each stage once its MMAs are done. 128x128 tiles: one
  consumer warpgroup (two m64 MMAs); 128x256 (``tile_n=256``): two "cooperative" ones;
* persistent (``persistent=True``): a static scheduler over ``min(tiles, SMs / cluster)``
  CTAs, so the producer runs ahead into the next tile while the consumers run the
  epilogue; ``persistent=False`` launches one CTA per tile (the same code);
* epilogue on the fp32 accumulators in registers (``partition_C`` of an identity tensor
  gives each register's ``(m, n)``): ``acc * sx[m] * sw[n]`` (FP8) ``+ bias[n]``
  ``+ residual[m, n]``, ``/ out_scale`` and saturation to ±448 for an e4m3 output, then
  ``stmatrix`` (16-bit) or plain stores (8-bit) to shared memory and a TMA store;
* swap-AB (rows ≤ ``SWAP_ROWS``): ``Cᵀ = W · Xᵀ``. The weight fills the MMA's M side
  (64-row tiles over N) and the tokens its N side (one 64-wide tile; TMA zero-fills the
  rows past M), and the output is stored through an M-major view of the row-major C. A
  decode GEMM so pads 1-64 tokens to a 64-wide tile instead of 128 rows, with N / 64 CTAs
  (no split-K yet: with few CTAs the weight stream is under-subscribed);
* rows ``M`` are dynamic (``cute.sym_int``: one compilation per weight shape); ``N`` must be
  a multiple of the tile N and ``K`` of 128, else ``build`` keeps the reference.

Precision: ``build(reference, fp8=True)`` is W8A8 (``precision: fp8_w8a8``): e4m3 weights
with one fp32 scale per output channel, quantised once in ``build``, and e4m3 activations
with one scale per token, quantised every call by ``_quant_rows_kernel`` (in a fused layer,
quantise in the producer instead). Long K: Hopper's FP8 ``wgmma`` accumulates with fewer
bits than fp32 (DeepSeek-V3: about 14); promote partial sums to fp32 every 128 K (CUTLASS's
``dense_gemm_fp8_2xacc.py``) when the tier is tight. ``fp8=False`` is a bf16 GEMM (another
summation order than cuBLAS: the near-lossless tier, or bf16 tolerances).

Baseline to beat (measure it here): cuBLASLt (``torch._scaled_mm`` row-wise / tensor-wise,
``examples/cuda_cublaslt_fp8.py``) and Triton ``tl.dot`` on e4m3 (it emits wgmma on sm_90).
Compiled once per (N, K, flags) with fake tensors and TVM-FFI through
``kernel_agent.cute_dsl.compile_cached`` (plain ``cute.compile`` outside kernel-agent).
"""

import math
import re

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
import torch.nn.functional as F
from cutlass.cute.nvgpu import cpasync, warpgroup
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
ARCHS = "sm_90"
ARCHS_WHY = (
    "wgmma (warpgroup MMA) exists on sm_90a only: sm_100 has tcgen05 instead, sm_120 neither "
    "(mma.sync only)"
)

_NS = re.sub(r"\W", "_", __name__)
TILE_M = 128
SWAP_TILE = (64, 64)  # swap-AB: 64 weight rows x 64 tokens per CTA (one consumer warpgroup)
SWAP_ROWS = 64  # rows up to which the module takes the swap-AB kernel
K_ALIGN = 128  # K tile of the e4m3 MMA (4 x k32); bf16 uses 64
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


class _WgmmaGemm:
    """``C[m, n] = epi(sum_k X[m, k] W[n, k])``, X [M, K] and W [N, K] K-major (bf16, or
    e4m3 passed as uint8), C [M, N] row-major bf16 or e4m3. ``epi``: ``* sx[m] * sw[n]``
    (``fp8``), ``+ bias[n]``, ``+ res[m, n]`` and, for an e4m3 C, ``/ out_scale[0]``.
    ``swap_ab`` computes ``Cᵀ = W Xᵀ`` (the MMA's M side runs over N)."""

    def __init__(
        self,
        *,
        fp8: bool,
        has_bias: bool,
        has_residual: bool,
        out_fp8: bool,
        swap_ab: bool,
        tile_n: int,
        cluster_m: int,
    ):
        self.acc_dtype = cutlass.Float32
        self.fp8 = fp8
        self.has_bias = has_bias
        self.has_residual = has_residual
        self.out_fp8 = out_fp8
        self.swap_ab = swap_ab
        self.fused = fp8 or has_bias or has_residual or out_fp8
        tile_mn = SWAP_TILE if swap_ab else (TILE_M, tile_n)
        self.tile_shape_mnk = (*tile_mn, 1)  # K is set in _setup_attributes
        self.cluster_shape_mn = (cluster_m, 1)
        # one consumer warpgroup spills on 128x256: two cooperative ones there
        self.atom_layout_mnk = (2, 1, 1) if tile_mn[0] > 64 and tile_mn[1] > 128 else (1, 1, 1)
        self.occupancy = 1
        self.num_dma_warp_groups = 1
        self.num_mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_warps_per_warp_group = 4
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = (
            self.num_dma_warp_groups + self.num_mma_warp_groups
        ) * self.num_threads_per_warp_group
        self.load_warp_id = 0
        self.epi_store_warp_id = self.num_dma_warp_groups * self.num_warps_per_warp_group
        self.load_register_requirement = 40
        self.mma_register_requirement = 232
        self.smem_capacity = get_smem_capacity_in_bytes("sm_90")
        self.buffer_align_bytes = 1024
        self.num_mma_threads = self.num_mma_warp_groups * self.num_threads_per_warp_group
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.num_mma_threads
        )

    def _setup_attributes(self):
        if self.tile_shape_mnk[0] not in (64, 128):
            raise ValueError("CTA tile M must be 64 or 128")
        if self.tile_shape_mnk[1] not in (64, 128, 256):
            raise ValueError("CTA tile N must be 64, 128 or 256")
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1]),
        )
        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        self.tile_shape_mnk = (
            self.tile_shape_mnk[0],
            self.tile_shape_mnk[1],
            mma_inst_shape_k * 4,  # bf16: 64, e4m3: 128
        )
        self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))
        self.num_mcast_ctas_a = self.cluster_shape_mn[1]
        self.num_mcast_ctas_b = self.cluster_shape_mn[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.epi_tile = sm90_utils.compute_tile_shape_or_override(
            self.tile_shape_mnk, self.c_dtype, is_cooperative=self.atom_layout_mnk == (2, 1, 1)
        )
        self.ab_stage, self.epi_stage = self._compute_stages()
        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            self.a_layout, self.tile_shape_mnk, self.a_dtype, self.ab_stage
        )
        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            self.b_layout, self.tile_shape_mnk, self.b_dtype, self.ab_stage
        )
        self.epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.epi_stage
        )

    def _operand(self, t: cute.Tensor) -> cute.Tensor:
        """(1, rows, K) -> (rows, K, 1), CuTe's (mode0, mode1, batch); uint8 -> e4m3."""
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
        c_ptr = c.iterator
        if cutlass.const_expr(self.out_fp8):
            c_ptr = cute.recast_ptr(c.iterator, dtype=cutlass.Float8E4M3FN)
        if cutlass.const_expr(self.swap_ab):
            # Cᵀ = W Xᵀ: the MMA's A is the weight, its B the tokens, and its C an (N, M)
            # view of the row-major C (mode 0 contiguous: an M-major output)
            a, b = self._operand(w), self._operand(x)
            out = cute.make_tensor(c_ptr, cute.select(c.layout, mode=[2, 1, 0]))
        else:
            a, b = self._operand(x), self._operand(w)
            out = cute.make_tensor(c_ptr, cute.select(c.layout, mode=[1, 2, 0]))
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = out.element_type
        self.a_layout = LayoutEnum.from_tensor(a)
        self.b_layout = LayoutEnum.from_tensor(b)
        self.c_layout = LayoutEnum.from_tensor(out)
        self._setup_attributes()

        tma_atom_a, tma_tensor_a = self._tma_load(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[1],
        )
        tma_atom_b, tma_tensor_b = self._tma_load(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[0],
        )
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            out,
            cute.slice_(self.epi_smem_layout_staged, (None, None, 0)),
            self.epi_tile,
        )
        tile_sched_params, grid = self._compute_grid(out, max_active_clusters)

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
            self.epi_smem_layout_staged,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
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
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)
        a_mcast_mask = cute.make_layout_image_mask(cta_layout_mnk, cluster_coord_mnk, mode=1)
        b_mcast_mask = cute.make_layout_image_mask(cta_layout_mnk, cluster_coord_mnk, mode=0)
        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(self.a_dtype, a_smem_layout) + cute.size_in_bytes(
            self.b_dtype, b_smem_layout
        )

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        # every consumer warp arrives once per CTA that multicasts into this one
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_arrive_cnt = mcast_size * self.num_mma_warp_groups * self.num_warps_per_warp_group
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mainloop_pipeline_array_ptr.data_ptr(),
            num_stages=self.ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, consumer_arrive_cnt),
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
            defer_sync=True,
        )
        # the cluster's barriers are initialised before any CTA multicasts into them
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )

        # (bM, bK, RestM, RestK, RestL) etc.
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.tile_shape_mnk, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.tile_shape_mnk, (0, None, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.tile_shape_mnk, (None, None, 0)), (None, None, None)
        )
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            cluster_coord_mnk[1],
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            cluster_coord_mnk[0],
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )

        # wgmma operands are partitioned per warpgroup (shared-memory descriptors)
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        mma_warp_group_thread_layout = cute.make_layout(
            self.num_mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma = tiled_mma.get_slice(
            mma_warp_group_thread_layout(warp_group_idx - self.num_dma_warp_groups)
        )
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCgC = thr_mma.partition_C(gC_mnl)
        accumulators = cute.make_rmem_tensor(tCgC.shape[:3], self.acc_dtype)
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups
        if is_dma_warp_group:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

        if warp_idx == self.load_warp_id:
            # ---------------------------------------------------------- TMA producer warp
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.ab_stage
            )
            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                tBgB_nkl = tBgB[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]
                producer_state.reset_count()
                for _k_tile in range(k_tile_cnt):
                    mainloop_pipeline.producer_acquire(producer_state)
                    bar = mainloop_pipeline.producer_get_barrier(producer_state)
                    cute.copy(
                        tma_atom_a,
                        tAgA_mkl[(None, producer_state.count)],
                        tAsA[(None, producer_state.index)],
                        tma_bar_ptr=bar,
                        mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_nkl[(None, producer_state.count)],
                        tBsB[(None, producer_state.index)],
                        tma_bar_ptr=bar,
                        mcast_mask=b_mcast_mask,
                    )
                    mainloop_pipeline.producer_commit(producer_state)  # a no-op for TMA
                    producer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            mainloop_pipeline.producer_tail(producer_state)

        if not is_dma_warp_group:
            # ---------------------------------------------------------- consumer warpgroups
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            num_k_blocks = cute.size(tCrA, mode=[2])
            mma_tidx = tidx - self.num_dma_warp_groups * self.num_threads_per_warp_group
            # (m, n) inside the tile of every accumulator register of this thread
            tCcC = tiled_mma.get_slice(mma_tidx).partition_C(
                cute.make_identity_tensor(cute.slice_(self.tile_shape_mnk, (None, None, 0)))
            )
            # the output's rows (the tokens): a clamp for the epilogue's loads past M
            rows = cute.size(mC_mnl, mode=[1]) if self.swap_ab else cute.size(mC_mnl, mode=[0])
            row_last = rows - 1

            # epilogue copies: registers -> shared memory -> TMA store
            copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                self.c_layout, elem_ty_d=self.c_dtype, elem_ty_acc=self.acc_dtype
            )
            copy_atom_C = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(self.c_layout.is_m_major_c(), 4),
                self.c_dtype,
            )
            tiled_copy_r2s = cute.make_tiled_copy_S(
                copy_atom_r2s, cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
            )
            thr_copy_r2s = tiled_copy_r2s.get_slice(mma_tidx)
            tRS_sD = thr_copy_r2s.partition_D(sC)  # (R2S, R2S_M, R2S_N, PIPE_D)
            tRS_rAcc = tiled_copy_r2s.retile(accumulators)
            rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
            tRS_rD = cute.make_rmem_tensor(rD_shape[:3], self.acc_dtype)
            tRS_rD_out = cute.make_rmem_tensor(rD_shape[:3], self.c_dtype)
            size_tRS_rD = cute.size(tRS_rD)
            k_pipe_mmas = 1
            prologue_mma_cnt = min(k_pipe_mmas, k_tile_cnt)
            tma_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.epi_stage,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, self.num_mma_threads
                ),
            )

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                gC_mnl_slice = gC_mnl[(None, None, *tile_coord_mnl)]

                # ------------------------------------------------ mainloop
                read_state.reset_count()
                release_state.reset_count()
                accumulators.fill(0.0)
                tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
                warpgroup.fence()
                for _k_tile in range(prologue_mma_cnt):
                    mainloop_pipeline.consumer_wait(read_state)
                    for k_block in cutlass.range_constexpr(num_k_blocks):
                        crd = (None, None, k_block, read_state.index)
                        cute.gemm(tiled_mma, accumulators, tCrA[crd], tCrB[crd], accumulators)
                    warpgroup.commit_group()
                    read_state.advance()
                for _k_tile in range(prologue_mma_cnt, k_tile_cnt):
                    mainloop_pipeline.consumer_wait(read_state)
                    for k_block in cutlass.range_constexpr(num_k_blocks):
                        crd = (None, None, k_block, read_state.index)
                        cute.gemm(tiled_mma, accumulators, tCrA[crd], tCrB[crd], accumulators)
                    warpgroup.commit_group()
                    # one MMA group stays in flight; the stage before it is free again
                    warpgroup.wait_group(k_pipe_mmas)
                    mainloop_pipeline.consumer_release(release_state)
                    release_state.advance()
                    read_state.advance()
                warpgroup.wait_group(0)
                for _k_tile in range(prologue_mma_cnt):
                    mainloop_pipeline.consumer_release(release_state)
                    release_state.advance()

                # ------------------------------------------------ fused epilogue math
                if cutlass.const_expr(self.fused):
                    tile_m0 = tile_coord_mnl[0] * self.tile_shape_mnk[0]
                    tile_n0 = tile_coord_mnl[1] * self.tile_shape_mnk[1]
                    inv_out = cutlass.Float32(1.0)
                    if cutlass.const_expr(self.out_fp8):
                        inv_out = cutlass.Float32(1.0) / mOutScale[0]
                    for i in cutlass.range_constexpr(cute.size(accumulators)):
                        coord = tCcC[i]
                        if cutlass.const_expr(self.swap_ab):
                            n = tile_m0 + coord[0]  # the MMA's M side runs over N
                            m = cutlass.min(tile_n0 + coord[1], row_last)
                        else:
                            m = cutlass.min(tile_m0 + coord[0], row_last)
                            n = tile_n0 + coord[1]
                        v = accumulators[i]
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
                        accumulators[i] = v

                # ------------------------------------------------ store: smem + TMA
                tCgC_epi = cute.zipped_divide(gC_mnl_slice, self.epi_tile)
                bSG_sD, bSG_gD = cpasync.tma_partition(
                    tma_atom_c, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2), tCgC_epi
                )
                epi_tile_num = cute.size(tCgC_epi, mode=[1])
                epi_tile_shape = tCgC_epi.shape[1]
                epi_tile_layout = cute.make_layout(epi_tile_shape, stride=(epi_tile_shape[1], 1))
                num_prev_epi_tiles = tile_sched.num_tiles_executed * epi_tile_num
                for epi_idx in cutlass.range_constexpr(epi_tile_num):
                    for epi_v in cutlass.range_constexpr(size_tRS_rD):
                        tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]
                    tRS_rD_out.store(tRS_rD.load().to(self.c_dtype))
                    epi_buffer = (num_prev_epi_tiles + epi_idx) % cute.size(tRS_sD, mode=[3])
                    cute.copy(tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, epi_buffer)])
                    # the TMA store (async proxy) must see the generic-proxy smem writes
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()
                    gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                    if warp_idx == self.epi_store_warp_id:
                        cute.copy(
                            tma_atom_c, bSG_sD[(None, epi_buffer)], bSG_gD[(None, gmem_coord)]
                        )
                        tma_store_pipeline.producer_commit()
                        tma_store_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            tma_store_pipeline.producer_tail()  # the last TMA stores read shared memory

    # -------------------------------------------------------------- host-side helpers

    def _compute_stages(self):
        tm, tn, tk = self.tile_shape_mnk
        ab_bytes = (tm * tk * self.a_dtype.width + tn * tk * self.b_dtype.width) // 8
        epi_stage = 4
        epi_bytes = cute.size(self.epi_tile) * self.c_dtype.width // 8 * epi_stage
        mbar_bytes = 1024
        ab_stage = (self.smem_capacity // self.occupancy - (mbar_bytes + epi_bytes)) // ab_bytes
        return ab_stage, epi_stage

    def _compute_grid(self, c: cute.Tensor, max_active_clusters: cutlass.Constexpr):
        gc = cute.zipped_divide(c, tiler=cute.slice_(self.tile_shape_mnk, (None, None, 0)))
        params = utils.PersistentTileSchedulerParams(
            gc[(0, (None, None, None))].shape, (*self.cluster_shape_mn, 1), 1, True
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(params, max_active_clusters)
        return params, grid

    @staticmethod
    def _tma_load(tensor, smem_layout_staged, tile, mcast_dim):
        op = (
            cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        return cpasync.make_tiled_tma_atom(
            op,
            tensor,
            cute.slice_(smem_layout_staged, (None, None, 0)),
            tile,
            num_multicast=mcast_dim,
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
    fp8: bool = True,
    has_bias: bool = False,
    has_residual: bool = False,
    out_fp8: bool = False,
    swap_ab: bool = False,
    tile_n: int = 128,
    cluster_m: int = 1,
    max_ctas: int = 1 << 20,
):
    """The GEMM for an [n, k] weight (``k % 128 == 0``; ``n`` a multiple of the tile N: 64
    with ``swap_ab``, else ``tile_n``), any number of rows. ``max_ctas``: clusters of the
    persistent schedule (SMs / ``cluster_m``), huge for one CTA per tile."""
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
    gemm = _WgmmaGemm(
        fp8=fp8,
        has_bias=has_bias,
        has_residual=has_residual,
        out_fp8=out_fp8,
        swap_ab=swap_ab,
        tile_n=tile_n,
        cluster_m=cluster_m,
    )
    key = ("sm90", n, k, fp8, has_bias, has_residual, out_fp8, swap_ab, tile_n, cluster_m)
    return _compile(
        gemm, x, w, c, sx, sw, bias, res, out_scale, max_ctas, stream, key=(*key, max_ctas)
    )


# ------------------------------------------------------------------ the module


def _launch(mod: "CuteSm90Linear", x: torch.Tensor, residual: torch.Tensor | None):
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
    gemm = mod.kernel(swap=mod.small_m and rows <= SWAP_ROWS, residual=residual is not None)
    gemm(
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


class CuteSm90Linear(nn.Module):
    """``nn.Linear`` on ``wgmma``: e4m3 weights (one fp32 scale per output channel) and e4m3
    activations (one fp32 scale per token, every call) with ``fp8``, else bf16."""

    def __init__(
        self,
        reference: nn.Linear,
        *,
        fp8: bool = True,
        persistent: bool = True,
        tile_n: int = 128,
        cluster_m: int = 1,
        small_m: bool = True,
    ) -> None:
        super().__init__()
        self.in_features, self.out_features = reference.in_features, reference.out_features
        n, k = self.out_features, self.in_features
        dev = reference.weight.device
        self.fp8, self.small_m = fp8, small_m
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
        self.config = {"fp8": fp8, "has_bias": reference.bias is not None}
        self.tile = {"tile_n": tile_n, "cluster_m": cluster_m}
        self.max_ctas = sms // cluster_m if persistent else 1 << 20
        self.swap_ctas = sms if persistent else 1 << 20
        self._kernels: dict[tuple[bool, bool], object] = {}
        self.kernel(swap=False, residual=False)  # compiled here, never inside a capture
        if small_m:
            self.kernel(swap=True, residual=False)
        self._key = len(_MODULES)  # the custom op's handle on this module
        _MODULES.append(self)

    def kernel(self, *, swap: bool, residual: bool):
        if (swap, residual) not in self._kernels:
            n, k = self.out_features, self.in_features
            tile = {"swap_ab": True} if swap else self.tile
            ctas = self.swap_ctas if swap else self.max_ctas
            self._kernels[swap, residual] = compile_gemm(
                n, k, has_residual=residual, max_ctas=ctas, **self.config, **tile
            )
        return self._kernels[swap, residual]

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        if x.dtype != torch.bfloat16 or not x.is_cuda:
            if self.fp8:
                y = fp8_w8a8_linear(x, self.weight_fp8, self.weight_scale, self.bias)
            else:
                y = F.linear(x, self.weight_op.to(x.dtype), self.bias)
            return y if residual is None else y + residual
        # compiled: one opaque op; eager: the launcher itself (no custom-op dispatch cost)
        launch = cute_sm90_linear if torch.compiler.is_compiling() else _eager
        return launch(x, residual, self._key)


_MODULES: list[CuteSm90Linear] = []  # compiled kernels are not tensors: ops get an index


def _eager(x: torch.Tensor, residual: torch.Tensor | None, key: int) -> torch.Tensor:
    return _launch(_MODULES[key], x, residual)


# One opaque op under torch.compile / CUDA graphs (fresh output, no in-place writes).
cute_sm90_linear = torch.library.custom_op(f"{_NS}::cute_sm90_linear", _eager, mutates_args=())


@cute_sm90_linear.register_fake
def _(x, residual, key):
    return x.new_empty((*x.shape[:-1], _MODULES[key].out_features), dtype=torch.bfloat16)


def build(
    reference: nn.Module,
    fp8: bool = True,
    persistent: bool = True,
    tile_n: int = 128,
    cluster_m: int = 1,
    small_m: bool = True,
) -> nn.Module:
    """``fp8``: W8A8 (True) or bf16; ``persistent``: a static persistent schedule over the
    SMs or one CTA per tile; ``tile_n``: 128 (one consumer warpgroup) or 256 (two
    cooperative ones); ``cluster_m``: 2 multicasts the weight tile over a CTA pair;
    ``small_m``: swap-AB for ≤ 64 rows. Tuning keywords for ``sweep_candidate``."""
    ok = (
        isinstance(reference, nn.Linear)
        and reference.weight.dtype == torch.bfloat16
        and reference.weight.is_cuda
        and torch.cuda.get_device_capability(reference.weight.device) == (9, 0)
        and reference.out_features % tile_n == 0
        and reference.in_features % K_ALIGN == 0
        and (reference.bias is None or reference.bias.dtype == torch.bfloat16)
    )
    if not ok:
        return reference
    return CuteSm90Linear(
        reference,
        fp8=fp8,
        persistent=persistent,
        tile_n=tile_n,
        cluster_m=cluster_m,
        small_m=small_m,
    )


# ------------------------------------------------------------------ selftest (GPU)


@torch.no_grad()
def selftest(m: int = 352, n: int = 1024, k: int = 1024, seed: int = 0) -> dict:
    """Run on an sm_90 GPU: the FP8 module against ``fp8_w8a8_linear`` (same quantisation;
    the outputs differ only by the summation order and one bf16 rounding) with a bias, with
    a residual, at 16 rows (swap-AB) and with an e4m3 output; the bf16 module against
    ``F.linear``. Returns the relative errors; raises AssertionError past the bounds."""
    torch.manual_seed(seed)
    lin = nn.Linear(k, n, bias=True, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    mod = build(lin)
    assert isinstance(mod, CuteSm90Linear), "build() fell back to the reference"
    ref = fp8_w8a8_linear(x, mod.weight_fp8, mod.weight_scale, lin.bias).float()

    def rel(a, b):
        a, b = a.detach().float(), b.detach().float()
        return float((a - b).norm() / b.norm().clamp_min(1e-30))

    out = {"bias": rel(mod(x), ref)}
    r = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    out["residual"] = rel(mod(x, r), ref + r.float())
    xs = x[:16]
    out["swap_ab"] = rel(
        mod(xs), fp8_w8a8_linear(xs, mod.weight_fp8, mod.weight_scale, lin.bias).float()
    )
    fp8 = compile_gemm(n, k, has_bias=True, out_fp8=True, max_ctas=mod.max_ctas)
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
    bounds = {"bias": 1e-2, "residual": 1e-2, "swap_ab": 1e-2, "e4m3_out": 8e-2, "bf16": 1e-2}
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
