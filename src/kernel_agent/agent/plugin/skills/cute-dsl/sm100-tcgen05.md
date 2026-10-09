# CuTe DSL on sm_100 (datacenter Blackwell): tcgen05, TMEM, 2-CTA pairs

Part of the `cute-dsl` skill. The template is `examples/cute_sm100_gemm_tcgen05.py`
(`ARCHS = "sm_10x"`): a W8A8 (e4m3) or bf16 `nn.Linear` with a fused epilogue, ported from
CUTLASS 4.8's CuTe DSL `cute/blackwell/kernel/dense_gemm/dense_gemm_alpha_beta_persistent.py`
and `dense_gemm_persistent.py` (BSD-3-Clause) for `nvidia-cutlass-dsl` 4.8. **Compiled for
sm_100a on the CPU (PTX checked), not run on a B200 yet**: evaluate a copy with
`mode="quick"` first and report what fails.

## The instruction and tensor memory

* `tcgen05.mma` is issued by **one thread** for the whole CTA (or CTA pair) from
  shared-memory descriptors; the accumulator lives in tensor memory (TMEM), not registers.
  M = 64 / 128 (one CTA) or 128 / 256 (`cta_group::2`), N up to 256 (CUTLASS's CuTe DSL
  GEMMs take 32-256 in steps of 32), K = 16 (bf16, `kind::f16`) or 32 (e4m3,
  `kind::f8f6f4`); MXFP8 (`kind::mxf8f6f4.block_scale`) runs at the FP8 rate, NVFP4 at
  twice it.
* **TMEM budget**: 128 lanes x 512 32-bit columns per SM. An fp32 accumulator of 128xN
  takes N columns; the template double-buffers it (2 x `tile_n` ≤ 512, so `tile_n` ≤ 256),
  and block-scaled MMAs also keep their scale factors there. Allocations are powers of two
  ≥ 32 columns.
* `tcgen05.alloc` / `dealloc` are warp-wide and must pair: `utils.TmemAllocator(holding_buf
  .ptr, barrier_for_retrieve=NamedBarrier, allocator_warp_id, is_two_cta,
  two_cta_tmem_dealloc_mbar_ptr)`, then `allocate(cols)` in the allocating warp,
  `wait_for_alloc()` + `retrieve_ptr(Float32)` in every warp that uses it, and at the end
  `relinquish_alloc_permit()` (other CTAs on the SM may allocate), a barrier (every
  `tcgen05.ld` done) and `free(ptr)`: all TMEM must be deallocated before the CTA exits.
* `tiled_mma.set(tcgen05.Field.ACCUMULATE, False)` at the start of each tile and `True`
  after its first K block (TMEM is not zeroed).

## Pipelines and mbarrier phases

* `PipelineTmaUmma` (TMA warp → MMA warp): the MMA warp's `consumer_release` is a
  `tcgen05.commit` to the stage's empty barrier, so the stage frees when the MMAs that read
  it complete, not when they are issued. `PipelineUmmaAsync` (MMA warp → epilogue warps):
  the accumulator buffer's full barrier is a `tcgen05.commit`; the epilogue releases it
  with `elect_one()`. `PipelineTmaStore` for the output.
* Phase bookkeeping as on Hopper (`sm90-wgmma.md`): `PipelineState(index, phase, count)`,
  `reset_count()` per tile, `producer_tail()` before exit; `producer_try_acquire` /
  `consumer_try_wait` peek so the wait can overlap other work. The template's warps: 0-3
  epilogue (and TMEM allocation), 4 MMA, 5 TMA.

## 2-CTA pairs and clusters

* `two_cta=True`: `tcgen05.CtaGroup.TWO`, a 256-row MMA over a (2, 1) cluster. The even
  CTA issues the MMAs for both (`is_leader_cta`), each CTA loads half of A, B is multicast
  (`cluster_shape_to_tma_atom_A / _B` pick the 2-SM TMA ops), the leader's barrier counts
  both CTAs' bytes (`x atom_thr_size`) and the accumulator consumers of both CTAs arrive.
  Half the B traffic per CTA: worth about +8 % at 4096³ on a B200 (gau-nernst's tcgen05
  GEMM, the authors' numbers); measure it here.
* Larger clusters multicast A and B further (`cluster_shape_mn`, ≤ 16 CTAs). Cluster launch
  control (a dynamic persistent scheduler that steals tiles from not-yet-launched clusters)
  is CUTLASS's `dense_gemm_persistent_dynamic.py`: the next schedule to try when tiles are
  uneven.

## Epilogue

`sm100_utils.get_tmem_load_op(...)` + `tcgen05.make_tmem_copy` give `tcgen05.ld.32x32b.xN`
(each thread owns one TMEM lane, i.e. one row, and N consecutive columns), sub-tile by
sub-tile (`compute_epilogue_tile_shape`). Partition an identity tensor of C's shape exactly
like C (`thr_mma.partition_C`, `flat_divide` by the epilogue tile, `thr_copy_t2r.
partition_D`) and element `i` is the `(m, n, l)` of register `i`: the template applies
`* sx[m] * sw[n] + bias[n] + residual[m, n]` there, then `get_smem_store_op` into shared
memory and a TMA store. Before an accumulator buffer goes back to the MMA warp,
`cute.arch.fence_view_async_tmem_load()` (`tcgen05.wait::ld`) makes sure every load from it
has completed.

## Triton first for GEMM-shaped glue

Triton's `tl.dot` lowers to `tcgen05.mma` on sm_100 (e4m3: `kind::f8f6f4`; FlashInfer-Bench
§4.4): keep it first for GEMM-shaped glue and check its SASS (`UTCQMMA` / `UTCHMMA`, not
`HMMA`). Compiled on the CPU with Triton 3.8 (the FP8 W8A8 example's GEMM): 128-row tiles
issue `UTCQMMA` for `tl.dot` and the block-scaled `UTCQMMA.SF` (the census's mark of the
scale operand) for `tl.dot_scaled`; 64-row tiles of `tl.dot_scaled` fall back to 16-bit
`UTCHMMA`, 32-row tiles to `mma.sync` (`HMMA`). This template is for what Triton cannot express: warp roles, TMEM buffering, 2-CTA
pairs, epilogues over TMEM. MXFP8 (`fp8_mx`) is CUTLASS's `blockscaled_gemm/
dense_blockscaled_gemm_persistent.py` (`make_blockscaled_trivial_tiled_mma`, scale factors
copied into TMEM with `tcgen05.cp`): the next template to port.

## What not to copy elsewhere

* sm_120 / sm_121 (GeForce Blackwell): no `tcgen05`, no TMEM, no 2-CTA MMA, no TMA
  multicast, 99 KB of shared memory: `sm120-blockscaled.md`. The names match
  (`blackwell_helpers`), the instructions do not.
* sm_90: no `tcgen05` (`wgmma` instead: `sm90-wgmma.md`).
* sm_103 (B300): the same kernels compile for `sm_103a`; INT8 MMA (`kind::i8`) does not
  exist there.

## Checking without a B200

`CUTE_DSL_ARCH=sm_100a CUDA_VISIBLE_DEVICES= CUTE_DSL_KEEP=ptx CUTE_DSL_DUMP_DIR=<dir>` and
the example's `compile_gemm(...)` (read `fn.__ptx__` right after each compile). Expect
`.target sm_100a`, `tcgen05.mma.cta_group::1.kind::f8f6f4` (bf16: `kind::f16`; pairs:
`cta_group::2` and `multicast::cluster`), `tcgen05.alloc`, `tcgen05.ld.sync.aligned.32x32b`,
`tcgen05.dealloc`, `cp.async.bulk.tensor`, no `wgmma` and no `mma.sync`. With
`CUTE_DSL_KEEP=sass` (nvdisasm ≥ 12.9 under `CUDA_HOME`): `UTCQMMA` / `UTCHMMA` (`.2CTA`),
`LDTM`, `UTMALDG`, `UTMASTG`. `tests/test_cute_examples.py` runs these checks.

## Measured

Nothing measured on a B200 yet (not run on that GPU yet). When it runs, record here,
labelled with the GPU: TFLOP/s against cuBLASLt (bf16, FP8, MXFP8) at 4096³ and at M = 1-64
decode shapes, 1-CTA vs 2-CTA, and the evaluator's verdict in the near-lossless tier.
