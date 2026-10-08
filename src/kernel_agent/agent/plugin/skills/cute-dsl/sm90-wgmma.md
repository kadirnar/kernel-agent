# CuTe DSL on sm_90 (Hopper): wgmma, TMA, warp specialisation

Part of the `cute-dsl` skill. The template is `examples/cute_sm90_gemm_ws.py` (`ARCHS =
"sm_90"`): a W8A8 (e4m3) or bf16 `nn.Linear` with a fused epilogue, ported from CUTLASS
4.8's CuTe DSL `cute/hopper/kernel/dense_gemm/dense_gemm_persistent.py` (BSD-3-Clause) for
`nvidia-cutlass-dsl` 4.8. **Compiled for sm_90a on the CPU (PTX checked), not run on an
H100 yet**: evaluate a copy with `mode="quick"` first and report what fails.

## The instruction

* `wgmma.mma_async` is a **warpgroup** MMA (4 warps, 128 threads): M = 64 per warpgroup, N
  8-256 in steps of 8, K = 16 (bf16 / fp16) or 32 (e4m3 / e5m2, int8). A and B come from
  shared memory through 64-bit descriptors (A may also come from registers); the fp32
  accumulators live in the warpgroup's registers (64xN / 128 threads per MMA).
* It is asynchronous: `warpgroup.fence()` before the first MMA that touches accumulators
  written by ordinary code, `cute.gemm(...)` per K block, `warpgroup.commit_group()`, and
  `warpgroup.wait_group(n)` before reading the accumulators or releasing a stage (n MMA
  groups may stay in flight; the template keeps 1).
* FP8 needs K-major A and B (an `nn.Linear` weight is K-major already). FP8 accumulation
  keeps fewer bits than fp32 (DeepSeek-V3: ~14): for long K promote to an fp32 register
  copy every 128 K (CUTLASS's `dense_gemm_fp8_2xacc.py`) when the tier is tight.
* CuTe DSL: `sm90_utils.make_trivial_tiled_mma(a_dtype, b_dtype, a_major, b_major, Float32,
  atom_layout_mnk, tiler_mn=(64, N))` (`import cutlass.utils.hopper_helpers as sm90_utils`),
  `tiled_mma.set(warpgroup.Field.ACCUMULATE, True)`; shared-memory layouts with the 128-byte
  swizzle the descriptors expect: `sm90_utils.make_smem_layout_a / _b / _epi`.

## TMA and mbarrier phase bookkeeping

* Loads: `cpasync.make_tiled_tma_atom(CopyBulkTensorTileG2SOp(), tensor, smem_layout,
  tile)` on the host (with `G2SMulticastOp()` and `num_multicast` over a cluster), then
  `cpasync.tma_partition` and `cute.copy(atom, gmem, smem, tma_bar_ptr=bar,
  mcast_mask=mask)` from **one** producer warp. Stores: `CopyBulkTensorTileS2GOp()` after
  `cute.arch.fence_proxy("async.shared", space="cta")` and a named barrier. Out-of-range
  rows are zero-filled on load and clipped on store: dynamic M needs no padding.
* `pipeline.PipelineTmaAsync` keeps a full and an empty mbarrier per stage. A
  `PipelineState` carries `(index, phase, count)`: `advance()` moves the index and flips the
  phase bit when it wraps. The producer acquires the empty barrier (its phase starts at 1,
  so the first pass over the ring does not block), the TMA's transaction bytes complete the
  full barrier (`tx_count` = bytes per stage), consumers wait on full with their phase and
  release by arriving on empty.
* What hangs: a consumer arrive count that differs from the arrivals (the template:
  consumer warps x CTAs that multicast into this one), a stage's `tx_count` that differs
  from the bytes the TMA writes, resetting a phase (call `reset_count()` per tile, never
  rebuild the state), and a CTA exiting while a peer still multicasts into it
  (`producer_tail()` at the end). `pipeline_init_arrive` / `pipeline_init_wait` around the
  barrier init make a cluster's barriers exist before any CTA signals them.

## Warp specialisation and persistence

* Warpgroup 0 produces (`setmaxregister_decrease(40)`: `setmaxnreg` works per warpgroup),
  the consumer warpgroups take the registers (`setmaxregister_increase(232)`). 128x128
  tiles: one consumer warpgroup (two m64 MMAs per K block); 128x256: two cooperative ones
  (one would spill). The epilogue syncs the consumers only: `pipeline.NamedBarrier`
  (id ≥ 1; id 0 is `__syncthreads`).
* `utils.StaticPersistentTileScheduler` over `PersistentTileSchedulerParams(tiles_mnl,
  cluster_mnl, swizzle, raster_along_m)`, grid = `min(tiles, max_active_clusters)`: every
  role builds its own scheduler and walks the same tile sequence, so the producer runs
  into the next tile while the consumers store. Ping-pong (each consumer warpgroup its own
  tile, epilogues overlapping the other's mainloop) is the next schedule to try when the
  epilogue is heavy.
* wgmma operands are partitioned per **warpgroup** (`tiled_mma.get_slice(wg * 128)`); the
  `(m, n)` of a thread's accumulator registers needs the **thread's** slice:
  `tiled_mma.get_slice(tidx - 128).partition_C(cute.make_identity_tensor((BM, BN)))`
  (element `i` is the coordinate of `acc[i]`). The template's epilogue applies
  `* sx[m] * sw[n] + bias[n] + residual[m, n]` there, before `stmatrix` (16-bit outputs) or
  a plain copy (e4m3) into shared memory.

## Small M (decode): swap-AB

At M ≤ 64 a 128-row MMA tile is mostly padding. Swap the operands: `Cᵀ = W · Xᵀ`, the
weight fills the MMA's M side (64-row tiles over N), the tokens its N side (one 64-wide
tile), and the output goes through an M-major view of C (`cute.select(c.layout, mode=[2,
1, 0])`, transposed `stmatrix`). The template does this below `SWAP_ROWS`; with N / 64
CTAs the weight stream can be under-subscribed: split-K / stream-K is the next step to
measure.

## What not to copy elsewhere

* sm_100 / sm_103: no `wgmma` (`tcgen05.mma` with tensor memory instead: `sm100-tcgen05.md`).
* sm_120 / sm_121: no `wgmma`, no TMA multicast (clusters 1x1x1), 99 KB of shared memory
  instead of 227 (the stage counts and 128x256 tiles here do not fit): `sm120-blockscaled.md`.
* An `ARCHS` declaration keeps an example on its family; a copy for another GPU starts from
  that GPU's template, not from this one.

## Checking without an H100

`CUTE_DSL_ARCH=sm_90a CUDA_VISIBLE_DEVICES= CUTE_DSL_KEEP=ptx CUTE_DSL_DUMP_DIR=<dir>` and
the example's `compile_gemm(...)` (the dump name is truncated: read `fn.__ptx__` right
after each compile). Expect `.target sm_90a`, `wgmma.mma_async.sync.aligned.m64n128k32.
f32.e4m3.e4m3` (bf16: `...k16.f32.bf16.bf16`), `cp.async.bulk.tensor`, `setmaxnreg`, no
`mma.sync`. `CUTE_DSL_KEEP=sass` (nvdisasm ≥ 12.9 under `CUDA_HOME`) shows `QGMMA` (e4m3) or
`HGMMA` (bf16), `UTMALDG` and `UTMASTG`. `tests/test_cute_examples.py` runs these checks.

## Measured

Nothing measured on an H100 yet (not run on that GPU yet). When it runs, record here,
labelled with the GPU: TFLOP/s against cuBLASLt (bf16, FP8) at 4096³ and at M = 1-64
decode shapes, and the evaluator's verdict in the near-lossless tier.
