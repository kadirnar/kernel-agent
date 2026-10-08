# CuTe DSL on sm_120: block-scaled MMA, pipelines, CUTLASS ports

Part of the `cute-dsl` skill.

## sm_120 (GeForce Blackwell) specifics

* Tensor cores: `mma.sync` only. No `wgmma` (`cute.nvgpu.warpgroup` MMA needs sm_90a), no
  `tcgen05` / TMEM / 2-SM MMA (sm_100a). TMA (`cp.async.bulk.tensor`) works, but **no
  multicast, cluster 1×1×1**. ~99 KB shared memory per block
  (`get_smem_capacity_in_bytes("sm_120")` = 101376): Hopper/B200 tile configs do not fit.
  `setmaxnreg` works (producer warp 40 registers, MMA warps 232).
* FP8 MMA atoms (`cute.nvgpu.warp`), measured instruction rates on an RTX 5070 Ti
  (RESEARCH-TRITON §1.1):

  | atom | PTX / SASS | TFLOP/s |
  |---|---|---|
  | `MmaFP8Op(Float8E4M3FN, Float32, (16, 8, 32))` | `mma.sync...e4m3.e4m3.f32` / `QMMA.F32` | 208 (half rate) |
  | `MmaMXF8Op(Float8E4M3FN, Float32, Float8E8M0FNU)` | `mma.sync.kind::mxf8f6f4.block_scale.scale_vec::1X` / `QMMA.SF` | **416** |
  | `MmaMXF8F6F4Op(a, b, Float32, Float8E8M0FNU)` | mixed FP4 × FP8 only | — |
  | `MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))` | `HMMA.16816.F32.BF16` | 104 |

  **Never use `MmaFP8Op` (or any `QMMA.F32` path) for a compute-bound FP8 GEMM**: it is
  capped at 208 TFLOP/s, below cuBLASLt (325 TFLOP/s at 4096³). Its 4-bit siblings
  (`MmaMXF4Op`, `MmaMXF4NVF4Op`) only when the run allows 4-bit precisions.
* **Unit scales**: a W8A8 layer's per-token / per-channel scales are not 32-element blocks,
  so feed `MmaMXF8Op` scale bytes of 127 (ue8m0 2^0: the product equals the plain
  fp32-accumulated e4m3 GEMM bit for bit) and apply the real scales in the epilogue. Real
  MXFP8 scales (`fp8_mx` targets) use the swizzled 128×4 blocks of cuBLASLt
  `VEC32_UE8M0` / torch `SWIZZLE_32_4_4`, which is the MMA's `tile_atom_to_shape_SF`
  layout: `kernel_agent.kernels.quant.quantize_mxfp8` (the non-saturating
  `2^ceil(log2(amax / 448))` rule, never the OCP floor rule: docs/FP8.md §3.2) and
  `swizzle_mx_scales(scales).view(torch.uint8)` give the bytes.
* Block-scaled plumbing (`cutlass.utils.blockscaled_layout as bs`,
  `cutlass.utils.blackwell_helpers as sm120_utils`): `bs.tile_atom_to_shape_SF(a.shape,
  32)` (global SF layout), `bs.sm120_make_smem_layout_sfa/sfb(tiled_mma, tile, 32, stages)`,
  `sm120_utils.partition_fragment_SFA/SFB`, `get_layoutSFA_TV/SFB_TV` (smem → register
  copies), `get_permutation_mnk(tile, 32, True)`; the MMA call is
  `cute.gemm(tiled_mma, acc, [tCrA[..., k], tCrSFA[..., k]], [tCrB[..., k], tCrSFB[..., k]], acc)`.
  FP8 tile: (128, 128, 128) with a (4, 2, 1) warp layout; tile K a multiple of 128.

## Pipelining, warp specialisation, persistence (the GEMM example)

* `pipeline.PipelineTmaAsync.create(num_stages, producer_group, consumer_group, tx_count,
  barrier_storage, cta_layout_vmnk)`: the producer warp does `producer_acquire` → TMA
  copies with `tma_bar_ptr=producer_get_barrier(state)` → `producer_commit` →
  `state.advance()`, and `producer_tail` at the end; consumers `consumer_try_wait` (peek) →
  `consumer_wait` → `ldmatrix` + MMA → `consumer_release`. Stages = what is left of shared
  memory after the epilogue buffers (2 at 128×128×128 FP8 with 4 epilogue sub-tiles).
* Warp roles: `warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())`; MMA warps
  `setmaxregister_increase(232)`, the producer `setmaxregister_decrease(40)`; epilogue
  warps sync on a `pipeline.NamedBarrier` (not `barrier()`, the producer is elsewhere).
* Persistent: `utils.StaticPersistentTileScheduler` over
  `min(tiles, max_active_clusters)` CTAs (`max_active_clusters` = SM count); the producer
  runs ahead into the next tile while the MMA warps finish the epilogue. With
  `max_active_clusters` huge the same code launches one CTA per tile. Cooperative (both
  MMA warpgroups on one tile) is the upstream FP8 example; ping-pong (each warpgroup its
  own tile, epilogue overlapping the other's mainloop) exists for GeForce from CUTLASS 4.8
  (`dense_blockscaled_gemm_persistent_pingpong.py`): try it when the epilogue is heavy.
* Epilogue: fused math on the fp32 accumulators in registers (identity-tensor coordinates),
  then registers → shared memory (`stmatrix` for 16-bit outputs, universal copy for 8-bit)
  → `fence_proxy("async.shared", space="cta")` → named barrier → TMA store from one warp
  (`PipelineTmaStore`: `producer_commit`, `producer_acquire`, `producer_tail` before exit).
  **8-bit outputs need an epilogue tile of the full 128 rows** (the 8-bit copy covers 128
  rows at once; a 64-row sub-tile copies nothing and stores garbage: assert
  `epi_tile // mma_sub_tile >= 1` at trace time). Rows past M: TMA zero-fills loads and
  clips stores; clamp the row index of your own epilogue loads (`cutlass.min(m, M - 1)`).

## Porting the CUTLASS sm_120 C++ examples

| CUTLASS C++ | CuTe DSL |
|---|---|
| `SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e4m3_t, float_e4m3_t, float, float_ue8m0_t, 32>` (commented "MMA.SF", `cute/arch/mma_sm120.hpp`) | `cute.nvgpu.warp.MmaMXF8Op(Float8E4M3FN, Float32, Float8E8M0FNU)` |
| `SM120_16x8x32_TN<e4m3, e4m3, float>` (PyTorch's row-wise `_scaled_mm`) | `MmaFP8Op`: half rate, avoid |
| `OpClassBlockScaledTensorOp`, `mx_float8_t<e4m3>`, `KernelTmaWarpSpecializedCooperative` | `examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent_cooperative.py` (our GEMM example is a trimmed copy) |
| `Sm1xxBlockScaledConfig::tile_atom_to_shape_SFA` | `bs.tile_atom_to_shape_SF(shape, 32)` |
| `TiledMMA<..., Layout<Shape<_4,_2,_1>>, Tile<_128, Layout<Shape<_8,_2,_2>, Stride<_1,_16,_8>>, _32>>` | `cute.make_tiled_mma(op, cute.make_layout((4, 2, 1)), permutation_mnk=sm120_utils.get_permutation_mnk(...))` |
| examples 79 (`blackwell_geforce_gemm`, NVFP4 / MXFP8×MXFP6), 87 (blockwise 1×128 / 128×128 with software promotion: `QMMA.F32` speed) | `blackwell_geforce/kernel/{dense_gemm, blockscaled_gemm}` |

Our CUTLASS 4.1 C++ build of the SM120 block-scaled kernel compiled at 128×128×128 but
`initialize()` returned `Error Internal` (docs/FP8.md §2): the DSL route is the tested one.
Fetch the upstream files from GitHub `NVIDIA/cutlass` (`examples/python/CuTeDSL/`) when
you need more than the examples here.
