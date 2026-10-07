# CuTe DSL backend (`nvidia-cutlass-dsl`, `import cutlass.cute as cute`)

The examples below target sm_120 (their `ARCHS`); on Hopper the peak MMA is the warpgroup
one (`cute.nvgpu.warpgroup`, wgmma) and on datacenter Blackwell `cute.nvgpu.tcgen05`
(CUTLASS's CuTe DSL `cute/hopper/` and `cute/blackwell/` examples: `sources.md`), with ~227 KB of
shared memory per block instead of 99.

When to use it (backend policy, docs/RESEARCH-TRITON.md §5.1): **compute-bound FP8 GEMMs
on sm_120** (W8A8, M ≳ 128 rows), because CuTe DSL exposes the block-scaled tensor-core
MMA that runs at full rate with fp32 accumulation, with an epilogue you write yourself;
launch-bound micro-ops (TVM-FFI calls cost ~25 us of host time); persistent / fused layer
blocks that need explicit pipelines and warp specialisation (Triton's automatic warp
specialisation does not pay on sm_120). Small-M GEMVs (memory bound) go to CUDA C++ first.

Examples (copy their structure; `kernel-agent doctor --smoke` runs them):
* `examples/cute_rmsnorm.py`: verified. `@cute.kernel` + `@cute.jit`, TVM-FFI, one CTA per row.
* `examples/cute_fp8_blockscaled_gemm.py`: W8A8 `nn.Linear` on `MmaMXF8Op`, persistent,
  warp-specialised (1 TMA producer warp + 8 MMA warps), fused epilogue (per-token ×
  per-channel scales, bias, residual, optional e4m3 output), per-token quantiser kernel.
  **Compiled for sm_120a on the CPU, not yet run on a GPU**: evaluate a copy with
  `mode="quick"` first and report what fails.
* `examples/cute_fp8_decoder_block.py`: fused small-M block (M ≤ 16) with FP8 weights:
  RMSNorm → gate|up → silu·up in one launch, down + residual in a second. Same status.

## Structure and compilation

```python
import cutlass, cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream


@cute.kernel  # device code
def k(gX: cute.Tensor, gY: cute.Tensor, eps: cutlass.Float32): ...


@cute.jit  # host launcher, traced at compile time
def launch(mX: cute.Tensor, mY: cute.Tensor, eps: cutlass.Float32, stream):
    k(mX, mY, eps).launch(grid=(mX.shape[0], 1, 1), block=(256, 1, 1), stream=stream)


m = cute.sym_int()  # dynamic rows: one compilation serves every M
x = make_fake_compact_tensor(cutlass.BFloat16, (m, 1024), stride_order=(1, 0), assumed_align=16)
y = make_fake_compact_tensor(cutlass.BFloat16, (m, 1024), stride_order=(1, 0), assumed_align=16)
fn = cute.compile(
    launch,
    x,
    y,
    cutlass.Float32(1e-6),
    make_fake_stream(use_tvm_ffi_env_stream=True),
    options="--enable-tvm-ffi",
)
fn(x_torch, y_torch, 1e-6)  # torch tensors directly, on torch.cuda.current_stream()
```

* **Fake tensors** (`make_fake_compact_tensor`, `cute.sym_int(divisibility=...)`) compile
  without device memory; `stride_order` gives each mode's rank, **0 = innermost** (a
  row-major `[M, K]` is `(1, 0)`, `(1, M, K)` is `(2, 1, 0)`). `Constexpr` arguments
  (`max_active_clusters: cutlass.Constexpr`) and a TVM-FFI env stream are not passed at
  call time.
* **Compile cache across evaluations**: every evaluation is a fresh process, so wrap
  `cute.compile` in `kernel_agent.cute_dsl.compile_cached(fn, *args, key=..., options=...)`
  (an exported object file per source digest + arch + DSL version + options + `key`; a
  reload takes ~0.04 s instead of 0.2-2 s). `key` must name every static specialisation:
  dtypes, static dims, tile sizes, flags. Both FP8 examples do this.
* FP8 / scale-factor dtypes through DLPack: pass `torch.uint8` views and recast inside the
  jit: `cute.make_tensor(cute.recast_ptr(t.iterator, dtype=cutlass.Float8E4M3FN), t.layout)`.
* GEMM convention: `(mode0, mode1, batch)`. Permute a `(1, M, K)` tensor inside the jit with
  `cute.make_tensor(t.iterator, cute.select(t.layout, mode=[1, 2, 0]))`.
* Under `torch.compile`, call through a `torch.library.custom_op` (the GEMM example); eager
  calls skip it (dispatch costs host time).

## Language facts

* Python `if`/`for` on runtime values become device control flow;
  `cutlass.range_constexpr(n)` unrolls at trace time; `cutlass.const_expr(flag)` for
  compile-time branches (flags kept as Python attributes of a kernel class specialise it).
* Types: `cutlass.Float32 / BFloat16 / Float16 / Float8E4M3FN / Float8E8M0FNU / Uint8 /
  Int32`; `.to(...)` converts scalars; vector conversion: `frag.store(other.load().to(T))`
  (`load()` gives a `TensorSSA`; f32 → e4m3 lowers to `cvt.rn.satfinite.e4m3x2.f32`,
  e4m3 → f32 via `cvt.rn.f16x2.e4m3x2`).
* Registers: `cute.make_rmem_tensor(shape, dtype)` (4.8 renamed `make_fragment`); shared
  memory: `cutlass.utils.SmemAllocator().allocate_tensor(dtype, layout, byte_alignment)` or a
  `@cute.struct` with `MemRange` / `Align` fields.
* Vector loads: `cute.autovec_copy(cute.local_tile(g, (1, 16), (row, chunk)), frag)` (128-bit
  when aligned); `cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype,
  num_bits_per_copy=128)` + `cute.make_tiled_copy_tv` for tiled copies.
* Sync / reductions: `cute.arch.barrier()`, `cute.arch.warp_reduction_sum / _max`,
  `cute.arch.shuffle_sync_bfly`; math: `cute.math.rsqrt / exp / exp2 / absf`;
  `cutlass.min / max` on dynamic scalars.
* Coordinates of register fragments: `thr_mma.partition_C(cute.make_identity_tensor((BM,
  BN)))` has the accumulator's shape; element `i` is the `(m, n)` of `acc[i]`.
* 4.8 API moves: `cutlass.tensor_utils.LayoutEnum`, `cutlass.memory.
  get_smem_capacity_in_bytes`; `cpasync.make_tiled_tma_atom` returns a `TmaInfo` that still
  unpacks into `(atom, tma_tensor)`. API docstrings are in the installed sources
  (`python -c "import cutlass, os; print(os.path.dirname(cutlass.__file__))"`).

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

## Checking without a GPU

Set `CUTE_DSL_ARCH=sm_120a` (and `CUDA_VISIBLE_DEVICES=` to be sure nothing runs) and call
the example's `compile_*` functions: fake tensors trace and compile on the CPU in 0.2-2 s.
`CUTE_DSL_KEEP=ptx CUTE_DSL_DUMP_DIR=<dir>` writes the PTX: check for
`mma.sync.aligned.kind::mxf8f6f4.block_scale` (not plain `.e4m3.e4m3.f32`),
`cp.async.bulk.tensor` and `setmaxnreg`; `CUTE_DSL_COMPILER_OPT="remarks{ptx}"` prints
registers and spills. `print(tensor.layout)` inside a jit prints at trace time. A kernel
whose output registers never get written still compiles: look for conversions or stores
of undefined registers (one `cvt` feeding every store) before you blame the GPU.

## Plan amnesia and false infeasibility

CuTe DSL converges slower than Triton, and the cause is process: every change takes
several compile rounds, then several correctness rounds, before it has a number. Three
failure modes follow:
* **Plan amnesia**: compile-error churn crowds out the goal. After every fix, re-read the
  idea in `NOTES.md` (and `plan.md` if there is one) and state the goal in one sentence
  before the next edit. If you cannot, you have drifted.
* **Abandoned restructurings**: big changes are expensive, so they get dropped halfway.
  Keep the last correct candidate, change one tile, layout or atom at a time, and revert to
  the last good file rather than pressing on through a fourth compile round.
* **False infeasibility**: an attempt you gave up on is not evidence. Write
  "abandoned after N compile rounds: <why>", never "X doesn't work", unless X measured
  correct and slower. A wrongly recorded infeasibility blocks the idea for every later
  session. Keep its `idea_id` so a later session can retry it.
