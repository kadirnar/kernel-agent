---
name: cute-dsl
description: CuTe DSL (cutlass.cute) kernels — when to choose it, kernel / jit structure, fake-tensor compilation and the compile cache, language facts, CPU-only PTX checks, sm_120 block-scaled MMA, sm_90 wgmma and sm_100 tcgen05 / TMEM templates, TMA pipelines, warp specialisation, CUTLASS ports. Use when a target's backend is cute.
---

# CuTe DSL backend (`nvidia-cutlass-dsl`, `import cutlass.cute as cute`)

Each example declares the GPUs it runs on (`ARCHS`): the peak MMA differs per family and
none of them runs on another. sm_120: block-scaled `mma.sync` (the examples below the
first two). Hopper: the warpgroup MMA (`cute.nvgpu.warpgroup`, wgmma). Datacenter
Blackwell: `cute.nvgpu.tcgen05` with tensor memory. Hopper and B200 have ~227 KB of shared
memory per block instead of 99 (CUTLASS's CuTe DSL `cute/hopper/` and `cute/blackwell/`
examples: the `documentation-sources` skill).

When to use it (backend policy, docs/RESEARCH-TRITON.md §5.1): **compute-bound FP8 GEMMs
on sm_120** (W8A8, M ≳ 128 rows), because CuTe DSL exposes the block-scaled tensor-core
MMA that runs at full rate with fp32 accumulation, with an epilogue you write yourself;
launch-bound micro-ops (TVM-FFI calls cost ~25 us of host time); persistent / fused layer
blocks that need explicit pipelines and warp specialisation (Triton's automatic warp
specialisation does not pay on sm_120). Small-M GEMVs (memory bound) go to CUDA C++ first.

Examples (copy their structure; `kernel-agent doctor --smoke` runs them):
* `examples/cute_rmsnorm.py`: verified. `@cute.kernel` + `@cute.jit`, TVM-FFI, one CTA per row.
* `examples/cute_sm90_gemm_ws.py` (sm_90): bf16 / W8A8 `nn.Linear` on `wgmma`, TMA, a
  producer and consumer warpgroups, persistent, fused epilogue, swap-AB for M ≤ 64:
  [sm90-wgmma.md](sm90-wgmma.md). `examples/cute_sm100_gemm_tcgen05.py` (sm_100 / sm_103):
  the same on `tcgen05.mma` with TMEM accumulators and optional 2-CTA pairs:
  [sm100-tcgen05.md](sm100-tcgen05.md). **Compiled for sm_90a / sm_100a on the CPU, not run
  on an H100 / B200 yet**: evaluate a copy with `mode="quick"` first.
* `examples/cute_fp8_blockscaled_gemm.py`: W8A8 `nn.Linear` on `MmaMXF8Op`, persistent,
  warp-specialised (1 TMA producer warp + 8 MMA warps), fused epilogue (per-token ×
  per-channel scales, bias, residual, optional e4m3 output), per-token quantiser kernel.
  **Compiled for sm_120a on the CPU, not yet run on a GPU**: evaluate a copy with
  `mode="quick"` first and report what fails.
* `examples/cute_nvfp4_w4a4_gemm.py`: W4A4 `nn.Linear` (`fp4_w4a4`, opt-in) on
  `MmaMXF4NVF4Op` (e2m1 x e2m1, ue4m3 scales per 16): the FP8 GEMM's structure with the FP4
  operand / scale types, the per-token scale and bias in the epilogue, a per-token NVFP4
  quantiser bit for bit `quant.quantize_fp4_activations`. Run on an RTX 5070 Ti: 650 TFLOP/s
  at 4096³ (cuBLASLt NVFP4 661, FP8 332; skill `fp4-w4a4`).
* `examples/cute_fp8_decoder_block.py`: fused small-M block (M ≤ 16) with FP8 weights:
  RMSNorm → gate|up → silu·up in one launch, down + residual in a second. Same status.

GeForce Blackwell (sm_120) specifics, the block-scaled FP8 MMA, pipelining / warp specialisation / persistence and porting CUTLASS C++ examples: [sm120-blockscaled.md](sm120-blockscaled.md).

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

## Checking without a GPU

Set `CUTE_DSL_ARCH=sm_120a` (and `CUDA_VISIBLE_DEVICES=` to be sure nothing runs) and call
the example's `compile_*` functions: fake tensors trace and compile on the CPU in 0.2-2 s.
`CUTE_DSL_KEEP=ptx CUTE_DSL_DUMP_DIR=<dir>` writes the PTX: check for
`mma.sync.aligned.kind::mxf8f6f4.block_scale` (not plain `.e4m3.e4m3.f32`),
`cp.async.bulk.tensor` and `setmaxnreg`; `CUTE_DSL_COMPILER_OPT="remarks{ptx}"` prints
registers and spills. `print(tensor.layout)` inside a jit prints at trace time. A kernel
whose output registers never get written still compiles: look for conversions or stores
of undefined registers (one `cvt` feeding every store) before you blame the GPU. Any
architecture compiles on any machine: `sm_90a` (expect `wgmma.mma_async`) and `sm_100a`
(`tcgen05.mma`, `tcgen05.alloc`, `tcgen05.ld`) as in the two files above; dump names are
truncated, so read `fn.__ptx__` right after each compile; `CUTE_DSL_KEEP=sass` (nvdisasm ≥
12.9 under `CUDA_HOME`) adds the SASS.

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

## Examples and sources

* Examples: `examples/cute_rmsnorm.py`, `examples/cute_fp8_blockscaled_gemm.py`, `examples/cute_nvfp4_w4a4_gemm.py`, `examples/cute_fp8_decoder_block.py`, `examples/cute_sm90_gemm_ws.py`, `examples/cute_sm100_gemm_tcgen05.py`. All in kernel-agent's examples directory (`kernel_agent/agent/examples/`; a session's prompt gives the directory): copy their structure; an example's `ARCHS` names the GPUs it runs on.
* Sources: the `documentation-sources` skill's `sources.md`, sections "CUTLASS / CuTe"; "Local reference code".
* Code: `kernel_agent.cute_dsl.compile_cached`.
