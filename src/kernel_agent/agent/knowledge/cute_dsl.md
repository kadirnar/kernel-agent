# CuTe DSL backend (`nvidia-cutlass-dsl`, `import cutlass.cute as cute`)

Verified example: `examples/cute_rmsnorm.py` (TVM-FFI calling convention).

Structure
```python
import cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.runtime import make_fake_stream


@cute.kernel  # device code
def k(gA: cute.Tensor, gB: cute.Tensor, scale: cutlass.Float32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    ...


@cute.jit  # host launcher, traced at compile time
def launch(mA: cute.Tensor, mB: cute.Tensor, scale: cutlass.Float32, stream):
    k(mA, mB, scale).launch(grid=(G, 1, 1), block=(T, 1, 1), stream=stream)


# compile ONCE (cache it), then call with torch tensors directly:
mA = from_dlpack(a, assumed_align=16).mark_layout_dynamic(leading_dim=1)
fn = cute.compile(
    launch,
    mA,
    mB,
    cutlass.Float32(1.0),
    make_fake_stream(use_tvm_ffi_env_stream=True),
    options="--enable-tvm-ffi",
)
fn(a_torch, b_torch, 1.0)  # launches on torch.cuda.current_stream()
```

Key facts
* Tensors passed to `from_dlpack` must not require grad (`.detach()`).
* Python `if`/`for` on runtime values become device control flow;
  `cutlass.range_constexpr(n)` unrolls at compile time; `cutlass.const_expr(...)`
  for compile-time branches.
* Types: `cutlass.Float32`, `cutlass.BFloat16`, `cutlass.Float16`,
  `cutlass.Int32`; convert with `.to(cutlass.Float32)`; `tensor.element_type`.
* Shared memory: `smem = cutlass.utils.SmemAllocator()`;
  `smem.allocate_tensor(cutlass.Float32, cute.make_layout(n))`.
* Sync / reductions: `cute.arch.barrier()`, `cute.arch.warp_reduction_sum(x)`,
  `cute.arch.shuffle_sync_bfly`. Math: `cute.math.rsqrt`, `cute.math.exp2`,
  `cute.math.tanh` (fastmath variants exist).
* Layout algebra: `cute.make_layout(shape, stride=...)`, `cute.zipped_divide`,
  `cute.local_tile(tensor, tiler, coord)`, `cute.make_fragment_like`,
  `cute.autovec_copy` / `cute.copy(atom, src, dst)` with
  `cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=128)`
  for vectorised loads; `cute.make_tiled_copy_tv(atom, thr_layout, val_layout)`.
* Tensor cores: `cute.nvgpu.warp.MmaF16BF16Op` (mma.sync; works on sm_80+ incl.
  sm_120), `cute.make_tiled_mma`, `cute.gemm(tiled_mma, acc, a_frag, b_frag, acc)`.
  Hopper WGMMA (`cute.nvgpu.warpgroup`) needs sm_90a; tcgen05 (`cute.nvgpu.tcgen05`)
  needs sm_100a — neither runs on sm_120.
* Reference kernels (elementwise, softmax, layernorm/rmsnorm, dense GEMM,
  flash attention) live in GitHub `NVIDIA/cutlass` under `examples/python/CuTeDSL`
  — fetch them with WebFetch when you need a full idiom. API docstrings are in
  the installed sources: `python -c "import cutlass, os; print(os.path.dirname(cutlass.__file__))"`.
* Compile time is a few seconds; always cache the compiled function per
  (dtype, static shape) and use `mark_layout_dynamic` for varying dims.
* Errors are raised at trace time with MLIR diagnostics: read them carefully,
  usually a type mismatch (Float32 vs BFloat16) or a non-constexpr value used
  where a compile-time constant is required.

Plan amnesia and false infeasibility
CuTe DSL converges slower than Triton, and the cause is process: every change
takes several compile rounds, then several correctness rounds, before it has a
number. Three failure modes follow:
* **Plan amnesia**: compile-error churn crowds out the goal. After every fix,
  re-read the idea in `NOTES.md` (and `plan.md` if there is one) and state the
  goal in one sentence before the next edit. If you cannot, you have drifted.
* **Abandoned restructurings**: big changes are expensive, so they get dropped
  halfway. Keep the last correct candidate, change one tile, layout or atom at
  a time, and revert to the last good file rather than pressing on through a
  fourth compile round.
* **False infeasibility**: an attempt you gave up on is not evidence. Write
  "abandoned after N compile rounds: <why>", never "X doesn't work", unless X
  measured correct and slower. A wrongly recorded infeasibility blocks the idea
  for every later session. Keep its `idea_id` so a later session can retry it.
