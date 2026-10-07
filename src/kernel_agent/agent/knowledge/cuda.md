# CUDA C++ backends

Two verified ways to run hand-written CUDA C++:

## A. `cuda` — `torch.utils.cpp_extension.load_inline` (nvcc)

Verified example: `examples/cuda_rmsnorm.py`.
* Full C++/CUDA: templates, CUTLASS/CuTe C++ headers (TileLang ships them at
  `tilelang/3rdparty/cutlass/include`; pass via `extra_include_paths`), inline
  PTX (`asm volatile`), `mma.sync`, `cp.async`, warp shuffles, cooperative groups.
* Lowest host overhead of all backends (~19 us measured for a tiny op), because
  the call goes straight through pybind into a C++ launcher.
* First build of a source takes 20-60 s; builds are cached by `name`. Give each
  distinct source a distinct `name` (hash the source) so edits are not ignored.
* Use `at::cuda::getCurrentCUDAStream()` for launches and
  `C10_CUDA_KERNEL_LAUNCH_CHECK()` after them.
* Do NOT use `AT_DISPATCH_FLOATING_TYPES*` with templated device helpers that
  only specialise half/bf16/float — it also instantiates `double`. Switch on
  `scalar_type()` explicitly.
* `__nv_bfloat16` / `__half` arithmetic: convert to float
  (`__bfloat162float`) for math; vectorise with `__nv_bfloat162` / `float4`
  loads (16-byte aligned).
* Environment: the toolchain sets `CUDA_HOME`, `NVCC_APPEND_FLAGS` and
  `TORCH_CUDA_ARCH_LIST` automatically (see `toolchain.json`).

## B. `nvrtc` — runtime compilation with `cuda.core`

Verified example: `examples/nvrtc_rmsnorm.py`.
* No nvcc / no build step: `Program(src, code_type="c++", options=ProgramOptions(arch=f"sm_{dev.arch}", std="c++17", include_path=cuda_include_dirs()))`
  `.compile("cubin").get_kernel("name")`; kernels must be `extern "C"` (or use
  `name_expressions` for templates).
* Launch on torch's stream:
  `launch(dev.create_stream(torch.cuda.current_stream()), LaunchConfig(grid=..., block=..., shmem_size=...), kernel, t.data_ptr(), np.int32(n), np.float32(x))`.
  Scalars must be numpy scalars with the exact C type.
* No torch headers: pass raw pointers + sizes; allocate outputs with torch.
* `from kernel_agent.toolchain import cuda_include_dirs` gives header paths
  (needed for `cuda_bf16.h`, `cuda_fp16.h`, `cuda/std/*`).

## Architecture cheat sheet

| Arch | GPUs | Tensor-core path | Async copy |
|---|---|---|---|
| sm_80/86/89 | A100, RTX 30xx/40xx | `mma.sync` (bf16/fp16/tf32/int8, fp8 on sm_89) | `cp.async` |
| sm_90a | H100/H200 | WGMMA (warpgroup), `mma.sync` | TMA + mbarrier, clusters |
| sm_100a | B200/GB200 | `tcgen05` MMA + TMEM, 2-CTA | TMA |
| sm_120 | RTX 50xx (consumer Blackwell) | `mma.sync` incl. FP8 and block-scaled FP4/FP6/FP8 (`mma.sync...kind::mxf8f6f4`); **no WGMMA, no tcgen05/TMEM** | TMA + `cp.async`, ~99 KB SMEM per block |

General rules: coalesced 128-bit global loads, avoid SMEM bank conflicts
(swizzle / padding), keep occupancy reasonable (registers ≤ 128/thread for
memory-bound kernels), use warp shuffles for reductions, `__launch_bounds__`.

Long debug loops (nvcc builds take 20-60 s, wrong results need several rounds):
* **Plan amnesia**: after every compile or correctness fix, re-read the idea in
  `NOTES.md` (and `plan.md`) and state its goal in one sentence before the next
  edit; fix rounds are not experiments. Keep the last correct candidate and
  revert to it instead of stacking a third untested change on a broken one.
* **False infeasibility**: if you stop, write "abandoned after N build rounds:
  <why>", never "X doesn't work", unless X measured correct and slower; keep its
  `idea_id` so a later session can retry it.

Under `torch.compile` (the compiled baseline, e.g. VoxCPM's `model.optimize()`
with `fullgraph=True`) a `load_inline` / NVRTC launcher is a graph break.
Register it with `torch.library.custom_op` + `register_fake`
(`examples/triton_rmsnorm_custom_op.py` shows the wrapper; only the launcher
body changes) and check it with `evaluate_candidate(..., compile_check=true)`.

## Streams, PDL and SM partitions (`kernel_agent.concurrency`)

Overlap is legal when it is joined and declared (the evaluator rejects the rest):
launch from the calling thread, join every stream before the call returns, name
side streams with `kernel_agent.concurrency` (`cc`).

* **Fork/join** (Python, around your launchers):
  `with cc.fork("aux"): y = branch(x)` (the stream waits for the caller's on entry,
  the caller's waits for it on exit), or `h = cc.launch("aux", fn, *args)` …
  `h.result()` (joins exactly that work; `cc.join_all()` joins the rest). Inside
  `torch.cuda.graph(...)` the same fork/join becomes graph edges (multi-stream
  capture): two independent branches in one graph ran 1.09-1.16x faster than
  serial (docs/PARALLEL.md §4.4). Your C++ launchers keep using
  `at::cuda::getCurrentCUDAStream()`: inside `cc.fork` that is the side stream.
* **PDL** (programmatic dependent launch, sm_90+): the next kernel on the stream
  launches while the current one drains. Header `ka_launch.cuh`:
  `load_inline(..., extra_include_paths=[str(cc.include_dir())])` and
  `#include "ka_launch.cuh"`; launch with
  `KaLaunch opt; opt.pdl = true; C10_CUDA_CHECK(ka_launch(kernel, grid, block, smem, stream, opt, args...));`
  (`cudaLaunchKernelEx` with programmatic stream serialization; dropped below
  sm_90). Device idiom: issue the loads that do not depend on the previous kernel
  (weights, constants), then `ka_pdl_launch_dependents(); ka_pdl_wait();`, then read
  what the previous kernel wrote; write nothing before the wait. Hash the header
  into the extension `name` with your source.
  - Pays for graph-captured chains of short memory-bound kernels whose prologue is
    a weight load: 28 dependent GEMVs of 2 MB, one CUDA graph 101.2 us -> graph +
    PDL 77.7 us = the DRAM floor (76.7 us); a graph kernel boundary costs ~0.9 us.
    Eager launches gain little (host-bound): capture the chain in a graph.
  - Hurt or noise where the wait came before the independent loads, and on
    GEMM -> GEMM edges in our runs; cuBLAS / cuBLASLt kernels between yours do not
    take part.
  - Example: `examples/cuda_pdl_gemv_chain.py` (the chain captured in `build()`,
    `pdl` a `build()` keyword for `sweep_candidate`).
* **Cooperative / persistent grids**: `opt.cooperative = true` (grid-wide sync)
  needs every block resident: size the grid by
  `ka_coresident_blocks(kernel, threads, smem, stream)`, which counts the SMs of the
  stream's green-context partition (`ka_sm_count(stream)`; in Python
  `cc.sm_count()`). `cudaDevAttrMultiProcessorCount` reports the whole device even
  inside a partition, and a cooperative grid sized from it is rejected there. A
  hand-rolled grid barrier cost ~2.2-2.4 us per layer on sm_120, more than a PDL
  graph boundary (~0.9 us): prefer one kernel per phase + PDL in a graph for
  memory-bound chains.
* **Partitions** (`cc.partition(sms=k)`: green contexts, two disjoint SM sets from
  one driver split, None where unsupported): they buy isolation, not throughput
  (two jobs on two plain streams finished first in every pair measured).
