---
name: cuda-kernels
description: CUDA C++ candidates through load_inline (nvcc) or NVRTC — builds, launches, dtype pitfalls, vectorisation, the tensor-core path per architecture, FP8 GEMMs via cuBLASLt from C++, long debug loops, torch.compile registration. Use when a target's backend is cuda or nvrtc.
---

# CUDA C++ backends

Two verified ways to run hand-written CUDA C++:

Streams, PDL, cooperative grids and SM partitions: the `cuda-graphs-streams-pdl` skill. Per-architecture facts: `gpu-architectures`. FP8 / FP4 formats and contracts: the precision skills (`precision-tiers` first).

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
* No nvcc / no build step: `kernel = nvrtc_kernel(src, "name", capability=torch.cuda.get_device_capability())`
  (`from kernel_agent.toolchain import nvrtc_kernel`; `nvrtc_kernels(src, ["a", "b"])` for
  several: a dict). It compiles for this GPU (`sm_XY` cubin; under an emulated older GPU,
  `KERNEL_AGENT_EMULATE_ARCH`, `compute_XY` PTX, which the driver JIT-compiles: a
  hard-coded `sm_XY` cubin does not load there), C++17 with CUDA's headers; other
  `ProgramOptions` go as keywords (`arch="sm_120a"` for an arch-specific target,
  `max_register_count=...`, extra `include_path`). It keeps the cubin, so the SASS census of
  `profile=true` sees the kernel (cuda.core frees the `ObjectCode` after `get_kernel`), and
  under `profile="ncu"` adds line info, so ncu's stall lines are source lines. Kernels must
  be `extern "C"`, or a template instantiation by name (`"k<128>"`: passed to NVRTC as a
  name expression).
* Calling `Program(src, code_type="c++", options=ProgramOptions(arch=arch, ...)).compile(kind).get_kernel("name")`
  directly (`arch, kind = nvrtc_target(...)`, `include_path=cuda_include_dirs()`) also
  works: keep the `ObjectCode` in a module global for the census, `lineinfo=True` for
  source lines.
* Launch on torch's stream:
  `launch(dev.create_stream(torch.cuda.current_stream()), LaunchConfig(grid=..., block=..., shmem_size=...), kernel, t.data_ptr(), np.int32(n), np.float32(x))`.
  Scalars must be numpy scalars with the exact C type.
* No torch headers: pass raw pointers + sizes; allocate outputs with torch.
* `from kernel_agent.toolchain import cuda_include_dirs` gives header paths
  (needed for `cuda_bf16.h`, `cuda_fp16.h`, `cuda/std/*`).

## Architecture cheat sheet

| Arch | GPUs | Tensor-core path | Async copy |
|---|---|---|---|
| sm_75 | T4, RTX 20xx | `mma.sync` m16n8k8 fp16 (fp32 or fp16 accumulation), m8n8k16 s8 / m8n8k32 s4 (IMMA), `ldmatrix`; WMMA 16x16x16 `half` / `signed char`; **no bf16, tf32 or FP8** (bf16 `__hfma` does not compile: math in fp32) | none (`cp.async` is sm_80+): plain loads into shared memory; 64 KB SMEM per block |
| sm_80 | A100, A30 | `mma.sync` m16n8k16 bf16/fp16, m16n8k8 tf32, m16n8k32 s8 | `cp.async` + `mbarrier`, 163 KB SMEM per block |
| sm_86 / sm_89 | A10, A40, RTX 30xx / L4, L40S, RTX 40xx | as sm_80; on sm_89 also FP8 `mma.sync` m16n8k32 e4m3 / e5m2 and the hardware e4m3 `cvt` | `cp.async`, 99 KB SMEM per block |
| sm_90a | H100/H200 | WGMMA (warpgroup) for the peak; `mma.sync` ~2/3 of it | TMA + mbarrier, clusters |
| sm_100a | B200/GB200 | `tcgen05` MMA + TMEM, 2-CTA, block-scaled; no WGMMA | TMA |
| sm_120 | RTX 50xx (consumer Blackwell) | `mma.sync` incl. FP8 and block-scaled FP4/FP6/FP8 (`mma.sync...kind::mxf8f6f4`); **no WGMMA, no tcgen05/TMEM** | TMA + `cp.async`, ~99 KB SMEM per block |

FP8 GEMMs from C++ (sm_89+): call cuBLASLt directly (`cublasLtMatmul` with the
descriptors, layouts and algorithm cached per shape; `-lcublasLt`): ~6 us of host time
per call against ~19 for `at::_scaled_mm` from C++ (ATen's checks and a fresh output).
Example to copy: `examples/cuda_cublaslt_fp8.py` (written from measured code,
not yet run on a GPU): tensor-wise (`SCALAR_32F`, unit scales: the unscaled product, scales
applied by a small epilogue or by the consumer) and MXFP8 (`VEC32_UE8M0`, ue8m0 scales
per 32 along K in the 128 x 4 blocked layout; sm_100+ / sm_120) modes, split-K as a
strided batch over K slices, the heuristic's algorithms timed once, several GEMMs on
one input behind one pybind call. On sm_120 cuBLASLt has no row-wise (`OUTER_VEC_32F`)
or DeepSeek blockwise (`VEC128_32F`, `BLK128x128_32F`) scale mode (they are sm_90 only).
In hand-written kernels, the instruction that reaches the FP8 peak depends on the GPU
(the prompt's "This GPU" section): on sm_120 the block-scaled MMA (`mma.sync ...
kind::mxf8f6f4.block_scale`, `sm_120a`, 416 TFLOP/s on an RTX 5070 Ti; unit ue8m0 scales
127 when the real scales are per tensor / row), since plain `mma.sync ...
e4m3.e4m3.f32` runs at half that rate on GeForce; on sm_89 plain e4m3 `mma.sync`; on
sm_90 `wgmma` and on sm_100 `tcgen05.mma` (e4m3 `mma.sync` is emulated through fp16
there).

Per-arch details, verified compiles and what to measure first: the `gpu-architectures` skill (`turing.md`, `ampere.md`, `ada.md`, `measure-first.md`). On a newer GPU, `TORCH_CUDA_ARCH_LIST="8.6+PTX"` (with its own `TORCH_EXTENSIONS_DIR`) runs a `load_inline` kernel's sm_86 code path through PTX JIT: a correctness check, never a timing.

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

## Examples and sources

* Examples: `examples/cuda_rmsnorm.py`, `examples/nvrtc_rmsnorm.py`, `examples/cuda_cublaslt_fp8.py`, `cuda_fp8_gemv.py`, `cuda_fp8_skinny_gemm.py`, `cuda_fp4_gemv.py`, `examples/cuda_pdl_gemv_chain.py`. All in kernel-agent's examples directory (`kernel_agent/agent/examples/`; a session's prompt gives the directory): copy their structure; an example's `ARCHS` names the GPUs it runs on.
* Sources: the `documentation-sources` skill's `sources.md`, sections "Local reference code"; "CUDA C++ and PTX"; "cuBLASLt (FP8 / FP4 GEMMs, epilogues, heuristics)"; "CUTLASS / CuTe".
