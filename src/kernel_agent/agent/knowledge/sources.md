# Sources: where to look things up

One line per source: what it answers. Local reference code first (Grep / Read it:
exact text, no fetch), then official docs and library examples, then papers.
`WebFetch` answers your question from one page with a small model, ~100K characters
per call; the result says where it stopped, and `offset` reads on. Two NVIDIA
references are single pages longer than WebFetch reaches: the PTX ISA ends near
§9.7.9 (before mma, ldmatrix, cp.async.bulk) and cuBLAS before §3 (cuBLASLt). For
those, use the local headers and the examples below. All URLs were checked (2026-10).

## Local reference code (paths: the prompt's "Local sources")
* CUTLASS `cute/arch/mma_sm120.hpp`: exact PTX of sm_120 block-scaled `mma.sync`
  (`kind::mxf8f6f4`, `kind::mxf4nvf4`, ue8m0 / ue4m3 scales) and its operand registers.
* CUTLASS `cute/arch/mma_sm89.hpp`, `mma_sm80.hpp`: e4m3/e5m2 and bf16/f16 `mma.sync` PTX.
* CUTLASS `cute/arch/copy_sm80.hpp` (cp.async), `copy_sm75.hpp` (ldmatrix),
  `copy_sm90_tma.hpp` (TMA `cp.async.bulk.tensor`).
* `cublasLt.h`: every cuBLASLt matmul attribute, scale mode, epilogue and heuristic
  call, documented in its comments.
* `triton/language/core.py`: signatures and docstrings of `tl.dot`, `tl.dot_scaled`, ...

## CUDA C++ and PTX
* https://docs.nvidia.com/cuda/cuda-programming-guide/index.html — CUDA Programming Guide (split into short pages, start here).
* https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html — cp.async / TMA / bulk copies from C++, pipelines.
* https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html — PDL: overlap a kernel's prologue with the previous kernel.
* https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html — CUDA graphs: capture, update, launch cost.
* https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html — L2 persistence / access policy windows.
* https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html — per-architecture limits (sm_120: smem, registers, features).
* https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html — coalescing, occupancy, memory optimisations.
* https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html — Blackwell tuning (L2, smem, tensor cores).
* https://docs.nvidia.com/cuda/parallel-thread-execution/index.html — PTX ISA (only up to §9.7.9 via WebFetch; see above).
* https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html — `asm volatile` constraints and clobbers.
* https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html — FP8 conversion intrinsics.
* https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__BFLOAT16.html — bf16 intrinsics.
* https://docs.nvidia.com/cuda/nvrtc/index.html — NVRTC options and name expressions.
* https://nvidia.github.io/cuda-python/cuda-core/latest/ — `cuda.core` Program / launch API.
* https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/ — 128-bit vector loads.

## cuBLASLt (FP8 / FP4 GEMMs, epilogues, heuristics)
* https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLASLt — runnable samples: `LtFp8Matmul` (tensor-wide scales), `LtFp8CustomFind` (algo search), `LtMxfp8Matmul`, `LtNvfp4Matmul`, `LtBlk128x128Fp8Matmul`.
* https://developer.nvidia.com/blog/boosting-matrix-multiplication-speed-and-flexibility-with-nvidia-cublas-12-9/ — cuBLAS 12.9 scaling modes (block, outer-vector) and where they run.
* https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ScaledBlas.cpp — how `torch._scaled_mm` checks layouts and scale modes before cuBLASLt.

## CUTLASS / CuTe
* https://github.com/NVIDIA/cutlass/tree/main/examples/79_blackwell_geforce_gemm — sm_120 NVFP4 x bf16, NVFP4 x NVFP4 and MXFP8 x MXFP6 GEMMs.
* https://github.com/NVIDIA/cutlass/tree/main/examples/87_blackwell_geforce_gemm_blockwise — sm_120 FP8 x FP8 -> bf16 GEMMs with blockwise / groupwise scales.
* https://github.com/NVIDIA/cutlass/blob/main/examples/91_fp4_gemv/91_fp4_gemv.cu — FP4 GEMV.
* https://github.com/NVIDIA/cutlass/blob/main/examples/94_ada_fp8_blockwise/ada_fp8_blockwise.cu — sm_89-class (mma.sync) FP8 blockwise GEMM.
* https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/cute/blackwell_geforce/kernel — CuTe DSL sm_120 kernels: `dense_gemm`, `blockscaled_gemm`.
* https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/cute/ampere — CuTe DSL mma.sync kernels (`dense_gemm`, `attention`, `elementwise`) and a tutorial.
* https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html — CuTe DSL documentation.
* https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_warp.html — CuTe DSL warp MMA ops: `MmaFP8Op` (sm_89+), `MmaMXF8Op` / `MmaMXF8F6F4Op` / `MmaMXF4NVF4Op` (sm_120a).
* https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/tvm_ffi_compilation.html — CuTe DSL with TVM-FFI: lower host overhead per launch.
* https://github.com/HazyResearch/ThunderKittens — ThunderKittens: C++ register / shared tile primitives, TMA, Blackwell (MXFP8, NVFP4).

## Triton
* https://triton-lang.org/main/python-api/triton.language.html — `triton.language` reference (every op).
* https://triton-lang.org/main/python-api/generated/triton.language.dot_scaled.html — `tl.dot_scaled`: MX / NVFP4 block-scaled dot.
* https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html — block-scaled (MXFP4/NVFP4/MXFP8) matmul tutorial.
* https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html — persistent matmul, TMA descriptors, FP8.
* https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html — flash attention in Triton.
* https://github.com/triton-lang/triton/tree/main/python/tutorials — tutorial sources (incl. programmatic dependent launch, gluon).
* https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html — tiled matmul, GROUP_SIZE_M L2 grouping, FP8 (its largest configs exceed sm_120's 101 KB smem).
* https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html — TMA tensor descriptors in a kernel: alignment rules, `triton.set_allocator`.
* https://raw.githubusercontent.com/triton-lang/triton/main/python/tutorials/11-programmatic-dependent-launch.py — PDL in Triton (`gdc_wait`, `gdc_launch_dependents`, `launch_pdl=True`).
* https://github.com/triton-lang/triton/tree/main/python/triton/tools — ahead-of-time compile / link of Triton kernels (launch from C without Python).
* https://github.com/triton-lang/triton/pull/7918 — native MXFP8 `tl.dot_scaled` on sm_120 and its limits (1X scale vector, both scales).
* https://github.com/flashinfer-ai/flashinfer/issues/5963 — GeForce sm_120: FP8 MMA with fp32 accumulation at half rate; block-scaled MMA with unit ue8m0 scales, bit-identical, 2x.
* https://github.com/vllm-project/vllm/pull/58877 — Triton kernels on SM12x: 101 KB smem, `block_m`, stages, persistent / epilogue-subtile changes.

## Triton libraries (reference kernels and techniques)
* https://github.com/linkedin/Liger-Kernel/tree/main/src/liger_kernel/ops — Liger-Kernel Triton ops: RMSNorm (+ fused add), RoPE, SwiGLU / GeGLU, cross entropy, fused MoE.
* https://arxiv.org/abs/2410.10989 — Liger-Kernel paper: fusion and chunking techniques, test tolerances, int32-offset and contiguity bugs.
* https://github.com/unslothai/unsloth/tree/main/unsloth/kernels — Unsloth: in-place RoPE, FP8 block quant / W8A8 block matmul, `triton_launch.py` (direct `CompiledKernel` launches).
* https://github.com/dropbox/gemlite — Triton GEMV / split-K / GEMM chosen by M, low-bit and MXFP8 (`tl.dot_scaled`) kernels with sm_120 notes.
* https://github.com/flagos-ai/FlagGems — 180+ ATen ops in Triton, pointwise codegen, `LibTuner` / `LibEntry` (persistent tuning cache, cheap launches).
* https://github.com/ByteDance-Seed/Triton-distributed — communication inside Triton kernels (multi-GPU compute / communication overlap).

## TileLang
* https://tilelang.com/programming_guides/overview.html — programming guide.
* https://github.com/tile-ai/tilelang/tree/main/examples — examples: `gemm`, `gemm_fp8`, `flash_attention`, `deepseek_mla`, ...
* https://arxiv.org/abs/2504.17577 — TileLang paper: scheduling annotations, GEMM / attention against Triton.
* https://github.com/tile-ai/tilelang/pull/3099 — TileLang block-scaled `mxf8f6f4` MMA on sm_120 (FP8 at full rate with fp32 accumulation).

## PyTorch (2.14)
* https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs — CUDA graphs in PyTorch (capture rules, static inputs, pools).
* https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html — `mode="reduce-overhead"` CUDA graph trees.
* https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler.html — torch.compile.
* https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html — custom ops (`torch.library`), compile-friendly kernels.
* https://docs.pytorch.org/docs/2.14/cpp_extension.html — `load_inline` and its build options.

## Attention and LLM kernels (reference code)
* https://docs.flashinfer.ai/api/attention.html — FlashInfer decode / prefill attention API; sources: https://github.com/flashinfer-ai/flashinfer/tree/main/include/flashinfer
* https://github.com/Dao-AILab/flash-attention — FlashAttention-2 (and `hopper/`: FA3).
* https://crfm.stanford.edu/2023/10/12/flashdecoding.html — flash-decoding: split-KV for one-query decode.
* https://github.com/vllm-project/vllm/tree/main/csrc/quantization/w8a8 — production W8A8 kernels: `fp8/` (activation quantisation), `cutlass/` (scaled GEMMs).
* https://github.com/IST-DASLab/marlin — mixed-precision (int4 weight x fp16) GEMM near the bandwidth limit at small batch.

## Low precision (formats, scaling, accuracy)
* https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ — NVFP4: e2m1, e4m3 scale per 16, tensor scale.
* https://arxiv.org/abs/2310.10537 — Microscaling (MX) formats; spec: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
* https://arxiv.org/abs/2209.05433 — FP8 formats (e4m3 / e5m2) for deep learning.
* https://arxiv.org/abs/2211.10438 — SmoothQuant: migrate activation outliers into the weights for W8A8.
* https://arxiv.org/abs/2405.04532 — QServe: W4A8KV4 and its GEMM design (code: https://github.com/mit-han-lab/omniserve).
* https://github.com/pytorch/ao/tree/main/torchao/prototype/mx_formats — MX / NVFP4 reference quantisation in PyTorch.

## Attention and sampling algorithms (papers)
* https://arxiv.org/abs/2307.08691 — FlashAttention-2: work partitioning.
* https://arxiv.org/abs/2407.08608 — FlashAttention-3: asynchrony, FP8 attention.
* https://arxiv.org/abs/2210.02747 — Flow Matching (the ODE a CFM / DiT sampler integrates).
* https://arxiv.org/abs/2209.03003 — Rectified flow: straighter paths, fewer steps.
* https://arxiv.org/abs/2310.19075 — Bespoke solvers: few-step solvers for flow models.
* https://arxiv.org/abs/2211.01095 — DPM-Solver++: higher-order solvers for guided sampling.
* https://arxiv.org/abs/2410.12557 — Shortcut models: one- / few-step flow sampling.

## Kernel-generation agents: failures and reward hacks (for reviews of a candidate)
* https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/src/kernelbench/kernel_static_checker.py — KernelBench's static anti-hack rules per backend (Triton, CuTe, TileLang).
* https://arxiv.org/abs/2502.14752 — TritonBench: what LLM-written Triton gets wrong (API / type errors, wrong results).
* https://arxiv.org/abs/2510.17891 — TritonRL: Triton reward hacks (torch delegation, constants, identity) and a robust verifier.
* https://arxiv.org/abs/2507.14111 — CUDA-L1: side-stream, lazy-output, caching and shape-shrinking hacks.
* https://arxiv.org/abs/2603.19173 — SOL-ExecBench: hack frequencies on Blackwell and the harness controls against them.
* https://arxiv.org/abs/2607.16241 — KernelBench-Verified: scaled / negated hidden inputs, TF32 baseline, peak memory.
