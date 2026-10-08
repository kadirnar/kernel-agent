---
name: optimisation-playbook
description: Optimisation methodology for any model on any NVIDIA GPU — classify hot spots (launch, memory or compute bound), the engineering loop for kernel candidates, host overhead per backend, hot paths per model family, model-level transforms. Use when planning targets or starting and steering a kernel.
---

# Optimisation playbook

The methodology for every target, any model, any GPU. More in this skill's directory:

* [model-families.md](model-families.md): where the time goes per model family (LLM and LLM-style TTS, diffusion, STT, diffusion-autoregressive TTS with the VoxCPM2 measurements, vocoders) and what wins there.
* [model-transforms.md](model-transforms.md): algorithm-level changes (static caches, CUDA graphs, merged projections, precomputed tables, host syncs, speculative decoding).

Related skills: `profiling-and-roofline` (reading the profile and the evaluator's numbers), the backend skills (`triton-kernels`, `cuda-kernels`, `cute-dsl`, `tilelang-kernels`), `precision-tiers`, `systems-patterns`.

## Diagnose before you write code

Classify every hot spot before choosing a technique:

| Regime | Symptom | What wins |
|---|---|---|
| **Launch / CPU bound** | GPU kernel time is a small share of latency; thousands of kernels of 1-5 us; decode loops with batch 1 | Fuse many small ops into one kernel, fewer Python ops per step, CUDA graphs, static KV cache, remove host syncs (`.item()`, `.cpu()`, Python control flow on tensors) |
| **Memory bound** | Elementwise / norm / softmax / GEMV (M=1..16) / attention decode; arithmetic intensity < ~50 FLOP/byte | Read every byte once: fusion, vectorised 128-bit loads, keep intermediates in registers/SMEM, lower precision weights, split-K / split-KV for parallelism |
| **Compute bound** | Large GEMM / conv / prefill attention; tensor-core kernels dominate | Tensor cores with the right MMA, tiling + pipelining (cp.async / TMA), epilogue fusion (bias, activation, residual, quantisation), FlashAttention-style tiling |

Roofline numbers: the evaluator does this for you. Peak copy bandwidth (DRAM
and L2), dense matmul TFLOP/s per dtype and the launch floor are measured on
this GPU (see "measured peaks" in the toolchain section), and every timed
result reports per case `flops`, `min_bytes` (what the reference must read and
write at least once), `sol_ms` = max(FLOPs / peak, bytes / bandwidth),
`pct_of_sol` and `bound`. Achieved bandwidth = `min_bytes` / `new_ms`. A
`memory` case at ≥ 80 % of SOL is at the bound of a single kernel: fuse it with its
neighbours (fewer bytes) or lower its precision to move that bound.
For a `launch` case the work is smaller than one launch from Python: only
fewer launches and less host work per call help. Cases that fit in L2
(`l2_resident`) are compared with the L2 bandwidth, because the benchmark runs
them with a warm cache.

## Engineering loop

1. Read the captured module's source (`spec.json → source_file`) and the
   captured shapes/dtypes. Write down the exact math, including dtype casts.
2. First candidate: simplest correct fused kernel. Evaluate. Give every evaluation a
   `title`, a commit subject of at most 72 characters saying what this version changes
   (`split-K=4, RED epilogue`): it names the experiment in the ledger, the charts and
   `kernel-agent exp`; the `hypothesis` says why it should be faster.
3. Then optimise with evidence: run `evaluate_candidate` with `profile=true`
   to see which kernels remain and how long each takes, which instructions they
   issue (the SASS census) and the directives (each with its numbers: test the
   one that names the largest share first, or say why its numbers do not hold).
4. Cover every captured case (prefill AND decode shapes). Specialise per shape
   inside `forward` if needed (e.g. GEMV path for M ≤ 16, tensor-core path for
   larger M). Fall back to the reference math for shapes you do not handle.
5. Tune with one sweep per idea: make block sizes, `num_warps`, `num_stages`
   and vector widths keyword arguments of `build(reference, BLOCK=..., ...)`
   and call `sweep_candidate` with the plausible grid (powers of two around
   the row/tile size, 1-8 warps, 2-4 stages). It checks and times every config
   in one GPU session and counts as one evaluation; a plain sweep beat agents
   that measured one config per evaluation (InferenceBench: 11.5x vs 8.1x).
6. Keep the best version; record what you learned in `NOTES.md`.

## Host overhead matters for tiny kernels

At decode shapes a kernel runs for 2-10 us, so Python-side launch cost is
visible in the measurement (it is real latency too). Measured on this machine
for a 2048-wide RMSNorm: CUDA C++ via `load_inline` ≈ 19 us, NVRTC ≈ 28 us,
CuTe DSL with TVM-FFI ≈ 25 us, TileLang ≈ 32 us, Triton ≈ 43 us, HF eager
(6 kernels) ≈ 74 us. Therefore: fuse *more* work per launch (a whole block, not
one op), avoid per-call Python work (no `from_dlpack`, dict lookups, shape
math or allocations you can hoist), and cache compiled kernels per shape.

## Examples and sources

* Examples: `examples/triton_rmsnorm.py`, `examples/cuda_rmsnorm.py`. All in kernel-agent's examples directory (`kernel_agent/agent/examples/`; a session's prompt gives the directory): copy their structure; an example's `ARCHS` names the GPUs it runs on.
* Sources: the `documentation-sources` skill's `sources.md`, sections "CUDA C++ and PTX"; "Triton".
