---
name: triton-kernels
description: Triton kernels for kernel-agent candidates — GEMM template, numerics, autotuning and the tuned-config cache, launch overhead (CachedLaunch), tiny-M GEMVs, masking, torch.compile custom ops, measured sm_120 facts (tl.dot_scaled FP8, TMA, tiles). Use when a target's backend is triton.
---

# Triton backend

Measured facts for GeForce Blackwell (sm_120: codegen, FP8 rates, `tl.dot_scaled`, TMA, warp specialisation, winning tiles, short attention) are in [sm120.md](sm120.md); read it on that GPU and before any FP8 Triton GEMM.

Verified example: `examples/triton_rmsnorm.py`.

```python
import triton, triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(
            a_ptr + rm[:, None] * stride_am + (k * BLOCK_K + rk)[None, :] * stride_ak,
            mask=(rm[:, None] < M) & ((k * BLOCK_K + rk)[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + (k * BLOCK_K + rk)[:, None] * stride_bk + rn[None, :] * stride_bn,
            mask=((k * BLOCK_K + rk)[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(a, b, acc)  # tensor cores; fp32 accumulate
    # fused epilogue goes here (bias, activation, residual, cast)
    tl.store(
        c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        acc.to(c_ptr.dtype.element_ty),
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )
```

Notes
* Launch: `kernel[grid](...)` where `grid` is a tuple or `lambda meta: (...)`.
* Block sizes must be powers of two (`triton.next_power_of_2`). `tl.dot` needs
  every dimension ≥ 16.
* Reductions: `tl.sum`, `tl.max`, `tl.argmax`; math: `tl.exp`, `tl.exp2`,
  `tl.rsqrt`, `tl.sigmoid`, `tl.math.*`; `tl.where`; `tl.cast`/`.to()`.
* Always compute in fp32 (`.to(tl.float32)`) and cast on store, matching the
  reference's cast points exactly (HF RMSNorm casts the normalised value to the
  input dtype *before* multiplying by the weight).
* Autotune only on shapes that vary; `key=[...]`. Autotuning runs on first
  call (cost paid during warm-up). Avoid autotune when shapes change every call.
* Tuned-config cache (`kernel_agent.kernels.tuned`): `@triton.autotune` tunes
  again in every process (each evaluation, sweep config, re-check and A/B is
  one). `tuned.best_config(op, {"M": m, "N": n, "K": k, "dtype": "bf16"},
  candidates, bench, exact=["N", "K"], default=...)` returns the stored config of
  this GPU, library versions (torch, CUDA, driver, Triton / CUTLASS / cuBLAS by
  `backend=`) and shape bucket (dims rounded up to a power of two except `exact`),
  else times the candidates with `bench(config) -> ms` (`tuned.time_ms(fn)`),
  stores the fastest in `~/.cache/kernel-agent/tuned-configs.sqlite` and returns
  it; an upgrade invalidates the entry; it never times while a CUDA graph is
  captured. Memoise the result per shape in your module (no lookup per call),
  name the op after the kernel's source digest (an edited kernel tunes again),
  and keep every candidate correct: the evaluator checks the one picked.
  `python -m kernel_agent.kernels.tuned` lists the entries. Also for cuBLASLt
  algorithm indices and CUTLASS configs chosen from C++ (`backend="cublaslt"`).
* Launch overhead is ~30-40 us of Python per call. For decode-time micro-ops,
  fuse more per kernel; for elementwise chains do everything in one kernel.
  When the module evaluator times eager calls, several Triton launches per call
  can make the candidate host bound: the VoxCPM2 `dit_layer` (cuBLAS GEMMs + 5
  Triton glue kernels) measured 2.21x with ~216 us host vs ~167 us GPU per call;
  the same math behind one `load_inline` C++ entry measured 2.70x. Launch less,
  or launch through `kernel_agent.kernels.triton_launch.CachedLaunch`: wrap the
  kernel once (`_k = CachedLaunch(_kernel)`), call `_k[grid](...)` exactly like
  `_kernel[grid](...)`; from the second call of a specialisation it calls the cached
  `CompiledKernel`'s C launcher directly, skipping the JIT's binding and
  specialisation (Unsloth's `triton_launch.py`, FlagGems' `LibEntry`). Plain
  `@triton.jit` kernels only (no `@triton.autotune` / `@triton.heuristics`); use
  the plain `kernel[grid]` when `torch.compiler.is_compiling()` (Dynamo traces it).
  `examples/triton_cheap_launch.py` (a pre-norm gated MLP block, 2 Triton
  launches + 2 GEMMs per call, `build(..., fast_launch=False)` for the JIT path to
  compare with `sweep_candidate`). The other route: Triton AOT
  (`python -m triton.tools.compile` + `triton.tools.link`) called from one
  `load_inline` entry. Under CUDA graphs / torch.compile host time does not count.
* Persistent kernels: `grid = (num_SMs,)` and loop over tiles inside.
* Debug: `TRITON_INTERPRET=1` runs kernels on CPU (slow) for logic bugs.
* Matmul with tiny M (decode GEMV): use a split-K or row-per-program design with
  `tl.sum(a[None, :] * w, axis=1)` rather than `tl.dot`, and vector loads of the
  weight (memory bound — goal is to stream weights at full bandwidth). From
  M ≈ 8 a `tl.dot` with BM = 16 (rows masked) and a deep K tile works too: at
  M = 16 an e4m3 x e4m3 GEMM with BM 16, BN 64, BK 256, 4 warps, 3 stages streamed
  a [12288, 2048] weight at 774 GB/s (the DRAM peak). gemlite's thresholds: GEMV
  up to M = 2-4, split-K GEMM below M = 64, plain GEMM from 64.
* Index arithmetic: `pid * stride` in int32 overflows past 2^31 elements (a
  Liger-Kernel bug): cast to `tl.int64` for large tensors. Mask every load whose
  address derives from a row index, also in the epilogue: an unmasked per-batch
  scale load `sc[rm // T]` read past the buffer for tail rows (M % BM != 0) and
  was refused by memcheck at integration; clamp (`tl.minimum(rm, M - 1)`).
* TF32: `tl.dot` on fp32 inputs uses `input_precision="tf32"`, which truncates
  the mantissa; cuDNN / cuBLAS TF32 rounds to nearest. Against a TF32 cuDNN
  reference at the strict fp32 tier (VoxCPM2 VAE) the bias compounded to relative
  L2 0.018; `"ieee"` still differs in summation order. Use `"tf32x3"` / `"ieee"`,
  or work in bf16 only when the target's precision tier allows it.
* torch.compile: the model may run under `torch.compile` (the compiled baseline,
  e.g. VoxCPM's `model.optimize()` with `fullgraph=True`). Wrap the launcher in
  `torch.library.custom_op` + `register_fake` (`examples/triton_rmsnorm_custom_op.py`)
  so it is one opaque op instead of a graph break, and check it with
  `evaluate_candidate(..., compile_check=true)`.

## When Triton is the right backend (VoxCPM2 ledgers, docs/RESEARCH-TRITON.md §4-5)

* Good: fused glue (norm, RoPE, activation, quantisation) and GEMM-shaped work
  with prologue / epilogue fusion inside CUDA graphs; attention with tiny
  sequences; split-K variants; channels-last conv-as-GEMM (VAE: 7.9x).
* Poor: eager-timed modules with many launches (host time); fused halo stencil +
  GEMM with large channel counts (a fused conv residual unit at C >= 64 spilled at
  255 registers; CUDA with an smem halo won); exact-tier GEMMs that must match
  cuBLAS's summation order; anything that needs warp specialisation on sm_120.
* Libraries to read for patterns: Liger-Kernel (RMSNorm, RoPE, SwiGLU, cross
  entropy), Unsloth `unsloth/kernels/` (RoPE, FP8 block quantisation, direct
  launches), gemlite (GEMV / split-K selection by M, `tl.dot_scaled` MXFP8),
  FlagGems (ATen ops, persistent tuning cache); URLs in the `documentation-sources` skill's `sources.md`.

## Examples and sources

* Examples: `examples/triton_rmsnorm.py`, `examples/triton_rmsnorm_custom_op.py`, `examples/triton_cheap_launch.py`, `examples/triton_short_attention.py`, `triton_fp8_w8a8_gemm.py`, `triton_fp8_producers.py`, `triton_mxfp8_gemm.py`, `triton_fp8_kv_decode.py`. All in kernel-agent's examples directory (`kernel_agent/agent/examples/`; a session's prompt gives the directory): copy their structure; an example's `ARCHS` names the GPUs it runs on.
* Sources: the `documentation-sources` skill's `sources.md`, sections "Triton"; "Triton libraries (reference kernels and techniques)".
