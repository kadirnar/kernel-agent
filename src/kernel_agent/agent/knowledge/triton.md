# Triton backend

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

## sm_120 (RTX 50xx, consumer Blackwell), measured with Triton 3.8

Source: `docs/RESEARCH-TRITON.md` §1 (RTX 5070 Ti, CUDA-graph timing over 12 L2-cold
weights).

* Codegen: `tl.dot` lowers to `mma.sync` (`HMMA` bf16, `QMMA` e4m3) with `ldmatrix`
  and `cp.async`; there is no `wgmma`, no `tcgen05` / TMEM on sm_120. Shared memory
  is 101 376 B per block: Hopper / B200 configs (128x256x64 with 3-4 stages, the
  matmul tutorial's) fail with `OutOfResources`; the autotuner skips them, Inductor
  may silently fall back to `num_stages=1`.
* **FP8 with fp32 accumulation runs at half rate on GeForce, except block-scaled.**
  Register-only rates: bf16 104, e4m3 `tl.dot` (`QMMA.F32`) 208, block-scaled e4m3
  (`QMMA.SF`, `mma ... kind::mxf8f6f4.block_scale`) 416 TFLOP/s. A plain e4m3
  `tl.dot` GEMM therefore stops at ~190 TFLOP/s (4096^3); cuBLASLt's tensor-wise
  FP8 kernel reaches 325.
* **Use `tl.dot_scaled` for FP8 GEMMs.** It lowers to native `QMMA.SF` on sm_120
  (since triton-lang/triton#7918). With all scales 127 (ue8m0 2^0) the result is
  bit-identical to `tl.dot(a, b, acc)` with fp32 accumulation, so keep the
  per-token / per-channel scales in the epilogue:
  ```python
  # a: [BM, BK] e4m3, b: [BN, BK] e4m3 (weight rows), acc fp32 [BM, BN]
  ones_a = tl.full((BM, BK // 32), 127, tl.uint8)
  ones_b = tl.full((BN, BK // 32), 127, tl.uint8)
  acc = tl.dot_scaled(a, ones_a, "e4m3", b.T, ones_b, "e4m3", acc)
  ```
  With these constant scales and the row / column scales applied to `acc` in the
  epilogue, the gate|up GEMM [352, 1024] x [1024, 8192] ran 33.3 us (pointer loads)
  vs 36.4 us for `tl.dot`, output bit-identical to row-wise `torch._scaled_mm`.
  Scales may also be loaded from uint8 tensors [M, K // 32] / [N, K // 32]
  (slightly slower when they are all 127; real MX scales give MXFP8 blockwise
  scaling along K). Loaded unit scales at M = 352: gate|up 34.3 us (31.7 with TMA
  descriptors), q|k|v 14.3 -> 12.4, o_proj 14.2 -> 11.9, down 26.4 -> 25.4 (24.1
  TMA); 4096^3 190 -> 214 TFLOP/s (227 TMA). Check the PTX once:
  `kernel[grid](...).asm["ptx"]` must contain `block_scale`; without it Triton
  emulates through bf16 (much slower; gemlite notes that constant unit scales broke
  the SM120 lowering in Triton 3.7, they work in 3.8). The check needs no GPU:
  `triton.compile(ASTSource(fn=kernel, signature=..., constexprs=...),
  target=GPUTarget("cuda", 120, 32))` and read `.asm["ptx"]` (`gemm_ptx` /
  `block_scale_mma` in `examples/triton_fp8_w8a8_gemm.py`, whose GEMM uses this
  recipe; it keeps `tl.dot` below sm_100). Under `TRITON_INTERPRET=1` (Triton 3.8)
  `tl.dot` on bf16 operands, fp32 -> bf16 casts (truncated) and fp32 -> e4m3 casts
  (no carry into the exponent) are wrong: check numerics of such kernels on a GPU.
* Quantise in the producer: an RMSNorm / SiLU-mul row program that already holds the
  row writes e4m3 + its per-token scale (or 1 x 32 ue8m0 MXFP8 scales) for +0.1 us
  instead of a separate pass (+1.3 us at [352, 1024]): `examples/triton_fp8_producers.py`.
  Use `tl.math.div_rn` for the scale and the codes when they must equal torch's
  (Triton 3.8 lowers `/` on fp32 to `div.full.f32`, an approximate division).
* cuBLASLt tensor-wise FP8 (`torch._scaled_mm` with scalar scales, an
  `nvjet_sm120_qqtst_*` kernel) is still faster on a plain GEMM (gate|up 30.2 us).
  A Triton FP8 GEMM pays when it fuses work cuBLASLt cannot (row-wise scales,
  split-K for under-filled N, epilogues) or runs under a CUDA graph where its host
  cost does not count.
* TMA: host descriptors (`from triton.tools.tensor_descriptor import
  TensorDescriptor`; `TensorDescriptor.from_tensor(w, [BN, BK])`, then
  `desc.load([off_n, k])` in the kernel) compile to `cp.async.bulk.tensor` and work.
  Same speed as pointer loads for `tl.dot`; 5-8 % faster for `tl.dot_scaled` at
  our shapes (gemlite turns TMA off for MXFP8 on sm_120: time both). Base 16-byte
  aligned, leading strides multiples of 16 bytes.
* Do not use `tl.range(..., warp_specialize=True)` on sm_120: `tl.dot` got 11-29 %
  slower, `tl.dot_scaled` failed to compile (`TritonGPUOptimizePartitionWarps`).
* Tiles that won at M = 352 (graph-timed): e4m3 `tl.dot` BM 64, BN 64, BK 64, 4 warps,
  3 stages (N = 8192) or 128 x 64 x 128, 8 warps (N = 1024-2560); `tl.dot_scaled`
  64 x 64 x 128, 4 warps, 3 stages on most shapes. N = 1024 at M = 352 leaves SMs idle (48 tiles of
  128 x 64 on 70 SMs): 2-way split-K over 64 x 64 tiles with fp32 partials and a
  small reduce kernel cut down [4096 -> 1024] from 26.3 to 21.7 us.
* Short attention (VoxCPM2 LocDiT: S = 11, 16 q heads, 2 kv heads, D = 128): one
  program per (sequence, q head), whole Q / K / V tiles of 16 rows in registers,
  fp32 softmax, bf16 P, `tl.dot` for QK^T and PV: 5.7 us vs cuDNN SDPA 15.9 us and
  FlashAttention 35 us (128-row tiles mostly padding). It was routed from
  `F.scaled_dot_product_attention` by a `torch.overrides.TorchFunctionMode`
  entered inside the module's forward, into a `torch.library.custom_op`, which
  traces under `torch.compile(fullgraph=True)`. `examples/triton_short_attention.py`
  is the general version: query / key length <= 32, any head counts, GQA ratio and
  head dim (padded to a power of two), causal / boolean / additive masks, fp16 /
  bf16, `sdpa(...)` drop-in, `ShortAttentionMode` routing, cached eager launch,
  `num_warps` from the tuned-config cache.

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
  FlagGems (ATen ops, persistent tuning cache); URLs in `sources.md`.
