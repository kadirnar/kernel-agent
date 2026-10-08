---
name: fp8-weights
description: FP8 weight-only kernels (precision fp8_weights) — the contract, dequantisation in registers on any GPU, GEMV and skinny-GEMM designs, sm_120 tensor cores, measured speedups. Use for memory-bound decode GEMVs and skinny GEMMs whose spec says fp8_weights.
---

# FP8 weight-only (`precision: fp8_weights`)

Read the `precision-tiers` skill first (when it is allowed, the tolerance tier, formats, scale granularity). Measured speedups and how the evaluator times FP8 weights: [measured.md](measured.md).

## The contract

* **Quantise once, in `build()`**: never per call, and keep no bf16 copy of the
  weight (half the bytes to stream is the point; half the memory comes with it).
* **One scale per output channel** (a row of `nn.Linear`'s `[out, in]` weight):
  symmetric, `scale = amax(|row|) / 448`, codes `round(w / scale)` in e4m3
  (round to nearest even, clamped to ±448). The scale is constant along k, so it
  is applied once per output: `y[m, n] = scale[n] * sum_k x[m, k] q[n, k] + bias[n]`.
* **Activations stay bf16.** Never quantise activations in an `fp8_weights`
  target (that is W8A8, the `fp8_w8a8` class of the `fp8-w8a8` skill, for compute-bound GEMMs).
* **Accumulate in fp32**; apply scale and bias in fp32 and round to bf16 once.
* **Report the numerical error** in `NOTES.md`: the weight report of
  `fp8_error(weight, q, scale)` (`rel_l2`, `worst_channel_rel_l2`, `underflow`,
  `crest`) and the evaluator's per-case `min_cosine` / `max_rel_l2` (reported in
  the near-lossless tier), plus `metrics.perceptual` of an `evaluate_e2e`.
* Fallback (shapes the kernel does not cover): the dequantised FP8 weight through
  cuBLAS (`dequantize_fp8` or `q.float() * scale[:, None]`), the same numerics.

## Dequantise in registers (CUDA; hardware conversion from sm_89)

* Weight-only formats run on every GPU (the prompt's "This GPU" section says what yours
  has): below sm_89 there is no FP8 hardware, the conversion below compiles to a software
  routine and the bf16 math is unchanged (vLLM runs FP8 checkpoints on Ampere this way).
* 128-bit loads = 16 e4m3 codes; stream them past L1 (`ld.global.nc.L1::no_allocate`).
* `__nv_cvt_fp8x2_to_halfraw2(v, __NV_E4M3)` (`cuda_fp8.h`) converts two codes to
  `f16x2` exactly (the low byte is `.x`); `__half22float2` -> fp32 for FMA
  (GEMV), `__float22bfloat162_rn` -> bf16x2 (exact: every e4m3 value is a bf16
  value) for a bf16 MMA. Convert each weight once, then reuse it for every
  activation row.
* GEMV (M = 1): warp per output row, fp32 FMA, warp-shuffle reduction, scale in
  the epilogue. At M = 2..4 the per-row FMA work makes it ALU bound when the
  weight is L2-resident: the tensor-core kernel is better from M = 2.
* Skinny GEMM (M <= 32): `mma.sync.m16n8k16.bf16` with the weight as the A
  operand (16 output channels x k16) and the tokens as B, both loaded straight
  from global memory with the same k permutation (thread t of a quad loads 16
  consecutive k; mma j uses k 16t + 4j .. 16t + 4j + 3 of A and B). Several channel
  tiles per block divide the activation traffic from L2; split k across warps
  and reduce through shared memory.
* Triton (sm_89+: Triton has no e4m3 type below; there, load `uint8` codes and build
  the bf16 bits with integer ops): load the codes from a `torch.float8_e4m3fn` tensor
  (`tl.float8e4nv`) and `w.to(tl.bfloat16)` before `tl.dot(x, w, acc)` (verified on sm_120;
  [22, 1024] x [1024, 4096] in 5.6 us with a warm L2). Triton's host overhead
  (an eager call took ~30 us here, the CUDA examples ~19 us) dominates decode
  calls; under CUDA graphs it does not matter.

## Tensor cores and libraries on sm_120 (RTX 50xx), verified here

* `mma.sync` bf16/f16 (weight-only after upcast: the examples).
* `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` runs on plain `sm_120`
  (exact results), but both operands are FP8: W8A8 (`fp8_w8a8`), not weight-only.
* Block-scaled MMA (`mma.sync ... .kind::mxf4 / mxf4nvf4 / mxf8f6f4 .block_scale`,
  FP4/FP6/FP8 with ue8m0 / ue4m3 scales) needs the arch-specific target:
  `extra_cuda_cflags=["-gencode=arch=compute_120a,code=sm_120a"]` (ptxas rejects
  it for `sm_120`). No WGMMA, no tcgen05 / TMEM on sm_120.
* `torch._scaled_mm`: FP8 row-wise scales and NVFP4 (`float4_e2m1fn_x2` with
  e4m3 block scales, 128-row padded scale layout) both work, but they quantise
  the activations too (W8A8 / W4A4; usage and limits: the `fp8-w8a8` skill).
* Rates (register-only `mma.sync`, docs/RESEARCH-TRITON.md §1.1): e4m3 with fp32
  accumulation (`QMMA.F32`) 208 TFLOP/s, the block-scaled e4m3 instruction
  (`kind::mxf8f6f4.block_scale`, `QMMA.SF`) 416, bf16 104. Every FP8 kernel on the
  plain instruction (row-wise `_scaled_mm`, Triton `tl.dot`, CUTLASS SM120 dense or
  blockwise) stops below 208; cuBLASLt's tensor-wise and MXFP8 kernels reach 316-329
  at 8192³. FP8 scaling recipes on this GPU: the `fp8-w8a8` skill's `scaling-recipes-sm120.md`.

## Examples and sources

* Examples: `cuda_fp8_gemv.py`, `cuda_fp8_skinny_gemm.py`. All in kernel-agent's examples directory (`kernel_agent/agent/examples/`; a session's prompt gives the directory): copy their structure; an example's `ARCHS` names the GPUs it runs on.
* Sources: the `documentation-sources` skill's `sources.md`, sections "Low precision (formats, scaling, accuracy)"; "CUDA C++ and PTX"; "CUTLASS / CuTe".
* Code: `kernel_agent.kernels.quant` (`quantize_fp8`, `dequantize_fp8`, `fp8_error`).
