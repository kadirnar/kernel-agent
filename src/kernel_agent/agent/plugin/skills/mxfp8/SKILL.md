---
name: mxfp8
description: MXFP8 W8A8 kernels (precision fp8_mx) on block-scaled tensor cores — when they pay (M >= ~64, N >= ~2560), the ceil scale rule and its guard, the swizzled scale layout, F.scaled_mm and tl.dot_scaled, accuracy, host time. Use when a target's spec says fp8_mx.
---

# MXFP8 W8A8 (`precision: fp8_mx`)

Read the `precision-tiers` skill first; GEMMs at N <= ~1024 inside an `fp8_mx` target follow the `fp8-w8a8` skill.

For a target whose spec says `"precision": "fp8_mx"`: W8A8 on the block-scaled
tensor cores (sm_100 / sm_120, `mma.sync kind::mxf8f6f4.block_scale`): e4m3 weights
and activations with one power-of-two ue8m0 scale per 32 consecutive elements along K
on both operands, applied by the tensor core (no promotion step). Same
near-lossless tier as `fp8_w8a8`; allowed by default in near-lossless runs (8-bit).
Verified-style example (written from the measurements, not yet run on a GPU):
`examples/triton_mxfp8_gemm.py`. Helpers: `quantize_mxfp8`, `dequantize_mxfp8`,
`swizzle_mx_scales`, `mx_scale_offset`, `mxfp8_linear` (reference), `mxfp8_error`,
`mxfp8_saturation` in `kernel_agent.kernels.quant`.

**When it pays** (decide from the target's bound and shapes, not the model's name):
compute-bound GEMMs (M >= ~64 rows per call; clearly FLOP bound from the bf16 ridge,
M ~130 on an RTX 5070 Ti) with wide outputs (N >= ~2560), on a GPU whose *Ceilings*
table has an *MXFP8* column (block-scaled MMA measured). Measured there (docs/FP8.md
§3.1, M = 352, L2-cold CUDA graph): cuBLASLt MXFP8 runs [352 x 1024] x [1024 x 8192]
in 24.7 us (239 TFLOP/s) vs 29.5 tensor-wise FP8 and 69.0 bf16, N = 2560 in 9.3 vs
10.6, and its output arrives scaled (no s_x * s_w pass in the consumer). Not at
N <= ~1024 at such M: cuBLASLt offers one MXFP8 algorithm and no split-K, its large
tiles leave SMs idle (N = 1024: 14.8 / 26.6 us vs 8.8 / 15.2 for tensor-wise FP8
batched over two K halves). Inside an `fp8_mx` target run those GEMMs tensor-wise
with batched split-K (`fp8_w8a8` numerics, same tier) until a split-K or small-tile
MXFP8 kernel exists (Triton `tl.dot_scaled` with split-K, CuTe), or plan the target
`fp8_w8a8`. Not at a few rows per call: memory bound, the same bytes as FP8
weight-only (`fp8_weights`), and cuBLASLt's single MXFP8 algorithm is slower there.

The contract:

* **Weights** once in `build()`: `quantize_mxfp8(weight)` → e4m3 codes `[N, K]` and
  e8m0 scales `[N, K / 32]`; swizzle the scales once (`swizzle_mx_scales`); no bf16
  copy kept.
* **Activations per block of 32, every call** (dynamic), scale
  `2^ceil(log2(amax / 448))`: the smallest power of two that keeps the block within
  ±448, so the block maximum lands in (224, 448] and nothing saturates. Exact from the
  exponent bits of `amax = m * 2^E` (m in [1, 2)): `e = E - 8`, plus one when
  `m > 1.75` (448 = 1.75 * 2^8); a zero block gets code 0. **Never the OCP reference
  rule** `2^(floor(log2 amax) - 8)`: it maps block maxima in [256, 512) x scale onto
  e4m3 and clamps those above 448, a quarter of the blocks on Gaussian data and the
  largest elements of every block holding a massive activation (shrunk by up to
  12.5 %): measured, it fails the tier on a DiT layer whose o_proj / down_proj inputs
  carry outlier channels (norm off by 4.0 % / 2.0 %) and passes on the same layer's
  redrawn inputs, which have none.
* **The scale-rule guard**: define a module-level `quantize_activations(x) ->
  (codes, scales)` with the rule your kernels use (codes e4m3 `[rows, K]`, scales
  e8m0 or uint8 biased exponents `[rows, K / 32]`, unswizzled). The evaluator runs it
  on the captured input and on a stress input of the same shape whose block maxima
  sit at 1.9 x 2^e (where the OCP rule saturates every block) and rejects the
  candidate when a block maximum exceeds 448 x scale (`stage: scale_rule`, with the
  reason). Its `scale_rule` report also gives the captured input's outliers: the
  largest `|x| / RMS` of a row (`crest`) and the largest channel amax over the median
  channel's (`channel_ratio`; hundreds and more: massive activations).
* **Math**: `F.scaled_mm(x_q, w_q.t(), scale_a=sa, scale_recipe_a=
  ScalingType.BlockWise1x32, scale_b=sb, scale_recipe_b=ScalingType.BlockWise1x32,
  swizzle_a=SwizzleType.SWIZZLE_32_4_4, swizzle_b=SwizzleType.SWIZZLE_32_4_4,
  output_dtype=torch.bfloat16)` (cuBLASLt `VEC32_UE8M0`; K a multiple of 128, any M),
  the scales in 128 x 4 blocks: row `r`, column `c` of `[rows, K / 32]` at
  `((r // 128) * ceil(K / 128) + c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4
  + c % 4` (`mx_scale_offset`; rows padded to 128). A quantiser can write its scales
  there directly (the example does). Or Triton `tl.dot_scaled(a, sa, "e4m3", b, sb,
  "e4m3", acc)` (`QMMA.SF`; 215 TFLOP/s at 8192^3 with real scale loads vs cuBLASLt's
  316-320). fp32 accumulation, bias once, one rounding to bf16.
* **Report** `mxfp8_error(weight, q, scales, x)` on captured activations (weight
  error, `activation_rel_l2`, `activation_saturation`, output relative L2 / cosine /
  norm ratio) and the evaluator's per-case `min_cosine` / `max_rel_l2` in `NOTES.md`.
* **Fallback / reference**: `mxfp8_linear(x, q, scales, bias)` (`F.scaled_mm` where it
  applies, else the same math in fp32).

**Accuracy** with the ceil rule (docs/FP8.md §3.2, fake quant on real captures):
the error of per-token W8A8 (DiT layer at M = 352: relative L2 0.0205 vs 0.0204,
norm 0.92 %, passes 30 redrawn draws); finer than per-token at M = 1 where an
activation's crest is high (an LM q_proj at decode: 0.023 vs 0.037, which fails),
but decode is memory bound: `fp8_weights` there.

**Scales: dynamic by default.** A static (offline-calibrated) activation scale saves
<= 0.3 us per producer call and saturates on inputs larger than the calibration set
(measured: up to 8.8 % of another text's calls; a layer with static scales failed
the redrawn-input check in 1 of 10 draws). Use one only when the target passes the
evaluator's redrawn-input check with it; group scales (1 x 32, 1 x 128) are the local
alternative when a producer cannot see a whole row.

**Host time**: `F.scaled_mm` (MXFP8) costs ~27 us per eager call, a direct
`cublasLtMatmul` with cached descriptors ~6 us; inside CUDA graphs neither counts.

## Examples and sources

* Examples: `triton_mxfp8_gemm.py`, `cuda_cublaslt_fp8.py`, `triton_fp8_producers.py`. All in kernel-agent's examples directory (`kernel_agent/agent/examples/`; a session's prompt gives the directory): copy their structure; an example's `ARCHS` names the GPUs it runs on.
* Sources: the `documentation-sources` skill's `sources.md`, sections "Low precision (formats, scaling, accuracy)"; "cuBLASLt (FP8 / FP4 GEMMs, epilogues, heuristics)"; "Triton".
* Code: `kernel_agent.kernels.quant` (`quantize_mxfp8`, `dequantize_mxfp8`, `swizzle_mx_scales`, `mx_scale_offset`, `mxfp8_linear`, `mxfp8_error`, `mxfp8_saturation`).
