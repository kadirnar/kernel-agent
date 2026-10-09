---
name: fp4-weights
description: FP4 weight-only kernels (precision fp4_weights, NVFP4) — when over FP8, the e2m1 + block-scale format and its tier, the contract, dequantising in registers, measured GEMVs, per-layer selection. Use for memory-bound decode GEMVs whose spec says fp4_weights.
---

# FP4 weights (`fp4_weights`)

Read the `precision-tiers` skill first; the skinny-GEMM design it reuses is in `fp8-weights`.

**When**: only memory-bound decode GEMVs / skinny GEMMs (weights streamed once
per call) where `fp8_weights` is already in use or the planner's ceilings table
shows the target still bound by streaming weights. Per weight NVFP4 streams 4.5 bits (56 %
of FP8's bytes, 28 % of bf16's) and moves outputs ~4x more than FP8.

**Format** (`quantize_fp4(weight)`, NVFP4; `kernels/quant.py`):

* codes: e2m1 (1 sign, 2 exponent, 1 mantissa bit; magnitudes 0, 0.5, 1, 1.5, 2,
  3, 4, 6; code `c | 8` is the negative), two per byte, element 2j in the low
  nibble of byte j: `uint8 [out, in / 2]` (`.view(torch.float4_e2m1fn_x2)`
  for torch's FP4 ops);
* block scales: e4m3, one per 16 consecutive weights of a row, `amax(block) / 6
  / tensor_scale` rounded to e4m3: `float8_e4m3fn [out, in / 16]`;
* tensor scale: fp32, `amax(W) / (6 * 448)` (keeps every block scale in e4m3's
  range); `w[n, k] = e2m1[n, k] * scale[n, k / 16] * tensor_scale`;
* `fmt="mxfp4"`: an e8m0 (power-of-two) scale per 32, 4.25 bits, ~20 % more error
  (0.116 vs 0.095 relative L2 on VoxCPM2's weights), and it fails the FP4 tier on
  some calls (3 of 147 GEMVs, 4 of 18 MLPs): prefer NVFP4. `quantize_fp4` takes
  the smallest power-of-two scale that keeps the block within ±6; the OCP
  reference conversion (`2^(floor(log2 amax) - 2)`) saturates block maxima,
  which real activations meet: VoxCPM2's outputs shrank by up to 13 %.

**Error** (VoxCPM2, all 343 `nn.Linear` weights): weight relative L2 0.095
(worst channel 0.128; FP8 0.026); `underflow` ~7 % (weights below a quarter of
their block's step become 0: normal for FP4, not a bug). On real inputs: GEMV /
GEMM outputs mean 0.055, worst 0.21 (a small, 256-wide decode output); MLPs
<= 0.13; the 12-layer LocDiT estimator <= 0.16.

**Tier** `near-lossless-fp4`: per output tensor cosine >= 0.96, relative L2 <=
0.28, norm within ±4 % (noise adds energy, `sqrt(1 + rel_l2^2)`, 2.1 % at
worst), every element within 1.25 x RMS + 0.25 x |reference|. Swapped nibbles,
block scales off by one block, a wrong tensor scale fail it; a dropped output
channel often does not (FP4 noise hides it in a GEMV): the perceptual gate is
the real judge. On the perturbed-input check's redrawn inputs: cosine >= 0.94,
relative L2 <= 0.40, norm within ±12 %, every element within 2.5 x RMS (the
larger of the tensor's and its channel's) + 0.25 x |reference| (an LM decode
step's 256-value V-cache slot reaches relative L2 0.33 and +9.7 % there, the LM
down_proj GEMV 2.06 x RMS in its massive-activation channel).

**The contract** (as FP8, with the block scales):

* quantise once in `build()` with `quantize_fp4`, keep no bf16 copy;
* activations stay in the model's dtype, bf16 or fp16 (FP4 activations are W4A4:
  `fp4_w4a4`, skill `fp4-w4a4`);
* fp32 accumulation: a partial sum per 16-weight block times its block scale
  (or the weights dequantised with their scale), the tensor scale and bias once
  per output, one rounding to that dtype;
* report `fp4_error(weight, codes, scales, tensor_scale)` and the evaluator's
  per-case numbers in `NOTES.md`;
* fallback: the dequantised weight (`dequantize_fp4`) through cuBLAS.

**Dequantise in registers** (`examples/cuda_fp4_gemv.py`): for two codes in
bytes 0 and 2 of a word (`__byte_perm(w, w >> 4, sel)` picks the nibbles of
byte j), `((t & 0x00070007) << 9) | ((t & 0x00080008) << 12)` is an `f16x2`
equal to `e2m1 * 2^-14` exactly (exponent and mantissa bits to f16 bits 11..9,
sign to bit 15; code 1 = 0.5 becomes an f16 subnormal), `__half22float2` to
fp32, and fold `2^14` into the block scale. A few integer ops per two weights,
no lookup table; at M = 1 the GEMV's outputs matched an fp32 matmul with the
dequantised weight bit for bit on the shapes below. `cvt.rn.f16x2.e2m1x2` exists only for the arch-specific
`sm_120a` target. One lane loads 16 bytes of codes (32 weights, two blocks) and
the two block scales (`__nv_fp8x2_storage_t`, converted with
`__nv_cvt_fp8x2_to_halfraw2`).

**Tensor cores**: `e2m1 x e4m3` has at most 6 significant bits, so a weight
dequantised with its block scale is exact in bf16: an FP4 skinny GEMM can feed
`cuda_fp8_skinny_gemm.py`'s bf16 `mma.sync` with `e2m1 * scale` and apply the
tensor scale in the epilogue. The block-scaled FP4 MMA (`kind::mxf4nvf4`,
`sm_120a`) needs FP4 activations (W4A4: precision `fp4_w4a4`, skill `fp4-w4a4`).

**Measured** (RTX 5070 Ti, streamed as above; GB/s of codes + scales):

| GEMV | cuBLAS bf16 | FP8 GEMV | FP4 GEMV | vs bf16 / FP8 |
|---|---|---|---|---|
| [1, 2048] x [2048, 6144] (LM gate / up) | 32.3 us | 16.2 us | 9.0 us, 784 GB/s | 3.6x / 1.8x |
| [1, 2048] x [2048, 12288] (gate + up merged) | 62.8 us | 30.8 us | 16.3 us, 868 GB/s | 3.9x / 1.9x |
| [1, 6144] x [6144, 2048] (LM down) | 31.5 us | 16.3 us | 9.7 us (`unroll=2`: 9.3 us) | 3.2x / 1.7x |
| [1, 2048] x [2048, 2048] (LM q / o) | 11.9 us | 6.3 us | 3.9 us, 606 GB/s | 3.1x / 1.6x |
| [4, 2048] x [2048, 6144] | 31.5 us | 17.9 us | 19.9 us | 1.6x / 0.9x |

At M = 2..4 the GEMV is ALU bound (dequantisation plus a bf16-to-fp32
conversion of the activations per output row): use tensor cores there. With a
warm L2 (the evaluator) FP4 and FP8 are about equal (7.2 vs 6.3 us on the
gate / up weight, ~21 vs ~20 us eager with host overhead): judge FP4 streamed
and end to end. `pct_of_sol` of an `fp4_weights` target counts 4 bits per
weight plus one byte per 16 and 4 bytes per tensor.

**VoxCPM2** (fake-quantised weights; README "Quality modes"): NVFP4 in all 343
`nn.Linear` of both LMs and the LocDiT passes the perceptual gate (WER +0,
speaker similarity 0.978 / worst 0.957, MOS -0.09) but misses the sanity
floor's teacher forcing by a hair (mean step cosine 0.948 < 0.95); NVFP4 in
both LMs with the LocDiT in FP8 (89 % of the weights in FP4) passes everything
(0.982 / 0.959, MOS -0.01, teacher forcing 0.975). Per-layer selection, FP4
where the gate shows headroom and FP8 elsewhere, is the way to use it.

## Examples and sources

* Examples: `cuda_fp4_gemv.py`, `cuda_fp8_skinny_gemm.py`. All in kernel-agent's examples directory (`kernel_agent/agent/examples/`; a session's prompt gives the directory): copy their structure; an example's `ARCHS` names the GPUs it runs on.
* Sources: the `documentation-sources` skill's `sources.md`, sections "Low precision (formats, scaling, accuracy)"; "CUTLASS / CuTe".
* Code: `kernel_agent.kernels.quant` (`quantize_fp4`, `dequantize_fp4`, `fp4_error`).
