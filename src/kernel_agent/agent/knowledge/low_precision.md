# Low-precision weights (FP8, FP4)

Verified examples: `examples/cuda_fp8_gemv.py` (decode GEMV, M <= 4) and
`examples/cuda_fp8_skinny_gemm.py` (bf16 tensor cores, M <= 32; groups of 32
tokens beyond, slower than cuBLAS bf16 from M ~ 64); FP4:
`examples/cuda_fp4_gemv.py` (NVFP4 decode GEMV, M <= 4; "FP4 weights" below).
Helpers:
`kernel_agent.kernels.quant` (`quantize_fp8`, `dequantize_fp8`, `fp8_error`;
`quantize_fp4`, `dequantize_fp4`, `fp4_error`).

## When it is allowed

Only for a target whose spec says `"precision": "fp8_weights"`, which the planner
may set in a `--quality near-lossless` run (an exact run refuses such targets).
The target is then captured in the **near-lossless tolerance tier**
(`kernels/compare.py`): instead of per-element (atol, rtol), every output tensor
needs cosine >= 0.996, relative L2 error <= 0.08, its norm within ±2 % of the
reference (rounding noise is unbiased, a wrong scale is not) and every element
within 0.5 x RMS + 0.125 x |reference|. End to end, the run's perceptual gate
(WER, speaker similarity, MOS for TTS) decides. Without that spec field the
exact tier applies and FP8 weights fail it (~2.6 % relative L2 per GEMM, ~20 %
of the elements outside the bf16 tolerance). `"precision": "fp4_weights"`
(block-scaled FP4 weights) has its own tier, **near-lossless-fp4**: cosine >=
0.96, relative L2 error <= 0.28, norm within ±4 %, every element within 1.25 x
RMS + 0.25 x |reference| ("FP4 weights" below); FP4 fails the FP8 tier.

## The contract

* **Quantise once, in `build()`**: never per call, and keep no bf16 copy of the
  weight (half the bytes to stream is the point; half the memory comes with it).
* **One scale per output channel** (a row of `nn.Linear`'s `[out, in]` weight):
  symmetric, `scale = amax(|row|) / 448`, codes `round(w / scale)` in e4m3
  (round to nearest even, clamped to ±448). The scale is constant along k, so it
  is applied once per output: `y[m, n] = scale[n] * sum_k x[m, k] q[n, k] + bias[n]`.
* **Activations stay bf16.** Never quantise activations (that is W8A8, a
  different precision class that `fp8_weights` does not allow).
* **Accumulate in fp32**; apply scale and bias in fp32 and round to bf16 once.
* **Report the numerical error** in `NOTES.md`: the weight report of
  `fp8_error(weight, q, scale)` (`rel_l2`, `worst_channel_rel_l2`, `underflow`,
  `crest`) and the evaluator's per-case `min_cosine` / `max_rel_l2` (reported in
  the near-lossless tier), plus `metrics.perceptual` of an `evaluate_e2e`.
* Fallback (shapes the kernel does not cover): the dequantised FP8 weight through
  cuBLAS (`dequantize_fp8` or `q.float() * scale[:, None]`), the same numerics.

## Formats

| format | torch dtype | bits (s/e/m) | max | step near 1 | use |
|---|---|---|---|---|---|
| e4m3 ("fn": no inf, one NaN) | `torch.float8_e4m3fn` | 1/4/3 | 448 | 2^-3 | weights |
| e5m2 | `torch.float8_e5m2` | 1/5/2 | 57344 | 2^-2 | gradients; too coarse for weights |
| e2m1 (FP4) | `torch.float4_e2m1fn_x2` (2 per byte) | 1/2/1 | 6 | 0.5 | with block scales only |

e4m3 values: normals 2^-6 .. 448 with 3 mantissa bits, subnormals down to 2^-9.
Per-channel e4m3 on Gaussian-like rows: weight (and GEMM output) relative L2
error ~0.026, cosine ~0.9997. Errors add up over layers: a VoxCPM2-sized MLP
(gate, up and down in FP8) is ~0.046 against the bf16 MLP. FP4 needs block scales: NVFP4 =
e2m1 + an e4m3 scale per 16 elements (+ an fp32 tensor scale), MXFP4 = e2m1 +
a power-of-two (e8m0) scale per 32.

## Scale granularity and outliers

* **Per tensor**: one outlier channel sets the scale of all; the other channels
  lose mantissa bits or flush to zero. Avoid.
* **Per output channel** (the default): free at run time (epilogue), accurate on
  LLM / TTS weights (VoxCPM2: 147 `nn.Linear` calls, cosine >= 0.9992, relative
  L2 <= 0.04).
* **Per group along k** (e.g. 128): more accurate for rows with outliers, but the
  scale changes inside the k loop (scale partial sums per group); use only if
  per-channel fails the tier for a layer.
* Pitfalls: a high `crest` (amax / RMS of a row; ~4-5 for Gaussian rows)
  leaves few FP8 steps for the bulk of the row; `underflow` > 0 means weights
  became 0. A row of zeros gets scale 1 (never divide by 0). Clamp to ±448
  before the cast (`quantize_fp8` does). Merged weights (gate + up concatenated,
  QKV) keep their per-row scales: merge before quantising or after, the same.

## Dequantise in registers (sm_89+, CUDA)

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
* Triton: load the codes from a `torch.float8_e4m3fn` tensor (`tl.float8e4nv`)
  and `w.to(tl.bfloat16)` before `tl.dot(x, w, acc)` (verified on sm_120;
  [22, 1024] x [1024, 4096] in 5.6 us with a warm L2). Triton's host overhead
  (an eager call took ~30 us here, the CUDA examples ~19 us) dominates decode
  calls; under CUDA graphs it does not matter.

## Tensor cores and libraries on sm_120 (RTX 50xx), verified here

* `mma.sync` bf16/f16 (weight-only after upcast: the examples).
* `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` runs on plain `sm_120`
  (exact results), but both operands are FP8: W8A8, not weight-only.
* Block-scaled MMA (`mma.sync ... .kind::mxf4 / mxf4nvf4 / mxf8f6f4 .block_scale`,
  FP4/FP6/FP8 with ue8m0 / ue4m3 scales) needs the arch-specific target:
  `extra_cuda_cflags=["-gencode=arch=compute_120a,code=sm_120a"]` (ptxas rejects
  it for `sm_120`). No WGMMA, no tcgen05 / TMEM on sm_120.
* `torch._scaled_mm`: FP8 row-wise scales and NVFP4 (`float4_e2m1fn_x2` with
  e4m3 block scales, 128-row padded scale layout) both work, but they quantise
  the activations too (W8A8 / W4A4).

## Measured (RTX 5070 Ti, DRAM 767 GB/s copy peak, L2 48 MB)

"Streamed": CUDA graph of back-to-back calls over enough weight copies to exceed
L2, as in the model, where every layer's weights evict the others'.

| GEMM | cuBLAS bf16 | FP8 example | speedup |
|---|---|---|---|
| [1, 2048] x [2048, 6144] (LM gate / up) | 32.3 us, 780 GB/s | GEMV 16.2 us, 777 GB/s | 1.99x |
| [1, 2048] x [2048, 12288] (gate + up merged) | 62.8 us | GEMV 30.8 us, 819 GB/s | 2.04x |
| [1, 6144] x [6144, 2048] (LM down) | 31.7 us | GEMV 16.5 us, 764 GB/s | 1.93x |
| [22, 1024] x [1024, 4096] (LocDiT up) | 12.1 us | skinny 6.9 us, 610 GB/s | 1.75x |
| [22, 4096] x [4096, 1024] (LocDiT down) | 13.2 us | skinny 7.6 us, 556 GB/s | 1.75x |
| [32, 2048] x [2048, 6144] | 34.9 us | skinny 18.9 us, 667 GB/s | 1.85x |
| [8, 2048] x [2048, 6144] (batch-8 LM gate / up) | 31.5 us | skinny 16.3 us, 775 GB/s | 1.94x |
| [16, 2048] x [2048, 6144] (batch-16) | 31.7 us | skinny 16.6 us, 760 GB/s | 1.91x |
| [8, 6144] x [6144, 2048] (batch-8 LM down) | 39.0 us | skinny 17.0 us, 743 GB/s | 2.30x |
| [16, 6144] x [6144, 2048] (batch-16) | 39.1 us | skinny 17.6 us, 714 GB/s | 2.22x |
| [176, 1024] x [1024, 4096] (batch-8 LocDiT, CFG) | 20.0 us | skinny 26.6 us | 0.75x |

Batched decode (`-o batch_size=N`, `workloads/voxcpm_batch.py`): the LM GEMMs
have M = N and stay weight-bandwidth bound, so FP8 weights pay as at M = 1;
the LocDiT runs CFG at M = 2N x 11 (176 at N = 8), where cuBLAS bf16 already
reaches ~74 TFLOP/s: compute bound, weight-only FP8 does not help (keep it bf16;
faster FP8 math would need FP8 activations, which `fp8_weights` does not allow).

Whole MLPs against the VoxCPM2 run's fused bf16 MLP kernel: base LM decode MLP
92.1 -> 48.8 us (1.89x), LocDiT MLP 33.7 -> 22.8 us (1.48x), with an FP8 MLP
composed of the two examples and `silu * up` in torch; fuse the activation
into the gate / up epilogue (one launch) for the eager case.

The module evaluator times eager calls with a warm L2: weights that fit in L2
stay cached between its calls, so FP8 gains there are smaller than in the
model (the GEMV on [2048, 6144]: 1.3-1.7x, the skinny GEMM on `[2, 11, 1024]`:
1.4x, host overhead ~16-19 us per call included; the merged gate + up weight,
50 MB, measures 2.0x). A bf16 reference that only just fits in L2 can time
slower next to your FP8 weights than alone, and the reference-timing check then
reports an `integrity_violation` (rare): evaluate again before you suspect the
kernel. `pct_of_sol` of a `fp8_weights` target counts the weights at 1 byte (+
4 bytes of scale per channel). Judge FP8 by achieved GB/s, by `pct_of_sol` and
end to end.

## FP4 weights (`fp4_weights`)

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
the real judge.

**The contract** (as FP8, with the block scales):

* quantise once in `build()` with `quantize_fp4`, keep no bf16 copy;
* activations stay bf16 (FP4 activations are W4A4, not this class);
* fp32 accumulation: a partial sum per 16-weight block times its block scale
  (or the weights dequantised with their scale), the tensor scale and bias once
  per output, one rounding to bf16;
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
`sm_120a`) needs FP4 activations (W4A4).

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
