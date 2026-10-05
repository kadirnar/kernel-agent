# Low-precision weights (FP8, FP4)

Verified examples: `examples/cuda_fp8_gemv.py` (decode GEMV, M <= 4) and
`examples/cuda_fp8_skinny_gemm.py` (bf16 tensor cores, M <= 32; groups of 32
tokens beyond, slower than cuBLAS bf16 from M ~ 64). Helpers:
`kernel_agent.kernels.quant` (`quantize_fp8`, `dequantize_fp8`, `fp8_error`).

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
of the elements outside the bf16 tolerance).

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

## FP4 / NVFP4: when it is worth it

Not yet: no `nvfp4_weights` precision exists. FP4 halves the bytes again, and
sm_120 has the tensor cores for it (block-scaled MMA, `sm_120a`), but weight-only
NVFP4 has ~0.10 relative L2 error on Gaussian rows (MXFP4 0.12; FP8 0.026),
above the module tier's 0.08: it would need per-layer selection (FP4 only where
the perceptual gate shows headroom, FP8 or bf16 elsewhere) and recalibration.
Consider it only for large DRAM-bound GEMVs that already run FP8 at >= 80 % of
their speed of light.
