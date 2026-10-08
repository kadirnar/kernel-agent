---
name: precision-tiers
description: Precision classes (fp8_weights, fp8_w8a8, fp8_mx, fp8_kv, fp4_weights, int8_weights, int8_w8a8, reduced) and when a run allows them, the near-lossless and relaxed tolerance tiers and redrawn-input checks, FP8 / FP4 / INT8 formats, scale granularity and outliers. Use when planning a precision or before any low-precision kernel.
---

# Low-precision weights (FP8, FP4, INT8): precision classes and tolerance tiers

Verified examples: `examples/cuda_fp8_gemv.py` (decode GEMV, M <= 4) and
`examples/cuda_fp8_skinny_gemm.py` (bf16 tensor cores, M <= 32; groups of 32
tokens beyond, slower than cuBLAS bf16 from M ~ 64) for `fp8_weights`; FP4:
`examples/cuda_fp4_gemv.py` (NVFP4 decode GEMV, M <= 4; skill `fp4-weights`);
W8A8: `examples/triton_fp8_w8a8_gemm.py` (e4m3 tensor cores, compute-bound
GEMMs; skill `fp8-w8a8`); MXFP8 W8A8 (`fp8_mx`): `examples/triton_mxfp8_gemm.py`
(skill `mxfp8`). Helpers: `kernel_agent.kernels.quant` (`quantize_fp8`,
`dequantize_fp8`, `fp8_error`; `quantize_fp4`, `dequantize_fp4`, `fp4_error`;
`quantize_fp8_activations`, `fp8_w8a8_linear`, `fp8_w8a8_error`; `quantize_mxfp8`,
`mxfp8_linear`, `mxfp8_error`). FP8 toolkit (written from measured research code, not
yet run on a GPU): `cuda_cublaslt_fp8.py` (direct cuBLASLt FP8 GEMMs, tensor-wise and
MXFP8), `triton_fp8_producers.py` (RMSNorm / SiLU-mul emitting e4m3 + scales),
`triton_fp8_kv_decode.py` (split-KV decode attention over an e4m3 KV cache, `fp8_kv`;
helpers in `kernel_agent.kernels.kv_quant`: skill `fp8-kv-cache`). INT8 (the 8-bit classes
of GPUs without FP8 tensor cores, and an option elsewhere): `examples/cuda_int8_gemv.py`
(weight-only decode GEMV; skill `int8-weights`), `examples/triton_int8_w8a8_gemm.py` (IMMA
GEMM, compute bound) and `examples/cuda_int8_skinny_gemm.py` (IMMA skinny GEMM, decode; skill
`int8-w8a8`); helpers `quantize_int8`, `quantize_int8_activations`, `int8_matmul`,
`int8_w8a8_linear`, `int8_weights_linear`, `int8_error`, `int8_w8a8_error`,
`smoothquant_factors`.

One skill per precision class (load the one of your target's `precision`):

| precision | skill | for |
|---|---|---|
| `fp8_weights` | `fp8-weights` | memory-bound decode GEMVs and skinny GEMMs: e4m3 weights, bf16 activations |
| `fp8_w8a8` | `fp8-w8a8` | compute-bound GEMMs: e4m3 weights and per-token activations on the FP8 tensor cores |
| `fp8_mx` | `mxfp8` | compute-bound GEMMs with wide outputs on block-scaled tensor cores (sm_100 / sm_120) |
| `fp8_kv` | `fp8-kv-cache` | decode attention over long KV caches (opt-in) |
| `fp4_weights` | `fp4-weights` | memory-bound decode GEMVs where FP8 weights are not enough (NVFP4) |
| `int8_weights` | `int8-weights` | memory-bound decode GEMVs and skinny GEMMs: int8 weights, bf16 activations (no e4m3 conversion: GPUs before sm_89) |
| `int8_w8a8` | `int8-w8a8` | compute-bound GEMMs on the IMMA tensor cores: int8 weights and per-token activations (the 8-bit compute class of Turing / Ampere) |

## When it is allowed

Only for a target whose spec says `"precision": "fp8_weights"` (or `"fp8_w8a8"`,
`"fp8_mx"`, `"fp8_kv"`, `"int8_weights"`, `"int8_w8a8"`: their skills), which the planner
may set in a `--quality near-lossless` or `--quality relaxed` run (an exact run
refuses such targets; relaxed, the default of new runs, has the same tiers with
about twice the error budgets: **relaxed**, cosine >= 0.99, relative L2 error <=
0.16, norm within ±4 %, every element within 0.75 x RMS + 0.125 x |reference|;
**relaxed-fp4** and **relaxed-kv** likewise; your prompt states the bounds of the
tier your target was captured in). In near-lossless the target is captured in
the **near-lossless tolerance tier**
(`kernels/compare.py`): instead of per-element (atol, rtol), every output tensor
needs cosine >= 0.996, relative L2 error <= 0.08, its norm within ±2 % of the
reference (rounding noise is unbiased, a wrong scale is not) and every element
within 0.5 x RMS + 0.125 x |reference|. End to end, the run's perceptual gate
(WER, speaker similarity, MOS for TTS) decides. Without that spec field the
exact tier applies and FP8 weights fail it (~2.6 % relative L2 per GEMM, ~20 %
of the elements outside the bf16 tolerance). `"precision": "fp4_weights"`
(block-scaled FP4 weights) has its own tier, **near-lossless-fp4**: cosine >=
0.96, relative L2 error <= 0.28, norm within ±4 %, every element within 1.25 x
RMS + 0.25 x |reference| (skill `fp4-weights`); FP4 fails the FP8 tier.
After timing the evaluator checks again on inputs redrawn from each tensor's own
mean and std (the perturbed-input check). They have no outlier channels, so a
weight row that writes a massive activation (VoxCPM2's LocDiT o_proj / down_proj
row 497, 10x the median row norm) carries 10x the rounding error into a channel
whose values are now small: there the RMS of the element bound is the larger of
the tensor's and the element's channel's (last dimension), and the tiers have
wider bounds (`compare.PERTURBED_BOUNDS`; near-lossless: every element within
0.75 x RMS + 0.125 x |reference|). The reference math of every precision passes
both checks; if only the perturbed check fails, run `fp8_w8a8_linear` /
`dequantize_fp8` on the same redrawn inputs before you hunt the kernel. The same
check also runs the captured inputs x 3, x 0.01 and x -1 (`scaled_x3`, ...):
activation scales must follow the input; a scale calibrated once saturates at x 3.

## Formats

| format | torch dtype | bits (s/e/m) | max | step near 1 | use |
|---|---|---|---|---|---|
| e4m3 ("fn": no inf, one NaN) | `torch.float8_e4m3fn` | 1/4/3 | 448 | 2^-3 | weights |
| e5m2 | `torch.float8_e5m2` | 1/5/2 | 57344 | 2^-2 | gradients; too coarse for weights |
| e2m1 (FP4) | `torch.float4_e2m1fn_x2` (2 per byte) | 1/2/1 | 6 | 0.5 | with block scales only |
| int8 (symmetric, -128 unused) | `torch.int8` | integer | 127 | uniform: `amax / 127` | weights (`int8_weights`), weights and per-token activations (`int8_w8a8`) |

e4m3 values: normals 2^-6 .. 448 with 3 mantissa bits, subnormals down to 2^-9.
Per-channel e4m3 on Gaussian-like rows: weight (and GEMM output) relative L2
error ~0.026, cosine ~0.9997. Errors add up over layers: a VoxCPM2-sized MLP
(gate, up and down in FP8) is ~0.046 against the bf16 MLP. FP4 needs block scales: NVFP4 =
e2m1 + an e4m3 scale per 16 elements (+ an fp32 tensor scale), MXFP4 = e2m1 +
a power-of-two (e8m0) scale per 32.

int8's step is uniform: the error per value is about `amax / (127 x √12)`, i.e. `crest /
440` of the row's RMS. On Gaussian-like rows (crest ~4) that is ~0.9 % (e4m3: ~2.6 %, a
relative step): INT8 weight-only is the most accurate 8-bit weight format. At crest 30 it is
~7 % and the values below half a step (`amax / 254`) become 0: **a biased error** (the
products shrink), where e4m3 keeps its relative step whatever the crest. So activation
outlier channels cost INT8 W8A8 more than FP8 W8A8 (the `int8-w8a8` skill: SmoothQuant).

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

## Examples and sources

* Sources: the `documentation-sources` skill's `sources.md`, sections "Low precision (formats, scaling, accuracy)".
* Examples: `cuda_int8_gemv.py`, `triton_int8_w8a8_gemm.py`, `cuda_int8_skinny_gemm.py` (INT8); the other precisions' in their skills.
* Code: `kernel_agent.kernels.quant`, `kernel_agent.kernels.kv_quant`, `kernel_agent.kernels.compare` (`PRECISIONS`, `NEAR_LOSSLESS_BOUNDS`, `PERTURBED_BOUNDS`); README "Quality modes".
