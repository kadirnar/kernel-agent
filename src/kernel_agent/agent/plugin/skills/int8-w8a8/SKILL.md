---
name: int8-w8a8
description: INT8 W8A8 kernels (precision int8_w8a8) for compute-bound GEMMs on the IMMA tensor cores — the 8-bit compute class of GPUs without FP8 (Turing, Ampere) and an option elsewhere. Dynamic per-token scales, exact int32 accumulation, mma.sync s8 fragments, outliers and SmoothQuant, the path per GPU family. Use when a target's spec says int8_w8a8.
---

# INT8 W8A8 (`precision: int8_w8a8`): compute-bound GEMMs on IMMA

Read the `precision-tiers` skill first (when it is allowed, the tolerance tiers, the int8 format and its uniform step). Accuracy on real captures and the broken variants the tiers reject: [calibration.md](calibration.md); measured rates, peaks and example timings on sm_120: [measured.md](measured.md). INT8 weight-only: the `int8-weights` skill; FP8 W8A8: the `fp8-w8a8` skill.

For a target whose spec says `"precision": "int8_w8a8"`: GEMMs with ~64+ rows per call where
the bf16 tensor cores are the limit. It is the 8-bit compute class of GPUs without FP8 tensor
cores (Turing sm_75, Ampere sm_80 / sm_86: there `fp8_w8a8` / `fp8_mx` are refused), and an
option on every other GPU: INT8 where its measured floor (the *Ceilings* table's *INT8 W8A8*
column) is at or below the *W8A8* one and the GEMMs' input activations have no outlier
channels; FP8 where they do, or where the INT8 peak is the lower one (Blackwell Ultra, sm_103).

## The contract

* **Weights** once in `build()` (`kernel_agent.kernels.quant.quantize_int8`): int8 codes in
  [-127, 127], one fp32 scale per output channel, no bf16 copy kept.
* **Activations per token, every call** (`quantize_int8_activations`): `scale = amax(|row|) *
  (1 / 127)` (a product with the fp32 constant `quant.INT8_STEP`: kernels write `amax *
  (1.0f / 127.0f)`; torch divides by a Python scalar through its reciprocal on the GPU only),
  codes `round(x / scale)` to nearest even with an IEEE division (`__fdiv_rn` + `rintf` in
  CUDA, `tl.div_rn` + `libdevice.rint` in Triton), clamped to ±127: then a kernel's codes equal
  the reference's bit for bit. Never static, calibrated, cached or per-tensor scales (the
  evaluator's scaled x 3 / x 0.01 and redrawn checks fail them). One pass per row (amax, then
  codes) in its own kernel, or fused into the producer (RMSNorm, `silu(gate) * up`).
* **Math**: s8 x s8 products on the tensor cores with an **int32** accumulator: Triton
  `tl.dot(a, b, acc, out_dtype=tl.int32)` on int8 tiles (sm_80+), `mma.sync.aligned.m16n8k32
  .row.col.s32.s8.s8.s32` (sm_80+; Turing: `m8n8k16`), cuBLASLt through `torch._int_mm(a [M,
  K], w.t())` (M > 16, K and N multiples of 8: `int8_matmul` pads M). The sums are exact (no
  fp32 rounding, no split-K order to match); the epilogue `acc * x_scale[m] * w_scale[n] (+
  bias[n])` in fp32 without FMA contraction (`__fmul_rn`, `__fadd_rn`) and one rounding to
  bf16 reproduce `int8_w8a8_linear` bit for bit (both examples do). Norms, softmax / attention
  math and residual adds stay as in eager.
* **Report** `int8_w8a8_error(weight, q, scale, x)` on captured activations: `activation_crest`
  (the largest amax / RMS of a token) and `activation_underflow` say whether the activations
  suit INT8; and the evaluator's per-case `min_cosine` / `max_rel_l2` in `NOTES.md`.
* **Reference / fallback**: `int8_w8a8_linear(x, q, scale, bias, smooth=None)` (`torch._int_mm`
  on the GPU, exact chunked fp32 elsewhere).

## Fragments without conversion (`mma.sync` m16n8k32, 8-bit operands)

A (16 x 32, row) holds per thread rows g / g + 8 (g = lane / 4) at k 4t .. 4t + 3 and 16 + 4t
.. (t = lane % 4); B (32 x 8, col) k 4t .. 4t + 3 and 16 + 4t .. of column g; C as for fp32. A
dot product does not care about the order of k: let thread t load 16 consecutive bytes (one
128-bit load) of its weight rows and of its token and feed bytes 8j .. 8j + 3 / 8j + 4 .. 8j +
7 as the two halves of product j. The weight is the A operand straight from global memory,
the tokens the B operand: no conversion and no shared memory for the operands, and the
cross-warp reduction stays exact in int32 (`examples/cuda_int8_skinny_gemm.py`, decode and up
to 32 rows per weight read; groups of 32 beyond).

## Outliers and SmoothQuant

int8's step is uniform (`amax / 127`): at token crest 30 the bulk keeps ~4 steps per RMS and
the values below half a step become 0, a biased error (the products shrink), where e4m3 keeps
its relative step. On VoxCPM2's LocDiT MLP (crest 29-49) per-token INT8 W8A8 moves the layer's
norm by -2.1 %: near-lossless (±2 %) rejects it, relaxed (±4 %) accepts it
([calibration.md](calibration.md)). Where it must fit, move the range into the weight:
`s = smoothquant_factors(amax_per_input_channel, weight, alpha)` (`amax` over many captured
tokens: `x.reshape(-1, K).abs().amax(0)`), quantise `weight * s` (`quantize_int8(weight, s)`)
and `x / s` (`quantize_int8_activations(x, s)`, or fold `1 / s` into the producer's weight,
e.g. the RMSNorm's); `int8_w8a8_linear(..., smooth=s)` is the reference. The factors are
static, so the redrawn and scaled checks judge them: start at alpha 0.4 and keep the smallest
alpha that passes the captured cases (larger ones, or factors from a single decode token, fit
the captured outliers and fail the scaled or redrawn inputs: on the VoxCPM2 LocDiT layer
alpha >= 0.6 fails x 0.01). Or keep those GEMMs in FP8 / bf16 inside
the target.

## The path per GPU family

The toolchain block's measured peaks decide (`int8` in the GPU peaks: `torch._int_mm`, TOPS;
`s8 IMMA.S32`: the `mma.sync` rate); documented rates otherwise:

| GPU | INT8 tensor-core path | rate |
|---|---|---|
| Turing sm_75 (T4, RTX 20xx) | `mma.sync` m8n8k16 s8, cuBLASLt (`torch._int_mm`); Triton `tl.dot` has no tensor cores below sm_80 | 2x fp16 |
| Ampere sm_80 / sm_86 | `mma.sync` m16n8k32 s8 (IMMA), Triton `tl.dot` on int8, cuBLASLt | 2x bf16 (A100: 624 vs 312 TOPS); GeForce RTX 30xx 4x its fp32-accumulating bf16 (RTX 3090: 284 vs 71) |
| Ada sm_89 | as Ampere; FP8 also | = FP8 with fp16 accumulation; GeForce RTX 40xx 2x FP8 with fp32 accumulation (RTX 4090: 661 vs 330) |
| Hopper sm_90 | `wgmma` s8 (Triton `tl.dot`, CUTLASS sm_90); `mma.sync` stops below it | = FP8 (2x bf16) |
| Blackwell sm_100 (B200) | `tcgen05.mma kind::i8` (cuBLASLt, CUTLASS / CuTe DSL) | = FP8 |
| Blackwell Ultra sm_103 (B300) | warp-level IMMA only: no `tcgen05.mma kind::i8` in the PTX ISA, no CUTLASS INT8 kernels (arXiv 2608.11693) | **~1/30 of FP8** (NVIDIA's B300 specification): prefer FP8 |
| GeForce Blackwell sm_120 | `mma.sync` m16n8k32 s8 at full rate ([measured.md](measured.md)) | 410 TOPS = block-scaled FP8 |

## Examples and sources

* Examples: `triton_int8_w8a8_gemm.py` (one-pass per-token quantisation kernel + `tl.dot` on int8 tiles with an int32 accumulator, scales and bias in the epilogue, per-shape tiles, a `custom_op`), `cuda_int8_skinny_gemm.py` (IMMA skinny GEMM for decode). All in kernel-agent's examples directory (`kernel_agent/agent/examples/`; a session's prompt gives the directory): copy their structure; an example's `ARCHS` names the GPUs it runs on.
* Sources: the `documentation-sources` skill's `sources.md`, sections "Low precision (formats, scaling, accuracy)" (SmoothQuant, LLM.int8(), INT8 on Blackwell Ultra); "CUDA C++ and PTX"; "Triton". Per family: the `gpu-architectures` skill.
* Code: `kernel_agent.kernels.quant` (`quantize_int8`, `quantize_int8_activations`, `int8_matmul`, `int8_w8a8_linear`, `int8_w8a8_error`, `smoothquant_factors`); research scripts and raw results: `docs/research-scripts/int8-178`.
