---
name: fp8-w8a8
description: FP8 W8A8 kernels (precision fp8_w8a8) for compute-bound GEMMs — the contract with dynamic per-token activation scales, accuracy and outliers, torch._scaled_mm and custom e4m3 GEMMs, the FP8 peak and host time on sm_120, measured scaling recipes. Use when a target's spec says fp8_w8a8.
---

# FP8 W8A8 (`precision: fp8_w8a8`): compute-bound GEMMs

Read the `precision-tiers` skill first. GEMM paths (`torch._scaled_mm`, custom e4m3 GEMMs, the FP8 peak and host time on sm_120): [gemm-paths.md](gemm-paths.md); measured scaling recipes on sm_120 (cuBLASLt scale modes, MXFP8, static scales, quantising in the producer): [scaling-recipes-sm120.md](scaling-recipes-sm120.md). Calling cuBLASLt from C++ and the FP8 instruction per GPU: the `cuda-kernels` skill.

For a target whose spec says `"precision": "fp8_w8a8"`: GEMMs with hundreds of
rows per call, where bf16 tensor cores are the limit and the weight bytes are
not (e.g. the VoxCPM2 LocDiT at batch 16 under CFG: M = 2 x 16 x 11 = 352, 80
TFLOP per run, cuBLAS bf16 at ~70 % of its 99 TFLOP/s). FP8 tensor cores run
e4m3 x e4m3 at two to three times the bf16 rate, but both operands must be FP8
(peaks: "The FP8 peak on sm_120" in `gemm-paths.md`). The
recipe below is the one the systems agent of the VoxCPM2 throughput run found
(`runs/openbmb--VoxCPM2/20261006-004718`, transforms `fp8_locdit_mlp`,
`fp8_locdit_attn_proj`, `triton_fp8_locdit_gemm`; ledger exp 40-47), verified
in `examples/triton_fp8_w8a8_gemm.py`.

The contract:

* **Weights** as for `fp8_weights`: quantised once in `build()` (`quantize_fp8`),
  e4m3, one fp32 scale per output channel, no bf16 copy kept.
* **Activations per token, every call**: `scale = amax(|row|) / 448` of each row
  of `x.reshape(-1, K)`, codes `round(x / scale)` in e4m3 (round to nearest
  even, clamp to ±448 first). `quantize_fp8_activations` is the reference math.
  Never a static (offline-calibrated) or per-tensor scale: one outlier token
  would set the scale of every token. Fuse the quantisation into the op that
  produces `x` (RMSNorm, `silu(gate) * up`, the attention output's transpose)
  or into the GEMM's prologue; a separate pass reads and writes `x` once more
  (the run let Inductor fuse it into the producers).
* **Math**: e4m3 x e4m3 products, fp32 accumulation, epilogue
  `y[m, n] = sx[m] * sw[n] * acc[m, n] + bias[n]` in fp32, one rounding to the model's
  dtype (bf16 or fp16: the examples take it from the input).
  Norms, softmax / attention math, RoPE and residual adds stay as in eager.
* **Report** `fp8_w8a8_error(weight, q, scale, x)` on captured activations
  (weight error, `activation_rel_l2`, `activation_crest`, the output's relative
  L2 / cosine / norm ratio) and the evaluator's per-case `min_cosine` /
  `max_rel_l2` in `NOTES.md`.
* **Fallback / reference**: `fp8_w8a8_linear(x, q, scale, bias)`
  (`torch._scaled_mm` where it applies, else the same math in fp32).

Accuracy against the bf16 module, every `nn.Linear` W8A8 (fake quant with the
numerics above, on real VoxCPM2 capture inputs; on the GPU, `_scaled_mm` and the
example give the same numbers for the LocDiT layer; the near-lossless tier allows
cosine >= 0.996, relative L2 <= 0.08, norm ±2 %, element ratio <= 1):

| module (rows per call) | rel L2 | min cosine | norm change | element ratio | tier |
|---|---|---|---|---|---|
| LocDiT decoder layer (352; outputs: hidden, k, v) | 0.020 | 0.99979 | 0.24 % | 0.19 | pass |
| LocDiT MLP alone (352) | 0.009 | 0.99997 | 0.25 % | 0.15 | pass |
| LocDiT q_proj alone (352) | 0.008 | 0.99994 | 0.15 % | 0.11 | pass |
| LocDiT decoder layer (22, batch 1) | 0.021 | 0.99979 | 0.24 % | 0.16 | pass |
| LM decoder layer, decode (1) | 0.006 | 0.99998 | 0.03 % | 0.04 | pass |
| LM MLP alone, decode (1) | 0.014 | 0.99990 | 0.37 % | 0.11 | pass |
| LM q_proj alone, decode (1; activation crest ~30) | 0.041 | 0.99917 | 2.4 % | 0.38 | **fail** (norm) |

FP8 weight-only on the same modules: 0.015 (LocDiT layer), 0.007 (MLP), 0.006
(q_proj). Broken scales fail every module: weight scales x 1.05 (norm +5 %), a
neighbour channel's scale (relative L2 0.89), the first token's scale for every
token (relative L2 0.96). Per-tensor activation scales happen to pass on these
inputs (0.021) but break on an outlier token: use per-token scales.

**Outliers.** A token's crest factor (amax / RMS) decides how many e4m3 steps
the bulk of its row gets: 23-50 on the LocDiT GEMM inputs, ~30 on the LM's
attention input; a per-token quantised activation has relative L2 0.015-0.03.
Where an output fails the tier (the LM q_proj above: memory bound at decode
anyway) keep that GEMM weight-only (`fp8_weights`) or bf16. If a compute-bound
GEMM fails, move a per-input-channel factor from the activations into the
weights (SmoothQuant) before quantising; not needed on VoxCPM2.

**Redrawn inputs.** In the VoxCPM2 run the first W8A8 LocDiT layer kernels
(`dit_layer__fp8_w8a8`, exp 100-101) passed the captured cases and failed the
perturbed-input check on the element bound alone (element ratio 1.42 and 1.65 at
cosine 0.9999, relative L2 0.012); the engineer spent two slices hunting a GEMM.
The reference math (`fp8_w8a8_linear` on every Linear) failed it as well, in 98
of 200 draws: the error sat in channel 497, the massive-activation channel, whose
values are small on redrawn inputs. The check now scales the element bound per
channel and the same kernels pass (README "Quality modes"). Per-token scales are
still checked there: activation scales cached from the first call pass the
captured cases and fail the redrawn ones.

## Examples and sources

* Examples: `triton_fp8_w8a8_gemm.py`, `cuda_cublaslt_fp8.py`, `triton_fp8_producers.py`, `cute_fp8_blockscaled_gemm.py`, `cute_fp8_decoder_block.py`. All in kernel-agent's examples directory (`kernel_agent/agent/examples/`; a session's prompt gives the directory): copy their structure; an example's `ARCHS` names the GPUs it runs on.
* Sources: the `documentation-sources` skill's `sources.md`, sections "Low precision (formats, scaling, accuracy)"; "cuBLASLt (FP8 / FP4 GEMMs, epilogues, heuristics)"; "Triton".
* Code: `kernel_agent.kernels.quant` (`quantize_fp8`, `quantize_fp8_activations`, `fp8_w8a8_linear`, `fp8_w8a8_error`).
