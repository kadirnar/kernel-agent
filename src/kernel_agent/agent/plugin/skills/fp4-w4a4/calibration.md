# W4A4 accuracy on real captures and the broken kernels the tiers reject

Every `nn.Linear` of a captured module replaced by the W4A4 reference math
(`kernels/quant.py`: `quantize_fp4` weights, `quantize_fp4_activations` per call,
`fp4_w4a4_linear` through `F.scaled_mm` NVFP4 on an RTX 5070 Ti), judged by the evaluator's
own comparisons on the captured inputs, 10 seeds of both redraw kinds per case and the x 3 /
x 0.01 / x -1 inputs, in `near-lossless-fp4a` and `relaxed-fp4a` at once
(`docs/research-scripts/w4a4-233`: `calibrate_w4a4.py`, raw results and `summary.md` in
`results/`). Measured on VoxCPM2 and Qwen3-0.6B captures; numbers of another model are
evidence from those.

| capture | numerics | captured: min cosine / max rel L2 / max norm change | near-lossless-fp4a fails (captured, redrawn, scaled) | relaxed-fp4a fails |
|---|---|---|---|---|
| LocDiT layer (M = 352, 176) | NVFP4 W4A4 | 0.9968 / 0.080 / 3.4 % | 0/2, 0/40, 0/6 | 0/2, 0/40, 0/6 |
| LocDiT layer | NVFP4, one outer scale per call | 0.9969 / 0.079 / 2.9 % | 0/2, 0/40, 0/6 | 0/2, 0/40, 0/6 |
| LocDiT layer | NVFP4 + Hadamard 16 | 0.9965 / 0.084 / 2.4 % | 0/2, 0/40, 0/6 | 0/2, 0/40, 0/6 |
| LocDiT layer | MXFP4 W4A4 | 0.9935 / 0.116 / **6.5 %** | 2/2, 0/40, 0/6 | 0/2, 0/40, 0/6 |
| LocEnc, 12 layers (M = 80) | NVFP4 W4A4 | 0.9929 / 0.119 / 0.2 % | 0/4, 0/80, 0/12 | 0/4, 0/80, 0/12 |
| LocEnc, 12 layers | MXFP4 W4A4 | 0.9811 / 0.194 / 0.6 % | 0/4, 0/80, 0/12 | 0/4, 0/80, 0/12 |
| base-LM decode layer (M = 1) | NVFP4 W4A4 | 0.9988 / 0.050 / 1.8 % | 0/3, 0/60, 0/9 | 0/3, 0/60, 0/9 |
| Qwen3-0.6B decode layer (M = 1) | NVFP4 W4A4 | 0.9839 / 0.179 / 2.3 % | 0/4, **4/80**, 0/12 | 0/4, 0/80, 0/12 |
| Qwen3-0.6B decode layer | MXFP4 W4A4 | 0.9718 / 0.236 / 4.1 % | 0/4, **32/80**, 1/12 | 0/4, 6/80, 1/12 |

The Qwen3 failures are its one-token output's cosine on redrawn inputs (0.893 < 0.90): W4A4
at M = 1 is memory bound and not what the class is for (`fp4_weights` is). On redrawn and
scaled inputs the LocDiT layer reaches 0.9885 / 0.151 / 3.2 %, the base-LM layer at x 0.01
0.9684 / 0.256 / 2.8 %.

**The norm shrink.** The LocDiT layer's hidden output keeps its direction (cosine 0.99994)
and loses 3.4 % of its norm. Per GEMM, quantising the activations alone shrinks the output by
up to about 1 % (gate_proj -0.96 % with unrounded fp32 block scales, -0.91 % with e4m3 ones:
the e2m1 grid, not the scale rounding; down_proj -1.05 %; q_proj +0.02 %:
`results/norm_shrink.out`); MLP gate x up then down compound it. Keeping gate / up in
FP8 W8A8: relative L2 0.036 -> 0.014, norm -3.4 % -> -0.9 %; o_proj and down_proj in FP8
instead changes less (0.024, -2.4 %; down_proj alone 0.026, -2.5 %).

Per `nn.Linear` alone (captured input; output relative L2 / norm change against the exact
GEMM; token crest of the input):

| LocDiT layer, M = 352 | NVFP4 | NVFP4 + H16 | MXFP4 | MXFP4 + H32 | FP8 W8A8 |
|---|---|---|---|---|---|
| q_proj (1024 -> 2048, crest 24) | 0.029 / -0.4 % | 0.035 / -1.7 % | 0.044 / -1.9 % | 0.040 / -1.8 % | 0.008 / -0.1 % |
| k_proj (1024 -> 256) | 0.048 / -0.5 % | 0.055 / -2.0 % | 0.071 / -1.8 % | 0.064 / -1.4 % | 0.012 / -0.2 % |
| v_proj (1024 -> 256) | 0.079 / +0.1 % | 0.084 / -1.4 % | 0.116 / -2.7 % | 0.105 / -0.7 % | 0.020 / -0.2 % |
| o_proj (2048 -> 1024, crest 23) | 0.018 / -0.5 % | 0.024 / -0.7 % | 0.039 / -0.1 % | 0.033 / -2.2 % | 0.006 / -0.3 % |
| gate_proj (1024 -> 4096, crest 29) | 0.088 / -1.1 % | 0.126 / +1.1 % | 0.133 / -2.6 % | 0.139 / -1.0 % | 0.019 / -0.2 % |
| up_proj (1024 -> 4096) | 0.086 / -1.0 % | 0.124 / +1.0 % | 0.129 / -1.3 % | 0.137 / -0.4 % | 0.018 / -0.1 % |
| down_proj (4096 -> 1024, crest 49) | 0.012 / -0.7 % | 0.026 / +2.0 % | 0.059 / **-5.6 %** | 0.051 / -4.7 % | 0.007 / +0.4 % |

NVFP4 is about 4-5x FP8 W8A8's error per GEMM; MXFP4 1.5x NVFP4's, and its power-of-two
scales lose the massive-activation channel's norm (down_proj -5.6 %). A Hadamard rotation of
16 makes NVFP4 worse here (its 16-element blocks already isolate outliers; rotated values are
more Gaussian, which e2m1's grid fits worse); on the LocEnc and the Qwen3 layer the same.

**Broken kernels** (rejected = at least one failing check):

| broken kernel | LocDiT layer | base-LM decode | Qwen3 decode | LocEnc stack |
|---|---|---|---|---|
| weight tensor scale x 1.05 | yes (captured norm 11 %) | yes | yes | no (final RMSNorm) |
| weight tensor scale x 1.2 | yes | yes | yes | no (final RMSNorm) |
| nibbles swapped | yes | yes | yes | yes |
| activation block scales shifted by one block | yes | yes | yes | yes |
| the first token's outer scale for every token | yes | (M = 1: not a bug) | yes | no |
| activation codes truncated (round toward zero) | yes | yes | yes | no (final RMSNorm) |
| activation scales cached from the first call | yes (redrawn 40/40) | yes | yes | near-lossless only |
| int4 per tensor | yes | yes | yes | yes |
| output row 0 of every GEMM never written | no | no | yes | no |
| KV head 0 dropped | yes | yes | yes | yes |
| q heads 0 / last swapped (layout) | no | no | yes | yes (x 0.01) |

An unwritten output channel stays within W4A4's noise (as with FP4 weights), and a module
that ends in a normalisation hides a scale error: the perceptual gate judges those end to
end. The tiers are calibrated for single layers and blocks; a deep stack accumulates more.

## Norm bias and its correction (opt-in `unbiased=True`)

Where the shrink comes from: e2m1's bins at 0 (everything below a quarter step), 2 and 4 round
more values down than up, so a quantised block keeps less of its component along itself
(`x^ ~ a x + e`, `a < 1`) while the noise `e` adds to its norm. A GEMM's output then has an
in-phase gain `<y^, y> / |y|^2` below 1, partly hidden in its norm by the noise.
`quant.fp4_bias_correction` returns `c = sum x^2 / sum x x^` per row (= `1 / a`): per token
for the activations (a factor of the epilogue's per-token scale), per output channel for the
weight (in its tensor scale, `quantize_fp4(w, unbiased=True)`). Codes, block scales and the
outer scale the guard checks do not change.

Measured on Qwen3-0.6B (every q / k / v / o / gate / up / down of its 28 layers on 334 tokens
of six texts, the GEMMs emulated from the codes on the CPU; `norm_bias.py`,
`results/norm_bias.out`); means over the 196 GEMMs, the MLP (gate / up, SiLU, down, every
GEMM quantised) median over the layers:

| variant | gain - 1 | norm | rel L2 | cosine | MLP gain - 1 / norm |
|---|---|---|---|---|---|
| NVFP4 (block maximum to 6) | -0.86 % | -0.20 % | 0.1121 | 0.99337 | -2.45 % / -0.26 % |
| adaptive (maximum to 4 or 6, lower block error), activations | -1.09 % | -0.46 % | 0.1094 | 0.99369 | -2.74 % / -0.70 % |
| adaptive, both operands | -1.00 % | -0.41 % | 0.1058 | 0.99410 | -2.41 % / -0.59 % |
| per-token factor (least squares `sum x x^ / sum x^^2`) | -1.34 % | -0.69 % | 0.1118 | 0.99340 | -3.80 % / -1.74 % |
| per-token factor (`|x| / |x^|`) | -0.94 % | -0.28 % | 0.1118 | 0.99340 | -2.71 % / -0.58 % |
| per-token factor (`sum x^2 / sum x x^`) | -0.54 % | +0.13 % | 0.1119 | 0.99340 | -1.60 % / +0.57 % |
| per-token and per-channel factors (`unbiased`) | -0.10 % | +0.57 % | 0.1122 | 0.99341 | -0.37 % / +1.86 % |
| adaptive (both) and per-token factor | -0.24 % | +0.35 % | 0.1059 | 0.99412 | -0.37 % / +1.57 % |
| adaptive (both) and both factors | +0.11 % | +0.70 % | 0.1061 | 0.99412 | +0.53 % / +2.57 % |

The adaptive rule (with the outer scale `amax / (4 x 448)`, so that a block mapped to 4 never
clamps at 448) lowers the error but deepens the shrink: it is not in the library. The factors
remove the shrink at no error cost; the norm then shows the noise (+rel L2^2 / 2: Qwen3's MLPs
at relative L2 ~0.2 go to +1.9 %). The last layer's MLP loses half its in-phase gain with
every variant (cosine 0.89): the sensitivity probe's 8-bit layers are the answer there.

When: a W4A4 target that fails its gate on the norm with a small error (the LocDiT layer:
relative L2 0.036, norm -3.4 %: bias, not noise; the correction is expected to take most of it,
not measured here), or a deep stack where the in-phase loss compounds. Not by default: where
the noise dominates the norm already sits near 1 and the correction lifts it above. On an A10
the producers' factor (`triton_fp4_producers.py`, `unbiased=True`) costs 5-35 % of the
producer and is within 4e-7 of the reference; the MLP module with it moves Qwen3-0.6B layers
5 / 14 / 20 from an in-phase gain of 0.977 / 0.972 / 0.980 to 0.995 / 0.995 / 0.998 (relative L2
0.211 -> 0.214). Both factors are epilogue scales: free in a custom epilogue (CuTe's `acc *
sx[m] * sw[n]`); `F.scaled_mm`'s two-level recipe takes tensor-wise second-level scales only
(fp32 output and the epilogue in torch there, as `fp4_w4a4_linear`).

## The scale-rule guard (`kernels/scale_guard.py`)

The hook `quantize_activations(x) -> (codes, scales, outer)`: codes packed two per byte
`[rows, K / 2]` (not read by the guard: the tiers judge them), scales **unswizzled** `[rows,
K / 16]` e4m3 (MXFP4: `[rows, K / 32]` e8m0; uint8 bits or values too; the swizzled 128 x 4
layout is refused with that reason), outer fp32 `[rows]` or one per call (MXFP4 may return
`(codes, scales)`). A quantiser that rotates first (`hadamard_rotate`) returns the rotated
activations as a fourth element: its scales are checked against them.

The evaluator runs it on the captured input and on a stress input of the same shape
(`quant.fp4_stress_input`): every row's maximum is 64 (per token and per call give the same
outer scale), the other blocks' scale before rounding sits at 1.1125 x 2^e (rounding to
nearest gives 1.125 x 2^e, truncation 2^e) and at (k + 0.7) x 2^-9 in e4m3's subnormal
range; MXFP4: block maxima at 1.9 x 2^e. It rejects the candidate (`stage: scale_rule`, with
the reason) when a block maximum exceeds 6 x scale x outer beyond the rounding of an e4m3
scale (`quant.fp4_scale_check`: 6.375 for a normal scale rounded to nearest; below 2^-6 the
reference's own subnormal rounding), when a non-zero block gets scale 0 where rounding to
nearest gives a subnormal one (a flush to zero), or when a scale is not finite. Caught (CPU
tests): block scales rounded down (6.75 x step with a normal scale, more with a subnormal
one), an outer scale 0.8x too small (block scales clamp at 448), subnormal scales flushed to
zero (on Gaussian rows only the stress input shows it), MXFP4's OCP rule
`2^(floor(log2 amax) - 2)` (up to 8 x scale). The reference's own saturation (up to 6.375 x
step, more with subnormal scales: `fp4_w4a4_error`'s `activation_saturation`) passes.
