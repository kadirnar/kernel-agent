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
