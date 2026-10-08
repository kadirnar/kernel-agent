# INT8 accuracy on real captures and the broken variants the tiers reject

Every `nn.Linear` of a captured module replaced by the reference math (`kernels/quant.py`, on
the GPU through `torch._int_mm`), judged by the evaluator's own comparisons: the captured
inputs, then its redrawn (normal + mixed) draws and its scaled (x 3, x 0.01, x -1) inputs with
the redrawn bounds (`docs/research-scripts/int8-178`: `calib_int8.py`, raw output in
`results/`). Both INT8 classes share the 8-bit tiers (`near-lossless`, `relaxed`): no bounds of
their own.

| module (rows per call) | precision | rel L2 | min cosine | norm | redrawn fail | scaled |
|---|---|---|---|---|---|---|
| LocDiT decoder layer (352) | INT8 weights | 0.0054 | 0.99999 | 0.07 % | 0 / 30 | pass |
| LocDiT decoder layer (352) | INT8 W8A8 | 0.0229 | 0.99984 | **2.10 %** | 0 / 30 | **x 0.01 fails** (norm 4.0 %) |
| LocDiT decoder layer (352) | INT8 W8A8 + SmoothQuant alpha 0.4 | 0.0082 | 0.99997 | 0.24 % | 0 / 60 | pass |
| LocDiT decoder layer (352) | INT8 W8A8 + SmoothQuant alpha 0.5 / 0.6 / 0.7 | 0.008-0.010 | 0.99997 | < 1 % | 1 / 10 / 60 of 60 | x 0.01 fails from 0.6 |
| LocDiT decoder layer (352) | FP8 W8A8 / FP8 weights | 0.0204 / 0.0151 | 0.99979 | 0.21 % | 0 / 30 | pass |
| LocDiT decoder layer (176) | INT8 W8A8 | 0.0207 | 0.99984 | 1.89 % | 0 / 30 | x 0.01 fails (norm 4.1 %) |
| base-LM decode layer (1, KV cache; 3 steps) | INT8 W8A8 / INT8 weights | <= 0.0070 / 0.0020 | 0.99998 | 0.02 % | 0 / 36 | pass |
| Qwen3-0.6B MLP, decode (1) | INT8 W8A8 / INT8 weights | 0.060 / 0.014 | 0.99823 / 0.9999 | 0.86 % | 0 / 24 | pass |
| Qwen3-0.6B MLP, decode (1) | FP8 W8A8 / FP8 weights | 0.052 / 0.038 | 0.99867 | 0.57 % | 0 / 24 | pass |
| Qwen3-0.6B MLP, decode (1) | INT8 W8A8 + SmoothQuant from one decode token | 0.44 | 0.906 | 3.4 % | 12 / 12 | fails |

The verdicts above are the near-lossless tier's. In the **relaxed** tier (#175:
`results/calib_*_relaxed.out`) every correct INT8 recipe passes every check of these
captures, plain per-token INT8 W8A8 on the LocDiT layer too (norm 2.1 % captured and 4.0 % at
x 0.01, within ±4 % / ±6 %); every broken variant below still fails at least one capture (a
static calibrated activation scale passes the base-LM decode layer there and fails the LocDiT
and Qwen3 ones).

Per `nn.Linear` of the LocDiT layer (relative L2, captured input; token crest): q / k / v /
o_proj (crest 23-24) 0.007 / 0.011 / 0.018 / 0.007 with INT8 W8A8 against 0.008 / 0.012 /
0.020 / 0.007 FP8 W8A8; gate / up_proj (crest 29) 0.059 vs 0.019, down_proj (crest 49) 0.022
and norm -2.0 % (fails) vs 0.007. INT8 weight-only: 0.0025-0.0085 per Linear (FP8 weights: up
to 0.026). So: INT8 weights are the most accurate 8-bit weight format; INT8 W8A8 equals FP8
W8A8 on activations without outlier channels and needs SmoothQuant (alpha ~0.4, factors from
many tokens) or FP8 / bf16 where they have them in a near-lossless run.

Broken INT8 kernels fail: weight scales x 1.05 (norm +4 to +15 %), a neighbour channel's
scale, the first token's scale, a zeroed output channel, a static calibrated activation
scale, activations clipped at their 99.9th percentile (relative L2 0.13 on the LocDiT layer),
and a per-tensor activation scale (passes the captured LocDiT inputs at 0.041, fails the x
0.01 check: use per-token scales). At one row per call (decode) the first token's scale and a
per-tensor scale are the per-token math, and output channel 0 of the base-LM layer is small
(a zeroed channel hides in the noise there, as with FP8). The synthetic CPU fixture of
`tests/test_perturbed_calibration.py` runs both INT8 classes, these broken variants and #175's
blatant bugs (int4 per tensor, scales x 1.2, an unwritten row, gate / up swapped) in both
modes.
