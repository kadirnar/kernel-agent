---
name: int8-weights
description: INT8 weight-only kernels (precision int8_weights) — symmetric int8 codes per output channel, bf16 or fp16 activations, exact in-register conversion without e4m3 emulation, the decode GEMV, accuracy against FP8 weights, measured numbers. Use for memory-bound decode GEMVs and skinny GEMMs whose spec says int8_weights, above all on GPUs before sm_89.
---

# INT8 weight-only (`precision: int8_weights`)

Read the `precision-tiers` skill first (when it is allowed, the tolerance tiers, the int8 format and its uniform step). INT8 W8A8 (activations in int8 too, IMMA tensor cores): the `int8-w8a8` skill.

For a target whose spec says `"precision": "int8_weights"`: the same memory-bound targets as
`fp8_weights` (decode GEMVs, skinny GEMMs: the weight bytes set the floor), with the same bytes
(one per weight) and the same tier. Prefer it on GPUs without hardware e4m3 conversion (before
sm_89: e4m3 codes convert in software there, int8 codes in two cheap ops); elsewhere it is the
more accurate 8-bit weight format on rows without outliers, at the same speed.

## The contract

* **Quantise once, in `build()`** (`kernel_agent.kernels.quant.quantize_int8`): symmetric int8
  codes in [-127, 127] (-128 unused), one fp32 scale per output channel, `scale = amax(|row|) *
  (1 / 127)` (the fp32 constant `quant.INT8_STEP`), codes `round(w / scale)` to nearest even;
  keep no bf16 copy of a quantised weight.
* **Activations stay in the model's dtype** (bf16 or fp16; the example is a template on
  it). Never quantise them in an `int8_weights` target (that is `int8_w8a8`).
* **Accumulate in fp32**; apply the scale (and the bias) once per output in the epilogue,
  `y[m, n] = scale[n] * sum_k x[m, k] q[n, k] + bias[n]`, and round to that dtype once.
* **Report** `int8_error(weight, q, scale)` (`rel_l2`, `worst_channel_rel_l2`, `underflow`,
  `crest`: a row with an outlier loses its small weights to the uniform step) and the
  evaluator's per-case `min_cosine` / `max_rel_l2` in `NOTES.md`.
* **Reference / fallback**: `int8_weights_linear(x, q, scale, bias)` (fp32 math), or the
  dequantised weight (`dequantize_int8`) through cuBLAS.

## Convert in registers

* `I2F` runs at quarter rate on sm_80+; build the fp32 value instead: for byte `j` of a word
  `w` of four codes, `__int_as_float(__byte_perm(w ^ 0x80808080u, 0x4B000000u, 0x7440u | j)) -
  8388736.0f` (the byte biased to `code + 128` in the mantissa of 2^23, minus `2^23 + 128`)
  gives the code exactly: one `prmt` and one FADD per weight, no e4m3 conversion (which before
  sm_89 is a software routine).
* For the bf16 / fp16 tensor cores the codes are exact in both formats (|code| <= 127).
* GEMV (M <= 4): warp per output row, 128-bit loads of 16 codes streamed past L1
  (`ld.global.nc.L1::no_allocate`), each weight converted once and reused by every activation
  row, warp-shuffle reduction, the scale in the epilogue: `examples/cuda_int8_gemv.py`.
* More rows: the bf16 skinny-GEMM design of the `fp8-weights` skill with int8 codes, or (when
  the activations may be int8) the IMMA skinny GEMM of the `int8-w8a8` skill, which needs no
  conversion at all.

## Accuracy and speed (measured)

* Accuracy against the bf16 module (reference math on real captures, the evaluator's captured,
  redrawn and scaled checks): VoxCPM2 LocDiT layer relative L2 0.0054 (FP8 weights 0.0151),
  base-LM decode layer 0.0016-0.0020 (0.0034-0.0043), Qwen3-0.6B MLP at decode 0.014 (0.038);
  0.0025-0.0085 per `nn.Linear` (FP8 weights up to 0.026). Every check passes in both the
  near-lossless and the relaxed tier; weight scales x 1.05, a neighbour channel's scale and a
  zeroed output channel fail (`tests/test_perturbed_calibration.py`).
* RTX 5070 Ti, weights streamed from DRAM (`docs/research-scripts/int8-178`): `cuda_int8_gemv.py`
  [1, 2048] x [2048, 12288] in 31.5 us (799 GB/s; FP8 GEMV 31.2, cuBLAS bf16 63.1), [1, 6144] x
  [6144, 2048] 16.8 us (751 GB/s; 16.4 / 31.5); output relative L2 0.0085-0.0092 against bf16
  (FP8 weights 0.026). The module evaluator (eager, warm L2): 1.87x at M = 1.

## Examples and sources

* Examples: `cuda_int8_gemv.py` (weight-only decode GEMV); for more rows, `cuda_fp8_skinny_gemm.py` (the bf16 tensor-core design to adapt) or `cuda_int8_skinny_gemm.py` (INT8 W8A8). All in kernel-agent's examples directory (`kernel_agent/agent/examples/`; a session's prompt gives the directory): copy their structure; an example's `ARCHS` names the GPUs it runs on.
* Sources: the `documentation-sources` skill's `sources.md`, sections "Low precision (formats, scaling, accuracy)"; "CUDA C++ and PTX".
* Code: `kernel_agent.kernels.quant` (`quantize_int8`, `dequantize_int8`, `int8_weights_linear`, `int8_error`).
