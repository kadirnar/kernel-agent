# FP8 scaling recipes on sm_120 (measured, docs/FP8.md)

Part of the `fp8-w8a8` skill.

The `fp8_w8a8` contract above (per-channel weights, per-token activations, dynamic)
is unchanged; these are the measured alternatives (RTX 5070 Ti, cuBLASLt 13.1, torch
2.14; scripts in `docs/research-scripts/fp8-sm120/`).

* **Supported here**: cuBLASLt scale modes `SCALAR_32F` (tensor-wise, 8 algorithms
  per shape) and `VEC32_UE8M0` (MXFP8: e4m3 + one power-of-two ue8m0 scale per 32
  along K on both operands, applied by the tensor core; 1 algorithm per shape).
  `OUTER_VEC_32F`, `VEC128_32F`, `BLK128x128_32F` return
  `CUBLAS_STATUS_NOT_SUPPORTED`; `F.scaled_mm` with `BlockWise1x128` /
  `BlockWise128x128` (DeepSeek-style blockwise) raises "only supported in CUDA for
  SM90". MXFP8 through torch: `F.scaled_mm(x_q, w_q.t(), scale_a=sa, scale_recipe_a=
  ScalingType.BlockWise1x32, scale_b=sb, scale_recipe_b=ScalingType.BlockWise1x32,
  swizzle_a=SwizzleType.SWIZZLE_32_4_4, swizzle_b=SwizzleType.SWIZZLE_32_4_4,
  output_dtype=torch.bfloat16)` with `float8_e8m0fnu` scales `[rows, K / 32]`
  rearranged into 128 x 4 blocks (rows padded to 128); equals the fp32 math of the
  codes.
* **Speed at M = 352** (LocDiT, us, L2-cold graph): q|k|v 10.6 tensor-wise / **9.3
  MXFP8**; gate|up 29.5 / **24.7** (239 TFLOP/s); o_proj **8.8** (tensor-wise,
  2-batch split-K) / 14.8; down **15.2** / 26.6. MXFP8 has no split-K algorithm, so
  at N = 1024 it leaves SMs idle. Software blockwise (1x128 activations, 128x128
  weights, partial sums promoted every 128 K: CUTLASS SM120 example-87 kernel or
  Triton) is exact but runs on `QMMA.F32`: 15.9 / 15.7 / 40.8 / 28.0 us here.
* **MXFP8 scale rule**: use `2^ceil(log2(amax / 448))` per block. The OCP reference
  rule `2^(floor(log2 amax) - 8)` saturates block maxima above 448 x scale; on
  VoxCPM2's massive activations it fails the near-lossless tier (LocDiT o_proj /
  down_proj norm off by 4.0 % / 2.0 %). With the ceil rule MXFP8 has W8A8's error
  (LocDiT layer relative L2 0.0205 vs 0.0204 per token; passes captured and redrawn
  inputs). MXFP8 targets are their own class, `fp8_mx` (skill `mxfp8`), whose
  evaluations reject a saturating rule.
* **Where finer activation scales matter**: not at M = 352 (tensor-wise, per-token,
  1x128, blockwise and MXFP8 all give relative L2 0.0195-0.0205 on the LocDiT layer).
  At decode (M = 1) the LM q_proj fails the tier with per-token or tensor-wise scales
  (0.037, norm off by 2.4 %) and passes with 1x128 groups (0.018) or MXFP8 (0.023);
  decode is memory bound, so FP8 weight-only (0.017) is the right class there anyway.
* **Static (calibrated) activation scales**: calibrated on one text, a per-tensor
  scale saturates in up to 8.8 % of another text's calls (activation error up to
  0.33 vs 0.03 dynamic) and a layer with static scales fails the redrawn-input check
  in 1 of 10 LM decode draws (SmoothQuant + static: 3 of 10). It saves <= 0.3 us per
  producer call. Keep dynamic scales; group scales (1x32, 1x128) are the local
  alternative when a producer cannot see the whole row.
* **Quantising in the producer** costs little: RMSNorm [352, 1024] 1.35 us (bf16
  out) -> 1.48 us (e4m3 + per-token scale) vs 2.67 us with a separate quantisation
  pass; silu(gate) * up [352, 4096] 3.34 -> 3.03 us (half the bytes written) vs 5.50
  separate. Applying the tensor-wise GEMM's s_x * s_w in that kernel costs +2.7 us
  (5.74 us); an MXFP8 GEMM's output is already scaled. Example:
  `examples/triton_fp8_producers.py` (`rmsnorm_fp8`, `silu_mul_fp8`: e4m3 + per-token
  scales, or MXFP8 1 x 32 ue8m0 scales with `fp8_mx`'s rule, row-major for
  `tl.dot_scaled` / blocked for cuBLASLt; a gated MLP whose SiLU-mul feeds `down_proj` in
  e4m3). Never a separate
  quantisation pass after a producer you control.
* **Host time per eager call**: `torch._scaled_mm` 18-20 us, `F.scaled_mm` MXFP8 27,
  `at::_scaled_mm` from C++ 19, a direct `cublasLtMatmul` with cached descriptors and
  algorithm 6.1 (5.5 each for several behind one C++ call). Under CUDA graphs it does
  not count.
* **FP8 attention / KV cache**: e4m3 K/V (per-tensor or per-token scales) or FP8
  QK^T / PV change a VoxCPM2 layer's output by <= 0.004 relative L2, but there is no
  speed to win: the LocDiT attention covers 11 tokens (latency bound) and the LM's
  KV cache holds <= ~77 tokens (2.6 % of a batch-16 decode step's bytes); a decode
  kernel over e4m3 K/V is slower than bf16 at 77 tokens and 1.3x faster only from
  ~2k tokens. A static K/V scale from one text is exceeded by another in ~50 % of
  the layers. FlashAttention-3's FP8 path is Hopper only. For long contexts: the
  opt-in `fp8_kv` class (skill `fp8-kv-cache`).
