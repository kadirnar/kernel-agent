# FP8 beyond weight-only and row-wise W8A8 on sm_120 (issue #132)

What the five FP8 techniques of #132 can and cannot do on the RTX 5070 Ti (GeForce
Blackwell, sm_120) at VoxCPM2's shapes, measured here, and the PRs that follow from
it. The user's direction (#131, #133, #134, #136): no 4-bit; push FP8 further; CUDA
C++ / CuTe DSL first-class. This is the research write-up the issue asks for before
any library code; nothing under `src/` changes except the two knowledge files
(`knowledge/low_precision.md`, `knowledge/sources.md`). Compiled 2026-10-07.

Every number below is either measured for this note (method in §1, scripts and raw
outputs in [`docs/research-scripts/fp8-sm120/`](research-scripts/fp8-sm120/)) or
quoted with its source: a URL, a file of our runs (`runs/openbmb--VoxCPM2/…`, read
only), or [RESEARCH-TRITON.md](RESEARCH-TRITON.md) (#135, the sm_120 instruction
rates, which this note builds on rather than re-measures). Run names as there: **R2**
= `20261005-192504` (1 request, 522.2 ms, 10.49x eager), **R3** =
`20261006-004718-retest2` (batch 16, 925 ms per batched run of 153.6 audio s = 6.02 ms
per audio second, 6.24x eager).

## 0. Summary

1. **"Blockwise" on sm_120 means MXFP8.** The fine-grained scaling that runs at full
   rate here is the block-scaled tensor-core instruction (`mma.sync
   kind::mxf8f6f4.block_scale`, e4m3 with one power-of-two ue8m0 scale per 32
   elements along K on *both* operands; 416 TFLOP/s instruction rate vs 208 for
   plain e4m3 with fp32 accumulation, RESEARCH-TRITON §1.1). cuBLASLt runs it
   (`VEC32_UE8M0`): **316–320 TFLOP/s at 8192³ (tensor-wise: 329), and at the
   LocDiT's M = 352 faster than tensor-wise on the wide GEMMs** (gate|up 24.7 vs
   29.5 us, q|k|v 9.3 vs 10.6), slower on the N = 1024 ones (o_proj 14.8 vs 8.8,
   down 26.6 vs 15.2 us: one algorithm, no split-K).
2. **DeepSeek-style blockwise (1×128 activations, 128×128 weights) is not available
   at speed on sm_120.** cuBLASLt 13.1 refuses `VEC128_32F`, `BLK128x128_32F` and
   `OUTER_VEC_32F` here (`CUBLAS_STATUS_NOT_SUPPORTED`); torch 2.14 raises "only
   supported in CUDA for SM90". Software promotion (CUTLASS's SM120 blockwise
   kernel, example 87 style, built here; or Triton) works but runs on the half-rate
   instruction: **188 TFLOP/s at 8192³, 94–145 at M = 352**: 100 us per LocDiT layer
   against 64 us for the GEMMs in production.
3. **Accuracy does not need finer scales on the compute-bound GEMMs.** On the
   captured LocDiT layer (M = 352) every recipe lands at relative L2 ≈ 0.020 and
   passes the near-lossless tier on captured and 30 redrawn inputs (tensor-wise,
   row-wise, 1×128, blockwise, MXFP8 with the scale rule of item 4). Finer scales
   matter at **decode (M = 1)**: the LM `q_proj` fails the tier with per-token (and tensor-wise) scales (norm off
   by 2.4 %) and passes with 1×128 groups (0.018) or MXFP8 (0.023) — where W8A8 has
   no speed to offer (memory bound).
4. **The MXFP8 quantiser must not use the OCP reference rule.** Its scale
   `2^(floor(log2 amax) − 8)` saturates block maxima between 448 and 512 × scale;
   on VoxCPM2's massive activations that fails the tier (LocDiT o_proj / down: norm
   off by 4.0 % / 2.0 %). The non-saturating rule `2^ceil(log2(amax / 448))` passes
   everywhere with W8A8's error.
5. **FP8 activations through fused layers pay only in the glue.** Quantising inside
   the producer costs +0.1 us (RMSNorm, M = 352) instead of +1.3 us for a separate
   pass, and silu·up → e4m3 is cheaper than silu·up → bf16. Fusing the quantisation
   into the *GEMM's* epilogue loses on this GPU today: our Triton gate|up GEMM with a
   silu·up + 1×128 e4m3 epilogue takes 44.6 us against 35.6 us for cuBLASLt +
   a glue kernel and 28.1 us for cuBLASLt MXFP8 + glue, because every
   epilogue-capable GEMM we could build runs on the plain FP8 instruction (CUTLASS
   SM120 dense FP8: 192 TFLOP/s at 8192³, under its 208 cap).
6. **Static (calibrated) activation scales: no.** Calibrated on one text and tested
   on another, a per-tensor static scale saturates in up to 8.8 % of calls (worst
   activation error 0.33 vs 0.03 dynamic), and a layer with static scales fails the
   evaluator's redrawn-input check in 1 of 10 LM draws (SmoothQuant + static: 3 of
   10 LM, 6 of 30 LocDiT). The saving it buys is ≤ 0.3 us per producer call.
   SmoothQuant with dynamic per-token scales is harmless and unnecessary here.
7. **FP8 attention and FP8 KV cache do not pay on VoxCPM2.** The LocDiT attention
   covers 11 tokens (4.7 us per call at 7 TFLOP/s: latency, not math), the LM's KV
   cache holds 47–77 tokens (2.6 % of the decode bytes at batch 16, 0.16 % at batch
   1), and a decode kernel reading e4m3 K/V is slower than bf16 at 77 tokens (3.07
   vs 2.75 us) and 1.3x faster only from 2048 tokens. Accuracy is not the issue: e4m3
   K/V or FP8 QKᵀ/PV change a layer's output by ≤ 0.004 relative L2.
8. **Host overhead** per eager call: `torch._scaled_mm` 18–20 us, `F.scaled_mm`
   (MXFP8) 27 us, `at::_scaled_mm` from C++ 19 us, a cached direct cuBLASLt call
   6.1 us (5.5 us each when four run behind one binding), the bundled Triton
   W8A8 launcher 44 us (69 through `torch.library`). Inside CUDA graphs none of it
   counts.
9. **Expected end-to-end gains** (§8): batch 16, −40 ms per batched run from MXFP8 on
   the wide LocDiT GEMMs and −15 to −18 ms from the simpler glue once their output
   arrives scaled: ≈ −55 ms of 925 (−6 %, ~5.7 ms per audio second). Single-request
   latency (522 ms): ≈ 0 from all five techniques, because every GEMM there runs at
   M ≤ 22 where FP8 weight-only storage already sets the byte floor.

## 1. Setup and method

RTX 5070 Ti (70 SMs, 48 MB L2, DRAM 768 GB/s copy peak, `~/.cache/kernel-agent/peaks-*`),
torch 2.14.1+cu130, cuBLASLt 13.1.1 (`cublasLtGetVersion` 130101), Triton 3.8.0, nvcc
13.4 (kernel-agent's toolchain shim), CUTLASS 4.1 headers from the tilelang wheel
(`tilelang/3rdparty/cutlass`, the "local sources" of `knowledge/sources.md`). Every GPU
job ran under the GPU lock after `kernel_agent.kernels.bench.warm_gpu()` /
`ensure_clocks()`.

* **GEMM timing** (`gemm_bench.py`, `cutlass_bench.py`, `fusion_bench.py`): a CUDA
  graph of back-to-back calls rotating over enough weight copies (≥ 160 MB) to keep
  them out of L2, as in the model; minimum over 7 replays of 3 passes. Activations
  are already quantised unless a row says otherwise. Shapes are VoxCPM2's
  (`config.json`; LocDiT / LocEnc hidden 1024, FFN 4096, 16 heads × 128, 2 KV heads,
  merged q|k|v = 2560; LM hidden 2048, FFN 6144): LocDiT at M = 352 (batch 16, CFG:
  `targets/dit_layer__fp8_w8a8/spec.json` of R3) and M = 22 (batch 1), LocEnc at M =
  80, LM decode at M = 16 / 1, LM prefill at M = 272.
* **Accuracy** (`accuracy.py`, `fp8lib.py`): fake quantisation in fp32 (the reference
  math of each recipe), judged with the evaluator's own code: `compare_tensors` in the
  `near-lossless` tier on the captured inputs, and on inputs redrawn by
  `kernels.verify.perturb_` (20 normal + 10 mixed draws per LocDiT case, 6 + 4 per LM
  case) with the redrawn bounds of #109 (`compare.PERTURBED_BOUNDS`). Inputs and
  weights are the captures of R3's `dit_layer__fp8_w8a8` (LocDiT layer 0, `[32, 11,
  1024]`, plus a `[16, 11, 1024]` case from another run used for calibration) and R2's
  `lm_step_fp8` (base-LM layer 0, `forward_step` at decode steps 1 / 31 / 60 with its
  KV cache).
* **Model-wide statistics** (`calib.py`): VoxCPM2 loaded with hooks on every
  `nn.Linear` with K a multiple of 128; 30 patches of the workload's text (A) for
  calibration, 30 patches of its `natural_text` (B) for testing, every 3rd call.
* **Host time** (`host_peak.py`): wall time per call of 3000 back-to-back eager calls
  on a 16×256×256 GEMM (a few us of GPU work), median of 5 rounds.

## 2. What sm_120 supports

| feature | status on sm_120 | source |
|---|---|---|
| `mma.sync` e4m3 / e5m2, fp32 accumulate (`QMMA.F32`) | yes, **208 TFLOP/s** | [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) `mma` ("requires sm_89 or higher"); rate: RESEARCH-TRITON §1.1 |
| `mma.sync kind::mxf8f6f4.block_scale` (e4m3 + ue8m0 per 32, `QMMA.SF`) | yes (`sm_120a`), **416 TFLOP/s** with fp32 accumulation | PTX ISA: ".kind, .block_scale … requires sm_120a"; RESEARCH-TRITON §1.1 |
| e4m3 with fp16 accumulate | 416 TFLOP/s (fp16 partial sums; no W8A8 recipe here uses them) | RESEARCH-TRITON §1.1 |
| `wgmma`, `tcgen05` / TMEM, 2-SM MMA | no | CUTLASS SM120 docs (mma.sync only); RESEARCH-TRITON §1.2 |
| TMA (`cp.async.bulk.tensor`) | yes; **no multicast, cluster 1×1×1**; TN layout only for SM120 GEMMs | [CUTLASS blackwell_functionality.md](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md) |
| cuBLASLt `SCALAR_32F` (tensor-wise) | yes, 8 algorithms per shape | measured (`probe_lt.py`) |
| cuBLASLt `VEC32_UE8M0` (MXFP8) | **yes, 1 algorithm per shape** (no split-K variants) | measured; cuBLAS docs list it for CC 10.0+ |
| cuBLASLt `OUTER_VEC_32F`, `VEC128_32F`, `BLK128x128_32F` | **no** (heuristic: `NOT_SUPPORTED`); mixed scalar / outer-vector: `INVALID_VALUE` | measured; cuBLAS docs: CC 9.0 only |
| `torch._scaled_mm` / `F.scaled_mm` tensor-wise, row-wise | yes (row-wise = PyTorch's CUTLASS 3.x kernel on `QMMA.F32`) | measured; RESEARCH-TRITON §1.2 |
| `F.scaled_mm` `BlockWise1x128` / `BlockWise128x128` | **no**: `NotImplementedError: DeepSeek style (1x128, 128x128) scaling only supported in CUDA for SM90` | measured (`probe_support.py`); [ScaledBlas.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ScaledBlas.cpp) `_check_deepseek_support` |
| `F.scaled_mm` `BlockWise1x32` + `SWIZZLE_32_4_4` (MXFP8) | **yes**, equals the fp32 math of the codes (rel. error 1.7e-3 = bf16 output rounding) | measured |
| CUTLASS SM120 blockwise collective (`KernelTmaWarpSpecializedBlockwise*Sm120`, `Sm120BlockwiseScaleConfig<1,128,128>`) | builds and is exact (local CUTLASS 4.1); `QMMA.F32` speed | measured (`cutlass_bw.py`) |
| CUTLASS SM120 block-scaled (`OpClassBlockScaledTensorOp`, `mx_float8_t<e4m3>`) | 128×128×128 builds, `initialize()` returned `Error Internal`; 64-wide tiles fail TMA static asserts — not resolved here | measured (`cutlass_mx_bench.out`) |
| Triton 3.8 `tl.dot_scaled(e4m3, ue8m0)` | native `QMMA.SF` (unit or real MX scales) | [triton #7918](https://github.com/triton-lang/triton/pull/7918); RESEARCH-TRITON §1.2; measured |
| FlashAttention-3 FP8 | Hopper only (`sm_90a`) | [flash-attention README](https://github.com/Dao-AILab/flash-attention); `hopper/flash_api.cpp` |
| FlashAttention-4 (CuTe) | sm_120 forward with `mma.sync` m16n8k16 (bf16 / fp16); FP8 "only supported on SM100" | [`flash_fwd_sm120.py`](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm120.py), `interface.py` |
| FlashInfer FP8 KV cache | sm_120 listed ("RTX 50 series"); FP8 K/V (e4m3 / e5m2) with one scalar `k_scale` / `v_scale` | [README](https://github.com/flashinfer-ai/flashinfer), [`decode.py`](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/decode.py) |
| SageAttention 2 | sm_120 supported: INT8 QKᵀ (per-warp) + FP8 PV with fp32 accumulation ("sm120 has accurate fp32 accumulator for fp8 mma") | [SageAttention](https://github.com/thu-ml/SageAttention) `setup.py`, `core.py`; [arXiv 2411.10958](https://arxiv.org/abs/2411.10958) |
| SageAttention 3 | FP4 (microscaling) attention on RTX 5090 — 4-bit, out of scope (#131) | [arXiv 2505.11594](https://arxiv.org/abs/2505.11594) |

Two corrections to sources this note relied on. The CUTLASS SM120 page lists
`kind::f8f6f4` and `kind::mxf8f6f4.block_scale` at the same rate ("1x Ada Fp8 Tensor
Core (2x for FP32 accumulator)"); measured, only the block-scaled form runs at the
doubled rate with fp32 accumulation (208 vs 416, RESEARCH-TRITON §1.1).
SageAttention 2's paper describes an FP22 accumulator for `mma(f32.f8.f8.f32)` "on the
Ada and Hopper architecture"; its code treats sm_120's as fp32.

Dense peaks at 8192³ (`host_peak.py`, TFLOP/s): bf16 cuBLAS 99.5 · tensor-wise
cuBLASLt 328.8 · **MXFP8 cuBLASLt 316.5** (320.0 in a second run) · row-wise
`_scaled_mm` 194.1 · CUTLASS SM120 dense (TMA, cooperative 128×128×128) 192.3 ·
CUTLASS SM120 blockwise 187.9 (cooperative) / 117.7 (ping-pong 64×128) · Triton
`tl.dot` row-wise 186.7 · Triton `tl.dot_scaled` MXFP8 215.3 · Triton blockwise
(software promotion) 152.2. Everything on `QMMA.F32` stops below its 208 cap; only
cuBLASLt's kernels get near the 416 instruction rate (nvjet for tensor-wise scales is
inferred to use `QMMA.SF` with unit scales, RESEARCH-TRITON §1.2).

## 3. Technique 1: blockwise-scaled FP8 GEMMs

**How it works.** A scale per *block* instead of per tensor / row lets an outlier
spoil only its block. DeepSeek-V3's recipe
([arXiv 2412.19437](https://arxiv.org/abs/2412.19437) §3.3): activations scaled per
1×128 tile, weights per 128×128 block, both computed online (amax of the block), and
because the scale changes along K the partial sums are promoted to fp32 registers on
CUDA cores every N_C = 128 elements (4 WGMMAs on H800). CUTLASS example
[87](https://github.com/NVIDIA/cutlass/tree/main/examples/87_blackwell_geforce_gemm_blockwise)
ports this to sm_120 (cooperative 128×128×128 / ping-pong 64×128×128, scale
granularity M = 1, N = K = 128). MXFP8 ([OCP MX
v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf))
moves the block scale into the instruction: e4m3 elements with one E8M0 (power of two)
scale per 32 consecutive K elements, on both operands; the tensor core applies them, so
there is no promotion step.

### 3.1 Speed at the LocDiT shapes (M = 352, batch 16)

us per call (TFLOP/s), L2-cold, CUDA graph. Tensor-wise kernels compute the unscaled
product (scales applied by the consumer, as R3's `dit_layer__fp8_w8a8` does);
"split-K 2" = cuBLASLt batched over two K halves, the consumer adds the two bf16
partials (R3 slice 33).

| recipe / kernel | q\|k\|v 352×2560×1024 | o_proj 352×1024×2048 | gate\|up 352×8192×1024 | down 352×1024×4096 | layer (4 GEMMs) |
|---|---|---|---|---|---|
| bf16 cuBLAS | 25.6 | 22.5 | 69.0 | 38.9 | 156.0 |
| tensor-wise cuBLASLt, best of 8 algos | 10.6 (174) | 10.1 (146) | 29.5 (200) | 18.3 (161) | 68.5 |
| … split-K 2 for N = 1024 (R3 production) | | **8.8** (168) | | **15.2** (195) | 64.1 |
| row-wise `_scaled_mm` (CUTLASS, `QMMA.F32`) | 15.7 | 15.4 | 41.5 | 26.9 | 99.5 |
| **MXFP8 cuBLASLt** (`VEC32_UE8M0`) | **9.3** (199) | 14.8 (100) | **24.7** (239) | 26.6 (111) | 75.4 |
| Triton row-wise `tl.dot` (example configs) | 14.0 | 14.0 | 35.3 | 25.9 | 89.2 |
| Triton MXFP8 `tl.dot_scaled` (real-scale loads, pointers) | 12.6 | 12.8 | 34.6 | 25.8 | 85.8 |
| Triton blockwise 1×128 / 128×128 (promotion per 128) | 16.0 | 15.1 | 40.2 | 28.5 | 99.8 |
| CUTLASS SM120 blockwise (best of coop 128² / ping-pong 64×128) | 15.9 (116) | 15.7 (94) | 40.8 (145) | 28.0 (105) | 100.4 |
| CUTLASS SM120 dense FP8, scalar scale (best tile) | 15.1 | 15.2 | 39.1 | 27.1 | 96.5 |
| **mix: MXFP8 q\|k\|v + gate\|up, tensor-wise split-K o + down** | 9.3 | 8.8 | 24.7 | 15.2 | **58.0** |

* The blockwise promotion costs 8–13 % on top of the same kernel without it (Triton
  row-wise vs Triton blockwise with the same tiles: o_proj, gate|up, down; results
  `gemm_dit352_tuned.out`), and the blockwise kernels run on the plain FP8 MMA (their
  8192³ peaks, 152–188 TFLOP/s, sit under its 208 cap): 94–145 TFLOP/s here against
  nvjet's 146–200.
* MXFP8 is the fastest FP8 GEMM we have on the wide shapes (239 TFLOP/s on gate|up,
  72 % of the GEMM peak), but cuBLASLt offers one algorithm and no split-K for it; at
  N = 1024 its large tiles leave SMs idle (24 output tiles for 70 SMs) and it loses to
  tensor-wise split-K by 6–11 us. A split-K or small-tile MXFP8 kernel (Triton
  `tl.dot_scaled` + split-K, or CuTe) is a follow-up (§9).
* R3's production layer runs the tensor-wise column with split-K: its nvjet kernels
  total 418 ms of 906 ms GPU time per batched run (`rounds/3/profile/summary.md`:
  346.0 + 57.8 + 14.6 ms, 26664 calls) ≈ 64.5 us per LocDiT layer call (6480 per
  run, 4 GEMMs each), matching the 64.1 us above.

Other M (best of each family, us; full tables in `results/gemm_*.out`):

| shape | bf16 | tensor-wise cuBLASLt | MXFP8 cuBLASLt | best Triton (row-wise tuned / blockwise / MXFP8) | note |
|---|---|---|---|---|---|
| LocDiT gate\|up, M = 22 (batch 1) | 21.4 | 12.0 | 13.6 | 11.6 (724 GB/s) | memory bound: the FP8 weight-only skinny example streams the same bytes (6.9 us per 4096-wide half, 610 GB/s; low_precision.md) |
| LocDiT down, M = 22 | 13.2 | 9.2 | 26.3 | 7.8 | weight-only skinny: 7.6 us; MXFP8: 1 algo, idle SMs |
| LocEnc gate\|up, M = 80 | 26.0 | 13.3 | 13.2 | 14.5 | |
| LM gate\|up decode, M = 16 | 65.8 | 31.9 (788 GB/s) | 38.1 | 31.2 | DRAM bound; R3 runs FP8 weight-only skinny GEMMs here |
| LM down decode, M = 1 | 31.7 | 20.5 | 38.5 | — | weight-only GEMV: 16.5 us (low_precision.md) |
| LM prefill gate\|up, M = 272 | 172.9 | 68.7 | **68.0** (201) | 85.5 | |
| LM prefill down, M = 272 | 90.1 | **38.1** (180) | 38.8 | 57.0 | CUTLASS blockwise 73.4 |

### 3.2 Accuracy

LocDiT layer 0 at M = 352 (outputs hidden, k, v; worst of the three), every
`nn.Linear` with the recipe, attention and norms as in bf16:

| recipe (weights × activations) | captured: rel L2 / element ratio / norm | redrawn (30): fails, max rel L2, max element ratio |
|---|---|---|
| fp32 math, no quantisation (noise floor) | 0.0033 / 0.06 / 0.10 % | 0 |
| FP8 weights only (per channel) | 0.0151 / 0.14 / 0.22 % | 0, 0.027, 0.38 |
| tensor-wise × tensor-wise | 0.0205 / 0.25 / 0.46 % | 0, 0.038, 0.44 |
| per channel × per token (today's `fp8_w8a8`) | 0.0204 / 0.19 / 0.21 % | 0, 0.038, 0.38 |
| per channel × 1×128 | 0.0203 / 0.17 / 0.11 % | 0, 0.038, 0.49 |
| blockwise 128×128 × 1×128 | 0.0195 / 0.18 / 0.66 % | 0, 0.038, 0.53 |
| MXFP8, OCP floor scale | **0.0399 / 0.41 / 3.90 % — fails** | 0, 0.044, 0.50 |
| MXFP8, ceil (non-saturating) scale | 0.0205 / 0.17 / 0.92 % | 0, 0.038, 0.44 |

Each `nn.Linear` alone (captured input, relative L2 against its bf16 output; **fail**
= outside the near-lossless tier):

| linear (rows) | FP8 w only | tensor-wise | per token | 1×128 | blockwise | MXFP8 floor | MXFP8 ceil |
|---|---|---|---|---|---|---|---|
| LocDiT q_proj (352) | 0.006 | 0.009 | 0.008 | 0.008 | 0.008 | 0.013 | 0.008 |
| LocDiT o_proj (352) | 0.005 | 0.009 | 0.007 | 0.006 | 0.006 | **0.041 fail** (norm off 4.0 %) | 0.007 |
| LocDiT gate_proj (352) | 0.015 | 0.020 | 0.019 | 0.019 | 0.018 | 0.027 | 0.023 |
| LocDiT down_proj (352) | 0.006 | 0.008 | 0.007 | 0.005 | 0.006 | **0.021 fail** (norm off 2.0 %) | 0.007 |
| LM q_proj, decode (1) | 0.017 | **0.036 fail** (2.3 %) | **0.037 fail** (2.4 %) | 0.018 | 0.019 | **0.037 fail** (2.0 %) | 0.023 |
| LM k_proj, decode (1) | 0.012 | 0.026 | 0.021 | 0.013 | 0.019 | **0.038 fail** (2.2 %) | 0.026 |
| LM gate_proj, decode (1) | 0.017 | 0.024 | 0.024 | 0.023 | 0.024 | 0.028 | 0.025 |

(k/v/up/o/down rows behave like their neighbours: `results/acc_all.out`.) At layer
level the LM decode layer passes with every recipe except static ones (§6): the
residual dominates its output.

* The OCP rule ("largest power of two ≤ amax, divided by the largest power of two of
  the element type", §6.3 of the MX spec, which clamps what overflows) maps a block
  maximum in [256, 512) × scale onto e4m3's [256, 448]: maxima above 448 × scale
  saturate. VoxCPM2's massive activations (LocDiT `down_proj` input up to 1053× the
  median channel, base-LM `down_proj` 6345×, §6) hit that and shrink by up to 12.5 %.
  The ceil rule gives up one exponent step of range instead: error like per-token.
* Where accuracy needs finer scales (LM `q_proj` at decode: input crest ~30), 1×128
  groups fix it; but M = 1 is memory bound and FP8 weight-only (0.017) is both more
  accurate and as fast.

### 3.3 When it pays, how to check it, what it implies

* **Pays**: compute-bound GEMMs (M ≳ 130, the bf16 ridge on this GPU) with wide N
  (≥ 2560 at M = 352): MXFP8 on cuBLASLt beats tensor-wise by 12–16 % and carries
  its scales inside the GEMM. Not at N = 1024 (until a split-K MXFP8 kernel exists),
  not at M ≤ 80 (memory bound: same bytes as weight-only FP8), never
  DeepSeek-style promotion on sm_120 (half rate).
* **Check**: a verified example (`selftest`) whose output equals the fp32 math of the
  MX-dequantised operands to bf16 rounding; reference math `quantize_mxfp8` (ceil
  rule) next to `quantize_fp8_activations`; the near-lossless tier unchanged (the
  ceil rule passes captured and redrawn inputs above), plus a selftest that the
  floor rule fails VoxCPM2's o_proj (guards the quantiser); perceptual gate end to
  end as for every W8A8 change.
* **Precision class**: stays `fp8_w8a8` (8-bit weights and activations; MXFP8 is a
  scale layout, not a new tier): allowed by default in near-lossless runs (#131), no
  4-bit. Ceilings (#90): the W8A8 column's 330 TFLOP/s peak holds for MXFP8 (316–320
  measured); kernels on `QMMA.F32` (row-wise CUTLASS, Triton `tl.dot`,
  DeepSeek-style blockwise) are capped at 208 — the table should say which (§9).
  *Implemented in #144 as its own class*, `fp8_mx` (same near-lossless tier, 8-bit:
  allowed by default), so that the planner policy, an *MXFP8* ceilings column at the
  measured MXFP8 peak and the evaluator's scale-rule guard can address it
  (README, "MXFP8 W8A8").

## 4. Technique 2: FP8 activations kept between fused ops

**How it works.** A W8A8 GEMM needs its input in e4m3 plus scales. Quantising in a
separate pass reads and writes the activation once more and costs a launch; doing it
in the producer (RMSNorm, silu(gate)·up, the attention output) writes half the bytes
instead of more. The catch is the scale: a per-token scale needs the amax of the whole
row (K = 1024–4096 values), which a GEMM epilogue tile or a per-head attention CTA
does not see. Per-group scales (1×32 MXFP8, 1×128) and static scales are local, so
any tile can emit them.

**Measured** (`fusion_bench.py`, M = 352, us per call, CUDA graph):

| producer | bf16 out | bf16 out + separate per-token quant | fused per-token e4m3 | fused static e4m3 | fused MXFP8 (1×32) |
|---|---|---|---|---|---|
| RMSNorm [352, 1024] | 1.35 | 2.67 | **1.48** | 1.27 | 1.54 |
| silu(g)·u [352, 8192] → [352, 4096] | 3.34 | 5.50 | **3.03** | 2.71 | 3.35 |
| same, input = unscaled tensor-wise GEMM output (× s_x · s_w first) | | | 5.74 | | |

gate|up GEMM + silu·up + e4m3 for `down_proj` (M = 352, 20 L2-cold weights):

| chain | us |
|---|---|
| cuBLASLt tensor-wise (bf16, unscaled) + one kernel: s_x·s_w, silu·up, per-token e4m3 (R3's recipe) | 35.6 (GEMM alone 29.5) |
| **cuBLASLt MXFP8 (scales in the GEMM) + silu·up → MXFP8 kernel** | **28.1** |
| Triton GEMM (e4m3 `tl.dot`, interleaved gate/up rows) with silu·up epilogue, bf16 h | 45.6 |
| same with silu·up + 1×128 e4m3 quantisation in the epilogue (one kernel; best of 6 tiles) | 44.6 |

* Fused producer quantisation is close to free (+0.13 us on RMSNorm against +1.3 us
  for a separate pass); R3 already does it for norm1 / norm2 and silu·up (its
  `quant_k<1,256>`: 35.7 ms per batched run, 5.35 us per call ≈ the 5.74 us row).
* The expensive part of R3's glue is applying s_x·s_w to the unscaled tensor-wise
  output (+2.7 us per call). MXFP8 removes it: the GEMM output is already scaled.
* Fusing into the GEMM epilogue needs a GEMM that is both fast and ours. The ones we
  could build here (Triton `tl.dot`, CUTLASS SM120 dense or blockwise) stay under the
  plain FP8 MMA's 208 TFLOP/s; the fused epilogue saves the 6.1 us glue kernel, but
  the fused Triton kernel (BN = 256 tiles, so that one tile holds 128 matching gate
  and up columns) is 9 us slower than the tensor-wise chain and 16.5 us slower than
  the MXFP8 one. The way out is an epilogue-capable kernel on the block-scaled
  instruction (CuTe DSL `MmaMXF8Op` / CUTLASS `OpClassBlockScaledTensorOp`; our
  CUTLASS 4.1 build of it did not initialise, §2) — RESEARCH-TRITON §5.3 issue 2
  and #133.
* The attention-output quantisation (R3: `quant_k<0,128>`, 10.6 ms per batched run)
  needs a per-token amax across 16 heads = 8 CTAs; R3 slices 33–38 fused it per
  K-half of o_proj's split-K and measured no gain (`targets/dit_layer__fp8_w8a8/NOTES.md`).
  With 1×32 / 1×128 scales it becomes per-head local, but o_proj would then need a
  group-scaled GEMM, and the only full-rate one (MXFP8 cuBLASLt) is 6 us slower at
  N = 1024.

**When it pays**: always fuse the quantisation into an elementwise / row-reduction
producer (norm, silu·up) — a separate pass is never right. Fuse into a GEMM epilogue
only with a full-rate (block-scaled) GEMM. **Check**: same tier as `fp8_w8a8` (the
numerics are identical to quantising in the next GEMM's prologue); the evaluator's
module check sees the fused producer + GEMM as one candidate. **Class / ceilings**:
`fp8_w8a8`; activation I/O at one byte would halve the I/O term of the W8A8 floor,
which is small next to FLOPs at M = 352.

## 5. Technique 3: FP8 attention and an FP8 KV cache

**How it works.** FlashAttention-3's FP8 forward
([arXiv 2407.08608](https://arxiv.org/abs/2407.08608)) quantises Q, K, V per block
(one scale per Br×d / Bc×d tile), runs QKᵀ and P·V on FP8 WGMMA (V transposed:
"only the k-major format is supported"), and reduces outlier error with incoherent
processing (random ±1 diagonal × Hadamard on Q and K); Hopper only. SageAttention 2
(INT8 QKᵀ with K smoothing, FP8 P·V) is the sm_120 option. An FP8 KV cache stores K/V
in e4m3 with scales per tensor (vLLM: "a single scale is applied for each Q, K, and V
tensor", default 1.0 without calibration,
[quantized_kvcache.md](https://github.com/vllm-project/vllm/blob/main/docs/features/quantization/quantized_kvcache.md);
FlashInfer: scalar `k_scale` / `v_scale`) and halves the bytes a decode step reads.

**Accuracy** (emulated in fp32, `accuracy.py`):

| where | variant | op output vs torch SDPA (rel L2) | layer output: captured rel L2 / redrawn fails |
|---|---|---|---|
| LocDiT layer 0, 11 tokens | fp32 math, no FP8 | 0.0012 | 0.0011 / 0 of 30 |
| | Q, K e4m3, one scale per (batch, head) block | 0.030 | 0.0033 / 0 of 30 |
| | Q, K, V e4m3 + P e4m3 (×448), FA3-style | 0.038 | 0.0035 / 0 of 30 |
| | same + W8A8 row-wise GEMMs | | 0.0204 / 0 of 30 (= GEMMs alone) |
| base-LM decode layer 0, 17 / 47 / 76 cached tokens | K/V e4m3, per-tensor scale from the cache | 0.007–0.016 | ≤ 0.0009 / 0 of 10 |
| | K/V e4m3, scale 1.0 (uncalibrated) | 0.011–0.034 (V's values sit near e4m3's subnormals: amax 0.20, RMS 0.013) | ≤ 0.0011 / 0 of 10 |
| | K/V e4m3, per token and head | 0.006–0.015 | ≤ 0.0009 / 0 of 10 |

Model-wide (`calib.py`, base LM, 28 layers): a static per-tensor K/V scale taken from
text A is exceeded on text B in 43 % (K) / 54 % (V) of the layers; the e4m3 error of
the cache is then 0.028 on average (max 0.055) vs 0.026 / 0.025 for dynamic per-tensor
/ per-token-and-head scales. Harmless for VoxCPM2's tiny contexts, but a calibrated
KV scale needs margin or a per-token scale.

**Speed.** What there is to win on VoxCPM2:

* **KV bytes**: 28 KB per token for the base LM + 8 KB for the residual LM (bf16;
  `calib.py`); contexts of 47–77 tokens (prompt + 30–60 patches). Per decode step at
  batch 16 and 77 tokens that is 44 MB against 1.70 GB of FP8 weights (36 layers ×
  47.2 M parameters): 2.6 % (batch 1: 0.16 %).
* **R3's decode attention** (Inductor `triton_red_fused__softmax…`): 19.2 ms per
  batched run, 8.9 us per call for ~1.3 MB of K/V: latency bound, not bandwidth.
* **A decode kernel on e4m3 K/V** (`kv_bench.py`: Triton, one program per (sequence,
  KV head), GQA 8, batch 16, 36 layer-sized caches):

  | cached tokens | K/V per layer (bf16) | bf16 K/V | e4m3 K/V (per-tensor scale) | speedup |
  |---|---|---|---|---|
  | 77 (VoxCPM2) | 1.3 MB | 2.75 us | 3.07 us | 0.90x |
  | 512 | 8.4 MB | 11.8 us (709 GB/s) | 9.8 us | 1.21x |
  | 2048 | 33.6 MB | 42.3 us (794 GB/s) | 32.6 us | 1.29x |
  | 8192 | 134 MB | 161.7 us (830 GB/s) | 123.0 us | 1.31x |

  With 32 programs for 70 SMs and a convert per element, the e4m3 kernel reaches
  only ~550 GB/s of codes; the 2x would need a split-KV (flash-decoding) kernel.
* **LocDiT attention** (R3 `rope_attn4_k`): 31.1 ms per batched run, 4.7 us per call
  for 32 MFLOP (2 × 2 × 32 × 16 × 11² × 128) = 7 TFLOP/s: latency, not math. FP8
  QKᵀ cannot shorten it.

**When it pays**: decode over long contexts (thousands of tokens, KV bytes ≳ weight
bytes per step) and prefill attention with long sequences — not VoxCPM2 (≤ 2 ms per
batched run upper bound). **Check**: an FP8 KV cache changes what the attention reads,
so it is a precision of its own (proposed `fp8_kv`, 8-bit, near-lossless tier; the
measured layer errors sit far inside it: relative L2 ≤ 0.0011 against 0.08), checked on captured caches and on redrawn
caches (the perturbed check already redraws KV-cache contents); FP8 attention math
(QKᵀ / PV) the same. **Ceilings**: KV-cache reads are not counted today ("Approximate:
… KV-cache reads … are not counted"); an `fp8_kv` column needs them first (§9).

## 6. Technique 4: static (calibrated) vs dynamic scales, outlier migration

**How it works.** Dynamic scales take the amax of the data at run time (per token,
per group); static scales are calibrated offline (vLLM's old `activation_scheme=
"static"`; Transformer Engine's delayed scaling uses an amax history). Static scales
save the row reduction and make every producer able to quantise locally, at the risk
of saturating inputs larger than the calibration set. SmoothQuant
([arXiv 2211.10438](https://arxiv.org/abs/2211.10438)) divides each input channel by
s_j = max|X_j|^α / max|W_j|^(1−α) (α = 0.5) and multiplies the weight's column by it,
moving outliers from activations into weights. VoxCPM2 has massive activations
([Sun et al., arXiv 2402.17762](https://arxiv.org/abs/2402.17762): few fixed channels
up to ~10⁴–10⁵ × the median, acting as biases):

| GEMM input (text A, all layers) | median of max/median channel amax | worst (channel, layer) |
|---|---|---|
| LocDiT q/k/v, gate/up | 7–10 | 21 (channel 497) |
| LocDiT down_proj (silu·up) | 13 | 1053 (channel 3882, layer 0) |
| base-LM q/k/v | 29 | 396 (channel 1299, layer 0) |
| base-LM down_proj | 41 | 6345 (channel 523, layer 2) |
| residual-LM down_proj | 54 | 91 |

**Model-wide, text A → text B** (`calib.py`; relative L2 of the quantised activation
itself, mean / max over calls; "saturates" = calls with amax above A's):

| GEMM input | per token | 1×128 | MXFP8 ceil | per tensor, dynamic | **static per tensor (A)** | static saturates |
|---|---|---|---|---|---|---|
| LocDiT down_proj | 0.025 / 0.027 | 0.020 / 0.023 | 0.027 / 0.042 | 0.026 / 0.028 | 0.027 / **0.139** | 2.6 % |
| LocDiT q/k/v | 0.021 / 0.025 | 0.017 / 0.023 | 0.027 / 0.041 | 0.025 / 0.030 | 0.026 / 0.030 | 0.6 % |
| LocDiT o_proj | 0.026 / 0.027 | 0.025 / 0.027 | 0.027 / 0.031 | 0.026 / 0.028 | 0.027 / 0.029 | 2.6 % |
| LocEnc down_proj | 0.024 / 0.027 | 0.019 / 0.021 | 0.026 / 0.031 | 0.025 / 0.029 | 0.032 / **0.282** | 5.8 % |
| base-LM down_proj | 0.024 / 0.029 | 0.020 / 0.024 | 0.026 / 0.055 | 0.024 / 0.029 | 0.028 / **0.152** | 1.4 % |
| base-LM o_proj | 0.026 / 0.029 | 0.025 / 0.030 | 0.027 / 0.031 | 0.026 / 0.029 | 0.027 / 0.083 | 2.1 % |
| residual-LM down_proj | 0.025 / 0.029 | 0.019 / 0.021 | 0.027 / 0.037 | 0.025 / 0.029 | 0.031 / **0.328** | 1.2 % |
| residual-LM o_proj | 0.026 / 0.028 | 0.026 / 0.027 | 0.027 / 0.029 | 0.026 / 0.028 | 0.027 / 0.031 | 8.8 % |

Module level (`accuracy.py`, §3.2's layers): static per-tensor scales pass the
captured cases but fail the redrawn-input check in 1 of 10 LM decode draws at each of
the three steps (norm off by up to 4.3 %); SmoothQuant + static fails captured LM
inputs (norm off by 4.7 % at step 1) and 3 of 10 / 6 of 30 redrawn draws;
SmoothQuant + dynamic per-token passes everything with the per-token error (0.0209
vs 0.0204 LocDiT).

**Cost side**: a static scale saves the amax: 1.27 vs 1.48 us (RMSNorm) and 2.71 vs
3.03 us (silu·up) per call, ≤ 0.3 us × 2 producers × 6480 layer calls ≈ 4 ms per
batched run at most.

**Verdict**: keep dynamic scales (the `fp8_w8a8` contract already forbids static /
cached ones, and the evaluator catches them: activation scales cached from the first
call fail 24 of 32 redrawn draws, `compare.py`). When a producer must quantise
locally, use group scales (1×32 / 1×128), not static ones. SmoothQuant stays the
documented fallback for a compute-bound GEMM that fails the tier; none does on
VoxCPM2. **Check**: the existing redrawn-input check; amplitude-scaled redraws
(RESEARCH-TRITON §5.3 issue 6) would make static scales fail deterministically.
**Class / ceilings**: no change.

## 7. Technique 5: host overhead of FP8 GEMM calls

Eager wall time per call (`host_peak.py`; GPU work a few us, so these are host bound):

| call path | us per call |
|---|---|
| `F.linear` bf16 (cuBLAS) | 10.0 |
| `torch._scaled_mm` tensor-wise / row-wise | 18.3 / 19.6 |
| `F.scaled_mm` MXFP8 (Python recipe dispatch) | 27.1 |
| `at::_scaled_mm` called from C++ (pybind11) | 19.4 |
| **direct cuBLASLt, cached descriptors + algo (pybind11)** | **6.1** |
| four direct cuBLASLt GEMMs behind one pybind11 call | 21.8 (5.5 each) |
| pybind11 call with an empty body | 0.4 |
| CUTLASS SM120 blockwise via `load_inline` (args + `initialize` every call) | 5.4 |
| Triton e4m3 GEMM launch (pre-quantised x) | 12.2 |
| Triton quant + GEMM (`triton_fp8_w8a8_gemm.py` launcher) / through `torch.library` | 43.5 / 69.2 |
| CUDA graph replay of four cuBLASLt GEMMs | 5.0 |
| launch floor: CuTe DSL RMSNorm (TVM-FFI) / CUDA C++ RMSNorm (`load_inline`) / torch RMSNorm module | 12.4 / 6.9 / 63.3 |

* `at::_scaled_mm`'s cost is in ATen (scale checks, layout logic, a fresh output),
  not in Python: 19.4 us from C++ vs 6.1 for the same cuBLASLt kernel called
  directly. `cublasLtMatmul` itself plus descriptor attribute updates is ~5 us.
* R3 measured the consequence on an eager-timed module: the W8A8 layer was host bound
  at 130 us per call (4 × `at::_scaled_mm`, 16 us each) with 100 us of GPU work;
  direct cuBLASLt with cached descriptors and the best of the heuristic's 8 algos
  timed once (the top-1 picked a slower `sm89_xmma` split-K kernel for one shape)
  took it from 3.46x to 5.42x (`targets/dit_layer__fp8_w8a8/NOTES.md`, slice 31).
* Under CUDA graphs (R2 and R3's integrated models) host time is replay time; none of
  the above reaches the end-to-end number.

**When it pays**: eager-timed module targets (the evaluator) and any path not
captured in a graph; ≥ 2 FP8 GEMMs per call. **Check**: host time is part of the
evaluator's wall time already; a selftest that the helper's output equals
`fp8_w8a8_linear` / the MXFP8 reference. **Ceilings**: the "calls × launch floor"
term (15.5 us) is a torch-op floor; a direct-cuBLASLt or one-launcher path is ~6 us
per GEMM.

## 8. What it adds up to for VoxCPM2

**Batch 16** (R3: 925 ms per batched run, 6.02 ms per audio second; LocDiT layer
called 6480 times at M = 352):

| change | per layer call | per batched run |
|---|---|---|
| MXFP8 q\|k\|v and gate\|up (cuBLASLt), tensor-wise split-K o / down unchanged | GEMMs 64.1 → 58.0 us | −40 ms |
| silu·up + per-token e4m3 glue without s_x·s_w (gate\|up output already scaled): R3's 5.35 us (`quant_k<1,256>`) / 5.74 measured → 3.03 us | −2.3 to −2.7 us | −15 to −18 ms |
| norm1 / norm2 emit MXFP8 instead of per-token e4m3 (+0.06 us each) | +0.1 us | +1 ms |
| FP8 KV cache, FP8 attention, static scales | | ≤ 2, 0, ≤ 4 ms |
| **total** | | **≈ −55 ms (−6 %): ~870 ms, ~5.7 ms per audio s** |

The LocEnc (M = 80) and the LM (M = 16 decode, 272 prefill once) gain nothing worth a
change: MXFP8 equals tensor-wise there or loses.

**Single request** (R2: 522 ms): the LocDiT runs at M = 22, the LMs at M = 1 — memory
bound, where R2's FP8 weight-only kernels already stream one byte per weight; W8A8 /
MXFP8 add no bandwidth (and MXFP8's single algorithm is slower at N = 1024). Expected
gain from all five techniques: ≈ 0. The latency headroom is elsewhere: R2's re-profile
puts both LMs' decode at ~305 ms (hooked, approximate) against a 130 ms FP8 floor
(`rounds/2/profile/ceilings.md`) — kernel efficiency and launch structure (#134),
overlap (#136). [PARALLEL.md](PARALLEL.md) §8.1 (#136) puts the largest measured
latency headroom in weight streaming of the fused batch-1 LocDiT layer (−130 ms),
also a bytes question, not a precision one.

## 9. Plan

One PR per technique, in priority order. Each: reference math in
`kernels/quant.py`, a verified example (selftest, `doctor --smoke`), tier
calibration on captured *and* redrawn inputs (#109), a `low_precision.md` section,
planner policy (bound and M), and the ceilings column (#90). 4-bit stays opt-in (#131).

| technique | precision class (#131) | tier | ceilings (#90) |
|---|---|---|---|
| 1 blockwise → MXFP8 | `fp8_w8a8` (a scale layout, not a class) | near-lossless, unchanged (§3.2) | W8A8 column (330 TFLOP/s); mark the 208 cap of `QMMA.F32` kernels |
| 2 FP8 between fused ops | `fp8_w8a8` | same | W8A8 (halved activation I/O: negligible at M = 352) |
| 3 FP8 attention / KV cache | new `fp8_kv` (8-bit: allowed by default in near-lossless) | near-lossless on captured and redrawn caches | KV bytes in decode floors first, then an FP8-KV column |
| 4 static scales, SmoothQuant | none: scales stay dynamic | — | — |
| 5 host overhead | none | — | launch-floor term per call path (~6 us direct cuBLASLt vs 15.5 us) |

1. **MXFP8 as the sm_120 "blockwise" recipe (technique 1).** `quantize_mxfp8`
   (e4m3 + ue8m0 per 32, the non-saturating ceil rule, 128×4 swizzled scale layout)
   and `mxfp8_linear` (reference: `F.scaled_mm` `BlockWise1x32`); an example W8A8
   GEMM that picks MXFP8 for wide N and tensor-wise + batched split-K for N ≤ 1024;
   selftest: equals the fp32 math; the OCP floor rule fails the o_proj case;
   `low_precision.md`: scaling recipes on sm_120 (§2), when MXFP8 pays (§3.3).
   Planner: `fp8_w8a8` targets at M ≳ 130 name the recipe per GEMM. Expected: −40 ms
   per batched run at batch 16 (−4 %), 0 at batch 1.
2. **Direct cuBLASLt helper (technique 5).** A bundled C++ helper / example
   (`load_inline`): descriptors and layouts cached per (M, N, K, scale mode), the
   heuristic's ≤ 8 algorithms timed interleaved once per shape (R3: sequential timing
   on a busy GPU picked a 10 % slower kernel), tensor-wise, MXFP8 and batched
   split-K, several GEMMs behind one call, graph-capturable (fixed workspace).
   `fp8_w8a8_linear` keeps `_scaled_mm` as the reference. Expected: 0 end to end
   (graphs); eager-timed W8A8 targets lose 12–14 us of host time per GEMM (R3's
   layer: 3.46x → 5.42x).
3. **Producers emit FP8 (technique 2).** Verified fused RMSNorm → e4m3 (per token /
   MXFP8 with swizzled scales) and silu·up → e4m3 kernels; contract text: never a
   separate quantisation pass; with MXFP8 / tensor-wise outputs, where the scales are
   applied. Expected on top of 1: −15 to −18 ms per batched run (−2 %). GEMM
   epilogue fusion waits for follow-up A.
4. **Scaling policy and checks (technique 4).** No static scales in the contract
   (already); `low_precision.md` gets the model-wide table (§6) and the KV-scale
   finding; a test that a static-scale layer fails the redrawn check on the captured
   LM layer (regression guard for #109's bounds); SmoothQuant reference helper
   (`smooth_weights(w, amax_x, alpha)`) for the rare failing GEMM. Expected: 0 ms
   (prevents wasted slices).
5. **FP8 KV cache and FP8 attention (technique 3), for long-context workloads.**
   Precision `fp8_kv` (8-bit, near-lossless tier), a split-KV FP8 decode example
   (`llm.py` workloads), KV bytes in the decode floors of the ceilings table.
   Expected on VoxCPM2: ≤ 2 ms per batched run — do it only when an LLM target with
   ≥ 2k tokens of context is planned.

Proposed follow-up issues (not opened):

* **A. Block-scaled (QMMA.SF) GEMM with fused epilogue in CuTe DSL / CUTLASS.**
  The only route to nvjet-class FP8 GEMMs that also fuse silu·up, residual and
  e4m3 output: CuTe DSL `MmaMXF8Op` or CUTLASS `OpClassBlockScaledTensorOp` with
  `mx_float8_t<e4m3>` (unit or real scales). Our CUTLASS 4.1 (tilelang copy) build
  compiled at 128×128×128 but `initialize()` returned `Error Internal`, 64-wide
  tiles failed TMA static asserts. Target: gate|up with silu·up + e4m3 epilogue
  under 28 us at M = 352 (today's best chain, cuBLASLt MXFP8 + glue: 28.1 us). Same
  as RESEARCH-TRITON §5.3 issue 2; part of #133.
* **B. MXFP8 for N = 1024 at M = 352 (split-K / small tiles).** cuBLASLt returns one
  MXFP8 algorithm: o_proj 14.8 and down 26.6 us vs tensor-wise split-K 8.8 / 15.2.
  Try batched MXFP8 (scale offsets per batch), Triton `tl.dot_scaled` with split-K
  (12.8 / 25.8 us without), or a CuTe kernel; would let the whole LocDiT layer run
  MXFP8 and quantise the attention output per head inside the attention kernel (R3's
  separate pass: 10.6 ms per batched run).
* **C. Ceilings: which FP8 instruction a kernel needs, and KV bytes.** Show the W8A8
  floor at 330 TFLOP/s *and* the 208 cap of `QMMA.F32` kernels (row-wise CUTLASS,
  Triton `tl.dot`, DeepSeek-style blockwise), so planners stop expecting the floor
  from them; count KV-cache bytes in decode floors (needed before an `fp8_kv`
  column). Overlaps RESEARCH-TRITON §5.3 issue 3.
* **D. MXFP8 quantiser guard.** Selftest + knowledge: the OCP reference rule
  saturates massive activations (VoxCPM2 o_proj / down: norm off by 4 % / 2 %); the
  library and examples use the ceil rule; a candidate that quantises with the floor
  rule fails the tier on captured inputs already, the selftest makes the reason
  explicit.
* **E. Split-KV FP8 decode attention for `llm.py` workloads.** Flash-decoding over
  e4m3 K/V with per-token-head or margin-calibrated per-tensor scales (static K/V
  scales from one text are exceeded by another in ~50 % of layers); target: ≥ 1.8x
  over bf16 at ≥ 4k tokens of context, where the naive kernel reached 1.31x.

## 10. Measured: the toolkit's FP8 GEMM paths (after #144, #145, #133)

The examples the agents start from, on the RTX 5070 Ti (torch 2.14.1, Triton 3.8, CuTe DSL
4.8, cuBLASLt 13.4), each wrapping a bf16 `nn.Linear` with Gaussian weights; one call per
shape `M x K x N`, median of 15 x 10 calls, inside a CUDA graph of 20 calls (`graph`, GPU
time) and eagerly (`eager`, with host time). Every path is W8A8 (e4m3 activations per token
or MXFP8 per 32, e4m3 weights per channel or per 32): relative L2 vs bf16 0.037 everywhere
(the FP8 weight-only skinny GEMV: 0.027). Script:
[`research-scripts/fp8-sm120/toolkit_fp8_gemms.py`](research-scripts/fp8-sm120/toolkit_fp8_gemms.py),
raw numbers in `results/toolkit_fp8_gemms.json`.

| M x K x N | bf16 cuBLAS | `_scaled_mm` row-wise | CuTe block-scaled (#133) | cuBLASLt tensor-wise (#145) | cuBLASLt MXFP8 (#145) | Triton `tl.dot_scaled` (#145) | Triton MXFP8 (#144) |
|---|---|---|---|---|---|---|---|
| 16 x 2048 x 6144 | 13.9 | 26.3 | 17.3 | 13.3 | 18.8 | **9.7** | 18.0 |
| 352 x 1024 x 2560 | 25.4 | 28.5 | **10.1** (182 TFLOP/s) | 15.3 | 11.4 | 10.7 | 11.5 |
| 352 x 1024 x 8192 | 69.2 | 53.9 | **24.9** (237) | 36.7 | 26.3 | 33.1 | 26.6 |
| 352 x 4096 x 1024 | 38.9 | 59.2 | 28.5 | 20.8 | 31.5 | **18.7** (158) | 32.9 |
| 2048 x 4096 x 4096 | 731 | 653 | **259** (265) | 268 | 263 | 352 | 268 |
| 8192 x 8192 x 8192 | 11261 | 8718 | 5586 | 3952 | **3746** (294) | 5569 | 3776 |

(µs per call, graph-timed.) What it says for the backend policy (`backends.py`):

* Mid-size compute-bound GEMMs with wide N (M in the hundreds to thousands): the CuTe DSL
  block-scaled kernel is the fastest (2.8x bf16 at 352 x 1024 x 8192), cuBLASLt MXFP8 next.
* N = 1024 at M = 352 leaves SMs idle for single-pass tiles: split-K wins, here Triton
  `tl.dot_scaled` and cuBLASLt tensor-wise (its split-K algorithms); the CuTe kernel has no
  split-K yet (§9 follow-up B).
* Very large square GEMMs: cuBLASLt (MXFP8 294 TFLOP/s, tensor-wise 278); the CuTe kernel's
  tile schedule falls to 197 there.
* Decode (M = 16): the GEMM is a weight stream; `tl.dot_scaled` with BM = 16 is the fastest
  W8A8 path, and FP8 weight-only GEMVs are the right class anyway (§3).
* Eager host time is large for every FP8 path (CuTe ~20 µs, cuBLASLt ~18 µs, Triton ~40 µs
  per call over the graph time): eager-timed targets need one launcher per module or graphs
  (§7).
