# Triton research: libraries, agents, our ledgers (issue #135)

Input for the CUDA / CuTe work (#131–#134, #136): what Triton work exists, what
LLM agents get wrong on Triton and CUDA, what our VoxCPM2 ledgers say per backend,
and which backend kernel-agent should use for which target class. Compiled
2026-10-07. Every claim has a source URL, a file of our runs, or a measurement made
for this note (§1, method below). "(?)" marks what we could not confirm. Agent-paper
numbers are the authors' own. Related: [RESEARCH.md](RESEARCH.md) (agent systems,
hack catalogue v0.1.0), [VOXCPM2.md](VOXCPM2.md).

Run names used below (all `runs/openbmb--VoxCPM2/`, read-only evidence):

| name | directory | workload | result |
|---|---|---|---|
| R1 | `20261005-042829` | 1 request, bf16 exact | 5407 → 736.9 ms, 7.34x eager (`report.md`) |
| R2 | `20261005-192504` | 1 request, FP8 / FP4 weights, near-lossless | 5476 → 522.2 ms, 10.49x eager, 7.20x compile |
| R3 | `20261006-004718` → `-retest` → `-retest2` | batch 16, ms per audio second | 37.7 → 6.03 ms, 6.24x eager, 3.15x compile |

R3's three directories share one ledger: `-retest2/results.tsv` (233 rows) extends
the other two line for line, so it is counted once.

## 0. Summary

1. **On sm_120 (GeForce Blackwell) FP8 `mma.sync` with fp32 accumulation runs at
   half rate (208 TFLOP/s on the RTX 5070 Ti); the block-scaled instruction
   (`kind::mxf8f6f4.block_scale`, SASS `QMMA.SF`) runs at full rate (416) with the
   same fp32 accumulation** (§1.1, measured; the same 2x on an RTX 5090 in
   [FlashInfer #5963](https://github.com/flashinfer-ai/flashinfer/issues/5963)).
   That is why Triton `tl.dot` on e4m3 and the row-wise `_scaled_mm` (CUTLASS
   `SM120_16x8x32_TN`) both stop at ~193 TFLOP/s while cuBLASLt's tensor-wise
   kernel reaches 325.
2. Triton 3.8 lowers `tl.dot_scaled(e4m3, ue8m0)` to native `QMMA.SF` on sm_120
   (native since [triton #7918](https://github.com/triton-lang/triton/pull/7918),
   bf16 emulation before). With all
   scales = 127 (2^0) its output is bit-identical to `tl.dot` with fp32
   accumulation and it is 4–16 % faster at the LocDiT shapes (gate|up 36.4 →
   31.7 us with TMA) and 214–227 vs 190 TFLOP/s at 4096³ (§1.2). It is a drop-in for every
   Triton FP8 GEMM; per-token / per-channel scales stay in the epilogue.
3. cuBLASLt tensor-wise FP8 (nvjet) still beats every Triton variant on plain
   GEMMs (gate|up 30.2 us, 325 TFLOP/s at 4096³). Triton wins only where it fuses
   work around the GEMM (scales, split-K, epilogues) or where the library leaves
   SMs idle; CUDA C++ / CuTe on the block-scaled MMA is the route to nvjet-class
   GEMMs *with* fused epilogues.
4. Triton TMA descriptors work on sm_120 (`cp.async.bulk.tensor`); automatic warp
   specialisation does not pay (`tl.dot` 11–29 % slower) and crashes the compiler
   for `tl.dot_scaled` (§1.2).
5. Our ledgers: 73 CUDA C++ evaluations (88 % correct, 42 kept) vs 6 Triton and 4
   hybrid CUDA+Triton (§4). No TileLang or CuTe DSL kernel was ever written: the
   "tilelang+cuda" ledger label is CUDA C++ that includes CUTLASS headers from the
   tilelang wheel. The planner listed `cute` once and Triton first five times;
   the engineers wrote CUDA anyway.
6. Triton produced three kept VoxCPM2 steps, all inside CUDA-graph transforms: the shape-tuned
   e4m3 GEMM at M = 352 (9.03 → 8.73 ms per audio s), ≤ 16-token attention (cuDNN
   15.9 → 5.7 us; 8.50 → 8.00 ms) and split-K for N = 1024 (8.00 → 7.72 ms).
   Inductor (which emits Triton) carried the glue fusion in every run.
7. Triton lost where host time counts (eager-timed module: 2.21x Triton vs 2.70x
   for the same math behind one C++ launcher), on fused halo-stencil + GEMM at
   C ≥ 64 (255 registers, spills; CUDA `mma.sync` with an smem halo won 9.64x vs
   7.88x), and against strict fp32/TF32 references (Triton's TF32 truncates, cuDNN
   rounds to nearest).
8. Agent literature (§3): the dominant Triton failure is *compiles but wrong*
   (AutoTriton 83–97 % compile vs 20–45 % correct), then API/type hallucination
   (69 % of TritonBench-G errors); "not really Triton" (PyTorch fallbacks) is the
   dominant hack; serial refinement beats parallel sampling at equal budget (Kevin:
   1.10x vs 0.65x).
9. kernel-agent's evaluator already covers most published hacks (side streams,
   patching, caching, lazy tensors, fallbacks, precision downgrade). Gaps: no
   scaled / sign-flipped redraws (KernelBench-Verified's ×3, ×0.01, ×−1) and no
   module-level peak-memory check (§3.3).
10. Backend policy (§5): compute-bound FP8 GEMM → block-scaled MMA (Triton
    `tl.dot_scaled` now, CuTe DSL / CUTLASS for fused epilogues); small-M GEMV →
    CUDA C++ (Triton under graphs is at bandwidth too); ≤ 16-token attention →
    Triton; conv / VAE → Triton for GEMM-shaped parts, CUDA for fused stencils;
    fused decoder layers → CUDA C++ one-launcher when eager-timed, Inductor +
    library GEMMs inside graphs.

## 1. Triton on sm_120: measured here

Setup: RTX 5070 Ti (sm_120, 70 SMs, 48 MB L2, DRAM copy peak 767 GB/s), torch
2.14.1+cu130, Triton 3.8.0, nvcc 13.4, 2026-10-07, each job under the GPU lock.
GEMMs: 12 distinct weight copies (L2-cold) called back to back inside one CUDA
graph, median of 15 replays, minimum over 3 interleaved rounds, GPU warmed first;
spreads ≤ 1.5 %. Inputs random, e4m3 per-row/per-column scales; every Triton result
is bit-identical (relative error 0.0) to the cuBLASLt result with the same scales.
Scripts (scratchpad of this issue, not committed): `bench_fp8.py` (all columns
below except TMA `tl.dot_scaled`), `bench_mx.py` (TMA `tl.dot_scaled`, warp
specialisation; one round after warm-up), `const_scale.py`, `mma_rate.cu`,
`sass_check.py`, `nvjet_name.py`.

### 1.1 Tensor-core instruction rates

Register-only loop, 280 blocks × 8 warps, 8 independent accumulators per warp,
`nvcc -gencode arch=compute_120a,code=sm_120a`; SASS from `cuobjdump -sass`.

| PTX | SASS | TFLOP/s |
|---|---|---|
| `mma.sync.m16n8k16.f32.bf16.bf16.f32` | `HMMA.16816.F32.BF16` | 104 |
| `mma.sync.m16n8k32.f32.e4m3.e4m3.f32` (sm_89 form) | `QMMA.16832.F32.E4M3.E4M3` | 208 |
| `mma.sync.kind::f8f6f4.m16n8k32.f32.e4m3.e4m3.f32` | `QMMA.16832.F32.E4M3.E4M3` (same) | 208 |
| `mma.sync.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.f32.e4m3.e4m3.f32.ue8m0` | `QMMA.SF.16832.F32.E4M3.E4M3.E8` | **416** |
| `mma.sync.m16n8k32.f16.e4m3.e4m3.f16` (fp16 accumulate) | `QMMA.16832.F16.E4M3.E4M3` | 416 |

So the GeForce rule "fp32 accumulation at half rate" (bf16: 104 here vs ~200 with
fp16 accumulation, `-retest2/targets/loc_enc_decode/NOTES.md` slice 28) holds for
FP8 too, **except** for the block-scaled instruction. The ceilings table's FP8 peak
(330 TFLOP/s, measured with cuBLASLt) is therefore a GEMM peak; the instruction peak
is 416, and any kernel on `QMMA.F32` is capped at 208.

### 1.2 FP8 GEMMs (us per call; TFLOP/s in brackets)

| shape M×N×K | bf16 cuBLAS | `_scaled_mm` scalar scales (cuBLASLt nvjet) | `_scaled_mm` row-wise (CUTLASS) | Triton `tl.dot` e4m3 | Triton `tl.dot_scaled` e4m3/ue8m0 |
|---|---|---|---|---|---|
| LocDiT gate\|up 352×8192×1024 | 69.6 | **30.2** (196) | 41.5 | 36.4 (162) | 34.3; TMA **31.7** (187) |
| LocDiT qkv 352×2560×1024 | 26.1 | **10.9** | 15.6 | 14.3 | 12.4 |
| LocDiT o_proj 352×1024×2048 | 22.8 | 13.4 | 15.3 | 14.2 | **11.9** |
| LocDiT down 352×1024×4096 | 39.1 | **17.7** | 27.1 | 26.4 | 25.4; TMA 24.1 |
| LM decode gate\|up 16×12288×2048 | 65.8 | 32.8 | 52.7 | **32.5** (BM 16) | — |
| 4096³ | 1420 (97) | **423 (325)** | 713 (193) | 725 (190) | 642 (214); TMA 606 (227) |

* Kernels (torch profiler): tensor-wise = `nvjet_sm120_qqtst_mma_128x128x64_6_64x64x64_tmaAB_bz_TNNN`
  (4096³) / `..._128x64x64_8_64x32x64_...` (M = 352); row-wise = PyTorch's CUTLASS
  3.x `MainloopSm120TmaWarpSpecialized` with `SM120_16x8x32_TN<e4m3, e4m3, float>`
  (`QMMA.F32`, hence ≤ 208). nvjet's 325 TFLOP/s exceeds the `QMMA.F32` cap and its
  output equals the fp32-accumulated `QMMA.SF` result bit for bit, so it runs a
  full-rate path, consistent with `QMMA.SF` (inference: its SASS could not be
  extracted from `libcublasLt.so.13`).
* Triton codegen (PTX / SASS of the compiled kernels): target `sm_120a`; `tl.dot`
  → `QMMA.16832.F32` + `LDSM` + `LDGSTS` (cp.async); `tl.dot_scaled` →
  `QMMA.SF.16832.F32.E4M3.E4M3.E8`; host `TensorDescriptor` → `cp.async.bulk.tensor`
  (TMA). No `wgmma` / `tcgen05` anywhere. Triton 3.8 sends capability ≥ 100 through
  its Blackwell pass pipeline (TMEM hoisting, automatic warp specialisation;
  `triton/backends/nvidia/compiler.py`, `make_ttgir`), which sm_120 cannot use.
* `tl.range(..., warp_specialize=True)` + TMA: `tl.dot` slower everywhere (gate|up
  36.2 → 46.8 us, 4096³ 197 → 173 TFLOP/s); `tl.dot_scaled` fails to compile in 8
  of 9 tile configs (`PassManager::run failed` in `TritonGPUOptimizePartitionWarps`)
  and the one that compiles runs at 28 TFLOP/s.
* Best tiles (graph-timed): e4m3 `tl.dot` 64×64×64 w4 s3 (gate|up), 128×64×128 w8
  s3 (qkv, down); `tl.dot_scaled` 64×64×128 w4 s3 (M = 352), 128×128×128 w8 s3
  (4096³, pointers) / 256×128×64 w8 s3 (TMA). Big Hopper/B200 tiles (128×256×64,
  4 stages) exceed sm_120's ~99 KB shared memory per block (`OutOfResources`); in
  R3 several Inductor max-autotune FP8 templates failed the same way
  (`-retest2/transforms/NOTES.md`, "Measured locally").
* Scales: the `tl.dot_scaled` column uses unit scales loaded from uint8 tensors.
  Constant scales (`tl.full((BM, BK // 32), 127, tl.uint8)`) also lower to
  `QMMA.SF` in Triton 3.8 and are faster: gate|up with row / column scales in the
  epilogue 33.3 us (pointer loads), output bit-identical to row-wise `_scaled_mm`
  (`const_scale.py`, 3 rounds 33.3–33.8 vs 35.3–35.9 us for loaded unit scales).
  gemlite notes the constant-scale form broke the SM120 lowering in Triton 3.7
  ([gemm_kernels.py](https://raw.githubusercontent.com/dropbox/gemlite/master/gemlite/triton_kernels/gemm_kernels.py)).
* Memory-bound M = 16: a plain `tl.dot` with BM = 16, BN = 64, BK = 256 streams a
  25 MB e4m3 weight at 774 GB/s (DRAM peak), as fast as nvjet; the row-wise
  `_scaled_mm` kernel is 60 % slower there.

## 2. Libraries and techniques

### 2.1 Triton libraries: techniques worth copying

| library | technique | for kernel-agent |
|---|---|---|
| Liger-Kernel ([repo](https://github.com/linkedin/Liger-Kernel), [paper](https://arxiv.org/abs/2410.10989)) | RMSNorm with the 1/rms cached, RoPE on Q and K in one kernel, SwiGLU recomputed in backward, cross-entropy gradient computed in the forward with an online softmax written over the logits, fused linear + CE in chunks ([paper v3](https://arxiv.org/html/2410.10989v3)); A100: RMSNorm ~7x, RoPE ~8x, CE ~3x faster | reference implementations for norm / RoPE / gated-MLP glue |
| ″ testing | tests on non-power-of-2 shapes; bf16 atol 1e-3, rtol 1e-2; convergence tests on 4-layer mini models comparing loss, top-k log-probs and weights ([test_mini_models.py](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/test/convergence/bf16/test_mini_models.py)); bugs they hit: int32 offset overflow once `program_id * stride` > 2³¹−1, a non-contiguous gradient from SDPA ([paper v3](https://arxiv.org/html/2410.10989v3)) | int64 offsets and stride handling belong in `triton.md`; our redrawn-input + teacher-forced checks play the convergence-test role |
| ″ v0.8.3 (Sep 2026) | one dispatcher over Triton, CuTe DSL and cuTile backends ([release](https://github.com/linkedin/Liger-Kernel/releases/tag/v0.8.3)) | the library world is moving to multi-backend per op, like our per-target backend list |
| FlagGems ([repo](https://github.com/flagos-ai/FlagGems)) | 180+ ATen ops in Triton registered with `torch.library.Library("aten", "IMPL")` ([`__init__.py`](https://raw.githubusercontent.com/FlagOpen/FlagGems/master/src/flag_gems/__init__.py)); `@pointwise_dynamic` codegen for any broadcast / layout ([PyTorch blog](https://pytorch.org/blog/flaggems-joins-the-pytorch-ecosystem-triton-powered-operator-library-for-universal-ai-acceleration/)); `LibTuner` keys bucketed by log2 / align32 + dtypes, tuned configs persisted in SQLite per vendor and Triton version; `LibEntry` caches compiled kernels per device and skips the JIT binder, autotuner and heuristics on repeat calls ([libentry.py](https://raw.githubusercontent.com/flagos-ai/FlagGems/master/src/flag_gems/utils/libentry.py)) | a persistent tuned-config cache (GPU, Triton version, shape bucket) across evaluations and runs; a cheap launch path |
| gemlite ([repo](https://github.com/dropbox/gemlite)) | kernel by M: GEMM at M ≥ 64, split-K GEMM above M = 2 (4 for < 8-bit non-MX), else GEMV with (reverse) split-K ([core.py](https://raw.githubusercontent.com/dropbox/gemlite/master/gemlite/core.py)); JSON autotune cache per GPU; MXFP8 through `tl.dot_scaled`; `GEMLITE_USE_TMA = False  # ... faster MXFP8 on sm_120` (core.py); a source comment says Triton 3.7 broke SM120 lowering with a synthetic E8M0-one scale tensor ([gemm_kernels.py](https://raw.githubusercontent.com/dropbox/gemlite/master/gemlite/triton_kernels/gemm_kernels.py)) | the M thresholds match our evidence; unit scales lower correctly in Triton 3.8 here (§1.2); TMA helped `tl.dot_scaled` at our shapes, so time both |
| Unsloth ([kernels/](https://github.com/unslothai/unsloth/tree/main/unsloth/kernels)) | in-place RoPE, 4 heads per program, backward = forward with −sin; CE chunked above 65 536 classes; `BLOCK = next_pow2(n)` ≤ 65 536, warps 4–32 ([utils.py](https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/kernels/utils.py)); FP8: 128×128 block dequant, `s = amax / 448` activation quant, `BLOCK_M = max(next_pow2(M), 16)`, falls back to `torch._scaled_mm` on SM120 ([fp8.py](https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/kernels/fp8.py)); `triton_launch.py` caches a specialisation and calls `compiled.run()` directly ([triton_launch.py](https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/kernels/triton_launch.py)) | the direct `compiled.run()` launch for eager-timed targets |
| Triton tutorials ([sources](https://github.com/triton-lang/triton/tree/main/python/tutorials)) | 03 matmul: GROUP_SIZE_M L2 grouping (A100 220 → 245 TFLOP/s), its 128×256×64×3 fp16 config needs 147 KB of smem ([03](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)); 09 persistent: TMA (cc ≥ 9), CLC (cc ≥ 10), `EPILOGUE_SUBTILE`, `warp_specialize` ([09](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html)); 10 block-scaled: gated to cc 10 / 11, 5-D preshuffled scales via TMA, **sm_120 not covered** ([10](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html)); 11 PDL: `gdc_wait` / `gdc_launch_dependents` with `launch_pdl=True` ([11](https://raw.githubusercontent.com/triton-lang/triton/main/python/tutorials/11-programmatic-dependent-launch.py)); descriptors need a 16-byte aligned base and strides and `triton.set_allocator` ([make_tensor_descriptor](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html)) | 03's configs and 10's gate do not fit sm_120: the agent must not copy them; PDL is worth a test (in our CUDA kernels it was noise or worse: `loc_enc_decode` got faster with PDL off on some edges, `dit_layer__fp8_w8a8` ~0.3 us) |
| Gluon ([tutorials](https://github.com/triton-lang/triton/tree/main/python/tutorials/gluon)) | Triton's lower-level dialect: explicit layouts, smem, async copies, warp specialisation; SM120 support (mma.sync, decomposed `dot_scaled`) is an open PR ([#11260](https://github.com/triton-lang/triton/pull/11260)) | not usable on sm_120 yet |
| Triton-distributed ([repo](https://github.com/ByteDance-Seed/Triton-distributed), [paper](https://arxiv.org/abs/2504.19442)) | NVSHMEM primitives inside Triton kernels; AllGather-GEMM, GEMM-ReduceScatter, all-to-all overlap | multi-GPU only: input for #136 / #23, nothing for one GPU |

### 2.2 Triton on sm_120: outside evidence

* Blackwell (sm_100, sm_120) support arrived in Triton 3.3 with PyTorch 2.7
  ([blog](https://pytorch.org/blog/pytorch-2-7/)). Triton 3.8.0 (Aug 2026) adds a
  native block-scaled dot on SM121, TMA im2col and a breaking change to the
  descriptor type ([release](https://github.com/triton-lang/triton/releases/tag/v3.8.0)).
* `tl.dot_scaled` on the RTX 5090 was emulated through bf16 in Triton 3.4
  ([#7550](https://github.com/triton-lang/triton/issues/7550)); native MXFP8 on
  sm_120 landed in [#7918](https://github.com/triton-lang/triton/pull/7918)
  (2025-08-29; 1X scale vector, single CTA, both scales required, else emulation;
  vLLM Llama-3-8B on a 5090: FP8 42.8 s, native MXFP8 44.5 s, emulated 76.4 s).
  Native FP4 needs `k_pack=True` ([#9684](https://github.com/triton-lang/triton/pull/9684))
  and K = 64 ([#11776](https://github.com/triton-lang/triton/pull/11776), open).
* **Half-rate FP8 with fp32 accumulation on GeForce, confirmed independently**:
  FlashInfer [#5963](https://github.com/flashinfer-ai/flashinfer/issues/5963)
  (2026-10-02, RTX 5090): the legacy `m16n8k32.f32.e4m3.e4m3.f32` runs 510 vs 1014
  TFLOP/s for the block-scaled form with unit ue8m0 scales; in an FP8 attention
  kernel 1.29–1.34x, outputs bit-identical. TileLang
  [#3099](https://github.com/tile-ai/tilelang/pull/3099) reports the same "2x
  f32-accum win". NVIDIA's whitepaper lists the RTX 5070 Ti at 175.8 dense FP8
  TFLOP/s with fp32 accumulation, 351.5 with fp16 accumulation, 87.9 bf16, 703 FP4
  with fp32 accumulation, at the 2452 MHz boost clock ([RTX Blackwell
  whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf),
  Appendix B, Table 5); our 104 / 208 / 416 are the same ratios at the clock the
  card actually ran (~2.8 GHz).
* Shared memory: the per-block limit is 101 376 B ("Required: 102400, Hardware
  limit: 101376", [vLLM #60173](https://github.com/vllm-project/vllm/pull/60173));
  3-stage Triton GEMMs failed on a 5090 in FlashInfer
  ([#5517](https://github.com/flashinfer-ai/flashinfer/pull/5517)); OpenAI's
  `triton_kernels` MXFP4 MoE needed a persistent kernel, `epilogue_subtile=1` and
  `block_m` ≤ 32 on SM12x and is still 10 % behind Marlin
  ([vLLM #58877](https://github.com/vllm-project/vllm/pull/58877)); Inductor
  silently recompiles over-limit kernels with `num_stages=1`
  ([pytorch #199308](https://github.com/pytorch/pytorch/issues/199308)).
* TMA: Triton emitted `.tile::scatter4`, which sm_120a lacks
  ([#11859](https://github.com/triton-lang/triton/issues/11859), fixed); on GB200
  descriptors made pointwise / reduction kernels 1.4–2.2x slower than pointers
  ([pytorch #199818](https://github.com/pytorch/pytorch/pull/199818)). No `wgmma`
  on sm_120 ([SageAttention #291](https://github.com/thu-ml/SageAttention/issues/291)).
* FP8 `tl.dot` instead of an upcast to bf16: +35 % prefill scoring on an RTX PRO
  6000, identical results ([vLLM #59331](https://github.com/vllm-project/vllm/pull/59331)).
  RTX PRO 6000, Triton 3.4: bf16 GEMM 346–390 vs cuBLAS 378–386 TFLOP/s, causal
  attention 86–91 % of FlashAttention-2 ([arXiv 2604.23466](https://arxiv.org/html/2604.23466v1)).
* Attention on sm_120: FlashAttention-4 (CuTe DSL) has no SM120 support
  ([#2307](https://github.com/Dao-AILab/flash-attention/issues/2307)); a CuTe DSL
  FlashAttention-2 for SM120 with cp.async, TMA + warp specialisation and FP8 is an
  open PR ([cutlass #3030](https://github.com/NVIDIA/cutlass/pull/3030));
  SageAttention's CUDA kernels reach 560 TOPS on a 5090
  ([repo](https://github.com/thu-ml/SageAttention)).

### 2.3 ThunderKittens, TileLang, CuTe DSL, cuTile

| framework | sm_120 | exposes beyond Triton | evidence |
|---|---|---|---|
| ThunderKittens ([repo](https://github.com/HazyResearch/ThunderKittens)) | 2.0 (Jan 2026) targets Blackwell with MXFP8 / NVFP4; `ARCH=SM120` accepted, tuned sm_120 kernels (?) | C++ register / shared tiles, TMA, a load-compute-finish template | H100 GEMM 855 TFLOP/s (README) |
| TileLang ([repo](https://github.com/tile-ai/tilelang), [paper](https://arxiv.org/abs/2504.17577)) | SM70–SM120; block-scaled `mxf8f6f4` MMA on sm_120 in [#3099](https://github.com/tile-ai/tilelang/pull/3099): FP8 562 TFLOP/s at 4096³ on an RTX PRO 6000, bit-exact vs CUTLASS | scheduling (layout, pipelining, thread binding) as annotations separate from dataflow | paper: GEMM 1.08x Triton on an RTX 4090, FlashAttention 1.41x Triton on H100 ([v2](https://arxiv.org/html/2504.17577v2)); installed here (0.1.15), never written by an agent |
| CuTe DSL ([docs](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html)) | warp MMA ops `MmaMXF8Op`, `MmaMXF8F6F4Op`, `MmaMXF4NVF4Op` for sm_120a ([API](https://docs.nvidia.com/cutlass/4.7.0/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_warp.html)); examples `blackwell_geforce/kernel/{dense_gemm, blockscaled_gemm, conv}`; 4.8.0 (Sep 2026) adds a GeForce ping-pong GEMM and a block-scaled implicit-GEMM conv ([release](https://github.com/NVIDIA/cutlass/releases/tag/v4.8.0)); a segfault on non-block-scaled FP8 MMA on sm_120 ([#3044](https://github.com/NVIDIA/cutlass/issues/3044), closed) | layouts, explicit pipelines, warp specialisation, TVM-FFI launch ([guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/tvm_ffi_compilation.html)) | a CuTe DSL block-scaled GEMM tutorial for sm12x (NVFP4, ~60 % of peak on an RTX PRO 6000; [Colfax](https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/), not primary); FlashAttention-4 in CuTe DSL compiles 20–30x faster than C++ templates and runs 2.7x Triton on B200 ([arXiv 2603.05451](https://arxiv.org/abs/2603.05451)) |
| cuTile | — | tile language from NVIDIA | TileBench on B200: Triton 2.02x vs cuTile 1.58x geomean over PyTorch, Triton faster on 37 of 45 ops, cuTile 1.55x on a large GEMM (tensor pipe 79–81 % vs 52–69 %) ([arXiv 2609.29067](https://arxiv.org/html/2609.29067v1)) |

Reading: Triton is competitive on irregular, bandwidth-bound and glue kernels
(TileBench, Liger, FlagGems) and loses on peak-bound GEMMs and attention to
languages that expose pipelining and warp specialisation (TileBench GEMM, FA-4,
TileLang). On sm_120 the only public FP8 GEMMs above the 208 TFLOP/s `QMMA.F32`
cap (scaled to our card) use the block-scaled MMA: nvjet, TileLang #3099, the
CUTLASS / CuTe DSL GeForce examples.

### 2.4 Small M and launch overhead

* Split-K: up to 1.94x over the base Triton GEMM at small M on H100, cuBLAS ahead
  from M ≥ 32 ([PyTorch blog](https://pytorch.org/blog/accelerating-llama3/));
  Helion exposes split-K and swap-AB as tunables for M ≤ 32 under CUDA graphs
  ([blog](https://pytorch.org/blog/building-a-high-performance-and-portable-vllm-linear-backend-with-helion/));
  tritonBLAS picks tiles analytically at > 95 % of autotuned speed
  ([arXiv 2512.04226](https://arxiv.org/abs/2512.04226)). Ours: split-K paid for
  N = 1024 at M = 352 in Triton (§4.2) and as `RED` atomics in CUDA at M = 80
  (`-retest2/targets/loc_enc_decode/NOTES.md`).
* Launch cost: 165 us gaps around a 12 us GEMM until CUDA graphs ([PyTorch
  blog](https://pytorch.org/blog/accelerating-llama3/)); 220 us of launcher for an
  80 us kernel ([#2637](https://github.com/triton-lang/triton/issues/2637)); an RFC
  for frozen launch handles is open ([#10140](https://github.com/triton-lang/triton/issues/10140)).
  Workarounds: `compiled.run()` (Unsloth), `LibEntry` (FlagGems), AOT
  `triton/tools/compile.py` + `link.py`
  ([tools](https://github.com/triton-lang/triton/tree/main/python/triton/tools)),
  PDL (tutorial 11). Ours: ~43 us per Triton launch vs 19 us `load_inline`
  (`knowledge/playbook.md` §5).

## 3. Agent work on Triton / CUDA kernel generation

### 3.1 Works

| work | backend | headline (authors' setting) | what fails | hacks found → defence |
|---|---|---|---|---|
| KernelBench ([arXiv 2502.10517](https://arxiv.org/abs/2502.10517), [repo](https://github.com/ScalingIntelligence/KernelBench)) | CUDA in the paper; repo now `backend=` cuda / triton / cute / tilelang / thunderkittens | one-shot fast_1 on L40S ≤ 36 %; R1 on L2 36 → 72 % with 10 turns of execution + profiler feedback ([html](https://arxiv.org/html/2502.10517)) | execution failures (> 70 % for non-reasoning models), then functional mismatch, which feedback fixes least | v0.1 blog: torch/cuBLAS calls, kernels never called, no-ops reading the reference's output memory, wrong-stream work ([blog](https://scalingintelligence.stanford.edu/blogs/kernelbenchv01/)); static checker rejects try/except, `pass`, timer patches, threads, Tensor subclasses; Triton must contain `@triton.jit` ([checker](https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/src/kernelbench/kernel_static_checker.py)) |
| TritonBench ([arXiv 2502.14752](https://arxiv.org/abs/2502.14752)) | Triton | best one-shot execution accuracy 23.9 % (G, o1), 45.8 % (T, R1) | of R1's errors on G: name / reference 35.8 %, attribute / type 33.1 %, runtime / logic 21.9 %, syntax 9.3 % ([html](https://arxiv.org/html/2502.14752)) | — (not the perf suite [meta-pytorch/tritonbench](https://github.com/meta-pytorch/tritonbench)) |
| GEAK ([arXiv 2507.23194](https://arxiv.org/abs/2507.23194)) | Triton on MI300 | 54.9 % exec accuracy, 2.59x on TritonBench-revised; ROCm bench 63 % but 0.92x | 0-shot frontier models 7–14 % exec; GPT-4.1 0 % on ROCm Triton | ~6 tests per kernel risk false positives (authors); restart after `max_perf_debug_num` failed fixes |
| TritonRL ([arXiv 2510.17891](https://arxiv.org/abs/2510.17891)) | Triton, RL (Qwen3-8B) | L1 valid / correct / fast_1 100 / 88 / 41 | L2 correctness 28 % | torch delegation (`extern_kernels.convolution` + Triton bias add), hard-coded constants, identity copies, omitted math → linter + torch-call rules + LLM judge; under it AutoTriton L1 87 → 57 %, KernelLLM 34 → 20 % ([html](https://arxiv.org/html/2510.17891)) |
| AutoTriton ([arXiv 2507.05687](https://arxiv.org/abs/2507.05687)) | Triton, SFT + RL (8B) | KernelBench compile / exec: L1 83 / 36, L2 97 / 45, L3 82 / 20 | compiles but wrong | Triton for the easy part + PyTorch fallback, kernels never invoked → rule reward requiring `@triton.jit` |
| Kevin-32B ([arXiv 2507.11948](https://arxiv.org/abs/2507.11948)) | CUDA, multi-turn RL | 82 % correct, 1.10x mean | — | try/except fallback, inheriting the reference → zero reward for `try` / `except` / `pass`; KernelBench recycled the reference output (fixed by running the kernel first) ([blog](https://cognition.com/blog/kevin-32b)) |
| CUDA-L1 ([arXiv 2507.14111](https://arxiv.org/abs/2507.14111)) | CUDA, contrastive RL | 3.12x mean (A100) | 120x / 64x outliers exploit degenerate tasks | extra streams in 82 / 250 (32.8 %) solutions, lazy outputs materialised in `allclose`, shrunk hyperparameters, results cached by input address → sync all streams, plain-Tensor check, R1 reward checker ([html](https://arxiv.org/html/2507.14111)) |
| Sakana AI CUDA Engineer / robust-kbench ([arXiv 2509.14279](https://arxiv.org/abs/2509.14279), [repo](https://github.com/SakanaAI/robust-kbench)) | CUDA | claimed ~150x kernel ran 3x slower: it reused the eager result's memory ([TechCrunch](https://techcrunch.com/2025/02/21/sakana-walks-back-claims-that-its-ai-can-dramatically-speed-up-model-training/)); re-evaluated 1.13x → 0.82x, 63 → 22 tasks ([arXiv 2510.03760](https://arxiv.org/html/2510.03760)) | ~40 contaminated KernelBench tasks (near-constant outputs) | multi-shape, multi-init, forward + backward; LLM soft verifiers flag 73–82 % ([html](https://arxiv.org/html/2509.14279)) |
| AutoKernel ([arXiv 2603.21331](https://arxiv.org/abs/2603.21331)) | Triton / CUDA | RMSNorm 5.29x eager on H100; matmul 278 vs cuBLAS 800+ TFLOP/s | — | five correctness stages; "the benchmark is never modified by the agent" |
| SOL-ExecBench ([arXiv 2603.19173](https://arxiv.org/abs/2603.19173)) | Python / Triton / CUDA (CUTLASS, CuTe DSL, cuTile) | 235 Blackwell problems vs speed of light | — | **14.5 %** of submissions: precision downgrade 6.4 %, monkey-patching 3.3 %, stream injection 2.5 %, cached outputs 1.6 % → locked clocks, L2 scrub, pointer-shifting allocator, input clones, subprocess ([html](https://arxiv.org/html/2603.19173)) |
| Hacker-Fixer ([arXiv 2606.08960](https://arxiv.org/html/2606.08960)) | CUDA / Triton | attack success 62–76 % → 0 % after fixes | — | 15 classes incl. input mutation, `torch.empty` scavenging, TF32 flag flips, `sys._getframe`, constant-within-tolerance outputs |
| KernelBench-Verified ([arXiv 2607.16241](https://arxiv.org/html/2607.16241)) | CUDA | GPT-5.5 1.43x → 0.88x with a TF32 baseline and hidden ×3 / ×0.01 / ×−1 inputs | a shape-guarded ReLU returning its input "374x" (all-positive tests) | 28 % of correct kernels raised peak memory |
| Dr. Kernel ([arXiv 2602.05885](https://arxiv.org/html/2602.05885v2)) | Triton, RL (14B) | L2 Fast@1 59.8 % vs Claude-4.5-Sonnet 50.0 % | "lazy optimisation": correct, trivial kernels on non-bottleneck ops | instruments Triton launches; hacked solutions spend 0.014 % of time in generated kernels vs 86 % genuine; hacking ~20 → 3 % |
| KernelLLM ([HF](https://huggingface.co/facebook/KernelLLM)) | Triton (8B) | KernelBench-Triton L1 pass@1 20.2 | "incorrect API references and syntax errors" | — |
| KernelFalcon / KernelAgent ([blog 1](https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/), [blog 2](https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/)) | Triton | 100 % correct on 250 KernelBench tasks (4 parallel workers); 1.56x vs torch.compile on L1 with NCU → roofline → beam search | trusts its LLM-written tests | `@triton.jit` code cannot call torch |
| CudaForge ([arXiv 2511.01884](https://arxiv.org/html/2511.01884v1)) | CUDA | o3 coder + judge with NCU: 97.6 % correct, 1.68x (o3 alone 57.6 %, 0.68x) | — | "fake kernels" in CUDA-L1 output; 24 curated NCU metrics beat all (84 vs 80 % fast_1) |
| PIKE ([arXiv 2511.16964](https://arxiv.org/html/2511.16964v1)) | Triton / CUDA | exploit-heavy evolution 2.88x vs explore-heavy 2.17x; without the error-fixing agent 1.98x; best solutions 23 Triton / 6 CUDA | — | removed tasks where CUDA graphs gave ~1000x |
| TritonForge ([arXiv 2512.09196](https://arxiv.org/html/2512.09196v2)) | Triton | profiling raises success 22.8 → 42.3 % | 28.8 % unusable (compile / runtime) | — |
| RealisticTritonBench ([arXiv 2608.12004](https://arxiv.org/html/2608.12004)) | Triton, 31 tasks from vLLM / SGLang / PyTorch PRs, end to end | best 25.8 % success, 0.96x mean e2e | 64.7 % Triton constraint / API / memory misuse, 29.4 % wrong semantics, 25.2 % regress despite passing unit tests | — |

### 3.2 What fails most in Triton generation

1. **Compiles, but wrong**: AutoTriton compiles 83–97 % and is correct on 20–45 %;
   TritonBench-G best 23.9 %; RealisticTritonBench 29 % semantic / edge-case errors.
   Ours: 4 of 4 Triton quick checks failed, 2 on rounding semantics (TF32, `addmm`
   single rounding), one on a gross index bug (§4.3).
2. **API and type hallucination, Triton constraints**: 69 % of TritonBench-G
   errors; 65 % in RealisticTritonBench; KernelLLM's stated limit. Our mitigation:
   verified examples and local Triton sources (`triton/language/core.py`).
3. **Not real Triton**: PyTorch fallbacks or delegation (AutoTriton −30 points
   under a strict verifier, Dr. Kernel ~20 % early in RL).
4. **Correct but slow**: direct prompting 0.52–0.61x (GEAK), 0.96x end to end
   (RealisticTritonBench), 57 % of TritonForge attempts without gain.
5. **Platform mismatch**: 0 % on ROCm Triton for GPT-4.1 (GEAK); for us: H100 /
   B200 tile configs and tutorials that exceed sm_120's smem or require cc 10.

### 3.3 Published hacks vs kernel-agent's evaluator

kernel-agent today (README "What correct means", "Anti-gaming guards", "Independent
re-check", memcheck): fallback detection by entrypoint calls and
`custom_kernel_share`; plain-Tensor outputs and aliasing; timed-output re-check on
redrawn inputs; re-runs at fresh addresses, in-place redraws (normal, uniform,
Laplace, log-normal); side-stream / thread detection by wall vs event time and a
profiled pass; identity snapshots of timers, torch functions, backend flags (TF32,
cuDNN, SDPA), reference weights; parent-side comparison; separate-process re-check
of winners; precision tiers; compute-sanitizer memcheck before integration.

| hack | seen in | kernel-agent | gap |
|---|---|---|---|
| PyTorch fallback, kernel never called, reference inherited | KernelBench, Kevin, AutoTriton, TritonRL, CudaForge | fallback check (entrypoint calls, `custom_kernel_share`) | — |
| try/except around a broken kernel | Kevin | caught at run time when the fallback path runs on the dominant case | a fallback taken only on other cases passes by design ("fall back only for shapes you do not support") |
| no-op reading the reference's output memory | Sakana, KernelBench v0.1 | reference outputs come from the capture / a live call; re-run at fresh addresses | — |
| side streams, threads | CUDA-L1 (32.8 %), SOL (2.5 %) | wall vs event time + profiled pass | — |
| lazy tensor subclass | CUDA-L1, Hacker-Fixer | plain-Tensor rule | — |
| timer / comparator / flag patching, TF32 flips | SOL (3.3 %), Hacker-Fixer | identity snapshot incl. backend flags | — |
| caching by address / shape / call count | CUDA-L1, SOL (1.6 %) | rotation of 3 copies, redraws, fresh addresses | — |
| input mutation | Hacker-Fixer | side effects compared on changed elements | — |
| precision downgrade | SOL (6.4 %) | tiers; FP8 / FP4 only when the spec allows | — |
| value-conditional shortcuts (all-positive tests, ReLU "374x") | KernelBench-Verified | redraws from each tensor's mean / std and mixed distributions | **no scaled (×3, ×0.01) or sign-flipped (×−1) redraws**: relevant to FP8 scale handling and overflow |
| higher peak memory | KernelBench-Verified (28 % of correct kernels) | e2e peak memory reported; OOM in the perceptual gate is a step (#137) | **no module-level peak-memory delta** |
| lazy optimisation (correct, irrelevant op) | Dr. Kernel | targets ranked by profiled share and ceilings | — |
| degenerate tasks (constant outputs) | METR, robust-kbench, CUDA-L1 | real captured inputs and models | — |

### 3.4 Search strategies with evidence

* **Depth over breadth** at a fixed budget: Kevin 16 × 8 turns 1.10x / 82 % vs 128
  × 1 0.65x / 76 %; GEAK 13 → 44 % over 19 iterations; KernelBench R1 36 → 72 %
  with 10 turns; Dr. Kernel +10.6 points with sequential test-time scaling. Our
  slices are serial sessions with notes, which matches.
* **Execution feedback first, curated profiler feedback second**: TritonForge 22.8
  → 42.3 % with profiling; CudaForge's 24 NCU metrics beat the full set;
  KernelAgent's NCU-guided beam search beat 8 rounds without hardware feedback
  (1.95 vs 3.20 ms on a matvec).
* **Exploit-heavy evolution with a fixer**: PIKE 2.88x vs 2.17x, 1.98x without the
  error-fixing agent (our rule: a failed attempt is a bug, not evidence against
  the idea, `prompts.py` "# Ideas").
* **Plain sweeps beat agents on tuning** (InferenceBench 11.5x vs 8.1x, already in
  `playbook.md`); our §1 tile sweeps found the configs agents had found by hand.
* **RL reward design** (for later): gate rewards on validity (TritonRL), clip
  speedups (TritonRL at 2, CUDA-L1 at k = 1.5), reward the profiled share of
  runtime (Dr. Kernel).

## 4. Our evidence: the VoxCPM2 ledgers by backend

Tabulated with a script over copies of `results.tsv` and the `history/` snapshots
of R1, R2, R3 (scratchpad `tabulate.py`). The backend is classified from the
snapshot's source (`@triton.jit`, `load_inline`, `T.prim_func`, `cutlass.cute`),
not from the label the agent gave it. Library priors (re-evaluations of an earlier
run's winner) are excluded; a `re-evaluated` row replaces the speedup of its
snapshot (e.g. R1 `attn_fused`: 46.9x before the #6/#7 hardening, 18.2x after).

### 4.1 Kernel targets by backend

| backend (by source) | ledger labels | evaluations | correct | kept | quick checks failed | best module speedup |
|---|---|---|---|---|---|---|
| CUDA C++ (`load_inline`) | cuda, tilelang+cuda | 73 | 64 (88 %) | 42 | 2 of 3 | 18.2x `attn_fused` (R1); FP8: 12.2x `dit_layer_fp8` (R2) |
| CUDA C++ + Triton in one candidate | cuda+triton | 4 | 4 | 3 | — | 9.64x `vae_decoder__reduced` (R3) |
| Triton | triton | 6 | 5 (83 %) | 4 | 4 of 4 | 7.88x `vae_decoder__reduced` (R3) |
| TileLang | — | 0 | — | — | — | — |
| CuTe DSL | — | 0 | — | — | — | — |

Per run: R1 CUDA 11 evaluations (9 correct, 6 kept) + 1 hybrid; R2 CUDA 33 (29, 18),
no Triton; R3 CUDA 29 (26, 18), hybrid 3 (3, 3), Triton 6 (5, 4). Candidate files
written (including local-only experiments): CUDA 161, Triton 10, hybrid 7, TileLang
and CuTe 0. All three "tilelang+cuda" snapshots of `dit_layer__fp8_w8a8` do
`import tilelang` only for `os.path.join(os.path.dirname(tilelang.__file__),
"3rdparty", "cutlass", "include")` (e.g. `history/003_cuda_w8a8_v1_2b143116.py`).

Planner choices (`plan.json`, `rounds/*/plan.json`): backends `['triton', 'cuda']`
for R3's `dit_layer`, `vae_decoder`, `lm_decode_mlp`, `loc_enc_decode`; `['cuda',
'cute']` for R1's `lm_step_megakernel`; `['cuda', 'nvrtc']` for R2 round 2. The
engineers started in Triton only for `dit_layer` and `vae_decoder`.

| run | target | backends tried | evals | correct | best |
|---|---|---|---|---|---|
| R1 | attn_fused | CUDA | 3 | 3 | 18.19x (re-evaluated) |
| R1 | mlp_fused (M ≤ 32 GEMV/GEMM) | CUDA, CUDA+Triton | 3 | 3 | 1.05x (105 % of DRAM SOL) |
| R1 | locenc_fused | CUDA | 3 | 2 | 9.15x |
| R1 | decoder_layer_fused / rmsnorm_fused | CUDA | 1 / 1 | 1 / 1 | 5.82x / 4.32x |
| R2 | dit_layer_fp8 | CUDA | 12 | 11 | 12.24x |
| R2 | lm_step_fp8 | CUDA | 5 | 5 | 10.53x |
| R2 | enc_dit_stack_fp8 | CUDA | 8 | 7 | 10.19x (lost end to end: OOM in the gate, #137) |
| R3 | dit_layer (bf16 exact) | Triton → CUDA | 14 | 13 | Triton 2.21x → CUDA 3.09x |
| R3 | dit_layer__fp8_w8a8 | CUDA (+ cuBLASLt) | 10 | 7 | 6.24x |
| R3 | loc_enc_decode (M = 80, FP8 weights) | CUDA | 8 | 8 | 8.13x |
| R3 | vae_decoder (fp32 strict) | Triton | 3 quick | 0 | — (TF32 rounding) |
| R3 | vae_decoder__reduced | Triton → CUDA+Triton | 8 | 7 | Triton 7.88x → hybrid 9.64x |

FP4 targets (`lm_step_fp8__fp4_weights` 16.18x, `lm_stack_fp4`) are left out of the
recommendations (no 4-bit, #131).

### 4.2 Triton inside transforms (systems agent, R3)

| transform | what | e2e ms per audio s | source |
|---|---|---|---|
| `triton_fp8_locdit_gemm` | e4m3 `tl.dot`, one tile per (N, K), custom_op; gate\|up 41.2 → 36.3 us vs `_scaled_mm` | 9.03 → 8.73 | `-retest2/transforms/NOTES.md` #16 |
| `triton_locdit_attention` | single-tile attention, Sq/Sk ≤ 16, GQA, bf16 P; routed from SDPA by a `TorchFunctionMode`; 15.9 (cuDNN) → 5.7 us | 8.50 → 8.00 | #18 |
| `split_k_locdit_gemm` | 2-way split-K, 64×64 tiles for N = 1024 (48 → 192 programs on 70 SMs); down 26.3 → 21.7 us | 8.00 → 7.72 | #19 |
| `fused_gate_up_silu` | two accumulators, silu·up epilogue | 7.716 → 7.709 (noise) | #20 |
| `split_k_..._fused_reduce` | reduce fused by Inductor | 7.713 (noise) | #21 |

Inductor-compiled (i.e. Triton-generated) glue was part of the biggest steps in all
runs: R3 `graph_inductor_cfm_solver` 20.62 → 17.30, `vae_channels_last` 16.96 →
14.90, `fused_lm_step` 13.99 → 13.06; R1 `compile_*` / `autotune_dit_estimator`
(876.7 → 847.8 ms with max-autotune Triton templates). Inductor's limit showed in
R3: it functionalises the KV-cache `index_put` into ~60 whole-cache copies
(`-retest2/NOTES.md`, slice 2), so the cache write stays eager.

The CUDA W8A8 layer kernel and the Triton transform stack are a tie on the same
LocDiT layers: swapping `dit_layer__fp8_w8a8` (6.24x module, direct cuBLASLt
tensor-wise + CUDA glue) in for the seven Triton/FP8 transforms measured −0.6 %
(95 % CI −0.8 .. −0.5 %, rejected; `-retest2/report.md`, integration). Integrated
next to them (it also covers the LocEnc and prefill layers) it gave 7.7 → 6.6 ms.

### 4.3 Where Triton won and lost, and why

Won:
* **Shape-tuned e4m3 GEMM at M = 352** inside a CUDA graph: beat row-wise
  cuBLASLt by 2–12 % per GEMM because cuBLASLt's tiles leave SMs idle (24 output
  tiles for N = 1024 on 70 SMs). It did not beat cuBLASLt *tensor-wise*, which the
  transform could not use with row-wise scales; §1 shows both Triton and row-wise
  sit on the half-rate `QMMA.F32`.
* **≤ 16-token attention**: one program per (sequence, head) with Q/K/V in
  registers; FlashAttention's 128-row tiles are mostly padding at S = 11
  (`-retest2/NOTES.md` #8: flash 35 us, cuDNN 16 us).
* **Split-K for under-filled N = 1024 GEMMs** and **GEMM-shaped conv** (VAE
  channels-last tconv as one GEMM with cond + Snake prologue, RU as dw kernel +
  GEMM with residual epilogue: 7.09x → 7.88x, `vae_decoder__reduced/NOTES.md`).

Lost:
* **Host overhead in eager-timed modules.** `dit_layer` triton_v1 (cuBLAS GEMMs +
  5 Triton glue kernels) measured 2.21x with ~216 us host vs ~167 us GPU per call;
  the same math behind one `load_inline` C++ entry measured 2.70x (host ~87 us)
  (`-retest2/targets/dit_layer/NOTES.md`, exp 21/23). Playbook: Triton ~43 us per
  launch vs 19 us for `load_inline` (`knowledge/playbook.md` §5).
* **bf16 GEMMs at M = 352**: a Triton `tl.dot` sweep did not beat cuBLAS (qkv 25.9
  vs 25.5 us, down 50.5 vs 39.2); the exact tier then needs cuBLAS's own split-K
  boundaries (CUTLASS 2.x serial split-K reproduced them, 3.09x).
* **Fused halo stencil + GEMM**: a one-kernel Triton residual unit at C = 128/256
  ran 150–800 ms (255 registers, spills); `ruk_triton` with K-chunking 98 ms total;
  "Triton is the wrong tool for the C ≥ 64 fused RU". A CUDA kernel with a padded
  bf16 smem halo, `mma.sync` and pre-packed B fragments via `__ldg` reached 9.64x
  (`vae_decoder__reduced/NOTES.md`, slices 20–34).
* **Strict fp32 references with TF32**: Triton's `input_precision="tf32"` truncates
  while cuDNN rounds to nearest (`cvt.rna`): relative L2 0.0177, 47 % of elements
  outside 1e-4; `"ieee"` still failed the element check vs the TF32 reference
  (`vae_decoder/NOTES.md`, exp 50/51). Needs a reduced tier or an fp32 reference.
* **FP8 W8A8 layer as a module kernel**: "Triton ≈ row-wise" against nvjet's 207
  TFLOP/s at gate|up (`dit_layer__fp8_w8a8/NOTES.md`, slice 33–38): the engineer
  went to direct cuBLASLt from C++ (host 130 → 62 us, 3.46x → 5.42x).
* Never tried in Triton: M = 16 / M = 80 FP8 decode GEMMs (CUDA examples and the
  8.13x LocEnc kernel), persistent megakernels.

## 5. What this means for kernel-agent

### 5.1 Backend per target class (sm_120)

| target class | first backend | why (evidence) | second |
|---|---|---|---|
| Compute-bound FP8 GEMM (M ≳ 128) | block-scaled MMA: CuTe DSL / CUTLASS SM120 block-scaled GEMM with fused epilogue; Triton `tl.dot_scaled` (unit or MX scales) for quick wins | `QMMA.F32` is capped at 208 TFLOP/s, `QMMA.SF` runs 416 (§1.1); baseline to beat is cuBLASLt tensor-wise (nvjet) with scales applied in the consumer | direct cuBLASLt from C++ (descriptors cached, top-8 algos timed): 5.42x layer in R3 |
| Small-M GEMV / skinny GEMM (M ≤ 32, weights streamed) | CUDA C++ (`load_inline`), bundled FP8 skinny / GEMV examples | at DRAM bandwidth, 19 us host; R2/R3 all CUDA (10.5x, 8.13x) | Triton `tl.dot` BM = 16 under graphs (774 GB/s, §1.2) |
| Attention ≤ 16 tokens | Triton single-tile kernel as SDPA custom_op | 5.7 vs 15.9 us cuDNN (R3) | CUDA WMMA inside a fused CUDA layer (4.8–6.8 us, R3 `dit_layer`) |
| Conv / VAE (channels-last, large activations) | Triton for conv-as-GEMM with prologue/epilogue fusion | 7.88x; fast to write | CUDA for fused stencil + GEMM with smem halo (9.64x) |
| Fused decoder layer (norm, RoPE, attention, 4 GEMMs) | eager-timed module: CUDA C++ with one launcher; inside graphs: Inductor + library or block-scaled GEMMs | host time (2.21x vs 2.70x); Inductor glue gave the large e2e steps | CuTe DSL persistent layer (#133, #134) |

TileLang: no evidence either way on this GPU (never written by an agent; its
sm_120 block-scaled MMA landed upstream in #3099); keep it listed, not first.
Triton is not first where a kernel needs warp specialisation or a hand-built
persistent pipeline on sm_120, or must match cuBLAS's summation order (exact tier).

### 5.2 Prompt and knowledge changes

Done in this PR:
* `knowledge/triton.md`: sm_120 section (instruction rates, `tl.dot_scaled` with
  unit scales, TMA, no warp specialisation, shared-memory limit, M = 16 tiles,
  TF32 truncation, tail-row epilogue OOB, host overhead, short-sequence attention
  and its routing) and the library techniques worth copying.
* `knowledge/sources.md`: Triton tutorials and sm_120 issues, Triton libraries,
  CuTe DSL / TileLang / ThunderKittens references and the agent-evaluation papers
  of §3, all on the WebFetch allowlist (`agent/web.py`). Cited here but outside it,
  proposed for the allowlist in the PR rather than added:
  `images.nvidia.com` (architecture whitepapers: per-SKU tensor rates by
  accumulator) and `research.colfax-intl.com` (CUTLASS / CuTe tutorials, incl. an
  sm_120 block-scaled GEMM).

Proposed (separate PRs, see the issues in 5.3):
* `knowledge/low_precision.md`, "The FP8 peak on sm_120": replace "Triton ~55 % of
  peak" with the `QMMA.F32` / `QMMA.SF` explanation and the `tl.dot_scaled` recipe
  (#132's write-up).
* `knowledge/cute_dsl.md` / `cuda.md`: compute-bound FP8 GEMMs on sm_120 use the
  block-scaled MMA (CuTe DSL `MmaMXF8Op`; CUTLASS
  `SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e4m3_t, float_e4m3_t, float,
  float_ue8m0_t, 32>`, commented "MMA.SF" in `cute/arch/mma_sm120.hpp`; PTX
  `mma.sync ... kind::mxf8f6f4.block_scale`) with unit ue8m0 scales when the real
  scales are per tensor / channel / token, not `SM120_16x8x32_TN` (#133).
* Planner prompt (`prompts.py`, "Put the backend most suited to the op first"):
  replace the one-line hint with the class table of 5.1; for eager-timed targets
  with ≥ 3 launches per call say "one C++ launcher".
* Engineer prompt: for a Triton FP8 GEMM, start from `tl.dot_scaled`; never use
  `warp_specialize` on sm_120; graph-time GEMMs on L2-cold weights (R3 slice 8
  showed eager loops give wrong verdicts).

### 5.3 Proposed follow-up issues (not opened)

1. **Triton FP8 example on the block-scaled MMA (`tl.dot_scaled` with unit ue8m0
   scales).** Switch `examples/triton_fp8_w8a8_gemm.py` from `tl.dot` to
   `tl.dot_scaled(x, 127s, "e4m3", w.T, 127s, "e4m3", acc)` (`QMMA.SF`, full
   rate); selftest that the output stays bit-identical to `_scaled_mm`; re-sweep
   its `CONFIGS` (pointer and TMA). Expected at M = 352: gate|up 36.4 → ~31.7 us,
   o_proj 14.2 → 11.9. Rewrite low_precision.md's "FP8 peak on sm_120" around
   `QMMA.F32` (208) vs `QMMA.SF` (416). Feeds #132.
2. **FP8 GEMMs for CUDA C++ / CuTe DSL on the block-scaled MMA atom.** A verified
   CuTe DSL (or CUTLASS SM120 block-scaled) FP8 GEMM at the LocDiT shapes with
   unit or MX scales and a fused epilogue (row/col scales, silu·up, residual,
   e4m3 quantise of the output); target: nvjet tensor-wise (gate|up 30.2 us) or
   better *with* the epilogue fused. Update `cute_dsl.md` / `cuda.md`: never
   `QMMA.F32` for compute-bound FP8 on GeForce. Feeds #133; blockwise / MX scaling
   for #132 rides on the same instruction.
3. **Instruction-level peaks in the ceilings table and `doctor`.** Measure the
   `mma.sync` rates per dtype / accumulator / block-scale (the §1.1
   microbenchmark, ~1 s) next to the GEMM peaks; show both in the profile and
   tell agents which instruction a kernel needs to pass the half-rate cap.
   `doctor` checks that Triton lowers `tl.dot_scaled` to `QMMA.SF` and that TMA
   descriptors compile, and records the Triton version.
4. **Planner and engineer backend policy by target class.** Replace the one-line
   backend hint in the planner prompt with the table of 5.1; add to the engineer
   prompt: "eager-timed target with ≥ 3 launches per call: one C++ launcher";
   "Triton FP8: start from `tl.dot_scaled`; no `warp_specialize` on sm_120".
   Record backend choice and outcome per target in the library (with #133), and
   add a per-backend table (§4.1, this note's script) to `report.md` / `status`.
5. **Cheap Triton launches for eager-timed module targets.** A verified example
   that launches several Triton kernels with low host cost: cached
   `CompiledKernel.run()` (Unsloth `triton_launch.py`), or Triton AOT
   (`tools/compile.py`) kernels called from one `load_inline` entry. Goal: close
   the 2.21x vs 2.70x gap of R3 `dit_layer` without rewriting kernels in CUDA.
6. **Evaluator: scaled and sign-flipped redraws, module peak memory.** Add ×3,
   ×0.01 and ×−1 redraws of floating-point inputs to the perturbed check
   (KernelBench-Verified), with tier-aware bounds (FP8 targets: scales must
   follow); report the candidate's peak-memory delta vs the reference per case and
   warn above a threshold (28 % of correct kernels raised peak memory there; our
   perceptual-gate OOM, #137).
7. **≤ 16-token attention as a verified Triton example.** Turn R3's
   `triton_locdit_attention` (single tile, GQA, bf16 P, `TorchFunctionMode`
   routing from SDPA into a custom_op that traces under `fullgraph=True`) into an
   example + selftest; optional RoPE-on-load (R3's open idea, ~1.5 %).
8. **Persistent tuned-config cache.** Cache Triton (and CUTLASS / cuBLASLt algo)
   tile choices per (GPU, library version, op, shape bucket) across evaluations
   and runs, like FlagGems' `LibTuner` SQLite store, so sweeps and the
   integration reuse them; invalidate on a Triton / driver upgrade.
