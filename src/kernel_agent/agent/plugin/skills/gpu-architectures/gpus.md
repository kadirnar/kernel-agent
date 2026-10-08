# GPU architectures: what is fast, what to avoid

One section per architecture family; every prompt gets the section of its GPU
(`kernel_agent/gpu_arch.py` picks it from the compute capability). Shared memory per block
is the opt-in maximum (CUDA Programming Guide, compute capabilities). Facts checked
2026-10 against the sources at the end. Numbers measured by kernel-agent itself come from
an RTX 5070 Ti (sm_120) unless a section says otherwise: on another GPU, the toolchain
block's measured peaks and instruction rates are the numbers to use.

## [pre_ampere] Volta / Turing (sm_70, sm_75: V100, T4, RTX 20xx)

* Tensor cores: fp16 `mma.sync` only (int8 on sm_75); **no bf16, TF32 or FP8 tensor
  cores**. A bf16 model's matmuls do not get tensor-core rates here; fp16 does.
* Triton's `tl.dot` falls back to FMA (CUDA cores) below sm_80: use Triton for
  memory-bound glue, cuBLAS or CUDA C++ fp16 `mma.sync` for GEMMs.
* No `cp.async`, TMA, clusters or PDL. Shared memory: 64 KB per block on sm_75, 96 KB on
  sm_70. CUDA 13 dropped sm_70 / sm_72 (Turing is its minimum).
* Weight-only FP8 / FP4 and FP8 KV caches still run (software dequantisation, as on
  Ampere); W8A8 and MXFP8 do not.
* INT8 tensor cores on sm_75 (IMMA, `mma.sync` m8n8k16 s8, 2x the fp16 rate): `int8_w8a8`
  runs there through cuBLASLt (`torch._int_mm`) or CUDA C++ (Triton's `tl.dot` has no
  tensor cores below sm_80; the `m16n8k32` form needs sm_80). sm_70 has no INT8 MMA:
  `int8_w8a8` is refused there; `int8_weights` runs everywhere.

## [ampere] Ampere (sm_80: A100, A30; sm_86 / sm_87: A10, A40, RTX A6000, RTX 30xx, Orin)

* Tensor cores: `mma.sync` m16n8k16 bf16 / fp16, tf32, int8 (HMMA). The fast path is
  `ldmatrix` + `mma.sync` with a `cp.async` multi-stage pipeline (Triton `num_stages` 3-4).
* **No FP8 tensor cores**: `fp8_w8a8` / `fp8_mx` are refused on this GPU. Weight-only
  formats pay where weights stream (decode GEMVs, skinny GEMMs): store e4m3 (or NVFP4)
  codes and dequantise to bf16 in registers. There is no hardware e4m3 conversion before
  sm_89: CUDA's `__nv_cvt_fp8x2_to_halfraw2` falls back to a software conversion (correct;
  a few integer ops per value, hidden under the DRAM time of a GEMV), and Triton has no
  e4m3 type below sm_89 (load the codes as `uint8` and convert with bit operations). vLLM
  runs FP8 checkpoints on Ampere the same way (weight-only W8A16, FP8 Marlin kernels).
* **INT8 is the 8-bit compute class here**: `int8_w8a8` (IMMA `mma.sync` m16n8k32 s8 with
  int32 accumulation at twice the bf16 rate: A100 624 vs 312 TOPS dense; Triton `tl.dot` on
  int8 tiles, `torch._int_mm`), and `int8_weights` for weight streams (int8 codes convert
  in two ops per value, no e4m3 emulation). Activation outlier channels need SmoothQuant
  or bf16 for those GEMMs (low_precision.md, "INT8").
* No TMA, no clusters, no PDL (`griddepcontrol` needs sm_90): CUDA graphs are the way to
  cut launch gaps.
* Shared memory: 163 KB per block on sm_80 (and sm_87), 99 KB on sm_86. L2: 40 MB on A100
  (with L2 residency control); the toolchain block has this GPU's.

## [ada] Ada Lovelace (sm_89: RTX 40xx, RTX 6000 Ada, L4, L40S)

* Tensor cores: `mma.sync` bf16 / fp16 / tf32 / int8 **and FP8** e4m3 / e5m2
  (`mma.sync ... m16n8k32 ... e4m3.e4m3.f32`, SASS `QMMA`). FP8 with fp32 accumulation
  runs at twice the fp16 rate with fp32 accumulation (SageAttention2, Table 1). **No
  block-scaled MMA**: MXFP8 (`fp8_mx`) is refused here, and Triton's `tl.dot_scaled`
  is emulated through bf16 (never use it on sm_89).
* W8A8 FP8: Triton `tl.dot` on e4m3 (native on sm_89), cuBLASLt tensor-wise scales
  (row-wise / blockwise modes are sm_90-only in cuBLASLt; `torch._scaled_mm` row-wise
  runs a CUTLASS sm_89 kernel), CUTLASS's Ada blockwise example from C++.
* The FP8 accumulator of `mma ... f32.e4m3.e4m3.f32` keeps fewer bits than fp32 (about
  FP22: SageAttention2) on long K: promote partial sums to fp32 every few K blocks when
  the tolerance tier is tight.
* INT8 `mma.sync` (IMMA) runs at the rate of FP8 with fp16 accumulation; GeForce Ada runs
  FP8 with fp32 accumulation at half of it, so INT8 W8A8 (`int8_w8a8`) can be the faster
  8-bit path there where the activations suit it: the toolchain block's measured `s8 IMMA`
  and `e4m3 QMMA.F32` rates say.
* No TMA, clusters or PDL: `cp.async` pipelines, CUDA graphs for launch gaps.
  Shared memory 99 KB per block; large L2 (the AD102 die has 96 MB; products enable less:
  the toolchain block has this GPU's).

## [hopper] Hopper (sm_90: H100, H200, GH200, H20)

* Tensor cores: **`wgmma`** (warpgroup MMA, operands from shared memory, asynchronous;
  `sm_90a` only) reaches the peak; `mma.sync` reaches only about 2/3 of it, and FP8
  `mma.sync` is emulated through fp16 HMMA on sm_90 (Triton's own lowering notes).
  FP8 wgmma runs at twice bf16. Hand-written compute-bound kernels: CuTe DSL
  (`cute.nvgpu.warpgroup`), CUTLASS 3.x sm_90 kernels, Triton `tl.dot` (it emits
  wgmma on sm_90) or cuBLAS(Lt); never a plain `mma.sync` GEMM.
* TMA (`cp.async.bulk.tensor`, multicast across a cluster), thread-block clusters with
  distributed shared memory, warp specialisation (producer warpgroup on TMA, consumer
  warpgroups on wgmma, `setmaxnreg`), PDL. Shared memory 227 KB per block, L2 50 MB.
* **No block-scaled MMA**: MXFP8 (`fp8_mx`) is refused here and `tl.dot_scaled` is
  emulated. Fine-grained FP8 scaling is DeepSeek-style blockwise (1 x 128 activation
  tiles, 128 x 128 weight blocks; cuBLASLt `VEC128_32F` / `BLK128x128_32F` and row-wise
  `OUTER_VEC_32F` modes exist on sm_90 only).
* FP8 wgmma accumulates with fewer bits than fp32 (DeepSeek-V3: about 14 bits): promote
  to fp32 every 128 along K (4 wgmmas) for long reductions.
* FlashAttention-3 (Hopper-only, wgmma + TMA, FP8 attention) is the reference attention.
* INT8 (`int8_w8a8`): `wgmma` on s8 with int32 accumulators at the FP8 rate (Triton
  `tl.dot` on int8, CUTLASS sm_90); a `mma.sync` s8 kernel stops below it.

## [blackwell] Blackwell, datacenter (sm_100 / sm_103 / sm_110: B200, GB200, B300, GB300)

* Tensor cores: **`tcgen05.mma`** with accumulators in tensor memory (TMEM), issued by
  one thread for the CTA, 2-CTA pairs (`cta_group::2`) and TMA multicast; FP8 and the
  block-scaled formats (MXFP8, MXFP4, NVFP4: `kind::mxf8f6f4.block_scale`, scale vectors
  per 16 / 32) at full rate; NVFP4 runs at twice FP8. **No `wgmma`** (sm_90a only).
  `mma.sync` still compiles but saturates near a quarter of the B200 peak, and FP8
  `mma.sync` is emulated through fp16 HMMA: never hand-write a `mma.sync` GEMM here
  (the bundled sm_120 CuTe GEMM uses `mma.sync`).
* Paths to the peak: cuBLASLt (tensor-wise FP8, MXFP8 `VEC32_UE8M0`, NVFP4 `VEC16_UE4M3`;
  all layouts, not just TN), Triton `tl.dot` (e4m3: `tcgen05.mma kind::f8f6f4`, full
  rate) and `tl.dot_scaled` (`kind::mxf8f6f4.block_scale` with 128-row tiles; 64-row
  tiles fall back to `kind::f16`, an upcast: Triton 3.8, compiled for sm_100 here), CuTe
  DSL `cute.nvgpu.tcgen05` (CUTLASS's Blackwell CuTe DSL examples: dense, persistent,
  block-scaled GEMMs), CUTLASS sm_100 kernels.
* Shared memory 227 KB per block (deeper pipelines and larger tiles than the 99 KB of the
  sm_120 examples), clusters, PDL. L2: 126 MB on GB200.
* INT8 (`int8_w8a8`): `tcgen05.mma kind::i8` on sm_100 (B200: INT8 at the FP8 rate).
  **Blackwell Ultra (sm_103: B300, GB300) keeps INT8 at about 1/30 of its FP8 rate**, the
  PTX ISA exposes no `kind::i8` there (only warp-level IMMA) and CUTLASS generates no INT8
  kernels for it: prefer FP8 W8A8 on sm_103 unless INT8 is measured faster.

## [blackwell_geforce] Blackwell GeForce / RTX PRO / DGX Spark (sm_120 / sm_121)

Most of kernel-agent's own measurements are from this family (RTX 5070 Ti): the guides'
sm_120 sections hold them.

* Tensor cores: `mma.sync` only (CUTLASS's sm_120 GEMMs use nothing else): bf16 / fp16
  HMMA, FP8 QMMA, and **block-scaled `mma.sync`** (`kind::mxf8f6f4.block_scale`,
  `kind::mxf4nvf4`; `sm_120a` / `sm_121a`). No `wgmma`, no `tcgen05` / TMEM.
* **FP8 with fp32 accumulation runs at half rate on GeForce** (RTX 5090 whitepaper: 419
  vs 838 TFLOP/s with fp16 accumulation); the block-scaled `QMMA.SF` with unit ue8m0
  scales runs at the full rate and gives the same result (RTX 5070 Ti: 208 vs 416
  TFLOP/s; RTX 5090: 510 vs 1014, flashinfer #5963). Compute-bound FP8: the block-scaled
  instruction (Triton `tl.dot_scaled` with unit scales, CuTe DSL `MmaMXF8Op`); the
  toolchain block's measured rates say whether this GPU (RTX PRO cards included) differs.
* INT8 `mma.sync` (`IMMA.16832.S8.S8`, int32 accumulation) runs at the full rate: 410
  TOPS on an RTX 5070 Ti, as fast as the block-scaled FP8 instruction and twice plain e4m3
  with fp32 accumulation, so a plain Triton `tl.dot` on int8 reaches it (`int8_w8a8`).
* TMA works without multicast (CUTLASS fixes the cluster shape to 1x1x1); PDL works.
  Shared memory 99 KB per block: Hopper / B200 tile configs do not fit (Triton raises
  `OutOfResources`). cuBLASLt has tensor-wise FP8 and MXFP8 here, but no row-wise or
  128-blockwise modes. Triton's `warp_specialize` was slower here (RTX 5070 Ti).

## [newer] Newer than this table

The family is not described here yet: trust the toolchain block (measured peaks and
`mma.sync` rates), run `kernel-agent doctor` (its probes say whether `tl.dot_scaled`,
TMA and PDL work) and look the architecture up (the `documentation-sources` skill's `sources.md`: the CUDA Programming
Guide's compute capabilities and the PTX ISA's target notes) before choosing an
instruction.

## Sources

* https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html — shared memory per SM / block, TMA / cluster / DSMEM support per compute capability.
* https://docs.nvidia.com/cuda/parallel-thread-execution/index.html — target notes: `mma.sync` sm_80+, e4m3 `mma` sm_89+, `wgmma` sm_90a, `tcgen05` sm_100a / sm_103a / sm_110a, block-scaled `mma.sync` sm_120a, `griddepcontrol` sm_90+, FP8 wgmma accumulation below fp32.
* https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html , https://docs.nvidia.com/cuda/ada-tuning-guide/index.html , https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html , https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html — L2 sizes and per-architecture tuning.
* https://docs.nvidia.com/cuda/cublas/index.html — "Scaling Mode Support Overview": tensor-wise FP8 8.9+, outer-vector / 128-block 9.0, MXFP8 / NVFP4 10.0+; TN layout on 8.9 / 9.0 / 12.x.
* https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ScaledBlas.cpp — `torch._scaled_mm` needs sm_89 or sm_90+; row-wise via CUTLASS on sm_89; blockwise sm_90 only.
* https://github.com/triton-lang/triton/blob/main/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp — `tl.dot`: wgmma on sm_90, tcgen05 on sm_100-119, `mma.sync` on sm_120; FP8 `mma.sync` emulated on sm_90 / sm_100; native `tl.dot_scaled` on sm_100-119 and sm_12x; FMA below sm_80.
* https://github.com/triton-lang/triton/blob/main/third_party/nvidia/backend/compiler.py — e4m3 (`fp8e4nv`) only from sm_89.
* https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md — sm_100 vs sm_120 GEMMs; GeForce: no multicast, cluster 1x1x1.
* https://pytorch.org/blog/flashattention-3/ — `mma.sync` reaches about 2/3 of the Hopper tensor-core peak.
* https://arxiv.org/abs/2609.15311 — `mma.sync` saturates near 66 % of the H100 and 25 % of the B200 per-cycle tensor throughput.
* https://arxiv.org/abs/2411.10958 — SageAttention2: FP8 rates vs fp16 on Ada / Hopper, the FP22 accumulator of FP8 `mma`.
* https://arxiv.org/abs/2412.19437 — DeepSeek-V3 §3.3: FP8 accumulation (14 bits), promotion every 128 K.
* https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/quantization/llm_compressor/fp8.md — vLLM: W8A8 FP8 needs sm_89+; FP8 checkpoints run weight-only (W8A16, FP8 Marlin) from sm_75.
* https://github.com/flashinfer-ai/flashinfer/issues/5963 — GeForce Blackwell: FP8 fp32-accumulate half rate, block-scaled MMA with unit scales at full rate.
* https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/ — B200 / B300 dense FP8 and NVFP4 rates.
* https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/ — A100: INT8 tensor cores at 624 TOPS dense, twice bf16's 312 TFLOP/s; IMMA `mma.sync` m16n8k32.
* https://arxiv.org/abs/2608.11693 — INT8 on Blackwell Ultra (B300, sm_103): ~30:1 FP8 to INT8 dense rate (1:1 on H200 / B200), no `tcgen05.mma kind::i8` on sm_103a, no CUTLASS INT8 kernels for it.
* https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html — CUDA 13 drops Maxwell / Pascal / Volta; SM101 renumbered SM110.
