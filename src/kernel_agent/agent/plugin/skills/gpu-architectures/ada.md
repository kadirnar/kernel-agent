# Ada Lovelace (sm_89: L4, L40, L40S, RTX 6000 Ada, GeForce RTX 40xx)

Part of the `gpu-architectures` skill. The `[ada]` section of [gpus.md](gpus.md) is the short
version. Nothing here has run on an Ada GPU yet. Each fact says how it was checked:

* **compiled** for sm_89 on the CPU (nvcc 13.4, NVRTC 13.0, Triton 3.8, TileLang 0.1.15,
  CuTe DSL 4.8);
* **run** with the sm_89 code path on an RTX 5070 Ti through PTX JIT (correctness only,
  [measure-first.md](measure-first.md));
* **measured** on the RTX 5070 Ti (labelled so);
* or **NVIDIA's documentation** (Sources).

Scripts and raw output: `docs/research-scripts/older-gpus-259/`. The toolchain block's
measured numbers count before every table here. Follow [measure-first.md](measure-first.md)
on the GPU itself.

## First decisions

* **FP8 tensor cores (QMMA):** `mma.sync m16n8k32` on e4m3 / e5m2 with fp32 or fp16
  accumulation, and the hardware conversions `cvt.rn.f16x2.e4m3x2` and
  `cvt.rn.satfinite.e4m3x2.f32`. `fp8_w8a8` runs here. `fp8_mx` does not: there is no
  block-scaled MMA, and Triton's `tl.dot_scaled` becomes a bf16 `mma` (emulation, compiled
  for sm_89).
* **Triton FP8 GEMM:** `tl.dot` on e4m3 becomes QMMA, pipelined with `cp.async` when the
  pointers are 16-byte aligned. The launched `triton_fp8_w8a8_gemm.py` GEMM under emulated
  sm_89 had 24 `cp.async`, 32 KB of shared memory, and a rel L2 error of 0.0368 against
  `nn.Linear`. Its `gemm_ptx` helper compiles with the alignment hints a launch gets, and
  shows the same 24 `cp.async` (the `triton-kernels` skill's `sm75-sm89.md`).
* **fp32 vs fp16 accumulation is the SKU question.** `kernel-agent doctor --remeasure-peaks`
  measures `e4m3 QMMA.F32` and `e4m3 QMMA.F16`, fp16 HMMA with both accumulations, bf16,
  TF32 and s8 on the board itself. The toolchain block then states "fp32-accumulating HMMA
  at N % of fp16-accumulating here".
  * The RTX 5070 Ti (GeForce Blackwell) measured 206.6 vs 413.5 TFLOP/s for these two;
    there, fp32 accumulation is the half-rate path.
  * cuBLASLt's FP8 matmul on Ada takes only `CUBLAS_COMPUTE_32F` (cuBLAS 13.4).
  * Where the board halves fp32 accumulation, an INT8 W8A8 kernel (`s8 IMMA`) or fp16
    accumulation can be the faster 8-bit path. The board's measured rates decide.
* **cuBLASLt FP8 on 8.9** (cuBLAS 13.4):
  * tensor-wide scaling only (`SCALAR_32F`); outer-vector (row-wise) and 128-element block
    scaling are 9.0 modes;
  * "TN" layout only;
  * compute type `CUBLAS_COMPUTE_32F`, scale type `CUDA_R_32F`.

  Row-wise scales go in an epilogue or a Triton / CUTLASS kernel ([gpus.md](gpus.md):
  `torch._scaled_mm` row-wise runs a CUTLASS sm_89 kernel).
* **Shared memory and occupancy** (Ada tuning guide):
  * 128 KB of L1 + shared memory per SM, with shared carveouts of 0, 8, 16, 32, 64 or
    100 KB;
  * up to 99 KB per block;
  * 48 warps and 24 blocks per SM;
  * twice sm_80's fp32 operations per clock per SM.
* **L2:** 98304 KB on the AD102 die, 16x GA102 (Ada tuning guide). The toolchain block
  reports the L2 of the board at hand; size L2-resident working sets with that number.
* **No TMA, clusters or PDL:** `cp.async.bulk` and `griddepcontrol` need sm_90 (ptxas).
  Pipeline with `cp.async` and cut launch gaps with CUDA graphs.

## Instructions (compiled with NVRTC 13.0; `ptx_forms.txt`)

| form | sm_89 | minimum |
|---|---|---|
| `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`, `.e5m2`, `.f16.e4m3.e4m3.f16` (fp16 accumulation) | yes | sm_89 |
| `cvt.rn.f16x2.e4m3x2`, `cvt.rn.satfinite.e4m3x2.f32` | yes | sm_89 |
| every sm_80 form: `m16n8k16` f16 / bf16, `m16n8k8` tf32, `m16n8k32` s8, `ldmatrix`, `cp.async`, `mbarrier.init`, `redux.sync` | yes | sm_80 |
| `cp.async.bulk` (TMA), `griddepcontrol` (PDL) | no | sm_90 |

## Backends on sm_89

* **Triton 3.8** (the `triton-kernels` skill's `sm75-sm89.md`):
  * e4m3 and e5m2 `tl.dot` become QMMA `m16n8k32` with `cp.async`;
  * `fp8e4b15` is converted to fp16 (not QMMA);
  * fp16 / bf16 become `m16n8k16`, and int8 becomes `m16n8k32` s8;
  * `tl.dot_scaled` is a bf16 `mma`.
* **CuTe DSL 4.8:** an `sm_89` target; `cute_rmsnorm.py` compiles for it.
  `examples/cute_fp8_decoder_block.py` is declared sm_89+.
* **TileLang 0.1.15:** fp16 / bf16 `T.gemm` becomes `tl::mma_sync<..., 16, 8, 16>` with
  `ldmatrix` and `cp_async` (`backends_per_arch.txt`).
* **PyTorch 2.14 SDPA:** flash, cuDNN and memory-efficient attention cover sm89, bf16
  included.

## Rates

`mma.sync` rates, register-only kernel (`mma_peaks` keys `f16_f32`, `f16_f16`, `bf16_f32`, `s8_s32`, `e4m3_f32`, `e4m3_f16`; RTX 5070 Ti row from `mma_rates.py`):

| GPU | f16, fp32 acc | f16, fp16 acc | bf16 | s8 m16n8k32 | e4m3, fp32 acc | e4m3, fp16 acc |
|---|---|---|---|---|---|---|
| RTX 5070 Ti (sm_120, measured) | 103.1 | 206.3 | 103.5 | 405.9 TOPS | 206.6 | 413.5 |
| L4, L40S, RTX 40xx | not measured on this family yet: `kernel-agent doctor --remeasure-peaks` measures every column | | | | | |

## Datasheet (NVIDIA; labelled, secondary to the measured peaks)

| GPU | cc | memory | power | notes |
|---|---|---|---|---|
| L4 | 8.9 | 24 GB GDDR6 with ECC | 72 W, passive, PCIe 4.0 low-profile single-slot | FP8 support |
| L40S | 8.9 | 48 GB GDDR6 with ECC | 350 W, passive, PCIe 4.0 dual-slot FHFL | 568 fourth-generation tensor cores, 142 third-generation RT cores |
| L40, RTX 6000 Ada, GeForce RTX 4090 / 4080 | 8.9 | – | – | – |

No allowlisted NVIDIA page gives these boards' bandwidth or tensor peaks. The L4's datasheet
numbers (www.nvidia.com, outside the agents' allowlist) are in [skus.md](skus.md). The
toolchain block measures them.

## Power and memory

* **Power:** the L4 is a 72 W card (NVIDIA's L4 blog and vWS sizing guide); the L40S is
  rated 350 W. `doctor`'s sustained line measures 2 s of GEMM load next to the burst
  peaks ([measure-first.md](measure-first.md), steps 2 and 5).
* **Memory:** 24 GB (L4), 48 GB (L40S). Plan with the toolchain block's memory and the
  measured baseline peak.

## Verified with the sm_89 code path (PTX JIT on the RTX 5070 Ti; correctness only, performance not measured)

| path | rel L2 error vs reference |
|---|---|
| `triton_fp8_w8a8_gemm.py` (e4m3 QMMA `m16n8k32`, `cp.async`), M = 704 | 0.0368 |
| `triton_int8_w8a8_gemm.py` (`m16n8k32` s8, `cp.async`) | 0.0091 |
| Triton fp16 / bf16 `tl.dot` GEMM (`m16n8k16`, `cp.async`) | 6.6e-7 / 4.8e-7 |
| `triton_short_attention.py` vs fp32 SDPA, fp16 / bf16 | 0.0003 / 0.0020 |
| e4m3 -> fp16 conversion (compute_89: hardware `cvt.rn.f16x2.e4m3x2` in the PTX) | equal to torch on all 256 codes |

## Not measured on this family yet

* the e4m3 fp32 vs fp16 accumulation ratio and the s8 rate of an L4, L40S or RTX 40xx: run
  `kernel-agent doctor --remeasure-peaks`;
* Triton's FP8 W8A8 GEMM against cuBLASLt's tensor-wise FP8 (`torch._scaled_mm`) at the
  model's shapes: time both on the board;
* the sustained SM clock of a 72 W L4: run `kernel-agent doctor --remeasure-peaks` (its
  sustained line), or `sustained_clocks.py` for 10 s.

## Sources

* https://docs.nvidia.com/cuda/ada-tuning-guide/index.html — L2 98304 KB on AD102 (16x GA102); L1 / shared memory carveouts; 99 KB per block; 48 warps and 24 blocks per SM; 2x fp32 per clock vs 8.0; fourth-generation tensor cores with FP8.
* https://docs.nvidia.com/vgpu/sizing/virtual-workstation/latest/gpus-vws.html — L4: 24 GB GDDR6 with ECC, 72 W, passive, low-profile single-slot; L40S: 48 GB, 350 W, passive, 568 tensor cores, 142 RT cores.
* https://developer.nvidia.com/blog/supercharging-ai-video-and-ai-inference-performance-with-nvidia-l4-gpus/ — L4: 24 GB GDDR6, 72 W, FP8 support.
* https://developer.nvidia.com/cuda/gpus — compute capability 8.9: L4, L40, L40S, RTX 6000 Ada, GeForce RTX 4090 / 4080.
* https://docs.nvidia.com/cuda/parallel-thread-execution/index.html — FP8 `mma` and `cvt` from sm_89, `cp.async.bulk` and `griddepcontrol` from sm_90.
* https://docs.nvidia.com/cuda/cublas/index.html#narrow-precision-data-types-usage — FP8 from compute capability 8.9; Scaling Mode Support Overview: tensorwide 8.9+, outer vector and 128-element blocks 9.0.
* https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul — FP8: "TN" on Ada, `CUBLAS_COMPUTE_32F`, `CUDA_R_32F` scale type.
