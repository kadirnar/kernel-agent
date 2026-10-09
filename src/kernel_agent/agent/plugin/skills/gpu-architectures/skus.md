# Boards of one architecture: datacenter vs GeForce, power-capped boards, the NVIDIA A10

`kernel_agent/gpu_arch.py` decides by compute capability: an NVIDIA A10, an A40 and a
GeForce RTX 3090 are all "Ampere, sm_86" to it, and no code matches GPU names (the AWS
A10G is another sm_86 board, with more SMs and other clocks). What differs between boards
of one architecture comes from measurements, which the toolchain block of every prompt
carries:

| what differs | measured as | toolchain block |
|---|---|---|
| fp32-accumulating HMMA rate (GeForce: half the fp16-accumulating one; datacenter: the same) | the `mma.sync` rates `f16_f32` and `f16_f16` (RTX 5070 Ti, GeForce: 104 vs 208) | "fp32-accumulating HMMA at N % of fp16-accumulating here" |
| INT8 vs bf16 tensor-core rate | `s8 IMMA.S32` vs `bf16 HMMA.F32`; the `int8` vs `bfloat16` matmul peaks | the INT8 GEMM policy row: "measured on this GPU" |
| the clock a power cap leaves under sustained load | `tflops_sustained` and `sustained` of the peaks | "sustained bf16/fp16 ..." in the peaks, "sustained load slows this GPU down" |
| SMs, L2, memory, shared memory per block | `toolchain.GPUInfo` | `GPU: ...` |

The datasheet numbers below are NVIDIA's specifications (dense math, boost clocks), not
kernel-agent measurements: they say what to expect; the toolchain block decides. Nothing
in this file was measured on the boards it describes yet.

## NVIDIA A10 (sm_86, datacenter GA102)

* Datasheet: 72 SMs, 24 GB GDDR6 (about 22 GB usable), 600 GB/s, 6 MB L2, 150 W,
  passively cooled, single slot, 1695 MHz boost; dense bf16 / fp16 125 TFLOP/s (fp32 or
  fp16 accumulation alike), TF32 62.5, fp32 31.2, INT8 250 TOPS. Shared memory 99 KB per
  block (sm_86).
* **Full-rate fp32 accumulation** (datacenter GA102, like the A40 and RTX A6000): bf16
  GEMMs get the whole tensor-core rate, fp16 accumulation buys none; keep fp32
  accumulators. A GeForce RTX 30xx runs the same bf16 code at half its fp16-accumulating
  rate.
* **INT8 W8A8 is 2x bf16** (250 vs 125 TOPS), as on the A100, not the 4x of GeForce RTX
  30xx: `int8_w8a8` halves a compute-bound GEMM's floor and its weight bytes. FP8 W8A8 and
  MXFP8 do not run (no FP8 tensor cores); weight-only FP8 / FP4 and FP8 KV caches
  dequantise in software (the Ampere section of `gpus.md`). bf16 ridge from the datasheet:
  125 TFLOP/s / 600 GB/s ≈ 208 rows per weight read.
* **6 MB L2**: the weights of a transformer layer of any sizeable model do not fit, so
  decode weight streams are DRAM bound at 600 GB/s and a warm-L2 evaluation of a
  weight-streaming module runs about as fast as a cold one. Fewer bytes per weight
  (`int8_weights`, `fp8_weights`) and fusion pay; keeping weights L2-resident between
  calls pays only for modules whose weights fit in a few MB.
* **150 W power cap**: under a model's sustained GEMM load the SM clock settles well below
  the 1695 MHz boost (`sw_power_cap`). The ceilings' compute floors use the measured
  sustained peak; a short module timing (`evaluate_candidate`) runs at a higher clock than
  the end-to-end run it is projected into ("Power-capped boards" below).
* **Verified so far**: the CUDA and Triton sm_86 code paths of the bundled examples ran
  correctly through compute_86 PTX on an RTX 5070 Ti (correctness only, not speed; FP8
  W8A8 refused). [ampere.md](ampere.md) has the error of each path, the instruction forms
  and the backends on sm_86. On an A10, run `kernel-agent doctor --remeasure-peaks` and
  `doctor --smoke` first: the peaks, the instruction rates and the sustained line are the
  facts to decide with.

## Ampere SKU classes

| board (datasheet, dense) | arch | SMs | L2 | DRAM GB/s | bf16 / fp16, fp32 acc. TFLOP/s | fp16, fp16 acc. | INT8 TOPS | power |
|---|---|---|---|---|---|---|---|---|
| A100 SXM4 80 GB | sm_80 | 108 | 40 MB | 2039 | 312 | 312 | 624 | 400 W |
| A10 | sm_86 | 72 | 6 MB | 600 | 125 | 125 | 250 | 150 W, passive |
| A40 | sm_86 | 84 | 6 MB | 696 | 149.7 | 149.7 | 299.3 | 300 W, passive |
| GeForce RTX 3090 | sm_86 | 82 | 6 MB | 936 | 71 | 142 | 284 | 350 W |

* **A100 (sm_80)**: 163 KB shared memory per block, a 40 MB L2 with residency control,
  HBM2e; INT8 2x bf16. Its bandwidth puts the bf16 ridge near 150 rows.
* **Datacenter GA102 (A10, A40, RTX A6000)**: full-rate fp32 accumulation, INT8 2x bf16,
  power-capped boards (150-300 W) with GDDR6.
* **GeForce GA102 (RTX 3080 / 3090)**: fp32-accumulating HMMA at half rate, so a bf16 GEMM
  runs at half the fp16-accumulating rate and INT8 is 4x bf16 (`int8_w8a8` pays most
  here). fp16 operands with fp16 accumulation double the rate where the tolerance tier
  allows their rounding (never for an `exact` target).

## Power-capped boards (A10 150 W, L4 72 W, T4 70 W)

| board (datasheet, dense) | arch | SMs | L2 | DRAM GB/s | 16-bit tensor TFLOP/s | 8-bit | power |
|---|---|---|---|---|---|---|---|
| T4 | sm_75 | 40 | 4 MB | 320 | fp16 65 (no bf16 tensor cores) | INT8 130 TOPS | 70 W |
| L4 | sm_89 | 58 | 48 MB | 300 | bf16 / fp16 121 | FP8 242, INT8 242 | 72 W |

* **What is measured**: after the burst peaks, `kernel-agent doctor` runs ~2 s of
  back-to-back 8192³ GEMMs per 16-bit dtype and records `tflops_sustained` with the SM
  clock, power, the enforced power limit and the clock-event reasons (`sustained`):
  `sustained bf16/fp16 ... TFLOP/s (2 s; SM ... MHz, burst ...; ... W of a ... W limit;
  sw_power_cap)` in the toolchain block. Measured on an RTX 5070 Ti (a 300 W GeForce
  board, for scale): `sustained bf16/fp16 94 / 92 TFLOP/s (2 s; SM 2670 MHz, burst 2782;
  300 W of a 300 W limit; sw_power_cap)` against 100 / 97 in the burst, 6 % below it.
* **What uses it**: the ceilings table's floors take the sustained peak when it is at
  least 10 % below the burst one (an end-to-end run is sustained load). The evaluator's
  speed of light (`pct_of_sol`) keeps the burst peak: a module timing is a burst.
* **What follows for kernels** (reasoning, not measured on these boards yet): compute-bound
  work there is bounded by power, not only by the issue rate, so math in fewer bits (INT8
  W8A8) and fused passes save energy as well as instructions. A/B timings stay valid (both
  sides run under the same cap); absolute numbers from separate processes drift with
  temperature, so compare candidates paired.

## Sources

Datasheets and whitepapers on www.nvidia.com (outside the agents' `WebFetch` allowlist):

* NVIDIA A10 datasheet: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a10/pdf/a10-datasheet.pdf
* NVIDIA Ampere GA102 architecture whitepaper (RTX 3090: fp16 tensor TFLOP/s with fp32 vs
  fp16 accumulation, INT8): https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf
* NVIDIA A40 datasheet: https://images.nvidia.com/content/Solutions/data-center/a40/nvidia-a40-datasheet.pdf
* NVIDIA T4 datasheet: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf
* NVIDIA A100: https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/
