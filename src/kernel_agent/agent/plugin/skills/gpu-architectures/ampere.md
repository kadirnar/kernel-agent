# Ampere (sm_80: A100, A30; sm_86: A10, A40, RTX A6000, GeForce RTX 30xx)

Part of the `gpu-architectures` skill. The `[ampere]` section of [gpus.md](gpus.md) is the
short version. Nothing here has run on an Ampere GPU yet. Each fact says how it was checked:

* **compiled** for sm_80 / sm_86 on the CPU (nvcc 13.4, NVRTC 13.0, Triton 3.8, TileLang
  0.1.15, CuTe DSL 4.8);
* **run** with the sm_86 code path on an RTX 5070 Ti through PTX JIT (correctness only,
  [measure-first.md](measure-first.md));
* **measured** on the RTX 5070 Ti (labelled so);
* or **NVIDIA's documentation** (Sources).

Scripts and raw output: `docs/research-scripts/older-gpus-259/`. The toolchain block's
measured numbers count before every table here. Follow [measure-first.md](measure-first.md)
on the GPU itself.

## First decisions

* **bf16 and fp16 GEMMs:** `ldmatrix` + `mma.sync m16n8k16` + a `cp.async` multi-stage
  pipeline. Triton `tl.dot` (pipelined when the pointers are 16-byte aligned) and TileLang
  `T.gemm` both compile to it for sm_80 / sm_86.
* **fp32 models:** TF32 (`mma m16n8k8 .tf32`) changes the precision. Triton's fp32
  `tl.dot` uses TF32 by default; `input_precision="ieee"` keeps fp32 (FMA). PyTorch 2.14
  defaults: `torch.backends.cuda.matmul.allow_tf32=False` (fp32 matmuls stay fp32) and
  `torch.backends.cudnn.allow_tf32=True` (cuDNN convolutions may use TF32).
* **No FP8 tensor cores** (e4m3 `mma` and `cvt` need sm_89). `fp8_w8a8` and `fp8_mx` are
  refused. Weight-only FP8 dequantises in software, which equals torch's decoding on all
  256 codes (compute_80 and compute_86 PTX runs). Triton refuses `fp8e4nv` below sm_89; it accepts `fp8e5`
  and `fp8e4b15`, but converts both to fp16 before an fp16 `mma`.
* **8-bit compute is INT8** (`int8_w8a8`): `mma.sync m16n8k32 .s32.s8.s8.s32` (IMMA),
  Triton `tl.dot` on int8 (pipelined), `torch._int_mm` (cuBLASLt).
* **Boards of one arch differ too.** [skus.md](skus.md) has the SKU classes (A100; A10 /
  A40 / RTX A6000; GeForce RTX 30xx) and the NVIDIA A10, with NVIDIA's datasheet numbers.
* **sm_80 and sm_86 differ** (Ampere tuning guide):

  | | sm_80 (A100) | sm_86 |
  |---|---|---|
  | shared memory per SM / per block | 164 / 163 KB | 100 / 99 KB |
  | L1 + shared memory per SM | 192 KB | 128 KB |
  | warps / blocks per SM | 64 / 32 | 48 / 16 |
  | fp32 operations per clock per SM | 1x | 2x |

  A tile config tuned on an A100 can raise `OutOfResources` on an A10. Check `metadata.shared`
  against the toolchain block's shared memory per block.
* **L2:** 40 MB on the A100, with persistence controls (an access policy window per
  stream: the CUDA Programming Guide's "L2 Cache Control"). The GA102 die has 6 MB: the
  Ada tuning guide gives AD102's 98304 KB as 16x GA102's. The toolchain block reports the
  board's L2. A weight matrix larger than the L2 streams from DRAM on every call, so time
  decode GEMVs with a cold L2 there.

## Instructions (compiled with NVRTC 13.0; `ptx_forms.txt`)

sm_80 and sm_86 accept the same forms:

| form | sm_80 / sm_86 | minimum |
|---|---|---|
| `mma.sync.aligned.m16n8k16.row.col.f32.{f16,bf16}...f32`, `.f16.f16.f16.f16` (fp16 accumulation) | yes | sm_80 |
| `mma ... m16n8k8 .tf32` and `.bf16` | yes | sm_80 |
| `mma ... m16n8k32 .s32.s8.s8.s32`, `m16n8k64 ... s4` | yes | sm_80 |
| Turing's `m16n8k8` f16 and `m8n8k16` s8 | yes (see Rates) | sm_75 |
| `ldmatrix`, `movmatrix` | yes | sm_75 |
| `cp.async.{ca,cg}.shared.global` + `commit_group` / `wait_group`, `mbarrier.init` | yes | sm_80 |
| `redux.sync`, `cvt.rn.bf16x2.f32`, `fma.rn.bf16x2` | yes | sm_80 |
| e4m3 / e5m2 `mma`, `cvt.rn.f16x2.e4m3x2`, `cvt.rn.satfinite.e4m3x2.f32` | no | sm_89 |
| `cp.async.bulk` (TMA), `griddepcontrol` (PDL) | no | sm_90 |

## Backends on sm_80 / sm_86

* **Triton 3.8** (the `triton-kernels` skill's `sm75-sm89.md`):
  * fp16 / bf16 `tl.dot` becomes `m16n8k16` with `ldmatrix`, plus `cp.async` at
    `num_stages=3`;
  * fp32 becomes TF32 `m16n8k8` by default;
  * int8 becomes `m16n8k32` s8, pipelined;
  * `fp8e4nv` is refused.
* **CuTe DSL 4.8:** targets `sm_80`, `sm_86` and `sm_87`, and `cute_rmsnorm.py` compiles
  for each. Its FP8 example is sm_89+ by its `ARCHS`.
* **TileLang 0.1.15:** the fp16 and bf16 `T.gemm` template becomes
  `tl::mma_sync<..., 16, 8, 16>` with `ldmatrix` and `cp_async` (`backends_per_arch.txt`).
* **PyTorch 2.14 SDPA:** flash, cuDNN and memory-efficient attention all cover sm80+, bf16
  included (`libtorch_cuda.so` messages and kernels).
* **cuBLASLt:** no FP8 below 8.9 (cuBLAS 13.4: FP8 types "with Ada and Hopper GPUs
  (compute capability 8.9 and above)"). IMMA takes the regular "TN" layout, or the COL32
  orders (`CUBLASLT_ORDER_COL32` with `COL4_4R2_8C` or, on Ampere, `COL32_2R_4R4` for B).

## Rates

`mma.sync` rates, register-only kernel (`mma_peaks` keys `f16_f32`, `f16_f16`, `bf16_f32`, `tf32_f32`, `s8_s32`; RTX 5070 Ti row from `mma_rates.py`):

| GPU | f16, fp32 acc | f16, fp16 acc | bf16, fp32 acc | tf32 m16n8k8 | s8 m16n8k32 |
|---|---|---|---|---|---|
| RTX 5070 Ti (sm_120, measured) | 103.1 | 206.3 | 103.5 | 51.4 | 405.9 TOPS |
| A100, A10, A40, RTX A6000, RTX 30xx | not measured on this family yet: `kernel-agent doctor --remeasure-peaks` measures every column | | | | |

The f16 and bf16 columns are `m16n8k16`. On the RTX 5070 Ti the Turing shapes reach half
(`m16n8k8`: 51.4) and a quarter (`m8n8k16`: 98.2 TOPS) of these. From sm_80 on, prefer
`m16n8k16` and `m16n8k32`. No Ampere GPU has measured the two shapes yet; `doctor
--remeasure-peaks` measures both (`f16_f32_k8`, `s8_s32_k16`).

**GeForce vs datacenter.** Whether fp32-accumulating HMMA runs at half the fp16-accumulating
rate is a property of the board. After `kernel-agent doctor --remeasure-peaks` the
toolchain block states it ("fp32-accumulating HMMA at N % of fp16-accumulating here"):

* GeForce Turing runs it at half rate (NVIDIA, [turing.md](turing.md));
* the RTX 5070 Ti (GeForce Blackwell) measured 103 vs 206;
* no allowlisted NVIDIA page gives the ratio for the A10, A40, RTX A6000 or GeForce RTX 30xx.
  [skus.md](skus.md) cites the A10 datasheet and the GA102 whitepaper: full rate on
  datacenter GA102, half rate on GeForce RTX 30xx.

If the board halves it, fp16 accumulation pays where the tier allows it. The `int8_w8a8`
gain is likewise the measured `s8_s32 / bf16_f32` ratio of the board: the A100's datasheet
ratio (624 / 312) is evidence for the A100 only.

## Datasheet (NVIDIA; labelled, secondary to the measured peaks)

| GPU | cc | SMs | memory | bandwidth | power | dense tensor peaks | L2 |
|---|---|---|---|---|---|---|---|
| A100 (40 GB) | 8.0 | 108 | 40 GB HBM2 | 1555 GB/s | 400 W | fp16 / bf16 312 (fp32 acc), TF32 156, INT8 624 TOPS, FP64 tensor 19.5; fp32 (no tensor cores) 19.5 | 40 MB |
| A30 | 8.0 | – | 24 GB HBM2 | 933 GB/s | 165 W | – | – |
| A10 | 8.6 | – | 24 GB GDDR6 with ECC | – | 140 W (vWS guide; 150 W in the datasheet, [skus.md](skus.md)), passive, single-slot FHFL | – | – |
| A40 | 8.6 | – | 48 GB GDDR6 with ECC | – | 300 W, passive, dual-slot | – | – |
| RTX A6000, GeForce RTX 3090 | 8.6 | – | – | – | – | – | – |

A dash means no allowlisted NVIDIA page gives the number. NVIDIA's datasheet and
whitepaper numbers for the A100 80 GB, A10, A40 and RTX 3090 (www.nvidia.com, outside the
agents' allowlist) are in [skus.md](skus.md). The toolchain block measures the SMs, L2,
memory, bandwidth and peaks of the board at hand.

## Power and memory

* **Power:** the A10 is a passively cooled card, 140 W in NVIDIA's vWS sizing guide and 150 W
  in its datasheet ([skus.md](skus.md)). The A40 is rated 300 W and the A100 400 W.
  `doctor`'s sustained line measures 2 s of GEMM load next to the burst peaks
  ([measure-first.md](measure-first.md), steps 2 and 5).
* **Memory:** 24 GB (A10, A30), 48 GB (A40), 40 GB (A100 in the table). Plan with the
  toolchain block's memory and the measured baseline peak. A second weight copy or in-process
  A/B doubles the weights.

## Verified with the sm_86 code path (PTX JIT on the RTX 5070 Ti; correctness only, performance not measured)

| path | rel L2 error vs reference |
|---|---|
| `cuda_int8_skinny_gemm.py` (INT8 W8A8, IMMA `m16n8k32`), M = 3 / 16 | 0.0097 / 0.0096 |
| `cuda_int8_gemv.py` (int8 weights), M = 1 / 4 | 0.0045 / 0.0046 |
| `cuda_fp8_gemv.py` (e4m3 weights, software conversion), M = 1 / 4 | 0.0259 / 0.0256 |
| `triton_int8_w8a8_gemm.py` (`m16n8k32` s8, `cp.async`) | 0.0091 |
| Triton fp16 / bf16 `tl.dot` GEMM (`m16n8k16`, `cp.async`) | 6.6e-7 / 4.8e-7 |
| `triton_short_attention.py` vs fp32 SDPA, fp16 / bf16 | 0.0003 / 0.0020 |
| e4m3 -> fp16 conversion (`__nv_cvt_fp8_to_halfraw`, compute_80 / compute_86) | equal to torch on all 256 codes |

## Not measured on this family yet

* every `mma.sync` rate in the Rates table, the fp32 vs fp16 accumulation ratio and the INT8
  : bf16 ratio of an A10 or RTX 30xx: run `kernel-agent doctor --remeasure-peaks`;
* the sustained SM clock of an A10 under a long GEMM: run `kernel-agent doctor
  --remeasure-peaks` (its sustained line), or `sustained_clocks.py` for 10 s;
* tile configs on 99 KB of shared memory (sm_86) at the board's SM count: sweep them there.

## Sources

* https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/ — A100: 108 SMs, dense tensor peaks (fp16 / bf16 312, TF32 156, INT8 624), 40 GB HBM2 at 1555 GB/s, 40960 KB L2, 400 W, 1024 dense fp16 FMA per clock per SM.
* https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html — shared memory, L1, warps and blocks per SM for 8.0 vs 8.6; 2x fp32 per clock on 8.6; L2 persistence; asynchronous copy; BF16 / TF32 HMMA.
* https://docs.nvidia.com/cuda/ada-tuning-guide/index.html — AD102's 98304 KB L2 is 16x GA102's.
* https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html — L2 set-aside and access policy windows for persisting accesses.
* https://docs.nvidia.com/vgpu/sizing/virtual-workstation/latest/gpus-vws.html — A10: 24 GB GDDR6 with ECC, 140 W, passive, single-slot; A40: 48 GB, 300 W, passive.
* https://developer.nvidia.com/blog/accelerating-ai-inference-workloads-with-nvidia-a30-gpu/ — A30: 24 GB HBM2, 933 GB/s, 165 W.
* https://developer.nvidia.com/cuda/gpus — compute capability 8.0: A100, A30; 8.6: A10, A40, RTX A6000, GeForce RTX 3090.
* https://docs.nvidia.com/cuda/parallel-thread-execution/index.html — `mma`, `ldmatrix`, `cp.async` and their minimum targets.
* https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul — IMMA layouts; FP8 from compute capability 8.9.
