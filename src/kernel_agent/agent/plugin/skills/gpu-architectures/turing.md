# Turing (sm_75: T4, GeForce RTX 20xx)

Part of the `gpu-architectures` skill. The `[pre_ampere]` section of [gpus.md](gpus.md) is the
short version. Nothing here has run on a Turing GPU yet. Each fact says how it was checked:

* **compiled** for sm_75 on the CPU (nvcc 13.4, NVRTC 13.0, Triton 3.8, TileLang 0.1.15,
  CuTe DSL 4.8);
* **run** with the sm_75 code path on an RTX 5070 Ti through PTX JIT (correctness only,
  [measure-first.md](measure-first.md));
* **measured** on the RTX 5070 Ti (labelled so);
* or **NVIDIA's documentation** (Sources).

Scripts and raw output: `docs/research-scripts/older-gpus-259/`. The toolchain block's
measured numbers count before every table here. Follow [measure-first.md](measure-first.md)
on the GPU itself.

## First decisions

* **fp16 is the tensor-core dtype; bf16 gets no tensor cores here.** Checks on sm_75:

  | what | bf16 on sm_75 | source |
  |---|---|---|
  | `mma` / WMMA | `.bf16` needs sm_80 | ptxas: "Feature '.bf16' requires .target sm_80" |
  | Triton `tl.dot` | FMA | compiled |
  | TileLang `T.gemm` | compiles without an MMA | compiled |
  | PyTorch flash and cuDNN attention | need sm_80 | PyTorch 2.14 binary |
  | PyTorch memory-efficient attention | no bf16 kernel below sm_80 | PyTorch 2.14 binary |

  A bf16 model therefore runs its GEMMs and attention without tensor cores. Run the 16-bit
  work in fp16 once an fp16 run matches an fp32 run within the target's tier. fp16 tops out
  at 65504; bf16 reaches 3.4e38, so activations that bf16 holds can overflow in fp16.
  `--dtype auto` (the default) does this in `analyze`: a bf16 checkpoint gets the fp16
  candidate, checked once against a float32 run, and `run.json` → `dtype` says why.
* **GEMMs:** use cuBLAS fp16, CUDA C++ `mma.sync m16n8k8` with `ldmatrix`, WMMA 16x16x16
  `half`, or TileLang `T.gemm` on fp16 (it emits `m16n8k8` + `ldmatrix`). Not Triton: its
  fp16 / bf16 `tl.dot` is FMA on sm_75. Not CuTe DSL: 4.8 has no sm_75 target.
* **Triton for memory-bound kernels** (norms, elementwise ops, reductions): they need no
  tensor cores, and they ran correctly with the sm_75 code path (below).
* **INT8 W8A8 (`int8_w8a8`):** IMMA `mma.sync m8n8k16` s8 in CUDA C++, or cuBLASLt
  (`torch._int_mm`). Triton's int8 `tl.dot` does not compile for sm_75.
* **Weight-only formats** (`int8_weights`, `fp8_weights`, `fp4_weights`) dequantise in
  registers. The bundled CUDA GEMVs compile for sm_75 and ran correctly through compute_75
  PTX (below), although their `ARCHS` say sm_80+ (bf16 activations). Below sm_89 the e4m3
  codes convert in software, and that conversion equals torch's on all 256 codes.
* **No `cp.async`** (sm_80+). TileLang's sm_75 GEMM fills shared memory with plain 16-byte
  loads. In CUDA C++, load the next tile into registers while the MMAs read the current one
  from shared memory.

## Instructions (compiled for sm_75 with NVRTC 13.0; `ptx_forms.txt`)

| form | sm_75 | minimum (PTX ISA) |
|---|---|---|
| `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32` (also `.f16` accumulation) | yes | sm_75 |
| `mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32`, `m8n8k32 ... s4` | yes | sm_75 |
| `ldmatrix.sync.aligned.m8n8.x4[.trans].shared.b16`, `movmatrix` | yes | sm_75 |
| WMMA 16x16x16 `half` -> `float`, `signed char` -> `int` (`<mma.h>`) | yes | sm_70 / sm_72 |
| `mma.sync.aligned.m8n8k4` f16 (Volta's shape) | compiles | sm_70 |
| any `m16n8k16` (f16 / bf16 / s8), `m16n8k32` s8, `.bf16`, `.tf32`, bf16 WMMA | no | sm_80 |
| `cp.async`, `mbarrier.init`, `redux.sync`, `cvt.rn.bf16x2.f32`, `fma.rn.bf16x2` | no | sm_80 |
| e4m3 / e5m2 `mma` and `cvt` | no | sm_89 |

CUDA 13.4's `cuda_bf16.h` has no bf16 `__hfma` or `__hfma2` for sm_75 ("no suitable
user-defined conversion"). `__hadd`, the operators and the float conversions compile
(`nvcc_bf16_ops.txt`). Do bf16 math in fp32. fp16 `__hfma2` compiles.

## Backends on sm_75

* **Triton 3.8:** fp16, bf16 and fp32 `tl.dot` become FMA, with no `ldmatrix` and no
  `cp.async`. The int8 dot fails to compile, and e4m3 is refused. Details: the
  `triton-kernels` skill's `sm75-sm89.md`.
* **CuTe DSL 4.8:** its `Arch` targets start at sm_80. `examples/cute_rmsnorm.py` fails
  with `KeyError: 'sm_75'` (`backends_per_arch.txt`). `doctor` lists `cute` under
  `backends unavailable` with the reason.
* **TileLang 0.1.15:** compiles for sm_75 (`backends_per_arch.txt`):
  * `tilelang_rmsnorm.py` compiles;
  * the fp16 `T.gemm` template becomes
    `tl::mma_sync<kFloat16, kFloat16, kFloat32, 16, 8, 8>` with `ldmatrix` and no
    `cp.async`;
  * the bf16 one compiles without an MMA.
* **CUDA C++:** sm_75 is the oldest target CUDA 13 compiles for. NVRTC 13.0 rejects sm_70
  and sm_72 (`ptx_forms.txt`), and the CUDA 13.0 release notes drop Maxwell, Pascal and
  Volta.
* **PyTorch 2.14 SDPA** (messages and kernel names in `libtorch_cuda.so`):
  * flash and cuDNN attention support "[sm80, sm121]";
  * memory-efficient attention supports "[sm50, sm121]", with fp16 and fp32 kernels for
    sm50 / 70 / 75 / 80 and bf16 kernels for sm80 only.

  So on sm_75, fp16 attention can take the memory-efficient kernel and bf16 attention runs
  the math path.
* **cuBLASLt IMMA** (int8 -> int32; cuBLAS 13.4 docs, `cublasLtMatmul`) takes either:
  * the regular layout ("TN", 4-byte aligned pointers, m and k multiples of 4), the
    preferred one;
  * or the IMMA orders `CUBLASLT_ORDER_COL32` for A / C and `CUBLASLT_ORDER_COL4_4R2_8C`
    for B on Turing.

  `torch._int_mm` on a Turing GPU has not been measured on this family yet: run
  `kernel-agent doctor --remeasure-peaks`. Its INT8 peak calls `_int_mm`, and
  `tflops_unavailable` names the error if it fails. Where `_int_mm` raises, the INT8 W8A8
  reference (`kernels.quant.int8_matmul`) takes an exact fp32 path with the same integers.

## Rates

NVIDIA: each Turing SM has 8 tensor cores, and each runs 64 fp16 FMA per clock, so 512 per
SM. INT8 runs at twice that (Turing blog, Turing tuning guide).

`mma.sync` rates, register-only kernel (`mma_peaks` keys `f16_f32_k8`, `f16_f16_k8`, `s8_s32_k16`; RTX 5070 Ti row from `mma_rates.py`):

| GPU | f16 m16n8k8, fp32 acc | f16 m16n8k8, fp16 acc | s8 m8n8k16 |
|---|---|---|---|
| RTX 5070 Ti (sm_120, measured) | 51.4 TFLOP/s | 102.2 TFLOP/s | 98.2 TOPS |
| T4, RTX 20xx | not measured on this family yet: run `kernel-agent doctor --remeasure-peaks` | | |

The Turing shapes are slow on newer GPUs. On the RTX 5070 Ti, `m16n8k16` f16 reaches 103.1 /
206.3 TFLOP/s and `m16n8k32` s8 reaches 405.9 TOPS: twice and four times the Turing shapes.
Use `m16n8k8` and `m8n8k16` only on sm_75.

## Datasheet (NVIDIA; labelled, secondary to the measured peaks)

| GPU | cc | SMs | memory | bandwidth | power | fp16 tensor, fp16 / fp32 acc | INT8 tensor | fp32 | L2 |
|---|---|---|---|---|---|---|---|---|---|
| T4 | 7.5 | 40 (derived: 2,560 CUDA cores at 64 per SM; 320 tensor cores at 8 per SM) | 16 GB GDDR6 | 320 GB/s | 70 W, PCIe Gen3 single-slot low profile | not on an allowlisted NVIDIA page: measure | – | – | – |
| RTX 2080 Ti (Founders Edition) | 7.5 | 68 | 11 GB GDDR6 | 616 GB/s | 260 W (reference 250 W) | 113.8 / 56.9 TFLOP/s | 227.7 TOPS (INT4 455.4) | 14.2 TFLOP/s | 5.5 MB |

NVIDIA's T4 datasheet numbers (www.nvidia.com, outside the agents' allowlist) are in
[skus.md](skus.md) ("Power-capped boards").

**GeForce vs datacenter.** NVIDIA lists GeForce Turing's fp32-accumulating HMMA at half the
fp16-accumulating rate (RTX 2080 Ti, above). On a GeForce Turing board, fp16 accumulation
therefore doubles the tensor-core peak where the tolerance tier allows it. The RTX 5070 Ti
(GeForce Blackwell) shows the same 2x, measured. No allowlisted NVIDIA page gives the T4's
ratio. After `kernel-agent doctor --remeasure-peaks` the toolchain block states it
("fp32-accumulating HMMA at N % of fp16-accumulating here", from the Turing shapes).

## Power, memory, shared memory

* **Power:** the T4 is a 70 W card. NVIDIA built the Turing Tesla GPUs "to operate under 70
  Watts". `doctor`'s sustained line measures 2 s of GEMM load next to the burst peaks
  ([measure-first.md](measure-first.md), steps 2 and 5).
* **Memory:** the T4 has 16 GB. An fp32 fallback doubles the weight bytes of a bf16 / fp16
  checkpoint. Plan with the toolchain block's memory and the measured baseline peak.
* **Shared memory and occupancy** (Turing tuning guide):
  * 96 KB of L1 + shared memory per SM, carved out as 32 or 64 KB shared;
  * a block addresses all 64 KB through the dynamic opt-in (static allocations stay at
    48 KB);
  * 32 warps and 16 blocks per SM;
  * 64K 32-bit registers per SM, at most 255 per thread.

## Verified with the sm_75 code path (PTX JIT on the RTX 5070 Ti; correctness only, performance not measured)

| path | rel L2 error vs reference |
|---|---|
| `cuda_int8_gemv.py` (int8 weights), M = 1 / 4 | 0.0045 / 0.0046 |
| `cuda_fp8_gemv.py` (e4m3 weights, software conversion), M = 1 / 4 | 0.0259 / 0.0256 |
| e4m3 -> fp16 conversion (`__nv_cvt_fp8_to_halfraw`, compute_75) | equal to torch on all 256 codes |
| Triton fp16 / bf16 `tl.dot` GEMM (FMA) | 4.1e-7 / 1.8e-7 |
| `triton_rmsnorm.py` fp16 / bf16 | < 5e-5 / < 5e-5 |
| `triton_short_attention.py` (FMA) vs fp32 SDPA, fp16 / bf16 | 0.0003 / 0.0020 |

## Not measured on this family yet

* tensor-core and IMMA rates, and the fp32 vs fp16 accumulation ratio of a T4: run
  `kernel-agent doctor --remeasure-peaks`;
* `torch._int_mm` and the cuBLAS fp16 peak: run `kernel-agent doctor --remeasure-peaks`;
* the sustained SM clock under a long GEMM: run `kernel-agent doctor --remeasure-peaks`
  (its sustained line), or `sustained_clocks.py` for 10 s;
* which SDPA backend PyTorch picks for fp16: wrap the call in
  `torch.nn.attention.sdpa_kernel` with one backend at a time and time each.

## Sources

* https://developer.nvidia.com/blog/nvidia-turing-architecture-in-depth/ — RTX 2080 Ti table (SMs, fp16 tensor TFLOPS with fp16 / fp32 accumulation, INT8 / INT4, bandwidth, 5632 KB L2, TDP); 8 tensor cores per SM at 64 fp16 FMA per clock, INT8 at twice that; Turing Tesla GPUs under 70 W.
* https://docs.nvidia.com/cuda/turing-tuning-guide/index.html — shared memory per SM and per block, warps and blocks per SM, registers, WMMA shapes, integer tensor cores.
* https://developer.nvidia.com/blog/accelerating-ai-inference-workloads-with-nvidia-a30-gpu/ — T4: 16 GB GDDR6, 320 GB/s, 70 W, PCIe Gen3 single-slot low profile.
* https://developer.nvidia.com/blog/nvidia-dgx-2-server-and-new-tesla-t4-set-image-recognition-records-for-training-and-inference/ — T4: 2,560 CUDA cores, 320 tensor cores, 70 W.
* https://developer.nvidia.com/cuda/gpus — compute capability 7.5: T4, GeForce RTX 2080 Ti.
* https://docs.nvidia.com/cuda/parallel-thread-execution/index.html — `mma` shapes and their minimum targets.
* https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul — IMMA layouts (regular "TN" or COL32 / COL4_4R2_8C on Turing).
* https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html — CUDA 13 drops Maxwell, Pascal and Volta.
