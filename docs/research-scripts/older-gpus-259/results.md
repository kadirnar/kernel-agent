# Older GPUs (Turing, Ampere, Ada): verified facts behind the skills (issue #259)

Date 2026-10-09. Toolchain: torch 2.14.1+cu130, Triton 3.8.0, nvidia-cutlass-dsl 4.8.0,
TileLang 0.1.15, cuda.core 1.2.1 (NVRTC 13.0), nvcc 13.4. The only GPU here is an RTX 5070
Ti (sm_120, 70 SMs, driver 615.71.09). Nothing ran on a Turing, Ampere or Ada GPU:

* **compile** results are CPU-only (`CUDA_VISIBLE_DEVICES=`);
* **PTX JIT** runs executed the older arch's code path on the RTX 5070 Ti: correctness only,
  the SASS is sm_120's;
* **rates and clocks** are the RTX 5070 Ti's and are labelled so in the skills.

GPU jobs ran under `flock ~/.cache/kernel-agent/gpu.lock env KERNEL_AGENT_LOCK_HELD=1`.
The skills that use these results are `gpu-architectures/turing.md`, `ampere.md`, `ada.md` and
`measure-first.md`, `triton-kernels/sm75-sm89.md`, the `cuda-kernels` cheat sheet and the
`[pre_ampere]` / `[ampere]` / `[ada]` sections of `gpus.md`.

| script | needs | output | what it shows |
|---|---|---|---|
| `ptx_forms.py` | CPU | `ptx_forms.txt` | every PTX form (`mma` shapes and types, `ldmatrix`, `cp.async`, `mbarrier`, TMA, PDL, e4m3 `cvt`, bf16x2 ops, WMMA) per sm_75 / 80 / 86 / 89 / 120. It matches the PTX ISA minimum of every form. |
| `nvcc_bf16_ops.py` | CPU | `nvcc_bf16_ops.txt` | CUDA 13.4's `cuda_bf16.h`: bf16 `__hfma` / `__hfma2` do not compile for sm_75; `__hadd`, operators and conversions do |
| `lowering_triton.py` | CPU | `lowering_triton.txt` | Triton `tl.dot` per dtype x arch (`mma` form, `cp.async`, `ldmatrix`, shared memory), `tl.dot_scaled`, the bundled W8A8 GEMMs, and the alignment caveat |
| `backends_per_arch.py` | CPU | `backends_per_arch.txt` | CuTe DSL 4.8 targets (from sm_80), TileLang per arch (its sm_75 fp16 GEMM is `m16n8k8`), PyTorch SDPA arch ranges and kernels, TF32 defaults |
| `ptx_jit_nvrtc.py` | GPU, < 5 s | `ptx_jit_nvrtc.txt` | compute_75 / 80 / 86 / 89 PTX runs with that `__CUDA_ARCH__`; the software e4m3 conversion equals torch on all 256 codes |
| `ptx_jit_cuda.py` | CPU build, then GPU | `ptx_jit_cuda.txt` | `load_inline` with `TORCH_CUDA_ARCH_LIST=8.6+PTX` / `7.5+PTX` (compute_86 / compute_75 only) runs the INT8 / FP8 GEMV and INT8 skinny GEMM examples correctly |
| `ptx_jit_triton.py <cc>` | GPU, ~30 s each | `ptx_jit_triton.txt` | Triton examples with the sm_75 / 86 / 89 lowering; the launched FP8 W8A8 GEMM at sm_89 is pipelined (24 `cp.async`) |
| `mma_rates.py` | GPU, ~2 s | `mma_rates.txt` | RTX 5070 Ti `mma.sync` rates of every form (the measured row of the skills' rate tables) |
| `sustained_clocks.py` | GPU, ~15 s | `sustained_clocks.txt` | RTX 5070 Ti: 60 ms burst 97.1 TFLOP/s at 2902 MHz vs 92–94 at ~2680 MHz under its 300 W cap |

## Main results

1. **Triton int8 `tl.dot` does not compile for sm_75.** It fails with `PassManager::run
   failed` in `TritonGPUAccelerateMatmul`. The skills used to say it "falls back to FMA". The
   fp16, bf16 and fp32 dots do compile to FMA there, without `ldmatrix` or `cp.async`.
2. **Triton's FP8 W8A8 GEMM is pipelined on sm_89.** The audit's "no `cp.async`" came from
   `triton_fp8_w8a8_gemm.gemm_ptx`, which then compiled without the `tt.divisibility`
   attributes a launch adds for 16-byte aligned arguments (it passes them since #256);
   without them no Triton kernel gets `cp.async` (an fp16 GEMM on sm_80 neither). With
   them, and in the launched kernel under emulated sm_89, the GEMM has 24 `cp.async`.
3. **The RTX 5070 Ti halves fp32 accumulation**, measured with the register-only rate kernel:

   | form | rate |
   |---|---|
   | f16 m16n8k16, fp32 / fp16 acc | 103.1 / 206.3 TFLOP/s |
   | e4m3 m16n8k32, fp32 / fp16 acc | 206.6 / 413.5 TFLOP/s |
   | bf16 m16n8k16 | 103.5 TFLOP/s |
   | tf32 m16n8k8 | 51.4 TFLOP/s |
   | s8 m16n8k32 | 405.9 TOPS |

   The Turing shapes run at half (`m16n8k8`: 51.4 / 102.2) and a quarter (`m8n8k16` s8:
   98.2 TOPS) of the sm_80 shapes there.
4. **Other backends below sm_80:**
   * TileLang compiles an fp16 `T.gemm` for sm_75 to `m16n8k8` + `ldmatrix` (no
     `cp.async`); a bf16 one gets no MMA.
   * CuTe DSL 4.8 cannot target sm_75 (`KeyError: 'sm_75'`).
   * PyTorch 2.14 has flash and cuDNN attention for [sm80, sm121] only, and its
     memory-efficient attention has bf16 kernels for sm80 only.
5. **PTX JIT correctness** (rel L2 vs reference):

   | path | arch | rel L2 |
   |---|---|---|
   | INT8 skinny GEMM (IMMA m16n8k32) | sm_86 | 0.0097 / 0.0096 |
   | INT8 weight GEMV | sm_86 and sm_75 | 0.0045 / 0.0046 |
   | e4m3 weight GEMV (software conversion) | sm_86 and sm_75 | 0.0259 / 0.0256 |
   | Triton INT8 W8A8 | sm_86 / sm_89 | 0.0091 |
   | Triton FP8 W8A8 (QMMA) | sm_89 | 0.0368 |
   | Triton short attention (fp16 / bf16, FMA on sm_75) | sm_75 / 86 / 89 | 0.0003 / 0.0020 |

   Performance was not measured.

## What NVIDIA's allowlisted pages do not give

These are the pages kernel-agent's agents may fetch (`agent/web.py` `DOMAINS`). The T4, A10,
A40, L4, L40S, RTX 3090 and RTX 4090 tensor peaks and bandwidths (except the T4's
320 GB/s) are in NVIDIA datasheets and whitepapers on www.nvidia.com / images.nvidia.com,
outside that allowlist. The family files therefore leave those cells empty and say to
measure (`kernel-agent doctor --remeasure-peaks`, `mma_rates.py`).
