# Measure first on Turing, Ampere and Ada, and test their code paths on a newer GPU

Part of the `gpu-architectures` skill, for [turing.md](turing.md), [ampere.md](ampere.md)
and [ada.md](ada.md). Most of kernel-agent's numbers were measured on an RTX 5070 Ti
(sm_120). On another GPU these steps give that GPU's numbers before a decision rests on
them. Write a gap as "not measured on this family yet: run …", never as "not possible".
Scripts and raw results: `docs/research-scripts/older-gpus-259/` (`results.md`).

## On the GPU itself

1. **`kernel-agent doctor`** prints the toolchain block: compute capability, SMs, memory,
   L2, the opt-in shared memory per block, the measured peaks, the backends and the feature
   probes. These numbers count before any table in the skills, the datasheet rows of the
   family files included.
2. **`kernel-agent doctor --remeasure-peaks`** measures again:
   * the DRAM copy bandwidth;
   * bf16, fp16 and fp32 GEMMs (cuBLAS);
   * FP8 and INT8 GEMMs where torch has a kernel (`tflops_unavailable` says why not);
   * the launch floor;
   * the `mma.sync` rates of `kernel_agent.kernels.mma_peaks` (`mma_tflops`):
     * from sm_80: `bf16_f32`, `f16_f32`, `f16_f16` and `tf32_f32`, plus `s8_s32` in TOPS;
     * from sm_75: Turing's shapes `f16_f32_k8`, `f16_f16_k8` and `s8_s32_k16`;
     * from sm_89: `e4m3_f32` and `e4m3_f16`;
   * the sustained bf16 / fp16 peaks: 2 s of back-to-back 8192^3 GEMMs with the SM clock,
     the power and the clock-event reasons. They appear as the toolchain block's
     `sustained bf16/fp16 ...` line, and the ceilings' floors use them when they are at
     least 10 % below the burst ([skus.md](skus.md)).
3. **fp32 vs fp16 accumulation is the SKU fact.** After step 2 the toolchain block says
   "fp32-accumulating HMMA at N % of fp16-accumulating here".
   * Where that is about 50 %, fp16 accumulation pays wherever the tolerance tier allows
     it. The RTX 5070 Ti measured 103 vs 206 TFLOP/s on f16 m16n8k16.
   * NVIDIA documents the same half rate for GeForce Turing (RTX 2080 Ti: 56.9 vs 113.8
     TFLOP/s, [turing.md](turing.md)). No allowlisted NVIDIA page gives the ratio for the
     other boards, so this line is their measurement.
   * `docs/research-scripts/older-gpus-259/mma_rates.py` times every form with the same
     register-only kernel, outside the peaks cache.
4. **`kernel-agent doctor --smoke`** compiles and runs one kernel per backend and every
   bundled example whose `ARCHS` holds this GPU. It lists the rest with the reason.
5. **Sustained clocks.** The other peaks are bursts of about 60 ms per shape, and a board
   with a low power limit can settle at a lower SM clock under longer tensor-core load.
   Step 2's sustained line is the 2 s measurement. NVIDIA lists:
   * the T4 at 70 W and the L4 at 72 W;
   * the A10 at 140 W in its vWS sizing guide and 150 W in its datasheet ([skus.md](skus.md)).

   For a longer look, run `python docs/research-scripts/older-gpus-259/sustained_clocks.py`
   (10 s of bf16 8192^3 GEMMs with SM clock, power and clock-event reasons per second), or
   watch `nvidia-smi --query-gpu=clocks.sm,power.draw,power.limit,clocks_event_reasons.active
   --format=csv -lms 500` while a long GEMM runs.
   * RTX 5070 Ti (measured): a 60 ms burst ran 97.1 TFLOP/s at 2902 MHz. Under 10 s of
     load the board held its 300 W limit (`sw_power_cap`) at about 2680 MHz and
     92–94 TFLOP/s, 3–5 % below the burst.
   * The gap belongs to the board and its power limit: measure it there before a floor or
     a `pct_of_sol` relies on the burst peak.
6. **Read the code Triton made.** `compiled = kernel[grid](...)`, then
   `compiled.asm["ptx"]` holds:
   * `.target sm_XX`;
   * the `mma.sync.aligned.<shape>.<types>` forms (none means FMA on CUDA cores);
   * `cp.async` (the K loop is pipelined) and `ldmatrix`.

   A compile-only check (`triton.compile(ASTSource(fn, signature, constexprs, attrs),
   target=GPUTarget("cuda", cc, 32))`) needs
   `attrs={(arg_index,): [["tt.divisibility", 16]]}` for the pointers and sizes that a
   launch sees as 16-byte aligned. Without them no load is vectorised, and no `cp.async`
   appears even on sm_80+ (`lowering_triton.txt`, Triton 3.8).
7. **Memory.** Plan with the toolchain block's memory (what torch sees), not the
   datasheet's. Measure a baseline's peak (`torch.cuda.max_memory_allocated`) before
   adding candidates that keep a second copy of the weights or run in-process A/B.

## Testing an older GPU's code path on a newer one (PTX JIT)

A newer GPU's driver JIT-compiles older PTX. The kernel then runs with the older
`__CUDA_ARCH__` and the older Triton lowering. This checks **correctness only**. The SASS,
clocks, SM count, DRAM, L2 and the libraries (cuBLAS, cuDNN, SDPA backend choice,
`torch._int_mm`) stay the newer GPU's, so never record a time measured this way.
`--emulate-arch sm_86` on `doctor --smoke`, `eval`, `recheck` and `memcheck` (the variable
`KERNEL_AGENT_EMULATE_ARCH`, also for `pytest -m gpu`; `kernel_agent/emulate.py`) applies
all three routes in every process, with the old GPU's facts, and keeps nothing it measures.
The three routes below were verified on the RTX 5070 Ti for sm_75, sm_80, sm_86 and sm_89:

* **CUDA C++ (`load_inline`):** set `TORCH_CUDA_ARCH_LIST="8.6+PTX"` (or `"7.5+PTX"`) and a
  separate `TORCH_EXTENSIONS_DIR`. nvcc then builds `-gencode=arch=compute_86,code=sm_86`
  plus `code=compute_86`. The sm_86 SASS does not load on sm_120, so the driver
  JIT-compiles the PTX (`ptx_jit_cuda.py`).
* **NVRTC:** use `ProgramOptions(arch="compute_86")`, then `program.compile("ptx")`, then
  `cuda.core.ObjectCode.from_ptx(ptx).get_kernel(name)`. The kernel saw `__CUDA_ARCH__`
  750 / 800 / 860 / 890 (`ptx_jit_nvrtc.py`). `kernel_agent.kernels.mma_peaks.measure((7, 5))`
  runs an older GPU's rate kernels this way; the rates it returns are the newer GPU's.
* **Triton:** three patches in the process (`ptx_jit_triton.py`):
  * `triton.runtime.driver.active.get_current_target = lambda: GPUTarget("cuda", 86, 32)`;
  * `CompiledKernel._init_handles` replaces `self.kernel` (the cubin) with
    `self.asm["ptx"].encode() + b"\0"`, and refuses a kernel whose `metadata.shared` exceeds
    the older arch's limit;
  * `torch.cuda.get_device_capability` returns the old capability, so the examples'
    own checks behave as on that GPU.
