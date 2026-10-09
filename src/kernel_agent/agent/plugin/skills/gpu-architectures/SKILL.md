---
name: gpu-architectures
description: What is fast and what to avoid per NVIDIA GPU family (Volta / Turing to Blackwell and newer) — tensor-core instructions, FP8 and block-scaled support, TMA / clusters / PDL, shared memory, L2, with sources. Use to choose an instruction, precision or tile size, or for another GPU than the prompt's.
---

# GPU architectures

Every agent prompt already has the section of its own GPU (`# This GPU`, picked by `kernel_agent/gpu_arch.py` from the compute capability). [gpus.md](gpus.md) has every family, one `## <family> [<key>]` section each, and the sources they rest on:

| key | family | compute capability |
|---|---|---|
| `pre_ampere` | Volta / Turing (V100, T4, RTX 20xx) | sm_70, sm_75 |
| `ampere` | Ampere (A100, A10, RTX 30xx, Orin) | sm_80, sm_86, sm_87 |
| `ada` | Ada Lovelace (RTX 40xx, L4, L40S) | sm_89 |
| `hopper` | Hopper (H100, H200, GH200, H20) | sm_90 |
| `blackwell` | Blackwell datacenter (B200, GB200, B300) | sm_100, sm_103, sm_110 |
| `blackwell_geforce` | Blackwell GeForce / RTX PRO / DGX Spark | sm_120, sm_121 |
| `newer` | a family not described yet | — |

On any GPU, the toolchain block's measured peaks and instruction rates come first; `kernel-agent doctor` probes `tl.dot_scaled`, TMA, PDL and green contexts.

Boards of one architecture differ too: [skus.md](skus.md) has what tells them apart from measurements (datacenter vs GeForce fp32 accumulation, INT8 vs bf16, the sustained clock of a power-capped board), the NVIDIA A10 (sm_86) and the power-capped A10 / T4 / L4, datasheet numbers labelled.

Turing, Ampere and Ada in depth: [turing.md](turing.md), [ampere.md](ampere.md) and [ada.md](ada.md). Each holds the instruction forms per arch, the backends' behaviour (Triton, CuTe DSL, TileLang, PyTorch SDPA, cuBLASLt), datasheet rows per SKU, the code paths verified through PTX JIT and what is not measured on the family yet. Before relying on a number on such a GPU, follow [measure-first.md](measure-first.md): doctor, the instruction rates incl. fp32 vs fp16 accumulation, sustained clocks, and running an older arch's code path on a newer GPU.

An older GPU's code paths (`__CUDA_ARCH__` branches, Triton's sm_8x / sm_75 lowering) can be checked for correctness on a newer one: `kernel-agent eval|recheck|memcheck|doctor --smoke --emulate-arch sm_86` (`KERNEL_AGENT_EMULATE_ARCH`) runs that architecture's PTX through the driver's JIT with its facts (capability, shared memory per block). Timings there are not that GPU's, cuBLAS / cuDNN / SDPA / `_scaled_mm` / `_int_mm` run natively, CuTe DSL and TileLang only compile; NVRTC code must take its target from `toolchain.nvrtc_target` (the `cuda-kernels` skill).

## Examples and sources

* Sources: the `documentation-sources` skill's `sources.md`, sections "CUDA C++ and PTX".
* Per-family sources: the `## Sources` section of [gpus.md](gpus.md).
