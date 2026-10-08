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

## Examples and sources

* Sources: the `documentation-sources` skill's `sources.md`, sections "CUDA C++ and PTX".
* Per-family sources: the `## Sources` section of [gpus.md](gpus.md).
