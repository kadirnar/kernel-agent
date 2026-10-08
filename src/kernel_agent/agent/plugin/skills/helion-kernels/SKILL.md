---
name: helion-kernels
description: Helion kernels (PyTorch with tiles, compiled to Triton) — the candidate contract (a plain kernel body, helion_tune.kernel in build), hl.tile loops, configs, Helion's own autotuner through sweep_candidate strategy helion, arch limits, when it runs here. Use when a target's backend is helion.
---

# Helion backend (`import helion.language as hl`)

Helion compiles "PyTorch with tiles" to Triton and searches a space it derives from each
kernel: tile sizes of every `hl.tile` loop, their order, L2 grouping of the program ids,
persistent programs, looped or persistent reductions, pointer / `block_ptr` / TMA
(`tensor_descriptor`) indexing, eviction hints, warps and stages. Its authors' numbers (blog,
B200): geomean 3.27x over eager vs 2.70x for `torch.compile` max-autotune; one matmul
searched 1,520 configs in 586 s. Whether it runs on this GPU: `kernel-agent doctor` (the
`helion` probe compiles and runs `examples/helion_rmsnorm.py`); Helion lists A100 / H100 /
B200, and the backend is unavailable, with the reason, where the probe fails.

## The candidate contract

Write each kernel body as a plain function and make the kernel in `build` with
`kernel_agent.kernels.helion_tune.kernel(fn, helion_configs)`: every built candidate gets its
own kernel, an evaluation uses Helion's default config (it never autotunes inside an
evaluation: `autotune_effort="none"`), and tuned configs arrive as a `build` keyword:

```python
import helion.language as hl
from kernel_agent.kernels import helion_tune


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):  # the grid: one program per tile of rows
        row = x[tile_m, :].to(torch.float32)
        rstd = torch.rsqrt(torch.mean(row * row, dim=-1) + eps)
        out[tile_m, :] = (row * rstd[:, None]).to(x.dtype) * weight[:]
    return out


def build(reference, helion_configs=None):
    return HelionRMSNorm(reference, helion_tune.kernel(rmsnorm, helion_configs))
```

* Code outside the `for` loops runs on the host (allocations, shapes); the loops are one
  GPU kernel. A nested `hl.tile` is a loop inside the kernel (the K loop of a GEMM:
  `examples/helion_gemm_epilogue.py`, `acc = torch.addmm(acc, x[tm, tk], w[tk, tn])` on an
  `hl.zeros` fp32 accumulator, the epilogue before the store).
* PyTorch ops inside the loops lower through Inductor: keep the reference's casts (normalise
  in fp32, cast, then scale) or the exact tier rejects the rounding.
* `static_shapes=True` (the default of `helion_tune.kernel`): one compiled kernel per input
  shape and stride; decode shapes repeat, so each compiles once per process.
* The kernel takes tensors and scalars, no `*args` / `**kwargs`; reshape to 2-D on the host.

## Tuning: `sweep_candidate(..., strategy="helion")`

No space and no configs: the sweep checks the default config, runs Helion's autotuner on
the case with the most work (calls per run x elements) with most of the sweep's time as its
budget, a seed, earlier tuned configs of this GPU / versions / shape bucket as seeds, and no
warp specialisation where kernel-agent's arch rules refuse it (sm_120: slower or failing,
docs/RESEARCH-TRITON.md §1.2; TMA only on sm_90+ is Helion's own check). The tuned config is
checked and timed next to the default; the faster is bound into the snapshot
(`_KA_SWEEP_CONFIG = {"helion_configs": {"rmsnorm": {...}}}`), so integration and export never
tune again. One config serves every shape of a kernel: tune where the time is.

## When to choose it

Where its implicit search pays: GEMM-like and reduction kernels whose best tiles, indexing
and grouping differ per GPU, and epilogue fusions you would otherwise sweep by hand in
Triton. Host overhead per call is Helion's launcher plus Triton's: measure it on decode
shapes (an eager-timed small op pays it; inside CUDA graphs it does not count) and compare
with the backends table of `optimisation-playbook` before choosing Helion for tiny ops.

Measured on an RTX 5070 Ti (sm_120, Helion 1.4.0, torch 2.14.1, Triton 3.8.0): the RMSNorm
example bit-exact, its decode call 48 us (the Triton example 55, CUDA C++ 22 in the same
session); the GEMM example on a `[704, 1024] x [1024, 4096]` bf16 Linear 0.21x of cuBLAS
with Helion's default config and 0.80x after 175 s of `strategy="helion"`: tune before you
judge Helion, and keep cuBLAS where it wins (bf16 GEMMs that need no fused epilogue).

## Examples and sources

* Examples: `examples/helion_rmsnorm.py`, `examples/helion_gemm_epilogue.py`. All in
  kernel-agent's examples directory (`kernel_agent/agent/examples/`; a session's prompt
  gives the directory): copy their structure.
* Sources: the `documentation-sources` skill's `sources.md`, section "Helion".
