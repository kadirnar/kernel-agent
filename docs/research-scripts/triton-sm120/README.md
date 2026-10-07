# Triton on sm_120: research scripts (issue #135)

The scripts behind the measurements in [`docs/RESEARCH-TRITON.md`](../../RESEARCH-TRITON.md),
kept as they were run (RTX 5070 Ti, Triton 3.8.0, torch 2.14.1+cu130). One-off benchmarks,
not part of the library (excluded from ruff). Run GPU jobs under the GPU lock:
`flock ~/.cache/kernel-agent/gpu.lock env KERNEL_AGENT_LOCK_HELD=1 python bench_fp8.py`.

* `mma_rate.cu`: register-only `mma.sync` rates (bf16, e4m3 `QMMA.F32`, block-scaled `QMMA.SF`).
* `bench_fp8.py`, `bench_mx.py`, `const_scale.py`: FP8 GEMMs at the LocDiT shapes
  (`tl.dot`, `tl.dot_scaled`, TMA, `_scaled_mm`).
* `sass_check.py`, `nvjet_name.py`: which instruction / cuBLASLt kernel a GEMM uses.
* `tabulate.py`, `plans.py`: the VoxCPM2 ledgers by backend (§4).
