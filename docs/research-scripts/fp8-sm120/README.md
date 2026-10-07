# FP8 on sm_120: research scripts (issue #132)

The scripts behind the measurements in [`docs/FP8.md`](../../FP8.md), kept as they were
run (RTX 5070 Ti, torch 2.14.1+cu130, cuBLASLt 13.1.1, Triton 3.8.0, nvcc 13.4, CUTLASS
4.1 headers from the tilelang wheel), with their raw output in `results/`. One-off
benchmarks, not part of the library (excluded from ruff). They import `kernel_agent`
(put `src/` on `PYTHONPATH`) and read the VoxCPM2 captures of `runs/openbmb--VoxCPM2/`
(read only). Run GPU jobs under the GPU lock:
`flock ~/.cache/kernel-agent/gpu.lock env KERNEL_AGENT_LOCK_HELD=1 python gemm_bench.py dit352`.

* `probe_support.py`: which `torch._scaled_mm` / `F.scaled_mm` recipes run on sm_120 and
  whether they equal the fp32 math of the codes (`results/probe_support.out`).
* `lt_ext.py`, `probe_lt.py`: direct cuBLASLt from C++ (`load_inline`): heuristic per
  scale mode (`results/probe_lt.out`: status 4015 = `CUBLAS_STATUS_NOT_SUPPORTED`,
  4007 = `INVALID_VALUE`), algorithm timing, cached call path, batched split-K.
* `gemm_bench.py <set> [variant filter]`: FP8 GEMMs by recipe and backend at the
  VoxCPM2 shapes (`dit352`, `dit352n1024`, `dit22`, `enc80`, `lm16`, `lm1`, `lm272`),
  L2-cold CUDA-graph timing. `results/gemm_dit352.out`'s `Lt tensorwise split2` row for
  o_proj and its `down` failures come from a split-K layout bug fixed before
  `gemm_dit352n1024.out` (use that file for the N = 1024 shapes).
* `cutlass_bw.py`, `cutlass_bench.py`: CUTLASS SM120 blockwise (example-87 style) and
  dense FP8 GEMMs; `cutlass_mx_bench.py`, `mx_debug.py`: the SM120 block-scaled (MXFP8)
  instantiation, which built at 128×128×128 but whose `initialize()` returned
  `Error Internal` (`mx_debug.py` prints "init failed: Error Internal cuda: no error
  smem 81920" for M = 352 / 384 / 4096).
* `fp8lib.py`, `accuracy.py`: fake-quant reference math of every recipe (incl. FP8
  attention / KV-cache emulation) judged with `kernels.compare` / `kernels.verify` on
  the captured LocDiT and LM layers (`results/acc_all.out`).
* `calib.py`: model-wide activation statistics, static vs dynamic scales, KV-cache
  scales, text A -> text B (`results/calib.out`).
* `fusion_bench.py`: quantisation fused into producers vs separate; gate|up + silu·up +
  e4m3 chains; CUTLASS dense FP8 vs cuBLASLt (`results/fusion_bench.out`).
* `host_peak.py`: host time per eager call of each FP8 GEMM path and dense 8192³ peaks
  (`results/host_peak.out`).
* `kv_bench.py`: decode attention over bf16 vs e4m3 KV caches (`results/kv_bench.out`).
