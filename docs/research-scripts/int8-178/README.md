# INT8 W8A8 / weight-only: research scripts (issue #178)

The scripts behind the INT8 numbers in the README ("INT8"), `knowledge/low_precision.md`
("INT8") and `kernels/compare.py`, kept as they were run (RTX 5070 Ti, sm_120, torch
2.14.1+cu130, Triton 3.8.0), with their raw output in `results/`. One-off benchmarks, not
part of the library (excluded from ruff). They import `kernel_agent` (put `src/` and this
directory on `PYTHONPATH`) and read the VoxCPM2 / Qwen3 captures of `runs/` (read only).
Run GPU jobs under the GPU lock:
`flock ~/.cache/kernel-agent/gpu.lock env KERNEL_AGENT_LOCK_HELD=1 python calib_int8.py dit`.

* `peaks_int8.py`: `torch._int_mm` TOPS at 4096³-16384 x 8192², the `mma.sync` rates of
  `kernels/mma_peaks.py` (s8 IMMA next to bf16 / e4m3), `int8_matmul` on the GPU against
  its exact CPU path, and `_int_mm` / `_scaled_mm` / bf16 at the VoxCPM2 LocDiT shapes
  (`results/peaks.out`).
* `bench_triton.py [sweep]`: `examples/triton_int8_w8a8_gemm.py` at the LocDiT shapes (M =
  352 / 704): bit for bit against `quant.int8_w8a8_linear`, CUDA-graph time against cuBLAS
  bf16, the `_int_mm` reference path and the FP8 W8A8 example; `sweep`: the tile configs
  behind its `CONFIGS` (`results/bench_triton.out`).
* `bench_cuda.py [rows ...] | sweep`: `examples/cuda_int8_skinny_gemm.py` (W8A8, IMMA) and
  `examples/cuda_int8_gemv.py` (weight-only) streamed from DRAM against cuBLAS bf16 and the
  FP8 skinny GEMM / GEMV examples; `sweep`: the skinny kernel's rows / warps / unroll behind
  its auto rule (`results/bench_cuda.out`).
* `calib_int8.py dit | lm | alpha | capture <path>`: the tier calibration: every
  `nn.Linear` of a captured module replaced by a recipe's reference math (INT8 W8A8, INT8
  weight-only, SmoothQuant, FP8 for comparison, and broken variants: weight scales x 1.05, a
  neighbour channel's scale, the first token's scale, a per-tensor activation scale, a zeroed
  output channel, a static calibrated activation scale, activations clipped at their 99.9th
  percentile), judged by `kernels.compare` in the near-lossless tier on the captured inputs,
  on the evaluator's redrawn draws (`verify.perturb_`) and on its scaled checks (x 3, x 0.01,
  x -1), with a per-`nn.Linear` table (`results/calib_dit.out`, `calib_lm.out`,
  `calib_alpha.out`, `calib_qwen3_mlp.out`); with `CALIB_TIER=relaxed` judged in #175's
  relaxed tier (`results/calib_*_relaxed.out`).
* `int8lib.py`: the examples as modules, CUDA-graph and streamed timing.
