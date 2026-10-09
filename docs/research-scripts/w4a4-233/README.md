# W4A4 (`fp4_w4a4`): research scripts (issue #233)

The scripts behind the W4A4 numbers in the README ("W4A4"), the `fp4-w4a4` skill and
`kernels/compare.py`'s `near-lossless-fp4a` / `relaxed-fp4a` tiers, kept as they were run
(RTX 5070 Ti, sm_120, torch 2.14.1+cu130, Triton 3.8.0, CUTLASS DSL 4.8.0), with their raw
output in `results/`. One-off research code, not part of the library (excluded from ruff).
They import `kernel_agent` (put `src/` on `PYTHONPATH`) and read the VoxCPM2 / Qwen3-0.6B
captures of `runs/` (read only). Run GPU jobs under the GPU lock:
`flock ~/.cache/kernel-agent/gpu.lock env KERNEL_AGENT_LOCK_HELD=1 python calibrate_w4a4.py ...`.

* `calibrate_w4a4.py [--linears] CAPTURE.pt ...`: the tier calibration, through
  `../relaxed-175/calibrate_tiers.py` (every `nn.Linear` of a capture replaced by a recipe's
  reference math, judged by `kernels.compare` on the captured inputs, 10 seeds of both
  per-channel redraws and the x 3 / x 0.01 / x -1 checks, in both W4A4 tiers): NVFP4 W4A4
  (per-token and per-call outer scales, a Hadamard rotation of 16), MXFP4 W4A4 (with a
  rotation of 32) and broken kernels (weight scale x 1.05 / x 1.2, swapped nibbles, shifted
  activation block scales, the first token's outer scale, truncated codes, cached activation
  scales, int4, a skipped row, a dropped KV head, swapped q heads).
  `results/calib_dit_lm.*` (LocDiT layer, base-LM decode layer), `results/calib_qwen3_locenc.*`
  (Qwen3-0.6B decode layer, the 12-layer LocEnc at M = 80), `results/summary.md`
  (`summarize.py`); `--linears`: every `nn.Linear` alone (`results/linears.out`).
* `norm_shrink.py CAPTURE.pt`: where W4A4's norm shrink comes from (weights, activations or
  both quantised; e4m3 or unrounded block scales): `results/norm_shrink.out`.
* `fp8_subsets.py CAPTURE.pt`: the LocDiT layer's outputs with W4A4 everywhere and named
  layers in FP8 (`results/fp8_subsets.out`).
* `locdit_gate.py CAPTURE.pt WORKDIR`: the LocDiT layer captured again in the W4A4 tiers, two
  candidates built from the CuTe examples through `run_evaluation`, the sensitivity probe, and
  the layer's GPU time in a CUDA graph (`results/locdit_gate.out`).
* `bench_triton.py`: `F.scaled_mm` NVFP4 (two-level; fp32 out with the per-token epilogue),
  `examples/triton_nvfp4_w4a4_gemm.py`, cuBLASLt FP8 tensor-wise and bf16 at 4096³ and the
  LocDiT shapes (`results/bench_triton.out`).
* `bench_cute.py`: `examples/cute_nvfp4_w4a4_gemm.py` (GEMM alone, quantiser + GEMM, the
  quantiser alone) against the same baselines, cold and warm L2 (`results/bench_cute.out`).
* `host_time.py`, `host_parts.py`: the examples' eager (host-bound) time and where the
  Triton one's goes (`results/host_time.out`).
