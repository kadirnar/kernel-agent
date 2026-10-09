# W4A4 speed (measured on an RTX 5070 Ti, sm_120)

Measured on one GPU: evidence from it, not a ceiling. Measure on the GPU at hand (the
*Ceilings* table's *W4A4* column uses this GPU's measured NVFP4 peak).

**Peaks** (`kernel_agent.kernels.roofline` peaks cache): NVFP4 `F.scaled_mm` 661 TFLOP/s,
FP8 e4m3 tensor-wise 332, MXFP8 332, INT8 328, bf16 101. The block-scaled FP4 MMA
(`mma.sync ... kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64`, CuTe `MmaMXF4NVF4Op`)
does twice the K of the FP8 `QMMA.SF` per instruction at the same issue rate.

**The GEMM alone** (operands quantised beforehand; CUDA graphs; the speed-up over cuBLASLt
FP8 tensor-wise, `torch._scaled_mm` with scalar scales):

| M x K -> N | bf16 | FP8 tensor-wise | NVFP4 `F.scaled_mm` (two-level, bf16 out) | CuTe `MmaMXF4NVF4Op` GEMM |
|---|---|---|---|---|
| 4096 x 4096 -> 4096 | 1397 us | 413 us (333 TFLOP/s) | 209 us (659, 1.98x) | 212 us (650, 1.96x) |
| 352 x 1024 -> 8192 (gate \| up) | 68 us | 29 us | 15.7 us (1.85x) | 14.1 us (2.10x, cold L2) |
| 352 x 1024 -> 4096 | 38 us | 18.6 us | 11.5 us (1.62x) | 9.5 us (1.93x, cold L2) |
| 352 x 1024 -> 2560 (q \| k \| v) | 25.7 us | 10.7 us | 7.5 us (1.42x) | |
| 352 x 2048 -> 1024 (o) | 22.7 us | 13.2 us | 10.1 us (1.31x) | 8.2 us (1.61x) |
| 352 x 4096 -> 1024 (down) | 39 us | 17.7 us | 16.0 us (1.11x) | 14.5 us (1.28x) |

At N = 1024 with large K a 128 x 128 tile grid leaves SMs idle (24 tiles on 70 SMs): split-K
is the next step there, as for FP8.

**With the activation quantiser** (every call; the FP8 column above has none):

| M x K -> N | Triton quantiser + `F.scaled_mm` (`triton_nvfp4_w4a4_gemm.py`) | CuTe quantiser + GEMM (`cute_nvfp4_w4a4_gemm.py`) |
|---|---|---|
| 4096 x 4096 -> 4096 | 295 us (466 TFLOP/s; quantiser 54 us, two launches) | 262 us (1.58x FP8's GEMM; quantiser 41 us) |
| 352 x 1024 -> 8192 | 21.2 us | 16.8 us |
| 352 x 1024 -> 4096 | 16.6 us | 11.7 us |
| 352 x 2048 -> 1024 | 15.2 us | 12.8 us |
| 352 x 4096 -> 1024 | 24.0 us | 22.2 us |

**A whole layer** (the VoxCPM2 LocDiT decoder layer at M = 352, every GEMM replaced, norms,
rotary attention and residuals as in eager; GPU time per call in a CUDA graph): bf16 301 us,
W4A4 (CuTe example) 201 us (1.50x), W4A4 with gate / up in FP8 210 us (1.43x), every GEMM
FP8 W8A8 (`cute_fp8_blockscaled_gemm.py`) 267 us (1.13x). It passes `near-lossless-fp4a`
through the evaluator (`locdit_gate.py`).

A separate quantiser launch costs 2.5-8 us at M = 352: fuse it into the producer of the
activations (RMSNorm, `silu(gate) * up`, the previous GEMM's epilogue), which then writes
codes, swizzled block scales and the per-token outer scale.

**Per-token outer scales cost nothing in a custom epilogue** (the CuTe example multiplies
`acc * outer[m] * tensor_scale` before its one rounding), but `F.scaled_mm` takes only a
tensor-wise second-level scale: per token through it needs fp32 output and an epilogue
pass, 27.9 us at 352 x 1024 -> 4096 against 11.9 us two-level with the bias fused. So
the `F.scaled_mm` example uses one outer scale per call (measured as accurate, skill
calibration).

**Host time** (eager, no CUDA graph; 704 x 1024 -> 8192, a loaded machine): bf16 126 us,
the CuTe example 41-50 us, the Triton + `F.scaled_mm` example 147-176 us (`F.scaled_mm`'s Python
recipe dispatch ~49 us, two Triton launches and their buffers ~75 us), the FP8 W8A8 Triton
example 70-77 us. The evaluator times eagerly: run the `F.scaled_mm` path inside a CUDA graph
(or call cuBLASLt directly, `cuda_cublaslt_fp8.py`'s approach) where host time matters.

Raw output and scripts: `docs/research-scripts/w4a4-233` (`bench_triton.py`,
`bench_cute.py`, `results/`).
