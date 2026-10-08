# Measured (RTX 5070 Ti, DRAM 767 GB/s copy peak, L2 48 MB)

Part of the `fp8-weights` skill.

"Streamed": CUDA graph of back-to-back calls over enough weight copies to exceed
L2, as in the model, where every layer's weights evict the others'.

| GEMM | cuBLAS bf16 | FP8 example | speedup |
|---|---|---|---|
| [1, 2048] x [2048, 6144] (LM gate / up) | 32.3 us, 780 GB/s | GEMV 16.2 us, 777 GB/s | 1.99x |
| [1, 2048] x [2048, 12288] (gate + up merged) | 62.8 us | GEMV 30.8 us, 819 GB/s | 2.04x |
| [1, 6144] x [6144, 2048] (LM down) | 31.7 us | GEMV 16.5 us, 764 GB/s | 1.93x |
| [22, 1024] x [1024, 4096] (LocDiT up) | 12.1 us | skinny 6.9 us, 610 GB/s | 1.75x |
| [22, 4096] x [4096, 1024] (LocDiT down) | 13.2 us | skinny 7.6 us, 556 GB/s | 1.75x |
| [32, 2048] x [2048, 6144] | 34.9 us | skinny 18.9 us, 667 GB/s | 1.85x |
| [8, 2048] x [2048, 6144] (batch-8 LM gate / up) | 31.5 us | skinny 16.3 us, 775 GB/s | 1.94x |
| [16, 2048] x [2048, 6144] (batch-16) | 31.7 us | skinny 16.6 us, 760 GB/s | 1.91x |
| [8, 6144] x [6144, 2048] (batch-8 LM down) | 39.0 us | skinny 17.0 us, 743 GB/s | 2.30x |
| [16, 6144] x [6144, 2048] (batch-16) | 39.1 us | skinny 17.6 us, 714 GB/s | 2.22x |
| [176, 1024] x [1024, 4096] (batch-8 LocDiT, CFG) | 20.0 us | skinny 26.6 us | 0.75x |

Batched decode (`-o batch_size=N`, `workloads/voxcpm_batch.py`): the LM GEMMs
have M = N and stay weight-bandwidth bound, so FP8 weights pay as at M = 1;
the LocDiT runs CFG at M = 2N x 11 (176 at N = 8), where cuBLAS bf16 already
reaches ~74 TFLOP/s: compute bound, weight-only FP8 does not help (faster FP8
math needs FP8 activations: `fp8_w8a8`, skill `fp8-w8a8`).

Whole MLPs against the VoxCPM2 run's fused bf16 MLP kernel: base LM decode MLP
92.1 -> 48.8 us (1.89x), LocDiT MLP 33.7 -> 22.8 us (1.48x), with an FP8 MLP
composed of the two examples and `silu * up` in torch; fuse the activation
into the gate / up epilogue (one launch) for the eager case.

The module evaluator times eager calls with a warm L2: weights that fit in L2
stay cached between its calls, so FP8 gains there are smaller than in the
model (the GEMV on [2048, 6144]: 1.3-1.7x, the skinny GEMM on `[2, 11, 1024]`:
1.4x, host overhead ~16-19 us per call included; the merged gate + up weight,
50 MB, measures 2.0x). A bf16 reference that only just fits in L2 can time
slower next to your FP8 weights than alone, and the reference-timing check then
reports an `integrity_violation` (rare): evaluate again before you suspect the
kernel. `pct_of_sol` of a `fp8_weights` target counts the weights at 1 byte (+
4 bytes of scale per channel). Judge FP8 by achieved GB/s, by `pct_of_sol` and
end to end. For a `fp8_w8a8` target the GEMMs on its weights also count at the
measured FP8 tensor-core peak (`float8_e4m3fn` in the GPU peaks), so its
`pct_of_sol` is against FP8 math, not bf16.
