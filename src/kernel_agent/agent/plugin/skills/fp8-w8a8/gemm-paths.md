# FP8 GEMM paths: `torch._scaled_mm`, custom e4m3 GEMMs, the sm_120 peak, eager timing

Part of the `fp8-w8a8` skill (the W8A8 contract is in its SKILL.md).

**`torch._scaled_mm`** (cuBLASLt, sm_120, torch 2.14):
`torch._scaled_mm(x_q, w_q.t(), scale_a=x_s[:, None], scale_b=w_s[None, :],
out_dtype=torch.bfloat16)`: `x_q` [M, K] e4m3 row-major, the weight [N, K] as
stored passed transposed (column-major [K, N], the layout cuBLASLt needs for
the second FP8 operand; a row-major one is refused: `b.stride(0) == 1`), fp32
scales [M, 1] and [1, N] (a [N] scale is refused). K and N multiples of 16
(else a RuntimeError), any M (1, 11, 353 work). `bias=` (bf16) and
`out_dtype=` bf16 / fp16 / fp32 work. Its result equals the fp32 math of
`fp8_w8a8_linear`'s fallback up to the bf16 rounding of the output. At M = 352
(graph-timed) it runs gate|up [1024 -> 8192] in 41 us vs 68 us bf16, down
[4096 -> 1024] 27 vs 39, q|k|v [1024 -> 2560] 15.5 vs 25.5, o_proj
[2048 -> 1024] 15.1 vs 22.5.

**When a custom e4m3 GEMM beats cuBLASLt.** Its sm_120 FP8 kernels use large
tiles, so at M = 352 an N = 1024 GEMM has 24 output tiles for 70 SMs. A plain
Triton e4m3 GEMM (`tl.dot` on `tl.float8e4nv` operands from
`torch.float8_e4m3fn` tensors, fp32 accumulator, both scales in the epilogue)
with one tile config per weight shape, picked by timing 12 L2-cold weights in a
CUDA graph, ran gate|up in 36.3 vs 41.2 us, q|k|v 14.4 vs 15.6, o_proj 14.0 vs
15.1, down 26.5 vs 27.0 (the example's `CONFIGS`; end to end ~4 % of the run).
Write one when the GEMM is a large share and the output tiles of the library
kernel do not fill the SMs a few times over; otherwise `_scaled_mm` is the
baseline to beat. `x.to(tl.float8e4nv)` rounds to nearest even; clamp to ±448
before it. CUDA C++: `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` on
plain sm_120. The example's quantisation + GEMM in a CUDA graph (warm L2,
M = 352): gate|up 36.1 us (cuBLAS bf16 68.2, `_scaled_mm` alone 41.3), q|k|v
15.1 (25.5 / 15.5), o_proj 15.5 (22.5 / 15.1), down 28.0 (38.9 / 26.6); its
output is bit-identical to `_scaled_mm`'s.

**The FP8 peak on sm_120** (RTX 5070 Ti, dense, fp32 accumulation, 8192^3; each
exact against the fp32 product of the same e4m3 values): cuBLASLt's FP8 GEMM
with scalar (tensor-wise) scales, an `nvjet_sm120_qqtst_mma_*` TMA kernel,
reaches 338 TFLOP/s, the `float8_e4m3fn` peak of the roofline; row-wise scales
in torch 2.14 (a CUTLASS 3.x kernel) reach 195, Triton 3.8's `tl.dot` on e4m3
(`mma.sync` m16n8k32) 183-195 over six tile configs; bf16 99. At M = 352 the
tensor-wise kernel runs gate|up in 28.9 us, down 17.5, q|k|v 10.7, o_proj 13.2
(204 / 169 / 172 / 112 TFLOP/s), well ahead of the example and of row-wise
`_scaled_mm`. It computes the unscaled product: apply `sx[m] * sw[n]` where the
output is read next (the `silu(gate) * up`, the residual add: Inductor fuses
it under torch.compile, a fused custom kernel does it in registers). As separate
torch ops on an fp32 output the scales cost more than they save (gate|up
52.6 us); in bf16 the product of e4m3 codes is exact enough (one more bf16
rounding, 2^-9, next to FP8's ~3 %). Plain e4m3 `tl.dot` (~55 % of the FP8 peak)
runs on `QMMA.F32`, capped at 208 TFLOP/s; `tl.dot_scaled` with unit ue8m0 scales
(127) runs on `QMMA.SF` (416), bit-identical and 4-16 % faster at M = 352
(docs/RESEARCH-TRITON.md §1.2). The example's GEMM uses `tl.dot_scaled` on sm_120 /
sm_121 (constant `tl.full((BM, BK // 32), 127, tl.uint8)` scales, row / column scales
in the epilogue) and `tl.dot` elsewhere: on sm_89 / sm_90 Triton would emulate the
block-scaled form through bf16, and on sm_100 plain `tl.dot` on e4m3 is already
full-rate `tcgen05.mma kind::f8f6f4` while `tl.dot_scaled` at 64-row tiles falls back to
`kind::f16` (128-row tiles reach `kind::mxf8f6f4.block_scale`; Triton 3.8, compiled for
sm_100 without a GPU); check the lowering once per Triton version: the
compiled kernel's PTX must contain `mma ... kind::mxf8f6f4.block_scale`
(`gemm_ptx(capability)` + `block_scale_mma(ptx)` in the example; compiling needs no
GPU).

**Eager timing.** The module evaluator times eager calls. The example's host
time per call (two Triton launches and three allocations, ~47 us; ~89 us through
the `torch.library.custom_op` dispatcher, which is why it calls the launcher
directly when not compiling) is above its GPU time at M = 352, so it measures
0.95x against cuBLAS bf16 there, 1.79x at M = 704. Under torch.compile / CUDA
graphs, as in the model, host time does not count. For an eager-timed target,
launch less per call: fuse the activation quantisation into its producer, or
put several GEMMs (a whole MLP or layer) behind one C++ launcher (`load_inline`,
~19 us per call). For the GEMMs themselves call cuBLASLt directly with cached
descriptors and algorithm: `examples/cuda_cublaslt_fp8.py` (6.1 us of host time per
GEMM, 5.5 each with several behind one call, against 18-20 for `torch._scaled_mm`
and 19 for `at::_scaled_mm` from C++, which spends it in ATen's checks).

End to end in that run (batch 16, near-lossless, every step within the
perceptual gate): W8A8 `_scaled_mm` on the LocDiT MLP took it from 11.36 to 9.74
ms per audio second (3.32x -> 3.87x vs eager), on q|k|v and o_proj to 9.03
(4.17x), the Triton GEMM to 8.73 (4.31x), while the exact-tier kernel arm on the
same layer stayed at 3.01x module speedup.
