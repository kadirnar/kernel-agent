# INT8 on sm_120 (RTX 5070 Ti), measured

`torch._int_mm`, the instruction rates of `kernels/mma_peaks.py` and the bundled examples
(`docs/research-scripts/int8-178/results`; CUDA graphs; "streamed": enough weight copies to
exceed the 48 MB L2, as in a model):

| what | INT8 | FP8 | bf16 |
|---|---|---|---|
| `mma.sync` rate (`IMMA.16832.S8.S8` / `QMMA.16832.F32.E4M3` / `HMMA`) | 411 TOPS | 208 (block-scaled `QMMA.SF` 414) | 104 |
| cuBLASLt GEMM peak (`torch._int_mm` / `_scaled_mm` tensor-wise / cuBLAS) | 327 TOPS | 331 | 99 |
| `torch._int_mm` [352, 1024] x [1024, 8192] (no quantisation, no scales) | 28.4 us | 29.6 (`_scaled_mm`) | 68.5 |
| `triton_int8_w8a8_gemm.py` gate\|up [352, 1024] -> 8192, quantisation included | 25.1 us | 32.0 (`triton_fp8_w8a8_gemm.py`) | 68.4 |
| the same at M = 704 | 45.8 us | 57.7 | 126.6 |
| down [352, 4096] -> 1024 / q\|k\|v [352, 1024] -> 2560 / o_proj [352, 2048] -> 1024 | 20.0 / 10.6 / 11.7 us | 18.8 / 10.8 / 10.7 | 39.0 / 25.5 / 22.6 |
| `cuda_int8_skinny_gemm.py` (W8A8, IMMA) [1 / 16 / 32, 2048] -> 12288, streamed | 32.1 / 33.0 / 33.7 us | skinny weight-only 31.0 / 31.5 / 35.7 | 63.1 / 66.1 / 62.8 |
| the same [32, 6144] -> 2048 / [64, 4096] -> 1024 | 22.5 / 11.5 us | 36.3 / 25.1 | 35.9 / 13.9 |
| `cuda_int8_gemv.py` (weight-only) [1, 2048] -> 12288 / [1, 6144] -> 2048, streamed | 31.5 / 16.8 us (799 / 751 GB/s) | GEMV 31.2 / 16.4 | 63.1 / 31.5 |

On this GPU s8 `mma.sync` runs at twice plain e4m3's rate: a hand-written INT8 kernel reaches
the full tensor-core rate with the simple instruction (`tl.dot` on int8), where FP8 needs the
block-scaled one. Up to 16 rows the skinny W8A8 kernel is ~1 us slower than the weight-only
FP8 skinny kernel (its separate per-token quantisation launch); from 32 rows it is faster,
where converting every weight in registers makes weight-only kernels ALU bound. Without
`torch._int_mm`'s own epilogue the reference path (`int8_w8a8_linear`, quantisation and scales
as torch ops) costs ~80 us at M = 352: a fallback, not a kernel.

The module evaluator (eager, warm L2; `doctor --smoke`): the Triton example 1.70x at M = 704
(gate|up, 540 calls; the FP8 W8A8 example 1.72x in the same run, at relative L2 0.037 against
INT8's 0.011), the skinny W8A8 kernel 1.09x at 16 rows (two launches per call: host bound when
timed eagerly; its GPU time is half of bf16's, above). Tile configs of the Triton example were
swept at M = 352 / 704 (`bench_triton.py sweep`), the skinny kernel's rows / warps rule
streamed (`bench_cuda.py sweep`): re-sweep on another GPU.
