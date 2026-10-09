| capture | variant | inputs | draws | near-lossless fails | relaxed fails | min cosine | max rel L2 | max norm change | max element ratio (NL / relaxed) |
|---|---|---|---|---|---|---|---|---|---|
| 20261006-004718/dit_layer | NVFP4 W4A4 | captured | 2 | 0 | 0 | 0.9968 | 0.080 | 3.4 % | 0.26 / 0.20 |
| 20261006-004718/dit_layer | NVFP4 W4A4 | redrawn | 40 | 0 | 0 | 0.9885 | 0.151 | 2.3 % | 0.28 / 0.24 |
| 20261006-004718/dit_layer | NVFP4 W4A4 | scaled | 6 | 0 | 0 | 0.9965 | 0.084 | 3.2 % | 0.15 / 0.13 |
| 20261006-004718/dit_layer | NVFP4 W4A4, per-tensor activations | captured | 2 | 0 | 0 | 0.9969 | 0.079 | 2.9 % | 0.24 / 0.18 |
| 20261006-004718/dit_layer | NVFP4 W4A4, per-tensor activations | redrawn | 40 | 0 | 0 | 0.9885 | 0.151 | 2.3 % | 0.28 / 0.24 |
| 20261006-004718/dit_layer | NVFP4 W4A4, per-tensor activations | scaled | 6 | 0 | 0 | 0.9964 | 0.084 | 2.5 % | 0.15 / 0.13 |
| 20261006-004718/dit_layer | NVFP4 W4A4 + Hadamard 16 | captured | 2 | 0 | 0 | 0.9965 | 0.084 | 2.4 % | 0.28 / 0.21 |
| 20261006-004718/dit_layer | NVFP4 W4A4 + Hadamard 16 | redrawn | 40 | 0 | 0 | 0.9810 | 0.197 | 3.7 % | 0.33 / 0.29 |
| 20261006-004718/dit_layer | NVFP4 W4A4 + Hadamard 16 | scaled | 6 | 0 | 0 | 0.9957 | 0.093 | 5.1 % | 0.19 / 0.16 |
| 20261006-004718/dit_layer | MXFP4 W4A4 | captured | 2 | 2 | 0 | 0.9935 | 0.116 | 6.5 % | 0.48 / 0.36 |
| 20261006-004718/dit_layer | MXFP4 W4A4 | redrawn | 40 | 0 | 0 | 0.9727 | 0.234 | 10.8 % | 0.41 / 0.35 |
| 20261006-004718/dit_layer | MXFP4 W4A4 | scaled | 6 | 0 | 0 | 0.9925 | 0.123 | 6.3 % | 0.24 / 0.21 |
| 20261006-004718/dit_layer | MXFP4 W4A4 + Hadamard 32 | captured | 2 | 2 | 0 | 0.9946 | 0.104 | 6.5 % | 0.35 / 0.27 |
| 20261006-004718/dit_layer | MXFP4 W4A4 + Hadamard 32 | redrawn | 40 | 0 | 0 | 0.9752 | 0.224 | 1.9 % | 0.42 / 0.36 |
| 20261006-004718/dit_layer | MXFP4 W4A4 + Hadamard 32 | scaled | 6 | 0 | 0 | 0.9938 | 0.111 | 4.9 % | 0.24 / 0.21 |
| 20261006-004718/dit_layer | BUG w4a4: weight scale x1.05 | captured | 2 | 2 | 2 | 0.9969 | 0.114 | 11.3 % | 0.51 / 0.50 |
| 20261006-004718/dit_layer | BUG w4a4: weight scale x1.05 | redrawn | 40 | 0 | 0 | 0.9885 | 0.163 | 5.1 % | 0.29 / 0.25 |
| 20261006-004718/dit_layer | BUG w4a4: weight scale x1.05 | scaled | 6 | 0 | 0 | 0.9950 | 0.125 | 11.0 % | 0.24 / 0.21 |
| 20261006-004718/dit_layer | BUG w4a4: weight scale x1.2 | captured | 2 | 2 | 2 | 0.9968 | 0.668 | 66.7 % | 2.75 / 2.69 |
| 20261006-004718/dit_layer | BUG w4a4: weight scale x1.2 | redrawn | 40 | 40 | 40 | 0.9444 | 0.354 | 20.1 % | 1.09 / 0.97 |
| 20261006-004718/dit_layer | BUG w4a4: weight scale x1.2 | scaled | 6 | 6 | 6 | 0.9257 | 0.625 | 62.1 % | 1.38 / 1.20 |
| 20261006-004718/dit_layer | BUG w4a4: nibbles swapped | captured | 2 | 2 | 2 | 0.0045 | 1.193 | 94.8 % | 3.66 / 3.59 |
| 20261006-004718/dit_layer | BUG w4a4: nibbles swapped | redrawn | 40 | 40 | 40 | -0.0323 | 1.770 | 71.3 % | 2.67 / 2.34 |
| 20261006-004718/dit_layer | BUG w4a4: nibbles swapped | scaled | 6 | 6 | 6 | -0.0907 | 1.193 | 85.1 % | 2.16 / 1.88 |
| 20261006-004718/dit_layer | BUG w4a4: activation block scales shifted | captured | 2 | 2 | 2 | 0.8946 | 0.562 | 23.5 % | 2.41 / 1.82 |
| 20261006-004718/dit_layer | BUG w4a4: activation block scales shifted | redrawn | 40 | 40 | 40 | 0.5370 | 1.597 | 89.3 % | 4.59 / 3.95 |
| 20261006-004718/dit_layer | BUG w4a4: activation block scales shifted | scaled | 6 | 6 | 6 | 0.3951 | 6.374 | 572.0 % | 11.79 / 10.12 |
| 20261006-004718/dit_layer | BUG w4a4: first token's outer scale | captured | 2 | 2 | 2 | 0.9793 | 0.793 | 78.8 % | 3.02 / 2.96 |
| 20261006-004718/dit_layer | BUG w4a4: first token's outer scale | redrawn | 40 | 12 | 9 | 0.9075 | 0.580 | 49.2 % | 0.59 / 0.52 |
| 20261006-004718/dit_layer | BUG w4a4: first token's outer scale | scaled | 6 | 2 | 2 | 0.9535 | 0.701 | 67.9 % | 0.63 / 0.56 |
| 20261006-004718/dit_layer | BUG w4a4: activation codes truncated | captured | 2 | 2 | 2 | 0.9941 | 0.468 | 46.7 % | 1.88 / 1.84 |
| 20261006-004718/dit_layer | BUG w4a4: activation codes truncated | redrawn | 40 | 40 | 23 | 0.9628 | 0.318 | 22.9 % | 0.61 / 0.55 |
| 20261006-004718/dit_layer | BUG w4a4: activation codes truncated | scaled | 6 | 6 | 6 | 0.9202 | 0.412 | 40.6 % | 1.01 / 0.88 |
| 20261006-004718/dit_layer | BUG w4a4: cached activation scales | captured | 2 | 0 | 0 | 0.9968 | 0.080 | 3.4 % | 0.26 / 0.20 |
| 20261006-004718/dit_layer | BUG w4a4: cached activation scales | redrawn | 40 | 40 | 40 | 0.6643 | 0.771 | 54.9 % | 1.13 / 1.02 |
| 20261006-004718/dit_layer | BUG w4a4: cached activation scales | scaled | 6 | 0 | 0 | 0.9965 | 0.084 | 3.4 % | 0.32 / 0.28 |
| 20261006-004718/dit_layer | BUG w4a4: int4 per tensor | captured | 2 | 2 | 2 | 0.9902 | 0.142 | 1.9 % | 1.48 / 1.23 |
| 20261006-004718/dit_layer | BUG w4a4: int4 per tensor | redrawn | 40 | 7 | 0 | 0.9558 | 0.310 | 16.2 % | 1.06 / 0.95 |
| 20261006-004718/dit_layer | BUG w4a4: int4 per tensor | scaled | 6 | 2 | 2 | 0.9436 | 0.582 | 42.3 % | 0.74 / 0.65 |
| 20261006-004718/dit_layer | BUG w4a4: row 0 of every GEMM skipped | captured | 2 | 0 | 0 | 0.9967 | 0.082 | 3.4 % | 0.59 / 0.46 |
| 20261006-004718/dit_layer | BUG w4a4: row 0 of every GEMM skipped | redrawn | 40 | 0 | 0 | 0.9871 | 0.161 | 2.2 % | 0.82 / 0.73 |
| 20261006-004718/dit_layer | BUG w4a4: row 0 of every GEMM skipped | scaled | 6 | 0 | 0 | 0.9948 | 0.102 | 3.1 % | 0.47 / 0.41 |
| 20261006-004718/dit_layer | BUG w4a4: KV head 0 dropped | captured | 2 | 2 | 2 | 0.7795 | 0.626 | 22.2 % | 3.06 / 2.84 |
| 20261006-004718/dit_layer | BUG w4a4: KV head 0 dropped | redrawn | 40 | 40 | 40 | 0.7229 | 1.056 | 90.6 % | 0.99 / 0.88 |
| 20261006-004718/dit_layer | BUG w4a4: KV head 0 dropped | scaled | 6 | 6 | 6 | 0.5561 | 1.007 | 25.6 % | 1.55 / 1.37 |
| 20261006-004718/dit_layer | BUG w4a4: q heads 0/last swapped (layout) | captured | 2 | 0 | 0 | 0.9968 | 0.080 | 3.9 % | 0.26 / 0.21 |
| 20261006-004718/dit_layer | BUG w4a4: q heads 0/last swapped (layout) | redrawn | 40 | 0 | 0 | 0.9885 | 0.151 | 3.1 % | 0.28 / 0.24 |
| 20261006-004718/dit_layer | BUG w4a4: q heads 0/last swapped (layout) | scaled | 6 | 0 | 0 | 0.9804 | 0.197 | 3.2 % | 0.37 / 0.32 |
| 20261005-192504/lm_step_fp8 | NVFP4 W4A4 | captured | 3 | 0 | 0 | 0.9988 | 0.050 | 1.8 % | 0.12 / 0.09 |
| 20261005-192504/lm_step_fp8 | NVFP4 W4A4 | redrawn | 60 | 0 | 0 | 0.9965 | 0.086 | 2.7 % | 0.13 / 0.12 |
| 20261005-192504/lm_step_fp8 | NVFP4 W4A4 | scaled | 9 | 0 | 0 | 0.9684 | 0.256 | 2.8 % | 0.36 / 0.32 |
| 20261005-192504/lm_step_fp8 | NVFP4 W4A4, per-tensor activations | captured | 3 | 0 | 0 | 0.9988 | 0.050 | 1.8 % | 0.12 / 0.09 |
| 20261005-192504/lm_step_fp8 | NVFP4 W4A4, per-tensor activations | redrawn | 60 | 0 | 0 | 0.9965 | 0.086 | 2.7 % | 0.13 / 0.12 |
| 20261005-192504/lm_step_fp8 | NVFP4 W4A4, per-tensor activations | scaled | 9 | 0 | 0 | 0.9684 | 0.256 | 2.8 % | 0.36 / 0.32 |
| 20261005-192504/lm_step_fp8 | NVFP4 W4A4 + Hadamard 16 | captured | 3 | 1 | 1 | 0.9977 | 0.069 | 3.4 % | 0.17 / 0.14 |
| 20261005-192504/lm_step_fp8 | NVFP4 W4A4 + Hadamard 16 | redrawn | 60 | 0 | 0 | 0.9945 | 0.110 | 6.0 % | 0.14 / 0.12 |
| 20261005-192504/lm_step_fp8 | NVFP4 W4A4 + Hadamard 16 | scaled | 9 | 0 | 0 | 0.9274 | 0.419 | 11.6 % | 0.54 / 0.46 |
| 20261005-192504/lm_step_fp8 | MXFP4 W4A4 | captured | 3 | 1 | 1 | 0.9916 | 0.135 | 3.1 % | 0.22 / 0.18 |
| 20261005-192504/lm_step_fp8 | MXFP4 W4A4 | redrawn | 60 | 3 | 0 | 0.9702 | 0.245 | 15.8 % | 0.34 / 0.30 |
| 20261005-192504/lm_step_fp8 | MXFP4 W4A4 | scaled | 9 | 0 | 0 | 0.9356 | 0.372 | 7.3 % | 0.49 / 0.42 |
| 20261005-192504/lm_step_fp8 | MXFP4 W4A4 + Hadamard 32 | captured | 3 | 2 | 2 | 0.9988 | 0.049 | 2.4 % | 0.14 / 0.11 |
| 20261005-192504/lm_step_fp8 | MXFP4 W4A4 + Hadamard 32 | redrawn | 60 | 2 | 0 | 0.9928 | 0.182 | 16.0 % | 0.19 / 0.17 |
| 20261005-192504/lm_step_fp8 | MXFP4 W4A4 + Hadamard 32 | scaled | 9 | 2 | 0 | 0.8903 | 0.494 | 10.9 % | 0.57 / 0.49 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: weight scale x1.05 | captured | 3 | 2 | 0 | 0.9988 | 0.072 | 5.7 % | 0.23 / 0.22 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: weight scale x1.05 | redrawn | 60 | 0 | 0 | 0.9962 | 0.109 | 7.8 % | 0.36 / 0.33 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: weight scale x1.05 | scaled | 9 | 1 | 1 | 0.9684 | 0.272 | 12.9 % | 1.22 / 1.12 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: weight scale x1.2 | captured | 3 | 3 | 3 | 0.9912 | 0.282 | 24.0 % | 1.14 / 1.09 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: weight scale x1.2 | redrawn | 60 | 60 | 59 | 0.9662 | 0.370 | 23.2 % | 1.58 / 1.47 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: weight scale x1.2 | scaled | 9 | 9 | 9 | 0.8534 | 0.770 | 63.8 % | 5.91 / 5.43 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: nibbles swapped | captured | 3 | 3 | 3 | -0.0024 | 1.006 | 90.9 % | 1.82 / 1.74 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: nibbles swapped | redrawn | 60 | 60 | 60 | -0.3124 | 1.082 | 90.2 % | 2.25 / 2.07 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: nibbles swapped | scaled | 9 | 9 | 9 | -0.0051 | 3.698 | 259.3 % | 7.63 / 7.00 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: activation block scales shifted | captured | 3 | 3 | 3 | 0.2002 | 0.982 | 80.2 % | 1.46 / 1.19 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: activation block scales shifted | redrawn | 60 | 60 | 60 | -0.2381 | 1.395 | 77.2 % | 3.96 / 3.64 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: activation block scales shifted | scaled | 9 | 9 | 9 | 0.1691 | 5.752 | 484.1 % | 6.83 / 5.90 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: first token's outer scale | captured | 3 | 0 | 0 | 0.9988 | 0.050 | 1.8 % | 0.12 / 0.09 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: first token's outer scale | redrawn | 60 | 0 | 0 | 0.9965 | 0.086 | 2.7 % | 0.13 / 0.12 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: first token's outer scale | scaled | 9 | 0 | 0 | 0.9684 | 0.256 | 2.8 % | 0.36 / 0.32 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: activation codes truncated | captured | 3 | 3 | 3 | 0.9911 | 0.307 | 28.6 % | 0.74 / 0.71 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: activation codes truncated | redrawn | 60 | 18 | 14 | 0.9569 | 0.369 | 35.2 % | 1.06 / 0.98 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: activation codes truncated | scaled | 9 | 7 | 7 | 0.8901 | 0.464 | 41.5 % | 3.59 / 3.29 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: cached activation scales | captured | 3 | 1 | 0 | 0.9988 | 0.078 | 6.4 % | 0.12 / 0.10 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: cached activation scales | redrawn | 60 | 28 | 27 | 0.8400 | 0.696 | 64.5 % | 0.69 / 0.61 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: cached activation scales | scaled | 9 | 0 | 0 | 0.9712 | 0.244 | 6.4 % | 0.47 / 0.41 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: int4 per tensor | captured | 3 | 2 | 1 | 0.9810 | 0.197 | 5.6 % | 0.69 / 0.52 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: int4 per tensor | redrawn | 60 | 0 | 0 | 0.9275 | 0.384 | 9.5 % | 0.49 / 0.42 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: int4 per tensor | scaled | 9 | 3 | 1 | 0.8444 | 0.555 | 16.2 % | 0.80 / 0.70 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: row 0 of every GEMM skipped | captured | 3 | 0 | 0 | 0.9971 | 0.078 | 2.0 % | 0.59 / 0.46 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: row 0 of every GEMM skipped | redrawn | 60 | 0 | 0 | 0.9870 | 0.161 | 2.4 % | 0.65 / 0.57 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: row 0 of every GEMM skipped | scaled | 9 | 0 | 0 | 0.9682 | 0.257 | 3.1 % | 0.36 / 0.32 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: KV head 0 dropped | captured | 3 | 3 | 3 | 0.9988 | 0.050 | 1.8 % | 0.12 / 0.09 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: KV head 0 dropped | redrawn | 60 | 21 | 18 | 0.9965 | 0.086 | 2.7 % | 0.14 / 0.13 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: KV head 0 dropped | scaled | 9 | 5 | 5 | 0.7173 | 0.698 | 24.2 % | 1.70 / 1.55 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: q heads 0/last swapped (layout) | captured | 3 | 0 | 0 | 0.9988 | 0.050 | 1.8 % | 0.12 / 0.09 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: q heads 0/last swapped (layout) | redrawn | 60 | 0 | 0 | 0.9965 | 0.086 | 2.7 % | 0.13 / 0.12 |
| 20261005-192504/lm_step_fp8 | BUG w4a4: q heads 0/last swapped (layout) | scaled | 9 | 0 | 0 | 0.9684 | 0.256 | 2.8 % | 0.36 / 0.32 |
