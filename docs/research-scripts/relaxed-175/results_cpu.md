| capture | variant | inputs | draws | near-lossless fails | relaxed fails | min cosine | max rel L2 | max norm change | max element ratio (NL / relaxed) |
|---|---|---|---|---|---|---|---|---|---|
| 20261006-004718/dit_layer | FP8 weights | captured | 2 | 0 | 0 | 0.9999 | 0.015 | 0.3 % | 0.14 / 0.10 |
| 20261006-004718/dit_layer | FP8 weights | redrawn | 40 | 0 | 0 | 0.9996 | 0.027 | 0.1 % | 0.28 / 0.14 |
| 20261006-004718/dit_layer | FP8 weights | scaled | 6 | 0 | 0 | 0.9999 | 0.015 | 0.3 % | 0.12 / 0.06 |
| 20261006-004718/dit_layer | FP8 W8A8 | captured | 2 | 0 | 0 | 0.9998 | 0.020 | 0.2 % | 0.19 / 0.14 |
| 20261006-004718/dit_layer | FP8 W8A8 | redrawn | 40 | 0 | 0 | 0.9993 | 0.038 | 0.1 % | 0.39 / 0.22 |
| 20261006-004718/dit_layer | FP8 W8A8 | scaled | 6 | 0 | 0 | 0.9998 | 0.024 | 0.9 % | 0.16 / 0.10 |
| 20261006-004718/dit_layer | MXFP8 W8A8 | captured | 2 | 0 | 0 | 0.9998 | 0.021 | 0.9 % | 0.17 / 0.14 |
| 20261006-004718/dit_layer | MXFP8 W8A8 | redrawn | 40 | 0 | 0 | 0.9993 | 0.038 | 0.2 % | 0.36 / 0.19 |
| 20261006-004718/dit_layer | MXFP8 W8A8 | scaled | 6 | 0 | 0 | 0.9998 | 0.021 | 0.8 % | 0.16 / 0.10 |
| 20261006-004718/dit_layer | NVFP4 weights | captured | 2 | 0 | 0 | 0.9984 | 0.056 | 0.9 % | 0.19 / 0.14 |
| 20261006-004718/dit_layer | NVFP4 weights | redrawn | 40 | 0 | 0 | 0.9951 | 0.099 | 0.2 % | 0.30 / 0.25 |
| 20261006-004718/dit_layer | NVFP4 weights | scaled | 6 | 0 | 0 | 0.9984 | 0.057 | 0.9 % | 0.12 / 0.10 |
| 20261006-004718/dit_layer | MXFP4 weights | captured | 2 | 2 | 0 | 0.9976 | 0.070 | 4.9 % | 0.27 / 0.21 |
| 20261006-004718/dit_layer | MXFP4 weights | redrawn | 40 | 0 | 0 | 0.9924 | 0.123 | 0.5 % | 0.40 / 0.34 |
| 20261006-004718/dit_layer | MXFP4 weights | scaled | 6 | 0 | 0 | 0.9972 | 0.083 | 4.4 % | 0.15 / 0.12 |
| 20261006-004718/dit_layer | BUG fp8: scales x1.05 | captured | 2 | 2 | 2 | 0.9999 | 0.159 | 15.9 % | 1.32 / 1.30 |
| 20261006-004718/dit_layer | BUG fp8: scales x1.05 | redrawn | 40 | 40 | 0 | 0.9989 | 0.057 | 5.0 % | 0.73 / 0.50 |
| 20261006-004718/dit_layer | BUG fp8: scales x1.05 | scaled | 6 | 6 | 4 | 0.9942 | 0.147 | 14.6 % | 1.16 / 0.64 |
| 20261006-004718/dit_layer | BUG fp8: scales x1.2 | captured | 2 | 2 | 2 | 0.9996 | 0.732 | 73.1 % | 6.01 / 5.88 |
| 20261006-004718/dit_layer | BUG fp8: scales x1.2 | redrawn | 40 | 40 | 40 | 0.9797 | 0.230 | 20.0 % | 3.29 / 2.23 |
| 20261006-004718/dit_layer | BUG fp8: scales x1.2 | scaled | 6 | 6 | 6 | 0.9177 | 0.679 | 67.5 % | 5.37 / 2.96 |
| 20261006-004718/dit_layer | BUG fp8: int4 per tensor | captured | 2 | 2 | 2 | 0.9902 | 0.142 | 1.9 % | 3.71 / 2.96 |
| 20261006-004718/dit_layer | BUG fp8: int4 per tensor | redrawn | 40 | 40 | 40 | 0.9718 | 0.242 | 3.1 % | 3.20 / 1.78 |
| 20261006-004718/dit_layer | BUG fp8: int4 per tensor | scaled | 6 | 6 | 6 | 0.9434 | 0.585 | 42.5 % | 2.42 / 1.47 |
| 20261006-004718/dit_layer | BUG fp8: row 0 of every GEMM skipped | captured | 2 | 2 | 2 | 0.9994 | 0.034 | 0.4 % | 1.61 / 1.17 |
| 20261006-004718/dit_layer | BUG fp8: row 0 of every GEMM skipped | redrawn | 40 | 40 | 40 | 0.9968 | 0.080 | 0.3 % | 3.52 / 2.21 |
| 20261006-004718/dit_layer | BUG fp8: row 0 of every GEMM skipped | scaled | 6 | 2 | 0 | 0.9981 | 0.061 | 0.2 % | 1.65 / 0.94 |
| 20261006-004718/dit_layer | BUG fp8: KV head 0 dropped | captured | 2 | 2 | 2 | 0.7811 | 0.624 | 21.9 % | 6.64 / 6.12 |
| 20261006-004718/dit_layer | BUG fp8: KV head 0 dropped | redrawn | 40 | 40 | 40 | 0.6648 | 0.747 | 33.5 % | 3.78 / 2.48 |
| 20261006-004718/dit_layer | BUG fp8: KV head 0 dropped | scaled | 6 | 6 | 6 | 0.5328 | 1.070 | 25.1 % | 5.06 / 3.08 |
| 20261006-004718/dit_layer | BUG fp8: q heads 0/last swapped (layout) | captured | 2 | 0 | 0 | 0.9999 | 0.015 | 0.1 % | 0.33 / 0.23 |
| 20261006-004718/dit_layer | BUG fp8: q heads 0/last swapped (layout) | redrawn | 40 | 0 | 0 | 0.9996 | 0.027 | 0.1 % | 0.28 / 0.14 |
| 20261006-004718/dit_layer | BUG fp8: q heads 0/last swapped (layout) | scaled | 6 | 2 | 2 | 0.9862 | 0.166 | 0.9 % | 1.32 / 0.73 |
| 20261006-004718/dit_layer | BUG w8a8: first token's activation scale | captured | 2 | 2 | 2 | 0.9812 | 0.783 | 77.7 % | 6.09 / 5.96 |
| 20261006-004718/dit_layer | BUG w8a8: first token's activation scale | redrawn | 40 | 20 | 3 | 0.9946 | 0.105 | 2.3 % | 1.96 / 1.19 |
| 20261006-004718/dit_layer | BUG w8a8: first token's activation scale | scaled | 6 | 6 | 4 | 0.9557 | 0.695 | 67.4 % | 2.06 / 1.25 |
| 20261006-004718/dit_layer | BUG fp4: nibbles swapped | captured | 2 | 2 | 2 | 0.0140 | 1.188 | 94.8 % | 3.70 / 3.62 |
| 20261006-004718/dit_layer | BUG fp4: nibbles swapped | redrawn | 40 | 40 | 40 | -0.0214 | 1.477 | 7.4 % | 4.00 / 3.42 |
| 20261006-004718/dit_layer | BUG fp4: nibbles swapped | scaled | 6 | 6 | 6 | -0.0729 | 1.188 | 85.1 % | 2.54 / 2.16 |
| 20261006-004718/dit_layer | BUG fp4: scales x1.2 | captured | 2 | 2 | 2 | 0.9984 | 0.711 | 71.0 % | 2.88 / 2.82 |
| 20261006-004718/dit_layer | BUG fp4: scales x1.2 | redrawn | 40 | 40 | 40 | 0.9797 | 0.229 | 20.1 % | 1.22 / 1.09 |
| 20261006-004718/dit_layer | BUG fp4: scales x1.2 | scaled | 6 | 6 | 6 | 0.9214 | 0.662 | 65.9 % | 1.68 / 1.43 |
| 20261006-004718/dit_layer | BUG fp4: int4 per tensor | captured | 2 | 2 | 2 | 0.9902 | 0.142 | 1.9 % | 1.65 / 1.34 |
| 20261006-004718/dit_layer | BUG fp4: int4 per tensor | redrawn | 40 | 1 | 0 | 0.9718 | 0.242 | 3.1 % | 1.04 / 0.89 |
| 20261006-004718/dit_layer | BUG fp4: int4 per tensor | scaled | 6 | 2 | 2 | 0.9434 | 0.585 | 42.5 % | 0.85 / 0.74 |
| 20261006-004718/dit_layer | BUG fp4: row 0 of every GEMM skipped | captured | 2 | 0 | 0 | 0.9982 | 0.059 | 0.8 % | 0.68 / 0.52 |
| 20261006-004718/dit_layer | BUG fp4: row 0 of every GEMM skipped | redrawn | 40 | 20 | 4 | 0.9925 | 0.122 | 0.4 % | 1.26 / 1.11 |
| 20261006-004718/dit_layer | BUG fp4: row 0 of every GEMM skipped | scaled | 6 | 0 | 0 | 0.9967 | 0.082 | 0.9 % | 0.55 / 0.47 |
| 20261006-004718/dit_layer | BUG fp4: KV head 0 dropped | captured | 2 | 2 | 2 | 0.7803 | 0.625 | 22.0 % | 3.19 / 2.94 |
| 20261006-004718/dit_layer | BUG fp4: KV head 0 dropped | redrawn | 40 | 40 | 40 | 0.6620 | 0.750 | 33.5 % | 1.40 / 1.24 |
| 20261006-004718/dit_layer | BUG fp4: KV head 0 dropped | scaled | 6 | 6 | 6 | 0.5216 | 1.095 | 25.3 % | 1.78 / 1.55 |
| 20261006-004718/dit_layer | BUG fp4: q heads 0/last swapped (layout) | captured | 2 | 0 | 0 | 0.9984 | 0.056 | 1.4 % | 0.19 / 0.14 |
| 20261006-004718/dit_layer | BUG fp4: q heads 0/last swapped (layout) | redrawn | 40 | 0 | 0 | 0.9951 | 0.099 | 0.2 % | 0.30 / 0.25 |
| 20261006-004718/dit_layer | BUG fp4: q heads 0/last swapped (layout) | scaled | 6 | 0 | 0 | 0.9849 | 0.173 | 1.4 % | 0.41 / 0.34 |
| 20261005-192504/lm_step_fp8 | FP8 weights | captured | 3 | 0 | 0 | 0.9998 | 0.022 | 0.1 % | 0.15 / 0.11 |
| 20261005-192504/lm_step_fp8 | FP8 weights | redrawn | 60 | 0 | 0 | 0.9995 | 0.032 | 0.6 % | 0.26 / 0.14 |
| 20261005-192504/lm_step_fp8 | FP8 weights | scaled | 9 | 0 | 0 | 0.9997 | 0.025 | 0.4 % | 0.13 / 0.07 |
| 20261005-192504/lm_step_fp8 | FP8 W8A8 | captured | 3 | 0 | 0 | 0.9995 | 0.031 | 1.3 % | 0.27 / 0.20 |
| 20261005-192504/lm_step_fp8 | FP8 W8A8 | redrawn | 60 | 0 | 0 | 0.9989 | 0.048 | 1.5 % | 0.29 / 0.15 |
| 20261005-192504/lm_step_fp8 | FP8 W8A8 | scaled | 9 | 0 | 0 | 0.9994 | 0.036 | 1.3 % | 0.22 / 0.12 |
| 20261005-192504/lm_step_fp8 | MXFP8 W8A8 | captured | 3 | 1 | 0 | 0.9993 | 0.039 | 3.0 % | 0.24 / 0.18 |
| 20261005-192504/lm_step_fp8 | MXFP8 W8A8 | redrawn | 60 | 12 | 1 | 0.9983 | 0.083 | 6.3 % | 0.48 / 0.31 |
| 20261005-192504/lm_step_fp8 | MXFP8 W8A8 | scaled | 9 | 1 | 0 | 0.9992 | 0.049 | 3.0 % | 0.26 / 0.14 |
| 20261005-192504/lm_step_fp8 | NVFP4 weights | captured | 3 | 0 | 0 | 0.9991 | 0.042 | 0.5 % | 0.15 / 0.11 |
| 20261005-192504/lm_step_fp8 | NVFP4 weights | redrawn | 60 | 0 | 0 | 0.9978 | 0.067 | 1.2 % | 0.14 / 0.12 |
| 20261005-192504/lm_step_fp8 | NVFP4 weights | scaled | 9 | 0 | 0 | 0.9803 | 0.200 | 1.5 % | 0.27 / 0.23 |
| 20261005-192504/lm_step_fp8 | MXFP4 weights | captured | 3 | 0 | 0 | 0.9927 | 0.120 | 2.0 % | 0.25 / 0.20 |
| 20261005-192504/lm_step_fp8 | MXFP4 weights | redrawn | 60 | 0 | 0 | 0.9878 | 0.163 | 6.7 % | 0.37 / 0.32 |
| 20261005-192504/lm_step_fp8 | MXFP4 weights | scaled | 9 | 0 | 0 | 0.9571 | 0.291 | 2.9 % | 0.45 / 0.38 |
| 20261005-192504/lm_step_fp8 | BUG fp8: scales x1.05 | captured | 3 | 3 | 3 | 0.9994 | 0.062 | 5.1 % | 0.52 / 0.49 |
| 20261005-192504/lm_step_fp8 | BUG fp8: scales x1.05 | redrawn | 60 | 60 | 0 | 0.9985 | 0.064 | 5.8 % | 0.93 / 0.69 |
| 20261005-192504/lm_step_fp8 | BUG fp8: scales x1.05 | scaled | 9 | 9 | 6 | 0.9881 | 0.169 | 15.0 % | 3.76 / 2.60 |
| 20261005-192504/lm_step_fp8 | BUG fp8: scales x1.2 | captured | 3 | 3 | 3 | 0.9904 | 0.298 | 25.5 % | 2.55 / 2.43 |
| 20261005-192504/lm_step_fp8 | BUG fp8: scales x1.2 | redrawn | 60 | 60 | 60 | 0.9745 | 0.284 | 20.8 % | 4.34 / 3.30 |
| 20261005-192504/lm_step_fp8 | BUG fp8: scales x1.2 | scaled | 9 | 9 | 9 | 0.8513 | 0.782 | 66.1 % | 18.23 / 11.99 |
| 20261005-192504/lm_step_fp8 | BUG fp8: int4 per tensor | captured | 3 | 3 | 3 | 0.9810 | 0.197 | 5.6 % | 2.06 / 1.39 |
| 20261005-192504/lm_step_fp8 | BUG fp8: int4 per tensor | redrawn | 60 | 60 | 59 | 0.9485 | 0.324 | 7.7 % | 2.04 / 1.20 |
| 20261005-192504/lm_step_fp8 | BUG fp8: int4 per tensor | scaled | 9 | 9 | 7 | 0.8444 | 0.555 | 16.2 % | 2.82 / 1.60 |
| 20261005-192504/lm_step_fp8 | BUG fp8: row 0 of every GEMM skipped | captured | 3 | 3 | 2 | 0.9977 | 0.068 | 0.4 % | 1.64 / 1.18 |
| 20261005-192504/lm_step_fp8 | BUG fp8: row 0 of every GEMM skipped | redrawn | 60 | 18 | 7 | 0.9573 | 0.289 | 4.5 % | 3.41 / 2.19 |
| 20261005-192504/lm_step_fp8 | BUG fp8: row 0 of every GEMM skipped | scaled | 9 | 6 | 0 | 0.9977 | 0.068 | 0.4 % | 1.18 / 0.64 |
| 20261005-192504/lm_step_fp8 | BUG fp8: KV head 0 dropped | captured | 3 | 3 | 3 | 0.9998 | 0.022 | 0.1 % | 0.15 / 0.11 |
| 20261005-192504/lm_step_fp8 | BUG fp8: KV head 0 dropped | redrawn | 60 | 57 | 48 | 0.9995 | 0.032 | 0.6 % | 0.26 / 0.14 |
| 20261005-192504/lm_step_fp8 | BUG fp8: KV head 0 dropped | scaled | 9 | 9 | 7 | 0.7446 | 0.667 | 25.5 % | 4.77 / 3.40 |
| 20261005-192504/lm_step_fp8 | BUG fp8: q heads 0/last swapped (layout) | captured | 3 | 0 | 0 | 0.9998 | 0.022 | 0.1 % | 0.15 / 0.11 |
| 20261005-192504/lm_step_fp8 | BUG fp8: q heads 0/last swapped (layout) | redrawn | 60 | 0 | 0 | 0.9995 | 0.032 | 0.6 % | 0.26 / 0.14 |
| 20261005-192504/lm_step_fp8 | BUG fp8: q heads 0/last swapped (layout) | scaled | 9 | 0 | 0 | 0.9997 | 0.025 | 0.4 % | 0.34 / 0.17 |
| 20261005-192504/lm_step_fp8 | BUG w8a8: first token's activation scale | captured | 3 | 0 | 0 | 0.9995 | 0.031 | 1.3 % | 0.27 / 0.20 |
| 20261005-192504/lm_step_fp8 | BUG w8a8: first token's activation scale | redrawn | 60 | 0 | 0 | 0.9989 | 0.048 | 1.5 % | 0.29 / 0.15 |
| 20261005-192504/lm_step_fp8 | BUG w8a8: first token's activation scale | scaled | 9 | 0 | 0 | 0.9994 | 0.036 | 1.3 % | 0.22 / 0.12 |
| 20261005-192504/lm_step_fp8 | BUG fp4: nibbles swapped | captured | 3 | 3 | 3 | -0.0079 | 1.007 | 90.9 % | 1.85 / 1.76 |
| 20261005-192504/lm_step_fp8 | BUG fp4: nibbles swapped | redrawn | 60 | 60 | 60 | -0.3548 | 1.120 | 88.8 % | 2.74 / 2.52 |
| 20261005-192504/lm_step_fp8 | BUG fp4: nibbles swapped | scaled | 9 | 9 | 9 | -0.0084 | 3.662 | 261.7 % | 8.64 / 7.67 |
| 20261005-192504/lm_step_fp8 | BUG fp4: scales x1.2 | captured | 3 | 3 | 3 | 0.9907 | 0.293 | 25.0 % | 1.22 / 1.16 |
| 20261005-192504/lm_step_fp8 | BUG fp4: scales x1.2 | redrawn | 60 | 60 | 60 | 0.9753 | 0.278 | 21.5 % | 1.79 / 1.65 |
| 20261005-192504/lm_step_fp8 | BUG fp4: scales x1.2 | scaled | 9 | 9 | 9 | 0.8511 | 0.781 | 65.3 % | 6.59 / 6.00 |
| 20261005-192504/lm_step_fp8 | BUG fp4: int4 per tensor | captured | 3 | 2 | 1 | 0.9810 | 0.197 | 5.6 % | 0.83 / 0.59 |
| 20261005-192504/lm_step_fp8 | BUG fp4: int4 per tensor | redrawn | 60 | 0 | 0 | 0.9485 | 0.324 | 7.7 % | 0.70 / 0.60 |
| 20261005-192504/lm_step_fp8 | BUG fp4: int4 per tensor | scaled | 9 | 4 | 3 | 0.8444 | 0.555 | 16.2 % | 0.93 / 0.80 |
| 20261005-192504/lm_step_fp8 | BUG fp4: row 0 of every GEMM skipped | captured | 3 | 0 | 0 | 0.9973 | 0.073 | 0.5 % | 0.69 / 0.52 |
| 20261005-192504/lm_step_fp8 | BUG fp4: row 0 of every GEMM skipped | redrawn | 60 | 1 | 1 | 0.9559 | 0.294 | 4.4 % | 1.24 / 1.09 |
| 20261005-192504/lm_step_fp8 | BUG fp4: row 0 of every GEMM skipped | scaled | 9 | 0 | 0 | 0.9801 | 0.201 | 1.5 % | 0.38 / 0.32 |
| 20261005-192504/lm_step_fp8 | BUG fp4: KV head 0 dropped | captured | 3 | 3 | 3 | 0.9991 | 0.042 | 0.5 % | 0.15 / 0.11 |
| 20261005-192504/lm_step_fp8 | BUG fp4: KV head 0 dropped | redrawn | 60 | 25 | 20 | 0.9978 | 0.067 | 1.2 % | 0.14 / 0.12 |
| 20261005-192504/lm_step_fp8 | BUG fp4: KV head 0 dropped | scaled | 9 | 6 | 5 | 0.7279 | 0.686 | 25.5 % | 1.88 / 1.70 |
| 20261005-192504/lm_step_fp8 | BUG fp4: q heads 0/last swapped (layout) | captured | 3 | 0 | 0 | 0.9991 | 0.042 | 0.5 % | 0.15 / 0.11 |
| 20261005-192504/lm_step_fp8 | BUG fp4: q heads 0/last swapped (layout) | redrawn | 60 | 0 | 0 | 0.9978 | 0.067 | 1.2 % | 0.14 / 0.12 |
| 20261005-192504/lm_step_fp8 | BUG fp4: q heads 0/last swapped (layout) | scaled | 9 | 0 | 0 | 0.9803 | 0.200 | 1.5 % | 0.27 / 0.23 |
| 20261008-021850/layer_decode | FP8 weights | captured | 4 | 0 | 0 | 0.9994 | 0.034 | 0.4 % | 0.23 / 0.15 |
| 20261008-021850/layer_decode | FP8 weights | redrawn | 80 | 72 | 53 | 0.8939 | 0.480 | 7.8 % | 3.77 / 1.99 |
| 20261008-021850/layer_decode | FP8 weights | scaled | 12 | 3 | 0 | 0.9985 | 0.055 | 1.5 % | 1.71 / 0.97 |
| 20261008-021850/layer_decode | FP8 W8A8 | captured | 4 | 0 | 0 | 0.9989 | 0.048 | 0.8 % | 0.30 / 0.20 |
| 20261008-021850/layer_decode | FP8 W8A8 | redrawn | 80 | 76 | 62 | 0.8376 | 0.571 | 13.9 % | 3.72 / 1.97 |
| 20261008-021850/layer_decode | FP8 W8A8 | scaled | 12 | 6 | 1 | 0.9974 | 0.073 | 3.4 % | 2.31 / 1.21 |
| 20261008-021850/layer_decode | MXFP8 W8A8 | captured | 4 | 0 | 0 | 0.9985 | 0.055 | 0.7 % | 0.40 / 0.27 |
| 20261008-021850/layer_decode | MXFP8 W8A8 | redrawn | 80 | 78 | 69 | 0.8087 | 0.609 | 8.8 % | 3.50 / 1.85 |
| 20261008-021850/layer_decode | MXFP8 W8A8 | scaled | 12 | 5 | 2 | 0.9979 | 0.073 | 3.4 % | 2.86 / 1.48 |
| 20261008-021850/layer_decode | NVFP4 weights | captured | 4 | 0 | 0 | 0.9925 | 0.123 | 1.3 % | 0.36 / 0.26 |
| 20261008-021850/layer_decode | NVFP4 weights | redrawn | 80 | 71 | 56 | 0.5209 | 1.026 | 15.4 % | 1.84 / 1.55 |
| 20261008-021850/layer_decode | NVFP4 weights | scaled | 12 | 4 | 4 | 0.9890 | 0.151 | 7.5 % | 1.86 / 1.57 |
| 20261008-021850/layer_decode | MXFP4 weights | captured | 4 | 0 | 0 | 0.9892 | 0.146 | 2.0 % | 0.39 / 0.28 |
| 20261008-021850/layer_decode | MXFP4 weights | redrawn | 80 | 75 | 69 | 0.3955 | 1.031 | 24.8 % | 1.48 / 1.23 |
| 20261008-021850/layer_decode | MXFP4 weights | scaled | 12 | 4 | 4 | 0.9691 | 0.247 | 3.6 % | 2.29 / 1.95 |
| 20261008-021850/layer_decode | BUG fp8: scales x1.05 | captured | 4 | 4 | 4 | 0.9985 | 0.126 | 11.4 % | 0.61 / 0.49 |
| 20261008-021850/layer_decode | BUG fp8: scales x1.05 | redrawn | 80 | 80 | 80 | 0.8809 | 0.564 | 18.9 % | 3.81 / 2.01 |
| 20261008-021850/layer_decode | BUG fp8: scales x1.05 | scaled | 12 | 12 | 12 | 0.9973 | 0.161 | 14.5 % | 1.77 / 0.95 |
| 20261008-021850/layer_decode | BUG fp8: scales x1.2 | captured | 4 | 4 | 4 | 0.9853 | 0.558 | 51.7 % | 2.58 / 2.09 |
| 20261008-021850/layer_decode | BUG fp8: scales x1.2 | redrawn | 80 | 80 | 80 | 0.8624 | 0.921 | 70.5 % | 5.93 / 3.01 |
| 20261008-021850/layer_decode | BUG fp8: scales x1.2 | scaled | 12 | 12 | 12 | 0.9820 | 0.707 | 66.6 % | 3.87 / 3.16 |
| 20261008-021850/layer_decode | BUG fp8: int4 per tensor | captured | 4 | 4 | 4 | 0.4691 | 0.933 | 28.0 % | 5.04 / 3.83 |
| 20261008-021850/layer_decode | BUG fp8: int4 per tensor | redrawn | 80 | 80 | 80 | 0.0164 | 2.081 | 95.7 % | 9.20 / 5.85 |
| 20261008-021850/layer_decode | BUG fp8: int4 per tensor | scaled | 12 | 12 | 12 | 0.2831 | 1.682 | 66.5 % | 19.02 / 10.54 |
| 20261008-021850/layer_decode | BUG fp8: row 0 of every GEMM skipped | captured | 4 | 4 | 3 | 0.9769 | 0.214 | 1.5 % | 4.96 / 4.14 |
| 20261008-021850/layer_decode | BUG fp8: row 0 of every GEMM skipped | redrawn | 80 | 78 | 62 | 0.8904 | 0.492 | 8.8 % | 3.72 / 1.97 |
| 20261008-021850/layer_decode | BUG fp8: row 0 of every GEMM skipped | scaled | 12 | 10 | 10 | 0.9821 | 0.195 | 9.0 % | 5.79 / 4.54 |
| 20261008-021850/layer_decode | BUG fp8: KV head 0 dropped | captured | 4 | 4 | 4 | 0.9429 | 0.334 | 8.1 % | 3.92 / 3.13 |
| 20261008-021850/layer_decode | BUG fp8: KV head 0 dropped | redrawn | 80 | 80 | 80 | 0.8939 | 0.480 | 7.8 % | 3.77 / 2.06 |
| 20261008-021850/layer_decode | BUG fp8: KV head 0 dropped | scaled | 12 | 11 | 10 | 0.9369 | 0.371 | 15.5 % | 7.06 / 6.32 |
| 20261008-021850/layer_decode | BUG fp8: q heads 0/last swapped (layout) | captured | 4 | 4 | 3 | 0.9181 | 0.416 | 4.4 % | 2.56 / 1.79 |
| 20261008-021850/layer_decode | BUG fp8: q heads 0/last swapped (layout) | redrawn | 80 | 80 | 80 | 0.7137 | 0.766 | 12.8 % | 11.36 / 6.00 |
| 20261008-021850/layer_decode | BUG fp8: q heads 0/last swapped (layout) | scaled | 12 | 11 | 7 | 0.8153 | 0.633 | 7.7 % | 2.68 / 1.44 |
| 20261008-021850/layer_decode | BUG w8a8: first token's activation scale | captured | 4 | 0 | 0 | 0.9989 | 0.048 | 0.8 % | 0.30 / 0.20 |
| 20261008-021850/layer_decode | BUG w8a8: first token's activation scale | redrawn | 80 | 76 | 65 | 0.7103 | 0.753 | 13.9 % | 6.80 / 3.56 |
| 20261008-021850/layer_decode | BUG w8a8: first token's activation scale | scaled | 12 | 6 | 1 | 0.9974 | 0.073 | 3.4 % | 2.31 / 1.21 |
| 20261008-021850/layer_decode | BUG fp4: nibbles swapped | captured | 4 | 4 | 4 | -0.1133 | 1.612 | 27.0 % | 4.43 / 3.92 |
| 20261008-021850/layer_decode | BUG fp4: nibbles swapped | redrawn | 80 | 80 | 80 | -0.0755 | 1.520 | 38.6 % | 7.74 / 6.69 |
| 20261008-021850/layer_decode | BUG fp4: nibbles swapped | scaled | 12 | 12 | 12 | -0.1039 | 1.883 | 71.8 % | 26.09 / 21.79 |
| 20261008-021850/layer_decode | BUG fp4: scales x1.2 | captured | 4 | 4 | 4 | 0.9786 | 0.594 | 53.6 % | 1.21 / 0.99 |
| 20261008-021850/layer_decode | BUG fp4: scales x1.2 | redrawn | 80 | 80 | 80 | 0.5073 | 1.515 | 81.5 % | 2.43 / 2.10 |
| 20261008-021850/layer_decode | BUG fp4: scales x1.2 | scaled | 12 | 12 | 12 | 0.9736 | 0.700 | 66.2 % | 2.34 / 1.97 |
| 20261008-021850/layer_decode | BUG fp4: int4 per tensor | captured | 4 | 4 | 4 | 0.4691 | 0.933 | 28.0 % | 2.18 / 1.71 |
| 20261008-021850/layer_decode | BUG fp4: int4 per tensor | redrawn | 80 | 80 | 80 | 0.0164 | 2.081 | 95.7 % | 3.33 / 2.93 |
| 20261008-021850/layer_decode | BUG fp4: int4 per tensor | scaled | 12 | 12 | 12 | 0.2831 | 1.682 | 66.5 % | 6.17 / 5.27 |
| 20261008-021850/layer_decode | BUG fp4: row 0 of every GEMM skipped | captured | 4 | 1 | 1 | 0.9716 | 0.238 | 1.9 % | 2.26 / 1.91 |
| 20261008-021850/layer_decode | BUG fp4: row 0 of every GEMM skipped | redrawn | 80 | 73 | 58 | 0.4953 | 1.029 | 16.3 % | 1.85 / 1.56 |
| 20261008-021850/layer_decode | BUG fp4: row 0 of every GEMM skipped | scaled | 12 | 8 | 7 | 0.9755 | 0.221 | 6.4 % | 2.44 / 2.27 |
| 20261008-021850/layer_decode | BUG fp4: KV head 0 dropped | captured | 4 | 3 | 3 | 0.9356 | 0.354 | 8.3 % | 1.74 / 1.42 |
| 20261008-021850/layer_decode | BUG fp4: KV head 0 dropped | redrawn | 80 | 73 | 57 | 0.5209 | 1.026 | 15.4 % | 1.84 / 1.55 |
| 20261008-021850/layer_decode | BUG fp4: KV head 0 dropped | scaled | 12 | 7 | 4 | 0.9344 | 0.380 | 12.2 % | 3.27 / 3.16 |
| 20261008-021850/layer_decode | BUG fp4: q heads 0/last swapped (layout) | captured | 4 | 2 | 1 | 0.9083 | 0.435 | 3.2 % | 1.15 / 0.86 |
| 20261008-021850/layer_decode | BUG fp4: q heads 0/last swapped (layout) | redrawn | 80 | 80 | 80 | 0.4758 | 1.062 | 19.5 % | 3.60 / 3.03 |
| 20261008-021850/layer_decode | BUG fp4: q heads 0/last swapped (layout) | scaled | 12 | 8 | 8 | 0.8016 | 0.664 | 12.8 % | 1.86 / 1.57 |
| decode attention (synthetic) | FP8 KV cache | captured | 48 | 0 | 0 | 0.9996 | 0.029 | 0.9 % | 0.16 / 0.12 |
| decode attention (synthetic) | FP8 KV cache | redrawn | 192 | 0 | 0 | 0.9968 | 0.081 | 2.7 % | 0.38 / 0.24 |
| decode attention (synthetic) | FP8 KV cache | scaled | 144 | 0 | 0 | 0.9996 | 0.030 | 0.9 % | 0.07 / 0.04 |
| decode attention (synthetic) | BUG kv: v scales x1.05 | captured | 48 | 48 | 48 | 0.9996 | 0.065 | 5.9 % | 0.31 / 0.24 |
| decode attention (synthetic) | BUG kv: v scales x1.05 | redrawn | 192 | 185 | 16 | 0.9968 | 0.105 | 7.0 % | 0.43 / 0.28 |
| decode attention (synthetic) | BUG kv: v scales x1.05 | scaled | 144 | 132 | 0 | 0.9996 | 0.065 | 5.9 % | 0.13 / 0.08 |
| decode attention (synthetic) | BUG kv: v scales x1.2 | captured | 48 | 48 | 48 | 0.9996 | 0.213 | 21.1 % | 0.85 / 0.66 |
| decode attention (synthetic) | BUG kv: v scales x1.2 | redrawn | 192 | 192 | 192 | 0.9968 | 0.237 | 22.2 % | 0.69 / 0.48 |
| decode attention (synthetic) | BUG kv: v scales x1.2 | scaled | 144 | 132 | 132 | 0.9996 | 0.213 | 21.1 % | 0.35 / 0.23 |
| decode attention (synthetic) | BUG kv: first token's scale | captured | 48 | 48 | 44 | 0.9427 | 0.451 | 39.5 % | 2.03 / 1.56 |
| decode attention (synthetic) | BUG kv: first token's scale | redrawn | 192 | 180 | 180 | 0.1813 | 1.746 | 77.5 % | 9.42 / 5.68 |
| decode attention (synthetic) | BUG kv: first token's scale | scaled | 144 | 129 | 57 | 0.9427 | 0.452 | 39.5 % | 0.77 / 0.49 |
