| capture | variant | inputs | draws | near-lossless fails | relaxed fails | min cosine | max rel L2 | max norm change | max element ratio (NL / relaxed) |
|---|---|---|---|---|---|---|---|---|---|
| 20261008-021850/layer_decode | NVFP4 W4A4 | captured | 4 | 0 | 0 | 0.9839 | 0.179 | 2.3 % | 0.50 / 0.39 |
| 20261008-021850/layer_decode | NVFP4 W4A4 | redrawn | 80 | 4 | 0 | 0.8934 | 0.479 | 15.4 % | 0.77 / 0.67 |
| 20261008-021850/layer_decode | NVFP4 W4A4 | scaled | 12 | 0 | 0 | 0.9797 | 0.205 | 11.5 % | 0.28 / 0.25 |
| 20261008-021850/layer_decode | NVFP4 W4A4, per-tensor activations | captured | 4 | 0 | 0 | 0.9839 | 0.179 | 2.3 % | 0.50 / 0.39 |
| 20261008-021850/layer_decode | NVFP4 W4A4, per-tensor activations | redrawn | 80 | 4 | 0 | 0.8934 | 0.479 | 15.4 % | 0.77 / 0.67 |
| 20261008-021850/layer_decode | NVFP4 W4A4, per-tensor activations | scaled | 12 | 0 | 0 | 0.9797 | 0.205 | 11.5 % | 0.28 / 0.25 |
| 20261008-021850/layer_decode | NVFP4 W4A4 + Hadamard 16 | captured | 4 | 0 | 0 | 0.9820 | 0.189 | 3.8 % | 0.57 / 0.43 |
| 20261008-021850/layer_decode | NVFP4 W4A4 + Hadamard 16 | redrawn | 80 | 1 | 0 | 0.8990 | 0.471 | 13.4 % | 0.94 / 0.81 |
| 20261008-021850/layer_decode | NVFP4 W4A4 + Hadamard 16 | scaled | 12 | 0 | 0 | 0.9763 | 0.217 | 7.0 % | 0.42 / 0.36 |
| 20261008-021850/layer_decode | MXFP4 W4A4 | captured | 4 | 0 | 0 | 0.9718 | 0.236 | 4.1 % | 0.72 / 0.54 |
| 20261008-021850/layer_decode | MXFP4 W4A4 | redrawn | 80 | 32 | 6 | 0.7988 | 0.606 | 37.4 % | 0.74 / 0.64 |
| 20261008-021850/layer_decode | MXFP4 W4A4 | scaled | 12 | 1 | 1 | 0.9563 | 0.361 | 20.6 % | 0.86 / 0.79 |
| 20261008-021850/layer_decode | MXFP4 W4A4 + Hadamard 32 | captured | 4 | 0 | 0 | 0.9703 | 0.242 | 4.8 % | 0.65 / 0.51 |
| 20261008-021850/layer_decode | MXFP4 W4A4 + Hadamard 32 | redrawn | 80 | 16 | 4 | 0.8390 | 0.552 | 20.8 % | 1.18 / 1.03 |
| 20261008-021850/layer_decode | MXFP4 W4A4 + Hadamard 32 | scaled | 12 | 3 | 1 | 0.8908 | 0.454 | 19.6 % | 0.61 / 0.56 |
| 20261008-021850/layer_decode | BUG w4a4: weight scale x1.05 | captured | 4 | 4 | 3 | 0.9824 | 0.221 | 13.6 % | 0.55 / 0.43 |
| 20261008-021850/layer_decode | BUG w4a4: weight scale x1.05 | redrawn | 80 | 25 | 3 | 0.8860 | 0.521 | 31.6 % | 1.16 / 1.05 |
| 20261008-021850/layer_decode | BUG w4a4: weight scale x1.05 | scaled | 12 | 1 | 1 | 0.9810 | 0.311 | 25.3 % | 0.51 / 0.47 |
| 20261008-021850/layer_decode | BUG w4a4: weight scale x1.2 | captured | 4 | 4 | 4 | 0.9759 | 0.591 | 52.5 % | 1.10 / 0.93 |
| 20261008-021850/layer_decode | BUG w4a4: weight scale x1.2 | redrawn | 80 | 80 | 80 | 0.8754 | 1.012 | 86.0 % | 1.88 / 1.71 |
| 20261008-021850/layer_decode | BUG w4a4: weight scale x1.2 | scaled | 12 | 12 | 12 | 0.9641 | 0.738 | 70.2 % | 1.50 / 1.42 |
| 20261008-021850/layer_decode | BUG w4a4: nibbles swapped | captured | 4 | 4 | 4 | -0.7327 | 1.703 | 78.0 % | 4.72 / 4.29 |
| 20261008-021850/layer_decode | BUG w4a4: nibbles swapped | redrawn | 80 | 80 | 80 | -0.8675 | 2.366 | 109.0 % | 2.33 / 2.00 |
| 20261008-021850/layer_decode | BUG w4a4: nibbles swapped | scaled | 12 | 12 | 12 | -0.7373 | 1.888 | 78.0 % | 2.76 / 2.60 |
| 20261008-021850/layer_decode | BUG w4a4: activation block scales shifted | captured | 4 | 4 | 4 | 0.3385 | 2.238 | 136.9 % | 4.25 / 3.28 |
| 20261008-021850/layer_decode | BUG w4a4: activation block scales shifted | redrawn | 80 | 80 | 80 | -0.3619 | 2.202 | 106.1 % | 1.78 / 1.54 |
| 20261008-021850/layer_decode | BUG w4a4: activation block scales shifted | scaled | 12 | 12 | 12 | 0.1863 | 2.942 | 198.6 % | 3.58 / 3.07 |
| 20261008-021850/layer_decode | BUG w4a4: first token's outer scale | captured | 4 | 0 | 0 | 0.9839 | 0.179 | 2.3 % | 0.50 / 0.39 |
| 20261008-021850/layer_decode | BUG w4a4: first token's outer scale | redrawn | 80 | 6 | 1 | 0.7952 | 0.774 | 27.6 % | 0.91 / 0.83 |
| 20261008-021850/layer_decode | BUG w4a4: first token's outer scale | scaled | 12 | 0 | 0 | 0.9797 | 0.205 | 11.5 % | 0.28 / 0.25 |
| 20261008-021850/layer_decode | BUG w4a4: activation codes truncated | captured | 4 | 4 | 4 | 0.9641 | 0.398 | 33.2 % | 0.76 / 0.67 |
| 20261008-021850/layer_decode | BUG w4a4: activation codes truncated | redrawn | 80 | 80 | 80 | 0.5847 | 0.948 | 43.7 % | 0.91 / 0.80 |
| 20261008-021850/layer_decode | BUG w4a4: activation codes truncated | scaled | 12 | 12 | 12 | 0.9217 | 0.512 | 42.8 % | 1.20 / 1.13 |
| 20261008-021850/layer_decode | BUG w4a4: cached activation scales | captured | 4 | 0 | 0 | 0.9840 | 0.179 | 2.7 % | 0.46 / 0.35 |
| 20261008-021850/layer_decode | BUG w4a4: cached activation scales | redrawn | 80 | 5 | 3 | 0.7419 | 0.678 | 36.5 % | 0.75 / 0.65 |
| 20261008-021850/layer_decode | BUG w4a4: cached activation scales | scaled | 12 | 4 | 3 | 0.7412 | 0.687 | 40.3 % | 1.56 / 1.43 |
| 20261008-021850/layer_decode | BUG w4a4: int4 per tensor | captured | 4 | 4 | 4 | 0.4688 | 0.933 | 28.0 % | 1.92 / 1.54 |
| 20261008-021850/layer_decode | BUG w4a4: int4 per tensor | redrawn | 80 | 80 | 80 | 0.0001 | 1.933 | 89.9 % | 2.56 / 2.27 |
| 20261008-021850/layer_decode | BUG w4a4: int4 per tensor | scaled | 12 | 12 | 12 | 0.2830 | 1.683 | 66.5 % | 1.93 / 1.70 |
| 20261008-021850/layer_decode | BUG w4a4: row 0 of every GEMM skipped | captured | 4 | 1 | 1 | 0.9663 | 0.259 | 3.0 % | 2.07 / 1.78 |
| 20261008-021850/layer_decode | BUG w4a4: row 0 of every GEMM skipped | redrawn | 80 | 5 | 1 | 0.8462 | 0.533 | 17.0 % | 0.87 / 0.79 |
| 20261008-021850/layer_decode | BUG w4a4: row 0 of every GEMM skipped | scaled | 12 | 3 | 3 | 0.9695 | 0.250 | 7.7 % | 1.28 / 1.15 |
| 20261008-021850/layer_decode | BUG w4a4: KV head 0 dropped | captured | 4 | 3 | 3 | 0.9072 | 0.421 | 9.0 % | 1.45 / 1.19 |
| 20261008-021850/layer_decode | BUG w4a4: KV head 0 dropped | redrawn | 80 | 42 | 14 | 0.7337 | 0.715 | 19.6 % | 1.02 / 0.91 |
| 20261008-021850/layer_decode | BUG w4a4: KV head 0 dropped | scaled | 12 | 0 | 0 | 0.9071 | 0.421 | 11.6 % | 0.89 / 0.78 |
| 20261008-021850/layer_decode | BUG w4a4: q heads 0/last swapped (layout) | captured | 4 | 3 | 1 | 0.9187 | 0.406 | 3.7 % | 0.85 / 0.64 |
| 20261008-021850/layer_decode | BUG w4a4: q heads 0/last swapped (layout) | redrawn | 80 | 47 | 18 | 0.7661 | 0.778 | 20.5 % | 1.06 / 0.92 |
| 20261008-021850/layer_decode | BUG w4a4: q heads 0/last swapped (layout) | scaled | 12 | 4 | 2 | 0.8037 | 0.636 | 10.9 % | 0.74 / 0.64 |
| 20261006-004718/loc_enc_decode | NVFP4 W4A4 | captured | 4 | 0 | 0 | 0.9929 | 0.119 | 0.2 % | 0.50 / 0.37 |
| 20261006-004718/loc_enc_decode | NVFP4 W4A4 | redrawn | 80 | 0 | 0 | 0.9927 | 0.121 | 0.2 % | 0.20 / 0.17 |
| 20261006-004718/loc_enc_decode | NVFP4 W4A4 | scaled | 12 | 0 | 0 | 0.9868 | 0.162 | 0.4 % | 0.22 / 0.19 |
| 20261006-004718/loc_enc_decode | NVFP4 W4A4, per-tensor activations | captured | 4 | 0 | 0 | 0.9920 | 0.126 | 0.1 % | 0.45 / 0.34 |
| 20261006-004718/loc_enc_decode | NVFP4 W4A4, per-tensor activations | redrawn | 80 | 0 | 0 | 0.9938 | 0.112 | 0.3 % | 0.19 / 0.17 |
| 20261006-004718/loc_enc_decode | NVFP4 W4A4, per-tensor activations | scaled | 12 | 0 | 0 | 0.9866 | 0.164 | 0.3 % | 0.21 / 0.18 |
| 20261006-004718/loc_enc_decode | NVFP4 W4A4 + Hadamard 16 | captured | 4 | 0 | 0 | 0.9908 | 0.136 | 0.3 % | 0.51 / 0.39 |
| 20261006-004718/loc_enc_decode | NVFP4 W4A4 + Hadamard 16 | redrawn | 80 | 0 | 0 | 0.9923 | 0.124 | 0.2 % | 0.21 / 0.18 |
| 20261006-004718/loc_enc_decode | NVFP4 W4A4 + Hadamard 16 | scaled | 12 | 0 | 0 | 0.9741 | 0.227 | 0.8 % | 0.31 / 0.27 |
| 20261006-004718/loc_enc_decode | MXFP4 W4A4 | captured | 4 | 0 | 0 | 0.9811 | 0.194 | 0.6 % | 0.62 / 0.47 |
| 20261006-004718/loc_enc_decode | MXFP4 W4A4 | redrawn | 80 | 0 | 0 | 0.9830 | 0.184 | 0.7 % | 0.41 / 0.35 |
| 20261006-004718/loc_enc_decode | MXFP4 W4A4 | scaled | 12 | 0 | 0 | 0.9631 | 0.270 | 2.1 % | 0.44 / 0.38 |
| 20261006-004718/loc_enc_decode | MXFP4 W4A4 + Hadamard 32 | captured | 4 | 0 | 0 | 0.9879 | 0.155 | 0.6 % | 0.62 / 0.47 |
| 20261006-004718/loc_enc_decode | MXFP4 W4A4 + Hadamard 32 | redrawn | 80 | 0 | 0 | 0.9886 | 0.151 | 0.5 % | 0.24 / 0.21 |
| 20261006-004718/loc_enc_decode | MXFP4 W4A4 + Hadamard 32 | scaled | 12 | 0 | 0 | 0.9755 | 0.221 | 0.5 % | 0.30 / 0.26 |
| 20261006-004718/loc_enc_decode | BUG w4a4: weight scale x1.05 | captured | 4 | 0 | 0 | 0.9932 | 0.117 | 0.2 % | 0.42 / 0.32 |
| 20261006-004718/loc_enc_decode | BUG w4a4: weight scale x1.05 | redrawn | 80 | 0 | 0 | 0.9930 | 0.118 | 0.3 % | 0.22 / 0.19 |
| 20261006-004718/loc_enc_decode | BUG w4a4: weight scale x1.05 | scaled | 12 | 0 | 0 | 0.9852 | 0.172 | 0.3 % | 0.23 / 0.20 |
| 20261006-004718/loc_enc_decode | BUG w4a4: weight scale x1.2 | captured | 4 | 0 | 0 | 0.9880 | 0.155 | 0.6 % | 0.52 / 0.40 |
| 20261006-004718/loc_enc_decode | BUG w4a4: weight scale x1.2 | redrawn | 80 | 0 | 0 | 0.9868 | 0.163 | 0.6 % | 0.29 / 0.25 |
| 20261006-004718/loc_enc_decode | BUG w4a4: weight scale x1.2 | scaled | 12 | 0 | 0 | 0.9674 | 0.255 | 2.1 % | 0.34 / 0.30 |
| 20261006-004718/loc_enc_decode | BUG w4a4: nibbles swapped | captured | 4 | 4 | 4 | -0.0416 | 1.619 | 23.2 % | 4.00 / 3.66 |
| 20261006-004718/loc_enc_decode | BUG w4a4: nibbles swapped | redrawn | 80 | 80 | 80 | -0.0640 | 1.636 | 24.0 % | 3.17 / 3.00 |
| 20261006-004718/loc_enc_decode | BUG w4a4: nibbles swapped | scaled | 12 | 12 | 12 | -0.0297 | 1.582 | 21.7 % | 2.83 / 2.67 |
| 20261006-004718/loc_enc_decode | BUG w4a4: activation block scales shifted | captured | 4 | 4 | 4 | 0.0689 | 1.518 | 21.3 % | 5.35 / 4.15 |
| 20261006-004718/loc_enc_decode | BUG w4a4: activation block scales shifted | redrawn | 80 | 80 | 80 | 0.0461 | 1.546 | 22.7 % | 2.93 / 2.65 |
| 20261006-004718/loc_enc_decode | BUG w4a4: activation block scales shifted | scaled | 12 | 12 | 12 | 0.0830 | 1.498 | 20.1 % | 3.09 / 2.88 |
| 20261006-004718/loc_enc_decode | BUG w4a4: first token's outer scale | captured | 4 | 0 | 0 | 0.9626 | 0.273 | 0.4 % | 0.95 / 0.72 |
| 20261006-004718/loc_enc_decode | BUG w4a4: first token's outer scale | redrawn | 80 | 0 | 0 | 0.9496 | 0.317 | 0.8 % | 0.59 / 0.51 |
| 20261006-004718/loc_enc_decode | BUG w4a4: first token's outer scale | scaled | 12 | 0 | 0 | 0.9574 | 0.290 | 1.5 % | 0.38 / 0.33 |
| 20261006-004718/loc_enc_decode | BUG w4a4: activation codes truncated | captured | 4 | 0 | 0 | 0.9734 | 0.231 | 1.1 % | 0.78 / 0.59 |
| 20261006-004718/loc_enc_decode | BUG w4a4: activation codes truncated | redrawn | 80 | 0 | 0 | 0.9778 | 0.212 | 0.9 % | 0.46 / 0.40 |
| 20261006-004718/loc_enc_decode | BUG w4a4: activation codes truncated | scaled | 12 | 0 | 0 | 0.9282 | 0.374 | 3.0 % | 0.49 / 0.44 |
| 20261006-004718/loc_enc_decode | BUG w4a4: cached activation scales | captured | 4 | 1 | 0 | 0.9768 | 0.215 | 0.5 % | 1.08 / 0.83 |
| 20261006-004718/loc_enc_decode | BUG w4a4: cached activation scales | redrawn | 80 | 0 | 0 | 0.9768 | 0.215 | 0.4 % | 0.49 / 0.42 |
| 20261006-004718/loc_enc_decode | BUG w4a4: cached activation scales | scaled | 12 | 1 | 0 | 0.9085 | 0.421 | 4.1 % | 1.10 / 0.95 |
| 20261006-004718/loc_enc_decode | BUG w4a4: int4 per tensor | captured | 4 | 4 | 0 | 0.9408 | 0.343 | 1.1 % | 1.11 / 0.83 |
| 20261006-004718/loc_enc_decode | BUG w4a4: int4 per tensor | redrawn | 80 | 0 | 0 | 0.9375 | 0.352 | 1.2 % | 0.83 / 0.73 |
| 20261006-004718/loc_enc_decode | BUG w4a4: int4 per tensor | scaled | 12 | 4 | 4 | 0.7444 | 0.706 | 3.2 % | 0.85 / 0.74 |
| 20261006-004718/loc_enc_decode | BUG w4a4: row 0 of every GEMM skipped | captured | 4 | 0 | 0 | 0.9912 | 0.133 | 0.5 % | 0.56 / 0.44 |
| 20261006-004718/loc_enc_decode | BUG w4a4: row 0 of every GEMM skipped | redrawn | 80 | 0 | 0 | 0.9906 | 0.137 | 0.4 % | 0.42 / 0.36 |
| 20261006-004718/loc_enc_decode | BUG w4a4: row 0 of every GEMM skipped | scaled | 12 | 0 | 0 | 0.9817 | 0.191 | 0.6 % | 0.38 / 0.33 |
| 20261006-004718/loc_enc_decode | BUG w4a4: KV head 0 dropped | captured | 4 | 4 | 4 | 0.7712 | 0.693 | 4.6 % | 2.65 / 2.06 |
| 20261006-004718/loc_enc_decode | BUG w4a4: KV head 0 dropped | redrawn | 80 | 80 | 80 | 0.7793 | 0.680 | 4.6 % | 1.32 / 1.18 |
| 20261006-004718/loc_enc_decode | BUG w4a4: KV head 0 dropped | scaled | 12 | 12 | 10 | 0.6005 | 0.868 | 6.2 % | 1.39 / 1.19 |
| 20261006-004718/loc_enc_decode | BUG w4a4: q heads 0/last swapped (layout) | captured | 4 | 0 | 0 | 0.9879 | 0.156 | 0.2 % | 0.68 / 0.53 |
| 20261006-004718/loc_enc_decode | BUG w4a4: q heads 0/last swapped (layout) | redrawn | 80 | 0 | 0 | 0.9863 | 0.166 | 0.5 % | 0.35 / 0.30 |
| 20261006-004718/loc_enc_decode | BUG w4a4: q heads 0/last swapped (layout) | scaled | 12 | 4 | 4 | 0.6890 | 0.792 | 0.7 % | 1.29 / 1.12 |
