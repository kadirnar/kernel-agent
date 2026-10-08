| capture | numerics | redrawn fails, NL: before -> after | redrawn fails, relaxed: before -> after | after, redrawn: min cosine / max rel L2 / max norm change / max element ratio (relaxed) | captured, scaled fails (NL · relaxed) |
|---|---|---|---|---|---|
| Qwen3-0.6B decoder layer, decode (M = 1) | FP8 weights | 70/80 -> **4/80** | 54/80 -> **0/80** | 0.9953 / 0.096 / 3.2 % / 0.26 | 0/4 · 0/4 · 3/12 · 0/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | FP8 W8A8 | 74/80 -> **65/80** | 66/80 -> **0/80** | 0.9919 / 0.130 / 3.8 % / 0.43 | 0/4 · 0/4 · 6/12 · 1/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | MXFP8 W8A8 | 75/80 -> **65/80** | 70/80 -> **1/80** | 0.9844 / 0.176 / 3.0 % / 0.37 | 0/4 · 0/4 · 5/12 · 2/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | NVFP4 weights | 71/80 -> **0/80** | 59/80 -> **0/80** | 0.9484 / 0.319 / 7.8 % / 0.53 | 0/4 · 0/4 · 4/12 · 4/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | MXFP4 weights | 77/80 -> **6/80** | 68/80 -> **0/80** | 0.9035 / 0.429 / 14.3 % / 0.50 | 0/4 · 0/4 · 4/12 · 4/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | INT8 weights | 27/80 -> **0/80** | 2/80 -> **0/80** | 0.9994 / 0.033 / 0.7 % / 0.10 | 0/4 · 0/4 · 0/12 · 0/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | INT8 W8A8 | 64/80 -> **4/80** | 38/80 -> **0/80** | 0.9923 / 0.125 / 1.7 % / 0.28 | 0/4 · 0/4 · 4/12 · 1/12 |
| VoxCPM2 LocDiT layer (M = 352, 176) | FP8 weights | 0/40 -> **0/40** | 0/40 -> **0/40** | 0.9997 / 0.025 / 1.0 % / 0.09 | 0/2 · 0/2 · 0/6 · 0/6 |
| VoxCPM2 LocDiT layer (M = 352, 176) | FP8 W8A8 | 0/40 -> **0/40** | 0/40 -> **0/40** | 0.9994 / 0.034 / 1.1 % / 0.12 | 0/2 · 0/2 · 0/6 · 0/6 |
| VoxCPM2 LocDiT layer (M = 352, 176) | MXFP8 W8A8 | 0/40 -> **0/40** | 0/40 -> **0/40** | 0.9994 / 0.035 / 1.1 % / 0.14 | 0/2 · 0/2 · 0/6 · 0/6 |
| VoxCPM2 LocDiT layer (M = 352, 176) | NVFP4 weights | 0/40 -> **0/40** | 0/40 -> **0/40** | 0.9947 / 0.103 / 0.6 % / 0.18 | 0/2 · 0/2 · 0/6 · 0/6 |
| VoxCPM2 LocDiT layer (M = 352, 176) | MXFP4 weights | 0/40 -> **0/40** | 0/40 -> **0/40** | 0.9907 / 0.137 / 6.2 % / 0.24 | 2/2 · 0/2 · 0/6 · 0/6 |
| VoxCPM2 LocDiT layer (M = 352, 176) | INT8 weights | 0/40 -> **0/40** | 0/40 -> **0/40** | 0.9999 / 0.012 / 0.1 % / 0.06 | 0/2 · 0/2 · 0/6 · 0/6 |
| VoxCPM2 LocDiT layer (M = 352, 176) | INT8 W8A8 | 0/40 -> **0/40** | 0/40 -> **0/40** | 0.9982 / 0.060 / 1.5 % / 0.33 | 1/2 · 0/2 · 2/6 · 0/6 |
| VoxCPM2 base-LM decode layer (M = 1) | FP8 weights | 0/60 -> **0/60** | 0/60 -> **0/60** | 0.9992 / 0.040 / 0.5 % / 0.13 | 0/3 · 0/3 · 0/9 · 0/9 |
| VoxCPM2 base-LM decode layer (M = 1) | FP8 W8A8 | 0/60 -> **0/60** | 0/60 -> **0/60** | 0.9987 / 0.051 / 1.7 % / 0.15 | 0/3 · 0/3 · 0/9 · 0/9 |
| VoxCPM2 base-LM decode layer (M = 1) | MXFP8 W8A8 | 5/60 -> **11/60** | 0/60 -> **0/60** | 0.9980 / 0.066 / 4.5 % / 0.20 | 1/3 · 0/3 · 1/9 · 0/9 |
| VoxCPM2 base-LM decode layer (M = 1) | NVFP4 weights | 0/60 -> **0/60** | 0/60 -> **0/60** | 0.9975 / 0.070 / 0.6 % / 0.09 | 0/3 · 0/3 · 0/9 · 0/9 |
| VoxCPM2 base-LM decode layer (M = 1) | MXFP4 weights | 0/60 -> **0/60** | 0/60 -> **0/60** | 0.9840 / 0.178 / 1.6 % / 0.25 | 0/3 · 0/3 · 0/9 · 0/9 |
| VoxCPM2 base-LM decode layer (M = 1) | INT8 weights | 0/60 -> **0/60** | 0/60 -> **0/60** | 1.0000 / 0.009 / 0.1 % / 0.03 | 0/3 · 0/3 · 0/9 · 0/9 |
| VoxCPM2 base-LM decode layer (M = 1) | INT8 W8A8 | 1/60 -> **2/60** | 0/60 -> **0/60** | 0.9992 / 0.041 / 1.3 % / 0.11 | 0/3 · 0/3 · 3/9 · 3/9 |
| fp8_kv decode attention (synthetic, #145) | FP8 KV cache | 0/192 -> **5/192** | 0/192 -> **0/192** | 0.9934 / 0.115 / 4.0 % / 0.36 | 0/48 · 0/48 · 0/144 · 0/144 |

| capture | broken variant | still rejected (NL · relaxed) | redrawn fails, NL: before -> after | redrawn fails, relaxed: before -> after | captured fails (NL · relaxed) | scaled fails (NL · relaxed) |
|---|---|---|---|---|---|---|
| Qwen3-0.6B decoder layer, decode (M = 1) | fp8: scales x1.05 | yes · yes | 80/80 -> 80/80 | 80/80 -> 80/80 | 4/4 · 4/4 | 12/12 · 12/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | fp8: scales x1.2 | yes · yes | 80/80 -> 80/80 | 80/80 -> 80/80 | 4/4 · 4/4 | 12/12 · 12/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | fp8: int4 per tensor | yes · yes | 80/80 -> 80/80 | 80/80 -> 80/80 | 4/4 · 4/4 | 12/12 · 12/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | fp8: row 0 of every GEMM skipped | yes · yes | 75/80 -> 69/80 | 60/80 -> 29/80 | 4/4 · 3/4 | 10/12 · 10/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | fp8: KV head 0 dropped | yes · yes | 80/80 -> 80/80 | 80/80 -> 80/80 | 4/4 · 4/4 | 11/12 · 10/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | fp8: q heads 0/last swapped (layout) | yes · yes | 80/80 -> 80/80 | 80/80 -> 80/80 | 4/4 · 3/4 | 11/12 · 7/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | w8a8: first token's activation scale | yes · yes | 74/80 -> 66/80 | 65/80 -> 7/80 | 0/4 · 0/4 | 6/12 · 1/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | fp4: nibbles swapped | yes · yes | 80/80 -> 80/80 | 80/80 -> 80/80 | 4/4 · 4/4 | 12/12 · 12/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | fp4: scales x1.2 | yes · yes | 80/80 -> 80/80 | 80/80 -> 80/80 | 4/4 · 4/4 | 12/12 · 12/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | fp4: int4 per tensor | yes · yes | 80/80 -> 80/80 | 80/80 -> 80/80 | 4/4 · 4/4 | 12/12 · 12/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | fp4: row 0 of every GEMM skipped | yes · yes | 71/80 -> 3/80 | 59/80 -> 1/80 | 1/4 · 1/4 | 8/12 · 7/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | fp4: KV head 0 dropped | yes · yes | 72/80 -> 53/80 | 61/80 -> 21/80 | 3/4 · 3/4 | 7/12 · 4/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | fp4: q heads 0/last swapped (layout) | yes · yes | 80/80 -> 63/80 | 78/80 -> 33/80 | 2/4 · 1/4 | 8/12 · 8/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | w8a8: cached activation scales | yes · yes | 77/80 -> 73/80 | 67/80 -> 11/80 | 0/4 · 0/4 | 12/12 · 11/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | int8: scales x1.05 | yes · yes | 80/80 -> 80/80 | 80/80 -> 80/80 | 4/4 · 4/4 | 12/12 · 12/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | int8 w8a8: scales x1.05 | yes · yes | 80/80 -> 80/80 | 80/80 -> 80/80 | 4/4 · 4/4 | 12/12 · 12/12 |
| VoxCPM2 LocDiT layer (M = 352, 176) | fp8: scales x1.05 | yes · yes | 40/40 -> 40/40 | 0/40 -> 0/40 | 2/2 · 2/2 | 6/6 · 4/6 |
| VoxCPM2 LocDiT layer (M = 352, 176) | fp8: scales x1.2 | yes · yes | 40/40 -> 40/40 | 40/40 -> 40/40 | 2/2 · 2/2 | 6/6 · 6/6 |
| VoxCPM2 LocDiT layer (M = 352, 176) | fp8: int4 per tensor | yes · yes | 40/40 -> 40/40 | 40/40 -> 40/40 | 2/2 · 2/2 | 6/6 · 6/6 |
| VoxCPM2 LocDiT layer (M = 352, 176) | fp8: row 0 of every GEMM skipped | yes · yes | 40/40 -> 40/40 | 40/40 -> 40/40 | 2/2 · 2/2 | 2/6 · 0/6 |
| VoxCPM2 LocDiT layer (M = 352, 176) | fp8: KV head 0 dropped | yes · yes | 40/40 -> 40/40 | 40/40 -> 40/40 | 2/2 · 2/2 | 6/6 · 6/6 |
| VoxCPM2 LocDiT layer (M = 352, 176) | fp8: q heads 0/last swapped (layout) | yes · yes | 0/40 -> 0/40 | 0/40 -> 0/40 | 0/2 · 0/2 | 2/6 · 2/6 |
| VoxCPM2 LocDiT layer (M = 352, 176) | w8a8: first token's activation scale | yes · yes | 20/40 -> 36/40 | 5/40 -> 27/40 | 2/2 · 2/2 | 6/6 · 4/6 |
| VoxCPM2 LocDiT layer (M = 352, 176) | fp4: nibbles swapped | yes · yes | 40/40 -> 40/40 | 40/40 -> 40/40 | 2/2 · 2/2 | 6/6 · 6/6 |
| VoxCPM2 LocDiT layer (M = 352, 176) | fp4: scales x1.2 | yes · yes | 40/40 -> 40/40 | 40/40 -> 40/40 | 2/2 · 2/2 | 6/6 · 6/6 |
| VoxCPM2 LocDiT layer (M = 352, 176) | fp4: int4 per tensor | yes · yes | 0/40 -> 40/40 | 0/40 -> 1/40 | 2/2 · 2/2 | 2/6 · 2/6 |
| VoxCPM2 LocDiT layer (M = 352, 176) | fp4: row 0 of every GEMM skipped | **no** (before: yes) · **no** (before: yes) | 23/40 -> 0/40 | 6/40 -> 0/40 | 0/2 · 0/2 | 0/6 · 0/6 |
| VoxCPM2 LocDiT layer (M = 352, 176) | fp4: KV head 0 dropped | yes · yes | 40/40 -> 40/40 | 40/40 -> 40/40 | 2/2 · 2/2 | 6/6 · 6/6 |
| VoxCPM2 LocDiT layer (M = 352, 176) | fp4: q heads 0/last swapped (layout) | **no** · **no** | 0/40 -> 0/40 | 0/40 -> 0/40 | 0/2 · 0/2 | 0/6 · 0/6 |
| VoxCPM2 LocDiT layer (M = 352, 176) | w8a8: cached activation scales | yes · yes | 37/40 -> 40/40 | 14/40 -> 40/40 | 0/2 · 0/2 | 2/6 · 0/6 |
| VoxCPM2 LocDiT layer (M = 352, 176) | int8: scales x1.05 | yes · yes | 40/40 -> 40/40 | 0/40 -> 0/40 | 2/2 · 2/2 | 6/6 · 6/6 |
| VoxCPM2 LocDiT layer (M = 352, 176) | int8 w8a8: scales x1.05 | yes · yes | 40/40 -> 40/40 | 0/40 -> 0/40 | 2/2 · 2/2 | 6/6 · 5/6 |
| VoxCPM2 base-LM decode layer (M = 1) | fp8: scales x1.05 | yes · yes | 60/60 -> 60/60 | 0/60 -> 0/60 | 3/3 · 3/3 | 9/9 · 6/9 |
| VoxCPM2 base-LM decode layer (M = 1) | fp8: scales x1.2 | yes · yes | 60/60 -> 60/60 | 60/60 -> 60/60 | 3/3 · 3/3 | 9/9 · 9/9 |
| VoxCPM2 base-LM decode layer (M = 1) | fp8: int4 per tensor | yes · yes | 60/60 -> 60/60 | 59/60 -> 60/60 | 3/3 · 3/3 | 9/9 · 7/9 |
| VoxCPM2 base-LM decode layer (M = 1) | fp8: row 0 of every GEMM skipped | yes · yes | 26/60 -> 36/60 | 3/60 -> 5/60 | 3/3 · 2/3 | 6/9 · 0/9 |
| VoxCPM2 base-LM decode layer (M = 1) | fp8: KV head 0 dropped | yes · yes | 58/60 -> 57/60 | 50/60 -> 51/60 | 3/3 · 3/3 | 9/9 · 7/9 |
| VoxCPM2 base-LM decode layer (M = 1) | fp8: q heads 0/last swapped (layout) | **no** · **no** | 0/60 -> 0/60 | 0/60 -> 0/60 | 0/3 · 0/3 | 0/9 · 0/9 |
| VoxCPM2 base-LM decode layer (M = 1) | w8a8: first token's activation scale | **no** · **no** | 0/60 -> 0/60 | 0/60 -> 0/60 | 0/3 · 0/3 | 0/9 · 0/9 |
| VoxCPM2 base-LM decode layer (M = 1) | fp4: nibbles swapped | yes · yes | 60/60 -> 60/60 | 60/60 -> 60/60 | 3/3 · 3/3 | 9/9 · 9/9 |
| VoxCPM2 base-LM decode layer (M = 1) | fp4: scales x1.2 | yes · yes | 60/60 -> 60/60 | 60/60 -> 60/60 | 3/3 · 3/3 | 9/9 · 9/9 |
| VoxCPM2 base-LM decode layer (M = 1) | fp4: int4 per tensor | yes · yes | 1/60 -> 2/60 | 0/60 -> 0/60 | 2/3 · 1/3 | 4/9 · 3/9 |
| VoxCPM2 base-LM decode layer (M = 1) | fp4: row 0 of every GEMM skipped | **no** · **no** | 0/60 -> 0/60 | 0/60 -> 0/60 | 0/3 · 0/3 | 0/9 · 0/9 |
| VoxCPM2 base-LM decode layer (M = 1) | fp4: KV head 0 dropped | yes · yes | 25/60 -> 26/60 | 22/60 -> 21/60 | 3/3 · 3/3 | 6/9 · 5/9 |
| VoxCPM2 base-LM decode layer (M = 1) | fp4: q heads 0/last swapped (layout) | **no** · **no** | 0/60 -> 0/60 | 0/60 -> 0/60 | 0/3 · 0/3 | 0/9 · 0/9 |
| VoxCPM2 base-LM decode layer (M = 1) | w8a8: cached activation scales | yes · yes | 44/60 -> 37/60 | 38/60 -> 33/60 | 2/3 · 1/3 | 3/9 · 0/9 |
| VoxCPM2 base-LM decode layer (M = 1) | int8: scales x1.05 | yes · yes | 60/60 -> 60/60 | 0/60 -> 0/60 | 3/3 · 3/3 | 9/9 · 6/9 |
| VoxCPM2 base-LM decode layer (M = 1) | int8 w8a8: scales x1.05 | yes · yes | 60/60 -> 60/60 | 0/60 -> 2/60 | 3/3 · 3/3 | 9/9 · 6/9 |
| fp8_kv decode attention (synthetic, #145) | kv: v scales x1.05 | yes · yes | 192/192 -> 181/192 | 43/192 -> 29/192 | 48/48 · 48/48 | 132/144 · 0/144 |
| fp8_kv decode attention (synthetic, #145) | kv: v scales x1.2 | yes · yes | 192/192 -> 192/192 | 192/192 -> 192/192 | 48/48 · 48/48 | 132/144 · 132/144 |
| fp8_kv decode attention (synthetic, #145) | kv: first token's scale | yes · yes | 180/192 -> 179/192 | 175/192 -> 177/192 | 48/48 · 44/48 | 129/144 · 57/144 |
