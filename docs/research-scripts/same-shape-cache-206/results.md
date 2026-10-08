| capture | numerics | captured fails (NL · relaxed) | redrawn fails | scaled fails | scaled: max element ratio NL / relaxed |
|---|---|---|---|---|---|
| Qwen3-0.6B decoder layer, decode (M = 1), returned static cache | FP8 weights | 0 · 0 /4 | 8 · 0 /80 -> **9 · 0 /80** | 4 · 3 /12 -> **0 · 0 /12** | 2.31 / 1.30 -> 0.24 / 0.13 |
| Qwen3-0.6B decoder layer, decode (M = 1), returned static cache | FP8 W8A8 | 0 · 0 /4 | 63 · 3 /80 -> **63 · 4 /80** | 6 · 4 /12 -> **2 · 0 /12** | 3.14 / 1.68 -> 0.40 / 0.29 |
| Qwen3-0.6B decoder layer, decode (M = 1), returned static cache | MXFP8 W8A8 | 0 · 0 /4 | 68 · 4 /80 -> **68 · 6 /80** | 5 · 4 /12 -> **1 · 0 /12** | 3.95 / 2.06 -> 0.39 / 0.26 |
| Qwen3-0.6B decoder layer, decode (M = 1), returned static cache | NVFP4 weights | 0 · 0 /4 | 0 · 0 /80 | 4 · 4 /12 -> **0 · 0 /12** | 2.55 / 2.16 -> 0.27 / 0.25 |
| Qwen3-0.6B decoder layer, decode (M = 1), returned static cache | MXFP4 weights | 0 · 0 /4 | 5 · 1 /80 -> **6 · 1 /80** | 4 · 4 /12 -> **0 · 0 /12** | 3.17 / 2.64 -> 0.31 / 0.26 |
| Qwen3-0.6B decoder layer, decode (M = 1), returned static cache | INT8 weights | 0 · 0 /4 | 0 · 0 /80 | 1 · 0 /12 -> **0 · 0 /12** | 1.02 / 0.51 -> 0.10 / 0.05 |
| Qwen3-0.6B decoder layer, decode (M = 1), returned static cache | INT8 W8A8 | 0 · 0 /4 | 4 · 0 /80 | 4 · 3 /12 -> **0 · 0 /12** | 2.50 / 1.39 -> 0.31 / 0.17 |

| capture | broken variant | still rejected (NL · relaxed) | captured fails (NL · relaxed) | redrawn fails | scaled fails |
|---|---|---|---|---|---|
| Qwen3-0.6B decoder layer, decode (M = 1), returned static cache | fp8: scales x1.05 | yes · yes | 4 · 4 /4 | 80 · 80 /80 | 12 · 12 /12 |
| Qwen3-0.6B decoder layer, decode (M = 1), returned static cache | fp8: scales x1.2 | yes · yes | 4 · 4 /4 | 80 · 80 /80 | 12 · 12 /12 |
| Qwen3-0.6B decoder layer, decode (M = 1), returned static cache | fp8: int4 per tensor | yes · yes | 4 · 4 /4 | 80 · 80 /80 | 12 · 12 /12 |
| Qwen3-0.6B decoder layer, decode (M = 1), returned static cache | fp8: row 0 of every GEMM skipped | yes · yes | 4 · 3 /4 -> **4 · 2 /4** | 73 · 30 /80 -> **74 · 29 /80** | 10 · 10 /12 -> **10 · 8 /12** |
| Qwen3-0.6B decoder layer, decode (M = 1), returned static cache | fp8: KV head 0 dropped | yes · yes | 4 · 4 /4 | 80 · 80 /80 | 11 · 10 /12 -> **12 · 12 /12** |
| Qwen3-0.6B decoder layer, decode (M = 1), returned static cache | fp8: q heads 0/last swapped (layout) | yes · yes | 4 · 3 /4 | 80 · 80 /80 | 12 · 10 /12 -> **8 · 7 /12** |
| Qwen3-0.6B decoder layer, decode (M = 1), returned static cache | w8a8: first token's activation scale | yes · yes | 0 · 0 /4 | 66 · 11 /80 -> **66 · 12 /80** | 6 · 4 /12 -> **2 · 0 /12** |
| Qwen3-0.6B decoder layer, decode (M = 1), returned static cache | fp4: nibbles swapped | yes · yes | 4 · 4 /4 | 80 · 80 /80 | 12 · 12 /12 |
| Qwen3-0.6B decoder layer, decode (M = 1), returned static cache | fp4: scales x1.2 | yes · yes | 4 · 4 /4 | 80 · 80 /80 | 12 · 12 /12 |
| Qwen3-0.6B decoder layer, decode (M = 1), returned static cache | fp4: int4 per tensor | yes · yes | 4 · 4 /4 | 80 · 80 /80 | 12 · 12 /12 |
| Qwen3-0.6B decoder layer, decode (M = 1), returned static cache | fp4: row 0 of every GEMM skipped | yes · yes | 1 · 1 /4 | 2 · 0 /80 | 8 · 7 /12 -> **4 · 3 /12** |
| Qwen3-0.6B decoder layer, decode (M = 1), returned static cache | fp4: KV head 0 dropped | yes · yes | 4 · 3 /4 -> **3 · 3 /4** | 44 · 20 /80 -> **76 · 20 /80** | 8 · 7 /12 -> **10 · 0 /12** |
| Qwen3-0.6B decoder layer, decode (M = 1), returned static cache | fp4: q heads 0/last swapped (layout) | yes · yes | 2 · 1 /4 | 69 · 33 /80 | 8 · 8 /12 -> **4 · 4 /12** |
| Qwen3-0.6B decoder layer, decode (M = 1), returned static cache | w8a8: cached activation scales | yes · yes | 0 · 0 /4 | 71 · 16 /80 -> **71 · 17 /80** | 12 · 11 /12 -> **8 · 8 /12** |
| Qwen3-0.6B decoder layer, decode (M = 1), returned static cache | int8: scales x1.05 | yes · yes | 4 · 4 /4 | 80 · 80 /80 | 12 · 12 /12 |
| Qwen3-0.6B decoder layer, decode (M = 1), returned static cache | int8 w8a8: scales x1.05 | yes · yes | 4 · 4 /4 | 80 · 80 /80 | 12 · 12 /12 |
