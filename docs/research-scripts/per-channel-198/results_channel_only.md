| capture | numerics | redrawn fails, NL: before -> after | redrawn fails, relaxed: before -> after | after, redrawn: min cosine / max rel L2 / max norm change / max element ratio (relaxed) | captured, scaled fails (NL · relaxed) |
|---|---|---|---|---|---|
| Qwen3-0.6B decoder layer, decode (M = 1) | FP8 weights | 70/80 -> **50/80** | 54/80 -> **3/80** | 0.9211 / 0.390 / 7.4 % / 2.08 | 0/4 · 0/4 · 3/12 · 0/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | FP8 W8A8 | 74/80 -> **74/80** | 66/80 -> **28/80** | 0.9113 / 0.418 / 7.2 % / 1.05 | 0/4 · 0/4 · 6/12 · 1/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | MXFP8 W8A8 | 75/80 -> **79/80** | 70/80 -> **28/80** | 0.9060 / 0.430 / 5.9 % / 1.21 | 0/4 · 0/4 · 5/12 · 2/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | NVFP4 weights | 71/80 -> **20/80** | 59/80 -> **6/80** | 0.7319 / 0.731 / 15.1 % / 0.99 | 0/4 · 0/4 · 4/12 · 4/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | MXFP4 weights | 77/80 -> **49/80** | 68/80 -> **22/80** | 0.6216 / 0.876 / 11.9 % / 1.03 | 0/4 · 0/4 · 4/12 · 4/12 |

| capture | broken variant | still rejected (NL · relaxed) | redrawn fails, NL: before -> after | redrawn fails, relaxed: before -> after | captured fails (NL · relaxed) | scaled fails (NL · relaxed) |
|---|---|---|---|---|---|---|
| Qwen3-0.6B decoder layer, decode (M = 1) | fp8: scales x1.05 | yes · yes | 80/80 -> 80/80 | 80/80 -> 80/80 | 4/4 · 4/4 | 12/12 · 12/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | fp8: scales x1.2 | yes · yes | 80/80 -> 80/80 | 80/80 -> 80/80 | 4/4 · 4/4 | 12/12 · 12/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | fp8: int4 per tensor | yes · yes | 80/80 -> 80/80 | 80/80 -> 80/80 | 4/4 · 4/4 | 12/12 · 12/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | fp8: row 0 of every GEMM skipped | yes · yes | 75/80 -> 76/80 | 60/80 -> 31/80 | 4/4 · 3/4 | 10/12 · 10/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | fp8: KV head 0 dropped | yes · yes | 80/80 -> 80/80 | 80/80 -> 80/80 | 4/4 · 4/4 | 11/12 · 10/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | fp8: q heads 0/last swapped (layout) | yes · yes | 80/80 -> 80/80 | 80/80 -> 80/80 | 4/4 · 3/4 | 11/12 · 7/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | w8a8: first token's activation scale | yes · yes | 74/80 -> 75/80 | 65/80 -> 30/80 | 0/4 · 0/4 | 6/12 · 1/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | fp4: nibbles swapped | yes · yes | 80/80 -> 80/80 | 80/80 -> 80/80 | 4/4 · 4/4 | 12/12 · 12/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | fp4: scales x1.2 | yes · yes | 80/80 -> 80/80 | 80/80 -> 80/80 | 4/4 · 4/4 | 12/12 · 12/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | fp4: int4 per tensor | yes · yes | 80/80 -> 80/80 | 80/80 -> 80/80 | 4/4 · 4/4 | 12/12 · 12/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | fp4: row 0 of every GEMM skipped | yes · yes | 71/80 -> 21/80 | 59/80 -> 6/80 | 1/4 · 1/4 | 8/12 · 7/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | fp4: KV head 0 dropped | yes · yes | 72/80 -> 41/80 | 61/80 -> 16/80 | 3/4 · 3/4 | 7/12 · 4/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | fp4: q heads 0/last swapped (layout) | yes · yes | 80/80 -> 78/80 | 78/80 -> 66/80 | 2/4 · 1/4 | 8/12 · 8/12 |
| Qwen3-0.6B decoder layer, decode (M = 1) | w8a8: cached activation scales | yes · yes | 77/80 -> 78/80 | 67/80 -> 36/80 | 0/4 · 0/4 | 12/12 · 11/12 |
