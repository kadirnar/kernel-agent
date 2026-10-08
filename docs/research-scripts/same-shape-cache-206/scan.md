| capture | cases | outputs with an input of their shape | grown caches | written caches (box of shape) | the others: most elements left as they were |
|---|---|---|---|---|---|
| 20261005-192504/dit_layer_fp8 | 1 | 1 | 0 | none | 0.26% |
| 20261005-192504/lm_step_fp8 | 3 | 3 | 0 | none | 0.88% |
| 20261005-192504/lm_step_fp8__fp4_weights | 3 | 3 | 0 | none | 0.88% |
| 20261005-192504/enc_dit_stack_fp8 | 1 | 1 | 0 | none | 0.06% |
| 20261005-192504/native_residual_lm | 5 | 5 | 0 | none | 0.05% |
| 20261005-192504/native_base_lm | 5 | 5 | 0 | none | 0.05% |
| 20261005-192504/lm_stack_fp4 | 3 | 3 | 0 | none | 0.05% |
| 20261006-004718/dit_layer | 2 | 2 | 0 | none | 0.22% |
| 20261006-004718/dit_layer__fp8_w8a8 | 2 | 2 | 0 | none | 0.22% |
| 20261006-004718/loc_enc_decode | 4 | 0 | 0 | none | - |
| 20261006-004718/vae_decoder | 3 | 0 | 0 | none | - |
| 20261008-021850/layer_decode | 4 | 4 | 0 | none | 0.10% |
| 20261008-063823/decoder_layer_decode | 4 | 4 | 0 | none | 0.10% |
| 20261008-063823/attention_decode | 4 | 4 | 0 | none | 0.20% |
| 20261008-063823/mlp_decode | 4 | 4 | 0 | none | 0.10% |
| qwen3-static/layer_decode_static | 4 | 12 | 0 | [1, 8, 1, 128] of [1, 8, 1024, 128] x2, [1, 8, 1, 128] of [1, 8, 1086, 128] x2, [1, 8, 1, 128] of [1, 8, 1148, 128] x2, [2, 8, 1, 128] of [2, 8, 602, 128] x2 | 0.10% |
