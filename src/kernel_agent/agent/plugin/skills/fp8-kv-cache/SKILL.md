---
name: fp8-kv-cache
description: FP8 KV caches (precision fp8_kv, opt-in) — when the cache's share of a decode step pays, per-(token, KV head) e4m3 storage, scales folded into the scores and P, accuracy, speed by context length. Use for decode attention whose spec says fp8_kv.
---

# FP8 KV cache (`fp8_kv`, opt-in)

Read the `precision-tiers` skill first.

For a target whose spec says `"precision": "fp8_kv"`: decode attention whose time
goes into reading the KV cache. Not in a near-lossless run's default precisions
(`--precisions ...,fp8_kv`): it pays only where the cache is a large share of what a
decode step reads, which depends on the model and the workload, so measure it on the
model at hand: the *Ceilings* table's *KV GB* of the decode rows against their weight
bytes, or `kv_quant.kv_cache_share(batch=, tokens=, layers=, kv_heads=, head_dim=,
other_bytes=)` from the model's config and the workload's context lengths
(`other_bytes`: the weights and activations a step streams). Thousands of cached
tokens at a real batch, not tens.

* **Storage**: K and V in e4m3 with one fp32 scale per (token, KV head): `amax(|row|)
  / 448` over the head dimension, written once when tokens are appended
  (`kv_quant.Fp8KVCache.append`, `quantize_fp8_kv`); never re-quantise the cache per
  step, never a static (calibrated) per-tensor scale (one text's scale was exceeded by
  another's in ~50 % of a model's layers; vLLM / FlashInfer use per-tensor scales).
* **Math**: Q, the fp32 softmax and the output as in eager; fold the K scale into the
  scores (`(q · codes) * k_scale * sm_scale`: bf16 x e4m3 products are exact in fp32)
  and the V scale into P (rounded to bf16 for the P·V MMA, as FlashAttention rounds P).
  Reference / fallback: `kv_quant.fp8_kv_attention` (grouped-query heads, per-sequence
  lengths). Report `kv_quant.fp8_kv_error(k, v)`.
* **Accuracy** (docs/FP8.md §5, real captures): e4m3 K / V per token and head change a
  decoder layer's output by <= 0.0009 relative L2 (the op itself 0.006-0.015), far
  inside the near-lossless tier (0.08); broken scales (V scale x 1.05, one token's
  scale for all) fail it.
* **Speed** (RTX 5070 Ti, Triton, batch 16, 16 query / 2 KV heads, head dim 128): one
  program per (sequence, KV head) reading e4m3 K / V ran 0.90x of bf16 at 77 cached
  tokens (latency bound: the conversion costs more than the bytes save), 1.21x at 512,
  1.29x at 2048, 1.31x at 8192 (~550 GB/s of codes: 32 programs on 70 SMs). Split the
  cache over programs (flash-decoding) to approach the 2x of half the bytes:
  `examples/triton_fp8_kv_decode.py` (splits for ~4 programs per SM, fp32 partials
  merged by a second kernel; not measured yet).
* **Ceilings**: decode rows count the KV-cache reads (*KV GB*) in their floors: the
  share to compare with the weight bytes. There is no `fp8_kv` floor column (it would
  halve the KV bytes); the target's floor is its exact one.

## Examples and sources

* Examples: `triton_fp8_kv_decode.py`. All in kernel-agent's examples directory (`kernel_agent/agent/examples/`; a session's prompt gives the directory): copy their structure; an example's `ARCHS` names the GPUs it runs on.
* Sources: the `documentation-sources` skill's `sources.md`, sections "Attention and LLM kernels (reference code)"; "Low precision (formats, scaling, accuracy)".
* Code: `kernel_agent.kernels.kv_quant` (`quantize_fp8_kv`, `Fp8KVCache`, `fp8_kv_attention`, `fp8_kv_error`, `kv_cache_share`).
