# Algorithm-level changes (model transforms)

Part of the `optimisation-playbook` skill; the systems agent's patterns are in the `systems-patterns` and `speculative-decoding` skills.

These change *how* the model runs, not the math, and often beat any kernel:
* `model.generation_config.cache_implementation = "static"` + `torch.compile` /
  CUDA graphs for decode loops (removes launch overhead almost entirely).
* `torch.backends.cuda.matmul.allow_tf32`, SDPA backend choice
  (`torch.nn.attention.sdpa_kernel`), `channels_last` for conv nets.
* Precompute constant tensors (rotary tables, masks, positional embeddings).
* Fusing linear layers that share an input (QKV, gate/up) — done in `build()`.
* Avoid recomputation (cache encoder output, cross-attention K/V).
* Exact speculative / lookahead decoding keeps greedy outputs identical; its gain depends
  on the content, so it is also timed on a diverse input set (skill `speculative-decoding`).
* Host syncs in a generation loop (the profile's "Host synchronisation" table: `.item()` /
  `.cpu()` of a stop flag, `torch.tensor(..., device=...)` per step, pageable copies, with
  call sites): read flags asynchronously (`Workload.async_flags()`), build constants once,
  pin and copy without blocking; a post-processing stage (vocoder, VAE, detokeniser) on a
  joined side stream (`serving.SideStage`). The systems agent's guide: skill `systems-patterns`.
Quality gates still apply end to end; anything that changes outputs beyond the
workload's tolerance is rejected.
