---
name: speculative-decoding
description: Exact data-dependent speedups (prompt-lookup drafts, draft models, early exit) — keeping them exact for greedy decoding, the diverse input set they are timed on, the data_dependent label, report_stats counters. Use when a transform's gain depends on the generated content.
---

# Data-dependent speedups: speculative decoding, early exit

Some exact techniques are as fast as the *content* lets them be: prompt-lookup (n-gram)
drafts are right where the continuation copies the prompt, a draft model where its guesses
match, an early exit where the input is easy. They are welcome. Keep them exact for greedy
decoding (verify every draft with the model's own forward and keep the longest prefix where
draft == argmax, plus the model's next token), and measure them honestly:

* Every passing `evaluate_e2e` also runs the workload's diverse input set
  (`metrics.diverse`; LLM: news, dialogue, code, a recipe, a poem, a question, maths, a CSV
  table, German, French, Spanish and Chinese requests at the end of a `prompt_len` prompt;
  TTS: a few other texts), judged like the main input and timed against the baseline per
  input. Its median, min and max speedup are reported next to the benchmark speedup.
* A candidate whose per-input speedups spread beyond the noise (10 % at least), or whose
  decode steps per token change with the input, is labelled `data_dependent` in the ledger
  (`flags`), `status`, the integration and the report. It is never rejected for it; the
  report shows the diverse-set median next to the benchmark number.
* Report the loop's counters so the report can explain the gain:

  ```python
  # one verification of a K-token draft that accepted n tokens and emitted m (= n + 1)
  workload.report_stats(steps=1, verifies=1, drafted=K, accepted=n, tokens=m)
  # one plain decode step
  workload.report_stats(steps=1, tokens=1)
  ```

  Every timed run starts from zero; `metric_detail.decode_stats` holds the medians plus
  `acceptance_rate`, `tokens_per_verify` and `tokens_per_step`.
* The benchmark prompt is long non-repeating text. A prompt made of one paragraph repeated
  (the LLM workload before #170) makes a greedy model repeat it, and prompt lookup then looks
  40-67x faster than on real text (Qwen3-0.6B: 65x on the repeated paragraph, 11-14x on
  READMEs, code and a licence).
* With `--quality near-lossless` or `relaxed` an LLM is judged teacher forced (KL and top-1 agreement on
  eager's continuations, the likelihood of its own): the gate calls `model(input_ids)` on a
  prompt plus 64 tokens, so keep plain forward calls of the model working when a transform
  replaces the decode loop.

Related: `systems-patterns` (host syncs, serving), `model-transforms.md` of `optimisation-playbook`.

## Examples and sources

* Sources: the `documentation-sources` skill's `sources.md`, sections "Attention and LLM kernels (reference code)".
* Code: `Workload.report_stats`, `kernel_agent.workloads.diverse`.
