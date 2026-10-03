---
description: Optimise a Hugging Face model at kernel level (CUDA / CuTe DSL / Triton / TileLang)
argument-hint: <hf-url> [extra kernel-agent options]
---

Optimise the model `$ARGUMENTS` with kernel-agent.

1. Run `kernel-agent doctor` and report which backends are available.
2. Run `kernel-agent analyze $ARGUMENTS` and read `profile/summary.md` in the
   printed run directory. Explain where the time goes (launch-bound vs
   memory-bound vs compute-bound) and propose up to 4 target module classes,
   with an approach and backends for each.
3. Ask me which targets to run, then either:
   * run the autonomous pipeline: `kernel-agent resume <run_dir>` (it continues
     with plan → capture → kernels → transforms → integrate → report), or
   * work interactively: for each target, launch the `kernel-engineer` subagent
     (in parallel when there are several) with the run directory and target id.
4. Finish with `kernel-agent report <run_dir>` and summarise the speedups.
