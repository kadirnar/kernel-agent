---
description: Optimise a Hugging Face model at kernel level (CUDA / CuTe DSL / Triton / TileLang)
argument-hint: <hf-url> [extra kernel-agent options]
---

Optimise the model `$ARGUMENTS` with kernel-agent.

kernel-agent's skills (`optimisation-playbook`, `profiling-and-roofline`, the backend and
precision skills, ...) and subagents (`planner`, `kernel-engineer`, `systems-engineer`,
`native-engineer`, `researcher`, `reviewer`, `doc-lookup`, `profile-analyst`, ...) are
installed in this project: load a skill whenever a step touches its topic.

1. Run `kernel-agent doctor` and report which backends are available and what the GPU is.
2. Run `kernel-agent analyze $ARGUMENTS`, then hand the printed run directory to the
   `planner` subagent: where the time goes (launch / memory / compute bound, with the
   ceilings table's numbers) and up to 4 targets with an approach, backends and precision
   each, plus model-level transforms.
3. Ask me which targets to run, then either:
   * run the autonomous pipeline: `kernel-agent resume <run_dir>` (it continues with plan →
     capture → kernels → transforms → integrate → report), or `kernel-agent improve
     <run_dir>` for the continuous loop, or
   * work interactively: capture the targets (`kernel-agent resume <run_dir> --until
     capture`), then launch a `kernel-engineer` subagent per target (in parallel when there
     are several) with the run directory and target id, a `systems-engineer` for the
     transforms, and the `reviewer` on a candidate before it is kept.
4. Finish with `kernel-agent report <run_dir>` and summarise the speedups and the next
   ideas worth testing.
