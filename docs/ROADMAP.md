# Roadmap

Goal: turn kernel-agent into a continuous, autoresearch-style optimisation
loop with an evaluator that cannot be gamed, and use it to make
`openbmb/VoxCPM2` fast. Background and sources: [RESEARCH.md](RESEARCH.md).
Every item is a GitHub issue; this file is the overview.

## Principles

1. **Trust before search.** More search on a weak evaluator only finds more
   hacks (CUDA-L1, SOL-ExecBench, METR). Evaluator integrity (M3 P0 items)
   comes before parallel search and long-running loops.
2. **Real inputs, measured end to end.** Keep what already sets kernel-agent
   apart: captured prefill/decode inputs, side-effect checks, measured greedy
   integration. Report speedups against eager **and** a strong compiled
   baseline.
3. **Everything is an experiment.** One hypothesis per evaluation, a ledger
   row per experiment, charts generated from the ledger, budgets on every loop.

## M1 — VoxCPM-ready

| issue | item | priority |
|---|---|---|
| [#1](https://github.com/kadirnar/kernel-agent/issues/1) | profile, capture and evaluate non-`forward` entrypoints (`forward_step`) | P0 |
| [#2](https://github.com/kadirnar/kernel-agent/issues/2) | teacher-forced e2e validation for chaotic AR workloads + built-in VoxCPM2 workload | P0 |
| [#12](https://github.com/kadirnar/kernel-agent/issues/12) | strong baselines (workload-native / `torch.compile`), report vs eager and compiled | P0 |
| [#24](https://github.com/kadirnar/kernel-agent/issues/24) | tracking: optimise VoxCPM2 end to end | P0 |

## M2 — Autoresearch loop

| issue | item | priority |
|---|---|---|
| [#3](https://github.com/kadirnar/kernel-agent/issues/3) | experiment ledger (`results.tsv`), progress charts, `status`, HTML dashboard | P0 |
| [#4](https://github.com/kadirnar/kernel-agent/issues/4) | run budgets (hours, USD, per-agent, per-eval) and advice signals | P1 |
| [#13](https://github.com/kadirnar/kernel-agent/issues/13) | continuous `kernel-agent improve`: Amdahl/UCB scheduler, slices, stop rules, rounds | P1 |
| [#14](https://github.com/kadirnar/kernel-agent/issues/14) | `program.md`: human-editable instructions per role | P2 |
| [#15](https://github.com/kadirnar/kernel-agent/issues/15) | idea ledger + clean-context research agent on plateau | P2 |
| [#16](https://github.com/kadirnar/kernel-agent/issues/16) | live dashboard (`kernel-agent watch`) | P2 |

## M3 — Robust evaluation

| issue | item | priority |
|---|---|---|
| [#5](https://github.com/kadirnar/kernel-agent/issues/5) | tamper-proof evaluator state | P0 |
| [#6](https://github.com/kadirnar/kernel-agent/issues/6) | strict compare, timed-output check, perturbed re-verification | P0 |
| [#7](https://github.com/kadirnar/kernel-agent/issues/7) | side streams/threads, monkey-patch integrity, reference-fallback detection | P0 |
| [#8](https://github.com/kadirnar/kernel-agent/issues/8) | extra capture settings, KV buckets, held-out e2e, fp64 tolerance | P1 |
| [#9](https://github.com/kadirnar/kernel-agent/issues/9) | roofline / speed-of-light with measured peaks | P1 |
| [#11](https://github.com/kadirnar/kernel-agent/issues/11) | paired in-process A/B integration with bootstrap CI | P1 |
| [#10](https://github.com/kadirnar/kernel-agent/issues/10) | Nsight Compute curated metrics + compiler stats | P2 |

## M4 — Search & memory

| issue | item | priority |
|---|---|---|
| [#17](https://github.com/kadirnar/kernel-agent/issues/17) | parallel isolated workers per hot target, dedup, quick eval tier | P2 |
| [#18](https://github.com/kadirnar/kernel-agent/issues/18) | cross-run kernel library and distilled lessons | P2 |
| [#19](https://github.com/kadirnar/kernel-agent/issues/19) | sweep / autotune tool | P2 |
| [#20](https://github.com/kadirnar/kernel-agent/issues/20) | workload statistics at capture, phase-specific targets, TTS playbook | P2 |
| [#21](https://github.com/kadirnar/kernel-agent/issues/21) | red-team agent and KernelBench regression suite | P2 |
| [#22](https://github.com/kadirnar/kernel-agent/issues/22) | region targets across module boundaries | P2 |
| [#23](https://github.com/kadirnar/kernel-agent/issues/23) | multi-GPU lock pool | P2 |

## Order of work

```
wave 1 (parallel):  #1 forward_step   #2 teacher forcing + VoxCPM   #3 ledger/charts   #4 budgets
wave 2 (parallel):  #5 truth files    #6 strict compare             #12 strong baseline #9 roofline
wave 3:             #24 VoxCPM run    #7 anti-gaming   #8 coverage   #13 improve loop    #11 A/B
wave 4+:            #14 #15 #16 #17 #18 #19 #20 #21 #10 #22 #23
```

Each wave is implemented by parallel agents in separate git worktrees, one PR
per issue, reviewed and merged after lint, type checks and tests pass.
