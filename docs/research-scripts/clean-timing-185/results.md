# Clean timing under concurrent load: A/A validation (issue #185)

One fixed candidate (Qwen3-0.6B `decoder_layer_decode`: one cooperative CUDA kernel via
`load_inline`, 8x over the eager decoder layer; its reference is launch-bound, so CPU
contention shows in it first) evaluated by kernel-agent's own evaluator (`run_evaluation`,
the `evaluate_candidate` path) on the RTX 5070 Ti, in blocks of 2 evaluations that alternate
between the conditions (`aa.py`), so the rest of this shared machine's load falls on all of
them alike. Simulated agents: each builds a `load_inline`-like extension (main.cpp + one .cu,
both with `torch/extension.h`; one time in four 3 variants at once), thinks 15-35 s, runs a
3-8 s GPU script (bf16 GEMMs and small launches; half of them "benchmarks"), thinks again.
Every run held the machine's GPU lock (`flock ~/.cache/kernel-agent/gpu.lock`). `loadavg` is
`/proc/loadavg` at the start of each evaluation (in C, L and B with the simulated agents' own
load), `foreign CPUs` the CPUs the rest of the machine (not this experiment) used during it.

## Runs r5-r7 (final code, host load recorded)

| condition | evaluations | speedup mean | sd (CV) | min - max | reference ms / run (CV) | candidate ms / run (CV) | measured twice | dirty twice | not ok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q: N = 1, nothing else (`--agents 1`) | 12 | 8.008x | 0.089 (1.1 %) | 7.90 - 8.15 | 44.1 (1.2 %) | 5.51 (0.6 %) | 0 | 0 | 0 |
| C: N = 3 agents, clean timing (`--agents 3`) | 12 | 8.028x | 0.058 (0.7 %) | 7.93 - 8.11 | 44.1 (0.8 %) | 5.49 (0.4 %) | 6 | 3 | 0 |
| L: N = 3 agents, no clean timing (control) | 8 | 8.128x | 0.406 (5.0 %) | 7.80 - 9.11 | 45.1 (5.5 %) | 5.55 (0.8 %) | 0 | 0 | 0 |
| B: N = 3, scripts outside the lock, clean timing on | 4 | 7.976x | 0.042 (0.5 %) | 7.91 - 8.00 | 43.7 (0.6 %) | 5.48 (0.2 %) | 0 | 0 | 0 |

The spread of the speedup with 3 concurrent agents and clean timing (C, CV 0.7 %) is not
wider than with one agent and nothing else (Q, CV 1.1 %); without clean timing (L) it is
5.0 %, with a +14 % outlier. Of the 12 C evaluations 6 were measured twice. Why the first did
not count: the host was swapping (`/proc/pressure/memory`, 2x), another user's process used
the GPU outside the GPU lock (a process of another agent's test suite with a CUDA context,
once at 87 % SM, 3x), other processes on the timing cores (39 %, 1x). 3 second measurements
were dirty too (swapping, a foreign GPU process) and are kept with `timing_dirty`.

| run | block | condition | speedup | ref ms | new ms | loadavg (1 min) | foreign CPUs | builds / dev runs at start | GPU holds | first measurement not counted |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| r5 | 0 | Q | 7.926 | 43.6 | 5.50 | 2.34 | 1.33 | 0 / 0 | 1 |  |
| r5 | 0 | Q | 7.914 | 43.5 | 5.49 | 2.3 | 1.85 | 0 / 0 | 1 |  |
| r5 | 1 | C | 7.939 | 43.7 | 5.51 | 2.29 | 2.37 | 1 / 0 | 1 |  |
| r5 | 1 | C | 8.025 | 44.4 | 5.53 | 5.2 | 2.76 | 3 / 0 | 2 | the host was out of memory |
| r5 | 2 | L | 7.797 | 43.3 | 5.56 | 4.32 | 2.09 | 0 / 0 | 1 |  |
| r5 | 2 | L | 8.028 | 44.9 | 5.59 | 3.94 | 2.35 | 0 / 1 | 1 |  |
| r5 | 3 | Q | 8.069 | 45.0 | 5.58 | 5.1 | 6.16 | 0 / 0 | 1 |  |
| r5 | 3 | Q | 7.917 | 44.1 | 5.57 | 5.85 | 2.27 | 0 / 0 | 1 |  |
| r5 | 4 | C | 8.061 | 44.5 | 5.52 | 5.98 | 4.86 | 2 / 0 | 2 | other processes used 39 % of the timing cores while it timed |
| r5 | 4 | C | 8.11 | 44.3 | 5.47 | 8.13 | 2.76 | 3 / 0 | 2 | another process used the GPU outside the GPU lock |
| r5 | 5 | L | 9.108 | 51.1 | 5.61 | 4.45 | 0.24 | 1 / 0 | 1 |  |
| r5 | 5 | L | 8.058 | 45.1 | 5.59 | 5.01 | 0.45 | 3 / 0 | 1 |  |
| r6 | 0 | Q | 8.151 | 44.8 | 5.50 | 5.42 | 0.98 | 0 / 0 | 1 |  |
| r6 | 0 | Q | 7.901 | 43.5 | 5.51 | 4.52 | 1.52 | 0 / 0 | 1 |  |
| r6 | 1 | C | 7.989 | 43.8 | 5.48 | 3.16 | 1.4 | 1 / 0 | 1 |  |
| r6 | 1 | C | 8.027 | 44.1 | 5.49 | 5.09 | 1.38 | 3 / 0 | 2 | the host was out of memory |
| r6 | 2 | L | 7.987 | 44.1 | 5.52 | 4.31 | 4.8 | 0 / 1 | 1 |  |
| r6 | 2 | L | 7.947 | 43.8 | 5.51 | 4.94 | 1.31 | 0 / 2 | 1 |  |
| r6 | 3 | Q | 8.04 | 44.2 | 5.50 | 3.63 | 1.75 | 0 / 0 | 1 |  |
| r6 | 3 | Q | 8.006 | 44.1 | 5.50 | 3.98 | 1.31 | 0 / 0 | 1 |  |
| r6 | 4 | C | 8.107 | 44.4 | 5.47 | 3.8 | 0.9 | 2 / 0 | 1 |  |
| r6 | 4 | C | 8.056 | 44.3 | 5.50 | 3.85 | 0.16 | 3 / 0 | 1 |  |
| r6 | 5 | L | 8.091 | 44.4 | 5.49 | 2.82 | 0.07 | 0 / 0 | 1 |  |
| r6 | 5 | L | 8.011 | 44.1 | 5.50 | 2.45 | 0.07 | 0 / 3 | 1 |  |
| r7 | 0 | Q | 8.147 | 44.5 | 5.47 | 4.23 | 0.05 | 0 / 0 | 1 |  |
| r7 | 0 | Q | 8.072 | 44.3 | 5.49 | 3.52 | 0.05 | 0 / 0 | 1 |  |
| r7 | 1 | B | 8.0 | 44.0 | 5.50 | 2.15 | 0.15 | 1 / 0 | 1 |  |
| r7 | 1 | B | 7.997 | 43.8 | 5.48 | 4.26 | 0.23 | 3 / 0 | 1 |  |
| r7 | 2 | C | 7.927 | 43.4 | 5.47 | 3.42 | 0.09 | 0 / 1 | 2 | another process used the GPU outside the GPU lock |
| r7 | 2 | C | 7.997 | 43.8 | 5.48 | 2.32 | 0.1 | 2 / 0 | 1 |  |
| r7 | 3 | Q | 7.982 | 43.7 | 5.47 | 2.38 | 0.05 | 0 / 0 | 1 |  |
| r7 | 3 | Q | 7.965 | 43.8 | 5.49 | 1.99 | 0.07 | 0 / 0 | 1 |  |
| r7 | 4 | B | 7.913 | 43.3 | 5.47 | 1.92 | 0.85 | 2 / 0 | 1 |  |
| r7 | 4 | B | 7.992 | 43.7 | 5.46 | 4.29 | 0.42 | 3 / 0 | 1 |  |
| r7 | 5 | C | 8.041 | 44.0 | 5.47 | 2.84 | 0.06 | 0 / 0 | 2 | another process used the GPU outside the GPU lock |
| r7 | 5 | C | 8.058 | 44.1 | 5.47 | 2.13 | 0.08 | 0 / 0 | 1 |  |

The candidate's extension was built once (35 s, the warm-up evaluation). In C and B every
evaluation's prebuild found it built (0.02-0.06 s in ninja; the prebuild took 5-12 s in all,
off the GPU: torch's import, the 287 MB inputs-only capture and `build()` on fake CUDA
weights) and the evaluator's `compile_s` was 0.1-0.2 s.

## Earlier rounds (not counted above)

* r1 (aborted, no file): builds of 5 template-heavy translation units per agent at once put
  12 `cudafe++` (2 GB each) on the 32 GB host and swapped it: one C evaluation held the GPU
  77 s instead of 20 s, L read 14.0x and 16.2x plus a false `integrity_violation`. Hence
  `MAX_JOBS` is a session's share of the other cores, and a hold during which the host swaps
  is dirty.
* r2, r3 (`results/aa-r2.jsonl`, `results/aa-r3.jsonl`): before the quiet timing phases (r2),
  and with them but partly under other agents' load (r3: loadavg up to 17.7). C read 8.73x
  twice (+8 %): builds pinned to the other cores still slow the launch-bound reference through
  the shared L3 cache, memory bandwidth and all-core clocks (one also on a swapping host).
  Hence a timed subprocess's timing pauses the agents' background work. L read 4.07x once (two
  cases at 0.13 ms with a 440 % spread) and once a false `integrity_violation` (the reference
  3.7x slower beside the candidate than without it).
* r4 (aborted, no file): another agent's synthetic load (8 CPU burners, loadavg 19). N = 1 read
  9.16x and 9.31x (+15 %, nothing flags it); with clean timing both evaluations were dirty
  twice (`cpu_wait_share` 6-22 %) and kept with `timing_dirty`.

Reproduce: `aa.py` (its docstring has the layout and the command), `summarize.py r5 r6 r7`.
