---
name: profiling-and-roofline
description: Reading kernel-agent's measurements — the profile summary (timeline, phase split, host syncs, ceilings and floors per precision), the evaluator's roofline per case (sol_ms, pct_of_sol, bound), profile=true tables, Nsight Compute. Use to rank targets, diagnose a slow kernel or judge a result against its bound.
---

# Profiling and roofline: reading kernel-agent's numbers

Every number below is measured on the run's GPU (peaks, launch floor, instruction rates in
the toolchain block). The details and their evidence are in the repository README ("What
faster means", "Speed of light", "Kernel feedback", "Ceilings for the planner", "Workload
statistics", "Host synchronisation in the profile"). What to do about each regime: the
`optimisation-playbook` skill.

## Before the plan: `profile/summary.md`

* **Kernel view** (`torch.profiler`, two profiled runs at full clocks): kernel times,
  launches and the GPU busy share behind "launch/CPU bound" or "GPU bound". It must agree
  with the unprofiled run (within 105 %) and between the two runs (20 %), else the summary
  marks it unreliable and draws no conclusion from it.
* **Timeline**: busy time per stream (the union over streams, so overlap does not add up),
  per stage (every call of the workload's roots and their children as a `ka::<qualname>`
  range: calls, kernels, GPU and busy ms, host lead, SM fill), idle gaps binned and
  attributed to a cause (*host sync*, *pageable copy*, *host late*, *graph launch*,
  *launch latency*), and the critical path between stages (work off it could overlap on
  another stream).
* **Phase split**: classes called in both phases (`decode`: a `*step*` entrypoint or a
  `[batch, 1, ...]` first input; else `prefill`). Split such a class into one target per
  phase when both phases are large and their bottlenecks differ.
* **Host synchronisation (transform opportunities)**: every call that makes the host wait
  (`.item()`, `.cpu()`, a blocking or pageable copy, a tensor built on the host per call,
  a data-dependent output size) by call site, with calls per run, host time and the fix
  (the `systems-patterns` skill). CUDA-graph replays make no Python calls and are not
  listed.
* **Ceilings** (`profile/ceilings.md`, `ceilings.json`): one row per class × instance
  group × phase. *M* = FLOPs / (2 × weight elements): the rows per weight read; a bf16
  GEMM turns compute bound near M = peak / bandwidth (≈ 130 on an RTX 5070 Ti). *now*,
  TFLOP and weight GB per run, the bound, and **floors** = max(FLOPs / peak, bytes / DRAM
  bandwidth, calls × launch floor) per precision: *exact*, *FP8 w*, *W8A8*, *MXFP8*,
  *FP4 w*, *W4A4* (only the columns the run's precisions allow are shown; `?` = peak not
  measured). *saves* = now − exact floor; rows rank by it. *FP8 MMA* says which FP8
  instruction a W8A8 kernel needs (*SF*: the block-scaled one; *any*), *KV GB* the KV-cache
  bytes of decode rows. The end-to-end line puts every class at its floor. Approximate:
  attention scores, element-wise math and caches held as attributes are not counted.
* **Workload profile** of a target (`workload_profile.md`): every call of the target's
  instances during the capture run (signatures and shares, masks, strides, flags, integer
  ranges, KV-cache valid lengths): what may be specialised behind a run-time check.

Rank targets by ceiling × share (*saves*), not by share alone, and name each target's bound
with its number.

## After a kernel evaluation: the roofline of its recipe

Every timed `evaluate_candidate` result reports per case `flops`, `min_bytes` (every byte
range the reference reads once, outputs and changed state once), `sol_ms` = max(Σ flops /
peak, min_bytes / bandwidth), `pct_of_sol` = 100 × sol_ms / new_ms and `bound`
(`compute`, `memory`, or `launch` when `sol_ms` is below the launch floor), plus the
weighted `pct_of_sol`, `sol_ms_weighted`, the dominant `bound` and `launch_floor_ms`.

* `l2_resident`: the case's bytes fit in L2 and are compared with the L2 bandwidth (the
  benchmark reuses warm inputs).
* Reduced precisions count their weights at the stored bits (`fp8_weights`: one byte plus a
  scale per channel; `fp8_w8a8` FLOPs on its weights at the FP8 peak).
* `suspicious_faster_than_sol`: faster than the hardware allows; check that the kernel does
  all the work. `sol_unreliable`: the reference itself beats the estimate (masked work
  outside SDPA, strided views counted whole).
* At ≥ 90 % the recipe is at its bound: the next gain needs another recipe (fusion with
  neighbouring calls, a precision the run allows, another algorithm or layout), never "done".

Timing itself: reference and candidate alternate in rounds with CUDA events at full clocks
(a DRAM-bandwidth probe gates each round), medians weighted by calls per run; inputs rotate
between copies and mutable ones are copied outside the timed region.

## Deeper: `profile=true` and Nsight Compute

* `profile=true`: per-kernel GPU time tables for candidate and reference, and
  `compiler_stats` (registers, spills, shared memory of Triton, NVRTC and `load_inline`
  kernels) with a warning per spilling kernel.
* `profile="ncu"`: Nsight Compute on the most-called case, 24 curated metrics (SM and
  memory throughput, DRAM bytes, L1 / L2 hit rates, achieved and theoretical occupancy,
  tensor-pipe activity, registers, shared memory, grid, block, waves, top warp stalls);
  each kernel classed *memory*, *compute* or *under-utilised* (both below 60 % of peak).
  ncu runs per launch with flushed caches at base clocks: compare kernels with each other,
  not with the evaluator's timings. `ncu.status: unavailable` says why and how to fix it.

## Examples and sources

* Code: `kernel_agent.kernels.roofline`, `kernel_agent.kernels.ncu`,
  `kernel_agent.profiling.ceilings` (`python -m kernel_agent.profiling.ceilings
  profile.json --baseline-ms <ms>` prints the table of any profile),
  `kernel_agent.profiling.timeline`, `kernel_agent.profiling.host_sync`.
* Sources: the `documentation-sources` skill's `sources.md`, sections "CUDA C++ and PTX"
  (best practices, tuning guides).
