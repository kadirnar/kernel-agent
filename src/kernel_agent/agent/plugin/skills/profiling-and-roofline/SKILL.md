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
* **Fusion candidates (measured)** (`profile/fusions.md`, `fusions.json`): chains of
  memory-bound ops (element-wise, norms, reductions, casts) found from the tensor storages
  of every op of one run, with the GEMM / convolution / attention that could take them as
  an epilogue or prologue (*fuse*), across module boundaries (*crosses*), never across a
  host sync; layers deduplicated (*calls*). *saves* = the DRAM part of the intermediates'
  round trips / DRAM bandwidth + launches saved × the launch floor (eager) or ~0.9 us per
  CUDA-graph boundary. LRU-ish L2: of an intermediate, what exceeds L2 − the bytes other
  ops touch between its write and its last read counts (0 when it fits with that traffic;
  `fusions.md` says per intermediate: in L2, evicted by N MB of traffic, larger than the
  L2). The counted rows share no op and add up; a row marked ↳ is an alternative that
  shares a GEMM with the row above it (a prologue and an epilogue of one projection):
  plan one or the other. A row is a region target: `parent_class` = its parent class,
  `region` = its ops, `fusion` = its id.
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
  benchmark reuses warm inputs; never with `l2: cold`).
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

Timing context (`context`, `l2`, `context_reason` in every timed result): the module is
timed as it runs in the model, from the newest profile. `graph` when its timeline stage is
mostly launched by CUDA-graph replays (12 calls captured in one graph: host launch time does
not count, as inside a graphed stage), else `eager` (host time counts); `cold` L2 when the
rest of the model touches more than the L2 between two of its calls (weights stream from
DRAM, and `l2_resident` no longer applies), else `warm`. A winner is timed in the other
context too: `speedup_by_context` and per case `timing`. `graph: unavailable (<why>)` means
the candidate cannot be captured in a CUDA graph (a host sync, a CPU tensor, a tensor of its
inputs replaced instead of written in place) or computes something else when replayed: it
would break a graphed stage, whatever its eager speedup. Measured: an FP8 GEMM of two Triton
launches at M = 352 was 0.58x eagerly and 2.01x graph-timed with a cold L2 (RTX 5070 Ti),
so compare candidates in the context the result names.

## Deeper: `profile=true` and Nsight Compute

* `profile=true`: per-kernel GPU time tables for candidate and reference, and
  `compiler_stats` (registers, spills, shared memory of Triton, NVRTC and `load_inline`
  kernels) with a warning per spilling kernel.
* The SASS census (`sass`, every `profile` evaluation, no GPU work): what each kernel was
  compiled to. `tensor`: its tensor-core opcodes whole (`HMMA` / `IMMA` / `QMMA` / `OMMA`:
  `mma.sync`; `QMMA.SF`: block-scaled; `HGMMA` / `QGMMA` / `IGMMA`: wgmma; `UTC*MMA`:
  tcgen05); `categories`: global loads and stores, `cp_async` (`LDGSTS`), `tma`, `local`
  (`LDL` / `STL`: spills or a run-time indexed array), shared memory, tensor memory,
  barriers, shuffles, atomics, fp32 / fp16 math; `global_load_bits`: load widths. Counts
  are static (instructions in the binary). `F2FP...E4M3.UNPACK` feeding `HMMA` is e4m3
  `mma.sync` emulated through fp16 (sm_90, sm_100). Compare the tensor-core opcode with
  this GPU's full-rate path (the toolchain block's measured rates, `gpu-architectures`):
  on sm_120 `tl.dot` on e4m3 issues `QMMA.16832.F32` at half the rate of the
  `QMMA.SF` that `tl.dot_scaled` issues. `missing`: kernels that ran without SASS here
  (library kernels; an NVRTC kernel whose `ObjectCode` was freed: keep the result of
  `Program(...).compile("cubin")` in a module global to include it).
* `profile="ncu"`: Nsight Compute on the most-called case, 24 curated metrics (SM and
  memory throughput, DRAM bytes, L1 / L2 hit rates, achieved and theoretical occupancy,
  tensor-pipe activity, registers, shared memory, grid, block, waves, top warp stalls);
  each kernel classed *memory*, *compute* or *under-utilised* (both below 60 % of peak).
  ncu runs per launch with flushed caches at base clocks: compare kernels with each other,
  not with the evaluator's timings. `ncu.status: unavailable` says why and how to fix it.
  Then `rules`: Nsight's 3 rules with the largest estimated speedup (global: share of the
  kernel's time; local: a unit's efficiency), `lines`: the 5 source lines (SASS
  instructions without `-lineinfo`) with most warp-stall samples, their share and
  dominant stall (`stall_long_sb`: waiting on global memory; `stall_math`: a pipe is
  saturated; `stall_barrier`; ...), `flagged`: uncoalesced or bank-conflicting lines.
* `directives` (at most 5, in the summary): what the documented rules of
  `kernel_agent.kernels.directives` conclude from all of the above, each with its numbers
  (spills, emulated FP8, an instruction below the full rate, missing tensor cores, an
  idle tensor pipe, the hottest stall line, uncoalesced lines, narrow loads, Nsight's top rule, register-staged
  MMA operands). A directive is a hypothesis: test it, or say which number contradicts it.
* In a kernel-agent session both write the full tables to `profiles/<snapshot>.json` in
  your working directory; the result keeps `profile`: the top kernels of candidate and
  reference with their share of GPU time, the spill warnings, the top ncu kernels' bounds,
  one census line per top kernel, the directives and the file's path. Read the file for a
  detail (the census per kernel, the directives' evidence), or hand the path and your
  question to the `profile-analyst` helper.

## Examples and sources

* Code: `kernel_agent.kernels.roofline`, `kernel_agent.kernels.ncu`,
  `kernel_agent.kernels.sass`, `kernel_agent.kernels.directives`,
  `kernel_agent.profiling.ceilings` (`python -m kernel_agent.profiling.ceilings
  profile.json --baseline-ms <ms>` prints the table of any profile),
  `kernel_agent.profiling.timeline`, `kernel_agent.profiling.host_sync`,
  `kernel_agent.profiling.fusion` (`python -m kernel_agent.profiling.fusion profile.json
  --window-ms <ms>` ranks the chains of a profile again).
* Sources: the `documentation-sources` skill's `sources.md`, sections "CUDA C++ and PTX"
  (best practices, tuning guides).
