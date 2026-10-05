# Research notes: agentic GPU-kernel optimisation (2024 – Oct 2026)

What kernel-agent can learn from related systems and papers. Compiled in
October 2026 from source reading (cloned repositories) and paper pages. Numbers
are the authors' own and unverified unless stated; "(?)" marks details we could
not confirm. The roadmap built from these notes is in [ROADMAP.md](ROADMAP.md).
Which feature came from which source: [REFERENCES.md](REFERENCES.md).

## 1. Closest systems, read at source level

### karpathy/autoresearch (`master@228791f`, Mar 2026)

* No orchestrator code: Claude Code / Codex follows `program.md` (114 lines).
  The agent edits only `train.py`; `prepare.py` (data, metric, `TIME_BUDGET =
  300` s) is read-only; the human edits only `program.md` ("research org code").
* Loop forever: edit → `git commit` → run (output to `run.log`, grep the
  metric) → append to untracked `results.tsv` (`commit, val_bpb, memory_gb,
  status keep|discard|crash, description`) → keep the commit if better, else
  `git reset`. "NEVER STOP … if you run out of ideas, think harder."
* Fixed 5-minute training budget makes experiments comparable (~12/hour).
  Simplicity criterion: tiny gains that add hacky code are not worth keeping;
  deleting code at equal performance is a win.
* `analysis.ipynb` → `progress.png`: x = experiment #, y = val_bpb; discarded
  runs faint grey, kept runs green with black edge annotated with their
  description, running-best step line.
* `agenthub` branch: many agents share improving commits as git bundles and
  post every result (including failures) to a shared board.
* Pitfall seen in its own published log: a pure "lower is better" rule kept a
  random-seed change as an improvement — keep decisions must be noise-aware.

### RightNow-AI/autokernel (`main@7843582`, Mar 2026; paper arXiv 2603.21331)

* `profile.py` (torch.profiler, one forward pass) → `extract.py` (starter
  Triton/CUDA kernel per op type, 9 types by kernel-name substrings) → agent
  loop on a single `kernel.py` (commit → `bench.py` → keep if ≥ 1 % faster,
  else `git reset --hard HEAD~1`) → `verify.py`.
* Orchestrator move-on rules: 5 consecutive reverts, 90 % of peak, 2 h per
  kernel, or 2× speedup; targets ranked by profiled GPU time (Amdahl).
* Five-stage correctness on synthetic inputs: smoke, shape sweep (8–10 sizes ×
  3 dtypes), adversarial numerics, 3-run bitwise determinism, non-power-of-two
  sizes.
* `analysis.py` → `progress.png` (TFLOPS scatter kept/reverted/failed +
  running-max frontier + PyTorch baseline line).
* Weaknesses found in the source: synthetic canonical shapes (the extracted
  model shapes are not used by `bench.py`), profiler double-counts aten rows
  and their kernels, end-to-end check is a single prefill forward and can only
  swap 3 of the 9 op types, `do_bench` mean (comment says median), reference
  and kernel timed sequentially, docs and code disagree on thresholds. The
  paper's headline table benchmarks the hand-written *starter* kernels, and
  reports no measured end-to-end model speedup.

### meta-pytorch/KernelAgent (KernelFalcon, `e064717`)

* Deterministic Python control plane; each step is a raw LLM call.
  Generation: LLM-written test → N seeded workers (T = 0.8) race, first PASS
  wins. Complex problems: a Fuser rewrites the model into fusable subgraphs,
  an extractor emits shape contracts, kernels are generated per subgraph and
  a composer stitches one Triton program.
* Optimisation: beam search over parents × NCU-diagnosed bottlenecks × LLMs ×
  samples (90 workers in the diverse config), Nsight Compute (28 metrics) →
  SOL roofline classification → LLM bottleneck ranking → RAG over a hardware
  knowledge base → reflexion records shared across workers; PTX-fingerprint
  dedup; early stop at ≥ 95 % SOL; per-GPU locks.
* Defences: regex bans on PyTorch compute and frame introspection in kernel
  files; network-blocked sandbox with process-group kill.
* Weakness: correctness is only as good as the LLM-written test; no
  reintegration into a real model (except the hand-wired `oink` ATen
  overrides).

### Dogacel/auto-gpu-kernel (`db02153`)

* `kopt run` supervisor (persistent agent session, one optimisation per
  iteration), hard USD budget, stall detection and recovery prompts, NDJSON
  event log. Won the MLSys 2026 FlashInfer contest DSA track (34.9× on B200,
  per its report).
* Best measurement hygiene of all sources: absolute latencies only, < 5 %
  deltas are noise unless a paired A/B confirms them, harness and candidate
  hashed before and after every benchmark (`kbench/task.py`), results only
  comparable with the same harness revision.
* Subagents: profiler (phase breakdown + memory-floor anchor),
  workload-inspector (input distributions, padding, sparsity — the source of
  its biggest wins), clean-context research agent on plateaus (pathology
  checklist, judge directions by their *ceiling*, "do not try" list).
* `kopt watch`: stdlib HTTP + SSE live dashboard with a best-so-far chart.

## 2. Literature (selected)

| work | org, date | key idea | take-away for kernel-agent |
|---|---|---|---|
| KernelBench | Stanford, Feb 2025 | 250 tasks, fast_p metric | report fast_p-style metrics vs eager and vs compile |
| KernelBench v0.1 blog | Stanford, 2025 | `rand_mix` inputs, hack catalogue, hardware-minimum sanity check | perturbed inputs; faster-than-roofline = hack |
| METR kernel engineering | Feb 2025 | best-of-k tree search; removed stream/memory-reuse hacks | detect streams; diversity adds gains |
| AI CUDA Engineer → robust-kbench | Sakana, 2025 | launch claims withdrawn after a memory-reuse exploit; multi-shape tests, LLM soft verifiers | test unseen shapes; LLM verifiers flag, never gate |
| SOL-ExecBench | NVIDIA, Mar 2026 | roofline-relative scoring; 14.5 % of agent submissions were hacks | checklist for our evaluator (§3) |
| Hacker-Fixer loops | Jun 2026 (?) | hacker agent vs verifier-fixer agent | red-team our evaluator |
| NVIDIA DeepSeek-R1 attention | Feb 2025 | generator + verifier loop | correctness rises with wall-clock budget |
| Stanford "fast kernels" | May 2025 | branch at the level of NL ideas, several implementations per idea | idea ledger; tolerance must not admit precision downgrades |
| GEAK v1→v3 | AMD, 2025–26 | git workspace per agent, profiler/retrieval as MCP tools, cross-session memory | worktree per worker; persistent memory |
| Astra / STARK / CudaForge / PIKE | 2025 | role splits; tree search; curated 3–4 NCU metrics per round (full report was worse) | curated NCU subset; one hypothesis per evaluation |
| K-Search | Berkeley, Feb 2026 | separate strategy from implementation | a buggy attempt must not kill a good idea |
| EvoEngineer, Kernel Scientist, KernelFoundry, KernelBand | 2025–26 | evolutionary archives, MAP-Elites, hierarchical bandits | archive of diverse winners; bandit over targets/strategies |
| AlphaEvolve / AVO | DeepMind 2025 / NVIDIA 2026 | long-horizon evolutionary lineages (AVO beat FA-4 by 10.5 % in 7 days) | continuous mode pays off at the top end |
| AccelOpt, KernelBlaster, KernelSkill, AdaExplore | 2025–26 | slow→fast pairs, persistent lessons, "validity rules" from failures | cross-run kernel library + lessons |
| Kevin-32B, CUDA-L1, AutoTriton/TritonRL, Dr.Kernel, CUDA Agent | 2025–26 (RL) | taxonomy of reward hacks; serial refinement beats parallel sampling at fixed budget | depth over breadth; verify custom kernels actually run; compare against `torch.compile` |
| InferenceBench | 2026 | a plain sweep (11.5×) beat agents (8.1×) | give agents a sweep tool |
| LLM4LLM | EMNLP'26 | phase-aware (prefill/decode) whole-model optimisation with in-model validation | closest system to ours; phase-specific targets |
| KernelOPT | Sep 2026 | fp64-fallback verification cascade, optimise Inductor's leftover kernels | fp64-calibrated tolerance; compile-first path |
| MEP, FlashInfer-Bench | 2025–26 | extract minimal programs / serialisable traces, reintegrate | validates our capture + `apply.py` design |
| Faster IndexTTS-2 | NVIDIA, Jul 2026 | hand-optimised TTS (TensorRT-LLM for the AR part) | **no agent work targets TTS — open ground** |

Survey and index: arXiv 2601.15727 and
github.com/flagos-ai/awesome-LLM-driven-kernel-generation.

## 3. Reward-hacking failure modes vs. our evaluator (as of v0.1.0)

| # | failure mode | seen in | kernel-agent v0.1.0 | roadmap |
|---|---|---|---|---|
| 1 | work on side streams / threads invisible to CUDA events | CUDA-L1, METR, SOL | exposed (`bench.py` times the current stream) | #7 |
| 2 | output cached by `data_ptr` / shape / call count | CUDA-L1, SOL | exposed (one correctness call; timing reuses tensors; timed outputs unchecked) | #6 |
| 3 | stale CUDA-graph replay / skipped work | SOL | exposed (same cause) | #6 |
| 4 | lazy tensor subclasses | CUDA-L1, METR | exposed (`isinstance`) | #6 |
| 5 | monkey-patching timer / comparator / reference / backend flags | METR o3, SOL | exposed (candidate imported into the evaluator process) | #7 |
| 6 | reading the answer key | METR o3 | exposed (`capture.pt` with outputs in the agent cwd) | #5 |
| 7 | tampering with results / baselines | METR o3 | exposed (`bypassPermissions` in dirs holding `results.jsonl`, `history/`, `../baseline.json`) | #5 |
| 8 | memory-reuse no-op (Sakana) | Sakana, METR | low risk (reference outputs come from disk) | #6 |
| 9 | calling / inheriting the reference on the main path | Kevin, TritonRL | partial (identity check only) | #7 |
| 10 | precision downgrade | SOL | partial (bf16/fp16 tolerance + 0.1 % outliers) | #8 |
| 11 | unbounded outliers | AutoKernel | exposed (0.1 % of elements may be wrong by any amount; cosine unused) | #6 |
| 12 | NaN masking | code reading | exposed (candidate NaNs pass when the reference has any −inf/NaN) | #6 |
| 13 | overfitting to captured shapes | robust-kbench, LLM4LLM | exposed (≤ 3 cases; one KV length; e2e on the timed input) | #8 |
| 14 | transform memoising `workload.run` | METR | prompt rule only | #8 |
| 15 | races / latent out-of-bounds writes | AutoKernel, LLM4LLM | exposed (one run per case, no sanitizer) | #8 |
| 16 | weak eager baseline | CUDA Agent, KernelOPT | partial (`compile_baseline` off) | #12 |
| 17 | acceptance noise | METR, SOL | partial (5 iterations, 1 % threshold, no CI) | #11 |
| 18 | hot-L2 timing | LLM4LLM, SOL | partial (`l2_flush` exists, never enabled by tools) | #6/#9 |

What we already do well and must keep: real captured inputs (prefill + decode),
checking in-place side effects (KV-cache appends), rejecting `build()` that
returns the reference, subprocess isolation under a GPU lock, interleaved
reference/candidate timing, end-to-end acceptance by measured latency, and
host-overhead-aware backend choice.

## 4. Feature comparison

| | autoresearch | AutoKernel | KernelAgent | auto-gpu-kernel | kernel-agent v0.1.0 |
|---|---|---|---|---|---|
| unit optimised | training recipe | op-type kernel | KernelBench problem / subgraph | one kernel or repo | `nn.Module` class + model transforms |
| test inputs | fixed eval set | synthetic sweep | LLM-written test | trace set / generated harness | **real captured I/O** |
| continuous loop | yes | yes | fixed rounds | yes (supervisor) | no (one-shot) |
| keep/revert | git | git | beam keep top-N | git + log | best-so-far snapshot |
| charts | `progress.png` | `progress.png` | none (Gradio logs) | live SSE chart | none |
| hardware feedback | — | % of peak | NCU + SOL roofline | memory-floor anchor | torch.profiler tables |
| parallel search | agenthub | none | seeded race + beam | none | one agent per target |
| cross-run memory | git | git + TSV | program DB + RAG | LESSONS.md | none |
| end-to-end composition | the metric itself | 3 op types, one forward | LLM-composed program | n/a | **greedy measured integration + `apply.py`** |
| cost control | none | per-kernel time | iteration counts | hard USD budget | per-agent USD/turns |
| modalities | 1 LM | LMs, BERT | KernelBench | kernels | LLM, STT, TTS, diffusion, harness |

## 5. VoxCPM2 findings (motivating model)

Measured on an RTX 5070 Ti (16 GB, sm_120), torch 2.14.1+cu130, voxcpm 2.0.3,
60 patches (9.6 s of 48 kHz audio), 10 CFM steps, CFG 2.0, bf16:

* Latency: **5.46 s eager**, **3.80 s** with VoxCPM's own `optimize()`
  (`torch.compile(mode="reduce-overhead")`).
* Launch-bound: GPU busy 47 %, 507k kernel launches per run (avg 5 µs).
* The decode path runs through `forward_step` methods that `nn.Module` hooks
  never see (2160 decoder-layer and 2160 attention calls per run).
* Top kernel `fmha_cutlassF` (1151 ms, 45 % of GPU time) is the decode
  attention in `MiniCPMAttention.forward_step`, which attends over the whole
  8192-slot static KV cache with a mask while ~70 positions are valid.
* The output is chaotic under numerically-correct changes: the bundled,
  verified Triton RMSNorm drops the spectral cosine to 0.69 (gate 0.97);
  per-step latents agree to 0.9999 at step 0 and diverge after ~18 steps.
  Free-running comparison therefore rejects every valid kernel; teacher-forced
  validation is required (#2).
