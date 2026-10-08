# Issue #174, our own data: where the wall time goes today and what concurrent agents would buy on one GPU

Measured on four finished runs (RTX 5070 Ti 16 GB, 12 CPU threads, Claude Opus 5.5 over the subscription login).
Scripts and raw extracts: `scratchpad/174/data/` (see "Reproduce" at the end). Nothing in the repo or in any
run directory was modified; the active Qwen3 run `runs/Qwen--Qwen3-0.6B/20261008-063823` was not read.

## Key numbers

* **Data:** 107 agent sessions (19.4 session-hours), 156 agent evaluations, 21 integration windows (8.2 h, 292 A/B measurements), 28.3 h of active run wall time, $177 API-equivalent.
* **Run wall time:** agent sessions take 68%, integration 29% (16–52% per run), analyze + capture 1.5%, other orchestrator work 1%. The runs produce **5.5 agent evaluations per active hour** (4.8–7.0 per run).
* **Inside a session:** 46% of the time a model call is in flight (about half of the output tokens are thinking), 11% is a locked GPU evaluation, 40% is the agent's own Bash runs outside the lock (compiling, correctness checks, microbenchmarks), and 2% is idle. For kernel sessions the split is 45% model / 6% locked GPU / 46% own runs.
* **Compilation is the largest single activity in kernel sessions:** nvcc / load_inline builds (from `.ninja_log`) took 3.3 of 10.9 kernel session-hours (31%), which is 64% of their outside-lock run time. The evaluator itself rarely compiles (median compile_s 0.5 s) because the agents build first.
* **The GPU is mostly idle during sessions:** the lock is held 11% of session time. Even counting every unlocked Python run as GPU time, the GPU is used at most 53% of the time; after removing the measured compile time, at most about 34%.
* **Nobody waits for the lock:** 0 of 156 evaluations waited and no lock intervals overlapped, because only one session runs at a time and integration runs between slices. Agents do wait on their own work: 53 polling loops took 0.83 h, and 3 sessions were held open 10 min each by a background benchmark.
* **Cadence:** sessions are short (median 12 min, 27 API calls, ~12 s per call). The first evaluation comes after a median 5.2 min (kernel), 4.4 min (systems) or 8.6 min (native), and later ones every 2.2–5.9 min. 20 of 86 evaluating-arm sessions ended with no evaluation.
* **GPU job sizes:** a kernel evaluation takes a median 8.7 s (p90 61 s), a sweep 33 s (max 312 s), an e2e evaluation 71 s (max 756 s). One integration A/B measurement takes a median 84 s (p90 237 s, max 791 s; 214 s median on the throughput run). Integration costs 100–350 GPU-seconds per agent evaluation.
* **Simulation (pooled):** without integration, evaluations per hour grow almost linearly with k. But if the agents' unlocked runs stay as they are, 40–88% of the timed GPU time would overlap another agent's run, so timing would no longer be clean. With clean timing and today's integration cost, the GPU saturates at k = 3–4 (11.6 evals/h at k = 3, 2.1x today's 5.5). With integration made 4x cheaper and a reader/writer GPU lock: **k = 2: 13.5/h, k = 3: 17.7/h, k = 4: 20.7/h** (1.7x, 2.3x and 2.7x the simulated k = 1; 2.5x, 3.2x and 3.8x today's measured rate). Agents are then blocked 18 / 28 / 37% of their time. k = 6 saturates the GPU.
* **Cost and limits:** about $9.3 per agent-hour and $1.14 per evaluation (API-equivalent), with no API errors or rejected requests in any session. One session uses about 10 points of the 5-hour window and 2.3 points of the 7-day window per hour, and the 7-day window already reached 0.86 (`allowed_warning`) with one agent at a time. k = 2 fills the 5-hour window. k = 3–4 empties it in 2.4–3.2 h and empties a fresh week in 11–14 h. **On a subscription, the usage budget limits k more than the GPU does.**

## 1. Data and method

| key | run | kind | sessions | notes |
|---|---|---|---:|---|
| V-lat-r1 | `openbmb--VoxCPM2/20261005-042829` | latency, rounds 1–2 (older) | 20 | logs `voxcpm2-improve*.log`; two manual re-integrations (`voxcpm2-reintegrate*.log`, 96 min) are outside events.jsonl and not counted |
| V-lat | `openbmb--VoxCPM2/20261005-192504` | latency, incl. Oct 7 round and the wave-5 native continuation (Oct 7 23:28 – Oct 8 02:18) | 42 | logs `voxcpm2-round3-latency*.log`, `wave5-voxcpm2-latency.log` |
| V-thr | `openbmb--VoxCPM2/20261006-004718` | throughput | 33 | log `voxcpm2-throughput.log` (later invocations have no main log) |
| Q-lat | `Qwen--Qwen3-0.6B/20261008-021850` | latency, wave 5 | 12 | log `wave5-qwen3.log` |

The `-retest`, `-retest2` and `-backup-before-wave5` directories are copies and were left out.

**Timeline reconstruction.** Session windows come from the `agent_start` / `agent_done` events in `events.jsonl`. Session ids come from the init messages in `logs/agent-*.jsonl`, matched by order.
The main logs only time-stamp the start of each tool call, so I used the Claude Code transcripts instead (`~/.claude/projects/*/<session_id>.jsonl`, found for 107 of 107 sessions). They time-stamp every assistant block and every tool result, which gives:
* **model** intervals: from the last user or tool-result entry to the last block of each API call. Their sum matches `duration_api_ms` in the ResultMessage: 8.76 h against 8.80 h over the 103 sessions that have a ResultMessage; the per-session ratio is 0.97–1.00.
* **tool** intervals: from each tool_use to its tool_result.
* **background-job** intervals: from the tool_use to the `<task-notification>` enqueue, or to the end of the session.

Evaluation tool calls last exactly `results.tsv` eval_s + ~0.6 s (154 calls matched). Integration windows run from `reintegrate` to `integrated`; an invocation that stopped mid-integration is closed at its last evaluation.

**Bash classes.** A Bash call counts as **dev GPU** if it:
* runs Python with torch, triton, tilelang or load_inline,
* runs a `.py` script or `-m` module, or a profiler (ncu, nsys, compute-sanitizer), or a built binary,
* polls a background job or the GPU, or
* runs ≥ 15 s and is not an obvious CPU tool.

All other Bash calls count as CPU. This is an upper bound on GPU use. To measure compilation, I read every `.ninja_log` under `~/.cache/torch_extensions` and `~/.cache/kernel-agent` (1650 build steps). A step runs from `mtime` to `mtime + duration`: I checked this against tool-call times. Those build steps were then intersected with the dev intervals.

## 2. Today: where a run's wall time goes (T1)

GPU-lock busy = evaluation calls inside sessions + integration windows + analyze/capture. Dev GPU = the agents' own Python / benchmark / profiler runs outside the lock (upper bound, compiles included).

| run | active h | sessions | agent sessions h | integration h | analyze + capture h | orchestrator other h | agent evals | integration measurements | agent evals / active h | GPU-lock busy | dev GPU (outside lock) | GPU idle | $ (API-equiv.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V-lat-r1 | 4.7 | 20 | 3.6 (77%) | 0.9 (20%) | 0.08 (2%) | 0.07 (2%) | 33 | 109 | 7.0 | 39% | 23% | 38% | 33 |
| V-lat | 11.3 | 42 | 9.3 (83%) | 1.8 (16%) | 0.11 (1%) | 0.09 (1%) | 54 | 62 | 4.8 | 20% | 41% | 39% | 81 |
| V-thr | 9.6 | 33 | 4.4 (45%) | 5.0 (52%) | 0.18 (2%) | 0.12 (1%) | 51 | 79 | 5.3 | 62% | 15% | 24% | 43 |
| Q-lat | 2.7 | 12 | 2.1 (78%) | 0.6 (20%) | 0.05 (2%) | 0.00 (0%) | 18 | 42 | 6.6 | 30% | 32% | 37% | 20 |
| **all** | 28.3 | 107 | 19.4 (68%) | 8.2 (29%) | 0.42 (1%) | 0.29 (1%) | 156 | 292 | 5.5 | 38% | 28% | 33% | 177 |

Integration is already the main locked-GPU consumer: 8.2 h of A/B work against 2.2 h of agent evaluations. On the throughput run it takes half the wall time, because one A/B measurement takes 214 s there.

## 3. Agent sessions by role (T2)

Each session's wall time is split exclusively, in priority order: GPU lock > model > dev GPU > CPU tools > idle. "GPU any" = lock ∪ dev GPU (dev counted even while the model thinks).

| role | sessions | hours | min/session | model | GPU lock | dev GPU | CPU tools | idle | GPU any | evals | evals / session-h | median min to 1st eval | sessions w/o eval | $ / session | $ / eval | turns / session |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| planner | 9 | 0.4 | 2.7 | 97% | 0% | 0% | 1% | 1% | 0% | 0 | 0.0 | - | 9 | 0.78 | - | 9 |
| dossier | 4 | 0.0 | 0.6 | 80% | 0% | 0% | 17% | 3% | 0% | 0 | 0.0 | - | 4 | 0.19 | - | 6 |
| research | 1 | 0.1 | 3.9 | 99% | 0% | 0% | 0% | 1% | 0% | 0 | 0.0 | - | 1 | 1.17 | - | 19 |
| kernel | 54 | 10.9 | 12.1 | 45% | 6% | 46% | 0% | 3% | 54% | 83 | 7.6 | 5.2 | 11 | 1.77 | 1.15 | 22 |
| systems | 25 | 5.2 | 12.5 | 39% | 26% | 34% | 1% | 0% | 60% | 59 | 11.3 | 4.4 | 6 | 1.68 | 0.71 | 26 |
| native | 7 | 2.7 | 23.1 | 53% | 6% | 41% | 0% | 0% | 47% | 14 | 5.2 | 8.6 | 3 | 4.04 | 2.02 | 47 |
| librarian | 7 | 0.0 | 0.4 | 95% | 0% | 0% | 0% | 5% | 0% | 0 | 0.0 | - | 7 | 0.31 | - | 2 |
| **all** | 107 | 19.4 | 10.9 | 46% | 11% | 40% | 0% | 2% | 53% | 156 | 8.0 | 5.2 | 41 | 1.65 | 1.14 | 22 |

* **Thinking and writing.** Model time runs at about 100 output tokens/s. Thinking tokens are 49% of output tokens (48% kernel, 52% systems, 51% native, 58% planner), so model time splits roughly in half between thinking and writing. A median API call takes 12.4 s.
* **Compiling.** These are ninja build steps inside dev intervals:

  | role | compile time | share of dev time |
  |---|---:|---:|
  | kernel | 3.33 h | 64% (31% of kernel wall time) |
  | native | 0.30 h | 27% |
  | systems | 0.03 h | 2% (their runs load models and run torch.compile / inductor, which ninja does not log) |

  Overall, at least 46% of dev time is nvcc. Only 0.1 h of builds ran inside the lock.
* **GPU actually computing:** at most (2.2 h lock + 8.04 h dev − 3.66 h compile) / 19.4 h ≈ 34% of session time. The GPU is idle at least two thirds of the time an agent works.
* **Planner, dossier, librarian and research roles** are short, use no GPU and take 3% of session time. They could run beside the arms; for example, the dossier for the next target could be written while the current arm runs.
* `--seeds-per-target` / `--parallel` were 1 in all four runs (the `worker` column of results.tsv is empty).

Per run, evaluating arms only (T2b):

| run | role | sessions | hours | model | GPU lock | dev GPU | GPU idle | evals | evals / h | $ / eval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V-lat-r1 | kernel | 9 | 1.27 | 55% | 14% | 32% | 54% | 12 | 9.4 | 1.13 |
| V-lat-r1 | systems | 7 | 2.16 | 37% | 30% | 31% | 39% | 21 | 9.7 | 0.76 |
| V-lat | kernel | 25 | 6.93 | 39% | 2% | 56% | 42% | 40 | 5.8 | 1.34 |
| V-lat | systems | 5 | 0.26 | 50% | 28% | 20% | 53% | 4 | 15.1 | 0.88 |
| V-lat | native | 4 | 1.96 | 54% | 8% | 39% | 54% | 10 | 5.1 | 2.06 |
| V-thr | kernel | 19 | 2.31 | 54% | 11% | 34% | 55% | 30 | 13.0 | 0.85 |
| V-thr | systems | 8 | 1.86 | 38% | 28% | 34% | 38% | 21 | 11.3 | 0.69 |
| Q-lat | kernel | 1 | 0.40 | 44% | 22% | 35% | 43% | 1 | 2.5 | 3.24 |
| Q-lat | systems | 5 | 0.94 | 45% | 12% | 43% | 45% | 13 | 13.8 | 0.62 |
| Q-lat | native | 3 | 0.74 | 50% | 3% | 46% | 50% | 4 | 5.4 | 1.91 |

The V-lat kernel sessions (FP8/FP4 megakernels) spent only 2% of their time in the lock and 56% in their own runs: they benchmark in Bash with `bench_cold.py`-style scripts, sometimes in the background for 10 minutes, and evaluate rarely.

## 4. GPU job durations (T3, seconds)

| run | role | job | n | median | mean | p90 | max | total min |
|---|---|---|---:|---:|---:|---:|---:|---:|
| all | integration | A/B measurement | 292 | 84.0 | 118.6 | 237.1 | 790.8 | 577 |
| all | kernel | evaluate_candidate | 69 | 8.7 | 21.9 | 61.2 | 175.1 | 25 |
| all | kernel | sweep_candidate | 12 | 33.1 | 66.3 | 131.7 | 311.7 | 13 |
| all | kernel | evaluate_e2e | 2 | 67.1 | 67.1 | 68.3 | 68.6 | 2 |
| all | native | evaluate_e2e | 12 | 63.2 | 50.6 | 68.5 | 69.0 | 10 |
| all | systems | evaluate_e2e | 59 | 71.1 | 82.5 | 112.3 | 756.1 | 81 |
| V-lat | integration | A/B measurement | 62 | 88.2 | 112.5 | 210.6 | 241.8 | 116 |
| V-lat | kernel | evaluate_candidate | 35 | 7.0 | 8.0 | 11.6 | 22.2 | 5 |
| V-thr | integration | A/B measurement | 79 | 214.4 | 219.3 | 321.8 | 790.8 | 289 |
| V-thr | systems | evaluate_e2e | 21 | 86.9 | 89.8 | 107.3 | 113.7 | 31 |
| Q-lat | integration | A/B measurement | 42 | 25.0 | 31.2 | 47.7 | 85.9 | 22 |
| Q-lat | systems | evaluate_e2e | 13 | 17.3 | 31.7 | 71.0 | 71.1 | 7 |
| Q-lat | kernel | sweep_candidate | 1 | 311.7 | 311.7 | 311.7 | 311.7 | 5 |

(Full per-run table: `data/tables.md`.)

* Peak GPU memory of e2e jobs:

  | run | peak memory |
  |---|---|
  | VoxCPM2 latency | 7.5–8.7 GB |
  | VoxCPM2 throughput | 7.2–10.9 GB |
  | Qwen3 | 1.3–4.3 GB |

  On a 16 GB GPU, two VoxCPM2-scale runs cannot be resident at the same time.
* Integration work per agent evaluation, in GPU-seconds:

  | run | GPU-s per agent eval |
  |---|---:|
  | V-lat-r1 | 103 |
  | V-lat | 117 |
  | V-thr | 352 |
  | Q-lat | 111 |

## 5. Agent time between evaluations (T4, minutes)

| role | gap | n | median | mean | p90 |
|---|---|---:|---:|---:|---:|
| kernel | start → 1st eval | 43 | 5.2 | 6.2 | 12.1 |
| kernel | between evals | 40 | 2.9 | 3.4 | 7.4 |
| kernel | last eval → end | 43 | 1.7 | 3.6 | 8.5 |
| kernel | whole session, no eval | 11 | 1.6 | 5.1 | 14.9 |
| systems | start → 1st eval | 19 | 4.4 | 5.2 | 8.9 |
| systems | between evals | 40 | 2.2 | 2.6 | 5.6 |
| systems | last eval → end | 19 | 0.8 | 1.4 | 3.2 |
| native | start → 1st eval | 4 | 8.6 | 9.9 | 17.1 |
| native | between evals | 10 | 5.9 | 7.5 | 14.4 |
| native | last eval → end | 4 | 2.6 | 3.6 | 6.8 |

Evaluations per arm session:

| evals | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sessions | 20 | 23 | 15 | 12 | 14 | 1 | 1 |

## 6. Waiting on the GPU lock today (T5)

* 156 evaluation calls and 21 integration windows; **0 overlapping lock intervals, 0 waits.** Every evaluation call took its eval_s + ~0.6 s. eval_s starts before the lock is taken, so any wait would have shown up.
* The lock is uncontended only because work is sequential. The GPU itself is shared without the lock:
  * 31 background Bash jobs (benchmarks) ran outside it.
  * 53 polling commands (`until ! pgrep ...`, `nvidia-smi` memory loops) took 0.83 h while agents waited on their own jobs.
  * Three V-lat `lm_step_fp8` sessions stayed open 10 min after their last message, kept alive by a background benchmark. For example, #7 logged 27.8 min of SDK time against 37.8 min of wall time.

## 7. Cost and subscription limits (T6)

| run | $ total (API-equivalent) | agent h | $ / agent-h | agent evals | $ / agent eval | output Mtok |
|---|---:|---:|---:|---:|---:|---:|
| V-lat-r1 | 32.76 | 3.6 | 9.11 | 33 | 0.99 | 0.61 |
| V-lat | 80.97 | 9.3 | 8.70 | 54 | 1.50 | 1.44 |
| V-thr | 43.38 | 4.4 | 9.94 | 51 | 0.85 | 0.74 |
| Q-lat | 19.96 | 2.1 | 9.39 | 18 | 1.11 | 0.38 |

* All runs authenticated with the subscription login, so the dollar figures are the SDK's list-price equivalent, not billed. Per session-hour: 10.5 M cache-read, 0.47 M cache-write (1-hour TTL) and 0.16 M output tokens.
* The sessions saw 284 RateLimitEvents: 249 `allowed` (five_hour) and 35 `allowed_warning` (seven_day, on Oct 7, utilisation 0.83–0.86). There were **no `rejected` events, no API errors, no retries and no session stopped at a usage limit**.
* Utilisation slope per session-hour, measured inside each session over 42 sessions:

  | window | pooled | median | p90 |
  |---|---:|---:|---:|
  | 5-hour | +10.3 points | 9.7 | 17.6 |
  | 7-day | +2.34 points | 2.8 | - |

  These numbers are account-wide, so they include interactive use at the same time; the agents alone consume less.

| k sessions in parallel | 5-h window used per 5 h | hours to exhaust a fresh 5-h window | hours to exhaust a fresh 7-day window |
|---|---:|---:|---:|
| 1 | 52% | never (resets first) | 43 h (1.8 d) |
| 2 | 103% | 4.9 | 21 h (0.9 d) |
| 3 | 155% | 3.2 | 14 h (0.6 d) |
| 4 | 206% | 2.4 | 11 h (0.4 d) |
| 6 | 309% | 1.6 | 7 h (0.3 d) |

## 8. Simulation: k concurrent sessions on one GPU

**Model** (`data/simulate.py`, a discrete-event simulation over 400 simulated hours with 20 h warm-up and 4 seeds):

* **Agent slots.** Each of k slots replays real sessions, drawn from the 86 evaluating-arm sessions (kernel, systems, native) with replacement. A session is its measured sequence of pieces:
  * *agent*: model time and CPU tools, no GPU;
  * *eval*: a timed GPU job with its measured duration;
  * *dev*: the agent's own runs outside the lock.

  After a session ends, the slot starts another one.
* **Integration.** Every finished evaluation adds the run's integration GPU-seconds per evaluation (103–352 s) as low-priority A/B chunks. Chunk durations are drawn from the run's A/B measurement durations.
* **How dev runs use the GPU:**
  * *unlocked*: as today;
  * *strict*: a fraction f of each dev piece is an exclusive GPU job and the rest is CPU compile;
  * *rw*: a reader/writer lock where timed jobs are exclusive and dev runs share the GPU with each other but never with a timed job (writer preference for evaluations; integration yields to dev runs).
* **f.** f = 0.5 is the central value: the ninja logs show at least 46% of dev time is compile. f = 0.25 and f = 1.0 are tested as sensitivity cases.
* **Queues.** One FIFO queue, or priority evaluations > dev > integration. Jobs are not preemptive.
* **Validation:** at k = 1 without integration, the simulation gives 8.2 evals per session-hour against 8.3 measured for the arms.

### S1. Agent evaluations per hour vs k (cell: evals/h, GPU busy, mean wait of an evaluation)

† timed GPU work overlapped another agent's unlocked dev run for more than 5% of timed GPU time, so timing is not clean.
‡ integration falls behind (more than 1 h of A/B work queued after 400 simulated hours), so the setup is not sustainable.
Today's measured rate is 5.5 evals per active hour.

**Pooled (all four runs)**

| scenario | k=1 | k=2 | k=3 | k=4 | k=6 |
|---|---:|---:|---:|---:|---:|
| A: no integration; dev runs unlocked (as today) | 8.2 (12%, 0s) | 16.2 (23%, 9s)† | 23.8 (34%, 18s)† | 30.9 (44%, 29s)† | 43.8 (62%, 57s)† |
| B: integration at today's cost, one FIFO queue; dev unlocked | 7.2 (48%, 64s)† | 11.9 (79%, 166s)† | 14.2 (95%, 322s)† | 14.9 (99%, 529s)† | 15.0 (100%, 1002s)† |
| C: integration at today's cost, priority eval > integration; dev unlocked | 7.5 (50%, 48s)† | 13.6 (91%, 96s)† | 19.7 (100%, 112s)†‡ | 26.2 (100%, 113s)†‡ | 37.9 (100%, 134s)†‡ |
| D: clean: dev runs exclusive too (f=0.5), priority eval > dev > integration | 6.4 (59%, 21s) | 9.4 (87%, 44s) | 10.8 (100%, 64s)‡ | 14.4 (100%, 54s)‡ | 20.5 (100%, 40s)‡ |
| E: clean: reader/writer lock (f=0.5), integration lowest | 6.4 (59%, 21s) | 9.6 (85%, 42s) | 11.6 (97%, 64s) | 13.6 (100%, 73s)‡ | 19.3 (100%, 72s)‡ |
| **F: as E, integration 4x cheaper** | **7.7 (41%, 5s)** | **13.5 (67%, 19s)** | **17.7 (83%, 30s)** | **20.7 (92%, 41s)** | 24.5 (100%, 62s)‡ |
| G: as D, integration 4x cheaper | 7.7 (41%, 5s) | 12.9 (69%, 17s) | 16.2 (86%, 26s) | 18.0 (96%, 34s) | 21.8 (100%, 35s)‡ |
| H: as E, no integration | 8.2 (33%, 0s) | 15.2 (57%, 13s) | 20.8 (73%, 23s) | 25.3 (84%, 32s) | 31.8 (94%, 47s) |
| G with f = 1.0 (every dev second on the GPU) | 7.7 (60%, 5s) | 11.1 (88%, 33s) | 12.5 (98%, 50s) | 13.5 (100%, 51s)‡ | 15.0 (100%, 49s)‡ |
| G with f = 0.25 | 7.8 (31%, 5s) | 13.6 (55%, 12s) | 17.9 (72%, 19s) | 21.1 (85%, 25s) | 24.5 (99%, 35s) |

**Per run (evals/h; same scenarios)**

| run | scenario | k=1 | k=2 | k=3 | k=4 | k=6 |
|---|---|---:|---:|---:|---:|---:|
| V-lat (latency; kernel-heavy, 7 s evals) | A unlocked, no integration | 5.8 | 11.8† | 17.7† | 23.6† | 35.2† |
| | E clean rw, today's integration | 5.4 | 9.3 | 12.1 | 14.0 | 16.3‡ |
| | F clean rw, integration ÷4 | 5.8 | 10.8 | 15.0 | 18.4 | 23.4 |
| V-thr (throughput; 214 s A/B rounds) | A | 12.1 | 23.7† | 34.2† | 43.5† | 56.4† |
| | E | 6.2 | 7.6 | 8.2 | 10.1‡ | 14.5‡ |
| | F | 9.8 | 14.8 | 17.7 | 19.5 | 21.5 |
| Q-lat (Qwen3) | A | 8.7 | 16.9† | 25.0† | 32.7† | 46.6† |
| | E | 8.0 | 13.8 | 18.5‡ | 23.5‡ | 30.8‡ |
| | F | 8.6 | 15.4 | 21.1 | 25.5 | 31.7 |

(Every scenario for every run, with all the metrics: `data/sim_tables.md`, `data/sim_results.json`.)

### S2. Pooled detail, scenario F (clean timing with a reader/writer lock, integration 4x cheaper)

| k | evals/h | vs k=1 | per-agent efficiency | GPU: eval / integration / dev | GPU busy | eval wait mean / p90 s | dev wait s | agent time blocked |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.7 | 1.00x | 100% | 11% / 10% / 20% | 41% | 5 / 0 | 5 | 6% |
| 2 | 13.5 | 1.74x | 87% | 19% / 18% / 31% | 67% | 19 / 55 | 16 | 18% |
| 3 | 17.7 | 2.28x | 76% | 25% / 23% / 35% | 83% | 30 / 79 | 28 | 28% |
| 4 | 20.7 | 2.68x | 67% | 29% / 27% / 36% | 92% | 41 / 98 | 43 | 37% |
| 6 | 24.5 | 3.17x | 53% | 35% / 32% / 33% | 100% | 62 / 136 | 75 | 50% (integration backlog 1.8 h) |

What the simulation says:

1. **Agent evaluations alone barely load the GPU.** They use 12% of the GPU at k = 1 and 62% at k = 6 (scenario A). Concurrency scales nearly linearly until something else fills the GPU.
2. **Unlocked runs and concurrency do not mix.** With dev runs outside the lock (today's practice), 40% (k = 2) to 88% (k = 6) of timed GPU time overlaps another agent's run (scenario A, †). Clean timing needs every GPU-touching process under the queue or the reader/writer lock.
3. **Integration is the first thing to saturate the GPU.** At today's cost (100–350 GPU-seconds of A/B per evaluation), one FIFO queue saturates at about 15 evals/h with waits of 5–17 min (scenario B). Giving evaluations priority keeps them moving but starves integration from k = 3 (scenario C, ‡). With clean timing, the useful limit is k ≈ 3 (E: 11.6/h, 2.1x today) unless integration gets cheaper.
4. **Making integration 4x cheaper moves the knee to k ≈ 4.** That gives 2.5–3.8x today's measured rate at k = 2–4 (F), with mean evaluation waits under 45 s. Beyond k = 4 the GPU is full and half the agent time is blocked.
5. **Reader/writer lock against exclusive dev jobs:** similar at k ≤ 2. The reader/writer lock is ahead at k = 3–4 (F 17.7 / 20.7 against G 16.2 / 18.0). The result is sensitive to f: if every dev second really is GPU time (f = 1), k = 3 gives only 12.5/h.
6. **Long jobs block evaluations even at k = 1.** Today's 84–214 s A/B chunks already cause 21–64 s mean evaluation waits once integration runs asynchronously (B and E at k = 1).

## 9. Design implications for #174

1. **The GPU is not the bottleneck for agent work; use concurrency on one GPU, with k = 3–4.**
   * The lock is held 11% of session time, and the GPU computes at most about 34% of the time an agent works.
   * With clean timing and cheaper integration, k = 3–4 gives 2.3–2.7x the simulated single-agent rate (3.2–3.8x today's 5.5 evals per active hour).
   * k > 4 mostly adds blocked agents.
2. **Put every GPU-touching process under the scheduler, not only the evaluator.**
   * Agents' Bash runs (40% of session time) and background benchmarks run outside the lock today. Under concurrency they would corrupt 40–88% of timed measurements.
   * Provide a queue API for agent dev runs: a "quick/dev" class and a "timed" class. Timed jobs are exclusive. Dev jobs are exclusive when they time something (agents' own microbenchmarks), or shared through a reader/writer lock when they only check correctness.
3. **Use priority classes and small jobs.**
   * Order: agent evaluation (an agent is blocked) > agent dev run > integration A/B (batch).
   * Split jobs so none blocks the GPU for minutes: integration into single A/B rounds, sweeps into configs, e2e into runs. Today a single A/B measurement lasts up to 791 s, a sweep up to 312 s and an e2e evaluation up to 756 s, and they are not preemptible.
4. **Make integration cheaper before scaling k.** It is the dominant GPU consumer: 29% of run wall time and 8.2 h of lock time against 2.2 h for agent evaluations. Options:
   * sequential A/B that stops early on clear wins and losses,
   * reuse measurements whose content is unchanged (already partly done: "1 of 11 measurements reused"),
   * integrate only kept results predicted to matter,
   * fewer rounds for the combination,
   * cheaper A/B on throughput workloads, where one measurement takes 214 s.
5. **Move compilation out of the GPU path and manage it as CPU work.**
   * nvcc builds are 31% of kernel session wall time, and agents compile seven variants in parallel. On a 12-thread CPU, k = 4 agents doing this would contend for cores.
   * A compile service gives a CPU budget per agent and a shared build cache. The evaluator already compiles native projects outside the lock (`_prebuilt`); keep it that way for load_inline too.
6. **Admit GPU work by memory, not only by time.** VoxCPM2 e2e and dev runs peak at 7.5–11 GB on a 16 GB GPU, so at most one of them can be resident. Even with one job at a time, V-lat's results.tsv has two `oom` rows, so memory is already tight. Kernel-level dev runs on captured module inputs are much smaller and can share the GPU.
7. **Replace polling with notifications.** Agents spent 0.83 h polling their own background jobs, and sessions were held open for 10 min by them. A submit/await job API with completion messages (a shared blackboard or messages) gives that time back and keeps sessions from hanging.
8. **Make concurrency budget-aware; on a subscription the usage budget, not the GPU, sets k.**
   * One session uses about 10 points of the 5-hour window and 2.3 points of the 7-day window per hour, and one agent at a time already drove the 7-day window to 0.86.
   * k = 2 fills the 5-hour window. k = 3–4 empties it in 2.4–3.2 h and empties a fresh week in 11–14 h. Concurrency shortens wall time but does not add budget.
   * The scheduler can read the live `five_hour` / `seven_day` utilisation from the RateLimitEvents already in `logs/agent-*.jsonl`, choose k and the model / effort per role from it, and back off on `allowed_warning`.
9. **Cost per evaluation stays about the same under concurrency** ($1.14 API-equivalent). Tokens are spent thinking, not waiting, and the 1-hour prompt cache (ephemeral_1h) survives queue waits of seconds to a few minutes. Cost per wall-hour grows with k (about $9.3 per agent-hour).
10. **Run the non-GPU roles beside the arms, and use concurrency to hide ramp-up.**
    * Arm sessions are short (median 12 min) with 4–9 min before their first evaluation, and 20 of 86 end without one. Running several arms at once hides this ramp-up. Longer sessions, or sessions that keep their context, would cut it.
    * Planner, dossier, research and librarian use no GPU and take 3% of the time. They can run concurrently, for example preparing the next target's dossier or research while an arm runs.

## 10. Caveats

* "Dev GPU" is a heuristic upper bound. Measured compile time covers ninja-built extensions only; Triton, TileLang, CuTe and inductor JIT compiles and model loading are not separated out, so the true GPU share of dev time is likely below 0.5 for kernel work. The simulation brackets this with f = 0.25 and f = 1.0.
* The simulation resamples whole sessions independently. It does not model:
  * CPU contention (parallel nvcc builds) or GPU-memory contention,
  * any slowdown of shared dev runs,
  * search-quality effects of concurrency (duplicate ideas, agents learning from each other),
  * the scheduler's slice ordering.

  It also assumes integration work is proportional to agent evaluations.
* Integration counts for V-lat-r1 leave out two manual re-integrations (96 min) that are not in its events.jsonl, so its 103 GPU-s per evaluation is low.
* Rate-limit slopes are account-wide (they include interactive use) and per-session: an upper bound on agent consumption, so the "hours to exhaust" figures are lower bounds. The plan's absolute limits are not visible in the data.
* Dollar figures are the SDK's list-price equivalent; the runs used subscription auth.

## Reproduce

All in `scratchpad/174/data/`. The scripts only read the run directories, `~/.claude/projects` transcripts and `~/.cache` ninja logs.

```
python3 extract.py        # sessions.json, evals.json, integrations.json, ratelimits.json, phases.json
python3 analyze.py        # tables.md (T1-T6), session_table.csv (one row per session)
python3 compile_share.py  # compile_share.json (ninja build time inside dev intervals)
python3 simulate.py       # sim_results.json (about 2 min)
python3 report_sim.py     # sim_tables.md
```
