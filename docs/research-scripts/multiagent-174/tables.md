## T1. Where the wall time of a run goes (today: one agent at a time)

Active wall = union of agent sessions, integrations and the analyze/plan/capture phases, gaps under 10 min merged (overnight pauses between invocations excluded); the plan phase is the planner session, so it is counted there. GPU-lock busy = evaluation tool calls inside sessions + integration windows + analyze/capture. Dev GPU = Bash commands that run Python/benchmarks/profilers (incl. background jobs and waits on them), outside the lock.

| run | active h | sessions | agent sessions h | integration h | analyze + capture h | orchestrator other h | agent evals | integration measurements | agent evals / active h | GPU-lock busy | dev GPU (outside lock) | GPU idle | $ (API-equiv.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V-lat-r1 | 4.7 | 20 | 3.6 (77%) | 0.9 (20%) | 0.08 (2%) | 0.07 (2%) | 33 | 109 | 7.0 | 39% | 23% | 38% | 33 |
| V-lat | 11.3 | 42 | 9.3 (83%) | 1.8 (16%) | 0.11 (1%) | 0.09 (1%) | 54 | 62 | 4.8 | 20% | 41% | 39% | 81 |
| V-thr | 9.6 | 33 | 4.4 (45%) | 5.0 (52%) | 0.18 (2%) | 0.12 (1%) | 51 | 79 | 5.3 | 62% | 15% | 24% | 43 |
| Q-lat | 2.7 | 12 | 2.1 (78%) | 0.6 (20%) | 0.05 (2%) | 0.00 (0%) | 18 | 42 | 6.6 | 30% | 32% | 37% | 20 |
| **all** | 28.3 | 107 | 19.4 (68%) | 8.2 (29%) | 0.42 (1%) | 0.29 (1%) | 156 | 292 | 5.5 | 38% | 28% | 33% | 177 |

## T2. Agent sessions by role (all four runs)

Exclusive split of session wall time, priority GPU lock > model > dev GPU > CPU tools > idle. model = an API call in flight (thinking + writing; its total matches duration_api_ms of the ResultMessage within 2%); GPU lock = evaluate_candidate / evaluate_e2e / sweep_candidate calls; dev GPU = Bash running Python / benchmarks / profilers, background jobs and waits on them (outside the lock; includes load_inline / nvcc compiles started from Python); CPU tools = other Bash, Read/Write/Grep, WebFetch, ToolSearch; idle = none of these (SDK start-up, hooks, a session tail kept open by a background job). GPU any = lock or dev GPU (the dev part counted even while the model thinks).

| role | sessions | hours | min/session | model | GPU lock | dev GPU | CPU tools | idle | GPU any | evals | evals / session-h | median min to 1st eval | sessions w/o eval | $ / session | $ / eval | turns / session |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| planner | 9 | 0.4 | 2.7 | 97% | 0% | 0% | 1% | 1% | 0% | 0 | 0.0 | - | 9 | 0.78 | - | 9 |
| dossier | 4 | 0.0 | 0.6 | 80% | 0% | 0% | 17% | 3% | 0% | 0 | 0.0 | - | 4 | 0.19 | - | 6 |
| research | 1 | 0.1 | 3.9 | 99% | 0% | 0% | 0% | 1% | 0% | 0 | 0.0 | - | 1 | 1.17 | - | 19 |
| kernel | 54 | 10.9 | 12.1 | 45% | 6% | 46% | 0% | 3% | 54% | 83 | 7.6 | 5.2 | 11 | 1.77 | 1.15 | 22 |
| systems | 25 | 5.2 | 12.5 | 39% | 26% | 34% | 1% | 0% | 60% | 59 | 11.3 | 4.4 | 6 | 1.68 | 0.71 | 26 |
| native | 7 | 2.7 | 23.1 | 53% | 6% | 41% | 0% | 0% | 47% | 14 | 5.2 | 8.6 | 3 | 4.04 | 2.02 | 47 |
| librarian | 7 | 0.0 | 0.4 | 95% | 0% | 0% | 0% | 5% | 0% | 0 | 0.0 | - | 7 | 0.31 | - | 2 |
| all | 107 | 19.4 | 10.9 | 46% | 11% | 40% | 0% | 2% | 53% | 156 | 8.0 | 5.2 | 41 | 1.65 | 1.14 | 22 |

Sessions kept open > 1 min after their last message (Claude Code waiting for a background job): 3 (V-lat:kernel-lm_step_fp8#7 10 min, V-lat:kernel-lm_step_fp8#8 10 min, V-lat:kernel-lm_step_fp8#10 10 min).

### T2b. Same split per run, evaluating arms only (kernel + systems + native)

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

## T3. GPU job durations (seconds)

Evaluation tool calls (start of tool_use to its result; equals results.tsv eval_s + ~0.6 s) and integration A/B measurements (results.tsv eval_s of `integrate` / `re-evaluated` rows).

| run | role | job | n | median | mean | p90 | max | total min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | integration | A/B measurement | 292 | 84.0 | 118.6 | 237.1 | 790.8 | 577 |
| all | kernel | evaluate_candidate | 69 | 8.7 | 21.9 | 61.2 | 175.1 | 25 |
| all | kernel | evaluate_e2e | 2 | 67.1 | 67.1 | 68.3 | 68.6 | 2 |
| all | kernel | sweep_candidate | 12 | 33.1 | 66.3 | 131.7 | 311.7 | 13 |
| all | native | evaluate_candidate | 2 | 6.9 | 6.9 | 7.7 | 7.9 | 0 |
| all | native | evaluate_e2e | 12 | 63.2 | 50.6 | 68.5 | 69.0 | 10 |
| all | systems | evaluate_e2e | 59 | 71.1 | 82.5 | 112.3 | 756.1 | 81 |
| Q-lat | integration | A/B measurement | 42 | 25.0 | 31.2 | 47.7 | 85.9 | 22 |
| Q-lat | kernel | sweep_candidate | 1 | 311.7 | 311.7 | 311.7 | 311.7 | 5 |
| Q-lat | native | evaluate_e2e | 4 | 19.0 | 23.1 | 30.7 | 35.8 | 2 |
| Q-lat | systems | evaluate_e2e | 13 | 17.3 | 31.7 | 71.0 | 71.1 | 7 |
| V-lat | integration | A/B measurement | 62 | 88.2 | 112.5 | 210.6 | 241.8 | 116 |
| V-lat | kernel | evaluate_candidate | 35 | 7.0 | 8.0 | 11.6 | 22.2 | 5 |
| V-lat | kernel | evaluate_e2e | 2 | 67.1 | 67.1 | 68.3 | 68.6 | 2 |
| V-lat | kernel | sweep_candidate | 3 | 6.7 | 39.8 | 87.1 | 107.2 | 2 |
| V-lat | native | evaluate_candidate | 2 | 6.9 | 6.9 | 7.7 | 7.9 | 0 |
| V-lat | native | evaluate_e2e | 8 | 64.1 | 64.3 | 68.8 | 69.0 | 9 |
| V-lat | systems | evaluate_e2e | 4 | 66.8 | 65.6 | 67.7 | 68.0 | 4 |
| V-lat-r1 | integration | A/B measurement | 109 | 66.5 | 82.8 | 141.6 | 288.6 | 150 |
| V-lat-r1 | kernel | evaluate_candidate | 12 | 19.2 | 53.9 | 129.1 | 175.1 | 11 |
| V-lat-r1 | systems | evaluate_e2e | 21 | 46.7 | 110.0 | 225.8 | 756.1 | 39 |
| V-thr | integration | A/B measurement | 79 | 214.4 | 219.3 | 321.8 | 790.8 | 289 |
| V-thr | kernel | evaluate_candidate | 22 | 10.1 | 26.4 | 59.3 | 145.6 | 10 |
| V-thr | kernel | sweep_candidate | 8 | 33.1 | 45.5 | 93.9 | 134.4 | 6 |
| V-thr | systems | evaluate_e2e | 21 | 86.9 | 89.8 | 107.3 | 113.7 | 31 |

compile_s reported inside evaluate_candidate results: n=76, median 0.5 s, p90 0.9 s, max 36.0 s (agents build in Bash first, so the evaluator mostly hits the extension cache; compiles happen in the dev-GPU Bash time instead).

## T4. Agent time between GPU-lock jobs (minutes)

Gaps from session start (or the end of the previous evaluation) to the next evaluation call; 'tail' is the last evaluation to session end. Includes model, dev-GPU Bash and CPU tools.

| role | gap | n | median | mean | p90 |
|---|---:|---:|---:|---:|---:|
| systems | first | 19 | 4.4 | 5.2 | 8.9 |
| systems | between | 40 | 2.2 | 2.6 | 5.6 |
| systems | tail | 19 | 0.8 | 1.4 | 3.2 |
| systems | no-eval session | 6 | 0.5 | 0.6 | 0.8 |
| kernel | first | 43 | 5.2 | 6.2 | 12.1 |
| kernel | between | 40 | 2.9 | 3.4 | 7.4 |
| kernel | tail | 43 | 1.7 | 3.6 | 8.5 |
| kernel | no-eval session | 11 | 1.6 | 5.1 | 14.9 |
| native | first | 4 | 8.6 | 9.9 | 17.1 |
| native | between | 10 | 5.9 | 7.5 | 14.4 |
| native | tail | 4 | 2.6 | 3.6 | 6.8 |
| native | no-eval session | 3 | 3.0 | 7.6 | 14.2 |

## T5. Waiting on the GPU lock and GPU overlap today

* GPU-lock jobs: 156 agent evaluation calls + 21 integration windows. Overlapping lock intervals across sessions/integration: 0 pairs [].
* Every evaluation call lasted its eval_s + ~0.6 s (eval_s starts before the lock is taken), so no agent waited for the lock: with one session at a time and integration run between slices, the lock is never contended.
* Background Bash jobs started by agents: 31; jobs still running when their session ended: 0 (). Such jobs, and every dev-GPU Bash run, use the GPU outside the lock.
* Agents instead wait on their *own* GPU work: 53 Bash polls (until/while/pgrep/nvidia-smi loops, Monitor) took 0.83 h; 3 sessions stayed open 10 min after their last message for a background benchmark.

## T6. Cost and subscription limits

| run | $ total (API-equivalent) | agent h | $ / agent-h | agent evals | $ / agent eval | output Mtok |
|---|---:|---:|---:|---:|---:|---:|
| V-lat-r1 | 32.76 | 3.6 | 9.11 | 33 | 0.99 | 0.61 |
| V-lat | 80.97 | 9.3 | 8.70 | 54 | 1.50 | 1.44 |
| V-thr | 43.38 | 4.4 | 9.94 | 51 | 0.85 | 0.74 |
| Q-lat | 19.96 | 2.1 | 9.39 | 18 | 1.11 | 0.38 |

* Rate-limit events seen by the sessions: {('allowed', 'five_hour'): 249, ('allowed_warning', 'seven_day'): 35}; no `rejected` status, no session stopped at a usage limit (`allowed_warning` = the seven-day window above its warning threshold; highest seven-day utilisation seen 0.86).
* Five-hour window: utilisation rises 10.3 percentage points per session-hour (pooled over 42 sessions; median per session 9.7, p90 17.6). Seven-day window: 2.34 points per session-hour (median 2.81). These are account-wide (they include any interactive use at the same time), so they are upper bounds for the agents alone.
| k sessions in parallel | 5-h window used per 5 h | hours to exhaust a fresh 5-h window | hours to exhaust a fresh 7-day window |
|---|---:|---:|---:|
| 1 | 52% | never (resets first) | 43 h (1.8 d) |
| 2 | 103% | 4.9 | 21 h (0.9 d) |
| 3 | 155% | 3.2 | 14 h (0.6 d) |
| 4 | 206% | 2.4 | 11 h (0.4 d) |
| 6 | 309% | 1.6 | 7 h (0.3 d) |
