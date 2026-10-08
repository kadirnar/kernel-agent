> Design for [#174](https://github.com/kadirnar/kernel-agent/issues/174): kernel-agent as a multi-agent system on one GPU. Literature review: [MULTIAGENT-LITERATURE.md](MULTIAGENT-LITERATURE.md). Measured time split and k-agent simulation from our runs: [MULTIAGENT-DATA.md](MULTIAGENT-DATA.md) (it replaces the illustrative numbers of §4). Implementation follows the staged PR plan in §5 (one issue per PR).

# #174 Multi-agent architecture: codebase audit and design proposal

Scope: the *architecture* part of issue #174. It covers how orchestration works today
(at `5994b67`), what stops several agent sessions from running at once, and a concrete
design with a staged PR plan. It also covers efficiency (model mix per role, prompt
caching, in-session subagents, critic, early termination, batching: §3.12) and how the
coordinator uses #176's agent definitions and skills and #177's doc tools (§3.13). The
literature survey and the measured time split from run logs belong to the other parts of
the issue. The numbers in §4 are an illustrative
queueing model to be replaced by those measurements.

Line numbers refer to `src/kernel_agent/` at `5994b67`.

---

## 0. Summary

* Today one coordinator process (`improve.Improver._loop`) gives one slice to one arm
  and waits for it. The only concurrency is inside a slice (`--seeds-per-target` workers,
  `--parallel` semaphore, `improve.py:998`) and in the non-improve `kernels` phase
  (`orchestrator.py:545`). Roles run one after another.
* Keep the process model: **one coordinator process, N agent sessions as asyncio tasks**.
  The in-memory `Truth`, the ledger `RLock`, the dedup `_inflight` map and the in-process
  MCP tools all assume one process. A second process would corrupt the truth digests in
  `run.json`.
* Several things block concurrency, and they must be fixed first (PR 0):
  1. The integration, captures, pivots and native stage captures call `subprocess.run`
     **on the event-loop thread**, which freezes every other session for up to an hour.
  2. `run.json` is read-modify-written from both the loop thread and worker threads
     (`Truth._persist` in library seeding), so truth digests can be lost and a target's
     records can then be refused.
  3. `Budget` holds per-run globals (`kernel_evals`, `transform_evals`) that each slice
     overwrites.
  4. Slices attribute ledger rows by `exp` range, which breaks as soon as two sessions
     share an arm.
  5. USD accounting only sees finished sessions, so N sessions can overspend by up to N
     per-session caps.
* The **GPU job queue** sits *inside* `gpulock.gpu_lock()`. It is a priority admission
  controller fed by a `contextvars` job tag that the MCP tool sets before
  `asyncio.to_thread`. The tag reaches every existing lock site (evaluate, sweep, recheck,
  memcheck, ncu, region, worker, profiler) without signature changes. The flock stays as
  the outer, cross-process layer. Priorities from high to low: final integration in the
  reserve window, quick/debug, full eval, e2e, sweep, background (re-integration A/B,
  recheck, seeding, capture). Ties go to the shortest job and long waits age upward. Each
  session's timeout is extended by the time it waits in the queue.
* **Clean timing** needs more than an exclusive lock:
  * load_inline candidates should compile off the lock (projects already do).
  * Timed jobs should get pinned cores and compile jobs a lower CPU priority.
  * Other GPU contexts present during a timed run should be detected, and the run marked
    dirty and repeated.
  * Agents' Bash scripts that use the GPU should go through a `run_on_gpu` tool. Today
    "short debug scripts are fine" (`prompts.py:65`), which bypasses the lock.
* **Blackboard**: a coordinator-written `board.jsonl` plus `post_note` / `read_board` MCP
  tools. New entries also ride on every evaluation result (the natural interrupt point,
  like the budget advice). It carries insights, winners, claims, critic reviews and
  messages addressed to an arm's mailbox.
* **Critic**: it runs *while the candidate waits in the GPU queue*. It adds no latency.
  It can withdraw a queued job with a reason, and the agent can override with `force`.
  Its verdicts on jobs that ran anyway give free calibration data.
* **Islands**: generalise `workers.py` into persistent islands with island-level UCB,
  migration of elites (inspirations), and culling or reseeding. An arm's concurrency cap
  equals its island count.
* **Coordinator**: an event-driven loop over a `running` set. Slots go to arms by the
  existing scores, with virtual pulls for running sessions and per-arm caps. When the GPU
  is saturated, free slots go to GPU-free roles (research, dossier, critic). The
  re-integration runs as a background task. A round boundary is a barrier.
  `--agents N` (default 1 = today's behaviour), and `auto` uses a governor driven by queue
  measurements and `RateLimitEvent.utilization`.
* **Dry run**: switch to a virtual-time asyncio loop so `--dry-run --agents N` simulates
  real overlap. Simulated agents then `await` think time and GPU jobs through the same
  queue, which checks the invariants and the knee of N.
* **Efficiency** (§3.12):
  * *Model mix per role*: Opus 5.5 for the engineers, native, planner and research;
    Sonnet 5.5 for dossier, doc lookup, compile triage and the librarian; Haiku 4.5 for
    critic triage. This also spares the separate Opus usage window on a subscription.
  * *Prompt-cache-friendly prompts*: today the engineer prompt puts the target-specific
    header *before* the large stable blocks (rules, playbook, backend guides, env;
    `prompts.engineer_prompt`). So no cache prefix is shared across targets. Fix: stable
    first, volatile last, byte-identical tool lists per role, and staggered starts so
    concurrent sessions read the first one's cache. Measure the hit rate from
    `ResultMessage.usage`.
  * *In-session delegation*: read-only SDK subagents for doc lookup, compile-error triage
    and profile analysis keep the engineer's context small.
  * *Critic* before the GPU evaluation.
  * *Early termination of hopeless branches* (idea, island, session, and early discards in
    the evaluator, sweep racing and sequential A/B). It never retires a target or the run
    on a guess: every terminated branch hands its slot to the next.
  * *Batching*: same-target jobs under one lease, a same-session multi-candidate tool, and
    one model load for a batch of e2e sets with in-process undo.
* **Roles as SDK agent definitions + skills (#176) + doc tools (#177)** (§3.13): one
  `roles.py` registry (model, effort, tools, skills, subagents, writable roots, GPU need,
  caps). It builds both the top-level `ClaudeAgentOptions` and the `AgentDefinition`s.
  Skills come from #176's packaged local plugin (`plugins=[...]` with
  `setting_sources=[]`, so #126 isolation holds; verified in SDK 0.2.163 that `skills=`
  does not override an explicit `setting_sources`). `doc_search`/`doc_read` (#177) are
  GPU-free tools for every role. The coordinator also pre-retrieves "docs to read first"
  for each digest at zero LLM cost.
* **PR plan** (§5), in order:
  * PR 0: safety prerequisites (no behaviour change)
  * PR 1: role registry, per-role models and cache-friendly prompts (useful at N = 1)
  * PR 2: GPU queue
  * PR 3: concurrent coordinator and virtual-time dry run
  * PR 4: observability
  * PR 5: timing hygiene
  * PR 6: subagents + doc tools
  * PR 7: blackboard
  * PR 8: critic
  * PR 9: islands
  * PR 10: early termination and batching in measurement
  * PR 11: governor and async evaluations

---

## 1. How orchestration works today

```
kernel-agent improve
 └─ Improver.improve → _loop (improve.py:751), strictly sequential:
     _keep_time()                     final-integration reserve (Budget.final_reserve_s)
     seed_library(targets)            GPU work (to_thread), before any slice
     arms = _pickable()               scheduler.build_arms from ledger + slices + profiles
     arm = pick(arms); _in_time(arms) ONE arm with time for a slice
     research_due → _research         read-only session writing plan.md (pivot.json)
     dossier_due  → _dossier          web session writing research.md
     _slice(arm)  → orch.kernel_slice | systems_slice | native_slice | _workers(team)
                     └─ Orchestrator._agent → runner.run_agent (Claude Agent SDK query)
                          MCP tools (agent/tools.py, in-process) → asyncio.to_thread(
                            run_evaluation | run_sweep | call_worker) → gpu_lock()
     every --integrate-every keeps → reintegrate() → orch.integrate (BLOCKING, see §2.1)
     no live arm → _next_round: reintegrate, re-profile, re-plan, capture (barrier)
   _finish → final integration → report → librarian
```

Concurrency that exists:

| Where | Mechanism | Shared state it already handles |
|---|---|---|
| `Orchestrator.kernels` (`optimize` path) | `asyncio.Semaphore(cfg.parallel)` over targets and workers | per-target dirs; shared ledger; GPU lock |
| `Improver._workers` (one slice) | `gather` of worker sessions under `Semaphore(cfg.parallel)` | `workers.Binding` per session (own dir, agent name, eval budget); shared `results.jsonl`/`history/` via symlinks; `_inflight` dedup of identical sources (`tools.py:36,587`) |
| Evaluations | `asyncio.to_thread` + `gpulock.gpu_lock()` (thread lock + flock, pool of GPUs) | one GPU job at a time per GPU, across threads and processes |

Everything else is serial: one arm at a time, research and dossier sessions block the
loop, and so do the integration, the re-profile and the round.

---

## 2. Audit: what is safe with several sessions in one process

Legend: **OK** safe as is · **Fix** must change before N > 1 · **Risk** works but degrades.

### 2.1 Event loop and process model

| Component | Today | Verdict |
|---|---|---|
| `Orchestrator.integrate` (`orchestrator.py:798`) | `async def`, but every A/B step runs `self._worker` → `worker.call_worker` → `subprocess.run` **on the loop thread** (`ab()` → `_paired` → `_integration_call:1463`; `_recheck_kernels` likewise). A re-integration took 45 min (12 A/B × 3.8 min) and a full one 113 min (`scheduler.py:139-143`). | **Fix**: every session's SDK stream and MCP tool call freezes meanwhile. Run the sync body in `asyncio.to_thread` (`_integrate_sync`). |
| `Orchestrator.capture_targets` → `_capture` (`:463`) | sync `self._worker("capture")` on the loop thread; used by `pivot` (`:2029`), `Improver._native_stage` (`improve.py:941`) and `_start_round` | **Fix**: same; `to_thread`. |
| `Orchestrator.refactor` install | already `to_thread` | OK |
| `reprofile` | `to_thread` (`improve.py:974,1318`) | OK |
| Agent sessions | each `query()` spawns its own Claude Code CLI subprocess; the SDK's MCP server routes requests per call | OK for N tasks. One server instance per session is still wanted for bindings (§3.6). |
| Truth (`truth.of`: one instance per run **per process**) | in-memory digests, persisted into `run.json` | **Constraint**: never two coordinator processes on one run. Write a pid lock on `<run>/.coordinator.lock` and refuse a second `improve`. |
| Ctrl-C (`interrupt.py`) | cancels the main task and kills descendants | OK for N tasks, provided the coordinator owns them in a `TaskGroup` and every task closes its record (§3.8). |

### 2.2 Shared files and in-memory state

| File / state | Writers today | Concurrency verdict |
|---|---|---|
| `.truth/**/results.jsonl` (`Truth.append`, `truth.py:241`) | MCP tools on the loop thread; `library.seed_target` **in a worker thread** (`orchestrator.py:2150` → `library.py:552` `record_candidate`) | Appends are under `Truth._lock`. But `_persist` (`truth.py:186`) read-modify-writes **`run.json`**, and so do `budget.note` (`budget.py:148`), `Orchestrator._mark` (`:195`), `program.for_agent` (`program.py:151,171`), `library.remember_*` (`library.py:657,667,894`), the kernels-phase lists (`:609,740`) and `check_harness` (`tools.py:955`). None of those take the Truth lock. **Fix**: a lost update of the `truth.files` section turns into a `TamperError`, and then *all* of the target's records are ignored (`ranked_for_target`). Use one `workspace.update_json(path, fn)` with a per-path lock for every RMW site. |
| `write_json` tmp name (`workspace.py:58`) | fixed `<name>.json.tmp` | **Fix**: two threads writing the same file race on the tmp (torn content or a failed `replace`). Use a unique tmp name (`.tmp.<pid>.<thread>`). |
| `results.tsv` (ledger) | `ledger.append` under `_lock` (RLock), exp = line count | Writers are OK. **Risk**: `ledger.rows()` reads without the lock and can parse a partly written line from a thread writer. Read under the lock, or drop a last line without a newline. |
| `events.jsonl`, `logs/agent-<name>.jsonl` | `append_jsonl` (single `write`) | OK while **agent names are unique among running sessions**; two sessions with the same name would interleave one log. Invariant enforced by the coordinator (§3.2). |
| `costs.json` | `_agent` end (loop thread, sync RMW) | OK in asyncio. **Fix** for budgets: it only contains *finished* sessions (§2.4). |
| `improve.json` (`Improver.save`) | loop thread | OK, but `Improver.doing` is one string and `_interrupted` records one activity: **Fix** to a set of running activities. |
| `history/NNN_*.py` numbering (`tools._snapshot`) | loop thread (evaluate) and **a thread** (`sweep` `prepare` callback, `tools.py:762`) | **Risk**: two snapshots can take the same `NNN` (names stay distinct through stem and sha). Take a per-directory lock and create with `O_EXCL`. |
| `dashboard.html` (`dashboard.write_dashboard`, fixed `.dashboard.tmp.html`) | `to_thread(refresh)` after every evaluation | **Risk**: concurrent refreshes race on the tmp. Charts are already serialised (`charts._lock`). Use one debounced refresher task (one thread, coalesced requests). |
| `NOTES.md`, `plan.md`, `research.md` | agents (own dirs); research sessions are write-guarded | OK per arm. Classic session and island sessions of one target must not run together (classic `cwd` contains `workers/`). |
| `integration.json` | integration (end of run, atomic replace) | OK. Digests read the last complete one. |
| `tools._inflight` | per (run, target, source key) | OK, designed for concurrent workers. |
| `budget.final_reserve_s` re-estimate after every evaluation (`budget.py:310`, `Improver._integration_estimate` hashes every item file) | loop thread | **Risk** (CPU): N sessions mean N× re-estimates. Cache by ledger length. |

### 2.3 GPU lock semantics (`gpulock.py`)

What it is: per GPU, a `threading.Lock` (in-process) plus an `flock` (cross-process),
re-entrant per thread. A child gets `KERNEL_AGENT_LOCK_HELD=1`. With several GPUs it
takes the first free one, else the one with the fewest waiters.

| Property | Today | Problem with N agents |
|---|---|---|
| Ordering | In-process: `Lock.acquire(timeout=0.25)` polling, so **no FIFO, no priorities**. Across processes: kernel flock order. | A 5-minute sweep or a 4-minute A/B step can starve the quick checks an agent is blocked on. Long jobs repeatedly win. |
| What the hold covers | The whole evaluator subprocess: import, **load_inline/nvcc and Triton JIT compile** (`compile_s` is measured inside, `evaluate.py:640`), correctness, timing, reference re-timing (`_check_reference_timing`). For e2e: model load + workload. | CPU-only compile time occupies the GPU. Only native projects prebuild outside (`tools._prebuilt`, `native/project.py:930`, `CUDA_VISIBLE_DEVICES=""`). |
| Granularity | One exclusive holder per GPU. | Fine for timing. Quick checks and correctness stages could share, but only with memory admission (§3.3, later stage). |
| Agents' own GPU use | `prompts._env_block`: "Do not run long GPU jobs yourself; short compile/debug scripts are fine". Bash runs without the lock. | With N agents, debug scripts overlap **timed** runs of other agents. Today the only defences are `_check_reference_timing` (a candidate-free re-time catches a slowed reference) and `telemetry.Monitor.processes` on e2e. |
| CPU contention | Nothing isolates the timed process. | nvcc/ninja builds, N Claude Code CLIs and agents' Bash steal cores during host-launch-bound timings (VoxCPM2 and Qwen decode are launch-bound; see memory note on `_scaled_mm` host overhead). |
| Session timeouts | `asyncio.timeout(agent_minutes)` around the whole session (`orchestrator.py:255`) | Queue waits eat the session's minutes. Sessions time out with 0 evaluations, which the scheduler counts as `idle` and then stops the arm (`IDLE_SLICES`, `scheduler.py:911`). |
| MCP tool latency | Tool call blocks for queue + evaluation | **Verify** the Claude Code MCP tool timeout for SDK servers (not set in `SESSION_ENV`). Long waits plus a long e2e may exceed it. |

### 2.4 Budget (`budget.py`)

| Field / method | Problem | Fix |
|---|---|---|
| `kernel_evals`, `transform_evals` (run-wide) | Overwritten per slice (`orchestrator.py:1812,1840,1884`; `improve.py:734`). A systems slice changes a running kernel session's `evals_budget`, which `tools.py:580,939` reads at each call. | Bind the evaluation budget per session in the MCP server: generalise `workers.Binding` into a `SessionBinding` (label, role, target, island, evaluations, cwd). `build_server` already takes `agent=`/`evaluations=` for native. |
| `deadlines`, `evals`, `restarted`, `minutes_by_agent` keyed by agent *name* | Correct only while names are unique among running sessions. | Enforce the invariant in the coordinator; key by session label where a name can repeat (resumes). |
| `spent_usd()` = Σ `costs.json` | Running sessions are invisible. `agent_config` gives *each* new session `max_budget_usd = usd_left`, so N sessions can spend N × the remainder. | USD reservations: `Budget.reserve(label, expected_usd)` from the role's median cost (costs.json). `usd_left` subtracts open reservations. Per-session cap = min(cfg cap, (left − others' reservations)). |
| `exhausted()` | Checked before each start. | OK. Add "reserved USD" and "rate gate closed" reasons. |
| `sessions` / `max_sessions` | Counted at start. | OK. |
| `_wait_for_limit` (`orchestrator.py:321`) | Each session sleeps on its own. `MAX_LIMIT_WAITS` per session. `blocked` is global. | A subscription limit is account-wide: N sessions hit it together and N timers wake together. Use a shared `RateGate` (asyncio Event + reset time). Sessions wait on it, the coordinator starts nothing while it is closed, and waits count once. |
| `feedback()` advice | per session name | OK once evals are bound per session. With deadline extension, `minutes_left` must use the extended deadline. |

### 2.5 Scheduler assumptions (`scheduler.py`, `improve.py`)

The scheduler is stateless (it rebuilds arms from ledger + slice log), which is a strength:
it stays restartable. Assumptions of a single active slice:

1. `pick()` / `_in_time()` return **one** arm. Assigning N slots needs `assign()` with
   **virtual pulls**. Otherwise every free slot goes to the top-scoring arm, because a
   running session's outcome is not in the ledger yet (the "virtual loss" problem of
   parallel bandits/MCTS).
2. `_slice` attributes work by `exp_before` and the arm (`_close`, `improve.py:1062`):
   `evals`, `keeps`, `improved = best > best_before`. With two sessions on one arm
   (islands) rows and improvements are cross-credited, and `stale`/`idle` (`build_arms`
   `:803-810`) become wrong. **Fix**: a `session` column in ledger rows and
   `results.jsonl` records, with attribution by session.
3. `arm.hours` sums slice seconds. With concurrency that is *agent-hours*, so
   `target_hours` caps effort, not wall time. Keep the semantics and rename them in the
   docs and the `why`.
4. `slice_seconds()` = warm-up + median `eval_s` + wrap-up. It needs the expected queue
   wait added, or `_in_time` starts slices that cannot finish an evaluation.
5. `research_due` runs research *before* the arm's next slice while the loop waits. With
   islands, the arm needs a **paused** state: no new sessions until the research session
   ends; running ones continue.
6. `_native_gate` holds the native arm while any kernel arm is live and not plateaued.
   With free slots and a busy GPU that wastes capacity. Relax it to "slots free and no
   higher-scoring arm can use the slot".
7. `reintegrate` runs inline every `--integrate-every` keeps. Make it a background task,
   at most one at a time.
8. `_next_round` must run only when **no session is running** (barrier). A round
   re-plans from a re-profile, and running sessions would be on the old profile's targets.
9. `MAX_FAILED_SLICES` (3 in a row) ordered by start becomes ordered by completion, and
   needs a per-arm circuit breaker plus the global one.
10. `max_slices` should count started sessions.

### 2.6 Ownership and conflicting edits

* Engineer, systems and native sessions run with `BASE_TOOLS` and **no write guard**
  (`run_agent(writable=None)`). Only the prompt says "Keep everything inside your working
  directory" (`prompts.COMMON_RULES`). Research, dossier, refactor and planner sessions
  are guarded (`runner.write_guard`, exact files).
* `systems` works in `transforms/`, and `native` works in `transforms/native/`, a subtree
  of the systems agent's cwd. Concurrent systems and native sessions can edit each
  other's projects.
* A classic kernel session's cwd `targets/<id>/` contains `workers/<k>/`.
* Module overlap (#112, `integrate/owners.py`) is only known **after** patching, during
  integration (`touched`/`owns`). Nothing stops a kernel arm, a systems transform that
  CUDA-graphs the same layer and a native stage that subsumes it from working on the
  same modules at once. That is not a correctness problem (the integration's `replace`
  steps resolve alternatives), but it is a wasted-effort and moving-target problem.
* The integrity of *scores* does not depend on ownership: `.truth/` digests plus
  snapshot sha256 already defend against any session writing anywhere (`truth.py`
  docstring).

### 2.7 Dry run (`dryrun.py`)

`World.run_agent` is synchronous apart from a final `sleep(0)`. It advances one global
`SimClock` serially (`clock.advance(think)`, then `record_candidate` directly, bypassing
the tools and the GPU lock). Two simulated sessions cannot overlap in simulated time,
so the dry run cannot show what concurrency buys. **Fix**: virtual-time event loop (§3.9).

### 2.8 Observability

`status.render`, `watch.py` and `dashboard` read the ledger, events and agent logs, so
they are fine with N writers. The slices chart (`improve._draw_slices`) has one lane per
arm, and concurrent sessions on one arm would overlap. There is no per-agent state
(thinking / waiting for the GPU / on GPU) and no GPU-queue view.

---

## 3. Design

### 3.1 Principles

1. **One coordinator process per run.** It holds the in-memory Truth, the ledger lock,
   dedup, the GPU queue and the blackboard. N agent sessions are asyncio tasks, each
   driving its own Claude Code CLI subprocess. MCP tools run in the coordinator, so every
   GPU job and every shared write passes through it.
2. **The coordinator is Python, not an LLM.** The bandit scheduler is grounded,
   auditable and restartable from the ledger. LLM "leads" exist only where judgement pays
   (the planner at round boundaries, research at plateaus, the critic). Claude Agent SDK
   `agents=` subagents are not the unit of concurrency: they live inside one parent
   session's lifetime and context, cannot be resumed independently and blur budgets.
   Independent `query()` sessions remain the unit.
3. **Shared truth is append-only and coordinator-written.** Agents write only inside
   their own roots. Everything they share goes through tools (evaluations, board posts).
4. **N = 1 must behave exactly as today.** Every PR keeps `--agents 1` byte-identical in
   the dry run (the reproducibility tests already exist: `test_dry_run_is_reproducible`).

```
                        ┌──────────── coordinator process ─────────────┐
 Claude Code CLI ◄──────┤ Session task (label, role, arm, island)      │
 Claude Code CLI ◄──────┤ Session task ...         ──► MCP tools ──┐   │
 Claude Code CLI ◄──────┤ Session task ...                         │   │
                        │                                          ▼   │
                        │  Coordinator loop ◄── events ── GpuQueue (admission in gpu_lock)
                        │   (assign slots,                    │  flock ──► evaluator /
                        │    background integration,          │            worker subprocesses
                        │    rate gate, governor)             │
                        │  Blackboard (board.jsonl) ◄── posts │ Truth / ledger / events
                        └─────────────────────────────────────────────────┘
```

### 3.2 Coordinator: `Improver._loop` becomes event-driven

New module `kernel_agent/coordinator.py` (keeps `improve.py` as the policy and record
keeper). Sketch:

```python
async def _loop(self) -> str:
    async with asyncio.TaskGroup() as tg:          # failure isolation: see 3.8
        while True:
            interrupt.check(); self._keep_time()
            if reason := self._stop_reason():      # budget, max_slices, every arm stopped
                if not self.running: return reason
                self.draining = True               # start nothing; wait for the rest
            else:
                for job in self.assign(self.free_slots()):    # slices, research, dossiers
                    self.running[job.label] = tg.create_task(self._run(job))
                self._maybe_background_integration(tg)
            if not self.running and not self.background:
                if await self._next_round(self._pickable()): continue
                return self._why_none()
            await self._wake.wait(); self._wake.clear()       # set by task done, eval recorded,
                                                              # rate gate open, integration done
```

**Slot assignment** (`scheduler.assign(arms, slots, running, policy, gpu)`), in order:

1. Eligible arms: live (`stop is None`), not paused (research running), below their
   concurrency cap (kernel: its live islands, 1 without islands; systems 1; native 1),
   no running session with the same agent name, and `_in_time` with the queue wait added.
2. **Virtual pulls**: for each running session on an arm, add its expected evaluations to
   `arm.evals` (UCB `n`) and apply one step of `decay` as if it found nothing. Then
   re-rank and take the best arm for this slot. Repeat for each slot.
3. **GPU-aware role choice**: when the queue's expected wait exceeds `gpu_saturated_s`
   (§3.3) and a GPU-free job is due (dossier for a not-yet-started arm, research on a
   plateaued arm, next round's dossier pre-work), it takes the slot before another
   engineer. Engineers on a saturated GPU only lengthen the queue (§4).
4. Research and dossier jobs for an arm are exclusive per arm. While research runs, the
   arm is paused for new sessions.

**Roles and mix**:

| Role | Agent name | Needs GPU | Concurrency | Trigger |
|---|---|---|---|---|
| kernel engineer | `kernel-<id>` / `kernel-<id>-w<k>` | yes (evals) | ≤ islands per arm | score |
| systems engineer | `systems` | yes (e2e, long) | 1 | score |
| native engineer | `native` | yes (build off-GPU, e2e) | 1 | score; gate relaxed when slots free |
| researcher | `research-<id>` | no | 1 per arm | `plateau()` |
| dossier | `dossier-<id>` | no | any | arm's first session |
| critic | `critic` (no tools, cheap model) | no | many, short | queued full evals (§3.4) |
| planner | `planner` | no | 1 | round barrier |
| integrator | Python (`Orchestrator.integrate`) | yes (A/B, background) | 1 | `--integrate-every`, final |
| librarian | `librarian` | no | 1 | end |

Records: the slice record in `improve.json` gains `session` (label), `island` and
`queue_s` (time its evaluations waited). `improved`, `evals`, `keeps` and `failures` are
computed from the session's own ledger rows, not from `exp` ranges. `_recover` already
closes every `running` record. It needs no change beyond the per-session attribution.

### 3.3 GPU job queue

**Where**: inside `gpulock.gpu_lock()`, as an admission controller in front of the
existing `_acquire`. Every GPU entry point already goes through `gpu_lock()`:
`kernels/evaluate.run_evaluation:983`, `sweep:838`, `recheck:508`, `memcheck:317,393`,
`ncu:401`, `region:384`, `worker.call_worker:863`, `profiler:952`, `roofline:436,930`,
`suite:333`. So the queue reaches all of them with no signature changes:

* `gpuqueue.Job` (kind, priority class, estimated seconds, session label, target,
  submitted/started/ended, `exclusive=True`, optional `mem_gb`). It is set as a
  `contextvars.ContextVar` by the MCP tool (or the coordinator for background work)
  **before** `asyncio.to_thread`. `to_thread` copies the context, so the thread that
  calls `gpu_lock()` sees the tag. Untagged calls (CLI tools, tests) take class
  `default`.
* `gpulock._admit(job)`: a `threading.Condition` plus a heap per GPU pool. A thread
  waits until it is at the head *and* a GPU is free, then proceeds to `_acquire` (flock).
  On release it notifies the next. Re-entrant holds (`held[name]`) and children
  (`KERNEL_AGENT_LOCK_HELD`) skip admission exactly as they skip the lock today.
* The flock stays as the outer layer. Another kernel-agent process or a
  `kernel-agent bench` still excludes. In-process ordering is the queue's.

**Priorities** (class first, then SJF on the estimate, then aging):

| Class | Jobs | Why |
|---|---|---|
| P0 `deadline` | final integration A/B steps once the loop is in its reserve window | the run must finish within `--max-hours` |
| P1 `interactive` | `mode="quick"` checks, `run_on_gpu` debug (≤ 120 s), `verify_rewrite` | an agent is blocked, short |
| P2 `eval` | full `evaluate_candidate`, native stage evals | an agent is blocked |
| P3 `e2e` | `evaluate_e2e` (systems, native) | an agent is blocked, long (model load) |
| P4 `sweep` | `sweep_candidate` | long; one tool call times many configs |
| P5 `background` | re-integration A/B, recheck, memcheck, library seeding, captures, re-profile | no agent waiting |

* Estimates come from `scheduler.median_eval_s` per (kind, target), falling back to
  `EVAL_SECONDS`.
* Aging: the effective class improves one step per `age_s` (default 10 min) of waiting,
  so background integration still progresses on a busy GPU.
* Integration guarantee: when `Budget.final_reserve_s` grows past the share kept, P5
  integration jobs move to P2.
* Within a class: round-robin across sessions (fair share), so one sweep-heavy agent
  cannot monopolise.

**Session clock**: while a session's job waits in the queue, its `asyncio.timeout` is
pushed back by the wait (`Timeout.reschedule(when + waited)`; the `timer` lives in
`Orchestrator._agent`, registered in a `SessionClock` the queue updates). The same goes
for `Budget.deadlines[label]`, so `minutes_left` and the `stop` advice ignore queue time.
Always capped by `agent_seconds_left()`, so the run's budget still holds. This prevents
the "no evaluation in 2 slices → stopped" false positive (§2.3).

**CPU-only work off the queue**:

* Native projects: already prebuilt without a GPU (`tools._prebuilt`).
* **load_inline single-file candidates**: a new `kernels/prebuild.py` imports the
  candidate's extension build in a subprocess with `CUDA_VISIBLE_DEVICES=""` and
  `TORCH_CUDA_ARCH_LIST` from the toolchain, into the same `TORCH_EXTENSIONS_DIR`. The
  evaluator's compile is then a cache hit. Triton JIT stays in the lock (it compiles on
  first launch, usually seconds).
* Dedup, critic, digests and charts are CPU work that never enters the queue.

**Clean timing**:

1. One exclusive holder per GPU (unchanged).
2. CPU isolation for timed jobs. The evaluator and `e2e` subprocesses get
   `sched_setaffinity` to `--timing-cores` (default: the last 2 physical cores) and
   `nice -5`. Prebuilds and Claude Code CLIs get `nice 10` and `MAX_JOBS` capped to the
   other cores.
3. Stray GPU contexts. `telemetry.Monitor.processes()` is sampled at the start and end
   of every P2–P4 job. A foreign compute context, or an `integrity` reference slowdown
   confirmed by the clean re-time (already done in `_check_reference_timing`), marks the
   result `timing_dirty`. The job is re-queued once at its class head, and the first
   result is not recorded.
4. Agents' Bash GPU use. Add `--agent-gpu {shared,tool}`, default `tool` when `--agents`
   > 1: `agent_env` sets `CUDA_VISIBLE_DEVICES=""` for the CLI (Bash inherits it). A
   new MCP tool `run_on_gpu(script, args, timeout≤120)` runs the script as a P1 job and
   returns the output tail. The `_env_block` text changes accordingly. `shared` keeps
   today's behaviour.

**Batching quick checks** (later): queued P1 quick checks of the same target can run in
one evaluator process (capture loaded once). This needs a multi-candidate mode in
`kernels/evaluate.py`. Shared (non-exclusive) leases for correctness-only phases with
memory admission (`mem_gb` from the evaluator's peak-memory field) are a further step.
They need the evaluator to request an exclusive upgrade before timing (it already prints
`COMPILE_MARKER`, so a phase protocol over stderr/stdin is plausible). Not in the first
PRs.

**Events and records**: `gpu_job` events (`queued`/`start`/`done`, id, class, session,
`wait_s`, `hold_s`). The ledger and `results.jsonl` get `queue_s`, and `eval_s` keeps
meaning the hold. `scheduler.integration_estimate` keeps using hold times.

### 3.4 Blackboard and messaging

**Store**: `<run>/board.jsonl`, append-only, written **only by the coordinator**
(`kernel_agent/board.py`). It is advisory: never read for scoring. Entry:

```json
{"id": 41, "ts": ..., "author": "kernel-attn-w2#17" | "coordinator",
 "kind": "insight|warning|request|reply|claim|release|winner|integration|review|round",
 "to": null | "arm:native" | "role:kernel" | "session:<label>",
 "tags": {"targets": ["attn"], "module_class": "Qwen3Attention", "backend": "triton",
          "precision": "fp8_w8a8"},
 "text": "≤1500 chars", "refs": ["history/012_...py", "exp:233"], "reply_to": 37}
```

**Writes**:

* Agents use the MCP tool `post_note(kind, text, tags, to, refs)`. It is rate-limited per
  session (say 6) and deduplicated by text hash.
* The coordinator posts itself:
  * `winner` on every `keep` (target, snapshot, speedup, backend, precision, idea);
  * `integration` after each re-integration (accepted set, owners);
  * `round` at round start;
  * `claim`/`release` at session start and end (§3.6);
  * `review` for critic verdicts.

**Reads** (three paths, all bounded):

1. At session start: `improve.board_section(run, binding)` in the digest. It holds the
   last K entries relevant to the session's tags and its arm's mailbox (unread `to:
   arm:<id>` entries), and replaces nothing that already exists in digests.
2. **Piggyback**: every evaluation result gets `board: {new: n, items: [first lines]}`
   for entries since the session's cursor that match its subscriptions. The cursor is in
   memory per session. This is the same mechanism the budget advice uses, and the
   natural interrupt point of a session.
3. `read_board(since, kinds, tags)` for the full text.

**Subscriptions by role**:

| Role | Subscribes to |
|---|---|
| kernel | its target, the same `module_class`, the same backend/precision `insight`s |
| native | every `winner` (its building blocks become live rather than frozen at session start, cf. `native_engine.building_blocks`), `integration` |
| systems | `winner`, `integration`, `claim` |
| research | everything about its target, including the board history in `research.evidence` |

**Messages to arms**: sessions are short, so a message `to: arm:<id>` lands in the arm's
mailbox. The arm's next session sees it in its digest, and a running session on that arm
sees it piggybacked. Typical messages:

* native → kernel ("need a fused RMSNorm+QKV for stage X at these shapes"). The
  coordinator may turn this into a new kernel target via a planner-lite step (later).
* kernel → systems ("my kernel needs the static KV cache to be contiguous").

**Critic** (`kernel_agent/critic.py`):

* When `evaluate_candidate` (full) or `evaluate_e2e` is queued and the expected wait
  exceeds `critic_min_wait_s` (with `--critic auto`; always with `on`), it starts a
  critic task **concurrently with the queue wait**.
* Static checks first: AST near-duplicate of a failed snapshot (beyond `dedup`'s exact
  key), reference `forward` calls, precision outside the target's tier, no `build()`.
* Then a no-tools, single-turn `query()` with a cheap model (`--critic-model`, the same
  pattern as the librarian config). It sees the diff against `parent`, the idea's
  history (`ledger.ideas`) and the plan's do-not-try list.
* If it returns `reject` (high confidence) *before the job starts*, the job is withdrawn
  and the tool returns `status: "reviewed"`, the critique, and "evaluate again with
  `force=true` to override". This counts as no evaluation.
* If the job has already started, the verdict is only recorded (`review` column). Those
  are free labels for calibrating the critic. Additionally, 10 % of rejects are evaluated
  anyway (audit) to measure false negatives.
* The critic's precision and recall go in the report. It is disabled automatically when
  its precision falls below a floor.

### 3.5 Population / island search per target (`workers.py` → islands)

* An **island** is a persistent worker `k` of a target (`targets/<id>/workers/<k>/`, own
  `NOTES.md`, own lineage via `parent`). Its seed is `workers.seeds` (planner
  `alternatives`, else backend rotation). State lives in `improve.json` →
  `islands[target][k]`: seed, best snapshot, evals, last improvement, status
  (`live`/`culled`/`reseeded`).
* **Selection inside an arm**: the arm-level score decides *whether* the arm gets a
  slot. The island is chosen by its own UCB (rate per evaluation, exploration on few
  evaluations), skipping islands with a running session.
* **Migration**: every `--migrate-every M` evaluations of the target, each island's next
  digest gets an "Inspirations" section (OpenEvolve-style). It holds the global elite
  and a diversity elite: the best snapshot of another (backend, idea family, precision)
  cell (MAP-Elites cells from ledger `backend`, `idea`, spec precision). It may use either
  as `parent`. This replaces the one-shot `workers.reseeds` (`--reseed-workers` becomes
  "migration on").
* **Culling**: an island with best < `(1 − cull_gap)` × the target's best after
  ≥ `cull_after` evals and `stale` ≥ 2 sessions is reseeded. It gets a fresh direction
  (the next unused planner alternative or backend) and an elite as parent. Its old notes
  are archived.
* **Budget**: no fixed split (`workers.split`). Evaluations flow to islands that pay,
  through the island UCB. `--slice` stays per session.
* **Scheduler integration**: arm stop rules count across islands (`streak` over all the
  arm's measured rows), which is the same rule in evaluation units. `plateau` triggers
  one research session for the arm, and its `plan.md` is read by every island's next
  digest (already: `plan.md` is in `workers.SHARED`).
* The classic session and islands are mutually exclusive per target. With `--islands ≥ 2`
  every session of the target is an island session.

### 3.6 Ownership and locking

**File ownership** (enforced for every session, not only research). This extends
`runner.write_guard` with `writable_roots` (directory prefixes) alongside exact files:

| Session | Writable roots |
|---|---|
| kernel (classic) | `targets/<id>/` minus `workers/` |
| kernel island k | `targets/<id>/workers/<k>/` (symlinked shared files are not writable) |
| systems | `transforms/` minus `transforms/native/` |
| native | `transforms/native/` |
| research | `plan.md`, `pivot.json`, `research.md` of its target (as today) |
| dossier | `research.md` of its target (as today) |
| planner | `rounds/<n>/` |
| critic | nothing (no tools) |
| coordinator only | `run.json`, `improve.json`, `results.tsv`, `events.jsonl`, `costs.json`, `integration.json`, `board.jsonl`, `.truth/**` |

Bash is not covered by hooks (as today, `claude_files_guard` notes it). Scoring
integrity does not rely on it (truth digests). An optional post-session audit compares
the mtimes of other arms' roots and posts a `warning` to the board.

**Coordinator-side locks**:

* `workspace.update_json(path, fn)` (per-path `threading.Lock`, unique tmp names) for
  every RMW site listed in §2.2. `Truth._persist` uses it too, so the truth section and
  notes never overwrite each other.
* `history/` snapshot numbering under a per-directory lock with `O_EXCL`.
* Dashboard: one debounced refresher (`dashboard.Refresher`, a single thread, coalesces
  `refresh(target)` requests, at most once per 5 s).
* Coordinator pid lock (`<run>/.coordinator.lock`, flock) so two `improve` processes
  never share a run.

**Module claims (#112)**:

* At session start the coordinator posts a `claim` with the modules the arm works on.
  For a kernel target, the qualnames or instance groups from `spec.json` (and
  `projection.build` groups). For a native stage, its group (`native_engine.Stage`). For
  systems, the modules its plan names, and the agent may add claims with
  `post_note(kind="claim")` before writing a transform.
* Overlap is tested with `integrate.owners.inside`.
* `--overlap {allow,warn,avoid}`, default `warn`:
  * `warn` starts the session but adds a digest note naming the concurrent overlapping
    session and how the integration will treat alternatives (`replace` steps). A native
    stage that subsumes a live kernel target gets that target's `winner`s as building
    blocks.
  * `avoid` skips an overlapping arm for this slot.
* After an integration, `owners` from `integration.json` refine the claims
  (`touched`/`owns` are exact).

### 3.7 Budget- and rate-limit-aware concurrency

Effective concurrency `k = min(--agents, k_usd, k_rate, k_gpu, live eligible arms)`,
recomputed at every wake-up:

* **USD** (`k_usd`): reservations (§2.4). A session starts only if `usd_left −
  Σ reservations ≥ expected cost of its role`. The reservation is released at the
  session's end with the actual cost.
* **Time**: unchanged rules (`agent_seconds_left`, integration reserve, `slice_seconds` +
  expected queue wait). Draining starts when `agent_seconds_left` falls below the longest
  running session's remaining need. The final integration then has the GPU alone.
* **Rate limits** (`k_rate`, subscription):
  * `runner.run_agent` already sees `RateLimitEvent` (`auth.LimitWatch.see`). Forward
    `rate_limit_info.utilization`, `status` and `resets_at` to the coordinator.
  * AIMD: on `allowed_warning`, `k_rate -= 1` (min 1). After each quiet half hour with
    utilization < 0.6, `k_rate += 1`. On `rejected`, the `RateGate` closes until
    `resets_at`: no new sessions, and sessions stopped at the limit wait on the gate and
    are resumed with `auth.RESUME_PROMPT` (as today, but one shared wait).
  * Project the utilization at reset: `u + du/dt × (resets_at − now)`. If it exceeds
    0.9, lower `k_rate` pre-emptively.
* **GPU** (`k_gpu`, `--agents auto`): `1 + Z/S` from the measured think time `Z` (session
  time outside GPU jobs per evaluation) and hold time `S` (§4), plus one, capped by
  `--agents`.

### 3.8 Failure isolation

* Each session is a task in the coordinator's `TaskGroup`, wrapped so that an
  `Exception` closes its record (`failed`) and wakes the loop. It never cancels the
  group. `KeyboardInterrupt`/`CancelledError`/`interrupt.Interrupted` propagate, and the
  group cancels every session. Each `_close` records `interrupted`, the Agent SDK closes
  its CLI and `interrupt.run` kills the descendants.
* Circuit breakers:
  * Per arm: 3 failed sessions in a row stop the arm for the round (`stop = "failing"`).
  * Per role: e.g. every native session fails; the role is disabled for 30 min.
  * Global: `MAX_FAILED_SLICES` across all completions, meaning SDK/auth is broken, so
    stop.
* GPU jobs: per-job timeouts as today. A watchdog kills a job at timeout + grace (the
  process tree via `interrupt`'s `/proc` walk). A small health probe (matmul and sync)
  runs before the next job, and a failing probe pauses the queue and posts a
  `warning`.
* A crashed coordinator: `_recover` already closes `running` slice, research and dossier
  records. Add `islands` and board cursors (cursors are in memory; a restart re-sends
  the last K entries).

### 3.9 `--dry-run` with N agents

* **Virtual time**: replace `SimClock` with a `VirtualClock` that drives an asyncio loop
  whose `time()` is virtual. When no task is runnable, the loop jumps to the earliest
  timer (the pattern of `aiotools.VirtualClock`: patch `loop.time` and the selector
  timeout). `asyncio.sleep`, `asyncio.timeout` (session limits) and `Budget.elapsed_s`
  then all run in simulated seconds. Set `ledger.clock = VirtualClock.now`.
* **Simulated agents become concurrent coroutines**. `World.run_agent` awaits
  `clock.sleep(think)` per candidate (today `clock.advance(uniform(150, 330))`). It then
  submits the evaluation **through the real GPU queue** with a simulated executor (the
  job body is `await clock.sleep(eval_s)` while holding the admission slot, in virtual
  time) and records through `record_candidate` as today.
* Simulated tool calls go through the same per-session `SessionBinding`, so budget
  advice, board piggyback and critic paths are exercised.
* Determinism: draws are already seeded by (seed, arm, evaluation index). Ties in the
  queue break by job id, and the scheduler is deterministic, so a dry run with N agents
  is reproducible. `test_dry_run_is_reproducible` gets an `--agents 3` variant.
* Invariants checked in tests:
  * GPU jobs never overlap on one GPU;
  * no two running sessions share an agent name;
  * USD spent ≤ `max_usd` + one session's cap;
  * every record is closed after an interrupt;
  * rounds start only with nothing running;
  * the integration finishes inside the reserve.
* Metrics vs N from the simulation feed the report and the docs: evals/h, GPU busy share,
  queue wait p50/p95 and agent idle share. They are cross-checked with §4.
* The simulated think and eval distributions should be fitted to the measured ones (the
  data part of #174) instead of `uniform(150, 330)` and `uniform(25, 70)`.

### 3.10 Config / CLI

| Flag | Default | Meaning |
|---|---|---|
| `--agents N\|auto` | 1 | concurrent agent sessions (`auto`: governor §3.7, capped at 6) |
| `--roles LIST` | `kernel,systems,native,research,dossier` | roles the coordinator may start (`critic` with `--critic`) |
| `--role-max kernel=4,systems=1,native=1` | as shown | per-role caps |
| `--islands K\|auto` | 1 | islands per kernel target (alias and successor of `--seeds-per-target`) |
| `--migrate-every M` / `--cull-gap F` | 4 / 0.15 | island migration and culling |
| `--critic off\|auto\|on`, `--critic-model` | `auto` with `--agents > 1` | §3.4 |
| `--agent-gpu shared\|tool` | `tool` when N > 1 | §3.3 point 4 |
| `--timing-cores LIST` | auto | §3.3 clean timing |
| `--overlap allow\|warn\|avoid` | `warn` | §3.6 |
| `--parallel` | — | kept for the `optimize` kernels phase; in `improve` an alias of `--agents` with a deprecation note |
| `--role-model ROLE=MODEL` (repeatable), `--role-effort ROLE=LEVEL` | table in §3.12 | per-role model and effort; `inherit` = `--claude-model` |
| `--subagents off\|on` | `on` | in-session delegation (§3.12, §3.13) |
| `--stagger-starts` | on | start one session per role, then the rest after its first streamed message (cache reuse) |
| `--early-stop off\|on` | `on` | early discards in evaluator, sweeps and A/B (§3.12) |

The values go in `run.json` `config` and `improve.json` `config`
(`ImproveConfig.agents`, `.islands`, ...), and `improve()` overrides them on resume like
the existing ones (`improve.py:1650`).

### 3.11 Observability

* **Live state**: `improve.json` → `sessions` (running) and `gpu` (queue snapshot),
  updated on state changes. Each session's `state` is one of `starting`, `thinking`,
  `bash` (PreToolUse/PostToolUse hooks in `runner.run_agent` mark Bash/Write spans),
  `queued(<class>, waited s)`, `on_gpu(<kind>)`, `critic` or `waiting_rate_limit`.
  Fields: evals, best, USD so far and minutes.
* **`kernel-agent status`** (`status.render`):
  * an **Agents** table (label, role, arm/island, state, evals, best, $, min);
  * a **GPU** block (running job, queue by class with waits, busy share over the last
    hour, wait p50/p95);
  * the last 5 board entries.
* **`watch`** (`watch.state`, `watch.html`): per-session swimlanes (thinking / Bash /
  queued / GPU) plus a GPU lane, from `gpu_job` and `session_state` events, and a board
  panel.
* **Charts**: `improve._draw_slices` uses sub-lanes per island and session (one row per
  arm, stacked bars) and adds a GPU busy strip.
* **Report** (`improve.report_lines`): a "Concurrency" section.
  * GPU busy share and queue waits.
  * Agent-time split (thinking/Bash/queued/GPU), measured from the states above. This is
    the data #174 asks for, captured from now on.
  * Evals/h, and critic precision and recall.
  * Board activity, and how many `winner`s were reused by native or systems.

### 3.12 Efficiency: tokens, quota, GPU time

The goal is to do more per Opus token, per subscription window and per GPU-second. Each
item is measurable: `costs.json` gets per-session token usage, and the ledger gets
`queue_s`, `review` and `early` columns.

#### 3.12.1 Model mix per role

Today every session uses `cfg.claude_model` (default `claude-opus-5-5`, `config.py`) at
`effort="high"`, except the dossier (`DOSSIER_CONFIG`: effort low, 20 turns) and the
librarian (`librarian_model`, default `None` = same model, effort low). Proposal, as
`roles.py` defaults, all overridable with `--role-model` / `--role-effort`:

| Role | Model (alias) | Effort | Why |
|---|---|---|---|
| kernel engineer, native, systems | `inherit` (Opus 5.5) | high | the creative work; quality per evaluation is what the GPU budget buys |
| planner, research (plateau review) | `inherit` (Opus 5.5) | high | few sessions with high leverage (the targets, the next plan) |
| dossier | `sonnet` (Sonnet 5.5) | low | retrieval and summarisation from the local doc library (#177), web only for gaps |
| subagent `doc-lookup` | `sonnet` | low | API questions answered with doc citations |
| subagent `compile-triage` | `sonnet` | medium | reads a build or runtime error and the candidate, proposes the fix |
| subagent `profile-analyst` | `sonnet` | medium | reads the profile/ncu tables, returns the bottleneck and the top 3 actions |
| critic triage | `haiku` (Haiku 4.5) | (none) | static and short checks, single turn; escalates when uncertain |
| critic escalation | `sonnet` | low | only for triage-uncertain candidates |
| librarian | `sonnet` | low | distillation |

Notes:

* With `--auth subscription` the binding constraint is the usage windows, not USD. The
  `auth._LIMIT_TEXT` pattern already distinguishes "opus" and "sonnet" limits. Moving
  high-volume, low-judgement roles to Sonnet or Haiku keeps the Opus window for engineers.
* Opus 5.5's default effort is `medium`, so every role sets effort explicitly (`roles.py`).
* `costs.json` records model and effort per session. The report gives cost and quota per
  role, which shows whether a cheaper model hurt: compare the critic's precision and the
  dossier-to-first-keep rate against an `inherit` baseline run.

#### 3.12.2 Prompt-cache-friendly sessions

Facts that drive the design (Claude API prompt caching):

* Prefix match in the order tools → system → messages; any changed byte invalidates
  everything after it.
* The default TTL is 5 min and a read refreshes it.
* Cache reads cost about 0.1× of input (0.05× on Opus 5.5) and writes 1.25×.
* Caches are scoped per model and per workspace.
* An entry becomes readable only after the first response **starts streaming**: N
  identical requests fired at once all pay full price.

What breaks sharing today:

1. `prompts.engineer_prompt` puts `# Target <id>`, the spec, cases and workload *first*,
   then `COMMON_RULES`, `playbook.md`, the backend guides (`triton.md` 13 KB, `cuda.md`
   8 KB, `low_precision.md` 41 KB when reduced precisions apply) and `_env_block`. The
   large stable part therefore sits after the per-target part and is never shared across
   targets or islands. `systems_prompt` and `native_prompt` have the same shape.
2. `{evaluations}` and the target id are embedded mid-prompt. `Budget.prompt_note` ("you
   have about N min") and the program note are appended after the role prompt but
   *before* the digest. Each differs per session.
3. The tool list differs per role (and must be byte-identical within a role: same MCP
   server schema and order, sorted skill listing).
4. Unverified: the `claude_code` preset's environment section (cwd, date) sits before
   `append`. If it varies per session, nothing after it is shared across directories.
   **Measure first** (PR 1 records `cache_read_input_tokens`). If the cwd breaks it, use
   one common cwd per role (the run root) with absolute paths to the session's own
   directory, or a custom `system_prompt` string that puts our stable prefix first.

Design:

* `prompts.py` builds every role prompt as `stable_prefix(role, backends, precisions,
  toolchain)` + `target_block(...)` + `digest` + `volatile_tail` (budget minutes, program
  note, board). The stable prefix (rules, playbook, guides or the skill listing under
  #176, env, GPU block) is byte-identical for every session of a role on a run. It is
  sorted deterministically, with no timestamps or session ids.
* With #176, progressive-disclosure skills shrink the stable prefix to a skill listing.
  Skill bodies load on demand as tool results in `messages`, after the cached prefix.
* **Staggered starts** (`coordinator.fill_slots`): when several sessions of the same role
  start together, the first one starts alone. The others start after its first streamed
  `AssistantMessage` (or 30 s), so they read its cache instead of each writing one.
* `runner.run_agent` records `ResultMessage.usage` (`input_tokens`,
  `cache_read_input_tokens`, `cache_creation_input_tokens`, `output_tokens`) in
  `AgentResult` and `costs.json`. The report gives the cache hit rate per role, and a
  test asserts the stable prefix is identical across two targets of one role.

#### 3.12.3 In-session delegation (subagents)

Long sessions degrade as their context fills with doc excerpts, compiler logs and
profile tables. The native agent's sessions are already 3× longer (`NATIVE_FACTOR`).
Delegation keeps the main thread on the design. Subagents are SDK `AgentDefinition`s
passed in `ClaudeAgentOptions.agents` (built from `roles.py`, §3.13). The session gets the
`Agent` tool.

| Subagent | Tools | Input → output | Replaces |
|---|---|---|---|
| `doc-lookup` | `mcp__ka__doc_search`, `mcp__ka__doc_read` (#177), `Read`/`Grep` on `knowledge/`, `examples/` | an API question → answer with doc ids and versions (≤ 300 words) | the engineer paging through docs and guides in its own context |
| `compile-triage` | `Read`, `Grep`, doc tools | a candidate path + the `build_error`/`runtime_error` text → root cause + minimal patch (as text) | a long Bash and edit loop on compiler output |
| `profile-analyst` | `Read`, `Grep` | the path of a profile/ncu file → bottleneck vs roofline (`pct_of_sol`), top 3 actions | 24 KB of tables (`tools._text` cap) in the engineer's context |
| `reviewer` | `Read`, `Grep` | candidate + parent → the critic checklist on demand | (optional; the critic of §3.4 runs anyway) |

Rules:

* Subagents are **read-only**: no Write, Edit, Bash or GPU tools. So there is no ownership
  conflict, no GPU accounting and no hidden evaluations; the engineer decides and
  evaluates.
* To make delegation worth it, `evaluate_candidate(profile=…)` writes full tables to
  `<session dir>/profiles/<snapshot>.json` and returns a summary plus the path, instead of
  inlining up to 24 KB.
* Accounting: `runner.run_agent` already counts turns only for `parent_tool_use_id is
  None`. Split `tool_calls` into main and subagent calls likewise. Check that the
  session's `total_cost_usd` includes subagent usage; if not, sum it from the subagent
  `ResultMessage`s.
* Hooks (`write_guard`, `claude_files_guard`, web `Lookups`) are session-level and also
  apply to subagent tool calls. Verify this in a test.

#### 3.12.4 Critic before the GPU evaluation

§3.4 describes the flow (it runs concurrently with the queue wait, withdraws, has a
`force` override, audit sampling and calibration). The efficiency angle:

* Static checks are free and run first.
* The Haiku triage costs a fraction of a cent, compared with 60–300 s of GPU time per full
  evaluation (`EVAL_SECONDS`) plus the engineer's turn spent reading a failure.
* With an empty queue the critic still runs in `on` mode, overlapping the evaluation.
  A reject then only annotates the result ("the critic predicted this"), so the engineer
  learns without losing the measurement.

#### 3.12.5 Early termination of hopeless branches, without self-limiting

The principle (#176's "never self-limit"; `prompts.COMMON_RULES`; move-on rules retire an
arm for a *round*, #166): a branch may be cut early, but only with evidence. Cutting a
branch always hands its slot and budget to the next branch. It never ends a target, a
round or the run "because nothing more is possible".

| Branch | Rule | Freed capacity goes to |
|---|---|---|
| Idea | `ledger.ideas` verdict `refuted` (k ≥ 3 correct tries, none within noise of the best) → the result and the digest say "refuted: stop variations of `idea_id`"; the critic rejects further variants unless `force` | the next open idea in `## Open ideas` / `plan.md` |
| Island | culling (§3.5) | an elite reseed with a new direction |
| Session | no evaluation by `2 × WARMUP_SECONDS`, or 3 build errors on one idea in a row → mid-session advice via a `PostToolUse` hook's `additionalContext` (verify support) or the next tool result: "delegate to `compile-triage` or switch idea"; still nothing at 3 × warm-up → end the session (`budget_short`-like, not counted as stale) | the next slot assignment |
| Evaluation | **early discard** in `kernels/evaluate.py`: after the minimum timing rounds, if the candidate is slower than `best_so_far / (1 + noise)` with 95 % confidence (and slower than the reference), stop timing and record `discard` with `early: true` and the bound. Correctness is always checked in full; an early stop only skips more timing of a clear loser. | the queue |
| Sweep | racing (successive halving) in `kernels/sweep.run_sweep`: time all configs briefly, drop the slower half beyond the noise, re-time, repeat; the winner gets the full evaluation as today | the queue |
| Integration A/B | sequential test in `abtest.py`: stop after round r < `ab_rounds` when the CI already excludes `ab_min_gain` in either direction | background capacity |
| Arm | unchanged: move-on rules per round; plateau → research → new plan; precision pivot; native | other arms; next round revives what matters (`_revival`) |

The critic and the advice never reject on "unlikely to help" alone. Rejects are limited to
rule violations, duplicates and refuted ideas, so novel ideas always get measured.

#### 3.12.6 Evaluation batching

Candidates of different agents never share an evaluator process. Each candidate runs in
its own subprocess (`run_evaluation`), and cross-candidate state (CUDA context, monkey
patches, caches) would undermine the anti-gaming guarantees. Batching is therefore
layered:

1. **Lease batching** (GpuQueue). When a job reaches the head, queued jobs of the *same
   target and class* (up to B, total estimate ≤ `batch_s`) run back to back under one
   lease. This saves lock hand-offs, keeps GPU clocks and thermal state steady across
   comparable measurements, and reuses the parent's clean reference timing
   (`evaluate._CLEAN_REF_MS` is per parent process and capture, so it is shared already).
2. **Same-session multi-candidate tool**: `evaluate_candidates([...])` for one agent's
   variants of one idea. It works like `sweep_candidate` across files: one subprocess
   loads the capture once, checks each, times them interleaved with the reference, and
   records each as its own row. It counts as one evaluation per candidate (budget
   honesty) but costs one process start, one capture load and one reference warm-up.
3. **e2e batching**: `evaluate_e2e_batch([set1, set2, …])` for the systems and native
   agents. One worker process loads the model once (the dominant e2e cost) and applies,
   measures and undoes each set in-process. This reuses `integrate/undo.py` and the
   alternating timing of `worker e2e_ab`. Sets that cannot be undone in-process
   (`irreversible`) fall back to their own process, as in the integration.
4. **Quick checks**: queued `mode="quick"` checks of the same target, batched as in (1).

### 3.13 Roles as agent definitions, skills (#176) and doc tools (#177)

#176 and #177 are being implemented in parallel. This section fixes the interfaces the
coordinator relies on, so the three can land in any order.

**One registry, `kernel_agent/roles.py`**:

```python
@dataclass(frozen=True)
class RoleSpec:
    name: str                    # "kernel", "systems", "native", "research", "dossier",
                                 # "planner", "critic", "librarian", subagents: "doc-lookup", ...
    description: str             # AgentDefinition.description (when delegation applies)
    model: str = "inherit"       # alias or full id; "inherit" = cfg.claude_model
    effort: str | None = "high"
    max_turns: int | None = None
    builtin_tools: tuple[str, ...] = runner.BASE_TOOLS
    mcp_tools: tuple[str, ...] = ()       # tool_names(...), incl. doc_search/doc_read (#177)
    skills: Callable[[dict], list[str]] = lambda spec: []  # #176 skill names for this session
    subagents: tuple[str, ...] = ()       # RoleSpec names it may delegate to
    writable: Callable[[RunDir, "SessionBinding"], WritePolicy] | None = None  # §3.6
    needs_gpu: bool = True                # coordinator: GPU-free roles fill slots on a saturated GPU
    max_concurrent: int | None = None     # §3.2 caps
    program_section: str | None = None    # program.md role (program.ROLES)
```

* `roles.options_for(role, cfg, binding) -> dict` builds the per-session
  `ClaudeAgentOptions` fields: `model`, `effort`, `max_turns`, `allowed_tools` (builtin +
  MCP + `Agent` if subagents), `skills=role.skills(spec)`, `plugins=[{"type": "local",
  "path": skills.PLUGIN_DIR}]` (#176), `agents={n: roles.agent_definition(n) for n in
  role.subagents}`, `hooks` (write policy) and `setting_sources=[]` (unchanged).
* `roles.agent_definition(name) -> AgentDefinition(description, prompt, tools, model,
  skills, mcpServers=["ka"], maxTurns, effort)`. Subagents take the `ka` MCP server for
  the doc tools only (their `tools` list restricts them).
* `Orchestrator._agent(name, ...)` gets `role=` and calls `roles.options_for`. The
  scattered per-call settings move into the registry: `tool_names(...)` lists in
  `kernel_slice`/`systems_slice`/`native_slice`/`research`/`dossier`/`replan`,
  `DOSSIER_CONFIG`, `NATIVE_FACTOR` turns, and `librarian_*`. `program.ROLES` and
  `program.role_of` derive from it (adding `critic`, `dossier`, subagents).
* `src/kernel_agent/claude_code/agents/kernel-engineer.md` (the Claude Code
  `/optimize-model` path) and #176's `install-claude-code` are generated from the same
  RoleSpecs. One definition serves both the library sessions and the user's Claude Code.

**Contract with #176 (skills)**:

* `kernel_agent.skills.PLUGIN_DIR`: a local plugin (`skills/<name>/SKILL.md`, optionally
  `agents/`) shipped in the package.
* `skills.for_role(role, *, backends, precision, arch) -> list[str]`: plugin-qualified
  names. Examples: kernel/triton → `triton-kernels`, `profiling-roofline`,
  `anti-gaming`, plus `fp8-gemm-sm120` when the precision is FP8 on sm_120. native →
  `native-engines`, `cuda-graphs-pdl`, `cute-dsl`. Deterministic order (cache, §3.12.2).
* Isolation (#126): pass `plugins=[...]` plus `skills=[...]` with `setting_sources=[]`.
  The SDK only defaults `setting_sources` to `["user","project"]` when it is `None`
  (`subprocess_cli._apply_skills_defaults`, SDK 0.2.163), so the user's `~/.claude`
  skills stay out. A test asserts that the CLI command line carries
  `--setting-sources=` empty.
* Prompts shrink: `prompts.engineer_prompt` drops the inlined guides
  (`BACKEND_GUIDES`, `low_precision.md`) for the skill listing. That is the
  stable-prefix win of §3.12.2.

**Contract with #177 (doc library)**:

* MCP tools `doc_search(query, library=?)` and `doc_read(id)` in `agent/tools.py`
  `build_server`. They are in-process, read-only and thread-safe (the index is opened
  read-only once per process). They are **GPU-free and never enter the GPU queue**.
  Citations go to `research/sources.jsonl` with the session label (`web.record`).
* Every role gets them; the `doc-lookup` subagent is built around them.
* `kernel-agent docs build` runs before a run or at `doctor`. If the installed versions
  changed at run start, the coordinator rebuilds as a background CPU task, never during
  timed GPU work (CPU contention, §3.3).
* **Zero-LLM retrieval in digests**: `improve.kernel_digest`/`native_digest` call
  `docs.search(spec keywords: module class, backend, precision, arch)` and list the top 5
  doc ids under "Docs to read first". This is cheap, deterministic, and part of the
  volatile tail, so it does not disturb the cache.
* Dossier sessions start from `doc_search` (Sonnet, low effort) and use WebFetch only for
  what the library lacks.

**Coordinator use of roles**: slot assignment (§3.2) consults `RoleSpec.needs_gpu` and
`max_concurrent`. The governor's quota model (§3.7) uses `RoleSpec.model` to predict which
usage window a session draws on. The dry run's simulated agents are keyed by role name,
not by the agent name prefixes `dryrun.World.run_agent` parses today.

---

## 4. What N buys on one GPU (illustrative; replace with measured distributions)

This is a closed queueing model: N agents think for `Z` per evaluation, then hold the
GPU for `S` (exact MVA, one FCFS server plus a delay station; script in
`docs/research-scripts/multiagent-174/mva.py`).

Assumed `Z ≈ 300 s`: warm-up 240 s (`scheduler.WARMUP_SECONDS`) amortised over a 4-eval
slice plus about 240 s of writing per candidate (the dry run's `uniform(150, 330)`).
Assumed `S`: 60 s for a kernel eval (`EVAL_SECONDS`) and 120 s for e2e.

| N | kernel S=60 s: evals/h · GPU busy · wait | kernel S=90 s (compile in lock) | e2e S=120 s, Z=360 s |
|---|---|---|---|
| 1 | 10.0 · 17 % · 0 s | 9.2 · 23 % · 0 s | 7.5 · 25 % · 0 s |
| 2 | 19.5 · 32 % · 10 s | 17.5 · 44 % · 21 s | 14.1 · 47 % · 30 s |
| 3 | 28.2 · 47 % · 23 s | 24.6 · 62 % · 49 s | 19.6 · 65 % · 71 s |
| 4 | 36.1 · 60 % · 39 s | 30.3 · 76 % · 85 s | 23.8 · 79 % · 125 s |
| 6 | 48.5 · 81 % · 85 s | 37.1 · 93 % · 192 s | 28.4 · 95 % · 280 s |
| 8 | 55.8 · 93 % · 156 s | 39.5 · 99 % · 340 s | 29.8 · 99 % · 488 s |

Readings:

1. Throughput scales almost linearly up to the knee `N* ≈ 1 + Z/S`: about 6 for kernel
   work and about 4 for e2e-heavy work.
2. Moving compile off the lock (S 90 → 60 s) moves the knee from about 4 to about 6.
   That is why PR 5 matters as soon as N > 3.
3. Past the knee, queue waits grow without more throughput. Free slots should go to
   GPU-free roles, and session clocks must exclude queue waits.
4. A first real run should use `--agents 3` (systems + 2 kernel arms, or 1 arm with
   2 islands). Raise N only once the measured `Z`/`S` are in.
5. Tokens and USD scale with N, and evals per dollar stay about constant. The gain is
   wall-clock. Subscription limits then become the binding constraint (§3.7).

---

## 5. Staged PR plan (smallest useful first)

Each PR keeps `--agents 1` identical to today's behaviour (dry-run reproducibility
tests), adds tests and is independently useful.

### PR 0: concurrency safety prerequisites (no behaviour change)

* `workspace.py`: `update_json(path, fn)` with a per-path lock; unique tmp names in
  `write_json`.
* Port the run.json RMW sites to `update_json`:
  * `budget.note`;
  * `Orchestrator._mark`, `Orchestrator.resume`, the kernels-phase `finished`/`workers`
    lists (`orchestrator.py:182,199,387,609,740`);
  * `Truth._persist`;
  * `program.for_agent`/`install` (`program.py:151,171`);
  * `library.remember_*` (`library.py:657,667,894`);
  * `tools.check_harness`;
  * `suite.py:365`.
* `orchestrator.py`:
  * `integrate` runs its body via `asyncio.to_thread(self._integrate_sync, reuse)`;
  * `capture_targets` runs `_capture` via `to_thread`;
  * check every `async def` for sync `_worker` calls (`pivot`, `refactor`,
    `_native_stage` path).
* `ledger.py`: `rows()` under `_lock` (or tolerate a partial last line); new `session`
  column (and `queue_s`, empty for now); `record_kernel`/`record_e2e` take `session=`.
* `agent/tools.py`: `SessionBinding` (generalises `workers.Binding`: label, role,
  target, island, evaluations, cwd). `build_server` is used per session. Remove the
  reads of `budget.kernel_evals`/`transform_evals` from the tools. `record_candidate`
  and `record_e2e_result` stamp `session`.
* `orchestrator.kernel_slice`/`systems_slice`/`native_slice` build a per-session server;
  remove the writes to `budget.kernel_evals`/`transform_evals`.
* `_snapshot`: per-directory lock and `O_EXCL`.
* `dashboard.py`: `Refresher` (debounced, single thread). Tools call `refresher.request(target)`.
* Coordinator pid lock on the run.
* Tests: threads hammering `update_json` and `Truth.append` together with `budget.note`
  (truth digests survive); the integration does not block the loop (a concurrent task
  makes progress while a fake worker sleeps); ledger session attribution.
* **Risk**: low; a large mechanical diff. Order: `update_json` first, then the loop-blocking
  fixes.

### PR 1: role registry, per-role models, cache-friendly prompts (useful at N = 1)

* New `roles.py` (`RoleSpec`, `options_for`, `agent_definition`; §3.13).
  `Orchestrator._agent(role=...)` uses it. Per-call tool lists, `DOSSIER_CONFIG`,
  `NATIVE_FACTOR` turns and `librarian_*` move into the registry. `program.ROLES` and
  `role_of` derive from it.
* `config.py`: `role_models`, `role_efforts` (dicts) with the §3.12.1 defaults.
  `cli.py`: `--role-model`, `--role-effort`. `improve()` passes them through on resume.
* `prompts.py`: `stable_prefix` / `target_block` / `volatile_tail` ordering for the
  engineer, systems, native and research prompts. `Orchestrator._agent` appends the budget
  note, the program note and the digest after the stable part, in a fixed order.
* `runner.run_agent`: record `ResultMessage.usage` (cache read/creation, input, output
  tokens) and the model in `AgentResult` and `costs.json`. Report: cache hit rate and cost
  per role.
* Coordinate with #176: if it lands first, `skills.for_role` feeds `RoleSpec.skills`;
  otherwise the guides stay inlined but move into the stable prefix.
* Tests: identical stable prefix across two targets of one role (byte compare); role
  options per role; `--setting-sources` stays empty with skills/plugins on; costs.json
  carries cache fields (fake runner).
* **Risks**: prompt reordering changes agent behaviour slightly (compare first-keep rate
  on the dry run and one real run). Cheaper models on dossier and librarian may lower
  quality (configurable, measured).

### PR 2: GPU job queue

* New `gpuqueue.py`: `Job`, classes, the `job` ContextVar, `estimate(kind, target)`
  from ledger medians, wait/hold accounting and events.
* `gpulock.py`: `_admit(job)` before `_acquire`, notify on release; re-entrancy and
  child paths unchanged.
* `agent/tools.py`: tag each tool's job (quick, eval, e2e, sweep, verify_rewrite) before
  `to_thread`, then put `queue_s` in the record and the ledger. The coordinator tags
  background work: `Orchestrator._integration_call`, `_recheck_one`, `_memcheck`,
  `seed_library`, `_capture`, `reprofile`.
* `orchestrator._agent` registers its `asyncio.timeout` with a `SessionClock`. The queue
  extends it by the wait, and `Budget.deadlines` is extended the same way.
* `scheduler.slice_seconds` adds the expected queue wait.
* `status.render`: GPU block.
* Tests: priority order with fake jobs in threads; aging; per-session round-robin;
  re-entrant hold skips admission; a child with `KERNEL_AGENT_LOCK_HELD` skips; deadline
  extension; flock still excludes a second process (`test_gpulock.py` patterns).
* **Risk**: deadlock if a job holds the GPU and waits on admission for a nested call.
  Mitigated by the re-entrancy short-circuit before admission (same thread), and child
  processes skip admission.

### PR 3: concurrent coordinator, `--agents N`, virtual-time dry run

* New `coordinator.py` (event loop, `running`, `TaskGroup`, wake events, drain,
  background integration task, `RateGate`, circuit breakers). `Improver._loop` delegates
  to it when `agents > 1`, and the N = 1 path is unchanged.
* `scheduler.py`:
  * `assign(arms, slots, running, policy, gpu_wait)` with virtual pulls and per-arm caps;
  * `Arm.max_sessions`; `paused` while research runs;
  * relaxed `_native_gate` when slots are free;
  * `stale`/`idle`/`improved` from per-session attribution.
* `improve.py`:
  * slice records with `session`, `queue_s`;
  * `_close` by session rows;
  * `doing` becomes a set;
  * `_next_round` only when nothing is running;
  * `reintegrate` callable as a background task;
  * `max_slices` counts starts.
* `budget.py`: USD reservations; per-label keys; `exhausted()` includes reservations and
  the rate gate.
* `orchestrator._wait_for_limit` waits on the shared `RateGate`.
* `runner.run_agent` forwards `RateLimitEvent` info.
* `runner.write_guard` gets `writable_roots`. Every session gets its roots (§3.6), and
  systems excludes `transforms/native/`.
* `dryrun.py`: `VirtualClock` loop; `World.run_agent` becomes concurrent coroutines;
  evaluations go through the GPU queue with simulated executors.
* `cli.py`: `--agents`, `--role-max`, `--overlap`; `--parallel` alias.
* Staggered starts per role (`coordinator.fill_slots`, §3.12.2): GPU-free roles fill slots
  when the queue is saturated (`RoleSpec.needs_gpu`).
* Tests:
  * dry run with `--agents 3`: reproducible, invariants (§3.9);
  * the slot assignment spreads across arms (virtual pulls);
  * a round barrier waits for running sessions;
  * USD reservations stop new starts;
  * a failing arm's breaker;
  * a rate gate closes once and resumes all sessions.
* **Risks**:
  * Scheduler pathologies (all slots on one arm, or starvation of systems/native). Covered
    by virtual pulls, caps and tests.
  * Restartability with several `running` records (`_recover` handles it; test it).
  * Subtle N = 1 drift. Gate the new path on `agents > 1` until PR 4's data validates it.

### PR 4: observability

* Session state tracking: PreToolUse/PostToolUse hooks in `runner.run_agent`, and
  queue states from `gpuqueue`.
* `improve.json` `sessions`/`gpu`; `session_state`/`gpu_job` events.
* `status.render`: Agents table and GPU block; `watch.state` + `watch.html`: swimlanes;
  `improve._draw_slices`: sub-lanes and a GPU strip; `report_lines`: Concurrency section
  with the measured time split.
* **Risk**: low. Note that this PR produces the measurements §4 needs.

### PR 5: clean-timing hygiene

* `kernels/prebuild.py`: load_inline off the lock, with a shared `TORCH_EXTENSIONS_DIR`.
* Evaluator and worker subprocess CPU affinity and nice (`kernels/evaluate.run_evaluation`,
  `worker.call_worker`); nice and `MAX_JOBS` for prebuilds and agent CLIs
  (`runner.agent_env`).
* `telemetry.Monitor.processes` around timed jobs; `timing_dirty` triggers a re-queue.
* `--agent-gpu tool`: `agent_env` hides the GPU, a new `run_on_gpu` MCP tool, and the
  `prompts._env_block` text changes.
* Tests: prebuild cache hit; a dirty-timing re-run with a fake foreign process.
* **Risks**: agents lose the habit of quick Bash GPU runs (mitigated by `run_on_gpu`
  and `mode="quick"`). Affinity on machines with few cores (auto-disable below 8 cores).

### PR 6: subagents and doc tools in sessions

* `roles.py`: `doc-lookup`, `compile-triage`, `profile-analyst` (and optional
  `reviewer`) RoleSpecs as read-only `AgentDefinition`s. Engineer, native and systems
  roles list them in `subagents`, and `options_for` adds `agents=` and the `Agent` tool.
* `agent/tools.py`: wire #177's `doc_search`/`doc_read` into `build_server` (if #177 has
  not already). `evaluate_candidate(profile=…)` writes the full tables to
  `<session dir>/profiles/<snapshot>.json` and returns a summary plus the path.
* `runner.run_agent`: split `tool_calls` into main and subagent calls
  (`parent_tool_use_id`); check that subagent cost is included in the session total.
* `improve.kernel_digest`/`native_digest`: "Docs to read first" from `docs.search` (no
  LLM).
* `prompts.py`: one short section on when to delegate (doc questions, compiler output
  longer than a screen, profile tables).
* Tests: options carry the agent definitions with read-only tools; hooks apply to
  subagent tool calls; the digest lists doc ids (fixture corpus from #177).
* **Risk**: delegation overhead on small questions (the prompt says when it pays; measure
  turns and tokens per keep).

### PR 7: blackboard

* New `board.py` (append, query, subscriptions, cursors, dedup, rate limit).
* `agent/tools.py`: `post_note`, `read_board`; piggyback `board` in evaluation results.
* Coordinator posts: `winner` (on `ledger.KEEP` in `record_candidate`/`record_e2e`
  callers), `integration`, `round`, `claim`/`release`.
* Digests: `improve.board_section` in `kernel_digest`, `systems_digest` and
  `native_digest`. `research.evidence` includes the target's board history.
* `prompts.py`: one short section on the board (when to post; posts are advisory).
* `library` and the librarian prompt: distil board `insight`s into lessons.
* **Risks**: chatter and context bloat (bounded by caps, piggyback shows only first
  lines). Agents posting misleading insights (advisory only; never scored).

### PR 8: critic

* New `critic.py` (static checks plus a cheap single-turn session).
* `tools.evaluate_candidate`/`evaluate_e2e`: start the critic concurrently with the
  queued job, withdraw on reject, `force` override, record the verdict.
* Ledger `review` column; audit sampling; precision and recall in the report;
  `program.ROLES` adds `critic`.
* **Risks**: false rejects lose good ideas (override, audit, auto-off); cost (only when a
  wait exceeds the threshold).

### PR 9: islands

* `workers.py`: island state, island UCB, migration (inspirations), culling and reseeding.
  `--islands` (alias `--seeds-per-target`); `split` replaced by dynamic allocation.
* `scheduler.py`: arm cap = live islands; per-island choice.
* `improve.kernel_digest`: an "Inspirations" section.
* `workers.prompt_note`: island wording.
* Dry run: islands with distinct simulated ceilings per backend, so migration shows
  value.
* **Risk**: breadth versus depth. Kevin's result is that serial refinement beats parallel
  sampling at a fixed budget. Here the budget is not fixed: islands use idle GPU time.
  Keep the default `--islands 1` until the dry run and a real run show a gain per
  agent-hour.

### PR 10: early termination and batching in measurement

* `kernels/evaluate.py`: early discard after the minimum timing rounds (95 % bound vs
  `best_so_far` and the reference), correctness always in full; `early: true` in the
  record and the ledger. The tool passes `best_so_far`.
* `kernels/sweep.py`: racing (successive halving) of configs.
* `abtest.py`: sequential stop of an A/B once the CI excludes `ab_min_gain`;
  `integrate/reuse.py` keys include the stop rule (the evaluator schema).
* `ledger.ideas` verdict `refuted` feeds the critic and the evaluation result. Session
  advice via the next tool result (and a `PostToolUse` hook if `additionalContext` is
  supported).
* `gpuqueue.py`: lease batching of same-target, same-class jobs.
* `agent/tools.py`: `evaluate_candidates([...])` (same session, one subprocess, interleaved
  timing, one row each).
* `worker.py` + `integrate/undo.py`: `e2e_batch` (one model load, apply, measure, undo per
  set; irreversible sets in their own process); tool `evaluate_e2e_batch`.
* Tests:
  * an early discard never flips a winner (synthetic timings at the noise edge);
  * racing keeps the best config on fixtures;
  * the sequential A/B matches the fixed-round verdict on recorded A/B data
    (`tests/ab_fake.py`);
  * e2e batch results equal separate runs within noise (`toy_decoder`).
* **Risk**: early stops bias measurements if the bound is wrong. Keep them conservative,
  log the bound, and make them switchable with `--early-stop off`.

### PR 11: governor and asynchronous evaluations (optional)

* `--agents auto`: `k_gpu` from measured Z/S, AIMD on rate-limit utilization.
* Optional `submit_evaluation` / `evaluation_result` tools, so one agent can write its
  next candidate while the previous one is queued. This needs new advice semantics.
* Optional shared correctness leases with memory admission and quick-check batching.

---

## 6. Risks and open questions

1. **Timing contamination** is the main technical risk. Mitigations: an exclusive hold,
   CPU isolation, stray-context detection, `run_on_gpu`, the existing reference re-time.
   It should be validated by A/A runs of a fixed candidate with N = 1 vs N = 4
   background load: the speedup spread must not widen.
2. **Cost and limits**: N× token burn. Subscription windows are reached N× sooner, and
   all sessions stall together. The governor, the shared rate gate and the USD
   reservations are mandatory before `--agents > 2` is advertised.
3. **MCP tool timeout** of Claude Code for long blocking tool calls (queue + e2e). Verify
   the default for SDK servers and set it in `SESSION_ENV` if needed. Otherwise the
   async evaluation API (PR 11) becomes necessary sooner.
4. **Memory on the host**: N Claude Code CLIs (a few hundred MB each), nvcc jobs and the
   evaluator, on a desktop with one GPU. The governor needs a host-memory term.
5. **Scheduler correctness** with virtual pulls and session attribution. It is
   stateless-from-ledger today; keep it that way (claims, islands and board cursors live
   in `improve.json` and the board, so a restart rebuilds them).
6. **Moving targets**: a transform or native stage changing the modules a running kernel
   arm optimises. Claims and `warn` notes handle the effort. The integration's `replace`
   steps handle correctness.
7. **Do LLM roles pay?** The critic and the board must earn their place in the report
   (critic precision, `winner` reuse). Both can be disabled per flag.
8. **Rounds**: the barrier idles agents while the re-profile and re-plan run (about
   10 min). Possible later: dossiers for likely next targets during the barrier.

9. **Cache assumptions**: the shared-prefix win depends on the `claude_code` preset not
   putting per-session content (cwd, date) before our appended prompt. PR 1 measures
   `cache_read_input_tokens` before committing to a common-cwd or custom-system-prompt
   change.
10. **Parallel work on #176/#177**: they touch the same files (`prompts.py`,
    `agent/tools.py`, `runner.py`). Land `roles.py` (PR 1) with the interface stubs
    (`skills.for_role`, `skills.PLUGIN_DIR`, the `doc_search`/`doc_read` tool names)
    first, or agree on them in the issues, so each PR only fills its side.
11. **Cheaper models in judgement roles**: a Haiku critic that rejects good candidates, or
    a Sonnet dossier that misses the key API, costs more than it saves. Every downgrade is
    a config default with a measured metric (critic precision, dossier-to-first-keep), and
    `inherit` is one flag away.

---

## 7. Follow-up issues to open (one per PR)

1. Concurrency safety prerequisites: non-blocking integration and capture, `update_json`,
   per-session tool bindings, ledger `session` column (PR 0).
2. Role registry: per-role models and effort, cache-friendly prompt order, token and cache
   accounting (PR 1; coordinates with #176).
3. GPU job queue: priority admission in `gpu_lock`, session clocks exclude queue waits
   (PR 2).
4. `--agents N`: event-driven coordinator, slot assignment with virtual pulls, staggered
   starts, USD reservations, shared rate gate, virtual-time dry run (PR 3).
5. Per-agent observability: states, swimlanes, GPU queue, concurrency report and the
   measured time split (PR 4).
6. Clean timing under load: off-lock load_inline builds, CPU isolation, dirty-timing
   re-runs, `run_on_gpu` (PR 5).
7. In-session subagents (doc lookup, compile triage, profile analysis) with #177's
   `doc_search`/`doc_read`; profile tables to files; "docs to read first" in digests
   (PR 6).
8. Blackboard: `board.jsonl`, `post_note`/`read_board`, piggyback, live winners for
   native and systems (PR 7).
9. Critic during the queue wait, with Haiku triage, Sonnet escalation, audit and
   calibration (PR 8).
10. Islands per target: migration, culling, island UCB (PR 9).
11. Early termination in measurement (evaluator early discard, sweep racing, sequential
    A/B) and batching (lease batching, `evaluate_candidates`, `evaluate_e2e_batch`)
    (PR 10).
12. Governor `--agents auto` (GPU knee, quota per model window) and asynchronous
    evaluations (PR 11).
