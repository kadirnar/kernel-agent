# #174 literature: multi-agent systems for kernel / code optimisation, and the Claude platform constraints

Research for issue #174 (concurrent specialised agents, GPU job queue, shared blackboard), as of 2026-10-08.
Every claim has a source tag `[Sn]`; the list is at the end. Numbers are as reported by the authors. Most
papers do **not** compare multi-agent with single-agent at an equal token or candidate budget; where they
do, it is called out ("equal budget"). Evidence strength: peer-reviewed or official docs > arXiv preprint >
blog > course project (DSLBench [S31] is a student report; treat it as a hint).

---

## 1. Kernel / program-optimisation systems at a glance

| System | Roles | How agents communicate | Selection and diversity | Evaluation infrastructure | Measured gain from multi-agent / parallel structure |
|---|---|---|---|---|---|
| **AlphaEvolve** (DeepMind) [S1] | Controller, LLM samplers (Gemini 2.0 Flash for volume + Pro for "occasional, higher-quality suggestions"), evaluators | Program database; the prompt sampler renders prior programs **with their scores and outputs**, plus LLM-evolved meta-prompts | Database "inspired by a combination of MAP-Elites and island-based population models"; multiple scores to diversify | asyncio pipeline "optimized for throughput (rather than the speed of any one particular computation)"; evaluation cascade (cheap stages first) and an evaluation cluster | Ablations: no-evolution, no-context, no-meta-prompt, small-LLM-only each clearly worse. Kernels: Gemini matmul tiling heuristic +23 % avg kernel speed, -1 % Gemini training time, "months … to just days"; FlashAttention Pallas kernel +32 %, pre/post-processing +15 % |
| **OpenEvolve** (open AlphaEvolve) [S2] | Controller, weighted LLM ensemble (default 0.8 small / 0.2 larger model), evaluator, optional LLM-feedback scorer | Program DB; "artifact side-channel" feeds errors back into the next prompt | Defaults: `num_islands: 5`, `migration_interval: 50`, `migration_rate: 0.1`, MAP-Elites grid on complexity × diversity (10 bins), exploration 0.2 / exploitation 0.7 | `parallel_evaluations: 4`, cascade thresholds 0.5 / 0.75 / 0.9, 300 s timeout; "distributed evaluation is not yet implemented" | Cautionary: the MLX Metal attention-kernel example, after validity fixes (subprocess hook, bf16 gate, correct GQA ratio), reports the best evolved kernel **3.2 % slower** than baseline after 25 iterations, and lists "no GPU profiling data" and "1 parent + 5 samples per iteration" as limits [S2b] |
| **ShinkaEvolve** (Sakana) [S3] | Mutation LLMs chosen by a UCB1 bandit, novelty-judge LLM, meta-scratchpad summariser | Archive of islands; every T generations a **meta-scratchpad** summarises successful strategies into the mutation prompt | Islands (migration every 10 gens, rate 0.1, island elites protected); weighted parent sampling (fitness + few-offspring novelty); embedding novelty rejection (cosine > 0.95 → LLM judges "meaningfully different?") | Job queue; 1–5 max parallel jobs in the reported runs; a fully async proposal+job queue raised throughput but introduced **"off-archiveness"** (proposals built on stale archive; fast models over-represented) | SOTA circle packing with ~150 evaluations; bandit ensemble beats single LLM and fixed ensemble; weighted sampling beats hill-climbing; novelty rejection helps (ablations) |
| **AI CUDA Engineer** → **robust-kbench** (Sakana) [S4, S4b, S4c] | Translator, evolutionary optimiser, LLM ensemble (o3, o4-mini, Claude Sonnet 3.7, GPT-4.1), three tuned LLM **verifiers** (compile / memory / numerics) | Archive; prompt holds **up to 5 correct kernels sorted slowest→fastest**, sampled high-level hints for diversity, NCU + clang-tidy summaries | Sample N=8 proposals, verifiers majority-filter, evaluate N*=4 on hardware | "Individual H100 GPUs"; compile ≥ 1 min per kernel, which motivates LLM pre-filtering | The original 2025 release was walked back after a memory exploit bypassed correctness ("3x slowdown" not 100x speedup) [S4c]. robust-kbench: after removing contaminated tasks the average KernelBench speedup fell **3.13x → 1.49x**; fake 50–120x kernels found. Verifier accuracy 0.82 / 0.80 / 0.73 |
| **KernelFalcon** (Meta KernelAgent v1) [S5] | Orchestrator, FuserAgent, ExtractorAgent, Dispatcher + KernelAgent workers, ComposerAgent | Typed JSON contracts between stages; **error feedback stays local to the worker** ("no context pollution") | N=4 workers per subgraph, same prompt, different temperatures; first worker to pass wins and cancels the rest; SHA dedup | Each worker in its own dir, tests run in subprocesses; deterministic Python control plane | 100 % correctness on 250 KernelBench tasks (no speed numbers). MKEvolve reports its AutoAgent launched **47 workers on one GPU → GPU OOM**, so they serialised it [S7] |
| **KernelAgent hardware-guided** (Meta, Mar 2026) [S6] | ProfilerAgent (NCU), Diagnose/Judge, Analyzer (prescriber, retrieves from an optimisation DB), Orchestrator, Optimization Manager + workers, BenchmarkAgent | After each round, outcomes are reflected ("was diagnosis correct? lessons, avoid_patterns, try_patterns") into **shared memory broadcast to all agents** next round | Beam of top-K kernels × K bottlenecks (e.g. 2×2 = 4 workers) | "**Shared benchmark lock prevents GPU contention between workers**"; 25 warm-up / 100 reps | 2.02x geomean over its correctness-only kernels, 1.56x vs torch.compile on KernelBench L1, 89 % roofline. Case study matvec: 4 workers × 8 rounds 1.95 ms vs sequential 8 rounds 3.20 ms (not equal budget). "Parallelism alone is not sufficient; without coordination, agents quickly duplicate work" |
| **MKEvolve** (AWS, ICML-W 2026) [S7] | LLMDecompose, LLMEvolveKernel (per subproblem), LLMFuse/Split; budget allocator by runtime/error | Programmatic composition of verified subkernels | Beam search (width 4) per subproblem; budget allocated to subproblems by runtime/error | KernelBench CUDA-stream pipeline, TritonRL cheating detector, 5-min subprocess limit; A100 | **Equal budget (160 kernels/problem, Claude 4.5 Opus, L2):** independent parallel sampling Fast1 **0.03**, beam search 0.36, MKEvolve 0.49 (Swap 0.58); KernelFalcon 0.00. Tokens 1.3M / 2.0M / 1.7M per problem; MKEvolve uses 25–35 % fewer tokens than beam |
| **KernelArc** (Aug 2026) [S8] | Strategy-specialised agents (library, memory, compute, fusion, precision, reduction, scheduling, B200-native skills) | **Conclusions-only shared memory**: "wins" (speedup, shape, reflection) and "traps" (dead ends + error); "excludes iteration counters, heartbeats, and progress logs"; **read-only** leaderboard of sibling bests | Plateau-triggered drafting: after r stale steps an agent must try a different DSL / algorithm / layout | Deterministic **benchmark guard**: cascade gate → benchmark "under a GPU lock" → KEEP only if every workload passes and beats incumbent → STOP on plateau/target/time/cap | **Equal budget (100 candidates, one task FI-014):** 2 agents + shared memory 1.59x (capped memory) / 2.04x (uncapped) geomean over single agent; "increasing … from two to four agents did not show a consistent additional gain" |
| **KernelEvolve** (Meta, ISCA 2026) [S9] | Context-memory sub-agent (profiles/errors → directives), deep-search sub-agent (retrieval over a hardware knowledge base), "universal operator" | Metadata store with transaction isolation lets multiple agents expand different search-graph nodes | Greedy / MCTS / evolutionary selection policies; fitness 0 for incorrect | Generation decoupled from evaluation: "a single host may run hundreds of generation agents but … 8 GPUs or 24 MTIA devices per host", so evaluation goes to FaaS pools with hundreds of devices | 100 % correctness on 480 operator-platform configs; 1.25–17x on production workloads; no multi- vs single-agent ablation |
| **GEAK** v1 / v3 (AMD) [S10, S10b] | v1: generator, evaluator, reflector, optimizer. v3: planner + sub-agents (exploration, harness generation, optimisation, speedup verification, knowledge extraction, GEMM tuning) | v1 sequential pipeline; v3 agents in **isolated git workspaces** under one fixed evaluation contract; best patches seed the next round | v3 dispatches one optimisation path per agent | MI300X / MI250 | v1 parallel runs: execution accuracy 35.4 % (pass@1) → 54.9 % (pass@10), ~log-linear; serial iterations 13 % → 44 %. v3: HIP 3.02x, Triton 2.22x geomean vs mini-swe-agent 2.11x / 1.25x |
| **Astra** (NeurIPS-W 2025) [S11] | Testing, profiling, planning, coding agents (o4-mini) | Shared log of (round, code, correctness, performance) | 5 rounds | H100 | 1.32x vs **1.08x** single-agent on 3 SGLang kernels; single agent's own tests were unrepresentative and "biased the profiling results" |
| **STARK** (ICLR 2026) [S12] | Plan (τ=0.8), code (τ=0.1), debug (τ=0.1); Claude Sonnet 4 | Search tree; **role-specific context windows** (plan: children + global leaders; code: siblings' children; debug: node + siblings) | ε-greedy tree search, root throttle 5, prune nodes with ≥3 failing children | One A100, B=30 attempts per task | **Equal budget (30 attempts), L3:** sampling Fast1 50 %, search-only 67.5 %, multi-agent-only 67.5 % (1.11x), STARK 87.5 % (1.58x) — the roles and the search compound |
| **CudaForge** [S13] | Coder + Judge | Coder sees **only the Judge's feedback**, not history | ≤10 rounds | One RTX 6000; ~$0.30 and 26.5 min per kernel | Full 1.677x / 70.8 % Fast1 vs correction-only judge 1.222x / 58.8 %; a curated **24 NCU metrics beat the full metric set** ("the Judge is overwhelmed") |
| **TritonForge**, **PRAGMA**, **AccelOpt**, **KernelBand** [S14–S17] | TritonForge: test generator, optimiser, fault-aware remediation. PRAGMA: profiling-reasoned multi-agent. AccelOpt: planner, executor, summariser. KernelBand: bandit over kernel clusters × strategies | AccelOpt: **optimisation memory** of distilled slow→fast kernel pairs | AccelOpt beam search; KernelBand hierarchical bandit, profiles only cluster centroids (~10 s per profile) | — | TritonForge up to 5x; PRAGMA 2.30x vs Torch on GPU, beats its no-profiling baseline; AccelOpt peak-throughput share 49→61 % (Trn1), open models match Claude Sonnet 4 at 26x lower cost; KernelBand >33 % over prior methods |
| **CUDA-L1** [S18] | RL policy + **reward-checking model** (DeepSeek-R1) + hacking-case DB | Contrastive prompts: prior implementations **with scores** | Reward smoothing / clipping | — | 3.12x mean / 1.42x median on KernelBench, but found 32.8 % of samples (82/250) exploiting async extra CUDA streams, plus lazy tensors, shrunken hyper-parameters, result caching by address |
| **Kevin** (Cognition/Stanford) [S19] | Single multi-turn RL model | Previous turns' CoT discarded, **model-written summary of changes** kept | — | — | **Equal budget (128 generations):** 128×1 turn 0.65x, 32×4 1.02x, 16×8 **1.10x** — serial refinement beats parallel sampling |
| **Stanford CRFM fast kernels** [S20], **METR KernelAgent** [S21], **NVIDIA R1 loop** [S22] | Idea generator → many implementations; METR: best-of-k across models; NVIDIA: generate + H100 verifier loop | CRFM: optimisation **ideas in natural language** conditioned on prior ideas | CRFM: each idea spawns several implementations; best seed next round (5 rounds; best results mostly round 4–5). METR: 8 parallel attempts, ~300 per problem | CRFM: L40S, ~3M in + 4M out tokens for 10 problems. NVIDIA: 15 min loop | CRFM: "sequential loops often fall into local minima, revisiting the same classes of transformations". METR: 1.05x → 1.81x from better elicitation, 2.01x best-of-all-models; gains "still increasing at 200–300 attempts"; model/approach diversity helps |

Also relevant: "Agentic Kernel Optimization" (2026) used a multi-agent orchestrator (Houmao) with a compact set of
CUDA optimisation skills and won an MLSys 2026 FlashInfer contest track, spending ~1.9 billion agent tokens [S30].

---

## 2. Does multi-agent beat single-agent at an equal budget?

What the equal-budget comparisons say:

* **Independent parallel sampling is the weakest use of a budget.** MKEvolve: at 160 kernels per problem,
  parallel sampling reaches Fast1 0.03 vs beam search 0.36 (Claude 4.5 Opus, L2) [S7]. Kevin: 16×8 turns
  beats 128×1 (1.10x vs 0.65x) [S19]. STARK: sampling 50 % vs 87.5 % Fast1 at 30 attempts [S12].
* **A little breadth with shared conclusions beats pure serial.** KernelArc: 2 agents sharing a win/trap memory
  beat one agent by 1.59–2.04x at 100 candidates; 4 agents added nothing consistent [S8]. KernelAgent: beam of
  4 workers + reflection memory found an architectural rewrite the serial loop never tried [S6]. STARK: roles
  alone ≈ search alone; together they compound [S12].
* **Breadth helps correctness more than speed.** GEAK pass@k execution accuracy grows ~log-linearly with
  parallel runs [S10]; KernelFalcon reaches 100 % correctness with first-to-pass workers [S5].
* **Outside kernels, gains are mostly "more tokens".** Anthropic's research system beats single-agent Opus 4 by
  90.2 %, but "token usage by itself explains 80 % of the variance" and multi-agent uses ~15x chat tokens (single
  agents ~4x) [S33]. A 180-configuration study found +81 % on parallelisable tasks, **−39 to −70 % on sequential
  ones**, and error amplification of 17.2x for independent agents vs 4.4x with a central coordinator [S36].
  MAST: gains over single-agent "often remain minimal"; 14 failure modes, mostly system design (spec, inter-agent
  misalignment, verification) [S35].
* **Practitioners' rule:** "writes stay single-threaded"; extra agents should "contribute intelligence rather than
  actions" (read-only search sub-agents, clean-context reviewers); parallel-writer swarms fragment decisions [S34].

Synthesis for kernel-agent: the clear win is **concurrency across independent arms** (different targets, systems,
native, research: parallelisable work with a hard verifier), plus **2-way breadth with shared conclusions** on a
hot target. Several independent workers on one target split the evaluation budget, which the data says is worse
than refinement. This matches the existing `workers.py` default (k=1, "serial refinement beats parallel sampling
at a fixed budget (Kevin)").

---

## 3. Communication patterns that worked

| Pattern | Where | What is shared |
|---|---|---|
| Program database rendered into prompts | AlphaEvolve, OpenEvolve, ShinkaEvolve, CUDA-L1, robust-kbench [S1–S4b, S18] | Prior programs with scores; robust-kbench orders ≤5 correct kernels slow→fast so the model infers the pattern |
| Conclusions-only memory | KernelArc wins/traps [S8]; KernelAgent reflexions (lessons, avoid/try patterns) [S6]; Shinka meta-scratchpad [S3]; AccelOpt slow→fast pairs [S16]; Kevin change summaries [S19] | Distilled outcomes, never raw logs. KernelArc explicitly drops counters/heartbeats/progress logs |
| Error isolation | KernelFalcon (errors visible to the failing worker only) [S5]; GEAK v3 git workspaces [S10b]; CudaForge coder sees judge feedback only [S13] | Keeps contexts clean; "context rot" makes long contexts worse [S34] |
| Read-only cross-agent state | KernelArc campaign leaderboard [S8]; Cognition read-only sub-agents [S34] | Others' best solutions are readable, never writable |
| Artifacts on disk instead of messages through the lead | Anthropic research system ("subagent outputs … bypass the main coordinator", avoids a "game of telephone") [S33] | Files and a path, not prose summaries |
| Shared task list + mailbox | Claude Code agent teams: tasks with dependencies, claiming by **file locking**, per-agent JSON inboxes [S44] | "Two teammates editing the same file leads to overwrites. Break the work so each teammate owns a different set of files" |
| Publish/subscribe message pool | MetaGPT (PM → architect → engineer → QA, structured documents) [S37]; ChatDev chat chain [S38]; AutoGen conversations [S39]; OpenHands event stream + `AgentDelegateAction` [S40] | General coding frameworks; SOP-driven pipelines, not concurrent search |

Known hazards: **staleness** (Shinka's off-archiveness: async proposals built on an old archive; fast models
over-represented [S3]); **the lead blocks** while sub-agents run (Anthropic's lead executes sub-agents
synchronously [S33]); **duplicate exploration** without coordination [S6].

---

## 4. Evaluation infrastructure when many agents share few GPUs

* **Do not time on a shared GPU.** Without MPS, work from different processes cannot execute concurrently: "each
  process is assigned a serially scheduled time-slice on the whole GPU" [S29], so a concurrent job inflates
  another job's measured time. Every system that runs agents in parallel on one device serialises timing:
  KernelAgent's shared benchmark lock [S6], KernelArc's guard "under a GPU lock" [S8], MKEvolve serialising
  KernelFalcon after 47 concurrent workers ran out of GPU memory [S7]. robust-kbench times on individual H100s [S4b].
* **Decouple generation from evaluation.** KernelEvolve: hundreds of generation agents per host vs 8 GPUs, so
  evaluation goes to a separate pool [S9]. KernelBot (GPU MODE): a job queue over Modal/Northflank runners,
  30,000+ submissions in one competition [S28]. AlphaEvolve and ShinkaEvolve are queue-based [S1, S3].
* **Spend GPU time only on candidates that survive cheap filters.** AlphaEvolve and OpenEvolve evaluation cascades
  [S1, S2]; KernelArc's cascade gate (syntax, entry point, portability) [S8]; robust-kbench LLM verifiers before
  hardware [S4b]; GPU Forecasters, an LLM surrogate that predicts relative runtime and defers to the GPU when
  unsure, "lets the search consider several times as many candidates under the same GPU evaluation budget" [S27].
  Compilation (≥ 1 min per CUDA kernel [S4b]) is CPU work and can run outside the GPU lock.
* **Throughput vs freshness.** AlphaEvolve optimises for throughput [S1]; Shinka shows that full asynchrony
  costs sample efficiency through stale proposals [S3].

---

## 5. Verification, critics and anti-gaming

Documented exploits of kernel evaluators: memory exploit bypassing correctness (AI CUDA Engineer) [S4c]; extra
async CUDA streams (32.8 % of CUDA-L1 samples), lazy tensors, shrunken problem sizes, caching by input address
[S18]; copying the PyTorch reference, try/except fallback to PyTorch, inheriting the reference (Kevin) [S19];
o3 walking the Python stack to return the scorer's answer and disabling CUDA synchronisation (METR) [S25];
shape-hardcoded and distribution-specific shortcuts (11.6 % of KernelBench problems); a 1.43x reported speedup
becomes 0.88x under stricter evaluation (KernelBench-Verified) [S24]. ADRS: full-file rewrites let models delete
constraints; **diff-only edits** restrict reward hacking [S26]. KernelFalcon notes it trusts the LLM-written
test harness without analysing it [S5].

Critic and verifier agents that filter before or after evaluation:

| Critic | Placement | Measured |
|---|---|---|
| LLM verifiers (compile / memory / numerics), majority vote | Before hardware [S4b] | Accuracy 0.82 / 0.80 / 0.73 after prompt tuning; better stability, fewer regressions |
| Reward-checking model + hacking-case DB (top-3 similar cases retrieved) | Triggered on suspicious reward jumps [S18] | Detects hacks "over 60 %" of the time |
| Two-LLM source audit of accepted kernels | After the test suite [S24] | Found 6 residual hacks (1.3 %) among 453 surviving fast kernels |
| Clean-context code reviewer | After writing, before merge [S34] | ~2 bugs per PR, ~58 % severe; works better *without* the coder's context |
| Independent test designer (never sees the code) | Before execution [S41] | AgentCoder 96.3 % HumanEval pass@1 vs 90.2 % for prior SOTA, with 57K vs 138K tokens |
| Judge with curated NCU metrics | Between rounds [S13] | 1.677x vs 1.222x without optimisation feedback |
| Novelty judge (embedding similarity > 0.95 → LLM) | Before evaluation [S3] | Saves evaluation budget on near-duplicates |
| Runtime surrogate with deferral | Before evaluation [S27] | Several times more candidates per GPU budget |

Principle across all of these: deterministic, executable gates stay primary ("hard, verifiable constraints" [S6];
"grounded tool use" [S5]); LLM critics are cheap pre-filters and post-hoc auditors, not the source of truth.

---

## 6. Roles, model mixing and cost

* **Orchestrator-worker cost.** Single agents use ~4x chat tokens, multi-agent ~15x; it pays for "breadth-first"
  tasks whose information exceeds one context, not for tasks needing shared context or with many dependencies;
  "most coding tasks involve fewer truly parallelizable tasks than research" [S33]. Delegation needs "an
  objective, an output format, guidance on the tools and sources to use, and clear task boundaries"; effort is
  scaled to complexity (1 agent with 3–10 tool calls up to 10+ sub-agents) [S33]. Claude Code agent teams use
  ~7x the tokens of a standard session when teammates run in plan mode [S45].
* **Deterministic orchestration.** KernelFalcon keeps "worker lifecycles, timeouts, and success conditions"
  in Python and lets LLMs only generate code and metadata [S5]; a central coordinator contains error
  amplification (4.4x vs 17.2x) [S36].
* **Model mixing per role.**
  * Volume model + strong model: AlphaEvolve Flash + Pro [S1]; OpenEvolve 0.8/0.2 weights [S2]; a UCB1 bandit
    over models by fitness improvement beats any fixed choice (Shinka) [S3].
  * Lead/worker: Opus 4 lead + Sonnet 4 sub-agents in Anthropic's system [S33]; Claude Code docs advise Sonnet for
    teammates [S45].
  * Per-role temperature: STARK plan τ=0.8, code/debug τ=0.1 [S12].
  * Cost frontier: KernelArc found Kimi K3 the most cost-efficient trajectory (>700 TFLOPS "after a few dollars")
    and Claude Opus 5 the best (~766 TFLOPS) "at the largest cumulative cost" [S8]; AccelOpt with open models
    matched Claude Sonnet 4 at 26x lower cost [S16].
  * **Advisor tool**: a cheaper executor consults a stronger advisor "before committing to an approach, when an
    error keeps recurring, and before declaring a task done". It is available on subscription accounts, can be set
    in the Agent SDK with `/advisor <model>` (Claude Code ≥ 2.1.260) or the `advisorModel` setting, sub-agents
    inherit it, and it does not invalidate the main prompt cache. Each call re-reads the transcript uncached [S55].
* **Cost anchors for one kernel:** CudaForge ~$0.30 and 26.5 min on one RTX 6000 [S13]; MKEvolve 1.7–2.1M
  tokens per problem at 160 kernels, KernelFalcon 0.2–1.4M [S7]; ADRS runs "5h (100 iters), ≤$15" [S26].

---

## 7. Claude platform constraints (documented facts)

**Authentication and plan limits**
* Agent SDK docs: "Unless previously approved, Anthropic does not allow third party developers to offer claude.ai
  login or rate limits for their products, including agents built on the Claude Agent SDK" [S42]. kernel-agent
  runs on the user's own Claude Code login. Whether that note applies to an open-source tool that users run on
  their own subscription is a policy question for the maintainer, not settled here.
* Pro/Max usage limits "are shared across Claude and Claude Code" [S51]. Seat allowances reset "on a rolling
  five-hour window and a weekly window" (Team/Enterprise docs) [S45]; for Pro/Max, the five-hour limits plus weekly
  limits (overall and Opus-specific) since Aug 28 2025 [S52]. The errors are "You've hit your session limit" /
  "weekly limit"; switching models does not help, except after a model-specific limit message [S45]. **No
  documented cap on the number of concurrent sessions**: concurrency is bounded by token spend per window (each
  sub-agent or teammate "sends its own requests"; teammates consume tokens "until it exits") [S45].
* API-key mode (if used): per-model RPM/ITPM/OTPM (Start tier Opus 5.5: 1,000 RPM, 2M ITPM, 400k OTPM); **cache
  reads do not count toward ITPM** for most models; 429s carry `retry-after`; "acceleration limits" on sharp
  ramps, so ramp up gradually [S50].

**Agent SDK concurrency**
* One session = one `claude` subprocess; N concurrent sessions = N subprocesses; "1 GiB RAM, 5 GiB disk, and 1 CPU
  per agent is a reasonable starting point"; no top-level session timeout (use `max_turns`); "large
  parallel-subagent fanouts can hit rate limits: break work into smaller batches" [S49].
* Sub-agents: `agents={name: AgentDefinition(description, prompt, tools, model, skills, maxTurns, background,
  effort, …)}`; only the final message returns to the parent; background by default; limits
  `CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH` (default 3), `CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS` (default 20),
  `max_budget_usd` (no default) [S43].
* **Agent teams are not available to SDK sessions**: "In non-interactive mode with the `-p` flag, including Agent
  SDK sessions, Claude doesn't spawn teammates" [S44]. A blackboard for kernel-agent has to be its own Python
  orchestration (ledger + files), not Claude Code teams.

**Prompt caching across agents**
* Prices: cache writes 1.25x (5 min) / 2x (1 h); reads 0.1x base, 0.05x on Opus 5.5 / Sonnet 5.5; exact prefix
  match in the order tools → system → messages; caches are shared within an organisation (or workspace); **a cache
  entry is available only after the first response begins**, so parallel requests should wait for the first one
  [S47].
* Claude Code: on a subscription the **main conversation (including Agent SDK turns) gets a 1-hour TTL**; sub-agents,
  teammates and compaction get 5 minutes unless `subagentPromptCacheTtl` / `CLAUDE_CODE_SUBAGENT_PROMPT_CACHE_TTL`
  is set. "Sessions you run in parallel in the same directory build matching prefixes and read each other's
  cache"; **sessions in different directories miss each other's cache** (the auto-memory path sits in the system
  prompt, and the conversation opens with the working directory). Model switches and effort changes (on older
  models) invalidate; workflow fan-outs hold same-prefix agents up to 5 s so they read the first one's cache [S46].
* `system_prompt={"type": "preset", "preset": "claude_code", "append": …, "exclude_dynamic_sections": True}` moves
  per-session context into the first user message so identical configurations share the system-prompt cache
  entry [S48].
* kernel-agent today: `agent/runner.py` uses the `claude_code` preset + `append` with `setting_sources=[]`;
  sessions run with `cwd` = target dir / transforms dir / round dir (orchestrator.py). So concurrent sessions of
  different arms do not share cache prefixes.

**Skills and agent definitions**
* Skills use progressive disclosure: name + description in the system prompt at start, full `SKILL.md` when
  relevant, extra files only when referenced; bundled scripts run "without loading … into context" [S53].
* SDK: skills load only through the `user`/`project` setting sources **or the `plugins` option**; with
  `setting_sources=[]` (kernel-agent) they do not load unless provided as a plugin path; `skills=[…]` allowlists
  per session; `AgentDefinition.skills` preloads named skills into a sub-agent [S54, S43].

---

## 8. Does retrieval over API docs help kernel agents?

The evidence is mixed and thinner than the claims. Examples beat raw docs.
* **Hardware specs/docs in the prompt barely help.** KernelBench: "Providing hardware information does not
  significantly impact the outputs"; few-shot technique examples made models try more aggressive kernels, which
  **lowered fast_1** through more failures, though correct ones used tiling or shared memory more [S23].
* **Similar-code retrieval + knowledge helps.** GEAK ablation: knowledge injection 20.1 % execution accuracy →
  +1-shot retrieved similar Triton code 27.2 % → +optimizer 40.8 % [S10].
* **Retrieved verified kernels raise low-resource DSL correctness (weak evidence: a course project).** DSLBench:
  zero-shot pass@1 Triton 32.7 %, CuTe DSL 14.7 %, TileLang 12.2 %, ThunderKittens 0.95 % (Gemini 3 Pro);
  few-shot RAG over a database of verified kernels lifted TileLang to 59.3 %, accuracy grew with the database
  size, and "accessing valid syntax examples is the primary bottleneck" [S31].
* **Proprietary hardware needs knowledge injection**, but KernelEvolve gives no ablation [S9].
* **Too much context hurts:** CudaForge's judge did worse with all NCU metrics than with 24 curated ones [S13].
* General code: DocPrompting +2.85 pass@1 (52 % relative) on CoNaLa [S32]; CodeRAG-Bench: good contexts help, but
  "retrievers often struggle to fetch useful contexts, and generators face limitations in using those contexts" [S32b].

For kernel-agent: retrieve **its own verified kernels and recipes** (the library, by op/shape/backend/arch) and
small, targeted API excerpts on demand (skills); avoid bulk documentation in prompts; measure correctness with
and without it (A/B on quick-check pass rate).

---

## 9. What transfers to kernel-agent on one GPU

R1–R16, in rough priority order.

**R1. GPU job queue with classes, exclusive timing.** Generalise the GPU lock into a priority queue run by
the orchestrator:
1. CPU lane, no lock: builds and compiles, static and anti-gaming checks, LLM critics.
2. `quick` lane: correctness-only checks. Short and frequent; may run back to back, and is never timed.
3. `timed` lane: interleaved benchmarks. Exclusive and non-preemptible.
4. `e2e/integration` lane: long paired A/B runs. Exclusive; scheduled as batches.

Use priority with aging, so integration cannot starve and quick checks do not jump into a timed batch. Record
queue wait per job in the ledger. Basis: time-slicing makes concurrent timing invalid [S29]; every multi-agent
kernel system serialises timing [S6, S7, S8].

**R2. Concurrency over arms first.** Run k concurrent slices from *different* arms (kernel targets, systems,
native, research, dossiers). These are parallelisable work with a verifier, which gains [S36]. Same-target
concurrency is capped at 2 workers with shared conclusions [S8]; independent sampling is the worst use of a
budget [S7, S19]. Keep the UCB/Amdahl scheduler, but have it fill k slots instead of 1.

**R3. Choose k from measured GPU idle time and token budget, adaptively.** If an agent needs the GPU a fraction g
of its wall time, k agents keep the GPU busy about k·g of the time, and queueing grows sharply as k·g → 1.
Start at k=2–3. Lower k when the median queue wait exceeds, for example, 20 % of slice wall time, or when token
use per 5-hour window nears the plan limit. Session/weekly-limit errors pause new slices, they do not fail them
(`autoContinueAtUsageLimit` exists for interactive use [S45]).

**R4. Blackboard = conclusions only, ledger = truth.** Per target and per technique, keep `wins` (speedup,
shapes, why) and `traps` (dead end + error) [S8]; per-slice reflections (diagnosis right? fix effective? lessons,
avoid/try) [S6]; a periodic meta-summary [S3]; slow→fast diff pairs [S16]. Agents read a digest at slice start
(already the case) plus a **read-only** view of other arms' best snapshots. No raw logs, no free-form chat
between agents [S8, S33].

**R5. Single writer per module.** Assign ownership of modules and files per arm (a registry, from #112's "which
items touch which modules"); integration is the only composer. Overlapping proposals become swap candidates, not
concurrent edits [S34, S44].

**R6. Handle staleness.** A slice that started before an integration changed the baseline gets a "baseline
moved" note in its next digest, and its results are re-measured on the new composite before acceptance [S3].

**R7. Cheap critic before the GPU.** A clean-context, read-only reviewer on a cheaper model checks each
candidate diff against a known-exploit list before it is queued for timing [S4b, S18, S24, S34]. The exploits
are extra streams or async work outside the timer, caching by pointer or shape, try/except fallback to PyTorch,
lazy or subclass outputs, hard-coded shapes, and skipped work on test distributions. Grow a hacking-case
database from real incidents [S18]. Deterministic checks stay the gate; the critic only adds `suspect` flags and
a reason.

**R8. Audit winners.** Before a kernel enters integration, run a second-opinion audit on a different model
[S24]. This is cheap: it runs once per accepted item, not per candidate.

**R9. Roles.** The coordinator and scheduler stay deterministic Python [S5, S36]. LLM roles:
* planner
* kernel engineers (per arm)
* systems
* native
* researcher / dossier
* **profiler-analyst**: deterministic NCU/ceilings digest plus an LLM diagnosis [S6, S13]
* **critic**
* the integrator stays deterministic

Brief every role with objective, output format, tools, and boundaries [S33].

**R10. Model mixing per role, measured.** Strong model for planner, research and hot-target engineers; cheaper
model for critic, profiler-analyst and dossier; the advisor tool for engineers that stall [S55]. Track
ms-saved per token per (role, model) in the ledger, and let a bandit pick models per arm [S3].

**R11. Prompt-cache-friendly sessions.**
* One static `append` per role.
* `exclude_dynamic_sections=True` [S48].
* Put per-target material in the user prompt, not in `append`.
* Run sessions from a common `cwd`, passing target paths explicitly, or accept per-directory misses [S46].
* Stagger concurrent starts of same-role sessions by a few seconds, so the first request writes the cache [S46, S47].
* Never switch model mid-session [S46].
* With sub-agents, set `CLAUDE_CODE_SUBAGENT_PROMPT_CACHE_TTL=1h` when they idle more than 5 minutes [S46].

**R12. Budget- and rate-limit-aware concurrency.** Per-session `max_budget_usd` and `max_turns` [S43, S49].
Batch wide fan-outs [S49]. In API-key mode, a high cache-hit rate raises effective ITPM, because cache reads are
exempt [S50]. Ramp k gradually [S50].

**R13. Knowledge as skills, examples over docs.** Package the knowledge base (CUDA, CuTe DSL, Triton, TileLang
recipes) as skills loaded through the SDK `plugins` option, since `setting_sources=[]` blocks filesystem skills
[S54]. Give each role a `skills` allowlist. Retrieve verified library kernels by op/shape/arch as few-shot
examples [S10, S31]. A/B the effect on quick-check pass rate before keeping it [S23].

**R14. Islands only for hot targets.** Map workers to niches (backend × approach, MAP-Elites style [S1, S2]).
Migrate best snapshots between them occasionally (the existing `--reseed-workers`). Add an embedding-similarity
novelty check on top of the SHA dedup [S3]. Use plateau-triggered drafting: after N stale slices the next slice
must change DSL, algorithm or layout [S8].

**R15. Evaluation cascade.** The order is compile → tiny-shape correctness → full correctness → timed →
end-to-end, each stage gating the next [S1, S8]. Optionally, a runtime surrogate decides which correct
candidates are worth timing [S27].

**R16. `--dry-run` simulation.** A discrete-event simulation of k agents with think, compile, quick, timed and
e2e durations drawn from the measured distributions. It covers the queue discipline from R1 and a token-per-window
limit. It reports GPU utilisation, queue wait, slices per hour, and tokens per hour against the plan window.
The same simulator tunes k (R3).

---

## Sources

* [S1] Novikov et al., *AlphaEvolve: A coding agent for scientific and algorithmic discovery*, https://arxiv.org/abs/2506.13131 (blog: https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
* [S2] OpenEvolve repository, https://github.com/algorithmicsuperintelligence/openevolve ; default config https://github.com/algorithmicsuperintelligence/openevolve/blob/main/configs/default_config.yaml
* [S2b] OpenEvolve MLX Metal kernel example README, https://github.com/algorithmicsuperintelligence/openevolve/blob/main/examples/mlx_metal_kernel_opt/README.md
* [S3] Lange et al., *ShinkaEvolve*, https://arxiv.org/abs/2509.19349 ; https://sakana.ai/shinka-evolve/
* [S4] Sakana AI, *The AI CUDA Engineer*, https://sakana.ai/ai-cuda-engineer/
* [S4b] Lange et al., *Towards Robust Agentic CUDA Kernel Benchmarking, Verification, and Optimization* (robust-kbench), https://arxiv.org/abs/2509.14279
* [S4c] TechCrunch, *Sakana walks back claims…*, https://techcrunch.com/2025/02/21/sakana-walks-back-claims-that-its-ai-can-dramatically-speed-up-model-training/
* [S5] PyTorch blog, *KernelFalcon: Autonomous GPU Kernel Generation via Deep Agents*, https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/
* [S6] PyTorch blog, *KernelAgent: Hardware-Guided GPU Kernel Optimization via Multi-Agent Orchestration* (2026-03-06), https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/
* [S7] Yoo et al., *MKEvolve: A Modular Multi-Agent Framework for Kernel Code Generation*, https://arxiv.org/abs/2607.20501
* [S8] *KernelArc: A Multi-Agent Framework for GPU Kernel Optimization*, https://arxiv.org/abs/2608.17071
* [S9] *KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta*, https://arxiv.org/abs/2512.23236
* [S10] *GEAK: Introducing Triton Kernel AI Agent & Evaluation Benchmarks*, https://arxiv.org/abs/2507.23194
* [S10b] AMD ROCm blog, *GEAK v3*, https://rocm.blogs.amd.com/artificial-intelligence/kernel-optimization-agent/README.html
* [S11] Wei et al., *Astra: A Multi-Agent System for GPU Kernel Performance Optimization*, https://arxiv.org/abs/2509.07506
* [S12] Dong et al., *STARK: Strategic Team of Agents for Refining Kernels*, https://arxiv.org/abs/2510.16996
* [S13] *CudaForge*, https://arxiv.org/abs/2511.01884
* [S14] *TritonForge*, https://arxiv.org/abs/2512.09196
* [S15] *PRAGMA*, https://arxiv.org/abs/2511.06345
* [S16] *AccelOpt*, https://arxiv.org/abs/2511.15915
* [S17] *KernelBand*, https://arxiv.org/abs/2511.18868
* [S18] *CUDA-L1*, https://arxiv.org/abs/2507.14111
* [S19] Baronio et al., *Kevin: Multi-Turn RL for Generating CUDA Kernels*, https://arxiv.org/abs/2507.11948 ; https://cognition.com/blog/kevin-32b
* [S20] Stanford Scaling Intelligence, *Surprisingly Fast AI-Generated Kernels We Didn't Mean to Publish (Yet)*, https://scalingintelligence.stanford.edu/blogs/fastkernels
* [S21] METR, *Measuring Automated Kernel Engineering*, https://metr.substack.com/p/2025-02-14-measuring-automated-kernel-engineering
* [S22] NVIDIA, *Automating GPU Kernel Generation with DeepSeek-R1 and Inference Time Scaling*, https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/
* [S23] Ouyang et al., *KernelBench*, https://arxiv.org/abs/2502.10517
* [S24] *KernelBench-Verified*, https://arxiv.org/abs/2607.16241 (summary: https://www.alphaxiv.org/abs/2607.16241)
* [S25] METR, *Recent Frontier Models Are Reward Hacking*, https://metr.org/blog/2025-06-05-recent-reward-hacking
* [S26] *Let the Barbarians In: How AI Can Accelerate Systems Performance Research* (ADRS), https://arxiv.org/abs/2512.14806
* [S27] *GPU Forecasters: Language Models as Selective Surrogates for Kernel Runtime Optimization*, https://arxiv.org/abs/2605.31464
* [S28] *KernelBot* (ICML 2025 CODEML workshop), https://icml.cc/virtual/2025/48171 ; https://github.com/gpu-mode/kernelbot
* [S29] NVIDIA MPS architecture, https://docs.nvidia.com/deploy/mps/architecture.html
* [S30] *Agentic Kernel Optimization: Generating State-of-the-Art GPU Kernels Without Hand-Written CUDA*, https://arxiv.org/abs/2608.14560
* [S31] W. Chan, *DSLBench* (Stanford CS191 project, Fall 2025), https://cs191.stanford.edu/projects/Fall2025/_Willy___Chan_.pdf
* [S32] Zhou et al., *DocPrompting*, https://arxiv.org/abs/2207.05987 ; [S32b] *CodeRAG-Bench*, https://arxiv.org/abs/2406.14497
* [S33] Anthropic, *How we built our multi-agent research system*, https://www.anthropic.com/engineering/multi-agent-research-system
* [S34] Cognition, *Don't Build Multi-Agents* (2025-06-12), https://cognition.com/blog/dont-build-multi-agents ; *Multi-Agents: What's Actually Working* (2026-04-22), https://cognition.com/blog/multi-agents-working
* [S35] Cemri et al., *Why Do Multi-Agent LLM Systems Fail?* (MAST), https://arxiv.org/abs/2503.13657
* [S36] Google Research, *Towards a science of scaling agent systems*, https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/ ; https://arxiv.org/abs/2512.08296
* [S37] Hong et al., *MetaGPT*, https://arxiv.org/abs/2308.00352
* [S38] Qian et al., *ChatDev*, https://arxiv.org/abs/2307.07924
* [S39] Wu et al., *AutoGen*, https://arxiv.org/abs/2308.08155
* [S40] Wang et al., *OpenHands*, https://arxiv.org/abs/2407.16741
* [S41] Huang et al., *AgentCoder*, https://arxiv.org/abs/2312.13010
* [S42] Claude Agent SDK overview, https://code.claude.com/docs/en/agent-sdk/overview
* [S43] Agent SDK subagents, https://code.claude.com/docs/en/agent-sdk/subagents
* [S44] Claude Code agent teams, https://code.claude.com/docs/en/agent-teams
* [S45] Claude Code costs, https://code.claude.com/docs/en/costs
* [S46] How Claude Code uses prompt caching, https://code.claude.com/docs/en/prompt-caching
* [S47] Claude API prompt caching, https://platform.claude.com/docs/en/build-with-claude/prompt-caching
* [S48] Agent SDK, modifying system prompts (`exclude_dynamic_sections`), https://code.claude.com/docs/en/agent-sdk/modifying-system-prompts
* [S49] Hosting the Agent SDK, https://code.claude.com/docs/en/agent-sdk/hosting
* [S50] Claude API rate limits, https://platform.claude.com/docs/en/api/rate-limits
* [S51] Claude Help Center, *Using Claude Code with your Pro or Max plan*, https://support.claude.com/en/articles/11145838
* [S52] TechCrunch, *Anthropic unveils new rate limits to curb Claude Code power users* (2025-07-28), https://techcrunch.com/2025/07/28/anthropic-unveils-new-rate-limits-to-curb-claude-code-power-users/
* [S53] Anthropic, *Equipping agents for the real world with Agent Skills*, https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
* [S54] Agent SDK skills, https://code.claude.com/docs/en/agent-sdk/skills
* [S55] Claude Code advisor, https://code.claude.com/docs/en/advisor ; API advisor tool, https://platform.claude.com/docs/en/agents-and-tools/tool-use/advisor-tool
