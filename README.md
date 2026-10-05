# kernel-agent

Give it a Hugging Face URL; it profiles the model on your GPU, finds the slow
parts, and has Claude write custom GPU kernels for them in **CUDA C++,
CuTe DSL, Triton or TileLang**. It also tries model-level algorithm changes
(static caches, CUDA graphs, merged projections). Each candidate is checked
for correctness against real captured inputs, benchmarked, and kept only if
the whole model still produces the same output and runs faster.

Works on **LLM**, **STT**, **TTS** and **diffusion** models, plus a built-in
workload for **VoxCPM2** (diffusion-autoregressive TTS). When no built-in
workload can run a model (custom TTS stacks, for example), Claude writes a
benchmark harness for it first.

```bash
uv sync --extra all
uv run kernel-agent doctor --smoke      # check GPU, compilers, all 5 backends
uv run kernel-agent optimize https://huggingface.co/Qwen/Qwen3-0.6B
```

## How it works

```
HF URL ─► resolve (modality, arch, family, size)
        ─► analyze   load model, baseline latency, determinism check,
                     sensitivity probe, teacher-forcing self-check,
                     held-out input baseline, module-level + kernel-level
                     profile, compiled baseline (the model's own
                     torch.compile path)                           [GPU worker]
        ─► (harness) Claude writes harness.py if the built-in workload fails
        ─► plan      Claude reads the profile + source, picks target modules,
                     an approach and backends for each, and model transforms
        ─► capture   each target module is saved with real inputs/outputs
                     (prefill + decode shapes, first/middle/last decode step,
                     KV-cache side effects, every entrypoint such as
                     forward_step, correctness-only cases from extra
                     settings), plus statistics of every call
                     (workload_profile.md); a region target is first
                     refactored into its own submodule by a Claude
                     "refactor" agent, verified bitwise     [GPU worker]
        ─► kernels   one Claude "kernel engineer" per target (or k isolated
                     workers, --seeds-per-target) writes candidates,
                     calls evaluate_candidate (correctness + interleaved
                     benchmark + optional per-kernel profile) and iterates
        ─► transforms  Claude "systems engineer" writes model-level transforms,
                     validated end to end with evaluate_e2e
        ─► integrate every winner measured alone, then combined greedily from
                     the fastest single item (or from the systems agent's
                     fastest measured combination, when it is faster again),
                     validated against the baseline output; then the accepted
                     kernels under the compiled baseline
        ─► report    report.md + optimized/ (kernels + apply.py)
```

All GPU work runs in subprocesses under a GPU lock (one per GPU, see "GPUs and
the GPU lock"), so a crashing kernel or an illegal memory access can't kill the
run, and parallel agents (`--parallel N`) never benchmark on the same GPU at the
same time.

### What "correct" means

* **Module level** (`evaluate_candidate`): every captured case must match the
  reference outputs **and** the in-place side effects (for example KV-cache
  appends) within dtype-aware tolerances (bf16 2e-2, fp16 1e-2, fp32 1e-4;
  at most 0.1 % of elements outside). Side effects are compared on the
  elements the reference or the candidate changed, so writing one position of
  an 8192-long cache (or forgetting to) is not lost in the 0.1 %. If `build()`
  hands back the reference module unchanged, the candidate is rejected.
* **More than one KV length and one setting**: decode steps share their
  primary input while the KV cache grows, so `capture` runs the workload once
  to count them and keeps the **first, middle and last** decode step of every
  decode signature as cases (`"bucket"`, signature `... @ decode step 31/60`).
  A kernel that only works for the first step's KV length fails the later
  buckets, and each bucket weighs a third of the calls in the timing.
  `Workload.variants()` declares extra settings (LLM: prompt length 37 and
  301 at batch 2; VoxCPM: a short text with 8 patches; diffusion: ¾ of the
  resolution); calls whose primary input the main run lacks become
  **correctness-only** cases (`count` 0, `[correctness only: ...]` in the
  signature): checked like the others, never timed. A kernel that only
  handles the captured length fails them.
* **Model level** (`e2e`): the workload's own comparison. For LLM/STT that is
  identical greedy tokens for the first N tokens plus first-step logits cosine
  ≥ 0.99. For TTS it is spectral cosine. For diffusion it is PSNR ≥ 25 dB on
  the same seed. Timing is always the free-running workload.
* **Chaotic workloads** (teacher forcing): in an autoregressive model with
  continuous, sampled outputs (VoxCPM: LM → flow-matching head with fresh noise
  → latent fed back), any numerically-correct kernel makes the free-running
  output diverge. The bundled Triton RMSNorm gives spectral cosine 0.69 on
  VoxCPM2, a different seed 0.37. Such workloads implement
  `run_teacher_forced`: the candidate replays the baseline trajectory with the
  same per-step noise, and its own prediction for every step is compared with
  the baseline's (per-step cosine, mean and minimum, plus an RMS ratio). The
  free-running output only has to pass a sanity check: finite, same shapes, RMS
  energy within ±25 % (VoxCPM: ±60 % for audio). The full free-running
  comparison is reported as `metrics.free_running` and does not gate.
  Teacher-forced metrics are in `metrics.teacher_forced`.
* **Sensitivity probe**: `analyze` runs the workload once more with every
  `nn.Linear` output multiplied by `1 ± 2^-8` (about one bf16 rounding step) and
  records whether the free-running comparison still passes
  (`baseline.json` → `sensitivity`, and `profile/summary.md`). If it fails, the
  workload counts as chaotic. If it also has no teacher forcing, a loud warning
  says that end-to-end validation will reject (almost) every kernel.
  Teacher-forced workloads also replay their own trajectory once, which must be
  exact (`baseline.json` → `teacher_forcing`).
* **Held-out input** (`workloads/holdout.py`): every e2e run times the same
  main input, so a transform that memoised outputs across runs would pass all
  of the above. `analyze` therefore also stores the baseline output of a
  held-out input (`Workload.holdout_options(1)`: LLM another prompt of the
  same length, STT rotated audio, TTS / VoxCPM another text and seed with the
  same patches, diffusion another prompt and seed), and `e2e` judges every
  candidate on it too with the same check (teacher forced where supported),
  untimed. It fails when the held-out check fails, or when the candidate's
  held-out output equals its main output although the baseline's differ.
  **Memoisation probe**: after the timed runs, one run on a *fresh* input
  (`holdout_options(k)`, random `k >= 2`: the held-out input's shapes, another
  seed or rotation, never seen by any process) is timed. Real inference takes
  about as long as the repeated main runs; the candidate fails when it takes
  more than 3× their median (and ≥ 10 ms more) on a second fresh input as
  well (one slow run can be a busy GPU), or when its output equals the
  held-out output although their baseline outputs differ. A first run at new
  shapes (compilation, CUDA-graph capture) is not penalised: the held-out run
  warms those shapes up before the probe. Details are in `metrics.holdout`.

### What "faster" means

Reference and candidate are timed in alternating rounds with CUDA events
after a GPU warm-up, and the median round is reported. Mutable inputs (caches)
are deep-copied outside the timed region. A module's speedup is weighted by how
often each captured shape runs per inference. The end-to-end speedup is
wall-clock latency of the whole workload.

### Ground truth the agents cannot quietly change

Agents have a Bash tool, so file permissions alone cannot protect what the
evaluator trusts. Everything it trusts lives in `<run>/.truth/`, outside the
agents' working directories: the full captures (with reference outputs and
post-call state), the baseline output, every evaluation record and every
evaluated snapshot. The agent's `targets/<id>/` holds `capture_inputs.pt`
(module + inputs, no outputs) for local debugging, plus copies of its
snapshots (`history/`) and records (`results.jsonl`) that nothing reads back.

Truth files are `chmod a-w`, and their sha256 is recorded when kernel-agent
writes them: in the orchestrator's memory (the authority) and in `run.json` →
`truth` for a resumed run. They are checked before use:

* `evaluate_candidate` passes the capture's digest to the evaluator
  subprocess, which hashes the bytes it loads; a changed capture is refused
  (`status: tampered`). A snapshot that changes during its evaluation voids
  the result.
* `e2e` gets the baseline latency from the orchestrator (`--baseline-ms`) and
  verifies `baseline.json` and the baseline outputs, main and held-out
  (`--verify`).
* Each record stores its snapshot's sha256. Winners (`best_for_target`), the
  integration and the export use only records kernel-agent wrote (lines
  appended by anyone else are ignored) whose snapshot still has that digest.
  The integration's reuse cache comes only from an unmodified
  `integration.json`.

A mismatch is refused with a `TAMPER` line on stderr and a `tamper` event in
`events.jsonl`. Runs created before `.truth/` existed are read from their old
paths without these checks. The evaluator runs candidate code in the same
user account, so a candidate could still read `.truth/` at run time; process
isolation is a separate step.

### Strong baseline: eager vs compiled

Speedups over eager PyTorch overstate the gain when the model already has a
fast path that users get for free. `Workload.reference_optimizations()` is an
optional hook that applies what a competent user would do without custom
kernels and returns a one-line description:

| workload | reference optimisations |
|---|---|
| VoxCPM | `model.optimize()`: `torch.compile(mode="reduce-overhead", fullgraph=True)` of both LMs' `forward_step`, the LocEnc and the LocDiT estimator |
| LLM | static KV cache (`cache_implementation="static"`); transformers then compiles the decode step (`reduce-overhead`). Only for architectures that declare compile support. SmolLM2-135M, 512 + 64 tokens: 985 → 187 ms, identical tokens |
| diffusion | `torch.compile` of the denoiser (`pipe.transformer` / `pipe.unet`; not with `cpu_offload`) |
| STT, TTS, harnesses | none by default (implement the hook, or pass `--compile-baseline`) |

`analyze` measures the eager baseline first, releases the model and measures
the reference optimisations in a fresh process (`python -m
kernel_agent.strong_baseline`, log in `logs/strong-baseline.log`): 2 untimed
warm-up runs (compilation, CUDA-graph recording), then the same timed runs as
the eager baseline. `--compile-baseline` also measures workloads without the
hook, with a generic `torch.compile` of their root modules. `baseline.json`
gets `compiled_ms` (median, or `null` when it failed) and `compiled_detail`:
what was applied, timings, `warmup_s`, `compile_s` (warm-up time beyond the
steady-state runs), `speedup_vs_eager`, and `quality`: the compiled output
judged against the eager output by the workload's own end-to-end check
(teacher forced for chaotic workloads). A failure is recorded and reported,
never fatal. Harness checks (`--no-profile`) and the re-profiles of improve
rounds do not measure it (rounds copy the run's values).

* `report.md`, `kernel-agent status`, `dashboard.html`, `kernel-agent watch`
  and `profile/summary.md` show every result vs eager **and** vs compiled.
* The planner and systems prompts state the real headroom: the compiled
  latency, its speedup over eager, and that an optimisation only helps users
  when it beats it (re-discovering `torch.compile` / CUDA graphs is no
  progress).
* `integrate` also measures *reference optimisations + accepted kernels*
  (the built-in transform `integrate/builtin/reference_optimizations.py`
  applied after the kernels) and records in `integration.json` → `reference`
  whether the kernels compose with it: `composes` (beats the compiled baseline
  by > 1 %), `no_gain`, `fails_quality`, or `breaks` (for example a kernel
  launcher that is a graph break under `fullgraph=True`, with the error). The
  greedy result (`accepted`, `final`) is unchanged.
* `evaluate_candidate(..., compile_check=true)` / `kernel-agent eval
  --compile-check` adds an optional stage on a fresh `build()`:
  `torch._dynamo.explain` per entrypoint (graphs, graph breaks and where they
  happen) and `torch.compile(fullgraph=False)` of the candidate on every
  captured case, with outputs and side effects checked
  (`result["compile_check"]`: `passed`, `graph_breaks`, `fullgraph_ok`,
  `break_reasons`). `examples/triton_rmsnorm_custom_op.py` shows how to wrap
  a kernel in `torch.library.custom_op` + `register_fake` so it is one opaque
  op for Dynamo, Inductor and CUDA graphs instead of a graph break.

### Entrypoints other than `forward`

Custom decode loops often call module methods directly, e.g. VoxCPM's
`MiniCPMModel.forward_step → MiniCPMDecoderLayer.forward_step →
MiniCPMAttention.forward_step`. Those calls bypass `nn.Module` hooks, so
kernel-agent finds them by name (`forward*`, `step`, `decode*`, `prefill*`,
`generate_step` defined on the model's own classes; add others with
`-o entrypoints=Cls.method,...` or `Workload.entrypoints`) and wraps them per
instance:

* the profile shows a *methods* column (`forward_step×720 (638.4 ms) ·
  forward×2448 (1208.7 ms)`) and signatures like `forward_step: a0[1, 2048]`;
* capture records cases from every entrypoint (`"method": "forward_step"`),
  keeping at least one case per entrypoint (`--max-cases`, default 4; the
  middle / last decode-step cases of a signature come on top);
* the evaluator replays each case through its method (`candidate.forward_step(...)`)
  for correctness and timing; a candidate without a captured method is a
  `build_error`, and integration refuses a replacement that lacks one.

Arguments that are views into a much larger storage (one layer's K/V slice of
a static `[2, layers, B, H, T, D]` cache) are captured compactly: only the
memory the views span, with the same shapes, strides and aliasing between
overlapping views (a full deep copy would store the whole cache for every
case, before and after the call). Side effects are therefore checked on the
memory the arguments cover, not on the rest of the buffer they were sliced from.

### Workload statistics and phase-specific targets

A capture keeps a few cases, but the kernel engineer also needs to know how
the model calls the module the rest of the time. During the capture run every
call of the target's instances is summarised in
`targets/<id>/workload_profile.md` (+ `.json`), and its top facts go into the
engineer's prompt:

* calls and share per entrypoint and primary-input signature, with the phase;
* masks: `None`, all ones, causal, prefix (a decode step that may only read
  the first *L* slots) or other;
* contiguity / strides and dtypes of tensor arguments, zero biases (argument
  or parameter), values of flags such as `is_causal`;
* integer ranges (`position_id`, `cache_position`: min / max / distinct);
* KV-cache arguments: cache slots and the valid length per call (from the
  position, else from the mask). VoxCPM2's decode attention, for example,
  reads a static cache of 8192 slots of which only the first `position_id + 1`
  hold data.

Values are reduced on the GPU without a sync per call and read back once at
the end of the run.

A call is `decode` when it goes through a `*step*` entrypoint or its first
tensor argument is `[batch, 1, ...]`, otherwise `prefill`. The profile summary
has a *Phase split* table for classes called in both phases (with the
qualname patterns of their instances). The planner can then split such a class
into one target per phase (`"phase": "prefill" | "decode"` in the plan; `all`
is the default), and restrict a target to some instances with
`"qualname_regex"` (searched in the full qualname, e.g. `feat_decoder\.`).
A phase target captures only that phase's calls (from the instance with the
most of them, unless `qualname` is set). Integration keeps the instance and
sends only that phase's calls of the captured entrypoints to the kernel. The
other calls keep the reference, so a `prefill` and a `decode` target on the
same class combine. `optimized/apply.py` applies them the same way.

### Region targets: fusions across module boundaries

A target replaces one module class, so ops that sit in a parent's `forward`
between its children cannot be fused by a kernel target: the residual add
after attention and the next RMSNorm of a decoder layer, or an output
projection + residual add + norm (`kernel_agent/region.py`, after the Fuser
of KernelAgent). The planner marks such a fusion `"kind": "region"` with a
`parent_class` (e.g. `LlamaDecoderLayer`) and a `region` (the ops, e.g. "the
residual add after self_attn + post_attention_layernorm"). Before the kernel
engineer starts:

1. the parent class is captured (every phase) to `.truth/captures/<id>.parent.pt`;
2. a `refactor-<id>` agent (read-only tools plus `verify_rewrite`; it may
   write only `targets/<id>/rewrite.py`, no shell) writes
   `rewrite(parent) -> nn.Module`: a drop-in parent whose entrypoints call a
   new submodule of class `Region_<id>` for the region, the same ops in the
   same order;
3. kernel-agent seals a copy of `rewrite.py` and replays every captured call
   of the parent with it applied. Outputs and in-place side effects (KV
   caches) must be bitwise identical, or within about one unit in the last
   place with no mismatch allowance, and the parent must call `Region_<id>`.
   A rewrite that changes the math (say, the norm before the residual add) is
   rejected and the target dropped;
4. the workload runs with the rewrite applied to every parent instance and
   `Region_<id>` is captured. From here on it is a normal target.

Wherever the kernel is applied (`e2e`, integration, re-profiles,
`optimized/apply.py`), the target's verified rewrite runs first on every
instance of the parent class (`qualname_regex` selects parent instances),
then the kernel replaces the `Region_<id>` modules. A rewrite that changed
after its verification keeps the kernel out of the integration. A
`targets/<id>/rewrite.py` that exists before the refactor step (a resumed run,
or one written by hand) is verified first; the agent runs only when it fails.
The classes a rewrite defines are pickled by value, so the region's capture
loads in the evaluator and in the agent's own scripts without the rewrite on
the import path; a candidate can subclass the region class with
`from ka_region_<id> import Region_<id>` or `type(reference)`.

### Speed of light

Every timed `evaluate_candidate` result says how close each case is to the
hardware limit (`kernel_agent/kernels/roofline.py`).

* **Peaks** are measured on the GPU once per GPU model and torch version, in a
  subprocess under the GPU lock (never inside a timed evaluation), and cached
  in `~/.cache/kernel-agent/peaks-<gpu>-torch<version>.json`. They are copy
  bandwidth from DRAM (512 MiB buffers) and from L2 (counting bytes read plus
  bytes written), dense matmul TFLOP/s for bf16, fp16 and fp32 (best of a few
  large shapes, TF32 off), and the launch floor (a module call that launches
  one tiny kernel, timed like a candidate). `kernel-agent doctor` measures and
  prints them (`--remeasure-peaks` measures again). `toolchain.json` and the
  agents' prompts include them. On the RTX 5070 Ti: copy DRAM 767 GB/s, L2
  2970 GB/s, matmul bf16/fp16/fp32 99 / 94 / 34 TFLOP/s, launch floor ~16 µs.
* **Work** is counted on the reference call. FLOPs come from
  `torch.utils.flop_counter.FlopCounterMode`, split by the dtype of each op's
  inputs. Attention (SDPA) FLOPs count only the query/key pairs that the mask
  or `is_causal` allows. `min_bytes` counts every byte range of the inputs,
  parameters and buffers that the reference reads, each one once. Unused
  weights, intermediates and metadata-only uses do not count. Gather ops
  (embedding, index) count only the rows they read, and SDPA counts only the
  key/value positions that some query attends to. Outputs and new state (for
  example a concatenated cache) count once. Inputs updated in place count only
  the elements that changed, so a decode step over a static KV cache costs the
  slot it writes plus the valid positions, not the whole cache.
* Per case: `flops`, `min_bytes`, `sol_ms = max(Σ flops / peak, min_bytes /
  bandwidth)`, `pct_of_sol = 100 × sol_ms / new_ms` and `bound`. `bound` is
  `compute`, `memory`, or `launch` when `sol_ms` is below the launch floor
  (one launch from Python costs more than the work). Cases whose bytes fit in
  L2 are compared with the L2 bandwidth (`l2_resident`), because the benchmark
  reuses the same inputs and runs them with a warm cache. The result also has
  the weighted `pct_of_sol` (cases weighted by calls per run), `sol_ms_weighted`,
  the dominant `bound` and `launch_floor_ms`.
* `new_ms < 0.9 × sol_ms` is faster than the hardware allows. That case and the
  result get `suspicious_faster_than_sol`, which is a warning and not a
  rejection. If the reference itself beats 0.9 × `sol_ms`, the estimate is wrong
  and the case gets `sol_unreliable` instead.
* Limitations: the bytes are what the reference touches. Masked work that
  happens outside SDPA (for example a `repeat_kv` copy of a whole static cache)
  counts in full, so a kernel that skips it can legitimately beat the estimate.
  Such a result is flagged, never rejected. Strided views count their whole
  span. Element-wise math counts as free. The launch floor is measured for a
  torch op, and backend launch paths add their own overhead on top of it
  (CUDA C++ ~19 µs, Triton ~43 µs, see Backends).

### Budgets

All limits are off by default (`kernel_agent/budget.py`).

* `--max-hours` / `--max-usd` cover the whole run. Before each kernel or
  transform agent starts, the elapsed time and the sum of `costs.json` are
  checked. Once the budget is spent, the remaining agents are skipped, but
  integrate and report always run on whatever results exist. 15 % of
  `--max-hours` (`--budget-reserve`) is kept for them. Each agent's own USD cap
  (`--budget`) is lowered to what is left of `--max-usd`. Time counts from the
  start of the current `optimize`/`resume` process; USD counts the whole run.
* `--agent-minutes` stops an agent session after that many minutes. The Claude
  Code subprocess is terminated. Its session id, turns and tool calls are still
  written to `costs.json`. Its USD cost is not, because Claude Code reports cost
  only when a session ends.
* `--eval-timeout` (default 300 s) limits one `evaluate_candidate` subprocess.
  Results include `compile_s` (import, build and the first call, which is where
  JIT backends compile). A timeout result says whether it ran out of time while
  compiling or while checking and benchmarking.
* Agents get their remaining time and USD in the system prompt. Every
  `evaluate_candidate` / `evaluate_e2e` result has
  `budget: {evals_used, evals_budget, minutes_left, non_improving}` and an
  `advice`. The advice is `continue`, or `consider_stopping` after 4 evaluations
  in a row that did not beat the best result by more than max(1 %, 2 × timing
  spread), or `stop` when the evaluation, time or USD budget is spent. A
  correct kernel result whose weighted `pct_of_sol` is at least 90 % also gets
  `stop` ("within X % of speed of light"), unless it is flagged
  `suspicious_faster_than_sol` or `sol_unreliable` (see "Speed of light").
* Timeouts and budget stops are recorded under `run.json` → `phases.<phase>`
  (`timed_out`, `budget_skipped`), in `events.jsonl` and in `report.md`.

### program.md: steering the agents

You can steer the agents by editing a Markdown file instead of the code, as
in karpathy/autoresearch (`kernel_agent/program.py`). Each `## <section>` is
appended to the system prompt of the matching agents. `## all` goes to every
agent. `## planner`, `## kernel` (each `kernel-<target>` agent), `## systems`,
`## harness`, `## research` (each `research-<target>` session, see "Research
on plateaus") and `## refactor` (each `refactor-<target>` session of a region
target) go to that role only. A heading can name several roles
(`## kernel, systems`). Unknown sections are ignored with a warning. Text
above the first `##` heading and `<!-- comments -->` are not sent to agents.

* `kernel-agent program init [path]` writes the default program for editing.
  It covers measurement hygiene (absolute latencies, deltas below
  max(1 %, 2 × timing spread) are noise), one hypothesis per evaluation, the
  simplicity criterion, "abandoned after N attempts" instead of "X doesn't
  work", optimising inference rather than the benchmark, and a few rules per
  role.
* `--program FILE` on `optimize` / `analyze` copies FILE to `<run>/program.md`.
  Without the flag, the default is copied. `resume --program FILE` replaces
  the run's copy. A plain `resume` keeps it, including your edits.
* The run's `program.md` is read again before every agent session. Edits made
  during a run apply to the next agent that starts, while running agents keep
  the version they started with.
* Provenance: `costs.json` and the `agent_start` events in `events.jsonl`
  store `program_sha256` for each agent.
  `run.json` → `program` stores the source and every version in use (sha256,
  the first agent that used it, time). Each version is saved as
  `logs/program-<sha12>.md`.

### VoxCPM2

`openbmb/VoxCPM2` (and VoxCPM 1.x) is detected from the `"architecture"`
field of its `config.json` (family `voxcpm`) and runs through the built-in
workload in `kernel_agent/workloads/voxcpm.py`, so no harness agent is needed:

```bash
uv sync --extra all --extra voxcpm
uv pip install --no-deps "voxcpm>=2.0.3"
uv pip install --no-deps torchaudio --index-url https://download.pytorch.org/whl/cu130
uv run --no-sync kernel-agent analyze openbmb/VoxCPM2 --no-harness-agent
```

`voxcpm` is installed with `--no-deps` because its own requirements pin
`datasets<4`, gradio and funasr, and pull a `torchaudio` build that may not
match your torch. It imports `torchaudio` at import time, so install a
`torchaudio` from the same index as your torch (same CUDA variant), also with
`--no-deps`. Never let either replace torch. A later `uv sync` removes both
again; use `uv run --no-sync` afterwards.

The workload generates exactly `patches` latent patches (default 60, 9.6 s
of 48 kHz audio) with `retry_badcase=False`, and records every patch the
LocDiT sampler (`model.feat_decoder`) produces. Teacher forcing wraps
`forward` of the current `model.feat_decoder` instance at run time, so it
works after kernel replacements and with transforms that wrap `workload.run`.
The wrapper runs the sampler (same noise as the free run), records its
prediction and returns the reference patch to the loop. The audio of that run
is decoded from the reference latents, which also checks the AudioVAE decoder.
The defaults `min_mean_step_cosine=0.99` and `min_step_cosine=0.7` (plus a
±10 % RMS ratio) were calibrated at 60 patches on two texts. The table shows,
for each change, the value of the two texts that is closest to the threshold:

| change | mean step cosine | min step cosine | verdict |
|---|---|---|---|
| bundled `triton_rmsnorm.py` on every `MiniCPMRMSNorm` | 0.9993 | 0.976 | pass |
| fp32 RMSNorm, MATH-backend SDPA | 0.9983 | 0.927 | pass |
| every `nn.Linear` output × (1 ± 2^-8), 6 sign patterns | 0.9959 | 0.868 | pass |
| RMSNorm with eps=1e-2 | 0.70 | 0.10 | fail |
| RMSNorm without its weight | 0.26 | −0.06 | fail |
| attention softmax scale × 1.25 | 0.973 | 0.79 | fail |
| attention with one KV head dropped | 0.30 | −0.01 | fail |
| Snake1d (AudioVAE) ignoring alpha | 1.0 (decoded audio: spectral cosine 0.90) | 1.0 | fail |

Limitations: a candidate must still call `model.feat_decoder` from Python once
per patch and draw its noise with `torch.randn` like the original. A transform
that captures a whole step or the whole loop in one CUDA graph cannot be
teacher-forced and is rejected with a clear reason. Errors as small as one
bf16 rounding step per layer cannot be told apart from correct rounding
changes: an attention softmax scale off by 5 % reaches mean cosine 0.994 and
passes; the module-level check has to catch errors of that size. The model
runs in its checkpoint dtype (bf16), and `--dtype` is ignored.

Compiled baseline (`model.optimize()`, see "Strong baseline"), RTX 5070 Ti,
60 patches: eager 5,499 ms, compiled 3,798 ms (1.45×; 2 warm-up runs take
15.5 s), teacher-forced against the eager output with mean step cosine 0.9984
(min 0.958). On top of it the bundled Triton RMSNorm on all 124
`MiniCPMRMSNorm` instances survives `fullgraph=True` and passes (3,789 ms, no
gain: Inductor already fuses the norm); the `load_inline` CUDA RMSNorm breaks
it (Dynamo cannot trace the pybind function) until it is wrapped in a custom op.

### kernel-agent improve: the continuous loop

```bash
kernel-agent improve <run_dir | hf-url> [--max-hours 6] [--max-usd 60] [--slice 4] [--rounds 2]
kernel-agent improve Qwen/Qwen3-0.6B --dry-run     # simulated: no GPU, no Claude
```

`optimize` gives each target one agent session with a fixed evaluation
budget. `improve` keeps going, like autoresearch: it gives short sessions
("slices") to whichever target pays most, keeps or discards every result in
the ledger, re-integrates end to end as results come in, and stops only when
the budget is spent or every target has stopped (`kernel_agent/improve.py`,
`kernel_agent/scheduler.py`).

* **Start or continue.** With a Hugging Face URL it runs analyze, plan and
  capture first. With a run directory it continues that run, including one made
  by `optimize`. Ctrl-C leaves the run consistent: run the same command again
  and it continues. A slice that was running is recorded as `interrupted`
  together with the evaluations it made. `--max-hours` and `--max-usd` are the
  budget of this invocation: hours from now, and USD on top of what the run has
  spent already. Without them the loop runs until every target has stopped.
  `--agent-minutes` caps each slice.
* **Scheduler.** Every kernel target is an arm, and so is the systems agent
  (model-level transforms). The expected gain of an arm, in ms per model run,
  is `remaining_ms × headroom × 0.7^k`:
  * `remaining_ms`: the target's share of the profiled time × the baseline ms
    ÷ its best module speedup. For the systems agent it is the end-to-end time
    of its best run. A transform-only run is a new best when it beats the best
    transform-only run; transforms on top of kernels when they beat the best
    run measured with the same kernels (the integration, or the agent's previous
    best with them), so the kernels' own gain is never the systems agent's.
  * `headroom`: `1 − pct_of_sol` of the best result when the evaluator reports
    a speed-of-light estimate. Otherwise `1 − 1/further`, where `further =
    max(2 ÷ best, 1.1)` is the speedup still assumed possible. For the systems
    agent, `further` starts at 1 ÷ the GPU-busy share of the profile (at least
    1.25).
  * `k`: slices of this arm in a row that found no new best.

  The arm with the highest `expected gain × UCB index` gets the next slice. The
  index is the arm's observed gain per evaluation (ms saved by its kept results
  ÷ its evaluations, relative to the best arm's) plus `sqrt(2 ln(N + 2) / (n +
  1))`, as in KernelBand. Untried arms get explored, and arms that keep paying
  get more slices.
* **Slices.** Each slice is a fresh agent session with `--slice` evaluations
  (the evaluation advice says `stop` after them), so no context grows. It is
  seeded with a digest in the system prompt: the arm's last 15 ledger rows with
  ideas, hypotheses and status, the per-idea table (see "Experiment ledger"),
  the best snapshot with its speedup (and % of SOL), the research `plan.md` if
  there is one, the `## Open ideas` section and the tail of `NOTES.md`, and how
  far the target is from its stop rules. The agent is asked to keep `NOTES.md`
  and its open ideas current for the next session. Slices run through the same
  code as `optimize` agents (budgets, timeouts, `program.md`, events). Their
  cost is in `costs.json` as `kernel-<target>#<slice>` and `systems#<slice>`.
* **Stop rules per arm** (AutoKernel's move-on rules): `--patience 5`
  evaluations in a row without a new best, across slices; `--sol-stop 0.9` of
  the speed of light; `--target-hours 2` spent in its slices; the module
  `--speedup-goal 2` reached. `0` turns a rule off. An arm whose last two slices
  made no evaluation stops too. The loop ends when the budget is spent or every
  arm has stopped, or after 3 agent sessions in a row failed.
* **Research on plateaus** (`kernel_agent/research.py`, auto-gpu-kernel's
  research subagent). When a kernel target has plateaued, i.e. 4 evaluations in
  a row without a new best (the advice turns `consider_stopping`), `--patience`
  reached, or 3 failed evaluations in a row, and no other stop rule applies, the
  loop runs a `research-<target>` session before the target's next slice, or
  before the patience rule stops it. The session starts from a clean context
  with read-only tools (`Read`, `Glob`, `Grep`, `best_result`, web if allowed).
  It may write one file, `targets/<id>/plan.md`, which a `PreToolUse` hook
  enforces. It gets the target's ledger rows with ideas and statuses, the
  per-idea table, the best result with `pct_of_sol` and `bound` per case, and
  pointers to `NOTES.md`, `workload_profile.md`, the snapshots and the previous
  plan. It writes a diagnosis along a 9-item pathology checklist (repetition
  loop, local minimum, correctness wall, wrong bottleneck, missing fundamental,
  over-engineering, ignored prior research, host overhead, overlooked
  shortcuts), directions ranked by their ceiling, ideas to retry (failed, not
  refuted) and a do-not-try list. The next slice of the target is a fresh
  session with the plan in its digest. A written plan restarts the target's
  count of evaluations without a new best. A target gets at most one session
  per `--research-every 3` of its slices (`0`: never), and none again when its
  last plan brought no new best: the target then stops at its next plateau.
  The sessions go through the same code as the slices (budgets, `program.md`
  `## research`, events `research_start` / `research_done`, `costs.json` as
  `research-<target>#<slice>`). They are recorded in `improve.json` →
  `research`, shown as diamonds in `improve.png` and listed in `report.md`.
* **Re-integration.** After every `--integrate-every 4` kept results, the
  integration of `optimize` measures the combination end to end. Combinations of
  the same snapshot files that were measured before are reused instead of
  measured again. Every measurement is a ledger row, so the progress chart shows
  the measured latency going down. `optimized/` is re-exported each time.
* **Rounds.** With `--rounds R` above 1, once every arm of a round has stopped
  and the round brought a real end-to-end gain, the optimised model (the
  accepted integration applied) is profiled again into `rounds/<n>/`
  (`worker analyze --out-dir D --kernel ... --transform ...`). The planner then
  proposes new targets, with the earlier rounds and the existing targets as
  context, and the loop continues with them. A target whose module was replaced
  in the re-profile keeps the share it had in the first profile.
* **Files.** `improve.json` holds the slices (arm, scores, evaluations,
  outcome), the research sessions, the re-integrations, the rounds and why the
  loop stopped. `improve.png` is drawn from it, and `report.md` gets an
  "Improve loop" section.
* **Dry run.** `--dry-run` replaces Claude and the GPU with a simulated
  Qwen3-0.6B decode workload (`kernel_agent/dryrun.py`). Targets approach a
  hidden ceiling with noise, failures and plateaus. Some report a speed-of-light
  estimate and some do not. The simulated engineer tags every candidate with an
  `idea_id`, retries an idea once after a failed attempt and starts from the
  directions of a research plan. The simulated research agent writes `plan.md`
  from the ledger, so plateaus exercise the research trigger and its cap (the
  simulated outcomes do not depend on the plan). The CUDA graph the systems
  agent finds is incompatible with the MLP kernel, and round 2 finds a new
  target. Time is simulated too, so `--max-hours` counts simulated hours. The
  images below come from `kernel-agent improve Qwen/Qwen3-0.6B --dry-run
  --rounds 2`.

![improve progress](docs/images/example-improve-progress.png)

`improve.png` has one lane per arm and a bar per slice: green when the slice
found a new best, grey when it did not. Dashed lines are the re-integrations,
labelled with the measured end-to-end speedup. Diamonds are research sessions
that wrote a plan.

![improve slices](docs/images/example-improve-slices.png)

Limits: the evaluation budget of a slice is advice to the agent. The hard caps
are `--agent-minutes` and the run budgets. Targets found by a re-plan are
captured from the unmodified model.

### Parallel workers, duplicates and quick checks

* **Workers** (`kernel_agent/workers.py`). `--seeds-per-target K` gives every
  target K isolated workers; `auto` gives 2 to targets with at least 20 % of the
  profiled time and 1 to the rest (default: 1, one session as before). Each
  worker is its own agent session (`kernel-<id>-w<k>`) in
  `targets/<id>/workers/<k>/` with its own `candidates/` and `NOTES.md`; the
  target's shared files (`spec.json`, `reference_source.py`, `history/`,
  `results.jsonl`, ...) are links there. Worker 1 starts from the planner's
  approach, worker k from the planner's `alternatives[k-2]` (`approach` +
  `backends`, an optional field of the plan) or else from the next backend
  (KernelFalcon's seeded workers, GEAK's workspace per agent). All workers of a
  target share its verified evaluation store and the GPU lock, so
  `best_for_target`, `best_result` and `best_so_far` are the best across workers
  and the integration takes it. Ledger rows, records and evaluation events carry
  the `worker`. The target's evaluation budget stays the same and is split across
  its workers (`--evaluations 12`, 2 workers: 6 each; an improve slice of 4: 2
  each), because serial refinement beats parallel sampling at a fixed budget
  (Kevin). Worker sessions run concurrently up to `--parallel`, in the `kernels`
  phase and in a slice of `improve`; a resumed `kernels` phase skips finished
  worker sessions. `--reseed-workers` adds round 2: after the first worker
  sessions of a target, the next ones start from its two best snapshots (with
  half of the budget; in `improve`, every slice after the target's first).
* **Duplicates** (`kernel_agent/dedup.py`). Before it takes the GPU lock,
  `evaluate_candidate` hashes the candidate's source normalised by
  `ast.unparse(ast.parse(...))` (comments, blank lines and formatting do not
  count) and looks it up in the target's verified records. A match is not
  evaluated again: the result is the earlier one, labelled `duplicate of
  history/NNN_...py (exp N)`, the ledger gets a `duplicate` row, and no budget,
  streak or idea counts it. A full evaluation reuses only full results; results
  that may not repeat (`timeout`, `crash`, tampered) and calls that ask for more
  than the record has (`profile=true`, `compile_check=true`) run again. Two
  workers submitting the same source at the same time evaluate it once.
* **Quick checks.** `evaluate_candidate(mode="quick")` (and `kernel-agent eval
  --quick`) checks correctness on the smallest and the largest captured case
  only, untimed. The result says it is not a benchmark, has no speedup and is
  not counted against the evaluation budget; it is recorded in
  `.truth/targets/<id>/quick.jsonl` and in the ledger as `quick_ok` /
  `quick_fail`, which the charts, the scheduler, the streaks and the idea
  aggregates ignore. A full evaluation of a quick-checked source still runs; a
  quick check of an evaluated source returns the full result as a duplicate.
* **Views.** `status` shows `<target>/w<k>` in the latest rows and the number of
  workers per target; a target's `progress.png` has one marker shape per
  worker; `kernel-agent watch` shows the worker in the ledger, the events and
  the tooltips, and quick checks and duplicates as neutral rows.

### GPUs and the GPU lock

Every evaluation, worker command (`analyze`, `capture`, `e2e`) and peak
measurement holds the lock of one GPU while its subprocess runs
(`kernel_agent/gpulock.py`).

* **Pool.** The GPUs `nvidia-smi --query-gpu=index,name,uuid` lists (the
  orchestrator never initialises CUDA to find them), restricted to an inherited
  `CUDA_VISIBLE_DEVICES` (indices or `GPU-` UUIDs, in its order).
  `KERNEL_AGENT_GPUS=0,2` replaces both. Indices are `nvidia-smi`'s (PCI bus
  order, which is CUDA's order with `CUDA_DEVICE_ORDER=PCI_BUS_ID`).
  `kernel-agent doctor` prints the pool and its lock files.
* **Locks.** GPU `i` locks `~/.cache/kernel-agent/gpu<i>.lock` (an `flock`
  across processes, plus a thread lock for the agents of one process). GPU 0
  keeps `gpu.lock`, the lock file from before the pool, so an older
  kernel-agent (or a `flock ~/.cache/kernel-agent/gpu.lock ...` wrapper) and a
  new one still exclude each other on GPU 0, the only GPU of a single-GPU
  machine. The lock takes the first free GPU, else waits for the GPU with the
  fewest waiters (and stays with it); a nested lock in the same thread keeps
  its GPU. Without `nvidia-smi`, or with no GPU visible,
  the pool is GPU 0 alone (`gpu.lock`), as before.
* **Child processes.** A subprocess started under the lock gets
  `KERNEL_AGENT_LOCK_HELD=1` (it does not wait for its parent) and, when more
  than one GPU is visible, `CUDA_VISIBLE_DEVICES=<i>` with
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so it sees the locked GPU alone. With a
  single visible GPU its environment is the same as before.
* **Results** carry `gpu_index`, in the targets' and the transforms'
  `results.jsonl`. A candidate and its reference always run on the same GPU
  (one subprocess). With N GPUs, `--parallel N` agents evaluate at the same
  time, one GPU each.
* **Mixed GPU models.** Roofline peaks are cached per GPU model, measured on
  the GPU the lock hands out and named after the orchestrator's first visible
  GPU. The e2e speedups divide by the baseline that `analyze` measured once,
  on one GPU. Both assume identical GPUs: on a mixed machine, restrict the pool
  to one model (`KERNEL_AGENT_GPUS` or `CUDA_VISIBLE_DEVICES`); `doctor` says
  when the pool mixes models.
* `gpu` tests hold the lock of the GPU the test process uses. With several
  GPUs, the test session runs on the first GPU of the pool.

### Kernel library and lessons (memory across runs)

Runs used to forget everything. Now the kernels that won are kept per GPU
architecture, and short lessons are distilled from the agents' notes, so the
next run starts from them (AccelOpt, KernelBlaster and AdaExplore: slow→fast
examples and validity rules; FlashInfer-Bench: a stored definition, solution
and evaluation per kernel). The code is in `kernel_agent/library.py`.

```
~/.cache/kernel-agent/library/          ($KERNEL_AGENT_LIBRARY overrides it)
  <sm_arch>/<module_class>/<entry-id>/  kernel.py  spec.json  result.json  NOTES.md  entry.json
  lessons/<backend>.md  lessons/<module_family>.md
```

* **Store.** After every integration (`optimize` and each re-integration of
  `improve`), kernel-agent stores two kinds of kernels. The first is the
  verified best kernel of every target with at least `min_speedup` (1.03×),
  flagged `module_winner`. The second is every kernel the integration accepted
  end to end, flagged `accepted` (with the measured e2e speedup). `entry.json`
  holds the model repo, torch version, GPU, dates and the module speedup and %
  of SOL. It also holds a signature summary (entrypoints, dtypes, shapes of the
  captured cases) and the sha256 of each of the other files. The entry id is
  `<signature hash>-<code hash>`, so storing the same kernel again for the same
  shapes updates the entry. `runs` lists every run that stored it, and `origin`
  points to the entry it was first reused from.
* **Reuse.** Library entries are evaluated before a target's first agent
  session: at the start of the `optimize` kernels phase, and in `improve`
  before the scheduler picks a slice. This costs no LLM calls. An entry
  matches when it has the same module class and GPU architecture, implements
  the target's entrypoints (`forward_step`, ...) and was verified on the
  target's dtypes. Shapes may differ, because kernels read their sizes from the
  module. Up to 3 matches are tried, closest shapes first. Each one is copied
  to `candidates/prior_<entry-id>.py` and evaluated through the normal
  truth-verified path (`capture_sha256`, snapshot, `results.jsonl`). Its ledger
  row has the hypothesis `prior winner from <repo> (library entry <id>, 1.50x
  there)`. A prior can set the target's best, but it does not count as an
  attempt without a gain: neither the `consider_stopping` advice nor
  `--patience` counts it. When a prior already reaches 90 % of its speed of
  light, `optimize` skips the target's agent, and `improve` stops the arm.
* **Prompt.** The engineer prompt lists the evaluated priors slow → fast, with
  their numbers in the run that stored them and on this target. It then gives
  the lessons for the target's module family and backends, at most 1,500
  characters of whole rules:

  ```
  # Prior kernels from the library
  1. `candidates/prior_b5dee5cf-cebbc64f.py` (triton, from `HuggingFaceTB/SmolLM2-135M-Instruct`):
     1.67x at 0.017 % of SOL there; here 1.68x at 0.018 % of SOL,
     `history/001_prior_b5dee5cf-cebbc64f_996c6f7d.py`. Approach: bundled Triton example:
     one program per row, fp32 accumulation
  ```

  That is from a run on SmolLM2-135M with batch 2 and a 128-token prompt. The
  kernel came from an earlier run with batch 1 and a 256-token prompt. The
  reused kernel was then accepted end to end (487.4 → 454.7 ms), and no agent
  wrote a line of it.
* **Librarian.** After the report, a cheap agent distils the run's `NOTES.md`
  files and ledger rows (status, speedup, % of SOL, hypothesis) into short
  rules, such as "do X when Y" or "Z fails because W (abandoned after N
  attempts)". It merges them into `lessons/<backend>.md` and
  `lessons/<module_family>.md` (`norm`, `attention`, `mlp`, `rope`, ...),
  dropping duplicates and contradicted rules. It runs through the same code as
  the other agents (budgets, `program.md` `## all`, events, `costs.json` →
  `librarian`). It uses `--librarian-model` (default `--claude-model`) at
  effort `low` and answers with structured output only. kernel-agent writes the
  files itself: only names of this run's backends and module families, at most
  20 rules each. The librarian runs again only when the ledger has new rows.
* **Safety.** Entries are code that will run. An entry is reused only when
  every file still has its recorded sha256. Otherwise a `LIBRARY: ... refused`
  line and a `library_rejected` event are emitted. Entries of another GPU
  architecture are never read: only `<this sm_arch>/` is searched, and each
  entry's own `sm_arch` must match. A reused kernel passes the same correctness
  checks as any candidate before it counts. The digests are tamper evidence,
  not a signature, because the library is as reachable from an agent's Bash
  tool as a run directory is.
* `--no-library` turns off reading and writing the library, and
  `--no-librarian` skips the lessons agent. Without a GPU (`improve --dry-run`),
  the library is off. `run.json` → `library` records the priors tried, the
  entries stored and the librarian's files, and `report.md` has a "Kernel
  library" section.

```bash
kernel-agent library list [--arch sm_120] [--module-class LlamaRMSNorm]
kernel-agent library show <id or prefix>       # metadata, integrity, cases, runs, notes
kernel-agent library prune [--older-than DAYS] [--dry-run]   # broken (and old) entries
kernel-agent library path
```

## Backends

| backend | how | host overhead (tiny op, measured) |
|---|---|---|
| `cuda` | CUDA C++ via `torch.utils.cpp_extension.load_inline` (nvcc) | ~19 µs |
| `cute` | CuTe DSL (`nvidia-cutlass-dsl`), TVM-FFI calling convention | ~25 µs |
| `nvrtc` | CUDA C++ compiled at runtime with NVRTC (`cuda.core`) | ~28 µs |
| `tilelang` | TileLang (`tilelang.language`) | ~32 µs |
| `triton` | Triton | ~43 µs |

There are verified example kernels for every backend in
`src/kernel_agent/agent/examples/`, and backend guides plus an optimisation
playbook in `src/kernel_agent/agent/knowledge/`. Both are fed to the agents.

**No system CUDA toolkit needed.** If `nvcc` is missing, the pip wheels
(`nvidia-cuda-nvcc`, `nvidia-cuda-cccl`, ...) are assembled into a
`CUDA_HOME` shim with the unversioned `libcudart.so` the linker needs.
A host GCC that is too new and nvcc/CUDA-header version skew are handled
through `NVCC_APPEND_FLAGS` (`kernel_agent/toolchain.py`).

## CLI

```bash
kernel-agent optimize <hf-url> [options]
  --modality {llm,stt,tts,diffusion}   override auto-detection
  --dtype bfloat16|float16|float32
  -o KEY=VALUE                         workload options, e.g.
                                       LLM: prompt_len, new_tokens, batch_size, min_prefix
                                       STT: audio=/path.wav, audio_seconds, new_tokens
                                       TTS: text, seed, min_spec_cosine
                                       diffusion: steps, height, width, prompt, cpu_offload, min_psnr
                                       any: entrypoints=Cls.method,... (extra non-forward methods)
                                       VoxCPM: text, patches, timesteps, cfg, seed, compile,
                                               min_step_cosine, min_mean_step_cosine, min_spec_cosine
  --backends cuda,triton,cute,tilelang,nvrtc
  --max-targets 4 --evaluations 12     targets and evaluation budget per target
  --parallel 2                         kernel agents at the same time
  --seeds-per-target 2|auto            isolated workers per target, budget split across them
  --reseed-workers                     round 2 of the workers from the two best snapshots
  --no-transforms                      kernels only
  --claude-model claude-opus-5-5 --effort high --budget 10 (USD per agent)
  --max-hours 3 --max-usd 40           budget for the whole run (see "Budgets")
  --agent-minutes 45                   time limit per agent session
  --eval-timeout 300                   seconds per evaluate_candidate
  --budget-reserve 0.15                share of --max-hours kept for integrate + report
  --harness my_harness.py              your own workload
  --program my_program.md              instructions for the agents (see "program.md")
  --until analyze|plan|capture|kernels|transforms|integrate
  --compile-baseline                   also measure a generic torch.compile baseline for
                                       workloads without reference_optimizations()
  --no-library --no-librarian          cross-run kernel library / lessons agent off
  --librarian-model MODEL              (see "Kernel library and lessons")

kernel-agent analyze <hf-url>          baseline + profile only (no Claude)
kernel-agent improve <run_dir | hf-url> [--max-hours H] [--max-usd U] [--slice 4] [--rounds R]
  --integrate-every 4 --patience 5 --sol-stop 0.9 --target-hours 2 --speedup-goal 2
  --max-slices N --dry-run [--seed 0]  continuous loop (see "kernel-agent improve")
kernel-agent resume <run_dir> [--redo kernels] [--program FILE]
kernel-agent program init [path]       write the default program.md for editing
kernel-agent eval capture.pt candidate.py [--profile] [--compile-baseline] [--compile-check]
                                       [--quick] [--timeout 300]
                                       (a full capture, e.g. <run_dir>/.truth/captures/<id>.pt)
kernel-agent report <run_dir>          report.md + charts + dashboard.html
kernel-agent status <run_dir> [--watch 10]   per-target progress, e2e, cost, last evaluations
kernel-agent watch <run_dir> [--port 8765]   live dashboard in the browser (see "Live dashboard")
kernel-agent library list|show <id>|prune [--older-than DAYS]|path   cross-run kernel library
kernel-agent doctor [--smoke] [--remeasure-peaks]
kernel-agent install-claude-code <project-dir>
```

Python API:

```python
import asyncio
from kernel_agent import OptimizeConfig, optimize

run = asyncio.run(
    optimize(
        OptimizeConfig(
            model_ref="https://huggingface.co/openai/whisper-large-v3-turbo",
            backends=["cuda", "triton"],
            max_targets=3,
        )
    )
)
print(run.report.read_text())
```

## Run directory

```
runs/<org>--<name>/<timestamp>/
  run.json  toolchain.json
  baseline.json               eager baseline (+ compiled_ms / compiled_detail, see "Strong baseline")
  program.md                  agent instructions; edit it mid-run to steer the agents
  profile/summary.md          profile handed to the planner
  plan.json                   targets + transforms
  .truth/                     what the evaluator trusts (read-only, sha256 in run.json):
    baseline_output.pt          output of the baseline run
    baseline_output_holdout.pt  ... of the held-out input
    captures/<id>.pt            module + real inputs/outputs + post-call state
    captures/<id>.parent.pt     region target: the capture of its parent class
    targets/<id>/history/       snapshot of every evaluated version (region: + rewrite.py)
    targets/<id>/results.jsonl  every evaluation (full record + snapshot sha256)
    targets/<id>/quick.jsonl    every quick check (mode="quick", not a benchmark)
    transforms/                 history/ + results.jsonl of the transforms
  targets/<id>/capture_inputs.pt  module + real inputs (no outputs), for the agent
  targets/<id>/workload_profile.md  statistics of every call of the target (+ .json)
  targets/<id>/reference_source.py
  targets/<id>/candidates/    files the agent writes
  targets/<id>/workers/<k>/   --seeds-per-target: a worker's candidates/ + NOTES.md (+ links)
  targets/<id>/plan.md        improve: the research plan of a plateaued target
  targets/<id>/rewrite.py     region target: the refactor agent's rewrite (+ parent/: its
                              parent's capture_inputs.pt, reference_source.py, profile)
  targets/<id>/history/ results.jsonl   the agent's copies (never read back)
  targets/<id>/progress.png   speedup per evaluation (see "Charts")
  transforms/                 model-level transforms (+ the agent's copies)
  results.tsv                 experiment ledger: one row per evaluation
  events.jsonl                phase changes, agent start/stop, evaluations
  progress.png  amdahl.png  integration.png  dashboard.html
  integration.json  report.md  logs/  (incl. logs/program-<sha12>.md)
  improve.json  improve.png   improve loop: slices, research sessions, re-integrations, rounds
  rounds/<n>/                 re-profile (baseline.json, profile/) + plan.json of round n
  costs.json                  per agent: $, turns, minutes, tools, session_id, program_sha256
  optimized/                  apply.py + manifest.json + kernels/ (+ rewrites/ of region targets)
```

To use the result in your own code:

```python
import sys

sys.path.insert(0, "runs/.../optimized")
from apply import apply_kernels

apply_kernels(model)  # replaces every matching module instance
```

## Experiment ledger

Every evaluation of a run is one row of `results.tsv`: kernel candidates,
model-level transforms and integration steps (`target = e2e`).

```
exp  time  target  backend  snapshot  parent  status  correct  speedup  ref_ms  new_ms
est_saved_ms  spread  pct_of_sol  eval_s  worker  idea  hypothesis
```

`pct_of_sol` is the weighted share of the speed of light for kernel rows (see
"Speed of light"). `worker` is the target's worker that evaluated a kernel
candidate (empty without workers). `idea` is the `idea_id` of a kernel candidate. A ledger
written before a column existed keeps its own layout, and its rows have no
value for that column.

* `evaluate_candidate` requires a `hypothesis` (one sentence: what changed and
  why it should be faster) and accepts an optional `parent` snapshot.
  `evaluate_e2e` takes an optional `hypothesis`.
* **Idea ledger** (Stanford's NL-level branching, K-Search's split of strategy
  from implementation). `evaluate_candidate` also accepts `idea_id`, a short
  slug that names the idea a candidate implements (the same id for every
  attempt and fix of it), and `expected_speedup`. Both go into the record and
  `idea` into the ledger. The result's `idea` shows expected next to measured,
  the idea's tries so far and its verdict. A failed attempt gets a note that it
  is a bug to fix, not evidence against the idea. `best_result` aggregates per
  idea: tries, best speedup, `kept`, `slow` (correct, not a new best), `bugs`
  (failed) with their statuses, the expected speedup, the last hypothesis and a
  verdict (`kept`, `slow`, or `buggy`: never correct, so untested rather than
  refuted). The engineer prompt and `program.md` ask for 3-5 distinct ideas
  with expected gain and ceiling before any code, a retry of a buggy idea before
  it is dropped, and "abandoned after N attempts: <why>" instead of "X doesn't
  work". The CUDA and CuTe DSL guides describe plan amnesia and false
  infeasibility. `status`, `dashboard.html` and `kernel-agent watch` show the
  idea next to the hypothesis.
* `status` is `keep` when the result is correct and beats the best kept result
  by more than the noise: speedup > best × (1 + max(1 %, 2 × timing spread)).
  Kernels start from the reference module (1.0×), `e2e` rows from the baseline.
  A correct result that is not better is `discard`. Failures are `incorrect`,
  `build_error`, `runtime_error`, `crash` or `timeout`. The keep rule is the
  same one the budget advice uses. `quick_ok` / `quick_fail` (quick checks) and
  `duplicate` rows are not benchmark evaluations: no chart, count, budget or
  streak uses them (see "Parallel workers, duplicates and quick checks").
* `backend` is read from the candidate's imports (`load_inline` → `cuda`,
  `cuda.core` → `nvrtc`, `cutlass` → `cute`, `tilelang`, `triton`; `torch` when
  there is no custom kernel).
* `results.jsonl` (in `.truth/`) still has the full records (cases, errors)
  plus `exp`, `ledger_status`, `hypothesis` and the snapshot's sha256. For
  runs that predate the ledger, the rows are rebuilt from the `results.jsonl`
  files.

`kernel-agent status <run_dir> [--watch SECONDS]` prints the current phase, the
baseline, the projected and measured end-to-end latency, the total cost from
`costs.json`, a table per target (evaluations, keeps, failures, best speedup,
its % of speed of light, estimated ms saved, last hypothesis) and the last 10
ledger rows.

`dashboard.html` in the run directory is self-contained: charts inlined as
PNG, the target table, the latest evaluations and the agent costs. It supports
light and dark mode and reloads every 30 s while the run is going.

## Charts

The charts need matplotlib, which is in the optional `viz` extra:
`uv sync --extra viz` (`all` includes it). Without matplotlib nothing is
drawn, and the ledger, `status` and the dashboard tables still work. The charts
and `dashboard.html` are redrawn after every evaluation and phase, and by
`kernel-agent report`. `report.md` embeds them. The colours mean the same thing
in every chart: green = kept, grey = discarded, red = failed.

`progress.png`: end-to-end latency over wall-clock time. The blue step line
is the projection from the best kernels (baseline − Σ est. saved ms of each
target's best kept candidate). Diamonds are measured end-to-end runs
(transforms and integration steps). Dashed lines mark the baseline and, when it
is known, the `torch.compile` baseline. The shaded bands are the pipeline phases.

![run progress](docs/images/example-progress.png)

`targets/<id>/progress.png`: module speedup per evaluation. Each kept
candidate is labelled with its hypothesis. Failures sit on the floor. The
step line is the running best, and the dashed line is the reference module.

![target progress](docs/images/example-target-progress.png)

`amdahl.png`: the baseline time split by each target's share of the module
profile, then the same bar with every target at its best module speedup
(Amdahl's law), then the measured integrated result.

![time split](docs/images/example-amdahl.png)

`integration.png`: the greedy integration as a waterfall. It starts at the
baseline, then the best single item, then each item added on top: accepted
(green, ms saved), rejected for no gain (hatched) or failed (red ×).

The candidates are every kernel winner and every transform of a passing
`evaluate_e2e` record that beat the baseline, alone or combined with other
transforms and kernels (the systems agent evaluates on top of the kernel
winners); of each transform idea (file stem, without the `NNN_` / `_<sha>`
of history snapshots) the version of the fastest such record. The fastest
passing combination the systems agent measured is measured again as a whole
(`integration.json` → `composite`: its `items`, `exp` and whether it
`seeded`); when it beats the best single item it seeds the search instead
(`exp N combination` in the waterfall), and items that are already part of it,
or another version of one, are not added again.

![integration waterfall](docs/images/example-integration.png)

The example images come from the synthetic run in `tests/synthetic_run.py`
(`python tests/synthetic_run.py /tmp/demo` writes one and draws its charts).

## Live dashboard

`kernel-agent watch <run_dir> [--port 8765] [--host 127.0.0.1]` serves a live
view of a run on http://127.0.0.1:8765. It needs only the standard library
(`http.server` + Server-Sent Events) and only reads the run directory, so it
can be started before, during or after a run. Every second it reads what was
appended to `results.tsv`, `events.jsonl` and `logs/agent-*.jsonl` since the
last byte offset (a half-written last line waits for the next poll) and
re-reads `costs.json`, `integration.json` and `baseline.json` when they
change. The page updates without reloading.

The page shows the model, GPU, eager and `torch.compile` baselines, the best
measured and projected end-to-end latency, elapsed time, USD spent, the
pipeline phases and the agents that are running. Below that: a
speedup-per-evaluation chart per target (running best as a step line,
hypothesis on hover), projected and measured end-to-end latency over
wall-clock time, cumulative agent cost, the integration waterfall, the latest
events and agent tool calls, the ledger (filter by target and status, search
the hypotheses) and a read-only file browser for candidates, snapshots,
`NOTES.md` and `program.md`. The charts are inline SVG drawn in the browser
(no CDN, works offline) in the same colours as the PNG charts. Light and dark
mode follow the system (`?theme=dark` forces dark), tooltips also work from
the keyboard (focus a chart, then use the arrow keys), and the layout works at
phone width.

![live dashboard](docs/images/example-watch.png)

Endpoints: `/` the page, `/api/state` a JSON snapshot, `/events` the SSE
stream (a `state` event on connect, then `delta` events with new ledger rows,
events, agent-log lines and the updated summary), `/api/files` and
`/file?path=` (text files inside the run directory only, at most 512 KB). The
server binds to localhost by default and rejects requests whose `Host` header
is not a local name. `--host 0.0.0.0` exposes the run, read-only, to your
network.

## Candidate contract

```python
def build(reference: torch.nn.Module) -> torch.nn.Module:
    """Return a drop-in replacement for `reference`: same forward signature,
    same outputs, same in-place side effects, sharing its weights. Return
    `reference` for instances the kernel does not support."""
```

If the model also calls the module through another entrypoint (the capture's
cases say `"method": "forward_step"`), the replacement must implement that
method too, with the same signature and side effects. Subclassing the
reference class keeps the methods a kernel does not touch:

```python
class Fast(MiniCPMAttention):
    def forward_step(self, hidden_states, position_emb, position_id, kv_cache): ...


def build(reference):
    new = copy.copy(reference)  # shares the weights
    new.__class__ = Fast
    return new
```

To keep a kernel working when the model runs under `torch.compile` (the
compiled baseline), register its launcher as a custom op
(`examples/triton_rmsnorm_custom_op.py`) and check it with `--compile-check`.

## Harness contract (custom models)

`harness.py` defines `create(spec) -> Workload`. Implement `load`, `roots`,
`make_inputs`, `run` and `compare` (`kernel_agent/workloads/base.py`). If the
inference loop calls module methods whose names the entrypoint pattern misses,
list them in the class attribute `entrypoints = {"ClassName": ["method"]}`.

For chaotic autoregressive models, also set `supports_teacher_forcing = True`
(and `chaotic = True`) and implement `run_teacher_forced(inputs, reference)`
and `compare_teacher_forced(reference, candidate)`. `reference` is the
baseline output of `run()`, so `run()` must record the per-step trajectory.
`compare_steps` in `base.py` is the per-step comparison, and
`workloads/voxcpm.py` is a complete example.

Optionally implement `reference_optimizations()` (apply the model's own fast
path in place, return a one-line description, or `None`) so that `analyze`
measures a compiled baseline (see "Strong baseline").

`holdout_options(variant)` returns option overrides for the held-out input
(variant 1: other content, the same shapes where possible) and the
memoisation probe's fresh inputs (variants >= 2: the shapes of variant 1,
other content); the default changes the `seed` option when there is one.
They apply to `self.options` around `make_inputs`, `run` and teacher forcing,
so read prompts, texts and seeds from `self.options`. `variants()` returns
option overrides of extra settings (other lengths, batch sizes) whose new
shapes `capture` records as correctness-only cases.

## Using it interactively from Claude Code

`kernel-agent install-claude-code <project>` copies a `/optimize-model`
slash command and a `kernel-engineer` subagent into `<project>/.claude/`. You
can then drive the same tools (`kernel-agent analyze/eval/...`) from an
interactive Claude Code session instead of the autonomous pipeline.

## Authentication and safety

The agents run through the Claude Agent SDK and use your Claude Code login
or `ANTHROPIC_API_KEY`. By default they run with `bypassPermissions` inside
the run directory, because they need to compile and run code without prompts.
Use `--permission-mode acceptEdits` for a stricter setup. Agents never
install or change torch/CUDA packages.

## Development

```bash
uv sync --extra all --group dev
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest -q
```

GPU tests are marked `gpu` and skipped when no CUDA device is present.
