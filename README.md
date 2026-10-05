# kernel-agent

Give it a Hugging Face URL; it profiles the model on your GPU, finds the slow
parts, and has Claude write custom GPU kernels for them in **CUDA C++,
CuTe DSL, Triton or TileLang**. It also tries model-level algorithm changes
(static caches, CUDA graphs, merged projections). Each candidate is checked
for correctness against real captured inputs, benchmarked, and kept only if
the whole model still produces the same output and runs faster.

Works on **LLM**, **STT**, **TTS** and **diffusion** models. When no built-in
workload can run a model (custom TTS stacks, for example), Claude writes a
benchmark harness for it first.

```bash
uv sync --extra all
uv run kernel-agent doctor --smoke      # check GPU, compilers, all 5 backends
uv run kernel-agent optimize https://huggingface.co/Qwen/Qwen3-0.6B
```

## How it works

```
HF URL ─► resolve (modality, arch, size)
        ─► analyze   load model, baseline latency, determinism check,
                     module-level + kernel-level profile           [GPU worker]
        ─► (harness) Claude writes harness.py if the built-in workload fails
        ─► plan      Claude reads the profile + source, picks target modules,
                     an approach and backends for each, and model transforms
        ─► capture   each target module is saved with real inputs/outputs
                     (prefill + decode shapes, KV-cache side effects)  [GPU worker]
        ─► kernels   one Claude "kernel engineer" per target writes candidates,
                     calls evaluate_candidate (correctness + interleaved
                     benchmark + optional per-kernel profile) and iterates
        ─► transforms  Claude "systems engineer" writes model-level transforms,
                     validated end to end with evaluate_e2e
        ─► integrate all winners are applied together, validated against the
                     baseline output; greedy fallback if the combination fails
        ─► report    report.md + optimized/ (kernels + apply.py)
```

All GPU work runs in subprocesses under a GPU lock, so a crashing kernel or an
illegal memory access can't kill the run, and parallel agents
(`--parallel N`) never benchmark at the same time.

### What "correct" means

* **Module level** (`evaluate_candidate`): every captured case must match the
  reference outputs **and** the in-place side effects (for example KV-cache
  appends) within dtype-aware tolerances (bf16 2e-2, fp16 1e-2, fp32 1e-4;
  at most 0.1 % of elements outside). If `build()` hands back the reference
  module unchanged, the candidate is rejected.
* **Model level** (`e2e`): the workload's own comparison. For LLM/STT that is
  identical greedy tokens for the first N tokens plus first-step logits cosine
  ≥ 0.99. For TTS it is spectral cosine. For diffusion it is PSNR ≥ 25 dB on
  the same seed.

### What "faster" means

Reference and candidate are timed in alternating rounds with CUDA events
after a GPU warm-up, and the median round is reported. Mutable inputs (caches)
are deep-copied outside the timed region. A module's speedup is weighted by how
often each captured shape runs per inference. The end-to-end speedup is
wall-clock latency of the whole workload.

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
  spread), or `stop` when the evaluation, time or USD budget is spent.
* Timeouts and budget stops are recorded under `run.json` → `phases.<phase>`
  (`timed_out`, `budget_skipped`), in `events.jsonl` and in `report.md`.

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
  --backends cuda,triton,cute,tilelang,nvrtc
  --max-targets 4 --evaluations 12     targets and evaluation budget per target
  --parallel 2                         kernel agents at the same time
  --no-transforms                      kernels only
  --claude-model claude-opus-5-5 --effort high --budget 10 (USD per agent)
  --max-hours 3 --max-usd 40           budget for the whole run (see "Budgets")
  --agent-minutes 45                   time limit per agent session
  --eval-timeout 300                   seconds per evaluate_candidate
  --budget-reserve 0.15                share of --max-hours kept for integrate + report
  --harness my_harness.py              your own workload
  --until analyze|plan|capture|kernels|transforms|integrate

kernel-agent analyze <hf-url>          baseline + profile only (no Claude)
kernel-agent resume <run_dir> [--redo kernels]
kernel-agent eval capture.pt candidate.py [--profile] [--compile-baseline] [--timeout 300]
kernel-agent report <run_dir>          report.md + charts + dashboard.html
kernel-agent status <run_dir> [--watch 10]   per-target progress, e2e, cost, last evaluations
kernel-agent doctor [--smoke]
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
  run.json  toolchain.json  baseline.json  baseline_output.pt
  profile/summary.md          profile handed to the planner
  plan.json                   targets + transforms
  targets/<id>/capture.pt     module + real inputs/outputs
  targets/<id>/reference_source.py
  targets/<id>/candidates/    files the agent writes
  targets/<id>/history/       snapshot of every evaluated version
  targets/<id>/results.jsonl  every evaluation (full record)
  targets/<id>/progress.png   speedup per evaluation (see "Charts")
  transforms/                 model-level transforms + results.jsonl
  results.tsv                 experiment ledger: one row per evaluation
  events.jsonl                phase changes, agent start/stop, evaluations
  progress.png  amdahl.png  integration.png  dashboard.html
  integration.json  report.md  logs/
  costs.json                  per agent: $, turns, minutes, tools, session_id
  optimized/                  apply.py + manifest.json + kernels/
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
est_saved_ms  spread  eval_s  hypothesis
```

* `evaluate_candidate` requires a `hypothesis` (one sentence: what changed and
  why it should be faster) and accepts an optional `parent` snapshot.
  `evaluate_e2e` takes an optional `hypothesis`.
* `status` is `keep` when the result is correct and beats the best kept result
  by more than the noise: speedup > best × (1 + max(1 %, 2 × timing spread)).
  Kernels start from the reference module (1.0×), `e2e` rows from the baseline.
  A correct result that is not better is `discard`. Failures are `incorrect`,
  `build_error`, `runtime_error`, `crash` or `timeout`. The keep rule is the
  same one the budget advice uses.
* `backend` is read from the candidate's imports (`load_inline` → `cuda`,
  `cuda.core` → `nvrtc`, `cutlass` → `cute`, `tilelang`, `triton`; `torch` when
  there is no custom kernel).
* `results.jsonl` still has the full records (cases, errors) plus `exp`,
  `ledger_status` and `hypothesis`. For runs that predate the ledger, the
  rows are rebuilt from the `results.jsonl` files.

`kernel-agent status <run_dir> [--watch SECONDS]` prints the current phase, the
baseline, the projected and measured end-to-end latency, the total cost from
`costs.json`, a table per target (evaluations, keeps, failures, best speedup,
estimated ms saved, last hypothesis) and the last 10 ledger rows.

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

![integration waterfall](docs/images/example-integration.png)

The example images come from the synthetic run in `tests/synthetic_run.py`
(`python tests/synthetic_run.py /tmp/demo` writes one and draws its charts).

## Candidate contract

```python
def build(reference: torch.nn.Module) -> torch.nn.Module:
    """Return a drop-in replacement for `reference`: same forward signature,
    same outputs, same in-place side effects, sharing its weights. Return
    `reference` for instances the kernel does not support."""
```

## Harness contract (custom models)

`harness.py` defines `create(spec) -> Workload`. Implement `load`, `roots`,
`make_inputs`, `run` and `compare` (`kernel_agent/workloads/base.py`).

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
