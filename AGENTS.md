# Developing kernel-agent with coding agents

This file is for agents (and people) who **change kernel-agent's code**. kernel-agent's own
agent sessions never read it: they run with `setting_sources=[]` and load only the package's
skills (#126, #176).

## Principles

* **Model-agnostic.** Library code and the agents' prompts, skills and definitions name no
  model; model-specific facts are labelled evidence ("measured on VoxCPM2"). A workload
  file (`src/kernel_agent/workloads/`) is the only place for one model's code.
* **Any NVIDIA GPU.** Decisions come from the measured facts of the GPU at hand
  (`gpu_arch.py`, the toolchain block's peaks and instruction rates, `kernel-agent doctor`
  probes); numbers measured on one GPU are labelled with it (most were measured on an
  RTX 5070 Ti, sm_120).
* **Never self-limit.** Floors, ceilings and `pct_of_sol` are bounds of today's recipe, not
  of the model. No prompt, skill or agent may end a session claiming nothing more is
  possible: it ends with the next ideas worth testing.
* **Multi-agent by design.** Roles are agent definitions (`src/kernel_agent/agent/agents/`)
  a session can delegate to and a coordinator can start uniformly; know-how is in skills.
* **Evidence over claims.** A result counts only when the evaluator reports it correct and
  faster; every PR states measured numbers or the tests that show the behaviour.

## Where things are

| path | what |
|---|---|
| `src/kernel_agent/agent/plugin/skills/<name>/` | the agents' know-how as Agent Skills (`SKILL.md` + linked files); the plugin every session loads |
| `src/kernel_agent/agent/agents/<name>.md` | agent definitions of the roles and helpers (frontmatter + prompt) |
| `src/kernel_agent/roles.py` | the role registry: model, effort, turns, tools, GPU need, helpers per role; token and cache accounting |
| `src/kernel_agent/agent/prompts.py` | each role's prompt: a stable prefix (the system prompt every session of the role shares) and its target block (the first message) |
| `src/kernel_agent/agent/runner.py` | one SDK session: tools, hooks, skills, subagents, isolation |
| `src/kernel_agent/agent/tools.py` | the MCP tools (`evaluate_candidate`, `sweep_candidate`, `evaluate_e2e`, ...) |
| `src/kernel_agent/agent/examples/` | verified example candidates of every backend (selftests run them) |
| `src/kernel_agent/orchestrator.py`, `improve.py`, `coordinator.py` | the pipeline, the continuous loop and its concurrent sessions (`--agents N`) |
| `src/kernel_agent/kernels/` | the evaluator: correctness, timing, roofline, integrity, memcheck |
| `docs/` | design and research documents with their measurements |

Knowledge changes go into a skill: keep `SKILL.md` short (what the role needs first), put
long material in files it links, keep the frontmatter `name` equal to the directory and the
`description` a plain one-line YAML scalar (no `": "`). `tests/test_skills.py` validates
the tree, the definitions and every link.

## Environment

* Python 3.12 with `uv`: `uv sync --extra all --group dev`. Never install, upgrade or
  downgrade torch or CUDA packages in an existing environment.
* One GPU job at a time: kernel-agent's processes share `~/.cache/kernel-agent/gpu.lock`.
  Wrap every GPU-heavy command of your own (model loads, `pytest -m gpu`, evaluations):
  `flock ~/.cache/kernel-agent/gpu.lock env KERNEL_AGENT_LOCK_HELD=1 <command>`; keep such
  jobs short and never leave GPU processes running.
* Claude-powered phases (`optimize`, `improve`, `plan`, the kernel / systems / native
  sessions, the harness agent) use the user's Claude account: do not start them unless
  your task says so. `analyze --no-harness-agent`, `eval`, `memcheck`, `recheck`,
  `python -m kernel_agent.worker ...` and `improve --dry-run` (simulated agents and GPU)
  are fine.
* Run directories (`runs/`) are evidence: read them, never write into them, never signal
  processes you did not start.

## Tests and checks

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run pytest -q -m "not gpu"     # CPU tests; GPU tests are marked `gpu`
```

All four must pass before a PR. Add CPU tests for new behaviour (a fake SDK stream, a fake
Claude Code CLI or a local fake Messages API where a session is involved; never a real
model call), GPU tests marked `@pytest.mark.gpu`.

### Compile matrix (`ARCHS`)

`tests/test_arch_matrix.py` compiles every bundled example and kernel-agent's own device
code for sm_75 / sm_80 / sm_86 / sm_89 / sm_120 on the CPU and checks it against the
example's `ARCHS` (the GPUs it runs on) or `ARCHS_COMPILES` (where it compiles, when a limit
is runtime-only). A changed example recompiles only its own units (verdicts are cached by
content); the first run on a machine takes minutes. When it fails, change the declaration
or the code as its message says; a new Triton kernel, CuTe DSL or TileLang example needs
its entry in `tests/arch_matrix.py` (`TRITON` with the launch's specialisation, `CUTE`,
`TILELANG`). `uv run --no-sync python tests/arch_matrix.py` prints the matrix.

### Flaky tests

Several agents often run the suite at once (load average 20+): a CPU test whose verdict
depends on wall-clock time or thread scheduling then fails now and then, passes on a rerun
and hides real regressions (#211). Keep CPU tests deterministic:

* Times a verdict compares (A/B rounds, the memoisation probe, debounce intervals, queue
  aging) come from a simulated clock, `tests/fake_clock.py`'s `Clock` in place of the
  measuring module's clock (`workloads.base.time`, `gpuqueue.clock`, `dashboard.time`),
  not from real runs and sleeps.
* Threads and processes meet on a `threading.Barrier`, `Event` or `Semaphore` (or a file
  a child writes), never on a `sleep` after which they "are there by now". A timeout on
  such a wait (60 s) is reached only when the code under test is broken.
* Random draws are seeded and exclude what the check compares against (a fresh input
  that happens to be the held-out one).
* A real-time bound that remains (an agent timeout, a grace period) leaves a wide margin
  between what passes and what the bug would take.

Check a new or changed test that uses threads, processes or time 50 times on a loaded
machine (`--load 8`: 8 busy processes for the session; `--repeat 50`: each test 50 times;
both in `tests/conftest.py`); it must not fail once:

```bash
uv run pytest -q --repeat 50 --load 8 tests/test_x.py::test_y
```

## Pull requests

* One issue per branch, `issue-<N>-<slug>`, rebased on `origin/main`.
* Match the surrounding code: `from __future__ import annotations`, type hints, ruff line
  length 100, comments that say why. Touch shared files (`prompts.py`, `config.py`,
  `tools.py`, `worker.py`, `orchestrator.py`, `improve.py`, `cli.py`, `README.md`) in small,
  self-contained hunks: several agents work in parallel.
* Update the README sections that describe behaviour you changed.
* The PR body says `Closes #<N>`, what changed, the measured evidence and the test output.
