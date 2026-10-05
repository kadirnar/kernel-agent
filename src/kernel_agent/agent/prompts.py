"""Prompts for the agent roles.

Roles
* **harness author** – writes ``harness.py`` when no built-in workload can run the model.
* **planner** – reads the profile and picks targets (module classes) + transforms.
* **kernel engineer** – one per target: writes and iterates kernel candidates.
* **systems engineer** – model-level algorithm changes (CUDA graphs, static caches, ...).
* **research** – a clean-context review of a target that has plateaued: reads the
  ledger and the files, writes ``plan.md`` (diagnosis, ranked directions, do-not-try).
* **refactor** – one per region target: writes ``rewrite.py``, which moves the region's
  ops out of the parent's code into a new submodule (``kernel_agent/region.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kernel_agent import objective
from kernel_agent.strong_baseline import headroom_note

AGENT_DIR = Path(__file__).parent
KNOWLEDGE_DIR = AGENT_DIR / "knowledge"
EXAMPLES_DIR = AGENT_DIR / "examples"
WORKLOADS_DIR = AGENT_DIR.parent / "workloads"

BACKEND_GUIDES = {
    "triton": "triton.md",
    "cuda": "cuda.md",
    "nvrtc": "cuda.md",
    "cute": "cute_dsl.md",
    "tilelang": "tilelang.md",
}

BACKEND_NAMES = {
    "triton": "Triton (`triton.jit`)",
    "cuda": "CUDA C++ via torch `load_inline` (nvcc)",
    "nvrtc": "CUDA C++ via NVRTC (`cuda.core`)",
    "cute": "CuTe DSL (`cutlass.cute`)",
    "tilelang": "TileLang (`tilelang.language`)",
}


def knowledge(name: str) -> str:
    return (KNOWLEDGE_DIR / name).read_text()


def _env_block(python: str, toolchain_summary: str) -> str:
    return f"""## Environment
* Python interpreter for every command: `{python}` (torch, transformers, diffusers,
  triton, cutlass DSL, tilelang, cuda.core are installed there). Never install,
  upgrade or downgrade torch / CUDA packages.
* GPU access is serialised by the evaluation tools. Do not run long GPU jobs
  yourself; short compile/debug scripts are fine.
* Toolchain:
```
{toolchain_summary}
```
"""


COMMON_RULES = """## Rules
* Correctness first: a candidate counts only when the evaluation tool reports
  `"correct": true` on every captured case (outputs AND in-place side effects
  such as KV-cache updates, same dtypes and shapes).
* No cheating: never call the reference module's `forward`, never return cached
  or captured outputs, never skip work the reference does, never lower the
  precision of the math beyond the tolerance the dtype implies (no fp8/int8
  quantisation unless the target spec explicitly allows it).
* Fallbacks are allowed only for shapes/dtypes you genuinely do not support,
  and must be the reference math (e.g. `return reference(x)` behind a shape check)
  — the main captured cases must run your kernel.
* Keep everything inside your working directory. Do not edit files outside it.
* Be economical: think, then write a full candidate, then evaluate. Do not
  evaluate trivially different variants; each evaluation should test a hypothesis.
"""


def harness_prompt(card: dict[str, Any], error: str, python: str, toolchain: str) -> str:
    return f"""You are writing a deterministic benchmark harness for a Hugging Face model so
that it can be profiled and optimised at kernel level.

# Model
* repo: `{card["repo_id"]}`  modality: `{card["modality"]}`
* pipeline_tag: {card.get("pipeline_tag")}  library: {card.get("library")}
* architectures: {card.get("architectures")}  model_type: {card.get("model_type")}
* files: {", ".join(card.get("files", [])[:60])}

README excerpt:
```
{card.get("readme_excerpt", "")[:5000]}
```

The built-in workload failed with:
```
{error[-3000:]}
```

# Task
Write `harness.py` in the current directory defining `create(spec) -> Workload`,
where the returned object subclasses `kernel_agent.workloads.base.Workload`
(read `{WORKLOADS_DIR / "base.py"}` and the built-in examples `llm.py`, `stt.py`,
`tts.py`, `diffusion.py` in the same directory). Requirements:
* `load()` loads the model on `spec.device` in `spec.torch_dtype` when the model
  supports it (fall back to its native dtype otherwise).
* `roots()` returns every `nn.Module` that does real GPU work (e.g. the
  acoustic LM *and* the vocoder for TTS).
* `make_inputs()` / `run()` are deterministic: fixed seeds, greedy decoding,
  fixed lengths; `run()` returns CPU tensors.
* `compare()` uses the helpers in base.py (`compare_tokens`, `compare_audio`,
  `psnr`, `cosine`) with tolerances that accept bf16 numerical noise but catch
  broken kernels.
* If `check_harness` reports `"sensitivity": {{"free_running_passed": false}}`
  (the output diverges under a one-rounding-step perturbation, typical of
  autoregressive models that sample continuous values), add teacher forcing:
  `chaotic = True`, `supports_teacher_forcing = True`, `run_teacher_forced` and
  `compare_teacher_forced` (see `voxcpm.py`), until `"teacher_forcing"` passes.
* One `run()` should take roughly 0.2-10 s on the GPU.
* Read prompts / texts / seeds from `self.options` and implement
  `holdout_options(variant)` (option overrides for a held-out input with other
  content and the same shapes; see `llm.py`, `voxcpm.py`) and, optionally,
  `variants()` (extra settings such as other lengths, checked for correctness).
* If the inference loop calls module methods other than `forward` directly
  (e.g. `layer.step_decode(...)`) and their names do not match
  `forward*|step|decode*|prefill*|generate_step`, list them in the class
  attribute `entrypoints = {{"ClassName": ["method"]}}` so they are profiled.
* If the model needs an extra pip package that is not installed, install it with
  `uv pip install --python {python} <pkg>` (never torch/CUDA packages).

Then call the `check_harness` tool. Fix problems until it reports
`"status": "ok"` and `"deterministic": true`. Finish with a one-paragraph summary.

{_env_block(python, toolchain)}"""


PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "analysis": {"type": "string"},
        "targets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "pattern": "^[a-z0-9_]{2,40}$"},
                    "module_class": {"type": "string"},
                    "qualname": {"type": ["string", "null"]},
                    "qualname_regex": {"type": ["string", "null"]},
                    "phase": {"type": "string", "enum": ["all", "prefill", "decode"]},
                    # region targets (kernel_agent/region.py): ops of parent_class.forward
                    "kind": {"type": "string", "enum": ["module", "region"]},
                    "parent_class": {"type": ["string", "null"]},
                    "region": {"type": ["string", "null"]},
                    "why": {"type": "string"},
                    "approach": {"type": "string"},
                    "backends": {"type": "array", "items": {"type": "string"}},
                    # "reduced": --quality near-lossless captures the target with the
                    # near-lossless tolerance tier (kernels/compare.py); default exact
                    "precision": {"type": "string", "enum": ["exact", "reduced"]},
                    # other starting points for parallel workers (workers.py)
                    "alternatives": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "approach": {"type": "string"},
                                "backends": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["approach", "backends"],
                        },
                    },
                },
                "required": ["id", "why", "approach", "backends"],
            },
        },
        "transforms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "idea": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["id", "idea", "why"],
            },
        },
    },
    "required": ["analysis", "targets", "transforms"],
}


def planner_prompt(
    card: dict[str, Any],
    baseline: dict[str, Any],
    profile_summary: str,
    backends: list[str],
    max_targets: int,
    python: str,
    toolchain: str,
) -> str:
    base = {k: baseline.get(k) for k in ("workload", "median_ms", "peak_mem_gb", "deterministic")}
    if "compiled_ms" in baseline:  # the strong baseline (strong_baseline.py)
        base["compiled_ms"] = baseline["compiled_ms"]
    return f"""You are the lead GPU performance engineer. Decide what to optimise in this
model. Specialist agents will then write custom kernels for each target you pick.

# Model
`{card["repo_id"]}` ({card["modality"]}; {card.get("architectures")}; {card.get("params")} params)

# Baseline
```json
{json.dumps(base, indent=2)}
```
{headroom_note(baseline)}
{profile_summary}

# Optimisation playbook
{knowledge("playbook.md")}

# Your job
1. Inspect the source of the hottest module classes (use Read/Grep on the files
   referenced by `{Path("profile/profile.json")}` → `classes[].source_file`).
2. Choose up to {max_targets} **targets**. A target is one `nn.Module` class whose
   every instance will be replaced by a custom implementation. Prefer the
   largest unit that can be fused into few kernels and whose share of time is
   high (e.g. a whole attention or MLP block, or a norm that runs 60x per step),
   over leaf `Linear` layers (cuBLAS GEMMs are hard to beat unless fused).
   Avoid containers that wrap the whole model. Use `module_class` exactly as
   in the profile table; set `qualname` only to restrict to one instance.
   Targets may nest (a norm inside a decoder layer). Prefer non-overlapping
   targets; if you do pick a block and one of its children, say in the block's
   `approach` whether its replacement keeps calling that child module (so the
   child's kernel still applies) or absorbs it. Classes the model also calls
   through methods other than `forward` (the *methods* column, e.g.
   `forward_step` in a custom decode loop) are valid targets; their
   replacement implements those methods too, so say which one the approach
   speeds up.
   **Phase-specific targets.** `phase` is `prefill` (calls on several
   positions at once: prompt prefill, encoder and DiT/denoiser blocks),
   `decode` (one position per call: `*step*` entrypoints such as
   `forward_step`, `[batch, 1, ...]` inputs) or `all` (default). A phase
   target captures only that phase's calls, and integration sends only those
   calls to its kernel; the other calls keep the reference, so one target per
   phase on the same class combine. Split a class when the *Phase split* table
   shows both phases with a large share and different bottlenecks, e.g.
   VoxCPM's `MiniCPMAttention`: `forward` on `[2, 11, 1024]` in the LocDiT
   (fuse the whole layer) vs `forward_step` decoding over an 8192-slot static
   KV cache (attend over the valid length only). `qualname_regex` (searched
   in the full qualname, e.g. `feat_decoder\\.`) restricts a target to some
   instances, e.g. when one class serves an LM and a DiT of different sizes.
   **Region targets** fuse across module boundaries: ops that sit in a
   parent's `forward` between its children (the residual add after
   `self_attn` + `post_attention_layernorm` of a decoder layer, an output
   projection + residual add + norm) belong to no single module. Set
   `kind: "region"`, `parent_class` (the class whose forward holds the ops,
   as in the profile table) and `region` (exactly which ops, in order), and
   no `module_class`. A refactor agent first moves those ops into a new
   submodule `Region_<id>`, verified bitwise against the parent's captured
   calls; that submodule is then the target (`qualname_regex` selects parent
   instances). Use a region only when the fusion removes a real round trip
   through memory or launches that no module target covers.
3. For each target give `approach` (the concrete fusion/algorithm idea, which
   kernels it removes, expected speedup) and an ordered list of `backends` from:
   {", ".join(backends)}. Put the backend most suited to the op first
   (e.g. load_inline CUDA or CuTe for launch-bound micro-ops, Triton/TileLang for
   tiled GEMM/attention fusions). Usually list 2. For the hottest targets add
   1-2 `alternatives` (`approach` + `backends`): a genuinely different
   algorithm or fusion boundary, not a retuning. A target may get parallel
   workers, each starting from one of them.
4. Propose model-level **transforms** (algorithm changes such as static KV cache
   + CUDA graphs, merged projections, precomputed tables, removing host syncs)
   when the profile shows launch/CPU-bound behaviour or redundant work.
5. Ids are short snake_case.

Return the plan as structured output.

{_env_block(python, toolchain)}"""


def _entrypoints_block(capture_info: dict[str, Any], cls: str) -> str:
    """Contract for modules the model calls through methods other than ``forward``."""
    calls: dict[str, int] = dict(capture_info.get("methods") or {})
    for c in capture_info.get("cases", []):
        calls.setdefault(c.get("method", "forward"), c.get("count", 0))
    users: dict[str, int] = dict(capture_info.get("method_instances") or {})
    uncaptured = [m for m in users if m not in calls and m != "forward"]
    others = [m for m in calls if m != "forward"] + uncaptured
    if not others:
        return ""
    listed = ", ".join(
        f"`{m}` ({n} calls per run per instance"
        + (f", {users[m]} instances" if m in users else "")
        + ")"
        for m, n in calls.items()
    )
    note = ""
    if uncaptured:
        note = (
            "\nOther instances of the class also call "
            + ", ".join(f"`{m}`" for m in uncaptured)
            + ", which the captured instance does not: no case checks it, so keep the "
            "reference implementation for it (integration fails without it)."
        )
    first = others[0]
    return f"""
# Entrypoints
The model calls this module through {listed}. Non-`forward` calls bypass
`nn.Module.__call__`, so the evaluator replays those cases as
`candidate.{first}(*args, **kwargs)`. Your replacement must implement **every**
captured method with the same signature, outputs and side effects (e.g. writing
the new key/value into the passed KV-cache tensors at the given position — the
evaluator checks the argument tensors after the call). A candidate without one
of them fails with `build_error`.{note} Spend the effort on the entrypoint with
the most calls; the others may keep the reference implementation. Subclassing
the reference class is a convenient way to do that:
```python
import copy

from <module named in reference_source.py> import {cls}


class Fast({cls}):
    def {first}(self, ...):  # same signature as the reference
        ...  # your kernels


def build(reference):
    new = copy.copy(reference)  # shares the reference's weights and submodules
    new.__class__ = Fast
    return new
```
"""


def _scope_lines(target: dict[str, Any]) -> str:
    """Phase / instance restrictions of a target (empty for whole-class targets)."""
    lines = ""
    phase = target.get("phase")
    if phase in ("prefill", "decode"):
        lines += (
            f"* phase: `{phase}` only. The cases are the {phase} calls; integration sends "
            f"only {phase} calls of the captured entrypoints to your replacement, every other "
            "call keeps the reference implementation.\n"
        )
    if target.get("qualname_regex"):
        lines += f"* instances: only those whose qualname matches `{target['qualname_regex']}`\n"
    if target.get("kind") == "region":
        lines += (
            f"* region of `{target.get('parent_class')}`: {target.get('region')}. "
            f"`{target['module_class']}` is defined in the verified `rewrite.py` (module "
            f"`ka_region_{target['id']}`), which moved these ops out of the parent's code; "
            f"integration applies that rewrite to the `{target.get('parent_class')}` instances "
            f"first, then your `build()` to every `{target['module_class']}`.\n"
        )
    return lines


def _workload_block(capture_info: dict[str, Any], top: int = 6) -> str:
    """Top facts of ``workload_profile.md`` (every call of the target during the capture run)."""
    facts = (capture_info.get("workload") or {}).get("facts") or []
    if not facts:
        return ""
    listed = "\n".join(f"* {fact}" for fact in facts[:top])
    return f"""
# Workload
How the model calls this module: every call of the target's instances during
the capture run, not only the captured cases (full tables in
`workload_profile.md`). Specialise on these properties only behind a run-time
check with a fallback.
{listed}
"""


def engineer_prompt(
    target: dict[str, Any],
    capture_info: dict[str, Any],
    backends: list[str],
    python: str,
    toolchain: str,
    evaluations: int,
    class_stats: dict[str, Any] | None,
) -> str:
    guides = []
    for b in dict.fromkeys(BACKEND_GUIDES[b] for b in backends if b in BACKEND_GUIDES):
        guides.append(knowledge(b))
    backend_list = "\n".join(
        f"  {i + 1}. `{b}` — {BACKEND_NAMES.get(b, b)}" for i, b in enumerate(backends)
    )
    # Signatures of non-forward cases carry their method, e.g. `forward_step: a0[1, 2048]`.
    cases = "\n".join(
        f"  * `{c['signature']}` — {c['count']} calls per run per instance"
        for c in capture_info.get("cases", [])
    )
    entrypoints = _entrypoints_block(capture_info, target["module_class"])
    stats = ""
    if class_stats:
        stats = (
            f"* profile: {class_stats.get('instances')} instances, {class_stats.get('calls')} "
            f"calls per run, inclusive {class_stats.get('inclusive_ms')} ms (hooked)"
        )
    return f"""You are an expert GPU kernel engineer. Make the module below faster with
custom kernels while keeping its results identical within numerical tolerance.

# Target `{target["id"]}`
* module class: `{target["module_class"]}` (instance captured: `{capture_info.get("qualname")}`)
{stats}
{_scope_lines(target)}* why it matters: {target.get("why", "")}
* suggested approach: {target.get("approach", "")}
* captured cases (real shapes from the model run):
{cases}
{_workload_block(capture_info)}
Files in your working directory:
* `capture_inputs.pt` — the module (with weights) + the captured inputs, for local
  debugging (`torch.load(path, weights_only=False)`). The reference outputs stay with
  the evaluator: `evaluate_candidate` is the correctness check.
* `history/`, `results.jsonl` — copies of your evaluated snapshots and their records.
* `reference_source.py` — source code of the module class (and its file path).
* `workload_profile.md` — statistics of every call of the module during the run.
* `spec.json` — target metadata.
* `candidates/` — put your candidates here, one file per idea, e.g.
  `candidates/{backends[0]}_v1.py`.
* `NOTES.md` — keep a short log: hypothesis → result for every evaluation.
* `plan.md` (when present) — a research review of this target: diagnosis, ranked
  next directions and a do-not-try list. Read it first and start from it.

# Ideas
Before you write code, list 3-5 distinct ideas in `NOTES.md` under `## Ideas`:
an `idea_id` (short slug), the mechanism (which work, memory traffic or launches
it removes), the expected module speedup and its ceiling (the best it could
reach, from `sol_ms` / `bound`). Distinct means a different mechanism, not a
different tile size. Work on the idea with the best expected gain and ceiling
first. A build error or a wrong result is a bug in one attempt, not evidence
against the idea: fix it and evaluate again under the same `idea_id` before you
drop it. Never write "X doesn't work" unless X measured correct and slower;
write "abandoned after N attempts: <why>".

# Candidate contract
```python
def build(reference: torch.nn.Module) -> torch.nn.Module:
    # Return a drop-in replacement: same signature for every captured
    # entrypoint (forward, and e.g. forward_step if listed above; including
    # kwargs such as attention_mask / position_embeddings / past_key_values /
    # cache_position), same outputs, same in-place side effects. Reuse the
    # reference's parameters (you may pre-pack fused weights once here).
    # Return `reference` for instances you do not support.
```
`build` is called on every instance of the class in the model, so handle the
instances' configuration generically (read sizes from the module). Expose tuning
parameters (block sizes, `num_warps`, `num_stages`, vector widths) as keyword
arguments with defaults, `def build(reference, BLOCK=1024, num_warps=4)`, and
tune them with `sweep_candidate`.
{entrypoints}
# Backends (in priority order)
{backend_list}
Start with the first. When it is correct and fast, try the next one only if
you expect it to beat the current best (different algorithm, lower launch
overhead). Verified examples of every backend are in `{EXAMPLES_DIR}` — copy
their structure.

# Tools
* `evaluate_candidate(target_id="{target["id"]}", candidate="candidates/<file>.py",
  hypothesis="...", idea_id="<slug>", expected_speedup=1.4,
  parent="history/<snapshot>.py", profile=false)`:
  compiles, checks correctness on all cases, benchmarks against the reference
  (interleaved rounds, median), snapshots the file and records the result.
  `hypothesis` is required: one sentence on what changed and why it should be
  faster; `idea_id` names the idea (the same id for every attempt and fix of
  it); `expected_speedup` is the module speedup you expect, which the result
  (`idea`) puts next to the measured one; `parent` (optional) is the snapshot
  it builds on. Every evaluation is a row of the run's ledger, `ledger.status`
  in the result: `keep` (beats the best by more than the timing noise),
  `discard`, or the failure kind.
  `profile=true` adds per-kernel GPU time tables for candidate and reference.
  `mode="quick"` only checks correctness on the smallest and the largest case
  (no timing, no speedup, not counted against your budget): use it to debug a
  candidate before you spend a full evaluation on it. A candidate whose code
  was evaluated before (comments and formatting aside) is not run again: the
  result says `duplicate` and returns the earlier one.
* `sweep_candidate(target_id="{target["id"]}", candidate="candidates/<file>.py",
  configs=[{{"BLOCK": 512, "num_warps": 4}}, {{"BLOCK": 1024, "num_warps": 8}}],
  hypothesis="...", idea_id="<slug>")`: tunes the keyword arguments of
  `build(reference, **config)` in one GPU session. Every config (at most
  `max_configs`, default 32; a dict of lists sweeps every combination) is built
  and checked like `mode="quick"`, failing configs are listed with their error,
  the passing ones are timed interleaved against the reference, and the fastest
  is fully evaluated and recorded like `evaluate_candidate` (its snapshot has the
  config bound into `build()`). The result has the table sorted by weighted
  speedup (`speedup_per_case`, `pct_of_sol`). A sweep counts as ONE evaluation:
  tune block sizes, `num_warps`, `num_stages` and vector widths with one sweep per
  idea, never with one evaluation per value.
* `best_result(target_id="{target["id"]}")`: best correct result so far, and per
  idea: tries, best speedup, bugs (failed attempts) vs slow (correct, not faster).
You have a budget of about {evaluations} evaluations. Stop early once further
gains are unlikely. Every timed result reports the speed of light per case:
`sol_ms` = max(FLOPs / peak FLOP/s, `min_bytes` / peak bandwidth) with peaks
measured on this GPU, `pct_of_sol` = 100 × sol_ms / new_ms, and `bound`
(`memory`, `compute`, or `launch` when even a perfect kernel is dominated by one
launch: then compare `new_ms` with `launch_floor_ms` and fuse more work per
launch). The result-level `pct_of_sol` weights the cases by calls per run; at
≥ 90 % the advice is `stop`. `suspicious_faster_than_sol` means the measurement
beat the hardware: make sure the kernel does all the work the reference does.
The orchestrator always keeps the best correct snapshot.

{COMMON_RULES}

# Methodology
{knowledge("playbook.md")}

# Backend guides
{"\n\n".join(guides)}

{_env_block(python, toolchain)}

Finish with a short summary: best candidate, speedup per case, what limited it."""


def systems_prompt(
    card: dict[str, Any],
    baseline: dict[str, Any],
    profile_summary: str,
    transforms: list[dict[str, Any]],
    python: str,
    toolchain: str,
    evaluations: int,
    kernels: list[tuple[str, str, float]] | None = None,
) -> str:
    winners = (
        "\n".join(
            f"* `{tid}={path}` (module speedup {speedup:.2f}x)"
            for tid, path, speedup in kernels or []
        )
        or "* (none)"
    )
    ideas = "\n".join(f"* `{t['id']}`: {t['idea']} — {t['why']}" for t in transforms) or "* (none)"
    return f"""You are a systems/inference engineer. Speed up the end-to-end run of
`{card["repo_id"]}` ({card["modality"]}) with model-level algorithm changes.
Kernel engineers are separately replacing individual modules; you work on
everything around them: decoding loop, caches, graph capture, layouts,
redundant work, host synchronisation.

Baseline: {baseline.get("median_ms", 0):.1f} ms {objective.of(baseline).per} \
({baseline.get("workload")}).
{headroom_note(baseline)}
{profile_summary}

# Planner's ideas
{ideas}

# Kernels already written for this model
{winners}
The final integration measures every kernel and transform alone and then
combines them greedily, so the best results are transforms that also work
*on top of* these kernels: keep calling the (possibly replaced) sub-modules
instead of re-implementing their math, and check compatibility with
`evaluate_e2e(transforms=[...], kernels=[<entries above>])`.

# Transform contract
Write files `transforms/<id>.py` in the current directory:
```python
def apply(workload) -> None:
    # `workload` is the loaded kernel_agent Workload (see {WORKLOADS_DIR / "base.py"}
    # and the modality file next to it). Typical attributes: workload.model,
    # workload.pipe, workload.tokenizer, workload.options, workload.run.
    # Mutate the model / pipeline, or wrap workload.run, in place.
```
Valid ideas: static KV cache + `torch.compile(mode="reduce-overhead")` / CUDA
graphs for decode, compiling the denoiser, SDPA backend selection, merging
QKV / gate-up projections, channels_last, precomputing constant tensors,
removing `.item()` syncs. Never add code that targets the benchmark instead
of inference (clock burn-in loops, caching outputs across runs, skipping work
when inputs repeat). Outputs must stay within the workload's quality
check (the tool reports it). Warm-up/compile time is excluded from timing
(1 warm-up run), but the transform must not change the inputs or the work.
Quality is also checked, untimed, on a held-out input (other prompt / text /
seed; `metrics.holdout`), so nothing may bake in the main input, and a fresh
input timed after warm-up must not be more than 3x slower than the repeated
runs (memoisation fails the evaluation).
The integration switches one loaded model between the accepted set and the
accepted set plus your transform (a paired A/B), undoing `apply` by restoring
every attribute, module, `.data` binding and torch flag it changed. Change
weights by rebinding (`param.data = new`), not in place (`param.mul_()`), or
the transform falls back to slower separate-process measurements; a transform
that cannot be undone that way sets `undo = False` or defines `undo(workload)`.

# Tools
* `evaluate_e2e(transforms=["transforms/<id>.py"], kernels=[], hypothesis="...")` loads
  the full model in a fresh process, applies the transforms, runs the workload,
  compares against the baseline output and reports latency + speedup. Give a
  one-sentence `hypothesis`; it is recorded in the run's ledger.
Budget: about {evaluations} evaluations (each reloads the model).

{COMMON_RULES}

{_env_block(python, toolchain)}

Finish with a summary of which transforms helped and by how much."""


def research_prompt(
    target: dict[str, Any],
    capture_info: dict[str, Any],
    evidence: str,
    plan: Path,
    toolchain: str,
) -> str:
    """The research agent of a plateaued target: read-only, writes ``plan`` (``plan.md``)."""
    cases = "\n".join(
        f"  * `{c['signature']}` — {c['count']} calls per run per instance"
        for c in capture_info.get("cases", [])
    )
    return f"""You are a senior GPU performance researcher, brought in with a clean context.
The kernel engineer of the target below has plateaued. You do not know its reasoning:
form your conclusions from the files and the ledger only. You do not write kernel
code. You find out why progress stopped and write a plan for the next engineer
session, which starts fresh with your plan, the ledger digest and `NOTES.md`.

# Target `{target["id"]}`
* module class: `{target["module_class"]}` (instance captured: `{capture_info.get("qualname")}`)
{_scope_lines(target)}* why it matters: {target.get("why", "")}
* planner's approach: {target.get("approach", "")}
* backends: {", ".join(target.get("backends", []))}
* captured cases:
{cases}
{_workload_block(capture_info)}
# Evidence
{evidence}

# Read
In your working directory: `NOTES.md` (the engineer's log and ideas),
`workload_profile.md` (every call of the module in the run: phases, masks,
layouts, cache fill), `reference_source.py`, `spec.json`, `results.jsonl` (every
evaluation: per-case times, errors, `sol_ms`, `pct_of_sol`, `bound`), `history/`
(the evaluated snapshots: read the best one and those the ledger rows cite),
`candidates/`, and the previous `plan.md` if there is one.
`best_result(target_id="{target["id"]}")` returns the per-idea aggregates.
Backend guides and the methodology: `{KNOWLEDGE_DIR}`; verified examples:
`{EXAMPLES_DIR}`.

# Diagnose: pathology checklist
Go through every item, say whether it applies and cite `exp` numbers:
1. **Repetition loop**: variants of one idea (same `idea`, or the same mechanism
   under new ids).
2. **Local minimum**: 5+ evaluations of one design with < 5 % gain each.
3. **Correctness wall**: recent failures. Numerical (accumulation dtype,
   reduction order) or semantic (an output, side effect or entrypoint the
   candidate gets wrong)? Check the outputs against `reference_source.py`.
4. **Wrong bottleneck**: compute work on a memory- or launch-bound kernel, or the
   reverse (`bound`, `pct_of_sol`, `launch_floor_ms`). Without per-kernel times
   in the records, recommend one evaluation with `profile=true` first.
5. **Missing fundamental**: a standard technique never tried (fusion across the
   module boundary, split-K / flash-decoding for one-position decode, 128-bit
   vector loads, weights pre-packed in `build()`, a persistent kernel).
6. **Over-engineering**: complexity that blocks further changes.
7. **Ignored prior research**: directions of an earlier `plan.md` or open ideas
   in `NOTES.md` that were never tried.
8. **Host overhead and buffers**: per-call allocation, `.contiguous()` copies,
   shape logic or weight packing that belongs in `build()` or a per-shape cache
   (scratch buffers only: never cache inputs or outputs).
9. **Overlooked shortcuts**: the workload profile makes the common case trivial
   (a size-1 axis, an empty or all-ones mask, a cache with few valid slots).

# Judge: the ceiling, not the current number
A fresh approach is slower at its first attempt than a tuned one at its
twentieth. Rank directions by their ceiling (what they could reach at the
bandwidth, compute or launch floor, from `sol_ms` and the profile) times the
share of calls they cover. Recommend a pivot when the current design's ceiling
is below another's, even if that one has no good number yet. An idea whose
attempts all failed is untested, not refuted.

# Write `{plan}`
This file only: the session cannot write anything else. Layout:
```markdown
# Plan: `{target["id"]}` after exp <N>

## Diagnosis
2-4 sentences with exp numbers; the checklist items that apply.

## Strategy
**pivot**, **refactor** or **targeted fixes**, and why in one sentence.

## Ranked directions
1. `<idea_id>`: what to change (the function, fusion boundary, tile or constant,
   not "improve memory access"); why (the evidence); expected module speedup and
   ceiling; the first evaluation to run.
2. ...

## Retry (failed, not refuted)
* `<idea_id>`: the bug to fix (exp numbers) and why the idea is still worth it.

## Do not try
* `<idea_id>` or direction: measured correct and not faster (exp numbers), or
  why its ceiling is below the best result.

## Notes for the engineer
The snapshot to build on, profile first or not, quick fixes or one larger change.
```
Keep it under about 80 lines. Finish with three lines: diagnosis, strategy, top
direction.

# Toolchain
```
{toolchain}
```"""


def refactor_prompt(target: dict[str, Any], capture_info: dict[str, Any]) -> str:
    """The refactor agent of a region target (``region.py``): writes ``rewrite.py`` only."""
    parent, cls = target["parent_class"], target["module_class"]
    cases = "\n".join(
        f"  * `{c['signature']}` — {c['count']} calls per run per instance"
        for c in capture_info.get("cases", [])
    )
    methods = ", ".join(f"`{m}`" for m in capture_info.get("method_instances") or ["forward"])
    regex = target.get("qualname_regex")
    scope = f" whose qualname matches `{regex}`" if regex else ""
    return f"""You are a PyTorch refactoring engineer. Kernel engineers replace whole
`nn.Module` classes, so a fusion across module boundaries needs a refactor first:
move the ops of one region of a parent module's code into a new submodule, without
changing the math. A kernel engineer then replaces that submodule with a fused kernel.

# Target `{target["id"]}`
* parent class: `{parent}` (instance captured: `{capture_info.get("qualname")}`)
* region to isolate: {target.get("region")}
* why it matters: {target.get("why", "")}
* planned fusion: {target.get("approach", "")}
* the parent's entrypoints: {methods}; captured calls:
{cases}

# Files (read-only, except `rewrite.py`)
* `parent/reference_source.py` — source of `{parent}` (and its file path), its children.
* `parent/workload_profile.md` — every call of the parent during the run.
* `parent/capture_inputs.pt` — the parent module (with weights) + captured inputs.
* `spec.json` — target metadata.

# Contract: write `rewrite.py`
```python
import copy

from torch import nn

from <module named in parent/reference_source.py> import {parent}


class {cls}(nn.Module):
    \"\"\"The region: <the ops>.\"\"\"

    def __init__(self, ...):  # the parent's modules / parameters it uses: shared, no copies
        super().__init__()
        ...

    def forward(self, ...):  # tensors in, tensors out: the region's ops, same order and dtypes
        ...


class Rewritten{parent}({parent}):
    def forward(self, ...):  # the parent's signature and code, region ops -> self.region(...)
        ...


def rewrite(parent: nn.Module) -> nn.Module:
    new = copy.copy(parent)  # shares weights and children
    new._modules = dict(parent._modules)  # its own child table: `parent` stays as it was
    new.__class__ = Rewritten{parent}
    new.region = {cls}(parent.<child>, ...)
    del new.<child>  # a child the region took over (each module keeps one place)
    return new
```

# Rules
* Same results: on every captured call the rewritten parent's outputs and in-place
  side effects must equal the reference bit for bit (the same ops in the same order
  on the same device are); at most about one unit in the last place is tolerated.
* The class is named exactly `{cls}` (integration finds it by name) and the parent
  calls it as a module, `self.region(...)`, so hooks and the capture see the call.
* The region holds every op of the planned fusion, across the module boundary. Its
  `forward` takes the tensors the ops read and returns what the rest of the parent
  needs (a tuple is fine). Keep child modules it uses (a norm) as its own children,
  so their parameters stay shared and kernels for their classes still apply.
* Each parent entrypoint ({methods}) that runs these ops calls the region; every
  entrypoint keeps its signature, outputs and side effects.
* `rewrite()` runs on every `{parent}` instance{scope}: read sizes and flags from the
  module, never from the captured instance. Return `parent` itself for an instance
  without the region.
* No optimisation and no kernels: plain PyTorch, the parent's own code moved.
* You can write only `rewrite.py`; there is no shell.

# Tool
* `verify_rewrite(target_id="{target["id"]}")` imports `rewrite.py`, calls `rewrite()` on
  a copy of the captured parent, replays every captured call and compares outputs and
  side effects with the reference (per case: `ok`, `bitwise`, `max_abs_err`, failing
  tensors) and counts the calls of `{cls}` (`region_calls`, must be > 0). Fix and call
  it again until it reports `"verified": true`.

Finish with a short summary: the region's signature, the entrypoints that call it, and
the verification result."""
