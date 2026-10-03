"""Prompts for the agent roles.

Roles
* **harness author** – writes ``harness.py`` when no built-in workload can run the model.
* **planner** – reads the profile and picks targets (module classes) + transforms.
* **kernel engineer** – one per target: writes and iterates kernel candidates.
* **systems engineer** – model-level algorithm changes (CUDA graphs, static caches, ...).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
* One `run()` should take roughly 0.2-10 s on the GPU.
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
                    "why": {"type": "string"},
                    "approach": {"type": "string"},
                    "backends": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "module_class", "why", "approach", "backends"],
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
    return f"""You are the lead GPU performance engineer. Decide what to optimise in this
model. Specialist agents will then write custom kernels for each target you pick.

# Model
`{card["repo_id"]}` ({card["modality"]}; {card.get("architectures")}; {card.get("params")} params)

# Baseline
```json
{json.dumps(base, indent=2)}
```

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
3. For each target give `approach` (the concrete fusion/algorithm idea, which
   kernels it removes, expected speedup) and an ordered list of `backends` from:
   {", ".join(backends)}. Put the backend most suited to the op first
   (e.g. load_inline CUDA or CuTe for launch-bound micro-ops, Triton/TileLang for
   tiled GEMM/attention fusions). Usually list 2.
4. Propose model-level **transforms** (algorithm changes such as static KV cache
   + CUDA graphs, merged projections, precomputed tables, removing host syncs)
   when the profile shows launch/CPU-bound behaviour or redundant work.
5. Ids are short snake_case.

Return the plan as structured output.

{_env_block(python, toolchain)}"""


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
    cases = "\n".join(
        f"  * `{c['signature']}` — {c['count']} calls per run per instance"
        for c in capture_info.get("cases", [])
    )
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
* why it matters: {target.get("why", "")}
* suggested approach: {target.get("approach", "")}
* captured cases (real shapes from the model run):
{cases}

Files in your working directory:
* `capture.pt` — the module (with weights) + inputs + reference outputs. Read-only.
* `reference_source.py` — source code of the module class (and its file path).
* `spec.json` — target metadata.
* `candidates/` — put your candidates here, one file per idea, e.g.
  `candidates/{backends[0]}_v1.py`.
* `NOTES.md` — keep a short log: hypothesis → result for every evaluation.

# Candidate contract
```python
def build(reference: torch.nn.Module) -> torch.nn.Module:
    # Return a drop-in replacement: same forward signature (including kwargs
    # such as attention_mask / position_embeddings / past_key_values /
    # cache_position), same outputs, same in-place side effects. Reuse the
    # reference's parameters (you may pre-pack fused weights once here).
    # Return `reference` for instances you do not support.
```
`build` is called on every instance of the class in the model, so handle the
instances' configuration generically (read sizes from the module).

# Backends (in priority order)
{backend_list}
Start with the first. When it is correct and fast, try the next one only if
you expect it to beat the current best (different algorithm, lower launch
overhead). Verified examples of every backend are in `{EXAMPLES_DIR}` — copy
their structure.

# Tools
* `evaluate_candidate(target_id="{target["id"]}", candidate="candidates/<file>.py", profile=false)`:
  compiles, checks correctness on all cases, benchmarks against the reference
  (interleaved rounds, median), snapshots the file and records the result.
  `profile=true` adds per-kernel GPU time tables for candidate and reference.
* `best_result(target_id="{target["id"]}")`: best correct result so far.
You have a budget of about {evaluations} evaluations. Stop early once further
gains are unlikely (e.g. you are within ~10 % of the memory-bandwidth or launch
overhead floor). The orchestrator always keeps the best correct snapshot.

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
) -> str:
    ideas = "\n".join(f"* `{t['id']}`: {t['idea']} — {t['why']}" for t in transforms) or "* (none)"
    return f"""You are a systems/inference engineer. Speed up the end-to-end run of
`{card["repo_id"]}` ({card["modality"]}) with model-level algorithm changes.
Kernel engineers are separately replacing individual modules; you work on
everything around them: decoding loop, caches, graph capture, layouts,
redundant work, host synchronisation.

Baseline: {baseline.get("median_ms", 0):.1f} ms per run ({baseline.get("workload")}).

{profile_summary}

# Planner's ideas
{ideas}

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
removing `.item()` syncs. Outputs must stay within the workload's quality
check (the tool reports it). Warm-up/compile time is excluded from timing
(1 warm-up run), but the transform must not change the inputs or the work.

# Tools
* `evaluate_e2e(transforms=["transforms/<id>.py"], kernels=[])` loads the full
  model in a fresh process, applies the transforms, runs the workload, compares
  against the baseline output and reports latency + speedup.
Budget: about {evaluations} evaluations (each reloads the model).

{COMMON_RULES}

{_env_block(python, toolchain)}

Finish with a summary of which transforms helped and by how much."""
