# Native engines: rewriting a model's inference path natively (#134)

kernel-agent optimises a model module by module: each target is one `nn.Module` class whose
instances a kernel replaces, verified on calls captured from the real run. That approach
plateaus where the remaining time is not *inside* one module but *between* modules: an
iterative solver re-streams its network's weights at every step, a decode step is dozens of
short launches and grid syncs, glue kernels sit between stages, the host round-trips inside
the loop. A **native engine** hands such a part of the inference path to a systems-native
agent, which rewrites it as native CUDA C++ / CuTe (a multi-file project) that keeps the
weights streaming, fuses across module boundaries and runs the part in one persistent kernel
or a few launches.

This document is the design; `src/kernel_agent/native/` implements it,
`src/kernel_agent/agent/knowledge/native.md` is the agents' contract and
`src/kernel_agent/agent/examples/native_project/` the project template. Everything here is
model-agnostic: targets come from the profile's structure, never from model names. VoxCPM2
appears only as the worked example in §10.

## 0. Summary

* **Targets** (§2): a `stage` (one module group of the profile), a `group` (the stages of one
  iteration of the generation loop) or the whole `loop`. Each stage has a *pattern* read from
  the profile: `solver` (a module calling an inner network k times per call: diffusion and
  flow samplers), `stack` (repeated layers: LM bodies and decode steps, encoders, vision
  backbones) or `other` (vocoders, VAEs, convolution stacks).
* **Where they come from** (§3): the stage graph of the newest ceilings table (topmost
  non-leaf rows with ≥ 5 % of the run, their floors at the precisions the run allows, calls
  per run), ordered by time above the floor; or the planner's `native` entry.
* **Interface** (§4): weights shared with the PyTorch model, outputs handed to its unchanged
  modules, the workload's quality seams kept.
* **Correctness** (§5): a stage is captured as a kernel target `native_<id>` and checked
  teacher forced on its recorded inputs by the unchanged evaluator; groups and the loop end
  to end with the workload's quality checks (the perceptual gate in near-lossless runs);
  memcheck and the integration's re-check as for any kernel.
* **Integration** (§6): a stage target is a kernel replacing the stage's module; any scope
  can be a transform. Ownership (#112) makes the module kernels inside a stage its
  alternatives.
* **Library support** (§7): multi-file project candidates (headers, several `.cu` files, a
  manifest, a build script) with content-digest compile caching; a project is evaluated,
  snapshotted, deduplicated, swept, memchecked, integrated and exported through its
  *bundle*, one `.py` file whose sha256 covers every file.
* **Agent and scheduler** (§8, §9): a `native` session with a longer budget, this run's and
  the library's best kernels as building blocks; the improve loop opens the native arm only
  with `--native on` or when the plan asks for it (`--native plan`, the default), and only
  once every module arm has stopped or plateaued. Each stage must beat the best
  module-level result end to end before the next one starts.

## 1. When module kernels plateau

The improve loop already knows when a module arm stops paying (scheduler stop rules):
`patience` evaluations without a new best, the speed-of-light stop, the time cap, the
speedup goal; a plateaued arm gets a research session first. The native arm uses exactly
that signal: it **waits while any kernel arm is live and has not plateaued**
(`scheduler._native_gate`). This is model-independent: it depends only on the ledger.

Why the plateau is often not the end: a module kernel is bounded by its module's floor *at
its call boundaries*. What a native engine can still remove, visible in any profile:

| signal (profile, ceilings) | what it means | native remedy |
|---|---|---|
| a stage far above its floor while the module kernels inside it are near theirs | the time is in boundaries: launches, syncs, glue, re-streamed weights | fuse the stage; stream weights across phases and layers |
| an inner row called k ≥ 2 times per call of its parent (`solver`) | each step reads the same weights again | fuse the steps; keep weights on chip or stream them once per step at the floor |
| many short kernels per call (kernel launches per call in the evaluator's results, idle gaps between launches) | launch / grid-sync bound | persistent kernels, PDL between launches |
| several stages with equal calls per run | one loop iteration split into stages that each drain the GPU | a `group`: one or two persistent kernels per iteration |

## 2. What a native engine target is

| scope | replaces | checked by | integrated as |
|---|---|---|---|
| `stage` | every instance of one module group (a ceilings row: class at an instance group) | its captured kernel target `native_<id>`, then end to end | kernel (or transform) |
| `group` | the stages called once per loop iteration, fused across their boundaries | end to end | transform |
| `loop` | the whole generation loop | end to end | transform |

Patterns, per model family:

| family | typical stage | pattern | typical native engine |
|---|---|---|---|
| LLM decode | the model body called once per token; the decode loop | `stack`, many calls per run; `loop` | persistent decode step (all layers, weights streamed by producer warps, split-K across the grid at M = 1–16); the loop with on-device sampling |
| diffusion / flow matching | the sampler / solver module | `solver` | all steps fused; guidance branches as one batched pass; the update math in the network's epilogue |
| encoders (text, audio, vision) | the encoder stack | `stack`, one call | one persistent kernel per few layers, activations on chip |
| vocoders, VAE decoders | the decoder | `other`, one call | fused convolution chains in a layout of their own |
| vision backbones | the backbone | `stack` | as encoders, with fused patch embedding / attention |
| autoregressive generators of several networks (TTS, speech, multimodal) | the per-iteration stages | several stages with equal calls → `group` | one or two persistent kernels per iteration, split at the quality seam (§4) |

## 3. Deriving targets from the profile: the stage graph

`native.engine.stage_graph(table)` reads the newest ceilings table (`profile/ceilings.json`,
or a round's re-profile): one row per module class at one instance group and phase, with
`now_ms`, `share`, `calls` per run, `instances` and the floor per precision.

1. **Stages**: the rows with at least `MIN_SHARE` (5 %) of the run that contain other rows
   (non-leaf), keeping the topmost: a row inside a kept row is part of it. At most six.
2. **Pattern** (`engine.pattern`): `solver` when a non-leaf row inside it, not a layer group,
   holds ≥ 50 % of its time and runs k ≥ 2 times per call; `stack` when a layer group
   (`….*`, folded indices) holds ≥ 50 %; else `other`.
3. **Floor**: the lowest floor among the precisions the run allows (`--precisions`);
   `headroom_ms = now − floor`.
4. **Loop body**: the largest set of stages with equal calls per run (≥ 2) is the `group`
   `loop_body` (two or more stages); when it covers ≥ 60 % of the run the whole `loop` is a
   target too.
5. **Order** (the staged plan): stages by `headroom_ms`, then the loop body, then the loop.

The planner may replace the derived plan with a `native` entry in `plan.json` (optional in
the plan schema):

```json
"native": {
  "why": "the solver re-streams its 200 MB network 9 times per call",
  "stages": [
    {"id": "solver", "scope": "stage", "group": "model.solver", "pattern": "solver",
     "idea": "warp-specialised weight streaming across the steps", "expected_speedup": 1.3},
    {"id": "loop", "scope": "loop", "members": ["solver", "step"], "idea": "..."}
  ]
}
```

Stages of the entry take the derived row's numbers when their `group` matches; a `stage`
without a module group is dropped (plan it as a `group`). A round's re-plan can add or
change the entry (the newest one wins).

## 4. Interface to the PyTorch model

* **Weights are shared**: an engine reads the reference modules' parameters and buffers; a
  format change (packing, FP8 with scales, a tiled layout) happens once in `build()` /
  `apply()` by rebinding (`param.data = packed`), which the integration's undo restores.
* **Outputs go to the model's unchanged modules** (decoder, vocoder, the solver's caller) in
  the dtype, shape, layout and device they expect, with the same in-place side effects (KV
  caches, state) as the code replaced.
* **Quality seams**: a workload may check quality teacher forced by wrapping one module call
  per iteration (a chaotic sampler whose free-running output cannot be compared). A `loop`
  engine keeps calling that module from Python once per iteration with the noise drawn as
  before, i.e. it splits its persistent kernels at the seam (one or two per iteration).
  Workloads say which module is the seam (`workloads/*.py`), and the e2e check rejects an
  engine that bypasses it with a clear reason.
* **Precision**: what the run allows. A stage target is captured at the reduced precision
  the run's module targets already use (`engine.module_precision`), so its tolerance tier
  matches the kernels it competes with.

## 5. Correctness

1. **Teacher forced per stage.** Before the native session works on a stage with a module
   group, the improve loop captures it as a kernel target (`native_<id>`, spec
   `native: true`, `qualname_regex` from its instance group) from the real run: every
   distinct call with its inputs, outputs and side effects. `evaluate_candidate` then runs
   the unchanged evaluator: captured cases, aliasing, redrawn inputs at fresh addresses,
   timed-output checks, fallback and hidden-work guards, the parent-side output comparison.
   That is teacher forcing at stage granularity: the stage always sees the reference's
   inputs.
2. **End to end.** `evaluate_e2e` runs the full workload with the engine (transform) and the
   accepted kernels: output comparison, held-out input, natural-length run, and in a
   near-lossless run the perceptual gate (#71).
3. **Memory safety.** The integration runs every kernel under compute-sanitizer memcheck on
   the captured cases and odd-size variants (#115); a project loads through the same path.

## 6. Integration

A stage target's best kernel enters the integration like any kernel; a transform project
like any transform (its snapshot is the bundle, §7). Each item is measured alone and added
by paired A/B. **Ownership (#112)**: the patcher records what an item replaces; a native
stage owns its modules, so the module kernels inside it overlap it and the integration tries
the stage *in their place* (a replace step) instead of stacking them. A transform that only
wraps the stage (a CUDA graph around it) does not overlap and still composes.

The export copies the bundle; importing it needs `kernel_agent` (the build cache and the
toolchain discovery), as compiled kernels already need the toolchain.

## 7. Library support: project candidates

```
<stage id>/
  kernel_project.toml     [project] name, entry, kind; [build] backend, sources, flags
  candidate.py            build(reference) (kind = "kernel") or apply(workload) ("transform")
  include/*.cuh           shared device code
  csrc/*.cu  csrc/binding.cpp  build.sh
```

**Manifest** (`native/project.py`, validated strictly: unknown keys are errors):

| key | meaning |
|---|---|
| `project.name` | `[a-z][a-z0-9_]{1,47}`; names the bundle and the snapshot |
| `project.entry` | the entry module (default `candidate.py`) |
| `project.kind` | `kernel` (`build`) or `transform` (`apply`) |
| `build.backend` | `torch_extension` (default; `torch.utils.cpp_extension.load`) or `command` |
| `build.sources` | globs of `.cu` / `.cpp` / `.c` files (each must match) |
| `build.include_dirs`, `cflags`, `cuda_cflags`, `ldflags` | compiler inputs (the project root is always an include dir) |
| `build.command`, `outputs`, `load` | a `command` build (`{src}`, `{build}`, `{python}` substituted; `KA_SRC_DIR`, `KA_BUILD_DIR`, `KA_PYTHON`, `KA_TORCH_CMAKE_PREFIX` set), its shared libraries, how they load (`torch_ops`, `python`, `none`) |
| `build.timeout_s` | build time limit (≤ 3600 s) |

**Files and digest.** Text files only (sources, headers, scripts), at most 200 files / 4 MB;
`build/`, hidden and `__pycache__` directories and build products are not part of a project;
symlinks are refused. The digest is the sha256 of the sorted (path, size, bytes).

**Bundles.** `pack()` writes one Python file embedding every file (one string literal per
line, deterministic) and the digest. Importing it checks the digest (a hand-edited bundle is
refused), writes the files to the cache and re-exports the entry module. Because a bundle is
a normal candidate file, the existing machinery works unchanged: snapshots and their sha256
(`history/NNN_<name>_<sha1>.py`, the sha256 covers the whole project), the duplicate check,
`sweep_candidate`'s config binding, memcheck, re-checks, the integration, the export and the
kernel library. The evaluator, memcheck and the patcher also accept a project directory.

**Build cache** (`~/.cache/kernel-agent/native`, `$KERNEL_AGENT_NATIVE_CACHE`):
`<digest>/src/` holds the files (rewritten from the bundle on every load; stray files and
`__pycache__` removed, so a stale `.pyc` cannot stand in for the entry) and
`<digest>/build-<key>/` one build per toolchain (key: digest + torch, CUDA, arch list, nvcc
flags, Python, C++ ABI) with a `stamp.json` of every output's sha256; a damaged build is
rebuilt. `project.load(__file__)` in the entry compiles once per key and process.

**Integrity.** The snapshot's sha256 covers every project file; the bundle checks its own
digest before any file is written; the evaluator's thread check covers the project's
directory in the cache. The build cache has the trust level of torch's extension cache
(within reach of an agent's Bash tool): the stamp catches accidents; a deliberate
replacement of a compiled library is the same exposure `load_inline` has today
(`KERNEL_AGENT_NATIVE_REBUILD=1` rebuilds from the snapshot).

**Prebuild.** `evaluate_candidate`, `sweep_candidate` and `evaluate_e2e` compile a project in
a subprocess that sees no GPU (arch from `TORCH_CUDA_ARCH_LIST`) before the evaluation, so a
compiler error is reported without taking the GPU lock and the evaluation finds the build in
the cache. `python -m kernel_agent.native.project check|pack|build` does the same by hand.

## 8. The systems-native agent

A `native` session (role `native` in `program.md`):

* **Budget**: `--native-minutes` per session (default 3 × `--agent-minutes`), 3 × the turns,
  `--native-evaluations` per slice (default 6); `native_patience` 8 runs without a new best
  and `native_hours` 6 h of slices before the arm stops.
* **Context**: the contract (`knowledge/native.md`), the template project, the workload
  files, the digest (the bar, the staged plan with each stage's state and teacher-forced
  target, the module arms and why they stopped, the last native evaluations, its
  `NOTES.md`), and **building blocks**: this run's best kernel per target and the kernel
  library's entries for this GPU architecture (earlier runs' winners), fastest first.
* **Tools**: `evaluate_candidate` / `sweep_candidate` / `best_result` on the stage targets,
  `evaluate_e2e` and `run_info`; relative paths resolve in its working directory
  (`transforms/native/`) first.
* **Web**: 6 lookups (PTX / CuTe instructions, reference engines of the model family).

## 9. Scheduler and planner wiring

* `--native off|plan|on` (default `plan`): the arm exists only when the plan has a `native`
  entry, or always with `on`, or never with `off`.
* The arm (`scheduler.NATIVE`) waits while a kernel arm is live and has not plateaued; then
  it competes like any arm. `ref_ms` is the run at the bar (the best module-level result end
  to end: integrations, the systems agent's runs), `best` its fastest native run over the
  bar, `estimate` `Policy.native_estimate` (1.3), and its runs are recorded with the ledger
  backend `native` (no longer the systems agent's).
* Before a native slice the loop captures the current stage's target once.
* The arm stops when every stage of the staged plan has a native run that beat the bar, by
  its patience or time cap, or when the budget ends.

## 10. Worked example: VoxCPM2 on one RTX 5070 Ti

Numbers from [PARALLEL.md](PARALLEL.md) §3.1 / §4.6 / §8 and [FP8.md](FP8.md) §8 (batch 1,
FP8 weights, near-lossless, 522 ms accepted; 60 patches per run). Per steady patch the GPU
runs the LocDiT CFM solve (9 estimator calls, CFG as one batch-2 call), the LocEnc, the base
LM step (28 layers), the residual LM step (8 layers); the AudioVAE runs once.

| stage (from the profile) | pattern | calls / run | ms / run now | FP8 DRAM floor | of floor |
|---|---|---|---|---|---|
| LocDiT CFM solver (`feat_decoder`) | solver (estimator × 9 per call) | 60 | 320.5 (fused layer kernel `layer_coop<3>`: 303.5 = 58 % of the run) | ~135 | **44 %** |
| base LM decode step | stack, 28 layers | 60 | 119.1 | ~95 | 80 % |
| residual LM decode step | stack, 8 layers | 60 | 34.8 | ~27 | 79 % |
| LocEnc | stack, 12 layers | 60 | 23.2 | ~15 | 64 % |
| AudioVAE decode | other | 1 | 15.1 | | off the critical path |

The stages called 60 times form the loop body (≥ 60 % of the run: the loop is a target too).
Ordered by headroom, the staged plan is:

1. **Warp-specialised LocDiT layer / CFM solver.** Producer warps (or bulk / TMA copies)
   stream the next phase's and the next layer's weights into a shared-memory ring across the
   grid syncs while consumer warps compute; PDL between the layer launches; the 9 steps'
   Euler update and CFG combine folded into the layer kernels (245 glue kernels per patch
   today). At the LM kernel's 80 % of floor a layer takes ~26 µs instead of 46.8 µs:
   **−130 ms, 522 → ~390 ms (1.34×)**. Stage target: the `UnifiedCFM` module, teacher forced
   on its recorded calls.
2. **LM decode steps** (both LMs, FP8): one persistent kernel per step with weight streaming
   and PDL instead of per-layer cooperative launches (572 inter-launch gaps of ~2.3 µs per
   run); bounded by the 80 % → ~90 % of floor: ~−15 ms.
3. **LocEnc**: 64 % → 80 % of floor: ~−5 ms.
4. **One persistent kernel per patch** (the loop body), split at the `feat_decoder` seam
   the VoxCPM workload's teacher forcing wraps: [LocEnc + both LM steps + projections] and
   [the CFM solver] as two persistent kernels per patch; removes the remaining boundaries
   (~0.9 µs each, 474 per patch).
5. **Optional AudioVAE** (off the critical path at batch 1: ≤ 15 ms, and only with
   streaming).

Each stage must beat the best module-level result end to end (522 ms) before the next
starts; the systems-native agent gets the run's FP8 decoder-layer kernels (`dit_layer_fp8`,
`lm_step_fp8`) and the library's entries as building blocks. At batch 16 the LocDiT is
GEMM-bound (M = 352) and the native gain is small (FP8.md §8: MXFP8 and simpler glue, −55
ms per batched run).

## 11. What this PR verifies and what it does not

Implemented and covered by CPU tests: manifest parsing and validation, file collection and
limits, digests, build keys, bundles (round trip, determinism, tamper refusal), cache
materialisation, `command` builds with the stamp cache and rebuilds, the prebuild
subprocess, the evaluator, sweep binding, patcher and tools accepting projects (a project
candidate passes the CPU evaluation; a thread it leaves running is an integrity violation),
the stage graph for diffusion, LLM and multi-stage loops, the planner entry, the gate, the
native arm's scoring and stop rules, and a dry-run improve loop that reaches the native arm
after the module arms.

Not run here (no GPU in this PR): the template project's `torch_extension` build with nvcc
and its GPU evaluation (`pytest -m gpu tests/test_native_project.py`,
`kernel-agent doctor --smoke`), memcheck of a project, and any native engine for a real
model. The §10 gains are measured headroom, not results.

## 12. Follow-ups

* A verified warp-specialised streaming example (shared-memory ring, mbarriers, PDL) in the
  examples, with a selftest, for stage 1 of §10 (#136 PR 5, #133 for TMA on sm_120).
* Timeline-based signals (#136 §7.3) in the stage graph: idle gaps and kernel boundaries per
  stage, so the planner sees launch-bound stages directly.
* Per-stage teacher forcing for `group` scopes (a capture of several modules' calls per
  iteration), so a fused group is checked before its end-to-end run.
