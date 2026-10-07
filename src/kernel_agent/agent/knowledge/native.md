# Native engines: contract for the systems-native agent

Module kernels stop paying where the remaining time sits *between* modules: an iterative
solver that re-streams its network's weights at every step, a layer stack or decode step
made of many short launches and grid syncs, glue kernels between stages, host round trips
inside a loop. A native engine rewrites such a part of the inference path as native CUDA
C++ / CuTe code, keeps the weights streaming, fuses across module boundaries and runs the
part in one persistent kernel or a few launches.

## Scopes (what you may replace)

| scope | what | checked by | integrated as |
|---|---|---|---|
| `stage` | one module group of the profile (a row of the ceilings table): an iterative solver module, a layer stack, a decode step, a one-shot decoder | its kernel target `native_<id>` (captured from the real run: teacher forced on the stage's recorded inputs), then end to end | a kernel replacing every instance of the stage's module (or a transform) |
| `group` | the stages that run once per iteration of the generation loop, fused across their boundaries | end to end | a transform |
| `loop` | the whole generation loop | end to end | a transform |

Patterns of a stage (from the profile's structure, any model family):

* **solver**: a module that calls an inner network k times per call (diffusion and
  flow-matching samplers, iterative refinement): fuse the steps, keep the network's weights
  streaming across them (each step reads the same weights: on-chip reuse or one pass per
  step at the DRAM floor), fold guidance branches into one batched pass and the update
  math into the network's last kernel.
* **stack**: repeated layers (LM bodies and decode steps, encoders, vision backbones,
  decoder stacks): one persistent kernel per call or per few layers; producer warps (or
  bulk / TMA copies) stream the next phase's and the next layer's weights into a
  shared-memory ring while consumer warps compute (warp specialisation); programmatic
  dependent launch (PDL) between launches, so layer *l*+1's weight loads overlap layer
  *l*'s tail; split-K across the co-resident grid at small M.
* **other**: one-off networks (vocoders, VAE decoders, convolution stacks): fused
  convolutions in a layout of your own, activations kept on chip between ops.

## The staged plan

The digest lists the stages in order (the planner's `native` entry, else derived from the
ceilings table: stages by their time above the floor, then the loop body, then the loop).
Work on the **current** stage. It is done when a native end-to-end run of it beats the
**bar** (the best module-level result end to end: the integration or the systems agent's
best run); only then does the next stage start. Name every project directory after its
stage (`<stage id>/`, manifest `name = "<stage id>"` or `<stage id>_<variant>`): the
ledger finds a stage's runs by that name.

Every stage beating the bar once does not end the work. The digest's **After the staged
plan** section then lists the stages of a re-profile of your best run (the stage times
moved) by their time above the floor and marks the **focus**: the stage with the most time
left. Keep improving your best run there, or on a group of stages fused across their
boundaries when the time sits between them.

## Interface to the PyTorch model

* **Weights are shared.** Read the reference modules' parameters and buffers; convert them
  once (packing, FP8 with scales, a tiled layout) in `build()` / `apply()` and rebind
  (`param.data = packed`), never per call. No copies of the checkpoint.
* **Outputs go to the model's unchanged modules** (the decoder, the vocoder, the sampler's
  caller) in the dtype, shape, layout and device they expect, with the same in-place side
  effects (KV caches, state buffers) as the code you replace.
* **Keep the quality seams.** A workload may check quality teacher forced by wrapping one
  module call per loop iteration (its file in `workloads/` says which). A loop engine must
  keep calling that module once per iteration from Python, with the noise drawn as before:
  split the persistent kernels at that seam (one or two persistent kernels per iteration,
  not one per run).
* **Precision**: what the run allows (the `# Precision` section). A stage target is
  captured at the precision the module kernels already use.

## Correctness

1. Stage targets: `evaluate_candidate(target_id="native_<id>", candidate="<dir>")` checks
   every captured case (outputs and side effects), redrawn inputs and aliasing, exactly as
   for module kernels, then times the stage against the reference stage. `mode="quick"`
   is free.
2. End to end: `evaluate_e2e(transforms=["<dir>"], kernels=[...])` runs the full workload:
   output comparison, the held-out input, the natural-length run and, in a near-lossless
   run, the perceptual gate.
3. The integration re-checks kernels in fresh processes and runs them under
   compute-sanitizer memcheck (odd-size variants included): bounds-check every tail tile.

## Integration and ownership

The integration measures each item alone and adds items one at a time by paired A/B. A
native stage owns the modules it replaces: module kernels inside it are its alternatives
(the integration tries the native stage in their place), a transform that only wraps it
(a CUDA graph around it) still composes. Make a stage engine a drop-in for the stage's
module so the rest of the accepted set keeps applying.

## Projects

```
<stage id>/
  kernel_project.toml     [project] name, entry, kind; [build] backend, sources, flags
  candidate.py            build(reference) (kind = "kernel") / apply(workload) ("transform")
  include/*.cuh           device helpers shared by the .cu files
  csrc/*.cu csrc/binding.cpp
```

* `project.load(__file__)` (`from kernel_agent.native import project`) compiles the
  project once per content digest and toolchain (`~/.cache/kernel-agent/native/`) and
  returns it; its attributes are the functions `binding.cpp` binds.
* kernel-agent's toolkit headers are on every build's include path:
  `#include "ka_launch.cuh"` for PDL and cooperative launches (`ka_launch`,
  `ka_pdl_wait`, `ka_pdl_launch_dependents`, `ka_coresident_blocks`; see `cuda.md`).
* `python -m kernel_agent.native.project check <dir>` validates the manifest and files;
  `... build <dir>` compiles on the CPU (no GPU, no evaluation used): fix compiler errors
  there. The tools also compile a project before its evaluation, outside the GPU lock.
* `backend = "command"` runs your own build (`command = ["bash", "{src}/build.sh"]`,
  CMake, make) and loads `outputs` (`load = "torch_ops"` for `TORCH_LIBRARY` libraries,
  `"python"` for extension modules, `"none"` for ctypes).
* Text files only (sources, headers, scripts), at most 200 files / 4 MB; `build/` and
  hidden directories are not part of the project.
* Evaluations snapshot the project's **bundle**: one `.py` file holding every file and the
  project's sha256, which the evaluator, the sweep (`build(reference, **config)` keyword
  arguments), memcheck, the integration and the export use like any candidate file.

## Timing

Stage targets are timed per captured case against the reference stage (CUDA events, full
clocks, interleaved rounds), end-to-end runs as the median of the workload's runs after a
warm-up (compile and capture time excluded). Inside CUDA graphs host overhead does not
count; a persistent engine is still judged by the whole stage's GPU time.

## Rules

* No work on hidden streams or threads (the evaluator checks it): join every side stream.
* No caching of outputs across calls, no skipped work, no reading of unused cache slots.
* Fallbacks only for shapes you do not support, never for the captured dominant case.
