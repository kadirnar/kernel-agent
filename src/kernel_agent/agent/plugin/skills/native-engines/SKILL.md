---
name: native-engines
description: Contract for native engines — scopes (stage, group, loop) and patterns, the staged plan and its bar, the interface to the PyTorch model, correctness, integration, the project layout, build and timing. Use when rewriting part of the inference path as a native CUDA / CuTe engine.
---

# Native engines: contract for the systems-native agent

Module kernels stop paying where the remaining time sits *between* modules: an iterative
solver that re-streams its network's weights at every step, a layer stack or decode step
made of many short launches and grid syncs, glue kernels between stages, host round trips
inside a loop. A native engine rewrites such a part of the inference path as native CUDA
C++ / CuTe code, keeps the weights streaming, fuses across module boundaries and runs the
part in one persistent kernel or a few launches.

Project layout, building and timing: [projects.md](projects.md).

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
   output comparison, the held-out input, the natural-length run and, in a near-lossless or
   relaxed run, the perceptual gate.
3. The integration re-checks kernels in fresh processes and runs them under
   compute-sanitizer memcheck (odd-size variants included): bounds-check every tail tile.

## Integration and ownership

The integration measures each item alone and adds items one at a time by paired A/B. A
native stage owns the modules it replaces: module kernels inside it are its alternatives
(the integration tries the native stage in their place), a transform that only wraps it
(a CUDA graph around it) still composes. Make a stage engine a drop-in for the stage's
module so the rest of the accepted set keeps applying.

A transform that builds on an evaluated engine (speculative decoding around a decode-step
engine, say) loads that engine's bundle by its snapshot name:
`artifacts.load("NNN_<project>_<sha8>.py")` (`from kernel_agent import artifacts`; it looks
next to the calling file, in its `history/`, `../history/` and `..`). Never load a run file
by an absolute path, relative to the run directory or from the project directory: the
export copies what an accepted item loads through `artifacts` (or names in a string
literal) into `optimized/`, and its self-test applies the package in a fresh process that
cannot read the run directory.

## Rules

* No work on hidden streams or threads (the evaluator checks it): join every side stream.
* No caching of outputs across calls, no skipped work, no reading of unused cache slots.
* Fallbacks only for shapes you do not support, never for the captured dominant case.

## Examples and sources

* Examples: `examples/native_project`. All in kernel-agent's examples directory (`kernel_agent/agent/examples/`; a session's prompt gives the directory): copy their structure; an example's `ARCHS` names the GPUs it runs on.
* Sources: the `documentation-sources` skill's `sources.md`, sections "CUDA C++ and PTX"; "CUTLASS / CuTe".
* Related skills: `cuda-kernels`, `cute-dsl`, `cuda-graphs-streams-pdl`, `systems-patterns`; design document `docs/NATIVE.md` in the repository.
