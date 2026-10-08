---
name: systems-patterns
description: Model-level patterns for generation loops — removing host syncs (async flag reads, constants built once, pinned copies, static shapes), a post stage on a joined side stream, continuous batching, loading run files through kernel_agent.artifacts. Use when writing transforms or when the profile shows host syncs or idle gaps.
---

# Systems patterns: host syncs, side streams, serving, run files

Model-level patterns for any generation loop (LLM decode, a TTS patch or frame loop, an STT
decoder, a sampler with a per-request output stage). They change *when* the host and the
GPU wait for each other, not the math, so they are exact when done right.

Data-dependent speedups (speculative decoding, early exit) are in the `speculative-decoding` skill; overlap inside kernels (PDL, streams in CUDA graphs) in `cuda-graphs-streams-pdl`.

Give every `evaluate_e2e` (and every set of `evaluate_e2e_batch`) a `title`: a commit subject of at most 72 characters saying what this combination changes (`graph the decode step, static KV cache`). It names the experiment in the ledger, the charts and `kernel-agent exp`; the `hypothesis` says why.

## Host synchronisation

A loop that reads a device value on the host every step makes the host wait until the GPU
has drained everything queued so far; then the GPU idles while the host launches the next
step. Typical sources: `.item()` / `.cpu()` / `bool(tensor)` of a stop token or a finished
mask, `torch.tensor([pos], device=...)` per step (built on the host, copied from pageable
memory, which synchronises the stream), `.to(device)` of an unpinned host tensor, and
data-dependent shapes (`nonzero`, boolean-mask indexing, `unique`).

The profile's **Host synchronisation** table lists every such call of the run by call site
(file, line, the source line, calls per run, host time): start there, not with a guess.

Fixes, in order of preference:

* **Async flag read.** Compute the flag as early in the step as it exists (a stop head
  depends only on the hidden state of the previous step), copy it to pinned memory without
  blocking and read it after the rest of the step is queued, or one step later:

  ```python
  flags = workload.async_flags()  # kernel_agent.workloads.serving.AsyncFlags
  for step in range(max_steps):
      ticket = flags.send(stop_logits(hidden).argmax(-1))  # pinned, non_blocking, event
      out = sample(...)  # queue the step's heavy work
      if flags.read(ticket).all():  # waits for the flag copy, not for `sample`
          break
      hidden = advance(out)
  ```

  Reading in the same step is bit-identical (same values, same decisions). Reading a step
  later lets the host run one step ahead but computes one speculative step after the stop:
  discard it, and keep the output length unchanged.
* **Build constants once.** Positions, masks and index tensors made with
  `torch.tensor(..., device=...)` per call: make them once and keep them on the device
  (advance a device-side position with `+= 1`; precompute masks for every length you meet).
  A constant the model rebuilds on every call (a sample-rate condition, a timestep table)
  can usually be cached on the module.
* **Pinned, non-blocking copies.** A host tensor that must go to the device every step:
  pin it once (`pin_memory()`), copy with `non_blocking=True`, and do not touch the host
  buffer again before the copy has completed (an event, `serving.HostCopy`).
* **Static shapes.** Masks and `torch.where` instead of selecting rows; a known bound
  (`output_size=`, padding) instead of a size only the device knows.

A sync inside the workload's own loop (*where* = workload) is the benchmark's code: a
transform can still replace the method that holds it (`workload.<method> = ...`) as long as
the outputs, the teacher-forcing hooks and the per-request bookkeeping stay the same. A sync
in the model's package (*where* = model) is the usual transform target: wrap or replace the
method around the call site.

## A post-processing stage on a side stream

Vocoders, VAEs, codec decoders and detokenisers run after the generation of a request (or of
a chunk) and nothing in the loop depends on them. Run them on a side stream so they overlap
the next request's (or the next step's) generation:

```python
stage = kernel_agent.workloads.serving.SideStage(device)  # a torch stream, fork/join
copy = stage.submit(vocoder, latents)  # waits for the work queued so far, runs on the side
...  # the loop goes on
stage.join()  # before run() returns
audio = copy.wait()  # host copy of the result, its own event only
```

Rules (the evaluator times every stream: the run ends with a device-wide synchronize):
issue all GPU work from the calling thread, join every stream before `run()` returns (work
that outlives `run()` is hidden work and fails), keep the inputs of side-stream work alive
until the join (`SideStage` does), and do not read a host copy before its event. Gains are
bounded by how much of the stage the GPU can overlap: a GPU-bound loop hides only its
kernel-boundary bubbles (a few % of the stage); a host-bound stage (an eager vocoder per
chunk) hides almost entirely behind a GPU-bound loop. A stateful chunk decoder per step is
host-bound unless it is captured in a CUDA graph with static state buffers.

`torch.cuda.graph` captures multi-stream work when the fork and the join happen inside the
capture (`side.wait_stream(cur)` ... `cur.wait_stream(side)`).

## Serving: continuous batching

A batch of requests with natural lengths runs until its longest request stops; the finished
slots compute nothing useful. `kernel_agent.workloads.serving.serve` drives a workload's
batched loop as a `SlotModel` (`admit`, `step`, `advance`, `finish`) over a request queue:
static batches, or continuous batching (a slot is refilled as soon as its request stops),
with the async stop check, the post stage per finished request (on a `SideStage` with
`-o pipeline=true`) and each request's latency (`metric_detail.request_ms`, `_mean`,
`_max`). It is a benchmark option of the workload (`-o serving=continuous`), not a
transform: the default fixed-length benchmark is unchanged and every candidate is compared
on the same schedule.

## Building on another file of the run

A transform that uses another evaluated file (a native project's bundle, an earlier
snapshot, a helper module) loads it by its snapshot name through `kernel_agent.artifacts`:

```python
from kernel_agent import artifacts

engine = artifacts.load("012_decode_loop_engine_1a2b3c4d.py")  # a history/ snapshot: the module
path = artifacts.find("012_decode_loop_engine_1a2b3c4d.py")  # or only its path
```

It looks next to the calling file, then in its `history/`, `../history/` and `..`, so the
same call works from your copy, from the evaluated snapshot and from the exported
`optimized/` package. The export copies what an accepted transform loaded this way (and any
run file whose name is a string literal in it) next to it, then applies the package in a
fresh process that cannot read the run directory; a transform that reads a run file by an
absolute path, relative to the run directory or by a glob for the newest snapshot fails
that self-test.

## Examples and sources

* Sources: the `documentation-sources` skill's `sources.md`, sections "PyTorch (2.14)"; "CUDA C++ and PTX".
* Code: `kernel_agent.workloads.serving` (`AsyncFlags`, `SideStage`, `HostCopy`, `serve`), `kernel_agent.artifacts`, `kernel_agent.concurrency`.
