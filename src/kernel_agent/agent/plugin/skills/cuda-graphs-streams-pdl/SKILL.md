---
name: cuda-graphs-streams-pdl
description: Overlap the evaluator accepts — declared fork / join side streams (also inside CUDA graphs), programmatic dependent launch (PDL), cooperative and persistent grids, SM partitions, and where the CUDA-graph rules are. Use when chains of short kernels or independent branches leave the GPU idle.
---

# CUDA graphs, streams and PDL

## Streams, PDL and SM partitions (`kernel_agent.concurrency`)

Overlap is legal when it is joined and declared (the evaluator rejects the rest):
launch from the calling thread, join every stream before the call returns, name
side streams with `kernel_agent.concurrency` (`cc`).

* **Fork/join** (Python, around your launchers):
  `with cc.fork("aux"): y = branch(x)` (the stream waits for the caller's on entry,
  the caller's waits for it on exit), or `h = cc.launch("aux", fn, *args)` …
  `h.result()` (joins exactly that work; `cc.join_all()` joins the rest). Inside
  `torch.cuda.graph(...)` the same fork/join becomes graph edges (multi-stream
  capture): two independent branches in one graph ran 1.09-1.16x faster than
  serial (docs/PARALLEL.md §4.4). Your C++ launchers keep using
  `at::cuda::getCurrentCUDAStream()`: inside `cc.fork` that is the side stream.
* **PDL** (programmatic dependent launch, sm_90+): the next kernel on the stream
  launches while the current one drains. Header `ka_launch.cuh`:
  `load_inline(..., extra_include_paths=[str(cc.include_dir())])` and
  `#include "ka_launch.cuh"`; launch with
  `KaLaunch opt; opt.pdl = true; C10_CUDA_CHECK(ka_launch(kernel, grid, block, smem, stream, opt, args...));`
  (`cudaLaunchKernelEx` with programmatic stream serialization; dropped below
  sm_90). Device idiom: issue the loads that do not depend on the previous kernel
  (weights, constants), then `ka_pdl_launch_dependents(); ka_pdl_wait();`, then read
  what the previous kernel wrote; write nothing before the wait. Hash the header
  into the extension `name` with your source.
  - Pays for graph-captured chains of short memory-bound kernels whose prologue is
    a weight load: 28 dependent GEMVs of 2 MB, one CUDA graph 101.2 us -> graph +
    PDL 77.7 us = the DRAM floor (76.7 us); a graph kernel boundary costs ~0.9 us.
    Eager launches gain little (host-bound): capture the chain in a graph.
  - Hurt or noise where the wait came before the independent loads, and on
    GEMM -> GEMM edges in our runs; cuBLAS / cuBLASLt kernels between yours do not
    take part.
  - Example: `examples/cuda_pdl_gemv_chain.py` (the chain captured in `build()`,
    `pdl` a `build()` keyword for `sweep_candidate`).
* **Cooperative / persistent grids**: `opt.cooperative = true` (grid-wide sync)
  needs every block resident: size the grid by
  `ka_coresident_blocks(kernel, threads, smem, stream)`, which counts the SMs of the
  stream's green-context partition (`ka_sm_count(stream)`; in Python
  `cc.sm_count()`). `cudaDevAttrMultiProcessorCount` reports the whole device even
  inside a partition, and a cooperative grid sized from it is rejected there. A
  hand-rolled grid barrier cost ~2.2-2.4 us per layer on sm_120, more than a PDL
  graph boundary (~0.9 us): prefer one kernel per phase + PDL in a graph for
  memory-bound chains.
* **Partitions** (`cc.partition(sms=k)`: green contexts, two disjoint SM sets from
  one driver split, None where unsupported): they buy isolation, not throughput
  (two jobs on two plain streams finished first in every pair measured).

## CUDA graphs elsewhere in the skills

* Static KV cache + `torch.compile(mode="reduce-overhead")` / CUDA graphs for decode loops, graphs on a diffusion denoiser: `optimisation-playbook` (`model-transforms.md`, `model-families.md`; the VoxCPM2 entry also says which graphs teacher forcing allows).
* Multi-stream capture of a side stage and its rules: `systems-patterns`.
* Host time does not count under graphs (Triton launches, `torch._scaled_mm`, cuBLASLt): `triton-kernels`, `fp8-w8a8`.

## Examples and sources

* Examples: `examples/cuda_pdl_gemv_chain.py`. All in kernel-agent's examples directory (`kernel_agent/agent/examples/`; a session's prompt gives the directory): copy their structure; an example's `ARCHS` names the GPUs it runs on.
* Sources: the `documentation-sources` skill's `sources.md`, sections "CUDA C++ and PTX"; "PyTorch (2.14)"; "Triton".
* Code: `kernel_agent.concurrency` (`fork`, `launch`, `join_all`, `include_dir`, `sm_count`, `partition`), header `ka_launch.cuh`.
