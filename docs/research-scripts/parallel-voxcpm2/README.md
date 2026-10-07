# Parallelisation on VoxCPM2: research scripts (issue #136)

The scripts behind the measurements in [`docs/PARALLEL.md`](../../PARALLEL.md), kept as they
were run (RTX 5070 Ti, torch 2.14.1+cu130, the accepted sets of the latency and batch-16 runs
applied from read-only copies). One-off benchmarks, not part of the library (excluded from
ruff); paths to scratch copies are as they were. Run GPU jobs under the GPU lock.

* `prof.py`, `analyze.py`, `gaps_detail.py`, `gaps_patch.py`, `topk.py`: torch.profiler traces,
  stage attribution, idle gaps, the slowest kernels against their DRAM floors.
* `bench_streams.py`, `bench_b16_pipeline.py`, `bench_graph_ms.py`: two stages on two streams,
  pipelined VAE, multi-stream CUDA graph capture.
* `bench_green.py`, `bench_green2.py`, `coop_green.py`, `green_sync.py`, `api_check.py`: green
  contexts (SM partitions).
* `bench_persistent.py`, `gemv_ext.py`: a 28-layer GEMV chain as eager launches, one graph,
  graph + PDL, one persistent cooperative kernel.
