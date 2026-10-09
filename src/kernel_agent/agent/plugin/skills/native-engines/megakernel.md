# Megakernels: the kit and the milestone ladder

Part of the `native-engines` skill. A megakernel runs a whole stage (a decode step, a layer
stack) as **one launch**: one block per SM interprets a queue of instructions that Python
scheduled ahead of time, and instructions wait for their inputs on **counters** in global
memory instead of grid barriers. Reach for it when a stage's best kernel synchronises its
whole grid or launches more than ~3 kernels per call (the digest says so under
`## Megakernel kit`): each hand-rolled grid barrier drains the weight stream (~2.2–2.4 µs
per barrier, a graph kernel boundary ~0.9 µs; measured on an RTX 5070 Ti, docs/PARALLEL.md
§4.6), and a counter lets every tile start the moment *its* inputs exist.

## The kit (`kernel_agent.native.megakernel`)

| piece | what |
|---|---|
| `ka_mk.cuh` (on every project's include path, with `ka_launch.cuh`) | the interpreter `ka_mk::kernel<Ops>`: consumer threads run the instructions, a **producer warp** streams their weights into the shared-memory **page pool** (full / empty mbarrier per page; `cp.async.bulk` on sm_90+ incl. sm_120, `cp.async` + mbarrier on sm_80–sm_89; an L2 evict-first / evict-last hint; `inflight` caps the pages in flight); `ka_mk::wait` (an acquire read, relaxed polls and an acquire fence, nanosleep back-off) and `ka_mk::signal` (`red.release.gpu`); `ka_mk::sync<kThreads>()` (the consumers' barrier); the **watchdog**; `KA_MK_TRACE` (+ two `ctx.mark(k)` stamps an opcode sets) |
| `schedule.py` | `Op` (opcode, tiles, cost per tile, `Prefetch` per tile, opcode arguments) and `Edge` (`ALL`, `SAME`, `span(k)`, `shard(k)` or any tile map) → per-SM queues (static wave order + longest-first, earliest-finish placement), every counter's target (its producers' tile count; chunked per output tile where consumers need only part of a producer), the int32 program (`Schedule.tensor()`, built once in `build()` / `apply()`) |
| `simulate.py` | `check(schedule, draws=1000)`: deadlocks (the stuck counter and instruction first) and instructions started before their producers, over random SM speeds; `simulate(..., durations=costs_from_trace(...))` predicts the time; `report()` and `trace_summary()` say where it goes (wait, land, run per op; how often a weight load overlapped a counter wait) |
| `runtime.py` | `Runtime(schedule, tensors, trace=..., pool_bytes=...)`: program, counters (self-zeroing: the last block of a launch resets them), tensor table, pinned status words; `check()` raises `MegakernelHang` before a launch after a hang |

An `Ops` struct holds the project's opcodes (`kThreads` consumer threads, `kPageBytes`,
`kScratchBytes`, `run(ctx)`: `ctx.op()`, `ctx.arg(k)`, `ctx.ptr<T>(tensor)`,
`ctx.weights(byte)` from the pool, `ctx.scratch`, `ctx.mark(k)`). Opcodes synchronise with
`ka_mk::sync<kThreads>()`, never `__syncthreads()` (the producer warp does not take part).
Host side: `ka_mk::pool_pages<Ops>()` (pages that fit this GPU's
shared memory), `ka_mk::queues<Ops>(pages, stream)` (resident blocks via
`ka_coresident_blocks`, green-context aware: build the schedule for exactly that many
queues), `ka_mk::launch<Ops>(params, queues, stream)` (cooperative attribute: every block is
resident, so a waiting block never waits for one that has not started).

Start from `examples/native_megakernel`: an RMSNorm → GEMV → residual chain with generic
opcodes (RMSNorm, GEMV tiles with bf16 or e4m3 weights, fused norm prologue, residual or
split-K-partial epilogue, residual add, split-K reduce, argmax) and two baselines from the
same math in the same project (`mode="graph_pdl"`: one kernel per layer with PDL edges;
`mode="coop_barrier"`: one cooperative kernel with a grid barrier per layer), so
`sweep_candidate` compares all three on your GPU.

## Rules of the interpreter

* Activations another block wrote in this launch are read after the wait's acquire with
  plain loads (L1-cacheable: the warps' repeated reads hit L1) or `__ldcg`, never `__ldg`
  (its read-only cache can be stale). Weights come from the pool (`ctx.weights`, read with
  `ld.shared`), constants with `__ldg`.
* Every global load an instruction needs after its wait goes out at once (x, the norm's
  weights, the residual): one round trip on the critical path, not three. Then all the
  pool loads, then the arithmetic: back-to-back loads, independent FMA chains.
* Keep few pages in flight per block (`inflight`, 1 on an RTX 5070 Ti): the consumers'
  critical loads queue behind the block's own outstanding weight copies.
* Every edge the data needs is declared, including write-after-read reuse of a buffer
  (or give each layer its own activation row, as the example does).
* `simulate.check(schedule)` before the first launch; a launch that hangs anyway stops after
  the watchdog's limit (`KA_MK_WATCHDOG_MS`, 2 s; 10 min under the sanitizers) and the
  evaluator records status `hang` with the instruction, counter, value and target.
* Native candidates must give the same bits twice (the evaluator's determinism check).
  Declare `ORDER_DEPENDENT_ATOMICS = "<why>"` in the entry only when float atomics sum in a
  varying order on purpose (split-K reduced with `atomicAdd`); prefer a reduce instruction.
* The integration runs every native candidate under memcheck, **racecheck** (shared-memory
  hazards: a page reused while still read) and **synccheck** (barriers some threads miss).

## Reuse across calls (the stress check)

Every launch reuses what the previous one left: the counters (zeroed by the launch's last
block, after every block finished its queue), the activation rows, the pages, the program.
The rule that keeps that safe:

* **One launch of a runtime at a time, in stream order.** The next launch may only start
  once the previous one completed: launch every call on the same stream (or join streams),
  never two calls of one `Runtime` concurrently, and no PDL edge into a launch that reuses
  the previous launch's counters before its blocks exit.
* **Nothing is reused within a launch before its last reader is done.** A page is refilled
  only after every consumer thread arrived on its empty barrier; a buffer row is rewritten
  only behind an edge from every instruction that reads it (or never: one row per layer);
  words the interpreter needs after an instruction's last barrier (its signal) are read
  before it (warp 0 may already refill the row ring for a later chunk).
* **Counters start at zero and end at zero.** Every signal a launch sends is waited for in
  that launch, so the last block's reset never races a late `red`; a counter nobody waits on
  is not signalled (`schedule.py` gives such tiles no counter).

The evaluator checks this for every native candidate whose sources use counters or atomics:
after the two-run check it runs the main case 256 times back to back
(`$KERNEL_AGENT_STRESS_CALLS`), on four inputs in turn and with GPU-side gaps before some, and
every call must equal an isolated call on the same input bit for bit (`determinism.stress`).
On the example: 0 of 2,000 calls differ (28 layers, 1 or all pages in flight); the same
schedule with its counter waits removed: 500 of 500 calls differ.

Not every rare failure is a race: the example's exact-tier check against eager (cuBLAS
rounding) fails on 1.3 % of random inputs at 8 layers and on none at 4 (3,000 draws, issue
#248), and the graph + PDL and grid-barrier baselines, which use no counters, fail at 1.0 %.
Tell them apart with the stress check: a race differs from an isolated call on the same
input, a rounding drift does not. The evaluator judges a failed redrawn-input check again
with the reference's own rounding spread (#250): at 8 layers none of 3,000 draws and none of
100 evaluations are refused any more; the captured inputs keep the plain tolerance.

## Measured (RTX 5070 Ti, sm_120; the example, 28 layers)

| | H = 1024 (58.7 MB) | H = 2048 (235 MB) |
|---|---|---|
| DRAM floor (833 GB/s) | 70.5 µs | 282.0 µs |
| graph + PDL, one kernel per layer | 77.9 µs | 289.8 µs |
| **megakernel** (counters, producer warp, 16-row tiles) | **84–87 µs** | **335 µs** |
| one cooperative kernel, grid barrier per layer | 198 µs | 451 µs |

The megakernel is 2.3× faster than the grid-barrier kernel and 8–12 % slower than graph +
PDL on this chain. Its trace says where the rest goes: per layer ~1.1 µs waiting for the
previous layer's slowest tile and its signal, ~0.3 µs for the last weight page, ~1.4 µs of
run (activation loads ~0.5, products ~0.3, reduction, signal). Every weight load was issued
before its counter was met (trace `overlap` 1.0). These are the next rungs to climb, not a
limit: a chain this short is the megakernel's worst case (one dependent tile per SM per
layer); stages with independent ops per layer (Q, K, V, gate, up) overlap them.

## The milestone ladder

Climb it one rung at a time; each rung is evaluated (`evaluate_candidate`, the stage target)
before the next. A rung that does not pay is a measurement, not a reason to stop: the next
rungs attack other costs.

1. **One correct opcode.** The interpreter with one opcode (the stage's dominant GEMV
   tile), every tile independent; correct against the stage, deterministic.
2. **Counters replace barriers.** The real edges (`Edge(producer, consumer, ALL)` per layer);
   `simulate.check` clean; compare against the grid-barrier version.
3. **Page pool and cross-instruction prefetch.** Weights stream into the pool while blocks
   wait (`Prefetch` per tile, the L2 hint for weights larger than half the L2): the trace's
   `overlap` near 1, `land` small.
4. **Chunked counters.** Edges at tile granularity (`span`, `shard`, a map): a consumer
   waits only for the producer tiles it reads (a down projection per chunk of its input).
5. **Split-K across SMs.** Skinny GEMMs with too few output tiles to fill the SMs: K splits
   write fp32 partials, a reduce instruction sums them (deterministic, unlike atomics).
6. **Attention opcode.** Split-KV attention per (head, KV chunk) and a combine opcode for
   decode; prefill keeps its own kernels (not in the kit's example yet: write it in the
   project's `Ops`).
7. **On-device argmax and token advance.** The sampler (argmax opcode) and the next step's
   inputs inside the kernel: no host round trip between tokens.
8. **Tuned wave order.** `Op.wave` overrides the depth order (start the next layer's
   independent work earlier); the scheduler checks every edge still goes forward.
9. **Trace-driven rebalancing.** `trace=True` → `costs_from_trace` → `simulate` with the
   measured costs: move the tiles of the critical path, split slow ops into more tiles,
   change rows per tile.
10. **PDL / graph integration.** Capture the launch with the stage's neighbours in one CUDA
    graph; let the megakernel's prologue (weight prefetch) overlap the previous kernel.

## Examples and sources

* Example: `examples/native_megakernel` (in kernel-agent's examples directory; its `ARCHS`).
* Sources: the `documentation-sources` skill's `sources.md`, section "CUDA C++ and PTX"
  (memory consistency model: release / acquire patterns; `cp.async.bulk`, mbarrier).
* Design: docs/PARALLEL.md §4.6 (grid barrier vs PDL measurements) in the repository.
