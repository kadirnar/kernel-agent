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
| `captured.py` | the schedule's input built from one recorded call of the stage instead of by hand: `python -m kernel_agent.native.megakernel.schedule --from-capture <capture>` (a target's `capture.pt` / `capture_inputs.pt`) or `captured.from_module(reference, args)` in `build()` → a `Plan`: ops mapped to opcode families, unsupported ops with why, tiles, the tensor table (`plan.tensors()`, `plan.bind()`), tile-level edges from the storages, roofline costs; `plan.schedule(queues)` (next section) |
| `opcodes.py` | the generic opcodes' numbers and argument slots (`gemv_args`, `rmsnorm_args`, ...; the example's `include/mk_ops.cuh` is their device code), `tile_rows`, `l2_hint` |
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
split-K-partial epilogue, residual add, split-K reduce, argmax, gated activation `GLU`),
its op DAG declared by hand (`chain_dag`) or built from a recorded call of the reference
(`schedule="captured"`, next section), and two baselines from the
same math in the same project (`mode="graph_pdl"`: one kernel per layer with PDL edges;
`mode="coop_barrier"`: one cooperative kernel with a grid barrier per layer), so
`sweep_candidate` compares all three on your GPU. PDL needs sm_90+: before (sm_80–sm_89)
`graph_pdl` is a plain graph, and the engine's `label` says so (`graph_label`): compare
against it as a graph, not as graph + PDL. `rows=0` (the default) takes 16 rows per tile
where they fit the page pool and fewer where not (`tile_rows`: 16 rows of a 4096-wide layer
are 128 KB, a 99 KB GPU's pool holds 88 KB, so 8).

## Schedules from the stage's captured ops

Do not declare a stage's ops and edges by hand when its capture can say them:

```bash
python -m kernel_agent.native.megakernel.schedule \
    --from-capture targets/native_<id>/capture_inputs.pt [--case K] [--queues N] \
    [--pool-bytes B] [--split-k auto|N] [--no-fuse] [--ceilings profile/ceilings.json] [--json]
```

It records one call of the stage (the case with the most calls per run; `--device cuda` for
modules that only run there) under the fusion miner's op recorder and maps each aten op, by
its structure, never its module name: **RMSNorm** (rsqrt(mean(x²) + eps) · x, the casts and
the weight in either order, or `rms_norm` as one op), **GEMV / skinny GEMM tiles** (`linear` /
`mm` / `matmul` of ≤ 8 activation rows by a bf16 weight the stage only reads, no bias),
**residual add**, **gated activation** (`silu` / `gelu` of one activation times another:
the `GLU` opcode), **argmax**, and **split-K reduce** (a GEMV split over K: always when K
exceeds the 4096-column slice, `--split-k N` / `auto` to fill the SMs). Every other op is
listed as **unsupported, with why** (attention, softmax, LayerNorm, a biased linear, a cast,
a cache write...), and `plan.schedule()` refuses the plan until it is mapped: write its
opcode in your project's `Ops` and schedule it with the rest (`schedule.Op` / `Edge`).
Then it fuses as the example does (a norm read only by GEMVs into their prologue, a
residual into the GEMV's epilogue or its reduce), tiles every op (`tile_rows` for this
GPU's page pool), lays out the tensor table (the GEMV weights, the norms' weights, one arena
per dtype, every value its own slice: no write-after-read edges), derives the edges at tile
granularity from the byte ranges each tile reads and writes (`ALL`, `SAME` or a per-tile
map: chunked counters for free), costs each tile from the roofline of the GPU at hand
(`kernel-agent doctor`'s peaks or a ceilings table's: max(bytes / bandwidth, FLOPs / fp32
peak) on a 1/queues share, labelled), builds the schedule and runs `simulate.check`. It
prints the DAG, the unsupported ops, the schedule and the simulator's report (exit 0: a
clean schedule, 1: unsupported ops or a problem). In a project: `plan =
captured.from_module(reference, (x,), pool_bytes=pages * page_bytes, queues=queues)`,
`plan.schedule(queues)`, `tensors = plan.tensors()`, `Runtime(schedule, tensors, ...)`,
`plan.bind(tensors, plan.inputs[0])` / `plan.outputs[0]` for the call's tensors (the
example's `_setup_captured`).

Measured on an NVIDIA A10 (sm_86, `docs/research-scripts/megakernel-captured-225/`): the
example's chain from its capture is the hand-declared program byte for byte (28 layers at
1024 and 2048; the same bits out, the same time within the bench's ±2 % order effect;
predicted 152.7 µs at 1024, measured 152–162 µs best to median); a decode gated MLP
[1024 → 3072 → 1024] from its recorded call runs in 54.5 µs as one launch against 66.2 µs
for its PyTorch ops in a CUDA graph (71 % of the DRAM floor's speed, bit-identical to
eager), [2048 → 8192 → 2048] in 245.8 µs against 250.8 µs (its down projection split over
K, 8192 > 4096). Splitting every GEMV in two was slower there (74.8 and 302.3 µs): measure
`--split-k`, do not assume it.

## Rules of the interpreter

* Activations another block wrote in this launch are read after the wait's acquire with
  plain loads (L1-cacheable: the warps' repeated reads hit L1) or `__ldcg`, never `__ldg`
  (its read-only cache can be stale). Weights come from the pool (`ctx.weights`, read with
  `ld.shared`), constants with `__ldg`.
* Every global load an instruction needs after its wait goes out at once (x, the norm's
  weights, the residual): one round trip on the critical path, not three. Then all the
  pool loads, then the arithmetic: back-to-back loads, independent FMA chains.
* Keep few pages in flight per block (`inflight`, 1 on an RTX 5070 Ti): the consumers'
  critical loads queue behind the block's own outstanding weight copies. Measure it: on an
  A10 (`cp.async` path, 8 KB per page) 2 was 1–4 % faster than 1, 4 and all slower at 1024.
* Every thread that arrives on a page's empty barrier observes its previous phase there
  before arriving again (`ka_mk.cuh` does; keep it in an interpreter of your own): the
  protocol is correct without it, but compute-sanitizer synccheck reports "Missing wait",
  stops the warp and the integration refuses the kernel (an A10 from 6 layers of the
  example on). Cost there: 1–4 % at 1024, none measurable at 2048.
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
  `kernel-agent doctor`'s `racecheck` / `synccheck` probes say whether each tool reports a
  deliberate hazard on your GPU: on an A10 with compute-sanitizer 2025.2.1 synccheck reports
  no divergent `__syncthreads()` (a clean run says nothing about those barriers there) but
  does check mbarrier phases.

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
PDL on this chain (measured before the consumers observed their previous empty phase,
which cost 1–4 % at 1024 on an A10). Its trace says where the rest goes: per layer ~1.1 µs waiting for the
previous layer's slowest tile and its signal, ~0.3 µs for the last weight page, ~1.4 µs of
run (activation loads ~0.5, products ~0.3, reduction, signal). Every weight load was issued
before its counter was met (trace `overlap` 1.0). These are the next rungs to climb, not a
limit: a chain this short is the megakernel's worst case (one dependent tile per SM per
layer); stages with independent ops per layer (Q, K, V, gate, up) overlap them.

## Measured (NVIDIA A10, sm_86; the example, 28 layers)

No PDL before sm_90, so the launch-per-layer baseline is a plain graph; page loads take
the `cp.async` path; 72 SMs (72 queues), 99 KB of shared memory per block (11 pages of
8 KB), 150 W power cap, a GPU shared with another tenant (medians of 15 x 100 replays;
runs differ by up to ~5 %). `docs/research-scripts/megakernel-a10-225/bench_chain.py` in the
repository measures it on any GPU.

| | H = 1024 (58.7 MB) | H = 2048 (235 MB) | H = 4096 (940 MB) |
|---|---|---|---|
| DRAM floor (487 GB/s copy) | 120.7 µs | 482.6 µs | 1930 µs |
| plain graph, one kernel per layer | 204.4 µs | **532.4 µs** | **1946 µs** |
| **megakernel**, inflight 1 / 2 (16 rows; 8 at 4096) | **157.3 / 155.7 µs** | 571.4 / 566.2 µs | 2578 / 2481 µs |
| one cooperative kernel, grid barrier per layer | 431.8 µs | 848.5 µs | 2140 µs |

At 1024 the megakernel is 1.3x faster than the plain graph and 2.8x faster than the
grid-barrier kernel (77 % of the floor's speed); at 2048 the graph leads by 6 %, at 4096 by
27 %. Its trace at 1024, per layer: ~1.2 µs waiting for the previous layer, ~0.4 µs for
the last weight page, ~3.4 µs of run (x loaded and normalised ~1.2, products ~0.7,
reduction + store + signal ~1.5; an RTX 5070 Ti ran the whole tile in ~1.4); every weight
load issued before its counter was met. Rungs worth measuring there: the tail (the release
fence of the signal, the reduction), tiles per layer against the SM count (64 tiles leave 8
of 72 SMs idle per layer, 128 or 512 tiles end each layer with a partial round) and, at
4096, tiles the 11-page pool can double-buffer (an 8-page tile cannot) and the whole-row
GEMV path, which covers K of 1024 and 2048 only.

## The milestone ladder

Climb it one rung at a time; each rung is evaluated (`evaluate_candidate`, the stage target)
before the next. A rung that does not pay is a measurement, not a reason to stop: the next
rungs attack other costs.

0. **The DAG from the capture.** `--from-capture` on the stage's capture: what maps,
   what does not (the opcodes to write first), the predicted time per op.
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
