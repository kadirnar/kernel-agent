# Decode-step megakernels: split-KV attention and the on-device token advance

Part of the `native-engines` skill, rungs 6 and 7 of the megakernel ladder
([megakernel.md](megakernel.md)). A whole decode step (one token of a batch-1 decoder) as one
launch of the interpreter, and a run of steps as graph replays with no host value between
them.

## The example (`examples/native_megakernel`, the decode step)

`TinyDecoder` is the torch reference: a pre-norm decoder (RMSNorm, fused QKV,
rotate-half RoPE from fp32 cos / sin tables, grouped-query attention over a KV cache with the
softmax in fp32, O projection + residual, SwiGLU MLP, final norm, LM head, greedy argmax),
rounded as an eager model rounds. `build_decode(reference, mode="megakernel")` builds
`DecodeStep`; `mode="graph"` the same opcodes as one launch per op in a CUDA graph (kernel
boundaries instead of counters; its results are the megakernel's, bit for bit). Its ops, in
`decode_program` (pure Python: the CPU tests build and simulate it):

| op | opcode | tiles |
|---|---|---|
| embedding of the device's token | `EMBED` | 1 |
| per layer: RMSNorm + QKV projection | `GEMV` (fused norm) | rows / 16 |
| RoPE + KV append at the device's position | `ROPE_KV` | one per KV head |
| split-KV attention | `ATTN_DECODE` | KV heads x q blocks x splits |
| combine | `ATTN_COMBINE` | one per KV head |
| O projection + residual | `GEMV` | rows / 16 |
| RMSNorm + gate / up projection | `GEMV` (fused norm) | rows / 16 |
| SwiGLU | `GLU` | intermediate / 256 |
| down projection + residual | `GEMV` | rows / 16 |
| final RMSNorm + LM head | `GEMV` (fused norm) | vocab / 16 |
| argmax + token advance | `ARGMAX` (`advance=1`) | 1 |

Every layer has its own activation rows, so no buffer is rewritten while an instruction of
the same step may read it. The step state is two int32 words on the device
(`decode.STEP_TOKEN`, `STEP_POS`); `reset(token, pos, cache)` sets it (and the caches) from
the host once, `step()` replays the graph, `generate(n)` replays it n times; the history
(int32 per position) records the tokens on the device. The host counts the position itself
(no device read) and refuses a step once the cache is full.

## Split-KV attention (`ATTN_DECODE`, `ATTN_COMBINE`, `decode.SplitKV`)

* **Tiles.** One attention tile per (KV head, block of up to 4 of its q heads, split): it
  reads its chunk of that head's cache once for all the block's q heads (GQA) and writes the
  chunk's max (log2 domain), sum and unnormalised output, fp32, into rows
  `prow0 + q_head * splits + split`. A combine tile per KV head waits on one counter (target:
  that head's attention tiles, `SplitKV.combine_needs`) and reduces the non-empty chunks:
  `o = sum_c exp(m_c - M) o_c / sum_c exp(m_c - M) l_c`. The RoPE tile of a KV head is its
  attention tiles' only producer (`SplitKV.kv_head_of`: one counter per KV head, target 1).
* **The length is a device value**: the step state's position + 1, read by the opcode, never
  baked into the schedule. The schedule has a fixed number of splits; the chunk follows the
  length (`decode.kv_split`, the opcode's rule): `chunk` keys per split, or with `chunk=0`
  the length spread over the splits (rounded up to 16), so every split has work at every
  length and short contexts do not leave SMs idle. A chunk past the length writes nothing;
  the combine derives the same chunks. Default splits: about one wave of the queues
  (`default_splits`: queues / (KV heads x q blocks)).
* **Inside a tile.** Thread t owns 8 columns (one 16-byte load) of key t / (D / 8) of each
  pass: a warp reads whole contiguous rows, the 256 threads 4 KB of K and of V per pass, two
  passes in flight. The score sums over the key's D / 8 threads (a shuffle butterfly); each
  thread keeps an online softmax over its keys; at the end the threads of a warp combine
  (butterfly), then the warps in turn through shared memory (as many slots as the 8 KB
  scratch holds). Every reduction has a fixed order: the same bits on every call.
* **Reads.** The current position's K / V row is appended in this launch (by `ROPE_KV`):
  read the cache with plain or `__ldcg` loads after the acquire, never `__ldg`.
* **Registers.** The interpreter has 288 threads, so ptxas gives each 168 registers. Four
  heads' q and output columns plus the loads in flight fit two passes (four spilled ~100
  bytes on sm_80-sm_120); a larger GQA group is split into q blocks of 4 (K / V read once
  per block). Heavy opcodes go in their own `Ops` struct (`DecodeOps`: the generic opcodes
  and the decode ones): the chain's `kernel<Ops>` keeps its 128 registers and no spills.
  `nvcc -Xptxas -v` says it; check it after every opcode you add.
* **Layouts the opcode lacks** (head dim 96, more than 4 q heads per tile, a dtype other
  than bf16 / fp16) stop the interpreter with status `bad_opcode` and the instruction;
  `SplitKV` refuses them when the schedule is built.

## The token advance (`ARGMAX` with `advance=1`)

`argmax_args(..., advance=1, step=<state>, hist=<history>, hist_len=...)`: thread 0 writes
the argmax (first maximum, NaN first, as `torch.argmax`) as the next token, the position + 1,
and the history at that position. Everything that reads the state in the step (the
embedding, every RoPE, attention and combine tile) must precede the advance through the
schedule's edges, or the advance could overwrite the position under a late reader:
`decode.check_advance(schedule, readers, "argmax")` lists any reader that does not (the
example checks it when it builds). The acquire / release chain is transitive, so a reader
needs a path to the advance, not an edge.

## Checking it

* The opcodes against torch: the fp32 softmax at lengths 1, 2, around the chunk and granule
  boundaries and the maximum (the length changed on the device between launches of one
  schedule), every head dim, bf16 and fp16, GQA groups that split into q blocks; within one
  ulp of the output's scale; a repeated launch bit-identical.
* The step against the reference at prefilled lengths (logits within bf16 rounding of the
  projections, the appended K / V, the advanced state), then a generation teacher-forced on
  the engine's tokens: every token is the reference's argmax or a bf16 near tie.
* `simulate.check` over random SM speeds, with `decode.attention_durations` at a short and a
  long length (most chunks empty at a short one).
* compute-sanitizer memcheck, racecheck and synccheck on a step and on the attention alone,
  and a deliberately short cache that memcheck must report (a clean run that saw nothing
  proves nothing). Under memcheck a faulting warp stops: a consumer waiting on its counter
  waits for the watchdog (10 minutes under the sanitizers), so a broken run should have no
  waits.

## Measured (NVIDIA A10, sm_86)

The example's decoder, 2 layers (hidden 1024, 16 q / 4 KV heads of 64, intermediate 2048,
vocabulary 8192, KV capacity 4096; 52.4 MB of weights per step), bf16, 18 splits per KV
head, medians of 15 x 16 steps (each step advancing the position on the device; a GPU
shared with another tenant: runs differ by a few %).
`docs/research-scripts/megakernel-a10-225/bench_decode.py` in the repository measures it
on any GPU.

| KV length | DRAM floor (487 GB/s) | **megakernel** (inflight 2) | graph, one launch per op (19) |
|---|---|---|---|
| 16 | 107.7 µs | **158.7 µs** | 257.1 µs |
| 128 | 108.1 µs | **160.9 µs** | 258.4 µs |
| 1024 | 111.9 µs | **166.8 µs** | 266.4 µs |
| 4096 | 124.8 µs | **188.0 µs** | 303.3 µs |

The megakernel is 1.6x faster than the same opcodes launched per op at every length, at
66-68 % of the floor's speed; every engine gives the same bits and the reference's
argmax. Its trace per layer at length 16 (mean per tile): RoPE waits 3.5 µs and runs 1.5,
attention waits 5.4 and runs 1.2, the combine waits 5.4 and runs 2.3, the O projection
waits 8.0 for it: the attention's chain of three dependent hops is the step's longest
wait. At 4096 an attention tile runs 15.5 µs: ~12 µs of keys (240 keys of K and V, about
380 GB/s summed over the SMs while the producer warps stream the next weights) and ~3.5 µs
of reductions, a fixed cost at every length.

## Next rungs

Measured next steps, not limits:

* **RoPE and the KV append inside the attention tiles** (each tile rotates its q block, the
  tile holding the position appends K / V): one dependent hop less per layer (~5 µs of
  wait and run on the critical path above). Its arguments need more than the 20 argument
  words: a second row or a packed table.
* **A cheaper intra-tile reduction** (~3.5 µs per tile at any length): the warps' max and
  sum once per head instead of in every thread, fewer shared-memory rounds.
* **K / V through the page pool** at long contexts: the producer warp prefetching the
  chunk's rows ahead of the counters needs a length-dependent prefetch (the length is a
  device value).
* **SwiGLU in the down projection's prologue** (the GLU hop: ~2.8 µs per layer) and more
  combine tiles (one per q head) to shorten the combine.
