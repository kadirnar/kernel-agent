"""Megakernel schedules: a stage's ops as per-SM instruction queues whose dependencies are
global-memory counters, not grid barriers (issue #225).

Inputs, op level and model-agnostic (built from a stage's captured ops and the ceilings
table, never from module names):

* :class:`Op`: an opcode, its number of tiles (output tiles, K splits, ...), a cost
  estimate per tile, the weight bytes each tile streams (:class:`Prefetch`: loaded into the
  shared-memory page pool *before* the tile's activation counters are met) and each tile's
  opcode arguments;
* :class:`Edge`: producer → consumer at tile granularity: which producer tiles each consumer
  tile needs (:data:`ALL`, :data:`SAME`, :func:`span`, :func:`shard` or any function).

:func:`build` turns them into a :class:`Schedule`:

* **Counters.** A producer op's tiles are split into *chunks*: the coarsest partition in
  which every consumer tile's needed set is a union of chunks (one chunk when every consumer
  needs every tile; one per tile for a one-to-one edge: chunked counters per output tile).
  A chunk is one counter: each of its tiles adds 1 after writing its output
  (``red.release.gpu``) and a consumer tile waits (``ld.acquire.gpu``) until every counter of
  its needed chunks reaches its *target*, the chunk's tile count.
* **Order.** Ops run in waves: an op's ``wave``, by default its depth in the op graph; every
  producer's wave precedes its consumers'. Within a wave the tiles are placed longest first
  (LPT) on the queue where they would finish earliest, given their producers' estimated
  finish times (list scheduling). Every queue runs its tiles in this one global order, so the
  earliest unfinished instruction can always run: no deadlock while every block is resident,
  which the cooperative launch guarantees.
* **Serialisation** (:meth:`Schedule.words`): one int32 array, built once in ``build()`` /
  ``apply()`` and reused by every call. The ABI, mirrored in ``include/ka_mk.cuh``:

  ====================  =====================================================================
  header (16 words)     MAGIC, VERSION, queues, instructions, counters, waits, WORDS,
                        queue table offset, instruction offset, wait offset, total words
  queue table           queues + 1 instruction indices: queue q runs rows [t[q], t[q + 1])
  instructions          WORDS (32) int32 each: opcode, id, the first wait inline (counter,
                        -1: none, and target), the other waits (first index in the wait
                        list, count), signal counter (-1: none), signal amount, prefetch
                        tensor (-1: none), prefetch offset (16-byte units), prefetch bytes,
                        prefetch L2 hint (:data:`EVICT_FIRST`, ...), 20 opcode arguments
  waits                 (counter, target) pairs: the waits after an instruction's first
  ====================  =====================================================================

An instruction's id is its row in the instruction array (queue 0's rows first): the trace
and the watchdog report it, :attr:`Schedule.instrs` maps it back to its op and tile.
:mod:`.simulate` checks a schedule for deadlocks and ordering and predicts its time.
"""

from __future__ import annotations

import itertools
import struct
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

MAGIC = 0x4B414D4B  # "KAMK"
VERSION = 1
WORDS = 32  # int32 words per instruction: the opcode and 31 arguments
HEADER_WORDS = 16
# header words
H_MAGIC, H_VERSION, H_QUEUES, H_INSTRS, H_COUNTERS, H_WAITS, H_WORDS = range(7)
H_QUEUE_OFF, H_INSTR_OFF, H_WAIT_OFF, H_TOTAL = range(7, 11)
# instruction words (the first wait inline: most instructions wait on one counter, which
# the interpreter then reads without a trip to the wait list)
OP, ID, W0_COUNTER, W0_TARGET, WAIT_FIRST, WAIT_COUNT, SIGNAL, SIGNAL_N = range(8)
PF_TENSOR, PF_OFFSET, PF_BYTES, PF_HINT = range(8, 12)
ARG0 = 12
# PF_HINT: the L2 policy of a prefetch (evict_first: streamed once; evict_last: keep in L2)
EVICT_NORMAL, EVICT_FIRST, EVICT_LAST = 0, 1, 2
ARGS = WORDS - ARG0  # opcode-specific arguments per instruction
NONE = -1
PREFETCH_ALIGN = 16  # bulk / cp.async copies move 16-byte units
INT32_MAX = 2**31 - 1

#: Every consumer tile needs every producer tile.
ALL = "all"
#: Consumer tile t needs producer tile t (equal tile counts).
SAME = "same"


class ScheduleError(ValueError):
    """Ops and edges that do not make a schedule."""


def span(k: int) -> Callable[[int], range]:
    """Consumer tile ``t`` needs producer tiles ``[t * k, (t + 1) * k)`` (a reduction of k
    partial tiles, a down projection that waits per chunk of its input)."""
    if k < 1:
        raise ScheduleError(f"span({k}): k must be at least 1")
    return lambda t: range(t * k, (t + 1) * k)


def shard(k: int) -> Callable[[int], tuple[int]]:
    """Consumer tile ``t`` needs producer tile ``t // k`` (k consumers per producer tile)."""
    if k < 1:
        raise ScheduleError(f"shard({k}): k must be at least 1")
    return lambda t: (t // k,)


@dataclass(frozen=True)
class Prefetch:
    """Weight bytes an instruction streams into the page pool before its dependencies are
    met: ``nbytes`` from byte ``offset`` of tensor ``tensor`` of the runtime's tensor table
    (both multiples of 16). ``hint``: the L2 policy: :data:`EVICT_FIRST` for weights larger
    than the L2 that every call streams once (they leave the program and the activations in
    L2), :data:`EVICT_LAST` for weights that fit and should stay, :data:`EVICT_NORMAL`."""

    tensor: int
    offset: int
    nbytes: int
    hint: int = 0


TileArgs = Sequence[Sequence[int]] | Callable[[int], Sequence[int]]
TilePrefetch = Sequence[Prefetch | None] | Callable[[int], Prefetch | None]


@dataclass(frozen=True, eq=False)
class Op:
    """One op of a stage: ``tiles`` instructions of ``opcode``. ``cost``: estimated time per
    tile (any unit; a number or one per tile), ``args``: each tile's opcode arguments (at
    most :data:`ARGS` int32), ``weights``: each tile's :class:`Prefetch` (or None), ``wave``:
    its place in the static order (default: its depth in the op graph)."""

    name: str
    opcode: int
    tiles: int
    cost: float | Sequence[float] = 1.0
    args: TileArgs = ()
    weights: TilePrefetch | None = None
    wave: int | None = None

    def tile_cost(self, t: int) -> float:
        return float(self.cost) if isinstance(self.cost, int | float) else float(self.cost[t])

    def tile_args(self, t: int) -> tuple[int, ...]:
        if callable(self.args):
            return tuple(int(a) for a in self.args(t))
        return tuple(int(a) for a in self.args[t]) if self.args else ()

    def tile_prefetch(self, t: int) -> Prefetch | None:
        if self.weights is None:
            return None
        return self.weights(t) if callable(self.weights) else self.weights[t]


@dataclass(frozen=True, eq=False)
class Edge:
    """``consumer`` reads what ``producer`` writes: ``needs`` maps a consumer tile to the
    producer tiles it needs (:data:`ALL`, :data:`SAME`, :func:`span`, :func:`shard` or a
    function returning tile indices)."""

    producer: str
    consumer: str
    needs: str | Callable[[int], Iterable[int]] = ALL


@dataclass(frozen=True)
class Counter:
    """A chunk of a producer op's tiles: each signals it once; ``target``: their count."""

    id: int
    op: str
    tiles: tuple[int, ...]
    target: int


@dataclass(frozen=True)
class Instr:
    """One instruction: ``id`` is its row in the instruction array."""

    id: int
    op: str
    opcode: int
    tile: int
    queue: int
    cost: float
    waits: tuple[tuple[int, int], ...]  # (counter, target)
    signal: int  # counter id or NONE
    prefetch: Prefetch | None
    args: tuple[int, ...]
    deps: tuple[int, ...]  # ids of the producer instructions it needs
    est_start: float = 0.0
    est_finish: float = 0.0


@dataclass(frozen=True)
class Schedule:
    """Per-queue instruction lists (one queue per resident block), their counters and the
    list-scheduling estimate of the makespan."""

    queues: tuple[tuple[int, ...], ...]
    instrs: tuple[Instr, ...]
    counters: tuple[Counter, ...]
    ops: tuple[str, ...] = ()
    est_makespan: float = 0.0
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_queues(self) -> int:
        return len(self.queues)

    @property
    def max_prefetch(self) -> int:
        """The largest prefetch of one instruction, in bytes (it must fit the page pool)."""
        return max((i.prefetch.nbytes for i in self.instrs if i.prefetch), default=0)

    def words(self) -> list[int]:
        """The serialised schedule (the ABI of the module docstring)."""
        n_instr = len(self.instrs)
        waits: list[int] = []
        rows: list[int] = []
        for instr in self.instrs:
            first = len(waits) // 2
            inline = instr.waits[0] if instr.waits else (NONE, 0)
            for counter, target in instr.waits[1:]:
                waits += [counter, target]
            pf = instr.prefetch
            row = [
                instr.opcode,
                instr.id,
                *inline,
                first,
                len(instr.waits) - 1 if instr.waits else 0,
                instr.signal,
                1 if instr.signal != NONE else 0,
                pf.tensor if pf else NONE,
                pf.offset // PREFETCH_ALIGN if pf else 0,
                pf.nbytes if pf else 0,
                pf.hint if pf else EVICT_NORMAL,
                *instr.args,
            ]
            rows += row + [0] * (WORDS - len(row))
        queue_off = HEADER_WORDS
        # rows start on a 128-byte boundary: one row is one cache line, read in 16-byte units
        instr_off = -(-(queue_off + len(self.queues) + 1) // WORDS) * WORDS
        wait_off = instr_off + n_instr * WORDS
        total = wait_off + len(waits)
        header = [0] * HEADER_WORDS
        header[H_MAGIC], header[H_VERSION] = MAGIC, VERSION
        header[H_QUEUES], header[H_INSTRS] = len(self.queues), n_instr
        header[H_COUNTERS], header[H_WAITS] = len(self.counters), len(waits) // 2
        header[H_WORDS] = WORDS
        header[H_QUEUE_OFF], header[H_INSTR_OFF], header[H_WAIT_OFF] = (
            queue_off,
            instr_off,
            wait_off,
        )
        header[H_TOTAL] = total
        table = [0]
        for queue in self.queues:
            table.append(table[-1] + len(queue))
        pad = [0] * (instr_off - queue_off - len(table))
        words = header + table + pad + rows + waits
        if bad := [w for w in words if not -(2**31) <= w <= INT32_MAX]:
            raise ScheduleError(f"words outside int32: {bad[:3]}")
        return words

    def to_bytes(self) -> bytes:
        """:meth:`words` as little-endian int32 (the same ops give the same bytes)."""
        words = self.words()
        return struct.pack(f"<{len(words)}i", *words)

    def tensor(self, device: Any = "cpu") -> Any:
        """:meth:`words` as an int32 torch tensor on ``device`` (build it once, reuse it)."""
        import torch

        return torch.tensor(self.words(), dtype=torch.int32, device=device)

    def describe(self) -> str:
        """A few lines: instructions per op, queues, counters, waits, the estimate."""
        per_op: dict[str, int] = {}
        for instr in self.instrs:
            per_op[instr.op] = per_op.get(instr.op, 0) + 1
        sizes = [len(q) for q in self.queues]
        n_waits = sum(len(i.waits) for i in self.instrs)
        ops = ", ".join(f"{name} x{n}" for name, n in per_op.items())
        return (
            f"{len(self.instrs)} instructions ({ops}) on {len(self.queues)} queues "
            f"({min(sizes, default=0)}-{max(sizes, default=0)} each), "
            f"{len(self.counters)} counters, {n_waits} waits, "
            f"estimated makespan {self.est_makespan:.4g}"
        )


# ------------------------------------------------------------------ building


def _needs(edge: Edge, producer: Op, consumer: Op) -> list[tuple[int, ...]]:
    """The producer tiles each consumer tile needs (sorted, unique, in range)."""
    out: list[tuple[int, ...]] = []
    for t in range(consumer.tiles):
        if edge.needs == ALL:
            tiles: Iterable[int] = range(producer.tiles)
        elif edge.needs == SAME:
            if producer.tiles != consumer.tiles:
                raise ScheduleError(
                    f"edge {producer.name} -> {consumer.name}: SAME needs equal tile counts "
                    f"({producer.tiles} vs {consumer.tiles})"
                )
            tiles = (t,)
        elif callable(edge.needs):
            tiles = edge.needs(t)
        else:
            raise ScheduleError(
                f"edge {producer.name} -> {consumer.name}: bad needs {edge.needs!r}"
            )
        need = tuple(sorted({int(p) for p in tiles}))
        if not need:
            raise ScheduleError(
                f"edge {producer.name} -> {consumer.name}: tile {t} needs no producer tile"
            )
        if need[0] < 0 or need[-1] >= producer.tiles:
            raise ScheduleError(
                f"edge {producer.name} -> {consumer.name}: tile {t} needs producer tiles "
                f"{need[0]}..{need[-1]}, the producer has {producer.tiles}"
            )
        out.append(need)
    return out


def _chunks(tiles: int, sets: Iterable[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """The coarsest partition of the needed tiles of ``range(tiles)`` in which every set of
    ``sets`` is a union of blocks (partition refinement), blocks ordered by first tile.
    Tiles no set holds are in no block (nobody waits for them)."""
    block = [-1] * tiles  # block of each tile; -1: not needed
    members: list[set[int]] = []
    for need in sets:
        touched: dict[int, list[int]] = {}
        for p in need:
            touched.setdefault(block[p], []).append(p)
        for b, inside in touched.items():
            if b == -1 or len(inside) < len(members[b]):
                if b != -1:
                    members[b].difference_update(inside)
                members.append(set(inside))
                for p in inside:
                    block[p] = len(members) - 1
    return sorted((tuple(sorted(m)) for m in members if m), key=lambda m: m[0])


def _waves(ops: Sequence[Op], producers: Mapping[str, list[str]]) -> dict[str, int]:
    """Each op's wave: its own ``wave`` or 1 + its producers' latest (0 without any)."""
    by_name = {op.name: op for op in ops}
    waves: dict[str, int] = {}
    visiting: set[str] = set()

    def wave(name: str) -> int:
        if name in waves:
            return waves[name]
        if name in visiting:
            raise ScheduleError(f"the op graph has a cycle through {name}")
        visiting.add(name)
        deps = [wave(p) for p in producers.get(name, [])]
        explicit = by_name[name].wave
        waves[name] = explicit if explicit is not None else (1 + max(deps) if deps else 0)
        visiting.discard(name)
        return waves[name]

    for op in ops:
        wave(op.name)
    return waves


def _check_op(op: Op) -> None:
    if not op.name or op.tiles < 1:
        raise ScheduleError(f"op {op.name!r}: a name and at least one tile")
    if not 0 <= op.opcode <= INT32_MAX:
        raise ScheduleError(f"op {op.name}: opcode {op.opcode} outside int32")
    for t in range(op.tiles):
        if op.tile_cost(t) < 0:
            raise ScheduleError(f"op {op.name}: negative cost of tile {t}")
        args = op.tile_args(t)
        if len(args) > ARGS:
            raise ScheduleError(f"op {op.name}: tile {t} has {len(args)} arguments (max {ARGS})")
        pf = op.tile_prefetch(t)
        if pf is not None and (
            pf.nbytes <= 0
            or pf.offset < 0
            or pf.tensor < 0
            or pf.offset % PREFETCH_ALIGN
            or pf.nbytes % PREFETCH_ALIGN
            or pf.offset // PREFETCH_ALIGN > INT32_MAX
            or pf.nbytes > INT32_MAX
        ):
            raise ScheduleError(
                f"op {op.name}: tile {t} prefetch {pf}: offset and size must be non-negative "
                f"multiples of {PREFETCH_ALIGN} bytes (size > 0) within int32"
            )
        if pf is not None and pf.hint not in (EVICT_NORMAL, EVICT_FIRST, EVICT_LAST):
            raise ScheduleError(f"op {op.name}: tile {t} prefetch hint {pf.hint} (0, 1 or 2)")


def build(
    ops: Sequence[Op],
    edges: Sequence[Edge],
    queues: int,
    *,
    meta: Mapping[str, Any] | None = None,
) -> Schedule:
    """The schedule of ``ops`` and ``edges`` on ``queues`` resident blocks (one per SM of
    the launch: ``ka_coresident_blocks``). Deterministic: the same ops and edges give the
    same schedule, byte for byte (:meth:`Schedule.to_bytes`)."""
    if queues < 1:
        raise ScheduleError("a schedule needs at least one queue")
    by_name: dict[str, Op] = {}
    for op in ops:
        if op.name in by_name:
            raise ScheduleError(f"two ops named {op.name}")
        _check_op(op)
        by_name[op.name] = op
    index = {op.name: i for i, op in enumerate(ops)}
    producers: dict[str, list[str]] = {}
    incoming: dict[str, list[tuple[Edge, list[tuple[int, ...]]]]] = {}
    outgoing: dict[str, list[tuple[int, ...]]] = {}
    for edge in edges:
        if edge.producer not in by_name or edge.consumer not in by_name:
            raise ScheduleError(f"edge {edge.producer} -> {edge.consumer}: unknown op")
        if edge.producer == edge.consumer:
            raise ScheduleError(f"edge {edge.producer} -> {edge.consumer}: a self edge")
        needs = _needs(edge, by_name[edge.producer], by_name[edge.consumer])
        producers.setdefault(edge.consumer, []).append(edge.producer)
        incoming.setdefault(edge.consumer, []).append((edge, needs))
        outgoing.setdefault(edge.producer, []).extend(needs)
    waves = _waves(ops, producers)
    for edge in edges:
        if waves[edge.producer] >= waves[edge.consumer]:
            raise ScheduleError(
                f"edge {edge.producer} -> {edge.consumer}: the producer's wave "
                f"{waves[edge.producer]} must precede the consumer's {waves[edge.consumer]}"
            )

    # counters: the chunks of each producer op, numbered by op order then first tile
    counters: list[Counter] = []
    chunk_of: dict[tuple[str, int], int] = {}  # (op, tile) -> counter id
    for op in ops:
        for chunk in _chunks(op.tiles, outgoing.get(op.name, [])):
            counter = Counter(len(counters), op.name, chunk, len(chunk))
            counters.append(counter)
            for t in chunk:
                chunk_of[(op.name, t)] = counter.id

    # placement: waves in order; LPT within a wave; earliest estimated finish per tile, ties
    # to the queue with the least work so far (serial ops do not pile up on queue 0)
    free = [0.0] * queues
    busy = [0.0] * queues
    placed: dict[tuple[str, int], tuple[int, float, float]] = {}  # -> queue, start, finish
    lists: list[list[tuple[str, int]]] = [[] for _ in range(queues)]
    for wave in sorted(set(waves.values())):
        tiles = [(op.name, t) for op in ops if waves[op.name] == wave for t in range(op.tiles)]
        tiles.sort(key=lambda nt: (-by_name[nt[0]].tile_cost(nt[1]), index[nt[0]], nt[1]))
        for name, t in tiles:
            ready = 0.0
            for edge, needs in incoming.get(name, []):
                for p in needs[t]:
                    ready = max(ready, placed[(edge.producer, p)][2])
            cost = by_name[name].tile_cost(t)
            best = min(range(queues), key=lambda q: (max(free[q], ready) + cost, busy[q], q))
            start = max(free[best], ready)
            placed[(name, t)] = (best, start, start + cost)
            free[best] = start + cost
            busy[best] += cost
            lists[best].append((name, t))

    ids: dict[tuple[str, int], int] = {}
    for q in range(queues):
        for key in lists[q]:
            ids[key] = len(ids)
    instrs: list[Instr] = []
    for q in range(queues):
        for name, t in lists[q]:
            op = by_name[name]
            waits: set[tuple[int, int]] = set()
            deps: set[int] = set()
            for edge, needs in incoming.get(name, []):
                for p in needs[t]:
                    counter = counters[chunk_of[(edge.producer, p)]]
                    waits.add((counter.id, counter.target))
                    deps.add(ids[(edge.producer, p)])
            _, start, finish = placed[(name, t)]
            instrs.append(
                Instr(
                    id=ids[(name, t)],
                    op=name,
                    opcode=op.opcode,
                    tile=t,
                    queue=q,
                    cost=op.tile_cost(t),
                    waits=tuple(sorted(waits)),
                    signal=chunk_of.get((name, t), NONE),
                    prefetch=op.tile_prefetch(t),
                    args=op.tile_args(t),
                    deps=tuple(sorted(deps)),
                    est_start=start,
                    est_finish=finish,
                )
            )
    rows = tuple(tuple(ids[key] for key in lists[q]) for q in range(queues))
    return Schedule(
        queues=rows,
        instrs=tuple(instrs),
        counters=tuple(counters),
        ops=tuple(op.name for op in ops),
        est_makespan=max(free),
        meta=dict(meta or {}),
    )


def check_words(words: Sequence[int]) -> str | None:
    """Why ``words`` is not a well-formed serialised schedule (None: it is): the checks the
    interpreter's header validation and the watchdog rely on."""
    if len(words) < HEADER_WORDS:
        return "shorter than the header"
    if words[H_MAGIC] != MAGIC or words[H_VERSION] != VERSION or words[H_WORDS] != WORDS:
        return "bad magic, version or instruction width"
    n_q, n_i, n_c, n_w = words[H_QUEUES], words[H_INSTRS], words[H_COUNTERS], words[H_WAITS]
    if words[H_TOTAL] != len(words):
        return f"total {words[H_TOTAL]} words, the array has {len(words)}"
    if words[H_INSTR_OFF] % WORDS:
        return "the instruction rows do not start on a 128-byte boundary"
    table = words[words[H_QUEUE_OFF] : words[H_QUEUE_OFF] + n_q + 1]
    if table[0] != 0 or table[-1] != n_i or any(b < a for a, b in itertools.pairwise(table)):
        return "the queue table is not a partition of the instructions"
    base = words[H_INSTR_OFF]
    waits = words[words[H_WAIT_OFF] :]
    for i in range(n_i):
        row = words[base + i * WORDS : base + (i + 1) * WORDS]
        if row[ID] != i:
            return f"instruction {i} has id {row[ID]}"
        if row[WAIT_FIRST] < 0 or row[WAIT_FIRST] + row[WAIT_COUNT] > n_w:
            return f"instruction {i}: waits outside the wait list"
        if not NONE <= row[W0_COUNTER] < n_c or (row[W0_COUNTER] != NONE and row[W0_TARGET] < 1):
            return f"instruction {i}: inline wait on counter {row[W0_COUNTER]}"
        if not NONE <= row[SIGNAL] < n_c:
            return f"instruction {i}: signal counter {row[SIGNAL]} of {n_c}"
    for k in range(n_w):
        counter, target = waits[2 * k], waits[2 * k + 1]
        if not 0 <= counter < n_c or target < 1:
            return f"wait {k}: counter {counter}, target {target}"
    return None
