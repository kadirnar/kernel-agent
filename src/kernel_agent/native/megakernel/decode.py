"""Decode-step pieces of a megakernel schedule (issue #225, milestones 6 and 7): split-KV
attention over a KV cache, its combine, and the step state a decode step advances on the
device.

**Split-KV attention** (:class:`SplitKV`). One query token's attention over a KV cache, split
along the cache across SMs: an *attention tile* reads one chunk of one KV head's keys and
values for a block of that head's q heads (GQA: up to :data:`MAX_Q_BLOCK` q heads share every
K / V load) and writes the chunk's max, sum and unnormalised output (fp32); a *combine tile*
per KV head waits on one counter, its attention tiles' (target: their count, a chunked
counter), and reduces them: ``o = sum_c exp(m_c - M) o_c / sum_c exp(m_c - M) l_c`` (the
flash-decoding math).

**The length is a device value.** The schedule has a fixed number of attention tiles,
``splits`` per KV head and q block; the opcode reads the valid length on the device (the step
state's position + 1) and derives each tile's chunk from it (:func:`kv_split`, the
opcode's rule): ``chunk`` keys per split, or with ``chunk=0`` the length spread evenly over
the splits (rounded up to :data:`CHUNK_GRANULE` keys), so every split has work at every
length. A chunk past the length writes nothing and the combine reads only the non-empty
ones. Nothing about the length is known when the schedule is built: its costs
(:meth:`SplitKV.ops`) assume a length, and :func:`attention_durations` gives the simulator
the tiles' durations at any length.

**The step state** (:data:`STEP_TOKEN`, :data:`STEP_POS`): int32 words on the device. A decode
step's embedding reads the token, its RoPE / KV append and its attention read the position,
and its last instruction (argmax + advance) writes the next token and the position + 1: the
next launch needs no host value, and a loop of steps is a loop of graph replays. Every
reader of the state precedes the advance through the schedule's edges (directly or through
other instructions: the acquire / release chain is transitive), so the advance never races a
read; :func:`check_advance` says which reader does not.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from kernel_agent.native.megakernel.schedule import Edge, Instr, Op, Schedule, ScheduleError

#: The step state's words (int32 on the device): the token the step embeds, the position it
#: appends its K / V at (and attends up to, inclusive).
STEP_TOKEN, STEP_POS = 0, 1
STEP_WORDS = 2
#: q heads one attention tile serves: each thread keeps 8 columns of each one's q and output
#: in registers (the example's opcode; a larger group is split into blocks of this many).
MAX_Q_BLOCK = 4
#: Head dimensions the example's attention opcode has (16-byte loads, D / 8 threads per key).
HEAD_DIMS = (64, 128, 256)
#: Keys per chunk of a balanced split (``chunk=0``) are a multiple of this.
CHUNK_GRANULE = 16
#: Floats of shared scratch the combine has for the chunk weights of its q heads.
COMBINE_SLOTS = 2048
#: The cost of an instruction that moves no bytes worth counting (an empty chunk, a combine):
#: the same unit as the bytes the GEMV tiles cost.
FIXED_COST = 4096.0


def kv_split(length: int, splits: int, chunk: int = 0) -> tuple[int, int]:
    """``(keys per chunk, non-empty chunks)`` of a KV length: the attention opcode's rule.
    ``chunk > 0``: that many keys per split (``splits * chunk`` must cover the capacity);
    ``chunk == 0``: ``ceil(length / splits)`` rounded up to :data:`CHUNK_GRANULE`, at least one
    granule. Split ``s`` covers keys ``[s * keys, min((s + 1) * keys, length))``."""
    if splits < 1:
        raise ValueError(f"splits {splits}: at least one")
    length = max(0, length)
    if chunk > 0:
        keys = chunk
    else:
        per = -(-length // splits)
        keys = max(CHUNK_GRANULE, -(-per // CHUNK_GRANULE) * CHUNK_GRANULE)
    return keys, min(splits, -(-length // keys))


def q_blocks(group: int, block: int = MAX_Q_BLOCK) -> list[tuple[int, int]]:
    """``(first, count)`` of the q-head blocks of one KV head's group of ``group`` q heads
    (offsets within the group), balanced: 7 heads in blocks of 4 are 4 + 3."""
    if group < 1 or not 1 <= block <= MAX_Q_BLOCK:
        raise ValueError(f"group {group}, block {block}: group >= 1, 1 <= block <= {MAX_Q_BLOCK}")
    n = -(-group // block)
    base, extra = divmod(group, n)
    out, first = [], 0
    for k in range(n):
        count = base + (1 if k < extra else 0)
        out.append((first, count))
        first += count
    return out


@dataclass(frozen=True)
class AttentionTile:
    """Attention tile ``index``: q heads ``[q0, q0 + qn)`` (global indices) of KV head
    ``kv_head`` over chunk ``split``."""

    index: int
    kv_head: int
    q0: int
    qn: int
    split: int


@dataclass(frozen=True)
class SplitKV:
    """One layer's decode attention over a KV cache of ``capacity`` positions per KV head:
    ``kv_heads`` KV heads of ``group`` q heads each (GQA; 1: multi-head), head dimension
    ``head_dim``, ``splits`` attention tiles per KV head and q block, ``chunk`` keys per split
    (0: the length spread over the splits), ``elem_bytes`` per cached element (2: bf16 /
    fp16), q heads per attention tile at most ``q_block``.

    Tile order: attention tile ``(kv_head * blocks + block) * splits + split``; combine tile
    ``h``: all q heads of KV head ``h``. Partial rows (the opcode's buffers):
    ``q_head * splits + split``."""

    kv_heads: int
    group: int
    head_dim: int
    capacity: int
    splits: int
    chunk: int = 0
    elem_bytes: int = 2
    q_block: int = MAX_Q_BLOCK

    def __post_init__(self) -> None:
        if self.kv_heads < 1 or self.group < 1 or self.capacity < 1 or self.splits < 1:
            raise ScheduleError(f"{self}: heads, group, capacity and splits must be >= 1")
        if self.head_dim not in HEAD_DIMS:
            raise ScheduleError(f"head_dim {self.head_dim}: one of {HEAD_DIMS}")
        if not 1 <= self.q_block <= MAX_Q_BLOCK:
            raise ScheduleError(f"q_block {self.q_block}: 1 to {MAX_Q_BLOCK}")
        if self.chunk < 0 or (self.chunk and self.splits * self.chunk < self.capacity):
            raise ScheduleError(
                f"{self.splits} splits of {self.chunk} keys do not cover {self.capacity} "
                "positions (chunk=0 spreads the length over the splits)"
            )
        if self.group * self.splits > COMBINE_SLOTS:
            raise ScheduleError(
                f"{self.group} q heads x {self.splits} splits: the combine weighs at most "
                f"{COMBINE_SLOTS} chunks"
            )

    @property
    def q_heads(self) -> int:
        return self.kv_heads * self.group

    @property
    def blocks(self) -> list[tuple[int, int]]:
        """The q-head blocks of a KV head's group (``(first, count)`` within the group)."""
        return q_blocks(self.group, self.q_block)

    @property
    def attention_tiles(self) -> int:
        return self.kv_heads * len(self.blocks) * self.splits

    @property
    def combine_tiles(self) -> int:
        return self.kv_heads

    @property
    def partial_rows(self) -> int:
        """Rows of the partial buffers: one per q head and split."""
        return self.q_heads * self.splits

    def attention_tile(self, t: int) -> AttentionTile:
        if not 0 <= t < self.attention_tiles:
            raise IndexError(f"attention tile {t} of {self.attention_tiles}")
        blocks = self.blocks
        head_block, split = divmod(t, self.splits)
        kv_head, block = divmod(head_block, len(blocks))
        first, count = blocks[block]
        return AttentionTile(t, kv_head, kv_head * self.group + first, count, split)

    def combine_tile(self, t: int) -> tuple[int, int, int]:
        """``(kv_head, q0, qn)`` of combine tile ``t``: every q head of KV head ``t``."""
        if not 0 <= t < self.combine_tiles:
            raise IndexError(f"combine tile {t} of {self.combine_tiles}")
        return t, t * self.group, self.group

    def kv_head_of(self, t: int) -> tuple[int]:
        """An :class:`Edge`'s ``needs`` from a producer with one tile per KV head (the RoPE /
        KV append) to the attention tiles: tile ``t`` needs its KV head's."""
        return (self.attention_tile(t).kv_head,)

    def combine_needs(self, t: int) -> range:
        """An :class:`Edge`'s ``needs`` from the attention tiles to the combine tiles: combine
        ``t`` needs every chunk of every q block of KV head ``t`` (one counter, that target)."""
        per = len(self.blocks) * self.splits
        return range(t * per, (t + 1) * per)

    def keys(self, t: int, length: int) -> int:
        """Keys attention tile ``t`` reads at a KV length."""
        keys, _ = kv_split(min(length, self.capacity), self.splits, self.chunk)
        first = self.attention_tile(t).split * keys
        return max(0, min(min(length, self.capacity), first + keys) - first)

    def tile_bytes(self, t: int, length: int) -> int:
        """K and V bytes attention tile ``t`` reads at a KV length (0: an empty chunk)."""
        return 2 * self.keys(t, length) * self.head_dim * self.elem_bytes

    def ops(
        self,
        attention: str,
        combine: str,
        opcodes: tuple[int, int],
        attention_args: Callable[[AttentionTile], Sequence[int]],
        combine_args: Callable[[int, int, int], Sequence[int]],
        *,
        length: int | None = None,
    ) -> tuple[Op, Op, Edge]:
        """The attention op (``opcodes[0]``, one tile per :meth:`attention_tile`, its
        arguments from ``attention_args``), the combine op (``opcodes[1]``, its arguments
        from ``combine_args(kv_head, q0, qn)``) and the edge between them (chunked counters
        per KV head). Costs: the bytes each tile reads at ``length`` (default: the capacity,
        the longest a step can be) plus :data:`FIXED_COST`."""
        at = self.capacity if length is None else length
        n = self.attention_tiles
        attn = Op(
            attention,
            opcodes[0],
            n,
            cost=[FIXED_COST + self.tile_bytes(t, at) for t in range(n)],
            args=lambda t: attention_args(self.attention_tile(t)),
        )
        comb = Op(
            combine,
            opcodes[1],
            self.combine_tiles,
            cost=FIXED_COST,
            args=lambda t: combine_args(*self.combine_tile(t)),
        )
        return attn, comb, Edge(attention, combine, self.combine_needs)


def attention_durations(
    attention: Mapping[str, SplitKV],
    length: int,
    *,
    ns_per_byte: float,
    fixed_ns: float,
    other: Callable[[Instr], float] | Mapping[str | int, float] | None = None,
) -> Callable[[Instr], float]:
    """Durations for :func:`.simulate.simulate` at a KV length: an instruction of an attention
    op named in ``attention`` takes ``fixed_ns`` plus its tile's K / V bytes at ``length``
    times ``ns_per_byte`` (an empty chunk: ``fixed_ns``); the others take ``other`` (a
    function, a table per op name or opcode, or their cost)."""

    def duration(instr: Instr) -> float:
        split = attention.get(instr.op)
        if split is not None:
            return fixed_ns + split.tile_bytes(instr.tile, length) * ns_per_byte
        if callable(other):
            return other(instr)
        if other is not None:
            for key in (instr.op, instr.opcode):
                if key in other:
                    return float(other[key])
        return instr.cost

    return duration


def ancestors(schedule: Schedule, instr: int) -> set[int]:
    """The instructions that ``instr`` waits for, directly or through others."""
    seen: set[int] = set()
    todo = list(schedule.instrs[instr].deps)
    while todo:
        i = todo.pop()
        if i not in seen:
            seen.add(i)
            todo.extend(schedule.instrs[i].deps)
    return seen


def check_advance(schedule: Schedule, readers: Iterable[str], advance: str) -> list[str]:
    """Why the token advance (the instructions of op ``advance``) could race a read of the
    step state: every instruction of the ``readers`` ops must precede it through the
    schedule's edges. Empty: none can."""
    adv = [i.id for i in schedule.instrs if i.op == advance]
    if not adv:
        return [f"no instruction of {advance}"]
    names = set(readers)
    problems = []
    for a in adv:
        before = ancestors(schedule, a)
        for instr in schedule.instrs:
            if instr.op in names and instr.id not in before:
                problems.append(
                    f"{instr.op}[{instr.tile}] (instruction {instr.id}) reads the step state "
                    f"but does not precede the advance {a}"
                )
    return problems
