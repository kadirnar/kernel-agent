"""Discrete-event simulation of a megakernel schedule (issue #225).

:func:`simulate` runs a :class:`~.schedule.Schedule` the way the interpreter does: every
queue executes its instructions in order; an instruction starts when its queue is free and
every counter it waits on has reached its target; when it ends, it adds its signal to its
counter (after ``latency``, the time a release takes to become visible). It reports

* a **deadlock** (:class:`Blocked`: every queue that cannot go on, with the instruction id,
  the counter, its value and its target) when no instruction can start and some are left:
  a mis-targeted counter, a wait on a counter nobody signals, an order that is not
  topological;
* **ordering violations**: an instruction that started before one of the producer tiles it
  needs (:attr:`Instr.deps`) had finished: a counter target set too low;
* the **predicted time** (makespan, per-op and per-queue busy and waiting time, the
  critical path), from durations per op or opcode: the schedule's cost estimates, or the
  per-op times measured by a trace (:func:`costs_from_trace`), so the agent sees where the
  time goes before and after a run.

:func:`check` repeats the simulation over seeded random per-queue speed draws (some SMs
slower than others, as clocks, L2 slices and co-running work make them) and returns every
problem found.

Traces (``KA_MK_TRACE``, ``include/ka_mk.cuh``): one row of :data:`TRACE_COLUMNS` per
instruction, ``globaltimer`` nanoseconds. :func:`trace_summary` reads them: per op the mean
execution and wait times, and how often the instruction's weight load was issued before its
counters were met (the overlap the page pool exists for).
"""

from __future__ import annotations

import heapq
import itertools
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from kernel_agent.native.megakernel.schedule import NONE, Instr, Schedule

#: Columns of one trace row: when the instruction's weight prefetch was issued, when the
#: queue reached it (its wait began), when its counters were met, when its weights had
#: landed, when it ended (signal issued; ns of the GPU's globaltimer, 0: not recorded), the
#: SM it ran on, and two marks its opcode may set inside its run (``ctx.mark(0 | 1)``).
TRACE_COLUMNS = (
    "prefetch_ns", "start_ns", "ready_ns", "landed_ns", "end_ns", "smid", "mark0_ns", "mark1_ns"
)  # fmt: skip

Durations = Mapping[str | int, float] | Callable[[Instr], float] | None


@dataclass(frozen=True)
class Blocked:
    """A queue that cannot go on: its next instruction waits on a counter below target.
    ``stuck``: every instruction that signals the counter has run, so the target can never
    be reached (a mis-targeted counter, a wait nobody signals): the cause; the other
    blocked queues wait on these."""

    queue: int
    instr: int
    op: str
    tile: int
    counter: int
    value: int
    target: int
    stuck: bool = False

    def describe(self) -> str:
        why = " (every producer has run: the target is wrong)" if self.stuck else ""
        return (
            f"queue {self.queue}: instruction {self.instr} ({self.op}[{self.tile}]) waits on "
            f"counter {self.counter}: {self.value} of {self.target}{why}"
        )


@dataclass
class Simulation:
    """What :func:`simulate` found. Times are in the unit of the durations."""

    makespan: float
    start: list[float | None]
    finish: list[float | None]
    deadlock: list[Blocked] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    busy: list[float] = field(default_factory=list)  # per queue
    waiting: list[float] = field(default_factory=list)  # per queue: idle before a ready instr
    per_op: dict[str, float] = field(default_factory=dict)  # busy time per op
    critical: list[int] = field(default_factory=list)  # instruction ids, first to last

    @property
    def ok(self) -> bool:
        return not self.deadlock and not self.violations

    def problems(self) -> list[str]:
        out = []
        if self.deadlock:
            out.append("deadlock: " + "; ".join(b.describe() for b in self.deadlock))
        return out + self.violations


def _duration(durations: Durations) -> Callable[[Instr], float]:
    if durations is None:
        return lambda instr: instr.cost
    if callable(durations):
        return durations
    table = dict(durations)

    def lookup(instr: Instr) -> float:
        for key in (instr.op, instr.opcode):
            if key in table:
                return float(table[key])
        return instr.cost

    return lookup


def simulate(
    schedule: Schedule,
    *,
    durations: Durations = None,
    speeds: Sequence[float] | None = None,
    latency: float = 0.0,
) -> Simulation:
    """Run ``schedule``: ``durations`` per op name or opcode (or a function of the
    instruction; default its cost estimate), divided by its queue's ``speeds`` factor;
    ``latency``: delay between an instruction's end and its signal's visibility."""
    duration = _duration(durations)
    n = len(schedule.instrs)
    queues = schedule.queues
    speed = list(speeds) if speeds is not None else [1.0] * len(queues)
    if len(speed) != len(queues) or any(s <= 0 for s in speed):
        raise ValueError("one positive speed per queue")
    instrs = schedule.instrs
    value = [0] * len(schedule.counters)
    start: list[float | None] = [None] * n
    finish: list[float | None] = [None] * n
    head = [0] * len(queues)
    free = [0.0] * len(queues)  # when the queue finished its last instruction
    busy = [0.0] * len(queues)
    waiting = [0.0] * len(queues)
    per_op: dict[str, float] = {}
    running = [False] * len(queues)
    # events: (time, kind, seq, payload); kind 0: an instruction ends, 1: a signal lands
    events: list[tuple[float, int, int, int]] = []
    seq = 0
    now = 0.0

    def ready(i: int) -> bool:
        return all(value[c] >= target for c, target in instrs[i].waits)

    def dispatch() -> None:
        nonlocal seq
        for q, queue in enumerate(queues):
            if running[q] or head[q] >= len(queue):
                continue
            i = queue[head[q]]
            if not ready(i):
                continue
            begin = max(now, free[q])
            waiting[q] += begin - free[q]
            took = max(0.0, duration(instrs[i])) / speed[q]
            start[i], finish[i] = begin, begin + took
            busy[q] += took
            per_op[instrs[i].op] = per_op.get(instrs[i].op, 0.0) + took
            running[q] = True
            heapq.heappush(events, (begin + took, 0, seq, i))
            seq += 1

    dispatch()
    while events:
        now, kind, _, i = heapq.heappop(events)
        if kind == 0:  # instruction i ended: its queue moves on, its signal is in flight
            q = instrs[i].queue
            running[q] = False
            head[q] += 1
            free[q] = now
            if instrs[i].signal != NONE:
                heapq.heappush(events, (now + latency, 1, seq, i))
                seq += 1
        else:
            value[instrs[i].signal] += 1
        if events and events[0][0] == now:
            continue  # simultaneous events first: the order within one instant is not real
        dispatch()

    sim = Simulation(
        makespan=max((f for f in finish if f is not None), default=0.0),
        start=start,
        finish=finish,
        busy=busy,
        waiting=waiting,
        per_op=per_op,
    )
    pending: dict[int, int] = {}  # counter -> signals not yet sent
    for instr in instrs:
        if instr.signal != NONE and finish[instr.id] is None:
            pending[instr.signal] = pending.get(instr.signal, 0) + 1
    for q, queue in enumerate(queues):
        if head[q] < len(queue):
            i = queue[head[q]]
            for counter, target in instrs[i].waits:
                if value[counter] < target:
                    stuck = pending.get(counter, 0) == 0
                    instr = instrs[i]
                    blocked = Blocked(
                        q, i, instr.op, instr.tile, counter, value[counter], target, stuck
                    )
                    sim.deadlock.append(blocked)
                    break
    sim.deadlock.sort(key=lambda b: (not b.stuck, b.queue))  # the causes first
    for instr in instrs:
        begin = start[instr.id]
        if begin is None:
            continue
        for dep in instr.deps:
            end = finish[dep]
            if end is None or end > begin:
                sim.violations.append(
                    f"instruction {instr.id} ({instr.op}[{instr.tile}]) started at {begin:.4g} "
                    f"before producer {dep} ({instrs[dep].op}[{instrs[dep].tile}]) "
                    + ("ran" if end is None else f"finished at {end:.4g}")
                )
    if not sim.deadlock:
        sim.critical = _critical_path(schedule, start, finish)
    return sim


def _critical_path(
    schedule: Schedule, start: list[float | None], finish: list[float | None]
) -> list[int]:
    """The chain that set the makespan: from the last instruction to end, back through what
    each one started after (its queue's previous instruction or its latest producer)."""
    if not schedule.instrs:
        return []
    prev: dict[int, int] = {}
    for queue in schedule.queues:
        for a, b in itertools.pairwise(queue):
            prev[b] = a
    i = max(range(len(finish)), key=lambda k: finish[k] or 0.0)
    path = [i]
    while True:
        begin = start[i] or 0.0
        options = [*schedule.instrs[i].deps, *([prev[i]] if i in prev else [])]
        options = [k for k in options if finish[k] is not None]
        if not options or begin <= 0.0:
            break
        i = max(options, key=lambda k: (finish[k] or 0.0, k))
        path.append(i)
    return path[::-1]


def random_speeds(
    queues: int, rng: random.Random, low: float = 0.5, high: float = 2.0
) -> list[float]:
    """One speed factor per queue, uniform in [low, high]."""
    return [rng.uniform(low, high) for _ in range(queues)]


def check(
    schedule: Schedule,
    *,
    draws: int = 1000,
    seed: int = 0,
    durations: Durations = None,
    latency: float = 0.0,
) -> list[str]:
    """Problems found by :func:`simulate` over ``draws`` seeded random speed draws (and the
    estimate's own speeds first); empty: no deadlock and no ordering violation."""
    rng = random.Random(seed)
    found: list[str] = []
    for k in range(draws + 1):
        speeds = None if k == 0 else random_speeds(schedule.n_queues, rng)
        sim = simulate(schedule, durations=durations, speeds=speeds, latency=latency)
        for problem in sim.problems():
            found.append(f"draw {k}: {problem}")
        if found:
            break  # the first draw that fails says enough
    return found


# ------------------------------------------------------------------ traces


def trace_rows(trace: Any) -> list[list[int]]:
    """A trace tensor ([instructions, len(TRACE_COLUMNS)] int64) or nested list as rows."""
    if hasattr(trace, "tolist"):
        trace = trace.tolist()
    return [[int(v) for v in row] for row in trace]


def costs_from_trace(schedule: Schedule, trace: Any) -> dict[str, float]:
    """Mean execution time per op (``end − ready``: waiting for its weights and running, ns)
    of the traced instructions: the ``durations`` that make :func:`simulate` predict a run."""
    sums: dict[str, list[float]] = {}
    for instr, row in zip(schedule.instrs, trace_rows(trace), strict=False):
        ready, end = row[2], row[4]
        if ready and end >= ready:
            sums.setdefault(instr.op, []).append(float(end - ready))
    return {op: sum(v) / len(v) for op, v in sums.items()}


def trace_summary(schedule: Schedule, trace: Any) -> dict[str, Any]:
    """Per op: instructions traced, mean wait (counters), land (weights after the counters)
    and run time (ns), and ``overlap``: the share of its prefetching instructions whose
    weight load was issued before their counters were met. ``span_ns``: first start to last
    end."""
    rows = trace_rows(trace)
    out: dict[str, Any] = {"ops": {}}
    starts = [r[1] for r in rows if r[1]]
    ends = [r[4] for r in rows if r[4]]
    out["span_ns"] = (max(ends) - min(starts)) if starts and ends else 0
    for instr, row in zip(schedule.instrs, rows, strict=False):
        pf, begin, ready, landed, end = row[:5]
        if not end:
            continue
        stats = out["ops"].setdefault(
            instr.op,
            {"n": 0, "wait_ns": 0.0, "land_ns": 0.0, "run_ns": 0.0, "prefetching": 0, "overlap": 0},
        )
        stats["n"] += 1
        stats["wait_ns"] += ready - begin
        stats["land_ns"] += landed - ready
        stats["run_ns"] += end - landed
        if instr.prefetch is not None and pf:
            stats["prefetching"] += 1
            stats["overlap"] += int(pf < ready)
    for stats in out["ops"].values():
        n = stats["n"]
        for key in ("wait_ns", "land_ns", "run_ns"):
            stats[key] = round(stats[key] / n, 1)
        stats["overlap"] = (
            round(stats["overlap"] / stats["prefetching"], 3) if stats["prefetching"] else None
        )
    return out


def report(schedule: Schedule, sim: Simulation, unit: str = "") -> str:
    """Where the time goes: makespan, busy and waiting share, per-op busy time, the
    critical path by op."""
    if sim.deadlock:
        return "deadlock: " + "; ".join(b.describe() for b in sim.deadlock)
    total = sim.makespan * max(1, schedule.n_queues)
    busy = sum(sim.busy)
    lines = [
        f"makespan {sim.makespan:.4g}{unit} on {schedule.n_queues} queues: busy "
        f"{busy / max(total, 1e-30):.0%}, waiting {sum(sim.waiting) / max(total, 1e-30):.0%}",
    ]
    for op, t in sorted(sim.per_op.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {op}: {t:.4g}{unit} busy ({t / max(busy, 1e-30):.0%})")
    if sim.critical:
        ops: dict[str, int] = {}
        for i in sim.critical:
            ops[schedule.instrs[i].op] = ops.get(schedule.instrs[i].op, 0) + 1
        lines.append(
            f"  critical path: {len(sim.critical)} instructions ("
            + ", ".join(f"{op} x{k}" for op, k in ops.items())
            + ")"
        )
    if sim.violations:
        lines.append(f"  {len(sim.violations)} ordering violation(s): {sim.violations[0]}")
    return "\n".join(lines)
