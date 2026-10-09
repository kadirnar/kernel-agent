"""Device state of one megakernel schedule (issue #225): the buffers a project's launch
function passes to ``ka_mk::launch`` (``include/ka_mk.cuh``), built once in ``build()`` /
``apply()`` and reused by every call.

* ``program``: the serialised schedule (int32, :meth:`.schedule.Schedule.words`);
* ``counters``: its counters and the interpreter's internal words, zero (the last block of
  every launch zeroes them again);
* ``table``: the tensor table, the device pointers of ``tensors`` (instructions name tensors
  by index), which the runtime keeps alive;
* ``status``: :data:`STATUS_WORDS` int32 of pinned host memory the watchdog writes to; a hang
  can be read without a CUDA call (:meth:`Runtime.failure`), also after the kernel stopped;
* ``trace`` (optional, ``KA_MK_TRACE`` builds): [instructions, 8] int64 timestamps
  (:data:`.simulate.TRACE_COLUMNS`).

A hang (a counter wait longer than the watchdog's limit: ``watchdog_ms``,
``$KA_MK_WATCHDOG_MS``, default :data:`DEFAULT_WATCHDOG_MS`) stops the kernel instead of the
process. :meth:`Runtime.check` raises :class:`MegakernelHang` (call it before every launch:
a hung schedule is not launched again) and :func:`hangs` lists the failures of every live
runtime, which the evaluator reads to record status ``hang`` with the instruction id.
"""

from __future__ import annotations

import os
import weakref
from collections.abc import Sequence
from typing import Any

from kernel_agent.native.megakernel.schedule import Schedule, ScheduleError

STATUS_WORDS = 16
S_CODE, S_INSTR, S_QUEUE, S_OPCODE, S_COUNTER, S_VALUE, S_TARGET, S_SM = range(8)
CODES = {0: "ok", 1: "hang", 2: "bad_program", 3: "bad_opcode", 4: "pool_too_small"}
INTERNAL = 2  # words after the counters: blocks done, abort flag
TRACE_COLS = 8
WATCHDOG_ENV = "KA_MK_WATCHDOG_MS"
#: A wait on one counter longer than this is a hang: legitimate waits are microseconds.
DEFAULT_WATCHDOG_MS = 2000.0

_LIVE: weakref.WeakSet[Runtime] = weakref.WeakSet()
_RAISED: list[dict[str, Any]] = []  # the infos of the MegakernelHang raised since forget()


class MegakernelHang(RuntimeError):
    """A megakernel stopped by its watchdog (or a bad program): ``info`` says where."""

    def __init__(self, info: dict[str, Any]) -> None:
        super().__init__(describe(info))
        self.info = info
        _RAISED.append(info)


def describe(info: dict[str, Any]) -> str:
    """One line on a failure (:meth:`Runtime.failure`)."""
    where = f"instruction {info.get('instr')}"
    if info.get("op") is not None:
        where += f" ({info['op']}[{info.get('tile')}])"
    if info.get("code") == "hang":
        return (
            f"megakernel hang: {where} on queue {info.get('queue')} waited longer than "
            f"{info.get('watchdog_ms')} ms on counter {info.get('counter')} "
            f"({info.get('value')} of {info.get('target')})"
        )
    if info.get("code") == "pool_too_small":
        return (
            f"megakernel stopped: {where} prefetches {info.get('value')} bytes, the page pool "
            f"holds {info.get('target')}"
        )
    return f"megakernel stopped ({info.get('code')}): {where}, opcode {info.get('opcode')}"


def watchdog_limit(value: float | None = None) -> float:
    """The watchdog's limit in ms: ``value``, else ``$KA_MK_WATCHDOG_MS``, else the default."""
    if value is not None:
        return float(value)
    try:
        return float(os.environ.get(WATCHDOG_ENV, "") or DEFAULT_WATCHDOG_MS)
    except ValueError:
        return DEFAULT_WATCHDOG_MS


class Runtime:
    """The buffers of ``schedule`` on ``device`` (default: the first tensor's).
    ``pool_bytes``: the page pool of the launch (pages x page bytes): a schedule whose
    largest prefetch does not fit is refused here, not on the GPU."""

    def __init__(
        self,
        schedule: Schedule,
        tensors: Sequence[Any],
        *,
        device: Any = None,
        trace: bool = False,
        watchdog_ms: float | None = None,
        pool_bytes: int | None = None,
    ) -> None:
        import torch

        if pool_bytes is not None and schedule.max_prefetch > pool_bytes:
            raise ScheduleError(
                f"an instruction prefetches {schedule.max_prefetch} bytes; the page pool holds "
                f"{pool_bytes}: use smaller tiles"
            )
        used = {i.prefetch.tensor for i in schedule.instrs if i.prefetch is not None}
        if used and max(used) >= len(tensors):
            raise ScheduleError(f"a prefetch names tensor {max(used)} of {len(tensors)}")
        for k in sorted(used):
            if tensors[k].data_ptr() % 16:
                raise ScheduleError(f"tensor {k} is not 16-byte aligned (bulk copies need it)")
        device = torch.device(device) if device is not None else tensors[0].device
        self.schedule = schedule
        self.tensors = list(tensors)  # the table holds their addresses
        self.program = schedule.tensor(device)
        self.counters = torch.zeros(
            len(schedule.counters) + INTERNAL, dtype=torch.int32, device=device
        )
        self.table = torch.tensor(
            [t.data_ptr() for t in self.tensors], dtype=torch.int64, device=device
        )
        # pinned host memory: with unified addressing its pointer is valid on the device
        self.status = torch.zeros(STATUS_WORDS, dtype=torch.int32, pin_memory=True)
        self.trace = (
            torch.zeros((len(schedule.instrs), TRACE_COLS), dtype=torch.int64, device=device)
            if trace
            else None
        )
        self.watchdog_ms = watchdog_limit(watchdog_ms)
        self.timeout_ns = int(self.watchdog_ms * 1e6)
        _LIVE.add(self)

    def args(self) -> tuple[Any, ...]:
        """``(program, counters, table, status pointer, trace or None, timeout_ns)``: the
        launch arguments of ``ka_mk::Params`` before ``n_pages``."""
        return (
            self.program,
            self.counters,
            self.table,
            self.status.data_ptr(),
            self.trace,
            self.timeout_ns,
        )

    def failure(self) -> dict[str, Any] | None:
        """What the watchdog recorded (pinned host memory: no CUDA call), or None."""
        words = self.status.tolist()
        code = int(words[S_CODE])
        if code == 0:
            return None
        info: dict[str, Any] = {
            "code": CODES.get(code, str(code)),
            "instr": int(words[S_INSTR]),
            "queue": int(words[S_QUEUE]),
            "opcode": int(words[S_OPCODE]),
            "counter": int(words[S_COUNTER]),
            "value": int(words[S_VALUE]),
            "target": int(words[S_TARGET]),
            "sm": int(words[S_SM]),
            "watchdog_ms": self.watchdog_ms,
        }
        if 0 <= info["instr"] < len(self.schedule.instrs):
            instr = self.schedule.instrs[info["instr"]]
            info.update(op=instr.op, tile=instr.tile)
        return info

    def check(self) -> None:
        """Raise :class:`MegakernelHang` when a launch so far stopped (call before each)."""
        if (info := self.failure()) is not None:
            raise MegakernelHang(info)

    def clear(self) -> None:
        """Forget a failure and zero the counters (a stopped launch zeroed them already;
        this is for a launch that never reached its end)."""
        self.status.zero_()
        self.counters.zero_()

    def trace_rows(self) -> list[list[int]]:
        """The trace of the last launch (synchronises), [] without one."""
        return [] if self.trace is None else self.trace.tolist()


def hangs() -> list[dict[str, Any]]:
    """The failures raised (:class:`MegakernelHang`) or recorded by the runtimes of this
    process since :func:`forget` (the evaluator's check after a failed candidate)."""
    return [*_RAISED, *(info for rt in list(_LIVE) if (info := rt.failure()) is not None)]


def forget() -> None:
    """Stop reporting the runtimes and hangs so far (the evaluator, before each candidate:
    the next one's failures are its own)."""
    _RAISED.clear()
    _LIVE.clear()
