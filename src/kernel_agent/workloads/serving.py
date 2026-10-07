"""Serving helpers for autoregressive and generative workloads (issue #149): host syncs off
the critical path, a post-processing stage on a side stream, continuous batching.

Model agnostic. A generation loop (LLM decode, a TTS patch loop, an STT decoder, a diffusion
sampler's per-request output stage) uses what it needs:

* :class:`AsyncFlags` (``Workload.async_flags()``): per-step device flags (stop tokens, a
  finished mask) read on the host without draining the GPU. ``send(flags)`` copies them to
  pinned memory with ``non_blocking=True`` and records an event; ``read(ticket)`` waits for
  that copy only. Send as early in the step as the flags exist, queue the rest of the step,
  then read: the host waits for the flags, not for the whole step, and queues the next step
  while the GPU still works (``depth`` tickets stay readable, so a loop may also read them a
  step later). The values are those of ``flags.cpu()``: a loop that reads them in the same
  step stays bit-identical.
* :class:`HostCopy`: a device tensor copied to the host the same way (``ready()`` polls,
  ``wait()`` waits for this copy only).
* :class:`SideStage`: a post-processing stage (vocoder, VAE, detokeniser) run on a side
  stream, so that request *r*'s stage overlaps request *r* + 1's generation, or chunk *i*'s
  overlaps step *i* + 1. ``submit(fn, ...)`` makes the side stream wait for the work queued
  so far on the caller's stream, runs ``fn`` there and copies its result to the host
  (:class:`HostCopy`); ``join()`` makes the caller's stream wait for the side stream and
  must run before the workload's ``run()`` returns (the evaluator times every stream: the
  run ends with a device-wide synchronize; work that outlives ``run()`` is hidden work).
  A plain torch stream until ``kernel_agent.concurrency`` (#147) declares named streams.
  Without CUDA, or ``pipelined=False``, ``fn`` runs inline.
* :func:`serve`: a request queue over the fixed batch slots of a :class:`SlotModel` (the
  workload's batched loop, split into ``admit`` / ``step`` / ``advance`` / ``finish``):
  static batches (a new batch when every slot's request has stopped) or **continuous
  batching** (a slot is refilled as soon as its request stops), the asynchronous stop check,
  the post stage of each finished request (pipelined on a :class:`SideStage` when asked) and
  each request's latency (``on_ready`` when its output reached the host; the workload marks
  it with :meth:`Workload.mark_ready <kernel_agent.workloads.base.Workload.mark_ready>`, so
  ``metric_detail`` reports per-request latency). :func:`per_request` regroups per-step
  batch records (the latents a teacher-forcing hook saw, say) per request.
* :func:`serving_options`: the opt-in, ``-o serving=static|continuous`` with
  ``-o requests=N`` (the queue) and ``-o pipeline=true`` (the post stage on a side stream).
  Without ``serving`` a workload runs its default fixed-length benchmark unchanged.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

#: ``-o serving=`` values.
SERVING_MODES = ("static", "continuous")


def _cuda(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "cuda"


class AsyncFlags:
    """Device flags read on the host without stalling the GPU (see the module docstring).

    ``send`` returns a ticket; ``read(ticket)`` returns the flags of that send as a CPU
    tensor (the last send when ``ticket`` is None). The last ``depth`` sends stay readable
    (one pinned buffer and event each, reused); an older ticket raises ``ValueError``.
    CPU tensors are cloned (nothing to overlap); other devices copy synchronously."""

    def __init__(self, depth: int = 2) -> None:
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.depth = depth
        self._host: list[torch.Tensor | None] = [None] * depth
        self._events: list[Any] = [None] * depth
        self._pending: list[bool] = [False] * depth
        self.sent = 0

    def send(self, flags: torch.Tensor) -> int:
        slot = self.sent % self.depth
        flags = flags.detach()
        if _cuda(flags):
            host = self._host[slot]
            if host is None or host.shape != flags.shape or host.dtype != flags.dtype:
                host = self._host[slot] = torch.empty(
                    flags.shape, dtype=flags.dtype, pin_memory=True
                )
            host.copy_(flags, non_blocking=True)  # after the copy `depth` sends ago (stream order)
            if self._events[slot] is None:
                self._events[slot] = torch.cuda.Event()
            self._events[slot].record(torch.cuda.current_stream(flags.device))
            self._pending[slot] = True
        else:
            self._host[slot] = flags.to("cpu", copy=True)
        self.sent += 1
        return self.sent - 1

    def read(self, ticket: int | None = None) -> torch.Tensor:
        ticket = self.sent - 1 if ticket is None else ticket
        if not 0 <= ticket < self.sent:
            raise ValueError(f"ticket {ticket} was never sent ({self.sent} sends)")
        if ticket < self.sent - self.depth:
            raise ValueError(
                f"ticket {ticket} was overwritten: only the last {self.depth} sends stay "
                "readable (raise `depth`)"
            )
        slot = ticket % self.depth
        if self._pending[slot]:
            self._events[slot].synchronize()
            self._pending[slot] = False
        host = self._host[slot]
        assert host is not None
        return host.clone()


class HostCopy:
    """``tensor`` copied to the host without blocking the caller: pinned memory,
    ``non_blocking=True`` and an event on the current stream (call it on the stream that
    produced ``tensor``). ``ready()`` polls the copy, ``wait()`` waits for it and returns
    the host tensor. The source stays referenced until then (the caching allocator cannot
    hand its memory to another stream early). CPU tensors: no copy, ready at once."""

    def __init__(self, tensor: torch.Tensor) -> None:
        tensor = tensor.detach()
        self._source: torch.Tensor | None = None
        self._event: Any = None
        if _cuda(tensor):
            self.host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
            self.host.copy_(tensor, non_blocking=True)
            self._event = torch.cuda.Event()
            self._event.record(torch.cuda.current_stream(tensor.device))
            self._source = tensor
        else:
            self.host = tensor.to("cpu")

    def ready(self) -> bool:
        if self._event is None or self._event.query():
            self._source = None
            return True
        return False

    def wait(self) -> torch.Tensor:
        if self._event is not None:
            self._event.synchronize()
        self._source = None
        return self.host


class SideStage:
    """A post-processing stage on a side stream (see the module docstring): ``submit``
    forks, ``join`` joins. ``pipelined=False``, or a device without CUDA, runs every job
    inline on the caller's stream (the same results, nothing overlapped)."""

    def __init__(self, device: torch.device | str | None = None, *, pipelined: bool = True) -> None:
        self.device = torch.device(device) if device is not None else None
        cuda = self.device is not None and self.device.type == "cuda"
        self.stream: Any = torch.cuda.Stream(self.device) if pipelined and cuda else None
        self._keep: list[Any] = []
        self.submitted = 0

    @property
    def pipelined(self) -> bool:
        return self.stream is not None

    def submit(self, fn: Callable[..., torch.Tensor], *args: Any, **kwargs: Any) -> HostCopy:
        """Run ``fn(*args, **kwargs)`` after the work queued so far on the caller's stream,
        on the side stream, and copy its result to the host (:class:`HostCopy`)."""
        self.submitted += 1
        if self.stream is None:
            return HostCopy(fn(*args, **kwargs))
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream):
            out = fn(*args, **kwargs)
            copy = HostCopy(out)
        # inputs made on the caller's stream and the outputs stay referenced until join():
        # after it the caller's stream is ordered behind every use on the side stream
        self._keep.append((args, kwargs, out))
        return copy

    def join(self) -> None:
        """The caller's stream waits for everything submitted (call before ``run()``
        returns)."""
        if self.stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
        self._keep.clear()


class SlotModel(Protocol):
    """A workload's batched generation loop over ``slots`` fixed batch rows, as
    :func:`serve` drives it. Every step computes every slot (static shapes: idle slots
    compute something that is never read); ``serve`` keeps the bookkeeping."""

    @property
    def slots(self) -> int: ...

    def admit(self, slots: list[int], requests: list[int]) -> None:
        """Prefill ``requests`` (indices into the queue) into ``slots`` (a batch): their
        state replaces whatever those slots held; their first step is the next ``step``."""

    def step(self, send: Callable[[torch.Tensor], int]) -> None:
        """One step of every slot: produce its next output (token, patch, ...). Call
        ``send(flags)`` once with the per-slot stop flags (``[slots]``, nonzero = stop) as
        early as they exist, then queue the rest of the step; ``serve`` reads them after
        ``step`` returns. A run without a stop check (``stop=False``) need not send."""

    def advance(self) -> None:
        """Feed every slot's last output back (the next step's state): called after the
        stop check when any request goes on; the slots refilled right after are then
        overwritten by ``admit``."""

    def finish(
        self, slots: list[int], requests: list[int], steps: int
    ) -> Callable[[], torch.Tensor] | None:
        """``requests`` (in ``slots``) stopped after ``steps`` steps each. Return the post
        stage's job, a function of no arguments returning one row per request (decoded
        audio, text, ...), which :func:`serve` runs on its :class:`SideStage`; or None
        (no post stage). The job may run on another stream later: read the slots' state
        now (or keep it alive) and do the work inside the job."""


@dataclass
class Served:
    """What :func:`serve` did: per request its ``steps``, ``slot`` and ``first`` (the
    iteration of its first step) and its post-stage ``outputs`` row (host; None without a
    post stage); per iteration the slot ``tables`` (``(request, step)`` per slot, None for
    an idle slot)."""

    steps: list[int]
    slots: list[int]
    first: list[int]
    tables: list[list[tuple[int, int] | None]] = field(default_factory=list)
    outputs: list[torch.Tensor | None] = field(default_factory=list)

    @property
    def iterations(self) -> int:
        return len(self.tables)


def serve(
    model: SlotModel,
    requests: int,
    *,
    continuous: bool,
    max_steps: int | Sequence[int],
    min_steps: int = 0,
    stop: bool = True,
    stage: SideStage | None = None,
    post_batch: int = 0,
    flags: AsyncFlags | None = None,
    on_ready: Callable[[int, int, torch.Tensor | None], None] | None = None,
) -> Served:
    """Generate ``requests`` requests on ``model``'s slots (see the module docstring).

    A request stops after its step *k* (0-based) when its stop flag is set and
    ``k >= min_steps`` (``stop=True``), or after ``max_steps`` steps (one number, or one
    per request). ``continuous``: a slot is refilled from the queue as soon as its request
    stops; else a new batch is admitted when every slot's request has stopped (static
    batching). Requests that stop together with the same length go to one ``finish`` call
    (at most ``post_batch`` per call; 0: no limit), whose job runs on ``stage`` (inline when
    None). ``on_ready(request, steps, output_row)`` is called once per request, in the order
    the host sees the outputs arrive: polled after every step, the rest after
    ``stage.join()`` at the end (a request without a post stage is ready when it stops)."""
    if requests < 1:
        raise ValueError(f"requests must be >= 1, got {requests}")
    limits = [int(max_steps)] * requests if isinstance(max_steps, int) else list(max_steps)
    if len(limits) != requests or min(limits) < 1:
        raise ValueError(f"max_steps: one limit >= 1 for each of {requests} requests")
    slots = int(model.slots)
    stage = stage or SideStage(pipelined=False)
    flags = flags or AsyncFlags()
    queue = deque(range(requests))
    table: list[tuple[int, int] | None] = [None] * slots
    served = Served([0] * requests, [-1] * requests, [-1] * requests)
    served.outputs = [None] * requests
    pending: list[tuple[list[int], int, HostCopy]] = []

    def admit(free: list[int], iteration: int) -> None:
        take = [queue.popleft() for _ in range(min(len(free), len(queue)))]
        if not take:
            return
        model.admit(free[: len(take)], take)
        for s, m in zip(free, take, strict=False):
            table[s] = (m, 0)
            served.slots[m], served.first[m] = s, iteration

    def ready(group: list[int], steps: int, rows: torch.Tensor | None) -> None:
        for j, m in enumerate(group):
            served.outputs[m] = None if rows is None else rows[j]
            if on_ready is not None:
                on_ready(m, steps, served.outputs[m])

    def finish(done: list[tuple[int, int]]) -> None:  # (slot, request)
        by_steps: dict[int, list[tuple[int, int]]] = {}
        for s, m in done:
            by_steps.setdefault(served.steps[m], []).append((s, m))
        for steps, members in by_steps.items():
            size = post_batch or len(members)
            for i in range(0, len(members), size):
                chunk = members[i : i + size]
                group = [m for _, m in chunk]
                job = model.finish([s for s, _ in chunk], group, steps)
                if job is None:
                    ready(group, steps, None)
                else:
                    pending.append((group, steps, stage.submit(job)))

    def poll() -> None:
        for entry in list(pending):
            if entry[2].ready():
                pending.remove(entry)
                ready(entry[0], entry[1], entry[2].wait())

    tickets: list[int] = []

    def send(step_flags: torch.Tensor) -> int:
        tickets.append(flags.send(step_flags))
        return tickets[-1]

    admit(list(range(slots)), 0)
    while any(e is not None for e in table):
        served.tables.append(list(table))
        tickets.clear()
        model.step(send)
        if stop and not tickets:
            raise RuntimeError("SlotModel.step() sent no stop flags (call send(flags))")
        hits = flags.read(tickets[-1]).reshape(-1).tolist() if stop else None
        done: list[tuple[int, int]] = []
        for s, entry in enumerate(table):
            if entry is None:
                continue
            m, k = entry
            if (hits is not None and k >= min_steps and hits[s]) or k + 1 >= limits[m]:
                served.steps[m] = k + 1
                table[s] = None
                done.append((s, m))
            else:
                table[s] = (m, k + 1)
        if done:
            finish(done)
        poll()
        active = any(e is not None for e in table)
        if not active and not queue:
            break
        if active:
            model.advance()
        free = [s for s, e in enumerate(table) if e is None]
        if queue and free and (continuous or not active):
            admit(free, len(served.tables))
    stage.join()
    for group, steps, copy in pending:
        ready(group, steps, copy.wait())
    return served


def per_request(
    records: Sequence[torch.Tensor],
    tables: Sequence[Sequence[tuple[int, int] | None]],
    requests: int,
) -> tuple[torch.Tensor, list[int]]:
    """Per-iteration batch records (``[slots, ...]``, one per iteration of :func:`serve`,
    e.g. what a hook saw each step) regrouped per request: ``([max steps, requests, ...]``
    (zero past a request's last step), the steps recorded per request)``."""
    if len(records) != len(tables):
        raise ValueError(
            f"{len(records)} records for {len(tables)} steps: record exactly once per step"
        )
    steps = [0] * requests
    if not records:
        return torch.empty(0), steps
    stacked = torch.stack(list(records))
    index: list[tuple[int, int, int, int]] = []  # (iteration, slot, step, request)
    for i, table in enumerate(tables):
        for s, entry in enumerate(table):
            if entry is not None:
                m, k = entry
                index.append((i, s, k, m))
                steps[m] = max(steps[m], k + 1)
    out = stacked.new_zeros((max(steps), requests, *stacked.shape[2:]))
    if index:
        cols = torch.tensor(index, dtype=torch.long).t().to(stacked.device)
        out[cols[2], cols[3]] = stacked[cols[0], cols[1]]
    return out, steps


@dataclass(frozen=True)
class ServingOptions:
    """``-o serving=static|continuous``: ``requests`` (None: the workload's choice) and
    ``pipeline`` (the post stage on a :class:`SideStage`)."""

    mode: str
    requests: int | None = None
    pipeline: bool = False

    @property
    def continuous(self) -> bool:
        return self.mode == "continuous"


def _flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def serving_options(options: Mapping[str, Any]) -> ServingOptions | None:
    """The serving opt-in of a workload's options (None: not asked for, the default
    benchmark; ``requests`` and ``pipeline`` alone mean nothing). ``ValueError`` for an
    unknown mode or ``requests < 1``."""
    mode = options.get("serving")
    if mode is None or (isinstance(mode, str) and mode.strip().lower() in ("", "off", "none")):
        return None
    mode = str(mode).strip().lower()
    if mode not in SERVING_MODES:
        raise ValueError(f"-o serving={mode}: choose one of {', '.join(SERVING_MODES)}")
    raw = options.get("requests")
    requests = None if raw in (None, "") else int(raw)
    if requests is not None and requests < 1:
        raise ValueError(f"-o requests must be >= 1, got {requests}")
    return ServingOptions(mode, requests, _flag(options.get("pipeline")))
