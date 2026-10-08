"""GPU job queue: which job gets the GPU next (docs/MULTIAGENT.md §3.3, issue #182).

Every GPU job of a run passes :func:`kernel_agent.gpulock.gpu_lock`. A job is described by
a :class:`Job` (kind, priority class, expected seconds, the agent session waiting for it,
target, memory) in a context variable: the evaluation tools and the coordinator set it
(:func:`run`, :func:`tagged`) before ``asyncio.to_thread``, which copies the context into
the thread that takes the lock. So the queue reaches every lock site without signature
changes. An untagged call (a CLI command, a test) is a job of class ``default``.

**Admission** (:class:`Gate`, ``gpulock._admit``): a job takes a GPU when it is the head of
the queue and a GPU is free. The head is the job with the best class (:data:`CLASSES`,
from ``deadline`` to ``background``); a class improves one step per :data:`AGE_S` of
waiting (never to ``deadline``), so background work still progresses on a busy GPU. Within
a class the session served least recently goes first (round robin: one sweep-heavy agent
cannot monopolise the GPU), then the shortest job (:func:`estimate`), then the oldest.
Every acquisition queues on its own, so a long sequence of acquisitions (an integration,
one A/B step each) lets waiting jobs of a better class go between its steps; a step that
runs is never interrupted. A measurement found dirty (``hygiene.py``: a foreign GPU process
or CPU contention during its timing) queues again first of its class (:meth:`Job.requeue`).
The ``flock`` stays the outer, cross-process layer: another process still excludes, and the
queue orders this process's jobs. A coroutine of the event loop's thread waits its turn
without blocking the loop (:meth:`Gate.admit_async`, :func:`holding`): the simulated
executor of a virtual-time dry run (``dryrun.py``, whose clock replaces :data:`clock`) holds
the GPU that way.

**Exclusive and shared**: every timed job holds its GPU alone (``exclusive``, all of
today's jobs). A job that only checks correctness may be non-exclusive (agents' dev runs,
class ``dev``, #185): it shares a GPU with other non-exclusive jobs while their memory
estimates fit in it (:func:`fits`, so two jobs that load a whole model never share), and
never with an exclusive one.

**Time**: waiting is not the session's. A :class:`SessionClock` (``Orchestrator._agent``)
holds the session's ``asyncio.timeout`` while one of its jobs waits and then pushes it back
by the wait, and pushes back its ``Budget.deadlines`` entry the same way (``minutes_left``
and the ``stop`` advice), never past the run's agent time. A job withdrawn by its session
(cancelled: a timeout, Ctrl-C) leaves the queue without running. Ledger rows get
``queue_s``; their ``eval_s`` stays the evaluation's own time.

**Records**: ``gpu_queue.jsonl`` in the run directory has a ``gpu_job`` event per state of
a tagged job (``queued`` when it has to wait, ``start``, ``done``, ``withdrawn``; class,
kind, session, target, ``wait_s``, ``hold_s``); ``kernel-agent status`` summarises it
(:func:`status_lines`) and :func:`listen` hands every event to a callback (a completion
notification instead of polling).
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

from kernel_agent import interrupt
from kernel_agent.workspace import RunDir, append_jsonl, read_json, read_jsonl

DEADLINE = "deadline"  # the final integration's steps once the run is in its reserve window
INTERACTIVE = "interactive"  # quick checks, verify_rewrite: an agent waits for a short job
EVAL = "eval"  # full kernel evaluations: an agent waits
E2E = "e2e"  # end-to-end evaluations: an agent waits for a long job (a model load)
SWEEP = "sweep"  # sweep_candidate: long, one call times many configs
DEV = "dev"  # agents' own GPU runs (run_on_gpu, #185)
BACKGROUND = "background"  # integration, recheck, memcheck, seeding, captures, re-profiles
CLASSES = (DEADLINE, INTERACTIVE, EVAL, E2E, SWEEP, DEV, BACKGROUND)  # best first
DEFAULT = "default"  # untagged (CLI commands, tests): ranked as an evaluation
RANK = {c: i for i, c in enumerate(CLASSES)} | {DEFAULT: CLASSES.index(EVAL)}
AGE_S = 600.0  # a waiting job's class improves one step per this many seconds
MEM_SHARE = 0.9  # non-exclusive jobs share a GPU while their memory fits in this share of it
WAIT_S = 0.25  # a waiter re-checks its turn (aging) and whether the run is stopping this often
FILE = "gpu_queue.jsonl"  # in the run directory

#: kind -> (class, seconds a job of that kind takes before the run has timed one). The
#: seconds are medians of the runs studied for #174: kernel evaluation 8.7 s, sweep 33 s,
#: end-to-end evaluation 71 s, integration A/B step 84 s.
KINDS: dict[str, tuple[str, float]] = {
    "quick": (INTERACTIVE, 9.0),
    "verify_rewrite": (INTERACTIVE, 30.0),
    "eval": (EVAL, 9.0),
    "ncu": (EVAL, 60.0),
    "e2e": (E2E, 71.0),
    "harness": (E2E, 71.0),
    "sweep": (SWEEP, 33.0),
    "dev": (DEV, 60.0),
    "integration": (BACKGROUND, 84.0),
    "reevaluate": (BACKGROUND, 9.0),
    "recheck": (BACKGROUND, 60.0),
    "memcheck": (BACKGROUND, 60.0),
    "seed": (BACKGROUND, 9.0),
    "capture": (BACKGROUND, 60.0),
    "reprofile": (BACKGROUND, 300.0),
    DEFAULT: (DEFAULT, 60.0),
}
#: kinds that load the whole model: their memory is the model's measured peak
MODEL_KINDS = frozenset({"e2e", "harness", "integration", "capture", "reprofile"})

#: The clock of waits, holds and aging (seconds); a virtual-time dry run (``dryrun.py``,
#: ``--dry-run --agents N``) replaces it with its simulated one
clock: Callable[[], float] = time.monotonic

_current: ContextVar[Job | None] = ContextVar("kernel_agent_gpu_job", default=None)
_session: ContextVar[SessionClock | None] = ContextVar("kernel_agent_session", default=None)
_ids = itertools.count(1)
_account = threading.Lock()
_attach = threading.Lock()  # a job's waiting and its session clock (Job.attach)
_listeners: list[Callable[[dict[str, Any]], None]] = []


class Withdrawn(Exception):
    """A queued job whose caller is gone (its session was cancelled): it never ran."""


# ------------------------------------------------------------------ jobs


@dataclass(eq=False)
class Job:
    """One GPU job: what it is (``kind``, ``job_class``), how long it should take
    (``estimate_s``), who waits for it (``session``, ``clock``) and, accumulated over its
    acquisitions (and those of the jobs tagged inside it), how long it waited for and held
    the GPU."""

    kind: str = DEFAULT
    job_class: str = DEFAULT
    estimate_s: float = KINDS[DEFAULT][1]
    session: str | None = None
    target: str | None = None
    run: RunDir | None = None
    mem_gb: float | None = None
    exclusive: bool = True
    parent: Job | None = None
    clock: SessionClock | None = None
    id: int = field(default_factory=lambda: next(_ids))
    seq: int = 0  # order of arrival of its current wait
    submitted: float = 0.0  # clock() its current wait started
    started: float | None = None  # ... and its current hold
    wait_s: float = 0.0
    hold_s: float = 0.0
    holds: int = 0
    withdrawn: bool = False
    front: bool = False  # its next wait: first of its class (a dirty measurement's re-run)
    # an evaluation its session submitted and does not wait for (submit_evaluation, #191):
    # its waits stop the session's clock only once the session waits for it (attach)
    detached: bool = False
    waiting: bool = False  # it waits for the GPU now (under _attach)

    @classmethod
    def of(
        cls,
        run: RunDir | None,
        kind: str,
        target: str | None = None,
        *,
        job_class: str | None = None,
        exclusive: bool = True,
        mem_gb: float | None = None,
        detached: bool = False,
    ) -> Job:
        """A job of ``kind`` of this context: its session (the agent session whose tool
        call this is) and the job it is tagged inside of, if any (:func:`tagged`). A
        ``detached`` job is its session's, but its waits are not (:meth:`attach`)."""
        outer, clock = current(), _session.get()
        clock = clock or (outer.clock if outer else None)
        detached = detached or (outer is not None and outer.detached)
        return cls(
            kind=kind,
            job_class=job_class or KINDS.get(kind, KINDS[DEFAULT])[0],
            estimate_s=estimate(run, kind, target),
            session=clock.label if clock else (outer.session if outer else None),
            target=target,
            run=run,
            mem_gb=mem_gb if mem_gb is not None else memory_gb(run, kind),
            exclusive=exclusive,
            parent=outer,
            clock=None if detached else clock,
            detached=detached,
        )

    def requeue(self) -> None:
        """Its next acquisition goes first of its class: the re-run of a measurement whose
        timing was dirty (``hygiene.py``)."""
        self.front = True

    def withdraw(self) -> None:
        """Its caller is gone: if it still waits, it leaves the queue (:class:`Withdrawn`)."""
        self.withdrawn = True

    def attach(self, clock: SessionClock | None) -> None:
        """Its session waits for it from now on (``evaluation_result``): its waits for the GPU
        stop ``clock`` (the one it waits now too)."""
        if clock is None:
            return
        with _attach:
            if self.clock is not None:
                return
            self.clock = clock
            if self.waiting:
                clock.pause()

    def _blocked(self) -> None:
        """It starts waiting for the GPU (``Wait.block``)."""
        with _attach:
            self.waiting = True
            if self.clock is not None:
                self.clock.pause()

    def _unblocked(self) -> None:
        """It stops waiting (``Wait``)."""
        with _attach:
            if self.waiting and self.clock is not None:
                self.clock.resume()
            self.waiting = False

    def add(self, *, wait: float = 0.0, hold: float = 0.0, holds: int = 0) -> None:
        """Account a wait / hold to this job and the jobs it is tagged inside of."""
        with _account:
            job: Job | None = self
            while job is not None:
                job.wait_s += wait
                job.hold_s += hold
                job.holds += holds
                job = job.parent

    @property
    def queue_s(self) -> float | None:
        """Its time waiting for the GPU (a record's ``queue_s``; None: it never took it)."""
        return round(self.wait_s, 1) if self.holds or self.wait_s else None


def current() -> Job | None:
    """The job the GPU locks of this context belong to (None: untagged)."""
    return _current.get()


def session() -> SessionClock | None:
    """The clock of the agent session this context runs in (None: not in one)."""
    return _session.get()


@contextlib.contextmanager
def tagged(
    kind: str,
    run: RunDir | None = None,
    target: str | None = None,
    *,
    job_class: str | None = None,
    exclusive: bool = True,
    mem_gb: float | None = None,
) -> Iterator[Job]:
    """GPU locks taken in this context, and in threads ``asyncio.to_thread`` starts from it,
    are a job of ``kind``. Tagged inside another job, it takes that job's run, target and
    class unless given, and its waits and holds count for that job too."""
    outer = current()
    if outer is not None:
        run, target = run or outer.run, target or outer.target
        job_class = job_class or outer.job_class
    job = Job.of(run, kind, target, job_class=job_class, exclusive=exclusive, mem_gb=mem_gb)
    with using(job):
        yield job


@contextlib.contextmanager
def using(job: Job) -> Iterator[Job]:
    """GPU locks taken in this context (and in threads ``asyncio.to_thread`` starts from it)
    are ``job``'s."""
    token = _current.set(job)
    try:
        yield job
    finally:
        _current.reset(token)


async def run[T](job: Job, fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """``fn(*args, **kwargs)`` in a worker thread (``asyncio.to_thread``) as ``job``. A caller
    cancelled while it waits (its session timed out or the run stops) withdraws ``job``:
    still queued, it leaves the queue without running."""
    with using(job):
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        except asyncio.CancelledError:
            job.withdraw()
            raise


# ------------------------------------------------------------------ estimates


def _rows_of(kind: str, target: str | None) -> Callable[[dict[str, Any]], bool] | None:
    """Which ledger rows time a job of ``kind`` on ``target`` (None: no row does)."""
    from kernel_agent import ledger

    quick = (ledger.QUICK_OK, ledger.QUICK_FAIL)

    def sweep(r: dict[str, Any]) -> bool:
        return "[sweep:" in str(r.get("hypothesis") or "")

    def integration(r: dict[str, Any]) -> bool:
        return r["target"] == ledger.E2E and r.get("backend") == "integrate"

    selects: dict[str, Callable[[dict[str, Any]], bool]] = {
        "quick": lambda r: r["target"] == target and r["status"] in quick,
        "eval": lambda r: (
            r["target"] == target and r["status"] not in ledger.UNMEASURED and not sweep(r)
        ),
        "reevaluate": lambda r: r["target"] == target and r["status"] == ledger.REEVALUATED,
        "sweep": lambda r: r["target"] == target and sweep(r),
        "e2e": lambda r: r["target"] == ledger.E2E and not integration(r),
        "harness": lambda r: r["target"] == ledger.E2E and not integration(r),
        "integration": integration,
    }
    return selects.get(kind)


def estimate(run: RunDir | None, kind: str, target: str | None = None) -> float:
    """Seconds a job of ``kind`` (on ``target``) is expected to hold the GPU: the median
    ``eval_s`` of the run's ledger rows of that kind (``scheduler.median_eval_s``), else the
    default of :data:`KINDS`."""
    fallback = KINDS.get(kind, KINDS[DEFAULT])[1]
    select = _rows_of(kind, target)
    if run is None or select is None or not run.ledger.exists():
        return fallback
    from kernel_agent import ledger, scheduler

    try:
        rows = ledger.read_tsv(run.ledger)
    except (OSError, ValueError):
        return fallback
    return scheduler.median_eval_s(r for r in rows if select(r)) or fallback


def memory_gb(run: RunDir | None, kind: str) -> float | None:
    """GPU memory (GB) a job of ``kind`` is expected to need: for the jobs that load the whole
    model (:data:`MODEL_KINDS`) the model's measured peak (``baseline.json``); None: unknown."""
    if run is None or kind not in MODEL_KINDS:
        return None
    try:
        peak = (read_json(run.baseline_json, {}) or {}).get("peak_mem_gb")
    except (OSError, ValueError):
        return None
    return float(peak) if isinstance(peak, int | float) and peak > 0 else None


def fits(group: list[Job], job: Job, capacity_gb: float | None) -> bool:
    """Whether non-exclusive ``job`` may join ``group``, the non-exclusive jobs on a GPU of
    ``capacity_gb``: every memory estimate is known and together they fit in
    :data:`MEM_SHARE` of it."""
    if capacity_gb is None or job.mem_gb is None or any(j.mem_gb is None for j in group):
        return False
    return sum(float(j.mem_gb or 0.0) for j in group) + job.mem_gb <= MEM_SHARE * capacity_gb


def rank(job: Job, now: float) -> int:
    """``job``'s class rank after aging (0 is the best): one step better per :data:`AGE_S`
    it has waited, never better than ``interactive``."""
    base = RANK.get(job.job_class, RANK[DEFAULT])
    if base <= RANK[INTERACTIVE]:
        return base
    return max(RANK[INTERACTIVE], base - int((now - job.submitted) // AGE_S))


# ------------------------------------------------------------------ admission


class Gate:
    """The admission controller of one lock (``gpulock``): the waiting jobs and the jobs on
    each GPU of this process. :meth:`admit` blocks until a job may take a GPU and returns
    the GPUs it may take; :meth:`placed` and :meth:`leave` record the GPU it took and its
    release (which wakes the waiters)."""

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.waiting: list[Job] = []
        self.holders: dict[int, list[Job]] = {}  # GPU index -> jobs on it
        self.unplaced: list[Job] = []  # exclusive jobs admitted, not yet on a GPU
        self.served: dict[str | None, int] = {}  # session -> turn it was last admitted in
        self.gpus = 1  # GPUs of the pool at the last admission
        self.wakers: list[Callable[[], None]] = []  # the coroutines that wait (admit_async)
        self._seq = itertools.count()
        self._turn = itertools.count()

    def _key(self, job: Job, now: float) -> tuple[int, bool, int, float, int]:
        served = self.served.get(job.session, -1)
        return (rank(job, now), not job.front, served, job.estimate_s, job.seq)

    def head(self, now: float | None = None) -> Job | None:
        """The job that goes next (None: nothing waits)."""
        now = clock() if now is None else now
        return min(self.waiting, key=lambda j: self._key(j, now), default=None)

    def _free(self, gpus: tuple[int, ...]) -> list[int]:
        return [g for g in gpus if not self.holders.get(g)]

    def _room(
        self, job: Job, gpus: tuple[int, ...], capacity: dict[int, float | None]
    ) -> tuple[int, ...] | None:
        """The GPUs ``job`` may take now (None: none). An exclusive job: those without a job
        of this process (``gpulock._acquire`` takes the first one no other process holds);
        a non-exclusive one: a GPU of non-exclusive jobs it fits with, else a free GPU."""
        free = self._free(gpus)
        if job.exclusive:
            return tuple(free) if len(free) > len(self.unplaced) else None
        for g in gpus:
            group = self.holders.get(g) or []
            if group and not any(j.exclusive for j in group) and fits(group, job, capacity.get(g)):
                return (g,)
        return (free[0],) if free and not self.unplaced else None

    def admit(
        self,
        job: Job,
        wait: Wait,
        gpus: tuple[int, ...],
        capacity: dict[int, float | None] | None = None,
    ) -> tuple[int, ...]:
        """Wait until ``job`` is the head and a GPU it may take is free; returns those GPUs
        (``gpus``: the pool's indices; ``capacity``: their memory in GB, for non-exclusive
        jobs). Raises ``interrupt.Interrupted`` when the run is stopping and
        :class:`Withdrawn` when the job was withdrawn while it waited."""
        room: tuple[int, ...] | None = None
        with self.cond:
            self.gpus = len(gpus)
            job.submitted, job.seq = clock(), next(self._seq)
            self.waiting.append(job)
            try:
                while True:
                    if job.withdrawn:
                        raise Withdrawn(f"GPU job {job.id} ({job.kind}) withdrawn while queued")
                    if self.head() is job:
                        room = self._room(job, gpus, capacity or {})
                        if room is not None:
                            break
                    wait.block()
                    self.cond.wait(WAIT_S)
                    interrupt.check()
            finally:
                self.waiting.remove(job)
                self._notify()  # the next head checks its turn
            assert room is not None
            return self._admitted(job, room)

    async def admit_async(
        self,
        job: Job,
        wait: Wait,
        gpus: tuple[int, ...] = (0,),
        capacity: dict[int, float | None] | None = None,
    ) -> tuple[int, ...]:
        """:meth:`admit` for a job of the event loop's own thread: it waits as a coroutine
        (woken by every release), so a simulated executor (``dryrun.py``: the job's body is a
        sleep in simulated time) goes through the same queue as the threads' jobs."""
        loop = asyncio.get_running_loop()
        turn = asyncio.Event()

        def wake() -> None:
            with contextlib.suppress(RuntimeError):  # the loop closed meanwhile
                loop.call_soon_threadsafe(turn.set)

        with self.cond:
            self.gpus = len(gpus)
            job.submitted, job.seq = clock(), next(self._seq)
            self.waiting.append(job)
            self.wakers.append(wake)
        try:
            while True:
                turn.clear()
                with self.cond:
                    if job.withdrawn:
                        raise Withdrawn(f"GPU job {job.id} ({job.kind}) withdrawn while queued")
                    room = self._room(job, gpus, capacity or {}) if self.head() is job else None
                    if room is not None:
                        self._unwait(job, wake)
                        return self._admitted(job, room)
                wait.block()
                await turn.wait()
                interrupt.check()
        except BaseException:
            with self.cond:
                self._unwait(job, wake)
            raise

    def _unwait(self, job: Job, wake: Callable[[], None]) -> None:
        """``job`` (a coroutine's, :meth:`admit_async`) waits no more (under ``cond``)."""
        if job in self.waiting:
            self.waiting.remove(job)
        if wake in self.wakers:
            self.wakers.remove(wake)
        self._notify()

    def _admitted(self, job: Job, room: tuple[int, ...]) -> tuple[int, ...]:
        self.served[job.session] = next(self._turn)
        job.front = False
        if job.exclusive:
            self.unplaced.append(job)
        else:
            self.holders.setdefault(room[0], []).append(job)
        return room

    def _notify(self) -> None:
        """Wake every waiter (under ``cond``): threads and coroutines check their turn."""
        self.cond.notify_all()
        for wake in list(self.wakers):
            wake()

    def placed(self, job: Job, gpu: int) -> None:
        """An admitted exclusive job took GPU ``gpu``."""
        with self.cond:
            self.unplaced.remove(job)
            self.holders.setdefault(gpu, []).append(job)

    def leave(self, job: Job) -> None:
        """An admitted job let its GPU go (or never took one): the waiters check their turn."""
        with self.cond:
            if job in self.unplaced:
                self.unplaced.remove(job)
            for group in self.holders.values():
                if job in group:
                    group.remove(job)
            self._notify()

    def withdraw(self, job: Job) -> bool:
        """Withdraw ``job`` if it has not taken a GPU yet (it waits, or is not queued yet):
        it leaves the queue without running (:class:`Withdrawn`, the critic's reject,
        ``critic.py``). False: it took one already (or held one and let it go)."""
        with self.cond:
            admitted = job in self.unplaced or any(job in g for g in self.holders.values())
            if job.holds or admitted:
                return False
            job.withdraw()
            self._notify()
        return True

    def expected_wait(self, job_class: str) -> float:
        """Seconds a new job of ``job_class`` should wait: what the jobs on the GPUs have
        left and the jobs that would go before it, spread over the pool's GPUs."""
        with self.cond:
            now = clock()
            busy = [j for group in self.holders.values() for j in group] + self.unplaced
            left = sum(max(j.estimate_s - (now - (j.started or now)), 0.0) for j in busy)
            ahead = RANK.get(job_class, RANK[DEFAULT])
            left += sum(j.estimate_s for j in self.waiting if rank(j, now) <= ahead)
            return left / max(self.gpus, 1)


_gates: dict[str, Gate] = {}
_gates_lock = threading.Lock()


def gate(name: str = "gpu") -> Gate:
    """The admission controller of lock ``name`` (one per process)."""
    with _gates_lock:
        return _gates.setdefault(name, Gate())


@contextlib.asynccontextmanager
async def holding(job: Job, name: str = "gpu", gpus: tuple[int, ...] = (0,)) -> AsyncIterator[int]:
    """Hold a GPU of lock ``name``'s queue as ``job``, from a coroutine of the event loop's
    thread (:meth:`Gate.admit_async`); yields its index. The queue's order, accounting and
    ``gpu_job`` events are ``gpulock.gpu_lock``'s; there is no ``flock`` and no process: the
    simulated executor of a virtual-time dry run (``dryrun.py``) holds it while it sleeps for
    the job's simulated seconds."""
    wait, queue = Wait(job), gate(name)
    try:
        room = await queue.admit_async(job, wait, gpus)
    except BaseException as exc:
        wait.abandoned(exc)
        raise
    index = room[0]
    if job.exclusive:
        queue.placed(job, index)
    wait.started(index)
    try:
        yield index
    finally:
        queue.leave(job)
        wait.ended()


def expected_wait(job_class: str = EVAL, name: str = "gpu") -> float:
    """Seconds a job of ``job_class`` would wait for the GPU now (0: it would go at once)."""
    return gate(name).expected_wait(job_class)


# ------------------------------------------------------------------ accounting


class Wait:
    """One acquisition of the GPU by ``job`` (``gpulock.gpu_lock``): its wait pauses the
    session's clock (:meth:`block`, only when it has to wait), its wait and hold are
    accounted to the job, and each step is a ``gpu_job`` event."""

    def __init__(self, job: Job) -> None:
        self.job = job
        self.t0 = clock()
        self.blocked = False
        self.waited = 0.0

    def block(self) -> None:
        """The job has to wait (the GPU is busy, or other jobs go first)."""
        if self.blocked:
            return
        self.blocked = True
        self.job._blocked()
        _event(self.job, "queued", estimate_s=round(self.job.estimate_s, 1))

    def _end_wait(self) -> None:
        self.waited = clock() - self.t0
        self.job.add(wait=self.waited)
        if self.blocked:
            self.job._unblocked()

    def started(self, gpu: int) -> None:
        """It holds GPU ``gpu`` now."""
        self._end_wait()
        self.job.started = clock()
        self.job.add(holds=1)
        mem = {"mem_gb": self.job.mem_gb} if self.job.mem_gb is not None else {}
        _event(self.job, "start", gpu=gpu, wait_s=round(self.waited, 2), **mem)

    def ended(self) -> None:
        """It let the GPU go."""
        held = clock() - (self.job.started or self.t0)
        self.job.add(hold=held)
        self.job.started = None
        _event(self.job, "done", hold_s=round(held, 2), wait_s=round(self.waited, 2))

    def abandoned(self, exc: BaseException) -> None:
        """It never got the GPU: withdrawn, the run stopping, or an error."""
        self._end_wait()
        _event(self.job, "withdrawn", wait_s=round(self.waited, 2), reason=type(exc).__name__)


def listen(callback: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
    """Call ``callback`` with every ``gpu_job`` event (in the thread that has it); returns
    the function that stops it."""
    _listeners.append(callback)

    def stop() -> None:
        with contextlib.suppress(ValueError):
            _listeners.remove(callback)

    return stop


def _event(job: Job, state: str, **data: Any) -> None:
    from kernel_agent import ledger

    now = ledger.clock()  # the run's clock (a dry run's simulated one)
    record = {
        "ts": round(now, 3),
        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
        "event": "gpu_job",
        "state": state,
        "id": job.id,
        "kind": job.kind,
        "class": job.job_class,
        **({"session": job.session} if job.session else {}),
        **({"target": job.target} if job.target else {}),
        **({"shared": True} if not job.exclusive else {}),
        **({"detached": True} if job.detached else {}),
        **data,
    }
    if job.run is not None:
        with contextlib.suppress(OSError):  # a record: it never fails the job
            append_jsonl(job.run.root / FILE, record)
    for callback in list(_listeners):
        with contextlib.suppress(Exception):
            callback(record)


# ------------------------------------------------------------------ session clocks


class SessionClock:
    """The time limits of one agent session (``label``) that its jobs' GPU waits do not use.

    While one of its jobs waits (:meth:`pause`, from any thread), the session's
    ``asyncio.timeout`` stays at the run's agent deadline (``cap``: seconds the run still
    allows agents; None: no limit), so it cannot run out during the wait; when no job waits
    any more (:meth:`resume`), it is pushed back by the wait from where it was, never past
    that deadline (nor earlier than it was), and ``extend`` gets the seconds waited
    (``Budget.extend_deadline``)."""

    def __init__(
        self,
        label: str,
        *,
        cap: Callable[[], float | None] | None = None,
        extend: Callable[[float], None] | None = None,
    ) -> None:
        self.label = label
        self.cap = cap
        self.extend = extend
        self.waited = 0.0  # seconds its jobs waited for the GPU (overlapping waits once)
        self.loop: asyncio.AbstractEventLoop | None = None
        self.timer: asyncio.Timeout | None = None
        self._lock = threading.Lock()
        self._waiting = 0
        self._since = 0.0
        self._when: float | None = None  # the timer's deadline when the wait started
        self._next: asyncio.Timeout | None = None
        self._token: Token[SessionClock | None] | None = None

    def running(self, timer: asyncio.Timeout | None = None) -> SessionClock:
        """``with`` / ``async with`` it: the session runs in this context (its timeout:
        ``timer``, entered first), so the jobs of its tool calls are its jobs (:meth:`Job.of`)."""
        self._next = timer
        return self

    def __enter__(self) -> SessionClock:
        self.loop, self.timer, self._when = asyncio.get_running_loop(), self._next, None
        self._token = _session.set(self)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._token is not None:
            _session.reset(self._token)
            self._token = None
        self.timer = None

    async def __aenter__(self) -> SessionClock:
        return self.__enter__()

    async def __aexit__(self, *exc: object) -> None:
        self.__exit__(*exc)

    def pause(self) -> None:
        """One of its jobs starts waiting for the GPU."""
        with self._lock:
            self._waiting += 1
            if self._waiting > 1:
                return
            self._since = clock()
        self._call(self._hold)

    def resume(self) -> None:
        """One of its jobs stops waiting."""
        with self._lock:
            self._waiting -= 1
            if self._waiting > 0:
                return
            waited = clock() - self._since
            self.waited += waited
        self._call(self._release, waited)

    def _call(self, fn: Callable[..., None], *args: Any) -> None:
        loop = self.loop
        if loop is None or loop.is_closed():
            fn(*args)
            return
        with contextlib.suppress(RuntimeError):  # the loop closed meanwhile
            loop.call_soon_threadsafe(fn, *args)

    def _deadline(self) -> float | None:
        """The run's agent deadline on the event loop's clock (None: no limit)."""
        left = self.cap() if self.cap is not None else None
        if left is None or self.loop is None:
            return None
        return self.loop.time() + max(left, 0.0)

    def _reschedule(self, when: float | None) -> None:
        timer = self.timer
        if timer is None or timer.expired():
            return
        with contextlib.suppress(RuntimeError):  # the session's timeout ended meanwhile
            timer.reschedule(when)

    def _hold(self) -> None:
        timer = self.timer
        if timer is None or (when := timer.when()) is None or self._when is not None:
            return
        self._when = when
        cap = self._deadline()
        self._reschedule(None if cap is None else max(when, cap))

    def _release(self, waited: float) -> None:
        if (when := self._when) is not None:
            self._when = None
            new = when + waited
            if (cap := self._deadline()) is not None:
                new = max(when, min(new, cap))
            self._reschedule(new)
        if self.extend is not None:
            self.extend(waited)


# ------------------------------------------------------------------ status


def _span(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    return f"{seconds / 60:.1f} min" if seconds < 5400 else f"{seconds / 3600:.1f} h"


def _who(e: dict[str, Any]) -> str:
    kind, cls = e.get("kind"), e.get("class")
    what = str(kind) if kind == cls else " ".join(str(x) for x in (cls, kind) if x)
    where = ", ".join(str(e[k]) for k in ("target", "session") if e.get(k))
    return f"{what} ({where})" if where else what


def status_lines(run: RunDir, width: int = 120) -> list[str]:
    """The GPU block of ``kernel-agent status``: GPU acquisitions, time on the GPU and
    waiting for it (per class), what is on the GPU and what waits ([] before the run's
    first tagged job)."""
    try:
        log = read_jsonl(run.root / FILE)
    except (OSError, ValueError):
        return []
    if not log:
        return []
    last: dict[Any, dict[str, Any]] = {}  # job id -> its fields, with its latest state
    for e in log:
        last[e.get("id")] = {**last.get(e.get("id"), {}), **e}
    starts = [e for e in log if e.get("state") == "start"]
    waits = [float(e.get("wait_s") or 0.0) for e in starts]
    hold = sum(float(e.get("hold_s") or 0.0) for e in log if e.get("state") == "done")
    head = f"GPU queue: {len(starts)} jobs, {_span(hold)} on the GPU, {_span(sum(waits))} waiting"
    lines = [(head + (f" (max {_span(max(waits))})" if any(waits) else ""))[:width]]
    by_class = []
    for c in (*CLASSES, DEFAULT):
        jobs = [e for e in starts if e.get("class") == c]
        if jobs:
            waited = sum(float(e.get("wait_s") or 0.0) for e in jobs)
            by_class.append(f"{c} {len(jobs)}" + (f" (waited {_span(waited)})" if waited else ""))
    if by_class:
        lines.append(("  " + ", ".join(by_class))[:width])
    now = time.time()
    for state, title in (("start", "on the GPU"), ("queued", "waiting")):
        jobs = [e for e in last.values() if e.get("state") == state]
        if jobs:
            what = "; ".join(f"{_who(e)} {_span(now - float(e['ts']))}" for e in jobs)
            lines.append(f"  {title} ({len(jobs)}): {what}"[:width])
    return lines
