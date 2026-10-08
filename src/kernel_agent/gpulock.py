"""Exclusive GPU locks so concurrent agents never benchmark on the same GPU at once.

Every GPU of the pool (:func:`pool`) has a lock: an ``flock`` on a file in the cache
dir (across processes) plus a ``threading.Lock`` (across the threads of this process:
parallel agents run their evaluation tools in worker threads of one orchestrator).
:func:`gpu_lock` takes the first free GPU without waiting, or else waits for the GPU
with the fewest waiters in this process, and yields its index. It is re-entrant
within a thread and keeps that GPU. A child process started while the lock is held
gets ``KERNEL_AGENT_LOCK_HELD=1`` through :func:`child_env`, so it does not wait for
its own parent, and, when more than one GPU is visible, ``CUDA_VISIBLE_DEVICES`` of
the locked GPU, so it runs where its parent locked. It also dies with its parent
(``interrupt.die_with_parent``). A run that is stopping (Ctrl-C, :mod:`kernel_agent.interrupt`)
takes no lock and starts no child: a waiter stops waiting, and work that held the lock
when the stop came raises ``Interrupted`` as it lets it go (it was cut short: not a result).

The pool is what ``nvidia-smi`` lists (the parent never initialises CUDA to find
it), restricted to an inherited ``CUDA_VISIBLE_DEVICES`` (indices or ``GPU-`` UUIDs);
``KERNEL_AGENT_GPUS=0,2`` replaces both. Indices are ``nvidia-smi``'s, i.e. PCI bus
order (CUDA's own order too with ``CUDA_DEVICE_ORDER=PCI_BUS_ID``, which pinned
children get, or on identical GPUs). GPU 0 locks ``gpu.lock``, the one lock file of
kernel-agent before the pool, so old and new processes on GPU 0 (on a single-GPU
machine: always) still exclude each other; GPU ``i`` locks ``gpu{i}.lock``. Without
``nvidia-smi`` (or with no GPU visible) the pool is GPU 0 alone and a child's
environment only gains ``KERNEL_AGENT_LOCK_HELD``, as before the pool. A waiter stays
with the GPU it chose, even if another one frees up first.

The threads of this process take the lock in the order of the GPU job queue
(:mod:`kernel_agent.gpuqueue`): :func:`_admit` lets a thread's job (the context's
``gpuqueue.Job``) go when it is the queue's head and a GPU is free, then it takes the
locks as above; a release wakes the queue. A re-entrant hold and a child process skip the
queue as they skip the lock. A non-exclusive job (correctness only) shares its GPU with
other non-exclusive jobs whose memory fits (``LOCK_SH``, no thread lock), never with an
exclusive one, and another process's exclusive lock still excludes it.
"""

from __future__ import annotations

import collections
import contextlib
import fcntl
import functools
import itertools
import os
import subprocess
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from kernel_agent import gpuqueue, interrupt
from kernel_agent.toolchain import CACHE_DIR

ENV = "KERNEL_AGENT_LOCK_HELD"
GPUS_ENV = "KERNEL_AGENT_GPUS"  # the pool as nvidia-smi indices, e.g. "0,2"
INDEX_ENV = "KERNEL_AGENT_GPU"  # a pinned child: the GPU its parent locked
WAIT_S = 0.25  # a waiter checks this often whether the run is stopping

_guard = threading.Lock()
_thread_locks: dict[str, threading.Lock] = {}
_waiting: collections.Counter[str] = collections.Counter()  # threads waiting per lock file
_turn = itertools.count()
_held = threading.local()


@dataclass(frozen=True)
class GPU:
    index: int  # nvidia-smi index (PCI bus order)
    name: str = ""
    uuid: str = ""


@dataclass(frozen=True)
class Pool:
    gpus: tuple[GPU, ...]
    pin: bool  # children see only the locked GPU (they would see more than one)


def _nvidia_smi() -> list[GPU]:
    """Every GPU ``nvidia-smi`` lists ([] without it)."""
    cmd = ["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in proc.stdout.splitlines() if proc.returncode == 0 else []:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            gpus.append(GPU(int(parts[0]), ", ".join(parts[1:-1]), parts[-1]))
    return gpus


def _visible(gpus: list[GPU], devices: str) -> list[GPU]:
    """The GPUs a ``CUDA_VISIBLE_DEVICES`` value names, in its order. Like CUDA, the
    list ends at the first entry that names none of them (``-1``, a MIG instance, ...)."""
    out: list[GPU] = []
    for entry in (e.strip() for e in devices.split(",")):
        gpu = next(
            (
                g
                for g in gpus
                if (entry.isdigit() and int(entry) == g.index)
                or (entry.startswith("GPU-") and g.uuid.startswith(entry))
            ),
            None,
        )
        if gpu is None or gpu in out:
            break
        out.append(gpu)
    return out


@functools.cache
def pool() -> Pool:
    """The GPUs :func:`gpu_lock` hands out, found once per process."""
    override = os.environ.get(GPUS_ENV, "").strip()
    if override:
        parts = [p.strip() for p in override.split(",") if p.strip()]
        if not parts or not all(p.isdigit() for p in parts):
            raise ValueError(f"{GPUS_ENV}={override!r}: expected GPU indices, e.g. 0,2")
        return Pool(tuple(GPU(i) for i in dict.fromkeys(map(int, parts))), pin=True)
    gpus = _nvidia_smi()
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = gpus if devices is None else _visible(gpus, devices)
    if not visible:
        return Pool((GPU(0),), pin=False)
    return Pool(tuple(visible), pin=len(visible) > 1)


def describe() -> str:
    """One line for ``kernel-agent doctor``: the pool and its lock files."""
    gpus = pool().gpus
    line = "GPU locks: " + ", ".join(
        f"{g.index}{' ' + g.name if g.name else ''} ({_lock_file('gpu', g.index)})" for g in gpus
    )
    if len({g.name for g in gpus}) > 1:
        line += "; mixed GPU models (roofline peaks and e2e baselines assume identical GPUs)"
    return line


def _lock_file(name: str, index: int) -> str:
    """GPU 0 keeps the lock file of kernel-agent before the pool (``gpu.lock``)."""
    return f"{name}.lock" if index == 0 else f"{name}{index}.lock"


def _held_gpus() -> dict[str, int]:
    """Lock name -> GPU index of the locks this thread holds."""
    gpus: dict[str, int] | None = getattr(_held, "gpus", None)
    if gpus is None:
        gpus = _held.gpus = {}
    return gpus


def _take(path: Path, lock: threading.Lock, *, wait: bool) -> IO[str] | None:
    """``path`` open and flocked, with ``lock`` held; None if either is busy and not
    ``wait``. A wait ends with ``interrupt.Interrupted`` when the run is stopping."""
    if wait:
        while not lock.acquire(timeout=WAIT_S):
            interrupt.check()
    elif not lock.acquire(blocking=False):
        return None
    try:
        fh = open(path, "w")  # noqa: SIM115 - closed by the caller when it unlocks
        try:
            if wait:
                _flock_wait(fh)
            else:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            fh.close()
            raise
    except BlockingIOError:  # another process holds it
        lock.release()
        return None
    except BaseException:
        lock.release()
        raise
    return fh


def _take_shared(path: Path, on_wait: Callable[[], None]) -> IO[str]:
    """``path`` open with a shared ``flock`` (a non-exclusive job, :mod:`gpuqueue`), waited
    for while another process holds it exclusively (``on_wait`` first)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w")  # noqa: SIM115 - closed by the caller when it unlocks
    try:
        try:
            fcntl.flock(fh, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            on_wait()
            _flock_wait(fh, fcntl.LOCK_SH)
    except BaseException:
        fh.close()
        raise
    return fh


def _flock_wait(fh: IO[str], mode: int = fcntl.LOCK_EX) -> None:
    """``mode`` (``LOCK_EX``) on ``fh``, waited for until the run is stopping (``Interrupted``).

    The blocking ``flock`` runs in a daemon thread on a duplicate of the descriptor, so
    waiters get the lock in the kernel's order (no polling against other processes'
    blocking waits) and an abandoned wait neither blocks the exit nor keeps the lock:
    the thread closes its duplicate, and with ``fh`` closed by the caller, a lock it
    gets late is free again."""
    fd = os.dup(fh.fileno())
    done = threading.Event()
    error: list[OSError] = []

    def wait() -> None:
        try:
            fcntl.flock(fd, mode)
        except OSError as exc:
            error.append(exc)
        finally:
            os.close(fd)
            done.set()

    threading.Thread(target=wait, name="gpu-lock-wait", daemon=True).start()
    while not done.wait(WAIT_S):
        interrupt.check()
    if error:
        raise error[0]


def _acquire(
    name: str, gpus: tuple[GPU, ...], on_wait: Callable[[], None] | None = None
) -> tuple[int, threading.Lock, IO[str]]:
    """The first free GPU, else the one with the fewest waiters of this process (round
    robin on ties), waited for (``on_wait`` first)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    files = [_lock_file(name, g.index) for g in gpus]
    with _guard:
        locks = [_thread_locks.setdefault(f, threading.Lock()) for f in files]
    for gpu, file, lock in zip(gpus, files, locks, strict=True):
        if (fh := _take(CACHE_DIR / file, lock, wait=False)) is not None:
            return gpu.index, lock, fh
    if on_wait is not None:
        on_wait()
    with _guard:
        turn = next(_turn)
        k = min(range(len(gpus)), key=lambda i: (_waiting[files[i]], (i - turn) % len(gpus)))
        _waiting[files[k]] += 1
    try:
        fh = _take(CACHE_DIR / files[k], locks[k], wait=True)
    finally:
        with _guard:
            _waiting[files[k]] -= 1
    assert fh is not None
    return gpus[k].index, locks[k], fh


def _inherited_index() -> int:
    value = os.environ.get(INDEX_ENV, "")
    return int(value) if value.isdigit() else 0


@contextlib.contextmanager
def gpu_lock(name: str = "gpu") -> Iterator[int]:
    """Hold one GPU of the pool exclusively; yields its (``nvidia-smi``) index. Raises
    ``interrupt.Interrupted`` when the run is stopping: before it takes the GPU, and as
    the work under it ends (whatever it measured was cut short by the stop)."""
    interrupt.check()
    held = _held_gpus()
    if name in held:
        yield held[name]  # this thread already holds it: the same GPU
        interrupt.check()
        return
    if os.environ.get(ENV) == "1":
        yield _inherited_index()  # our parent process holds it
        interrupt.check()
        return
    job = gpuqueue.current() or gpuqueue.Job()  # untagged: class "default"
    wait, gate = gpuqueue.Wait(job), gpuqueue.gate(name)
    try:
        room = _admit(name, job, wait, pool().gpus)
    except BaseException as exc:
        wait.abandoned(exc)
        raise
    thread_lock: threading.Lock | None = None
    try:
        if job.exclusive:
            index, thread_lock, fh = _acquire(name, room, wait.block)
            gate.placed(job, index)
        else:
            index = room[0].index
            fh = _take_shared(CACHE_DIR / _lock_file(name, index), wait.block)
    except BaseException as exc:
        gate.leave(job)
        wait.abandoned(exc)
        raise
    wait.started(index)
    held[name] = index
    try:
        yield index
    finally:
        del held[name]
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()
        if thread_lock is not None:
            thread_lock.release()
        gate.leave(job)
        wait.ended()
    interrupt.check()


@functools.cache
def _memory_table() -> dict[int, float]:
    """GPU index -> its memory in GB (``nvidia-smi``; {} without it)."""
    cmd = ["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return {}
    table: dict[int, float] = {}
    for line in proc.stdout.splitlines() if proc.returncode == 0 else []:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            with contextlib.suppress(ValueError):
                table[int(parts[0])] = float(parts[1]) / 1024  # MiB
    return table


def _memory_gb(index: int) -> float | None:
    """Memory of GPU ``index`` in GB (None: unknown, so no job shares it)."""
    return _memory_table().get(index)


def _admit(
    name: str, job: gpuqueue.Job, wait: gpuqueue.Wait, gpus: tuple[GPU, ...]
) -> tuple[GPU, ...]:
    """Wait until ``job`` is the head of the GPU job queue of lock ``name`` and a GPU of
    ``gpus`` it may take is free (:meth:`gpuqueue.Gate.admit`); returns those GPUs."""
    capacity = None if job.exclusive else {g.index: _memory_gb(g.index) for g in gpus}
    room = gpuqueue.gate(name).admit(job, wait, tuple(g.index for g in gpus), capacity)
    return tuple(g for g in gpus if g.index in room)


def pinned(index: int) -> dict[str, str]:
    """Environment that shows a process GPU ``index`` alone; {} when it would see no
    other GPU anyway (one visible GPU, or a child of the lock holder)."""
    if os.environ.get(ENV) == "1" or not pool().pin:
        return {}
    return {
        "CUDA_VISIBLE_DEVICES": str(index),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        INDEX_ENV: str(index),
    }


def child_env() -> dict[str, str]:
    """Environment for a subprocess started while this thread holds the GPU lock: it
    does not wait for the lock, runs on the locked GPU and dies with this process
    (``interrupt.die_with_parent``). A stopping run starts none (``Interrupted``)."""
    interrupt.check()
    env = dict(os.environ)
    env[interrupt.PARENT_ENV] = str(os.getpid())
    held = _held_gpus()
    if held or os.environ.get(ENV) == "1":
        env[ENV] = "1"
    if held:
        env.update(pinned(next(iter(held.values()))))
    return env
