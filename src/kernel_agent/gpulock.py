"""Exclusive GPU locks so concurrent agents never benchmark on the same GPU at once.

Every GPU of the pool (:func:`pool`) has a lock: an ``flock`` on a file in the cache
dir (across processes) plus a ``threading.Lock`` (across the threads of this process:
parallel agents run their evaluation tools in worker threads of one orchestrator).
:func:`gpu_lock` takes the first free GPU without waiting, or else waits for the GPU
with the fewest waiters in this process, and yields its index. It is re-entrant
within a thread and keeps that GPU. A child process started while the lock is held
gets ``KERNEL_AGENT_LOCK_HELD=1`` through :func:`child_env`, so it does not wait for
its own parent, and, when more than one GPU is visible, ``CUDA_VISIBLE_DEVICES`` of
the locked GPU, so it runs where its parent locked.

The pool is what ``nvidia-smi`` lists (the parent never initialises CUDA to find
it), restricted to an inherited ``CUDA_VISIBLE_DEVICES`` (indices or ``GPU-`` UUIDs);
``KERNEL_AGENT_GPUS=0,2`` replaces both. Indices are ``nvidia-smi``'s (PCI bus order;
pinned children also get ``CUDA_DEVICE_ORDER=PCI_BUS_ID``). GPU 0 locks ``gpu.lock``,
the one lock file of kernel-agent before the pool, so old and new processes on the
same GPU still exclude each other; GPU ``i`` locks ``gpu{i}.lock``. Without
``nvidia-smi`` (or with no GPU visible) the pool is GPU 0 alone and a child's
environment only gains ``KERNEL_AGENT_LOCK_HELD``, as before the pool.
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
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from kernel_agent.toolchain import CACHE_DIR

ENV = "KERNEL_AGENT_LOCK_HELD"
GPUS_ENV = "KERNEL_AGENT_GPUS"  # the pool as nvidia-smi indices, e.g. "0,2"
INDEX_ENV = "KERNEL_AGENT_GPU"  # a pinned child: the GPU its parent locked

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
    ``wait``."""
    if not lock.acquire(blocking=wait):
        return None
    try:
        fh = open(path, "w")  # noqa: SIM115 - closed by the caller when it unlocks
        try:
            fcntl.flock(fh, fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB)
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


def _acquire(name: str, gpus: tuple[GPU, ...]) -> tuple[int, threading.Lock, IO[str]]:
    """The first free GPU, else the one with the fewest waiters of this process (round
    robin on ties), waited for."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    files = [_lock_file(name, g.index) for g in gpus]
    with _guard:
        locks = [_thread_locks.setdefault(f, threading.Lock()) for f in files]
    for gpu, file, lock in zip(gpus, files, locks, strict=True):
        if (fh := _take(CACHE_DIR / file, lock, wait=False)) is not None:
            return gpu.index, lock, fh
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
    """Hold one GPU of the pool exclusively; yields its (``nvidia-smi``) index."""
    held = _held_gpus()
    if name in held:
        yield held[name]  # this thread already holds it: the same GPU
        return
    if os.environ.get(ENV) == "1":
        yield _inherited_index()  # our parent process holds it
        return
    index, thread_lock, fh = _acquire(name, pool().gpus)
    held[name] = index
    try:
        yield index
    finally:
        del held[name]
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()
        thread_lock.release()


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
    does not wait for the lock and runs on the locked GPU."""
    env = dict(os.environ)
    held = _held_gpus()
    if held or os.environ.get(ENV) == "1":
        env[ENV] = "1"
    if held:
        env.update(pinned(next(iter(held.values()))))
    return env
