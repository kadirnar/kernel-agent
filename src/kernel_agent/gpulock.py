"""Exclusive GPU lock so concurrent agents never benchmark at once.

The lock is an ``flock`` on a file in the cache dir (across processes) plus a
``threading.Lock`` (across the threads of this process: parallel agents run
their evaluation tools in worker threads of one orchestrator). It is
re-entrant within a thread. A child process started while the lock is held
gets ``KERNEL_AGENT_LOCK_HELD=1`` through :func:`child_env`, so it does not wait
for its own parent.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import threading
from collections.abc import Iterator

from kernel_agent.toolchain import CACHE_DIR

ENV = "KERNEL_AGENT_LOCK_HELD"

_guard = threading.Lock()
_thread_locks: dict[str, threading.Lock] = {}
_held = threading.local()


def _held_names() -> set[str]:
    names: set[str] | None = getattr(_held, "names", None)
    if names is None:
        names = _held.names = set()
    return names


@contextlib.contextmanager
def gpu_lock(name: str = "gpu") -> Iterator[None]:
    held = _held_names()
    if os.environ.get(ENV) == "1" or name in held:
        yield  # our parent process, or this thread, already holds it
        return
    with _guard:
        thread_lock = _thread_locks.setdefault(name, threading.Lock())
    with thread_lock:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_DIR / f"{name}.lock", "w") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            held.add(name)
            try:
                yield
            finally:
                held.discard(name)
                fcntl.flock(fh, fcntl.LOCK_UN)


def child_env() -> dict[str, str]:
    """Environment for a subprocess started while this thread holds the GPU lock."""
    env = dict(os.environ)
    if _held_names() or os.environ.get(ENV) == "1":
        env[ENV] = "1"
    return env
