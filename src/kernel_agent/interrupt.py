"""Ctrl-C and SIGTERM for ``kernel-agent improve``: stop at once, leave nothing running.

:func:`run` runs the command's coroutine like ``asyncio.run``, with its own SIGINT and
SIGTERM handler. The first signal

* makes :func:`requested` true. No GPU work starts any more: ``gpulock.gpu_lock`` and
  ``gpulock.child_env`` raise :class:`Interrupted`, and a thread waiting for the GPU lock
  stops waiting. Work that holds the GPU lock when the stop comes raises
  :class:`Interrupted` as it lets the lock go, so a measurement cut short is never
  recorded as a result (a crash, a failed evaluation). No agent session starts either;
* cancels the main task. The improve loop records the slice, research session or
  integration that was running as interrupted (``improve.json``) and the Agent SDK
  closes its sessions;
* terminates every process the command started, with their children (worker and
  evaluator subprocesses, Claude Code and the commands its sessions run): SIGTERM at
  once, SIGKILL for what is still alive after :data:`GRACE_S` seconds.

The command then exits with 130. A second signal kills what is left and exits at once.
When the command ends, any process it started that is still alive is terminated the
same way, so nothing holds the GPU after the run.

A child started with ``gpulock.child_env()`` (every worker and evaluator subprocess)
also dies with the process that started it when that one is killed outright:
``import kernel_agent`` in the child sets Linux's ``PR_SET_PDEATHSIG``
(:func:`die_with_parent`).

Processes are found in ``/proc``: the descendants of this process that it did not
already have when :func:`run` started. Without ``/proc`` (not Linux) only the first two
points apply.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
import threading
import time
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any

GRACE_S = 10.0  # SIGTERM → SIGKILL
EXIT_CODE = 130  # of a command stopped by a signal (SIGINT or SIGTERM)
#: Set by ``gpulock.child_env``: the pid of the process that starts the child.
PARENT_ENV = "KERNEL_AGENT_PARENT"
PR_SET_PDEATHSIG = 1  # linux/prctl.h

#: A process: (pid, start time in clock ticks since boot). A pid can be reused, the
#: pair cannot.
Proc = tuple[int, str]

_stop = threading.Event()  # set by the first signal; only ever waited on through is_set()
_signal: list[int] = []  # the signal that stopped the run


class Interrupted(KeyboardInterrupt):
    """The run is stopping (Ctrl-C or SIGTERM): what was running is not a result."""


def requested() -> bool:
    """A signal has asked the run to stop."""
    return _stop.is_set()


def check() -> None:
    """Raise :class:`Interrupted` once the run is stopping."""
    if _stop.is_set():
        raise Interrupted


def signal_name() -> str | None:
    """``SIGINT`` / ``SIGTERM``: the signal that stopped the run (None: none did)."""
    return signal.Signals(_signal[-1]).name if _signal else None


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ processes


def _stat(pid: int) -> tuple[int, str, str] | None:
    """(parent pid, start time, state) of ``pid`` from ``/proc`` (None: no such process)."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            stat = fh.read()
    except OSError:
        return None
    fields = stat[stat.rfind(b")") + 2 :].split()  # the name may contain spaces and ")"
    return int(fields[1]), fields[19].decode(), fields[0].decode()


def descendants(pid: int | None = None) -> set[Proc]:
    """The processes below ``pid`` (default: this process), at any depth."""
    try:
        pids = [int(p) for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return set()
    children: dict[int, list[Proc]] = {}
    for p in pids:
        if (stat := _stat(p)) is not None:
            children.setdefault(stat[0], []).append((p, stat[1]))
    out: set[Proc] = set()
    todo = [os.getpid() if pid is None else pid]
    while todo:
        for proc in children.get(todo.pop(), []):
            if proc not in out:
                out.add(proc)
                todo.append(proc[0])
    return out


def alive(proc: Proc) -> bool:
    """``proc`` runs (not ended, not a zombie, its pid not reused)."""
    stat = _stat(proc[0])
    return stat is not None and stat[1] == proc[1] and stat[2] != "Z"


def _send(proc: Proc, sig: int) -> None:
    """Send ``sig`` to ``proc`` unless it has ended (its pid may be another process's
    then): through a pidfd, opened before the start time is compared."""
    try:
        fd = os.pidfd_open(proc[0])
    except OSError:
        return
    try:
        if (stat := _stat(proc[0])) is not None and stat[1] == proc[1]:
            signal.pidfd_send_signal(fd, sig)
    except OSError:
        pass  # it ended in between
    finally:
        os.close(fd)


def process(pid: int) -> Proc | None:
    """``pid`` with its start time (None: no such process)."""
    stat = _stat(pid)
    return (pid, stat[1]) if stat is not None else None


def send(procs: set[Proc], sig: int) -> None:
    """Send ``sig`` to each of ``procs`` that has not ended (``hygiene.py``: SIGSTOP and
    SIGCONT of the agents' background work)."""
    for proc in procs:
        _send(proc, sig)


def kill(pid: int) -> None:
    """SIGKILL process ``pid`` and every process below it, also those in process groups of
    their own (nvcc starts one: a ``killpg`` of the caller's group misses it)."""
    procs = descendants(pid)
    if (root := process(pid)) is not None:
        procs.add(root)
    send(procs, signal.SIGKILL)


def terminate(procs: set[Proc], grace: float = GRACE_S) -> None:
    """SIGTERM ``procs``, then SIGKILL the ones still alive after ``grace`` seconds."""
    for proc in procs:
        _send(proc, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while (left := {p for p in procs if alive(p)}) and time.monotonic() < deadline:
        time.sleep(0.05)
    for proc in left:
        _send(proc, signal.SIGKILL)


# ------------------------------------------------------------------ in a child


def die_with_parent() -> None:
    """In a child started with ``gpulock.child_env()`` (:data:`PARENT_ENV` names its
    parent): die when the parent dies, also when it is killed outright (Linux
    ``PR_SET_PDEATHSIG``), so no worker keeps the GPU after its run is gone.

    No-op in any other process: one not started that way, or a grandchild that only
    inherited the variable (its parent is not the one named)."""
    parent = os.environ.get(PARENT_ENV, "")
    if not sys.platform.startswith("linux") or parent != str(os.getppid()):
        return
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
            return
    except (OSError, AttributeError):
        return
    if str(os.getppid()) != parent:  # the parent died before prctl took effect
        os._exit(EXIT_CODE)


# ------------------------------------------------------------------ the run


@dataclass
class _Run:
    before: set[Proc]  # processes this one had before run(): not the run's to stop
    grace: float
    task: asyncio.Task[Any] | None = None
    loop: asyncio.AbstractEventLoop | None = None
    done: bool = False

    def mine(self) -> set[Proc]:
        return descendants() - self.before

    def handler(self, signum: int, frame: Any) -> None:
        """Runs in the main thread between two bytecodes of whatever it was doing, so it
        takes no lock that code may hold (``print`` included): the stopper thread logs and
        terminates. ``_stop`` is only ever waited on by that thread."""
        if _stop.is_set():  # the second signal: exit at once
            for proc in self.mine():
                _send(proc, signal.SIGKILL)
            os.write(2, b"\nsecond signal: exiting at once\n")
            os._exit(EXIT_CODE)
        _signal.append(signum)
        _stop.set()
        task, loop = self.task, self.loop
        if task is not None and loop is not None and not task.done():
            task.cancel()
            with contextlib.suppress(RuntimeError):  # the loop is closed
                loop.call_soon_threadsafe(lambda: None)  # wake it if it waits in select()

    def stopper(self) -> None:
        """Daemon thread: on the first signal, terminate the run's processes."""
        while not _stop.wait(0.1):
            if self.done:
                return
        _log(
            f"{signal_name()}: stopping. No new work starts, running work is stopped and "
            "recorded as interrupted; Ctrl-C again exits at once."
        )
        for proc in self.mine():
            _send(proc, signal.SIGTERM)
        deadline = time.monotonic() + self.grace
        while not self.done and time.monotonic() < deadline:
            time.sleep(0.05)
        if not self.done:  # started since, or ignoring SIGTERM
            for proc in self.mine():
                _send(proc, signal.SIGKILL)


def run[T](main: Coroutine[Any, Any, T], *, grace: float | None = None) -> T:
    """``asyncio.run(main)``, stopped by SIGINT / SIGTERM as the module docstring says:
    then it raises :class:`Interrupted` (exit code :data:`EXIT_CODE`). ``grace``: seconds
    from SIGTERM to SIGKILL (default :data:`GRACE_S`). The signal handlers are installed
    only in the main thread."""
    grace = GRACE_S if grace is None else grace
    _stop.clear()  # before the handlers: it takes the lock that set() takes
    _signal.clear()
    state = _Run(before=descendants(), grace=grace)

    async def main_task() -> T:
        state.task, state.loop = asyncio.current_task(), asyncio.get_running_loop()
        return await main

    previous: dict[int, Any] = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, state.handler)
    threading.Thread(target=state.stopper, name="interrupt", daemon=True).start()
    try:
        return asyncio.run(main_task())
    except BaseException as exc:
        if _stop.is_set() and not isinstance(exc, KeyboardInterrupt):
            raise Interrupted from exc  # a cancellation, or an error of work cut short
        raise
    finally:
        state.done = True
        terminate(state.mine(), grace)  # nothing the run started outlives it
        for signum, handler in previous.items():
            signal.signal(signum, handler)
