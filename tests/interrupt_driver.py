"""Test helper (issue #94): ``kernel-agent improve --dry-run`` with a real child process
that hangs until it is killed, in a process of its own so a test can interrupt it.

    python interrupt_driver.py MODE PIDS_FILE IMPROVE_ARGS...

The simulated loop (``kernel_agent.dryrun``) runs as the CLI runs it (``cli.main``). The
child starts where real ones start (``MODE``):

* ``integration``: the first ``e2e_ab`` worker of an integration, as ``worker.call_worker``
  starts it: under the GPU lock, with ``gpulock.child_env()``, in the event loop's thread;
* ``evaluation``: an evaluation in the first kernel session, the same way but in a worker
  thread (``asyncio.to_thread``, as the ``evaluate`` tool runs);
* ``agent``: the Claude Code process of the first kernel session (an asyncio subprocess
  with the plain environment, as the Agent SDK starts it);
* ``none``: no child.

The child starts a grandchild and appends ``<child pid> <grandchild pid>`` to PIDS_FILE.
With ``KA_TEST_TRAP_SIGTERM=1`` both ignore SIGTERM (only SIGKILL ends them);
``KA_TEST_GRACE`` replaces ``interrupt.GRACE_S``. The GPU locks are files next to
PIDS_FILE, not the machine's.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from kernel_agent import charts, cli, dryrun, gpulock, interrupt, orchestrator

HANG = """
import os, signal, subprocess, sys, time
import kernel_agent  # as every worker does
if os.environ.get("KA_TEST_TRAP_SIGTERM") == "1":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)  # inherited by the grandchild
grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
with open(sys.argv[1], "a") as fh:
    fh.write(f"{os.getpid()} {grandchild.pid}\\n")
time.sleep(600)
"""


def main() -> int:
    mode, pids = sys.argv[1], Path(sys.argv[2])
    cmd = [sys.executable, "-c", HANG, str(pids)]
    gpulock.CACHE_DIR = pids.parent / "locks"
    interrupt.GRACE_S = float(os.environ.get("KA_TEST_GRACE", interrupt.GRACE_S))
    orchestrator.toolchain.setup = dryrun.SimToolchain  # type: ignore[assignment]
    charts.available = lambda: False  # type: ignore[assignment]
    pending = [mode != "none"]  # the child starts once

    def gpu_child() -> None:
        with gpulock.gpu_lock():
            subprocess.run(cmd, capture_output=True, text=True, env=gpulock.child_env())

    simulated_worker, simulated_agent = dryrun.World.worker, dryrun.World.run_agent

    def worker(self: Any, run: Any, command: str, *args: str, **kwargs: Any) -> Any:
        if mode == "integration" and command == "e2e_ab" and pending:
            pending.clear()
            gpu_child()
        return simulated_worker(self, run, command, *args, **kwargs)

    async def run_agent(self: Any, name: str, **kwargs: Any) -> Any:
        if mode in ("evaluation", "agent") and name.startswith("kernel-") and pending:
            pending.clear()
            if mode == "evaluation":
                await asyncio.to_thread(gpu_child)
            else:
                proc = await asyncio.create_subprocess_exec(*cmd)
                await proc.wait()
        return await simulated_agent(self, name, **kwargs)

    dryrun.World.worker = worker  # type: ignore[method-assign]
    dryrun.World.run_agent = run_agent  # type: ignore[method-assign]
    return cli.main(["improve", *sys.argv[3:]])


if __name__ == "__main__":
    raise SystemExit(main())
