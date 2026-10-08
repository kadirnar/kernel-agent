"""An agent's own GPU script, run through the GPU job queue: the ``run_on_gpu`` tool
(docs/MULTIAGENT.md §3.3 "Clean timing", issue #185).

With ``--agent-gpu tool`` (the default with several sessions at once) the agents' Bash
commands see no GPU, so their correctness checks and microbenchmarks cannot run on top of
another session's timed evaluation. They hand the script to ``run_on_gpu`` instead, which
runs it here as a GPU job of class ``dev`` (``gpuqueue.py``): after the evaluations waiting
in the queue, never during a timed job. A script that times something holds the GPU alone;
one that only checks correctness and says how much GPU memory it needs may share it with
other such runs while their memory estimates fit (``gpuqueue.fits``).
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from kernel_agent import gpulock, hygiene, interrupt

MAX_TIMEOUT_S = 120.0  # a dev run is short: longer work is an evaluation
OUTPUT_CHARS = 8000  # the tail of its output the agent gets
#: Variables of the hold (``gpulock.child_env``) that win over the session's environment: the
#: locked GPU, the hold itself and its CPUs
_HOLD_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    gpulock.ENV,
    gpulock.INDEX_ENV,
    gpulock.HOLD_ENV,
    interrupt.PARENT_ENV,
    hygiene.CPUS_ENV,
    hygiene.NICE_ENV,
    "MAX_JOBS",
)


def script_env(session: Mapping[str, str] | None) -> dict[str, str]:
    """The environment of a dev run in this thread's hold: the session's own (``session``:
    its ``runner.agent_env``, so its API key variables stay blanked) on top of the hold's
    (``gpulock.child_env``), whose GPU it sees."""
    hold = gpulock.child_env()
    env = {**hold, **(session or {})}
    for key in _HOLD_KEYS:
        if key in hold:
            env[key] = hold[key]
        else:
            env.pop(key, None)  # the session's CUDA_VISIBLE_DEVICES="" hid the GPU
    return env


def run_script(
    script: Path,
    args: list[str] | tuple[str, ...] = (),
    *,
    cwd: Path,
    timeout: float = MAX_TIMEOUT_S,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """``python SCRIPT ARGS`` in ``cwd`` under the GPU lock as the context's GPU job (the
    tool tags it ``dev``), stopped with everything it started after ``timeout`` seconds (at
    most :data:`MAX_TIMEOUT_S`). ``env``: the session's environment (:func:`script_env`).
    Returns ``status`` (``ok``, ``failed``: a non-zero exit, ``timeout``), ``returncode``,
    ``seconds``, ``output`` (the tail of stdout and stderr) and ``gpu_index``."""
    limit = min(max(float(timeout), 1.0), MAX_TIMEOUT_S)
    cmd = [sys.executable, str(script), *map(str, args)]
    with gpulock.gpu_lock() as gpu:
        start = time.perf_counter()
        full = script_env(env)
        with subprocess.Popen(
            cmd,
            cwd=cwd,
            env=full,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        ) as proc:
            hygiene.apply_env(proc.pid, full)  # it need not import kernel_agent to apply them
            status = "ok"
            try:
                output, _ = proc.communicate(timeout=limit)
            except subprocess.TimeoutExpired:
                interrupt.kill(proc.pid)  # with everything it started
                output, _ = proc.communicate()
                status = "timeout"
        seconds = round(time.perf_counter() - start, 2)
    if status == "ok" and proc.returncode != 0:
        status = "failed"
    output = output or ""
    if len(output) > OUTPUT_CHARS:
        output = f"... ({len(output) - OUTPUT_CHARS} characters cut)\n" + output[-OUTPUT_CHARS:]
    return {
        "status": status,
        "returncode": proc.returncode,
        "seconds": seconds,
        **({"timeout_s": limit} if status == "timeout" else {}),
        "output": output,
        "gpu_index": gpu,
    }
