"""Megakernel kit for native engines (issue #225): an on-GPU instruction interpreter whose
dependencies are global-memory counters instead of grid barriers.

* ``include/ka_mk.cuh`` (on every native project's include path,
  :func:`kernel_agent.native.project.toolkit_includes`): the interpreter (one block per SM,
  cooperative launch), ``ka_mk::wait`` / ``ka_mk::signal`` (``ld.acquire.gpu`` /
  ``red.release.gpu``), a producer warp per block that streams the next instructions'
  weights into a shared-memory page pool before their counters are met (``cp.async`` +
  mbarrier on sm_80-sm_89, bulk copies on sm_90+), the watchdog and ``KA_MK_TRACE``;
* :mod:`.schedule`: a stage's ops and tile-level edges as per-SM queues, counter targets
  and the serialised int32 program;
* :mod:`.simulate`: a discrete-event simulation of a schedule (deadlocks, ordering, the
  predicted time from per-op costs or a trace);
* :mod:`.runtime`: the device buffers of one schedule (program, counters, tensor table,
  watchdog status, trace) and :class:`.runtime.MegakernelHang`.

The example project ``kernel_agent/agent/examples/native_megakernel`` builds an RMSNorm →
GEMV → residual chain with it, next to a graph + PDL and a grid-barrier baseline; the
``native-engines`` skill's ``megakernel.md`` is the milestone ladder.
"""

from __future__ import annotations

from pathlib import Path

HEADER = "ka_mk.cuh"


def include_dir() -> Path:
    """Directory of :data:`HEADER` (it includes ``ka_launch.cuh``: put
    :func:`kernel_agent.concurrency.include_dir` on the path too)."""
    return Path(__file__).resolve().parent / "include"
