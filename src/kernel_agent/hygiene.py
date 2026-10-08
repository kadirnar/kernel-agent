"""Clean timing under concurrent load (docs/MULTIAGENT.md §3.3 "Clean timing", issue #185).

With several agent sessions at once (``improve --agents N``) one session's evaluation is timed
while the others compile (nvcc builds are 31 % of a kernel session) and run their own
scripts. :func:`active` (``improve`` enters it when more than one session runs) turns on what
keeps the timed jobs clean:

* **CPU isolation** (:class:`Plan`): the CPUs this process may use are split into the
  *timing cores* (``--timing-cores``, default the last 2 physical cores with their hardware
  threads) and the *other cores*. A subprocess of a timed GPU job (``gpulock.child_env``
  while an exclusive job holds the GPU: the evaluator, the end-to-end worker, a sweep) runs
  on the timing cores at nice :data:`TIMED_NICE`; what kernel-agent starts for the agents
  (the Claude Code CLIs and so their Bash commands, ``kernels/prebuild.py`` builds,
  ``run_on_gpu`` correctness runs) runs on the other cores at nice :data:`BACKGROUND_NICE`,
  with ``MAX_JOBS`` (the parallel nvcc / ninja jobs of ``torch.utils.cpp_extension``) at
  each session's share of them (a template-heavy nvcc job takes about 2 GB of host memory:
  N sessions building at once must not swap). Below :data:`MIN_CPUS` CPUs the affinity is
  off (two cores less would starve the builds); the nice values and ``MAX_JOBS`` still apply.
* **Builds off the GPU lock**: ``run_evaluation`` and ``run_sweep`` compile a candidate's
  ``load_inline`` extensions first, in a process without a GPU (``kernels/prebuild.py``).
* **Dirty-timing re-runs**: ``telemetry.HoldWatch`` around a timed job and the evaluator's
  own CPU wait mark a measurement ``timing_dirty``; it is measured once more, first of its
  class in the GPU queue, and the first result is not recorded.

A child process gets its CPUs and nice value through the environment (:data:`CPUS_ENV`,
:data:`NICE_ENV`): ``import kernel_agent`` applies them (:func:`apply_inherited`) before the
child imports torch. The Claude Code CLIs are not Python: the session runner moves each one
once it started (:func:`move_session`). Without ``/proc`` and ``sched_setaffinity`` (not
Linux) only ``MAX_JOBS`` applies.
"""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

CPUS_ENV = "KERNEL_AGENT_CPUS"  # a child's CPUs, e.g. "4,5,10,11"
NICE_ENV = "KERNEL_AGENT_NICE"  # ... and its nice value
SESSION_ENV = "KERNEL_AGENT_SESSION"  # a Claude Code CLI's mark (move_session finds it by it)
TIMED_NICE = -5  # a timed job's processes (needs RLIMIT_NICE; else they keep their own)
BACKGROUND_NICE = 10  # builds, the agents' CLIs and what they run
TIMING_CORES = 2  # physical cores for the timed jobs (--timing-cores)
MIN_CPUS = 8  # fewer CPUs: no affinity (the nice values and MAX_JOBS still apply)

_active: Hygiene | None = None


@dataclass(frozen=True)
class Plan:
    """A split of this process's CPUs: ``timing`` for the timed jobs, the rest for everything
    else; ``affinity``: whether processes are pinned to their part (False below
    :data:`MIN_CPUS`); ``agents``: the sessions that share the rest (``MAX_JOBS``)."""

    cpus: tuple[int, ...]
    timing: tuple[int, ...]
    affinity: bool = True
    agents: int = 1

    @property
    def other(self) -> tuple[int, ...]:
        return tuple(c for c in self.cpus if c not in self.timing) or self.cpus

    @property
    def max_jobs(self) -> int:
        """``MAX_JOBS`` of a build: a session's share of the other cores."""
        return max(1, len(self.other) // max(self.agents, 1))

    @classmethod
    def detect(cls, cores: int | None = None, agents: int = 1) -> Plan | None:
        """The split of this process's CPUs with the last ``cores`` physical cores (default
        :data:`TIMING_CORES`) for timing and the rest shared by ``agents`` sessions; None for
        0 cores, or when they would leave no other core."""
        cores = TIMING_CORES if cores is None else cores
        cpus = _cpus()
        groups = _physical_cores(cpus)
        if cores <= 0 or cores >= len(groups):
            return None
        timing = tuple(sorted(c for group in groups[-cores:] for c in group))
        return cls(cpus=cpus, timing=timing, affinity=len(cpus) >= MIN_CPUS, agents=agents)

    def describe(self) -> str:
        """One line for the run's log."""
        what = f"timing cores {_ranges(self.timing)}, the rest on {_ranges(self.other)}"
        if not self.affinity:
            what = f"no CPU affinity (fewer than {MIN_CPUS} CPUs)"
        return f"{what}; timed jobs at nice {TIMED_NICE}, builds and agents at nice " + (
            f"{BACKGROUND_NICE} with MAX_JOBS={self.max_jobs}"
        )


@dataclass(frozen=True)
class Hygiene:
    """What :func:`active` turned on: ``plan`` (None: no CPU isolation, ``--timing-cores 0``)."""

    plan: Plan | None = None


def _cpus() -> tuple[int, ...]:
    try:
        return tuple(sorted(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return tuple(range(os.cpu_count() or 1))


def _physical_cores(cpus: tuple[int, ...]) -> list[tuple[int, ...]]:
    """``cpus`` grouped by physical core (hardware threads together), in the order of their
    first CPU; one group per CPU when the topology is not readable."""
    groups: dict[tuple[str, str], list[int]] = {}
    for cpu in cpus:
        top = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            key = ((top / "physical_package_id").read_text(), (top / "core_id").read_text())
        except OSError:
            key = ("", str(cpu))
        groups.setdefault(key, []).append(cpu)
    return sorted((tuple(g) for g in groups.values()), key=lambda g: g[0])


def _ranges(cpus: tuple[int, ...]) -> str:
    """``0-3,6-9`` for a CPU list."""
    spans: list[list[int]] = []
    for cpu in sorted(cpus):
        if spans and cpu == spans[-1][1] + 1:
            spans[-1][1] = cpu
        else:
            spans.append([cpu, cpu])
    return ",".join(f"{a}-{b}" if b > a else str(a) for a, b in spans)


# ------------------------------------------------------------------ the process's setting


@contextlib.contextmanager
def active(timing_cores: int | None = None, agents: int = 1) -> Iterator[Hygiene]:
    """Clean timing for the work of this process while it is entered (``improve --agents N``,
    ``agents``: N): CPU isolation with ``timing_cores`` physical cores (default
    :data:`TIMING_CORES`; 0: none), builds before the GPU lock and dirty-timing re-runs."""
    global _active
    before, _active = _active, Hygiene(Plan.detect(timing_cores, agents))
    try:
        yield _active
    finally:
        _active = before


def current() -> Hygiene | None:
    """The clean-timing setting of this process (None: off, as with one session at a time)."""
    return _active


def timed_env() -> dict[str, str]:
    """Environment of a timed job's subprocess: the timing cores, nice :data:`TIMED_NICE`
    ({} when off)."""
    plan = _active.plan if _active is not None else None
    if plan is None:
        return {}
    cpus = {CPUS_ENV: ",".join(map(str, plan.timing))} if plan.affinity else {}
    return {**cpus, NICE_ENV: str(TIMED_NICE)}


def background_env() -> dict[str, str]:
    """Environment of a build or of what an agent runs: the other cores, nice
    :data:`BACKGROUND_NICE` and ``MAX_JOBS`` ({} when off)."""
    plan = _active.plan if _active is not None else None
    if plan is None:
        return {}
    cpus = {CPUS_ENV: ",".join(map(str, plan.other))} if plan.affinity else {}
    return {**cpus, NICE_ENV: str(BACKGROUND_NICE), "MAX_JOBS": str(plan.max_jobs)}


# ------------------------------------------------------------------ applying it


def _parse_cpus(value: str) -> set[int]:
    return {int(c) for c in value.split(",") if c.strip().isdigit()}


def _tasks(pid: int) -> list[int]:
    """The threads of process ``pid`` ([pid] without ``/proc``)."""
    try:
        return [int(t) for t in os.listdir(f"/proc/{pid}/task") if t.isdigit()] or [pid]
    except OSError:
        return [pid]


def apply(pid: int, cpus: set[int] | None, nice: int | None) -> None:
    """Every thread of process ``pid`` (0: this one) on ``cpus`` at ``nice``; what the system
    refuses (a nice value below the limit, a CPU outside the cgroup) is left as it was."""
    for tid in _tasks(pid or os.getpid()):
        if cpus:
            with contextlib.suppress(AttributeError, OSError, ValueError):
                os.sched_setaffinity(tid, cpus)
        if nice is not None:
            with contextlib.suppress(AttributeError, OSError):
                os.setpriority(os.PRIO_PROCESS, tid, nice)


def background_thread() -> None:
    """The calling thread (a helper thread that starts processes) on the other cores at nice
    :data:`BACKGROUND_NICE`: what it starts inherits that (nothing when off)."""
    plan = _active.plan if _active is not None else None
    if plan is None:
        return
    tid = threading.get_native_id()
    if plan.affinity:
        with contextlib.suppress(AttributeError, OSError, ValueError):
            os.sched_setaffinity(tid, set(plan.other))
    with contextlib.suppress(AttributeError, OSError):
        os.setpriority(os.PRIO_PROCESS, tid, BACKGROUND_NICE)


def apply_env(pid: int, env: Mapping[str, str]) -> None:
    """Process ``pid`` (0: this one) on the CPUs and at the nice value of ``env``
    (:data:`CPUS_ENV`, :data:`NICE_ENV`; nothing without them)."""
    cpus, nice = env.get(CPUS_ENV, ""), env.get(NICE_ENV, "")
    if cpus or nice:
        level = int(nice) if nice.lstrip("-").isdigit() else None
        apply(pid, _parse_cpus(cpus) or None, level)


def apply_inherited() -> None:
    """The CPUs and nice value a parent gave this process (:data:`CPUS_ENV`,
    :data:`NICE_ENV`; ``import kernel_agent`` calls it). A child that is not kernel-agent's
    (an agent's script, ``devrun.py``) gets them from its parent (:func:`apply_env`)."""
    apply_env(0, os.environ)


def _children(pid: int) -> list[int]:
    """The child processes of ``pid`` (from ``/proc``; [] without it)."""
    out = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return []
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            stat = Path(f"/proc/{entry}/stat").read_text()
        except OSError:
            continue
        fields = stat.rsplit(")", 1)[-1].split()  # after "pid (comm)": state, ppid, ...
        if len(fields) > 1 and fields[1] == str(pid):
            out.append(int(entry))
    return out


def environ_has(pid: int, key: str, value: str) -> bool:
    """Whether process ``pid`` started with ``key=value`` in its environment."""
    try:
        data = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return False
    return f"{key}={value}".encode() in data.split(b"\0")


def move_session(mark: str) -> list[int]:
    """Move the Claude Code CLI started with ``SESSION_ENV=mark`` (a child of this process)
    and its children to the other cores at nice :data:`BACKGROUND_NICE`; what it starts
    later (its Bash commands, their builds) inherits that. Returns the moved pids ([]:
    clean timing is off or the CLI is not found)."""
    plan = _active.plan if _active is not None else None
    if plan is None:
        return []
    cpus = set(plan.other) if plan.affinity else None
    moved: list[int] = []
    todo = [p for p in _children(os.getpid()) if environ_has(p, SESSION_ENV, mark)]
    while todo:
        pid = todo.pop()
        apply(pid, cpus, BACKGROUND_NICE)
        moved.append(pid)
        todo += _children(pid)
    return moved
