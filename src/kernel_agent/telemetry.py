"""GPU clocks, temperature and power around timed end-to-end runs.

Read through NVML (``pynvml``, from the ``nvidia-ml-py`` package) when it is
importable, else through ``nvidia-smi``; neither is a dependency, and without
both there are no samples. A measurement is *throttled*, and logged as a
warning, when the GPU actually slowed down during it:

* a thermal or hardware slowdown reason was active (:data:`SLOWDOWN`), or
* the SM clock of a sample taken under load fell below :data:`CLOCK_DROP` of the
  highest SM clock of the measurement (``clock_drop``).

The clock-event reasons of every sample are recorded (``reasons``); ``sw_power_cap``
alone is informational, since consumer GPUs report it routinely while they run at
full boost. What a power-capped board (A10, T4, L4) sustains is measured once with the
peaks instead (``roofline.sustained_peaks``: the sustained 16-bit matmul rate, the SM
clock and :meth:`Monitor.power_limit_w`). The board's maximum SM clock
(``clocks.max.sm``, ``sm_max_mhz``) is recorded too but is not the reference: boards
often never reach it under load (an RTX 5070 Ti boosts to ~2.9 of its 3.1 GHz). The
paired A/B design keeps throttling
out of the comparison, but the absolute latencies of a throttled measurement are
pessimistic and separate-process numbers noisy.

:meth:`Monitor.processes` lists the other processes with a context on the GPU: one
that computes meanwhile (outside kernel-agent's GPU lock) slows every kernel timed.

Clean timing under concurrent load (``hygiene.py``, issue #185): a :class:`HoldWatch` samples
the GPU's processes around a timed job from the process that holds the GPU lock (no CUDA
there, ``nvidia-smi`` by index), and the timed process measures how long its main thread
waited for a CPU while it timed (:func:`schedstat`, ``cpu_wait_share``); :func:`dirty` says
why such a measurement is not clean.
"""

from __future__ import annotations

import os
import shutil
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

#: NVML clock-event reasons recorded with a sample (``nvmlClocksEventReason*``).
REASONS = {
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x20: "sw_thermal",
    0x40: "hw_thermal",
    0x80: "hw_power_brake",
}
#: Reasons that mean the GPU slowed down (``sw_power_cap`` is not one of them).
SLOWDOWN = frozenset({"hw_slowdown", "sw_thermal", "hw_thermal", "hw_power_brake"})
#: A sample under load whose SM clock is below this share of the highest one is a drop.
CLOCK_DROP = 0.9
_QUERY = "clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,clocks_event_reasons.active"
_FORMAT = "csv,noheader,nounits"
_APPS = "pid,used_memory,process_name"
#: Seconds between the samples of a GPU's processes while a timed job holds it (HoldWatch)
WATCH_S = 10.0
#: A timed process whose main thread waited for a CPU longer than this share of its timing
#: ran on a contended CPU (``cpu_wait_share``): the launches of its kernels were delayed
CPU_WAIT_SHARE = 0.05
#: A timed process pinned to the timing cores (``hygiene.py``) whose cores other processes used
#: more than this share of its timing ran beside them (``cpu_others_share``)
CPU_OTHERS_SHARE = 0.1
#: A hold during which every task of the host stalled on memory (it swapped) longer than this
#: share of it was not clean (``/proc/pressure/memory`` ``full``)
MEMORY_STALL_SHARE = 0.02


def _bus_id() -> str | None:
    """PCI bus id of the current CUDA device (``00000000:0B:00.0``), None without CUDA."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        p: Any = torch.cuda.get_device_properties(torch.cuda.current_device())
        return f"{p.pci_domain_id:08X}:{p.pci_bus_id:02X}:{p.pci_device_id:02X}.0"
    except Exception:
        return None


def reasons(mask: int) -> list[str]:
    return [name for bit, name in REASONS.items() if mask & bit]


def _mhz(value: str) -> int | None:
    """An ``nvidia-smi`` clock or MiB (None: ``[N/A]``)."""
    try:
        return int(float(value))
    except ValueError:
        return None


class Monitor:
    """Samples of one GPU; :meth:`summary` for results, :func:`warning` for logs."""

    def __init__(self, bus_id: str | None = None) -> None:
        self.samples: list[dict[str, Any]] = []
        self.backend: str | None = None
        self._nvml: Any = None
        self._handle: Any = None
        self._bus = bus_id or _bus_id()
        self._start = time.monotonic()
        if self._bus is None:
            return
        try:
            import pynvml  # type: ignore[import-not-found,unused-ignore]

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByPciBusId(self._bus.encode())
            self._nvml, self.backend = pynvml, "nvml"
        except Exception:
            if shutil.which("nvidia-smi"):
                self.backend = "nvidia-smi"

    def _read(self) -> dict[str, Any]:
        if self.backend == "nvml":
            nvml, h = self._nvml, self._handle
            reasons_of = getattr(nvml, "nvmlDeviceGetCurrentClocksEventReasons", None)
            reasons_of = reasons_of or nvml.nvmlDeviceGetCurrentClocksThrottleReasons  # old name
            return {
                "sm_mhz": nvml.nvmlDeviceGetClockInfo(h, nvml.NVML_CLOCK_SM),
                "sm_max_mhz": nvml.nvmlDeviceGetMaxClockInfo(h, nvml.NVML_CLOCK_SM),
                "mem_mhz": nvml.nvmlDeviceGetClockInfo(h, nvml.NVML_CLOCK_MEM),
                "temp_c": nvml.nvmlDeviceGetTemperature(h, nvml.NVML_TEMPERATURE_GPU),
                "power_w": round(nvml.nvmlDeviceGetPowerUsage(h) / 1000.0, 1),
                "mask": int(reasons_of(h)),
            }
        out = subprocess.run(
            ["nvidia-smi", "-i", str(self._bus), f"--query-gpu={_QUERY}", f"--format={_FORMAT}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        values = [v.strip() for v in out.strip().splitlines()[0].split(",")]
        sm, sm_max, mem, temp, power, mask = values
        return {
            "sm_mhz": int(float(sm)),
            "sm_max_mhz": _mhz(sm_max),
            "mem_mhz": int(float(mem)),
            "temp_c": int(float(temp)),
            "power_w": round(float(power), 1),
            "mask": int(mask, 16),
        }

    def sample(self, label: str = "", *, loaded: bool = True) -> dict[str, Any] | None:
        """One sample (None: no backend, or it failed and is now off). ``loaded``: taken
        right after timed GPU work (False: e.g. before it, when the clocks may be idling);
        only such samples count for a clock drop."""
        if self.backend is None:
            return None
        try:
            values = self._read()
        except Exception:
            self.backend = None
            return None
        mask = values.pop("mask")
        sample = {
            "t_s": round(time.monotonic() - self._start, 2),
            **({"label": label} if label else {}),
            **values,
            "reasons": reasons(mask),
            **({} if loaded else {"loaded": False}),
        }
        self.samples.append(sample)
        return sample

    def power_limit_w(self) -> float | None:
        """The board's enforced power limit in W (None: no backend, or it does not say):
        150 W on an A10, 70 W on a T4, the cap a sustained load runs into (#253)."""
        if self.backend is None:
            return None
        try:
            if self.backend == "nvml":
                milliwatts = self._nvml.nvmlDeviceGetEnforcedPowerLimit(self._handle)
                return round(milliwatts / 1000.0, 1)
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    str(self._bus),
                    "--query-gpu=enforced.power.limit",
                    f"--format={_FORMAT}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            ).stdout
            return round(float(out.strip().splitlines()[0]), 1)
        except Exception:  # [N/A], no such query on an old driver, ...
            return None

    def processes(self) -> list[dict[str, Any]]:
        """The other processes on this GPU, most memory first: ``pid``, ``used_mib`` and
        ``name`` (this process left out; [] without a backend or when it fails)."""
        if self.backend is None:
            return []
        try:
            procs = self._processes()
        except Exception:
            return []
        procs = [p for p in procs if p["pid"] != os.getpid()]
        return sorted(procs, key=lambda p: -(p["used_mib"] or 0))

    def _processes(self) -> list[dict[str, Any]]:
        if self.backend == "nvml":
            nvml, h = self._nvml, self._handle
            out = []
            for proc in nvml.nvmlDeviceGetComputeRunningProcesses(h):
                used = getattr(proc, "usedGpuMemory", None)
                out.append(
                    {
                        "pid": int(proc.pid),
                        "used_mib": round(used / 2**20) if isinstance(used, int) else None,
                        "name": _process_name(int(proc.pid)),
                    }
                )
            return out
        return _compute_apps(str(self._bus))

    def summary(self) -> dict[str, Any]:
        """Ranges over the samples (``{}``: none), the clock-event reasons seen and whether
        the GPU slowed down (``throttle``: the :data:`SLOWDOWN` reasons and ``clock_drop``
        seen, ``throttled_samples``)."""
        if not self.samples:
            return {}
        loaded = [s for s in self.samples if s.get("loaded", True)] or self.samples
        sm = [s["sm_mhz"] for s in loaded]
        floor = round(CLOCK_DROP * max(sm))
        slowed: list[list[str]] = []
        for s in self.samples:
            why = sorted(SLOWDOWN.intersection(s["reasons"]))
            if s.get("loaded", True) and s["sm_mhz"] < floor:
                why.append("clock_drop")
            if why:
                slowed.append(why)
        mem = [s["mem_mhz"] for s in self.samples]
        sm_max = [s["sm_max_mhz"] for s in self.samples if s.get("sm_max_mhz")]
        throttle = sorted({r for why in slowed for r in why})
        return {
            "backend": self.backend or "off",
            "samples": len(self.samples),
            "sm_mhz": [min(sm), round(statistics.median(sm)), max(sm)],
            **({"sm_max_mhz": max(sm_max)} if sm_max else {}),
            "clock_drop_below_mhz": floor,
            "mem_mhz": [min(mem), max(mem)],
            "temp_c_max": max(s["temp_c"] for s in self.samples),
            "power_w_max": max(s["power_w"] for s in self.samples),
            "reasons": sorted({r for s in self.samples for r in s["reasons"]}),
            "throttle": throttle,
            "throttled_samples": len(slowed),
            "throttled": bool(slowed),
        }


def _compute_apps(gpu: str) -> list[dict[str, Any]]:
    """``nvidia-smi``'s processes with a compute context on ``gpu`` (an index or a bus id);
    raises when it fails."""
    text = subprocess.run(
        ["nvidia-smi", "-i", gpu, f"--query-compute-apps={_APPS}", f"--format={_FORMAT}"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    ).stdout
    out = []
    for line in text.strip().splitlines():
        pid, used, name = (v.strip() for v in line.split(",", 2))
        if pid.isdigit():
            out.append({"pid": int(pid), "used_mib": _mhz(used), "name": Path(name).name})
    return out


def compute_processes(gpu: int | str) -> list[dict[str, Any]]:
    """The processes with a compute context on GPU ``gpu`` (an ``nvidia-smi`` index, as
    ``gpulock`` hands out, or a bus id): ``pid``, ``used_mib``, ``name``. Initialises no
    CUDA in this process; [] without ``nvidia-smi`` or when it fails."""
    try:
        return _compute_apps(str(gpu))
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def sm_utilization(gpu: int | str) -> dict[int, int] | None:
    """pid -> SM utilisation (%) of each process on GPU ``gpu`` over about a second
    (``nvidia-smi pmon``; 0: an idle context); None when it cannot tell."""
    cmd = ["nvidia-smi", "pmon", "-c", "1", "-s", "u", "-i", str(gpu)]
    try:
        text = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    out: dict[int, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if line.startswith("#") or len(parts) < 4 or not parts[1].isdigit():
            continue
        out[int(parts[1])] = int(parts[3]) if parts[3].isdigit() else 0
    return out


class HoldWatch:
    """The processes on GPU ``gpu`` (a ``gpulock`` index) around one timed job: sampled when
    it starts, every ``interval`` seconds (default :data:`WATCH_S`) while it runs and when it
    ends. This process and the hold's own processes (``token``: their ``gpulock.HOLD_ENV``)
    do not count; another process was *active* during the job when it ended meanwhile, or
    when it computes at the end (``nvidia-smi pmon``; without it, whenever it is there; an
    idle context, a notebook holding tensors, is no load): it ran outside the GPU lock and
    slowed what was timed (:attr:`active`)."""

    def __init__(self, gpu: int, token: str | None, *, interval: float | None = None) -> None:
        self.gpu = gpu
        self.token = token
        self.interval = WATCH_S if interval is None else interval
        self.samples: list[dict[int, dict[str, Any]]] = []
        self.active: list[dict[str, Any]] = []
        self.memory_stall_share: float | None = None  # of the hold (memory_stall)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stalled: tuple[float, int] | None = None

    def _foreign(self) -> dict[int, dict[str, Any]]:
        from kernel_agent.gpulock import HOLD_ENV
        from kernel_agent.hygiene import environ_has

        token = self.token
        return {
            p["pid"]: p
            for p in compute_processes(self.gpu)
            if p["pid"] != os.getpid() and not (token and environ_has(p["pid"], HOLD_ENV, token))
        }

    def __enter__(self) -> HoldWatch:
        self.samples.append(self._foreign())
        if (stalled := memory_stall()) is not None:
            self._stalled = (time.monotonic(), stalled)
        if self.interval > 0:
            self._thread = threading.Thread(target=self._watch, name="hold-watch", daemon=True)
            self._thread.start()
        return self

    def _watch(self) -> None:
        from kernel_agent import hygiene

        hygiene.background_thread()  # nvidia-smi, started from here, stays off the timing cores
        while not self._stop.wait(self.interval):
            self.samples.append(self._foreign())

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
        self.samples.append(self._foreign())
        self.active = self._active()
        if self._stalled is not None and (stalled := memory_stall()) is not None:
            start, before = self._stalled
            span = time.monotonic() - start
            self.memory_stall_share = round((stalled - before) / 1e6 / span, 4) if span else None

    def _active(self) -> list[dict[str, Any]]:
        seen = {pid: p for sample in self.samples for pid, p in sample.items()}
        last = self.samples[-1] if self.samples else {}
        busy = sm_utilization(self.gpu) if last else {}
        out = []
        for pid, proc in seen.items():
            if pid not in last:
                why = "ended during the timing"
            elif busy is None:
                why = "on the GPU during the timing"
            elif busy.get(pid, 0) > 0:
                why = f"computing at {busy[pid]} % SM"
            else:
                continue  # an idle context (a notebook holding tensors): no load
            out.append({**proc, "why": why})
        return out


def memory_stall() -> int | None:
    """Microseconds all the host's runnable tasks were stalled on memory at once since boot
    (``/proc/pressure/memory`` ``full``: it swaps); None without pressure stall information."""
    try:
        lines = Path("/proc/pressure/memory").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("full") and "total=" in line:
            return int(line.rsplit("total=", 1)[1])
    return None


def schedstat() -> tuple[int, int] | None:
    """This thread's nanoseconds on a CPU and waiting for one (``/proc/thread-self/schedstat``;
    None without it)."""
    try:
        run_ns, wait_ns = Path("/proc/thread-self/schedstat").read_text().split()[:2]
        return int(run_ns), int(wait_ns)
    except (OSError, ValueError):
        return None


def cpu_wait_share(before: tuple[int, int] | None) -> float | None:
    """The share of the time since ``before`` (:func:`schedstat`) this thread waited for a
    CPU while it could run (None: unknown)."""
    now = schedstat()
    if before is None or now is None:
        return None
    ran, waited = now[0] - before[0], now[1] - before[1]
    return round(waited / (ran + waited), 4) if ran + waited > 0 else None


def cores_busy() -> tuple[float, int, int] | None:
    """For a process pinned to its CPUs (``hygiene.CPUS_ENV``: a timed job on the timing
    cores): now, the clock ticks those CPUs were busy (``/proc/stat``) and the ticks this
    process ran (all its threads); None when it is not pinned or without ``/proc``."""
    from kernel_agent.hygiene import CPUS_ENV

    if not os.environ.get(CPUS_ENV):
        return None
    try:
        cpus = {f"cpu{c}" for c in os.sched_getaffinity(0)}
        busy = 0
        for line in Path("/proc/stat").read_text().splitlines():
            name, *ticks = line.split()
            if name in cpus:  # user nice system idle iowait irq softirq steal ...
                values = [int(t) for t in ticks]
                busy += sum(values) - values[3] - values[4]
        fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
        own = int(fields[11]) + int(fields[12])  # utime + stime of every thread
    except (OSError, ValueError, IndexError):
        return None
    return time.monotonic(), busy, own


def others_share(before: tuple[float, int, int] | None) -> float | None:
    """The share of its CPUs' time since ``before`` (:func:`cores_busy`) that other processes
    used them (another process on a timing core or on its hardware-thread sibling slows the
    timed one down without making it wait); None: unknown."""
    now = cores_busy()
    if before is None or now is None:
        return None
    seconds = now[0] - before[0]
    capacity = seconds * os.sysconf("SC_CLK_TCK") * len(os.sched_getaffinity(0))
    if capacity <= 0:
        return None
    others = (now[1] - before[1]) - (now[2] - before[2])
    return round(min(max(others / capacity, 0.0), 1.0), 4)


def dirty(watch: HoldWatch | None, result: dict[str, Any]) -> str | None:
    """Why a timed ``result`` is not clean (None: it is): another process computed on the GPU
    during it (``watch``), the host swapped (all its tasks stalled on memory longer than
    :data:`MEMORY_STALL_SHARE` of the hold), or its timed process waited for a CPU longer
    than :data:`CPU_WAIT_SHARE` of its timing (``cpu_wait_share``)."""
    if watch is not None and watch.active:
        procs = "; ".join(
            f"{p.get('name') or '?'} (pid {p['pid']}, {p.get('used_mib')} MiB): {p['why']}"
            for p in watch.active[:3]
        )
        return f"another process used the GPU outside the GPU lock: {procs}"
    stall = watch.memory_stall_share if watch is not None else None
    if stall is not None and stall > MEMORY_STALL_SHARE:
        return (
            f"the host was out of memory: every task stalled on memory {100 * stall:.0f} % of "
            "the time (swapping; builds or scripts running beside it)"
        )
    share = result.get("cpu_wait_share")
    if isinstance(share, int | float) and share > CPU_WAIT_SHARE:
        return (
            f"the timed process waited for a CPU {100 * share:.0f} % of its timing "
            f"(more than {100 * CPU_WAIT_SHARE:.0f} %): a contended CPU delayed its launches"
        )
    others = result.get("cpu_others_share")
    if isinstance(others, int | float) and others > CPU_OTHERS_SHARE:
        return (
            f"other processes used {100 * others:.0f} % of the timing cores while it timed "
            f"(more than {100 * CPU_OTHERS_SHARE:.0f} %; a process on a core's other hardware "
            "thread slows it down too)"
        )
    return None


def _process_name(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip() or None
    except OSError:
        return None


def warning(summary: dict[str, Any] | None) -> str | None:
    """Log line for a throttled measurement (None: no samples, or the GPU did not slow
    down, e.g. ``sw_power_cap`` reported at full clocks)."""
    if not summary or not summary.get("throttled"):
        return None
    lo, _, hi = summary["sm_mhz"]
    why = [
        f"SM clock below {summary.get('clock_drop_below_mhz')} MHz" if r == "clock_drop" else r
        for r in summary.get("throttle") or []
    ]
    return (
        f"GPU throttled ({', '.join(why)}) in {summary['throttled_samples']}/"
        f"{summary['samples']} samples: SM clock {lo}-{hi} MHz, up to {summary['temp_c_max']} "
        f"°C and {summary['power_w_max']} W"
    )
