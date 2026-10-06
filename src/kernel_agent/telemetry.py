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
full boost. The board's maximum SM clock (``clocks.max.sm``, ``sm_max_mhz``) is
recorded too but is not the reference: boards often never reach it under load (an
RTX 5070 Ti boosts to ~2.9 of its 3.1 GHz). The paired A/B design keeps throttling
out of the comparison, but the absolute latencies of a throttled measurement are
pessimistic and separate-process numbers noisy.

:meth:`Monitor.processes` lists the other processes with a context on the GPU: one
that computes meanwhile (outside kernel-agent's GPU lock) slows every kernel timed.
"""

from __future__ import annotations

import os
import shutil
import statistics
import subprocess
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
        text = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(self._bus),
                f"--query-compute-apps={_APPS}",
                f"--format={_FORMAT}",
            ],
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
