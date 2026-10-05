"""GPU clocks, temperature and power around timed end-to-end runs.

Read through NVML (``pynvml``, from the ``nvidia-ml-py`` package) when it is
importable, else through ``nvidia-smi``; neither is a dependency, and without
both there are no samples. A clock-event reason that lowers the clocks (power
cap, thermal or hardware slowdown) during a measurement is a warning: the
paired A/B design keeps such drift out of the comparison, but the absolute
latencies of that session are pessimistic and separate-process numbers noisy.
"""

from __future__ import annotations

import shutil
import statistics
import subprocess
import time
from typing import Any

#: NVML clock-event reasons that slow the GPU down (``nvmlClocksEventReason*``).
THROTTLE = {
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x20: "sw_thermal",
    0x40: "hw_thermal",
    0x80: "hw_power_brake",
}
_QUERY = "clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_event_reasons.active"
_FORMAT = "csv,noheader,nounits"


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
    return [name for bit, name in THROTTLE.items() if mask & bit]


class Monitor:
    """Samples of one GPU; :meth:`summary` for results, :meth:`warning` for logs."""

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
        sm, mem, temp, power, mask = (v.strip() for v in out.strip().splitlines()[0].split(","))
        return {
            "sm_mhz": int(float(sm)),
            "mem_mhz": int(float(mem)),
            "temp_c": int(float(temp)),
            "power_w": round(float(power), 1),
            "mask": int(mask, 16),
        }

    def sample(self, label: str = "") -> dict[str, Any] | None:
        """One sample (None: no backend, or it failed and is now off)."""
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
            "throttle": reasons(mask),
        }
        self.samples.append(sample)
        return sample

    def summary(self) -> dict[str, Any]:
        """Ranges over the samples (``{}``: none) and whether any was throttled."""
        if not self.samples:
            return {}
        sm = [s["sm_mhz"] for s in self.samples]
        mem = [s["mem_mhz"] for s in self.samples]
        seen = sorted({r for s in self.samples for r in s["throttle"]})
        return {
            "backend": self.backend or "off",
            "samples": len(self.samples),
            "sm_mhz": [min(sm), round(statistics.median(sm)), max(sm)],
            "mem_mhz": [min(mem), max(mem)],
            "temp_c_max": max(s["temp_c"] for s in self.samples),
            "power_w_max": max(s["power_w"] for s in self.samples),
            "throttle": seen,
            "throttled_samples": sum(bool(s["throttle"]) for s in self.samples),
            "throttled": bool(seen),
        }


def warning(summary: dict[str, Any] | None) -> str | None:
    """Log line for a throttled measurement (None: not throttled or no samples)."""
    if not summary or not summary.get("throttled"):
        return None
    lo, _, hi = summary["sm_mhz"]
    return (
        f"GPU throttled ({', '.join(summary['throttle'])}) in {summary['throttled_samples']}/"
        f"{summary['samples']} samples: SM clock {lo}-{hi} MHz, up to {summary['temp_c_max']} "
        f"°C and {summary['power_w_max']} W"
    )
