"""Peaks on a power-capped board (#253): the sustained 16-bit matmul peak next to the burst
one, the clocks and power it ran at, and the ceilings' floors on it. CPU tests with a fake
monitor, fake GEMM batches and a simulated clock (deterministic); one GPU test."""

from __future__ import annotations

import subprocess

import pytest
from fake_clock import Clock
from test_gpu_skus import SKUS, fake, peaks_of

from kernel_agent import gpu_arch, telemetry, toolchain
from kernel_agent.kernels import roofline
from kernel_agent.profiling import ceilings

BOOST_MHZ = 1695  # the faked boards' boost clock (the A10's, datasheet)
#: Enforced power limits of the power-capped boards (W, datasheets)
LIMIT_W = {"A10": 150.0, "T4": 70.0, "L4": 72.0}


class PowerCapped:
    """:class:`telemetry.Monitor`'s interface on a board at its power limit: the first
    ``burst`` samples at the boost clock, then ``capped_mhz`` with ``sw_power_cap``."""

    def __init__(self, capped_mhz: int = 1320, burst: int = 3, limit_w: float = 150.0) -> None:
        self.samples: list[dict] = []
        self.capped_mhz, self.burst, self.limit_w = capped_mhz, burst, limit_w

    def sample(self, label: str = "", *, loaded: bool = True) -> dict:
        capped = len(self.samples) >= self.burst
        sample = {
            "label": label,
            "sm_mhz": self.capped_mhz if capped else BOOST_MHZ,
            "sm_max_mhz": BOOST_MHZ,
            "mem_mhz": 6251,
            "temp_c": 71,
            "power_w": self.limit_w - 0.5 if capped else 120.0,
            "reasons": ["sw_power_cap"] if capped else [],
        }
        self.samples.append(sample)
        return sample

    def power_limit_w(self) -> float:
        return self.limit_w


class Batches:
    """GEMM batches whose rate falls from ``burst`` to ``sustained`` TFLOP/s after ``fast``
    batches; ``log`` records the order of launches and waits."""

    def __init__(self, burst: float, sustained: float, fast: int = 4) -> None:
        self.burst, self.sustained, self.fast = burst, sustained, fast
        self.launched = 0
        self.log: list[tuple[str, int]] = []

    def launch(self) -> int:
        self.launched += 1
        self.log.append(("launch", self.launched))
        return self.launched

    def wait(self, batch: int) -> float:
        self.log.append(("wait", batch))
        return self.burst if batch <= self.fast else self.sustained


def measure_with(rates: dict[str, tuple[float, float]]):
    """``measure`` for :func:`roofline.sustained_peaks`: :func:`roofline.sustain` over fake
    batches on a simulated clock (0.125 s per reading)."""

    def measure(name: str, monitor):
        batches = Batches(*rates[name])
        return roofline.sustain(batches.launch, batches.wait, 2.0, monitor, Clock(0.125).monotonic)

    return measure


def test_sustain_keeps_the_gpu_busy_and_takes_the_settled_half():
    batches, monitor = Batches(125.0, 100.0), PowerCapped()
    rate, samples = roofline.sustain(
        batches.launch, batches.wait, 2.0, monitor, Clock(0.125).monotonic
    )
    assert rate == 100.0  # the settled half, not the burst at the start
    assert batches.launched == 16 and len(monitor.samples) == 16  # 2 s of 0.125 s readings
    assert samples == monitor.samples[8:] and all(s["reasons"] for s in samples)
    # every batch but the last is waited for only once the next one is queued
    for i in range(1, batches.launched):
        assert batches.log.index(("launch", i + 1)) < batches.log.index(("wait", i))
    # without a monitor: the rate alone
    assert roofline.sustain(Batches(90.0, 90.0).launch, lambda b: 90.0, 0.0) == (90.0, [])


@pytest.mark.parametrize("sku", ["A10", "T4", "L4"])
def test_a_power_capped_board_records_and_uses_its_sustained_peak(sku):
    burst = SKUS[sku][1]["tflops"]
    peaks = peaks_of(sku)
    limit = LIMIT_W[sku]
    monitor = PowerCapped(limit_w=limit)
    for name in ("bfloat16", "float16", "float32"):  # measure_peaks samples after each burst
        monitor.sample(f"burst {name}")
    rates = {k: (burst[k], round(0.8 * burst[k], 1)) for k in ("bfloat16", "float16")}
    roofline.sustained_peaks(peaks, monitor, measure_with(rates))
    assert peaks["tflops_sustained"] == {k: r for k, (_, r) in rates.items()}  # 20 % lower
    assert peaks["sustained"] == {
        "seconds": roofline.SUSTAINED_S,
        "sm_mhz": 1320,
        "power_w": limit - 0.5,
        "reasons": ["sw_power_cap"],
        "burst_sm_mhz": BOOST_MHZ,
        "sm_max_mhz": BOOST_MHZ,
        "power_limit_w": limit,
    }
    assert roofline.sustained_drops(peaks) == rates
    assert roofline.floor_tflops(peaks)["bfloat16"] == rates["bfloat16"][1]
    assert roofline.floor_tflops(peaks)["float32"] == burst["float32"]  # no ratio assumed

    # the ceilings' floors run at the sustained rate (and so does the scheduler's floor_ms)
    flops = 2 * 352 * 8192 * 4096 * 100
    profile = _profile(flops)
    table = ceilings.build(profile, peaks, 2000.0)
    want = flops / (rates["bfloat16"][1] * 1e9)
    assert table["rows"][0]["floors"]["exact"] == pytest.approx(want, rel=1e-3)
    floor = ceilings.floor_ms(table["rows"], ceilings.PRECISIONS["exact"], table["peaks"])
    assert floor == pytest.approx(want, rel=1e-3)
    assert "*Sustained*: the compute floors use this GPU's sustained matmul peaks" in (
        ceilings.markdown(table)
    )
    # the toolchain block (doctor, every prompt) shows both peaks, the clocks and the cap
    gpu = toolchain.GPUInfo(*SKUS[sku][0])
    tc = toolchain.Toolchain(gpu, "2.14", "13.0", None, "13.0", {}, [], {}, peaks)
    text = tc.summary()
    assert f"sustained bf16/fp16 {rates['bfloat16'][1]:.0f} / {rates['float16'][1]:.0f}" in text
    load = f"(2 s; SM 1320 MHz, burst {BOOST_MHZ}; {limit:.0f} W of a {limit:.0f} W limit; "
    assert load + "sw_power_cap)" in text
    assert "sustained load slows this GPU down" in text and f"{limit:.0f} W limit" in text


def _profile(flops: int) -> dict:
    work = {
        "group": "model.layers.*.mlp",
        "phase": "prefill",
        "instances": 1,
        "calls": 100,
        "inclusive_ms": 900.0,
        "flops": {"bfloat16": flops},
        "weight_elems": 8192 * 4096,
        "weight_bytes": 2 * 8192 * 4096,
        "io_bytes": 0,
        "signature": "",
    }
    return {"hooked_wall_ms": 2000.0, "classes": [{"cls": "Mlp", "is_leaf": False, "work": [work]}]}


def test_a_board_that_holds_its_clock_keeps_the_burst_peaks():
    peaks = peaks_of("A10")
    monitor = PowerCapped(capped_mhz=BOOST_MHZ - 15)
    roofline.sustained_peaks(
        peaks, monitor, measure_with({k: (125.0, 119.0) for k in ("bfloat16", "float16")})
    )
    assert peaks["tflops_sustained"] == {"bfloat16": 119.0, "float16": 119.0}  # 5 % lower
    assert roofline.sustained_drops(peaks) == {}
    assert roofline.floor_tflops(peaks)["bfloat16"] == 125.0
    flops = 2 * 352 * 8192 * 4096 * 100
    table = ceilings.build(_profile(flops), peaks, 2000.0)
    assert table["rows"][0]["floors"]["exact"] == pytest.approx(flops / 125e9, rel=1e-3)
    assert "*Sustained*" not in ceilings.markdown(table)
    text = fake(
        "A10", tflops_sustained=peaks["tflops_sustained"], sustained=peaks["sustained"]
    ).summary()
    assert "sustained bf16/fp16 119 / 119 TFLOP/s" in text  # doctor prints it anyway
    assert "sustained load slows this GPU down" not in text
    assert gpu_arch.sustained_line(peaks) is None


def test_peaks_without_a_sustained_measurement_are_as_before():
    peaks = peaks_of("A10")
    assert roofline.sustained_drops(peaks) == {} and roofline.sustained_drops(None) == {}
    assert roofline.floor_tflops(peaks) == peaks["tflops"]
    assert "sustained" not in toolchain.format_peaks(peaks)
    # no 16-bit burst peak, or the measurement failed: nothing recorded
    empty = {"tflops": {"float32": 30.0}}
    assert roofline.sustained_peaks(empty, PowerCapped(), measure_with({})) == empty
    failed = peaks_of("A10")
    roofline.sustained_peaks(failed, None, lambda name, mon: None)
    assert "tflops_sustained" not in failed and "sustained" not in failed


def test_power_limit_from_nvidia_smi_and_nvml(monkeypatch):
    answers = iter(["150.00\n", "[N/A]\n"])

    def run(cmd, **kwargs):
        assert cmd[0] == "nvidia-smi" and "--query-gpu=enforced.power.limit" in cmd
        return subprocess.CompletedProcess(cmd, 0, next(answers), "")

    monkeypatch.setattr(telemetry.subprocess, "run", run)
    mon = telemetry.Monitor(bus_id="00000000:0B:00.0")
    mon.backend = "nvidia-smi"
    assert mon.power_limit_w() == 150.0
    assert mon.power_limit_w() is None  # [N/A]: unknown, never a failure

    class Nvml:
        @staticmethod
        def nvmlDeviceGetEnforcedPowerLimit(handle):
            return 70000  # mW: a T4

    mon.backend, mon._nvml = "nvml", Nvml()
    assert mon.power_limit_w() == 70.0
    mon.backend = None
    assert mon.power_limit_w() is None


@pytest.mark.gpu
def test_sustained_peak_on_this_gpu():
    """The sustained measurement on the real GPU: a rate, and the clocks and power when NVML
    or nvidia-smi is there (an RTX 5070 Ti reports ``sw_power_cap`` routinely)."""
    import torch

    monitor = telemetry.Monitor()
    free, _ = torch.cuda.mem_get_info()
    found = roofline._sustained_tflops(torch.bfloat16, free, monitor, seconds=0.5)
    assert found is not None
    rate, samples = found
    assert rate > 1.0
    if monitor.backend is not None:
        assert samples and all(s["sm_mhz"] > 0 for s in samples)
    peaks = {"tflops": {"bfloat16": rate * 1.5}}
    roofline.sustained_peaks(peaks, monitor, lambda name, mon: found)
    assert peaks["tflops_sustained"]["bfloat16"] == rate
    assert roofline.sustained_drops(peaks)  # a third below the (made-up) burst: used
