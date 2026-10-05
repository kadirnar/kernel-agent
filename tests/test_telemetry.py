"""GPU telemetry around timed runs (issue #59): only a real slowdown is throttling.

CPU only: ``nvidia-smi`` is replaced by recorded output lines.
"""

import subprocess

from kernel_agent import telemetry

# `nvidia-smi --query-gpu=<telemetry._QUERY> --format=csv,noheader,nounits` on an RTX 5070
# Ti (boost ~2.9 GHz of a 3.1 GHz maximum) during a bf16 GEMM loop at its 300 W limit:
# sw_power_cap (0x4) at full boost.
FULL_BOOST = [
    "2902, 3105, 13801, 59, 80.33, 0x0000000000000004",
    "2902, 3105, 13801, 63, 306.70, 0x0000000000000004",
    "2902, 3105, 13801, 65, 299.36, 0x0000000000000004",
    "2917, 3105, 13801, 65, 299.99, 0x0000000000000004",
]
#: The same GPU idle (before a timed run): clocks down, no slowdown reason.
IDLE = "870, 3105, 810, 40, 26.06, 0x0000000000000400"
#: A hardware slowdown (0x8) holding the SM clock at 2100 MHz.
HW_SLOWDOWN = "2100, 3105, 13801, 91, 285.10, 0x0000000000000008"
#: The SM clock down to 2400 MHz with no reason but the power cap.
POWER_CAP_DROP = "2400, 3105, 13801, 70, 300.02, 0x0000000000000004"


def monitor(monkeypatch, lines: list[str]) -> telemetry.Monitor:
    """A Monitor whose ``nvidia-smi`` prints ``lines``, one per sample."""
    out = iter(lines)

    def run(cmd, **kwargs):
        assert cmd[0] == "nvidia-smi" and f"--query-gpu={telemetry._QUERY}" in cmd
        return subprocess.CompletedProcess(cmd, 0, next(out) + "\n", "")

    monkeypatch.setattr(telemetry.subprocess, "run", run)
    mon = telemetry.Monitor(bus_id="00000000:0B:00.0")
    mon.backend = "nvidia-smi"
    return mon


def test_sw_power_cap_at_full_boost_is_not_throttling(monkeypatch):
    mon = monitor(monkeypatch, [IDLE, *FULL_BOOST])
    first = mon.sample("before", loaded=False)  # idle clocks are not a drop
    assert first is not None and first["loaded"] is False and first["reasons"] == []
    for _ in FULL_BOOST:
        mon.sample("B")
    gpu = mon.summary()
    assert gpu["reasons"] == ["sw_power_cap"]  # recorded, informational
    assert gpu["throttle"] == [] and gpu["throttled_samples"] == 0 and not gpu["throttled"]
    assert gpu["sm_mhz"] == [2902, 2902, 2917] and gpu["sm_max_mhz"] == 3105
    assert gpu["clock_drop_below_mhz"] == 2625  # 90 % of the highest clock under load
    assert gpu["temp_c_max"] == 65 and gpu["power_w_max"] == 306.7
    assert telemetry.warning(gpu) is None


def test_hw_slowdown_at_2100_mhz_is_throttling(monkeypatch):
    mon = monitor(monkeypatch, [HW_SLOWDOWN] * 3)  # the whole measurement: by its reason
    for _ in range(3):
        mon.sample()
    gpu = mon.summary()
    assert gpu["throttle"] == ["hw_slowdown"] and gpu["throttled_samples"] == 3
    message = telemetry.warning(gpu)
    assert message == (
        "GPU throttled (hw_slowdown) in 3/3 samples: SM clock 2100-2100 MHz, up to 91 °C "
        "and 285.1 W"
    )

    mon = monitor(monkeypatch, [*FULL_BOOST[:2], HW_SLOWDOWN])  # during it: a drop too
    for _ in range(3):
        mon.sample()
    gpu = mon.summary()
    assert gpu["throttle"] == ["clock_drop", "hw_slowdown"] and gpu["throttled_samples"] == 1
    message = telemetry.warning(gpu)
    assert message is not None
    assert message.startswith("GPU throttled (SM clock below 2612 MHz, hw_slowdown) in 1/3")


def test_a_clock_drop_under_the_power_cap_is_throttling(monkeypatch):
    mon = monitor(monkeypatch, [*FULL_BOOST, POWER_CAP_DROP])
    for _ in range(5):
        mon.sample()
    gpu = mon.summary()
    assert gpu["throttle"] == ["clock_drop"] and gpu["throttled_samples"] == 1
    message = telemetry.warning(gpu)
    assert message is not None and "SM clock below 2625 MHz) in 1/5 samples" in message


def test_no_samples_and_an_unreadable_maximum(monkeypatch):
    assert telemetry.Monitor(bus_id="0").summary() == {}
    assert telemetry.warning({}) is None and telemetry.warning(None) is None
    mon = monitor(monkeypatch, ["2902, [N/A], 13801, 59, 80.33, 0x4"])
    sample = mon.sample()
    assert sample is not None and sample["sm_max_mhz"] is None
    assert "sm_max_mhz" not in mon.summary()
