"""Burst vs sustained GEMM rate and the SM clock under load (issue #259). GPU, ~15 s.

``roofline.measure_peaks`` times short bursts (~60 ms per shape). On a power-capped board
(T4 70 W, L4 72 W, A10: NVIDIA's sizing guides) the SM clock under sustained tensor-core
load can settle below the burst clock. This measures both here: the bf16 8192^3 GEMM rate
of a 60 ms burst, then of each second of a 10 s run, with the SM clock, power draw and
clock-event reasons (``kernel_agent.telemetry.Monitor``, NVML or nvidia-smi) sampled after
each second. The same script on an older GPU gives its sustained numbers. Run:

    flock ~/.cache/kernel-agent/gpu.lock env KERNEL_AGENT_LOCK_HELD=1 PYTHONPATH=src \
        python docs/research-scripts/older-gpus-259/sustained_clocks.py \
        > docs/research-scripts/older-gpus-259/sustained_clocks.txt
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import torch  # noqa: E402

from kernel_agent.telemetry import Monitor  # noqa: E402

N = 8192
FLOP = 2.0 * N**3


def rate(a: torch.Tensor, b: torch.Tensor, seconds: float) -> float:
    """TFLOP/s of back-to-back GEMMs for about ``seconds``."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    count = 0
    while True:
        for _ in range(4):
            torch.mm(a, b)
        count += 4
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if elapsed >= seconds:
            return FLOP * count / elapsed / 1e12


def main() -> None:
    limit = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.limit,power.max_limit,clocks.max.sm", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"# {torch.cuda.get_device_name()}; power.limit, power.max_limit, clocks.max.sm: {limit}")
    a = torch.randn(N, N, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, N, device="cuda", dtype=torch.bfloat16)
    monitor = Monitor()
    rate(a, b, 0.2)  # warm up (cuBLAS heuristics, clocks)
    time.sleep(2.0)  # let the board cool back to its idle state between the two
    burst = rate(a, b, 0.06)
    sample = monitor.sample("burst")
    print(f"burst 60 ms: {burst:.1f} TFLOP/s; {sample}")
    for second in range(10):
        r = rate(a, b, 1.0)
        sample = monitor.sample(f"s{second + 1}")
        print(f"second {second + 1:2d}: {r:.1f} TFLOP/s; {sample}")


if __name__ == "__main__":
    main()
