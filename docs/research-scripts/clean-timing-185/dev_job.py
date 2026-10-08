"""A simulated agent's own GPU run: a few seconds of GEMMs and small launches (a correctness
check or a microbenchmark of its kernel)."""

import sys
import time

import torch

seconds = float(sys.argv[1])
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1 << 16, device="cuda")
start = time.perf_counter()
n = 0
while time.perf_counter() - start < seconds:
    for _ in range(4):
        y = x @ x
    for _ in range(200):
        v = v * 1.0001 + 1e-6
    torch.cuda.synchronize()
    n += 1
print(f"dev run: {n} rounds in {time.perf_counter() - start:.1f}s", float(y[0, 0]), flush=True)
