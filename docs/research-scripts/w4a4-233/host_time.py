"""Eager (host-bound) time per call at 704 x 1024 -> 8192 (#233): bf16 against the two W4A4
examples and the FP8 W8A8 Triton one, as the evaluator times them.

    python host_time.py   (results/host_time.out)
"""

import importlib.util
import time
from pathlib import Path

import torch

base = str(Path(__file__).resolve().parents[3] / "src/kernel_agent/agent/examples") + "/"


def load(name):
    spec = importlib.util.spec_from_file_location(name, base + name + ".py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def eager_us(fn, iters=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e6


torch.manual_seed(0)
lin = torch.nn.Linear(1024, 8192, bias=False, device="cuda", dtype=torch.bfloat16)
x = torch.randn(64, 11, 1024, device="cuda", dtype=torch.bfloat16)
with torch.no_grad():
    print("bf16 eager", eager_us(lambda: lin(x)))
    for name in ("triton_nvfp4_w4a4_gemm", "cute_nvfp4_w4a4_gemm", "triton_fp8_w8a8_gemm"):
        try:
            mod = load(name).build(lin)
            print(name, "eager", eager_us(lambda m=mod: m(x)))
        except Exception as exc:  # noqa: BLE001
            print(name, "failed", exc)
