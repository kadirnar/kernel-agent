"""Cooperative (grid-barrier) kernels inside a green context of k SMs: does the launch size
the grid by the device's 70 SMs (oversubscribed -> error or deadlock) or by the partition?
Runs in its own process (the parent kills it on a hang).

    python coop_green.py <num_sms> [--lm]
"""

from __future__ import annotations

import sys
import time

import torch
from torch.cuda._utils import _cuda_bindings_driver as drv
from torch.cuda._utils import _cuda_bindings_runtime as rt
from torch.cuda.green_contexts import GreenContext

from gemv_ext import ext

k = int(sys.argv[1])
torch.cuda.init()
e = ext()
g = GreenContext(num_sms=k)
res, = drv.cuGreenCtxGetDevResource(g._green_ctx, drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM)[1:]
print("green ctx SMs:", res.sm.smCount, flush=True)
# what does a kernel's host code see as the SM count with the green context current?
_, before = rt.cudaDeviceGetAttribute(rt.cudaDeviceAttr.cudaDevAttrMultiProcessorCount, 0)
drv.cuCtxPushCurrent(g._context)
_, inside = rt.cudaDeviceGetAttribute(rt.cudaDeviceAttr.cudaDevAttrMultiProcessorCount, 0)
drv.cuCtxPopCurrent()
print(f"multiProcessorCount: primary ctx {before}, green ctx current {inside}", flush=True)
s = g.Stream()
L, H = 28, 2048
W = (torch.randn(L, H, H, device="cuda") / H**0.5).to(torch.bfloat16)
buf = torch.zeros(2, H, device="cuda", dtype=torch.bfloat16)
bar = torch.zeros(3, device="cuda", dtype=torch.int32)
torch.cuda.synchronize()
rejected = False
for blocks in (k, 0):  # k blocks (fits the partition), 0 = per-SM occupancy x 70 SMs
    try:
        bar.zero_()
        torch.cuda.synchronize()
        with torch.cuda.stream(s):
            t0 = time.perf_counter()
            used = e.chain_persistent(W, buf, bar, L, True, blocks)
        s.synchronize()
        hung = int(bar[2].item())
        print(f"persistent chain, {used} blocks on {k} SMs: "
              f"{'BARRIER TIMEOUT (grid not co-resident)' if hung else 'ok'}, "
              f"{(time.perf_counter() - t0) * 1e3:.3f} ms", flush=True)
    except Exception as exc:
        rejected = rejected or blocks == 0
        print(f"persistent chain, blocks={blocks or 'max'} on {k} SMs: {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:200]}", flush=True)
        torch.cuda.synchronize()
if "--lm" in sys.argv and not rejected:
    print("FP8 LM step not run: an oversubscribed cooperative launch was accepted, the LM "
          "kernel's unbounded grid barrier would hang", flush=True)
if "--lm" in sys.argv and rejected:
    import common

    w = common.load(1, optimized=True, patches=4)
    m = w.model
    x = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
    pos = torch.tensor([3], device="cuda")
    with torch.inference_mode():
        m.base_lm.forward_step(x, pos)
        torch.cuda.synchronize()
        try:
            with torch.cuda.stream(s):
                t0 = time.perf_counter()
                m.base_lm.forward_step(x, pos)
            s.synchronize()
            print(f"FP8 LM step (cooperative kernels) on {k} SMs: ok "
                  f"{(time.perf_counter() - t0) * 1e3:.3f} ms", flush=True)
        except Exception as exc:
            print(f"FP8 LM step on {k} SMs: {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:200]}", flush=True)
