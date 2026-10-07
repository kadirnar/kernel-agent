"""Green contexts, part 2 (#136): are two torch GreenContexts disjoint, and partitions that are
disjoint by construction (one split: k SMs + the remaining SMs).

1. Two torch GreenContext(num_sms=32): the compute-bound VAE decode on both at once vs alone
   (disjoint SMs: ~the time alone; the same SMs: ~twice).
2. Disjoint partitions from cuDevSmResourceSplitByCount (k SMs and the remainder), wrapped as
   torch GreenContexts: GEMV chain x40 on the remainder || VAE on k SMs.
"""

from __future__ import annotations

import json
import statistics
import time

import torch
from torch.cuda._utils import _check_cuda_bindings as chk
from torch.cuda._utils import _cuda_bindings_driver as drv
from torch.cuda.green_contexts import GreenContext

import common
from gemv_ext import ext

torch.cuda.init()
e = ext()
res: dict = {}


def wrap(resource) -> GreenContext:
    dev = chk(drv.cuDeviceGet(0))
    desc = chk(drv.cuDevResourceGenerateDesc([resource], 1))
    g = chk(drv.cuGreenCtxCreate(desc, dev, drv.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM))
    ctx = chk(drv.cuCtxFromGreenCtx(g))
    gc = GreenContext.__new__(GreenContext)
    gc._init_from_cuda_objects(0, g, ctx)
    gc._sms = resource.sm.smCount
    return gc


def disjoint(k: int) -> tuple[GreenContext, GreenContext]:
    dev = chk(drv.cuDeviceGet(0))
    sm = chk(drv.cuDeviceGetDevResource(dev, drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))
    groups, n, rest = chk(drv.cuDevSmResourceSplitByCount(1, sm, 0, k))
    return wrap(groups[0]), wrap(rest)


w = common.load(1, optimized=True, patches=60)
m = w.model
z = torch.randn(1, 64, 240, device="cuda")
L, H = 28, 2048
W = (torch.randn(L, H, H, device="cuda") / H**0.5).to(torch.bfloat16)
buf = torch.zeros(2, H, device="cuda", dtype=torch.bfloat16)


def gemv_chain():
    for l in range(L):
        e.gemv(W[l], buf[l & 1], buf[(l + 1) & 1], False)


def vae():
    return m.audio_vae.decode(z)


def run(jobs, reps=5):
    """jobs: [(stream, fn, n)] issued interleaved; wall time until every stream is done."""
    walls = []
    with torch.inference_mode():
        for s, f, _ in jobs:  # warm-up on each stream
            with torch.cuda.stream(s):
                f()
            s.synchronize()
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            left = [n for _, _, n in jobs]
            total = sum(left)
            while any(left):
                # issue the job that is furthest behind its share
                i = min((j for j in range(len(jobs)) if left[j]),
                        key=lambda j: 1 - left[j] / jobs[j][2])
                with torch.cuda.stream(jobs[i][0]):
                    jobs[i][1]()
                left[i] -= 1
            for s, _, _ in jobs:
                s.synchronize()
            walls.append((time.perf_counter() - t0) * 1e3)
            del total
    return statistics.median(walls)


from kernel_agent.kernels.bench import warm_gpu

warm_gpu(500.0)
plain1, plain2 = torch.cuda.Stream(), torch.cuda.Stream()
a32, b32 = GreenContext(num_sms=32), GreenContext(num_sms=32)
sa, sb = a32.Stream(), b32.Stream()
t1 = run([(sa, vae, 1)])
t2 = run([(sa, vae, 1), (sb, vae, 1)])
p1 = run([(plain1, vae, 1)])
res["torch_two_32sm_contexts"] = {
    "vae_alone_on_A_ms": t1, "vae_on_A_and_B_at_once_ms": t2,
    "vae_alone_plain_ms": p1,
    "verdict": "same SMs" if t2 > 1.6 * t1 else "disjoint SMs" if t2 < 1.25 * t1 else "partial",
}
print(json.dumps(res["torch_two_32sm_contexts"]), flush=True)

NA = 40
base = {"gemv_x40_alone_ms": run([(plain1, gemv_chain, NA)]), "vae_alone_ms": p1}
base["plain_two_streams_ms"] = run([(plain1, gemv_chain, NA), (plain2, vae, 1)])
base["plain_one_stream_ms"] = run([(plain1, gemv_chain, NA), (plain1, vae, 1)])
res["unpartitioned"] = base
print(json.dumps(base), flush=True)
part = {}
for k in (8, 16, 24, 32):
    try:
        gv, gg = disjoint(k)
        sv, sg = gv.Stream(), gg.Stream()
        row = {"vae_sms": gv._sms, "gemv_sms": gg._sms,
               "vae_alone_ms": run([(sv, vae, 1)]),
               "gemv_x40_alone_ms": run([(sg, gemv_chain, NA)]),
               "together_ms": run([(sg, gemv_chain, NA), (sv, vae, 1)])}
        part[k] = row
        print(k, json.dumps(row), flush=True)
    except Exception as exc:
        part[k] = f"{type(exc).__name__}: {str(exc)[:200]}"
        print(k, part[k], flush=True)
res["disjoint_partitions"] = part
(common.OUT / "bench_green2.json").write_text(json.dumps(res, indent=1, default=str))
