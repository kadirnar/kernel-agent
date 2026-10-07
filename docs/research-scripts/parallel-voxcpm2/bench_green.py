"""Green contexts (SM partitioning) on the RTX 5070 Ti with torch 2.14 / CUDA 13 (#136).

1. Which SM counts a green context can get (split granularity).
2. SM scaling of a DRAM-bound decode GEMV chain (28 x [2048 x 2048] bf16, 224 MB) and of
   VoxCPM2 stages from the optimised set (graphed LocDiT solve, AudioVAE decode): how many
   SMs saturate DRAM, how many the compute-bound VAE needs.
3. Partitioned concurrency: the GEMV chain on (70 - k) SMs while the VAE runs on k SMs, vs
   both on unpartitioned streams, vs one after the other.
4. Cooperative grid-barrier kernels (the FP8 LM step) inside a partition: in a subprocess
   (coop_green.py), since an oversubscribed grid barrier can hang.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time

import torch
from torch.cuda._utils import _cuda_bindings_driver as drv
from torch.cuda.green_contexts import GreenContext

import common
from gemv_ext import ext

HERE = common.HERE
res: dict = {}
torch.cuda.init()
e = ext()

# 4 first (fresh process each, before this process holds a big model)
coop = {}
for k in (16, 35):
    try:
        p = subprocess.run([sys.executable, str(HERE / "coop_green.py"), str(k), "--lm"],
                           capture_output=True, text=True, timeout=180, cwd=HERE)
        coop[k] = [l for l in p.stdout.splitlines() if not l.startswith(("USDT", "  "))
                   and ("SMs" in l or "multiProcessor" in l)]
        coop[k].append(f"exit {p.returncode}")
        if p.returncode:
            coop[k].append(p.stderr.strip().splitlines()[-1][:300] if p.stderr.strip() else "")
    except subprocess.TimeoutExpired:
        coop[k] = ["TIMEOUT after 180 s (hang: grid barrier never released)"]
    print(k, coop[k], flush=True)
res["cooperative_in_green_ctx"] = coop


def sm_count(g):
    _, r = drv.cuGreenCtxGetDevResource(g._green_ctx, drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM)
    return r.sm.smCount


# 1. split granularity
gran = {}
for k in (1, 2, 3, 4, 6, 8, 10, 12, 16, 24, 32, 35, 40, 48, 56, 64, 68, 70):
    try:
        g = GreenContext(num_sms=k)
        gran[k] = sm_count(g)
        del g
    except Exception as exc:
        gran[k] = f"{type(exc).__name__}: {str(exc)[:120]}"
res["requested_vs_granted_sms"] = gran
print("granularity", gran, flush=True)

w = common.load(1, optimized=True, patches=60)
m = w.model
inputs = w.make_inputs()
with torch.inference_mode():
    w.run(inputs)
    w.run(inputs)
captured = {}


def grab(obj, attr, key):
    inner = getattr(obj, attr)
    had = attr in vars(obj)

    def f(*a, **k):
        captured.setdefault(key, ([x.clone() if isinstance(x, torch.Tensor) else x for x in a],
                                  {kk: (v.clone() if isinstance(v, torch.Tensor) else v)
                                   for kk, v in k.items()}))
        return inner(*a, **k)

    setattr(obj, attr, f)
    return lambda: setattr(obj, attr, inner) if had else delattr(obj, attr)


undo = [grab(m.feat_decoder, "forward", "dit"), grab(m.audio_vae, "decode", "vae")]
with torch.inference_mode():
    w.run(inputs)
for u in undo:
    u()

L, H = 28, 2048
W = (torch.randn(L, H, H, device="cuda") / H**0.5).to(torch.bfloat16)
buf = torch.zeros(2, H, device="cuda", dtype=torch.bfloat16)


def gemv_chain():
    for l in range(L):
        e.gemv(W[l], buf[l & 1], buf[(l + 1) & 1], False)


def dit():
    a, k = captured["dit"]
    return m.feat_decoder(*a, **k)


def vae():
    a, k = captured["vae"]
    return m.audio_vae.decode(*a, **k)


jobs = {"gemv_chain": gemv_chain, "dit_solve": dit, "vae_decode": vae}


def run_on(stream, fn, reps):
    """Wall time per call on ``stream`` (stream-synchronised; events may not cross contexts)."""
    with torch.inference_mode(), torch.cuda.stream(stream):
        fn()
    stream.synchronize()
    ts = []
    with torch.inference_mode():
        for _ in range(reps):
            t0 = time.perf_counter()
            with torch.cuda.stream(stream):
                fn()
            stream.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


from kernel_agent.kernels.bench import warm_gpu

warm_gpu(500.0)
plain = torch.cuda.Stream()
scaling = {"all_70_plain_stream": {j: run_on(plain, f, 10) for j, f in jobs.items()}}
print(scaling, flush=True)
ctxs = {}
for k in (8, 16, 24, 32, 40, 48, 56, 64, 70):
    try:
        g = GreenContext(num_sms=k)
        ctxs[k] = g
        s = g.Stream()
        row = {"granted_sms": sm_count(g)}
        for j, f in jobs.items():
            try:
                row[j] = run_on(s, f, 10)
            except Exception as exc:
                row[j] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
                torch.cuda.synchronize()
        scaling[k] = row
        print(k, row, flush=True)
    except Exception as exc:
        scaling[k] = f"{type(exc).__name__}: {str(exc)[:160]}"
        print(k, scaling[k], flush=True)
res["sm_scaling_ms"] = scaling

# 3. partitioned concurrency: gemv chain x n on (70-k) SMs || VAE x 1 on k SMs


def concurrent(sa, fa, na, sb, fb, nb, reps=5):
    walls = []
    with torch.inference_mode():
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            ia = ib = 0
            while ia < na or ib < nb:
                if ia < na and (ib >= nb or ia / na <= ib / nb):
                    with torch.cuda.stream(sa):
                        fa()
                    ia += 1
                else:
                    with torch.cuda.stream(sb):
                        fb()
                    ib += 1
            sa.synchronize()
            sb.synchronize()
            walls.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(walls)


warm_gpu(300.0)
part = {}
NA = 40  # gemv chains (~0.3 ms each)
a0 = scaling["all_70_plain_stream"]["gemv_chain"]
b0 = scaling["all_70_plain_stream"]["vae_decode"]
d0 = scaling["all_70_plain_stream"]["dit_solve"]
s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
part["sequential_sum_ms"] = NA * a0 + b0
part["plain_two_streams_ms"] = concurrent(s1, gemv_chain, NA, s2, vae, 1)
part["plain_one_stream_ms"] = concurrent(s1, gemv_chain, NA, s1, vae, 1)
for k in (8, 16, 24):
    try:
        ga, gb = GreenContext(num_sms=70 - k), GreenContext(num_sms=k)
        part[f"green_{70 - k}+{k}_ms"] = concurrent(ga.Stream(), gemv_chain, NA, gb.Stream(), vae, 1)
    except Exception as exc:
        part[f"green_{70 - k}+{k}_ms"] = f"{type(exc).__name__}: {str(exc)[:160]}"
# DiT solve (DRAM-bound skinny FP8 GEMMs) with the VAE
ND = 6
part["dit_vae_sequential_sum_ms"] = ND * d0 + b0
part["dit_vae_plain_two_streams_ms"] = concurrent(s1, dit, ND, s2, vae, 1)
part["dit_vae_plain_one_stream_ms"] = concurrent(s1, dit, ND, s1, vae, 1)
for k in (8, 16):
    try:
        ga, gb = GreenContext(num_sms=70 - k), GreenContext(num_sms=k)
        part[f"dit_vae_green_{70 - k}+{k}_ms"] = concurrent(ga.Stream(), dit, ND, gb.Stream(), vae, 1)
    except Exception as exc:
        part[f"dit_vae_green_{70 - k}+{k}_ms"] = f"{type(exc).__name__}: {str(exc)[:160]}"
res["partitioned_concurrency"] = part
print(json.dumps(part, indent=1), flush=True)
(common.OUT / "bench_green.json").write_text(json.dumps(res, indent=1, default=str))
