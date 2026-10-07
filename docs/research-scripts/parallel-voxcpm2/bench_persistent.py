"""Persistent kernel vs many launches for a decode GEMV chain (#136).

L = 28 layers (the base LM's depth) of y = W x, W [H, H] bf16, distinct weights per layer
(L x H x H x 2 bytes, larger than the 48 MB L2 for H = 2048: DRAM streaming like the LM
decode). Times per chain (median of 20 CUDA-event timed replays, after warm-up):
eager launches, one CUDA graph, one graph with PDL edges, one persistent cooperative
kernel (grid barrier per layer), the same with the next layer's weights loaded before the
barrier; and the DRAM floor (one GEMV of the whole L*H rows in one launch).
"""

from __future__ import annotations

import json
import statistics
import time

import torch

import common
from gemv_ext import chain_ref, ext

torch.manual_seed(0)
e = ext()
dev = "cuda"
L = 28
results = {}


def ev_time(fn, reps=20, inner=1):
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(inner):
            fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / inner)
    return statistics.median(ts)


def wall_time(fn, reps=20):
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(ts)


from kernel_agent.kernels.bench import warm_gpu

for H in (1024, 2048):
    W = (torch.randn(L, H, H, device=dev) / H**0.5).to(torch.bfloat16)
    x0 = torch.randn(H, device=dev).to(torch.bfloat16)
    buf = torch.zeros(2, H, device=dev, dtype=torch.bfloat16)
    bar = torch.zeros(3, device=dev, dtype=torch.int32)
    ref = chain_ref(W, x0)

    def eager(pdl=False):
        buf[0].copy_(x0)
        for l in range(L):
            e.gemv(W[l], buf[l & 1], buf[(l + 1) & 1], pdl)
        return buf[L & 1]

    def persistent(pf=False, blocks=0):
        buf[0].copy_(x0)
        e.chain_persistent(W, buf, bar, L, pf, blocks)
        return buf[L & 1]

    def err(out):
        return float((out.float() - ref.float()).abs().max() / ref.float().abs().max())

    row = {"bytes_MB": L * H * H * 2 / 1e6}
    row["err_eager"] = err(eager())
    row["err_pdl"] = err(eager(True))
    row["err_persistent"] = err(persistent())
    row["err_persistent_pf"] = err(persistent(True))
    warm_gpu(500.0)
    # graphs (plain and PDL)
    graphs = {}
    for pdl in (False, True):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            eager(pdl)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            eager(pdl)
        graphs[pdl] = g
    graphs[False].replay()
    row["err_graph"] = err(buf[L & 1])
    graphs[True].replay()
    row["err_graph_pdl"] = err(buf[L & 1])
    # DRAM floor: one launch over all L*H rows (x reused)
    Wflat = W.view(L * H, H)
    yflat = torch.empty(L * H, device=dev, dtype=torch.bfloat16)
    row["floor_us"] = ev_time(lambda: e.gemv(Wflat, x0, yflat, False))
    row["floor_GBps"] = row["bytes_MB"] * 1e6 / (row["floor_us"] * 1e-6) / 1e9
    row["eager_wall_us"] = wall_time(eager)
    row["eager_event_us"] = ev_time(eager)
    row["eager_pdl_event_us"] = ev_time(lambda: eager(True))
    row["graph_us"] = ev_time(graphs[False].replay)
    row["graph_pdl_us"] = ev_time(graphs[True].replay)
    row["persistent_us"] = ev_time(persistent)
    row["persistent_pf_us"] = ev_time(lambda: persistent(True))
    for bps in (1, 2):
        row[f"persistent_pf_{bps}bps_us"] = ev_time(lambda: persistent(True, 70 * bps))
    # one layer alone (launch + tail), graph-timed 28x
    one = torch.cuda.CUDAGraph()
    with torch.cuda.graph(one):
        for _ in range(L):
            e.gemv(W[0], buf[0], buf[1], False)
    row["same_layer_x28_graph_us"] = ev_time(one.replay)  # L2-resident weights (2-8 MB)
    results[f"H{H}"] = row
    print(H, json.dumps(row, indent=1), flush=True)
    del W, Wflat

(common.OUT / "bench_persistent.json").write_text(json.dumps(results, indent=1))
