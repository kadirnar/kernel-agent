"""CUDA graphs with multi-stream capture (#136): one graph holding two independent branches,
captured serially on one stream vs forked onto two streams (fork/join with wait_stream inside
the capture), for (a) DRAM-bound + compute-bound branches, (b) two DRAM-bound branches.

Branch "mem": 28 x [2048 x 2048] bf16 GEMV chain (224 MB of weights, DRAM streaming, like an
LM decode step). Branch "mem2": the same with other weights. Branch "compute": a bf16 GEMM
chain at M = 352 ([352, 1024] x [1024, 4096] x [4096, 1024], 4 times), tensor-core bound like
the batch-16 LocDiT.
"""

from __future__ import annotations

import json
import statistics

import torch

import common
from gemv_ext import ext

e = ext()
L, H = 28, 2048
Wa = (torch.randn(L, H, H, device="cuda") / H**0.5).to(torch.bfloat16)
Wb = (torch.randn(L, H, H, device="cuda") / H**0.5).to(torch.bfloat16)
bufa = torch.zeros(2, H, device="cuda", dtype=torch.bfloat16)
bufb = torch.zeros(2, H, device="cuda", dtype=torch.bfloat16)
x = torch.randn(352, 1024, device="cuda", dtype=torch.bfloat16)
W1 = torch.randn(1024, 4096, device="cuda", dtype=torch.bfloat16) / 32
W2 = torch.randn(4096, 1024, device="cuda", dtype=torch.bfloat16) / 64
outs = {}


def mem(W=Wa, buf=bufa):
    for l in range(L):
        e.gemv(W[l], buf[l & 1], buf[(l + 1) & 1], False)


def mem2():
    mem(Wb, bufb)


def compute():
    h = x
    for _ in range(4):
        h = (h @ W1).relu() @ W2
    outs["c"] = h


def ev(fn, reps=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    return statistics.median(ts)


def capture(fa, fb, forked):
    side = torch.cuda.Stream()
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        fa()
        fb()
    torch.cuda.current_stream().wait_stream(warm)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        if forked:
            cur = torch.cuda.current_stream()
            side.wait_stream(cur)  # fork
            fa()
            with torch.cuda.stream(side):
                fb()
            cur.wait_stream(side)  # join
        else:
            fa()
            fb()
    return g


def eager_two(fa, fb):
    side = torch.cuda.Stream()

    def run():
        cur = torch.cuda.current_stream()
        side.wait_stream(cur)
        fa()
        with torch.cuda.stream(side):
            fb()
        cur.wait_stream(side)

    return run


from kernel_agent.kernels.bench import warm_gpu

warm_gpu(500.0)
res = {"alone_us": {"mem": ev(mem), "mem2": ev(mem2), "compute": ev(compute)}}
for name, (fa, fb) in {"mem+compute": (mem, compute), "mem+mem": (mem, mem2)}.items():
    gs, gf = capture(fa, fb, False), capture(fa, fb, True)
    row = {
        "eager_one_stream_us": ev(lambda: (fa(), fb())),
        "eager_two_streams_us": ev(eager_two(fa, fb)),
        "graph_serial_us": ev(gs.replay),
        "graph_forked_us": ev(gf.replay),
    }
    row["graph_fork_speedup"] = row["graph_serial_us"] / row["graph_forked_us"]
    res[name] = row
    print(name, json.dumps(row), flush=True)
print(json.dumps(res, indent=1))
(common.OUT / "bench_graph_ms.json").write_text(json.dumps(res, indent=1))
