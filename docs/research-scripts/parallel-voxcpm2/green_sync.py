"""Does torch.cuda.synchronize() (cudaDeviceSynchronize in the primary context) wait for work on
a green-context stream? And do CUDA events record/wait across it?"""

import time

import torch
from torch.cuda.green_contexts import GreenContext

from gemv_ext import ext

torch.cuda.init()
e = ext()
L, H = 28, 2048
W = (torch.randn(L, H, H, device="cuda") / H**0.5).to(torch.bfloat16)
buf = torch.zeros(2, H, device="cuda", dtype=torch.bfloat16)
g = GreenContext(num_sms=8)
s = g.Stream()


def chain(n):
    for _ in range(n):
        for l in range(L):
            e.gemv(W[l], buf[l & 1], buf[(l + 1) & 1], False)


with torch.cuda.stream(s):
    chain(2)
s.synchronize()
for trial in range(3):
    t0 = time.perf_counter()
    with torch.cuda.stream(s):
        chain(300)  # ~150 ms on 8 SMs
    t_issue = time.perf_counter() - t0
    torch.cuda.synchronize()
    t_dev = time.perf_counter() - t0
    pending = not s.query()
    s.synchronize()
    t_stream = time.perf_counter() - t0
    print(f"issue {t_issue * 1e3:.1f} ms, torch.cuda.synchronize returned at {t_dev * 1e3:.1f} ms "
          f"(work still pending: {pending}), stream done at {t_stream * 1e3:.1f} ms", flush=True)
# events across contexts
a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
try:
    a.record(s)
    with torch.cuda.stream(s):
        chain(10)
    b.record(s)
    b.synchronize()
    print(f"events on a green-context stream: {a.elapsed_time(b):.2f} ms", flush=True)
    torch.cuda.current_stream().wait_event(b)
    print("primary stream can wait on an event recorded on the green stream", flush=True)
except Exception as exc:
    print("events:", type(exc).__name__, exc)
