"""Profile one VoxCPM2 generation per stage (#136).

    python prof.py --batch 1 --patches 60 [--optimized] --tag b1_opt
"""

from __future__ import annotations

import argparse
import json

import torch

import common

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--patches", type=int, default=60)
ap.add_argument("--optimized", action="store_true")
ap.add_argument("--tag", required=True)
ap.add_argument("--iters", type=int, default=3)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--async-stop", action="store_true")
ns = ap.parse_args()

w = common.load(ns.batch, ns.optimized, ns.patches)
inputs = w.make_inputs()
if ns.async_stop:
    w._generate_batch = common.async_stop_variant(w)
with torch.inference_mode():
    for _ in range(ns.warmup):
        w.run(inputs)
torch.cuda.synchronize()
times = common.timed(w, inputs, ns.iters) if ns.iters else []
print(f"{ns.tag}: unprofiled run {times} ms", flush=True)
common.tag_stages(w)
with torch.inference_mode():
    w.run(inputs)  # warm-up with the ranges installed
from kernel_agent.kernels.bench import warm_gpu

warm_gpu(500.0)
data = common.profile_run(w, inputs, ns.tag)
(common.OUT / f"{ns.tag}.times.json").write_text(
    json.dumps({"times_ms": times, "profiled_wall_ms": data["wall_ms"],
                "peak_gb": torch.cuda.max_memory_allocated() / 2**30})
)
