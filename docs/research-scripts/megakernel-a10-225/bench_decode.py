"""The decode-step megakernel against its graph baseline on this GPU (issue #225, follow-up 1:
split-KV attention and on-device token advance).

The example's ``TinyDecoder`` (``agent/examples/native_megakernel``: a pre-norm GQA decoder,
bf16, random weights of a trained model's scale) as ``build_decode(mode=...)``: the
megakernel (one launch of ``kernel<DecodeOps>`` per step) and the graph of one launch per op
(the same opcodes and tiles, kernel boundaries instead of counters). For each KV length of
``--lengths``: first one step of each engine from the same state against the torch reference
(logits max |diff|, the argmax, megakernel == graph bit for bit), then each engine timed as
the GPU time of ``--steps`` consecutive steps (graph replays back to back between two events,
each advancing the position on the device: the lengths ``L - steps + 1 .. L``), ``--trials``
times, the engines in turn within a trial. The DRAM floor is the bytes a step must read (every
weight but the embedding table, one embedding row, the K / V cache up to the length) over the
GPU's copy bandwidth (``roofline._copy_gbps``, as ``doctor``'s peaks measure it). Then the
megakernel's trace (``KA_MK_TRACE``) per op at the shortest and the longest length. Run (GPU,
~1 min):

    flock ~/.cache/kernel-agent/gpu.lock env KERNEL_AGENT_LOCK_HELD=1 PYTHONPATH=src \\
        python docs/research-scripts/megakernel-a10-225/bench_decode.py \\
        > docs/research-scripts/megakernel-a10-225/bench_decode.txt
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import torch  # noqa: E402

from kernel_agent import toolchain  # noqa: E402
from kernel_agent.agent.prompts import EXAMPLES_DIR  # noqa: E402
from kernel_agent.kernels import bench, roofline  # noqa: E402
from kernel_agent.native import project  # noqa: E402

MiB = 1 << 20


def copy_bandwidth() -> float:
    free, _ = torch.cuda.mem_get_info()
    n = int(min(512 * MiB, free // 8)) // 16 * 16
    src = torch.ones(n // 2, device="cuda", dtype=torch.bfloat16)
    dst = torch.empty_like(src)
    gbps = roofline._copy_gbps(src, dst)
    del src, dst
    torch.cuda.empty_cache()
    return gbps


def step_bytes(ref, length: int) -> int:
    """What one step must read: every weight but the embedding table, one embedding row, the
    K and V caches up to ``length``."""
    weights = sum(p.numel() * p.element_size() for p in ref.parameters())
    table = ref.embed.weight
    weights += -table.numel() * table.element_size() + ref.hidden * table.element_size()
    kv = 2 * len(ref.layers) * ref.kv_heads * length * ref.head_dim * table.element_size()
    return weights + kv


def start_of(length: int, steps: int, capacity: int) -> int:
    """The position a run of ``steps`` steps ending at KV length ``length`` starts at."""
    return max(0, min(length - steps, capacity - steps))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--vocab", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--inter", type=int, default=2048)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--lengths", type=int, nargs="+", default=[16, 128, 1024, 4096])
    parser.add_argument("--splits", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--inflight", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--no-trace", action="store_true")
    ns = parser.parse_args()

    tc = toolchain.setup()
    gpu = tc.gpu
    print(f"GPU: {gpu.name} ({gpu.arch}), {gpu.sm_count} SMs, L2 {gpu.l2_cache_mb:.0f} MB")
    print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), nvcc {tc.nvcc_version}")
    mod = project.import_project(EXAMPLES_DIR / "native_megakernel")
    torch.manual_seed(0)
    ref = mod.TinyDecoder(
        vocab=ns.vocab, hidden=ns.hidden, layers=ns.layers, heads=ns.heads,
        kv_heads=ns.kv_heads, head_dim=ns.head_dim, inter=ns.inter, capacity=ns.capacity,
    )  # fmt: skip
    ref = ref.cuda().to(torch.bfloat16).eval().randomize_(1)
    engines = {}
    for inflight in ns.inflight:
        engines[f"megakernel (inflight {inflight})"] = mod.build_decode(
            ref, "megakernel", ns.splits, ns.chunk, inflight=inflight
        )
    graph = mod.build_decode(ref, "graph", ns.splits, ns.chunk)
    engines[f"graph ({len(graph.runtimes)} launches per step)"] = graph
    mk = next(iter(engines.values()))
    weights = step_bytes(ref, 0)
    print(f"date {time.strftime('%Y-%m-%d %H:%M')}; decoder: {ns.layers} layers, hidden "
          f"{ns.hidden}, {ns.heads} q / {ns.kv_heads} kv heads of {ns.head_dim}, inter "
          f"{ns.inter}, vocab {ns.vocab}, KV capacity {ns.capacity}; {weights / 1e6:.1f} MB of "
          f"weights per step")
    print(f"megakernel: {mk.pages} pages, {mk.queues} queues; {mk.schedule.describe()}")
    print(f"split-KV: {mk.split.splits} splits per KV head (chunk "
          f"{mk.split.chunk or 'balanced over the splits'}), {mk.split.attention_tiles} "
          f"attention tiles per layer")
    bench.warm_gpu()
    copy = copy_bandwidth()
    print(f"DRAM: copy {copy:.1f} GB/s; {ns.trials} trials of {ns.steps} steps per engine "
          "(per-step us: min / median)")
    for length in ns.lengths:
        kc, vc = ref.new_cache()
        kc.normal_()
        vc.normal_()
        pos = length - 1
        rk, rv = kc.clone(), vc.clone()
        want = ref.step(5, pos, rk, rv).float()
        outs = {}
        for name, engine in engines.items():
            engine.reset(5, pos, (kc, vc))
            engine.step()
            torch.cuda.synchronize()
            outs[name] = (engine.logits.clone(), int(engine.best))
        first = next(iter(outs.values()))
        same = all(torch.equal(first[0], o[0]) for o in outs.values())
        err = (first[0].float() - want).abs().max().item()
        top = int(want.argmax())
        start = start_of(length, ns.steps, ns.capacity)
        print(f"\nKV length {length} (timed: lengths {start + 1}..{start + ns.steps}): logits "
              f"max |diff| vs torch {err:.3g}, argmax {first[1]} (torch {top}), every engine "
              f"bit for bit the same: {same}")
        times: dict[str, list[float]] = {name: [] for name in engines}
        for _ in range(ns.trials):
            for name, engine in engines.items():
                engine.reset(5, start)
                torch.cuda.synchronize()
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(ns.steps):
                    engine._graph.replay()
                end.record()
                end.synchronize()
                times[name].append(begin.elapsed_time(end) / ns.steps * 1e3)
                engine.position = start + ns.steps
                engine.check()
        floor = step_bytes(ref, start + ns.steps // 2) / copy / 1e3
        print(f"  {'DRAM floor (copy bandwidth)':44s} {floor:7.1f} us")
        for name, ts in times.items():
            med = statistics.median(ts)
            print(f"  {name:44s} {min(ts):7.1f} / {med:7.1f} us ({floor / med:.0%} of floor)")
    if not ns.no_trace:
        engine = mod.build_decode(ref, "megakernel", ns.splits, ns.chunk, trace=True)
        for length in (min(ns.lengths), max(ns.lengths)):
            start = start_of(length, ns.steps, ns.capacity)
            engine.reset(5, start)
            for _ in range(ns.steps):
                engine.step()
            torch.cuda.synchronize()
            summary = engine.trace_summary()
            print(f"\ntrace at KV length {start + ns.steps} (the last step; per op: mean wait for "
                  f"its counters / weights landing after them / run, us; tiles): span "
                  f"{summary['span_ns'] / 1e3:.1f} us")
            per: dict[str, list] = {}
            for op, s in summary["ops"].items():
                kind = op.split(".")[-1]
                per.setdefault(kind, []).append(s)
            for kind, stats in per.items():
                n = sum(s["n"] for s in stats)
                w = statistics.mean(s["wait_ns"] for s in stats) / 1e3
                land = statistics.mean(s["land_ns"] for s in stats) / 1e3
                run = statistics.mean(s["run_ns"] for s in stats) / 1e3
                print(f"  {kind:9s} x{n:4d}: wait {w:6.2f}, land {land:5.2f}, run {run:6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
