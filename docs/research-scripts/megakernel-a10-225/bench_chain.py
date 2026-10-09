"""The megakernel example against its two baselines on this GPU (issue #225, follow-up 6).

For ``--layers`` layers of [H, H] (bf16, ``kernel_agent.selftest.NormGemvChain``) and each H
of ``--hidden``: every mode of ``agent/examples/native_megakernel`` (the megakernel at each
``--rows`` x ``--inflight``, the graph of one kernel per layer, with PDL edges where the GPU
has PDL, and the cooperative kernel with a grid barrier per layer), each one's output
compared with the eager reference (max |diff| and the elements outside the GPU tests'
tolerance: 28 bf16 layers drift from cuBLAS's rounding, issue #248) and timed as the GPU time
of one call: its CUDA graph replayed ``--reps`` times back to back between two events,
``--trials`` times, the modes in turn within each trial (a power-capped board's clock drifts
alike for all). The DRAM floor is the weights' bytes over this GPU's copy bandwidth, measured
as ``doctor``'s peaks measure it (``roofline._copy_gbps``, read + write bytes of a 512 MiB
copy); the read-only stream (a ``torch.sum`` over the same buffer) is printed next to it.
Then the megakernel's trace (``KA_MK_TRACE``) per layer at the first H and rows, for each
inflight. Run (GPU, ~1 min):

    flock ~/.cache/kernel-agent/gpu.lock env KERNEL_AGENT_LOCK_HELD=1 PYTHONPATH=src \
        python docs/research-scripts/megakernel-a10-225/bench_chain.py \
        > docs/research-scripts/megakernel-a10-225/bench_chain.txt
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
from kernel_agent.selftest import NormGemvChain  # noqa: E402

MiB = 1 << 20


def chain(hidden: int, layers: int) -> torch.nn.Module:
    """The reference (projections ~ N(0, hidden ** -0.5), norm weights ~ N(1, 0.1))."""
    torch.manual_seed(0)
    ref = NormGemvChain(hidden, layers).cuda().to(torch.bfloat16).eval()
    with torch.no_grad():
        for name, weight in ref.named_parameters():
            if name.startswith("norms."):
                weight.normal_(1.0, 0.1)
            else:
                weight.normal_(0.0, hidden**-0.5)
    return ref


def bandwidth() -> tuple[float, float]:
    """(copy GB/s as doctor's peaks measure it, read-only GB/s of a sum)."""
    free, _ = torch.cuda.mem_get_info()
    n = int(min(512 * MiB, free // 8)) // 16 * 16
    src = torch.ones(n // 2, device="cuda", dtype=torch.bfloat16)
    dst = torch.empty_like(src)
    copy = roofline._copy_gbps(src, dst)
    del dst
    ms = roofline._best_ms(lambda: torch.sum(src, dtype=torch.float32), iters=10)
    read = src.numel() * src.element_size() / ms / 1e6
    del src
    torch.cuda.empty_cache()
    return copy, read


def time_graphs(engines: dict[str, torch.nn.Module], reps: int, trials: int) -> dict:
    """Per engine: the per-call GPU times (us) of ``trials`` runs of ``reps`` replays."""
    times: dict[str, list[float]] = {name: [] for name in engines}
    for engine in engines.values():  # warm-up
        for _ in range(10):
            engine._graph.replay()
    torch.cuda.synchronize()
    for _ in range(trials):
        for name, engine in engines.items():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(reps):
                engine._graph.replay()
            end.record()
            end.synchronize()
            times[name].append(start.elapsed_time(end) / reps * 1e3)
    return times


def timings(mod, ns, pages: int, page_bytes: int, copy_gbps: float, read_gbps: float) -> None:
    for hidden in ns.hidden:
        ref = chain(hidden, ns.layers)
        # a tile must fit the page pool (16 rows of 4096 columns do not fit 99 KB: 8 rows)
        fits = list(dict.fromkeys(mod.tile_rows(hidden, pages * page_bytes, r) for r in ns.rows))
        weights = sum(layer.weight.numel() * 2 for layer in ref.layers)
        x = torch.randn(1, hidden, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            want = ref(x)
        engines: dict[str, torch.nn.Module] = {}
        for rows in fits:
            for inflight in ns.inflight:
                engines[f"megakernel ({rows} rows, inflight {inflight or 'all'})"] = mod.build(
                    ref, mode="megakernel", rows=rows, inflight=inflight
                )
        baselines = [mod.build(ref, mode=mode) for mode in ("graph_pdl", "coop_barrier")]
        for engine in baselines:
            engines[engine.label] = engine  # graph_pdl: says whether PDL edges ran
        print(f"\nH = {hidden}: {ns.layers} x [{hidden}, {hidden}], {weights / 1e6:.1f} MB of "
              f"weights, megakernel tiles of {', '.join(map(str, fits))} rows")
        with torch.no_grad():
            outs = {name: engine(x) for name, engine in engines.items()}
        mk = [out for name, out in outs.items() if name.startswith("megakernel")]
        graph, coop = (outs[engine.label] for engine in baselines)
        print("  outputs against eager (cuBLAS; the GPU tests' tolerance atol 0.1, rtol 0.05):")
        for name, out in (("megakernel", mk[0]), ("graph", graph), ("coop_barrier", coop)):
            diff = (out.float() - want.float()).abs()
            outside = ~torch.isclose(out.float(), want.float(), atol=0.1, rtol=0.05)
            print(f"    {name:13s} max |diff| {diff.max().item():.3g}, {int(outside.sum())} of "
                  f"{hidden} outside")
        same = all(torch.equal(mk[0], out) for out in mk)
        print(f"    megakernel bit-identical across its settings: {same}; graph == coop_barrier: "
              f"{torch.equal(graph, coop)}; megakernel == graph: {torch.equal(mk[0], graph)}")
        times = time_graphs(engines, ns.reps, ns.trials)
        floor_us = weights / copy_gbps / 1e3
        read_us = weights / read_gbps / 1e3
        print(f"  {'DRAM floor (copy bandwidth)':62s} {floor_us:7.1f} us")
        print(f"  {'(read-only stream)':62s} {read_us:7.1f} us")
        for name, ts in times.items():
            best, med = min(ts), statistics.median(ts)
            print(f"  {name:62s} {best:7.1f} / {med:7.1f} us  ({floor_us / med:.0%} of floor "
                  f"speed)")
        del engines, baselines, mk, outs
        torch.cuda.empty_cache()


def traces(mod, ns) -> None:
    hidden, rows = ns.hidden[0], ns.rows[0]
    ref = chain(hidden, ns.layers)
    x = torch.randn(1, hidden, device="cuda", dtype=torch.bfloat16)
    print(f"\ntrace (H = {hidden}, {rows} rows; the last of 20 calls, with time stamps), per "
          "layer after the first (mean over its tiles): wait = for the previous layer's "
          "counter, page = the last weight page after it, run = page landed to signal (h: x "
          "loaded and normalised, mark 0; dot: the products, mark 1; tail: the reduction, the "
          "store and the signal); overlap = share of tiles whose weight load was issued before "
          "their counter was met")
    for inflight in ns.inflight:
        engine = mod.build(ref, mode="megakernel", rows=rows, trace=True, inflight=inflight)
        for _ in range(20):
            engine(x)
        torch.cuda.synchronize()
        summary = engine.trace_summary()
        later = [s for op, s in summary["ops"].items() if op != "layer0"]
        mean = {k: statistics.mean(s[k] for s in later) for k in ("wait_ns", "land_ns", "run_ns")}
        stamps = engine.rt.trace_rows()  # issued, begin, met, landed, end, SM, mark 0, mark 1
        tiles = [r for i, r in zip(engine.schedule.instrs, stamps, strict=True)
                 if i.op != "layer0" and r[4]]
        h = statistics.mean(r[6] - r[3] for r in tiles)
        dot = statistics.mean(r[7] - r[6] for r in tiles)
        tail = statistics.mean(r[4] - r[7] for r in tiles)
        first = summary["ops"]["layer0"]
        print(f"  inflight {inflight or 'all'}: span {summary['span_ns'] / 1e3:.1f} us; wait "
              f"{mean['wait_ns'] / 1e3:.2f}, page {mean['land_ns'] / 1e3:.2f}, run "
              f"{mean['run_ns'] / 1e3:.2f} us (h {h / 1e3:.2f}, dot {dot / 1e3:.2f}, tail "
              f"{tail / 1e3:.2f}); overlap min {min(s['overlap'] for s in later):.3f}; layer0: "
              f"page {first['land_ns'] / 1e3:.2f}, run {first['run_ns'] / 1e3:.2f} us")
        del engine


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--hidden", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--rows", type=int, nargs="+", default=[16])
    parser.add_argument("--inflight", type=int, nargs="+", default=[1, 2, 4, 0])
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--project", type=Path, default=EXAMPLES_DIR / "native_megakernel",
                        help="a copy of the example to compare (default: the bundled one)")
    ns = parser.parse_args()

    tc = toolchain.setup()
    gpu = tc.gpu
    print(f"GPU: {gpu.name} ({gpu.arch}), {gpu.sm_count} SMs, L2 {gpu.l2_cache_mb:.0f} MB, "
          f"smem {gpu.smem_per_block_kb:.0f} KB per block / {gpu.smem_per_sm_kb:.0f} KB per SM")
    print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), nvcc {tc.nvcc_version}, "
          f"driver CUDA {'.'.join(map(str, toolchain.driver_cuda_version() or ()))}")
    print(f"date {time.strftime('%Y-%m-%d %H:%M')}; {ns.layers} layers, {ns.trials} trials of "
          f"{ns.reps} graph replays per mode (per-call us: min / median)")
    mod = project.import_project(ns.project)
    ext = mod.extension()
    pages, queues, page_bytes, _ = ext.mk_info()
    print(f"megakernel: {pages} pages of {page_bytes} B per block, {queues} queues "
          f"(resident interpreter blocks)")
    bench.warm_gpu()
    copy_gbps, read_gbps = bandwidth()
    print(f"DRAM: copy {copy_gbps:.1f} GB/s (doctor's dram_gbps), read-only {read_gbps:.1f} GB/s")
    timings(mod, ns, pages, page_bytes, copy_gbps, read_gbps)
    if not ns.no_trace:
        traces(mod, ns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
