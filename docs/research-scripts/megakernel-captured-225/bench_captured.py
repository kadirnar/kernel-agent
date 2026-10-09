"""Megakernel schedules built from a recorded call against the hand-declared one (issue #225,
follow-up 5: ``kernel_agent.native.megakernel.captured``).

1. The example's chain (``kernel_agent.selftest.NormGemvChain``, ``--layers`` layers of [H, H]
   for each H of ``--hidden``): ``build(ref)`` (the op DAG declared by hand, ``chain_dag``)
   and ``build(ref, schedule="captured")`` (the DAG of one recorded call of the reference),
   at each ``--inflight``: whether the two programs are byte-identical, whether their outputs
   are bit-identical, their difference from eager, the simulator's prediction from the
   roofline costs of this GPU's measured peaks, and the GPU time of one call (its CUDA graph
   replayed ``--reps`` times between two events, ``--trials`` times, the engines in turn).
2. A decode-step gated MLP (``kernel_agent.selftest.GatedMlpBlock``, one row, each
   ``--mlp HIDDEN:INTER``) as one megakernel launch from its recorded call (the norm fused
   into gate and up, the GLU opcode on chunked counters, down + residual; ``split_k`` 1,
   ``auto`` and 2), against the same PyTorch ops captured in a CUDA graph.

The DRAM floor is the weights' bytes over this GPU's copy bandwidth (``bench_chain.
bandwidth``, as ``doctor`` measures it). Run (GPU, ~1 min):

    flock ~/.cache/kernel-agent/gpu.lock env KERNEL_AGENT_LOCK_HELD=1 PYTHONPATH=src \\
        python docs/research-scripts/megakernel-captured-225/bench_captured.py \\
        > docs/research-scripts/megakernel-captured-225/bench_captured.txt
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "docs" / "research-scripts" / "megakernel-a10-225"))

import torch  # noqa: E402
from bench_chain import bandwidth, chain, time_graphs  # noqa: E402

from kernel_agent import toolchain  # noqa: E402
from kernel_agent.agent.prompts import EXAMPLES_DIR  # noqa: E402
from kernel_agent.kernels import bench  # noqa: E402
from kernel_agent.native import project  # noqa: E402
from kernel_agent.native.megakernel import captured, runtime, simulate  # noqa: E402
from kernel_agent.selftest import GatedMlpBlock  # noqa: E402


class Graph:
    """A CUDA graph of ``launch`` (what ``time_graphs`` replays)."""

    def __init__(self, launch) -> None:
        launch()
        torch.cuda.synchronize()
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            launch()


def summary(name: str, ts: list[float], floor_us: float) -> str:
    best, med = min(ts), statistics.median(ts)
    return f"  {name:58s} {best:7.1f} / {med:7.1f} us  ({floor_us / med:.0%} of floor speed)"


def chains(mod, ns, copy_gbps: float) -> None:
    for hidden in ns.hidden:
        ref = chain(hidden, ns.layers)
        weights = sum(layer.weight.numel() * 2 for layer in ref.layers)
        x = torch.randn(1, hidden, device="cuda", dtype=torch.bfloat16)
        engines = {}
        for inflight in ns.inflight:
            for source in ("captured", "hand") if ns.captured_first else ("hand", "captured"):
                engines[f"megakernel, {source} schedule (inflight {inflight})"] = mod.build(
                    ref, inflight=inflight, schedule=source
                )
        hand, auto = (engines[f"megakernel, {s} schedule (inflight {ns.inflight[0]})"]
                      for s in ("hand", "captured"))  # fmt: skip
        plan = auto.plan
        print(f"\nH = {hidden}: {ns.layers} x [{hidden}, {hidden}], {weights / 1e6:.1f} MB of "
              f"weights; captured plan: {len(plan.ops)} ops ({plan.info[0].family} x "
              f"{plan.info[0].tiles} tiles, {plan.info[0].shape}; "
              f"{', '.join(plan.info[0].fused)}), {len(plan.edges)} edges, "
              f"{len(plan.unsupported)} unsupported ops")  # fmt: skip
        print(f"  costs: {plan.cost_source}")
        sim = simulate.simulate(auto.schedule)
        print(f"  programs byte-identical: {auto.schedule.to_bytes() == hand.schedule.to_bytes()}"
              f"; simulator over 1000 SM-speed draws: "
              f"{simulate.check(auto.schedule) or 'no deadlock'}; predicted "
              f"{sim.makespan:.1f} us (roofline costs, every queue a 1/{auto.schedule.n_queues}"
              f" share of the DRAM bandwidth)")  # fmt: skip
        with torch.no_grad():
            want = ref(x)
            outs = {name: engine(x) for name, engine in engines.items()}
        first = next(iter(outs.values()))
        same = all(torch.equal(first, out) for out in outs.values())
        diff = (first.float() - want.float()).abs().max().item()
        print(f"  outputs bit-identical across the engines: {same}; max |diff| from eager "
              f"{diff:.3g}")
        times = time_graphs(engines, ns.reps, ns.trials)
        floor_us = weights / copy_gbps / 1e3
        print(f"  {'DRAM floor (copy bandwidth)':58s} {floor_us:7.1f} us")
        for name, ts in times.items():
            print(summary(name, ts, floor_us))
        del engines, hand, auto, outs
        torch.cuda.empty_cache()


def mlps(mod, ns, copy_gbps: float) -> None:
    ext = mod.extension()
    pages, queues, page_bytes, _ = ext.mk_info()
    peaks, source = captured.peaks_here()
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    for spec in ns.mlp:
        hidden, inter = (int(v) for v in spec.split(":"))
        torch.manual_seed(0)
        block = GatedMlpBlock(hidden, inter).cuda().to(torch.bfloat16).eval()
        with torch.no_grad():
            for name, w in block.named_parameters():
                w.normal_(1.0, 0.1) if "norm" in name else w.normal_(0.0, w.shape[-1] ** -0.5)
        weights = sum(p.numel() * 2 for n, p in block.named_parameters() if "proj" in n)
        x = torch.randn(1, hidden, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            want = block(x)
        print(f"\ngated MLP [{hidden} -> {inter} -> {hidden}], one row, {weights / 1e6:.1f} MB "
              f"of weights")  # fmt: skip
        engines: dict[str, Graph] = {}
        for split in ("1", "auto", "2"):
            plan = captured.from_module(
                block, (x,), pool_bytes=pages * page_bytes, queues=queues,
                split_k=split if split == "auto" else int(split), peaks=peaks,
                peaks_source=source, l2_bytes=l2,
            )  # fmt: skip
            sched = plan.schedule(queues)
            tensors = plan.tensors()
            plan.bind(tensors, plan.inputs[0]).copy_(x)
            rt = runtime.Runtime(sched, tensors, pool_bytes=pages * page_bytes)
            name = f"megakernel from the recorded call, split_k={split}"
            engines[name] = Graph(lambda rt=rt: ext.mk_run(*rt.args(), pages, queues, 1))
            rt.check()
            out = plan.bind(tensors, plan.outputs[0])
            ops = ", ".join(f"{i.name} {i.family} x{i.tiles}" for i in plan.info)
            print(f"  split_k={split}: {ops}; {len(sched.counters)} counters; simulator: "
                  f"{simulate.check(sched) or 'no deadlock'}, predicted "
                  f"{simulate.simulate(sched).makespan:.1f} us; max |diff| from eager "
                  f"{(out.float() - want.float()).abs().max().item():.3g}")  # fmt: skip
            engines[name]._keep = (plan, tensors, rt)  # the graph holds their addresses
        static = x.clone()

        def eager(static=static, block=block) -> None:
            with torch.no_grad():
                block(static)

        engines["PyTorch ops in a CUDA graph"] = Graph(eager)
        times = time_graphs(engines, ns.reps, ns.trials)
        floor_us = weights / copy_gbps / 1e3
        print(f"  {'DRAM floor (copy bandwidth)':58s} {floor_us:7.1f} us")
        for name, ts in times.items():
            print(summary(name, ts, floor_us))
        del engines
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--hidden", type=int, nargs="+", default=[1024, 2048])
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--inflight", type=int, nargs="+", default=[2, 1])
    parser.add_argument("--mlp", nargs="+", default=["1024:3072", "2048:8192"])
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--captured-first", action="store_true", help="time it first")
    parser.add_argument("--no-mlp", action="store_true")
    ns = parser.parse_args()

    tc = toolchain.setup()
    gpu = tc.gpu
    print(f"GPU: {gpu.name} ({gpu.arch}), {gpu.sm_count} SMs, L2 {gpu.l2_cache_mb:.0f} MB")
    print(f"torch {torch.__version__} (CUDA {torch.version.cuda}); date "
          f"{time.strftime('%Y-%m-%d %H:%M')}; {ns.trials} trials of {ns.reps} graph replays "
          "per engine (per-call us: min / median)")  # fmt: skip
    mod = project.import_project(EXAMPLES_DIR / "native_megakernel")
    pages, queues, page_bytes, _ = mod.extension().mk_info()
    print(f"megakernel: {pages} pages of {page_bytes} B per block, {queues} queues")
    bench.warm_gpu()
    copy_gbps, read_gbps = bandwidth()
    print(f"DRAM: copy {copy_gbps:.1f} GB/s (doctor's dram_gbps), read-only {read_gbps:.1f} GB/s")
    chains(mod, ns, copy_gbps)
    if not ns.no_mlp:
        mlps(mod, ns, copy_gbps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
