"""Helpers of the INT8 research scripts: the bundled examples as modules, CUDA-graph timing
(warm L2) and streamed timing (enough weight copies to exceed L2, as in a model)."""

import importlib.util

import torch

from kernel_agent import toolchain
from kernel_agent.agent.prompts import EXAMPLES_DIR

toolchain.setup()  # CUDA_HOME for load_inline


def example(name):
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), EXAMPLES_DIR / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _graph_time(calls, reps):
    """Best us per call of ``calls`` (a list of zero-argument callables) replayed in a graph."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode():
        for fn in calls:
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.inference_mode():
        for _ in range(reps):
            for fn in calls:
                fn()
    g.replay()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        best = min(best, a.elapsed_time(b) * 1000 / (reps * len(calls)))
    return best


def graph_us(fn, reps=20):
    """One callable, warm L2."""
    return _graph_time([fn], reps)


def streamed_us(mods, x, reps=3):
    """Every module copy called once per pass: weights larger than L2 stream from DRAM."""
    return _graph_time([lambda m=m: m(x) for m in mods], reps)
