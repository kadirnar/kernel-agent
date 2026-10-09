"""Example: a toy decoder's greedy generation loop run on the device with
``kernel_agent.graphloop.device_loop`` (#232).

``ToyDecoder`` is a stand-in for any decoder with a static KV cache (an LLM, the LM of a TTS
model): token and position embeddings, ``layers`` pre-norm blocks (attention over the whole
static cache with a position mask, a GELU MLP) and the tied output projection. Its ``step``
reads the token and its position from device tensors and writes the KV cache at that
position, so a graph can replay it with nothing from the host.

The same greedy tokens, four ways:

* :func:`generate_host`: the plain loop, the stop read with ``.item()`` every step;
* :func:`generate_graph_steps`: one CUDA graph per step replayed from Python, the stop read
  every step (the usual "CUDA graphs for decode");
* :class:`DeviceGenerator` with ``mode="while"``: one graph whose WHILE node runs every
  step and checks the stop on the device (no host check at all);
* :class:`DeviceGenerator` with ``mode="unrolled"``: the fallback, K steps per graph replay,
  the stop read once per block; steps after the stop are masked
  (``graphloop.masked_copy_`` / ``masked_index_copy_``: no write past the stop).

With ``--compile`` the step is also ``torch.compile``'d (the default mode: Inductor's
kernels, no CUDA graphs of its own; :func:`compiled_request`) and run as a host loop, a
graph per step and a WHILE loop, judged against the compiled host loop's tokens (Inductor's
fusions change the numerics, so they may differ from the eager ones).

``python graph_while_decode.py`` prints, per way, the tokens' agreement with the host loop,
the time per token (GPU-synchronised runs), the host time per token (how long the call
held the host) and the host checks per run. Measured on an RTX 5070 Ti (sm_120) and an A10
(sm_86): see ``main``'s docstring.
"""

from __future__ import annotations

import argparse
import copy
import math
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from kernel_agent import graphloop

#: GPUs this example runs on (``graphloop`` falls back where conditional nodes are missing).
ARCHS = "sm_70+"
ARCHS_WHY = (
    "CUDA graphs (the WHILE node needs a CUDA 12.4+ driver; elsewhere the unrolled fallback)"
)


def _rms(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype) * weight


class ToyDecoder(nn.Module):
    """A random decoder with a static KV cache ``[layers, heads, max_len, head_dim]``."""

    def __init__(
        self,
        vocab: int = 2048,
        dim: int = 512,
        heads: int = 8,
        layers: int = 4,
        max_len: int = 512,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        seed: int = 0,
    ) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)

        def weight(*shape: int, scale: float = 1.0) -> nn.Parameter:
            w = torch.randn(*shape, generator=gen) * scale / math.sqrt(shape[0])
            return nn.Parameter(w.to(device=device, dtype=dtype), requires_grad=False)

        self.heads, self.head_dim, self.max_len = heads, dim // heads, max_len
        self.emb = weight(vocab, dim, scale=math.sqrt(vocab))
        # random weights collapse greedy decoding into a repeated token; a position
        # embedding keeps the toy's tokens changing, so a late stop token exists
        self.pos = weight(max_len, dim, scale=math.sqrt(max_len))
        self.wqkv = weight(layers, dim, 3 * dim)
        self.wo = weight(layers, dim, dim)
        self.w1 = weight(layers, dim, 4 * dim)
        self.w2 = weight(layers, 4 * dim, dim)
        self.norm = nn.Parameter(torch.ones(2 * layers + 1, dim, device=device, dtype=dtype))
        with torch.inference_mode(False):  # written in place in and out of inference mode
            shape = (layers, heads, max_len, self.head_dim)
            self.k = torch.zeros(shape, device=device, dtype=dtype)
            self.v = torch.zeros(shape, device=device, dtype=dtype)
            self.slots = torch.arange(max_len, device=device)

    def step(
        self, token: torch.Tensor, pos: torch.Tensor, active: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Logits ``[vocab]`` of the token after ``token`` (``[1]``, int64) at position
        ``pos`` (a device int64 scalar); writes the KV cache at ``pos``. ``active`` false: a
        masked step that writes nothing (the unrolled loop's steps after the stop)."""
        at = pos.view(1)
        x = self.emb.index_select(0, token) + self.pos.index_select(0, at)  # [1, dim]
        valid = self.slots <= pos
        for layer in range(self.wqkv.shape[0]):
            h = _rms(x, self.norm[2 * layer])
            q, k, v = (h @ self.wqkv[layer]).view(3, self.heads, 1, self.head_dim).unbind(0)
            if active is None:
                self.k[layer].index_copy_(1, at, k)
                self.v[layer].index_copy_(1, at, v)
            else:
                graphloop.masked_index_copy_(self.k[layer], 1, at, k, active)
                graphloop.masked_index_copy_(self.v[layer], 1, at, v, active)
            scores = (q @ self.k[layer].transpose(1, 2)).float() / math.sqrt(self.head_dim)
            p = scores.masked_fill(~valid, float("-inf")).softmax(-1).to(x.dtype)
            x = x + (p @ self.v[layer]).reshape(1, -1) @ self.wo[layer]
            h = _rms(x, self.norm[2 * layer + 1])
            x = x + F.gelu(h @ self.w1[layer]) @ self.w2[layer]
        return (_rms(x, self.norm[-1]) @ self.emb.t())[0]


class Request:
    """One request's static state: the last token, where its position starts, the output
    buffer; :meth:`prefill` writes the prompt into the cache."""

    def __init__(self, model: ToyDecoder, max_new: int) -> None:
        dev = model.emb.device
        with torch.inference_mode(False):
            self.last = torch.zeros(1, dtype=torch.int64, device=dev)
            self.start = torch.zeros((), dtype=torch.int64, device=dev)
            self.out = torch.full((max_new,), -1, dtype=torch.int64, device=dev)
        self.model, self.max_new = model, max_new

    def prefill(self, prompt: list[int]) -> None:
        model, dev = self.model, self.last.device
        model.k.zero_()
        model.v.zero_()
        tokens = torch.tensor(prompt, dtype=torch.int64, device=dev)
        for i in range(len(prompt) - 1):  # token by token through the same step
            model.step(tokens[i : i + 1], torch.tensor(i, device=dev))
        self.last.copy_(tokens[-1:])
        self.start.fill_(len(prompt) - 1)
        self.out.fill_(-1)

    def step(self, index: torch.Tensor, active: torch.Tensor | None) -> None:
        """Decode step ``index``: the token after ``last`` into ``out[index]`` and ``last``."""
        logits = self.model.step(self.last, self.start + index, active)
        token = logits.argmax(-1, keepdim=True)
        if active is None:
            self.out.index_copy_(0, index.view(1), token)
            self.last.copy_(token)
        else:
            graphloop.masked_index_copy_(self.out, 0, index.view(1), token, active)
            graphloop.masked_copy_(self.last, token, active)


def compiled_request(req: Request) -> Request:
    """``req`` (its buffers shared) with a ``torch.compile`` step in the default mode:
    Inductor's kernels, no CUDA graphs of its own, so a graph or a device loop captures
    them (``mode="reduce-overhead"`` would make graphs of its own)."""
    compiled = copy.copy(req)
    compiled.step = torch.compile(req.step)  # type: ignore[method-assign]
    return compiled


def generate_host(req: Request, prompt: list[int], eos: int) -> tuple[list[int], int]:
    """The plain loop: ``(tokens, host checks)``, one ``.item()`` per step."""
    req.prefill(prompt)
    index = torch.zeros((), dtype=torch.int64, device=req.last.device)
    checks = 0
    for _ in range(req.max_new):
        req.step(index, None)
        index += 1
        checks += 1
        if int(req.last.item()) == eos:
            break
    return req.out[: int(index)].tolist(), checks


class GraphSteps:
    """One CUDA graph per step, replayed from Python; the stop read every step."""

    def __init__(self, req: Request) -> None:
        self.req = req
        with torch.inference_mode(False):
            self.index = torch.zeros((), dtype=torch.int64, device=req.last.device)
        # warm-up on these buffers: a torch.compile step compiles for them here, not under the
        # capture (where compiling fails); generate() prefills before it replays
        req.step(self.index, None)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            req.step(self.index, None)
            self.index += 1

    def generate(self, prompt: list[int], eos: int) -> tuple[list[int], int]:
        self.req.prefill(prompt)
        self.index.zero_()
        checks = 0
        for _ in range(self.req.max_new):
            self.graph.replay()
            checks += 1
            if int(self.req.last.item()) == eos:
                break
        return self.req.out[: int(self.index)].tolist(), checks


class DeviceGenerator:
    """The loop on the device: ``graphloop.device_loop`` over :meth:`Request.step`.
    ``masked``: the step gates its writes with ``active`` (the unrolled fallback needs it;
    a WHILE graph runs no step after the stop, so it may skip the masking's extra ops)."""

    def __init__(
        self, req: Request, eos: int, mode: str = "auto", masked: bool = True, **options: Any
    ) -> None:
        self.req = req

        def step(index: torch.Tensor, active: torch.Tensor) -> None:
            req.step(index, active if masked else None)

        self.loop = graphloop.device_loop(
            step,
            lambda: req.last == eos,
            req.max_new,
            mode=mode,
            masked=masked,
            check_state=[req.out, req.last, req.model.k] if masked else (),
            **options,
        )

    def generate(self, prompt: list[int]) -> tuple[list[int], int]:
        self.req.prefill(prompt)
        checks = self.loop.stats["host_checks"]
        steps = int(self.loop.run())  # the one read of the run, after the loop
        return self.req.out[:steps].tolist(), self.loop.stats["host_checks"] - checks


def _time(fn: Any, runs: int) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / runs


def compare(
    max_new: int = 256,
    prompt_len: int = 16,
    runs: int = 10,
    compile: bool = False,
    **model_options: Any,
) -> dict[str, dict[str, Any]]:
    """Every way on one model and prompt: per way the tokens' agreement with the host loop,
    ms per token (prefill excluded), the host time per token of the loop call (a WHILE
    graph: its launch) and the host checks per run; ``eos`` is the first token the host
    loop generates for the first time from 3/4 of ``max_new`` on, so the loop stops on the
    device's own decision there. ``compile``: the ``torch.compile`` ways too, judged
    against the compiled host loop."""
    model = ToyDecoder(**model_options)
    gen = torch.Generator().manual_seed(1)
    prompt = torch.randint(0, model.emb.shape[0], (prompt_len,), generator=gen).tolist()
    req = Request(model, max_new)
    free, _ = generate_host(req, prompt, eos=-1)
    late = range(3 * max_new // 4, max_new)
    eos = next((free[i] for i in late if free[i] not in free[:i]), -1)  # -1: max_new
    reference, _ = generate_host(req, prompt, eos)
    graphed = GraphSteps(req)
    loops = {
        "device loop (while)": DeviceGenerator(req, eos, mode="while", masked=False),
        "device loop (while, masked step)": DeviceGenerator(req, eos, mode="while"),
        "device loop (unrolled)": DeviceGenerator(req, eos, mode="unrolled"),
    }
    # per way: the call and the tokens it must give
    ways: dict[str, tuple[Any, list[int]]] = {
        "host loop": (lambda: generate_host(req, prompt, eos), reference),
        "graph per step": (lambda: graphed.generate(prompt, eos), reference),
    }
    ways |= {name: (lambda g=g: g.generate(prompt), reference) for name, g in loops.items()}
    if compile:
        fast = compiled_request(req)
        compiled, _ = generate_host(fast, prompt, eos)  # compiles the step before any capture
        fast_graphed = GraphSteps(fast)
        fast_loop = DeviceGenerator(fast, eos, mode="while", masked=False)
        loops["device loop (while, torch.compile step)"] = fast_loop
        ways |= {
            "host loop (torch.compile step)": (
                lambda: generate_host(fast, prompt, eos),
                compiled,
            ),
            "graph per step (torch.compile step)": (
                lambda: fast_graphed.generate(prompt, eos),
                compiled,
            ),
            "device loop (while, torch.compile step)": (
                lambda: fast_loop.generate(prompt),
                compiled,
            ),
        }
    prefill_s = _time(lambda: req.prefill(prompt), runs)
    out: dict[str, dict[str, Any]] = {}
    for name, (call, want) in ways.items():
        call()  # warm-up (a device loop's first run is a host run, then it builds)
        tokens, checks = call()
        total_s = _time(call, runs)
        n = len(tokens)
        row: dict[str, Any] = {
            "same_tokens": tokens == want,
            "tokens": n,
            "ms_per_token": round((total_s - prefill_s) / n * 1e3, 4),
            "host_checks": checks,
        }
        if name in loops:
            loop = loops[name].loop
            req.prefill(prompt)
            torch.cuda.synchronize()
            start = time.perf_counter()
            loop.run()  # a WHILE graph returns at its launch
            row["host_us_per_token"] = round((time.perf_counter() - start) / n * 1e6, 3)
            torch.cuda.synchronize()
            row |= {"mode": loop.mode, "reason": loop.reason}
            if "unroll" in loop.stats:
                row["unroll"] = loop.stats["unroll"]
        out[name] = row
    return out


def main() -> None:
    """Print :func:`compare` for one toy model.

    RTX 5070 Ti, driver 615.71 (CUDA 13.4), torch 2.14.1+cu130, cuda.core 1.2.1, bf16, 256
    new tokens with the stop at ~193, ms per token over 20 runs (the same tokens every way):

    ============================  ==========  =========  =================
    way                           4 x 512     1 x 256    1 x 128, vocab 512
    ============================  ==========  =========  =================
    host loop                     1.875       0.585      0.570
    graph per step                0.269       0.067      0.063
    device loop (while)           0.253       0.062      0.060
    device loop (while, masked)   0.276       0.068      0.067
    device loop (unrolled, K=1)   0.281       0.077      0.073
    ============================  ==========  =========  =================

    The WHILE loop held the host ~0.3 µs per token and checked nothing on the host; the
    unrolled fallback chose K = 1 (GPU-bound steps) and pays for its masked writes.

    Measured on an NVIDIA A10, sm_86 (150 W, shared with another tenant), driver 570.86
    (CUDA 12.9), torch 2.10.0+cu128, cuda.core 1.2.1, bf16, the same model and stops, median
    of 3 repetitions of 20 runs (``--compile``; the same tokens every way):

    ===================================  ==========  =========  =================
    way                                  4 x 512     1 x 256    1 x 128, vocab 512
    ===================================  ==========  =========  =================
    host loop                            1.861       0.650      0.617
    graph per step                       0.377       0.120      0.099
    device loop (while)                  0.369       0.095      0.092
    device loop (while, masked)          0.409       0.108      0.102
    device loop (unrolled, K=1)          0.428       0.113      0.111
    graph per step (torch.compile)       0.270
    device loop (while, torch.compile)   0.258
    ===================================  ==========  =========  =================

    The torch.compile ways ran 256 tokens (Inductor's numerics never generate the eager
    stop token there) and match the compiled host loop (0.866 ms per token); on the 1-layer
    models they stop after 78 and 24 tokens, too few for a stable number. The WHILE loop
    held the host 0.25-0.35 µs per token."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--vocab", type=int, default=2048)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--compile", action="store_true", help="the torch.compile ways too")
    ns = parser.parse_args()
    with torch.inference_mode():
        rows = compare(
            ns.max_new,
            runs=ns.runs,
            compile=ns.compile,
            layers=ns.layers,
            dim=ns.dim,
            vocab=ns.vocab,
        )
    for name, row in rows.items():
        print(f"{name:42s} {row}")


if __name__ == "__main__":
    main()
