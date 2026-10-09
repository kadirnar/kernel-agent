"""A compute-bound layer through the evaluator in the W4A4 tier (#233): the VoxCPM2 LocDiT
decoder layer (M = 352 / 176) captured again in ``near-lossless-fp4a`` (and ``relaxed-fp4a``)
from the run's capture (read only), and two candidates built from the bundled examples:

* ``all W4A4``: every ``nn.Linear`` by ``examples/cute_nvfp4_w4a4_gemm.py``;
* ``W4A4, gate / up in FP8``: the sensitivity probe's top two layers by
  ``examples/cute_fp8_blockscaled_gemm.py`` (W8A8), the rest W4A4,

each through ``kernels.evaluate.run_evaluation`` (captured, redrawn and scaled inputs, timing
against the bf16 layer, eager), plus the probe's ranking (``quant.fp4_w4a4_sensitivity``).

    python locdit_gate.py CAPTURE.pt WORKDIR   (results/locdit_gate.out)
"""

import copy
import sys
from pathlib import Path

import torch

from kernel_agent.kernels import quant
from kernel_agent.kernels.evaluate import run_evaluation
from kernel_agent.profiling.capture import capture_calls, load_capture

CANDIDATE = """import copy
import importlib.util
import math

from torch import nn

from kernel_agent.agent.prompts import EXAMPLES_DIR


def _load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES_DIR / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W4A4 = _load("cute_nvfp4_w4a4_gemm")
FP8 = _load("cute_fp8_blockscaled_gemm")
KEEP_FP8 = {keep!r}


class Layer(nn.Module):
    # MiniCPMDecoderLayer.forward written out (the evaluator rejects a candidate that runs the
    # reference's entrypoint with most of its time in the reference's kernels): the norms,
    # rotary attention and residuals as in eager, every GEMM a W4A4 (or FP8) example module
    def __init__(self, reference):
        super().__init__()
        self.ref = copy.deepcopy(reference)
        for name, mod in list(self.ref.named_modules()):
            for child_name, child in list(mod.named_children()):
                if isinstance(child, nn.Linear):
                    full = f"{{name}}.{{child_name}}" if name else child_name
                    fp8 = any(k in full for k in KEEP_FP8)
                    setattr(mod, child_name, (FP8 if fp8 else W4A4).build(child))

    def forward(self, hidden_states, position_emb, is_causal):
        r = self.ref
        residual = hidden_states
        hidden_states = r.input_layernorm(hidden_states)
        hidden_states, present_key_value = r.self_attn(
            hidden_states=hidden_states, position_emb=position_emb, is_causal=is_causal
        )
        if r.use_mup:
            hidden_states = residual + hidden_states * (r.scale_depth / math.sqrt(r.num_hidden_layers))
        else:
            hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = r.mlp(r.post_attention_layernorm(hidden_states))
        if r.use_mup:
            hidden_states = residual + hidden_states * (r.scale_depth / math.sqrt(r.num_hidden_layers))
        else:
            hidden_states = residual + hidden_states
        return hidden_states, present_key_value


def build(reference):
    return Layer(reference)
"""


def main(src: str, workdir: str) -> None:
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    cap = load_capture(Path(src), device="cuda")
    module = cap["module"].eval()
    calls = [(case["args"], case["kwargs"], case["count"]) for case in cap["cases"]]

    first = calls[0]
    probe = quant.fp4_w4a4_sensitivity(module, lambda: module(*copy.deepcopy(first[0])))
    print("## the sensitivity probe (M = 352)")
    for row in probe:
        print(
            f"  {row['name']:18s} W4A4 rel L2 {row['w4a4_rel_l2']:.4f} norm "
            f"{row['w4a4_norm_change'] * 100:+.2f} %  FP8 {row['fp8_rel_l2']:.4f}  "
            f"FLOP share {row['flop_share']:.3f}  crest {row['activation_crest']:.0f}"
        )

    for tier in ("near-lossless-fp4a", "relaxed-fp4a"):
        path = work / f"dit_layer.{tier}.pt"
        capture_calls(module, copy.deepcopy(calls), path, tier=tier, precision="fp4_w4a4")
        for label, keep in (("all W4A4", ()), ("W4A4, gate / up in FP8", ("gate_proj", "up_proj"))):
            candidate = work / f"cand_{len(keep)}.py"
            candidate.write_text(CANDIDATE.format(keep=keep))
            result = run_evaluation(path, candidate, timeout=900)
            cases = result.get("cases") or []
            print(
                f"\n## {tier}: {label}\n  status {result.get('status')} correct "
                f"{result.get('correct')} speedup {result.get('speedup')}x (eager, vs the bf16 "
                f"layer) checks {sorted(result.get('checks') or [])}"
            )
            for i, case in enumerate(cases):
                print(
                    f"  case {i}: min cosine {case.get('min_cosine')} max rel L2 "
                    f"{case.get('max_rel_l2')} ref {case.get('ref_ms')} ms new "
                    f"{case.get('new_ms')} ms"
                )
            if not result.get("correct"):
                print(f"  why: {str(result.get('error', ''))[:600]}")
    graph_times(module, first, work)
    torch.cuda.synchronize()


def graph_ms(fn, calls: int = 20, replays: int = 5) -> float:
    """GPU time of ``fn`` per call inside a CUDA graph (best of ``replays``; warm L2)."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    best = float("inf")
    for _ in range(replays):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / calls)
    return best


def graph_times(module, call, work: Path) -> None:
    """The layer's GPU time at M = 352 in a CUDA graph (the evaluator times eagerly, where
    this layer is host bound): bf16 against the two candidates."""
    import importlib.util

    args = copy.deepcopy(call[0])
    print("\n## GPU time per layer call, CUDA graph (M = 352)")
    with torch.inference_mode():
        ref_ms = graph_ms(lambda: module(*args))
        print(f"  bf16 layer (reference)        {ref_ms * 1e3:7.1f} us")
        (work / "cand_fp8.py").write_text(CANDIDATE.format(keep=("_proj",)))  # every GEMM FP8
        for label, n in (("all W4A4", 0), ("W4A4, gate / up in FP8", 2), ("all FP8 W8A8", "fp8")):
            spec = importlib.util.spec_from_file_location(f"cand{n}", work / f"cand_{n}.py")
            cand = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cand)
            layer = cand.build(module)
            ms = graph_ms(lambda layer=layer: layer(*args))
            print(f"  {label:29s} {ms * 1e3:7.1f} us  ({ref_ms / ms:.2f}x)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
