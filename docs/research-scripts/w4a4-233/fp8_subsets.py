"""Per-output metrics of a captured layer with every nn.Linear in W4A4 and named subsets in FP8
W8A8 (#233: which layers to keep in FP8).

    python fp8_subsets.py CAPTURE.pt   (the VoxCPM2 LocDiT layer: results/fp8_subsets.out)
"""

import copy
import sys

import torch
from torch import nn

from kernel_agent.kernels import compare, quant
from kernel_agent.profiling.capture import load_capture
from kernel_agent.profiling.state import Replay

path = sys.argv[1]
cap = load_capture(path, device="cuda")
ref_mod = cap["module"].eval()


class W4(nn.Module):
    def __init__(self, lin, kind):
        super().__init__()
        self.bias, self.kind = lin.bias, kind
        w = lin.weight.detach()
        if kind == "fp8":
            self.q, self.s = quant.quantize_fp8(w)
        else:
            self.q = quant.quantize_fp4(w.float())

    def forward(self, x):
        if self.kind == "fp8":
            return quant.fp8_w8a8_linear(x, self.q, self.s, self.bias)
        return quant.fp4_w4a4_linear(x, *self.q, self.bias)


def build(keep_fp8=()):
    m = copy.deepcopy(ref_mod)
    for name, mod in list(m.named_modules()):
        for cn, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                full = f"{name}.{cn}" if name else cn
                setattr(mod, cn, W4(child, "fp8" if any(k in full for k in keep_fp8) else "w4"))
    return m.eval()


for keep in [(), ("down_proj",), ("o_proj", "down_proj"), ("gate_proj", "up_proj")]:
    cand = build(keep)
    print(f"\n### W4A4 except FP8 in {keep or 'none'}")
    for i, case in enumerate(cap["cases"][:2]):
        fn = Replay(cap, cand).call(case, cand)
        args, kwargs = copy.deepcopy((case["args"], case["kwargs"]))
        with torch.inference_mode():
            out = fn(*args, **kwargs)
        for tier in ("near-lossless-fp4a",):
            checks = compare.compare_structures(case["output"], out, "out", tier=tier)
            for c in checks:
                if "cosine" in c:
                    print(f"  case {i} {c['name']:20s} ok={c['ok']} cos {c['cosine']:.5f} rel {c['rel_l2']:.4f} "
                          f"norm {(c['norm_ratio'] - 1) * 100:+.2f} % el {c.get('element_ratio', 0):.2f}")
