"""Where W4A4's norm shrink comes from (#233): one nn.Linear of a capture quantised to NVFP4 on
the weights, the activations or both, with e4m3 ("rn") or unrounded fp32 ("fp32") block scales.

    python norm_shrink.py CAPTURE.pt   (the VoxCPM2 LocDiT layer: results/norm_shrink.out)
"""

import copy
import sys

import torch
from torch import nn

from kernel_agent.kernels import quant
from kernel_agent.profiling.capture import load_capture
from kernel_agent.profiling.state import Replay

cap = load_capture(sys.argv[1], device="cuda")
mod = cap["module"].eval()
seen = {}


def keep(name):
    def hook(m, a):
        seen.setdefault(name, a[0].detach())

    return hook


for n, m in mod.named_modules():
    if isinstance(m, nn.Linear):
        m.register_forward_pre_hook(keep(n))
case = cap["cases"][0]
with torch.inference_mode():
    Replay(cap, mod).call(case, mod)(*copy.deepcopy(case["args"]), **copy.deepcopy(case["kwargs"]))
lut = torch.tensor(quant.E2M1_VALUES, device="cuda")
lut = torch.cat([lut, -lut])


def fq(a, rule):
    """fake-quant rows of a [R, K] to NVFP4 with a block-scale rule."""
    r, k = a.shape
    b = a.reshape(r, k // 16, 16)
    bmax = b.abs().amax(-1)
    amax = a.abs().amax(1)
    outer = torch.where(amax > 0, amax / 2688, torch.ones_like(amax))  # a row of zeros: 1
    s = bmax / (outer[:, None] * 6)
    if rule == "rn":  # e4m3 block scales (quant.quantize_fp4_activations)
        s = s.clamp(max=448).to(torch.float8_e4m3fn).float()
    step = s * outer[:, None]  # rule "fp32": unrounded block scales
    v = (b / step[..., None].clamp_min(1e-38)).clamp(-6, 6)
    codes = quant._round_e2m1(v.reshape(r, -1)).long()
    return (lut[codes].reshape(r, k // 16, 16) * step[..., None]).reshape(r, k)


for name in ("mlp.gate_proj", "mlp.down_proj", "self_attn.q_proj"):
    x = seen[name].reshape(-1, seen[name].shape[-1]).float()
    w = dict(mod.named_modules())[name].weight.float()
    ref = x @ w.T
    for rule in ("rn", "fp32"):
        for which in ("w", "x", "both"):
            xa = fq(x, rule) if which in ("x", "both") else x
            wa = fq(w, rule) if which in ("w", "both") else w
            y = xa @ wa.T
            print(f"{name:18s} {rule:5s} {which:5s} rel {((y - ref).norm() / ref.norm()).item():.4f} "
                  f"norm {(y.norm() / ref.norm() - 1).item() * 100:+.2f} %  "
                  f"act norm {(xa.norm() / x.norm() - 1).item() * 100:+.2f} % w norm {(wa.norm() / w.norm() - 1).item() * 100:+.2f} %")
