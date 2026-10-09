"""Which systematic errors does the calibrated check newly accept? Rejection rates, before
(plain) and after (with the rounding spread), of the reference's output with a bias of a
fraction of its RMS, a scale error, one layer skipped, and of the previous draw's output
(the first draw's "previous output" is its own: 99 of 100 is every draw).

    python systematic.py 100 > systematic.txt
"""

from __future__ import annotations

import sys

import torch

from kernel_agent.kernels import bench, compare, verify
from kernel_agent.selftest import NormGemvChain


def chain(layers, hidden=1024):
    torch.manual_seed(0)
    m = NormGemvChain(hidden, layers).cuda().to(torch.bfloat16)
    with torch.no_grad():
        for name, w in m.named_parameters():
            w.normal_(1.0, 0.1) if name.startswith("norms.") else w.normal_(0.0, hidden**-0.5)
    return m.eval()


def main():
    draws = int(sys.argv[1])
    for layers in (1, 8, 28):
        for rows in (1, 64):
            ref = chain(layers)
            captured = torch.randn(rows, 1024, device="cuda", dtype=torch.bfloat16)
            bad = {}
            ratios = {}
            for i in range(draws):
                args, kwargs = bench._perturbed_copy((captured,), {})
                with torch.inference_mode():
                    r = ref(*args)
                    skipped = args[0]
                    for norm, layer in zip(ref.norms[:-1], ref.layers[:-1], strict=True):
                        skipped = skipped + layer(norm(skipped))
                alt = verify.rerounded_call(ref, args, kwargs)[0]
                rms = r.float().pow(2).mean().sqrt()
                prev = bad.setdefault("_prev", r)
                variants = {
                    "bias 1%": (r.float() + 0.01 * rms).to(r.dtype),
                    "bias 2%": (r.float() + 0.02 * rms).to(r.dtype),
                    "bias 4%": (r.float() + 0.04 * rms).to(r.dtype),
                    "scale 1.01": (r.float() * 1.01).to(r.dtype),
                    "scale 1.03": (r.float() * 1.03).to(r.dtype),
                    "scale 1.05": (r.float() * 1.05).to(r.dtype),
                    "noise 1%": (r.float() + 0.01 * rms * torch.randn_like(r.float())).to(r.dtype),
                    "noise 2%": (r.float() + 0.02 * rms * torch.randn_like(r.float())).to(r.dtype),
                    "last layer skipped": skipped,
                    "previous output": prev,
                }
                bad["_prev"] = r
                for name, v in variants.items():
                    b = compare.compare_tensors("o", r, v, perturbed=True)
                    a = compare.compare_tensors("o", r, v, perturbed=True, rerounded=alt)
                    s = bad.setdefault(name, [0, 0])
                    s[0] += not b["ok"]
                    s[1] += not a["ok"]
                    ratios.setdefault(name, []).append(a.get("error_over_spread"))
            bad.pop("_prev")
            print(f"layers={layers} rows={rows} ({draws} draws): rejected before/after")
            for name, (b, a) in bad.items():
                med = sorted(ratios[name])[len(ratios[name]) // 2]
                print(f"   {name:20s} {b:4d} {a:4d}   error/spread median {med}")


if __name__ == "__main__":
    main()
