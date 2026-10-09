"""The calibration of the rounding spread (#250): per redrawn input of a deep bf16 chain, how
far honest candidates are from eager in units of the reference's own rounding spread
(``kernels.verify.Rerounding``: the reference with every op's rounding redrawn), and how far
the outputs the check must reject are.

Chains: ``selftest.NormGemvChain`` ("norm": RMSNorm -> GEMV -> residual) with the megakernel
example's three modes, and ``selftest.GemvChain`` ("plain") with the PDL example; both with
``torch.compile`` of the reference. Per draw and candidate: elements outside the plain
tolerance (``mis``), the RMS error relative to the spread (``d_over_s``), the ``K`` that puts
every element (``k0``) / all but one (``k1``) within ``tol + K x spread``, eager's own error
against an fp32 recompute (``s32_rel``), and for the cheats (previous draw's output, last
layer skipped, a stale 16-element tile, one stale element) the ``K`` / the widening in units
of the RMS (``*_relcap``) at which they would pass. ``summarize.py`` prints the table.

    python calibrate.py 200 2,4,8,16,28 > calibrate.json  # seeded: 2.3 MB, not kept
    python summarize.py calibrate.json > calibrate.txt
"""

from __future__ import annotations

import json
import sys
import time

import torch

from kernel_agent import toolchain
from kernel_agent.agent.prompts import EXAMPLES_DIR
from kernel_agent.kernels.evaluate import load_candidate_module
from kernel_agent.kernels.verify import Rerounding
from kernel_agent.native import project
from kernel_agent.selftest import GemvChain, NormGemvChain


def chain(kind, hidden, layers, seed=0):
    torch.manual_seed(seed)
    if kind == "norm":
        m = NormGemvChain(hidden, layers).cuda().to(torch.bfloat16)
        with torch.no_grad():
            for name, w in m.named_parameters():
                w.normal_(1.0, 0.1) if name.startswith("norms.") else w.normal_(0.0, hidden**-0.5)
    else:
        m = GemvChain(hidden, layers).cuda().to(torch.bfloat16)
        with torch.no_grad():
            for w in m.parameters():
                w.normal_(0.0, hidden**-0.5)
    return m.eval()


def main():
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    depths = [int(d) for d in sys.argv[2].split(",")] if len(sys.argv) > 2 else [8]
    toolchain.setup()
    mk = project.import_project(EXAMPLES_DIR / "native_megakernel")
    pdl = load_candidate_module(EXAMPLES_DIR / "cuda_pdl_gemv_chain.py")
    gen = torch.Generator(device="cuda").manual_seed(1234)
    out = {}
    for layers in depths:
        for kind in ("norm", "plain"):
            ref = chain(kind, 1024, layers)
            ref32 = chain(kind, 1024, layers).float()
            if kind == "norm":
                cands = {m: mk.build(ref, mode=m) for m in ("megakernel", "graph_pdl", "coop_barrier")}
                skip = NormGemvChain(1024, layers - 1).cuda().to(torch.bfloat16).eval()
                skip.norms = ref.norms[: layers - 1]
                skip.layers = ref.layers[: layers - 1]
            else:
                cands = {"pdl": pdl.build(ref)}
                skip = GemvChain(1024, layers - 1).cuda().to(torch.bfloat16).eval()
                skip.layers = ref.layers[: layers - 1]
            cands["compiled"] = torch.compile(ref, mode="max-autotune-no-cudagraphs")
            rows = {c: [] for c in cands}
            prev = None
            t0 = time.time()
            for i in range(draws):
                x = torch.randn(1, 1024, device="cuda", dtype=torch.bfloat16)
                with torch.inference_mode():
                    r = ref(x)
                    with Rerounding(1000 + 2 * i + layers):
                        alt = ref(x)
                    with Rerounding(5000 + 2 * i + layers):
                        alt2 = ref(x)
                    r32 = ref32(x.float())
                    sk = skip(x)
                    outs = {c: f(x).clone() for c, f in cands.items()}
                torch.cuda.synchronize()
                a = r.float()
                tol = 0.02 + 0.02 * a.abs()
                rms = a.pow(2).mean().sqrt().item()
                s_mca = (alt.float() - a).pow(2).mean().sqrt().item()
                s_mca2 = (alt2.float() - a).pow(2).mean().sqrt().item()
                s32 = (a - r32).pow(2).mean().sqrt().item()
                mca_mis = int(((alt.float() - a).abs() > tol).sum())
                for c, o in outs.items():
                    d = (o.float() - a).abs().flatten()
                    excess = ((d - tol.flatten()) / max(s_mca, 1e-12)).sort().values
                    cheats = {}
                    if prev is not None:
                        stale = o.float().clone().flatten()
                        stale[16:32] = prev[c].float().flatten()[16:32]
                        one = o.float().clone().flatten()
                        one[100] = prev[c].float().flatten()[100]
                        cheats = {
                            "cached": prev[c].float().flatten(),
                            "skip_last": sk.float().flatten(),
                            "stale_tile": stale,
                            "stale_one": one,
                        }
                    kpass = {}
                    for nm, v in cheats.items():
                        dc = (v - a.flatten()).abs()
                        need_mis = ((dc - tol.flatten()) / s_mca).sort().values[-2].item()
                        need_out = ((dc / 10 - tol.flatten()) / s_mca).max().item()
                        kpass[nm] = round(max(need_mis, need_out), 1)
                        kpass[nm + "_relcap"] = round(
                            max(
                                ((dc - tol.flatten()) / rms).sort().values[-2].item(),
                                ((dc / 10 - tol.flatten()) / rms).max().item(),
                            ),
                            4,
                        )
                    mis = int((d > tol.flatten()).sum())
                    rows[c].append(
                        dict(
                            mis=mis,
                            ratio=round((d / tol.flatten()).max().item(), 3),
                            rel=round((d.pow(2).mean().sqrt() / rms).item(), 5),
                            rms=round(rms, 3),
                            s_mca_rel=round(s_mca / rms, 5),
                            s_mca2_rel=round(s_mca2 / rms, 5),
                            s32_rel=round(s32 / rms, 5),
                            cand32_rel=round(((o.float() - r32).pow(2).mean().sqrt() / rms).item(), 5),
                            mca_mis=mca_mis,
                            d_over_s=round((d.pow(2).mean().sqrt() / s_mca).item(), 3),
                            k0=round(excess[-1].item(), 2),
                            k1=round(excess[-2].item(), 2),
                            kpass=kpass,
                        )
                    )
                prev = outs
            print(f"{kind} L={layers}: {draws} draws in {time.time() - t0:.0f}s", file=sys.stderr)
            out[f"{kind}{layers}"] = rows
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
