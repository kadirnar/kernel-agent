"""Before/after false-rejection rates of the timed-output check on deep bf16 chains (#250).

Each draw: redraw the captured input as the timed check does (bench._perturbed_copy), run a
candidate on it, and judge the kept call with bench.check_timed_output: "before" with the
rounding spread disabled (verify.rerounded_call -> None: the old comparator), "after" as is.
Cheats on the same draws must fail "after": the previous draw's output (cached), a stale
16-row tile, the last layer skipped.

    python timed_check.py norm:8:3000 > timed_norm8.json
    python timed_check.py norm:4:1000,plain:8:1000,norm:16:1000,plain:28:1000,norm:28:1000 \\
        > timed_more.json
"""

from __future__ import annotations

import json
import sys
import time

import torch

from kernel_agent import toolchain
from kernel_agent.agent.prompts import EXAMPLES_DIR
from kernel_agent.kernels import bench, verify
from kernel_agent.kernels.evaluate import load_candidate_module
from kernel_agent.native import project
from kernel_agent.selftest import GemvChain, NormGemvChain

REAL = verify.rerounded_call


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


def judge(ref, kept, plain):
    verify.rerounded_call = (lambda *a, **k: None) if plain else REAL
    try:
        return bench.check_timed_output(ref, kept)["failures"]
    finally:
        verify.rerounded_call = REAL


def main():
    plan = [(p.split(":")[0], int(p.split(":")[1]), int(p.split(":")[2])) for p in sys.argv[1].split(",")]
    toolchain.setup()
    mk = project.import_project(EXAMPLES_DIR / "native_megakernel")
    pdl = load_candidate_module(EXAMPLES_DIR / "cuda_pdl_gemv_chain.py")
    report = {}
    for kind, layers, draws in plan:
        ref = chain(kind, 1024, layers)
        if kind == "norm":
            cands = {m: mk.build(ref, mode=m) for m in ("megakernel", "graph_pdl", "coop_barrier")}
            skip = NormGemvChain(1024, layers - 1).cuda().to(torch.bfloat16).eval()
            skip.norms, skip.layers = ref.norms[: layers - 1], ref.layers[: layers - 1]
        else:
            cands = {"pdl": pdl.build(ref)}
            skip = GemvChain(1024, layers - 1).cuda().to(torch.bfloat16).eval()
            skip.layers = ref.layers[: layers - 1]
        cands["compiled"] = torch.compile(ref, mode="max-autotune-no-cudagraphs")
        captured = torch.randn(1, 1024, device="cuda", dtype=torch.bfloat16)
        stats = {c: {"before": 0, "after": 0} for c in cands}
        cheats = {"cached": 0, "stale_tile": 0, "skip_last": 0, "n": 0}
        prev = None
        t0 = time.time()
        for i in range(draws):
            args, kwargs = bench._perturbed_copy((captured,), {})
            pre = bench._snapshot((args, kwargs))
            outs = {}
            with torch.inference_mode():
                for c, f in cands.items():
                    outs[c] = f(*args).clone()
                sk = skip(*args)
            torch.cuda.synchronize()
            for c, out in outs.items():
                kept = {"iteration": i, "pre": pre, "post": pre, "output": out}
                before = judge(ref, kept, plain=True)
                stats[c]["before"] += bool(before)
                if before:
                    after = judge(ref, kept, plain=False)
                    stats[c]["after"] += bool(after)
                    if after:
                        stats[c].setdefault("after_errors", []).append(after[0].get("error"))
            if prev is not None and i % 10 == 0:  # cheats: every 10th draw (each fails anyway)
                c0 = next(iter(cands))
                stale = outs[c0].clone()
                stale[:, 16:32] = prev[:, 16:32]
                for name, out in (("cached", prev), ("stale_tile", stale), ("skip_last", sk)):
                    kept = {"iteration": i, "pre": pre, "post": pre, "output": out}
                    cheats[name] += bool(judge(ref, kept, plain=False))
                cheats["n"] += 1
            prev = outs[next(iter(cands))]
        took = time.time() - t0
        report[f"{kind}{layers}"] = {"draws": draws, "seconds": round(took), "candidates": stats,
                                     "cheats_rejected": cheats}
        print(f"{kind}{layers}: {json.dumps(report[f'{kind}{layers}'])}", file=sys.stderr)
    json.dump(report, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
