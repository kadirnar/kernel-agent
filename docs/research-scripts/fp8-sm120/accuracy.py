"""Accuracy of FP8 recipes against bf16 at module level, on the VoxCPM2 captures (read
only): the LocDiT decoder layer 0 at M = 352 (dit_layer__fp8_w8a8) and the base-LM decoder
layer 0 at decode, M = 1 (lm_step_fp8, with its KV cache). Judged by the evaluator's own
near-lossless tier (kernels.compare.compare_tensors) on the captured inputs and on inputs
redrawn the evaluator's way (kernels.verify.perturb_), 20 normal + 10 mixed draws."""

import copy
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fp8lib import RECIPES, calibrate, patched_linears, patched_sdpa  # noqa: E402

from kernel_agent.kernels.compare import compare_tensors, flatten
from kernel_agent.kernels.verify import perturb_

RUNS = "/home/kadir/kadir_projects/kernel-agent/runs/openbmb--VoxCPM2"
DIT = f"{RUNS}/20261006-004718-retest2/targets/dit_layer__fp8_w8a8/capture_inputs.pt"
LM = f"{RUNS}/20261005-192504/targets/lm_step_fp8/capture_inputs.pt"
TIER = "near-lossless"
torch.backends.cuda.matmul.allow_tf32 = False


def to_cuda(v):
    if torch.is_tensor(v):
        return v.cuda()
    if isinstance(v, (list, tuple)):
        return type(v)(to_cuda(x) for x in v)
    return v


def outputs(out):
    return {k: v for k, v in flatten(out).items() if torch.is_tensor(v) and v.is_floating_point()}


def judge(ref, new, perturbed):
    worst = {"ok": True, "cos": 1.0, "rel": 0.0, "elem": 0.0, "norm": 0.0}
    for name, r in outputs(ref).items():
        c = compare_tensors(name, r, outputs(new)[name], tier=TIER, perturbed=perturbed)
        worst["ok"] &= c["ok"]
        worst["cos"] = min(worst["cos"], c.get("cosine", 1.0))
        worst["rel"] = max(worst["rel"], c.get("rel_l2", 0.0))
        worst["elem"] = max(worst["elem"], c.get("element_ratio", 0.0))
        worst["norm"] = max(worst["norm"], abs(c.get("norm_ratio", 1.0) - 1))
    return worst


def run_module(call, args, ctx_new, draws=(20, 10), fresh=None):
    """(captured verdict, redrawn verdicts) of ctx_new vs the plain module."""
    with torch.no_grad():
        a = fresh(args) if fresh else args
        ref = call(a)
        a = fresh(args) if fresh else args
        with ctx_new():
            new = call(a)
        cap = judge(ref, new, False)
        red = []
        for kind, n in (("normal", draws[0]), ("mix", draws[1])):
            for seed in range(n):
                g = torch.Generator(device="cuda").manual_seed(1000 + seed)
                a0 = copy.deepcopy(fresh(args) if fresh else args)
                perturb_(a0, g, kind)
                a1 = copy.deepcopy(a0)
                ref = call(a0)
                with ctx_new():
                    new = call(a1)
                red.append(judge(ref, new, True))
    return cap, red


def fmt(cap, red):
    fails = sum(not r["ok"] for r in red)
    return (f"cap: {'pass' if cap['ok'] else 'FAIL'} cos {cap['cos']:.5f} rel {cap['rel']:.4f} "
            f"elem {cap['elem']:.2f} norm {cap['norm'] * 100:.2f}% | redrawn: {fails}/{len(red)} fail, "
            f"max rel {max(r['rel'] for r in red):.4f} max elem {max(r['elem'] for r in red):.2f} "
            f"max norm {max(r['norm'] for r in red) * 100:.2f}%")


TABLE = ["FP8 weights only (per channel)", "W8A8 tensorwise (per tensor x per tensor)",
         "W8A8 rowwise (per channel x per token)", "W8A8 per channel x 1x128 groups",
         "W8A8 blockwise (128x128 x 1x128)", "MXFP8 OCP floor scale (1x32 x 1x32)",
         "MXFP8 ceil scale (1x32 x 1x32)", "W8A8 static per-tensor activations (calibrated)",
         "SmoothQuant a=0.5 + rowwise"]


def linear_table(layer, call, calib):
    """Each nn.Linear alone on its captured input: recipe output vs the bf16 output."""
    seen = {}
    def keep(mod, a, o, n):
        seen.setdefault(n, (a[0].detach(), o.detach()))

    hooks = [m.register_forward_hook(lambda mod, a, o, n=n: keep(mod, a, o, n))
             for n, m in layer.named_modules() if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        call()
    for h in hooks:
        h.remove()
    short = {r: r.split(" (")[0].replace("W8A8 ", "") for r in TABLE}
    print("| linear (rows) | " + " | ".join(short[r] for r in TABLE) + " |")
    print("|---" * (len(TABLE) + 1) + "|")
    for n, (x, y) in seen.items():
        lin = dict(layer.named_modules())[n]
        cells = []
        for r in TABLE:
            with torch.no_grad(), patched_linears(layer, r, calib, only=[n]):
                new = lin(x)
            c = compare_tensors(n, y, new, tier=TIER)
            cells.append(f"{c['rel_l2']:.3f}{'' if c['ok'] else ' **fail** (norm ' + format(abs(c.get('norm_ratio', 1) - 1) * 100, '.1f') + '%)'}")
        rows = x.reshape(-1, x.shape[-1]).shape[0]
        print(f"| {n} ({rows}) | " + " | ".join(cells) + " |")


def dit():
    d = torch.load(DIT, map_location="cpu", weights_only=False)
    layer = d["module"].cuda().eval()
    test = to_cuda(list(d["cases"][0]["args"]))  # [32, 11, 1024], the timed case
    calib_args = to_cuda(list(d["cases"][1]["args"]))  # [16, 11, 1024], another run
    calib = calibrate(layer, lambda: layer(*calib_args))
    x = test[0].float().reshape(-1, 1024)
    x = x[x.abs().amax(1) > 0]
    print(f"LocDiT layer 0 input: {x.shape[0]} non-zero rows of 352, crest (amax/RMS) per token "
          f"median {float((x.abs().amax(1) / x.pow(2).mean(1).sqrt()).median()):.1f}")
    for name in ("self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj"):
        a = calib[name]["amax"]
        top = torch.topk(a, 3)
        print(f"  {name} input channel amax: median {float(a.median()):.3f}, top "
              f"{[(int(i), round(float(v), 1)) for v, i in zip(top.values, top.indices)]}")
    print("\n## LocDiT layer 0, each nn.Linear alone (captured input, rel L2 vs bf16)")
    linear_table(layer, lambda: layer(*test), calib)
    print("\n## LocDiT decoder layer 0, M = 352 (outputs: hidden, k, v)")
    for recipe in RECIPES:
        cap, red = run_module(lambda a: layer(*a), test,
                              lambda r=recipe: patched_linears(layer, r, calib))
        print(f"{recipe:48s} {fmt(cap, red)}")
    print("\n## LocDiT layer 0, attention in FP8 (GEMMs bf16)")
    for mode in ("none", "qk", "qkpv"):
        rec = []
        with torch.no_grad(), patched_sdpa(mode, record=rec):
            layer(*test)
        cap, red = run_module(lambda a: layer(*a), test, lambda m=mode: patched_sdpa(m))
        print(f"attention {mode:10s} (SDPA output rel L2 vs torch {rec[0]:.4f}) {fmt(cap, red)}")
    print("\n## LocDiT layer 0, rowwise W8A8 GEMMs + FP8 attention (qkpv)")

    def both():
        import contextlib

        st = contextlib.ExitStack()
        st.enter_context(patched_linears(layer, "W8A8 rowwise (per channel x per token)", calib))
        st.enter_context(patched_sdpa("qkpv"))
        return st

    cap, red = run_module(lambda a: layer(*a), test, both)
    print(f"{'rowwise + qkpv':48s} {fmt(cap, red)}")


def lm():
    d = torch.load(LM, map_location="cpu", weights_only=False)
    layer = d["module"].cuda().eval()
    cases = [to_cuda(list(c["args"])) for c in d["cases"]]

    def fresh(args):  # forward_step writes the KV cache: a new copy per call
        return [args[0], args[1], args[2], (args[3][0].clone(), args[3][1].clone())]

    # calibration on the other two decode steps for each tested one
    print("\n## base-LM decoder layer 0, decode (M = 1), forward_step with its KV cache")
    for i, args in enumerate(cases):
        pos = int(args[2])
        k, v = args[3][0][:, :, : pos + 1].float(), args[3][1][:, :, : pos + 1].float()
        others = [c for j, c in enumerate(cases) if j != i]
        calib = calibrate(layer, lambda: [layer.forward_step(*fresh(c)) for c in others])
        print(f"\n### decode step {d['cases'][i]['decode_step'] + 1} (position {pos}, "
              f"{pos + 1} cached tokens; K amax {float(k.abs().max()):.1f} rms "
              f"{float(k.pow(2).mean().sqrt()):.2f}, V amax {float(v.abs().max()):.2f} rms "
              f"{float(v.pow(2).mean().sqrt()):.3f})")
        call = lambda a: layer.forward_step(*a)  # noqa: E731
        if i == 1:
            print("\nEach nn.Linear alone (captured input, rel L2 vs bf16):")
            linear_table(layer, lambda: call(fresh(args)), calib)
        for recipe in RECIPES:
            cap, red = run_module(call, args, lambda r=recipe: patched_linears(layer, r, calib),
                                  draws=(6, 4), fresh=fresh)
            print(f"{recipe:48s} {fmt(cap, red)}")
        kv_cal = (k.abs().max() / 448, v.abs().max() / 448)
        one = (torch.tensor(1.0, device="cuda"), torch.tensor(1.0, device="cuda"))
        for label, mode, sc in (("KV fp32 math (no quant)", "none", None),
                                ("KV e4m3 per-tensor, calibrated scale", "kv_tensor", kv_cal),
                                ("KV e4m3 per-tensor, scale 1.0 (uncalibrated)", "kv_tensor", one),
                                ("KV e4m3 per token and head", "kv_token", None)):
            rec = []
            with torch.no_grad(), patched_sdpa(mode, sc, record=rec):
                call(fresh(args))
            cap, red = run_module(call, args, lambda m=mode, s=sc: patched_sdpa(m, s),
                                  draws=(6, 4), fresh=fresh)
            print(f"{label:48s} (SDPA rel L2 {rec[0]:.4f}) {fmt(cap, red)}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "dit"):
        dit()
    if which in ("all", "lm"):
        lm()
