"""INT8 tier calibration on real VoxCPM2 captures (read only), like docs/research-scripts/
fp8-sm120/accuracy.py: every nn.Linear of a module replaced by a recipe's reference math
(kernels/quant.py, on the GPU: torch._int_mm), judged by the evaluator's own comparisons
(kernels.compare) in the near-lossless tier on the captured inputs, on redrawn inputs
(kernels.verify.perturb_: normal and mixed draws) and on the scaled inputs (x3, x0.01, x-1),
both with the redrawn bounds. Broken variants run fewer draws.

Usage: calib_int8.py dit|lm|alpha [normal_draws mixed_draws] | capture <capture_inputs.pt>
[normal_draws mixed_draws [cases]] (`alpha`: SmoothQuant alpha 0.3-0.8 on the LocDiT layer;
`capture`: any capture, e.g. the Qwen3-0.6B MLP decode target of runs/Qwen--Qwen3-0.6B)."""

import contextlib
import copy
import sys

import torch

from kernel_agent.kernels import quant
from kernel_agent.kernels.compare import compare_tensors, flatten
from kernel_agent.kernels.verify import SCALED, perturb_, scale_

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


# ------------------------------------------------------------------ recipes


def make_forward(lin, recipe, calib):
    w = lin.weight.detach()
    bias = lin.bias
    name = recipe
    if name == "bf16":
        return None
    if name == "fp8 W8A8 (per channel x per token)":
        q, s = quant.quantize_fp8(w)
        return lambda x: quant.fp8_w8a8_linear(x, q, s, bias)
    if name == "fp8 weights only":
        q, s = quant.quantize_fp8(w)
        wd = quant.dequantize_fp8(q, s, w.dtype)
        return lambda x: torch.nn.functional.linear(x, wd, bias)
    if name == "int8 weights only":
        q, s = quant.quantize_int8(w)
        return lambda x: quant.int8_weights_linear(x, q, s, bias)
    if name.startswith("int8 W8A8 + SmoothQuant"):
        alpha = float(name.split("a=")[1].split()[0])
        sm = quant.smoothquant_factors(calib[id(lin)], w, alpha)
        q, s = quant.quantize_int8(w, sm)
        return lambda x: quant.int8_w8a8_linear(x, q, s, bias, smooth=sm)
    q, s = quant.quantize_int8(w)
    if name == "int8 W8A8 (per channel x per token)":
        return lambda x: quant.int8_w8a8_linear(x, q, s, bias)
    # broken variants
    if name == "BUG weight scales x1.05":
        return lambda x: quant.int8_w8a8_linear(x, q, s * 1.05, bias)
    if name == "BUG neighbour channel's scale":
        return lambda x: quant.int8_w8a8_linear(x, q, s.roll(1), bias)
    if name == "BUG zeroed output channel":
        q0 = q.clone()
        q0[0] = 0
        return lambda x: quant.int8_w8a8_linear(x, q0, s, bias)

    def epilogue(x, xq, xs):
        acc = quant.int8_matmul(xq, q)
        y = acc.float() * xs[:, None] * s[None, :]
        if bias is not None:
            y = y + bias.float()
        return y.to(x.dtype).reshape(*x.shape[:-1], q.shape[0])

    def codes(a, xs):
        return torch.round(a / xs[:, None]).clamp(-127, 127).to(torch.int8)

    if name == "BUG first token's scale":
        def fwd(x):
            xq, xs = quant.quantize_int8_activations(x)
            a = x.reshape(-1, x.shape[-1]).float()
            xs = xs[:1].expand_as(xs).contiguous()
            return epilogue(x, codes(a, xs), xs)
        return fwd
    if name == "BUG per-tensor activation scale":
        def fwd(x):
            a = x.reshape(-1, x.shape[-1]).float()
            amax = a.abs().amax()
            xs = torch.full((a.shape[0],), float(amax) / 127 if amax > 0 else 1.0, device=a.device)
            return epilogue(x, codes(a, xs), xs)
        return fwd
    if name == "BUG static activation scale (calibrated)":
        static = float(calib[id(lin)].max()) / 127

        def fwd(x):
            a = x.reshape(-1, x.shape[-1]).float()
            xs = torch.full((a.shape[0],), static, device=a.device)
            return epilogue(x, codes(a, xs), xs)
        return fwd
    if name == "BUG clipped activations (99.9th percentile)":
        def fwd(x):
            a = x.reshape(-1, x.shape[-1]).float()
            xs = torch.quantile(a.abs(), 0.999, dim=1).clamp_min(1e-30) / 127
            return epilogue(x, codes(a, xs), xs)
        return fwd
    raise ValueError(name)


GOOD = [
    "int8 W8A8 (per channel x per token)",
    "int8 weights only",
    "int8 W8A8 + SmoothQuant a=0.4",
    "fp8 W8A8 (per channel x per token)",
    "fp8 weights only",
]
BUGS = [
    "BUG weight scales x1.05",
    "BUG neighbour channel's scale",
    "BUG first token's scale",
    "BUG per-tensor activation scale",
    "BUG zeroed output channel",
    "BUG static activation scale (calibrated)",
    "BUG clipped activations (99.9th percentile)",
]


@contextlib.contextmanager
def patched(module, recipe, calib, only=None):
    saved = []
    for n, m in module.named_modules():
        if isinstance(m, torch.nn.Linear) and (only is None or n in only):
            fwd = make_forward(m, recipe, calib)
            if fwd is not None:
                saved.append(m)
                m.forward = fwd
    try:
        yield
    finally:
        for m in saved:
            del m.forward


def calibrate(module, run):
    """Per nn.Linear (by id): max |x| per input channel over the calls of ``run``."""
    amax = {}
    hooks = []
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            def pre(mod, args):
                a = args[0].detach().reshape(-1, args[0].shape[-1]).float().abs().amax(0)
                amax[id(mod)] = torch.maximum(amax.get(id(mod), torch.zeros_like(a)), a)
            hooks.append(m.register_forward_pre_hook(pre))
    with torch.no_grad():
        run()
    for h in hooks:
        h.remove()
    return amax


# ------------------------------------------------------------------ judging


def outputs(out):
    return {k: v for k, v in flatten(out).items() if torch.is_tensor(v) and v.is_floating_point()}


def judge(ref, new, perturbed, scale=1.0):
    worst = {"ok": True, "cos": 1.0, "rel": 0.0, "elem": 0.0, "norm": 0.0, "why": ""}
    new_out = outputs(new)
    for name, r in outputs(ref).items():
        c = compare_tensors(name, r, new_out[name], tier=TIER, perturbed=perturbed,
                            input_scale=scale)
        if not c["ok"] and worst["ok"]:
            worst["why"] = f"{name}: {c.get('error', '')[:90]}"
        worst["ok"] &= c["ok"]
        worst["cos"] = min(worst["cos"], c.get("cosine", 1.0))
        worst["rel"] = max(worst["rel"], c.get("rel_l2", 0.0))
        worst["elem"] = max(worst["elem"], c.get("element_ratio", 0.0))
        worst["norm"] = max(worst["norm"], abs(c.get("norm_ratio", 1.0) - 1))
    return worst


def run_module(call, args, ctx, draws, fresh=None):
    """(captured, [redrawn], {scaled check: verdict}) of ctx vs the plain module."""
    def get(a):
        return copy.deepcopy(fresh(a) if fresh else a)

    with torch.no_grad():
        ref = call(get(args))
        with ctx():
            new = call(get(args))
        cap = judge(ref, new, False)
        red = []
        for kind, n in (("normal", draws[0]), ("mix", draws[1])):
            for seed in range(n):
                g = torch.Generator(device="cuda").manual_seed(1000 + seed)
                a0 = get(args)
                perturb_(a0, g, kind)
                a1 = copy.deepcopy(a0)
                ref = call(a0)
                with ctx():
                    new = call(a1)
                red.append(judge(ref, new, True))
        scaled = {}
        for name, factor in SCALED:
            a0 = get(args)
            scale_(a0, factor)
            a1 = copy.deepcopy(a0)
            ref = call(a0)
            with ctx():
                new = call(a1)
            scaled[name] = judge(ref, new, True, factor)
    return cap, red, scaled


def fmt(cap, red, scaled):
    fails = sum(not r["ok"] for r in red)
    sc = " ".join(f"{k.replace('scaled_', '').replace('sign_flipped', 'x-1')}:"
                  f"{'ok' if v['ok'] else 'FAIL'}" for k, v in scaled.items())
    text = (f"cap {'pass' if cap['ok'] else 'FAIL'} cos {cap['cos']:.5f} rel {cap['rel']:.4f} "
            f"elem {cap['elem']:.2f} norm {cap['norm'] * 100:.2f}% | redrawn {fails}/{len(red)} fail, "
            f"max rel {max(r['rel'] for r in red):.4f} elem {max(r['elem'] for r in red):.2f} "
            f"norm {max(r['norm'] for r in red) * 100:.2f}% | {sc} (worst rel "
            f"{max(v['rel'] for v in scaled.values()):.4f} elem "
            f"{max(v['elem'] for v in scaled.values()):.2f} norm "
            f"{max(v['norm'] for v in scaled.values()) * 100:.2f}%)")
    why = cap["why"] or next((r["why"] for r in red if not r["ok"]), "") or next(
        (f"{k}: {v['why']}" for k, v in scaled.items() if not v["ok"]), "")
    return text + (f"\n      first failure: {why}" if why else "")


# ------------------------------------------------------------------ modules


def linear_table(layer, call, calib, recipes):
    seen = {}

    def keep(mod, a, o, n):
        seen.setdefault(n, (a[0].detach(), o.detach()))

    hooks = [m.register_forward_hook(lambda mod, a, o, n=n: keep(mod, a, o, n))
             for n, m in layer.named_modules() if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        call()
    for h in hooks:
        h.remove()
    print("| linear (rows, act crest) | " + " | ".join(recipes) + " |")
    for n, (x, y) in seen.items():
        lin = dict(layer.named_modules())[n]
        cells = []
        for r in recipes:
            with torch.no_grad(), patched(layer, r, calib, only=[n]):
                new = lin(x)
            c = compare_tensors(n, y, new, tier=TIER)
            cells.append(f"{c['rel_l2']:.4f}" + ("" if c["ok"] else f" FAIL(norm {abs(c.get('norm_ratio', 1) - 1) * 100:.1f}%, elem {c.get('element_ratio', 0):.2f})"))
        a = x.reshape(-1, x.shape[-1]).float()
        rms = a.pow(2).mean(1).sqrt()
        crest = float((a.abs().amax(1) / rms.clamp_min(1e-30))[rms > 0].max())
        print(f"| {n} ({a.shape[0]}, {crest:.0f}) | " + " | ".join(cells) + " |")


def dit(draws):
    d = torch.load(DIT, map_location="cpu", weights_only=False)
    layer = d["module"].cuda().eval()
    cases = [to_cuda(list(c["args"])) for c in d["cases"]]
    print(f"LocDiT layer: cases {[tuple(c[0].shape) for c in cases]}")
    for i, test in enumerate(cases[:2]):
        other = cases[1 - i]
        calib = calibrate(layer, lambda o=other: layer(*o))  # SmoothQuant / static: the other call
        rows = test[0].reshape(-1, test[0].shape[-1]).shape[0]
        if i == 0:
            print("\n## each nn.Linear alone (captured input, rel L2 vs bf16)")
            linear_table(layer, lambda: layer(*test), calib, GOOD[:4])
        print(f"\n## LocDiT decoder layer, M = {rows} (outputs: hidden, k, v)")
        for recipe in GOOD + BUGS:
            n = draws if recipe in GOOD else (draws[0] // 4, draws[1] // 4)
            res = run_module(lambda a: layer(*a), test, lambda r=recipe: patched(layer, r, calib), n)
            print(f"{recipe:46s} {fmt(*res)}", flush=True)


def lm(draws):
    d = torch.load(LM, map_location="cpu", weights_only=False)
    layer = d["module"].cuda().eval()
    cases = [to_cuda(list(c["args"])) for c in d["cases"]]

    def fresh(args):
        return [args[0], args[1], args[2], (args[3][0].clone(), args[3][1].clone())]

    print("\n## base-LM decoder layer, decode (M = 1), forward_step with its KV cache")
    for i, args in enumerate(cases):
        others = [c for j, c in enumerate(cases) if j != i]
        calib = calibrate(layer, lambda: [layer.forward_step(*fresh(c)) for c in others])
        call = lambda a: layer.forward_step(*a)  # noqa: E731
        print(f"\n### decode case {i} (position {int(args[2])})")
        if i == 1:
            linear_table(layer, lambda: call(fresh(args)), calib, GOOD[:4])
        for recipe in GOOD + BUGS:
            n = draws if recipe in GOOD else (2, 1)
            res = run_module(call, args, lambda r=recipe: patched(layer, r, calib), n, fresh=fresh)
            print(f"{recipe:46s} {fmt(*res)}", flush=True)


def alpha(draws):
    d = torch.load(DIT, map_location="cpu", weights_only=False)
    layer = d["module"].cuda().eval()
    cases = [to_cuda(list(c["args"])) for c in d["cases"]]
    for i, test in enumerate(cases[:2]):
        calib = calibrate(layer, lambda o=cases[1 - i]: layer(*o))
        print(f"\n## M = {test[0].reshape(-1, test[0].shape[-1]).shape[0]}")
        for a in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            r = f"int8 W8A8 + SmoothQuant a={a}"
            res = run_module(lambda x: layer(*x), test, lambda r=r: patched(layer, r, calib), draws)
            print(f"{r:40s} {fmt(*res)}", flush=True)


def generic(path, draws, max_cases):
    from kernel_agent.profiling.capture import load_capture

    cap = load_capture(path)
    module = cap["module"].cuda().eval()

    def dev(v):
        if torch.is_tensor(v):
            return v.cuda()
        if isinstance(v, dict):
            return {k: dev(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return type(v)(dev(x) for x in v)
        return v

    cases = [
        (dev(list(c["args"])), dev(dict(c.get("kwargs") or {})), c.get("method") or "forward")
        for c in cap["cases"]
    ][:max_cases]
    print(path.split("/runs/")[-1], [tuple(a[0].shape) for a, _, _ in cases])

    def fn_of(method):
        return module if method == "forward" else getattr(module, method)

    for i, (args, kwargs, method) in enumerate(cases):
        others = [c for j, c in enumerate(cases) if j != i] or [cases[i]]
        calib = calibrate(
            module,
            lambda o=others: [fn_of(m)(*copy.deepcopy(a), **copy.deepcopy(k)) for a, k, m in o],
        )
        call = lambda ak, fn=fn_of(method): fn(*ak[0], **ak[1])  # noqa: E731
        print(f"\n### case {i}")
        if i == 0:
            linear_table(module, lambda: call(copy.deepcopy((args, kwargs))), calib, GOOD[:4])
        for recipe in GOOD + BUGS:
            n = draws if recipe in GOOD else (2, 1)
            res = run_module(call, (args, kwargs), lambda r=recipe: patched(module, r, calib), n)
            print(f"{recipe:46s} {fmt(*res)}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "capture":
        draws = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (8, 4)
        generic(sys.argv[2], draws, int(sys.argv[5]) if len(sys.argv) > 5 else 3)
    else:
        draws = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (20, 10)
        {"dit": dit, "lm": lm, "alpha": alpha}[which](draws)
