"""Calibrate the LLM near-lossless gate (teacher-forced KL / top-1 / NLL) on Qwen3-0.6B.

Honest numerics changes (bf16 rounding, FP8 weight-only fake quant) vs broken ones, on the
14 natural prompts of LLMWorkload.perceptual_samples() (inputs version 2)."""

from __future__ import annotations

import contextlib
import json
import sys
import time

import torch
from torch import nn

from kernel_agent import toolchain
from kernel_agent.workloads import create_workload, perceptual
from kernel_agent.workloads.base import WorkloadSpec, cosine
from kernel_agent.workloads.quality import perturb_linears

toolchain.setup()
spec = WorkloadSpec(repo_id="Qwen/Qwen3-0.6B", modality="llm")
wl = create_workload(spec)
wl.load()
samples = wl.perceptual_samples()
print("samples", [s["sample"] for s in samples], flush=True)

t0 = time.time()
ref_gen, _ = perceptual.generate(wl, samples)
ref_scores, _ = perceptual.score(wl, ref_gen)
print(f"baseline generated + scored in {time.time() - t0:.1f}s", flush=True)
saved = {k: v.detach().clone() for k, v in wl.model.state_dict().items()}


def restore() -> None:
    with torch.no_grad():
        for k, v in wl.model.state_dict().items():
            v.copy_(saved[k])


def decoder_linears():
    for name, mod in wl.model.named_modules():
        if isinstance(mod, nn.Linear) and ".layers." in name:
            yield name, mod


def fake_fp8(per_channel: bool, scale_error: float = 1.0) -> None:
    with torch.no_grad():
        for _, mod in decoder_linears():
            w = mod.weight.float()
            amax = w.abs().amax(1, keepdim=True) if per_channel else w.abs().amax()
            s = amax.clamp_min(1e-12) / 448
            q = (w / s).to(torch.float8_e4m3fn).float() * s * scale_error
            mod.weight.copy_(q.to(mod.weight.dtype))


def fake_int4(group: int | None) -> None:
    with torch.no_grad():
        for _, mod in decoder_linears():
            w = mod.weight.float()
            shape = w.shape
            if group:
                w = w.reshape(shape[0], -1, group)
            amax = w.abs().amax(-1, keepdim=True).clamp_min(1e-12)
            s = amax / 7
            q = (w / s).round().clamp(-8, 7) * s
            mod.weight.copy_(q.reshape(shape).to(mod.weight.dtype))


def rmsnorm_eps(eps: float) -> None:
    for mod in wl.model.modules():
        if "RMSNorm" in type(mod).__name__:
            if hasattr(mod, "variance_epsilon"):
                mod.variance_epsilon = eps
            if hasattr(mod, "eps"):
                mod.eps = eps


def drop_kv_head() -> None:
    with torch.no_grad():
        for name, mod in decoder_linears():
            if name.endswith("k_proj"):
                d = mod.weight.shape[0] // wl.model.config.num_key_value_heads
                mod.weight[:d] = 0


def evaluate(label: str, ctx=None) -> dict:
    t = time.time()
    with ctx if ctx is not None else contextlib.nullcontext():
        gen, _ = perceptual.generate(wl, samples)
        for g, s in zip(gen, ref_scores, strict=True):
            g["reference"] = s
        scores, _ = perceptual.score(wl, gen)
    cmp = perceptual.compare_llm(ref_scores, scores)
    rows = []
    for s, r, c, rg, g in zip(samples, ref_scores, scores, ref_gen, gen, strict=True):
        a, b = rg["output"]["tokens"][0].tolist(), g["output"]["tokens"][0].tolist()
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), len(a))
        margins = rg["output"]["margins"][0]
        exact = {}
        for tie in (0.0, 0.25, 0.5, 1.0):
            with wl.with_options({"near_tie": tie}):
                exact[str(tie)] = wl.compare(rg["output"], g["output"]).passed
        rows.append(
            {
                "sample": s["sample"],
                "prefix": first,
                "margin_at_div": round(float(margins[first]), 4) if first < len(a) else None,
                "exact_pass": exact,
                "first_cos": round(
                    cosine(rg["output"]["first_logits"], g["output"]["first_logits"]), 6
                ),
                "kl": c["kl"],
                "kl_max": c["kl_max"],
                "top1": c["top1"],
                "nll_inc": round(c["nll"] - r["nll"], 4),
            }
        )
    out = {
        "variant": label,
        "passed": cmp.passed,
        "reason": cmp.reason,
        **cmp.metrics,
        "min_prefix": min(r["prefix"] for r in rows),
        "prefix_lt16": sum(r["prefix"] < 16 for r in rows),
        "min_first_cos": min(r["first_cos"] for r in rows),
        "exact_pass": {
            t: sum(r["exact_pass"][t] for r in rows) for t in ("0.0", "0.25", "0.5", "1.0")
        },
        "div_margins": sorted(r["margin_at_div"] for r in rows if r["margin_at_div"] is not None),
        "rows": rows,
        "s": round(time.time() - t, 1),
    }
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}), flush=True)
    return out


def broken_loop() -> dict:
    """A decode loop that writes the wrong text: the honest model scores a continuation
    whose tokens are shifted by one position (the right words in the wrong places)."""
    gen = []
    for rg, s in zip(ref_gen, ref_scores, strict=True):
        tokens = rg["output"]["tokens"].clone()
        tokens[0] = torch.roll(tokens[0], 1)
        gen.append({"options": rg["options"], "output": {"tokens": tokens}, "reference": s})
    scores, _ = perceptual.score(wl, gen)
    cmp = perceptual.compare_llm(ref_scores, scores)
    out = {"variant": "broken loop (continuation rolled by 1)", "passed": cmp.passed, "reason": cmp.reason, **cmp.metrics}
    print(json.dumps(out), flush=True)
    return out


results = [evaluate("self")]
results.append(evaluate("bf16 rounding (Linear x (1 +- 2^-8))", perturb_linears(wl.roots())))
for label, fn in [
    ("fp8 e4m3 weight-only, per channel", lambda: fake_fp8(True)),
    ("fp8 e4m3 weight-only, per tensor", lambda: fake_fp8(False)),
    ("int4 weight-only, group 128", lambda: fake_int4(128)),
    ("int4 weight-only, per channel", lambda: fake_int4(None)),
    ("fp8 per channel, scales x1.05", lambda: fake_fp8(True, 1.05)),
    ("fp8 per channel, scales x1.2", lambda: fake_fp8(True, 1.2)),
    ("RMSNorm eps 1e-2", lambda: rmsnorm_eps(1e-2)),
    ("one KV head dropped (k_proj rows 0)", drop_kv_head),
]:
    fn()
    results.append(evaluate(label))
    restore()
    rmsnorm_eps(wl.model.config.rms_norm_eps)
results.append(broken_loop())
json.dump(results, open(sys.argv[1], "w"), indent=1)
