"""The LLM gate (#170) at near-lossless and relaxed thresholds (#175) on Qwen3-0.6B.

Each variant is scored teacher forced on the 14 natural prompts of
``LLMWorkload.perceptual_samples()`` (64 tokens each) like ``e2e`` does, and judged with
the near-lossless thresholds (``perceptual.LLM_*``) and the relaxed ones
(``perceptual.RELAXED_GATE``); the first-step logits cosine is the sanity floor
(``min_cosine``: near-lossless 0.98, relaxed 0.96). Honest numerics changes, more
aggressive ones (NVFP4 / MXFP4 weights, W8A8 on every decoder Linear) and broken ones.

    python calibrate_llm_relaxed.py out.json
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable

import torch
from torch import nn

from kernel_agent import toolchain
from kernel_agent.kernels import quant
from kernel_agent.workloads import create_workload, perceptual
from kernel_agent.workloads.base import WorkloadSpec, cosine

toolchain.setup()
wl = create_workload(WorkloadSpec(repo_id="Qwen/Qwen3-0.6B", modality="llm"))
wl.load()
samples = wl.perceptual_samples()
ref_gen, _ = perceptual.generate(wl, samples)
ref_scores, _ = perceptual.score(wl, ref_gen)
saved = {k: v.detach().clone() for k, v in wl.model.state_dict().items()}
RELAXED = {
    "max_kl": perceptual.RELAXED_GATE["max_kl"],
    "max_kl_worst": perceptual.RELAXED_GATE["max_kl_worst"],
    "min_top1": perceptual.RELAXED_GATE["min_top1"],
    "max_nll_increase": perceptual.RELAXED_GATE["max_nll_increase"],
}
LLM_FLOOR = 0.96  # LLMWorkload.relaxed_options["min_cosine"] (near-lossless: 0.98)


def linears():
    for name, mod in wl.model.named_modules():
        if isinstance(mod, nn.Linear) and ".layers." in name:
            yield name, mod


def restore() -> None:
    with torch.no_grad():
        for k, v in wl.model.state_dict().items():
            v.copy_(saved[k])
    for _, mod in linears():
        mod.__dict__.pop("forward", None)  # w8a8's patched forward
    for mod in wl.model.modules():
        if "RMSNorm" in type(mod).__name__:
            mod.variance_epsilon = wl.model.config.rms_norm_eps


def weights(fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
    with torch.no_grad():
        for _, mod in linears():
            mod.weight.copy_(fn(mod.weight).to(mod.weight.dtype))


def fp8(scale: float = 1.0) -> None:
    def q(w):
        codes, s = quant.quantize_fp8(w)
        return quant.dequantize_fp8(codes, s * scale, w.dtype)

    weights(q)


def fp4(fmt: str) -> None:
    weights(lambda w: quant.dequantize_fp4(*quant.quantize_fp4(w, fmt), w.dtype))


def int4(group: int | None) -> None:
    def q(w):
        shape, x = w.shape, w.float()
        x = x.reshape(shape[0], -1, group) if group else x
        s = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 7
        return ((x / s).round().clamp(-8, 7) * s).reshape(shape)

    weights(q)


def w8a8() -> None:
    for _, mod in linears():
        codes, s = quant.quantize_fp8(mod.weight)
        mod.forward = lambda x, c=codes, s=s, b=mod.bias: quant.fp8_w8a8_linear(x, c, s, b)


def eps() -> None:
    for mod in wl.model.modules():
        if "RMSNorm" in type(mod).__name__:
            mod.variance_epsilon = 1e-2


def kv_head() -> None:
    with torch.no_grad():
        for name, mod in linears():
            if name.endswith("k_proj"):
                mod.weight[: mod.weight.shape[0] // wl.model.config.num_key_value_heads] = 0


def judge(label: str, scores: list[dict], first_cos: float | None, t: float) -> dict:
    near = perceptual.compare_llm(ref_scores, scores)
    relaxed = perceptual.compare_llm(ref_scores, scores, **RELAXED)
    out = {
        "variant": label,
        **{k: near.metrics[k] for k in ("kl", "kl_worst", "top1", "nll_increase")},
        "first_logits_cosine_min": first_cos,
        "near_lossless": near.passed and (first_cos is None or first_cos >= 0.98),
        "relaxed": relaxed.passed and (first_cos is None or first_cos >= LLM_FLOOR),
        "near_lossless_reason": near.reason,
        "relaxed_reason": relaxed.reason,
        "s": round(time.time() - t, 1),
    }
    print(json.dumps(out), flush=True)
    return out


def evaluate(label: str, setup: Callable[[], None]) -> dict:
    t = time.time()
    setup()
    gen, _ = perceptual.generate(wl, samples)
    for g, s in zip(gen, ref_scores, strict=True):
        g["reference"] = s
    scores, _ = perceptual.score(wl, gen)
    first = min(
        cosine(r["output"]["first_logits"], g["output"]["first_logits"])
        for r, g in zip(ref_gen, gen, strict=True)
    )
    restore()
    return judge(label, scores, round(first, 4), t)


def shifted_loop() -> dict:
    """A decode loop that writes the right tokens one place late (#170)."""
    t = time.time()
    gen = []
    for rg, s in zip(ref_gen, ref_scores, strict=True):
        tokens = rg["output"]["tokens"].clone()
        tokens[0] = torch.roll(tokens[0], 1)
        gen.append({"options": rg["options"], "output": {"tokens": tokens}, "reference": s})
    scores, _ = perceptual.score(wl, gen)
    return judge("decode loop one token late", scores, None, t)


results = [
    evaluate(label, fn)
    for label, fn in [
        ("FP8 weights, per channel", fp8),
        ("FP8 W8A8 (every decoder Linear)", w8a8),
        ("FP8 weights, scales x1.05", lambda: fp8(1.05)),
        ("FP8 weights, scales x1.1", lambda: fp8(1.1)),
        ("NVFP4 weights", lambda: fp4("nvfp4")),
        ("MXFP4 weights", lambda: fp4("mxfp4")),
        ("int4 weights, group 128", lambda: int4(128)),
        ("int4 weights, per channel", lambda: int4(None)),
        ("FP8 weights, scales x1.2", lambda: fp8(1.2)),
        ("RMSNorm eps 1e-2", eps),
        ("one KV head dropped", kv_head),
    ]
]
results.append(shifted_loop())
json.dump(results, open(sys.argv[1], "w"), indent=1)
