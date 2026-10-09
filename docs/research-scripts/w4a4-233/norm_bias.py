"""W4A4's norm bias and what removes it (#233 follow-up 8), on Qwen3-0.6B's real activations
and weights: every q / k / v / o / gate / up / down projection of the 28 decoder layers,
captured on a fixed text (bf16 model, eager), each GEMM emulated as dequantise -> float64
matmul (no FP4 hardware needed), against the exact GEMM of the captured bf16 activations.

Variants (activations per token; weights per tensor, as ``quant.quantize_fp4``):

* ``nvfp4``: the current rule, every block maximum to 6 (``quantize_fp4_activations``);
* ``4/6 act``: adaptive block scales on the activations: per block of 16, the block maximum
  to 6 or to 4, whichever gives the lower squared error of the block (the outer scale
  ``amax / (4 x 448)`` so that a block mapped to 4 never clamps at 448); weights as nvfp4;
* ``4/6 both``: the same on the weights (tensor scale ``amax / (4 x 448)``);
* ``token unbiased``: nvfp4, then each token's dequantised activations times ``c = sum x^2 /
  sum x x^`` (the in-phase shrink of the row: ``x^ ~ a x + e``, ``c = 1 / a``), a factor of the
  per-token outer scale (free in the epilogue);
* ``token LS``: ``c = sum x x^ / sum x^^2`` (least squares; for comparison);
* ``token norm``: ``c = |x| / |x^|`` (for comparison);
* ``unbiased + channel``: ``token unbiased`` and the weight's per-output-channel ``c_n = sum
  w^2 / sum w w^`` (a per-channel epilogue factor);
* ``4/6 act + token``, ``4/6 both + token`` (``+ channel``): the combinations.

Per GEMM: the output's norm change, its in-phase gain ``<y^, y> / |y|^2`` (the bias; the noise
lifts the norm above it), cosine and relative L2 against the exact GEMM. The MLP rows run
gate / up / silu * up / down with every GEMM quantised (the bias compounds; one layer's
massive activations dominate the mean, hence the median). The library's opt-in
(``quant.quantize_fp4(..., unbiased=True)``, ``fp4_w4a4_error(..., unbiased=True)``) is
checked against the ``unbiased + channel`` prototype.

    PYTHONPATH=src python norm_bias.py [--device cuda] [--variants ...]
    (results/norm_bias.out: CPU, torch 2.10)
"""

import argparse
import math
import os
import time
from collections import defaultdict

import torch
from torch import nn

from kernel_agent.kernels import quant

STEP6 = quant.NVFP4_OUTER_STEP
STEP4 = float(torch.tensor(1.0 / (4 * 448.0), dtype=torch.float32))
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

TEXT = [
    "The committee met on Tuesday to review the quarterly budget. After a long discussion "
    "about rising energy costs, the members agreed to postpone the purchase of new servers "
    "until the second half of the year, and to ask the vendors for revised quotes.",
    "def merge_sorted(a, b):\n    out, i, j = [], 0, 0\n    while i < len(a) and j < len(b):\n"
    "        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n"
    "            out.append(b[j]); j += 1\n    return out + a[i:] + b[j:]\n",
    "Photosynthesis converts light energy into chemical energy stored in glucose. In the "
    "light-dependent reactions, water is split and oxygen is released; the Calvin cycle then "
    "fixes carbon dioxide using ATP and NADPH produced earlier.",
    "Q: A train leaves at 14:35 and arrives at 18:10. How long is the journey? A: From 14:35 "
    "to 18:35 is four hours; subtract 25 minutes, so the journey takes 3 hours 35 minutes.",
    "Der Zug nach München hatte zwanzig Minuten Verspätung, weil ein Signal ausgefallen war. "
    "Les passagers attendaient sur le quai en regardant le tableau d'affichage. 列车终于到站了。",
    "In 1969, Apollo 11 landed on the Moon. Neil Armstrong and Buzz Aldrin spent about 21 "
    "hours on the lunar surface, while Michael Collins orbited above in the command module "
    "Columbia, waiting to bring them home.",
]


def capture(device: str) -> tuple[nn.Module, dict[str, torch.Tensor], int]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "Qwen/Qwen3-0.6B"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(device).eval()
    seen: dict[str, list[torch.Tensor]] = defaultdict(list)

    def keep(n):
        def hook(_, args):
            seen[n].append(args[0].detach().reshape(-1, args[0].shape[-1]))

        return hook

    hooks = [
        m.register_forward_pre_hook(keep(n))
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and n.split(".")[-1] in PROJ
    ]
    tokens = 0
    with torch.inference_mode():
        for text in TEXT:
            ids = tok(text, return_tensors="pt").input_ids.to(device)
            tokens += ids.shape[1]
            model(ids)
    for h in hooks:
        h.remove()
    return model, {n: torch.cat(v) for n, v in seen.items()}, tokens


# ------------------------------------------------------------------ quantisers (prototypes)


def nvfp4(a: torch.Tensor, *, adaptive: bool = False, per_row: bool = True) -> torch.Tensor:
    """Dequantised NVFP4 of fp32 rows ``a [R, K]`` (blocks of 16 along K): the outer scale per
    row (activations) or per tensor (weights); ``adaptive``: each block's maximum to 6 or to
    4, whichever has the lower squared error (outer scale amax / (4 x 448))."""
    r, k = a.shape
    blocks = a.reshape(r, k // 16, 16)
    bmax = blocks.abs().amax(-1)
    amax = a.abs().amax(1) if per_row else a.abs().amax().expand(r)
    outer = torch.where(amax > 0, amax * (STEP4 if adaptive else STEP6), torch.ones_like(amax))

    def to(top: float) -> torch.Tensor:
        s = (bmax / (outer * top)[:, None]).clamp(max=448.0).to(torch.float8_e4m3fn)
        codes = quant._fp4_codes(blocks, s.float() * outer[:, None])
        return quant.fp4_values(codes, s).reshape(r, k // 16, 16) * outer[:, None, None]

    deq = to(6.0)
    if adaptive:
        four = to(4.0)
        pick = ((blocks - four) ** 2).sum(-1) < ((blocks - deq) ** 2).sum(-1)
        deq = torch.where(pick[..., None], four, deq)
    return deq.reshape(r, k)


def unbiased(a: torch.Tensor, q: torch.Tensor, kind: str = "unbiased") -> torch.Tensor:
    """Per-row factor of ``q`` (the quantised ``a``)."""
    aa, aq, qq = (a * a).sum(1), (a * q).sum(1), (q * q).sum(1)
    if kind == "unbiased":
        c = aa / aq
    elif kind == "ls":
        c = aq / qq
    else:
        c = (aa / qq).sqrt()
    return torch.where(torch.isfinite(c) & (c > 0), c, torch.ones_like(c))


VARIANTS = {
    # name: (activation quantiser, weight quantiser, token correction, channel correction)
    "nvfp4": ({}, {}, None, False),
    "4/6 act": ({"adaptive": True}, {}, None, False),
    "4/6 both": ({"adaptive": True}, {"adaptive": True}, None, False),
    "token unbiased": ({}, {}, "unbiased", False),
    "token LS": ({}, {}, "ls", False),
    "token norm": ({}, {}, "norm", False),
    "unbiased + channel": ({}, {}, "unbiased", True),
    "4/6 act + token": ({"adaptive": True}, {}, "unbiased", False),
    "4/6 both + token": ({"adaptive": True}, {"adaptive": True}, "unbiased", False),
    "4/6 both + token + channel": ({"adaptive": True}, {"adaptive": True}, "unbiased", True),
}


def quantised_linear(x: torch.Tensor, w: torch.Tensor, variant: str) -> torch.Tensor:
    act, wq, token, channel = VARIANTS[variant]
    xq = nvfp4(x, **act)
    if token:
        xq = xq * unbiased(x, xq, token)[:, None]
    if wq:
        wd = nvfp4(w, per_row=False, **wq)
    else:  # the library's weight quantiser (its tensor scale amax / 2688 rounded once)
        wd = quant.dequantize_fp4(*quant.quantize_fp4(w), torch.float32)
    if channel:
        wd = wd * unbiased(w, wd)[:, None]
    return xq.double() @ wd.double().T


def metrics(ref: torch.Tensor, new: torch.Tensor) -> tuple[float, float, float, float]:
    """Norm ratio, cosine, relative L2 and the in-phase gain <new, ref> / |ref|^2 (the bias:
    the norm ratio is about sqrt(gain^2 + rel L2^2 - ...), noise adds to it)."""
    ref, new = ref.double().flatten(), new.double().flatten()
    rn, nn_ = ref.norm(), new.norm()
    dot = ref @ new
    return float(nn_ / rn), float(dot / (rn * nn_)), float((new - ref).norm() / rn), float(dot / rn**2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--variants", nargs="*", default=list(VARIANTS))
    p.add_argument("--layers", type=int, default=28)
    args = p.parse_args()
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    t0 = time.time()
    model, seen, tokens = capture(args.device)
    print(f"Qwen3-0.6B, {len(TEXT)} texts, {tokens} tokens, captured in {time.time() - t0:.1f} s")
    layers = model.model.layers[: args.layers]
    table: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    library = []  # |gain of quant's unbiased=True - gain of the "unbiased + channel" prototype|
    for i, layer in enumerate(layers):
        mods = {**dict(layer.self_attn.named_children()), **dict(layer.mlp.named_children())}
        for proj in PROJ:
            group = "self_attn" if proj in PROJ[:4] else "mlp"
            x = seen[f"model.layers.{i}.{group}.{proj}"].float()
            w = mods[proj].weight.detach().float()
            ref = x.double() @ w.double().T
            for v in args.variants:
                table[(proj, v)].append(metrics(ref, quantised_linear(x, w, v)))
            if "unbiased + channel" in args.variants:
                lib = quant.fp4_w4a4_error(
                    w, *quant.quantize_fp4(w, unbiased=True), x, unbiased=True
                )
                library.append(abs(lib["output_gain"] - table[(proj, "unbiased + channel")][-1][3]))
        # the whole MLP: gate / up -> silu(gate) * up -> down, every GEMM quantised
        x = seen[f"model.layers.{i}.mlp.gate_proj"].float()
        wg, wu, wd = (layer.mlp.get_submodule(n).weight.detach().float() for n in PROJ[4:])
        g, u = x.double() @ wg.double().T, x.double() @ wu.double().T
        ref = (nn.functional.silu(g) * u) @ wd.double().T
        for v in args.variants:
            g, u = quantised_linear(x, wg, v), quantised_linear(x, wu, v)
            h = (nn.functional.silu(g) * u).float()
            table[("MLP", v)].append(metrics(ref, quantised_linear(h, wd, v)))
    print(f"{len(layers)} layers in {time.time() - t0:.1f} s; per projection over the layers:")
    print("norm change and in-phase gain - 1 mean [worst], cosine mean [min], rel L2 mean [max]\n")
    head = f"| {'projection':10s} | {'variant':18s} | norm change | gain - 1 | cosine | rel L2 |"
    print(head)
    print("|" + "---|" * 6)
    for proj in (*PROJ, "MLP"):
        for v in args.variants:
            rows = table[(proj, v)]
            norm = [r[0] - 1 for r in rows]
            worst = max(norm, key=abs)
            gain = [r[3] - 1 for r in rows]
            cos = [r[1] for r in rows]
            rel = [r[2] for r in rows]
            if proj == "MLP":  # one layer's massive activations dominate the mean: the median
                gain_median = sorted(gain)[len(gain) // 2]
                norm_median = sorted(norm)[len(norm) // 2]
                v = f"{v} (median {100 * norm_median:+.2f} / {100 * gain_median:+.2f} %)"
            print(
                f"| {proj:10s} | {v:18s} | {100 * sum(norm) / len(norm):+.2f} % "
                f"[{100 * worst:+.2f} %] | {100 * sum(gain) / len(gain):+.2f} % "
                f"[{100 * max(gain, key=abs):+.2f} %] | {sum(cos) / len(cos):.5f} [{min(cos):.5f}] | "
                f"{sum(rel) / len(rel):.4f} [{max(rel):.4f}] |"
            )
    worst = min(range(len(layers)), key=lambda i: table[("MLP", "nvfp4")][i][3])
    print(f"\nthe MLP of layer {worst} (the worst): " + "; ".join(
        f"{v} gain {100 * (table[('MLP', v)][worst][3] - 1):+.1f} %" for v in args.variants))
    print("\nall projections (7 x layers):")
    for v in args.variants:
        rows = [r for proj in PROJ for r in table[(proj, v)]]
        norm = [r[0] - 1 for r in rows]
        mean_abs = sum(abs(n) for n in norm) / len(norm)
        gain = [r[3] - 1 for r in rows]
        print(
            f"  {v:18s} norm change mean {100 * sum(norm) / len(norm):+.3f} %, mean |.| "
            f"{100 * mean_abs:.3f} %, worst {100 * max(norm, key=abs):+.2f} %; gain - 1 mean "
            f"{100 * sum(gain) / len(gain):+.3f} % (worst {100 * max(gain, key=abs):+.2f} %); cosine mean "
            f"{sum(r[1] for r in rows) / len(rows):.5f}; rel L2 mean "
            f"{sum(r[2] for r in rows) / len(rows):.4f} (max {max(r[2] for r in rows):.4f})"
        )
    if library:
        print(
            "\nquant.quantize_fp4(unbiased=True) + fp4_w4a4_error(unbiased=True) against the "
            f"'unbiased + channel' prototype: in-phase gain within {max(library):.1e}"
        )
    assert all(math.isfinite(r[2]) for rows in table.values() for r in rows)


if __name__ == "__main__":
    main()
