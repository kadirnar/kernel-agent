"""Calibrate the relaxed tolerance tiers (#175) against the near-lossless ones on captures.

For each capture (the VoxCPM2 LocDiT layer at M = 352 / 176, a Qwen3-0.6B decoder layer at
decode with its KV cache, ...) and each variant (the reference math of every reduced
precision; broken kernels), the candidate (the capture's module with every ``nn.Linear``
replaced) is compared with the reference on

* the captured inputs, against the captured outputs and side effects (``captured``);
* ``--seeds`` redrawn copies of every case's inputs per redraw kind (``verify.perturb_``:
  normal, then the uniform / Laplace / log-normal mix), against the reference called live,
  with the bounds of redrawn inputs (``redrawn``);
* the captured inputs x 3, x 0.01 and x -1 (``verify.SCALED``, ``input_scale``; ``scaled``),

each judged in the near-lossless tier and in the relaxed tier of the variant's precision
(one candidate and one reference call per input: both tiers judge the same outputs). Per
(capture, variant, input class): draws, failed draws per tier, worst cosine, relative L2
error, |norm - 1| and element ratio (of each tier's own element bound). ``--kv`` adds the
``fp8_kv`` calibration (a decode-attention step, synthetic GQA shapes as in #145).

    python calibrate_tiers.py --device cuda --seeds 10 --out results.json CAPTURE.pt ...
"""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from kernel_agent.kernels import compare, quant, verify
from kernel_agent.profiling.capture import capture_calls, load_capture
from kernel_agent.profiling.state import Replay

QUALITIES = ("near-lossless", "relaxed")


# ------------------------------------------------------------------ variants


def _int4_per_tensor(w: torch.Tensor) -> torch.Tensor:
    scale = w.float().abs().amax() / 7
    return (torch.round(w.float() / scale).clamp(-8, 7) * scale).to(w.dtype)


class QLinear(nn.Module):
    """An ``nn.Linear`` with the numerics of ``kind``; ``bug``: a broken kernel."""

    def __init__(self, linear: nn.Linear, kind: str, bug: str | None) -> None:
        super().__init__()
        w = linear.weight.detach()
        self.bias = linear.bias
        self.kind, self.bug = kind, bug
        if bug == "int4 per tensor":
            self.w = _int4_per_tensor(w)
            return
        if kind == "fp8_mx":
            self.q, self.s = quant.quantize_mxfp8(w)
            return
        if kind in ("nvfp4", "mxfp4"):
            codes, scales, ts = quant.quantize_fp4(w, kind)
            if bug == "nibbles swapped":
                codes = (codes >> 4) | ((codes & 0xF) << 4)
            if bug == "scales x1.2":
                ts = ts * 1.2
            self.w = quant.dequantize_fp4(codes, scales, ts, w.dtype)
            return
        self.q, self.s = quant.quantize_fp8(w)
        if bug and bug.startswith("scales x"):
            self.s = self.s * float(bug.removeprefix("scales x"))
        self.w = quant.dequantize_fp8(self.q, self.s, w.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "fp8_mx" and self.bug != "int4 per tensor":
            return quant.mxfp8_linear(x, self.q, self.s, self.bias)
        if self.kind != "fp8_w8a8" or self.bug == "int4 per tensor":
            return nn.functional.linear(x, self.w.to(x.device), self.bias)
        if self.bug != "first token's activation scale":
            return quant.fp8_w8a8_linear(x, self.q, self.s, self.bias)
        xq, xs = quant.quantize_fp8_activations(x)
        xs = xs[:1].expand_as(xs)
        a = x.detach().reshape(-1, x.shape[-1]).float()
        xq = (a / xs[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        y = (xq.float() * xs[:, None]) @ (self.q.float() * self.s[:, None]).T
        return y.to(x.dtype).reshape(*x.shape[:-1], -1)


def _attention(module: nn.Module) -> tuple[nn.Module, int, int] | None:
    """(attention module, head dim, KV heads) of the first submodule with q/k/v projections."""
    for mod in module.modules():
        if all(isinstance(getattr(mod, n, None), nn.Linear) for n in ("q_proj", "v_proj")):
            hd = int(getattr(mod, "head_dim", 0) or 0)
            if hd:
                return mod, hd, mod.v_proj.out_features // hd
    return None


def _edit_weights(module: nn.Module, edit: str | None) -> None:
    """Weight edits of the broken variants, before quantisation."""
    if edit is None:
        return
    with torch.no_grad():
        if edit == "row 0 skipped":  # an off-by-one row loop: output channel 0 never written
            for mod in module.modules():
                if isinstance(mod, nn.Linear):
                    mod.weight[0] = 0
            return
        found = _attention(module)
        if found is None:
            raise ValueError(f"{edit}: no attention module with q_proj / v_proj / head_dim")
        attn, hd, _ = found
        if edit == "KV head 0 dropped":  # its value rows zeroed: its query heads read zeros
            attn.v_proj.weight[:hd] = 0
        elif edit == "q heads 0 and last swapped":  # a head-layout bug (across KV groups)
            w = attn.q_proj.weight
            first = w[:hd].clone()
            w[:hd] = w[-hd:]
            w[-hd:] = first
        else:
            raise ValueError(edit)


#: (label, precision of the target, numerics kind, quantisation bug, weight edit)
VARIANTS: list[tuple[str, str, str, str | None, str | None]] = [
    ("FP8 weights", "fp8_weights", "fp8_weights", None, None),
    ("FP8 W8A8", "fp8_w8a8", "fp8_w8a8", None, None),
    ("MXFP8 W8A8", "fp8_mx", "fp8_mx", None, None),
    ("NVFP4 weights", "fp4_weights", "nvfp4", None, None),
    ("MXFP4 weights", "fp4_weights", "mxfp4", None, None),
    ("BUG fp8: scales x1.05", "fp8_weights", "fp8_weights", "scales x1.05", None),
    ("BUG fp8: scales x1.2", "fp8_weights", "fp8_weights", "scales x1.2", None),
    ("BUG fp8: int4 per tensor", "fp8_weights", "fp8_weights", "int4 per tensor", None),
    ("BUG fp8: row 0 of every GEMM skipped", "fp8_weights", "fp8_weights", None, "row 0 skipped"),
    ("BUG fp8: KV head 0 dropped", "fp8_weights", "fp8_weights", None, "KV head 0 dropped"),
    (
        "BUG fp8: q heads 0/last swapped (layout)",
        "fp8_weights",
        "fp8_weights",
        None,
        "q heads 0 and last swapped",
    ),
    (
        "BUG w8a8: first token's activation scale",
        "fp8_w8a8",
        "fp8_w8a8",
        "first token's activation scale",
        None,
    ),
    ("BUG fp4: nibbles swapped", "fp4_weights", "nvfp4", "nibbles swapped", None),
    ("BUG fp4: scales x1.2", "fp4_weights", "nvfp4", "scales x1.2", None),
    ("BUG fp4: int4 per tensor", "fp4_weights", "nvfp4", "int4 per tensor", None),
    ("BUG fp4: row 0 of every GEMM skipped", "fp4_weights", "nvfp4", None, "row 0 skipped"),
    ("BUG fp4: KV head 0 dropped", "fp4_weights", "nvfp4", None, "KV head 0 dropped"),
    (
        "BUG fp4: q heads 0/last swapped (layout)",
        "fp4_weights",
        "nvfp4",
        None,
        "q heads 0 and last swapped",
    ),
]


def linear_candidate(reference: nn.Module, kind: str, bug: str | None, edit: str | None) -> Any:
    candidate = copy.deepcopy(reference)
    _edit_weights(candidate, edit)
    for name, mod in list(candidate.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                setattr(mod, child_name, QLinear(child, kind, bug))
    return candidate.eval()


# ------------------------------------------------------------------ judging


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _call(fn: Callable[..., Any], args: Any, kwargs: Any) -> Any:
    with torch.inference_mode():
        out = fn(*args, **kwargs)
    _sync()
    return copy.deepcopy(out)


def _judge(
    tiers: dict[str, str],
    expected: Any,
    out: Any,
    pre: tuple[Any, Any],
    ref_post: tuple[Any, Any],
    new_post: tuple[Any, Any],
    *,
    perturbed: bool,
    input_scale: float = 1.0,
) -> dict[str, list[dict[str, Any]]]:
    found = {}
    for quality, tier in tiers.items():
        kw: dict[str, Any] = {"tier": tier, "perturbed": perturbed, "input_scale": input_scale}
        checks = compare.compare_structures(expected, out, "output", **kw)
        checks += compare.compare_side_effects(pre[0], ref_post[0], new_post[0], "args", **kw)
        checks += compare.compare_side_effects(pre[1], ref_post[1], new_post[1], "kwargs", **kw)
        found[quality] = checks
    return found


def _live(
    ref_fn: Callable[..., Any],
    new_fn: Callable[..., Any],
    args: Any,
    kwargs: Any,
    tiers: dict[str, str],
    *,
    input_scale: float = 1.0,
    finite: bool = False,
) -> dict[str, list[dict[str, Any]]] | None:
    pre = copy.deepcopy((args, kwargs))
    out = _call(new_fn, args, kwargs)
    ref_args, ref_kwargs = copy.deepcopy(pre)
    expected = _call(ref_fn, ref_args, ref_kwargs)
    if finite and not verify._finite((expected, ref_args, ref_kwargs)):
        return None
    return _judge(
        tiers,
        expected,
        out,
        pre,
        (ref_args, ref_kwargs),
        (args, kwargs),
        perturbed=True,
        input_scale=input_scale,
    )


class Stats:
    """Failed draws per quality mode and the worst metrics of the relaxed tier's checks."""

    def __init__(self) -> None:
        self.draws = 0
        self.fails = dict.fromkeys(QUALITIES, 0)
        self.min_cos, self.max_rel, self.max_norm = 1.0, 0.0, 0.0
        self.element = dict.fromkeys(QUALITIES, 0.0)
        self.example: dict[str, str] = {}

    def add(self, found: dict[str, list[dict[str, Any]]]) -> None:
        self.draws += 1
        for quality, checks in found.items():
            bad = [c for c in checks if not c.get("ok")]
            if bad:
                self.fails[quality] += 1
                self.example.setdefault(quality, f"{bad[0]['name']}: {bad[0].get('error')}")
            for c in checks:
                if "element_ratio" not in c:  # not judged by the tier (no signal, integer)
                    continue
                self.element[quality] = max(self.element[quality], c["element_ratio"])
                if quality != "relaxed":
                    continue
                self.min_cos = min(self.min_cos, c["cosine"])
                self.max_rel = max(self.max_rel, c["rel_l2"])
                self.max_norm = max(self.max_norm, abs(c["norm_ratio"] - 1.0))

    def row(self) -> dict[str, Any]:
        return {
            "draws": self.draws,
            "fails": dict(self.fails),
            "min_cosine": round(self.min_cos, 5),
            "max_rel_l2": round(self.max_rel, 4),
            "max_norm_change": round(self.max_norm, 4),
            "max_element_ratio": {k: round(v, 3) for k, v in self.element.items()},
            "example": self.example,
        }


def calibrate(
    capture: dict[str, Any],
    build: Callable[[nn.Module], Any],
    precision: str,
    *,
    seeds: int,
    max_cases: int,
    device: str,
) -> dict[str, dict[str, Any]]:
    tiers = {q: compare.tier_for(q, precision) for q in QUALITIES}
    reference = capture["module"].eval()
    replay = Replay(capture, reference)
    candidate = build(reference)
    cases = capture["cases"][:max_cases]
    stats = {"captured": Stats(), "redrawn": Stats(), "scaled": Stats()}
    gen = torch.Generator(device=device).manual_seed(1)
    for case in cases:
        ref_fn = replay.call(case, reference)
        new_fn = replay.call(case, candidate)
        args, kwargs = copy.deepcopy((case["args"], case["kwargs"]))
        out = _call(new_fn, args, kwargs)
        post = (case["post_args"], case["post_kwargs"])
        stats["captured"].add(
            _judge(
                tiers,
                case["output"],
                out,
                (case["args"], case["kwargs"]),
                post,
                (args, kwargs),
                perturbed=False,
            )
        )
        for seed in range(seeds):
            for kind in ("normal", "mix"):
                args, kwargs = copy.deepcopy((case["args"], case["kwargs"]))
                verify.perturb_((args, kwargs), gen, kind)
                found = _live(ref_fn, new_fn, args, kwargs, tiers)
                assert found is not None
                stats["redrawn"].add(found)
        finite = verify._finite(case["output"])
        for _, factor in verify.SCALED:
            args, kwargs = copy.deepcopy((case["args"], case["kwargs"]))
            if not verify.scale_((args, kwargs), factor):
                continue
            found = _live(
                ref_fn, new_fn, args, kwargs, tiers, input_scale=factor, finite=finite
            )
            if found is not None:
                stats["scaled"].add(found)
    return {k: s.row() for k, s in stats.items()}


# ------------------------------------------------------------------ fp8_kv


class DecodeAttention(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)


class Fp8KV(nn.Module):
    def __init__(self, bug: str | None) -> None:
        super().__init__()
        self.bug = bug

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        from kernel_agent.kernels.kv_quant import fp8_kv_attention, quantize_fp8_kv

        kc, ks = quantize_fp8_kv(k)
        vc, vs = quantize_fp8_kv(v)
        if self.bug and self.bug.startswith("v scales x"):
            vs = vs * float(self.bug.removeprefix("v scales x"))
        elif self.bug == "first token's scale":
            ks, vs = ks[..., :1].expand_as(ks), vs[..., :1].expand_as(vs)
        return fp8_kv_attention(q, kc, ks, vc, vs)


KV_VARIANTS = [
    ("FP8 KV cache", None),
    ("BUG kv: v scales x1.05", "v scales x1.05"),
    ("BUG kv: v scales x1.2", "v scales x1.2"),
    ("BUG kv: first token's scale", "first token's scale"),
]
#: (query heads, KV heads, head dim, cached tokens) of the decode steps (#145's shapes).
KV_SHAPES = [(16, 2, 128, 4096), (16, 8, 128, 544), (32, 8, 64, 77), (14, 2, 64, 1024)]


def kv_capture(path: Path, seed: int, device: str) -> dict[str, Any]:
    gen = torch.Generator().manual_seed(seed)
    cases = []
    for heads, kv_heads, dim, tokens in KV_SHAPES:
        q = torch.randn(1, heads, 1, dim, generator=gen)
        k = torch.randn(1, kv_heads, tokens, dim, generator=gen)
        v = torch.randn(1, kv_heads, tokens, dim, generator=gen)
        q = q + 2.0 * k[:, :, -1:].repeat_interleave(heads // kv_heads, 1)  # a sharper softmax
        args = tuple(t.to(device, torch.bfloat16) for t in (q, k, v))
        cases.append((args, {}, 1))
    capture_calls(DecodeAttention(), cases, path, tier="near-lossless-kv", precision="fp8_kv")
    return load_capture(path, device=device)


# ------------------------------------------------------------------ main


def table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| capture | variant | inputs | draws | near-lossless fails | relaxed fails | "
        "min cosine | max rel L2 | max norm change | max element ratio (NL / relaxed) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        for cls, s in r["classes"].items():
            e = s["max_element_ratio"]
            lines.append(
                f"| {r['capture']} | {r['variant']} | {cls} | {s['draws']} | "
                f"{s['fails']['near-lossless']} | {s['fails']['relaxed']} | "
                f"{s['min_cosine']:.4f} | {s['max_rel_l2']:.3f} | "
                f"{s['max_norm_change'] * 100:.1f} % | "
                f"{e['near-lossless']:.2f} / {e['relaxed']:.2f} |"
            )
    return "\n".join(lines)


def save(out: Path | None, ns: argparse.Namespace, results: list[dict[str, Any]]) -> None:
    """The results so far as ``out`` (JSON, with the bounds) and ``out``.md (the table)."""
    if out is None:
        return
    bounds = {
        t: {"captured": compare.NEAR_LOSSLESS_BOUNDS[t], "redrawn": compare.PERTURBED_BOUNDS[t]}
        for t in compare.NEAR_LOSSLESS_BOUNDS
    }
    data = {"device": ns.device, "seeds": ns.seeds, "bounds": bounds, "results": results}
    out.write_text(json.dumps(data, indent=1, default=str))
    out.with_suffix(".md").write_text(table(results) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("captures", nargs="*", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--variants", default="", help="comma list of label prefixes")
    parser.add_argument("--kv", action="store_true", help="also calibrate fp8_kv")
    parser.add_argument("--kv-seeds", type=int, default=12)
    parser.add_argument("--out", type=Path)
    ns = parser.parse_args()
    wanted = [w.strip() for w in ns.variants.split(",") if w.strip()]
    results: list[dict[str, Any]] = []
    for path in ns.captures:
        capture = load_capture(path, device=ns.device)
        name = f"{path.parent.parent.parent.name}/{path.stem}"
        for label, precision, kind, bug, edit in VARIANTS:
            if wanted and not any(label.startswith(w) for w in wanted):
                continue
            t0 = time.time()
            try:
                classes = calibrate(
                    capture,
                    lambda ref, k=kind, b=bug, e=edit: linear_candidate(ref, k, b, e),
                    precision,
                    seeds=ns.seeds,
                    max_cases=ns.max_cases,
                    device=ns.device,
                )
            except ValueError as exc:
                print(f"{name} {label}: skipped ({exc})", flush=True)
                continue
            results.append({"capture": name, "variant": label, "classes": classes})
            save(ns.out, ns, results)
            print(
                f"{name} {label} ({time.time() - t0:.0f} s): "
                + "; ".join(
                    f"{c} NL {s['fails']['near-lossless']}/{s['draws']} relaxed "
                    f"{s['fails']['relaxed']}/{s['draws']} cos {s['min_cosine']} rel "
                    f"{s['max_rel_l2']} norm {s['max_norm_change']} el "
                    f"{s['max_element_ratio']}"
                    for c, s in classes.items()
                ),
                flush=True,
            )
        del capture
    if ns.kv:
        with tempfile.TemporaryDirectory() as tmp:
            for label, bug in KV_VARIANTS:
                rows = []
                for seed in range(ns.kv_seeds):
                    cap = kv_capture(Path(tmp) / f"kv{seed}.pt", seed, ns.device)
                    rows.append(
                        calibrate(
                            cap,
                            lambda ref, b=bug: Fp8KV(b),
                            "fp8_kv",
                            seeds=2,
                            max_cases=len(KV_SHAPES),
                            device=ns.device,
                        )
                    )
                classes = {}
                for cls in rows[0]:
                    group = [r[cls] for r in rows]
                    classes[cls] = {
                        "draws": sum(g["draws"] for g in group),
                        "fails": {q: sum(g["fails"][q] for g in group) for q in QUALITIES},
                        "min_cosine": min(g["min_cosine"] for g in group),
                        "max_rel_l2": max(g["max_rel_l2"] for g in group),
                        "max_norm_change": max(g["max_norm_change"] for g in group),
                        "max_element_ratio": {
                            q: max(g["max_element_ratio"][q] for g in group) for q in QUALITIES
                        },
                        "example": next((g["example"] for g in group if g["example"]), {}),
                    }
                results.append(
                    {"capture": "decode attention (synthetic)", "variant": label, "classes": classes}
                )
                save(ns.out, ns, results)
                print(label, json.dumps(classes), flush=True)
    save(ns.out, ns, results)
    print(table(results))


if __name__ == "__main__":
    main()
