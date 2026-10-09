"""Calibrate the W4A4 tiers (#233: ``near-lossless-fp4a`` / ``relaxed-fp4a``) on real captures.

Runs ``../relaxed-175/calibrate_tiers.py`` (each capture's module with every ``nn.Linear``
replaced by a recipe's reference math, judged by ``kernels.compare`` on the captured inputs,
on redrawn inputs (``verify.perturb_``, per channel since #198) and on the x 3 / x 0.01 /
x -1 checks, in the near-lossless and the relaxed tier of the variant's precision at once)
with the W4A4 recipes of ``kernels/quant.py`` (``quantize_fp4`` weights,
``quantize_fp4_activations`` per call, ``fp4_w4a4_linear``: ``F.scaled_mm`` NVFP4 on the
GPU) and broken W4A4 kernels:

* NVFP4 W4A4 (e2m1 + e4m3 per 16, the activations' outer fp32 scale per token; per tensor;
  with a block Hadamard rotation of 16), MXFP4 W4A4 (e8m0 per 32; with a Hadamard 32);
* broken: weight tensor scale x 1.05 / x 1.2, nibbles swapped (weights), activation block
  scales shifted by one block, the first token's outer scale for every token, activation
  codes truncated instead of rounded (``cvt.rz``), activation scales cached from the first
  call per shape, a row of every GEMM skipped, a KV head dropped, q heads swapped, int4 per
  tensor (the ``calibrate_tiers`` edits and bugs).

``--linears`` prints instead, per capture (its first case), every ``nn.Linear`` alone:
its input's token crest and the output's relative L2 error / norm change of W4A4 NVFP4,
NVFP4 + Hadamard 16, MXFP4 and FP8 W8A8 against the exact GEMM (``quant.fp4_w4a4_error``):
which layers are sensitive.

    python calibrate_w4a4.py --device cuda --seeds 10 --out results.json CAPTURE.pt ...
    python calibrate_w4a4.py --linears CAPTURE.pt ...
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "relaxed-175"))

import calibrate_tiers as ct

from kernel_agent.kernels import quant

CACHED = "cached activation scales"
ACT_BUGS = (
    "first token's outer scale",
    "activation block scales shifted",
    "activation codes truncated",
    CACHED,
)


def _recipe(kind: str) -> tuple[str, str, int | None]:
    """(fmt, granularity, rotate) of a ``w4a4_*`` kind."""
    fmt = "mxfp4" if "mxfp4" in kind else "nvfp4"
    granularity = "tensor" if kind.endswith("_tensor") else "token"
    rotate = 16 if kind.endswith("_h16") else 32 if kind.endswith("_h32") else None
    return fmt, granularity, rotate


class QLinear(ct.QLinear):
    """``calibrate_tiers.QLinear`` plus the W4A4 recipes (``kind`` ``w4a4_*``)."""

    def __init__(self, linear: nn.Linear, kind: str, bug: str | None) -> None:
        if not kind.startswith("w4a4") or bug == "int4 per tensor":
            super().__init__(linear, kind, bug)
            return
        nn.Module.__init__(self)
        self.bias, self.kind, self.bug = linear.bias, kind, bug
        self.fmt, self.granularity, self.rotate = _recipe(kind)
        w = linear.weight.detach()
        if self.rotate:
            w = quant.hadamard_rotate(w, self.rotate).to(w.dtype)
        codes, scales, ts = quant.quantize_fp4(w.float(), self.fmt)
        if bug == "nibbles swapped":
            codes = (codes >> 4) | ((codes & 0xF) << 4)
        if bug and bug.startswith("weight scale x"):
            ts = ts * float(bug.removeprefix("weight scale x"))
        self.codes, self.scales, self.ts = codes, scales, ts
        self.cache: dict[tuple[int, ...], torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.kind.startswith("w4a4") or self.bug == "int4 per tensor":
            return super().forward(x)
        if self.bug not in ACT_BUGS:
            return quant.fp4_w4a4_linear(
                x,
                self.codes,
                self.scales,
                self.ts,
                self.bias,
                fmt=self.fmt,
                granularity=self.granularity,
                rotate=self.rotate,
            )
        a = x.detach().reshape(-1, x.shape[-1]).float()
        if self.rotate:
            a = quant.hadamard_rotate(a, self.rotate)
        xq, xs, outer = quant.quantize_fp4_activations(a, self.fmt, self.granularity)
        block = quant.FP4_FORMATS[self.fmt][0]
        if self.bug == "first token's outer scale" or self.bug == CACHED:
            if self.bug == CACHED:
                outer = self.cache.setdefault(tuple(x.shape), outer)
            else:
                outer = outer[:1].expand_as(outer).contiguous()
            # the block scales and codes of a kernel that kept that outer scale
            blocks = a.reshape(a.shape[0], -1, block)
            bmax = blocks.abs().amax(-1)
            xs = (bmax / (outer * 6.0)[:, None]).clamp(max=448.0).to(xs.dtype)
            xq = quant._fp4_codes(blocks, xs.float() * outer[:, None])
        elif self.bug == "activation block scales shifted":
            xs = xs.float().roll(1, dims=1).to(xs.dtype)
        elif self.bug == "activation codes truncated":
            blocks = a.reshape(a.shape[0], -1, block)
            step = (xs.float() * outer[:, None])[..., None].clamp_min(1e-38)
            v = (blocks / step).clamp(-6, 6)
            lut = torch.tensor(quant.E2M1_VALUES, device=v.device)
            mag = v.abs()
            idx = (mag[..., None] >= lut).sum(-1) - 1  # round toward zero
            codes = (idx + 8 * ((v < 0) & (idx > 0))).to(torch.uint8).reshape(a.shape[0], -1)
            xq = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
        acc = quant.fp4_values(xq, xs) @ quant.fp4_values(self.codes, self.scales).T
        y = acc * (outer[:, None] * float(self.ts))
        if self.bias is not None:
            y = y + self.bias.float()
        return y.to(x.dtype).reshape(*x.shape[:-1], -1)


W4A4 = "fp4_w4a4"
VARIANTS: list[tuple[str, str, str, str | None, str | None]] = [
    ("NVFP4 W4A4", W4A4, "w4a4_nvfp4", None, None),
    ("NVFP4 W4A4, per-tensor activations", W4A4, "w4a4_nvfp4_tensor", None, None),
    ("NVFP4 W4A4 + Hadamard 16", W4A4, "w4a4_nvfp4_h16", None, None),
    ("MXFP4 W4A4", W4A4, "w4a4_mxfp4", None, None),
    ("MXFP4 W4A4 + Hadamard 32", W4A4, "w4a4_mxfp4_h32", None, None),
    ("BUG w4a4: weight scale x1.05", W4A4, "w4a4_nvfp4", "weight scale x1.05", None),
    ("BUG w4a4: weight scale x1.2", W4A4, "w4a4_nvfp4", "weight scale x1.2", None),
    ("BUG w4a4: nibbles swapped", W4A4, "w4a4_nvfp4", "nibbles swapped", None),
    ("BUG w4a4: activation block scales shifted", W4A4, "w4a4_nvfp4", ACT_BUGS[1], None),
    ("BUG w4a4: first token's outer scale", W4A4, "w4a4_nvfp4", ACT_BUGS[0], None),
    ("BUG w4a4: activation codes truncated", W4A4, "w4a4_nvfp4", ACT_BUGS[2], None),
    ("BUG w4a4: cached activation scales", W4A4, "w4a4_nvfp4", CACHED, None),
    ("BUG w4a4: int4 per tensor", W4A4, "w4a4_nvfp4", "int4 per tensor", None),
    ("BUG w4a4: row 0 of every GEMM skipped", W4A4, "w4a4_nvfp4", None, "row 0 skipped"),
    ("BUG w4a4: KV head 0 dropped", W4A4, "w4a4_nvfp4", None, "KV head 0 dropped"),
    (
        "BUG w4a4: q heads 0/last swapped (layout)",
        W4A4,
        "w4a4_nvfp4",
        None,
        "q heads 0 and last swapped",
    ),
]


def linears(paths: list[Path], device: str) -> None:
    """Every nn.Linear of each capture's first case alone: W4A4 recipes vs FP8 W8A8."""
    from kernel_agent.profiling.capture import load_capture
    from kernel_agent.profiling.state import Replay

    recipes = [("nvfp4", None), ("nvfp4", 16), ("mxfp4", None), ("mxfp4", 32)]
    for path in paths:
        capture = load_capture(path, device=device)
        module = capture["module"].eval()
        seen: dict[str, torch.Tensor] = {}

        def keep(mod: nn.Module, args: tuple, name: str) -> None:
            seen.setdefault(name, args[0].detach())

        hooks = [
            m.register_forward_pre_hook(lambda mod, a, n=n: keep(mod, a, n))
            for n, m in module.named_modules()
            if isinstance(m, nn.Linear)
        ]
        case = capture["cases"][0]
        fn = Replay(capture, module).call(case, module)
        with torch.inference_mode():
            fn(*copy.deepcopy(case["args"]), **copy.deepcopy(case["kwargs"]))
        for h in hooks:
            h.remove()
        print(f"\n## {path.parent.parent.parent.name}/{path.stem}")
        head = " | ".join(
            f"{f}{' H' + str(r) if r else ''} rel / norm" for f, r in recipes
        )
        print(f"| nn.Linear (rows x K -> N, token crest) | {head} | FP8 W8A8 rel / norm |")
        print("|---|" + "---|" * (len(recipes) + 1))
        mods = dict(module.named_modules())
        for name, x in seen.items():
            w = mods[name].weight.detach().float()
            cells = []
            for fmt, rot in recipes:
                wr = quant.hadamard_rotate(w, rot) if rot else w
                q = quant.quantize_fp4(wr, fmt)
                e = quant.fp4_w4a4_error(w, *q, x, fmt=fmt, rotate=rot)
                cells.append(f"{e['output_rel_l2']:.3f} / {(e['output_norm_ratio'] - 1) * 100:+.1f} %")
                crest = e["activation_crest"]
            e8 = quant.fp8_w8a8_error(w, *quant.quantize_fp8(w), x)
            cells.append(f"{e8['output_rel_l2']:.3f} / {(e8['output_norm_ratio'] - 1) * 100:+.1f} %")
            rows = x.reshape(-1, x.shape[-1]).shape[0]
            print(f"| {name} ({rows} x {w.shape[1]} -> {w.shape[0]}, {crest:.0f}) | " + " | ".join(cells) + " |")
        del capture, module


def main() -> None:
    if "--linears" in sys.argv:
        sys.argv.remove("--linears")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        linears([Path(p) for p in sys.argv[1:]], device)
        return
    ct.QLinear = QLinear  # type: ignore[misc]
    ct.VARIANTS[:] = VARIANTS
    ct.main()


if __name__ == "__main__":
    main()
