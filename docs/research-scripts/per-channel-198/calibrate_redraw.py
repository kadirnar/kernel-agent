"""The redrawn-input check per channel (#198): the #175 calibration before and after.

``kernels.verify.perturb_`` redraws a floating-point input from each channel's mean and std
(``verify.redraw_stats``: one position of the last dimension, over all the other dimensions;
the tensor's for fewer than 16 values) and keeps rotary tables (``verify.rotary_tables``:
cos / sin of the positions); before #198 it drew every tensor, rotary tables too, from the
tensor's mean and std. This runs ``../relaxed-175/calibrate_tiers.py`` (the reference math
of every reduced precision and the broken variants, on the captured, redrawn and x 3 /
x 0.01 / x -1 inputs, judged in the near-lossless and the relaxed tier at once) with

* ``--redraw after``: the evaluator's redraw;
* ``--redraw channel``: per-channel statistics, rotary tables redrawn;
* ``--redraw before``: the redraw before #198 (global statistics, rotary tables redrawn),

on the same seeds (the scaled checks are the same in all three), plus the INT8 recipes of
#178 (``quant.int8_weights_linear`` / ``int8_w8a8_linear``, and their weight scales x 1.05)
and one more broken W8A8 variant: per-token activation scales cached from the first call per
input shape (a kernel that skips the amax pass after its first call). ``summarize.py``
writes ``results.md`` (before -> after) and ``results_channel_only.md``; ``int8/``: #178's
``calib_int8.py`` with this redraw (``*_before.out``: the old one on the same capture).

    python calibrate_redraw.py --redraw after --device cuda --seeds 10 \
        --out results_after.json CAPTURE.pt ...
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "relaxed-175"))

import calibrate_tiers as ct

from kernel_agent.kernels import quant, verify

CACHED = "cached activation scales"
REDRAWS = ("before", "channel", "after")


class QLinear(ct.QLinear):
    """``calibrate_tiers.QLinear``, the INT8 recipes of #178 (``int8_weights``,
    ``int8_w8a8``: ``quant.int8_*_linear``) and a W8A8 kernel that caches its per-token
    activation scales from the first call per input shape."""

    def __init__(self, linear: torch.nn.Linear, kind: str, bug: str | None) -> None:
        if kind.startswith("int8"):
            torch.nn.Module.__init__(self)
            self.bias, self.kind, self.bug = linear.bias, kind, bug
            self.q, self.s = quant.quantize_int8(linear.weight.detach())
            if bug and bug.startswith("scales x"):
                self.s = self.s * float(bug.removeprefix("scales x"))
        else:
            super().__init__(linear, kind, bug)
        self.cache: dict[tuple[int, ...], torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "int8_weights":
            return quant.int8_weights_linear(x, self.q, self.s, self.bias)
        if self.kind == "int8_w8a8":
            return quant.int8_w8a8_linear(x, self.q, self.s, self.bias)
        if self.bug != CACHED:
            return super().forward(x)
        _, xs = quant.quantize_fp8_activations(x)
        xs = self.cache.setdefault(tuple(x.shape), xs)
        a = x.detach().reshape(-1, x.shape[-1]).float()
        xq = (a / xs[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        y = (xq.float() * xs[:, None]) @ (self.q.float() * self.s[:, None]).T
        if self.bias is not None:
            y = y + self.bias.float()
        return y.to(x.dtype).reshape(*x.shape[:-1], -1)


def global_stats(values: torch.Tensor) -> tuple[Any, Any]:
    """The redraw statistics before #198: the tensor's non-zero mean and std."""
    return verify._global_stats(values)


def no_rotary_tables(tensors: list[torch.Tensor]) -> set[int]:
    """Before #198 rotary tables were redrawn and scaled like any other input."""
    return set()


def main() -> None:
    redraw = "after"
    if "--redraw" in sys.argv:
        i = sys.argv.index("--redraw")
        redraw = sys.argv[i + 1]
        del sys.argv[i : i + 2]
    if redraw not in REDRAWS:
        raise SystemExit(f"--redraw {'|'.join(REDRAWS)}, not {redraw}")
    if redraw in ("before", "channel"):
        verify.rotary_tables = no_rotary_tables  # type: ignore[assignment]
    if redraw == "before":
        verify.redraw_stats = global_stats  # type: ignore[assignment]
    print(f"redraw: {redraw}", flush=True)
    ct.QLinear = QLinear  # type: ignore[misc]
    ct.VARIANTS.append((f"BUG w8a8: {CACHED}", "fp8_w8a8", "fp8_w8a8", CACHED, None))
    ct.VARIANTS += [
        ("INT8 weights", "int8_weights", "int8_weights", None, None),
        ("INT8 W8A8", "int8_w8a8", "int8_w8a8", None, None),
        ("BUG int8: scales x1.05", "int8_weights", "int8_weights", "scales x1.05", None),
        ("BUG int8 w8a8: scales x1.05", "int8_w8a8", "int8_w8a8", "scales x1.05", None),
    ]
    ct.main()


if __name__ == "__main__":
    main()
