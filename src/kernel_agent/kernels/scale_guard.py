"""The MXFP8 scale-rule guard of ``fp8_mx`` targets (issue #144), run by the evaluator.

MXFP8 (``precision: fp8_mx``) scales each block of 32 elements along K by a power of two.
The OCP reference rule ``2^(floor(log2 amax) - 8)`` maps block maxima in [256, 512) x scale
onto e4m3, whose largest value is 448: maxima above 448 x scale saturate. On activations
with massive outlier channels that shrinks outputs by up to 12.5 % and fails the
near-lossless tier (docs/FP8.md §3.2: VoxCPM2's LocDiT o_proj / down_proj, norm off by
4.0 % / 2.0 %); on inputs without such channels it passes the tier while still clamping a
quarter of the blocks. The rule ``2^ceil(log2(amax / 448))`` never saturates and has
per-token W8A8's error (:func:`kernel_agent.kernels.quant.quantize_mxfp8`).

An ``fp8_mx`` candidate exposes the activation quantiser its kernels use as a module-level
``quantize_activations(x) -> (codes, scales)`` (:data:`QUANTIZER`; codes e4m3 ``[rows, K]``,
scales ``[rows, K / 32]`` unswizzled: e8m0, uint8 biased exponents or values). :func:`check`
runs it on the captured input of the main case and on a stress input of the same shape
(:func:`kernel_agent.kernels.quant.mxfp8_stress_input`: blocks whose maxima sit where the
OCP rule saturates) and rejects the candidate when a block maximum exceeds 448 x scale, or
the quantiser fails, with that reason (``status: incorrect``, ``stage: scale_rule``). A
candidate without the function is not rejected for it; its report says the rule went
unchecked. The report also gives the captured input's outlier statistics (the largest
``|x| / RMS`` of a row, the largest channel amax over the median channel's), measured on the
model at hand.
"""

from __future__ import annotations

from typing import Any

import torch

from kernel_agent.kernels import quant

#: The module-level function of an ``fp8_mx`` candidate the guard runs.
QUANTIZER = "quantize_activations"


def activation(cases: list[dict[str, Any]]) -> torch.Tensor | None:
    """The input the guard quantises: the first floating-point tensor argument of the case
    with the most calls per run whose last dimension is a multiple of 32 (the module's
    input), or None."""
    ranked = sorted(cases, key=lambda c: -int(c.get("count") or 0))
    for case in ranked:
        for value in [*case.get("args", ()), *(case.get("kwargs") or {}).values()]:
            if (
                isinstance(value, torch.Tensor)
                and value.is_floating_point()
                and value.dim() >= 2
                and value.shape[-1] % quant.MX_BLOCK == 0
                and value.numel() > 0
            ):
                return value
    return None


def outliers(x: torch.Tensor) -> dict[str, float]:
    """Outlier statistics of activations ``x [..., K]``: ``crest``, the largest ``amax / RMS``
    of a row (many e4m3 steps lost to one element: ~4 for Gaussian rows, tens with massive
    activations), and ``channel_ratio``, the largest channel amax over the median channel's
    (a massive-activation channel: hundreds or more)."""
    a = x.detach().reshape(-1, x.shape[-1]).float()
    a = torch.where(torch.isfinite(a), a, torch.zeros_like(a))
    rms = a.pow(2).mean(dim=1).sqrt()
    crest = a.abs().amax(dim=1) / rms.clamp_min(1e-30)
    channel = a.abs().amax(dim=0)
    median = float(channel.median())
    return {
        "crest": round(float(crest[rms > 0].max()), 1) if bool((rms > 0).any()) else 0.0,
        "channel_ratio": round(float(channel.max()) / median, 1) if median > 0 else 0.0,
    }


def check(module: Any, cases: list[dict[str, Any]]) -> dict[str, Any]:
    """The scale-rule report of an ``fp8_mx`` candidate (``module``: its file's module):
    ``ok``, ``checked``, per input (``captured``, ``stress``) its
    :func:`~kernel_agent.kernels.quant.mxfp8_saturation` summary, ``outliers`` of the
    captured input, and ``error`` (the reason) when it fails."""
    x = activation(cases)
    report: dict[str, Any] = {"ok": True, "checked": False}
    if x is not None:
        report["outliers"] = outliers(x)
    fn = getattr(module, QUANTIZER, None)
    if not callable(fn):
        report["note"] = (
            f"no module-level {QUANTIZER}(x) -> (codes, scales) in the candidate: its MXFP8 "
            "scale rule is not checked (it must be 2^ceil(log2(amax / 448)); the OCP rule "
            "2^(floor(log2 amax) - 8) saturates block maxima above 448 x scale)"
        )
        return report
    if x is None:
        report["note"] = "no floating-point input with K a multiple of 32: nothing to quantise"
        return report
    report["checked"] = True
    stress = quant.mxfp8_stress_input(tuple(x.shape), dtype=x.dtype, device=x.device)
    for name, sample in (("captured", x), ("stress", stress)):
        what = f"{QUANTIZER} on the {name} input {list(sample.shape)} {sample.dtype}"
        try:
            with torch.inference_mode():
                out = fn(sample.clone())
            scales = out[1]
            found = quant.mxfp8_saturation(sample, scales)
        except Exception as exc:  # the candidate's function, or scales of the wrong shape
            report.update(ok=False, input=name, error=f"{what}: {type(exc).__name__}: {exc}"[:600])
            return report
        report[name] = {k: found[k] for k in ("blocks", "saturated", "worst_ratio", "coarser")}
        problem = quant.mxfp8_scale_problem(sample, scales)
        if problem is not None:
            report.update(ok=False, input=name, error=f"{what}: {problem}")
            return report
    return report
