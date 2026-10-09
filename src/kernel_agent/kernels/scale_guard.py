"""The scale-rule guard of block-scaled activations, run by the evaluator: MXFP8 for
``fp8_mx`` targets (issue #144), NVFP4 / MXFP4 for ``fp4_w4a4`` targets (W4A4, issue #233).

MXFP8 (``precision: fp8_mx``) scales each block of 32 elements along K by a power of two.
The OCP reference rule ``2^(floor(log2 amax) - 8)`` maps block maxima in [256, 512) x scale
onto e4m3, whose largest value is 448: maxima above 448 x scale saturate. On activations
with massive outlier channels that shrinks outputs by up to 12.5 % and fails the
near-lossless tier (docs/FP8.md §3.2: VoxCPM2's LocDiT o_proj / down_proj, norm off by
4.0 % / 2.0 %); on inputs without such channels it passes the tier while still clamping a
quarter of the blocks. The rule ``2^ceil(log2(amax / 448))`` never saturates and has
per-token W8A8's error (:func:`kernel_agent.kernels.quant.quantize_mxfp8`).

W4A4 (``precision: fp4_w4a4``) quantises the activations to e2m1 (largest value 6) with an
e4m3 scale per 16 elements and an fp32 outer scale per token or per call (NVFP4), or an
e8m0 scale per 32 (MXFP4). The reference (:func:`kernel_agent.kernels.quant.
quantize_fp4_activations`) rounds the NVFP4 block scale ``bmax / (outer x 6)`` to nearest,
so a block maximum reaches at most 6.375 x step (scale x outer) with a normal scale; a scale
rounded down reaches 6.75, one clamped at 448 by too small an outer scale more, and a
subnormal scale flushed to zero zeroes its block (:func:`kernel_agent.kernels.quant.
fp4_scale_check`). The MXFP4 rule ``2^ceil(log2(amax / 6))`` never saturates; the OCP rule
``2^(floor(log2 amax) - 2)`` saturates maxima in (6, 8) x scale. Such errors are biased
(they shrink the largest elements, as e2m1's grid already shrinks a GEMM's output by up to
~1 %) and can stay inside the looser ``near-lossless-fp4a`` tier on one input while failing
on another: the guard names them on every input.

A candidate of either precision exposes the activation quantiser its kernels use as a
module-level ``quantize_activations(x)`` (:data:`QUANTIZER`), the hook of both bundled
examples of each:

* ``fp8_mx``: ``-> (codes, scales)``, codes e4m3 ``[rows, K]``, scales ``[rows, K / 32]``
  unswizzled (e8m0, uint8 biased exponents or values);
* ``fp4_w4a4``: ``-> (codes, scales, outer)``: codes packed two e2m1 per byte ``[rows, K /
  2]`` (as :func:`~kernel_agent.kernels.quant.quantize_fp4_activations`; not read by the
  guard: the tiers judge them), scales unswizzled ``[rows, K / 16]`` e4m3 (NVFP4) or ``[rows,
  K / 32]`` e8m0 (MXFP4; uint8 bits or values too), ``outer`` fp32 ``[rows]`` or one value
  per call (MXFP4: may be left out, ``(codes, scales)``); a quantiser that rotates its input
  first (``quant.hadamard_rotate``) returns the rotated activations as a fourth element,
  which its scales are checked against. The swizzled 128 x 4 layout the
  GEMMs read (``swizzle_fp4_scales``) is refused with that reason: the hook returns the
  unswizzled scales (both examples have that path).

:func:`check` runs it on the captured input of the main case and on a stress input of the
same shape (:func:`kernel_agent.kernels.quant.mxfp8_stress_input`, :func:`kernel_agent.
kernels.quant.fp4_stress_input`: blocks whose maxima sit where a saturating rule shows) and
rejects the candidate when its scales saturate (or, W4A4, flush non-zero blocks to scale 0),
or the quantiser fails, with that reason (``status: incorrect``, ``stage: scale_rule``). A
candidate without the function is not rejected for it; its report says the rule went
unchecked. The report also gives the captured input's outlier statistics (the largest
``|x| / RMS`` of a row, the largest channel amax over the median channel's), measured on the
model at hand.
"""

from __future__ import annotations

from typing import Any

import torch

from kernel_agent.kernels import quant

#: The module-level function of a candidate the guard runs.
QUANTIZER = "quantize_activations"
#: The precisions whose candidates the evaluator guards, and the block of K their activation
#: input must be a whole number of (W4A4: NVFP4's 16; an MXFP4 quantiser refuses K % 32).
GUARDED = {"fp8_mx": quant.MX_BLOCK, "fp4_w4a4": quant.FP4_FORMATS["nvfp4"][0]}
#: What the note of an unchecked candidate says the rule must be.
_RULES = {
    "fp8_mx": (
        "(codes, scales): its MXFP8 scale rule is not checked (it must be "
        "2^ceil(log2(amax / 448)); the OCP rule 2^(floor(log2 amax) - 8) saturates block "
        "maxima above 448 x scale)"
    ),
    "fp4_w4a4": (
        "(codes, scales, outer): its W4A4 scale rule is not checked (NVFP4: the e4m3 block "
        "scale bmax / (outer x 6) rounded to nearest, outer = amax / (6 x 448); MXFP4: "
        "2^ceil(log2(amax / 6)))"
    ),
}


def activation(cases: list[dict[str, Any]], block: int = quant.MX_BLOCK) -> torch.Tensor | None:
    """The input the guard quantises: the first floating-point tensor argument of the case
    with the most calls per run whose last dimension is a multiple of ``block`` (the
    module's input), or None."""
    ranked = sorted(cases, key=lambda c: -int(c.get("count") or 0))
    for case in ranked:
        for value in [*case.get("args", ()), *(case.get("kwargs") or {}).values()]:
            if (
                isinstance(value, torch.Tensor)
                and value.is_floating_point()
                and value.dim() >= 2
                and value.shape[-1] % block == 0
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


def _mxfp8(sample: torch.Tensor, out: Any) -> tuple[dict[str, Any], str | None]:
    """The MXFP8 summary and problem of a quantiser's output ``(codes, scales)``."""
    scales = out[1]
    found = quant.mxfp8_saturation(sample, scales)
    summary = {k: found[k] for k in ("blocks", "saturated", "worst_ratio", "coarser")}
    return summary, quant.mxfp8_scale_problem(sample, scales)


def _fp4(sample: torch.Tensor, out: Any, fmt: str | None) -> tuple[dict[str, Any], str | None]:
    """The W4A4 summary and problem of a quantiser's output ``(codes, scales[, outer[,
    quantised]])``: ``quantised``, the activations it quantised when that is not ``sample``
    itself (a quantiser that rotates first, ``quant.hadamard_rotate``), of its shape."""
    scales, outer = out[1], (out[2] if len(out) > 2 else None)
    x = out[3] if len(out) > 3 and out[3] is not None else sample
    if tuple(x.shape[-1:]) != tuple(sample.shape[-1:]) or x.numel() != sample.numel():
        raise ValueError(f"the quantised activations {tuple(x.shape)} are not of the input's size")
    found = quant.fp4_scale_check(x, scales, outer, fmt)
    keys = ("format", "blocks", "saturated", "worst_ratio", "underflow", "coarser")
    return {k: found[k] for k in keys}, quant.fp4_scale_problem(x, scales, outer, fmt)


def check(module: Any, cases: list[dict[str, Any]], precision: str = "fp8_mx") -> dict[str, Any]:
    """The scale-rule report of a candidate of ``precision`` (:data:`GUARDED`; ``module``: its
    file's module): ``ok``, ``checked``, per input (``captured``, ``stress``) its summary
    (:func:`~kernel_agent.kernels.quant.mxfp8_saturation`, :func:`~kernel_agent.kernels.quant.
    fp4_scale_check`: ``blocks``, ``saturated``, ``worst_ratio``, ``coarser``; W4A4 also
    ``format`` and ``underflow``), ``outliers`` of the captured input, and ``error`` (the
    reason) when it fails."""
    block = GUARDED.get(precision, quant.MX_BLOCK)
    x = activation(cases, block)
    report: dict[str, Any] = {"ok": True, "checked": False}
    if x is not None:
        report["outliers"] = outliers(x)
    fn = getattr(module, QUANTIZER, None)
    if not callable(fn):
        rule = _RULES.get(precision, _RULES["fp8_mx"])
        report["note"] = f"no module-level {QUANTIZER}(x) -> {rule} in the candidate"
        return report
    if x is None:
        report["note"] = (
            f"no floating-point input with K a multiple of {block}: nothing to quantise"
        )
        return report
    report["checked"] = True
    fmt: str | None = None  # W4A4: the format of the captured run's scales, for the stress
    for name in ("captured", "stress"):
        if name == "captured":
            sample = x
        elif precision == "fp4_w4a4":
            sample = quant.fp4_stress_input(
                tuple(x.shape), fmt or "nvfp4", dtype=x.dtype, device=x.device
            )
        else:
            sample = quant.mxfp8_stress_input(tuple(x.shape), dtype=x.dtype, device=x.device)
        what = f"{QUANTIZER} on the {name} input {list(sample.shape)} {sample.dtype}"
        try:
            with torch.inference_mode():
                out = fn(sample.clone())
            if precision == "fp4_w4a4":
                fmt = fmt or quant.fp4_format_of(sample.shape[-1], out[1])
                summary, problem = _fp4(sample, out, fmt)
            else:
                summary, problem = _mxfp8(sample, out)
        except Exception as exc:  # the candidate's function, or scales of the wrong shape
            report.update(ok=False, input=name, error=f"{what}: {type(exc).__name__}: {exc}"[:600])
            return report
        report[name] = summary
        if problem is not None:
            report.update(ok=False, input=name, error=f"{what}: {problem}")
            return report
    return report
