"""Structural comparison of nested module outputs (tensors, tuples, dicts, caches).

A floating-point tensor matches its reference when

* it is a plain ``torch.Tensor`` (no subclass: a ``__torch_function__`` override
  could compute lazily, after the timer stopped) of the same shape, dtype and
  device type;
* NaN, +inf and -inf sit at exactly the same positions;
* at most :data:`MAX_MISMATCH` of the finite elements are outside ``atol + rtol ·
  |ref|`` (:data:`TOLERANCES`), and none of them by more than :data:`MAX_OUTLIER`
  times that tolerance;
* tensors with more than :data:`GLOBAL_MIN_NUMEL` elements whose reference has
  signal above the absolute tolerance (``‖ref‖ > atol · √n``) also match as a
  whole: cosine similarity and relative L2 error ``‖new − ref‖ / ‖ref‖`` within
  :data:`GLOBAL_TOLERANCES`.

Integer and boolean tensors must match exactly. In the ``near-lossless`` tier
(:data:`TIERS`: ``--quality near-lossless`` and a target whose spec allows reduced
precision) whole-tensor bounds replace the per-element tolerances.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import torch

#: (atol, rtol) per dtype.  Fused kernels change accumulation order, so these
#: are looser than ``torch.testing`` defaults but tight enough to catch bugs.
TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    torch.float32: (1e-4, 1e-4),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (2e-2, 2e-2),
}
#: Fraction of elements allowed outside tolerance (rounding-boundary flips).
MAX_MISMATCH = 1e-3
#: ... and none of them by more than this multiple of its tolerance.
MAX_OUTLIER = 10.0
#: Tensors with more elements than this are also compared as a whole.
GLOBAL_MIN_NUMEL = 1024
#: (min cosine, max relative L2 error) per dtype for that whole-tensor check.  On
#: VoxCPM2 modules, fp32-accumulating and MATH-SDPA variants of bf16 attention and
#: MLP stay below a relative L2 error of 0.0035 (cosine >= 0.99999), also on
#: perturbed inputs.
GLOBAL_TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    torch.float32: (0.99999, 0.005),
    torch.float16: (0.9995, 0.01),
    torch.bfloat16: (0.999, 0.02),
}
#: Tolerance tiers. ``exact``: the checks above. ``near-lossless``
#: (``--quality near-lossless``, a target whose spec allows reduced precision,
#: ``"precision": "reduced"``; recorded in its capture as ``tier``): reduced-precision
#: weights (FP8) move every element of a GEMM's output by a few per cent of the output's
#: RMS, so the per-element (atol, rtol) checks are replaced by whole-tensor bounds. Tensors
#: without signal keep the exact checks. Calibrated on FP8 (e4m3, per output channel)
#: weight-only nn.Linear GEMVs / GEMMs, MLPs and the LocDiT estimator of VoxCPM2 on real
#: inputs: cosine >= 0.9985, relative L2 error <= 0.055, norm within 0.7 %, every element
#: within 0.47 of its bound below (README, "Quality modes").
EXACT_TIER = "exact"
NEAR_LOSSLESS_TIER = "near-lossless"
TIERS = (EXACT_TIER, NEAR_LOSSLESS_TIER)
NEAR_LOSSLESS_MIN_COSINE = 0.996
NEAR_LOSSLESS_MAX_REL_L2 = 0.08
#: ``‖new‖ / ‖ref‖`` within this of 1: rounding noise is unbiased, a wrong scale is not.
NEAR_LOSSLESS_MAX_NORM_CHANGE = 0.02
#: Every element within ``a * RMS(ref) + r * |ref|`` (``(a, r)``): a corrupted element or
#: row fails, a massive activation keeps FP8's relative rounding step.
NEAR_LOSSLESS_ELEMENT = (0.5, 0.125)
#: The tier of this process's comparisons when a call passes none. The evaluator sets it
#: from the capture before the candidate is imported, so the integrity snapshot
#: (:mod:`kernel_agent.kernels.integrity`) watches it like the constants above.
TIER = EXACT_TIER
_PLAIN = (torch.Tensor, torch.nn.Parameter)


def tier_of(capture: dict[str, Any] | None) -> str:
    """The tolerance tier recorded in a capture (``exact`` when none or unknown)."""
    name = (capture or {}).get("tier")
    return name if name in TIERS else EXACT_TIER


def tier_for(quality: str | None, precision: str | None) -> str:
    """The tier of a target: ``near-lossless`` when the run's quality mode is
    ``near-lossless`` and the target's spec allows reduced precision (``"reduced"``)."""
    near = quality == NEAR_LOSSLESS_TIER and precision == "reduced"
    return NEAR_LOSSLESS_TIER if near else EXACT_TIER


def flatten(value: Any, prefix: str = "out", depth: int = 0) -> dict[str, torch.Tensor]:
    """Flatten nested containers / dataclasses / cache objects into named tensors."""
    found: dict[str, torch.Tensor] = {}
    if depth > 6 or value is None:
        return found
    if isinstance(value, torch.Tensor):
        found[prefix] = value
    elif isinstance(value, dict):
        for k, v in value.items():
            found.update(flatten(v, f"{prefix}.{k}", depth + 1))
    elif isinstance(value, tuple | list):
        for i, v in enumerate(value):
            found.update(flatten(v, f"{prefix}[{i}]", depth + 1))
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            found.update(flatten(getattr(value, f.name, None), f"{prefix}.{f.name}", depth + 1))
    elif hasattr(value, "__dict__") and not isinstance(value, torch.nn.Module):
        for k, v in vars(value).items():
            if not k.startswith("__"):
                found.update(flatten(v, f"{prefix}.{k}", depth + 1))
    return found


def type_error(new: Any) -> str | None:
    """Why ``new`` cannot stand in for a reference tensor (None if it can)."""
    if type(new) in _PLAIN:
        return None
    if isinstance(new, torch.Tensor):
        return (
            f"got a {type(new).__name__} (a torch.Tensor subclass); return plain tensors "
            "(a subclass can defer the computation until the result is read)"
        )
    return f"expected tensor, got {type(new).__name__}"


def _non_finite_error(a: torch.Tensor, b: torch.Tensor) -> str | None:
    """Positions of NaN / +inf / -inf must be identical in reference and candidate."""
    problems = []
    for label, in_ref, in_new in (
        ("NaN", torch.isnan(a), torch.isnan(b)),
        ("+inf", a == math.inf, b == math.inf),
        ("-inf", a == -math.inf, b == -math.inf),
    ):
        differ = int((in_ref != in_new).sum())
        if differ:
            problems.append(
                f"{label} at {differ} positions that differ "
                f"(candidate {int(in_new.sum())}, reference {int(in_ref.sum())})"
            )
    return "; ".join(problems) or None


def compare_tensors(
    name: str,
    ref: torch.Tensor,
    new: torch.Tensor,
    tol: tuple[float, float] | None = None,
    *,
    tier: str | None = None,
) -> dict[str, Any]:
    """One tensor against its reference (module docstring); ``tier``: the tolerance tier
    (default :data:`TIER`)."""
    result: dict[str, Any] = {"name": name, "ok": False}
    error = type_error(new)
    if error is not None:
        result["error"] = error
        return result
    if ref.shape != new.shape:
        result["error"] = f"shape {tuple(new.shape)} != reference {tuple(ref.shape)}"
        return result
    if ref.dtype != new.dtype:
        result["error"] = f"dtype {new.dtype} != reference {ref.dtype}"
        return result
    if new.layout != ref.layout or new.device.type != ref.device.type:
        result["error"] = (
            f"{new.layout} tensor on {new.device.type} != reference "
            f"{ref.layout} tensor on {ref.device.type}"
        )
        return result
    if not ref.is_floating_point():
        mismatches = (ref.to(new.device) != new).float().mean().item() if ref.numel() else 0.0
        result.update(ok=mismatches == 0, mismatch_frac=mismatches)
        return result
    atol, rtol = tol or TOLERANCES.get(ref.dtype, (1e-3, 1e-3))
    a = ref.detach().to(new.device).float()
    b = new.detach().float()
    error = _non_finite_error(a, b)
    if error is not None:
        result["error"] = error
        return result
    finite = torch.isfinite(a)
    if not bool(finite.all()):  # identical non-finite positions: compare the rest
        a, b = a[finite], b[finite]
    diff = (a - b).abs()
    allowed = atol + rtol * a.abs()
    mismatch = (diff > allowed).float().mean().item() if a.numel() else 0.0
    outlier = float((diff / allowed.clamp_min(1e-30)).max()) if a.numel() else 0.0
    ref_norm, new_norm = float(a.norm()), float(b.norm())
    denom = ref_norm * new_norm
    cos = float((a.flatten() @ b.flatten()) / denom) if denom > 0 else 1.0
    rel_l2 = float(diff.norm()) / ref_norm if ref_norm > 0 else 0.0
    result.update(
        max_abs_err=float(diff.max()) if diff.numel() else 0.0,
        mean_abs_err=float(diff.mean()) if diff.numel() else 0.0,
        mismatch_frac=round(mismatch, 6),
        max_err_ratio=round(outlier, 3),
        cosine=round(cos, 7),
        rel_l2=float(f"{rel_l2:.4g}"),
        atol=atol,
        rtol=rtol,
    )
    problems = []
    min_cos, max_rel = GLOBAL_TOLERANCES.get(ref.dtype, GLOBAL_TOLERANCES[torch.float32])
    signal = a.numel() > 0 and ref_norm > atol * math.sqrt(a.numel())
    if (tier or TIER) == NEAR_LOSSLESS_TIER and signal:
        min_cos, max_rel = NEAR_LOSSLESS_MIN_COSINE, NEAR_LOSSLESS_MAX_REL_L2
        rms = ref_norm / math.sqrt(a.numel())
        e_atol, e_rtol = NEAR_LOSSLESS_ELEMENT
        element = float((diff / (e_atol * rms + e_rtol * a.abs())).max())
        norm_ratio = new_norm / ref_norm
        result.update(
            tier=NEAR_LOSSLESS_TIER,
            max_err_over_rms=round(float(diff.max()) / rms, 4),
            element_ratio=round(element, 4),
            norm_ratio=round(norm_ratio, 6),
        )
        if element > 1.0:
            problems.append(
                f"an element is {element:.3g}x its near-lossless tolerance away "
                f"({e_atol:g} x RMS + {e_rtol:g} x |reference|)"
            )
        if abs(norm_ratio - 1.0) > NEAR_LOSSLESS_MAX_NORM_CHANGE:
            problems.append(
                f"norm x{norm_ratio:.4f} (allowed ±{NEAR_LOSSLESS_MAX_NORM_CHANGE:.0%}): "
                "a systematic error, not rounding noise"
            )
        whole = True
    else:
        if mismatch > MAX_MISMATCH:
            problems.append(
                f"{mismatch:.4%} of elements outside tolerance (max {MAX_MISMATCH:.2%})"
            )
        if outlier > MAX_OUTLIER:
            problems.append(
                f"an element is {outlier:.3g}x its tolerance away (max {MAX_OUTLIER:g}x per "
                "element)"
            )
        whole = a.numel() > GLOBAL_MIN_NUMEL and signal
    if whole:
        if cos < min_cos:
            problems.append(f"cosine {cos:.6f} < {min_cos}")
        if rel_l2 > max_rel:
            problems.append(f"relative L2 error {rel_l2:.4g} > {max_rel}")
    if problems:
        result["error"] = "; ".join(problems)
    result["ok"] = not problems
    return result


def compare_structures(
    ref: Any, new: Any, prefix: str = "out", *, tier: str | None = None
) -> list[dict[str, Any]]:
    ref_flat = flatten(ref, prefix)
    new_flat = flatten(new, prefix)
    results = []
    for name, tensor in ref_flat.items():
        if name not in new_flat:
            results.append({"name": name, "ok": False, "error": "missing in candidate output"})
            continue
        results.append(compare_tensors(name, tensor, new_flat[name], tier=tier))
    return results


def compare_side_effects(
    pre: Any, ref_post: Any, new_post: Any, prefix: str = "args", *, tier: str | None = None
) -> list[dict[str, Any]]:
    """Compare the post-call state of a call's arguments (in-place side effects).

    For argument tensors that keep their shape and dtype, only the elements that
    the reference *or* the candidate changed are compared, so the mismatch
    allowance (:data:`MAX_MISMATCH`) is relative to the update.  Otherwise
    writing one position of an 8192-long KV cache (0.01 % of its elements), or
    forgetting to, would vanish inside the allowance.  Other tensors (e.g. caches
    that grow by concatenation) are compared whole."""
    return compare_side_effects_flat(
        flatten(pre, prefix), flatten(ref_post, prefix), flatten(new_post, prefix), tier=tier
    )


def compare_side_effects_flat(
    pre_flat: dict[str, torch.Tensor],
    ref_flat: dict[str, torch.Tensor],
    new_flat: dict[str, torch.Tensor],
    *,
    tier: str | None = None,
) -> list[dict[str, Any]]:
    """:func:`compare_side_effects` on already flattened ``{name: tensor}`` states."""
    results: list[dict[str, Any]] = []
    for name, ref in ref_flat.items():
        new = new_flat.get(name)
        if new is None:
            results.append({"name": name, "ok": False, "error": "missing in candidate arguments"})
            continue
        before = pre_flat.get(name)
        if (
            before is None
            or type_error(new) is not None
            or not (before.shape == ref.shape == new.shape)
            or not (before.dtype == ref.dtype == new.dtype)
        ):
            results.append(compare_tensors(name, ref, new, tier=tier))
            continue
        before, ref = before.to(new.device), ref.to(new.device)
        changed = (ref != before) | (new != before)
        count = int(changed.sum())
        if count == 0:
            results.append({"name": name, "ok": True, "changed_elements": 0, "max_abs_err": 0.0})
            continue
        result = compare_tensors(name, ref[changed], new[changed], tier=tier)
        result["changed_elements"] = count
        results.append(result)
    return results
