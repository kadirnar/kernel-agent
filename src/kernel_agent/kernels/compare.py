"""Structural comparison of nested module outputs (tensors, tuples, dicts, caches)."""

from __future__ import annotations

import dataclasses
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


def compare_tensors(
    name: str, ref: torch.Tensor, new: torch.Tensor, tol: tuple[float, float] | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {"name": name, "ok": False}
    if not isinstance(new, torch.Tensor):
        result["error"] = f"expected tensor, got {type(new).__name__}"
        return result
    if ref.shape != new.shape:
        result["error"] = f"shape {tuple(new.shape)} != reference {tuple(ref.shape)}"
        return result
    if ref.dtype != new.dtype:
        result["error"] = f"dtype {new.dtype} != reference {ref.dtype}"
        return result
    if not ref.is_floating_point():
        mismatches = (ref.to(new.device) != new).float().mean().item() if ref.numel() else 0.0
        result.update(ok=mismatches == 0, mismatch_frac=mismatches)
        return result
    atol, rtol = tol or TOLERANCES.get(ref.dtype, (1e-3, 1e-3))
    a = ref.detach().to(new.device).float()
    b = new.detach().float()
    if not torch.isfinite(b).all() and torch.isfinite(a).all():
        result["error"] = "candidate produced NaN/Inf"
        return result
    diff = (a - b).abs()
    bad = diff > (atol + rtol * a.abs())
    mismatch = bad.float().mean().item() if a.numel() else 0.0
    denom = a.norm() * b.norm()
    cos = float((a.flatten() @ b.flatten()) / denom) if denom > 0 else 1.0
    result.update(
        ok=mismatch <= MAX_MISMATCH,
        max_abs_err=float(diff.max()) if diff.numel() else 0.0,
        mean_abs_err=float(diff.mean()) if diff.numel() else 0.0,
        mismatch_frac=round(mismatch, 6),
        cosine=round(cos, 7),
        atol=atol,
        rtol=rtol,
    )
    return result


def compare_structures(ref: Any, new: Any, prefix: str = "out") -> list[dict[str, Any]]:
    ref_flat = flatten(ref, prefix)
    new_flat = flatten(new, prefix)
    results = []
    for name, tensor in ref_flat.items():
        if name not in new_flat:
            results.append({"name": name, "ok": False, "error": "missing in candidate output"})
            continue
        results.append(compare_tensors(name, tensor, new_flat[name]))
    return results


def compare_side_effects(
    pre: Any, ref_post: Any, new_post: Any, prefix: str = "args"
) -> list[dict[str, Any]]:
    """Compare the post-call state of a call's arguments (in-place side effects).

    For argument tensors that keep their shape and dtype, only the elements that
    the reference *or* the candidate changed are compared, so the mismatch
    allowance (:data:`MAX_MISMATCH`) is relative to the update.  Otherwise
    writing one position of an 8192-long KV cache (0.01 % of its elements), or
    forgetting to, would vanish inside the allowance.  Other tensors (e.g. caches
    that grow by concatenation) are compared whole."""
    pre_flat = flatten(pre, prefix)
    new_flat = flatten(new_post, prefix)
    results: list[dict[str, Any]] = []
    for name, ref in flatten(ref_post, prefix).items():
        new = new_flat.get(name)
        if new is None:
            results.append({"name": name, "ok": False, "error": "missing in candidate arguments"})
            continue
        before = pre_flat.get(name)
        if (
            before is None
            or not isinstance(new, torch.Tensor)
            or not (before.shape == ref.shape == new.shape)
            or not (before.dtype == ref.dtype == new.dtype)
        ):
            results.append(compare_tensors(name, ref, new))
            continue
        before, ref = before.to(new.device), ref.to(new.device)
        changed = (ref != before) | (new != before)
        count = int(changed.sum())
        if count == 0:
            results.append({"name": name, "ok": True, "changed_elements": 0, "max_abs_err": 0.0})
            continue
        result = compare_tensors(name, ref[changed], new[changed])
        result["changed_elements"] = count
        results.append(result)
    return results
