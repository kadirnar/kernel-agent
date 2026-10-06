"""Live correctness checks beyond the captured cases (run by :mod:`kernels.evaluate`).

* **Aliasing parity** (:func:`alias_errors`): an output tensor shares memory with
  an input tensor in the candidate's call exactly when it does in the reference's
  (returning ``x`` itself where the reference allocates, or a copy where it
  returns a view, changes what in-place updates downstream do).
* **Re-verification** (:func:`reverify_case`), after timing, per case:

  1. ``fresh_addresses``: the captured inputs, deep-copied to new addresses,
     against the captured outputs and side effects;
  2. ``perturbed_same_addresses``: the floating-point tensors of those same input
     objects redrawn in place from a normal distribution with each tensor's own
     mean and std, so an output cached by input address, shape or call count no
     longer matches;
  3. ``perturbed_mixed``: a fresh copy redrawn from a random mix of uniform,
     Laplace and (standardised) log-normal distributions.

  Perturbed draws are compared with the reference called live on copies of the
  same inputs (outputs, in-place side effects and aliasing), a reduced-precision
  tier with its bounds for redrawn inputs (:data:`kernels.compare.PERTURBED_BOUNDS`:
  redrawn inputs have no outlier channels).  The candidate
  always runs first and its output is copied right away, so it cannot return
  memory that the reference's call just freed.

What is redrawn: every floating-point tensor in the arguments (KV-cache contents
too), with mean and std taken over its non-zero finite elements (unused cache
slots are zero).  Integer and boolean tensors (ids, positions, masks), additive
masks (any value <= -1e4) and tensors with non-finite values are left alone.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from typing import Any

import torch

from kernel_agent.kernels.compare import compare_side_effects, compare_structures, flatten

MASKED = -1e4  # additive masks use values at or below this: never redrawn
MIX = ("uniform", "laplace", "lognormal")


# ------------------------------------------------------------------ aliasing


def _byte_range(t: torch.Tensor) -> tuple[str, int, int] | None:
    """(device, first byte, end byte) of the memory a tensor's elements span."""
    if t.numel() == 0 or t.layout != torch.strided:
        return None
    try:
        base = t.untyped_storage().data_ptr()
    except (RuntimeError, NotImplementedError, TypeError):
        return None
    if not base:
        return None
    item = t.element_size()
    start = base + int(t.storage_offset()) * item
    span = 1 + sum((int(n) - 1) * abs(int(s)) for n, s in zip(t.shape, t.stride(), strict=True))
    return str(t.device), start, start + span * item


def alias_map(output: Any, args: Any, kwargs: Any) -> dict[str, list[str]]:
    """Output tensor name -> names of the argument tensors whose memory it overlaps."""
    spans = {
        name: span
        for name, t in {**flatten(args, "args"), **flatten(kwargs, "kwargs")}.items()
        if isinstance(t, torch.Tensor) and (span := _byte_range(t)) is not None
    }
    found: dict[str, list[str]] = {}
    for name, t in flatten(output, "output").items():
        if not isinstance(t, torch.Tensor):
            continue
        mine = _byte_range(t)
        found[name] = sorted(
            other
            for other, (device, start, end) in spans.items()
            if mine is not None and device == mine[0] and start < mine[2] and mine[1] < end
        )
    return found


def alias_errors(
    ref_out: Any, ref_inputs: tuple[Any, Any], new_out: Any, new_inputs: tuple[Any, Any]
) -> list[dict[str, Any]]:
    """Failed checks for output tensors whose aliasing of the call's arguments differs
    from the reference's.  ``*_inputs`` are ``(args, kwargs)`` of each call."""
    ref_map, new_map = alias_map(ref_out, *ref_inputs), alias_map(new_out, *new_inputs)
    errors = []
    for name, ref_alias in ref_map.items():
        new_alias = new_map.get(name)
        if new_alias is None or new_alias == ref_alias:
            continue
        if new_alias and not ref_alias:
            why = f"shares memory with {', '.join(new_alias)}; the reference returns a new tensor"
        elif ref_alias and not new_alias:
            why = f"is a new tensor; the reference returns (a view of) {', '.join(ref_alias)}"
        else:
            why = f"shares memory with {', '.join(new_alias)}; the reference with " + ", ".join(
                ref_alias
            )
        errors.append({"name": name, "ok": False, "error": f"aliasing: the output {why}"})
    return errors


# ------------------------------------------------------------------ perturbation


def _perturbable(t: torch.Tensor) -> bool:
    if not t.is_floating_point() or t.numel() == 0 or t.layout != torch.strided:
        return False
    values = t.detach().float()
    return bool(torch.isfinite(values).all()) and float(values.min()) > MASKED


def _draw(
    t: torch.Tensor, kind: str, mean: float, std: float, gen: torch.Generator
) -> torch.Tensor:
    """Samples with the given mean and std (``kind`` sets the shape of the distribution),
    drawn on the generator's device (an argument may live on another one)."""
    shape, device = t.shape, gen.device
    if kind == "normal":
        z = torch.randn(shape, generator=gen, device=device)
    elif kind == "uniform":
        z = (torch.rand(shape, generator=gen, device=device) * 2 - 1) * math.sqrt(3)
    elif kind == "laplace":
        u = torch.rand(shape, generator=gen, device=device).clamp(1e-7, 1 - 1e-7) - 0.5
        z = -torch.sign(u) * torch.log1p(-2 * u.abs()) / math.sqrt(2)
    elif kind == "lognormal":  # sigma 0.5, standardised: skewed, heavy right tail
        sigma = 0.5
        m = math.exp(sigma**2 / 2)
        sd = math.sqrt((math.exp(sigma**2) - 1) * math.exp(sigma**2))
        z = (torch.exp(sigma * torch.randn(shape, generator=gen, device=device)) - m) / sd
    else:
        raise ValueError(kind)
    return mean + std * z


def perturb_(value: Any, gen: torch.Generator, kind: str) -> int:
    """Redraw the floating-point tensors inside ``value`` in place (same shape, dtype,
    strides and storage).  ``kind`` is ``normal`` or ``mix`` (one of :data:`MIX`
    per tensor).  Returns the number of tensors redrawn."""
    done = 0
    with torch.inference_mode():
        for t in flatten(value).values():
            if not isinstance(t, torch.Tensor) or not _perturbable(t):
                continue
            values = t.detach().float()
            nonzero = values[values != 0]
            if nonzero.numel():
                mean, std = float(nonzero.mean()), float(nonzero.std(unbiased=False))
            else:
                mean, std = 0.0, 1.0
            if std == 0:
                std = abs(mean) * 0.1 or 1.0
            shape = kind
            if kind == "mix":
                pick = int(torch.randint(len(MIX), (1,), generator=gen, device=gen.device))
                shape = MIX[pick]
            t.copy_(_draw(t, shape, mean, std, gen).to(t.device, t.dtype))
            done += 1
    return done


# ------------------------------------------------------------------ re-verification

CHECKS = {
    "fresh_addresses": "captured inputs at fresh addresses",
    "perturbed_same_addresses": "inputs redrawn in place (same addresses as the previous call)",
    "perturbed_mixed": "inputs redrawn from uniform/Laplace/log-normal at fresh addresses",
}


def _snapshot(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _call(fn: Callable[..., Any], args: Any, kwargs: Any, sync: Callable[[], None]) -> Any:
    with torch.inference_mode():
        out = fn(*args, **kwargs)
    sync()
    return _snapshot(out)


def _against_reference(
    ref_fn: Callable[..., Any],
    new_fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    sync: Callable[[], None],
) -> list[dict[str, Any]]:
    """Call the candidate on ``args``/``kwargs`` (mutated in place, like the model
    would), then the reference on copies of the same pre-call inputs; compare."""
    pre_args, pre_kwargs = copy.deepcopy(args), copy.deepcopy(kwargs)
    out = _call(new_fn, args, kwargs, sync)
    ref_args, ref_kwargs = copy.deepcopy(pre_args), copy.deepcopy(pre_kwargs)
    expected = _call(ref_fn, ref_args, ref_kwargs, sync)
    checks = compare_structures(expected, out, "output", perturbed=True)
    checks += compare_side_effects(pre_args, ref_args, args, "args", perturbed=True)
    checks += compare_side_effects(pre_kwargs, ref_kwargs, kwargs, "kwargs", perturbed=True)
    return checks


def reverify_case(
    ref_fn: Callable[..., Any],
    new_fn: Callable[..., Any],
    case: dict[str, Any],
    pristine: tuple[tuple[Any, ...], dict[str, Any]],
    gen: torch.Generator,
    sync: Callable[[], None],
) -> list[dict[str, Any]]:
    """Run the three re-verification checks of one case; returns the failed ones as
    ``{"check": name, "what": description, "failures": [...]}``."""
    failed = []
    pre_args, pre_kwargs = pristine
    args, kwargs = copy.deepcopy(pre_args), copy.deepcopy(pre_kwargs)
    out = _call(new_fn, args, kwargs, sync)
    checks = compare_structures(case["output"], out, "output")
    checks += compare_side_effects(pre_args, case["post_args"], args, "args")
    checks += compare_side_effects(pre_kwargs, case["post_kwargs"], kwargs, "kwargs")
    runs = [("fresh_addresses", checks)]

    perturb_((args, kwargs), gen, "normal")  # the objects the previous call just saw
    runs.append(
        ("perturbed_same_addresses", _against_reference(ref_fn, new_fn, args, kwargs, sync))
    )

    args, kwargs = copy.deepcopy(pre_args), copy.deepcopy(pre_kwargs)
    perturb_((args, kwargs), gen, "mix")
    runs.append(("perturbed_mixed", _against_reference(ref_fn, new_fn, args, kwargs, sync)))
    for name, checks in runs:
        bad = [c for c in checks if not c.get("ok")]
        if bad:
            failed.append({"check": name, "what": CHECKS[name], "failures": bad[:5]})
    return failed
