"""Live correctness checks beyond the captured cases (run by :mod:`kernels.evaluate`).

* **Aliasing parity** (:func:`alias_errors`): an output tensor shares memory with
  an input tensor in the candidate's call exactly when it does in the reference's
  (returning ``x`` itself where the reference allocates, or a copy where it
  returns a view, changes what in-place updates downstream do).
* **Re-verification** (:func:`reverify_case`), after timing, per case:

  1. ``fresh_addresses``: the captured inputs, deep-copied to new addresses,
     against the captured outputs and side effects;
  2. ``perturbed_same_addresses``: the floating-point tensors of those same input
     objects redrawn in place from a normal distribution with each channel's own
     mean and std, so an output cached by input address, shape or call count no
     longer matches;
  3. ``perturbed_mixed``: a fresh copy redrawn from a random mix of uniform,
     Laplace and (standardised) log-normal distributions;
  4. ``scaled_x3``, ``scaled_x0.01``, ``sign_flipped`` (:data:`SCALED`): fresh copies
     of the captured inputs with every floating-point tensor multiplied by 3, 0.01 and
     −1 (KernelBench-Verified caught a "374x" kernel that only worked on positive
     inputs this way; docs/RESEARCH-TRITON.md §3.3). Value-conditional shortcuts,
     constants calibrated on the captured values (an FP8 activation scale: ×3
     saturates it, the scales must follow the input), absolute epsilons and overflow
     handling show up here. Compared with ``input_scale`` (:func:`kernels.compare.
     compare_tensors`): the absolute tolerance grows with ×3, the signal threshold
     shrinks with ×0.01. A check whose reference output turns non-finite (overflow)
     where the captured one is finite is skipped and recorded (``skipped``).

  Perturbed draws are compared with the reference called live on copies of the
  same inputs (outputs, in-place side effects and aliasing), a reduced-precision
  tier with its bounds for redrawn inputs (:data:`kernels.compare.PERTURBED_BOUNDS`:
  a single token is redrawn without its outlier channels; the scaled checks too: a sign
  flip moves a massive activation to other channels).  The candidate
  always runs first and its output is copied right away, so it cannot return
  memory that the reference's call just freed.

What is redrawn: every floating-point tensor in the arguments (KV-cache contents
too, every slot), each channel (one position of the last dimension) from its own
mean and std over its non-zero elements (unused cache slots are zero), a tensor with
too few rows for that (a single token) from the tensor's (:func:`redraw_stats`, #198).
Integer and boolean tensors (ids, positions, masks), additive masks (any value <= -1e4),
tensors with non-finite values and rotary tables (cos / sin of the positions,
:func:`rotary_tables`) are left alone.
"""

from __future__ import annotations

import copy
import itertools
import math
from collections.abc import Callable
from typing import Any

import torch

from kernel_agent.kernels.compare import (
    CHANNEL_MIN_ROWS,
    compare_side_effects,
    compare_structures,
    flatten,
)

MASKED = -1e4  # additive masks use values at or below this: never redrawn
#: Rotary tables (:func:`rotary_tables`): ``cos² + sin²`` within this share of its mean.
ROTARY_TOL = 0.02
MIX = ("uniform", "laplace", "lognormal")
#: (check, factor) of the scaled checks: the captured floating-point inputs times the factor.
SCALED = (("scaled_x3", 3.0), ("scaled_x0.01", 0.01), ("sign_flipped", -1.0))


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


def rotary_tables(tensors: list[torch.Tensor]) -> set[int]:
    """``id`` of the rotary tables among ``tensors`` (in flattening order): two consecutive
    floating-point tensors of one shape whose squares sum to the same positive value at
    every element (within :data:`ROTARY_TOL`): the cos and sin of the positions, scaled
    or not (``(cos, sin)`` of HF's ``position_embeddings``, VoxCPM's ``position_emb``).
    They are a function of the positions, which stay as captured (integer), so they stay
    too: drawn independently they are no rotation, and a model whose keys have huge
    low-frequency channels (Qwen3: ``sin`` ≈ 0 there at every position) gets them leaked
    into the other channels of the query and key (#198)."""
    floats = [
        t
        for t in tensors
        if t.is_floating_point() and t.numel() and t.dim() and t.layout == torch.strided
    ]
    found: set[int] = set()
    for a, b in itertools.pairwise(floats):
        if a.shape != b.shape or a is b:
            continue
        x, y = a.detach(), b.detach()
        row = (0,) * (x.dim() - 1)  # one row first: other pairs fail it at once
        if _constant_norm(x[row], y[row]) and _constant_norm(x, y):
            found |= {id(a), id(b)}
    return found


def _constant_norm(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether ``a² + b²`` is the same positive value at every element (:data:`ROTARY_TOL`)."""
    r = a.float().pow(2) + b.to(a.device).float().pow(2)
    level = float(r.mean())
    return level > 0 and float((r - level).abs().max()) <= ROTARY_TOL * level


def _global_stats(values: torch.Tensor) -> tuple[float, float]:
    """Mean and std of the non-zero elements of ``values`` (0 and 1 when there are none;
    a constant tensor gets a tenth of its magnitude as std)."""
    nonzero = values[values != 0]
    if nonzero.numel():
        mean, std = float(nonzero.mean()), float(nonzero.std(unbiased=False))
    else:
        mean, std = 0.0, 1.0
    return mean, std or abs(mean) * 0.1 or 1.0


def redraw_stats(values: torch.Tensor) -> tuple[torch.Tensor | float, torch.Tensor | float]:
    """The mean and std to redraw ``values`` (float, finite) from: per channel where it
    has the rows for it, the tensor's own otherwise.

    A channel is one position of the last dimension, its statistics taken over all the
    other dimensions (as :func:`kernels.compare._channel_rms` defines it) and over its
    non-zero elements only (unused cache slots are zero); a channel with fewer than
    :data:`CHANNEL_MIN_ROWS` of them gets the tensor's statistics (every channel of a
    single token or a short cache). So outlier channels keep their scale: Qwen3-0.6B's
    K cache after ``k_norm`` (layer 0: channel RMS 225 and 69, the median 1.6), drawn from
    the tensor's std (21), had every channel 13 times its size, attention logits far wider
    than real ones, and the reference math of every reduced precision failed most draws
    (#198)."""
    mean, std = _global_stats(values)
    if values.dim() < 2 or values.numel() < CHANNEL_MIN_ROWS * values.shape[-1]:
        return mean, std
    dims = tuple(range(values.dim() - 1))
    used = values != 0
    count = used.sum(dims, keepdim=True)
    n = count.clamp_min(1)
    ch_mean = torch.where(used, values, 0.0).sum(dims, keepdim=True) / n
    ch_var = torch.where(used, values - ch_mean, 0.0).pow(2).sum(dims, keepdim=True) / n
    ch_std = ch_var.sqrt()
    ch_std = torch.where(ch_std > 0, ch_std, ch_mean.abs() * 0.1)
    few = count < CHANNEL_MIN_ROWS
    ch_mean = torch.where(few, torch.full_like(ch_mean, mean), ch_mean)
    ch_std = torch.where(few, torch.full_like(ch_std, std), ch_std)
    return ch_mean, ch_std


def _draw(
    t: torch.Tensor,
    kind: str,
    mean: torch.Tensor | float,
    std: torch.Tensor | float,
    gen: torch.Generator,
) -> torch.Tensor:
    """Samples with the given mean and std (``kind`` sets the shape of the distribution;
    tensors broadcast against ``t``), drawn on the generator's device (an argument may
    live on another one)."""
    shape, device = t.shape, gen.device
    if isinstance(mean, torch.Tensor):
        mean = mean.to(device)
    if isinstance(std, torch.Tensor):
        std = std.to(device)
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
    strides and storage) from their channels' statistics (:func:`redraw_stats`); rotary
    tables stay (:func:`rotary_tables`).  ``kind`` is ``normal`` or ``mix`` (one of
    :data:`MIX` per tensor).  Returns the number of tensors redrawn."""
    done = 0
    tensors = [t for t in flatten(value).values() if isinstance(t, torch.Tensor)]
    rotary = rotary_tables(tensors)
    with torch.inference_mode():
        for t in tensors:
            if id(t) in rotary or not _perturbable(t):
                continue
            mean, std = redraw_stats(t.detach().float())
            shape = kind
            if kind == "mix":
                pick = int(torch.randint(len(MIX), (1,), generator=gen, device=gen.device))
                shape = MIX[pick]
            t.copy_(_draw(t, shape, mean, std, gen).to(t.device, t.dtype))
            done += 1
    return done


def scale_(value: Any, factor: float) -> int:
    """Multiply the floating-point tensors inside ``value`` by ``factor`` in place (the
    tensors :func:`perturb_` would redraw, and rotary tables too: the scaled checks scale
    the whole call's inputs, as calibrated in #175; with them kept the x 0.01 inputs of the
    VoxCPM2 LocDiT layer move honest MXFP8's output norm by 6.8 %. No additive masks, no
    non-finite tensors). Returns the number of tensors scaled."""
    done = 0
    with torch.inference_mode():
        for t in flatten(value).values():
            if not isinstance(t, torch.Tensor) or not _perturbable(t):
                continue
            t.mul_(factor)
            done += 1
    return done


# ------------------------------------------------------------------ re-verification

CHECKS = {
    "fresh_addresses": "captured inputs at fresh addresses",
    "perturbed_same_addresses": "inputs redrawn in place (same addresses as the previous call)",
    "perturbed_mixed": "inputs redrawn from uniform/Laplace/log-normal at fresh addresses",
    "scaled_x3": "the captured floating-point inputs x 3",
    "scaled_x0.01": "the captured floating-point inputs x 0.01",
    "sign_flipped": "the captured floating-point inputs x -1 (signs flipped)",
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


def _finite(value: Any) -> bool:
    """Whether every floating-point tensor inside ``value`` is finite."""
    return all(
        bool(torch.isfinite(t).all())
        for t in flatten(value).values()
        if isinstance(t, torch.Tensor) and t.is_floating_point() and t.numel()
    )


def _against_reference(
    ref_fn: Callable[..., Any],
    new_fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    sync: Callable[[], None],
    *,
    input_scale: float = 1.0,
    finite: bool = False,
) -> list[dict[str, Any]] | None:
    """Call the candidate on ``args``/``kwargs`` (mutated in place, like the model
    would), then the reference on copies of the same pre-call inputs; compare (with
    ``input_scale``, :func:`kernels.compare.compare_tensors`). ``finite``: None (skip)
    when the reference's output or post-call state is not finite."""
    pre_args, pre_kwargs = copy.deepcopy(args), copy.deepcopy(kwargs)
    out = _call(new_fn, args, kwargs, sync)
    ref_args, ref_kwargs = copy.deepcopy(pre_args), copy.deepcopy(pre_kwargs)
    expected = _call(ref_fn, ref_args, ref_kwargs, sync)
    if finite and not _finite((expected, ref_args, ref_kwargs)):
        return None
    kw: dict[str, Any] = {"perturbed": True, "input_scale": input_scale}
    checks = compare_structures(expected, out, "output", inputs=(pre_args, pre_kwargs), **kw)
    checks += compare_side_effects(pre_args, ref_args, args, "args", **kw)
    checks += compare_side_effects(pre_kwargs, ref_kwargs, kwargs, "kwargs", **kw)
    return checks


def reverify_case(
    ref_fn: Callable[..., Any],
    new_fn: Callable[..., Any],
    case: dict[str, Any],
    pristine: tuple[tuple[Any, ...], dict[str, Any]],
    gen: torch.Generator,
    sync: Callable[[], None],
    record: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run the re-verification checks of one case (:data:`CHECKS`); returns the failed
    ones as ``{"check": name, "what": description, "failures": [...]}``. ``record``
    collects the scaled checks that ran (``ran``: check -> count) and were skipped
    (``skipped``: ``{"check", "why"}``)."""
    failed = []
    record = record if record is not None else {}
    ran, skipped = record.setdefault("ran", {}), record.setdefault("skipped", [])
    pre_args, pre_kwargs = pristine
    args, kwargs = copy.deepcopy(pre_args), copy.deepcopy(pre_kwargs)
    out = _call(new_fn, args, kwargs, sync)
    checks = compare_structures(case["output"], out, "output", inputs=pristine)
    checks += compare_side_effects(pre_args, case["post_args"], args, "args")
    checks += compare_side_effects(pre_kwargs, case["post_kwargs"], kwargs, "kwargs")
    runs: list[tuple[str, list[dict[str, Any]] | None]] = [("fresh_addresses", checks)]

    perturb_((args, kwargs), gen, "normal")  # the objects the previous call just saw
    runs.append(
        ("perturbed_same_addresses", _against_reference(ref_fn, new_fn, args, kwargs, sync))
    )

    args, kwargs = copy.deepcopy(pre_args), copy.deepcopy(pre_kwargs)
    perturb_((args, kwargs), gen, "mix")
    runs.append(("perturbed_mixed", _against_reference(ref_fn, new_fn, args, kwargs, sync)))

    captured_finite = _finite(case["output"])  # else non-finite positions are compared
    for name, factor in SCALED:
        args, kwargs = copy.deepcopy(pre_args), copy.deepcopy(pre_kwargs)
        if not scale_((args, kwargs), factor):
            continue  # no floating-point inputs: nothing to scale
        scaled = _against_reference(
            ref_fn, new_fn, args, kwargs, sync, input_scale=factor, finite=captured_finite
        )
        if scaled is None:
            skipped.append({"check": name, "why": "the reference's output is not finite"})
            continue
        ran[name] = ran.get(name, 0) + 1
        runs.append((name, scaled))
    for name, found in runs:
        bad = [c for c in found or [] if not c.get("ok")]
        if bad:
            failed.append({"check": name, "what": CHECKS[name], "failures": bad[:5]})
    return failed
