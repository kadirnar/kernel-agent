"""Workload contract: how to load, run and compare one model end to end.

A workload is the ground truth for an optimisation run.  It must be
deterministic (fixed seeds, greedy decoding) so that the output of the
optimised model can be compared against the baseline output.

Autoregressive models with continuous, sampled outputs (diffusion/flow-matching
TTS heads fed back into an LM) are *chaotic*: any numerically-correct kernel
change makes the free-running trajectory diverge.  Such workloads implement
teacher forcing: the candidate replays the baseline trajectory and only its
per-step predictions are compared, so errors cannot compound.
"""

from __future__ import annotations

import contextlib
import statistics
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

import torch
from torch import nn

from kernel_agent.hub import Modality

DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}
#: :func:`compare_stop`: a baseline stop logit margin this close to zero is a near-tie.
STOP_NEAR_TIE = 0.5


@dataclass
class WorkloadSpec:
    repo_id: str
    modality: str
    revision: str | None = None
    dtype: str = "bfloat16"
    device: str = "cuda"
    trust_remote_code: bool = False
    harness: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    #: Model family with a dedicated built-in workload (``hub.detect_family``).
    family: str | None = None

    @property
    def torch_dtype(self) -> torch.dtype:
        return DTYPES[self.dtype]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkloadSpec:
        return cls(**data)


@dataclass
class Comparison:
    passed: bool
    metrics: dict[str, float | int | str] = field(default_factory=dict)
    reason: str = ""


class Workload(ABC):
    """Base class for built-in and agent-written workloads."""

    modality: ClassVar[Modality] = Modality.UNKNOWN
    #: Default knobs; overridden by ``spec.options``.
    defaults: ClassVar[dict[str, Any]] = {}
    #: Extra module entrypoints besides ``forward`` that the run calls directly
    #: (class name -> method names, e.g. ``{"Attention": ["step_decode"]}``) and
    #: the name pattern in :mod:`kernel_agent.profiling.methods` misses.  The
    #: ``entrypoints=Cls.method,...`` option adds more.
    entrypoints: ClassVar[dict[str, list[str]]] = {}
    #: The free-running output diverges under any numerically-correct change, so
    #: :meth:`compare` on it is informational only (quality = teacher forcing).
    chaotic: ClassVar[bool] = False
    #: :meth:`run_teacher_forced` and :meth:`compare_teacher_forced` are implemented.
    supports_teacher_forcing: ClassVar[bool] = False
    #: Where teacher forcing hooks in (shown to the agents in the profile summary).
    teacher_forcing_note: ClassVar[str] = ""
    #: Per-output RMS tolerance of the free-running sanity check (default ±25 %),
    #: keyed like the output dict, e.g. ``{"audio": 0.6}``.
    sanity_rms_tolerance: ClassVar[dict[str, float]] = {}

    def __init__(self, spec: WorkloadSpec) -> None:
        self.spec = spec
        self.options = {**self.defaults, **spec.options}

    @property
    def device(self) -> torch.device:
        return torch.device(self.spec.device)

    @property
    def dtype(self) -> torch.dtype:
        return self.spec.torch_dtype

    @abstractmethod
    def load(self) -> None:
        """Download (if needed) and load the model onto the device."""

    @abstractmethod
    def roots(self) -> dict[str, nn.Module]:
        """Top-level ``nn.Module``s that are profiled and patched (name -> module)."""

    @abstractmethod
    def make_inputs(self) -> Any:
        """Deterministic inputs for :meth:`run`."""

    @abstractmethod
    def run(self, inputs: Any) -> Any:
        """One end-to-end inference.  Return CPU tensors / plain data for :meth:`compare`."""

    @abstractmethod
    def compare(self, reference: Any, candidate: Any) -> Comparison:
        """Decide whether ``candidate`` output is acceptable versus the baseline."""

    def run_teacher_forced(self, inputs: Any, reference: Any) -> Any:
        """Replay the trajectory recorded in ``reference`` (the baseline output of
        :meth:`run`): at every step feed the *reference* state back into the loop
        and record the model's own prediction for that step.

        Sampling noise must be drawn exactly as in :meth:`run` (same seed, same
        RNG calls in the same order) so that step *i* sees the same noise."""
        raise NotImplementedError(f"{type(self).__name__} does not support teacher forcing")

    def compare_teacher_forced(self, reference: Any, candidate: Any) -> Comparison:
        """Compare the per-step predictions of :meth:`run_teacher_forced` with the
        steps recorded in ``reference``."""
        raise NotImplementedError(f"{type(self).__name__} does not support teacher forcing")

    def reference_optimizations(self) -> str | None:
        """Optional hook: apply, in place and after :meth:`load`, what a competent
        user would do to speed this model up without custom kernels (the model's
        own ``torch.compile`` path, a static KV cache, a compiled denoiser, ...).
        Returns a one-line description, or ``None`` when there is nothing to apply.

        ``analyze`` measures it in a fresh process as the *compiled baseline*
        (``baseline.json`` ``compiled_ms``) and the reports show speedups vs eager
        and vs compiled; the integration also measures it on top of the accepted
        kernels.  The default does nothing (no compiled baseline unless
        ``--compile-baseline`` asks for a generic ``torch.compile`` of the roots)."""
        return None

    def holdout_options(self, variant: int = 1) -> dict[str, Any] | None:
        """Option overrides for held-out input number ``variant`` (``None``: none).

        Variant 1 is the held-out e2e input: ``analyze`` stores its baseline output
        and ``e2e`` judges every candidate on it as well, untimed
        (:mod:`kernel_agent.workloads.holdout`). It differs in content from the main
        input (another prompt, text, audio or seed) and keeps its shapes where it
        can. Variants ``>= 2`` are the memoisation probe's fresh inputs: the shapes of
        variant 1, content that differs from variant 1 (another seed, say).

        Overrides apply to ``self.options`` around :meth:`make_inputs`, :meth:`run`
        and teacher forcing (:meth:`with_options`), so seeds read in ``run`` count.
        The default changes ``seed`` when the workload has one."""
        if "seed" not in self.options:
            return None
        return {"seed": int(self.options["seed"]) + variant}

    def variants(self) -> list[dict[str, Any]]:
        """Option overrides of extra workload settings (other prompt lengths, batch
        sizes, texts) whose calls ``capture`` records as *correctness-only* cases:
        checked by the evaluator, not timed, not weighted. Keep them short (few
        decode steps); only calls with shapes the main run lacks are kept."""
        return []

    def natural_length_run(self, reference: Any = None) -> dict[str, Any] | None:
        """Optional hook for autoregressive models that decide their own output length
        (a stop head, an end-of-sequence token): one untimed run in which that decision
        is live. ``None`` (the default): the workload has no stop condition.

        A fixed-length workload (VoxCPM forces ``min_len = max_len``) never lets the stop
        condition decide, so a transform that skipped or delayed it would pass every other
        check (:mod:`kernel_agent.workloads.stopping`). ``analyze`` calls this free running
        (``reference=None``) and stores the result in ``.truth/``; ``e2e`` calls it with
        ``reference`` set to that result and replays it teacher forced where the workload
        supports it, so every stop decision sees the baseline history.

        Returns a dict with ``steps`` (generated steps), ``min_steps`` / ``max_steps``
        (the limits of the run) and, optionally, ``stop_margins`` (baseline: stop minus
        continue logit of every step), ``output_length`` (e.g. audio samples) and whatever
        the teacher-forced replay needs (CPU tensors). Keep it short (tens of steps)."""
        return None

    def compare_natural_length(self, reference: Any, candidate: Any) -> Comparison:
        """Results of :meth:`natural_length_run`: the candidate must stop at the same step
        (:func:`compare_stop`; options ``stop_tolerance`` and ``stop_near_tie``)."""
        return compare_stop(
            reference,
            candidate,
            tolerance=int(self.options.get("stop_tolerance", 0)),
            near_tie=float(self.options.get("stop_near_tie", STOP_NEAR_TIE)),
        )

    @contextlib.contextmanager
    def with_options(self, overrides: dict[str, Any] | None) -> Iterator[None]:
        """Apply option ``overrides`` (in place, so references to ``self.options``
        see them) and restore the previous options afterwards."""
        saved = dict(self.options)
        self.options.update(overrides or {})
        try:
            yield
        finally:
            self.options.clear()
            self.options.update(saved)

    def describe(self) -> str:
        opts = ", ".join(f"{k}={v}" for k, v in sorted(self.options.items()))
        return f"{type(self).__name__}({self.spec.repo_id}, {self.spec.dtype}; {opts})"


# ---------------------------------------------------------------- helpers


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure(workload: Workload, inputs: Any, *, warmup: int = 1, iters: int = 3) -> dict[str, Any]:
    """Wall-clock latency of ``workload.run`` (GPU-synchronised), in milliseconds."""
    output = None
    if torch.cuda.is_available():
        # Same clock warm-up for baseline and optimised runs (fair comparison).
        from kernel_agent.kernels.bench import warm_gpu

        warm_gpu(500.0)
    with torch.inference_mode():
        for _ in range(warmup):
            output = workload.run(inputs)
        synchronize()
        times = []
        for _ in range(iters):
            start = time.perf_counter()
            output = workload.run(inputs)
            synchronize()
            times.append((time.perf_counter() - start) * 1000)
    return {
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "times_ms": times,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3
        if torch.cuda.is_available()
        else 0.0,
        "output": output,
    }


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    n = min(a.numel(), b.numel())
    if n == 0:
        return 1.0
    a, b = a[:n], b[:n]
    finite = torch.isfinite(a) & torch.isfinite(b)
    if not bool(finite.all()):
        # Masked logits (-inf) must match exactly; compare the rest numerically.
        if not torch.equal(torch.isfinite(a), torch.isfinite(b)):
            return 0.0
        a, b = a[finite], b[finite]
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if torch.equal(a, b) else 0.0
    return float((a @ b) / denom)


def psnr(a: torch.Tensor, b: torch.Tensor, data_range: float | None = None) -> float:
    a = a.detach().float()
    b = b.detach().float()
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0:
        return float("inf")
    rng = data_range if data_range is not None else float(a.max() - a.min()) or 1.0
    return 10.0 * torch.log10(torch.tensor(rng**2 / mse)).item()


def compare_tokens(
    ref_tokens: torch.Tensor,
    new_tokens: torch.Tensor,
    ref_logits: torch.Tensor | None,
    new_logits: torch.Tensor | None,
    *,
    min_prefix: int,
    min_cosine: float,
) -> Comparison:
    """Greedy-decoding comparison: first-step logits must agree and the first
    ``min_prefix`` generated tokens must be identical.  Later divergence is
    tolerated because low-precision kernels legitimately flip near-ties."""
    ref = ref_tokens.flatten().tolist()
    new = new_tokens.flatten().tolist()
    prefix = 0
    for x, y in zip(ref, new, strict=False):
        if x != y:
            break
        prefix += 1
    total = max(len(ref), 1)
    metrics: dict[str, float | int | str] = {
        "token_prefix_match": prefix,
        "token_match_ratio": round(prefix / total, 4),
        "generated_tokens": len(new),
    }
    passed = prefix >= min(min_prefix, len(ref))
    reason = "" if passed else f"tokens diverge at position {prefix}"
    if ref_logits is not None and new_logits is not None:
        cos = cosine(ref_logits, new_logits)
        metrics["first_logits_cosine"] = round(cos, 6)
        same_top1 = bool(torch.equal(ref_logits.float().argmax(-1), new_logits.float().argmax(-1)))
        metrics["first_top1_match"] = int(same_top1)
        if cos < min_cosine:
            passed = False
            reason = f"first-step logits cosine {cos:.5f} < {min_cosine}"
    return Comparison(passed, metrics, reason)


def compare_audio(
    ref: torch.Tensor, new: torch.Tensor, *, min_spec_cosine: float, max_len_ratio: float = 0.1
) -> Comparison:
    """Spectral comparison (robust to tiny phase/sample differences)."""
    ref = ref.detach().float().flatten()
    new = new.detach().float().flatten()
    len_diff = abs(ref.numel() - new.numel()) / max(ref.numel(), 1)
    n = min(ref.numel(), new.numel())
    if n < 1024:
        return Comparison(False, {"samples": n}, "audio too short to compare")

    def spec(x: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(1024)
        mag = torch.stft(x[:n], 1024, 256, window=window, return_complex=True).abs()
        return torch.log1p(mag)

    cos = cosine(spec(ref), spec(new))
    metrics: dict[str, float | int | str] = {
        "spectral_cosine": round(cos, 5),
        "length_diff_ratio": round(len_diff, 4),
        "waveform_cosine": round(cosine(ref[:n], new[:n]), 5),
    }
    if len_diff > max_len_ratio:
        return Comparison(False, metrics, f"length differs by {len_diff:.1%}")
    if cos < min_spec_cosine:
        return Comparison(False, metrics, f"spectral cosine {cos:.4f} < {min_spec_cosine}")
    return Comparison(True, metrics)


def compare_steps(
    ref_steps: torch.Tensor,
    new_steps: torch.Tensor,
    *,
    min_step_cosine: float,
    min_mean_step_cosine: float,
    max_rms_change: float = 0.1,
) -> Comparison:
    """Teacher-forced comparison of per-step predictions (dim 0 = step).

    ``new_steps[i]`` was predicted from the *reference* history, so a kernel's
    error shows up once per step instead of compounding along the trajectory.
    The mean cosine catches small systematic errors, the minimum a single bad
    step, and the RMS ratio magnitude errors that cosines cannot see."""
    if ref_steps.shape[0] == 0:
        return Comparison(False, {"steps": 0}, "the reference recorded no steps")
    if ref_steps.shape[0] != new_steps.shape[0]:
        return Comparison(
            False,
            {"steps": int(new_steps.shape[0]), "reference_steps": int(ref_steps.shape[0])},
            f"{new_steps.shape[0]} teacher-forced steps != {ref_steps.shape[0]} reference steps",
        )
    if ref_steps.shape[1:] != new_steps.shape[1:]:
        return Comparison(
            False,
            {"steps": int(new_steps.shape[0])},
            f"step shape {tuple(new_steps.shape[1:])} != {tuple(ref_steps.shape[1:])}",
        )
    ref = ref_steps.detach().float().flatten(1)
    new = new_steps.detach().float().flatten(1)
    cosines = [cosine(r, n) for r, n in zip(ref, new, strict=True)]
    rel_err = (new - ref).norm(dim=1) / ref.norm(dim=1).clamp_min(1e-12)
    worst = min(range(len(cosines)), key=cosines.__getitem__)
    mean_cos = sum(cosines) / len(cosines)
    ref_rms = float(ref.pow(2).mean().sqrt())
    rms_ratio = float(new.pow(2).mean().sqrt()) / ref_rms if ref_rms > 0 else 1.0
    metrics: dict[str, float | int | str] = {
        "steps": len(cosines),
        "min_step_cosine": round(cosines[worst], 6),
        "mean_step_cosine": round(mean_cos, 6),
        "worst_step": worst,
        "mean_rel_error": round(float(rel_err.mean()), 6),
        "max_rel_error": round(float(rel_err.max()), 6),
        "rms_ratio": round(rms_ratio, 6),
    }
    if not bool(torch.isfinite(new).all()):
        return Comparison(False, metrics, "non-finite teacher-forced predictions")
    if mean_cos < min_mean_step_cosine:
        return Comparison(
            False, metrics, f"mean step cosine {mean_cos:.5f} < {min_mean_step_cosine}"
        )
    if cosines[worst] < min_step_cosine:
        return Comparison(
            False, metrics, f"step {worst} cosine {cosines[worst]:.5f} < {min_step_cosine}"
        )
    if abs(rms_ratio - 1.0) > max_rms_change:
        return Comparison(False, metrics, f"RMS x{rms_ratio:.4f} (allowed ±{max_rms_change:.0%})")
    return Comparison(True, metrics)


def compare_stop(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    tolerance: int = 0,
    near_tie: float = STOP_NEAR_TIE,
) -> Comparison:
    """Natural-length comparison (:meth:`Workload.natural_length_run`): the candidate
    must stop after as many steps as the baseline.

    Replayed teacher forced, every stop decision is an argmax over a hidden state on
    the baseline trajectory, so the check is exact by default. ``tolerance`` > 0 accepts
    a candidate that stops up to that many steps away when the baseline's stop margin
    (stop minus continue logit, ``reference["stop_margins"]``) at the first step where
    the two decisions differ is within ``near_tie`` of zero, a near-tie that rounding can
    flip. Without recorded margins the check stays exact. The margins are reported."""
    ref_steps, new_steps = int(reference["steps"]), int(candidate["steps"])
    margins = reference.get("stop_margins")
    if margins is not None and len(margins) != ref_steps:
        margins = None
    metrics: dict[str, float | int | str] = {"steps": new_steps, "reference_steps": ref_steps}
    if margins:
        metrics["reference_stop_margin"] = round(float(margins[-1]), 4)
    if new_steps == ref_steps:
        ref_len, new_len = reference.get("output_length"), candidate.get("output_length")
        if ref_len is None or new_len is None or int(ref_len) == int(new_len):
            return Comparison(True, metrics)
        metrics.update(output_length=int(new_len), reference_output_length=int(ref_len))
        return Comparison(
            False, metrics, f"output length {new_len} != {ref_len} after the same {new_steps} steps"
        )
    max_steps = reference.get("max_steps")
    if new_steps > ref_steps and max_steps is not None and new_steps >= int(max_steps):
        what = f"never fires (ran to max_steps={max_steps})"
    elif new_steps > ref_steps:
        what = f"fires {new_steps - ref_steps} step(s) late"
    else:
        what = f"fires {ref_steps - new_steps} step(s) early"
    reason = f"the stop condition {what}: {new_steps} steps, the baseline {ref_steps}"
    if not margins:
        return Comparison(False, metrics, reason)
    # The first step whose decisions differ: the baseline stops at its last step while
    # the candidate goes on, or the candidate stops at an earlier one.
    step = min(ref_steps, new_steps) - 1
    margin = float(margins[step])
    metrics.update(differing_step=step, differing_step_margin=round(margin, 4))
    reason += f" (baseline stop margin {margin:+.3f} at step {step})"
    consulted = step > int(reference.get("min_steps", 0))  # before that the stop is ignored
    if abs(new_steps - ref_steps) <= tolerance and abs(margin) <= near_tie and consulted:
        metrics["tolerated"] = f"{reason}: a near-tie (|margin| <= {near_tie:g})"
        return Comparison(True, metrics)
    return Comparison(False, metrics, reason)
