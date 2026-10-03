"""Workload contract: how to load, run and compare one model end to end.

A workload is the ground truth for an optimisation run.  It must be
deterministic (fixed seeds, greedy decoding) so that the output of the
optimised model can be compared against the baseline output.
"""

from __future__ import annotations

import statistics
import time
from abc import ABC, abstractmethod
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
