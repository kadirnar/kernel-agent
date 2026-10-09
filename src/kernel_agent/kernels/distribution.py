"""A distributional comparison for sampling targets (issue #227).

A sampler's output is a random draw. A library sampler (FlashInfer's top-k / top-p kernels,
a fused Triton sampler) draws from the same distribution as the reference's
``softmax`` / ``sort`` / ``cumsum`` / ``multinomial`` chain but from another random stream,
so even from the same seed its tokens are not the reference's: an output compared with the
captured one fails every correct sampler. What has to match is the distribution of the
draws. :func:`compare` draws ``n`` times from the reference and from the candidate on the
same inputs, seeded (``torch.manual_seed(seed + i)`` before the i-th draw of each: the same
verdict run to run), and tests per output position (one request's token) whether the two
samples come from one categorical distribution:

* **the test**: the two-sample chi-square homogeneity test of each position's token counts
  (:func:`homogeneity`), the rare tokens pooled into one bin until every bin holds
  :data:`MIN_BIN` draws of the two samples together (the condition of the chi-square
  approximation), the statistics and degrees of freedom summed over the positions (their
  draws are independent);
* **the bound**: the sum exceeds the chi-square quantile at ``1 - alpha``
  (:func:`chi2_quantile`, Wilson-Hilferty: within 1 % of the exact quantile from 10 degrees
  of freedom, above it at fewer for small alphas: the test errs towards passing) with
  probability at most about ``alpha`` when both samplers draw from one distribution: with
  :data:`ALPHA` = 1e-6 a correct sampler fails about one comparison in a million. A wrong
  one (another temperature, top-p or top-k) moves probability between tokens: the
  statistic grows like ``n`` times the chi-square divergence of the two distributions,
  summed over positions. The tests
  (``tests/test_distribution.py``: 4 requests over 50 tokens) measure that a temperature
  of 1.0 or 0.9 against 0.8, a top-p of 0.8 or 0.95 against 0.9 and a top-k of 10 against
  20 each exceed the bound more than twice at 1000 draws (half of :data:`DRAWS`), while the
  right sampler on another random stream passes.

What it does not catch: a sampler whose distribution differs by less than its power at
``n`` draws (raise ``n``: the statistic grows linearly with it), and anything a single
draw shows that a distribution does not (the evaluator's other checks stay on: aliasing,
in-place updates, timing).

Opt-in, for sampling targets only (:func:`sampling_reason`: the reference's trace has the
``sampling`` family of ``libscout.detect``: a ``multinomial`` draw, or ``sort`` / ``argsort``
with ``cumsum``). The evaluator does not call it yet: README, *Library scout*, says how it
is meant to (the integer outputs of a case compared by :func:`compare`, every other check of
a call made under a fixed seed).
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from typing import Any

import torch

#: The false-failure probability of a correct sampler per comparison
ALPHA = 1e-6
#: Draws per sampler (each position's counts; the test's power grows linearly with it)
DRAWS = 2000
#: Draws of both samples together a bin needs (5 expected per sample: chi-square's rule)
MIN_BIN = 10
#: Positions listed in a report, the largest statistics first
WORST = 3


def chi2_quantile(df: int, alpha: float = ALPHA) -> float:
    """The chi-square quantile at ``1 - alpha`` for ``df`` degrees of freedom (the
    Wilson-Hilferty approximation; 0.0 for ``df`` 0). Above the exact quantile for small
    ``df`` at small ``alpha`` (df 1, alpha 1e-6: 27.5 against 23.9): conservative."""
    if df <= 0:
        return 0.0
    z = statistics.NormalDist().inv_cdf(1.0 - alpha)
    h = 2.0 / (9.0 * df)
    return float(df * (1.0 - h + z * h**0.5) ** 3)


def _bins(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The counts of each token in two samples of one position (int64 vectors), the tokens
    with fewer than :data:`MIN_BIN` draws together pooled into one bin (merged with the
    smallest other bin while it is still short of it)."""
    tokens, inverse = torch.unique(torch.cat([a, b]), return_inverse=True)
    first = torch.bincount(inverse[: a.numel()], minlength=tokens.numel())
    second = torch.bincount(inverse[a.numel() :], minlength=tokens.numel())
    total = first + second
    small = total < MIN_BIN
    if not bool(small.any()):
        return first, second
    keep_a, keep_b = first[~small], second[~small]
    pool_a, pool_b = first[small].sum(), second[small].sum()
    if int(pool_a + pool_b) < MIN_BIN and keep_a.numel():
        smallest = int(torch.argmin(keep_a + keep_b))
        keep_a = keep_a.clone()
        keep_b = keep_b.clone()
        keep_a[smallest] += pool_a
        keep_b[smallest] += pool_b
        return keep_a, keep_b
    return torch.cat([keep_a, pool_a[None]]), torch.cat([keep_b, pool_b[None]])


def _statistic(first: torch.Tensor, second: torch.Tensor) -> tuple[float, int]:
    """The two-sample chi-square statistic of binned counts and its degrees of freedom:
    ``sum((a * sqrt(m / n) - b * sqrt(n / m)) ** 2 / (a + b))`` over the bins (n, m: the
    sample sizes; ``(a - b) ** 2 / (a + b)`` when they are equal)."""
    n, m = float(first.sum()), float(second.sum())
    a, b = first.double(), second.double()
    terms = (a * (m / n) ** 0.5 - b * (n / m) ** 0.5) ** 2 / (a + b)
    return float(terms.sum()), max(int(first.numel()) - 1, 0)


def homogeneity(
    reference: torch.Tensor, candidate: torch.Tensor, *, alpha: float = ALPHA
) -> dict[str, Any]:
    """Whether two samples of draws (``[n, ...]`` and ``[m, ...]`` integer tensors: one draw
    per row, the other dimensions the output's positions) come from one distribution at
    every position (the module docstring): ``ok``, the summed ``statistic`` and ``df``,
    the ``threshold`` it was held to, and the :data:`WORST` positions with their most drawn
    tokens."""
    if reference.shape[1:] != candidate.shape[1:]:
        return {
            "ok": False,
            "error": f"draws of shape {list(reference.shape[1:])} vs {list(candidate.shape[1:])}",
        }
    if reference.is_floating_point() or candidate.is_floating_point():
        return {"ok": False, "error": "draws are integers (token ids); got floating point"}
    ref = reference.reshape(reference.shape[0], -1).cpu()
    cand = candidate.reshape(candidate.shape[0], -1).cpu()
    total, df = 0.0, 0
    positions = []
    for j in range(ref.shape[1]):
        stat, dof = _statistic(*_bins(ref[:, j], cand[:, j]))
        total += stat
        df += dof
        positions.append((stat, dof, j))
    threshold = chi2_quantile(df, alpha)
    worst = []
    for stat, dof, j in sorted(positions, reverse=True)[:WORST]:
        if dof == 0:
            continue
        top = {}
        for name, sample in (("reference", ref[:, j]), ("candidate", cand[:, j])):
            tokens, counts = torch.unique(sample, return_counts=True)
            order = torch.argsort(counts, descending=True)[:3]
            top[name] = {int(tokens[i]): round(float(counts[i]) / len(sample), 4) for i in order}
        worst.append({"position": j, "statistic": round(stat, 2), "df": dof, "top": top})
    return {
        "ok": total <= threshold,
        "statistic": round(total, 2),
        "df": df,
        "threshold": round(threshold, 2),
        "alpha": alpha,
        "draws": [int(reference.shape[0]), int(candidate.shape[0])],
        "positions": int(ref.shape[1]),
        "worst": worst,
    }


def draws(sample: Callable[[], torch.Tensor], n: int = DRAWS, *, seed: int = 0) -> torch.Tensor:
    """``n`` outputs of ``sample()`` stacked (``[n, ...]``), the i-th drawn after
    ``torch.manual_seed(seed + i)`` (every device's generator: the reference and the
    candidate see the same seeds, a rerun the same draws); the caller's generator states
    are restored afterwards."""
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    out = []
    with torch.random.fork_rng(devices=devices):
        for i in range(n):
            torch.manual_seed(seed + i)
            value = sample()
            if not isinstance(value, torch.Tensor):
                kind = type(value).__name__
                raise TypeError(f"a sampler returns a tensor of token ids, not {kind}")
            out.append(value.detach().reshape(-1).to("cpu", torch.int64))
    return torch.stack(out)


def compare(
    reference: Callable[[], torch.Tensor],
    candidate: Callable[[], torch.Tensor],
    *,
    n: int = DRAWS,
    seed: int = 0,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """:func:`homogeneity` of ``n`` seeded :func:`draws` of the reference and of the
    candidate (zero-argument calls of each on one case's inputs)."""
    return homogeneity(draws(reference, n, seed=seed), draws(candidate, n, seed=seed), alpha=alpha)


#: Why a target that draws nothing gets no distributional comparison
NOT_SAMPLING = (
    "the distributional comparison is for sampling targets only (the reference draws: "
    "multinomial, or sort / argsort with cumsum); this reference's outputs compare one to one"
)


def sampling_reason(fn: Callable[..., Any], args: Any = (), kwargs: Any = None) -> str | None:
    """None when ``fn(*args, **kwargs)`` (the reference on a captured case) draws: the
    ``sampling`` family of ``libscout.detect`` in its trace; else :data:`NOT_SAMPLING` (the
    opt-in is refused: a deterministic output keeps its exact comparison)."""
    from kernel_agent.libscout import detect

    found = detect.families(detect.trace(fn, args, kwargs))
    return None if "sampling" in found else NOT_SAMPLING
