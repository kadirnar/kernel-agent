"""The distributional comparison of sampling targets (``kernels/distribution.py``, #227).

CPU and deterministic: fake samplers on fixed logits, every draw seeded. The reference
samples as transformers' generation code does (temperature, top-k, top-p by ``sort`` and
``cumsum``, ``multinomial``); the "library" draws the same distribution from another random
stream (an inverse CDF of ``torch.rand``: never the reference's tokens for a seed), so only
a distributional comparison can pass it; wrong temperatures, top-p and top-k must fail.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from libscout_toy import Head

from kernel_agent.kernels import distribution

#: 4 requests over a 50-token vocabulary (their own generator: the global one is untouched)
LOGITS = 2.0 * torch.randn(4, 50, generator=torch.Generator().manual_seed(1234))
#: Draws per sampler here (the default is 2000): every wrong sampler below is then more than
#: twice the bound (at 1000 draws, measured: temperature 1.0 against 0.8: 517 against a
#: threshold of 122; 0.9: 232 against 106; top-p 0.95 against 0.9: 246 against 109)
N = 1000


def _filtered(logits: torch.Tensor, temperature: float, top_p: float, top_k: int | None):
    """The sampling distribution: tokens sorted by probability, those outside top-k and
    outside the nucleus (the smallest prefix whose mass reaches top-p) zeroed."""
    probs = torch.softmax(logits / temperature, dim=-1)
    ordered, index = probs.sort(dim=-1, descending=True)
    beyond = ordered.cumsum(-1) - ordered > top_p  # starts after the nucleus is full
    ordered = ordered.masked_fill(beyond, 0.0)
    if top_k is not None:
        ordered[:, top_k:] = 0.0
    return ordered / ordered.sum(-1, keepdim=True), index


def reference(temperature: float = 0.8, top_p: float = 0.9, top_k: int | None = None):
    def sample() -> torch.Tensor:
        probs, index = _filtered(LOGITS, temperature, top_p, top_k)
        return index.gather(-1, torch.multinomial(probs, 1))

    return sample


def library(temperature: float = 0.8, top_p: float = 0.9, top_k: int | None = None):
    """The same distribution from another random stream (inverse CDF of uniforms)."""

    def sample() -> torch.Tensor:
        probs, index = _filtered(LOGITS, temperature, top_p, top_k)
        u = torch.rand(probs.shape[0], 1)
        pick = torch.searchsorted(probs.cumsum(-1), u).clamp(max=probs.shape[-1] - 1)
        return index.gather(-1, pick)

    return sample


def test_a_correct_sampler_on_another_random_stream_passes():
    ref, lib = distribution.draws(reference(), N), distribution.draws(library(), N)
    assert not torch.equal(ref, lib)  # not one token for token: a one-to-one compare fails
    found = distribution.homogeneity(ref, lib)
    assert found["ok"], found
    assert found["positions"] == 4 and found["draws"] == [N, N]
    assert found["statistic"] < found["threshold"]
    same = distribution.compare(reference(), reference(), n=N)  # one stream: equal draws
    assert same["ok"] and same["statistic"] == 0.0


@pytest.mark.parametrize(
    "wrong",
    [
        {"temperature": 1.0},  # the reference's 0.8
        {"temperature": 0.9},
        {"top_p": 0.8},  # 0.9
        {"top_p": 0.95},
        {"top_k": 10, "top_p": 1.0},  # against top-k 20, no nucleus
    ],
)
def test_a_wrong_temperature_top_p_or_top_k_fails(wrong: dict[str, Any]):
    right = {"temperature": 0.8, "top_p": 0.9}
    if "top_k" in wrong:
        right = {"temperature": 0.8, "top_p": 1.0, "top_k": 20}
    found = distribution.compare(reference(**right), library(**{**right, **wrong}), n=N)
    assert not found["ok"], found
    assert found["statistic"] > 2 * found["threshold"]  # far beyond the bound, not marginal
    assert found["worst"][0]["top"]["reference"] != found["worst"][0]["top"]["candidate"]


def test_the_bound_holds_for_correct_samplers():
    """At alpha 0.05 the right sampler fails about 5 % of seeded comparisons (the test's
    calibration; the default alpha is 1e-6)."""
    failed = 0
    for seed in range(20):
        found = distribution.compare(reference(), library(), n=200, seed=1000 * seed, alpha=0.05)
        failed += not found["ok"]
    assert failed <= 3  # 20 comparisons at 5 %: 1 expected (this seeding: 1, measured)


def test_chi_square_quantiles():
    # known quantiles (statistical tables): within 1 % from df 10, conservative at df 1
    for df, alpha, exact in ((10, 0.05, 18.307), (10, 0.001, 29.588), (100, 0.01, 135.807)):
        assert distribution.chi2_quantile(df, alpha) == pytest.approx(exact, rel=0.01)
    assert distribution.chi2_quantile(1, 1e-6) > 23.928
    assert distribution.chi2_quantile(0) == 0.0


def test_rare_tokens_are_pooled_into_one_bin():
    a = torch.tensor([0] * 50 + [1] * 40 + [2, 3, 4] * 3 + [5])
    b = torch.tensor([0] * 52 + [1] * 38 + [2, 3] * 4 + [6, 7])
    first, second = distribution._bins(a, b)
    assert first.tolist() == [50, 40, 10] and second.tolist() == [52, 38, 10]
    stat, df = distribution._statistic(first, second)
    assert df == 2 and stat == pytest.approx(4 / 102 + 4 / 78)


def test_draws_leave_the_callers_generator_as_it_was():
    torch.manual_seed(7)
    want = torch.rand(3)
    torch.manual_seed(7)
    distribution.draws(reference(), 5, seed=100)
    assert torch.equal(torch.rand(3), want)


def test_draws_must_be_token_ids_of_one_shape():
    ints = torch.zeros(10, 4, dtype=torch.int64)
    assert "shape" in distribution.homogeneity(ints, torch.zeros(10, 3, dtype=torch.int64))["error"]
    assert "integers" in distribution.homogeneity(ints, ints.float())["error"]
    with pytest.raises(TypeError, match="tensor of token ids"):
        distribution.draws(lambda: [1, 2], 2)


def test_only_a_sampling_reference_opts_in():
    head = Head().eval()
    x = torch.randn(2, 16)
    assert distribution.sampling_reason(head, (x,)) is None
    ranked = distribution.sampling_reason(lambda t: head(t, draw=False), (x,))
    assert ranked == distribution.NOT_SAMPLING
