"""The reference's own rounding spread (issue #250): a check on redrawn inputs that fails is
judged again against the reference with its rounding redrawn (``verify.Rerounding``), which
widens the exact tolerance for noise-like errors only.

CPU, deterministic: seeded draws and redraws (``verify._seed`` pinned). A deep bf16 chain
(``selftest.NormGemvChain``, 8 layers) computed in fp32 throughout and rounded once (as a
fused kernel or ``torch.compile`` does: closer to fp32 than eager) fails the plain exact
tolerance on redrawn inputs and passes within the spread; outputs of other inputs, a skipped
layer, a stale tile and a systematic bias still fail. The GPU side (the megakernel example
and its baselines at 8 layers): ``tests/test_megakernel_gpu.py``.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch import nn

from kernel_agent.kernels import bench, compare, recheck, verify
from kernel_agent.profiling.capture import capture_calls
from kernel_agent.profiling.state import StatefulCall
from kernel_agent.selftest import NormGemvChain

HIDDEN, LAYERS = 256, 8

FUSED = """import torch
from torch import nn


class Fused(nn.Module):
    '''The chain in fp32 throughout, rounded once at the end.'''

    def __init__(self, reference):
        super().__init__()
        self.reference = reference

    def forward(self, x):
        h = x.float()
        for norm, layer in zip(self.reference.norms, self.reference.layers):
            n = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
            h = h + (n * norm.weight.float()) @ layer.weight.float().t()
        return h.to(x.dtype)


def build(reference):
    return Fused(reference)
"""


@pytest.fixture(autouse=True)
def _seeded(monkeypatch):
    monkeypatch.setattr(verify, "_seed", lambda: 250)


def _chain(layers: int = LAYERS) -> NormGemvChain:
    torch.manual_seed(0)
    module = NormGemvChain(HIDDEN, layers).to(torch.bfloat16)
    with torch.no_grad():
        for name, weight in module.named_parameters():
            if name.startswith("norms."):
                weight.normal_(1.0, 0.1)
            else:
                weight.normal_(0.0, HIDDEN**-0.5)
    return module.eval()


def _fused(ref: NormGemvChain, x: torch.Tensor) -> torch.Tensor:
    h = x.float()
    for norm, layer in zip(ref.norms, ref.layers, strict=True):
        n = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
        h = h + (n * norm.weight.float()) @ layer.weight.float().t()
    return h.to(x.dtype)


def _draws(n: int, rows: int = 1) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(1)
    return [torch.randn(rows, HIDDEN, generator=gen).to(torch.bfloat16) for _ in range(n)]


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16).clone()


def test_rerounding_moves_rounded_results_by_one_ulp_at_most_and_nothing_else():
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(64, 64, generator=gen).to(torch.bfloat16)
    y = torch.randn(64, 64, generator=gen).to(torch.bfloat16)
    x[0, :4] = torch.tensor([0.0, math.inf, -math.inf, math.nan])
    kept = _bits(x)
    with torch.inference_mode():
        plain, narrowed_plain = x + y, (y.float() * 1.1).to(torch.bfloat16)
        with verify.Rerounding(1):
            summed = x + y
            viewed, cloned, joined = x.view(-1), x.clone(), torch.cat([x, y])
            widened = x.float()
            narrowed = (y.float() * 1.1).to(torch.bfloat16)
            zeros = y * 0
    finite = torch.isfinite(plain)
    moved = (summed != plain) & finite
    assert 0.05 < float(moved.float().mean()) < 0.6  # about a quarter of the elements
    step = (summed.float() - plain.float()).abs()[finite]
    assert bool((step <= plain.float().abs()[finite] * 2**-7).all())  # one ulp at most
    assert summed[0, 1] == math.inf and summed[0, 2] == -math.inf  # infinities and NaN stay
    assert bool(summed[0, 3].isnan())
    assert torch.equal(_bits(x), kept)  # the input (and its view) untouched
    assert torch.equal(_bits(viewed), _bits(x.view(-1))) and torch.equal(_bits(cloned), kept)
    assert torch.equal(_bits(joined[:64]), kept)
    assert torch.equal(widened[1:], x.float()[1:])  # bf16 -> fp32 is exact
    assert not torch.equal(narrowed, narrowed_plain)  # fp32 -> bf16 rounds
    assert bool((zeros == 0).all())


def test_in_place_results_are_redrawn_but_views_and_weights_are_not():
    torch.manual_seed(0)
    linear = nn.Linear(64, 64, bias=False).to(torch.bfloat16)
    weight, version = linear.weight.detach().clone(), linear.weight._version
    x, y = (torch.randn(8, 64).to(torch.bfloat16) for _ in range(2))
    z = x.clone()
    with torch.inference_mode(), verify.Rerounding(2):
        out = linear(x)
        z.add_(y)
        z.unsqueeze_(0).squeeze_(0)
        snapshot = _bits(z)
        z.t_().t_()  # in-place views move nothing
    assert torch.equal(linear.weight, weight) and linear.weight._version == version
    assert out.shape == (8, 64)
    assert not torch.equal(z, x + y)  # the in-place add rounds: redrawn
    assert torch.equal(_bits(z), snapshot)


def test_rerounded_call_runs_on_copies_restores_state_outside_the_mode_and_survives_errors():
    depth: list[int] = []

    def restore() -> None:
        depth.append(torch._C._len_torch_dispatch_stack())

    def step(x: torch.Tensor) -> torch.Tensor:
        x.mul_(3)
        return x + 1

    x = torch.randn(32).to(torch.bfloat16)
    before = x.clone()
    found = verify.rerounded_call(StatefulCall(step, restore), (x,), {}, seed=0)
    assert found is not None
    out, args, kwargs = found
    assert torch.equal(x, before)  # the caller's inputs are not touched
    assert depth == [0]  # the case's state restored once, outside the mode
    assert args[0] is not x and kwargs == {}
    assert float((args[0].float() - before.float() * 3).abs().max()) <= 0.1
    assert out.shape == x.shape

    def broken(x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("an op the mode cannot run")

    assert verify.rerounded_call(broken, (x,), {}) is None


def test_a_deep_chain_that_rounds_differently_passes_within_the_spread():
    ref = _chain()
    failed_plain = 0
    for i, x in enumerate(_draws(8)):
        with torch.inference_mode():
            r = ref(x)
        new = _fused(ref, x)
        found = verify.rerounded_call(ref, (x,), {}, seed=i)
        assert found is not None
        plain = compare.compare_tensors("out", r, new, perturbed=True)
        judged = compare.compare_tensors("out", r, new, perturbed=True, rerounded=found[0])
        failed_plain += not plain["ok"]
        assert "rounding_spread" not in plain
        assert judged["ok"], judged
        assert 0 < judged["error_over_spread"] < 1  # no larger than another rounding of eager
        assert 0.003 < judged["rounding_spread"] < 0.05
    assert failed_plain >= 4  # on this CPU: all 8 draws are past the plain tolerance


def test_what_the_check_is_for_still_fails_within_the_spread():
    ref = _chain()
    draws = _draws(3)
    with torch.inference_mode():
        outs = [ref(x) for x in draws]
        x = draws[1]
        skipped = x
        for norm, layer in zip(ref.norms[:-1], ref.layers[:-1], strict=True):
            skipped = skipped + layer(norm(skipped))
    r = outs[1]
    found = verify.rerounded_call(ref, (x,), {}, seed=1)
    assert found is not None
    alt = found[0]
    rms = float(r.float().pow(2).mean().sqrt())
    stale = r.clone()
    stale[:, 16:32] = outs[0][:, 16:32]
    cheats = {
        "the previous call's output": outs[0],
        "the last layer skipped": skipped,
        "a stale 16-element tile": stale,
        "a bias of 5 % of the RMS": (r.float() + 0.05 * rms).to(r.dtype),
    }
    for what, new in cheats.items():
        judged = compare.compare_tensors("out", r, new, perturbed=True, rerounded=alt)
        assert not judged["ok"], what
        assert judged["error_over_spread"] > compare.ROUNDING_SPREAD_RMS, what
        assert "not rounding" in judged["error"], what


def test_the_widening_is_capped_and_needs_the_references_shape():
    torch.manual_seed(3)
    r = torch.randn(64, 32).to(torch.bfloat16)
    rms = float(r.float().pow(2).mean().sqrt())
    spread = (r.float() + rms * torch.randn(64, 32)).to(torch.bfloat16)  # a reference that
    new = (r.float() + 0.5 * rms * torch.randn(64, 32)).to(torch.bfloat16)  # spreads far
    judged = compare.compare_tensors("out", r, new, perturbed=True, rerounded=spread)
    assert judged["error_over_spread"] <= compare.ROUNDING_SPREAD_RMS
    assert not judged["ok"]  # widened by ROUNDING_SPREAD_CAP x RMS only, not 4 x RMS
    assert "with 4 x the reference's own rounding spread" in judged["error"]
    for other in (spread[:, :16], spread.float(), None):
        plain = compare.compare_tensors("out", r, r.clone(), perturbed=True, rerounded=other)
        assert plain["ok"] and "rounding_spread" not in plain


def test_caches_take_the_spread_of_the_elements_they_compare():
    ref = _chain()
    x = _draws(1)[0]
    with torch.inference_mode():
        r = ref(x)
    new = _fused(ref, x)
    found = verify.rerounded_call(ref, (x,), {}, seed=0)
    assert found is not None
    alt = found[0]
    kept = torch.randn(5, HIDDEN).to(torch.bfloat16)

    def cache(row: torch.Tensor) -> torch.Tensor:  # an in-place write of row 2
        c = kept.clone()
        c[2] = row[0]
        return c

    for rerounded, ok in ((None, False), ({"c": cache(alt)}, True)):
        checks = compare.compare_side_effects(
            {"c": kept}, {"c": cache(r)}, {"c": cache(new)}, perturbed=True, rerounded=rerounded
        )
        assert checks[0]["ok"] is ok, checks
    grown_ref, grown_new, grown_alt = (torch.cat([kept, t]) for t in (r, new, alt))
    plain = compare.compare_output("g", grown_ref, grown_new, [kept], perturbed=True)
    judged = compare.compare_output(
        "g", grown_ref, grown_new, [kept], perturbed=True, rerounded=grown_alt
    )
    assert "grown" in judged and not plain["ok"] and judged["ok"], judged
    static = torch.full((4, 16, HIDDEN), 100.0, dtype=torch.bfloat16)  # no output value

    def written(row: torch.Tensor) -> torch.Tensor:  # a functional write of slot 5
        s = static.clone()
        s[:, 5] = row
        return s

    plain = compare.compare_output("w", written(r), written(new), [static], perturbed=True)
    judged = compare.compare_output(
        "w", written(r), written(new), [static], perturbed=True, rerounded=written(alt)
    )
    assert "written" in judged and not plain["ok"] and judged["ok"], judged


def test_the_timed_output_check_and_the_reverification_judge_a_failure_again(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    calls: list[int] = []
    real = verify.rerounded_call

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(verify, "rerounded_call", counted)
    ref = _chain()
    x, other = _draws(2)
    with torch.inference_mode():
        r, previous = ref(x), ref(other)

    def kept(output: torch.Tensor) -> dict:
        return {"iteration": 7, "pre": ((x,), {}), "post": ((x,), {}), "output": output}

    assert bench.check_timed_output(ref, kept(r.clone())) == {"iteration": 7, "failures": []}
    assert calls == []  # a pass costs no extra reference call
    assert bench.check_timed_output(ref, kept(_fused(ref, x)))["failures"] == []
    assert len(calls) == 1
    failures = bench.check_timed_output(ref, kept(previous))["failures"]
    assert failures and "not rounding" in failures[0]["error"]

    def fused(x: torch.Tensor) -> torch.Tensor:
        return _fused(ref, x)

    checks = verify._against_reference(ref, fused, (x.clone(),), {}, lambda: None)
    assert checks and all(c["ok"] for c in checks), checks
    assert "rounding_spread" in checks[0]

    def cached(_: torch.Tensor) -> torch.Tensor:
        return previous.clone()

    checks = verify._against_reference(ref, cached, (x.clone(),), {}, lambda: None)
    assert not checks[0]["ok"]


def test_the_recheck_keeps_the_rerounded_reference_and_judges_with_it():
    def step(x: torch.Tensor, cache: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        cache[0] = x * 1.5
        return x * 3

    x = torch.randn(4, 8).to(torch.bfloat16)
    cache = torch.zeros(2, 4, 8, dtype=torch.bfloat16)
    ids = torch.zeros(3)
    entry = {"args": (x, cache, ids), "kwargs": {}}
    written = cache.clone()
    out = step(x, written, ids)
    expected = recheck._flat(output=out, args=(x, written, ids), kwargs={})
    kept = recheck._rerounded(step, entry, expected)
    assert set(kept["output"]) == {"output"}
    assert set(kept["args"]) == {"args[1]"}  # the cache the call wrote; x and ids as they were
    assert torch.equal(cache, torch.zeros_like(cache))  # the entry's inputs are not touched

    ref = _chain()
    xs = _draws(1)
    with torch.inference_mode():
        r = ref(xs[0])
    found = verify.rerounded_call(ref, (xs[0],), {}, seed=0)
    assert found is not None
    exp = {"output": {"output": r}, "pre": {}, "args": {}, "kwargs": {}}
    saved = [{"output": {"output": _fused(ref, xs[0])}, "args": {}, "kwargs": {}}]
    assert recheck.compare_entries([exp], saved, [(0, 0)])[0]  # plain: rejected
    exp["rerounded"] = {"output": {"output": found[0]}}
    assert recheck.compare_entries([exp], saved, [(0, 0)]) == [[]]


@pytest.fixture
def cpu_children(monkeypatch):
    """The re-check's subprocesses on the CPU, and no GPU lock to wait for."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("KERNEL_AGENT_LOCK_HELD", "1")


def test_the_recheck_passes_a_deep_chain_that_rounds_differently(tmp_path: Path, cpu_children):
    ref = _chain()
    capture = tmp_path / "chain.pt"
    capture_calls(ref, [((x,), {}, 1) for x in _draws(2, rows=2)], capture)
    candidate = tmp_path / "fused.py"
    candidate.write_text(FUSED)
    verdict = {"status": "ok", "correct": True, "speedup": 1.4}
    result = recheck.run_recheck(capture, candidate, verdict=verdict, seed=5)
    assert result["status"] == "ok" and result["passed"] and result["correct"], result
