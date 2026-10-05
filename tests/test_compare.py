from dataclasses import dataclass

import torch

from kernel_agent.kernels.compare import compare_structures, compare_tensors, flatten
from kernel_agent.workloads.base import compare_audio, compare_tokens, cosine, psnr


@dataclass
class Out:
    hidden: torch.Tensor
    extra: tuple


class Cache:
    def __init__(self):
        self.keys = [torch.ones(2, 3)]


def test_flatten_nested():
    out = Out(torch.zeros(2), (torch.ones(1), {"a": torch.ones(3)}))
    flat = flatten({"o": out, "c": Cache(), "none": None})
    assert set(flat) == {
        "out.o.hidden",
        "out.o.extra[0]",
        "out.o.extra[1].a",
        "out.c.keys[0]",
    }


def test_compare_tensors_tolerance_by_dtype():
    a = torch.randn(1000, dtype=torch.bfloat16)
    assert compare_tensors("x", a, a.clone())["ok"]
    b = a.clone()
    b[:5] += 10
    res = compare_tensors("x", a, b)
    assert not res["ok"] and res["mismatch_frac"] > 1e-3
    noisy = (a.float() * (1 + 1e-3)).to(torch.bfloat16)
    assert compare_tensors("x", a, noisy)["ok"]


def test_compare_tensors_errors():
    a = torch.zeros(4)
    assert "shape" in compare_tensors("x", a, torch.zeros(5))["error"]
    assert "dtype" in compare_tensors("x", a, torch.zeros(4, dtype=torch.float16))["error"]
    assert "NaN" in compare_tensors("x", a, torch.full((4,), float("nan")))["error"]
    assert not compare_tensors("i", torch.arange(4), torch.arange(4) + 1)["ok"]


def test_compare_structures_missing():
    res = compare_structures((torch.ones(2), torch.ones(3)), (torch.ones(2),))
    assert [r["ok"] for r in res] == [True, False]


def test_cosine_masked_logits():
    a = torch.tensor([1.0, float("-inf"), 2.0])
    assert abs(cosine(a, a.clone()) - 1.0) < 1e-6
    b = torch.tensor([1.0, 0.0, 2.0])
    assert cosine(a, b) == 0.0


def test_compare_tokens():
    ref = torch.arange(32)
    res = compare_tokens(ref, ref.clone(), None, None, min_prefix=16, min_cosine=0.99)
    assert res.passed and res.metrics["token_prefix_match"] == 32
    late = ref.clone()
    late[20] = -1
    assert compare_tokens(ref, late, None, None, min_prefix=16, min_cosine=0.99).passed
    early = ref.clone()
    early[3] = -1
    assert not compare_tokens(ref, early, None, None, min_prefix=16, min_cosine=0.99).passed
    logits = torch.randn(1, 100)
    res = compare_tokens(ref, ref, logits, -logits, min_prefix=4, min_cosine=0.99)
    assert not res.passed and "cosine" in res.reason


def test_compare_audio_and_psnr():
    t = torch.linspace(0, 1, 24000)
    wav = torch.sin(2 * torch.pi * 220 * t)
    assert compare_audio(wav, wav + 1e-4 * torch.randn_like(wav), min_spec_cosine=0.97).passed
    assert not compare_audio(wav, torch.randn_like(wav), min_spec_cosine=0.97).passed
    assert not compare_audio(wav, wav[:12000], min_spec_cosine=0.97).passed
    img = torch.rand(1, 3, 8, 8)
    assert psnr(img, img) == float("inf")
    assert psnr(img, img + 0.01, data_range=1.0) > 35


def test_compare_tensors_non_finite_positions_must_match():
    ref = torch.tensor([1.0, float("-inf"), 2.0, 3.0])
    assert compare_tensors("x", ref, ref.clone())["ok"]
    nan = compare_tensors("x", ref, torch.full((4,), float("nan")))
    assert not nan["ok"] and "NaN at 4 positions" in nan["error"]
    finite = compare_tensors("x", ref, torch.tensor([1.0, -1e30, 2.0, 3.0]))
    assert not finite["ok"] and "-inf" in finite["error"]
    flipped = compare_tensors("x", torch.tensor([float("inf")]), torch.tensor([float("-inf")]))
    assert not flipped["ok"]


def test_compare_tensors_caps_outliers_and_checks_the_whole_tensor():
    torch.manual_seed(0)
    a = torch.randn(4096, dtype=torch.bfloat16)
    one_off = a.clone()
    one_off[7] = 1e4  # 1 of 4096 elements (< 0.1 %), but 1e5 tolerances away
    res = compare_tensors("x", a, one_off)
    assert not res["ok"] and res["mismatch_frac"] < 1e-3 and "tolerance away" in res["error"]
    near = a.clone()
    near[7] = a[7] + 5 * (2e-2 + 2e-2 * a[7].abs())  # 5x its tolerance: a rounding flip
    assert compare_tensors("x", a, near)["ok"]
    # small outputs: every element is within atol, the tensor as a whole is 10 % off
    small = (a.float() * 0.1).to(torch.bfloat16)
    biased = small + 0.01
    res = compare_tensors("x", small, biased)
    assert res["mismatch_frac"] == 0 and not res["ok"] and "relative L2" in res["error"]
    # small tensors and outputs below the absolute tolerance skip the whole-tensor check
    assert compare_tensors("x", small[:1000], biased[:1000])["ok"]
    tiny = torch.full((4096,), 1e-3, dtype=torch.bfloat16)
    assert compare_tensors("x", tiny, tiny * 1.5)["ok"]


def test_compare_tensors_rejects_subclasses_and_other_devices():
    class Sub(torch.Tensor):
        pass

    a = torch.randn(8)
    res = compare_tensors("x", a, a.clone().as_subclass(Sub))
    assert not res["ok"] and "subclass" in res["error"]
    assert compare_tensors("w", a, torch.nn.Parameter(a.clone()))["ok"]
    assert "meta" in compare_tensors("x", a, torch.empty(8, device="meta"))["error"]
