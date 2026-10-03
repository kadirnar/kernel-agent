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
