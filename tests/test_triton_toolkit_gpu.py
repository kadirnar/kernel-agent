"""GPU tests of the Triton toolkit (issue #148): cached launches, the short-sequence attention
example against SDPA (eager and under ``torch.compile(fullgraph=True)``), the tuned-config
cache with real timings, both examples through the evaluator (``selftest``), and the
evaluator's peak-memory delta."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from kernel_agent.agent.prompts import EXAMPLES_DIR
from kernel_agent.kernels import triton_launch, tuned
from kernel_agent.kernels.evaluate import evaluate
from kernel_agent.selftest import make_rmsnorm_capture, smoke_triton_tools

pytestmark = pytest.mark.gpu

triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")


def _example(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"ka_test_{name}", EXAMPLES_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@triton.jit
def _axpy(x, y, out, n, a, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = i < n
    tl.store(out + i, a * tl.load(x + i, mask=ok) + tl.load(y + i, mask=ok), mask=ok)


def test_cached_launch_matches_the_jit_launch():
    launcher = triton_launch.CachedLaunch(_axpy)
    for n in (4096, 1000, 1):  # n % 16 == 0, not, == 1: three specialisations
        x, y = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
        want, got = torch.empty_like(x), torch.empty_like(x)
        grid = (triton.cdiv(n, 256),)
        _axpy[grid](x, y, want, n, 2.5, BLOCK=256)
        for _ in range(3):
            got.zero_()
            launcher[grid](x, y, got, n, 2.5, BLOCK=256)
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        off = torch.randn(n + 1, device="cuda")[1:]  # 4-byte offset: another specialisation
        launcher[grid](off, y, got, n, 2.5, BLOCK=256)
        _axpy[grid](off, y, want, n, 2.5, BLOCK=256)
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert launcher.hits >= 6 and launcher.misses == 6


def _qkv(b, hq, hk, sq, sk, d, dtype=torch.bfloat16, strided=False):
    q = torch.randn(b, hq, sq, d, device="cuda", dtype=dtype)
    k = torch.randn(b, hk, sk, d, device="cuda", dtype=dtype)
    v = torch.randn(b, hk, sk, d, device="cuda", dtype=dtype)
    if strided:  # [B, S, H, D] storage seen as [B, H, S, D], as after a projection
        q = q.transpose(1, 2).contiguous().transpose(1, 2)
        k = k.transpose(1, 2).contiguous().transpose(1, 2)
    return q, k, v


SHAPES = [
    # b, hq, hk, sq, sk, d, options
    (32, 16, 2, 11, 11, 128, {}),
    (4, 8, 8, 32, 32, 64, {"is_causal": True}),
    (2, 12, 4, 7, 7, 80, {}),
    (3, 6, 1, 1, 24, 96, {}),
    (5, 32, 8, 16, 16, 256, {}),
    (1, 3, 3, 2, 30, 40, {"scale": 0.3}),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("strided", [False, True])
def test_short_attention_matches_sdpa(shape, strided):
    example = _example("triton_short_attention")
    b, hq, hk, sq, sk, d, options = shape
    q, k, v = _qkv(b, hq, hk, sq, sk, d, strided=strided)
    gqa = hq != hk
    assert example.supported(q, k, v, enable_gqa=gqa, **options)
    want = F.scaled_dot_product_attention(q, k, v, enable_gqa=gqa, **options)
    got = example.sdpa(q, k, v, enable_gqa=gqa, **options)
    assert got.shape == want.shape and got.dtype == want.dtype
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)


def test_short_attention_masks_fp16_and_fallbacks():
    example = _example("triton_short_attention")
    q, k, v = _qkv(2, 12, 4, 7, 7, 80)
    padding = torch.ones(2, 1, 1, 7, device="cuda", dtype=torch.bool)
    padding[1, ..., 5:] = False
    want = F.scaled_dot_product_attention(q, k, v, attn_mask=padding, enable_gqa=True)
    got = example.sdpa(q, k, v, attn_mask=padding, enable_gqa=True)
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)

    q, k, v = _qkv(3, 6, 1, 1, 24, 96, torch.float16)
    bias = torch.randn(1, 24, device="cuda", dtype=torch.float16)
    want = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, enable_gqa=True)
    got = example.sdpa(q, k, v, attn_mask=bias, enable_gqa=True)
    torch.testing.assert_close(got, want, rtol=1e-2, atol=1e-2)

    long_q, long_k, long_v = _qkv(2, 4, 2, 40, 40, 64)  # > MAX_SEQ: SDPA itself
    assert not example.supported(long_q, long_k, long_v, enable_gqa=True)
    fp32 = [t.float() for t in _qkv(1, 2, 2, 8, 8, 64)]
    assert not example.supported(*fp32)


def test_short_attention_mode_routes_sdpa_and_traces_under_fullgraph():
    example = _example("triton_short_attention")
    core = example.ShortAttentionCore()
    q, k, v = _qkv(32, 16, 2, 11, 11, 128)
    want = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    want = want.transpose(1, 2).reshape(32, 11, -1)
    calls = example._launch.hits + example._launch.misses
    with torch.inference_mode():
        eager = core(q, k, v)
    assert example._launch.hits + example._launch.misses > calls  # the kernel ran
    torch.testing.assert_close(eager, want, rtol=2e-2, atol=2e-2)
    torch._dynamo.reset()
    compiled = torch.compile(core.forward, fullgraph=True)
    with torch.inference_mode():
        out = compiled(q, k, v)
    torch.testing.assert_close(out, eager, rtol=0, atol=0)
    torch._dynamo.reset()


def test_tuned_cache_times_real_configs(tmp_path):
    store = tuned.TunedConfigs(tmp_path / "t.sqlite")
    x, y, out = (torch.randn(1 << 20, device="cuda") for _ in range(3))

    def bench(config: dict[str, int]) -> float:
        grid = (triton.cdiv(x.numel(), config["BLOCK"]),)
        return tuned.time_ms(lambda: _axpy[grid](x, y, out, x.numel(), 1.0, **config))

    candidates = [{"BLOCK": 256}, {"BLOCK": 1024}, {"BLOCK": 4096}]
    best = store.best_config("axpy", {"n": x.numel()}, candidates, bench)
    assert best in candidates
    (entry,) = store.entries()
    assert entry["ms"] > 0 and entry["versions"]["triton"] == triton.__version__
    major, minor = torch.cuda.get_device_capability()
    assert entry["gpu"] == tuned.gpu_name() and entry["gpu"].endswith(f"sm_{major}{minor}")


def test_toolkit_examples_pass_the_evaluator(tmp_path):
    assert smoke_triton_tools(tmp_path, verbose=True)


HUNGRY = """import torch


class M(torch.nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.reference = reference

    def forward(self, x):
        scratch = torch.empty(64 << 20, dtype=torch.uint8, device=x.device)  # 64 MiB
        scratch.zero_()
        h = x.float()
        var = h.pow(2).mean(-1, keepdim=True)
        return self.reference.weight * (h * torch.rsqrt(var + 1e-5)).to(x.dtype)


def build(reference):
    return M(reference)
"""


def test_the_evaluator_reports_the_peak_memory_delta(tmp_path):
    capture = make_rmsnorm_capture(tmp_path / "rms.pt", hidden=1024)
    honest = evaluate(capture, EXAMPLES_DIR / "triton_rmsnorm.py")
    assert honest["status"] == "ok", honest
    assert all("ref_peak_mib" in c and "new_peak_mib" in c for c in honest["cases"])
    assert "warning" not in honest["peak_memory"]
    path = Path(tmp_path / "hungry.py")
    path.write_text(HUNGRY)
    hungry = evaluate(capture, path)
    assert hungry["correct"], hungry
    assert hungry["peak_memory"]["delta_mib"] >= 63.0 and "warning" in hungry["peak_memory"]
