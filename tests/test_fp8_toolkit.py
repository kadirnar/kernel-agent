"""FP8 toolkit (#145): the ``fp8_kv`` precision class (opt-in, near-lossless tier) and its
reference math, the MXFP8 scale rule and cuBLASLt's blocked scale layout of the direct
cuBLASLt example, and the Triton examples compiled for sm_120 without a GPU (the W8A8 GEMM on
the block-scaled MMA: PTX ``block_scale``). ``gpu``: every new example on the evaluator and
against its reference math (written with the examples; not run in this PR)."""

import importlib.util
import math

import pytest
import torch

from kernel_agent import orchestrator, precisions, selftest
from kernel_agent.agent import prompts
from kernel_agent.kernels import compare, kv_quant, quant
from kernel_agent.kernels.evaluate import evaluate
from kernel_agent.profiling import ceilings
from kernel_agent.profiling.capture import capture_calls

NEAR = "near-lossless"


def _example(name: str):
    spec = importlib.util.spec_from_file_location(
        f"_toolkit_{name[:-3]}", prompts.EXAMPLES_DIR / name
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- fp8_kv: the class


def test_fp8_kv_is_an_opt_in_near_lossless_precision():
    assert "fp8_kv" in compare.REDUCED_PRECISIONS and "fp8_kv" in precisions.OPT_IN
    assert compare.tier_for(NEAR, "fp8_kv") == compare.NEAR_LOSSLESS_KV_TIER
    assert compare.tier_for("exact", "fp8_kv") == compare.EXACT_TIER
    # not in a near-lossless run's default: --precisions must name it
    assert "fp8_kv" not in precisions.default(NEAR)
    assert precisions.parse("fp8_kv") == ["exact", "fp8_kv"]
    assert precisions.allowed(NEAR, ["fp8_kv"]) == ("exact", "fp8_kv")
    assert precisions.allowed("exact", ["fp8_kv"]) == ("exact",)
    why = precisions.refusal("fp8_kv", precisions.default(NEAR))
    assert "not allowed" in why and "fp8_kv is opt-in" in why and "4-bit" not in why
    assert "4-bit precisions are opt-in" in precisions.refusal("fp4_weights", ("exact",))
    assert precisions.refusal("fp8_kv", ("exact", "fp8_kv")) is None

    target = {"id": "attn", "precision": "fp8_kv", "precision_why": "8k contexts: KV 60 %"}
    assert "not allowed" in orchestrator._precision(dict(target), NEAR)
    assert orchestrator._precision(dict(target), NEAR, ("exact", "fp8_kv")) is None
    assert "needs --quality near-lossless" in orchestrator._precision(dict(target), "exact")
    # no ceilings column of its own (KV-cache bytes are not counted yet): the exact floor
    assert ceilings.target_precision("fp8_kv") is ceilings.PRECISIONS["exact"]


def test_fp8_kv_in_the_planner_and_engineer_prompts():
    enum = prompts.PLAN_SCHEMA["properties"]["targets"]["items"]["properties"]["precision"]
    assert "fp8_kv" in enum["enum"]
    assert "fp8_kv" not in str(prompts.plan_schema(precisions.default(NEAR)))
    default = prompts.precision_policy(NEAR)
    assert '`precision: "fp8_kv"`' not in default and "fp8_kv" not in default
    asked = prompts.precision_policy(NEAR, (*precisions.default(NEAR), "fp8_kv"))
    assert '`precision: "fp8_kv"`' in asked and "kv_cache_share" in asked
    assert "On short caches it is slower" in asked

    spec = {"id": "attn", "module_class": "Attention", "why": "w", "approach": "a"}
    spec |= {"backends": ["triton"], "precision": "fp8_kv", "precision_why": "8k contexts"}
    capture = {"cases": [], "tier": NEAR, "precision": "fp8_kv"}
    text = prompts.engineer_prompt(spec, capture, ["triton"], "py", "tc", 10, None)
    assert prompts.reduced_precision(spec, capture) == "fp8_kv"
    for needle in (
        "# Precision: `fp8_kv`",
        "FP8 KV cache:",
        "one fp32 scale per (token, KV head)",
        "never re-quantise the whole cache per step",
        "triton_fp8_kv_decode.py",
        "fp8_kv_error(k, v)",
    ):
        assert needle in text, needle
    w8a8 = prompts.engineer_prompt(
        {**spec, "precision": "fp8_w8a8"},
        {**capture, "precision": "fp8_w8a8"},
        ["cuda"],
        "py",
        "tc",
        10,
        None,
    )
    assert "cuda_cublaslt_fp8.py" in w8a8 and "triton_fp8_producers.py" in w8a8


# ---------------------------------------------------------------- fp8_kv: reference math


def test_quantize_fp8_kv_per_token_and_head():
    gen = torch.Generator().manual_seed(0)
    k = torch.randn(2, 3, 17, 64, generator=gen).to(torch.bfloat16)
    k[0, 1, 5] = 0  # a token of zeros
    k[1, 2, 3, 7] = 50.0  # an outlier: only its own row's scale moves
    codes, scales = kv_quant.quantize_fp8_kv(k)
    assert codes.dtype == torch.float8_e4m3fn and codes.shape == k.shape
    assert scales.shape == (2, 3, 17) and scales.dtype == torch.float32
    amax = k.float().abs().amax(-1)
    nonzero = amax > 0
    assert torch.allclose(scales[nonzero], amax[nonzero] / 448.0)
    assert scales[0, 1, 5] == 1.0 and not codes[0, 1, 5].float().any()
    assert torch.equal(codes.float().abs().amax(-1)[nonzero], torch.full_like(amax[nonzero], 448))
    deq = kv_quant.dequantize_fp8_kv(codes, scales)
    rel = float((deq - k.float()).norm() / k.float().norm())
    assert 0.01 < rel < 0.04
    report = kv_quant.fp8_kv_error(k, k)
    assert report["k_rel_l2"] == pytest.approx(rel, rel=1e-3)
    rows = 2 * 3 * 17
    assert report["bytes"] == {"before": 2 * rows * 64 * 2, "after": 2 * (rows * 64 + 4 * rows)}


def test_the_fp8_kv_cache_quantises_on_append():
    gen = torch.Generator().manual_seed(1)
    k = torch.randn(2, 2, 9, 32, generator=gen).to(torch.bfloat16)
    v = torch.randn(2, 2, 9, 32, generator=gen).to(torch.bfloat16)
    cache = kv_quant.Fp8KVCache(2, 2, 12, 32)
    cache.append(k[:, :, :5], v[:, :, :5])
    for t in range(5, 9):  # decode: one token per step
        kc, ks, vc, vs = cache.append(k[:, :, t : t + 1], v[:, :, t : t + 1])
    assert cache.length == 9 and kc.shape == (2, 2, 9, 32) and vs.shape == (2, 2, 9)
    whole = kv_quant.quantize_fp8_kv(k), kv_quant.quantize_fp8_kv(v)
    assert torch.equal(kc.float(), whole[0][0].float()) and torch.equal(ks, whole[0][1])
    assert torch.equal(vc.float(), whole[1][0].float()) and torch.equal(vs, whole[1][1])
    with pytest.raises(ValueError, match="cache full"):
        cache.append(k[:, :, :4], v[:, :, :4])


def test_fp8_kv_attention_against_sdpa():
    gen = torch.Generator().manual_seed(2)
    q = torch.randn(3, 8, 1, 64, generator=gen)
    k = torch.randn(3, 2, 40, 64, generator=gen)
    v = torch.randn(3, 2, 40, 64, generator=gen)
    kc, ks = kv_quant.quantize_fp8_kv(k)
    vc, vs = kv_quant.quantize_fp8_kv(v)
    out = kv_quant.fp8_kv_attention(q, kc, ks, vc, vs)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    assert out.shape == ref.shape and float((out - ref).norm() / ref.norm()) < 0.04
    exact = kv_quant.fp8_kv_attention(  # no quantisation error left: the math is SDPA's
        q,
        k.to(torch.float8_e4m3fn),
        torch.ones(3, 2, 40),
        v.to(torch.float8_e4m3fn),
        torch.ones(3, 2, 40),
    )
    k8, v8 = k.to(torch.float8_e4m3fn).float(), v.to(torch.float8_e4m3fn).float()
    sdpa8 = torch.nn.functional.scaled_dot_product_attention(q, k8, v8, enable_gqa=True)
    assert torch.allclose(exact, sdpa8, atol=1e-5)
    # per-sequence lengths: the first n tokens only
    lengths = torch.tensor([40, 1, 17])
    masked = kv_quant.fp8_kv_attention(q, kc, ks, vc, vs, scale=0.1, lengths=lengths)
    for b, n in enumerate(lengths.tolist()):
        alone = kv_quant.fp8_kv_attention(
            q[b : b + 1],
            kc[b : b + 1, :, :n],
            ks[b : b + 1, :, :n],
            vc[b : b + 1, :, :n],
            vs[b : b + 1, :, :n],
            scale=0.1,
        )
        assert torch.allclose(masked[b : b + 1], alone, atol=1e-6)


def test_kv_cache_share_from_the_models_shapes():
    # 28 layers x 8 KV heads x 128 dims, batch 8: 2 x 8 x 28 x 8 x 128 x 2 B = 917.5 KB per
    # token; 8k tokens: 7.5 GB of KV against 16 GB of bf16 weights per step
    share = kv_quant.kv_cache_share(
        batch=8, tokens=8192, layers=28, kv_heads=8, head_dim=128, other_bytes=16e9
    )
    kv = 2 * 8 * 8192 * 28 * 8 * 128 * 2
    assert share["kv_bytes"] == kv and share["share"] == pytest.approx(kv / (kv + 16e9))
    fp8 = 2 * 8 * 8192 * 28 * 8 * (128 + 4)
    assert share["saved_share"] == pytest.approx((kv - fp8) / (kv + 16e9))
    short = kv_quant.kv_cache_share(
        batch=1, tokens=77, layers=28, kv_heads=8, head_dim=128, other_bytes=16e9
    )
    assert short["share"] < 0.01  # a short cache: nothing to win


KV_CANDIDATE = """import torch
from kernel_agent.kernels.kv_quant import fp8_kv_attention, quantize_fp8_kv

BUG = {bug!r}


class Fp8KV(torch.nn.Module):
    def forward(self, q, k, v):
        kc, ks = quantize_fp8_kv(k)
        vc, vs = quantize_fp8_kv(v)
        if BUG == "v scale x1.05":
            vs = vs * 1.05
        elif BUG == "first token's scale":
            ks, vs = ks[..., :1].expand_as(ks), vs[..., :1].expand_as(vs)
        return fp8_kv_attention(q, kc, ks, vc, vs)


def build(reference):
    return Fp8KV()
"""


@pytest.fixture(scope="module", params=[NEAR, "relaxed"])
def kv_capture(tmp_path_factory, request):
    path = tmp_path_factory.mktemp("kv") / f"decode.{request.param}.pt"
    gen = torch.Generator().manual_seed(3)
    cases = []
    for batch, tokens, count in ((2, 192, 28), (1, 33, 0)):
        q = torch.randn(batch, 8, 1, 64, generator=gen).to(torch.bfloat16)
        k = torch.randn(batch, 2, tokens, 64, generator=gen).to(torch.bfloat16)
        v = torch.randn(batch, 2, tokens, 64, generator=gen).to(torch.bfloat16)
        cases.append(((q, k, v), {}, count))
    tier = compare.tier_for(request.param, "fp8_kv")  # near-lossless-kv, relaxed-kv (#175)
    capture_calls(selftest.DecodeAttention(), cases, path, tier=tier, precision="fp8_kv")
    return path, tier


@pytest.mark.parametrize(
    ("bug", "status"),
    [(None, "ok"), ("v scale x1.05", "incorrect"), ("first token's scale", "incorrect")],
)
def test_the_fp8_kv_reference_math_fits_the_near_lossless_tier(tmp_path, kv_capture, bug, status):
    """e4m3 K / V with per-token scales pass captured and redrawn inputs; broken scales fail."""
    capture, tier = kv_capture
    path = tmp_path / "fp8_kv.py"
    path.write_text(KV_CANDIDATE.format(bug=bug))
    result = evaluate(capture, path, device="cpu")
    assert result["status"] == status, result
    if bug is None:
        assert result["correct"] and result["tolerance_tier"] == tier
        assert tier in (compare.NEAR_LOSSLESS_KV_TIER, compare.RELAXED_KV_TIER)
        assert "perturbed" in result["checks"]
        assert all(c["max_rel_l2"] < 0.04 for c in result["cases"])


# ---------------------------------------------------------------- MXFP8 (the fp8_mx rule)


def _bits_rule(amax: torch.Tensor) -> torch.Tensor:
    """The exponent rule of the cuBLASLt example's ``quant_mx_k`` and the producers' ``_ue8m0``
    (C++ / Triton), mirrored: from the bits of amax = m * 2^E, E - 8 plus one when m > 1.75;
    zero / subnormal maxima -127; clamped to [-127, 126]."""
    bits = amax.float().contiguous().view(torch.int32)
    field = (bits >> 23) & 0xFF
    e = field - 127 - 8 + ((bits & 0x7FFFFF) > 0x600000).to(torch.int32)
    e = torch.where(field == 0, torch.full_like(e, -127), e)
    return e.clamp(-127, 126)


def test_the_kernels_mxfp8_exponent_rule_is_fp8_mx_s():
    gen = torch.Generator().manual_seed(4)
    edges = 448.0 * torch.pow(2.0, torch.arange(-120, 110).float())
    amax = torch.cat(
        [
            torch.rand(20000, generator=gen) * torch.logspace(-40, 38, 20000),
            torch.tensor([0.0, 1e-45, 448.0, 448.0 * 2**-3, 449.0, 500.0, 1.75, 1.7500001]),
            edges,  # 448 x powers of two: exactly at the boundary
            torch.nextafter(edges, torch.tensor(1e38)),  # just above it
        ]
    )
    # e <= 126 covers every finite bf16 activation block; a subnormal fp32 block maximum
    # (< 1.2e-38) gets -127 here and 0 from quant (amax / 448 underflows): neither saturates
    normal = (amax == 0) | (amax >= torch.finfo(torch.float32).tiny)
    checked = normal & (amax < 2.0**126)
    assert int(checked.sum()) > 19000
    assert torch.equal(_bits_rule(amax)[checked], quant.mx_scale_exponents(amax)[checked])
    tiny = amax[~normal & (amax > 0)]
    assert tiny.numel() and bool((tiny * 2.0**127 <= 448).all())  # -127: within e4m3's range


def test_cublaslt_example_mxfp8_helpers_on_cpu():
    lt = _example("cuda_cublaslt_fp8.py")
    assert lt.shape_ok(8192, 1024, lt.TENSORWISE) and lt.shape_ok(1024, 1040, lt.TENSORWISE)
    assert not lt.shape_ok(1024, 1040, lt.MXFP8) and not lt.shape_ok(1000, 1024, lt.TENSORWISE)
    lin = torch.nn.Linear(64, 32).to(torch.bfloat16)
    assert lt.build(lin) is lin  # CPU: the reference stays
    # the fp8_mx scale-rule guard runs this module-level quantiser: never saturates
    x = quant.mxfp8_stress_input((16, 256))
    codes, scales = lt.quantize_activations(x)
    ref_codes, ref_scales = quant.quantize_mxfp8(x)
    assert torch.equal(codes.float(), ref_codes.float())
    assert torch.equal(scales.view(torch.uint8), ref_scales.view(torch.uint8))
    assert quant.mxfp8_scale_problem(x, scales) is None


# ---------------------------------------------------------------- Triton, compiled without a GPU


def test_block_scale_parsing():
    w8a8 = _example("triton_fp8_w8a8_gemm.py")
    scaled = (
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3"
        ".e4m3.f32.ue8m0 { %r1, %r2, %r3, %r4 }, ..."
    )
    plain = "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { %f1, %f2, %f3, %f4 }, ..."
    assert w8a8.block_scale_mma(f"ld.global.b32 %r1;\n{scaled}\n")
    assert not w8a8.block_scale_mma(plain) and not w8a8.block_scale_mma("// block_scale\n")
    assert w8a8.block_scale_capable((12, 0)) and w8a8.block_scale_capable((12, 1))
    assert not w8a8.block_scale_capable((9, 0)) and not w8a8.block_scale_capable((8, 9))
    assert not w8a8.block_scale_capable((10, 0))  # plain tl.dot is full rate on sm_100
    assert all(cfg[2] % 32 == 0 for cfg in (*w8a8.CONFIGS.values(), w8a8.DEFAULT))
    assert selftest.smoke_block_scale((8, 9))  # nothing to check below sm_100


def test_w8a8_example_lowers_to_the_block_scaled_mma():
    """``tl.dot_scaled`` with unit scales -> ``mma ... kind::mxf8f6f4.block_scale`` on sm_120
    (``QMMA.SF``); ``tl.dot`` on sm_89. Compiled by Triton's own ptxas, no GPU."""
    w8a8 = _example("triton_fp8_w8a8_gemm.py")
    assert w8a8.block_scale_mma(w8a8.gemm_ptx((12, 0), w8a8.CONFIGS[(8192, 1024)]))
    ada = w8a8.gemm_ptx((8, 9), w8a8.DEFAULT)
    assert not w8a8.block_scale_mma(ada) and "e4m3.e4m3.f32" in ada
    assert not w8a8.block_scale_mma(w8a8.gemm_ptx((12, 0), w8a8.DEFAULT, scaled=False))


def test_w8a8_example_reaches_each_gpus_full_rate_fp8_mma():
    """#165: compiled (no GPU) for Hopper the GEMM's ``tl.dot`` on e4m3 is ``wgmma``, for
    datacenter Blackwell ``tcgen05.mma kind::f8f6f4``; ``tl.dot_scaled`` at the example's
    64-row tiles would fall back to ``kind::f16`` there, so it is used on sm_12x only."""
    w8a8 = _example("triton_fp8_w8a8_gemm.py")
    hopper = w8a8.gemm_ptx((9, 0), w8a8.DEFAULT)
    assert "wgmma.mma_async" in hopper and ".e4m3.e4m3" in hopper
    blackwell = w8a8.gemm_ptx((10, 0), w8a8.DEFAULT)
    assert "tcgen05.mma.cta_group::1.kind::f8f6f4" in blackwell
    emulated = w8a8.gemm_ptx((10, 0), w8a8.DEFAULT, scaled=True)
    assert "kind::f16" in emulated and not w8a8.block_scale_mma(emulated)


def _compile(fn, signature, constexprs, warps=4):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    signature = {**signature, **dict.fromkeys(constexprs, "constexpr")}
    source = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    target = GPUTarget("cuda", 120, 32)
    return triton.compile(source, target=target, options={"num_warps": warps})


def test_producer_and_kv_kernels_compile_for_sm120():
    prod = _example("triton_fp8_producers.py")
    rows = {"q": "*fp8e4nv", "M": "i32", "K": "i32"}
    for mx, blocked in ((False, False), (True, True)):
        consts = {"MX": mx, "BLOCKED": blocked, "BLOCK": 1024}
        s = {"s": "*u8" if mx else "*fp32"}
        sig = {"x": "*bf16", "w": "*bf16", **rows, **s, "ldx": "i32", "eps": "fp32"}
        ptx = _compile(prod._rmsnorm_fp8_kernel, sig, consts).asm["ptx"]
        assert "cvt.rn.satfinite.e4m3x2.f32" in ptx  # round to nearest even, saturated
        sig = {"g": "*bf16", "u": "*bf16", **rows, **s, "ldg": "i32", "ldu": "i32"}
        _compile(prod._silu_mul_fp8_kernel, sig, consts, warps=8)
    kv = _example("triton_fp8_kv_decode.py")
    i32 = "i32"
    sig = {"q": "*bf16", "kc": "*fp8e4nv", "ks": "*fp32", "vc": "*fp8e4nv", "vs": "*fp32"}
    sig |= {"lens": "*i32", "po": "*fp32", "pm": "*fp32", "pl": "*fp32"}
    sig |= {"Hkv": i32, "L": i32, "chunk": i32, "sm_scale": "fp32"}
    strides = ("sqb", "sqh", "skb", "skh", "skl", "ssb", "ssh", "svb", "svh", "svl", "stb", "sth")
    sig |= dict.fromkeys(strides, i32)
    consts = {"G": 8, "GP": 16, "D": 128, "BL": kv.BL, "HAS_LENS": True}
    split = _compile(kv._split_kernel, sig, consts)
    assert split.metadata.shared <= 99 * 1024  # sm_120's shared memory per block
    sig = {"po": "*fp32", "pm": "*fp32", "pl": "*fp32", "out": "*bf16", "S": i32, "Hkv": i32}
    sig |= {"sob": i32, "soh": i32}
    _compile(kv._combine_kernel, sig, {"G": 8, "GP": 16, "D": 128})


def test_the_toolkit_examples_exist():
    for name in (
        *selftest.CUBLASLT_EXAMPLES,
        *selftest.PRODUCER_EXAMPLES,
        *selftest.FP8_KV_EXAMPLES,
    ):
        source = (prompts.EXAMPLES_DIR / name).read_text()
        assert "def build(" in source and "from kernel_agent" in source, name
    text = prompts.knowledge("low_precision.md")
    for needle in ("fp8_kv", "cuda_cublaslt_fp8.py", "triton_fp8_producers.py", "kv_cache_share"):
        assert needle in text, needle


# ---------------------------------------------------------------- GPU (written, not run here)


def _gpu(backend: str = "triton", name: str = "triton_fp8_w8a8_gemm.py") -> tuple[int, int]:
    from kernel_agent import toolchain

    tc = toolchain.setup()
    if (why := selftest.example_skip(name, tc, backend)) is not None or tc.gpu is None:
        pytest.skip(f"{name}: {why or 'no GPU'}")
    return tuple(tc.gpu.capability)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "name", [*selftest.CUBLASLT_EXAMPLES, *selftest.PRODUCER_EXAMPLES, *selftest.FP8_KV_EXAMPLES]
)
def test_toolkit_example_on_the_evaluator(name, tmp_path):
    _gpu("cuda" if name in selftest.CUBLASLT_EXAMPLES else "triton", name)
    cuda = name in selftest.CUBLASLT_EXAMPLES
    assert selftest.smoke_fp8_toolkit(tmp_path, verbose=True, cuda=cuda, triton=not cuda)


@pytest.mark.gpu
def test_w8a8_dot_scaled_is_bit_identical_to_dot():
    capability = _gpu()
    w8a8 = _example("triton_fp8_w8a8_gemm.py")
    if not w8a8.block_scale_capable(capability):
        pytest.skip("no block-scaled MMA on this GPU")
    torch.manual_seed(0)
    lin = torch.nn.Linear(1024, 8192, bias=False).cuda().to(torch.bfloat16)
    x = torch.randn(352, 1024, device="cuda", dtype=torch.bfloat16)
    scaled, plain = w8a8.build(lin, scaled=1), w8a8.build(lin, scaled=0)
    with torch.no_grad():
        assert torch.equal(scaled(x), plain(x))
    from kernel_agent.kernels.quant import fp8_w8a8_linear

    ref = fp8_w8a8_linear(x, scaled.weight_fp8, scaled.weight_scale)
    assert torch.equal(scaled(x), ref)  # as the tl.dot version: bit-identical to _scaled_mm


@pytest.mark.gpu
def test_cublaslt_helper_against_the_reference_math():
    capability = _gpu("cuda", "cuda_cublaslt_fp8.py")
    from kernel_agent.kernels.quant import fp8_w8a8_linear

    lt = _example("cuda_cublaslt_fp8.py")
    torch.manual_seed(0)
    lins = [torch.nn.Linear(1024, n).cuda().to(torch.bfloat16) for n in (2560, 1024)]
    x = torch.randn(2, 176, 1024, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for lin in lins:
            for splits in (0, 1, 2):
                mod = lt.build(lin, splits=splits)
                ref = fp8_w8a8_linear(x, mod.weight_fp8, mod.weight_scale, lin.bias)
                rel = float((mod(x).float() - ref.float()).norm() / ref.float().norm())
                assert rel < 5e-3, (lin.out_features, splits, rel)  # one more bf16 rounding
        group = lt.Fp8LtGroup(lins)
        for got, lin in zip(group(x), lins, strict=True):
            assert torch.equal(got, lt.build(lin, splits=0)(x))
        info = lt.plan_info(352, 2560, 1024)
        assert info["algorithms"] >= 1 and info["us"] > 0
        if capability >= (10, 0):
            mod = lt.build(lins[0], mxfp8=1)
            q, scales = quant.quantize_mxfp8(lins[0].weight)
            ref = quant.mxfp8_linear(x, q, scales, lins[0].bias)
            rel = float((mod(x).float() - ref.float()).norm() / ref.float().norm())
            assert rel < 5e-3, rel
            codes, act = lt.quantize_activations(x)  # the CUDA kernel, read back
            ref_codes, ref_act = quant.quantize_mxfp8(x)
            assert torch.equal(act.view(torch.uint8), ref_act.view(torch.uint8))
            assert (codes.float() == ref_codes.float()).float().mean() > 0.999
        # graph capture after one eager call per shape
        mod = lt.build(lins[1])
        eager = mod(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = mod(x)
        graph.replay()
        assert torch.equal(captured, eager)


@pytest.mark.gpu
def test_producers_against_the_reference_math():
    _gpu(name="triton_fp8_producers.py")
    from kernel_agent.kernels.quant import quantize_fp8_activations

    prod = _example("triton_fp8_producers.py")
    torch.manual_seed(0)
    x = torch.randn(353, 1024, device="cuda", dtype=torch.bfloat16) * 3
    w = (torch.randn(1024, device="cuda") * 0.1 + 1).to(torch.bfloat16)
    norm = selftest.RMSNorm(1024, eps=1e-6).cuda().to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(w)
        h = norm(x)
    q, s = prod.rmsnorm_fp8(x, w, 1e-6)
    rq, rs = quantize_fp8_activations(h)
    assert torch.allclose(s, rs, rtol=1e-2) and (q.float() == rq.float()).float().mean() > 0.99
    q, s = prod.rmsnorm_fp8(x, w, 1e-6, mx=True)
    rq, rs = quant.quantize_mxfp8(h)
    rs = rs.view(torch.uint8)
    assert (s.int() - rs.int()).abs().max() <= 1 and (q.float() == rq.float()).float().mean() > 0.99
    _, blocked = prod.rmsnorm_fp8(x, w, 1e-6, mx=True, blocked=True)
    assert torch.equal(blocked, quant.swizzle_mx_scales(s).view(torch.uint8))
    gu = torch.randn(353, 2 * 4096, device="cuda", dtype=torch.bfloat16)
    g, u = gu.chunk(2, dim=-1)
    q, s = prod.silu_mul_fp8(gu)
    rq, rs = quantize_fp8_activations(torch.nn.functional.silu(g) * u)
    assert torch.allclose(s, rs, rtol=1e-2) and (q.float() == rq.float()).float().mean() > 0.99


@pytest.mark.gpu
def test_fp8_kv_decode_against_the_reference_math():
    _gpu(name="triton_fp8_kv_decode.py")
    kv = _example("triton_fp8_kv_decode.py")
    torch.manual_seed(0)
    shapes = ((4, 16, 2, 4096, 128, None), (3, 8, 8, 77, 64, [77, 1, 40]))
    for b, hq, hkv, length, d, lengths in shapes:
        q = torch.randn(b, hq, 1, d, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(b, hkv, length, d, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(b, hkv, length, d, device="cuda", dtype=torch.bfloat16)
        kc, ks = kv.quantize_kv(k)
        ref_kc, ref_ks = kv_quant.quantize_fp8_kv(k)
        assert torch.allclose(ks, ref_ks) and (kc.float() == ref_kc.float()).float().mean() > 0.995
        vc, vs = kv.quantize_kv(v)
        lens = torch.tensor(lengths, device="cuda") if lengths else None
        ref = kv_quant.fp8_kv_attention(q, kc, ks, vc, vs, lengths=lens)
        for splits in (0, 1, 3):
            out = kv.fp8_kv_decode(q, kc, ks, vc, vs, lengths=lens, splits=splits)
            rel = float((out.float() - ref.float()).norm() / ref.float().norm())
            assert out.shape == q.shape and rel < 1e-2 and math.isfinite(rel), (splits, rel)
