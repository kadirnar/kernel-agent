"""Turing (sm_75, #254): what kernel-agent offers on a T4 matches what runs there. Backend
availability follows each backend's targets (CuTe DSL 4.8 has none below sm_80), ``doctor
--smoke`` never runs a refused backend, no policy row or rule names an example whose
``ARCHS`` excludes the GPU, the INT8 W8A8 reference survives a ``torch._int_mm`` error and
the library scout takes fp16 GEMMs to cuBLASLt there. CPU only (faked GPUs, a faked CuTe DSL
probe and evaluator), but the forced ``_int_mm`` fallback on a GPU (``gpu``)."""

from __future__ import annotations

import re

import pytest
import torch

from kernel_agent import backends, cute_dsl, gpu_arch, selftest, toolchain
from kernel_agent.agent.prompts import EXAMPLES_DIR
from kernel_agent.kernels import evaluate, quant
from kernel_agent.libscout import adapters, registry

#: name, capability, memory GB, SMs, L2 MB, smem per block KB (no peaks: CPU only)
GPUS = {
    (7, 5): ("Tesla T4", 14.6, 40, 4.0, 64.0),
    (8, 0): ("NVIDIA A100-SXM4-80GB", 80.0, 108, 40.0, 163.0),
    (8, 6): ("NVIDIA A10", 22.1, 72, 6.0, 99.0),
    (8, 9): ("NVIDIA L4", 22.0, 58, 48.0, 99.0),
    (12, 0): ("NVIDIA GeForce RTX 5070 Ti", 15.5, 70, 48.0, 99.0),
}
#: CuTe DSL 4.8.0's targets (``cutlass.base_dsl.enums.Arch``, read on the CPU)
CUTE_48 = ["sm_80", "sm_86", "sm_87", "sm_89", "sm_90", "sm_90a", "sm_100", "sm_100a"]
CUTE_48 += ["sm_103", "sm_103a", "sm_110", "sm_110a", "sm_120", "sm_120a", "sm_121a"]


@pytest.fixture
def fake_dsl(tmp_path, monkeypatch):
    """CuTe DSL 4.8.0 installed (its probe faked, counted), the targets cache in tmp."""
    calls = []

    def probe():
        calls.append(1)
        return {"version": "4.8.0", "arches": list(CUTE_48)}

    monkeypatch.delenv(cute_dsl.ARCH_ENV, raising=False)
    monkeypatch.setattr(cute_dsl, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cute_dsl, "dsl_version", lambda: "4.8.0")
    monkeypatch.setattr(cute_dsl, "_probe", probe)
    return calls


def fake_setup(monkeypatch, tmp_path, capability: tuple[int, int]) -> toolchain.Toolchain:
    """``toolchain.setup`` (uncached, no env changes) on a faked GPU with every backend's
    package installed and an nvcc 13.0."""
    name, mem, sms, l2, kb = GPUS[capability]
    gpu = toolchain.GPUInfo(name, capability, mem, sms, l2, kb, kb + 1)
    home = tmp_path / "cuda"
    (home / "bin").mkdir(parents=True, exist_ok=True)
    (home / "bin" / "nvcc").write_text("")
    monkeypatch.setenv("CUDA_HOME", str(home))
    monkeypatch.setattr(toolchain, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(toolchain, "gpu_info", lambda: gpu)
    monkeypatch.setattr(toolchain, "_module_available", lambda name: True)
    monkeypatch.setattr(toolchain, "_nvcc_version", lambda nvcc: (13, 0))
    monkeypatch.setattr(toolchain, "_gcc_major", lambda: None)
    return toolchain.setup.__wrapped__(apply_env=False)


# ------------------------------------------------------------------ backend availability


def test_a_t4_toolchain_lists_cute_as_unavailable_with_the_reason(fake_dsl, monkeypatch, tmp_path):
    tc = fake_setup(monkeypatch, tmp_path, (7, 5))
    why = "CuTe DSL 4.8.0 has no sm_75 target; its targets: sm_80 to sm_121"
    assert tc.backends["cute"] is False and tc.unavailable == {"cute": why}
    assert all(tc.backends[b] for b in ("cuda", "triton", "nvrtc", "tilelang"))
    text = tc.summary()
    assert "backends available: triton, cuda, nvrtc, tilelang, helion" in text
    assert f"backends unavailable: cute ({why})" in text
    assert selftest.example_skip("cute_rmsnorm.py", tc, "cute") == (
        f"the cute backend is not available ({why})"
    )
    # sm_80 and the development GPU keep it; the DSL's targets were probed once (cached)
    for cap in ((8, 0), (12, 0)):
        tc = fake_setup(monkeypatch, tmp_path, cap)
        assert tc.backends["cute"] is True and tc.unavailable == {}
    assert len(fake_dsl) == 1


def test_the_targets_are_cached_per_dsl_version(fake_dsl, monkeypatch):
    assert cute_dsl.unsupported((7, 5)) and len(fake_dsl) == 1
    assert cute_dsl.unsupported((8, 6)) is None and len(fake_dsl) == 1  # from the cache
    monkeypatch.setattr(cute_dsl, "dsl_version", lambda: "4.9.0")  # an upgrade probes again
    sm75 = lambda: {"arches": ["sm_75", *CUTE_48]}  # noqa: E731
    assert cute_dsl.unsupported((7, 5), probe=sm75) is None
    assert cute_dsl.unsupported(None) is None  # no GPU: nothing to refuse


def test_a_broken_dsl_or_check_refuses_nothing(fake_dsl, monkeypatch):
    def broken():
        raise ImportError("libcute_dsl_runtime.so: cannot open shared object file")

    monkeypatch.setattr(cute_dsl, "_probe", broken)
    assert cute_dsl.unsupported((7, 5)) is None  # doctor's check reports the import error

    def fails(capability):
        raise RuntimeError("probe crashed")

    monkeypatch.setitem(toolchain.ARCH_SUPPORT, "cute", fails)
    assert toolchain.arch_unsupported({"cute": True}, (7, 5)) == {}


def test_doctor_smoke_never_runs_a_refused_backend(fake_dsl, monkeypatch, tmp_path, capsys):
    tc = fake_setup(monkeypatch, tmp_path, (7, 5))
    monkeypatch.setattr(toolchain, "setup", lambda apply_env=True: tc)
    ran: list[str] = []

    def run_evaluation(capture, example, **kw):
        ran.append(example.name)
        return {"status": "ok", "correct": True, "speedup": 1.0, "cases": [{}]}

    monkeypatch.setattr(evaluate, "run_evaluation", run_evaluation)
    monkeypatch.setattr(selftest, "make_rmsnorm_capture", lambda path: path)
    smoked: list[str] = []
    for name in (
        "smoke_fp8",
        "smoke_fp4",
        "smoke_pdl",
        "smoke_megakernel",
        "smoke_block_scale",
        "smoke_triton_tools",
        "smoke_helion",
        "smoke_fp8_toolkit",
    ):
        monkeypatch.setattr(selftest, name, lambda *a, _n=name, **k: smoked.append(_n) or True)
    assert selftest.smoke_backends(verbose=True)
    assert "cute_rmsnorm.py" not in ran
    assert {"cuda_rmsnorm.py", "triton_rmsnorm.py", "nvrtc_rmsnorm.py"} <= set(ran)
    # nothing sm_80+ (FP8 / FP4 / INT8 / megakernel examples), no CuTe example at all
    assert set(smoked) <= {"smoke_triton_tools", "smoke_helion"}
    out = capsys.readouterr().out
    assert re.search(r"cute_rmsnorm\s+the cute backend is not available \(CuTe DSL 4\.8\.0", out)
    # asking for cute explicitly does not run it either
    ran.clear()
    assert selftest.smoke_backends(["cute"])
    assert ran == []


# ------------------------------------------------------------------ the policy


def _named_unrunnable(text: str, cap: tuple[int, int]) -> list[str]:
    """Bundled examples ``text`` names whose ``ARCHS`` exclude ``cap`` (independent of
    ``backends.unrunnable``)."""
    names = {p.name for p in EXAMPLES_DIR.iterdir()}
    out = []
    for name in sorted(set(re.findall(r"\w+\.py", text)) & names):
        spec = gpu_arch.example_requirement(EXAMPLES_DIR / name)[0]
        if not gpu_arch.supports(spec, cap):
            out.append(f"{name} ({spec})")
    return out


def _spec(cls, signature, precision=None):
    spec = {"module_class": cls, "capture": {"cases": [{"signature": signature, "count": 10}]}}
    return {**spec, "precision": precision} if precision else spec


#: One target per policy class (and the GEMMs inside fused layers at each 8-bit precision).
TARGETS = [
    _spec("Qwen3MLP", "a0[4, 176, 1024]:bfloat16", "fp8_w8a8"),
    _spec("Qwen3MLP", "a0[4, 176, 1024]:bfloat16", "int8_w8a8"),
    _spec("Qwen3MLP", "a0[4, 176, 1024]:bfloat16"),
    _spec("Linear", "a0[80, 1024]:bfloat16", "fp8_w8a8"),
    _spec("LlamaMLP", "a0[16, 1, 2048]:bfloat16", "int8_weights"),
    _spec("GPT2Attention", "a0[32, 11, 1024]:bfloat16"),
    _spec("Conv1d", "a0[16, 64, 240]:float32"),
    _spec("LlamaDecoderLayer", "a0[2, 176, 1024]:bfloat16", "fp8_w8a8"),
    _spec("LlamaDecoderLayer", "a0[2, 176, 1024]:bfloat16", "int8_w8a8"),
    _spec("Qwen3RMSNorm", "a0[1, 1, 1024]:bfloat16"),
    _spec("Mystery", "a0[3]:float32"),
]


@pytest.mark.parametrize("cap", [(7, 5), (8, 0), (8, 6), (8, 9)])
def test_the_policy_names_no_example_the_gpu_cannot_run(cap):
    facts = gpu_arch.Facts(GPUS[cap][0], cap)
    every = ["cuda", "triton", "cute", "tilelang", "nvrtc", "helion"]
    assert {backends.target_class(t) for t in TARGETS} == {c.id for c in backends.POLICY}
    texts = {"policy_text": backends.policy_text(every, facts)}
    texts["policy_text (no list)"] = backends.policy_text(None, facts)
    for target in TARGETS:
        for plan in (every, ["cute", "triton"]):
            texts[f"{target['module_class']}/{plan[0]}"] = backends.engineer_note(
                target, plan, facts
            )
    for row in backends.POLICY:  # every field of every row, also those policy_text hides
        r = backends.policy(row.id, facts)
        texts[row.id] = " ".join((r.first, r.why, r.second, r.never))
    for where, text in texts.items():
        assert _named_unrunnable(text, cap) == [], (cap, where)
    assert backends.unrunnable(" ".join(texts.values()), cap) == []


def test_the_filter_keeps_what_runs_and_drops_only_what_does_not():
    # the development GPU: every named example runs, the rows stay as written
    facts = gpu_arch.Facts("NVIDIA GeForce RTX 5070 Ti", (12, 0))
    for row in backends.POLICY:
        assert backends.policy(row.id, facts) == row
    assert "cute_fp8_blockscaled_gemm.py" in backends.policy_text(None, facts)
    ada = backends.policy_text(None, gpu_arch.Facts("NVIDIA L4", (8, 9)))
    assert "triton_fp8_w8a8_gemm.py" in ada and "triton_int8_w8a8_gemm.py" in ada
    # a parenthesis that names it goes, the clause stays; a clause naming it goes
    t4 = (7, 5)
    text = "CUDA C++ (INT8: `cuda_int8_gemv.py` weight-only); Triton (`cuda_rmsnorm.py`) too"
    assert backends.runnable_text(text, t4) == "CUDA C++; Triton (`cuda_rmsnorm.py`) too"
    text = "CuTe DSL layer from `examples/cute_fp8_decoder_block.py` (one launch); cuBLAS"
    assert backends.runnable_text(text, (8, 6)) == "cuBLAS"
    assert backends.runnable_text(text, (8, 9)) == text
    assert backends.runnable_text("a (b; `cute_fp8_decoder_block.py`); c", t4) == "a; c"
    assert backends.runnable_text(text, None) == text  # unknown GPU: unchanged
    # a first / second backend left empty falls back to the "other" row's
    row = backends.TargetClass("x", "X", "`cuda_int8_gemv.py`", "why", "`cute_rmsnorm.py`")
    got = backends._runnable_row(row, t4)
    assert (got.first, got.second) == (backends.policy("other").first, "`cute_rmsnorm.py`")


def test_turing_rows_use_what_sm75_has():
    facts = gpu_arch.Facts("Tesla T4", (7, 5))
    small = backends.policy("small_m_gemm", facts)
    assert "m16n8k8" in small.first and "fp16" in small.first and small.order[0] == "cuda"
    attention = backends.policy("short_attention", facts)
    assert "WMMA fp16" in attention.first and "memory-efficient" in attention.second
    assert attention.order == ("cuda", "triton")
    gemm = backends.policy("bf16_gemm", facts)
    assert "fp16 GEMMs get the tensor cores" in gemm.first and "cute" not in gemm.order
    layer = backends.policy("decoder_layer", facts)
    assert "cute" not in layer.order and "fp16" in layer.first
    int8 = backends.policy("int8_gemm", facts)
    assert "does not compile below sm_80" in int8.why and "int8 tiles" in int8.never
    text = backends.policy_text(["cuda", "triton"], facts)
    assert "does not compile on int8 (Triton 3.8)" in text
    assert "CuTe DSL 4.8 has no target below sm_80" in text
    assert "runs on FMA units below sm_80: keep" not in text  # the old, wrong int8 claim
    note = gpu_arch.precision_note("int8_w8a8", (7, 5))
    assert "does not compile below sm_80" in str(note)
    # Ampere keeps the sm_120 rows for these classes (none names an sm_80+ gap there)
    ampere = gpu_arch.Facts("NVIDIA A10", (8, 6))
    assert backends.policy("short_attention", ampere) == backends.policy("short_attention")


# ------------------------------------------------------------------ the INT8 reference


def test_int8_matmul_survives_an_int_mm_error(monkeypatch):
    """Where ``torch._int_mm`` raises (cuBLASLt IMMA with regular layouts on Turing is
    unverified), the exact fp32 path gives the same int32 products; tried once per device
    and layout, the error recorded."""
    calls = []

    def int_mm(a, b):
        calls.append((tuple(a.shape), tuple(b.shape)))
        raise RuntimeError("CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling cublasLtMatmul")

    monkeypatch.setattr(quant, "INT_MM_FAILED", {})
    monkeypatch.setattr(quant, "_int_mm_ok", lambda a, b: True)
    monkeypatch.setattr(torch, "_int_mm", int_mm)
    gen = torch.Generator().manual_seed(0)
    for m, k, n in ((5, 3000, 24), (40, 2048, 16)):
        a = torch.randint(-127, 128, (m, k), generator=gen, dtype=torch.int8)
        b = torch.randint(-127, 128, (n, k), generator=gen, dtype=torch.int8)
        got = quant.int8_matmul(a, b)
        assert got.dtype == torch.int32 and torch.equal(got.long(), a.long() @ b.long().T)
    assert len(calls) == 1
    assert list(quant.INT_MM_FAILED) == [("cpu", "column-major")]
    assert "CUBLAS_STATUS_NOT_SUPPORTED" in quant.INT_MM_FAILED[("cpu", "column-major")]
    # the W8A8 reference through it: the same integers, scales and rounding
    x = torch.randn(7, 2048, generator=gen).to(torch.bfloat16)
    q, s = quant.quantize_int8((torch.randn(16, 2048, generator=gen) * 0.02).bfloat16())
    got = quant.int8_w8a8_linear(x, q, s)
    monkeypatch.setattr(quant, "_int_mm_ok", lambda a, b: False)
    assert torch.equal(got, quant.int8_w8a8_linear(x, q, s)) and len(calls) == 1


@pytest.mark.gpu
def test_the_int_mm_fallback_equals_int_mm_on_the_gpu(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    gen = torch.Generator().manual_seed(0)
    shapes = ((1, 2048, 1024), (33, 3000, 40), (352, 4096, 1024))
    pairs = []
    for m, k, n in shapes:
        a = torch.randint(-127, 128, (m, k), generator=gen, dtype=torch.int8).cuda()
        b = torch.randint(-127, 128, (n, k), generator=gen, dtype=torch.int8).cuda()
        pairs.append((a, b, quant.int8_matmul(a, b)))  # torch._int_mm

    def int_mm(a, b):
        raise RuntimeError("forced: no IMMA kernel")

    monkeypatch.setattr(quant, "INT_MM_FAILED", {})
    monkeypatch.setattr(torch, "_int_mm", int_mm)
    for a, b, want in pairs:
        assert torch.equal(quant.int8_matmul(a, b), want)
    assert list(quant.INT_MM_FAILED) == [(str(pairs[0][0].device), "column-major")]


# ------------------------------------------------------------------ the library scout


VOLTA = "needs sm_75+ (fp16 tensor cores); this GPU is sm_70"


@pytest.mark.parametrize(
    ("cap", "fp16", "bf16"),
    [
        (
            (7, 5),
            None,
            "no call site it takes: linear in bfloat16 needs sm_80+ (bf16 tensor cores); "
            "this GPU is sm_75",
        ),
        ((8, 6), None, None),
        ((12, 0), None, None),
        ((7, 0), VOLTA, VOLTA),
    ],
)
def test_cublaslt_takes_fp16_gemms_from_sm75_and_bf16_from_sm80(cap, fp16, bf16):
    lt = adapters.BY_NAME["cublaslt"]
    probes = {"cublaslt": registry.Availability("cublaslt", "torch", "2.14")}

    def decide(dtype):
        found = {"linear": {"sites": [{"dtype": dtype, "n": 1024, "k": 2048}]}}
        (decision,) = registry.applicable(
            [lt], found, capability=cap, precision=None, available=probes, backends={"cuda": True}
        )
        return decision

    half, brain = decide("float16"), decide("bfloat16")
    assert (half.reason, brain.reason) == (fp16, bf16)
    assert half.run is (fp16 is None) and brain.run is (bf16 is None)
