"""The low-precision ``nn.Linear`` examples take the model's dtype, bf16 or fp16 (#258): every
``build()`` guard accepts ``torch.float16`` (read with ``ast``), the examples declare it
(``DTYPES``, which the selftest reads to add an fp16 capture), their ``ARCHS`` follow the
evidence (CPU compiles for sm_75 / sm_80 / sm_86 / sm_89 / sm_120 and runs through each arch's
PTX on an RTX 5070 Ti) and the skinny GEMMs carry Turing's MMA forms. CPU only."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
import torch

from kernel_agent import gpu_arch, selftest
from kernel_agent.agent.prompts import EXAMPLES_DIR

#: The low-precision examples and the GPUs they declare.
EXAMPLES = {
    "cuda_fp8_gemv.py": "sm_75+",
    "cuda_int8_gemv.py": "sm_75+",
    "cuda_fp4_gemv.py": "sm_75+",
    "cuda_fp8_skinny_gemm.py": "sm_75+",  # Turing: fp16 as two m16n8k8, bf16 falls back
    "cuda_int8_skinny_gemm.py": "sm_75+",  # Turing: four m8n8k16
    "triton_int8_w8a8_gemm.py": "sm_80+",  # Triton's int8 tl.dot does not compile for sm_75
    "triton_fp8_w8a8_gemm.py": "sm_89+",
    "cuda_cublaslt_fp8.py": "sm_89+",
}


def _tree(name: str) -> ast.Module:
    return ast.parse((EXAMPLES_DIR / name).read_text())


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_build_guards_accept_fp16(name):
    tree = _tree(name)
    declared = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "DTYPES"
    ]
    assert len(declared) == 1 and isinstance(declared[0], ast.Tuple), name
    assert {ast.unparse(e) for e in declared[0].elts} == {"torch.bfloat16", "torch.float16"}
    build = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build")
    guards = [
        node
        for node in ast.walk(build)
        if isinstance(node, ast.Compare) and ast.unparse(node.left) == "reference.weight.dtype"
    ]
    # the weight's dtype is checked against DTYPES, the bias against the weight's
    assert [ast.unparse(g) for g in guards] == ["reference.weight.dtype in DTYPES"], name
    assert "reference.bias.dtype == reference.weight.dtype" in ast.unparse(build), name
    # no bf16 pinned anywhere else in the module (forwards, fake ops, empty biases)
    pins = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and ast.unparse(node) == "torch.bfloat16"
    ]
    assert len(pins) == 1, (name, [p.lineno for p in pins])
    assert selftest.example_dtypes(name) == ("bfloat16", "float16")


@pytest.mark.parametrize("name", sorted(n for n in EXAMPLES if n.startswith("cuda_")))
def test_cuda_sources_dispatch_both_dtypes(name):
    source = (EXAMPLES_DIR / name).read_text()
    assert "at::kHalf" in source and "at::kBFloat16" in source
    assert "<__half>" in source and "<bf16>" in source  # both template instantiations


def test_archs_follow_the_evidence():
    for name, spec in EXAMPLES.items():
        assert gpu_arch.example_requirement(EXAMPLES_DIR / name)[0] == spec, name
        assert gpu_arch.supports(spec, (7, 5)) is (spec == "sm_75+"), name
    # correct on sm_75 through Triton's FMA lowering, without tensor cores: documented
    spec, why = gpu_arch.example_requirement(EXAMPLES_DIR / "triton_short_attention.py")
    assert spec == "sm_75+" and "FMA" in why


@pytest.mark.parametrize(
    ("name", "ampere", "turing"),
    [
        (
            "cuda_fp8_skinny_gemm.py",
            "m16n8k16.row.col.f32.f16.f16.f32",
            "m16n8k8.row.col.f32.f16.f16.f32",
        ),
        (
            "cuda_int8_skinny_gemm.py",
            "m16n8k32.row.col.s32.s8.s8.s32",
            "m8n8k16.row.col.s32.s8.s8.s32",
        ),
    ],
)
def test_skinny_gemms_have_turings_mma_forms(name, ampere, turing):
    source = (EXAMPLES_DIR / name).read_text()
    at = source.index(ampere)
    branch = source.rfind("#if __CUDA_ARCH__ >= 800", 0, at)
    other = source.index("#else", at)
    assert 0 <= branch < at < other < source.index(turing, other) < source.index("#endif", other)
    if name == "cuda_fp8_skinny_gemm.py":  # no bf16 MMA before sm_80: the loaded code says so
        assert "ptxVersion >= 80" in source and "bf16_mma()" in source


def test_selftest_adds_an_fp16_capture(monkeypatch, tmp_path):
    from kernel_agent.kernels import evaluate

    captured: list[tuple[str, Any]] = []

    def capture(path: Path, k, n, calls, *, tier=None, precision=None, dtype=torch.bfloat16):
        captured.append((path.name, dtype))
        return path

    def run_evaluation(capture_path, example, quick=False, **_):
        return {"status": "incorrect"} if quick else {"status": "ok", "correct": True}

    monkeypatch.setattr(evaluate, "run_evaluation", run_evaluation)
    examples = {"cuda_fp8_gemv.py": (64, 32, [((1,), 1)]), "triton_mxfp8_gemm.py": (64, 32, [])}
    assert selftest.smoke_fp8(tmp_path, precision="fp8_weights", examples=examples, capture=capture)
    dtypes = [(path, dtype) for path, dtype in captured if path.startswith("cuda_fp8_gemv")]
    assert [d for _, d in dtypes] == [torch.bfloat16] * 2 + [torch.float16] * 2
    assert len({path for path, _ in captured}) == len(captured)  # one file per dtype
    # an example without DTYPES, and a capture without a dtype: bf16 only
    assert [d for p, d in captured if p.startswith("triton_mxfp8")] == [torch.bfloat16] * 2

    def untyped(path: Path, k, n, calls, *, tier=None, precision=None):
        captured.append((path.name, None))
        return path

    captured.clear()
    assert selftest.smoke_fp8(tmp_path, examples=examples, capture=untyped)
    assert len(captured) == 4 and not any(".float16." in p for p, _ in captured)

    captured.clear()
    monkeypatch.setattr(selftest, "make_linear_capture", capture)
    assert selftest.smoke_fp4(tmp_path, examples={"cuda_fp4_gemv.py": (64, 32, [])})
    assert [d for _, d in captured] == [torch.bfloat16] * 2 + [torch.float16] * 2
