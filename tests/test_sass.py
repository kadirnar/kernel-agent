"""SASS opcode census (#230): fixture dumps of tiny kernels compiled on the CPU for sm_75,
sm_80, sm_86, sm_89, sm_90a, sm_100a and sm_120a (tests/fixtures/sass, make_fixtures.py)
give the expected census; the opcode tables name only opcodes these cubins contain; the
cubins of a process are collected and disassembled without a GPU."""

from __future__ import annotations

import re
import subprocess
import types
from pathlib import Path
from typing import Any

import pytest

from kernel_agent.kernels import sass

FIXTURES = Path(__file__).parent / "fixtures" / "sass"
ARCHS = ("sm_75", "sm_80", "sm_86", "sm_89", "sm_90a", "sm_100a", "sm_120a")


def census_of(name: str) -> dict[str, dict[str, Any]]:
    """The census rows of a fixture dump, by kernel."""
    rows = [sass.summarise(k) for k in sass.parse((FIXTURES / name).read_text())]
    return {r["kernel"]: r for r in rows}


# ------------------------------------------------------------------ parsing


RAW = """
Fatbin elf code:
================
arch = sm_120
code version = [1,8]
host = linux
compile_size = 64bit

	code for sm_120
		Function : _Z12rmsnorm_bf16PK13__nv_bfloat16PS_if
	.headerflags	@"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM120"
        /*0000*/                   LDC R1, c[0x0][0x37c] ;              /* 0x0000df00ff017b82 */
                                                                        /* 0x000fe20000000800 */
        /*0010*/              @!P0 LDG.E.128 R4, desc[UR4][R2.64] ;     /* 0x0000000402047981 */
        /*0020*/              @!UPT UIADD3 URZ, URZ, URZ, URZ ;         /* 0x0000003f3f3ff290 */
        /*0030*/                   HMMA.16816.F32.BF16 R8, R4, R6, RZ ; /* 0x000000060408723c */
        /*0040*/                   NOP ;                                /* 0x0000000000007918 */
"""
NVDISASM = """
	.text._gemm_kernel:
        /*0000*/                   LDC R1, c[0x0][0x37c] ;
        /*0010*/                   QMMA.SF.16832.F32.E4M3.E4M3.E8 R12, R4, R8, RZ, R0, R0, URZ ;
"""


def test_cuobjdump_output_with_encodings_and_predicates_parses():
    (kernel,) = sass.parse(RAW)
    assert kernel.symbol == "_Z12rmsnorm_bf16PK13__nv_bfloat16PS_if"
    assert kernel.arch == "sm_120"
    assert kernel.ops == {
        "LDC": 1,
        "LDG.E.128": 1,
        "UIADD3": 1,
        "HMMA.16816.F32.BF16": 1,
        "NOP": 1,
    }
    row = sass.summarise(kernel)
    assert row["kernel"] == "rmsnorm_bf16" and row["symbol"].startswith("_Z12")
    assert row["instructions"] == 4  # NOP is padding
    assert row["tensor"] == {"HMMA.16816.F32.BF16": 1}
    assert row["global_load_bits"] == {"128": 1}


def test_nvdisasm_sections_parse():
    (kernel,) = sass.parse(NVDISASM)
    assert kernel.symbol == "_gemm_kernel"
    assert sass.summarise(kernel)["tensor"] == {"QMMA.SF.16832.F32.E4M3.E4M3.E8": 1}


def test_load_widths_names_and_matching():
    assert sass.load_bits("LDG.E.128.CONSTANT") == 128
    assert sass.load_bits("LDG.E.64") == 64
    assert sass.load_bits("LDG.E.U16") == 16
    assert sass.load_bits("LDG.E.LTC128B.U8") == 8  # an L2 hint is no width
    assert sass.load_bits("LDG.E") == 32
    assert sass.short_name("_ZN2ns6kernelEv") == "ns::kernel"
    assert sass.short_name("_Z6kernelPf") == "kernel"
    assert sass.short_name("_gemm_kernel") == "_gemm_kernel"
    assert sass.matches("rmsnorm_bf16", "void rmsnorm_bf16<128>(const float *, float *)")
    assert sass.matches("ns::kernel", "void ns::kernel(float *)")
    assert sass.matches("triton_poi_fused_add_0", "triton_poi_fused_add_0")
    assert not sass.matches("_gemm_kernel", "_quant_kernel")
    assert sass.fp8_unpacks("F2FP.F16.E4M3.UNPACK_B")
    assert not sass.fp8_unpacks("F2FP.BF16.F32.PACK_AB")


# ------------------------------------------------------------------ the fixtures per arch


def test_every_tabled_opcode_occurs_in_a_compiled_fixture():
    """sass.py's tables name only opcodes found in the cubins of tests/fixtures/sass."""
    found = set()
    for path in FIXTURES.glob("*.sass"):
        for kernel in sass.parse(path.read_text()):
            found |= {sass.base(op) for op in kernel.ops}
    assert set(sass.CATEGORIES) <= found, set(sass.CATEGORIES) - found
    assert set(sass.TENSOR_OPS) <= found


def test_fixtures_were_compiled_for_their_arch():
    for arch in ARCHS:
        text = (FIXTURES / f"{arch}.sass").read_text()
        assert text.startswith("// nvcc ")
        assert {k.arch for k in sass.parse(text)} == {arch}


@pytest.mark.parametrize("arch", ARCHS)
def test_memory_barriers_shuffles_and_local_memory_on_every_arch(arch):
    rows = census_of(f"{arch}.sass")
    ldst = rows["ka_ldst"]["categories"]
    assert rows["ka_ldst"]["global_load_bits"] == {"32": 1, "128": 1}
    for cat in ("global_store", "shared_load", "shared_store", "barrier", "shuffle", "atomic"):
        assert ldst.get(cat), (arch, cat)
    assert rows["ka_ldst"]["categories"]["fp32_math"] >= 1
    assert rows["ka_half2"]["categories"]["fp16_math"] >= 1
    assert set(rows["ka_local"]["local"]) == {"LDL", "STL"}
    spill = rows["ka_spill"]["local"]  # 48 live values under __maxnreg__(32)
    assert spill["LDL"] > 40 and spill["STL"] > 40
    if arch == "sm_75":  # Turing: m16n8k8 fp16 and m8n8k16 s8 only, no cp.async
        turing = rows["ka_mma_turing"]
        assert turing["tensor"] == {"HMMA.1688.F32": 1, "IMMA.8816.S8.S8": 1}
        assert "ka_mma_bf16" not in rows and "ka_mma_s8" not in rows
        return
    bf16 = rows["ka_mma_bf16"]
    assert bf16["tensor"] == {"HMMA.16816.F32.BF16": 1}
    assert bf16["categories"]["cp_async"] == 2  # LDGSTS + LDGDEPBAR
    assert bf16["top"]["LDSM"] == 2
    assert rows["ka_mma_s8"]["tensor"] == {"IMMA.16832.S8.S8": 1}


def test_fp8_mma_sync_per_arch():
    for arch in ("sm_75", "sm_80", "sm_86"):  # no e4m3 mma.sync before sm_89
        assert "ka_mma_e4m3" not in census_of(f"{arch}.sass"), arch
    for arch in ("sm_89", "sm_120a"):
        row = census_of(f"{arch}.sass")["ka_mma_e4m3"]
        assert row["tensor"] == {"QMMA.16832.F32.E4M3.E4M3": 1}, arch
        assert "fp8_unpacks" not in row
    for arch in ("sm_90a", "sm_100a"):  # emulated: unpacked to fp16, HMMA
        row = census_of(f"{arch}.sass")["ka_mma_e4m3"]
        assert row["tensor"] == {"HMMA.16816.F32": 2}, arch
        assert row["fp8_unpacks"] == 12


def test_hopper_wgmma_and_tma():
    rows = census_of("sm_90a.sass")
    assert rows["ka_wgmma"]["tensor"] == {
        "HGMMA.64x8x16.F32.BF16": 1,
        "QGMMA.64x8x32.F32.E4M3.E4M3": 1,
        "IGMMA.64x8x32.S8.S8": 1,
    }
    tma = rows["ka_tma"]
    assert tma["categories"]["tma"] >= 3  # UTMALDG, UBLKCP, UTMASTG (+ its commit)
    assert tma["top"]["SYNCS"] >= 3  # mbarrier init / arrive / wait
    assert rows["ka_stmatrix"]["categories"]["shared_store"] == 1  # STSM


def test_datacenter_blackwell_tcgen05():
    row = census_of("sm_100a.sass")["ka_tcgen05"]
    assert row["tensor"] == {"UTCQMMA": 2, "UTCHMMA": 1, "UTCIMMA": 1, "UTCOMMA.4X": 1}
    assert row["categories"]["tensor_memory"] >= 4  # LDTM, STTM, alloc / dealloc


def test_geforce_blackwell_block_scaled_mma():
    row = census_of("sm_120a.sass")["ka_mma_block_scaled"]
    assert row["tensor"] == {
        "QMMA.SF.16832.F32.E4M3.E4M3.E8": 1,
        "OMMA.SF.16864.F32.E2M1.E2M1.E8": 1,
    }


def test_triton_fp8_gemm_tl_dot_against_tl_dot_scaled_on_sm120():
    """The FP8 example's GEMM compiled on the CPU for sm_120: tl.dot issues the half-rate
    QMMA.F32, tl.dot_scaled the block-scaled QMMA.SF (docs/RESEARCH-TRITON.md §1)."""
    dot = census_of("triton_fp8_dot_sm120a.sass")["_gemm_kernel"]
    scaled = census_of("triton_fp8_dot_scaled_sm120a.sass")["_gemm_kernel"]
    assert set(dot["tensor"]) == {"QMMA.16832.F32.E4M3.E4M3"}
    assert set(scaled["tensor"]) == {"QMMA.SF.16832.F32.E4M3.E4M3.E8"}
    for row in (dot, scaled):
        assert row["arch"] == "sm_120a" and row["categories"]["cp_async"] > 0
        assert "local" not in row["categories"]
    assert sass.line(scaled).startswith("_gemm_kernel (sm_120a): QMMA.SF.16832")


# ------------------------------------------------------------------ the cubins of a process


class _Meta:
    name = "matmul_kernel"


class CompiledKernel:
    """Triton's ``CompiledKernel`` as the census sees it."""

    __module__ = "triton.compiler.compiler"

    def __init__(self, cubin: bytes, name: str = "matmul_kernel") -> None:
        self.asm = {"cubin": cubin, "ptx": "..."}
        self.name = name
        self.metadata = _Meta()


class ObjectCode:
    """``cuda.core``'s ``ObjectCode`` as the census sees it."""

    __module__ = "cuda.core._module"

    def __init__(self, code: bytes, code_type: str = "cubin") -> None:
        self.code = code
        self.code_type = code_type
        self.name = "rmsnorm"


class JitCompiledFunction:
    """CuTe DSL's compiled function as the census sees it (its subclasses too)."""

    __module__ = "cutlass.base_dsl.jit_executor"
    function_name = "rmsnorm_kernel"
    __cubin__ = b"\x7fELF-cute"


class TVMFFIJitCompiledFunction(JitCompiledFunction):
    __module__ = "cutlass.cutlass_dsl.tvm_ffi_provider"


def test_binaries_come_from_the_collector_once_each(monkeypatch, tmp_path):
    so = tmp_path / "ext.so"
    monkeypatch.setattr(sass, "_extension_files", lambda: [so])
    keep = [
        CompiledKernel(b"\x7fELF-triton-a"),
        CompiledKernel(b"\x7fELF-triton-a"),  # the same cubin twice: once
        ObjectCode(b"\x7fELF-nvrtc"),
        ObjectCode(b"ptx text", code_type="ptx"),  # PTX: no SASS to read
        TVMFFIJitCompiledFunction(),
    ]
    found = sass.binaries({"k": object()})
    data = (b"\x7fELF-triton-a", b"\x7fELF-nvrtc", b"\x7fELF-cute")
    mine = [b for b in found if b.data in data or b.path]
    assert sorted((b.source, b.name) for b in mine) == [
        ("cuda", "ext.so"),
        ("cute", "rmsnorm_kernel"),
        ("nvrtc", "rmsnorm"),
        ("triton", "matmul_kernel"),
    ]
    assert keep  # alive until here: the collector tracks them


def test_shared_objects_built_here_are_read_and_shipped_ones_are_not():
    import sys

    home = "/home/u/.cache"
    assert sass._built_here(f"{home}/torch_extensions/py312_cu130/ka_rms/ka_rms.so")
    assert sass._built_here(f"{home}/kernel-agent/native/0123/build/engine.so")
    assert sass._built_here("/home/u/.tilelang/cache/abc/kernel_lib.so")
    assert not sass._built_here(f"{sys.prefix}/lib/python3.12/site-packages/torch/lib/x.so")
    assert not sass._built_here("/usr/lib/libcuda.so.1")
    assert not sass._built_here("/opt/cuda/lib64/libcublas.so.13")
    assert not sass._built_here(f"{home}/.triton/cache/ab/cuda_utils.so")
    assert not sass._built_here("/srv/env/lib/python3.12/dist-packages/triton/x.so")


def _fake_disassembler(texts: dict[bytes, str]) -> Any:
    def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert cmd[1] == "-sass"
        data = Path(cmd[2]).read_bytes()
        return subprocess.CompletedProcess(cmd, 0, texts.get(data, ""), "")

    return run


def test_census_orders_kernels_that_ran_first_and_merges_variants(monkeypatch):
    spill = (FIXTURES / "sm_120a.sass").read_text()
    dot = (FIXTURES / "triton_fp8_dot_sm120a.sass").read_text()
    scaled = (FIXTURES / "triton_fp8_dot_scaled_sm120a.sass").read_text()
    monkeypatch.setattr(sass, "find_cuobjdump", lambda: "/bin/cuobjdump")
    monkeypatch.setattr(
        sass,
        "binaries",
        lambda namespace=None: [
            sass.Binary("cuda", "ext", b"cubin-1"),
            sass.Binary("triton", "_gemm_kernel", b"cubin-2"),
            sass.Binary("triton", "_gemm_kernel", b"cubin-3"),  # an autotuning variant
        ],
    )
    run = _fake_disassembler({b"cubin-1": spill, b"cubin-2": dot, b"cubin-3": scaled})
    gpu = {"arch": "sm_120", "capability": [12, 0], "mma_tflops": {"e4m3_f32": 206.0}}
    out = sass.census(ran=["_gemm_kernel", "void ka_spill(float*, int)"], gpu=gpu, run=run)
    assert out["status"] == "ok" and out["gpu"] == gpu
    names = [k["kernel"] for k in out["kernels"]]
    assert names[:2] == ["ka_spill", "_gemm_kernel"]  # those that ran, larger first
    assert "ka_tma" in names[2:]
    gemm = out["kernels"][1]
    assert gemm["variants"] == 2 and gemm["source"] == "triton"
    assert set(gemm["tensor"]) == {
        "QMMA.16832.F32.E4M3.E4M3",
        "QMMA.SF.16832.F32.E4M3.E4M3.E8",
    }
    assert len(out["kernels"]) <= sass.MAX_KERNELS
    assert "missing" not in out  # every kernel that ran has its SASS
    library = ["_gemm_kernel", "void at::native::vectorized_elementwise_kernel<4>(int)"]
    out = sass.census(ran=library, gpu=gpu, run=run)
    assert out["missing"] == library[1:] and "library kernels" in out["note"]


def test_census_without_cuobjdump_says_how_to_get_one(monkeypatch):
    monkeypatch.setattr(sass, "find_cuobjdump", lambda: None)
    out = sass.census(gpu={"arch": "sm_80"})
    assert out["status"] == "unavailable" and "cuobjdump" in out["reason"]
    assert out["gpu"] == {"arch": "sm_80"}
    assert "unavailable" in sass.describe()


def test_census_never_raises(monkeypatch):
    monkeypatch.setattr(sass, "find_cuobjdump", lambda: "/bin/cuobjdump")

    def broken(namespace=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(sass, "binaries", broken)
    out = sass.census()
    assert out["status"] == "error" and "boom" in out["reason"]


def test_census_of_a_triton_kernel_compiled_on_the_cpu():
    """The real path without a GPU: Triton compiles for sm_120, cuobjdump disassembles."""
    triton = pytest.importorskip("triton")
    if sass.find_cuobjdump() is None:
        pytest.skip("no cuobjdump")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    source = ASTSource(
        _add_kernel(),
        {"x": "*fp32", "y": "*fp32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": 256},
        attrs={(0,): [["tt.divisibility", 16]], (1,): [["tt.divisibility", 16]]},
    )
    compiled = triton.compile(source, target=GPUTarget("cuda", 120, 32))
    module = types.SimpleNamespace(kernel=compiled)
    out = sass.census(module, gpu={"arch": "sm_120", "capability": [12, 0]})
    assert out["status"] == "ok", out
    row = next(k for k in out["kernels"] if k["kernel"] == "ka_add_kernel")
    assert row["arch"].startswith("sm_120") and row["source"] == "triton"
    assert re.match(r"ka_add_kernel \(sm_120a?\): no tensor-core MMA; LDG", sass.line(row))


SPILLING_RMSNORM = """
import torch
import triton
import triton.language as tl


@triton.jit
def _rms_rows_kernel(x_ptr, w_ptr, y_ptr, n_rows, n_cols, eps, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * 8 + tl.arange(0, 8)  # 8 rows of 2048 at one warp: spills
    cols = tl.arange(0, BLOCK)
    mask = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)
    offs = rows[:, None] * n_cols + cols[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x, axis=1) / n_cols + eps)[:, None]
    w = tl.load(w_ptr + cols, mask=cols < n_cols, other=0.0)
    tl.store(y_ptr + offs, y.to(w.dtype) * w[None, :], mask=mask)


class Spilling(torch.nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.weight, self.eps = reference.weight, float(reference.variance_epsilon)

    def forward(self, h):
        x = h.reshape(-1, h.shape[-1]).contiguous()
        y = torch.empty_like(x)
        grid = (triton.cdiv(x.shape[0], 8),)
        _rms_rows_kernel[grid](x, self.weight, y, x.shape[0], x.shape[1], self.eps,
                               BLOCK=triton.next_power_of_2(x.shape[1]), num_warps=1)
        return y.view(h.shape)


def build(reference):
    return Spilling(reference)
"""


@pytest.mark.gpu
def test_the_census_and_directives_of_evaluations_on_this_gpu(tmp_path):
    """On sm_12x the FP8 example's tl.dot GEMM issues QMMA.F32 and is told about QMMA.SF,
    its tl.dot_scaled GEMM issues QMMA.SF; on any GPU a spilling kernel is flagged."""
    import torch

    from kernel_agent import selftest
    from kernel_agent.agent.prompts import EXAMPLES_DIR
    from kernel_agent.kernels import directives
    from kernel_agent.kernels.evaluate import run_evaluation

    rms = selftest.make_rmsnorm_capture(tmp_path / "rms.pt")
    spilling = tmp_path / "spilling.py"
    spilling.write_text(SPILLING_RMSNORM)
    result = run_evaluation(rms, spilling, profile=True)
    assert result["correct"], result.get("error")
    row = next(k for k in result["sass"]["kernels"] if k["kernel"] == "_rms_rows_kernel")
    assert row["local"]["LDL"] > 0 and row["local"]["STL"] > 0
    found = directives.build(result, result)
    assert found and found[0]["rule"] == "local_memory"
    if torch.cuda.get_device_capability()[0] != 12:
        return  # QMMA.SF is sm_12x's
    lin = selftest.make_linear_capture(
        tmp_path / "lin.pt",
        1024,
        8192,
        [((64, 11), 54), ((32, 11), 0)],
        tier="near-lossless",
        precision="fp8_w8a8",
    )
    example = EXAMPLES_DIR / "triton_fp8_w8a8_gemm.py"
    plain = tmp_path / "fp8_tl_dot.py"
    old = "    scaled: int = -1,\n) -> nn.Module:"
    plain.write_text(example.read_text().replace(old, "    scaled: int = 0,\n) -> nn.Module:"))
    for candidate, want, rules in (
        (example, "QMMA.SF.", []),
        (plain, "QMMA.16832.F32", ["tensor_rate"]),
    ):
        result = run_evaluation(lin, candidate, profile=True)
        assert result["correct"], result.get("error")
        gemm = next(k for k in result["sass"]["kernels"] if k["kernel"] == "_gemm_kernel")
        assert all(op.startswith(want) for op in gemm["tensor"]), gemm
        found = directives.build(result, result)
        assert [d["rule"] for d in found if d.get("kernel") == "_gemm_kernel"] == rules


def _add_kernel() -> Any:
    import triton
    import triton.language as tl

    @triton.jit
    def ka_add_kernel(x, y, BLOCK: tl.constexpr):
        i = tl.arange(0, BLOCK)
        tl.store(y + i, tl.load(x + i) + 1.0)

    return ka_add_kernel
