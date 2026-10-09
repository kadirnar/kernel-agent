"""The NVRTC helper and line info (#230): ``toolchain.nvrtc_kernels`` compiles for this GPU
and keeps each kernel's cubin for the SASS census while the kernel lives; in the
evaluator's ``--ncu-mode`` (``toolchain.lineinfo_env``) CUDA C++ builds carry line info:
``-lineinfo`` for nvcc, ``load_inline`` builds in their own directory, NVRTC programs with
``lineinfo`` named by a file holding their source. No GPU: a fake ``cuda.core`` for the
kernels, the real NVRTC compiling to PTX."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from kernel_agent import toolchain
from kernel_agent.kernels import ncu, sass

SRC = r"""
template <int N> __global__ void scale(float* x) { x[threadIdx.x] *= N; }
extern "C" __global__ void add_one(float* x) {
  x[threadIdx.x] += 1.f;
}
"""


@pytest.fixture
def plain(monkeypatch, tmp_path):
    """A process outside ``--ncu-mode`` with fixed include directories and cache."""
    monkeypatch.setattr(toolchain, "cuda_include_dirs", lambda: ["/cuda/include"])
    monkeypatch.setattr(toolchain, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv(toolchain.LINEINFO_ENV, "0")


def test_options_take_this_gpus_target_and_the_callers_options(plain):
    opts, kind = toolchain.nvrtc_options(SRC, (8, 6))
    assert (opts, kind) == (
        {"arch": "sm_86", "std": "c++17", "include_path": ["/cuda/include"]},
        "cubin",
    )
    opts, kind = toolchain.nvrtc_options(
        SRC, (12, 0), arch="sm_120a", include_path="/mine", max_register_count=64
    )
    assert kind == "cubin" and opts["arch"] == "sm_120a" and opts["max_register_count"] == 64
    assert opts["include_path"] == ["/mine", "/cuda/include"]
    assert toolchain.nvrtc_options(SRC, arch="compute_86")[1] == "ptx"  # JIT by the driver
    assert "lineinfo" not in opts and "name" not in opts


def test_options_carry_line_info_in_ncu_mode(plain, monkeypatch, tmp_path):
    monkeypatch.setenv(toolchain.LINEINFO_ENV, "1")
    opts, _ = toolchain.nvrtc_options(SRC, (8, 6))
    assert opts["lineinfo"] is True
    source = Path(opts["name"])  # the line table's file: Nsight Compute reads lines there
    assert source.parent == tmp_path / "cache" / "nvrtc-src" and source.read_text() == SRC
    assert toolchain.nvrtc_options(SRC, (8, 6))[0]["name"] == str(source)  # one per source
    assert toolchain.nvrtc_options(SRC, (8, 6), lineinfo=False)[0]["lineinfo"] is False


class _Kernel:
    """A ``cuda.core`` ``Kernel``: weakly referable, its cubin not exposed."""

    def __init__(self, name: str) -> None:
        self.name = name


class _ObjectCode:
    def __init__(self, code: bytes, code_type: str) -> None:
        self.code, self.code_type = code, code_type

    def get_kernel(self, name: str) -> _Kernel:
        return _Kernel(name)


def _fake_cuda_core(monkeypatch) -> list[dict[str, Any]]:
    """``cuda.core`` with a ``Program`` that records each compile."""
    compiles: list[dict[str, Any]] = []

    class Program:
        def __init__(self, source: str, code_type: str, options: dict[str, Any]) -> None:
            self.source, self.options = source, options
            assert code_type == "c++"

        def compile(self, kind: str, name_expressions: Any = ()) -> _ObjectCode:
            compiles.append(
                {"options": self.options, "kind": kind, "names": list(name_expressions)}
            )
            body = b"ptx text" if kind == "ptx" else b"\x7fELF" + self.source.encode()[:16]
            return _ObjectCode(body, kind)

    module = types.ModuleType("cuda.core")
    module.Program = Program  # type: ignore[attr-defined]
    module.ProgramOptions = lambda **kwargs: dict(kwargs)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cuda.core", module)
    return compiles


def test_kernels_keep_their_cubin_for_the_census_while_they_live(plain, monkeypatch):
    compiles = _fake_cuda_core(monkeypatch)
    kernels = toolchain.nvrtc_kernels(SRC, ["add_one", "scale<4>"], capability=(8, 6))
    assert list(kernels) == ["add_one", "scale<4>"]
    (made,) = compiles
    assert made["kind"] == "cubin" and made["options"]["arch"] == "sm_86"
    assert made["names"] == ["scale<4>"]  # a template instantiation is a name expression
    cubin = b"\x7fELF" + SRC.encode()[:16]
    held = sorted(b.name for b in sass.kept() if b.data == cubin)
    assert held == ["add_one", "scale<4>"]
    one = toolchain.nvrtc_kernel(SRC, "add_one", capability=(8, 6))
    assert isinstance(one, _Kernel) and one.name == "add_one"
    del kernels, one  # dropped kernels drop their cubins: no leak across candidates
    assert not [b for b in sass.kept() if b.data == cubin]
    # PTX (an emulated older GPU's target): no cubin, nothing kept
    toolchain.nvrtc_kernels(SRC, ["add_one"], arch="compute_86")
    assert compiles[-1]["kind"] == "ptx"
    assert not [b for b in sass.kept() if b.data == b"ptx text"]


def test_kernels_compile_with_line_info_in_ncu_mode(plain, monkeypatch):
    compiles = _fake_cuda_core(monkeypatch)
    monkeypatch.setenv(toolchain.LINEINFO_ENV, "1")
    toolchain.nvrtc_kernel(SRC, "add_one", capability=(8, 6))
    options = compiles[-1]["options"]
    assert options["lineinfo"] is True and Path(options["name"]).read_text() == SRC


def test_line_info_reaches_real_nvrtc_output(plain, monkeypatch):
    """The real NVRTC (PTX: no GPU): the line table names the source file and its lines."""
    core = pytest.importorskip("cuda.core")
    monkeypatch.setenv(toolchain.LINEINFO_ENV, "1")
    opts, kind = toolchain.nvrtc_options(SRC, (8, 6), arch="compute_86")
    opts["include_path"] = []  # the fixture's directory does not exist
    program = core.Program(SRC, code_type="c++", options=core.ProgramOptions(**opts))
    ptx = bytes(program.compile(kind).code).decode()
    assert f'\t.file\t1 "{opts["name"]}"' in ptx.splitlines()
    assert "\t.loc\t1 4 3" in ptx.splitlines()  # x[threadIdx.x] += 1.f: line 4
    monkeypatch.setenv(toolchain.LINEINFO_ENV, "0")
    opts, kind = toolchain.nvrtc_options(SRC, (8, 6), arch="compute_86")
    opts["include_path"] = []
    program = core.Program(SRC, code_type="c++", options=core.ProgramOptions(**opts))
    assert ".loc" not in bytes(program.compile(kind).code).decode()


# ------------------------------------------------------------------ --ncu-mode


def test_line_info_env_adds_the_flag_and_moves_extension_builds(tmp_path):
    env = toolchain.lineinfo_env(
        {"NVCC_APPEND_FLAGS": "-allow-unsupported-compiler", "TORCH_EXTENSIONS_DIR": str(tmp_path)}
    )
    assert env == {
        toolchain.LINEINFO_ENV: "1",
        "NVCC_APPEND_FLAGS": "-allow-unsupported-compiler -lineinfo",
        "TORCH_EXTENSIONS_DIR": str(tmp_path / toolchain.LINEINFO_DIR),
    }
    assert toolchain.lineinfo_env(env) == env  # applied twice: the same
    default = toolchain.lineinfo_env({})
    assert default["NVCC_APPEND_FLAGS"] == "-lineinfo"
    lineinfo_dir = os.path.join(toolchain.torch_extensions_default(), toolchain.LINEINFO_DIR)
    assert default["TORCH_EXTENSIONS_DIR"] == lineinfo_dir


def test_ncu_mode_builds_the_candidate_with_line_info(monkeypatch, tmp_path):
    """``--ncu-mode`` sets the line-info environment before the candidate builds anything;
    the evaluations' own builds (other processes) keep theirs."""
    from kernel_agent.profiling import capture

    monkeypatch.setenv("NVCC_APPEND_FLAGS", "-DX=1")
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "ext"))
    monkeypatch.setenv(toolchain.LINEINFO_ENV, "0")
    monkeypatch.setattr(toolchain, "setup", lambda *args, **kwargs: None)
    keys = (toolchain.LINEINFO_ENV, "NVCC_APPEND_FLAGS", "TORCH_EXTENSIONS_DIR")
    seen: dict[str, str | None] = {}

    class Loaded(Exception):
        pass

    def load_capture(*args: Any, **kwargs: Any) -> None:
        seen.update({k: os.environ.get(k) for k in keys})
        raise Loaded

    monkeypatch.setattr(capture, "load_capture", load_capture)
    with pytest.raises(Loaded):
        ncu.ncu_entry(tmp_path / "capture.pt", tmp_path / "candidate.py")
    assert seen == {
        toolchain.LINEINFO_ENV: "1",
        "NVCC_APPEND_FLAGS": "-DX=1 -lineinfo",
        "TORCH_EXTENSIONS_DIR": str(tmp_path / "ext" / "lineinfo"),
    }
    assert toolchain.lineinfo()
