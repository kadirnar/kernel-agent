"""CPU compile matrix (#256): every bundled example and kernel-agent's own device code,
compiled without a GPU for each GPU of :data:`MATRIX`, against the GPUs the code declares.

An example declares the GPUs it runs on (``ARCHS``) and, where a limit is runtime-only
(PDL, FP8 tensor cores, a library call), the wider set its device code compiles for
(``ARCHS_COMPILES``; :func:`kernel_agent.gpu_arch.example_compiles`). Its code must compile
exactly there: a declaration that is too narrow or too wide fails ``test_arch_matrix.py``
with the arch and the compiler's error. How each piece compiles (workers run with
``CUDA_VISIBLE_DEVICES=``):

==================================  ============================================================
code                                compile
==================================  ============================================================
CUDA examples' ``CUDA_SRC``         ``load_inline``'s preamble and flags, ``nvcc -cubin``
                                    ``-arch=sm_XX`` with the toolchain's ``NVCC_APPEND_FLAGS``
native projects' ``.cu`` sources    the same ``nvcc`` command, the project's include dirs and
                                    ``cuda_cflags``
NVRTC sources                       ``cuda.core.Program``: ``nvrtc_*`` examples, ``graphloop``,
                                    ``mma_peaks`` (each instruction), ``probes.PDL_SRC``
Triton kernels                      ``triton.compile(ASTSource, GPUTarget)`` with the JIT's
                                    specialisation, recorded from launches (:data:`TRITON`)
CuTe DSL                            ``CUTE_DSL_ARCH`` and fake tensors (:data:`CUTE`)
TileLang                            ``TILELANG_DEFAULT_TARGET`` (:data:`TILELANG`)
==================================  ============================================================

Verdicts are cached by content (source, flags, arch, tool versions) in
``~/.cache/kernel-agent/arch-matrix`` (``KERNEL_AGENT_ARCH_MATRIX_CACHE``: another directory,
or ``off``). A cold run compiles 305 units, 94 % of the time nvcc parsing the torch headers
(176 s on 12 cores at load 6, 213 s at load 13); a warm one reads them back in a second, and
a changed example recompiles only its own units. ``KERNEL_AGENT_ARCH_MATRIX_JOBS`` bounds the
parallel compiles (default: half the CPUs, 2 to 8); concurrent test sessions wait for each
other's compiles instead of repeating them (a lock per unit).

Print the matrix::

    uv run --no-sync python tests/arch_matrix.py          # --json: every unit's result
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import subprocess
import sys
import sysconfig
import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if __name__ == "__main__" and str(SRC) not in sys.path:  # a worker or the CLI
    sys.path.insert(0, str(SRC))

from kernel_agent import gpu_arch  # noqa: E402

PACKAGE = SRC / "kernel_agent"
EXAMPLES = PACKAGE / "agent" / "examples"
#: The GPUs of the matrix: Turing, Ampere (A100; A10 / RTX 30xx), Ada, GeForce Blackwell.
MATRIX = ("sm_75", "sm_80", "sm_86", "sm_89", "sm_120")
CAPS = {arch: gpu_arch.capability_of(arch) or (0, 0) for arch in MATRIX}
#: Part of every cache key: bump it when a compile path changes.
FORMAT = 1
CACHE_ENV = "KERNEL_AGENT_ARCH_MATRIX_CACHE"
JOBS_ENV = "KERNEL_AGENT_ARCH_MATRIX_JOBS"
#: ``torch.utils.cpp_extension.load_inline`` puts these lines in front of ``cuda_sources``.
INLINE_PREAMBLE = "#include <torch/types.h>\n#include <cuda.h>\n#include <cuda_runtime.h>\n"
NVCC_TIMEOUT_S = 900
WORKER_TIMEOUT_S = 1800
_MARK = "@@arch-matrix@@"


# ------------------------------------------------------------------ what is compiled


@dataclass(frozen=True)
class TritonSpec:
    """One specialisation of a Triton kernel as its example launches it (recorded with a
    hook on ``JITFunction._do_compile``). ``args``: ``"names: type; ..."`` in Triton's
    signature types; pointers are 16-byte aligned (torch allocations), ``/16`` marks an
    integer the JIT specialises as divisible by 16. ``consts``: constexprs, including the
    integers the JIT specialises to 1. ``where``: the GPUs that launch it (None: all).
    ``mma``: ``(GPUs, mma.sync form)`` pairs its PTX must show where it compiles ("": no
    ``mma.sync`` at all). ``pipelined``: a ``tl.dot`` K loop whose loads must be ``cp.async``
    on sm_80+."""

    kernel: str
    args: str
    consts: dict[str, Any] = field(default_factory=dict)
    warps: int = 4
    stages: int = 3
    label: str = ""
    where: str | None = None
    mma: tuple[tuple[str, str], ...] = ()
    pipelined: bool = False


def _args(text: str) -> dict[str, str]:
    """``"a b: *bf16; n: i32/16"`` -> ``{"a": "*bf16", "b": "*bf16", "n": "i32/16"}``."""
    out: dict[str, str] = {}
    for part in filter(None, (p.strip() for p in text.split(";"))):
        names, _, ty = part.partition(":")
        for name in names.split():
            out[name] = ty.strip()
    return out


HALF = ("bf16", "fp16")
_PTX_TYPE = {"bf16": "bf16", "fp16": "f16"}
_ATTN = (
    "q k v o m: *{dt}; Hq D: i32/16; G Sk: i32; sqb sqh sqs skb skh sks svb svh svs: i32/16; "
    "sob soh sos smb smh smq smk: i32/16; scale: fp32"
)
_ATTN_CONSTS = {"sqd": 1, "skd": 1, "svd": 1, "MASK": 0, "S": 16, "BD": 128}
#: The W8A8 GEMMs' scales and output (without a bias, its slot holds the fp32 scales).
_W8A8 = "sa sb bias: *fp32; c: *{dt}; M N K: i32/16"
#: GPUs that are not GeForce Blackwell (``supports`` has no negation).
NOT_SM12X = "sm_7x, sm_8x, sm_9x, sm_10x, sm_11x"
#: The FP8 producers' arguments besides the scales (``s``).
_FP8_ROWS = {
    "_quant_rows_kernel": "x: *bf16; q: *fp8e4nv; M K ldx: i32/16",
    "_rmsnorm_fp8_kernel": "x w: *bf16; q: *fp8e4nv; M K ldx: i32/16; eps: fp32",
    "_silu_mul_fp8_kernel": "g u: *bf16; q: *fp8e4nv; M K ldg ldu: i32/16",
}

#: The Triton kernels of each example (and of kernel-agent's own Triton probes), as their
#: launches specialise them (recorded on the RTX 5070 Ti with tiny inputs; the FP8 GEMM's
#: ``tl.dot`` form is what every GPU but sm_12x launches). Every launched ``@triton.jit``
#: function needs one (``test_triton_specs_match_the_kernels``).
TRITON: dict[str, list[TritonSpec]] = {
    "triton_rmsnorm.py": [
        TritonSpec(
            "_rmsnorm_kernel",
            f"x_ptr w_ptr y_ptr: *{dt}; stride_row n_cols: i32/16; eps: fp32",
            {"BLOCK": 2048},
            label=dt,
        )
        for dt in HALF
    ],
    "triton_rmsnorm_custom_op.py": [
        TritonSpec(
            "_rmsnorm_kernel",
            f"x_ptr w_ptr y_ptr: *{dt}; stride_row n_cols: i32/16; eps: fp32",
            {"BLOCK": 2048},
            label=dt,
        )
        for dt in HALF
    ],
    "triton_cheap_launch.py": [
        *(
            TritonSpec(
                "_rmsnorm_kernel",
                f"x w y: *{dt}; stride_x stride_y n_cols: i32/16; eps: fp32",
                {"BLOCK": 1024},
                label=dt,
            )
            for dt in HALF
        ),
        *(
            TritonSpec(
                "_silu_mul_kernel",
                f"gu out: *{dt}; n stride_gu stride_out: i32/16",
                {"BLOCK": 1024},
                label=dt,
            )
            for dt in HALF
        ),
    ],
    "triton_short_attention.py": [
        spec
        for dt in HALF
        for spec in (
            TritonSpec(
                "_attn_kernel",
                _ATTN.format(dt=dt) + "; Sq: i32",
                {**_ATTN_CONSTS, "CAUSAL": True},
                label=f"{dt},causal",
                # Triton has no tensor-core MMA below sm_80: tl.dot runs on FMA there
                mma=(("sm_75", ""), ("sm_80+", f"m16n8k16.row.col.f32.{_PTX_TYPE[dt]}")),
            ),
            TritonSpec(
                "_attn_kernel",
                _ATTN.format(dt=dt),
                {**_ATTN_CONSTS, "Sq": 1, "CAUSAL": False},
                warps=8,
                label=f"{dt},decode",
            ),
        )
    ],
    # the W8A8 GEMMs take the activations' dtype (bf16 or fp16) from the input (#258)
    "triton_int8_w8a8_gemm.py": [
        spec
        for dt in HALF
        for spec in (
            TritonSpec(
                "_quant_kernel",
                f"x_ptr: *{dt}; q_ptr: *i8; s_ptr: *fp32; K stride_x: i32/16",
                {"BLOCK": 1024},
                label=dt,
            ),
            TritonSpec(
                "_gemm_kernel",
                "a b: *i8; " + _W8A8.format(dt=dt),
                {"HAS_BIAS": False, "BM": 128, "BN": 128, "BK": 64, "GM": 8},
                warps=8,
                label=dt,
                mma=(("sm_80+", "m16n8k32.row.col.satfinite.s32.s8.s8.s32"),),
                pipelined=True,
            ),
        )
    ],
    "triton_fp8_w8a8_gemm.py": [
        spec
        for dt in HALF
        for spec in (
            TritonSpec(
                "_quant_kernel",
                f"x_ptr: *{dt}; q_ptr: *fp8e4nv; s_ptr: *fp32; K stride_x: i32/16",
                {"BLOCK": 1024},
                label=dt,
            ),
            TritonSpec(
                "_gemm_kernel",
                "a b: *fp8e4nv; " + _W8A8.format(dt=dt),
                {"HAS_BIAS": False, "BM": 64, "BN": 64, "BK": 128, "GM": 8, "SCALED": False},
                label=f"{dt},tl.dot",
                where=NOT_SM12X,
                mma=(("sm_89", "m16n8k32.row.col.f32.e4m3.e4m3.f32"),),
                pipelined=True,
            ),
            TritonSpec(
                "_gemm_kernel",
                "a b: *fp8e4nv; " + _W8A8.format(dt=dt),
                {"HAS_BIAS": False, "BM": 64, "BN": 64, "BK": 128, "GM": 8, "SCALED": True},
                label=f"{dt},tl.dot_scaled",
                where="sm_12x",
                mma=(("sm_12x", "kind::mxf8f6f4.block_scale"),),
                pipelined=True,
            ),
        )
    ],
    "triton_fp8_kv_decode.py": [
        TritonSpec(
            "_quant_kv_kernel", "x: *bf16; q: *fp8e4nv; s: *fp32; R: i32/16", {"D": 128, "BR": 16}
        ),
        TritonSpec(
            "_split_kernel",
            "q: *bf16; kc vc: *fp8e4nv; ks vs lens po pm pl: *fp32; Hkv: i32; L chunk: i32/16; "
            "sm_scale: fp32; sqb sqh skb skh skl ssb ssh svb svh svl stb sth: i32/16",
            {"G": 8, "GP": 16, "D": 128, "BL": 64, "HAS_LENS": False},
        ),
        TritonSpec(
            "_combine_kernel",
            "po pm pl: *fp32; out: *bf16; S Hkv: i32; sob soh: i32/16",
            {"G": 8, "GP": 16, "D": 128},
        ),
    ],
    "triton_fp8_producers.py": [
        TritonSpec(
            kernel,
            args + ("; s: *u8" if mx else "; s: *fp32"),
            {"MX": mx, "BLOCKED": mx, "BLOCK": 1024},
            label="mx" if mx else "per-token",
        )
        for kernel, args in _FP8_ROWS.items()
        for mx in (False, True)
    ],
    "triton_mxfp8_gemm.py": [
        TritonSpec(
            "_mx_quant_kernel",
            "x_ptr: *bf16; q_ptr: *fp8e4nv; s_ptr: *u8; K stride_x scale_cols: i32/16",
            {"CEIL": True, "SWIZZLE": swizzle, "BLOCKS": 4},
            warps=1,
            label="swizzled" if swizzle else "rows",
        )
        for swizzle in (False, True)
    ],
    "triton_nvfp4_w4a4_gemm.py": [
        TritonSpec(
            "_amax_kernel", "x_ptr: *bf16; amax_ptr: *fp32; K stride_x: i32/16", {"CHUNK": 512}
        ),
        *(
            TritonSpec(
                "_nvfp4_quant_kernel",
                "x_ptr: *bf16; amax_ptr: *fp32; q_ptr s_ptr: *u8; K stride_x scale_cols: i32/16; "
                "outer_step: fp32",
                {"SWIZZLE": swizzle, "BLOCKS": 32},
                warps=2,
                label="swizzled" if swizzle else "rows",
            )
            for swizzle in (False, True)
        ),
    ],
}
#: kernel-agent's own Triton kernels (module path under the package -> specs).
LIBRARY_TRITON: dict[str, list[TritonSpec]] = {
    "kernels/memcheck_probe.py": [
        TritonSpec(kernel, "x_ptr y_ptr: *fp32; n: i32", {"BLOCK": 1024})
        for kernel in ("_ka_memcheck_inbounds", "_ka_memcheck_overrun")
    ],
}

_RMS_FAKE = """
for dt in (cutlass.BFloat16, cutlass.Float16):
    x = fake(dt, (4, 2048), stride_order=(1, 0), assumed_align=16)
    y = fake(dt, (4, 2048), stride_order=(1, 0), assumed_align=16)
    w = fake(dt, (2048,), assumed_align=16)
    s = stream(use_tvm_ffi_env_stream=True)
    cute.compile(ex._rmsnorm, x, w, y, cutlass.Float32(1e-5), s, options="--enable-tvm-ffi")
"""
#: CuTe DSL examples: unit -> code compiling it with fake tensors (``ex``: the example;
#: ``cutlass``, ``cute``, ``fake`` = ``make_fake_compact_tensor``, ``stream`` =
#: ``make_fake_stream``).
CUTE: dict[str, dict[str, str]] = {
    "cute_rmsnorm.py": {"rmsnorm[bf16,fp16]": _RMS_FAKE},
    "cute_fp8_decoder_block.py": {
        "rows_gemv[norm,gated]": "ex.compile_rows_gemv(16, 1024, 4096, norm=True, gated=True)",
        "rows_gemv[residual]": "ex.compile_rows_gemv(4, 4096, 1024, residual=True)",
    },
    "cute_fp8_blockscaled_gemm.py": {
        "quant": "ex.compile_quant(1024)",
        "gemm": "ex.compile_gemm(1024, 1024, max_ctas=70)",
    },
    "cute_nvfp4_w4a4_gemm.py": {
        "quant": "ex.compile_quant(1024)",
        "gemm": "ex.compile_gemm(1024, 1024)",
    },
}
#: TileLang examples: unit -> code building it (``ex``: the example).
TILELANG: dict[str, dict[str, str]] = {
    "tilelang_rmsnorm.py": {
        f"rmsnorm[{dt}]": f"ex._rmsnorm(16, 2048, 1e-5, {dt!r})" for dt in ("bfloat16", "float16")
    },
}
#: Bundled examples the matrix does not compile, and why. (Examples whose declarations name
#: no GPU of :data:`MATRIX`, the Hopper / datacenter Blackwell templates, are left out too.)
OUT_OF_MATRIX = {
    "graph_while_decode.py": "no device code of its own: graphloop's NVRTC kernels are in the "
    "matrix (graphloop.KERNELS_SRC, graphloop.COUNT_SRC)",
    "helion_gemm_epilogue.py": "Helion compiles through Triton at its first call (autotuned "
    "configs): no CPU compile path; doctor's helion probe and the GPU selftests cover it",
    "helion_rmsnorm.py": "Helion compiles through Triton at its first call (autotuned configs): "
    "no CPU compile path; doctor's helion probe and the GPU selftests cover it",
}


@dataclass
class Unit:
    """One compile: ``backend`` builds ``payload`` for a GPU of the matrix."""

    name: str
    backend: str  # nvcc | nvrtc | triton | cute | tilelang
    payload: dict[str, Any]
    files: tuple[Path, ...] = ()  # whose contents the verdict depends on, besides the payload
    where: str | None = None  # the GPUs that build it (gpu_arch.supports); None: every one
    spec: TritonSpec | None = None

    def applies(self, arch: str) -> bool:
        return self.where is None or gpu_arch.supports(self.where, CAPS[arch])


@dataclass
class Group:
    """What one declaration covers: a bundled example or one of kernel-agent's sources."""

    name: str
    declared: str  # where the expected GPUs come from, for messages
    expected: dict[str, bool]  # arch -> it must compile there
    units: list[Unit]
    example: bool = False  # a bundled example (else kernel-agent's own code)
    runs: str | None = None  # an example's ARCHS


@dataclass
class Result:
    ok: bool
    error: str = ""
    skipped: str = ""  # not compiled: a tool is missing
    mma: list[str] = field(default_factory=list)  # mma.sync forms in the PTX (Triton)
    cp_async: bool = False
    shared: int = 0  # bytes of shared memory per block (Triton)
    seconds: float = 0.0  # the compile's wall time
    cacheable: bool = True


def _constants(path: Path) -> dict[str, str]:
    """Top-level string constants of a Python file."""
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
            if name and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                out[name] = node.value.value
    return out


def _inline_flags(path: Path) -> list[str]:
    """``extra_cuda_cflags`` of the example's ``load_inline`` call (``["-O3"]`` if none)."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "load_inline":
            for kw in node.keywords:
                if kw.arg == "extra_cuda_cflags":
                    return [str(f) for f in ast.literal_eval(kw.value)]
    return ["-O3"]


def _toolkit_headers() -> tuple[Path, ...]:
    """kernel-agent's C++ headers on every build's include path (``ka_launch.cuh``,
    ``ka_mk.cuh``)."""
    from kernel_agent.native import project

    return tuple(sorted(p for d in project.toolkit_includes() for p in d.glob("*") if p.is_file()))


def _expected(spec: str | None) -> dict[str, bool]:
    return {arch: gpu_arch.supports(spec, CAPS[arch]) for arch in MATRIX}


def _example_units(path: Path) -> list[Unit]:
    """The compiles of one bundled example (empty: not compiled by the matrix)."""
    name = path.name
    headers = _toolkit_headers()
    if path.is_dir() and (path / "kernel_project.toml").is_file():
        from kernel_agent.native import project

        proj = project.Project.from_dir(path)
        m = proj.manifest
        # what the device code is built from: sources, headers, the manifest (not the
        # Python entry point: an ARCHS edit recompiles nothing)
        built = {rel: text for rel, text in proj.files.items() if not rel.endswith(".py")}
        digest = _digest(json.dumps(built, sort_keys=True))
        return [
            Unit(
                source,
                "nvcc",
                {
                    "kind": "native",
                    "project": name,
                    "file": source,
                    "digest": digest,
                    "flags": list(m.cuda_cflags),
                    "include_dirs": list(m.include_dirs),
                },
                headers,
            )
            for source in m.sources
            if source.endswith(".cu")
        ]
    if name.startswith("cuda_") and "CUDA_SRC" in (consts := _constants(path)):
        payload = {
            "kind": "inline",
            "name": path.stem,
            "source": INLINE_PREAMBLE + "\n" + consts["CUDA_SRC"],
            "flags": _inline_flags(path),
        }
        return [Unit("CUDA_SRC", "nvcc", payload, headers)]
    if name.startswith("nvrtc_") and "SRC" in (consts := _constants(path)):
        payload = {"source": consts["SRC"], "includes": True, "specific": False}
        return [Unit("SRC", "nvrtc", payload)]
    if name.startswith("triton_"):
        return _triton_units(path, TRITON.get(name, []))
    for prefix, table, backend in (("cute_", CUTE, "cute"), ("tilelang_", TILELANG, "tilelang")):
        if name.startswith(prefix):
            return [
                Unit(unit, backend, {"path": _rel(path), "code": code}, (path,))
                for unit, code in table.get(name, {}).items()
            ]
    return []


def _rel(path: Path) -> str:
    return path.relative_to(SRC).as_posix()


def _triton_units(path: Path, specs: list[TritonSpec]) -> list[Unit]:
    units = []
    for spec in specs:
        label = f"[{spec.label}]" if spec.label else ""
        payload = {
            "path": _rel(path),
            "kernel": spec.kernel,
            "args": _args(spec.args),
            "consts": spec.consts,
            "warps": spec.warps,
            "stages": spec.stages,
        }
        units.append(Unit(spec.kernel + label, "triton", payload, (path,), spec.where, spec))
    return units


def _library_groups() -> list[Group]:
    """kernel-agent's own device code: graphloop, mma_peaks, the PDL probe, Triton probes."""
    from kernel_agent import graphloop, probes
    from kernel_agent.kernels import mma_peaks

    every = _expected(None)
    groups = [
        Group(
            f"graphloop.{name}",
            "every GPU",
            every,
            [Unit(name, "nvrtc", {"source": source, "includes": False, "specific": False})],
        )
        for name, source in (
            ("KERNELS_SRC", graphloop.KERNELS_SRC),
            ("COUNT_SRC", graphloop.COUNT_SRC),
        )
    ]
    for ins in mma_peaks.INSTRUCTIONS:
        source = mma_peaks.source(ins)
        groups.append(
            Group(
                f"mma_peaks.{ins.key}",
                "mma_peaks.available()",
                {arch: mma_peaks.available(ins, CAPS[arch]) is None for arch in MATRIX},
                [Unit(ins.key, "nvrtc", {"source": source, "includes": False, "specific": True})],
            )
        )
    pdl = probes.ARCHS["pdl"][0]
    groups.append(
        Group(
            "probes.PDL_SRC",
            f"probes.ARCHS['pdl'] = {pdl!r}",
            _expected(pdl),
            [Unit("PDL_SRC", "nvrtc", {"source": probes.PDL_SRC, "includes": False})],
        )
    )
    for rel, specs in LIBRARY_TRITON.items():
        groups.append(Group(rel, "every GPU", every, _triton_units(PACKAGE / rel, specs)))
    return groups


def examples() -> list[Path]:
    """Every bundled example (files and project directories)."""
    return [
        p
        for p in sorted(EXAMPLES.iterdir())
        if not p.name.startswith(("_", "."))
        and (p.suffix == ".py" or (p.is_dir() and any(p.glob("*.py"))))
    ]


def collect() -> tuple[list[Group], dict[str, str]]:
    """``(groups, left out: example -> why)``: the matrix's compiles, from the examples'
    declarations and :data:`TRITON` / :data:`CUTE` / :data:`TILELANG`."""
    groups: list[Group] = []
    out: dict[str, str] = {}
    for path in examples():
        if path.name in OUT_OF_MATRIX:
            out[path.name] = OUT_OF_MATRIX[path.name]
            continue
        runs, _why = gpu_arch.example_requirement(path)
        compiles = gpu_arch.example_compiles(path)
        expected = _expected(compiles)
        if not any(expected.values()) and not any(_expected(runs).values()):
            out[path.name] = f"ARCHS = {runs!r}: no GPU of the matrix ({', '.join(MATRIX)})"
            continue
        if compiles != runs:
            declared = f"ARCHS_COMPILES = {compiles!r}"
        else:
            declared = f"ARCHS = {runs!r}" if runs else "no ARCHS (every GPU)"
        units = _example_units(path)
        groups.append(Group(path.name, declared, expected, units, example=True, runs=runs))
    return groups + _library_groups(), out


# ------------------------------------------------------------------ tools and cache


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _tool_line(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = [line for line in out.stdout.splitlines() if line.strip()]
    return lines[-1] if lines else None


#: The Python module each Python-hosted backend needs.
BACKEND_MODULE = {
    "triton": "triton",
    "cute": "cutlass",
    "tilelang": "tilelang",
    "nvrtc": "cuda.core",
}


@dataclass
class Tools:
    """The compilers this machine has, their versions (part of every cache key) and the
    environment the toolchain sets (``CUDA_HOME``, ``NVCC_APPEND_FLAGS``)."""

    nvcc: str | None
    env: dict[str, str]
    versions: dict[str, str | None]

    @classmethod
    def find(cls) -> Tools:
        from kernel_agent import toolchain

        tc = toolchain.setup(apply_env=False)
        nvcc = str(Path(tc.cuda_home) / "bin" / "nvcc") if tc.cuda_home else None
        if nvcc is not None and not Path(nvcc).is_file():
            nvcc = None
        env = {k: v for k, v in tc.env.items() if k in ("CUDA_HOME", "NVCC_APPEND_FLAGS")}
        if tc.cuda_home:
            env.setdefault("CUDA_HOME", tc.cuda_home)
        versions = {
            dist: _version(dist)
            for dist in ("torch", "triton", "nvidia-cutlass-dsl", "tilelang", "cuda-core")
        }
        versions["nvcc"] = _tool_line([nvcc, "--version"]) if nvcc else None
        versions["c++"] = _tool_line(["c++", "-dumpfullversion"])
        versions["python"] = sys.version.split()[0]
        versions["NVCC_APPEND_FLAGS"] = env.get("NVCC_APPEND_FLAGS", "")
        return cls(nvcc, env, versions)

    @staticmethod
    def installed(module: str) -> bool:
        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            return False

    def missing(self, backend: str) -> str | None:
        """Why ``backend`` cannot compile here (None: it can)."""
        module = BACKEND_MODULE.get(backend)
        if backend in ("nvcc", "tilelang") and self.nvcc is None:
            return "no nvcc (a CUDA toolkit or the nvidia-cuda-nvcc wheel)"
        if module is not None and not self.installed(module):
            return f"{module} is not installed"
        return None

    def fingerprint(self, backend: str) -> dict[str, Any]:
        need = {
            "nvcc": ("torch", "nvcc", "c++", "NVCC_APPEND_FLAGS"),
            "nvrtc": ("cuda-core", "nvcc"),
            "triton": ("triton", "torch"),
            "cute": ("nvidia-cutlass-dsl", "torch"),
            "tilelang": ("tilelang", "torch", "nvcc", "c++", "NVCC_APPEND_FLAGS"),
        }[backend]
        return {name: self.versions.get(name) for name in (*need, "python")}


def _digest(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def key_of(unit: Unit, arch: str, tools: Tools) -> str:
    """The cache key of one compile: everything its verdict depends on."""
    files = {
        _rel(p) if p.is_relative_to(SRC) else p.name: _digest(p.read_bytes()) for p in unit.files
    }
    blob = {
        "format": FORMAT,
        "backend": unit.backend,
        "arch": arch,
        "payload": unit.payload,
        "files": files,
        "tools": tools.fingerprint(unit.backend),
    }
    return _digest(json.dumps(blob, sort_keys=True, default=str))


class Cache:
    """Verdicts by key (``<key>.json``); None: off."""

    def __init__(self, root: Path | None) -> None:
        self.root = root
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def default(cls) -> Cache:
        value = os.environ.get(CACHE_ENV, "")
        if value.lower() == "off":
            return cls(None)
        if value:
            return cls(Path(value))
        from kernel_agent.toolchain import CACHE_DIR

        return cls(CACHE_DIR / "arch-matrix")

    def get(self, key: str) -> Result | None:
        if self.root is None:
            return None
        try:
            return Result(**json.loads((self.root / f"{key}.json").read_text()))
        except (OSError, ValueError, TypeError):
            return None

    def put(self, key: str, result: Result) -> None:
        if self.root is None or not result.cacheable:
            return
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump(asdict(result), fh)
        os.replace(tmp, self.root / f"{key}.json")

    @contextlib.contextmanager
    def lock(self, key: str) -> Iterator[None]:
        """Held while ``key`` compiles: another session waits for it instead of compiling the
        same unit (it reads the verdict once the lock is free)."""
        if self.root is None:
            yield
            return
        path = self.root / f"{key}.lock"
        with path.open("a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                path.unlink(missing_ok=True)
                fcntl.flock(fh, fcntl.LOCK_UN)


# ------------------------------------------------------------------ compiling


@dataclass
class Job:
    group: str
    unit: Unit
    arch: str
    key: str


_ANSI = re.compile(r"\x1b\[[0-9;]*m")
#: ``ptxas /tmp/tmpxft_..._x.ptx, line 949; error   : ...`` -> ``ptxas error : ...``
_PTXAS_AT = re.compile(r"^ptxas (?:\S+|application ptx input), line \d+; ")


def _short_error(text: str) -> str:
    """The compiler's diagnostics (the first and the last ``error: ...`` line: the summary
    and the most specific one; else its last line), without colours, temporary paths and
    repeats: one line, at most 400 chars."""
    lines: list[str] = []
    for raw in _ANSI.sub("", text).splitlines():
        line = _PTXAS_AT.sub("ptxas ", re.sub(r"\s+", " ", raw).strip())
        if line and line not in lines:
            lines.append(line)
    diagnostics = [line for line in lines if re.search(r"\berror ?:", line, re.I)]
    other = [line for line in lines if re.search(r"error|unsupported|not supported", line, re.I)]
    found = diagnostics or other
    picked = list(dict.fromkeys(found[:1] + found[-1:])) or lines[-1:]
    return " | ".join(picked)[:400]


def _nvcc_command(tools: Tools, unit: Unit, arch: str, tmp: Path) -> list[str]:
    """``load_inline`` / ``load``'s nvcc flags (torch's ``_write_ninja_file_to_build_library``)
    for one arch, compiled to a cubin only."""
    from torch.utils import cpp_extension

    payload = unit.payload
    includes: list[str] = []
    if payload["kind"] == "native":
        root = EXAMPLES / payload["project"]
        source = root / payload["file"]
        includes += [str(root), *(str(root / d) for d in payload["include_dirs"])]
    else:
        source = tmp / f"{payload['name']}.cu"
        source.write_text(payload["source"])
    from kernel_agent.native import project

    includes += [str(d) for d in project.toolkit_includes()]
    system = [
        *cpp_extension.include_paths("cpu"),
        sysconfig.get_path("include", scheme="posix_prefix"),
    ]
    assert tools.nvcc is not None
    flags = list(payload["flags"])
    return [
        tools.nvcc,
        "-cubin",
        f"-arch={arch}",
        "-o",
        str(tmp / "out.cubin"),
        str(source),
        "-DTORCH_EXTENSION_NAME=ka_arch_matrix",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        *(f"-I{d}" for d in includes),
        *(f for d in system for f in ("-isystem", d)),
        *cpp_extension.COMMON_NVCC_FLAGS,
        *flags,
        *([] if any(f.startswith("-std=") for f in flags) else ["-std=c++20"]),
    ]


def _run_nvcc(tools: Tools, job: Job) -> Result:
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="ka-arch-matrix-") as tmp:
        cmd = _nvcc_command(tools, job.unit, job.arch, Path(tmp))
        env = {**os.environ, **tools.env, "CUDA_VISIBLE_DEVICES": ""}
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, env=env, cwd=tmp, timeout=NVCC_TIMEOUT_S
            )
        except subprocess.TimeoutExpired:
            return Result(False, f"nvcc exceeded {NVCC_TIMEOUT_S} s", cacheable=False)
    seconds = round(time.monotonic() - start, 1)
    if proc.returncode == 0:
        return Result(True, seconds=seconds)
    return Result(False, _short_error(proc.stderr + proc.stdout), seconds=seconds)


def _worker_env(tools: Tools, backend: str, arch: str, tmp: str) -> dict[str, str]:
    env = {
        **os.environ,
        **tools.env,
        "CUDA_VISIBLE_DEVICES": "",  # nothing may run on a GPU
        "PYTHONPATH": os.pathsep.join(filter(None, [str(SRC), os.environ.get("PYTHONPATH")])),
        "TRITON_CACHE_DIR": os.path.join(tmp, "triton"),
        "KERNEL_AGENT_CUTE_CACHE": "off",
        "TILELANG_CACHE_DIR": os.path.join(tmp, "tilelang"),
        "TVM_FFI_CACHE_DIR": os.path.join(tmp, "tvm-ffi"),
        "KERNEL_AGENT_TUNED_DB": os.path.join(tmp, "tuned.sqlite"),
    }
    if backend == "cute":
        major, minor = CAPS[arch]
        env["CUTE_DSL_ARCH"] = f"sm_{major}{minor}" + ("a" if major >= 9 else "")
    if backend == "tilelang":
        env["TILELANG_DEFAULT_TARGET"] = json.dumps({"kind": "cuda", "arch": arch})
    return env


def _run_worker(tools: Tools, backend: str, arch: str | None, jobs: list[Job]) -> list[Result]:
    """Compile ``jobs`` in a fresh Python process (``backend`` per arch: CuTe DSL and TileLang
    read their target from the environment)."""
    payload = [{"backend": j.unit.backend, "arch": j.arch, "payload": j.unit.payload} for j in jobs]
    with tempfile.TemporaryDirectory(prefix="ka-arch-matrix-") as tmp:
        try:
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker"],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                env=_worker_env(tools, backend, arch or MATRIX[0], tmp),
                cwd=tmp,
                timeout=WORKER_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return [Result(False, "worker timed out", cacheable=False) for _ in jobs]
    for line in proc.stdout.splitlines():
        if line.startswith(_MARK):
            out = [Result(**r) for r in json.loads(line[len(_MARK) :])]
            if len(out) == len(jobs):
                return out
    why = f"worker failed (exit {proc.returncode}): {_short_error(proc.stderr)}"
    return [Result(False, why, cacheable=False) for _ in jobs]


def _jobs() -> int:
    try:
        return max(1, int(os.environ[JOBS_ENV]))
    except (KeyError, ValueError):
        return max(2, min(8, (os.cpu_count() or 4) // 2))


@dataclass
class Matrix:
    """Every unit's result per arch: ``results[(group, unit, arch)]``."""

    groups: list[Group]
    out: dict[str, str]
    results: dict[tuple[str, str, str], Result]

    def group(self, name: str) -> Group:
        return next(g for g in self.groups if g.name == name)

    def verdict(self, group: Group, arch: str) -> bool | None:
        """Whether every unit of ``group`` built for ``arch`` compiles (None: nothing is built
        there, or a tool is missing)."""
        rows = [self.results[(group.name, u.name, arch)] for u in group.units if u.applies(arch)]
        if not rows or any(r.skipped for r in rows):
            return None
        return all(r.ok for r in rows)

    def errors(self, group: Group, arch: str) -> list[str]:
        return [
            f"{u.name}: {r.error}"
            for u in group.units
            if u.applies(arch) and not (r := self.results[(group.name, u.name, arch)]).ok
        ]

    def mismatches(self, group: Group) -> list[str]:
        """What contradicts the declaration, one sentence per arch."""
        out = []
        for arch in MATRIX:
            got = self.verdict(group, arch)
            if got is None or got == group.expected[arch]:
                continue
            if got:
                fix = (
                    "widen ARCHS if it runs there, or declare ARCHS_COMPILES (it compiles but "
                    "does not run there)"
                    if group.example
                    else "update the declaration"
                )
                out.append(
                    f"{group.name} compiles for {arch} but {group.declared} excludes it: {fix}"
                )
            else:
                errors = "; ".join(self.errors(group, arch))[:600]
                out.append(
                    f"{group.name}: {group.declared} includes {arch} but it does not compile "
                    f"there ({errors}): narrow the declaration or fix the code"
                )
        return out

    def table(self) -> str:
        """Markdown: one row per group, C (compiles) / x (fails) per arch; ! marks a
        contradiction of the declaration, - nothing built, ? a missing tool."""
        head = "| code | declared | " + " | ".join(MATRIX) + " |"
        lines = [head, "|---|---|" + "---|" * len(MATRIX)]
        for g in self.groups:
            cells = []
            for arch in MATRIX:
                got = self.verdict(g, arch)
                cell = {None: "-", True: "C", False: "x"}[got]
                if any(
                    self.results[(g.name, u.name, arch)].skipped for u in g.units if u.applies(arch)
                ):
                    cell = "?"
                if got is not None and got != g.expected[arch]:
                    cell += "!"
                cells.append(cell)
            lines.append(f"| {g.name} | {g.declared} | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def failures(self) -> list[str]:
        """``group arch: error`` of every unit that does not compile (as expected or not)."""
        out = []
        for g in self.groups:
            for arch in MATRIX:
                out += [f"{g.name} {arch}: {e}" for e in self.errors(g, arch) if e]
        return out


def build(groups: list[Group] | None = None, cache: Cache | None = None) -> Matrix:
    """Compile (or read back) every unit of ``groups`` (default: :func:`collect`) for every
    arch of the matrix, at most :func:`_jobs` compiles at a time."""
    out: dict[str, str] = {}
    if groups is None:
        groups, out = collect()
    cache = cache or Cache.default()
    tools = Tools.find()
    unavailable = {b: tools.missing(b) for b in {u.backend for g in groups for u in g.units}}
    results: dict[tuple[str, str, str], Result] = {}
    todo: list[Job] = []
    for g in groups:
        for unit in g.units:
            for arch in MATRIX:
                if not unit.applies(arch):
                    continue
                if why := unavailable[unit.backend]:
                    results[(g.name, unit.name, arch)] = Result(False, skipped=why)
                    continue
                key = key_of(unit, arch, tools)
                if (hit := cache.get(key)) is not None:
                    results[(g.name, unit.name, arch)] = hit
                else:
                    todo.append(Job(g.name, unit, arch, key))

    # nvcc: one job per compile; Python backends: one process per backend and arch (CuTe DSL
    # and TileLang take the target from the environment; Triton for parallelism), NVRTC: one.
    batches: dict[tuple[str, str], list[Job]] = {}
    for job in todo:
        b = job.unit.backend
        where = job.key if b == "nvcc" else ("all" if b == "nvrtc" else job.arch)
        batches.setdefault((b, where), []).append(job)
    # the longest batches first (a Python process compiles many units; the biggest sources
    # take nvcc the longest)
    order = {"cute": 0, "triton": 1, "tilelang": 2, "nvrtc": 3, "nvcc": 4}

    def cost(item: tuple[tuple[str, str], list[Job]]) -> tuple[int, int]:
        (backend, _), jobs = item
        size = sum(len(json.dumps(j.unit.payload)) for j in jobs)
        return order[backend], -size

    def run(item: tuple[tuple[str, str], list[Job]]) -> list[tuple[Job, Result]]:
        (backend, where), jobs = item
        lock = jobs[0].key if len(jobs) == 1 else _digest("".join(j.key for j in jobs))
        with cache.lock(lock):
            missing = [j for j in jobs if cache.get(j.key) is None]
            if backend == "nvcc":
                fresh = [_run_nvcc(tools, j) for j in missing]
            elif missing:
                arch = None if where == "all" else where
                fresh = _run_worker(tools, backend, arch, missing)
            else:
                fresh = []
            for job, result in zip(missing, fresh, strict=True):
                cache.put(job.key, result)
        done = dict(zip((j.key for j in missing), fresh, strict=True))
        return [
            (j, done[j.key] if j.key in done else cache.get(j.key) or Result(False)) for j in jobs
        ]

    items = sorted(batches.items(), key=cost)
    with ThreadPoolExecutor(_jobs()) as pool:
        for pairs in pool.map(run, items):
            for job, result in pairs:
                results[(job.group, job.unit.name, job.arch)] = result
    return Matrix(groups, out, results)


# ------------------------------------------------------------------ the worker


_MODULES: dict[str, Any] = {}


def _module(rel: str) -> Any:
    """The example (or kernel-agent module) at ``rel`` under ``src``, loaded once per worker."""
    if rel not in _MODULES:
        path = SRC / rel
        name = "ka_arch_matrix_" + re.sub(r"\W", "_", path.stem)
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        _MODULES[rel] = mod
    return _MODULES[rel]


def triton_signature(
    arg_names: list[str], args: dict[str, str], consts: dict[str, Any]
) -> tuple[dict[str, str], dict[tuple[int, ...], list[list[Any]]]]:
    """``(signature, attrs)`` for ``ASTSource``: a spec's types in the kernel's parameter
    order, 16-byte divisibility on pointers and ``/16`` integers (as the JIT specialises)."""
    unknown = sorted((set(args) | set(consts)) - set(arg_names))
    missing = [n for n in arg_names if n not in args and n not in consts]
    if unknown or missing:
        raise ValueError(
            f"the spec does not match the kernel: missing {missing}, unknown {unknown}"
        )
    signature: dict[str, str] = {}
    attrs: dict[tuple[int, ...], list[list[Any]]] = {}
    for i, name in enumerate(arg_names):
        if name in consts:
            signature[name] = "constexpr"
            continue
        ty, _, div = args[name].partition("/")
        signature[name] = ty
        if ty.startswith("*") or div:
            attrs[(i,)] = [["tt.divisibility", 16]]
    return signature, attrs


def _compile_triton(payload: dict[str, Any], arch: str) -> Result:
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    fn = getattr(_module(payload["path"]), payload["kernel"])
    consts = payload["consts"]
    signature, attrs = triton_signature(list(fn.arg_names), payload["args"], consts)
    source = ASTSource(fn, signature, constexprs=consts, attrs=attrs)
    major, minor = CAPS[arch]
    options = {"num_warps": payload["warps"], "num_stages": payload["stages"]}
    compiled = triton.compile(
        source, target=GPUTarget("cuda", major * 10 + minor, 32), options=options
    )
    ptx = compiled.asm["ptx"]
    return Result(
        True,
        mma=sorted(set(re.findall(r"mma\.sync\.aligned\.([\w.:]+)", ptx))),
        cp_async="cp.async" in ptx,
        shared=int(compiled.metadata.shared),
    )


def _compile_nvrtc(payload: dict[str, Any], arch: str) -> Result:
    from cuda.core import Program, ProgramOptions

    from kernel_agent.toolchain import cuda_include_dirs

    major, minor = CAPS[arch]
    name = f"sm_{major}{minor}" + ("a" if payload.get("specific") and major >= 9 else "")
    extra = {"include_path": cuda_include_dirs()} if payload.get("includes") else {}
    options = ProgramOptions(arch=name, std="c++17", **extra)
    Program(payload["source"], code_type="c++", options=options).compile("cubin")
    return Result(True)


def _compile_cute(payload: dict[str, Any], arch: str) -> Result:
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

    scope = {
        "ex": _module(payload["path"]),
        "cutlass": cutlass,
        "cute": cute,
        "fake": make_fake_compact_tensor,
        "stream": make_fake_stream,
    }
    exec(payload["code"], scope)
    return Result(True)


def _compile_tilelang(payload: dict[str, Any], arch: str) -> Result:
    exec(payload["code"], {"ex": _module(payload["path"])})
    return Result(True)


def _worker() -> int:
    runners = {
        "triton": _compile_triton,
        "nvrtc": _compile_nvrtc,
        "cute": _compile_cute,
        "tilelang": _compile_tilelang,
    }
    out = []
    for job in json.loads(sys.stdin.read()):
        start = time.monotonic()
        try:
            result = runners[job["backend"]](job["payload"], job["arch"])
        except Exception as exc:  # the verdict: it does not compile
            result = Result(False, _short_error(f"{type(exc).__name__}: {exc}"))
        result.seconds = round(time.monotonic() - start, 1)
        out.append(asdict(result))
    sys.stdout.write(_MARK + json.dumps(out) + "\n")
    sys.stdout.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", action="store_true", help="every unit's result as JSON")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return _worker()
    matrix = build()
    if args.json:
        rows = [
            {"group": g, "unit": u, "arch": a, **asdict(r)}
            for (g, u, a), r in sorted(matrix.results.items())
        ]
        print(json.dumps(rows, indent=1))
        return 0
    print(matrix.table())
    print("\nnot compiled (x):")
    for line in matrix.failures():
        print(f"* {line}")
    print("\nleft out:")
    for name, why in matrix.out.items():
        print(f"* {name}: {why}")
    bad = [m for g in matrix.groups for m in matrix.mismatches(g)]
    for line in bad:
        print(f"MISMATCH {line}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
