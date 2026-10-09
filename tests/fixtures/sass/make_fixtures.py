"""Refresh the SASS fixtures of the opcode census (kernel_agent/kernels/sass.py, #230).

Everything compiles on the CPU (no GPU needed):

* ``kernels.cu`` for every architecture of :data:`ARCHS` with the toolkit's ``nvcc -cubin``
  -> ``<arch>.sass``;
* the FP8 W8A8 example's Triton GEMM (``agent/examples/triton_fp8_w8a8_gemm.py``) for
  sm_120 and sm_100 (:data:`TRITON_TARGETS`) with ``tl.dot`` and with ``tl.dot_scaled``
  (Triton's own compiler, specialised like the JIT specialises 16-byte aligned pointers
  and sizes divisible by 16) -> ``triton_fp8_<dot|dot_scaled>_<arch>.sass``.

Each file is the ``cuobjdump -sass`` text of the cubin (the format the census reads at run
time) without the instruction encodings. Run it after a CUDA or Triton upgrade and check
what ``tests/test_sass.py`` says: opcode names drift between versions, and sass.py's
tables list only names these cubins contain::

    uv run --no-sync python tests/fixtures/sass/make_fixtures.py
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
EXAMPLE = (
    HERE.parents[2] / "src" / "kernel_agent" / "agent" / "examples" / "triton_fp8_w8a8_gemm.py"
)
#: (nvcc -arch, KA_SM): one GPU per family, and sm_86 (A10, RTX 30xx) next to sm_80; the
#: ``a`` targets carry wgmma, tcgen05 and the block-scaled mma.sync.
ARCHS = (
    ("sm_75", 75),
    ("sm_80", 80),
    ("sm_86", 86),
    ("sm_89", 89),
    ("sm_90a", 90),
    ("sm_100a", 100),
    ("sm_120a", 120),
)
#: The Triton GEMM's targets: (capability, fixture arch, tile (BM, BN, BK, warps, stages)),
#: small tiles so the fixtures stay small. sm_100 lowers to tcgen05 from 128-row tiles
#: (Triton 3.8: a 32-row tile issues mma.sync, which for e4m3 is unpacked to fp16 there).
TRITON_TARGETS = ((120, "sm120a", (32, 32, 64, 4, 2)), (100, "sm100a", (128, 16, 32, 4, 1)))
_ENCODING = re.compile(r"\s*/\* 0x[0-9a-f]+ \*/")
_PADDING = re.compile(r"(/\*[0-9a-f]{4,}\*/)\s+")


def tools() -> tuple[str, str]:
    """``(nvcc, cuobjdump)`` of kernel-agent's toolchain."""
    from kernel_agent import toolchain
    from kernel_agent.kernels.sass import find_cuobjdump

    home = toolchain.setup().cuda_home
    nvcc = str(Path(home) / "bin" / "nvcc") if home else None
    cuobjdump = find_cuobjdump()
    if not nvcc or not Path(nvcc).is_file() or cuobjdump is None:
        sys.exit(f"needs nvcc ({nvcc}) and cuobjdump ({cuobjdump})")
    return nvcc, cuobjdump


def version(tool: str) -> str:
    out = subprocess.run([tool, "--version"], capture_output=True, text=True).stdout
    match = re.search(r"release (\S+), V(\S+)", out)
    return match.group(2) if match else "?"


def dump(cuobjdump: str, cubin: Path) -> str:
    """The SASS of ``cubin`` without encodings, column padding and empty lines."""
    sass = subprocess.run(
        [cuobjdump, "-sass", str(cubin)], check=True, capture_output=True, text=True
    ).stdout
    lines = [_PADDING.sub(r"\1 ", _ENCODING.sub("", line)).rstrip() for line in sass.splitlines()]
    return "\n".join(line for line in lines if line.strip())


def triton_gemm(scaled: bool, capability: int, tile: tuple[int, ...]) -> bytes:
    """The cubin of the FP8 example's ``_gemm_kernel`` for a GPU of ``capability`` (120:
    sm_120) with ``tile`` (``scaled``: the ``tl.dot_scaled`` product, else ``tl.dot``)."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    spec = importlib.util.spec_from_file_location("triton_fp8_w8a8_gemm", EXAMPLE)
    assert spec is not None and spec.loader is not None
    example: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    bm, bn, bk, warps, stages = tile
    ptr = {"a": "*fp8e4nv", "b": "*fp8e4nv", "sa": "*fp32", "sb": "*fp32", "bias": "*bf16"}
    consts = {"HAS_BIAS": False, "BM": bm, "BN": bn, "BK": bk, "GM": 8, "SCALED": scaled}
    signature = {**ptr, "c": "*bf16", "M": "i32", "N": "i32", "K": "i32"}
    signature.update(dict.fromkeys(consts, "constexpr"))
    aligned = {(i,): [["tt.divisibility", 16]] for i in range(len(ptr) + 4)}  # ptrs, M, N, K
    source = ASTSource(example._gemm_kernel, signature, constexprs=consts, attrs=aligned)
    options = {"num_warps": warps, "num_stages": stages}
    compiled = triton.compile(source, target=GPUTarget("cuda", capability, 32), options=options)
    return bytes(compiled.asm["cubin"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=HERE)
    ns = parser.parse_args(argv)
    nvcc, cuobjdump = tools()
    stamp = f"nvcc {version(nvcc)}, cuobjdump {version(cuobjdump)}"
    with tempfile.TemporaryDirectory() as tmp:
        for arch, sm in ARCHS:
            cubin = Path(tmp) / f"{arch}.cubin"
            cmd = [nvcc, "-cubin", "-O3", f"-arch={arch}", f"-DKA_SM={sm}", "-o", str(cubin)]
            subprocess.run([*cmd, str(HERE / "kernels.cu")], check=True)
            text = dump(cuobjdump, cubin)
            (ns.out / f"{arch}.sass").write_text(f"// {stamp}: kernels.cu, -arch={arch}\n{text}\n")
            print(f"{arch}: {len(text.splitlines())} lines")
        import triton

        for capability, arch, tile in TRITON_TARGETS:
            for scaled, name in ((False, "dot"), (True, "dot_scaled")):
                cubin = Path(tmp) / f"triton_{name}_{arch}.cubin"
                cubin.write_bytes(triton_gemm(scaled, capability, tile))
                text = dump(cuobjdump, cubin)
                head = f"// Triton {triton.__version__}, cuobjdump {version(cuobjdump)}"
                out = ns.out / f"triton_fp8_{name}_{arch}.sass"
                out.write_text(f"{head}: {EXAMPLE.name} tl.{name}, tile {tile}\n{text}\n")
                print(f"{out.name}: {len(text.splitlines())} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
