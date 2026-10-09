"""Which backends compile for which older arch (issue #259). CPU only, no GPU visible.

* CuTe DSL: its ``Arch`` targets, and ``examples/cute_rmsnorm.py`` compiled with
  ``CUTE_DSL_ARCH`` per arch (fake tensors).
* TileLang: ``examples/tilelang_rmsnorm.py`` and the skill's fp16 GEMM template
  (``T.Pipelined`` + ``T.gemm``) compiled per arch; the GEMM's CUDA source is searched for
  the tensor-core instruction and ``cp.async``.
* PyTorch SDPA: the arch ranges in the installed ``libtorch_cuda.so`` messages and which
  memory-efficient attention kernels it ships (dtype x arch).

Each compile runs in a child process (the target is read at import). Run:

    CUDA_VISIBLE_DEVICES= PYTHONPATH=src python docs/research-scripts/older-gpus-259/backends_per_arch.py \
        > docs/research-scripts/older-gpus-259/backends_per_arch.txt
"""

from __future__ import annotations

import collections
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "src" / "kernel_agent" / "agent" / "examples"
ARCHS = ("sm_75", "sm_80", "sm_86", "sm_89")

CUTE = r"""
import importlib.util, os, sys
import cutlass, cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
spec = importlib.util.spec_from_file_location("cute_rmsnorm", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
x = make_fake_compact_tensor(cutlass.Float16, (4, 2048), stride_order=(1, 0), assumed_align=16)
y = make_fake_compact_tensor(cutlass.Float16, (4, 2048), stride_order=(1, 0), assumed_align=16)
w = make_fake_compact_tensor(cutlass.Float16, (2048,), assumed_align=16)
try:
    cute.compile(m._rmsnorm, x, w, y, cutlass.Float32(1e-5),
                 make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")
    print("OK")
except Exception as e:
    print("FAIL", type(e).__name__, " ".join(str(e).split())[:200])
"""

TILELANG = r"""
import importlib.util, re, sys
from kernel_agent import toolchain
toolchain.setup()  # CUDA_HOME, NVCC_APPEND_FLAGS
import tilelang, tilelang.language as T
spec = importlib.util.spec_from_file_location("tl_rms", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
try:
    m._rmsnorm(16, 2048, 1e-5, "float16")
    print("rmsnorm fp16: OK")
except Exception as e:
    print("rmsnorm fp16: FAIL", type(e).__name__, " ".join(str(e).split())[-200:])

@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, bM=128, bN=128, bK=32, dtype="float16", acc="float"):
    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=128) as (bx, by):
            As = T.alloc_shared((bM, bK), dtype)
            Bs = T.alloc_shared((bK, bN), dtype)
            Cl = T.alloc_fragment((bM, bN), acc)
            T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K, bK), num_stages=3):
                T.copy(A[by * bM, k * bK], As)
                T.copy(B[k * bK, bx * bN], Bs)
                T.gemm(As, Bs, Cl)
            T.copy(Cl, C[by * bM, bx * bN])
    return main

for dtype in ("float16", "bfloat16"):
    try:
        k = matmul(1024, 1024, 1024, dtype=dtype)
        src = k.get_kernel_source()
        calls = sorted(set(re.findall(r"tl::(?:mma_sync<[^;(]{0,120}>|ptx_ldmatrix\w*|cp_async\w*)", src)))
        print(f"gemm {dtype}: OK; tl:: calls {calls or 'none (no tensor-core MMA)'}", flush=True)
    except Exception as e:
        print(f"gemm {dtype}: FAIL", type(e).__name__, " ".join(str(e).split())[-300:], flush=True)
"""


def child(code: str, env: dict, *args: str) -> str:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT / "src"), **env}
    with tempfile.TemporaryDirectory() as tmp:  # a file: TileLang's jit reads the source
        script = Path(tmp) / "child.py"
        script.write_text(code)
        done = subprocess.run(
            [sys.executable, str(script), *args],
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
    lines = [x for x in done.stdout.splitlines() if "[TileLang:" not in x]
    if done.returncode:
        lines.append(f"(exit {done.returncode}) " + " ".join(done.stderr.splitlines()[-2:])[-300:])
    return "\n".join(lines).strip()


def sdpa() -> None:
    import torch

    lib = Path(torch.__file__).parent / "lib" / "libtorch_cuda.so"
    text = subprocess.run(["strings", lib], capture_output=True, text=True).stdout
    print(f"\n## PyTorch {torch.__version__} SDPA ({lib.name})")
    for line in sorted(set(text.splitlines())):
        if "only supports gpu architectures" in line:
            print("message:", line.strip())
    kernels = collections.Counter(
        re.sub(r"_\d+x\d+(_[a-z0-9]+)?_", "_", k)
        for k in re.findall(r"fmha_cutlassF_(?:f16|bf16|f32)_[a-z]+_\d+x\d+(?:_[a-z0-9]+)?_sm\d+", text)
    )
    by_dtype: dict[str, set[str]] = collections.defaultdict(set)
    for name in kernels:
        dtype, arch = re.match(r"fmha_cutlassF_(\w+?)_\w+_(sm\d+)", name).groups()  # type: ignore[union-attr]
        by_dtype[dtype].add(arch)
    for dtype, archs in sorted(by_dtype.items()):
        print(f"memory-efficient forward kernels {dtype}: {sorted(archs)}")
    print(
        f"TF32 defaults: torch.backends.cuda.matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}"
        f", torch.backends.cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}"
    )
    print(f"fp16 max {torch.finfo(torch.float16).max}, bf16 max {torch.finfo(torch.bfloat16).max:.3e}")


def main() -> None:
    from importlib.metadata import version

    arches = child(
        "from cutlass.base_dsl.enums import Arch; print([a.name for a in Arch])", {}
    )
    print(f"## CuTe DSL {version('nvidia-cutlass-dsl')}: Arch targets {arches}")
    for arch in ARCHS:
        result = child(CUTE, {"CUTE_DSL_ARCH": arch}, str(EXAMPLES / "cute_rmsnorm.py"))
        print(f"cute_rmsnorm fp16 {arch}: {result}")
    print(f"\n## TileLang {version('tilelang')}")
    for arch in ARCHS:
        target = json.dumps({"kind": "cuda", "arch": arch})
        result = child(TILELANG, {"TILELANG_DEFAULT_TARGET": target}, str(EXAMPLES / "tilelang_rmsnorm.py"))
        for line in result.splitlines():
            print(f"{arch} {line}")
    sdpa()


if __name__ == "__main__":
    main()
