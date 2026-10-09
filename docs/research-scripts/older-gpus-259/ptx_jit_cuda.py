"""Run the CUDA C++ examples' older-arch code path on this GPU through PTX JIT (issue #259).

``TORCH_CUDA_ARCH_LIST="8.6+PTX"`` (or ``"7.5+PTX"``) makes ``load_inline`` build
``-gencode arch=compute_86,code=[sm_86,compute_86]``: sm_86 SASS, which an sm_120 GPU cannot
load, and compute_86 PTX, which its driver JIT-compiles. The kernels then run their
``__CUDA_ARCH__ == 860`` (or 750) code path: correctness only, the SASS is this GPU's.

Two steps, so the GPU lock is held only for the run:

    # build (CPU): compile each example's extension, list what its .so embeds (cuobjdump)
    CUDA_VISIBLE_DEVICES= TORCH_CUDA_ARCH_LIST=8.6+PTX TORCH_EXTENSIONS_DIR=<scratch>/ext86 \
        PYTHONPATH=src python docs/research-scripts/older-gpus-259/ptx_jit_cuda.py build
    # run (GPU): each example against nn.Linear on random bf16 inputs (rel L2 error)
    flock ~/.cache/kernel-agent/gpu.lock env KERNEL_AGENT_LOCK_HELD=1 TORCH_CUDA_ARCH_LIST=8.6+PTX \
        TORCH_EXTENSIONS_DIR=<scratch>/ext86 PYTHONPATH=src \
        python docs/research-scripts/older-gpus-259/ptx_jit_cuda.py run
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from kernel_agent import toolchain  # noqa: E402

EXAMPLES = ROOT / "src" / "kernel_agent" / "agent" / "examples"
ARCH_LIST = os.environ["TORCH_CUDA_ARCH_LIST"]
#: example -> (in, out features, row counts); the skinny GEMM only for sm_80+ code paths
CASES = {
    "cuda_int8_gemv": (2048, 12288, (1, 4)),  # INT8 weight-only (prmt + FADD dequant)
    "cuda_fp8_gemv": (2048, 12288, (1, 4)),  # e4m3 weight-only (software cvt below sm_89)
    "cuda_int8_skinny_gemm": (2048, 12288, (3, 16)),  # INT8 W8A8, IMMA mma.sync m16n8k32
}


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_{ARCH_LIST}", EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def names() -> list[str]:
    major = int(ARCH_LIST.split(".")[0])
    return [n for n in CASES if not (n == "cuda_int8_skinny_gemm" and major < 8)]


def build() -> None:
    print(f"# TORCH_CUDA_ARCH_LIST={ARCH_LIST}")
    for name in names():
        module = load(name)
        ext = module._load()
        ninja = Path(ext.__file__).parent / "build.ninja"
        flags = sorted(set(re.findall(r"-gencode=?\s*arch=\S+", ninja.read_text())))
        print(f"{name}: built {Path(ext.__file__).name}; nvcc {' '.join(flags)}")


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def run() -> None:
    cap = torch.cuda.get_device_capability()
    print(f"# GPU {torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]}; TORCH_CUDA_ARCH_LIST={ARCH_LIST}")
    torch.manual_seed(0)
    for name in names():
        k, n, rows = CASES[name]
        module = load(name)
        ref = nn.Linear(k, n, bias=False).cuda().to(torch.bfloat16).eval()
        with torch.no_grad():
            cand = module.build(ref)
            if cand is ref:
                print(f"{name}: build() returned the reference")
                continue
            errs = []
            for m in rows:
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                errs.append(f"M={m}: {rel(cand(x), ref(x)):.4f}")
            torch.cuda.synchronize()
        print(f"{name}: ran through compute_{ARCH_LIST.split('+')[0].replace('.', '')} PTX; rel L2 vs nn.Linear {', '.join(errs)}")


if __name__ == "__main__":
    toolchain.setup()  # CUDA_HOME, NVCC_APPEND_FLAGS, ninja (keeps TORCH_CUDA_ARCH_LIST)
    os.environ["TORCH_CUDA_ARCH_LIST"] = ARCH_LIST
    build() if sys.argv[1:] == ["build"] else run()
