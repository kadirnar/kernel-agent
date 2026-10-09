"""Run Triton kernels with an older arch's code path on this GPU (issue #259). GPU.

Three patches make Triton compile for an older target and run it here:

* ``driver.active.get_current_target`` returns ``GPUTarget("cuda", <cc>, 32)``: the JIT
  compiles for that arch (its ``tl.dot`` lowering, its fp8 types);
* ``CompiledKernel._init_handles`` hands the driver the PTX instead of the cubin (an sm_75 /
  sm_8x cubin does not load on sm_120; PTX is JIT-compiled for this GPU), and refuses a
  kernel whose shared memory exceeds the emulated arch's per-block limit;
* ``torch.cuda.get_device_capability`` returns the old capability, so the examples' own
  checks behave as on that GPU.

Every launched kernel's PTX is summarised (``mma.sync`` forms, ``cp.async`` count) next to
the error against the reference. Correctness only: the SASS is this GPU's. Run:

    flock ~/.cache/kernel-agent/gpu.lock env KERNEL_AGENT_LOCK_HELD=1 PYTHONPATH=src \
        python docs/research-scripts/older-gpus-259/ptx_jit_triton.py 89 \
        >> docs/research-scripts/older-gpus-259/ptx_jit_triton.txt
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

CC = int(sys.argv[1])
CAP = (CC // 10, CC % 10)

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from torch import nn  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler.compiler import CompiledKernel  # noqa: E402
from triton.runtime import driver  # noqa: E402

#: opt-in shared memory per block (CUDA Programming Guide; tuning guides)
LIMIT = {75: 64 * 1024, 80: 163 * 1024, 86: 99 * 1024, 89: 99 * 1024}[CC]
EXAMPLES = ROOT / "src" / "kernel_agent" / "agent" / "examples"
REAL = torch.cuda.get_device_capability()
SEEN: dict[str, str] = {}

driver.active.get_current_target = lambda: GPUTarget("cuda", CC, 32)
_init_handles = CompiledKernel._init_handles


def _init(self):
    if getattr(self, "module", None) is None and isinstance(self.kernel, bytes):
        ptx = self.asm["ptx"]
        assert f".target sm_{CC}" in ptx, "not compiled for the emulated arch"
        forms = sorted({m.replace("row.col.", "") for m in re.findall(r"mma\.sync\.aligned\.([\w.]+)", ptx)})
        cp_async = len(re.findall(r"cp\.async\.c[ag]", ptx))
        SEEN[self.name] = f"{'/'.join(forms) or 'no mma (FMA)'}, cp.async {cp_async}, smem {self.metadata.shared}"
        if self.metadata.shared > LIMIT:
            raise RuntimeError(f"shared memory {self.metadata.shared} B > sm_{CC} limit {LIMIT}")
        self.kernel = ptx.encode() + b"\0"  # the driver JIT-compiles PTX
    return _init_handles(self)


CompiledKernel._init_handles = _init
torch.cuda.get_device_capability = lambda *a, **k: CAP


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@triton.jit
def gemm(a, b, c, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        x = tl.load(a + rm[:, None] * K + (k + rk)[None, :])
        y = tl.load(b + rn[None, :] * K + (k + rk)[:, None])  # b: [N, K]
        acc = tl.dot(x, y, acc)
    tl.store(c + rm[:, None] * (tl.num_programs(1) * BN) + rn[None, :], acc)


def report(name: str, fn) -> None:
    SEEN.clear()
    try:
        with torch.no_grad():
            message = fn()
        torch.cuda.synchronize()
        kernels = "; ".join(f"{k}: {v}" for k, v in SEEN.items())
        print(f"sm_{CC} {name:28s} {message} [{kernels}]", flush=True)
    except Exception as exc:
        text = " ".join(str(exc).split())
        print(f"sm_{CC} {name:28s} FAIL {type(exc).__name__}: {text[:200]}", flush=True)


def main() -> None:
    print(f"# real GPU sm_{REAL[0]}{REAL[1]}, emulated sm_{CC}; Triton {triton.__version__}")
    torch.manual_seed(0)
    for dtype in (torch.float16, torch.bfloat16):
        tag = str(dtype).split(".")[1]

        def dot(dtype=dtype):
            a = torch.randn(256, 512, device="cuda", dtype=dtype)
            b = torch.randn(256, 512, device="cuda", dtype=dtype)
            c = torch.empty(256, 256, device="cuda", dtype=torch.float32)
            gemm[(4, 4)](a, b, c, 512, BM=64, BN=64, BK=64, num_warps=4, num_stages=3)
            return f"rel L2 {rel(c, a.float() @ b.float().T):.2e}"

        report(f"tl.dot GEMM [{tag}]", dot)

        def rmsnorm(dtype=dtype):
            from kernel_agent.selftest import RMSNorm

            ref = RMSNorm(2048, 1e-5).cuda().to(dtype)
            x = torch.randn(4, 64, 2048, device="cuda", dtype=dtype)
            return f"rel L2 {rel(load('triton_rmsnorm').build(ref)(x), ref(x)):.4f}"

        report(f"triton_rmsnorm [{tag}]", rmsnorm)

        def attention(dtype=dtype):
            module = load("triton_short_attention")
            q = torch.randn(32, 16, 11, 128, device="cuda", dtype=dtype)
            k = torch.randn(32, 2, 11, 128, device="cuda", dtype=dtype)
            v = torch.randn_like(k)
            got = module.sdpa(q, k, v, is_causal=True, enable_gqa=True)
            want = torch.nn.functional.scaled_dot_product_attention(
                q.float(), k.float(), v.float(), is_causal=True, enable_gqa=True
            )
            return f"rel L2 vs fp32 SDPA {rel(got, want):.4f}"

        report(f"triton_short_attention [{tag}]", attention)

    def int8():
        ref = nn.Linear(1024, 8192, bias=False).cuda().to(torch.bfloat16)
        cand = load("triton_int8_w8a8_gemm").build(ref)
        if cand is ref:
            return "build() returned the reference (its capability check)"
        x = torch.randn(704, 1024, device="cuda", dtype=torch.bfloat16)
        return f"rel L2 vs nn.Linear {rel(cand(x), ref(x)):.4f}"

    report("triton_int8_w8a8_gemm [bf16]", int8)

    def fp8():
        module = load("triton_fp8_w8a8_gemm")
        ref = nn.Linear(1024, 8192, bias=False).cuda().to(torch.bfloat16)
        cand = module.build(ref)
        if cand is ref:
            return "build() returned the reference (its capability check)"
        x = torch.randn(704, 1024, device="cuda", dtype=torch.bfloat16)
        return f"rel L2 vs nn.Linear {rel(cand(x), ref(x)):.4f}"

    report("triton_fp8_w8a8_gemm [bf16]", fp8)


if __name__ == "__main__":
    main()
