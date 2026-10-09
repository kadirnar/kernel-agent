"""bf16 / fp16 arithmetic from the CUDA headers below sm_80 (issue #259), compiled with the
toolchain's nvcc (what ``load_inline`` uses). CPU only. Run:

    PYTHONPATH=src python docs/research-scripts/older-gpus-259/nvcc_bf16_ops.py \
        > docs/research-scripts/older-gpus-259/nvcc_bf16_ops.txt
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from kernel_agent import toolchain  # noqa: E402

SRC = r"""
#include <cuda_bf16.h>
#include <cuda_fp16.h>
__global__ void k(__nv_bfloat16* p, __half* q, float* f) {
  __nv_bfloat16 x = p[threadIdx.x];
  __half y = q[threadIdx.x];
#if defined(OP_BF16_HFMA)
  x = __hfma(x, x, x);
#elif defined(OP_BF16_HADD)
  x = __hadd(x, x);
#elif defined(OP_BF16_OPERATOR)
  x = x * x + x;
#elif defined(OP_BF16_HFMA2)
  __nv_bfloat162 v = __halves2bfloat162(x, x);
  v = __hfma2(v, v, v);
  x = __low2bfloat16(v);
#elif defined(OP_BF16_VIA_FLOAT)
  float t = __bfloat162float(x);
  x = __float2bfloat16(t * t + t);
#elif defined(OP_FP16_HFMA2)
  __half2 w = __halves2half2(y, y);
  w = __hfma2(w, w, w);
  y = __low2half(w);
#endif
  p[threadIdx.x] = x;
  q[threadIdx.x] = y;
}
"""
OPS = (
    "OP_BF16_HFMA",
    "OP_BF16_HADD",
    "OP_BF16_OPERATOR",
    "OP_BF16_HFMA2",
    "OP_BF16_VIA_FLOAT",
    "OP_FP16_HFMA2",
)


def main() -> None:
    toolchain.setup()
    nvcc = Path(os.environ["CUDA_HOME"]) / "bin" / "nvcc"
    version = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
    print("#", version.strip().splitlines()[-1])
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "ops.cu"
        src.write_text(SRC)
        for op in OPS:
            for arch in ("sm_75", "sm_80"):
                cmd = [nvcc, "-std=c++17", f"-D{op}", f"-arch={arch}", "-cubin", "-o"]
                done = subprocess.run(
                    [*cmd, str(Path(tmp) / "k.cubin"), str(src)], capture_output=True, text=True
                )
                if done.returncode == 0:
                    print(f"{op:20s} {arch}: OK")
                else:
                    first = next((x for x in done.stderr.splitlines() if "error" in x), "")
                    print(f"{op:20s} {arch}: FAIL {first.split('error', 1)[-1][:140]}")


if __name__ == "__main__":
    main()
