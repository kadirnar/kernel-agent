"""SASS of Triton's e4m3 tl.dot and tl.dot_scaled kernels on sm_120 (#135)."""

import subprocess
import sys
import tempfile
from collections import Counter
import re

import torch
import triton

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench_fp8 import FP8, fp8_ptr, mxfp8_ptr  # noqa: E402

NVDISASM = triton.__file__.rsplit("/", 1)[0] + "/backends/nvidia/bin/nvdisasm"
M, N, K = 352, 8192, 1024
x = torch.randn(M, K, device="cuda").to(FP8)
w = torch.randn(N, K, device="cuda").to(FP8)
o = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
s1 = torch.ones(M, device="cuda")
s2 = torch.ones(N, device="cuda")
sa = torch.full((M, K // 32), 127, device="cuda", dtype=torch.uint8)
sb = torch.full((N, K // 32), 127, device="cuda", dtype=torch.uint8)
grid = (triton.cdiv(M, 64) * (N // 64),)
k1 = fp8_ptr[grid](x, w, s1, s2, o, M, N, K, BM=64, BN=64, BK=128, GM=8, num_warps=4, num_stages=3)
k2 = mxfp8_ptr[grid](x, w, sa, sb, o, M, N, K, BM=64, BN=64, BK=128, GM=8, num_warps=4, num_stages=3)
for name, k in [("tl.dot e4m3", k1), ("tl.dot_scaled e4m3/ue8m0", k2)]:
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(k.asm["cubin"])
        f.flush()
        sass = subprocess.run([NVDISASM, f.name], capture_output=True, text=True).stdout
    ops = Counter(re.findall(r"\b((?:HMMA|QMMA|OMMA|UTMALDG|LDSM|LDGSTS)[.A-Z0-9_]*)", sass))
    print(name, dict(ops))
