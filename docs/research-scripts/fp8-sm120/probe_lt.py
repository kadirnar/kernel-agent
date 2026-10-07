"""cuBLASLt (direct) scale-mode support on sm_120: heuristic status / algo count per mode."""

import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lt_ext import load  # noqa: E402

ext = load()
torch.cuda.init()
print("cublasLt version", ext.version(), torch.cuda.get_device_name())
MODES = {
    "tensorwise (SCALAR_32F x SCALAR_32F)": (0, 0),
    "outer-vector (OUTER_VEC_32F x OUTER_VEC_32F)": (3, 3),
    "DeepSeek blockwise (W BLK128x128_32F, X VEC128_32F)": (5, 4),
    "1x128 both (VEC128_32F x VEC128_32F)": (4, 4),
    "MXFP8 (VEC32_UE8M0 x VEC32_UE8M0)": (2, 2),
    "W SCALAR, X OUTER_VEC": (0, 3),
}
SHAPES = [
    (352, 2560, 1024),
    (352, 1024, 2048),
    (352, 8192, 1024),
    (352, 1024, 4096),
    (22, 8192, 1024),
    (16, 12288, 2048),
    (1, 12288, 2048),
]
for name, (ma, mb) in MODES.items():
    row = []
    for M, N, K in SHAPES:
        status, n = ext.probe(M, N, K, ma, mb, 1)
        row.append(f"{M}x{N}x{K}: " + (f"{n} algos" if status == 0 else f"status {status}"))
    print(f"{name}\n   " + "; ".join(row))
