"""CUTLASS SM120 MXFP8 (block-scaled MMA, QMMA.SF) GEMM, 128x128x128 tiles, unit ue8m0
scales: correctness (equals the fp32 product of the e4m3 codes) and L2-cold graph timing
next to cuBLASLt (tensorwise nvjet, MXFP8)."""

import math
import sys

import torch
import torch.nn.functional as F
from torch.nn.functional import ScalingType as ST
from torch.nn.functional import SwizzleType as SW

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from cutlass_bw import load as load_cutlass  # noqa: E402
from gemm_bench import graph_us, to_blocked  # noqa: E402
from lt_ext import load  # noqa: E402

from kernel_agent.kernels.bench import ensure_clocks, warm_gpu

E4M3 = torch.float8_e4m3fn
cut = load_cutlass()
ext = load()
warm_gpu()
print("clock state", ensure_clocks())
one = torch.ones((), device="cuda")
for M, N, K, label in ((352, 2560, 1024, "LocDiT q|k|v"), (352, 1024, 2048, "LocDiT o_proj"),
                       (352, 8192, 1024, "LocDiT gate|up"), (352, 1024, 4096, "LocDiT down"),
                       (8192, 8192, 8192, "8192^3")):
    copies = max(3, math.ceil(160e6 / (N * K)))
    x = (torch.randn(M, K, device="cuda") * 0.5).to(E4M3)
    ws = [(torch.randn(N, K, device="cuda") * 0.05).to(E4M3) for _ in range(copies)]
    y = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    na, nb = cut.mx_sf_sizes(M, N, K)
    sfa = torch.full((na,), 127, device="cuda", dtype=torch.uint8)
    sfb = torch.full((nb,), 127, device="cuda", dtype=torch.uint8)
    ref = x.float() @ ws[0].float().T
    sxm = to_blocked(torch.ones(M, K // 32, device="cuda").to(torch.float8_e8m0fnu))
    swm = to_blocked(torch.ones(N, K // 32, device="cuda").to(torch.float8_e8m0fnu))
    ext.tune(ws[0], x, one, one, y, 0, 0, 1, 10, 3)
    row = [f"{label:16s}"]
    for name, fn in (
        ("cuBLASLt tensorwise", lambda i: ext.run(ws[i], x, one, one, y, 0, 0, 1)),
        ("cuBLASLt MXFP8", lambda i: F.scaled_mm(
            x, ws[i].t(), scale_a=sxm, scale_recipe_a=ST.BlockWise1x32, scale_b=swm,
            scale_recipe_b=ST.BlockWise1x32, swizzle_a=SW.SWIZZLE_32_4_4,
            swizzle_b=SW.SWIZZLE_32_4_4, output_dtype=torch.bfloat16)),
        ("CUTLASS MXFP8 coop 128x128x128", lambda i: cut.mx_coop128(x, ws[i], sfa, sfb, y)),
        ("CUTLASS MXFP8 auto 128x128x128", lambda i: cut.mx_ping64(x, ws[i], sfa, sfb, y)),
    ):
        try:
            out = fn(0)
            torch.cuda.synchronize()
            res = out if isinstance(out, torch.Tensor) else y
            err = float((res.float() - ref).norm() / ref.norm())
            us = graph_us([lambda i=i, fn=fn: fn(i) for i in range(copies)], reps=1 if M == 8192 else 3)
            row.append(f"{name} {us:8.2f} us ({2 * M * N * K / us / 1e6:5.1f} TF, err {err:.0e})")
        except Exception as exc:  # noqa: BLE001
            row.append(f"{name} {type(exc).__name__}: {str(exc).splitlines()[0][:60]}")
    print(" | ".join(row))
    del ws
    torch.cuda.empty_cache()
