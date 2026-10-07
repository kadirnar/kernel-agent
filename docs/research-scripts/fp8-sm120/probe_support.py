"""Which FP8 scaling recipes torch 2.14's scaled_mm runs on this GPU (sm_120), and are
their results the fp32 math of the dequantised operands?"""

import math

import torch
import torch.nn.functional as F
from torch.nn.functional import ScalingType as ST
from torch.nn.functional import SwizzleType as SW

torch.manual_seed(0)
dev = "cuda"
E4M3 = torch.float8_e4m3fn
print(torch.cuda.get_device_name(), torch.cuda.get_device_capability(), torch.__version__)


def ceil_div(a, b):
    return (a + b - 1) // b


def to_blocked(m):
    """[rows, cols] scale matrix -> cuBLAS / CUTLASS 128x4 blocked layout (flattened)."""
    rows, cols = m.shape
    nrb, ncb = ceil_div(rows, 128), ceil_div(cols, 4)
    p = torch.zeros(nrb * 128, ncb * 4, device=m.device, dtype=m.dtype)
    p[:rows, :cols] = m
    blocks = p.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def q_rows(x, group):
    """x [R, K] -> codes e4m3 [R, K], fp32 scales [R, K // group] (amax / 448 per group)."""
    R, K = x.shape
    g = x.float().view(R, K // group, group)
    s = g.abs().amax(-1).clamp_min(1e-12) / 448.0
    q = (g / s[..., None]).clamp(-448, 448).to(E4M3).view(R, K)
    return q, s


def q_blocks(w, b=128):
    """w [N, K] -> codes, fp32 scales [N // b, K // b] (amax / 448 per b x b block)."""
    N, K = w.shape
    g = w.float().view(N // b, b, K // b, b)
    s = g.abs().amax(dim=(1, 3)).clamp_min(1e-12) / 448.0
    q = (g / s[:, None, :, None]).clamp(-448, 448).to(E4M3).view(N, K)
    return q, s


def q_mx(x):
    """MXFP8: e4m3 codes, e8m0 power-of-two scale per 32 (OCP: 2^(floor(log2 amax) - 8))."""
    R, K = x.shape
    g = x.float().view(R, K // 32, 32)
    amax = g.abs().amax(-1).clamp_min(2.0**-126)
    e = torch.floor(torch.log2(amax)) - 8  # e4m3 emax = 8
    s = torch.exp2(e)
    q = (g / s[..., None]).clamp(-448, 448).to(E4M3).view(R, K)
    return q, s.to(torch.float8_e8m0fnu)


def ref_mm(xd, wd):
    return (xd.double() @ wd.double().T).float()


def report(name, fn, ref):
    try:
        y = fn()
        torch.cuda.synchronize()
        err = (y.float() - ref).norm() / ref.norm()
        print(f"  {name:44s} OK   rel err vs fp32 math of the codes {err:.2e}")
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).strip().splitlines()[0][:150]
        print(f"  {name:44s} FAIL {type(exc).__name__}: {msg}")


for M, N, K in [(352, 2560, 1024), (22, 2560, 1024), (16, 2048, 6144)]:
    print(f"M={M} N={N} K={K}")
    x = torch.randn(M, K, device=dev)
    w = torch.randn(N, K, device=dev) * 0.02
    # tensor-wise
    sx, sw = x.abs().max() / 448, w.abs().max() / 448
    xq, wq = (x / sx).to(E4M3), (w / sw).to(E4M3)
    ref = ref_mm(xq.float() * sx, wq.float() * sw)
    report(
        "_scaled_mm tensorwise",
        lambda: torch._scaled_mm(xq, wq.t(), scale_a=sx, scale_b=sw, out_dtype=torch.bfloat16),
        ref,
    )
    # row-wise
    xr, xs = q_rows(x, K)
    wr, ws = q_rows(w, K)
    ref = ref_mm(xr.float() * xs, wr.float() * ws)
    report(
        "_scaled_mm rowwise",
        lambda: torch._scaled_mm(
            xr, wr.t(), scale_a=xs, scale_b=ws.view(1, N), out_dtype=torch.bfloat16
        ),
        ref,
    )
    # DeepSeek-V3 blockwise: activations 1x128, weights 128x128
    xb, xbs = q_rows(x, 128)
    wb, wbs = q_blocks(w, 128)
    xd = (xb.float().view(M, K // 128, 128) * xbs[..., None]).view(M, K)
    wd = (wb.float().view(N // 128, 128, K // 128, 128) * wbs[:, None, :, None]).view(N, K)
    ref = ref_mm(xd, wd)
    xbs_mmajor = xbs.t().contiguous().t()  # [M, K/128], M-major (outer-dim-major)
    for la, sa in (("row-major", xbs), ("M-major", xbs_mmajor)):
        for lb, sb in (("[N/128,K/128]", wbs), ("[K/128,N/128]", wbs.t())):
            report(
                f"F.scaled_mm 1x128 ({la}) x 128x128 {lb}",
                lambda sa=sa, sb=sb: F.scaled_mm(
                    xb,
                    wb.t(),
                    scale_a=sa,
                    scale_recipe_a=ST.BlockWise1x128,
                    scale_b=sb,
                    scale_recipe_b=ST.BlockWise128x128,
                    output_dtype=torch.bfloat16,
                ),
                ref,
            )
    # 1x128 x 1x128 (both per-group along K)
    wg, wgs = q_rows(w, 128)
    wd2 = (wg.float().view(N, K // 128, 128) * wgs[..., None]).view(N, K)
    ref = ref_mm(xd, wd2)
    report(
        "F.scaled_mm 1x128 x 1x128",
        lambda: F.scaled_mm(
            xb,
            wg.t(),
            scale_a=xbs_mmajor,
            scale_recipe_a=ST.BlockWise1x128,
            scale_b=wgs.t().contiguous().t(),
            scale_recipe_b=ST.BlockWise1x128,
            output_dtype=torch.bfloat16,
        ),
        ref,
    )
    # MXFP8: e8m0 per 32 on both, swizzled
    xm, xms = q_mx(x)
    wm, wms = q_mx(w)
    xd = (xm.float().view(M, K // 32, 32) * xms.float()[..., None]).view(M, K)
    wd = (wm.float().view(N, K // 32, 32) * wms.float()[..., None]).view(N, K)
    ref = ref_mm(xd, wd)
    report(
        "F.scaled_mm MXFP8 (1x32 e8m0, swizzled)",
        lambda: F.scaled_mm(
            xm,
            wm.t(),
            scale_a=to_blocked(xms),
            scale_recipe_a=ST.BlockWise1x32,
            scale_b=to_blocked(wms),
            scale_recipe_b=ST.BlockWise1x32,
            swizzle_a=SW.SWIZZLE_32_4_4,
            swizzle_b=SW.SWIZZLE_32_4_4,
            output_dtype=torch.bfloat16,
        ),
        ref,
    )
    report(
        "_scaled_mm MXFP8 (old API)",
        lambda: torch._scaled_mm(
            xm,
            wm.t(),
            scale_a=to_blocked(xms),
            scale_b=to_blocked(wms),
            out_dtype=torch.bfloat16,
        ),
        ref,
    )
    print(f"  (fp8 codes: {math.prod(x.shape)} activations)")
