"""Does tl.dot_scaled with constant (tl.full) unit scales lower to QMMA.SF on sm_120? (#135)"""

import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench_fp8 import FP8, R, graph_time, mxfp8_ptr  # noqa: E402


@triton.jit
def mx_const(a, b, sa, sb, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM)
    npn = tl.cdiv(N, BN)
    group = GM * npn
    first_m = (pid // group) * GM
    group_m = min(npm - first_m, GM)
    pm = first_m + (pid % group) % group_m
    pn = (pid % group) // group_m
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptr = a + rm[:, None] * K + rk[None, :]
    b_ptr = b + rn[None, :] * K + rk[:, None]
    row_ok = rm[:, None] < M
    ones_a = tl.full((BM, BK // 32), 127, tl.uint8)
    ones_b = tl.full((BN, BK // 32), 127, tl.uint8)
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(0, K, BK):
        acc = tl.dot_scaled(tl.load(a_ptr, mask=row_ok, other=0.0), ones_a, "e4m3", tl.load(b_ptr), ones_b, "e4m3", acc)
        a_ptr += BK
        b_ptr += BK
    acc = acc * tl.load(sa + rm, mask=rm < M, other=0.0)[:, None] * tl.load(sb + rn)[None, :]
    tl.store(c + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=row_ok)


M, N, K = 352, 8192, 1024
x = torch.randn(M, K, device="cuda")
xs = (x.abs().amax(1, keepdim=True) / 448).clamp(min=1e-12)
xq = (x / xs).to(FP8)
ws = [torch.randn(N, K, device="cuda") for _ in range(R)]
wss = [(w.abs().amax(1, keepdim=True) / 448).clamp(min=1e-12) for w in ws]
wqs = [(w / s).to(FP8) for w, s in zip(ws, wss)]
outs = [torch.empty(M, N, device="cuda", dtype=torch.bfloat16) for _ in range(R)]
grid = (triton.cdiv(M, 64) * (N // 64),)
k = mx_const[grid](xq, wqs[0], xs.reshape(M), wss[0].reshape(N), outs[0], M, N, K, BM=64, BN=64, BK=128, GM=8,
                   num_warps=4, num_stages=3)
ref = torch._scaled_mm(xq, wqs[0].t(), scale_a=xs, scale_b=wss[0].reshape(1, N), out_dtype=torch.bfloat16)
print("block_scale in PTX:", "block_scale" in k.asm["ptx"], "bit-identical to row-wise _scaled_mm:",
      torch.equal(outs[0], ref))
warm = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
for _ in range(200):
    warm @ warm
fns = [lambda w=w, s=s, o=o: mx_const[grid](xq, w, xs.reshape(M), s.reshape(N), o, M, N, K, BM=64, BN=64, BK=128,
                                            GM=8, num_warps=4, num_stages=3) for w, s, o in zip(wqs, wss, outs)]
sa = torch.full((M, K // 32), 127, device="cuda", dtype=torch.uint8)
sb = torch.full((N, K // 32), 127, device="cuda", dtype=torch.uint8)
fns2 = [lambda w=w, o=o: mxfp8_ptr[grid](xq, w, sa, sb, o, M, N, K, BM=64, BN=64, BK=128, GM=8, num_warps=4,
                                         num_stages=3) for w, o in zip(wqs, outs)]
for _ in range(3):
    print("const scales + row/col epilogue %.2f us | loaded unit scales %.2f us" % (graph_time(fns), graph_time(fns2)))
