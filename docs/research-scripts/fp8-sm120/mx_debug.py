import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from cutlass_bw import load  # noqa: E402

c = load()
import os
M, N, K = int(os.environ.get("MM", 352)), 8192, 1024
x = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
w = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn)
na, nb = c.mx_sf_sizes(M, N, K)
print("sf sizes", na, nb)
a = torch.full((na,), 127, device="cuda", dtype=torch.uint8)
b = torch.full((nb,), 127, device="cuda", dtype=torch.uint8)
y = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
try:
    c.mx_coop128(x, w, a, b, y)
    torch.cuda.synchronize()
    ref = x.float() @ w.float().T
    print("ok, rel err", float((y.float() - ref).norm() / ref.norm()))
except Exception as e:  # noqa: BLE001
    print(str(e)[:400])
