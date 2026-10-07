"""Which cuBLASLt kernels run torch._scaled_mm (tensor-wise / row-wise) on sm_120 (#135)."""

import torch
from torch.profiler import ProfilerActivity, profile

FP8 = torch.float8_e4m3fn
one = torch.ones((), device="cuda")
for M, N, K in [(4096, 4096, 4096), (352, 8192, 1024)]:
    a = torch.randn(M, K, device="cuda").to(FP8)
    b = torch.randn(N, K, device="cuda").to(FP8)
    xs = torch.ones(M, 1, device="cuda")
    ws = torch.ones(1, N, device="cuda")
    for name, kw in [("tensorwise", dict(scale_a=one, scale_b=one)), ("rowwise", dict(scale_a=xs, scale_b=ws))]:
        torch._scaled_mm(a, b.t(), out_dtype=torch.bfloat16, **kw)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            torch._scaled_mm(a, b.t(), out_dtype=torch.bfloat16, **kw)
            torch.cuda.synchronize()
        print(M, N, K, name, [e.name for e in p.events() if e.device_type.name == "CUDA"])
