"""#133 / #144 / #145: FP8 GEMM paths at transformer shapes on this GPU (eager and CUDA-graph timing)."""
import importlib.util, json, sys, time
import torch
import torch.nn as nn
from kernel_agent import toolchain
from kernel_agent.agent.prompts import EXAMPLES_DIR
toolchain.setup()
torch.manual_seed(0)

def load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES_DIR / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def timed(fn, x, graph):
    for _ in range(5): fn(x)
    torch.cuda.synchronize()
    if graph:
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): fn(x)
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            for _ in range(20): fn(x)
        run = g.replay; per = 20
    else:
        run = lambda: fn(x); per = 1
    for _ in range(3): run()
    torch.cuda.synchronize()
    ts = []
    for _ in range(15):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); [run() for _ in range(10)]; b.record(); b.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / (10 * per))
    ts.sort(); return ts[len(ts) // 2]

shapes = [(16, 2048, 6144), (352, 1024, 2560), (352, 1024, 8192), (352, 4096, 1024), (2048, 4096, 4096), (8192, 8192, 8192)]
names = {"cute_block_scaled": ("cute_fp8_blockscaled_gemm", {}), "cublaslt_tensorwise": ("cuda_cublaslt_fp8", {}),
         "cublaslt_mxfp8": ("cuda_cublaslt_fp8", {"mxfp8": 1}), "triton_dot_scaled": ("triton_fp8_w8a8_gemm", {}),
         "triton_mxfp8": ("triton_mxfp8_gemm", {}), "cuda_skinny": ("cuda_fp8_skinny_gemm", {})}
mods = {k: load(v[0]) for k, v in names.items()}
rows = []
for m, k, n in shapes:
    ref = nn.Linear(k, n, bias=False).cuda().to(torch.bfloat16)
    with torch.no_grad(): ref.weight.normal_(0, k ** -0.5)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    want = ref(x).float()
    flops = 2 * m * k * n
    cands = {"bf16_cublas": ref}
    wq = (ref.weight.float() / (ref.weight.float().abs().amax(1, keepdim=True) / 448)).to(torch.float8_e4m3fn)
    ws = (ref.weight.float().abs().amax(1, keepdim=True) / 448).t().contiguous()
    def rowwise(x):
        s = x.float().abs().amax(1, keepdim=True).clamp_min(1e-12) / 448
        return torch._scaled_mm((x.float() / s).to(torch.float8_e4m3fn), wq.t(), scale_a=s, scale_b=ws, out_dtype=torch.bfloat16)
    cands["torch_scaled_mm_rowwise"] = rowwise
    for label, (name, kw) in names.items():
        try:
            mod = mods[label].build(ref, **kw)
            if mod is ref: raise RuntimeError("build returned the reference (shape unsupported)")
            cands[label] = mod
        except Exception as e:
            rows.append({"shape": [m, k, n], "path": label, "error": f"{type(e).__name__}: {str(e)[:120]}"})
    for label, fn in cands.items():
        rec = {"shape": [m, k, n], "path": label}
        try:
            with torch.no_grad():
                out = fn(x).float()
                rec["rel_l2"] = round(float((out - want).norm() / want.norm()), 4)
                rec["eager_us"] = round(timed(fn, x, False), 2)
                try: rec["graph_us"] = round(timed(fn, x, True), 2)
                except Exception as e: rec["graph_error"] = f"{type(e).__name__}: {str(e)[:80]}"
                t = rec.get("graph_us") or rec["eager_us"]
                rec["tflops"] = round(flops / t / 1e6, 1)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        rows.append(rec); print(json.dumps(rec), flush=True)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
