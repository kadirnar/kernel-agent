"""Which concurrency APIs does this torch build expose (no GPU work)."""

import torch

import kernel_agent

print("kernel_agent:", kernel_agent.__file__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
names = [n for n in dir(torch.cuda) if any(s in n.lower() for s in ("green", "context", "partition"))]
print("torch.cuda names:", names)
gc = getattr(torch.cuda, "green_contexts", None)
print("torch.cuda.green_contexts:", gc)
if gc is not None:
    print([n for n in dir(gc) if not n.startswith("__")])
    if hasattr(gc, "GreenContext"):
        print(gc.GreenContext.__doc__)
        print([n for n in dir(gc.GreenContext) if not n.startswith("__")])
try:
    import cuda.bindings.driver as drv

    print("cuda.bindings", [n for n in dir(drv) if "GreenCtx" in n or "DevResource" in n][:40])
except Exception as exc:
    print("cuda.bindings:", exc)
try:
    import cuda.core

    print("cuda.core", cuda.core.__version__ if hasattr(cuda.core, "__version__") else "?")
except Exception as exc:
    print("cuda.core:", exc)
print("graph APIs:", [n for n in dir(torch.cuda) if "graph" in n.lower()])
