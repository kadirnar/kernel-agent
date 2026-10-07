import json
import os
import tempfile

import torch
from torch.profiler import ProfilerActivity, profile

a = torch.randn(256, 256, device="cuda")
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    (a @ a).sum()
    torch.cuda.synchronize()
for e in prof.profiler.kineto_results.events():
    if str(e.activity_type()) == "kernel":
        print(repr(e.metadata_json())[:300])
        print([m for m in dir(e) if "meta" in m])
        break
path = os.path.join(tempfile.mkdtemp(), "t.json")
prof.export_chrome_trace(path)
d = json.load(open(path))
for ev in d["traceEvents"]:
    if ev.get("cat") == "kernel":
        print(ev["args"])
        break
