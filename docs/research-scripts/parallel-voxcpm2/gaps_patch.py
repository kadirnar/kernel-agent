"""Idle GPU time per steady patch, split by what follows a device->host copy (host syncs)."""

import bisect
import collections
import gzip
import pickle
import statistics
import sys

d = pickle.load(gzip.open(sys.argv[1], "rb"))
names = d["names"]
ev = d["events"]
GPU = {"kernel", "gpu_memcpy", "gpu_memset"}
gpu = sorted((s, e, names[i], k) for k, i, s, e, c, r, m in ev if k in GPU)
dits = sorted(s for k, i, s, e, c, r, m in ev if k == "user_annotation" and names[i] == "stage::dit")
launch = {c: s for k, i, s, e, c, r, m in ev if k in {"cuda_runtime", "cuda_driver"}}
# patch of a GPU event: by its own start relative to the first GPU work of each patch is hard;
# use GPU start times between consecutive DtoH copies instead: report every gap >= 2 us with
# the event before it
per_kind = collections.Counter()
per_kind_n = collections.Counter()
last_end, last = None, None
for s, e, n, k in gpu:
    if last_end is not None and s - last_end >= 2000:
        key = ("after DtoH copy (host sync)" if "DtoH" in last else
               "after HtoD copy" if "HtoD" in last else
               "after DtoD copy" if "DtoD" in last else "after a kernel")
        per_kind[key] += (s - last_end) / 1e6
        per_kind_n[key] += 1
    if last_end is None or e >= last_end:
        last_end, last = e, n
for k, v in per_kind.most_common():
    print(f"{k}: {per_kind_n[k]} gaps, {v:.2f} ms")
