"""Gaps >= 2 us by (event before, stage of the next GPU work, launched by a graph?)."""

import bisect
import collections
import gzip
import pickle
import statistics
import sys

import analyze

d = pickle.load(gzip.open(sys.argv[1], "rb"))
names = d["names"]
ev = d["events"]
ann = [(s, e, names[i][7:]) for k, i, s, e, c, r, m in ev if k == "user_annotation"]
segs = analyze.flatten(ann)
starts = [s for s, _, _ in segs]
api = {c: (s, names[i]) for k, i, s, e, c, r, m in ev if k in analyze.API}


def stage(t):
    j = bisect.bisect_right(starts, t) - 1
    return segs[j][2] if j >= 0 and segs[j][0] <= t < segs[j][1] else "outside"


gpu = sorted((s, e, names[i], c) for k, i, s, e, c, r, m in ev if k in analyze.GPU)
agg = collections.defaultdict(list)
last_end, last = None, None
for s, e, n, c in gpu:
    if last_end is not None and s - last_end >= 2000:
        a = api.get(c)
        how = "graph" if a and "Graph" in a[1] else "launch"
        before = ("DtoH" if "DtoH" in last else "HtoD" if "HtoD" in last else
                  "DtoD" if "DtoD" in last else "kernel")
        nxt = stage(a[0]) if a else "?"
        # host lateness: the next work's launch call started after the GPU went idle?
        late = a is not None and a[0] > last_end
        agg[(before, nxt, how, "host late" if late else "host early")].append((s - last_end) / 1e3)
    if last_end is None or e >= last_end:
        last_end, last = e, n
rows = sorted(agg.items(), key=lambda x: -sum(x[1]))
for (b, nx, how, late), gs in rows[:14]:
    print(f"{sum(gs) / 1e3:7.2f} ms {len(gs):5d} gaps median {statistics.median(gs):7.1f} us | "
          f"after {b:6s} -> {nx:20s} via {how:6s} {late}")
