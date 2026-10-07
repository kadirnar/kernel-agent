"""Top kernels of one stage: calls, total, mean, grid, est. occupancy."""

import collections
import gzip
import pickle
import sys

import analyze

path, stage = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 15
d = pickle.load(gzip.open(path, "rb"))
names = d["names"]
ev = d["events"]
ann = [(s, e, names[i][7:]) for k, i, s, e, c, r, m in ev if k == "user_annotation"]
segs = analyze.flatten(ann)
import bisect

starts = [s for s, _, _ in segs]
launch = {c: s for k, i, s, e, c, r, m in ev if k in analyze.API}
agg = collections.defaultdict(lambda: [0, 0.0, None])
total = 0.0
for k, i, s, e, c, r, m in ev:
    if k not in analyze.GPU:
        continue
    t = launch.get(c, s)
    j = bisect.bisect_right(starts, t) - 1
    st = segs[j][2] if j >= 0 and segs[j][0] <= t < segs[j][1] else "outside"
    if st != stage:
        continue
    a = agg[names[i][:90]]
    a[0] += 1
    a[1] += (e - s) / 1e3
    a[2] = m
    total += (e - s) / 1e3
print(f"{stage}: {total / 1e3:.2f} ms GPU")
for name, (cnt, us, m) in sorted(agg.items(), key=lambda x: -x[1][1])[:n]:
    g = (m or {}).get("grid")
    occ = (m or {}).get("est. achieved occupancy %")
    print(f"{us / 1e3:8.2f} ms {100 * us / total:5.1f}% {cnt:6d} x {us / cnt:7.2f} us grid {g} occ {occ}  {name}")
