"""How much of the dev-GPU Bash time was compilation: union of build-step intervals from
every .ninja_log under ~/.cache/torch_extensions and ~/.cache/kernel-agent, intersected with
the sessions' dev intervals. The log's mtime field matches the step's start (checked against
tool-call times: a 64 s Bash call 15:21:16-15:22:20 covers steps logged at 15:21:18 + 60 s), so
a step is [mtime, mtime + (end_ms - start_ms)]."""
import glob, json
from pathlib import Path
D = Path(__file__).resolve().parent
S = json.load(open(D / "sessions.json"))
steps = []
for f in glob.glob(str(Path.home() / ".cache/torch_extensions/**/.ninja_log"), recursive=True) + glob.glob(
    str(Path.home() / ".cache/kernel-agent/**/.ninja_log"), recursive=True
):
    for line in open(f):
        if line.startswith("#"):
            continue
        p = line.split("\t")
        if len(p) < 4:
            continue
        a, b, m = int(p[0]), int(p[1]), int(p[2])
        start = m / 1e9
        steps.append((start, start + (b - a) / 1000))

def union(iv):
    o = []
    for a, b in sorted(iv):
        if o and a <= o[-1][1]:
            o[-1][1] = max(o[-1][1], b)
        else:
            o.append([a, b])
    return o

def inter(x, y):
    tot = 0.0
    j = 0
    for a, b in x:
        for c, d in y:
            lo, hi = max(a, c), min(b, d)
            if hi > lo:
                tot += hi - lo
    return tot

B = union(steps)
print("ninja build steps:", len(steps), "union hours:", round(sum(b - a for a, b in B) / 3600, 2))
rows = {}
for s in S:
    if not s.get("dev_iv"):
        continue
    lo, hi = s["start"], s["end"]
    Bs = [(max(a, lo), min(b, hi)) for a, b in B if b > lo and a < hi]
    dev = sum(b - a for a, b in s["dev_iv"])
    c_dev = inter(s["dev_iv"], Bs)
    c_lock = inter(s.get("lock_iv", []), Bs)
    c_all = sum(b - a for a, b in Bs)
    r = rows.setdefault(s["role"], [0, 0, 0, 0])
    r[0] += dev; r[1] += c_dev; r[2] += c_lock; r[3] += c_all
out = {}
for k, (dev, cd, cl, ca) in rows.items():
    out[k] = {"dev_h": round(dev / 3600, 2), "compile_in_dev_h": round(cd / 3600, 2), "share": round(cd / dev, 3) if dev else None, "compile_in_lock_h": round(cl / 3600, 3), "compile_in_session_h": round(ca / 3600, 2)}
    print(k, out[k])
tot = [sum(v[i] for v in rows.values()) for i in range(4)]
print("all", round(tot[0] / 3600, 2), round(tot[1] / 3600, 2), round(tot[1] / tot[0], 3))
(D / "compile_share.json").write_text(json.dumps({"by_role": out, "all_share": tot[1] / tot[0]}, indent=1))
