"""Offline analysis of one profiled run (common.profile_run): per-stage GPU busy / idle,
kernel counts per patch, idle gaps by cause, host lead (queue delay), SM fill.

    python analyze.py out/<tag>.pkl.gz [--skip 2]
"""

from __future__ import annotations

import bisect
import collections
import gzip
import json
import pickle
import statistics
import sys
from pathlib import Path

GPU = {"kernel", "gpu_memcpy", "gpu_memset"}
API = {"cuda_runtime", "cuda_driver"}
SMS = 70


def flatten(intervals):
    """Nested (start, end, label) ranges -> non-overlapping segments of the innermost label."""
    intervals = sorted(intervals, key=lambda x: (x[0], -x[1]))
    segs, stack, cur = [], [], None
    for s, e, label in intervals:
        while stack and stack[-1][1] <= s:
            top = stack.pop()
            if cur < top[1]:
                segs.append((cur, top[1], top[2]))
            cur = top[1]
        if stack and cur < s:
            segs.append((cur, s, stack[-1][2]))
        cur = s if cur is None or s > cur else cur
        stack.append((s, e, label))
    while stack:
        top = stack.pop()
        if cur < top[1]:
            segs.append((cur, top[1], top[2]))
        cur = max(cur, top[1])
    return segs


def union(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return out


def analyze(path: str, skip: int = 2) -> dict:
    with gzip.open(path, "rb") as fh:
        d = pickle.load(fh)
    names = d["names"]
    ev = d["events"]
    ann = [(s, e, names[i][7:]) for k, i, s, e, c, r, m in ev if k == "user_annotation"]
    run = [a for a in ann if a[2] == "run"]
    run_start = run[0][0]
    launches = {c: (s, e, r, names[i]) for k, i, s, e, c, r, m in ev if k in API}
    gpu = [(s, e, c, r, names[i], k, m) for k, i, s, e, c, r, m in ev if k in GPU]
    gpu.sort()
    # stage of every GPU event: innermost annotation around its launch
    segs = flatten([a for a in ann])
    starts = [s for s, _, _ in segs]

    def stage_at(t):
        j = bisect.bisect_right(starts, t) - 1
        if j >= 0 and segs[j][0] <= t < segs[j][1]:
            return segs[j][2]
        return "outside"

    # patches: the i-th "dit" range starts patch i
    dits = sorted(a[0] for a in ann if a[2] == "dit")
    rows = []
    for s, e, c, r, name, k, m in gpu:
        launch = launches.get(c)
        t = launch[0] if launch else s
        st = stage_at(t)
        if st == "run":
            st = "other"
        patch = bisect.bisect_right(dits, t) - 1  # -1: before the first patch (prefill)
        rows.append(dict(s=s, e=e, stage=st, patch=patch, name=name, kind=k, meta=m,
                         stream=r, launch=t, api=launch[3] if launch else None))
    end = max(r["e"] for r in rows)
    window = (end - run_start) / 1e6
    busy_iv = union([[r["s"], r["e"]] for r in rows])
    busy = sum(e - s for s, e in busy_iv) / 1e6
    kernel_sum = sum(r["e"] - r["s"] for r in rows) / 1e6
    streams = collections.Counter(r["stream"] for r in rows)

    # per stage totals
    per = collections.defaultdict(lambda: dict(n=0, kernels=0, gpu_ms=0.0, wsum=0.0, fill=0.0,
                                               occ=0.0, qdelay=[]))
    graph_launches = collections.Counter()
    for r in rows:
        p = per[r["stage"]]
        dur = (r["e"] - r["s"]) / 1e6
        p["n"] += 1
        p["gpu_ms"] += dur
        if r["kind"] == "kernel":
            p["kernels"] += 1
            m = r["meta"] or {}
            g = m.get("grid")
            if g:
                blocks = g[0] * g[1] * g[2]
                p["fill"] += min(1.0, blocks / SMS) * dur
                p["wsum"] += dur
                occ = m.get("est. achieved occupancy %")
                if occ is not None:
                    p["occ"] += float(occ) * dur
        p["qdelay"].append((r["s"] - r["launch"]) / 1e3)
        if r["api"] and "Graph" in r["api"]:
            graph_launches[r["stage"]] += 1
    # stage-instance spans: CPU range vs GPU span and idle inside
    inst = collections.defaultdict(list)
    for s, e, label in ann:
        if label == "run":
            continue
        inst[label].append((s, e))
    by_stage_rows = collections.defaultdict(list)
    for r in rows:
        by_stage_rows[r["stage"]].append(r)
    span_stats = {}
    for label, ranges in inst.items():
        rs = sorted(by_stage_rows.get(label, []), key=lambda r: r["launch"])
        launch_t = [r["launch"] for r in rs]
        cpu, span, idle, busyl = [], [], [], []
        for s, e in ranges:
            a, b = bisect.bisect_left(launch_t, s), bisect.bisect_right(launch_t, e)
            sub = rs[a:b]
            cpu.append((e - s) / 1e3)
            if not sub:
                continue
            g0, g1 = min(x["s"] for x in sub), max(x["e"] for x in sub)
            u = sum(y - x for x, y in union([[x["s"], x["e"]] for x in sub]))
            span.append((g1 - g0) / 1e3)
            busyl.append(u / 1e3)
            idle.append((g1 - g0 - u) / 1e3)
        if span:
            span_stats[label] = dict(calls=len(ranges), cpu_us=statistics.median(cpu),
                                     gpu_span_us=statistics.median(span),
                                     gpu_busy_us=statistics.median(busyl),
                                     idle_in_span_us=statistics.median(idle))
    # gaps between consecutive GPU work, by the stages around them
    gaps = []
    top = []
    rows_by_s = sorted(rows, key=lambda r: r["s"])
    last_end, last_stage, last_name = None, None, None
    for r in rows_by_s:
        if last_end is not None and r["s"] > last_end:
            g = (r["s"] - last_end) / 1e3
            gaps.append((g, last_stage, r["stage"]))
            top.append((g, (last_end - run_start) / 1e6, f"{last_stage}:{last_name[:40]}",
                        f"{r['stage']}:{r['name'][:40]}"))
        if last_end is None or r["e"] >= last_end:
            last_end, last_stage, last_name = r["e"], r["stage"], r["name"]
    top.sort(reverse=True)
    kdur = collections.defaultdict(list)
    for r in rows:
        if r["kind"] == "kernel":
            kdur[r["stage"]].append((r["e"] - r["s"]) / 1e3)
    kstats = {}
    for st, ds in kdur.items():
        tot = sum(ds)
        kstats[st] = dict(mean_us=round(tot / len(ds), 2), median_us=round(statistics.median(ds), 2),
                          share_time_lt10us=round(sum(x for x in ds if x < 10) / tot, 3) if tot else 0,
                          n_lt5us=sum(1 for x in ds if x < 5))
    first_gpu = rows_by_s[0]["s"]
    lead_in = (first_gpu - run_start) / 1e6
    hist = collections.OrderedDict(
        (k, [0, 0.0]) for k in ("<2us", "2-10us", "10-50us", "50-500us", ">500us"))
    pair = collections.defaultdict(lambda: [0, 0.0])
    for g, a, b in gaps:
        key = ("<2us" if g < 2 else "2-10us" if g < 10 else "10-50us" if g < 50 else
               "50-500us" if g < 500 else ">500us")
        hist[key][0] += 1
        hist[key][1] += g / 1e3
        if g >= 2:
            pair[f"{a} -> {b}"][0] += 1
            pair[f"{a} -> {b}"][1] += g / 1e3
    # steady-state per patch
    npatch = len(dits)
    pp = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0.0]))
    for r in rows:
        if r["patch"] >= 0:
            x = pp[r["patch"]][r["stage"]]
            x[0] += r["kind"] == "kernel"
            x[1] += (r["e"] - r["s"]) / 1e3
    steady = [p for p in range(skip, npatch - 1)]
    stages = sorted({st for p in steady for st in pp[p]})
    per_patch = {}
    for st in stages:
        ks = [pp[p][st][0] for p in steady]
        ms = [pp[p][st][1] for p in steady]
        per_patch[st] = dict(kernels=statistics.median(ks), gpu_us=statistics.median(ms))
    # wall per patch from dit starts (CPU) and from GPU
    dit_gpu_start = []
    for p in range(npatch):
        xs = [r["s"] for r in rows if r["patch"] == p]
        dit_gpu_start.append(min(xs) if xs else None)
    gpu_patch = [(dit_gpu_start[p + 1] - dit_gpu_start[p]) / 1e3 for p in steady
                 if dit_gpu_start[p + 1] and dit_gpu_start[p]]
    cpu_patch = [(dits[p + 1] - dits[p]) / 1e3 for p in steady]
    patch_busy = []
    for p in steady:
        iv = [[r["s"], r["e"]] for r in rows if r["patch"] == p]
        patch_busy.append(sum(e - s for s, e in union(iv)) / 1e3)
    summary = dict(
        tag=d["tag"], wall_ms_unprofiled_window=d["wall_ms"], window_ms=window, busy_ms=busy,
        kernel_sum_ms=kernel_sum, idle_ms=window - busy, busy_frac=busy / window,
        lead_in_ms=lead_in, gpu_events=len(rows),
        kernels=sum(1 for r in rows if r["kind"] == "kernel"), streams=dict(streams),
        patches=npatch,
        per_patch_wall_gpu_us=statistics.median(gpu_patch) if gpu_patch else None,
        per_patch_cpu_issue_us=statistics.median(cpu_patch) if cpu_patch else None,
        per_patch_busy_us=statistics.median(patch_busy) if patch_busy else None,
        stages={st: dict(events=p["n"], kernels=p["kernels"], gpu_ms=round(p["gpu_ms"], 3),
                         sm_fill=round(p["fill"] / p["wsum"], 3) if p["wsum"] else None,
                         occupancy_pct=round(p["occ"] / p["wsum"], 1) if p["wsum"] else None,
                         queue_delay_us_median=round(statistics.median(p["qdelay"]), 1),
                         graph_kernels=graph_launches.get(st, 0))
                for st, p in sorted(per.items(), key=lambda x: -x[1]["gpu_ms"])},
        stage_calls=span_stats,
        per_patch=per_patch,
        gaps={k: dict(count=v[0], ms=round(v[1], 3)) for k, v in hist.items()},
        top_gaps=[dict(us=round(g, 1), at_ms=round(t, 2), after=a, before=b) for g, t, a, b in top[:10]],
        kernel_durations=kstats,
        gap_pairs=dict(sorted(((k, dict(count=v[0], ms=round(v[1], 3))) for k, v in pair.items()),
                              key=lambda x: -x[1]["ms"])[:12]),
    )
    return summary


def table(s: dict) -> str:
    lines = [f"## {s['tag']}",
             f"window {s['window_ms']:.1f} ms (profiled; unprofiled-equivalent run wall "
             f"{s['wall_ms_unprofiled_window']:.1f} ms with profiler on), GPU busy "
             f"{s['busy_ms']:.1f} ms ({100 * s['busy_frac']:.1f} %), idle {s['idle_ms']:.1f} ms, "
             f"kernel time sum {s['kernel_sum_ms']:.1f} ms, {s['kernels']} kernels, "
             f"{s['patches']} patches, streams {s['streams']}, lead-in {s['lead_in_ms']:.2f} ms",
             f"steady patch: GPU {s['per_patch_wall_gpu_us']} us start-to-start, busy "
             f"{s['per_patch_busy_us']} us, host issue {s['per_patch_cpu_issue_us']} us", "",
             "| stage | kernels | GPU ms | SM fill | occ % | queue delay us | graph kernels |",
             "|---|---|---|---|---|---|---|"]
    for st, p in s["stages"].items():
        lines.append(f"| {st} | {p['kernels']} | {p['gpu_ms']:.2f} | {p['sm_fill']} | "
                     f"{p['occupancy_pct']} | {p['queue_delay_us_median']} | "
                     f"{p['graph_kernels']} |")
    lines += ["", "| stage call | calls | CPU us | GPU span us | GPU busy us | idle in span us |",
              "|---|---|---|---|---|---|"]
    for st, p in s["stage_calls"].items():
        lines.append(f"| {st} | {p['calls']} | {p['cpu_us']:.1f} | {p['gpu_span_us']:.1f} | "
                     f"{p['gpu_busy_us']:.1f} | {p['idle_in_span_us']:.1f} |")
    lines += ["", "| per patch (steady) | kernels | GPU us |", "|---|---|---|"]
    for st, p in sorted(s["per_patch"].items(), key=lambda x: -x[1]["gpu_us"]):
        lines.append(f"| {st} | {p['kernels']} | {p['gpu_us']:.1f} |")
    lines += ["", "| gap | count | ms |", "|---|---|---|"]
    for k, v in s["gaps"].items():
        lines.append(f"| {k} | {v['count']} | {v['ms']:.2f} |")
    lines += ["", "| kernel durations | mean us | median us | time share in kernels < 10 us | kernels < 5 us |",
              "|---|---|---|---|---|"]
    for st, k in s["kernel_durations"].items():
        lines.append(f"| {st} | {k['mean_us']} | {k['median_us']} | {k['share_time_lt10us']} | {k['n_lt5us']} |")
    lines += ["", "| largest gaps us | at ms | after | before |", "|---|---|---|---|"]
    for g in s["top_gaps"]:
        lines.append(f"| {g['us']} | {g['at_ms']} | {g['after']} | {g['before']} |")
    lines += ["", "| gap >= 2us between | count | ms |", "|---|---|---|"]
    for k, v in s["gap_pairs"].items():
        lines.append(f"| {k} | {v['count']} | {v['ms']:.2f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    path = sys.argv[1]
    skip = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    s = analyze(path, skip)
    Path(path.replace(".pkl.gz", ".summary.json")).write_text(json.dumps(s, indent=1))
    md = table(s)
    Path(path.replace(".pkl.gz", ".summary.md")).write_text(md)
    print(md)
