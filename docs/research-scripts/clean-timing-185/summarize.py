"""The A/A tables of aa.py runs: per condition the speedup's spread and what happened, then
every measurement with the host's load.

    python summarize.py r5 r6 r7 > results.md
"""

import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = {
    "Q": "Q: N = 1, nothing else (`--agents 1`)",
    "C": "C: N = 3 agents, clean timing (`--agents 3`)",
    "L": "L: N = 3 agents, no clean timing (control)",
    "B": "B: N = 3, scripts outside the lock, clean timing on",
    "U": "U: as C, timed process not pinned",
}


def spread(xs):
    m = statistics.mean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return m, sd, 100 * sd / m, min(xs), max(xs)


def main(tags):
    rows = []
    for tag in tags:
        for line in (HERE / "results" / f"aa-{tag}.jsonl").read_text().splitlines():
            rows.append({"run": tag, **json.loads(line)})
    print("| condition | evaluations | speedup mean | sd (CV) | min - max | reference ms / run "
          "(CV) | candidate ms / run (CV) | measured twice | dirty twice | not ok |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cond in NAMES:
        sel = [r for r in rows if r["cond"] == cond]
        if not sel:
            continue
        ok = [r for r in sel if r["status"] == "ok"]
        sp, ref, new = (spread([r[k] for r in ok]) for k in ("speedup", "ref_ms", "new_ms"))
        print(
            f"| {NAMES[cond]} | {len(sel)} | {sp[0]:.3f}x | {sp[1]:.3f} ({sp[2]:.1f} %) "
            f"| {sp[3]:.2f} - {sp[4]:.2f} | {ref[0]:.1f} ({ref[2]:.1f} %) | "
            f"{new[0]:.2f} ({new[2]:.1f} %) | {sum(bool(r['retimed']) for r in sel)} | "
            f"{sum(bool(r['timing_dirty']) for r in sel)} | "
            f"{sum(r['status'] != 'ok' for r in sel)} |"
        )
    print()
    print("| run | block | condition | speedup | ref ms | new ms | loadavg (1 min) | foreign "
          "CPUs | builds / dev runs at start | GPU holds | first measurement not counted |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        why = re.split(r"[:(]", r["retimed"] or "")[0].strip()
        print(
            f"| {r['run']} | {r['block']} | {r['cond']} | {r['speedup']} | {r['ref_ms']:.1f} | "
            f"{r['new_ms']:.2f} | {r['loadavg']} | {r.get('foreign_cpus', '-')} | "
            f"{r['builds_at_start']} / {r['devs_at_start']} | {r['holds']} | {why} |"
        )


main(sys.argv[1:] or ["r5", "r6", "r7"])
