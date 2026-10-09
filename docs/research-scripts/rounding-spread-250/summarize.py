"""Summary of ``calibrate.py``'s output, per chain, depth and candidate.

    python summarize.py calibrate.json
"""

import json
import sys

data = json.load(open(sys.argv[1]))
for key, rows in data.items():
    for c, rs in rows.items():
        n = len(rs)
        fail = sum(1 for r in rs if r["mis"] > 1 or r["ratio"] > 10)
        q = lambda k, p=0.5: sorted(r[k] for r in rs)[min(n - 1, int(p * n))]  # noqa: E731
        print(
            f"{key:8s} {c:13s} n={n} plainfail={fail} mis_max={max(r['mis'] for r in rs)} "
            f"ratio_max={max(r['ratio'] for r in rs):.2f} rel med={q('rel'):.4f} max={max(r['rel'] for r in rs):.4f} "
            f"s_mca_rel med={q('s_mca_rel'):.4f} min={min(r['s_mca_rel'] for r in rs):.4f} "
            f"s32_rel med={q('s32_rel'):.4f} cand32 med={q('cand32_rel'):.4f} "
            f"d/s med={q('d_over_s'):.2f} max={max(r['d_over_s'] for r in rs):.2f} "
            f"k0 max={max(r['k0'] for r in rs):.2f} k1 max={max(r['k1'] for r in rs):.2f} mca_mis max={max(r['mca_mis'] for r in rs)}"
        )
        kp = {}
        for r in rs:
            for nm, v in r["kpass"].items():
                kp.setdefault(nm, []).append(v)
        if kp:
            print("   cheats K-to-pass min:", {nm: min(v) for nm, v in kp.items()})
