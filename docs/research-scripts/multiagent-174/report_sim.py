"""Markdown tables from sim_results.json (simulate.py) -> sim_tables.md."""

import json
from pathlib import Path

D = Path(__file__).resolve().parent
R = json.load(open(D / "sim_results.json"))
out = []


def emit(s=""):
    out.append(s)


def table(head, rows):
    emit("| " + " | ".join(head) + " |")
    emit("|" + "|".join("---:" if i else "---" for i in range(len(head))) + "|")
    for r in rows:
        emit("| " + " | ".join(str(x) for x in r) + " |")
    emit()


def flags(m):
    f = ""
    if m["timed_overlap"] > 0.05:
        f += "†"
    if m["integ_backlog_h"] > 1.0:
        f += "‡"
    return f


KS = ["1", "2", "3", "4", "6"]
emit("### S1. Agent evaluations per hour vs k (cell: evals/h, GPU busy, mean wait of an evaluation)")
emit()
emit("† timed GPU work overlapped another agent's unlocked dev run (> 5% of timed GPU time: timing not clean). "
     "‡ integration falls behind (backlog > 1 h of A/B work after 400 simulated hours: not sustainable).")
emit()
for g, v in R.items():
    emit(f"**{g}** (bootstrap pool: {v['pool_sessions']} evaluating-arm sessions; integration GPU-s per agent eval: "
         + ", ".join(f"{k} {a:.0f}" for k, a in v["alpha_s_per_eval"].items()) + ")")
    emit()
    rows = []
    for sid, sc in v["scenarios"].items():
        if sid in ("D1", "D2") and g != "pooled (all 4 runs)":
            continue
        row = [f"{sid}: {sc['desc']}"]
        for k in KS:
            m = sc["k"][k]
            row.append(f"{m['evals_per_h']:.1f} ({100 * m['gpu_util_any']:.0f}%, {m['eval_wait_mean']:.0f}s){flags(m)}")
        rows.append(row)
    table(["scenario"] + [f"k={k}" for k in KS], rows)

emit("### S2. Pooled detail for the clean-timing scenarios")
emit()
g = "pooled (all 4 runs)"
for sid in ("E", "F", "G", "H"):
    sc = R[g]["scenarios"][sid]
    emit(f"**{sid}: {sc['desc']}**")
    emit()
    base = sc["k"]["1"]["evals_per_h"]
    rows = []
    for k in KS:
        m = sc["k"][k]
        rows.append([
            k,
            f"{m['evals_per_h']:.1f}",
            f"{m['evals_per_h'] / base:.2f}x",
            f"{m['evals_per_h'] / int(k) / base:.0%}",
            f"{100 * m['util_eval']:.0f}% / {100 * m['util_integ']:.0f}% / {100 * m['util_dev']:.0f}%",
            f"{100 * m['gpu_util_any']:.0f}%",
            f"{m['eval_wait_mean']:.0f} / {m['eval_wait_p90']:.0f}",
            f"{m['dev_wait_mean']:.0f}",
            f"{100 * m['blocked_share']:.0f}%",
            f"{m['integ_backlog_h']:.1f}",
        ])
    table(["k", "evals/h", "vs k=1", "per-agent efficiency", "GPU eval / integration / dev", "GPU busy", "eval wait mean / p90 s", "dev wait s", "agent time blocked", "integration backlog h"], rows)

(D / "sim_tables.md").write_text("\n".join(out))
print("\n".join(out))
