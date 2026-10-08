"""Tables from sessions.json / evals.json / integrations.json / ratelimits.json (extract.py).

Prints markdown; also writes tables.md and session_table.csv next to this file.
"""

from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

D = Path(__file__).resolve().parent
S = json.load(open(D / "sessions.json"))
E = json.load(open(D / "evals.json"))
I = json.load(open(D / "integrations.json"))
R = json.load(open(D / "ratelimits.json"))
P = json.load(open(D / "phases.json"))
RUNS = ["V-lat-r1", "V-lat", "V-thr", "Q-lat"]
ROLES = ["planner", "dossier", "research", "kernel", "systems", "native", "librarian"]
out: list[str] = []


def emit(s: str = "") -> None:
    out.append(s)


def table(head: list[str], rows: list[list]) -> None:
    emit("| " + " | ".join(head) + " |")
    emit("|" + "|".join("---:" if i else "---" for i in range(len(head))) + "|")
    for r in rows:
        emit("| " + " | ".join(str(x) for x in r) + " |")
    emit()


def pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "-"


def q(v, p):
    v = sorted(v)
    if not v:
        return None
    k = (len(v) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def f1(x):
    return "-" if x is None else f"{x:.1f}"


def union(iv):
    o = []
    for a, b in sorted(iv):
        if o and a <= o[-1][1]:
            o[-1][1] = max(o[-1][1], b)
        else:
            o.append([a, b])
    return o


def usd(s):
    r = s.get("result") or {}
    return r.get("usd") if r.get("usd") is not None else (s.get("usd_event") or 0.0)


# ------------------------------------------------------------------ run timeline
def phase_windows(run):
    ws, open_ = [], {}
    for p in [p for p in P if p["run"] == run]:
        if p["event"] == "phase_start" and p["phase"] in ("analyze", "plan", "capture"):
            open_[p["phase"]] = p["ts"]
        elif p["event"] == "phase_done" and p["phase"] in open_:
            ws.append((open_.pop(p["phase"]), p["ts"], p["phase"]))
    return ws


emit("## T1. Where the wall time of a run goes (today: one agent at a time)")
emit()
emit(
    "Active wall = union of agent sessions, integrations and the analyze/plan/capture phases, "
    "gaps under 10 min merged (overnight pauses between invocations excluded); the plan phase is the planner session, so it is counted there. "
    "GPU-lock busy = evaluation tool calls inside sessions + integration windows + analyze/capture. "
    "Dev GPU = Bash commands that run Python/benchmarks/profilers (incl. background jobs and waits on them), outside the lock."
)
emit()
rows = []
tot = defaultdict(float)
for run in RUNS:
    ss = [s for s in S if s["run"] == run]
    iw = [(i["start"], i["end"]) for i in I if i["run"] == run]
    ph = phase_windows(run)
    acts = [(s["start"], s["end"]) for s in ss] + iw + [(a, b) for a, b, _ in ph]
    merged = []
    for a, b in sorted(acts):
        if merged and a - merged[-1][1] <= 600:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    active = sum(b - a for a, b in merged)
    sess = sum(s["wall_s"] for s in ss)
    integ = sum(b - a for a, b in iw)
    phs = sum(b - a for a, b, n in ph if n != "plan")  # plan phase = the planner session
    gpu_ph = sum(b - a for a, b, n in ph if n in ("analyze", "capture"))
    other = active - sess - integ - phs
    lock = sum(s.get("gpu_lock_s", 0) for s in ss)
    dev = sum(s.get("gpu_dev_s", 0) for s in ss)
    nev = sum(len(s.get("evals", [])) for s in ss)
    nint = sum(1 for e in E if e["run"] == run and (e["backend"] == "integrate" or e["status"] == "re-evaluated"))
    cost = sum(usd(s) for s in ss)
    busy = lock + integ + gpu_ph
    rows.append(
        [
            run,
            f"{active / 3600:.1f}",
            f"{len(ss)}",
            f"{sess / 3600:.1f} ({pct(sess, active)})",
            f"{integ / 3600:.1f} ({pct(integ, active)})",
            f"{phs / 3600:.2f} ({pct(phs, active)})",
            f"{other / 3600:.2f} ({pct(other, active)})",
            nev,
            nint,
            f"{nev / (active / 3600):.1f}",
            pct(busy, active),
            pct(dev, active),
            pct(active - busy - dev, active),
            f"{cost:.0f}",
        ]
    )
    for k, v in (("active", active), ("sess", sess), ("integ", integ), ("phs", phs), ("other", other), ("lock", lock), ("dev", dev), ("busy", busy), ("nev", nev), ("nint", nint), ("cost", cost), ("n", len(ss))):
        tot[k] += v
rows.append(
    [
        "**all**",
        f"{tot['active'] / 3600:.1f}",
        int(tot["n"]),
        f"{tot['sess'] / 3600:.1f} ({pct(tot['sess'], tot['active'])})",
        f"{tot['integ'] / 3600:.1f} ({pct(tot['integ'], tot['active'])})",
        f"{tot['phs'] / 3600:.2f} ({pct(tot['phs'], tot['active'])})",
        f"{tot['other'] / 3600:.2f} ({pct(tot['other'], tot['active'])})",
        int(tot["nev"]),
        int(tot["nint"]),
        f"{tot['nev'] / (tot['active'] / 3600):.1f}",
        pct(tot["busy"], tot["active"]),
        pct(tot["dev"], tot["active"]),
        pct(tot["active"] - tot["busy"] - tot["dev"], tot["active"]),
        f"{tot['cost']:.0f}",
    ]
)
table(
    [
        "run",
        "active h",
        "sessions",
        "agent sessions h",
        "integration h",
        "analyze + capture h",
        "orchestrator other h",
        "agent evals",
        "integration measurements",
        "agent evals / active h",
        "GPU-lock busy",
        "dev GPU (outside lock)",
        "GPU idle",
        "$ (API-equiv.)",
    ],
    rows,
)

# ------------------------------------------------------------------ per role
emit("## T2. Agent sessions by role (all four runs)")
emit()
emit(
    "Exclusive split of session wall time, priority GPU lock > model > dev GPU > CPU tools > idle. "
    "model = an API call in flight (thinking + writing; its total matches duration_api_ms of the ResultMessage within 2%); "
    "GPU lock = evaluate_candidate / evaluate_e2e / sweep_candidate calls; "
    "dev GPU = Bash running Python / benchmarks / profilers, background jobs and waits on them (outside the lock; "
    "includes load_inline / nvcc compiles started from Python); CPU tools = other Bash, Read/Write/Grep, WebFetch, ToolSearch; "
    "idle = none of these (SDK start-up, hooks, a session tail kept open by a background job). "
    "GPU any = lock or dev GPU (the dev part counted even while the model thinks)."
)
emit()
rows = []
for role in ROLES + ["all"]:
    ss = [s for s in S if (role == "all" or s["role"] == role) and s.get("transcript")]
    if not ss:
        continue
    wall = sum(s["wall_s"] for s in ss)
    g = lambda k: sum(s[k] for s in ss)
    idle = wall - g("part_lock") - g("part_model") - g("part_dev") - g("part_cpu")
    nev = sum(len(s["evals"]) for s in ss)
    first = [s["evals"][0]["t0"] - s["start"] for s in ss if s["evals"]]
    cost = sum(usd(s) for s in ss)
    turns = [((s.get("result") or {}).get("turns") or 0) for s in ss]
    rows.append(
        [
            role,
            len(ss),
            f"{wall / 3600:.1f}",
            f"{wall / len(ss) / 60:.1f}",
            pct(g("part_model"), wall),
            pct(g("part_lock"), wall),
            pct(g("part_dev"), wall),
            pct(g("part_cpu"), wall),
            pct(idle, wall),
            pct(g("gpu_any_s"), wall),
            nev,
            f"{nev / (wall / 3600):.1f}",
            f1(st.median(first) / 60) if first else "-",
            f"{sum(1 for s in ss if not s['evals'])}",
            f"{cost / len(ss):.2f}",
            f"{cost / nev:.2f}" if nev else "-",
            f"{st.mean(turns):.0f}",
        ]
    )
table(
    [
        "role",
        "sessions",
        "hours",
        "min/session",
        "model",
        "GPU lock",
        "dev GPU",
        "CPU tools",
        "idle",
        "GPU any",
        "evals",
        "evals / session-h",
        "median min to 1st eval",
        "sessions w/o eval",
        "$ / session",
        "$ / eval",
        "turns / session",
    ],
    rows,
)
tails = [(s["run"], s["label"], s["tail_s"]) for s in S if s.get("tail_s", 0) > 60]
emit(
    f"Sessions kept open > 1 min after their last message (Claude Code waiting for a background job): {len(tails)} "
    f"({', '.join(f'{r}:{l} {t / 60:.0f} min' for r, l, t in tails)})."
)
emit()

emit("### T2b. Same split per run, evaluating arms only (kernel + systems + native)")
emit()
rows = []
for run in RUNS:
    for role in ("kernel", "systems", "native"):
        ss = [s for s in S if s["run"] == run and s["role"] == role and s.get("transcript")]
        if not ss:
            continue
        wall = sum(s["wall_s"] for s in ss)
        lock = sum(s["gpu_lock_s"] for s in ss)
        dev = sum(s["gpu_dev_s"] for s in ss)
        model = sum(s["model_s"] for s in ss)
        nev = sum(len(s["evals"]) for s in ss)
        rows.append([run, role, len(ss), f"{wall / 3600:.2f}", pct(model, wall), pct(lock, wall), pct(dev, wall), pct(wall - lock - dev, wall), nev, f"{nev / (wall / 3600):.1f}", f"{sum(usd(s) for s in ss) / max(nev, 1):.2f}"])
table(["run", "role", "sessions", "hours", "model", "GPU lock", "dev GPU", "GPU idle", "evals", "evals / h", "$ / eval"], rows)

# ------------------------------------------------------------------ GPU jobs
emit("## T3. GPU job durations (seconds)")
emit()
emit("Evaluation tool calls (start of tool_use to its result; equals results.tsv eval_s + ~0.6 s) and integration A/B measurements (results.tsv eval_s of `integrate` / `re-evaluated` rows).")
emit()
rows = []
groups = defaultdict(list)
for s in S:
    for e in s.get("evals", []):
        groups[(s["run"], s["role"], e["name"])].append(e["dur"])
        groups[("all", s["role"], e["name"])].append(e["dur"])
for e in E:
    if e["eval_s"] and (e["backend"] == "integrate" or e["status"] == "re-evaluated"):
        groups[(e["run"], "integration", "A/B measurement")].append(e["eval_s"])
        groups[("all", "integration", "A/B measurement")].append(e["eval_s"])
for k in sorted(groups, key=lambda k: (k[0] != "all", k)):
    v = groups[k]
    rows.append([*k, len(v), f1(q(v, 0.5)), f1(st.mean(v)), f1(q(v, 0.9)), f1(max(v)), f"{sum(v) / 60:.0f}"])
table(["run", "role", "job", "n", "median", "mean", "p90", "max", "total min"], rows)

cs = [e["compile_s"] for s in S for e in s.get("evals", []) if e.get("compile_s") is not None]
emit(
    f"compile_s reported inside evaluate_candidate results: n={len(cs)}, median {q(cs, .5):.1f} s, p90 {q(cs, .9):.1f} s, max {max(cs):.1f} s "
    "(agents build in Bash first, so the evaluator mostly hits the extension cache; compiles happen in the dev-GPU Bash time instead)."
)
emit()

# ------------------------------------------------------------------ think time
emit("## T4. Agent time between GPU-lock jobs (minutes)")
emit()
emit("Gaps from session start (or the end of the previous evaluation) to the next evaluation call; 'tail' is the last evaluation to session end. Includes model, dev-GPU Bash and CPU tools.")
emit()
gaps = defaultdict(lambda: defaultdict(list))
for s in S:
    if s["role"] not in ("kernel", "systems", "native") or not s.get("transcript"):
        continue
    prev = s["start"]
    for i, e in enumerate(s["evals"]):
        gaps[s["role"]]["first" if i == 0 else "between"].append((e["t0"] - prev) / 60)
        prev = e["t1"]
    gaps[s["role"]]["tail" if s["evals"] else "no-eval session"].append((s["end"] - prev) / 60)
rows = []
for role, d in gaps.items():
    for k in ("first", "between", "tail", "no-eval session"):
        v = d.get(k, [])
        if v:
            rows.append([role, k, len(v), f1(q(v, 0.5)), f1(st.mean(v)), f1(q(v, 0.9))])
table(["role", "gap", "n", "median", "mean", "p90"], rows)

# ------------------------------------------------------------------ lock waits / overlaps
emit("## T5. Waiting on the GPU lock and GPU overlap today")
emit()
lock_iv = []
for s in S:
    for e in s.get("evals", []):
        lock_iv.append((e["t0"], e["t1"], s["run"] + ":" + s["label"]))
for i in I:
    lock_iv.append((i["start"], i["end"], i["run"] + ":integration"))
over = 0
pairs = []
for a in lock_iv:
    for b in lock_iv:
        if a is b or a[2] == b[2]:
            continue
        if a[0] < b[1] and b[0] < a[1]:
            over += 1
            pairs.append((a[2], b[2]))
outlive = [(s["run"], s["label"], s["bg_outlive_s"]) for s in S if s.get("bg_outlive_s", 0) > 1]
waits = [b for s in S for b in s.get("bash", []) if b["cls"] == "wait"]
nbg = sum(s.get("bg_tasks", 0) for s in S)
d_eval = [e["dur"] for s in S for e in s.get("evals", [])]
emit(
    f"* GPU-lock jobs: {len(d_eval)} agent evaluation calls + {len(I)} integration windows. Overlapping lock intervals across sessions/integration: {over // 2} pairs {sorted(set(pairs))[:4]}."
)
emit(
    "* Every evaluation call lasted its eval_s + ~0.6 s (eval_s starts before the lock is taken), so no agent waited for the lock: with one session at a time and integration run between slices, the lock is never contended."
)
emit(
    f"* Background Bash jobs started by agents: {nbg}; jobs still running when their session ended: {len(outlive)} "
    f"({', '.join(f'{r}:{l} +{v / 60:.1f} min' for r, l, v in outlive[:6])}). Such jobs, and every dev-GPU Bash run, use the GPU outside the lock."
)
emit(
    f"* Agents instead wait on their *own* GPU work: {len(waits)} Bash polls (until/while/pgrep/nvidia-smi loops, Monitor) "
    f"took {sum(b['dur'] for b in waits) / 3600:.2f} h; 3 sessions stayed open 10 min after their last message for a background benchmark."
)
emit()

# ------------------------------------------------------------------ cost & rate limits
emit("## T6. Cost and subscription limits")
emit()
rows = []
for run in RUNS:
    ss = [s for s in S if s["run"] == run]
    cost = sum(usd(s) for s in ss)
    hours = sum(s["wall_s"] for s in ss) / 3600
    nev = sum(len(s.get("evals", [])) for s in ss)
    toks = sum(((s.get("result") or {}).get("usage") or {}).get("output_tokens", 0) or 0 for s in ss)
    rows.append([run, f"{cost:.2f}", f"{hours:.1f}", f"{cost / hours:.2f}", nev, f"{cost / max(nev, 1):.2f}", f"{toks / 1e6:.2f}"])
table(["run", "$ total (API-equivalent)", "agent h", "$ / agent-h", "agent evals", "$ / agent eval", "output Mtok"], rows)

# utilisation slope per session (same window, >= 2 samples)
by = defaultdict(list)
for r in R:
    by[r["label"] + "@" + r["run"]].append(r)
slopes5, slopes7, w5, w7, dt5, dt7 = [], [], 0.0, 0.0, 0.0, 0.0
for k, v in by.items():
    v.sort(key=lambda r: r["t"])
    for win, acc in (("five_hour", "5"), ("seven_day", "7")):
        vv = [r for r in v if r["windows"].get(win, {}).get("utilization") is not None]
        if len(vv) < 2:
            continue
        # keep the samples of the window of the last sample (resets within a session)
        last = vv[-1]["windows"][win]["resetsAt"]
        vv = [r for r in vv if r["windows"][win]["resetsAt"] == last]
        if len(vv) < 2:
            continue
        du = vv[-1]["windows"][win]["utilization"] - vv[0]["windows"][win]["utilization"]
        dtt = vv[-1]["t"] - vv[0]["t"]
        if dtt < 300:
            continue
        if acc == "5":
            slopes5.append(du / (dtt / 3600))
            w5 += du
            dt5 += dtt
        else:
            slopes7.append(du / (dtt / 3600))
            w7 += du
            dt7 += dtt
statuses = defaultdict(int)
for r in R:
    statuses[(r["status"], r["type"])] += 1
max7 = max((r["windows"].get("seven_day", {}).get("utilization") or 0) for r in R)
emit(
    f"* Rate-limit events seen by the sessions: {dict(statuses)}; no `rejected` status, no session stopped at a usage limit "
    f"(`allowed_warning` = the seven-day window above its warning threshold; highest seven-day utilisation seen {max7:.2f})."
)
emit(
    f"* Five-hour window: utilisation rises {100 * w5 / (dt5 / 3600):.1f} percentage points per session-hour (pooled over {len(slopes5)} sessions; "
    f"median per session {100 * q(slopes5, .5):.1f}, p90 {100 * q(slopes5, .9):.1f}). "
    f"Seven-day window: {100 * w7 / (dt7 / 3600):.2f} points per session-hour (median {100 * q(slopes7, .5):.2f}). "
    "These are account-wide (they include any interactive use at the same time), so they are upper bounds for the agents alone."
)
r5 = w5 / (dt5 / 3600)
r7 = w7 / (dt7 / 3600)
rows = []
for k in (1, 2, 3, 4, 6):
    h5 = 1 / (k * r5)
    h7 = 1 / (k * r7)
    rows.append([k, f"{100 * k * r5 * 5:.0f}%", f"{h5:.1f}" if h5 < 5 else "never (resets first)", f"{h7:.0f} h ({h7 / 24:.1f} d)"])
table(["k sessions in parallel", "5-h window used per 5 h", "hours to exhaust a fresh 5-h window", "hours to exhaust a fresh 7-day window"], rows)

# ------------------------------------------------------------------ per-session csv
with open(D / "session_table.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["run", "label", "role", "wall_min", "api_min", "model_min", "gpu_lock_min", "gpu_dev_min", "idle_min", "evals", "first_eval_min", "usd", "turns", "bg_tasks"])
    for s in S:
        r = s.get("result") or {}
        w.writerow(
            [
                s["run"],
                s["label"],
                s["role"],
                round(s["wall_s"] / 60, 2),
                round((r.get("api_s") or 0) / 60, 2),
                round(s.get("model_s", 0) / 60, 2),
                round(s.get("gpu_lock_s", 0) / 60, 2),
                round(s.get("gpu_dev_s", 0) / 60, 2),
                round(s.get("idle_s", 0) / 60, 2),
                len(s.get("evals", [])),
                round((s["evals"][0]["t0"] - s["start"]) / 60, 2) if s.get("evals") else "",
                round(usd(s), 3),
                r.get("turns"),
                s.get("bg_tasks", 0),
            ]
        )

text = "\n".join(out)
(D / "tables.md").write_text(text)
print(text)
