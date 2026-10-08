"""Extract per-session timelines from kernel-agent runs (read-only).

Sources per run: events.jsonl (session / integration windows, evaluation times),
results.tsv (eval_s per evaluation), logs/agent-*.jsonl (session ids, ResultMessage
with duration_api_ms / cost / turns, RateLimitEvent utilisation), and the Claude Code
transcripts ~/.claude/projects/*/<session_id>.jsonl (timestamp of every assistant block
and every tool result -> model intervals and tool intervals).

Writes sessions.json, evals.json, integrations.json, ratelimits.json next to this file.
"""

from __future__ import annotations

import csv
import datetime as dt
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

RUNS_ROOT = Path(__file__).resolve().parents[3] / "runs"  # the repo's runs/
PROJECTS = Path.home() / ".claude/projects"
OUT = Path(__file__).resolve().parent

RUNS = {
    "V-lat-r1": "openbmb--VoxCPM2/20261005-042829",  # older latency run (rounds 1-2)
    "V-lat": "openbmb--VoxCPM2/20261005-192504",  # latency run incl. wave-5 native continuation
    "V-thr": "openbmb--VoxCPM2/20261006-004718",  # throughput run
    "Q-lat": "Qwen--Qwen3-0.6B/20261008-021850",  # Qwen3 latency run (wave 5)
}

GPU_TOOLS = {
    "mcp__ka__evaluate_candidate",
    "mcp__ka__evaluate_e2e",
    "mcp__ka__sweep_candidate",
    "mcp__ka__verify_rewrite",
}


def ts(s: str) -> float:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def role_of(agent: str) -> str:
    for p in ("planner", "systems", "native", "librarian"):
        if agent.startswith(p):
            return p
    if agent.startswith("kernel-"):
        return "kernel"
    if agent.startswith("research-"):
        return "research"
    if agent.startswith("dossier-"):
        return "dossier"
    return agent


# --------------------------------------------------------------------------- Bash classes
PY_RUN = re.compile(r"(?:^|[\s;&|(])(?:timeout\s+\d+\s+)?(?:\S*/)?python[0-9.]*\s+(?:-u\s+)?(?!-\s|-c\b)(?:-m\s+\S+|\S+\.py)\b")
PY_INLINE = re.compile(r"python[0-9.]*\s+(?:-u\s+)?(-\s*<<\s*'?\"?(\w+)'?\"?|-c\s)")
GPU_WORDS = re.compile(
    r"import torch|from torch|torch\.|triton|tilelang|load_inline|cuda\.|subprocess|os\.system|kernel_agent\.(?:kernels|worker|profiling)|cutlass"
)
PROFILERS = re.compile(r"(?:^|[\s;&|])(?:ncu|nsys|compute-sanitizer|cuda-memcheck)\b")
WAIT = re.compile(r"(until|while)\b[^\n]*(pgrep|kill -0|ps -p|grep -c|grep -q)|for i in \$\(seq[^\n]*(ps -p|kill -0|pgrep)|^\s*sleep \d+")
NVCC = re.compile(r"(?:^|[\s;&|/])nvcc\b|cuobjdump|nvdisasm|ptxas")


PY_VAR = re.compile(r"\$\{?(?:PY|P|PYTHON)\w*\}?\s+(?:-u\s+)?(?:\S+\.py|-m\s)")
GPU_POLL = re.compile(r"nvidia-smi[^\n]*(memory\.used|compute-apps)[^\n]*|until[^\n]*nvidia-smi")
CPU_ONLY = re.compile(r"^\s*(find|git|pip|uv|grep|rg|ls|du|cat|sed|head|tail|wc)\b")


def bash_class(cmd: str, dur: float = 0.0) -> str:
    """gpu: likely runs GPU work (python with torch / scripts / profilers, or a command that
    ran >= 15 s and is not an obvious CPU tool); wait: polls a background job or the GPU;
    compile: nvcc-only; cpu: everything else (read, grep, sed, edits)."""
    if WAIT.search(cmd) or (GPU_POLL.search(cmd) and re.search(r"\b(for|until|while)\b", cmd)):
        return "wait"
    if PY_VAR.search(cmd):
        return "gpu"
    c = _bash_class(cmd)
    if c == "cpu" and dur >= 15 and not CPU_ONLY.search(cmd):
        return "gpu"
    return c


def _bash_class(cmd: str) -> str:
    if PROFILERS.search(cmd):
        return "gpu"
    if PY_RUN.search(cmd):
        # `python3 x.py` / `python -m mod`; exclude obvious CPU-only modules
        m = PY_RUN.search(cmd)
        if not re.search(r"-m\s+(json\.tool|py_compile|ast|pip)\b", m.group(0)):
            return "gpu"
    for m in PY_INLINE.finditer(cmd):
        body = cmd[m.end() : m.end() + 20000]
        if m.group(2):  # heredoc: body up to the terminator
            end = re.search(rf"^\s*{re.escape(m.group(2))}\s*$", body, re.M)
            body = body[: end.start()] if end else body
        if GPU_WORDS.search(body):
            return "gpu"
    if re.search(r"(?:^|[\s;&|])\./[\w.-]+(?:\s|$)", cmd):  # runs a built binary
        return "gpu"
    if NVCC.search(cmd):
        return "compile"
    return "cpu"


# --------------------------------------------------------------------------- intervals
def union(iv: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted((a, b) for a, b in iv if b > a):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def total(iv) -> float:
    return sum(b - a for a, b in union(iv))


def clip(iv, lo, hi):
    return [(max(a, lo), min(b, hi)) for a, b in iv if min(b, hi) > max(a, lo)]


def minus(iv, sub):
    """iv minus the union of sub."""
    sub = union(sub)
    out = []
    for a, b in union(iv):
        cur = a
        for c, d in sub:
            if d <= cur or c >= b:
                continue
            if c > cur:
                out.append((cur, c))
            cur = max(cur, d)
        if cur < b:
            out.append((cur, b))
    return out


def pieces(lo, hi, lock, dev):
    marks = sorted({lo, hi, *[x for iv in lock + dev for x in iv]})
    out = []
    for a, b in zip(marks, marks[1:]):
        m = (a + b) / 2
        kind = "eval" if any(x <= m < y for x, y in lock) else "dev" if any(x <= m < y for x, y in dev) else "agent"
        if out and out[-1][0] == kind:
            out[-1][1] += b - a
        else:
            out.append([kind, b - a])
    return out


# --------------------------------------------------------------------------- transcripts
_index: dict[str, Path] | None = None


def transcript(sid: str) -> Path | None:
    global _index
    if _index is None:
        _index = {}
        for f in glob.glob(str(PROJECTS / "*kernel-agent-runs-*" / "*.jsonl")):
            _index.setdefault(Path(f).stem, Path(f))
    return _index.get(sid)


def parse_transcript(path: Path) -> dict:
    entries = [json.loads(l) for l in open(path)]
    calls: dict[str, dict] = {}  # message.id -> first/last ts, first index
    order = []
    tool_uses: dict[str, dict] = {}
    results: dict[str, float] = {}
    user_times: list[float] = []
    bg_done: dict[str, float] = {}  # tool_use_id -> background completion time
    bg_ids: set[str] = set()
    eval_payloads: dict[str, str] = {}
    out_tokens = 0
    for i, d in enumerate(entries):
        t = d.get("timestamp")
        if t is None:
            continue
        t = ts(t)
        typ = d.get("type")
        if typ == "assistant" and not d.get("isSidechain"):
            m = d["message"]
            mid = m.get("id") or d.get("uuid")
            c = calls.get(mid)
            if c is None:
                c = calls[mid] = {"first": t, "last": t, "idx": i, "usage": m.get("usage") or {}}
                order.append(mid)
            c["last"] = max(c["last"], t)
            c["usage"] = m.get("usage") or c["usage"]
            for b in m.get("content", []):
                if b.get("type") == "tool_use":
                    tool_uses[b["id"]] = {"t": t, "name": b["name"], "input": b.get("input") or {}, "call": mid}
        elif typ == "user":
            user_times.append((i, t))
            c = d.get("message", {}).get("content")
            if isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        results.setdefault(b.get("tool_use_id"), t)
                        if b.get("tool_use_id") in tool_uses and tool_uses[b["tool_use_id"]]["name"] in GPU_TOOLS:
                            eval_payloads[b["tool_use_id"]] = json.dumps(b.get("content"))[:20000]
            tur = d.get("toolUseResult")
            if isinstance(tur, dict) and tur.get("backgroundTaskId") and isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        bg_ids.add(b.get("tool_use_id"))
        elif typ == "queue-operation" and d.get("operation") == "enqueue":
            s = d.get("content") or ""
            m = re.search(r"<tool-use-id>(\w+)</tool-use-id>", s)
            if m and "<task-notification>" in s:
                bg_done.setdefault(m.group(1), t)
    for mid in order:
        out_tokens += int((calls[mid]["usage"] or {}).get("output_tokens") or 0)
    # model interval of a call: from the latest user entry before its first block to its last block
    model_iv = []
    for mid in order:
        c = calls[mid]
        prev = [t for (j, t) in user_times if j < c["idx"]]
        start = prev[-1] if prev else c["first"]
        model_iv.append((start, c["last"]))
    tools = []
    for tid, u in tool_uses.items():
        end = results.get(tid)
        name = u["name"]
        rec = {"id": tid, "name": name, "t0": u["t"], "t1": end if end is not None else u["t"]}
        if name == "Bash":
            cmd = u["input"].get("command", "")
            rec["cls"] = bash_class(cmd, rec["t1"] - rec["t0"])
            rec["cmd"] = cmd[:160]
            rec["bg"] = bool(u["input"].get("run_in_background")) or tid in bg_ids
            if rec["bg"]:
                rec["bg_end"] = bg_done.get(tid)
        if name in GPU_TOOLS:
            rec["input"] = {k: (v if len(json.dumps(v)) < 300 else "...") for k, v in u["input"].items() if k != "hypothesis"}
            p = eval_payloads.get(tid, "")
            m = re.search(r'\\"compile_s\\": ([0-9.]+)', p)
            rec["compile_s"] = float(m.group(1)) if m else None
            m = re.search(r'\\"status\\": \\"(\w+)\\"', p)
            rec["status"] = m.group(1) if m else None
            m = re.search(r'\\"exp\\": (\d+)', p)
            rec["exp"] = int(m.group(1)) if m else None
        tools.append(rec)
    times = [ts(d["timestamp"]) for d in entries if d.get("timestamp")]
    return {
        "first": min(times),
        "last": max(times),
        "model_iv": model_iv,
        "n_calls": len(order),
        "tools": tools,
        "out_tokens": out_tokens,
        "call_last": {mid: calls[mid]["last"] for mid in order},
    }


# --------------------------------------------------------------------------- per run
def agent_logs(run_dir: Path) -> dict:
    """session id -> ResultMessage summary, per agent log the ordered init session ids,
    and the rate-limit events with the message id they follow."""
    inits: dict[str, list[str]] = {}
    results: dict[str, dict] = {}
    rl = []
    for f in sorted(run_dir.glob("logs/agent-*.jsonl")):
        name = f.stem.removeprefix("agent-")
        inits[name] = []
        last_mid = None
        sid = None
        for line in open(f):
            d = json.loads(line)
            t = d["type"]
            if t == "SystemMessage" and d["data"].get("subtype") == "init":
                sid = d["data"]["data"]["session_id"]
                inits[name].append(sid)
            elif t == "AssistantMessage":
                last_mid = d["data"].get("message_id")
            elif t == "ResultMessage":
                r = d["data"]
                results[r["session_id"]] = {
                    "duration_s": r["duration_ms"] / 1000,
                    "api_s": r["duration_api_ms"] / 1000,
                    "turns": r["num_turns"],
                    "usd": r["total_cost_usd"],
                    "usage": r.get("usage"),
                    "subtype": r.get("subtype"),
                    "is_error": r.get("is_error"),
                }
            elif t == "RateLimitEvent":
                ri = d["data"]["rate_limit_info"]
                raw = ri.get("raw") or {}
                rl.append(
                    {
                        "agent": name,
                        "sid": sid,
                        "after_mid": last_mid,
                        "status": ri.get("status"),
                        "type": ri.get("rate_limit_type"),
                        "utilization": ri.get("utilization"),
                        "windows": {k: v for k, v in (raw.get("unifiedWindows") or {}).items()},
                        "resets_at": ri.get("resets_at"),
                    }
                )
    return {"inits": inits, "results": results, "rl": rl}


def main() -> None:
    sessions, evals, integrations, ratelimits, phases = [], [], [], [], []
    for key, rel in RUNS.items():
        rd = RUNS_ROOT / rel
        ev = [json.loads(l) for l in open(rd / "events.jsonl")]
        rows = {int(r["exp"]): r for r in csv.DictReader(open(rd / "results.tsv"), delimiter="\t")}
        logs = agent_logs(rd)
        # sessions from events
        counters: dict[str, int] = defaultdict(int)
        open_s: dict[str, dict] = {}
        sess = []
        for e in ev:
            if e["event"] == "agent_start":
                a = e["agent"]
                k = counters[a]
                counters[a] += 1
                ids = logs["inits"].get(a, [])
                s = {
                    "run": key,
                    "agent": a,
                    "label": e.get("label") or a,
                    "role": role_of(a),
                    "start": e["ts"],
                    "end": None,
                    "sid": ids[k] if k < len(ids) else None,
                    "done": False,
                }
                open_s[a] = s
                sess.append(s)
            elif e["event"] == "agent_done":
                s = open_s.pop(e["agent"], None)
                if s:
                    s.update(end=e["ts"], done=True, usd_event=e.get("usd"), minutes_event=e.get("minutes"))
            elif e["event"] in ("slice_done", "interrupted"):
                for a, s in list(open_s.items()):
                    if e["event"] == "interrupted" or e.get("status") == "interrupted":
                        s["end"] = e["ts"]
                        open_s.pop(a)
        # integration windows: reintegrate -> integrated; an invocation that stops during
        # one (next phase_start / interrupted) closes it at its last evaluation or the stop
        start = None
        last_eval = None
        for e in ev:
            if e["event"] == "reintegrate":
                if start is None:
                    start, last_eval = e["ts"], e["ts"]
            elif e["event"] == "evaluation" and start is not None:
                last_eval = e["ts"]
            elif e["event"] == "integrated" and start is not None:
                integrations.append({"run": key, "start": start, "end": e["ts"], "speedup": e.get("speedup")})
                start = None
            elif e["event"] in ("phase_start", "interrupted") and start is not None:
                end = e["ts"] if e["event"] == "interrupted" else last_eval
                integrations.append({"run": key, "start": start, "end": end, "interrupted": True})
                start = None
            if e["event"] in ("phase_start", "phase_done", "interrupted"):
                phases.append({"run": key, "event": e["event"], "phase": e.get("phase"), "ts": e["ts"]})
        # evaluations (all rows) with epoch end time from events
        ev_ts = {e["exp"]: e["ts"] for e in ev if e["event"] == "evaluation" and "exp" in e}
        for exp, r in rows.items():
            t1 = ev_ts.get(exp)
            evals.append(
                {
                    "run": key,
                    "exp": exp,
                    "t1": t1,
                    "eval_s": float(r["eval_s"]) if r.get("eval_s") else None,
                    "target": r["target"],
                    "backend": r["backend"],
                    "status": r["status"],
                }
            )
        # transcripts
        for s in sess:
            if s["end"] is None:
                s["end"] = s["start"]
            s["wall_s"] = s["end"] - s["start"]
            res = logs["results"].get(s["sid"] or "")
            s["result"] = res
            p = transcript(s["sid"]) if s["sid"] else None
            s["transcript"] = str(p) if p else None
            if not p:
                sessions.append(s)
                continue
            tr = parse_transcript(p)
            lo, hi = s["start"], s["end"]
            model_iv = clip(tr["model_iv"], lo, hi)
            tools = tr["tools"]
            cat_iv = defaultdict(list)
            bg_iv = []
            for t in tools:
                n = t["name"]
                if n in GPU_TOOLS:
                    c = "gpu_eval"
                elif n == "Bash":
                    c = "bash_" + t["cls"]
                    if t.get("bg"):
                        end = t.get("bg_end") or hi
                        bg_iv.append((t["t0"], end, t["cls"]))
                elif n in ("Read", "Write", "Edit", "Grep", "Glob", "NotebookEdit"):
                    c = "files"
                elif n in ("WebFetch", "WebSearch"):
                    c = "web"
                elif n == "Monitor":
                    c = "bash_wait"
                else:
                    c = "other_tool"
                t["cat"] = c
                cat_iv[c].append((t["t0"], t["t1"]))
            cats = {c: total(clip(iv, lo, hi)) for c, iv in cat_iv.items()}
            all_tools = [iv for ivs in cat_iv.values() for iv in ivs]
            busy = union(clip(model_iv + all_tools + [(a, b or hi) for a, b, c in bg_iv], lo, hi))
            gpu_lock = union(clip(cat_iv.get("gpu_eval", []), lo, hi))
            bg_gpu = [(a, b) for a, b, c in bg_iv if c in ("gpu", "wait")]
            gpu_dev = union(clip(cat_iv.get("bash_gpu", []) + cat_iv.get("bash_wait", []) + bg_gpu, lo, hi))
            s.update(
                model_s=total(model_iv),
                tool_s=total(clip(all_tools, lo, hi)),
                cats=cats,
                model_only_s=total(minus(model_iv, all_tools)),
                idle_s=(hi - lo) - total(busy),
                gpu_lock_s=total(gpu_lock),
                gpu_dev_s=total(minus(gpu_dev, gpu_lock)),
                gpu_any_s=total(union(gpu_lock + gpu_dev)),
                # exclusive partition of the wall time: lock > model > dev GPU > CPU tools > idle
                part_lock=total(gpu_lock),
                part_model=total(minus(model_iv, gpu_lock)),
                part_dev=total(minus(gpu_dev, union(gpu_lock + model_iv))),
                part_cpu=total(minus(clip(all_tools, lo, hi), union(gpu_lock + model_iv + gpu_dev))),
                tail_s=max(0.0, hi - tr["last"]),
                bg_tasks=len(bg_iv),
                bg_outlive_s=sum(max(0.0, (b or hi) - hi) for a, b, c in bg_iv),
                n_calls=tr["n_calls"],
                out_tokens=tr["out_tokens"],
                transcript_first=tr["first"],
                transcript_last=tr["last"],
            )
            # eval tool calls
            # the session as a sequence of pieces: eval (GPU-lock job) > dev (GPU-likely Bash,
            # background job or wait on one) > agent (model, CPU tools, overhead)
            s["pieces"] = pieces(lo, hi, gpu_lock, gpu_dev)
            s["dev_iv"] = minus(gpu_dev, gpu_lock)
            s["lock_iv"] = gpu_lock
            s["evals"] = [
                {
                    "name": t["name"].removeprefix("mcp__ka__"),
                    "t0": t["t0"],
                    "t1": t["t1"],
                    "dur": t["t1"] - t["t0"],
                    "compile_s": t.get("compile_s"),
                    "status": t.get("status"),
                    "exp": t.get("exp"),
                }
                for t in sorted(tools, key=lambda x: x["t0"])
                if t["name"] in GPU_TOOLS
            ]
            s["bash"] = [
                {"t0": t["t0"], "dur": t["t1"] - t["t0"], "cls": t["cls"], "bg": t.get("bg", False), "cmd": t["cmd"]}
                for t in tools
                if t["name"] == "Bash"
            ]
            # rate limit events of this session: time = last block of the call they follow
            for r in logs["rl"]:
                if r["sid"] == s["sid"]:
                    r2 = dict(r, run=key, label=s["label"], t=tr["call_last"].get(r["after_mid"], s["start"]))
                    ratelimits.append(r2)
            sessions.append(s)
    for name, obj in (
        ("sessions", sessions),
        ("evals", evals),
        ("integrations", integrations),
        ("ratelimits", ratelimits),
        ("phases", phases),
    ):
        (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, default=str))
    print(len(sessions), "sessions;", sum(1 for s in sessions if s.get("transcript")), "with transcripts;", len(evals), "result rows;", len(integrations), "integrations;", len(ratelimits), "rate-limit events")


if __name__ == "__main__":
    main()
