"""Discrete-event simulation: k concurrent agent sessions sharing one GPU.

Each agent slot replays real sessions (bootstrap from the evaluating arms: kernel, systems,
native) as their measured sequence of pieces:
  agent - model thinking/writing, CPU tools (no GPU)
  eval  - a timed GPU job (evaluate_candidate / evaluate_e2e / sweep_candidate): exclusive
  dev   - the agent's own Python/benchmark/compile runs outside the lock today
After a session ends the slot starts another one drawn at random (with replacement).

Integration (paired A/B of kept results) becomes GPU work proportional to agent evals: each
completed eval adds alpha seconds of integration work (alpha = the run's integration time /
its agent evals), enqueued as chunks drawn from the run's A/B measurement durations.

GPU modes for dev pieces:
  lenient - dev runs need no lock (as today); timed jobs may overlap other agents' dev runs:
            we report the share of timed GPU time that overlapped (= timing not clean)
  strict  - a fraction f of every dev piece is a GPU job in the same exclusive queue
            (the rest, 1-f, is compile / CPU time before it)
  rw      - reader/writer lock: timed jobs (eval, integration) exclusive; dev runs (fraction
            f) share the GPU with each other but never with a timed job (writer preference)
Queue disciplines: fifo (one queue for everything) and prio (eval > dev > integration).
"""

from __future__ import annotations

import heapq
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

D = Path(__file__).resolve().parent
S = json.load(open(D / "sessions.json"))
E = json.load(open(D / "evals.json"))
I = json.load(open(D / "integrations.json"))

ARMS = ("kernel", "systems", "native")


def run_params(runs):
    pool = [s for s in S if s["run"] in runs and s["role"] in ARMS and s.get("pieces")]
    alpha, chunks = {}, {}
    for r in runs:
        nev = sum(len(s["evals"]) for s in S if s["run"] == r)
        integ = sum(i["end"] - i["start"] for i in I if i["run"] == r)
        alpha[r] = integ / max(nev, 1)
        chunks[r] = [e["eval_s"] for e in E if e["run"] == r and e["eval_s"] and (e["backend"] == "integrate" or e["status"] == "re-evaluated")]
    return pool, alpha, chunks


class Sim:
    def __init__(self, k, pool, alpha, chunks, mode="lenient", disc="prio", f=0.5, integration=True, integ_scale=1.0, seed=0, hours=400, warm=20):
        self.k, self.pool, self.chunks = k, pool, chunks
        self.alpha = {r: a * integ_scale for r, a in alpha.items()}
        self.next_chunk = {}
        self.eval_q = 0  # eval jobs waiting in the queue
        self.mode, self.disc, self.f, self.integration = mode, disc, f, integration
        self.rng = random.Random(seed)
        self.T, self.W = hours * 3600, warm * 3600
        self.t = 0.0
        self.ev = []  # (time, seq, kind, data)
        self.seq = 0
        self.queue = []  # (prio, seq, job)
        self.running_x = None  # exclusive job running
        self.dev_running = 0  # rw: dev runs sharing the GPU
        self.dev_wait = []  # rw: dev jobs waiting (writer preference)
        self.credit = 0.0
        self.integ_q = 0.0  # seconds of integration work waiting in the queue
        self.stats = defaultdict(float)
        self.waits = defaultdict(list)
        self.lenient_dev_active = 0  # lenient: agents in a dev piece right now
        self.last = 0.0

    # -- bookkeeping of time-weighted quantities
    def advance(self, t):
        dt = max(0.0, min(t, self.T) - max(self.last, self.W))
        if dt > 0:
            if self.running_x is not None:
                self.stats["busy_" + self.running_x["kind"]] += dt
                if self.running_x["kind"] in ("eval", "integ") and (self.lenient_dev_active > 0 or self.dev_running > 0):
                    self.stats["timed_overlap"] += dt
            if self.dev_running > 0:
                self.stats["busy_devshared"] += dt
        self.last = t
        self.t = t

    def push(self, t, kind, data):
        self.seq += 1
        heapq.heappush(self.ev, (t, self.seq, kind, data))

    # -- agents
    def new_session(self, a):
        s = self.rng.choice(self.pool)
        pieces = []
        for kind, d in s["pieces"]:
            if kind == "dev" and self.mode in ("strict", "rw"):
                if (1 - self.f) * d > 0:
                    pieces.append(("agent", (1 - self.f) * d))
                pieces.append(("devjob", self.f * d))
            else:
                pieces.append((kind, d))
        a["pieces"], a["i"], a["run"] = pieces, 0, s["run"]
        if self.t >= self.W:
            self.stats["sessions"] += 1

    def step(self, a):
        if a["i"] >= len(a["pieces"]):
            self.new_session(a)
        kind, d = a["pieces"][a["i"]]
        a["i"] += 1
        if kind == "agent":
            self.push(self.t + d, "agent", a)
        elif kind == "dev":  # lenient: runs without the lock
            self.lenient_dev_active += 1
            a["in_dev"] = True
            self.push(self.t + d, "dev_end", a)
        elif kind == "eval":
            self.submit({"kind": "eval", "dur": d, "owner": a, "t_sub": self.t, "run": a["run"]})
        elif kind == "devjob":
            job = {"kind": "dev", "dur": d, "owner": a, "t_sub": self.t}
            if self.mode == "strict":
                self.submit(job)
            else:
                self.submit_shared(job)

    # -- GPU
    def prio(self, job):
        if self.disc == "fifo":
            return 0
        return {"eval": 0, "dev": 1, "integ": 2}[job["kind"]]

    def submit(self, job):
        if job["kind"] == "integ":
            self.integ_q += job["dur"]
        if job["kind"] == "eval":
            self.eval_q += 1
        self.seq += 1
        heapq.heappush(self.queue, (self.prio(job), self.seq, job))
        self.dispatch()

    def submit_shared(self, job):
        # rw: a dev run shares the GPU with other dev runs; it waits while a timed job runs
        # or an eval waits (writer preference for evals; integration yields to dev runs)
        if self.running_x is None and self.eval_q == 0:
            self.start_shared(job)
        else:
            self.dev_wait.append(job)

    def release_shared(self):
        if self.running_x is None and self.eval_q == 0 and self.dev_wait:
            for j in self.dev_wait:
                self.start_shared(j)
            self.dev_wait = []

    def start_shared(self, job):
        self.dev_running += 1
        if self.t >= self.W:
            self.waits["dev"].append(self.t - job["t_sub"])
        self.push(self.t + job["dur"], "dev_done", job)

    def dispatch(self):
        if self.running_x is not None or not self.queue:
            return
        if self.mode == "rw" and self.dev_running > 0:
            return  # writer waits for the readers to drain
        _, _, job = heapq.heappop(self.queue)
        if job["kind"] == "integ":
            self.integ_q -= job["dur"]
        if job["kind"] == "eval":
            self.eval_q -= 1
        self.running_x = job
        if self.t >= self.W and job["kind"] in ("eval", "dev"):
            self.waits[job["kind"]].append(self.t - job["t_sub"])
        self.push(self.t + job["dur"], "gpu_done", job)

    def after_eval(self, run):
        if not self.integration:
            return
        self.credit += self.alpha[run]
        ch = self.chunks[run]
        while ch:
            d = self.next_chunk.get(run) or self.rng.choice(ch)
            if self.credit < d:
                self.next_chunk[run] = d
                break
            self.next_chunk[run] = None
            self.credit -= d
            self.submit({"kind": "integ", "dur": d, "owner": None, "t_sub": self.t})

    def run(self):
        agents = [{"id": i} for i in range(self.k)]
        for a in agents:
            self.new_session(a)
            # desynchronise the slots: start each at a random point of its first session
            self.push(self.rng.uniform(0, 600), "agent", a)
        while self.ev:
            t, _, kind, data = heapq.heappop(self.ev)
            if t > self.T:
                break
            self.advance(t)
            if kind == "agent":
                self.step(data)
            elif kind == "dev_end":
                self.lenient_dev_active -= 1
                self.step(data)
            elif kind == "gpu_done":
                job = data
                self.running_x = None
                if job["kind"] == "eval":
                    if self.t >= self.W:
                        self.stats["evals"] += 1
                    self.after_eval(job["run"])
                if job["owner"] is not None:
                    self.step(job["owner"])
                if self.mode == "rw":
                    self.release_shared()
                self.dispatch()
            elif kind == "dev_done":
                self.dev_running -= 1
                self.step(data["owner"])
                self.dispatch()
        self.advance(self.T)
        H = (self.T - self.W) / 3600
        busy_x = sum(v for k, v in self.stats.items() if k.startswith("busy_") and k != "busy_devshared")
        timed = self.stats["busy_eval"] + self.stats["busy_integ"]
        ew = self.waits["eval"]
        dw = self.waits["dev"]
        return {
            "evals_per_h": self.stats["evals"] / H,
            "sessions_per_h": self.stats["sessions"] / H,
            "gpu_util_timed": timed / (self.T - self.W),
            "gpu_util_any": (busy_x + self.stats["busy_devshared"]) / (self.T - self.W),
            "util_eval": self.stats["busy_eval"] / (self.T - self.W),
            "util_integ": self.stats["busy_integ"] / (self.T - self.W),
            "util_dev": (self.stats["busy_dev"] + self.stats["busy_devshared"]) / (self.T - self.W),
            "eval_wait_mean": st.mean(ew) if ew else 0.0,
            "eval_wait_p90": sorted(ew)[int(0.9 * (len(ew) - 1))] if ew else 0.0,
            "dev_wait_mean": st.mean(dw) if dw else 0.0,
            "blocked_share": (sum(ew) + sum(dw)) / (self.k * (self.T - self.W)),
            "timed_overlap": self.stats["timed_overlap"] / timed if timed else 0.0,
            "integ_backlog_h": self.integ_q / 3600,
        }


def avg(results):
    return {k: st.mean(r[k] for r in results) for k in results[0]}


def simulate(runs, ks, seeds=4, **kw):
    pool, alpha, chunks = run_params(runs)
    out = {}
    for k in ks:
        out[k] = avg([Sim(k, pool, alpha, chunks, seed=s, **kw).run() for s in range(seeds)])
    return out


SCENARIOS = [
    ("A", "no integration; dev runs unlocked (as today)", dict(mode="lenient", disc="fifo", integration=False)),
    ("B", "integration at today's ratio, one FIFO queue; dev unlocked", dict(mode="lenient", disc="fifo")),
    ("C", "integration at today's ratio, priority eval > integration; dev unlocked", dict(mode="lenient", disc="prio")),
    ("D", "clean: dev runs exclusive too (f=0.5), priority eval > dev > integration", dict(mode="strict", disc="prio", f=0.5)),
    ("E", "clean: reader/writer lock (timed exclusive, dev shared, f=0.5), integration lowest", dict(mode="rw", disc="prio", f=0.5)),
    ("F", "as E, integration 4x cheaper", dict(mode="rw", disc="prio", f=0.5, integ_scale=0.25)),
    ("G", "as D, integration 4x cheaper", dict(mode="strict", disc="prio", f=0.5, integ_scale=0.25)),
    ("H", "as E, no integration", dict(mode="rw", disc="prio", f=0.5, integration=False)),
    ("D1", "as G with f=1.0 (every dev second on the GPU)", dict(mode="strict", disc="prio", f=1.0, integ_scale=0.25)),
    ("D2", "as G with f=0.25", dict(mode="strict", disc="prio", f=0.25, integ_scale=0.25)),
]
KS = [1, 2, 3, 4, 6]


def main():
    groups = {
        "pooled (all 4 runs)": ["V-lat-r1", "V-lat", "V-thr", "Q-lat"],
        "V-lat (latency)": ["V-lat"],
        "V-thr (throughput)": ["V-thr"],
        "Q-lat (Qwen3)": ["Q-lat"],
    }
    which = sys.argv[1:] or list(groups)
    res = {}
    for g in which:
        runs = groups[g]
        pool, alpha, _ = run_params(runs)
        res[g] = {"alpha_s_per_eval": alpha, "pool_sessions": len(pool), "scenarios": {}}
        for sid, desc, kw in SCENARIOS:
            res[g]["scenarios"][sid] = {"desc": desc, "k": simulate(runs, KS, **kw)}
            print(g, sid, {k: round(v["evals_per_h"], 1) for k, v in res[g]["scenarios"][sid]["k"].items()}, flush=True)
    (D / "sim_results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
