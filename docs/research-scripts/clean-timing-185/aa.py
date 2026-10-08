"""A/A validation of clean timing (issue #185), interleaved: one fixed candidate's speedup,
evaluated by kernel-agent's evaluator (run_evaluation: the evaluate_candidate path) in blocks
that alternate between

  Q  N = 1: nothing else runs; clean timing off (today's --agents 1)
  C  N = 3 with clean timing (the new --agents 3 default): 3 simulated agents build
     load_inline-like extensions (main.cpp + a .cu, sometimes 3 variants at once; their Bash:
     nice 10 on the other cores, MAX_JOBS, paused while a timed job times) and run their own
     GPU scripts through the GPU job queue (run_on_gpu's path, class dev); the evaluator on
     the timing cores at nice -5, prebuilds, dirty-timing re-runs
  L  N = 3 without it (control): the same agents, their GPU scripts outside the GPU lock
     (--agent-gpu bash), builds on every core at the session's nice
  B  N = 3, GPU scripts outside the lock (bash) but clean timing on
  U  as C, but the timed process not pinned to the timing cores

so that the load of the rest of this (shared) machine falls on every condition alike; every
record has /proc/loadavg at its start and ``foreign_cpus``: the CPUs the rest of the machine
used meanwhile.

Layout next to this script: ``cand.py`` (the candidate), ``run/.truth/captures/<target>.pt``
and ``run/targets/<target>/capture_inputs.pt`` (copies or links of a run's captures; here
Qwen3-0.6B ``decoder_layer_decode`` and its 8.05x cooperative-kernel prior). Run each under
the machine's GPU lock (this process uses a private lock file for its own queue):

  flock ~/.cache/kernel-agent/gpu.lock python aa.py --schedule QCLQCL --per-block 2 --tag r5
"""

import argparse
import contextlib
import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--schedule", default="QCLQCL")
    p.add_argument("--per-block", type=int, default=3)
    p.add_argument("--agents", type=int, default=3)
    p.add_argument("--tag", default="r")
    ns = p.parse_args()

    from kernel_agent import devrun, gpulock, gpuqueue, hygiene, interrupt, toolchain
    from kernel_agent.agent import runner
    from kernel_agent.kernels.evaluate import run_evaluation

    gpulock.CACHE_DIR = HERE / "locks"  # the outer flock excludes every other process
    os.environ["TORCH_EXTENSIONS_DIR"] = str(HERE / "ext")
    toolchain.setup()
    capture = HERE / "run" / ".truth" / "captures" / "decoder_layer_decode.pt"
    cand = HERE / "cand.py"
    out_file = HERE / "results" / f"aa-{ns.tag}.jsonl"
    out_file.parent.mkdir(exist_ok=True)

    mode = {"cond": "Q", "block": -1}
    active = threading.Event()  # agents start new steps only while set
    stop = threading.Event()
    guard = threading.Lock()
    procs: list[subprocess.Popen] = []
    busy = {"dev": 0, "build": 0}
    events: list[dict] = []

    def spawn(cmd, env):
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            start_new_session=True,
        )
        hygiene.apply_env(proc.pid, env)  # as a Bash command of a moved CLI inherits it
        if hasattr(hygiene, "background"):  # a Bash command of a registered CLI: paused
            hygiene.background(proc.pid)  # while a timed job times (off: no-op)
        with guard:
            procs.append(proc)
        out, _ = proc.communicate()
        with guard:
            procs.remove(proc)
        if hasattr(hygiene, "done"):
            hygiene.done(proc.pid)
        return proc.returncode, out

    def agent(i: int) -> None:
        rng = random.Random(1000 + i)
        n = 0
        seen = None
        while not stop.is_set():
            if not active.wait(0.5):
                continue
            cond = mode["cond"]
            if (mode["block"], cond) != seen:  # a new block: the agents are out of phase
                seen = (mode["block"], cond)
                if stop.wait(rng.uniform(0, 20)) or not active.is_set():
                    continue
            via_queue = cond in "CU"
            session = runner.agent_env({}, gpu=runner.GPU_TOOL if via_queue else runner.GPU_BASH)
            env = {**os.environ, **session}
            n += 1
            t0 = time.time()
            variants = 3 if rng.random() < 0.25 else 1  # sometimes variants side by side
            with guard:
                busy["build"] += 1
            try:
                rc, _ = spawn([sys.executable, str(HERE / "build_job.py"), str(HERE / "builds"),
                               f"a{i}_{ns.tag}_{os.getpid()}_{n}", str(variants)], env)
            finally:
                with guard:
                    busy["build"] -= 1
            events.append({"what": "build", "agent": i, "cond": cond, "t0": t0,
                           "t1": time.time(), "rc": rc, "variants": variants})
            if stop.wait(rng.uniform(15, 35)) or not active.is_set():  # the model thinks
                continue
            secs = rng.uniform(3, 8)
            t0 = time.time()
            with guard:
                busy["dev"] += 1
            try:
                if via_queue:  # run_on_gpu: the queue, class dev; half of them time something
                    bench = rng.random() < 0.5
                    job = gpuqueue.Job(
                        kind="dev", job_class=gpuqueue.DEV, session=f"agent{i}",
                        estimate_s=secs, exclusive=bench, mem_gb=None if bench else 2.0,
                    )
                    with gpuqueue.using(job):
                        r = devrun.run_script(HERE / "dev_job.py", [f"{secs:.1f}"], cwd=HERE,
                                              timeout=60, env=session)
                    rc, wait = r["returncode"], job.wait_s
                else:  # its own Bash: on the GPU whenever it wants
                    rc, _ = spawn([sys.executable, str(HERE / "dev_job.py"), f"{secs:.1f}"], env)
                    wait = 0.0
            finally:
                with guard:
                    busy["dev"] -= 1
            events.append({"what": "dev", "agent": i, "cond": cond, "t0": t0,
                           "t1": time.time(), "rc": rc, "wait_s": round(wait, 2)})
            stop.wait(rng.uniform(15, 35))

    def quiesce() -> None:
        """No agent work: no new steps, running dev runs finish, builds are killed."""
        active.clear()
        deadline = time.time() + 60
        while busy["dev"] and time.time() < deadline:
            time.sleep(0.2)
        with guard:
            for proc in list(procs):
                interrupt.kill(proc.pid)  # with nvcc, which runs in a group of its own
        time.sleep(2)

    hz = os.sysconf("SC_CLK_TCK")

    def cpu_now() -> tuple[float, int, int]:
        """(time, busy ticks of every CPU, ticks of this process tree): what the rest of the
        machine (not this experiment) used is the difference."""
        values = [int(v) for v in Path("/proc/stat").read_text().split("\n")[0].split()[1:]]
        busy = sum(values) - values[3] - values[4]
        own = 0
        for pid in [os.getpid(), *(p for p, _ in interrupt.descendants())]:
            try:
                f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            except OSError:
                continue
            own += int(f[11]) + int(f[12]) + int(f[13]) + int(f[14])  # with reaped children
        return time.monotonic(), busy, own

    def evaluate_once() -> dict:
        job = gpuqueue.Job(kind="eval", job_class=gpuqueue.EVAL, session="measure", estimate_s=17)
        t0 = time.time()
        load = Path("/proc/loadavg").read_text().split()[0]
        with guard:
            running = dict(busy)
        c0 = cpu_now()
        with gpuqueue.using(job):
            r = run_evaluation(capture, cand, timeout=300)
        c1 = cpu_now()
        foreign = ((c1[1] - c0[1]) - (c1[2] - c0[2])) / hz / max(c1[0] - c0[0], 1e-9)
        cases = [
            {k: c.get(k) for k in ("ref_ms", "new_ms", "speedup", "timing_spread", "clock")}
            for c in r.get("cases") or [] if c.get("ref_ms")
        ]
        return {
            "status": r.get("status"), "speedup": r.get("speedup"),
            "ref_ms": r.get("ref_ms_weighted"), "new_ms": r.get("new_ms_weighted"),
            "cpu_wait_share": r.get("cpu_wait_share"), "retimed": r.get("retimed"),
            "cpu_others_share": r.get("cpu_others_share"),
            "timing_dirty": r.get("timing_dirty"), "compile_s": r.get("compile_s"),
            "prebuild": r.get("prebuild"), "queue_s": round(job.wait_s, 2),
            "hold_s": round(job.hold_s, 2), "holds": job.holds,
            "wall_s": round(time.time() - t0, 1), "t0": t0, "loadavg": float(load),
            "builds_at_start": running["build"], "devs_at_start": running["dev"],
            "foreign_cpus": round(max(foreign, 0.0), 2),
            "cases": cases, "error": (r.get("error") or "")[-300:] or None,
        }

    threads = [threading.Thread(target=agent, args=(i,), daemon=True) for i in range(ns.agents)]
    for t in threads:
        t.start()
    try:
        warm = evaluate_once()  # compiles once, caches the reference re-time
        print("warm-up:", warm["status"], warm["speedup"], warm["wall_s"], flush=True)
        for b, cond in enumerate(ns.schedule):
            quiesce()
            mode["cond"], mode["block"] = cond, b
            clean = hygiene.active(agents=ns.agents) if cond in "CBU" else contextlib.nullcontext()
            # U: clean timing, but the timed process not pinned to the timing cores (only
            # the agents' work is kept off them)
            timed_env = hygiene.timed_env
            if cond == "U":

                def unpinned(original=timed_env):
                    return {k: v for k, v in original().items() if k != hygiene.CPUS_ENV}

                hygiene.timed_env = unpinned
            with clean:
                if cond != "Q":
                    active.set()
                    time.sleep(15)  # builds and dev runs ramp up
                for k in range(ns.per_block):
                    rec = evaluate_once()
                    with out_file.open("a") as fh:
                        fh.write(json.dumps({"cond": cond, "block": b, "k": k, **rec}) + "\n")
                    print(cond, b, k, rec["status"], rec["speedup"], rec["ref_ms"], rec["new_ms"],
                          rec["cpu_wait_share"], rec["cpu_others_share"], rec["builds_at_start"],
                          rec["devs_at_start"],
                          "R" if rec["retimed"] else "",
                          "D" if rec["timing_dirty"] else "", rec["queue_s"], rec["wall_s"],
                          rec["loadavg"], rec["foreign_cpus"], flush=True)
                quiesce()
            hygiene.timed_env = timed_env
    finally:
        stop.set()
        active.set()
        quiesce()
        for t in threads:
            t.join(30)
        (HERE / "results" / f"aa-{ns.tag}.events.json").write_text(json.dumps(events))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
