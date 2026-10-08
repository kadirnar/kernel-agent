"""Clean timing under concurrent load (issue #185): CPU isolation (hygiene.py), builds off the
GPU lock (kernels/prebuild.py), dirty-timing re-runs (telemetry.HoldWatch) and run_on_gpu
(devrun.py). CPU tests with fake GPUs, but the last one."""

import asyncio
import json
import os
import subprocess
import sys
import textwrap
import time

import pytest

from kernel_agent import devrun, gpulock, gpuqueue, hygiene, telemetry, worker
from kernel_agent.agent import prompts, runner
from kernel_agent.agent import tools as tools_mod
from kernel_agent.kernels import evaluate, prebuild
from kernel_agent.workspace import RunDir, read_jsonl

POOL_ENV = (gpulock.ENV, gpulock.GPUS_ENV, gpulock.INDEX_ENV, "CUDA_VISIBLE_DEVICES")
CPUS = sorted(os.sched_getaffinity(0))
needs_cpus = pytest.mark.skipif(len(CPUS) < 2, reason="needs two CPUs")
MEMORY_STALL = telemetry.memory_stall


@pytest.fixture(autouse=True)
def lock_dir(tmp_path, monkeypatch, request):
    """Lock files in a temporary directory, one fake GPU, a fresh queue (not for the GPU test,
    which holds the real lock)."""
    if request.node.get_closest_marker("gpu"):
        yield None
        return
    locks = tmp_path / "locks"
    monkeypatch.setattr(gpulock, "CACHE_DIR", locks)
    for name in POOL_ENV:
        monkeypatch.delenv(name, raising=False)
    gpus = [gpulock.GPU(0, "NVIDIA Fake GPU", "GPU-000-fake")]
    monkeypatch.setattr(gpulock, "_nvidia_smi", lambda: list(gpus))
    gpulock.pool.cache_clear()
    monkeypatch.setattr(gpuqueue, "_gates", {})
    monkeypatch.setattr(telemetry, "memory_stall", lambda: None)  # this host's swapping
    yield locks
    gpulock.pool.cache_clear()


def plan(timing=None):
    """A plan of this machine's CPUs: the last one (or ``timing``) for the timed jobs."""
    timing = tuple(timing or CPUS[-1:])
    return hygiene.Plan(cpus=tuple(CPUS), timing=timing, affinity=True)


@pytest.fixture
def clean(monkeypatch):
    """Clean timing on, with :func:`plan`."""
    detect = classmethod(lambda cls, cores=None, agents=1: plan())
    monkeypatch.setattr(hygiene.Plan, "detect", detect)
    with hygiene.active() as on:
        yield on


# ------------------------------------------------------------------ CPU isolation


def test_plan_takes_the_last_physical_cores_and_turns_affinity_off_below_8_cpus(monkeypatch):
    smt = [(c, c + 6) for c in range(6)]  # a 6-core CPU with 2 threads per core
    monkeypatch.setattr(hygiene, "_cpus", lambda: tuple(range(12)))
    monkeypatch.setattr(hygiene, "_physical_cores", lambda cpus: smt)
    split = hygiene.Plan.detect()
    assert split is not None and split.affinity
    assert split.timing == (4, 5, 10, 11) and split.other == (0, 1, 2, 3, 6, 7, 8, 9)
    assert split.max_jobs == 8
    assert "timing cores 4-5,10-11, the rest on 0-3,6-9" in split.describe()
    assert hygiene.Plan.detect(agents=3).max_jobs == 2  # a session's share of the 8 others
    assert hygiene.Plan.detect(0) is None and hygiene.Plan.detect(6) is None
    monkeypatch.setattr(hygiene, "_cpus", lambda: tuple(range(6)))
    monkeypatch.setattr(hygiene, "_physical_cores", lambda cpus: [(c,) for c in cpus])
    small = hygiene.Plan.detect()
    assert small is not None and not small.affinity and small.max_jobs == 4
    assert "no CPU affinity" in small.describe()


def test_physical_cores_of_this_machine_cover_every_cpu():
    groups = hygiene._physical_cores(tuple(CPUS))
    assert sorted(c for g in groups for c in g) == CPUS


def test_envs_are_empty_when_off_and_split_when_on(clean):
    timed, background = hygiene.timed_env(), hygiene.background_env()
    assert timed == {hygiene.CPUS_ENV: str(CPUS[-1]), hygiene.NICE_ENV: "-5"}
    assert background[hygiene.NICE_ENV] == "10"
    assert background[hygiene.CPUS_ENV] == ",".join(map(str, CPUS[:-1] or CPUS))
    assert background["MAX_JOBS"] == str(max(1, len(CPUS) - 1))


def test_envs_are_empty_when_off():
    assert hygiene.current() is None
    assert hygiene.timed_env() == {} and hygiene.background_env() == {}


def test_child_env_of_a_timed_hold_and_of_a_shared_one(clean):
    with gpulock.gpu_lock():
        env = gpulock.child_env()
        token = gpulock.hold_token()
    assert token and env[gpulock.HOLD_ENV] == token
    assert env[hygiene.CPUS_ENV] == str(CPUS[-1]) and env[hygiene.NICE_ENV] == "-5"
    shared = gpuqueue.Job(kind="dev", job_class=gpuqueue.DEV, exclusive=False, mem_gb=1.0)
    with gpuqueue.using(shared), gpulock.gpu_lock():
        env = gpulock.child_env()
    assert env[hygiene.NICE_ENV] == "10" and "MAX_JOBS" in env
    assert gpulock.hold_token() is None  # released


def test_child_env_is_unchanged_when_off():
    with gpulock.gpu_lock():
        env = gpulock.child_env()
    assert gpulock.HOLD_ENV not in env and hygiene.NICE_ENV not in env


@needs_cpus
def test_a_child_applies_the_cpus_and_nice_value_it_was_given(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(p for p in sys.path if p))
    code = "import os, kernel_agent; print(sorted(os.sched_getaffinity(0)), os.nice(0))"
    env = {**os.environ, hygiene.CPUS_ENV: str(CPUS[0]), hygiene.NICE_ENV: "12"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.stdout.split() == [f"[{CPUS[0]}]", "12"], out.stderr


@needs_cpus
def test_move_session_moves_the_cli_and_its_children(clean):
    mark = f"{os.getpid()}-kernel-t"
    script = "import subprocess, sys, time; subprocess.Popen(['sleep', '30']); time.sleep(30)"
    cli = subprocess.Popen(
        [sys.executable, "-c", script], env={**os.environ, hygiene.SESSION_ENV: mark}
    )
    other = subprocess.Popen(["sleep", "30"])  # another child: not this session's
    try:
        deadline = time.monotonic() + 10
        while not hygiene._children(cli.pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        moved = hygiene.move_session(mark)
        assert cli.pid in moved and len(moved) == 2 and other.pid not in moved
        for pid in moved:
            assert sorted(os.sched_getaffinity(pid)) == CPUS[:-1]
            assert os.getpriority(os.PRIO_PROCESS, pid) == hygiene.BACKGROUND_NICE
    finally:
        for proc in (cli, other):
            proc.kill()
            proc.wait()


def _state(pid):
    from kernel_agent import interrupt

    stat = interrupt._stat(pid)
    return stat[2] if stat else None


def _until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


def test_quiet_pauses_the_agents_work_while_a_timed_subprocess_times(tmp_path, monkeypatch, clean):
    """A Claude Code CLI's Bash commands and a prebuild stop while the hold's subprocess
    says it times, and go on when it is done or the hold ends; the CLI itself never stops."""
    code = "import subprocess, time; subprocess.Popen(['sleep', '30']); time.sleep(30)"
    cli = subprocess.Popen([sys.executable, "-c", code])
    build = subprocess.Popen(["sleep", "30"])
    other = subprocess.Popen(["sleep", "30"])  # not the agents' work
    phase = tmp_path / "phase"
    monkeypatch.setenv(hygiene.PHASE_ENV, str(phase))  # as the timed subprocess has it
    try:
        _until(lambda: hygiene._children(cli.pid))
        bash = hygiene._children(cli.pid)[0]
        hygiene.background(cli.pid, itself=False)
        hygiene.background(build.pid)
        with hygiene.Quiet(phase) as quiet:
            time.sleep(0.2)
            assert _state(bash) != "T" and _state(build.pid) != "T"
            hygiene.phase(hygiene.TIMING)
            _until(lambda: _state(bash) == "T" and _state(build.pid) == "T")
            assert _state(cli.pid) != "T" and _state(other.pid) != "T"
            hygiene.phase("")
            _until(lambda: _state(bash) != "T" and _state(build.pid) != "T")
            with hygiene.timing():
                _until(lambda: _state(bash) == "T")
                time.sleep(0.2)
        assert _state(bash) != "T" and _state(build.pid) != "T"  # the hold ended: going on
        assert quiet.paused_s > 0.1 and not phase.exists()
    finally:
        hygiene.done(cli.pid)
        hygiene.done(build.pid)
        for proc in (cli, build, other):
            proc.kill()
            proc.wait()
    assert hygiene.work() == set()


def test_work_paused_by_two_holds_goes_on_when_both_are_done(tmp_path, clean):
    """Two GPUs' timed jobs at once: one ending its timing does not continue what the other
    one still pauses."""
    build = subprocess.Popen(["sleep", "30"])
    first, second = tmp_path / "a", tmp_path / "b"
    try:
        hygiene.background(build.pid)
        with hygiene.Quiet(first), hygiene.Quiet(second):
            first.write_text(hygiene.TIMING)
            second.write_text(hygiene.TIMING)
            _until(lambda: _state(build.pid) == "T")
            first.write_text("")
            time.sleep(0.3)
            assert _state(build.pid) == "T"  # the second still times
            second.write_text("")
            _until(lambda: _state(build.pid) != "T")
    finally:
        hygiene.done(build.pid)
        build.kill()
        build.wait()
    assert not hygiene._paused


def test_an_exclusive_hold_tells_its_subprocess_where_to_say_it_times(clean):
    with gpulock.gpu_lock():
        env = gpulock.child_env()
    assert env[hygiene.PHASE_ENV].endswith(env[gpulock.HOLD_ENV])
    shared = gpuqueue.Job(kind="dev", job_class=gpuqueue.DEV, exclusive=False, mem_gb=1.0)
    with gpuqueue.using(shared), gpulock.gpu_lock():
        assert hygiene.PHASE_ENV not in gpulock.child_env()


def test_agent_env_hides_the_gpu_in_tool_mode(clean):
    env = runner.agent_env({"TORCH_CUDA_ARCH_LIST": "12.0"}, gpu=runner.GPU_TOOL)
    assert env["CUDA_VISIBLE_DEVICES"] == "" and env["TORCH_CUDA_ARCH_LIST"] == "12.0"
    assert env["MAX_JOBS"] == hygiene.background_env()["MAX_JOBS"]
    assert "CUDA_VISIBLE_DEVICES" not in runner.agent_env({}, gpu=runner.GPU_BASH)


def test_prompts_say_how_the_agents_reach_the_gpu():
    assert set(prompts.GPU_ACCESS) == set(runner.GPU_MODES)
    default = prompts._env_block("python", "GPU: none detected")
    assert "short compile/debug scripts are fine" in default and "run_on_gpu" not in default
    with prompts.gpu_access(runner.GPU_TOOL):
        text = prompts.stable_prefix("kernel", "python", "GPU: none detected")
    assert "run_on_gpu(" in text and "see no GPU" in text
    assert "run_on_gpu" not in prompts._env_block("python", "GPU: none detected")  # restored
    with pytest.raises(ValueError, match="agent-gpu"), prompts.gpu_access("x"):
        pass


def test_gpu_roles_have_run_on_gpu():
    from kernel_agent import roles

    for role in ("kernel", "systems", "native", "harness"):
        assert "mcp__ka__run_on_gpu" in roles.mcp_tools(role), role
    assert "mcp__ka__run_on_gpu" not in roles.mcp_tools("research")


def test_orchestrator_gpu_access_rebuilds_the_agents_environment(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from kernel_agent.config import OptimizeConfig
    from kernel_agent.orchestrator import Orchestrator

    orch = Orchestrator.__new__(Orchestrator)
    orch.tc = SimpleNamespace(env={})
    orch.cfg = OptimizeConfig(model_ref="org/m")
    orch.env = before = runner.agent_env({})
    with orch.gpu_access(runner.GPU_TOOL):
        assert orch.env["CUDA_VISIBLE_DEVICES"] == ""
        assert prompts._agent_gpu == runner.GPU_TOOL
    assert orch.env is before and prompts._agent_gpu == runner.GPU_BASH


# ------------------------------------------------------------------ the GPU queue


def test_a_requeued_job_goes_first_of_its_class():
    gate = gpuqueue.Gate()
    now = gpuqueue.clock()
    old = gpuqueue.Job(kind="eval", job_class=gpuqueue.EVAL, session="a", submitted=now, seq=0)
    again = gpuqueue.Job(kind="eval", job_class=gpuqueue.EVAL, session="b", submitted=now, seq=1)
    gate.served["b"] = 5  # served last: round robin would put it behind "a"
    gate.waiting += [old, again]
    assert gate.head() is old
    again.requeue()
    assert gate.head() is again
    quick = gpuqueue.Job(kind="quick", job_class=gpuqueue.INTERACTIVE, submitted=now, seq=2)
    gate.waiting.append(quick)
    assert gate.head() is quick  # the class still comes first
    gate._admitted(again, (0,))
    assert not again.front


# ------------------------------------------------------------------ dirty timing


def test_hold_watch_counts_only_foreign_processes(monkeypatch):
    token = "123.4"
    own = subprocess.Popen(["sleep", "30"], env={**os.environ, gpulock.HOLD_ENV: token})
    foreign = subprocess.Popen(["sleep", "30"])
    idle = subprocess.Popen(["sleep", "30"])
    procs = [{"pid": p.pid, "used_mib": 100, "name": "python"} for p in (own, foreign, idle)]
    procs.append({"pid": os.getpid(), "used_mib": 300, "name": "python"})  # this process
    monkeypatch.setattr(telemetry, "compute_processes", lambda gpu: list(procs))
    monkeypatch.setattr(telemetry, "sm_utilization", lambda gpu: {foreign.pid: 87, idle.pid: 0})
    try:
        with telemetry.HoldWatch(0, token, interval=0) as watch:
            pass
        assert [p["pid"] for p in watch.active] == [foreign.pid]
        assert "87 % SM" in watch.active[0]["why"]
        why = telemetry.dirty(watch, {})
        assert why and "outside the GPU lock" in why and str(foreign.pid) in why
    finally:
        for proc in (own, foreign, idle):
            proc.kill()
            proc.wait()


def test_cpu_wait_above_the_share_is_dirty():
    assert telemetry.dirty(None, {"cpu_wait_share": 0.01}) is None
    assert "waited for a CPU 20 %" in str(telemetry.dirty(None, {"cpu_wait_share": 0.2}))
    assert "used 30 % of the timing cores" in str(telemetry.dirty(None, {"cpu_others_share": 0.3}))
    swapped = telemetry.HoldWatch(0, None, interval=0)
    swapped.memory_stall_share = 0.1  # every task stalled on memory a tenth of the hold
    assert "out of memory" in str(telemetry.dirty(swapped, {}))
    stalled = MEMORY_STALL()  # this host's (the fixture fakes it for the other tests)
    assert stalled is None or stalled >= 0
    before = telemetry.schedstat()
    sum(i * i for i in range(200_000))
    share = telemetry.cpu_wait_share(before)
    assert share is None or 0.0 <= share <= 1.0


def test_others_share_of_the_timing_cores(monkeypatch):
    assert telemetry.cores_busy() is None  # not pinned: not measured
    monkeypatch.setenv(hygiene.CPUS_ENV, ",".join(map(str, CPUS)))
    before = telemetry.cores_busy()
    assert before is not None
    sum(i * i for i in range(300_000))
    share = telemetry.others_share(before)
    assert share is None or 0.0 <= share <= 1.0


def _fake_evaluator(monkeypatch, results):
    """``subprocess.run`` of the evaluator: the next of ``results`` (callables may act first)."""
    calls = []

    def run(cmd, *, env, **kwargs):
        calls.append(env)
        result = results[len(calls) - 1]
        if callable(result):
            result = result()
        tag = kwargs["input"].strip() + "@@"
        return subprocess.CompletedProcess(cmd, 0, "@@KA_RESULT@@" + tag + json.dumps(result), "")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(evaluate, "ensure_peaks", lambda: None)
    monkeypatch.setattr(evaluate, "_check_outputs", lambda *a, **k: None)
    return calls


def test_dirty_timing_reruns_with_a_fake_foreign_process(tmp_path, monkeypatch, clean):
    """A process outside the GPU lock on the GPU during the first measurement: it does not
    count, the evaluation is measured again first of its class and the second one counts."""
    foreign = subprocess.Popen(["sleep", "60"])
    on_gpu = {foreign.pid}

    def processes(gpu):
        return [{"pid": p, "used_mib": 2048, "name": "bench.py"} for p in sorted(on_gpu)]

    def first():
        foreign.kill()  # it ends during the timing
        foreign.wait()
        on_gpu.clear()
        return {"status": "ok", "correct": True, "speedup": 1.4}

    monkeypatch.setattr(telemetry, "compute_processes", processes)
    calls = _fake_evaluator(monkeypatch, [first, {"status": "ok", "correct": True, "speedup": 2.0}])
    job = gpuqueue.Job(kind="eval", job_class=gpuqueue.EVAL)
    with gpuqueue.using(job):
        result = evaluate.run_evaluation(tmp_path / "c.pt", tmp_path / "k.py")
    assert len(calls) == 2 and result["speedup"] == 2.0
    assert "bench.py" in result["retimed"] and "ended during" in result["retimed"]
    assert "timing_dirty" not in result
    assert job.holds == 2 and not job.front  # requeued at the head, then admitted
    assert all(env[gpulock.HOLD_ENV] and env[hygiene.NICE_ENV] == "-5" for env in calls)


def test_a_measurement_dirty_twice_is_kept_and_flagged(tmp_path, monkeypatch, clean):
    busy = {"status": "ok", "correct": True, "speedup": 1.1, "cpu_wait_share": 0.4}
    calls = _fake_evaluator(monkeypatch, [busy, busy])
    monkeypatch.setattr(telemetry, "compute_processes", lambda gpu: [])
    result = evaluate.run_evaluation(tmp_path / "c.pt", tmp_path / "k.py")
    assert len(calls) == 2 and "waited for a CPU 40 %" in result["timing_dirty"]
    assert "waited for a CPU" in result["retimed"]


def test_a_reference_slowdown_is_measured_again_once(tmp_path, monkeypatch, clean):
    ok = {"status": "ok", "correct": True, "speedup": 1.5}
    calls = _fake_evaluator(monkeypatch, [ok, ok])
    monkeypatch.setattr(telemetry, "compute_processes", lambda gpu: [])
    seen = []

    def check(data, *args, **kwargs):
        seen.append(1)
        if len(seen) == 1:
            evaluate._violation(data, "reference_timing", "the reference runs slower")

    monkeypatch.setattr(evaluate, "_check_reference_timing", check)
    result = evaluate.run_evaluation(tmp_path / "c.pt", tmp_path / "k.py")
    assert len(calls) == 2 and result["status"] == "ok"
    assert "reference ran slower" in result["retimed"]


def test_quick_checks_and_runs_without_clean_timing_are_measured_once(tmp_path, monkeypatch):
    busy = {"status": "ok", "correct": True, "cpu_wait_share": 0.5}
    calls = _fake_evaluator(monkeypatch, [busy, busy])
    result = evaluate.run_evaluation(tmp_path / "c.pt", tmp_path / "k.py")
    assert len(calls) == 1 and "retimed" not in result and "timing_dirty" not in result
    with hygiene.active():
        result = evaluate.run_evaluation(tmp_path / "c.pt", tmp_path / "k.py", quick=True)
    assert len(calls) == 2 and "retimed" not in result


def test_call_worker_measures_a_dirty_e2e_again(tmp_path, monkeypatch, clean):
    outputs = [
        {"status": "ok", "passed": True, "speedup": 1.2, "cpu_wait_share": 0.3},
        {"status": "ok", "passed": True, "speedup": 1.5, "cpu_wait_share": 0.0},
    ]
    calls = []

    def run(cmd, *, env, **kwargs):
        calls.append(cmd)
        out = worker.MARKER + json.dumps(outputs[len(calls) - 1])
        return subprocess.CompletedProcess(cmd, 0, out, "")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(telemetry, "compute_processes", lambda gpu: [])
    result = worker.call_worker(RunDir(tmp_path), "e2e", "--warmup", "2")
    assert len(calls) == 2 and result["speedup"] == 1.5 and "waited for a CPU" in result["retimed"]
    calls.clear()
    worker.call_worker(RunDir(tmp_path), "capture")  # not a timed command: once
    assert len(calls) == 1


# ------------------------------------------------------------------ run_on_gpu


def test_run_on_gpu_runs_the_script_as_a_dev_job(tmp_path, monkeypatch, clean):
    run = RunDir.create(tmp_path, "org/m")
    home = run.target("t")
    home.mkdir(parents=True)
    (home / "probe.py").write_text(
        textwrap.dedent(
            """
            import json, os, sys
            print(json.dumps({
                "argv": sys.argv[1:],
                "visible": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
                "held": os.environ.get("KERNEL_AGENT_LOCK_HELD"),
                "secret": os.environ.get("ANTHROPIC_API_KEY"),
                "nice": os.nice(0),
                "cwd": os.getcwd(),
            }))
            """
        )
    )
    monkeypatch.setattr(tools_mod, "create_sdk_mcp_server", lambda n, version, tools: tools)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")  # the orchestrator's; blank for agents
    session = runner.agent_env({}, gpu=runner.GPU_TOOL) | {"ANTHROPIC_API_KEY": ""}
    binding = tools_mod.SessionBinding(label="kernel-t", role="kernel", target_id="t")
    server = {t.name: t for t in tools_mod.build_server(run, None, None, binding, env=session)}
    args = {"script": "probe.py", "args": ["--n", "3"], "timeout": 30, "mem_gb": 1.5}

    async def call(args):
        async with gpuqueue.SessionClock("kernel-t").running(None):  # Orchestrator._agent
            return await server["run_on_gpu"].handler(args)

    out = json.loads(asyncio.run(call(args))["content"][0]["text"])
    assert out["status"] == "ok" and out["returncode"] == 0 and out["shared"], out
    seen = json.loads(out["output"])
    assert seen["argv"] == ["--n", "3"] and seen["visible"] == "unset"  # the GPU is visible
    assert seen["held"] == "1" and seen["secret"] == "" and seen["cwd"] == str(home)
    assert seen["nice"] == hygiene.BACKGROUND_NICE  # a correctness run: background
    events = read_jsonl(run.root / gpuqueue.FILE)
    assert {(e["kind"], e["class"]) for e in events} == {("dev", gpuqueue.DEV)}
    assert all(e["session"] == "kernel-t" and e["shared"] for e in events)
    bench = asyncio.run(server["run_on_gpu"].handler({"script": "probe.py", "benchmark": True}))
    out = json.loads(bench["content"][0]["text"])
    assert "shared" not in out and json.loads(out["output"])["nice"] in (-5, os.nice(0))
    missing = asyncio.run(server["run_on_gpu"].handler({"script": "nope.py"}))
    assert "is not a file" in missing["content"][0]["text"]


def test_kill_stops_children_in_process_groups_of_their_own(tmp_path):
    """nvcc runs in a process group of its own: a timed-out prebuild or dev run stops it."""
    from kernel_agent import interrupt

    pid_file = tmp_path / "grandchild"
    code = (
        "import subprocess, sys, time; "
        "p = subprocess.Popen(['sleep', '60'], start_new_session=True); "
        f"open({str(pid_file)!r}, 'w').write(str(p.pid)); time.sleep(60)"
    )
    child = subprocess.Popen([sys.executable, "-c", code])
    deadline = time.monotonic() + 10
    while not (pid_file.exists() and pid_file.read_text()) and time.monotonic() < deadline:
        time.sleep(0.05)
    grandchild = int(pid_file.read_text())
    assert os.getpgid(grandchild) != os.getpgid(child.pid)
    interrupt.kill(child.pid)
    child.wait(10)
    deadline = time.monotonic() + 10
    while (stat := interrupt._stat(grandchild)) is not None and stat[2] != "Z":
        assert time.monotonic() < deadline, "the grandchild still runs"
        time.sleep(0.05)


def test_run_script_stops_a_script_at_its_timeout(tmp_path):
    (tmp_path / "slow.py").write_text("import time\nprint('started', flush=True)\ntime.sleep(60)\n")
    start = time.monotonic()
    out = devrun.run_script(tmp_path / "slow.py", cwd=tmp_path, timeout=1)
    assert out["status"] == "timeout" and "started" in out["output"]
    assert time.monotonic() - start < 20


# ------------------------------------------------------------------ prebuild


CANDIDATE = """
import json
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

LOG = Path(__file__).with_suffix(".log")


def _note(**what):
    with LOG.open("a") as fh:
        fh.write(json.dumps(what) + "\\n")


def _load():
    _note(fn="_load", cuda=torch.cuda.is_available(), visible=os.environ["CUDA_VISIBLE_DEVICES"],
          nice=os.nice(0), jobs=os.environ.get("MAX_JOBS"))
    if os.environ.get("KA_REAL_BUILD"):
        return load_inline(name="never", cpp_sources="")


def _variant(width):
    _note(fn="_variant", width=width)
    if os.environ.get("KA_REAL_BUILD"):
        return load_inline(name=f"never{width}", cpp_sources="")


def build(reference, WIDTH=4):
    _variant(WIDTH * reference.in_features)
    return reference
"""


def test_loaders_finds_the_getters_and_the_loaders_with_arguments():
    assert prebuild.loaders(CANDIDATE) == (["_load"], True)
    plain = "import torch.utils.cpp_extension as ce\ndef _ext():\n    return ce.load_inline()\n"
    assert prebuild.loaders(plain) == (["_ext"], False)
    other = "from torch.utils.cpp_extension import load\ndef build(r, n=1):\n    load('x', [])\n"
    assert prebuild.loaders(other) == ([], True)
    unrelated = "import torch\ndef _w():\n    return torch.load('w.pt')\n"
    assert prebuild.loaders(unrelated) == ([], False)
    assert prebuild.loaders("def (") == ([], False)


def test_inputs_capture_never_hands_out_the_answer_key(tmp_path):
    run = RunDir.create(tmp_path, "org/m")
    full = run.truth_dir / "captures" / "t.pt"
    assert prebuild.inputs_capture(full) is None  # no inputs-only copy
    inputs = run.target("t") / "capture_inputs.pt"
    inputs.parent.mkdir(parents=True)
    inputs.write_bytes(b"")
    assert prebuild.inputs_capture(full) == inputs
    assert prebuild.inputs_capture(run.target("t") / "capture.pt") == run.target("t") / (
        "capture.pt"
    )
    assert prebuild.inputs_capture(tmp_path / "c.pt") is None


def test_prebuild_skips_what_it_cannot_build(tmp_path, monkeypatch):
    plain = tmp_path / "triton_kernel.py"
    plain.write_text("import triton\ndef build(r):\n    return r\n")
    assert prebuild.prebuild(plain)["reason"] == "no load_inline"
    assert prebuild.prebuild(tmp_path)["status"] == "skipped"  # a project directory
    assert prebuild.prebuild(tmp_path / "missing.py")["reason"] == "unreadable"
    cand = tmp_path / "cand.py"
    cand.write_text(CANDIDATE)
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    assert "TORCH_CUDA_ARCH_LIST" in prebuild.prebuild(cand)["reason"]


def test_prebuild_calls_the_loaders_without_a_gpu(tmp_path, monkeypatch, clean):
    import torch

    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(p for p in sys.path if p))
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "12.0")
    run = RunDir.create(tmp_path, "org/m")
    inputs = run.target("t") / "capture_inputs.pt"
    inputs.parent.mkdir(parents=True)
    torch.save({"module": torch.nn.Linear(2, 2), "cases": [], "inputs_only": True}, inputs)
    cand = tmp_path / "cand.py"
    cand.write_text(CANDIDATE)
    capture = prebuild.inputs_capture(run.truth_dir / "captures" / "t.pt")
    out = prebuild.prebuild(cand, capture, [{"WIDTH": 1}, {"WIDTH": 3}])
    assert out["status"] == "ok" and out["built"] == [], out
    notes = [json.loads(line) for line in cand.with_suffix(".log").read_text().splitlines()]
    load = notes[0]
    assert load["fn"] == "_load" and not load["cuda"] and load["visible"] == ""
    assert load["nice"] == hygiene.BACKGROUND_NICE and load["jobs"] == str(max(1, len(CPUS) - 1))
    assert [n["width"] for n in notes[1:]] == [2, 6]  # build() per config, on the CPU


@pytest.mark.gpu
def test_prebuild_makes_the_evaluators_build_a_cache_hit(tmp_path, monkeypatch):
    """Not in the CPU suite: nvcc builds a tiny extension without the GPU, then a process
    that sees the GPU (as the evaluator does) loads it without compiling."""
    from kernel_agent import toolchain

    toolchain.setup()
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "ext"))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(p for p in sys.path if p))
    cand = tmp_path / "ext_cand.py"
    cand.write_text(
        textwrap.dedent(
            '''
            from torch.utils.cpp_extension import load_inline

            CUDA = """
            #include <torch/extension.h>
            __global__ void twice(float* x, int n) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n) x[i] *= 2.0f;
            }
            void run(torch::Tensor x) {
                int n = x.numel();
                twice<<<(n + 255) / 256, 256>>>(x.data_ptr<float>(), n);
            }
            """
            _ext = None


            def _load():
                global _ext
                if _ext is None:
                    _ext = load_inline(
                        name="ka_prebuild_test",
                        cpp_sources="void run(torch::Tensor x);",
                        cuda_sources=CUDA,
                        functions=["run"],
                    )
                return _ext


            def build(reference):
                return reference
            '''
        )
    )
    out = prebuild.prebuild(cand)
    assert out["status"] == "ok" and out["built"][0]["name"] == "ka_prebuild_test", out
    assert "error" not in out["built"][0]
    code = (
        "import time, torch; from kernel_agent import toolchain; toolchain.setup(); "
        "from kernel_agent.kernels.evaluate import load_candidate_module as load; "
        f"m = load(__import__('pathlib').Path({str(cand)!r})); t = time.perf_counter(); "
        "ext = m._load(); s = time.perf_counter() - t; x = torch.ones(8, device='cuda'); "
        "ext.run(x); print(s, x.sum().item())"
    )
    env = gpulock.child_env()  # the evaluator's environment (the test holds the GPU lock)
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    seconds, total = proc.stdout.split()[-2:]
    assert float(total) == 16.0, proc.stderr
    assert float(seconds) < min(5.0, out["built"][0]["seconds"] / 3), (seconds, out)
