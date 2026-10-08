"""End-to-end hidden-work check (#147): the analysis of synthetic profiler traces (joined
and unjoined side streams, late threads, other devices, graph replays, spoofed ranges),
the thread scan and the worker's verdict, on the CPU; the profiled run itself on the GPU
(marked)."""

from __future__ import annotations

import functools
import importlib.util
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import torch
from test_truth import sealed_run

from kernel_agent import toolchain, truth, worker
from kernel_agent.kernels import e2e_activity
from kernel_agent.kernels.e2e_activity import Event, analyse, live_threads
from kernel_agent.workloads.base import WorkloadSpec
from kernel_agent.workspace import RunDir

TOY = Path(__file__).with_name("chaotic_toy.py")
TAG = "#t1"
MAIN, OTHER = 1, 2  # threads
CALLER, AUX, RAW = 7, 9, 11  # streams


class Trace:
    """A synthetic profile of one checked run: ``run()`` from 0 to 1000 ns with one kernel
    on the caller's stream, then the check's markers (the caller's stream drains at 1200)."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self.corr = 0
        self.range(f"ka::run{TAG}", 0, 1000)
        self.launch(10, CALLER, 100, 900)
        self.range(f"ka::mark{TAG}", 1010, 1020)
        self.launch(1012, CALLER, 1200, 1201, name="spin")
        self.range(f"ka::stream:aux{TAG}", 1030, 1040)
        self.launch(1032, AUX, 1300, 1301, name="spin")

    def range(self, name: str, start: int, end: int) -> None:
        self.events.append(Event("user_annotation", name, start, end, 0, MAIN))

    def launch(
        self,
        at: int,
        stream: int,
        start: int,
        end: int,
        *,
        thread: int = MAIN,
        name: str = "gemm",
        device: int = 0,
        kernels: int = 1,
    ) -> None:
        self.corr += 1
        self.events.append(Event("cuda_runtime", "cudaLaunchKernel", at, at + 3, self.corr, thread))
        step = (end - start) // kernels
        for k in range(kernels):  # a graph launch: several kernels, one correlation id
            begin = start + k * step
            self.events.append(
                Event("kernel", name, begin, begin + step, self.corr, stream, device)
            )

    def verdict(self, **kwargs: Any) -> dict[str, Any]:
        return analyse(self.events, devices=kwargs.pop("devices", {0}), tag=TAG, **kwargs)


def test_joined_streams_pass_and_raw_ones_are_a_note() -> None:
    trace = Trace()
    trace.launch(20, AUX, 150, 1100)  # declared, joined: ends before the caller drains
    verdict = trace.verdict()
    assert verdict["passed"] and verdict["reason"] == "", verdict
    assert "undeclared_streams" not in verdict
    trace.launch(30, RAW, 150, 1150)  # a raw torch.cuda.Stream(), joined
    verdict = trace.verdict()
    assert verdict["passed"] and verdict["undeclared_streams"] == ["stream 11: 1 GPU operations"]


def test_an_unjoined_side_stream_is_hidden_work() -> None:
    trace = Trace()
    trace.launch(30, RAW, 150, 5000)  # still running when the caller's stream drained
    verdict = trace.verdict()
    assert not verdict["passed"] and verdict["unjoined"], verdict
    assert verdict["reason"].startswith("hidden work: GPU work on a stream the caller's")
    assert "cc.fork" in verdict["reason"]


def test_late_and_foreign_threads_are_hidden_work() -> None:
    late = Trace()
    late.launch(1500, 13, 1600, 1700, thread=OTHER)  # a thread launches after run()
    verdict = late.verdict()
    assert not verdict["passed"]
    assert verdict["foreign_threads"] and verdict["late_launches"], verdict
    during = Trace()
    during.launch(500, CALLER, 600, 700, thread=OTHER)  # synchronous, but another thread
    verdict = during.verdict()
    assert not verdict["passed"] and verdict["foreign_threads"] and not verdict["late_launches"]
    main_late = Trace()
    main_late.launch(1100, CALLER, 1210, 1220)  # the calling thread, outside the check's markers
    verdict = main_late.verdict()
    assert not verdict["passed"] and verdict["late_launches"] and not verdict["foreign_threads"]


def test_work_on_another_device_fails_unless_the_workload_uses_it() -> None:
    trace = Trace()
    trace.launch(40, 21, 150, 800, device=1)
    assert not trace.verdict()["passed"]
    assert trace.verdict()["other_devices"] == ["gemm on cuda:1"]
    assert trace.verdict(devices={0, 1})["passed"]


def test_a_graph_replay_on_the_callers_stream_passes() -> None:
    trace = Trace()
    trace.launch(50, CALLER, 120, 990, kernels=28, name="graph_kernel")
    assert trace.verdict()["passed"]


def test_ranges_without_the_checks_tag_are_not_trusted() -> None:
    trace = Trace()
    trace.range("ka::mark", 400, 450)  # the code under test names its own range like ours
    trace.launch(410, RAW, 420, 5000)
    verdict = trace.verdict()
    assert not verdict["passed"] and verdict["unjoined"], verdict


def test_what_run_leaves_behind_fails_it() -> None:
    for kwargs, needle in (
        ({"outstanding": ["aux"]}, "handles not joined"),
        ({"stream_changed": True}, "another current stream"),
        ({"live_threads": ["Thread-3 (late.py)"]}, "threads still run"),
    ):
        verdict = Trace().verdict(**kwargs)
        assert not verdict["passed"] and needle in verdict["reason"], (kwargs, verdict)


def test_no_gpu_activity_is_a_note() -> None:
    verdict = analyse([Event("user_annotation", "ka::run", 0, 10, 0, MAIN)], devices={0})
    assert verdict["passed"] and "no GPU activity" in verdict["note"]


# ------------------------------------------------------------------ threads


def _module(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "def wait(event, entered=None):\n"
        "    if entered is not None:\n"
        "        entered.release()  # this thread runs the evaluated code now\n"
        "    event.wait(30)\n\n\n"
        "def noop():\n    pass\n",
    )
    spec = importlib.util.spec_from_file_location(f"ka_threads_{path.parent.name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_threads_finds_threads_running_the_evaluated_code(tmp_path: Path) -> None:
    module = _module(tmp_path / "transforms" / "late.py")
    stop, entered = threading.Event(), threading.Semaphore(0)
    direct = threading.Thread(target=module.wait, args=(stop, entered), name="direct")
    wrapped = threading.Thread(target=functools.partial(module.wait, stop, entered), name="wrapped")
    timer = threading.Timer(60, module.noop)
    timer.name = "timer"
    for t in (direct, wrapped, timer):
        t.start()
    try:
        # both are inside module.wait (the wrapped one is found by its frames), however
        # long the scheduler took to start them
        assert entered.acquire(timeout=60) and entered.acquire(timeout=60)
        found = live_threads([tmp_path / "transforms"])
        assert {f.split(" ")[0] for f in found} == {"direct", "wrapped", "timer"}, found
        assert all("late.py" in f for f in found)
        assert live_threads([tmp_path / "elsewhere"]) == []
        assert live_threads([]) == []
    finally:
        stop.set()
        timer.cancel()
        for t in (direct, wrapped, timer):
            t.join(5)


def test_check_skips_without_a_gpu_workload(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(inputs: Any) -> None:
        raise AssertionError("not profiled")

    assert e2e_activity.check(run, None, device=torch.device("cpu"))["skipped"].endswith("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    verdict = e2e_activity.check(run, None, device=0)
    assert verdict == {"passed": True, "reason": "", "skipped": "no CUDA device"}


# ------------------------------------------------------------------ worker


def _call(capsys: Any, run: RunDir, *argv: Any) -> dict[str, Any]:
    worker.main([str(a) for a in (argv[0], "--run-dir", run.root, *argv[1:])])
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith(worker.MARKER)]
    return json.loads(line[-1][len(worker.MARKER) :])


def test_worker_fails_an_e2e_evaluation_with_hidden_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(toolchain, "setup", lambda *a, **k: None)
    spec = WorkloadSpec(repo_id="toy/chaotic", modality="tts", device="cpu", harness=str(TOY))
    run = sealed_run(tmp_path, workload=spec.to_dict())
    baseline = _call(capsys, run, "analyze", "--no-profile", "--iters", 1)
    keeper = truth.of(run)
    keeper.seal_baseline(baseline["median_ms"])
    transform = tmp_path / "transforms" / "noop.py"
    transform.parent.mkdir()
    transform.write_text("def apply(workload):\n    pass\n")

    clean = _call(capsys, run, "e2e", "--transform", transform, "--iters", 1, *keeper.worker_args())
    assert clean["passed"], clean
    assert clean["metrics"]["concurrency"]["skipped"] == "the workload runs on cpu"

    seen: dict[str, Any] = {}
    verdicts = [
        {"passed": True, "reason": "", "streams": ["aux"]},
        {"passed": False, "reason": "hidden work: GPU work launched after run() returned"},
    ]

    def fake_check(run_fn: Any, inputs: Any, **kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return verdicts.pop(0)

    monkeypatch.setattr(e2e_activity, "check", fake_check)
    declared = _call(
        capsys, run, "e2e", "--transform", transform, "--iters", 1, *keeper.worker_args()
    )
    assert declared["passed"] and declared["streams"] == ["aux"], declared
    assert seen["dirs"] == [transform.parent.resolve()] and seen["device"] == torch.device("cpu")
    hidden = _call(
        capsys, run, "e2e", "--transform", transform, "--iters", 1, *keeper.worker_args()
    )
    assert hidden["status"] == "ok" and not hidden["passed"], hidden
    assert hidden["reason"] == "hidden work: GPU work launched after run() returned"
    assert not hidden["metrics"]["concurrency"]["passed"]


# ------------------------------------------------------------------ GPU (not run on CPU)


@pytest.mark.gpu
def test_profiled_run_rejects_unjoined_streams_and_late_threads(tmp_path: Path) -> None:
    from kernel_agent import concurrency as cc

    x = torch.randn(2048, 2048, device="cuda")

    def heavy(t: torch.Tensor) -> torch.Tensor:
        for _ in range(4):
            t = torch.tanh(t @ x)
        return t

    def joined(inputs: torch.Tensor) -> torch.Tensor:
        with cc.fork("e2e-test"):
            a = heavy(inputs)
        return a + heavy(inputs)

    side = torch.cuda.Stream()

    def unjoined(inputs: torch.Tensor) -> torch.Tensor:
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            return heavy(inputs)

    late_dir = tmp_path / "transforms"
    module = _module(late_dir / "late.py")
    stop = threading.Event()

    def late(inputs: torch.Tensor) -> torch.Tensor:
        def work() -> None:
            time.sleep(0.005)
            with torch.inference_mode():
                heavy(inputs)
            module.wait(stop)

        threading.Thread(target=work, daemon=True).start()
        return inputs

    try:
        ok = e2e_activity.check(joined, x, device=0, dirs=[late_dir])
        assert ok["passed"] and "e2e-test" in ok["streams"], ok
        bad = e2e_activity.check(unjoined, x, device=0)
        assert not bad["passed"] and bad["unjoined"], bad
        thread = e2e_activity.check(late, x, device=0, dirs=[late_dir])
        assert not thread["passed"], thread
        assert thread["foreign_threads"] and thread["late_launches"], thread
    finally:
        stop.set()
        torch.cuda.synchronize()
