"""KernelBench regression suite (issue #21): problem files -> captures, fast_p, a dry run.

CPU only, no Claude and no network: two tiny problems in KernelBench format
(``tests/fixtures/kernelbench/level1/``), a tarball of them served from a ``file://``
URL, and the suite's fake engineer (``--dry-run``).
"""

import asyncio
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import torch

from kernel_agent import charts, kernelbench, orchestrator, suite, truth
from kernel_agent.dryrun import SimToolchain
from kernel_agent.kernels.evaluate import evaluate
from kernel_agent.profiling.capture import load_capture
from kernel_agent.workspace import RunDir, read_json, write_json

FIXTURES = Path(__file__).parent / "fixtures" / "kernelbench"


def sealed_run(tmp_path) -> RunDir:
    run = RunDir.create(tmp_path, "KernelBench/level1")
    write_json(
        run.run_json, {"card": {"repo_id": "KernelBench/level1"}, "truth": truth.new_section()}
    )
    return run


def tarball(path: Path) -> Path:
    """The fixtures laid out like GitHub's archive of the repository, plus other files."""
    with tarfile.open(path, "w:gz") as tar:

        def add(name: str, data: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        for file in sorted((FIXTURES / "level1").glob("*.py")):
            add(f"KernelBench-main/KernelBench/level1/{file.name}", file.read_bytes())
        add("KernelBench-main/README.md", b"# KernelBench\n")
        add("KernelBench-main/src/eval.py", b"print('not a problem')\n")
        add("KernelBench-main/KernelBench/level1/notes.txt", b"not a problem\n")
    return path


def test_fetch_keeps_only_problem_files_and_caches_them(tmp_path):
    url = tarball(tmp_path / "kb.tar.gz").as_uri()
    root = kernelbench.fetch("main", cache=tmp_path / "cache", url=url)
    assert root == tmp_path / "cache" / "main" / "KernelBench"
    assert sorted(p.name for p in (root / "level1").iterdir()) == [
        "1_Linear_scale.py",
        "2_RMSNorm_.py",
    ]
    assert not (tmp_path / "cache" / "main" / "README.md").exists()
    # the second call reads the cache (this URL does not exist)
    again = kernelbench.fetch("main", cache=tmp_path / "cache", url=url + ".missing")
    assert again == root
    assert kernelbench.find_root(FIXTURES) == FIXTURES
    assert kernelbench.find_root(FIXTURES.parent) is None


def test_problems_by_id():
    found = kernelbench.problems(FIXTURES, 1)
    assert [(p.pid, p.name, p.target_id) for p in found] == [
        (1, "Linear_scale", "l1_001_linear_scale"),
        (2, "RMSNorm_", "l1_002_rmsnorm"),
    ]
    assert [p.pid for p in kernelbench.problems(FIXTURES, 1, n=1)] == [1]
    assert [p.pid for p in kernelbench.problems(FIXTURES, 1, ids=[2])] == [2]
    with pytest.raises(KeyError):
        kernelbench.problems(FIXTURES, 1, ids=[3])
    with pytest.raises(FileNotFoundError):
        kernelbench.problems(FIXTURES, 2)


def test_problem_files_become_sealed_captures(tmp_path):
    run = sealed_run(tmp_path)
    keeper = truth.of(run)
    for problem in kernelbench.problems(FIXTURES, 1):
        spec = kernelbench.prepare(
            run, problem, keeper, device="cpu", max_mb=16.0, backends=["triton"]
        )
        target_id = problem.target_id
        assert spec["id"] == target_id and spec["module_class"] == "Model"
        assert spec["kernelbench"]["scaled"] == {}  # small enough already
        assert read_json(run.target(target_id) / "spec.json") == spec

        # the sealed copy of the module and the capture: digests recorded
        sealed = run.truth_dir / "kernelbench" / f"{problem.module_name}.py"
        assert sealed.read_text() == problem.path.read_text()
        capture = run.capture_file(target_id)
        assert keeper.verify(sealed) and keeper.verify(capture)

        # one timed case from seed 0, two correctness-only cases from seeds 1 and 2
        data = load_capture(capture, sha256=keeper.expect(capture))
        cases = data["cases"]
        assert [c["count"] for c in cases] == [1, 0, 0]
        assert [c.get("correctness_only", False) for c in cases] == [False, True, True]
        assert [c["count"] for c in spec["capture"]["cases"]] == [1, 0, 0]
        inputs = [c["args"][0] for c in cases]
        assert all(x.shape == inputs[0].shape for x in inputs)
        assert not torch.equal(inputs[0], inputs[1]) and not torch.equal(inputs[1], inputs[2])
        torch.manual_seed(0)
        assert torch.equal(inputs[0], problem_inputs(problem)[0])  # get_inputs() at seed 0
        model = data["module"]
        assert type(model).__name__ == "Model" and type(model).__module__ == problem.module_name
        for case in cases:  # the outputs are the module's on the captured inputs
            torch.testing.assert_close(model(*case["args"]), case["output"], rtol=0, atol=0)

        # the agent's copy has no answers; reference_source.py shows the problem
        agent = torch.load(run.target(target_id) / "capture_inputs.pt", weights_only=False)
        assert agent["inputs_only"] and "output" not in agent["cases"][0]
        source = (run.target(target_id) / "reference_source.py").read_text()
        assert f"problem {problem.pid}: {problem.path.name}" in source
        assert "def get_inputs" in source

        # the evaluator takes it like any capture: an honest rewrite passes every case
        _, text, _ = suite.dry_run_candidate(spec, source, triton=False)
        candidate = tmp_path / f"{target_id}_rewrite.py"
        candidate.write_text(text)
        result = evaluate(capture, candidate, device="cpu", capture_sha256=keeper.expect(capture))
        assert result["status"] == "ok" and len(result["cases"]) == 3, result

    # the pickled Model loads in a fresh process without any path setup (by value)
    capture = run.capture_file("l1_002_rmsnorm")
    code = (
        "import sys, torch; from kernel_agent.profiling.capture import load_capture; "
        "m = load_capture(sys.argv[1])['module']; print(type(m).__module__, m.eps)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(capture)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )
    assert proc.stdout.split() == ["ka_kernelbench_l1_002_rmsnorm", "1e-05"], proc.stderr


def problem_inputs(problem: kernelbench.Problem) -> list:
    module = kernelbench.load_module(problem.path.read_text(), "ka_test_inputs", "x.py")
    return list(module.get_inputs())


def test_sizes_are_scaled_to_fit_and_written_into_the_source(tmp_path):
    problem = kernelbench.problems(FIXTURES, 1, ids=[1])[0]
    module = kernelbench.load_module(problem.path.read_text(), "ka_test_fit", "x.py")
    assert kernelbench.footprint(module) == 512 * (64 + 32) * 4
    changed, size = kernelbench.fit_sizes(module, 50_000)
    # batch_size first: in_features is also read by get_init_inputs (the model's shape)
    assert changed == {"batch_size": [512, 128]} and size == 128 * 96 * 4
    assert module.batch_size == 128 and module.in_features == 64

    run = sealed_run(tmp_path)
    spec = kernelbench.prepare(
        run, problem, truth.of(run), device="cpu", max_mb=50_000 / 2**20, backends=["triton"]
    )
    assert spec["kernelbench"]["scaled"] == {"batch_size": [512, 128]}
    sealed = (run.truth_dir / "kernelbench" / f"{problem.module_name}.py").read_text()
    assert sealed.startswith(problem.path.read_text().rstrip("\n"))
    assert sealed.rstrip().endswith("batch_size = 128  # KernelBench: 512")
    data = load_capture(run.capture_file(problem.target_id))
    assert [tuple(c["args"][0].shape) for c in data["cases"]] == [(128, 64)] * 3

    # a value the model rejects is put back: in_features must stay a multiple of 4 here
    source = problem.path.read_text()
    for a, b in (
        ("        self.linear =", "        assert in_features % 4 == 0\n        self.linear ="),
        ("(self, x: torch.Tensor)", "(self, x: torch.Tensor, y: torch.Tensor)"),
        ("self.linear(x) * self.scale", "self.linear(x + y.sum()) * self.scale"),
        ("    return [x]", "    return [x, torch.rand(in_features)]"),
    ):
        assert a in source
        source = source.replace(a, b)
    module = kernelbench.load_module(source, "ka_test_fit2", "y.py")
    changed, size = kernelbench.fit_sizes(module, 150)  # cannot fit: the output is 128 bytes
    assert changed == {"batch_size": [512, 1], "in_features": [64, 4]}
    assert module.in_features == 4 and size == 4 * 4 + 4 * 4 + 32 * 4


def test_fast_p_counts_every_problem():
    rows = [
        {"correct": True, "speedup": 2.5, "speedup_vs_compile": 1.2},
        {"correct": True, "speedup": 1.3, "speedup_vs_compile": 0.9},
        {"correct": True, "speedup": 1.05, "speedup_vs_compile": None},  # compile failed
        {"correct": True, "speedup": None},  # not timed (CPU)
        {"correct": False, "speedup": 3.0},  # an incorrect kernel never counts
        {"correct": False, "status": "capture_failed"},
    ]
    assert suite.fast_p(rows) == {
        "1": 0.5,
        "1.1": 0.3333,
        "1.25": 0.3333,
        "1.5": 0.1667,
        "2": 0.1667,
    }
    assert suite.fast_p(rows, "speedup_vs_compile") == {
        "1": 0.1667,
        "1.1": 0.1667,
        "1.25": 0.0,
        "1.5": 0.0,
        "2": 0.0,
    }
    assert suite.fast_p([]) == dict.fromkeys(("1", "1.1", "1.25", "1.5", "2"), 0.0)
    summary = suite.summarise([{**r, "usd": 0.5, "evaluations": 2} for r in rows])
    assert summary["problems"] == 6 and summary["correct"] == 4
    assert summary["usd"] == 3.0 and summary["evaluations"] == 12
    assert summary["geomean_speedup"] == pytest.approx((2.5 * 1.3 * 1.05) ** (1 / 3), abs=1e-3)


def test_dry_run_candidates():
    spec = {"kernelbench": {"name": "RMSNorm_"}}
    source = "import torch\nclass Model(torch.nn.Module):\n    pass\n"
    name, text, _ = suite.dry_run_candidate(spec, source, triton=True)
    assert name == "triton_rmsnorm_v1.py" and "triton_rmsnorm.py" in text
    name, text, _ = suite.dry_run_candidate(spec, source, triton=False)
    assert name == "torch_rewrite_v1.py" and "class Rewrite(" in text and "Model" not in text


def test_suite_dry_run_end_to_end_on_cpu(tmp_path, monkeypatch):
    """Fixtures -> sealed captures -> Orchestrator.kernels() with the fake engineer ->
    re-evaluation with the torch.compile baseline -> suite.json / suite.md (CPU: correctness
    only, so nothing is faster: fast_p 0)."""
    monkeypatch.setattr(orchestrator.toolchain, "setup", SimToolchain)
    monkeypatch.setattr(suite.toolchain, "setup", SimToolchain)
    monkeypatch.setattr(charts, "available", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # the evaluator's subprocesses: CPU
    monkeypatch.setenv("KERNEL_AGENT_LOCK_HELD", "1")  # no GPU to lock
    cfg = suite.SuiteConfig(
        kernelbench_dir=FIXTURES,
        n=2,
        evaluations=1,
        dry_run=True,
        device="cpu",
        runs_dir=tmp_path,
        backends=["triton"],
    )
    run = asyncio.run(suite.run_suite(cfg))

    data = json.loads((run.root / suite.SUITE_JSON).read_text())
    rows = data["rows"]
    assert [r["target"] for r in rows] == ["l1_001_linear_scale", "l1_002_rmsnorm"]
    for row in rows:
        assert row["correct"] and row["status"] == "ok", row
        assert row["evaluations"] == 1 and row["usd"] == 0.0
        assert "torch_rewrite_v1" in row["snapshot"]
        assert row["speedup"] is None  # CPU: not timed
    summary = data["summary"]
    assert summary["problems"] == 2 and summary["correct"] == 2
    assert summary["fast_p"]["eager"] == dict.fromkeys(("1", "1.1", "1.25", "1.5", "2"), 0.0)
    loaded = run.load()
    assert loaded["config"]["use_library"] is False and loaded["config"]["do_transforms"] is False
    assert set(loaded["phases"]["kernels"]["finished"]) == {r["target"] for r in rows}
    costs = read_json(run.root / "costs.json")
    assert set(costs) == {"kernel-l1_001_linear_scale", "kernel-l1_002_rmsnorm"}
    md = (run.root / suite.SUITE_MD).read_text()
    assert "# KernelBench level 1 suite" in md and "| vs torch.compile |" in md
    assert "fake engineer (--dry-run)" in md and "`Linear_scale`" in md


def test_bench_suite_command_line(monkeypatch):
    from kernel_agent import cli

    seen = {}
    monkeypatch.setattr(suite, "main", lambda ns: seen.update(vars(ns)) or 0)
    assert cli.main(["bench-suite", "--kernelbench-level", "1", "--n", "5", "--dry-run"]) == 0
    assert seen["kernelbench_level"] == 1 and seen["n"] == 5 and seen["dry_run"]
    assert seen["problems"] is None and seen["evaluations"] == 4
    cli.main(["bench-suite", "--problems", "1,19,36"])
    assert seen["problems"] == [1, 19, 36]
