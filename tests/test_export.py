"""A self-contained ``optimized/`` (issue #171): the export copies the run files its items
load (a snapshot by name, a helper module, a lookup recorded through
``kernel_agent.artifacts``) to where their own lookup finds them, and its self-test applies
the package in a fresh process outside the run directory, naming what it lacks.

CPU only: the chaotic toy workload (``chaotic_toy.py``) as the run's harness, the worker
commands called in this process (``export_check`` from the package copy's directory).
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from test_recheck import kernel, make, simulated, worker  # noqa: F401 (fixture)
from test_truth import sealed_run
from torch import nn

from kernel_agent import artifacts, ledger, toolchain, truth
from kernel_agent import worker as worker_mod
from kernel_agent.agent.tools import snapshot
from kernel_agent.integrate import deps, export
from kernel_agent.integrate.export import check_export, export_optimized, hidden, missing
from kernel_agent.workloads.base import WorkloadSpec
from kernel_agent.workspace import RunDir, read_json, read_jsonl, write_json

TOY = Path(__file__).with_name("chaotic_toy.py")

#: What the transform loads: a one-rounding-step weight change (passes teacher forcing).
HELPER = "FACTOR = 1 + 2**-10\n"

#: The issue's pattern: a transform that loads another snapshot of the run by file name
#: from its own directory, ``history/`` or ``../history/`` (as the Qwen3 run's
#: ``prompt_lookup_decode_gqa_verify`` loads its decode-step engine's bundle).
BY_NAME = """import importlib.util
from pathlib import Path

import torch

_HELPER = "{helper}"


def _load():
    here = Path(__file__).resolve().parent
    for d in (here, here / "history", here.parent / "history"):
        p = d / _HELPER
        if p.exists():
            spec = importlib.util.spec_from_file_location("toy_helper", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(_HELPER)


def apply(workload):
    helper = _load()
    with torch.no_grad():
        workload.model.backbone.weight.mul_(helper.FACTOR)
"""

#: Through the helper, with a name computed at run time: only the recorded lookup knows it.
COMPUTED = """import torch

from kernel_agent import artifacts

_PARTS = ("{head}", "{tail}")


def apply(workload):
    helper = artifacts.load("".join(_PARTS))
    with torch.no_grad():
        workload.model.backbone.weight.mul_(helper.FACTOR)
"""

#: Reads the run directory by an absolute path and falls back silently when it cannot.
ABSOLUTE = """import torch

_PATH = {path!r}


def apply(workload):
    try:
        factor = float(open(_PATH).read().split("=")[1].split("+")[0]) + 2**-10
    except OSError:
        factor = 1.0
    with torch.no_grad():
        workload.model.backbone.weight.mul_(factor)
"""


@pytest.fixture
def cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(toolchain, "setup", lambda *a, **k: None)


def _call(capsys, run: RunDir, *argv: Any) -> dict[str, Any]:
    worker_mod.main([str(a) for a in (argv[0], "--run-dir", run.root, *argv[1:])])
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith(worker_mod.MARKER)]
    return json.loads(line[-1][len(worker_mod.MARKER) :])


def in_process(capsys, seen: list[dict[str, Any]]):
    """``call_worker`` in this process: the command runs in ``cwd`` (the package copy's
    directory), as the real worker process would."""

    def call(run: RunDir, command: str, *args: str, cwd=None, timeout=None) -> dict[str, Any]:
        before = os.getcwd()
        os.chdir(cwd)
        try:
            seen.append({"command": command, "cwd": os.getcwd(), "args": list(args)})
            return _call(capsys, run, command, *args)
        finally:
            os.chdir(before)

    return call


def toy_run(tmp_path: Path, capsys) -> RunDir:
    """A sealed run of the chaotic toy with its baseline (``worker analyze``)."""
    spec = WorkloadSpec(repo_id="toy/chaotic", modality="tts", device="cpu", harness=str(TOY))
    run = sealed_run(tmp_path, workload=spec.to_dict())
    _call(capsys, run, "analyze", "--no-profile", "--iters", 1)
    run.transforms_dir.mkdir(parents=True, exist_ok=True)
    return run


def evaluated(run: RunDir, name: str, source: str) -> Path:
    """The agent's ``transforms/<name>.py`` snapshotted as an evaluation does (``.truth/``)."""
    src = run.transforms_dir / f"{name}.py"
    src.write_text(source)
    return snapshot(run, src)


# ------------------------------------------------------------------ kernel_agent.artifacts


def test_artifacts_find_load_and_record(tmp_path, monkeypatch):
    history = tmp_path / "transforms" / "history"
    history.mkdir(parents=True)
    (history / "001_helper_0a1b2c3d.py").write_text(HELPER)
    user = history / "002_user_1a2b3c4d.py"
    user.write_text(
        "from kernel_agent import artifacts\n\n"
        "def run():\n"
        '    return artifacts.load("001_helper_0a1b2c3d.py").FACTOR\n'
    )
    log = tmp_path / "logs" / "artifacts.jsonl"
    monkeypatch.setenv(artifacts.LOG_ENV, str(log))
    spec = importlib.util.spec_from_file_location("ka_test_user", user)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.run() == 1 + 2**-10  # next to the calling file

    entry = artifacts.lookups()[-1]
    assert entry["caller"] == str(user) and entry["rel"] == "001_helper_0a1b2c3d.py"
    assert read_jsonl(log)[-1]["caller_sha256"] == hashlib.sha256(user.read_bytes()).hexdigest()
    assert artifacts.recorded(tmp_path)[-1]["path"] == str(history / "001_helper_0a1b2c3d.py")

    # the agent's copy one level up finds it in history/; a missing name is named
    agent = tmp_path / "transforms" / "user.py"
    assert (
        artifacts.find("001_helper_0a1b2c3d.py", near=agent) == history / "001_helper_0a1b2c3d.py"
    )
    assert artifacts.lookups()[-1]["rel"] == "history/001_helper_0a1b2c3d.py"
    with pytest.raises(FileNotFoundError) as info:
        artifacts.find("009_gone_00000000.py", near=agent)
    assert info.value.filename == "009_gone_00000000.py"


# ------------------------------------------------------------------ dependency scan


def test_needs_follow_names_imports_projects_and_recorded_lookups(tmp_path):
    root = tmp_path / "run"
    history = root / "transforms" / "history"
    history.mkdir(parents=True)
    (history / "001_helper_0a1b2c3d.py").write_text('DEEP = "005_deep_2c2c2c2c.py"\n')
    (history / "003_other_0000aaaa.py").write_text("X = 1\n")
    (history / "004_dyn_1111bbbb.py").write_text("Y = 2\n")
    (history / "005_deep_2c2c2c2c.py").write_text("Z = 3\n")
    (history / "toy_helper.py").write_text("W = 4\n")
    project = root / "transforms" / "engine"
    (project / "csrc").mkdir(parents=True)
    (project / "kernel_project.toml").write_text('[project]\nname = "engine"\n')
    (project / "csrc" / "step.cu").write_text("// step\n")
    (project / "__pycache__").mkdir()
    (project / "__pycache__" / "x.pyc").write_bytes(b"\0")
    elsewhere = tmp_path / "outside.py"
    elsewhere.write_text("")
    item = history / "006_main_3d3d3d3d.py"
    item.write_text(
        "import os\nimport toy_helper\nfrom kernel_agent import artifacts\n"
        'A = "001_helper_0a1b2c3d.py"\n'
        'B = "003_other_0000aaaa"  # a snapshot name without .py\n'
        f'C = "{elsewhere}"  # outside the run: not part of it\n'
        'D = "../engine"  # a native project directory\n'
        'E = "not_a_file.py"\n'
        'F = "".join(("004_dyn", "_1111bbbb.py"))  # only the recorded lookup knows it\n'
    )
    records = [
        {
            "caller_sha256": hashlib.sha256(item.read_bytes()).hexdigest(),
            "path": str(history / "004_dyn_1111bbbb.py"),
            "rel": "004_dyn_1111bbbb.py",
        },
        {"caller_sha256": "0" * 64, "path": str(history / "003_other_0000aaaa.py")},  # not it
    ]
    found = {n.place: n for n in deps.needs(item, root, records)}
    assert set(found) == {
        "001_helper_0a1b2c3d.py",
        "003_other_0000aaaa.py",
        "004_dyn_1111bbbb.py",
        "005_deep_2c2c2c2c.py",  # named by what the item needs: followed
        "toy_helper.py",
        "../engine/kernel_project.toml",  # its files, at the same place (no build products)
        "../engine/csrc/step.cu",
    }
    assert found["004_dyn_1111bbbb.py"].how == "recorded"
    assert found["toy_helper.py"].how == "import"
    assert found["005_deep_2c2c2c2c.py"].by == "001_helper_0a1b2c3d.py"
    assert found["../engine/csrc/step.cu"].how == "project"
    # more than one level up leaves the package: next to the item, by name
    assert deps._place("", "../../targets/t/history/x.py", "x.py") == "x.py"
    assert deps._place("", "../apply.py", "apply.py") == "apply.py"
    assert deps._place("", "../history/x.py", "x.py") == "../history/x.py"


# ------------------------------------------------------------------ the hidden run directory


def test_hidden_refuses_reads_of_the_run_directory(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "x.py").write_text("x")
    (run / "harness.py").write_text("h")
    with hidden(run, allow=[run / "harness.py"]) as refused:
        with pytest.raises(FileNotFoundError) as info:
            (run / "x.py").read_text()
        assert (run / "harness.py").read_text() == "h"
        assert (tmp_path / "elsewhere.txt").write_text("ok") == 2
    assert (run / "x.py").read_text() == "x"  # only while hidden
    assert refused == [str((run / "x.py").resolve())]
    assert missing(info.value, refused, run) == ["x.py"]
    named = FileNotFoundError("028_engine_2c930eef.py")  # what the issue's transform raises
    assert missing(named, [], run) == ["028_engine_2c930eef.py"]


# ------------------------------------------------------------------ export + self-test


def test_export_copies_the_snapshot_a_transform_loads_and_self_tests(tmp_path, cpu, capsys):
    run = toy_run(tmp_path, capsys)
    helper = evaluated(run, "factor", HELPER)
    main = evaluated(run, "by_name", BY_NAME.format(helper=helper.name))
    assert main.parent == helper.parent == run.root / ".truth/transforms/history"

    out = export_optimized(run, [("transform", str(main), 0.0)])
    assert (out / "transforms" / helper.name).read_text() == HELPER
    entry = read_json(out / "manifest.json")["transforms"][0]
    assert entry["file"] == f"transforms/{main.name}"
    assert [(n["file"], n["source"], n["how"]) for n in entry["needs"]] == [
        (f"transforms/{helper.name}", f".truth/transforms/history/{helper.name}", "name")
    ]

    seen: list[dict[str, Any]] = []
    call = in_process(capsys, seen)
    result = check_export(run, call, expected={"replaced": {}})
    assert result["passed"], result
    assert result["transforms"] == [f"transforms/{main.name}"]
    assert seen[0]["command"] == "export_check"
    package = Path(seen[0]["args"][1])
    assert seen[0]["cwd"] == str(package.parent) and not package.is_relative_to(run.root)
    assert read_json(out / "export_check.json")["passed"]
    again = check_export(run, call)  # the same package: the passing result is reused
    assert again["reused"] and len(seen) == 1

    # The issue's export: only the transform. The self-test fails and names the file.
    (out / "transforms" / helper.name).unlink()
    failed = check_export(run, call)
    assert not failed["passed"] and failed["status"] == "missing"
    assert failed["missing"] == [helper.name]
    assert f"not self-contained, missing {helper.name}" in failed["reason"]
    assert len(read_jsonl(run.root / "logs" / "export_checks.jsonl")) == 2


def test_recorded_lookup_and_reads_of_the_run_directory(tmp_path, cpu, capsys):
    run = toy_run(tmp_path, capsys)
    helper = evaluated(run, "factor", HELPER)
    head, tail = helper.name[:4], helper.name[4:]
    computed = evaluated(run, "computed", COMPUTED.format(head=head, tail=tail))

    out = export_optimized(run, [("transform", str(computed), 0.0)])
    assert not (out / "transforms" / helper.name).exists()  # no literal names it

    # an evaluation (worker e2e) records the lookup; the next export copies the file
    e2e = _call(capsys, run, "e2e", "--transform", computed, "--iters", 1)
    assert e2e["passed"], e2e
    assert artifacts.recorded(run.root)[-1]["path"] == str(helper)
    assert artifacts.LOG_ENV not in os.environ  # restored after the command
    out = export_optimized(run, [("transform", str(computed), 0.0)])
    assert (out / "transforms" / helper.name).exists()
    needs = read_json(out / "manifest.json")["transforms"][0]["needs"]
    assert [n["how"] for n in needs] == ["recorded"]
    seen: list[dict[str, Any]] = []
    assert check_export(run, in_process(capsys, seen))["passed"]

    # an absolute path into the run: the fallback hides it, the self-test does not
    absolute = evaluated(
        run, "absolute", ABSOLUTE.format(path=str(run.transforms_dir / "factor.py"))
    )
    export_optimized(run, [("transform", str(absolute), 0.0)])
    failed = check_export(run, in_process(capsys, seen))
    assert not failed["passed"] and failed["missing"] == ["transforms/factor.py"]
    assert "opened files of the run directory" in failed["reason"]


def test_export_puts_kernel_helpers_on_the_import_path(tmp_path):
    """A kernel that imports a module next to it (the run's loaders put its directory on
    ``sys.path``): exported next to ``kernels/<target>.py``, imported by ``apply.py``."""
    run = sealed_run(tmp_path)
    write_json(run.target("lin") / "spec.json", {"module_class": "Linear"})
    history = run.history_dir("lin")
    history.mkdir(parents=True)
    (history / "ka_lin_helper.py").write_text("SCALE = 2.0\n")
    candidate = history / "001_lin_0a0a0a0a.py"
    candidate.write_text(
        "import ka_lin_helper\nfrom torch import nn\n\n\n"
        "class Scaled(nn.Module):\n"
        "    def __init__(self, ref):\n        super().__init__()\n        self.ref = ref\n\n"
        "    def forward(self, x):\n        return self.ref(x) * ka_lin_helper.SCALE\n\n\n"
        "def build(reference):\n    return Scaled(reference)\n"
    )
    out = export_optimized(run, [("kernel", f"lin={candidate}", 0.0)])
    assert (out / "kernels" / "ka_lin_helper.py").is_file()
    assert read_json(out / "manifest.json")["kernels"][0]["needs"][0]["how"] == "import"
    (history / "ka_lin_helper.py").unlink()  # not reachable from the run any more
    sys.modules.pop("ka_lin_helper", None)
    spec = importlib.util.spec_from_file_location("ka_test_apply", out / "apply.py")
    apply = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(apply)
    model = nn.Sequential(nn.Linear(2, 2))
    assert apply.apply_kernels(model) == {"lin": 1}
    x = torch.ones(1, 2)
    assert torch.equal(model(x), model[0].ref(x) * 2.0)
    sys.modules.pop("ka_lin_helper", None)
    sys.path.remove(str(out / "kernels"))


def test_cli_re_exports_a_moved_run(tmp_path, capsys):
    """``python -m kernel_agent.integrate.export RUN --no-check`` on a copied run: the
    accepted items' paths name the run's old place."""
    run = sealed_run(tmp_path)
    history = run.history_dir()
    history.mkdir(parents=True)
    (history / "001_helper_0a1b2c3d.py").write_text(HELPER)
    (history / "002_main_1b1b1b1b.py").write_text(BY_NAME.format(helper="001_helper_0a1b2c3d.py"))
    old = "/elsewhere/runs/org--m/1/.truth/transforms/history/002_main_1b1b1b1b.py"
    integration = run.root / "integration.json"
    write_json(integration, {"accepted": [{"kind": "transform", "item": old}], "final": {}})
    truth.of(run).seal(integration)
    assert export.main([str(run.root), "--no-check"]) == 0
    assert "needs transforms/001_helper_0a1b2c3d.py" in capsys.readouterr().out
    assert (run.optimized_dir / "transforms" / "001_helper_0a1b2c3d.py").read_text() == HELPER
    snap = run.history_dir("attn") / "003_k_00000000.py"
    snap.parent.mkdir(parents=True)
    snap.write_text("")
    moved = export._moved(run, "kernel", "attn=/x/.truth/targets/attn/history/003_k_00000000.py")
    assert moved == f"attn={snap}"
    gone = "attn=/x/.truth/targets/attn/history/004_k_11111111.py"
    assert export._moved(run, "kernel", gone) == gone  # not in the run: unchanged


# ------------------------------------------------------------------ the integration


def test_integration_self_tests_its_export(tmp_path, simulated):  # noqa: F811
    orch = make(tmp_path / "on")
    kernel(orch.run, "attn", 2.0)
    orch.worker = worker([])
    calls: list[dict[str, Any]] = []

    def checker(run, call, args, *, expected):
        calls.append({"run": run, "args": args, "expected": expected})
        reason = "missing: not self-contained, missing x.py"
        return {"passed": False, "reason": reason, "missing": ["x.py"]}

    orch.export_checker = checker
    asyncio.run(orch.integrate())
    final = read_json(orch.run.root / "integration.json")["final"]
    assert len(calls) == 1 and calls[0]["run"] is orch.run
    assert calls[0]["expected"] == final.get("patches")
    failed = [e for e in ledger.events(orch.run) if e["event"] == "export_failed"]
    assert failed and failed[-1]["missing"] == ["x.py"]
    assert (orch.run.optimized_dir / "manifest.json").exists()  # the run goes on

    orch = make(tmp_path / "off", export_check=False)
    kernel(orch.run, "attn", 2.0)
    orch.worker = worker([])
    orch.export_checker = lambda *a, **k: pytest.fail("--no-export-check still checked")
    asyncio.run(orch.integrate())

    orch = make(tmp_path / "sim")  # a simulated run: nothing to import the package in
    kernel(orch.run, "attn", 2.0)
    orch.worker = worker([])
    asyncio.run(orch.integrate())
    assert not [e for e in ledger.events(orch.run) if e["event"].startswith("export")]
