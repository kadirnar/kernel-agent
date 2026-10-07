"""Multi-file project candidates (kernel_agent/native/project.py, issue #134): manifest,
digest, bundles, build cache and the evaluator, sweep, patcher and tools accepting them."""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import textwrap
from pathlib import Path

import pytest

from kernel_agent import dedup, ledger
from kernel_agent.agent import tools
from kernel_agent.agent.prompts import EXAMPLES_DIR
from kernel_agent.integrate.patcher import load_transform
from kernel_agent.kernels import evaluate as evaluate_mod
from kernel_agent.kernels.evaluate import evaluate, load_candidate_module
from kernel_agent.kernels.sweep import bind_config
from kernel_agent.native import engine
from kernel_agent.native import project as proj

MANIFEST = """
[project]
name = "{name}"
kind = "{kind}"

[build]
sources = ["csrc/*.cu", "csrc/*.cpp"]
include_dirs = ["include"]
cuda_cflags = ["-O3"]
"""

ENTRY = '''
"""Entry of a test project: pure torch, nothing to compile."""
import torch
from torch import nn


class Scale(nn.Module):
    def __init__(self, reference, factor=1.0):
        super().__init__()
        self.weight = reference.weight
        self.eps = reference.variance_epsilon
        self.factor = factor

    def forward(self, x):
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * h.to(x.dtype) * self.factor


def build(reference, factor=1.0):
    return Scale(reference, factor)
'''


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    """A private build cache per test."""
    root = tmp_path / "native-cache"
    monkeypatch.setenv(proj.CACHE_ENV, str(root))
    return root


def make_project(
    root: Path, name: str = "demo", *, kind: str = "kernel", entry: str = ENTRY
) -> Path:
    (root / "csrc").mkdir(parents=True)
    (root / "include").mkdir()
    (root / proj.MANIFEST).write_text(MANIFEST.format(name=name, kind=kind))
    (root / "candidate.py").write_text(entry)
    (root / "include" / "common.cuh").write_text("#pragma once\n")
    (root / "csrc" / "b.cu").write_text("// b\n")
    (root / "csrc" / "a.cu").write_text("// a\n")
    (root / "csrc" / "binding.cpp").write_text("// binding\n")
    return root


# ------------------------------------------------------------------ manifest


def test_manifest_expands_globs_in_pattern_order():
    files = ["kernel_project.toml", "candidate.py", "csrc/b.cu", "csrc/a.cu", "csrc/x.cpp"]
    text = MANIFEST.format(name="demo", kind="kernel").replace('include_dirs = ["include"]\n', "")
    m = proj.parse_manifest(text, files)
    assert m.sources == ("csrc/a.cu", "csrc/b.cu", "csrc/x.cpp")
    assert m.backend == "torch_extension" and m.load == "python" and m.entry == "candidate.py"
    command = proj.parse_manifest(
        '[build]\nbackend = "command"\ncommand = ["make"]\noutputs = ["libx.so"]\n',
        ["kernel_project.toml", "candidate.py"],
        default_name="p_7",
    )
    assert command.name == "p_7" and command.load == "torch_ops" and command.sources == ()


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ('[project]\nname = "Bad-Name"\n[build]\nsources = ["a.cu"]\n', "project.name"),
        ('[project]\nnmae = "x"\n[build]\nsources = ["a.cu"]\n', "unknown key"),
        ("[tool]\n", "unknown section"),
        ('[build]\nsources = ["*.txt"]\n', "matches no file"),
        ('[build]\nsources = ["notes.md"]\n', "must be C / C"),
        ('[build]\nsources = ["../x.cu"]\n', "relative"),
        ("[build]\nsources = []\n", "build.sources is empty"),
        ('[build]\nbackend = "command"\ncommand = ["make"]\n', "outputs"),
        ('[build]\nsources = ["a.cu"]\ntimeout_s = 99999\n', "timeout_s"),
        ('[build]\nsources = ["a.cu"]\ninclude_dirs = ["inc"]\n', "include directory"),
        ('[project]\nentry = "main.py"\n[build]\nsources = ["a.cu"]\n', "not a file"),
        ('[build]\nsources = ["a.cu"]\nload = "dlopen"\n', "build.load"),
        ("[build\n", "kernel_project.toml"),
    ],
)
def test_manifest_errors(text, error):
    files = ["kernel_project.toml", "candidate.py", "a.cu", "notes.md"]
    with pytest.raises(proj.ProjectError, match=error):
        proj.parse_manifest(text, files)


# ------------------------------------------------------------------ files + digest


def test_collect_skips_build_products_and_refuses_binaries_and_symlinks(tmp_path):
    root = make_project(tmp_path / "p")
    (root / "build").mkdir()
    (root / "build" / "x.o").write_text("object")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "c.pyc").write_bytes(b"\x00")
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref")
    (root / ".hidden").write_text("x")
    (root / "csrc" / "old.so").write_bytes(b"\x7fELF")
    files = proj.collect(root)
    assert sorted(files) == [
        "candidate.py",
        "csrc/a.cu",
        "csrc/b.cu",
        "csrc/binding.cpp",
        "include/common.cuh",
        "kernel_project.toml",
    ]
    (root / "weights.bin").write_bytes(b"\xff\xfe\x00\x81")
    with pytest.raises(proj.ProjectError, match="UTF-8"):
        proj.collect(root)
    (root / "weights.bin").unlink()
    (root / "link.cu").symlink_to(root / "csrc" / "a.cu")
    with pytest.raises(proj.ProjectError, match="symlink"):
        proj.collect(root)
    with pytest.raises(proj.ProjectError, match="not a project"):
        proj.collect(tmp_path)


def test_collect_limits(tmp_path, monkeypatch):
    root = make_project(tmp_path / "p")
    monkeypatch.setattr(proj, "MAX_FILES", 3)
    with pytest.raises(proj.ProjectError, match="more than 3 files"):
        proj.collect(root)
    monkeypatch.setattr(proj, "MAX_FILES", 200)
    monkeypatch.setattr(proj, "MAX_BYTES", 100)
    with pytest.raises(proj.ProjectError, match="bytes"):
        proj.collect(root)


def test_digest_covers_paths_and_contents():
    files = {"kernel_project.toml": "x", "a.cu": "1", "b.cu": "2"}
    same = dict(reversed(list(files.items())))
    assert proj.digest(files) == proj.digest(same)
    assert proj.digest(files) != proj.digest({**files, "a.cu": "1 "})
    assert proj.digest(files) != proj.digest({"kernel_project.toml": "x", "c.cu": "1", "b.cu": "2"})
    # a path / content boundary cannot be shifted to give the same digest
    assert proj.digest({"a": "bc"}) != proj.digest({"ab": "c"})


def test_build_key_is_per_digest_and_toolchain(tmp_path, cache):
    fp = {"torch": "2.14", "arch": "12.0", "cuda": "13.0"}
    key = proj.build_key("d" * 64, fp)
    assert key == proj.build_key("d" * 64, dict(reversed(list(fp.items()))))
    assert key != proj.build_key("e" * 64, fp)
    assert key != proj.build_key("d" * 64, {**fp, "arch": "9.0"})
    assert proj.cache_dir("d" * 64) == cache / ("d" * 24)
    assert proj.src_dir("d" * 64) == cache / ("d" * 24) / "src"


# ------------------------------------------------------------------ bundles


def test_bundle_round_trip_is_deterministic_python(tmp_path):
    root = make_project(tmp_path / "p")
    (root / "csrc" / "a.cu").write_text('// quotes " \\\\ and unicode µ\n\tint x;\n')
    text, project = proj.pack(root)
    assert text == proj.pack(root)[0]  # same files, same bytes
    ast.parse(text)  # valid Python
    bundle = tmp_path / "demo.py"
    bundle.write_text(text)
    payload = proj.read_bundle(bundle)
    assert payload is not None and payload["digest"] == project.digest
    again = proj.from_payload(payload)
    assert again.files == project.files and again.manifest == project.manifest
    assert proj.read_bundle(root / "candidate.py") is None  # not a bundle
    assert ledger.detect_backend(text) == "native"
    assert dedup.source_key(text) == dedup.source_key(proj.pack(root)[0])


def test_a_bundle_edited_by_hand_is_refused(tmp_path):
    root = make_project(tmp_path / "p")
    bundle = proj.write_bundle(root, tmp_path / "out")
    text = bundle.read_text().replace('"// a\\n"', '"// changed\\n"')
    assert text != bundle.read_text()
    bundle.write_text(text)
    payload = proj.read_bundle(bundle)
    with pytest.raises(proj.ProjectError, match="digest mismatch"):
        proj.from_payload(payload)
    with pytest.raises(proj.ProjectError, match="digest mismatch"):
        load_candidate_module(bundle)


def test_materialise_rewrites_the_cache_from_the_project(tmp_path):
    project = proj.Project.from_dir(make_project(tmp_path / "p"))
    src = proj.materialise(project)
    assert src == proj.src_dir(project.digest)
    (src / "csrc" / "a.cu").write_text("// tampered\n")  # the cache is within an agent's reach
    (src / "stray.py").write_text("import os\n")
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "candidate.cpython-312.pyc").write_bytes(b"stale")
    proj.materialise(project)
    assert (src / "csrc" / "a.cu").read_text() == "// a\n"
    assert not (src / "stray.py").exists() and not (src / "__pycache__").exists()
    assert proj.collect(src) == project.files


def test_evaluator_patcher_and_memcheck_load_projects_and_bundles(tmp_path):
    root = make_project(tmp_path / "p")
    module = load_candidate_module(root)  # what the evaluator and memcheck import
    assert callable(module.build) and module.__ka_project__["name"] == "demo"
    assert Path(module.__file__).parent == proj.src_dir(module.__ka_project__["digest"])
    bundle = proj.write_bundle(root, tmp_path / "out")
    via_bundle = load_candidate_module(bundle)
    assert via_bundle.build is module.build  # one entry module per digest and process
    transform = make_project(
        tmp_path / "t",
        "engine",
        kind="transform",
        entry="def apply(workload):\n    workload.applied = True\n",
    )
    loaded = load_transform(transform)
    assert callable(loaded.apply)
    loaded_bundle = load_transform(proj.write_bundle(transform, tmp_path / "out"))
    assert callable(loaded_bundle.apply)
    with pytest.raises(AttributeError, match="build"):
        load_candidate_module(transform)


def test_evaluator_code_dirs_cover_the_materialised_project(tmp_path):
    root = make_project(tmp_path / "p")
    bundle = proj.write_bundle(root, tmp_path / "out")
    project = proj.Project.from_dir(root)
    assert proj.code_dirs(bundle) == [proj.src_dir(project.digest)]
    assert proj.code_dirs(root) == [root.resolve(), proj.src_dir(project.digest)]
    assert proj.code_dirs(root / "candidate.py") == []
    dirs = evaluate_mod._code_dirs(bundle)
    assert dirs == (bundle.parent, proj.src_dir(project.digest))


def test_project_candidate_passes_the_cpu_evaluation(tmp_path):
    from test_evaluator_exploits import make_capture, restored_globals

    capture = make_capture("rmsnorm", tmp_path / "rms.pt", "cpu")
    root = make_project(tmp_path / "p")
    bundle = proj.write_bundle(root, tmp_path / "out")
    with restored_globals():
        for candidate in (root, bundle):
            result = evaluate(capture, candidate, device="cpu")
            assert result["status"] == "ok" and result["correct"], result


def test_a_thread_left_running_by_a_project_is_an_integrity_violation(tmp_path):
    from test_evaluator_exploits import make_capture, restored_globals

    entry = ENTRY + textwrap.dedent(
        """
        import threading
        import time

        _stop = threading.Event()


        def _spin():
            while not _stop.wait(0.01):
                pass


        _build = build


        def build(reference):
            threading.Thread(target=_spin, daemon=True).start()
            return _build(reference)
        """
    )
    capture = make_capture("rmsnorm", tmp_path / "rms.pt", "cpu")
    bundle = proj.write_bundle(make_project(tmp_path / "p", "spinner", entry=entry), tmp_path)
    try:
        with restored_globals():
            result = evaluate(capture, bundle, device="cpu")
        assert result["status"] == "integrity_violation", result
        assert "runs candidate code" in result["error"]
    finally:
        entry = next(m for m in sys.modules if m.startswith("ka_native_entry_spinner_"))
        sys.modules[entry]._stop.set()


def test_sweep_binds_configs_into_a_bundle(tmp_path):
    root = make_project(tmp_path / "p", "swept")
    text, _ = proj.pack(root)
    bound = tmp_path / "bound.py"
    bound.write_text(bind_config(text, {"factor": 2.0}))
    module = load_candidate_module(bound)

    class Ref:
        weight = 1.0
        variance_epsilon = 1e-6

    built = module.build(Ref())
    assert built.factor == 2.0
    assert module.build(Ref(), factor=3.0).factor == 3.0


# ------------------------------------------------------------------ build + cache


def _command_project(root: Path, name: str = "cmd") -> Path:
    root.mkdir(parents=True)
    (root / proj.MANIFEST).write_text(
        textwrap.dedent(
            f"""
            [project]
            name = "{name}"

            [build]
            backend = "command"
            command = ["{{python}}", "{{src}}/build.py", "{{build}}"]
            outputs = ["out/lib.txt"]
            load = "none"
            """
        )
    )
    (root / "candidate.py").write_text("def build(reference):\n    return reference\n")
    (root / "build.py").write_text(
        "import os, pathlib, sys\n"
        "out = pathlib.Path(sys.argv[1]) / 'out'\n"
        "out.mkdir()\n"
        "(out / 'lib.txt').write_text(os.environ['KA_PROJECT_DIGEST'])\n"
        "n = pathlib.Path(os.environ['KA_SRC_DIR']) / '..' / 'builds'\n"
        "n.write_text(str(int(n.read_text()) + 1) if n.exists() else '1')\n"
    )
    return root


def test_command_build_is_cached_by_digest_and_rebuilt_when_damaged(tmp_path, monkeypatch):
    monkeypatch.setattr(proj, "fingerprint", lambda: {"torch": "t", "arch": "a"})
    project = proj.Project.from_dir(_command_project(tmp_path / "c"))
    builds = proj.cache_dir(project.digest) / "builds"
    first = proj.build(project, setup=False)
    assert not first.cached and first.libraries[0].read_text() == project.digest
    assert builds.read_text() == "1"
    stamp = json.loads((first.build_dir / proj.STAMP).read_text())
    assert stamp["key"] == first.key and list(stamp["outputs"]) == ["out/lib.txt"]
    proj._BUILT.clear()  # a new process
    again = proj.build(project, setup=False)
    assert again.cached and builds.read_text() == "1" and again.build_dir == first.build_dir
    assert proj.build(project, setup=False) is again  # one load per process
    proj._BUILT.clear()
    first.libraries[0].write_text("damaged")  # the stamp no longer matches: rebuilt
    assert proj.stamp_ok(first.build_dir, first.key) is None
    rebuilt = proj.build(project, setup=False)
    assert not rebuilt.cached and builds.read_text() == "2"
    proj._BUILT.clear()
    monkeypatch.setattr(proj, "fingerprint", lambda: {"torch": "t", "arch": "b"})
    other = proj.build(project, setup=False)  # another toolchain: its own build
    assert not other.cached and other.build_dir != first.build_dir
    proj._BUILT.clear()
    monkeypatch.setenv(proj.REBUILD_ENV, "1")
    assert not proj.build(project, setup=False).cached
    proj._BUILT.clear()


def test_failed_command_build_reports_the_output(tmp_path, monkeypatch):
    monkeypatch.setattr(proj, "fingerprint", lambda: {"torch": "t"})
    root = _command_project(tmp_path / "c", "broken")
    (root / "build.py").write_text("import sys\nprint('nvcc: error in gemv.cu')\nsys.exit(3)\n")
    with pytest.raises(proj.ProjectError, match="exit 3"):
        proj.build(proj.Project.from_dir(root), setup=False)


def test_prebuild_runs_outside_and_skips_without_an_arch(tmp_path, monkeypatch):
    root = _command_project(tmp_path / "c", "pre")
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    assert proj.prebuild(root)["status"] == "skipped"
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "12.0")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(p for p in sys.path if p))
    result = proj.prebuild(root)
    assert result["status"] == "ok", result
    assert Path(result["outputs"][0]).read_text() == proj.Project.from_dir(root).digest
    assert proj.prebuild(root)["cached"]
    (root / "build.py").write_text("raise SystemExit('ptxas fatal')\n")
    failed = proj.prebuild(root)
    assert failed["status"] == "build_error" and "ptxas fatal" in failed["error"]
    broken = tmp_path / "nomanifest"
    broken.mkdir()
    assert proj.prebuild(broken)["status"] == "build_error"


def test_cli_check_and_pack(tmp_path, capsys):
    root = make_project(tmp_path / "p", "cli_demo")
    assert proj.main(["check", str(root)]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["name"] == "cli_demo" and info["files"] == 6
    out = tmp_path / "b.py"
    assert proj.main(["pack", str(root), "-o", str(out)]) == 0
    assert proj.read_bundle(out)["digest"] == info["digest"]
    assert proj.main(["check", str(tmp_path)]) == 2


# ------------------------------------------------------------------ tools


def test_snapshot_of_a_project_is_its_bundle(tmp_path):
    from synthetic_run import make_run

    run = make_run(tmp_path / "run", integrate=False)
    root = make_project(run.transforms_dir / "native" / "engine", "engine", kind="transform")
    snap = tools.snapshot(run, root)
    assert re.fullmatch(r"\d{3}_engine_[0-9a-f]{8}\.py", snap.name), snap.name
    assert proj.read_bundle(snap)["digest"] == proj.Project.from_dir(root).digest
    assert engine.e2e_backend([snap], []) == "native"
    assert engine.e2e_backend([snap], ["attn=x.py"]) == "native+kernels"
    assert engine.e2e_backend([], ["native_solver=x.py"]) == "native"
    plain = run.transforms_dir / "graph.py"
    plain.write_text("def apply(workload):\n    pass\n")
    assert engine.e2e_backend([tools.snapshot(run, plain)], []) == "transform"


def test_template_project_is_valid():
    project = proj.Project.from_dir(EXAMPLES_DIR / "native_project")
    m = project.manifest
    assert m.name == "native_rmsnorm" and m.kind == "kernel"
    assert m.sources == ("csrc/binding.cpp", "csrc/rmsnorm.cu")
    assert "include/ka_native.cuh" in project.files


@pytest.mark.gpu
def test_template_project_passes_the_evaluator_on_gpu(tmp_path):
    """Not run in the CPU suite: compiles the template with nvcc and evaluates it."""
    from kernel_agent.kernels.evaluate import run_evaluation
    from kernel_agent.selftest import make_rmsnorm_capture

    capture = make_rmsnorm_capture(tmp_path / "rms.pt")
    result = run_evaluation(capture, EXAMPLES_DIR / "native_project")
    assert result["correct"], result
