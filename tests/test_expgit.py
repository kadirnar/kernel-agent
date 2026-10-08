"""The experiments' git history (issue #224): ``<run>/experiments.git``, one commit per
experiment, ``best/<lineage>`` branches, ``kernel-agent exp git``. CPU only and
deterministic: synthetic and simulated runs; threads and a child process meet on a barrier
or the child's first line of output, never on a sleep. Skipped where git is missing (except
the test of exactly that)."""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import os
import random
import shutil
import subprocess
import sys
import threading
import types
from pathlib import Path
from typing import Any

import pytest
from synthetic_run import make_run

from kernel_agent import charts, dashboard, dryrun, expgit, ledger, orchestrator, truth
from kernel_agent.agent import tools as tools_mod
from kernel_agent.budget import Standing
from kernel_agent.cli import main
from kernel_agent.config import OptimizeConfig
from kernel_agent.improve import ImproveConfig, Improver
from kernel_agent.native import project as native_project
from kernel_agent.workspace import RunDir, write_json

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
OK = {"status": "ok", "correct": True, "cases": []}


def git(repo: Path, *args: str) -> str:
    """git on ``repo`` as expgit runs it (none of the user's configuration)."""
    exe = shutil.which("git")
    assert exe is not None
    return expgit.Git(exe, repo).run(*args)


def refs(repo: Path) -> dict[str, str]:
    """Every ref of ``repo`` → its commit."""
    listing = git(repo, "for-each-ref", "--format=%(refname) %(objectname)")
    return dict(line.split() for line in listing.splitlines())


def tags(repo: Path) -> list[int]:
    return sorted(int(r.rsplit("/", 1)[1]) for r in refs(repo) if r.startswith(expgit.TAGS))


def trailer(repo: Path, rev: str, key: str) -> str:
    return git(repo, "log", "-1", f"--format=%(trailers:key={key},valueonly)", rev, "--").strip()


def objects(repo: Path) -> list[str]:
    """The repository's object files (all loose: nothing here packs them)."""
    return sorted(str(p.relative_to(repo)) for p in (repo / "objects").rglob("*") if p.is_file())


def last_standing(rows: list[dict[str, Any]], lineage: str) -> int:
    """The ``exp`` of the standing best of a target, as the ledger keeps it."""
    stand = Standing()
    for r in rows:
        if r["target"] != lineage:
            continue
        if r["status"] == ledger.REEVALUATED:
            stand.replace(r)
        elif r["status"] == ledger.KEEP:
            stand.keep(r)
    assert stand.top is not None
    return int(stand.top["exp"])


def kernel(run: RunDir, target: str, name: str, speedup: float, **kw: Any) -> dict[str, Any]:
    """A kernel row whose snapshot ``history/<name>`` exists (its code names the speedup)."""
    history = run.history_dir(target)
    history.mkdir(parents=True, exist_ok=True)
    (history / name).write_text(f"def build(reference):  # {name}\n    return reference\n")
    if not (run.target(target) / "reference_source.py").exists():
        (run.target(target) / "reference_source.py").write_text(f"class {target.title()}: ...\n")
    result = OK | {"speedup": speedup} if speedup else {"status": "incorrect", "correct": False}
    return ledger.record_kernel(run, target, result, snapshot=name, hypothesis=name, **kw)


@pytest.fixture(scope="module")
def synced(tmp_path_factory):
    """The synthetic run, a quick check and a duplicate on top, synced."""
    run = make_run(tmp_path_factory.mktemp("runs"))
    attn = [r for r in ledger.rows(run) if r["target"] == "attn"]
    ledger.record_kernel(
        run, "attn", OK, snapshot="099_quick_0a1b2c3d.py", hypothesis="q", status=ledger.QUICK_OK
    )
    ledger.record_kernel(
        run, "attn", OK, snapshot=attn[0]["snapshot"], hypothesis="d", status=ledger.DUPLICATE
    )
    done = expgit.sync(run)
    assert done is not None and done.repo == expgit.repo_path(run)
    return run


# ------------------------------------------------------------------ the synthetic run


@needs_git
def test_one_commit_per_measured_experiment(synced):
    run, repo = synced, expgit.repo_path(synced)
    rows = ledger.rows(run)
    measured = [r["exp"] for r in ledger.measured(rows)]
    unmeasured = [r["exp"] for r in rows if r["status"] in (ledger.QUICK_OK, ledger.DUPLICATE)]
    assert len(unmeasured) == 2
    assert tags(repo) == [0, *measured]  # none for the quick check and the duplicate
    assert git(repo, "symbolic-ref", "HEAD").strip() == "refs/heads/best/model"
    assert git(repo, "log", "-1", "--format=%P", "exp/0").strip() == ""  # the root
    for r in rows:
        if r["exp"] in measured and r["target"] != ledger.E2E:  # the snapshot's bytes
            snap = run.history_dir(r["target"]) / r["snapshot"]
            shown = git(repo, "show", f"exp/{r['exp']}:targets/{r['target']}/candidate.py")
            assert shown == snap.read_text()
            sha = hashlib.sha256(snap.read_bytes()).hexdigest()
            assert trailer(repo, f"exp/{r['exp']}", "Snapshot-Sha256") == sha
    for target in run.target_ids():
        mine = [r for r in rows if r["target"] == target]
        kept = [r for r in mine if r["status"] == ledger.KEEP]
        # the reference and the baseline below the kept line
        count = git(repo, "rev-list", "--first-parent", "--count", f"best/{target}")
        assert int(count) == len(kept) + 2
        assert trailer(repo, f"best/{target}", "Exp") == str(last_standing(rows, target))
        for a, b in itertools.pairwise(kept):  # consecutive kept versions differ
            assert git(repo, "diff", f"exp/{a['exp']}", f"exp/{b['exp']}").startswith("diff --git")
        first = git(repo, "log", "--format=%s", f"best/{target}").splitlines()[-2:]
        assert first[0] == f"reference [{target}] 1.00×: the module as captured"
        assert first[1].startswith("exp 0 [model] baseline 1,532.4 ms: Qwen/Qwen3-0.6B")
    assert trailer(repo, "best/model", "Exp") == str(last_standing(rows, ledger.E2E))
    keep = next(r for r in rows if r["status"] == ledger.KEEP and r["target"] == "attn")
    message = git(repo, "log", "-1", "--format=%an|%ad|%s%n%b", "--date=raw", f"exp/{keep['exp']}")
    head, body = message.split("\n", 1)
    assert head.startswith(f"kernel-agent|{expgit._date(keep['time'])}|exp {keep['exp']} [attn]")
    assert f" keep {keep['speedup']:.2f}× (+" in head and head.endswith(f": {ledger.title(keep)}")
    assert keep["hypothesis"] in body and "Kind: kernel\nStatus: keep\n" in body


@needs_git
def test_a_second_sync_writes_nothing_and_deleted_tags_come_back(synced):
    repo = expgit.repo_path(synced)
    before, files = refs(repo), objects(repo)
    done = expgit.sync(synced)
    assert done is not None and done.added == 0
    assert refs(repo) == before and objects(repo) == files  # no new object, no ref moved
    k = max(tags(repo)) // 2
    above = [n for n in tags(repo) if n > k]
    for n in above:
        git(repo, "update-ref", "-d", f"refs/tags/exp/{n}")
    assert max(tags(repo)) == k
    done = expgit.sync(synced)
    assert done is not None and done.added == len(above)
    assert refs(repo) == before and objects(repo) == files  # the same hashes again


@needs_git
def test_rebuilds_are_bit_identical(synced, tmp_path):
    repo = expgit.repo_path(synced)
    before = refs(repo)
    out = tmp_path / "out.git"
    done = expgit.rebuild(synced, out)
    assert done is not None and refs(out) == before
    assert not (synced.root / "out.git").exists()
    with pytest.raises(FileExistsError):
        expgit.rebuild(synced, out)  # not an empty directory
    done = expgit.rebuild(synced)  # in place, swapped in under the lock
    assert done is not None and refs(repo) == before
    assert not [p for p in synced.root.iterdir() if p.name.startswith(f".{expgit.REPO}")]


# ------------------------------------------------------------------ re-evaluations, files


@needs_git
def test_a_reevaluation_that_demotes_the_best_moves_the_branch_back(tmp_path):
    run = RunDir.create(tmp_path, "org/m")
    kernel(run, "t", "001_a_00000001.py", 1.5)
    kernel(run, "t", "002_b_00000002.py", 2.0)
    repo = expgit.repo_path(run)
    assert expgit.sync(run) is not None
    assert refs(repo)["refs/heads/best/t"] == refs(repo)["refs/tags/exp/2"]
    # the integration's re-evaluation of exp 2: slower now, so exp 1 is the best again
    kernel(run, "t", "002_b_00000002.py", 1.2, status=ledger.REEVALUATED)
    assert expgit.sync(run) is not None
    after = refs(repo)
    assert after["refs/heads/best/t"] == after["refs/tags/exp/1"]
    assert git(repo, "rev-parse", "exp/3^").strip() == after["refs/tags/exp/2"]
    assert git(repo, "diff", "exp/2", "exp/3") == ""  # the same tree
    subject = git(repo, "log", "-1", "--format=%s", "exp/3")
    assert subject.startswith("exp 3 [t] re-evaluated exp 2 1.20×: ")
    # it is confirmed again: the branch follows the standing re-evaluation
    kernel(run, "t", "002_b_00000002.py", 2.1, status=ledger.REEVALUATED)
    assert expgit.sync(run) is not None
    assert refs(repo)["refs/heads/best/t"] == refs(repo)["refs/tags/exp/4"]
    # a failed re-evaluation of the only kept version: back to the reference
    kernel(run, "u", "001_c_00000003.py", 1.4)
    kernel(run, "u", "001_c_00000003.py", 0.0, status=ledger.REEVALUATED)
    assert expgit.sync(run) is not None
    assert refs(repo)["refs/heads/best/u"] == refs(repo)["refs/tags/reference/u"]


@needs_git
def test_an_e2e_row_waits_for_its_files(tmp_path, monkeypatch):
    """An end-to-end row is committed once its event lists its files; one whose event never
    came is committed after ``SETTLE_S`` without them, and its tree is its parent's."""
    run = RunDir.create(tmp_path, "org/m")
    kernel(run, "t", "001_a_00000001.py", 1.5)
    row = ledger.append(
        run,
        {
            "time": ledger.stamp(),
            "target": ledger.E2E,
            "kind": ledger.E2E,
            "status": ledger.KEEP,
            "correct": True,
            "speedup": 1.3,
            "new_ms": 10.0,
            "hypothesis": "graph",
        },
    )  # its event: not yet
    kernel(run, "t", "002_b_00000002.py", 1.7)
    repo = expgit.repo_path(run)
    done = expgit.sync(run)
    assert done is not None and done.last == 1  # exp 2 waits, and exp 3 behind it
    now = ledger.epoch(row["time"])
    assert now is not None
    monkeypatch.setattr(ledger, "clock", lambda: now + expgit.SETTLE_S)
    done = expgit.sync(run)
    assert done is not None and done.last == 3
    assert "files not recorded" in git(repo, "log", "-1", "--format=%b", "exp/2")
    assert git(repo, "rev-parse", "exp/2^{tree}") == git(repo, "rev-parse", "exp/0^{tree}")


@needs_git
def test_truth_is_untouched_and_kernel_lines_merge_into_the_model(tmp_path, monkeypatch):
    monkeypatch.setenv(native_project.CACHE_ENV, str(tmp_path / "native-cache"))
    run = RunDir.create(tmp_path, "org/m")
    write_json(run.run_json, {"card": {"repo_id": "org/m"}, "truth": truth.new_section()})
    write_json(run.baseline_json, {"median_ms": 100.0, "workload": "decode 8 tokens"})
    write_json(run.target("t") / "spec.json", {"id": "t", "module_class": "M"})
    (run.target("t") / "reference_source.py").write_text("class M: ...\n")
    (run.target("t") / "candidates").mkdir()
    snaps = []
    for i, speedup in enumerate((1.4, 1.6)):
        src = run.target("t") / "candidates" / f"v{i}.py"
        src.write_text(f"def build(reference):  # v{i}\n    return reference\n")
        snaps.append(tools_mod.snapshot(run, src, "t"))
        tools_mod.record_candidate(run, "t", src, snaps[-1], OK | {"speedup": speedup},
                                   hypothesis=f"v{i}", eval_s=1.0)  # fmt: skip
    project = tmp_path / "proj"
    (project / "csrc").mkdir(parents=True)
    (project / native_project.MANIFEST).write_text(
        '[project]\nname = "stack"\nkind = "transform"\n\n[build]\nsources = ["csrc/*.cu"]\n'
    )
    (project / "candidate.py").write_text("def apply(workload):\n    pass\n")
    (project / "csrc" / "a.cu").write_text("// a\n")
    run.transforms_dir.mkdir(parents=True, exist_ok=True)
    bundle = tools_mod.snapshot(run, project, None)
    result = {"status": "ok", "passed": True, "speedup": 1.25, "median_ms": 80.0,
              "baseline_ms": 100.0, "times_ms": [80.0, 80.0]}  # fmt: skip
    tools_mod.record_e2e_result(run, result, [bundle], [f"t={snaps[0]}"], hypothesis="stack")

    def digests() -> dict[str, tuple[str, int, int]]:
        return {
            str(p.relative_to(run.root)): (
                hashlib.sha256(p.read_bytes()).hexdigest(),
                p.stat().st_mtime_ns,
                p.stat().st_mode,
            )
            for p in sorted(run.truth_dir.rglob("*"))
            if p.is_file()
        }

    before = digests()
    assert any("history" in name for name in before)
    repo = expgit.repo_path(run)
    assert expgit.sync(run) is not None
    assert digests() == before  # read, never written
    assert git(repo, "show", "exp/1:targets/t/candidate.py") == snaps[0].read_text()
    tree = git(repo, "ls-tree", "-r", "--name-only", "exp/3").split()
    assert tree == [
        "native/stack/candidate.py",
        "native/stack/csrc/a.cu",
        f"native/stack/{native_project.MANIFEST}",
        "targets/t/candidate.py",
    ]
    parents = git(repo, "log", "-1", "--format=%P", "exp/3").split()
    assert parents == [refs(repo)["refs/tags/exp/0"], refs(repo)["refs/tags/exp/1"]]
    graph = git(repo, "log", "--oneline", "--graph", "best/model")
    assert "|\\" in graph and "exp 1 [t] keep" in graph  # the kernel line merges in


# ------------------------------------------------------------------ determinism


@needs_git
def test_seeded_dry_runs_give_identical_hashes(tmp_path, monkeypatch):
    """Two virtual-time dry runs with ``--agents 3`` (their phase ends sync as they go)."""
    monkeypatch.setattr(orchestrator.toolchain, "setup", dryrun.SimToolchain)
    monkeypatch.setattr(charts, "available", lambda: False)
    monkeypatch.setattr(dryrun, "time", types.SimpleNamespace(time=lambda: 1_790_000_000.0))
    found = []
    for k in range(2):
        config = OptimizeConfig(
            model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path / str(k), dossier=False
        )
        orch = orchestrator.Orchestrator(dryrun.create_run(config), config)
        world = dryrun.World(orch, virtual=True)
        improver = Improver(
            orch,
            ImproveConfig(agents=3, rounds=1, max_slices=6),
            require_capture=False,
            live_charts=False,
        )

        async def go(improver: Improver = improver, world: dryrun.World = world) -> None:
            with world.driving():
                await improver.improve()

        with world.installed():
            asyncio.run(go())
        assert expgit.sync(orch.run) is not None
        found.append(refs(expgit.repo_path(orch.run)))
    assert found[0] == found[1] and len(found[0]) > 20
    assert "refs/heads/best/model" in found[0]


# ------------------------------------------------------------------ concurrency


THREADS, CALLS = 4, 8


@needs_git
def test_recording_threads_and_a_syncing_thread(tmp_path):
    """4 threads record (kernel rows, end-to-end rows whose files follow them in their
    event) while a fifth syncs in a loop: the tags are 1..N, every branch is its target's
    standing best, and the repository equals one rebuilt from the finished ledger."""
    run = RunDir.create(tmp_path, "org/m")
    for target in ("t0", "t1"):  # before any sync can read them
        run.target(target).mkdir(parents=True)
        (run.target(target) / "reference_source.py").write_text(f"class {target}: ...\n")
    start = threading.Barrier(THREADS + 1)
    recording = threading.Event()
    errors: list[BaseException] = []

    def record(k: int) -> None:
        rng = random.Random(k)
        start.wait(timeout=60)
        try:
            for i in range(CALLS):
                target = f"t{k % 2}"
                name = f"{k}{i:02d}_v_{k:04d}{i:04d}.py"
                kernel(run, target, name, round(rng.uniform(0.8, 4.0), 4), session=f"s{k}")
                if k == 0 and i % 2:
                    path = run.history_dir(target) / name
                    ledger.record_e2e(
                        run,
                        {"status": "ok", "passed": True, "speedup": rng.uniform(1.0, 2.0)},
                        backend="kernels",
                        snapshot=target,
                        hypothesis=f"e2e {i}",
                        files=ledger.item_files(run, [f"{target}={path}"]),
                    )
        except BaseException as exc:  # reported below
            errors.append(exc)

    def syncing() -> None:
        start.wait(timeout=60)
        try:
            while not recording.is_set():
                assert expgit.sync(run) is not None, expgit.last_error(expgit.repo_path(run))
        except BaseException as exc:
            errors.append(exc)

    recorders = [threading.Thread(target=record, args=(k,)) for k in range(THREADS)]
    syncer = threading.Thread(target=syncing)
    for t in [*recorders, syncer]:
        t.start()
    for t in recorders:
        t.join()
    recording.set()
    syncer.join()
    assert not errors
    repo = expgit.repo_path(run)
    assert expgit.sync(run) is not None
    rows = ledger.rows(run)
    assert tags(repo) == list(range(len(rows) + 1)) and len(rows) == THREADS * CALLS + CALLS // 2
    for target in ("t0", "t1"):
        assert trailer(repo, f"best/{target}", "Exp") == str(last_standing(rows, target))
    rebuilt = tmp_path / "rebuilt.git"
    assert expgit.rebuild(run, rebuilt) is not None
    assert refs(rebuilt) == refs(repo)
    git(repo, "fsck", "--no-dangling")


_CHILD = """
import sys
from pathlib import Path
from kernel_agent import expgit
from kernel_agent.workspace import RunDir
run = RunDir(Path(sys.argv[1]))
print("ready", flush=True)  # the parent syncs from now on, as this process does
sys.exit(0 if expgit.sync(run) is not None else 1)
"""


@needs_git
def test_two_processes_sync_at_once(tmp_path):
    """The ``flock``: whichever process gets it first writes, the other finds the commits
    there; one consistent repository either way."""
    run = make_run(tmp_path / "runs")
    src = str(Path(expgit.__file__).parents[1])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([src, os.environ.get("PYTHONPATH", "")])}
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(run.root)], stdout=subprocess.PIPE, text=True, env=env
    )
    try:
        assert child.stdout is not None and child.stdout.readline() == "ready\n"
        assert expgit.sync(run) is not None
        assert child.wait(timeout=120) == 0
    finally:
        child.kill()
    repo = expgit.repo_path(run)
    git(repo, "fsck", "--no-dangling")
    rebuilt = tmp_path / "rebuilt.git"
    assert expgit.rebuild(run, rebuilt) is not None
    assert refs(rebuilt) == refs(repo)
    assert tags(repo) == [0, *(r["exp"] for r in ledger.measured(ledger.rows(run)))]


# ------------------------------------------------------------------ no git, the dashboard


def test_without_git_recording_goes_on(tmp_path, monkeypatch):
    run = RunDir.create(tmp_path, "org/m")
    write_json(run.run_json, {"card": {"repo_id": "org/m"}, "phases": {}})
    monkeypatch.setenv("PATH", str(tmp_path / "no-bin"))
    monkeypatch.setattr(charts, "available", lambda: False)
    kernel(run, "t", "001_a_00000001.py", 1.5)
    assert expgit.sync(run) is None
    dashboard._refresh(run, None)  # the refresher thread's rewrite: no exception either
    kernel(run, "t", "002_b_00000002.py", 1.7)
    assert expgit.sync(run) is None
    assert [r["status"] for r in ledger.rows(run)] == ["keep", "keep"]
    failed = [e for e in ledger.events(run) if e["event"] == expgit.EVENT]
    assert len(failed) == 1 and "git is not on PATH" in failed[0]["error"]  # once
    assert expgit.last_error(expgit.repo_path(run)) == "git is not on PATH"
    assert not expgit.repo_path(run).exists()


@needs_git
def test_the_dashboard_refresh_syncs(tmp_path, monkeypatch):
    run = RunDir.create(tmp_path, "org/m")
    write_json(run.run_json, {"card": {"repo_id": "org/m"}, "phases": {}})
    monkeypatch.setattr(charts, "available", lambda: False)
    dashboard._refresh(run, None)
    assert not expgit.repo_path(run).exists()  # no row yet: no repository
    kernel(run, "t", "001_a_00000001.py", 1.5)
    dashboard._refresh(run, ["t"])
    assert tags(expgit.repo_path(run)) == [0, 1]
    assert run.dashboard.exists()


# ------------------------------------------------------------------ the CLI


@needs_git
def test_exp_git_cli(tmp_path, capfd):
    run = make_run(tmp_path / "runs")
    root = str(run.root)
    assert main(["exp", "git", root]) == 0
    assert f"kernel-agent exp git {run.root} --sync" in capfd.readouterr().out
    attn = [r for r in ledger.rows(run) if r["target"] == "attn" and r["status"] == "keep"]
    a, b = attn[-1]["exp"], attn[0]["exp"]
    assert main(["exp", "diff", root, str(a), str(b)]) == 0
    assert capfd.readouterr().out.startswith(f"--- exp {b}: ")  # difflib: no repository yet

    assert main(["exp", "git", root, "--sync"]) == 0
    out = capfd.readouterr().out
    rows = ledger.rows(run)
    assert out.startswith(f"{expgit.repo_path(run)}: {len(rows) + 1 + 4} new commits, up to")
    assert "best/model (HEAD)" in out and "best/attn" in out
    line = next(x for x in out.splitlines() if x.startswith("best/attn "))
    assert line.split()[2:4] == [str(attn[-1]["exp"]), str(len(attn))]  # tip exp, kept

    assert main(["exp", "diff", root, str(a), str(b)]) == 0
    assert capfd.readouterr().out.startswith("diff --git a/targets/attn/candidate.py")

    assert main(["exp", "git", root, "--", "log", "--oneline", "best/attn"]) == 0
    log = capfd.readouterr().out.splitlines()
    assert len(log) == len(attn) + 2 and log[-1].endswith("baseline 1,532.4 ms: Qwen/Qwen3-0.6B")

    out_dir = tmp_path / "elsewhere.git"
    out_dir.mkdir()
    assert main(["exp", "git", root, "--rebuild", "--out", str(out_dir)]) == 0
    assert refs(out_dir) == refs(expgit.repo_path(run))
    capfd.readouterr()
    assert main(["exp", "git", root, "--rebuild", "--out", str(out_dir)]) == 1
    assert "not an empty directory" in capfd.readouterr().err
