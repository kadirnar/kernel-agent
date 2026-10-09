"""A run's experiments as a git history (issue #224): ``<run>/experiments.git``.

In autoresearch and AutoKernel the research *is* a git branch: ``git log`` is the kept line
and ``git diff`` shows each change. This module derives such a repository from the run's
ledger (:mod:`kernel_agent.experiments`), one commit per experiment, in ``exp`` order.

* **Bare, plumbing only.** It is written with ``hash-object -w``, ``mktree``, ``commit-tree``
  and ``update-ref`` alone: no working tree, index or checkout, so it never touches
  ``.truth/`` and is safe while sessions run. It is derived data. The evaluator, the
  integration and the winner selection never read it (they keep using the ``.truth/``
  records and their sha256), so deleting it loses nothing: :func:`rebuild` recreates it
  bit for bit.
* **Trees.** A commit's tree is what its experiment measured, at stable paths so that
  consecutive versions diff: a kernel row's snapshot is ``targets/<id>/candidate.py`` (a
  region target's ``rewrite.py`` sits beside it), a transform is ``transforms/<stem>.py``, and
  each kernel item of an ``e2e``, integration or probe row is its target's
  ``candidate.py``. A native project's bundle is unpacked into ``native/<project>/…``.
  Identical contents are one blob. A row whose files were not recorded keeps its parent's
  tree, and its body says so.
* **Parents.** The first parent is the commit of the row's ``parent`` snapshot, else the
  lineage's standing best when the row was recorded (``Experiment.parent_exp``): "start
  from the current branch". A target's first experiment starts from the target's
  ``reference`` commit (``reference_source.py``), a child of ``exp 0``, the baseline. An
  end-to-end commit also gets the commits of its kernel items as parents, so ``git log
  --graph best/model`` shows the kernel lines merging into the model's. A ``re-evaluated``
  row is a child of the experiment it re-evaluates, with the same tree.
* **Messages.** The subject is ``exp 42 [dit_layer_fp8] keep 12.11× (+1.1 %): <title>``. The
  body holds the hypothesis and the files measured, then the trailers ``Exp``, ``Lineage``,
  ``Kind``, ``Status``, ``Value``, ``Unit``, ``Snapshot``, ``Snapshot-Sha256``, ``Idea``,
  ``Session``, ``Worker`` and ``Review``. The author is ``kernel-agent (<session>)``. The
  author and committer dates are the row's ``time``: the ledger's wall clock, written as
  UTC because the ledger keeps no zone. A rebuild therefore gives the same hashes on any
  machine.
* **Refs.** ``refs/tags/exp/<N>`` marks every experiment, so discarded and failed code
  stays reachable as evidence. ``refs/tags/reference/<id>`` marks the roots.
  ``refs/heads/best/<lineage>`` (``model`` and every kernel target) is the lineage's
  standing best: it advances on ``keep`` and moves back when a re-evaluation demotes the
  best (``ledger.standing``). ``HEAD`` is ``best/model``.

:func:`sync` is an idempotent catch-up. It commits every row from the first one without its
tag, strictly in ``exp`` order, and updates the refs in one compare-and-swap transaction.
An ``flock`` of ``experiments.git/ka-sync.lock`` serialises the run's writer with a CLI
``--sync`` from another process; readers (``git log``, ``exp diff``) need no lock. The run's
``dashboard.Refresher`` thread calls it before the charts (``dashboard._refresh``, also at
phase ends and in ``kernel-agent report``), so no evaluation waits for git. A missing or
failing ``git`` becomes an ``exp_git`` event, and the run goes on: the feature is optional,
like the charts.

``kernel-agent exp git RUN`` (:func:`main`) prints the repository and its lineages; it also
takes ``--sync``, ``--rebuild [--out DIR]`` and ``-- <git arguments>``.
"""

from __future__ import annotations

import argparse
import calendar
import contextlib
import fcntl
import hashlib
import os
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernel_agent import experiments, ledger, objective
from kernel_agent.budget import Standing
from kernel_agent.experiments import MODEL, MS, Experiment, X
from kernel_agent.native import project as native_project
from kernel_agent.workspace import RunDir, read_json

REPO = "experiments.git"
LOCK = "ka-sync.lock"
EVENT = "exp_git"  # events.jsonl: git is missing or failed
EMAIL = "kernel-agent@localhost"
TAGS, REFERENCES, BEST = "refs/tags/exp/", "refs/tags/reference/", "refs/heads/best/"
ZERO = "0" * 40  # update-ref's "this ref does not exist"
#: An end-to-end row is committed once its ``evaluation`` event (its files) is in
#: ``events.jsonl``. The event follows the row within microseconds, so a row still without one
#: this many seconds after it was recorded never gets one (its recorder died in between).
SETTLE_S = 60.0


class GitError(RuntimeError):
    """A git command failed."""


@dataclass(frozen=True)
class Synced:
    """What :func:`sync` did: the ``repo``, the commits it ``added``, the highest ``exp``
    with a commit (None: none yet)."""

    repo: Path
    added: int
    last: int | None


def repo_path(run: RunDir) -> Path:
    return run.root / REPO


def _env() -> dict[str, str]:
    """git's environment: none of the caller's ``GIT_*`` variables (a hook's ``GIT_DIR``) and
    neither the user's nor the system's configuration, where commit signing or another hash
    function would change the hashes."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return env | {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "LC_ALL": "C",
    }


class Git:
    """The plumbing of one bare repository. Within an instance, contents already written are
    not handed to git again (a target's tree recurs in every row that does not change it)."""

    def __init__(self, exe: str, repo: Path) -> None:
        self.exe, self.repo, self.env = exe, repo, _env()
        self._blobs: dict[bytes, str] = {}  # sha256 of the content → blob
        self._trees: dict[tuple[str, ...], str] = {}  # sorted entries → tree

    def run(self, *args: str, data: bytes | None = None, env: dict[str, str] | None = None) -> str:
        proc = subprocess.run(
            [self.exe, f"--git-dir={self.repo}", *args],
            input=data,
            capture_output=True,
            env=self.env | (env or {}),
            check=False,
        )
        if proc.returncode:
            error = proc.stderr.decode(errors="replace").strip()
            raise GitError(f"git {args[0]} exited with {proc.returncode}: {error[-500:]}")
        return proc.stdout.decode(errors="replace")

    def init(self) -> None:
        """Create the repository (bare, no hooks or other templates), ``HEAD`` → best/model."""
        if (self.repo / "HEAD").is_file():
            return
        proc = subprocess.run(
            [self.exe, "init", "--bare", "--quiet", "--template=", str(self.repo)],
            capture_output=True,
            env=self.env,
            check=False,
        )
        if proc.returncode:
            raise GitError(f"git init: {proc.stderr.decode(errors='replace').strip()[-500:]}")
        self.run("symbolic-ref", "HEAD", f"{BEST}{MODEL}")

    def blob(self, data: bytes) -> str:
        key = hashlib.sha256(data).digest()
        if (sha := self._blobs.get(key)) is None:
            sha = self._blobs[key] = self.run("hash-object", "-w", "--stdin", data=data).strip()
        return sha

    def tree(self, node: dict[str, Any]) -> str:
        """The tree of ``node`` ({name: bytes, or a node for a directory})."""
        entries = []
        for name, value in node.items():
            if isinstance(value, dict):
                entries.append(f"040000 tree {self.tree(value)}\t{name}")
            else:
                entries.append(f"100644 blob {self.blob(value)}\t{name}")
        key = tuple(sorted(entries))
        if (sha := self._trees.get(key)) is None:
            listing = "".join(f"{entry}\0" for entry in key).encode()
            sha = self._trees[key] = self.run("mktree", "-z", data=listing).strip()
        return sha

    def commit(self, tree: str, parents: list[str], message: str, author: str, date: str) -> str:
        who = {"NAME": author, "EMAIL": EMAIL, "DATE": date}
        env = {f"GIT_{role}_{k}": v for role in ("AUTHOR", "COMMITTER") for k, v in who.items()}
        args = ["commit-tree", "--no-gpg-sign", tree]
        for parent in parents:
            args += ["-p", parent]
        return self.run(*args, data=message.encode(), env=env).strip()

    def refs(self) -> dict[str, tuple[str, str]]:
        """Every experiment tag, reference tag and best branch: ref → (commit, its tree)."""
        listing = self.run(
            "for-each-ref",
            "--format=%(refname) %(objectname) %(tree)",
            TAGS.rstrip("/"),
            REFERENCES.rstrip("/"),
            BEST.rstrip("/"),
        )
        out = {}
        for line in listing.splitlines():
            ref, commit, tree = ([*line.split(), "", "", ""])[:3]
            out[ref] = (commit, tree)
        return out

    def update(self, changes: list[tuple[str, str, str | None]]) -> None:
        """Set every ``(ref, new, old)`` in one transaction, each only if the ref is still at
        ``old`` (None: if it does not exist): compare and swap."""
        if changes:
            lines = "".join(f"update {ref} {new} {old or ZERO}\n" for ref, new, old in changes)
            self.run("update-ref", "--stdin", data=lines.encode())


# ------------------------------------------------------------------ the lock


@contextlib.contextmanager
def _locked(repo: Path) -> Iterator[None]:
    """Hold the ``flock`` of ``repo``'s :data:`LOCK` (between processes, and between threads:
    each call opens its own descriptor). A lock taken on a repository that :func:`rebuild`
    replaced meanwhile is let go and taken again on the new one."""
    while True:
        repo.mkdir(parents=True, exist_ok=True)
        fd = os.open(repo / LOCK, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            here = os.fstat(fd)
            try:
                there = os.stat(repo / LOCK)
                same = (here.st_dev, here.st_ino) == (there.st_dev, there.st_ino)
            except FileNotFoundError:
                same = False
        except BaseException:
            os.close(fd)
            raise
        if same:
            break
        os.close(fd)
    try:
        yield
    finally:
        os.close(fd)  # releases the flock


# ------------------------------------------------------------------ what a commit holds


def _date(text: str | None) -> str | None:
    """git's date of a ledger time: the wall clock written as UTC (the ledger keeps no zone,
    so the hashes do not depend on the machine's)."""
    try:
        return f"{calendar.timegm(time.strptime(str(text), ledger.TIME_FORMAT))} +0000"
    except ValueError:
        return None


def _one_line(value: Any) -> str:
    return " ".join(str(value).split())


def _ref(name: str) -> str:
    """``name`` as one component of a ref name."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).replace("..", "_").strip("._")
    return re.sub(r"\.lock$", "_lock", safe) or "_"


def _author(session: str) -> str:
    session = re.sub(r"[<>\n]+", " ", session).strip()
    return f"kernel-agent ({session})" if session else "kernel-agent"


def _message(subject: str, body: list[str], trailers: list[tuple[str, Any]]) -> str:
    """A commit message: the subject, the body's paragraphs and the trailers that have a
    value (one line each)."""
    lines = [_one_line(subject)]
    paragraphs = [p for p in body if p.strip()]
    for paragraph in paragraphs:
        lines += ["", paragraph.rstrip()]
    pairs = [(k, _one_line(v)) for k, v in trailers if v is not None and _one_line(v)]
    if pairs:
        lines += ["", *(f"{k}: {v}" for k, v in pairs)]
    return "\n".join(lines) + "\n"


def _wrap(text: str) -> str:
    return textwrap.fill(" ".join(text.split()), 72, break_long_words=False, break_on_hyphens=False)


def _number(value: float | None) -> str | None:
    return None if value is None else f"{value:.6g}"


def _put(node: dict[str, Any], place: str, data: bytes) -> bool:
    """Put ``data`` at ``place`` in a tree; False when a file or directory is there already
    with other content (a path conflict)."""
    *dirs, name = place.split("/")
    for part in dirs:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            return False
    if name in node:
        return node[name] == data
    node[name] = data
    return True


def _read(run: RunDir, path: str, target: str | None) -> tuple[Path, bytes] | None:
    """The file a measured path names and its bytes: run-relative, or absolute (an older
    run's: when the run directory has moved since, its history snapshot of that name)."""
    p = Path(path)
    places = [p if p.is_absolute() else run.root / p]
    if p.is_absolute() and not p.is_relative_to(run.root):
        places.append(run.history_dir(target) / p.name)
    for where in places:
        try:
            if where.is_file():
                return where, where.read_bytes()
        except OSError:
            continue
    return None


def _bundle(where: Path, data: bytes) -> tuple[str, dict[str, str]] | None:
    """A native project's name and files when ``where`` is its bundle
    (``native/project.read_bundle``)."""
    if f"\n{native_project.BUNDLE_VAR} = ".encode() not in data:
        return None
    payload = native_project.read_bundle(where)
    if not payload:
        return None
    name, files = payload.get("name"), payload.get("files")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name.strip(".")):
        return None
    if not isinstance(files, dict) or not all(
        isinstance(k, str)
        and isinstance(v, str)
        and not ({"", ".", ".."} & set(k.split("/")))  # relative, inside the project
        for k, v in files.items()
    ):
        return None
    return name.strip("."), files


def contents(run: RunDir, files: Iterable[str]) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    """The tree of measured ``files`` (``ledger.kernel_files`` / ``ledger.item_files``):
    the tree, the sha256 of each file read and notes on what is left out (missing, not a
    snapshot, a path conflict)."""
    node: dict[str, Any] = {}
    digests: dict[str, str] = {}
    notes: list[str] = []
    for file in files:
        target, sep, path = file.partition("=")
        item = bool(sep) and "/" not in target
        if not item:
            target, path = "", file
        parts = Path(path).parts
        if item:
            place = f"targets/{target}/candidate.py"
        elif "targets" in parts[:-1]:
            target = parts[parts.index("targets") + 1]
            name = "rewrite.py" if parts[-1] == "rewrite.py" else "candidate.py"
            place = f"targets/{target}/{name}"
        else:
            place = f"transforms/{ledger.snapshot_stem(path)}.py"
        if "history" not in parts[:-1]:  # an agent's own file: it may have changed since
            notes.append(f"not a snapshot, left out: {path}")
            continue
        found = _read(run, path, target or None)
        if found is None:
            notes.append(f"not in the run directory: {path}")
            continue
        where, data = found
        digests[file] = hashlib.sha256(data).hexdigest()
        bundle = _bundle(where, data)
        if bundle is not None:  # the project's own files, so its versions diff file by file
            name, sources = bundle
            placed = [_put(node, f"native/{name}/{rel}", t.encode()) for rel, t in sources.items()]
            ok = all(placed)
        else:
            ok = _put(node, place, data)
        if not ok:
            notes.append(f"path conflict, left out: {path}")
    return node, digests, notes


# ------------------------------------------------------------------ the integration's files


def _integration_files(run: RunDir) -> dict[tuple[str, str], tuple[str, ...]]:
    """(snapshot cell, ms) → the files of an ``integration.json`` step, for integration rows
    recorded before their ``evaluation`` events listed files (only the last integration's
    steps are there; a row matches its step by the items' names and the measured ms)."""
    data = read_json(run.root / "integration.json", {}) or {}
    out: dict[tuple[str, str], tuple[str, ...]] = {}
    for step in data.get("history") or []:
        if not isinstance(step, dict) or not isinstance(step.get("median_ms"), int | float):
            continue
        items = [str(i) for i in step.get("items") or []]
        names = "+".join(ledger.item_label(i) for i in items) or "baseline"
        out.setdefault((names, f"{step['median_ms']:.6g}"), tuple(_item_files(run, items)))
    return out


def _item_files(run: RunDir, items: list[str]) -> list[str]:
    """``ledger.item_files``, with a transform's absolute path from where the run directory
    was then mapped to its history snapshot of that name."""
    out = []
    for f in ledger.item_files(run, items):
        p = Path(f)
        if p.is_absolute() and (run.history_dir() / p.name).is_file():
            f = ledger.relative(run, run.history_dir() / p.name)
        out.append(f)
    return out


def _files(
    e: Experiment,
    evaluated: set[int],
    integration: dict[tuple[str, str], tuple[str, ...]],
) -> tuple[str, ...] | None:
    """What an experiment measured (None: not recorded): its ``files``, else an older
    integration row's step in ``integration.json``; () for a row recorded with its event and
    no files: it measured the unmodified model."""
    if e.files:
        return e.files
    if e.kind in (ledger.INTEGRATION, ledger.PROBE) and e.row.get("new_ms") is not None:
        found = integration.get((str(e.row.get("snapshot") or ""), f"{e.row['new_ms']:.6g}"))
        if found is not None:
            return found
    if e.row.get("kind") and e.exp in evaluated:
        return ()
    return None


# ------------------------------------------------------------------ messages


def _subject(e: Experiment, reevaluates: int | None) -> str:
    if e.status == ledger.REEVALUATED:
        what = f"re-evaluated exp {reevaluates}" if reevaluates is not None else e.status
        value = experiments.value_text(e.value, e.unit) if e.value is not None else "not correct"
        return f"exp {e.exp} [{e.lineage}] {what} {value}: {e.title}"
    head = f"exp {e.exp} [{e.lineage}] {e.status}"
    if e.value is not None:
        head += f" {experiments.value_text(e.value, e.unit)}"
    if e.gain is not None:
        head += f" ({e.gain * 100:+.1f} %)"
    return f"{head}: {e.title}"


def _row_message(
    e: Experiment,
    files: tuple[str, ...] | None,
    digests: dict[str, str],
    notes: list[str],
    reevaluates: int | None,
) -> str:
    body = [_wrap(e.hypothesis)] if e.hypothesis.strip() else []
    if reevaluates is not None:
        body.append(f"Re-evaluates exp {reevaluates}: the same files, the current evaluator.")
    if files is None:
        body.append("files not recorded: the tree is its parent's")
    elif not files:
        body.append("measured no files: the unmodified model")
    else:
        body.append("\n".join(["measured:", *(f"  {f}" for f in files)]))
    if notes:
        body.append("\n".join(notes))
    snapshot = str(e.row.get("snapshot") or "")
    sha = digests.get(files[0]) if e.kind == ledger.KERNEL and files else None
    trailers: list[tuple[str, Any]] = [
        ("Exp", e.exp),
        ("Lineage", e.lineage),
        ("Kind", e.kind),
        ("Status", e.status),
        ("Value", _number(e.value)),
        ("Unit", e.unit),
        ("Snapshot", snapshot),
        ("Snapshot-Sha256", sha),
        ("Re-Evaluates", reevaluates),
        ("Idea", e.idea),
        ("Session", e.session),
        ("Worker", e.worker),
        ("Review", e.row.get("review")),
    ]
    return _message(_subject(e, reevaluates), body, trailers)


def _baseline_message(run: RunDir) -> str:
    base = read_json(run.baseline_json, {}) or {}
    ms = base.get("median_ms") if isinstance(base.get("median_ms"), int | float) else None
    data = run.load() if run.run_json.exists() else {}
    repo_id = (data.get("card") or {}).get("repo_id") or run.root.name
    gpu = ((read_json(run.toolchain_json, {}) or {}).get("gpu") or {}).get("name")
    value = f" {experiments.value_text(ms, MS)}" if ms is not None else ""
    facts = [
        f"metric: {objective.of(base).label}",
        f"GPU: {gpu or 'not recorded'}",
        f"workload: {base.get('workload') or 'not recorded'}",
    ]
    trailers: list[tuple[str, Any]] = [("Exp", 0), ("Lineage", MODEL), ("Kind", "baseline")]
    trailers += [("Value", _number(ms)), ("Unit", MS)]
    return _message(
        f"exp 0 [{MODEL}] baseline{value}: {repo_id}",
        ["The unmodified model: every lineage starts here.", "\n".join(facts)],
        trailers,
    )


def _reference_message(target: str, found: bool) -> str:
    where = f"targets/{target}/reference_source.py"
    text = (
        f"{where}: the module every experiment of this target is measured against."
        if found
        else f"{where} is not in the run directory."
    )
    return _message(
        f"reference [{target}] 1.00{X}: the module as captured",
        [text],
        [("Lineage", target), ("Kind", "reference"), ("Value", "1"), ("Unit", X)],
    )


# ------------------------------------------------------------------ sync


def _evaluated(run: RunDir) -> set[int]:
    """The ``exp`` of every row whose ``evaluation`` event is in ``events.jsonl``."""
    return {
        ev["exp"]
        for ev in experiments._jsonl(run.events)
        if ev.get("event") == "evaluation" and isinstance(ev.get("exp"), int)
    }


def _settled(e: Experiment, evaluated: set[int], now: float) -> bool:
    """Whether a row's files are final: a kernel row's follow from its snapshot, an
    end-to-end row's are in its event (:data:`SETTLE_S`)."""
    if e.kind == ledger.KERNEL or e.exp in evaluated:
        return True
    t = ledger.epoch(e.time)
    return t is None or now - t >= SETTLE_S


def _committed(e: Experiment) -> bool:
    """Whether a row becomes a commit: every measured row and every re-evaluation (quick
    checks and duplicates measured nothing)."""
    return e.measured or e.status == ledger.REEVALUATED


class _Writer:
    """One sync's commits: what the repository has (``refs``), what this sync adds."""

    def __init__(self, run: RunDir, git: Git, items: list[Experiment], evaluated: set[int]):
        self.run, self.git, self.evaluated = run, git, evaluated
        self.refs = git.refs()
        self.trees = {commit: tree for commit, tree in self.refs.values()}  # commit → tree
        self.commits: dict[int, str] = {}  # exp → commit
        self.roots: dict[str, str] = {}  # a target's ref name → its reference commit
        for ref, (commit, _) in self.refs.items():
            if ref.startswith(TAGS) and ref[len(TAGS) :].isdigit():
                self.commits[int(ref[len(TAGS) :])] = commit
            elif ref.startswith(REFERENCES):
                self.roots[ref[len(REFERENCES) :]] = commit
        self.created: list[tuple[str, str, str | None]] = []  # (ref, commit, None): new refs
        self.latest: dict[tuple[str, str], int] = {}  # (target, snapshot name) → latest exp
        self.first = _date(next((e.time for e in items if e.time), None)) or "0 +0000"
        self.started: dict[str, str] = {}  # lineage → the date of its first experiment
        for e in items:
            if _committed(e) and (date := _date(e.time)):
                self.started.setdefault(e.lineage, date)
        self._integration: dict[tuple[str, str], tuple[str, ...]] | None = None

    def _new(self, ref: str, commit: str, tree: str) -> str:
        self.created.append((ref, commit, None))
        self.trees[commit] = tree
        return commit

    def baseline(self) -> str:
        """``exp 0``: the unmodified model, dated by ``run.json`` ``created``."""
        if 0 not in self.commits:
            data = self.run.load() if self.run.run_json.exists() else {}
            empty = self.git.tree({})
            date = _date(data.get("created")) or self.first
            commit = self.git.commit(empty, [], _baseline_message(self.run), _author(""), date)
            self.commits[0] = self._new(f"{TAGS}0", commit, empty)
        return self.commits[0]

    def root(self, lineage: str) -> str:
        """Where a lineage starts: the baseline, or a target's reference commit (made with
        the target's first experiment and dated by it, also for a target added in a later
        round)."""
        if lineage == MODEL:
            return self.baseline()
        name = _ref(lineage)
        if name not in self.roots:
            source = self.run.target(lineage) / "reference_source.py"
            node: dict[str, Any] = {}
            if found := source.is_file():
                _put(node, f"targets/{lineage}/candidate.py", source.read_bytes())
            tree = self.git.tree(node)
            message = _reference_message(lineage, found)
            date = self.started.get(lineage, self.first)
            commit = self.git.commit(tree, [self.baseline()], message, _author(""), date)
            self.roots[name] = self._new(f"{REFERENCES}{name}", commit, tree)
        return self.roots[name]

    def files(self, e: Experiment) -> tuple[str, ...] | None:
        if not e.files and self._integration is None and e.lineage == MODEL:
            self._integration = _integration_files(self.run)
        return _files(e, self.evaluated, self._integration or {})

    def commit(self, e: Experiment) -> str:
        """The commit of one experiment (its tag goes into :attr:`created`)."""
        date = _date(e.time) or self.first
        root = self.root(e.lineage)
        reevaluates = None
        if e.status == ledger.REEVALUATED:  # a child of the experiment it re-evaluates
            reevaluates = self.latest.get((e.lineage, Path(str(e.row.get("snapshot"))).name))
        start = reevaluates if reevaluates is not None else e.parent_exp
        parents = [self.commits.get(start, root) if start is not None else root]
        if e.lineage == MODEL:  # the lines of its kernel items merge into the model's
            for file in e.files:
                target, sep, path = file.partition("=")
                item = sep and "/" not in target  # target=path, not a transform's path
                exp = self.latest.get((target, Path(path).name)) if item else None
                if exp is not None and self.commits[exp] not in parents:
                    parents.append(self.commits[exp])
        files = self.files(e)
        node, digests, notes = contents(self.run, files or ())
        if files is None:  # not recorded: no tree change
            first = parents[0]
            tree = self.trees.get(first) or self.git.run("rev-parse", f"{first}^{{tree}}").strip()
        else:
            tree = self.git.tree(node)
        message = _row_message(e, files, digests, notes, reevaluates)
        commit = self.git.commit(tree, parents, message, _author(e.session), date)
        self.commits[e.exp] = self._new(f"{TAGS}{e.exp}", commit, tree)
        return commit

    def tip(self, lineage: str, stand: Standing) -> str:
        """``best/<lineage>``: the commit of its standing best, else where it starts."""
        top = stand.top
        return self.commits[int(top["exp"])] if top is not None else self.root(lineage)


def _sync(run: RunDir, git: Git) -> Synced:
    """:func:`sync` under the lock: every row without its commit, in ``exp`` order, then the
    new tags and the moved branches in one transaction."""
    rows = ledger.rows(run)
    if not rows:
        return Synced(git.repo, 0, None)
    git.init()
    # the events before all_rows reads them again: a row this sync calls settled has its
    # event's files in its Experiment too
    evaluated = _evaluated(run)
    items = experiments.all_rows(run, rows)
    now = max([ledger.clock(), *(t for e in items if (t := ledger.epoch(e.time)) is not None)])
    writer = _Writer(run, git, items, evaluated)
    stands: dict[str, Standing] = {}
    for e in items:
        if not _committed(e):
            continue
        if e.exp not in writer.commits:
            if not _settled(e, evaluated, now):
                break  # strictly in exp order: the rows after it wait as well
            writer.commit(e)
        if e.kind == ledger.KERNEL:
            writer.latest[(e.lineage, Path(str(e.row.get("snapshot") or "")).name)] = e.exp
        stand = stands.setdefault(e.lineage, Standing())  # the bar, as the ledger keeps it
        if e.status == ledger.REEVALUATED:
            stand.replace(e.row)
        elif e.status == ledger.KEEP:
            stand.keep(e.row)
    if 0 in writer.commits:
        stands.setdefault(MODEL, Standing())  # HEAD's branch, from exp 0 on
    changes = list(writer.created)
    for lineage, stand in stands.items():
        ref, tip = f"{BEST}{_ref(lineage)}", writer.tip(lineage, stand)
        old = writer.refs[ref][0] if ref in writer.refs else None
        if old != tip:
            changes.append((ref, tip, old))
    git.update(changes)
    return Synced(git.repo, len(writer.created), max(writer.commits, default=None))


_errors: dict[Path, str] = {}  # repository → the error of its last failed sync in this process
_errors_lock = threading.Lock()


def last_error(repo: Path) -> str | None:
    """Why the last :func:`sync` of ``repo`` in this process failed (None: it did not)."""
    with _errors_lock:
        return _errors.get(repo.resolve())


def sync(run: RunDir, repo: Path | None = None) -> Synced | None:
    """Commit the ledger's rows that ``repo`` (default ``<run>/experiments.git``) has no
    commit for yet, in ``exp`` order, under the repository's lock. Idempotent and never
    raises: None when git is missing or failed. Into the run's repository, a failure is an
    ``exp_git`` event (once per error and process; :func:`last_error`); a ``repo`` elsewhere
    leaves the run directory alone."""
    target = (repo or repo_path(run)).resolve()
    exe = shutil.which("git")
    try:
        if exe is None:
            raise GitError("git is not on PATH")
        if not ledger.rows(run):  # nothing to commit: no repository yet either
            return Synced(target, 0, None)
        with _locked(target):
            done = _sync(run, Git(exe, target))
    except Exception as exc:  # derived data: a failure never stops the run
        error = str(exc) if isinstance(exc, GitError) else f"{type(exc).__name__}: {exc}"
        with _errors_lock:
            again, _errors[target] = _errors.get(target) == error, error
        if repo is None and not again:
            with contextlib.suppress(OSError):
                ledger.event(run, EVENT, error=error[:500], repo=REPO)
        return None
    with _errors_lock:
        _errors.pop(target, None)
    return done


def rebuild(run: RunDir, out: Path | None = None) -> Synced | None:
    """A new repository from the whole ledger: at ``out`` (missing or an empty directory;
    the run directory is not written), else in place of ``<run>/experiments.git``, built
    beside it and swapped in under its lock (None: git is missing or failed,
    :func:`last_error`)."""
    if out is not None:
        out = out.resolve()
        if out.exists() and (not out.is_dir() or any(out.iterdir())):
            raise FileExistsError(f"{out} exists and is not an empty directory")
        return sync(run, out)
    target = repo_path(run).resolve()
    fresh = target.with_name(f".{REPO}.new-{os.getpid()}-{threading.get_native_id()}")
    old = target.with_name(f".{REPO}.old-{os.getpid()}-{threading.get_native_id()}")
    shutil.rmtree(fresh, ignore_errors=True)
    built = sync(run, fresh)
    if built is None:
        with _errors_lock:
            error = _errors.pop(fresh, None)
            if error is not None:
                _errors[target] = error
        shutil.rmtree(fresh, ignore_errors=True)
        return None
    with _locked(target):  # a writer waiting for the old one takes the new one's lock
        target.rename(old)
        fresh.rename(target)
    shutil.rmtree(old, ignore_errors=True)
    return Synced(target, built.added, built.last)


def diff(run: RunDir, a: int, b: int) -> str | None:
    """``git diff exp/B exp/A``: what A changes against B, when the run's repository has
    both (None: no git, no repository, or not synced that far). Reads only, no lock."""
    repo = repo_path(run)
    exe = shutil.which("git")
    if exe is None or not (repo / "HEAD").is_file():
        return None
    try:
        return Git(exe, repo).run("diff", f"{TAGS}{b}", f"{TAGS}{a}", "--")
    except GitError:
        return None


# ------------------------------------------------------------------ the CLI


def render(run: RunDir, repo: Path) -> str:
    """The repository, how far it is synced and its lineages: each ``best/<lineage>`` with
    its tip, the tip's experiment, the keeps along its first parents and its subject."""
    exe = shutil.which("git")
    if exe is None:
        return "git is not on PATH"
    if not (repo / "HEAD").is_file():
        return f"no {repo}: kernel-agent exp git {run.root} --sync"
    git = Git(exe, repo)
    listing = git.run(
        "for-each-ref",
        "--format=%(refname)%00%(objectname:short)%00%(objectname)%00%(contents:subject)",
        TAGS.rstrip("/"),
        BEST.rstrip("/"),
    )
    tags: dict[str, int] = {}
    branches = []
    for line in listing.splitlines():
        ref, short, full, subject = ([*line.split("\x00"), "", "", ""])[:4]
        if ref.startswith(TAGS) and ref[len(TAGS) :].isdigit():
            tags[full] = int(ref[len(TAGS) :])
        elif ref.startswith(BEST):
            branches.append((ref[len("refs/heads/") :], short, full, subject))
    recorded = max((r["exp"] for r in ledger.rows(run) if isinstance(r.get("exp"), int)), default=0)
    synced = max(tags.values(), default=None)
    lines = [
        f"{repo} (bare; derived from results.tsv: --rebuild recreates it)",
        f"synced up to exp {synced} of {recorded}: {len(tags)} experiment commits "
        "(exp 0 is the baseline; quick checks and duplicates have none)"
        if synced is not None
        else "no experiment committed yet",
    ]
    if not branches:
        return "\n".join(lines)
    head = f"best/{MODEL}"
    table_rows = []
    for name, short, full, subject in sorted(branches, key=lambda b: (b[0] != head, b[0])):
        statuses = git.run(
            "log", "--first-parent", "--format=%(trailers:key=Status,valueonly)", name, "--"
        )
        kept = sum(line.strip() == ledger.KEEP for line in statuses.splitlines())
        exp = tags.get(full)
        table_rows.append(
            [
                name + (" (HEAD)" if name == head else ""),
                short,
                "ref" if exp is None else str(exp),
                str(kept),
                subject,
            ]
        )
    lines.append("")
    lines += experiments.table(["branch", "tip", "exp", "kept", "subject"], table_rows, {2, 3})
    lines += [
        "",
        f"kernel-agent exp git {run.root} -- log --oneline --graph {head}",
        f"git --git-dir {repo} show exp/N",
    ]
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kernel-agent exp git",
        description="The run's experiments as a bare git repository, <run>/experiments.git: "
        "one commit per experiment (tags exp/N), best/<lineage> branches of the kept line.\n"
        "  kernel-agent exp git RUN                  the repository and its lineages\n"
        "  kernel-agent exp git RUN --sync           commit the rows it has no commit for\n"
        "  kernel-agent exp git RUN --rebuild [--out DIR]\n"
        "                                            a fresh repository from the ledger\n"
        "  kernel-agent exp git RUN -- ARGS...       git ARGS with --git-dir set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("run_dir")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--sync", action="store_true", help="catch up with the ledger")
    mode.add_argument("--rebuild", action="store_true", help="build a fresh repository")
    p.add_argument(
        "--out",
        type=Path,
        metavar="DIR",
        help="the repository at DIR instead of <run>/experiments.git (the run directory is "
        "not written; --rebuild: DIR missing or empty)",
    )
    return p


def main(argv: list[str]) -> int:
    """``kernel-agent exp git RUN [--sync | --rebuild] [--out DIR] [-- GIT ARGS...]``."""
    passthrough: list[str] | None = None
    if "--" in argv:
        at = argv.index("--")
        argv, passthrough = argv[:at], argv[at + 1 :]
    args = _parser().parse_args(argv)
    run = experiments._run(args.run_dir)
    repo = args.out.resolve() if args.out else repo_path(run).resolve()
    if args.sync or args.rebuild:
        try:
            done = rebuild(run, args.out) if args.rebuild else sync(run, args.out)
        except FileExistsError as exc:
            print(f"kernel-agent exp git: {exc}", file=sys.stderr)
            return 1
        if done is None:
            print(f"kernel-agent exp git: {last_error(repo) or 'failed'}", file=sys.stderr)
            return 1
        last = "no experiment yet" if done.last is None else f"up to exp {done.last}"
        print(f"{done.repo}: {done.added} new commits, {last}")
    if passthrough is not None:
        exe = shutil.which("git")
        if exe is None:
            print("kernel-agent exp git: git is not on PATH", file=sys.stderr)
            return 1
        sys.stdout.flush()  # what this printed comes before git's output
        return subprocess.run([exe, f"--git-dir={repo}", *passthrough], check=False).returncode
    print(render(run, repo))
    return 0
