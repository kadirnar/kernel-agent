"""Tamper-evident ground truth of a run (what the evaluator trusts).

Agents run with ``bypassPermissions`` and a Bash tool, so every file of the
run directory is within their reach. Everything the evaluator trusts therefore
lives outside the agents' working directories, in ``<run>/.truth/``::

    .truth/baseline_output.pt         reference output of the baseline run
    .truth/baseline_output_holdout.pt ... of the held-out input (workloads/holdout.py)
    .truth/baseline_output_natural.pt ... of the natural-length run (workloads/stopping.py)
    .truth/baseline_output_diverse.pt ... of the diverse input set (workloads/diverse.py)
    .truth/baseline_output_perceptual.pt samples + scores of the perceptual gate
                                      (--quality near-lossless / relaxed, perceptual.py)
    .truth/captures/<id>.pt           module + inputs + reference outputs + post-call state
    .truth/targets/<id>/results.jsonl evaluation records (each with its snapshot's sha256)
    .truth/targets/<id>/history/      the evaluated snapshots
    .truth/transforms/results.jsonl   ... and history/, for the model-level transforms

plus ``baseline.json`` and ``integration.json`` in the run root. The agent's
``targets/<id>/`` keeps ``spec.json``, ``reference_source.py``, ``candidates/``,
``NOTES.md``, an inputs-only ``capture_inputs.pt`` (no outputs, no post-call
state) for local debugging, and copies of its snapshots (``history/``) and
records (``results.jsonl``) that nothing reads back.

File permissions (``chmod a-w``) only stop accidents: a Bash tool can ``chmod``.
The defence is the sha256 (and size) of every truth file, recorded when
kernel-agent writes it, held in this process's memory (:class:`Truth`, one per
run and process, see :func:`of`) and mirrored in ``run.json`` ``truth`` for a
resumed run. Before use, the evaluator (``capture_sha256``), the e2e worker
(``--verify`` + ``--baseline-ms``; ``--quality`` passes the run's quality mode the same
way), :func:`kernel_agent.agent.tools.best_for_target`,
the integration and the export check the digests; a mismatch is refused with
a ``TAMPER`` line on stderr and a ``tamper`` event in ``events.jsonl``.
Records appended to a results file outside kernel-agent are ignored (the
recorded size marks the authentic prefix); a record whose snapshot is missing
or changed is ignored as well.

Runs created before this layout have no ``truth`` section in ``run.json``;
they keep their old paths and are read without checks.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import stat
import sys
import threading
import time
from pathlib import Path
from typing import Any

from kernel_agent import ledger
from kernel_agent.workspace import TRUTH_DIR, RunDir, write_json

VERSION = 1
#: Case fields of a capture that hold the answer key (reference outputs + side effects; a
#: case's module ``state`` before its call is an input and stays, profiling/state.py).
ANSWER_KEYS = ("output", "post_args", "post_kwargs", "post_state")

__all__ = ["TRUTH_DIR", "TamperError", "Truth", "of"]


class TamperError(RuntimeError):
    """A truth file does not match the digest kernel-agent recorded for it."""


# ------------------------------------------------------------------ digests + files


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def read_verified(path: Path, sha256: str | None) -> bytes:
    """The bytes of ``path``, checked against ``sha256`` (None: unchecked).

    Read once, so the file cannot change between the check and its use."""
    data = Path(path).read_bytes()
    if sha256 is not None and (found := sha256_bytes(data)) != sha256:
        raise TamperError(
            f"{path} was modified after kernel-agent wrote it "
            f"(sha256 {found[:12]}…, expected {sha256[:12]}…); refusing to use it"
        )
    return data


def read_only(path: Path) -> None:
    """``chmod a-w`` (a speed bump only: the digests are the defence)."""
    with contextlib.suppress(OSError):
        mode = Path(path).stat().st_mode
        os.chmod(path, mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def writable(path: Path) -> None:
    with contextlib.suppress(OSError):
        os.chmod(path, Path(path).stat().st_mode | stat.S_IWUSR)


def replace(path: Path) -> Path:
    """Make room for a new version of a (read-only) truth file; returns ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)  # a read-only file in a writable directory
    return path


def alarm(run: RunDir, file: str, problem: str, action: str = "refused") -> None:
    """Report a tampered truth file: a loud line on stderr and a ``tamper`` event."""
    print(
        f"[{time.strftime('%H:%M:%S')}] TAMPER: {file}: {problem} ({action})",
        file=sys.stderr,
        flush=True,
    )
    with contextlib.suppress(OSError):
        ledger.event(run, "tamper", file=file, problem=problem[:500], action=action)


# ------------------------------------------------------------------ captures


def inputs_only(capture: dict[str, Any]) -> dict[str, Any]:
    """A capture without the answer key: module + args/kwargs of every case."""
    cases = [{k: v for k, v in c.items() if k not in ANSWER_KEYS} for c in capture["cases"]]
    return {**capture, "cases": cases, "inputs_only": True}


def write_inputs_capture(full: Path, dst: Path) -> Path:
    """``dst``: the inputs-only copy of the capture file ``full`` (for the agent's cwd)."""
    import torch

    from kernel_agent.profiling.capture import load_capture

    torch.save(inputs_only(load_capture(full)), replace(dst))
    return dst


# ------------------------------------------------------------------ the registry


def new_section() -> dict[str, Any]:
    """``run.json`` ``truth`` of a new run: marks the ``.truth/`` layout."""
    return {"version": VERSION, "baseline_ms": None, "files": {}}


class Truth:
    """Digests of one run's truth files; this process's memory is the authority.

    Created from ``run.json`` (trusted when the process starts) and updated by
    :meth:`seal` / :meth:`append`, which write the files' digests back to
    ``run.json``. Use :func:`of` to share one instance per run in a process.
    """

    def __init__(self, run: RunDir) -> None:
        self.run = run
        data = run.load()
        section = data.get("truth")
        self.enabled = isinstance(section, dict)
        section = section if isinstance(section, dict) else {}
        self.files: dict[str, dict[str, Any]] = dict(section.get("files") or {})
        ms = section.get("baseline_ms")
        self.recorded_baseline_ms: float | None = float(ms) if ms else None
        #: ``--quality`` of the run (exact / near-lossless / relaxed): how e2e judges, so it is
        #: passed to the worker from here, not read back from the agent-writable ``run.json``.
        self.quality = str((data.get("config") or {}).get("quality") or "exact")
        self._lock = threading.RLock()
        self._reported: set[tuple[str, str]] = set()

    # -------------------------------------------------------- bookkeeping

    def rel(self, path: Path) -> str:
        path = Path(path)
        try:
            return path.relative_to(self.run.root).as_posix()
        except ValueError:
            try:
                return path.resolve().relative_to(self.run.root.resolve()).as_posix()
            except ValueError:
                return str(path)

    def _persist(self) -> None:
        data = self.run.load()
        data["truth"] = {
            "version": VERSION,
            "baseline_ms": self.recorded_baseline_ms,
            "files": self.files,
        }
        write_json(self.run.run_json, data)

    def alarm(self, path: Path | str, problem: str, action: str = "refused") -> None:
        """:func:`alarm`, once per file and problem in this process."""
        file = self.rel(Path(path)) if isinstance(path, Path) else path
        if (file, problem) not in self._reported:
            self._reported.add((file, problem))
            alarm(self.run, file, problem, action)

    def _refuse(self, path: Path, problem: str) -> TamperError:
        self.alarm(path, problem)
        return TamperError(f"{self.rel(path)}: {problem}; refusing to use it")

    # -------------------------------------------------------- writing

    def seal(self, path: Path) -> str:
        """Record the digest of a file kernel-agent just wrote and make it read-only."""
        digest = sha256_file(path)
        if not self.enabled:
            return digest
        with self._lock:
            self.files[self.rel(path)] = {"sha256": digest, "bytes": Path(path).stat().st_size}
            read_only(path)
            self._persist()
        return digest

    def _baseline_files(self) -> tuple[Path, ...]:
        run = self.run
        return (
            run.baseline_json,
            run.baseline_output(),
            run.baseline_output_holdout(),
            run.baseline_output_natural(),
            run.baseline_output_diverse(),
            run.baseline_output_perceptual(),
        )

    def seal_baseline(self, median_ms: float) -> None:
        """``baseline.json`` + the baseline outputs, and the latency every speedup divides."""
        if not self.enabled:
            return
        with self._lock:
            self.recorded_baseline_ms = float(median_ms)
            for path in self._baseline_files():
                if path.exists():
                    self.seal(path)
            self._persist()

    def append(self, path: Path, record: dict[str, Any]) -> None:
        """Append a record to a results file (verified first; foreign lines dropped)."""
        line = (json.dumps(record, default=str) + "\n").encode()
        with self._lock:
            if not self.enabled:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("ab") as fh:
                    fh.write(line)
                return
            head = self._authentic(path)  # raises when the recorded part was changed
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                writable(path)
            with path.open("r+b" if path.exists() else "wb") as fh:
                fh.seek(len(head))
                fh.truncate()  # lines appended outside kernel-agent go
                fh.write(line)
            self.files[self.rel(path)] = {
                "sha256": sha256_bytes(head + line),
                "bytes": len(head) + len(line),
            }
            read_only(path)
            self._persist()

    # -------------------------------------------------------- checking

    def expect(self, path: Path) -> str | None:
        """The recorded sha256 of ``path`` (None for a run without ``.truth/``)."""
        if not self.enabled:
            return None
        entry = self.files.get(self.rel(path))
        if entry is None:
            raise self._refuse(path, "no digest recorded (not written by kernel-agent)")
        return str(entry["sha256"])

    def verify(self, path: Path) -> str | None:
        """Check a file against its recorded digest; returns it (None: nothing to check)."""
        expected = self.expect(path)
        if expected is None:
            return None
        try:
            found = sha256_file(path)
        except OSError:
            raise self._refuse(path, "missing") from None
        if found != expected:
            raise self._refuse(path, f"modified (sha256 {found[:12]}…, recorded {expected[:12]}…)")
        return expected

    def _authentic(self, path: Path) -> bytes:
        """The part of an append-only file that kernel-agent wrote (b"" when none)."""
        entry = self.files.get(self.rel(path))
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            if entry is not None:
                raise self._refuse(path, "missing") from None
            return b""
        if entry is None:
            if data:
                self.alarm(path, "not written by kernel-agent", "ignored")
            return b""
        size = int(entry["bytes"])
        head = data[:size]
        if len(head) < size or sha256_bytes(head) != entry["sha256"]:
            raise self._refuse(path, "records written by kernel-agent were modified")
        if len(data) > size:
            self.alarm(path, f"{len(data) - size} bytes appended outside kernel-agent", "ignored")
        return head

    def records(self, path: Path) -> list[dict[str, Any]]:
        """The records of a results file that kernel-agent wrote (raises :class:`TamperError`
        when they were modified; lines appended by anyone else are ignored)."""
        with self._lock:
            data = self._authentic(path) if self.enabled else _read(path)
        return [json.loads(line) for line in data.decode().splitlines() if line.strip()]

    def snapshot_ok(self, path: Path, sha256: str | None) -> bool:
        """Whether a record's snapshot still is the file that was evaluated."""
        if sha256 is None and not self.enabled:
            return True  # a record of a run without ``.truth/``
        if sha256 is None:
            self.alarm(path, "record without a snapshot digest", "ignored")
            return False
        try:
            found = sha256_file(path)
        except OSError:
            self.alarm(path, "snapshot of a record is missing", "ignored")
            return False
        if found != sha256:
            self.alarm(path, "snapshot changed since it was evaluated", "ignored")
            return False
        return True

    def load_json(self, path: Path) -> Any:
        """A sealed JSON file ({} when missing or tampered with; tampering is reported)."""
        if not path.exists():
            return {}
        try:
            expected = self.expect(path)
        except TamperError:
            return {}
        data = path.read_bytes()
        if expected is not None and sha256_bytes(data) != expected:
            self.alarm(path, "modified since kernel-agent wrote it", "ignored")
            return {}
        return json.loads(data)

    def baseline_ms(self) -> float:
        """The baseline latency that every end-to-end speedup divides."""
        if self.enabled and self.recorded_baseline_ms:
            return self.recorded_baseline_ms
        data = json.loads(_read(self.run.baseline_json) or b"{}")
        return float(data["median_ms"])

    def worker_args(self) -> list[str]:
        """``e2e`` worker flags: the baseline latency, the digests it must verify and the
        quality mode."""
        if not self.enabled:
            return []
        args = []
        if self.recorded_baseline_ms:
            args += ["--baseline-ms", repr(self.recorded_baseline_ms)]
        for path in self._baseline_files():
            entry = self.files.get(self.rel(path))
            if entry is not None:
                args += ["--verify", f"{self.rel(path)}={entry['sha256']}"]
        return [*args, "--quality", self.quality]


def _read(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return b""


_registry: dict[Path, Truth] = {}
_registry_lock = threading.Lock()


def of(run: RunDir) -> Truth:
    """The :class:`Truth` of ``run`` in this process (one per run: memory is the authority)."""
    key = run.root.resolve()
    with _registry_lock:
        truth = _registry.get(key)
        if truth is None or (not truth.enabled and run.sealed()):  # sealed since: reload
            truth = _registry[key] = Truth(run)
        return truth
