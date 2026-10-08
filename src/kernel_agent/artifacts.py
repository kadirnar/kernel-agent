"""Load another file of the run from a transform or a kernel (issue #171).

A transform or kernel that uses another evaluated file of the run (a snapshot in
``history/``: a native project's bundle, an earlier version, a helper module) loads it
through this module instead of building a path to it::

    from kernel_agent import artifacts

    mk = artifacts.load("028_decode_step_megakernel_2c930eef.py")  # the module
    path = artifacts.find("028_decode_step_megakernel_2c930eef.py")  # or just its path

A name (a file name, a relative path, or a native project directory) is looked up from
the directory of the calling file: there, in its ``history/``, in ``../history/`` and in
``..`` (:data:`SEARCH`). The same call then finds the file next to the evaluated snapshot
(``.truth/transforms/history/``), next to the agent's copy (``transforms/history/``) and
in the exported package (``optimized/transforms/``, where the export puts it).

Every lookup is recorded: in this process (:func:`lookups`) and, when
``$KERNEL_AGENT_ARTIFACTS_LOG`` names a file (the worker sets it to
``<run>/logs/artifacts.jsonl``), as a line there with the calling file's sha256. The
export (``integrate/deps.py``) copies what an accepted item looked up into ``optimized/``,
also a name computed at run time; names in string literals it finds without this record.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

#: Where a name is looked for, relative to the directory of the file that asks, in order.
SEARCH = (".", "history", "../history", "..")
#: The file the lookups of this process are appended to (JSON lines), when set.
LOG_ENV = "KERNEL_AGENT_ARTIFACTS_LOG"
LOG_NAME = "artifacts.jsonl"

_LOOKUPS: list[dict[str, Any]] = []


def find(name: str | os.PathLike[str], *, near: str | os.PathLike[str] | None = None) -> Path:
    """The run file or directory ``name``, looked up from the directory of ``near`` (default:
    the calling file) along :data:`SEARCH`; raises ``FileNotFoundError`` naming it."""
    caller = _caller(near)
    here = caller if caller.is_dir() else caller.parent
    named = Path(name)
    dirs = [here] if named.is_absolute() else [Path(os.path.normpath(here / d)) for d in SEARCH]
    for d in dict.fromkeys(dirs):
        path = d / named
        if path.exists():
            found = Path(os.path.normpath(path))
            _record(caller, here, str(name), found)
            return found
    tried = ", ".join(str(d) for d in dict.fromkeys(dirs))
    raise FileNotFoundError(errno.ENOENT, f"run artifact not found (looked in {tried})", str(name))


def load(name: str | os.PathLike[str], *, near: str | os.PathLike[str] | None = None) -> ModuleType:
    """Import the run's Python file ``name`` (a snapshot, a native project's bundle, a helper)
    or native project directory, found as :func:`find` finds it; a fresh module per call."""
    path = find(name, near=near if near is not None else _caller(None))
    if path.is_dir():
        from kernel_agent.native import project

        return project.import_project(path)
    data = path.read_bytes()
    stem = re.sub(r"\W", "_", path.stem)
    module_name = f"ka_artifact_{stem}_{hashlib.sha1(data).hexdigest()[:10]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def lookups() -> list[dict[str, Any]]:
    """The lookups of this process: ``caller``, ``name``, ``path`` and ``rel`` (the found
    path relative to the caller's directory)."""
    return [dict(x) for x in _LOOKUPS]


def log_file(root: Path) -> Path:
    """Where the worker processes of the run in ``root`` record their lookups."""
    return Path(root) / "logs" / LOG_NAME


def recorded(root: Path) -> list[dict[str, Any]]:
    """The lookups recorded for the run in ``root`` (lines that do not parse are skipped)."""
    out: list[dict[str, Any]] = []
    with contextlib.suppress(OSError):
        for line in log_file(root).read_text().splitlines():
            with contextlib.suppress(ValueError):
                if isinstance(entry := json.loads(line), dict):
                    out.append(entry)
    return out


def _caller(near: str | os.PathLike[str] | None, depth: int = 2) -> Path:
    """``near``, else the file of the frame ``depth`` levels up (the caller of the public
    function that called this), else the working directory."""
    if near is None:
        near = sys._getframe(depth).f_globals.get("__file__") or os.getcwd()
    return Path(os.path.abspath(near))


def _record(caller: Path, here: Path, name: str, found: Path) -> None:
    sha256 = None
    if caller.is_file():
        with contextlib.suppress(OSError):
            sha256 = hashlib.sha256(caller.read_bytes()).hexdigest()
    entry = {
        "caller": str(caller),
        "caller_sha256": sha256,
        "name": name,
        "path": str(found),
        "rel": Path(os.path.relpath(found, here)).as_posix(),
    }
    _LOOKUPS.append(entry)
    if log := os.environ.get(LOG_ENV):
        with contextlib.suppress(OSError):
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            with open(log, "a") as fh:
                fh.write(json.dumps(entry) + "\n")
