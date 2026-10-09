"""The run files an accepted item needs besides itself (issue #171).

A transform or kernel can load another file of the run: a snapshot by its history name (a
native project's bundle, an earlier version), a helper module next to it, a native project
directory. The export (``integrate/export.py``) copies each into ``optimized/`` where the
item's own lookup finds it, without rewriting the item: at the same position relative to
the item's directory (a sibling snapshot next to it, ``../history/x.py`` in
``optimized/history/``), or next to the item (by name, where :mod:`kernel_agent.artifacts`
looks first) when that position would be outside the package.

They are found two ways, and followed: a needed ``.py`` file's own needs count too.

* statically (:func:`_static`): string literals that name an existing source file
  (:data:`~kernel_agent.integrate.reuse.SOURCES`; a history snapshot's name also without
  ``.py``) or a native project directory, relative to the file's directory along
  :data:`kernel_agent.artifacts.SEARCH` or to the run directory, or an absolute path in the
  run directory; and the modules it imports from its own directory, which the run's loaders
  and ``apply.py`` put on ``sys.path``;
* at run time (:func:`_recorded`): the lookups through :mod:`kernel_agent.artifacts` the
  run's worker processes recorded (``logs/artifacts.jsonl``) for a file with the same
  sha256, which also covers a name computed at run time.

Only files in the run directory count: the installed packages (``kernel_agent`` included)
are the package's requirements, not its contents.
"""

from __future__ import annotations

import ast
import hashlib
import os
import posixpath
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernel_agent import artifacts
from kernel_agent.integrate.reuse import SOURCES

#: A history snapshot's file name without ``.py`` (``028_decode_step_megakernel_2c930eef``).
SNAPSHOT_NAME = re.compile(r"\d{3,}_\w+")
#: Files at the top of ``optimized/`` that a needed file never replaces.
RESERVED = frozenset(
    {"apply.py", "manifest.json", "phases.py", "export_check.json", "requirements.txt"}
)
#: Needed files per item at most (a literal naming a large directory tree is not followed).
MAX_FILES = 500


@dataclass(frozen=True)
class Need:
    """A run file an item needs: ``source``, and ``place``: where it goes, relative to the
    directory of the exported item (POSIX, may start with one ``../``)."""

    source: Path
    place: str
    by: str  # the file (name) that loads it
    how: str  # "name" (a string literal), "import", "project" (a native project's file), "recorded"


def needs(path: Path, root: Path, records: Iterable[dict[str, Any]] = ()) -> list[Need]:
    """What the item ``path`` of the run in ``root`` needs (see the module docstring), in the
    order found; ``records``: the run's recorded lookups (``artifacts.recorded``)."""
    path, root = Path(path).resolve(), Path(root).resolve()
    by_sha: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        if isinstance(sha := rec.get("caller_sha256"), str):
            by_sha.setdefault(sha, []).append(rec)
    found: dict[tuple[str, Path], Need] = {}  # two files for one place: the export says so
    todo, seen = [(path, "")], {path}
    while todo and len(found) < MAX_FILES:
        current, base = todo.pop(0)
        for source, rel, how in [*_static(current, root), *_recorded(current, root, by_sha)]:
            place = _place(base, rel, source.name)
            if source.is_dir():  # a native project: its files, at the same place
                for name in _project_files(source):
                    file, at = source / name, posixpath.join(place, name)
                    found.setdefault((at, file), Need(file, at, current.name, "project"))
                continue
            if source == path:
                continue
            found.setdefault((place, source), Need(source, place, current.name, how))
            if source.suffix == ".py" and source not in seen:
                seen.add(source)
                todo.append((source, posixpath.dirname(place)))
    return list(found.values())[:MAX_FILES]


def _place(base: str, rel: str, name: str) -> str:
    """Where a file the file at ``base`` reaches as ``rel`` goes (relative to the exported
    item's directory): the same relative position, or by ``name`` next to the item when that
    leaves the package (more than one level up) or hits one of :data:`RESERVED`."""
    place = posixpath.normpath(posixpath.join(base, rel))
    if posixpath.isabs(place) or place.split("/").count("..") > 1 or place in {".", ".."}:
        return name
    if place.startswith("../") and place.removeprefix("../") in RESERVED:
        return name
    return place


def _static(path: Path, root: Path) -> list[tuple[Path, str, str]]:
    """(file, its path relative to ``path``'s directory, how) of what ``path`` names or
    imports; nothing for a file that does not parse."""
    try:
        tree = ast.parse(path.read_bytes())
    except (OSError, SyntaxError, ValueError):
        return []
    here = path.parent
    dirs = list(dict.fromkeys(Path(os.path.normpath(here / d)) for d in artifacts.SEARCH))
    out: list[tuple[Path, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for file in _named(node.value, dirs, root):
                out.append((file, _rel(file, here), "name"))
        elif isinstance(node, ast.Import | ast.ImportFrom):
            if isinstance(node, ast.ImportFrom) and (node.level or not node.module):
                continue
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
            for name in names:
                for file in _module(str(name).split(".")[0], here):
                    out.append((file, _rel(file, here), "import"))
    return out


def _named(text: str, dirs: list[Path], root: Path) -> list[Path]:
    """The run file or native project directory a string literal names, if any."""
    if not text or "\n" in text or len(text) > 1024 or text.strip() != text:
        return []
    candidates = [text]
    if not text.endswith(SOURCES) and SNAPSHOT_NAME.fullmatch(text):
        candidates.append(f"{text}.py")
    for name in candidates:
        named = Path(name)
        places = [named] if named.is_absolute() else [*(d / named for d in dirs), root / named]
        for p in places:
            p = Path(os.path.normpath(p))
            if not p.is_relative_to(root):
                continue
            if p.is_file() and name.endswith(SOURCES):
                return [p]
            if p.is_dir() and _is_project(p):
                return [p]
    return []


def _module(name: str, here: Path) -> list[Path]:
    """The files of the module or package ``name`` in ``here`` (not ``kernel_agent``)."""
    if name == "kernel_agent":
        return []
    if (module := here / f"{name}.py").is_file():
        return [module]
    if (package := here / name).is_dir():
        return sorted(p for p in package.rglob("*.py") if "__pycache__" not in p.parts)
    return []


def _recorded(
    path: Path, root: Path, by_sha: dict[str, list[dict[str, Any]]]
) -> list[tuple[Path, str, str]]:
    """The recorded lookups of a file with ``path``'s content that found a run file."""
    if not by_sha:
        return []
    try:
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return []
    out = []
    for rec in by_sha.get(sha, []):
        source = Path(os.path.normpath(str(rec.get("path") or "")))
        if not source.is_absolute() or not source.is_relative_to(root) or not source.exists():
            continue
        rel = str(rec.get("rel") or source.name)
        out.append((source, rel, "recorded"))
    return out


def _rel(file: Path, here: Path) -> str:
    return Path(os.path.relpath(file, here)).as_posix()


def _is_project(path: Path) -> bool:
    from kernel_agent.native.project import MANIFEST

    return (path / MANIFEST).is_file()


def _project_files(path: Path) -> list[str]:
    """The files of a native project directory (``project.collect``: no build products)."""
    from kernel_agent.native import project

    try:
        return sorted(project.collect(path))
    except project.ProjectError:
        return []
