"""Multi-file CUDA / C++ project candidates (issue #134).

A *project* is a directory with a build manifest (:data:`MANIFEST`), an entry module and
its sources: headers, several ``.cu`` / ``.cpp`` files, a build script::

    my_stage/
      kernel_project.toml     [project] name, entry, kind; [build] backend, sources, flags
      candidate.py            build(reference) (kind = "kernel") or apply(workload) ("transform")
      include/common.cuh
      csrc/binding.cpp  csrc/gemv.cu  csrc/attention.cu

The entry module gets the compiled project with ``project.load(__file__)``
(:func:`load`): the project's files are copied to the build cache, compiled once per
content digest and toolchain, and loaded (a torch extension as a Python module).

**Bundles.** Everything that takes a single-file candidate takes a project through its
*bundle* (:func:`pack`): one generated ``.py`` file that embeds every project file and the
project's digest. Importing it checks the digest, writes the files to the build cache and
re-exports the entry module (its ``build`` / ``apply`` / ``undo``), so the evaluator,
snapshots and their sha256, the duplicate check, the sweep, memcheck (#115), the
integration and the export treat a project like any candidate file, and the snapshot's
sha256 covers the whole project. ``evaluate_candidate`` / ``evaluate_e2e`` given a project
directory snapshot its bundle; the evaluator, memcheck and the patcher also load a project
directory directly (:func:`import_project`).

**Digest** (:func:`digest`): sha256 over the sorted (path, size, bytes) of every project file
(:func:`collect`: text files only; ``build/``, hidden and ``__pycache__`` directories and
build products are left out).

**Cache** (:func:`cache_root`, ``~/.cache/kernel-agent/native``, ``$KERNEL_AGENT_NATIVE_CACHE``
overrides it)::

    <digest[:24]>/src/              the project's files: rewritten from the bundle on every
                                    load, other files and ``__pycache__`` removed
    <digest[:24]>/build-<key[:16]>/ the build for one toolchain (:func:`build_key`: digest +
                                    torch, CUDA, arch list, nvcc flags, Python, C++ ABI) and
                                    its ``stamp.json``: the sha256 of every output

A build is reused only when its stamp matches the outputs on disk; a stale or damaged one is
rebuilt. Like torch's own extension cache, the cache is within reach of an agent's Bash
tool: the stamp catches accidents, not an attacker (the snapshot digest says what *should*
have been compiled; ``KERNEL_AGENT_NATIVE_REBUILD=1`` ignores the cache).

**Build backends** (``[build] backend``):

* ``torch_extension`` (default): ``torch.utils.cpp_extension.load`` over ``sources`` (globs)
  with ``include_dirs`` (the project root and kernel-agent's toolkit headers,
  :func:`toolkit_includes`: ``ka_launch.cuh`` for PDL / cooperative launches and the
  megakernel kit's ``ka_mk.cuh``, are always on the path), ``cflags``, ``cuda_cflags`` and
  ``ldflags``; loaded as a Python module
  (``PYBIND11_MODULE``) or, with ``load = "torch_ops"``, as ``TORCH_LIBRARY`` ops.
* ``command``: ``command`` (e.g. ``["bash", "{src}/build.sh"]``; ``{src}``, ``{build}``,
  ``{python}`` are substituted) runs in the build directory with ``KA_SRC_DIR``,
  ``KA_BUILD_DIR``, ``KA_PYTHON``, ``KA_TORCH_CMAKE_PREFIX`` and the toolchain's environment
  (CMake, make, nvcc directly; ``KA_TOOLKIT_INCLUDE``: the toolkit headers,
  ``KA_MK_INCLUDE``: the megakernel kit's) and must
  produce ``outputs`` (shared libraries, relative to
  the build directory), loaded per ``load``: ``torch_ops`` (``torch.ops.load_library``),
  ``python`` (extension modules) or ``none`` (``Built.libraries``, e.g. for ctypes).

:func:`prebuild` compiles a project in a subprocess that sees no GPU (the arch list comes
from ``TORCH_CUDA_ARCH_LIST``), outside the GPU lock: ``evaluate_candidate`` reports a
compiler error without taking the GPU, and the evaluation finds the build in the cache.

CLI::

    python -m kernel_agent.native.project check DIR     # manifest, files, digest
    python -m kernel_agent.native.project pack DIR [-o OUT.py]
    python -m kernel_agent.native.project build PATH    # a directory or a bundle
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import fcntl
import fnmatch
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST = "kernel_project.toml"
FORMAT = 1  # of the digest, the bundle and the cache layout
BUNDLE_VAR = "KA_PROJECT"
CACHE_ENV = "KERNEL_AGENT_NATIVE_CACHE"
REBUILD_ENV = "KERNEL_AGENT_NATIVE_REBUILD"
RESULT_MARKER = "@@KA_NATIVE@@"
STAMP = "stamp.json"
MAX_FILES = 200
MAX_BYTES = 4_000_000
DEFAULT_BUILD_TIMEOUT_S = 900.0
MAX_BUILD_TIMEOUT_S = 3600.0
KINDS = ("kernel", "transform")
BACKENDS = ("torch_extension", "command")
LOADERS = ("torch_ops", "python", "none")
SOURCE_SUFFIXES = (".cu", ".cpp", ".cc", ".cxx", ".c")
#: Directories and files of a project directory that are not part of the project.
IGNORED_DIRS = frozenset({"__pycache__", "build", "node_modules"})
IGNORED_SUFFIXES = frozenset(
    {".pyc", ".pyo", ".o", ".so", ".a", ".obj", ".ptx", ".cubin", ".fatbin", ".pt", ".pth"}
)
_NAME = re.compile(r"^[a-z][a-z0-9_]{1,47}$")
_PROJECT_KEYS = {"name", "entry", "kind", "description"}
_BUILD_KEYS = {
    "backend",
    "sources",
    "include_dirs",
    "cflags",
    "cuda_cflags",
    "ldflags",
    "command",
    "outputs",
    "load",
    "timeout_s",
}


class ProjectError(ValueError):
    """A project directory or bundle that cannot be packed, built or loaded."""


# ------------------------------------------------------------------ manifest


@dataclass(frozen=True)
class Manifest:
    """``kernel_project.toml``: ``[project]`` and ``[build]``."""

    name: str
    entry: str = "candidate.py"
    kind: str = "kernel"  # kernel: build(reference); transform: apply(workload)
    description: str = ""
    backend: str = "torch_extension"
    sources: tuple[str, ...] = ()  # expanded globs, relative to the project
    include_dirs: tuple[str, ...] = ()
    cflags: tuple[str, ...] = ()
    cuda_cflags: tuple[str, ...] = ()
    ldflags: tuple[str, ...] = ()
    command: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    load: str = "python"  # torch_ops / python / none (command builds default to torch_ops)
    timeout_s: float = DEFAULT_BUILD_TIMEOUT_S


def safe_name(text: str) -> str:
    """A project name from a directory name (lower case, ``[a-z0-9_]``)."""
    name = re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_")
    if not name or not name[0].isalpha():
        name = f"p_{name}".rstrip("_")
    return name[:48] if len(name) > 1 else "project"


def _rel_ok(path: str) -> bool:
    """A relative POSIX path inside the project (no ``..``, no absolute path)."""
    if not path or "\\" in path or path.startswith("/"):
        return False
    parts = PurePosixPath(path).parts
    return all(p not in ("", ".", "..") for p in parts)


def _old_std(build: Mapping[str, Any]) -> str:
    """A ``-std=c++NN`` (or ``--std``) flag of ``build`` older than C++20, or ""."""
    for key in ("cflags", "cuda_cflags"):
        for flag in build.get(key) or []:
            m = re.fullmatch(r"--?std=(?:c|gnu)\+\+(\d+)", str(flag))
            if m and int(m.group(1)) in (98, 3, 11, 14, 17):
                return f"build.{key} {flag}"
    return ""


def _strings(table: Mapping[str, Any], key: str, where: str) -> tuple[str, ...]:
    value = table.get(key, [])
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ProjectError(f"{where}.{key} must be a list of strings")
    return tuple(value)


def expand(patterns: Iterable[str], files: Iterable[str]) -> tuple[str, ...]:
    """Project files matching the glob ``patterns`` (``*`` crosses directories), in the
    order of the patterns, sorted within each; a pattern that matches nothing is an error."""
    names = sorted(files)
    out: list[str] = []
    for pattern in patterns:
        if not _rel_ok(pattern):
            raise ProjectError(f"source pattern {pattern!r} must be relative to the project")
        found = [n for n in names if fnmatch.fnmatchcase(n, pattern)]
        if not found:
            raise ProjectError(f"source pattern {pattern!r} matches no file of the project")
        out += [n for n in found if n not in out]
    return tuple(out)


def parse_manifest(
    text: str, files: Iterable[str] | None = None, *, default_name: str = "project"
) -> Manifest:
    """Parse and validate a manifest. ``files``: the project's file paths, against which the
    entry, the include directories and the source globs are checked (globs expanded)."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ProjectError(f"{MANIFEST}: {exc}") from None
    unknown = sorted(set(data) - {"project", "build"})
    if unknown:
        raise ProjectError(f"{MANIFEST}: unknown section(s) {', '.join(unknown)}")
    project, build = data.get("project", {}), data.get("build", {})
    if not isinstance(project, dict) or not isinstance(build, dict):
        raise ProjectError(f"{MANIFEST}: [project] and [build] must be tables")
    for table, keys, where in ((project, _PROJECT_KEYS, "project"), (build, _BUILD_KEYS, "build")):
        if extra := sorted(set(table) - keys):
            raise ProjectError(f"{MANIFEST}: unknown key(s) in [{where}]: {', '.join(extra)}")
    name = str(project.get("name") or default_name)
    if not _NAME.match(name):
        raise ProjectError(f"project.name {name!r} must match {_NAME.pattern}")
    entry = str(project.get("entry") or "candidate.py")
    if not _rel_ok(entry) or not entry.endswith(".py"):
        raise ProjectError(f"project.entry {entry!r} must be a relative path to a .py file")
    kind = str(project.get("kind") or "kernel")
    if kind not in KINDS:
        raise ProjectError(f"project.kind {kind!r} must be one of {', '.join(KINDS)}")
    backend = str(build.get("backend") or "torch_extension")
    if backend not in BACKENDS:
        raise ProjectError(f"build.backend {backend!r} must be one of {', '.join(BACKENDS)}")
    # a torch extension is a Python module unless it only registers TORCH_LIBRARY ops
    load = str(build.get("load") or ("python" if backend == "torch_extension" else "torch_ops"))
    if load not in LOADERS:
        raise ProjectError(f"build.load {load!r} must be one of {', '.join(LOADERS)}")
    try:
        timeout = float(build.get("timeout_s", DEFAULT_BUILD_TIMEOUT_S))
    except (TypeError, ValueError):
        raise ProjectError("build.timeout_s must be a number of seconds") from None
    if not 0 < timeout <= MAX_BUILD_TIMEOUT_S:
        raise ProjectError(f"build.timeout_s must be in (0, {MAX_BUILD_TIMEOUT_S:.0f}]")
    sources = _strings(build, "sources", "build")
    includes = _strings(build, "include_dirs", "build")
    command = _strings(build, "command", "build")
    outputs = _strings(build, "outputs", "build")
    for path in (*includes, *outputs):
        if not _rel_ok(path):
            raise ProjectError(f"{path!r} must be a relative path inside the project")
    if backend == "torch_extension" and (old := _old_std(build)):
        raise ProjectError(  # torch's headers need the standard torch compiles them with
            f"build: {old} is older than torch's C++ standard (-std=c++20): leave -std to "
            "torch's extension builder (g++ rejects ATen's headers under C++17)"
        )
    if backend == "torch_extension" and not sources:
        raise ProjectError("build.sources is empty: a torch_extension needs its .cu / .cpp files")
    if backend == "command" and (not command or not outputs):
        raise ProjectError("a command build needs build.command and build.outputs")
    if files is not None:
        names = set(files)
        if entry not in names:
            raise ProjectError(f"project.entry {entry} is not a file of the project")
        sources = expand(sources, names)
        if bad := [s for s in sources if not s.endswith(SOURCE_SUFFIXES)]:
            raise ProjectError(f"build.sources must be C / C++ / CUDA files, not {bad[:3]}")
        for d in includes:
            if not any(n.startswith(d.rstrip("/") + "/") for n in names):
                raise ProjectError(f"include directory {d!r} holds no file of the project")
    return Manifest(
        name=name,
        entry=entry,
        kind=kind,
        description=str(project.get("description") or ""),
        backend=backend,
        sources=sources,
        include_dirs=includes,
        cflags=_strings(build, "cflags", "build"),
        cuda_cflags=_strings(build, "cuda_cflags", "build"),
        ldflags=_strings(build, "ldflags", "build"),
        command=command,
        outputs=outputs,
        load=load,
        timeout_s=timeout,
    )


# ------------------------------------------------------------------ files + digest


def _ignored_dir(name: str) -> bool:
    return name in IGNORED_DIRS or name.startswith(".")


def collect(root: Path) -> dict[str, str]:
    """The files of a project directory: POSIX relative path → text (UTF-8).

    Hidden files and directories, ``build/``, ``__pycache__`` and build products
    (:data:`IGNORED_SUFFIXES`) are not part of it; symlinks and binary files are refused,
    as are more than :data:`MAX_FILES` files or :data:`MAX_BYTES` bytes."""
    root = Path(root)
    if not (root / MANIFEST).is_file():
        raise ProjectError(f"{root} is not a project: no {MANIFEST}")
    files: dict[str, str] = {}
    total = 0
    for here, dirs, names in os.walk(root, followlinks=False):
        base = Path(here)
        dirs[:] = sorted(d for d in dirs if not _ignored_dir(d))
        for d in dirs:
            if (base / d).is_symlink():
                raise ProjectError(f"{(base / d).relative_to(root)}: symlinks are not allowed")
        for name in sorted(names):
            path = base / name
            rel = path.relative_to(root).as_posix()
            if name.startswith(".") or path.suffix.lower() in IGNORED_SUFFIXES:
                continue
            if path.is_symlink():
                raise ProjectError(f"{rel}: symlinks are not allowed (copy the file in)")
            data = path.read_bytes()
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                raise ProjectError(
                    f"{rel} is not a UTF-8 text file: projects hold sources, headers and "
                    "scripts; generate binaries in the build"
                ) from None
            files[rel] = text
            total += len(data)
            if len(files) > MAX_FILES:
                raise ProjectError(f"more than {MAX_FILES} files in {root}")
            if total > MAX_BYTES:
                raise ProjectError(f"{root} holds more than {MAX_BYTES} bytes of sources")
    return files


def digest(files: Mapping[str, str]) -> str:
    """sha256 of a project's files (sorted paths, each with its size and bytes)."""
    h = hashlib.sha256(f"ka-native-project\0{FORMAT}\0".encode())
    for rel in sorted(files):
        data = files[rel].encode()
        h.update(f"{rel}\0{len(data)}\0".encode())
        h.update(data)
    return h.hexdigest()


@dataclass(frozen=True)
class Project:
    files: dict[str, str] = field(repr=False)
    manifest: Manifest
    digest: str

    @classmethod
    def from_files(cls, files: Mapping[str, str], *, default_name: str = "project") -> Project:
        if MANIFEST not in files:
            raise ProjectError(f"no {MANIFEST} in the project")
        if bad := [p for p in files if not _rel_ok(p)]:
            raise ProjectError(f"bad file path(s) in the project: {bad[:3]}")
        files = dict(files)
        manifest = parse_manifest(files[MANIFEST], files, default_name=default_name)
        return cls(files, manifest, digest(files))

    @classmethod
    def from_dir(cls, root: Path) -> Project:
        return cls.from_files(collect(root), default_name=safe_name(Path(root).resolve().name))

    def info(self) -> dict[str, Any]:
        m = self.manifest
        return {
            "name": m.name,
            "kind": m.kind,
            "entry": m.entry,
            "backend": m.backend,
            "digest": self.digest,
            "files": len(self.files),
            "bytes": sum(len(t.encode()) for t in self.files.values()),
            "sources": list(m.sources),
        }


def find_root(path: Path, levels: int = 6) -> Path:
    """The project directory holding ``path`` (a file or directory inside it)."""
    here = Path(path).resolve()
    here = here if here.is_dir() else here.parent
    for _ in range(levels):
        if (here / MANIFEST).is_file():
            return here
        if here.parent == here:
            break
        here = here.parent
    raise ProjectError(f"no {MANIFEST} in {path} or the directories above it")


def is_project(path: Path) -> bool:
    return Path(path).is_dir() and (Path(path) / MANIFEST).is_file()


# ------------------------------------------------------------------ bundles

_BUNDLE_HEAD = '''"""kernel-agent project bundle: {name} ({count} files, sha256 {digest}).

Generated by kernel_agent.native.project.pack from the project directory: edit the project,
not this file. Importing it checks the digest, writes the files to the build cache and
re-exports the project's entry module ({entry}), so it stands in for a single-file
candidate (build) or transform (apply).
"""

'''
_BUNDLE_TAIL = """
from kernel_agent.native.project import unpack as _ka_unpack  # noqa: E402

globals().update(_ka_unpack(KA_PROJECT, __name__))
"""


def _literal(text: str, indent: str) -> str:
    """``text`` as a parenthesised run of one JSON string literal per line (valid Python,
    deterministic, readable)."""
    lines = text.splitlines(keepends=True)
    if not lines:
        return '""'
    if len(lines) == 1:
        return json.dumps(lines[0], ensure_ascii=False)
    body = "".join(f"{indent}    {json.dumps(line, ensure_ascii=False)}\n" for line in lines)
    return f"(\n{body}{indent})"


def render(project: Project) -> str:
    """The bundle source of ``project`` (the same files give the same bytes)."""
    m = project.manifest
    out = [
        _BUNDLE_HEAD.format(
            name=m.name, count=len(project.files), digest=project.digest, entry=m.entry
        ),
        f"{BUNDLE_VAR} = {{\n",
        f'    "format": {FORMAT},\n',
        f'    "name": {json.dumps(m.name)},\n',
        f'    "entry": {json.dumps(m.entry)},\n',
        f'    "digest": {json.dumps(project.digest)},\n',
        '    "files": {\n',
    ]
    for rel in sorted(project.files):
        out.append(f"        {json.dumps(rel)}: {_literal(project.files[rel], '        ')},\n")
    out += ["    },\n", "}\n", _BUNDLE_TAIL]
    return "".join(out)


def pack(root: Path) -> tuple[str, Project]:
    """The bundle source of the project at ``root`` and the project."""
    project = Project.from_dir(root)
    return render(project), project


def bundle_name(root: Path) -> str:
    """File name of a project's bundle: ``<project name>.py``."""
    return f"{Project.from_dir(root).manifest.name}.py"


def write_bundle(root: Path, dest_dir: Path) -> Path:
    """Write the bundle of the project at ``root`` to ``dest_dir/<name>.py``."""
    text, project = pack(root)
    dest = Path(dest_dir) / f"{project.manifest.name}.py"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)
    return dest


def source_of(path: Path) -> str:
    """The candidate source of ``path``: a file's text, a project directory's bundle."""
    path = Path(path)
    return pack(path)[0] if path.is_dir() else path.read_text(errors="replace")


def read_bundle(path: Path) -> dict[str, Any] | None:
    """The ``KA_PROJECT`` payload of a bundle file, read without importing it (None: not a
    bundle)."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    if f"\n{BUNDLE_VAR} = ".encode() not in data:
        return None
    try:
        tree = ast.parse(data)
    except (SyntaxError, ValueError):
        return None
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == BUNDLE_VAR
        ):
            try:
                payload = ast.literal_eval(node.value)
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                return None
            return payload if isinstance(payload, dict) else None
    return None


def from_payload(payload: Mapping[str, Any]) -> Project:
    """The project of a bundle payload, its digest checked (:class:`ProjectError`)."""
    if payload.get("format") != FORMAT:
        raise ProjectError(f"bundle format {payload.get('format')!r}, this is {FORMAT}")
    files = payload.get("files")
    if not isinstance(files, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in files.items()
    ):
        raise ProjectError("bundle without its files")
    found = digest(files)
    if found != payload.get("digest"):
        raise ProjectError(
            f"bundle digest mismatch (files {found[:12]}…, recorded "
            f"{str(payload.get('digest'))[:12]}…): the bundle was edited by hand; edit the "
            "project directory and pack it again"
        )
    project = Project.from_files(files, default_name=str(payload.get("name") or "project"))
    if project.manifest.name != payload.get("name"):
        raise ProjectError("bundle name differs from its manifest's")
    return project


def code_dirs(candidate: Path) -> list[Path]:
    """Directories whose code is the candidate's (integrity checks: threads running it):
    a project directory itself, a bundle's source directory in the cache."""
    candidate = Path(candidate)
    if candidate.is_dir():  # its entry runs from the cache (import_project)
        try:
            return [candidate.resolve(), src_dir(Project.from_dir(candidate).digest)]
        except ProjectError:
            return [candidate.resolve()]
    payload = read_bundle(candidate)
    if payload is None or not isinstance(payload.get("digest"), str):
        return []
    return [src_dir(str(payload["digest"]))]


# ------------------------------------------------------------------ cache


def cache_root() -> Path:
    if env := os.environ.get(CACHE_ENV):
        return Path(env).expanduser()
    from kernel_agent.toolchain import CACHE_DIR

    return CACHE_DIR / "native"


def cache_dir(project_digest: str) -> Path:
    return cache_root() / project_digest[:24]


def src_dir(project_digest: str) -> Path:
    return cache_dir(project_digest) / "src"


def toolkit_include() -> Path:
    """kernel-agent's C++ toolkit headers (``ka_launch.cuh``: PDL and cooperative launches;
    the directory of ``kernel_agent.concurrency.include_dir()``), on every build's path."""
    return Path(__file__).resolve().parent.parent / "include"


def toolkit_includes() -> list[Path]:
    """Every toolkit include directory of a build: :func:`toolkit_include` and the
    megakernel kit's (``ka_mk.cuh``, :mod:`kernel_agent.native.megakernel`, issue #225)."""
    from kernel_agent.native import megakernel

    return [toolkit_include(), megakernel.include_dir()]


def _toolkit_digest() -> str:
    """sha256 of the toolkit headers: a changed header rebuilds every project."""
    h = hashlib.sha256()
    for root in toolkit_includes():
        for path in sorted(root.glob("*")) if root.is_dir() else []:
            if path.is_file():
                h.update(path.name.encode() + b"\0" + path.read_bytes())
    return h.hexdigest()[:16]


def fingerprint() -> dict[str, str]:
    """What a build depends on besides the sources (call after ``toolchain.setup()``)."""
    import torch

    abi = getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None)
    return {
        "toolkit": _toolkit_digest(),
        "format": str(FORMAT),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "arch": os.environ.get("TORCH_CUDA_ARCH_LIST", ""),
        "nvcc_flags": os.environ.get("NVCC_APPEND_FLAGS", ""),
        "cuda_home": os.environ.get("CUDA_HOME", ""),
        "cxx11_abi": str(abi),
    }


def build_key(project_digest: str, fp: Mapping[str, str]) -> str:
    """The cache key of one build: the project digest and the toolchain fingerprint."""
    blob = json.dumps({"digest": project_digest, "toolchain": dict(fp)}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    """An exclusive ``flock`` on ``path`` (processes building the same project wait)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


#: Projects materialised by this process: source directory → project.
_MATERIALISED: dict[Path, Project] = {}


def materialise(project: Project) -> Path:
    """Write the project's files to its source directory in the cache (only what differs)
    and remove everything else there (stray files, ``__pycache__``: a stale ``.pyc`` must
    not stand in for the entry). Returns the directory."""
    src = src_dir(project.digest)
    with _locked(cache_dir(project.digest) / ".lock"):
        for rel, text in project.files.items():
            path = src / rel
            data = text.encode()
            try:
                same = path.read_bytes() == data and not path.is_symlink()
            except OSError:
                same = False
            if not same:
                if path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                _write(path, data)
        keep = set(project.files)
        for here, dirs, names in os.walk(src, topdown=False):
            base = Path(here)
            for name in names:
                path = base / name
                if path.relative_to(src).as_posix() not in keep:
                    path.unlink(missing_ok=True)
            for d in dirs:
                path = base / d
                if path.is_symlink():
                    path.unlink()
                elif d == "__pycache__" or not any(path.iterdir()):
                    shutil.rmtree(path, ignore_errors=True)
    _MATERIALISED[src.resolve()] = project
    return src


# ------------------------------------------------------------------ build


@dataclass
class Built:
    """A compiled project: the Python extension module (``torch_extension``, ``python``
    loads), the shared libraries, where they are and whether the cache had them."""

    name: str
    digest: str
    key: str
    build_dir: Path
    module: Any = None
    libraries: list[Path] = field(default_factory=list)
    cached: bool = False
    seconds: float = 0.0

    def __getattr__(self, attr: str) -> Any:  # ext.my_op(...) on the extension module
        module = self.__dict__.get("module")
        if module is None or attr.startswith("__"):
            raise AttributeError(attr)
        return getattr(module, attr)


#: Builds loaded by this process, per build key (an extension module loads once).
_BUILT: dict[str, Built] = {}
_toolchain_ready = False


def _setup_toolchain() -> None:
    global _toolchain_ready
    if not _toolchain_ready:
        from kernel_agent import toolchain

        toolchain.setup()
        _toolchain_ready = True


def _sha256(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def stamp_ok(build_dir: Path, key: str) -> dict[str, Any] | None:
    """The build's stamp when it is for ``key`` and every output still has its sha256."""
    try:
        stamp = json.loads((build_dir / STAMP).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(stamp, dict) or stamp.get("key") != key:
        return None
    outputs = stamp.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        return None
    for rel, sha in outputs.items():
        path = build_dir / rel
        try:
            if not _rel_ok(rel) or _sha256(path) != sha:
                return None
        except OSError:
            return None
    return stamp


def _ext_name(manifest: Manifest, key: str) -> str:
    return f"ka_native_{manifest.name}_{key[:12]}"


def _import_extension(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ProjectError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def _compile_extension(project: Project, src: Path, out: Path, name: str) -> tuple[list[str], Any]:
    """Compile a ``torch_extension`` project into ``out``; returns its library and the
    module torch loaded (None for ``load = "torch_ops"``: the ops are registered)."""
    from torch.utils.cpp_extension import load as load_extension

    m = project.manifest
    module = load_extension(
        name=name,
        sources=[str(src / s) for s in m.sources],
        extra_include_paths=[
            str(src),
            *(str(src / d) for d in m.include_dirs),
            *(str(d) for d in toolkit_includes()),
        ],
        extra_cflags=list(m.cflags),
        extra_cuda_cflags=list(m.cuda_cflags),
        extra_ldflags=list(m.ldflags),
        build_directory=str(out),
        verbose=False,
        is_python_module=m.load == "python",
    )
    libs = sorted(p.name for p in out.glob(f"{name}*.so"))
    if not libs:
        raise ProjectError(f"the build left no {name}*.so in {out}")
    return libs, module if m.load == "python" else None


def _run_command(project: Project, src: Path, out: Path) -> list[str]:
    import torch

    m = project.manifest
    subst = {"src": str(src), "build": str(out), "python": sys.executable}
    try:
        command = [arg.format(**subst) for arg in m.command]
    except (KeyError, IndexError, ValueError) as exc:
        raise ProjectError(f"build.command: bad placeholder {exc}") from None
    env = dict(os.environ)
    env.update(
        KA_SRC_DIR=str(src),
        KA_BUILD_DIR=str(out),
        KA_PYTHON=sys.executable,
        KA_TORCH_CMAKE_PREFIX=str(getattr(torch.utils, "cmake_prefix_path", "")),
        KA_TOOLKIT_INCLUDE=str(toolkit_include()),
        KA_MK_INCLUDE=str(toolkit_includes()[1]),
        KA_PROJECT_NAME=m.name,
        KA_PROJECT_DIGEST=project.digest,
    )
    try:
        proc = subprocess.run(
            command, cwd=out, env=env, capture_output=True, text=True, timeout=m.timeout_s
        )
    except subprocess.TimeoutExpired:
        raise ProjectError(f"build command exceeded {m.timeout_s:.0f}s") from None
    except OSError as exc:
        raise ProjectError(f"build command {command[0]!r} cannot run: {exc}") from None
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr)[-4000:]
        raise ProjectError(f"build command failed (exit {proc.returncode}):\n{tail}")
    missing = [o for o in m.outputs if not (out / o).is_file()]
    if missing:
        raise ProjectError(f"the build command did not produce {missing}")
    return list(m.outputs)


def _load_outputs(project: Project, built: Built, outputs: list[str], fresh: bool) -> None:
    """Load what a build produced (``fresh``: a torch extension torch just compiled and
    loaded; ``built.module`` is already set for a Python one)."""
    m = project.manifest
    paths = [built.build_dir / o for o in outputs]
    built.libraries = paths
    if m.backend == "torch_extension" and fresh:
        return
    if m.backend == "torch_extension" and m.load == "python":
        built.module = _import_extension(_ext_name(m, built.key), paths[0])
    elif m.load == "torch_ops":
        import torch

        for path in paths:
            torch.ops.load_library(str(path))
    elif m.load == "python":
        modules = [_import_extension(p.name.split(".")[0], p) for p in paths]
        built.module = modules[0] if modules else None


def build(project: Project, src: Path | None = None, *, setup: bool = True) -> Built:
    """Compile ``project`` (or take it from the cache) and load it. ``src``: its
    materialised source directory (default: materialise it now). ``setup``: discover the
    toolchain first (CUDA_HOME, nvcc flags, arch list: they are part of the build key)."""
    if setup:
        _setup_toolchain()
    src = materialise(project) if src is None else src
    key = build_key(project.digest, fingerprint())
    if key in _BUILT:
        return _BUILT[key]
    out = cache_dir(project.digest) / f"build-{key[:16]}"
    start = time.perf_counter()
    with _locked(cache_dir(project.digest) / f".build-{key[:16]}.lock"):
        stamp = None if os.environ.get(REBUILD_ENV) == "1" else stamp_ok(out, key)
        cached = stamp is not None
        module = None
        if stamp is not None:
            outputs = list(stamp["outputs"])
        else:
            if out.exists():
                shutil.rmtree(out)
            out.mkdir(parents=True)
            if project.manifest.backend == "torch_extension":
                name = _ext_name(project.manifest, key)
                outputs, module = _compile_extension(project, src, out, name)
            else:
                outputs = _run_command(project, src, out)
            stamp = {
                "key": key,
                "digest": project.digest,
                "name": project.manifest.name,
                "toolchain": fingerprint(),
                "outputs": {o: _sha256(out / o) for o in outputs},
                "seconds": round(time.perf_counter() - start, 1),
                "built": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            (out / STAMP).write_text(json.dumps(stamp, indent=1))
        built = Built(project.manifest.name, project.digest, key, out, module, cached=cached)
        _load_outputs(project, built, outputs, fresh=not cached)
    built.seconds = round(time.perf_counter() - start, 2)
    _BUILT[key] = built
    return built


def load(path: str | Path) -> Built:
    """The compiled project that ``path`` (usually the entry module's ``__file__``) belongs
    to: materialised, built once per content digest and toolchain, loaded."""
    root = find_root(Path(path))
    project = _MATERIALISED.get(root) or Project.from_dir(root)
    src = root if _MATERIALISED.get(root) is project else materialise(project)
    return build(project, src)


# ------------------------------------------------------------------ import


def import_entry(project: Project, src: Path) -> Any:
    """Import the project's entry module from its materialised directory ``src`` (once per
    digest and process)."""
    name = f"ka_native_entry_{project.manifest.name}_{project.digest[:10]}"
    if name in sys.modules:
        return sys.modules[name]
    path = src / project.manifest.entry
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ProjectError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__ka_project__"] = project.info()
    sys.modules[name] = module
    if str(src) not in sys.path:  # helper modules next to the entry
        sys.path.insert(0, str(src))
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _exports(module: Any) -> dict[str, Any]:
    out = {k: v for k, v in vars(module).items() if not k.startswith("__")}
    out["__ka_project__"] = vars(module).get("__ka_project__")
    return out


def unpack(payload: Mapping[str, Any], module_name: str = "") -> dict[str, Any]:
    """What a bundle's module namespace gets: the entry module's names (``build``,
    ``apply``, ``undo``, ...) after the digest check and materialisation."""
    project = from_payload(payload)
    return _exports(import_entry(project, materialise(project)))


def import_project(root: Path) -> Any:
    """The entry module of the project directory ``root`` (materialised in the cache
    first, so it runs from the files its digest names)."""
    project = Project.from_dir(root)
    return import_entry(project, materialise(project))


# ------------------------------------------------------------------ prebuild


def _project_of(path: Path) -> Project:
    path = Path(path)
    if path.is_dir():
        return Project.from_dir(path)
    payload = read_bundle(path)
    if payload is None:
        raise ProjectError(f"{path} is neither a project directory nor a bundle")
    return from_payload(payload)


def prebuild(path: Path, *, timeout: float | None = None) -> dict[str, Any]:
    """Build the project of ``path`` (a directory or a bundle) in a subprocess that sees no
    GPU, outside the GPU lock. ``status``: ``ok`` (``cached``, ``seconds``), ``build_error``
    (``error``: the compiler's output), ``timeout`` or ``skipped`` (no
    ``TORCH_CUDA_ARCH_LIST`` to compile for without a GPU: the evaluator builds it)."""
    try:
        project = _project_of(path)
    except ProjectError as exc:
        return {"status": "build_error", "error": str(exc)}
    if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return {"status": "skipped", "reason": "TORCH_CUDA_ARCH_LIST is not set"}
    limit = min(timeout or project.manifest.timeout_s, MAX_BUILD_TIMEOUT_S)
    cmd = [sys.executable, "-m", "kernel_agent.native.project", "build", str(path), "--json"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=limit, env=env)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"the build exceeded {limit:.0f}s"}
    seconds = round(time.perf_counter() - start, 1)
    for line in proc.stdout.splitlines()[::-1]:
        if line.startswith(RESULT_MARKER):
            return {**json.loads(line[len(RESULT_MARKER) :]), "seconds": seconds}
    tail = (proc.stderr or proc.stdout)[-4000:]
    return {"status": "build_error", "error": tail, "seconds": seconds}


# ------------------------------------------------------------------ CLI


def _cmd_check(ns: argparse.Namespace) -> int:
    project = _project_of(ns.path)
    print(json.dumps(project.info(), indent=1))
    return 0


def _cmd_pack(ns: argparse.Namespace) -> int:
    text, project = pack(ns.path)
    if ns.output:
        Path(ns.output).write_text(text)
        print(f"{ns.output}: {project.manifest.name}, sha256 {project.digest}")
    else:
        sys.stdout.write(text)
    return 0


def _cmd_build(ns: argparse.Namespace) -> int:
    try:
        project = _project_of(ns.path)
        built = build(project)
        result = {
            "status": "ok",
            "cached": built.cached,
            "key": built.key,
            "build_dir": str(built.build_dir),
            "outputs": [str(p) for p in built.libraries],
            "build_seconds": built.seconds,
        }
    except Exception as exc:
        result = {"status": "build_error", "error": f"{type(exc).__name__}: {exc}"[-6000:]}
    if ns.json:
        print(RESULT_MARKER + json.dumps(result), flush=True)
    else:
        print(json.dumps(result, indent=1))
    return 0 if result["status"] == "ok" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check", help="validate a project (directory or bundle)")
    p.add_argument("path", type=Path)
    p.set_defaults(func=_cmd_check)
    p = sub.add_parser("pack", help="write the bundle of a project directory")
    p.add_argument("path", type=Path)
    p.add_argument("-o", "--output", type=Path)
    p.set_defaults(func=_cmd_pack)
    p = sub.add_parser("build", help="compile a project (directory or bundle) into the cache")
    p.add_argument("path", type=Path)
    p.add_argument("--json", action="store_true", help="one result line for prebuild()")
    p.set_defaults(func=_cmd_build)
    ns = parser.parse_args(argv)
    try:
        return int(ns.func(ns))
    except ProjectError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
