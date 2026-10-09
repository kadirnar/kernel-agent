"""Export the accepted optimisations as a small self-contained package.

``optimized/`` holds the accepted kernels (``kernels/<target>.py``), region rewrites
(``rewrites/``) and transforms (``transforms/``) in the integrated order
(``manifest.json``), ``apply.py`` and ``phases.py``, and every run file they load
(:mod:`kernel_agent.integrate.deps`: a history snapshot by name, a native project's bundle,
a helper module), placed where the item's own lookup finds it; ``manifest.json`` lists them
per item (``needs``). Kernels that call a library (the library scout's, ``KA_LIBRARY``) add
``requirements.txt`` with its exact version and ``manifest.json`` → ``libraries`` with its
licence.

**Self-test** (:func:`check_export`, issue #171): a copy of the package, in a new temporary
directory, is imported and applied to the workload in a fresh worker process
(``worker export_check``) whose working directory is that copy's directory and which cannot
read the run directory (:func:`hidden`); one run's output must pass the workload's quality
check against the baseline output, and each kernel must replace as many module instances
as in the integration. A failure names the files the package lacks.

CLI (re-export a run from its ``integration.json``, then the self-test)::

    python -m kernel_agent.integrate.export RUN_DIR [--no-check]
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from kernel_agent import artifacts, phases, region
from kernel_agent.integrate import deps
from kernel_agent.truth import TamperError, alarm, read_verified, sha256_bytes
from kernel_agent.workspace import RunDir, append_jsonl, read_json, read_jsonl, write_json

#: The self-test's result, in ``optimized/``; every result also goes to ``logs/``.
CHECK_FILE = "export_check.json"
CHECK_LOG = "export_checks.jsonl"
CHECK_TIMEOUT_S = 3600.0

APPLY_TEMPLATE = '''"""Optimised kernels for {repo_id}, produced by kernel-agent.

Usage::

    import sys; sys.path.insert(0, "{root}")
    from apply import apply_kernels, apply_transforms
    apply_kernels(model)            # model: the nn.Module(s) loaded as in the benchmark
    apply_transforms(model)         # model-level transforms, in the integrated order

Model-level transforms (``transforms/*.py``) receive the kernel-agent workload
object. Most only use ``workload.model``; ``apply_transforms(model)`` passes a
stand-in holding ``model`` (pass ``workload=`` for transforms that need more).

Files the kernels and transforms load (a history snapshot by name, a native project's
bundle, a helper module) are next to them, where their own lookup finds them
(``manifest.json``: ``needs``); kernel-agent checked that the package applies outside the
run directory (``export_check.json``).

Kernels compiled at load time (CUDA C++ via ``load_inline``, TileLang) need the
compiler environment kernel-agent found (nvcc from pip wheels, ninja on PATH,
nvcc flags for new host compilers): with kernel-agent importable this module
sets it up; otherwise put a CUDA toolkit and ninja on PATH yourself.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

from torch import nn

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "manifest.json").read_text())


def _toolchain() -> None:
    try:
        from kernel_agent import toolchain
    except ImportError:
        return
    toolchain.setup()


_toolchain()


def _load(path: Path, name: str | None = None):
    name = name or f"ka_opt_{{path.stem}}"
    if str(path.parent) not in sys.path:  # as in the run: modules next to it import by name
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _rewrite_parents(roots, entry) -> int:
    """Region target: ``rewrite()`` every instance of its parent class first, which adds
    the ``Region_<id>`` modules that the entry's kernel then replaces."""
    rewrite = entry["rewrite"]
    module = _load(HERE / rewrite["file"], f"ka_region_{{entry['target']}}")
    regex = rewrite.get("qualname_regex")
    n = 0
    for root in roots:
        for name, child in list(root.named_modules()):
            if not name or type(child).__name__ != rewrite["parent_class"]:
                continue
            names = [name, *(f"{{r}}.{{name}}" for r in MANIFEST.get("roots", []))]
            if regex and not any(re.search(regex, q) for q in names):
                continue
            new = module.rewrite(child)
            parent_name, _, attr = name.rpartition(".")
            parent = root.get_submodule(parent_name) if parent_name else root
            if isinstance(parent, (nn.ModuleList, nn.Sequential)) and attr.isdigit():
                parent[int(attr)] = new
            else:
                setattr(parent, attr, new)
            n += 1
    return n


def apply_kernels(*roots: nn.Module) -> dict[str, int]:
    """Replace every matching module instance in ``roots``; returns counts.

    An entry with a ``phase`` keeps the instance and sends only that phase's
    calls to the kernel (``phases.py``), as the integration did. Region targets'
    rewrites (``rewrites/``) come first."""
    counts: dict[str, int] = {{}}
    for entry in MANIFEST["kernels"]:
        if entry.get("rewrite"):
            counts[f"{{entry['target']}}:rewrite"] = _rewrite_parents(roots, entry)
    for entry in MANIFEST["kernels"]:
        module = _load(HERE / entry["file"])
        regex = entry.get("qualname_regex")
        n = 0
        for root in roots:
            for name, child in list(root.named_modules()):
                if not name or type(child).__name__ != entry["module_class"]:
                    continue
                # qualnames in the run start with the workload's root name, e.g. "model."
                names = [name, *(f"{{r}}.{{name}}" for r in MANIFEST.get("roots", []))]
                if regex and not any(re.search(regex, q) for q in names):
                    continue
                new = module.build(child)
                if new is None or new is child:
                    continue
                # Entrypoints other than forward the model calls (e.g. forward_step).
                missing = [
                    m
                    for m in entry.get("methods", [])
                    if callable(getattr(child, m, None)) and not callable(getattr(new, m, None))
                ]
                if missing:
                    raise RuntimeError(
                        f"{{entry['target']}}: the replacement for {{name}} lacks {{missing}}"
                    )
                if entry.get("phase"):
                    phases = _load(HERE / "phases.py")
                    phases.route(child, new, entry["phase"], entry.get("routed") or ["forward"])
                    n += 1
                    continue
                parent_name, _, attr = name.rpartition(".")
                parent = root.get_submodule(parent_name) if parent_name else root
                if isinstance(parent, (nn.ModuleList, nn.Sequential)) and attr.isdigit():
                    parent[int(attr)] = new
                else:
                    setattr(parent, attr, new)
                n += 1
        counts[entry["target"]] = n
    return counts


def apply_transforms(model: nn.Module, workload=None) -> list[str]:
    """Apply the model-level transforms in the order the integration accepted them."""
    from types import SimpleNamespace

    target = workload if workload is not None else SimpleNamespace(model=model)
    done = []
    for entry in MANIFEST.get("transforms", []):
        _load(HERE / entry["file"]).apply(target)
        done.append(entry["file"])
    return done
'''


def _copy(run: RunDir, src: Path, dst: Path, digests: dict[str, str | None] | None) -> bytes:
    """Copy ``src`` after checking it against its recorded sha256 (when ``digests`` has one)."""
    sha256 = (digests or {}).get(str(src))
    try:
        data = read_verified(src, sha256)
    except TamperError as exc:
        rel = src.relative_to(run.root).as_posix() if src.is_relative_to(run.root) else str(src)
        alarm(run, rel, str(exc))
        raise
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    return data


def export_optimized(
    run: RunDir,
    accepted: list[tuple[str, str, float]],
    *,
    digests: dict[str, str | None] | None = None,
) -> Path:
    """``optimized/``: the accepted snapshots, the run files they need + ``apply.py``.
    ``digests`` (snapshot path → sha256 of the evaluated file) makes a snapshot changed
    since then refuse the export."""
    out = run.optimized_dir
    if out.exists():
        shutil.rmtree(out)
    (out / "kernels").mkdir(parents=True)
    (out / "transforms").mkdir()
    card = run.load()["card"]
    manifest: dict[str, list[Any]] = {"kernels": [], "transforms": []}
    records = artifacts.recorded(run.root)  # the lookups the run's evaluations recorded
    conflicts: list[str] = []

    def need(src: Path, dst: Path) -> dict[str, Any]:
        """Copy what the item ``src`` (exported as ``dst``) needs; its manifest field."""
        entries = []
        for n in deps.needs(src, run.root, records):
            target = Path(os.path.normpath(dst.parent / n.place))
            rel = target.relative_to(out).as_posix()
            source = n.source.relative_to(run.root).as_posix()
            if target.exists():  # another item's file or need, or an accepted item itself
                sha = sha256_bytes(target.read_bytes())
                if sha != (theirs := sha256_bytes(n.source.read_bytes())):
                    conflicts.append(f"{rel}: {source} (sha256 {theirs[:12]}…) differs")
                    continue
            else:
                sha = sha256_bytes(_copy(run, n.source, target, digests))
            entries.append({"file": rel, "source": source, "sha256": sha, "by": n.by, "how": n.how})
        return {"needs": entries} if entries else {}

    for kind, arg, _ in accepted:
        if kind == "kernel":
            target_id, _, path = arg.partition("=")
            spec = read_json(run.target(target_id) / "spec.json")
            dst = out / "kernels" / f"{target_id}.py"
            _copy(run, Path(path), dst, digests)
            methods = spec.get("capture", {}).get("method_instances", {})
            scope = {k: spec[k] for k in ("phase", "qualname_regex") if spec.get(k)}
            if "phase" in scope:
                scope["routed"] = list(methods)  # entrypoints whose phase calls go to the kernel
            if (rewrite := region.rewrite_of(run, target_id, spec)) is not None:
                (out / "rewrites").mkdir(exist_ok=True)
                _copy(run, rewrite.path, out / "rewrites" / f"{target_id}.py", digests)
                scope["rewrite"] = {"file": f"rewrites/{target_id}.py"}
                scope["rewrite"]["parent_class"] = rewrite.parent_class
                if regex := scope.pop("qualname_regex", None):  # it selects parent instances
                    scope["rewrite"]["qualname_regex"] = regex
                scope["rewrite"].update(need(rewrite.path, out / "rewrites" / f"{target_id}.py"))
            manifest["kernels"].append(
                {
                    "target": target_id,
                    "module_class": spec["module_class"],
                    "file": f"kernels/{dst.name}",
                    "methods": [m for m in methods if m != "forward"],
                    **scope,
                    **need(Path(path), dst),
                }
            )
        else:
            src = Path(arg)
            dst = out / "transforms" / src.name
            _copy(run, src, dst, digests)
            manifest["transforms"].append({"file": f"transforms/{dst.name}", **need(src, dst)})
    # Root names prefix the qualnames that `qualname_regex` was written against.
    profile = read_json(run.profile_dir / "profile.json", {}) or {}
    manifest["roots"] = sorted({c["root"] for c in profile.get("classes", []) if c.get("root")})
    if conflicts:  # two items need different files at one place: the self-test fails
        manifest["conflicts"] = conflicts
    _requirements(out, manifest)
    shutil.copy2(Path(phases.__file__), out / "phases.py")  # phase routing for apply.py
    write_json(out / "manifest.json", manifest)
    (out / "apply.py").write_text(APPLY_TEMPLATE.format(repo_id=card["repo_id"], root=out))
    return out


def _requirements(out: Path, manifest: dict[str, list[Any]]) -> None:
    """``requirements.txt`` with the exact versions (and the licences, in ``manifest.json``
    → ``libraries``) of the libraries the exported files call: library scout kernels declare
    theirs (``KA_LIBRARY``, ``KA_LICENCE``; libscout/, issue #227). None without any."""
    from kernel_agent.libscout.scout import requirements, requirements_text

    sources = {p.relative_to(out).as_posix(): p.read_text() for p in sorted(out.rglob("*.py"))}
    if entries := requirements(sources):
        (out / "requirements.txt").write_text(requirements_text(entries))
        manifest["libraries"] = entries


# ------------------------------------------------------------------ self-test (issue #171)


def package_digest(out: Path) -> str:
    """sha256 over the package's files (path and bytes; the self-test's result and
    ``__pycache__`` left out)."""
    h = hashlib.sha256()
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != CHECK_FILE and "__pycache__" not in p.parts:
            h.update(p.relative_to(out).as_posix().encode() + b"\0")
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def check_export(
    run: RunDir,
    worker: Callable[..., dict[str, Any]],
    args: Sequence[str] = (),
    *,
    expected: dict[str, Any] | None = None,
    timeout: float = CHECK_TIMEOUT_S,
) -> dict[str, Any]:
    """The self-test of ``optimized/`` (see the module docstring): ``worker`` (``call_worker``)
    runs ``export_check`` on a copy of the package with ``cwd`` set to the copy's directory;
    ``args``: its ``--verify`` / ``--quality`` flags (``Truth.worker_args``); ``expected``: the
    integration's final ``patches`` (the module instances each kernel replaced there). A
    passing result of the same package and flags is reused. The result (``passed``,
    ``reason``, ``missing``: the files the package lacks) goes to ``optimized/`` and ``logs/``."""
    out = run.optimized_dir
    digest = package_digest(out)
    key = sha256_bytes(json.dumps([digest, list(args)]).encode())
    log = run.root / "logs" / CHECK_LOG
    previous = [r for r in read_jsonl(log) if r.get("key") == key and r.get("passed")]
    if previous:
        record = {**previous[-1], "reused": True}
        write_json(out / CHECK_FILE, record)
        return record
    manifest = read_json(out / "manifest.json", {}) or {}
    start = time.perf_counter()
    result: dict[str, Any]
    if conflicts := manifest.get("conflicts"):
        reason = "two items need different files at one place: " + "; ".join(conflicts)
        result = {"status": "conflict", "passed": False, "reason": reason}
    else:
        try:
            with tempfile.TemporaryDirectory(prefix="ka-export-check-") as tmp:
                package = Path(tmp) / out.name
                shutil.copytree(out, package, ignore=shutil.ignore_patterns("__pycache__"))
                result = worker(
                    run, "export_check", "--package", str(package), *args, cwd=tmp, timeout=timeout
                )
        except Exception as exc:  # a package that cannot be copied, a worker that cannot start
            result = {"status": "error", "passed": False, "error": f"{type(exc).__name__}: {exc}"}
        result = _with_counts(result, manifest, expected)
    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "key": key,
        "package_sha256": digest,
        "passed": bool(result.get("passed")),
        "status": result.get("status"),
        "reason": describe(result),
        "missing": result.get("missing") or [],
        "check_s": round(time.perf_counter() - start, 1),
        **{k: result[k] for k in ("counts", "transforms", "median_ms", "metrics") if k in result},
        **({"error": str(result["error"])[-3000:]} if result.get("error") else {}),
    }
    write_json(out / CHECK_FILE, record)
    append_jsonl(log, record)
    return record


def _with_counts(
    result: dict[str, Any], manifest: dict[str, Any], expected: dict[str, Any] | None
) -> dict[str, Any]:
    """``result``, failed when a kernel replaced another number of module instances than in
    the integration (``expected``: its ``patches``) or a transform of the manifest was not
    applied."""
    if not result.get("passed"):
        return result
    counts = result.get("counts") or {}
    wrong = [
        f"kernels/{t}.py replaced {counts[t]} module instances, {n} in the integration"
        for t, n in ((expected or {}).get("replaced") or {}).items()
        if t in counts and counts[t] != n
    ]
    files = [e["file"] for e in manifest.get("transforms") or []]
    if (applied := result.get("transforms")) is not None and list(applied) != files:
        wrong.append(f"applied {applied}, the manifest lists {files}")
    if not wrong:
        return result
    return {**result, "passed": False, "status": "mismatch", "reason": "; ".join(wrong)}


def describe(result: dict[str, Any]) -> str:
    """One line on a self-test result, the missing files first."""
    if result.get("passed"):
        return "applied outside the run directory; its output passes the quality check"
    reason = str(result.get("reason") or "")
    if missing := result.get("missing"):
        reason = f"not self-contained, missing {', '.join(missing)}" + (
            f" ({reason})" if reason else ""
        )
    if not reason:
        error = str(result.get("error") or "").strip().splitlines()
        reason = error[-1] if error else "no result"
    return f"{result.get('status') or 'failed'}: {reason}"


# ------------------------------------------------------------------ the worker's side


class _Hidden:
    """The directories whose files this process may not open (:func:`hidden`). Audit hooks
    cannot be removed, so the one hook consults this list."""

    hook = False
    active: list[tuple[str, frozenset[str], list[str]]] = []  # (root, allowed, refused)


def _audit(event: str, args: tuple[Any, ...]) -> None:
    if event != "open" or not _Hidden.active or not args:
        return
    path = args[0]
    if path is None or isinstance(path, int):  # a file descriptor
        return
    try:
        name = os.fsdecode(path)
    except TypeError:
        return
    full = os.path.realpath(name)
    for root, allowed, refused in _Hidden.active:
        if full.startswith(root + os.sep) and full not in allowed:
            refused.append(full)
            raise FileNotFoundError(
                errno.ENOENT, "a file of the run directory, not of the exported package", name
            )


@contextlib.contextmanager
def hidden(root: Path, allow: Sequence[Path] = ()) -> Iterator[list[str]]:
    """While active, opening a file in ``root`` (but ``allow``) raises ``FileNotFoundError``
    in this process (an audit hook: also imports, ``torch.load``, ``os.open``); yields the
    list of the refused paths."""
    if not _Hidden.hook:
        sys.addaudithook(_audit)
        _Hidden.hook = True
    refused: list[str] = []
    allowed = frozenset(os.path.realpath(p) for p in allow)
    entry = (os.path.realpath(root), allowed, refused)
    _Hidden.active.append(entry)
    try:
        yield refused
    finally:
        _Hidden.active.remove(entry)


def apply_package(package: Path, workload: Any) -> dict[str, Any]:
    """Import ``package/apply.py`` as a user would and apply it to ``workload``: the kernels
    to its roots, then the transforms (given the workload); the counts and the files."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("ka_export_apply", Path(package) / "apply.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {package}/apply.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    counts = module.apply_kernels(*workload.roots().values())
    done = module.apply_transforms(getattr(workload, "model", None), workload=workload)
    return {"counts": counts, "transforms": done}


def missing(exc: BaseException | None, refused: Sequence[str], root: Path) -> list[str]:
    """What a failed self-test lacks: the run files it tried to open (relative to ``root``)
    and the files and modules the exception chain of ``exc`` names."""
    base = Path(os.path.realpath(root))
    names = [Path(p).relative_to(base).as_posix() for p in refused if Path(p).is_relative_to(base)]
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, ModuleNotFoundError) and exc.name:
            names.append(exc.name)
        elif isinstance(exc, FileNotFoundError):
            name = exc.filename or (exc.args[0] if len(exc.args) == 1 else None)
            if isinstance(name, str) and not Path(os.path.realpath(name)).is_relative_to(base):
                names.append(name)
        exc = exc.__cause__ or exc.__context__
    return list(dict.fromkeys(names))


# ------------------------------------------------------------------ CLI


def _moved(run: RunDir, kind: str, item: str) -> str:
    """An integration item of a run that was moved or copied: its path (a kernel's after
    ``target=``) in ``run``, by the part from ``.truth/``, ``targets/`` or ``transforms/``."""
    target, sep, path = item.partition("=") if kind == "kernel" else ("", "", item)
    parts = Path(path).parts
    if not Path(path).is_relative_to(run.root):
        for i, part in enumerate(parts):
            here = run.root.joinpath(*parts[i:])
            if part in (".truth", "targets", "transforms") and here.exists():
                path = str(here)
                break
    return f"{target}{sep}{path}"


def main(argv: list[str] | None = None) -> int:
    """Re-export a run's ``optimized/`` from its ``integration.json`` and self-test it."""
    from kernel_agent import truth
    from kernel_agent.worker import call_worker

    parser = argparse.ArgumentParser(prog="python -m kernel_agent.integrate.export")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--no-check", action="store_true", help="skip the self-test")
    ns = parser.parse_args(argv)
    run = RunDir(ns.run_dir.resolve())
    keeper = truth.of(run)
    integration = keeper.load_json(run.root / "integration.json")
    if not integration:
        print(f"{run.root}: no integration.json (or a modified one)", file=sys.stderr)
        return 2
    items = integration.get("accepted") or []
    accepted = [(a["kind"], _moved(run, a["kind"], a["item"]), 0.0) for a in items]
    out = export_optimized(run, accepted)
    manifest = read_json(out / "manifest.json", {}) or {}
    print(f"exported {len(accepted)} item(s) to {out}")
    for entry in [*manifest.get("kernels", []), *manifest.get("transforms", [])]:
        for n in entry.get("needs") or []:
            print(f"  {entry['file']} needs {n['file']} (from {n['source']})")
    if ns.no_check:
        return 0
    final = integration.get("final") or {}
    result = check_export(run, call_worker, keeper.worker_args(), expected=final.get("patches"))
    print(f"export self-test: {result['reason']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
