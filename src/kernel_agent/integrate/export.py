"""Export the accepted optimisations as a small self-contained package."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from kernel_agent import phases, region
from kernel_agent.truth import TamperError, alarm, read_verified
from kernel_agent.workspace import RunDir, read_json, write_json

APPLY_TEMPLATE = '''"""Optimised kernels for {repo_id}, produced by kernel-agent.

Usage::

    import sys; sys.path.insert(0, "{root}")
    from apply import apply_kernels, apply_transforms
    apply_kernels(model)            # model: the nn.Module(s) loaded as in the benchmark
    apply_transforms(model)         # model-level transforms, in the integrated order

Model-level transforms (``transforms/*.py``) receive the kernel-agent workload
object. Most only use ``workload.model``; ``apply_transforms(model)`` passes a
stand-in holding ``model`` (pass ``workload=`` for transforms that need more).

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


def _copy(run: RunDir, src: Path, dst: Path, digests: dict[str, str | None] | None) -> None:
    """Copy ``src`` after checking it against its recorded sha256 (when ``digests`` has one)."""
    sha256 = (digests or {}).get(str(src))
    try:
        dst.write_bytes(read_verified(src, sha256))
    except TamperError as exc:
        rel = src.relative_to(run.root).as_posix() if src.is_relative_to(run.root) else str(src)
        alarm(run, rel, str(exc))
        raise


def export_optimized(
    run: RunDir,
    accepted: list[tuple[str, str, float]],
    *,
    digests: dict[str, str | None] | None = None,
) -> Path:
    """``optimized/``: the accepted snapshots + ``apply.py``. ``digests`` (snapshot path →
    sha256 of the evaluated file) makes a snapshot changed since then refuse the export."""
    out = run.optimized_dir
    if out.exists():
        shutil.rmtree(out)
    (out / "kernels").mkdir(parents=True)
    (out / "transforms").mkdir()
    card = run.load()["card"]
    manifest: dict[str, list[Any]] = {"kernels": [], "transforms": []}
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
            manifest["kernels"].append(
                {
                    "target": target_id,
                    "module_class": spec["module_class"],
                    "file": f"kernels/{dst.name}",
                    "methods": [m for m in methods if m != "forward"],
                    **scope,
                }
            )
        else:
            src = Path(arg)
            dst = out / "transforms" / src.name
            _copy(run, src, dst, digests)
            manifest["transforms"].append({"file": f"transforms/{dst.name}"})
    # Root names prefix the qualnames that `qualname_regex` was written against.
    profile = read_json(run.profile_dir / "profile.json", {}) or {}
    manifest["roots"] = sorted({c["root"] for c in profile.get("classes", []) if c.get("root")})
    shutil.copy2(Path(phases.__file__), out / "phases.py")  # phase routing for apply.py
    write_json(out / "manifest.json", manifest)
    (out / "apply.py").write_text(APPLY_TEMPLATE.format(repo_id=card["repo_id"], root=out))
    return out
