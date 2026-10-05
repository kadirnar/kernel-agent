"""Export the accepted optimisations as a small self-contained package."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from kernel_agent.truth import TamperError, alarm, read_verified
from kernel_agent.workspace import RunDir, read_json, write_json

APPLY_TEMPLATE = '''"""Optimised kernels for {repo_id}, produced by kernel-agent.

Usage::

    import sys; sys.path.insert(0, "{root}")
    from apply import apply_kernels
    apply_kernels(model)            # model: the nn.Module(s) loaded as in the benchmark

Model-level transforms (``transforms/*.py``) take the kernel-agent workload
object; port them by hand if you use your own inference code.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from torch import nn

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "manifest.json").read_text())


def _load(path: Path):
    name = f"ka_opt_{{path.stem}}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def apply_kernels(*roots: nn.Module) -> dict[str, int]:
    """Replace every matching module instance in ``roots``; returns counts."""
    counts: dict[str, int] = {{}}
    for entry in MANIFEST["kernels"]:
        module = _load(HERE / entry["file"])
        n = 0
        for root in roots:
            for name, child in list(root.named_modules()):
                if not name or type(child).__name__ != entry["module_class"]:
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
                parent_name, _, attr = name.rpartition(".")
                parent = root.get_submodule(parent_name) if parent_name else root
                if isinstance(parent, (nn.ModuleList, nn.Sequential)) and attr.isdigit():
                    parent[int(attr)] = new
                else:
                    setattr(parent, attr, new)
                n += 1
        counts[entry["target"]] = n
    return counts
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
    manifest: dict[str, list[dict[str, Any]]] = {"kernels": [], "transforms": []}
    for kind, arg, _ in accepted:
        if kind == "kernel":
            target_id, _, path = arg.partition("=")
            spec = read_json(run.target(target_id) / "spec.json")
            dst = out / "kernels" / f"{target_id}.py"
            _copy(run, Path(path), dst, digests)
            methods = spec.get("capture", {}).get("method_instances", {})
            manifest["kernels"].append(
                {
                    "target": target_id,
                    "module_class": spec["module_class"],
                    "file": f"kernels/{dst.name}",
                    "methods": [m for m in methods if m != "forward"],
                }
            )
        else:
            src = Path(arg)
            dst = out / "transforms" / src.name
            _copy(run, src, dst, digests)
            manifest["transforms"].append({"file": f"transforms/{dst.name}"})
    write_json(out / "manifest.json", manifest)
    (out / "apply.py").write_text(APPLY_TEMPLATE.format(repo_id=card["repo_id"], root=out))
    return out
