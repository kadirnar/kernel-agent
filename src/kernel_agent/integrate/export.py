"""Export the accepted optimisations as a small self-contained package."""

from __future__ import annotations

import shutil
from pathlib import Path

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


def export_optimized(run: RunDir, accepted: list[tuple[str, str, float]]) -> Path:
    out = run.optimized_dir
    if out.exists():
        shutil.rmtree(out)
    (out / "kernels").mkdir(parents=True)
    (out / "transforms").mkdir()
    card = run.load()["card"]
    manifest: dict[str, list[dict[str, str]]] = {"kernels": [], "transforms": []}
    for kind, arg, _ in accepted:
        if kind == "kernel":
            target_id, _, path = arg.partition("=")
            spec = read_json(run.target(target_id) / "spec.json")
            dst = out / "kernels" / f"{target_id}.py"
            shutil.copy2(path, dst)
            manifest["kernels"].append(
                {
                    "target": target_id,
                    "module_class": spec["module_class"],
                    "file": f"kernels/{dst.name}",
                }
            )
        else:
            src = Path(arg)
            dst = out / "transforms" / src.name
            shutil.copy2(src, dst)
            manifest["transforms"].append({"file": f"transforms/{dst.name}"})
    write_json(out / "manifest.json", manifest)
    (out / "apply.py").write_text(APPLY_TEMPLATE.format(repo_id=card["repo_id"], root=out))
    return out
