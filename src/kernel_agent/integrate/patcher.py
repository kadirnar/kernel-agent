"""Apply optimised kernels (module replacements) and model-level transforms."""

from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch import nn

from kernel_agent.kernels.evaluate import load_candidate_module


@dataclass
class KernelPatch:
    target_id: str
    module_class: str
    candidate: Path
    qualname_regex: str | None = None
    #: Entrypoints the model calls on this class (``spec["capture"]["method_instances"]``,
    #: e.g. ``["forward", "forward_step"]``); replacements must provide them.
    methods: list[str] = field(default_factory=list)


class PatchError(RuntimeError):
    """A replacement cannot stand in for the module it replaces."""


@dataclass
class PatchReport:
    replaced: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)
    transforms: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _set_child(root: nn.Module, qualname: str, new: nn.Module) -> None:
    parent_name, _, child = qualname.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    if isinstance(parent, nn.ModuleList | nn.Sequential) and child.isdigit():
        parent[int(child)] = new
    else:
        setattr(parent, child, new)


def apply_kernels(
    roots: dict[str, nn.Module], patches: list[KernelPatch], report: PatchReport | None = None
) -> PatchReport:
    """Replace every instance of each patch's module class with ``build(instance)``.

    Raises :class:`PatchError` when a replacement lacks one of the patch's
    entrypoints that the original instance has (e.g. ``forward_step``): the
    model would call it and either crash or silently bypass the kernel."""
    report = report or PatchReport()
    for patch in patches:
        module = load_candidate_module(patch.candidate)
        pattern = re.compile(patch.qualname_regex) if patch.qualname_regex else None
        replaced = skipped = 0
        for root_name, root in roots.items():
            targets = [
                (name, m)
                for name, m in root.named_modules()
                if name
                and type(m).__name__ == patch.module_class
                and (pattern is None or pattern.search(f"{root_name}.{name}"))
            ]
            for name, instance in targets:
                try:
                    new = module.build(instance)
                except Exception as exc:
                    report.errors.append(f"{patch.target_id}: build({root_name}.{name}): {exc}")
                    skipped += 1
                    continue
                if new is None or new is instance:
                    skipped += 1
                    continue
                missing = [
                    m
                    for m in patch.methods
                    if m != "forward"
                    and callable(getattr(instance, m, None))
                    and not callable(getattr(new, m, None))
                ]
                if missing:
                    raise PatchError(
                        f"{patch.target_id}: build({root_name}.{name}) returned a "
                        f"{type(new).__name__} without {', '.join(missing)}(), which the model "
                        f"calls on {patch.module_class}"
                    )
                _set_child(root, name, new)
                replaced += 1
        report.replaced[patch.target_id] = replaced
        report.skipped[patch.target_id] = skipped
    return report


def load_transform(path: Path) -> Any:
    path = path.resolve()
    name = f"ka_transform_{path.stem}_{abs(hash(path.read_text()))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import transform {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    if not callable(getattr(module, "apply", None)):
        raise AttributeError(f"{path.name} must define apply(workload) -> None")
    return module


def apply_transforms(workload: Any, paths: list[Path], report: PatchReport) -> PatchReport:
    for path in paths:
        load_transform(path).apply(workload)
        report.transforms.append(path.stem)
    return report
