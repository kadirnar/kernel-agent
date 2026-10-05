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
from kernel_agent.phases import PHASES, route
from kernel_agent.profiling.methods import entrypoints_of
from kernel_agent.region import Rewrite, apply_rewrites


@dataclass
class KernelPatch:
    target_id: str
    module_class: str
    candidate: Path
    qualname_regex: str | None = None
    #: Entrypoints the model calls on this class (``spec["capture"]["method_instances"]``,
    #: e.g. ``["forward", "forward_step"]``); replacements must provide them.
    methods: list[str] = field(default_factory=list)
    #: ``prefill`` / ``decode``: only that phase's calls of ``methods`` go to the
    #: replacement; the instance stays in the model (:func:`kernel_agent.phases.route`).
    phase: str | None = None
    #: Region target (:mod:`kernel_agent.region`): the rewrite that adds the
    #: ``module_class`` modules to its parent class, applied before every kernel.
    rewrite: Rewrite | None = None


class PatchError(RuntimeError):
    """A replacement cannot stand in for the module it replaces."""


@dataclass
class PatchReport:
    replaced: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)
    transforms: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    rewritten: dict[str, int] = field(default_factory=dict)  # parents, per region target


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

    A patch with a ``phase`` keeps the instance and routes only that phase's
    calls to the replacement, so a second patch for the other phase of the
    same class still applies.

    Raises :class:`PatchError` when a replacement lacks one of the patch's
    entrypoints that the original instance has (e.g. ``forward_step``): the
    model would call it and either crash or silently bypass the kernel.

    The rewrites of region targets' parents come first, so their kernels (and
    the kernels of classes the region modules contain) find the new modules."""
    report = report or PatchReport()
    report.rewritten.update(apply_rewrites(roots, [p.rewrite for p in patches if p.rewrite]))
    for patch in patches:
        module = load_candidate_module(patch.candidate)
        pattern = re.compile(patch.qualname_regex) if patch.qualname_regex else None
        phase = patch.phase if patch.phase in PHASES else None
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
                if phase is None:
                    _set_child(root, name, new)
                else:
                    methods = patch.methods or ["forward", *entrypoints_of(type(instance))]
                    route(instance, new, phase, methods)
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
