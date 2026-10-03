"""Load an agent-written (or user-written) ``harness.py`` workload.

A harness module must define ``create(spec: WorkloadSpec) -> Workload``.  The
returned object subclasses :class:`kernel_agent.workloads.base.Workload`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from kernel_agent.workloads.base import Workload, WorkloadSpec


def load_harness(path: str | Path, spec: WorkloadSpec) -> Workload:
    path = Path(path).resolve()
    name = f"kernel_agent_harness_{abs(hash(str(path)))}"
    module_spec = importlib.util.spec_from_file_location(name, path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot import harness {path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    module_spec.loader.exec_module(module)
    create = getattr(module, "create", None)
    if create is None:
        raise AttributeError(f"{path} must define create(spec) -> Workload")
    workload = create(spec)
    if not isinstance(workload, Workload):
        raise TypeError(f"{path}: create() returned {type(workload).__name__}, not a Workload")
    return workload
