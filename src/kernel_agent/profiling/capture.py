"""Capture a hot module together with real inputs/outputs from the model run.

The capture file is the self-contained unit an agent optimises: it holds the
original module (weights included), up to ``max_cases`` calls with distinct
shape signatures, the reference outputs and the post-call state of the
arguments (so in-place updates such as KV-cache appends are verified too).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from torch import nn

from kernel_agent.profiling.profiler import signature_of
from kernel_agent.workloads.base import Workload, synchronize


def _detach(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


class _Recorder:
    def __init__(self, module: nn.Module, max_cases: int) -> None:
        self.module = module
        self.max_cases = max_cases
        self.cases: dict[str, dict[str, Any]] = {}
        self._pending: list[tuple[str, Any, Any] | None] = []
        self.handles = [
            module.register_forward_pre_hook(self._pre, with_kwargs=True),
            module.register_forward_hook(self._post, with_kwargs=True),
        ]

    def _pre(self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        # Group calls by the primary input only: decode steps share it even
        # though masks / cache positions grow every step.
        sig = signature_of(args, kwargs, limit=1)
        case = self.cases.get(sig)
        if case is not None:
            case["count"] += 1
            self._pending.append(None)
        elif len(self.cases) < self.max_cases:
            self._pending.append((sig, _detach(args), _detach(kwargs)))
        else:
            self._pending.append(None)

    def _post(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any
    ) -> None:
        pending = self._pending.pop() if self._pending else None
        if pending is None:
            return
        sig, pre_args, pre_kwargs = pending
        self.cases[sig] = {
            "signature": sig,
            "full_signature": signature_of(pre_args, pre_kwargs, limit=8),
            "count": 1,
            "args": pre_args,
            "kwargs": pre_kwargs,
            "output": _detach(output),
            "post_args": _detach(args),
            "post_kwargs": _detach(kwargs),
        }

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


def find_instance(
    roots: dict[str, nn.Module], cls: str, qualname: str | None = None
) -> tuple[str, nn.Module]:
    for root_name, root in roots.items():
        for name, module in root.named_modules():
            full = f"{root_name}.{name}" if name else root_name
            if qualname is not None and full != qualname:
                continue
            if type(module).__name__ == cls:
                return full, module
    raise LookupError(f"no module of class {cls!r} (qualname={qualname!r})")


def count_calls(roots: dict[str, nn.Module], cls: str) -> int:
    return sum(1 for root in roots.values() for m in root.modules() if type(m).__name__ == cls)


def capture_module(
    workload: Workload,
    inputs: Any,
    cls: str,
    path: Path,
    *,
    qualname: str | None = None,
    max_cases: int = 3,
) -> dict[str, Any]:
    """Run the workload once and save one instance of ``cls`` plus its calls."""
    roots = workload.roots()
    full, module = find_instance(roots, cls, qualname)
    recorder = _Recorder(module, max_cases)
    try:
        with torch.inference_mode():
            workload.run(inputs)
            synchronize()
    finally:
        recorder.remove()
    if not recorder.cases:
        raise RuntimeError(f"{full} ({cls}) was never called during the workload run")
    cases = sorted(recorder.cases.values(), key=lambda c: -c["count"])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "module": module,
            "qualname": full,
            "class": cls,
            "module_path": f"{type(module).__module__}.{type(module).__qualname__}",
            "instances": count_calls(roots, cls),
            "cases": cases,
        },
        path,
    )
    return {
        "qualname": full,
        "cases": [{"signature": c["signature"], "count": c["count"]} for c in cases],
        "bytes": path.stat().st_size,
    }


def load_capture(path: Path, device: str | None = None) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except ModuleNotFoundError:
        # Classes from ``trust_remote_code`` repos live in the HF modules cache.
        from transformers.dynamic_module_utils import init_hf_modules

        init_hf_modules()
        return torch.load(path, map_location=device, weights_only=False)


def capture_calls(
    module: nn.Module,
    calls: list[tuple[tuple[Any, ...], dict[str, Any], int]],
    path: Path,
    *,
    instances: int = 1,
) -> None:
    """Build a capture file from explicit ``(args, kwargs, count)`` calls.

    Used by tests and for synthetic shapes (e.g. other batch sizes) that the
    workload run did not exercise."""
    cases = []
    with torch.inference_mode():
        for args, kwargs, count in calls:
            pre_args, pre_kwargs = _detach(args), _detach(kwargs)
            output = module(*args, **kwargs)
            synchronize()
            cases.append(
                {
                    "signature": signature_of(pre_args, pre_kwargs, limit=1),
                    "full_signature": signature_of(pre_args, pre_kwargs, limit=8),
                    "count": count,
                    "args": pre_args,
                    "kwargs": pre_kwargs,
                    "output": _detach(output),
                    "post_args": _detach(args),
                    "post_kwargs": _detach(kwargs),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "module": module,
            "qualname": type(module).__name__,
            "class": type(module).__name__,
            "module_path": f"{type(module).__module__}.{type(module).__qualname__}",
            "instances": instances,
            "cases": cases,
        },
        path,
    )
