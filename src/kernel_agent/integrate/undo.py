"""Undo handles for kernel replacements and model-level transforms.

The paired A/B integration (``worker e2e_ab``, :mod:`kernel_agent.abtest`) loads
the model once and switches it between two optimised states. Every
``apply_kernels`` / ``apply_transforms`` call made with ``handles=`` records
what it changed in an :class:`Undo` handle, by comparing snapshots taken before
and after:

* attribute bindings: the ``__dict__`` of the workload object, of every
  ``nn.Module`` under its roots, of plain objects those hold (one level, e.g. a
  KV cache) and of all their classes; dict and set values one level deep
  (``_modules``, ``_parameters``, ``_buffers``, hooks); ``__class__``;
* ``.data`` rebinding of parameters and buffers;
* the globals of the model's own packages (and of the workload's module),
  callables of ``torch`` / ``torch.nn.functional`` rebound by a monkeypatch,
  torch backend flags and the dynamo / inductor configs.

:meth:`Undo.undo` puts the original objects back, so the model holds the very
same module objects and methods as before; :meth:`Undo.redo` re-installs what
``apply`` made, with the state it built lazily since (captured CUDA graphs,
compiled code) intact.

An in-place change of a parameter's or buffer's values cannot be seen this way
and makes the handle irreversible (:attr:`Undo.problems`). A transform module
can declare ``undo = False`` (it cannot be undone in-process) or define
``undo(workload)``, called after the restore to revert what the snapshot cannot
see; such a transform is applied again to re-enter its state
(:attr:`Undo.reapply`). Irreversible handles make ``e2e_ab`` fall back to
measuring in separate processes.
"""

from __future__ import annotations

import ast
import contextlib
import sys
import types
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

_MISSING: Any = type("_Missing", (), {"__repr__": lambda self: "<missing>"})()
#: Dicts larger than this held by a tracked object are not tracked one level deep.
MAX_DICT = 4096
#: Packages whose module globals are tracked only for rebound callables.
_LIBRARIES = ("torch", "builtins", "typing", "abc", "collections", "functools", "kernel_agent")


class IrreversibleError(RuntimeError):
    """A handle cannot undo (or redo) what its ``apply`` did."""


@dataclass
class Change:
    """One binding ``apply`` changed: ``kind`` says how to set it."""

    kind: str  # item (dict key) | attr (class attribute) | class | data | set | flag | config
    target: Any
    key: Any
    old: Any
    new: Any

    def set(self, value: Any) -> None:
        kind, target, key = self.kind, self.target, self.key
        if kind == "item":
            if value is _MISSING:
                target.pop(key, None)
            else:
                target[key] = value
        elif kind == "attr":
            if value is _MISSING:
                if key in vars(target):
                    delattr(target, key)
            else:
                setattr(target, key, value)
        elif kind == "class":
            object.__setattr__(target, "__class__", value)
        elif kind == "data":
            with torch.no_grad():
                target.data = value
        elif kind == "set":
            target.clear()
            target.update(value)
        elif kind == "flag":
            key(value)  # the flag's setter
        elif kind == "config":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                setattr(target, key, value)
        else:
            raise ValueError(f"unknown change kind {kind!r}")


@dataclass
class Undo:
    """What one ``apply_kernels`` call or one transform's ``apply`` changed."""

    label: str
    changes: list[Change] = field(default_factory=list)
    #: Why the handle cannot undo (empty: it can).
    problems: list[str] = field(default_factory=list)
    #: The transform's own ``undo()``: called after the restore; re-entering means re-applying.
    revert: Callable[[], None] | None = None
    applied: bool = True

    @property
    def reversible(self) -> bool:
        return not self.problems

    @property
    def reapply(self) -> bool:
        """Re-entering this state needs ``apply`` again (and a warm-up run), not :meth:`redo`."""
        return self.revert is not None

    def undo(self) -> None:
        if not self.applied:
            return
        if self.problems:
            raise IrreversibleError(f"{self.label}: {'; '.join(self.problems)}")
        for change in reversed(self.changes):
            change.set(change.old)
        if self.revert is not None:
            self.revert()
        self.applied = False

    def redo(self) -> None:
        if self.applied:
            return
        if self.revert is not None:
            raise IrreversibleError(f"{self.label}: defines undo(), so it is applied again")
        for change in self.changes:
            change.set(change.new)
        self.applied = True

    def summary(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for change in self.changes:
            kinds[change.kind] = kinds.get(change.kind, 0) + 1
        return {"label": self.label, "changes": kinds, "problems": self.problems}


# ------------------------------------------------------------------ declarations


def declaration(path: Path) -> bool | None:
    """A transform file's ``undo``: False (``undo = False`` / ``None``: cannot be undone
    in-process), True (``def undo``), None (not declared: the snapshot decides).
    Read from the source, without importing it."""
    try:
        tree = ast.parse(Path(path).read_text())
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "undo":
            return True
        if isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {t.id for t in targets if isinstance(t, ast.Name)}
            if "undo" in names and isinstance(node.value, ast.Constant):
                return bool(node.value.value)
    return None


def declared(handle: Undo, module: Any, workload: Any) -> None:
    """Apply a loaded transform module's ``undo`` declaration to its ``handle``."""
    if not hasattr(module, "undo"):
        return
    undo = module.undo
    if callable(undo):
        handle.revert = lambda: undo(workload)
    elif not undo:
        handle.problems.append(f"it declares undo = {undo!r} (cannot be undone in-process)")


# ------------------------------------------------------------------ snapshots


def _flags() -> dict[str, tuple[Callable[[], Any], Callable[[Any], None]]]:
    """Global torch switches a transform may flip: name -> (getter, setter)."""
    b = torch.backends
    out: dict[str, tuple[Callable[[], Any], Callable[[Any], None]]] = {
        "float32_matmul_precision": (
            torch.get_float32_matmul_precision,
            torch.set_float32_matmul_precision,
        ),
        "default_dtype": (torch.get_default_dtype, torch.set_default_dtype),
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled,
            torch.use_deterministic_algorithms,
        ),
        "sdp.flash": (b.cuda.flash_sdp_enabled, b.cuda.enable_flash_sdp),
        "sdp.mem_efficient": (b.cuda.mem_efficient_sdp_enabled, b.cuda.enable_mem_efficient_sdp),
        "sdp.math": (b.cuda.math_sdp_enabled, b.cuda.enable_math_sdp),
    }
    if hasattr(b.cuda, "cudnn_sdp_enabled"):
        out["sdp.cudnn"] = (b.cuda.cudnn_sdp_enabled, b.cuda.enable_cudnn_sdp)

    def attr(obj: Any, name: str) -> tuple[Callable[[], Any], Callable[[Any], None]]:
        return (lambda: getattr(obj, name)), (lambda v: setattr(obj, name, v))

    for path, obj, names in (
        (
            "cuda.matmul",
            b.cuda.matmul,
            (
                "allow_tf32",
                "allow_fp16_reduced_precision_reduction",
                "allow_bf16_reduced_precision_reduction",
                "fp32_precision",
            ),
        ),
        ("cudnn", b.cudnn, ("enabled", "benchmark", "deterministic", "allow_tf32")),
    ):
        for name in names:
            try:
                getattr(obj, name)
            except Exception:
                continue
            out[f"{path}.{name}"] = attr(obj, name)
    return out


def _configs() -> list[Any]:
    """The dynamo and inductor config modules (imported here: a transform may set them)."""
    import torch._dynamo.config as dynamo_config
    import torch._inductor.config as inductor_config

    return [dynamo_config, inductor_config]


def _read_flags(
    flags: dict[str, tuple[Callable[[], Any], Callable[[Any], None]]],
) -> dict[str, Any]:
    values = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, (get, _) in flags.items():
            try:
                values[name] = get()
            except Exception:
                continue
    return values


def _plain(value: Any) -> bool:
    """An object whose ``__dict__`` is tracked when a module or the workload holds it."""
    skip = (type, types.ModuleType, types.FunctionType, types.MethodType, torch.Tensor, nn.Module)
    return hasattr(value, "__dict__") and not isinstance(value, skip) and not callable(value)


def _tensor_key(t: torch.Tensor) -> tuple[Any, ...]:
    try:
        return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)
    except RuntimeError:  # no storage (a tensor subclass, say): its object identity only
        return (id(t),)


def _version(t: torch.Tensor) -> int | None:
    try:
        return int(t._version)
    except RuntimeError:  # inference tensors have no version counter
        return None


class Snapshot:
    """What :func:`record` compares: taken before ``apply``, diffed after it."""

    def __init__(self, roots: dict[str, nn.Module], extra: tuple[Any, ...] = ()) -> None:
        # first: importing dynamo rebinds torch functions (torch.manual_seed, ...)
        self.configs = [(cfg, cfg.get_config_copy()) for cfg in _configs()]
        objects: dict[int, Any] = {}

        def add(obj: Any) -> None:
            objects.setdefault(id(obj), obj)

        for obj in extra:
            add(obj)
            for value in list(vars(obj).values()) if hasattr(obj, "__dict__") else []:
                if isinstance(value, nn.Module):
                    for module in value.modules():
                        add(module)
        for root in roots.values():
            for module in root.modules():
                add(module)
        for obj in list(objects.values()):  # plain objects held by them, one level
            for value in list(vars(obj).values()):
                if _plain(value):
                    add(value)
        self.objects = list(objects.values())
        self.types = [(obj, type(obj)) for obj in self.objects]
        self.dicts: list[tuple[dict[Any, Any], dict[Any, Any]]] = []
        self.sets: list[tuple[set[Any], frozenset[Any]]] = []
        seen: set[int] = set()
        for obj in self.objects:
            d = vars(obj)
            self._dict(d, seen)
            for value in list(d.values()):
                if isinstance(value, dict) and len(value) <= MAX_DICT:
                    self._dict(value, seen)
                elif isinstance(value, set) and id(value) not in seen:
                    seen.add(id(value))
                    self.sets.append((value, frozenset(value)))
        classes: dict[type, None] = {}
        for _, cls in self.types:
            for klass in cls.__mro__[:-1]:  # not `object`
                classes.setdefault(klass)
        self.classes = [(cls, dict(vars(cls))) for cls in classes]
        self.tensors: list[tuple[str, torch.Tensor, torch.Tensor, tuple[Any, ...], int | None]] = []
        tseen: set[int] = set()
        for obj in self.objects:
            if not isinstance(obj, nn.Module):
                continue
            for group in (obj._parameters, obj._buffers):
                for name, t in group.items():
                    if t is None or id(t) in tseen or t.is_meta:
                        continue
                    tseen.add(id(t))
                    label = f"{type(obj).__name__}.{name}"
                    self.tensors.append((label, t, t.detach(), _tensor_key(t), _version(t)))
        self.modules = self._python_modules(classes)
        self.flag_fns = _flags()
        self.flags = _read_flags(self.flag_fns)

    def _dict(self, d: dict[Any, Any], seen: set[int]) -> None:
        if id(d) not in seen:
            seen.add(id(d))
            self.dicts.append((d, dict(d)))

    @staticmethod
    def _python_modules(
        classes: dict[type, None],
    ) -> list[tuple[dict[str, Any], dict[str, Any], bool]]:
        """(globals, copy, strict) of the modules whose globals are tracked: every module of
        the packages that define the tracked classes (strict: every rebinding counts), the
        modules of library classes and ``torch`` / ``torch.nn.functional`` (only rebound
        callables count)."""
        names: dict[str, bool] = {"torch": False, "torch.nn.functional": False}
        packages = set()
        for cls in classes:
            defined = cls.__module__ or ""
            top = defined.partition(".")[0]
            if top in _LIBRARIES:
                names.setdefault(defined, top == "kernel_agent")
            else:
                packages.add(top)
        for name in list(sys.modules):
            if name.partition(".")[0] in packages:
                names[name] = True
        out: list[tuple[dict[str, Any], dict[str, Any], bool]] = []
        for name, strict in names.items():
            module = sys.modules.get(name)
            if isinstance(module, types.ModuleType):
                d = vars(module)
                out.append((d, dict(d), strict))
        return out

    def diff(self) -> tuple[list[Change], list[str]]:
        """The changes since the snapshot, and what makes them irreversible."""
        changes: list[Change] = []
        problems: list[str] = []
        for obj, cls in self.types:
            if type(obj) is not cls:
                changes.append(Change("class", obj, None, cls, type(obj)))
        for d, before in self.dicts:
            changes += _dict_changes(d, before)
        for s, before_set in self.sets:
            if s != before_set:
                changes.append(Change("set", s, None, set(before_set), set(s)))
        for cls, before in self.classes:
            now = vars(cls)
            for key in before.keys() | now.keys():
                old, new = before.get(key, _MISSING), now.get(key, _MISSING)
                if old is not new:
                    changes.append(Change("attr", cls, key, old, new))
        for label, t, alias, key, version in self.tensors:
            if _tensor_key(t) != key:
                changes.append(Change("data", t, None, alias, t.detach()))
            elif version is not None and _version(t) != version:
                problems.append(f"{label} was modified in place")
        for d, before, strict in self.modules:
            for key, new in list(d.items()):
                old = before.get(key, _MISSING)
                if old is new or isinstance(new, types.ModuleType):
                    continue
                if strict or (old is not _MISSING and callable(new)):
                    changes.append(Change("item", d, key, old, new))
            if strict:
                for key in before.keys() - d.keys():
                    changes.append(Change("item", d, key, before[key], _MISSING))
        now = _read_flags(self.flag_fns)
        for name, old in self.flags.items():
            if name in now and now[name] != old:
                changes.append(Change("flag", name, self.flag_fns[name][1], old, now[name]))
        for cfg, before in self.configs:
            for key, value in cfg.get_config_copy().items():
                if key in before and before[key] != value:
                    changes.append(Change("config", cfg, key, before[key], value))
        return changes, problems


def _dict_changes(d: dict[Any, Any], before: dict[Any, Any]) -> list[Change]:
    out = []
    for key in list(before.keys() | d.keys()):
        old, new = before.get(key, _MISSING), d.get(key, _MISSING)
        if old is not new:
            out.append(Change("item", d, key, old, new))
    return out


@contextlib.contextmanager
def record(label: str, roots: dict[str, nn.Module], *extra: Any) -> Iterator[Undo]:
    """Record what the body changes in the model under ``roots`` (and the ``extra``
    objects, e.g. the workload) in the yielded handle, also when the body fails."""
    snapshot = Snapshot(roots, extra)
    handle = Undo(label)
    try:
        yield handle
    finally:
        handle.changes, problems = snapshot.diff()
        handle.problems += problems if handle.revert is None else []


def rollback(handles: list[Undo]) -> None:
    """Undo ``handles`` newest first (a partially applied state, say)."""
    for handle in reversed(handles):
        handle.undo()
