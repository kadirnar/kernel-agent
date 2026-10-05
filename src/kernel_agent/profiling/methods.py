"""Module entrypoints other than ``forward``.

Custom autoregressive models often run their decode loop through methods such
as ``forward_step`` (VoxCPM: ``MiniCPMModel.forward_step`` →
``MiniCPMDecoderLayer.forward_step`` → ``MiniCPMAttention.forward_step``).
Those calls bypass ``nn.Module.__call__`` and therefore every forward hook, so
the profiler and the capture recorder would not see them.

:func:`discover_entrypoints` finds such methods by name and :func:`instrument`
wraps them per instance so that the same ``pre``/``post`` callbacks fire for
``forward`` (through hooks) and for every other entrypoint (through wrappers),
with the method name attached.
"""

from __future__ import annotations

import collections
import contextlib
import functools
import inspect
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

from torch import nn

#: Methods matching this pattern are entrypoints when a class (or one of its
#: non-``torch`` base classes) defines them.
ENTRYPOINT_PATTERN = re.compile(r"^(forward\w*|step|decode\w*|prefill\w*|generate_step)$")

PreFn = Callable[[nn.Module, str, tuple[Any, ...], dict[str, Any]], None]
PostFn = Callable[[nn.Module, str, tuple[Any, ...], dict[str, Any], Any], None]


def parse_entrypoints(raw: Any) -> dict[str, list[str]]:
    """``"Cls.method,Cls2.method2"`` (or a list of such items, or a dict) -> {class: [methods]}."""
    out: dict[str, list[str]] = {}
    if not raw:
        return out
    if isinstance(raw, Mapping):
        for cls, names in raw.items():
            items = [names] if isinstance(names, str) else list(names)
            for name in items:
                out.setdefault(str(cls), []).append(str(name).strip())
        return out
    items = raw.split(",") if isinstance(raw, str) else [str(x) for x in raw]
    for item in items:
        cls, sep, name = item.strip().rpartition(".")
        if not sep or not cls or not name:
            raise ValueError(f"entrypoint {item!r} must look like ClassName.method")
        out.setdefault(cls, []).append(name)
    return out


def workload_entrypoints(workload: Any) -> dict[str, list[str]]:
    """Extra entrypoints from ``Workload.entrypoints`` and the ``entrypoints`` option."""
    out = parse_entrypoints(getattr(type(workload), "entrypoints", None))
    options = getattr(workload, "options", None) or {}
    for cls, names in parse_entrypoints(options.get("entrypoints")).items():
        out.setdefault(cls, []).extend(names)
    return out


def entrypoints_of(cls: type, extra: Mapping[str, Sequence[str]] | None = None) -> list[str]:
    """Non-``forward`` entrypoints of one module class.

    Public methods matching :data:`ENTRYPOINT_PATTERN` that the class or one of
    its user-level base classes defines (methods inherited from ``nn.Module``
    and other ``torch`` classes are ignored; so are generator methods), plus
    ``extra[cls.__name__]``.  ``forward`` itself is always an entrypoint and is
    seen through hooks, so it is never listed."""
    names: list[str] = []
    for klass in cls.__mro__:
        if klass is nn.Module or klass is object or klass.__module__.startswith("torch."):
            continue
        for name, value in vars(klass).items():
            if (
                name != "forward"
                and ENTRYPOINT_PATTERN.match(name)
                and inspect.isfunction(value)
                and not inspect.isgeneratorfunction(value)
                and name not in names
            ):
                names.append(name)
    for name in (extra or {}).get(cls.__name__, []):
        if name != "forward" and name not in names and callable(getattr(cls, name, None)):
            names.append(name)
    return names


def discover_entrypoints(
    roots: Mapping[str, nn.Module], extra: Mapping[str, Sequence[str]] | None = None
) -> dict[type, list[str]]:
    """``{module class: [non-forward entrypoints]}`` for every class in ``roots`` that has any."""
    found: dict[type, list[str]] = {}
    seen: set[type] = set()
    for root in roots.values():
        for module in root.modules():
            cls = type(module)
            if cls in seen:
                continue
            seen.add(cls)
            names = entrypoints_of(cls, extra)
            if names:
                found[cls] = names
    return found


def describe(methods: Mapping[type, Sequence[str]]) -> list[str]:
    """``["MiniCPMAttention.forward_step", ...]`` for logs and profile summaries."""
    return sorted(f"{cls.__name__}.{name}" for cls, names in methods.items() for name in names)


def entrypoint(module: Any, method: str) -> Callable[..., Any]:
    """The callable for one captured case: the module itself for ``forward`` (hooks run
    as in the model), otherwise the bound method."""
    return module if method == "forward" else getattr(module, method)


@contextlib.contextmanager
def instrument(
    modules: Iterable[nn.Module],
    methods: Mapping[type, Sequence[str]],
    pre: PreFn,
    post: PostFn,
) -> Iterator[None]:
    """Call ``pre(module, method, args, kwargs)`` / ``post(..., output)`` around every
    entrypoint call of ``modules``.

    ``forward`` is observed through forward hooks; the methods listed for a
    module's class in ``methods`` are wrapped on the instance (an instance
    attribute, restored on exit, so other instances and the class are untouched).

    Only the outermost entrypoint call of an instance is reported: when an
    entrypoint calls another entrypoint of the *same* instance (``forward`` →
    ``forward_features``, ``forward_step`` → ``self(x)``), the inner call is an
    implementation detail of the outer one and is not reported.  As with plain
    hooks, ``post`` is not called when the entrypoint raises.
    """
    depth: collections.Counter[int] = collections.Counter()
    frames: list[bool] = []  # forward hooks only; they nest LIFO
    handles: list[Any] = []
    restore: list[tuple[nn.Module, str, Any]] = []
    missing = object()

    def fwd_pre(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        outer = depth[id(module)] == 0
        depth[id(module)] += 1
        frames.append(outer)
        if outer:
            pre(module, "forward", args, kwargs)

    def fwd_post(
        module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any
    ) -> None:
        outer = frames.pop() if frames else False
        depth[id(module)] = max(depth[id(module)] - 1, 0)
        if outer:
            post(module, "forward", args, kwargs, output)

    def wrap(module: nn.Module, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = id(module)
            outer = depth[key] == 0
            depth[key] += 1
            try:
                if outer:
                    pre(module, name, args, kwargs)
                output = fn(*args, **kwargs)
                if outer:
                    post(module, name, args, kwargs, output)
                return output
            finally:
                depth[key] -= 1

        return wrapper

    try:
        done: set[int] = set()
        for module in modules:
            if id(module) in done:
                continue
            done.add(id(module))
            handles.append(module.register_forward_pre_hook(fwd_pre, with_kwargs=True))
            handles.append(module.register_forward_hook(fwd_post, with_kwargs=True))
            for name in methods.get(type(module), ()):
                fn = getattr(module, name, None)
                if not callable(fn):
                    continue
                restore.append((module, name, module.__dict__.get(name, missing)))
                object.__setattr__(module, name, wrap(module, name, fn))
        yield
    finally:
        for handle in handles:
            handle.remove()
        for module, name, previous in reversed(restore):
            if previous is missing:
                module.__dict__.pop(name, None)
            else:
                object.__setattr__(module, name, previous)
