"""Module state outside the arguments of a call: snapshot per case, restore before each one.

A module can keep state between calls in its attributes instead of its arguments: a KV
cache (a tensor, an object holding tensors such as a static cache class or
``transformers``' ``DynamicCache``), a step counter, an RNN's hidden state. A call reads it
and changes it, so a captured case is reproducible only from the state the module had
*before that call*, not from the state it has when the capture is saved: the end of the
run, e.g. after a correctness variant's prefill. Replayed on that, the unmodified reference
fails its own capture (issue #162).

**State** (:func:`entries`): every value reachable from the attributes and buffers of the
module and its submodules, except parameters (and tensors sharing their storage),
submodules, hooks and callables: tensors, plain values (numbers, strings, dtypes, ...) and
the lists, tuples, dicts and objects holding them (up to :data:`MAX_DEPTH` levels and
:data:`MAX_ITEMS` items per container). Anything else (a lock, a tokenizer's huge vocabulary)
is opaque: never copied, compared or restored.

**Capture** (:class:`Log`, used by :mod:`kernel_agent.profiling.capture`): the recorder
snapshots the state to the CPU before and after every recorded call, each snapshot kept as a
diff against the one before. At the end of the run (:func:`finalize`) a case keeps

* ``state``: the state before its call, as a diff against the module as saved;
* ``post_state``: what its call changed (a diff against ``state``).

Only what differs is stored: a tensor of the same shape as the 8 KiB chunks
(:data:`CHUNK`) that differ, so a KV cache costs the positions that differ, not the cache.
``capture["state"]`` names the tracked entries (``keys``: the state some case needs or some
call changes) and what their diffs cost (``bytes``). Capturing costs two CPU copies of the
module's non-parameter tensors per recorded call (only while capturing).

**Replay** (:class:`Replay`): before every call of a case its state is written into the
module (in place where shapes match, so views of a cache taken at build time stay valid);
:meth:`Replay.call` wraps an entrypoint so that each call restores first
(:class:`StatefulCall`; timers restore outside the timed region, :func:`split`), and
:meth:`Replay.check` compares the state after a call with ``post_state``: only the elements
either side changed, as for in-place updates of arguments. A candidate keeps the state where
the reference keeps it (the model sets it there, e.g. ``setup_cache``), in its format.
"""

from __future__ import annotations

import functools
import types
from collections.abc import Callable, Iterator
from typing import Any

import torch
from torch import nn

from kernel_agent.profiling.methods import entrypoint

#: Bytes per chunk of a tensor diff.
CHUNK = 8192
#: Containers and objects deeper than this below an attribute are opaque.
MAX_DEPTH = 6
#: ... and so are containers and objects with more items than this.
MAX_ITEMS = 4096
#: Immutable values kept as they are.
_PLAIN = (int, float, bool, complex, str, bytes, type(None), torch.dtype, torch.device, torch.Size)
#: Attributes of every ``nn.Module`` (hooks, parameters, submodules, ...): not state.
_INTERNALS = frozenset(vars(nn.Module())) - {"_buffers"}
#: Objects of these packages are opaque (locks, loggers, streams, generators, ...).
_FOREIGN = (
    "torch",
    "builtins",
    "threading",
    "_thread",
    "logging",
    "multiprocessing",
    "concurrent",
    "asyncio",
    "functools",
    "types",
    "typing",
    "weakref",
    "collections",
    "io",
)


class _Marker:
    """A singleton that survives pickling (``OPAQUE``, ``MISSING``)."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"<{self.name.lower()}>"

    def __reduce__(self) -> str:
        return self.name


#: A value the snapshots do not copy; never compared or restored.
OPAQUE = _Marker("OPAQUE")
#: An attribute that does not exist (yet).
MISSING = _Marker("MISSING")


# ------------------------------------------------------------------ walking the state


def _ignored(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return isinstance(value, nn.Parameter)
    return isinstance(value, nn.Module | type | types.ModuleType) or callable(value)


def entries(module: nn.Module) -> dict[str, Any]:
    """The live state values of ``module`` by name: ``<submodule path>.<attribute>``."""
    found: dict[str, Any] = {}
    for prefix, sub in module.named_modules():
        dot = f"{prefix}." if prefix else ""
        for name, value in vars(sub).items():
            if name in _INTERNALS or name.startswith("__") or _ignored(value):
                continue
            if name == "_buffers":
                found.update({dot + k: v for k, v in value.items() if v is not None})
            else:
                found[dot + name] = value
    return found


def weight_storages(module: nn.Module) -> frozenset[int]:
    """Storage addresses of the parameters: tensors on them are weights, not state."""
    found = set()
    for p in module.parameters():
        try:
            found.add(p.untyped_storage().data_ptr())
        except (RuntimeError, NotImplementedError):
            continue
    return frozenset(found)


def _walkable(value: Any) -> bool:
    module = getattr(type(value), "__module__", "") or ""
    return hasattr(value, "__dict__") and module.split(".")[0] not in _FOREIGN


def _new(cls: Any) -> Any:
    """An instance of ``cls`` without running its ``__init__`` (None: not possible)."""
    try:
        return cls.__new__(cls)
    except Exception:
        return None


class _Copier:
    """Copies of state values (:func:`mirror`), one copy per object (aliasing kept).
    ``shared``: what is not copied stays a reference to the original instead of
    :data:`OPAQUE` (copies that never leave the process, e.g. :class:`Replay`'s base)."""

    def __init__(
        self, device: str | None, weights: frozenset[int] = frozenset(), *, shared: bool = False
    ) -> None:
        self.device = device
        self.weights = weights
        self.shared = shared
        self.memo: dict[int, Any] = {}

    def __call__(self, value: Any, depth: int = 0) -> Any:
        if value is OPAQUE or value is MISSING or isinstance(value, _PLAIN):
            return value
        key = id(value)
        if key in self.memo:
            return self.memo[key]
        out = self._copy(value, depth)
        if out is OPAQUE and self.shared:
            out = value
        self.memo[key] = out
        return out

    def _copy(self, value: Any, depth: int) -> Any:
        if isinstance(value, torch.Tensor):
            if (
                isinstance(value, nn.Parameter)
                or value.layout != torch.strided
                or value.is_quantized
                or value.device.type == "meta"
            ):
                return OPAQUE
            try:
                if value.untyped_storage().data_ptr() in self.weights:
                    return OPAQUE
            except (RuntimeError, NotImplementedError):
                return OPAQUE
            t = value.detach()
            return t.to(self.device, copy=True) if self.device else t.clone()
        if depth >= MAX_DEPTH or _ignored(value):
            return OPAQUE
        if type(value) in (list, tuple) or (isinstance(value, tuple) and hasattr(value, "_fields")):
            if len(value) > MAX_ITEMS:
                return OPAQUE
            items = [self(v, depth + 1) for v in value]
            if type(value) is list:
                return items
            return tuple(items) if type(value) is tuple else type(value)(*items)
        if type(value) is dict:
            if len(value) > MAX_ITEMS:
                return OPAQUE
            return {k: self(v, depth + 1) for k, v in value.items()}
        if _walkable(value) and len(vars(value)) <= MAX_ITEMS:
            new = _new(type(value))
            if new is None or not hasattr(new, "__dict__"):
                return OPAQUE
            self.memo[id(value)] = new  # cycles
            vars(new).update({k: self(v, depth + 1) for k, v in vars(value).items()})
            return new
        return OPAQUE


def mirror(
    value: Any,
    device: str | None = "cpu",
    weights: frozenset[int] = frozenset(),
    *,
    shared: bool = False,
) -> Any:
    """A copy of a state value: tensors copied (to ``device``; None: where they are),
    containers and objects rebuilt, the rest :data:`OPAQUE` (``shared``: the original)."""
    return _Copier(device, weights, shared=shared)(value)


def _copied(value: Any) -> bool:
    """Whether a snapshot copies ``value`` (anything else is :data:`OPAQUE` or shared)."""
    if isinstance(value, torch.Tensor):
        return not isinstance(value, nn.Parameter)
    if isinstance(value, _PLAIN) or type(value) in (list, tuple, dict):
        return True
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return True
    return not isinstance(value, _Marker) and not _ignored(value) and _walkable(value)


def _bytes(t: torch.Tensor) -> torch.Tensor | None:
    """The bytes of a tensor, flat (None: no such view)."""
    try:
        return t.detach().contiguous().view(-1).view(torch.uint8)
    except (RuntimeError, TypeError):
        return None


def same(a: Any, b: Any) -> bool:
    """Whether two snapshots of a value are equal (tensors bitwise)."""
    if a is b:
        return True
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return False
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        x, y = _bytes(a), _bytes(b)
        return x is not None and y is not None and torch.equal(x, y.to(x.device))
    if type(a) is not type(b) or isinstance(a, _Marker):
        return False
    if isinstance(a, _PLAIN):
        try:
            return bool(a == b) or (a != a and b != b)  # NaN
        except Exception:
            return False
    if isinstance(a, list | tuple):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(v, b[k]) for k, v in a.items())
    if hasattr(a, "__dict__"):
        return same(vars(a), vars(b))
    return False


# ------------------------------------------------------------------ diffs


def _chunks(new: torch.Tensor, old: torch.Tensor) -> dict[str, Any] | None:
    """The whole chunks of ``new``'s bytes that differ from ``old``'s (``index``, ``data``)
    and the bytes after the last whole chunk if they differ (``tail``); None: no byte view.
    Applying it needs no device synchronisation (:func:`_write_chunks`)."""
    a, b = _bytes(new), _bytes(old)
    if a is None or b is None:
        return None
    b = b.to(a.device)
    whole = a.numel() // CHUNK * CHUNK
    rows_a, rows_b = a[:whole].view(-1, CHUNK), b[:whole].view(-1, CHUNK)
    index = (rows_a != rows_b).any(1).nonzero().flatten()
    out = {"kind": "chunks", "index": index, "data": rows_a[index].clone()}
    if not torch.equal(a[whole:], b[whole:]):
        out["tail"] = a[whole:].clone()
    return out


def diff(new: Any, old: Any, memo: dict[tuple[int, int], Any] | None = None) -> Any:
    """What turns snapshot ``old`` into ``new`` (None: they are equal): changed chunks of a
    tensor of the same shape and dtype, changed items of a container or object of the same
    layout, else ``new`` itself (``{"kind": "value"}``)."""
    memo = {} if memo is None else memo
    key = (id(new), id(old))
    if key in memo:
        return memo[key]
    out: Any = None
    if not same(new, old):
        out = _diff(new, old, memo)
    memo[key] = out
    return out


def _diff(new: Any, old: Any, memo: dict[tuple[int, int], Any]) -> Any:
    if (
        isinstance(new, torch.Tensor)
        and isinstance(old, torch.Tensor)
        and new.shape == old.shape
        and new.dtype == old.dtype
        and (found := _chunks(new, old)) is not None
    ):
        return found
    if type(new) is type(old) and not isinstance(new, (*_PLAIN, _Marker, torch.Tensor)):
        items: dict[Any, Any] | None = None
        kind = "items"
        if isinstance(new, list | tuple) and len(new) == len(old):
            items = {i: diff(x, y, memo) for i, (x, y) in enumerate(zip(new, old, strict=True))}
        elif isinstance(new, dict) and new.keys() == old.keys():
            items = {k: diff(v, old[k], memo) for k, v in new.items()}
        elif hasattr(new, "__dict__") and vars(new).keys() == vars(old).keys():
            kind = "attrs"
            items = {k: diff(v, vars(old)[k], memo) for k, v in vars(new).items()}
        if items is not None:
            return {"kind": kind, "items": {k: v for k, v in items.items() if v is not None}}
    return {"kind": "value", "value": mirror(new, device=None)}


def diff_maps(new: dict[str, Any], old: dict[str, Any]) -> dict[str, Any]:
    """:func:`diff` per entry of two snapshots (an entry one of them lacks is
    :data:`MISSING` there); only the entries that differ."""
    memo: dict[tuple[int, int], Any] = {}
    out = {}
    for key in [*new, *(k for k in old if k not in new)]:
        found = diff(new.get(key, MISSING), old.get(key, MISSING), memo)
        if found is not None:
            out[key] = found
    return out


def _write_chunks(t: torch.Tensor, enc: dict[str, Any]) -> torch.Tensor:
    """``t`` with the chunks of ``enc`` written into it (in place when it is contiguous)."""
    target = t if t.is_contiguous() else t.contiguous()
    flat = target.view(-1).view(torch.uint8)
    whole = flat.numel() // CHUNK * CHUNK
    if enc["index"].numel():
        index, data = enc["index"].to(flat.device), enc["data"].to(flat.device)
        flat[:whole].view(-1, CHUNK)[index] = data
    if (tail := enc.get("tail")) is not None:
        flat[whole:] = tail.to(flat.device)
    if target is not t:
        t.copy_(target)
    return t


def _to(value: Any, device: torch.device) -> Any:
    """A diff (:func:`diff`) with its tensors on ``device``."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if type(value) is dict:
        return {k: _to(v, device) for k, v in value.items()}
    if type(value) in (list, tuple):
        return type(value)(_to(v, device) for v in value)
    if isinstance(value, (*_PLAIN, _Marker)):
        return value
    return mirror(value, device=str(device))  # a snapshot in a "value" diff


def apply(enc: Any, old: Any, *, exact: bool = False) -> Any:
    """``old`` changed by the diff ``enc`` (in place where possible); returns the new value.
    ``exact``: ``old`` is a snapshot, whose ``"value"`` parts are replaced by copies as they
    are; otherwise it is live state, made equal to them in place (:func:`assign`)."""
    if enc is None:
        return old
    kind = enc["kind"]
    if kind == "value":
        return mirror(enc["value"], device=None) if exact else assign(old, enc["value"])
    if kind == "chunks":
        return _write_chunks(old, enc)
    if kind == "attrs":
        attrs = vars(old)
        for name, sub in enc["items"].items():
            value = apply(sub, attrs.get(name, MISSING), exact=exact)
            if value is MISSING:
                attrs.pop(name, None)
            else:
                attrs[name] = value
        return old
    if isinstance(old, tuple):
        items = list(old)
        for i, sub in enc["items"].items():
            items[i] = apply(sub, items[i], exact=exact)
        return tuple(items) if type(old) is tuple else type(old)(*items)
    for k, sub in enc["items"].items():
        old[k] = apply(sub, old[k], exact=exact)
    return old


def assign(current: Any, desired: Any) -> Any:
    """``current`` made equal to the snapshot ``desired``, in place where the layout matches
    (a tensor of the same shape, dtype and device is copied into); returns the value to
    keep (a fresh copy where it does not match: ``desired`` itself is never handed out).
    What snapshots do not copy (:data:`OPAQUE`, or shared originals) leaves ``current``
    alone; it only fills a :data:`MISSING` one."""
    if desired is OPAQUE or (desired is not MISSING and not _copied(desired)):
        return desired if current is MISSING else current
    if desired is MISSING or isinstance(desired, _PLAIN):
        return desired
    if isinstance(desired, torch.Tensor):
        mine = isinstance(current, torch.Tensor) and not isinstance(current, nn.Parameter)
        if mine and current.shape == desired.shape and current.dtype == desired.dtype:
            if current is not desired:
                current.copy_(desired)  # in place, on the device the state lives on
            return current
        return desired.to(current.device if mine else desired.device, copy=True)
    if type(desired) is list:
        if type(current) is list and len(current) == len(desired):
            current[:] = [assign(c, d) for c, d in zip(current, desired, strict=True)]
            return current
        return [assign(MISSING, d) for d in desired]
    if isinstance(desired, tuple):
        olds = list(current) if type(current) is type(desired) else []
        if len(olds) != len(desired):
            olds = [MISSING] * len(desired)
        items = [assign(c, d) for c, d in zip(olds, desired, strict=True)]
        return tuple(items) if type(desired) is tuple else type(desired)(*items)
    if type(desired) is dict:
        if type(current) is dict:
            for k in [k for k in current if k not in desired]:
                del current[k]
            for k, d in desired.items():
                current[k] = assign(current.get(k, MISSING), d)
            return current
        return {k: assign(MISSING, d) for k, d in desired.items()}
    if type(current) is not type(desired):
        current = _new(type(desired))
        if current is None:
            return desired
    attrs = vars(current)
    for k, d in vars(desired).items():
        value = assign(attrs.get(k, MISSING), d)
        if value is MISSING:
            attrs.pop(k, None)
        else:
            attrs[k] = value
    return current


def nbytes(value: Any) -> int:
    """Bytes of the tensors in a diff or snapshot (each tensor counted once)."""
    seen: set[int] = set()

    def walk(v: Any, depth: int = 0) -> int:
        if depth > 2 * MAX_DEPTH + 4 or id(v) in seen:
            return 0
        seen.add(id(v))
        if isinstance(v, torch.Tensor):
            return v.numel() * v.element_size()
        if isinstance(v, dict):
            return sum(walk(x, depth + 1) for x in v.values())
        if isinstance(v, list | tuple):
            return sum(walk(x, depth + 1) for x in v)
        if hasattr(v, "__dict__") and not isinstance(v, _Marker):
            return sum(walk(x, depth + 1) for x in vars(v).values())
        return 0

    return walk(value)


# ------------------------------------------------------------------ capture


class Log:
    """Snapshots of one module's state on the CPU, in the order taken: the first one in
    full, every later one as its diff against the one before (:meth:`take`)."""

    def __init__(self, module: nn.Module) -> None:
        self.module = module
        self.weights = weight_storages(module)
        self.first: dict[str, Any] | None = None
        self.last: dict[str, Any] | None = None
        #: ``diffs[i - 1]`` turns snapshot ``i - 1`` into snapshot ``i``.
        self.diffs: list[dict[str, Any]] = []

    def snapshot(self) -> dict[str, Any]:
        copier = _Copier("cpu", self.weights)
        with torch.inference_mode():
            return {k: copier(v) for k, v in entries(self.module).items()}

    def take(self) -> int | None:
        """Snapshot the state now; its index (None: not possible during a CUDA graph capture)."""
        if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
            return None
        now = self.snapshot()
        if self.last is None:
            self.first = now
        else:
            self.diffs.append(diff_maps(now, self.last))
        self.last = now
        return len(self.diffs)

    def changes(self, before: int | None, after: int | None) -> dict[str, Any] | None:
        """What changed from snapshot ``before`` to the next one, ``after`` (None: not
        consecutive snapshots)."""
        if before is None or after != before + 1:
            return None
        return self.diffs[after - 1]

    def replay(self) -> Iterator[tuple[int, dict[str, Any]]]:
        """``(index, snapshot)`` in order, rebuilt on one working copy (use each before
        advancing)."""
        if self.first is None:
            return
        state = {k: mirror(v, device=None) for k, v in self.first.items()}
        yield 0, state
        for i, step in enumerate(self.diffs, 1):
            for key, enc in step.items():
                value = apply(enc, state.get(key, MISSING), exact=True)
                if value is MISSING:
                    state.pop(key, None)
                else:
                    state[key] = value
            yield i, state


#: Case key of the recorders' bookkeeping until :func:`finalize`: ``(Log, index of the
#: snapshot before the call)``.
LOG_KEY = "_state_log"


def record(case: dict[str, Any], log: Log, before: int | None, after: int | None) -> None:
    """Note on ``case`` the snapshots ``log`` took around its call (``post_state``: what
    the call changed)."""
    case[LOG_KEY] = (log, before)
    if changes := log.changes(before, after):
        case["post_state"] = changes


def finalize(cases: list[dict[str, Any]], module: nn.Module) -> dict[str, Any]:
    """Turn the recorders' snapshots into ``state`` (the state before the call, a diff
    against the module as it is now: as saved) and ``post_state`` per case; returns
    ``capture["state"]``: the tracked ``keys``, the ``bytes`` of the diffs and the
    ``cases`` that need a state other than the saved one ({} without state changes)."""
    logs: dict[int, Log] = {}
    wanted: dict[int, dict[int, list[dict[str, Any]]]] = {}
    for case in cases:
        log, index = case.pop(LOG_KEY, (None, None))
        if log is None or index is None:
            continue
        logs[id(log)] = log
        wanted.setdefault(id(log), {}).setdefault(index, []).append(case)
    if not logs:
        for case in cases:
            case.pop("post_state", None)
        return {}
    final = next(iter(logs.values())).snapshot()
    for key, log in logs.items():
        for index, snap in log.replay():
            for case in wanted[key].get(index, []):
                if found := diff_maps(snap, final):
                    case["state"] = found
    keys = sorted({k for c in cases for k in (*c.get("state", {}), *c.get("post_state", {}))})
    for case in cases:
        if not case.get("post_state"):
            case.pop("post_state", None)
    if not keys:
        return {}
    return {
        "keys": keys,
        "bytes": nbytes([c.get(k) for c in cases for k in ("state", "post_state")]),
        "cases": sum(1 for c in cases if c.get("state")),
    }


def describe(info: dict[str, Any]) -> str:
    """One line on a capture's tracked state (``capture["state"]``)."""
    if not info.get("keys"):
        return "no module state changes between or during the recorded calls"
    keys = ", ".join(f"`{k}`" for k in info["keys"][:6])
    more = f" (+{len(info['keys']) - 6})" if len(info["keys"]) > 6 else ""
    return (
        f"module state {keys}{more} restored per case ({info.get('cases', 0)} cases with a "
        f"state of their own, {info.get('bytes', 0) / 1024:.1f} KiB of diffs)"
    )


# ------------------------------------------------------------------ replay


class StatefulCall:
    """An entrypoint of a stateful module bound to one case: every call first restores the
    case's state (:meth:`Replay.restore`). Timers call :func:`split` and restore outside
    the timed region."""

    def __init__(self, fn: Callable[..., Any], restore: Callable[[], None]) -> None:
        self.fn = fn
        self.restore = restore

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.restore()
        return self.fn(*args, **kwargs)


def split(fn: Callable[..., Any]) -> tuple[Callable[..., Any], Callable[[], None] | None]:
    """``(the callable, its restore step or None)``."""
    if isinstance(fn, StatefulCall):
        return fn.fn, fn.restore
    return fn, None


def _owner(holder: Any, key: str) -> tuple[Any, str]:
    path, _, name = key.rpartition(".")
    try:
        owner = holder.get_submodule(path) if path else holder
    except (AttributeError, TypeError):
        return None, name
    return (owner, name) if isinstance(owner, nn.Module) else (None, name)


def _get(owner: nn.Module, name: str) -> Any:
    if name in owner._buffers:
        return owner._buffers[name]
    return vars(owner).get(name, MISSING)


def _set(owner: nn.Module, name: str, value: Any) -> None:
    if name in owner._buffers:
        if value is MISSING:
            del owner._buffers[name]
        else:
            owner._buffers[name] = value
    elif value is MISSING:
        vars(owner).pop(name, None)
    else:
        vars(owner)[name] = value


class Replay:
    """The state of a loaded capture's cases (``capture["state"]``), written into a module
    before each call. ``module``: the capture's module before any call (its state is the
    base the cases' diffs apply to). Falsy for a capture without tracked state."""

    def __init__(self, capture: dict[str, Any], module: nn.Module) -> None:
        self.keys: list[str] = list((capture.get("state") or {}).get("keys") or [])
        live = entries(module) if self.keys else {}
        copier = _Copier(None, weight_storages(module) if self.keys else frozenset(), shared=True)
        with torch.inference_mode():
            self.base = {k: copier(live.get(k, MISSING)) for k in self.keys}
        #: The class of the submodule that holds each entry in the reference.
        self.owners = {k: type(_owner(module, k)[0] or module) for k in self.keys}
        tensors = [*module.parameters(), *module.buffers()]
        #: Where the module lives: the cases' diffs are moved there once (a capture loaded
        #: without ``map_location`` keeps them on the CPU, where they were recorded).
        self.device = tensors[0].device if tensors else torch.device("cpu")
        self._moved: dict[int, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}

    def __bool__(self) -> bool:
        return bool(self.keys)

    def _diffs(self, case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """``case``'s ``state`` and ``post_state`` on the module's device."""
        moved = self._moved.get(id(case))
        if moved is None or moved[0] is not case:
            with torch.inference_mode():
                pre = _to(case.get("state") or {}, self.device)
                post = _to(case.get("post_state") or {}, self.device)
            moved = self._moved[id(case)] = (case, pre, post)
        return moved[1], moved[2]

    def _place(self, holder: Any, key: str) -> tuple[Any, str, Any]:
        """``(owner, attribute, current value)`` of an entry in ``holder``; owner None when
        it does not keep the entry (not the reference's class there, and no such attribute:
        a wrapper keeps it in the reference copy it holds)."""
        owner, name = _owner(holder, key)
        if owner is None:
            return None, name, MISSING
        current = _get(owner, name)
        if current is MISSING and not isinstance(owner, self.owners[key]):
            return None, name, MISSING
        return owner, name, current

    def restore(self, case: dict[str, Any], *holders: Any) -> None:
        """Write ``case``'s state into every holder (the module the call runs on, and the
        reference copy a candidate was built from: either may hold the state)."""
        if not self.keys:
            return
        diffs, _ = self._diffs(case)
        seen: set[int] = set()
        with torch.inference_mode():
            for holder in holders:
                if id(holder) in seen:
                    continue
                seen.add(id(holder))
                for key in self.keys:
                    owner, name, current = self._place(holder, key)
                    if owner is None:
                        continue
                    value = apply(diffs.get(key), assign(current, self.base[key]))
                    if value is not current:
                        _set(owner, name, value)

    def state(self, *holders: Any) -> dict[str, Any]:
        """The live tracked state: each entry from the first holder that keeps it."""
        out: dict[str, Any] = {}
        for key in self.keys:
            for holder in holders:
                owner, _, value = self._place(holder, key)
                if owner is not None:
                    out[key] = value
                    break
        return out

    def expected(self, case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """``(pre, post)``: copies of ``case``'s state before and after its call."""
        state, changes = self._diffs(case)
        with torch.inference_mode():
            pre = {k: apply(state.get(k), assign(MISSING, v)) for k, v in self.base.items()}
            post = {k: apply(changes.get(k), assign(MISSING, v)) for k, v in pre.items()}
        return pre, post

    def check(self, case: dict[str, Any], *holders: Any, **kw: Any) -> list[dict[str, Any]]:
        """The state after ``case``'s call on ``holders`` against the capture's (in-place
        side effects, :func:`kernels.compare.compare_side_effects_flat`; ``kw``: its tier
        options); plain values must be equal where either side changed them."""
        from kernel_agent.kernels.compare import compare_side_effects_flat, flatten

        if not self.keys:
            return []
        pre, post = self.expected(case)
        now = self.state(*holders)
        checks = compare_side_effects_flat(
            flatten(pre, "state"), flatten(post, "state"), flatten(now, "state"), **kw
        )
        for check in checks:
            if check.get("error") == "missing in candidate arguments":
                check["error"] = "missing in the module's state (keep it where the reference does)"
        before, after, got = _plain(pre, "state"), _plain(post, "state"), _plain(now, "state")
        for name, value in after.items():
            mine = got.get(name, MISSING)
            if (not same(value, before.get(name)) or not same(mine, before.get(name))) and not (
                same(mine, value)
            ):
                checks.append(
                    {
                        "name": name,
                        "ok": False,
                        "error": f"{mine!r} after the call, expected {value!r}",
                    }
                )
        return checks

    def call(self, case: dict[str, Any], *holders: Any) -> Callable[..., Any]:
        """The entrypoint of ``case`` on ``holders[0]``; restoring the case's state into
        every holder before each call when the capture tracks state."""
        return self.wrap(entrypoint(holders[0], case.get("method", "forward")), case, *holders)

    def wrap(
        self, fn: Callable[..., Any], case: dict[str, Any], *holders: Any
    ) -> Callable[..., Any]:
        """``fn`` restoring ``case``'s state into ``holders`` before each call."""
        if not self.keys:
            return fn
        return StatefulCall(fn, functools.partial(self.restore, case, *holders))


def _plain(value: Any, prefix: str, depth: int = 0) -> dict[str, Any]:
    """Plain values (numbers, strings, ...) of a state tree by name, as ``flatten`` names
    tensors."""
    if depth > 2 * MAX_DEPTH or isinstance(value, torch.Tensor) or not _copied(value):
        return {}  # what snapshots do not copy is not compared either
    if isinstance(value, _PLAIN):
        return {prefix: value}
    found: dict[str, Any] = {}
    if isinstance(value, dict):
        for k, v in value.items():
            found.update(_plain(v, f"{prefix}.{k}", depth + 1))
    elif isinstance(value, list | tuple):
        for i, v in enumerate(value):
            found.update(_plain(v, f"{prefix}[{i}]", depth + 1))
    elif hasattr(value, "__dict__") and not isinstance(value, nn.Module | _Marker):
        for k, v in vars(value).items():
            if not k.startswith("__"):
                found.update(_plain(v, f"{prefix}.{k}", depth + 1))
    return found
