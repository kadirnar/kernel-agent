"""Cheap Triton launches for eager-timed kernels (issue #148; docs/RESEARCH-TRITON.md §2.4
and §4.3).

``kernel[grid](*args, **meta)`` runs Triton's JIT launcher on every call: it binds the
arguments to the signature, computes the specialisation key, looks the compiled kernel up
and checks the globals it captured, then launches. That is tens of microseconds of host
time per launch (skill ``optimisation-playbook``: ~43 us per Triton launch vs ~19 us for one
``load_inline`` entry), so when the module evaluator times eager calls a candidate with
several Triton launches per call is host bound: a fused decoder layer of cuBLAS GEMMs +
5 Triton glue kernels measured 2.21x with ~216 us of host time vs ~167 us of GPU time per
call, the same math behind one C++ entry 2.70x. Under CUDA graphs / ``torch.compile``
host time does not count; there this module changes nothing.

:class:`CachedLaunch` keeps the ``CompiledKernel`` that the first JIT launch of each
specialisation returns and launches it directly afterwards (``CompiledKernel.run``, the
driver's C launcher; Unsloth's ``triton_launch.py`` and FlagGems' ``LibEntry`` do the
same)::

    _norm = CachedLaunch(_norm_kernel)          # once, next to the @triton.jit kernel
    _norm[(rows,)](x, w, y, n, eps, BLOCK=block, num_warps=4)   # as kernel[grid](...)

The key of a specialisation (:func:`spec_key`) is at least as fine as the JIT's: per
positional argument a tensor's dtype and whether its data pointer is 16-byte aligned, an
integer's ``== 1`` / ``% 16 == 0`` / 32-bit range, a float / bool / None by type, a
constexpr parameter by value; every keyword argument (constexprs and launch options such
as ``num_warps``) by value; and the current device. A call with any other argument type
(tuples, tensor descriptors, NumPy scalars) goes through the JIT every time. Not for
``@triton.autotune`` / ``@triton.heuristics`` wrappers (they add constexprs this class does
not see): pick their configs once, e.g. with :mod:`kernel_agent.kernels.tuned`, and launch
the plain ``@triton.jit`` kernel.

What it skips: the JIT's check that the globals a kernel reads have not changed since it
was compiled (keep globals constant), ``pre_run_hooks`` and, while no profiler is attached,
the launch metadata. A profiler's ``launch_enter_hook`` / ``launch_exit_hook`` still run.

The other route is Triton AOT (``python -m triton.tools.compile`` per specialisation,
``triton.tools.link`` into one C dispatcher) called from one ``load_inline`` entry that
launches every kernel of the module: no Python per launch at all, at the cost of a C++
build. ``examples/triton_cheap_launch.py`` uses this module.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Sequence
from typing import Any

import torch

_I32 = 1 << 31


def spec_key(args: Sequence[Any], constexprs: frozenset[int] = frozenset()) -> tuple[Any, ...]:
    """The specialisation of positional ``args`` (module docstring); the indices in
    ``constexprs`` are keyed by value. Raises TypeError for argument types it does not
    know (such a call takes the JIT path)."""
    out: list[Any] = []
    for i, a in enumerate(args):
        if i in constexprs:
            hash(a)  # TypeError when unhashable
            out.append(("c", a))
        elif isinstance(a, torch.Tensor):
            out.append((a.dtype, a.data_ptr() % 16 == 0))
        elif isinstance(a, bool):
            out.append(("b",))
        elif isinstance(a, int):
            out.append(("i", a == 1, a % 16 == 0, -_I32 <= a < _I32, a >= 1 << 63))
        elif isinstance(a, float):
            out.append(("f",))
        elif a is None:
            out.append(None)
        else:
            raise TypeError(f"no cached launch for a {type(a).__name__} argument")
    return tuple(out)


class CachedLaunch:
    """A ``@triton.jit`` kernel launched through the ``CompiledKernel`` of its first JIT
    launch per specialisation (module docstring). ``launcher[grid](*args, **meta)`` like
    ``kernel[grid](...)``; returns the ``CompiledKernel``. ``hits`` / ``misses`` count
    direct and JIT launches."""

    def __init__(self, kernel: Any) -> None:
        from triton.runtime.jit import JITFunction

        if not isinstance(kernel, JITFunction):
            raise TypeError(
                f"CachedLaunch needs a @triton.jit function, got {type(kernel).__name__} "
                "(autotune / heuristics wrappers add constexprs it cannot see)"
            )
        self.kernel = kernel
        params = inspect.signature(kernel.fn).parameters
        self.names = list(params)
        self.defaults = {
            n: p.default for n, p in params.items() if p.default is not inspect.Parameter.empty
        }
        self.constexprs = frozenset(int(i) for i in getattr(kernel, "constexprs", ()))
        self.compiled: dict[tuple[Any, ...], tuple[Any, tuple[Any, ...]]] = {}
        self.hits = 0
        self.misses = 0

    def __getitem__(self, grid: Any) -> Callable[..., Any]:
        return functools.partial(self.launch, grid)

    def _tail(self, n_args: int, kwargs: dict[str, Any]) -> tuple[Any, ...]:
        """The parameters after the positional ones, from ``kwargs`` or their defaults, in
        signature order (the launcher takes every parameter, constexprs included)."""
        return tuple(
            kwargs[name] if name in kwargs else self.defaults[name] for name in self.names[n_args:]
        )

    def launch(self, grid: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            key = (_device(), spec_key(args, self.constexprs), tuple(kwargs.items()))
            hit = self.compiled.get(key)
        except TypeError:  # an argument type without a cached path, or unhashable meta
            self.misses += 1
            return self.kernel[grid](*args, **kwargs)
        if hit is None:
            self.misses += 1
            compiled = self.kernel[grid](*args, **kwargs)  # compiles once, launches
            if compiled is not None:
                self.compiled[key] = (compiled, self._tail(len(args), kwargs))
            return compiled
        self.hits += 1
        compiled, tail = hit
        full = (*args, *tail)
        if callable(grid):
            grid = grid(dict(zip(self.names, full, strict=True)))
        gx = grid[0]
        gy = grid[1] if len(grid) > 1 else 1
        gz = grid[2] if len(grid) > 2 else 1
        stream = _stream(key[0])
        enter, leave = _hooks()
        meta = compiled.launch_metadata(grid, stream, *full) if enter is not None else None
        compiled.run(
            gx, gy, gz, stream, compiled.function, compiled.packed_metadata, meta, enter, leave,
            *full,
        )  # fmt: skip
        return compiled


def _hooks() -> tuple[Any, Any]:
    """Triton's launch enter / exit hooks (a profiler's), None for an unset or empty hook
    chain (Triton 3.8 always has a ``HookChain``): no launch metadata to build per call."""
    from triton import knobs

    hooks = knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook
    enter, leave = (None if h is None or not getattr(h, "calls", True) else h for h in hooks)
    return enter, leave


def _device() -> int:
    return int(torch._C._cuda_getDevice())


def _stream(device: int) -> int:
    return int(torch._C._cuda_getCurrentRawStream(device))


def cached(kernel: Any) -> CachedLaunch:
    """``CachedLaunch(kernel)``, usable as a decorator under ``@triton.jit``."""
    return CachedLaunch(kernel)
