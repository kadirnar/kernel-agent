"""Helion kernels in kernel-agent (issue #229): whether Helion runs here, and tuning a
candidate's Helion kernels with Helion's own autotuner under the GPU lease.

Helion (https://pypi.org/project/helion, "PyTorch with tiles") compiles a ``@helion.kernel``
to Triton and searches a space it derives from the kernel itself: block sizes, loop orders,
L2 grouping, pointer / block_ptr / TMA indexing, persistent program ids, reduction loops,
eviction hints, warps and stages (its blog: one matmul searched 1,520 configs in 586 s). A
candidate writes each kernel body as a plain function and makes its kernel in ``build``
(:func:`kernel`), so every built candidate has its own kernel and config, an evaluation never
tunes (``autotune_effort="none"``: Helion's default config) and the tuned configs come in as
a ``build`` keyword argument::

    def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor: ...

    def build(reference, helion_configs=None):
        return HelionRMSNorm(reference, helion_tune.kernel(rmsnorm, helion_configs))

* :func:`status`: whether Helion can run with this process's torch and Triton (``doctor``'s
  ``helion`` probe; the backend is skipped with the reason otherwise): installed, every
  requirement it declares on torch / Triton met by the installed versions (installing Helion
  must never change torch: ``doctor`` refuses a mismatch instead), importable.
* :func:`kernels`: the ``helion.Kernel`` objects of a module or a built candidate.
* :func:`autotune` (``sweep_candidate(..., strategy="helion")``, :mod:`kernels.sweep`): one
  call of the built candidate records the arguments of each Helion kernel it launches, then
  Helion's autotuner runs on each (``Kernel.autotune(args, force=True)``) with the sweep's
  remaining time as its budget (``autotune_budget_seconds``), a seed, earlier tuned configs
  as seeds (``autotune_seed_configs``) and no warp specialisation where
  :func:`kernels.search.arch_reason` refuses it. It returns ``{kernel name: config}``; the
  sweep checks and times that config next to the default one and binds the faster into the
  snapshot (``_KA_SWEEP_CONFIG = {"helion_configs": ...}``), so integration and export never
  tune again.

Nothing here imports Helion at module level: without it, :func:`status` says why.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

#: The ``build`` keyword argument that carries tuned Helion configs (by kernel name)
KEYWORD = "helion_configs"
#: What doctor and the sweep say when Helion is missing
MISSING = (
    "helion is not installed (pip install helion: it declares no torch requirement, so it "
    "keeps this torch; `kernel-agent doctor` checks that)"
)


def _requirements() -> list[str]:
    try:
        return list(importlib.metadata.requires("helion") or [])
    except importlib.metadata.PackageNotFoundError:
        return []


def mismatches(requirements: Iterable[str], installed: Mapping[str, str | None]) -> list[str]:
    """The requirements (``Requires-Dist`` lines) on an ``installed`` distribution that its
    version does not meet (extras' requirements ignored)."""
    try:
        from packaging.requirements import Requirement
        from packaging.version import Version
    except ImportError:  # packaging ships with pip and setuptools; without it nothing is checked
        return []
    out = []
    for line in requirements:
        try:
            req = Requirement(line)
        except Exception:
            continue
        if req.marker is not None and "extra" in str(req.marker):
            continue
        have = installed.get(req.name.lower())
        if have is None or not str(req.specifier):
            continue
        base = Version(Version(have.split("+")[0]).base_version)
        if not req.specifier.contains(base, prereleases=True):
            out.append(f"helion requires {req}, this environment has {req.name} {have}")
    return out


def status() -> dict[str, Any]:
    """Whether Helion can run here: ``{"ok": bool, "version", "why"}`` (``why``: what is
    missing or refused). Imports Helion (no GPU work)."""
    try:
        version = importlib.metadata.version("helion")
    except importlib.metadata.PackageNotFoundError:
        return {"ok": False, "version": None, "why": MISSING}
    installed: dict[str, str | None] = {}
    for name in ("torch", "triton"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            installed[name] = importlib.metadata.version(name)
    if bad := mismatches(_requirements(), installed):
        why = "; ".join(bad) + " (refused: installing Helion must not change torch / Triton)"
        return {"ok": False, "version": version, "why": why}
    try:
        import helion  # noqa: F401
    except Exception as exc:
        return {"ok": False, "version": version, "why": f"import helion: {exc!r}"[:300]}
    return {"ok": True, "version": version, "why": None}


def is_kernel(obj: Any) -> bool:
    """Whether ``obj`` is a ``helion.Kernel`` (by its type's module: Helion is not imported)."""
    cls = type(obj)
    return cls.__module__.startswith("helion") and callable(getattr(obj, "autotune", None))


def kernels(*namespaces: Mapping[str, Any]) -> list[Any]:
    """The Helion kernels among the values of ``namespaces`` (a module's ``vars``, a built
    candidate's), once each."""
    out: list[Any] = []
    for namespace in namespaces:
        for obj in namespace.values():
            if is_kernel(obj) and not any(obj is k for k in out):
                out.append(obj)
    return out


def name_of(kernel: Any) -> str:
    return str(
        getattr(kernel, "name", None) or getattr(getattr(kernel, "fn", None), "__name__", "")
    )


def kernel(
    fn: Callable[..., Any], configs: Mapping[str, Any] | None = None, **settings: Any
) -> Any:
    """A ``helion.Kernel`` of ``fn`` for one built candidate: the config ``configs`` has for
    ``fn``'s name (``build``'s :data:`KEYWORD`: a dict of ``helion.Config`` fields), else
    Helion's default config; it never autotunes inside an evaluation (``autotune_effort=
    "none"``). ``settings``: more ``helion.kernel`` settings (``static_shapes`` defaults to
    True: one compiled kernel per shape)."""
    import helion

    found = (configs or {}).get(fn.__name__)
    options: dict[str, Any] = {"static_shapes": True, "autotune_effort": "none", **settings}
    if found is not None:
        options["config"] = helion.Config(**dict(found))
    return helion.kernel(fn, **options)


@contextlib.contextmanager
def _recording(found: list[Any]) -> Iterator[dict[int, tuple[Any, tuple[Any, ...]]]]:
    """The first call's arguments of each kernel of ``found`` while the block runs (their
    class's ``__call__`` wrapped, restored after)."""
    calls: dict[int, tuple[Any, tuple[Any, ...]]] = {}
    classes = {type(k): type(k).__call__ for k in found}

    def recorder(original: Callable[..., Any]) -> Callable[..., Any]:
        def call(self: Any, *args: Any, **kwargs: Any) -> Any:
            if id(self) not in calls:
                normal = self.normalize_args(*args, **kwargs) if kwargs else tuple(args)
                calls[id(self)] = (self, normal)
            return original(self, *args, **kwargs)

        return call

    try:
        for cls, original in classes.items():
            cls.__call__ = recorder(original)
        yield calls
    finally:
        for cls, original in classes.items():
            cls.__call__ = original


def _plain(config: Any) -> dict[str, Any]:
    """A ``helion.Config`` (a mapping) as plain JSON data."""
    data = getattr(config, "config", None)
    return dict(json.loads(json.dumps(dict(data if isinstance(data, Mapping) else config))))


#: The Helion settings :func:`autotune` sets for its search (restored after)
_TUNING = (
    "autotune_effort",
    "autotune_budget_seconds",
    "autotune_random_seed",
    "autotune_progress_bar",
    "allow_warp_specialize",
    "autotune_seed_configs",
)


def autotune(
    found: list[Any],
    call: Callable[[], Any],
    *,
    seconds: float | None = None,
    seed: int = 0,
    effort: str = "full",
    warp_specialize: bool = True,
    seeds: Mapping[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Tune the Helion kernels ``found`` (:func:`kernels`; module docstring) on the arguments
    ``call`` (one call of the built candidate) gives them: ``{"configs": {name: config},
    "kernels": [names], "seconds"}``, or ``{"error"}``. ``seconds``: the whole budget, shared
    by the kernels; ``seeds``: earlier configs per kernel name to start from."""
    if not found:
        return {"error": "no Helion kernel on the built candidate or at its module level"}
    with _recording(found) as calls:
        call()
    if not calls:
        return {"error": "the candidate's call launched none of its Helion kernels"}
    began = time.monotonic()
    configs: dict[str, Any] = {}
    for tuned, args in calls.values():
        settings = tuned.settings
        saved = {n: getattr(settings, n) for n in _TUNING if hasattr(settings, n)}
        left = None if seconds is None else seconds - (time.monotonic() - began)
        share = None if left is None else max(1, int(left / max(len(calls) - len(configs), 1)))
        wanted = {
            "autotune_effort": effort,
            "autotune_budget_seconds": share,
            "autotune_random_seed": seed,
            "autotune_progress_bar": False,
            "allow_warp_specialize": bool(saved.get("allow_warp_specialize", True))
            and warp_specialize,
        }
        starts = (seeds or {}).get(name_of(tuned)) or []
        try:
            for n, v in wanted.items():
                if n in saved:
                    setattr(settings, n, v)
            if starts and "autotune_seed_configs" in saved:
                import helion

                settings.autotune_seed_configs = [helion.Config(**dict(c)) for c in starts]
            try:
                config = tuned.autotune(args, force=True)
            except Exception:
                if not starts:
                    raise
                settings.autotune_seed_configs = saved.get("autotune_seed_configs")
                config = tuned.autotune(args, force=True)  # a seed it refused: without them
        finally:
            for n, v in saved.items():
                setattr(settings, n, v)
            # Helion's autotune leaves the tuned kernel running what it found: the candidate
            # it belongs to is timed as the default config next to the tuned one
            if callable(reset := getattr(tuned, "reset", None)):
                reset()
        configs[name_of(tuned)] = _plain(config)
    return {
        "configs": configs,
        "kernels": sorted(configs),
        "seconds": round(time.monotonic() - began, 1),
    }
