"""The library scout's adapter registry (issue #227).

An adapter maps one op family of a reference (:mod:`kernel_agent.libscout.detect`) onto a
library's kernels:

* **detector**: the families it serves (:attr:`Adapter.families`) and :meth:`Adapter.sites_ok`,
  what a call site needs (dtype, head size, no mask, ...);
* **availability probe** (:func:`availability`): the package is installed
  (``importlib.util.find_spec``), its version, and the library's own check
  (:meth:`Adapter.check`: an import that fails on an incompatible wheel, an attribute the
  template needs). An adapter whose package is missing, broken or too old is skipped with
  the reason;
* **architectures and dtypes** it declares (:attr:`Adapter.archs`, ``gpu_arch.supports``
  syntax: ``sm_80+``, ``sm_90``, ``sm_12x``; :attr:`Adapter.dtypes`), checked against the GPU
  at hand: FlashAttention 3 needs sm_90, QuACK lists Hopper / Blackwell / RTX 50;
* **candidate template** (:meth:`Adapter.code`: the op functions and the graph rewrite of a
  scout candidate, rendered by :mod:`kernel_agent.libscout.template`) and the configs the
  scout sweeps (:meth:`Adapter.configs`).

:func:`applicable` decides, from the detected families, the GPU, the target's precision and
the probes, which adapters run on a target and why each other one does not. Nothing here
imports a library: the probes run in the scout's GPU subprocess and in ``doctor``.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from kernel_agent import gpu_arch

#: Where a missing library comes from (``pip install kernel-agent[libs]`` holds the ones
#: that ship wheels; the rest need their own build).
INSTALL_HINT = "pip install 'kernel-agent[libs]'"


@dataclass(frozen=True)
class Availability:
    """Whether an adapter's library can be used here: its ``version`` (None: not installed)
    and ``reason`` (None: usable)."""

    adapter: str
    package: str
    version: str | None
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.reason is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "package": self.package,
            "version": self.version,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Availability:
        return cls(
            str(data.get("adapter")),
            str(data.get("package")),
            data.get("version"),
            data.get("reason"),
        )


@dataclass(frozen=True)
class Site:
    """The detected call sites of one family on a target (``detect.summary`` form)."""

    family: str
    sites: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Adapter:
    """One library's kernels for some op families (see the module docstring)."""

    name: str  # registry key, also in the candidate's file name: ``libscout_<name>.py``
    title: str  # what it does, for the ledger's hypothesis and the digest
    package: str  # the distribution (``pip install <package>``, requirements.txt)
    module: str  # what it imports
    licence: str
    families: tuple[str, ...]
    archs: str = ""  # gpu_arch.supports spec; "" = every GPU torch runs on
    archs_why: str = ""
    dtypes: tuple[str, ...] = ("bfloat16", "float16")
    #: target precisions it serves: "exact" (the reference's own precision: every target,
    #: whatever its tier) or reduced ones (``fp8_w8a8``: only targets planned at one of them)
    precisions: tuple[str, ...] = ("exact",)
    #: run on a GPU in kernel-agent's tests; False: written from the library's documentation
    #: (its library is not installed in the development environment), checked statically
    verified: bool = True
    #: builds a CUDA extension (``load_inline``): needs a CUDA toolkit; no op bars
    compiles: bool = False
    #: minimum package version the template was written against
    min_version: str | None = None
    #: helpers of :mod:`kernel_agent.libscout.fx_rewrites` its code calls
    helpers: tuple[str, ...] = ("swap_calls",)
    #: no template yet: listed and probed, never run (the reason)
    no_template: str | None = None

    #: subclasses: the candidate's op code (see :meth:`code`)
    CODE: ClassVar[str] = ""

    # ---------------------------------------------------------------- probes

    def check(self, module: Any) -> str | None:
        """The library's own check after a successful import (None: usable)."""
        return None

    def probe(self) -> Availability:
        """Installed? Which version? Does it import and pass :meth:`check`? (Imports the
        library: call it in a GPU worker or ``doctor``, not the coordinator.)"""
        top = self.module.split(".")[0]
        try:
            spec = importlib.util.find_spec(top)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            return Availability(
                self.name, self.package, None, f"{self.package} is not installed ({INSTALL_HINT})"
            )
        version = package_version(self.package)
        if version and self.min_version and _older(version, self.min_version):
            return Availability(
                self.name,
                self.package,
                version,
                f"{self.package} {version} is older than the {self.min_version} its template needs",
            )
        try:
            module = importlib.import_module(self.module)
        except Exception as exc:  # an incompatible wheel (ABI, CUDA version) fails here
            why = f"{type(exc).__name__}: {exc}".splitlines()[0][:200]
            return Availability(self.name, self.package, version, f"import failed: {why}")
        try:
            problem = self.check(module)
        except Exception as exc:
            problem = f"check failed: {type(exc).__name__}: {exc}"[:200]
        return Availability(self.name, self.package, version, problem)

    def arch_reason(self, capability: tuple[int, ...] | None) -> str | None:
        """Why this GPU cannot run it (None: it can, or every GPU can)."""
        if not self.archs:
            return None
        if capability is None:
            return "no GPU"
        if gpu_arch.supports(self.archs, capability):
            return None
        why = f" ({self.archs_why})" if self.archs_why else ""
        return f"needs {self.archs}{why}; this GPU is {gpu_arch.arch_of(capability)}"

    # ---------------------------------------------------------------- the target

    def sites_ok(self, family: str, site: Mapping[str, Any]) -> str | None:
        """Why the template cannot take this call site (None: it can). The base checks the
        dtype; subclasses add what their op needs."""
        dtype = site.get("dtype")
        if dtype and dtype != "?" and dtype not in self.dtypes:
            return f"{family} in {dtype} (takes {', '.join(self.dtypes)})"
        return None

    def configs(self, found: Mapping[str, Site], gpu: Mapping[str, Any]) -> list[dict[str, Any]]:
        """The ``build`` keyword arguments the scout sweeps (``gpu``: ``capability``,
        ``backends``)."""
        return [{}]

    def code(self) -> str:
        """The op code of a candidate: imports, the replacement functions, ``ops(**config)``
        (original callable → replacement: what op bars time) and ``rewrite(graph, **config)``
        (what the candidate's Dynamo backend runs on each FX graph; returns the nodes it
        changed)."""
        return self.CODE


def package_version(package: str) -> str | None:
    """The installed version of a distribution (None: not installed)."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _parts(version: str) -> tuple[int, ...]:
    out = []
    for piece in version.split("+")[0].split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


def _older(version: str, minimum: str) -> bool:
    return _parts(version) < _parts(minimum)


# ------------------------------------------------------------------ the decision


@dataclass(frozen=True)
class Decision:
    """One adapter on one target: ``run`` with ``configs``, or skipped for ``reason``."""

    adapter: Adapter
    reason: str | None
    configs: list[dict[str, Any]] = field(default_factory=list)
    sites: int = 0  # call sites it takes

    @property
    def run(self) -> bool:
        return self.reason is None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "adapter": self.adapter.name,
            "package": self.adapter.package,
            "families": list(self.adapter.families),
        }
        if self.reason is None:
            out.update(run=True, configs=self.configs, sites=self.sites)
        else:
            out.update(run=False, reason=self.reason)
        return out


def applicable(
    adapters: Iterable[Adapter],
    found: Mapping[str, Any],
    *,
    capability: tuple[int, ...] | None,
    precision: str | None,
    available: Mapping[str, Availability],
    backends: Mapping[str, bool] | None = None,
) -> list[Decision]:
    """Which ``adapters`` run on a target whose reference calls the families ``found``
    (``detect.summary`` form: ``{family: {"sites": [...]}}``), on a GPU of ``capability``, at
    the target's ``precision`` (None / ``exact``: the reference's), given the probes
    ``available`` (by adapter name) and the toolchain's ``backends`` (``cuda``: a CUDA
    toolkit for ``load_inline``). Only adapters of a family the target calls are listed."""
    target_precision = precision or "exact"
    decisions = []
    sites_of = {
        name: Site(name, list((fam or {}).get("sites") or []))
        for name, fam in found.items()
        if isinstance(fam, Mapping)
    }
    gpu = {"capability": capability, "backends": dict(backends or {})}
    for adapter in adapters:
        mine = [sites_of[f] for f in adapter.families if f in sites_of]
        if not mine:
            continue
        reason = _reason(adapter, mine, capability, target_precision, available, backends)
        if reason is not None:
            decisions.append(Decision(adapter, reason))
            continue
        taken = sum(1 for s in mine for site in s.sites if adapter.sites_ok(s.family, site) is None)
        if not taken:
            first = next(
                why for s in mine for site in s.sites if (why := adapter.sites_ok(s.family, site))
            )
            decisions.append(Decision(adapter, f"no call site it takes: {first}"))
            continue
        configs = adapter.configs(sites_of, gpu)
        if not configs:
            decisions.append(Decision(adapter, "no config applies on this GPU"))
            continue
        decisions.append(Decision(adapter, None, configs, taken))
    return decisions


def _reason(
    adapter: Adapter,
    mine: list[Site],
    capability: tuple[int, ...] | None,
    precision: str,
    available: Mapping[str, Availability],
    backends: Mapping[str, bool] | None,
) -> str | None:
    if adapter.no_template:
        return adapter.no_template
    # the reference's own precision passes every tier: an exact adapter runs on any target;
    # a reduced one only on targets planned at its precision
    if "exact" not in adapter.precisions and precision not in adapter.precisions:
        return f"serves {', '.join(adapter.precisions)} targets; this one is {precision}"
    if (why := adapter.arch_reason(capability)) is not None:
        return why
    probe = available.get(adapter.name)
    if probe is None:
        return "not probed"
    if not probe.ok:
        return probe.reason
    if adapter.compiles and backends is not None and not backends.get("cuda", False):
        return "builds a CUDA extension and no CUDA toolkit (nvcc) was found"
    return None


def availability(adapters: Iterable[Adapter]) -> dict[str, Availability]:
    """:meth:`Adapter.probe` of every adapter, by name."""
    return {a.name: a.probe() for a in adapters}
