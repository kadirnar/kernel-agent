"""CuTe DSL (``nvidia-cutlass-dsl``) toolchain: the install / architecture check of
``doctor`` and a compile cache for CuTe DSL kernels across evaluations.

**Check** (:func:`check`, static: imports and metadata, no GPU launch): the DSL version and
its library wheels, whether it imports, the architecture it compiles for
(``CUTE_DSL_ARCH``, else the GPU's ``sm_XYa``), whether that architecture is one it knows
and whether the block-scaled warp MMA (``MmaMXF8Op``: ``mma.sync kind::mxf8f6f4.block_scale``,
SASS ``QMMA.SF``, 416 TFLOP/s on an RTX 5070 Ti vs 208 for plain e4m3) admits it, TVM-FFI
(the low-overhead calling convention of the examples) and the compile cache.

**Compile cache** (:func:`compile_cached`). Every evaluation runs a candidate in a fresh
process, so ``cute.compile`` (seconds per kernel: tracing, MLIR passes, ``ptxas``) is paid
again on every evaluation and every sweep config. The cache keeps the compiled kernel as
the object file CuTe DSL exports (``JitCompiledFunction.export_to_c``: host launcher +
cubin) under ``<cache>/cute-dsl/<arch>/<key>/kernel.o`` and loads it back with
``cute.runtime.load_module`` (``enable_tvm_ffi`` when compiled with
``--enable-tvm-ffi``). The key is the sha256 of the kernel's source file, the
architecture, the DSL version, the compile options, the function's name and the caller's
specialisation key (dtypes, static shapes, tile config: everything ``cute.compile``
specialises on). Any failure to export or to load falls back to the freshly compiled
function (the cache is an optimisation, never a dependency); a corrupt entry is removed.
Each entry's ``meta.json`` records the measured compile time (``compile_s``), which
``doctor`` summarises. ``KERNEL_AGENT_CUTE_CACHE`` moves the cache (``off`` disables it).
"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import shutil
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kernel_agent.toolchain import CACHE_DIR

DIST = "nvidia-cutlass-dsl"
CACHE_ENV = "KERNEL_AGENT_CUTE_CACHE"
ARCH_ENV = "CUTE_DSL_ARCH"
TVM_FFI = "--enable-tvm-ffi"
OBJECT = "kernel.o"
META = "meta.json"


# ------------------------------------------------------------------ check (doctor)


def arch_name(capability: tuple[int, int] | None) -> str | None:
    """The architecture CuTe DSL compiles for on a GPU of ``capability`` (``sm_120a``:
    the ``a`` suffix from sm_90 on, as the DSL's own detection does); ``CUTE_DSL_ARCH``
    overrides it."""
    if env := os.environ.get(ARCH_ENV):
        return env
    if capability is None:
        return None
    major, minor = capability
    return f"sm_{major}{minor}{'a' if major >= 9 else ''}"


def _distributions() -> dict[str, str]:
    """Installed ``nvidia-cutlass-dsl*`` distributions → version."""
    out = {}
    for dist in importlib.metadata.distributions():
        name = str(dist.metadata.get("Name") or "")
        if name.lower().startswith(DIST):
            out[name.lower()] = dist.version
    return out


def _probe() -> dict[str, Any]:
    """What the installed DSL supports (imports only, no CUDA context)."""
    import cutlass
    import cutlass.cute as cute  # noqa: F401  (registers the export / load providers)
    from cutlass.base_dsl.enums import Arch

    info: dict[str, Any] = {
        "version": getattr(cutlass, "__version__", None),
        "arches": [a.name for a in Arch],
    }
    try:
        from cutlass.cute.nvgpu.warp.mma import MmaMXF8Op, MmaSM120BlockScaledOp

        info["mxf8"] = MmaMXF8Op.__name__
        info["block_scaled_arches"] = [a.name for a in MmaSM120BlockScaledOp.admissible_archs]
    except ImportError:
        info["mxf8"] = None
        info["block_scaled_arches"] = []
    return info


#: The CuTe DSL MMA that reaches the tensor-core peak, per architecture family
#: (``gpu_arch.Family.key``) where the warp-level block-scaled ``MmaMXF8Op`` does not apply.
FAMILY_MMA = {
    "pre_ampere": "warp MMA on fp16 only (no bf16 / FP8 tensor cores)",
    "ampere": "warp MMA (`MmaF16BF16Op`, mma.sync); no FP8 tensor cores",
    "ada": "warp MMA incl. FP8 (`MmaFP8Op`, the full FP8 rate here); no block-scaled MMA",
    "hopper": "warpgroup MMA (`cute.nvgpu.warpgroup`, wgmma) for the peak, FP8 included; "
    "no block-scaled MMA",
    "blackwell": "tcgen05 MMA (`cute.nvgpu.tcgen05`, block-scaled MXF8 / NVF4 included) for "
    "the peak; the warp-level `MmaMXF8Op` is sm_120-only",
    "blackwell_geforce": "no block-scaled MMA for this arch (FP8 only via `MmaFP8Op`)",
    "newer": "check the DSL's MMA ops for this arch",
}


@dataclass
class Status:
    """CuTe DSL install and support on this GPU (``doctor``)."""

    installed: bool
    version: str | None = None
    libs: list[str] = field(default_factory=list)  # e.g. ["cu12"]
    import_error: str | None = None
    arch: str | None = None
    arch_known: bool | None = None  # the DSL can target ``arch``
    block_scaled: bool | None = None  # MmaMXF8Op (QMMA.SF) admits ``arch``
    tvm_ffi: bool = False
    cache: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.installed and self.import_error is None and self.arch_known is not False

    def describe(self) -> list[str]:
        if not self.installed:
            return [f"CuTe DSL: not installed (`pip install {DIST}`): the `cute` backend is off"]
        libs = f", libs {'/'.join(self.libs)}" if self.libs else ""
        ffi = "yes" if self.tvm_ffi else "no"
        head = f"CuTe DSL {self.version or '?'} ({DIST}{libs}; TVM-FFI {ffi})"
        if self.import_error:
            return [f"{head}: import FAILED: {self.import_error}", *self._tail()]
        if self.arch is None:
            return [f"{head}: no GPU, architecture not checked", *self._tail()]
        if not self.arch_known:
            return [f"{head}: {self.arch} is not a target of this DSL version", *self._tail()]
        if self.block_scaled:
            mma = "block-scaled MMA `MmaMXF8Op` (QMMA.SF, full-rate FP8 with fp32 accumulation)"
        else:
            from kernel_agent.gpu_arch import capability_of, family

            fam = family(capability_of(self.arch))
            mma = FAMILY_MMA.get(fam.key if fam else "", FAMILY_MMA["blackwell_geforce"])
        return [f"{head}: {self.arch} supported, {mma}", *self._tail()]

    def _tail(self) -> list[str]:
        lines = []
        if self.cache:
            c = self.cache
            timing = (
                f", median compile {c['median_compile_s']:.1f} s"
                if c.get("median_compile_s") is not None
                else ""
            )
            lines.append(f"  compile cache: {c['root']} ({c['entries']} kernels{timing})")
        lines += [f"  note: {n}" for n in self.notes]
        return lines


def check(
    capability: tuple[int, int] | None,
    *,
    probe: Callable[[], dict[str, Any]] | None = None,
    installed: bool | None = None,
    distributions: Callable[[], dict[str, str]] = _distributions,
) -> Status:
    """Static check of the CuTe DSL install for a GPU of ``capability`` (None: no GPU).
    ``probe`` / ``installed`` / ``distributions`` replace the real imports (tests)."""
    if installed is None:
        installed = importlib.util.find_spec("cutlass") is not None
    if not installed:
        return Status(installed=False)
    dists = distributions()
    status = Status(
        installed=True,
        version=dists.get(DIST),
        libs=sorted(
            name.removeprefix(f"{DIST}-libs-")
            for name in dists
            if name.startswith(f"{DIST}-libs-") and not name.endswith(("-base", "-core"))
        ),
        tvm_ffi=importlib.util.find_spec("tvm_ffi") is not None,
        cache=cache_summary(),
    )
    try:
        info = (probe or _probe)()
    except Exception as exc:  # a broken install: say why, never crash doctor
        status.import_error = f"{type(exc).__name__}: {exc}"[:300]
        return status
    status.version = status.version or info.get("version")
    status.arch = arch_name(capability)
    if status.arch is not None:
        status.arch_known = status.arch in (info.get("arches") or [])
        status.block_scaled = bool(info.get("mxf8")) and status.arch in (
            info.get("block_scaled_arches") or []
        )
    if capability is not None and os.environ.get(ARCH_ENV):
        gpu = f"sm_{capability[0]}{capability[1]}"
        if not status.arch.startswith(gpu):  # type: ignore[union-attr]
            status.notes.append(f"{ARCH_ENV}={status.arch} differs from the GPU ({gpu})")
    if status.arch and status.arch.startswith("sm_12") and status.block_scaled is False:
        status.notes.append(
            "compute-bound FP8 GEMMs need the block-scaled MMA on sm_120 (plain e4m3 "
            "`mma.sync` runs at half rate): upgrade nvidia-cutlass-dsl (MmaMXF8Op)"
        )
    if not status.tvm_ffi:
        status.notes.append("apache-tvm-ffi missing: `--enable-tvm-ffi` (low host overhead) is off")
    return status


# ------------------------------------------------------------------ compile cache


def cache_root() -> Path | None:
    """The compile cache directory (None: disabled with ``KERNEL_AGENT_CUTE_CACHE=off``)."""
    env = os.environ.get(CACHE_ENV)
    if env is not None and env.strip().lower() in ("", "0", "off", "false", "no"):
        return None
    return Path(env).expanduser() if env else CACHE_DIR / "cute-dsl"


@functools.cache
def dsl_version() -> str:
    try:
        return importlib.metadata.version(DIST)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def source_digest(fn: Any) -> str:
    """sha256 of the source file that defines ``fn`` (a function, a bound method or a
    callable object): an edit anywhere in the file (helpers, constants) changes it."""
    target = getattr(fn, "__func__", fn)
    if not (inspect.isfunction(target) or inspect.ismethod(target)):
        target = type(fn)
    target = inspect.unwrap(target)
    try:
        path = inspect.getsourcefile(target)
        if path and Path(path).is_file():
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except TypeError:
        pass
    try:
        return hashlib.sha256(inspect.getsource(target).encode()).hexdigest()
    except (OSError, TypeError):
        return hashlib.sha256(repr(target).encode()).hexdigest()


def cache_key(*, source: str, arch: str, version: str, options: str, name: str, key: Any) -> str:
    """The entry name of a compiled kernel (32 hex digits)."""
    blob = json.dumps([source, arch, version, options, name, repr(key)])
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def device_arch() -> str:
    """The architecture CuTe DSL compiles for in this process (``CUTE_DSL_ARCH`` or the
    current CUDA device's)."""
    if env := os.environ.get(ARCH_ENV):
        return env
    import torch

    return arch_name(tuple(torch.cuda.get_device_capability())) or "unknown"  # type: ignore[arg-type]


@dataclass
class Stats:
    """This process's cache traffic (``compile_s``: time spent compiling on misses)."""

    hits: int = 0
    misses: int = 0
    compile_s: float = 0.0
    load_s: float = 0.0
    errors: list[str] = field(default_factory=list)


STATS = Stats()


def _compile(fn: Any, *args: Any, options: str = "") -> Any:
    import cutlass.cute as cute

    return cute.compile(fn, *args, options=options) if options else cute.compile(fn, *args)


def _export(compiled: Any, directory: Path, prefix: str, tvm_ffi: bool) -> None:
    """Write ``directory/kernel.o``: a TVM-FFI function exports its ``__tvm_ffi_<prefix>``
    symbol to an object file; a plain one a header + object with the ``prefix`` symbols."""
    if tvm_ffi:
        compiled.export_to_c(str(directory / OBJECT), function_name=prefix)
    else:
        compiled.export_to_c(str(directory), Path(OBJECT).stem, function_prefix=prefix)


def _load(obj: Path, prefix: str, tvm_ffi: bool) -> Any:
    from cutlass.cute.runtime import load_module

    return getattr(load_module(str(obj), enable_tvm_ffi=tvm_ffi), prefix)


def compile_cached(
    fn: Any,
    *args: Any,
    key: Any,
    options: str = "",
    name: str | None = None,
    source: str | None = None,
    arch: str | None = None,
    compile: Callable[..., Any] = _compile,
    export: Callable[[Any, Path, str, bool], None] = _export,
    load: Callable[[Path, str, bool], Any] = _load,
) -> Any:
    """``cute.compile(fn, *args, options=options)``, through the on-disk cache.

    ``key`` must name everything the compiled code specialises on besides the source:
    dtypes, static shapes, tile sizes, flags (``mark_layout_dynamic`` dimensions are not
    part of it). ``source`` replaces :func:`source_digest` (e.g. a digest of generated
    code). ``compile`` / ``export`` / ``load`` replace the CuTe DSL calls (tests)."""
    root = cache_root()
    if root is None:
        return _timed_compile(compile, fn, args, options)
    name = name or getattr(fn, "__qualname__", None) or type(fn).__qualname__
    arch = arch or device_arch()
    digest = cache_key(
        source=source or source_digest(fn),
        arch=arch,
        version=dsl_version(),
        options=options,
        name=str(name),
        key=key,
    )
    entry = root / arch / digest
    prefix = f"ka_{digest[:24]}"
    obj = entry / OBJECT
    if obj.is_file():
        start = time.perf_counter()
        try:
            loaded = load(obj, prefix, TVM_FFI in options)
        except Exception as exc:  # stale or corrupt: compile again, replace it
            STATS.errors.append(f"load {entry.name}: {type(exc).__name__}: {exc}"[:300])
            shutil.rmtree(entry, ignore_errors=True)
        else:
            STATS.hits += 1
            STATS.load_s += time.perf_counter() - start
            return loaded
    compiled, seconds = _timed_compile(compile, fn, args, options, timed=True)
    tmp = entry.with_name(f"{entry.name}.tmp{os.getpid()}")
    try:
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        export(compiled, tmp, prefix, TVM_FFI in options)
        meta = {
            "name": name,
            "key": repr(key),
            "arch": arch,
            "version": dsl_version(),
            "options": options,
            "compile_s": round(seconds, 3),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (tmp / META).write_text(json.dumps(meta, indent=2))
        if not (tmp / OBJECT).is_file():
            raise FileNotFoundError(f"export wrote no {OBJECT}")
        if entry.exists():  # another process stored it meanwhile
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            tmp.replace(entry)
    except Exception as exc:  # the cache is an optimisation: keep the compiled function
        STATS.errors.append(f"export {entry.name}: {type(exc).__name__}: {exc}"[:300])
        shutil.rmtree(tmp, ignore_errors=True)
    return compiled


def _timed_compile(
    compile: Callable[..., Any], fn: Any, args: tuple[Any, ...], options: str, timed: bool = False
) -> Any:
    start = time.perf_counter()
    compiled = compile(fn, *args, options=options)
    seconds = time.perf_counter() - start
    STATS.misses += 1
    STATS.compile_s += seconds
    return (compiled, seconds) if timed else compiled


def cache_entries(root: Path | None = None) -> list[dict[str, Any]]:
    """``meta.json`` of every complete cache entry (``path`` added)."""
    root = cache_root() if root is None else root
    if root is None or not root.is_dir():
        return []
    out = []
    for meta in sorted(root.glob(f"*/*/{META}")):
        if not (meta.parent / OBJECT).is_file():
            continue
        try:
            data = json.loads(meta.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            out.append({**data, "path": str(meta.parent)})
    return out


def cache_summary(root: Path | None = None) -> dict[str, Any]:
    """Entries and median compile time of the cache (``doctor``)."""
    root = cache_root() if root is None else root
    if root is None:
        return {}
    found = cache_entries(root)
    times = [float(e["compile_s"]) for e in found if isinstance(e.get("compile_s"), int | float)]
    return {
        "root": str(root),
        "entries": len(found),
        "median_compile_s": statistics.median(times) if times else None,
    }


def clear_cache(root: Path | None = None) -> int:
    """Remove every entry; returns how many there were."""
    root = cache_root() if root is None else root
    if root is None or not root.is_dir():
        return 0
    count = len(cache_entries(root))
    shutil.rmtree(root, ignore_errors=True)
    return count
