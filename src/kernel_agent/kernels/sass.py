"""SASS opcode census: the instructions a candidate's kernels actually issue (#230).

Registers, spills and ncu metrics say how a kernel runs; its SASS says what it was
compiled to. Which tensor-core instruction a GEMM issues decides its peak (on sm_120 a
Triton ``tl.dot`` on e4m3 lowers to ``QMMA.16832.F32`` at half the rate of the
block-scaled ``QMMA.SF`` that ``tl.dot_scaled`` lowers to: docs/RESEARCH-TRITON.md §1), and
local-memory traffic, narrow loads or missing async copies show in the opcodes before any
profiler runs. ``evaluate_candidate(profile=true)`` (and ``"ncu"``) adds :func:`census`:

* :func:`binaries` collects the candidate process's cubins: Triton's compiled kernels
  (``CompiledKernel.asm["cubin"]``, the candidate's own and those Inductor compiled for
  it), NVRTC kernels (the cubins :func:`kernel_agent.toolchain.nvrtc_kernels` compiled,
  :func:`kept` while their kernels live; a ``cuda.core`` ``ObjectCode`` of code type
  ``cubin`` still alive: ``Program(...).compile("cubin").get_kernel(...)`` frees it, so a
  candidate compiling without the helper keeps it in a global, else its kernel is listed
  under ``missing``), TileLang's ``JITKernel`` objects (:func:`_tilelang`: the cubin of
  their adapter's CUDA module or library file), the shared objects built on this machine
  and mapped into the process (``load_inline``, native projects: ``cuobjdump`` reads the
  ``.so``), and CuTe DSL's compiled functions (the fat binary their lowered IR module
  embeds).
* :func:`disassemble` runs ``cuobjdump -sass`` (the toolkit's or the one Triton bundles,
  :func:`find_cuobjdump`) on each; :func:`parse` reads its ``Function :`` blocks (and
  ``nvdisasm``'s ``.text.<kernel>:`` sections).
* :func:`summarise` counts each kernel's opcodes by :data:`CATEGORIES` (tensor-core MMA,
  global / shared / local memory, cp.async, TMA, tensor memory, barriers, shuffles,
  atomics, fp32 / fp16 math), keeps every tensor-core opcode whole (shape and types:
  ``QMMA.SF.16832.F32.E4M3.E4M3.E8``), the widths of its global loads and the most frequent
  opcodes raw (an opcode in no category is reported under its own name, never guessed into
  one). Counts are static (instructions in the binary), not executions. A tcgen05 MMA
  (sm_100) carries its block scaling as an operand, not as a modifier: a ``tmem[]`` scale
  operand after its instruction descriptor (``kind::mxf8f6f4`` / ``mxf4`` / ``mxf4nvf4``
  ``.block_scale``). The census marks such an MMA ``.SF`` like sm_120's own ``QMMA.SF`` /
  ``OMMA.SF`` (``UTCQMMA.SF``, ``UTCOMMA.SF.4X``; the ``.SF`` is the census's, the SASS
  shows the operand), so Triton's ``tl.dot_scaled`` (``UTCQMMA.SF``) and ``tl.dot``
  (``UTCQMMA``) differ there as on sm_120.

The tables name only opcodes found in cubins compiled on the CPU for sm_80, sm_89, sm_90a,
sm_100a and sm_120a (``tests/fixtures/sass``: ``kernels.cu`` and ``make_fixtures.py``, which
refreshes them after a CUDA upgrade). The census needs no GPU work, no ncu and no admin
counters; without ``cuobjdump`` it says so (``status: unavailable`` with the fix).
:mod:`kernel_agent.kernels.directives` turns it into short directives.
"""

from __future__ import annotations

import collections
import contextlib
import gc
import hashlib
import importlib
import re
import struct
import subprocess
import sys
import tempfile
import threading
import weakref
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Tensor-core opcodes (the base name before the first ``.``; the modifiers carry shape and
#: types) and what issues them, as compiled in ``tests/fixtures/sass``.
TENSOR_OPS: dict[str, str] = {
    "HMMA": "mma.sync fp16 / bf16 / tf32",
    "IMMA": "mma.sync int8",
    "QMMA": "mma.sync FP8 (sm_89, sm_12x; QMMA.SF: block-scaled, sm_12x)",
    "OMMA": "mma.sync block-scaled FP4 (sm_12x)",
    "HGMMA": "wgmma fp16 / bf16 / tf32 (sm_90a)",
    "QGMMA": "wgmma FP8 (sm_90a)",
    "IGMMA": "wgmma int8 (sm_90a)",
    "UTCHMMA": "tcgen05.mma kind::f16 / tf32 (sm_100a)",
    "UTCQMMA": "tcgen05.mma kind::f8f6f4: FP8 / FP6 / FP4 operands (UTCQMMA.SF: "
    "kind::mxf8f6f4.block_scale; sm_100a)",
    "UTCIMMA": "tcgen05.mma kind::i8 (sm_100a)",
    "UTCOMMA": "tcgen05.mma kind::mxf4 / mxf4nvf4, block-scaled FP4 (UTCOMMA.SF; .4X: "
    "scale_vec::4X; sm_100a)",
}
#: Category of each base opcode the fixtures contain (everything else: its own name).
CATEGORIES: dict[str, str] = {
    **dict.fromkeys(TENSOR_OPS, "tensor"),
    "LDG": "global_load",
    "STG": "global_store",
    "LDGSTS": "cp_async",  # cp.async global -> shared
    "LDGDEPBAR": "cp_async",  # cp.async.commit_group
    "UTMALDG": "tma",  # cp.async.bulk.tensor global -> shared
    "UTMASTG": "tma",  # cp.async.bulk.tensor shared -> global
    "UTMACMDFLUSH": "tma",  # cp.async.bulk.commit_group
    "UBLKCP": "tma",  # cp.async.bulk (no tensor map)
    "LDS": "shared_load",
    "LDSM": "shared_load",  # ldmatrix
    "STS": "shared_store",
    "STSM": "shared_store",  # stmatrix
    "LDL": "local",
    "STL": "local",
    "LDTM": "tensor_memory",  # tcgen05.ld
    "STTM": "tensor_memory",  # tcgen05.st
    "UTCATOMSWS": "tensor_memory",  # tcgen05.alloc / dealloc
    "BAR": "barrier",  # __syncthreads, named barriers
    "SYNCS": "barrier",  # mbarrier
    "DEPBAR": "barrier",  # cp.async.wait_group
    "WARPGROUP": "barrier",  # wgmma fence / wait
    "UTCBAR": "barrier",  # tcgen05.commit
    "SHFL": "shuffle",
    "RED": "atomic",  # global reduction without a result (sm_80, sm_89)
    "REDG": "atomic",  # the same from sm_90
    "ATOMG": "atomic",  # global atomic returning a value
    "ATOMS": "atomic",  # shared-memory atomic
    "FFMA": "fp32_math",
    "FADD": "fp32_math",
    "FMUL": "fp32_math",
    "HFMA2": "fp16_math",
    "HADD2": "fp16_math",
    "HMUL2": "fp16_math",
}
#: Opcodes left out of the counts: padding.
IGNORED = frozenset({"NOP"})
TOP_OPS = 8  # most frequent opcodes kept raw per kernel
MAX_BINARIES = 32  # cubins / files disassembled per evaluation
MAX_KERNELS = 16  # kernels kept in the census
MAX_MISSING = 5  # profiled kernels without SASS, named
TIMEOUT_S = 60.0
MAX_FILE_BYTES = 512 << 20  # a library file read for the cubins embedded in it
#: Why a kernel that ran has no SASS in the census (``missing``).
MISSING_NOTE = (
    "kernels that ran without SASS here (missing) are library kernels (torch, cuBLAS, "
    "cuDNN), NVRTC kernels compiled without kernel_agent.toolchain.nvrtc_kernels whose "
    "cuda.core ObjectCode was freed (compile through the helper, or keep the ObjectCode "
    "alive, e.g. in a module global, to include them) or PTX-only code"
)

_FUNCTION = re.compile(r"^\s*(?:Function\s*:\s*(?P<f>\S+)|\.text\.(?P<t>[^\s:]+):)")
_ARCH = re.compile(r"(?:code for|arch\s*=)\s*(sm_\w+)")
_INSTRUCTION = re.compile(
    r"^\s*/\*[0-9a-f]{4,}\*/\s+\{?\s*(?:@!?U?P(?:T|\d+)\s+)?"
    r"(?P<op>[A-Z][A-Z0-9_]*(?:\.[A-Za-z0-9_]+)*)"
)
#: A tcgen05 MMA (``UTCHMMA``, ``UTCQMMA``, ``UTCIMMA``, ``UTCOMMA``, with modifiers).
_TCGEN05_MMA = re.compile(r"UTC[A-Z]*MMA(?:\.|$)")
#: Its block-scale operand: a ``tmem[]`` operand after the instruction descriptor
#: (``UTCQMMA gdesc[UR12], gdesc[UR14], tmem[UR6], tmem[UR4], idesc[UR5], tmem[UR8], UPT``).
#: Counting ``tmem[]`` operands would not do: the A operand can be in tensor memory too
#: (``UTCQMMA tmem[UR7], gdesc[UR10], ...``), before the descriptor (tests/fixtures/sass).
_SCALE_OPERAND = re.compile(r"\bidesc\[[^\]]*\]\s*,\s*tmem\[")


# ------------------------------------------------------------------ parsing


@dataclass
class KernelSass:
    """One kernel's opcodes in one architecture's SASS."""

    symbol: str
    arch: str | None = None
    ops: collections.Counter[str] = field(default_factory=collections.Counter)


def parse(text: str) -> list[KernelSass]:
    """The kernels of ``cuobjdump -sass`` (``Function : <name>`` blocks under ``code for
    sm_XX``) or ``nvdisasm`` (``.text.<name>:`` sections) output, with every instruction's
    full opcode (predicates dropped) counted; a tcgen05 MMA with a block-scale operand is
    counted ``.SF`` (module docstring)."""
    out: list[KernelSass] = []
    arch: str | None = None
    current: KernelSass | None = None
    for line in text.splitlines():
        if match := _ARCH.search(line):
            arch = match.group(1)
            continue
        if match := _FUNCTION.match(line):
            current = KernelSass(match.group("f") or match.group("t"), arch)
            out.append(current)
            continue
        if current is not None and (match := _INSTRUCTION.match(line)):
            op = match.group("op")
            if _TCGEN05_MMA.match(op) and _SCALE_OPERAND.search(line, match.end()):
                name, _, mods = op.partition(".")
                op = f"{name}.SF" + (f".{mods}" if mods else "")
            current.ops[op] += 1
    return out


def base(opcode: str) -> str:
    """``LDG`` for ``LDG.E.128``."""
    return opcode.split(".", 1)[0]


def block_scaled(opcode: str) -> bool:
    """Whether a tensor-core opcode is a block-scaled MMA: ``QMMA.SF`` / ``OMMA.SF``
    (sm_12x ``mma.sync``) or a tcgen05 MMA :func:`parse` marked ``.SF`` (sm_100)."""
    return "SF" in opcode.split(".")[1:]


def load_bits(opcode: str) -> int:
    """Width of one thread's access of a load / store opcode: a numeric modifier (64, 128,
    256), 8 / 16 for the ``U8`` / ``S16`` forms, else 32."""
    mods = opcode.split(".")[1:]
    for mod in mods:
        if mod.isdigit() and int(mod) >= 64:
            return int(mod)
    for mod in mods:
        if mod in ("U8", "S8"):
            return 8
        if mod in ("U16", "S16"):
            return 16
    return 32


def fp8_unpacks(opcode: str) -> bool:
    """Whether ``opcode`` unpacks FP8 to fp16 (``F2FP.F16.E4M3.UNPACK_B``): how e4m3
    ``mma.sync`` is emulated where the GPU has no FP8 ``mma.sync`` (sm_90, sm_100)."""
    mods = opcode.split(".")
    return mods[0] == "F2FP" and "UNPACK" in opcode and bool({"E4M3", "E5M2"} & set(mods))


def short_name(symbol: str) -> str:
    """A readable kernel name: the identifiers of an Itanium-mangled C++ name
    (``_ZN2ns6kernelEv`` -> ``ns::kernel``, ``_Z6kernelPf`` -> ``kernel``), else
    ``symbol``."""
    if not symbol.startswith("_Z"):
        return symbol
    rest = symbol[2:]
    nested = rest.startswith("N")
    rest = rest[1:] if nested else rest
    parts = []
    while (match := re.match(r"(\d+)", rest)) is not None:
        n = int(match.group(1))
        start = len(match.group(1))
        parts.append(rest[start : start + n])
        rest = rest[start + n :]
        if not nested:
            break
    return "::".join(p for p in parts if p) or symbol


def summarise(kernel: KernelSass) -> dict[str, Any]:
    """The census row of one kernel (module docstring)."""
    categories: collections.Counter[str] = collections.Counter()
    bases: collections.Counter[str] = collections.Counter()
    tensor: dict[str, int] = {}
    widths: collections.Counter[str] = collections.Counter()
    local: collections.Counter[str] = collections.Counter()
    unpacks = 0
    for opcode, n in kernel.ops.items():
        name = base(opcode)
        if name in IGNORED:
            continue
        bases[name] += n
        category = CATEGORIES.get(name)
        if category is not None:
            categories[category] += n
        if category == "tensor":
            tensor[opcode] = tensor.get(opcode, 0) + n
        elif category == "local":
            local[name] += n
        elif name == "LDG":
            widths[str(load_bits(opcode))] += n
        if fp8_unpacks(opcode):
            unpacks += n
    row: dict[str, Any] = {"kernel": short_name(kernel.symbol)}
    if row["kernel"] != kernel.symbol:
        row["symbol"] = kernel.symbol[:200]
    row |= {
        "arch": kernel.arch,
        "instructions": sum(bases.values()),
        "categories": dict(categories.most_common()),
        "tensor": dict(sorted(tensor.items(), key=lambda kv: -kv[1])),
        "local": dict(sorted(local.items())),
        "global_load_bits": dict(sorted(widths.items(), key=lambda kv: int(kv[0]))),
        "top": dict(bases.most_common(TOP_OPS)),
    }
    if unpacks:
        row["fp8_unpacks"] = unpacks
    return {k: v for k, v in row.items() if v not in ({}, None)}


# ------------------------------------------------------------------ the cubins of a process


@dataclass
class Binary:
    """A cubin (``data``) or a file holding cubins (``path``: an extension ``.so``)."""

    source: str  # triton / nvrtc / cuda / cute / tilelang
    name: str
    data: bytes | None = None
    path: Path | None = None
    origin: Path | None = None  # the file a cubin of ``data`` was read from

    @property
    def key(self) -> str:
        if self.data is not None:
            return hashlib.sha256(self.data).hexdigest()
        return str(self.path)


def _type_is(obj: Any, name: str, module: str) -> bool:
    kind = type(obj)
    return kind.__name__ == name and str(kind.__module__).startswith(module)


def _triton(obj: Any) -> Binary | None:
    cubin = getattr(obj, "asm", {}).get("cubin") if hasattr(obj, "asm") else None
    if not isinstance(cubin, bytes | bytearray):
        return None
    meta = getattr(obj, "metadata", None)
    name = getattr(obj, "name", None) or getattr(meta, "name", None) or "triton"
    return Binary("triton", str(name), bytes(cubin))


def _nvrtc(obj: Any) -> Binary | None:
    try:
        if getattr(obj, "code_type", None) != "cubin":
            return None
        code = obj.code
    except Exception:  # an unloaded or foreign ObjectCode
        return None
    if isinstance(code, bytes | bytearray):
        return Binary("nvrtc", str(getattr(obj, "name", "") or "nvrtc"), bytes(code))
    return None


#: The cubins of NVRTC kernels compiled through :func:`kernel_agent.toolchain.nvrtc_kernels`,
#: by kernel: ``cuda.core`` frees the ``ObjectCode`` once ``get_kernel`` returns, and a
#: ``Kernel`` does not expose its cubin. Weak keys: an entry lives as long as its kernel, so
#: a candidate that drops its kernels drops their cubins too (several kernels of one
#: program share one ``bytes``).
_KEPT: weakref.WeakKeyDictionary[Any, tuple[str, bytes]] = weakref.WeakKeyDictionary()
_KEPT_LOCK = threading.Lock()


def keep(owner: Any, name: str, cubin: bytes) -> bool:
    """Keep ``cubin`` (holding kernel ``name``) for the census while ``owner`` (its
    ``cuda.core`` ``Kernel``) lives. False when ``owner`` takes no weak reference: nothing
    is kept then (never a leak; the census lists the kernel under ``missing``)."""
    try:
        with _KEPT_LOCK:
            _KEPT[owner] = (str(name), bytes(cubin))
    except TypeError:
        return False
    return True


def kept() -> list[Binary]:
    """The cubins :func:`keep` holds for kernels still alive."""
    with _KEPT_LOCK:
        entries = list(_KEPT.values())
    return [Binary("nvrtc", name, cubin) for name, cubin in entries]


#: The first bytes of a CUDA fat binary (0xBA55ED50) and of an ELF cubin.
_DEVICE_CODE = (b"P\xedU\xba", b"\x7fELF")
_EM_CUDA = 190  # the ELF machine of a cubin


def embedded_cubins(data: bytes, limit: int = MAX_BINARIES) -> list[bytes]:
    """The cubins stored raw in ``data`` (a file's bytes), at most ``limit``: TVM's module
    blob in TileLang's ``executable.so`` holds its kernels' cubin, which ``cuobjdump`` does
    not read ("does not contain device code"). A cubin is an ELF64 little-endian image of
    machine ``EM_CUDA``; it ends with its section or program header table, whichever is
    last (checked against the cubin TileLang caches beside the file)."""
    out: list[bytes] = []
    pos = data.find(b"\x7fELF")
    while pos >= 0 and len(out) < limit:
        head = data[pos : pos + 64]
        if len(head) == 64 and head[4] == 2 and head[5] == 1:  # ELF64, little-endian
            (machine,) = struct.unpack_from("<H", head, 18)
            phoff, shoff = struct.unpack_from("<QQ", head, 32)
            phentsize, phnum, shentsize, shnum = struct.unpack_from("<HHHH", head, 54)
            end = max(shoff + shentsize * shnum, phoff + phentsize * phnum)
            if machine == _EM_CUDA and 64 <= end <= len(data) - pos:
                out.append(data[pos : pos + end])
                pos = data.find(b"\x7fELF", pos + end)
                continue
        pos = data.find(b"\x7fELF", pos + 4)
    return out


def _tvm_modules(roots: Iterable[Any], depth: int = 3) -> Iterator[Any]:
    """TVM runtime modules ``roots`` and their imports, ``depth`` levels down."""
    for module in roots:
        if module is None:
            continue
        yield module
        if depth > 0:
            with contextlib.suppress(Exception):
                yield from _tvm_modules(list(module.imports), depth - 1)


def _tvm_cubin(module: Any) -> bytes | None:
    """The device code a TVM CUDA module holds: ``inspect_source("cubin")`` returns it as a
    string, which does not decode as UTF-8 (the bytes are the decode error's)."""
    for fmt in ("cubin", "fatbin"):
        try:
            data = str(module.inspect_source(fmt)).encode("utf-8", "surrogateescape")
        except UnicodeDecodeError as exc:
            data = bytes(exc.object)
        except Exception:
            continue
        if data[:4] in _DEVICE_CODE:  # else the module's CUDA source: another format
            return data
    return None


def _tilelang(obj: Any) -> list[Binary]:
    """The device code of a TileLang ``JITKernel`` (TileLang 0.1 internals: any failure
    leaves it out). Its kernels are in no ``.so`` cuobjdump reads: the tvm_ffi backend's
    cubin is in the CUDA module its adapter's runtime module (a fresh compile) or loaded
    library (TileLang's cache: ``libpath``) imports, and embedded raw in TVM's module blob
    of that ``executable.so``. So: the CUDA modules first, then the cubins embedded in the
    adapter's library file (``libpath``: also the nvrtc backend's ``.cubin``) or the cache
    entry's; a file without (the cython backend's nvcc-built ``.so``) goes to cuobjdump."""
    adapter = getattr(obj, "adapter", None)
    files: list[Path] = []
    with contextlib.suppress(Exception):
        if libpath := getattr(adapter, "libpath", None):
            files.append(Path(str(libpath)))
        if cache := getattr(obj, "_tilelang_cache_path", None):
            files += [Path(str(cache)) / n for n in ("executable.so", "kernel_lib.so")]
    roots = [getattr(adapter, attr, None) for attr in ("rt_mod", "executable")]
    found = [
        data
        for module in _tvm_modules(roots)
        if getattr(module, "kind", None) == "cuda" and (data := _tvm_cubin(module)) is not None
    ]
    if found:  # the loaded library's (libpath) when it came from one: no cuobjdump run on it
        origin = files[0] if files and getattr(adapter, "libpath", None) else None
        return [Binary("tilelang", "tilelang", data, origin=origin) for data in found]
    for path in files:
        try:
            if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
                continue
            data = path.read_bytes()
        except OSError:
            continue
        if cubins := embedded_cubins(data):
            return [Binary("tilelang", path.name, c, origin=path) for c in cubins]
        if data[:4] in _DEVICE_CODE:
            return [Binary("tilelang", path.name, path=path)]
    return []


def _is_cute(obj: Any) -> bool:
    kind = type(obj)
    return str(kind.__module__).startswith("cutlass") and any(
        c.__name__ == "JitCompiledFunction" for c in kind.__mro__
    )


def _cute(obj: Any) -> list[Binary]:
    """The device code of a CuTe DSL compiled function: its kept cubin (``CUTE_DSL_KEEP=
    cubin``), else the fat binary its lowered IR module embeds as a global (CuTe DSL 4.x
    internals: any failure leaves it out)."""
    name = str(getattr(obj, "function_name", None) or "cute")
    with contextlib.suppress(Exception):
        cubin = obj.__cubin__
        if isinstance(cubin, bytes | bytearray) and cubin:
            return [Binary("cute", name, bytes(cubin))]
    out: list[Binary] = []
    with contextlib.suppress(Exception):
        ir: Any = importlib.import_module("cutlass._mlir.ir")

        def visit(op: Any) -> Any:
            if op.name == "llvm.mlir.global" and "value" in op.attributes:
                with contextlib.suppress(Exception):
                    data = ir.StringAttr(op.attributes["value"]).value_bytes
                    if bytes(data[:4]) in _DEVICE_CODE:
                        out.append(Binary("cute", name, bytes(data)))
            return ir.WalkResult.ADVANCE

        obj.ir_module.operation.walk(visit)
    return out


def _built_here(path: str) -> bool:
    """Whether a shared object was built for this machine's kernels (``load_inline``,
    native projects, TileLang, ...): not shipped by Python packages or the system."""
    shipped = {sys.prefix, sys.base_prefix, sys.exec_prefix, "/usr", "/lib", "/lib64", "/opt"}
    if any(path.startswith(p.rstrip("/") + "/") for p in shipped if p):
        return False
    return "-packages/" not in path and "/.triton/" not in path  # Triton's launchers


def _extension_files() -> list[Path]:
    """Shared objects with this process's own kernels: the ``load_inline`` extensions
    (:func:`kernel_agent.kernels.ncu.extension_files`) and every shared object mapped into
    the process that :func:`_built_here` (Linux ``/proc/self/maps``: ``load_inline``
    modules are not in ``sys.modules``; TileLang loads its kernels with ``ctypes``)."""
    from kernel_agent.kernels.ncu import extension_files

    files = list(extension_files())
    with contextlib.suppress(OSError):
        for line in Path("/proc/self/maps").read_text().splitlines():
            parts = line.split(maxsplit=5)
            path = parts[5].strip() if len(parts) == 6 else ""
            if re.search(r"\.so(\.\d+)*$", path) and _built_here(path):
                files.append(Path(path))
    return sorted(set(files))


def binaries(namespace: Mapping[str, Any] | None = None) -> list[Binary]:
    """The cubins of this process's kernels (module docstring): the Triton kernels of
    ``namespace`` (a candidate module's globals) first, then those the garbage collector
    tracks (Triton ``CompiledKernel``, ``cuda.core`` ``ObjectCode``, CuTe DSL compiled
    functions, TileLang ``JITKernel``), the NVRTC cubins :func:`kept`, then the extension
    files (but those a TileLang library's cubins came from); each once."""
    from kernel_agent.kernels.ncu import _triton_kernels

    found: list[Binary] = []
    for obj in (namespace or {}).values():
        for kernel in _triton_kernels(obj):
            if (b := _triton(kernel)) is not None:
                found.append(b)
    for obj in gc.get_objects():
        if _type_is(obj, "CompiledKernel", "triton"):
            if (b := _triton(obj)) is not None:
                found.append(b)
        elif _type_is(obj, "ObjectCode", "cuda.core"):
            if (b := _nvrtc(obj)) is not None:
                found.append(b)
        elif _is_cute(obj):
            found += _cute(obj)
        elif _type_is(obj, "JITKernel", "tilelang"):
            found += _tilelang(obj)
    found += kept()
    read = {b.origin.resolve() for b in found if b.origin is not None}
    files = [p for p in _extension_files() if p.resolve() not in read]
    found += [Binary("cuda", p.name, path=p) for p in files]
    unique: dict[str, Binary] = {}
    for b in found:
        unique.setdefault(b.key, b)
    return list(unique.values())


# ------------------------------------------------------------------ disassembling


def find_cuobjdump(places: Iterable[Path] | None = None) -> str | None:
    """A ``cuobjdump`` that can disassemble (``places``: :func:`kernel_agent.kernels.ncu.
    cuobjdumps`, the toolkit's first, then Triton's). ``-sass`` runs ``nvdisasm`` from the
    tool's directory or ``PATH``, and a partial toolkit ships ``cuobjdump`` without it (CUDA
    12.9's on an A10 machine: "Could not find executable file 'nvdisasm'", and every kernel
    went to ``missing``): one with an ``nvdisasm`` beside it comes first (Triton bundles
    both), else the first (its ``nvdisasm`` from ``PATH``)."""
    if places is None:
        from kernel_agent.kernels.ncu import cuobjdumps

        places = cuobjdumps()
    found = list(places)
    paired = [p for p in found if (p.parent / "nvdisasm").is_file()]
    return str((paired or found)[0]) if found else None


def availability() -> dict[str, Any]:
    """``{"cuobjdump", "usable", "reason"}``: whether the census can run here (no GPU)."""
    tool = find_cuobjdump()
    if tool is None:
        return {
            "cuobjdump": None,
            "usable": False,
            "reason": "cuobjdump not found: install the CUDA toolkit (or the "
            "nvidia-cuda-cuobjdump wheel), or Triton, whose NVIDIA backend bundles one",
        }
    return {"cuobjdump": tool, "usable": True}


def disassemble(binary: Binary, tool: str, *, run: Any = subprocess.run) -> str:
    """``cuobjdump -sass`` of one :class:`Binary` ("" when it holds no SASS: PTX only)."""

    def sass(path: Path) -> str:
        proc = run([tool, "-sass", str(path)], capture_output=True, text=True, timeout=TIMEOUT_S)
        return proc.stdout or ""

    if binary.path is not None:
        return sass(binary.path)
    with tempfile.TemporaryDirectory(prefix="ka-sass-") as tmp:
        path = Path(tmp) / "kernel.cubin"
        path.write_bytes(binary.data or b"")
        return sass(path)


def _arch_rank(arch: str | None, capability: tuple[int, int] | None) -> int:
    """0 for SASS of the GPU's own architecture (``sm_120`` / ``sm_120a``), 1 otherwise."""
    if arch is None or capability is None:
        return 1
    digits = re.sub(r"[a-z]$", "", arch.removeprefix("sm_"))
    return 0 if digits == f"{capability[0]}{capability[1]}" else 1


def census(
    module: Any = None,
    candidate: Any = None,
    *,
    ran: Iterable[str] = (),
    capability: tuple[int, int] | None = None,
    gpu: Mapping[str, Any] | None = None,
    run: Any = subprocess.run,
) -> dict[str, Any]:
    """The opcode census of a candidate's kernels (module docstring): ``{"status": "ok",
    "kernels": [...], "gpu": {...}}``, or ``"unavailable"`` / ``"error"`` with the reason.
    ``ran``: kernel names that ran (the profiled table): their kernels come first; Triton
    kernels compiled several times (autotuning) are one row with ``variants`` and the
    union of their tensor-core opcodes. ``gpu``: the GPU facts the directives compare with
    (:func:`gpu_facts`). Never raises; no GPU work."""
    out: dict[str, Any] = {"status": "ok"}
    if gpu:
        out["gpu"] = dict(gpu)
        cap = gpu.get("capability")
        if capability is None and isinstance(cap, list | tuple) and len(cap) >= 2:
            capability = (int(cap[0]), int(cap[1]))
    found = availability()
    if not found["usable"]:
        return out | {"status": "unavailable", "reason": found["reason"]}
    names = [str(n) for n in ran]
    try:
        namespace = dict(vars(module)) if module is not None else {}
        if candidate is not None:
            namespace |= {f"candidate.{k}": v for k, v in vars(candidate).items()}
        found_binaries = binaries(namespace)  # those that ran first: within the cap
        found_binaries.sort(key=lambda b: not any(matches(b.name, n) for n in names))
        rows: dict[tuple[str, str], dict[str, Any]] = {}
        for binary in found_binaries[:MAX_BINARIES]:
            text = disassemble(binary, found["cuobjdump"], run=run)
            for kernel in parse(text):
                if not kernel.ops:
                    continue
                row = summarise(kernel)
                row["source"] = binary.source
                if (file := binary.path or binary.origin) is not None:
                    row["file"] = file.name
                key = (row["kernel"], str(row.get("arch")))
                if key in rows:
                    _merge(rows[key], row)
                else:
                    rows[key] = row
    except Exception as exc:  # never breaks an evaluation
        return out | {"status": "error", "reason": f"{type(exc).__name__}: {exc}"[:300]}
    ordered = sorted(
        rows.values(),
        key=lambda r: (
            _arch_rank(r.get("arch"), capability),
            0 if any(matches(r["kernel"], n) for n in names) else 1,
            -int(r.get("instructions") or 0),
        ),
    )
    out["kernels"] = ordered[:MAX_KERNELS]
    if missing := [n for n in names if not any(matches(r["kernel"], n) for r in ordered)]:
        out["missing"] = [n[:120] for n in missing[:MAX_MISSING]]
    if len(found_binaries) > MAX_BINARIES:
        out["note"] = f"{MAX_BINARIES} of {len(found_binaries)} binaries disassembled"
    if missing or not ordered:
        out["note"] = MISSING_NOTE
    return out


def _merge(row: dict[str, Any], other: dict[str, Any]) -> None:
    """Fold another compiled variant of the same kernel into ``row``: the union of their
    tensor-core opcodes, the largest counts, how many variants use local memory."""
    row["variants"] = int(row.get("variants") or 1) + 1
    local = int(bool(row.get("categories", {}).get("local")))
    local = int(row.get("local_variants", local)) + int(
        bool(other.get("categories", {}).get("local"))
    )
    if local:
        row["local_variants"] = local
    for key in ("tensor", "categories", "local", "global_load_bits"):
        merged = dict(row.get(key) or {})
        for op, n in (other.get(key) or {}).items():
            merged[op] = max(int(merged.get(op, 0)), int(n))
        if merged:
            row[key] = merged
    row["instructions"] = max(int(row.get("instructions") or 0), int(other["instructions"]))
    if other.get("fp8_unpacks"):
        row["fp8_unpacks"] = max(int(row.get("fp8_unpacks") or 0), int(other["fp8_unpacks"]))


def matches(kernel: str, name: str) -> bool:
    """Whether a census kernel name and a profiler kernel name (``void ns::k<...>(...)``,
    an Inductor ``triton_poi_fused_...``) are the same kernel: equal identifiers."""

    def ident(text: str) -> str:
        text = re.sub(r"^(?:void|__global__)\s+", "", text.strip())
        text = text.split("(", 1)[0].split("<", 1)[0]
        return text.rsplit("::", 1)[-1].strip()

    a, b = ident(kernel), ident(name)
    return bool(a) and a == b


def gpu_facts() -> dict[str, Any]:
    """The GPU facts the directives compare a census with: name, ``arch``, ``capability``
    and the measured tensor-core instruction rates (``mma_tflops``) and GEMM peaks
    (``tflops``) of the peaks cache. Empty without a GPU."""
    with contextlib.suppress(Exception):
        import torch

        if not torch.cuda.is_available():
            return {}
        index = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        out: dict[str, Any] = {
            "name": torch.cuda.get_device_name(index),
            "arch": f"sm_{major}{minor}",
            "capability": [major, minor],
        }
        from kernel_agent.kernels.roofline import current_peaks

        peaks = current_peaks() or {}
        for key in ("mma_tflops", "tflops"):
            if peaks.get(key):
                out[key] = dict(peaks[key])
        return out
    return {}


def line(row: Mapping[str, Any]) -> str:
    """One census row in a line (the profile summary): its tensor-core opcodes, async
    copies, TMA, local memory and global load widths."""
    tensor: Mapping[str, int] = row.get("tensor") or {}
    parts = [", ".join(f"{op} x{n}" for op, n in list(tensor.items())[:2]) or "no tensor-core MMA"]
    cats = row.get("categories") or {}
    for key, label in (("cp_async", "cp.async"), ("tma", "TMA"), ("local", "LDL/STL")):
        if cats.get(key):
            parts.append(f"{label} {cats[key]}")
    if widths := row.get("global_load_bits"):
        parts.append("LDG " + " ".join(f"{n}x{b}b" for b, n in widths.items()))
    if row.get("fp8_unpacks"):
        parts.append(f"FP8 unpacks {row['fp8_unpacks']}")
    if int(row.get("variants") or 1) > 1:
        parts.append(f"{row['variants']} variants")
    return f"{str(row.get('kernel'))[:60]} ({row.get('arch')}): " + "; ".join(parts)


def describe(found: Mapping[str, Any] | None = None) -> str:
    """One ``doctor`` line on :func:`availability`."""
    found = availability() if found is None else found
    if not found.get("usable"):
        return f"SASS census: unavailable ({found.get('reason')})"
    return f"SASS census: cuobjdump ({found['cuobjdump']})"
