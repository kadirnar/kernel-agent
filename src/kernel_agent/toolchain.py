"""GPU and compiler toolchain discovery.

Kernel backends need different things from the machine:

* Triton ships its own compiler (bundled ``ptxas``).
* ``nvrtc`` (``cuda.core``) and CuTe DSL compile at runtime without ``nvcc``.
* TileLang and ``torch.utils.cpp_extension.load_inline`` call ``nvcc``.

A system CUDA toolkit is used when present.  Otherwise the pip wheels
(``nvidia-cuda-nvcc``, ``nvidia-cuda-cccl``, ...) are assembled into a
``CUDA_HOME`` shim directory, because the wheels ship ``libcudart.so.13`` but
not the unversioned ``libcudart.so`` the linker looks for.  Host compilers that
are newer than ``nvcc`` officially supports, and header/compiler version skew
between wheels, are papered over with ``NVCC_APPEND_FLAGS``.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CACHE_DIR = Path(os.environ.get("KERNEL_AGENT_CACHE", Path.home() / ".cache" / "kernel-agent"))


@dataclass
class GPUInfo:
    name: str
    capability: tuple[int, int]
    memory_gb: float
    sm_count: int
    l2_cache_mb: float
    smem_per_block_kb: float = 0.0  # opt-in maximum per block (0: unknown)
    smem_per_sm_kb: float = 0.0
    emulated_on: str | None = None  # an emulated GPU (emulate.py): the real GPU's arch

    @property
    def arch(self) -> str:
        return f"sm_{self.capability[0]}{self.capability[1]}"


@dataclass
class Toolchain:
    gpu: GPUInfo | None
    torch_version: str
    torch_cuda: str | None
    cuda_home: str | None
    nvcc_version: str | None
    backends: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    peaks: dict[str, Any] | None = None  # measured roofline peaks (kernels/roofline.py)
    #: installed backends that cannot compile for this GPU, and why (:data:`ARCH_SUPPORT`)
    unavailable: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.gpu is not None:
            data["gpu"]["arch"] = self.gpu.arch
        return data

    def summary(self) -> str:
        lines = []
        if self.gpu:
            g = self.gpu
            smem = f", {g.smem_per_block_kb:.0f} KB smem per block" if g.smem_per_block_kb else ""
            lines.append(
                f"GPU: {g.name} ({g.arch}, {g.memory_gb:.1f} GB, {g.sm_count} SMs, "
                f"L2 {g.l2_cache_mb:.0f} MB{smem})"
            )
            from kernel_agent.emulate import summary_line

            if (emulated := summary_line(g)) is not None:
                lines.append(emulated)
        else:
            lines.append("GPU: none detected")
        lines.append(f"torch {self.torch_version} (CUDA {self.torch_cuda})")
        lines.append(f"nvcc: {self.nvcc_version or 'not found'} (CUDA_HOME={self.cuda_home})")
        enabled = [name for name, ok in self.backends.items() if ok]
        disabled = [name for name, ok in self.backends.items() if not ok]
        lines.append(f"backends available: {', '.join(enabled) or 'none'}")
        if disabled:
            why = self.unavailable
            off = [f"{name} ({why[name]})" if name in why else name for name in disabled]
            lines.append(f"backends unavailable: {', '.join(off)}")
        if self.peaks:
            lines.append(f"measured peaks: {format_peaks(self.peaks)}")
        elif self.gpu and self.gpu.emulated_on:
            lines.append(f"measured peaks: none under emulation (the {self.gpu.emulated_on}'s)")
        elif self.gpu:
            lines.append("measured peaks: not yet (`kernel-agent doctor` measures them)")
        from kernel_agent.gpu_arch import summary_lines

        lines.extend(summary_lines(self.gpu, self.peaks))  # family, full rate, precisions
        lines.extend(f"note: {n}" for n in self.notes)
        return "\n".join(lines)


def gpu_info() -> GPUInfo | None:
    """The GPU of this process: the current device, or under emulation (:mod:`kernel_agent.
    emulate`) the emulated GPU's facts on it (``emulate.Refused``: it cannot emulate that)."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    from kernel_agent import emulate

    props = emulate.real_properties(torch.cuda.current_device())
    l2 = getattr(props, "L2_cache_size", 0) or 0
    return emulate.apply(
        GPUInfo(
            name=props.name,
            capability=(props.major, props.minor),
            memory_gb=props.total_memory / 1024**3,
            sm_count=props.multi_processor_count,
            l2_cache_mb=l2 / 1024**2,
            smem_per_block_kb=(getattr(props, "shared_memory_per_block_optin", 0) or 0) / 1024,
            smem_per_sm_kb=(getattr(props, "shared_memory_per_multiprocessor", 0) or 0) / 1024,
        )
    )


def peaks_path(gpu: str, torch_version: str) -> Path:
    """Cache file of the measured roofline peaks of one GPU model + torch version."""
    name = re.sub(r"[^A-Za-z0-9.]+", "-", f"{gpu}-torch{torch_version}").strip("-").lower()
    return CACHE_DIR / f"peaks-{name}.json"


def load_peaks(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("dram_gbps") else None


def format_peaks(peaks: dict[str, Any]) -> str:
    """One line: copy bandwidths (read + write bytes), matmul TFLOP/s, launch floor."""
    parts = [f"copy DRAM {peaks['dram_gbps']:.0f} GB/s"]
    if peaks.get("l2_gbps"):
        parts.append(f"L2 {peaks['l2_gbps']:.0f} GB/s")
    short = {
        "bfloat16": "bf16",
        "float16": "fp16",
        "float32": "fp32",
        "float8_e4m3fn": "fp8",
        "float4_e2m1fn_x2": "nvfp4",
    }
    tflops = peaks.get("tflops") or {}
    if tflops:
        names = "/".join(short.get(k, k) for k in tflops)
        values = " / ".join(f"{v:.0f}" for v in tflops.values())
        parts.append(f"matmul {names} {values} TFLOP/s")
    if sustained := peaks.get("tflops_sustained"):  # under sustained load (#253)
        parts.append(_sustained(sustained, peaks.get("sustained") or {}, short))
    if peaks.get("tflops_unavailable"):  # low-precision matmuls torch has no kernel for here
        parts.append(
            "no " + "/".join(short.get(k, k) for k in peaks["tflops_unavailable"]) + " matmul"
        )
    if peaks.get("launch_floor_us"):
        parts.append(f"launch floor {peaks['launch_floor_us']:.1f} us")
    if peaks.get("mma_tflops"):  # tensor-core instruction rates (kernels/mma_peaks.py)
        from kernel_agent.kernels.mma_peaks import describe

        parts.append(describe(peaks["mma_tflops"]))
    return ", ".join(parts)


def _sustained(tflops: dict[str, Any], info: dict[str, Any], short: dict[str, str]) -> str:
    """``sustained bf16/fp16 100 / 101 TFLOP/s (2 s; SM 1200 MHz, burst 1695; 149 W of a
    150 W limit; sw_power_cap)``: the 16-bit peaks under sustained load and that load."""
    names = "/".join(short.get(k, k) for k in tflops)
    values = " / ".join(f"{float(v):.0f}" for v in tflops.values())
    load = [f"{float(info['seconds']):.0f} s"] if info.get("seconds") else []
    if info.get("sm_mhz"):
        burst = f", burst {info['burst_sm_mhz']}" if info.get("burst_sm_mhz") else ""
        load.append(f"SM {info['sm_mhz']} MHz{burst}")
    if info.get("power_w"):
        limit = f" of a {info['power_limit_w']:.0f} W limit" if info.get("power_limit_w") else ""
        load.append(f"{info['power_w']:.0f} W{limit}")
    if info.get("reasons"):
        load.append(", ".join(info["reasons"]))
    return f"sustained {names} {values} TFLOP/s" + (f" ({'; '.join(load)})" if load else "")


def _pip_cuda_root() -> Path | None:
    """Directory of the merged ``nvidia/cu1x`` wheel tree that contains ``bin/nvcc``."""
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return None
    for base in spec.submodule_search_locations:
        for sub in sorted(Path(base).glob("cu*"), reverse=True):
            if (sub / "bin" / "nvcc").exists():
                return sub
        legacy = Path(base) / "cuda_nvcc"
        if (legacy / "bin" / "nvcc").exists():
            return legacy
    return None


def _build_cuda_home_shim(root: Path) -> Path:
    """Mirror ``root`` into a cache dir and add unversioned ``lib*.so`` symlinks."""
    digest = hashlib.sha1(str(root.resolve()).encode()).hexdigest()[:12]
    shim = CACHE_DIR / f"cuda-home-{digest}"
    marker = shim / ".source"
    if marker.exists() and marker.read_text() == str(root):
        return shim
    if shim.exists():
        shutil.rmtree(shim)
    (shim / "lib").mkdir(parents=True)
    for entry in root.iterdir():
        if entry.name not in {"lib", "lib64"}:
            (shim / entry.name).symlink_to(entry)
    for lib_dir in (root / "lib", root / "lib64"):
        if not lib_dir.is_dir():
            continue
        for lib in lib_dir.iterdir():
            target = shim / "lib" / lib.name
            if not target.exists():
                target.symlink_to(lib)
            match = re.match(r"(lib.+?\.so)\.\d", lib.name)
            if match and not (shim / "lib" / match.group(1)).exists():
                (shim / "lib" / match.group(1)).symlink_to(lib)
    (shim / "lib64").symlink_to("lib")
    marker.write_text(str(root))
    return shim


def _nvcc_version(nvcc: Path) -> tuple[int, int] | None:
    try:
        out = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"release (\d+)\.(\d+)", out.stdout)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _gcc_major() -> int | None:
    cxx = shutil.which("g++") or shutil.which("c++")
    if cxx is None:
        return None
    try:
        out = subprocess.run([cxx, "-dumpversion"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.match(r"(\d+)", out.stdout.strip())
    return int(match.group(1)) if match else None


#: Capabilities whose peak MMA (``wgmma`` on sm_90, ``tcgen05`` on sm_100 / sm_103 /
#: sm_110) exists only in the architecture-specific target (``sm_90a``): ``load_inline``
#: builds for it there (#165). sm_12x keeps the plain target its examples were verified
#: with (the block-scaled ones ask for ``sm_120a`` themselves).
ARCH_SPECIFIC = {(9, 0), (10, 0), (10, 3), (11, 0)}


def cuda_arch_list(capability: tuple[int, int], ptx: bool | None = None) -> str:
    """``TORCH_CUDA_ARCH_LIST`` for a GPU of ``capability`` (``9.0a`` on Hopper). With ``ptx``
    (default: under emulation, :mod:`kernel_agent.emulate`) ``8.6+PTX``: the build embeds the
    PTX, which the driver JIT-compiles for the real GPU (an sm_86 cubin does not load there)."""
    major, minor = int(capability[0]), int(capability[1])
    if ptx is None:
        ptx = _emulating()
    if ptx:
        return f"{major}.{minor}+PTX"
    return f"{major}.{minor}" + ("a" if (major, minor) in ARCH_SPECIFIC else "")


def _emulating() -> bool:
    from kernel_agent.emulate import active

    return active()


def nvrtc_target(capability: tuple[int, ...] | None = None) -> tuple[str, str]:
    """``(arch, code type)`` NVRTC compiles for (``ProgramOptions(arch=...)``,
    ``Program.compile(code type)``) on a GPU of ``capability`` (default: this process's):
    ``("sm_120", "cubin")``; under emulation (:mod:`kernel_agent.emulate`) ``("compute_86",
    "ptx")``: ``get_kernel`` hands the PTX to the driver, which JIT-compiles it for the real
    GPU (an sm_86 cubin does not load there)."""
    if capability is None:
        gpu = setup().gpu
        if gpu is None:
            raise RuntimeError("no CUDA GPU to compile for")
        capability = gpu.capability
    major, minor = int(capability[0]), int(capability[1])
    if _emulating():
        return f"compute_{major}{minor}", "ptx"
    return f"sm_{major}{minor}", "cubin"


# ------------------------------------------------------------------ line info, NVRTC (#230)

#: ``1`` in a process whose CUDA C++ builds carry line info (:func:`lineinfo_env`): the
#: evaluator's ``--ncu-mode`` (:func:`kernel_agent.kernels.ncu.ncu_entry`), so Nsight
#: Compute's source counters name CUDA C++ lines, not SASS instructions only.
LINEINFO_ENV = "KERNEL_AGENT_LINEINFO"
#: The directory of the line-info extension builds inside ``TORCH_EXTENSIONS_DIR``.
LINEINFO_DIR = "lineinfo"


def lineinfo() -> bool:
    """Whether this process builds CUDA C++ with line info (:data:`LINEINFO_ENV`)."""
    return os.environ.get(LINEINFO_ENV) == "1"


def lineinfo_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The variables that make a process (of environment ``environ``, default this one's)
    build its CUDA C++ with line info: :data:`LINEINFO_ENV` (NVRTC through
    :func:`nvrtc_kernels`), ``-lineinfo`` in ``NVCC_APPEND_FLAGS`` (every nvcc:
    ``load_inline``, native projects, whose build key includes the flags; TileLang passes
    it itself) and torch's extension builds in a :data:`LINEINFO_DIR` directory of their
    own. torch's build hash does not cover ``NVCC_APPEND_FLAGS``: in the shared directory
    the cached build without line info would be loaded, and a rebuild in place would
    replace the one the evaluations load (an nvcc build of 20-60 s each way). ``-lineinfo``
    adds line tables only; the code is the same."""
    environ = os.environ if environ is None else environ
    flags = environ.get("NVCC_APPEND_FLAGS", "").split()
    root = environ.get("TORCH_EXTENSIONS_DIR") or torch_extensions_default()
    if os.path.basename(os.path.normpath(root)) != LINEINFO_DIR:
        root = os.path.join(root, LINEINFO_DIR)
    return {
        LINEINFO_ENV: "1",
        "NVCC_APPEND_FLAGS": " ".join(dict.fromkeys([*flags, "-lineinfo"])),
        "TORCH_EXTENSIONS_DIR": root,
    }


def nvrtc_source_file(source: str) -> Path:
    """A file holding ``source`` (``<cache>/nvrtc-src/<sha256>.cu``) to name an NVRTC
    program by under :func:`lineinfo`: its line table points at the program's name, and
    Nsight Compute reads a line's text from that file (an NVRTC program has none)."""
    path = CACHE_DIR / "nvrtc-src" / f"{hashlib.sha256(source.encode()).hexdigest()[:24]}.cu"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(source)
        os.replace(tmp, path)  # whole or absent for a concurrent reader
    return path


def nvrtc_options(
    source: str, capability: tuple[int, ...] | None = None, **options: Any
) -> tuple[dict[str, Any], str]:
    """``(ProgramOptions keyword arguments, code type)`` of :func:`nvrtc_kernels`: the
    target of :func:`nvrtc_target` (an ``arch`` in ``options`` wins, e.g. ``sm_120a``; a
    ``compute_XY`` one compiles to PTX), ``std="c++17"``, the :func:`cuda_include_dirs`
    after any ``include_path`` given, the other ``options`` as given; under
    :func:`lineinfo` also ``lineinfo=True`` and the program named by
    :func:`nvrtc_source_file` (unless ``options`` say otherwise)."""
    arch = options.pop("arch", None)
    if arch is None:
        arch, kind = nvrtc_target(capability)
    else:
        kind = "ptx" if str(arch).startswith("compute_") else "cubin"
    given = options.pop("include_path", None)
    paths = [given] if isinstance(given, str) else list(given or [])
    out: dict[str, Any] = {"arch": arch, "std": "c++17"}
    out["include_path"] = list(dict.fromkeys([*map(str, paths), *cuda_include_dirs()]))
    out.update(options)
    if lineinfo():
        out.setdefault("lineinfo", True)
        out.setdefault("name", str(nvrtc_source_file(source)))
    return out, kind


def nvrtc_kernels(
    source: str,
    names: Iterable[str],
    *,
    capability: tuple[int, ...] | None = None,
    name_expressions: Iterable[str] = (),
    **options: Any,
) -> dict[str, Any]:
    """``names``' ``cuda.core`` kernels, compiled from CUDA C++ ``source`` with NVRTC for this
    GPU: ``Program(source, "c++", ProgramOptions(**opts)).compile(kind)`` with
    :func:`nvrtc_options` (``capability``: another GPU's target), then ``get_kernel`` per
    name (a template instantiation such as ``"k<128>"`` is added to ``name_expressions``).

    The SASS census sees their cubin as long as a kernel lives
    (:func:`kernel_agent.kernels.sass.keep`): ``cuda.core`` frees the ``ObjectCode`` once
    ``get_kernel`` returns and a ``Kernel`` does not expose its cubin, so a candidate
    compiling with ``Program`` directly must keep the ``ObjectCode`` itself (#230). In the
    evaluator's ``--ncu-mode`` (:func:`lineinfo`) the program carries line info."""
    from cuda.core import Program, ProgramOptions

    from kernel_agent.kernels import sass

    wanted = list(dict.fromkeys(names))
    expressions = list(dict.fromkeys([*name_expressions, *(n for n in wanted if "<" in n)]))
    kwargs, kind = nvrtc_options(source, capability, **options)
    program = Program(source, code_type="c++", options=ProgramOptions(**kwargs))
    code = program.compile(kind, name_expressions=expressions)
    kernels = {name: code.get_kernel(name) for name in wanted}
    if kind == "cubin" and isinstance(cubin := code.code, bytes | bytearray):
        for name, kernel in kernels.items():
            sass.keep(kernel, name, bytes(cubin))
    return kernels


def nvrtc_kernel(source: str, name: str, **kwargs: Any) -> Any:
    """The ``cuda.core`` kernel ``name`` compiled from ``source`` (:func:`nvrtc_kernels`)."""
    return nvrtc_kernels(source, [name], **kwargs)[name]


# Newest GCC major officially accepted by nvcc, per CUDA major version.  Forcing
# a newer host compiler is harmless when nvcc would have accepted it anyway.
_MAX_GCC = {12: 14, 13: 15}


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _cute_targets(capability: tuple[int, int]) -> str | None:
    from kernel_agent import cute_dsl

    return cute_dsl.unsupported(capability)


#: Backends whose compiler targets only some architectures: why each cannot compile for a
#: GPU of a capability (None: it can). An installed backend refused here is off, with the
#: reason (``Toolchain.unavailable``), so neither a plan nor ``doctor --smoke`` uses it:
#: CuTe DSL 4.8's targets start at sm_80, and on a T4 every ``cute`` kernel fails to
#: compile (``KeyError: 'sm_75'``, #254). Triton, nvcc (CUDA C++, TileLang) and NVRTC
#: compile for sm_75 and newer; Helion compiles to Triton (``doctor``'s helion probe runs
#: its example on the GPU).
ARCH_SUPPORT: dict[str, Callable[[tuple[int, int]], str | None]] = {"cute": _cute_targets}


def arch_unsupported(backends: dict[str, bool], capability: tuple[int, int]) -> dict[str, str]:
    """Why each available backend of ``backends`` cannot compile for a GPU of
    ``capability`` (:data:`ARCH_SUPPORT`). A check that fails itself refuses nothing: the
    backend's own compile errors say more than a broken probe."""
    out: dict[str, str] = {}
    for name, why_not in ARCH_SUPPORT.items():
        if not backends.get(name):
            continue
        try:
            why = why_not(capability)
        except Exception:
            continue
        if why:
            out[name] = why
    return out


@functools.cache
def setup(apply_env: bool = True) -> Toolchain:
    """Discover the toolchain and (optionally) export the env vars backends need.

    Safe to call many times; the result is cached per process.
    """
    import torch

    notes: list[str] = []
    env: dict[str, str] = {}

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc: Path | None = None
    if cuda_home and (Path(cuda_home) / "bin" / "nvcc").exists():
        nvcc = Path(cuda_home) / "bin" / "nvcc"
    elif (system_nvcc := shutil.which("nvcc")) is not None:
        nvcc = Path(system_nvcc).resolve()
        cuda_home = str(nvcc.parent.parent)
    elif (root := _pip_cuda_root()) is not None:
        shim = _build_cuda_home_shim(root)
        cuda_home = str(shim)
        nvcc = shim / "bin" / "nvcc"
        env["CUDA_HOME"] = cuda_home
        notes.append(f"using pip CUDA toolchain from {root}")

    nvcc_ver = _nvcc_version(nvcc) if nvcc else None
    gpu = gpu_info()

    if nvcc_ver is not None:
        flags = os.environ.get("NVCC_APPEND_FLAGS", "").split()
        gcc = _gcc_major()
        max_gcc = _MAX_GCC.get(nvcc_ver[0])
        if gcc is not None and max_gcc is not None and gcc > max_gcc:
            flags.append("-allow-unsupported-compiler")
            notes.append(f"g++ {gcc} is newer than nvcc {nvcc_ver} supports; forcing it")
        torch_cuda = torch.version.cuda
        if torch_cuda and tuple(int(x) for x in torch_cuda.split(".")[:2]) != nvcc_ver:
            flags.append("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK")
            notes.append(f"nvcc {nvcc_ver} differs from torch CUDA {torch_cuda}")
        if flags:
            env["NVCC_APPEND_FLAGS"] = " ".join(dict.fromkeys(flags))
        emulated = gpu is not None and gpu.emulated_on is not None  # its PTX, always
        if gpu is not None and (emulated or "TORCH_CUDA_ARCH_LIST" not in os.environ):
            env["TORCH_CUDA_ARCH_LIST"] = cuda_arch_list(gpu.capability, ptx=emulated)

    has_ninja = _module_available("ninja") or shutil.which("ninja") is not None
    if apply_env and shutil.which("ninja") is None and _module_available("ninja"):
        # load_inline runs the `ninja` executable from PATH; the pip wheel puts it
        # next to the venv's python, which is not on PATH unless the venv is active.
        import ninja

        bin_dir = str(getattr(ninja, "BIN_DIR", ""))
        if bin_dir and Path(bin_dir, "ninja").exists():
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            notes.append(f"added {bin_dir} to PATH for ninja")
    backends = {
        "triton": _module_available("triton") and gpu is not None,
        "cuda": nvcc_ver is not None and has_ninja and gpu is not None,
        "nvrtc": _module_available("cuda.core") and gpu is not None,
        "cute": _module_available("cutlass") and gpu is not None,
        "tilelang": _module_available("tilelang") and nvcc_ver is not None and gpu is not None,
        # Helion compiles to Triton (#229); `doctor`'s helion probe says whether it runs here
        "helion": _module_available("helion") and _module_available("triton") and gpu is not None,
    }
    unavailable = arch_unsupported(backends, gpu.capability) if gpu is not None else {}
    backends.update(dict.fromkeys(unavailable, False))
    if nvcc_ver is not None and not has_ninja:
        notes.append("ninja missing: torch load_inline (cuda backend) disabled")
    if gpu is not None and gpu.emulated_on:
        _emulated_env(gpu, env, backends, unavailable)

    if apply_env:
        os.environ.update(env)
        if gpu is not None and gpu.emulated_on:
            from kernel_agent import emulate

            emulate.install(gpu)  # torch's device properties, Triton's target and loading
    # under emulation no peaks: the real GPU's are not the emulated one's (emulate.py)
    measured = gpu is not None and not gpu.emulated_on

    return Toolchain(
        gpu=gpu,
        torch_version=torch.__version__,
        torch_cuda=torch.version.cuda,
        cuda_home=cuda_home,
        nvcc_version=".".join(map(str, nvcc_ver)) if nvcc_ver else None,
        backends=backends,
        notes=notes,
        env=env,
        peaks=load_peaks(peaks_path(gpu.name, torch.__version__)) if gpu and measured else None,
        unavailable=unavailable,
    )


def torch_extensions_default() -> str:
    """torch's own ``TORCH_EXTENSIONS_DIR`` default (``cpp_extension._get_build_directory``:
    per Python and CUDA version), without importing ``torch.utils.cpp_extension``: it reads
    ``CUDA_HOME`` once, at its import, and :func:`setup`'s env may not have set it yet."""
    import sys

    import torch

    try:
        from torch._appdirs import user_cache_dir

        root = os.path.realpath(user_cache_dir(appname="torch_extensions"))
    except ImportError:
        root = str(Path.home() / ".cache" / "torch_extensions")
    cuda = f"cu{torch.version.cuda.replace('.', '')}" if torch.version.cuda else "cpu"
    python = f"py{sys.version_info.major}{sys.version_info.minor}{getattr(sys, 'abiflags', '')}"
    return os.path.join(root, f"{python}_{cuda}")


def _emulated_env(
    gpu: GPUInfo, env: dict[str, str], backends: dict[str, bool], unavailable: dict[str, str]
) -> None:
    """The builds of an emulated ``gpu`` (:mod:`kernel_agent.emulate`): extensions and
    Inductor's cache in their own directories (nothing built for the real GPU is reused),
    Inductor without its cubin-loading launcher, CuTe DSL compiling for it and the driver's
    JIT cache large enough; CuTe DSL and TileLang cannot run (SASS)."""
    import getpass
    import tempfile

    from kernel_agent import emulate

    current = os.environ.get("TORCH_EXTENSIONS_DIR")
    env["TORCH_EXTENSIONS_DIR"] = emulate.arch_dir(gpu.arch, current, torch_extensions_default())
    # Inductor: its own cache (its key names the GPU, not the target) and no static launcher
    # (it loads the cubin, which is the emulated GPU's)
    try:
        user = getpass.getuser()
    except (KeyError, OSError):  # no user name (a container): torch falls back the same way
        user = f"uid_{os.getuid()}"
    inductor = os.path.join(tempfile.gettempdir(), f"torchinductor_{user}")
    current = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    env["TORCHINDUCTOR_CACHE_DIR"] = emulate.arch_dir(gpu.arch, current, inductor)
    env["TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER"] = "0"
    env["CUTE_DSL_ARCH"] = gpu.arch
    if not os.environ.get("CUDA_CACHE_MAXSIZE"):  # read when a process's CUDA starts
        env["CUDA_CACHE_MAXSIZE"] = str(emulate.JIT_CACHE_BYTES)
    for name in emulate.COMPILE_ONLY:
        if backends.get(name):
            backends[name] = False
            unavailable[name] = emulate.compile_only(name, gpu) or ""


# ------------------------------------------------------------------ compute-sanitizer

#: The path of a ``compute-sanitizer`` to use (it wins over every other place).
SANITIZER_ENV = "KERNEL_AGENT_COMPUTE_SANITIZER"
#: What ``compute-sanitizer`` launches its target through and injects into it: they must be
#: next to the binary. The ``nvidia-cuda-sanitizer-api`` pip wheel ships the binary without
#: ``TreeLauncherSubreaper`` (and its libraries in ``lib/``), so every target ends with
#: "Target application terminated before first instrumented API call" (issue #115).
SANITIZER_PARTS = (
    "TreeLauncherSubreaper",
    "libTreeLauncherTargetInjection.so",
    "libInterceptorInjectionTarget.so",
)
#: NVIDIA's redistributable archives (``doctor --fetch-sanitizer``, :func:`fetch_sanitizer`).
REDIST_URL = "https://developer.download.nvidia.com/compute/cuda/redist/"


@dataclass
class Sanitizer:
    """A ``compute-sanitizer`` that can launch a target (``path``), or why there is none."""

    path: str | None
    version: str | None = None
    reason: str | None = None  # why none is usable (``path`` None)
    rejected: list[str] = field(default_factory=list)  # installs found but unusable, and why

    def describe(self) -> str:
        if self.path:
            return f"compute-sanitizer {self.version or '?'} ({self.path})"
        return f"compute-sanitizer unavailable: {self.reason}"


def _sanitizer_dir(binary: Path) -> Path:
    """The directory of the real binary: a toolkit's ``bin/compute-sanitizer`` is a script
    that runs ``../compute-sanitizer/compute-sanitizer``."""
    real = binary.resolve()
    beside = real.parent.parent / "compute-sanitizer"
    if real.parent.name == "bin" and (beside / "compute-sanitizer").is_file():
        return beside
    return real.parent


def _natural(path: Path) -> list[Any]:
    """Sort key: version numbers in a path compare as numbers (13.10 after 13.4)."""
    return [(int(x), "") if x.isdigit() else (-1, x) for x in re.split(r"(\d+)", str(path))]


def _sanitizer_places() -> list[Path]:
    """Every ``compute-sanitizer`` to consider, in order: :data:`SANITIZER_ENV`,
    ``CUDA_HOME``, ``PATH``, the usual toolkit prefixes, :func:`fetch_sanitizer`'s cache, the
    pip wheels."""
    places: list[Path] = []
    if override := os.environ.get(SANITIZER_ENV):
        places.append(Path(override).expanduser())
    homes = [os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")]
    for home in [*homes, "/usr/local/cuda", "/opt/cuda"]:
        if home:
            places += [Path(home) / "compute-sanitizer" / "compute-sanitizer"]
            places += [Path(home) / "bin" / "compute-sanitizer"]
    if found := shutil.which("compute-sanitizer"):
        places.append(Path(found))
    fetched = (CACHE_DIR / "compute-sanitizer").glob("*/compute-sanitizer/compute-sanitizer")
    places += sorted(fetched, key=_natural, reverse=True)  # the newest archive first
    spec = importlib.util.find_spec("nvidia")
    wheels = (spec.submodule_search_locations or []) if spec is not None else []
    for base in wheels:
        places += sorted(Path(base).glob("*/bin/compute-sanitizer"), reverse=True)
        places += sorted(Path(base).glob("*/compute-sanitizer/compute-sanitizer"), reverse=True)
    return list(dict.fromkeys(places))


def _sanitizer_version(binary: Path) -> str | None:
    try:
        out = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"Version (\S+)", out.stdout)
    return match.group(1) if match else None


def find_sanitizer(places: list[Path] | None = None) -> Sanitizer:
    """The first complete ``compute-sanitizer`` of ``places`` (default:
    :func:`_sanitizer_places`): an executable that has :data:`SANITIZER_PARTS` beside it
    and reports a version. Installs without them are listed in ``rejected``."""
    rejected: list[str] = []
    for binary in _sanitizer_places() if places is None else places:
        if not binary.is_file():
            continue
        if not os.access(binary, os.X_OK):
            rejected.append(f"{binary}: not executable")
            continue
        folder = _sanitizer_dir(binary)
        if missing := [p for p in SANITIZER_PARTS if not (folder / p).exists()]:
            rejected.append(
                f"{binary}: incomplete, no {', '.join(missing)} in {folder} (the pip wheel's "
                "layout: no target ever starts)"
            )
            continue
        if (version := _sanitizer_version(folder / "compute-sanitizer")) is None:
            rejected.append(f"{binary}: does not run (`--version`)")
            continue
        return Sanitizer(str(folder / "compute-sanitizer"), version, rejected=rejected)
    reason = rejected[0] if rejected else "not found (CUDA_HOME, PATH, pip wheels)"
    if len(rejected) > 1:
        reason += f" (+{len(rejected) - 1} more)"
    hint = f"; `kernel-agent doctor --fetch-sanitizer` installs NVIDIA's, or set {SANITIZER_ENV}"
    return Sanitizer(None, reason=reason + hint, rejected=rejected)


_found: list[Sanitizer] = []


def sanitizer() -> Sanitizer:
    """:func:`find_sanitizer`, once per process once it found one (until then it looks
    again each time: a long run sees one ``doctor --fetch-sanitizer`` installed meanwhile)."""
    if not _found or not _found[0].path:
        _found[:] = [find_sanitizer()]
    return _found[0]


def driver_cuda_version() -> tuple[int, int] | None:
    """The CUDA version the driver supports (``cuDriverGetVersion``; no CUDA context)."""
    import ctypes

    try:
        lib = ctypes.CDLL("libcuda.so.1")
        value = ctypes.c_int()
        if lib.cuDriverGetVersion(ctypes.byref(value)) != 0:
            return None
    except (OSError, AttributeError):
        return None
    return value.value // 1000, value.value % 1000 // 10


def _fetch(url: str, timeout: float = 120.0) -> bytes:
    import urllib.request

    with urllib.request.urlopen(url, timeout=timeout) as response:
        data: bytes = response.read()
    return data


def fetch_sanitizer(
    cuda: tuple[int, int] | None = None, fetch: Any = _fetch, log: Any = print
) -> Path:
    """Install ``compute-sanitizer`` from NVIDIA's CUDA redistributables into
    ``CACHE_DIR/compute-sanitizer/`` (where :func:`find_sanitizer` looks) and return it:
    the newest release for ``cuda`` (default: the driver's CUDA version) or older, its
    archive checked against the manifest's sha256."""
    import io
    import platform
    import tarfile

    want = cuda or driver_cuda_version()
    if want is None:
        raise RuntimeError("no CUDA driver: cannot tell which compute-sanitizer fits it")
    index = fetch(REDIST_URL).decode("utf-8", "replace")
    found = {
        tuple(int(x) for x in m.groups()): m.group(0)
        for m in re.finditer(r"redistrib_(\d+)\.(\d+)\.(\d+)\.json", index)
    }
    fits = sorted(v for v in found if v[:2] <= want)
    if not fits:
        raise RuntimeError(f"no CUDA redistributable for CUDA {want[0]}.{want[1]} or older")
    manifest = json.loads(fetch(REDIST_URL + found[fits[-1]]))
    arch = {"x86_64": "linux-x86_64", "aarch64": "linux-sbsa"}.get(platform.machine())
    entry = (manifest.get("cuda_sanitizer_api") or {}).get(arch or "")
    if not entry:
        raise RuntimeError(f"{found[fits[-1]]} has no cuda_sanitizer_api for {arch}")
    log(f"downloading {entry['relative_path']} ({int(entry.get('size', 0)) / 1e6:.0f} MB)")
    data = fetch(REDIST_URL + entry["relative_path"])
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise RuntimeError(f"{entry['relative_path']}: sha256 mismatch")
    root = CACHE_DIR / "compute-sanitizer"
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:xz") as tar:
        top = {Path(name).parts[0] for name in tar.getnames()}
        tar.extractall(root, filter="data")
    binaries = [root / t / "compute-sanitizer" / "compute-sanitizer" for t in sorted(top)]
    binary = next((p for p in binaries if p.is_file()), None)
    if binary is None:
        raise RuntimeError(f"{entry['relative_path']}: no compute-sanitizer/compute-sanitizer")
    _found.clear()
    return binary


def cuda_include_dirs() -> list[str]:
    """Header directories for runtime compilation (NVRTC needs ``cuda_bf16.h`` etc.)."""
    dirs: list[str] = []
    home = setup().cuda_home
    if home:
        dirs += [str(Path(home) / "include"), str(Path(home) / "include" / "cccl")]
    root = _pip_cuda_root()
    if root is not None:
        dirs += [str(root / "include"), str(root / "include" / "cccl")]
    return [d for d in dict.fromkeys(dirs) if Path(d).is_dir()]
