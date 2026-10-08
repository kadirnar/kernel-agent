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
        else:
            lines.append("GPU: none detected")
        lines.append(f"torch {self.torch_version} (CUDA {self.torch_cuda})")
        lines.append(f"nvcc: {self.nvcc_version or 'not found'} (CUDA_HOME={self.cuda_home})")
        enabled = [name for name, ok in self.backends.items() if ok]
        disabled = [name for name, ok in self.backends.items() if not ok]
        lines.append(f"backends available: {', '.join(enabled) or 'none'}")
        if disabled:
            lines.append(f"backends unavailable: {', '.join(disabled)}")
        if self.peaks:
            lines.append(f"measured peaks: {format_peaks(self.peaks)}")
        elif self.gpu:
            lines.append("measured peaks: not yet (`kernel-agent doctor` measures them)")
        from kernel_agent.gpu_arch import summary_lines

        lines.extend(summary_lines(self.gpu, self.peaks))  # family, full rate, precisions
        lines.extend(f"note: {n}" for n in self.notes)
        return "\n".join(lines)


def gpu_info() -> GPUInfo | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2 = getattr(props, "L2_cache_size", 0) or 0
    return GPUInfo(
        name=props.name,
        capability=(props.major, props.minor),
        memory_gb=props.total_memory / 1024**3,
        sm_count=props.multi_processor_count,
        l2_cache_mb=l2 / 1024**2,
        smem_per_block_kb=(getattr(props, "shared_memory_per_block_optin", 0) or 0) / 1024,
        smem_per_sm_kb=(getattr(props, "shared_memory_per_multiprocessor", 0) or 0) / 1024,
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


def cuda_arch_list(capability: tuple[int, int]) -> str:
    """``TORCH_CUDA_ARCH_LIST`` for a GPU of ``capability`` (``9.0a`` on Hopper)."""
    major, minor = int(capability[0]), int(capability[1])
    return f"{major}.{minor}" + ("a" if (major, minor) in ARCH_SPECIFIC else "")


# Newest GCC major officially accepted by nvcc, per CUDA major version.  Forcing
# a newer host compiler is harmless when nvcc would have accepted it anyway.
_MAX_GCC = {12: 14, 13: 15}


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


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
        if gpu is not None and "TORCH_CUDA_ARCH_LIST" not in os.environ:
            env["TORCH_CUDA_ARCH_LIST"] = cuda_arch_list(gpu.capability)

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
    if nvcc_ver is not None and not has_ninja:
        notes.append("ninja missing: torch load_inline (cuda backend) disabled")

    if apply_env:
        os.environ.update(env)

    return Toolchain(
        gpu=gpu,
        torch_version=torch.__version__,
        torch_cuda=torch.version.cuda,
        cuda_home=cuda_home,
        nvcc_version=".".join(map(str, nvcc_ver)) if nvcc_ver else None,
        backends=backends,
        notes=notes,
        env=env,
        peaks=load_peaks(peaks_path(gpu.name, torch.__version__)) if gpu else None,
    )


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
