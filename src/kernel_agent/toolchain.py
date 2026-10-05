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
            lines.append(
                f"GPU: {g.name} ({g.arch}, {g.memory_gb:.1f} GB, {g.sm_count} SMs, "
                f"L2 {g.l2_cache_mb:.0f} MB)"
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
    short = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}
    tflops = peaks.get("tflops") or {}
    if tflops:
        names = "/".join(short.get(k, k) for k in tflops)
        values = " / ".join(f"{v:.0f}" for v in tflops.values())
        parts.append(f"matmul {names} {values} TFLOP/s")
    if peaks.get("launch_floor_us"):
        parts.append(f"launch floor {peaks['launch_floor_us']:.1f} us")
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
            env["TORCH_CUDA_ARCH_LIST"] = f"{gpu.capability[0]}.{gpu.capability[1]}"

    has_ninja = _module_available("ninja") or shutil.which("ninja") is not None
    backends = {
        "triton": _module_available("triton") and gpu is not None,
        "cuda": nvcc_ver is not None and has_ninja and gpu is not None,
        "nvrtc": _module_available("cuda.core") and gpu is not None,
        "cute": _module_available("cutlass") and gpu is not None,
        "tilelang": _module_available("tilelang") and nvcc_ver is not None and gpu is not None,
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
