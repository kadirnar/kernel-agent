"""Emulate an older GPU on this one (#252): its exact code paths through the driver's PTX JIT.

``KERNEL_AGENT_EMULATE_ARCH=sm_86`` (or ``--emulate-arch sm_86`` on ``doctor``, ``eval``,
``recheck`` and ``memcheck``) makes a process and its children (the variable is inherited:
:func:`kernel_agent.gpulock.child_env` copies the environment) build and run every kernel as
on a GPU of that architecture: the code behind ``__CUDA_ARCH__ < 890``, Triton's sm_8x / sm_75
lowering, the megakernel's ``cp.async`` page loads, the e4m3 software conversion. The driver
JIT-compiles the older architecture's PTX for the real GPU (PTX runs on its own and every
newer architecture), so an RTX 5070 Ti (sm_120) checks the correctness of sm_75, sm_80, sm_86
and sm_89 code paths. Only architectures at or below the real one can be emulated, and not the
architecture-specific targets (``sm_90a``, ``sm_100a``: their PTX runs on no other GPU).

What changes in an emulated process (:func:`apply` and :func:`install`, both from
:func:`kernel_agent.toolchain.setup`):

* the GPU's facts: :class:`~kernel_agent.toolchain.GPUInfo` has the emulated capability, the
  name ``"<real name> emulating sm_86"`` and that architecture's shared memory per block
  (:data:`kernel_agent.gpu_arch.SMEM_PER_BLOCK_KB`), per SM and threads per SM; so the
  precisions it cannot run, the examples' ``ARCHS`` and the toolchain summary follow. The SM
  count and the L2 stay the real GPU's; :data:`MEM_ENV` lowers the memory (a per-process
  memory fraction);
* ``torch.cuda.get_device_capability`` and ``get_device_properties`` (``major``, ``minor``, the
  shared memory and thread limits, ``total_memory``) report them, so the checks of examples and
  candidates (``>= (8, 0)``) take the old GPU's branch. That also steers Inductor's Python-level
  choices in this process: ``torch.compile`` builds its Triton kernels for the emulated GPU
  (in its own ``TORCHINDUCTOR_CACHE_DIR``, without the static launcher, which loads cubins);
* builds: ``TORCH_CUDA_ARCH_LIST=8.6+PTX`` in a per-architecture ``TORCH_EXTENSIONS_DIR``
  (``toolchain.cuda_arch_list``), NVRTC ``compute_86`` PTX (``toolchain.nvrtc_target``),
  Triton compiled for sm_86 and loaded from its PTX, refused like on the old GPU when its
  shared memory exceeds the emulated opt-in limit (Triton's ``OutOfResources``: an autotuner
  skips that config). CuTe DSL (``CUTE_DSL_ARCH``) and TileLang compile for the emulated
  architecture but cannot run: their artefacts are SASS (:data:`COMPILE_ONLY`);
* library ops are not emulated: cuBLAS / cuBLASLt, cuDNN, SDPA, ``torch._scaled_mm`` and
  ``torch._int_mm`` run natively (:data:`NATIVE`), and every emulated result says so;
* records: evaluation, memcheck and probe results carry ``emulated`` (:func:`record`: ``{"arch":
  "sm_86", "real": "sm_120", ...}``); timings are not representative (the SASS is the real
  GPU's, as are its SMs, DRAM and L2). Nothing measured under emulation is kept: no peaks are
  measured or loaded, no probes cache, no backend track record (``backends.record_run``), no
  library entry (``library.store_run``), no tuned configs or search points (``tuned.py``).
  ``optimize``, ``analyze``, ``improve``, ``resume`` and ``integrate`` refuse to run under
  emulation (:func:`refuse_runs`): it is a correctness tool.

It misses what only the old GPU shows: its SASS (older ``ptxas`` bugs, registers, occupancy),
race windows that depend on its SM count or clocks, and cuBLAS / cuDNN behaviour there. An
emulated run is never presented as a run on that hardware.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

#: The architecture to emulate (``sm_75``, ``sm_80``, ``sm_86``, ``sm_89``).
ENV = "KERNEL_AGENT_EMULATE_ARCH"
#: The memory of the emulated GPU in GB (a T4: 16, an A10: 24): at most the real GPU's.
MEM_ENV = "KERNEL_AGENT_EMULATE_MEM_GB"
#: Library ops that run natively on the real GPU under emulation.
NATIVE = ("cuBLAS / cuBLASLt", "cuDNN", "SDPA", "torch._scaled_mm", "torch._int_mm")
#: Backends that compile for the emulated architecture but cannot run under emulation: they
#: build SASS (no PTX the driver could JIT-compile for the real GPU).
COMPILE_ONLY = {"cute": "CuTe DSL (CUTE_DSL_ARCH)", "tilelang": "TileLang"}
#: The driver's JIT cache (bytes): the first load of every emulated kernel is compiled from
#: its PTX, later processes reuse the cached SASS.
JIT_CACHE_BYTES = 4 * 1024**3
#: Shared memory per SM (KB) and resident threads per SM of the architectures emulated (CUDA
#: Programming Guide, compute capabilities); the opt-in limit per block is ``gpu_arch``'s.
SMEM_PER_SM_KB = {(7, 5): 64.0, (8, 0): 164.0, (8, 6): 100.0, (8, 7): 164.0, (8, 9): 100.0}
THREADS_PER_SM = {(7, 5): 1024, (8, 0): 2048, (8, 6): 1536, (8, 7): 2048, (8, 9): 1536}

_ARCH = re.compile(r"(?:sm_|compute_)?(\d{1,2})\.?(\d)[a-z]?")


class Refused(RuntimeError):
    """The requested emulation cannot run here (the message says why)."""


def active() -> bool:
    """Whether this process is asked to emulate a GPU (:data:`ENV` set)."""
    return bool(os.environ.get(ENV, "").strip())


def parse(raw: str) -> tuple[int, int]:
    """``(8, 6)`` for ``sm_86`` (also ``86``, ``8.6``, ``compute_86``; ``(12, 0)`` for
    ``sm_120``). ValueError: no architecture."""
    match = _ARCH.fullmatch(raw.strip().lower())
    if not match:
        raise ValueError(f"{raw!r} is not a GPU architecture (sm_75, sm_80, sm_86, sm_89)")
    return int(match.group(1)), int(match.group(2))


def requested() -> tuple[int, int] | None:
    """The capability :data:`ENV` asks for (None: none). Refused: not an architecture."""
    if not active():
        return None
    try:
        return parse(os.environ[ENV])
    except ValueError as exc:
        raise Refused(f"{ENV}: {exc}") from None


def refusal(want: tuple[int, int], real: tuple[int, int]) -> str | None:
    """Why a GPU of capability ``real`` cannot emulate ``want`` (None: it can)."""
    from kernel_agent.gpu_arch import SMEM_PER_BLOCK_KB, arch_of
    from kernel_agent.toolchain import ARCH_SPECIFIC

    arch, here = arch_of(want), arch_of(real)
    if want in ARCH_SPECIFIC:
        return (
            f"{arch}'s full-rate instructions exist only in its architecture-specific target "
            f"({arch}a), whose code runs on no other GPU: it cannot be emulated"
        )
    if want not in SMEM_PER_BLOCK_KB:
        known = ", ".join(
            arch_of(c) or "" for c in sorted(SMEM_PER_BLOCK_KB) if c not in ARCH_SPECIFIC
        )
        return f"{arch} is not a known architecture (one of {known})"
    if want > tuple(real):
        return (
            f"{arch} is newer than this GPU ({here}): PTX runs only on its own architecture "
            "and newer ones, so a GPU emulates only architectures at or below its own"
        )
    return None


def memory_gb(real_gb: float) -> float | None:
    """The emulated GPU's memory (:data:`MEM_ENV`; None: the real GPU's). Refused: not a
    number, or more than ``real_gb``."""
    raw = os.environ.get(MEM_ENV, "").strip()
    if not raw:
        return None
    try:
        gb = float(raw)
    except ValueError:
        raise Refused(f"{MEM_ENV}={raw!r} is not a number of GB") from None
    if not 0 < gb <= real_gb:
        raise Refused(f"{MEM_ENV}={raw}: this GPU has {real_gb:.1f} GB (0 < GB <= that)")
    return gb


def apply(gpu: Any) -> Any:
    """The :class:`~kernel_agent.toolchain.GPUInfo` this process works with: ``gpu`` (the real
    one) itself, or under emulation the emulated GPU's facts on it (``emulated_on`` names the
    real architecture). Refused: the emulation cannot run on ``gpu``."""
    want = requested()
    if want is None or gpu is None:
        return gpu
    from kernel_agent.gpu_arch import SMEM_PER_BLOCK_KB, arch_of

    if (why := refusal(want, tuple(gpu.capability))) is not None:
        raise Refused(f"cannot emulate {arch_of(want)} here: {why}")
    return replace(
        gpu,
        name=f"{gpu.name} emulating {arch_of(want)}",
        capability=want,
        memory_gb=memory_gb(gpu.memory_gb) or gpu.memory_gb,
        smem_per_block_kb=SMEM_PER_BLOCK_KB[want],
        smem_per_sm_kb=SMEM_PER_SM_KB.get(want, gpu.smem_per_sm_kb),
        emulated_on=gpu.arch,
    )


def record(gpu: Any = None) -> dict[str, Any] | None:
    """The ``emulated`` field of a record made in this process (``gpu``: default the
    toolchain's): the emulated and the real architecture, the library ops that ran natively,
    and that its timings are not representative. None: not emulated."""
    if gpu is None:
        if not active():
            return None
        from kernel_agent import toolchain

        gpu = toolchain.setup().gpu
    real = getattr(gpu, "emulated_on", None)
    if gpu is None or not real:
        return None
    return {
        "arch": gpu.arch,
        "real": real,
        "native": f"{', '.join(NATIVE)} ran natively ({real})",
        "timings": f"not representative (the SASS, SMs, DRAM and L2 are the {real}'s)",
    }


def summary_line(gpu: Any) -> str | None:
    """The toolchain summary's line about an emulated ``gpu`` (None: a real one)."""
    if not (real := getattr(gpu, "emulated_on", None)):
        return None
    return (
        f"emulated: {gpu.arch} code paths on this {real} through the driver's PTX JIT ({ENV}): "
        "correctness only, timings not representative; "
        f"{', '.join(NATIVE)} run natively; {', '.join(sorted(COMPILE_ONLY))} compile only"
    )


def compile_only(backend: str, gpu: Any) -> str | None:
    """Why ``backend`` does not run on an emulated ``gpu`` (None: it runs, or ``gpu`` is a
    real GPU)."""
    real = getattr(gpu, "emulated_on", None)
    if not real or backend not in COMPILE_ONLY:
        return None
    return (
        f"compile-only under emulation: {COMPILE_ONLY[backend]} builds SASS for {gpu.arch}, "
        f"which this {real} cannot load"
    )


def refuse_runs(command: str) -> str | None:
    """Why ``command`` (``optimize``, ``improve``, ...) must not run in this process: under
    emulation nothing is measured that a run could keep (None: it may run)."""
    if not active():
        return None
    return (
        f"{command} does not run under {ENV}={os.environ[ENV]}: emulation checks the "
        "correctness of an older GPU's code paths, its timings are not that GPU's (use doctor "
        "--smoke, eval, recheck or memcheck with --emulate-arch)"
    )


def arch_dir(arch: str, current: str | None, default: str) -> str:
    """A build cache of an emulated ``arch`` (``TORCH_EXTENSIONS_DIR``, Inductor's):
    ``emulate-<arch>`` inside the ``current`` one (else torch's ``default``), so nothing built
    for the real GPU is reused under emulation, nor the other way round."""
    base = current or default
    leaf = f"emulate-{arch}"
    return base if os.path.basename(os.path.normpath(base)) == leaf else os.path.join(base, leaf)


# ------------------------------------------------------------------ the process

#: The originals :func:`install` replaced (:func:`restore` puts them back; tests).
_SAVED: dict[str, Any] = {}


class Properties:
    """A device's properties (``torch.cuda.get_device_properties``) with the emulated GPU's
    capability and limits; everything else is the real device's."""

    def __init__(self, real: Any, overrides: Mapping[str, Any]) -> None:
        self.__dict__["_real"] = real
        self.__dict__["_overrides"] = dict(overrides)

    def __getattr__(self, name: str) -> Any:
        if name in ("_real", "_overrides"):  # not set yet (a copy being built)
            raise AttributeError(name)
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._real, name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"device properties are read-only ({name})")

    def __repr__(self) -> str:
        return f"{self._real!r} (emulated: {self._overrides})"


def overrides(gpu: Any) -> dict[str, Any]:
    """What :class:`Properties` reports for the emulated ``gpu`` instead of the real device."""
    out: dict[str, Any] = {
        "major": int(gpu.capability[0]),
        "minor": int(gpu.capability[1]),
        "shared_memory_per_block_optin": int(gpu.smem_per_block_kb * 1024),
        "shared_memory_per_multiprocessor": int(gpu.smem_per_sm_kb * 1024),
        "total_memory": int(gpu.memory_gb * 1024**3),
    }
    if (threads := THREADS_PER_SM.get(tuple(gpu.capability))) is not None:
        out["max_threads_per_multi_processor"] = threads
    return out


def real_properties(device: Any = None) -> Any:
    """The real device's properties, also once :func:`install` replaced them."""
    import torch

    cuda: Any = torch.cuda
    original = _SAVED.get("props")
    if original is None:
        return cuda.get_device_properties(device)
    return original(cuda._get_device_index(device, optional=True))


def install(gpu: Any) -> None:
    """Make this process see the emulated ``gpu`` (an emulated ``GPUInfo``; once per process):
    torch's device properties and capability, Triton's target and kernel loading, the memory
    limit."""
    if not getattr(gpu, "emulated_on", None) or _SAVED.get("installed") == gpu.capability:
        return
    import torch

    from kernel_agent.gpu_arch import capability_of

    cuda: Any = torch.cuda
    cuda.init()  # defines torch.cuda._get_device_properties
    original = _SAVED.setdefault("props", cuda._get_device_properties)
    fields = overrides(gpu)

    def emulated(device: int) -> Properties:
        return Properties(original(device), fields)

    # get_device_properties and get_device_capability (however imported) call this one
    cuda._get_device_properties = emulated
    real = capability_of(gpu.emulated_on) or tuple(gpu.capability)
    install_triton(tuple(gpu.capability), real, fields["shared_memory_per_block_optin"])
    inductor: Any = sys.modules.get("torch._inductor.config")
    if inductor is not None:
        # imported before TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0: its launcher loads cubins
        inductor.use_static_cuda_launcher = False
    total = original(cuda.current_device()).total_memory
    if fields["total_memory"] < total:
        cuda.set_per_process_memory_fraction(fields["total_memory"] / total)
    _SAVED["installed"] = gpu.capability


def ptx_image(kernel: Any, real: int, limit: int) -> bytes | None:
    """What a Triton ``CompiledKernel`` compiled for an emulated architecture loads on the
    real GPU (``real``: its ``major * 10 + minor``): its PTX (the cubin of an older
    architecture does not load), None for a kernel of the real architecture. Raises Triton's
    ``OutOfResources`` when it needs more shared memory than ``limit`` bytes, as the old GPU
    would."""
    from triton.runtime.errors import OutOfResources

    target = kernel.metadata.target
    if getattr(target, "backend", "cuda") != "cuda" or int(target.arch) == real:
        return None
    shared = int(getattr(kernel.metadata, "shared", 0) or 0)
    if shared > limit:
        raise OutOfResources(shared, limit, "shared memory")
    return str(kernel.asm["ptx"]).encode() + b"\0"  # cuModuleLoadData JIT-compiles PTX


def install_triton(capability: tuple[int, ...], real: tuple[int, ...], smem_limit: int) -> bool:
    """Triton compiles for ``capability`` (its CUDA driver's current target) and loads those
    kernels from their PTX on the ``real`` GPU (:func:`ptx_image`); False without Triton."""
    try:
        from triton.backends.compiler import GPUTarget
        from triton.backends.nvidia.driver import CudaDriver
        from triton.compiler.compiler import CompiledKernel
    except ImportError:
        return False
    cc = int(capability[0]) * 10 + int(capability[1])
    here = int(real[0]) * 10 + int(real[1])
    _SAVED.setdefault("triton_target", CudaDriver.get_current_target)
    original = _SAVED.setdefault("triton_init", CompiledKernel._init_handles)

    def get_current_target(self: Any) -> Any:
        return GPUTarget("cuda", cc, 32)

    def _init_handles(self: Any) -> Any:
        if self.module is None and (image := ptx_image(self, here, smem_limit)) is not None:
            self.kernel = image
        return original(self)

    CudaDriver.get_current_target = get_current_target
    CompiledKernel._init_handles = _init_handles
    return True


def restore() -> None:
    """Undo :func:`install` / :func:`install_triton` in this process (tests)."""
    if "props" in _SAVED:
        import torch

        cuda: Any = torch.cuda
        cuda._get_device_properties = _SAVED.pop("props")
    if "triton_target" in _SAVED:
        from triton.backends.nvidia.driver import CudaDriver
        from triton.compiler.compiler import CompiledKernel

        CudaDriver.get_current_target = _SAVED.pop("triton_target")
        CompiledKernel._init_handles = _SAVED.pop("triton_init")
    _SAVED.clear()
