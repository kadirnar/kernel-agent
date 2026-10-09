"""``doctor`` probes: GPU features the kernel toolkit relies on, and the versions (#146).

Each probe compiles (and, where cheap, runs) the smallest program that uses a feature
and says whether it works here and how:

* ``dot_scaled``: Triton lowers ``tl.dot_scaled`` (e4m3, ue8m0 scales) to the
  block-scaled ``mma ... block_scale`` (SASS ``QMMA.SF``), the full-rate FP8 instruction
  on sm_120 (docs/RESEARCH-TRITON.md §1.1; a plain ``tl.dot`` on e4m3 runs at half rate
  there; elsewhere plain e4m3 is the full-rate path).
* ``tma``: Triton host TMA descriptors (``TensorDescriptor``) compile to
  ``cp.async.bulk.tensor`` and copy a tile correctly.
* ``pdl``: a programmatic dependent launch (``griddepcontrol``; ``cuda.core``
  ``LaunchConfig(programmatic_stream_serialization=True)``) runs and orders correctly.
* ``green_contexts``: :func:`kernel_agent.concurrency.partition` splits the SMs into two
  disjoint green contexts (how many SMs it grants for 8 asked) whose streams run torch
  work.
* ``helion`` (#229): Helion is installed with a torch and Triton it accepts (a mismatch is
  refused: installing it must not change torch), and the bundled ``helion_rmsnorm.py``
  example compiles with Helion's default config and computes the reference's RMSNorm. Not
  installed: skipped with that reason (the ``helion`` backend is then unavailable).

:func:`run` records the results with :func:`versions` in
``<cache>/probes-<gpu>-torch<version>.json``. A probe that fails says why; none of them
fails ``doctor``. Written for #146 and not yet run on a GPU (``kernel-agent doctor`` runs
them). A probe whose feature the GPU lacks (:data:`ARCHS`: the block-scaled
``tl.dot_scaled`` outside sm_12x, TMA and PDL before sm_90) is skipped with the reason,
not failed (#165).
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import re
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kernel_agent import toolchain


@dataclass
class Probe:
    name: str
    ok: bool | None  # None: skipped (no GPU, the feature's package missing)
    detail: str

    def line(self) -> str:
        mark = {True: "ok", False: "FAILED", None: "skipped"}[self.ok]
        return f"  {self.name}: {mark}: {self.detail}"


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def versions() -> dict[str, str | None]:
    """The versions that decide what compiles and how fast (torch, its CUDA, the driver,
    Triton, cuda.core / cuda.bindings, nvcc, CuTe DSL, TileLang, ncu)."""
    import torch

    tc = toolchain.setup()
    driver = toolchain.driver_cuda_version()
    out: dict[str, str | None] = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "driver_cuda": f"{driver[0]}.{driver[1]}" if driver else None,
        "nvcc": tc.nvcc_version,
        "triton": _version("triton"),
        "cuda-core": _version("cuda-core") or _version("cuda_core"),
        "cuda-bindings": _version("cuda-bindings") or _version("cuda_bindings"),
        "cutlass-dsl": _version("nvidia-cutlass-dsl"),
        "tilelang": _version("tilelang"),
        "helion": _version("helion"),
    }
    from kernel_agent.kernels import ncu

    if (path := ncu.find_ncu()) is not None:
        out["ncu"] = ncu.ncu_version(path) or "?"
    return out


def _sass(cubin: bytes) -> str:
    """SASS of a cubin through the ``nvdisasm`` Triton bundles ("" without one)."""
    import triton

    tool = Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "nvdisasm"
    if not tool.is_file():
        return ""
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(cubin)
        f.flush()
        proc = subprocess.run([str(tool), f.name], capture_output=True, text=True, timeout=60)
    return proc.stdout


def probe_dot_scaled() -> Probe:
    """``tl.dot_scaled`` e4m3 x e4m3 with ue8m0 scales: compiled, PTX and SASS checked."""
    import torch

    from kernel_agent import probe_kernels

    bm = bn = 64
    bk = 128
    a = torch.zeros(bm, bk, device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.zeros(bk, bn, device="cuda", dtype=torch.float8_e4m3fn)
    sa = torch.full((bm, bk // 32), 127, device="cuda", dtype=torch.uint8)
    sb = torch.full((bn, bk // 32), 127, device="cuda", dtype=torch.uint8)
    c = torch.empty(bm, bn, device="cuda", dtype=torch.float32)
    kernel = probe_kernels.dot_scaled_kernel.warmup(
        a, b, sa, sb, c, BM=bm, BN=bn, BK=bk, grid=(1,), num_warps=4
    )
    ptx = str(kernel.asm.get("ptx", ""))
    sass = _sass(kernel.asm["cubin"]) if "cubin" in kernel.asm else ""
    ops = sorted(set(re.findall(r"\b((?:UTC)?[QHO]MMA[.A-Z0-9_]*|UTC\w*MMA[.A-Z0-9_]*)", sass)))
    block_scale = "block_scale" in ptx
    # sm_12x: the block-scaled mma.sync is SASS QMMA.SF; sm_100: tcgen05.mma ... block_scale
    geforce = torch.cuda.get_device_capability()[0] == 12
    sf = any(op.startswith("QMMA.SF") for op in ops)
    full_rate = block_scale and (not sass or not geforce or sf)
    detail = (
        "tl.dot_scaled lowers to a block_scale MMA"
        if block_scale
        else "tl.dot_scaled does NOT lower to a block_scale MMA (Triton emulates it through "
        "bf16: half-rate FP8 on sm_120)"
    )
    if ops:
        detail += f" (SASS {', '.join(ops[:3])})"
    return Probe("dot_scaled", full_rate, detail)


def probe_tma() -> Probe:
    """A tile copied through two host-built TMA descriptors."""
    import torch
    from triton.tools.tensor_descriptor import TensorDescriptor

    from kernel_agent import probe_kernels

    bm, bn = 64, 64
    src = torch.arange(2 * bm * bn, device="cuda", dtype=torch.float32).view(2 * bm, bn)
    dst = torch.zeros_like(src)
    kernel = probe_kernels.tma_copy_kernel[(2,)](
        TensorDescriptor.from_tensor(src, [bm, bn]),
        TensorDescriptor.from_tensor(dst, [bm, bn]),
        BM=bm,
        BN=bn,
    )
    torch.cuda.synchronize()
    copied = bool(torch.equal(src, dst))
    ptx = str(getattr(kernel, "asm", {}).get("ptx", "")) if kernel is not None else ""
    bulk = "cp.async.bulk.tensor" in ptx
    detail = "TensorDescriptor tiles " + ("copied" if copied else "WRONG after the copy")
    if ptx:
        detail += ", PTX " + ("uses" if bulk else "does NOT use") + " cp.async.bulk.tensor"
    return Probe("tma", copied and (bulk or not ptx), detail)


PDL_SRC = r"""
extern "C" __global__ void pdl_producer(int* x) {
  x[threadIdx.x] = (int)threadIdx.x + 1;
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
extern "C" __global__ void pdl_consumer(const int* x, int* y) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  y[threadIdx.x] = 2 * x[threadIdx.x];
}
"""


def probe_pdl() -> Probe:
    """Producer + consumer launched with programmatic stream serialization (NVRTC)."""
    import torch
    from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch

    index = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(index)
    if major < 9:
        return Probe("pdl", None, f"needs sm_90+, this GPU is sm_{major}{minor}")
    dev = Device(index)
    dev.set_current()
    options = ProgramOptions(arch=f"sm_{major}{minor}", std="c++17")
    code = Program(PDL_SRC, code_type="c++", options=options).compile("cubin")
    producer, consumer = code.get_kernel("pdl_producer"), code.get_kernel("pdl_consumer")
    stream = dev.create_stream(torch.cuda.current_stream())
    x = torch.zeros(32, device="cuda", dtype=torch.int32)
    y = torch.zeros(32, device="cuda", dtype=torch.int32)
    launch(stream, LaunchConfig(grid=1, block=32), producer, x.data_ptr())
    pdl = LaunchConfig(grid=1, block=32, programmatic_stream_serialization=True)
    launch(stream, pdl, consumer, x.data_ptr(), y.data_ptr())
    torch.cuda.synchronize()
    want = 2 * torch.arange(1, 33, device="cuda", dtype=torch.int32)
    ok = bool(torch.equal(y, want))
    return Probe(
        "pdl",
        ok,
        "griddepcontrol + programmatic stream serialization: "
        + ("the dependent kernel saw the producer's writes" if ok else "WRONG result"),
    )


def probe_green_contexts(sms: int = 8) -> Probe:
    """Two disjoint SM partitions (:func:`kernel_agent.concurrency.partition`, at least
    ``sms`` SMs and the rest): SMs granted, torch work on the small one's stream."""
    import torch

    from kernel_agent import concurrency

    if (why := concurrency.partition_unsupported()) is not None:
        return Probe("green_contexts", None, why)
    split = concurrency.partition(sms=sms)
    if split is None:
        return Probe("green_contexts", False, str(concurrency.partition_error()))
    part, rest = split
    stream = part.stream()
    with torch.cuda.stream(stream):
        x = torch.full((1024,), 2.0, device="cuda")
        total = (x * x).sum()
    stream.synchronize()
    ok = float(total.item()) == 4096.0
    return Probe(
        "green_contexts",
        ok,
        f"concurrency.partition(sms={sms}): {part.sms} + {rest.sms} SMs"
        + (", torch ops run on a partition's stream" if ok else ", WRONG result on its stream")
        + " (graphs captured outside a partition and device-sized cooperative grids ignore it)",
    )


def probe_helion() -> Probe:
    """Helion (#229): installed with a torch / Triton it accepts (:func:`kernels.helion_tune.
    status`), and the bundled ``helion_rmsnorm.py`` example compiled with Helion's default
    config (no autotuning) and run on a bf16 RMSNorm, equal to the reference within bf16
    rounding. Helion lists A100 / H100 / B200; elsewhere this probe is the evidence."""
    import torch

    from kernel_agent.kernels import helion_tune

    found = helion_tune.status()
    if not found["ok"]:
        return Probe("helion", None if found["version"] is None else False, str(found["why"]))
    from kernel_agent.agent.prompts import EXAMPLES_DIR
    from kernel_agent.kernels.evaluate import load_candidate_module
    from kernel_agent.selftest import RMSNorm

    torch.manual_seed(0)
    reference = RMSNorm(2048, eps=1e-5).cuda().to(torch.bfloat16)
    with torch.no_grad():
        reference.weight.copy_(torch.randn(2048) * 0.1 + 1)
    x = torch.randn(4, 64, 2048, device="cuda", dtype=torch.bfloat16)
    module = load_candidate_module(EXAMPLES_DIR / "helion_rmsnorm.py")
    began = time.monotonic()
    with torch.no_grad():
        got, want = module.build(reference)(x), reference(x)
    torch.cuda.synchronize()
    seconds = time.monotonic() - began
    error = float((got.float() - want.float()).abs().max())
    ok = error <= 0.0625  # bf16 rounding of values around 4
    return Probe(
        "helion",
        ok,
        f"helion {found['version']}: the RMSNorm example compiled and ran in {seconds:.1f} s, "
        f"max abs error {error:.3g}" + ("" if ok else " (WRONG result)"),
    )


PROBES: dict[str, Callable[[], Probe]] = {
    "dot_scaled": probe_dot_scaled,
    "tma": probe_tma,
    "pdl": probe_pdl,
    "green_contexts": probe_green_contexts,
    "helion": probe_helion,
}
#: The package each probe needs (skipped without it; ``helion`` says why itself).
NEEDS = {"dot_scaled": "triton", "tma": "triton", "pdl": "cuda.core", "green_contexts": "torch"}
#: The GPUs each probe's feature exists on (``gpu_arch.supports``) and what it is: skipped
#: elsewhere with the reason, never reported as a failure (#165).
ARCHS = {
    "dot_scaled": ("sm_12x", "block-scaled mma.sync; plain e4m3 tl.dot is full rate elsewhere"),
    "tma": ("sm_90+", "TMA (cp.async.bulk.tensor)"),
    "pdl": ("sm_90+", "griddepcontrol"),
}


def run(
    gpu: bool,
    probes: dict[str, Callable[[], Probe]] | None = None,
    capability: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Every probe (skipped without a GPU, its package, or on a GPU without its feature:
    :data:`ARCHS`; ``capability`` None: this machine's GPU), with :func:`versions`; written
    to the cache next to the peaks."""
    from kernel_agent.gpu_arch import arch_of, supports

    if gpu and capability is None:
        found_gpu = getattr(toolchain.setup(), "gpu", None)
        capability = tuple(found_gpu.capability) if found_gpu is not None else None
    found: list[Probe] = []
    for name, probe in (PROBES if probes is None else probes).items():
        if not gpu:
            found.append(Probe(name, None, "no CUDA GPU"))
            continue
        spec, what = ARCHS.get(name, (None, ""))
        if capability is not None and not supports(spec, capability):
            found.append(
                Probe(name, None, f"needs {spec} ({what}), this GPU is {arch_of(capability)}")
            )
            continue
        need = NEEDS.get(name)
        if need and not toolchain._module_available(need):
            found.append(Probe(name, None, f"{need} is not installed"))
            continue
        try:
            found.append(probe())
        except Exception as exc:  # the feature does not work here: say why
            found.append(Probe(name, False, f"{type(exc).__name__}: {exc}"[:300]))
    result = {
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "versions": versions(),
        "probes": [asdict(p) for p in found],
    }
    tc = toolchain.setup()
    if tc.gpu is not None:
        from kernel_agent.workspace import write_json

        name = toolchain.peaks_path(tc.gpu.name, tc.torch_version).name
        with contextlib.suppress(OSError):
            write_json(toolchain.CACHE_DIR / name.replace("peaks-", "probes-", 1), result)
    return result


def describe(result: dict[str, Any]) -> str:
    """``doctor`` lines."""
    lines = ["versions: " + ", ".join(f"{k} {v}" for k, v in result["versions"].items() if v)]
    lines.append("probes:")
    lines += [Probe(**p).line() for p in result["probes"]]
    return "\n".join(lines)
