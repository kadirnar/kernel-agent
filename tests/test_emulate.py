"""Emulating an older GPU on the development GPU (#252, ``emulate.py``): the facts of the
emulated GPU in the toolchain, the builds' targets, the refusals, the Triton hook (compiled
only), NVRTC's target, the records that carry ``emulated`` and the caches that stay empty.
CPU tests on a faked RTX 5070 Ti (sm_120) unless marked ``gpu``."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_agent import cli, emulate, gpu_arch, gpulock, precisions, probes, selftest, toolchain
from kernel_agent.kernels import mma_peaks, roofline, tuned

GiB = 1024**3


def props(capability=(12, 0), memory_gb=15.5):
    """torch's device properties of a faked GPU (an RTX 5070 Ti by default)."""
    return SimpleNamespace(
        name="NVIDIA GeForce RTX 5070 Ti" if capability == (12, 0) else "NVIDIA A10",
        major=capability[0],
        minor=capability[1],
        total_memory=int(memory_gb * GiB),
        multi_processor_count=70,
        L2_cache_size=48 * 1024**2,
        shared_memory_per_block_optin=101376,
        shared_memory_per_multiprocessor=102400,
    )


@pytest.fixture
def fake_gpu(monkeypatch, tmp_path):
    """A faked GPU (``fake_gpu(capability)``) with a faked nvcc 13.0 for ``toolchain.setup``;
    the emulation variables are unset unless a test sets them."""
    import torch

    names = (emulate.ENV, emulate.MEM_ENV, "TORCH_EXTENSIONS_DIR", "CUDA_CACHE_MAXSIZE")
    for name in (*names, "TORCHINDUCTOR_CACHE_DIR"):
        monkeypatch.delenv(name, raising=False)
    nvcc = tmp_path / "cuda" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("")
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "cuda"))
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "12.0")  # replaced under emulation
    monkeypatch.setattr(toolchain, "_nvcc_version", lambda path: (13, 0))
    monkeypatch.setattr(toolchain, "_gcc_major", lambda: None)
    monkeypatch.setattr(toolchain, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def use(capability=(12, 0), memory_gb=15.5):
        monkeypatch.setattr(
            emulate, "real_properties", lambda device=None: props(capability, memory_gb)
        )
        toolchain.setup.cache_clear()

    use()
    yield use
    toolchain.setup.cache_clear()


def test_architectures_parse_and_refusals():
    assert emulate.parse("sm_86") == (8, 6) and emulate.parse("8.6") == (8, 6)
    assert emulate.parse("compute_75") == (7, 5) and emulate.parse("sm_120") == (12, 0)
    with pytest.raises(ValueError):
        emulate.parse("ampere")
    for arch in ("sm_75", "sm_80", "sm_86", "sm_89"):
        assert emulate.refusal(emulate.parse(arch), (12, 0)) is None
    assert "sm_90a" in (emulate.refusal((9, 0), (12, 0)) or "")  # arch-specific PTX
    assert "newer than this GPU (sm_86)" in (emulate.refusal((8, 9), (8, 6)) or "")
    assert "not a known architecture" in (emulate.refusal((7, 2), (12, 0)) or "")
    assert emulate.arch_dir("sm_86", None, "/x/ext") == os.path.join("/x/ext", "emulate-sm_86")
    assert emulate.arch_dir("sm_86", "/x/ext/emulate-sm_86", "/y") == "/x/ext/emulate-sm_86"


def test_setup_reports_the_emulated_gpu(fake_gpu, monkeypatch):
    assert toolchain.cuda_arch_list((8, 6)) == "8.6"
    assert toolchain.nvrtc_target((12, 0)) == ("sm_120", "cubin")
    monkeypatch.setenv(emulate.ENV, "sm_86")
    tc = toolchain.setup(apply_env=False)
    gpu = tc.gpu
    assert gpu is not None and gpu.capability == (8, 6) and gpu.smem_per_block_kb == 99.0
    assert gpu.name == "NVIDIA GeForce RTX 5070 Ti emulating sm_86" and gpu.emulated_on == "sm_120"
    assert (gpu.sm_count, gpu.l2_cache_mb, gpu.memory_gb) == (70, 48.0, 15.5)  # the real ones
    assert gpu.smem_per_sm_kb == 100.0 and tc.peaks is None
    assert tc.env["TORCH_CUDA_ARCH_LIST"] == "8.6+PTX" == toolchain.cuda_arch_list((8, 6))
    assert tc.env["TORCH_EXTENSIONS_DIR"].endswith("emulate-sm_86")
    assert tc.env["TORCHINDUCTOR_CACHE_DIR"].endswith("emulate-sm_86")  # Inductor's own
    assert tc.env["TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER"] == "0"  # it loads cubins
    assert tc.env["CUTE_DSL_ARCH"] == "sm_86" and int(tc.env["CUDA_CACHE_MAXSIZE"]) == 4 * GiB
    assert toolchain.nvrtc_target(gpu.capability) == ("compute_86", "ptx")
    summary = tc.summary()
    assert "emulating sm_86" in summary and "measured peaks: none under emulation" in summary
    assert "emulated: sm_86 code paths on this sm_120" in summary
    assert "not representative" in summary and "torch._scaled_mm" in summary
    facts = gpu_arch.from_summary(summary)  # what the prompts would read
    assert facts.capability == (8, 6) and facts.name == gpu.name
    for name in emulate.COMPILE_ONLY:  # SASS: compiled, never run (with the reason)
        assert tc.backends.get(name) is not True
        if name in tc.unavailable:
            assert "compile-only under emulation" in tc.unavailable[name]
            assert f"{name} (compile-only under emulation" in summary
    allowed = precisions.allowed("relaxed", None, gpu.capability)
    assert not {"fp8_w8a8", "fp8_mx", "fp4_w4a4"} & set(allowed)
    assert {"fp8_weights", "int8_w8a8", "int8_weights"} <= set(allowed)
    e4m3 = mma_peaks.BY_KEY[mma_peaks.FP8_F32]
    assert "needs sm_89+" in (mma_peaks.available(e4m3, gpu.capability) or "")
    assert mma_peaks.available(e4m3, (12, 0)) is None  # the real GPU has it
    record = emulate.record(gpu)
    assert record is not None and (record["arch"], record["real"]) == ("sm_86", "sm_120")
    assert "not representative" in record["timings"] and "cuDNN" in record["native"]
    assert emulate.record(toolchain.GPUInfo("GPU", (12, 0), 15.5, 70, 48.0)) is None
    assert gpulock.child_env()[emulate.ENV] == "sm_86"  # the children emulate too


def test_emulated_memory_and_refused_emulations(fake_gpu, monkeypatch):
    monkeypatch.setenv(emulate.ENV, "sm_75")
    monkeypatch.setenv(emulate.MEM_ENV, "12")
    gpu = toolchain.setup(apply_env=False).gpu
    assert gpu is not None and gpu.memory_gb == 12.0 and gpu.smem_per_block_kb == 64.0
    assert emulate.overrides(gpu)["max_threads_per_multi_processor"] == 1024
    monkeypatch.setenv(emulate.MEM_ENV, "40")
    toolchain.setup.cache_clear()
    with pytest.raises(emulate.Refused, match=re.escape("this GPU has 15.5 GB")):
        toolchain.setup(apply_env=False)
    monkeypatch.delenv(emulate.MEM_ENV)
    fake_gpu((8, 6), 24.0)  # an A10: neither Hopper nor a newer GPU can be emulated on it
    for arch, why in (("sm_90", "sm_90a"), ("sm_89", "newer than this GPU (sm_86)")):
        monkeypatch.setenv(emulate.ENV, arch)
        toolchain.setup.cache_clear()
        with pytest.raises(emulate.Refused, match=re.escape(why)):
            toolchain.setup(apply_env=False)
    monkeypatch.setenv(emulate.ENV, "banana")
    with pytest.raises(emulate.Refused, match="not a GPU architecture"):
        toolchain.setup(apply_env=False)


def test_properties_report_the_emulated_gpu():
    gpu = toolchain.GPUInfo("X emulating sm_80", (8, 0), 15.5, 70, 48.0, 163.0, 164.0, "sm_120")
    seen = emulate.Properties(props(), emulate.overrides(gpu))
    assert (seen.major, seen.minor, seen.name) == (8, 0, "NVIDIA GeForce RTX 5070 Ti")
    assert seen.shared_memory_per_block_optin == 163 * 1024 and seen.multi_processor_count == 70
    assert seen.max_threads_per_multi_processor == 2048
    with pytest.raises(AttributeError):
        seen.major = 12


def test_cli_refuses_runs_and_bad_architectures(fake_gpu, monkeypatch, capsys):
    monkeypatch.setenv(emulate.ENV, "sm_86")
    for argv in (["optimize", "org/model"], ["improve", "org/model", "--dry-run"]):
        assert cli.main(argv) == 2
        assert "correctness of an older GPU's code paths" in capsys.readouterr().err
    monkeypatch.delenv(emulate.ENV)
    with pytest.raises(SystemExit):
        cli.main(["eval", "capture.pt", "candidate.py", "--emulate-arch", "ampere"])
    fake_gpu((8, 6), 24.0)
    assert cli.main(["eval", "capture.pt", "candidate.py", "--emulate-arch", "sm_89"]) == 2
    assert "cannot emulate sm_89 here" in capsys.readouterr().err
    assert os.environ[emulate.ENV] == "sm_89"  # the flag is the variable (children inherit)


# ------------------------------------------------------------------ Triton (compiled only)


@pytest.fixture
def triton_hook():
    pytest.importorskip("triton")
    yield emulate.install_triton
    emulate.restore()


try:  # the kernels of the Triton tests (compiled only)
    import triton
    import triton.language as tl
except ImportError:  # no Triton: those tests are skipped
    triton = None

if triton is not None:

    @triton.jit
    def _dot(a, b, c, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
        rm, rn, rk = tl.arange(0, BM), tl.arange(0, BN), tl.arange(0, K)
        x = tl.load(a + rm[:, None] * K + rk[None, :])
        y = tl.load(b + rk[:, None] * BN + rn[None, :])
        tl.store(c + rm[:, None] * BN + rn[None, :], tl.dot(x, y).to(tl.float16))

    @triton.jit
    def _transpose(a, c, M: tl.constexpr, N: tl.constexpr):
        rm, rn = tl.arange(0, M), tl.arange(0, N)
        x = tl.load(a + rm[:, None] * N + rn[None, :])
        tl.store(c + rn[:, None] * M + rm[None, :], tl.trans(x))


def _dot_kernel(target):
    """An fp16 ``tl.dot`` (32 x 32 x 32) compiled for ``target``."""
    from triton.compiler import ASTSource

    signature = {"a": "*fp16", "b": "*fp16", "c": "*fp16"}
    signature.update(dict.fromkeys(("K", "BM", "BN"), "constexpr"))
    source = ASTSource(_dot, signature, constexprs={"K": 32, "BM": 32, "BN": 32})
    return triton.compile(source, target=target, options={"num_warps": 4})


def _transpose_kernel(target):
    """fp64 64 x 256 through shared memory: 128 KB, more than sm_75 or sm_86 allow."""
    from triton.compiler import ASTSource

    signature = {"a": "*fp64", "c": "*fp64", "M": "constexpr", "N": "constexpr"}
    source = ASTSource(_transpose, signature, constexprs={"M": 64, "N": 256})
    return triton.compile(source, target=target, options={"num_warps": 4})


def test_triton_compiles_for_the_emulated_gpu_and_loads_its_ptx(triton_hook):
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.driver import CudaDriver
    from triton.compiler.compiler import CompiledKernel
    from triton.runtime.errors import OutOfResources

    original = CompiledKernel._init_handles
    assert triton_hook((7, 5), (12, 0), 64 * 1024)
    target = CudaDriver.get_current_target(None)  # what Triton's JIT compiles for
    assert target == GPUTarget("cuda", 75, 32)
    kernel = _dot_kernel(target)
    ptx = kernel.asm["ptx"]
    assert ".target sm_75" in ptx and "mma" not in ptx  # Turing: tl.dot on CUDA cores
    image = emulate.ptx_image(kernel, 120, 64 * 1024)
    assert image is not None and image.endswith(b"\0") and b".target sm_75" in image
    big = _transpose_kernel(target)
    assert big.metadata.shared > 64 * 1024
    with pytest.raises(OutOfResources):  # as on a T4, before any driver call
        big._init_handles()

    assert triton_hook((8, 6), (12, 0), 99 * 1024)
    target = CudaDriver.get_current_target(None)
    assert target == GPUTarget("cuda", 86, 32)
    assert "mma.sync.aligned.m16n8k16" in _dot_kernel(target).asm["ptx"]
    native = _dot_kernel(GPUTarget("cuda", 120, 32))
    assert emulate.ptx_image(native, 120, 99 * 1024) is None  # the real GPU's: its cubin
    emulate.restore()
    assert CompiledKernel._init_handles is original


# ------------------------------------------------------------------ nothing is kept


def emulated_toolchain(arch="sm_86"):
    cap = emulate.parse(arch)
    gpu = toolchain.GPUInfo(
        f"Fake GPU emulating {arch}", cap, 15.5, 70, 48.0, 99.0, 100.0, emulated_on="sm_120"
    )
    return toolchain.Toolchain(gpu, "2.14", "13.0", None, "13.0", {"triton": True}, [], {})


def test_the_smoke_lists_what_does_not_run_under_emulation():
    tc = emulated_toolchain("sm_89")
    tc.backends.update(cute=False, tilelang=False)  # as setup() leaves them under emulation
    tc.unavailable["cute"] = emulate.compile_only("cute", tc.gpu) or ""
    why = selftest.example_skip("cute_rmsnorm.py", tc, "cute") or ""
    assert "compile-only under emulation" in why and "SASS for sm_89" in why
    assert "needs sm_90" in (selftest.example_skip("cute_sm90_gemm_ws.py", tc, "cute") or "")
    assert selftest.example_skip("triton_fp8_w8a8_gemm.py", tc, "triton") is None  # Ada: runs
    real = toolchain.Toolchain(
        toolchain.GPUInfo("GPU", (12, 0), 15.5, 70, 48.0), "2.14", "13.0", None, "13.0", {}
    )
    assert selftest.example_skip("cute_rmsnorm.py", real, "cute") == (
        "the cute backend is not available"
    )


def test_probes_and_peaks_are_not_cached_under_emulation(tmp_path, monkeypatch):
    monkeypatch.setenv(emulate.ENV, "sm_86")
    monkeypatch.setattr(toolchain, "CACHE_DIR", tmp_path)
    tc = emulated_toolchain()
    monkeypatch.setattr(toolchain, "setup", lambda: tc)
    found = probes.run(True, {"x": lambda: probes.Probe("x", True, "ran")}, capability=(8, 6))
    assert found["emulated"]["arch"] == "sm_86" and found["emulated"]["real"] == "sm_120"
    assert "probes (emulated sm_86):" in probes.describe(found)
    assert list(tmp_path.iterdir()) == []  # no probes cache

    def no_measurement(*args, **kwargs):
        raise AssertionError("peaks measured under emulation")

    monkeypatch.setattr(roofline.subprocess, "run", no_measurement)
    assert roofline.ensure_peaks(remeasure=True) is None and roofline.current_peaks() is None
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(SystemExit):  # the measurement itself refuses to cache them
        roofline.main([])


def test_tuned_configs_are_not_stored_under_emulation(tmp_path, monkeypatch):
    path = tmp_path / "tuned.sqlite"
    versions = {"torch": "2.14", "triton": "3.8"}
    monkeypatch.setenv(emulate.ENV, "sm_86")
    store = tuned.TunedConfigs(path, gpu="Fake GPU sm_86", versions=versions)
    assert store.put("gemm", {"M": 64}, {"BM": 64}) == {"BM": 64}
    assert store.get("gemm", {"M": 64}) == {"BM": 64}  # this process remembers it
    assert store.add_points("gemm", {"M": 64}, [{"config": {"BM": 32}, "score": 1.2}]) == 0
    monkeypatch.delenv(emulate.ENV)
    fresh = tuned.TunedConfigs(path, gpu="Fake GPU sm_86", versions=versions)
    assert fresh.get("gemm", {"M": 64}) is None and fresh.entries() == []
    assert fresh.points("gemm", {"M": 64}) == []


# ------------------------------------------------------------------ on the GPU

CHILD = r"""
import json, torch, triton, triton.language as tl
from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch
from kernel_agent import toolchain

tc = toolchain.setup()

@triton.jit
def add_one(x, y, N: tl.constexpr):
    r = tl.arange(0, N)
    tl.store(y + r, tl.load(x + r) + 1.0)

x = torch.arange(64, device="cuda", dtype=torch.float32)
y = torch.empty_like(x)
compiled = add_one[(1,)](x, y, N=64)
arch, kind = toolchain.nvrtc_target()
SRC = 'extern "C" __global__ void probe(int* a) { if (threadIdx.x == 0) a[0] = __CUDA_ARCH__; }'
program = Program(SRC, code_type="c++", options=ProgramOptions(arch=arch, std="c++17"))
dev = Device(torch.cuda.current_device())
dev.set_current()
seen = torch.zeros(1, dtype=torch.int32, device="cuda")
stream = dev.create_stream(torch.cuda.current_stream())
launch(stream, LaunchConfig(grid=1, block=32), program.compile(kind).get_kernel("probe"),
       seen.data_ptr())
torch.cuda.synchronize()
print(json.dumps({
    "capability": list(torch.cuda.get_device_capability()),
    "smem": torch.cuda.get_device_properties(0).shared_memory_per_block_optin,
    "triton_target": compiled.metadata.target.arch,
    "triton_ok": bool(torch.equal(y, x + 1)),
    "cuda_arch": int(seen.item()),
    "nvrtc": [arch, kind],
    "arch_list": tc.env.get("TORCH_CUDA_ARCH_LIST"),
}))
"""


@pytest.mark.gpu
@pytest.mark.parametrize("arch", ["sm_75", "sm_86"])
def test_an_emulated_process_runs_the_old_code_paths(arch, tmp_path):
    import torch

    if emulate.refusal(emulate.parse(arch), torch.cuda.get_device_capability()) is not None:
        pytest.skip(f"this GPU cannot emulate {arch}")
    env = gpulock.child_env() | {emulate.ENV: arch, "TRITON_CACHE_DIR": str(tmp_path / "t")}
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    script = tmp_path / "child.py"  # Triton reads a @jit function's source from its file
    script.write_text(CHILD)
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=600
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    cap = emulate.parse(arch)
    assert out["capability"] == list(cap) and out["arch_list"] == f"{cap[0]}.{cap[1]}+PTX"
    assert out["smem"] == int(gpu_arch.SMEM_PER_BLOCK_KB[cap] * 1024)
    assert out["triton_target"] == cap[0] * 10 + cap[1] and out["triton_ok"]
    assert out["cuda_arch"] == cap[0] * 100 + cap[1] * 10  # the old code path ran
    assert out["nvrtc"] == [f"compute_{cap[0]}{cap[1]}", "ptx"]
