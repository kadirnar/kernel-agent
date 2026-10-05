"""GPU timing (kernels/bench.py): the clock guard of #81 (CPU, with stand-ins for the
probes) and, on the GPU, a bandwidth-bound cooperative kernel timed after the GPU idled
and the evaluator's stage order."""

import hashlib
import time

import pytest
import torch

from kernel_agent.kernels import bench


class FakeClock:
    """``bench._perf_counter`` that only moves when ``bench._spin`` runs."""

    def __init__(self, step: float = 0.125) -> None:
        self.now, self.step, self.spins = 0.0, step, 0

    def __call__(self) -> float:
        return self.now

    def spin(self) -> None:
        self.now += self.step
        self.spins += 1


@pytest.fixture
def guard(monkeypatch):
    """A fresh process's clock guard state, a fake clock and probes read from a list
    (the last value repeats)."""
    clock = FakeClock()
    probes: list[float] = []

    def probe() -> float:
        return probes.pop(0) if len(probes) > 1 else probes[0]

    for name, value in (
        ("_PEAK_GBPS", 800.0),  # the roofline's dram_gbps
        ("_BEST_GBPS", 0.0),
        ("_CALIBRATED", False),
        ("_SPENT_S", 0.0),
        ("_perf_counter", clock),
        ("_spin", clock.spin),
        ("dram_gbps", probe),
    ):
        monkeypatch.setattr(bench, name, value)
    return clock, probes


def test_ensure_clocks_spins_until_the_memory_clock_is_up(guard):
    clock, probes = guard
    probes[:] = [17.0, 17.0, 400.0, 790.0]  # idle state, half clock, full clock
    assert bench.ensure_clocks() == pytest.approx(790 / 800)
    assert clock.spins == 3
    probes[:] = [780.0]  # already up: one probe, no spinning
    assert bench.ensure_clocks() == pytest.approx(780 / 800) and clock.spins == 3


def test_ensure_clocks_gives_up_on_a_gpu_that_never_gets_there(guard):
    clock, probes = guard
    probes[:] = [300.0]  # busy or capped (or the roofline measured more than a probe reads)
    assert bench.ensure_clocks() == pytest.approx(1.0)  # now relative to the best probe
    assert clock.now == pytest.approx(bench.WARM_MAX_S) and bench._PEAK_GBPS == 0.0
    spins = clock.spins
    assert bench.ensure_clocks() == pytest.approx(1.0) and clock.spins == spins
    probes[:] = [100.0]  # a drop below the best probe still counts, within the budget
    assert bench.ensure_clocks() == pytest.approx(1 / 3)
    assert clock.now == pytest.approx(bench.WARM_BUDGET_S)
    spins = clock.spins
    assert bench.ensure_clocks() == pytest.approx(1 / 3) and clock.spins == spins  # spent


def test_ensure_clocks_without_the_roofline_spins_the_uncalibrated_warm_up(guard, monkeypatch):
    clock, probes = guard
    monkeypatch.setattr(bench, "_PEAK_GBPS", 0.0)
    probes[:] = [17.0, 17.0, 17.0, 800.0]
    assert bench.ensure_clocks() == pytest.approx(1.0)  # 800 is the best probe
    assert clock.now == pytest.approx(bench.WARM_UNCALIBRATED_S)
    probes[:] = [500.0, 790.0]  # the reference is the best probe now
    assert bench.ensure_clocks() == pytest.approx(790 / 800) and clock.spins == 21


def test_time_call_measures_again_below_full_clocks(monkeypatch):
    before: list[float | None] = [0.5, 1.0, 0.4, 0.4, 0.4, 0.4]
    after: list[float | None] = [1.0, 0.95, 0.3, 0.3, 0.3, 0.3]
    measured: list[int] = []

    def measure(fn, fresh, **kw):
        measured.append(len(measured))
        fresh()
        return {"median_ms": float(len(measured))}

    monkeypatch.setattr(bench, "ensure_clocks", lambda: before.pop(0))
    monkeypatch.setattr(bench, "clock_state", lambda: after.pop(0))
    monkeypatch.setattr(bench, "_measure", measure)
    x = torch.zeros(4)
    result = bench.time_call(lambda t: t, (x,), {})
    assert result == {"median_ms": 2.0, "clock": 0.95, "clock_retries": 1}
    result = bench.time_call(lambda t: t, (x,), {})  # never at full clocks: the last one
    assert result["median_ms"] == 2.0 + bench.CLOCK_RETRIES + 1
    assert result["clock"] == 0.3 and result["clock_retries"] == bench.CLOCK_RETRIES
    before[:] = after[:] = [None]  # no probe (no memory for it): measured once
    assert bench.time_call(lambda t: t, (x,), {}) == {"median_ms": float(len(measured))}


def test_median_round_reports_the_lowest_clock_and_all_retries():
    rounds = [
        {"median_ms": 1.0, "clock": 0.98},
        {"median_ms": 3.0, "clock": 0.91, "clock_retries": 2},
        {"median_ms": 2.0, "clock": 1.02, "clock_retries": 1},
    ]
    best = bench.median_round(rounds)
    assert best["median_ms"] == 2.0 and best["spread"] == 1.0
    assert best["clock"] == 0.91 and best["clock_retries"] == 3
    assert "clock" not in bench.median_round([{"median_ms": 1.0}])


# ------------------------------------------------------------------ GPU

COOP_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Streams w in `phases` slices with a grid-wide barrier after each (like the phases of a
// fused decode-step kernel), then sums it into out[0].
__global__ void __launch_bounds__(256) coop_stream(const float4* __restrict__ w,
    float* __restrict__ partial, float* __restrict__ out, long long n4, int phases) {
  cg::grid_group grid = cg::this_grid();
  __shared__ float red[8];
  float acc = 0.f;
  for (int p = 0; p < phases; ++p) {
    const long long begin = n4 * p / phases, end = n4 * (p + 1) / phases;
    for (long long i = begin + grid.thread_rank(); i < end; i += grid.size()) {
      const float4 v = __ldg(w + i);
      acc += v.x + v.y + v.z + v.w;
    }
    grid.sync();
  }
  for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = 0.f;
    for (int i = 0; i < 8; ++i) s += red[i];
    partial[blockIdx.x] = s;
  }
  grid.sync();
  if (grid.thread_rank() == 0) {
    float s = 0.f;
    for (unsigned b = 0; b < gridDim.x; ++b) s += partial[b];
    out[0] = s;
  }
}

// Launches with as many blocks as can be co-resident (at most 4 per SM); returns
// {blocks, co-resident capacity}.
std::vector<int64_t> coop_launch(torch::Tensor w, torch::Tensor partial, torch::Tensor out,
                                 int64_t phases) {
  int dev = 0, sms = 0, per_sm = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, coop_stream, 256, 0));
  const int blocks = sms * std::min(per_sm, 4);
  TORCH_CHECK(blocks > 0 && blocks <= partial.numel(), "partial buffer too small");
  const float4* wp = reinterpret_cast<const float4*>(w.data_ptr<float>());
  float* pp = partial.data_ptr<float>();
  float* op = out.data_ptr<float>();
  long long n4 = w.numel() / 4;
  int ph = (int)phases;
  void* args[] = {(void*)&wp, (void*)&pp, (void*)&op, (void*)&n4, (void*)&ph};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel((void*)coop_stream, dim3(blocks), dim3(256), args,
                                             0, at::cuda::getCurrentCUDAStream()));
  return {blocks, (int64_t)sms * per_sm};
}
"""
COOP_CPP = (
    "std::vector<int64_t> coop_launch(torch::Tensor w, torch::Tensor partial, "
    "torch::Tensor out, int64_t phases);"
)


@pytest.fixture(scope="module")
def coop():
    from kernel_agent import toolchain

    tc = toolchain.setup()  # CUDA_HOME, before cpp_extension reads it
    if not tc.backends.get("cuda"):
        pytest.skip("needs the cuda backend (nvcc)")
    from torch.utils import cpp_extension

    cpp_extension.CUDA_HOME = cpp_extension.CUDA_HOME or tc.cuda_home  # imported earlier
    tag = hashlib.sha1(COOP_SRC.encode()).hexdigest()[:8]
    return cpp_extension.load_inline(
        name=f"ka_test_coop_{tag}",
        cpp_sources=COOP_CPP,
        cuda_sources=COOP_SRC,
        functions=["coop_launch"],
        extra_cuda_cflags=["-O3"],
    )


@pytest.mark.gpu
def test_bandwidth_bound_cooperative_kernel_is_timed_at_full_clocks_after_idle(coop):
    """A DRAM-bound cooperative kernel with grid syncs (like a fused decode step) times the
    same after the GPU idled (its memory clock drops within seconds on GPUs with idle
    performance states) as when it is busy; before #81 it measured up to 2x slower."""
    w = torch.rand(48 * 2**20, device="cuda")  # 192 MiB: DRAM, not L2
    partial = torch.empty(4096, device="cuda")
    out = torch.empty(1, device="cuda")
    blocks, capacity = coop.coop_launch(w, partial, out, 4)
    assert 0 < blocks <= capacity  # all blocks co-resident: the cooperative launch contract
    torch.cuda.synchronize()
    assert out.item() == pytest.approx(w.double().sum().item(), rel=1e-3)

    def step(w, partial, out):
        coop.coop_launch(w, partial, out, 4)
        return out

    args = (w, partial, out)
    kw = {"target_ms": 60.0, "input_sets": 1}
    busy = min(bench.time_call(step, args, {}, **kw)["median_ms"] for _ in range(3))
    time.sleep(3.0)  # idle: the driver lowers the clocks
    after_idle = bench.time_call(step, args, {}, **kw)
    assert after_idle["clock"] >= bench.CLOCK_OK
    assert after_idle["median_ms"] < 1.25 * busy, (after_idle, busy)


@pytest.mark.gpu
def test_evaluator_times_before_the_profiled_pass(tmp_path, monkeypatch):
    """The activity pass's torch.profiler session keeps CUPTI attached: every later launch
    of the process is slower. The evaluator times every case before it."""
    from kernel_agent.agent.prompts import EXAMPLES_DIR
    from kernel_agent.kernels import integrity
    from kernel_agent.kernels.evaluate import evaluate
    from kernel_agent.selftest import make_rmsnorm_capture

    order: list[str] = []
    compare_timing, activity_check = bench.compare_timing, integrity.activity_check

    def timed(*args, **kwargs):
        order.append("timing")
        return compare_timing(*args, **kwargs)

    def profiled(*args, **kwargs):
        order.append("profiler")
        return activity_check(*args, **kwargs)

    monkeypatch.setattr(bench, "compare_timing", timed)  # before evaluate's snapshot
    monkeypatch.setattr(integrity, "activity_check", profiled)
    capture = make_rmsnorm_capture(tmp_path / "rms.pt", hidden=1024)
    result = evaluate(capture, EXAMPLES_DIR / "triton_rmsnorm.py")
    assert result["status"] == "ok", result
    assert order == ["timing", "timing", "profiler"]
    assert all(case["clock"] >= bench.CLOCK_OK for case in result["cases"]), result["cases"]
