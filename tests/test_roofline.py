"""Roofline / speed of light: FLOP and byte counting (CPU), SOL classification, advice."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from kernel_agent import ledger, toolchain
from kernel_agent.agent.tools import compact
from kernel_agent.budget import SOL_STOP_PCT, Budget
from kernel_agent.kernels import roofline
from kernel_agent.kernels.roofline import (
    CaseCost,
    annotate,
    apply_sol,
    attention_density,
    count_case,
    sol_signal,
    sol_time,
)
from kernel_agent.selftest import RMSNorm
from kernel_agent.workspace import RunDir, append_jsonl

F32 = 4
PEAKS = {
    "dram_gbps": 1000.0,
    "l2_gbps": 4000.0,
    "l2_mb": 1.0,
    "tflops": {"bfloat16": 100.0, "float16": 100.0, "float32": 25.0},
    "launch_floor_us": 10.0,
}


# ------------------------------------------------------------ counting


def test_linear_flops_and_bytes():
    torch.manual_seed(0)
    lin = nn.Linear(64, 128)
    x = torch.randn(4, 64)
    cost = count_case(lin, (x,), {})
    assert cost.flops == {"float32": 2 * 4 * 64 * 128}
    assert cost.read_bytes == (4 * 64 + 128 * 64 + 128) * F32  # x + weight + bias
    assert cost.write_bytes == 4 * 128 * F32
    assert torch.equal(lin(x), lin(x))  # inputs and module untouched


class NormWithJunk(RMSNorm):
    def __init__(self, hidden: int) -> None:
        super().__init__(hidden)
        self.register_buffer("unused_table", torch.zeros(4096, hidden))


def test_rmsnorm_counts_used_tensors_once():
    norm = NormWithJunk(256).to(torch.bfloat16)
    x = torch.randn(8, 256, dtype=torch.bfloat16)
    cost = count_case(norm, (x,), {})
    assert cost.total_flops == 0  # element-wise math is free
    # x (read by the fp32 upcast), weight, output; the unused buffer and the fp32
    # intermediates are not traffic the kernel has to do
    assert cost.min_bytes == 8 * 256 * 2 + 256 * 2 + 8 * 256 * 2


class Attention(nn.Module):
    def forward(self, q, k, v, mask=None, causal=False):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal)


def test_sdpa_flops_follow_the_mask():
    q, k, v = (torch.randn(1, 4, 16, 32) for _ in range(3))
    full = 2 * 4 * 16 * 16 * (32 + 32)
    io = 4 * 4 * 16 * 32 * F32  # q, k, v, out
    cost = count_case(Attention(), (q, k, v), {})
    assert cost.flops == {"float32": full} and cost.min_bytes == io

    causal = count_case(Attention(), (q, k, v), {"causal": True})
    assert causal.total_flops == full * (16 * 17 // 2) // (16 * 16)
    assert causal.min_bytes == io  # every key is attended by some query

    keep = torch.arange(16) < 4  # only the first 4 keys are valid
    mask = torch.where(keep, 0.0, torch.finfo(torch.float32).min).view(1, 1, 1, 16)
    masked = count_case(Attention(), (q, k, v, mask), {})
    assert masked.total_flops == full // 4
    kv = 4 * 16 * 32 * F32
    assert masked.read_bytes == kv + 2 * kv // 4 + 16 * F32  # q, valid k/v rows, mask


class StaticCacheStep(nn.Module):
    """VoxCPM-style decode step: write one slot of a static cache, attend with a mask."""

    def forward(self, q, kv, pos):
        k_cache, v_cache = kv
        k_cache[:, :, pos, :] = q[:, :, 0]
        v_cache[:, :, pos, :] = q[:, :, 0] * 2
        mask = (torch.arange(k_cache.size(2)) <= pos).view(1, 1, 1, -1)
        return F.scaled_dot_product_attention(q, k_cache, v_cache, attn_mask=mask)


def test_static_kv_cache_counts_only_the_touched_slice():
    torch.manual_seed(0)
    heads, slots, dim, pos = 4, 1024, 32, 99
    cache = torch.randn(2, 3, 1, heads, slots, dim)  # (k/v, layers, batch, heads, slots, dim)
    q = torch.randn(1, heads, 1, dim)
    args = (q, (cache[0, 1], cache[1, 1]), pos)
    before = cache.clone()
    cost = count_case(StaticCacheStep(), args, {})
    assert torch.equal(cache, before)  # counted on a deep copy
    row = heads * dim * F32
    valid = pos + 1
    assert cost.read_bytes == row + 2 * valid * row  # q + the valid k/v positions
    assert cost.write_bytes == 2 * row + row  # one k and one v slot + the output
    assert cost.total_flops == 2 * heads * 1 * valid * (dim + dim)
    full_cache = 2 * slots * row
    assert cost.min_bytes < full_cache / 4


class AppendCache:
    def __init__(self, k: torch.Tensor) -> None:
        self.k = k


class Append(nn.Module):
    """DynamicCache-style append: the cache attribute is replaced by a longer tensor."""

    def forward(self, x, cache):
        cache.k = torch.cat([cache.k, x], dim=2)
        return cache.k.sum(2)


class Accumulate(nn.Module):
    def forward(self, x, cache):
        cache.k.add_(x)
        return x * 2


def test_cache_objects_append_and_in_place_update():
    past = torch.randn(1, 2, 10, 8)
    x = torch.randn(1, 2, 1, 8)
    cost = count_case(Append(), (x, AppendCache(past)), {})
    assert cost.read_bytes == past.numel() * F32 + x.numel() * F32
    assert cost.write_bytes == (1 * 2 * 11 * 8 + 1 * 2 * 8) * F32  # new cache + output

    state = AppendCache(torch.zeros(64))
    x = torch.randn(64)
    cost = count_case(Accumulate(), (x,), {"cache": state})
    assert cost.read_bytes == 64 * F32
    assert cost.write_bytes == 64 * F32 + 64 * F32  # changed state elements + output
    assert torch.count_nonzero(state.k) == 0


def test_embedding_reads_only_gathered_rows():
    emb = nn.Embedding(1000, 16)
    ids = torch.tensor([[1, 5, 5, 7]])
    cost = count_case(emb, (ids,), {})
    assert cost.read_bytes == 4 * 16 * F32 + ids.numel() * 8
    assert cost.write_bytes == 4 * 16 * F32


def test_attention_density():
    assert attention_density(None, False, 8, 8) == (1.0, 1.0)
    pairs, kv = attention_density(None, True, 4, 4)
    assert pairs == pytest.approx(10 / 16) and kv == 1.0
    pairs, kv = attention_density(None, True, 2, 8)  # upper-left aligned like torch
    assert pairs == pytest.approx(3 / 16) and kv == pytest.approx(2 / 8)
    mask = torch.zeros(1, 1, 2, 8, dtype=torch.bool)
    mask[..., 0, :2] = True
    mask[..., 1, :3] = True
    pairs, kv = attention_density(mask, False, 2, 8)
    assert pairs == pytest.approx(5 / 16) and kv == pytest.approx(3 / 8)
    additive = torch.where(mask, 0.0, float("-inf")).expand(2, 16, 2, 8)  # broadcast, not copied
    assert attention_density(additive, False, 2, 8) == pytest.approx((5 / 16, 3 / 8))


# ------------------------------------------------------------ speed of light


def test_sol_time_classification():
    big = CaseCost(read_bytes=50_000_000, write_bytes=50_000_000)  # 100 MB, no FLOPs
    sol = sol_time(big, PEAKS)
    assert sol == {"sol_ms": pytest.approx(0.1), "bound": "memory", "l2_resident": False}

    gemm = CaseCost(flops={"bfloat16": 2 * 4096**3}, read_bytes=3 * 4096 * 4096 * 2)
    sol = sol_time(gemm, PEAKS)
    assert sol["bound"] == "compute" and sol["sol_ms"] == pytest.approx(2 * 4096**3 / 1e14 * 1e3)
    fp32 = CaseCost(flops={"float32": 2 * 4096**3})
    assert sol_time(fp32, PEAKS)["sol_ms"] == pytest.approx(4 * sol["sol_ms"])

    small = CaseCost(read_bytes=400_000, write_bytes=400_000)  # fits in L2 (1 MB here)
    hot = sol_time(small, PEAKS)
    assert hot["l2_resident"] and hot["sol_ms"] == pytest.approx(0.8e6 / 4000e9 * 1e3)
    assert hot["bound"] == "launch"  # 0.2 us of work vs a 10 us launch floor
    cold = sol_time(small, PEAKS, hot_l2=False)
    assert not cold["l2_resident"] and cold["sol_ms"] == pytest.approx(4 * hot["sol_ms"])


def _result(*cases):
    return {
        "status": "ok",
        "correct": True,
        "cases": [
            {"calls_per_run": n, "ref_ms": ref, "new_ms": new, "signature": f"s{i}"}
            for i, (n, ref, new) in enumerate(cases)
        ],
    }


def test_apply_sol_weights_and_flags():
    mem = CaseCost(read_bytes=50_000_000, write_bytes=50_000_000)  # sol 0.1 ms
    result = _result((1, 0.5, 0.2), (3, 0.5, 0.125))
    apply_sol(result, [mem, mem], PEAKS)
    first, second = result["cases"]
    assert first["pct_of_sol"] == 50.0 and second["pct_of_sol"] == 80.0
    assert first["bound"] == "memory" and first["min_bytes"] == 100_000_000
    assert result["pct_of_sol"] == pytest.approx(100 * 0.4 / (0.2 + 0.375), abs=0.1)
    assert result["sol_ms_weighted"] == pytest.approx(0.4)
    assert result["launch_floor_ms"] == 0.01 and result["bound"] == "memory"
    assert "suspicious_faster_than_sol" not in result and sol_signal(result) == 69.6

    hack = _result((1, 0.5, 0.05))  # twice as fast as the hardware allows
    apply_sol(hack, [mem], PEAKS)
    assert hack["suspicious_faster_than_sol"] and hack["cases"][0]["suspicious_faster_than_sol"]
    assert sol_signal(hack) is None

    wrong = _result((1, 0.05, 0.05))  # the reference beats it too: the estimate is off
    apply_sol(wrong, [mem], PEAKS)
    assert wrong["sol_unreliable"] and "suspicious_faster_than_sol" not in wrong
    assert sol_signal(wrong) is None
    assert sol_signal({"correct": False, "pct_of_sol": 99.0}) is None


def test_annotate_without_peaks_and_on_errors(monkeypatch):
    result = _result((1, 0.5, 0.2))
    monkeypatch.setattr(roofline, "current_peaks", lambda: None)
    annotate(result, nn.Identity(), [{"args": (torch.ones(1),), "kwargs": {}}])
    assert "not measured" in result["sol_note"] and "pct_of_sol" not in result

    broken = _result((1, 0.5, 0.2))
    annotate(broken, nn.Linear(2, 2), [{"args": (torch.ones(3),), "kwargs": {}}], peaks=PEAKS)
    assert broken["sol_error"].startswith("RuntimeError")  # never fails the evaluation

    ok = _result((1, 0.5, 0.2))
    lin = nn.Linear(64, 64)
    annotate(ok, lin, [{"args": (torch.ones(2, 64),), "kwargs": {}}], peaks=PEAKS)
    assert ok["cases"][0]["flops"] == 2 * 2 * 64 * 64 and ok["cases"][0]["bound"] == "launch"


def test_compact_keeps_sol_fields():
    result = _result((1, 0.5, 0.05))
    apply_sol(result, [CaseCost(read_bytes=100_000_000)], PEAKS)
    out = compact(result)
    assert out["pct_of_sol"] == result["pct_of_sol"] and out["suspicious_faster_than_sol"]
    case = out["cases"][0]
    assert {"sol_ms", "pct_of_sol", "bound", "min_bytes", "suspicious_faster_than_sol"} <= set(case)


# ------------------------------------------------------------ advice, ledger, toolchain


def test_feedback_stops_near_speed_of_light(tmp_path):
    run = RunDir.create(tmp_path, "org/m")
    results = run.target("t") / "results.jsonl"
    append_jsonl(results, {"correct": True, "speedup": 1.5})
    budget = Budget(run, kernel_evals=10)
    budget.start_agent("kernel-t")
    fb = budget.feedback("kernel-t", results, 10, pct_of_sol=SOL_STOP_PCT - 5)
    assert fb["advice"] == "continue"
    fb = budget.feedback("kernel-t", results, 10, pct_of_sol=95.2)
    assert fb["advice"] == "stop"
    assert fb["advice_reason"].startswith("at 95 % of this recipe's roofline (SOL)")
    fb = budget.feedback("kernel-t", results, 3, pct_of_sol=95.2)  # budget reasons come first
    assert fb["advice"] == "stop" and "evaluation budget" in fb["advice_reason"]
    note = budget.prompt_note("kernel-t", budget.agent_config(budget_cfg()), ["x__evaluate_e2e"])
    assert "90 % of its recipe's roofline" in note and "next ideas" in note


def budget_cfg():
    from kernel_agent.config import OptimizeConfig

    return OptimizeConfig(model_ref="org/m")


def test_ledger_records_pct_of_sol(tmp_path):
    run = RunDir.create(tmp_path, "org/m")
    result = {"status": "ok", "correct": True, "speedup": 1.4, "pct_of_sol": 72.5, "cases": []}
    row = ledger.record_kernel(run, "t", result, snapshot="001_a.py", hypothesis="h")
    assert row["pct_of_sol"] == 72.5 and ledger.rows(run)[0]["pct_of_sol"] == 72.5
    assert ledger.summary(run)["targets"][0]["best_pct_of_sol"] == 72.5

    # a ledger written before the column existed keeps its own layout
    old = RunDir.create(tmp_path / "old", "org/m")
    columns = [c for c in ledger.COLUMNS if c != "pct_of_sol"]
    old.ledger.write_text("\t".join(columns) + "\n")
    ledger.record_kernel(old, "t", result, snapshot="001_a.py", hypothesis="still aligned")
    (row,) = ledger.rows(old)
    assert row["hypothesis"] == "still aligned" and row["exp"] == 1 and "pct_of_sol" not in row


def test_toolchain_peaks_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(toolchain, "CACHE_DIR", tmp_path)
    path = toolchain.peaks_path("NVIDIA GeForce RTX 5070 Ti", "2.14.1+cu130")
    assert path == tmp_path / "peaks-nvidia-geforce-rtx-5070-ti-torch2.14.1-cu130.json"
    assert toolchain.load_peaks(path) is None
    path.write_text("not json")
    assert toolchain.load_peaks(path) is None
    path.write_text(
        '{"dram_gbps": 812.5, "l2_gbps": 2900, "tflops": {"bfloat16": 99.7}, '
        '"launch_floor_us": 9.5}'
    )
    peaks = toolchain.load_peaks(path)
    assert peaks is not None and peaks["dram_gbps"] == 812.5
    line = toolchain.format_peaks(peaks)
    assert line == "copy DRAM 812 GB/s, L2 2900 GB/s, matmul bf16 100 TFLOP/s, launch floor 9.5 us"
    tc = toolchain.Toolchain(
        gpu=None, torch_version="x", torch_cuda=None, cuda_home=None, nvcc_version=None, peaks=peaks
    )
    assert f"measured peaks: {line}" in tc.summary() and tc.to_dict()["peaks"] == peaks


def test_ensure_peaks_uses_the_cache_and_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(toolchain, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("KERNEL_AGENT_LOCK_HELD", "1")
    monkeypatch.setattr(roofline, "_MEASURE_FAILED", False)
    gpu = toolchain.GPUInfo("Fake GPU", (12, 0), 16.0, 70, 48.0)
    tc = toolchain.Toolchain(
        gpu=gpu, torch_version="2.0", torch_cuda=None, cuda_home=None, nvcc_version=None
    )
    monkeypatch.setattr(toolchain, "setup", lambda: tc)
    runs = []

    def failing_run(cmd, **kwargs):
        runs.append(cmd)
        raise OSError("no GPU here")

    monkeypatch.setattr(roofline.subprocess, "run", failing_run)
    assert roofline.ensure_peaks() is None and len(runs) == 1
    assert roofline.ensure_peaks() is None and len(runs) == 1  # not retried in this process

    toolchain.peaks_path("Fake GPU", "2.0").write_text('{"dram_gbps": 500.0}')  # e.g. doctor
    assert roofline.ensure_peaks() == {"dram_gbps": 500.0}  # read from the cache, no run
    assert tc.peaks == {"dram_gbps": 500.0} and len(runs) == 1

    tc.gpu = "not a GPUInfo"  # e.g. a test double: no name, no measurement
    assert roofline.ensure_peaks(remeasure=True) is None and len(runs) == 1


# ------------------------------------------------------------ GPU


@pytest.mark.gpu
def test_measured_peaks_and_rmsnorm_sol(tmp_path):
    from kernel_agent.agent.prompts import EXAMPLES_DIR
    from kernel_agent.kernels.evaluate import evaluate
    from kernel_agent.selftest import make_rmsnorm_capture

    peaks = roofline.ensure_peaks()
    assert peaks is not None, "peak measurement failed"
    assert 100 < peaks["dram_gbps"] < 10_000 and peaks["l2_gbps"] > peaks["dram_gbps"]
    assert peaks["tflops"]["bfloat16"] > peaks["tflops"]["float32"] > 1
    for low in (roofline.FP8, roofline.FP4):  # measured, or why not (never a ratio)
        assert low in peaks.get("tflops_unavailable", {}) or peaks["tflops"][low] > 1
    assert peaks["version"] == roofline.PEAKS_VERSION
    assert 0.5 < peaks["launch_floor_us"] < 500
    print("peaks:", toolchain.format_peaks(peaks))

    capture = make_rmsnorm_capture(tmp_path / "rms.pt")
    result = evaluate(capture, EXAMPLES_DIR / "triton_rmsnorm.py")
    assert result["status"] == "ok", result
    assert 0 < result["pct_of_sol"] < 100 / roofline.SUSPICIOUS_RATIO, result
    assert "suspicious_faster_than_sol" not in result and "sol_unreliable" not in result
    prefill, decode = sorted(result["cases"], key=lambda c: -c["min_bytes"])
    hidden = 2048
    assert prefill["min_bytes"] == 2 * 256 * hidden * 2 + hidden * 2  # x + out (bf16) + weight
    assert decode["bound"] == "launch" and decode["flops"] == 0
    for case in result["cases"]:
        print(
            f"{case['signature']}: new {case['new_ms'] * 1000:.1f} us, sol "
            f"{case['sol_ms'] * 1000:.2f} us, {case['pct_of_sol']} % of SOL, {case['bound']}"
        )
    print("weighted pct_of_sol:", result["pct_of_sol"], "bound:", result["bound"])

    # attention over a static cache on the GPU: SDPA's mask-aware accounting holds there too
    cache = torch.zeros(2, 2, 1, 4, 1024, 64, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(1, 4, 1, 64, device="cuda", dtype=torch.bfloat16)
    cost = count_case(StaticCacheGPU(), (q, (cache[0, 0], cache[1, 0]), 99), {})
    row = 4 * 64 * 2
    assert cost.read_bytes == row + 2 * 100 * row and cost.write_bytes == 3 * row


class StaticCacheGPU(StaticCacheStep):
    def forward(self, q, kv, pos):
        k_cache, v_cache = kv
        k_cache[:, :, pos, :] = q[:, :, 0]
        v_cache[:, :, pos, :] = q[:, :, 0] * 2
        mask = (torch.arange(k_cache.size(2), device=q.device) <= pos).view(1, 1, 1, -1)
        return F.scaled_dot_product_attention(q, k_cache, v_cache, attn_mask=mask)
