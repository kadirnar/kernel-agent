"""Search-based autotuning over declared spaces (issue #229, ``kernels/search.py``).

CPU, deterministic: spaces and constraints, the GPU's facts and its pruning (a fake sm_120
with its 101,376 B of shared memory per block, a fake sm_86 without TMA), pruning from
feedback, the strategies on seeded synthetic objectives (the optimum within the budget, the
same configs for the same seed), warm starts from the tuned-config cache. The sweep side
(batches in the subprocess, a simulated clock for its time bound, the tool's record and
budget, a restart after a crash, the target's tolerance tier) is in ``test_sweep.py``'s
neighbour, the second half of this file.
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from fake_clock import Clock

from kernel_agent import cli, gpulock, ledger, truth
from kernel_agent.agent import tools as tools_mod
from kernel_agent.budget import Budget
from kernel_agent.kernels import compare, tuned
from kernel_agent.kernels import evaluate as evaluate_mod
from kernel_agent.kernels import search as search_mod
from kernel_agent.kernels import sweep as sweep_mod
from kernel_agent.kernels.search import Arch, Constraints, Search, arch_reason, parse_space
from kernel_agent.profiling.capture import capture_calls
from kernel_agent.selftest import RMSNorm
from kernel_agent.workspace import RunDir, write_json

SM120 = Arch((12, 0), smem_per_block=101_376, sm_count=70)  # RTX 5070 Ti's facts
SM86 = Arch((8, 6), smem_per_block=101_376, sm_count=84)  # Ampere: no TMA
SM90 = Arch((9, 0), smem_per_block=232_448, sm_count=132)  # Hopper: TMA, wgmma

# ------------------------------------------------------------------ spaces


def test_spaces_expand_lists_ranges_powers_of_two_and_fixed_values():
    space = parse_space(
        {
            "BLOCK_M": {"pow2": [16, 256]},
            "num_stages": "2..5",
            "SPLIT": {"range": [0, 64, 16]},
            "BLOCK_N": "pow2:32..128",
            "num_warps": [8, 2, 4, 4],  # sorted, duplicates dropped
            "mode": ["b", "a"],  # categorical: the given order
            "fixed": True,
        }
    )
    values = {d.name: d.values for d in space.dims}
    assert values["BLOCK_M"] == (16, 32, 64, 128, 256)
    assert values["num_stages"] == (2, 3, 4, 5) and values["SPLIT"] == (0, 16, 32, 48, 64)
    assert values["BLOCK_N"] == (32, 64, 128) and values["num_warps"] == (2, 4, 8)
    assert values["mode"] == ("b", "a") and values["fixed"] == (True,)
    assert [d.ordinal for d in space.dims] == [True, True, True, True, True, False, False]
    assert space.size == 5 * 4 * 5 * 3 * 3 * 2
    config = space.config(space.decode(7))
    assert space.key(config) == space.decode(7) and set(config) == set(space.names)
    assert space.key({**config, "num_warps": 4.0}) is not None  # JSON's 4.0 is 4
    assert space.key({**config, "num_warps": 3}) is None and space.key({"x": 1}) is None
    assert parse_space('{"B": [1, 2]}').size == 2  # JSON text
    for bad in (
        {},
        [],
        {"not a name": [1]},
        {"sm": [1]},  # a GPU fact
        {"min": [1]},  # a constraint function
        {"B": []},
        {"B": {"pow2": [0, 8]}},
        {"B": {"pow2": [64, 32]}},
        {"B": {"range": [1]}},
        {"B": {"step": [1, 2]}},
        {"B": "1..1000000000"},
        {"B": "pow2:1..8:2"},
        {"B": [[1, 2]]},
        "{",
    ):
        with pytest.raises(ValueError):
            parse_space(bad)


def test_constraints_parse_safely_and_read_the_gpus_facts():
    params = ["BLOCK_M", "BLOCK_N", "BLOCK_K", "num_stages", "num_warps"]
    smem = "(BLOCK_M + BLOCK_N) * BLOCK_K * 2 * num_stages <= smem_per_block"
    found = Constraints.parse(f"{smem}; BLOCK_M * BLOCK_N >= 32 * num_warps", params)
    assert found.texts == [smem, "BLOCK_M * BLOCK_N >= 32 * num_warps"]
    config = {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "num_stages": 4, "num_warps": 8}
    why = found.violated(config, SM120.names())  # 196,608 B > 101,376 B
    assert why == f"{smem} (smem_per_block = 101376)"
    assert found.violated({**config, "num_stages": 1}, SM120.names()) is None  # 49,152 B
    assert found.violated(config, SM90.names()) is None  # 227 KB on Hopper
    assert found.violated(config, {}) is None  # no GPU: the facts' constraints wait
    assert found.violated({**config, "BLOCK_M": 1, "BLOCK_N": 1}, {}) is not None
    lists = Constraints.parse(
        ["num_warps in (4, 8)", "cdiv(BLOCK_M, 64) <= 2 if has_tma else 1"], params
    )
    assert lists.violated({"num_warps": 2, "BLOCK_M": 64}, SM90.names()) == "num_warps in (4, 8)"
    assert lists.violated({"num_warps": 4, "BLOCK_M": 192}, SM90.names()) is not None
    assert lists.violated({"num_warps": 4, "BLOCK_M": 64}, SM86.names()) is None
    # arithmetic on numbers only, no huge powers: a constraint that raises is violated
    weird = Constraints.parse(["BLOCK_M ** 1000 > 0", "'a' * BLOCK_M == 'aa'"], params)
    assert "out of range" in str(weird.violated({"BLOCK_M": 2}, {}))
    two = Constraints.parse("'a' * BLOCK_M == 'aa'", params)
    assert "TypeError" in str(two.violated({"BLOCK_M": 2}, {}))
    for bad in (
        "BLOCK_M.bit_length() > 2",  # attributes
        "__import__('os')",  # calls of anything but the functions
        "[x for x in (1, 2)]",
        "lambda: 1",
        "BLOCK_X > 1",  # neither a parameter nor a fact
        "min(BLOCK_M, key=1)",
        "BLOCK_M >",
        "(1, 2)[0] == 1",
        42,
    ):
        with pytest.raises(ValueError):
            Constraints.parse(bad, params)


def test_arch_pruning_before_compile_on_a_fake_sm120_and_sm86():
    assert arch_reason({"warp_specialize": True}, SM90) is None  # Hopper has it
    sm120 = arch_reason({"WARP_SPECIALIZE": 1}, SM120)
    assert sm120 is not None and "§1.2" in sm120 and "sm_120" in sm120
    assert "needs Hopper" in str(arch_reason({"warp_specialize": True}, SM86))
    assert arch_reason({"warp_specialize": False}, SM120) is None
    # Helion's per-loop list and its TMA indexing
    assert arch_reason({"range_warp_specializes": [None, True]}, SM120) is not None
    assert arch_reason({"range_warp_specializes": [None, False]}, SM120) is None
    assert "sm_86" in str(arch_reason({"indexing": "tensor_descriptor"}, SM86))
    assert arch_reason({"indexing": ["pointer", "tensor_descriptor"]}, SM86) is not None
    assert (
        arch_reason({"USE_TMA": 1}, SM86) is not None and arch_reason({"USE_TMA": 0}, SM86) is None
    )
    assert arch_reason({"USE_TMA": 1}, SM120) is None and arch_reason({"USE_TMA": 1}, SM90) is None
    assert arch_reason({"warp_specialize": True, "USE_TMA": True}, Arch()) is None  # no GPU

    space = parse_space(
        {
            "BLOCK_M": {"pow2": [32, 256]},
            "BLOCK_N": {"pow2": [32, 256]},
            "BLOCK_K": [32, 64, 128],
            "num_stages": "2..4",
            "USE_TMA": [0, 1],
            "warp_specialize": [False, True],
        }
    )
    smem = Constraints.parse(
        "(BLOCK_M + BLOCK_N) * BLOCK_K * 2 * num_stages <= smem_per_block", space.names
    )
    on_120 = Search(space, smem, arch=SM120)
    on_86 = Search(space, smem, arch=SM86)
    on_90 = Search(space, smem, arch=SM90)
    assert on_120.valid is not None and on_86.valid is not None and on_90.valid is not None
    assert on_90.valid > on_120.valid > on_86.valid  # more smem; no TMA on sm_86
    every = [space.config(space.decode(i)) for i in range(space.size)]
    for config in every:
        need = (config["BLOCK_M"] + config["BLOCK_N"]) * config["BLOCK_K"] * 2
        need *= config["num_stages"]
        fits = need <= 101_376
        assert (on_120.reason(config) is None) == (fits and not config["warp_specialize"])
        assert (on_86.reason(config) is None) == (
            fits and not config["warp_specialize"] and not config["USE_TMA"]
        )
    assert on_120.pruned["constraint"] and on_120.pruned["arch"] and not on_120.pruned["feedback"]
    assert "smem_per_block = 101376" in on_120.examples["constraint"]
    for config in on_86.ask(64):  # never asks for a pruned config
        assert on_86.reason(config) is None and not config["USE_TMA"]
    blind = Search(space, smem)  # no GPU: only what the constraints say without its facts
    assert blind.valid == space.size


def test_feedback_prunes_what_runs_out_of_resources_or_spills():
    space = parse_space(
        {"BLOCK": {"pow2": [16, 512]}, "num_warps": [1, 2, 4, 8], "mode": ["a", "b"]}
    )
    found = Search(space, strategy="grid", seed=0)
    error = (
        "triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: "
        "131072, Hardware limit: 101376. Reducing block sizes or `num_stages` may help."
    )
    assert search_mod.resource_error(error) and not search_mod.resource_error("TypeError: x")
    found.tell({"BLOCK": 128, "num_warps": 2, "mode": "a"}, None, error=error)
    larger = {"BLOCK": 256, "num_warps": 4, "mode": "a"}
    assert "ran out of resources (out of resource: shared memory" in str(found.reason(larger))
    assert found.reason({"BLOCK": 256, "num_warps": 1, "mode": "a"}) is None  # fewer warps
    assert found.reason({"BLOCK": 256, "num_warps": 4, "mode": "b"}) is None  # another mode
    found.tell({"BLOCK": 64, "num_warps": 4, "mode": "b"}, 1.2, spills=96)
    assert "spilled registers" in str(found.reason({"BLOCK": 128, "num_warps": 2, "mode": "b"}))
    assert found.reason({"BLOCK": 128, "num_warps": 8, "mode": "b"}) is None  # more warps
    asked = [c for batch in iter(lambda: found.ask(8), []) for c in batch]
    assert larger not in asked and {"BLOCK": 512, "num_warps": 8, "mode": "a"} not in asked
    assert found.pruned["feedback"] > 0 and "feedback" in found.summary()["pruned"]
    assert found.best() == ({"BLOCK": 64, "num_warps": 4, "mode": "b"}, 1.2)


# ------------------------------------------------------------------ strategies

SYNTH = {
    "BLOCK_M": {"pow2": [16, 256]},
    "BLOCK_N": {"pow2": [16, 256]},
    "BLOCK_K": {"pow2": [16, 256]},
    "num_warps": [1, 2, 4, 8, 16],
    "num_stages": "1..6",
    "GROUP_M": [1, 4, 8, 16],
}


def synthetic(config: dict[str, Any]) -> float:
    """A speedup with interactions and a second, lower optimum (15,000 configs)."""
    lm, ln = math.log2(config["BLOCK_M"]), math.log2(config["BLOCK_N"])
    lk, w = math.log2(config["BLOCK_K"]), math.log2(config["num_warps"])
    s, g = config["num_stages"], math.log2(config["GROUP_M"])
    main = -((lm - 6) ** 2) - (ln - 7) ** 2 - 0.5 * (lk - 6) ** 2 - (w - 2) ** 2
    main -= 0.3 * (s - 3) ** 2 + 0.2 * (g - 3) ** 2 + 0.3 * (lm - 6) * (w - 2)
    local = -((lm - 4) ** 2) - (ln - 4) ** 2 - (lk - 4) ** 2 - (w - 1) ** 2 - (s - 2) ** 2 - 2
    return math.exp(0.15 * max(main, local))


OPTIMUM = {
    "BLOCK_M": 64,
    "BLOCK_N": 128,
    "BLOCK_K": 64,
    "num_warps": 4,
    "num_stages": 3,
    "GROUP_M": 8,
}


def run(strategy: str, seed: int, budget: int, **kw: Any) -> tuple[Search, list[dict[str, Any]]]:
    found = Search(parse_space(SYNTH), strategy=strategy, seed=seed, **kw)
    asked: list[dict[str, Any]] = []
    while len(asked) < budget and (batch := found.ask(16)):
        for config in batch:
            found.tell(config, synthetic(config))
        asked += batch
    return found, asked


def test_pattern_and_tpe_find_the_optimum_within_the_budget_and_repeat_for_a_seed():
    space = parse_space(SYNTH)
    every = (space.config(space.decode(i)) for i in range(space.size))
    assert max(every, key=synthetic) == OPTIMUM and space.size == 15_000
    for seed in range(8):
        pattern, asked = run("pattern", seed, 160)  # about 1 % of the space
        assert pattern.best() == (OPTIMUM, 1.0), seed
        assert len({json.dumps(c, sort_keys=True) for c in asked}) == len(asked)  # never twice
        tpe, _ = run("tpe", seed, 320)
        assert tpe.best() == (OPTIMUM, 1.0), seed
    again = [run("pattern", 3, 160)[1], run("pattern", 3, 160)[1]]
    assert again[0] == again[1]  # the same seed: the same configs in the same order
    assert run("pattern", 4, 160)[1] != again[0]
    assert run("tpe", 3, 64)[1] == run("tpe", 3, 64)[1]
    auto, _ = run("auto", 0, 16)
    assert auto.strategy == "pattern" and auto.summary()["requested"] == "auto"


def test_grid_sweeps_every_valid_config_in_a_seeded_order():
    space = parse_space({"B": [1, 2, 3, 4], "W": [1, 2, 4]})
    rule = Constraints.parse("B * W <= 8", space.names)
    grid = Search(space, rule, strategy="auto", seed=1)
    assert grid.strategy == "grid" and grid.valid == 10

    def every(found: Search) -> list[dict[str, Any]]:
        return [c for batch in iter(lambda: found.ask(4), []) for c in batch]

    asked = every(grid)
    assert len(asked) == 10 and all(c["B"] * c["W"] <= 8 for c in asked)
    assert every(Search(space, rule, strategy="grid", seed=1)) == asked  # the same seed
    assert every(Search(space, rule, strategy="grid", seed=2)) != asked  # another order
    warm = Search(space, rule, strategy="grid", seed=1, warm=[{"B": 4, "W": 2}, {"B": 4, "W": 4}])
    assert warm.ask(2)[0] == {"B": 4, "W": 2} and warm.warm == [(3, 1)]  # B*W=16: not valid
    again = every(Search(space, rule, strategy="grid", seed=1, warm=[{"B": 4, "W": 2}]))
    assert len(again) == 10 and again[0] == {"B": 4, "W": 2}  # first, and never twice


def test_warm_starts_come_first_and_specs_are_validated():
    found = Search(parse_space(SYNTH), strategy="pattern", seed=0, warm=[OPTIMUM, {"x": 1}])
    assert found.ask(4)[0] == OPTIMUM and found.summary()["warm_starts"] == 1
    spec = search_mod.spec_from({"B": "pow2:16..64"}, "B >= 32", "TPE", "7")
    assert spec == {
        "space": {"B": "pow2:16..64"},
        "constraints": ["B >= 32"],
        "strategy": "tpe",
        "seed": 7,
    }
    assert search_mod.spec_from('{"B": [1]}')["space"] == {"B": [1]}
    assert search_mod.spec_from(None, None, "helion", 3) == {"strategy": "helion", "seed": 3}
    for args in (
        ({"B": [1, 2]}, "B > 5"),  # nothing valid
        ({"B": [1, 2]}, None, "annealing"),
        ({"B": [1, 2]}, None, None, "x"),
        (None,),
        ({"B": [1]}, None, "helion"),
        ({"B": [1, 2]}, "C > 1"),
    ):
        with pytest.raises(ValueError):
            search_mod.spec_from(*args)


def test_tuned_points_are_stored_and_found_at_neighbouring_buckets(tmp_path):
    store = tuned.TunedConfigs(tmp_path / "t.sqlite", gpu="GPU sm_120", versions={"torch": "2"})
    bucket = tuned.signature_bucket(["a0[1, 352, 1024]:bfloat16", "a0[1, 1, 1024]:bfloat16"])
    assert bucket == "a0[1, 1, 1024]:bfloat16;a0[1, 512, 1024]:bfloat16"
    assert tuned.bucket_distance(bucket, bucket) == 0.0
    near = "a0[1, 1, 1024]:bfloat16;a0[1, 256, 1024]:bfloat16"
    assert tuned.bucket_distance(bucket, near) == 1.0
    assert tuned.bucket_distance(bucket, near.replace("bfloat16", "float16")) is None
    assert tuned.bucket_distance(bucket, "a0[1, 1, 1024]:bfloat16") is None
    points = [
        {"config": {"B": 64}, "score": 1.5, "status": "ok"},
        {"config": {"B": 128}, "score": 2.0, "status": "ok"},
        {"config": {"B": 512}, "score": None, "status": "build_error"},
    ]
    assert store.add_points("sweep:RMSNorm:B", bucket, points) == 3
    store.add_points("sweep:RMSNorm:B", near, [{"config": {"B": 32}, "score": 9.0}])
    store.add_points(
        "sweep:RMSNorm:B",
        "a0[1, 1, 1024]:bfloat16;a0[1, 4096, 1024]:bfloat16",
        [{"config": {"B": 16}, "score": 9.0}],
    )
    found = store.points("sweep:RMSNorm:B", bucket)
    assert [p["config"] for p in found] == [{"B": 128}, {"B": 64}, {"B": 32}, {"B": 512}]
    assert found[2]["distance"] == 1.0 and found[-1]["score"] is None
    store.add_points("sweep:RMSNorm:B", bucket, [{"config": {"B": 64}, "score": 3.0}])
    assert store.points("sweep:RMSNorm:B", bucket)[0]["config"] == {"B": 64}  # replaced
    other = tuned.TunedConfigs(tmp_path / "t.sqlite", gpu="GPU sm_120", versions={"torch": "3"})
    assert other.points("sweep:RMSNorm:B", bucket) == []  # tuned under other versions
    assert (
        tuned.TunedConfigs(tmp_path / "t.sqlite", gpu="x", versions={"torch": "2"}).points(
            "sweep:RMSNorm:B", bucket
        )
        == []
    )
    hybrid = tuned.library_versions("cuda+triton")  # a hybrid: each backend's library
    assert "triton" in hybrid and "nvidia-cuda-nvcc" in hybrid and "helion" in tuned.LIBRARIES


# ------------------------------------------------------------------ the sweep side

# A config's cost is |A - 3| + |B - 5| + 1 passes (in simulated time: the cpu timer counts
# torch calls), so the optimum is A=3, B=5; `crash` kills the process.
TUNABLE = """
import os

import torch
from torch import nn


class Fast(nn.Module):
    def __init__(self, reference, passes):
        super().__init__()
        self.weight = reference.weight
        self.eps = reference.variance_epsilon
        self.passes = passes

    def forward(self, x):
        for _ in range(self.passes):
            h = x.float()
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
            y = self.weight * h.to(x.dtype)
        return y


def build(reference, A=1, B=1, USE_TMA=0, crash=False):
    if crash:
        os._exit(17)
    return Fast(reference, abs(A - 3) + abs(B - 5) + 1)
"""


def capture_of(path: Path, tier: str | None = None) -> Path:
    torch.manual_seed(0)
    module = RMSNorm(64, eps=1e-5)
    with torch.no_grad():
        module.weight.copy_(torch.randn(64) * 0.1 + 1)
    calls = [((torch.randn(1, 16, 64),), {}, 1), ((torch.randn(1, 1, 64),), {}, 7)]
    capture_calls(module.eval(), calls, path, tier=tier)
    return path


def cpu_timer(fn, args, kwargs, *, l2_flush=False, target_ms=60.0, keep=False):
    """kernels.bench.time_call's stand-in: 1 us per torch call (test_sweep.py's)."""
    from torch.overrides import TorchFunctionMode

    class Count(TorchFunctionMode):
        n = 0

        def __torch_function__(self, func, types, args=(), kwargs=None):
            Count.n += 1
            return func(*args, **(kwargs or {}))

    Count.n = 0
    with torch.inference_mode(), Count():
        fn(*args, **kwargs)
    return {"median_ms": Count.n * 1e-3}


SPACE = {"A": "1..8", "B": "1..8", "USE_TMA": [0, 1]}


def test_a_searched_sweep_finds_the_optimum_in_simulated_time_and_repeats(tmp_path, monkeypatch):
    capture = capture_of(tmp_path / "c.pt")
    candidate = tmp_path / "tunable.py"
    candidate.write_text(TUNABLE)
    spec = search_mod.spec_from(SPACE, "A + B <= 14", "pattern", 5)
    spec["arch"] = SM86.to_dict()  # USE_TMA=1 is pruned before anything compiles

    def searched() -> dict[str, Any]:
        monkeypatch.setattr(sweep_mod, "time", Clock(tick=0.01))  # deadline in simulated time
        deadline = sweep_mod.time.monotonic() + 2.0  # three batches of 16
        return sweep_mod.sweep(
            capture, candidate, [], device="cpu", timer=cpu_timer, deadline=deadline,
            search={**spec, "warm": False},
        )  # fmt: skip

    out = searched()
    table = out["table"]
    best = table[0]
    assert best["config"] == {"A": 3, "B": 5, "USE_TMA": 0} and best["final"], table[:3]
    assert best["speedup"] == max(r.get("speedup") or 0 for r in table)
    assert sum(bool(r.get("final")) for r in table) == sweep_mod.FINALISTS
    assert all(r["config"]["USE_TMA"] == 0 for r in table)
    found = out["search"]
    assert found["strategy"] == "pattern" and found["arch"] == "sm_86"
    # 128 configs: USE_TMA=1 is pruned for sm_86 (61, and 3 more fail A + B <= 14 first)
    assert found["pruned"] == {"constraint": 6, "arch": 61} and found["valid"] == 61
    assert 16 < found["measured"] == len(table) < 61  # a part of the space, then time is up
    assert [r["config"] for r in searched()["table"]] == [r["config"] for r in table]

    # every point went to the tuned cache; the next search starts from the best of them
    found_points = tuned.default().points(
        "sweep:RMSNorm:A,B,USE_TMA",
        tuned.signature_bucket(["a0[1, 16, 64]:float32", "a0[1, 1, 64]:float32"]),
        backend="torch",
    )
    assert found_points[0]["config"] == {"A": 3, "B": 5, "USE_TMA": 0}
    monkeypatch.setattr(sweep_mod, "time", Clock(tick=0.01))
    warm = sweep_mod.sweep(
        capture, candidate, [], device="cpu", timer=cpu_timer,
        deadline=sweep_mod.time.monotonic() + 3.0, search=spec,
    )  # fmt: skip
    assert warm["search"]["warm_starts"] > 0
    assert warm["table"][0]["config"] == {"A": 3, "B": 5, "USE_TMA": 0}
    assert (
        min(r["index"] for r in warm["table"] if r["config"]["A"] == 3 and r["config"]["B"] == 5)
        == 0
    )


def test_sweep_times_configs_in_the_targets_tolerance_tier(tmp_path, monkeypatch):
    """The sweep's timed-output check used the exact tier on near-lossless targets (every
    config of an FP8 target failed): evaluate() resets the comparator's tier on return."""
    capture = capture_of(tmp_path / "near.pt", tier="near-lossless")
    candidate = tmp_path / "tunable.py"
    candidate.write_text(TUNABLE)
    seen: list[str] = []
    original = sweep_mod._time_configs

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(compare.TIER)
        return original(*args, **kwargs)

    monkeypatch.setattr(sweep_mod, "_time_configs", spy)
    sweep_mod.sweep(capture, candidate, [{"A": 3}, {"A": 2}], device="cpu", timer=cpu_timer)
    spec = search_mod.spec_from({"A": [2, 3]}, None, "grid") | {"warm": False}
    sweep_mod.sweep(capture, candidate, [], device="cpu", timer=cpu_timer, search=spec)
    assert seen and set(seen) == {"near-lossless"}
    assert compare.TIER == compare.EXACT_TIER  # restored after


@pytest.fixture
def private_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(gpulock, "CACHE_DIR", tmp_path / "locks")
    monkeypatch.delenv(gpulock.ENV, raising=False)
    monkeypatch.setattr(sweep_mod, "ensure_peaks", lambda: None)


def test_the_tool_records_a_search_as_one_evaluation(tmp_path, monkeypatch, private_lock):
    run = RunDir.create(tmp_path, "org/m")
    write_json(run.run_json, {"card": {"repo_id": "org/m"}, "truth": truth.new_section()})
    keeper = truth.of(run)
    capture = capture_of(run.capture_file("t"))
    keeper.seal(capture)
    write_json(run.target("t") / "spec.json", {"id": "t", "module_class": "RMSNorm"})
    (run.target("t") / "candidates").mkdir(parents=True)
    (run.target("t") / "candidates" / "tunable.py").write_text(TUNABLE)
    spawned: list[dict[str, Any]] = []

    def spawn(capture_path, swept, configs, indices, *, capture_sha256, search=None, **_):
        spawned.append(search or {})
        result = sweep_mod.sweep(
            capture_path, swept, configs, indices=indices, device="cpu",
            capture_sha256=capture_sha256, timer=cpu_timer, search=search,
        )  # fmt: skip
        return {"rows": {}, "result": result, "culprit": None, "timed_out": False, "error": ""}

    def run_evaluation(capture_path, path, *, capture_sha256, timeout, **_):
        result = evaluate_mod.evaluate(
            capture_path, path, device="cpu", capture_sha256=capture_sha256
        )
        return result | {"speedup": 3.0} if result["correct"] else result

    monkeypatch.setattr(sweep_mod, "_spawn", spawn)
    monkeypatch.setattr(sweep_mod, "run_evaluation", run_evaluation)
    monkeypatch.setattr(tools_mod, "create_sdk_mcp_server", lambda n, version, tools: tools)
    monkeypatch.setattr(tools_mod, "refresh", lambda *a: None)
    session = tools_mod.SessionBinding(evaluations=5)
    server = {
        t.name: t
        for t in tools_mod.build_server(run, Budget(run, eval_timeout_s=60.0), keeper, session)
    }

    def call(**args: Any) -> dict[str, Any]:
        out = asyncio.run(
            server["sweep_candidate"].handler(
                {
                    "target_id": "t",
                    "candidate": "candidates/tunable.py",
                    "hypothesis": "passes",
                    **args,
                }
            )
        )
        return json.loads(out["content"][0]["text"])

    out = call(space={"A": "1..6", "B": [4, 5, 6]}, constraints="A != 1", strategy="grid", seed=2)
    assert out["status"] == "ok" and out["config"] == {"A": 3, "B": 5} and out["speedup"] == 3.0
    assert spawned[0]["strategy"] == "grid" and spawned[0]["constraints"] == ["A != 1"]
    found = out["sweep"]["search"]
    assert found["valid"] == 15 and found["measured"] == 15 and out["sweep"]["configs"] == 15
    assert out["sweep"]["table"][0]["final"] and len(out["sweep"]["table"]) == 15
    (row,) = ledger.rows(run)
    assert (
        row["hypothesis"]
        == "passes [sweep: A=3, B=5; best of 15/15 configs; grid search, 15 valid]"
    )
    (record,) = keeper.records(run.results_file("t"))
    assert record["config"] == {"A": 3, "B": 5} and record["sweep"]["search"]["seed"] == 2
    snap = run.history_dir("t") / Path(record["snapshot"]).name
    assert "_KA_SWEEP_CONFIG = {'A': 3, 'B': 5}" in snap.read_text()
    assert out["budget"]["evals_used"] == 1  # one sweep, one evaluation, 15 configs

    for bad, why in (
        ({}, "space (the values of every parameter) or configs is required"),
        ({"space": {"A": [1]}, "configs": [{"A": 1}]}, "not both"),
        ({"space": {"A": [1, 2]}, "constraints": "A > 2"}, "no config of the space"),
        ({"space": {"A": [1]}, "strategy": "simplex"}, "strategy is one of"),
    ):
        refused = call(**bad)
        assert refused["status"] == "error" and why in refused["error"], refused
    assert len(keeper.records(run.results_file("t"))) == 1


def test_a_search_restarts_after_a_crash_from_what_it_measured(
    tmp_path, monkeypatch, capsys, private_lock
):
    """The real subprocesses on the CPU (children see no GPU), through the CLI."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(evaluate_mod, "ensure_peaks", lambda: None)
    capture = capture_of(tmp_path / "c.pt")
    candidate = tmp_path / "tunable.py"
    candidate.write_text(TUNABLE)
    spec = tmp_path / "search.json"
    space = {"A": [1, 2, 3], "crash": [False, True]}
    spec.write_text(
        json.dumps({"space": space, "constraints": "A == 1 or not crash", "strategy": "grid"})
    )

    code = cli.main(["eval", str(capture), str(candidate), "--sweep", str(spec), "--timeout", "60"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert code == 0 and data["evaluation"]["correct"], data
    info = data["sweep"]
    assert info["runs"] == 2 and info["status"] == "ok" and info["configs"] == 4
    crashed = [r for r in info["table"] if r["status"] == "crash"]
    assert [r["config"] for r in crashed] == [{"A": 1, "crash": True}]
    assert "exit code 17" in crashed[0]["error"]
    assert sorted(r["index"] for r in info["table"]) == [0, 1, 2, 3]  # one number each
    found = info["search"]  # the last process's: everything told, nothing timed on the CPU
    assert found["measured"] == 0 and found["failed"] == 1 and found["untimed"] == 3
    assert "search: grid (seed 0)" in captured.err


# ------------------------------------------------------------------ GPU


@pytest.mark.gpu
def test_a_triton_search_on_this_gpu_binds_its_best_config(tmp_path, monkeypatch):
    from test_sweep import TRITON_LOOPED

    from kernel_agent.selftest import make_rmsnorm_capture

    monkeypatch.setenv(gpulock.ENV, "1")  # conftest holds this process's GPU lock
    run = RunDir.create(tmp_path, "org/m")
    write_json(run.run_json, {"card": {"repo_id": "org/m"}, "truth": truth.new_section()})
    keeper = truth.of(run)
    keeper.seal(make_rmsnorm_capture(run.capture_file("rms"), hidden=2048))
    write_json(run.target("rms") / "spec.json", {"id": "rms", "module_class": "RMSNorm"})
    (run.target("rms") / "candidates").mkdir(parents=True)
    (run.target("rms") / "candidates" / "looped.py").write_text(TRITON_LOOPED)
    monkeypatch.setattr(tools_mod, "create_sdk_mcp_server", lambda n, version, tools: tools)
    budget = Budget(run, eval_timeout_s=30.0)
    server = {t.name: t for t in tools_mod.build_server(run, budget, keeper)}
    args = {
        "target_id": "rms",
        "candidate": "candidates/looped.py",
        "space": {"BLOCK": {"pow2": [32, 4096]}, "num_warps": [1, 2, 4, 8, 16, 32]},
        "constraints": "BLOCK >= 32 * num_warps",
        "strategy": "pattern",
        "hypothesis": "tile size and warps",
    }
    out = asyncio.run(server["sweep_candidate"].handler(args))
    out = json.loads(out["content"][0]["text"])
    found = out["sweep"]["search"]
    major, minor = torch.cuda.get_device_capability()
    assert found["arch"] == f"sm_{major}{minor}" and found["strategy"] == "pattern"
    assert found["measured"] >= 8 and found["valid"] < found["space"]
    table = out["sweep"]["table"]
    assert table[0]["final"] and table[0]["config"] == out["config"], table[:3]
    assert out["status"] == "ok" and out["correct"] and out["speedup"] > 0
    (record,) = keeper.records(run.results_file("rms"))
    snap = run.history_dir("rms") / Path(record["snapshot"]).name
    assert f"_KA_SWEEP_CONFIG = {out['config']!r}" in snap.read_text()
    assert out["budget"]["evals_used"] == 1
