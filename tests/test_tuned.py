"""The persistent tuned-config cache (issue #148, ``kernels/tuned.py``) and the cached Triton
launcher (``kernels/triton_launch.py``): keys, buckets, persistence across processes,
invalidation on a library or driver upgrade, tuning, and the launcher's specialisation key
and direct launch path (with a stand-in for the compiled kernel; the GPU tests launch real
ones)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch

from kernel_agent.kernels import triton_launch, tuned

GPU = "Test GPU sm_120"
V1 = {"torch": "2.14.1", "cuda": "13.0", "driver": "615.71.09", "triton": "3.8.0"}
V2 = {**V1, "triton": "3.9.0"}


def _store(path: Path, versions: dict[str, Any] = V1, gpu: str = GPU) -> tuned.TunedConfigs:
    return tuned.TunedConfigs(path, gpu=gpu, versions=versions)


def test_shape_buckets_round_up_to_powers_of_two_except_exact_dims():
    assert [tuned.bucket(v) for v in (0, 1, 2, 3, 17, 352, 1024)] == [0, 1, 2, 4, 32, 512, 1024]
    shape = {"M": 352, "N": 2560, "K": 1024, "dtype": "bfloat16", "causal": True}
    assert tuned.shape_bucket(shape, exact=["N"]) == (
        "M=512,N=2560,K=1024,dtype=bfloat16,causal=True"
    )
    # the M of a batch of 11 and of 16 share a bucket, 17 does not
    assert tuned.shape_bucket({"M": 11}) == tuned.shape_bucket({"M": 16}) != "M=32"
    assert tuned.shape_bucket({"M": 17}) == "M=32"


def test_configs_persist_across_instances_and_come_back_as_json(tmp_path):
    db = tmp_path / "t.sqlite"
    store = _store(db)
    assert store.get("gemm", {"M": 352, "N": 1024}, exact=["N"]) is None
    kept = store.put("gemm", {"M": 352, "N": 1024}, (64, 64, 128, 4, 3), exact=["N"], ms=0.01)
    assert kept == [64, 64, 128, 4, 3]  # a tuple comes back as a list
    other = _store(db)  # another process: same GPU and versions
    assert other.get("gemm", {"M": 300, "N": 1024}, exact=["N"]) == [64, 64, 128, 4, 3]
    assert other.get("gemm", "M=512,N=1024") == [64, 64, 128, 4, 3]  # a bucket string
    assert other.get("gemm", {"M": 352, "N": 2048}, exact=["N"]) is None  # another N
    assert other.get("gemm", {"M": 352, "N": 1024}, exact=["N"], backend="cublaslt") is None
    assert _store(db, gpu="Other GPU sm_89").get("gemm", "M=512,N=1024") is None
    (entry,) = other.entries()
    assert entry["op"] == "gemm" and entry["ms"] == 0.01 and not entry["stale"]


def test_a_version_change_invalidates_the_entry(tmp_path):
    db = tmp_path / "t.sqlite"
    _store(db).put("attn", {"S": 16}, {"num_warps": 4})
    upgraded = _store(db, V2)
    assert [e["stale"] for e in upgraded.entries()] == [True]
    assert upgraded.get("attn", {"S": 16}) is None  # tuned with Triton 3.8: dropped
    assert upgraded.invalidated == [
        {"op": "attn", "bucket": "S=16", "backend": "triton", "versions": V1}
    ]
    assert _store(db).get("attn", {"S": 16}) is None  # deleted, not just hidden
    _store(db).put("attn", {"S": 16}, {"num_warps": 4})
    _store(db).put("attn", {"S": 32}, {"num_warps": 8})
    assert _store(db, {**V1, "driver": "620.0"}).purge_stale() == 2
    assert _store(db).entries() == []


def test_best_config_times_the_candidates_once_and_reuses_the_winner(tmp_path):
    db = tmp_path / "t.sqlite"
    timed: list[Any] = []

    def bench(config: dict[str, int]) -> float:
        timed.append(config["num_warps"])
        if config["num_warps"] == 16:
            raise RuntimeError("out of resources")  # skipped, like an over-limit config
        return {2: 3.0, 4: 1.0, 8: 2.0}[config["num_warps"]]

    candidates = [{"num_warps": w} for w in (2, 4, 8, 16)]
    store = _store(db)
    best = store.best_config("attn", {"programs": 512, "S": 16}, candidates, bench)
    assert best == {"num_warps": 4} and timed == [2, 4, 8, 16]
    assert store.best_config("attn", {"programs": 300, "S": 16}, candidates, bench) == best
    again = _store(db).best_config("attn", {"programs": 512, "S": 16}, candidates, bench)
    assert again == best and timed == [2, 4, 8, 16]  # another process: no timing
    (entry,) = store.entries()
    assert entry["tried"] == 4 and entry["ms"] == 1.0

    def failing(config: Any) -> float:
        raise RuntimeError("no config runs")

    fallback = store.best_config("x", {"M": 1}, candidates, failing, default={"num_warps": 8})
    assert fallback == {"num_warps": 8} and store.get("x", {"M": 1}) is None


def test_no_tuning_while_capturing_or_when_disabled(tmp_path, monkeypatch):
    calls: list[Any] = []

    def bench(config: Any) -> float:
        calls.append(config)
        return 1.0

    store = _store(tmp_path / "t.sqlite")
    candidates = [{"w": 1}, {"w": 2}]
    monkeypatch.setattr(tuned, "_no_tuning", lambda: "capturing")
    assert store.best_config("op", {"M": 4}, candidates, bench) == {"w": 1}
    assert store.best_config("op", {"M": 4}, candidates, bench, default={"w": 2}) == {"w": 2}
    assert calls == [] and store.get("op", {"M": 4}) is None
    monkeypatch.undo()

    monkeypatch.setenv(tuned.TUNE_ENV, "0")
    assert tuned._no_tuning() == "disabled"
    assert store.best_config("op", {"M": 4}, candidates, bench) == {"w": 1}
    assert calls == [] and store.get("op", {"M": 4}) is None
    monkeypatch.delenv(tuned.TUNE_ENV)
    assert tuned._no_tuning() is None  # CPU: no capture
    fresh = _store(tmp_path / "t.sqlite")
    assert fresh.best_config("op", {"M": 4}, candidates, bench) == {"w": 1}
    assert len(calls) == 2


def test_forget_and_the_default_store(tmp_path, monkeypatch):
    db = tmp_path / "default.sqlite"
    monkeypatch.setenv(tuned.DB_ENV, str(db))
    assert tuned.default_path() == db
    monkeypatch.setattr(tuned, "_DEFAULT", {})
    store = tuned.default()
    assert store is tuned.default() and store.path == db
    tuned.store("op", {"M": 8}, {"a": 1}, backend="cutlass")
    assert tuned.lookup("op", {"M": 8}, backend="cutlass") == {"a": 1}
    assert tuned.forget("op", backend="cutlass") == 1
    assert tuned.lookup("op", {"M": 8}, backend="cutlass") is None
    assert tuned.main(["--db", str(db)]) == 0


def test_library_versions_name_the_backend_library_torch_cuda_and_driver():
    versions = tuned.library_versions("triton")
    assert {"torch", "cuda", "driver", "triton"} <= set(versions)
    assert versions["torch"] == torch.__version__
    assert "nvidia-cublas" in tuned.library_versions("cublaslt")
    assert set(tuned.library_versions("some-backend")) == {"torch", "cuda", "driver"}


def test_tests_do_not_touch_the_users_cache():
    assert "KERNEL_AGENT_TUNED_DB" in os.environ  # conftest: a temporary database


# ------------------------------------------------------------------ cached launches


def test_spec_key_is_at_least_as_fine_as_the_jit_specialisation():
    base = torch.zeros(64, dtype=torch.float16)
    aligned, offset = base[:32], base[1:33]  # 2-byte offset: not 16-byte aligned
    key = triton_launch.spec_key
    assert key([aligned]) != key([offset])
    assert key([aligned]) != key([aligned.float()])
    assert key([1]) != key([2]) and key([16]) != key([17]) and key([32]) == key([48])
    assert key([3]) != key([1 << 40])
    assert key([3]) == key([5]) and key([0.5]) == key([2.0]) and key([None]) == (None,)
    assert key([True]) != key([1])
    assert key([64, 3], frozenset({0})) != key([128, 3], frozenset({0}))  # constexpr by value
    with pytest.raises(TypeError):
        key([(1, 2)])


triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")


@triton.jit
def _scale_kernel(x, y, n, factor, BLOCK: tl.constexpr, EVEN: tl.constexpr = False):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(y + i, tl.load(x + i, mask=i < n) * factor, mask=i < n)


class _Compiled:
    """Stands in for a CompiledKernel: records direct launches."""

    function, packed_metadata = "fn", "meta"

    def __init__(self) -> None:
        self.launches: list[tuple[Any, ...]] = []

    def run(self, *args: Any) -> None:
        self.launches.append(args)

    def launch_metadata(self, grid: Any, stream: Any, *args: Any) -> str:
        return "metadata"


class _Jit:
    """Stands in for the JIT launcher (``kernel[grid](...)``)."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.compiled = _Compiled()

    def __getitem__(self, grid: Any) -> Any:
        def launch(*args: Any, **kwargs: Any) -> _Compiled:
            self.calls.append((grid, args, kwargs))
            return self.compiled

        return launch


def test_cached_launch_goes_through_the_jit_once_per_specialisation(monkeypatch):
    launcher = triton_launch.CachedLaunch(_scale_kernel)
    assert launcher.names == ["x", "y", "n", "factor", "BLOCK", "EVEN"]
    assert launcher.defaults == {"EVEN": False} and launcher.constexprs == {4, 5}
    jit = _Jit()
    launcher.kernel = jit
    monkeypatch.setattr(triton_launch, "_device", lambda: 0)
    monkeypatch.setattr(triton_launch, "_stream", lambda device: 1234)
    x, y = torch.zeros(256), torch.zeros(256)

    launcher[(4,)](x, y, 256, 2.0, BLOCK=64, num_warps=4)
    assert len(jit.calls) == 1 and launcher.misses == 1 and not jit.compiled.launches
    launcher[(4,)](x, y, 256, 3.0, BLOCK=64, num_warps=4)  # same specialisation: direct
    assert len(jit.calls) == 1 and launcher.hits == 1
    (args,) = jit.compiled.launches
    assert args[:9] == (4, 1, 1, 1234, "fn", "meta", None, None, None)
    assert args[9:] == (x, y, 256, 3.0, 64, False)  # every parameter, defaults included

    launcher[lambda meta: (meta["n"] // meta["BLOCK"],)](x, y, 256, 1.0, BLOCK=64, num_warps=4)
    assert jit.compiled.launches[-1][:3] == (4, 1, 1)  # a grid function sees the parameters
    launcher[(4,)](x, y, 256, 2.0, BLOCK=128, num_warps=4)  # another constexpr
    launcher[(4,)](x[1:], y, 255, 2.0, BLOCK=64, num_warps=4)  # misaligned, n % 16 != 0
    launcher[(4,)](x, y, 256, 2.0, BLOCK=64, num_warps=8)  # another launch option
    assert len(jit.calls) == 4 and launcher.misses == 4
    launcher[(4,)](x, y, (256,), 2.0, BLOCK=64)  # an argument type without a cached path
    assert len(jit.calls) == 5

    monkeypatch.setattr(triton_launch, "_hooks", lambda: ("enter", "exit"))  # a profiler
    launcher[(4,)](x, y, 256, 2.0, BLOCK=64, num_warps=4)
    assert jit.compiled.launches[-1][6:9] == ("metadata", "enter", "exit")


def test_empty_triton_hook_chains_are_passed_as_none():
    from triton import knobs

    def hook(metadata: Any) -> None:
        pass

    assert triton_launch._hooks() == (None, None)
    knobs.runtime.launch_enter_hook.add(hook)
    try:
        assert triton_launch._hooks()[0] is knobs.runtime.launch_enter_hook
    finally:
        knobs.runtime.launch_enter_hook.remove(hook)


def test_cached_launch_needs_a_plain_jit_function():
    with pytest.raises(TypeError, match=r"triton\.jit"):
        triton_launch.CachedLaunch(lambda: None)
    assert isinstance(triton_launch.cached(_scale_kernel), triton_launch.CachedLaunch)
