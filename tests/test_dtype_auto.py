"""The workload dtype follows the GPU (#255): ``--dtype auto`` (``workloads/dtypes.py``),
analyze's float16 check against float32, ``Workload.set_dtype`` and the memory preflight
(``memfit.py``). CPU, with faked GPUs (sm_75 T4, sm_86 A10 and RTX 3070, sm_89 L4, sm_120
RTX 5070 Ti); the float16 check of a real LLM on the GPU (marked ``gpu``)."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from chaotic_toy import ChaoticToy
from test_integrate import COMBINATIONS, fake_worker, make, voxcpm2
from test_integrate import fast as fast  # the integration's simulated toolchain (fixture)
from torch import nn

from kernel_agent import cli, memfit, orchestrator, toolchain, worker
from kernel_agent.config import OptimizeConfig
from kernel_agent.hub import Modality, ModelCard
from kernel_agent.workloads import dtypes
from kernel_agent.workloads.base import Comparison, Workload, WorkloadSpec, cosine, runtime_dtype
from kernel_agent.workspace import RunDir, read_json, write_json

TOY = Path(__file__).with_name("toy_decoder.py")
#: Faked GPUs (name, capability, memory, SMs, L2): the memory torch reports for each board.
GPUS = {
    "T4": toolchain.GPUInfo("Tesla T4", (7, 5), 14.6, 40, 4.0),
    "A10": toolchain.GPUInfo("NVIDIA A10", (8, 6), 22.1, 72, 6.0),
    "RTX 3070": toolchain.GPUInfo("NVIDIA GeForce RTX 3070", (8, 6), 7.7, 46, 4.0),
    "L4": toolchain.GPUInfo("NVIDIA L4", (8, 9), 22.0, 58, 48.0),
    "RTX 5070 Ti": toolchain.GPUInfo("NVIDIA GeForce RTX 5070 Ti", (12, 0), 15.5, 70, 48.0),
}
#: VoxCPM2's card as recorded (``runs/openbmb--VoxCPM2/*/run.json``).
VOXCPM2 = {"repo_id": "openbmb/VoxCPM2", "size_gb": 4.266, "config": {"dtype": "bfloat16"}}
THROUGHPUT = {"batch_size": 16, "metric": "throughput"}


# ------------------------------------------------------------------ the choice


def test_runtime_dtype_follows_the_gpu():
    t4 = runtime_dtype("bfloat16", GPUS["T4"].capability)
    assert (t4.runtime, t4.check, t4.candidate) == ("float16", dtypes.PENDING, "float16")
    assert "sm_75 has no bf16 tensor cores" in t4.why and t4.arch == "sm_75"
    for gpu in ("A10", "RTX 3070", "L4", "RTX 5070 Ti"):
        kept = runtime_dtype("bfloat16", GPUS[gpu].capability)
        assert (kept.runtime, kept.check, kept.why) == ("bfloat16", None, "the checkpoint's dtype")
    for cap in [(7, 0), (7, 5), (8, 0), (8, 6), (8, 9), (12, 0), None]:  # fp16 everywhere
        half = runtime_dtype("float16", cap)
        assert (half.runtime, half.check) == ("float16", None)
    # float32 checkpoints run in half precision as before; on Turing the float16 candidate
    assert runtime_dtype("float32", (8, 6)).runtime == "bfloat16"
    assert runtime_dtype("float32", (7, 5)).check == dtypes.PENDING
    assert runtime_dtype(None, (12, 0)).runtime == "bfloat16"  # the checkpoint does not say
    assert runtime_dtype(None, (7, 5)).runtime == "float16"
    assert runtime_dtype("bfloat16", None).runtime == "bfloat16"  # no GPU known
    assert t4.to_dict() == {
        "requested": "auto",
        "checkpoint": "bfloat16",
        "runtime": "float16",
        "why": t4.why,
        "check": "pending",
        "candidate": "float16",
        "arch": "sm_75",
    }


def test_an_explicit_dtype_is_kept():
    kept = dtypes.resolve("bfloat16", "bfloat16", (7, 5))
    assert (kept.runtime, kept.check, kept.requested) == ("bfloat16", None, "bfloat16")
    assert kept.warning and "no bf16 tensor cores" in kept.warning
    assert kept.to_dict()["warning"] == kept.warning  # recorded in run.json
    assert dtypes.resolve("bfloat16", "bfloat16", (8, 6)).warning is None
    assert dtypes.resolve("float32", "bfloat16", (7, 5)).runtime == "float32"
    assert dtypes.resolve("float16", "bfloat16", (12, 0)).runtime == "float16"
    assert dtypes.resolve("bf16", "float16", (8, 9)).runtime == "bfloat16"
    assert dtypes.resolve("auto", "bfloat16", (7, 5)) == runtime_dtype("bfloat16", (7, 5))
    with pytest.raises(ValueError, match="unknown dtype 'int8'"):
        dtypes.resolve("int8", None, None)


def test_checkpoint_dtype_from_the_model_config():
    assert dtypes.checkpoint_dtype({"torch_dtype": "bfloat16"}) == "bfloat16"
    assert dtypes.checkpoint_dtype({"dtype": "float16"}) == "float16"  # newer configs, VoxCPM
    assert dtypes.checkpoint_dtype({"torch_dtype": "torch.float32"}) == "float32"
    assert dtypes.checkpoint_dtype({"torch_dtype": "auto"}) is None
    assert dtypes.checkpoint_dtype({}) is None and dtypes.checkpoint_dtype(None) is None


# ------------------------------------------------------------------ analyze's check


class ScaleToy(Workload):
    """Two linear layers and an RMS normalisation: finite in float32 at any ``scale``; at
    ``scale`` 1e5 the inputs exceed float16's range (65504) and its outputs are NaN."""

    modality = Modality.LLM
    defaults = {"scale": 1.0, "min_cosine": 0.99999}
    relaxed_options = {"min_cosine": 0.999}
    fail_in: str | None = None  # a dtype whose run raises

    def load(self) -> None:
        torch.manual_seed(0)
        self.model = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32)).eval()
        self.model.to(self.device, self.dtype)

    def roots(self) -> dict[str, nn.Module]:
        return {"model": self.model}

    def make_inputs(self) -> torch.Tensor:
        x = torch.randn(4, 32, generator=torch.Generator().manual_seed(1))
        return x.to(self.device, self.dtype)

    def run(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.spec.dtype == self.fail_in:
            raise RuntimeError("expected scalar type Half but found Float")
        h = self.model(inputs * float(self.options["scale"]))
        return {"out": (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)).float().cpu()}

    def compare(self, reference: Any, candidate: Any) -> Comparison:
        value = cosine(reference["out"], candidate["out"])
        minimum = float(self.options["min_cosine"])
        why = "" if value >= minimum else f"cosine {value:.5f} < {minimum}"
        return Comparison(value >= minimum, {"cosine": round(value, 6)}, why)


class HalfChaotic(ChaoticToy):
    """The chaotic toy in the dtype it is set to: its free-running trajectory diverges under
    any rounding change, so float16 is judged teacher forced."""

    chaotic = True

    def load(self) -> None:
        super().load()
        self.model.to(self.dtype)

    def make_inputs(self) -> torch.Tensor:
        return super().make_inputs().to(self.dtype)


def loader(cls: type[Workload] = ScaleToy, **options: Any):
    loaded: list[str] = []

    def load(dtype: str) -> Workload:
        loaded.append(dtype)
        workload = cls(WorkloadSpec("toy/scale", "llm", device="cpu", options=options))
        workload.set_dtype(dtype)
        workload.load()
        return workload

    load.loaded = loaded  # type: ignore[attr-defined]
    return load


def pending() -> dict[str, Any]:
    return {**runtime_dtype("bfloat16", (7, 5)).to_dict(), "reference": "float32"}


def test_float16_that_overflows_falls_back_to_float32():
    load = loader(scale=1e5)
    result = dtypes.check(load, "float16", "float32", gate=False)
    assert load.loaded == ["float32", "float16"]  # the reference first, then freed
    assert result["passed"] is False and "out.out: 128 non-finite values" in result["reason"]
    settled = dtypes.settle(pending(), {"status": "ok", **result}, fp32_fits=True)
    assert settled["runtime"] == "float32" and settled["check"]["passed"] is False
    assert settled["why"].startswith("float16 failed against float32 (out.out: 128 non-finite")
    assert settled["why"].endswith("float32, which fits this GPU")
    # where float32 does not fit: the checkpoint's bfloat16, which cannot overflow there
    tight = dtypes.settle(pending(), {"status": "ok", **result}, fp32_fits=False)
    assert tight["runtime"] == "bfloat16" and "float32 does not fit this GPU" in tight["why"]


def test_well_behaved_float16_is_kept_with_the_measured_reason():
    result = dtypes.check(loader(), "float16", "float32", gate=True)
    assert result["passed"] and result["reason"] == ""
    assert result["perceptual"]["status"] == "none"  # the toy declares no samples
    assert result["metrics"]["cosine"] > 0.999
    settled = dtypes.settle(pending(), {"status": "ok", **result}, fp32_fits=True)
    assert settled["runtime"] == "float16" and settled["check"]["passed"]
    assert settled["why"].startswith(
        "sm_75 has no bf16 tensor cores; float16 runs on them and matched a float32 run"
    )
    assert settled["why"].endswith("(cosine 1)")  # rounded by the comparison


def test_a_chaotic_workload_is_judged_teacher_forced():
    result = dtypes.check(loader(HalfChaotic, steps=40), "float16", "float32", gate=False)
    assert result["passed"], result["reason"]
    forced, free = result["metrics"]["teacher_forced"], result["metrics"]["free_running"]
    assert forced["passed"] and forced["mean_step_cosine"] > 0.9999
    assert not free["gating"]  # the diverged free run is informational


def test_a_float16_run_that_fails_is_a_failed_check():
    class Failing(ScaleToy):
        fail_in = "float16"

    result = dtypes.check(loader(Failing), "float16", "float32", gate=False)
    assert result["passed"] is False
    assert result["reason"].startswith("the float16 run failed (RuntimeError: expected scalar")


def test_no_verdict_keeps_the_checkpoints_dtype_and_retries_after_a_harness():
    error = {"status": "error", "error": "ValueError: no built-in workload for modality 'x'"}
    settled = dtypes.settle(pending(), error, fp32_fits=True)
    assert settled["runtime"] == "bfloat16" and settled["candidate"] == "float16"
    assert settled["why"].startswith("the float16 check could not run (ValueError: no built-in")
    assert settled["check"] == {"status": "error", "error": error["error"]}
    assert not dtypes.needs_check(settled) and dtypes.needs_check(settled, retry=True)
    assert dtypes.needs_check(pending()) and not dtypes.needs_check({"runtime": "bfloat16"})


# ------------------------------------------------------------------ the memory preflight


def recorded(runs: Path, stamp: str, options: dict[str, Any], peak: float) -> None:
    """A VoxCPM2 run recorded in ``runs`` with its baseline's peak memory."""
    root = runs / "openbmb--VoxCPM2" / stamp
    workload = {"repo_id": "openbmb/VoxCPM2", "dtype": "bfloat16", "options": options}
    write_json(root / "run.json", {"card": VOXCPM2, "workload": workload})
    write_json(root / "baseline.json", {"peak_mem_gb": peak})


def test_the_footprint_factor_is_learnt_from_recorded_runs(tmp_path):
    recorded(tmp_path, "20261005-192504", {}, 5.476819)  # the issue's evidence
    recorded(tmp_path, "20261006-004718", THROUGHPUT, 10.512156)
    factor, source = memfit.footprint_factor(tmp_path, "openbmb/VoxCPM2", {})
    assert factor == pytest.approx(1.284, abs=1e-3) and source.startswith("measured: 1 recorded")
    factor, _ = memfit.footprint_factor(tmp_path, "openbmb/VoxCPM2", THROUGHPUT)
    assert factor == pytest.approx(2.464, abs=1e-3)
    other = {"batch_size": 8, "metric": "throughput"}  # another batch: the metric's default
    assert memfit.footprint_factor(tmp_path, "openbmb/VoxCPM2", other)[0] == 2.46
    factor, source = memfit.footprint_factor(tmp_path, "Qwen/Qwen3-0.6B", {})
    assert factor == 1.29 and source.startswith("default")
    assert memfit.footprint_factor(None, "openbmb/VoxCPM2", {})[0] == 1.29


def test_the_preflight_refuses_what_does_not_fit_with_the_options(tmp_path):
    recorded(tmp_path, "20261006-004718", THROUGHPUT, 10.512156)  # a 10.5 GB footprint
    workload = {"repo_id": "openbmb/VoxCPM2", "options": THROUGHPUT}
    small = memfit.preflight(VOXCPM2, workload, "bfloat16", 8.0, tmp_path)
    assert not small["fits"] and small["estimate_gb"] == pytest.approx(10.51, abs=0.01)
    assert not small["fp32_fits"] and not small["ab_in_process"]
    message = memfit.refusal(small)
    assert message.startswith("memory preflight: the model in bfloat16 needs about 10.51 GB")
    for option in (
        "8-bit weight storage (weight-only INT8 or FP8, about 2.1 GB of weights)",
        "a smaller batch: -o batch_size=N (now 16)",
        "a GPU with more memory",
        "--no-memory-check: try anyway",
    ):
        assert option in message
    big = memfit.preflight(VOXCPM2, workload, "bfloat16", 24.0, tmp_path)
    assert big["fits"] and big["fp32_fits"] and big["ab_in_process"] and "options" not in big
    fp32 = memfit.preflight(VOXCPM2, {"repo_id": "openbmb/VoxCPM2"}, "float32", 8.0, tmp_path)
    assert fp32["weights_gb"] == pytest.approx(8.53, abs=0.01)
    assert "--dtype float16 or bfloat16 (float32 doubles the weights)" in memfit.refusal(fp32)
    assert "a smaller workload variant" in memfit.refusal(fp32)
    unknown = memfit.preflight({"repo_id": "x/y"}, {}, "bfloat16", 8.0)
    assert unknown["status"] == "unknown" and unknown["fits"] and unknown["ab_in_process"]
    assert memfit.describe(unknown) == "not estimated (the checkpoint size is unknown)"


@pytest.mark.parametrize(
    ("gpu", "options", "fits", "fp32_fits", "ab_in_process"),
    [
        ("T4", {}, True, True, True),  # 5.5 GB, float32 11.0 GB, A/B 9.8 GB of 14.1 usable
        ("T4", THROUGHPUT, True, False, False),  # 10.5 GB; A/B 14.8 GB
        ("RTX 3070", {}, True, False, False),  # 7.2 GB usable: A/B in two processes
        ("RTX 3070", THROUGHPUT, False, False, False),
        ("A10", THROUGHPUT, True, True, True),  # float32 21.0 GB of 21.6 usable
        ("L4", THROUGHPUT, True, True, True),
        ("RTX 5070 Ti", THROUGHPUT, True, False, True),  # A/B 14.8 GB of 15.0
    ],
)
def test_the_preflight_on_faked_gpus(tmp_path, gpu, options, fits, fp32_fits, ab_in_process):
    recorded(tmp_path, "20261005-192504", {}, 5.476819)
    recorded(tmp_path, "20261006-004718", THROUGHPUT, 10.512156)
    workload = {"repo_id": "openbmb/VoxCPM2", "options": options}
    gb = memfit.gpu_memory_gb(GPUS[gpu])
    fit = memfit.preflight(VOXCPM2, workload, "bfloat16", gb, tmp_path)
    assert (fit["fits"], fit["fp32_fits"], fit["ab_in_process"]) == (fits, fp32_fits, ab_in_process)


def test_the_emulated_memory_caps_the_gpus(monkeypatch):
    assert memfit.gpu_memory_gb(GPUS["RTX 5070 Ti"]) == 15.5
    monkeypatch.setenv(memfit.EMULATE_ENV, "8")
    assert memfit.gpu_memory_gb(GPUS["RTX 5070 Ti"]) == 8.0
    assert memfit.gpu_memory_gb(None) == 8.0
    monkeypatch.setenv(memfit.EMULATE_ENV, "32")
    assert memfit.gpu_memory_gb(GPUS["A10"]) == 22.1  # never more than the GPU has


# ------------------------------------------------------------------ the run


class FakeToolchain:
    env: dict[str, str] = {}
    backends = {"triton": True}
    gpu: Any = GPUS["T4"]

    def summary(self) -> str:
        return "GPU: none (test)"

    def to_dict(self) -> dict:
        return {}


def create(
    tmp_path, monkeypatch, gpu: str = "T4", size_gb: float = 1.4, runs: str = "runs", **cfg: Any
):
    """A new run of a bf16 checkpoint (``Orchestrator.create``) on a faked GPU."""
    toolchain_cls = type("GpuToolchain", (FakeToolchain,), {"gpu": GPUS[gpu]})
    monkeypatch.setattr(orchestrator.toolchain, "setup", toolchain_cls)
    monkeypatch.setattr("kernel_agent.kernels.roofline.ensure_peaks", lambda **k: None)
    card = ModelCard(
        repo_id="org/m",
        revision=None,
        modality=Modality.LLM,
        size_gb=size_gb,
        config={"torch_dtype": "bfloat16"},
    )
    monkeypatch.setattr(orchestrator.hub, "resolve", lambda ref, **kwargs: card)
    config = OptimizeConfig(model_ref="org/m", runs_dir=tmp_path / runs, **cfg)
    return orchestrator.Orchestrator.create(config)


def check_worker(calls: list[dict[str, Any]], **result: Any):
    """A ``dtype_check`` worker: what run.json said when it was called, then ``result``."""

    def call(run: RunDir, command: str, *args: str) -> dict[str, Any]:
        assert command == "dtype_check" and args == ()
        calls.append(dict(run.load()["dtype"]))
        return {"status": "ok", "reference": "float32", "candidate": "float16", **result}

    return call


def test_a_run_on_turing_checks_float16_before_analyze(tmp_path, monkeypatch):
    orch = create(tmp_path, monkeypatch)
    data = orch.run.load()
    assert data["config"]["dtype"] == "auto"
    assert data["dtype"]["check"] == "pending" and data["workload"]["dtype"] == "float16"
    calls: list[dict[str, Any]] = []
    passed = {"passed": True, "reason": "", "metrics": {"first_logits_cosine": 0.99987}}
    orch.worker = check_worker(calls, **passed)
    asyncio.run(orch.settle_dtype())
    assert calls[0]["reference"] == "float32"  # float32 fits the T4: the reference
    data = orch.run.load()
    choice = data["dtype"]
    assert choice["runtime"] == data["workload"]["dtype"] == "float16"
    assert choice["check"]["passed"] and "first_logits_cosine 0.99987" in choice["why"]
    fit = data["memory_fit"]
    assert fit["fits"] and fit["runtime"] == "float16" and fit["weights_gb"] == 1.4
    report = dtypes.report_lines(data)
    assert report[0].startswith("* dtype: **float16** (checkpoint: bfloat16; sm_75 has no")
    assert report[1].startswith("* memory preflight: 1.81 GB estimated in float16")
    section = dtypes.summary_section(choice)
    assert "## Run dtype" in section and "kernels take and return float16" in section

    calls.clear()  # settled: analyze runs again (a harness) without checking again
    asyncio.run(orch.settle_dtype(retry=True))
    assert calls == []


def test_a_failed_check_runs_float32_where_it_fits(tmp_path, monkeypatch):
    orch = create(tmp_path, monkeypatch)
    failed = {"passed": False, "reason": "out.first_logits: 12 non-finite values"}
    orch.worker = check_worker([], **failed)
    asyncio.run(orch.settle_dtype())
    data = orch.run.load()
    assert data["dtype"]["runtime"] == data["workload"]["dtype"] == "float32"
    assert "12 non-finite values" in data["dtype"]["why"]
    assert data["memory_fit"]["runtime"] == "float32" and data["memory_fit"]["weights_gb"] == 2.8


def test_a_model_that_does_not_fit_is_refused_before_loading(tmp_path, monkeypatch):
    orch = create(tmp_path, monkeypatch, gpu="RTX 3070", size_gb=10.0)
    called: list[dict[str, Any]] = []
    orch.worker = check_worker(called, passed=True)
    with pytest.raises(SystemExit, match=r"needs about 12\.9 GB") as exc:
        asyncio.run(orch.settle_dtype())
    assert "8-bit weight storage" in str(exc.value) and called == []
    assert orch.run.load()["memory_fit"]["fits"] is False  # recorded

    orch = create(tmp_path, monkeypatch, "RTX 3070", 10.0, runs="again", memory_check=False)
    asyncio.run(orch.settle_dtype())  # --no-memory-check: logged, and on it goes
    assert orch.run.load()["memory_fit"]["fits"] is False


def test_explicit_bfloat16_on_turing_is_kept_with_a_warning(tmp_path, monkeypatch):
    orch = create(tmp_path, monkeypatch, dtype="bfloat16")
    data = orch.run.load()
    assert data["dtype"]["runtime"] == data["workload"]["dtype"] == "bfloat16"
    assert "no bf16 tensor cores" in data["dtype"]["warning"]
    orch.worker = check_worker([])
    asyncio.run(orch.settle_dtype())  # nothing to check
    assert "check" not in orch.run.load()["dtype"]
    assert any(line.startswith("* **dtype warning**") for line in dtypes.report_lines(data))


def test_a_run_from_before_is_left_as_it_was(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator.toolchain, "setup", FakeToolchain)
    run = RunDir.create(tmp_path, "org/m")
    workload = {"repo_id": "org/m", "modality": "llm", "dtype": "bfloat16"}
    config = OptimizeConfig.from_dict({"model_ref": "org/m", "dtype": "bfloat16"})
    write_json(run.run_json, {"card": {"repo_id": "org/m"}, "workload": workload, "phases": {}})
    orch = orchestrator.Orchestrator(run, config)
    asyncio.run(orch.settle_dtype())
    assert "memory_fit" not in run.load() and config.dtype == "bfloat16"
    assert dtypes.summary_section(None) == "" and dtypes.report_lines({}) == []
    assert OptimizeConfig(model_ref="x").dtype == "auto"


def test_the_worker_loads_the_runs_dtype(tmp_path, monkeypatch):
    monkeypatch.setattr(toolchain, "setup", lambda *a, **k: None)
    run = RunDir.create(tmp_path, "toy/decoder")
    spec = WorkloadSpec("toy/decoder", "llm", dtype="bfloat16", device="cpu", harness=str(TOY))
    write_json(run.run_json, {"workload": spec.to_dict()})
    assert worker._workload(run).dtype == torch.bfloat16  # a run from before #255
    run.update(lambda d: d.__setitem__("dtype", {"runtime": "float32"}))
    workload = worker._workload(run)
    assert workload.dtype == torch.float32 and workload.spec.dtype == "float32"
    assert next(workload.model.parameters()).dtype == torch.float32
    assert worker._workload(run, "float16").dtype == torch.float16  # dtype_check's candidate
    with pytest.raises(ValueError, match="unknown dtype 'int4'"):
        workload.set_dtype("int4")


def test_two_states_that_do_not_fit_are_measured_in_two_processes(tmp_path):
    orch = make(tmp_path)
    run = orch.run
    voxcpm2(run)
    why = "two states of the model need about 14.8 GB of the 14.1 GB usable"
    run.update(lambda d: d.__setitem__("memory_fit", {"ab_in_process": False, "ab_why": why}))
    calls: list[list[str]] = []
    commands: list[str] = []
    measure = fake_worker(COMBINATIONS, calls)

    def counted(run: RunDir, command: str, *args: str) -> dict[str, Any]:
        commands.append(command)
        return measure(run, command, *args)

    orch.worker = counted
    asyncio.run(orch.integrate())
    history = read_json(run.root / "integration.json")["history"]
    pairs = [h["ab"] for h in history if "ab" in h]
    assert pairs and all(ab["mode"] == "separate" for ab in pairs)
    assert all(ab["fallback"] == f"the memory preflight: {why}" for ab in pairs)
    assert "e2e_ab" not in commands


def test_the_cli_defaults_to_auto(monkeypatch):
    seen: list[OptimizeConfig] = []
    monkeypatch.setattr(cli, "cmd_optimize", lambda ns: seen.append(cli._config(ns)) or 0)
    assert cli.main(["optimize", "org/m"]) == 0
    assert cli.main(["optimize", "org/m", "--dtype", "float16", "--no-memory-check"]) == 0
    assert (seen[0].dtype, seen[0].memory_check) == ("auto", True)
    assert (seen[1].dtype, seen[1].memory_check) == ("float16", False)
    old = OptimizeConfig.from_dict({"model_ref": "org/m", "dtype": "bfloat16"})  # recorded
    assert (old.dtype, old.memory_check) == ("bfloat16", True)


# ------------------------------------------------------------------ VoxCPM


@pytest.mark.skipif(importlib.util.find_spec("voxcpm") is None, reason="needs voxcpm")
def test_the_voxcpm_workload_honours_set_dtype(tmp_path, monkeypatch):
    import voxcpm.model.voxcpm2 as voxcpm2_module
    import voxcpm_tiny

    from kernel_agent.workloads.voxcpm import VoxCPMWorkload

    checkpoint = voxcpm_tiny.write_checkpoint(tmp_path / "ckpt", "bfloat16")
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda repo, **k: str(checkpoint))
    tokenizer = SimpleNamespace(from_pretrained=lambda path: SimpleNamespace(vocab={}))
    monkeypatch.setattr(voxcpm2_module, "LlamaTokenizerFast", tokenizer)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def load(dtype: str | None) -> VoxCPMWorkload:
        spec = WorkloadSpec(
            "openbmb/VoxCPM2", "tts", family="voxcpm", device="cpu", options={"patches": 3}
        )
        workload = VoxCPMWorkload(spec)
        if dtype is not None:
            workload.set_dtype(dtype)
        workload.load()
        workload.model.text_tokenizer = voxcpm_tiny.CharTokenizer()
        return workload

    native = load(None)  # a run from before #255: the checkpoint's dtype
    assert native.model.config.dtype == "bfloat16"
    assert next(native.model.base_lm.parameters()).dtype == torch.bfloat16
    for dtype, torch_dtype in (("float32", torch.float32), ("float16", torch.float16)):
        workload = load(dtype)
        assert workload.model.config.dtype == dtype and workload.config_dtype == dtype
        for module in (workload.model.base_lm, workload.model.feat_decoder):
            assert next(module.parameters()).dtype == torch_dtype
        assert next(workload.model.audio_vae.parameters()).dtype == torch.float32
        weight = workload.model.base_lm.layers[0].self_attn.q_proj.weight
        expected = native.model.base_lm.layers[0].self_attn.q_proj.weight.to(torch_dtype)
        assert torch.equal(weight, expected)  # the checkpoint's values, converted
    out = load("float32").run("Hello")  # the model runs in the dtype it was given
    assert out["latents"].shape[0] == 3 and torch.isfinite(out["audio"]).all()
    assert '"dtype": "bfloat16"' in (checkpoint / "config.json").read_text()  # untouched


# ------------------------------------------------------------------ GPU


def _cached(repo: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(repo, "config.json"), str)


@pytest.mark.gpu
@pytest.mark.skipif(not _cached("Qwen/Qwen3-0.6B"), reason="needs Qwen/Qwen3-0.6B in the HF cache")
def test_float16_of_a_real_llm_matches_float32():
    from kernel_agent.workloads.llm import LLMWorkload

    def load(dtype: str) -> Workload:
        spec = WorkloadSpec("Qwen/Qwen3-0.6B", "llm", options={"prompt_len": 128, "new_tokens": 8})
        workload = LLMWorkload(spec)
        workload.set_dtype(dtype)
        workload.load()
        return workload

    result = dtypes.check(load, "float16", "float32", gate=False)
    assert result["passed"], result["reason"]
    assert result["metrics"]["first_logits_cosine"] > 0.999
