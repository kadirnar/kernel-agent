"""Nsight Compute feedback (#10): the CSV parser and roofline classes on recorded-style
fixtures, graceful degradation (no ncu, ERR_NVGPUCTRPERM, unknown metrics), compiler stats."""

from __future__ import annotations

import contextlib
import subprocess
from pathlib import Path
from typing import Any

import pytest

from kernel_agent import gpulock
from kernel_agent.kernels import ncu

FIXTURES = Path(__file__).parent / "fixtures" / "ncu"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


def test_the_raw_csv_page_parses_with_its_prof_log_and_units_row():
    launches = ncu.parse_csv(fixture("raw_base_units.csv"))
    assert [launch["kernel"] for launch in launches] == [
        "rmsnorm_bf16",
        "fp8_gemm_kernel",
        "add_one",
        "rmsnorm_bf16",
        "fp8_gemm_kernel",
    ]
    first = launches[0]
    assert first["id"] == 0 and first["grid"] == "(4096, 1, 1)" and first["block"] == "(256, 1, 1)"
    m = first["metrics"]
    assert m["time_ns"] == 43520.0 and m["dram_read_bytes"] == 16777216.0
    assert m["sm_pct"] == 21.33 and m["memory_pct"] == 89.02 and m["registers"] == 32.0
    assert m["stall_long_scoreboard"] == 6.85
    assert set(m) == set(ncu.METRICS.values())  # every curated metric, by its short name


def test_scaled_units_and_thousands_separators_become_ns_and_bytes():
    (launch,) = ncu.parse_csv(fixture("raw_scaled_units.csv"))
    m = launch["metrics"]
    assert m["time_ns"] == pytest.approx(1_234_570.0)  # "1,234.57" usecond
    assert m["dram_read_bytes"] == pytest.approx(16.78e6)  # Mbyte


def test_no_report_parses_to_nothing():
    assert ncu.parse_csv(fixture("err_nvgpuctrperm.txt")) == []
    assert ncu.parse_csv("") == []


def test_kernels_aggregate_launches_and_classify_the_roofline():
    found = ncu.kernels(ncu.parse_csv(fixture("raw_base_units.csv")))
    by_name = {k["kernel"]: k for k in found}
    assert [k["kernel"] for k in found] == ["rmsnorm_bf16", "fp8_gemm_kernel", "add_one"]
    rms = by_name["rmsnorm_bf16"]
    assert rms["launches"] == 2 and rms["metrics"]["time_ns"] == 43520.0 + 43776.0
    assert rms["metrics"]["dram_read_bytes"] == 2 * 16777216.0  # summed
    assert 88.60 < rms["metrics"]["memory_pct"] < 89.02  # time-weighted
    assert rms["bound"] == "memory" and "DRAM 88 %" in rms["why"]
    gemm = by_name["fp8_gemm_kernel"]
    assert gemm["bound"] == "compute" and "tensor pipe 71 %" in gemm["why"]
    assert "math_pipe_throttle" in gemm["why"]
    tiny = by_name["add_one"]
    assert tiny["bound"] == "under-utilised"
    assert "0.03 waves: too few blocks" in tiny["why"]
    assert "long_scoreboard 3.4 (waiting on global / local memory loads)" in tiny["why"]


def test_classify_without_throughputs_is_unknown():
    assert ncu.classify({"time_ns": 1.0})["bound"] == "unknown"


def test_unknown_metrics_are_named_from_ncu_output():
    metrics = list(ncu.METRICS)
    bad = ncu.unknown_metrics(fixture("unknown_metric.txt"), metrics)
    assert bad == ["sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"]
    assert ncu.unknown_metrics("==PROF== fine", metrics) == []


def test_counters_restricted_reads_the_driver_parameter():
    assert ncu.counters_restricted("RmProfilingAdminOnly: 1\n", root=False) is True
    assert ncu.counters_restricted("RmProfilingAdminOnly: 1\n", root=True) is False
    assert ncu.counters_restricted("RmProfilingAdminOnly: 0\n", root=False) is False
    assert ncu.counters_restricted("RmGpuComputeExecTimeout: 0\n", root=False) is None


def test_root_without_cap_sys_admin_is_no_admin_to_the_driver():
    """Root in an unprivileged container (the A10 container's CapEff: no CAP_SYS_ADMIN) is
    refused the counters (ERR_NVGPUCTRPERM): only CAP_SYS_ADMIN counts."""
    container = "Name:\tpython\nUid:\t0\t0\t0\t0\nCapEff:\t00000000a80405fb\n"
    privileged = "Uid:\t0\t0\t0\t0\nCapEff:\t000001ffffffffff\n"
    assert not ncu.is_admin(container)
    assert ncu.is_admin(privileged)
    assert ncu.is_admin("CapEff:\t0000000000200000\n")  # CAP_SYS_ADMIN alone


def test_the_command_profiles_the_candidate_range_only(tmp_path):
    cmd = ncu.command(
        "/opt/ncu",
        Path("cap.pt"),
        Path("cand.py"),
        ["a.sum", "b.pct"],
        tmp_path / "log.csv",
        capture_sha256="abc",
    )
    assert cmd[:3] == ["/opt/ncu", "--csv", "--page"]
    assert cmd[cmd.index("--nvtx-include") + 1] == f"{ncu.NVTX_RANGE}/"
    assert cmd[cmd.index("--metrics") + 1] == "a.sum,b.pct"
    assert cmd[cmd.index("--replay-mode") + 1] == "kernel"
    assert "--ncu-mode" in cmd and cmd[-2:] == ["--capture-sha256", "abc"]


@pytest.fixture
def unlocked(monkeypatch):
    monkeypatch.setattr(gpulock, "gpu_lock", contextlib.nullcontext)
    monkeypatch.setattr(gpulock, "child_env", lambda: {})


def test_without_ncu_profile_degrades_with_the_reason(monkeypatch):
    monkeypatch.setattr(ncu, "find_ncu", lambda places=None: None)
    out = ncu.profile_candidate(Path("c.pt"), Path("k.py"))
    assert out["status"] == "unavailable" and "ncu" in out["reason"]


def test_restricted_counters_degrade_before_running(monkeypatch):
    monkeypatch.setattr(ncu, "find_ncu", lambda places=None: "/opt/ncu")
    monkeypatch.setattr(ncu, "ncu_version", lambda path: "2025.3.0")
    monkeypatch.setattr(ncu, "counters_restricted", lambda *a, **k: True)
    assert ncu.availability()["usable"] is False
    out = ncu.profile_candidate(Path("c.pt"), Path("k.py"), run=pytest.fail)
    assert out == {"status": "unavailable", "reason": ncu.PERMISSION_HINT}
    assert "NVreg_RestrictProfilingToAdminUsers=0" in ncu.describe(ncu.availability())


class FakeNcu:
    """``subprocess.run`` for ncu: writes the ``--log-file`` from ``outputs`` in turn."""

    def __init__(self, *outputs: str) -> None:
        self.outputs = list(outputs)
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(cmd)
        Path(cmd[cmd.index("--log-file") + 1]).write_text(self.outputs.pop(0))
        return subprocess.CompletedProcess(cmd, 0, "", "")


@pytest.fixture
def usable(monkeypatch, unlocked):
    monkeypatch.setattr(ncu, "find_ncu", lambda places=None: "/opt/ncu")
    monkeypatch.setattr(ncu, "ncu_version", lambda path: "2025.3.0")
    monkeypatch.setattr(ncu, "counters_restricted", lambda *a, **k: False)


def test_permission_error_at_run_time_degrades(usable):
    run = FakeNcu(fixture("err_nvgpuctrperm.txt"))
    out = ncu.profile_candidate(Path("c.pt"), Path("k.py"), run=run)
    assert out["status"] == "unavailable" and "ERR_NVGPUCTRPERM" in out["reason"]


def test_an_unknown_metric_is_dropped_and_the_run_repeated(usable):
    run = FakeNcu(fixture("unknown_metric.txt"), fixture("raw_base_units.csv"))
    out = ncu.profile_candidate(Path("c.pt"), Path("k.py"), run=run)
    assert out["status"] == "ok" and out["launches"] == 5 and out["ncu"] == "2025.3.0"
    assert out["dropped_metrics"] == [
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
    ]
    second = run.calls[1][run.calls[1].index("--metrics") + 1].split(",")
    assert "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active" not in second
    assert [k["bound"] for k in out["kernels"]] == ["memory", "compute", "under-utilised"]


def test_a_failure_without_a_report_says_what_ncu_printed(usable):
    out = ncu.profile_candidate(Path("c.pt"), Path("k.py"), run=FakeNcu("==ERROR== boom"))
    assert out["status"] == "error" and "boom" in out["reason"]


# ------------------------------------------------------------------ compiler stats


def test_ptxas_and_cuobjdump_resource_usage_parse():
    ptxas = ncu.parse_ptxas(fixture("ptxas_v.txt"))
    assert ptxas == [
        {
            "kernel": "_Z13rmsnorm_bf16PK13__nv_bfloat16S1_PS_if",
            "stack_bytes": 0,
            "spill_stores": 0,
            "spill_loads": 0,
            "registers": 18,
            "shared_bytes": 32,
        },
        {
            "kernel": "_Z9big_tilePf",
            "stack_bytes": 48,
            "spill_stores": 44,
            "spill_loads": 44,
            "registers": 255,
        },
    ]
    usage = ncu.parse_resource_usage(fixture("cuobjdump_resource_usage.txt"))
    assert [(k["registers"], k["local_bytes"], k["shared_bytes"]) for k in usage] == [
        (40, 0, 0),
        (255, 96, 1024),
    ]
    assert usage[0]["arch"] == "sm_120" and usage[1]["kernel"].startswith("_Z14rmsnorm")


class _Meta:
    name = "matmul_kernel"
    shared = 49152
    num_warps = 4
    num_stages = 3


class _Compiled:
    metadata = _Meta()
    n_regs = 168
    n_spills = 12


class _Jit:
    """A Triton JITFunction's compiled-kernel cache (``device_caches``)."""

    device_caches = {0: ({"key": _Compiled()}, None, None)}


class _Autotuner:
    fn = _Jit()


class _Attrs:
    num_regs = 40
    local_size_bytes = 0
    shared_size_bytes = 128
    max_threads_per_block = 1024


_NvrtcKernel = type("Kernel", (), {"__module__": "cuda.core._module", "attributes": _Attrs()})


def test_compiler_stats_of_triton_and_nvrtc_kernels(monkeypatch):
    monkeypatch.setattr(ncu, "extension_files", list)
    module = type("Candidate", (), {})()
    module.matmul = _Autotuner()
    module._kernel = _NvrtcKernel()
    module.unrelated = 3
    stats = ncu.compiler_stats(module)
    assert stats["triton"] == [
        {
            "kernel": "matmul_kernel",
            "registers": 168,
            "spills": 12,
            "shared_bytes": 49152,
            "num_warps": 4,
            "num_stages": 3,
        }
    ]
    assert stats["nvrtc"] == [
        {
            "kernel": "_kernel",
            "registers": 40,
            "local_bytes": 0,
            "shared_bytes": 128,
            "max_threads": 1024,
        }
    ]
    assert "cuda" not in stats
    (warning,) = stats["warnings"]
    assert warning.startswith("matmul_kernel: 12 spilled (registers 168)")


def test_cuda_stats_run_cuobjdump_on_extension_files(monkeypatch, tmp_path):
    so = tmp_path / "ext.so"
    so.write_bytes(b"")

    def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert cmd == ["/bin/cuobjdump", "--dump-resource-usage", str(so)]
        return subprocess.CompletedProcess(cmd, 0, fixture("cuobjdump_resource_usage.txt"), "")

    monkeypatch.setattr(ncu.subprocess, "run", run)
    rows = ncu.cuda_stats([so], tool="/bin/cuobjdump")
    assert [r["file"] for r in rows] == ["ext.so", "ext.so"] and rows[1]["local_bytes"] == 96


def test_the_tool_accepts_profile_ncu():
    from kernel_agent.agent import tools

    assert "compiler_stats" in tools.compact({"status": "ok", "compiler_stats": {"triton": []}})


# ------------------------------------------------------------------ rules and source lines (#230)


def test_details_page_rules_rank_by_estimated_speedup():
    rules = ncu.parse_details(fixture("details_rules.csv"))
    assert len(rules) == 9  # rule rows only; metric rows and the ==PROF== log are skipped
    assert {r["type"] for r in rules} == {"OPT", "WRN", "INF"}
    top = ncu.top_rules(rules)
    assert [(r["kernel"], r["rule"], r["speedup_pct"]) for r in top] == [
        ("_quant_kernel", "LaunchConfiguration", 50.0),
        ("_gemm_kernel", "CPIStall", 43.0),  # the larger of its two launches
        ("_gemm_kernel", "TheoreticalOccupancy", 33.3),
    ]
    assert top[1]["description"].endswith("(local, global, surface, texture) operation.")
    ranked = ncu.top_rules(rules, n=10)
    assert [r["rule"] for r in ranked][-2:] == ["UncoalescedGlobalAccess", "IssueSlotUtilization"]
    assert all(r["speedup_pct"] > 0 for r in ranked)  # no estimate: not ranked


def test_source_lines_with_line_info_rank_by_stall_samples():
    rows = ncu.parse_source(fixture("source_cuda_sass.csv"))
    assert {r["kernel"] for r in rows} == {"_gemm_kernel", "_quant_kernel"}
    assert sum(r["line"] is not None for r in rows) == 6  # the rest: SASS rows
    top = ncu.top_lines(rows)
    assert [(t["where"], t["samples"], t["share_pct"]) for t in top] == [
        ("003_fp8_gemm.py:39", 5200, 58.5),
        ("003_fp8_gemm.py:40", 3100, 34.9),
        ("003_fp8_gemm.py:18", 900, 81.8),
        ("003_fp8_gemm.py:47", 400, 4.5),
        ("003_fp8_gemm.py:17", 200, 18.2),
    ]
    assert (top[0]["stall"], top[0]["stall_pct"]) == ("stall_long_sb", 78.8)
    assert top[1]["stall"] == "stall_math" and "math pipe" in top[1]["why"]
    assert top[0]["source"] == "a_t = tl.load(a_ptr, mask=row_ok, other=0.0)"
    flagged = ncu.flagged_lines(rows)
    assert [(f["where"], f["flags"]) for f in flagged] == [
        ("003_fp8_gemm.py:47", ["uncoalesced global access (2048 of 4096 L2 sectors excessive)"]),
        ("003_fp8_gemm.py:18", ["shared-memory bank conflicts (4-way)"]),
    ]


def test_source_without_line_info_ranks_sass_instructions():
    rows = ncu.parse_source(fixture("source_sass.csv"))
    assert all(r["line"] is None and r["address"].startswith("0x") for r in rows)
    (top,) = ncu.top_lines(rows, n=1)
    assert top["where"] == "SASS 0x7f3a20000020" and top["source"] == "HMUL2.BF16_V2 R8, R4, R4"
    assert (top["share_pct"], top["stall"]) == (68.7, "stall_long_sb")
    assert ncu.flagged_lines(rows) == []


def test_the_details_command_exports_the_sections_of_one_call(tmp_path):
    cmd = ncu.details_command(
        "/opt/ncu", Path("cap.pt"), Path("cand.py"), tmp_path / "r.ncu-rep", tmp_path / "l"
    )
    sections = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--section"]
    assert sections == list(ncu.SECTIONS) and "SourceCounters" in sections
    assert cmd[cmd.index("--export") + 1] == str(tmp_path / "r.ncu-rep")
    assert cmd[cmd.index("--launch-count") + 1] == str(ncu.DETAILS_LAUNCHES)
    assert cmd[-2:] == ["--ncu-calls", "1"]
    assert ncu.import_command("/opt/ncu", Path("r"), "source", "sass")[-4:] == [
        "--page",
        "source",
        "--print-source",
        "sass",
    ]


class FakeNcuDetails(FakeNcu):
    """:class:`FakeNcu` for both runs: the metrics run's log, the details run's report
    file, and the imported pages on stdout (``pages``: page or print-source -> text)."""

    def __init__(self, raw: str, pages: dict[str, str], *, report: bool = True) -> None:
        super().__init__(raw)
        self.pages = pages
        self.report = report

    def __call__(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "--import" in cmd:
            self.calls.append(cmd)
            key = cmd[-1] if "--print-source" in cmd else cmd[cmd.index("--page") + 1]
            return subprocess.CompletedProcess(cmd, 0, self.pages.get(key, ""), "")
        if "--export" in cmd:
            self.calls.append(cmd)
            if self.report:
                Path(cmd[cmd.index("--export") + 1]).write_bytes(b"report")
            return subprocess.CompletedProcess(cmd, 0 if self.report else 9, "", "==ERROR== x")
        return super().__call__(cmd, **kwargs)


def test_profile_ncu_adds_rules_and_stall_lines(usable):
    pages = {
        "details": fixture("details_rules.csv"),
        "cuda,sass": "",  # no line info here: the SASS view is read instead
        "sass": fixture("source_sass.csv"),
    }
    run = FakeNcuDetails(fixture("raw_base_units.csv"), pages)
    out = ncu.profile_candidate(Path("c.pt"), Path("k.py"), run=run)
    assert out["status"] == "ok" and len(out["kernels"]) == 3
    assert [r["rule"] for r in out["rules"]] == [
        "LaunchConfiguration",
        "CPIStall",
        "TheoreticalOccupancy",
    ]
    assert out["lines"][0]["source"] == "HMUL2.BF16_V2 R8, R4, R4" and out["flagged"] == []
    views = [c[-1] for c in run.calls if "--print-source" in c]
    assert views == ["cuda,sass", "sass"]


def test_a_failed_details_run_keeps_the_metrics(usable):
    run = FakeNcuDetails(fixture("raw_base_units.csv"), {}, report=False)
    out = ncu.profile_candidate(Path("c.pt"), Path("k.py"), run=run)
    assert out["status"] == "ok" and len(out["kernels"]) == 3
    assert out["details"]["status"] == "error" and "==ERROR== x" in out["details"]["reason"]
    assert "rules" not in out
