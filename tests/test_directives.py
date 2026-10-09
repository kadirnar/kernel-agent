"""Directives from a candidate's profile (#230): the documented rules on fixture censuses
(tests/fixtures/sass) and ncu reports (tests/fixtures/ncu) with fake GPU facts, at most
five per evaluation, each with its numbers; the profile summary stays small."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kernel_agent.agent import tools as tools_mod
from kernel_agent.kernels import directives, ncu, sass

SASS = Path(__file__).parent / "fixtures" / "sass"
NCU = Path(__file__).parent / "fixtures" / "ncu"
#: Fake GPU facts (the census's ``gpu``): measured rates as on an RTX 5070 Ti.
SM120 = {
    "name": "GeForce RTX 5070 Ti",
    "arch": "sm_120",
    "capability": [12, 0],
    "mma_tflops": {"bf16_f32": 103.4, "e4m3_f32": 206.0, "e4m3_sf_f32": 412.6},
}
SM100 = {"arch": "sm_100", "capability": [10, 0]}
SM90 = {"arch": "sm_90", "capability": [9, 0]}
SM89 = {"arch": "sm_89", "capability": [8, 9]}
SM80 = {"arch": "sm_80", "capability": [8, 0]}
COMPUTE = {"bound": "compute", "pct_of_sol": 21.0}
MEMORY = {"bound": "memory", "pct_of_sol": 48.0}


def rows(name: str, *kernels: str) -> list[dict[str, Any]]:
    """Census rows of a fixture dump (only ``kernels`` when given)."""
    found = [sass.summarise(k) for k in sass.parse((SASS / name).read_text())]
    return [r for r in found if not kernels or r["kernel"] in kernels]


def tables(facts: dict[str, Any], census: list[dict[str, Any]], **more: Any) -> dict[str, Any]:
    return {"sass": {"status": "ok", "gpu": facts, "kernels": census}, **more}


def rules_of(found: list[dict[str, Any]]) -> list[tuple[str, str | None]]:
    return [(d["rule"], d.get("kernel")) for d in found]


# ------------------------------------------------------------------ tensor-core instructions


def test_tl_dot_on_sm120_is_told_the_block_scaled_rate():
    found = directives.build(tables(SM120, rows("triton_fp8_dot_sm120a.sass")), COMPUTE)
    assert rules_of(found) == [("tensor_rate", "_gemm_kernel")]
    text = found[0]["text"]
    assert "compute-bound at 21 % of SOL" in text
    assert "issues QMMA.16832.F32.E4M3.E4M3 only (4 in its SASS)" in text
    assert "QMMA.SF" in text and "tl.dot_scaled" in text
    assert text.endswith("measured 413 vs 206 TFLOP/s")
    assert found[0]["evidence"]["full_rate"] == ["QMMA.SF", "QMMA.16832.F16"]
    assert len(text) <= directives.MAX_CHARS


def test_tl_dot_scaled_on_sm120_is_at_the_full_rate():
    census = rows("triton_fp8_dot_scaled_sm120a.sass")
    assert directives.build(tables(SM120, census), COMPUTE) == []


def test_the_measured_rates_decide_on_sm12x():
    census = rows("triton_fp8_dot_sm120a.sass")
    unmeasured = {k: v for k, v in SM120.items() if k != "mma_tflops"}
    (found,) = directives.build(tables(unmeasured, census), COMPUTE)
    assert found["rule"] == "tensor_rate" and "measured" not in found["text"]
    same = SM120 | {"mma_tflops": {"e4m3_f32": 206.0, "e4m3_sf_f32": 210.0}}
    assert directives.build(tables(same, census), COMPUTE) == []


def test_a_memory_bound_kernel_is_not_told_about_instruction_rates():
    census = rows("triton_fp8_dot_sm120a.sass")
    assert directives.build(tables(SM120, census), MEMORY) == []
    # Nsight's class of the kernel overrides the evaluation's roofline bound
    report = {"status": "ok", "kernels": [{"kernel": "_gemm_kernel", "bound": "memory"}]}
    assert directives.build(tables(SM120, census, ncu=report), COMPUTE) == []


def test_hopper_mma_sync_and_emulated_fp8():
    census = rows("sm_90a.sass", "ka_mma_bf16", "ka_wgmma", "ka_mma_e4m3")
    found = directives.build(tables(SM90, census), COMPUTE)
    assert rules_of(found) == [("fp8_emulated", "ka_mma_e4m3"), ("tensor_rate", "ka_mma_bf16")]
    assert "12 F2FP e4m3 -> fp16 unpacks feed 2 HMMA" in found[0]["text"]
    assert "wgmma (HGMMA / QGMMA / IGMMA)" in found[1]["text"]


def test_datacenter_blackwell_wants_tcgen05():
    census = rows("sm_100a.sass", "ka_mma_bf16", "ka_tcgen05", "ka_mma_e4m3")
    found = directives.build(tables(SM100, census), COMPUTE)
    assert ("tensor_rate", "ka_tcgen05") not in rules_of(found)
    assert ("fp8_emulated", "ka_mma_e4m3") in rules_of(found)
    rate = next(d for d in found if d.get("kernel") == "ka_mma_bf16")
    assert "tcgen05.mma (UTCHMMA" in rate["text"]


def test_mma_sync_is_the_full_rate_on_ada_and_ampere():
    assert directives.build(tables(SM89, rows("sm_89.sass", "ka_mma_e4m3")), COMPUTE) == []
    census = rows("sm_80.sass", "ka_mma_bf16", "ka_mma_s8")
    assert directives.build(tables(SM80, census), COMPUTE) == []


def test_a_family_newer_than_the_table_gets_no_rate_directive():
    newer = {"arch": "sm_130", "capability": [13, 0]}
    census = rows("triton_fp8_dot_sm120a.sass")
    assert directives.build(tables(newer, census), COMPUTE) == []
    assert directives.full_rate(newer) == {}


# ------------------------------------------------------------------ memory


def test_spills_are_flagged_with_their_numbers():
    census = rows("sm_80.sass", "ka_spill")
    stats = {"nvrtc": [{"kernel": "ka_spill", "registers": 32, "local_bytes": 176}]}
    (found,) = directives.build(tables(SM80, census, compiler_stats=stats), {})
    assert found["rule"] == "local_memory" and found["kernel"] == "ka_spill"
    assert "79 LDL / 78 STL; registers 32, 176 spilled per compiler stats" in found["text"]
    assert found["evidence"] == {"ldl": 79, "stl": 78, "registers": 32}


def test_narrow_loads_of_a_memory_bound_kernel():
    row = {
        "kernel": "gather",
        "arch": "sm_120",
        "categories": {"global_load": 12},
        "global_load_bits": {"32": 10, "128": 2},
    }
    (found,) = directives.build(tables(SM120, [row]), MEMORY)
    assert found["rule"] == "narrow_loads"
    assert "memory-bound at 48 % of SOL" in found["text"]
    assert "10 x 32-bit, 2 x 128-bit" in found["text"]
    staged = row | {"categories": {"global_load": 12, "tma": 2}}
    assert directives.build(tables(SM120, [staged]), MEMORY) == []
    assert directives.build(tables(SM120, [row]), COMPUTE) == []


def test_fp32_math_without_tensor_cores_and_register_staged_operands():
    fma = {"kernel": "naive_gemm", "arch": "sm_80", "categories": {"fp32_math": 640}}
    staged = {
        "kernel": "mma_gemm",
        "arch": "sm_80",
        "tensor": {"HMMA.16816.F32.BF16": 32},
        "categories": {"tensor": 32, "global_load": 8, "shared_store": 4, "shared_load": 8},
    }
    found = directives.build(tables(SM80, [fma, staged]), COMPUTE)
    assert rules_of(found) == [("no_tensor_cores", "naive_gemm"), ("no_async_copy", "mma_gemm")]
    assert "640 fp32 and 0 fp16 math instructions" in found[0]["text"]
    assert "(8 LDG + 4 STS, no LDGSTS / UTMALDG)" in found[1]["text"]


# ------------------------------------------------------------------ Nsight Compute


def report() -> dict[str, Any]:
    lines = ncu.parse_source((NCU / "source_cuda_sass.csv").read_text())
    return {
        "status": "ok",
        "kernels": [
            {"kernel": "_gemm_kernel", "bound": "compute", "metrics": {"tensor_pipe_pct": 38.0}},
            {"kernel": "_quant_kernel", "bound": "under-utilised", "metrics": {}},
        ],
        "rules": ncu.top_rules(ncu.parse_details((NCU / "details_rules.csv").read_text())),
        "lines": ncu.top_lines(lines),
        "flagged": ncu.flagged_lines(lines),
    }


def test_ncu_stall_lines_flags_and_rules_become_directives():
    found = directives.build({"ncu": report()}, COMPUTE)
    assert len(found) == directives.MAX_DIRECTIVES
    assert rules_of(found) == [
        ("stall_line", "_gemm_kernel"),
        ("stall_line", "_gemm_kernel"),
        ("uncoalesced", "_gemm_kernel"),
        ("uncoalesced", "_quant_kernel"),
        ("ncu_rule", "_quant_kernel"),
    ]
    assert found[0]["text"] == (
        "_gemm_kernel 003_fp8_gemm.py:39: 58 % of its warp-stall samples, mostly stall_long_sb "
        "(78.8 %: waiting on global / local memory (long scoreboard)): "
        "`a_t = tl.load(a_ptr, mask=row_ok, other=0.0)`"
    )
    assert "uncoalesced global access (2048 of 4096 L2 sectors excessive)" in found[2]["text"]
    assert found[4]["text"].startswith(
        "Nsight LaunchConfiguration on _quant_kernel (global speedup 50.0 %)"
    )
    unavailable = {"ncu": {"status": "unavailable", "reason": "ERR_NVGPUCTRPERM"}}
    assert directives.build(unavailable, COMPUTE) == []


def test_ncu_bound_and_tensor_pipe_are_the_evidence():
    census = rows("triton_fp8_dot_sm120a.sass")
    found = directives.build(tables(SM120, census, ncu=report()), COMPUTE)
    rate = next(d for d in found if d["rule"] == "tensor_rate")
    assert rate["text"].startswith("_gemm_kernel: ncu: compute, tensor pipe 38 %; issues")
    assert found[0] is rate  # a rate directive comes before the stall lines


def test_an_idle_tensor_pipe_is_named_with_its_stalls():
    """KernelPro's directive: the MMAs are right but the tensor pipe is 0.7 % busy."""
    census = rows("triton_fp8_dot_scaled_sm120a.sass")
    why = "SM 71 % of peak (memory 40 %); top stalls: long_scoreboard 9.8 (waiting on global)"
    kernel = {"kernel": "_gemm_kernel", "bound": "compute", "why": why}
    report = {"status": "ok", "kernels": [kernel | {"metrics": {"tensor_pipe_pct": 0.7}}]}
    (found,) = directives.build(tables(SM120, census, ncu=report), COMPUTE)
    assert found["rule"] == "tensor_idle"
    assert found["text"].startswith(
        "_gemm_kernel: tensor pipe 0.7 % active (ncu) though it issues "
        "QMMA.SF.16832.F32.E4M3.E4M3.E8: the MMAs wait on their operands; top stalls: "
        "long_scoreboard 9.8"
    )
    busy = {"status": "ok", "kernels": [kernel | {"metrics": {"tensor_pipe_pct": 71.0}}]}
    assert directives.build(tables(SM120, census, ncu=busy), COMPUTE) == []


# ------------------------------------------------------------------ ranking and the summary


def test_kernels_that_did_not_run_are_left_out_and_larger_shares_come_first():
    census = rows("sm_80.sass", "ka_spill", "ka_local", "ka_mma_bf16")
    timed = [
        {"kernel": "ka_local(float*, const int*)", "calls": 1, "us": 90.0},
        {"kernel": "void ka_spill(float*, int)", "calls": 1, "us": 10.0},
    ]
    found = directives.build(tables(SM80, census, kernels_candidate=timed), {})
    assert rules_of(found) == [("local_memory", "ka_local"), ("local_memory", "ka_spill")]
    many = [{**r, "kernel": f"{r['kernel']}_{i}"} for i in range(4) for r in census]
    found = directives.build(tables(SM80, many), {})
    assert len(found) == directives.PER_RULE  # one rule: at most two of it


def test_the_profile_file_keeps_the_evidence_and_the_summary_stays_small(tmp_path):
    census = rows("triton_fp8_dot_sm120a.sass") + rows("sm_120a.sass") * 2
    timed = [{"kernel": "_gemm_kernel", "calls": 2, "us": 70.0}]
    result = {
        **COMPUTE,
        "correct": True,
        "sass": {"status": "ok", "gpu": SM120, "kernels": census[: sass.MAX_KERNELS]},
        "kernels_candidate": timed,
        "compiler_stats": {"triton": [{"kernel": "_gemm_kernel", "registers": 96, "spills": 0}]},
    }
    out: dict[str, Any] = {"status": "ok", "ncu": report()}
    tools_mod.profile_file(tmp_path, tmp_path / "003_fp8_gemm.py", out, result)
    profile = out["profile"]
    assert profile["directives"][0].startswith("[tensor_rate] _gemm_kernel: ncu: compute")
    assert len(profile["directives"]) <= directives.MAX_DIRECTIVES
    assert profile["sass"][0].startswith("_gemm_kernel (sm_120a): QMMA.16832.F32.E4M3.E4M3 x4")
    assert len(profile["sass"]) == tools_mod.PROFILE_TOP
    stored = json.loads(Path(profile["file"]).read_text())
    assert stored["directives"][0]["evidence"]["issued"] == ["QMMA.16832.F32.E4M3.E4M3"]
    assert len(stored["sass"]["kernels"]) == sass.MAX_KERNELS
    assert stored["ncu"]["lines"][0]["where"] == "003_fp8_gemm.py:39"
    assert len(json.dumps(profile)) < 4096  # the summary stays small (#186)


def test_evaluate_candidate_returns_the_directives_and_keeps_the_census_out_of_records(
    tmp_path, monkeypatch
):
    from test_workers import _server, _target

    from kernel_agent.workspace import RunDir, read_jsonl

    run = RunDir.create(tmp_path, "org/m")
    tdir = _target(run)
    census = {"status": "ok", "gpu": SM120, "kernels": rows("triton_fp8_dot_sm120a.sass")}

    def evaluated(snap: Path, kwargs: dict[str, Any]) -> dict[str, Any]:
        out = {"status": "ok", "correct": True, "speedup": 1.4, "cases": [], **COMPUTE}
        if kwargs.get("profile"):
            out["sass"] = census
            out["kernels_candidate"] = [{"kernel": "_gemm_kernel", "calls": 1, "us": 67.0}]
        return out

    call, _ = _server(run, monkeypatch, evaluated)
    (tdir / "candidates" / "v1.py").write_text("def build(r):\n    return r\n")
    args = {"target_id": "t", "candidate": "candidates/v1.py", "hypothesis": "tl.dot"}
    out = call("evaluate_candidate", **args, profile=True)
    (text,) = out["profile"]["directives"]
    assert text.startswith("[tensor_rate] _gemm_kernel: compute-bound at 21 % of SOL")
    assert "sass" not in out and out["profile"]["sass"][0].startswith("_gemm_kernel")
    (record,) = read_jsonl(run.results_file("t"))
    assert "sass" not in record and record["speedup"] == 1.4  # the census: the file only


def test_an_unavailable_census_is_summarised_with_its_reason(tmp_path):
    result = {"sass": {"status": "unavailable", "reason": "cuobjdump not found"}}
    out: dict[str, Any] = {}
    tools_mod.profile_file(tmp_path, tmp_path / "001_x.py", out, result)
    assert out["profile"]["sass"] == {"status": "unavailable", "reason": "cuobjdump not found"}
    assert "directives" not in out["profile"]


def test_texts_name_their_rule():
    found = [{"rule": "ncu_rule", "text": "Nsight X"}]
    assert directives.texts(found) == ["[ncu_rule] Nsight X"]
