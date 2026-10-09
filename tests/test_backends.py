"""Backend policy by target class, backend classification from a candidate's source, the
per-backend outcome tables of report.md / status and the library's backend record
(``kernel_agent/backends.py``, issue #133). CPU only."""

import json

import pytest

from kernel_agent import backends, ledger, library, report, status
from kernel_agent.agent import prompts
from kernel_agent.budget import PRIOR_HYPOTHESIS
from kernel_agent.workspace import RunDir, write_json

TRITON = "import torch\nimport triton\n\n@triton.jit\ndef k(x):\n    pass\n"
CUDA = "from torch.utils.cpp_extension import load_inline\n\nmod = load_inline(name='x')\n"
# CUDA C++ that imports tilelang only for the CUTLASS headers it ships (RESEARCH-TRITON §4.1)
CUDA_WITH_TILELANG_HEADERS = (
    "import os\nimport tilelang\nfrom torch.utils.cpp_extension import load_inline\n"
    "inc = os.path.join(os.path.dirname(tilelang.__file__), '3rdparty', 'cutlass')\n"
    "mod = load_inline(name='x', extra_include_paths=[inc])\n"
)
CUTE = "import cutlass\nimport cutlass.cute as cute\n\n@cute.kernel\ndef k(a):\n    pass\n"
TILELANG = "import tilelang\nimport tilelang.language as T\n\n@T.prim_func\ndef k(A):\n    pass\n"


def test_classify_reads_what_the_source_runs():
    assert backends.classify(TRITON) == "triton"
    assert backends.classify(CUDA) == "cuda"
    assert backends.classify(CUTE) == "cute"
    assert backends.classify(TILELANG) == "tilelang"
    assert backends.classify(TRITON + CUDA) == "cuda+triton"
    # the imports say tilelang+cuda, the code runs CUDA C++ (RESEARCH-TRITON §4.1)
    assert ledger.detect_backend(CUDA_WITH_TILELANG_HEADERS) == "tilelang+cuda"
    assert backends.classify(CUDA_WITH_TILELANG_HEADERS) == "cuda"
    # no kernel marker: the imports decide (the ledger's column), torch without any
    assert backends.classify("import triton\n") == "triton"
    assert backends.classify("import torch\n\ndef build(r):\n    return r\n") == "torch"


def _spec(cls, signature, precision=None, count=10, **extra):
    spec = {"module_class": cls, "capture": {"cases": [{"signature": signature, "count": count}]}}
    if precision:
        spec["precision"] = precision
    return {**spec, **extra}


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (_spec("Qwen3MLP", "a0[4, 176, 1024]:bfloat16", "fp8_w8a8"), "fp8_gemm"),
        (_spec("Qwen3MLP", "a0[4, 176, 1024]:bfloat16"), "bf16_gemm"),
        (_spec("LlamaMLP", "a0[16, 1, 2048]:bfloat16", "fp8_weights"), "small_m_gemm"),
        (_spec("Linear", "a0[80, 1024]:bfloat16", "fp8_w8a8"), "small_m_gemm"),
        (_spec("GPT2Attention", "a0[32, 11, 1024]:bfloat16"), "short_attention"),
        (_spec("GPT2Attention", "a0[1, 512, 1024]:bfloat16"), "other"),
        (_spec("LlamaDecoderLayer", "a0[1, 128, 4096]:bfloat16"), "decoder_layer"),
        (_spec("WhisperEncoderLayer", "a0[1, 1500, 512]:float32"), "decoder_layer"),
        (_spec("Qwen3RMSNorm", "a0[1, 1, 1024]:bfloat16"), "elementwise"),
        (_spec("Conv1d", "a0[16, 64, 240]:float32"), "conv"),
        (_spec("AutoencoderKL", "a0[1, 4, 64, 64]:float32"), "conv"),
        (_spec("Mystery", "a0[3]:float32"), "other"),
        ({"kind": "region", "parent_class": "LlamaDecoderLayer"}, "decoder_layer"),
    ],
)
def test_target_class_by_op_shape_and_precision(spec, expected):
    assert backends.target_class(spec) == expected


def test_dominant_case_ignores_correctness_only_calls():
    spec = _spec("Qwen3MLP", "a0[8, 1024]:bfloat16", "fp8_w8a8", count=0)
    spec["capture"]["cases"].append({"signature": "a0[2, 352, 1024]:bfloat16", "count": 40})
    assert backends.dominant_shape(spec) == (2, 352, 1024)
    assert backends.rows_of(backends.dominant_shape(spec)) == 704
    assert backends.target_class(spec) == "fp8_gemm"


def test_policy_text_and_planner_prompt():
    text = backends.policy_text(["cuda", "triton", "cute"])
    assert "| target class | first backend | why (evidence) | second |" in text
    assert "`QMMA.SF`" in text and "208 TFLOP/s" in text and "MmaMXF8Op" in text
    assert "never a plain `QMMA.F32` kernel" in text and "`tl.dot_scaled`" in text
    assert "one C++ launcher" in text and "warp_specialize" in text
    for c in backends.POLICY:
        if c.id not in ("other", "fp4_gemm"):  # W4A4: opt-in, in runs that allow it (#233)
            assert c.label in text
    # model-agnostic: classes, shapes, bounds and precisions, never a model's module names
    for name in ("VoxCPM", "LocDiT", "MiniCPM", "LocEnc"):
        assert name not in text
    plan = prompts.planner_prompt(
        {"repo_id": "org/model", "modality": "llm"},
        {"median_ms": 1.0},
        "# Profile",
        ["cuda", "triton", "cute"],
        3,
        "py",
        "tc",
        backend_record="**Backend track record on sm_120** RECORD",
    )
    assert "Backend policy by target class" in plan and "RECORD" in plan
    assert "launch-bound micro-ops, Triton/TileLang for" not in plan  # the old one-liner


def test_engineer_note_names_the_class_and_the_fp8_rules():
    layer = _spec("LlamaDecoderLayer", "a0[2, 176, 1024]:bfloat16", "fp8_w8a8")
    note = backends.engineer_note(layer, ["triton", "cuda"])
    assert "Fused decoder layer" in note and "The GEMMs inside (M = 352)" in note
    assert "never a plain `QMMA.F32` kernel" in note and "`tl.dot_scaled`" in note
    gemm = _spec("Qwen3MLP", "a0[4, 176, 1024]:bfloat16", "fp8_w8a8")
    note = backends.engineer_note(gemm, ["triton", "cute"])
    assert "Compute-bound FP8 GEMM" in note
    assert "would start with `cute`" in note  # the plan disagrees with the policy
    assert "would start with" not in backends.engineer_note(gemm, ["cute", "triton"])
    # only among the plan's backends: without cute, Triton first is what the policy says
    assert "would start with" not in backends.engineer_note(gemm, ["triton", "cuda"])
    # the engineer prompt carries it
    target = {"id": "mlp", "module_class": "Qwen3MLP", "precision": "fp8_w8a8"}
    text = prompts.engineer_prompt(target, gemm["capture"], ["cute"], "py", "tc", 4, None)
    assert "Target class: **Compute-bound FP8 GEMM" in text


# ------------------------------------------------------------------ outcomes of a run


def _ok(speedup):
    return {"status": "ok", "correct": True, "speedup": speedup, "cases": []}


def _eval(run, target, name, source, result, *, hypothesis="h", status=None):
    history = run.history_dir(target)
    history.mkdir(parents=True, exist_ok=True)
    (history / name).write_text(source)
    return ledger.record_kernel(
        run, target, result, snapshot=name, hypothesis=hypothesis, source=source, status=status
    )


@pytest.fixture
def run(tmp_path):
    run = RunDir(tmp_path / "run")
    write_json(run.run_json, {"card": {"repo_id": "org/model"}})
    specs = {
        "mlp": _spec(
            "Qwen3MLP", "a0[4, 176, 1024]:bfloat16", "fp8_w8a8", backends=["triton", "cuda"]
        ),
        "norm": _spec("Qwen3RMSNorm", "a0[1, 1, 1024]:bfloat16", backends=["cuda", "cute"]),
    }
    for target, spec in specs.items():
        write_json(run.target(target) / "spec.json", {"id": target, **spec})
    _eval(run, "mlp", "001_t1.py", TRITON, _ok(1.2))
    _eval(run, "mlp", "002_c1.py", CUDA_WITH_TILELANG_HEADERS, _ok(1.9))
    _eval(run, "mlp", "003_c2.py", CUDA, {"status": "build_error", "correct": False})
    _eval(
        run, "mlp", "004_q.py", CUTE, {"status": "ok", "correct": False}, status=ledger.QUICK_FAIL
    )
    _eval(run, "mlp", "005_prior.py", CUTE, _ok(3.0), hypothesis=f"{PRIOR_HYPOTHESIS}x/y")
    _eval(run, "norm", "001_cute.py", CUTE, _ok(2.5))
    _eval(run, "norm", "002_cuda.py", CUDA, _ok(2.0))
    # the integration re-evaluated the cute snapshot: 1.6x now
    ledger.append(
        run,
        {
            "target": "norm",
            "snapshot": "001_cute.py",
            "status": ledger.REEVALUATED,
            "correct": True,
            "speedup": 1.6,
            "backend": "cute",
        },
    )
    return run


def test_outcomes_per_target_and_backend(run):
    found = {(o.target, o.backend): o for o in backends.outcomes(run)}
    assert set(found) == {
        ("mlp", "triton"),
        ("mlp", "cuda"),
        ("mlp", "cute"),
        ("norm", "cute"),
        ("norm", "cuda"),
    }
    cuda = found[("mlp", "cuda")]
    assert (cuda.evaluations, cuda.correct, cuda.best) == (2, 1, 1.9)
    cute = found[("mlp", "cute")]  # only the quick check: the library prior is left out
    assert (cute.evaluations, cute.quick_fail, cute.best) == (0, 1, None)
    assert found[("norm", "cute")].best == 1.6  # the re-evaluation replaces 2.5
    assert backends.winners(found.values()) == {"mlp": "cuda", "norm": "cuda"}
    table = {a["backend"]: a for a in backends.by_backend(list(found.values()))}
    assert table["cuda"]["won"] == 2 and table["cuda"]["targets"] == 2
    assert table["cuda"]["best"] == 2.0 and table["cuda"]["best_target"] == "norm"
    assert table["cute"]["quick_fail"] == 1 and table["cute"]["won"] == 0


def test_report_and_status_tables(run):
    lines = backends.report_lines(run)
    text = "\n".join(lines)
    assert "## Backends" in text
    assert "| cuda | 2 (2) | 3 | 2 (67 %) | " in text
    assert "| `mlp` | Compute-bound FP8 GEMM (W8A8, M ≳ 128 rows) | triton, cuda |" in text
    assert "cuda 1/2 1.90x" in text
    terminal = "\n".join(backends.status_lines(run))
    assert "backends (by source)" in terminal and "cuda" in terminal and "2 won" in terminal
    assert "backends (by source)" in status.render(run, width=160)
    assert backends.report_lines(RunDir(run.root.parent / "empty")) == []


def test_report_md_has_the_backend_section(run, monkeypatch):
    monkeypatch.setattr(report, "refresh", lambda r: None)
    write_json(run.run_json, {"card": {"repo_id": "org/model", "modality": "llm"}})
    text = report.write_report(run).read_text()
    assert "## Backends" in text and "tried (by source)" in text


def test_library_record_and_track_record(run, tmp_path, monkeypatch):
    monkeypatch.setenv(library.ENV, str(tmp_path / "lib"))
    path = backends.record_run(run, "sm_120")
    assert path == tmp_path / "lib" / "sm_120" / "backends.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    mlp = next(r for r in records if r["target"] == "mlp")
    assert mlp["class"] == "fp8_gemm" and mlp["winner"] == "cuda" and mlp["rows"] == 704
    assert mlp["planned"] == ["triton", "cuda"] and mlp["backends"]["cuda"]["best"] == 1.9
    backends.record_run(run, "sm_120")  # recording the run again replaces its lines
    assert len(path.read_text().splitlines()) == len(records)
    assert library.entries("sm_120") == []  # not mistaken for a library entry
    record = backends.track_record("sm_120")
    first = record[0]
    assert (first["class"], first["backend"], first["won"]) == ("fp8_gemm", "cuda", 1)
    note = backends.track_record_note("sm_120")
    assert "Backend track record on sm_120" in note and "| Compute-bound FP8 GEMM" in note
    assert backends.track_record_note("sm_89") == "" and backends.track_record_note(None) == ""


def test_no_track_record_under_emulation(run, tmp_path, monkeypatch):
    """An emulated GPU's outcomes (#252) are not that architecture's: nothing is recorded."""
    monkeypatch.setenv(library.ENV, str(tmp_path / "lib"))
    monkeypatch.setenv("KERNEL_AGENT_EMULATE_ARCH", "sm_86")
    assert backends.record_run(run, "sm_86") is None
    assert not backends.record_path("sm_86").exists() and backends.track_record("sm_86") == []
