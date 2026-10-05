import shutil
import struct

import pytest
from synthetic_run import BASELINE_MS, GREEDY, SCRIPTS, SINGLES, TRANSFORMS, make_run

from kernel_agent import charts, dashboard, ledger
from kernel_agent.agent import prompts
from kernel_agent.cli import main
from kernel_agent.report import write_report
from kernel_agent.status import render
from kernel_agent.workspace import RunDir, append_jsonl, read_jsonl, write_json


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return make_run(tmp_path_factory.mktemp("runs"))


def _png_size(path):
    data = path.read_bytes()[:24]
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def test_detect_backend():
    for backend in ("cuda", "triton", "cute", "tilelang", "nvrtc"):
        source = (prompts.EXAMPLES_DIR / f"{backend}_rmsnorm.py").read_text()
        assert ledger.detect_backend(source) == backend
    assert ledger.detect_backend("import torch\n\ndef build(r):\n    return r\n") == "torch"
    both = "import triton\nfrom torch.utils.cpp_extension import load_inline\n"
    assert ledger.detect_backend(both) == "cuda+triton"


def test_classify():
    def ok(speedup, spread=0.0):
        cases = [{"timing_spread": spread, "calls_per_run": 1, "ref_ms": 1.0}]
        return {"status": "ok", "correct": True, "speedup": speedup, "cases": cases}

    assert ledger.classify(ok(1.05), 1.0) == "keep"
    assert ledger.classify(ok(1.005), 1.0) == "discard"  # inside the 1 % margin
    assert ledger.classify(ok(1.05, spread=0.03), 1.0) == "discard"  # inside 2 x spread
    assert ledger.classify(ok(1.9), 2.0) == "discard"
    assert ledger.classify({"status": "incorrect", "correct": False}, 1.0) == "incorrect"
    assert ledger.classify({"status": "build_error"}, 1.0) == "build_error"
    assert ledger.classify({"status": "harness_error"}, 1.0) == "crash"
    assert ledger.classify({"status": "timeout"}, 1.0) == "timeout"
    e2e = {"status": "ok", "passed": True, "speedup": 1.2, "times_ms": [99, 100, 101]}
    assert ledger.classify(e2e, 1.0, e2e=True) == "keep"
    assert ledger.classify({**e2e, "passed": False}, 1.0, e2e=True) == "incorrect"
    assert ledger.classify({"status": "patch_error"}, 1.0, e2e=True) == "build_error"
    assert ledger.item_label("rms=/x/targets/rms/history/003_cuda_v2_0a1b2c3d.py") == "rms"
    assert ledger.item_label("/x/transforms/history/002_static_cache_0a1b2c3d.py") == "static_cache"


def test_ledger_rows(run):
    header = run.ledger.read_text().splitlines()[0].split("\t")
    assert tuple(header) == ledger.COLUMNS
    rows = ledger.rows(run)
    n_kernel = sum(len(s) for s in SCRIPTS.values())
    assert len(rows) == n_kernel + len(TRANSFORMS) + len(SINGLES) + len(GREEDY)
    assert [r["exp"] for r in rows] == list(range(1, len(rows) + 1))
    assert {r["status"] for r in rows} <= set(ledger.STATUSES)
    assert [r["time"] for r in rows] == sorted(r["time"] for r in rows)

    attn = [r for r in rows if r["target"] == "attn"]
    assert len(attn) == len(SCRIPTS["attn"])
    assert [r["status"] for r in attn[:5]] == ["keep", "build_error", "keep", "incorrect", "keep"]
    assert attn[0]["backend"] == "triton" and attn[-1]["backend"] == "cuda"
    assert attn[0]["hypothesis"].startswith("fuse q/k RMSNorm")
    assert ledger.best_kept(attn) == max(r["speedup"] for r in attn if r["status"] == "keep")
    kept = [r["speedup"] for r in attn if r["status"] == "keep"]
    assert kept == sorted(kept)  # every keep beats the previous best

    e2e = [r for r in rows if r["target"] == ledger.E2E]
    assert e2e[0]["status"] == "keep" and e2e[0]["backend"] == "transform"
    assert e2e[0]["ref_ms"] == BASELINE_MS
    assert e2e[0]["est_saved_ms"] == pytest.approx(BASELINE_MS - 1389.0)
    integrate = [r for r in e2e if r["backend"] == "integrate"]
    assert [r["status"] for r in integrate[-5:]] == [
        "discard",
        "keep",
        "incorrect",
        "keep",
        "discard",
    ]

    record = read_jsonl(run.target("attn") / "results.jsonl")[0]
    assert record["ledger_status"] == "keep" and record["exp"] == 1
    assert record["hypothesis"] == attn[0]["hypothesis"] and record["backend"] == "triton"
    assert read_jsonl(run.transforms_dir / "results.jsonl")[0]["ledger_status"] == "keep"


def test_events(run):
    events = ledger.events(run)
    kinds = {e["event"] for e in events}
    assert {"phase_start", "phase_done", "agent_start", "agent_done", "evaluation"} <= kinds
    assert sum(e["event"] == "evaluation" for e in events) == len(ledger.rows(run))
    spans = ledger.phase_spans(run)
    assert [p for p, _, _ in spans][:4] == ["analyze", "plan", "capture", "kernels"]
    assert all(end is not None and end >= start for _, start, end in spans)


def test_backfill_matches_ledger(run, tmp_path):
    copy = RunDir(tmp_path / "copy")
    shutil.copytree(run.root, copy.root)
    copy.ledger.unlink()
    for path in [*copy.targets_dir.glob("*/results.jsonl"), copy.transforms_dir / "results.jsonl"]:
        records = read_jsonl(path)
        path.unlink()
        for rec in records:  # an older run: no ledger fields, HH:MM:SS times
            for key in ("ledger_status", "exp", "hypothesis", "backend", "parent"):
                rec.pop(key, None)
            append_jsonl(path, rec)
    original = [r for r in ledger.rows(run) if r["backend"] != "integrate"]
    rebuilt = ledger.rows(copy)
    assert len(rebuilt) == len(original)
    key = lambda r: (r["target"], r["time"])  # noqa: E731
    for a, b in zip(sorted(original, key=key), sorted(rebuilt, key=key), strict=True):
        assert (a["target"], a["status"], a["speedup"], a["backend"]) == (
            b["target"],
            b["status"],
            b["speedup"],
            b["backend"],
        )


def test_summary_and_status(run):
    s = ledger.summary(run)
    assert s["phase"] == "report" and not s["phase_running"] and s["phase_failed"] is None
    assert s["baseline_ms"] == BASELINE_MS
    assert s["final"]["median_ms"] == 889.0
    assert s["cost_usd"] == 20.62
    attn = next(t for t in s["targets"] if t["id"] == "attn")
    assert attn["evals"] == 16 and attn["keeps"] == 6 and attn["failures"] == 5
    assert attn["last_hypothesis"] == SCRIPTS["attn"][-1][2]

    text = render(run, width=160)
    assert "Qwen/Qwen3-0.6B (llm)" in text and "phase: report" in text
    assert "baseline 1,532.4 ms" in text and "compiled 1,104.6 ms (1.39x)" in text
    assert "measured 889.0 ms (1.72x vs eager, 1.24x vs compiled, integrated)" in text
    assert "cost $20.62" in text
    for target in SCRIPTS:
        assert f"\n{target} " in text
    assert "last 10 evaluations" in text
    table = [line for line in text.splitlines() if not line.startswith(("run:", "dashboard:"))]
    assert all(len(line) <= 160 for line in table)


def test_status_cli_and_failed_phase(run, tmp_path, capsys):
    assert main(["status", str(run.root)]) == 0
    assert "last 10 evaluations" in capsys.readouterr().out

    fresh = RunDir.create(tmp_path, "org/m")
    write_json(fresh.run_json, {"card": {"repo_id": "org/m", "modality": "tts"}, "phases": {}})
    ledger.event(fresh, "phase_start", phase="analyze")
    assert "phase: analyze (running)" in render(fresh)
    assert "no evaluations yet" in render(fresh)
    ledger.event(fresh, "phase_failed", phase="analyze", error="SystemExit")
    assert "phase: analyze (failed)" in render(fresh)
    with pytest.raises(SystemExit):
        main(["status", str(tmp_path)])


def test_without_matplotlib(run, tmp_path, monkeypatch):
    copy = RunDir(tmp_path / "copy")
    shutil.copytree(run.root, copy.root)
    for leftover in [*copy.root.rglob("*.png"), copy.dashboard]:
        leftover.unlink(missing_ok=True)
    monkeypatch.setattr(charts, "available", lambda: False)
    assert charts.write_charts(copy) == []
    report = write_report(copy).read_text()
    assert "![" not in report and "results.tsv" in report
    page = copy.dashboard.read_text()
    assert "data:image/png" not in page and "Qwen/Qwen3-0.6B" in page and "keep" in page


def test_integration_steps(run):
    from kernel_agent.workspace import read_json

    steps = charts.integration_steps(read_json(run.root / "integration.json"))
    assert [s["kind"] for s in steps] == [
        "total",
        "keep",
        "nogain",
        "keep",
        "fail",
        "keep",
        "nogain",
        "total",
    ]
    assert steps[1]["label"] == "cuda_graph_decode" and steps[1]["source"] == "transform"
    assert steps[3]["label"] == "attn" and steps[3]["source"] == "kernel"
    assert steps[0]["ms"] == BASELINE_MS and steps[-1]["ms"] == 889.0
    shares = dict((t, s) for t, _, s in charts.target_shares(run))
    assert shares["attn"] == pytest.approx(856.0 / 2086.0)


def test_charts_report_dashboard(run):
    pytest.importorskip("matplotlib")
    paths = charts.write_charts(run)
    expected = [run.target(t) / "progress.png" for t in SCRIPTS]
    expected += [run.root / "progress.png", run.root / "amdahl.png", run.root / "integration.png"]
    assert sorted(paths) == sorted(expected)
    for path in expected:
        width, height = _png_size(path)
        assert width > 1000 and height > 400  # 150 dpi
    assert not list(run.root.rglob(".*.tmp.png"))

    report = write_report(run).read_text()
    for image in ("progress.png", "amdahl.png", "integration.png", "targets/attn/progress.png"):
        assert f"]({image})" in report
    page = run.dashboard.read_text()
    assert page.count("data:image/png;base64,") == len(expected)
    assert "prefers-color-scheme: dark" in page and "<table>" in page


def test_refresh_never_raises(run, tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("chart bug")

    monkeypatch.setattr(charts, "write_charts", boom)
    dashboard.refresh(run)  # logs once, does not raise
