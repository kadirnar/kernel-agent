import shutil
import struct

import pytest
from synthetic_run import BASELINE_MS, GREEDY, SCRIPTS, SINGLES, TRANSFORMS, make_run

from kernel_agent import charts, dashboard, experiments, ledger
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
    expected += [run.root / "progress.png", run.root / "timeline.png"]  # experiments, hours
    expected += [run.root / "amdahl.png", run.root / "integration.png"]
    assert sorted(paths) == sorted(expected)
    for path in expected:
        width, height = _png_size(path)
        assert width > 1000 and height > 400  # 150 dpi
    assert not list(run.root.rglob(".*.tmp.png"))

    report = write_report(run).read_text()
    images = ("progress.png", "timeline.png", "amdahl.png", "integration.png")
    for image in (*images, "targets/attn/progress.png"):
        assert f"]({image})" in report
    # the experiments: the headline, the kept improvements by gain, then progress.png
    s = experiments.summary(run)
    section = report.split("## Experiments\n", 1)[1].split("\n## ", 1)[0]
    assert section.startswith(f"\n**{experiments.headline(s)}** (+{len(SINGLES)} integration")
    top = experiments.kept(run)[0]
    assert f"| {top.exp} | `{top.lineage}` |" in section
    assert section.index(f"| {top.exp} |") < section.index("](progress.png)")
    assert section.index("](progress.png)") < section.index("](timeline.png)")
    page = run.dashboard.read_text()
    assert page.count("data:image/png;base64,") == len(expected)
    assert 'alt="progress over experiment number"' in page
    assert 'alt="end-to-end latency over wall-clock time"' in page
    assert "prefers-color-scheme: dark" in page and "<table>" in page


def _drawn(monkeypatch):
    """``{chart: its axes}`` of the charts drawn from now on (``progress.png``: the model
    panel; the kernel panel is ``.figure.axes[1]``)."""
    drawn = {}
    render = charts._render

    def keep(path, size, draw):
        def capture(fig, ax):
            draw(fig, ax)
            target = path.parent.parent.name == "targets"
            drawn[f"{path.parent.name}/{path.name}" if target else path.name] = ax

        return render(path, size, capture)

    monkeypatch.setattr(charts, "_render", keep)
    return drawn


def _lines(ax, color, lw):
    return [line for line in ax.get_lines() if line.get_color() == color and line.get_lw() == lw]


def test_progress_is_drawn_as_lines_without_point_markers(run, monkeypatch):
    """The progress charts (and the integration waterfall) show the improvement as lines: the
    best so far prominent, every result a thin line, failures as ticks; no scattered points,
    and the legends show line samples (the failure tick's is a short vertical line).
    ``progress.png`` is drawn over experiment number from 0 (the baseline), measured values
    only: no projection (#223); the wall-clock chart is ``timeline.png``."""
    pytest.importorskip("matplotlib")
    from matplotlib.collections import LineCollection, PathCollection

    drawn = _drawn(monkeypatch)
    assert charts.target_progress(run, "attn") and charts.run_progress(run)
    assert charts.run_timeline(run) and charts.integration(run)
    assert set(drawn) == {"attn/progress.png", "progress.png", "timeline.png", "integration.png"}
    kernels = drawn["progress.png"].figure.axes[1]
    for name, ax in [*drawn.items(), ("progress.png kernels", kernels)]:
        assert not [c for c in ax.collections if isinstance(c, PathCollection)], name
        assert all(line.get_marker() in ("None", "", None) for line in ax.get_lines()), name
        if ax.get_legend() is None:  # progress.png: the legend is under its kernel panel
            continue
        handles = ax.get_legend().legend_handles
        assert all(
            h.get_marker() in ("None", "", None, "|") for h in handles if hasattr(h, "get_marker")
        )
    target, progress = drawn["attn/progress.png"], drawn["progress.png"]
    labels = [t.get_text() for t in target.get_legend().get_texts()]
    assert labels[:3] == ["running best", "each evaluation", "failed"]
    assert any(isinstance(c, LineCollection) for c in target.collections)  # the failure ticks
    best = max(target.get_lines(), key=lambda line: line.get_linewidth())
    assert best.get_linewidth() == charts.BEST_LW and best.get_color() == charts.KEEP_COLOR
    labels = [t.get_text() for t in drawn["timeline.png"].get_legend().get_texts()]
    assert labels[:3] == ["projected (kernels)", "measured, best so far", "each measured run"]

    labels = [t.get_text() for t in kernels.get_legend().get_texts()]
    assert labels[:5] == ["running best", "each experiment", "failed", "baseline", "torch.compile"]
    assert labels[5:] == [t for t, _, _ in charts.target_shares(run)]  # the target lines
    (best,) = _lines(progress, charts.KEEP_COLOR, charts.BEST_LW)
    xs = list(best.get_xdata())
    exps = {r["exp"] for r in ledger.rows(run)}
    assert xs[0] == 0 and xs == sorted(xs) and set(xs[1:]) <= exps  # exps, from the baseline
    assert list(best.get_ydata()) == sorted(best.get_ydata(), reverse=True)  # never up
    assert best.get_drawstyle() == "steps-post"
    assert not [line for line in progress.get_lines() if line.get_color() == charts.PROJECTED_COLOR]
    assert progress.get_xlabel() == "" and kernels.get_xlabel().startswith("experiment #")
    assert any(isinstance(c, LineCollection) for c in progress.collections)  # e2e failures
    assert any(isinstance(c, LineCollection) for c in kernels.collections)  # kernel failures


def _copy(run, tmp_path):
    copy = RunDir(tmp_path / "copy")
    shutil.copytree(run.root, copy.root)
    return copy


def _after(run, minutes=1.0):
    """A time ``minutes`` after the synthetic run's last row (a row recorded then is last)."""
    return ledger.epoch(ledger.rows(run)[-1]["time"]) + 60 * minutes


def _e2e(ms):
    return {
        "status": "ok",
        "passed": True,
        "median_ms": ms,
        "times_ms": [ms - 1, ms, ms + 1],
        "baseline_ms": BASELINE_MS,
        "speedup": BASELINE_MS / ms,
    }


def test_progress_leaves_out_probes_and_heads_with_the_experiments(run, tmp_path, monkeypatch):
    """The thin line of ``progress.png`` runs through every correct end-to-end and
    integration experiment, never through the integration's probes of one item alone (one at
    2x the baseline here); the subtitle counts them beside the experiments, and the title is
    experiments.summary's "N experiments, K kept improvements" (#223)."""
    pytest.importorskip("matplotlib")
    copy = _copy(run, tmp_path)
    slow = 2 * BASELINE_MS
    probe = ledger.record_e2e(
        copy,
        _e2e(slow),
        backend=ledger.INTEGRATE,
        snapshot="rope",
        hypothesis="integration: rope alone",
        title="probe rope #004 alone",
        when=_after(run),
    )
    assert probe["kind"] == ledger.PROBE and probe["status"] == ledger.DISCARD
    drawn = _drawn(monkeypatch)
    assert charts.run_progress(copy) == copy.root / "progress.png"
    ax = drawn["progress.png"]
    (each,) = _lines(ax, charts.DISCARD_COLOR, charts.EACH_LW)
    e2e = [
        r
        for r in ledger.rows(copy)
        if r["target"] == ledger.E2E and r["status"] in ("keep", "discard")
        if ledger.kind(r) != ledger.PROBE
    ]
    assert list(each.get_xdata()) == [r["exp"] for r in e2e]
    assert slow not in each.get_ydata() and max(each.get_ydata()) < BASELINE_MS

    s = experiments.summary(copy)
    measured = experiments.experiments(run)  # the probe is no experiment of N: N is the same
    assert s["probes"] == len(SINGLES) + 1
    assert s["experiments"] == len(measured) - len(SINGLES)
    texts = [t.get_text() for t in ax.texts]
    title = f"{s['repo_id']}: {s['experiments']} experiments, {s['kept']} kept improvements"
    assert experiments.headline(s) == title and title in texts
    subtitle = next(t for t in texts if t.startswith(f"{BASELINE_MS:,.1f} → "))
    first, second = subtitle.split("\n")
    assert first.startswith(f"{BASELINE_MS:,.1f} → 889.0 ms measured ({BASELINE_MS / 889:.2f}×)")
    assert "RTX 5070 Ti" in first and "NVIDIA" not in first
    kinds = s["by_kind"]
    assert first.endswith(
        f"{kinds['kernel']} kernel / {kinds['e2e']} end-to-end / {kinds['integration']} "
        f"integration (+{len(SINGLES) + 1} probes)"
    )
    by = s["by_status"]
    assert second.startswith(f"{by['keep']} kept  ·  {by['discard']} discarded  ·  ")
    assert second.endswith("$20.62")


def test_progress_scale_switches_to_log_at_three_times():
    """``progress.png``'s metric axis: linear from the best − 15 % of the gain to the baseline
    + 15 % (autoresearch's frame), log from baseline / best = 3 on (the same frame in log
    space); torch.compile's line stays inside."""
    assert charts.progress_scale(300.0, 100.0)[0] and not charts.progress_scale(299.0, 100.0)[0]
    _, bottom, top = charts.progress_scale(300.0, 100.0)
    assert (bottom, top) == pytest.approx((100 / 3**0.15, 300 * 3**0.15))
    assert charts.progress_scale(200.0, 100.0) == pytest.approx((False, 85.0, 215.0))
    assert charts.progress_scale(200.0, 100.0, [60.0])[1] == pytest.approx(60 - 0.15 * 140)
    assert charts.progress_scale(100.0, 100.0)[1:] == pytest.approx((99.25, 100.75))  # no gain


def test_progress_draws_a_log_axis_from_three_times(run, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    drawn = _drawn(monkeypatch)
    charts.run_progress(run)
    assert drawn["progress.png"].get_yscale() == "linear"  # 1.72x
    copy = _copy(run, tmp_path)
    fast = BASELINE_MS / 4
    kept = ledger.record_e2e(
        copy,
        _e2e(fast),
        backend="transform",
        snapshot="001_graph_all_0a1b2c3d.py",
        hypothesis="one CUDA graph around the whole generate call",
        title="CUDA graph of generate",
        when=_after(run),
    )
    assert kept["status"] == ledger.KEEP
    charts.run_progress(copy)
    ax = drawn["progress.png"]
    assert ax.get_yscale() == "log"
    assert ax.get_ylim()[0] < fast < BASELINE_MS < ax.get_ylim()[1]
    (best,) = _lines(ax, charts.KEEP_COLOR, charts.BEST_LW)
    assert list(best.get_xdata())[-2:] == [kept["exp"], kept["exp"]]
    texts = [t.get_text() for t in ax.texts]
    assert any(t.startswith(f"#{kept['exp']} {fast:,.1f} ms 4.00×  CUDA graph") for t in texts)


def test_progress_labels_every_kept_step_without_overlaps(run, monkeypatch):
    """Every kept end-to-end step is labelled (its number, value, ratio and title) where no
    other label, result or line is; no two labels of the chart overlap."""
    pytest.importorskip("matplotlib")
    drawn = _drawn(monkeypatch)
    charts.run_progress(run)
    ax = drawn["progress.png"]
    kernels = ax.figure.axes[1]
    kept = [e for e in experiments.experiments(run) if e.lineage == "model" and e.status == "keep"]
    texts = [t.get_text() for t in ax.texts]
    for e in kept:
        title = ledger.cut(e.title, 40)
        assert f"#{e.exp} {e.value:,.1f} ms {BASELINE_MS / e.value:.2f}×  {title}" in texts
    best = min(e.value for e in kept)
    assert f"best {best:,.1f} ms ({BASELINE_MS / best:.2f}×)" in texts
    ends = [t.get_text() for t in kernels.texts]
    for target, _, ys in charts.kernel_lines(run, experiments.all_rows(run)):
        assert f"{target} {ys[-1]:.2f}×" in ends
    renderer = ax.figure.canvas.get_renderer()
    boxes = [
        (t.get_text(), t.get_window_extent(renderer))
        for panel in (ax, kernels)
        for t in panel.texts
        if t.get_visible() and t.get_text()
    ]
    assert len(boxes) > len(kept) + 4
    for i, (a, box) in enumerate(boxes):
        for b, other in boxes[i + 1 :]:
            assert not box.overlaps(other), (a, b)


def test_phases_are_bands_over_experiments(tmp_path, monkeypatch):
    """A row belongs to the phase whose span holds its time: improve started again (a resumed
    run) stays one band, not labels printed over each other, a later improve round is a band
    of its own, a phase without rows has none; re-integrations are dashed lines after their
    last row (#223)."""
    run = RunDir.create(tmp_path, "org/m")
    t0 = 1_800_000_000.0
    ok = {"status": "ok", "correct": True, "cases": []}

    def phase(event, name, minute):
        ledger.event(run, event, when=t0 + 60 * minute, phase=name)

    def evaluate(minute, speedup):
        snapshot = f"{minute:03d}_k_0a1b2c3d.py"
        ledger.record_kernel(
            run,
            "k",
            ok | {"speedup": speedup},
            snapshot=snapshot,
            hypothesis="h",
            when=t0 + 60 * minute,
        )

    phase("phase_start", "analyze", 0)
    phase("phase_done", "analyze", 5)
    phase("phase_start", "improve", 6)
    evaluate(7, 1.2)
    evaluate(8, 1.3)
    phase("phase_start", "improve", 20)  # resumed: no phase_done before it
    evaluate(21, 1.1)
    evaluate(22, 1.5)
    evaluate(31, 1.6)
    phase("phase_done", "improve", 40)
    rounds = [{"n": 1, "speedup": 1.0}, {"n": 2, "started": t0 + 60 * 30}]
    write_json(
        run.root / "improve.json",
        {"rounds": rounds, "integrations": [{"n": 1, "exp_before": 2, "exp_after": 4}]},
    )
    items = experiments.all_rows(run)
    assert [p for p, _, _ in ledger.phase_spans(run)] == ["analyze", "improve", "improve"]
    assert charts.exp_bands(run, items) == [("improve", 0.5, 4.5), ("improve round 2", 4.5, 5.5)]
    assert charts.reintegrations(run) == [4.5]
    ((_, xs, ys),) = charts.kernel_lines(run, items)
    assert (xs, ys) == ([1.0, 1.0, 2.0, 4.0, 5.0, 5.0], [1.0, 1.2, 1.3, 1.5, 1.6, 1.6])

    if not charts.available():
        return
    write_json(run.baseline_json, {"median_ms": 100.0})
    drawn = _drawn(monkeypatch)
    assert charts.run_progress(run)
    ax = drawn["progress.png"]
    texts = [t.get_text() for t in ax.texts]
    assert texts.count("improve") == 1 and texts.count("improve round 2") == 1
    assert "analyze" not in texts
    dashed = [line for line in ax.get_lines() if line.get_color() == charts.MUTED]
    assert [list(line.get_xdata()) for line in dashed] == [[4.5, 4.5]]  # the re-integration
    assert "no kernel experiments" not in [t.get_text() for t in ax.figure.axes[1].texts]


def test_progress_of_a_run_without_experiments_yet(tmp_path, monkeypatch):
    """Right after the baseline, before any ledger row, ``progress.png`` is drawn: the
    baseline, an empty kernel panel and the headline "0 experiments, 0 kept improvements"."""
    pytest.importorskip("matplotlib")
    run = RunDir.create(tmp_path, "org/m")
    write_json(run.baseline_json, {"median_ms": 100.0})
    drawn = _drawn(monkeypatch)
    assert charts.run_progress(run) == run.root / "progress.png"
    ax = drawn["progress.png"]
    texts = [t.get_text() for t in ax.texts]
    assert f"{run.root.name}: 0 experiments, 0 kept improvements" in texts
    assert "baseline 100.0 ms" in texts
    assert "no kernel experiments" in [t.get_text() for t in ax.figure.axes[1].texts]


def test_kernel_lines_follow_the_standing_best(tmp_path):
    """A target's line in ``progress.png`` is its standing best (the ledger's bar): it steps
    up at each keep and down when a re-evaluation finds its best slower (#223)."""
    run = RunDir.create(tmp_path, "org/m")
    ok = {"status": "ok", "correct": True, "cases": []}
    for snapshot, speedup, status in (
        ("001_a_0a1b2c3d.py", 1.5, None),
        ("002_b_0a1b2c3d.py", 2.0, None),
        ("003_c_0a1b2c3d.py", 1.2, None),
        ("002_b_0a1b2c3d.py", 1.6, ledger.REEVALUATED),
    ):
        ledger.record_kernel(
            run, "k", ok | {"speedup": speedup}, snapshot=snapshot, hypothesis="h", status=status
        )
    ((target, xs, ys),) = charts.kernel_lines(run, experiments.all_rows(run))
    assert target == "k" and xs == [1.0, 1.0, 2.0, 4.0, 4.0] and ys == [1.0, 1.5, 2.0, 1.6, 1.6]


def test_target_progress_labels_name_the_experiment(run, monkeypatch):
    """A target's kept candidates are labelled with their experiment number and title, and
    its header says which experiments its evaluations are (#223)."""
    pytest.importorskip("matplotlib")
    drawn = _drawn(monkeypatch)
    charts.target_progress(run, "attn")
    ax = drawn["attn/progress.png"]
    rows = ledger.measured(r for r in ledger.rows(run) if r["target"] == "attn")
    kept = {r["exp"]: r for r in rows if r["status"] == "keep"}
    texts = [t.get_text() for t in ax.texts]
    full = [t for t in texts if t.startswith("#")]
    assert full
    for text in full:
        r = kept[int(text[1:].split()[0])]
        assert text == f"#{r['exp']} {r['speedup']:.2f}×  {ledger.cut(ledger.title(r), 46)}"
    assert any(f"exp {rows[0]['exp']}–{rows[-1]['exp']}" in t for t in texts)


def test_refresh_never_raises(run, tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("chart bug")

    monkeypatch.setattr(charts, "write_charts", boom)
    dashboard.refresh(run)  # logs once, does not raise
