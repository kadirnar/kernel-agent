"""Prediction error of the integration (issue #226; region targets for #231): each accepted
item's predicted saving next to the measured A/B gain of the step that added it, in
``report.md`` (*Prediction error*), the ledger (``pred_saved_ms``) and the kernel library
(``<sm_arch>/predictions.jsonl``, ``lessons/estimates.md``, the planner's note).

CPU only: fixture ``integration.json`` files written here (the current format, one from
before #121 without ``summed_ms``, one from before #114 of a throughput run whose kernel
savings are per run), a region target with a fusion candidate, and a simulated integration
with a fake measurement layer (``test_projection_overlaps``) for the ledger and the library.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from test_integrate import BASE as SIM_BASE
from test_integrate import e2e as e2e_record
from test_integrate import kernel, label, make
from test_integrate import write as write_transform
from test_projection_overlaps import SAVES, overlap_worker

from kernel_agent import charts, dryrun, ledger, library, orchestrator, prediction, projection
from kernel_agent.report import write_report
from kernel_agent.workspace import RunDir, append_jsonl, read_json, write_json

BASE = 100.0  # ms per run (metric=latency)
K1 = "attn=/r/.truth/targets/attn/history/001_cuda_v1_aaaaaaaa.py"
T1 = "/r/.truth/transforms/history/002_graph_lm_bbbbbbbb.py"
T2 = "/r/.truth/transforms/history/003_fuse_norms_cccccccc.py"
K3 = "layer=/r/.truth/targets/layer/history/004_cuda_v4_dddddddd.py"
#: The accepted sets: the systems agent's combination of a kernel and a transform (measured
#: against the best single item, the kernel), + a transform, + a kernel whose estimate is
#: more than the set before it takes (not projectable).
SETS = [
    {"items": [K1, T1], "est_saved_ms": {K1: 20.0, T1: 10.0}, "measured_ms": 75.0},
    {
        "items": [K1, T1, T2],
        "est_saved_ms": {K1: 20.0, T1: 10.0, T2: 8.0},
        "measured_ms": 70.0,
        "from_ms": 75.5,
    },
    {
        "items": [K1, T1, T2, K3],
        "est_saved_ms": {K1: 20.0, T1: 10.0, T2: 8.0, K3: 90.0},
        "measured_ms": 60.0,
        "from_ms": 70.2,
    },
]


def _ab(a_items: list[str], a_ms: float, ci: list[float]) -> dict:
    return {"mode": "paired", "accepted": True, "a_items": a_items, "a_median_ms": a_ms, "ci95": ci}


HISTORY = [
    {"items": [K1], "passed": True, "ab": _ab([], 100.2, [0.17, 0.19])},  # alone
    {"items": [T1], "passed": True, "ab": _ab([], 100.1, [0.09, 0.11])},
    {"items": [T2], "passed": True, "ab": _ab([], 99.9, [0.07, 0.09])},
    {"items": [K1, T1], "passed": True, "ab": _ab([K1], 82.0, [0.07, 0.10])},  # the seed
    {"items": [K1, T1, T2], "passed": True, "ab": _ab([K1, T1], 75.5, [0.05, 0.09])},
    {"items": [K1, T1, T2, K3], "passed": True, "ab": _ab([K1, T1, T2], 70.2, [0.12, 0.17])},
]


def make_run(tmp_path, baseline: dict, projection_entries: list[dict], history=HISTORY) -> RunDir:
    """A run with its baseline, the specs and evaluation records of the kernels and the
    ``integration.json`` of ``projection_entries``."""
    run = RunDir(tmp_path / "org--model" / "20261009-000000")
    run.root.mkdir(parents=True)
    write_json(run.run_json, {"card": {"repo_id": "org/model", "modality": "llm"}})
    write_json(run.baseline_json, baseline)
    for item, timing in ((K1, ("memory", "graph", "cold")), (K3, ("compute", "eager", "warm"))):
        target, _, path = item.partition("=")
        write_json(run.target(target) / "spec.json", {"id": target, "module_class": "Block"})
        bound, context, l2 = timing
        record = {"snapshot": f"history/{path.rsplit('/', 1)[1]}", "correct": True}
        timed = {**record, "bound": bound, "context": context}
        append_jsonl(run.results_file(target), {**record, "speedup": 0.9})  # older records
        append_jsonl(run.results_file(target), timed)
        append_jsonl(run.results_file(target), {**timed, "l2": l2})  # the last one counts
    integration = {"baseline_ms": baseline["median_ms"], "history": history}
    write_json(run.root / "integration.json", {**integration, "projection": projection_entries})
    return run


def current() -> list[dict]:
    return projection.of_sets(projection.Tree(), SETS_WITH_A, BASE)


#: As the orchestrator passes them: the A of each set's A/B (``from_ms``; the first set's
#: is not used: it is projected from the baseline).
SETS_WITH_A = [{**SETS[0], "from_ms": 82.0}, *SETS[1:]]


def test_each_accepted_item_next_to_the_measured_gain_of_its_step(tmp_path):
    run = make_run(tmp_path, {"median_ms": BASE}, current())
    found = prediction.of_run(run)
    assert [(r["set"], r["label"], r["kind"]) for r in found] == [
        (1, "attn", "kernel"),
        (1, "graph_lm", "transform"),
        (2, "fuse_norms", "transform"),
        (3, "layer", "kernel"),
    ]
    attn, graph, norms, layer = found
    # the first set: two items at once, measured against the best single item, so its gain
    # is the baseline of analyze − its B (not paired, no interval)
    step = attn["step"]
    assert step["items"] == [K1, T1] and step["predicted_ms"] == 30.0  # 100 − (100 − 20 − 10)
    assert step["measured_ms"] == 25.0 and not step["paired"] and step["ci95_ms"] is None
    assert step["against"] == [K1]
    assert (attn["error_ms"], attn["ratio"]) == (-5.0, 0.833)
    assert attn["shared"] == [T1] and graph["shared"] == [K1]  # the error is the step's
    assert graph["step"] is step and graph["error_ms"] == -5.0
    assert attn["predicted_ms"] == 20.0 and graph["predicted_ms"] == 10.0  # their own
    # the kernel's bound and timing context from its last evaluation record (#226)
    assert (attn["bound"], attn["context"], attn["l2"]) == ("memory", "graph", "cold")
    assert "bound" not in graph and attn["module_class"] == "Block"
    # a transform: its gain alone predicted, paired against the set before it (the CI × A)
    assert norms["step"]["predicted_ms"] == 8.0 and norms["step"]["measured_ms"] == 5.5
    assert norms["step"]["paired"] and norms["step"]["ci95_ms"] == [3.775, 6.795]
    assert (norms["error_ms"], norms["ratio"], norms["shared"]) == (-2.5, 0.688, [])
    # not projectable (90 ms estimated on a set of 70.2 ms): its error is still measured
    assert layer["not_additive"].startswith("the estimated gain of layer (90.0 ms) is more")
    assert layer["step"]["predicted_ms"] == 90.0 and layer["step"]["measured_ms"] == 10.2
    assert (layer["error_ms"], layer["ratio"]) == (-79.8, 0.113)
    assert layer["step"]["ci95_ms"] == [8.424, 11.934]

    s = prediction.summary(found)
    assert (s["steps"], s["median_ratio"]) == (3, 0.688)
    assert (s["min_ratio"], s["max_ratio"]) == (0.113, 0.833)
    assert s["worst"]["set"] == 3
    assert prediction.headline(found) == (
        "Measured ÷ predicted gain: median 0.69x over 3 steps (0.11x .. 0.83x); the worst "
        "miss: `layer` (set 3): predicted 90.00 ms, measured 10.20 ms (-79.80 ms, 0.11x)."
    )
    lines = prediction.report_lines(run)
    assert lines[1] == "## Prediction error" and "in ms per run." in lines[3]
    table = "\n".join(lines)
    assert (
        "| 1 | `attn` | kernel | 30.00 | 25.00 | -5.00 | 0.83x | the step added 2 items: its "
        "error, shared; est. saved 20.00 ms; measured against the baseline of analyze (its "
        "A/B's A: `attn`); memory-bound, timed graph / cold L2 |"
    ) in table
    assert (
        "| 1 | `graph_lm` | transform | ″ | ″ | ″ | ″ | the same step as `attn`; est. saved "
        "10.00 ms |"
    ) in table
    assert "| 2 | `fuse_norms` | transform | 8.00 | 5.50 (3.77 .. 6.79) | -2.50 | 0.69x |  |" in (
        table
    )
    assert (
        "| 3 | `layer` | kernel | 90.00 | 10.20 (8.42 .. 11.93) | -79.80 | 0.11x | its set is "
        "not projectable (see Integration); compute-bound, timed eager / warm L2 |"
    ) in table


def test_an_integration_from_before_121_is_projected_again(tmp_path):
    """No ``summed_ms`` and no ``step``: the sets are projected again from their savings and
    the history's A/Bs (``projection.accepted_sets``), and the table is the same."""
    stale = [
        {
            "items": p["items"],
            "est_saved_ms": p["est_saved_ms"],
            "measured_ms": p["measured_ms"],
            "projected_ms": BASE - sum(p["est_saved_ms"].values()),  # summed, as before #121
            "counted_ms": p["est_saved_ms"],
            "est_saved_unit": projection.SAVED_UNIT,
        }
        for p in SETS
    ]
    old = make_run(tmp_path / "old", {"median_ms": BASE}, stale)
    new = make_run(tmp_path / "new", {"median_ms": BASE}, current())
    assert prediction.of_run(old) == prediction.of_run(new)
    assert prediction.report_lines(old) == prediction.report_lines(new)


def test_a_throughput_run_converts_a_kernels_estimate_per_run(tmp_path):
    """An ``integration.json`` from before #114 holds a kernel's est. saved ms per run of
    the workload: in a throughput run it is ÷ the seconds of audio (ms per audio second)."""
    baseline = {
        "median_ms": 37.659,
        "metric": "throughput",
        "metric_detail": {"throughput": 26.55, "audio_s": 153.6, "run_ms": 5784.4},
    }
    entries = [{"items": [K1], "est_saved_ms": {K1: 1536.0}, "measured_ms": 30.0}]
    history = [{"items": [K1], "passed": True, "ab": _ab([], 37.7, [0.18, 0.22])}]
    run = make_run(tmp_path, baseline, entries, history)
    (row,) = prediction.of_run(run)
    assert row["predicted_ms"] == 10.0 and row["step"]["predicted_ms"] == 10.0
    assert row["step"]["measured_ms"] == 7.7 and row["step"]["paired"]
    assert row["step"]["ci95_ms"] == [6.786, 8.294] and row["ratio"] == 0.77
    lines = prediction.report_lines(run)
    assert "in ms per second of generated audio." in lines[3]
    assert "| 1 | `attn` | kernel | 10.00 | 7.70 (6.79 .. 8.29) | -2.30 | 0.77x |" in lines[-2]


def test_a_region_target_next_to_its_fusion_candidate(tmp_path):
    """A region kernel's module-level estimate and its fusion candidate's predicted saving
    (``profile/fusions.json``, ms per profiled window × baseline ÷ window) against the
    measured gain of the step that added it alone (#231)."""
    region = "res_norm=/r/.truth/targets/res_norm/history/001_cuda_v1_eeeeeeee.py"
    entries = [{"items": [region], "est_saved_ms": {region: 5.0}, "measured_ms": 97.0}]
    history = [{"items": [region], "passed": True, "ab": _ab([], 100.0, [0.02, 0.04])}]
    run = make_run(tmp_path, {"median_ms": BASE}, entries, history)
    spec = {"id": "res_norm", "kind": "region", "parent_class": "Block", "fusion": "f00d"}
    write_json(run.target("res_norm") / "spec.json", {**spec, "module_class": "Region_res_norm"})
    candidate = {"id": "f00d", "saving_ms": 2.0, "kind": "region", "parent_class": "Block"}
    write_json(run.profile_dir / "fusions.json", {"window_ms": 50.0, "candidates": [candidate]})
    (row,) = prediction.of_run(run)
    assert row["kind"] == "region" and row["ratio"] == 0.6  # 3 of the 5 ms estimated
    assert row["fusion"] == {
        "id": "f00d",
        "how": "fusion f00d",
        "predicted_ms": 4.0,  # 2 ms of a 50 ms window × 100 ms
        "error_ms": -1.0,
        "ratio": 0.75,
    }
    assert "fusion `f00d`: predicted 4.00 ms, measured ÷ predicted 0.75x" in prediction.note(row)
    # with a fusion table but no candidate for it: no fusion prediction
    write_json(run.profile_dir / "fusions.json", {"window_ms": 50.0, "candidates": []})
    assert "fusion" not in prediction.of_run(run)[0]


def test_no_integration_no_section(tmp_path):
    run = RunDir(tmp_path / "r")
    run.root.mkdir()
    assert prediction.of_run(run) == [] and prediction.report_lines(run) == []
    assert prediction.headline([]) == ""


def test_the_predicted_saving_of_a_ledger_row():
    """baseline − (A − the step's estimated gain): A from the A/B's median, its rounds, or
    the A measured in a process of its own (``a_ms``)."""
    rounds = {"ab": {"a_ms": [101.0, 99.0, 100.0]}}
    assert ledger.predicted_saving(100.0, rounds, 20.0) == 20.0
    assert ledger.predicted_saving(100.0, {"ab": {"a_median_ms": 90.0}}, 5.0) == 15.0
    assert ledger.predicted_saving(100.0, {}, 5.0, a_ms=95.0) == 10.0  # separate processes
    assert ledger.predicted_saving(100.0, {}, 5.0) is None  # no A
    assert ledger.predicted_saving(100.0, rounds, None) is None  # nothing estimated
    assert ledger.predicted_saving(None, rounds, 5.0) is None


# ------------------------------------------------------------------ an integration


class FakeToolchain(dryrun.SimToolchain):
    torch_version = "2.14.1+cu130"

    def __init__(self) -> None:
        self.gpu = SimpleNamespace(arch="sm_120", name="Fake GPU")


@pytest.fixture
def integrated(tmp_path, monkeypatch):
    """The overlap scenario of ``test_projection_overlaps`` integrated on a GPU of a known
    architecture (the library on): a layer kernel estimated at 0.5 x the baseline that saves
    0.4 x, two transforms of the LM step and an FP8 transform of the layers' MLPs."""
    monkeypatch.setattr(orchestrator.toolchain, "setup", FakeToolchain)
    monkeypatch.setattr(charts, "available", lambda: False)
    monkeypatch.setenv(library.ENV, str(tmp_path / "library"))
    orch = make(tmp_path)
    run = orch.run
    dryrun._write_target(run, {"id": "layer", "module_class": "Qwen3DecoderLayer"})
    kernel(run, "layer", 2.0, saved_ms=0.5 * SIM_BASE)
    for name, (gain, _) in SAVES.items():
        if name != "layer":
            e2e_record(run, 1 / (1 - gain), [write_transform(run, name)])
    orch.worker = overlap_worker
    asyncio.run(orch.integrate())
    return orch


def test_integration_rows_have_the_predicted_saving_next_to_the_measured_one(integrated):
    run = integrated.run
    header = run.ledger.read_text().splitlines()[0].split("\t")
    assert header[header.index("est_saved_ms") + 1] == "pred_saved_ms"
    rows = [r for r in ledger.rows(run) if r["backend"] == ledger.INTEGRATE]
    probes = {r["snapshot"]: r for r in rows if ledger.kind(r) == ledger.PROBE}
    # a kernel alone: its module-level estimate (A is the unmodified model, as the baseline)
    assert probes["layer"]["pred_saved_ms"] == pytest.approx(0.5 * SIM_BASE, rel=1e-3)
    assert probes["layer"]["est_saved_ms"] == pytest.approx(0.4 * SIM_BASE, rel=1e-3)
    # a transform alone: its gain alone is its estimate, nothing predicted it
    assert all(probes[n]["pred_saved_ms"] is None for n in ("graph_lm", "fused_lm", "fp8_mlp"))
    # every accepted step: measured − predicted saving = the step's measured − estimated gain
    entries = read_json(run.root / "integration.json")["projection"]
    steps = {r["snapshot"]: r for r in rows if ledger.kind(r) == ledger.INTEGRATION}
    assert len(entries) == 4
    for entry in entries[1:]:
        row = steps["+".join(label(i) for i in entry["items"])]
        step = entry["step"]
        assert row["est_saved_ms"] - row["pred_saved_ms"] == pytest.approx(
            step["measured_gain_ms"] - step["est_gain_ms"], abs=0.02
        )
    fused = steps["layer+graph_lm+fused_lm"]  # overlaps graph_lm: no gain estimated
    assert fused["pred_saved_ms"] == pytest.approx(SIM_BASE - 0.3 * SIM_BASE, rel=1e-3)

    text = write_report(run).read_text()
    assert "## Prediction error" in text
    assert prediction.headline(prediction.of_run(run)) in text
    assert "| 3 | `fused_lm` | transform | 0.00 |" in text
    assert "est. saved 429.07 ms, 0.00 counted: overlaps `graph_lm`" in text


def test_the_library_keeps_the_prediction_errors_as_a_lesson(integrated, tmp_path):
    run = integrated.run
    path = library.predictions_path("sm_120")
    assert path == tmp_path / "library" / "sm_120" / "predictions.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(r["item"], r["kind"], r["set"]) for r in records] == [
        ("layer", "kernel", 1),
        ("graph_lm", "transform", 2),
        ("fused_lm", "transform", 3),
        ("fp8_mlp", "transform", 4),
    ]
    layer = records[0]
    assert layer["family"] == "block" and layer["run"] == str(run.root)
    assert layer["predicted_ms"] == pytest.approx(0.5 * SIM_BASE, rel=1e-3)
    assert layer["ratio"] == pytest.approx(0.8, rel=1e-2) and layer["step_items"] == 1
    assert layer["metric"] == "latency" and layer["sm_arch"] == "sm_120"

    rules = library.estimate_rules(records)
    # the steps estimated to gain nothing (fused_lm, fp8_mlp: counted once) have no ratio
    assert rules == [
        "kernel estimates (module-level), bound not recorded, timing context not recorded: "
        "the integration measured 0.80x of the predicted gain (median of 1 step in 1 run)",
        "transforms (their gain measured alone, added to the accepted set): the integration "
        "measured 1.00x of the predicted gain (median of 1 step in 1 run)",
    ]
    # a kernel's bound and timing context group its steps; several steps: their range
    more = [
        {**layer, "run": f"r{i}", "bound": "memory", "context": "graph", "l2": "cold", "ratio": x}
        for i, x in enumerate((0.5, 0.25, 1.5))
    ]
    shared = [{**layer, "run": "r9", "step_items": 2, "ratio": 0.9}] * 2  # one step, 2 items
    assert library.estimate_rules([*more, *shared]) == [
        "kernel estimates (module-level), memory-bound, timed graph / cold L2: the integration "
        "measured 0.50x of the predicted gain (median of 3 steps in 3 runs; 0.25x .. 1.50x)",
        "steps that added several items at once (their summed estimate): the integration "
        "measured 0.90x of the predicted gain (median of 1 step in 1 run)",
    ]
    lesson = (tmp_path / "library" / "lessons" / "estimates.md").read_text()
    assert "- sm_120: kernel estimates (module-level)" in lesson
    note = library.prediction_note("sm_120")
    assert note.startswith("**Estimated vs measured gains on sm_120** (kernel library: the")
    assert note in integrated._backend_record()  # the planner's prompt
    assert library.prediction_note("sm_89") == "" and library.prediction_note(None) == ""
    assert "estimates" not in library.lesson_names(run)  # never the librarian's

    asyncio.run(integrated.integrate(reuse=True))  # a re-integration replaces the run's lines
    assert len(path.read_text().splitlines()) == len(records)


def test_no_prediction_errors_are_stored_under_emulation(integrated, monkeypatch):
    """An emulated GPU's timings (#252) are not that architecture's."""
    monkeypatch.setenv("KERNEL_AGENT_EMULATE_ARCH", "sm_86")
    assert library.store_predictions(integrated.run, "sm_86") is None
    assert not library.predictions_path("sm_86").exists()
