"""The projection of an accepted set added up items whose savings overlap (#121).

In runs/openbmb--VoxCPM2/20261006-004718-retest2 the final set projected −24.6 ms against
6.03 ms measured: three transforms of the LM step (``graph_lm_step``, ``fused_lm_step``,
``fp8_lm_attn_proj``) each counted their gain alone, and the W8A8 DiT layer kernel counted
its module-level estimate on top of the transforms that already rewrote those layers. Now
items whose modules overlap (what the patcher recorded of each, ``integrate/owners.py``)
count once, and every set after the first is projected from the set before it, as
measured, minus the estimated gain of its step.

CPU only: projections of synthetic sets, and a simulated integration with a fake
measurement layer whose items overlap the way the run's did.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from test_integrate import BASE, WIGGLE, kernel, label, make, parse, write
from test_integrate import e2e as e2e_record
from test_integration_oom import _patches

from kernel_agent import dryrun, projection, report, truth
from kernel_agent.integrate import owners
from kernel_agent.report import write_report
from kernel_agent.workspace import read_json, write_json

GRAPH, FUSED, FP8 = "/t/history/001_graph_lm.py", "/t/history/002_fused_lm.py", "/t/003_fp8_mlp.py"
CFM = "/t/history/004_graph_cfm.py"
LAYER = "layer=/r/targets/layer/history/001_cuda_v1.py"
DIT = "dit=/r/targets/dit/history/001_cuda_v1.py"
#: What the patcher records of each item: its touched and owned modules together.
CHANGES = {
    GRAPH: ["workload.lm_step"],
    FUSED: ["workload.lm_step"],
    FP8: ["model.layers.*.mlp", "model.layers.*.mlp.up_proj"],
    LAYER: ["model.layers.*"],
    CFM: ["model.decoder"],
    DIT: ["model.decoder.layers.*"],
}


def test_where_two_items_save_the_same_time():
    # the same module, one inside the other's (both ways), a module that contains the other's
    assert owners.common(["workload.lm_step"], ["workload.lm_step"]) == ["workload.lm_step"]
    assert owners.common(["model.layers.*"], ["model.layers.*.mlp"]) == ["model.layers.*.mlp"]
    assert owners.common(["model.decoder"], ["model.decoder.layers.*"]) == [
        "model.decoder.layers.*"
    ]
    assert owners.common(["model.decoder"], ["model.decoder_norm"]) == []  # a name, not a child
    assert owners.common(["model.a"], []) == []
    # what can be stacked (#112) is narrower: a module around another item's is no overlap
    record = owners.Owners()
    record.touched = {CFM: {"model.decoder"}, DIT: {"model.decoder.layers.*"}}
    record.owns = {CFM: set(), DIT: {"model.decoder.layers.*"}}
    assert record.shared(CFM, DIT) == []
    assert record.changes() == {CFM: ["model.decoder"], DIT: ["model.decoder.layers.*"]}


def test_items_that_change_the_same_modules_count_once():
    """Two transforms of the LM step and a layer kernel with a slower FP8 transform of the
    MLPs inside its layers: their gains alone add up to more than the baseline."""
    saved = {GRAPH: 30.0, FUSED: 28.0, FP8: -5.0, LAYER: 40.0, CFM: 10.0}
    before = projection.of_set(projection.Tree(), saved, 100.0)  # what the run did
    assert before["projected_ms"] == pytest.approx(100.0 - 30 - 28 - 40 - 10 + 5)  # −3 ms

    entry = projection.of_set(projection.Tree(), saved, 100.0, CHANGES)
    assert entry["projected_ms"] == pytest.approx(20.0)  # 100 − 30 − 40 − 10
    assert entry["counted_ms"] == {GRAPH: 30.0, FUSED: 0.0, LAYER: 40.0, CFM: 10.0}
    assert entry["overlaps"] == [
        {"items": [GRAPH, FUSED], "counted": [GRAPH], "where": ["workload.lm_step"]},
        {"items": [LAYER, FP8], "counted": [LAYER], "where": ["model.layers.*.mlp"]},
    ]  # FP8 is slower alone, but overlaps the layer kernel: not added back either

    # the transform around the DiT layers and the kernel of those layers: the larger counts
    entry = projection.of_set(projection.Tree(), {CFM: 10.0, DIT: 25.0}, 100.0, CHANGES)
    assert entry["counted_ms"] == {CFM: 0.0, DIT: 25.0} and entry["projected_ms"] == 75.0

    # of three items in a row (a-b, b-c), the two that do not overlap, when they save more
    row = {"/t/a.py": ["model.a"], "/t/b.py": ["model.a", "model.c"], "/t/c.py": ["model.c"]}
    chain = projection.of_set(
        projection.Tree(), {"/t/a.py": 10.0, "/t/b.py": 15.0, "/t/c.py": 10.0}, 100.0, row
    )
    assert chain["projected_ms"] == 80.0
    assert chain["overlaps"][0]["counted"] == ["/t/a.py", "/t/c.py"]

    # two kernels never overlap by their modules: the tree nests them, by their instances
    attn = "attn=/r/targets/attn/history/001_v1.py"
    changes = {**CHANGES, attn: ["model.layers.*.self_attn"]}
    both = projection.of_set(projection.Tree(), {LAYER: 40.0, attn: 20.0}, 100.0, changes)
    assert both["projected_ms"] == 40.0 and "overlaps" not in both


def test_a_later_set_is_projected_from_the_set_before_it():
    """Each step's estimate meets what the step measured, not a sum of gains alone."""
    layer2 = "layer=/r/targets/layer/history/002_cuda_v2.py"
    changes = {**CHANGES, layer2: ["model.layers.*"]}
    sets = [
        {"items": [GRAPH], "est_saved_ms": {GRAPH: 30.0}, "measured_ms": 70.0},
        {
            "items": [GRAPH, LAYER],
            "est_saved_ms": {GRAPH: 30.0, LAYER: 40.0},
            "measured_ms": 45.0,
            "from_ms": 71.0,  # A of this step: the set before it, measured again
        },
        {
            "items": [GRAPH, LAYER, FP8],
            "est_saved_ms": {GRAPH: 30.0, LAYER: 40.0, FP8: -5.0},
            "measured_ms": 44.0,
            "from_ms": 46.0,
        },
        {  # a newer version of the kernel, estimated at 120 ms: more than there is left
            "items": [GRAPH, layer2, FP8],
            "est_saved_ms": {GRAPH: 30.0, layer2: 120.0, FP8: -5.0},
            "measured_ms": 40.0,
            "from_ms": 44.0,
        },
    ]
    first, second, third, fourth = projection.of_sets(projection.Tree(), sets, 100.0, changes)
    assert first["projected_ms"] == first["summed_ms"] == 70.0 and "step" not in first
    assert second["step"] == {
        "new": [LAYER],
        "old": [],
        "from_ms": 71.0,
        "est_gain_ms": 40.0,
        "measured_gain_ms": 26.0,
    }
    assert second["summed_ms"] == 30.0 and second["projected_ms"] == 31.0  # 71 − 40
    # the FP8 MLPs lie in the layers the kernel replaces: no gain estimated, none added back
    assert third["step"]["est_gain_ms"] == 0.0 and third["projected_ms"] == 46.0
    assert all(p["est_saved_unit"] == projection.SAVED_UNIT for p in (first, fourth))
    assert fourth["step"]["old"] == [LAYER] and fourth["step"]["est_gain_ms"] == 80.0
    assert fourth["projected_ms"] is None
    assert fourth["not_additive"] == (
        "the estimated gain of layer (80.0 ms) is more than the 44.0 ms of the set before it, "
        "where it overlaps fp8_mlp (in model.layers.*.mlp)"
    )
    row = report._set_row(fourth, third)
    assert row.startswith(
        "| ↳ + `layer` #002 instead of `layer` #001 (est. gain 80.00 ms, measured 4.00 ms) "
        "(not additive: the estimated gain of layer (80.0 ms) is more than"
    )
    assert row.endswith("| not additive | 40.0 | — |")
    assert report._set_row(third, second).startswith(
        "| ↳ + `fp8_mlp` (not counted, overlapping a counted item: fp8_mlp) (est. gain 0.00 ms"
    )

    # a first set whose savings counted exceed the baseline
    (alone,) = projection.of_sets(
        projection.Tree(), [{"items": [LAYER], "est_saved_ms": {LAYER: 140.0}}], 100.0
    )
    assert alone["projected_ms"] is None and alone["summed_ms"] == -40.0
    assert alone["not_additive"] == (
        "the savings counted (140.0 ms) are more than the baseline (100.0 ms); the largest: "
        "layer 140.0 ms"
    )


# ------------------------------------------------------------------ the integration

#: What each item saves alone (× the baseline) and the modules it changes: items of one
#: group overlap, so a set saves the largest of a group and 0.02 more per other item in it.
SAVES = {"layer": (0.40, "layers"), "graph_lm": (0.30, "lm"), "fused_lm": (0.28, "lm")}
SAVES["fp8_mlp"] = (0.10, "layers")
MODULES = {  # (touched, owns), as the patcher records them
    "layer": (["model.layers.*"], ["model.layers.*"]),
    "graph_lm": (["workload.lm_step"], []),
    "fused_lm": (["workload.lm_step"], []),
    "fp8_mlp": (["model.layers.*.mlp"], []),
}


def overlap_worker(run, command, *args):
    """A paired A/B of the items above, with what the patcher records of each state."""

    def ms(items: list[str]) -> float:
        groups: dict[str, list[float]] = {}
        for item in items:
            gain, group = SAVES[label(item)]
            groups.setdefault(group, []).append(gain)
        return BASE * (1.0 - sum(max(g) + 0.02 * (len(g) - 1) for g in groups.values()))

    def patches(items: list[str]) -> dict:
        return _patches({label(i) if "=" in i else Path(i).stem: MODULES[label(i)] for i in items})

    assert command == "e2e_ab"
    ns = parse(args)
    a, b = [*ns.kernel, *ns.transform], [*ns.b_kernel, *ns.b_transform]
    r = dryrun._e2e_result(ms(b))
    r["times_ms"] = [round(ms(b) * w, 3) for w in WIGGLE[: ns.rounds]]
    a_ms = [round(ms(a) * w, 3) for w in reversed(WIGGLE[: ns.rounds])]
    r["ab"] = {"mode": "paired", "a_ms": a_ms, "b_ms": r["times_ms"], "a_patches": patches(a)}
    r["patches"] = patches(b)
    return r


def test_an_accepted_set_counts_two_transforms_of_one_module_and_a_kernel_once(tmp_path):
    """``graph_lm`` and ``fused_lm`` both rewrite the LM step; the ``layer`` kernel owns the
    layers whose MLPs ``fp8_mlp`` changes, and its module-level estimate (0.5 x the
    baseline) is above what it saves end to end (0.4 x). The four are accepted one by one;
    summing their savings projected −0.18 x the baseline for the final set."""
    orch = make(tmp_path)
    run = orch.run
    dryrun._write_target(run, {"id": "layer", "module_class": "Qwen3DecoderLayer"})
    kernel(run, "layer", 2.0, saved_ms=0.5 * BASE)
    for name, (gain, _) in SAVES.items():
        if name != "layer":
            e2e_record(run, 1 / (1 - gain), [write(run, name)])
    orch.worker = overlap_worker
    asyncio.run(orch.integrate())

    data = read_json(run.root / "integration.json")
    accepted = [label(a["item"]) for a in data["accepted"]]
    assert accepted == ["layer", "graph_lm", "fused_lm", "fp8_mlp"]
    entries = data["projection"]
    assert [[label(i) for i in p["step"]["new"]] for p in entries[1:]] == [
        ["graph_lm"],
        ["fused_lm"],
        ["fp8_mlp"],
    ]
    first, _, third, last = entries
    items = {label(i): i for i in last["items"]}
    saved = last["est_saved_ms"]
    assert saved[items["layer"]] == pytest.approx(0.5 * BASE)
    assert saved[items["fused_lm"]] == pytest.approx(0.28 * BASE, rel=1e-3)  # its gain alone
    # before: every saving in full, −0.18 x the baseline
    old = projection.of_set(projection.tree(run), saved, BASE)["projected_ms"]
    assert old == pytest.approx(-0.18 * BASE, rel=1e-2)
    # now: of each module the larger saving, 1 − 0.5 − 0.3
    assert last["summed_ms"] == pytest.approx(0.2 * BASE, rel=1e-2)
    assert last["overlaps"] == [
        {
            "items": [items["layer"], items["fp8_mlp"]],
            "counted": [items["layer"]],
            "where": ["model.layers.*.mlp"],
        },
        {
            "items": [items["graph_lm"], items["fused_lm"]],
            "counted": [items["graph_lm"]],
            "where": ["workload.lm_step"],
        },
    ]
    assert first["projected_ms"] == pytest.approx(0.5 * BASE)  # measured 0.6 x
    # the two later transforms add nothing to the estimate: projected at the set before them
    for p in (third, last):
        assert p["step"]["est_gain_ms"] == pytest.approx(0.0, abs=1e-3)
        assert p["projected_ms"] == pytest.approx(p["step"]["from_ms"], abs=1e-3)
        assert p["step"]["measured_gain_ms"] == pytest.approx(0.02 * BASE, rel=0.05)
    assert last["measured_ms"] == pytest.approx(0.26 * BASE, rel=1e-3)
    assert last["measured_ms"] / last["projected_ms"] == pytest.approx(0.26 / 0.28, rel=1e-2)

    text = write_report(run).read_text()
    assert "| `layer` | 766.2 | 919.4 | 1.20 |" in text
    assert "| ↳ + `graph_lm` (est. gain 459.7" in text
    assert "| ↳ + `fused_lm` (not counted, overlapping a counted item: fused_lm) (est. gain" in text
    assert "| ↳ + `fp8_mlp` (not counted, overlapping a counted item: fp8_mlp) (est. gain" in text

    # an integration.json from before #121: the report projects it again from its history
    stale = [
        {k: v for k, v in p.items() if k in ("items", "est_saved_ms", "measured_ms")}
        for p in entries
    ]
    for p, entry in zip(stale, entries, strict=True):
        p.update(projected_ms=-0.18 * BASE, est_saved_unit=projection.SAVED_UNIT)
        p["counted_ms"] = {i: v for i, v in entry["est_saved_ms"].items() if v and v > 0}
    truth.writable(run.root / "integration.json")
    write_json(run.root / "integration.json", {**data, "projection": stale})
    again = projection.of_integration(
        read_json(run.root / "integration.json"), projection.tree(run), projection.Units(), BASE
    )
    assert again == entries
    assert write_report(run).read_text() == text
