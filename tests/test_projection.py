"""Projected latency with nested targets counted once (projection.py) and the end labels
of progress.png."""

import itertools
from dataclasses import asdict

import pytest
from synthetic_run import BASELINE_MS, make_run

from kernel_agent import charts, ledger, projection, watch
from kernel_agent.status import render
from kernel_agent.workspace import RunDir, write_json


def cls(name, groups=None, *, instances=None, example=None, phases=None):
    """A ``profile.json`` class entry: ``groups`` (pattern → instances), or only what an
    older profile has (``instances`` and ``example_qualname``)."""
    entry = {
        "root": "model",
        "cls": name,
        "instances": instances if instances is not None else sum((groups or {}).values()),
        "inclusive_ms": 1.0,
        "example_qualname": example or next(iter(groups or {}), ""),
    }
    if groups is not None:
        entry["groups"] = groups
    if phases is not None:
        entry["phases"] = phases
    return entry


def spec(module_class, **extra):
    return {"module_class": module_class, **extra}


def tree_of(specs, classes):
    return projection.build(specs, [{"classes": classes}])


LAYER = cls("Layer", {"model.layers.*": 4})
ATTN = cls("Attn", {"model.layers.*.attn": 4})
MLP = cls("MLP", {"model.layers.*.mlp": 4})
SPECS = {"layer": spec("Layer"), "attn": spec("Attn"), "mlp": spec("MLP")}


def test_nested_targets_take_the_better_of_parent_and_children():
    tree = tree_of(SPECS, [LAYER, ATTN, MLP])
    assert tree.nested()
    pairs = zip(tree.groups, tree.parents, strict=True)
    assert {g.target: tree.groups[p].target for g, p in pairs if p >= 0} == {
        "attn": "layer",
        "mlp": "layer",
    }

    # the layer's kernel saves more than its attention and MLP kernels together
    proj = projection.project(tree, {"layer": 100.0, "attn": 30.0, "mlp": 20.0}, 1000.0)
    assert proj.projected_ms == pytest.approx(900.0)  # not 1000 − 150
    assert proj.counted == pytest.approx({"layer": 100.0})
    assert proj.used == ["layer"] and proj.left_out == ["attn", "mlp"]
    assert proj.describe() == "layer; not counted (nested): attn, mlp"

    # ... and when the children save more, they count instead of the parent
    proj = projection.project(tree, {"layer": 40.0, "attn": 30.0, "mlp": 20.0}, 1000.0)
    assert proj.projected_ms == pytest.approx(950.0)
    assert proj.used == ["attn", "mlp"] and proj.left_out == ["layer"]

    # no saving yet: nothing counted
    proj = projection.project(tree, {"layer": None, "attn": 0.0}, 1000.0)
    assert proj.projected_ms == 1000.0 and proj.describe() == ""


def test_independent_targets_add_up():
    tree = tree_of({"attn": spec("Attn"), "mlp": spec("MLP")}, [ATTN, MLP])
    assert not tree.nested()
    proj = projection.project(tree, {"attn": 30.0, "mlp": 20.0, "other": 5.0}, 1000.0)
    assert proj.projected_ms == pytest.approx(945.0)  # "other" is not in the tree: in full
    assert proj.used == ["attn", "mlp", "other"] and proj.left_out == []
    assert proj.describe() == "attn + mlp + other"


def test_phase_targets_on_one_class_add_up():
    tree = tree_of(
        {
            "attn_decode": spec("Attn", phase="decode"),
            "attn_prefill": spec("Attn", phase="prefill"),
        },
        [ATTN],
    )
    assert not tree.nested()
    proj = projection.project(tree, {"attn_decode": 30.0, "attn_prefill": 10.0}, 100.0)
    assert proj.projected_ms == pytest.approx(60.0)
    # the same instances and phase: one or the other
    tree = tree_of({"a": spec("Attn"), "b": spec("Attn")}, [ATTN])
    assert projection.project(tree, {"a": 30.0, "b": 10.0}, 100.0).projected_ms == 70.0


# VoxCPM2: a LocEnc (with its own 12-layer encoder), the base and residual LMs (28 + 8
# layers) and a DiT decoder (12 layers); RMSNorm twice per layer plus each model's norm.
MODELS = ["model.feat_encoder.encoder", "model.base_lm", "model.residual_lm"]
MODELS.append("model.feat_decoder.estimator.decoder")
LAYERS = dict(zip(MODELS, (12, 28, 8, 12), strict=True))
VOXCPM = [
    cls("MiniCPMModel", dict.fromkeys(MODELS, 1)),
    cls("VoxCPMLocEnc", {"model.feat_encoder": 1}),
    cls("MiniCPMDecoderLayer", {f"{m}.layers.*": n for m, n in LAYERS.items()}),
    cls("MiniCPMAttention", {f"{m}.layers.*.self_attn": n for m, n in LAYERS.items()}),
    cls("MiniCPMMLP", {f"{m}.layers.*.mlp": n for m, n in LAYERS.items()}),
    cls(
        "MiniCPMRMSNorm",
        {f"{m}.layers.*.{norm}": n for m, n in LAYERS.items() for norm in ("in_ln", "post_ln")}
        | {f"{m}.norm": 1 for m in MODELS},
    ),
    cls("CausalResidualUnit", {"model.audio_vae.decoder.model.*.block.*": 18}),
]
VOXCPM_SPECS = {
    "attn_fused": spec("MiniCPMAttention"),
    "decoder_layer_fused": spec("MiniCPMDecoderLayer"),
    "lm_step_megakernel": spec(
        "MiniCPMModel", qualname_regex=r"^model\.(base_lm|residual_lm)$", phase="decode"
    ),
    "locenc_fused": spec("VoxCPMLocEnc"),
    "mlp_fused": spec("MiniCPMMLP"),
    "rmsnorm_fused": spec("MiniCPMRMSNorm"),
    "vae_resunit_fused": spec("CausalResidualUnit"),
}
# est. saved ms per run of each target's best kept result in the real run (20261005-042829);
# attn_fused's is the inflated pre-#6 record
VOXCPM_SAVED = {
    "attn_fused": 3212.08,
    "decoder_layer_fused": 1246.28,
    "locenc_fused": 325.455,
    "mlp_fused": 25.663,
    "rmsnorm_fused": 422.493,
}
VOXCPM_BASE = 5406.99


def test_voxcpm_like():
    tree = tree_of(VOXCPM_SPECS, VOXCPM)
    naive = VOXCPM_BASE - sum(VOXCPM_SAVED.values())
    assert naive == pytest.approx(175.0, abs=0.1)  # what progress.png said

    proj = projection.project(tree, VOXCPM_SAVED, VOXCPM_BASE)
    # per layer the children (attention 3212 / 60 + MLP + 2 norms) beat the layer's 1246 / 60,
    # in the LocEnc its 12 layers' children beat the LocEnc's own 325 ms
    assert proj.used == ["attn_fused", "rmsnorm_fused", "mlp_fused"]
    assert proj.left_out == ["decoder_layer_fused", "locenc_fused"]
    assert proj.projected_ms == pytest.approx(VOXCPM_BASE - 3212.08 - 422.493 - 25.663)
    assert proj.describe() == (
        "attn_fused + rmsnorm_fused + mlp_fused; "
        "not counted (nested): decoder_layer_fused, locenc_fused"
    )

    # A smaller attention saving: the layer wins in every layer, but the LocEnc holds only
    # 12 of the 60 layers and beats them there, so the layer counts in part (48 / 60).
    saved = {**VOXCPM_SAVED, "attn_fused": 300.0}
    proj = projection.project(tree, saved, VOXCPM_BASE)
    layer, norm = 1246.28 / 60, 422.493 / 124
    assert 12 * layer + norm < 325.455  # the LocEnc beats its layers (+ its encoder's norm)
    assert proj.counted == pytest.approx(
        {
            "locenc_fused": 325.455,
            "decoder_layer_fused": 48 * layer,
            "rmsnorm_fused": 3 * norm,  # the norms of the LMs and the DiT, outside every layer
        }
    )
    assert proj.part("decoder_layer_fused") == pytest.approx(0.8)
    assert proj.left_out == ["attn_fused", "mlp_fused"]
    assert proj.describe().startswith("decoder_layer_fused (80%) + locenc_fused + rmsnorm_fused")

    # The LM step megakernel (decode, base + residual LM) holds 36 of the layers.
    saved = {**saved, "lm_step_megakernel": 1500.0}
    proj = projection.project(tree, saved, VOXCPM_BASE)
    assert proj.counted["lm_step_megakernel"] == 1500.0
    assert proj.counted["decoder_layer_fused"] == pytest.approx(12 * layer)  # the DiT's
    assert proj.counted["rmsnorm_fused"] == pytest.approx(norm)  # the DiT's norm


def test_older_profile_example_and_captured_qualnames():
    """A profile without ``groups``: the example instance of each class and the captured
    instance of each target, with the instances split evenly over those patterns."""
    examples = {
        "MiniCPMModel": ("model.feat_encoder.encoder", 4, "model.base_lm"),
        "VoxCPMLocEnc": ("model.feat_encoder", 1, "model.feat_encoder"),
        "MiniCPMDecoderLayer": (
            "model.feat_encoder.encoder.layers.0",
            60,
            "model.base_lm.layers.0",
        ),
        "MiniCPMAttention": ("model.feat_encoder.encoder.layers.0.self_attn", 60, "self_attn"),
        "MiniCPMMLP": ("model.feat_encoder.encoder.layers.0.mlp", 60, "mlp"),
        "MiniCPMRMSNorm": ("model.feat_encoder.encoder.layers.0.in_ln", 124, "in_ln"),
    }
    classes, specs = [], {}
    for target, s in VOXCPM_SPECS.items():
        name = s["module_class"]
        if name not in examples:
            continue
        example, instances, captured = examples[name]
        if not captured.startswith("model."):
            captured = f"model.base_lm.layers.0.{captured}"
        classes.append(cls(name, instances=instances, example=example))
        specs[target] = {**s, "capture": {"qualname": captured}}
    tree = tree_of(specs, classes)
    shares = {(g.target, g.pattern): g.share for g in tree.groups}
    assert shares[("decoder_layer_fused", "model.base_lm.layers.*")] == 0.5
    assert shares[("lm_step_megakernel", "model.base_lm")] == 1.0  # the regex drops the encoder
    proj = projection.project(tree, VOXCPM_SAVED, VOXCPM_BASE)
    assert proj.used == ["attn_fused", "rmsnorm_fused", "mlp_fused"]  # nothing counted twice
    assert proj.projected_ms == pytest.approx(VOXCPM_BASE - 3212.08 - 422.493 - 25.663)


def test_scope_and_regions():
    # a target restricted to one instance; a region inside some of the layers
    specs = {
        "layer0": spec("Layer", qualname="model.layers.0"),
        "attn": spec("Attn"),
        "region": spec(
            "Region_region", kind="region", parent_class="Layer", qualname_regex=r"layers\.\d+$"
        ),
    }
    tree = tree_of(specs, [LAYER, ATTN, cls("Other", {"model.other": 1})])
    groups = {g.target: g for g in tree.groups}
    assert groups["layer0"].pattern == "model.layers.*" and groups["layer0"].inside == 0.25
    assert groups["region"].pattern == "model.layers.*.[region]"
    # layer 0's kernel (10 ms) beats its attention + region (a quarter of 8 + 4 ms); the
    # other three layers keep theirs
    proj = projection.project(tree, {"layer0": 10.0, "attn": 8.0, "region": 4.0}, 100.0)
    assert proj.counted == pytest.approx({"layer0": 10.0, "attn": 6.0, "region": 3.0})
    assert proj.describe() == "layer0 + attn (75%) + region (75%)"
    proj = projection.project(tree, {"layer0": 2.0, "attn": 8.0, "region": 4.0}, 100.0)
    assert proj.projected_ms == 88.0 and proj.left_out == ["layer0"]

    # a class that is in no profile: its captured instance, else it contains nothing
    tree = tree_of(
        {"x": spec("X", capture={"qualname": "model.layers.1.x"}), "y": spec("Y")}, [LAYER]
    )
    assert {g.target: g.pattern for g in tree.groups} == {"x": "model.layers.*.x", "y": ""}


def test_series_uses_the_results_that_stand():
    """A re-evaluation (#58) replaces its snapshot's earlier row in the projection."""
    tree = tree_of({"attn": spec("Attn")}, [ATTN])

    def row(exp, status, snapshot, speedup, saved, target="attn", correct=True):
        return {
            "exp": exp,
            "target": target,
            "status": status,
            "snapshot": snapshot,
            "speedup": speedup,
            "est_saved_ms": saved,
            "correct": correct,
        }

    rows = [
        row(1, "keep", "a.py", 9.8, 100.0),
        row(2, "keep", "b.py", 46.9, 300.0),  # a stale record
        row(3, "keep", "e2e_1", 2.0, None, target="e2e"),
        row(4, "re-evaluated", "b.py", 18.3, 250.0),
        row(5, "discard", "c.py", 5.0, 50.0),
    ]
    steps = [(r["exp"], p.projected_ms) for r, p in projection.series(tree, 1000.0, rows)]
    assert steps == [(1, 900.0), (2, 700.0), (4, 750.0)]
    rows.append(row(6, "re-evaluated", "b.py", None, None, correct=False))  # it failed
    assert projection.series(tree, 1000.0, rows)[-1][1].projected_ms == 900.0


def test_profiler_groups_build_the_tree():
    import torch
    from torch import nn

    from kernel_agent.profiling.profiler import ModuleTimer

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x):
            return self.lin(x)

    model = nn.Module()
    model.enc = Block()
    model.layers = nn.ModuleList(Block() for _ in range(3))
    with torch.inference_mode(), ModuleTimer({"model": model}, cuda=False) as timer:
        x = model.enc(torch.randn(1, 4))
        for layer in model.layers:
            x = layer(x)
    classes = [asdict(s) for s in timer.class_stats()]
    groups = {c["cls"]: c["groups"] for c in classes}
    assert groups["Block"] == {"model.layers.*": 3, "model.enc": 1}
    assert groups["Linear"] == {"model.layers.*.lin": 3, "model.enc.lin": 1}
    tree = projection.build({"block": spec("Block"), "lin": spec("Linear")}, [{"classes": classes}])
    assert projection.project(tree, {"block": 4.0, "lin": 2.0}, 10.0).counted == {"block": 4.0}


def test_synthetic_run_series_and_views(tmp_path):
    run = make_run(tmp_path)
    tree = projection.tree(run)
    assert not tree.nested()  # attn, mlp, rmsnorm, rope: no target holds another
    rows = ledger.rows(run)
    series = projection.series(tree, BASELINE_MS, rows)
    kept = [r for r in rows if r["target"] != ledger.E2E and r["status"] == ledger.KEEP]
    assert [r["exp"] for r, _ in series] == [r["exp"] for r in kept]
    final = series[-1][1]
    assert final.used == ["attn", "mlp", "rmsnorm", "rope"]

    s = ledger.summary(run)
    assert s["projected_ms"] == s["projection"].projected_ms == final.projected_ms
    assert "projected from attn + mlp + rmsnorm + rope" in render(run, width=160)
    state = watch.state(run)["summary"]
    assert state["projected_ms"] == s["projected_ms"]
    assert state["projection"]["label"] == final.describe()
    assert state["projection"]["steps"] == {str(r["exp"]): p.projected_ms for r, p in series}


# ------------------------------------------------------------------ progress.png


def _progress_run(tmp_path, measured_ms, nested=True):
    """A run whose integrated result ends right above the projection's end (550 ms), with
    the layer's kernel (nested: attention + MLP not counted) or without."""
    run = RunDir(tmp_path / "run")
    run.root.mkdir(parents=True)
    write_json(run.run_json, {"card": {"repo_id": "org/model"}, "created": "2026-10-05 09:00:00"})
    write_json(run.baseline_json, {"median_ms": 1000.0})
    write_json(run.profile_dir / "profile.json", {"classes": [LAYER, ATTN, MLP]})
    for target, s in SPECS.items():
        write_json(run.target(target) / "spec.json", s)

    def row(minute, target, status, **values):
        ledger.append(
            run,
            {
                "time": f"2026-10-05 09:{minute:02d}:00",
                "target": target,
                "status": status,
                "correct": True,
                **values,
            },
        )

    row(10, "attn", "keep", speedup=1.5, est_saved_ms=180.0 if nested else 270.0)
    row(15, "mlp", "keep", speedup=1.3, est_saved_ms=90.0 if nested else 180.0)
    if nested:
        row(25, "layer", "keep", speedup=2.0, est_saved_ms=450.0)
    row(
        59,
        "e2e",
        "keep",
        speedup=1000.0 / measured_ms,
        new_ms=measured_ms,
        snapshot="layer+cuda_graph",
        backend="integrate",
    )
    return run


def _texts(run):
    """The visible text boxes (display px) of the run's progress chart."""
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    rows = ledger.rows(run)
    with matplotlib.rc_context(charts._RC):
        fig = Figure(figsize=(10.0, 5.6), dpi=charts.DPI)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        start = ledger.start_time(run, rows)
        charts._draw_run(ax, run, {"median_ms": 1000.0}, 1000.0, rows, start)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        return {t.get_text(): t.get_window_extent(renderer) for t in ax.texts if t.get_visible()}


@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("measured_ms", [560.0, 600.0, 640.0, 700.0])
def test_progress_end_labels_do_not_overlap(tmp_path, measured_ms, nested):
    pytest.importorskip("matplotlib")
    texts = _texts(_progress_run(tmp_path, measured_ms, nested))
    projected = next(t for t in texts if t.startswith("projected "))
    measured = next(t for t in texts if t.startswith("measured "))
    assert projected == "projected 550.0 ms"  # nested: the layer alone, not 1000 − 720
    assert measured.startswith(f"measured {measured_ms:,.1f} ms")
    subtitle = next(t for t in texts if "kernel evaluations" in t)
    if nested:
        assert "\nprojected from layer; not counted (nested): attn, mlp" in subtitle
    else:
        assert subtitle.endswith("\nprojected from attn + mlp")
    for (a, box_a), (b, box_b) in itertools.combinations(texts.items(), 2):
        assert not box_a.overlaps(box_b), (a, b)
