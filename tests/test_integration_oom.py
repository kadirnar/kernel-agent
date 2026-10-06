"""Round-2 kernels of runs/openbmb--VoxCPM2/20261006-004718 never reached the result (#112):

* every addition to the accepted composite (17 transforms) ran out of GPU memory in the
  in-process A/B (both states in one process) and was dropped as ``runtime_error``; now an
  out-of-memory step is ``oom``, measured again in separate processes with
  ``--expandable-segments``, and recorded as ``oom`` only when it still runs out;
* ``vae_decoder__reduced`` replaces the ``CausalDecoder`` that the accepted
  ``vae_channels_last`` transform rewrites, ``loc_enc_decode`` the LocEnc that
  ``graph_loc_enc`` CUDA-graphs: they cannot be stacked. The patcher records which modules
  each item changes and replaces (``integrate/owners.py``), and an item that overlaps
  accepted items is also tried in their place.

CPU only: the toy decoder for the patcher and the worker, a simulated run with a fake
measurement layer for the integration.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from test_integrate import BASE, WIGGLE, kernel, label, make, parse, write
from test_integrate import e2e as e2e_record
from test_undo import NORM, _file, _run_dir, _worker, cpu, toy  # noqa: F401 (fixtures)

from kernel_agent import abtest, charts, dryrun, improve, ledger
from kernel_agent.integrate import owners, undo
from kernel_agent.integrate.patcher import KernelPatch, PatchReport, apply_kernels, apply_transforms
from kernel_agent.report import write_report
from kernel_agent.workspace import read_json

#: What ``torch`` says when the caching allocator gives up (a step of the issue's run).
OOM_TEXT = (
    "CUDA out of memory. Tried to allocate 450.00 MiB. GPU 0 has a total capacity of "
    "15.55 GiB of which 120.06 MiB is free."
)


# ------------------------------------------------------------------ what each item changes


def test_the_patcher_records_the_modules_each_item_changes(tmp_path, toy):  # noqa: F811
    report = PatchReport()
    handles: list[undo.Undo] = []
    norm = _file(tmp_path, "norm.py", NORM)
    apply_kernels(toy.roots(), [KernelPatch("norm", "ToyRMSNorm", norm)], report, handles=handles)
    norms = ["model.layers.*.input_layernorm", "model.layers.*.post_attention_layernorm"]
    assert report.touched["norm"] == report.owns["norm"] == [*norms, "model.norm"]

    graph = _file(  # what a CUDA graph of a module does: its forward, an instance attribute
        tmp_path,
        "graph_mlp.py",
        "def apply(workload):\n"
        "    for layer in workload.model.layers:\n"
        "        mlp = layer.mlp\n"
        "        mlp.forward = lambda x, f=type(mlp).forward, m=mlp: f(m, x)\n",
    )
    swap = _file(  # another module in a child's place, weights converted, a workload hook
        tmp_path,
        "swap_attn.py",
        "import torch\nfrom torch import nn\n\n\nclass Wrapped(nn.Module):\n"
        "    def __init__(self, inner):\n        super().__init__()\n        self.inner = inner\n\n"
        "    def forward(self, *a, **k):\n        return self.inner(*a, **k)\n\n\n"
        "def apply(workload):\n"
        "    model = workload.model\n"
        "    model.layers[1].self_attn = Wrapped(model.layers[1].self_attn)\n"
        "    with torch.no_grad():\n"
        "        model.lm_head.weight.data = model.lm_head.weight.data.clone()\n"
        "    workload.step = lambda: None\n",
    )
    apply_transforms(toy, [graph, swap], report, handles=handles)
    assert report.touched["graph_mlp"] == ["model.layers.*.mlp"] and report.owns["graph_mlp"] == []
    touched = ["model.layers.*.self_attn", "model.lm_head", "workload.step"]
    assert report.touched["swap_attn"] == touched
    assert report.owns["swap_attn"] == ["model.layers.*.self_attn"]  # the replaced child
    assert handles[-1].owns == ["model.layers.1.self_attn"]  # the handle's: exact names
    for handle in reversed(handles):
        handle.undo()

    # without undo handles (a separate-process e2e), and a transform that fails half way
    broken = _file(
        tmp_path,
        "broken.py",
        "def apply(workload):\n    workload.model.norm.forward = print\n"
        "    raise AttributeError(\"module 'ka_candidate' has no attribute 'CausalDecoder'\")\n",
    )
    report = PatchReport()
    with pytest.raises(AttributeError):
        apply_transforms(toy, [graph, broken], report)
    assert report.touched == {"graph_mlp": ["model.layers.*.mlp"], "broken": ["model.norm"]}
    assert report.transforms == ["graph_mlp"]


def _patches(items: dict[str, tuple[list[str], list[str]]]) -> dict:
    """``patches`` of a worker result: (touched, owns) per kernel target / transform stem."""
    return {
        "touched": {k: t for k, (t, _) in items.items()},
        "owns": {k: o for k, (_, o) in items.items()},
    }


def test_items_overlap_when_they_change_the_same_modules():
    """The issue's items, as the patcher records them on the VoxCPM2 model."""
    dit = ["model.feat_decoder.estimator.decoder.layers.*", "model.feat_encoder.encoder.layers.*"]
    decoder = "model.audio_vae.decoder"
    items = {
        "loc_enc_decode": ("kernel", (["model.feat_encoder"], ["model.feat_encoder"])),
        "dit_layer": ("kernel", (dit, dit)),
        "vae_decoder__reduced": ("kernel", ([decoder], [decoder])),
        "graph_loc_enc": ("transform", (["model.feat_encoder"], [])),
        "graph_inductor_cfm_solver": ("transform", (["model.feat_decoder"], [])),
        "fp8_locdit_mlp": ("transform", ([dit[0] + ".mlp"], [])),
        "vae_bf16": ("transform", (["model.audio_vae", decoder + ".model.*"], [])),
        "vae_channels_last": ("transform", ([decoder, decoder + ".model.*"], [decoder])),
        "graph_lm_step": ("transform", (["workload.lm_step"], [])),
        "fused_lm_step": ("transform", (["workload.lm_step"], [])),
    }
    arg = {
        name: f"{name}=/run/history/007_v9.py"
        if kind == "kernel"
        else f"/run/history/001_{name}.py"
        for name, (kind, _) in items.items()
    }
    known = owners.Owners()
    known.note(
        [(kind, arg[name]) for name, (kind, _) in items.items()],
        _patches({owners.report_key(k, arg[n]): rec for n, (k, rec) in items.items()}),
    )

    def overlaps(name: str) -> dict[str, list[str]]:
        others = [(items[n][0], arg[n]) for n in items if n != name]
        found = known.overlapping((items[name][0], arg[name]), others)
        return {ledger.item_label(o[1]): where for o, where in found}

    # the LocEnc's CUDA graph; and dit_layer's kernels of the layers inside the LocEnc
    # (nested kernels: the outer one replaces what the inner one does)
    assert overlaps("loc_enc_decode") == {
        "graph_loc_enc": ["model.feat_encoder"],
        "dit_layer": ["model.feat_encoder.encoder.layers.*"],
    }
    # the layers' MLPs: not the CUDA graph of the solver around the layers
    assert overlaps("dit_layer") == {
        "loc_enc_decode": ["model.feat_encoder.encoder.layers.*"],
        "fp8_locdit_mlp": [dit[0] + ".mlp"],
    }
    assert set(overlaps("vae_decoder__reduced")) == {"vae_bf16", "vae_channels_last"}
    # what vae_channels_last replaced is its own: a transform inside it overlaps it
    assert set(overlaps("vae_bf16")) == {"vae_decoder__reduced", "vae_channels_last"}
    assert overlaps("graph_lm_step") == {"fused_lm_step": ["workload.lm_step"]}
    assert known.shared(arg["loc_enc_decode"], "/run/history/001_unknown.py") == []

    record = known.record([("transform", arg["graph_loc_enc"]), ("transform", "/x/none.py")])
    assert record == {arg["graph_loc_enc"]: {"touched": ["model.feat_encoder"], "owns": []}}
    line = owners.context_line(known.record([(k, arg[n]) for n, (k, _) in items.items()]))
    assert "`graph_loc_enc`: `model.feat_encoder`;" in line
    assert f"`vae_channels_last`: `{decoder}`;" in line  # the top-most module
    assert "loc_enc_decode" not in line  # a kernel: its target is listed already
    assert owners.context_line({}) == "" and owners.context_line(None) == ""
    names = ["model.layers.11.mlp", "model.layers.2.mlp", "m.conv1"]
    assert owners.compact(names) == ["m.conv1", "model.layers.*.mlp"]


# ------------------------------------------------------------------ the worker

#: A toy decoder harness for the worker.
HARNESS = """from toy_decoder import ToyDecoderWorkload


def create(spec):
    return ToyDecoderWorkload(spec)
"""

#: B's MLP runs out of memory (what an in-process A/B of a big composite does on the GPU).
OOM_TRANSFORM = f"""import torch


def apply(workload):
    def forward(x):
        raise torch.OutOfMemoryError("{OOM_TEXT}")

    workload.model.layers[0].mlp.forward = forward
"""


def test_the_worker_reports_running_out_of_memory_as_oom(
    tmp_path,
    capsys,
    cpu,  # noqa: F811
    monkeypatch,
):
    root = _run_dir(tmp_path, _file(tmp_path, "harness.py", HARNESS))
    _worker(capsys, "analyze", "--run-dir", root, "--no-profile", "--iters", 1)
    norm = f"norm={_file(tmp_path, 'norm.py', NORM)}"
    oom = _file(tmp_path, "oom.py", OOM_TRANSFORM)
    a, b = ["--kernel", norm], ["--b-kernel", norm, "--b-transform", oom]

    r = _worker(capsys, "e2e_ab", "--run-dir", root, "--rounds", 2, *a, *b)
    assert r["status"] == abtest.OOM and r["status"] in abtest.FALLBACK
    assert r["reason"] == f"out of GPU memory: torch.OutOfMemoryError: {OOM_TEXT}"
    norms = r["ab"]["a_patches"]["touched"]["norm"]  # what each state applied (owners.py)
    assert r["patches"]["touched"] == {"norm": norms, "oom": ["model.layers.*.mlp"]}
    r = _worker(capsys, "e2e", "--run-dir", root, "--transform", oom)
    assert r["status"] == abtest.OOM and r["patches"]["touched"] == {"oom": ["model.layers.*.mlp"]}
    assert ledger.classify(r, 1.0, e2e=True) == "oom"

    # B fails to apply in-process: what it applied before is reported
    broken = _file(tmp_path, "broken.py", "def apply(workload):\n    raise AttributeError('x')\n")
    r = _worker(capsys, "e2e_ab", "--run-dir", root, *a, *b[:2], "--b-transform", broken)
    assert r["status"] == "undo_failed" and r["patches"]["touched"]["broken"] == []
    assert "norm" in r["patches"]["touched"] and "norm" in r["ab"]["a_patches"]["touched"]
    fails = _file(tmp_path, "fails.py", "def apply(workload):\n    workload.model.norm = None\n")
    assert _worker(capsys, "e2e", "--run-dir", root, "--transform", fails)["status"] == (
        "runtime_error"
    )

    # the retry's allocator setting, before CUDA starts
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    _worker(capsys, "e2e", "--run-dir", root, "--transform", oom, "--expandable-segments")
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128,expandable_segments:True"
    assert abtest.out_of_memory(RuntimeError("CUDA error: out of memory")) is not None
    assert abtest.out_of_memory("CUDA error: an illegal memory access was encountered") is None


# ------------------------------------------------------------------ the integration

#: Latency factor of each item (product over a set): ``attn`` (a kernel: loc_enc_decode) replaces
#: the LocEnc that ``graph_attn`` (graph_loc_enc) CUDA-graphs, and beats it.
FACTOR = {"graph_attn": 0.80, "attn": 0.70, "lm": 0.90, "cfm": 0.60}
#: (touched, owns) the patcher records of each item.
MODULES = {
    "graph_attn": (["model.feat_encoder"], []),
    "attn": (["model.feat_encoder"], ["model.feat_encoder"]),
    "lm": (["workload.lm_step"], []),
    "cfm": (["model.feat_decoder"], []),
}


def oom_worker(calls: list[tuple[str, list[str]]], *, persist: bool = False):
    """An A/B with ``cfm`` and another item in both states runs out of memory in one
    process; with ``persist`` so does an ``e2e`` of three items. ``attn`` and ``graph_attn``
    cannot be applied together (the transform finds the kernel's module). Every result has
    the ``patches`` the worker records."""

    def patches(items: list[str]) -> dict:
        key = {i: label(i) if "=" in i else Path(i).stem for i in items}
        return _patches({key[i]: MODULES[label(i)] for i in items})

    def ms(items: list[str]) -> float:
        t = BASE
        for i in items:
            t *= FACTOR[label(i)]
        return t

    def worker(run, command, *args):
        ns = parse(args)
        calls.append((command, list(args)))
        a, b = [*ns.kernel, *ns.transform], [*ns.b_kernel, *ns.b_transform]
        if command == "e2e_ab" and len(a) > 1 and "cfm" in {label(i) for i in [*a, *b]}:
            reason = f"out of GPU memory: torch.OutOfMemoryError: {OOM_TEXT}"
            return {"status": abtest.OOM, "passed": False, "reason": reason, "error": OOM_TEXT}
        applied = a if command == "e2e" else b
        if {"attn", "graph_attn"} <= {label(i) for i in applied}:
            return {"status": "patch_error", "passed": False, "patches": patches(applied)}
        if command == "e2e":
            if persist and "--expandable-segments" in args and len(a) > 2:
                return {"status": abtest.OOM, "passed": False, "reason": "out of GPU memory"}
            r = dryrun._e2e_result(ms(a))
            r.update(
                times_ms=[round(ms(a) * w, 3) for w in [*WIGGLE, 1.0, 1.0]], patches=patches(a)
            )
            return r
        r = dryrun._e2e_result(ms(b))
        r["times_ms"] = [round(ms(b) * w, 3) for w in WIGGLE[: ns.rounds]]
        a_ms = [round(ms(a) * w, 3) for w in reversed(WIGGLE[: ns.rounds])]
        r["ab"] = {"mode": "paired", "a_ms": a_ms, "b_ms": r["times_ms"], "a_patches": patches(a)}
        r["patches"] = patches(b)
        return r

    return worker


def _setup(run) -> None:
    """The measured combination ``cfm`` + ``graph_attn`` seeds the search; ``attn`` (a kernel
    of the LocEnc) and ``lm`` were measured alone."""
    kernel(run, "attn", 3.0)
    e2e_record(run, 1.1, [write(run, "lm")])
    e2e_record(run, 2.0, [write(run, "cfm"), write(run, "graph_attn")])


def test_an_oom_step_is_measured_again_in_separate_processes(tmp_path):
    orch = make(tmp_path)
    run = orch.run
    _setup(run)
    calls: list[tuple[str, list[str]]] = []
    orch.worker = oom_worker(calls)
    asyncio.run(orch.integrate())

    data = read_json(run.root / "integration.json")
    assert [label(a["item"]) for a in data["accepted"]] == ["cfm", "attn", "lm"]
    steps = data["history"][4:]  # after the items alone
    assert [h.get("kind") for h in steps] == [None, None, "replace", None]
    seed, on_top, _, lm = steps
    assert seed["ab"]["mode"] == "paired"  # one item in A: it fitted
    # attn on top: out of memory in one process, then two that cannot apply it
    assert on_top["status"] == "patch_error" and on_top["ab"]["mode"] == "separate"
    assert on_top["ab"]["expandable_segments"] and "out of GPU memory" in on_top["ab"]["fallback"]
    # lm: both states hold what ran out of memory before: no in-process attempt
    assert "out of GPU memory before" in lm["ab"]["fallback"] and lm["ab"]["accepted"]
    commands = [c for c, _ in calls]
    assert commands[4:] == ["e2e_ab", *["e2e_ab", "e2e", "e2e"] * 2, "e2e", "e2e"]
    retries = [args for c, args in calls if c == "e2e"]
    assert len(retries) == 6 and all("--expandable-segments" in args for args in retries)
    assert data["final"]["median_ms"] == pytest.approx(BASE * 0.6 * 0.7 * 0.9, rel=1e-3)
    rows = [r for r in ledger.rows(run) if r["backend"] == "integrate"]
    assert "oom" not in {r["status"] for r in rows}  # measured in the end


def test_an_oom_that_persists_is_recorded_and_measured_again_later(tmp_path):
    orch = make(tmp_path)
    run = orch.run
    _setup(run)
    calls: list[tuple[str, list[str]]] = []
    orch.worker = oom_worker(calls, persist=True)
    asyncio.run(orch.integrate())

    data = read_json(run.root / "integration.json")
    (oom,) = [h for h in data["history"] if h["status"] == abtest.OOM]
    assert [label(i) for i in oom["items"]] == ["cfm", "attn", "lm"]  # in its own process too
    assert oom["reason"] == "out of GPU memory" and not oom["ab"]["accepted"]
    assert oom["ab"]["expandable_segments"]
    assert [label(a["item"]) for a in data["accepted"]] == ["cfm", "attn"]  # graph_attn replaced
    rows = [r for r in ledger.rows(run) if r["backend"] == "integrate"]
    assert [r["status"] for r in rows].count("oom") == 1

    calls.clear()  # a re-integration measures it again (not reused), reuses the rest
    asyncio.run(orch.integrate(reuse=True))
    assert read_json(run.root / "integration.json")["reuse"] == {"reused": 7, "measured": 1}
    measured = [
        {label(i) for i in [*parse(args).kernel, *parse(args).transform]} for _, args in calls
    ]
    assert [c for c, _ in calls] == ["e2e_ab", "e2e", "e2e"]  # in one process, then two
    assert measured[1:] == [{"cfm", "attn"}, {"cfm", "attn", "lm"}]


def test_a_kernel_that_owns_a_transforms_module_replaces_it(tmp_path):
    """``attn`` cannot go on top of ``graph_attn`` (both change ``model.feat_encoder``): it is
    measured in its place instead, accepted there (faster), and ``graph_attn`` is not tried
    again on top of it."""
    orch = make(tmp_path)
    run = orch.run
    _setup(run)
    calls: list[tuple[str, list[str]]] = []
    orch.worker = oom_worker(calls)
    asyncio.run(orch.integrate())

    data = read_json(run.root / "integration.json")
    (replace,) = [h for h in data["history"] if h.get("kind") == "replace"]
    assert [label(o) for o in replace["old"]] == ["graph_attn"] and label(replace["new"]) == "attn"
    assert [label(i) for i in replace["items"]] == ["cfm", "attn"]  # in its place
    assert [label(i) for i in replace["ab"]["a_items"]] == ["cfm", "graph_attn"]
    assert replace["ab"]["accepted"] and replace["ab"]["gain"] == pytest.approx(0.125, abs=2e-3)
    after = data["history"][data["history"].index(replace) + 1 :]
    assert [label(h["items"][-1]) for h in after] == ["lm"]  # graph_attn: not on top of attn
    owned = {label(k): v for k, v in data["owners"].items()}
    assert owned["attn"] == {"touched": ["model.feat_encoder"], "owns": ["model.feat_encoder"]}
    assert owned["lm"]["touched"] == ["workload.lm_step"]

    steps = charts.integration_steps(data)
    assert [s["kind"] for s in steps] == ["total", "keep", "fail", "keep", "keep", "total"]
    assert steps[3]["label"] == "attn for graph_attn" and steps[3]["replace"]
    assert steps[-1]["ms"] == pytest.approx(data["final"]["median_ms"])
    assert "* `attn` instead of `graph_attn`: passed=True" in write_report(run).read_text()

    # the next round's planner sees what the accepted transforms change
    state = {"rounds": [], "integrations": []}
    text = improve.rounds_context(run, state, 2, data["accepted"], [], owners=data["owners"])
    assert "Modules the accepted transforms change: `cfm`: `model.feat_decoder`" in text
    assert "`lm`: `workload.lm_step`" in text and "`attn`:" not in text
