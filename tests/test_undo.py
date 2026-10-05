"""Undo handles of kernel replacements and transforms, the in-process A/B session and
``worker e2e_ab`` (issue #11). CPU only: the toy decoder of the region tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch._inductor.config as inductor_config
import toy_decoder
from torch import nn
from toy_decoder import FUSED_CANDIDATE, GOOD_REWRITE, ToyDecoderWorkload

from kernel_agent import abtest, toolchain, worker
from kernel_agent.integrate import ab, undo
from kernel_agent.integrate.patcher import (
    KernelPatch,
    PatchReport,
    apply_kernels,
    apply_transforms,
)
from kernel_agent.region import Rewrite
from kernel_agent.workloads.base import WorkloadSpec
from kernel_agent.workloads.holdout import outputs_equal
from kernel_agent.workspace import write_json

TESTS = Path(__file__).parent

NORM = """from torch import nn


class Norm(nn.Module):
    def __init__(self, ref):
        super().__init__()
        self.weight = ref.weight
        self.variance_epsilon = ref.variance_epsilon

    def forward(self, x):
        y = x.float()
        y = y * (y.pow(2).mean(-1, keepdim=True) + self.variance_epsilon).rsqrt()
        return self.weight * y.to(x.dtype)


def build(ref):
    return Norm(ref)
"""

#: Everything a transform may change that the snapshot sees, and a lazily built cache.
TRANSFORM = """import torch
import torch._inductor.config as inductor_config
import toy_decoder
from torch import nn

CAPTURES = []


class Scaled(toy_decoder.ToyAttention):
    pass


class Wrapped(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x)


def apply(workload):
    model = workload.model
    forward, cache = model.forward, {}

    def cached_forward(ids):  # what a lazily captured CUDA graph looks like
        if ids.shape not in cache:
            CAPTURES.append(ids.shape)
            cache[ids.shape] = True
        return forward(ids)

    model.forward = cached_forward
    model.norm = Wrapped(model.norm)
    model.layers[0].self_attn.__class__ = Scaled
    model.layers[1].mlp.register_buffer("packed", torch.ones(2))
    with torch.no_grad():
        model.lm_head.weight.data = model.lm_head.weight.data.clone()
    sdpa = torch.nn.functional.scaled_dot_product_attention
    torch.nn.functional.scaled_dot_product_attention = lambda *a, **k: sdpa(*a, **k)
    toy_decoder.ToyRMSNorm.extra_repr = lambda self: "patched"
    toy_decoder.PATCHED = True
    torch.backends.cudnn.benchmark = not torch.backends.cudnn.benchmark
    inductor_config.max_autotune = not inductor_config.max_autotune
"""

#: A Biased toy decoder: ``bias`` (a list) is not seen by the snapshot.
BIASED = """from toy_decoder import ToyDecoderWorkload


class Biased(ToyDecoderWorkload):
    def load(self):
        super().load()
        self.bias = [0.0]

    def run(self, inputs):
        out = super().run(inputs)
        return {**out, "first_logits": out["first_logits"] + self.bias[0]}


def create(spec):
    return Biased(spec)
"""


@pytest.fixture
def toy() -> ToyDecoderWorkload:
    workload = ToyDecoderWorkload(
        WorkloadSpec(repo_id="toy/decoder", modality="llm", device="cpu", dtype="float32")
    )
    workload.load()
    return workload


def _file(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(source)
    return path


def _out(workload) -> dict:
    with torch.inference_mode():
        return workload.run(workload.make_inputs())


def _modules(model: nn.Module) -> dict[str, nn.Module]:
    return dict(model.named_modules())


def test_kernel_handles_restore_the_exact_modules(tmp_path, toy):
    model = toy.model
    before, ref = _modules(model), _out(toy)
    rewrite = Rewrite("resid_norm", "ToyDecoderLayer", _file(tmp_path, "rw.py", GOOD_REWRITE))
    fused = _file(tmp_path, "fused.py", FUSED_CANDIDATE)
    patches = [
        KernelPatch("norm", "ToyRMSNorm", _file(tmp_path, "norm.py", NORM)),
        KernelPatch("fused", "Region_resid_norm", fused, rewrite=rewrite),
    ]
    handles: list[undo.Undo] = []
    report = apply_kernels(toy.roots(), patches, handles=handles)
    assert report.rewritten == {"resid_norm": 2} and report.replaced["fused"] == 2
    assert type(model.layers[0]).__name__ == "RegionDecoderLayer"
    applied = _modules(model)
    out = _out(toy)

    (handle,) = handles
    assert handle.reversible and handle.summary()["changes"] == {"item": 3}  # 2 layers, norm
    handle.undo()
    assert _modules(model).keys() == before.keys()
    assert all(_modules(model)[k] is m for k, m in before.items())  # the very same objects
    assert outputs_equal(_out(toy), ref)
    handle.redo()
    assert all(_modules(model)[k] is m for k, m in applied.items())
    assert outputs_equal(_out(toy), out)


def test_phase_routed_kernel_is_undone(tmp_path, toy):
    """Phase routing keeps the instance and adds instance attributes: gone after undo."""

    attn = toy.model.layers[0].self_attn
    source = "import copy\n\n\ndef build(ref):\n    return copy.copy(ref)\n"
    copy = _file(tmp_path, "copy.py", source)
    handles: list[undo.Undo] = []
    patch = KernelPatch("attn", "ToyAttention", copy, phase="prefill", methods=["forward"])
    apply_kernels(toy.roots(), [patch], handles=handles)
    assert "forward" in vars(attn) and toy.model.layers[0].self_attn is attn
    handles[0].undo()
    assert "forward" not in vars(attn)
    handles[0].redo()
    assert getattr(vars(attn)["forward"], "ka_phase", None) == "prefill"


def test_transform_handles_restore_wrapped_attributes(tmp_path, toy):
    model = toy.model
    before, ref = _modules(model), _out(toy)
    sdpa = torch.nn.functional.scaled_dot_product_attention
    flags = (torch.backends.cudnn.benchmark, inductor_config.max_autotune)
    params = {n: (p, p.data_ptr()) for n, p in model.named_parameters()}
    path = _file(tmp_path, "tf.py", TRANSFORM)
    handles: list[undo.Undo] = []
    report = apply_transforms(toy, [path], PatchReport(), handles=handles)
    assert report.transforms == ["tf"]
    (handle,) = handles
    assert handle.reversible and not handle.reapply
    kinds = handle.summary()["changes"]
    assert {"item", "class", "data", "attr", "flag", "config"} <= kinds.keys()
    capture = sys_module("tf").CAPTURES
    wrapped = vars(model)["forward"]
    out = _out(toy)
    assert outputs_equal(out, ref) and type(model.norm).__name__ == "Wrapped"
    n = len(capture)

    handle.undo()
    assert "forward" not in vars(model) and model.norm is before["norm"]
    assert type(model.layers[0].self_attn) is toy_decoder.ToyAttention
    assert "packed" not in model.layers[1].mlp._buffers
    for name, param in model.named_parameters():
        assert param is params[name][0] and param.data_ptr() == params[name][1]
    assert torch.nn.functional.scaled_dot_product_attention is sdpa
    assert "extra_repr" not in vars(toy_decoder.ToyRMSNorm) and not hasattr(toy_decoder, "PATCHED")
    assert (torch.backends.cudnn.benchmark, inductor_config.max_autotune) == flags
    assert all(_modules(model)[k] is m for k, m in before.items())
    assert outputs_equal(_out(toy), ref)

    handle.redo()  # the same wrapper, its cache intact: nothing captured again
    assert vars(model)["forward"] is wrapped
    assert outputs_equal(_out(toy), out) and len(capture) == n
    handle.undo()
    assert torch.nn.functional.scaled_dot_product_attention is sdpa


def sys_module(stem: str):
    import sys

    return next(m for n, m in sys.modules.items() if n.startswith(f"ka_transform_{stem}_"))


def test_irreversible_transforms_are_detected(tmp_path, toy):
    inplace = _file(
        tmp_path,
        "inplace.py",
        "import torch\n\n\ndef apply(workload):\n"
        "    with torch.no_grad():\n        workload.model.lm_head.weight.mul_(1.0)\n",
    )
    declared = _file(tmp_path, "declared.py", "undo = False\n\n\ndef apply(workload):\n    pass\n")
    custom = _file(
        tmp_path,
        "custom.py",
        "import torch\n\nREVERTED = []\n\n\ndef apply(workload):\n"
        "    with torch.no_grad():\n        workload.model.lm_head.weight.mul_(2.0)\n\n\n"
        "def undo(workload):\n    REVERTED.append(1)\n"
        "    with torch.no_grad():\n        workload.model.lm_head.weight.div_(2.0)\n",
    )
    assert [undo.declaration(p) for p in (inplace, declared, custom)] == [None, False, True]
    handles: list[undo.Undo] = []
    apply_transforms(toy, [inplace, declared], PatchReport(), handles=handles)
    assert ["modified in place" in p for h in handles for p in h.problems] == [True, False]
    assert "undo = False" in handles[1].problems[0]
    with pytest.raises(undo.IrreversibleError):
        handles[0].undo()

    ref = _out(toy)
    handles.clear()
    apply_transforms(toy, [custom], PatchReport(), handles=handles)
    assert handles[0].reversible and handles[0].reapply
    handles[0].undo()
    assert sys_module("custom").REVERTED == [1] and outputs_equal(_out(toy), ref)
    with pytest.raises(undo.IrreversibleError):
        handles[0].redo()  # applied again instead (Session)


def test_session_switches_states_and_shares_kernels(tmp_path, toy):
    norm = _file(tmp_path, "norm.py", NORM)
    tf = _file(tmp_path, "tf.py", TRANSFORM)
    patches = {"norm": KernelPatch("norm", "ToyRMSNorm", norm)}
    session = ab.Session(toy, lambda items: [patches[i.partition("=")[0]] for i in items])
    pristine = _out(toy)
    a = session.build("A", [f"norm={norm}"], [])
    a_out = _out(toy)
    b = session.build("B", [f"norm={norm}"], [str(tf)], on=a)
    assert b.shared == 1 and b.handles[0] is a.handles[0]
    assert b.irreversible() == [] and a.irreversible(keep=1) == []
    b_out = _out(toy)
    captures = sys_module("tf").CAPTURES
    n = len(captures)
    for state, out in [(a, a_out), (b, b_out), (a, a_out), (b, b_out)]:
        assert not session.to(state)
        assert outputs_equal(_out(toy), out)
    assert len(captures) == n  # B's lazily built cache survived the switches
    session.to(ab.State("pristine", [], []))
    assert outputs_equal(_out(toy), pristine) and session.applied == []
    # a region rewrite must come before every kernel: B with one does not share A's kernels
    rewrite = Rewrite("resid_norm", "ToyDecoderLayer", _file(tmp_path, "rw.py", GOOD_REWRITE))
    fused = _file(tmp_path, "fused.py", FUSED_CANDIDATE)
    patches["fused"] = KernelPatch("fused", "Region_resid_norm", fused, rewrite=rewrite)
    assert session.shareable(a, [f"norm={norm}", f"fused={fused}"]) == 0
    assert session.shareable(a, [f"norm={norm}"]) == 1 and session.shareable(a, []) == 0


def _run_dir(tmp_path: Path, harness: Path) -> Path:
    root = tmp_path / "run"
    spec = WorkloadSpec(
        repo_id="toy/decoder", modality="llm", device="cpu", dtype="float32", harness=str(harness)
    )
    write_json(root / "run.json", {"workload": spec.to_dict()})
    write_json(
        root / "targets" / "norm" / "spec.json",
        {"module_class": "ToyRMSNorm", "capture": {"method_instances": {"forward": 5}}},
    )
    return root


def _worker(capsys, *argv) -> dict:
    worker.main([str(a) for a in argv])
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith(worker.MARKER)][-1]
    return json.loads(line[len(worker.MARKER) :])


@pytest.fixture
def cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(toolchain, "setup", lambda *a, **k: None)


def test_worker_e2e_ab_on_the_toy_decoder(tmp_path, capsys, cpu):
    harness = _file(tmp_path, "biased.py", BIASED)
    root = _run_dir(tmp_path, harness)
    baseline = _worker(capsys, "analyze", "--run-dir", root, "--no-profile", "--iters", 1)
    assert baseline["deterministic"]
    norm = _file(tmp_path, "norm.py", NORM)
    tf = _file(tmp_path, "tf.py", TRANSFORM.replace("toy_decoder.PATCHED = True", "pass"))

    a = ["--kernel", f"norm={norm}"]
    b = ["--b-kernel", f"norm={norm}", "--b-transform", tf]
    r = _worker(capsys, "e2e_ab", "--run-dir", root, "--rounds", 4, "--warmup", 1, *a, *b)
    assert r["status"] == "ok" and r["passed"], r
    rec = r["ab"]
    assert rec["mode"] == "paired" and rec["rounds"] == 4 and rec["shared_kernels"]
    assert rec["order"] == "A B B A A B B A" and len(rec["a_ms"]) == len(rec["b_ms"]) == 4
    assert rec["undo_check"] == {"A": "identical", "B": "identical"}
    assert r["times_ms"] == rec["b_ms"] and r["patches"]["replaced"] == {"norm": 5}
    assert r["metrics"]["holdout"]["passed"]
    judged = abtest.judge(rec)
    assert set(judged) >= {"accepted", "why", "ci95", "win_rate"}

    # B leaks into A through what the snapshot cannot see: detected, not measured
    leak = _file(tmp_path, "leak.py", "def apply(workload):\n    workload.bias[0] += 1e-3\n")
    r = _worker(capsys, "e2e_ab", "--run-dir", root, "--rounds", 2, "--b-transform", leak)
    assert r["status"] == "undo_failed" and "changed after switching" in r["reason"]

    inplace = _file(
        tmp_path,
        "inplace.py",
        "import torch\n\n\ndef apply(workload):\n"
        "    with torch.no_grad():\n        workload.model.lm_head.weight.mul_(1.0)\n",
    )
    r = _worker(capsys, "e2e_ab", "--run-dir", root, "--b-transform", inplace)
    assert r["status"] == "irreversible" and r["irreversible"] == [str(inplace)]
    assert "modified in place" in r["reason"]
    declared = _file(tmp_path, "declared.py", "undo = False\n\n\ndef apply(workload):\n    pass\n")
    r = _worker(capsys, "e2e_ab", "--run-dir", root, "--transform", declared)
    assert r["status"] == "irreversible" and "declares undo = False" in r["reason"]
    assert r["status"] in abtest.FALLBACK


# ------------------------------------------------------------------ VoxCPM2 on the GPU

#: The VoxCPM2 run's graph_cfm_solver transform (CUDA graph captured on the first call),
#: counting its captures.
GRAPH_CFM = """import torch

CAPTURES = []


def apply(workload):
    cfm = workload.model.feat_decoder
    orig = cfm.solve_euler
    graphs = {}

    def solve_euler(x, t_span, mu, cond, cfg_value=1.0, use_cfg_zero_star=True):
        if not x.is_cuda or torch.is_grad_enabled():
            kw = {"cfg_value": cfg_value, "use_cfg_zero_star": use_cfg_zero_star}
            return orig(x, t_span, mu, cond, **kw)
        args = (x, t_span, mu, cond)
        key = (tuple((a.shape, a.dtype) for a in args), float(cfg_value), bool(use_cfg_zero_star))
        if key not in graphs:
            static = [a.clone() for a in args]

            def run(*a):
                return orig(*a, cfg_value=cfg_value, use_cfg_zero_star=use_cfg_zero_star)

            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                run(*static)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = run(*static)
            graphs[key] = (graph, static, out)
            CAPTURES.append(key)
        graph, static, out = graphs[key]
        for s, a in zip(static, args):
            s.copy_(a)
        graph.replay()
        return out.clone()

    cfm.solve_euler = solve_euler
"""

#: The run's merge_gate_up: gate/up weights rebound (``.data``) into one packed buffer.
MERGE_GATE_UP = """import torch
import torch.nn.functional as F
from voxcpm.modules.minicpm4.model import MiniCPMMLP


def _merged_forward(self, x):
    gate, up = F.linear(x, self.gate_up_weight).split(self.intermediate_size, dim=-1)
    return self.down_proj(self.act_fn(gate) * up)


def apply(workload):
    for module in workload.model.modules():
        if type(module) is not MiniCPMMLP or module.gate_proj.bias is not None:
            continue
        with torch.no_grad():
            w = torch.cat([module.gate_proj.weight, module.up_proj.weight], 0).contiguous()
            module.gate_proj.weight.data = w[: module.intermediate_size]
            module.up_proj.weight.data = w[module.intermediate_size :]
        module.register_buffer("gate_up_weight", w, persistent=False)
        module.forward = _merged_forward.__get__(module)
"""


def _voxcpm2_cached() -> bool:
    import importlib.util

    if importlib.util.find_spec("voxcpm") is None:
        return False
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache("openbmb/VoxCPM2", "config.json"), str)


@pytest.mark.gpu
@pytest.mark.skipif(not _voxcpm2_cached(), reason="needs voxcpm + openbmb/VoxCPM2 in the HF cache")
def test_voxcpm2_graph_transform_undo_and_redo(tmp_path):
    """A transform that captures CUDA graphs lazily: undo restores the original method,
    redo the wrapper with its graphs (nothing captured again); `.data` rebinding too."""
    from kernel_agent.workloads import create_workload

    spec = WorkloadSpec(
        repo_id="openbmb/VoxCPM2", modality="tts", family="voxcpm", options={"patches": 6}
    )
    wl = create_workload(spec)
    wl.load()
    cfm = wl.model.feat_decoder
    _out(wl)  # the first run differs (lazy initialisation), as e2e_ab's warm-up allows for
    pristine = _out(wl)
    assert outputs_equal(_out(wl), pristine)  # bit-reproducible: the undo check applies

    handles: list[undo.Undo] = []
    paths = [_file(tmp_path, "graph_cfm.py", GRAPH_CFM), _file(tmp_path, "merge.py", MERGE_GATE_UP)]
    apply_transforms(wl, paths, PatchReport(), handles=handles)
    assert all(h.reversible for h in handles), [h.problems for h in handles]
    assert {"item", "data"} <= handles[1].summary()["changes"].keys()
    wrapper = vars(cfm)["solve_euler"]
    _out(wl)  # captures the graphs
    graphed = _out(wl)
    captures = sys_module("graph_cfm").CAPTURES
    n = len(captures)
    assert n >= 1

    for handle in reversed(handles):
        handle.undo()
    assert "solve_euler" not in vars(cfm) and cfm.solve_euler.__func__ is type(cfm).solve_euler
    mlps = [m for m in wl.model.modules() if type(m).__name__ == "MiniCPMMLP"]
    assert not any("gate_up_weight" in m._buffers or "forward" in vars(m) for m in mlps)
    assert outputs_equal(_out(wl), pristine)

    for handle in handles:
        handle.redo()
    assert vars(cfm)["solve_euler"] is wrapper
    assert outputs_equal(_out(wl), graphed) and len(captures) == n  # graphs replayed
