"""The megakernel kit on the GPU (issue #225; ``gpu``, not in the CPU suite): every opcode of
the example against torch, the watchdog stopping a mis-targeted schedule, the trace, the
example through the evaluator and under memcheck, racecheck and synccheck."""

from __future__ import annotations

import copy
import dataclasses
import random

import pytest
import torch

from kernel_agent import toolchain
from kernel_agent.agent.prompts import EXAMPLES_DIR
from kernel_agent.native import project
from kernel_agent.native.megakernel import runtime, simulate
from kernel_agent.native.megakernel import schedule as mks

pytestmark = pytest.mark.gpu
EXAMPLE = EXAMPLES_DIR / "native_megakernel"


@pytest.fixture(scope="module")
def mk():
    """The example's entry module and its compiled extension."""
    toolchain.setup()
    module = project.import_project(EXAMPLE)
    return module, module.extension()


def _run(ext, sched, tensors, **kw):
    pages, _, page_bytes, _ = ext.mk_info()
    rt = runtime.Runtime(sched, tensors, pool_bytes=pages * page_bytes, **kw)
    ext.mk_run(*rt.args(), pages, sched.n_queues)
    torch.cuda.synchronize()
    return rt


def _rms(x, g, eps):
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return g * h.to(x.dtype)


def test_every_opcode_against_torch(mk):
    """RMSNorm, a GEMV with the norm fused and a residual, an FP8 GEMV split over K with a
    reduce, a residual add and an argmax: one schedule, chained by counters."""
    mod, ext = mk
    _, queues, _, _ = ext.mk_info()
    torch.manual_seed(0)
    n, k, eps, rows, splits = 512, 1024, 1e-6, 16, 4
    dev = dict(device="cuda")
    x = torch.randn(k, dtype=torch.bfloat16, **dev)
    g = (torch.randn(k, **dev) * 0.1 + 1).to(torch.bfloat16)
    w = (torch.randn(n, k, **dev) * k**-0.5).to(torch.bfloat16)
    w8 = (torch.randn(n, n, **dev) * 0.5).to(torch.float8_e4m3fn)
    scale = torch.rand(n, **dev) * 0.05 + 0.01
    # split-K packing done once (as build() would): [splits][n][n / splits]
    w8_split = w8.view(n, splits, n // splits).permute(1, 0, 2).contiguous()
    normed = torch.zeros(k, dtype=torch.bfloat16, **dev)
    y = torch.zeros(n, dtype=torch.bfloat16, **dev)
    part = torch.zeros(splits, n, **dev)
    z = torch.zeros(n, dtype=torch.bfloat16, **dev)
    s = torch.zeros(n, dtype=torch.bfloat16, **dev)
    best = torch.full((1,), -1, dtype=torch.int64, **dev)
    T = dict(x=0, g=1, w=2, w8=3, scale=4, normed=5, y=6, part=7, z=8, s=9, best=10)
    tensors = [x, g, w, w8_split, scale, normed, y, part, z, s, best]
    ks = n // splits
    ops = [
        mks.Op(
            "norm",
            mod.RMSNORM,
            1,
            args=[
                mod.rmsnorm_args(
                    x=T["x"],
                    x_off=0,
                    gamma=T["g"],
                    gamma_off=0,
                    eps=eps,
                    out=T["normed"],
                    out_off=0,
                    n=k,
                )
            ],
        ),
        mks.Op(
            "gemv",
            mod.GEMV,
            n // rows,
            args=lambda t: mod.gemv_args(
                x=T["x"],
                x_off=0,
                gamma=T["g"],
                eps=eps,
                out=T["y"],
                out_off=0,
                res=T["normed"],
                res_off=0,
                row0=t * rows,
                rows=rows,
                k=k,
            ),
            weights=lambda t: mks.Prefetch(T["w"], t * rows * k * 2, rows * k * 2),
        ),
        mks.Op(
            "fp8",
            mod.GEMV_FP8,
            splits * (n // 64),
            args=lambda t: mod.gemv_args(
                x=T["y"],
                x_off=0,
                out=-1,
                out_off=0,
                row0=(t % (n // 64)) * 64,
                rows=64,
                k=n,
                k0=(t // (n // 64)) * ks,
                klen=ks,
                part=T["part"],
                part_off=(t // (n // 64)) * n,
                scale=T["scale"],
                scale_off=0,
            ),
            weights=lambda t: mks.Prefetch(
                T["w8"], (t // (n // 64)) * n * ks + (t % (n // 64)) * 64 * ks, 64 * ks
            ),
        ),
        mks.Op(
            "reduce",
            mod.SPLITK_REDUCE,
            n // 128,
            args=lambda t: mod.reduce_args(
                part=T["part"],
                part_off=0,
                splits=splits,
                stride=n,
                row0=t * 128,
                rows=128,
                out=T["z"],
                out_off=0,
                res=T["y"],
                res_off=0,
            ),
        ),
        mks.Op(
            "add",
            mod.RESIDUAL,
            2,
            args=lambda t: mod.residual_args(
                a=T["z"],
                a_off=0,
                b=T["y"],
                b_off=0,
                out=T["s"],
                out_off=0,
                i0=t * (n // 2),
                n=n // 2,
            ),
        ),
        mks.Op(
            "argmax",
            mod.ARGMAX,
            1,
            args=[mod.argmax_args(x=T["s"], x_off=0, n=n, out=T["best"], out_off=0)],
        ),
    ]
    edges = [
        mks.Edge("norm", "gemv"),  # the residual is the normed vector (an extra dependency)
        mks.Edge("gemv", "fp8"),
        mks.Edge(
            "fp8",
            "reduce",
            lambda t: [s * (n // 64) + (t * 128) // 64 + h for s in range(splits) for h in (0, 1)],
        ),
        mks.Edge("gemv", "reduce"),
        mks.Edge("reduce", "add", lambda t: range(2 * t, 2 * t + 2)),
        mks.Edge("gemv", "add"),
        mks.Edge("add", "argmax"),
    ]
    sched = mks.build(ops, edges, queues)
    assert simulate.check(sched, draws=200) == []
    rt = _run(ext, sched, tensors)
    rt.check()
    want_normed = _rms(x, g, eps)
    assert torch.equal(normed, want_normed)
    want_y = (want_normed[:n].float() + (want_normed @ w.t()).float()).to(torch.bfloat16)
    torch.testing.assert_close(y, want_y, atol=2e-2, rtol=2e-2)
    want_part = torch.stack(
        [
            (y.float()[i * ks : (i + 1) * ks] @ w8.float()[:, i * ks : (i + 1) * ks].t()) * scale
            for i in range(splits)
        ]
    )
    torch.testing.assert_close(part, want_part, atol=1e-3, rtol=1e-3)
    want_z = (y.float() + part.sum(0).to(torch.bfloat16).float()).to(torch.bfloat16)
    assert torch.equal(z, want_z)
    assert torch.equal(s, (z.float() + y.float()).to(torch.bfloat16))
    assert int(best) == int(torch.argmax(s.float()))
    assert rt.counters.abs().sum() == 0  # the last block zeroed them for the next launch


def test_a_split_k_gemv_with_the_norm_fused_and_its_reduce(mk):
    """bf16 GEMV tiles over half of K each (the shared-memory path: the norm over all K, the
    slice normalised), fp32 partials, a reduce adding a residual."""
    mod, ext = mk
    _, queues, _, _ = ext.mk_info()
    torch.manual_seed(1)
    n, k, eps, rows, splits = 256, 2048, 1e-5, 32, 2
    x = torch.randn(k, device="cuda", dtype=torch.bfloat16)
    g = (torch.randn(k, device="cuda") * 0.1 + 1).to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda") * k**-0.5).to(torch.bfloat16)
    res = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    ks = k // splits
    w_split = w.view(n, splits, ks).permute(1, 0, 2).contiguous()  # [splits][n][ks]
    part = torch.zeros(splits, n, device="cuda")
    out = torch.zeros(n, dtype=torch.bfloat16, device="cuda")
    tiles = n // rows
    ops = [
        mks.Op(
            "gemv",
            mod.GEMV,
            splits * tiles,
            args=lambda t: mod.gemv_args(
                x=0,
                x_off=0,
                gamma=1,
                eps=eps,
                out=-1,
                out_off=0,
                row0=(t % tiles) * rows,
                rows=rows,
                k=k,
                k0=(t // tiles) * ks,
                klen=ks,
                part=3,
                part_off=(t // tiles) * n,
            ),
            weights=lambda t: mks.Prefetch(
                2, ((t // tiles) * n * ks + (t % tiles) * rows * ks) * 2, rows * ks * 2
            ),
        ),
        mks.Op(
            "reduce",
            mod.SPLITK_REDUCE,
            1,
            args=[
                mod.reduce_args(
                    part=3,
                    part_off=0,
                    splits=splits,
                    stride=n,
                    row0=0,
                    rows=n,
                    out=4,
                    out_off=0,
                    res=5,
                    res_off=0,
                )
            ],
        ),
    ]
    sched = mks.build(ops, [mks.Edge("gemv", "reduce")], queues)
    rt = _run(ext, sched, [x, g, w_split, part, out, res])
    rt.check()
    h = _rms(x, g, eps).float()
    want_part = torch.stack(
        [h[i * ks : (i + 1) * ks] @ w.float()[:, i * ks : (i + 1) * ks].t() for i in range(splits)]
    )
    torch.testing.assert_close(part, want_part, atol=1e-3, rtol=1e-3)
    assert torch.equal(out, (res.float() + part.sum(0).to(torch.bfloat16).float()).bfloat16())


def test_the_watchdog_stops_a_mistargeted_schedule_and_names_the_instruction(mk):
    mod, ext = mk
    _, queues, _, _ = ext.mk_info()
    buf = torch.zeros(1024, dtype=torch.bfloat16, device="cuda")
    ops = [mks.Op("a", mod.NOP, 8), mks.Op("b", mod.NOP, 8)]
    sched = mks.build(ops, [mks.Edge("a", "b")], queues)
    victim = next(i for i in sched.instrs if i.op == "b" and i.tile == 3)
    counter, target = victim.waits[0]
    broken = dataclasses.replace(victim, waits=((counter, target + 1),))
    bad = dataclasses.replace(
        sched, instrs=tuple(broken if i.id == victim.id else i for i in sched.instrs)
    )
    assert simulate.simulate(bad).deadlock
    rt = _run(ext, bad, [buf], watchdog_ms=50)
    info = rt.failure()
    assert info is not None and info["code"] == "hang", info
    assert (info["counter"], info["value"], info["target"]) == (counter, target, target + 1)
    assert info["instr"] in {i.id for i in bad.instrs if i.op == "b"}
    with pytest.raises(runtime.MegakernelHang, match="megakernel hang"):
        rt.check()
    assert rt.counters.abs().sum() == 0  # zeroed after the abort as well
    good = _run(ext, sched, [buf])
    good.check()


def _chain(hidden=1024, layers=8):
    from kernel_agent.selftest import NormGemvChain

    torch.manual_seed(0)
    ref = NormGemvChain(hidden, layers).cuda().to(torch.bfloat16).eval()
    with torch.no_grad():
        for name, weight in ref.named_parameters():
            weight.normal_(1.0, 0.1) if name.startswith("norms.") else weight.normal_(
                0.0, hidden**-0.5
            )
    return ref


@pytest.mark.parametrize("mode", ["megakernel", "graph_pdl", "coop_barrier"])
def test_every_mode_of_the_example_matches_the_reference(mk, mode):
    mod, _ = mk
    ref = _chain()
    x = torch.randn(1, 1024, device="cuda", dtype=torch.bfloat16)
    engine = mod.build(ref, mode=mode)
    assert engine is not ref
    with torch.no_grad():
        want, got = ref(x), engine(x)
    torch.testing.assert_close(got, want, atol=0.1, rtol=0.05)


def test_the_trace_shows_weight_loads_overlapping_counter_waits(mk):
    mod, _ = mk
    engine = mod.build(_chain(layers=16), mode="megakernel", trace=True)
    x = torch.randn(1, 1024, device="cuda", dtype=torch.bfloat16)
    for _ in range(20):
        engine(x)
    torch.cuda.synchronize()
    summary = engine.trace_summary()
    later = [stats for op, stats in summary["ops"].items() if op != "layer0"]
    assert summary["span_ns"] > 0 and later
    assert min(s["overlap"] for s in later) > 0.9, summary  # issued before the counters met


def test_the_example_passes_the_evaluator(tmp_path):
    from kernel_agent.selftest import smoke_megakernel

    assert smoke_megakernel(tmp_path, verbose=True)


def test_the_evaluator_stresses_the_example_back_to_back(tmp_path):
    """Issue #248: the example uses counters, so the evaluator's determinism stage runs its
    main case STRESS_CALLS times back to back, every call bit-identical to an isolated one."""
    from kernel_agent.kernels import evaluate as ev
    from kernel_agent.selftest import MEGAKERNEL_EXAMPLES, make_norm_chain_capture

    hidden, layers, calls = MEGAKERNEL_EXAMPLES["native_megakernel"]
    capture = make_norm_chain_capture(tmp_path / "chain.pt", hidden, layers, calls)
    result = ev.run_evaluation(capture, EXAMPLE)
    assert result["correct"], result
    stress = result["determinism"]["stress"]
    assert (stress["calls"], stress["wrong"]) == (ev.STRESS_CALLS, 0), stress


#: Redrawn inputs of the 8-layer chain in the timed-output test (#250; 1.6-1.8 % of them reach
#: past the plain exact tolerance for each mode) and evaluations of the 8-layer example.
DEEP_DRAWS = 1000
DEEP_EVALUATIONS = 10


def test_deep_chains_pass_the_timed_output_check_on_every_draw(mk):
    """Issue #250: at 8 layers 1.6-1.8 % of redrawn inputs put the megakernel and its graph + PDL
    and grid-barrier baselines past the exact tier's plain tolerance against eager (cuBLAS
    rounding). The timed-output check judges such a draw again with the reference's own
    rounding spread on it: none is rejected. What the check is for still fails on every draw
    it is tried on: the previous draw's output (cached by shape or address), a stale 16-row
    tile, the last layer skipped."""
    from kernel_agent.kernels import bench, compare, verify

    mod, _ = mk
    ref = _chain(layers=8)
    engines = {m: mod.build(ref, mode=m) for m in ("megakernel", "graph_pdl", "coop_barrier")}
    gen = torch.Generator(device="cuda").manual_seed(250)
    captured = torch.randn(1, 1024, device="cuda", dtype=torch.bfloat16, generator=gen)
    plain = dict.fromkeys(engines, 0)
    tried = 0
    previous = None
    for i in range(DEEP_DRAWS):
        args = (captured.clone(),)
        verify.perturb_(args, gen, "normal")
        pre = (copy.deepcopy(args), {})
        with torch.inference_mode():
            expected = ref(*args)
            outs = {m: engine(*args) for m, engine in engines.items()}
        for m, out in outs.items():
            kept = {"iteration": i, "pre": pre, "post": pre, "output": out}
            assert bench.check_timed_output(ref, kept)["failures"] == [], (m, i)
            plain[m] += not compare.compare_tensors("output", expected, out, perturbed=True)["ok"]
        if previous is not None and i % 20 == 0:
            stale = outs["megakernel"].clone()
            stale[:, 16:32] = previous[:, 16:32]
            with torch.inference_mode():
                skipped = args[0]
                for norm, layer in zip(ref.norms[:-1], ref.layers[:-1], strict=True):
                    skipped = skipped + layer(norm(skipped))
            for what, out in (("previous", previous), ("stale", stale), ("skipped", skipped)):
                kept = {"iteration": i, "pre": pre, "post": pre, "output": out}
                assert bench.check_timed_output(ref, kept)["failures"], (what, i)
            tried += 1
        previous = outs["megakernel"]
    assert tried == DEEP_DRAWS // 20 - 1
    assert sum(plain.values()) > 0, plain  # the draws do reach past the plain tolerance
    print(f"\n  past the plain tolerance (of {DEEP_DRAWS} draws): {plain}; rejected: none")


def _deep_capture(tmp_path):
    """The example's capture (``MEGAKERNEL_EXAMPLES``) at 8 layers."""
    from kernel_agent.selftest import MEGAKERNEL_EXAMPLES, make_norm_chain_capture

    hidden, _, calls = MEGAKERNEL_EXAMPLES["native_megakernel"]
    return make_norm_chain_capture(tmp_path / "chain8.pt", hidden, 8, calls)


def test_the_example_passes_the_evaluator_at_eight_layers_every_time(tmp_path):
    """Issue #250: with the reference's own rounding spread the example passes every
    evaluation of an 8-layer capture (before: 4 of 100 refused, 2 at the timed output, 2 at
    the perturbed inputs)."""
    from kernel_agent.kernels import evaluate as ev

    capture = _deep_capture(tmp_path)
    for i in range(DEEP_EVALUATIONS):
        result = ev.run_evaluation(capture, EXAMPLE)
        assert result["correct"], (i, result.get("status"), result.get("error"))


#: The example's engine behind an output cache (#250: still rejected on the 8-layer chain).
CHEATING_ENGINES = {
    "memo": """
from torch import nn

from kernel_agent.agent.prompts import EXAMPLES_DIR
from kernel_agent.native import project

_EXAMPLE = project.import_project(EXAMPLES_DIR / "native_megakernel")


class Memo(nn.Module):
    '''The engine's output memoised by input shape.'''

    def __init__(self, engine):
        super().__init__()
        self.engine, self.memo = engine, {}

    def forward(self, x):
        if x.shape not in self.memo:
            self.memo[x.shape] = self.engine(x)
        return self.memo[x.shape].clone()


def build(reference):
    return Memo(_EXAMPLE.build(reference))
""",
    "stale": """
from torch import nn

from kernel_agent.agent.prompts import EXAMPLES_DIR
from kernel_agent.native import project

_EXAMPLE = project.import_project(EXAMPLES_DIR / "native_megakernel")


class Stale(nn.Module):
    '''One call behind: the previous call's output of the same shape (the first its own).'''

    def __init__(self, engine):
        super().__init__()
        self.engine, self.last = engine, {}

    def forward(self, x):
        out = self.engine(x)
        last = self.last.get(x.shape)
        self.last[x.shape] = out
        return out if last is None else last


def build(reference):
    return Stale(_EXAMPLE.build(reference))
""",
}


@pytest.mark.parametrize("kind", sorted(CHEATING_ENGINES))
def test_cached_and_stale_engines_are_still_rejected_at_eight_layers(tmp_path, kind):
    from kernel_agent.kernels import evaluate as ev

    capture = _deep_capture(tmp_path)
    candidate = tmp_path / f"{kind}.py"
    candidate.write_text(CHEATING_ENGINES[kind])
    result = ev.run_evaluation(capture, candidate)
    assert not result["correct"], result
    assert result["status"] in ("incorrect_timed_output", "incorrect_perturbed"), result


def _stress(launch, xs, golden, out, calls, seed=0):
    """``calls`` launches back to back on the inputs ``xs`` in turn (a GPU-side gap of up to
    ~0.1 ms before a quarter of them), each output compared on the GPU with ``golden``: the
    indices of the calls that differ in a bit."""
    rng = random.Random(seed)
    flags = []
    for i in range(calls):
        k = i % len(xs)
        if rng.random() < 0.25:
            torch.cuda._sleep(rng.randrange(1_000, 300_000))
        launch(xs[k])
        flags.append((out() != golden[k]).any())
    torch.cuda.synchronize()
    return [i for i, f in enumerate(flags) if bool(f)]


@pytest.mark.parametrize("inflight", [1, 0])
def test_two_thousand_back_to_back_calls_equal_isolated_ones(mk, inflight):
    """Counters reset by the last block, pages and activation rows reused by every launch:
    2,000 calls on 8 inputs, back to back and with gaps, each bit for bit the isolated call's."""
    mod, _ = mk
    engine = mod.build(_chain(layers=28), mode="megakernel", inflight=inflight)
    xs = [torch.randn(1, 1024, device="cuda", dtype=torch.bfloat16) for _ in range(8)]
    with torch.inference_mode():
        golden = []
        for x in xs:
            torch.cuda.synchronize()
            golden.append(engine(x))
            torch.cuda.synchronize()
        held = {}

        def launch(x):
            held["out"] = engine(x)

        wrong = _stress(launch, xs, golden, lambda: held["out"], 2000)
    assert wrong == [], f"{len(wrong)} of 2000 calls differ, the first: {wrong[:5]}"
    engine.rt.check()


def test_the_stress_catches_consumers_that_do_not_wait(mk):
    """The same stress on a broken schedule whose tiles do not wait for the previous layer:
    they read rows before the previous layer wrote them, i.e. what the previous call left, and
    the stress sees it. (A target of 1 of 64 was not enough here: each SM runs its own tile
    of layer l right before its tile of layer l + 1, by when the other tiles of layer l are
    done as well.)"""
    mod, ext = mk
    engine = mod.build(_chain(layers=8), mode="megakernel")
    sched = engine.schedule
    early = tuple(dataclasses.replace(i, waits=()) for i in sched.instrs)
    broken = dataclasses.replace(sched, instrs=early)
    assert simulate.check(broken, draws=50)  # early starts the simulator reports too
    pages, queues, page_bytes, _ = ext.mk_info()
    rt = runtime.Runtime(
        broken, [*engine._weights, *engine._gammas, engine._act], pool_bytes=pages * page_bytes
    )
    xs = [torch.randn(1, 1024, device="cuda", dtype=torch.bfloat16) for _ in range(8)]
    with torch.inference_mode():
        golden = [engine(x) for x in xs]  # the correct schedule's outputs

        def launch(x):
            engine._act[0].copy_(x.reshape(-1))
            ext.mk_run(*rt.args(), pages, queues, 1)

        wrong = _stress(launch, xs, golden, lambda: engine._act[-1].view(1, -1).clone(), 500)
    assert len(wrong) > 0, "the stress saw no stale row"
    print(f"\n  broken schedule: {len(wrong)} of 500 calls wrong")


def test_the_example_is_clean_under_memcheck_racecheck_and_synccheck(tmp_path):
    from kernel_agent.kernels.memcheck import run_memcheck
    from kernel_agent.selftest import make_norm_chain_capture

    tool = toolchain.sanitizer()
    if not tool.path:
        pytest.skip(tool.reason)
    capture = make_norm_chain_capture(tmp_path / "chain.pt", 512, 2, [((1,), 4)])
    result = run_memcheck(capture, EXAMPLE, tool=tool, timeout=1200)
    assert result["status"] == "ok", result
    assert [result["tools"][t]["status"] for t in ("memcheck", "racecheck", "synccheck")] == [
        "ok",
        "ok",
        "ok",
    ], result["tools"]
