"""The perturbed-input check of the reduced-precision tiers (#109): the reference math of
every reduced precision passes both of the evaluator's checks (captured inputs, and inputs
redrawn from each channel's mean and std by ``kernels.verify``, #198) on a synthetic capture
with a massive-activation writer row (VoxCPM2's LocDiT o_proj / down_proj row 497); where
a redrawn input has lost the outlier channel (a single token is drawn from the tensor's
statistics) the bounds of redrawn inputs scale the element bound per channel
(``_wide_channel`` below); broken scales fail.
Both quality modes with reduced precision: ``near-lossless`` and ``relaxed`` (#175, about
twice the error budgets), where blatant bugs (int4 per tensor, scales x 1.2, a skipped
output row, gate / up swapped) still fail. INT8 (#178: ``int8_weights``, ``int8_w8a8``)
shares the 8-bit tiers and is calibrated here the same way, W4A4 (#233: ``fp4_w4a4``) its
own near-lossless-fp4a / relaxed-fp4a tiers."""

import pytest
import torch
from torch import nn

from kernel_agent.agent import prompts
from kernel_agent.kernels import bench, compare, verify
from kernel_agent.kernels.evaluate import evaluate
from kernel_agent.kernels.recheck import compare_entries
from kernel_agent.profiling.capture import capture_calls

# The reference math of each reduced precision (no kernel; ``reduced`` has none). A new
# precision adds its own here, so its tier is calibrated on both input classes.
CANDIDATE = """import torch
from kernel_agent.kernels import quant

PRECISION, BUG = {precision!r}, {bug!r}
RULE = "floor" if BUG == "floor scale rule" else "ceil"


def int4_per_tensor(w):
    scale = w.float().abs().amax() / 7
    return (torch.round(w.float() / scale).clamp(-8, 7) * scale).to(w.dtype)


def quantize_activations(x):  # fp8_mx: the scale-rule guard's hook
    return quant.quantize_mxfp8(x, RULE)


PRECISION_OF = {{}}
ACTIVATION_BUGS = ("first token's scale", "cached activation scales", "static activation scale")


class Linear(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        w = linear.weight.detach().clone()
        if BUG == "row 0 skipped":  # an off-by-one row loop: output channel 0 never written
            w[0] = 0
        if BUG == "int4 per tensor":
            self.w = int4_per_tensor(w)
            PRECISION_OF[id(self)] = "int4"
            return
        if PRECISION == "fp8_mx":
            self.q, self.s = quant.quantize_mxfp8(w)
            if BUG == "neighbour scale":
                self.s = self.s.roll(1, dims=0)
            return
        if PRECISION in ("int8_weights", "int8_w8a8"):
            self.q, self.s = quant.quantize_int8(w)
            if BUG == "scale x1.05":
                self.s = self.s * 1.05
            elif BUG == "scale x1.2":
                self.s = self.s * 1.2
            elif BUG == "neighbour scale":
                self.s = self.s.roll(1)
            elif BUG == "zeroed channel":
                self.q = self.q.clone()
                self.q[0] = 0
            self.cache = {{}}
            self.static = None
            return
        if PRECISION in ("fp4_weights", "fp4_w4a4"):
            codes, scales, ts = quant.quantize_fp4(w)
            if BUG == "nibbles":
                codes = (codes >> 4) | ((codes & 0xF) << 4)
            elif BUG == "scale x1.2":
                ts = ts * 1.2
            self.w = quant.dequantize_fp4(codes, scales, ts, w.dtype)
            self.codes, self.scales, self.ts = codes, scales, ts  # fp4_w4a4 (#233)
            return
        self.q, self.s = quant.quantize_fp8(w)
        if BUG == "scale x1.05":
            self.s = self.s * 1.05
        elif BUG == "scale x1.2":
            self.s = self.s * 1.2
        elif BUG == "neighbour scale":
            self.s = self.s.roll(1)
        self.w = quant.dequantize_fp8(self.q, self.s, w.dtype)
        self.cache = {{}}
        self.static = None

    def forward(self, x):
        if PRECISION_OF.get(id(self)) == "int4":
            return torch.nn.functional.linear(x, self.w)
        if PRECISION == "fp8_mx":
            return quant.mxfp8_linear(x, self.q, self.s, rule=RULE)
        if PRECISION == "int8_weights":
            return quant.int8_weights_linear(x, self.q, self.s)
        if PRECISION == "int8_w8a8":
            return self.int8(x)
        if PRECISION == "fp4_w4a4":
            return quant.fp4_w4a4_linear(x, self.codes, self.scales, self.ts)
        if PRECISION != "fp8_w8a8":
            return torch.nn.functional.linear(x, self.w)
        if BUG not in ACTIVATION_BUGS:
            return quant.fp8_w8a8_linear(x, self.q, self.s)
        xq, xs = quant.quantize_fp8_activations(x)
        if BUG == "first token's scale":
            xs = xs[:1].expand_as(xs)
        elif BUG == "cached activation scales":  # the first call's, per input shape
            xs = self.cache.setdefault(tuple(x.shape), xs)
            a = x.detach().reshape(-1, x.shape[-1]).float()
            xq = (a / xs[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        elif BUG == "static activation scale":  # calibrated on the first call, 2x headroom
            if self.static is None:
                self.static = float(x.detach().abs().max()) * 2 / 448
            a = x.detach().reshape(-1, x.shape[-1]).float()
            xs = torch.full((a.shape[0],), self.static, device=x.device)
            xq = (a / self.static).clamp(-448, 448).to(torch.float8_e4m3fn)
        y = (xq.float() * xs[:, None]) @ (self.q.float() * self.s[:, None]).T
        return y.to(x.dtype).reshape(*x.shape[:-1], -1)

    def int8(self, x):
        if BUG not in ACTIVATION_BUGS:
            return quant.int8_w8a8_linear(x, self.q, self.s)
        a = x.detach().reshape(-1, x.shape[-1]).float()
        xq, xs = quant.quantize_int8_activations(x)
        if BUG == "first token's scale":
            xs = xs[:1].expand_as(xs)
        elif BUG == "cached activation scales":  # the first call's, per input shape
            xs = self.cache.setdefault(tuple(x.shape), xs)
        elif BUG == "static activation scale":  # calibrated on the first call, 2x headroom
            if self.static is None:
                self.static = float(a.abs().max()) * 2 / 127
            xs = torch.full((a.shape[0],), self.static, device=x.device)
        xq = torch.round(a / xs[:, None]).clamp(-127, 127).to(torch.int8)
        y = quant.int8_matmul(xq, self.q).float() * xs[:, None] * self.s[None, :]
        return y.to(x.dtype).reshape(*x.shape[:-1], -1)


class Mlp(torch.nn.Module):
    def __init__(self, ref):
        super().__init__()
        self.gate, self.up, self.down = (Linear(ref.gate_proj), Linear(ref.up_proj),
                                         Linear(ref.down_proj))

    def forward(self, x):
        if BUG == "gate-up swapped":  # a packed gate|up weight read in the wrong order
            return self.down(torch.nn.functional.silu(self.up(x)) * self.gate(x))
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


def build(reference):
    return Mlp(reference)
"""
INT8 = ("int8_weights", "int8_w8a8")
FOUR_BIT = ("fp4_weights", "fp4_w4a4")
#: Broken variants caught on the captured inputs but not on every redrawn draw: INT8 clamps
#: where e4m3 would round coarser, and the redrawn tokens' amax are within ~20 % of the
#: first one's (the captured tokens' are not).
CAPTURED_ONLY = {("int8_w8a8", "first token's scale")}
REFERENCE_MATH = (
    "fp8_weights",
    "fp8_w8a8",
    "fp4_weights",
    "fp8_mx",
    "int8_weights",
    "int8_w8a8",
    "fp4_w4a4",
)


class Mlp(nn.Module):
    def __init__(self, hidden: int = 256, inter: int = 1024) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


def _writer_mlp(hidden: int = 256, inter: int = 1024) -> tuple[Mlp, list[torch.Tensor]]:
    """A bf16 MLP whose output channel 0 writes a massive activation (its weight row follows
    the hidden units that input channel 0 drives, at 6x the usual row norm) and real-like
    inputs in which input channel 0 is an outlier channel: on them output channel 0 is
    massive, on redrawn inputs it is as small as the others but keeps 6x their rounding
    error."""
    gen = torch.Generator().manual_seed(0)
    mlp = Mlp(hidden, inter)
    with torch.no_grad():
        mlp.gate_proj.weight.copy_(torch.randn(inter, hidden, generator=gen) * hidden**-0.5)
        mlp.up_proj.weight.copy_(torch.randn(inter, hidden, generator=gen) * hidden**-0.5)
        mlp.down_proj.weight.copy_(torch.randn(hidden, inter, generator=gen) * inter**-0.5)
        outlier = torch.zeros(hidden)
        outlier[0] = 20.0
        h = nn.functional.silu(mlp.gate_proj.weight @ outlier) * (mlp.up_proj.weight @ outlier)
        mlp.down_proj.weight[0] = h / h.norm() * 6.0
    inputs = []
    for batch in (32, 16):
        x = torch.randn(batch, 11, hidden, generator=gen)
        x[..., 0] += 20.0
        inputs.append(x.to(torch.bfloat16))
    return mlp.to(torch.bfloat16).eval(), inputs


@pytest.fixture(scope="module")
def writer():
    return _writer_mlp()


QUALITIES = ("near-lossless", "relaxed")


def _capture(tmp_path, writer, precision, quality="near-lossless"):
    mlp, inputs = writer
    path = tmp_path / f"{precision}.{quality}.pt"
    tier = compare.tier_for(quality, precision)
    calls = [((x,), {}, count) for x, count in zip(inputs, (540, 0), strict=True)]
    capture_calls(mlp, calls, path, tier=tier, precision=precision)
    return path, tier


def _candidate(tmp_path, precision, bug=None):
    path = tmp_path / f"cand_{precision}_{(bug or 'ok').replace(' ', '_')}.py"
    path.write_text(CANDIDATE.format(precision=precision, bug=bug))
    namespace: dict = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    return path, namespace["build"]


def _redrawn(writer, build, tier, *, perturbed, seeds=6):
    """compare_tensors of the candidate against the reference on ``seeds`` redrawn copies of
    each captured input (verify.perturb_: normal draws, then uniform / Laplace / log-normal
    ones), with the bounds of redrawn inputs (``perturbed``) or of captured ones."""
    mlp, inputs = writer
    candidate = build(mlp)
    gen = torch.Generator().manual_seed(1)
    results = []
    for seed in range(seeds):
        for x in inputs:
            x = x.clone()
            verify.perturb_((x,), gen, "normal" if seed % 2 == 0 else "mix")
            with torch.inference_mode():
                ref, new = mlp(x), candidate(x)
            results.append(compare.compare_tensors("y", ref, new, tier=tier, perturbed=perturbed))
    return results


def test_every_reduced_precision_is_calibrated_on_both_input_classes():
    # fp8_kv (a KV cache, not a GEMM): calibrated on a decode-attention capture in
    # test_fp8_toolkit.py
    assert set(REFERENCE_MATH) == set(compare.REDUCED_PRECISIONS) - {"reduced", "fp8_kv"}
    assert set(compare.PERTURBED_BOUNDS) == set(compare.NEAR_LOSSLESS_BOUNDS)
    for tier, (cosine, rel_l2, norm, (a, r)) in compare.PERTURBED_BOUNDS.items():
        captured = compare.NEAR_LOSSLESS_BOUNDS[tier]  # never tighter on redrawn inputs
        assert cosine <= captured[0] and rel_l2 >= captured[1] and norm >= captured[2]
        assert a >= captured[3][0] and r >= captured[3][1]
    assert compare.CHANNEL_MIN_ROWS >= 2


def test_the_relaxed_tiers_mirror_the_near_lossless_ones_with_looser_bounds():
    """#175: one relaxed tier per near-lossless tier (per precision), never tighter, and the
    8-bit tier's whole-tensor error budgets twice near-lossless's on captured inputs."""
    pairs = {
        compare.NEAR_LOSSLESS_TIER: compare.RELAXED_TIER,
        compare.NEAR_LOSSLESS_FP4_TIER: compare.RELAXED_FP4_TIER,
        compare.NEAR_LOSSLESS_KV_TIER: compare.RELAXED_KV_TIER,
        compare.NEAR_LOSSLESS_FP4A_TIER: compare.RELAXED_FP4A_TIER,  # #233
    }
    for precision in compare.REDUCED_PRECISIONS:
        near, relaxed = (
            compare.tier_for(QUALITIES[0], precision),
            compare.tier_for(QUALITIES[1], precision),
        )
        assert pairs[near] == relaxed
    for bounds in (compare.NEAR_LOSSLESS_BOUNDS, compare.PERTURBED_BOUNDS):
        for near, relaxed in pairs.items():
            (c0, l0, n0, (a0, r0)), (c1, l1, n1, (a1, r1)) = bounds[near], bounds[relaxed]
            assert c1 <= c0 and l1 >= l0 and n1 >= n0 and a1 >= a0 and r1 == r0, (near, relaxed)
    (c0, l0, n0, _), (c1, l1, n1, _) = (
        compare.NEAR_LOSSLESS_BOUNDS[t] for t in (compare.NEAR_LOSSLESS_TIER, compare.RELAXED_TIER)
    )
    assert 1 - c1 >= 2 * (1 - c0) and l1 == 2 * l0 and n1 == 2 * n0


@pytest.mark.parametrize("quality", QUALITIES)
@pytest.mark.parametrize("precision", REFERENCE_MATH)
def test_reference_math_passes_captured_and_redrawn_inputs(tmp_path, writer, precision, quality):
    capture, tier = _capture(tmp_path, writer, precision, quality)
    path, build = _candidate(tmp_path, precision)
    result = evaluate(capture, path, device="cpu")  # captured, then redrawn (verify.py)
    assert result["status"] == "ok" and result["correct"], result
    assert "perturbed" in result["checks"] and result["tolerance_tier"] == tier

    redrawn = _redrawn(writer, build, tier, perturbed=True)
    assert all(r["ok"] for r in redrawn), [r.get("error") for r in redrawn if not r["ok"]]
    if quality != "near-lossless":
        return
    # #198: the inputs are redrawn per channel, so input channel 0 keeps its offset and
    # output channel 0 stays massive; these draws pass the bounds of captured inputs too.
    # #109's wider bounds are for inputs drawn from the tensor's statistics (a decode token):
    # there the massive channel is gone and its 6x rounding error is not
    old = _redrawn(writer, build, tier, perturbed=False)
    assert all(r["ok"] for r in old), [r.get("error") for r in old if not r["ok"]]


@pytest.mark.parametrize(
    ("precision", "bug", "status"),
    [
        ("fp8_w8a8", "scale x1.05", "incorrect"),
        ("fp8_w8a8", "neighbour scale", "incorrect"),
        ("fp8_w8a8", "first token's scale", "incorrect"),
        ("fp8_w8a8", "cached activation scales", "incorrect_perturbed"),
        ("fp8_weights", "scale x1.05", "incorrect"),
        ("fp8_weights", "neighbour scale", "incorrect"),
        ("fp8_mx", "neighbour scale", "incorrect"),
        ("fp4_weights", "nibbles", "incorrect"),
        ("fp4_w4a4", "nibbles", "incorrect"),
        ("int8_w8a8", "scale x1.05", "incorrect"),
        ("int8_w8a8", "neighbour scale", "incorrect"),
        ("int8_w8a8", "first token's scale", "incorrect"),
        ("int8_w8a8", "zeroed channel", "incorrect"),
        ("int8_w8a8", "cached activation scales", "incorrect_perturbed"),
        ("int8_weights", "scale x1.05", "incorrect"),
        ("int8_weights", "neighbour scale", "incorrect"),
        ("int8_weights", "zeroed channel", "incorrect"),
    ],
)
@pytest.mark.parametrize("quality", QUALITIES)
def test_broken_scales_still_fail(tmp_path, writer, precision, bug, status, quality):
    capture, tier = _capture(tmp_path, writer, precision, quality)
    path, build = _candidate(tmp_path, precision, bug)
    result = evaluate(capture, path, device="cpu")
    assert result["status"] == status, result
    if status == "incorrect" and (precision, bug) not in CAPTURED_ONLY:  # every redrawn too
        assert not any(r["ok"] for r in _redrawn(writer, build, tier, perturbed=True))


@pytest.mark.parametrize(
    ("bug", "status"),
    [
        # on the captured inputs the massive output channel 0 dominates the tensor's norm and
        # hides most of int4's error (near-lossless: element ratio 1.06; relaxed: 0.74, FP4
        # tiers less); the redrawn inputs (no massive channel) show it
        ("int4 per tensor", {"near-lossless": "incorrect", "relaxed": "incorrect_perturbed"}),
        ("scale x1.2", "incorrect"),
        ("row 0 skipped", "incorrect"),  # an output channel never written
        ("gate-up swapped", "incorrect"),  # a packed gate|up weight read in the wrong order
    ],
)
@pytest.mark.parametrize("precision", ("fp8_weights", "fp8_w8a8", "fp4_weights", "fp4_w4a4", *INT8))
@pytest.mark.parametrize("quality", QUALITIES)
def test_blatant_bugs_fail_the_relaxed_tiers_as_the_near_lossless_ones(
    tmp_path, writer, precision, bug, status, quality
):
    """#175: the relaxed bounds reject blatant bugs at the stage the near-lossless ones do
    (the evaluator's captured inputs, or its redrawn ones)."""
    capture, tier = _capture(tmp_path, writer, precision, quality)
    path, _ = _candidate(tmp_path, precision, bug)
    result = evaluate(capture, path, device="cpu")
    if isinstance(status, dict):
        status = "incorrect_perturbed" if precision in FOUR_BIT else status[quality]
    assert result["status"] == status and result["tolerance_tier"] == tier, result


def test_a_saturating_mxfp8_scale_rule_is_rejected_with_its_reason(tmp_path, writer):
    """The OCP floor rule clamps block maxima above 448 x scale: the scale-rule guard
    (kernels/scale_guard.py) names it even where the tier alone would not catch it."""
    capture, _ = _capture(tmp_path, writer, "fp8_mx")
    path, _ = _candidate(tmp_path, "fp8_mx", "floor scale rule")
    result = evaluate(capture, path, device="cpu")
    assert result["status"] == "incorrect" and result["stage"] == "scale_rule", result
    assert "saturate" in result["error"] and "2^ceil(log2(amax / 448))" in result["error"]
    assert result["scale_rule"]["checked"] and result["scale_rule"]["input"] in (
        "captured",
        "stress",
    )
    good, _ = _candidate(tmp_path, "fp8_mx")
    report = evaluate(capture, good, device="cpu")["scale_rule"]
    assert report["ok"] and report["checked"] and report["stress"]["saturated"] == 0
    assert report["stress"]["worst_ratio"] <= 448 and report["outliers"]["channel_ratio"] > 5


@pytest.mark.parametrize("precision", ["fp8_w8a8", "int8_w8a8"])
def test_a_static_activation_scale_fails_the_scaled_check(tmp_path, writer, monkeypatch, precision):
    """An activation scale calibrated on the first call with 2x headroom passes the captured
    inputs and most redrawn draws (no outlier channel there), and always saturates on the
    captured inputs x 3 (#148; x 0.01 pushes small values into e4m3's subnormals and often
    fails too): the scales must follow the input. The same for INT8 (#178)."""
    capture, tier = _capture(tmp_path, writer, precision)
    path, build = _candidate(tmp_path, precision, "static activation scale")
    result = evaluate(capture, path, device="cpu")
    assert result["status"] == "incorrect_perturbed", result

    mlp, inputs = writer
    candidate = build(mlp)
    monkeypatch.setattr(compare, "TIER", tier)  # as the evaluator sets it from the capture
    for x in inputs:
        with torch.inference_mode():
            out = mlp(x)
        case = {"output": out, "post_args": (x.clone(),), "post_kwargs": {}}
        failed = verify.reverify_case(
            mlp, candidate, case, ((x,), {}), torch.Generator().manual_seed(0), lambda: None
        )
        checks = [f["check"] for f in failed]
        assert "scaled_x3" in checks and "sign_flipped" not in checks  # symmetric codes
    redrawn = _redrawn(writer, build, tier, perturbed=True, seeds=10)
    assert sum(not r["ok"] for r in redrawn) <= len(redrawn) // 4  # what the redraws miss


def _wide_channel() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(ref, noise, error)``: ``ref`` [64, 32] whose channel 0 has 10x the spread of the
    others, rounding-like ``noise`` scaled the same way, and ``error``: 1.5x the captured
    inputs' element bound (0.5 x RMS of the tensor) at the smallest element of channel 0."""
    gen = torch.Generator().manual_seed(0)
    ref = torch.randn(64, 32, generator=gen)
    ref[:, 0] *= 10
    noise = torch.randn(64, 32, generator=gen) * 0.01
    noise[:, 0] *= 10
    a = compare.NEAR_LOSSLESS_BOUNDS[compare.NEAR_LOSSLESS_TIER][3][0]
    error = torch.zeros_like(ref)
    error[ref[:, 0].abs().argmin(), 0] = 1.5 * a * float(ref.pow(2).mean().sqrt())
    return ref, noise, error


def test_redrawn_inputs_scale_the_element_bound_per_channel():
    ref, noise, error = _wide_channel()
    tier = compare.NEAR_LOSSLESS_TIER
    new = ref + noise + error
    assert "tolerance away" in compare.compare_tensors("y", ref, new, tier=tier)["error"]
    redrawn = compare.compare_tensors("y", ref, new, tier=tier, perturbed=True)
    assert redrawn["ok"], redrawn  # within the bound of its own channel
    corrupted = ref + noise
    corrupted[5, 7] += 3.0  # an element of an ordinary channel: still caught
    bad = compare.compare_tensors("y", ref, corrupted, tier=tier, perturbed=True)
    assert "on redrawn inputs" in bad["error"] and "its channel's" in bad["error"]
    rows = compare.CHANNEL_MIN_ROWS - 1  # too few rows for a channel RMS: the tensor's
    assert compare._channel_rms(ref[:rows], 1.5) == 1.5
    scale = compare._channel_rms(ref, 1.5)
    assert isinstance(scale, torch.Tensor) and scale.shape == ref.shape
    assert float(scale[0, 0]) > 5 and float(scale[0, 1:].min()) >= 1.5
    assert compare.EXACT_TIER not in compare.PERTURBED_BOUNDS  # exact: the same checks


def test_a_tensor_without_signal_gets_the_tier_element_bound_on_redrawn_inputs():
    gen = torch.Generator().manual_seed(0)
    ref = (torch.randn(256, generator=gen) * 0.01).to(torch.bfloat16)  # RMS < atol 0.02
    new = ref.clone()
    new[0] += 0.03  # beyond the exact tolerance, within FP4's element bound at RMS atol
    fp4 = compare.NEAR_LOSSLESS_FP4_TIER
    assert not compare.compare_tensors("v", ref, new, tier=fp4)["ok"]
    assert compare.compare_tensors("v", ref, new, tier=fp4, perturbed=True)["ok"]
    near = compare.NEAR_LOSSLESS_TIER  # FP8: its element bound at atol is the tighter one
    assert not compare.compare_tensors("v", ref, new, tier=near, perturbed=True)["ok"]
    assert not compare.compare_tensors("v", ref, new + 1.0, tier=fp4, perturbed=True)["ok"]


def test_recheck_and_the_timed_output_check_use_the_redrawn_bounds(monkeypatch):
    ref, noise, error = _wide_channel()
    new = ref + noise + error
    expected = [{"output": {"output": ref}, "pre": {}, "args": {}, "kwargs": {}}]
    saved = [{"output": {"output": new}, "args": {}, "kwargs": {}}]
    tier = compare.NEAR_LOSSLESS_TIER
    assert compare_entries(expected, saved, [(0, 0)], tier=tier) == [[]]
    assert not compare.compare_tensors("output", ref, new, tier=tier)["ok"]

    # the kept timed call runs on redrawn inputs (bench._perturbed_copy)
    monkeypatch.setattr(compare, "TIER", tier)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    x = torch.zeros(1)
    kept = {"iteration": 3, "pre": ((x,), {}), "post": ((x,), {}), "output": new}
    assert bench.check_timed_output(lambda x: ref, kept) == {"iteration": 3, "failures": []}
    kept["output"] = new + 1.0
    assert bench.check_timed_output(lambda x: ref, kept)["failures"]


@pytest.mark.parametrize("quality", QUALITIES)
def test_engineer_prompt_states_the_redrawn_bounds(quality):
    for precision in REFERENCE_MATH:
        tier = compare.tier_for(quality, precision)
        cosine, rel_l2, norm, (a, r) = compare.PERTURBED_BOUNDS[tier]
        text = prompts._tier_bounds(precision, tier)
        assert f"the {tier} tolerance tier" in text
        assert "perturbed-input check" in text and f"cosine >= {cosine:g}" in text
        assert f"{a:g} x RMS (the larger of the tensor's and its channel's)" in text
        assert f"±{norm * 100:g} %" in text and f"{rel_l2:g}" in text and f"{r:g}" in text


def test_loosening_the_redrawn_bounds_is_an_integrity_violation(tmp_path, writer, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    capture, _ = _capture(tmp_path, writer, "fp8_w8a8")
    path, _ = _candidate(tmp_path, "fp8_w8a8", "cached activation scales")
    source = path.read_text().replace(
        "def build(reference):\n",
        "def build(reference):\n"
        "    from kernel_agent.kernels import compare\n"
        "    compare.PERTURBED_BOUNDS[compare.NEAR_LOSSLESS_TIER] = (-1.0, 9.0, 9.0, (9.0, 9.0))\n",
    )
    path.write_text(source)
    saved = dict(compare.PERTURBED_BOUNDS)
    try:
        result = evaluate(capture, path, device="cpu")
    finally:
        compare.PERTURBED_BOUNDS.clear()
        compare.PERTURBED_BOUNDS.update(saved)
        compare.TIER = compare.EXACT_TIER
    assert not result["correct"] and result["status"] == "integrity_violation", result
