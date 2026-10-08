"""Caches a call grows are compared relative to the update (#202).

A cache that a call extends by concatenation (a KV cache ``torch.cat``-ed with the new
token's K / V: a ``transformers`` ``DynamicCache`` layer in the arguments, or the legacy
``(keys, values)`` it returns) used to be compared whole against the reference. The kept
rows diluted the new ones: one wrong new row of a 4096-token cache was ~0.02 % of the
elements, inside the 0.1 % mismatch allowance (an exact-tier hole), and at the ×0.01 scaled
check the old rows (scaled) set the element bound of a new row computed from normalised
activations (not scaled), so honest FP8 weights failed near-lossless. Now
(``compare.grown_dim`` / ``compare_grown``) the appended rows are compared on their own and
the kept part must stay bit for bit where the reference kept it."""

import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from kernel_agent.kernels import compare, verify
from kernel_agent.kernels.evaluate import evaluate
from kernel_agent.kernels.integrity import compare_saved_outputs, flat_outputs
from kernel_agent.kernels.recheck import compare_entries
from kernel_agent.profiling.capture import capture_calls, load_capture

bf = torch.bfloat16


def _before_202(monkeypatch) -> None:
    """The comparison before #202: grown caches compared whole."""
    monkeypatch.setattr(compare, "grown_dim", lambda before, after: None)


# ------------------------------------------------------------------ the comparison itself


def test_grown_dim_finds_the_one_dimension_a_cache_grows_along():
    cache = torch.zeros(1, 2, 16, 8, dtype=bf)
    assert compare.grown_dim(cache, torch.zeros(1, 2, 17, 8, dtype=bf)) == 2
    assert compare.grown_dim(cache, torch.zeros(1, 2, 32, 8, dtype=bf)) == 2  # a prefill
    assert compare.grown_dim(cache, torch.zeros(3, 2, 16, 8, dtype=bf)) == 0
    for other in (
        torch.zeros(1, 2, 16, 8, dtype=bf),  # same shape: an in-place update
        torch.zeros(1, 2, 15, 8, dtype=bf),  # shrunk
        torch.zeros(1, 3, 17, 8, dtype=bf),  # two dimensions
        torch.zeros(1, 2, 17, 8),  # another dtype
        torch.zeros(2, 17, 8, dtype=bf),  # another rank
    ):
        assert compare.grown_dim(cache, other) is None, other.shape
    ids = torch.zeros(1, 16, dtype=torch.long)  # integers must match exactly anyway
    assert compare.grown_dim(ids, torch.zeros(1, 17, dtype=torch.long)) is None
    assert compare.grown_dim(torch.zeros(1, 0, 8), torch.zeros(1, 4, 8)) is None  # empty
    assert compare.grown_dim(None, cache) is None


def _cache(tokens: int = 4096, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """``(kept, new)``: a V cache of ``tokens`` rows (2 KV heads x 32, std 0.05: every value
    within 10x the bf16 tolerance of 0) and one new row."""
    gen = torch.Generator().manual_seed(seed)
    kept = (torch.randn(1, 2, tokens, 32, generator=gen) * 0.05).to(bf)
    return kept, (torch.randn(1, 2, 1, 32, generator=gen) * 0.05).to(bf)


def _wrong_rows(kept: torch.Tensor, row: torch.Tensor) -> dict[str, torch.Tensor]:
    """What a kernel that writes the new row wrongly leaves in the grown cache."""
    return {
        "new row x 1.2": torch.cat((kept, (row.float() * 1.2).to(bf)), 2),
        "new row zeroed": torch.cat((kept, torch.zeros_like(row)), 2),
        "the previous token's row": torch.cat((kept, kept[:, :, -1:]), 2),
        "new row also over the last kept one": torch.cat((kept[:, :, :-1], row, row), 2),
    }


def _verdicts(pre, ref, new) -> dict[tuple[str, bool], bool]:
    """ok per (tier, redrawn bounds) of the argument cache ``pre`` -> ``ref`` / ``new``."""
    found = {}
    for tier in compare.TIERS:
        for perturbed in (False, True):
            checks = compare.compare_side_effects(
                (pre,), (ref,), (new,), "args", tier=tier, perturbed=perturbed
            )
            found[tier, perturbed] = all(c["ok"] for c in checks)
    return found


def test_one_wrong_new_row_fails_every_tier(monkeypatch):
    kept, row = _cache()
    ref = torch.cat((kept, row), 2)
    assert all(_verdicts(kept, ref, ref.clone()).values())
    honest = ref.clone()
    honest[:, :, -1] += 0.001  # within every tier's tolerance
    assert all(_verdicts(kept, ref, honest).values())
    for label, new in _wrong_rows(kept, row).items():
        assert not any(_verdicts(kept, ref, new).values()), label
    _before_202(monkeypatch)  # compared whole: the 64 wrong values hide in 262 208
    passed = {label: _verdicts(kept, ref, new) for label, new in _wrong_rows(kept, row).items()}
    assert passed["new row x 1.2"][compare.EXACT_TIER, False]  # the exact-tier hole
    assert all(passed["new row x 1.2"].values())  # norm, cosine, element bound: all diluted
    assert passed["new row zeroed"][compare.EXACT_TIER, False]


def test_the_report_names_the_appended_and_the_kept_rows():
    kept, row = _cache(64)
    ref = torch.cat((kept, row), 2)
    wrong = _wrong_rows(kept, row)
    result = compare.compare_side_effects((kept,), (ref,), (wrong["new row x 1.2"],), "args")[0]
    assert not result["ok"] and result["grown"] == {"dim": 2, "kept": 64, "appended": 1}
    assert result["error"].startswith("the 1 appended row along dim 2 (after 64 kept): ")
    assert result["kept_changed"] == 0 and result["mismatch_frac"] > compare.MAX_MISMATCH
    over = wrong["new row also over the last kept one"]
    result = compare.compare_side_effects((kept,), (ref,), (over,), "args")[0]
    assert result["kept_changed"] == 64 and "keeps them as they were" in result["error"]
    assert result["mismatch_frac"] == 0  # the appended row itself is right


def test_the_kept_part_stays_bit_for_bit_where_the_reference_kept_it():
    kept, row = _cache(256)
    ref = torch.cat((kept, row), 2)
    nudged = ref.clone()
    nudged[0, 1, 7, 3] = (nudged[0, 1, 7, 3].float() * (1 + 2**-6)).to(bf)  # 1-2 bf16 steps
    assert not torch.equal(nudged, ref)
    for tier in compare.TIERS:  # one kept element moved by rounding: a rewrite, not an append
        result = compare.compare_side_effects((kept,), (ref,), (nudged,), "a", tier=tier)[0]
        assert not result["ok"] and result["kept_changed"] == 1, (tier, result)
    nan = kept.clone()
    nan[0, 0, 3, 0] = float("nan")  # a NaN the reference copies is kept as NaN
    ref_nan = torch.cat((nan, row), 2)
    assert compare.compare_side_effects((nan,), (ref_nan,), (ref_nan.clone(),), "a")[0]["ok"]


def test_a_reference_that_changes_the_kept_part_is_compared_where_it_changed():
    """A cache whose old rows the call rewrites too (decayed, renormalised) is compared on
    the elements either side changed, as an in-place update is."""
    kept, row = _cache(256)
    ref = torch.cat(((kept.float() * 0.9).to(bf), row), 2)
    rounded = ref.clone()
    rounded[:, :, :-1] = (kept.float() * 0.9 * (1 + 1e-3)).to(bf)  # its own rounding
    result = compare.compare_side_effects((kept,), (ref,), (rounded,), "args")[0]
    assert result["ok"], result
    assert result["changed_elements"] > 0 and "kept_changed" not in result
    forgot = torch.cat((kept, row), 2)  # the old rows not decayed
    result = compare.compare_side_effects((kept,), (ref,), (forgot,), "args")[0]
    assert not result["ok"] and "changed elements of the 256 kept rows" in result["error"]


def test_a_returned_grown_cache_is_matched_to_its_input():
    """Legacy caches: the call returns ``torch.cat((past, new))``; the output whose leading
    part along one dimension equals an input tensor is compared like a grown argument."""
    kept, row = _cache()
    ref = torch.cat((kept, row), 2)
    x = torch.randn(1, 1, 64).to(bf)
    inputs = ((x, (kept, kept.clone())), {})
    for label, new in _wrong_rows(kept, row).items():
        checks = compare.compare_structures(
            (x, ref), (x.clone(), new), "output", inputs=inputs, tier=compare.EXACT_TIER
        )
        assert checks[0]["ok"] and not checks[1]["ok"], label
        assert checks[1]["grown"]["kept"] == 4096
    scaled = _wrong_rows(kept, row)["new row x 1.2"]  # without the inputs: compared whole
    assert compare.compare_structures((x, ref), (x, scaled), "output")[1]["ok"]
    unrelated = torch.randn_like(ref)  # same shape as a grown input, not its continuation
    check = compare.compare_structures([unrelated], [unrelated.clone()], inputs=inputs)[0]
    assert check["ok"] and "grown" not in check


def test_the_out_of_process_checks_split_grown_caches_too():
    """integrity.compare_saved_outputs (the parent's re-check of the saved outputs) and
    recheck.compare_entries (the integration's redrawn re-check) see the same verdicts."""
    kept, row = _cache()
    ref = torch.cat((kept, row), 2)
    bad = _wrong_rows(kept, row)["new row x 1.2"]
    capture = {
        "cases": [
            {"args": (kept,), "kwargs": {}, "output": ref, "post_args": (kept,), "post_kwargs": {}}
        ]
    }
    for new, ok in ((ref.clone(), True), (bad, False)):
        saved = flat_outputs([{"output": new, "args": (kept,), "kwargs": {}}])
        failures = compare_saved_outputs(capture, {"cases": [0], "outputs": saved})
        assert (failures == []) == ok, failures
        expected = [
            {
                "pre": {"args": {"args[0]": kept}, "kwargs": {}},
                "output": {"output": ref},
                "args": {"args[0]": kept},
                "kwargs": {},
            }
        ]
        found = compare_entries(expected, saved, [(0, 0)], tier=compare.EXACT_TIER)
        assert (found == [[]]) == ok


def test_a_new_row_gets_its_own_rms_at_x001():
    """At ×0.01 the kept V rows are scaled and the new one (from normalised activations) is
    not: compared whole, the cache's RMS (32 x below the new row's at 1024 tokens) set the
    new row's element bound and FP8-like noise of 3 % of its RMS failed near-lossless."""
    kept, row = _cache(1024)
    gen = torch.Generator().manual_seed(1)
    small = (kept.float() * 0.01).to(bf)
    ref = torch.cat((small, row), 2)
    noise = 0.03 * row.float().pow(2).mean().sqrt() * torch.randn(row.shape, generator=gen)
    new = torch.cat((small, (row.float() + noise).to(bf)), 2)
    kw = {"tier": compare.NEAR_LOSSLESS_TIER, "perturbed": True, "input_scale": 0.01}
    result = compare.compare_side_effects((small,), (ref,), (new,), "args", **kw)[0]
    assert result["ok"] and result["element_ratio"] < 0.25, result
    with pytest.MonkeyPatch.context() as mp:
        _before_202(mp)
        whole = compare.compare_side_effects((small,), (ref,), (new,), "args", **kw)[0]
    assert not whole["ok"] and whole["element_ratio"] > 1.0, whole


def test_on_redrawn_inputs_a_new_row_takes_its_channels_rms_from_the_cache():
    """One decode token has 8 rows (KV heads), too few for a channel RMS of its own
    (``compare.CHANNEL_MIN_ROWS``): on redrawn inputs the appended row's element bound takes
    each channel's RMS in the whole grown cache (Qwen3-0.6B's ``k_norm`` makes key channel 50
    ~10x the row's RMS; where a redrawn token has a small value there, the FP8 noise of
    that channel is far above the row's RMS). The row's norm, cosine and relative L2 error
    stay its own, and captured inputs keep the row's RMS."""
    gen = torch.Generator().manual_seed(0)
    kept = torch.randn(1, 8, 512, 128, generator=gen)
    row = torch.randn(1, 8, 1, 128, generator=gen)
    kept[..., 50] *= 40
    row[..., 50] *= 40
    row[0, 0, 0, 50] = 0.5  # a small value in the outlier channel
    kept, ref = kept.to(bf), torch.cat((kept, row), 2).to(bf)
    new = ref.clone()
    new[0, 0, -1, 50] += 4.0  # 10 % of the channel's RMS, ~1.8x the row's
    tier = compare.NEAR_LOSSLESS_TIER

    def ok(perturbed: bool) -> bool:
        return compare.compare_side_effects(
            (kept,), (ref,), (new,), "a", tier=tier, perturbed=perturbed
        )[0]["ok"]

    assert ok(perturbed=True)
    alone = compare.compare_tensors("a", ref[:, :, -1:], new[:, :, -1:], tier=tier, perturbed=True)
    assert not alone["ok"] and alone["element_ratio"] > 1.5, alone  # the row's RMS only
    assert not ok(perturbed=False)
    wrong = ref.clone()
    wrong[:, :, -1] *= 1.2  # a wrong scale: the row's own norm, not the cache's channels
    result = compare.compare_side_effects((kept,), (ref,), (wrong,), "a", tier=tier, perturbed=True)
    assert "norm x1.2" in result[0]["error"], result


# ------------------------------------------------------------------ a decode step, end to end

HIDDEN, HEADS, KV_HEADS, HEAD_DIM, TOKENS = 128, 4, 2, 32, 1024


class KV:
    """A cache object the decode step grows (a ``DynamicCache`` layer: ``keys`` and
    ``values`` reassigned to ``torch.cat`` of the old and the new rows)."""

    def __init__(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        self.keys, self.values = keys, values


def _rms(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    return (y * weight.float()).to(x.dtype)


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), -1) * sin


class Decode(nn.Module):
    """A pre-norm attention decode step (q / k norms, RoPE, GQA) that grows its KV cache:
    ``KV`` in the arguments, or (``Legacy``) ``(keys, values)`` in and grown out. V comes
    from the normalised hidden state, so ×0.01 inputs leave the new V row unscaled."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.Parameter(torch.ones(HIDDEN))
        self.q_norm, self.k_norm = (
            nn.Parameter(torch.ones(HEAD_DIM)),
            nn.Parameter(torch.ones(HEAD_DIM)),
        )
        self.q_proj = nn.Linear(HIDDEN, HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(HEADS * HEAD_DIM, HIDDEN, bias=False)

    def step(self, x, position_embeddings, past_keys, past_values):
        cos, sin = (t[:, None] for t in position_embeddings)
        h = _rms(x, self.norm)
        q = _rms(self.q_proj(h).view(1, 1, HEADS, HEAD_DIM), self.q_norm).transpose(1, 2)
        k = _rms(self.k_proj(h).view(1, 1, KV_HEADS, HEAD_DIM), self.k_norm).transpose(1, 2)
        v = self.v_proj(h).view(1, 1, KV_HEADS, HEAD_DIM).transpose(1, 2)
        keys = torch.cat((past_keys, _rope(k, cos, sin)), 2)
        values = torch.cat((past_values, v), 2)
        group = HEADS // KV_HEADS
        s = _rope(q, cos, sin).float() @ keys.repeat_interleave(group, 1).float().transpose(-1, -2)
        p = (s * HEAD_DIM**-0.5).softmax(-1)
        out = (p @ values.repeat_interleave(group, 1).float()).to(x.dtype)
        return x + self.o_proj(out.transpose(1, 2).reshape(1, 1, -1)), keys, values

    def forward(self, x, position_embeddings, cache):
        y, cache.keys, cache.values = self.step(x, position_embeddings, cache.keys, cache.values)
        return y


class Legacy(Decode):
    def forward(self, x, position_embeddings, past):
        y, keys, values = self.step(x, position_embeddings, *past)
        return y, (keys, values)


CANDIDATE = """import torch
from kernel_agent.kernels import quant

BUG, FP8, LEGACY = {bug!r}, {fp8!r}, {legacy!r}
HEADS, KV_HEADS, HEAD_DIM = {heads}, {kv_heads}, {head_dim}


def rms(x, weight):
    y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    return (y * weight.float()).to(x.dtype)


def rope(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), -1) * sin


def weight(linear):
    w = linear.weight.detach().clone()
    return quant.dequantize_fp8(*quant.quantize_fp8(w), w.dtype) if FP8 else w


class Decode(torch.nn.Module):
    def __init__(self, ref):
        super().__init__()
        self.w = [weight(m) for m in (ref.q_proj, ref.k_proj, ref.v_proj, ref.o_proj)]
        self.norms = [p.detach() for p in (ref.norm, ref.q_norm, ref.k_norm)]

    def forward(self, x, position_embeddings, cache):
        wq, wk, wv, wo = self.w
        norm, q_norm, k_norm = self.norms
        cos, sin = (t[:, None] for t in position_embeddings)
        past_keys, past_values = cache if LEGACY else (cache.keys, cache.values)
        h = rms(x, norm)
        q = rms((h @ wq.T).view(1, 1, HEADS, HEAD_DIM), q_norm).transpose(1, 2)
        k = rms((h @ wk.T).view(1, 1, KV_HEADS, HEAD_DIM), k_norm).transpose(1, 2)
        v = (h @ wv.T).view(1, 1, KV_HEADS, HEAD_DIM).transpose(1, 2)
        keys = torch.cat((past_keys, rope(k, cos, sin)), 2)
        values = torch.cat((past_values, v), 2)
        group = HEADS // KV_HEADS
        s = rope(q, cos, sin).float() @ keys.repeat_interleave(group, 1).float().transpose(-1, -2)
        p = (s * HEAD_DIM**-0.5).softmax(-1)
        out = (p @ values.repeat_interleave(group, 1).float()).to(x.dtype)
        y = x + out.transpose(1, 2).reshape(1, 1, -1) @ wo.T
        # the attention read the right rows; what the kernel writes to the cache:
        if BUG == "new K row x 1.2":  # e.g. a wrong dequantisation scale on the write
            keys = torch.cat((past_keys, (keys[:, :, -1:].float() * 1.2).to(k.dtype)), 2)
        elif BUG == "new V row x 1.2":
            values = torch.cat((past_values, (v.float() * 1.2).to(v.dtype)), 2)
        elif BUG == "new V row zeroed":  # the write skipped, the slot left at zero
            values = torch.cat((past_values, torch.zeros_like(v)), 2)
        elif BUG == "previous token's V row":  # an off-by-one source row
            values = torch.cat((past_values, past_values[:, :, -1:]), 2)
        elif BUG == "new V row over the last kept one":  # an off-by-one destination
            values = torch.cat((past_values[:, :, :-1], v, v), 2)
        if LEGACY:
            return y, (keys, values)
        cache.keys, cache.values = keys, values
        return y


def build(reference):
    return Decode(reference)
"""
WRONG_ROWS = (
    "new K row x 1.2",
    "new V row x 1.2",
    "new V row zeroed",
    "previous token's V row",
    "new V row over the last kept one",
)


def _rotary(position: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1e4 ** (-torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM)
    emb = torch.cat((position * inv, position * inv))[None, None]
    return emb.cos().to(bf), emb.sin().to(bf)


def _decoder(legacy: bool) -> tuple[Decode, list[tuple[tuple, dict, int]]]:
    """The module and two decode steps (1024 and 1040 cached tokens; V std 0.05: every
    cached value within 10x the bf16 tolerance of 0)."""
    gen = torch.Generator().manual_seed(0)
    module = Legacy() if legacy else Decode()
    with torch.no_grad():
        for lin in (module.q_proj, module.k_proj, module.v_proj, module.o_proj):
            lin.weight.copy_(torch.randn(lin.weight.shape, generator=gen) * lin.in_features**-0.5)
        module.v_proj.weight.mul_(0.05)
    calls = []
    for tokens, count in ((TOKENS, 21), (TOKENS + 16, 0)):
        k = torch.randn(1, KV_HEADS, tokens, HEAD_DIM, generator=gen).to(bf)
        v = (torch.randn(1, KV_HEADS, tokens, HEAD_DIM, generator=gen) * 0.05).to(bf)
        x = (torch.randn(1, 1, HIDDEN, generator=gen) * 0.5).to(bf)
        cache = (k, v) if legacy else KV(k, v)
        calls.append(((x, _rotary(float(tokens)), cache), {}, count))
    return module.to(bf).eval(), calls


@pytest.fixture(scope="module", params=[False, True], ids=["cache-argument", "returned-cache"])
def decoder(request):
    return request.param, _decoder(request.param)


def _candidate(tmp_path: Path, legacy: bool, bug: str | None = None, fp8: bool = False):
    name = f"cand_{legacy}_{fp8}_{(bug or 'ok').replace(' ', '_')}.py"
    path = tmp_path / name.replace("'", "")
    source = CANDIDATE.format(
        bug=bug, fp8=fp8, legacy=legacy, heads=HEADS, kv_heads=KV_HEADS, head_dim=HEAD_DIM
    )
    path.write_text(source)
    namespace: dict = {}
    exec(compile(source, str(path), "exec"), namespace)
    return path, namespace["build"]


def _capture(tmp_path: Path, decoder, tier: str | None = None, precision: str | None = None):
    legacy, (module, calls) = decoder
    path = tmp_path / f"decode_{legacy}_{tier}.pt"
    capture_calls(module, copy.deepcopy(calls), path, tier=tier, precision=precision)
    return path


def test_the_capture_records_a_grown_cache(tmp_path, decoder):
    legacy, _ = decoder
    case = load_capture(_capture(tmp_path, decoder), device="cpu")["cases"][0]
    if legacy:
        pre, post = case["args"][2][1], case["output"][1][1]
    else:
        pre, post = case["args"][2].values, case["post_args"][2].values
    assert pre.shape[2] == TOKENS and post.shape[2] == TOKENS + 1
    assert compare.grown_dim(pre, post) == 2 and torch.equal(post[:, :, :TOKENS], pre)


def _cache_failure(failure: dict) -> bool:
    """Whether a failed check is the grown cache's (argument or returned)."""
    name = failure.get("name", "")
    return name.endswith((".keys", ".values")) or name.startswith("output[1]")


def test_the_honest_decoder_passes_and_wrong_rows_fail_the_evaluator(tmp_path, decoder):
    legacy, _ = decoder
    capture = _capture(tmp_path, decoder)
    path, _ = _candidate(tmp_path, legacy)
    result = evaluate(capture, path, device="cpu")
    assert result["status"] == "ok" and result["correct"], result
    for bug in WRONG_ROWS:
        path, _ = _candidate(tmp_path, legacy, bug)
        result = evaluate(capture, path, device="cpu")
        assert result["status"] == "incorrect" and result["stage"] == "correctness", (bug, result)
        failures = [f for case in result["cases"] for f in case["failures"]]
        assert failures and all(_cache_failure(f) for f in failures), failures


def test_before_202_a_wrong_k_row_passed_the_exact_tier(tmp_path, decoder, monkeypatch):
    """The exact-tier hole: the new K row x 1.2 is 64 wrong values of 65 600, each within
    10x its tolerance, on the captured, redrawn and scaled inputs (K follows the scaled
    rotary tables, so x 0.01 scales the new row with the kept ones). A wrong V row was
    caught at x 0.01 only, by accident: the new V row is not scaled, so it dominates the
    cache's norm there."""
    legacy, _ = decoder
    capture = _capture(tmp_path, decoder)
    path, _ = _candidate(tmp_path, legacy, "new K row x 1.2")
    _before_202(monkeypatch)
    result = evaluate(capture, path, device="cpu")
    assert result["status"] == "ok" and result["correct"], result
    assert result["redraws"]["ran"].get("scaled_x0.01")


def _live_failures(module, candidate, calls, tier: str, draws: int = 4) -> tuple[int, int]:
    """(failed, run): checks of ``candidate`` against the live reference
    (``verify._against_reference``, as the evaluator runs them) on ``draws`` redrawn copies
    of each decode step and on the first step's inputs x 3, x 0.01 and x -1."""
    failed = run = 0
    gen = torch.Generator().manual_seed(1)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(compare, "TIER", tier)
        inputs = []
        for i, (args, kwargs, _) in enumerate(calls):
            for j in range(draws):
                a, k = copy.deepcopy((args, kwargs))
                verify.perturb_((a, k), gen, "normal" if j % 2 == 0 else "mix")
                inputs.append((a, k, 1.0))
            for _, factor in verify.SCALED if i == 0 else ():
                a, k = copy.deepcopy((args, kwargs))
                verify.scale_((a, k), factor)
                inputs.append((a, k, factor))
        for a, k, factor in inputs:
            found = verify._against_reference(
                module, candidate, a, k, lambda: None, input_scale=factor
            )
            failed += any(not c.get("ok") for c in found or [])
            run += 1
    return failed, run


@pytest.mark.parametrize("quality", ["near-lossless", "relaxed"])
def test_wrong_rows_fail_the_reduced_tiers_on_every_input_class(tmp_path, decoder, quality):
    """FP8 weights whose cache write is wrong: rejected on the captured inputs and on every
    redrawn and scaled draw, against the live reference; honest FP8 weights pass them all."""
    legacy, (module, calls) = decoder
    tier = compare.tier_for(quality, "fp8_weights")
    capture = _capture(tmp_path, decoder, tier, "fp8_weights")
    _, build = _candidate(tmp_path, legacy, fp8=True)
    assert _live_failures(module, build(module), calls, tier)[0] == 0
    for bug in WRONG_ROWS:
        path, build = _candidate(tmp_path, legacy, bug, fp8=True)
        result = evaluate(capture, path, device="cpu")
        assert result["status"] == "incorrect", (bug, result)
        failed, run = _live_failures(module, build(module), calls, tier)
        assert failed == run, (bug, failed, run)


def test_honest_fp8_weights_pass_the_x001_check_in_near_lossless(tmp_path, decoder, monkeypatch):
    """At x 0.01 the kept V rows are scaled and the new one is not: compared whole, the
    cache's RMS (1024 kept rows: ~1/32 of the new row's) bounded the new row's FP8 noise and
    honest FP8 weights failed near-lossless; compared on its own the new row passes, and the
    evaluator accepts them (captured, redrawn and scaled inputs)."""
    legacy, (module, calls) = decoder
    tier = compare.tier_for("near-lossless", "fp8_weights")
    path, build = _candidate(tmp_path, legacy, fp8=True)
    candidate = build(module)

    def x001() -> list[dict]:
        a, k = copy.deepcopy(calls[0][:2])
        verify.scale_((a, k), 0.01)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(compare, "TIER", tier)
            found = verify._against_reference(
                module, candidate, a, k, lambda: None, input_scale=0.01
            )
        return [c for c in found or [] if not c.get("ok")]

    assert x001() == []
    capture = _capture(tmp_path, decoder, tier, "fp8_weights")
    result = evaluate(capture, path, device="cpu")
    assert result["status"] == "ok" and result["correct"], result
    assert result["redraws"]["ran"].get("scaled_x0.01")
    _before_202(monkeypatch)
    failed = x001()
    assert failed and all(_cache_failure(f) for f in failed), failed
    assert "an element is" in failed[0]["error"], failed
