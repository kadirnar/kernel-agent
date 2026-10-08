"""Same-shape caches a call returns with rows written are compared relative to the update (#206).

A static KV cache that a call updates functionally (``cache.index_copy(2, position, k)`` /
a ``scatter`` into a paged cache, returned; the input left as it was) used to be compared
whole against the reference: one wrong written row of a 4096-slot cache was ~0.02 % of the
elements, inside the 0.1 % mismatch allowance (the exact-tier hole #202 closed for
concatenated caches). Now (``compare.written_box`` / ``compare_written``) an output equal to
an input of its shape except for a box of written rows is compared in two parts: the box on
its own, with the tier's checks, and the rest bit for bit."""

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


def _before_206(monkeypatch) -> None:
    """The comparison before #206: same-shape returned caches compared whole."""
    monkeypatch.setattr(compare, "written_box", lambda before, after: None)


def _sizes(box) -> list[int] | None:
    return None if box is None else [int(i.numel()) for i in box]


# ------------------------------------------------------------------ the comparison itself


def test_written_box_finds_the_rows_a_call_wrote():
    gen = torch.Generator().manual_seed(0)
    cache = torch.randn(1, 2, 16, 8, generator=gen).to(bf)
    rows = torch.randn(1, 2, 3, 8, generator=gen).to(bf)
    one = cache.index_copy(2, torch.tensor([5]), rows[:, :, :1])
    assert _sizes(compare.written_box(cache, one)) == [1, 2, 1, 8]
    assert compare.written_box(cache, one)[2].tolist() == [5]
    three = cache.index_copy(2, torch.tensor([3, 4, 9]), rows)  # a short prefill, any slots
    assert _sizes(compare.written_box(cache, three)) == [1, 2, 3, 8]
    paged = torch.randn(8, 4, 2, 8, generator=gen).to(bf)  # blocks x block size x heads x dim
    written = paged.clone()
    written[5, 3] = rows[0, :, 0]
    assert _sizes(compare.written_box(paged, written)) == [1, 1, 2, 8]
    v = torch.randn(1, 2, 8, 16, generator=gen).to(bf)  # K^T layout: the slot is the last dim
    assert _sizes(compare.written_box(v, v.index_copy(3, torch.tensor([7]), v[..., :1] + 1)))
    lucky = one.clone()  # a written value equal to the slot's old one, by chance
    lucky[0, 0, 5, 0] = cache[0, 0, 5, 0]
    assert _sizes(compare.written_box(cache, lucky)) == [1, 2, 1, 8]
    for after in (
        cache.clone(),  # unchanged: the input itself, compared whole
        cache.index_copy(2, torch.arange(9), torch.randn(1, 2, 9, 8).to(bf)),  # over half
        torch.randn(1, 2, 17, 8).to(bf),  # another shape (grown: compare.grown_dim)
        one.float(),  # another dtype
    ):
        assert compare.written_box(cache, after) is None, after.shape
    ids = torch.zeros(1, 16, dtype=torch.long)  # integers must match exactly anyway
    assert compare.written_box(ids, ids.index_fill(1, torch.tensor([3]), 7)) is None
    assert compare.written_box(None, cache) is None
    assert compare.written_box(torch.tensor(1.0), torch.tensor(2.0)) is None


def test_a_residual_add_that_keeps_some_elements_is_not_a_write():
    """``x + f(x)`` in bf16 leaves an element as it was where ``|f| < ½ ulp(x)``: a short
    output can keep most of them. Those are not written rows (scattered, or scalars: no
    whole dimension of more than one element), so it is compared whole, as before."""
    gen = torch.Generator().manual_seed(0)
    for tokens in (1, 2, 4):
        x = (torch.randn(1, tokens, 1024, generator=gen) * 8).to(bf)
        update = torch.randn(1, tokens, 1024, generator=gen) * 0.012  # most below ½ ulp
        y = (x.float() + update).to(bf)
        kept = float(compare._same(x, y).float().mean())
        assert kept > 0.6, kept  # most elements kept, but not as rows
        assert compare.written_box(x, y) is None, tokens
        check = compare.compare_structures([y], [y.clone()], inputs=((x,), {}))[0]
        assert check["ok"] and "written" not in check
    scalars = torch.randn(1, 4096, generator=gen).to(bf)  # one scalar per slot: not rows
    assert compare.written_box(scalars, scalars.index_fill(1, torch.tensor([7]), 0.5)) is None


def _cache(slots: int = 4096, filled: int = 4000, seed: int = 0):
    """``(cache, row, ref)``: a static V cache of ``slots`` (2 KV heads x 32, std 0.05: every
    value within 10x the bf16 tolerance of 0; ``filled`` slots used, the rest zeros), the
    new token's row and the cache with it written at slot ``filled``."""
    gen = torch.Generator().manual_seed(seed)
    cache = (torch.randn(1, 2, slots, 32, generator=gen) * 0.05).to(bf)
    cache[:, :, filled:] = 0
    row = (torch.randn(1, 2, 1, 32, generator=gen) * 0.05).to(bf)
    return cache, row, cache.index_copy(2, torch.tensor([filled]), row)


def _wrong_rows(cache: torch.Tensor, row: torch.Tensor, slot: int) -> dict[str, torch.Tensor]:
    """What a kernel that writes the new row wrongly returns."""

    def at(*writes: tuple[int, torch.Tensor]) -> torch.Tensor:
        out = cache.clone()
        for s, r in writes:
            out[:, :, s : s + 1] = r
        return out

    return {
        "written row x 1.2": at((slot, (row.float() * 1.2).to(bf))),
        "write skipped": cache.clone(),
        "the previous token's row": at((slot, cache[:, :, slot - 1 : slot])),
        "written one slot late": at((slot + 1, row)),
        "written row also over the next slot": at((slot, row), (slot + 1, row)),
    }


def _verdicts(cache, ref, new) -> dict[tuple[str, bool], bool]:
    """ok per (tier, redrawn bounds) of the output ``new`` against ``ref`` (``cache`` with
    rows written), the cache among the call's inputs."""
    found = {}
    inputs = ((torch.zeros(1, 1, 64, dtype=bf), cache), {})
    for tier in compare.TIERS:
        for perturbed in (False, True):
            checks = compare.compare_structures(
                (ref,), (new,), "output", inputs=inputs, tier=tier, perturbed=perturbed
            )
            found[tier, perturbed] = all(c["ok"] for c in checks)
    return found


def test_one_wrong_written_row_fails_every_tier(monkeypatch):
    cache, row, ref = _cache()
    assert all(_verdicts(cache, ref, ref.clone()).values())
    honest = ref.clone()
    honest[:, :, 4000] += 0.001  # within every tier's tolerance
    assert all(_verdicts(cache, ref, honest).values())
    for label, new in _wrong_rows(cache, row, 4000).items():
        assert not any(_verdicts(cache, ref, new).values()), label
    _before_206(monkeypatch)  # compared whole: the 64 wrong values hide in 262 144
    passed = {
        label: _verdicts(cache, ref, new) for label, new in _wrong_rows(cache, row, 4000).items()
    }
    assert passed["written row x 1.2"][compare.EXACT_TIER, False]  # the exact-tier hole
    assert all(passed["written row x 1.2"].values())  # norm, cosine, element bound: all diluted
    assert passed["write skipped"][compare.EXACT_TIER, False]


def test_the_report_names_the_written_box_and_the_kept_part():
    cache, row, ref = _cache(64, 32)
    wrong = _wrong_rows(cache, row, 32)
    inputs = ((cache,), {})
    result = compare.compare_structures((ref,), (wrong["written row x 1.2"],), inputs=inputs)[0]
    assert not result["ok"] and result["written"] == {"box": [1, 2, 1, 32], "of": [1, 2, 64, 32]}
    assert result["error"].startswith("the written [1, 2, 1, 32] of [1, 2, 64, 32] (where ")
    assert result["kept_changed"] == 0 and result["mismatch_frac"] > compare.MAX_MISMATCH
    late = wrong["written row also over the next slot"]
    result = compare.compare_structures((ref,), (late,), inputs=inputs)[0]
    assert result["kept_changed"] == 64 and "keeps them as they were" in result["error"]
    assert result["mismatch_frac"] == 0  # the written row itself is right


def test_the_rest_stays_bit_for_bit():
    cache, _, ref = _cache(256, 128)
    nudged = ref.clone()
    nudged[0, 1, 7, 3] = (nudged[0, 1, 7, 3].float() * (1 + 2**-6)).to(bf)  # 1-2 bf16 steps
    assert not torch.equal(nudged, ref)
    for tier in compare.TIERS:  # one kept element moved by rounding: not what the call wrote
        result = compare.compare_structures([ref], [nudged], inputs=[cache], tier=tier)[0]
        assert not result["ok"] and result["kept_changed"] == 1, (tier, result)
    nan = cache.clone()
    nan[0, 0, 3, 0] = float("nan")  # a NaN the reference keeps is kept as NaN
    ref_nan = nan.index_copy(2, torch.tensor([128]), ref[:, :, 128:129])
    assert compare.compare_structures([ref_nan], [ref_nan.clone()], inputs=[nan])[0]["ok"]


def test_a_paged_write_is_compared_on_the_written_slot():
    """A paged cache (blocks x block size x heads x dim) written at one (block, offset): the
    box is that slot, not its whole block (whose other offsets would dilute it 16x)."""
    gen = torch.Generator().manual_seed(0)
    paged = (torch.randn(64, 16, 2, 32, generator=gen) * 0.05).to(bf)
    row = (torch.randn(2, 32, generator=gen) * 0.05).to(bf)
    ref, wrong = paged.clone(), paged.clone()
    ref[40, 3] = row
    wrong[40, 3] = (row.float() * 1.03).to(bf)  # 3 %: below FP8's noise per element
    kw = {"inputs": [paged], "tier": compare.NEAR_LOSSLESS_TIER}
    result = compare.compare_structures([ref], [wrong], **kw)[0]
    assert result["written"]["box"] == [1, 1, 2, 32]
    assert not result["ok"] and "a systematic error" in result["error"], result
    with pytest.MonkeyPatch.context() as mp:
        _before_206(mp)
        assert compare.compare_structures([ref], [wrong], **kw)[0]["ok"]  # one slot of 1024


def test_a_written_row_gets_its_own_rms_at_x001():
    """At ×0.01 the cache's old rows are scaled and the new V row (from normalised
    activations) is not: compared whole, the cache's RMS set the new row's element bound
    and FP8-like noise of 3 % of its RMS failed near-lossless."""
    cache, row, _ = _cache(1024, 512)
    gen = torch.Generator().manual_seed(1)
    small = (cache.float() * 0.01).to(bf)
    ref = small.index_copy(2, torch.tensor([512]), row)
    noise = 0.03 * row.float().pow(2).mean().sqrt() * torch.randn(row.shape, generator=gen)
    new = small.index_copy(2, torch.tensor([512]), (row.float() + noise).to(bf))
    kw = {"tier": compare.NEAR_LOSSLESS_TIER, "perturbed": True, "input_scale": 0.01}
    result = compare.compare_structures([ref], [new], inputs=[small], **kw)[0]
    assert result["ok"] and result["element_ratio"] < 0.25, result
    with pytest.MonkeyPatch.context() as mp:
        _before_206(mp)
        whole = compare.compare_structures([ref], [new], inputs=[small], **kw)[0]
    assert not whole["ok"] and whole["element_ratio"] > 1.0, whole


def test_on_redrawn_inputs_a_written_row_takes_its_channels_rms_from_the_cache():
    """One token has 8 rows (KV heads), too few for a channel RMS of its own: on redrawn
    inputs the written row's element bound takes each channel's RMS in the whole cache (an
    outlier channel ~40x the others), its norm and cosine stay its own."""
    gen = torch.Generator().manual_seed(0)
    cache = torch.randn(1, 8, 512, 128, generator=gen)
    row = torch.randn(1, 8, 1, 128, generator=gen)
    cache[..., 50] *= 40
    row[..., 50] *= 40
    row[0, 0, 0, 50] = 0.5  # a small value in the outlier channel
    cache = cache.to(bf)
    ref = cache.index_copy(2, torch.tensor([300]), row.to(bf))
    new = ref.clone()
    new[0, 0, 300, 50] += 4.0  # 10 % of the channel's RMS, ~1.8x the row's
    tier = compare.NEAR_LOSSLESS_TIER

    def ok(perturbed: bool) -> bool:
        kw = {"tier": tier, "perturbed": perturbed}
        return compare.compare_structures([ref], [new], inputs=[cache], **kw)[0]["ok"]

    assert ok(perturbed=True)
    assert not ok(perturbed=False)
    wrong = ref.clone()
    wrong[:, :, 300] *= 1.2  # a wrong scale: the row's own norm, not the cache's channels
    kw = {"tier": tier, "perturbed": True}
    result = compare.compare_structures([ref], [wrong], inputs=[cache], **kw)[0]
    assert "norm x1.2" in result["error"], result


def test_the_out_of_process_checks_compare_written_rows_too():
    """integrity.compare_saved_outputs (the parent's re-check of the saved outputs) and
    recheck.compare_entries (the integration's redrawn re-check) see the same verdicts."""
    cache, row, ref = _cache()
    bad = _wrong_rows(cache, row, 4000)["written row x 1.2"]
    capture = {
        "cases": [
            {
                "args": (cache,),
                "kwargs": {},
                "output": ref,
                "post_args": (cache,),
                "post_kwargs": {},
            }
        ]
    }
    for new, ok in ((ref.clone(), True), (bad, False)):
        saved = flat_outputs([{"output": new, "args": (cache,), "kwargs": {}}])
        failures = compare_saved_outputs(capture, {"cases": [0], "outputs": saved})
        assert (failures == []) == ok, failures
        expected = [
            {
                "pre": {"args": {"args[0]": cache}, "kwargs": {}},
                "output": {"output": ref},
                "args": {"args[0]": cache},
                "kwargs": {},
            }
        ]
        found = compare_entries(expected, saved, [(0, 0)], tier=compare.EXACT_TIER)
        assert (found == [[]]) == ok


# ------------------------------------------------------------------ a decode step, end to end

HIDDEN, HEADS, KV_HEADS, HEAD_DIM, SLOTS = 128, 4, 2, 32, 1280


def _rms(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    return (y * weight.float()).to(x.dtype)


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), -1) * sin


class StaticDecode(nn.Module):
    """A pre-norm attention decode step (q / k norms, RoPE, GQA) with a static KV cache it
    returns updated: ``keys.index_copy(2, position, k)`` (the inputs left as they were). V
    comes from the normalised hidden state, so ×0.01 inputs leave the new V row unscaled."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.Parameter(torch.ones(HIDDEN))
        self.q_norm = nn.Parameter(torch.ones(HEAD_DIM))
        self.k_norm = nn.Parameter(torch.ones(HEAD_DIM))
        self.q_proj = nn.Linear(HIDDEN, HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(HEADS * HEAD_DIM, HIDDEN, bias=False)

    def forward(self, x, position_embeddings, cache_position, keys, values):
        cos, sin = (t[:, None] for t in position_embeddings)
        h = _rms(x, self.norm)
        q = _rms(self.q_proj(h).view(1, 1, HEADS, HEAD_DIM), self.q_norm).transpose(1, 2)
        k = _rms(self.k_proj(h).view(1, 1, KV_HEADS, HEAD_DIM), self.k_norm).transpose(1, 2)
        v = self.v_proj(h).view(1, 1, KV_HEADS, HEAD_DIM).transpose(1, 2)
        keys = keys.index_copy(2, cache_position, _rope(k, cos, sin))
        values = values.index_copy(2, cache_position, v)
        n, group = int(cache_position[-1]) + 1, HEADS // KV_HEADS
        kk, vv = (t[:, :, :n].repeat_interleave(group, 1).float() for t in (keys, values))
        p = (_rope(q, cos, sin).float() @ kk.transpose(-1, -2) * HEAD_DIM**-0.5).softmax(-1)
        out = (p @ vv).to(x.dtype)
        return x + self.o_proj(out.transpose(1, 2).reshape(1, 1, -1)), keys, values


CANDIDATE = """import torch
from kernel_agent.kernels import quant

BUG, FP8 = {bug!r}, {fp8!r}
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

    def forward(self, x, position_embeddings, cache_position, keys, values):
        wq, wk, wv, wo = self.w
        norm, q_norm, k_norm = self.norms
        cos, sin = (t[:, None] for t in position_embeddings)
        h = rms(x, norm)
        q = rms((h @ wq.T).view(1, 1, HEADS, HEAD_DIM), q_norm).transpose(1, 2)
        k = rope(rms((h @ wk.T).view(1, 1, KV_HEADS, HEAD_DIM), k_norm).transpose(1, 2), cos, sin)
        v = (h @ wv.T).view(1, 1, KV_HEADS, HEAD_DIM).transpose(1, 2)
        n, group = int(cache_position[-1]) + 1, HEADS // KV_HEADS
        kk = torch.cat((keys[:, :, : n - 1], k), 2).repeat_interleave(group, 1).float()
        vv = torch.cat((values[:, :, : n - 1], v), 2).repeat_interleave(group, 1).float()
        p = (rope(q, cos, sin).float() @ kk.transpose(-1, -2) * HEAD_DIM**-0.5).softmax(-1)
        y = x + (p @ vv).to(x.dtype).transpose(1, 2).reshape(1, 1, -1) @ wo.T
        # the attention read the right rows; what the kernel writes to the returned cache:
        pos, late = cache_position, cache_position + 1
        if BUG == "new K row x 1.2":  # e.g. a wrong dequantisation scale on the write
            k = (k.float() * 1.2).to(k.dtype)
        elif BUG == "new V row x 1.2":
            v = (v.float() * 1.2).to(v.dtype)
        keys = keys.index_copy(2, pos, k)
        if BUG == "new V row skipped":  # the write skipped, the slot left as it was
            values = values.clone()
        elif BUG == "previous token's V row":  # an off-by-one source row
            values = values.index_copy(2, pos, values[:, :, pos - 1])
        elif BUG == "V row written one slot late":  # an off-by-one destination
            values = values.index_copy(2, late, v)
        elif BUG == "V row also over the next slot":
            values = values.index_copy(2, torch.cat((pos, late)), torch.cat((v, v), 2))
        else:
            values = values.index_copy(2, pos, v)
        return y, keys, values


def build(reference):
    return Decode(reference)
"""
WRONG_ROWS = (
    "new K row x 1.2",
    "new V row x 1.2",
    "new V row skipped",
    "previous token's V row",
    "V row written one slot late",
    "V row also over the next slot",
)


def _rotary(position: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1e4 ** (-torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM)
    emb = torch.cat((position * inv, position * inv))[None, None]
    return emb.cos().to(bf), emb.sin().to(bf)


@pytest.fixture(scope="module")
def decoder() -> tuple[StaticDecode, list[tuple[tuple, dict, int]]]:
    """The module and two decode steps (slots 1024 and 1040 of 1280; V std 0.05: every cached
    value within 10x the bf16 tolerance of 0; slots past the token zero)."""
    gen = torch.Generator().manual_seed(0)
    module = StaticDecode()
    with torch.no_grad():
        for lin in (module.q_proj, module.k_proj, module.v_proj, module.o_proj):
            lin.weight.copy_(torch.randn(lin.weight.shape, generator=gen) * lin.in_features**-0.5)
        module.v_proj.weight.mul_(0.05)
    calls = []
    for position, count in ((1024, 21), (1040, 0)):
        k = torch.randn(1, KV_HEADS, SLOTS, HEAD_DIM, generator=gen)
        v = torch.randn(1, KV_HEADS, SLOTS, HEAD_DIM, generator=gen) * 0.05
        k[:, :, position:], v[:, :, position:] = 0, 0
        x = (torch.randn(1, 1, HIDDEN, generator=gen) * 0.5).to(bf)
        at = torch.tensor([position])
        calls.append(((x, _rotary(float(position)), at, k.to(bf), v.to(bf)), {}, count))
    return module.to(bf).eval(), calls


def _candidate(tmp_path: Path, bug: str | None = None, fp8: bool = False):
    name = f"cand_{fp8}_{(bug or 'ok').replace(' ', '_')}.py".replace("'", "")
    path = tmp_path / name
    source = CANDIDATE.format(bug=bug, fp8=fp8, heads=HEADS, kv_heads=KV_HEADS, head_dim=HEAD_DIM)
    path.write_text(source)
    namespace: dict = {}
    exec(compile(source, str(path), "exec"), namespace)
    return path, namespace["build"]


def _capture(tmp_path: Path, decoder, tier: str | None = None, precision: str | None = None):
    module, calls = decoder
    path = tmp_path / f"static_{tier}.pt"
    capture_calls(module, copy.deepcopy(calls), path, tier=tier, precision=precision)
    return path


def test_the_capture_records_a_written_cache(tmp_path, decoder):
    case = load_capture(_capture(tmp_path, decoder), device="cpu")["cases"][0]
    pre, post = case["args"][4], case["output"][2]
    assert pre.shape == post.shape == (1, KV_HEADS, SLOTS, HEAD_DIM)
    box = compare.written_box(pre, post)
    assert _sizes(box) == [1, KV_HEADS, 1, HEAD_DIM] and box[2].tolist() == [1024]
    assert torch.equal(case["post_args"][4], pre)  # functional: the input left as it was


def _cache_failure(failure: dict) -> bool:
    """Whether a failed check is a returned cache's."""
    return failure.get("name", "") in ("output[1]", "output[2]")


def test_the_honest_decoder_passes_and_wrong_rows_fail_the_evaluator(tmp_path, decoder):
    capture = _capture(tmp_path, decoder)
    path, _ = _candidate(tmp_path)
    result = evaluate(capture, path, device="cpu")
    assert result["status"] == "ok" and result["correct"], result
    for bug in WRONG_ROWS:
        path, _ = _candidate(tmp_path, bug)
        result = evaluate(capture, path, device="cpu")
        assert result["status"] == "incorrect" and result["stage"] == "correctness", (bug, result)
        failures = [f for case in result["cases"] for f in case["failures"]]
        assert failures and all(_cache_failure(f) for f in failures), failures


def test_before_206_a_wrong_k_row_passed_the_exact_tier(tmp_path, decoder, monkeypatch):
    """The exact-tier hole: the written K row x 1.2 is 64 wrong values of 81 920, each
    within 10x its tolerance, on the captured, redrawn and scaled inputs."""
    capture = _capture(tmp_path, decoder)
    path, _ = _candidate(tmp_path, "new K row x 1.2")
    _before_206(monkeypatch)
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
    module, calls = decoder
    tier = compare.tier_for(quality, "fp8_weights")
    capture = _capture(tmp_path, decoder, tier, "fp8_weights")
    _, build = _candidate(tmp_path, fp8=True)
    assert _live_failures(module, build(module), calls, tier)[0] == 0
    for bug in WRONG_ROWS:
        path, build = _candidate(tmp_path, bug, fp8=True)
        result = evaluate(capture, path, device="cpu")
        assert result["status"] == "incorrect", (bug, result)
        failed, run = _live_failures(module, build(module), calls, tier)
        assert failed == run, (bug, failed, run)


def test_honest_fp8_weights_pass_the_x001_check_in_near_lossless(tmp_path, decoder, monkeypatch):
    """At x 0.01 the cache's old rows are scaled and the new V row is not: compared whole,
    the cache's RMS (1024 filled slots of 1280: ~1/36 of the new row's) bounded the new
    row's FP8 noise and honest FP8 weights failed near-lossless; compared on its own the
    written row passes, and the evaluator accepts them (captured, redrawn and scaled)."""
    module, calls = decoder
    tier = compare.tier_for("near-lossless", "fp8_weights")
    path, build = _candidate(tmp_path, fp8=True)
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
    _before_206(monkeypatch)
    failed = x001()
    assert failed and all(_cache_failure(f) for f in failed), failed
    assert "an element is" in failed[0]["error"], failed
