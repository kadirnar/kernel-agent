"""The redrawn-input check draws every channel from its own statistics (#198).

``kernels.verify.perturb_`` redraws a floating-point input from each channel's mean and std
(one position of the last dimension, over the other dimensions: ``verify.redraw_stats``)
and keeps rotary tables (``verify.rotary_tables``). Before, every tensor was drawn from its
global mean and std, rotary tables too: a KV cache with a few huge channels (Qwen3-0.6B's K
cache after ``k_norm``) came out with every channel that large, and the reference math of
every reduced precision failed most draws of a Qwen3 decoder layer at decode. A synthetic
decode step with such a cache (CPU) and the real layer (GPU) show it; the redraw's purpose
stays: fresh values in every slot, broken kernels still fail."""

import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from kernel_agent.kernels import compare, quant, verify
from kernel_agent.kernels.evaluate import evaluate
from kernel_agent.profiling.capture import capture_calls

# ------------------------------------------------------------------ the redraw itself


def _channel_rms(t: torch.Tensor) -> torch.Tensor:
    return t.float().reshape(-1, t.shape[-1]).pow(2).mean(0).sqrt()


def test_every_channel_keeps_its_scale_and_every_slot_is_redrawn():
    gen = torch.Generator().manual_seed(0)
    cache = torch.randn(1, 4, 256, 16, generator=gen)
    cache[..., 3] = torch.tensor([100.0, -80.0, 120.0, -90.0])[:, None] + 20 * cache[..., 3]
    cache[:, :, 200:] = 0  # unused slots
    before = cache.clone()
    used = _channel_rms(before[:, :, :200])
    for kind in ("normal", "mix"):
        x = before.clone()
        assert verify.perturb_((x,), torch.Generator().manual_seed(1), kind) == 1
        rms = _channel_rms(x)
        assert 0.75 < float(rms[3] / used[3]) < 1.3  # the huge channel stays huge
        others = torch.cat((rms[:3], rms[4:]))
        assert float(others.median()) < 2  # the global std (~51) would make every one huge
        assert torch.count_nonzero(x[:, :, 200:]) == x[:, :, 200:].numel()  # fresh slots
        assert not torch.equal(x[:, :, :200], before[:, :, :200])
    mean, std = verify.redraw_stats(before)
    assert isinstance(mean, torch.Tensor) and mean.shape == (1, 1, 1, 16)
    assert isinstance(std, torch.Tensor) and float(std[..., 3]) > 50 > 3 > float(std[..., 0])


def test_short_inputs_and_sparse_channels_get_the_tensors_statistics():
    gen = torch.Generator().manual_seed(0)
    token = torch.randn(1, 1, 64, generator=gen)
    token[..., 0] = 300.0  # one token: no channel statistics, the tensor's mean and std
    assert verify.redraw_stats(token) == verify._global_stats(token)
    rows = torch.randn(compare.CHANNEL_MIN_ROWS - 1, 64, generator=gen)
    assert verify.redraw_stats(rows) == verify._global_stats(rows)
    x = torch.randn(64, 8, generator=gen) * 0.1
    x[:, 5] = 0
    x[:4, 5] = 50.0  # four values: too few for a channel of its own
    mean, std = verify.redraw_stats(x)
    g_mean, g_std = verify._global_stats(x)
    assert isinstance(mean, torch.Tensor) and isinstance(std, torch.Tensor)
    assert float(mean[0, 5]) == pytest.approx(g_mean) and float(std[0, 5]) == pytest.approx(g_std)
    assert float(std[0, 0]) == pytest.approx(float(x[:, 0].std(unbiased=False)), rel=1e-4)


def _rotary(position: float, dim: int = 32, scale: float = 1.0, dtype=torch.bfloat16):
    inv = 1e6 ** (-torch.arange(0, dim, 2).float() / dim)
    emb = torch.cat((position * inv, position * inv))[None, None]
    return (emb.cos() * scale).to(dtype), (emb.sin() * scale).to(dtype)


@pytest.mark.parametrize(
    ("position", "scale", "dtype"),
    [(37.0, 1.0, torch.bfloat16), (0.0, 1.0, torch.bfloat16), (513.0, 1.2, torch.float32)],
)
def test_rotary_tables_stay_and_scale_with_the_scaled_checks(position, scale, dtype):
    """cos / sin of the positions (scaled by YaRN's attention factor or not; position 0:
    cos 1, sin 0) are kept by the redraw, like the integer positions they come from; the
    scaled checks multiply them as before (#175)."""
    cos, sin = _rotary(position, scale=scale, dtype=dtype)
    hidden = torch.randn(1, 1, 64)
    k, v = torch.randn(2, 1, 2, 40, 32).unbind(0)  # same shape, not a rotary pair
    inputs = ((hidden,), {"position_embeddings": (cos.clone(), sin.clone()), "k": k, "v": v})
    assert verify.rotary_tables([hidden, cos, sin, k, v]) == {id(cos), id(sin)}
    assert verify.perturb_(inputs, torch.Generator().manual_seed(0), "normal") == 3
    kept = inputs[1]["position_embeddings"]
    assert torch.equal(kept[0], cos) and torch.equal(kept[1], sin)
    assert verify.scale_(inputs, 3.0) == 5
    assert torch.equal(inputs[1]["position_embeddings"][0], cos * 3)


# ------------------------------------------------------------------ a synthetic decoder

HEADS, KV_HEADS, HEAD_DIM, HIDDEN, TOKENS = 8, 2, 32, 256, 256


class Rms(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        return (y * self.weight.float()).to(x.dtype)


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), -1) * sin


class OutlierDecode(nn.Module):
    """A Qwen3-like attention decode step (pre-norm, q / k norms, RoPE, GQA, KV cache as
    input): ``k_norm`` has two huge channels (Qwen3-0.6B layer 0: weight 96 and 38 at
    channels 50 and 115) that ``q_norm`` barely reads, so the cached keys carry a large
    per-head offset there that the attention logits hardly see."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = Rms(HIDDEN)
        self.q_proj = nn.Linear(HIDDEN, HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(HEADS * HEAD_DIM, HIDDEN, bias=False)
        self.q_norm, self.k_norm = Rms(HEAD_DIM), Rms(HEAD_DIM)

    def forward(self, x, position_embeddings, k_cache, v_cache):
        cos, sin = (t[:, None] for t in position_embeddings)
        h = self.norm(x)
        q = self.q_norm(self.q_proj(h).view(1, 1, HEADS, HEAD_DIM)).transpose(1, 2)
        k = self.k_norm(self.k_proj(h).view(1, 1, KV_HEADS, HEAD_DIM)).transpose(1, 2)
        v = self.v_proj(h).view(1, 1, KV_HEADS, HEAD_DIM).transpose(1, 2)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        group = HEADS // KV_HEADS
        keys = torch.cat((k_cache, k), 2).repeat_interleave(group, 1)
        values = torch.cat((v_cache, v), 2).repeat_interleave(group, 1)
        p = ((q.float() @ keys.float().transpose(-1, -2)) * HEAD_DIM**-0.5).softmax(-1)
        out = (p @ values.float()).to(x.dtype).transpose(1, 2).reshape(1, 1, -1)
        return x + self.o_proj(out)


def _decoder() -> tuple[OutlierDecode, list[tuple[tuple, dict, int]]]:
    """The module and two decode steps (256 and 272 cached tokens): a residual as small
    as layer 0's (embeddings), K channels 13 and 14 with a per-head offset of ±50-60 / ±20
    and std 15 / 5 (the rest N(0, 1), so the cache's global std is 15)."""
    gen = torch.Generator().manual_seed(0)
    module = OutlierDecode()
    with torch.no_grad():
        for lin in (module.q_proj, module.k_proj, module.v_proj, module.o_proj):
            lin.weight.copy_(torch.randn(lin.weight.shape, generator=gen) * lin.in_features**-0.5)
        module.k_norm.weight[13], module.k_norm.weight[14] = 60.0, 20.0
        module.q_norm.weight[13], module.q_norm.weight[14] = 0.02, 0.02  # Qwen3: |q[50]| ~ 0.01
    calls = []
    for tokens, count in ((TOKENS, 21), (TOKENS + 16, 0)):
        k = torch.randn(1, KV_HEADS, tokens, HEAD_DIM, generator=gen)
        k[..., 13] = torch.tensor([60.0, -50.0])[:, None] + 15 * k[..., 13]
        k[..., 14] = torch.tensor([-20.0, 25.0])[:, None] + 5 * k[..., 14]
        v = torch.randn(1, KV_HEADS, tokens, HEAD_DIM, generator=gen) * 0.5
        x = torch.randn(1, 1, HIDDEN, generator=gen) * 0.03
        bf = torch.bfloat16
        args = (x.to(bf), _rotary(float(tokens)), k.to(bf), v.to(bf))
        calls.append((args, {}, count))
    return module.to(torch.bfloat16).eval(), calls


@pytest.fixture(scope="module")
def decoder():
    return _decoder()


DECODER_CANDIDATE = """import torch
from kernel_agent.kernels import quant

BUG = {bug!r}
HEADS, KV_HEADS, HEAD_DIM = {heads}, {kv_heads}, {head_dim}


def rms(x, weight):
    y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    return (y * weight.float()).to(x.dtype)


class Linear(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        w = weight.detach().clone()
        if BUG == "row 0 skipped":  # an off-by-one row loop: output channel 0 never written
            w[0] = 0
        self.q, self.s = quant.quantize_fp8(w)
        if BUG and BUG.startswith("scales x"):
            self.s = self.s * float(BUG.removeprefix("scales x"))
        self.w = quant.dequantize_fp8(self.q, self.s, w.dtype)
        self.cache = {{}}

    def forward(self, x):
        if BUG != "cached activation scales":
            return torch.nn.functional.linear(x, self.w)
        _, xs = quant.quantize_fp8_activations(x)  # W8A8, per-token scales cached per shape
        xs = self.cache.setdefault(tuple(x.shape), xs)
        a = x.detach().reshape(-1, x.shape[-1]).float()
        xq = (a / xs[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        y = (xq.float() * xs[:, None]) @ (self.q.float() * self.s[:, None]).T
        return y.to(x.dtype).reshape(*x.shape[:-1], -1)


class Decode(torch.nn.Module):
    def __init__(self, ref):
        super().__init__()
        q = ref.q_proj.weight.detach().clone()
        if BUG == "q heads 0 and last swapped":  # a head-layout bug
            q[:HEAD_DIM], q[-HEAD_DIM:] = q[-HEAD_DIM:].clone(), q[:HEAD_DIM].clone()
        self.q_proj, self.o_proj = Linear(q), Linear(ref.o_proj.weight)
        self.k_proj, self.v_proj = Linear(ref.k_proj.weight), Linear(ref.v_proj.weight)
        self.norms = [m.weight.detach() for m in (ref.norm, ref.q_norm, ref.k_norm)]

    def forward(self, x, position_embeddings, k_cache, v_cache):
        norm, q_norm, k_norm = self.norms
        cos, sin = (t[:, None] for t in position_embeddings)
        half = HEAD_DIM // 2

        def rope(t):
            return t * cos + torch.cat((-t[..., half:], t[..., :half]), -1) * sin

        h = rms(x, norm)
        q = rms(self.q_proj(h).view(1, 1, HEADS, HEAD_DIM), q_norm).transpose(1, 2)
        k = rms(self.k_proj(h).view(1, 1, KV_HEADS, HEAD_DIM), k_norm).transpose(1, 2)
        v = self.v_proj(h).view(1, 1, KV_HEADS, HEAD_DIM).transpose(1, 2)
        keys = torch.cat((k_cache, rope(k)), 2).repeat_interleave(HEADS // KV_HEADS, 1)
        values = torch.cat((v_cache, v), 2)
        if BUG == "KV head 0 skipped":  # an attention loop that never reads KV head 0
            values = torch.cat((torch.zeros_like(values[:, :1]), values[:, 1:]), 1)
        values = values.repeat_interleave(HEADS // KV_HEADS, 1)
        p = ((rope(q).float() @ keys.float().transpose(-1, -2)) * HEAD_DIM**-0.5).softmax(-1)
        out = (p @ values.float()).to(x.dtype).transpose(1, 2).reshape(1, 1, -1)
        return x + self.o_proj(out)


def build(reference):
    return Decode(reference)
"""


def _candidate(tmp_path: Path, bug: str | None = None):
    path = tmp_path / f"cand_{(bug or 'ok').replace(' ', '_')}.py"
    path.write_text(
        DECODER_CANDIDATE.format(bug=bug, heads=HEADS, kv_heads=KV_HEADS, head_dim=HEAD_DIM)
    )
    namespace: dict = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    return path, namespace["build"]


def _redrawn_failures(decoder, build, quality: str, draws: int) -> int:
    """Failed draws of the candidate against the reference on ``draws`` redrawn copies of
    the first decode step (normal, then uniform / Laplace / log-normal draws)."""
    module, calls = decoder
    candidate = build(module)
    tier = compare.tier_for(quality, "fp8_weights")
    gen = torch.Generator().manual_seed(1)
    failed = 0
    for i in range(draws):
        args = copy.deepcopy(calls[0][0])
        verify.perturb_(args, gen, "normal" if i % 2 == 0 else "mix")
        with torch.inference_mode():
            ref, new = module(*args), candidate(*args)
        failed += not compare.compare_tensors("y", ref, new, tier=tier, perturbed=True)["ok"]
    return failed


QUALITIES = ("near-lossless", "relaxed")


def _old_redraw(monkeypatch) -> None:
    """The redraw before #198: every tensor from its global mean and std, rotary tables
    redrawn too."""
    monkeypatch.setattr(verify, "redraw_stats", verify._global_stats)
    monkeypatch.setattr(verify, "rotary_tables", lambda tensors: set())


def test_outlier_key_channels_redrawn_per_channel_pass_the_reference_math(
    tmp_path, decoder, monkeypatch
):
    _, build = _candidate(tmp_path)
    for quality in QUALITIES:
        assert _redrawn_failures(decoder, build, quality, 60) == 0, quality
    _old_redraw(monkeypatch)  # what #198 fixed: most draws failed honest FP8 weights
    assert _redrawn_failures(decoder, build, "near-lossless", 60) >= 20
    assert _redrawn_failures(decoder, build, "relaxed", 60) >= 2


def _capture(tmp_path: Path, decoder, quality: str, precision: str = "fp8_weights"):
    module, calls = decoder
    path = tmp_path / f"decode.{precision}.{quality}.pt"
    tier = compare.tier_for(quality, precision)
    capture_calls(module, calls, path, tier=tier, precision=precision)
    return path


def test_honest_fp8_weights_pass_the_evaluator(tmp_path, decoder):
    """In relaxed mode (the default). Near-lossless rejects this step on its x 3 inputs
    (cos / sin and the cache x 3: logits x 9; relative L2 0.093), as before #198: the
    scaled checks do not redraw."""
    capture = _capture(tmp_path, decoder, "relaxed")
    path, _ = _candidate(tmp_path)
    result = evaluate(capture, path, device="cpu")
    assert result["status"] == "ok" and result["correct"], result
    assert "perturbed" in result["checks"] and "scaled" in result["checks"]


@pytest.mark.parametrize(
    "bug",
    [
        "scales x1.05",
        "scales x1.2",
        "row 0 skipped",
        "KV head 0 skipped",
        "q heads 0 and last swapped",
        "cached activation scales",  # captured inputs pass: redrawn (NL) or x 3 (relaxed) fail
    ],
)
@pytest.mark.parametrize("quality", QUALITIES)
def test_broken_decoders_still_fail(tmp_path, decoder, bug, quality):
    precision = "fp8_w8a8" if bug == "cached activation scales" else "fp8_weights"
    capture = _capture(tmp_path, decoder, quality, precision)
    path, _ = _candidate(tmp_path, bug)
    result = evaluate(capture, path, device="cpu")
    assert result["status"] in ("incorrect", "incorrect_perturbed"), result


# ------------------------------------------------------------------ Qwen3-0.6B on the GPU


def _qwen3_cached() -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True)
        return True
    except Exception:
        return False


@pytest.mark.gpu
@pytest.mark.skipif(not _qwen3_cached(), reason="needs Qwen/Qwen3-0.6B in the HF cache")
def test_qwen3_decoder_layer_fp8_weights_pass_the_redraw_on_gpu(monkeypatch):
    """Layer 0 of Qwen3-0.6B at decode after a 400-token prompt (its K cache has channels
    of RMS ~200 against a median ~1.6): FP8 weights' reference math passes every redrawn
    draw in relaxed mode, as it did not with the global redraw."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.bfloat16, local_files_only=True
    )
    model = model.cuda().eval()
    layer = model.model.layers[0]
    gen = torch.Generator().manual_seed(0)
    ids = torch.randint(100, 20000, (1, 400), generator=gen).cuda()
    seen: list = []
    with torch.inference_mode():
        cache = model(ids, use_cache=True).past_key_values
        hook = layer.register_forward_pre_hook(
            lambda mod, args, kwargs: seen.append(copy.deepcopy((args, kwargs))), with_kwargs=True
        )
        model(ids[:, -1:], past_key_values=cache, use_cache=True)
        hook.remove()
    args, kwargs = seen[0]
    keys = kwargs["past_key_values"].layers[0].keys
    rms = _channel_rms(keys)
    assert float(rms.max()) > 50 * float(rms.median())  # the outlier channels
    candidate = copy.deepcopy(layer)
    with torch.no_grad():
        for mod in candidate.modules():
            if isinstance(mod, nn.Linear):
                mod.weight.copy_(
                    quant.dequantize_fp8(*quant.quantize_fp8(mod.weight), mod.weight.dtype)
                )
    del model

    def failures(draws: int) -> int:
        monkeypatch.setattr(compare, "TIER", compare.tier_for("relaxed", "fp8_weights"))
        sync = torch.cuda.synchronize
        rng = torch.Generator(device="cuda").manual_seed(1)
        failed = 0
        for i in range(draws):
            a, k = copy.deepcopy((args, kwargs))
            verify.perturb_((a, k), rng, "normal" if i % 2 == 0 else "mix")
            found = verify._against_reference(layer, candidate, a, k, sync)
            failed += any(not c.get("ok") for c in found or [])
        return failed

    assert failures(20) == 0
    _old_redraw(monkeypatch)
    assert failures(20) >= 8
