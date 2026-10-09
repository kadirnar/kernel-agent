"""KV-cache bytes in the decode rows of the ceilings table (#146, docs/FP8.md follow-up C):
the profiler reads the cache a decode call is given up to its position, once per subtree."""

from __future__ import annotations

import torch
from torch import nn

from kernel_agent.profiling import ceilings
from kernel_agent.profiling.profiler import ModuleTimer

B, H, S, D = 2, 2, 64, 8  # batch, KV heads, cache slots, head dim
SLOT_BYTES = 2 * B * H * D * 4  # K and V, fp32


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.o_proj = nn.Linear(16, 16, bias=False)

    def forward_step(
        self, x: torch.Tensor, position_id: int | torch.Tensor, kv_cache: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        k, _ = kv_cache
        k[:, :, position_id] = 1.0  # the step's own K, written in place
        return self.o_proj(x)


class DecoderLayer(nn.Module):
    """Passes its cache on: the layer and its attention read it once."""

    def __init__(self) -> None:
        super().__init__()
        self.self_attn = Attention()

    def forward_step(
        self, x: torch.Tensor, position_id: int | torch.Tensor, kv_cache: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        return self.self_attn.forward_step(x, position_id, kv_cache)


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(), DecoderLayer()])
        self.cache = [(torch.zeros(B, H, S, D), torch.zeros(B, H, S, D)) for _ in self.layers]

    def forward_step(self, x: torch.Tensor, position_id: int | torch.Tensor) -> torch.Tensor:
        for layer, cache in zip(self.layers, self.cache, strict=True):
            x = layer.forward_step(x, position_id, cache)
        return x


def test_decode_calls_count_their_kv_cache_up_to_the_position():
    model = Decoder()
    methods = {cls: ["forward_step"] for cls in (Attention, DecoderLayer, Decoder)}
    x = torch.randn(B, 16)
    with torch.inference_mode(), ModuleTimer({"m": model}, methods, cuda=False) as timer:
        for position in (4, 5, torch.tensor([6])):  # an int or a tensor of positions
            model.forward_step(x, position)
    stats = {c.cls: c for c in timer.class_stats()}
    slots = 5 + 6 + 7  # positions 4, 5, 6 attend to 5, 6, 7 cached positions
    (attn,) = stats["Attention"].work
    assert attn["phase"] == "decode" and attn["kv_bytes"] == 2 * slots * SLOT_BYTES
    (layer,) = stats["DecoderLayer"].work
    assert layer["kv_bytes"] == attn["kv_bytes"]  # not counted twice
    (decoder,) = stats["Decoder"].work
    assert decoder["kv_bytes"] == attn["kv_bytes"]  # held, not passed: from its layers
    (linear,) = stats["Linear"].work
    assert "kv_bytes" not in linear

    profile = {"hooked_wall_ms": 10.0, "classes": [asdict_(c) for c in stats.values()]}
    peaks = {"dram_gbps": 1.0, "tflops": {"float32": 1e6}}  # 1 GB/s: bytes are the floor
    table = ceilings.build(profile, peaks, 10.0)
    row = next(r for r in table["rows"] if r["cls"] == "Attention")
    memory = (row["weight_bytes"] + row["io_bytes"] + row["kv_bytes"]) / 1e6  # ms at 1 GB/s
    assert row["kv_bytes"] == attn["kv_bytes"] and row["floors"]["exact"] == round_(memory)
    md = ceilings.markdown(table, min_share=0.0)
    assert "| KV GB |" in md and "KV-cache reads are counted in decode rows" in md


class _CacheLayer:
    def __init__(self) -> None:
        self.keys = torch.zeros(B, H, S, D)
        self.values = torch.zeros(B, H, S, D)


class LayeredCache:
    """A cache object like transformers' ``Cache``: one entry per layer (``layers[i]``)."""

    def __init__(self, layers: int) -> None:
        self.layers = [_CacheLayer() for _ in range(layers)]


class ListCache:
    """The older layout: parallel ``key_cache`` / ``value_cache`` lists."""

    def __init__(self, layers: int) -> None:
        self.key_cache = [torch.zeros(B, H, S, D) for _ in range(layers)]
        self.value_cache = [torch.zeros(B, H, S, D) for _ in range(layers)]


class IndexedAttention(nn.Module):
    """Reads its layer (``layer_idx``) of the cache object it is given; the position comes in
    its ``**kwargs``."""

    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.o_proj = nn.Linear(16, 16, bias=False)

    def forward(
        self, x: torch.Tensor, past_key_values: object = None, **kwargs: object
    ) -> torch.Tensor:
        return self.o_proj(x)


class IndexedLayer(nn.Module):
    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = IndexedAttention(layer_idx)

    def forward(
        self, x: torch.Tensor, past_key_values: object = None, **kwargs: object
    ) -> torch.Tensor:
        return self.self_attn(x, past_key_values=past_key_values, **kwargs)


class IndexedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([IndexedLayer(0), IndexedLayer(1)])

    def forward(
        self, x: torch.Tensor, past_key_values: object = None, **kwargs: object
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, past_key_values=past_key_values, **kwargs)
        return x


def test_a_cache_object_counts_the_layer_of_each_call_once():
    """A cache object (every layer's K / V) passed down the model (#226): each attention call
    reads its own layer's (``layer_idx``) up to the position in its ``**kwargs``, not the
    whole object; the layer and the model, which pass it on, count those reads once."""
    for cache in (LayeredCache(2), ListCache(2)):
        model = IndexedModel()
        x = torch.randn(B, 1, 16)  # [batch, 1, ...]: a decode call
        with torch.inference_mode(), ModuleTimer({"m": model}, cuda=False) as timer:
            for position in (4, 9):
                model(x, past_key_values=cache, cache_position=torch.tensor([position]))
        stats = {c.cls: c for c in timer.class_stats()}
        (attn,) = stats["IndexedAttention"].work
        # 2 layers x (5 + 10 cached positions) x K and V of one layer
        assert attn["kv_bytes"] == 2 * (5 + 10) * SLOT_BYTES, type(cache).__name__
        assert stats["IndexedLayer"].work[0]["kv_bytes"] == attn["kv_bytes"]
        assert stats["IndexedModel"].work[0]["kv_bytes"] == attn["kv_bytes"]


def asdict_(stat: object) -> dict:
    from dataclasses import asdict

    return asdict(stat)  # type: ignore[call-overload]


def round_(value: float) -> float:
    return float(f"{value:.4g}")
