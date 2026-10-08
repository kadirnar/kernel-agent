"""A decoder layer with a returned static KV cache, built from a captured decode layer (#206).

A ``transformers`` decoder layer (its attention calls ``past_key_values.update(k, v,
layer_idx)``) captured at decode with a ``DynamicCache`` (Qwen3-0.6B,
``runs/Qwen--Qwen3-0.6B/20261008-021850/.truth/captures/layer_decode.pt``) is wrapped so that
its KV cache is a static one the call returns updated, the inputs left as they were:
``StaticLayer(hidden_states, position_embeddings, cache_position, keys, values) -> (hidden,
keys.index_copy(2, cache_position, k), values.index_copy(...))`` (a functional
``index_copy``, as export-friendly decoders write their caches). Each captured case becomes
one: the same weights, hidden state and rotary tables, the layer's cached K / V of ``T``
tokens in the first ``T`` of ``2 T`` slots (the rest zeros), the token written at slot
``T`` (the attention reads slots ``0 .. T``: the cached tokens and the new one, as the
``DynamicCache`` call did). On the RTX 5070 Ti the hidden outputs and the new K / V rows
are bit-identical to the captured ``DynamicCache`` call's (4 cases: T = 512, 543, 574, and
301 at batch 2).

    python static_layer.py LAYER_DECODE.pt OUT.pt [--device cuda]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn

from kernel_agent.profiling.capture import capture_calls, load_capture


class Written:
    """The cache the layer's attention updates: the static ``keys`` / ``values`` with the new
    token's K / V written at ``position`` (functional copies); attention reads the slots up
    to it."""

    def __init__(self, keys: torch.Tensor, values: torch.Tensor, position: torch.Tensor):
        self.keys, self.values, self.position = keys, values, position

    def update(self, k: torch.Tensor, v: torch.Tensor, *_: Any, **__: Any):
        self.keys = self.keys.index_copy(2, self.position, k)
        self.values = self.values.index_copy(2, self.position, v)
        n = int(self.position[-1]) + 1
        return self.keys[:, :, :n], self.values[:, :, :n]


class StaticLayer(nn.Module):
    """``layer`` with a static KV cache it returns updated (the inputs left as they were)."""

    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states, position_embeddings, cache_position, keys, values):
        cache = Written(keys, values, cache_position)
        out = self.layer(
            hidden_states,
            position_embeddings=position_embeddings,
            past_key_values=cache,
            use_cache=True,
        )
        return out, cache.keys, cache.values


def calls(capture: dict[str, Any], layer_idx: int) -> list[tuple[tuple, dict, int]]:
    """The captured cases as calls of :class:`StaticLayer`: ``T`` cached tokens in ``2 T``
    slots, the new token written at slot ``T``."""
    found = []
    for case in capture["cases"]:
        (hidden,), kw = case["args"], case["kwargs"]
        cached = kw["past_key_values"].layers[layer_idx]
        keys, values = (
            torch.cat((t, torch.zeros_like(t)), 2) for t in (cached.keys, cached.values)
        )
        at = torch.tensor([cached.keys.shape[2]], device=keys.device)
        args = (hidden, tuple(kw["position_embeddings"]), at, keys, values)
        found.append((args, {}, case["count"]))
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ns = parser.parse_args()
    import static_layer  # pickled as static_layer.StaticLayer, not __main__'s

    capture = load_capture(ns.capture, device=ns.device)
    layer = capture["module"].eval()
    module = static_layer.StaticLayer(layer).eval()
    found = calls(capture, layer.self_attn.layer_idx)
    capture_calls(module, found, ns.out)
    for (hidden, _, at, keys, _), _, count in found:
        shapes = f"hidden {tuple(hidden.shape)}, cache {tuple(keys.shape)}"
        print(f"case: {shapes}, slot {int(at)}, x{count}")
    print(f"wrote {ns.out}")


if __name__ == "__main__":
    main()
