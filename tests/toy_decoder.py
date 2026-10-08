"""A tiny decoder-only LM (CPU) for the region-target tests: each layer has the
Llama layout (input norm → attention → residual add → post-attention norm → MLP
→ residual add), so the residual add + ``post_attention_layernorm`` across the
attention/MLP boundary is a fusion no single module covers.

Usable as ``harness.py``. ``create`` imports the classes under this module's own
name (``toy_decoder``), so a subprocess with ``tests/`` on its path can unpickle
captures of them. The ``*_REWRITE`` strings are refactor fixtures for target
``resid_norm`` and ``FUSED_CANDIDATE`` a kernel candidate for its region.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from kernel_agent.hub import Modality
from kernel_agent.workloads.base import Comparison, Workload, WorkloadSpec, cosine


class ToyRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dtype)


class ToyAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        b, t, d = hidden_states.shape
        q, k, v = self.qkv(hidden_states).view(b, t, 3, self.heads, -1).unbind(2)
        out = nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        return self.o_proj(out.transpose(1, 2).reshape(b, t, d))


class ToyMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gate_up = nn.Linear(dim, 4 * dim, bias=False)
        self.down = nn.Linear(2 * dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(nn.functional.silu(gate) * up)


class ToyDecoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.input_layernorm = ToyRMSNorm(dim)
        self.self_attn = ToyAttention(dim, heads)
        self.post_attention_layernorm = ToyRMSNorm(dim)
        self.mlp = ToyMLP(dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class ToyDecoder(nn.Module):
    def __init__(self, vocab: int, dim: int, heads: int, layers: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList(ToyDecoderLayer(dim, heads) for _ in range(layers))
        self.norm = ToyRMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.lm_head(self.norm(hidden_states))


class ToyDecoderWorkload(Workload):
    """Greedy generation of ``new_tokens`` tokens (the full sequence each step, no cache)."""

    modality = Modality.LLM
    defaults = {"vocab": 96, "dim": 32, "heads": 4, "layers": 2, "prompt_len": 8}
    defaults |= {"new_tokens": 3, "offset": 0}

    def load(self) -> None:
        torch.manual_seed(0)
        o = self.options
        self.model = ToyDecoder(o["vocab"], o["dim"], o["heads"], o["layers"]).eval()
        with torch.no_grad():
            for module in self.model.modules():
                if isinstance(module, ToyRMSNorm):
                    module.weight.uniform_(0.5, 1.5)
        self.model.to(self.device, self.dtype)

    def roots(self) -> dict[str, nn.Module]:
        return {"model": self.model}

    def make_inputs(self) -> torch.Tensor:
        n, vocab = int(self.options["prompt_len"]), int(self.options["vocab"])
        ids = (torch.arange(n) * 7 + int(self.options["offset"])) % vocab
        return ids.unsqueeze(0).to(self.device)

    def holdout_options(self, variant: int = 1) -> dict[str, Any] | None:
        # The ids wrap around the vocabulary (96 = 3 x 32): offset 3 * variant would make
        # every 32nd fresh input of the memoisation probe (variant >= 2) the held-out input
        # (offset 3) or the main one (0). 3 * variant + 1 never is, nor are two in a row.
        return {"offset": 3 * variant + (variant >= 2)}

    def run(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        ids, first = inputs, None
        for _ in range(int(self.options["new_tokens"])):
            logits = self.model(ids)[:, -1]
            first = logits if first is None else first
            ids = torch.cat([ids, logits.argmax(-1, keepdim=True)], dim=1)
        assert first is not None
        return {"tokens": ids[:, inputs.shape[1] :].cpu(), "first_logits": first.float().cpu()}

    def compare(self, reference: Any, candidate: Any) -> Comparison:
        same = torch.equal(reference["tokens"], candidate["tokens"])
        cos = cosine(reference["first_logits"], candidate["first_logits"])
        passed = same and cos >= 0.9999
        return Comparison(passed, {"cosine": round(cos, 6)}, "" if passed else "outputs differ")


def create(spec: WorkloadSpec) -> Workload:
    from toy_decoder import ToyDecoderWorkload as Importable  # pickles as `toy_decoder.*`

    return Importable(spec)


#: The refactor of target ``resid_norm``: the residual add after attention and the
#: post-attention norm move into ``Region_resid_norm`` (the same ops, same order).
GOOD_REWRITE = '''import copy

from toy_decoder import ToyDecoderLayer
from torch import nn


class Region_resid_norm(nn.Module):
    """Residual add after self_attn + post_attention_layernorm."""

    def __init__(self, norm):
        super().__init__()
        self.post_attention_layernorm = norm

    def forward(self, hidden_states, residual):
        hidden_states = residual + hidden_states
        return self.post_attention_layernorm(hidden_states), hidden_states


class RegionDecoderLayer(ToyDecoderLayer):
    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states, residual = self.region(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


def rewrite(parent):
    new = copy.copy(parent)
    new._modules = dict(parent._modules)
    new.__class__ = RegionDecoderLayer
    new.region = Region_resid_norm(parent.post_attention_layernorm)
    del new.post_attention_layernorm
    return new
'''

#: Wrong order of ops: the norm runs before the residual add.
WRONG_ORDER_REWRITE = GOOD_REWRITE.replace(
    """        hidden_states = residual + hidden_states
        return self.post_attention_layernorm(hidden_states), hidden_states""",
    """        normed = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return normed, hidden_states""",
)

#: The region module is attached but the parent never calls it.
UNCALLED_REWRITE = GOOD_REWRITE.replace(
    "        hidden_states, residual = self.region(hidden_states, residual)",
    """        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.region.post_attention_layernorm(hidden_states)""",
)

#: No ``Region_resid_norm`` class: the region module has another name.
MISNAMED_REWRITE = GOOD_REWRITE.replace("Region_resid_norm", "AddNorm")

#: A "fused kernel" for the region (plain torch: the pipeline runs on the CPU).
FUSED_CANDIDATE = """import torch
from torch import nn


class FusedAddNorm(nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.weight = reference.post_attention_layernorm.weight
        self.eps = reference.post_attention_layernorm.variance_epsilon

    def forward(self, hidden_states, residual):
        hidden_states = residual + hidden_states
        x = hidden_states.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(hidden_states.dtype), hidden_states


def build(reference):
    return FusedAddNorm(reference)
"""
