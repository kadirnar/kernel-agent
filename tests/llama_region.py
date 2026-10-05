"""Region fixtures for ``transformers``' LlamaDecoderLayer (SmolLM2, Llama): the
refactor of target ``resid_norm`` (the residual add after self_attn +
post_attention_layernorm), the same with the ops in the wrong order, and a Triton
fused add + RMSNorm candidate for the region module (no Claude involved)."""

REWRITE = '''import copy

from torch import nn
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


class Region_resid_norm(nn.Module):
    """Residual add after self_attn + post_attention_layernorm."""

    def __init__(self, norm):
        super().__init__()
        self.post_attention_layernorm = norm

    def forward(self, hidden_states, residual):
        hidden_states = residual + hidden_states
        return self.post_attention_layernorm(hidden_states), hidden_states


class RegionLlamaDecoderLayer(LlamaDecoderLayer):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states, residual = self.region(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def rewrite(parent):
    new = copy.copy(parent)
    new._modules = dict(parent._modules)
    new.__class__ = RegionLlamaDecoderLayer
    new.region = Region_resid_norm(parent.post_attention_layernorm)
    del new.post_attention_layernorm
    return new
'''

# Wrong order: the norm reads the attention output before the residual add.
WRONG_ORDER = REWRITE.replace(
    """        hidden_states = residual + hidden_states
        return self.post_attention_layernorm(hidden_states), hidden_states""",
    """        normed = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return normed, hidden_states""",
)

FUSED = '''"""Fused residual add + RMSNorm (Triton): one program per row, one launch per call."""

import torch
import triton
import triton.language as tl
from torch import nn


@triton.jit
def _add_rmsnorm(x_ptr, r_ptr, w_ptr, y_ptr, h_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)
    r = tl.load(r_ptr + row * n_cols + cols, mask=mask, other=0.0)
    # as torch: the add in fp32, rounded to the input dtype
    h = (r.to(tl.float32) + x.to(tl.float32)).to(x.dtype)
    tl.store(h_ptr + row * n_cols + cols, h, mask=mask)
    hf = h.to(tl.float32)
    var = tl.sum(hf * hf, axis=0) / n_cols
    y = (hf * tl.rsqrt(var + eps)).to(x.dtype).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * n_cols + cols, (w * y).to(x.dtype), mask=mask)


class FusedAddRMSNorm(nn.Module):
    def __init__(self, reference):
        super().__init__()
        norm = reference.post_attention_layernorm
        self.weight = norm.weight
        self.eps = float(norm.variance_epsilon)

    def forward(self, hidden_states, residual):
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1]).contiguous()
        r = residual.reshape(-1, shape[-1]).contiguous()
        y, h = torch.empty_like(x), torch.empty_like(x)
        n = x.shape[-1]
        _add_rmsnorm[(x.shape[0],)](
            x, r, self.weight, y, h, n, self.eps, BLOCK=triton.next_power_of_2(n), num_warps=4
        )
        return y.view(shape), h.view(shape)


def build(reference):
    if reference.post_attention_layernorm.weight.dim() != 1:
        return reference
    return FusedAddRMSNorm(reference)
'''

IDENTITY = "def build(reference):\n    return reference  # the rewrite alone\n"
