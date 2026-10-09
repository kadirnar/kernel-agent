"""Example candidate (Helion): fused RMSNorm, a tile of rows per program.

Helion (``pip install helion``; "PyTorch with tiles", compiled to Triton) searches an
implicit space per kernel: the row tile, a persistent or looped reduction, indexing, warps,
stages. The kernel body is a plain function and ``build`` makes its kernel
(``kernel_agent.kernels.helion_tune.kernel``): Helion's default config in an evaluation (it
never tunes there), the tuned one when ``build`` gets ``helion_configs``.
``sweep_candidate(target_id, candidate, strategy="helion", hypothesis=...)`` runs Helion's
autotuner under the GPU lease and binds the config it finds into the snapshot, so
integration and export never tune again.

Helion lists A100 / H100 / B200; ``kernel-agent doctor`` compiles and runs this example on
any other GPU (its ``helion`` probe) and the backend is unavailable, with the reason, where
it fails. Measured on an RTX 5070 Ti (sm_120, Helion 1.4.0, #229): bit-exact with the
reference, 1.51x on the 2048-wide RMSNorm capture with Helion's default config, the decode
call 48 us (mostly host time: the Triton example 55 us, CUDA C++ 22 us in that session).

Contract: ``build(reference) -> nn.Module`` returning a drop-in replacement.
"""

import helion.language as hl
import torch
from torch import nn

from kernel_agent.kernels import helion_tune


def helion_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    m, _ = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        row = x[tile_m, :].to(torch.float32)
        variance = torch.mean(row * row, dim=-1)
        normed = row * torch.rsqrt(variance + eps)[:, None]
        # the reference's rounding: normalise in fp32, cast to the input dtype, then scale
        out[tile_m, :] = normed.to(x.dtype) * weight[:]
    return out


class HelionRMSNorm(nn.Module):
    def __init__(self, reference: nn.Module, kernel: object) -> None:
        super().__init__()
        self.weight = reference.weight
        self.eps = float(getattr(reference, "variance_epsilon", getattr(reference, "eps", 1e-6)))
        self.kernel = kernel  # this candidate's own Helion kernel and config

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        if hidden_states.dtype != self.weight.dtype:  # the reference would promote: its math
            h = hidden_states.float()
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
            return self.weight * h.to(hidden_states.dtype)
        x = hidden_states.reshape(-1, shape[-1]).contiguous()
        return self.kernel(x, self.weight, self.eps).view(shape)


def build(reference: nn.Module, helion_configs: dict | None = None) -> nn.Module:
    weight = getattr(reference, "weight", None)
    if weight is None or weight.dim() != 1:
        return reference  # unsupported instance: keep the original
    return HelionRMSNorm(reference, helion_tune.kernel(helion_rmsnorm, helion_configs))
