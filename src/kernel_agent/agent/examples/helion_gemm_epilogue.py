"""Example candidate (Helion): ``nn.Linear`` as one GEMM with its epilogue fused (the bias
added to the fp32 accumulator, one rounding to the output dtype).

Helion (``pip install helion``) searches the tile sizes of the three loops, their order, L2
grouping of the program ids, persistent programs, pointer / block_ptr / TMA indexing, warps
and stages of this one kernel body: ``sweep_candidate(target_id, candidate,
strategy="helion", hypothesis=...)`` runs its autotuner under the GPU lease (TMA where the
GPU has it, no warp specialisation on sm_120: kernel-agent's arch pruning) and binds the
config it finds into the snapshot. ``build`` makes the kernel from the plain function
(``kernel_agent.kernels.helion_tune.kernel``): Helion's default config in an evaluation (it
never tunes there), the tuned one with ``helion_configs``.

The weight is transposed once in ``build`` into a ``[K, N]`` copy (new tensors: the
reference's weights stay as they are), so the K loop reads both operands row-major. A
bf16 GEMM with fp32 accumulation sums in another order than cuBLAS: within the exact tier's
bf16 tolerance, not bit-identical. Calls on CPU or with other dtypes use the reference math.

Helion lists A100 / H100 / B200; on other GPUs ``kernel-agent doctor`` says whether it runs
(its ``helion`` probe). Measured on an RTX 5070 Ti (sm_120, Helion 1.4.0, #229) on a
``[704, 1024] x [1024, 4096]`` bf16 Linear: correct in the exact tier; Helion's default config
0.21x of cuBLAS, tuned by ``strategy="helion"`` in 175 s (64 x 64 x 32 tiles, 5 stages, L2
grouping 8, a TMA store) 0.80x: an example of the contract and the tuning, not a bf16 GEMM
that beats cuBLAS.

Contract: ``build(reference) -> nn.Module`` returning a drop-in replacement.
"""

import helion.language as hl
import torch
from torch import nn

from kernel_agent.kernels import helion_tune


def helion_linear(x: torch.Tensor, weight_t: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = weight_t.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], weight_t[tile_k, tile_n])
        # the epilogue: the bias in fp32, one rounding to the output dtype
        out[tile_m, tile_n] = (acc + bias[tile_n].to(torch.float32)).to(out.dtype)
    return out


class HelionLinear(nn.Module):
    def __init__(self, reference: nn.Linear, kernel: object) -> None:
        super().__init__()
        self.reference_weight = reference.weight
        self.reference_bias = reference.bias
        weight = reference.weight
        with torch.no_grad():
            self.weight_t = weight.t().contiguous()  # [K, N], packed once
            if reference.bias is not None:
                self.bias = reference.bias.detach().clone()
            else:  # no bias: the epilogue adds zeros (one kernel for both kinds of Linear)
                self.bias = weight.new_zeros(reference.out_features)
        self.kernel = kernel  # this candidate's own Helion kernel and config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda or x.dtype != self.weight_t.dtype:
            return nn.functional.linear(x, self.reference_weight, self.reference_bias)
        shape = x.shape
        y = self.kernel(x.reshape(-1, shape[-1]).contiguous(), self.weight_t, self.bias)
        return y.view(*shape[:-1], y.shape[-1])


def build(reference: nn.Module, helion_configs: dict | None = None) -> nn.Module:
    if not isinstance(reference, nn.Linear):
        return reference  # unsupported instance: keep the original
    return HelionLinear(reference, helion_tune.kernel(helion_linear, helion_configs))
