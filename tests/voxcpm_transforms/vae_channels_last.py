"""AudioVAE decoder in a channels-last 2D layout, depthwise convs as fused stencils.

The fp32 VAE tail (~22 ms after compile_vae_decoder) spends ~3.8 ms in cuDNN
NCHW<->NHWC transposes around every conv (the 1D tensors are NCL, the TF32
tensor-core kernels want NHWC) and ~4.8 ms in the generic depthwise kernel
(`conv_depthwise2d_forward_kernel_generic`, 19 calls, ~1.6x DRAM SOL), plus a
separate pass for the Snake that precedes each depthwise conv.

This transform (supersedes compile_vae_decoder):
* removes weight_norm once (the effective weights g * v / |v| are a
  weight-only constant; the reference recomputes them every decode);
* runs the decoder on [N, C, 1, L] tensors in channels_last memory format, so
  the 1x1 / k=7 convs and the transposed convs go to cuDNN's NHWC kernels
  without layout transposes (conv1d == conv2d with a unit height);
* writes each depthwise causal conv (groups == C, stride 1) as its k shifted
  multiply-adds over the causally padded input, which inductor fuses with the
  preceding Snake and the pad into one pass (same fp32 math, different
  summation order);
* runs the 1x1 convs as GEMMs on the channels-last view ([N, 1, L, C] @ W^T),
  in TF32 like the reference cuDNN convs (scoped flag, restored after);
* compiles the decoder with dynamic=True (one graph for every latent length).
"""

import sys

import torch
import torch.nn.functional as F
from torch import nn


def _snake_forward(m):
    def forward(x):
        a = m.alpha.view(1, -1, 1, 1)
        return x + (a + 1e-9).reciprocal() * torch.sin(a * x).pow(2)

    return forward


def _conv_forward(m, pad):
    k, d = m.kernel_size[0], m.dilation[0]
    depthwise = m.groups > 1 and m.groups == m.in_channels == m.out_channels and m.stride[0] == 1 and m.padding[0] == 0
    if depthwise:
        c = m.out_channels
        # taps[j] = w[:, 0, j] as [1, C, 1, 1]
        m.register_buffer("_taps", m.weight.detach()[:, 0, :].t().reshape(k, 1, c, 1, 1).contiguous(), persistent=False)
        m.register_buffer(
            "_bias4", (m.bias.detach() if m.bias is not None else m.weight.new_zeros(c)).view(1, c, 1, 1), persistent=False
        )

        def forward(x):
            xp = F.pad(x, (pad, 0))
            n = xp.shape[-1] - (k - 1) * d
            y = xp[..., 0:n] * m._taps[0]
            for j in range(1, k):
                y = y + xp[..., j * d : j * d + n] * m._taps[j]
            return y + m._bias4

        return forward

    if k == 1 and m.groups == 1 and m.stride[0] == 1 and m.padding[0] == 0 and pad == 0:
        # 1x1 conv on a channels-last [N, C, 1, L] tensor == [N, 1, L, C] @ W^T
        m.register_buffer("_w2", m.weight.detach()[:, :, 0].contiguous(), persistent=False)

        def forward(x):
            return F.linear(x.permute(0, 2, 3, 1), m._w2, m.bias).permute(0, 3, 1, 2)

        return forward

    m.register_buffer(
        "_w4", m.weight.detach().unsqueeze(2).contiguous(memory_format=torch.channels_last), persistent=False
    )

    def forward(x):
        return F.conv2d(
            F.pad(x, (pad, 0)),
            m._w4,
            m.bias,
            stride=(1, m.stride[0]),
            padding=(0, m.padding[0]),
            dilation=(1, d),
            groups=m.groups,
        )

    return forward


def _tconv_forward(m, trim):
    m.register_buffer(
        "_w4", m.weight.detach().unsqueeze(2).contiguous(memory_format=torch.channels_last), persistent=False
    )

    def forward(x):
        y = F.conv_transpose2d(
            x,
            m._w4,
            m.bias,
            stride=(1, m.stride[0]),
            padding=(0, m.padding[0]),
            output_padding=(0, m.output_padding[0]),
            groups=m.groups,
            dilation=(1, m.dilation[0]),
        )
        return y[..., :-trim]

    return forward


def _cond_forward(m):
    def forward(x, sr_cond):
        if m.cond_type in ("scale_bias", "scale_bias_init"):
            x = x * m.scale_embed(sr_cond)[:, :, None, None] + m.bias_embed(sr_cond)[:, :, None, None]
        else:  # "add"
            x = x + m.cond_embed(sr_cond)[:, :, None, None]
        return m.out_layer(x)

    return forward


def apply(workload) -> None:
    vae = workload.model.audio_vae
    dec = getattr(vae.decoder, "_orig_mod", vae.decoder)
    if not next(dec.parameters()).is_cuda:
        return
    lib = sys.modules[type(dec).__module__]
    mods = list(dec.modules())
    supported = (
        lib.CausalDecoder, lib.CausalDecoderBlock, lib.CausalResidualUnit, lib.Snake1d, lib.CausalConv1d,
        lib.CausalTransposeConv1d, lib.SampleRateConditionLayer, nn.Sequential, nn.ModuleList, nn.Identity,
        nn.Tanh, nn.Embedding,
    )
    if any(not isinstance(m, supported) for m in mods) or any(
        isinstance(m, lib.SampleRateConditionLayer) and m.cond_type == "concat" for m in mods
    ):
        return  # an unknown layer would see the 4D layout: keep the reference decoder

    with torch.no_grad():
        for m in mods:
            if hasattr(m, "weight_g"):
                torch.nn.utils.remove_weight_norm(m)
        for m in mods:
            if isinstance(m, lib.Snake1d):
                m.forward = _snake_forward(m)
            elif isinstance(m, lib.CausalConv1d):
                pad = m._CausalConv1d__padding * 2 - m._CausalConv1d__output_padding
                m.forward = _conv_forward(m, pad)
            elif isinstance(m, lib.CausalTransposeConv1d):
                trim = m._CausalTransposeConv1d__padding * 2 - m._CausalTransposeConv1d__output_padding
                m.forward = _tconv_forward(m, trim)
            elif isinstance(m, lib.SampleRateConditionLayer):
                m.forward = _cond_forward(m)

    base_forward = type(dec).forward

    def forward(x, sr_cond=None):
        x4 = x.unsqueeze(2).contiguous(memory_format=torch.channels_last)
        return base_forward(dec, x4, sr_cond).squeeze(2)

    dec.forward = forward
    vae.decoder = _TF32Scope(torch.compile(dec, dynamic=True))


class _TF32Scope(nn.Module):
    """The reference convs run in TF32 (cuDNN's default); the 1x1 convs that
    became GEMMs get the same precision through a scoped cuBLAS TF32 flag."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, *args):
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32
        try:
            return self.inner(*args)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
