"""Fake-quant (fp32) reference math of the FP8 scaling recipes compared in docs/FP8.md."""

import contextlib

import torch
import torch.nn.functional as F
from torch import nn

E4M3 = torch.float8_e4m3fn
FMAX = 448.0


def q8(v):
    """Round fp32 values (already divided by their scale) to e4m3 and back (RNE, clamped)."""
    return v.clamp(-FMAX, FMAX).to(E4M3).float()


def _safe(s):
    return torch.where(s > 0, s, torch.ones_like(s))


def fq_groups(x, group):
    """Per-row groups of `group` along the last dim (group == K: per row / per token)."""
    R, K = x.shape
    g = x.view(R, K // group, group)
    s = _safe(g.abs().amax(-1, keepdim=True) / FMAX)
    return (q8(g / s) * s).view(R, K)


def fq_tensor(x, scale=None):
    s = _safe(x.abs().amax() / FMAX) if scale is None else scale
    return q8(x / s) * s


def fq_block(w, b=128):
    N, K = w.shape
    g = w.view(N // b, b, K // b, b)
    s = _safe(g.abs().amax(dim=(1, 3), keepdim=True) / FMAX)
    return (q8(g / s) * s).view(N, K)


def fq_mx(x, ceil=False):
    """MXFP8: e4m3 elements, one power-of-two (e8m0) scale per 32. OCP rule (floor):
    2^(floor(log2 amax) - 8), block maxima above 448 saturate; ceil: the smallest power
    of two that keeps the block within 448 (no saturation, up to 1 bit less precision)."""
    R, K = x.shape
    g = x.view(R, K // 32, 32)
    amax = g.abs().amax(-1, keepdim=True).clamp_min(2.0**-120)
    if ceil:
        e = torch.ceil(torch.log2(amax / FMAX))
    else:
        e = torch.floor(torch.log2(amax)) - 8
    s = torch.exp2(e)
    return (q8(g / s) * s).view(R, K)


# recipe -> (weight quantiser, activation quantiser); "static" scales come from calibration
RECIPES = {
    "bf16 (reference math, fp32)": ("none", "none"),
    "FP8 weights only (per channel)": ("chan", "none"),
    "W8A8 tensorwise (per tensor x per tensor)": ("tensor", "tensor"),
    "W8A8 rowwise (per channel x per token)": ("chan", "token"),
    "W8A8 per channel x 1x128 groups": ("chan", "g128"),
    "W8A8 blockwise (128x128 x 1x128)": ("block", "g128"),
    "W8A8 1x128 x 1x128": ("g128", "g128"),
    "MXFP8 OCP floor scale (1x32 x 1x32)": ("mx", "mx"),
    "MXFP8 ceil scale (1x32 x 1x32)": ("mxc", "mxc"),
    "W8A8 static per-tensor activations (calibrated)": ("chan", "static"),
    "SmoothQuant a=0.5 + rowwise": ("chan", "token"),
    "SmoothQuant a=0.5 + static per-tensor": ("chan", "static"),
}


def quant_w(w, kind):
    return {
        "none": lambda: w,
        "chan": lambda: fq_groups(w, w.shape[1]),
        "tensor": lambda: fq_tensor(w),
        "block": lambda: fq_block(w),
        "g128": lambda: fq_groups(w, 128),
        "mx": lambda: fq_mx(w),
        "mxc": lambda: fq_mx(w, ceil=True),
    }[kind]()


def quant_x(x, kind, static_scale=None):
    if kind == "none":
        return x
    if kind == "token":
        return fq_groups(x, x.shape[1])
    if kind == "tensor":
        return fq_tensor(x)
    if kind == "g128":
        return fq_groups(x, 128)
    if kind == "mx":
        return fq_mx(x)
    if kind == "mxc":
        return fq_mx(x, ceil=True)
    if kind == "static":
        return fq_tensor(x, static_scale)
    raise ValueError(kind)


@contextlib.contextmanager
def patched_linears(root, recipe, calib, alpha=0.5, only=None):
    """Every nn.Linear under `root` computes y = Q(x) @ Q(W)^T in fp32 (one rounding to
    x's dtype) with the recipe's quantisers. `calib[name]`: {"amax": per-input-channel
    amax, "tmax": per-tensor amax} of that Linear's inputs on calibration data."""
    wk, xk = RECIPES[recipe]
    smooth = recipe.startswith("SmoothQuant")
    saved = {}
    for name, lin in root.named_modules():
        if not isinstance(lin, nn.Linear) or (only and not any(o in name for o in only)):
            continue
        w = lin.weight.detach().float()
        s = None
        static = None
        if smooth:
            amax = calib[name]["amax"].float().clamp_min(1e-5)
            s = amax.pow(alpha) / w.abs().amax(0).clamp_min(1e-5).pow(1 - alpha)
            w = w * s[None, :]
            static = (calib[name]["amax"] / s).amax() / FMAX
        elif xk == "static":
            static = calib[name]["tmax"] / FMAX
        wdq = quant_w(w, wk)
        bias = None if lin.bias is None else lin.bias.detach().float()

        def fwd(x, wdq=wdq, s=s, static=static, bias=bias, K=w.shape[1]):
            x2 = x.reshape(-1, K).float()
            if s is not None:
                x2 = x2 / s
            y = quant_x(x2, xk, static) @ wdq.T
            if bias is not None:
                y = y + bias
            return y.to(x.dtype).reshape(*x.shape[:-1], wdq.shape[0])

        saved[name] = lin
        lin.forward = fwd
    try:
        yield
    finally:
        for lin in saved.values():
            vars(lin).pop("forward", None)


def calibrate(root, run):
    """Per-Linear input statistics of `run()` (max over all calls)."""
    stats = {}
    hooks = []
    for name, lin in root.named_modules():
        if isinstance(lin, nn.Linear):

            def pre(mod, args, name=name):
                x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
                a = x.abs().amax(0)
                st = stats.setdefault(name, {"amax": torch.zeros_like(a), "tmax": a.new_zeros(())})
                st["amax"] = torch.maximum(st["amax"], a)
                st["tmax"] = torch.maximum(st["tmax"], a.amax())

            hooks.append(lin.register_forward_pre_hook(pre))
    try:
        with torch.no_grad():
            run()
    finally:
        for h in hooks:
            h.remove()
    return stats


# ---------------------------------------------------------------- FP8 attention / KV cache


def _fq_last2(t):
    """One scale per (batch, head) block over [tokens, head_dim] (FA3-style block scale)."""
    s = _safe(t.abs().amax(dim=(-2, -1), keepdim=True) / FMAX)
    return q8(t / s) * s


def _fq_token_head(t):
    s = _safe(t.abs().amax(dim=-1, keepdim=True) / FMAX)
    return q8(t / s) * s


def emulated_sdpa(q, k, v, attn_mask=None, is_causal=False, enable_gqa=False, scale=None,
                  mode="none", kv_scale=None):
    """fp32 attention with FP8 emulation. mode: 'none'; 'qk' (Q, K e4m3, one scale per
    (batch, head) block); 'qkpv' (also V per block and P = exp(s - max) as e4m3 x 1/448,
    FA3 / SageAttention style); 'kv_tensor' (K, V cache e4m3 with one static scale per
    tensor: kv_scale); 'kv_token' (K, V e4m3, one scale per token and head)."""
    qf, kf, vf = q.float(), k.float(), v.float()
    if mode in ("qk", "qkpv"):
        qf, kf = _fq_last2(qf), _fq_last2(kf)
    if mode == "qkpv":
        vf = _fq_last2(vf)
    if mode == "kv_tensor":
        kf = fq_tensor(kf, kv_scale[0])
        vf = fq_tensor(vf, kv_scale[1])
    if mode == "kv_token":
        kf, vf = _fq_token_head(kf), _fq_token_head(vf)
    rep = qf.shape[1] // kf.shape[1]
    if rep > 1:
        kf, vf = kf.repeat_interleave(rep, 1), vf.repeat_interleave(rep, 1)
    s = qf @ kf.transpose(-1, -2) * (scale if scale is not None else qf.shape[-1] ** -0.5)
    if attn_mask is not None:
        s = s.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else s + attn_mask
    if is_causal:
        L, S = s.shape[-2:]
        s = s.masked_fill(torch.ones(L, S, dtype=torch.bool, device=s.device).triu(1), float("-inf"))
    m = s.amax(-1, keepdim=True)
    p = torch.exp(s - m)
    denom = p.sum(-1, keepdim=True)
    if mode == "qkpv":
        p = q8(p * FMAX) / FMAX
    return ((p @ vf) / denom).to(q.dtype)


@contextlib.contextmanager
def patched_sdpa(mode, kv_scale=None, record=None):
    """F.scaled_dot_product_attention -> emulated_sdpa(mode); `record` (a list) gets the
    relative L2 error of each call's output against torch's own SDPA on the same inputs."""
    orig = F.scaled_dot_product_attention

    def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
             enable_gqa=False):
        out = emulated_sdpa(q, k, v, attn_mask, is_causal, enable_gqa, scale, mode, kv_scale)
        if record is not None:
            ref = orig(q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale,
                       enable_gqa=enable_gqa).float()
            record.append(float((out.float() - ref).norm() / ref.norm()))
        return out

    torch.nn.functional.scaled_dot_product_attention = sdpa
    try:
        yield
    finally:
        torch.nn.functional.scaled_dot_product_attention = orig
