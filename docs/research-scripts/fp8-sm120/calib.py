"""Static (calibrated) vs dynamic activation scales on the whole VoxCPM2 model.

Text A calibrates (per nn.Linear: max |x| per input channel and per tensor over every call),
text B tests: for every 3rd call of every nn.Linear, the relative L2 error of the quantised
activation itself (dynamic per token, 1x128 groups, MXFP8 ceil, static per tensor from A)
and how often the static scale saturates. Also the KV cache: its length, bytes per decode
step against the weights, and static per-tensor K/V scales from A applied to B."""

import collections
import json
import os
import sys

import torch
from torch import nn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fp8lib import fq_groups, fq_mx, fq_tensor  # noqa: E402

from kernel_agent.workloads.voxcpm import NATURAL_TEXT, TEXT

PATCHES = 30
torch.backends.cuda.matmul.allow_tf32 = False


def load():
    from huggingface_hub import snapshot_download
    from voxcpm.model.voxcpm2 import VoxCPM2Model

    path = snapshot_download("openbmb/VoxCPM2")
    return VoxCPM2Model.from_local(path, optimize=False, device="cuda")


def generate(model, text, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        model.generate(target_text=text, min_len=PATCHES, max_len=PATCHES, inference_timesteps=10,
                       cfg_value=2.0, retry_badcase=False, retry_badcase_ratio_threshold=float(PATCHES))


def group_of(name):
    for g in ("feat_decoder", "feat_encoder", "residual_lm", "base_lm"):
        if g in name:
            return {"feat_decoder": "LocDiT", "feat_encoder": "LocEnc", "residual_lm": "residual LM",
                    "base_lm": "base LM"}[g]
    return None


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def main():
    model = load()
    linears = {n: m for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and group_of(n) and m.in_features % 128 == 0}
    calib = {}
    hooks = []

    def pre_a(mod, args, name):
        x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
        a = x.abs().amax(0)
        st = calib.setdefault(name, {"amax": torch.zeros_like(a), "tmax": 0.0, "rows": x.shape[0]})
        st["amax"] = torch.maximum(st["amax"], a)
        st["tmax"] = max(st["tmax"], float(a.max()))

    for n, m in linears.items():
        hooks.append(m.register_forward_pre_hook(lambda mod, a, n=n: pre_a(mod, a, n)))
    generate(model, TEXT, 0)
    for h in hooks:
        h.remove()
    kv_a = model.base_lm.kv_cache
    len_a = kv_a.current_length
    kv_amax_a = kv_a.kv_cache[:, :, :, :, :len_a].float().abs().amax(dim=(2, 3, 4, 5))  # [2, L]

    # outlier channels per Linear type (calibration text)
    print("## input-channel outliers (text A): max channel amax / median channel amax")
    by_type = collections.defaultdict(list)
    for n, st in calib.items():
        key = (group_of(n), n.rsplit(".", 1)[-1])
        ratio = float(st["amax"].max() / st["amax"].median().clamp_min(1e-30))
        by_type[key].append((ratio, int(st["amax"].argmax()), n))
    for key, vals in sorted(by_type.items()):
        vals.sort()
        print(f"  {key[0]:12s} {key[1]:10s} median ratio {vals[len(vals) // 2][0]:8.1f}  max "
              f"{vals[-1][0]:8.1f} (channel {vals[-1][1]}, {vals[-1][2]})")

    stats = collections.defaultdict(lambda: collections.defaultdict(list))
    counter = collections.Counter()

    def pre_b(mod, args, name):
        counter[name] += 1
        if counter[name] % 3:
            return
        x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
        if float(x.abs().max()) == 0:
            return
        key = (group_of(name), name.rsplit(".", 1)[-1])
        st = stats[key]
        tmax = calib[name]["tmax"]
        st["per token"].append(rel(fq_groups(x, x.shape[1]), x))
        st["1x128"].append(rel(fq_groups(x, 128), x))
        st["MXFP8 ceil"].append(rel(fq_mx(x, ceil=True), x))
        st["per tensor dynamic"].append(rel(fq_tensor(x), x))
        st["static (A amax)"].append(rel(fq_tensor(x, torch.tensor(tmax / 448, device="cuda")), x))
        st["static sat calls"].append(float(x.abs().max()) > tmax)
        st["static sat elems"].append(float((x.abs() > tmax).float().mean()))
        st["amax B / amax A"].append(float(x.abs().max()) / tmax)

    for n, m in linears.items():
        hooks.append(m.register_forward_pre_hook(lambda mod, a, n=n: pre_b(mod, a, n)))
    generate(model, NATURAL_TEXT, 1)
    for h in hooks:
        h.remove()

    print("\n## text B: activation quantisation error (rel L2 of x itself; mean / max over calls)")
    cols = ["per token", "1x128", "MXFP8 ceil", "per tensor dynamic", "static (A amax)"]
    print("| module | linear | calls | " + " | ".join(cols) + " | static saturates (calls) | max amax B/A |")
    print("|---" * (len(cols) + 5) + "|")
    out = {}
    for key in sorted(stats):
        st = stats[key]
        cells = [f"{sum(st[c]) / len(st[c]):.4f} / {max(st[c]):.4f}" for c in cols]
        sat = sum(st["static sat calls"]) / len(st["static sat calls"])
        print(f"| {key[0]} | {key[1]} | {len(st['per token'])} | " + " | ".join(cells)
              + f" | {sat * 100:.1f} % | {max(st['amax B / amax A']):.2f} |")
        out[f"{key[0]}/{key[1]}"] = {c: [sum(st[c]) / len(st[c]), max(st[c])] for c in cols}

    kv = model.base_lm.kv_cache
    len_b = kv.current_length
    k = kv.kv_cache[:, :, :, :, :len_b].float()  # [2, L, B, H, T, D]
    kv_amax_b = k.abs().amax(dim=(2, 3, 4, 5))
    sat = (kv_amax_b > kv_amax_a).float().mean(dim=1)
    errs = {"static per-tensor (A)": [], "per-tensor dynamic": [], "per token+head": []}
    for i in range(2):
        for layer in range(k.shape[1]):
            t = k[i, layer]
            errs["static per-tensor (A)"].append(rel(fq_tensor(t, kv_amax_a[i, layer] / 448), t))
            errs["per-tensor dynamic"].append(rel(fq_tensor(t), t))
            s = (t.abs().amax(-1, keepdim=True) / 448).clamp_min(1e-30)
            errs["per token+head"].append(rel((t / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s, t))
    n_layers = k.shape[1]
    kv_bytes = 2 * n_layers * k.shape[3] * k.shape[5] * 2  # bf16 per token, base LM
    res = model.residual_lm.kv_cache
    print(f"\n## KV cache: base LM {n_layers} layers, length A {len_a}, B {len_b} tokens "
          f"(residual LM {res.kv_cache.shape[1]} layers, length {res.current_length}); "
          f"{kv_bytes / 1024:.0f} KB per token (bf16, base LM)")
    print(f"  K/V amax over layers (A): K {float(kv_amax_a[0].min()):.2f}..{float(kv_amax_a[0].max()):.2f}, "
          f"V {float(kv_amax_a[1].min()):.3f}..{float(kv_amax_a[1].max()):.3f}; layers where B exceeds "
          f"A's static scale: K {float(sat[0]) * 100:.0f} %, V {float(sat[1]) * 100:.0f} %")
    for name, v in errs.items():
        print(f"  e4m3 {name:24s} rel L2 of K/V: mean {sum(v) / len(v):.4f} max {max(v):.4f}")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "calib.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
