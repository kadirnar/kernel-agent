"""The FP4 producers of ``examples/triton_fp4_producers.py`` bit for bit against the reference
quantiser on real activations (#233 follow-up 1), on any CUDA GPU (no FP4 tensor cores
needed: the producers' e4m3 bits come from integer math).

Qwen3-0.6B (bf16, eager) on a fixed text: every decoder layer's RMSNorm inputs
(``input_layernorm``, ``post_attention_layernorm``: the residual stream, with its massive
activations) and the MLP's gate / up outputs. ``rmsnorm_fp4`` / ``silu_mul_fp4`` with ``out``
(the kernel's own bf16 output) must give ``quant.quantize_fp4_activations(out)`` bit for bit
(codes, e4m3 scales, outer scales; swizzled scales = ``swizzle_fp4_scales`` of them; MXFP4
too), and the kernel's bf16 output is compared with eager's (the share of equal elements).
Then synthetic rows at scales from 1e-30 to 1e30, the scale-rule guard on the captured
inputs, and the W4A4 MLP module (GEMMs emulated in fp32 where there are no FP4 tensor cores)
against the reference chain and through the evaluator in ``near-lossless-fp4a``.

    PYTHONPATH=src python verify_producers.py   (results/verify_producers.out: NVIDIA A10)
"""

import importlib.util
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

from kernel_agent import selftest
from kernel_agent.agent import prompts
from kernel_agent.kernels import quant, scale_guard
from kernel_agent.kernels.evaluate import run_evaluation

EXAMPLE = prompts.EXAMPLES_DIR / "triton_fp4_producers.py"
spec = importlib.util.spec_from_file_location("fp4_producers", EXAMPLE)
prod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prod)
sys.path.insert(0, str(Path(__file__).parent))
from norm_bias import TEXT  # noqa: E402  (the same texts as the norm-bias study)


def same(got, want) -> dict[str, bool]:
    got = [t.cpu() for t in got]
    want = [t.cpu() for t in want]
    return {
        "codes": torch.equal(got[0], want[0]),
        "scales": torch.equal(got[1].view(torch.uint8), want[1].view(torch.uint8)),
        "outer": torch.equal(got[2], want[2]),
    }


def check(name, fn, ref_input, eager, results):
    """fn(mx, swizzle, out) -> (codes, scales, outer); ref_input: the kernel's bf16 output."""
    out = torch.empty_like(eager)
    got = fn(False, False, out)
    want = quant.quantize_fp4_activations(out, "nvfp4")
    ok = same(got, want)
    swz = fn(False, True, None)
    ok["swizzled"] = torch.equal(
        swz[1].view(torch.uint8), quant.swizzle_fp4_scales(want[1]).view(torch.uint8)
    ) and torch.equal(swz[0], want[0])
    mx = fn(True, False, None)
    ok |= {f"mx {k}": v for k, v in same(mx, quant.quantize_fp4_activations(out, "mxfp4")).items()}
    results[name]["bit_exact"].append(all(ok.values()))
    if not all(ok.values()):
        print("  MISMATCH", name, ok)
    results[name]["h_equal"].append(float((out == eager).float().mean()))
    eager_q = quant.quantize_fp4_activations(eager, "nvfp4")
    results[name]["codes_vs_eager"].append(float((got[0] == eager_q[0]).float().mean()))
    # the opt-in per-token epilogue scale against outer * quant.fp4_bias_correction
    e = fn(False, True, None, True)[3]
    c = quant.fp4_bias_correction(out, *want)
    results[name]["factor_rel"].append(float(((e - want[2] * c) / (want[2] * c)).abs().max()))
    results[name]["factor"].extend(c.tolist())


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda"
    print(torch.cuda.get_device_name(), torch.__version__, "triton", prod.triton.__version__)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)
    model = model.to(dev).eval()
    seen = defaultdict(dict)

    def pre(name, key):
        def hook(_, args):
            seen[name][key] = args[0].detach().reshape(-1, args[0].shape[-1])

        return hook

    def post(name, key):
        def hook(_, args, out):
            seen[name][key] = out.detach().reshape(-1, out.shape[-1])

        return hook

    for i, layer in enumerate(model.model.layers):
        for norm in ("input_layernorm", "post_attention_layernorm"):
            m = getattr(layer, norm)
            m.register_forward_pre_hook(pre(f"{i}.{norm}", "x"))
            m.register_forward_hook(post(f"{i}.{norm}", "y"))
        layer.mlp.gate_proj.register_forward_hook(post(f"{i}.mlp", "g"))
        layer.mlp.up_proj.register_forward_hook(post(f"{i}.mlp", "u"))
        layer.mlp.down_proj.register_forward_pre_hook(pre(f"{i}.mlp", "h"))
    ids = tok(" ".join(TEXT), return_tensors="pt").input_ids.to(dev)
    with torch.inference_mode():
        model(ids)
    print(f"Qwen3-0.6B, {ids.shape[1]} tokens, {len(model.model.layers)} layers")
    results = defaultdict(lambda: defaultdict(list))
    crest = []
    for i, layer in enumerate(model.model.layers):
        for norm in ("input_layernorm", "post_attention_layernorm"):
            m, c = getattr(layer, norm), seen[f"{i}.{norm}"]
            x, y = c["x"], c["y"]
            crest.append(float((x.float().abs().amax(1) / x.float().pow(2).mean(1).sqrt()).max()))

            def fn(mx, swizzle, out, unbiased=False, x=x, m=m):
                eps = m.variance_epsilon
                return prod.rmsnorm_fp4(x, m.weight, eps, mx, swizzle, out, unbiased)

            check("rmsnorm_fp4", fn, None, y, results)
        c = seen[f"{i}.mlp"]

        def fn(mx, swizzle, out, unbiased=False, c=c):
            return prod.silu_mul_fp4(c["g"], c["u"], mx, swizzle, out, unbiased)

        check("silu_mul_fp4", fn, None, c["h"], results)
        merged = torch.cat([c["g"], c["u"]], dim=-1)  # a gate|up GEMM's merged output
        a = prod.silu_mul_fp4(merged)
        b = prod.silu_mul_fp4(c["g"], c["u"])
        results["silu_mul_fp4 merged"]["bit_exact"].append(all(same(a, b).values()))
        q = prod.quantize_rows(c["h"])
        results["quantize_rows"]["bit_exact"].append(
            all(same(q, quant.quantize_fp4_activations(c["h"])).values())
        )
    print(f"residual-stream token crest up to {max(crest):.0f} (massive activations)")
    for name, r in results.items():
        line = f"  {name:20s} bit for bit the reference in {sum(r['bit_exact'])}/{len(r['bit_exact'])}"
        if r["h_equal"]:
            line += (
                f"; bf16 output = eager's: {100 * min(r['h_equal']):.3f} % of the elements at "
                f"least (mean {100 * sum(r['h_equal']) / len(r['h_equal']):.3f} %); codes of "
                f"eager's output: {100 * min(r['codes_vs_eager']):.3f} % equal at least; bias "
                f"factor within {max(r['factor_rel']):.1e} of the reference (factors "
                f"{min(r['factor']):.4f} .. {max(r['factor']):.4f})"
            )
        print(line)

    # synthetic rows at scales from 1e-30 to 1e30: subnormal and zero scales, zero rows
    gen = torch.Generator().manual_seed(0)
    for rows, k in ((1, 16), (17, 48), (300, 1024), (129, 4096), (64, 16384)):
        x = torch.randn(rows, k, generator=gen) * torch.logspace(-30, 30, rows)[:, None]
        x[0, :16] = 0
        if rows > 3:
            x[1, 5] = 1e3
            x[2] = 0
            x[3, :] = torch.randn(k, generator=gen) * 1e-39  # bf16 subnormals
        x = x.to(torch.bfloat16)
        xc = x.to(dev)
        ok = same(prod.quantize_rows(xc), quant.quantize_fp4_activations(x, "nvfp4"))
        sw = prod.quantize_rows(xc, swizzle=True)[1].view(torch.uint8).cpu()
        ok["swizzled"] = torch.equal(
            sw, quant.swizzle_fp4_scales(quant.quantize_fp4_activations(x)[1]).view(torch.uint8)
        )
        if k % 32 == 0:
            mx = same(prod.quantize_rows(xc, mx=True), quant.quantize_fp4_activations(x, "mxfp4"))
            ok |= {f"mx {n}": v for n, v in mx.items()}
        print(f"  synthetic {rows} x {k} (1e-30 .. 1e30, CPU reference): {all(ok.values())}")
        assert all(ok.values()), ok

    # the scale-rule guard on the captured residual stream and its stress input
    x = seen["5.post_attention_layernorm"]["x"]
    report = scale_guard.check(prod, [{"args": (x,), "kwargs": {}, "count": 1}], "fp4_w4a4")
    print("  scale-rule guard:", {k: report[k] for k in ("ok", "checked", "captured", "stress")})

    # the W4A4 MLP module: emulated GEMMs here, against the reference chain (and unbiased)
    for layer in (5, 14, 20):
        mlp = model.model.layers[layer].mlp
        x = seen[f"{layer}.post_attention_layernorm"]["y"]
        for unbiased in (False, True):
            module = prod.build(mlp, unbiased=unbiased)
            with torch.inference_mode():
                got = module(x)
                gu_c, gu_s, gu_t, d_c, d_s, d_t = module.reference_args
                gu = quant.fp4_w4a4_linear(x, gu_c, gu_s, gu_t, unbiased=unbiased)
                g, u = gu.chunk(2, dim=-1)
                h = nn.functional.silu(g) * u
                want = quant.fp4_w4a4_linear(h, d_c, d_s, d_t, unbiased=unbiased)
                exact = mlp(x).float()
            rel = float((got.float() - want.float()).norm() / want.float().norm())
            err = float((got.float() - exact).norm() / exact.norm())
            gain = float((got.float() * exact).sum() / (exact * exact).sum())
            print(
                f"  W4A4 MLP (layer {layer}, GEMMs emulated, unbiased={unbiased}): equal to the "
                f"reference chain in {100 * float((got == want).float().mean()):.3f} % of the "
                f"outputs (rel L2 {rel:.2e}); against bf16: rel L2 {err:.4f}, in-phase gain "
                f"{gain:.4f}, norm {float(got.float().norm() / exact.norm()):.4f}"
            )
    with tempfile.TemporaryDirectory() as tmp:
        capture = selftest.make_mlp_capture(
            Path(tmp) / "mlp.pt",
            1024,
            3072,
            [((64, 11), 540), ((3, 5), 0)],
            tier="near-lossless-fp4a",
            precision="fp4_w4a4",
        )
        result = run_evaluation(capture, EXAMPLE)
        cases = result.get("cases") or [{}]
        print(
            f"  evaluator (near-lossless-fp4a, GatedMLP 1024 -> 3072): correct "
            f"{result.get('correct')}, status {result.get('status')}, max rel L2 "
            f"{cases[0].get('max_rel_l2')}, scale rule checked "
            f"{(result.get('scale_rule') or {}).get('checked')} ok "
            f"{(result.get('scale_rule') or {}).get('ok')}, speedup {result.get('speedup')} "
            "(emulated GEMMs: not a speed claim)"
        )
        if not result.get("correct"):
            print("   ", str(result.get("error"))[-600:])


if __name__ == "__main__":
    main()
