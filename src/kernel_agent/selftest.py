"""Backend smoke test: run every bundled example kernel through the evaluator (the FP8
weight-only, W8A8 and MXFP8 examples in the near-lossless tier, and their rejection by the
exact tier, MXFP8 with the OCP floor scale rule by the scale-rule guard; the FP4 one in the
near-lossless-fp4 tier, and its rejection by the FP8 tier; the PDL GEMV chain on sm_90+; the
Triton toolkit examples, cached launches and short-sequence attention:
:func:`smoke_triton_tools`; on sm_120 the CuTe DSL block-scaled W8A8 GEMM and the fused FP8
decoder block)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn

from kernel_agent.agent.prompts import EXAMPLES_DIR


class RMSNorm(nn.Module):
    """Same math as transformers' LlamaRMSNorm (kept here to avoid the dependency)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class GatedMlpBlock(nn.Module):
    """A pre-norm gated MLP block, ``x + down(silu(gate(norm(x))) * up(norm(x)))``: the
    reference of ``examples/triton_cheap_launch.py``."""

    def __init__(self, hidden: int, inter: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden, eps)
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + self.down_proj(nn.functional.silu(self.gate_proj(h)) * self.up_proj(h))


class AttentionCore(nn.Module):
    """SDPA over ``[B, H, S, D]`` heads (GQA when K / V have fewer heads), output
    ``[B, S, Hq * D]``: the reference of ``examples/triton_short_attention.py``."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        out = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal, enable_gqa=q.shape[1] != k.shape[1]
        )
        return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


#: The cheap-launch example's blocks: (hidden, intermediate, [(rows shape, calls per run)]):
#: a decode-sized block (host bound when timed eagerly) and one with sizes that are not
#: powers of two.
MLP_BLOCKS: list[tuple[int, int, list[tuple[tuple[int, ...], int]]]] = [
    (1024, 4096, [((2, 11), 40), ((1, 1), 0)]),
    (896, 4864, [((3, 5), 10), ((64,), 0)]),
]


def make_mlp_block_capture(
    path: Path, hidden: int, inter: int, calls: list[tuple[tuple[int, ...], int]]
) -> Path:
    from kernel_agent.profiling.capture import capture_calls

    torch.manual_seed(0)
    module = GatedMlpBlock(hidden, inter).cuda().to(torch.bfloat16)
    with torch.no_grad():
        module.norm.weight.copy_(torch.randn(hidden) * 0.1 + 1)
        for linear in (module.gate_proj, module.up_proj, module.down_proj):
            linear.weight.normal_(0.0, linear.in_features**-0.5)
    cases: list[tuple[Any, ...]] = [
        ((torch.randn(*shape, hidden, device="cuda", dtype=torch.bfloat16),), {}, count)
        for shape, count in calls
    ]
    capture_calls(module.eval(), cases, path)
    return path


def make_attention_capture(path: Path) -> Path:
    """Attention-core calls of several shapes: the timed one (32 sequences of 11 tokens,
    16 query / 2 KV heads, head dim 128) and correctness-only ones with other head counts,
    GQA ratios, head dims (incl. 80, 96), lengths up to 32, causal, boolean and additive
    masks, fp16, and one longer than the kernel takes (SDPA's path)."""
    from kernel_agent.profiling.capture import capture_calls

    torch.manual_seed(0)

    def qkv(
        b: int, hq: int, hk: int, sq: int, sk: int, d: int, dtype: torch.dtype = torch.bfloat16
    ) -> tuple[torch.Tensor, ...]:
        return (
            torch.randn(b, hq, sq, d, device="cuda", dtype=dtype),
            torch.randn(b, hk, sk, d, device="cuda", dtype=dtype),
            torch.randn(b, hk, sk, d, device="cuda", dtype=dtype),
        )

    padding = torch.ones(2, 1, 1, 7, device="cuda", dtype=torch.bool)
    padding[1, ..., 5:] = False  # the second sequence has 5 tokens
    bias = torch.randn(1, 24, device="cuda", dtype=torch.float16)
    cases: list[tuple[Any, ...]] = [
        (qkv(32, 16, 2, 11, 11, 128), {}, 540),
        (qkv(4, 8, 8, 32, 32, 64), {"is_causal": True}, 0),
        (qkv(2, 12, 4, 7, 7, 80), {"attn_mask": padding}, 0),
        (qkv(3, 6, 1, 1, 24, 96, torch.float16), {"attn_mask": bias}, 0),
        (qkv(2, 4, 2, 40, 40, 64), {}, 0),  # longer than MAX_SEQ: SDPA
    ]
    capture_calls(AttentionCore().eval(), cases, path)
    return path


def make_rmsnorm_capture(path: Path, hidden: int = 2048) -> Path:
    from kernel_agent.profiling.capture import capture_calls

    torch.manual_seed(0)
    module = RMSNorm(hidden, eps=1e-5).cuda().to(torch.bfloat16)
    with torch.no_grad():
        module.weight.copy_(torch.randn(hidden) * 0.1 + 1)
    prefill = torch.randn(1, 256, hidden, device="cuda", dtype=torch.bfloat16)
    decode = torch.randn(1, 1, hidden, device="cuda", dtype=torch.bfloat16)
    capture_calls(module, [((prefill,), {}, 1), ((decode,), {}, 31)], path, instances=2)
    return path


#: The FP8 weight-only examples (``precision: fp8_weights``, knowledge/low_precision.md) and
#: the ``nn.Linear`` they are checked on: in / out features and the captured calls
#: (input shape without the feature dimension, calls per run), on VoxCPM2 shapes.
FP8_EXAMPLES: dict[str, tuple[int, int, list[tuple[tuple[int, ...], int]]]] = {
    # base LM gate + up projections merged: decode GEMVs (and a few 4-row calls). 50 MB of
    # bf16 weights exceed the 48 MB L2 of an RTX 5070 Ti, as the model's layers do together;
    # a reference that just fits in L2 times erratically next to the candidate's weights
    "cuda_fp8_gemv.py": (2048, 12288, [((1,), 28), ((4,), 1)]),
    # LocDiT MLP up projection: [2, 11, 1024] per flow-matching step, a 32-row call and
    # (correctness only: 0 calls per run) 64 rows, two token groups
    "cuda_fp8_skinny_gemm.py": (1024, 4096, [((2, 11), 40), ((32,), 1), ((4, 16), 0)]),
}
#: The FP8 W8A8 examples (``precision: fp8_w8a8``), as :data:`FP8_EXAMPLES`: the VoxCPM2
#: LocDiT's merged gate|up projection under CFG at batch 32 (2 x 32 x 11 = 704 rows, compute
#: bound) and, correctness only, at batch 16 (352 rows; timed eagerly, the example's host
#: time, two Triton launches, would hide its gain there: 0.95x, in a CUDA graph 1.9x).
W8A8_EXAMPLES: dict[str, tuple[int, int, list[tuple[tuple[int, ...], int]]]] = {
    "triton_fp8_w8a8_gemm.py": (1024, 8192, [((64, 11), 540), ((32, 11), 0)]),
}
#: The MXFP8 W8A8 examples (``precision: fp8_mx``), as :data:`W8A8_EXAMPLES` (a wide-N GEMM,
#: where MXFP8 pays). Each also fails the scale-rule guard with the OCP floor rule (its
#: ``RULE = "floor"``). Not run on a GPU when written (#144).
MX_EXAMPLES: dict[str, tuple[int, int, list[tuple[tuple[int, ...], int]]]] = {
    "triton_mxfp8_gemm.py": (1024, 8192, [((64, 11), 540), ((32, 11), 0)]),
}
_EXAMPLES = {"fp8_w8a8": W8A8_EXAMPLES, "fp8_mx": MX_EXAMPLES}


#: The CuTe DSL W8A8 example on sm_120's block-scaled MMA, as :data:`W8A8_EXAMPLES`: a
#: compute-bound merged gate|up GEMM (704 rows) and, correctness only, 300 rows (not a
#: multiple of the 128-row tile: TMA's zero-filled tail).
CUTE_W8A8_EXAMPLES: dict[str, tuple[int, int, list[tuple[tuple[int, ...], int]]]] = {
    "cute_fp8_blockscaled_gemm.py": (1024, 8192, [((4, 176), 100), ((300,), 0)]),
}
#: The fused small-M CuTe DSL decoder block (``precision: fp8_weights``) on a gated MLP:
#: hidden and intermediate size, the captured calls (decode rows; 3 rows: a row bucket with
#: a tail; 40 rows, correctness only: the fallback past 16 rows).
CUTE_BLOCK_EXAMPLES: dict[str, tuple[int, int, list[tuple[tuple[int, ...], int]]]] = {
    "cute_fp8_decoder_block.py": (1024, 4096, [((16, 1), 100), ((3,), 0), ((40,), 0)]),
}


#: The FP4 weight-only examples (``precision: fp4_weights``) on the GEMV's shapes above; a
#: 22-row call (0 calls per run: correctness only) takes the dequantised fallback.
FP4_EXAMPLES: dict[str, tuple[int, int, list[tuple[tuple[int, ...], int]]]] = {
    "cuda_fp4_gemv.py": (2048, 12288, [((1,), 28), ((4,), 1), ((2, 11), 0)]),
}


class GemvChain(nn.Module):
    """``layers`` square bias-free ``nn.Linear`` applied in sequence: a chain of dependent
    decode GEMVs (the reference of the PDL example)."""

    def __init__(self, hidden: int, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


#: The PDL examples and their :class:`GemvChain`: hidden size, layers and the captured calls
#: (rows, calls per run). 28 layers of [1024, 1024] bf16 (59 MB, more than the 48 MB L2 of
#: an RTX 5070 Ti: streamed from DRAM), as in docs/PARALLEL.md §4.6; a 4-row call
#: (correctness only) takes the example's fallback.
PDL_EXAMPLES: dict[str, tuple[int, int, list[tuple[tuple[int, ...], int]]]] = {
    # 8 layers: an exact-tier chain of bf16 GEMVs drifts from cuBLAS's rounding with depth
    # (28 layers: 14 of 40 draws had > 0.1 % of elements outside bf16's tolerance; 8: none)
    "cuda_pdl_gemv_chain.py": (1024, 8, [((1,), 64), ((4,), 0)]),
}


def make_chain_capture(
    path: Path, hidden: int, layers: int, calls: list[tuple[tuple[int, ...], int]]
) -> Path:
    """A bf16 :class:`GemvChain` (Gaussian weights, std ``hidden ** -0.5``: activations keep
    their scale through the layers) and its calls."""
    from kernel_agent.profiling.capture import capture_calls

    torch.manual_seed(0)
    module = GemvChain(hidden, layers).cuda().to(torch.bfloat16)
    with torch.no_grad():
        for weight in module.parameters():  # the layers' weights (no biases)
            weight.normal_(0.0, hidden**-0.5)
    cases: list[tuple[Any, ...]] = [
        ((torch.randn(*shape, hidden, device="cuda", dtype=torch.bfloat16),), {}, count)
        for shape, count in calls
    ]
    capture_calls(module.eval(), cases, path)
    return path


def pdl_supported(tc: object) -> bool:
    """Whether the PDL examples can run: the ``cuda`` backend on sm_90 or newer
    (``griddepcontrol``)."""
    gpu = getattr(tc, "gpu", None)
    backends = getattr(tc, "backends", {}) or {}
    return bool(backends.get("cuda")) and gpu is not None and tuple(gpu.capability) >= (9, 0)


def smoke_pdl(tmp: Path, verbose: bool = False) -> bool:
    """Every PDL example (:data:`PDL_EXAMPLES`, ``ka_launch.cuh``) passes the evaluator."""
    from kernel_agent.kernels.evaluate import run_evaluation

    ok = True
    for name, (hidden, layers, calls) in PDL_EXAMPLES.items():
        capture = make_chain_capture(tmp / f"{name}.pt", hidden, layers, calls)
        result = run_evaluation(capture, EXAMPLES_DIR / name)
        passed = bool(result.get("correct"))
        ok &= passed
        if verbose:
            detail = (
                f"speedup {result.get('speedup')}x over {layers} eager layers"
                if passed
                else f"{result.get('status')}: {str(result.get('error', ''))[-300:]}"
            )
            print(f"  {name.removesuffix('.py'):22s} {'OK ' if passed else 'FAIL'} {detail}")
    return ok


def make_linear_capture(
    path: Path,
    in_features: int,
    out_features: int,
    calls: list[tuple[tuple[int, ...], int]],
    *,
    tier: str | None = None,
    precision: str | None = None,
) -> Path:
    """A bf16 ``nn.Linear`` (Gaussian weights, std ``in_features ** -0.5``) and its calls,
    captured in ``tier`` (``near-lossless`` for a reduced-precision target)."""
    from kernel_agent.profiling.capture import capture_calls

    torch.manual_seed(0)
    module = nn.Linear(in_features, out_features, bias=False).cuda().to(torch.bfloat16)
    with torch.no_grad():
        module.weight.normal_(0.0, in_features**-0.5)
    cases: list[tuple[Any, ...]] = [
        ((torch.randn(*shape, in_features, device="cuda", dtype=torch.bfloat16),), {}, count)
        for shape, count in calls
    ]
    capture_calls(module, cases, path, tier=tier, precision=precision)
    return path


def fp8_supported(tc: object) -> bool:
    """Whether the FP8 examples can run: the ``cuda`` backend on sm_89 or newer (hardware
    e4m3 conversion)."""
    gpu = getattr(tc, "gpu", None)
    backends = getattr(tc, "backends", {}) or {}
    return bool(backends.get("cuda")) and gpu is not None and tuple(gpu.capability) >= (8, 9)


def w8a8_supported(tc: object) -> bool:
    """Whether the W8A8 examples can run: the ``triton`` backend on sm_89 or newer (e4m3
    tensor cores)."""
    gpu = getattr(tc, "gpu", None)
    backends = getattr(tc, "backends", {}) or {}
    return bool(backends.get("triton")) and gpu is not None and tuple(gpu.capability) >= (8, 9)


def mxfp8_supported(tc: object) -> bool:
    """Whether the MXFP8 examples can run: the ``triton`` backend on a GPU with block-scaled
    tensor cores (sm_100 or newer) and a torch with ``F.scaled_mm``."""
    gpu = getattr(tc, "gpu", None)
    return (
        w8a8_supported(tc)
        and gpu is not None
        and tuple(gpu.capability) >= (10, 0)
        and hasattr(torch.nn.functional, "scaled_mm")
    )


def floor_rule_variant(tmp: Path, name: str) -> Path:
    """A copy of the MXFP8 example ``name`` with the OCP floor scale rule (``RULE =
    "floor"``): the evaluator's scale-rule guard must reject it."""
    source = (EXAMPLES_DIR / name).read_text()
    if 'RULE = "ceil"' not in source:
        raise ValueError(f'{name} has no RULE = "ceil" line')
    path = tmp / name.replace(".py", "_floor.py")
    path.write_text(source.replace('RULE = "ceil"', 'RULE = "floor"', 1))
    return path


def cute_fp8_supported(tc: object) -> bool:
    """Whether the CuTe DSL FP8 examples can run: the ``cute`` backend on sm_120 / sm_121
    (the block-scaled MMA the GEMM example uses exists only there)."""
    gpu = getattr(tc, "gpu", None)
    backends = getattr(tc, "backends", {}) or {}
    major = tuple(gpu.capability)[:1] if gpu is not None else ()
    return bool(backends.get("cute")) and major == (12,)


class GatedMLP(nn.Module):
    """``down(silu(gate(x)) * up(x))``, the common gated-MLP layout (bias-free)."""

    def __init__(self, hidden: int, inter: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def make_mlp_capture(
    path: Path,
    hidden: int,
    inter: int,
    calls: list[tuple[tuple[int, ...], int]],
    *,
    tier: str | None = None,
    precision: str | None = None,
) -> Path:
    """A bf16 :class:`GatedMLP` (Gaussian weights, std ``fan_in ** -0.5``) and its calls."""
    from kernel_agent.profiling.capture import capture_calls

    torch.manual_seed(0)
    module = GatedMLP(hidden, inter).cuda().to(torch.bfloat16)
    with torch.no_grad():
        for lin in (module.gate_proj, module.up_proj, module.down_proj):
            lin.weight.normal_(0.0, lin.in_features**-0.5)
    cases: list[tuple[Any, ...]] = [
        ((torch.randn(*shape, hidden, device="cuda", dtype=torch.bfloat16),), {}, count)
        for shape, count in calls
    ]
    capture_calls(module, cases, path, tier=tier, precision=precision)
    return path


def smoke_fp8(
    tmp: Path,
    verbose: bool = False,
    *,
    precision: str = "fp8_weights",
    examples: dict[str, tuple[int, int, list[tuple[tuple[int, ...], int]]]] | None = None,
    capture: Any = make_linear_capture,
) -> bool:
    """Every FP8 example of ``precision`` (:data:`FP8_EXAMPLES`, :data:`W8A8_EXAMPLES`,
    :data:`MX_EXAMPLES`, or ``examples`` captured with ``capture``) passes the evaluator in
    the near-lossless tier, and the exact tier (a quick check) rejects it; an MXFP8 example
    with the OCP floor scale rule fails the scale-rule guard."""
    from kernel_agent.kernels.evaluate import run_evaluation

    ok = True
    if examples is None:
        examples = _EXAMPLES.get(precision, FP8_EXAMPLES)
    for name, (k, n, calls) in examples.items():
        near = capture(
            tmp / f"{name}.near.pt", k, n, calls, tier="near-lossless", precision=precision
        )
        exact = capture(tmp / f"{name}.exact.pt", k, n, calls)
        result = run_evaluation(near, EXAMPLES_DIR / name)
        rejected = run_evaluation(exact, EXAMPLES_DIR / name, quick=True)
        passed = bool(result.get("correct")) and rejected.get("status") == "incorrect"
        floor: dict[str, Any] = {"stage": "scale_rule"}
        if precision == "fp8_mx":  # the guard names a saturating scale rule
            floor = run_evaluation(near, floor_rule_variant(tmp, name), quick=True)
            passed &= floor.get("stage") == "scale_rule"
        ok &= passed
        if verbose:
            if passed:
                cases = result.get("cases") or [{}]
                detail = (
                    f"speedup {result.get('speedup')}x, rel L2 {cases[0].get('max_rel_l2')} "
                    f"(near-lossless), exact tier rejects it"
                    + (
                        ", so does the scale-rule guard its floor rule"
                        if precision == "fp8_mx"
                        else ""
                    )
                )
            elif not result.get("correct"):
                detail = f"{result.get('status')}: {str(result.get('error', ''))[-300:]}"
            elif rejected.get("status") != "incorrect":
                detail = f"the exact tier did not reject it: {rejected.get('status')}"
            else:
                detail = f"the floor scale rule was not rejected: {floor.get('status')}"
            print(f"  {name.removesuffix('.py'):22s} {'OK ' if passed else 'FAIL'} {detail}")
    return ok


def smoke_fp4(tmp: Path, verbose: bool = False) -> bool:
    """Every FP4 example passes the evaluator in the near-lossless-fp4 tier, and the FP8
    near-lossless tier (a quick check) rejects it: FP4 needs its own tier."""
    from kernel_agent.kernels.evaluate import run_evaluation

    ok = True
    for name, (k, n, calls) in FP4_EXAMPLES.items():
        fp4 = make_linear_capture(
            tmp / f"{name}.fp4.pt", k, n, calls, tier="near-lossless-fp4", precision="fp4_weights"
        )
        fp8 = make_linear_capture(
            tmp / f"{name}.fp8.pt", k, n, calls, tier="near-lossless", precision="fp8_weights"
        )
        result = run_evaluation(fp4, EXAMPLES_DIR / name)
        rejected = run_evaluation(fp8, EXAMPLES_DIR / name, quick=True)
        passed = bool(result.get("correct")) and rejected.get("status") == "incorrect"
        ok &= passed
        if verbose:
            if passed:
                cases = result.get("cases") or [{}]
                detail = (
                    f"speedup {result.get('speedup')}x, rel L2 {cases[0].get('max_rel_l2')} "
                    f"(near-lossless-fp4), the FP8 tier rejects it"
                )
            elif not result.get("correct"):
                detail = f"{result.get('status')}: {str(result.get('error', ''))[-300:]}"
            else:
                detail = f"the FP8 tier did not reject it: {rejected.get('status')}"
            print(f"  {name.removesuffix('.py'):22s} {'OK ' if passed else 'FAIL'} {detail}")
    return ok


def cheap_launch_check() -> dict[str, Any]:
    """In this process: ``examples/triton_cheap_launch.py`` built with and without
    ``fast_launch`` gives bit-identical outputs, and its later calls launch through the
    cached ``CompiledKernel`` (``hits``)."""
    from kernel_agent.kernels.evaluate import load_candidate_module

    example = load_candidate_module(EXAMPLES_DIR / "triton_cheap_launch.py")
    torch.manual_seed(0)
    module = GatedMlpBlock(1024, 4096).cuda().to(torch.bfloat16).eval()
    x = torch.randn(2, 11, 1024, device="cuda", dtype=torch.bfloat16)
    fast, slow = example.build(module), example.build(module, fast_launch=False)
    with torch.inference_mode():
        outs = [fast(x), fast(x), slow(x)]
    hits = example._rmsnorm.hits + example._silu_mul.hits
    same = all(torch.equal(outs[0], o) for o in outs[1:])
    return {"passed": same and hits >= 2, "bit_identical": same, "hits": hits}


def smoke_triton_tools(tmp: Path, verbose: bool = False) -> bool:
    """The Triton toolkit examples (#148): ``triton_cheap_launch.py`` passes the evaluator on
    every block of :data:`MLP_BLOCKS` and its cached launches match the JIT's
    (:func:`cheap_launch_check`); ``triton_short_attention.py`` passes on
    :func:`make_attention_capture` and traces without a graph break (``compile_check``)."""
    from kernel_agent.kernels.evaluate import run_evaluation

    rows: list[tuple[str, bool, str]] = []
    for hidden, inter, calls in MLP_BLOCKS:
        capture = make_mlp_block_capture(tmp / f"mlp{hidden}.pt", hidden, inter, calls)
        result = run_evaluation(capture, EXAMPLES_DIR / "triton_cheap_launch.py")
        passed = bool(result.get("correct"))
        detail = (
            f"speedup {result.get('speedup')}x"
            if passed
            else f"{result.get('status')}: {str(result.get('error', ''))[-300:]}"
        )
        rows.append((f"cheap_launch {hidden}", passed, detail))
    try:
        check = cheap_launch_check()
    except Exception as exc:
        check = {"passed": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
    rows.append(("cheap_launch cache", bool(check["passed"]), str(check)))
    capture = make_attention_capture(tmp / "attention.pt")
    example = EXAMPLES_DIR / "triton_short_attention.py"
    result = run_evaluation(capture, example, compile_check=True)
    compiled = result.get("compile_check") or {}
    passed = bool(result.get("correct") and compiled.get("passed") and compiled.get("fullgraph_ok"))
    detail = (
        f"speedup {result.get('speedup')}x, compiled {compiled.get('passed')}, "
        f"fullgraph {compiled.get('fullgraph_ok')}"
        if result.get("correct")
        else f"{result.get('status')}: {str(result.get('error', ''))[-300:]}"
    )
    rows.append(("short_attention", passed, detail))
    ok = True
    for name, passed, detail in rows:
        ok &= passed
        if verbose:
            print(f"  {name:22s} {'OK ' if passed else 'FAIL'} {detail}")
    return ok


# ------------------------------------------------------------------ the FP8 toolkit (#145)

#: The direct cuBLASLt FP8 example (``cuda`` backend, ``fp8_w8a8``, tensor-wise mode), as
#: :data:`W8A8_EXAMPLES`.
CUBLASLT_EXAMPLES: dict[str, tuple[int, int, list[tuple[tuple[int, ...], int]]]] = {
    "cuda_cublaslt_fp8.py": (1024, 8192, [((64, 11), 540), ((32, 11), 0)]),
}
#: The producer example (``triton``, ``fp8_w8a8``): a gated SiLU MLP hidden -> inter ->
#: hidden and its calls (input shape without the feature dimension, calls per run).
PRODUCER_EXAMPLES: dict[str, tuple[int, int, list[tuple[tuple[int, ...], int]]]] = {
    "triton_fp8_producers.py": (1024, 4096, [((64, 11), 540), ((3, 5), 0)]),
}
#: The FP8 KV-cache example (``triton``, ``fp8_kv``): query heads, KV heads, head dim and the
#: decode calls (batch, cached tokens, calls per run): a long cache, and a short one.
FP8_KV_EXAMPLES: dict[str, tuple[int, int, int, list[tuple[int, int, int]]]] = {
    "triton_fp8_kv_decode.py": (16, 2, 128, [(4, 4096, 28), (2, 77, 0)]),
}


class DecodeAttention(nn.Module):
    """One decode step of attention: ``q [B, Hq, 1, D]`` over ``k / v [B, Hkv, L, D]``."""

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)


def make_decode_attention_capture(
    path: Path,
    heads: int,
    kv_heads: int,
    head_dim: int,
    calls: list[tuple[int, int, int]],
    *,
    tier: str | None = None,
    precision: str | None = None,
) -> Path:
    from kernel_agent.profiling.capture import capture_calls

    torch.manual_seed(0)
    cases: list[tuple[Any, ...]] = []
    for batch, tokens, count in calls:
        q = torch.randn(batch, heads, 1, head_dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(batch, kv_heads, tokens, head_dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(batch, kv_heads, tokens, head_dim, device="cuda", dtype=torch.bfloat16)
        cases.append(((q, k, v), {}, count))
    capture_calls(DecodeAttention(), cases, path, tier=tier, precision=precision)
    return path


def smoke_block_scale(capability: tuple[int, ...], verbose: bool = False) -> bool:
    """The Triton W8A8 example's GEMM lowers to the block-scaled MMA (PTX ``block_scale``,
    ``QMMA.SF``) on a GPU that has it (sm_100+): compiled for ``capability``, not run."""
    from kernel_agent.kernels.evaluate import load_candidate_module

    module = load_candidate_module(EXAMPLES_DIR / "triton_fp8_w8a8_gemm.py")
    if not module.block_scale_capable(capability):
        return True
    try:
        passed = bool(module.block_scale_mma(module.gemm_ptx(capability)))
        detail = "tl.dot_scaled -> block_scale MMA" if passed else "no block_scale MMA in PTX"
    except Exception as exc:  # a Triton that cannot compile it
        passed, detail = False, f"{type(exc).__name__}: {str(exc)[-200:]}"
    if verbose:
        print(f"  {'w8a8 block_scale':22s} {'OK ' if passed else 'FAIL'} {detail}")
    return passed


def smoke_fp8_toolkit(tmp: Path, verbose: bool = False, *, cuda: bool, triton: bool) -> bool:
    """The FP8 toolkit examples on the evaluator: the cuBLASLt helper (``cuda``) and the
    producer MLP (``triton``) pass ``fp8_w8a8``'s near-lossless tier and fail the exact one
    (:func:`smoke_fp8`); the FP8 KV decode attention (``triton``) passes ``fp8_kv``'s (e4m3
    K / V can stay inside the exact tier's tolerances: no rejection check)."""
    from kernel_agent.kernels.evaluate import run_evaluation

    ok = True
    if cuda:
        ok &= smoke_fp8(tmp, verbose, precision="fp8_w8a8", examples=CUBLASLT_EXAMPLES)
    if not triton:
        return ok
    ok &= smoke_fp8(
        tmp, verbose, precision="fp8_w8a8", examples=PRODUCER_EXAMPLES, capture=make_mlp_capture
    )
    for name, (heads, kv_heads, dim, decode) in FP8_KV_EXAMPLES.items():
        capture = make_decode_attention_capture(
            tmp / f"{name}.pt",
            heads,
            kv_heads,
            dim,
            decode,
            tier="near-lossless-kv",
            precision="fp8_kv",
        )
        result = run_evaluation(capture, EXAMPLES_DIR / name)
        passed = bool(result.get("correct"))
        ok &= passed
        if verbose:
            cases = result.get("cases") or [{}]
            detail = (
                f"speedup {result.get('speedup')}x, rel L2 {cases[0].get('max_rel_l2')} "
                "(near-lossless, fp8_kv)"
                if passed
                else f"{result.get('status')}: {str(result.get('error', ''))[-300:]}"
            )
            print(f"  {name.removesuffix('.py'):22s} {'OK ' if passed else 'FAIL'} {detail}")
    return ok


def smoke_backends(backends: list[str] | None = None, verbose: bool = False) -> bool:
    from kernel_agent import toolchain
    from kernel_agent.kernels.evaluate import run_evaluation

    tc = toolchain.setup()
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        capture = make_rmsnorm_capture(Path(tmp) / "rmsnorm.pt")
        for backend in backends or [b for b, avail in tc.backends.items() if avail]:
            example = EXAMPLES_DIR / f"{backend}_rmsnorm.py"
            if not example.exists():
                continue
            result = run_evaluation(capture, example)
            passed = bool(result.get("correct"))
            ok &= passed
            if verbose:
                detail = (
                    f"speedup {result.get('speedup')}x"
                    if passed
                    else f"{result.get('status')}: {str(result.get('error', ''))[-300:]}"
                )
                print(f"  {backend:9s} {'OK ' if passed else 'FAIL'} {detail}")
        native = EXAMPLES_DIR / "native_project"  # a multi-file project (native/project.py)
        if native.is_dir() and tc.backends.get("cuda") and (backends is None or "cuda" in backends):
            result = run_evaluation(capture, native)
            passed = bool(result.get("correct"))
            ok &= passed
            if verbose:
                detail = (
                    f"speedup {result.get('speedup')}x"
                    if passed
                    else f"{result.get('status')}: {str(result.get('error', ''))[-300:]}"
                )
                print(f"  {'project':9s} {'OK ' if passed else 'FAIL'} {detail}")
        if fp8_supported(tc) and (backends is None or "cuda" in backends):
            ok &= smoke_fp8(Path(tmp), verbose)
            ok &= smoke_fp4(Path(tmp), verbose)
        if pdl_supported(tc) and (backends is None or "cuda" in backends):
            ok &= smoke_pdl(Path(tmp), verbose)
        if w8a8_supported(tc) and (backends is None or "triton" in backends):
            ok &= smoke_fp8(Path(tmp), verbose, precision="fp8_w8a8")
            ok &= smoke_block_scale(tuple(tc.gpu.capability) if tc.gpu else (0, 0), verbose)
        if mxfp8_supported(tc) and (backends is None or "triton" in backends):
            ok &= smoke_fp8(Path(tmp), verbose, precision="fp8_mx")
        if tc.backends.get("triton") and (backends is None or "triton" in backends):
            ok &= smoke_triton_tools(Path(tmp), verbose)
        if cute_fp8_supported(tc) and (backends is None or "cute" in backends):
            ok &= smoke_fp8(Path(tmp), verbose, precision="fp8_w8a8", examples=CUTE_W8A8_EXAMPLES)
            ok &= smoke_fp8(
                Path(tmp), verbose, examples=CUTE_BLOCK_EXAMPLES, capture=make_mlp_capture
            )
        cuda = fp8_supported(tc) and (backends is None or "cuda" in backends)
        triton = w8a8_supported(tc) and (backends is None or "triton" in backends)
        if cuda or triton:
            ok &= smoke_fp8_toolkit(Path(tmp), verbose, cuda=cuda, triton=triton)
    return ok
