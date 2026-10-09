"""Backend smoke test: run every bundled example kernel through the evaluator (the FP8
weight-only, W8A8 and MXFP8 examples and the INT8 weight-only and W8A8 ones in the
near-lossless tier, and their rejection by the exact tier, MXFP8 with the OCP floor scale rule
by the scale-rule guard; the FP4 one in the
near-lossless-fp4 tier and the W4A4 ones in near-lossless-fp4a, and their rejection by the FP8
tier; the PDL GEMV chain on sm_90+; the
Triton toolkit examples, cached launches and short-sequence attention:
:func:`smoke_triton_tools`; on sm_120 the CuTe DSL block-scaled W8A8 and W4A4 GEMMs and the
fused FP8 decoder block; on sm_90 / sm_100 the CuTe DSL ``wgmma`` / ``tcgen05`` W8A8 GEMM
templates).

Every example declares the GPUs it runs on (its ``ARCHS``, ``kernel_agent/gpu_arch.py``):
:func:`smoke_backends` runs those this GPU supports and lists the others with the reason
(:func:`example_skip`), so ``doctor --smoke`` passes on any GPU. A backend the toolchain
refuses on this GPU (``cute`` below sm_80: ``toolchain.ARCH_SUPPORT``) is listed the same
way, never run. A low-precision example that declares fp16 activations (its ``DTYPES``,
:func:`example_dtypes`) is also checked on an fp16 ``nn.Linear``."""

from __future__ import annotations

import ast
import inspect
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from kernel_agent import emulate, gpu_arch
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


#: The FP8 weight-only examples (``precision: fp8_weights``, skill fp8-weights) and
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
#: The INT8 W8A8 examples (``precision: int8_w8a8``, #178), as :data:`W8A8_EXAMPLES`: the
#: Triton IMMA GEMM on the merged gate|up projection (704 rows, compute bound; 352 rows,
#: correctness only) and the CUDA IMMA skinny GEMM at decode sizes (16 rows; a 3-row call,
#: and 40 rows: groups of 32 tokens, correctness only).
INT8_W8A8_EXAMPLES: dict[str, tuple[int, int, list[tuple[tuple[int, ...], int]]]] = {
    "triton_int8_w8a8_gemm.py": (1024, 8192, [((64, 11), 540), ((32, 11), 0)]),
    "cuda_int8_skinny_gemm.py": (2048, 12288, [((16, 1), 28), ((3,), 0), ((40,), 0)]),
}
#: The INT8 weight-only examples (``precision: int8_weights``), as :data:`FP8_EXAMPLES`.
INT8_WEIGHT_EXAMPLES: dict[str, tuple[int, int, list[tuple[tuple[int, ...], int]]]] = {
    "cuda_int8_gemv.py": (2048, 12288, [((1,), 28), ((4,), 1), ((2, 11), 0)]),
}
_EXAMPLES = {
    "fp8_w8a8": W8A8_EXAMPLES,
    "fp8_mx": MX_EXAMPLES,
    "int8_w8a8": INT8_W8A8_EXAMPLES,
    "int8_weights": INT8_WEIGHT_EXAMPLES,
}


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
#: The CuTe DSL W8A8 GEMM templates of Hopper (``wgmma``) and datacenter Blackwell
#: (``tcgen05.mma``, #228), as :data:`CUTE_W8A8_EXAMPLES`; on sm_90 also 16 rows
#: (correctness only), the swap-AB kernel. Each runs only on its own family (its ``ARCHS``).
CUTE_ARCH_EXAMPLES: dict[str, tuple[int, int, list[tuple[tuple[int, ...], int]]]] = {
    "cute_sm90_gemm_ws.py": (1024, 8192, [((4, 176), 100), ((300,), 0), ((16,), 0)]),
    "cute_sm100_gemm_tcgen05.py": (1024, 8192, [((4, 176), 100), ((300,), 0)]),
}


#: The FP4 weight-only examples (``precision: fp4_weights``) on the GEMV's shapes above; a
#: 22-row call (0 calls per run: correctness only) takes the dequantised fallback.
FP4_EXAMPLES: dict[str, tuple[int, int, list[tuple[tuple[int, ...], int]]]] = {
    "cuda_fp4_gemv.py": (2048, 12288, [((1,), 28), ((4,), 1), ((2, 11), 0)]),
}
#: The W4A4 examples (``precision: fp4_w4a4``, #233; opt-in) in their near-lossless-fp4a
#: tier, as :data:`W8A8_EXAMPLES`: the merged gate|up GEMM at 704 rows (compute bound) and,
#: correctness only, 300 rows (zero-padded scale rows past the 128-row blocks).
W4A4_EXAMPLES: dict[str, tuple[int, int, list[tuple[tuple[int, ...], int]]]] = {
    "triton_nvfp4_w4a4_gemm.py": (1024, 8192, [((64, 11), 540), ((300,), 0)]),
}
#: The CuTe DSL W4A4 GEMM on sm_120's block-scaled FP4 MMA (``MmaMXF4NVF4Op``), as
#: :data:`CUTE_W8A8_EXAMPLES` (300 rows: TMA's zero-filled tail; 17: the per-token tail).
CUTE_W4A4_EXAMPLES: dict[str, tuple[int, int, list[tuple[tuple[int, ...], int]]]] = {
    "cute_nvfp4_w4a4_gemm.py": (1024, 8192, [((4, 176), 100), ((300,), 0), ((17,), 0)]),
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
    # (28 layers: 14 of 40 draws had > 0.1 % of elements outside bf16's tolerance; 8: none).
    # Since #250 the redrawn-input checks judge such a draw with the reference's own rounding
    # spread (28 layers: 342 of 1,000 draws past the plain tolerance, none rejected), but the
    # captured-input check keeps the plain tolerance, and a deeper capture whose own input is
    # past it on some GPU's cuBLAS would fail there on every run.
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


class NormGemvChain(nn.Module):
    """``x = x + W_l rmsnorm_l(x)`` for ``layers`` layers (bias-free square projections): a
    stack of decode layers, the reference of the megakernel example (issue #225)."""

    def __init__(self, hidden: int, layers: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norms = nn.ModuleList(RMSNorm(hidden, eps) for _ in range(layers))
        self.layers = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for norm, layer in zip(self.norms, self.layers, strict=True):
            x = x + layer(norm(x))
        return x


#: The megakernel example (kernel_agent.native.megakernel) and its capture: hidden, layers,
#: (leading shape, calls per run). 4 layers: the exact tier's bf16 tolerance against eager
#: (cuBLAS) rounding is reached with depth (issue #248, RTX 5070 Ti, 3000 random inputs: at 8
#: layers 1.3 % of the draws had > 0.1 % of elements outside it for the megakernel and 1.0 %
#: for the graph + PDL and grid-barrier baselines, which use no counters; at 4 layers none).
#: The redrawn-input checks judge such draws with the reference's own rounding spread since
#: #250 (8 layers: 0 of 100 evaluations refused, 4 before; ``tests/test_megakernel_gpu.py``);
#: the captured-input check does not, so the capture stays where its own input is far from
#: the plain tolerance on any GPU.
MEGAKERNEL_EXAMPLES: dict[str, tuple[int, int, list[tuple[tuple[int, ...], int]]]] = {
    "native_megakernel": (1024, 4, [((1,), 64), ((4,), 0)]),
}


def make_norm_chain_capture(
    path: Path, hidden: int, layers: int, calls: list[tuple[tuple[int, ...], int]]
) -> Path:
    """A bf16 :class:`NormGemvChain` (projections ~ N(0, hidden ** -0.5), norm weights
    ~ N(1, 0.1)) and its calls."""
    from kernel_agent.profiling.capture import capture_calls

    torch.manual_seed(0)
    module = NormGemvChain(hidden, layers).cuda().to(torch.bfloat16)
    with torch.no_grad():
        for name, weight in module.named_parameters():
            if name.startswith("norms."):
                weight.normal_(1.0, 0.1)
            else:  # the projections
                weight.normal_(0.0, hidden**-0.5)
    cases: list[tuple[Any, ...]] = [
        ((torch.randn(*shape, hidden, device="cuda", dtype=torch.bfloat16),), {}, count)
        for shape, count in calls
    ]
    capture_calls(module.eval(), cases, path)
    return path


def smoke_megakernel(tmp: Path, verbose: bool = False) -> bool:
    """The megakernel example passes the evaluator (its default mode)."""
    from kernel_agent.kernels.evaluate import run_evaluation

    ok = True
    for name, (hidden, layers, calls) in MEGAKERNEL_EXAMPLES.items():
        capture = make_norm_chain_capture(tmp / f"{name}.pt", hidden, layers, calls)
        result = run_evaluation(capture, EXAMPLES_DIR / name)
        passed = bool(result.get("correct"))
        ok &= passed
        if verbose:
            detail = (
                f"speedup {result.get('speedup')}x over {layers} eager layers"
                if passed
                else f"{result.get('status')}: {str(result.get('error', ''))[-300:]}"
            )
            print(f"  {name:22s} {'OK ' if passed else 'FAIL'} {detail}")
    return ok


def example_skip(name: str, tc: object, backend: str | None = None) -> str | None:
    """Why the bundled example ``name`` does not run with toolchain ``tc``: ``backend`` is
    not available (with the toolchain's reason: ``cute`` on sm_75), or this GPU is not one of
    its ``ARCHS`` (:func:`gpu_arch.example_skip`); None: it runs."""
    gpu = getattr(tc, "gpu", None)
    cap = tuple(gpu.capability) if gpu is not None else None
    if backend is not None and not (getattr(tc, "backends", {}) or {}).get(backend):
        arch = gpu_arch.example_skip(EXAMPLES_DIR / name, cap)
        if arch and emulate.compile_only(backend, gpu):  # #252: its ARCHS, not "compile-only"
            return arch
        why = (getattr(tc, "unavailable", {}) or {}).get(backend)
        return f"the {backend} backend is not available" + (f" ({why})" if why else "")
    return gpu_arch.example_skip(EXAMPLES_DIR / name, cap)


def examples_run(names: Iterable[str], tc: object, backend: str) -> bool:
    """Whether every example of ``names`` runs with ``tc`` (:func:`example_skip`)."""
    return all(example_skip(name, tc, backend) is None for name in names)


def example_dtypes(name: str) -> tuple[str, ...]:
    """The activation dtypes the bundled example ``name`` declares (``DTYPES =
    (torch.bfloat16, torch.float16)``, read with ``ast``: not imported), as ``torch``
    attribute names; ``("bfloat16",)`` without a declaration."""
    try:
        tree = ast.parse((EXAMPLES_DIR / name).read_text())
    except (OSError, SyntaxError, ValueError):
        return ("bfloat16",)
    for node in tree.body:
        if isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == [
            "DTYPES"
        ]:
            elts = node.value.elts if isinstance(node.value, ast.Tuple) else []
            found = tuple(e.attr for e in elts if isinstance(e, ast.Attribute))
            return found or ("bfloat16",)
    return ("bfloat16",)


def pdl_supported(tc: object) -> bool:
    """Whether the PDL examples can run: the ``cuda`` backend on sm_90 or newer
    (``griddepcontrol``; their ``ARCHS``)."""
    return examples_run(PDL_EXAMPLES, tc, "cuda")


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
    dtype: torch.dtype = torch.bfloat16,
) -> Path:
    """A bf16 (or ``dtype``) ``nn.Linear`` (Gaussian weights, std ``in_features ** -0.5``)
    and its calls, captured in ``tier`` (``near-lossless`` for a reduced-precision target)."""
    from kernel_agent.profiling.capture import capture_calls

    torch.manual_seed(0)
    module = nn.Linear(in_features, out_features, bias=False).cuda().to(dtype)
    with torch.no_grad():
        module.weight.normal_(0.0, in_features**-0.5)
    cases: list[tuple[Any, ...]] = [
        ((torch.randn(*shape, in_features, device="cuda", dtype=dtype),), {}, count)
        for shape, count in calls
    ]
    capture_calls(module, cases, path, tier=tier, precision=precision)
    return path


def fp8_supported(tc: object) -> bool:
    """Whether the FP8 weight-only examples can run: the ``cuda`` backend on a GPU of their
    ``ARCHS`` (sm_80+: e4m3 converted in registers, in software before sm_89)."""
    return examples_run(FP8_EXAMPLES, tc, "cuda")


def w8a8_supported(tc: object) -> bool:
    """Whether the W8A8 examples can run: the ``triton`` backend on sm_89 or newer (e4m3
    tensor cores; their ``ARCHS``)."""
    return examples_run(W8A8_EXAMPLES, tc, "triton")


def int8_supported(tc: object) -> bool:
    """Whether every INT8 example can run here: the ``triton`` and ``cuda`` backends on a
    GPU of their ``ARCHS`` (sm_80+: ``mma.sync`` m16n8k32 s8, Triton's int8 ``tl.dot``)."""
    triton = [n for n in INT8_W8A8_EXAMPLES if n.startswith("triton_")]
    cuda = [n for n in (*INT8_W8A8_EXAMPLES, *INT8_WEIGHT_EXAMPLES) if n.startswith("cuda_")]
    return examples_run(triton, tc, "triton") and examples_run(cuda, tc, "cuda")


def mxfp8_supported(tc: object) -> bool:
    """Whether the MXFP8 examples can run: the ``triton`` backend on a GPU with block-scaled
    tensor cores (sm_100 or newer: their ``ARCHS``) and a torch with ``F.scaled_mm``."""
    return examples_run(MX_EXAMPLES, tc, "triton") and hasattr(torch.nn.functional, "scaled_mm")


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
    """Whether the CuTe DSL FP8 GEMM example can run: the ``cute`` backend on sm_120 /
    sm_121 (the block-scaled ``mma.sync`` it uses exists only there; its ``ARCHS``)."""
    return examples_run(CUTE_W8A8_EXAMPLES, tc, "cute")


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
    """Every FP8 (or INT8) example of ``precision`` (:data:`FP8_EXAMPLES`,
    :data:`W8A8_EXAMPLES`, :data:`MX_EXAMPLES`, :data:`INT8_W8A8_EXAMPLES`,
    :data:`INT8_WEIGHT_EXAMPLES`, or ``examples`` captured with ``capture``) passes the evaluator in
    the near-lossless tier, and the exact tier (a quick check) rejects it; an MXFP8 example
    with the OCP floor scale rule fails the scale-rule guard. With a ``capture`` that takes a
    ``dtype`` (:func:`make_linear_capture`), once per activation dtype the example declares
    (:func:`example_dtypes`: an fp16 model too)."""
    from kernel_agent.kernels.evaluate import run_evaluation

    ok = True
    if examples is None:
        examples = _EXAMPLES.get(precision, FP8_EXAMPLES)
    typed = "dtype" in inspect.signature(capture).parameters
    for name, (k, n, calls), dtype in [
        (example, spec, dt)
        for example, spec in examples.items()
        for dt in (example_dtypes(example) if typed else ("bfloat16",))
    ]:
        kw: dict[str, Any] = {} if dtype == "bfloat16" else {"dtype": getattr(torch, dtype)}
        stem = name if dtype == "bfloat16" else f"{name}.{dtype}"
        near = capture(
            tmp / f"{stem}.near.pt", k, n, calls, tier="near-lossless", precision=precision, **kw
        )
        exact = capture(tmp / f"{stem}.exact.pt", k, n, calls, **kw)
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
            label = name.removesuffix(".py") + ("" if dtype == "bfloat16" else f" ({dtype})")
            print(f"  {label:22s} {'OK ' if passed else 'FAIL'} {detail}")
    return ok


def smoke_fp4(
    tmp: Path,
    verbose: bool = False,
    *,
    precision: str = "fp4_weights",
    examples: dict[str, tuple[int, int, list[tuple[tuple[int, ...], int]]]] | None = None,
) -> bool:
    """Every FP4 example of ``precision`` (:data:`FP4_EXAMPLES`; ``fp4_w4a4``:
    :data:`W4A4_EXAMPLES`, or ``examples``) passes the evaluator in its own tier
    (near-lossless-fp4, near-lossless-fp4a), and the 8-bit near-lossless tier (a quick check)
    rejects it: FP4 needs its own tier. Once per activation dtype the example declares
    (:func:`example_dtypes`)."""
    from kernel_agent.kernels.compare import tier_for
    from kernel_agent.kernels.evaluate import run_evaluation

    ok = True
    tier = tier_for("near-lossless", precision)
    if examples is None:
        examples = W4A4_EXAMPLES if precision == "fp4_w4a4" else FP4_EXAMPLES
    for name, (k, n, calls), dtype in [
        (example, spec, dt) for example, spec in examples.items() for dt in example_dtypes(example)
    ]:
        stem, dt = (name if dtype == "bfloat16" else f"{name}.{dtype}"), getattr(torch, dtype)
        fp4 = make_linear_capture(
            tmp / f"{stem}.fp4.pt", k, n, calls, tier=tier, precision=precision, dtype=dt
        )
        fp8 = make_linear_capture(
            tmp / f"{stem}.fp8.pt",
            k,
            n,
            calls,
            tier="near-lossless",
            precision="fp8_weights",
            dtype=dt,
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
                    f"({tier}), the FP8 tier rejects it"
                )
            elif not result.get("correct"):
                detail = f"{result.get('status')}: {str(result.get('error', ''))[-300:]}"
            else:
                detail = f"the FP8 tier did not reject it: {rejected.get('status')}"
            label = name.removesuffix(".py") + ("" if dtype == "bfloat16" else f" ({dtype})")
            print(f"  {label:22s} {'OK ' if passed else 'FAIL'} {detail}")
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


def smoke_triton_tools(tmp: Path, verbose: bool = False, *, attention: bool = True) -> bool:
    """The Triton toolkit examples (#148): ``triton_cheap_launch.py`` passes the evaluator on
    every block of :data:`MLP_BLOCKS` and its cached launches match the JIT's
    (:func:`cheap_launch_check`); ``triton_short_attention.py`` (``attention``: on a GPU of
    its ``ARCHS``) passes on :func:`make_attention_capture` and traces without a graph
    break (``compile_check``)."""
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
    if attention:
        capture = make_attention_capture(tmp / "attention.pt")
        example = EXAMPLES_DIR / "triton_short_attention.py"
        result = run_evaluation(capture, example, compile_check=True)
        compiled = result.get("compile_check") or {}
        passed = bool(
            result.get("correct") and compiled.get("passed") and compiled.get("fullgraph_ok")
        )
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
    ``QMMA.SF``) on the GPUs it uses it on (sm_12x): compiled for ``capability``, not run."""
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


#: The Helion examples besides the RMSNorm (#229) and the bf16 ``nn.Linear`` they are checked
#: on in the exact tier, as :data:`FP8_EXAMPLES`: in / out features and the captured calls
#: (704 rows; 3 rows, correctness only: a tile's masked tail).
HELION_EXAMPLES: dict[str, tuple[int, int, list[tuple[tuple[int, ...], int]]]] = {
    "helion_gemm_epilogue.py": (1024, 4096, [((64, 11), 20), ((3,), 0)]),
}


def smoke_helion(tmp: Path, verbose: bool = False) -> bool:
    """Every Helion example of :data:`HELION_EXAMPLES` passes the evaluator (exact tier)."""
    from kernel_agent.kernels.evaluate import run_evaluation

    ok = True
    for name, (k, n, calls) in HELION_EXAMPLES.items():
        result = run_evaluation(
            make_linear_capture(tmp / f"{name}.pt", k, n, calls), EXAMPLES_DIR / name
        )
        passed = bool(result.get("correct"))
        ok &= passed
        if verbose:
            detail = (
                f"speedup {result.get('speedup')}x"
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
    skipped: dict[str, str] = {}
    # the available backends, and those installed but refused on this GPU (listed with why)
    off = getattr(tc, "unavailable", {}) or {}
    if verbose and (emulated := emulate.record(tc.gpu)) is not None:  # #252
        print(
            f"  emulating {emulated['arch']} on this {emulated['real']} (PTX JIT): its code "
            f"paths' correctness; the speedups are not representative; {emulated['native']}"
        )
    with tempfile.TemporaryDirectory() as tmp:
        capture = make_rmsnorm_capture(Path(tmp) / "rmsnorm.pt")
        for backend in backends or [b for b, avail in tc.backends.items() if avail or b in off]:
            example = EXAMPLES_DIR / f"{backend}_rmsnorm.py"
            if not example.exists():
                continue
            if (why := example_skip(example.name, tc, backend)) is not None:
                skipped[example.name] = why
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

        def runs(names: Iterable[str], backend: str) -> bool:
            """Whether the examples ``names`` run here (``backend`` asked for, every one's
            ``ARCHS`` holds this GPU); the reason of each that does not is listed."""
            if backends is not None and backend not in backends:
                return False
            why = {name: example_skip(name, tc, backend) for name in names}
            skipped.update({k: v for k, v in why.items() if v is not None})
            return all(v is None for v in why.values())

        if runs(FP8_EXAMPLES, "cuda"):
            ok &= smoke_fp8(Path(tmp), verbose)
        if runs(FP4_EXAMPLES, "cuda"):
            ok &= smoke_fp4(Path(tmp), verbose)
        if runs(PDL_EXAMPLES, "cuda"):
            ok &= smoke_pdl(Path(tmp), verbose)
        if runs(MEGAKERNEL_EXAMPLES, "cuda"):
            ok &= smoke_megakernel(Path(tmp), verbose)
        if runs(W8A8_EXAMPLES, "triton"):
            ok &= smoke_fp8(Path(tmp), verbose, precision="fp8_w8a8")
            ok &= smoke_block_scale(tuple(tc.gpu.capability) if tc.gpu else (0, 0), verbose)
        if runs(MX_EXAMPLES, "triton") and mxfp8_supported(tc):
            ok &= smoke_fp8(Path(tmp), verbose, precision="fp8_mx")
        if runs(W4A4_EXAMPLES, "triton") and hasattr(torch, "float4_e2m1fn_x2"):  # #233
            ok &= smoke_fp4(Path(tmp), verbose, precision="fp4_w4a4")
        for precision, found in (
            ("int8_w8a8", INT8_W8A8_EXAMPLES),
            ("int8_weights", INT8_WEIGHT_EXAMPLES),
        ):  # INT8 (#178): each example with its own backend
            for name, spec in found.items():
                if runs([name], name.split("_", 1)[0]):
                    ok &= smoke_fp8(Path(tmp), verbose, precision=precision, examples={name: spec})
        if tc.backends.get("triton") and (backends is None or "triton" in backends):
            attention = runs(["triton_short_attention.py"], "triton")
            ok &= smoke_triton_tools(Path(tmp), verbose, attention=attention)
        if runs(CUTE_W8A8_EXAMPLES, "cute"):
            ok &= smoke_fp8(Path(tmp), verbose, precision="fp8_w8a8", examples=CUTE_W8A8_EXAMPLES)
        if runs(CUTE_W4A4_EXAMPLES, "cute"):  # #233
            ok &= smoke_fp4(Path(tmp), verbose, precision="fp4_w4a4", examples=CUTE_W4A4_EXAMPLES)
        if runs(CUTE_BLOCK_EXAMPLES, "cute"):
            ok &= smoke_fp8(
                Path(tmp), verbose, examples=CUTE_BLOCK_EXAMPLES, capture=make_mlp_capture
            )
        for name, spec in CUTE_ARCH_EXAMPLES.items():  # wgmma / tcgen05 templates (#228)
            if runs([name], "cute"):
                ok &= smoke_fp8(Path(tmp), verbose, precision="fp8_w8a8", examples={name: spec})
        if runs(HELION_EXAMPLES, "helion"):
            ok &= smoke_helion(Path(tmp), verbose)
        cuda = runs(CUBLASLT_EXAMPLES, "cuda")
        triton = runs([*PRODUCER_EXAMPLES, *FP8_KV_EXAMPLES], "triton")
        if cuda or triton:
            ok &= smoke_fp8_toolkit(Path(tmp), verbose, cuda=cuda, triton=triton)
        if verbose and skipped:
            print(f"  skipped here ({gpu_arch.from_toolchain(tc).label}):")
            for name, why in skipped.items():
                print(f"    {name.removesuffix('.py'):28s} {why}")
    return ok
