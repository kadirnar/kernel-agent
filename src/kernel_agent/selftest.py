"""Backend smoke test: run every bundled example kernel through the evaluator (the FP8
weight-only, W8A8 and MXFP8 examples in the near-lossless tier, and their rejection by the
exact tier, MXFP8 with the OCP floor scale rule by the scale-rule guard; the FP4 one in the
near-lossless-fp4 tier, and its rejection by the FP8 tier)."""

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


#: The FP4 weight-only examples (``precision: fp4_weights``) on the GEMV's shapes above; a
#: 22-row call (0 calls per run: correctness only) takes the dequantised fallback.
FP4_EXAMPLES: dict[str, tuple[int, int, list[tuple[tuple[int, ...], int]]]] = {
    "cuda_fp4_gemv.py": (2048, 12288, [((1,), 28), ((4,), 1), ((2, 11), 0)]),
}


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


def smoke_fp8(tmp: Path, verbose: bool = False, *, precision: str = "fp8_weights") -> bool:
    """Every FP8 example of ``precision`` (:data:`FP8_EXAMPLES`, :data:`W8A8_EXAMPLES`,
    :data:`MX_EXAMPLES`) passes the evaluator in the near-lossless tier, and the exact tier (a
    quick check) rejects it; an MXFP8 example with the OCP floor scale rule fails the
    scale-rule guard."""
    from kernel_agent.kernels.evaluate import run_evaluation

    ok = True
    examples = _EXAMPLES.get(precision, FP8_EXAMPLES)
    for name, (k, n, calls) in examples.items():
        near = make_linear_capture(
            tmp / f"{name}.near.pt", k, n, calls, tier="near-lossless", precision=precision
        )
        exact = make_linear_capture(tmp / f"{name}.exact.pt", k, n, calls)
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
        if fp8_supported(tc) and (backends is None or "cuda" in backends):
            ok &= smoke_fp8(Path(tmp), verbose)
            ok &= smoke_fp4(Path(tmp), verbose)
        if w8a8_supported(tc) and (backends is None or "triton" in backends):
            ok &= smoke_fp8(Path(tmp), verbose, precision="fp8_w8a8")
        if mxfp8_supported(tc) and (backends is None or "triton" in backends):
            ok &= smoke_fp8(Path(tmp), verbose, precision="fp8_mx")
    return ok
