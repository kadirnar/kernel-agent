"""What Triton makes of ``tl.dot`` per arch (issue #259). CPU only: ``triton.compile`` for a
``GPUTarget`` (Triton's own ptxas), no launch.

A K-loop GEMM (``num_stages=3``, 4 warps, 64 x 64 x 64 tiles; B either [K, N] or the
nn.Linear [N, K] layout) is compiled for sm_75 / 80 / 86 / 89 / 120 with each operand dtype.
Reported per case: compiled or the error, the ``mma.sync`` forms in the PTX ("FMA" when
there is none), the number of ``cp.async`` and ``ldmatrix`` instructions (pipelining and
tensor-core operand loads) and the shared memory the kernel needs. Then
``tl.dot_scaled`` (e4m3, unit ue8m0 scales) and the bundled FP8 / INT8 W8A8 GEMMs. Run:

    CUDA_VISIBLE_DEVICES= PYTHONPATH=src python docs/research-scripts/older-gpus-259/lowering_triton.py \
        > docs/research-scripts/older-gpus-259/lowering_triton.txt
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

ARCHS = (75, 80, 86, 89, 120)
EXAMPLES = ROOT / "src" / "kernel_agent" / "agent" / "examples"


@triton.jit
def gemm(
    a,
    b,
    c,
    K,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    B_NK: tl.constexpr,
    PREC: tl.constexpr,
    INT: tl.constexpr,
):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptr = a + rm[:, None] * K + rk[None, :]
    if B_NK:  # nn.Linear weight layout: b[n, k]
        b_ptr = b + rn[None, :] * K + rk[:, None]
    else:  # b[k, n]
        b_ptr = b + rk[:, None] * BN + rn[None, :]
    if INT:
        acc = tl.zeros((BM, BN), tl.int32)
    else:
        acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(0, K, BK):
        if INT:
            acc = tl.dot(tl.load(a_ptr), tl.load(b_ptr), acc, out_dtype=tl.int32)
        else:
            acc = tl.dot(tl.load(a_ptr), tl.load(b_ptr), acc, input_precision=PREC)
        a_ptr += BK
        if B_NK:
            b_ptr += BK
        else:
            b_ptr += BK * BN
    tl.store(c + rm[:, None] * BN + rn[None, :], acc)


@triton.jit
def gemm_scaled(a, b, c, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptr = a + rm[:, None] * K + rk[None, :]
    b_ptr = b + rn[None, :] * K + rk[:, None]
    ones_a = tl.full((BM, BK // 32), 127, tl.uint8)
    ones_b = tl.full((BN, BK // 32), 127, tl.uint8)
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(0, K, BK):
        acc = tl.dot_scaled(tl.load(a_ptr), ones_a, "e4m3", tl.load(b_ptr), ones_b, "e4m3", acc)
        a_ptr += BK
        b_ptr += BK
    tl.store(c + rm[:, None] * BN + rn[None, :], acc)


def summarise(ptx: str, shared: int) -> str:
    forms = sorted(set(re.findall(r"mma\.sync\.aligned\.([\w.]+)", ptx)))
    forms = [f.replace("row.col.", "") for f in forms]
    cp_async = len(re.findall(r"cp\.async\.c[ag]", ptx))
    ldmatrix = len(re.findall(r"ldmatrix", ptx))
    return (
        f"mma {', '.join(forms) if forms else 'none (FMA)'}; cp.async {cp_async}; "
        f"ldmatrix {ldmatrix}; smem {shared} B"
    )


def build(
    fn, signature: dict, consts: dict, cc: int, stages: int = 3, warps: int = 4, aligned=True
) -> str:
    """Compile like a launch would: pointers and integer arguments specialised as divisible
    by 16 (``tt.divisibility``, what the JIT does for 16-byte aligned tensors and sizes that
    are multiples of 16). Without it loads are not vectorised and never become cp.async."""
    sig = {**signature, **dict.fromkeys(consts, "constexpr")}
    names = list(fn.arg_names)
    attrs = {}
    if aligned:
        attrs = {
            (names.index(k),): [["tt.divisibility", 16]]
            for k, v in signature.items()
            if v.startswith("*") or v in ("i32", "i64")
        }
    try:
        compiled = triton.compile(
            ASTSource(fn=fn, signature=sig, constexprs=consts, attrs=attrs),
            target=GPUTarget("cuda", cc, 32),
            options={"num_warps": warps, "num_stages": stages},
        )
    except Exception as exc:
        text = " ".join(str(exc).split())
        return f"FAIL {type(exc).__name__}: {text[:150]}"
    return summarise(compiled.asm["ptx"], compiled.metadata.shared)


CASES = (
    # name, a / b pointer type, c type, input_precision, int accumulation
    ("fp16", "*fp16", "*fp32", None, False),
    ("bf16", "*bf16", "*fp32", None, False),
    ("fp32 (default tf32)", "*fp32", "*fp32", "tf32", False),
    ("fp32 ieee", "*fp32", "*fp32", "ieee", False),
    ("int8", "*i8", "*i32", None, True),
    ("fp8e4nv (e4m3)", "*fp8e4nv", "*fp32", None, False),
    ("fp8e5 (e5m2)", "*fp8e5", "*fp32", None, False),
    ("fp8e4b15", "*fp8e4b15", "*fp32", None, False),
)


def load_example(name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    print(f"# Triton {triton.__version__}; 64x64x64 tiles, 4 warps, num_stages=3 unless noted")
    for b_nk in (False, True):
        layout = "B [N, K] (nn.Linear)" if b_nk else "B [K, N]"
        print(f"\n## tl.dot, {layout}")
        for name, ptr, out, prec, integer in CASES:
            sig = {"a": ptr, "b": ptr, "c": out, "K": "i32"}
            consts = {"BM": 64, "BN": 64, "BK": 64, "B_NK": b_nk, "PREC": prec, "INT": integer}
            for cc in ARCHS:
                print(f"{name:22s} sm_{cc:<4d} {build(gemm, sig, consts, cc)}")
    print("\n## tl.dot_scaled e4m3, unit ue8m0 scales, B [N, K]")
    sig = {"a": "*fp8e4nv", "b": "*fp8e4nv", "c": "*fp32", "K": "i32"}
    for cc in ARCHS:
        print(f"{'dot_scaled e4m3':22s} sm_{cc:<4d} {build(gemm_scaled, sig, {'BM': 64, 'BN': 64, 'BK': 64}, cc)}")

    print("\n## the same GEMMs without the divisibility specialisation (unaligned pointers)")
    for name, ptr, prec in (("fp16", "*fp16", None), ("fp32 (tf32)", "*fp32", "tf32")):
        sig = {"a": ptr, "b": ptr, "c": "*fp32", "K": "i32"}
        consts = {"BM": 64, "BN": 64, "BK": 64, "B_NK": True, "PREC": prec, "INT": False}
        for cc in (80, 89):
            text = build(gemm, sig, consts, cc, aligned=False)
            print(f"{name + ' unaligned':22s} sm_{cc:<4d} {text}")

    print("\n## bundled W8A8 GEMMs (their kernels, BM, BN, BK, warps, stages)")
    sig = {
        "a": "*i8",
        "b": "*i8",
        "sa": "*fp32",
        "sb": "*fp32",
        "bias": "*bf16",
        "c": "*bf16",
        "M": "i32",
        "N": "i32",
        "K": "i32",
    }
    fp8 = load_example("triton_fp8_w8a8_gemm")
    for cc in (89, 120):
        for bm, bn, bk, warps, stages in (fp8.DEFAULT[:5], (128, 64, 128, 8, 3), (64, 64, 64, 4, 4)):
            consts = {"HAS_BIAS": False, "BM": bm, "BN": bn, "BK": bk, "GM": 8, "SCALED": False}
            fsig = {**sig, "a": "*fp8e4nv", "b": "*fp8e4nv"}
            text = build(fp8._gemm_kernel, fsig, consts, cc, stages=stages, warps=warps)
            print(f"triton_fp8_w8a8_gemm tl.dot sm_{cc} {(bm, bn, bk, warps, stages)}: {text}")
    int8 = load_example("triton_int8_w8a8_gemm")
    for cc in ARCHS:
        consts = {"HAS_BIAS": False, "BM": 64, "BN": 64, "BK": 128, "GM": 8}
        text = build(int8._gemm_kernel, sig, consts, cc)
        print(f"triton_int8_w8a8_gemm sm_{cc} (64, 64, 128, 4, 3): {text}")


if __name__ == "__main__":
    main()
