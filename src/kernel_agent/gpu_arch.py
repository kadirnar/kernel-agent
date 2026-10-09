"""What the GPU of a run is: its architecture's features and what it cannot run (#165).

kernel-agent runs on any NVIDIA GPU torch supports. What differs per architecture is
decided here from the detected compute capability (:class:`kernel_agent.toolchain.GPUInfo`)
and the peaks measured on the GPU (:mod:`kernel_agent.kernels.roofline`,
:mod:`kernel_agent.kernels.mma_peaks`), never from the GPU the toolkit was developed on (an
RTX 5070 Ti, sm_120: its measurements stay in the skills as labelled evidence).

* :func:`family`: the architecture family of a capability (:data:`FAMILIES`): its tensor-core
  instructions, the one a compute-bound kernel needs for the full rate, async copies,
  clusters / PDL, the typical shared memory per block.
* :func:`precision_unsupported`: why a target precision cannot run on a capability
  (:data:`PRECISION_NEEDS`: FP8 tensor-core math needs sm_89+, block-scaled MXFP8 and W4A4
  sm_100+, INT8 tensor-core math (IMMA) sm_75+). Weight-only and KV-cache formats dequantise in
  registers and run everywhere; before sm_89 without a hardware e4m3 conversion, and INT8
  W8A8's fast path differs per family (:func:`precision_note`).
* :func:`supports` / :func:`example_requirement` / :func:`example_skip`: the ``ARCHS``
  declaration of a bundled example (``"sm_89+"``, ``"sm_12x"``) against a capability, and
  why it does not run here (``doctor --smoke``).
* :func:`summary_lines` (in :meth:`Toolchain.summary <kernel_agent.toolchain.Toolchain.summary>`,
  so in ``doctor`` and every prompt's toolchain block): the family, what runs at full rate,
  the precisions this GPU cannot run, the 16-bit ridge from the measured peaks (at the
  16-bit tensor-core dtype, :func:`tensor_core_16bit`: fp16 on Volta / Turing) and what fp32
  accumulation costs (:func:`kernel_agent.kernels.mma_peaks.accumulation_line`);
  :func:`prompt_section`: those facts and this family's section of ``gpu-architectures/gpus.md``.
* :func:`from_summary`: the GPU, its capability and its measured instruction rates read
  back from a toolchain summary (the prompts receive the summary text).
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Every family's section (the ``gpu-architectures`` skill, issue #176).
KNOWLEDGE = Path(__file__).parent / "agent" / "plugin" / "skills" / "gpu-architectures" / "gpus.md"


@dataclass(frozen=True)
class Family:
    """One architecture family (a row of :data:`FAMILIES`)."""

    key: str  # gpu-architectures/gpus.md section, backends.ARCH_POLICY key
    name: str
    archs: str  # its compute capabilities
    gpus: str  # examples
    mma: str  # tensor-core instructions
    full_rate: str  # what a compute-bound kernel must issue to reach the tensor-core peak
    copies: str  # async copies, clusters, PDL
    smem_kb: float  # typical shared memory per block (opt-in), when the GPU does not say
    features: frozenset[str] = field(default_factory=frozenset)


#: Feature names of :attr:`Family.features`: ``bf16_tc`` bf16 tensor cores, ``fp8_tc`` FP8
#: (e4m3 / e5m2) tensor cores, ``block_scaled`` block-scaled MMA (MXFP8 / MXFP4 / NVFP4),
#: ``mma_block_scale`` block-scaled ``mma.sync`` (``kind::mxf8f6f4.block_scale``), ``fp4_tc``
#: FP4 tensor cores, ``wgmma`` warpgroup MMA, ``tcgen05`` tensor-memory MMA, ``tma`` /
#: ``clusters`` / ``pdl`` / ``cp_async``.
FAMILIES: tuple[Family, ...] = (
    Family(
        "pre_ampere",
        "Volta / Turing",
        "sm_70, sm_72, sm_75",
        "V100, T4, RTX 20xx",
        "`mma.sync` fp16 (and int8 on sm_75); no bf16, TF32 or FP8 tensor cores",
        "`mma.sync` on fp16 (bf16 models run their matmuls without tensor cores here)",
        "no `cp.async`, no TMA",
        64.0,
        frozenset(),
    ),
    Family(
        "ampere",
        "Ampere",
        "sm_80, sm_86, sm_87",
        "A100, A30, A10, A40, RTX A6000, RTX 30xx, Jetson Orin",
        "`mma.sync` bf16 / fp16 / tf32 / int8; no FP8 tensor cores",
        "`mma.sync` (HMMA) with `ldmatrix` and a `cp.async` multi-stage pipeline",
        "`cp.async` (no TMA, no clusters, no PDL)",
        99.0,
        frozenset({"bf16_tc", "cp_async"}),
    ),
    Family(
        "ada",
        "Ada Lovelace",
        "sm_89",
        "RTX 40xx, RTX 6000 Ada, L4, L40S",
        "`mma.sync` bf16 / fp16 / tf32 / int8 and FP8 e4m3 / e5m2 (QMMA); no block-scaled MMA",
        "`mma.sync` (HMMA; QMMA for FP8, the only FP8 instruction here)",
        "`cp.async` (no TMA, no clusters, no PDL)",
        99.0,
        frozenset({"bf16_tc", "fp8_tc", "cp_async"}),
    ),
    Family(
        "hopper",
        "Hopper",
        "sm_90",
        "H100, H200, GH200, H20",
        "`wgmma` (warpgroup MMA from shared memory, sm_90a) incl. FP8, and `mma.sync`; no "
        "block-scaled MMA, no `tcgen05`",
        "`wgmma` fed by TMA, warp-specialised (`mma.sync` reaches ~2/3 of it; FP8 `mma.sync` "
        "is emulated through fp16 here)",
        "TMA (`cp.async.bulk.tensor`, multicast), clusters with distributed shared memory, "
        "PDL, `cp.async`",
        227.0,
        frozenset({"bf16_tc", "fp8_tc", "wgmma", "tma", "clusters", "pdl", "cp_async"}),
    ),
    Family(
        "blackwell",
        "Blackwell (datacenter)",
        "sm_100, sm_103, sm_110",
        "B200, GB200, B300, GB300",
        "`tcgen05.mma` (tensor memory accumulators, 2-CTA pairs) incl. FP8 and block-scaled "
        "MXFP8 / MXFP4 / NVFP4, and `mma.sync`; no `wgmma`",
        "`tcgen05.mma` with TMEM accumulators fed by TMA (`mma.sync` saturates near a quarter "
        "of the B200 peak; FP8 `mma.sync` is emulated through fp16 here)",
        "TMA (multicast), clusters, PDL, `cp.async`",
        227.0,
        frozenset(
            {
                "bf16_tc",
                "fp8_tc",
                "block_scaled",
                "fp4_tc",
                "tcgen05",
                "tma",
                "clusters",
                "pdl",
                "cp_async",
            }
        ),
    ),
    Family(
        "blackwell_geforce",
        "Blackwell (GeForce / RTX PRO / DGX Spark)",
        "sm_120, sm_121",
        "RTX 50xx, RTX PRO 6000 Blackwell, DGX Spark",
        "`mma.sync` incl. FP8 and block-scaled FP8 / FP6 / FP4 "
        "(`kind::mxf8f6f4.block_scale`, sm_120a); no `wgmma`, no `tcgen05` / TMEM",
        "`mma.sync`; for FP8 the block-scaled `QMMA.SF` (plain e4m3 `QMMA.F32` with fp32 "
        "accumulation runs at half its rate on GeForce: the measured rates say it here)",
        "TMA without multicast (CUTLASS's GEMMs use 1x1x1 clusters), clusters, PDL, `cp.async`",
        99.0,
        frozenset(
            {
                "bf16_tc",
                "fp8_tc",
                "block_scaled",
                "mma_block_scale",
                "fp4_tc",
                "tma",
                "clusters",
                "pdl",
                "cp_async",
            }
        ),
    ),
    Family(
        "newer",
        "newer than this table",
        "sm_130+",
        "",
        "unknown: read the measured instruction rates and `doctor`'s probes",
        "unknown: measure",
        "unknown",
        99.0,
        frozenset(
            {"bf16_tc", "fp8_tc", "block_scaled", "fp4_tc", "tma", "clusters", "pdl", "cp_async"}
        ),
    ),
)
_BY_KEY = {f.key: f for f in FAMILIES}
#: Opt-in shared memory per block (KB) per compute capability (CUDA Programming Guide,
#: compute capabilities), for a GPU whose properties do not report it.
SMEM_PER_BLOCK_KB: dict[tuple[int, int], float] = {
    (7, 0): 96.0,
    (7, 5): 64.0,
    (8, 0): 163.0,
    (8, 6): 99.0,
    (8, 7): 163.0,
    (8, 9): 99.0,
    (9, 0): 227.0,
    (10, 0): 227.0,
    (10, 3): 227.0,
    (12, 0): 99.0,
    (12, 1): 99.0,
}


def family(capability: tuple[int, ...] | None) -> Family | None:
    """The family of a compute capability (None without one)."""
    if capability is None:
        return None
    major, minor = int(capability[0]), int(capability[1]) if len(capability) > 1 else 0
    if major < 8:
        return _BY_KEY["pre_ampere"]
    if major == 8:
        return _BY_KEY["ada" if minor >= 9 else "ampere"]
    if major == 9:
        return _BY_KEY["hopper"]
    if major in (10, 11):
        return _BY_KEY["blackwell"]
    if major == 12:
        return _BY_KEY["blackwell_geforce"]
    return _BY_KEY["newer"]


def has(capability: tuple[int, ...] | None, feature: str) -> bool:
    """Whether the family of ``capability`` has ``feature`` (:data:`FAMILIES`)."""
    fam = family(capability)
    return fam is not None and feature in fam.features


#: Short names of the 16-bit ``peaks["tflops"]`` keys.
SHORT_16BIT = {"bfloat16": "bf16", "float16": "fp16"}


def tensor_core_16bit(capability: tuple[int, ...] | None) -> str:
    """The 16-bit dtype (``peaks["tflops"]`` key) this GPU's tensor cores run: ``bfloat16``
    from Ampere on, ``float16`` on Volta / Turing (no bf16 tensor cores: bf16 matmuls run on
    CUDA cores there); ``bfloat16`` for an unknown GPU."""
    if capability is None or has(capability, "bf16_tc"):
        return "bfloat16"
    return "float16"


def arch_of(capability: tuple[int, ...] | None) -> str | None:
    """``sm_90`` for (9, 0)."""
    return None if capability is None else f"sm_{capability[0]}{capability[1]}"


def capability_of(arch: str | None) -> tuple[int, int] | None:
    """(12, 0) for ``sm_120`` / ``sm_120a`` / ``compute_120`` (None when not an arch)."""
    match = re.fullmatch(r"(?:sm|compute)_(\d{2,3})[a-z]?", str(arch or "").strip())
    if not match:
        return None
    digits = match.group(1)
    return int(digits[:-1]), int(digits[-1])


# ------------------------------------------------------------------ precisions

#: Target precisions whose math needs tensor cores a GPU may lack: (minimum capability, what).
PRECISION_NEEDS: dict[str, tuple[tuple[int, int], str]] = {
    "fp8_w8a8": ((8, 9), "FP8 (e4m3) tensor cores: sm_89+ (Ada, Hopper, Blackwell)"),
    "fp8_mx": (
        (10, 0),
        "block-scaled FP8 tensor cores (MXFP8): sm_100+ (Blackwell; cuBLASLt VEC32_UE8M0)",
    ),
    "int8_w8a8": ((7, 5), "INT8 tensor cores (IMMA, `mma.sync` s8): sm_75+ (Turing and newer)"),
    "fp4_w4a4": (
        (10, 0),
        "block-scaled FP4 tensor cores (NVFP4 / MXFP4: `tcgen05.mma kind::mxf4nvf4` on sm_100 / "
        "sm_103, `mma.sync kind::mxf4nvf4.block_scale` on sm_120 / sm_121): sm_100+ (Blackwell)",
    ),
}
#: Ceilings columns (``profiling/ceilings.py``) whose floor needs such tensor cores.
COLUMN_NEEDS: dict[str, tuple[tuple[int, int], str]] = {
    "w8a8": PRECISION_NEEDS["fp8_w8a8"],
    "mxfp8": PRECISION_NEEDS["fp8_mx"],
    "w4a4": PRECISION_NEEDS["fp4_w4a4"],
    "int8_w8a8": PRECISION_NEEDS["int8_w8a8"],
}
#: Weight-only / KV-cache formats: dequantised in registers, so any GPU runs them; below
#: sm_89 there is no hardware e4m3 conversion.
_SOFTWARE_FP8 = (
    "this GPU has no hardware e4m3 conversion (sm_89+): CUDA's `__nv_cvt_fp8x2_to_halfraw2` "
    "/ `__nv_cvt_fp8_to_halfraw` fall back to a software conversion (correct, a few integer "
    "ops more per value; still bandwidth bound in a GEMV), and Triton has no e4m3 type here "
    "(load the codes as `uint8` and convert with bit operations, or use CUDA C++)"
)


def _below(capability: tuple[int, ...], need: tuple[int, int]) -> bool:
    return (int(capability[0]), int(capability[1]) if len(capability) > 1 else 0) < need


def precision_unsupported(precision: str | None, capability: tuple[int, ...] | None) -> str | None:
    """Why target ``precision`` cannot run on a GPU of ``capability`` (None: it can, or no
    capability is known)."""
    need = PRECISION_NEEDS.get(str(precision or ""))
    if capability is None or need is None or not _below(capability, need[0]):
        return None
    return f"needs {need[1]}; this GPU is {arch_of(capability)}"


def column_unsupported(column: str, capability: tuple[int, ...] | None) -> str | None:
    """Why the ceilings column ``column`` is out of reach on a GPU of ``capability``."""
    need = COLUMN_NEEDS.get(column)
    if capability is None or need is None or not _below(capability, need[0]):
        return None
    return f"needs {need[1]}"


#: What INT8 W8A8 (``int8_w8a8``) needs per family for the INT8 tensor-core peak (#178).
_INT8_PATH = {
    "pre_ampere": (
        "Turing's IMMA is `mma.sync` m8n8k16 (`m16n8k32` needs sm_80) and Triton's int8 "
        "`tl.dot` does not compile below sm_80 (Triton 3.8): cuBLASLt (`torch._int_mm`) or "
        "CUDA C++ m8n8k16 here"
    ),
    "hopper": (
        "the INT8 peak needs `wgmma` s8 (Triton `tl.dot` on int8 emits it, CUTLASS sm_90); a "
        "hand-written `mma.sync` s8 kernel stops below it"
    ),
    "blackwell": (
        "the INT8 peak needs `tcgen05.mma kind::i8` (cuBLASLt via `torch._int_mm`, CUTLASS / "
        "CuTe DSL sm_100); a hand-written `mma.sync` s8 kernel stops below it"
    ),
}
#: Blackwell Ultra (sm_103: B300 / GB300) keeps INT8 tensor cores at about 1/30 of its FP8
#: rate (NVIDIA's B300 specifications; B200 and H100 / H200: 1:1) and its PTX exposes no
#: ``tcgen05.mma kind::i8`` (arXiv 2608.11693).
_INT8_CUT = (
    "Blackwell Ultra (sm_103) runs INT8 tensor-core math at a small fraction of its FP8 rate "
    "(NVIDIA's B300 specification: ~1/30) and has no `tcgen05.mma kind::i8` (warp-level IMMA "
    "only): compare the *INT8 W8A8* and *W8A8* floors of the *Ceilings* table (measured "
    "peaks) and prefer FP8 W8A8 unless INT8 is measured faster"
)


#: What W4A4 (``fp4_w4a4``, #233) needs per family for the FP4 tensor-core peak.
_W4A4_PATH = {
    "blackwell_geforce": (
        "the FP4 peak needs the block-scaled `mma.sync.aligned.kind::mxf4nvf4.block_scale."
        "scale_vec::4X.m16n8k64` (e2m1 x e2m1, a ue4m3 scale per 16; CuTe DSL `MmaMXF4NVF4Op`, "
        "sm_120a / sm_121a only) or cuBLASLt through `F.scaled_mm` (`BlockWise1x16`); torch "
        "2.14 has no MXFP4 GEMM here (CuTe `MmaMXF4Op` or Triton `tl.dot_scaled` on e2m1)"
    ),
    "blackwell": (
        "the FP4 peak needs `tcgen05.mma kind::mxf4nvf4` (TMEM accumulators fed by TMA: "
        "CUTLASS / CuTe DSL sm_100 block-scaled GEMMs, cuBLASLt through `F.scaled_mm`); the "
        "block-scaled `mma.sync` of sm_120 does not exist here"
    ),
}


def precision_note(precision: str | None, capability: tuple[int, ...] | None) -> str | None:
    """What an engineer of a ``precision`` target must know about this GPU (None: nothing)."""
    if capability is None:
        return None
    if precision == "fp4_w4a4":
        fam = family(capability)
        why = _W4A4_PATH.get(fam.key) if fam is not None else None
        return why + "." if why else None
    if precision == "int8_w8a8":
        if tuple(int(c) for c in capability[:2]) == (10, 3):
            return _INT8_CUT + "."
        fam = family(capability)
        why = _INT8_PATH.get(fam.key) if fam is not None else None
        return why + "." if why else None
    if precision not in ("fp8_weights", "fp4_weights", "fp8_kv"):
        return None
    if _below(capability, (8, 9)):
        return _SOFTWARE_FP8 + "."
    return None


# ------------------------------------------------------------------ example declarations

_SPEC = re.compile(r"sm_(\d{1,3})(x|\+)?")


def supports(spec: str | None, capability: tuple[int, ...] | None) -> bool:
    """Whether ``capability`` matches an ``ARCHS`` declaration: comma-separated items,
    ``sm_89+`` (that capability or newer), ``sm_12x`` (major 12), ``sm_90`` (exactly);
    None / empty: every GPU. False without a capability (no GPU)."""
    if not spec:
        return True
    if capability is None:
        return False
    cc = (int(capability[0]), int(capability[1]) if len(capability) > 1 else 0)
    for item in (s.strip() for s in spec.split(",")):
        match = _SPEC.fullmatch(item)
        if not match or (match.group(2) != "x" and len(match.group(1)) < 2):
            raise ValueError(f"bad ARCHS item {item!r} (sm_89+, sm_12x or sm_90)")
        digits, mode = match.groups()
        if mode == "x":
            if cc[0] == int(digits):
                return True
            continue
        want = (int(digits[:-1]), int(digits[-1]))
        if (cc >= want) if mode == "+" else (cc == want):
            return True
    return False


def _declarations(path: Path) -> dict[str, str]:
    """The ``ARCHS*`` constants at the top level of a bundled example (a file, or a project
    directory's ``build.py`` / first ``.py``), read with ``ast``: the example is not imported
    (it may need a backend this machine lacks)."""
    if path.is_dir():
        files = [path / "build.py"] if (path / "build.py").is_file() else sorted(path.glob("*.py"))
        if not files:
            return {}
        path = files[0]
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError, ValueError):
        return {}
    found: dict[str, str] = {}
    for node in tree.body:
        value: ast.expr | None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name, value = getattr(node.targets[0], "id", None), node.value
        elif isinstance(node, ast.AnnAssign):
            name, value = getattr(node.target, "id", None), node.value
        else:
            continue
        if name in ("ARCHS", "ARCHS_WHY", "ARCHS_COMPILES") and isinstance(value, ast.Constant):
            found[str(name)] = str(value.value)
    return found


def example_requirement(path: Path) -> tuple[str | None, str]:
    """``(ARCHS, ARCHS_WHY)`` declared by a bundled example (:func:`_declarations`);
    ``(None, "")`` without them."""
    found = _declarations(path)
    return found.get("ARCHS"), found.get("ARCHS_WHY", "")


def example_compiles(path: Path) -> str | None:
    """The GPUs a bundled example's device code compiles for: its ``ARCHS_COMPILES`` where a
    limit is runtime-only (it compiles for more GPUs than it runs on: PDL, FP8 tensor cores,
    a library call), else its ``ARCHS``. The CPU compile matrix
    (``tests/test_arch_matrix.py``) checks it on sm_75 / sm_80 / sm_86 / sm_89 / sm_120."""
    found = _declarations(path)
    return found.get("ARCHS_COMPILES", found.get("ARCHS"))


def example_skip(path: Path, capability: tuple[int, ...] | None) -> str | None:
    """Why the bundled example at ``path`` does not run on a GPU of ``capability`` (its
    ``ARCHS``), or None."""
    spec, why = example_requirement(path)
    if supports(spec, capability):
        return None
    here = arch_of(capability) or "no GPU"
    return f"needs {spec}" + (f" ({why})" if why else "") + f"; this GPU is {here}"


# ------------------------------------------------------------------ facts of one GPU


@dataclass(frozen=True)
class Facts:
    """The GPU an agent works on, as the prompts know it: name, capability and the measured
    tensor-core instruction rates (``peaks["mma_tflops"]``, :mod:`.kernels.mma_peaks`)."""

    name: str | None = None
    capability: tuple[int, int] | None = None
    mma: Mapping[str, float] = field(default_factory=dict)

    @property
    def arch(self) -> str | None:
        return arch_of(self.capability)

    @property
    def family(self) -> Family | None:
        return family(self.capability)

    def has(self, feature: str) -> bool:
        return has(self.capability, feature)

    @property
    def label(self) -> str:
        """``NVIDIA H100 (sm_90, Hopper)``, or ``an unknown GPU``."""
        if self.capability is None:
            return "an unknown GPU"
        fam = self.family
        return f"{self.name or 'GPU'} ({self.arch}, {fam.name if fam else '?'})"


def from_toolchain(tc: Any) -> Facts:
    """:class:`Facts` of a :class:`~kernel_agent.toolchain.Toolchain`."""
    gpu = getattr(tc, "gpu", None)
    if gpu is None:
        return Facts()
    peaks = getattr(tc, "peaks", None) or {}
    mma = {k: float(v) for k, v in (peaks.get("mma_tflops") or {}).items()}
    return Facts(gpu.name, (int(gpu.capability[0]), int(gpu.capability[1])), mma)


_GPU_LINE = re.compile(r"^GPU: (?P<name>.+?) \((?P<arch>sm_\d{2,3})[,)]", re.M)


def from_summary(text: str | None) -> Facts:
    """:class:`Facts` read back from a :meth:`Toolchain.summary
    <kernel_agent.toolchain.Toolchain.summary>` (its ``GPU:`` line and the measured
    ``mma.sync`` rates of its peaks line); empty :class:`Facts` when it names no GPU."""
    match = _GPU_LINE.search(text or "")
    if not match:
        return Facts()
    from kernel_agent.kernels.mma_peaks import INSTRUCTIONS

    mma = {}
    for instruction in INSTRUCTIONS:
        label = r"(?<![\w.])" + re.escape(instruction.label)  # not inside a longer label
        rate = re.search(label + r" (\d+(?:\.\d+)?)\b", text or "")
        if rate:
            mma[instruction.key] = float(rate.group(1))
    return Facts(match.group("name"), capability_of(match.group("arch")), mma)


def _features(fam: Family) -> str:
    names = {
        "fp8_tc": "FP8 tensor cores",
        "block_scaled": "block-scaled MMA (MXFP8 / NVFP4)",
        "wgmma": "wgmma",
        "tcgen05": "tcgen05 / TMEM",
        "tma": "TMA",
        "clusters": "clusters",
        "pdl": "PDL",
    }
    have = [label for key, label in names.items() if key in fam.features]
    lack = [label for key, label in names.items() if key not in fam.features]
    return f"has {', '.join(have) or 'none of the newer features'}" + (
        f"; no {', '.join(lack)}" if lack else ""
    )


def summary_lines(gpu: Any, peaks: Mapping[str, Any] | None) -> list[str]:
    """``Toolchain.summary`` lines about the GPU's architecture (none without a GPU): its
    family and features, the instruction a compute-bound kernel needs, the precisions it
    cannot run, the 16-bit ridge of the measured peaks (labelled with the tensor-core
    dtype: bf16, fp16 on Volta / Turing) and the measured fp32- vs fp16-accumulating HMMA
    rates."""
    if gpu is None:
        return []
    capability = (int(gpu.capability[0]), int(gpu.capability[1]))
    fam = family(capability)
    if fam is None:
        return []
    lines = [
        f"arch: {fam.name} ({arch_of(capability)}): {_features(fam)}",
        f"tensor cores: {fam.mma}; full rate needs {fam.full_rate}",
    ]
    if not getattr(gpu, "smem_per_block_kb", 0):
        kb = SMEM_PER_BLOCK_KB.get(capability, fam.smem_kb)
        lines.append(
            f"shared memory per block: not detected ({kb:.0f} KB for {arch_of(capability)} "
            "in the CUDA Programming Guide; `cudaDevAttrMaxSharedMemoryPerBlockOptin` says)"
        )
    refused = [
        f"{name} ({why})"
        for name in PRECISION_NEEDS
        if (why := precision_unsupported(name, capability)) is not None
    ]
    if refused:
        lines.append("precisions this GPU cannot run: " + "; ".join(refused))
    if (note := precision_note("fp8_weights", capability)) is not None:
        lines.append(f"weight-only FP8 / FP4 and FP8 KV caches run, but {note}")
    fp8_math = precision_unsupported("fp8_w8a8", capability) is None
    if not fp8_math and precision_unsupported("int8_w8a8", capability) is None:
        lines.append(
            "8-bit tensor-core math here is INT8 (`int8_w8a8`: IMMA, s8 x s8 -> int32), and "
            "`int8_weights` halves weight streams without an e4m3 conversion"
        )
    caveat = fam.key == "pre_ampere" or capability == (10, 3)  # no Triton IMMA / INT8 cut
    if caveat and (note := precision_note("int8_w8a8", capability)) is not None:
        lines.append(f"INT8 W8A8: {note}")
    tflops = (peaks or {}).get("tflops") or {}
    dtype = tensor_core_16bit(capability)
    short = SHORT_16BIT[dtype]
    if peaks and tflops.get(dtype) and peaks.get("dram_gbps"):
        dram = float(peaks["dram_gbps"])
        ridge = float(tflops[dtype]) * 1000 / dram
        line = (
            f"16-bit ridge ({short}, the 16-bit tensor-core dtype here): a GEMM turns compute "
            f"bound near M ≈ {ridge:.0f} rows per weight read (measured {short} "
            f"{float(tflops[dtype]):.0f} TFLOP/s / DRAM {dram:.0f} GB/s)"
        )
        if dtype != "bfloat16" and tflops.get("bfloat16"):
            line += (
                f"; bf16 matmuls run without tensor cores here ({float(tflops['bfloat16']):.0f} "
                f"TFLOP/s: M ≈ {float(tflops['bfloat16']) * 1000 / dram:.0f})"
            )
        lines.append(line)
    from kernel_agent.kernels.mma_peaks import accumulation_line

    if (accumulate := accumulation_line((peaks or {}).get("mma_tflops") or {})) is not None:
        lines.append(accumulate)
    if (sustained := sustained_line(peaks)) is not None:  # a power-capped board (#253)
        lines.append(sustained)
    return lines


def sustained_line(peaks: Mapping[str, Any] | None) -> str | None:
    """What the GPU sustains when its sustained 16-bit peaks are at least 10 % below the burst
    ones (a power-capped board: A10, T4, L4; #253), from the measured peaks only
    (:func:`kernel_agent.kernels.roofline.sustained_drops`), never from the GPU's name."""
    from kernel_agent.kernels.roofline import sustained_drops

    drops = sustained_drops(peaks)
    if not drops or peaks is None:
        return None
    rates = ", ".join(
        f"{SHORT_16BIT.get(k, k)} {rate:.0f} of the {burst:.0f} TFLOP/s burst"
        for k, (burst, rate) in drops.items()
    )
    info = peaks.get("sustained") or {}
    load = []
    if info.get("sm_mhz") and info.get("burst_sm_mhz"):
        load.append(f"SM clock {info['sm_mhz']} vs {info['burst_sm_mhz']} MHz")
    if info.get("reasons"):
        load.append(", ".join(info["reasons"]))
    if info.get("power_limit_w"):
        load.append(f"{float(info['power_limit_w']):.0f} W limit")
    after = f" after {float(info['seconds']):.0f} s of GEMMs" if info.get("seconds") else ""
    return (
        f"sustained load slows this GPU down: matmul {rates}{after}"
        + (f" ({'; '.join(load)})" if load else "")
        + ": the ceilings' floors use the sustained rate (an end-to-end run is sustained "
        "load), while a short module timing may see the burst one"
    )


# ------------------------------------------------------------------ knowledge per family


def knowledge_section(key: str | None, text: str | None = None) -> str:
    """The section of ``gpu-architectures/gpus.md`` whose heading carries ``[key]`` ("" without)."""
    if not key:
        return ""
    if text is None:
        try:
            text = KNOWLEDGE.read_text()
        except OSError:
            return ""
    sections = re.split(r"(?m)^(?=## )", text)
    for section in sections:
        head = section.split("\n", 1)[0]
        if head.startswith("## ") and f"[{key}]" in head:
            return section.replace(f" [{key}]", "", 1).strip()
    return ""


def prompt_section(toolchain_summary: str | None) -> str:
    """The ``# This GPU`` section of an agent prompt: what the toolchain summary's GPU is,
    that numbers measured on another GPU are evidence from it, and this family's section
    of ``gpu-architectures/gpus.md`` ("" when the summary names no GPU)."""
    facts = from_summary(toolchain_summary)
    fam = facts.family
    if fam is None:
        return ""
    lines = [
        f"# This GPU: {facts.label}",
        "Decide with this GPU's facts: the toolchain block (SMs, shared memory per block, L2, "
        "measured peaks and instruction rates, the precisions it cannot run) and the section "
        "below. Numbers the skills give for another GPU (most were measured on an RTX 5070 "
        "Ti, sm_120) are evidence from that GPU: measure here before relying on them. Every "
        "family, with its sources: the skill `kernel-agent:gpu-architectures` "
        f"(`{KNOWLEDGE}`).",
    ]
    if section := knowledge_section(fam.key):
        lines += ["", section]
    return "\n".join(lines) + "\n"
