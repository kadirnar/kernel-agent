"""Short directives from a candidate's profile: what to change, each with its numbers (#230).

Guidance in words beats raw metrics (KernelPro, arXiv 2606.26453: one directive, "TC
utilization: 0.7 %", took a task from 4.11x to 11.49x; KEET, arXiv 2605.04467), and a few
chosen numbers beat the full set (CudaForge). :func:`build` reads an evaluation's profile
tables, the SASS census (:mod:`kernel_agent.kernels.sass`), ``compiler_stats``, Nsight
Compute (:mod:`kernel_agent.kernels.ncu`: per-kernel bounds, rules, stall lines) and the
per-kernel GPU time (``kernels_candidate``), with the evaluation's roofline (``bound``,
``pct_of_sol``) and the GPU's facts (the census's ``gpu``: capability and measured
instruction rates), and returns at most :data:`MAX_DIRECTIVES` directives. The rules are
deterministic and documented here; each directive carries its numbers, so the engineer,
the critic or the profile-analyst can check and overrule it.

Rules (the ``rule`` of a directive), most urgent first; within a rule, the kernels with the
largest share of GPU time first, kernels that did not run in the profiled call left out:

1. ``local_memory``: a kernel with ``LDL`` / ``STL`` in its SASS (spills, or a per-thread
   array indexed at run time) or spills in its compiler stats.
2. ``fp8_emulated``: FP8 unpacked to fp16 (``F2FP.F16.E4M3.UNPACK_B``) feeding ``HMMA``
   with no FP8 MMA: e4m3 ``mma.sync`` on a GPU without it (sm_90, sm_100).
3. ``tensor_rate``: a kernel that is not memory or launch bound whose tensor-core opcodes
   for its operands are not this GPU's full-rate ones (:func:`full_rate`): ``mma.sync``
   where wgmma (sm_90) or tcgen05 (sm_100) runs faster, plain ``QMMA.F32`` where the
   measured block-scaled ``QMMA.SF`` rate is higher (sm_12x).
4. ``no_tensor_cores``: a compute-bound evaluation whose kernel runs fp32 / fp16 math and
   no tensor-core instruction on a GPU with tensor cores.
5. ``tensor_idle``: a kernel that issues tensor-core instructions while Nsight measures
   its tensor pipe below :data:`TENSOR_IDLE_PCT` % active (not memory bound): the MMAs
   wait on their operands (its top stalls say on what).
6. ``stall_line``: ncu's source counters: the line holding the most warp-stall samples of
   its kernel (at least :data:`STALL_LINE_SHARE` %) and its dominant stall.
7. ``uncoalesced``: a line Nsight flags (uncoalesced global access, bank conflicts).
8. ``narrow_loads``: a memory-bound evaluation whose kernel loads global memory mostly in
   accesses narrower than 128 bits (static counts) without cp.async or TMA.
9. ``ncu_rule``: Nsight's rule with the largest estimated speedup.
10. ``no_async_copy``: a tensor-core kernel that stages operands through registers
    (``LDG`` + ``STS``) with neither cp.async nor TMA on a GPU that has them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from kernel_agent import gpu_arch
from kernel_agent.kernels.sass import base, matches

MAX_DIRECTIVES = 5
PER_RULE = 2  # directives of one rule at most
MAX_CHARS = 360  # one directive's text
STALL_LINE_SHARE = 20.0  # % of a kernel's stall samples on one line
TENSOR_IDLE_PCT = 30.0  # tensor pipe active below this share in a kernel issuing MMAs
#: The measured block-scaled FP8 rate (``e4m3_sf_f32``) must beat plain ``QMMA.F32``
#: (``e4m3_f32``) by this factor for ``tensor_rate`` to name it (sm_120: 412 vs 206).
SF_GAIN = 1.3
PRIORITY = {
    "local_memory": 1,
    "fp8_emulated": 1,
    "tensor_rate": 2,
    "no_tensor_cores": 2,
    "tensor_idle": 2,
    "stall_line": 3,
    "uncoalesced": 3,
    "narrow_loads": 3,
    "ncu_rule": 4,
    "no_async_copy": 4,
}
#: Operand class of each tensor-core base opcode (``sass.TENSOR_OPS``).
OPERANDS = {
    "HMMA": "16-bit",
    "HGMMA": "16-bit",
    "UTCHMMA": "16-bit",
    "QMMA": "fp8",
    "QGMMA": "fp8",
    "UTCQMMA": "fp8",
    "IMMA": "int8",
    "IGMMA": "int8",
    "UTCIMMA": "int8",
    "OMMA": "fp4",
    "UTCOMMA": "fp4",
}
#: How a kernel reaches each family's full-rate path (the ``tensor_rate`` text).
HOW = {
    "wgmma": "wgmma (HGMMA / QGMMA / IGMMA): Triton tl.dot with 4+ warps and 64+ row tiles on "
    "sm_90a, CUTLASS / CuTe sm90 GEMMs (warpgroup MMA from shared memory, TMA-fed)",
    "tcgen05": "tcgen05.mma (UTCHMMA / UTCQMMA / UTCIMMA): Triton tl.dot on sm_100a, "
    "CUTLASS / CuTe sm100 GEMMs (TMEM accumulators, TMA-fed)",
    "block_scaled": "QMMA.SF (block-scaled mma.sync kind::mxf8f6f4.block_scale: Triton "
    "tl.dot_scaled with unit ue8m0 scales (127), CuTe MmaMXF8Op); bit-identical to unit-scale "
    "e4m3 products",
}


def full_rate(facts: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Per operand class, the opcode prefixes of this GPU's full-rate tensor-core path
    (from cubins compiled on the CPU: ``tests/fixtures/sass``); empty for an unknown GPU or
    a family newer than :data:`kernel_agent.gpu_arch.FAMILIES` knows (measure there)."""
    capability = _capability(facts)
    fam = gpu_arch.family(capability)
    if fam is None or fam.key == "newer":
        return {}
    if "tcgen05" in fam.features:
        return {"16-bit": ("UTCHMMA",), "fp8": ("UTCQMMA",), "int8": ("UTCIMMA",)}
    if "wgmma" in fam.features:
        return {"16-bit": ("HGMMA",), "fp8": ("QGMMA",), "int8": ("IGMMA",)}
    out: dict[str, tuple[str, ...]] = {"16-bit": ("HMMA",), "int8": ("IMMA",)}
    if "fp8_tc" in fam.features:
        out["fp8"] = ("QMMA",)
    if "mma_block_scale" in fam.features and _sf_faster(facts) is not False:
        # e4m3 with fp16 accumulation runs at the block-scaled rate too (sm_120: 412)
        out["fp8"] = ("QMMA.SF", "QMMA.16832.F16")
        out["fp4"] = ("OMMA",)
    return out


def _capability(facts: Mapping[str, Any]) -> tuple[int, int] | None:
    cap = facts.get("capability")
    if isinstance(cap, list | tuple) and len(cap) >= 2:
        return int(cap[0]), int(cap[1])
    return gpu_arch.capability_of(facts.get("arch"))


def _sf_faster(facts: Mapping[str, Any]) -> bool | None:
    """Whether the measured block-scaled FP8 rate beats plain ``QMMA.F32`` by
    :data:`SF_GAIN` (None: not measured)."""
    rates = facts.get("mma_tflops") or {}
    sf, plain = rates.get("e4m3_sf_f32"), rates.get("e4m3_f32")
    if not sf or not plain:
        return None
    return float(sf) >= SF_GAIN * float(plain)


def _rates(facts: Mapping[str, Any]) -> str:
    rates = facts.get("mma_tflops") or {}
    sf, plain = rates.get("e4m3_sf_f32"), rates.get("e4m3_f32")
    return f": measured {float(sf):.0f} vs {float(plain):.0f} TFLOP/s" if sf and plain else ""


# ------------------------------------------------------------------ inputs


def _share(rows: Any) -> dict[str, float]:
    """Per profiled kernel name, its share of the candidate's GPU time."""
    rows = [r for r in rows or [] if isinstance(r, dict)]
    total = sum(float(r.get("us") or 0.0) for r in rows)
    return {str(r.get("kernel")): float(r.get("us") or 0.0) / total for r in rows if total}


def _kernel_share(name: str, shares: Mapping[str, float]) -> float | None:
    found = [s for k, s in shares.items() if matches(name, k)]
    return sum(found) if found else None


def _ncu_kernel(name: str, ncu: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for k in ncu.get("kernels") or []:
        if isinstance(k, dict) and matches(name, str(k.get("kernel") or "")):
            return k
    return None


def _bound(name: str, result: Mapping[str, Any], ncu: Mapping[str, Any]) -> tuple[str, str]:
    """``(bound, evidence)`` of a kernel: Nsight's class of it when profiled, else the
    evaluation's roofline bound (``unknown`` without either)."""
    if (k := _ncu_kernel(name, ncu)) is not None and k.get("bound") not in (None, "unknown"):
        tensor = (k.get("metrics") or {}).get("tensor_pipe_pct")
        extra = f", tensor pipe {float(tensor):.0f} %" if isinstance(tensor, float) else ""
        return str(k["bound"]), f"ncu: {k['bound']}{extra}"
    bound = result.get("bound")
    if not bound:
        return "unknown", ""
    pct = result.get("pct_of_sol")
    at = f" at {float(pct):.0f} % of SOL" if isinstance(pct, int | float) else ""
    return str(bound), f"{bound}-bound{at}"


def _spills(name: str, stats: Mapping[str, Any]) -> tuple[Any, Any]:
    """``(registers, spilled bytes)`` of a kernel in the compiler stats."""
    for rows in (stats.get("triton"), stats.get("nvrtc"), stats.get("cuda")):
        for row in rows or []:
            if isinstance(row, dict) and matches(name, str(row.get("kernel") or "")):
                spilled = row.get("spills") or row.get("local_bytes") or row.get("spill_stores")
                return row.get("registers"), spilled
    return None, None


# ------------------------------------------------------------------ the rules


def _d(rule: str, kernel: str | None, text: str, **evidence: Any) -> dict[str, Any]:
    text = text if len(text) <= MAX_CHARS else text[: MAX_CHARS - 3] + "..."
    out: dict[str, Any] = {"rule": rule, "text": text}
    if kernel:
        out["kernel"] = kernel
    return out | ({"evidence": evidence} if evidence else {})


def _census_rules(
    row: Mapping[str, Any],
    facts: Mapping[str, Any],
    result: Mapping[str, Any],
    ncu: Mapping[str, Any],
    stats: Mapping[str, Any],
) -> Iterable[dict[str, Any]]:
    name = str(row.get("kernel"))
    cats = row.get("categories") or {}
    tensor: Mapping[str, int] = row.get("tensor") or {}
    bound, why = _bound(name, result, ncu)
    variants = int(row.get("variants") or 1)
    fam = gpu_arch.family(_capability(facts))

    local = row.get("local") or {}
    if cats.get("local"):
        regs, spilled = _spills(name, stats)
        n_local = row.get("local_variants")
        of = f" in {n_local} of {variants} compiled variants" if n_local else ""
        regs_text = f"; registers {regs}" if regs else ""
        spill_text = f", {spilled} spilled per compiler stats" if spilled else ""
        yield _d(
            "local_memory",
            name,
            f"{name}: local memory in its SASS{of}: {local.get('LDL', 0)} LDL / "
            f"{local.get('STL', 0)} STL{regs_text}{spill_text} (spills, or a per-thread "
            "array indexed at run time): fewer live values (smaller tiles, fewer stages or "
            "accumulators), constant indices after unrolling, or more registers per thread",
            ldl=local.get("LDL", 0),
            stl=local.get("STL", 0),
            registers=regs,
        )

    unpacks = int(row.get("fp8_unpacks") or 0)
    fp8_mma = [op for op in tensor if OPERANDS.get(base(op)) == "fp8"]
    hmma = sum(n for op, n in tensor.items() if base(op) == "HMMA")
    emulated = bool(unpacks and hmma and not fp8_mma)
    if emulated:
        path = HOW["tcgen05" if fam and "tcgen05" in fam.features else "wgmma"]
        yield _d(
            "fp8_emulated",
            name,
            f"{name}: FP8 runs emulated: {unpacks} F2FP e4m3 -> fp16 unpacks feed {hmma} HMMA "
            f"and no FP8 MMA is issued (e4m3 mma.sync on {facts.get('arch') or 'this GPU'}); "
            f"this GPU's FP8 rate needs {path}",
            unpacks=unpacks,
            hmma=hmma,
        )

    wanted = full_rate(facts)
    if tensor and bound not in ("memory", "launch") and wanted:
        classes: dict[str, list[str]] = {}
        for op in tensor:
            if (cls := OPERANDS.get(base(op))) is not None:
                classes.setdefault(cls, []).append(op)
        for cls, ops in classes.items():
            want = wanted.get(cls)
            if not want or any(op.startswith(w) for op in ops for w in want):
                continue
            if emulated and cls == "16-bit":  # those HMMA are the emulation: fp8_emulated
                continue
            if cls == "fp8" and want[0] == "QMMA.SF":
                how, rates = HOW["block_scaled"], _rates(facts)
            elif fam is not None and "tcgen05" in fam.features:
                how, rates = HOW["tcgen05"], ""
            elif fam is not None and "wgmma" in fam.features:
                how, rates = HOW["wgmma"], ""
            else:
                continue
            count = sum(int(tensor[op]) for op in ops)
            yield _d(
                "tensor_rate",
                name,
                f"{name}: {why + '; ' if why else ''}issues {', '.join(ops[:2])} only "
                f"({count} in its SASS); this GPU's full-rate {cls} path is {how}{rates}",
                issued=ops,
                full_rate=list(want),
                bound=bound,
            )

    profiled = _ncu_kernel(name, ncu)
    pipe = (profiled.get("metrics") or {}).get("tensor_pipe_pct") if profiled else None
    if tensor and isinstance(pipe, float) and pipe < TENSOR_IDLE_PCT and bound != "memory":
        stalls = str((profiled or {}).get("why") or "").split("top stalls: ", 1)
        top = f"; top stalls: {stalls[1][:140]}" if len(stalls) == 2 else ""
        yield _d(
            "tensor_idle",
            name,
            f"{name}: tensor pipe {pipe:.1f} % active (ncu) though it issues "
            f"{', '.join(list(tensor)[:2])}: the MMAs wait on their operands{top}; deeper "
            "pipelining (more stages, async copies), larger tiles per warp or more warps keep "
            "the tensor cores fed",
            tensor_pipe_pct=pipe,
        )

    math = int(cats.get("fp32_math") or 0) + int(cats.get("fp16_math") or 0)
    if (
        fam is not None
        and "bf16_tc" in fam.features
        and not tensor
        and bound == "compute"
        and (math >= 32)
    ):
        yield _d(
            "no_tensor_cores",
            name,
            f"{name}: {why}; no tensor-core instruction, {cats.get('fp32_math', 0)} fp32 and "
            f"{cats.get('fp16_math', 0)} fp16 math instructions in its SASS: matmul-shaped "
            f"work (dots, GEMMs, convolutions) belongs on tensor cores ({fam.full_rate})",
            fp32_math=cats.get("fp32_math", 0),
            fp16_math=cats.get("fp16_math", 0),
        )

    widths: Mapping[str, int] = row.get("global_load_bits") or {}
    loads = sum(int(n) for n in widths.values())
    wide = sum(int(n) for b, n in widths.items() if int(b) >= 128)
    staged = cats.get("cp_async") or cats.get("tma")
    if bound == "memory" and loads >= 4 and wide * 2 < loads and not staged:
        detail = ", ".join(f"{n} x {b}-bit" for b, n in widths.items())
        yield _d(
            "narrow_loads",
            name,
            f"{name}: {why}; its global loads are {detail} (static): 16-byte vector loads "
            "(contiguous inner dimension, 16-byte aligned rows and strides; CUDA float4 / "
            "uint4, Triton contiguous blocks with tl.multiple_of) move 4x the bytes per "
            "instruction",
            load_bits=dict(widths),
        )

    async_ok = fam is not None and "cp_async" in fam.features
    ldg, sts = int(cats.get("global_load") or 0), int(cats.get("shared_store") or 0)
    if tensor and async_ok and not staged and ldg >= 2 and sts >= 1 and bound != "memory":
        yield _d(
            "no_async_copy",
            name,
            f"{name}: tensor-core kernel stages operands through registers ({ldg} LDG + "
            f"{sts} STS, no LDGSTS / UTMALDG): a multi-stage cp.async or TMA pipeline "
            "(Triton num_stages >= 2, CUTLASS / CuTe pipelines) overlaps loads with MMA",
            ldg=ldg,
            sts=sts,
        )


def _ncu_rules(ncu: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    for line in ncu.get("lines") or []:
        share = float(line.get("share_pct") or 0.0)
        if share < STALL_LINE_SHARE:
            continue
        stall = (
            f", mostly {line['stall']} ({line.get('stall_pct')} %: {line.get('why')})"
            if line.get("stall")
            else ""
        )
        yield _d(
            "stall_line",
            str(line.get("kernel")),
            f"{_short(line.get('kernel'))} {line.get('where')}: {share:.0f} % of its "
            f"warp-stall samples{stall}: `{str(line.get('source'))[:80]}`",
            samples=line.get("samples"),
        )
    for line in ncu.get("flagged") or []:
        yield _d(
            "uncoalesced",
            str(line.get("kernel")),
            f"{_short(line.get('kernel'))} {line.get('where')}: "
            f"{'; '.join(line.get('flags') or [])}: `{str(line.get('source'))[:80]}`",
        )
    for rule in ncu.get("rules") or []:
        yield _d(
            "ncu_rule",
            str(rule.get("kernel")),
            f"Nsight {rule.get('rule')} on {_short(rule.get('kernel'))} "
            f"({rule.get('speedup_type') or 'est.'} speedup {rule.get('speedup_pct')} %): "
            f"{rule.get('description')}",
            speedup_pct=rule.get("speedup_pct"),
        )


def _short(name: Any) -> str:
    text = str(name or "?")
    return text.split("(", 1)[0][:80]


def _table(tables: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = tables.get(key)
    return value if isinstance(value, dict) else {}


def build(
    tables: Mapping[str, Any], result: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    """At most :data:`MAX_DIRECTIVES` directives (module docstring) from an evaluation's
    profile ``tables`` (``sass``, ``compiler_stats``, ``ncu``, ``kernels_candidate``) and
    its ``result`` (``bound``, ``pct_of_sol``): ``[{"rule", "kernel", "text",
    "evidence"}]``, most urgent first."""
    result = result or {}
    census = _table(tables, "sass")
    ncu = _table(tables, "ncu")
    if ncu.get("status") != "ok":
        ncu = {}
    stats = _table(tables, "compiler_stats")
    facts = census.get("gpu") or {}
    shares = _share(tables.get("kernels_candidate"))
    rows = [r for r in census.get("kernels") or [] if isinstance(r, dict)]
    arch = facts.get("arch")
    if arch:  # the SASS of this GPU's architecture when a file holds several
        own = [r for r in rows if str(r.get("arch") or "").rstrip("af") == arch]
        rows = own or rows
    ran = [r for r in rows if _kernel_share(str(r.get("kernel")), shares)]
    if ran:  # kernels that did not run in the profiled call say nothing about it
        rows = ran
    found: list[tuple[int, float, int, dict[str, Any]]] = []
    for row in rows:
        share = _kernel_share(str(row.get("kernel")), shares) or 0.0
        for d in _census_rules(row, facts, result, ncu, stats):
            found.append((PRIORITY[d["rule"]], -share, len(found), d))
    for d in _ncu_rules(ncu):
        share = _kernel_share(str(d.get("kernel")), shares) or 0.0
        found.append((PRIORITY[d["rule"]], -share, len(found), d))
    found.sort(key=lambda t: t[:3])
    out: list[dict[str, Any]] = []
    per_rule: dict[str, int] = {}
    for *_, d in found:
        if per_rule.get(d["rule"], 0) >= PER_RULE:
            continue
        per_rule[d["rule"]] = per_rule.get(d["rule"], 0) + 1
        out.append(d)
        if len(out) >= MAX_DIRECTIVES:
            break
    return out


def texts(directives: Iterable[Mapping[str, Any]]) -> list[str]:
    """The directives as the lines of a profile summary."""
    return [f"[{d['rule']}] {d['text']}" for d in directives]
