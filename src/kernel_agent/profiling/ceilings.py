"""Ceilings: how far each module class can go at its observed shapes, per precision.

The profile gives the planner each class's share of the run; this adds what a
perfect kernel could reach there, so targets can be ranked by ceiling × share.
``analyze`` writes ``profile/ceilings.json`` + ``ceilings.md`` and appends the table
to ``profile/summary.md``.

One row per module class at one instance group (qualname with layer indices folded)
and phase, from ``profile.json`` ``classes[].work``
(:meth:`~kernel_agent.profiling.profiler.ModuleTimer.class_stats`); sibling groups
of a leaf class (``q_proj``, ``k_proj``, ... of one attention) share a row. Per run:

* **work**: FLOPs (per dtype) and weights of the ``nn.Linear`` and convolution calls
  inside the row's calls (``2 × rows × in × out``; each weight read once per call),
  plus each call's first input and output (``io_bytes``). ``M`` = FLOPs / (2 × weight
  elements), the rows per weight read: a bf16 GEMM turns compute bound near
  ``M = peak FLOP/s / DRAM bandwidth``.
* **floor** per precision: ``max(FLOPs / peak, (weight + io bytes) / DRAM bandwidth,
  calls × launch floor)``, peaks measured on this GPU (:mod:`kernel_agent.kernels.roofline`):

  - ``exact``: as profiled, each dtype at its own peak;
  - ``fp8_weights``: one byte per weight (e4m3; per-channel scales neglected), math as profiled;
  - ``w8a8``: FP8 weights, all FLOPs at the FP8 tensor-core peak;
  - ``fp4_weights``: NVFP4 weights, 4.5 bits each (e2m1 + one e4m3 scale per 16), math
    as profiled;
  - ``w4a4``: NVFP4 weights, all FLOPs at the NVFP4 tensor-core peak.

  A precision whose peak was not measured is unknown: no ratio to bf16 is assumed.
* **bound**: the term that sets the exact floor (``compute``, ``memory`` or ``launch``).
* **now**: hooked inclusive ms scaled to the unhooked run (× baseline / hooked wall
  ms); **saves** = now − floor. Rows rank by the exact one; a row already below its exact
  floor (an optimised model that runs it at a lower precision) by its best lower-precision
  one (``saves_best``).
* **end to end**, per precision: the run with every row at its floor, nested rows
  counted once (Amdahl over the non-overlapping set, :func:`projection.project`).

Calls whose insides the hooks did not see (an optimised model in an improve round: a
compiled module, a CUDA-graph replay, a replaced kernel) take the work of the same call
in the unmodified model (``reference_calls``, ‡; :class:`~kernel_agent.profiling.profiler.
WorkReference`, the re-profile's hooked pass before the round's items are applied): the
work is a property of the math, not of its implementation. Without one, a compiled or
graph-replaying call's work is unknown (``unknown_calls``): its row has no floors (``?``)
and stays at its time in the end-to-end line, never "0 FLOP, memory bound". A call that
reads none of its module's weights (``F.linear`` on a child's weight) is estimated from
its module's ``nn.Linear`` weights at its input's rows (``estimated_calls``, †).

Approximations: attention scores, KV-cache reads and element-wise math are not
counted (the floors of attention-heavy rows are too low); weights stream from DRAM
on every call (no L2 reuse); fp32 convolutions are held to the fp32 peak (cuDNN may
use TF32); ‡ rows have the math and dtypes of the unmodified model (a transform that
merges or trims layers, or lowers the precision, makes them approximate).

The improve scheduler reads the table of the newest (re-profiled) run for its expected
gains (issue #122): the rows that hold a target's instance groups (:func:`holding`) and
their floor at the target's precision (:func:`floor_ms`, :data:`TARGET_PRECISIONS`).
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernel_agent import projection
from kernel_agent.kernels.roofline import FP4, FP8


@dataclass(frozen=True)
class Precision:
    label: str
    weight_bytes: float | None  # per weight element (None: as profiled)
    peak: str | None  # ``peaks["tflops"]`` key every FLOP runs at (None: each dtype's own)


PRECISIONS = {
    "exact": Precision("exact", None, None),
    "fp8_weights": Precision("FP8 w", 1.0, None),
    "w8a8": Precision("W8A8", 1.0, FP8),
    "fp4_weights": Precision("FP4 w", 4.5 / 8, None),
    "w4a4": Precision("W4A4", 4.5 / 8, FP4),
}
_SHORT = {FP8: "FP8", FP4: "NVFP4"}


# ------------------------------------------------------------------ rows


def entries(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The profile's work entries, one per (class, group, phase); sibling groups of a leaf
    class merged (``model.layers.*.self_attn.{k_proj,q_proj}``)."""
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for c in profile.get("classes") or []:
        for w in c.get("work") or []:
            group, phase = str(w.get("group") or ""), str(w.get("phase") or "")
            parent, _, leaf = group.rpartition(".")
            key = (str(c["cls"]), parent if c.get("is_leaf") and parent else group, phase)
            row = merged.setdefault(
                key,
                {
                    "cls": key[0],
                    "leaves": [],
                    "phase": phase,
                    "instances": 0,
                    "calls": 0,
                    "inclusive_ms": 0.0,
                    "flops": {},
                    "weight_elems": 0,
                    "weight_bytes": 0,
                    "io_bytes": 0,
                    "estimated_calls": 0,
                    "reference_calls": 0,
                    "unknown_calls": 0,
                    "signature": w.get("signature", ""),
                },
            )
            row["leaves"].append(leaf if key[1] != group else "")
            if w.get("inclusive_ms", 0.0) > row.get("_top_ms", -1.0):
                row["signature"], row["_top_ms"] = w.get("signature", ""), w.get("inclusive_ms", 0)
            for name in ("instances", "calls", "weight_elems", "weight_bytes", "io_bytes"):
                row[name] += int(w.get(name) or 0)
            for name in ("estimated_calls", "reference_calls", "unknown_calls"):
                row[name] += int(w.get(name) or 0)
            row["inclusive_ms"] += float(w.get("inclusive_ms") or 0.0)
            for dtype, n in (w.get("flops") or {}).items():
                row["flops"][dtype] = row["flops"].get(dtype, 0) + int(n)
    out = []
    for (_cls, base, _phase), row in merged.items():
        leaves = sorted(set(row.pop("leaves")) - {""})
        row.pop("_top_ms", None)
        row["group"] = (
            f"{base}.{leaves[0]}"
            if len(leaves) == 1
            else f"{base}.{{{','.join(leaves)}}}"
            if leaves
            else base
        )
        out.append(row)
    return out


def _peak(tflops: Mapping[str, float], dtype: str) -> float | None:
    """Matmul peak of ``dtype``; bf16's for a dtype without its own (e.g. integers)."""
    return tflops.get(dtype) or tflops.get("bfloat16")


def floor(
    row: Mapping[str, Any], precision: Precision, peaks: Mapping[str, Any]
) -> dict[str, Any] | None:
    """``{"ms", "bound", "compute_ms", "memory_ms", "launch_ms"}`` of one row at one
    precision; None when a peak it needs was not measured."""
    tflops = {k: float(v) for k, v in (peaks.get("tflops") or {}).items() if v}
    flops: dict[str, int] = row["flops"]
    if precision.peak is not None:
        peak = tflops.get(precision.peak)
        if not peak and any(flops.values()):
            return None
        compute = sum(flops.values()) / peak / 1e9 if peak else 0.0
    else:
        compute = 0.0
        for dtype, n in flops.items():
            if n and (peak := _peak(tflops, dtype)) is None:
                return None
            compute += n / peak / 1e9 if n and peak else 0.0
    weights = (
        row["weight_bytes"]
        if precision.weight_bytes is None
        else row["weight_elems"] * precision.weight_bytes
    )
    memory = (weights + row["io_bytes"]) / float(peaks["dram_gbps"]) / 1e6
    launch = row["calls"] * float(peaks.get("launch_floor_us") or 0.0) / 1000
    ms = max(compute, memory, launch)
    bound = (
        "launch" if launch > max(compute, memory) else "compute" if compute >= memory else "memory"
    )
    return {
        "ms": ms,
        "bound": bound,
        "compute_ms": compute,
        "memory_ms": memory,
        "launch_ms": launch,
    }


def _sig(value: float, digits: int = 4) -> float:
    return float(f"{value:.{digits}g}")


def build(
    profile: Mapping[str, Any],
    peaks: Mapping[str, Any] | None,
    baseline_ms: float,
    *,
    per: str = "per run",
) -> dict[str, Any]:
    """The ceilings table of a profile (``baseline_ms``: the profiled window, unhooked)."""
    hooked = float(profile.get("hooked_wall_ms") or 0.0)
    scale = baseline_ms / hooked if hooked > 0 and baseline_ms > 0 else 1.0
    usable = bool(peaks and peaks.get("dram_gbps"))
    tflops = (peaks or {}).get("tflops") or {}
    rows = []
    for row in entries(profile):
        known = not row["unknown_calls"]  # else a call hid work the unmodified model lacks
        total = sum(row["flops"].values())
        now = row["inclusive_ms"] * scale
        row.update(
            target=f"{row['cls']}@{row['group']}",
            now_ms=_sig(now),
            share=round(now / baseline_ms, 4) if baseline_ms > 0 else 0.0,
            work_known=known,
            flops_total=total if known else None,
            m=round(total / (2 * row["weight_elems"]), 1)
            if known and row["weight_elems"]
            else None,
            floors={},
            saves_ms={},
        )
        for name, precision in PRECISIONS.items():
            f = floor(row, precision, peaks) if usable and known and peaks is not None else None
            row["floors"][name] = _sig(f["ms"]) if f else None
            row["saves_ms"][name] = _sig(max(now - f["ms"], 0.0)) if f else None
            if name == "exact" and f:
                row["bound"] = f["bound"]
        exact = row["floors"]["exact"]
        lower = {k: v for k, v in row["saves_ms"].items() if k != "exact" and v}
        if exact is not None and now < exact and lower:
            best = max(lower, key=lambda k: lower[k])
            row["saves_best"] = {"precision": best, "ms": lower[best]}
        rows.append(row)
    rows.sort(key=lambda r: (-_rank_ms(r), -r["now_ms"], r["target"]))
    precisions = {}
    for name, p in PRECISIONS.items():
        info: dict[str, Any] = {"label": p.label, "weight_bytes": p.weight_bytes}
        if p.peak is not None:
            info["peak"] = p.peak
            info["peak_tflops"] = tflops.get(p.peak)
            why = ((peaks or {}).get("tflops_unavailable") or {}).get(p.peak)
            if info["peak_tflops"] is None:
                info["unknown"] = why or "peak not measured (`kernel-agent doctor` measures it)"
        precisions[name] = info
    table = {
        "baseline_ms": baseline_ms,
        "per": per,
        "hooked_ms": hooked or None,
        "peaks": None
        if not usable
        else {
            k: peaks[k]
            for k in ("gpu", "dram_gbps", "launch_floor_us", "tflops", "tflops_unavailable")
            if peaks is not None and k in peaks
        },
        "precisions": precisions,
        "rows": rows,
        "e2e": {name: _e2e(rows, name, baseline_ms) for name in PRECISIONS} if usable else {},
        "unknown_work": [r["target"] for r in rows if not r["work_known"]],
    }
    return table


def _rank_ms(row: Mapping[str, Any]) -> float:
    """What reaching the floor saves: the exact one, else the best lower-precision one."""
    best = row.get("saves_best")
    return float(best["ms"] if best else row["saves_ms"]["exact"] or 0.0)


def _e2e(rows: list[dict[str, Any]], precision: str, baseline_ms: float) -> dict[str, Any] | None:
    """The run with every row at its ``precision`` floor, nested rows counted once; a row
    whose work is unknown stays at its time."""
    savings: dict[str, float] = {}
    patterns: dict[str, str] = {}
    for row in rows:
        if not row["work_known"]:
            continue
        saved = row["saves_ms"][precision]
        if saved is None:
            return None
        # One node per class and group: its phases are different calls of the same
        # instances, and a call's children may be in another phase.
        savings[row["target"]] = savings.get(row["target"], 0.0) + saved
        patterns[row["target"]] = row["group"]
    tree = projection.of_groups(projection.Group(t, p, 1.0) for t, p in patterns.items())
    proj = projection.project(tree, savings, baseline_ms)
    floor_ms = proj.projected_ms
    return {
        "floor_ms": _sig(floor_ms),
        "speedup": round(baseline_ms / floor_ms, 2) if floor_ms > 0 else None,
        "counted": proj.used[:8],
    }


# ------------------------------------------------------------------ targets (the scheduler)

#: A target's ``precision`` (``spec.json``, ``kernels.compare.PRECISIONS``) → the precision
#: of its floor. ``reduced`` (another numerics-changing kernel): bf16 math and weights.
TARGET_PRECISIONS = {
    "exact": PRECISIONS["exact"],
    "fp8_weights": PRECISIONS["fp8_weights"],
    "fp8_w8a8": PRECISIONS["w8a8"],
    "fp4_weights": PRECISIONS["fp4_weights"],
    "reduced": Precision("bf16", 2.0, "bfloat16"),
}
_SIBLINGS = re.compile(r"(.*)\.\{([^{}]*)\}")


def target_precision(precision: str | None) -> Precision:
    """The floor's precision of a target's ``precision`` (none or unknown: ``exact``)."""
    return TARGET_PRECISIONS.get(str(precision or "exact"), PRECISIONS["exact"])


def patterns(row: Mapping[str, Any]) -> list[str]:
    """The instance groups a row times: ``a.{b,c}`` (sibling leaves that share a row) →
    ``a.b``, ``a.c``."""
    group = str(row.get("group") or "")
    found = _SIBLINGS.fullmatch(group)
    return [f"{found[1]}.{leaf}" for leaf in found[2].split(",")] if found else [group]


def holding(
    table: Mapping[str, Any], pattern: str, cls: str | None = None, phase: str | None = None
) -> tuple[list[dict[str, Any]], bool]:
    """The rows of ``table`` that time the instance group ``pattern`` (a qualname with the
    layer indices folded), and whether they hold more than it (``inside``).

    Its own rows: the module at that qualname (of class ``cls`` when one of them is; a
    transform may have wrapped it, e.g. ``_TF32Scope`` around VoxCPM2's VAE decoder), in
    ``phase`` (every phase without one). Else, when its calls are hidden inside a compiled
    or CUDA-graph-replayed parent (the optimised model of an improve round: the LocDiT
    layers inside the graph of ``UnifiedCFM`` ``model.feat_decoder``), the rows of the
    innermost group that encloses it, in every phase (a parent's call may be in another
    phase than its children's: VoxCPM2's LocEnc decode calls run their layers at prefill
    shapes): ``inside``. ``([], False)``: no row holds it, or it has rows only in other
    phases (its calls in ``phase`` ran outside any module call, e.g. a graph-replayed
    decode step: no row says what they take)."""
    rows = list(table.get("rows") or [])
    own = [r for r in rows if pattern in patterns(r)]
    if own:
        own = [r for r in own if r.get("cls") == cls] or own
        return [r for r in own if phase is None or r.get("phase") == phase], False
    outer = [(g, r) for r in rows for g in patterns(r) if g and pattern.startswith(g + ".")]
    if not outer:
        return [], False
    deepest = max(len(g) for g, _ in outer)
    held = {id(r): r for g, r in outer if len(g) == deepest}
    return list(held.values()), True


def floor_ms(
    rows: Iterable[Mapping[str, Any]], precision: Precision, peaks: Mapping[str, Any] | None
) -> float | None:
    """Σ floor ms of ``rows`` at ``precision`` (the table's ``peaks``); None when a row's
    work is unknown or a peak it needs was not measured."""
    if not peaks or not peaks.get("dram_gbps"):
        return None
    total = 0.0
    for row in rows:
        f = floor(row, precision, peaks) if row.get("work_known", True) else None
        if f is None:
            return None
        total += f["ms"]
    return total


# ------------------------------------------------------------------ markdown


def _ms(value: float | None) -> str:
    if value is None:
        return "?"
    if value >= 100:
        return f"{value:,.0f}"
    return f"{value:.3g}"


def _saves(row: Mapping[str, Any], precisions: Mapping[str, Any]) -> str:
    """The *saves ms* cell: ``0 (W8A8 424)`` for a row already below its exact floor."""
    best = row.get("saves_best")
    text = _ms(row["saves_ms"]["exact"])
    return f"{text} ({precisions[best['precision']]['label']} {_ms(best['ms'])})" if best else text


def markdown(table: Mapping[str, Any], *, top: int = 30, min_share: float = 0.01) -> str:
    """``## Ceilings`` section of ``profile/summary.md`` ("" without work in the profile):
    the rows with at least ``min_share`` of the run, by what reaching the floor saves."""
    if not table.get("rows"):  # a profile without work entries (made before #90)
        return ""
    peaks = table.get("peaks")
    per = table.get("per", "per run")
    precisions = table["precisions"]
    shown = [r for r in table["rows"] if r["share"] >= min_share][:top]
    lines = ["", "## Ceilings: floors at the observed shapes", ""]
    if peaks is None:
        lines += [
            "GPU peaks not measured yet (`kernel-agent doctor` measures them): the work of "
            "each class at its observed shapes, without floors.",
            "",
        ]
    else:
        ridge = (
            float((peaks.get("tflops") or {}).get("bfloat16") or 0.0)
            * 1000
            / float(peaks["dram_gbps"])
        )
        peak_text = []
        for name in ("w8a8", "w4a4"):
            p = precisions[name]
            peak_text.append(
                f"{p['label']} at {_SHORT.get(p['peak'], p['peak'])} "
                + (f"{p['peak_tflops']:.0f} TFLOP/s" if p.get("peak_tflops") else "unknown")
            )
        lines += [
            "What each module class could reach if its kernels ran at this GPU's roofline, "
            "at the shapes and call counts of this profile (every row: `profile/ceilings.json`). "
            "Work = the FLOPs and weights of the `nn.Linear` / convolution calls inside each "
            "call, plus its first input and output; *M* = FLOPs / (2 × weight elements), the "
            f"rows per weight read (a bf16 GEMM turns compute bound near M ≈ {ridge:.0f}). "
            f"Floor ({per}, ms) = max(FLOPs / peak, (weight + I/O bytes) / "
            f"{float(peaks['dram_gbps']):.0f} GB/s, calls × "
            f"{float(peaks.get('launch_floor_us') or 0):.1f} us launch floor) per precision: "
            "exact (as profiled), FP8 w (1 byte per weight, bf16 math), FP4 w (NVFP4, 4.5 "
            f"bits per weight, bf16 math), {', '.join(peak_text)}. *now* = the hooked time "
            "scaled to the unhooked run (approximate: hooks inflate many small calls more than "
            "a few large ones); *saves* = now − exact floor (ceiling × share; 0 when the floor "
            "is above *now*, then the best saving at a lower precision in brackets: the row "
            "already runs below its exact floor).",
            "",
        ]
    lines += [
        "| target | phase | inst | calls | M | now ms | share | TFLOP | weights GB | bound "
        "| exact | FP8 w | W8A8 | FP4 w | W4A4 | saves ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in shown:
        marks = {"†": r["estimated_calls"], "‡": r.get("reference_calls")}
        mark = "".join(f" {m}" for m, n in marks.items() if n)
        f = r["floors"]
        known = r.get("work_known", True)
        m = "" if r["m"] is None else f"{r['m']:.0f}"
        tflop = f"{r['flops_total'] / 1e12:.3g}" if known else "?"
        weights = f"{r['weight_bytes'] / 1e9:.3g}" if known else "?"
        lines.append(
            f"| `{r['cls']}` `{r['group']}`{mark} | {r['phase']} | {r['instances']} | "
            f"{r['calls']} | {m if known else '?'} | {_ms(r['now_ms'])} | "
            f"{r['share']:.1%} | {tflop} | {weights} | "
            f"{r.get('bound', '?')} | {_ms(f['exact'])} | {_ms(f['fp8_weights'])} | "
            f"{_ms(f['w8a8'])} | {_ms(f['fp4_weights'])} | {_ms(f['w4a4'])} | "
            f"{_saves(r, precisions)} |"
        )
    e2e = table.get("e2e") or {}
    if e2e:
        parts = []
        for name, p in precisions.items():
            e = e2e.get(name)
            if e is None:
                parts.append(f"{p['label']} unknown ({p.get('unknown', 'no peak')})"[:120])
            else:
                parts.append(f"{p['label']} ≥ {_ms(e['floor_ms'])} ms ({e['speedup']}x)")
        unknown = len(table.get("unknown_work") or [])
        lines += [
            "",
            f"**End to end** ({per}, baseline {_ms(table['baseline_ms'])} ms; every class at its "
            "floor, nested classes counted once, the time outside them unchanged"
            + (f", the {unknown} rows with unknown work (`?`) at their time" if unknown else "")
            + "): "
            + "; ".join(parts)
            + ".",
        ]
    lines += [
        "",
        "Approximate: attention scores, KV-cache reads and element-wise math are not counted "
        "(floors of attention-heavy rows are low); weights stream from DRAM on every call; "
        "fp32 convolutions are held to the fp32 peak (cuDNN may use TF32)."
        + (
            " ‡: the hooks did not see inside some calls (compiled, CUDA graph or replaced "
            "kernel); their work is that of the same calls in the unmodified model, with its "
            "math and dtypes (approximate where a transform merged or trimmed layers; *now* "
            "can be below the exact floor where the model already runs at a lower precision: "
            "compare it with the FP8 / FP4 floors)."
            if any(r.get("reference_calls") for r in shown)
            else ""
        )
        + (
            " ?: work unknown (hidden from the hooks, and the unmodified model made no such "
            "call): no floors, and the end-to-end line keeps its time."
            if any(not r.get("work_known", True) for r in shown)
            else ""
        )
        + (
            " †: partly estimated from the module's `nn.Linear` weights at its input's rows "
            "(its calls read none of them: a replaced kernel or `F.linear` on a child's weight)."
            if any(r["estimated_calls"] for r in shown)
            else ""
        ),
    ]
    return "\n".join(lines) + "\n"


def write(
    profile_dir: Path,
    profile: Mapping[str, Any],
    peaks: Mapping[str, Any] | None,
    baseline_ms: float,
    *,
    per: str = "per run",
) -> dict[str, Any]:
    """Build the table and write ``ceilings.json`` + ``ceilings.md`` into ``profile_dir``.
    Never raises on a malformed profile: the table then only holds the error."""
    from kernel_agent.workspace import write_json

    try:
        table = build(profile, peaks, baseline_ms, per=per)
    except Exception as exc:  # the ceilings never break an analyze
        table = {"error": f"{type(exc).__name__}: {exc}"[:300], "rows": []}
    write_json(profile_dir / "ceilings.json", table)
    (profile_dir / "ceilings.md").write_text(markdown(table).lstrip("\n"))
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ceilings table of a profile.json.")
    parser.add_argument("profile", type=Path, help="profile/profile.json of a run")
    parser.add_argument("--baseline-ms", type=float, required=True, help="unhooked window ms")
    parser.add_argument("--per", default="per run")
    parser.add_argument("--peaks", type=Path, default=None, help="peaks JSON (default: cached)")
    parser.add_argument("--json", action="store_true", help="print the JSON, not the markdown")
    ns = parser.parse_args(argv)
    if ns.peaks is not None:
        peaks = json.loads(ns.peaks.read_text())
    else:
        from kernel_agent.kernels.roofline import current_peaks

        peaks = current_peaks()
    table = build(json.loads(ns.profile.read_text()), peaks, ns.baseline_ms, per=ns.per)
    print(json.dumps(table, indent=1) if ns.json else markdown(table, top=40))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
