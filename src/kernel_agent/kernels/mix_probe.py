"""The per-layer precision probe of a W4A4 target (``precision: fp4_w4a4``, issue #233).

W4A4 moves a GEMM's output about 1.4x as far as FP4 weights alone and can shrink it by up to
~1 % per GEMM; through a block of several GEMMs that compounds (the VoxCPM2 LocDiT layer at
M = 352: relative L2 0.080, norm -3.4 %, the MLP's gate / up most). The usual remedy is to
keep the most sensitive layers at 8 bits inside the target. This module decides which, on
the target's own capture, without an agent:

* **Ranking** (:func:`kernel_agent.kernels.quant.fp4_w4a4_sensitivity`): every ``nn.Linear``
  of the reference module alone in W4A4 and in 8-bit on the inputs the captured cases feed
  it, the most sensitive first. Layers that read the same activations (q / k / v, gate /
  up: one quantised activation, usually one merged GEMM) form a **group**; groups keep the
  order of their most sensitive layer.
* **Ladder**: the reference module run on every captured case (its module state restored per
  case, as the evaluator does) with the reference math patched into its layers: W4A4
  (:func:`~kernel_agent.kernels.quant.fp4_w4a4_linear`) everywhere, then with the top 1, 2,
  ... groups in the GPU's 8-bit compute class (:func:`eight_bit_class`: FP8 W8A8, else INT8
  W8A8, else bf16), each rung compared with the captured outputs, side effects and state in
  the capture's tolerance tier (``near-lossless-fp4a`` / ``relaxed-fp4a``: the evaluator's
  first correctness stage). ``passes_at``: the fewest groups at 8 bits with which the
  reference math passes (0: W4A4 everywhere; None: not even with every group).

The worker runs :func:`probe_capture` when it captures a W4A4 target and stores the result in
``spec.json`` → ``capture`` → ``w4a4_probe``; :mod:`kernel_agent.demotion` turns it into the
target's precision mix and moves it on when a gate fails. The ladder is the reference math's
verdict, not a candidate's: a kernel adds its own error on top (a per-call outer scale, a
quantiser fused into a producer), so a later gate failure demotes the next group.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import nn

from kernel_agent.kernels import quant

#: Version of the probe's record (``spec.json`` → ``capture`` → ``w4a4_probe``).
VERSION = 1
#: Rungs of the ladder measured past W4A4 everywhere (groups moved to 8 bits): one per group
#: up to this many; a target with more groups gets its last rung with every group at 8 bits.
MAX_RUNGS = 12
#: Calls per layer the ranking keeps (their rows together).
CALLS = 4


def eight_bit_class(capability: tuple[int, ...] | None) -> str:
    """The 8-bit compute class a W4A4 target's sensitive layers move to on a GPU of
    ``capability``: ``fp8_w8a8`` where FP8 tensor cores exist (every GPU with FP4 ones, and an
    unknown GPU), else ``int8_w8a8`` (IMMA, sm_75+), else ``exact`` (bf16)."""
    from kernel_agent.gpu_arch import precision_unsupported

    for precision in ("fp8_w8a8", "int8_w8a8"):
        if precision_unsupported(precision, capability) is None:
            return precision
    return "exact"


def groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The groups of a ranking (layers with the same ``input_group``), the most sensitive
    first: ``layers`` (most sensitive first), ``w4a4_rel_l2`` and ``fp8_rel_l2`` (their
    largest), ``flop_share`` (their sum)."""
    found: dict[str, dict[str, Any]] = {}
    for row in rows:  # most sensitive first: a group's first row sets its place
        g = found.setdefault(
            str(row.get("input_group") or row["name"]),
            {"layers": [], "w4a4_rel_l2": 0.0, "fp8_rel_l2": 0.0, "flop_share": 0.0},
        )
        g["layers"].append(row["name"])
        g["w4a4_rel_l2"] = max(g["w4a4_rel_l2"], float(row["w4a4_rel_l2"]))
        g["fp8_rel_l2"] = max(g["fp8_rel_l2"], float(row["fp8_rel_l2"]))
        g["flop_share"] = round(g["flop_share"] + float(row["flop_share"]), 4)
    return list(found.values())


def _linear_fn(layer: nn.Linear, precision: str, fmt: str, granularity: str) -> Callable:
    """The reference math of ``layer`` at ``precision`` (``fp4_w4a4``, ``fp8_w8a8``,
    ``int8_w8a8``; ``exact``: its own forward), its weight quantised once."""
    bias = layer.bias
    if precision == "fp4_w4a4":
        codes, scales, ts = quant.quantize_fp4(layer.weight, fmt)
        return lambda x: quant.fp4_w4a4_linear(
            x, codes, scales, ts, bias, fmt=fmt, granularity=granularity
        )
    if precision == "fp8_w8a8":
        q, s = quant.quantize_fp8(layer.weight)
        return lambda x: quant.fp8_w8a8_linear(x, q, s, bias)
    if precision == "int8_w8a8":
        q8, s8 = quant.quantize_int8(layer.weight)
        return lambda x: quant.int8_w8a8_linear(x, q8, s8, bias)
    return type(layer).forward.__get__(layer)


@contextmanager
def _patched(layers: dict[str, nn.Linear], fns: dict[str, Callable]) -> Iterator[None]:
    """``layers`` with their ``forward`` replaced by ``fns`` (instance attributes) inside."""
    try:
        for name, fn in fns.items():
            layers[name].forward = fn  # type: ignore[method-assign]
        yield
    finally:
        for name in fns:
            layers[name].__dict__.pop("forward", None)


def _verdict(capture: dict[str, Any], reference: nn.Module, replay: Any, tier: str) -> dict:
    """``ok``, ``min_cosine``, ``max_rel_l2`` and the first ``failed`` check of ``reference``
    (patched) on every case of ``capture``, as the evaluator's captured-input stage compares
    them (outputs, in-place argument updates, module state) in ``tier``."""
    from kernel_agent.kernels.compare import compare_side_effects, compare_structures
    from kernel_agent.workloads.base import synchronize

    cosines, rel = [], []
    failed: str | None = None
    for i, case in enumerate(capture["cases"]):
        args, kwargs = copy.deepcopy(case["args"]), copy.deepcopy(case["kwargs"])
        try:
            with torch.inference_mode():
                out = replay.call(case, reference)(*args, **kwargs)
            synchronize()
        except Exception as exc:  # the reference math does not take this case
            failed = failed or f"case {i}: {type(exc).__name__}: {exc}"[:300]
            continue
        inputs = (case["args"], case["kwargs"])
        checks = compare_structures(case["output"], out, "output", inputs=inputs, tier=tier)
        checks += compare_side_effects(case["args"], case["post_args"], args, "args", tier=tier)
        checks += compare_side_effects(
            case["kwargs"], case["post_kwargs"], kwargs, "kwargs", tier=tier
        )
        checks += replay.check(case, reference, tier=tier)
        cosines += [float(c["cosine"]) for c in checks if "cosine" in c]
        rel += [float(c["rel_l2"]) for c in checks if "rel_l2" in c]
        bad = next((c for c in checks if not c.get("ok")), None)
        if bad is not None and failed is None:
            failed = f"case {i} {bad.get('name')}: {bad.get('error') or 'mismatch'}"[:300]
    return {
        "ok": failed is None,
        "min_cosine": round(min(cosines), 6) if cosines else None,
        "max_rel_l2": float(f"{max(rel):.4g}") if rel else None,
        **({"failed": failed} if failed else {}),
    }


def probe(
    capture: dict[str, Any],
    *,
    eight_bit: str = "fp8_w8a8",
    fmt: str = "nvfp4",
    granularity: str = "token",
    calls: int = CALLS,
) -> dict[str, Any]:
    """The probe of a loaded W4A4 ``capture`` (:func:`kernel_agent.profiling.capture.
    load_capture`): ``layers`` (the ranking), ``groups``, ``ladder`` (per rung: ``demoted``
    groups, their ``layers``, the verdict), ``passes_at``, and what it ran with."""
    from kernel_agent.kernels.compare import tier_of
    from kernel_agent.profiling.state import Replay

    t0 = time.perf_counter()
    reference = capture["module"].eval()
    replay = Replay(capture, reference)
    tier = tier_of(capture)
    cases = capture["cases"]

    def run() -> None:
        for case in cases:
            args, kwargs = copy.deepcopy(case["args"]), copy.deepcopy(case["kwargs"])
            with torch.inference_mode():
                replay.call(case, reference)(*args, **kwargs)

    rows = quant.fp4_w4a4_sensitivity(reference, run, fmt=fmt, granularity=granularity, calls=calls)
    ranked = groups(rows)
    layers = {n: m for n, m in reference.named_modules() if isinstance(m, nn.Linear)}
    w4a4 = {r["name"]: _linear_fn(layers[r["name"]], "fp4_w4a4", fmt, granularity) for r in rows}
    eight = {r["name"]: _linear_fn(layers[r["name"]], eight_bit, fmt, granularity) for r in rows}
    rungs = list(range(min(len(ranked), MAX_RUNGS) + 1))
    if len(ranked) > MAX_RUNGS:
        rungs.append(len(ranked))  # every group at 8 bits
    ladder = []
    for k in rungs:
        demoted = [name for g in ranked[:k] for name in g["layers"]]
        fns = {n: (eight[n] if n in demoted else w4a4[n]) for n in w4a4}
        with _patched(layers, fns):
            verdict = _verdict(capture, reference, replay, tier)
        ladder.append({"demoted": k, "layers": demoted, **verdict})
    passing = [rung["demoted"] for rung in ladder if rung["ok"]]
    return {
        "version": VERSION,
        "format": fmt,
        "granularity": granularity,
        "eight_bit": eight_bit,
        "tier": tier,
        "layers": rows,
        "groups": ranked,
        "ladder": ladder,
        "passes_at": passing[0] if passing else None,
        "seconds": round(time.perf_counter() - t0, 2),
        **({} if rows else {"note": "no nn.Linear whose K is a multiple of 16: nothing to rank"}),
    }


def probe_capture(
    path: Path, *, device: str | None = None, capability: tuple[int, ...] | None = None
) -> dict[str, Any]:
    """:func:`probe` of the capture file ``path`` with the 8-bit class of a GPU of
    ``capability`` (None: this process's GPU, else an unknown one); ``{"error": ...}`` when
    it fails (a capture is never refused for its probe)."""
    from kernel_agent.profiling.capture import load_capture

    try:
        if capability is None and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
        capture = load_capture(path, device=device)
        return probe(capture, eight_bit=eight_bit_class(capability))
    except Exception as exc:
        return {"version": VERSION, "error": f"{type(exc).__name__}: {exc}"[:600]}
