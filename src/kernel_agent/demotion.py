"""Per-layer demotion of W4A4 targets (issue #233): the precision mix of an ``fp4_w4a4``
target, and how it moves when a quality gate fails.

W4A4 is opt-in (``--precisions`` must name ``fp4_w4a4``), and so is everything here: only
targets planned at ``fp4_w4a4`` have a mix. Their sensitive ``nn.Linear`` layers run in the
GPU's 8-bit compute class inside the target (``fp8_w8a8``; ``int8_w8a8`` where FP8 is
absent), in the target's own tier (``near-lossless-fp4a`` / ``relaxed-fp4a``: 8-bit layers
are within it), so no new capture is needed. The evidence is the probe the worker runs when
it captures the target (:mod:`kernel_agent.kernels.mix_probe`, ``spec.json`` → ``capture``
→ ``w4a4_probe``): the layers ranked by their W4A4 error on the captured inputs, grouped by
the activations they read, and the ladder of reference-math verdicts with the top 0, 1, 2,
... groups at 8 bits.

The policy (deterministic, from records, no agent):

* **Start** (:func:`initial`): the fewest groups with which the reference math passes the
  tier on the captured cases (the ladder's ``passes_at``; every group when no rung passes).
* **Gate failures** move it on by the next group of the ranking (:func:`advance`), recorded
  in ``spec.json`` → ``w4a4_mix`` (its ``steps``: trigger, reason, the ledger row count) and
  as a ``w4a4_mix`` event:

  - the evaluator (:func:`evaluator_failure`): :data:`MODULE_FAILURES` evaluations since the
    mix was set failed the tier on the captured or redrawn inputs the way precision does
    (cosine at least :data:`PRECISION_COSINE`: a broken kernel is far below), and none passed;
  - the end-to-end gate (:func:`e2e_failure`): the integration measured one of the target's
    kernels, evaluated under the current mix, alone against the unmodified model and the
    quality checks (the perceptual gate, teacher forcing, the held-out input) rejected it.

* **Bound**: one step per group. With every group at 8 bits a further failure proposes the
  next precision mix, a pivot of the target to its 8-bit class (``<id>__fp8_w8a8``, a new
  arm in that tier, :mod:`kernel_agent.pivot`), and names the ideas left for W4A4 itself
  (:data:`NEXT_IDEAS`). Nothing here stops an arm: the W4A4 arm keeps its slices.

The engineer prompt (:func:`prompt_text`), the improve digest (:func:`digest_lines`) and the
round re-plan (:func:`label`) show the mix; ``Improver._w4a4_mix`` applies :func:`step` before
each slice of a W4A4 arm.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from kernel_agent import ledger
from kernel_agent.workspace import RunDir, read_json, write_json

PRECISION = "fp4_w4a4"
#: ``spec.json`` key of a target's mix, and of its probe in ``spec.json`` → ``capture``.
MIX = "w4a4_mix"
PROBE = "w4a4_probe"
#: The ``source`` of the pivots the policy proposes (``ledger`` events, improve.json).
SOURCE = "w4a4-demotion"
#: Evaluations failing the tier like precision does, with none correct, that count as a
#: failure of the mix (one buggy candidate is not evidence against the precision).
MODULE_FAILURES = 3
#: The cosine (worst over the failed evaluation's checks) at or above which a tier failure
#: looks like precision: W4A4's own error leaves cosines near 0.99 (the tier allows 0.96);
#: swapped nibbles, a wrong scale layout or a shifted block do not get near 0.9.
PRECISION_COSINE = 0.9
#: What remains for W4A4 when every group is at 8 bits and the gate still fails.
NEXT_IDEAS = (
    "per-token outer scales where one per call was used",
    "a Hadamard rotation of the remaining W4A4 layers (`quant.hadamard_rotate`, measure it)",
    "the opt-in correction of e2m1's norm bias (quantised activations shrink a GEMM's output by "
    "~1 % along the exact one: `unbiased=True` of `quant.quantize_fp4` / `fp4_w4a4_linear`, the "
    "producers' per-token factor in `triton_fp4_producers.py`)",
    "bf16 for the most sensitive group",
)


def applies(spec: Mapping[str, Any]) -> bool:
    """Whether ``spec`` is a W4A4 target with a usable probe."""
    from kernel_agent.precisions import of_spec

    return of_spec(spec) == PRECISION and probe_of(spec) is not None


def probe_of(spec: Mapping[str, Any]) -> dict[str, Any] | None:
    """The target's probe (``spec.json`` → ``capture`` → ``w4a4_probe``) when it ranked at
    least one group, else None."""
    found = (spec.get("capture") or {}).get(PROBE)
    if not isinstance(found, dict) or found.get("error") or not found.get("groups"):
        return None
    return found


def _rung(probe: Mapping[str, Any], k: int) -> dict[str, Any]:
    """The ladder rung with ``k`` groups at 8 bits ({} when it was not measured)."""
    return next((r for r in probe.get("ladder") or [] if r.get("demoted") == k), {})


def _numbers(rung: Mapping[str, Any]) -> str:
    """``cosine 0.9968, relative L2 0.08`` of a rung (worst over its checks)."""
    if rung.get("min_cosine") is None:
        return "no checks"
    return f"cosine {rung['min_cosine']:.4f}, relative L2 {rung['max_rel_l2']:.3g}"


def _layers(probe: Mapping[str, Any], k: int) -> list[str]:
    return [name for g in (probe.get("groups") or [])[:k] for name in g["layers"]]


def initial(spec: Mapping[str, Any]) -> dict[str, Any] | None:
    """The first mix of a W4A4 target from its probe (None without one): the ladder's
    ``passes_at`` groups at 8 bits (every group when no rung passes), with the reason."""
    probe = probe_of(spec)
    if probe is None:
        return None
    total = len(probe["groups"])
    at = probe.get("passes_at")
    k = total if at is None else min(int(at), total)
    tier, eight = probe.get("tier"), probe.get("eight_bit")
    first = _rung(probe, 0)
    if k == 0:
        reason = (
            f"the reference W4A4 math passes {tier} on the captured cases with every layer in "
            f"W4A4 ({_numbers(first)})"
        )
    elif at is not None:
        reason = (
            f"W4A4 everywhere fails {tier} on the captured cases with the reference math "
            f"({first.get('failed', _numbers(first))}); with the top {k} of {total} groups in "
            f"{eight} it passes ({_numbers(_rung(probe, k))})"
        )
    else:
        reason = (
            f"the reference math fails {tier} on the captured cases with every one of the "
            f"{total} groups in {eight} too ({_rung(probe, total).get('failed', '')}): the "
            "error is not in the GEMMs alone"
        )
    return {
        "eight_bit": eight,
        "demoted": k,
        "groups": total,
        "layers": _layers(probe, k),
        "exp": 0,
        "steps": [{"demoted": k, "trigger": "probe", "reason": reason, "exp": 0}],
    }


def mix_of(spec: Mapping[str, Any]) -> dict[str, Any] | None:
    """The target's mix: the recorded one (``spec.json`` → ``w4a4_mix``), else :func:`initial`."""
    found = spec.get(MIX)
    return dict(found) if isinstance(found, dict) and "demoted" in found else initial(spec)


def _cosine(record: Mapping[str, Any]) -> float | None:
    """The worst cosine of a failed evaluation (its cases and its failed check's failures)."""
    found = [c.get("min_cosine") for c in record.get("cases") or []]
    check = record.get("failed_check") or {}
    found += [f.get("cosine") for f in check.get("failures") or [] if isinstance(f, dict)]
    numbers = [float(c) for c in found if isinstance(c, int | float)]
    return min(numbers) if numbers else None


def _precision_like(record: Mapping[str, Any]) -> bool:
    """A tier failure on the captured or redrawn inputs (not the scale-rule guard, a crash or
    an integrity verdict) whose worst cosine is at least :data:`PRECISION_COSINE`."""
    if record.get("stage") not in ("correctness", "perturbed"):
        return False
    cosine = _cosine(record)
    return cosine is None or cosine >= PRECISION_COSINE


def _counted(records: Iterable[Mapping[str, Any]], since: int) -> list[Mapping[str, Any]]:
    """The benchmark evaluations recorded after ledger row ``since`` (no quick checks,
    duplicates or re-evaluations of an older snapshot)."""
    return [
        r
        for r in records
        if int(r.get("exp") or 0) > since
        and r.get("mode") != "quick"
        and not r.get("reevaluates")
        and r.get("ledger_status") not in (ledger.DUPLICATE, ledger.REEVALUATED)
    ]


def evaluator_failure(records: Iterable[Mapping[str, Any]], since: int) -> str | None:
    """Why the evaluator's verdicts since ledger row ``since`` count as a failure of the mix
    (None: they do not): at least :data:`MODULE_FAILURES` precision-like tier failures and no
    correct evaluation."""
    new = _counted(records, since)
    if any(r.get("correct") for r in new):
        return None
    failed = [r for r in new if _precision_like(r)]
    if len(failed) < MODULE_FAILURES:
        return None
    last = failed[-1]
    tier = last.get("tolerance_tier") or "its tier"
    return (
        f"{len(failed)} evaluations since the mix was set failed {tier} (none passed; last: "
        f"exp {last.get('exp')}, {str(last.get('error') or last.get('stage'))[:200]})"
    )


def e2e_failure(
    integration: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
    target_id: str,
    since: int,
) -> str | None:
    """Why the end-to-end gate rejected one of the target's kernels evaluated after ledger
    row ``since`` (``integration.json``: the kernel alone against the unmodified model, a
    quality verdict, not a crash or a run out of memory), or None."""
    from kernel_agent import abtest

    exp_of = {Path(str(r.get("snapshot") or "")).name: int(r.get("exp") or 0) for r in records}
    for entry in integration.get("history") or []:
        items = entry.get("items") or []
        if len(items) != 1 or (entry.get("ab") or {}).get("a_items"):
            continue
        target, sep, path = str(items[0]).partition("=")
        if not sep or target != target_id or entry.get("passed") is not False:
            continue
        if entry.get("status") != "ok" or abtest.step_out_of_memory(entry) is not None:
            continue
        snapshot = Path(path).name
        if exp_of.get(snapshot, 0) > since:
            why = str(entry.get("reason") or "the quality checks failed")
            return f"the end-to-end gate rejected `{snapshot}` alone: {why[:300]}"
    return None


def advance(
    mix: Mapping[str, Any],
    probe: Mapping[str, Any],
    *,
    trigger: str,
    reason: str,
    exp: int,
) -> dict[str, Any]:
    """The mix after a gate failure: the next group of the ranking at 8 bits; with every
    group there already, the same mix ``exhausted`` (its ``next``: the pivot and the ideas
    left, :data:`NEXT_IDEAS`)."""
    total = len(probe.get("groups") or [])
    k = int(mix.get("demoted") or 0)
    out = {**mix, "groups": total, "steps": list(mix.get("steps") or [])}
    if k < total:
        k += 1
        out.update(demoted=k, layers=_layers(probe, k), exp=exp)
        out["steps"].append({"demoted": k, "trigger": trigger, "reason": reason, "exp": exp})
        return out
    out.update(exhausted=True, exp=exp, next="; ".join(NEXT_IDEAS))
    out["steps"].append({"demoted": k, "trigger": trigger, "reason": reason, "exp": exp})
    return out


def pivot_proposal(spec: Mapping[str, Any], mix: Mapping[str, Any]) -> dict[str, Any] | None:
    """The pivot of an exhausted mix to its 8-bit class (None without one, or once it was
    tried: ``mix["pivot"]``)."""
    eight = mix.get("eight_bit")
    if not mix.get("exhausted") or mix.get("pivot") or eight in (None, "exact"):
        return None
    last = (mix.get("steps") or [{}])[-1]
    return {
        "precision": eight,
        "precision_why": (
            f"W4A4 with all {mix.get('groups')} GEMM groups of the target in {eight} still "
            f"failed the gate ({str(last.get('reason'))[:300]}); the per-layer probe's next "
            f"mix is {eight} everywhere, in its own tier"
        ),
    }


def step(
    run: RunDir,
    target_id: str,
    *,
    records: Iterable[Mapping[str, Any]],
    integration: Mapping[str, Any] | None = None,
    exp: int | None = None,
) -> dict[str, Any]:
    """Apply the policy to target ``target_id`` (its ``spec.json``) given its evaluation
    ``records`` (``results.jsonl``) and ``integration`` (``integration.json``); ``exp``: the
    ledger's row count now (None: read it). Records a new mix (``spec.json``, a ``w4a4_mix``
    event) and returns ``{"mix", "trigger", "reason"}`` and, when it proposes one, ``pivot``
    (for ``Orchestrator.pivot``); ``{}`` when nothing changed."""
    path = run.target(target_id) / "spec.json"
    spec = read_json(path, {}) or {}
    probe = probe_of(spec)
    if not applies(spec) or probe is None:
        return {}
    records = list(records)
    exp = len(ledger.rows(run)) if exp is None else exp
    recorded = isinstance(spec.get(MIX), dict)
    mix = mix_of(spec) or {}
    since = int(mix.get("exp") or 0)
    trigger, reason = "", None
    if mix.get("exhausted"):
        pass  # every group at 8 bits and the pivot tried: the arm goes on with its ideas
    elif reason := evaluator_failure(records, since):
        trigger = "evaluator"
    elif reason := e2e_failure(integration or {}, records, target_id, since):
        trigger = "e2e"
    if reason:
        mix = advance(mix, probe, trigger=trigger, reason=reason, exp=exp)
    elif recorded:
        return {}
    else:  # the probe's mix, recorded once (its step names the evidence)
        trigger, reason = "probe", mix["steps"][-1]["reason"]
    out: dict[str, Any] = {"mix": mix, "trigger": trigger, "reason": reason}
    if proposal := pivot_proposal(spec, mix):
        out["pivot"] = {"target": target_id, **proposal}
        mix["pivot"] = "proposed"
    spec[MIX] = mix
    write_json(path, spec)
    ledger.event(
        run,
        "w4a4_mix",
        target=target_id,
        demoted=mix["demoted"],
        groups=mix.get("groups"),
        layers=mix.get("layers"),
        trigger=trigger,
        reason=str(reason)[:500],
        **({"pivot": mix["eight_bit"]} if "pivot" in out else {}),
    )
    return out


def record_pivot(run: RunDir, target_id: str, outcome: Mapping[str, Any]) -> None:
    """What became of the pivot :func:`step` proposed (``Orchestrator.pivot``'s result: the
    new target, or why it was refused), in the target's mix."""
    path = run.target(target_id) / "spec.json"
    spec = read_json(path, {}) or {}
    if isinstance(spec.get(MIX), dict):
        moved = outcome.get("target")
        spec[MIX]["pivot"] = f"to {moved}" if moved else f"refused: {outcome.get('refused')}"
        write_json(path, spec)


def _name(precision: str | None) -> str:
    return {"fp8_w8a8": "FP8 W8A8", "int8_w8a8": "INT8 W8A8"}.get(str(precision), "bf16")


def _names(layers: Iterable[str], short: bool = False) -> str:
    """Layer names for a prompt (``short``: the last part, for tables); the target itself when
    it is one ``nn.Linear`` (its name is "")."""
    shown = [(str(n).rsplit(".", 1)[-1] if short else str(n)) for n in layers]
    shown = [n or "the target itself" for n in shown]
    return " / ".join(shown) if short else ", ".join(f"`{n}`" for n in shown)


def label(spec: Mapping[str, Any]) -> str:
    """``, gate_proj / up_proj in fp8_w8a8 (1 of 3 groups)``: a W4A4 target's mix for tables
    ("" when it has none or keeps every layer in W4A4)."""
    mix = mix_of(spec) if applies(spec) else None
    if not mix or not mix.get("demoted"):
        return ""
    names = _names(mix.get("layers") or [], short=True)
    return f", {names} in {mix['eight_bit']} ({mix['demoted']} of {mix['groups']} groups)"


def _groups_table(probe: Mapping[str, Any], limit: int = 6) -> list[str]:
    """The probe's groups, most sensitive first (layers, W4A4 / 8-bit relative L2, FLOPs)."""
    rows = [
        "| # | layers (same input) | W4A4 rel L2 | FP8 rel L2 | FLOP share |",
        "|---|---|---|---|---|",
    ]
    for i, g in enumerate((probe.get("groups") or [])[:limit], 1):
        names = _names(g["layers"])
        rows.append(
            f"| {i} | {names} | {g['w4a4_rel_l2']:.3g} | {g['fp8_rel_l2']:.3g} | "
            f"{g['flop_share']:.0%} |"
        )
    return rows


def prompt_text(spec: Mapping[str, Any]) -> str:
    """The engineer prompt's lines on the target's mix ("" without a probe)."""
    probe = probe_of(spec) if applies(spec) else None
    mix = mix_of(spec) if probe is not None else None
    if probe is None or not mix:
        return ""
    k, total, eight = mix["demoted"], mix["groups"], mix["eight_bit"]
    if k:
        layers = _names(mix.get("layers") or [])
        keep = (
            f"keep {layers} (the top {k} of {total} groups: layers that read the same "
            f"activations) in {_name(eight)} (`{eight}`'s numerics), every other layer W4A4"
        )
    else:
        keep = "every `nn.Linear` in W4A4"
    last = (mix.get("steps") or [{}])[-1]
    lines = [
        "",
        "Precision mix (kernel-agent's per-layer probe on this target's capture, `spec.json` "
        f"→ `{MIX}`; it moves one group to 8 bits each time a gate fails): {keep}. Why: "
        f"{last.get('reason')}.",
        "The probe's groups, most sensitive first (each layer alone on the captured inputs):",
        *_groups_table(probe),
    ]
    if mix.get("exhausted"):
        lines.append(f"Every group is at 8 bits; still to try for W4A4: {mix.get('next')}.")
    return "\n".join(lines)


def digest_lines(spec: Mapping[str, Any]) -> list[str]:
    """``## Precision mix (W4A4)`` of an improve digest ([] for other targets)."""
    probe = probe_of(spec) if applies(spec) else None
    mix = mix_of(spec) if probe is not None else None
    if probe is None or not mix:
        return []
    k, total, eight = mix["demoted"], mix["groups"], mix["eight_bit"]
    groups = probe.get("groups") or []
    now = "every layer W4A4"
    if k:
        now = f"{_names(mix.get('layers') or [])} in `{eight}`, the rest W4A4"
    lines = ["", "## Precision mix (W4A4)", f"* Now: {now} ({k} of {total} groups at 8 bits)."]
    for s in (mix.get("steps") or [])[-3:]:
        lines.append(f"* {s.get('trigger')} (exp {s.get('exp')}): {s.get('reason')}")
    if k < total:
        nxt = _names(groups[k]["layers"])
        lines.append(
            f"* When the evaluator ({MODULE_FAILURES} evaluations failing the tier near its "
            "bounds, none passing) or the end-to-end gate rejects this mix, kernel-agent "
            f"moves {nxt} to `{eight}` next."
        )
    else:
        pivot = mix.get("pivot")
        lines.append(
            "* Every group is at 8 bits"
            + (f"; the pivot to `{eight}`: {pivot}" if pivot else "")
            + f". Still to try for W4A4: {mix.get('next') or '; '.join(NEXT_IDEAS)}."
        )
    return lines
