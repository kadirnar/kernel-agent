"""Allowed precisions of a run (issue #131): the target precisions its kernels may use.

``--precisions exact,fp8_weights,fp8_w8a8,reduced`` (``run.json`` → ``config.precisions``)
lists the precisions (a target's ``precision`` in ``spec.json``,
:data:`kernel_agent.kernels.compare.PRECISIONS`) the run's targets may have; ``exact`` is
always one of them. Without the option, and in a run whose ``run.json`` has none (made
before it), the run's ``--quality`` decides (:func:`default`): ``exact`` allows ``exact``
only, ``near-lossless`` and ``relaxed`` (the default of new runs, #175) every reduced
precision but the opt-in ones (:data:`OPT_IN`: the
4-bit ones, :data:`FOUR_BIT`, and ``fp8_kv``), which ``--precisions`` must name
(``--precisions exact,fp8_weights,fp8_w8a8,reduced,fp4_weights``); the 8-bit classes
(``fp8_weights``, ``fp8_w8a8``, MXFP8 ``fp8_mx``, ``int8_weights``, ``int8_w8a8``) are
allowed by default. A run
continued with ``--precisions`` (``improve``, ``resume``, ``integrate``) records the new
list in its ``run.json``.

Enforced (:func:`refusal`, :func:`tier_allowed`) by the planner (its precision policy and
the plan schema offer only the allowed precisions; a target at another one is dropped),
precision pivots (``pivot.check``), the capture (a target at another one is refused, with
the reason), the library priors (an entry at another one is not evaluated), the kernels
phase and the improve scheduler (such a target gets no session, its arm stops), the
integration (its kernels are skipped; the reason is in ``integration.json`` → ``skipped``
and ``report.md``) and the ceilings table (only the allowed floors are shown and drive the
expected gains).

The GPU decides too (issue #165): a precision whose math needs tensor cores the GPU lacks
(:data:`kernel_agent.gpu_arch.PRECISION_NEEDS`: ``fp8_w8a8`` sm_89+, ``fp8_mx`` and
``fp4_w4a4`` sm_100+, ``int8_w8a8`` sm_75+) is not allowed on it (:func:`allowed` and
:func:`refusal` with the GPU's
``capability``; :func:`unsupported` says why); a run records the list its GPU allows in
``run.json``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

#: The 4-bit precisions: allowed only when ``--precisions`` names them (in no quality mode by
#: default): FP4 weights and W4A4 (``fp4_w4a4``, #233: FP4 weights and activations).
FOUR_BIT = ("fp4_weights", "fp4_w4a4")
#: Every precision allowed only when ``--precisions`` names it: the 4-bit ones and ``fp8_kv``
#: (an FP8 KV cache: it pays only where the cache is a large share of a decode step's bytes,
#: long contexts; on short caches it is slower, docs/FP8.md §5).
OPT_IN = (*FOUR_BIT, "fp8_kv")


def names() -> tuple[str, ...]:
    """Every target precision (``kernels.compare.PRECISIONS``), ``exact`` first."""
    from kernel_agent.kernels.compare import PRECISIONS

    return tuple(PRECISIONS)


def default(quality: str | None) -> tuple[str, ...]:
    """The precisions a run of ``quality`` allows without ``--precisions``: ``exact`` only,
    or (``near-lossless``, ``relaxed``) every precision but the opt-in ones (:data:`OPT_IN`)."""
    from kernel_agent.kernels.compare import EXACT_TIER, allows_reduced

    if not allows_reduced(quality):
        return (EXACT_TIER,)
    return tuple(p for p in names() if p not in OPT_IN)


def parse(raw: str | Iterable[str]) -> list[str]:
    """``--precisions`` (a comma list, or the list of ``run.json``): the known precisions in
    their canonical order, ``exact`` added. ValueError: empty, or an unknown name."""
    items = raw.split(",") if isinstance(raw, str) else list(raw)
    wanted = {str(p).strip().lower() for p in items} - {""}
    if not wanted:
        raise ValueError("--precisions needs at least one precision")
    unknown = sorted(wanted - set(names()))
    if unknown:
        raise ValueError(
            f"unknown precision {', '.join(map(repr, unknown))} (one of {', '.join(names())})"
        )
    return [p for p in names() if p in wanted or p == "exact"]


def check(quality: str | None, requested: Iterable[str] | None) -> str | None:
    """Why ``requested`` (``--precisions``) does not fit a run of ``quality``, or None: a
    reduced precision needs ``--quality near-lossless`` or ``relaxed``."""
    from kernel_agent.kernels.compare import allows_reduced

    if requested is None:
        return None
    try:
        wanted = parse(requested)
    except ValueError as exc:
        return str(exc)
    reduced = [p for p in wanted if p != "exact"]
    if reduced and not allows_reduced(quality):
        return (
            f"--precisions {','.join(reduced)} needs --quality near-lossless or relaxed "
            f"(this run: {quality or 'exact'})"
        )
    return None


def allowed(
    quality: str | None,
    precisions: Iterable[str] | None = None,
    capability: tuple[int, ...] | None = None,
) -> tuple[str, ...]:
    """The precisions a run allows: ``precisions`` (its ``--precisions``; None: the
    :func:`default` of ``quality``), never a reduced one outside ``near-lossless`` and
    ``relaxed``, nor one a GPU of ``capability`` cannot run (None: no GPU check)."""
    from kernel_agent.kernels.compare import allows_reduced

    found = default(quality)
    if precisions is not None:
        try:
            wanted = set(parse(precisions))
        except ValueError:  # an edited run.json: what the quality mode allows by default
            wanted = set(found)
        found = tuple(
            p for p in names() if p in wanted and (allows_reduced(quality) or p == "exact")
        )
    return tuple(p for p in found if unsupported(p, capability) is None)


def unsupported(precision: str | None, capability: tuple[int, ...] | None) -> str | None:
    """Why a GPU of ``capability`` cannot run target ``precision`` (None: it can, or no
    capability is known; :func:`kernel_agent.gpu_arch.precision_unsupported`)."""
    from kernel_agent.gpu_arch import precision_unsupported

    return precision_unsupported(precision, capability)


def gpu_refused(
    quality: str | None, precisions: Iterable[str] | None, capability: tuple[int, ...] | None
) -> dict[str, str]:
    """The precisions ``--precisions`` (None: the quality mode's default) would allow that a
    GPU of ``capability`` cannot run, each with the reason."""
    found = {p: unsupported(p, capability) for p in allowed(quality, precisions)}
    return {p: why for p, why in found.items() if why is not None}


def of_config(config: Mapping[str, Any] | None) -> tuple[str, ...]:
    """The precisions of a run's config (``run.json`` → ``config``)."""
    config = config or {}
    return allowed(config.get("quality"), config.get("precisions"))


def of_run(run: Any) -> tuple[str, ...]:
    """The precisions ``run`` (a ``RunDir``) allows, from its ``run.json``."""
    return of_config(run.load().get("config"))


def of_spec(spec: Mapping[str, Any]) -> str:
    """A target's precision: its spec's, else its capture's, else ``exact``."""
    return str(spec.get("precision") or (spec.get("capture") or {}).get("precision") or "exact")


def reduced(allowed: Iterable[str]) -> list[str]:
    """The reduced precisions of ``allowed`` (all but ``exact``)."""
    return [p for p in allowed if p != "exact"]


def refusal(
    precision: str | None, allowed: Iterable[str], capability: tuple[int, ...] | None = None
) -> str | None:
    """Why a target at ``precision`` (None: ``exact``) is refused in a run that allows
    ``allowed`` on a GPU of ``capability`` (None: no GPU check), or None."""
    allowed = tuple(allowed)
    name = str(precision or "exact")
    if (why := unsupported(name, capability)) is not None:
        return f"precision {name!r} cannot run on this GPU: {why}"
    if name in allowed:
        return None
    opt_in = " (4-bit precisions are opt-in)" if name in FOUR_BIT else ""
    if name in OPT_IN and not opt_in:
        opt_in = f" ({name} is opt-in: --precisions must name it)"
    return (
        f"precision {name!r} is not allowed in this run{opt_in}: --precisions {','.join(allowed)}"
    )


def tier_allowed(tier: str | None, allowed: Iterable[str]) -> bool:
    """Whether an evaluation in the tolerance ``tier`` (its ``tolerance_tier``; None: not
    recorded) is of a precision ``allowed`` holds: the exact tier, or the near-lossless or
    relaxed tier of one of them (``near-lossless-fp4``, ``relaxed-fp4``: ``fp4_weights``;
    ``near-lossless-fp4a``, ``relaxed-fp4a``: ``fp4_w4a4``)."""
    from kernel_agent.kernels.compare import EXACT_TIER, REDUCED_QUALITIES, TIERS, tier_for

    if tier is None or tier not in TIERS:
        return True
    tiers = {tier_for(quality, p) for quality in REDUCED_QUALITIES for p in allowed}
    return tier == EXACT_TIER or tier in tiers


def describe(allowed: Iterable[str], capability: tuple[int, ...] | None = None) -> str:
    """``exact, fp8_weights, fp8_w8a8, reduced (4-bit not allowed: fp4_weights)``; with a
    GPU's ``capability``, also the precisions it cannot run (``; not on sm_86: fp8_w8a8``)."""
    from kernel_agent.gpu_arch import PRECISION_NEEDS, arch_of

    allowed = tuple(allowed)
    text = ", ".join(allowed)
    missing = [p for p in FOUR_BIT if p not in allowed]
    text += f" (4-bit not allowed: {', '.join(missing)})" if missing and reduced(allowed) else ""
    gpu = [p for p in PRECISION_NEEDS if unsupported(p, capability) is not None]
    if gpu and capability is not None and reduced(allowed):
        text += f"; not on {arch_of(capability)}: {', '.join(gpu)}"
    return text
