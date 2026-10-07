"""Allowed precisions of a run (issue #131): the target precisions its kernels may use.

``--precisions exact,fp8_weights,fp8_w8a8,reduced`` (``run.json`` → ``config.precisions``)
lists the precisions (a target's ``precision`` in ``spec.json``,
:data:`kernel_agent.kernels.compare.PRECISIONS`) the run's targets may have; ``exact`` is
always one of them. Without the option, and in a run whose ``run.json`` has none (made
before it), the run's ``--quality`` decides (:func:`default`): ``exact`` allows ``exact``
only, ``near-lossless`` every reduced precision but the 4-bit ones (:data:`FOUR_BIT`),
which are opt-in (``--precisions exact,fp8_weights,fp8_w8a8,reduced,fp4_weights``); the
8-bit classes (``fp8_weights``, ``fp8_w8a8``, MXFP8 ``fp8_mx``) are allowed by default. A run
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
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

#: The 4-bit precisions: allowed only when ``--precisions`` names them.
FOUR_BIT = ("fp4_weights",)


def names() -> tuple[str, ...]:
    """Every target precision (``kernels.compare.PRECISIONS``), ``exact`` first."""
    from kernel_agent.kernels.compare import PRECISIONS

    return tuple(PRECISIONS)


def default(quality: str | None) -> tuple[str, ...]:
    """The precisions a run of ``quality`` allows without ``--precisions``: ``exact`` only,
    or (``near-lossless``) every precision but the 4-bit ones."""
    from kernel_agent.kernels.compare import EXACT_TIER, NEAR_LOSSLESS_TIER

    if quality != NEAR_LOSSLESS_TIER:
        return (EXACT_TIER,)
    return tuple(p for p in names() if p not in FOUR_BIT)


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
    reduced precision needs ``--quality near-lossless``."""
    from kernel_agent.kernels.compare import NEAR_LOSSLESS_TIER

    if requested is None:
        return None
    try:
        wanted = parse(requested)
    except ValueError as exc:
        return str(exc)
    reduced = [p for p in wanted if p != "exact"]
    if reduced and quality != NEAR_LOSSLESS_TIER:
        return (
            f"--precisions {','.join(reduced)} needs --quality near-lossless "
            f"(this run: {quality or 'exact'})"
        )
    return None


def allowed(quality: str | None, precisions: Iterable[str] | None = None) -> tuple[str, ...]:
    """The precisions a run allows: ``precisions`` (its ``--precisions``; None: the
    :func:`default` of ``quality``), never a reduced one outside ``near-lossless``."""
    if precisions is None:
        return default(quality)
    try:
        wanted = set(parse(precisions))
    except ValueError:  # an edited run.json: what the quality mode allows by default
        return default(quality)
    from kernel_agent.kernels.compare import NEAR_LOSSLESS_TIER

    return tuple(
        p for p in names() if p in wanted and (quality == NEAR_LOSSLESS_TIER or p == "exact")
    )


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


def refusal(precision: str | None, allowed: Iterable[str]) -> str | None:
    """Why a target at ``precision`` (None: ``exact``) is refused in a run that allows
    ``allowed``, or None."""
    allowed = tuple(allowed)
    name = str(precision or "exact")
    if name in allowed:
        return None
    opt_in = " (4-bit precisions are opt-in)" if name in FOUR_BIT else ""
    return (
        f"precision {name!r} is not allowed in this run{opt_in}: --precisions {','.join(allowed)}"
    )


def tier_allowed(tier: str | None, allowed: Iterable[str]) -> bool:
    """Whether an evaluation in the tolerance ``tier`` (its ``tolerance_tier``; None: not
    recorded) is of a precision ``allowed`` holds: the exact tier, or the near-lossless tier
    of one of them (``near-lossless-fp4``: ``fp4_weights``)."""
    from kernel_agent.kernels.compare import EXACT_TIER, NEAR_LOSSLESS_TIER, TIERS, tier_for

    if tier is None or tier not in TIERS:
        return True
    return tier == EXACT_TIER or tier in {tier_for(NEAR_LOSSLESS_TIER, p) for p in allowed}


def describe(allowed: Iterable[str]) -> str:
    """``exact, fp8_weights, fp8_w8a8, reduced (4-bit not allowed: fp4_weights)``."""
    allowed = tuple(allowed)
    text = ", ".join(allowed)
    missing = [p for p in FOUR_BIT if p not in allowed]
    return text + (
        f" (4-bit not allowed: {', '.join(missing)})" if missing and reduced(allowed) else ""
    )
