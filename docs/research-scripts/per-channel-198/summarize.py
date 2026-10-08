"""Before / after tables of ``calibrate_redraw.py`` results (#198; README, "Quality modes").

    python summarize.py BEFORE.json AFTER.json [BEFORE.json AFTER.json ...]

Per honest variant: failed draws per quality mode on the redrawn inputs (before -> after)
and on the captured / scaled inputs (the redraw does not change them). Per broken variant:
whether each quality mode still rejects it (any failed captured, redrawn or scaled input of
the capture) and its failed redrawn draws before -> after.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

NAMES = {
    "20261006-004718/dit_layer": "VoxCPM2 LocDiT layer (M = 352, 176)",
    "20261005-192504/lm_step_fp8": "VoxCPM2 base-LM decode layer (M = 1)",
    "20261008-021850/layer_decode": "Qwen3-0.6B decoder layer, decode (M = 1)",
    "decode attention (synthetic)": "fp8_kv decode attention (synthetic, #145)",
}
QUALITIES = ("near-lossless", "relaxed")


def _rows(paths: list[str]) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        for r in json.loads(Path(path).read_text())["results"]:
            rows[(r["capture"], r["variant"])] = r["classes"]
    return rows


def _fails(c: dict[str, Any], quality: str) -> str:
    return f"{c['fails'][quality]}/{c['draws']}"


def _rejected(classes: dict[str, Any], quality: str) -> bool:
    return any(c["fails"][quality] for c in classes.values())


def _yes(classes: dict[str, Any], quality: str) -> str:
    return "yes" if _rejected(classes, quality) else "no"


def main(paths: list[str]) -> None:
    before, after = _rows(paths[0::2]), _rows(paths[1::2])
    order = {c: i for i, c in enumerate(dict.fromkeys(k[0] for k in after))}
    keys = sorted((k for k in after if k in before), key=lambda k: order[k[0]])  # by capture
    print(
        "| capture | numerics | redrawn fails, NL: before -> after | redrawn fails, relaxed: "
        "before -> after | after, redrawn: min cosine / max rel L2 / max norm change / max "
        "element ratio (relaxed) | captured, scaled fails (NL · relaxed) |"
    )
    print("|---|---|---|---|---|---|")
    for key in keys:
        if key[1].startswith("BUG"):
            continue
        b, a = before[key], after[key]
        red = a["redrawn"]
        metrics = (
            f"{red['min_cosine']:.4f} / {red['max_rel_l2']:.3f} / "
            f"{red['max_norm_change'] * 100:.1f} % / {red['max_element_ratio']['relaxed']:.2f}"
        )
        fixed = " · ".join(
            f"{_fails(a[c], 'near-lossless')} · {_fails(a[c], 'relaxed')}"
            for c in ("captured", "scaled")
        )
        print(
            f"| {NAMES.get(key[0], key[0])} | {key[1]} | "
            + " | ".join(
                f"{_fails(b['redrawn'], q)} -> **{_fails(a['redrawn'], q)}**" for q in QUALITIES
            )
            + f" | {metrics} | {fixed} |"
        )
    print()
    print(
        "| capture | broken variant | still rejected (NL · relaxed) | redrawn fails, NL: before "
        "-> after | redrawn fails, relaxed: before -> after | captured fails (NL · relaxed) | "
        "scaled fails (NL · relaxed) |"
    )
    print("|---|---|---|---|---|---|---|")
    for key in keys:
        if not key[1].startswith("BUG"):
            continue
        b, a = before[key], after[key]
        kept = " · ".join(
            ("yes" if _rejected(a, q) else "**no**")
            + ("" if _rejected(a, q) == _rejected(b, q) else f" (before: {_yes(b, q)})")
            for q in QUALITIES
        )
        print(
            f"| {NAMES.get(key[0], key[0])} | {key[1].removeprefix('BUG ')} | {kept} | "
            + " | ".join(
                f"{_fails(b['redrawn'], q)} -> {_fails(a['redrawn'], q)}" for q in QUALITIES
            )
            + " | "
            + " | ".join(
                f"{_fails(a[c], 'near-lossless')} · {_fails(a[c], 'relaxed')}"
                for c in ("captured", "scaled")
            )
            + " |"
        )


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) % 2 == 0:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
