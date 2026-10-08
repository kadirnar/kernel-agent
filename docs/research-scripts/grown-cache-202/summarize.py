"""Before / after tables of ``calibrate_grown.py`` results (#202; README, "Quality modes").

    python summarize.py BEFORE.json AFTER.json [BEFORE.json AFTER.json ...] > results.md

Per honest variant: failed draws per quality mode (near-lossless · relaxed) on the captured,
redrawn and scaled inputs, before -> after, and the scaled inputs' worst element ratio. Per
broken variant: whether each quality mode still rejects it (any failed captured, redrawn or
scaled input of the capture) and its failed draws per input class, before -> after.
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
}
QUALITIES = ("near-lossless", "relaxed")
CLASSES = ("captured", "redrawn", "scaled")


def _rows(paths: list[str]) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        for r in json.loads(Path(path).read_text())["results"]:
            rows[(r["capture"], r["variant"])] = r["classes"]
    return rows


def _fails(c: dict[str, Any]) -> str:
    return " · ".join(str(c["fails"][q]) for q in QUALITIES) + f" /{c['draws']}"


def _change(b: dict[str, Any], a: dict[str, Any]) -> str:
    before, after = _fails(b), _fails(a)
    return after if before == after else f"{before} -> **{after}**"


def _rejected(classes: dict[str, Any], quality: str) -> bool:
    return any(c["fails"][quality] for c in classes.values())


def _yes(classes: dict[str, Any], quality: str) -> str:
    return "yes" if _rejected(classes, quality) else "no"


def main(paths: list[str]) -> None:
    before, after = _rows(paths[0::2]), _rows(paths[1::2])
    order = {c: i for i, c in enumerate(dict.fromkeys(k[0] for k in after))}
    keys = sorted((k for k in after if k in before), key=lambda k: order[k[0]])  # by capture
    print(
        "| capture | numerics | captured fails (NL · relaxed) | redrawn fails | scaled fails | "
        "scaled: max element ratio NL / relaxed |"
    )
    print("|---|---|---|---|---|---|")
    for key in keys:
        if key[1].startswith("BUG"):
            continue
        b, a = before[key], after[key]
        ratio = " -> ".join(
            " / ".join(f"{c['scaled']['max_element_ratio'][q]:.2f}" for q in QUALITIES)
            for c in (b, a)
        )
        print(
            f"| {NAMES.get(key[0], key[0])} | {key[1]} | "
            + " | ".join(_change(b[c], a[c]) for c in CLASSES)
            + f" | {ratio} |"
        )
    print()
    print(
        "| capture | broken variant | still rejected (NL · relaxed) | captured fails (NL · "
        "relaxed) | redrawn fails | scaled fails |"
    )
    print("|---|---|---|---|---|---|")
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
            + " | ".join(_change(b[c], a[c]) for c in CLASSES)
            + " |"
        )


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) % 2 == 0:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
