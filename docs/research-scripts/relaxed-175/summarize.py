"""The two summary tables of a ``calibrate_tiers.py`` result (README, "Quality modes").

    python summarize.py results.json
"""

from __future__ import annotations

import json
import sys
from typing import Any

NAMES = {
    "20261006-004718/dit_layer": "VoxCPM2 LocDiT layer (M = 352, 176)",
    "20261005-192504/lm_step_fp8": "VoxCPM2 base-LM decode layer (M = 1, KV cache)",
    "20261008-021850/layer_decode": "Qwen3-0.6B decoder layer, decode (M = 1, KV cache)",
    "decode attention (synthetic)": "decode attention, 12 seeds x 4 GQA shapes (#145)",
}
CLASSES = ("captured", "redrawn", "scaled")


def _fails(row: dict[str, Any], quality: str) -> str:
    return ", ".join(
        f"{c['fails'][quality]}/{c['draws']}" for c in (row["classes"][k] for k in CLASSES)
    )


def main(path: str) -> None:
    data = json.load(open(path))
    rows = data["results"]
    good = [r for r in rows if not r["variant"].startswith("BUG")]
    bugs = [r for r in rows if r["variant"].startswith("BUG")]
    print(
        "| capture | numerics | captured inputs, relaxed tier: min cosine / max rel L2 / "
        "max norm change / max element ratio | near-lossless fails (captured, redrawn, "
        "scaled) | relaxed fails |"
    )
    print("|---|---|---|---|---|")
    for r in good:
        c = r["classes"]["captured"]
        metrics = (
            f"{c['min_cosine']:.4f} / {c['max_rel_l2']:.3f} / {c['max_norm_change'] * 100:.1f} % / "
            f"{c['max_element_ratio']['relaxed']:.2f}"
        )
        print(
            f"| {NAMES.get(r['capture'], r['capture'])} | {r['variant']} | {metrics} | "
            f"{_fails(r, 'near-lossless')} | {_fails(r, 'relaxed')} |"
        )
    print()
    captures = list(dict.fromkeys(r["capture"] for r in bugs))
    print(
        "| broken variant | "
        + " | ".join(NAMES.get(c, c) for c in captures)
        + " |\n|---|"
        + "---|" * len(captures)
    )
    for variant in dict.fromkeys(r["variant"] for r in bugs):
        cells = []
        for capture in captures:
            row = next((r for r in bugs if r["capture"] == capture and r["variant"] == variant), None)
            if row is None:
                cells.append("—")
                continue
            c, red = row["classes"]["captured"], row["classes"]["redrawn"]
            cells.append(
                f"{c['fails']['near-lossless']}/{c['draws']} · {c['fails']['relaxed']}/{c['draws']} "
                f"(redrawn {red['fails']['near-lossless']} · {red['fails']['relaxed']} of "
                f"{red['draws']})"
            )
        print(f"| {variant.removeprefix('BUG ')} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1])
