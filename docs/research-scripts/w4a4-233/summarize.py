"""The summary tables of ``calibrate_w4a4.py`` results (README "W4A4", skill fp4-w4a4).

    python summarize.py results/calib_*.json > results/summary.md
"""

from __future__ import annotations

import json
import sys
from typing import Any

NAMES = {
    "20261006-004718/dit_layer": "VoxCPM2 LocDiT layer (M = 352, 176)",
    "20261006-004718/loc_enc_decode": "VoxCPM2 LocEnc, 12 layers (M = 80)",
    "20261005-192504/lm_step_fp8": "VoxCPM2 base-LM decode layer (M = 1)",
    "20261008-021850/layer_decode": "Qwen3-0.6B decoder layer, decode (M = 1)",
}
CLASSES = ("captured", "redrawn", "scaled")
QUALITIES = ("near-lossless", "relaxed")


def fails(row: dict[str, Any], quality: str) -> str:
    return ", ".join(
        f"{row['classes'][c]['fails'][quality]}/{row['classes'][c]['draws']}" for c in CLASSES
    )


def main(paths: list[str]) -> None:
    rows = [r for p in paths for r in json.load(open(p))["results"]]
    good = [r for r in rows if not r["variant"].startswith("BUG")]
    bugs = [r for r in rows if r["variant"].startswith("BUG")]
    print(
        "| capture | numerics | captured: min cosine / max rel L2 / max norm change | "
        "redrawn + scaled: min cosine / max rel L2 / max norm change | near-lossless-fp4a "
        "fails (captured, redrawn, scaled) | relaxed-fp4a fails |"
    )
    print("|---|---|---|---|---|---|")
    for r in good:
        c = r["classes"]["captured"]
        p = [r["classes"][k] for k in ("redrawn", "scaled")]
        print(
            f"| {NAMES.get(r['capture'], r['capture'])} | {r['variant']} | "
            f"{c['min_cosine']:.4f} / {c['max_rel_l2']:.3f} / {c['max_norm_change'] * 100:.1f} % | "
            f"{min(s['min_cosine'] for s in p):.4f} / {max(s['max_rel_l2'] for s in p):.3f} / "
            f"{max(s['max_norm_change'] for s in p) * 100:.1f} % | "
            f"{fails(r, 'near-lossless')} | {fails(r, 'relaxed')} |"
        )
    print()
    print(
        "| capture | broken variant | rejected (near-lossless-fp4a · relaxed-fp4a) | "
        "near-lossless-fp4a fails (captured, redrawn, scaled) | relaxed-fp4a fails |"
    )
    print("|---|---|---|---|---|")
    for r in bugs:
        caught = [
            "yes" if any(r["classes"][c]["fails"][q] for c in CLASSES) else "**no**"
            for q in QUALITIES
        ]
        print(
            f"| {NAMES.get(r['capture'], r['capture'])} | {r['variant'].removeprefix('BUG ')} | "
            f"{' · '.join(caught)} | {fails(r, 'near-lossless')} | {fails(r, 'relaxed')} |"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
