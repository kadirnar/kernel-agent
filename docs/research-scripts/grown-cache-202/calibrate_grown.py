"""Grown caches compared relative to the update (#202): the #175 / #198 calibration before
and after.

``kernels.compare`` compares a cache that a call grows along one dimension (``grown_dim``:
a ``DynamicCache`` layer ``torch.cat``-ed with the new token's K / V, or a returned grown
tensor whose leading part equals an input) in two parts (``compare_grown``): the appended
rows on their own, the kept part bit for bit. Before #202 it compared the whole cache, so
the kept rows diluted the new ones. This runs ``../per-channel-198/calibrate_redraw.py``
(the reference math of every reduced precision and the broken variants on the captured,
redrawn and x 3 / x 0.01 / x -1 inputs, judged in the near-lossless and the relaxed tier
at once; the evaluator's per-channel redraw) with

* ``--compare after``: the evaluator's comparison;
* ``--compare before``: grown caches compared whole (``grown_dim`` never finds one),

on the same seeds (RTX 5070 Ti). ``results_before_qwen3.json`` / ``results_after_qwen3.json``:
the Qwen3-0.6B decoder layer at decode (``runs/Qwen--Qwen3-0.6B/20261008-021850``, its
``DynamicCache`` grows by one token per call); ``results_voxcpm2.json``: the VoxCPM2 LocDiT
and base-LM decode layers (``20261006-004718/dit_layer``, ``20261005-192504/lm_step_fp8``:
in-place caches, no grown tensor), byte for byte the same in both modes and the same
verdicts as #198's ``results_after.json``. ``summarize.py`` writes ``results.md`` (before ->
after); ``exploit.py`` writes ``exploit.md`` (one wrong new row, every tier).

    python calibrate_grown.py --compare before --device cuda --seeds 10 \
        --out results_before_qwen3.json .../Qwen--Qwen3-0.6B/20261008-021850/.truth/captures/layer_decode.pt
    python summarize.py results_before_qwen3.json results_after_qwen3.json \
        results_voxcpm2.json results_voxcpm2.json > results.md
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "per-channel-198"))

import calibrate_redraw as cr

from kernel_agent.kernels import compare

COMPARES = ("before", "after")


def main() -> None:
    mode = "after"
    if "--compare" in sys.argv:
        i = sys.argv.index("--compare")
        mode = sys.argv[i + 1]
        del sys.argv[i : i + 2]
    if mode not in COMPARES:
        raise SystemExit(f"--compare {'|'.join(COMPARES)}, not {mode}")
    if mode == "before":
        compare.grown_dim = lambda before, after: None  # type: ignore[assignment]
    print(f"compare: {mode}", flush=True)
    cr.main()  # --redraw after (the evaluator's) by default


if __name__ == "__main__":
    main()
