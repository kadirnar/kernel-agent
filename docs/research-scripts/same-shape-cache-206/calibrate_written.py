"""Returned same-shape caches compared relative to the update (#206): the #175 / #198 / #202
calibration before and after.

``kernels.compare`` compares an output that equals an input of its shape except for a box of
written rows (``written_box``: a static KV cache returned by a functional ``index_copy``) in
two parts (``compare_written``): the box on its own, the rest bit for bit. Before #206 it
compared the whole tensor. This runs ``../per-channel-198/calibrate_redraw.py`` (the
reference math of every reduced precision and the broken variants on the captured, redrawn
and x 3 / x 0.01 / x -1 inputs, judged in the near-lossless and the relaxed tier at once;
the evaluator's per-channel redraw and #202's grown caches) with

* ``--compare after``: the evaluator's comparison;
* ``--compare before``: same-shape outputs compared whole (``written_box`` never finds one),

on the same seeds, and prints how many outputs ``written_box`` found (and their boxes). On
an RTX 5070 Ti:

* the captures of #202's calibration (the Qwen3-0.6B decoder layer at decode with its
  ``DynamicCache``, ``runs/Qwen--Qwen3-0.6B/20261008-021850/.truth/captures/layer_decode.pt``;
  the VoxCPM2 LocDiT and base-LM layers, ``20261006-004718/dit_layer``,
  ``20261005-192504/lm_step_fp8``): ``written_box`` finds nothing, and ``--compare after``
  gives byte for byte ``../grown-cache-202/results_after_qwen3.json`` and
  ``results_voxcpm2.json`` (the ``results`` lists are equal);
* the same Qwen3 layer with a returned static cache (``static_layer.py``):
  ``results_before_static.json`` / ``results_after_static.json``, ``written_box`` finds every
  K / V output (``[1, 8, 1, 128]`` of ``[1, 8, 2 T, 128]``; batch 2: ``[2, 8, 1, 128]``) and
  nothing else; ``summarize.py`` writes ``results.md``.

    python static_layer.py .../layer_decode.pt qwen3-static/.truth/captures/layer_decode_static.pt
    python calibrate_written.py --compare before --device cuda --seeds 10 \
        --out results_before_static.json qwen3-static/.truth/captures/layer_decode_static.pt
    python summarize.py results_before_static.json results_after_static.json > results.md
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "per-channel-198"))
sys.path.insert(0, str(HERE))  # static_layer: the class of the static-cache capture

import calibrate_redraw as cr

from kernel_agent.kernels import compare

COMPARES = ("before", "after")
FOUND: collections.Counter[str] = collections.Counter()


def main() -> None:
    mode = "after"
    if "--compare" in sys.argv:
        i = sys.argv.index("--compare")
        mode = sys.argv[i + 1]
        del sys.argv[i : i + 2]
    if mode not in COMPARES:
        raise SystemExit(f"--compare {'|'.join(COMPARES)}, not {mode}")
    written_box = compare.written_box

    def counted(before: Any, after: Any) -> Any:
        box = written_box(before, after) if mode == "after" else None
        if box is not None:
            FOUND[f"{[int(i.numel()) for i in box]} of {list(after.shape)}"] += 1
        return box

    compare.written_box = counted  # type: ignore[assignment]
    print(f"compare: {mode}", flush=True)
    cr.main()  # --redraw after (the evaluator's) by default
    print(f"written_box found: {dict(FOUND) or 'none'}", flush=True)


if __name__ == "__main__":
    main()
