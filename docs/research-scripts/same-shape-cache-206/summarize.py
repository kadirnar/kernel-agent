"""Before / after tables of ``calibrate_written.py`` results (#206): ``../grown-cache-202/
summarize.py`` with the names of this calibration's captures.

    python summarize.py BEFORE.json AFTER.json [BEFORE.json AFTER.json ...] > results.md
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "grown-cache-202"))

import summarize as grown

grown.NAMES["qwen3-static/layer_decode_static"] = (
    "Qwen3-0.6B decoder layer, decode (M = 1), returned static cache"
)

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) % 2 == 0:
        raise SystemExit(__doc__)
    grown.main(sys.argv[1:])
