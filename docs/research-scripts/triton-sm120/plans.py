"""Planner backend choices per target in the run plans (#135). python -I plans.py <runs-copy>"""

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for run in ("20261005-042829", "20261005-192504", "20261006-004718-retest2"):
    print("==", run)
    for plan in sorted((root / run).glob("**/plan.json")):
        try:
            d = json.loads(plan.read_text())
        except Exception:
            continue
        for t in d.get("targets", []):
            print(" ", plan.relative_to(root / run), "|", t.get("id"), "| backends:", t.get("backends"),
                  "| precision:", t.get("precision"))
