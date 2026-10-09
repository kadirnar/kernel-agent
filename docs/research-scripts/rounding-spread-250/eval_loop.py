"""Evaluate the megakernel example (or another candidate) on an L-layer NormGemvChain capture
N times through run_evaluation; print each verdict. Before: the same with ``PYTHONPATH`` on
``git archive origin/main src`` of the commit before #250.

    python eval_loop.py 50 8 > eval_after_1.txt
"""

from __future__ import annotations

import collections
import sys
import tempfile
import time
from pathlib import Path

import kernel_agent
from kernel_agent.agent.prompts import EXAMPLES_DIR
from kernel_agent.kernels.evaluate import run_evaluation
from kernel_agent.selftest import make_norm_chain_capture


def main():
    n, layers = int(sys.argv[1]), int(sys.argv[2])
    candidate = Path(sys.argv[3]) if len(sys.argv) > 3 else EXAMPLES_DIR / "native_megakernel"
    print(f"kernel_agent from {kernel_agent.__file__}", flush=True)
    tally = collections.Counter()
    with tempfile.TemporaryDirectory() as tmp:
        capture = make_norm_chain_capture(Path(tmp) / "chain.pt", 1024, layers, [((1,), 64), ((4,), 0)])
        for i in range(n):
            t0 = time.time()
            r = run_evaluation(capture, candidate)
            status = r.get("status")
            tally[status] += 1
            err = str(r.get("error", ""))[-220:] if status != "ok" else f"{r.get('speedup')}x"
            print(f"{i:3d} {status} {time.time() - t0:.0f}s {err}", flush=True)
    print(dict(tally), flush=True)


if __name__ == "__main__":
    main()
