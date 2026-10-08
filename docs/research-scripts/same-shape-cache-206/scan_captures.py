"""Which captured outputs ``compare.written_box`` takes for written caches (#206).

For every case of every capture given, every output tensor of an input tensor's shape (the
case's pre-call arguments) is checked as the evaluator's ``compare_output`` checks it: a grown
input first (``grown_dim`` and the leading part equal), then ``written_box``. Prints, per
capture, the outputs with an input of their shape, how many were taken for grown and for
written caches (with their boxes), and how many elements of the others were left as they
were (the rounding of a residual add: compared whole). On the CPU.

    python scan_captures.py CAPTURE.pt ... > scan.md
"""

from __future__ import annotations

import collections
import gc
import sys
from pathlib import Path

import torch

from kernel_agent.kernels import compare
from kernel_agent.profiling.capture import load_capture


def scan(path: Path) -> str:
    capture = load_capture(path, device="cpu")
    same = grown = 0
    written: collections.Counter[str] = collections.Counter()
    kept: list[float] = []
    for case in capture["cases"]:
        inputs = list(compare.flatten((case["args"], case["kwargs"]), "in").values())
        for ref in compare.flatten(case["output"], "output").values():
            if not ref.is_floating_point():
                continue
            if any(
                (d := compare.grown_dim(before, ref)) is not None
                and bool(compare._same(ref.narrow(d, 0, before.shape[d]), before).all())
                for before in inputs
            ):
                grown += 1
                continue
            shaped = [b for b in inputs if isinstance(b, torch.Tensor) and b.shape == ref.shape]
            shaped = [b for b in shaped if b.dtype == ref.dtype]
            if not shaped:
                continue
            same += 1
            for before in shaped:
                box = compare.written_box(before, ref)
                if box is not None:
                    written[f"{[int(i.numel()) for i in box]} of {list(ref.shape)}"] += 1
                    break
            else:
                kept.append(max(float(compare._same(b, ref).float().mean()) for b in shaped))
    name = f"{path.parent.parent.parent.name}/{path.stem}"
    found = ", ".join(f"{k} x{n}" for k, n in written.items()) or "none"
    most = f"{max(kept):.2%}" if kept else "-"
    cases = len(capture["cases"])
    del capture
    gc.collect()
    return f"| {name} | {cases} | {same} | {grown} | {found} | {most} |"


def main(paths: list[str]) -> None:
    print(
        "| capture | cases | outputs with an input of their shape | grown caches | written "
        "caches (box of shape) | the others: most elements left as they were |"
    )
    print("|---|---|---|---|---|---|")
    for p in paths:
        print(scan(Path(p)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
