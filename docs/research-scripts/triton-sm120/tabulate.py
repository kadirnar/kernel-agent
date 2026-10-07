"""Tabulate the VoxCPM2 run ledgers by kernel backend (issue #135).

Reads copies of the runs (results.tsv + history snapshots + transform files) and
classifies every evaluated candidate by what its source actually contains, not by the
backend label the agent declared. Usage: python -I tabulate.py <runs-copy-dir>
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

RUNS = {
    "R1 latency bf16": "20261005-042829",
    "R2 latency FP8": "20261005-192504",
    "R3 batch-16": "20261006-004718-retest2",  # superset of 20261006-004718 and -retest
}


def classify(src: str) -> str:
    """Backend of a candidate / transform by its source."""
    triton = "@triton.jit" in src
    cuda = "load_inline" in src or "cpp_extension" in src or "cuda.core" in src
    if "T.prim_func" in src or "@tilelang.jit" in src:
        return "tilelang"
    if "cutlass.cute" in src or "@cute.jit" in src or "@cute.kernel" in src:
        return "cute_dsl"
    if triton and cuda:
        return "cuda+triton"
    if triton:
        return "triton"
    if cuda:
        return "cuda"
    if "torch.compile" in src:
        return "inductor"
    if "_scaled_mm" in src:
        return "torch(_scaled_mm)"
    return "torch"


def flags(src: str) -> str:
    out = []
    if re.search(r"#include\s*[<\"]cutlass", src):
        out.append("cutlass-hdr")
    if "tilelang.__file__" in src:
        out.append("hdr-from-tilelang-wheel")
    if "cublasLt" in src:
        out.append("cublasLt")
    if "_scaled_mm" in src:
        out.append("_scaled_mm")
    if "mma.sync" in src or "wmma::" in src:
        out.append("mma")
    if "tl.dot" in src:
        out.append("tl.dot")
    if "float8_e4m3" in src or "e4m3" in src:
        out.append("e4m3")
    return ",".join(out)


def num(x: str) -> float | None:
    try:
        return float(x)
    except ValueError:
        return None


def main(root: Path) -> None:
    per_backend: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    per_target: list[tuple] = []
    transforms: list[tuple] = []
    for label, run in RUNS.items():
        rows = list(csv.DictReader((root / run / "results.tsv").open(), delimiter="\t"))
        by_target: dict[str, list] = defaultdict(list)
        reeval = {r["snapshot"]: r for r in rows if r["status"] == "re-evaluated"}
        best_e2e = None
        seen_kept: set[str] = set()
        for r in rows:
            t = r["target"]
            if t == "e2e":
                if r["backend"] in ("transform", "transform+kernels") and r["status"] == "keep":
                    parts = r["snapshot"].split("+")
                    names = []
                    for p in parts:
                        stem = Path(p).stem
                        while re.match(r"^\d+_", stem):
                            stem = re.sub(r"^\d+_", "", stem)
                        while re.search(r"_[0-9a-f]{8}$", stem):
                            stem = re.sub(r"_[0-9a-f]{8}$", "", stem)
                        names.append(stem)
                    new = [n for n in names if n not in seen_kept]
                    ms = num(r["new_ms"])
                    for n, p in zip(names, parts):
                        if n in new:
                            f = root / run / "transforms" / "history" / Path(p).name
                            src = f.read_text() if f.exists() else ""
                            kind = classify(src) if src else "kernel target (table B)"
                            transforms.append((label, n, kind, flags(src), best_e2e, ms, r["exp"]))
                    seen_kept.update(names)
                    best_e2e = ms
                continue
            if r["status"] == "re-evaluated":
                continue  # not an attempt; its speedup replaces the snapshot's (below)
            if r["snapshot"] in reeval:  # stale record re-measured by a later evaluator
                r = dict(r, speedup=reeval[r["snapshot"]]["speedup"])
            snap = root / run / "targets" / t / "history" / r["snapshot"]
            src = snap.read_text() if snap.exists() else ""
            be = classify(src) if src else f"?{r['backend']}"
            prior = "prior" in r["snapshot"]
            by_target[t].append((r, be, flags(src), prior))
            if prior:
                continue  # library re-evaluations of an earlier run's winner
            per_backend[be]["label"].append(r["backend"])
            per_backend[be]["rows"].append((label, t, r))
        for t, items in by_target.items():
            own = [(r, be, fl) for r, be, fl, prior in items if not prior]
            backends = sorted({be for _, be, _ in own})
            ok = [(num(r["speedup"]) or 0, be, r["snapshot"], fl) for r, be, fl in own if r["correct"] == "true" and num(r["speedup"])]
            best = max(ok) if ok else None
            per_target.append((label, t, backends, len(own), sum(r["correct"] == "true" for r, _, _ in own), best))

    print("## A. Kernel-target evaluations by backend (source-classified, library priors excluded)\n")
    print("| backend (by source) | declared labels | evaluations | correct | correct % | kept | quick checks (pass/fail) | best module speedup (target, run) |")
    print("|---|---|---|---|---|---|---|---|")
    for be, d in sorted(per_backend.items()):
        rows = d["rows"]
        full = [x for x in rows if not x[2]["status"].startswith("quick")]
        quick = [x for x in rows if x[2]["status"].startswith("quick")]
        correct = sum(x[2]["correct"] == "true" for x in full)
        kept = sum(x[2]["status"] == "keep" for x in full)
        best = max(((num(x[2]["speedup"]) or 0, x[1], x[0]) for x in full if x[2]["correct"] == "true"), default=None)
        qok = sum(x[2]["correct"] == "true" for x in quick)
        labels = ", ".join(sorted(set(d["label"])))
        pct = f"{100 * correct / len(full):.0f} %" if full else "—"
        bs = f"{best[0]:.2f}x ({best[1]}, {best[2]})" if best else "—"
        print(f"| {be} | {labels} | {len(full)} | {correct} | {pct} | {kept} | {qok}/{len(quick) - qok} | {bs} |")

    print("\n## A2. Per run and backend\n")
    print("| run | backend | evaluations | correct | kept | quick fail |")
    print("|---|---|---|---|---|---|")
    for be, d in sorted(per_backend.items()):
        for label in RUNS:
            rows = [x for x in d["rows"] if x[0] == label]
            if not rows:
                continue
            full = [x for x in rows if not x[2]["status"].startswith("quick")]
            quick = [x for x in rows if x[2]["status"].startswith("quick")]
            print(f"| {label} | {be} | {len(full)} | {sum(x[2]['correct'] == 'true' for x in full)} | "
                  f"{sum(x[2]['status'] == 'keep' for x in full)} | {sum(x[2]['correct'] != 'true' for x in quick)} |")

    print("\n## B. Per target\n")
    print("| run | target | backends tried (by source) | evals | correct | best (speedup, backend, snapshot, features) |")
    print("|---|---|---|---|---|---|")
    for label, t, backends, n, c, best in per_target:
        b = f"{best[0]:.3f}x, {best[1]}, `{best[2]}`, {best[3]}" if best else "—"
        print(f"| {label} | {t} | {', '.join(backends) or '—'} | {n} | {c} | {b} |")

    print("\n## C. Kept transforms (first time in a kept e2e config), by backend\n")
    print("| run | transform | backend (by source) | features | e2e before | e2e after | exp |")
    print("|---|---|---|---|---|---|---|")
    for label, n, be, fl, before, after, exp in transforms:
        print(f"| {label} | {n} | {be} | {fl} | {before} | {after} | {exp} |")

    print("\n## D. Candidate files written (incl. local-only, never evaluated) by backend\n")
    counts: dict[tuple, int] = defaultdict(int)
    for label, run in RUNS.items():
        for f in (root / run / "targets").glob("*/candidates/*.py"):
            counts[(label, classify(f.read_text()))] += 1
        for f in (root / run / "transforms").glob("*.py"):
            counts[(label, "transform:" + classify(f.read_text()))] += 1
    for k in sorted(counts):
        print(f"* {k[0]}: {k[1]} = {counts[k]}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
