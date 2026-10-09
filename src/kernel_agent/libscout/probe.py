"""The library scout's GPU step on one capture (issue #227), in a subprocess of its own.

It loads the capture, restores the module state of its dominant case (the most calls per
run) and runs that case once under :func:`kernel_agent.libscout.detect.trace`. Then, in the
same process:

* the op families of what the reference called (:func:`detect.families`);
* every adapter's availability (:meth:`Adapter.probe`: the imports happen here, never in
  the coordinator) and the decision for this target (:func:`registry.applicable`: the GPU's
  capability, the target's precision, the toolchain's backends);
* the candidate file of every adapter that runs (:func:`template.render`), written to
  ``--out-dir`` as ``libscout_<adapter>.py``, built and called once on the dominant case
  (:func:`_dry_run`): one that fails there, or whose rewrite finds nothing in TorchDynamo's
  graphs (an RMSNorm inside a TorchScript function: the trace sees into it, Dynamo does
  not), is not swept and says why;
* **op bars**: each recorded call that an adapter's ``ops(reference, **config)`` maps (an
  SDPA call, a softmax), called again on its recorded inputs with the reference's own op
  and with the library's, checked against the reference op's output in the capture's
  tolerance tier and timed interleaved (:data:`ROUNDS` rounds, the median of each) twice:
  as kernel time (the calls replayed from a CUDA graph) and per eager call (the host's
  launch cost included). A pattern an adapter folds (``patterns(reference, **config)``:
  the written-out RMSNorm, :func:`pattern_bars`) has its op bar too: its recorded calls
  replayed (the reference's own kernels, about six) against the library's fused kernel on
  the same x and weight. Adapters that build a CUDA extension (``compiles``) have none
  (their sweep builds it). The op bar is the library's speed on the op alone, apart from
  what the module around it costs (cuDNN attention can be 2x faster on its call and slower
  in an eager, launch-bound layer): what an engineer can count on when calling the library
  inside a kernel or a CUDA graph.

The result arrives on stdout on a line tagged ``@@KA_LIBSCOUT@@`` (``run_probe`` runs it under
the caller's GPU lock)::

    python -m kernel_agent.libscout.probe CAPTURE --out-dir DIR --target ID \\
        [--precision P] [--backends cuda,triton] [--sha256 S] [--no-bars]
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

MARKER = "@@KA_LIBSCOUT@@"
ROUNDS = 5  # op bar rounds, interleaved (the median of each function's rounds)
ROUND_MS = 2.0  # GPU time per function and round
MAX_BAR_CALLS = 12  # distinct call signatures timed per adapter and config


def candidate_name(adapter: str) -> str:
    """``libscout_torch_sdpa.py`` for ``torch-sdpa``."""
    return "libscout_" + adapter.replace("-", "_") + ".py"


def dominant_case(cases: Sequence[Mapping[str, Any]]) -> int:
    """The case with the most calls per run (the first of equals; correctness-only cases,
    count 0, only when there is nothing else)."""
    best, most = 0, -1
    for i, case in enumerate(cases):
        count = int(case.get("count") or 0)
        if count > most:
            best, most = i, count
    return best


# ------------------------------------------------------------------ op bars


def _signature(name: str, args: Any, kwargs: Mapping[str, Any]) -> str:
    import torch

    from kernel_agent.libscout.detect import _tensors

    parts = [str(list(t.shape)) for t in _tensors(args)]
    dtypes = sorted({str(t.dtype).removeprefix("torch.") for t in _tensors(args)})
    flags = [
        f"{k}={v!r}"
        for k, v in kwargs.items()
        if v is not None and (isinstance(v, (bool | int | float | str, torch.dtype)))
    ]
    return f"{name}({' '.join(parts)} {'/'.join(dtypes)}" + (
        f"; {', '.join(flags)})" if flags else ")"
    )


def _key(func: Any, args: Any, kwargs: Mapping[str, Any]) -> tuple[Any, ...]:
    from kernel_agent.libscout.detect import _plain, _tensors

    tensors = tuple((tuple(t.shape), str(t.dtype), t.device.type) for t in _tensors(args))
    tensors += tuple((tuple(t.shape), str(t.dtype), "kw") for t in _tensors(kwargs))
    plain = tuple(sorted((k, repr(v)) for k, v in kwargs.items() if _plain(v)))
    return (id(func), tensors, plain)


def _rounds(runs: Sequence[Callable[[], Any]], iters: int, rounds: int) -> list[float]:
    """Median microseconds per call of each ``runs`` entry (one call runs ``iters`` calls of
    its op), interleaved: every round times each in turn (CUDA events)."""
    import torch

    times: list[list[float]] = [[] for _ in runs]
    for _ in range(rounds):
        for i, run in enumerate(runs):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            run()
            end.record()
            end.synchronize()
            times[i].append(start.elapsed_time(end) * 1000.0 / iters)
    return [round(statistics.median(t), 2) for t in times]


def _graph(fn: Callable[[], Any], iters: int) -> Callable[[], Any]:
    """``iters`` calls of ``fn`` captured in one CUDA graph: its replay."""
    import torch

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # warm up off the capture: plans, workspaces
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn()
    return graph.replay


def time_functions(
    fns: Sequence[Callable[[], Any]], *, rounds: int = ROUNDS, round_ms: float = ROUND_MS
) -> dict[str, list[float] | None]:
    """Microseconds per call of each function: ``eager`` (a loop of calls: the host's launch
    cost included when it exceeds the GPU time, as in an eager module) and ``graph`` (the
    calls replayed from a CUDA graph: the kernels' own time; None when one does not
    capture). Medians of interleaved rounds; the iteration count from one warm call of the
    slowest."""
    import torch

    for fn in fns:  # warm up: cuDNN plans, autotuning, lazy imports
        fn()
        fn()
    torch.cuda.synchronize()
    first: list[float] = []
    for fn in fns:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        first.append(max(start.elapsed_time(end), 1e-3))
    iters = max(3, min(1000, int(round_ms / max(first))))

    def loop(fn: Callable[[], Any]) -> Callable[[], None]:
        def run() -> None:
            for _ in range(iters):
                fn()

        return run

    out: dict[str, list[float] | None] = {"eager": _rounds([loop(fn) for fn in fns], iters, rounds)}
    captured = min(iters, 50)
    try:
        replays = [_graph(fn, captured) for fn in fns]
    except Exception:  # an op that syncs with the host or allocates outside the pool
        torch.cuda.synchronize()
        out["graph"] = None
    else:
        out["graph"] = _rounds(replays, captured, rounds)
    return out


Timer = Callable[[Sequence[Callable[[], Any]]], Mapping[str, list[float] | None]]


def _measure(
    bar: dict[str, Any],
    mine: Callable[[], Any],
    theirs: Callable[[], Any],
    *,
    tier: str | None,
    timer: Timer,
) -> dict[str, Any]:
    """One op bar: the library's output (``theirs``) checked against the reference's
    (``mine``) in the capture's tolerance tier, both timed by ``timer``
    (:func:`time_functions`): ``ok`` / ``error``, eager and kernel times, speedups."""
    import torch

    from kernel_agent.kernels.compare import compare_structures

    try:
        with torch.inference_mode():  # the recorded tensors are inference tensors
            reports = compare_structures(mine(), theirs(), tier=tier)
            failed = [r for r in reports if not r.get("ok")]
            bar["ok"] = not failed
            if failed:
                bar["error"] = str(failed[0])[:300]
            times = timer([mine, theirs])
    except Exception as exc:  # a backend that cannot take this call
        bar.update(ok=False, error=f"{type(exc).__name__}: {exc}"[:300])
    else:
        eager, graph = times["eager"] or [], times["graph"]
        bar.update(ref_eager_us=eager[0], eager_us=eager[1])
        bar["eager_speedup"] = round(eager[0] / eager[1], 3)
        if graph is not None:  # the kernels' own time: what the bar is
            bar.update(ref_us=graph[0], us=graph[1])
            bar["speedup"] = round(graph[0] / graph[1], 3)
    return bar


def op_bars(
    reference: Any,
    invocations: Sequence[tuple[Any, Any, Any]],
    modules: Mapping[str, tuple[Any, list[dict[str, Any]]]],
    *,
    tier: str | None = None,
    timer: Timer | None = None,
) -> list[dict[str, Any]]:
    """The op bars of the adapters in ``modules`` (``{adapter: (candidate module,
    configs)}``) on the recorded ``invocations`` (see the module docstring); ``timer``:
    :func:`time_functions` (a fake in the CPU tests)."""
    timer = timer or time_functions
    bars: list[dict[str, Any]] = []
    for adapter, (module, configs) in modules.items():
        for config in configs:
            try:
                table = module.ops(reference, **config)
            except Exception as exc:
                bars.append({"adapter": adapter, "config": config, "error": repr(exc)[:300]})
                continue
            groups: dict[tuple[Any, ...], dict[str, Any]] = {}
            for func, args, kwargs in invocations:
                if func not in table:
                    continue
                key = _key(func, args, kwargs)
                if key in groups:
                    groups[key]["calls"] += 1
                elif len(groups) < MAX_BAR_CALLS:
                    groups[key] = {"func": func, "args": args, "kwargs": kwargs, "calls": 1}
            for group in groups.values():
                func, args, kwargs = group["func"], group["args"], group["kwargs"]
                name = getattr(func, "__name__", str(func))
                bar: dict[str, Any] = {
                    "adapter": adapter,
                    "config": config,
                    "op": name,
                    "signature": _signature(name, args, kwargs),
                    "calls": group["calls"],
                }
                mine = functools.partial(func, *args, **kwargs)
                theirs = functools.partial(table[func], *args, **kwargs)
                bars.append(_measure(bar, mine, theirs, tier=tier, timer=timer))
    return bars


#: The op bar name of a written-out RMSNorm (:func:`pattern_bars`)
RMS_PATTERN = "rms_norm (written out)"


def pattern_bars(
    reference: Any,
    invocations: Sequence[tuple[Any, Any, Any]],
    calls: Sequence[Any],
    modules: Mapping[str, tuple[Any, list[dict[str, Any]]]],
    *,
    tier: str | None = None,
    timer: Timer | None = None,
) -> list[dict[str, Any]]:
    """Op bars of the patterns the adapters in ``modules`` fold (``patterns(reference,
    **config)`` of a candidate: ``{"rms_norm": fn(x, weight, eps, dtype, mid)}``, what its
    rewrite puts in the pattern's place): every written-out RMSNorm of the trace
    (``detect.rms_spans`` of ``calls``) replayed from its recorded calls (the reference's
    own kernels, about six) against the adapter's function on the same x and weight,
    checked and timed like :func:`op_bars` (one bar per distinct signature)."""
    import torch

    from kernel_agent.libscout.detect import _tensors, rms_spans

    timer = timer or time_functions
    spans = rms_spans(list(calls))
    params = dict(reference.named_parameters()) if hasattr(reference, "named_parameters") else {}
    sites: dict[tuple[Any, ...], dict[str, Any]] = {}
    for span in spans:
        x = next(iter(_tensors(invocations[span.entry][1])), None)
        weight = params.get(span.weight) if span.weight else None
        dtype = getattr(torch, span.dtype, None)
        mid = getattr(torch, span.mid, None) if span.mid else None
        if x is None or span.eps is None or (span.weight and weight is None):
            continue
        if not isinstance(dtype, torch.dtype) or (span.mid and not isinstance(mid, torch.dtype)):
            continue
        args = (x,) if weight is None else (x, weight)
        signature = _signature(RMS_PATTERN, args, {"eps": span.eps})
        key = (signature, span.dtype, span.mid, len(span.calls))
        if key in sites:
            sites[key]["calls"] += 1
        elif len(sites) < MAX_BAR_CALLS:
            sites[key] = {
                "span": span,
                "args": (x, weight, span.eps, dtype, mid),  # the fused function's
                "sig": signature,
                "calls": 1,
            }
    bars: list[dict[str, Any]] = []
    for adapter, (module, configs) in modules.items():
        if not callable(getattr(module, "patterns", None)):
            continue
        for config in configs:
            try:
                fused = module.patterns(reference, **config).get("rms_norm")
            except Exception as exc:
                bars.append({"adapter": adapter, "config": config, "error": repr(exc)[:300]})
                continue
            if fused is None:
                continue
            for site in sites.values():
                span = site["span"]
                steps = [invocations[i] for i in span.calls]

                def replay(steps: list[Any] = steps) -> Any:
                    out = None
                    for func, a, kw in steps:
                        out = func(*a, **kw)
                    return out

                theirs = functools.partial(fused, *site["args"])
                bar = {
                    "adapter": adapter,
                    "config": config,
                    "op": RMS_PATTERN,
                    "signature": site["sig"],
                    "calls": site["calls"],
                    "pattern_ops": len(span.calls),
                }
                bars.append(_measure(bar, replay, theirs, tier=tier, timer=timer))
    return bars


#: A config whose op bars are all below this speedup over the reference's op, as kernel time
#: and per eager call, is not swept: it cannot make the module faster
PRUNE_BELOW = 0.9


def prune(
    configs: Sequence[Mapping[str, Any]], bars: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The configs worth a sweep and the others with why: a config whose op bar failed the
    tolerance, or whose op is slower than the reference's by more than :data:`PRUNE_BELOW`
    both as kernel time and per eager call. A config without bars stays; when none would,
    the fastest correct one does (its module-level verdict is still worth having)."""
    keep: list[dict[str, Any]] = []
    pruned: list[tuple[float, dict[str, Any]]] = []  # (best op speedup, -1: fails; entry)
    for config in configs:
        mine = [b for b in bars if b.get("config") == config]
        if not mine:
            keep.append(dict(config))
            continue
        failed = next((b for b in mine if not b.get("ok")), None)
        if failed is not None:
            why = f"op bar fails: {str(failed.get('error') or '')[:160]}"
            pruned.append((-1.0, {"config": dict(config), "why": why}))
            continue
        speeds = [s for b in mine for k in ("speedup", "eager_speedup") if (s := b.get(k))]
        if speeds and max(speeds) < PRUNE_BELOW:
            why = f"op bar at most {max(speeds):.2f}x"
            pruned.append((max(speeds), {"config": dict(config), "why": why}))
            continue
        keep.append(dict(config))
    if not keep and pruned:
        rescued = max(pruned, key=lambda p: p[0])
        pruned.remove(rescued)
        keep.append(rescued[1]["config"])
    return keep, [entry for _, entry in pruned]


# ------------------------------------------------------------------ in the subprocess


def _import(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"ka_libscout_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Why a candidate whose rewrite changed nothing is not swept
UNCHANGED = (
    "TorchDynamo's graphs of the reference show none of the calls it replaces (they run "
    "inside an opaque call, such as a TorchScript function)"
)


def _dry_run(
    path: Path, reference: Any, replay: Any, case: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[Any, str | None]:
    """The candidate module at ``path`` built with ``config`` and called once on the
    dominant case: the module and why it is not worth a sweep (None: its rewrite changed
    the reference's graphs). The trace saw inside every call, Dynamo does not: a TorchScript
    function is one opaque node there, so its RMSNorm is not the scout's to replace."""
    import torch

    from kernel_agent.profiling.methods import entrypoint

    try:
        module = _import(path)
        candidate = module.build(reference, **config)
        if replay:
            replay.restore(case, reference)
        with torch.inference_mode():
            entrypoint(candidate, case.get("method", "forward"))(*case["args"], **case["kwargs"])
    except Exception as exc:
        # what the rewrite raised, not Dynamo's BackendCompilerFailed around it
        inner = getattr(exc, "inner_exception", None) or exc
        why = f"{type(inner).__name__}: {inner}".splitlines()[0][:240]
        return None, f"its candidate failed on the dominant case in the probe: {why}"
    if not getattr(candidate, "rewritten", 0):
        return module, UNCHANGED
    return module, None


def probe(
    capture_path: Path,
    *,
    out_dir: Path,
    target: str,
    precision: str | None = None,
    backends: Mapping[str, bool] | None = None,
    sha256: str | None = None,
    bars: bool = True,
) -> dict[str, Any]:
    """Everything the scout needs from the GPU for one target (the module docstring)."""
    import torch

    from kernel_agent import toolchain
    from kernel_agent.kernels.compare import tier_of
    from kernel_agent.libscout import detect, registry, template
    from kernel_agent.libscout.adapters import ADAPTERS
    from kernel_agent.profiling.capture import load_capture
    from kernel_agent.profiling.methods import entrypoint
    from kernel_agent.profiling.state import Replay

    start = time.monotonic()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        toolchain.setup()  # the compiler environment of load_inline, as the evaluator's
    capture = load_capture(capture_path, device=device, sha256=sha256)
    reference = capture["module"].eval()
    cases = capture["cases"]
    index = dominant_case(cases)
    case = cases[index]
    replay = Replay(capture, reference)
    if replay:
        replay.restore(case, reference)
    invocations: list[Any] = []
    calls = detect.trace(
        entrypoint(reference, case.get("method", "forward")),
        case["args"],
        case["kwargs"],
        module=reference,
        invocations=invocations,
    )
    found = detect.summary(detect.families(calls))
    available = registry.availability(ADAPTERS)
    capability = tuple(torch.cuda.get_device_capability()) if device == "cuda" else None
    decisions = registry.applicable(
        ADAPTERS,
        found,
        capability=capability,
        precision=precision,
        available=available,
        backends=backends,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = list(capture.get("methods") or {"forward": 1})
    described = detect.describe(found)
    written: dict[str, str] = {}
    timed: dict[str, tuple[Any, list[dict[str, Any]]]] = {}
    rows = [d.to_dict() for d in decisions]
    for d, row in zip(decisions, rows, strict=True):
        if not d.run:
            continue
        version = available[d.adapter.name].version
        path = out_dir / candidate_name(d.adapter.name)
        path.write_text(
            template.render(
                d.adapter,
                target=target,
                version=version,
                configs=d.configs,
                families=described,
                methods=methods,
            )
        )
        module, why = _dry_run(path, reference, replay, case, d.configs[0])
        if why is not None:  # no sweep: it would fail, or be the reference's graph unchanged
            row.update(run=False, reason=why)
            row.pop("configs", None)
            continue
        written[d.adapter.name] = str(path)
        if bars and device == "cuda" and not d.adapter.compiles:
            timed[d.adapter.name] = (module, d.configs)
    measured = []
    if timed:  # one op for one op, and the written-out patterns the adapters fold
        measured = op_bars(reference, invocations, timed, tier=tier_of(capture))
        measured += pattern_bars(reference, invocations, calls, timed, tier=tier_of(capture))
    for d, row in zip(decisions, rows, strict=True):  # the op bars spare the sweep its losers
        if not row.get("run") or d.adapter.name not in timed:
            continue
        keep, pruned = prune(
            row["configs"], [b for b in measured if b["adapter"] == row["adapter"]]
        )
        if pruned:
            row.update(configs=keep, pruned=pruned)
            Path(written[d.adapter.name]).write_text(
                template.render(
                    d.adapter,
                    target=target,
                    version=available[d.adapter.name].version,
                    configs=keep,
                    families=described,
                    methods=methods,
                )
            )
    return {
        "target": target,
        "case": case.get("signature"),
        "case_index": index,
        "calls": len(calls),
        "families": found,
        "described": described,
        "capability": list(capability) if capability else None,
        "available": {k: v.to_dict() for k, v in available.items()},
        "decisions": rows,
        "candidates": written,
        "op_bars": measured,
        "seconds": round(time.monotonic() - start, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The library scout's GPU step on a capture.")
    parser.add_argument("capture", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target", default="target")
    parser.add_argument("--precision", default=None)
    parser.add_argument("--backends", default="", help="available backends, comma-separated")
    parser.add_argument("--sha256", default=None)
    parser.add_argument("--no-bars", action="store_true")
    ns = parser.parse_args(argv)
    backends = {b: True for b in ns.backends.split(",") if b} if ns.backends else None
    result = probe(
        ns.capture,
        out_dir=ns.out_dir,
        target=ns.target,
        precision=ns.precision,
        backends=backends,
        sha256=ns.sha256,
        bars=not ns.no_bars,
    )
    print(MARKER + json.dumps(result, default=str), flush=True)
    return 0


# ------------------------------------------------------------------ the caller's side


def run_probe(
    capture_path: Path,
    *,
    out_dir: Path,
    target: str,
    precision: str | None = None,
    backends: Mapping[str, bool] | None = None,
    sha256: str | None = None,
    bars: bool = True,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """:func:`probe` in a subprocess under the GPU lock (nested: the scout holds it for the
    whole target). ``{"error": ...}`` when it fails."""
    from kernel_agent.gpulock import child_env, gpu_lock

    cmd = [
        sys.executable,
        "-m",
        "kernel_agent.libscout.probe",
        str(capture_path),
        "--out-dir",
        str(out_dir),
        "--target",
        target,
    ]
    if precision:
        cmd += ["--precision", precision]
    if backends is not None:
        cmd += ["--backends", ",".join(b for b, ok in backends.items() if ok)]
    if sha256:
        cmd += ["--sha256", sha256]
    if not bars:
        cmd.append("--no-bars")
    with gpu_lock():
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, env=child_env()
            )
        except subprocess.TimeoutExpired:
            return {"error": f"the scout's probe timed out after {timeout:.0f} s"}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(MARKER):
            data: dict[str, Any] = json.loads(line[len(MARKER) :])
            return data
    tail = (proc.stderr or proc.stdout or "")[-3000:]
    return {"error": f"probe exited with {proc.returncode}: {tail}"}


if __name__ == "__main__":
    raise SystemExit(main())
