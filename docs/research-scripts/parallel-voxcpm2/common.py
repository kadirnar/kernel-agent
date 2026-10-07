"""Shared helpers for the #136 measurements: load VoxCPM2 (eager or the accepted optimised
set), tag the stages of one generation with profiler ranges, profile one run and keep the
raw Kineto events (kernels, runtime calls, annotations) for offline analysis."""

from __future__ import annotations

import gzip
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.autograd.profiler import record_function

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def load(batch: int = 1, optimized: bool = False, patches: int = 60, **options: Any) -> Any:
    from kernel_agent.workloads import WorkloadSpec, create_workload

    opts: dict[str, Any] = {"patches": patches, **options}
    if batch > 1:
        opts.update(batch_size=batch, metric="throughput")
    spec = WorkloadSpec(repo_id="openbmb/VoxCPM2", modality="tts", options=opts, family="voxcpm")
    w = create_workload(spec)
    t0 = time.perf_counter()
    w.load()
    print(f"loaded in {time.perf_counter() - t0:.1f} s", flush=True)
    if optimized:
        pkg = HERE / ("opt_b1" if batch == 1 else "opt_b16")
        sys.path.insert(0, str(pkg))
        import apply  # type: ignore[import-not-found]

        counts = apply.apply_kernels(w.model)
        done = apply.apply_transforms(w.model, workload=w)
        print("kernels", counts, "transforms", len(done), flush=True)
        sys.path.remove(str(pkg))
    return w


def _wrap(obj: Any, attr: str, label: Any) -> None:
    """Instance-level wrapper of ``obj.attr`` in a profiler range (``label``: str or
    callable(args) -> str)."""
    inner = getattr(obj, attr)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        name = label(args, kwargs) if callable(label) else label
        with record_function(f"stage::{name}"):
            return inner(*args, **kwargs)

    setattr(obj, attr, wrapped)


def tag_stages(w: Any) -> None:
    """Profiler ranges around every stage of one generation (applied after the optimised
    set, so they wrap whatever the transforms installed)."""
    m = w.model

    def enc_label(args: tuple, kwargs: dict) -> str:
        x = args[0] if args else next(iter(kwargs.values()))
        return "locenc" if x.dim() == 4 and x.size(1) == 1 else "prefill.locenc"

    _wrap(m.feat_encoder, "forward", enc_label)
    _wrap(m.feat_decoder, "forward", "dit")
    _wrap(m.audio_vae, "decode", "vae")
    _wrap(m.base_lm, "forward", "prefill.base_lm")
    _wrap(m.residual_lm, "forward", "prefill.residual_lm")
    from kernel_agent.workloads.voxcpm_batch import VoxCPMBatchWorkload

    if isinstance(w, VoxCPMBatchWorkload):
        inner = w.lm_step

        def lm_step(lm: Any, *a: Any, **k: Any) -> Any:
            name = "lm.base" if lm is m.base_lm else "lm.residual"
            with record_function(f"stage::{name}"):
                return inner(lm, *a, **k)

        w.lm_step = lm_step
    else:
        _wrap(m.base_lm, "forward_step", "lm.base")
        _wrap(m.residual_lm, "forward_step", "lm.residual")
    for name in ("lm_to_dit_proj", "res_to_dit_proj", "enc_to_lm_proj", "fusion_concat_proj"):
        _wrap(getattr(m, name), "forward", "proj")
    _wrap(m.fsq_layer, "forward", "proj")
    for name in ("stop_proj", "stop_head"):
        _wrap(getattr(m, name), "forward", "stop")


def profile_run(w: Any, inputs: Any, tag: str) -> dict[str, Any]:
    """One profiled run (after the caller's warm-up); keeps every Kineto event."""
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode(), profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
    ) as prof:
        with record_function("stage::run"):
            w.run(inputs)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1e3
    trace = OUT / f"{tag}.trace.json"
    prof.export_chrome_trace(str(trace))
    raw = json.loads(trace.read_text())
    trace.unlink()
    events = []
    names: dict[str, int] = {}
    keep = {"kernel", "gpu_memcpy", "gpu_memset", "cuda_runtime", "cuda_driver",
            "user_annotation"}
    for ev in raw.get("traceEvents", []):
        kind = ev.get("cat")
        if kind not in keep or ev.get("ph") != "X":
            continue
        name = ev.get("name", "")
        if kind == "user_annotation" and not name.startswith("stage::"):
            continue
        args = ev.get("args") or {}
        meta = None
        if kind == "kernel":
            meta = {k: args.get(k) for k in ("grid", "block", "blocks per SM", "warps per SM",
                                             "registers per thread", "shared memory",
                                             "est. achieved occupancy %", "graph id")}
        start = int(round(float(ev["ts"]) * 1000))
        stop = start + int(round(float(ev.get("dur", 0)) * 1000))
        resource = args.get("stream", ev.get("tid")) if kind in {"kernel", "gpu_memcpy",
                                                                  "gpu_memset"} else ev.get("tid")
        idx = names.setdefault(name, len(names))
        events.append((kind, idx, start, stop, int(args.get("correlation", -1)),
                       int(resource) if isinstance(resource, int) else hash(resource), meta))
    data = {"tag": tag, "wall_ms": wall, "names": list(names), "events": events}
    path = OUT / f"{tag}.pkl.gz"
    with gzip.open(path, "wb") as fh:
        pickle.dump(data, fh)
    print(f"profiled {tag}: wall {wall:.1f} ms, {len(events)} events -> {path}", flush=True)
    return data


def timed(w: Any, inputs: Any, iters: int = 3) -> list[float]:
    from kernel_agent.kernels.bench import warm_gpu

    warm_gpu(500.0)
    out = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            w.run(inputs)
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def async_stop_variant(w):
    """The workload's batched loop with the stop flag computed at the top of the patch (it
    depends on lm_hidden only), copied to pinned memory without blocking, and read after the
    patch's LocDiT and LocEnc are queued (as skip_dead_work does at batch 1)."""
    import inspect
    import textwrap
    import types

    from kernel_agent.workloads import voxcpm_batch as vb

    src = textwrap.dedent(inspect.getsource(vb.VoxCPMBatchWorkload._generate_batch))
    a = """        for i in range(n):
            dit_hidden"""
    b = """        _stop_host = torch.empty(batch, dtype=torch.int64).pin_memory()
        _ready = torch.cuda.Event()
        for i in range(n):
            _logits = model.stop_head(model.stop_actn(model.stop_proj(lm_hidden)))
            _stop_host.copy_(_logits.argmax(dim=-1), non_blocking=True)
            _ready.record()
            dit_hidden"""
    c = """            logits = model.stop_head(model.stop_actn(model.stop_proj(lm_hidden)))
            stop = logits.argmax(dim=-1).cpu()"""
    d = """            logits = _logits
            _ready.synchronize()
            stop = _stop_host.clone()"""
    src = textwrap.dedent(src)
    # the dedented source is indented 4 less than the class body
    a, b, c, d = (x.replace("\n    ", "\n")[4:] if x.startswith("    ") else x for x in (a, b, c, d))
    assert a in src and c in src, "loop text changed"
    src = src.replace(a, b).replace(c, d)
    ns = dict(vars(vb))
    exec(compile(src, "<async_stop>", "exec"), ns)
    return types.MethodType(ns["_generate_batch"], w)
