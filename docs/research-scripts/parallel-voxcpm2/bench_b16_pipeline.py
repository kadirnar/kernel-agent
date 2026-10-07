"""Throughput at batch 16 on the accepted optimised set (#136): request-level pipelining.

R batched runs back to back, (a) as the workload runs them (the AudioVAE decode of batch r
after its patch loop, on the main stream, then .cpu()), (b) pipelined: batch r's AudioVAE
decode issued on a second stream while batch r+1 prefills and decodes (the audio copied to
the host at the end). Also the VAE decode alone and a run without it (the loop alone).
"""

from __future__ import annotations

import json
import statistics
import time

import torch

import common

w = common.load(16, optimized=True, patches=60)
m = w.model
inputs = w.make_inputs()
with torch.inference_mode():
    for _ in range(2):
        w.run(inputs)
torch.cuda.synchronize()
side = torch.cuda.Stream()
pending: list = []
orig_decode = w._decode
patch = int(m.patch_size)


def async_decode(latent, steps):
    main = torch.cuda.current_stream()
    side.wait_stream(main)
    with torch.cuda.stream(side):
        z = latent[..., : max(steps) * patch].to(torch.float32)
        wav = m.audio_vae.decode(z)
    pending.append((latent, z, wav))
    return torch.zeros(len(steps), 1)


def no_decode(latent, steps):
    return torch.zeros(len(steps), 1)


def async_stop_variant():
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


orig_generate = w._generate_batch
async_generate = async_stop_variant()


def batches(n, mode):
    pending.clear()
    w._decode = {"sequential": orig_decode, "pipelined": async_decode, "no_vae": no_decode,
                 "async_stop": orig_decode, "async_stop+pipelined": async_decode}[mode]
    w._generate_batch = async_generate if mode.startswith("async_stop") else orig_generate
    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(n):
                w.run(inputs)
            torch.cuda.current_stream().wait_stream(side)
            audio = [p[2].float().cpu() for p in pending]
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3, audio
    finally:
        w._decode = orig_decode
        w._generate_batch = orig_generate


from kernel_agent.kernels.bench import warm_gpu

res = {}
R = 3
batches(1, "pipelined")
warm_gpu(500.0)
# correctness of the async stop variant: same latents as the workload's loop
with torch.inference_mode():
    torch.manual_seed(0)
    ref = w.run(inputs)["latents"]
    w._generate_batch = async_generate
    new = w.run(inputs)["latents"]
    w._generate_batch = orig_generate
    ref2 = w.run(inputs)["latents"]
res["workload_loop_deterministic"] = bool(torch.equal(ref, ref2))
res["async_stop_same_latents"] = bool(torch.equal(ref, new))
res["async_stop_max_abs_diff"] = float((ref - new).abs().max())
print("async stop identical latents:", res["async_stop_same_latents"], flush=True)
MODES = ("sequential", "pipelined", "no_vae", "async_stop", "async_stop+pipelined")
for mode in MODES + MODES:
    ms, _ = batches(R, mode)
    res.setdefault(mode, []).append(ms)
    print(mode, ms, flush=True)
# VAE decode alone (batch 16 x 60 patches)
a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
z = torch.randn(16, int(m.audio_vae.latent_dim), 60 * patch, device="cuda")
ts = []
with torch.inference_mode():
    m.audio_vae.decode(z)
    for _ in range(5):
        a.record()
        m.audio_vae.decode(z)
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
res["vae_alone_ms"] = statistics.median(ts)
res["runs_per_measure"] = R
audio_s = 16 * 60 * 0.16
for mode in MODES:
    best = min(res[mode])
    res[f"{mode}_ms_per_audio_s"] = best / (R * audio_s)
print(json.dumps(res, indent=1))
(common.OUT / "bench_b16_pipeline.json").write_text(json.dumps(res, indent=1))
