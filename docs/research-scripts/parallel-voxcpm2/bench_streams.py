"""Two streams running two VoxCPM2 stages concurrently, on the accepted optimised set (#136).

1. Stage alone vs pairs of stages on two streams (issued from one host thread, interleaved),
   against the same interleaving on one stream: what concurrency buys per pair.
2. Latency: the AudioVAE decode of finished patches on a second stream during the patch loop
   (streaming decode, no per-chunk host sync) vs the full decode after the loop (VoxCPM's
   generate) vs VoxCPM's own generate_streaming (decode + .cpu() per chunk on one stream).
3. Throughput at batch 1: R requests back to back, request r's AudioVAE decode on a second
   stream while request r+1 prefills and decodes, vs sequential.
"""

from __future__ import annotations

import json
import statistics
import time

import torch

import common

w = common.load(1, optimized=True, patches=60)
m = w.model
inputs = w.make_inputs()
with torch.inference_mode():
    for _ in range(2):
        w.run(inputs)
torch.cuda.synchronize()
res: dict = {}


def clone(x):
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    if isinstance(x, (list, tuple)):
        return type(x)(clone(v) for v in x)
    if isinstance(x, dict):
        return {k: clone(v) for k, v in x.items()}
    return x


captured: dict = {}
restore = []


def grab(obj, attr, key, when=lambda a, k: True):
    had = attr in vars(obj)
    inner = getattr(obj, attr)

    def f(*a, **k):
        if key not in captured and when(a, k):
            captured[key] = (clone(a), clone(k))
        return inner(*a, **k)

    setattr(obj, attr, f)
    restore.append((obj, attr, inner, had))


grab(m.feat_decoder, "forward", "dit")
grab(m.base_lm, "forward_step", "lm_base")
grab(m.residual_lm, "forward_step", "lm_res")
grab(m.feat_encoder, "forward", "locenc", lambda a, k: a[0].dim() == 4 and a[0].size(1) == 1)
grab(m.audio_vae, "decode", "vae")
with torch.inference_mode():
    out = w.run(inputs)
for obj, attr, inner, had in reversed(restore):
    if had:
        setattr(obj, attr, inner)
    else:
        delattr(obj, attr)
latents = out["latents"]  # [patches, 1, d, p]
print({k: [tuple(t.shape) for t in v[0] if isinstance(t, torch.Tensor)] for k, v in captured.items()})


def call(key):
    a, k = captured[key]
    fn = {"dit": m.feat_decoder, "locenc": m.feat_encoder, "vae": m.audio_vae.decode,
          "lm_base": m.base_lm.forward_step, "lm_res": m.residual_lm.forward_step}[key]
    return lambda: fn(*a, **k)


# streaming chunk decode: VoxCPM's stateful decoder, one patch latent per call
chunk_lat = captured["vae"][0][0][..., :4].contiguous()  # [1, d, 4] = one patch
fns = {k: call(k) for k in ("dit", "lm_base", "lm_res", "locenc", "vae")}
dec = None
fns["vae_chunk"] = lambda: dec.decode_chunk(chunk_lat)


def ev(fn, reps=20):
    with torch.inference_mode():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        ts, hs = [], []
        for _ in range(reps):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            t0 = time.perf_counter()
            fn()
            hs.append((time.perf_counter() - t0) * 1e3)
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b))
    return statistics.median(ts), statistics.median(hs)


from kernel_agent.kernels.bench import warm_gpu

warm_gpu(500.0)
alone = {}
for k, fn in fns.items():
    if k == "vae_chunk":
        continue
    gpu_ms, host_ms = ev(fn)
    alone[k] = {"ms": gpu_ms, "host_ms": host_ms}


def issue(fa, na, fb, nb, sa, sb):
    ia = ib = 0
    while ia < na or ib < nb:
        if ia < na and (ib >= nb or ia / na <= ib / nb):
            with torch.cuda.stream(sa):
                fa()
            ia += 1
        else:
            with torch.cuda.stream(sb):
                fb()
            ib += 1


def pair(a, na, b, nb, reps=5):
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    out = {}
    with torch.inference_mode():
        for mode in ("one_stream", "two_streams", "one_stream", "two_streams"):
            walls = []
            for _ in range(reps):
                torch.cuda.synchronize()
                cur = torch.cuda.current_stream()
                t0 = time.perf_counter()
                s1.wait_stream(cur)
                s2.wait_stream(cur)
                issue(fns[a], na, fns[b], nb, s1, s1 if mode == "one_stream" else s2)
                cur.wait_stream(s1)
                cur.wait_stream(s2)
                torch.cuda.synchronize()
                walls.append((time.perf_counter() - t0) * 1e3)
            out.setdefault(mode, []).append(statistics.median(walls))
    one, two = min(out["one_stream"]), min(out["two_streams"])
    sum_alone = na * alone[a]["ms"] + nb * alone[b]["ms"]
    small = min(na * alone[a]["ms"], nb * alone[b]["ms"])
    r = {"a": f"{a} x{na}", "b": f"{b} x{nb}", "sum_alone_ms": sum_alone, "one_stream_ms": one,
         "two_streams_ms": two, "speedup": one / two,
         "hidden_share_of_smaller": (one - two) / small}
    print(json.dumps(r), flush=True)
    return r


warm_gpu(300.0)
res["pairs"] = [
    pair("dit", 10, "vae", 4),
    pair("lm_base", 20, "vae", 4),
    pair("dit", 10, "lm_base", 20),
    pair("dit", 10, "locenc", 20),
    pair("lm_base", 20, "lm_res", 20),
]
# the stateful streaming decoder (eager 3-D decoder inside the context)
stream_ctx = m.audio_vae.streaming_decode()
dec = stream_ctx.__enter__()
gpu_ms, host_ms = ev(fns["vae_chunk"])
alone["vae_chunk"] = {"ms": gpu_ms, "host_ms": host_ms}
res["pairs"] += [pair("dit", 10, "vae_chunk", 10), pair("lm_base", 20, "vae_chunk", 10)]
stream_ctx.__exit__(None, None, None)
print("alone", json.dumps(alone, indent=1), flush=True)
res["alone"] = alone

# ------------------------------------------------------------------ end to end

def prep(text):
    tok = torch.LongTensor(m.text_tokenizer(text))
    tok = torch.cat([tok, torch.tensor([m.audio_start_token], dtype=torch.int32)])
    n = tok.shape[0]
    feat = torch.zeros((n, m.patch_size, m.audio_vae.latent_dim), dtype=torch.float32)
    dev = m.device
    return (tok.unsqueeze(0).to(dev), torch.ones(n, dtype=torch.int32).unsqueeze(0).to(dev),
            feat.unsqueeze(0).to(dev).to(torch.bfloat16),
            torch.zeros(n, dtype=torch.int32).unsqueeze(0).to(dev))


KW = dict(min_len=60, max_len=60, inference_timesteps=10, cfg_value=2.0)
TEXTS = [w.make_inputs(), "A held out sentence proves that every patch is computed.",
         "The library opens at nine and closes late on Thursdays.",
         "Rain is expected in the afternoon, so take an umbrella with you."]


def gen_full(text):
    torch.manual_seed(0)
    latent, _, _ = next(m._inference(*prep(text), streaming=False, **KW))
    return m.audio_vae.decode(latent.to(torch.float32)).squeeze(1).cpu()


def gen_voxcpm_streaming(text):
    torch.manual_seed(0)
    chunks = []
    with m.audio_vae.streaming_decode() as d:
        for lat, _, _ in m._inference(*prep(text), streaming=True, **KW):
            chunks.append(d.decode_chunk(lat.to(torch.float32)).squeeze(1).cpu())
    return torch.cat(chunks, -1)


side = torch.cuda.Stream()


def gen_side_streaming(text):
    torch.manual_seed(0)
    chunks, keep = [], []
    main = torch.cuda.current_stream()
    with m.audio_vae.streaming_decode() as d:
        for lat, _, _ in m._inference(*prep(text), streaming=True, **KW):
            side.wait_stream(main)
            keep.append(lat)
            with torch.cuda.stream(side):
                chunks.append(d.decode_chunk(lat.to(torch.float32)).squeeze(1))
    main.wait_stream(side)
    return torch.cat(chunks, -1).cpu()


def timed_e2e(fn, text, reps=3):
    ts = []
    with torch.inference_mode():
        fn(text)
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn(text)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts), ts


warm_gpu(300.0)
e2e = {}
for name, fn in (("generate_full_vae_after_loop", gen_full),
                 ("voxcpm_generate_streaming_one_stream", gen_voxcpm_streaming),
                 ("streaming_vae_on_side_stream", gen_side_streaming),
                 ("generate_full_vae_after_loop_again", gen_full)):
    best, ts = timed_e2e(fn, TEXTS[0])
    e2e[name] = {"best_ms": best, "runs": ts}
    print(name, best, ts, flush=True)
# loop alone (no VAE) for reference
def loop_only(text):
    torch.manual_seed(0)
    latent, _, _ = next(m._inference(*prep(text), streaming=False, **KW))
    return latent
best, ts = timed_e2e(loop_only, TEXTS[0])
e2e["loop_only_no_vae"] = {"best_ms": best, "runs": ts}
print("loop_only", best, flush=True)
res["e2e_latency"] = e2e


# pipelined requests at batch 1
def requests(pipelined: bool):
    outs, keep = [], []
    main = torch.cuda.current_stream()
    for text in TEXTS:
        torch.manual_seed(0)
        latent, _, _ = next(m._inference(*prep(text), streaming=False, **KW))
        if pipelined:
            side.wait_stream(main)
            keep.append(latent)
            with torch.cuda.stream(side):
                outs.append(m.audio_vae.decode(latent.to(torch.float32)).squeeze(1))
        else:
            outs.append(m.audio_vae.decode(latent.to(torch.float32)).squeeze(1).cpu())
    main.wait_stream(side)
    return [o.cpu() for o in outs]


pipe = {}
with torch.inference_mode():
    requests(True)
    for mode in (False, True, False, True):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        requests(mode)
        torch.cuda.synchronize()
        pipe.setdefault("pipelined" if mode else "sequential", []).append(
            (time.perf_counter() - t0) * 1e3)
pipe["requests"] = len(TEXTS)
print("pipelined", pipe, flush=True)
res["pipelined_batch1"] = pipe
(common.OUT / "bench_streams.json").write_text(json.dumps(res, indent=1))
