"""The decode-step opcodes of the megakernel kit on the GPU (issue #225, milestones 6 and 7;
``gpu``, not in the CPU suite): split-KV attention and its combine against torch across KV
lengths read from the device, the RoPE / KV append, SwiGLU, embedding and argmax + token
advance opcodes, the example's decode step against its torch reference and its graph
baseline, determinism, and compute-sanitizer memcheck / racecheck / synccheck."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
import torch

from kernel_agent import toolchain
from kernel_agent.agent.prompts import EXAMPLES_DIR
from kernel_agent.native import project
from kernel_agent.native.megakernel import decode as mkd
from kernel_agent.native.megakernel import runtime, simulate
from kernel_agent.native.megakernel import schedule as mks

pytestmark = pytest.mark.gpu
EXAMPLE = EXAMPLES_DIR / "native_megakernel"


@pytest.fixture(scope="module")
def mk():
    """The example's entry module and its compiled extension."""
    toolchain.setup()
    module = project.import_project(EXAMPLE)
    return module, module.extension()


def _run(ext, sched, tensors, **kw) -> runtime.Runtime:
    pages, _, page_bytes, _ = ext.mk_info(True)
    rt = runtime.Runtime(sched, tensors, pool_bytes=pages * page_bytes, **kw)
    ext.mk_run(*rt.args(), pages, sched.n_queues, 1, True)
    torch.cuda.synchronize()
    return rt


def _ulp(dtype: torch.dtype) -> float:
    return 2.0**-7 if dtype == torch.bfloat16 else 2.0**-10


# ------------------------------------------------------------------ split-KV attention

#: (kv heads, q heads per kv head, head dim, capacity, splits, chunk, dtype, KV lengths):
#: lengths of 1, not a multiple of the chunk (or of the granule) and the maximum; GQA groups
#: of 1 to 7 (7: two q blocks of 4 + 3), every head dim, both dtypes, balanced and fixed chunks
ATTENTION_CASES = [
    (4, 4, 64, 4096, 18, 0, torch.bfloat16, [1, 2, 15, 16, 17, 255, 257, 1000, 4095, 4096]),
    (4, 4, 64, 4096, 16, 256, torch.bfloat16, [1, 17, 255, 256, 257, 3000, 4096]),
    (2, 7, 64, 1024, 8, 0, torch.float16, [1, 33, 1023, 1024]),
    (8, 2, 128, 2048, 9, 0, torch.bfloat16, [1, 5, 129, 2047, 2048]),
    (1, 1, 256, 512, 5, 0, torch.float16, [1, 31, 500, 512]),
    (2, 4, 128, 1000, 4, 250, torch.float16, [1, 250, 251, 999, 1000]),
]


@pytest.mark.parametrize("case", ATTENTION_CASES, ids=lambda c: f"g{c[1]}-d{c[2]}-{c[6]}")
def test_split_kv_attention_against_torch_at_lengths_read_from_the_device(mk, case):
    """One schedule (attention tiles + their combine), the KV length a device value changed
    between launches: every length matches the fp32 softmax of torch within one ulp of the
    output's scale, and a repeated launch gives the same bits."""
    mod, ext = mk
    hkv, group, d, cap, splits, chunk, dtype, lengths = case
    _, queues, _, _ = ext.mk_info(True)
    torch.manual_seed(hkv * 100 + d)
    hq = hkv * group
    split = mkd.SplitKV(hkv, group, d, cap, splits, chunk)
    q = torch.randn(hq * d, device="cuda").to(dtype)
    k = torch.randn(hkv, cap, d, device="cuda").to(dtype)
    v = torch.randn(hkv, cap, d, device="cuda").to(dtype)
    state = torch.zeros(mkd.STEP_WORDS, dtype=torch.int32, device="cuda")
    po = torch.zeros(split.partial_rows, d, device="cuda")
    pml = torch.zeros(split.partial_rows, 2, device="cuda")
    out = torch.zeros(hq * d, dtype=dtype, device="cuda")
    length = {"length": 3, "length_off": mkd.STEP_POS, "length_add": 1}
    common = {"splits": splits, "chunk": chunk, **length, "dim": d, "dtype": mod.DTYPES[dtype]}
    attn, comb, edge = split.ops(
        "attn", "combine", (mod.ATTN_DECODE, mod.ATTN_COMBINE),
        lambda a: mod.attn_args(
            q=0, q_off=0, k=1, v=2, cap=cap, kv_head=a.kv_head, q0=a.q0, qn=a.qn, split=a.split,
            scale=d**-0.5, po=4, pml=5, prow0=0, **common,
        ),
        lambda h, q0, qn: mod.combine_args(
            po=4, pml=5, prow0=0, cap=cap, q0=q0, qn=qn, out=6, out_off=0, **common
        ),
    )  # fmt: skip
    sched = mks.build([attn, comb], [edge], queues)
    assert simulate.check(sched, draws=50) == []
    pages, _, page_bytes, _ = ext.mk_info(True)
    rt = runtime.Runtime(sched, [q, k, v, state, po, pml, out], pool_bytes=pages * page_bytes)
    for n in lengths:
        state[mkd.STEP_POS] = n - 1
        ext.mk_run(*rt.args(), pages, queues, 1, True)
        first = out.clone()
        ext.mk_run(*rt.args(), pages, queues, 1, True)
        torch.cuda.synchronize()
        rt.check()
        assert torch.equal(first, out), n  # deterministic
        scores = q.float().view(hkv, group, d) @ k[:, :n].float().transpose(1, 2) * d**-0.5
        want = (scores.softmax(-1) @ v[:, :n].float()).reshape(-1)
        err = (out.float() - want).abs().max().item()
        assert err <= _ulp(dtype) * max(1.0, want.abs().max().item()), (n, err)
    assert rt.counters.abs().sum() == 0


def test_an_attention_layout_the_opcode_lacks_stops_with_its_instruction(mk):
    """Head dim 96 (no instantiation): the interpreter stops with status bad_opcode and the
    instruction, instead of writing garbage."""
    mod, ext = mk
    buf = torch.zeros(4096, device="cuda")
    state = torch.zeros(2, dtype=torch.int32, device="cuda")
    args = mod.attn_args(
        q=0, q_off=0, k=0, v=0, cap=16, kv_head=0, q0=0, qn=1, split=0, splits=1, chunk=0,
        length=1, length_off=1, length_add=1, scale=1.0, po=0, pml=0, prow0=0, dim=96,
    )  # fmt: skip
    sched = mks.build([mks.Op("attn", mod.ATTN_DECODE, 1, args=[args])], [], 1)
    rt = _run(ext, sched, [buf, state])
    info = rt.failure()
    assert info is not None and info["code"] == "bad_opcode" and info["op"] == "attn", info


# ------------------------------------------------------------------ RoPE / KV append, SwiGLU, embed


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rope_kv_glu_and_embed_round_like_eager_torch(mk, dtype):
    """One KV head per RoPE tile: q and k rotated at the device's position exactly as
    ``q * cos + rotate_half(q) * sin`` in the dtype, k and v appended at the position, nothing
    else of the caches touched; the embedding of the device's token; the decode step's SwiGLU
    (the generic GLU opcode, bf16) as ``silu(gate) * up``."""
    mod, ext = mk
    torch.manual_seed(3)
    hq, hkv, d, cap, inter, vocab = 8, 2, 64, 32, 512, 100
    pos, token = 19, 77
    dev = {"device": "cuda"}
    qkv = torch.randn((hq + 2 * hkv) * d, **dev).to(dtype)
    freqs = torch.outer(torch.arange(cap, dtype=torch.float32), torch.rand(d // 2) * 0.5).cuda()
    cos, sin = freqs.cos().contiguous(), freqs.sin().contiguous()
    kc = torch.randn(hkv, cap, d, **dev).to(dtype)
    vc = torch.randn(hkv, cap, d, **dev).to(dtype)
    k0, v0 = kc.clone(), vc.clone()
    q_out = torch.zeros(hq * d, dtype=dtype, **dev)
    gu = torch.randn(2 * inter, **dev).to(dtype) * 3
    h = torch.zeros(inter, dtype=dtype, **dev)
    table = torch.randn(vocab, 256, **dev).to(dtype)
    row = torch.zeros(256, dtype=dtype, **dev)
    state = torch.tensor([token, pos], dtype=torch.int32, **dev)
    dt = mod.DTYPES[dtype]
    tensors = [qkv, cos, sin, kc, vc, q_out, gu, h, table, row, state]
    ops = [
        mks.Op("rope", mod.ROPE_KV, hkv, args=lambda g: mod.rope_kv_args(
            qkv=0, qkv_off=0, qheads=hq, kvheads=hkv, kv_head=g, group=hq // hkv, dim=d, qout=5,
            qout_off=0, k=3, v=4, cap=cap, pos=10, pos_off=1, cos=1, sin=2, dtype=dt,
        )),
        mks.Op("embed", mod.EMBED, 1, args=[mod.embed_args(
            table=8, vocab=vocab, dim=256, esize=2, token=10, token_off=0, out=9, out_off=0,
        )]),
    ]  # fmt: skip
    if dtype == torch.bfloat16:
        ops.append(mks.Op("glu", mod.GLU, 2, args=lambda t: mod.glu_args(
            a=6, a_off=0, b=6, b_off=inter, out=7, out_off=0, i0=t * 256, n=256, act=mod.GLU_SILU,
        )))  # fmt: skip
    _, queues, _, _ = ext.mk_info(True)
    _run(ext, mks.build(ops, [], queues), tensors).check()
    c = torch.cat((cos[pos], cos[pos])).to(dtype)
    s = torch.cat((sin[pos], sin[pos])).to(dtype)
    q = qkv[: hq * d].view(hq, d)
    k = qkv[hq * d : (hq + hkv) * d].view(hkv, d)
    v = qkv[(hq + hkv) * d :].view(hkv, d)
    assert torch.equal(q_out.view(hq, d), q * c + mod.rotate_half(q) * s)
    k0[:, pos], v0[:, pos] = k * c + mod.rotate_half(k) * s, v
    assert torch.equal(kc, k0) and torch.equal(vc, v0)
    assert torch.equal(row, table[token])
    if dtype == torch.bfloat16:
        gate, up = gu[:inter], gu[inter:]
        assert torch.equal(h, torch.nn.functional.silu(gate) * up)


def test_argmax_advances_the_step_state_on_the_device(mk):
    """The first maximum (torch.argmax: NaN first) becomes the token, the position moves on,
    the history records the token there; a full history is not written past its end."""
    mod, ext = mk
    logits = torch.randn(4096, device="cuda").to(torch.bfloat16)
    logits[1234] = logits[3000] = 9.0  # a tie: the first index wins
    state = torch.tensor([5, 7], dtype=torch.int32, device="cuda")
    hist = torch.zeros(9, dtype=torch.int32, device="cuda")
    best = torch.zeros(1, dtype=torch.int64, device="cuda")
    args = mod.argmax_args(
        x=0, x_off=0, n=4096, out=1, out_off=0, advance=1, step=2, step_off=0, hist=3, hist_len=9
    )
    sched = mks.build([mks.Op("argmax", mod.ARGMAX, 1, args=[args])], [], 1)
    rt = _run(ext, sched, [logits, best, state, hist])
    rt.check()
    assert int(best) == int(torch.argmax(logits.float())) == 1234
    assert state.tolist() == [1234, 8] and hist.tolist() == [0] * 8 + [1234]
    logits[77] = float("nan")
    ext.mk_run(*rt.args(), *ext.mk_info(True)[:1], 1, 1, True)
    torch.cuda.synchronize()
    assert int(best) == 77 and state.tolist() == [77, 9] and hist.tolist() == [0] * 8 + [1234]
    plain = mod.argmax_args(x=0, x_off=0, n=4096, out=1, out_off=0)  # no advance
    _run(ext, mks.build([mks.Op("a", mod.ARGMAX, 1, args=[plain])], [], 1), [logits, best])
    assert int(best) == 77 and state.tolist() == [77, 9]


# ------------------------------------------------------------------ the decode step


def _decoder(mod, layers=2, capacity=512, **kw):
    torch.manual_seed(0)
    cfg = {"vocab": 2048, "hidden": 1024, "layers": layers, "heads": 16, "kv_heads": 4,
           "head_dim": 64, "inter": 2048, "capacity": capacity, **kw}  # fmt: skip
    return mod.TinyDecoder(**cfg).cuda().to(torch.bfloat16).eval().randomize_(1)


@pytest.fixture(scope="module")
def engines(mk):
    mod, _ = mk
    ref = _decoder(mod)
    return ref, mod.build_decode(ref, "megakernel"), mod.build_decode(ref, "graph")


def _close(got: torch.Tensor, want: torch.Tensor) -> None:
    torch.testing.assert_close(got.float(), want.float(), atol=2e-2, rtol=2 * _ulp(got.dtype))


@pytest.mark.parametrize("length", [1, 2, 17, 100, 511, 512])
def test_one_decode_step_matches_the_reference_at_any_kv_length(mk, engines, length):
    """From the same caches (a random prefill) and state: the logits, the appended K / V and
    the advanced state of the megakernel match the torch reference (bf16 rounding of the
    projections), and the graph of per-op launches gives the megakernel's bits."""
    ref, mega, graph = engines
    kc, vc = ref.new_cache()
    kc.normal_()
    vc.normal_()
    pos, token = length - 1, 7 + length
    rk, rv = kc.clone(), vc.clone()
    want = ref.step(token, pos, rk, rv)
    for engine in (mega, graph):
        engine.reset(token, pos, (kc, vc))
        engine.step()
        torch.cuda.synchronize()
        engine.check()
    _close(mega.logits, want)
    _close(mega.k_cache, rk)
    _close(mega.v_cache, rv)
    best = int(mega.best)
    assert mega.state.tolist() == [best, pos + 1] and mega.position == pos + 1
    assert want[best] >= want.max() - 0.05  # the reference's argmax, or a bf16 near tie
    if pos + 1 < ref.capacity:
        assert int(mega.history[pos + 1]) == best and int(mega.history[pos]) == token
    for name in ("logits", "k_cache", "v_cache", "state", "history", "best"):
        assert torch.equal(getattr(mega, name), getattr(graph, name)), name


def test_generation_follows_the_reference_token_by_token(engines):
    """300 steps from position 0, no host value between them: every token the megakernel
    chose is the reference's argmax on the same prefix, or within bf16 rounding of it (a near
    tie), and the graph baseline chose the same tokens."""
    ref, mega, graph = engines
    steps = 300
    for engine in (mega, graph):
        engine.reset(3, 0, ref.new_cache())
    tokens = mega.generate(steps).tolist()
    assert graph.generate(steps).tolist() == tokens
    rk, rv = ref.new_cache()
    ties = 0
    for pos in range(steps):
        logits = ref.step(tokens[pos], pos, rk, rv).float()
        chosen = tokens[pos + 1]
        if chosen != int(logits.argmax()):
            ties += 1
            assert logits.max() - logits[chosen] <= 0.05, (pos, chosen)
    assert ties <= steps // 20, ties
    # the second layer's K carries the first layer's rounding of 300 steps (the GPU tests'
    # tolerance of the deep chain)
    torch.testing.assert_close(mega.k_cache.float(), rk.float(), atol=0.1, rtol=0.05)
    assert mega.position == steps and mega.state.tolist() == [tokens[steps], steps]


def test_decode_steps_are_bit_identical_run_to_run(engines):
    """The same start twice: 64 steps give the same tokens, logits and caches, bit for bit."""
    ref, mega, _ = engines
    runs = []
    for _ in range(2):
        mega.reset(11, 0, ref.new_cache())
        mega.generate(64)
        torch.cuda.synchronize()
        runs.append([t.clone() for t in (mega.history, mega.logits, mega.k_cache, mega.v_cache)])
    assert all(torch.equal(a, b) for a, b in zip(*runs, strict=True))


def test_a_full_cache_is_not_stepped_past(engines):
    ref, mega, _ = engines
    mega.reset(1, ref.capacity - 1, ref.new_cache())
    mega.step()
    with pytest.raises(ValueError, match="full"):
        mega.step()
    with pytest.raises(ValueError, match="outside"):
        mega.reset(1, ref.capacity)


def test_the_trace_shows_the_decode_steps_ops(mk):
    mod, _ = mk
    engine = mod.build_decode(_decoder(mod, layers=1, capacity=256), trace=True)
    engine.reset(5, 200)
    for _ in range(3):
        engine.step()
    torch.cuda.synchronize()
    summary = engine.trace_summary()
    assert {"embed", "l0.rope", "l0.attn", "l0.combine", "argmax"} <= set(summary["ops"])
    assert summary["span_ns"] > 0
    assert summary["ops"]["l0.qkv"]["overlap"] is not None


# ------------------------------------------------------------------ compute-sanitizer

SANITIZED = """
import sys

import torch
from kernel_agent import toolchain
from kernel_agent.agent.prompts import EXAMPLES_DIR
from kernel_agent.native import project
from kernel_agent.native.megakernel import decode as mkd, runtime
from kernel_agent.native.megakernel import schedule as mks

toolchain.setup()
mod = project.import_project(EXAMPLES_DIR / "native_megakernel")
ext = mod.extension()
pages, queues, page_bytes, _ = ext.mk_info(True)
torch.manual_seed(0)
broken = sys.argv[1:] == ["--out-of-bounds"]
# split-KV attention alone: fp16, head dim 64, a group of 7 (two q blocks), 3 balanced splits
hkv, group, d, cap = 2, 7, 64, 256
split = mkd.SplitKV(hkv, group, d, cap, 3)
k = torch.randn(hkv, cap // 2 if broken else cap, d, device="cuda").half()  # broken: too short
q, v = torch.randn(hkv * group * d, device="cuda").half(), torch.randn(hkv, cap, d).cuda().half()
state = torch.tensor([0, cap - 1], dtype=torch.int32, device="cuda")
po = torch.zeros(split.partial_rows, d, device="cuda")
pml = torch.zeros(split.partial_rows, 2, device="cuda")
out = torch.zeros(hkv * group * d, device="cuda").half()
common = dict(splits=3, chunk=0, length=3, length_off=1, length_add=1, dim=d, dtype=1)
attn, comb, edge = split.ops(
    "attn", "combine", (mod.ATTN_DECODE, mod.ATTN_COMBINE),
    lambda a: mod.attn_args(q=0, q_off=0, k=1, v=2, cap=cap, kv_head=a.kv_head, q0=a.q0, qn=a.qn,
                            split=a.split, scale=0.125, po=4, pml=5, prow0=0, **common),
    lambda h, q0, qn: mod.combine_args(po=4, pml=5, prow0=0, cap=cap, q0=q0, qn=qn, out=6,
                                       out_off=0, **common),
)
# broken: the attention tiles alone (memcheck stops a faulting warp: a combine waiting on its
# tiles' counter would wait for the watchdog, 10 minutes under the sanitizers)
sched = mks.build([attn], [], queues) if broken else mks.build([attn, comb], [edge], queues)
rt = runtime.Runtime(sched, [q, k, v, state, po, pml, out], pool_bytes=pages * page_bytes)
ext.mk_run(*rt.args(), pages, queues, 1, True)
torch.cuda.synchronize()
if not broken:
    # the decode step: head dim 128, 5 splits of 64 keys (two batches each) at length 251
    ref = mod.TinyDecoder(vocab=1024, hidden=1024, layers=1, heads=8, kv_heads=2, head_dim=128,
                          inter=1024, capacity=300).cuda().to(torch.bfloat16).eval().randomize_(1)
    for mode in ("megakernel", "graph"):
        engine = mod.build_decode(ref, mode, splits=5)
        engine.reset(3, 250)
        engine.generate(3)
        torch.cuda.synchronize()
        engine.check()
print("SANITIZED-OK", flush=True)
"""


def _sanitize(tmp_path, tool: str, *args: str):
    from kernel_agent.kernels import memcheck

    sanitizer = toolchain.sanitizer()
    if not sanitizer.path:
        pytest.skip(sanitizer.reason)
    script = tmp_path / "decode.py"
    script.write_text(textwrap.dedent(SANITIZED))
    log = tmp_path / f"{tool}.log"
    cmd = [sanitizer.path, *memcheck.TOOL_ARGS[tool], "--log-file", str(log), sys.executable,
           str(script), *args]  # fmt: skip
    env = {**memcheck.child_env(), **memcheck.ENV}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    return proc, log.read_text(errors="replace") if log.exists() else ""


@pytest.mark.parametrize("tool", ["memcheck", "racecheck", "synccheck"])
def test_the_decode_step_is_clean_under_compute_sanitizer(tmp_path, tool):
    """Split-KV attention (fp16, head dim 64, GQA 7) and a decode step (head dim 128, chunks of
    two batches, the page pool's producer warp, the advance) and its graph baseline under each
    tool: no error, and the tool saw the kernels."""
    from kernel_agent.kernels import memcheck

    proc, text = _sanitize(tmp_path, tool)
    assert "SANITIZED-OK" in proc.stdout, (proc.stdout[-2000:], proc.stderr[-2000:], text[-2000:])
    if tool == "racecheck":
        errors, warnings, report = memcheck.parse_race_log(text)
        assert (errors, warnings) == (0, 0), report
    else:
        errors, report = memcheck.parse_log(text)
        assert errors == 0, report
    assert not memcheck.unchecked(text), text[-1000:]


def test_memcheck_sees_an_attention_tile_read_past_its_cache(tmp_path):
    """The same run with a K cache half as long as the schedule says: memcheck reports the
    attention's reads past it (the clean runs above are not clean by not looking)."""
    from kernel_agent.kernels import memcheck

    _, text = _sanitize(tmp_path, "memcheck", "--out-of-bounds")
    errors, report = memcheck.parse_log(text)
    assert errors > 0 and "Invalid __global__ read" in report, text[-2000:]
