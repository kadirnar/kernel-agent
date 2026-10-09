"""Example native project: a megakernel built with kernel-agent's kit
(``kernel_agent.native.megakernel``, issue #225), next to two baselines from the same math.

Reference: ``reference.norms`` and ``reference.layers`` (``kernel_agent.selftest.
NormGemvChain``): ``x = x + W_l rmsnorm_l(x)`` for every layer l, bf16, ``W_l`` square and
bias-free with n a multiple of 256 up to 4096, one row (decode) per call: a stand-in for any
stack of decode layers (norm, projection, residual). Other row counts take the reference.

``mode`` (a ``build()`` keyword: ``sweep_candidate`` compares them):

* ``megakernel``: one launch of the interpreter (``ka_mk.cuh``) over a schedule built once
  here (:mod:`kernel_agent.native.megakernel.schedule`): one GEMV opcode per tile of ``rows``
  output rows (default :func:`tile_rows`: 16, fewer where 16 rows of weights do not fit this
  GPU's page pool) with the layer's RMSNorm fused into its prologue and the residual into its
  epilogue; a layer's tiles wait on one counter, the previous layer's (target: its tile
  count), not on a grid barrier, and every block's producer warp streams its next tiles'
  weights into the shared-memory page pool while its consumers wait and compute
  (``inflight``: pages in flight per block; 1 measured best on an RTX 5070 Ti, more pages
  in flight queue the activations' critical loads behind weight copies; on an A10, whose
  ``cp.async`` path keeps one 8 KB page per SM in flight at 1, 2 measured 1-4 % faster);
* ``graph_pdl``: one kernel per layer in a CUDA graph with programmatic dependent launch
  (weights into registers before ``ka_pdl_wait()``): the best launch-per-layer design of
  docs/PARALLEL.md §4.6. PDL needs sm_90+: on older GPUs (an A10, sm_86) the same graph has
  plain edges, each kernel starts once the previous one completed and its ``ka_pdl_wait()``
  is a no-op (correct, without the overlap); ``label`` says which one ran (:func:`graph_label`);
* ``coop_barrier``: one cooperative persistent kernel with a hand-rolled grid barrier after
  every layer (§4.6's slow persistent variant).

``schedule`` (a ``build()`` keyword): the megakernel's op DAG declared by hand
(:func:`chain_dag`, ``hand``) or built from one recorded call of the reference
(``captured``: ``kernel_agent.native.megakernel.captured`` maps its aten ops to the kit's
opcode families, fuses the norm and the residual into each GEMV, tiles them and derives the
edges from the storages: the same DAG here, from no knowledge of the module).

Every mode is captured once into a CUDA graph here; a call copies x into the activation
buffer (one row per layer, ``[layers + 1, n]``; a captured plan's input slice), replays the
graph and returns a copy of the last row. ``trace=True`` records per-instruction time
stamps of the megakernel (``KA_MK_TRACE``, set in ``kernel_project.toml``):
``module.trace_summary()``.

Opcodes (``include/mk_ops.cuh``): RMSNorm, GEMV tiles with bf16 or e4m3 weights (fused norm,
residual, split-K partials), residual add, split-K reduce, argmax (with the token advance of
a decode step), gated activation (GLU); their numbers and argument slots are the kit's
(``kernel_agent.native.megakernel.opcodes``).

**Decode step** (:class:`TinyDecoder` → :func:`build_decode`, issue #225 milestones 6 and 7):
a pre-norm GQA decoder's whole decode step as one launch of ``kernel<DecodeOps>``
(``include/mk_decode.cuh``: EMBED, ROPE_KV, ATTN_DECODE, ATTN_COMBINE next to the generic
opcodes): embedding of the token the device holds → per layer RMSNorm + QKV GEMV → RoPE +
KV append → split-KV attention (one chunk of a KV head per tile, GQA) → combine → O GEMV +
residual → RMSNorm + gate/up GEMV → SwiGLU (GLU) → down GEMV + residual → final RMSNorm + LM
head → argmax + token advance. The step state (token, position) lives on the device and the
last instruction advances it, so ``step()`` is a graph replay with no host value; the KV
length the attention reads is the device's position + 1 (:mod:`kernel_agent.native.
megakernel.decode`). ``mode="graph"``: the same opcodes, one launch per op in a CUDA graph
(kernel boundaries instead of counters: bit for bit the megakernel's results).
"""

import torch
from torch import nn
from torch.nn import functional as F

from kernel_agent.native import project
from kernel_agent.native.megakernel import decode as mkd
from kernel_agent.native.megakernel import opcodes as oc
from kernel_agent.native.megakernel import runtime, simulate
from kernel_agent.native.megakernel import schedule as mks

# The opcodes of include/mk_ops.cuh and include/mk_decode.cuh: numbers and argument slots
# (the kit's ABI)
from kernel_agent.native.megakernel.opcodes import (  # noqa: F401
    ARGMAX,
    ATTN_COMBINE,
    ATTN_DECODE,
    EMBED,
    GEMV,
    GEMV_FP8,
    GLU,
    GLU_SILU,
    NOP,
    RESIDUAL,
    RMSNORM,
    ROPE_KV,
    SPLITK_REDUCE,
    argmax_args,
    attn_args,
    combine_args,
    embed_args,
    f32_bits,
    gemv_args,
    glu_args,
    reduce_args,
    residual_args,
    rmsnorm_args,
    rope_kv_args,
)

#: GPUs this example runs on (``kernel_agent.gpu_arch.supports``).
ARCHS = "sm_80+"
ARCHS_WHY = (
    "bf16 opcodes (the chain's and the decode step's GEMV); page loads with cp.async "
    "(sm_80-sm_89) or bulk copies (sm_90+)"
)
#: GPUs its device code compiles for (sm_75: compiles, never run there).
ARCHS_COMPILES = "sm_75+"

MODES = ("megakernel", "graph_pdl", "coop_barrier")
#: Where the megakernel's op DAG comes from: declared by hand (:func:`chain_dag`) or built
#: from a recorded call of the reference (``kernel_agent.native.megakernel.captured``).
SCHEDULES = ("hand", "captured")
#: PDL (``griddepcontrol``, the programmatic stream serialization attribute) from sm_90 on.
PDL_SINCE = (9, 0)
#: What the other modes run (``Chain.label``).
LABELS = {
    "megakernel": "megakernel (counters, producer warp, page pool)",
    "coop_barrier": "coop_barrier (one cooperative kernel, grid barrier per layer)",
}


def graph_label(capability: tuple[int, int]) -> str:
    """What mode ``graph_pdl`` runs on a GPU of ``capability``: a graph with PDL edges, or
    before sm_90 the plain graph (a timing of it is not a graph + PDL number)."""
    if tuple(capability) >= PDL_SINCE:
        return "graph_pdl (graph + PDL, one kernel per layer)"
    arch = f"sm_{capability[0]}{capability[1]}"
    return f"graph_pdl (plain graph, one kernel per layer: PDL needs sm_90+, this GPU is {arch})"


#: Output rows per GEMV tile where they fit (the whole-row GEMV path takes up to 16).
ROWS = oc.ROWS


#: The dtype argument of the decode opcodes (activations and KV caches).
DTYPES = {torch.bfloat16: oc.DTYPE_BF16, torch.float16: oc.DTYPE_FP16}


def tile_rows(n: int, pool_bytes: int, want: int = ROWS) -> int:
    """The most output rows per tile, at most ``want``, that divide ``n`` and whose bf16
    weights (rows x n x 2 bytes) fit a page pool of ``pool_bytes`` (0: not one row fits).
    16 rows of a 4096-wide layer are 128 KB: more than the pool of a GPU with 99 KB of shared
    memory per block (A10 sm_86, RTX 5070 Ti sm_120: 11 pages of 8 KB), so 8 rows there; an
    A100's 163 KB holds 16."""
    return oc.tile_rows(n, n * 2, pool_bytes, want)


def chain_dag(
    n: int, layers: int, rows: int, eps: float, l2_hint: int = mks.EVICT_NORMAL
) -> tuple[list[mks.Op], list[mks.Edge]]:
    """The chain's ops and edges, declared by hand: per layer one GEMV opcode per tile of
    ``rows`` output rows with the layer's RMSNorm fused into its prologue and the residual
    into its epilogue, waiting on the previous layer (one counter, target its tile count).
    Tensor table: the layers' weights, the norms' weights, the activation buffer
    ``[layers + 1, n]`` (row l: layer l's input). ``schedule="captured"`` builds the same
    DAG from a recording of the reference instead (``kernel_agent.native.megakernel.
    captured``)."""
    act, tile_bytes = 2 * layers, rows * n * 2

    def tile(layer: int, t: int) -> list[int]:  # layer's RMSNorm fused, its residual added
        return gemv_args(
            x=act, x_off=layer * n, gamma=layers + layer, eps=eps, out=act,
            out_off=(layer + 1) * n, res=act, res_off=layer * n, row0=t * rows, rows=rows,
            k=n,
        )  # fmt: skip

    ops = [
        mks.Op(
            f"layer{layer}",
            GEMV,
            n // rows,
            cost=float(tile_bytes),
            args=lambda t, layer=layer: tile(layer, t),
            weights=lambda t, layer=layer: mks.Prefetch(layer, t * tile_bytes, tile_bytes, l2_hint),
        )
        for layer in range(layers)
    ]
    edges = [mks.Edge(f"layer{k - 1}", f"layer{k}", mks.ALL) for k in range(1, layers)]
    return ops, edges


def extension():
    """The compiled project (``project.load``: once per content digest and toolchain)."""
    return project.load(__file__)


class Chain(nn.Module):
    """The layers of ``reference`` as one CUDA graph of ``mode``'s launches (one-row calls)."""

    def __init__(
        self,
        reference: nn.Module,
        mode: str = "megakernel",
        rows: int = 0,
        trace: bool = False,
        pages: int = 0,
        inflight: int = 1,
        schedule: str = "hand",
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode {mode!r}: one of {', '.join(MODES)}")
        if schedule not in SCHEDULES:
            raise ValueError(f"schedule {schedule!r}: one of {', '.join(SCHEDULES)}")
        self.reference = reference  # other row counts (a fallback)
        self.mode, self.source = mode, schedule
        weights = [layer.weight.detach() for layer in reference.layers]
        gammas = [norm.weight.detach() for norm in reference.norms]
        self.eps = float(getattr(reference.norms[0], "variance_epsilon", 1e-6))
        self.n, self.layers = weights[0].shape[0], len(weights)
        device = weights[0].device
        with torch.inference_mode(False):  # buffers updated in place, in and out of inference
            self._weights, self._gammas = weights, gammas  # the graph holds their addresses
            self._act = torch.zeros(self.layers + 1, self.n, device=device, dtype=torch.bfloat16)
        self._in, self._out = self._act[0], self._act[self.layers]  # a captured plan: its own
        self.ext = extension()
        self.rt: runtime.Runtime | None = None
        self.inflight = inflight
        # PDL edges where the GPU has them (an emulated older GPU reports its own capability)
        capability = torch.cuda.get_device_capability(device)
        self.pdl = mode == "graph_pdl" and tuple(capability) >= PDL_SINCE
        self.label = graph_label(capability) if mode == "graph_pdl" else LABELS[mode]
        launch = getattr(self, f"_setup_{mode}")(rows, trace, pages)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.device(device):
            launch()  # warm-up: loads the kernels
            torch.cuda.synchronize()
            if self.rt is not None:
                self.rt.check()
            with torch.cuda.graph(self._graph):
                launch()

    def _setup_megakernel(self, rows: int, trace: bool, pages: int):
        fit, queues, page_bytes, max_slice = self.ext.mk_info()
        pages = min(pages, fit) if pages > 0 else fit
        n, layers = self.n, self.layers
        if self.source == "captured":
            return self._setup_captured(rows, trace, pages, queues, page_bytes)
        self.rows = rows = rows or tile_rows(n, pages * page_bytes)
        if rows < 1 or n % rows or n > max_slice:
            raise ValueError(
                f"rows {rows} must divide n {n} (at most {max_slice} columns) and fit the page "
                f"pool ({pages} pages of {page_bytes} bytes)"
            )
        # weights the L2 cannot keep stream once per call: evict them first, so the program,
        # the activations and the norms' weights stay in L2 (the GPU's own L2 size decides)
        l2 = torch.cuda.get_device_properties(self._act.device).L2_cache_size
        total = sum(w.numel() * w.element_size() for w in self._weights)
        self.l2_hint = oc.l2_hint(total, l2)
        ops, edges = chain_dag(n, layers, rows, self.eps, self.l2_hint)
        self.schedule = mks.build(ops, edges, queues, meta={"rows": rows, "pages": pages})
        self.rt = runtime.Runtime(
            self.schedule,
            [*self._weights, *self._gammas, self._act],
            trace=trace,
            pool_bytes=pages * page_bytes,
        )
        rt = self.rt
        return lambda: self.ext.mk_run(*rt.args(), pages, queues, self.inflight)

    def _setup_captured(self, rows: int, trace: bool, pages: int, queues: int, page_bytes: int):
        """The schedule of a recorded one-row call of the reference: its ops mapped to the
        kit's opcode families, fused, tiled, its edges from the storages, costs from this
        GPU's roofline (``kernel_agent.native.megakernel.captured``)."""
        from kernel_agent.native.megakernel import captured

        device = self._act.device
        probe = torch.zeros(1, self.n, device=device, dtype=torch.bfloat16)
        peaks, source = captured.peaks_here()
        self.plan = plan = captured.from_module(
            self.reference,
            (probe,),
            pool_bytes=pages * page_bytes,
            rows=rows,
            queues=queues,
            peaks=peaks,
            peaks_source=source,
            l2_bytes=torch.cuda.get_device_properties(device).L2_cache_size,
        )
        self.schedule = plan.schedule(queues, meta={"pages": pages})  # refuses unmapped ops
        self.rows = self.schedule.instrs[0].args[10]  # G_ROWS of a GEMV tile
        with torch.inference_mode(False):
            tensors = plan.tensors(device=device)
        self._in = plan.bind(tensors, plan.inputs[0]).view(-1)
        self._out = plan.bind(tensors, plan.outputs[0])
        self.rt = runtime.Runtime(
            self.schedule, tensors, trace=trace, pool_bytes=pages * page_bytes
        )
        rt = self.rt
        return lambda: self.ext.mk_run(*rt.args(), pages, queues, self.inflight)

    def _setup_graph_pdl(self, rows: int, trace: bool, pages: int):
        w, g, act, eps, pdl = self._weights, self._gammas, self._act, self.eps, self.pdl
        return lambda: self.ext.chain_pdl(w, g, act, eps, pdl)

    def _setup_coop_barrier(self, rows: int, trace: bool, pages: int):
        grid = min(self.ext.coop_blocks(self.n), self.n // 8)  # 8 rows per block at most
        device = self._act.device
        wt = torch.tensor([w.data_ptr() for w in self._weights], dtype=torch.int64, device=device)
        gt = torch.tensor([g.data_ptr() for g in self._gammas], dtype=torch.int64, device=device)
        bar = torch.zeros(2, dtype=torch.int32, device=device)
        self._coop = (wt, gt, bar)
        return lambda: self.ext.chain_coop(wt, gt, self._act, bar, self.layers, self.eps, grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() != self.n or x.dtype != torch.bfloat16 or x.device != self._act.device:
            return self.reference(x)
        if self.rt is not None:
            self.rt.check()  # a hung schedule is not launched again
        self._in.copy_(x.reshape(-1))
        self._graph.replay()
        return self._out.clone().view(x.shape)  # the buffer is reused

    def trace_summary(self) -> dict:
        """Per op: execution and wait times of the last call, the prefetch overlap."""
        if self.rt is None or self.rt.trace is None:
            return {}
        return simulate.trace_summary(self.schedule, self.rt.trace_rows())


def _supported(reference: nn.Module, rows: int) -> bool:
    layers, norms = getattr(reference, "layers", None), getattr(reference, "norms", None)
    if not isinstance(layers, nn.ModuleList) or not isinstance(norms, nn.ModuleList):
        return False
    if len(layers) == 0 or len(layers) != len(norms) or not torch.cuda.is_available():
        return False
    n = getattr(layers[0], "in_features", 0)
    return (
        n % 256 == 0
        and 256 <= n <= 4096
        and rows >= 0
        and n % max(rows, 1) == 0  # 0: tile_rows picks them
        and all(
            isinstance(layer, nn.Linear)
            and layer.bias is None
            and layer.in_features == layer.out_features == n
            and layer.weight.dtype == torch.bfloat16
            and layer.weight.is_cuda
            and layer.weight.is_contiguous()
            and layer.weight.data_ptr() % 16 == 0  # 128-bit and bulk loads
            and layer.weight.device == layers[0].weight.device
            for layer in layers
        )
        and all(
            getattr(norm, "weight", None) is not None
            and norm.weight.shape == (n,)
            and norm.weight.dtype == torch.bfloat16
            and norm.weight.device == layers[0].weight.device
            and norm.weight.data_ptr() % 16 == 0
            for norm in norms
        )
    )


def build(
    reference: nn.Module,
    mode: str = "megakernel",
    rows: int = 0,
    trace: bool = False,
    pages: int = 0,
    inflight: int = 1,
    schedule: str = "hand",
) -> nn.Module:
    """``rows``: output rows per GEMV tile (0: :func:`tile_rows`, 16 where they fit the page
    pool); ``pages``: the page pool's size (0: what fits this GPU's shared memory);
    ``inflight``: pages each producer warp keeps in flight (0: all); ``trace``:
    per-instruction time stamps (``trace_summary()``); ``schedule``: the megakernel's op DAG
    declared by hand (:func:`chain_dag`) or built from a recorded call of the reference
    (``captured``). A :class:`TinyDecoder` reference gets its decode step
    (:func:`build_decode`, the megakernel)."""
    if isinstance(reference, TinyDecoder) and mode == "megakernel":
        return build_decode(reference, trace=trace, pages=pages, inflight=inflight)
    if not _supported(reference, rows):
        return reference
    return Chain(reference, mode, rows, trace, pages, inflight, schedule)


# ------------------------------------------------------------------ the decode step


class RMSNorm(nn.Module):
    """RMSNorm as eager decoders compute it: normalised in fp32, rounded to the dtype, times
    the weight (the GEMV opcode's fused norm rounds the same way)."""

    def __init__(self, hidden: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * h.to(x.dtype)


class DecoderLayer(nn.Module):
    """Pre-norm attention (fused QKV projection) and SwiGLU MLP (fused gate / up), bias-free."""

    def __init__(
        self, hidden: int, heads: int, kv_heads: int, head_dim: int, inter: int, eps: float
    ) -> None:
        super().__init__()
        self.input_norm = RMSNorm(hidden, eps)
        self.qkv = nn.Linear(hidden, (heads + 2 * kv_heads) * head_dim, bias=False)
        self.o = nn.Linear(heads * head_dim, hidden, bias=False)
        self.post_norm = RMSNorm(hidden, eps)
        self.gate_up = nn.Linear(hidden, 2 * inter, bias=False)
        self.down = nn.Linear(inter, hidden, bias=False)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class TinyDecoder(nn.Module):
    """The decode step's torch reference: a pre-norm decoder at batch 1, one token per
    step, greedy. Per layer: RMSNorm, fused QKV projection, rotate-half RoPE (fp32 cos / sin
    tables of every position, rounded to the dtype), grouped-query attention over the KV
    cache (softmax in fp32), O projection + residual, RMSNorm, fused gate / up projection,
    SiLU(gate) * up, down projection + residual; then the final RMSNorm, the LM head and the
    argmax. Eager rounding: every op's result in the dtype, as transformers computes it.

    ``step(token, pos, k_cache, v_cache)`` (host values: it is the reference) appends the
    token's K / V at ``pos`` and returns the logits for position ``pos + 1``; the caches
    (:meth:`new_cache`) are [layers, kv_heads, capacity, head_dim]."""

    def __init__(
        self,
        vocab: int = 8192,
        hidden: int = 1024,
        layers: int = 1,
        heads: int = 16,
        kv_heads: int = 4,
        head_dim: int = 64,
        inter: int = 2048,
        capacity: int = 4096,
        theta: float = 10000.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if heads % kv_heads or head_dim % 2:
            raise ValueError("heads must be a multiple of kv_heads, head_dim even")
        self.vocab, self.hidden, self.heads, self.kv_heads = vocab, hidden, heads, kv_heads
        self.head_dim, self.inter, self.capacity = head_dim, inter, capacity
        self.embed = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList(
            DecoderLayer(hidden, heads, kv_heads, head_dim, inter, eps) for _ in range(layers)
        )
        self.norm = RMSNorm(hidden, eps)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.theta = theta
        self._rope: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    def rope_tables(self) -> tuple[torch.Tensor, torch.Tensor]:
        """fp32 cos and sin [capacity, head_dim / 2] of every position, on the weights'
        device. Not buffers: a module cast to bf16 would round them; the model rounds them
        to its dtype at use, after computing them in fp32 (as transformers does)."""
        device = self.embed.weight.device
        if device not in self._rope:
            d = self.head_dim
            inv_freq = 1.0 / self.theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d)
            pos = torch.arange(self.capacity, dtype=torch.float32)
            freqs = torch.outer(pos, inv_freq).to(device)
            self._rope[device] = (freqs.cos().contiguous(), freqs.sin().contiguous())
        return self._rope[device]

    @torch.no_grad()
    def randomize_(self, seed: int = 0) -> "TinyDecoder":
        """Weights of a trained model's scale: projections ~ N(0, fan_in ** -0.5), norm
        weights ~ N(1, 0.1), embeddings ~ N(0, 1)."""
        gen = torch.Generator(device=self.embed.weight.device).manual_seed(seed)
        for name, weight in self.named_parameters():
            noise = torch.randn(weight.shape, generator=gen, device=weight.device)
            if name.endswith("norm.weight"):
                weight.copy_(1.0 + 0.1 * noise)
            elif name == "embed.weight":
                weight.copy_(noise)
            else:
                weight.copy_(weight.shape[1] ** -0.5 * noise)
        return self

    def new_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero K and V caches [layers, kv_heads, capacity, head_dim] in the model's dtype."""
        w = self.embed.weight
        shape = (len(self.layers), self.kv_heads, self.capacity, self.head_dim)
        return (
            torch.zeros(shape, dtype=w.dtype, device=w.device),
            torch.zeros(shape, dtype=w.dtype, device=w.device),
        )

    @torch.no_grad()
    def step(
        self, token: int, pos: int, k_cache: torch.Tensor, v_cache: torch.Tensor
    ) -> torch.Tensor:
        dtype = self.embed.weight.dtype
        hq, hkv, d = self.heads, self.kv_heads, self.head_dim
        group = hq // hkv
        x = self.embed.weight[token].view(1, -1)
        cos_table, sin_table = self.rope_tables()
        cos, sin = cos_table[pos].to(dtype), sin_table[pos].to(dtype)
        cos, sin = torch.cat((cos, cos)), torch.cat((sin, sin))
        for i, layer in enumerate(self.layers):
            qkv = layer.qkv(layer.input_norm(x)).view(-1)
            q = qkv[: hq * d].view(hq, d)
            k = qkv[hq * d : (hq + hkv) * d].view(hkv, d)
            v = qkv[(hq + hkv) * d :].view(hkv, d)
            q = q * cos + rotate_half(q) * sin
            k_cache[i, :, pos] = k * cos + rotate_half(k) * sin
            v_cache[i, :, pos] = v
            keys = k_cache[i, :, : pos + 1].float()  # [hkv, L, d]
            values = v_cache[i, :, : pos + 1].float()
            scores = q.float().view(hkv, group, d) @ keys.transpose(1, 2) * d**-0.5
            att = (scores.softmax(-1) @ values).reshape(1, hq * d).to(dtype)
            x = x + layer.o(att)
            gate, up = layer.gate_up(layer.post_norm(x)).chunk(2, dim=-1)
            x = x + layer.down(F.silu(gate) * up)
        return self.lm_head(self.norm(x)).view(-1)

    def forward(
        self, token: int, pos: int, k_cache: torch.Tensor, v_cache: torch.Tensor
    ) -> torch.Tensor:
        return self.step(token, pos, k_cache, v_cache)


DECODE_MODES = ("megakernel", "graph")
DECODE_LABELS = {
    "megakernel": "megakernel (one launch per step: counters, split-KV attention, advance)",
    "graph": "graph (the same opcodes, one launch per op in a CUDA graph)",
}


def gemv_rows(m: int, k: int, pool_bytes: int, want: int = ROWS) -> int:
    """The most output rows per GEMV tile, at most ``want``, that divide ``m`` and whose bf16
    weights (rows x k x 2 bytes) fit a page pool of ``pool_bytes`` (0: not one row fits)."""
    return oc.tile_rows(m, k * 2, pool_bytes, want)


def tiles_covering(ranges: list[tuple[int, int]], rows: int) -> list[int]:
    """The GEMV tiles of ``rows`` output rows each that write any row of ``ranges``."""
    return sorted({r // rows for lo, hi in ranges for r in range(lo, hi)})


def default_splits(queues: int, heads: int) -> int:
    """Attention tiles per KV head and q block (``heads`` of them): about one wave of the
    queues for all of them (each split works at every length: the chunks follow it)."""
    return max(1, queues // max(1, heads))


def decode_program(
    dims: dict[str, int],
    T: dict[str, int],
    W: list[dict[str, int]],
    split: mkd.SplitKV,
    *,
    pool_bytes: int,
    eps: float,
    dtype: int = 0,
    hint: int = mks.EVICT_NORMAL,
) -> tuple[list[mks.Op], list[mks.Edge], list[str]]:
    """The decode step's ops, their edges and the ops that read the step state, from the
    model's ``dims`` (vocab, hidden, heads, kv_heads, head_dim, inter, capacity, layers), the
    tensor table's indices (``T``: the shared tensors, ``W``: each layer's) and the split-KV
    layout; no tensor and no GPU needed (the CPU tests build it).

    Buffers: ``x`` [layers + 1, hidden] (row 0 the embedding, row l + 1 layer l's output),
    ``xa`` [layers, hidden] (after the attention), ``qkv``, ``q`` (rotated), ``att``, ``gu``
    (gate | up), ``h`` per layer, ``po`` / ``pml`` (partials: ``split.partial_rows`` per
    layer), ``logits``, ``best``; every layer its own rows, so no buffer is rewritten while
    an instruction of the same step may still read it. The edges carry what each op reads
    from the op before it; a residual read (the O projection's of ``x``, the down
    projection's of ``xa``) rides the chain of edges through the attention or the MLP: the
    acquire / release chain is transitive, so it needs no edge (and no wait) of its own."""
    hq, hkv, d = dims["heads"], dims["kv_heads"], dims["head_dim"]
    hidden, inter, vocab, cap = dims["hidden"], dims["inter"], dims["vocab"], dims["capacity"]
    n_layers, group = dims["layers"], hq // hkv
    qkv_n, att_n = (hq + 2 * hkv) * d, hq * d
    ops: list[mks.Op] = []
    edges: list[mks.Edge] = []

    def gemv(name: str, w: int, m: int, k: int, **kw: int) -> int:
        rows = gemv_rows(m, k, pool_bytes)
        if rows < 1:
            raise mks.ScheduleError(f"{name}: not one row of {k} bf16 weights fits the pool")
        tile = rows * k * 2
        args = dict(kw, eps=eps if kw.get("gamma", -1) >= 0 else 0.0, rows=rows, k=k)
        ops.append(
            mks.Op(
                name,
                GEMV,
                m // rows,
                cost=float(tile),
                args=lambda t: gemv_args(**args, row0=t * rows),
                weights=lambda t: mks.Prefetch(w, t * tile, tile, hint),
            )
        )
        return rows

    def small(name: str, opcode: int, tiles: int, args: mks.TileArgs) -> None:
        ops.append(mks.Op(name, opcode, tiles, cost=mkd.FIXED_COST, args=args))

    length = {"length": T["state"], "length_off": mkd.STEP_POS, "length_add": 1}
    token = embed_args(
        table=T["embed"], vocab=vocab, dim=hidden, esize=2, token=T["state"],
        token_off=mkd.STEP_TOKEN, out=T["x"], out_off=0,
    )  # fmt: skip
    small("embed", EMBED, 1, [token])
    readers = ["embed"]
    glu_tiles = max(1, inter // 256)
    per_glu = inter // glu_tiles
    for i in range(n_layers):
        p, w = f"l{i}.", W[i]
        rows = gemv(
            p + "qkv", w["qkv"], qkv_n, hidden, x=T["x"], x_off=i * hidden, gamma=w["in_norm"],
            out=T["qkv"], out_off=i * qkv_n,
        )  # fmt: skip
        edges.append(mks.Edge("embed" if i == 0 else f"l{i - 1}.down", p + "qkv"))

        def rope(h: int, i: int = i, w: dict[str, int] = w) -> list[int]:
            return rope_kv_args(
                qkv=T["qkv"], qkv_off=i * qkv_n, qheads=hq, kvheads=hkv, kv_head=h, group=group,
                dim=d, qout=T["q"], qout_off=i * att_n, k=w["k"], v=w["v"], cap=cap,
                pos=T["state"], pos_off=mkd.STEP_POS, cos=T["cos"], sin=T["sin"], dtype=dtype,
            )  # fmt: skip

        def head_rows(h: int, rows: int = rows) -> list[int]:  # q heads, k and v of KV head h
            spans = [(h * group * d, (h + 1) * group * d), ((hq + h) * d, (hq + h + 1) * d)]
            return tiles_covering([*spans, ((hq + hkv + h) * d, (hq + hkv + h + 1) * d)], rows)

        small(p + "rope", ROPE_KV, hkv, rope)
        edges.append(mks.Edge(p + "qkv", p + "rope", head_rows))

        def attention(a: mkd.AttentionTile, i: int = i, w: dict[str, int] = w) -> list[int]:
            return attn_args(
                q=T["q"], q_off=i * att_n, k=w["k"], v=w["v"], cap=cap, kv_head=a.kv_head,
                q0=a.q0, qn=a.qn, split=a.split, splits=split.splits, chunk=split.chunk,
                **length, scale=d**-0.5, po=T["po"], pml=T["pml"], prow0=i * split.partial_rows,
                dim=d, dtype=dtype,
            )  # fmt: skip

        def combine(h: int, q0: int, qn: int, i: int = i) -> list[int]:
            return combine_args(
                po=T["po"], pml=T["pml"], prow0=i * split.partial_rows, splits=split.splits,
                chunk=split.chunk, **length, cap=cap, q0=q0, qn=qn, out=T["att"],
                out_off=i * att_n, dim=d, dtype=dtype,
            )  # fmt: skip

        attn, comb, to_combine = split.ops(
            p + "attn", p + "combine", (ATTN_DECODE, ATTN_COMBINE), attention, combine
        )
        ops += [attn, comb]
        edges += [mks.Edge(p + "rope", p + "attn", split.kv_head_of), to_combine]
        gemv(
            p + "o", w["o"], hidden, att_n, x=T["att"], x_off=i * att_n, out=T["xa"],
            out_off=i * hidden, res=T["x"], res_off=i * hidden,
        )  # fmt: skip
        edges.append(mks.Edge(p + "combine", p + "o"))
        rows = gemv(
            p + "gate_up", w["gate_up"], 2 * inter, hidden, x=T["xa"], x_off=i * hidden,
            gamma=w["post_norm"], out=T["gu"], out_off=i * 2 * inter,
        )  # fmt: skip
        edges.append(mks.Edge(p + "o", p + "gate_up"))

        def glu(t: int, i: int = i) -> list[int]:  # silu(gate) * up
            return glu_args(
                a=T["gu"], a_off=i * 2 * inter, b=T["gu"], b_off=i * 2 * inter + inter,
                out=T["h"], out_off=i * inter, i0=t * per_glu, n=per_glu, act=GLU_SILU,
            )  # fmt: skip

        def gate_and_up(t: int, rows: int = rows) -> list[int]:
            lo, hi = t * per_glu, (t + 1) * per_glu
            return tiles_covering([(lo, hi), (inter + lo, inter + hi)], rows)

        small(p + "glu", GLU, glu_tiles, glu)
        edges.append(mks.Edge(p + "gate_up", p + "glu", gate_and_up))
        gemv(
            p + "down", w["down"], hidden, inter, x=T["h"], x_off=i * inter, out=T["x"],
            out_off=(i + 1) * hidden, res=T["xa"], res_off=i * hidden,
        )  # fmt: skip
        edges.append(mks.Edge(p + "glu", p + "down"))
        readers += [p + "rope", p + "attn", p + "combine"]
    gemv(
        "lm", T["lm"], vocab, hidden, x=T["x"], x_off=n_layers * hidden, gamma=T["norm"],
        out=T["logits"], out_off=0,
    )  # fmt: skip
    edges.append(mks.Edge(f"l{n_layers - 1}.down", "lm"))
    advance = argmax_args(
        x=T["logits"], x_off=0, n=vocab, out=T["best"], out_off=0, advance=1, step=T["state"],
        step_off=0, hist=T["history"], hist_len=cap,
    )  # fmt: skip
    small("argmax", ARGMAX, 1, [advance])
    edges.append(mks.Edge("lm", "argmax"))
    return ops, edges, readers


#: The shared tensors of the decode step's table, in order (then each layer's: LAYER_TENSORS).
DECODE_TENSORS = (
    "embed", "state", "history", "best", "x", "xa", "qkv", "q", "att", "gu", "h", "po", "pml",
    "logits", "cos", "sin", "norm", "lm",
)  # fmt: skip
LAYER_TENSORS = ("qkv", "o", "gate_up", "down", "in_norm", "post_norm", "k", "v")


def decode_table(layers: int) -> tuple[dict[str, int], list[dict[str, int]]]:
    """The tensor-table indices of :data:`DECODE_TENSORS` and of each layer's tensors."""
    shared = {name: k for k, name in enumerate(DECODE_TENSORS)}
    base = len(DECODE_TENSORS)
    per = [
        {name: base + i * len(LAYER_TENSORS) + k for k, name in enumerate(LAYER_TENSORS)}
        for i in range(layers)
    ]
    return shared, per


def decoder_dims(reference: "TinyDecoder") -> dict[str, int]:
    ref = reference
    return {
        "vocab": ref.vocab, "hidden": ref.hidden, "heads": ref.heads, "kv_heads": ref.kv_heads,
        "head_dim": ref.head_dim, "inter": ref.inter, "capacity": ref.capacity,
        "layers": len(ref.layers),
    }  # fmt: skip


class DecodeStep(nn.Module):
    """``reference``'s decode step on the device. State (on the device, reused by every
    step): ``state`` (int32 [token, position]), the K / V caches ([layers, kv_heads,
    capacity, head_dim]), ``history`` (int32 [capacity]: the token at each position),
    ``logits`` and ``best`` (the last step's). ``reset(token, pos, cache)`` sets them from the
    host once; ``step()`` replays the step's CUDA graph: the embedding reads the device's
    token, the attention its position, the argmax writes the next ones, no host value.

    ``mode``: ``megakernel`` (one launch of ``kernel<DecodeOps>`` over one schedule) or
    ``graph`` (one launch per op, the same opcodes and tiles). ``splits``: attention tiles
    per KV head and q block (0: :func:`default_splits`), ``chunk``: keys per split (0: the
    length spread over the splits), ``pages`` / ``inflight`` / ``trace`` as for the chain."""

    def __init__(
        self,
        reference: TinyDecoder,
        mode: str = "megakernel",
        splits: int = 0,
        chunk: int = 0,
        trace: bool = False,
        pages: int = 0,
        inflight: int = 1,
    ) -> None:
        super().__init__()
        if mode not in DECODE_MODES:
            raise ValueError(f"mode {mode!r}: one of {', '.join(DECODE_MODES)}")
        ref = reference
        self.reference, self.mode, self.label = ref, mode, DECODE_LABELS[mode]
        dims = decoder_dims(ref)
        hq, hkv, d, hidden = ref.heads, ref.kv_heads, ref.head_dim, ref.hidden
        n_layers, group = len(ref.layers), hq // hkv
        weight = ref.embed.weight
        dev, dt = weight.device, weight.dtype
        self.ext = extension()
        fit, queues, page_bytes, _ = self.ext.mk_info(True)
        self.pages = min(pages, fit) if pages > 0 else fit
        self.queues, self.inflight = queues, inflight
        pool = self.pages * page_bytes
        heads = hkv * len(mkd.q_blocks(group))
        self.split = split = mkd.SplitKV(
            hkv, group, d, ref.capacity, splits or default_splits(queues, heads), chunk
        )
        with torch.inference_mode(False):  # updated in place, in and out of inference

            def zeros(*shape: int, dtype: torch.dtype = dt) -> torch.Tensor:
                return torch.zeros(shape, dtype=dtype, device=dev)

            self.state = zeros(mkd.STEP_WORDS, dtype=torch.int32)
            self.history = zeros(ref.capacity, dtype=torch.int32)
            self.best = zeros(1, dtype=torch.int64)
            self.k_cache, self.v_cache = ref.new_cache()
            self.x, self.xa = zeros(n_layers + 1, hidden), zeros(n_layers, hidden)
            self.qkv = zeros(n_layers, (hq + 2 * hkv) * d)
            self.q, self.att = zeros(n_layers, hq * d), zeros(n_layers, hq * d)
            self.gu, self.h = zeros(n_layers, 2 * ref.inter), zeros(n_layers, ref.inter)
            self.po = zeros(n_layers * split.partial_rows, d, dtype=torch.float32)
            self.pml = zeros(n_layers * split.partial_rows, 2, dtype=torch.float32)
            self.logits = zeros(ref.vocab)
        self.position = 0  # the host's count of where the device is (no device read)
        shared = {
            "embed": weight.detach(), "state": self.state, "history": self.history,
            "best": self.best, "x": self.x, "xa": self.xa, "qkv": self.qkv, "q": self.q,
            "att": self.att, "gu": self.gu, "h": self.h, "po": self.po, "pml": self.pml,
            "logits": self.logits, "cos": ref.rope_tables()[0], "sin": ref.rope_tables()[1],
            "norm": ref.norm.weight.detach(),
            "lm": ref.lm_head.weight.detach(),
        }  # fmt: skip
        tensors = [shared[name] for name in DECODE_TENSORS]
        for i, layer in enumerate(ref.layers):
            own = {
                "qkv": layer.qkv.weight, "o": layer.o.weight, "gate_up": layer.gate_up.weight,
                "down": layer.down.weight, "in_norm": layer.input_norm.weight,
                "post_norm": layer.post_norm.weight,
            }  # fmt: skip
            caches = {"k": self.k_cache[i], "v": self.v_cache[i]}
            tensors += [caches[n] if n in caches else own[n].detach() for n in LAYER_TENSORS]
        self.tensors = tensors  # the table holds their addresses
        T, W = decode_table(n_layers)
        # weights the L2 cannot keep stream once per step: evict them first
        l2 = torch.cuda.get_device_properties(dev).L2_cache_size
        total = sum(p.numel() * p.element_size() for p in ref.parameters())
        ops, edges, readers = decode_program(
            dims, T, W, split, pool_bytes=pool, eps=float(ref.norm.variance_epsilon),
            dtype=DTYPES[dt], hint=oc.l2_hint(total, l2),
        )  # fmt: skip
        self.ops, self.edges, self.readers = ops, edges, readers
        self.schedule = mks.build(ops, edges, queues, meta={"pages": self.pages, "mode": mode})
        if problems := mkd.check_advance(self.schedule, readers, "argmax"):
            raise mks.ScheduleError(problems[0])
        self.rt: runtime.Runtime | None = None
        self.runtimes: list[runtime.Runtime] = []
        launch = getattr(self, f"_setup_{mode}")(tensors, pool, trace)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.device(dev):
            launch()  # warm-up: loads the kernel (one step from the zero state, undone below)
            torch.cuda.synchronize()
            self.check()
            with torch.cuda.graph(self._graph):
                launch()
        self.reset(0, 0, ref.new_cache())

    def _setup_megakernel(self, tensors: list[torch.Tensor], pool: int, trace: bool):
        self.rt = rt = runtime.Runtime(self.schedule, tensors, trace=trace, pool_bytes=pool)
        self.runtimes = [rt]
        pages, queues, inflight = self.pages, self.queues, self.inflight
        return lambda: self.ext.mk_run(*rt.args(), pages, queues, inflight, True)

    def _setup_graph(self, tensors: list[torch.Tensor], pool: int, trace: bool):
        """One schedule (and launch) per op, in the op order: each op's tiles on as many
        queues as it has tiles, up to the resident blocks; no counters between ops."""
        launches = []
        for op in self.ops:
            n = min(op.tiles, self.queues)
            rt = runtime.Runtime(mks.build([op], [], n), tensors, pool_bytes=pool)
            self.runtimes.append(rt)
            launches.append((rt, n))
        pages, inflight = self.pages, self.inflight

        def launch() -> None:
            for rt, n in launches:
                self.ext.mk_run(*rt.args(), pages, n, inflight, True)

        return launch

    def check(self) -> None:
        """Raise :class:`runtime.MegakernelHang` when a launch so far stopped."""
        for rt in self.runtimes:
            rt.check()

    def reset(
        self,
        token: int,
        pos: int = 0,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """The step state from the host, once before a run of steps: ``token`` at position
        ``pos``; ``cache``: K and V caches to copy in ([layers, kv_heads, capacity,
        head_dim], e.g. a prefill's), else the caches keep what they hold."""
        cap = self.reference.capacity
        if not 0 <= pos < cap:
            raise ValueError(f"position {pos} outside the cache's {cap}")
        with torch.inference_mode(False):
            self.state.copy_(torch.tensor([token, pos], dtype=torch.int32))
            self.history.zero_()
            self.history[pos] = token
            if cache is not None:
                self.k_cache.copy_(cache[0])
                self.v_cache.copy_(cache[1])
        self.position = pos

    def step(self) -> None:
        """One decode step (a graph replay): no host value in, none out."""
        if self.position >= self.reference.capacity:
            raise ValueError(f"the cache is full ({self.reference.capacity} positions)")
        self.check()  # a hung schedule is not launched again
        self._graph.replay()
        self.position += 1

    def generate(self, steps: int) -> torch.Tensor:
        """``steps`` steps back to back; the history (int32 [capacity]) on the device."""
        for _ in range(steps):
            self.step()
        return self.history

    def forward(self) -> torch.Tensor:
        """One step; the next token (int64 [1], a copy)."""
        self.step()
        return self.best.clone()

    def trace_summary(self) -> dict:
        """Per op: execution and wait times of the last step, the prefetch overlap."""
        if self.rt is None or self.rt.trace is None:
            return {}
        return simulate.trace_summary(self.schedule, self.rt.trace_rows())


def decode_supported(reference: nn.Module) -> str | None:
    """Why the decode step cannot build for ``reference`` (None: it can): its GEMV opcodes take
    bf16 weights with K a multiple of 256 up to 4096, the attention head dims 64, 128, 256."""
    if not isinstance(reference, TinyDecoder):
        return "not a TinyDecoder"
    if not torch.cuda.is_available() or not reference.embed.weight.is_cuda:
        return "no CUDA weights"
    if reference.embed.weight.dtype != torch.bfloat16:
        return "the GEMV opcodes take bf16 weights"
    ks = (reference.hidden, reference.heads * reference.head_dim, reference.inter)
    if any(k % 256 or not 256 <= k <= 4096 for k in ks):
        return f"K of every projection must be a multiple of 256 up to 4096: {ks}"
    if reference.head_dim not in mkd.HEAD_DIMS:
        return f"head dim {reference.head_dim}: one of {mkd.HEAD_DIMS}"
    if any(not p.is_contiguous() or p.data_ptr() % 16 for p in reference.parameters()):
        return "parameters must be contiguous and 16-byte aligned"
    return None


def build_decode(
    reference: TinyDecoder,
    mode: str = "megakernel",
    splits: int = 0,
    chunk: int = 0,
    trace: bool = False,
    pages: int = 0,
    inflight: int = 1,
) -> nn.Module:
    """The decode step of ``reference`` (:class:`DecodeStep`), or ``reference`` itself where
    :func:`decode_supported` says no."""
    if decode_supported(reference) is not None:
        return reference
    return DecodeStep(reference, mode, splits, chunk, trace, pages, inflight)
