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
residual, split-K partials), residual add, split-K reduce, argmax, gated activation (GLU);
their numbers and argument slots are the kit's (``kernel_agent.native.megakernel.opcodes``).
"""

import torch
from torch import nn

from kernel_agent.native import project
from kernel_agent.native.megakernel import opcodes as oc
from kernel_agent.native.megakernel import runtime, simulate
from kernel_agent.native.megakernel import schedule as mks

# The opcodes of include/mk_ops.cuh: numbers and argument slots (the kit's generic ABI)
from kernel_agent.native.megakernel.opcodes import (  # noqa: F401
    ARGMAX,
    GEMV,
    GEMV_FP8,
    GLU,
    NOP,
    RESIDUAL,
    RMSNORM,
    SPLITK_REDUCE,
    argmax_args,
    f32_bits,
    gemv_args,
    glu_args,
    reduce_args,
    residual_args,
    rmsnorm_args,
)

#: GPUs this example runs on (``kernel_agent.gpu_arch.supports``).
ARCHS = "sm_80+"
ARCHS_WHY = "bf16 opcodes; page loads with cp.async (sm_80-sm_89) or bulk copies (sm_90+)"
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
    (``captured``)."""
    if not _supported(reference, rows):
        return reference
    return Chain(reference, mode, rows, trace, pages, inflight, schedule)
