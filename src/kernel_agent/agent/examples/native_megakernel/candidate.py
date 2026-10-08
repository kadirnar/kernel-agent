"""Example native project: a megakernel built with kernel-agent's kit
(``kernel_agent.native.megakernel``, issue #225), next to two baselines from the same math.

Reference: ``reference.norms`` and ``reference.layers`` (``kernel_agent.selftest.
NormGemvChain``): ``x = x + W_l rmsnorm_l(x)`` for every layer l, bf16, ``W_l`` square and
bias-free with n a multiple of 256 up to 4096, one row (decode) per call: a stand-in for any
stack of decode layers (norm, projection, residual). Other row counts take the reference.

``mode`` (a ``build()`` keyword: ``sweep_candidate`` compares them):

* ``megakernel``: one launch of the interpreter (``ka_mk.cuh``) over a schedule built once
  here (:mod:`kernel_agent.native.megakernel.schedule`): one GEMV opcode per tile of ``rows``
  output rows with the layer's RMSNorm fused into its prologue and the residual into its
  epilogue; a layer's tiles wait on one counter, the previous layer's (target: its tile
  count), not on a grid barrier, and every block's producer warp streams its next tiles'
  weights into the shared-memory page pool while its consumers wait and compute
  (``inflight``: pages in flight per block; 1 measured best on an RTX 5070 Ti, more pages
  in flight queue the activations' critical loads behind weight copies);
* ``graph_pdl``: one kernel per layer in a CUDA graph with programmatic dependent launch
  (weights into registers before ``ka_pdl_wait()``): the best launch-per-layer design of
  docs/PARALLEL.md §4.6;
* ``coop_barrier``: one cooperative persistent kernel with a hand-rolled grid barrier after
  every layer (§4.6's slow persistent variant).

Every mode is captured once into a CUDA graph here; a call copies x into the activation
buffer (one row per layer, ``[layers + 1, n]``), replays the graph and returns a copy of the
last row. ``trace=True`` records per-instruction time stamps of the megakernel
(``KA_MK_TRACE``, set in ``kernel_project.toml``): ``module.trace_summary()``.

Opcodes (``include/mk_ops.cuh``): RMSNorm, GEMV tiles with bf16 or e4m3 weights (fused norm,
residual, split-K partials), residual add, split-K reduce, argmax; the argument builders
below mirror their slots.
"""

import struct

import torch
from torch import nn

from kernel_agent.native import project
from kernel_agent.native.megakernel import runtime, simulate
from kernel_agent.native.megakernel import schedule as mks

#: GPUs this example runs on (``kernel_agent.gpu_arch.supports``).
ARCHS = "sm_80+"
ARCHS_WHY = "bf16 opcodes; page loads with cp.async (sm_80-sm_89) or bulk copies (sm_90+)"

MODES = ("megakernel", "graph_pdl", "coop_barrier")

# opcodes of include/mk_ops.cuh
NOP, RMSNORM, GEMV, GEMV_FP8, RESIDUAL, SPLITK_REDUCE, ARGMAX = range(7)


def f32_bits(value: float) -> int:
    """A float argument as the int32 word the device reads with ``ctx.arg_f``."""
    return struct.unpack("<i", struct.pack("<f", value))[0]


def gemv_args(
    *,
    x: int,
    x_off: int,
    out: int,
    out_off: int,
    row0: int,
    rows: int,
    k: int,
    gamma: int = -1,
    gamma_off: int = 0,
    eps: float = 0.0,
    res: int = -1,
    res_off: int = 0,
    k0: int = 0,
    klen: int | None = None,
    part: int = -1,
    part_off: int = 0,
    scale: int = -1,
    scale_off: int = 0,
) -> list[int]:
    """Arguments of a GEMV / GEMV_FP8 tile (tensors by table index, offsets in elements):
    rows [row0, row0 + rows) of ``W h`` over columns [k0, k0 + klen); ``gamma``: fuse the
    RMSNorm of x (over all k columns); ``res``: add a residual; ``part``: write fp32
    partials there instead (split-K); ``scale``: the e4m3 weights' per-row scales."""
    return [
        x, x_off, gamma, gamma_off, f32_bits(eps), out, out_off, res, res_off, row0, rows, k,
        k0, k if klen is None else klen, part, part_off, scale, scale_off,
    ]  # fmt: skip


def rmsnorm_args(
    *, x: int, x_off: int, gamma: int, gamma_off: int, eps: float, out: int, out_off: int, n: int
) -> list[int]:
    return [x, x_off, gamma, gamma_off, f32_bits(eps), out, out_off, n]


def residual_args(
    *, a: int, a_off: int, b: int, b_off: int, out: int, out_off: int, i0: int, n: int
) -> list[int]:
    return [a, a_off, b, b_off, out, out_off, i0, n]


def reduce_args(
    *,
    part: int,
    part_off: int,
    splits: int,
    stride: int,
    row0: int,
    rows: int,
    out: int,
    out_off: int,
    res: int = -1,
    res_off: int = 0,
) -> list[int]:
    return [part, part_off, splits, stride, row0, rows, out, out_off, res, res_off]


def argmax_args(*, x: int, x_off: int, n: int, out: int, out_off: int) -> list[int]:
    return [x, x_off, n, out, out_off]


def extension():
    """The compiled project (``project.load``: once per content digest and toolchain)."""
    return project.load(__file__)


class Chain(nn.Module):
    """The layers of ``reference`` as one CUDA graph of ``mode``'s launches (one-row calls)."""

    def __init__(
        self,
        reference: nn.Module,
        mode: str = "megakernel",
        rows: int = 16,
        trace: bool = False,
        pages: int = 0,
        inflight: int = 1,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode {mode!r}: one of {', '.join(MODES)}")
        self.reference = reference  # other row counts (a fallback)
        self.mode = mode
        weights = [layer.weight.detach() for layer in reference.layers]
        gammas = [norm.weight.detach() for norm in reference.norms]
        self.eps = float(getattr(reference.norms[0], "variance_epsilon", 1e-6))
        self.n, self.layers = weights[0].shape[0], len(weights)
        device = weights[0].device
        with torch.inference_mode(False):  # buffers updated in place, in and out of inference
            self._weights, self._gammas = weights, gammas  # the graph holds their addresses
            self._act = torch.zeros(self.layers + 1, self.n, device=device, dtype=torch.bfloat16)
        self.ext = extension()
        self.rt: runtime.Runtime | None = None
        self.inflight = inflight
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
        if n % rows or n > max_slice:
            raise ValueError(f"rows {rows} must divide n {n} (at most {max_slice} columns)")
        act = 2 * layers  # tensor table: weights, gammas, the activation buffer
        tile_bytes = rows * n * 2
        # weights the L2 cannot keep stream once per call: evict them first, so the program,
        # the activations and the norms' weights stay in L2 (the GPU's own L2 size decides)
        l2 = torch.cuda.get_device_properties(self._act.device).L2_cache_size
        total = sum(w.numel() * w.element_size() for w in self._weights)
        self.l2_hint = mks.EVICT_FIRST if total > l2 // 2 else mks.EVICT_NORMAL

        def tile(layer: int, t: int) -> list[int]:  # layer's RMSNorm fused, its residual added
            return gemv_args(
                x=act, x_off=layer * n, gamma=layers + layer, eps=self.eps, out=act,
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
                weights=lambda t, layer=layer: mks.Prefetch(
                    layer, t * tile_bytes, tile_bytes, self.l2_hint
                ),
            )
            for layer in range(layers)
        ]
        edges = [mks.Edge(f"layer{k - 1}", f"layer{k}", mks.ALL) for k in range(1, layers)]
        self.schedule = mks.build(ops, edges, queues, meta={"rows": rows, "pages": pages})
        self.rt = runtime.Runtime(
            self.schedule,
            [*self._weights, *self._gammas, self._act],
            trace=trace,
            pool_bytes=pages * page_bytes,
        )
        rt = self.rt
        return lambda: self.ext.mk_run(*rt.args(), pages, queues, self.inflight)

    def _setup_graph_pdl(self, rows: int, trace: bool, pages: int):
        return lambda: self.ext.chain_pdl(self._weights, self._gammas, self._act, self.eps, True)

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
        self._act[0].copy_(x.reshape(-1))
        self._graph.replay()
        return self._act[self.layers].clone().view(x.shape)  # the buffer is reused

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
        and rows >= 1
        and n % rows == 0
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
    rows: int = 16,
    trace: bool = False,
    pages: int = 0,
    inflight: int = 1,
) -> nn.Module:
    """``rows``: output rows per GEMV tile; ``pages``: the page pool's size (0: what fits this
    GPU's shared memory); ``inflight``: pages each producer warp keeps in flight (0: all);
    ``trace``: per-instruction time stamps (``trace_summary()``)."""
    if not _supported(reference, rows):
        return reference
    return Chain(reference, mode, rows, trace, pages, inflight)
