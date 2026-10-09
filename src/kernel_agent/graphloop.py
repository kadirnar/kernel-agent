"""Host-free generation loops: a CUDA-graph WHILE node with an on-device stop check (#232).

A generation loop (LLM decode, a TTS patch loop, an AR diffusion sampler) goes back to
Python every step: to launch the step and to read its stop flag. Both cost host time the GPU
waits for when steps are short: the batch-16 VoxCPM2 loop's per-patch stop-flag ``.cpu()``
left its GPU idle (busy 93.1 % → 96.0 % with an async read, docs/PARALLEL.md §0), and a
graph-launched kernel boundary costs ~0.9 µs on an RTX 5070 Ti (~1.3 µs on an H100, Hazy
Research 2025). :func:`device_loop` runs the whole loop on the device, model agnostic:

* ``while``: one CUDA graph whose WHILE node (a conditional node, CUDA 12.4+, built through
  ``cuda.core``) runs the body until the stop: the step, the loop's own index / stop update
  and a one-thread condition kernel that calls ``cudaGraphSetConditional``. One launch per
  run, no host decision per step. The step's torch ops (or a native engine's launches on
  the current stream) are captured on a torch ``ExternalStream`` of the body's capture
  stream, their memory from a private ``torch.cuda.MemPool`` kept alive with the graph.
* ``unrolled`` (the fallback where conditional nodes are unavailable or the capture
  fails): a ``torch.cuda.CUDAGraph`` of K steps replayed until the stop, the loop's status
  read on the host once per block, one block ahead (the GPU never waits for the read).
  Steps after the stop run *masked*: the step gates its writes with ``active``
  (:func:`masked_copy_`, :func:`masked_index_copy_`), so they write nothing. K comes from
  the measured step time and launch cost (:func:`measure_unroll`, :func:`choose_unroll`).
* ``host``: the plain loop, the step called from Python and the stop read every step (the
  warm-up runs, a replaced watched callable, the last resort).

Usage::

    from kernel_agent import graphloop

    def step(index, active):  # device int64 scalar (steps done) and bool scalar
        logits = decoder(last, pos=start + index)  # writes the KV cache at start + index
        token = logits.argmax(-1)
        graphloop.masked_index_copy_(out, 0, index.view(1), token, active)
        graphloop.masked_copy_(last, token, active)

    loop = graphloop.device_loop(step, lambda: last == eos, max_steps=256, masked=True)
    steps = loop.run()  # a device tensor: read it when the host needs it (once a request)

The first ``warmup_runs`` runs (default 1) are host runs that also warm the step up (lazy
initialisation, Triton / torch.compile compilations, cuBLAS handles); the graph is built
at the start of the next run, or by :meth:`DeviceLoop.build`. ``loop.mode`` and
``loop.reason`` say what was built and why (every fallback says why); ``loop.stats`` counts
launches and host checks.

The step's contract (as for any CUDA graph): it is a function of device state only, its
buffers static and updated in place, its shapes fixed; no host syncs (``.item()``,
``.cpu()``), no Python state that changes from step to step. Torch's RNG cannot be
captured into the WHILE body (torch refuses an RNG op under a capture it did not start):
such a step falls back to the unrolled graph, whose masked steps after the stop still draw
numbers, so the generator's state after the loop differs from the host loop's; draw the
noise from the loop's own generator or precompute it when later draws must match.

Streaming (``metric=ttfa``): with ``chunk_every=n`` the WHILE body ends with an IF node that
runs at every n-th step and at the stop: ``on_chunk()`` (captured, optional) and a kernel
that writes the steps done to a host-mapped counter. :meth:`DeviceLoop.chunks` launches the
loop and yields those counts as the host sees them, no device-wide synchronize; leaving
the iteration early (the metric window) cancels the rest of the loop through a host-mapped
flag the condition kernel reads and joins what was launched.

Integrity: the graph runs on the caller's current stream, so it is joined with that stream
and the evaluator's timing sees all of it; :meth:`DeviceLoop.chunks` waits for it before
it returns. It is launched with ``cudaGraphLaunch`` of the CUDA runtime torch loaded (the
profiler does not record cuda.core's driver-API launch), and it begins and ends with a
kernel outside the WHILE node, because the profiler lists only the last iteration's body
kernels (measured with torch 2.14, CUDA 13.0). So the end-to-end hidden-work check
(:mod:`kernel_agent.kernels.e2e_activity`) sees the launch and the loop's whole span: a
loop launched on a stream the caller never joins fails it as unjoined work.

Teacher forcing: a chaotic workload's teacher-forced replay wraps a Python callable that
the candidate must call once per step (VoxCPM2: ``model.feat_decoder.forward``). A graph
calls nothing from Python, so name such callables in ``watch=[(module, "forward")]``: a run
in which one of them is not the object it was when the loop was made (make it when the
transform is applied, before any run) runs host steps (the step from Python, the wrapper
sees every call), and the graph is never built while one is replaced. The free-running
runs then use the graph, judged by the free-running checks only, so for a chaotic workload
keep the teacher-forced call out of the device loop (graph the loop around it: the LM
sub-step, an inner sampler); ``watch`` is the safety net, not a way to validate a
whole-step graph.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import math
import time
from collections.abc import Callable, Generator, Iterator, Sequence
from typing import Any

import torch

#: The loop's modes, in fallback order.
MODES = ("while", "unrolled", "host")
#: Conditional graph nodes (IF / WHILE) need a driver of at least this CUDA version.
MIN_DRIVER = (12, 4)
#: Largest unrolled block (the graph's size and instantiation time grow with K).
MAX_UNROLL = 64
#: Host poll interval of :meth:`DeviceLoop.chunks` while a WHILE graph runs (s).
POLL_S = 50e-6
#: Replays per measurement of :func:`measure_unroll`.
MEASURE_REPLAYS = 8

#: The device side of the WHILE graph (NVRTC through cuda.core). ``index``: steps done
#: (int64), ``active``: the loop goes on (bool), ``limits``: (min_steps, max_steps),
#: ``host``: a host-mapped int64 pair (progress, cancel), null when not streaming.
KERNELS_SRC = r"""
extern "C" __device__ __cudart_builtin__ void CUDARTAPI cudaGraphSetConditional(
    cudaGraphConditionalHandle handle, unsigned int value);

// Before the WHILE node: step 0, active; the body runs at least once.
extern "C" __global__ void ka_loop_begin(cudaGraphConditionalHandle loop, long long* index,
                                         bool* active) {
  *index = 0;
  *active = true;
  cudaGraphSetConditional(loop, 1u);
}

// After the step and its stop flags (the same rule as DeviceLoop._advance): count the step;
// stop at max_steps, when every flag is set (from min_steps on) or when the host cancelled;
// set the WHILE condition and, with every > 0, the chunk IF condition (every ``every``
// steps and at the stop). One thread: ``n_flags`` is a batch's flags at most.
extern "C" __global__ void ka_loop_step(cudaGraphConditionalHandle loop,
                                        cudaGraphConditionalHandle chunk, const bool* flags,
                                        long long n_flags, long long* index, bool* active,
                                        const long long* limits, long long every,
                                        const volatile long long* host) {
  const long long i = *index + 1;
  bool done = n_flags > 0 && i >= limits[0];
  for (long long f = 0; done && f < n_flags; ++f) done = flags[f];
  bool stop = done || i >= limits[1];
  if (host != nullptr && host[1] != 0) stop = true;
  *index = i;
  *active = !stop;
  cudaGraphSetConditional(loop, stop ? 0u : 1u);
  if (every > 0) cudaGraphSetConditional(chunk, (stop || i % every == 0) ? 1u : 0u);
}

// In the chunk's IF body, after on_chunk: publish the steps done to the host.
extern "C" __global__ void ka_chunk_signal(const long long* index, volatile long long* host) {
  __threadfence_system();  // the chunk's writes before the count the host acts on
  host[0] = *index;
  __threadfence_system();
}

// After the WHILE node, outside it: the profiler does not list the kernels of a conditional
// body one by one, this one ends when the loop has ended (the hidden-work check sees the
// loop's span); it leaves the (index, active) status the unrolled blocks leave.
extern "C" __global__ void ka_loop_end(const long long* index, const bool* active,
                                       long long* status) {
  status[0] = *index;
  status[1] = *active ? 1 : 0;
}
"""
#: The same rule without a graph condition, for the unrolled and host modes (no conditional
#: node API needed): a masked step (``active`` false) changes nothing; ``status`` is the
#: (index, active) pair the host reads once per unrolled block.
COUNT_SRC = r"""
extern "C" __global__ void ka_loop_count(const bool* flags, long long n_flags,
                                         long long* index, bool* active,
                                         const long long* limits, long long* status) {
  if (*active) {
    const long long i = *index + 1;
    bool done = n_flags > 0 && i >= limits[0];
    for (long long f = 0; done && f < n_flags; ++f) done = flags[f];
    *index = i;
    *active = !(done || i >= limits[1]);
  }
  status[0] = *index;
  status[1] = *active ? 1 : 0;
}
"""
#: The kernels of each source.
NAMES = {
    "while": ("ka_loop_begin", "ka_loop_step", "ka_chunk_signal", "ka_loop_end"),
    "count": ("ka_loop_count",),
}


class Unsupported(RuntimeError):
    """A mode cannot run this loop here (the message says why; the loop falls back)."""


# ------------------------------------------------------------------ CUDA calls


class _Cuda:
    """The CUDA calls of this module (the CPU tests substitute a fake)."""

    def core(self) -> Any:
        import cuda.core as core

        return core

    def driver_version(self) -> tuple[int, int] | None:
        from kernel_agent import toolchain

        return toolchain.driver_cuda_version()

    def capability(self, device: int) -> tuple[int, int]:
        major, minor = torch.cuda.get_device_capability(device)
        return int(major), int(minor)

    def available(self) -> bool:
        return bool(torch.cuda.is_available())

    def graphs(self, device: torch.device) -> bool:
        """Whether CUDA graphs can capture work on ``device``."""
        return device.type == "cuda"

    def current_stream(self, device: int) -> Any:
        return torch.cuda.current_stream(device)

    def external_stream(self, handle: int, device: int) -> Any:
        return torch.cuda.ExternalStream(handle, device=device)

    def use_stream(self, stream: Any) -> contextlib.AbstractContextManager[Any]:
        return torch.cuda.stream(stream)

    def mem_pool(self) -> Any:
        return torch.cuda.MemPool()

    def use_pool(self, pool: Any, device: int) -> contextlib.AbstractContextManager[Any]:
        return torch.cuda.use_mem_pool(pool, device)

    def pool_handle(self) -> Any:
        return torch.cuda.graph_pool_handle()

    def capture_graph(self, fn: Callable[[], None], pool: Any) -> Any:
        """``fn`` captured into a ``torch.cuda.CUDAGraph`` (its ``replay()`` runs it)."""
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            fn()
        return graph

    def event(self) -> Any:
        return torch.cuda.Event()

    def synchronize(self, device: int | None) -> None:
        torch.cuda.synchronize(device)

    def pinned(self, n: int) -> torch.Tensor:
        """A host-mapped int64 buffer (pinned memory is mapped under unified addressing:
        the same pointer on the device)."""
        return torch.zeros(n, dtype=torch.int64, pin_memory=True)

    def end_capture(self, handle: int) -> None:
        """End a capture still open on stream ``handle`` (one a failed operation
        invalidated: until it ends, a device-wide synchronize fails) and drop its graph."""
        from cuda.bindings import driver

        stream = driver.CUstream(handle)
        _, status = driver.cuStreamIsCapturing(stream)
        if status != driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE:
            _, graph = driver.cuStreamEndCapture(stream)
            if graph is not None and int(graph):
                driver.cuGraphDestroy(graph)

    def runtime_launch(self, graph: Any, stream: Any) -> bool:
        """Launch a cuda.core graph on a torch stream with ``cudaGraphLaunch`` of the CUDA
        runtime torch loaded: the profiler records it like torch's own graph replays (it
        does not record cuda.core's driver-API ``cuGraphLaunch``), so the hidden-work check
        sees the launch and the work it made. False: that runtime was not found."""
        cudart = _cudart()
        if cudart is None:
            return False
        err = cudart.cudaGraphLaunch(
            ctypes.c_void_p(int(graph.handle)), ctypes.c_void_p(int(stream.cuda_stream))
        )
        if err != 0:
            raise RuntimeError(f"cudaGraphLaunch failed with CUDA error {err}")
        return True


_cuda: Any = _Cuda()


#: Builders of failed captures, kept alive: destroying a conditional body's builder after a
#: capture that failed inside it crashed the process (cuda.core 1.2.1).
_ABANDONED: list[Any] = []


def _abandon(builders: list[Any]) -> None:
    """End the captures of a failed build, innermost first (through the driver where
    cuda.core no longer counts an invalidated capture as building), and keep the builders
    for good: each gets one reference nothing releases, because a list alone is cleared at
    interpreter shutdown and destroying them then crashed the process on exit (#246)."""
    for builder in reversed(builders):
        with contextlib.suppress(Exception):
            if builder.is_building:
                builder.end_building()
        with contextlib.suppress(Exception):
            _cuda.end_capture(int(builder.stream.handle))
    for builder in builders:
        ctypes.pythonapi.Py_IncRef(ctypes.py_object(builder))
    _ABANDONED.append(builders)


@functools.cache
def _cudart() -> Any:
    """The CUDA runtime library torch loaded (Linux: from ``/proc/self/maps``), or None."""
    try:
        with open("/proc/self/maps") as maps:
            paths = [line.split()[-1] for line in maps if "libcudart.so" in line]
        if not paths:
            return None
        lib = ctypes.CDLL(paths[0])
        lib.cudaGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.cudaGraphLaunch.restype = ctypes.c_int
        return lib
    except (OSError, AttributeError):
        return None


def support_reason(
    *,
    cuda: bool,
    driver: tuple[int, int] | None,
    core: Any,
    mem_pool: bool,
) -> str | None:
    """Why conditional graph nodes cannot carry a loop here (None: they can be tried):
    ``cuda`` a CUDA device, ``driver`` the driver's CUDA version, ``core`` the imported
    ``cuda.core`` module (None: missing), ``mem_pool`` torch has ``MemPool`` and
    ``use_mem_pool``."""
    if not cuda:
        return "no CUDA device"
    if driver is None:
        return "the CUDA driver version is unknown (cuDriverGetVersion failed)"
    if tuple(driver) < MIN_DRIVER:
        need = ".".join(map(str, MIN_DRIVER))
        return f"the driver supports CUDA {driver[0]}.{driver[1]}, conditional nodes need {need}+"
    if core is None:
        return "cuda.core is not installed (pip install cuda-core)"
    builder = getattr(getattr(core, "graph", core), "GraphBuilder", None)
    if builder is None or not all(
        hasattr(builder, name) for name in ("create_condition", "while_loop", "if_then")
    ):
        version = getattr(core, "__version__", "?")
        return f"cuda.core {version} has no conditional graph builder (while_loop, if_then)"
    if not mem_pool:
        return f"torch {torch.__version__} has no torch.cuda.MemPool / use_mem_pool"
    return None


def conditional_support() -> str | None:
    """:func:`support_reason` for this process (the static checks; :func:`probe` builds and
    runs a graph)."""
    try:
        core = _cuda.core()
    except Exception:
        core = None
    cuda = _cuda.available()
    return support_reason(
        cuda=cuda,
        driver=_cuda.driver_version() if cuda else None,
        core=core,
        mem_pool=hasattr(torch.cuda, "MemPool") and hasattr(torch.cuda, "use_mem_pool"),
    )


@functools.cache
def _kernels(core: Any, arch: str, which: str = "while") -> dict[str, Any]:
    """The WHILE graph's kernels (``which="count"``: ``ka_loop_count``), compiled once per
    process, ``cuda.core`` and architecture."""
    source = KERNELS_SRC if which == "while" else COUNT_SRC
    options = core.ProgramOptions(arch=arch, std="c++17")
    module = core.Program(source, code_type="c++", options=options).compile("cubin")
    return {name: module.get_kernel(name) for name in NAMES[which]}


# ------------------------------------------------------------------ masked writes


def masked_copy_(dst: torch.Tensor, src: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """``dst.copy_(src)`` in an active step; a masked step (an unrolled block's steps after
    the stop) writes ``dst``'s own values back."""
    return dst.copy_(torch.where(active, src.to(dst.dtype), dst))


def masked_index_copy_(
    dst: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor, active: torch.Tensor
) -> torch.Tensor:
    """``dst.index_copy_(dim, index, src)`` in an active step; a masked step writes the old
    values back (a KV-cache slot or an output row past the stop stays as it was)."""
    old = dst.index_select(dim, index)
    return dst.index_copy_(dim, index, torch.where(active, src.to(dst.dtype), old))


# ------------------------------------------------------------------ unroll choice


def measure_unroll(
    replay: Callable[[], Any], sync: Callable[[], Any], n: int = MEASURE_REPLAYS, repeats: int = 3
) -> tuple[float, float]:
    """``(step_s, launch_s)`` of a one-step graph: ``replay(); sync()`` takes one launch and
    one step, ``n`` replays and one sync take ``n`` steps and one launch while the GPU is
    the bottleneck (``n`` host launches while the host is: then the "step" is the host's
    launch time, the rate the loop can go at). Minimum of ``repeats``."""
    replay()
    sync()
    one = many = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        replay()
        sync()
        one = min(one, time.perf_counter() - start)
        start = time.perf_counter()
        for _ in range(n):
            replay()
        sync()
        many = min(many, time.perf_counter() - start)
    step = max((many - one) / (n - 1), 0.0) if n > 1 else 0.0
    return step, max(one - step, 0.0)


def choose_unroll(
    step_s: float, launch_s: float, steps: int, max_unroll: int = MAX_UNROLL, lookahead: bool = True
) -> int:
    """The block size K with the least expected time for a run of ``steps`` steps: the run
    takes the longer of the GPU's ``steps * step_s`` and the host's ``ceil(steps / K)``
    block launches and checks (``launch_s`` each), plus the masked steps after the stop:
    (K - 1) / 2 on average in its block and, with the read one block ahead
    (``lookahead``), the whole next block. The smallest K among equals."""
    steps = max(int(steps), 1)
    best_k, best = 1, math.inf
    for k in range(1, max(1, min(max_unroll, steps)) + 1):
        blocks = math.ceil(steps / k)
        waste = (k - 1) / 2 + (k if lookahead else 0)
        cost = max(steps * step_s, blocks * launch_s) + waste * step_s
        if cost < best * (1 - 1e-9):
            best_k, best = k, cost
    return best_k


# ------------------------------------------------------------------ the loop


class DeviceLoop:
    """A generation loop run on the device (see the module docstring); made by
    :func:`device_loop`.

    ``index`` (int64) and ``active`` (bool) are the loop's device scalars: the step reads
    them (``step(index, active)``); after a run ``index`` holds the steps done."""

    def __init__(
        self,
        step: Callable[[torch.Tensor, torch.Tensor], Any],
        cond: Callable[[], torch.Tensor] | torch.Tensor | None,
        max_steps: int,
        *,
        min_steps: int = 0,
        mode: str = "auto",
        masked: bool = False,
        unroll: int | None = None,
        chunk_every: int | None = None,
        on_chunk: Callable[[], Any] | None = None,
        watch: Sequence[tuple[Any, str]] = (),
        check_state: Sequence[torch.Tensor] = (),
        warmup_runs: int = 1,
        strict: bool = False,
        device: torch.device | str | int | None = None,
    ) -> None:
        if mode not in ("auto", *MODES):
            raise ValueError(f"mode must be auto or one of {MODES}, got {mode!r}")
        if max_steps < 1 or min_steps < 0:
            raise ValueError(
                f"need max_steps >= 1 and min_steps >= 0, got {max_steps}, {min_steps}"
            )
        if chunk_every is not None and chunk_every < 1:
            raise ValueError(f"chunk_every must be >= 1, got {chunk_every}")
        if on_chunk is not None and chunk_every is None:
            raise ValueError("on_chunk needs chunk_every")
        if unroll is not None and unroll < 1:
            raise ValueError(f"unroll must be >= 1, got {unroll}")
        self.step = step
        self.cond = cond
        self.max_steps = int(max_steps)
        self.min_steps = int(min_steps)
        self.preferred = mode
        self.masked = masked
        self.unroll = unroll
        self.chunk_every = chunk_every
        self.on_chunk = on_chunk
        self.check_state = list(check_state)
        self.warmup_runs = max(int(warmup_runs), 0)
        self.strict = strict
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        # the CUDA device of the graph calls (the CPU tests run them on a fake)
        self._dev = (self.device.index or 0) if self.device.type == "cuda" else 0
        # what teacher forcing and other hooks may replace: never graph them replaced
        self._watched = [(obj, name, getattr(obj, name)) for obj, name in watch]
        with torch.inference_mode(False):  # updated in place, in and out of inference mode
            self.index = torch.zeros((), dtype=torch.int64, device=self.device)
            self.active = torch.ones((), dtype=torch.bool, device=self.device)
            # [min_steps, max_steps]: a run may lower the limit without a new graph
            self._limits = torch.tensor([self.min_steps, self.max_steps], device=self.device)
            self._status = torch.zeros(2, dtype=torch.int64, device=self.device)
        self._limit = self.max_steps
        #: The built mode (None until built) and why it is the one (the fallbacks' reasons).
        self.mode: str | None = None
        self.reason = ""
        self.stats: dict[str, Any] = {
            "runs": 0,
            "host_runs": 0,
            "launches": 0,
            "host_checks": 0,
        }
        self._graph: Any = None  # the WHILE graph (cuda.core) or the unrolled block (torch)
        self._pool: Any = None
        self._host: torch.Tensor | None = None  # host-mapped (progress, cancel)
        self._streams: dict[int, Any] = {}
        self._event: Any = None  # after the last launched work (chunks() joins it)
        self._cancel = False
        self._keep: list[Any] = []  # what the graph references (streams, buffers)
        self._count: Any = False  # ka_loop_count (None: torch ops; False: not looked up)

    # -------------------------------------------------------------- the loop's own update

    def _flags(self) -> torch.Tensor | None:
        """The stop flags after a step, flat bool (None without ``cond``)."""
        if self.cond is None:
            return None
        flags = self.cond() if callable(self.cond) else self.cond
        if flags.numel() == 0:
            raise ValueError("cond() returned no stop flags")
        return flags.reshape(-1).bool().contiguous()

    def _counter(self) -> Any:
        """``ka_loop_count`` on a CUDA device with cuda.core and NVRTC (None: torch ops)."""
        if self._count is False:
            try:
                if self.device.type != "cuda":
                    raise Unsupported(f"the loop runs on {self.device}")
                major, minor = _cuda.capability(self._dev)
                core, _ = self._core_device()
                self._count = _kernels(core, f"sm_{major}{minor}", "count")["ka_loop_count"]
                self.stats["counter"] = "kernel"
            except Exception as exc:
                self._count = None
                self.stats["counter"] = f"torch ops ({type(exc).__name__}: {exc})"[:200]
        return self._count

    def _advance(self) -> None:
        """After the step (the host and unrolled modes; a WHILE graph runs the same rule in
        ``ka_loop_step``): count the step when it was active, stop at ``max_steps`` or when
        every stop flag is set (from ``min_steps`` on). Sticky: a masked step changes
        nothing. One kernel (``ka_loop_count``) where it compiles, else ~10 torch ops."""
        index, active = self.index, self.active
        flags = self._flags()
        if (kernel := self._counter()) is not None:
            core = _cuda.core()
            ptr, n = (flags.data_ptr(), flags.numel()) if flags is not None else (0, 0)
            args = (index.data_ptr(), active.data_ptr(), self._limits.data_ptr())
            one = core.LaunchConfig(grid=1, block=1)
            core.launch(self._core_stream(), one, kernel, ptr, n, *args, self._status.data_ptr())
            return
        following = index + 1
        stop = following >= self._limits[1]
        if flags is not None:
            stop = stop | (flags.all() & (following >= self._limits[0]))
        index.add_(active.to(index.dtype))
        active.logical_and_(stop.logical_not())

    def _body(self) -> None:
        self.step(self.index, self.active)
        self._advance()

    def _reset(self) -> None:
        self.index.zero_()
        self.active.fill_(True)

    def _set_limit(self, max_steps: int | None) -> None:
        limit = self.max_steps if max_steps is None else int(max_steps)
        if limit < 1:
            raise ValueError(f"max_steps must be >= 1, got {limit}")
        if limit != self._limit:
            self._limits[1].fill_(limit)
            self._limit = limit

    # -------------------------------------------------------------- building

    def _replaced(self) -> str | None:
        for obj, name, seen in self._watched:
            if getattr(obj, name) != seen:
                return f"{type(obj).__name__}.{name} is replaced (teacher forcing or a hook)"
        return None

    def build(self) -> str:
        """Build the loop's graph now (the first mode of the fallback chain that works) and
        return the mode; ``reason`` says why the modes before it did not."""
        if self.mode is not None:
            return self.mode
        if (why := self._replaced()) is not None:
            raise Unsupported(f"not built while {why}")
        start = MODES.index(self.preferred) if self.preferred != "auto" else 0
        reasons: list[str] = []
        for mode in MODES[start:]:
            try:
                if mode == "while":
                    self._build_while()
                elif mode == "unrolled":
                    self._build_unrolled()
            except Exception as exc:  # this mode cannot carry the loop here: say why
                if self.strict:
                    raise
                self._graph = self._pool = None
                reasons.append(f"{mode}: {type(exc).__name__}: {exc}"[:400])
                continue
            self.mode = mode
            self.reason = "; ".join(reasons) if reasons else f"{mode} built"
            self.stats["mode"] = mode
            return mode
        raise AssertionError("the host mode always builds")  # pragma: no cover

    def _capture_torch(self, builder: Any, fn: Callable[[], Any]) -> None:
        """``fn``'s torch work captured into ``builder``'s graph: its capture stream as the
        current torch stream, its allocations from the loop's private pool."""
        stream = _cuda.external_stream(int(builder.stream.handle), self._dev)
        self._keep.append(stream)
        with _cuda.use_pool(self._pool, self._dev), _cuda.use_stream(stream):
            fn()

    def _build_while(self) -> None:
        if not _cuda.graphs(self.device):
            raise Unsupported(f"the loop runs on {self.device}")
        if (why := conditional_support()) is not None:
            raise Unsupported(why)
        core = _cuda.core()
        major, minor = _cuda.capability(self._dev)
        kernels = _kernels(core, f"sm_{major}{minor}")
        _, dev = self._core_device()
        one = core.LaunchConfig(grid=1, block=1)
        index, active = self.index.data_ptr(), self.active.data_ptr()
        host = 0
        if self.chunk_every is not None:
            self._host = _cuda.pinned(2)
            host = self._host.data_ptr()
        self._pool = _cuda.mem_pool()
        limits, every = self._limits.data_ptr(), int(self.chunk_every or 0)
        builder = dev.create_graph_builder()
        opened = [builder]  # the builders, outermost first (a failure ends them innermost first)
        builder.begin_building()  # relaxed: torch's allocator may cudaMalloc while capturing
        try:
            loop = builder.create_condition(default_value=1)
            core.launch(builder, one, kernels["ka_loop_begin"], loop, index, active)
            body = builder.while_loop(loop)
            opened.append(body)
            body.begin_building()
            found: list[torch.Tensor | None] = []

            def step() -> None:
                self.step(self.index, self.active)
                found.append(self._flags())

            self._capture_torch(body, step)
            flags = found[0]
            self._keep.append(flags)  # its address is in the graph
            chunk = body.create_condition(default_value=0) if every else 0
            args = (
                (flags.data_ptr(), flags.numel()) if flags is not None else (0, 0),
                (index, active, limits, every, host),
            )
            core.launch(body, one, kernels["ka_loop_step"], loop, chunk, *args[0], *args[1])
            if every:
                then = body.if_then(chunk)
                opened.append(then)
                then.begin_building()
                if self.on_chunk is not None:
                    self._capture_torch(then, self.on_chunk)
                core.launch(then, one, kernels["ka_chunk_signal"], index, host)
                then.end_building()
            body.end_building()
            status = self._status.data_ptr()
            core.launch(builder, one, kernels["ka_loop_end"], index, active, status)
            builder.end_building()
            graph = builder.complete()
        except BaseException:
            _abandon(opened)
            raise
        graph.upload(self._core_stream())
        self._graph = graph
        self._keep.append(builder)

    def _block(self, k: int) -> Callable[[], None]:
        def block() -> None:
            for _ in range(k):
                self._body()
            if self._counter() is None:  # ka_loop_count writes the status every step
                self._status[0].copy_(self.index)  # what the host reads once per block
                self._status[1].copy_(self.active)

        return block

    def _check_masking(self) -> None:
        """One masked step (``active`` false) must leave ``check_state`` as it was."""
        if not self.check_state:
            return
        before = [t.clone() for t in self.check_state]
        self.active.fill_(False)
        self._body()
        pairs = enumerate(zip(before, self.check_state, strict=True))
        changed = [i for i, (old, new) in pairs if not old.equal(new)]
        self._reset()
        if changed:
            raise Unsupported(
                f"a masked step changed check_state {changed}: gate the step's writes with "
                "`active` (graphloop.masked_copy_, masked_index_copy_)"
            )

    def _build_unrolled(self) -> None:
        if not self.masked:
            raise Unsupported(
                "the step is not declared masked (masked=True: it gates its writes with "
                "`active`), so steps after the stop would write"
            )
        if self.on_chunk is not None:
            raise Unsupported("on_chunk runs at chunk boundaries only in an IF node or on the host")
        if not _cuda.graphs(self.device):
            raise Unsupported(f"the loop runs on {self.device}")
        self._counter()  # compiled before any capture
        self._check_masking()
        pool = _cuda.pool_handle()
        k = self.unroll
        if k is None:
            self.active.fill_(False)  # masked replays: the state stays as it is
            one = _cuda.capture_graph(self._block(1), pool)
            step_s, launch_s = measure_unroll(one.replay, lambda: _cuda.synchronize(self._dev))
            self._reset()
            k = choose_unroll(step_s, launch_s, self.max_steps)
            self.stats |= {"step_us": round(step_s * 1e6, 2), "launch_us": round(launch_s * 1e6, 2)}
            self._graph = one if k == 1 else None
        if self._graph is None:
            self._graph = _cuda.capture_graph(self._block(k), pool)
        self._pool = pool
        self.stats["unroll"] = k

    # -------------------------------------------------------------- running

    def _core_device(self) -> tuple[Any, Any]:
        """``cuda.core`` and the loop's device, its (primary, torch's) context current: module
        loads and stream wrappers need it."""
        core = _cuda.core()
        dev = core.Device(self._dev)
        dev.set_current()
        return core, dev

    def _core_stream(self) -> Any:
        """The caller's current torch stream as a cuda.core stream (cached per stream)."""
        current = _cuda.current_stream(self._dev)
        key = int(current.cuda_stream)
        if key not in self._streams:
            self._streams[key] = self._core_device()[1].create_stream(current)
        return self._streams[key]

    def _record(self) -> None:
        """An event after the loop's work on the caller's stream (what :meth:`join` waits
        for)."""
        if _cuda.graphs(self.device):
            self._event = _cuda.event()
            self._event.record(_cuda.current_stream(self._dev))

    def _call_mode(self, max_steps: int | None) -> str:
        """The mode of this run (host for the warm-up runs and while a watched callable is
        replaced; builds the graph after the warm-up)."""
        self._set_limit(max_steps)
        self.stats["runs"] += 1
        if (why := self._replaced()) is not None:
            self.stats["host_runs"] += 1
            self.stats["host_why"] = why
            return "host"
        if self.mode is None and self.stats["runs"] <= self.warmup_runs:
            self.stats["host_runs"] += 1
            return "host"
        return self.build()

    def run(self, max_steps: int | None = None) -> torch.Tensor:
        """Run the loop to its stop on the current stream; returns ``index`` (the steps
        done, a device tensor). ``max_steps``: this run's limit (default the loop's; the
        step's buffers must hold it). A WHILE graph returns after its launch (no host
        sync); its work is ordered on the caller's stream."""
        self._cancel = False
        mode = self._call_mode(max_steps)
        if mode == "while":
            self._launch()
        else:
            for _ in self._drive(mode):
                pass
        return self.index

    def chunks(self, max_steps: int | None = None) -> Iterator[int]:
        """Run the loop and yield the steps done at each chunk boundary (every
        ``chunk_every`` steps and at the stop) as soon as the host can see them, without a
        device-wide synchronize (a WHILE graph: a host-mapped counter polled every
        :data:`POLL_S`; it yields the latest count, so a slow reader may skip one).
        Leaving the iteration early cancels the rest of the loop (after its current step)
        and joins what was launched."""
        if self.chunk_every is None:
            raise ValueError("chunks() needs chunk_every")
        self.join()  # a previous launch must not write the counter after the reset below
        self._cancel = False
        if self._host is not None:
            self._host.zero_()
        mode = self._call_mode(max_steps)
        steps = self._poll_while() if mode == "while" else self._drive(mode)
        try:
            yield from steps
        finally:
            self._cancel = True
            if self._host is not None:
                self._host[1] = 1  # the condition kernel ends the loop after this step
            steps.close()
            self.join()
            if self._host is not None:
                self._host.zero_()
            self._cancel = False

    def cancel(self) -> None:
        """End a running :meth:`chunks` loop after its current step."""
        self._cancel = True
        if self._host is not None:
            self._host[1] = 1

    def join(self) -> None:
        """Wait until the work the loop launched last has finished (the host waits; the
        caller's stream is ordered after it anyway)."""
        if self._event is not None:
            self._event.synchronize()
            self._event = None

    def _launch(self) -> None:
        """The WHILE graph on the caller's stream, launched where the profiler sees it."""
        if not _cuda.runtime_launch(self._graph, _cuda.current_stream(self._dev)):
            self._graph.launch(self._core_stream())
            # the profiler may not record a driver-API launch: a torch op ordered after the
            # loop on the same stream shows the hidden-work check where it ends
            self._status.add_(0)
        self.stats["launches"] += 1
        self._record()

    def _poll_while(self) -> Generator[int]:
        assert self._host is not None
        self._launch()
        event, seen = self._event, 0
        while True:
            done = event.query()
            count = int(self._host[0])
            if count != seen:
                seen = count
                yield count
            if done:
                return
            time.sleep(POLL_S)

    def _drive(self, mode: str) -> Generator[int]:
        """The host-driven modes: the unrolled blocks or the plain loop. Yields the steps
        done at each chunk boundary the host saw (every check without ``chunk_every``)."""
        if mode == "unrolled":
            yield from self._drive_unrolled()
        else:
            yield from self._drive_host()

    def _boundary(self, before: int, now: int, active: bool) -> bool:
        if self.chunk_every is None:
            return False
        return not active or now // self.chunk_every > before // self.chunk_every

    def _drive_host(self) -> Generator[int]:
        self._reset()
        done = 0
        try:
            for _ in range(self._limit):
                if self._cancel:
                    break
                self._body()
                self.stats["host_checks"] += 1
                active = bool(self.active.item())  # the plain loop: one read per step
                if self._boundary(done, done + 1, active):
                    if self.on_chunk is not None:
                        self.on_chunk()
                    yield done + 1
                done += 1
                if not active:
                    break
        finally:
            self._record()

    def _drive_unrolled(self) -> Generator[int]:
        from kernel_agent.workloads.serving import AsyncFlags

        k = int(self.stats["unroll"])
        self._reset()
        flags = AsyncFlags(depth=2)
        pending: int | None = None
        done = 0
        try:
            for _ in range(math.ceil(self._limit / k)):
                if self._cancel:
                    break
                self._graph.replay()
                ticket = flags.send(self._status)
                self.stats["launches"] += 1
                if pending is not None:  # the previous block's status, one block behind
                    index, active = (int(v) for v in flags.read(pending).tolist())
                    self.stats["host_checks"] += 1
                    if self._boundary(done, index, bool(active)):
                        yield index
                    done = index
                    if not active:
                        pending = None
                        break
                pending = ticket
            if pending is not None and self.chunk_every is not None and not self._cancel:
                # the limit ended the loop in the last block: its count is the last chunk's
                index, active = (int(v) for v in flags.read(pending).tolist())
                self.stats["host_checks"] += 1
                if self._boundary(done, index, bool(active)):
                    yield index
        finally:
            self._record()


def device_loop(
    step: Callable[[torch.Tensor, torch.Tensor], Any],
    cond: Callable[[], torch.Tensor] | torch.Tensor | None,
    max_steps: int,
    **options: Any,
) -> DeviceLoop:
    """A :class:`DeviceLoop` running ``step(index, active)`` until every element of
    ``cond()`` (or of the ``cond`` tensor; evaluated after each step, nonzero = stop) is
    set from ``min_steps`` on, or ``max_steps`` steps (see the module docstring). Options:
    ``min_steps``, ``mode`` (``auto``: while → unrolled → host), ``masked`` (the step gates
    its writes with ``active``: needed by the unrolled fallback), ``unroll`` (K; default
    measured), ``chunk_every`` / ``on_chunk`` (streaming), ``watch`` (callables that must be
    called from Python: teacher forcing), ``check_state`` (tensors a masked step must not
    change, checked once before the unrolled capture), ``warmup_runs``, ``strict`` (raise
    instead of falling back), ``device``."""
    return DeviceLoop(step, cond, max_steps, **options)


# ------------------------------------------------------------------ doctor


def probe() -> tuple[bool | None, str]:
    """``doctor``: a loop of three steps on the device, run twice through a WHILE graph with
    a chunk IF node (and once through :meth:`DeviceLoop.chunks`): ``(True, detail)`` when
    every count is right, ``(None, why)`` where conditional nodes are unavailable (loops use
    the unrolled graphs there), ``(False, why)`` when the graph is wrong."""
    if (why := conditional_support()) is not None:
        return None, f"{why}; device_loop uses K-step unrolled CUDA graphs"
    device = torch.device("cuda", torch.cuda.current_device())
    with torch.inference_mode(False):
        count = torch.zeros(2, dtype=torch.int64, device=device)  # steps, chunk boundaries

    def step(index: torch.Tensor, active: torch.Tensor) -> None:
        count[0].add_(1)

    loop = DeviceLoop(
        step,
        lambda: count[0].remainder(3) == 0,  # stops after 3 steps
        max_steps=8,
        mode="while",
        chunk_every=2,
        on_chunk=lambda: count[1].add_(1),  # at steps 2 and 3 (the stop)
        warmup_runs=0,
        strict=True,
        device=device,
    )
    start = time.perf_counter()
    loop.run()
    loop.run()
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    seen = list(loop.chunks())  # the host sees step 3; step 2 only when it polls in time
    got = (count.tolist(), int(loop.index), seen[-1:], loop.stats["host_checks"])
    want = ([9, 6], 3, [3], 0)
    if got != want:
        return False, (
            "a WHILE loop ran wrong: (steps and chunk boundaries, index, last chunk seen, "
            f"host checks) = {got}, want {want}"
        )
    return True, (
        "a WHILE graph ran 3 steps three times with no host check; its chunk IF node ran at "
        f"steps 2 and 3 and signalled the host ({seconds * 1e3:.1f} ms incl. build)"
    )
