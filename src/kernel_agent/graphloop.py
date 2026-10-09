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
``.cpu()``), no Python state that changes from step to step. A ``torch.compile`` step in
the default mode (Inductor's kernels, no CUDA graphs of its own) is captured into the WHILE
body like eager ops: the warm-up run compiles it, the graph replays its kernels (the same
tokens as the compiled step's host loop; measured on an NVIDIA A10, sm_86, torch 2.10, and
probed by ``doctor``). ``mode="reduce-overhead"`` makes CUDA graphs of its own: not inside a
device loop.

Random numbers: a step may draw from the device's default CUDA generator (``torch.randn``,
``rand``, ``multinomial``, dropout, without ``generator=``), and every mode gives the plain
loop's draws, step for step. Torch makes RNG kernels graph-safe for its own graphs (Note
[CUDA Graph-safe RNG states], ``ATen/cuda/CUDAGeneratorImpl.h``): under a capture they read
the seed and the Philox offset from two device scalars of the generator state, plus the
offset the capture counted so far; a replay fills the scalars from the host generator and
advances it by the graph's whole increment. A WHILE body runs many times per launch, so the
loop does the same per iteration (:class:`_GraphRng`): the step is captured while a clone of
the generator state is in capture mode, the launch fills the clone's scalars, and
``ka_loop_step`` advances the offset by the step's increment on the device, so step ``i``
reads base + ``i`` x increment, the plain loop's offsets. The unrolled blocks are torch's
own graphs. ``rng`` says where a run leaves the generator: ``"exact"`` (default) where the
plain loop leaves it (steps x increment; a WHILE ``run()`` then waits for the loop to read
its steps, an unrolled run corrects its masked steps' draws), ``"reserve"`` at max_steps x
increment in every mode whatever the stop (no wait; later draws differ from the plain
loop's after an early stop, they never repeat the loop's). ``on_chunk`` must not draw. Draws
from another generator (``generator=g``) are not captured: such a step runs host steps.

Streaming (``metric=ttfa``): with ``chunk_every=n`` the WHILE body ends with an IF node that
runs at every n-th step and at the stop: ``on_chunk()`` (captured, optional) and a kernel
that writes the steps done to a host-mapped counter. :meth:`DeviceLoop.chunks` launches the
loop and yields those counts as the host sees them, no device-wide synchronize; leaving
the iteration early (the metric window) cancels the rest of the loop through a host-mapped
flag the condition kernel reads and joins what was launched. The unrolled fallback has two
block graphs, K steps and K steps then ``on_chunk()``, with K a divisor of n so that every
chunk boundary ends a block: the second runs the blocks that end at a boundary, launched
once the block before is seen active (after a stop it would run ``on_chunk`` again: the host
waits once per chunk), and once more, its steps masked, for a stop in a plain block. So
``on_chunk`` runs at the same steps as in the WHILE graph, once each.

Counters: a run's steps stay on the device (``run()`` returns them as a tensor). With
``workload=`` the loop is a stats source of that workload
(:meth:`~kernel_agent.workloads.base.Workload.add_stats_source`): its graphs add each run's
steps to a device total at no extra launch (the WHILE graph's end kernel, ``ka_loop_count``),
which the workload reads before and after a timed run's clock, where the host waits
anyway, and reports as the ``report`` counters (default ``steps``; a plain decode loop:
``("steps", "tokens")``). No host sync is added to the timed run, and ``decode_stats`` of
an evaluation or an A/B sees the steps.

Integrity: the graph runs on the caller's current stream, so it is joined with that stream
and the evaluator's timing sees all of it; :meth:`DeviceLoop.chunks` waits for it before
it returns. It is launched with ``cudaGraphLaunch`` of the CUDA runtime torch loaded (the
profiler does not record cuda.core's driver-API launch), and it begins and ends with a
kernel outside the WHILE node, because the profiler does not list the body's kernels under
the launch: it lists the last iteration's only (measured with torch 2.14, CUDA 13.0) or,
once CUPTI was initialised before the graph was built (an earlier profiled run in the
process), every iteration's on a stream id of their own under correlation ids that are not
the launch's (measured on an NVIDIA A10, torch 2.10). So the end-to-end hidden-work check
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

from kernel_agent import toolchain

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
#: Where a run leaves the CUDA generator when the step draws random numbers (``rng=``).
RNG_POLICIES = ("exact", "reserve")

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
// steps and at the stop). One thread: ``n_flags`` is a batch's flags at most. A step that
// draws random numbers: its RNG kernels read the Philox offset at ``rng_offset`` (plus their
// offsets within the step), so step i reads the launch's base + i * rng_inc.
extern "C" __global__ void ka_loop_step(cudaGraphConditionalHandle loop,
                                        cudaGraphConditionalHandle chunk, const bool* flags,
                                        long long n_flags, long long* index, bool* active,
                                        const long long* limits, long long every,
                                        const volatile long long* host, long long* rng_offset,
                                        long long rng_inc) {
  const long long i = *index + 1;
  bool done = n_flags > 0 && i >= limits[0];
  for (long long f = 0; done && f < n_flags; ++f) done = flags[f];
  bool stop = done || i >= limits[1];
  if (host != nullptr && host[1] != 0) stop = true;
  if (rng_offset != nullptr) *rng_offset += rng_inc;
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
// loop's span); it leaves the (index, active) status the unrolled blocks leave and adds the
// run's steps to the loop's running total (read on the host outside the timed run).
extern "C" __global__ void ka_loop_end(const long long* index, const bool* active,
                                       long long* status, long long* total) {
  status[0] = *index;
  status[1] = *active ? 1 : 0;
  *total += *index;
}
"""
#: The same rule without a graph condition, for the unrolled and host modes (no conditional
#: node API needed): a masked step (``active`` false) changes nothing; ``status`` is the
#: (index, active) pair the host reads once per unrolled block, ``total`` the steps of
#: every run.
COUNT_SRC = r"""
extern "C" __global__ void ka_loop_count(const bool* flags, long long n_flags,
                                         long long* index, bool* active,
                                         const long long* limits, long long* status,
                                         long long* total) {
  if (*active) {
    const long long i = *index + 1;
    bool done = n_flags > 0 && i >= limits[0];
    for (long long f = 0; done && f < n_flags; ++f) done = flags[f];
    *index = i;
    *active = !(done || i >= limits[1]);
    *total += 1;
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
        """``fn`` captured into a ``torch.cuda.CUDAGraph`` (its ``replay()`` runs it).

        A capture that fails (a host sync in ``fn``) leaves torch 2.10's state behind: its
        ``torch.cuda.graph`` exit raises before it restores the current stream (later work
        would run on the capture stream) and the allocator still routes to ``pool``, so the
        next ``MemPool`` destructor aborts the process (``captures_underway.empty()``;
        measured on an A10 with torch 2.10). The outer stream context restores the stream,
        and ending the allocation to the pool clears the allocator (an error where torch
        already did it is ignored). It also leaves the default CUDA generator in capture mode
        (a step that draws, then syncs: every later draw of the process raised "Offset
        increment outside graph capture encountered unexpectedly"; A10, torch 2.10):
        :meth:`leave_rng_capture` ends it."""
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.stream(torch.cuda.current_stream()), torch.cuda.graph(graph, pool=pool):
                fn()
        except BaseException:
            device = torch.cuda.current_device()
            with contextlib.suppress(Exception):
                torch._C._cuda_endAllocateToPool(device, pool)
            with contextlib.suppress(Exception):
                self.leave_rng_capture(device)
            raise
        return graph

    def leave_rng_capture(self, device: int, *states: Any) -> None:
        """End the capture mode of the device's default generator state and of ``states``
        (generators): torch's capture end runs the generator states' capture epilogue only
        when the capture ended cleanly. A clean capture of a torch graph registered with
        them runs it (its prologue resets what a failed capture counted)."""
        side = torch.cuda.Stream(device=device)
        fix = torch.cuda.CUDAGraph()
        for state in states:
            fix.register_generator_state(state)
        with torch.cuda.stream(side):
            fix.capture_begin(capture_error_mode="relaxed")
            torch.cuda._sleep(1)  # not an empty graph
            fix.capture_end()

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

    def generator(self, device: int) -> Any:
        """The device's default CUDA generator (``torch.randn`` & co. without
        ``generator=``)."""
        torch.cuda.init()
        return torch.cuda.default_generators[device]

    def graph_rng(self, generator: Any, device: int) -> Any:
        """A :class:`_GraphRng` of ``generator`` (raises where torch does not allow it)."""
        return _GraphRng(generator, device)


#: The seed of the loop's clone of a generator state while its device scalars are told
#: apart (any value other than 0, the offset a capture prologue writes).
_SENTINEL = 0x5EED_1DE5_7A7E


def _int64(value: int) -> int:
    """A uint64 (a generator's seed) as the int64 of the same bits (a tensor's fill value)."""
    return value - (1 << 64) if value >= 1 << 63 else value


class _DeviceInt64:
    """``__cuda_array_interface__`` of one int64 at a device address (a tensor view of
    memory torch's generator state owns)."""

    def __init__(self, address: int) -> None:
        self.__cuda_array_interface__ = {
            "shape": (1,),
            "typestr": "<i8",
            "data": (address, False),
            "version": 3,
        }


class _GraphRng:
    """Torch's graph-safe Philox state, for a WHILE body (see the module docstring).

    Under a capture, an RNG kernel reads the seed and the Philox offset from two device
    scalars of the generator state (``seed_extragraph_``, ``offset_extragraph_``) plus the
    offset counted since the capture began; only torch's own captures put a state in that
    mode, and torch exposes neither scalar. So:

    * the scalars: a clone of the generator's state (``clone_state``) registered with a torch
      graph (``register_generator_state``, which allocates them) while the allocator routes
      to a private pool: they are that pool's two blocks (``MemPool.snapshot``), told apart
      by what a capture prologue writes (the clone's seed, offset 0);
    * capture mode (:meth:`capture`): a torch graph capture (the anchor, on a side stream) is
      open around the step's capture into the WHILE body, with the clone in the generator
      (``graphsafe_set_state``); the anchor's capture end ends capture mode and one replay of
      the anchor graph reads the step's increment (:meth:`increment`: torch advances the
      clone's offset by the graph's whole increment).

    The WHILE graph's RNG kernels then read the clone's scalars: each launch fills them from
    the generator (:attr:`seed`, :attr:`offset`) and ``ka_loop_step`` advances the offset.
    Measured on an NVIDIA A10 (torch 2.10): ``randn``, ``rand`` and ``multinomial`` steps
    draw the plain loop's numbers."""

    def __init__(self, generator: Any, device: int) -> None:
        self.generator = generator
        self.side = torch.cuda.Stream(device=device)
        self.clone = generator.clone_state()
        self.clone.manual_seed(_SENTINEL)
        self.pool = torch.cuda.MemPool()
        self.holder = torch.cuda.CUDAGraph()  # keeps the clone's scalars allocated
        with torch.cuda.use_mem_pool(self.pool, device):
            self.holder.register_generator_state(self.clone)
        blocks = [
            int(block["address"])
            for segment in self.pool.snapshot()
            for block in segment["blocks"]
            if block["state"] == "active_allocated"
        ]
        if len(blocks) != 2:
            raise Unsupported(
                f"torch {torch.__version__}: a generator state's device seed and offset were "
                f"not found ({len(blocks)} blocks in its pool, want 2)"
            )
        with torch.cuda.stream(self.side):
            self.holder.capture_begin(capture_error_mode="relaxed")  # its prologue writes them
            torch.cuda._sleep(1)  # not an empty graph
            self.holder.capture_end()
        self.side.synchronize()
        cuda = torch.device("cuda", device)
        views = [torch.as_tensor(_DeviceInt64(address), device=cuda) for address in blocks]
        values = [int(view) for view in views]
        if sorted(values) != sorted([0, _SENTINEL]):
            raise Unsupported(
                f"torch {torch.__version__}: a generator state's device scalars hold {values} "
                f"after a capture prologue, want its seed and offset 0"
            )
        #: The device scalars the captured RNG kernels read (int64 views).
        self.seed = views[values.index(_SENTINEL)]
        self.offset = views[values.index(0)]
        self.anchor = torch.cuda.CUDAGraph()
        self.anchor.register_generator_state(self.clone)

    @contextlib.contextmanager
    def capture(self) -> Iterator[None]:
        """The clone in capture mode and in the generator, for the step's capture."""
        with torch.cuda.stream(self.side):
            self.anchor.capture_begin(capture_error_mode="relaxed")
        original = self.generator.graphsafe_get_state()
        self.generator.graphsafe_set_state(self.clone)
        try:
            yield
        except BaseException:
            self._end(original, failed=True)
            raise
        self._end(original, failed=False)

    def _end(self, original: Any, failed: bool) -> None:
        """The generator's own state back, the anchor's capture ended; where that fails, the
        states still leave capture mode (the step's error is the one raised)."""
        self.generator.graphsafe_set_state(original)
        try:
            with torch.cuda.stream(self.side):
                torch.cuda._sleep(1)  # not an empty graph
                self.anchor.capture_end()
        except Exception:
            _cuda.leave_rng_capture(self.side.device.index, self.clone)
            if not failed:
                raise

    def increment(self) -> int:
        """The Philox offsets the captured step takes (0: it draws nothing)."""
        before = self.clone.get_offset()
        with torch.cuda.stream(self.side):
            self.anchor.replay()
        return int(self.clone.get_offset() - before)


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
    kind = "ptx" if arch.startswith("compute_") else "cubin"  # emulated GPU: PTX (#252)
    module = core.Program(source, code_type="c++", options=options).compile(kind)
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


def aligned_unroll(k: int, every: int) -> int:
    """The largest block size up to ``k`` that divides ``every``: with ``on_chunk`` every
    chunk boundary (each ``every``-th step) must end an unrolled block, the block that ends
    with ``on_chunk``."""
    return max(d for d in range(1, max(1, min(k, every)) + 1) if every % d == 0)


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
        workload: Any = None,
        report: Sequence[str] = ("steps",),
        rng: str = "exact",
    ) -> None:
        if mode not in ("auto", *MODES):
            raise ValueError(f"mode must be auto or one of {MODES}, got {mode!r}")
        if rng not in RNG_POLICIES:
            raise ValueError(f"rng must be one of {RNG_POLICIES}, got {rng!r}")
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
            # the steps of every run, counted on the device (read by stats_mark / since)
            self._total = torch.zeros((), dtype=torch.int64, device=self.device)
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
        self._chunk_graph: Any = None  # the unrolled block that ends with on_chunk
        self._pool: Any = None
        self._host: torch.Tensor | None = None  # host-mapped (progress, cancel)
        self._streams: dict[int, Any] = {}
        self._event: Any = None  # after the last launched work (chunks() joins it)
        self._cancel = False
        self._keep: list[Any] = []  # what the graph references (streams, buffers)
        self._count: Any = False  # ka_loop_count (None: torch ops; False: not looked up)
        self.rng = rng
        #: Philox offsets a step takes from the device's default generator (None: not yet
        #: measured; 0: the step draws nothing), the WHILE graph's :class:`_GraphRng`, the
        #: offsets this run took (``limit`` or the steps run, masked ones too) and the seed
        #: the WHILE graph's scalar holds.
        self._rng_inc: int | None = None
        self._graph_rng: Any = None
        self._rng_drawn = 0
        self._rng_seed: int | None = None
        #: The counters each run's steps are reported as (``workload``'s stats source).
        self.report = tuple(report)
        if workload is not None:
            workload.add_stats_source(self)

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
                arch, _ = toolchain.nvrtc_target(_cuda.capability(self._dev))
                core, _ = self._core_device()
                self._count = _kernels(core, arch, "count")["ka_loop_count"]
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
            out = (self._status.data_ptr(), self._total.data_ptr())
            one = core.LaunchConfig(grid=1, block=1)
            core.launch(self._core_stream(), one, kernel, ptr, n, *args, *out)
            return
        following = index + 1
        stop = following >= self._limits[1]
        if flags is not None:
            stop = stop | (flags.all() & (following >= self._limits[0]))
        counted = active.to(index.dtype)
        index.add_(counted)
        self._total.add_(counted)
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

    # -------------------------------------------------------------- random numbers

    def _generator(self) -> Any:
        """The device's default CUDA generator, which the step's draws advance (None off a
        CUDA device: host steps draw as the plain loop does)."""
        return _cuda.generator(self._dev) if _cuda.graphs(self.device) else None

    def _set_rng_inc(self, inc: int) -> None:
        self._rng_inc = int(inc)
        if inc:
            self.stats["rng_offsets_per_step"] = int(inc)

    def _new_graph_rng(self) -> Any:
        """A :class:`_GraphRng` for the WHILE body (None: the warm-up saw a step that draws
        nothing, or torch does not allow it here: a step that draws then fails its capture
        and the loop falls back; ``stats["rng"]`` says why)."""
        gen = self._generator()
        if gen is None or self._rng_inc == 0:
            return None
        try:
            return _cuda.graph_rng(gen, self._dev)
        except Exception as exc:
            self.stats["rng"] = f"no graph-safe RNG state: {type(exc).__name__}: {exc}"[:300]
            return None

    def _rng_launch(self) -> None:
        """Before a WHILE launch whose step draws: the graph's seed and offset scalars from
        the generator (stream-ordered fills, as torch's own replays make), and the generator
        past the run's draws, ``limit`` steps (:meth:`_rng_settle` moves it to the steps
        run where ``rng="exact"``)."""
        gen, rng, inc = self._generator(), self._graph_rng, int(self._rng_inc or 0)
        seed = int(gen.initial_seed())
        if seed != self._rng_seed:  # the scalar is the loop's own: only a new seed changes it
            rng.seed.fill_(_int64(seed))
            self._rng_seed = seed
        offset = int(gen.get_offset())
        rng.offset.fill_(offset)
        gen.set_offset(offset + self._limit * inc)
        self._rng_drawn = self._limit

    def _rng_settle(self, steps: int | None) -> None:
        """After a run whose step draws: the generator where ``rng`` leaves it, from where the
        run's draws left it (``_rng_drawn`` steps: a WHILE graph's ``limit``, an unrolled
        run's masked steps too): ``exact`` the run's ``steps`` (None: read from the device,
        which waits for the loop), ``reserve`` its limit."""
        inc = self._rng_inc
        if not inc or (gen := self._generator()) is None:
            return
        if self.rng == "reserve":
            target = self._limit
        else:
            target = int(self.index) if steps is None else int(steps)
        if target != self._rng_drawn:
            gen.set_offset(gen.get_offset() + (target - self._rng_drawn) * inc)
            self._rng_drawn = target

    # -------------------------------------------------------------- building

    def _replaced(self) -> str | None:
        for obj, name, seen in self._watched:
            if getattr(obj, name) != seen:
                return f"{type(obj).__name__}.{name} is replaced (teacher forcing or a hook)"
        return None

    def build(self) -> str:
        """Build the loop's graph now (the first mode of the fallback chain that works) and
        return the mode; ``reason`` says why the modes before it did not. The generator is
        where it was (the build's masked and measuring steps draw too)."""
        if self.mode is not None:
            return self.mode
        if (why := self._replaced()) is not None:
            raise Unsupported(f"not built while {why}")
        gen = self._generator()
        offset = gen.get_offset() if gen is not None else None
        try:
            return self._build()
        finally:
            if offset is not None:
                gen.set_offset(offset)

    def _build(self) -> str:
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
                self._graph = self._chunk_graph = self._pool = None
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
        kernels = _kernels(core, toolchain.nvrtc_target(_cuda.capability(self._dev))[0])
        _, dev = self._core_device()
        one = core.LaunchConfig(grid=1, block=1)
        index, active = self.index.data_ptr(), self.active.data_ptr()
        host = 0
        if self.chunk_every is not None:
            self._host = _cuda.pinned(2)
            host = self._host.data_ptr()
        self._pool = _cuda.mem_pool()
        limits, every = self._limits.data_ptr(), int(self.chunk_every or 0)
        rng = self._new_graph_rng()
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

            # the step's draws (and cond's) read the RNG state's device scalars; on_chunk's
            # are refused (captured after the RNG state left capture mode)
            with rng.capture() if rng is not None else contextlib.nullcontext():
                self._capture_torch(body, step)
            inc = rng.increment() if rng is not None else 0
            flags = found[0]
            self._keep.append(flags)  # its address is in the graph
            chunk = body.create_condition(default_value=0) if every else 0
            args = (
                (flags.data_ptr(), flags.numel()) if flags is not None else (0, 0),
                (index, active, limits, every, host),
                (rng.offset.data_ptr(), inc) if inc else (0, 0),
            )
            core.launch(
                body, one, kernels["ka_loop_step"], loop, chunk, *args[0], *args[1], *args[2]
            )
            if every:
                then = body.if_then(chunk)
                opened.append(then)
                then.begin_building()
                if self.on_chunk is not None:
                    self._capture_torch(then, self.on_chunk)
                core.launch(then, one, kernels["ka_chunk_signal"], index, host)
                then.end_building()
            body.end_building()
            status, total = self._status.data_ptr(), self._total.data_ptr()
            core.launch(builder, one, kernels["ka_loop_end"], index, active, status, total)
            builder.end_building()
            graph = builder.complete()
        except BaseException:
            _abandon(opened)
            raise
        graph.upload(self._core_stream())
        self._graph = graph
        self._keep.append(builder)
        self._set_rng_inc(inc)
        if inc:
            self._graph_rng = rng  # the graph reads its scalars
            self._keep.append(rng)

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

    def _chunk_block(self, k: int) -> Callable[[], None]:
        """K steps, then ``on_chunk`` (the unrolled block that ends at a chunk boundary)."""
        steps, on_chunk = self._block(k), self.on_chunk
        assert on_chunk is not None

        def block() -> None:
            steps()
            on_chunk()

        return block

    def _build_unrolled(self) -> None:
        if not self.masked:
            raise Unsupported(
                "the step is not declared masked (masked=True: it gates its writes with "
                "`active`), so steps after the stop would write"
            )
        if not _cuda.graphs(self.device):
            raise Unsupported(f"the loop runs on {self.device}")
        self._counter()  # compiled before any capture
        self._check_masking()
        pool = _cuda.pool_handle()
        k, one = self.unroll, None
        if k is None:
            self.active.fill_(False)  # masked replays: the state stays as it is
            one = _cuda.capture_graph(self._block(1), pool)
            step_s, launch_s = measure_unroll(one.replay, lambda: _cuda.synchronize(self._dev))
            self._reset()
            k = choose_unroll(step_s, launch_s, self.max_steps)
            self.stats |= {"step_us": round(step_s * 1e6, 2), "launch_us": round(launch_s * 1e6, 2)}
        if self.on_chunk is not None:
            assert self.chunk_every is not None
            k = aligned_unroll(k, self.chunk_every)  # every chunk boundary ends a block
        self._graph = one if one is not None and k == 1 else None
        if self._graph is None:
            self._graph = _cuda.capture_graph(self._block(k), pool)
        if (gen := self._generator()) is not None:
            # a masked block's replay: its steps draw (torch advances the generator by the
            # graph's increment), they write nothing; build() puts the generator back
            self.active.fill_(False)
            before = gen.get_offset()
            self._graph.replay()
            self._set_rng_inc((gen.get_offset() - before) // k)
            self._reset()
        if self.on_chunk is not None:
            # its own pool: the two block graphs replay in any order
            self._chunk_graph = _cuda.capture_graph(self._chunk_block(k), _cuda.pool_handle())
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
        self._rng_drawn = 0
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
            self._rng_settle(None)  # rng="exact" and a step that draws: waits for the loop
        else:
            for _ in self._drive(mode, streaming=False):
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
        steps = self._poll_while() if mode == "while" else self._drive(mode, streaming=True)
        try:
            yield from steps
        finally:
            self._cancel = True
            if self._host is not None:
                self._host[1] = 1  # the condition kernel ends the loop after this step
            steps.close()
            self.join()
            if mode == "while":
                self._rng_settle(None)  # joined: the read does not wait
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

    # -------------------------------------------------------------- counters

    def steps_total(self) -> int:
        """The steps of every run so far, counted on the device by the loop's own kernels
        (the WHILE graph's end kernel, ``ka_loop_count``; no extra launch). Reading it
        waits for the loop's work on the current stream: read it where the host waits
        anyway (a workload reads it after a timed run's synchronize)."""
        return int(self._total)

    def stats_mark(self) -> tuple[int, int]:
        """The runs and steps so far (:class:`~kernel_agent.workloads.base.StatsSource`:
        called after the synchronize before a timed run's clock starts)."""
        return self.stats["runs"], self.steps_total()

    def stats_since(self, mark: tuple[int, int] | None) -> dict[str, float]:
        """The steps of the runs since ``mark`` (None: since the loop was made) as each
        counter of ``report``; ``{}`` when the loop did not run (an A/B's other state)."""
        runs, steps = mark if mark is not None else (0, 0)
        if self.stats["runs"] == runs or not self.report:
            return {}
        return dict.fromkeys(self.report, self.steps_total() - steps)

    def _launch(self) -> None:
        """The WHILE graph on the caller's stream, launched where the profiler sees it."""
        if self._graph_rng is not None:
            self._rng_launch()
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

    def _drive(self, mode: str, streaming: bool) -> Generator[int]:
        """The host-driven modes: the unrolled blocks or the plain loop. Yields the steps
        done at each chunk boundary the host saw (``streaming``: :meth:`chunks` reads
        them)."""
        if mode == "unrolled":
            yield from self._drive_unrolled(streaming)
        else:
            yield from self._drive_host()

    def _boundary(self, before: int, now: int, active: bool) -> bool:
        if self.chunk_every is None:
            return False
        return not active or now // self.chunk_every > before // self.chunk_every

    def _drive_host(self) -> Generator[int]:
        self._reset()
        done = 0
        gen = self._generator()
        try:
            for _ in range(self._limit):
                if self._cancel:
                    break
                before = gen.get_offset() if gen is not None else 0
                self._body()
                if gen is not None and self._rng_inc is None:  # the step's draws, measured
                    self._set_rng_inc(gen.get_offset() - before)
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
            self._rng_drawn = done
            self._rng_settle(done)
            self._record()

    def _drive_unrolled(self, streaming: bool) -> Generator[int]:
        """K-step blocks until the stop, each block's status read one block behind (the
        GPU never waits for the host). With ``on_chunk``, a block that ends at a chunk
        boundary is the second graph (its K steps, then ``on_chunk``), launched only after
        the block before it was seen active: launched after a stop it would run
        ``on_chunk`` once more (the host waits there once per chunk). A stop in a plain
        block gets its ``on_chunk`` from one more replay of that graph, its steps masked.
        ``streaming`` (:meth:`chunks`): a chunk's work is done when its count is yielded."""
        from kernel_agent.workloads.serving import AsyncFlags

        k, chunk, every = int(self.stats["unroll"]), self._chunk_graph, self.chunk_every or 0
        flags = AsyncFlags(depth=2)
        done = 0
        stopped_at: int | None = None  # the steps, once a block's status showed the stop

        def launch(graph: Any) -> tuple[int, bool]:
            graph.replay()
            self.stats["launches"] += 1
            self._rng_drawn += k  # masked steps draw too
            return flags.send(self._status), graph is chunk

        def read(ticket: int) -> tuple[int, bool]:
            index, active = (int(v) for v in flags.read(ticket).tolist())
            self.stats["host_checks"] += 1
            return index, bool(active)

        def settle(ticket: int, chunked: bool) -> Generator[int, None, bool]:
            """A block's status on the host: yields the count at a chunk boundary or the
            stop, returns whether the loop stopped in it."""
            nonlocal done, stopped_at
            index, active = read(ticket)
            if not active:
                stopped_at = index
            if not active and chunk is not None and not chunked:
                ticket, _ = launch(chunk)  # the stop's on_chunk
                if streaming:
                    read(ticket)
            if self._boundary(done, index, active):
                yield index
            done = index
            return not active

        self._reset()
        pending: tuple[int, bool] | None = None  # the last block: its status, on_chunk ran
        try:
            for block in range(math.ceil(self._limit / k)):
                if self._cancel:
                    break
                chunked = chunk is not None and (block + 1) * k % every == 0
                if chunked and pending is not None:  # never on_chunk after a stop
                    stopped = yield from settle(*pending)
                    pending = None
                    if stopped:
                        break
                ticket = launch(chunk if chunked else self._graph)
                if pending is not None:  # the previous block's status, one block behind
                    stopped = yield from settle(*pending)
                    if stopped:
                        pending = None
                        break
                pending = ticket
            else:  # the loop stopped in its last block (at the limit at the latest)
                if pending is not None and chunk is not None and not pending[1]:
                    pending = launch(chunk)  # the stop's on_chunk
                if pending is not None and streaming:
                    yield from settle(*pending)  # the last chunk's count
        finally:
            self._rng_settle(stopped_at)  # unseen (rng="exact"): reads the index
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
    measured; with ``on_chunk`` the largest divisor of ``chunk_every`` up to it),
    ``chunk_every`` / ``on_chunk`` (streaming), ``watch`` (callables that must be called
    from Python: teacher forcing), ``check_state`` (tensors a masked step must not change,
    checked once before the unrolled capture), ``warmup_runs``, ``strict`` (raise instead
    of falling back), ``device``, ``workload`` / ``report`` (the workload whose timed runs
    count the loop's steps, read outside the clock, and the counters they are reported as;
    default ``("steps",)``), ``rng`` (a step that draws random numbers: ``"exact"`` leaves
    the generator where the plain loop does, a WHILE ``run()`` waiting for the loop's end;
    ``"reserve"`` at ``max_steps`` steps' draws, no wait)."""
    return DeviceLoop(step, cond, max_steps, **options)


# ------------------------------------------------------------------ doctor


def probe() -> tuple[bool | None, str]:
    """``doctor``: a loop of three steps on the device, run twice through a WHILE graph with
    a chunk IF node (and once through :meth:`DeviceLoop.chunks`), then a ``torch.compile``
    step (:func:`_probe_compiled`) and a step that draws random numbers (:func:`_probe_rng`)
    in a WHILE body: ``(True, detail)`` when every count is right, ``(None, why)`` where
    conditional nodes are unavailable (loops use the unrolled graphs there), ``(False,
    why)`` when a graph is wrong."""
    if (why := conditional_support()) is not None:
        return None, f"{why}; device_loop uses K-step unrolled CUDA graphs"
    ok, detail = _probe_while()
    if not ok:
        return ok, detail
    compiled, what = _probe_compiled()
    drawn, how = _probe_rng()
    return compiled is not False and drawn is not False, f"{detail}; {what}; {how}"


def _probe_while() -> tuple[bool, str]:
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


def _probe_compiled() -> tuple[bool | None, str]:
    """A ``torch.compile`` step (the default mode: Inductor's kernels, no CUDA graphs of its
    own) captured into a WHILE body: compiled in the warm-up host run, then two WHILE runs
    of three steps that read ``index`` on the device. ``(None, why)``: it does not compile
    or is not captured here (``device_loop`` falls back for such a step), ``(False, why)``:
    it ran wrong."""
    device = torch.device("cuda", torch.cuda.current_device())
    with torch.inference_mode(False):
        acc = torch.zeros(2, dtype=torch.int64, device=device)
        scale = torch.tensor([1, 2], device=device)

    def step(index: torch.Tensor, active: torch.Tensor) -> None:
        acc.copy_(acc + (index + 1) * scale)  # 1, 2, 3 (and twice that) at steps 1, 2, 3

    loop = DeviceLoop(
        torch.compile(step),
        lambda: acc[0] >= 6,  # stops after 3 steps
        max_steps=8,
        mode="while",
        strict=True,
        device=device,
    )
    start = time.perf_counter()
    stage = "compiled"
    try:
        for _ in range(3):  # a host run (it compiles), then the WHILE graph twice
            acc.zero_()
            loop.run()
            stage = "captured into a WHILE body"
    except Exception as exc:
        return None, (
            f"a torch.compile step is not {stage} here ({type(exc).__name__}: {exc}"[:300]
            + "): device_loop falls back for such a step"
        )
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    got = (acc.tolist(), int(loop.index), loop.mode, loop.stats["launches"])
    want = ([6, 12], 3, "while", 2)
    if got != want:
        return False, (
            "a torch.compile step ran wrong in a WHILE body: (sums, index, mode, launches) = "
            f"{got}, want {want}"
        )
    return True, (
        "a torch.compile step (no CUDA graphs of its own) ran 3 steps twice in a WHILE body "
        f"({seconds:.1f} s incl. compilation)"
    )


def _probe_rng() -> tuple[bool | None, str]:
    """A step that draws (``torch.rand``, ``torch.multinomial``) in a WHILE body: two runs of
    three steps against the plain loop's numbers and where it leaves the generator.
    ``(None, why)``: not captured here (``device_loop`` falls back for such a step; the
    unrolled blocks draw the plain loop's numbers too), ``(False, why)``: other numbers."""
    device = torch.device("cuda", torch.cuda.current_device())
    gen = _cuda.generator(device.index)
    with torch.inference_mode(False):
        out = torch.zeros(3, 8, device=device)

    def step(index: torch.Tensor, active: torch.Tensor | None) -> None:
        noise = torch.rand(1, 8, device=device)
        out.index_copy_(0, index.view(1), noise + torch.multinomial(noise[0], 1))

    saved = torch.cuda.get_rng_state(device)
    try:
        torch.cuda.manual_seed(5)
        for i in range(3):
            step(torch.tensor(i, device=device), None)
        want, offset = out.clone(), gen.get_offset()
        loop = DeviceLoop(step, None, 3, mode="while", warmup_runs=0, strict=True, device=device)
        got = []
        for _ in range(2):
            torch.cuda.manual_seed(5)
            out.zero_()
            loop.run()
            got.append((bool(out.equal(want)), gen.get_offset()))
    except Exception as exc:
        return None, (
            "a step that draws random numbers is not captured into a WHILE body here "
            f"({type(exc).__name__}: {exc}"[:300]
            + "): device_loop falls back for such a step"
        )
    finally:
        torch.cuda.set_rng_state(saved, device)
    if got != [(True, offset)] * 2:
        return False, (
            "a step that draws ran wrong in a WHILE body: (the plain loop's numbers, the "
            f"generator's offset) = {got}, want {[(True, offset)] * 2}"
        )
    per_step = loop.stats.get("rng_offsets_per_step")
    return True, (
        "a step that draws (torch.rand, torch.multinomial) ran in a WHILE body with the "
        f"plain loop's numbers ({per_step} Philox offsets a step)"
    )
