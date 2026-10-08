"""``kernel-agent improve --dry-run``: a simulated model, agents and GPU worker.

No GPU and no Claude. The simulated agents write the files a real run has
(candidates, snapshots, ``NOTES.md`` with open ideas, transforms, a research
``plan.md``, a research dossier ``research.md``) and record every evaluation
through the same code as the evaluation tools (``record_candidate`` with an
``idea_id``, ``record_e2e_result``), so the ledger, the budget advice, the research
trigger, the charts, the dashboard and the report are exercised end to end, with
workers per target too
(``--seeds-per-target``: a worker session writes to its own directory and follows
the evaluation budget of its ``# Worker`` section). An island session (``--islands``,
issue #189) builds on its own lineage in its direction's backend, whose ceiling is the
target's own per backend (:attr:`SimTarget.ceilings`; the planned first backend's is the
target's ceiling, so one session per target behaves as before), and on an inspiration of
its digest that is at least :data:`ADOPT` faster (a migration). The simulated engineer
retries an idea once after a failed attempt and starts from the plan's
directions; the research agent writes its plan from the ledger (the outcomes do
not depend on either).
The simulated worker runs the integration's end-to-end measurements, the
re-profile of a round or of the native arm's best run (with the run's ceilings table,
if it has one, scaled to the optimised model) and the capture of new targets.

The model is a Qwen3-0.6B decode workload (1532 ms baseline, launch bound).
Each kernel target has a hidden ceiling that it approaches in noisy,
diminishing steps, with failures (build errors, wrong results, timeouts) and
regressions on the way, so targets plateau. Some targets report a
speed-of-light estimate (``pct_of_sol``), others do not; one target only shows
up in the re-profile of round 2. The systems agent finds a static cache and a
CUDA graph, then plateaus; the graph is incompatible with the MLP kernel, which
the measured integration catches.

Every draw is seeded by (seed, arm, evaluation index), so a dry run is
reproducible and a restarted one continues like the original. Time is
simulated as well (``ledger.clock``, :class:`SimBudget`), so the charts show
hours of work and ``--max-hours`` stops the loop in simulated hours.

With ``--agents N`` (N > 1, ``coordinator.py``) the simulation runs in virtual time
(:class:`VirtualClock`, docs/MULTIAGENT.md §3.9): the event loop's clock is simulated and
jumps to the next timer when nothing can run, the simulated sessions are concurrent
coroutines that ``await`` their think time, and every evaluation, A/B step, capture and
re-profile holds the GPU through the real GPU job queue (``gpuqueue.holding``) for its
simulated seconds. Sessions then overlap as real ones would, and the run is still
reproducible. :attr:`World.limit` simulates an account-wide usage limit (the rate gate),
and :attr:`World.writes` records which session wrote which file (ownership).

With a board (``board.py``: ``--agents N``, ``--board on``) the simulated engineers post
now and then through the ``post_note`` checks (why a kept result wins, a trap after an idea
failed twice, an insight when a result pays well), and every simulated session reads what
its evaluation results carry (:attr:`World.seen`), as a real session's results do.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import concurrent.futures
import contextlib
import dataclasses
import functools
import math
import random
import re
import statistics
import threading
import time
from collections.abc import Callable, Coroutine, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from claude_agent_sdk import AssistantMessage, TextBlock

from kernel_agent import (
    board,
    gpuqueue,
    interrupt,
    ledger,
    pivot,
    program,
    sessions,
    truth,
    workers,
)
from kernel_agent.agent import auth, runner
from kernel_agent.agent.runner import AgentResult
from kernel_agent.agent.tools import SessionBinding, record_candidate, record_e2e_result, snapshot
from kernel_agent.budget import Budget
from kernel_agent.config import OptimizeConfig
from kernel_agent.kernels.roofline import sol_signal
from kernel_agent.native import engine as native_engine
from kernel_agent.research import PLAN_FILE
from kernel_agent.scheduler import class_shares, snapshot_record, systems_rows
from kernel_agent.workspace import RunDir, read_json, read_jsonl, write_json

if TYPE_CHECKING:
    from kernel_agent.orchestrator import Orchestrator

BASELINE_MS = 1532.4
HOOK_OVERHEAD = 2086.0 / BASELINE_MS  # profiled (hooked) time / real time
GPU_BUSY = 0.453
KERNEL_EFFICIENCY = 0.9  # share of a module's saving that shows up end to end
FIRST_MESSAGE_S = 12.0  # a simulated session's first streamed message (virtual time)
# virtual time: the share of a session's work that is its own runs (Bash: compiling,
# checks), not the model; docs/MULTIAGENT-DATA.md measured 46 % for kernel sessions
OWN_RUNS = 0.45
EVALUATION_TOOLS = {"eval": "evaluate_candidate", "e2e": "evaluate_e2e"}
# islands (#189): the ceiling of a backend a target lists no ceiling for, × the planned one's
OTHER_BACKEND = 0.85
ADOPT = 1.02  # a simulated island builds on an inspiration at least this much faster
_HEAD = re.compile(r"^\* Your island's best: `history/([^`]+)`", re.M)
_SNAPSHOT = re.compile(r"^\* `history/([^`]+)`: ", re.M)
_INSPIRATIONS = re.compile(r"^## Inspirations[^\n]*\n(.*?)(?=^#|\Z)", re.M | re.S)
_DIRECTION = re.compile(r"^\* Your direction: .*\(backends: ([a-z0-9_]+)", re.M)

HEADERS = {
    "triton": "import torch\nimport triton\nimport triton.language as tl\n",
    "cuda": "import torch\nfrom torch.utils.cpp_extension import load_inline\n",
    "cute": "import cutlass\nimport cutlass.cute as cute\nimport torch\n",
}

# (class, inclusive ms, instances, calls per run, leaf)
CLASSES: list[tuple[str, float, int, int, bool]] = [
    ("Qwen3ForCausalLM", 2086.0, 1, 128, False),
    ("Qwen3Model", 2010.0, 1, 128, False),
    ("Qwen3DecoderLayer", 1931.0, 28, 3584, False),
    ("Linear", 905.0, 197, 25_216, True),
    ("Qwen3Attention", 856.0, 28, 3584, False),
    ("Qwen3MLP", 588.0, 28, 3584, False),
    ("Qwen3RMSNorm", 321.0, 113, 14_464, True),
    ("SiLU", 54.0, 28, 3584, True),
    ("Qwen3RotaryEmbedding", 42.0, 1, 128, False),
    ("Embedding", 6.0, 1, 128, True),
]
CONTAINERS = ("Qwen3ForCausalLM", "Qwen3Model", "Qwen3DecoderLayer")


@dataclass(frozen=True)
class SimTarget:
    id: str
    cls: str
    ceiling: float  # module speedup the target approaches
    p_fail: float  # chance that an evaluation fails
    sol_at: float | None  # module speedup at 100 % of the speed of light (None: no estimate)
    backends: tuple[str, ...]
    hypotheses: tuple[str, ...]
    round: int = 1  # the round whose plan has it
    # the precision its research session proposes in a near-lossless run (pivot.py)
    pivot: str | None = None
    # an island's lineage in another backend approaches its own ceiling (islands, #189):
    # (backend, ceiling); the planned first backend's is ``ceiling``, an unlisted one
    # OTHER_BACKEND × it (:meth:`ceiling_of`)
    ceilings: tuple[tuple[str, float], ...] = ()

    def ceiling_of(self, backend: str) -> float:
        """The ceiling an island lineage in ``backend`` approaches."""
        found = dict(self.ceilings).get(backend)
        if found is not None:
            return found
        if not self.backends or backend == self.backends[0]:
            return self.ceiling
        return round(self.ceiling * OTHER_BACKEND, 3)


TARGETS = (
    SimTarget(
        "attn",
        "Qwen3Attention",
        2.3,
        0.25,
        2.65,
        ("triton", "cuda"),
        (
            "fuse q/k RMSNorm and RoPE into one kernel before SDPA",
            "write K/V straight into the cache slot",
            "split-K flash-decode for the 1-token decode case",
            "cp.async double-buffered KV tiles",
            "pre-pack q/k/v weights in build(): one GEMM instead of three",
            "fold the o_proj epilogue into the attention kernel",
            "fp32 accumulation in registers, no shared-memory reduction",
            "persistent kernel looping over heads",
            "separate static-shape prefill and decode kernels",
            "bf16x8 vector loads for K and V",
        ),
        ceilings=(("cuda", 2.6), ("cute", 2.0)),
    ),
    SimTarget(
        "mlp",
        "Qwen3MLP",
        1.8,
        0.2,
        1.95,
        ("cuda", "triton"),
        (
            "merge gate/up into one pre-packed GEMM with a fused SiLU*mul",
            "GEMV for M=1 decode: a warp per 8 rows, weights streamed once",
            "split-K for the down_proj GEMV with an fp32 scratch reduction",
            "128-bit loads and an L2 prefetch hint on the weights",
            "persistent CTAs, one per SM",
            "cuBLAS for prefill, the custom GEMV only for M<=4",
            "unroll the K loop by 4",
            "overlap the gate/up GEMV with the SiLU",
        ),
        pivot="fp8_weights",  # M=1 decode GEMVs: bound by streaming the bf16 weights
        ceilings=(("triton", 1.55), ("cute", 1.6)),
    ),
    SimTarget(
        "rmsnorm",
        "Qwen3RMSNorm",
        1.9,
        0.3,
        None,
        ("cuda", "cute"),
        (
            "warp-per-row RMSNorm, bf16x8 loads, fp32 accumulation",
            "two rows per warp",
            "skip .contiguous() and reuse the output buffer",
            "one launch for q_norm and k_norm",
            "__launch_bounds__(256), no shared memory",
            "CuTe DSL with the TVM-FFI calling convention",
        ),
        ceilings=(("cute", 2.1), ("triton", 1.7)),
    ),
    SimTarget(
        "rope",
        "Qwen3RotaryEmbedding",
        2.3,
        0.2,
        None,
        ("triton", "cuda"),
        (
            "precompute the cos/sin table once, gather by position",
            "load_inline gather kernel, one thread per pair",
            "vectorised 2×bf16 gather",
            "return views of the table instead of copies",
        ),
        round=2,
        ceilings=(("cuda", 2.0), ("cute", 1.9)),
    ),
)

# (file stem, hypothesis, family, e2e speedup alone or a failure status). Transforms of
# one family do not stack (a static cache is part of the CUDA-graph transforms).
SYSTEM_IDEAS: tuple[tuple[str, str, str, float | str], ...] = (
    ("static_cache", "static KV cache, no per-step reallocation", "graph", 1.10),
    ("cuda_graph_decode", "static cache + CUDA graph for the decode step", "graph", 1.52),
    ("sdpa_flash", "force the flash SDPA backend for prefill", "sdpa", 1.004),
    ("no_item_sync", "drop the per-step .item() sync in the stop check", "sync", "patch_error"),
    ("cuda_graph_v2", "CUDA graph + pinned sampling buffers, no host sync", "graph", 1.58),
    ("merged_qkv", "merge q/k/v projections into one Linear", "fusion", 1.02),
)
SYSTEM_TWEAKS = (
    "share one CUDA-graph memory pool across prompt buckets",
    "torch.compile the sampler",
    "pre-allocate the logits buffer",
    "prefill in two chunks to overlap with graph capture",
    "pin the KV cache layout to [layer, head, pos, dim]",
)
FAMILIES = {stem: family for stem, _, family, _ in SYSTEM_IDEAS}
INCOMPATIBLE = {("graph", "mlp"): "greedy tokens diverge at step 41"}


def _rng(seed: int, *key: Any) -> random.Random:
    return random.Random("|".join(map(str, (seed, *key))))


def sim_target(spec: dict[str, Any]) -> SimTarget:
    """The simulated behaviour of a target (made up from its id when it is not in TARGETS).
    A precision pivot (``pivot_of``) of a target gets 1.6x its ceiling (FP8 weights)."""
    for sim in TARGETS:
        if sim.cls == spec.get("module_class"):
            if spec.get("pivot_of"):
                faster = None if sim.sol_at is None else round(sim.sol_at * 1.9, 3)
                ceilings = tuple((b, round(c * 1.6, 3)) for b, c in sim.ceilings)
                return dataclasses.replace(
                    sim,
                    ceiling=round(sim.ceiling * 1.6, 3),
                    sol_at=faster,
                    pivot=None,
                    ceilings=ceilings,
                )
            return sim
    rng = _rng(0, "target", spec.get("id"))
    return SimTarget(
        str(spec.get("id")),
        str(spec.get("module_class")),
        rng.uniform(1.4, 2.4),
        rng.uniform(0.15, 0.35),
        None,
        tuple(spec.get("backends") or ("triton",)),
        ("fuse the elementwise ops into one kernel", "vectorised loads", "tune the block size"),
    )


# ------------------------------------------------------------------ clock and budget


class SimClock:
    """Simulated wall clock (seconds since the epoch)."""

    def __init__(self, start: float) -> None:
        self.t0 = self.t = start

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> float:
        self.t += seconds
        return self.t


class VirtualClock(SimClock):
    """Simulated time of an asyncio event loop (``--dry-run --agents N``, §3.9).

    While :meth:`driving` the running loop, the loop's ``time()`` is :meth:`time` and, when
    nothing can run, the loop jumps to its next timer instead of waiting: ``asyncio.sleep``,
    session timeouts, the rate gate and the GPU queue's waits (``gpuqueue.clock``) all take
    simulated seconds. Work the loop hands to threads (``asyncio.to_thread``: the
    integration, captures, re-profiles) runs while the loop is idle, one thread at a time (a
    baton), and the clock stands still while one runs, so a simulation is deterministic. A
    thread's simulated GPU job is a coroutine of the loop (:meth:`run_in_loop`) that the
    thread waits for without the baton. :meth:`now` (``ledger.clock``) starts at a whole
    second, so every run rounds the same way."""

    def __init__(self, start: float) -> None:
        super().__init__(float(math.floor(start)))
        self.elapsed = 0.0  # simulated seconds since the start
        self.loop: asyncio.AbstractEventLoop | None = None
        self._thread: int | None = None  # the loop's
        self._cond = threading.Condition()
        self._queue: collections.deque[object] = collections.deque()  # threads waiting to run
        self._holder: object | None = None  # the thread that runs now (None: the loop)
        self._closed = False
        self._tasks: set[asyncio.Task[None]] = set()

    def now(self) -> float:
        return self.t0 + self.elapsed

    def time(self) -> float:
        """The loop's clock (and the GPU queue's, the session deadlines')."""
        return self.elapsed

    def advance(self, seconds: float) -> float:
        """From a worker thread: wait ``seconds`` of simulated time."""
        self.run_in_loop(asyncio.sleep(seconds))
        return self.now()

    @contextmanager
    def driving(self) -> Iterator[VirtualClock]:
        """Drive the running event loop with this clock until the block ends (its timers
        keep their delays)."""
        loop = asyncio.get_running_loop()
        selector = loop._selector  # type: ignore[attr-defined]
        real_select, executor = selector.select, loop._default_executor  # type: ignore[attr-defined]
        shift = self.elapsed - loop.time()
        for handle in loop._scheduled:  # type: ignore[attr-defined]
            handle._when += shift
        self.loop, self._thread, self._closed = loop, threading.get_ident(), False
        setattr(loop, "time", self.time)  # noqa: B010 - an instance attribute shadows it
        setattr(selector, "select", functools.partial(self._select, real_select))  # noqa: B010
        loop.set_default_executor(_BatonExecutor(self))
        try:
            yield self
        finally:
            delattr(loop, "time")
            delattr(selector, "select")
            loop._default_executor = executor  # type: ignore[attr-defined]
            shift = loop.time() - self.elapsed
            for handle in loop._scheduled:  # type: ignore[attr-defined]
                handle._when += shift
            with self._cond:
                self._closed = True
                self._cond.notify_all()
            self.loop = None

    def _select(self, real: Callable[[float | None], list[Any]], timeout: float | None) -> Any:
        """The loop's ``select``: events first; when the loop is idle, a waiting thread runs
        alone, else simulated time jumps to the next timer."""
        if timeout == 0:
            return real(0)
        assert self.loop is not None
        while True:
            if events := real(0):  # the threads' results, signals
                return events
            with self._cond:
                if self._queue:
                    token = self._queue.popleft()
                    self._holder = token
                    self._cond.notify_all()
                    while self._holder is token and not self._closed:
                        self._cond.wait()
                    continue
            scheduled = self.loop._scheduled  # type: ignore[attr-defined]
            if scheduled:
                self.elapsed = max(self.elapsed, scheduled[0]._when)
                return []
            return real(1.0)  # nothing simulated waits: only an outside event (a signal)

    def _enqueue(self, token: object) -> None:
        with self._cond:
            self._queue.append(token)

    def _acquire(self, token: object) -> None:
        """A thread waits for the baton (``Interrupted`` once the loop stopped driving)."""
        with self._cond:
            while self._holder is not token:
                if self._closed:
                    raise interrupt.Interrupted
                self._cond.wait(0.5)

    def _release(self) -> None:
        with self._cond:
            self._holder = None
            self._cond.notify_all()

    def run_in_loop[T](self, coro: Coroutine[Any, Any, T]) -> T:
        """Run ``coro`` on the loop from a worker thread that holds the baton, and wait for
        it there without the baton (the loop and the other threads go on meanwhile)."""
        loop = self.loop
        if loop is None or threading.get_ident() == self._thread:
            coro.close()
            raise RuntimeError("VirtualClock.run_in_loop: only from a worker thread")
        done: concurrent.futures.Future[T] = concurrent.futures.Future()
        token = object()

        async def body() -> None:
            try:
                result = await coro
            except BaseException as exc:
                self._enqueue(token)  # before the thread wakes: the loop must wait for it
                done.set_exception(exc)
                if not isinstance(exc, Exception):
                    raise
            else:
                self._enqueue(token)
                done.set_result(result)

        def start() -> None:
            task = loop.create_task(body())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        loop.call_soon_threadsafe(start)
        self._release()
        try:
            while True:
                try:
                    return done.result(timeout=0.5)
                except concurrent.futures.TimeoutError:
                    if self._closed:
                        raise interrupt.Interrupted from None
        finally:
            self._acquire(token)


class _BatonExecutor(concurrent.futures.ThreadPoolExecutor):
    """The default executor of a loop a :class:`VirtualClock` drives: every call runs in its
    own thread once it holds the clock's baton (queued in the order of the calls)."""

    def __init__(self, clock: VirtualClock) -> None:
        super().__init__(max_workers=1)
        self.clock = clock

    def submit[T](
        self, fn: Callable[..., T], /, *args: Any, **kwargs: Any
    ) -> concurrent.futures.Future[T]:
        future: concurrent.futures.Future[T] = concurrent.futures.Future()
        token = object()
        self.clock._enqueue(token)  # now, in the loop's thread: before the loop can idle

        def body() -> None:
            try:
                self.clock._acquire(token)
            except BaseException as exc:
                future.set_exception(exc)
                return
            try:
                if future.set_running_or_notify_cancel():
                    try:
                        result = fn(*args, **kwargs)
                    except BaseException as exc:
                        future.set_exception(exc)
                    else:
                        future.set_result(result)
            finally:
                self.clock._release()  # after its result is on its way to the loop

        threading.Thread(target=body, name="dry-run-thread", daemon=True).start()
        return future


@dataclass
class _Lineage:
    """What a simulated island session builds on (``--islands``, #189): the snapshot, its
    speedup and its backend. It starts from its digest: its island's best (``## Your
    island``), or the fastest inspiration (``## Inspirations``) at least :data:`ADOPT` faster
    (a migration used), else the reference in its direction's first backend; then each
    faster result of its own."""

    parent: str | None
    level: float
    backend: str

    @classmethod
    def of(cls, run: RunDir, target_id: str, brief: str) -> _Lineage:
        rows = ledger.rows(run)
        found = {
            r["snapshot"]: r
            for r in rows
            if r["target"] == target_id and r["correct"] and r["speedup"]
        }
        direction = _DIRECTION.search(brief)
        out = cls(None, 1.0, direction[1] if direction else "")
        if (head := _HEAD.search(brief)) and head[1] in found:
            out = cls._from(found[head[1]])
        section = _INSPIRATIONS.search(brief)
        offered = _SNAPSHOT.findall(section[1]) if section else []
        better = [found[s] for s in offered if s in found]
        top = max(better, key=lambda r: float(r["speedup"]), default=None)
        if top is not None and float(top["speedup"]) > out.level * ADOPT:
            out = cls._from(top)
        return out

    @classmethod
    def _from(cls, row: dict[str, Any]) -> _Lineage:
        return cls(f"history/{row['snapshot']}", float(row["speedup"]), str(row["backend"]))

    def saw(self, row: dict[str, Any]) -> None:
        """Its own evaluation ``row``: a faster correct one is what it builds on next."""
        if row["correct"] and row["speedup"] and float(row["speedup"]) > self.level:
            self.parent, self.level = f"history/{row['snapshot']}", float(row["speedup"])


class _Limited(Exception):
    """A simulated session reached the simulated usage limit (:attr:`World.limit`)."""


@dataclass
class SimBudget(Budget):
    """A :class:`Budget` whose time is the simulated clock's (and in virtual time, whose
    session deadlines are too)."""

    clock: SimClock | None = None

    @classmethod
    def of(cls, budget: Budget, clock: SimClock) -> SimBudget:
        fields = {f.name: getattr(budget, f.name) for f in dataclasses.fields(Budget)}
        return cls(**fields, clock=clock)

    def elapsed_s(self) -> float:
        assert self.clock is not None
        return self.clock.now() - self.clock.t0


class SimToolchain:
    gpu = None
    backends = {b: True for b in ("cuda", "triton", "cute", "tilelang", "nvrtc")}
    env: dict[str, str] = {}

    def summary(self) -> str:
        return "GPU: simulated (kernel-agent improve --dry-run)"


# ------------------------------------------------------------------ the run


def _repo_id(ref: str) -> str:
    ref = re.sub(r"^https?://huggingface\.co/", "", ref.strip()).strip("/")
    return ref.split("@")[0] or "Qwen/Qwen3-0.6B"


def _profile(scale: dict[str, float] | None = None, busy: float = GPU_BUSY) -> dict[str, Any]:
    """Module profile; ``scale`` maps a class to a factor on its time (0: not in the model)."""
    classes: list[dict[str, Any]] = []
    for cls, ms, instances, calls, leaf in CLASSES:
        factor = 1.0 if scale is None else scale.get(cls, scale.get("*", 1.0))
        if factor <= 0:
            continue
        classes.append(
            {
                "root": "model",
                "cls": cls,
                "module_path": f"transformers.models.qwen3.modeling_qwen3.{cls}",
                "source_file": "/site-packages/transformers/models/qwen3/modeling_qwen3.py",
                "instances": instances,
                "calls": calls,
                "inclusive_ms": round(ms * factor, 3),
                "self_ms": round(ms * factor * 0.2, 3),
                "is_leaf": leaf,
                "example_qualname": f"model.{cls}",
                "signatures": [],
            }
        )
    return {
        "hooked_wall_ms": round(classes[0]["inclusive_ms"] * 1.05, 1),
        "classes": classes,
        "kernel_view": {"gpu_busy_fraction": round(busy, 3)},
    }


def _summary(profile: dict[str, Any], ms: float) -> str:
    rows = "\n".join(
        f"| {c['cls']} | {c['instances']} | {c['calls']} | {c['inclusive_ms']:.1f} |"
        for c in profile["classes"]
    )
    busy = profile["kernel_view"]["gpu_busy_fraction"]
    return (
        f"# Profile summary (simulated)\n\nwall {ms:.1f} ms per run, GPU busy {busy:.0%}\n\n"
        f"| class | instances | calls | inclusive ms |\n|---|---|---|---|\n{rows}\n"
    )


def _target_spec(sim: SimTarget) -> dict[str, Any]:
    return {
        "id": sim.id,
        "module_class": sim.cls,
        "qualname": None,
        "why": f"{sim.cls} is hot in decode",
        "approach": sim.hypotheses[0],
        "backends": list(sim.backends),
    }


def _capture_info(sim: SimTarget) -> dict[str, Any]:
    return {
        "qualname": f"model.layers.0.{sim.id}",
        "cases": [
            {"signature": "a0[1, 1, 1024]:bfloat16", "count": 127},
            {"signature": "a0[1, 512, 1024]:bfloat16", "count": 1},
        ],
    }


def _write_target(run: RunDir, spec: dict[str, Any]) -> None:
    target_dir = run.target(spec["id"])
    (target_dir / "candidates").mkdir(parents=True, exist_ok=True)
    write_json(target_dir / "spec.json", spec)
    (target_dir / "reference_source.py").write_text(f"# {spec['module_class']} (simulated)\n")
    (target_dir / "NOTES.md").touch()


def create_run(cfg: OptimizeConfig, seed: int = 0) -> RunDir:
    """A simulated run with analyze, plan and capture done (no GPU, no Claude, no network)."""
    repo = _repo_id(cfg.model_ref)
    run = RunDir.create(cfg.runs_dir, repo)
    t0 = time.time()
    profile = _profile()
    plan: dict[str, Any] = {
        "analysis": "Decode is launch bound (45 % GPU busy): fuse the small ops of attention, "
        "MLP and norms; a static cache + CUDA graph for the decode step.",
        "targets": [_target_spec(sim) for sim in TARGETS if sim.round == 1],
        "transforms": [
            {"id": "cuda_graph_decode", "idea": "CUDA graph for decode", "why": "launch bound"}
        ],
    }
    write_json(
        run.run_json,
        {
            "card": {
                "repo_id": repo,
                "revision": "main",
                "modality": "llm",
                "architectures": ["Qwen3ForCausalLM"],
                "params": 596_049_920,
                "size_gb": 1.4,
            },
            "workload": {"repo_id": repo, "modality": "llm", "dtype": cfg.dtype, "options": {}},
            "config": cfg.to_dict(),
            "phases": {p: {"done": True} for p in ("analyze", "plan", "capture")},
            "created": ledger.stamp(t0),
            "dry_run": {"seed": seed},
            "truth": truth.new_section(),
        },
    )
    write_json(run.toolchain_json, {"gpu": {"name": "simulated", "arch": "sm_120"}})
    write_json(
        run.baseline_json,
        {
            "workload": "llm: prompt 512 tokens, 128 new tokens, batch 1 (simulated)",
            "median_ms": BASELINE_MS,
            "times_ms": [1529.8, 1532.4, 1536.1],
            "deterministic": True,
        },
    )
    truth.of(run).seal_baseline(BASELINE_MS)
    write_json(run.profile_dir / "profile.json", profile)
    (run.profile_dir / "summary.md").write_text(_summary(profile, BASELINE_MS))
    write_json(run.plan_json, plan)
    for spec in plan["targets"]:
        _write_target(run, {**spec, "capture": _capture_info(sim_target(spec))})
    run.transforms_dir.mkdir(parents=True, exist_ok=True)
    program.install(run)
    for phase, a, b in (("analyze", 0.0, 4.5), ("plan", 4.5, 7.0), ("capture", 7.0, 9.0)):
        ledger.event(run, "phase_start", when=t0 + a * 60, phase=phase)
        ledger.event(run, "phase_done", when=t0 + b * 60, phase=phase)
    return run


# ------------------------------------------------------------------ the world


class World:
    """Simulated agents and worker for one run; :meth:`installed` plugs them into ``orch``.
    ``virtual``: concurrent sessions in virtual time (``--agents N``, :class:`VirtualClock`;
    the caller also enters :meth:`driving` inside the event loop)."""

    def __init__(
        self,
        orch: Orchestrator,
        hook: Callable[[str, int], None] | None = None,
        *,
        virtual: bool = False,
    ) -> None:
        self.orch = orch
        self.run = orch.run
        self.seed = int((self.run.load().get("dry_run") or {}).get("seed", 0))
        last = [float(e["ts"]) for e in ledger.events(self.run) if "ts" in e]
        start = max(last, default=time.time()) + 30
        self.virtual = virtual
        self.clock: SimClock = VirtualClock(start) if virtual else SimClock(start)
        self.hook = hook  # called after every simulated evaluation: (agent, evals so far)
        self.sessions: list[dict[str, Any]] = []
        profile = read_json(self.run.profile_dir / "profile.json", {}) or {}
        self.shares = class_shares(profile)
        # virtual time: a usage limit of the account, (seconds after the start, seconds until
        # it resets); every session that thinks inside it stops at it (the rate gate)
        self.limit: tuple[float, float] | None = None
        # virtual time: what each session wrote (label, file), its span and write policy
        self.writes: list[tuple[str, Path]] = []
        self.spans: dict[str, tuple[float, float]] = {}
        self.policies: dict[str, tuple[list[Path], list[Path]]] = {}
        # with a board (board.py): the entries each session's evaluation results carried
        self.seen: dict[str, list[int]] = {}

    @contextmanager
    def installed(self) -> Iterator[World]:
        orch = self.orch
        saved = (orch.agent_runner, orch.worker, orch.budget, orch.tc, ledger.clock, orch.clock)
        saved_queue = gpuqueue.clock
        orch.agent_runner, orch.worker = self.run_agent, self.worker
        orch.budget = SimBudget.of(orch.budget, self.clock)
        orch.tc = SimToolchain()  # type: ignore[assignment]
        ledger.clock = self.clock.now
        if isinstance(self.clock, VirtualClock):  # every clock of the run is the simulated one
            orch.clock = self.clock.now  # the usage-limit waits
            orch.budget.monotonic = self.clock.time  # the session deadlines
            gpuqueue.clock = self.clock.time  # the GPU queue's waits and holds
        try:
            yield self
        finally:
            orch.agent_runner, orch.worker, orch.budget, orch.tc, ledger.clock, orch.clock = saved
            gpuqueue.clock = saved_queue

    @contextmanager
    def driving(self) -> Iterator[World]:
        """Inside the event loop: drive it in virtual time (:class:`VirtualClock`) with a GPU
        job queue of its own (no job of an earlier run in this process ahead of it)."""
        assert isinstance(self.clock, VirtualClock), "World(virtual=True) drives a loop"
        with gpuqueue._gates_lock:
            saved = gpuqueue._gates.get("gpu")
            gpuqueue._gates["gpu"] = gpuqueue.Gate()
        try:
            with self.clock.driving():
                yield self
        finally:
            with gpuqueue._gates_lock:
                if saved is None:
                    gpuqueue._gates.pop("gpu", None)
                else:
                    gpuqueue._gates["gpu"] = saved

    def ref_ms(self, cls: str | None) -> float:
        share, _ = self.shares.get(str(cls), (0.0, 1))
        return share * BASELINE_MS

    @property
    def gated(self) -> bool:
        """The run's quality mode has the perceptual gate (near-lossless, relaxed): its
        ``e2e`` records carry ``metrics.perceptual``, as a real run's do."""
        from kernel_agent.kernels.compare import allows_reduced

        return allows_reduced(self.orch.cfg.quality)

    # -------------------------------------------------------- time

    async def _think(self, seconds: float) -> None:
        """A session works for ``seconds`` (virtual time: other sessions run meanwhile; a
        usage limit reached meanwhile stops it, :attr:`limit`)."""
        if not self.virtual:
            self.clock.advance(seconds)
            return
        own = seconds * OWN_RUNS if sessions.current() is not None else 0.0
        await asyncio.sleep(seconds - own)  # the model (sessions.py: thinking)
        if own:
            with sessions.tool("Bash"):  # its own runs
                await asyncio.sleep(own)
        if self.limit is not None:
            at, lasts = self.limit
            if at <= self.clock.now() - self.clock.t0 < at + lasts:
                raise _Limited

    async def _read(self, seconds: float) -> None:
        """A session reads its digest and files; in virtual time its first message streams
        (``runner.heard``: the coordinator's staggered starts) after :data:`FIRST_MESSAGE_S`."""
        if self.virtual:
            await asyncio.sleep(min(FIRST_MESSAGE_S, seconds))
            runner.heard(AssistantMessage(content=[TextBlock("reading")], model="dry-run"))
            seconds = max(seconds - FIRST_MESSAGE_S, 0.0)
        await self._think(seconds)

    async def _evaluate(self, kind: str, seconds: float, target: str | None = None) -> float | None:
        """The GPU job of a simulated evaluation (``eval``, ``e2e``): in virtual time it goes
        through the GPU queue as the session's job and holds the GPU for ``seconds`` (its
        ``eval_s``); returns its wait (``queue_s``). Without virtual time it takes no time.
        Either way it is the session's evaluation tool call (its states, ``sessions.py``)."""
        with sessions.tool(EVALUATION_TOOLS[kind]):
            if not self.virtual:
                return None
            job = gpuqueue.Job.of(self.run, kind, target)
            await self._held(job, seconds)
            return job.queue_s

    @staticmethod
    async def _held(job: gpuqueue.Job, seconds: float) -> None:
        async with gpuqueue.holding(job):
            await asyncio.sleep(seconds)

    def _hold(self, seconds: float) -> None:
        """The GPU time of a simulated worker job (an A/B step, a capture, a re-profile; from
        a worker thread): in virtual time the job ``Orchestrator._worker`` tagged holds the
        GPU through the queue for ``seconds``; else the clock advances."""
        if not isinstance(self.clock, VirtualClock):
            self.clock.advance(seconds)
            return
        job = gpuqueue.current() or gpuqueue.Job.of(self.run, "capture")
        self.clock.run_in_loop(self._held(job, seconds))

    def _write(self, bound: SessionBinding | None, path: Path, text: str) -> None:
        """A session writes ``path`` (recorded in virtual time: ownership checks)."""
        path.write_text(text)
        self._wrote(bound, path)

    def _wrote(self, bound: SessionBinding | None, path: Path) -> None:
        if self.virtual and bound is not None:
            self.writes.append((bound.label, path))

    # -------------------------------------------------------- agents

    async def run_agent(
        self,
        name: str,
        *,
        prompt: str,
        system_append: str,
        cwd: Path,
        result: AgentResult | None = None,
        writable: list[Path] | None = None,
        cfg: OptimizeConfig | None = None,
        resume: str | None = None,
        roots: list[Path] | None = None,
        excluded: list[Path] | None = None,
        **_: Any,
    ) -> AgentResult:
        result = result or AgentResult(name=name)
        result.is_error, result.usage_limit = False, None  # a resumed one: as runner.run_agent
        self.sessions.append({"name": name, "prompt": prompt, "system": system_append})
        sessions.thinking()  # its first turn (sessions.py; a real one's after its init message)
        result.session_id = resume or f"dry-{name}-{len(self.sessions)}"
        start = self.clock.now()
        rng = _rng(self.seed, "session", name, len(ledger.rows(self.run)))
        # what the session's tools are bound to (Orchestrator._agent): its evaluation budget
        # and the label its rows carry
        bound = self.orch.bindings.get(name)
        self.sessions[-1] |= {"label": bound.label if bound else name, "at": start}
        if self.virtual and bound is not None and roots is not None:
            self.policies[bound.label] = (list(roots), list(excluded or []))
        evals = 0
        try:
            await self._read(rng.uniform(40, 90))  # reading the digest and the files
            if name == "planner":
                result.structured = await self._plan(Path(cwd))
                result.cost_usd = 0.4
            elif name == "systems":
                evals = await self._systems(bound)
                result.cost_usd = 0.3 + 0.5 * evals * rng.uniform(0.8, 1.2)
            elif name == "native":
                evals = await self._native(bound)
                result.cost_usd = 0.8 + 1.2 * evals * rng.uniform(0.8, 1.2)
            elif name.startswith("kernel-"):
                target_id, worker = workers.parse_agent(name)
                brief = f"{system_append}\n{prompt}"  # the digest: in the first message (#181)
                evals = await self._kernel(target_id, brief, worker=worker, bound=bound)
                result.cost_usd = 0.25 + 0.35 * evals * rng.uniform(0.8, 1.2)
            elif name.startswith("research-"):
                self._research(name.removeprefix("research-"), writable or [], bound)
                await self._think(rng.uniform(240, 480))
                result.cost_usd = 0.6 * rng.uniform(0.8, 1.2)
            elif name.startswith("dossier-"):  # no web in a dry run: a dossier from the spec
                self._dossier(name.removeprefix("dossier-"), writable or [], bound)
                result.cost_usd = 0.15 * rng.uniform(0.8, 1.2)
        except _Limited:  # stopped at the simulated usage limit: resumed after it resets
            assert self.limit is not None
            evals = self.orch.budget.evals.get(name, 0)
            result.cost_usd = 0.25 + 0.35 * evals
            resets = self.clock.t0 + sum(self.limit)
            message = "You've hit your limit (simulated)"
            result.usage_limit = auth.UsageLimit(message, resets_at=resets, kind="five_hour")
            result.is_error = True
        if self.virtual and cfg is not None and cfg.budget_usd_per_agent is not None:
            result.cost_usd = min(result.cost_usd, cfg.budget_usd_per_agent)  # max_budget_usd
        result.tool_calls = {"evaluate": evals} if evals else {}
        result.turns = 4 + 5 * evals
        result.seconds = self.clock.now() - start
        result.text = f"simulated session: {evals} evaluations"
        if self.virtual and bound is not None:
            first = self.spans.get(bound.label, (start, start))[0]
            self.spans[bound.label] = (first, self.clock.now())
        await asyncio.sleep(0)
        return result

    def _advice(
        self,
        agent: str,
        results: Path,
        evals: int | None,
        rng: random.Random,
        pct_of_sol: float | None = None,
        label: str | None = None,
    ) -> bool:
        """Whether the simulated agent stops after this evaluation (it follows the advice)."""
        budget = self.orch.budget
        ok_key = "passed" if agent in ("systems", "native") else "correct"
        feedback = budget.feedback(agent, results, evals, ok_key=ok_key, pct_of_sol=pct_of_sol)
        advice = feedback["advice"]
        if advice == "stop" or budget.exhausted(label):
            return True
        return advice == "consider_stopping" and rng.random() < 0.5

    async def _kernel(
        self,
        target_id: str,
        system: str = "",
        worker: int | None = None,
        bound: SessionBinding | None = None,
    ) -> int:
        from kernel_agent.kernels.compare import tier_of

        target_dir = self.run.target(target_id)
        home = workers.directory(self.run, target_id, worker) if worker else target_dir
        agent = workers.agent_name(target_id, worker) if worker else f"kernel-{target_id}"
        bound = bound or SessionBinding(evaluations=self.orch.cfg.evaluations_per_target)
        spec = read_json(target_dir / "spec.json", {}) or {}
        sim = sim_target(spec)
        ref_ms = self.ref_ms(sim.cls) or 10.0
        instances = self.shares.get(sim.cls, (0.0, 1))[1]
        plan = _plan_directions(system)  # a research plan in the digest comes first
        # an island session (--islands, #189) builds on its own lineage, in its own backend
        lineage = (
            _Lineage.of(self.run, target_id, system) if workers.ISLAND_MARK in system else None
        )
        used = 0
        while True:
            rows = [r for r in ledger.rows(self.run) if r["target"] == target_id]
            k = len(rows)
            rng = _rng(self.seed, "kernel", target_id, k)
            kept = [r for r in rows if r["status"] == ledger.KEEP]
            best = ledger.best_kept(rows)
            if lineage is None:  # the target's best, the backends in turn
                backend = sim.backends[(k // 5 + (worker or 1) - 1) % len(sim.backends)]
                parent = f"history/{kept[-1]['snapshot']}" if kept else None
                level, model = best, sim
            else:  # its lineage's, toward its backend's ceiling
                backend, parent, level = lineage.backend, lineage.parent, lineage.level
                model = dataclasses.replace(sim, ceiling=sim.ceiling_of(backend))
            idea, hypothesis = _next_idea(sim, rows, plan)
            expected = level * _rng(self.seed, "expect", target_id, k).uniform(1.05, 1.4)
            outcome = self._kernel_outcome(model, level, rng)
            await self._think(rng.uniform(150, 330))
            src = home / "candidates" / f"{backend}_v{k + 1}.py"
            header = HEADERS.get(backend, "import torch\n")
            self._write(
                bound,
                src,
                f'{header}\n"""{hypothesis}"""\n\n\ndef build(reference):\n    return reference\n',
            )
            snap = snapshot(self.run, src, target_id)
            result = kernel_result(outcome, ref_ms, instances, sim)
            result["tolerance_tier"] = tier_of(spec.get("capture"))  # as the evaluator's
            eval_s = round(rng.uniform(25, 70), 1)
            queue_s = await self._evaluate("eval", eval_s, target_id)
            _, row = record_candidate(
                self.run,
                target_id,
                src,
                snap,
                result,
                hypothesis=hypothesis,
                parent=parent,
                eval_s=eval_s,
                when=self.clock.now(),
                idea=idea,
                expected_speedup=round(expected, 2),
                worker=worker,
                queue_s=queue_s,
                session=bound.label or None,
            )
            if lineage is not None:
                lineage.saw(row)
            used += 1
            _note(home / "NOTES.md", row, sim.hypotheses[(k + 1) % len(sim.hypotheses) :])
            self._wrote(bound, home / "NOTES.md")
            self._post(bound, target_id, sim, row, rows, backend)
            self._news(bound)
            if self.hook:
                self.hook(agent, used)
            results = self.run.results_file(target_id)
            stop = self._advice(
                agent, results, bound.evaluations, rng, sol_signal(result), bound.label
            )
            if stop:
                return used

    @staticmethod
    def _kernel_outcome(sim: SimTarget, best: float, rng: random.Random) -> float | str:
        if rng.random() < sim.p_fail:
            return rng.choice(("build_error", "incorrect", "incorrect", "runtime_error", "timeout"))
        gap = max(sim.ceiling - best, 0.0)
        progress = gap / max(sim.ceiling - 1.0, 1e-6)  # 1 at the start, 0 at the ceiling
        if rng.random() < 0.3 + 0.5 * progress:
            return best + gap * rng.uniform(0.2, 0.6)
        return best * rng.uniform(0.84, 1.006)

    async def _systems(self, bound: SessionBinding | None = None) -> int:
        bound = bound or SessionBinding(evaluations=self.orch.cfg.transform_evaluations)
        used = 0
        while True:
            rows = systems_rows(ledger.rows(self.run))
            k = len(rows)
            rng = _rng(self.seed, "systems", k)
            best = max(
                [1.0, *(r["speedup"] for r in rows if r["correct"] and r["speedup"])],
            )
            if k < len(SYSTEM_IDEAS):
                stem, hypothesis, _, outcome = SYSTEM_IDEAS[k]
            else:
                j = k - len(SYSTEM_IDEAS)
                stem, hypothesis = f"tweak_{j + 1}", SYSTEM_TWEAKS[j % len(SYSTEM_TWEAKS)]
                outcome = best * rng.uniform(0.93, 1.008)
            await self._think(rng.uniform(240, 420))
            src = self.run.transforms_dir / f"{stem}.py"
            self._write(bound, src, f'"""{hypothesis}"""\n\n\ndef apply(workload):\n    pass\n')
            snap = snapshot(self.run, src)
            if isinstance(outcome, str):
                result: dict[str, Any] = {
                    "status": outcome,
                    "passed": False,
                    "error": f"{outcome}: simulated failure",
                }
            else:
                ms = BASELINE_MS / (outcome * rng.uniform(0.996, 1.004))
                result = _e2e_result(ms, gated=self.gated)
            eval_s = round(rng.uniform(60, 110), 1)
            queue_s = await self._evaluate("e2e", eval_s)
            _, row = record_e2e_result(
                self.run,
                result,
                [snap],
                [],
                hypothesis=hypothesis,
                eval_s=eval_s,
                when=self.clock.now(),
                queue_s=queue_s,
                session=bound.label or None,
            )
            used += 1
            _note(self.run.transforms_dir / "NOTES.md", row, SYSTEM_TWEAKS[k % 3 :][:2])
            self._wrote(bound, self.run.transforms_dir / "NOTES.md")
            self._news(bound)
            if self.hook:
                self.hook("systems", used)
            results = self.run.results_file()
            if self._advice("systems", results, bound.evaluations, rng, label=bound.label):
                return used

    async def _native(self, bound: SessionBinding | None = None) -> int:
        """A simulated systems-native session: per evaluation a multi-file project of the
        current stage (once the plan is done: its focus; ``loop`` without either),
        snapshotted as its bundle like a real one, measured end to end around the
        module-level bar."""
        bound = bound or SessionBinding(evaluations=self.orch.cfg.transform_evaluations)
        used = 0
        while True:
            rows = ledger.rows(self.run)
            k = sum(native_engine.is_native(r) for r in rows)
            rng = _rng(self.seed, "native", k)
            level = native_engine.bar(rows)
            stage = native_engine.status(self.run, rows).stage
            name = stage.id if stage is not None else "loop"
            await self._think(rng.uniform(600, 1200))  # writing and compiling a project
            project = native_engine.native_dir(self.run) / name
            (project / "csrc").mkdir(parents=True, exist_ok=True)
            self._write(
                bound,
                project / "kernel_project.toml",
                f'[project]\nname = "{name}"\nkind = "transform"\n\n'
                '[build]\nsources = ["csrc/*.cu"]\n',
            )
            hypothesis = f"native engine of {name}, attempt {k + 1}"
            self._write(
                bound,
                project / "candidate.py",
                f'"""{hypothesis}"""\n\n\ndef apply(workload):\n    pass\n',
            )
            self._write(bound, project / "csrc" / "engine.cu", f"// {hypothesis}\n")
            snap = snapshot(self.run, project)
            outcome = level * rng.uniform(0.96, 1.1)
            result = _e2e_result(BASELINE_MS / outcome, gated=self.gated)
            eval_s = round(rng.uniform(200, 400), 1)
            queue_s = await self._evaluate("e2e", eval_s)
            record_e2e_result(
                self.run,
                result,
                [snap],
                [],
                hypothesis=hypothesis,
                eval_s=eval_s,
                when=self.clock.now(),
                queue_s=queue_s,
                session=bound.label or None,
            )
            used += 1
            self._news(bound)
            if self.hook:
                self.hook("native", used)
            results = self.run.results_file()
            if self._advice("native", results, bound.evaluations, rng, label=bound.label):
                return used

    # -------------------------------------------------------- the board (board.py)

    def _post(
        self,
        bound: SessionBinding | None,
        target_id: str,
        sim: SimTarget,
        row: dict[str, Any],
        rows: list[dict[str, Any]],
        backend: str,
    ) -> None:
        """A simulated engineer's note after an evaluation, through the ``post_note`` checks
        (its own draws, so the run's outcomes do not depend on the board): why a kept result
        wins (half of them), an insight that holds for every target when one pays well, a
        trap once an idea failed twice in a row. Refused notes (the session's notes spent,
        a duplicate) are dropped, as an agent would."""
        found = board.active(self.run)
        if found is None or bound is None or not bound.label:
            return
        draw = _rng(self.seed, "board", target_id, row["exp"]).random()
        idea, args = str(row.get("idea") or ""), None
        if row["status"] == ledger.KEEP and draw < 0.5:
            args = {
                "kind": board.WINNER,
                "target": target_id,
                "text": f"`{idea}` wins at {row['speedup']:.2f}x: {row['hypothesis']}",
                "refs": [f"exp:{row['exp']}"],
            }
        elif row["status"] == ledger.KEEP and draw < 0.7 and (row["speedup"] or 0) > 1.4:
            args = {
                "kind": board.INSIGHT,
                "text": f"{backend} pays on {sim.cls} ({row['speedup']:.2f}x): {row['hypothesis']}",
                "refs": [f"exp:{row['exp']}"],
            }
        elif row["status"] in ledger.FAILURES:
            same = [r for r in rows if idea and r.get("idea") == idea]
            if same and same[-1]["status"] in ledger.FAILURES:
                args = {
                    "kind": board.TRAP,
                    "target": target_id,
                    "text": f"`{idea}` failed twice in a row ({same[-1]['status']}, "
                    f"{row['status']}): {row['hypothesis']}",
                    "refs": [f"exp:{same[-1]['exp']}", f"exp:{row['exp']}"],
                }
        if args is not None:
            with contextlib.suppress(board.Refused):
                found.note(self.run, board.Reader.of_session(self.run, bound), args)

    def _news(self, bound: SessionBinding | None) -> None:
        """What a simulated session's evaluation result carries from the board (the entries
        new for it since its cursor, as the tools' piggyback), recorded in :attr:`seen`."""
        found = board.active(self.run)
        if found is None or bound is None or not bound.label:
            return
        if new := found.news(board.Reader.of_session(self.run, bound)):
            self.seen.setdefault(bound.label, []).extend(int(e["id"]) for e in new)

    def _research(
        self, target_id: str, writable: list[Path], bound: SessionBinding | None = None
    ) -> None:
        """A research session: ``plan.md`` from the target's ledger rows, if it may write it."""
        plan = self.run.target(target_id) / PLAN_FILE
        if plan.resolve() not in {p.resolve() for p in writable}:
            return
        spec = read_json(self.run.target(target_id) / "spec.json", {}) or {}
        rows = [r for r in ledger.rows(self.run) if r["target"] == target_id]
        sim = sim_target(spec)
        self._write(bound, plan, _plan_md(target_id, sim, rows))
        proposal = pivot.proposal_path(self.run, target_id)
        allowed = proposal.resolve() in {p.resolve() for p in writable}
        if allowed and sim.pivot and not spec.get("precision") and not proposal.exists():
            best = max((r["speedup"] or 0.0 for r in rows if r["correct"]), default=1.0)
            why = (
                f"{sim.cls} decode is M=1 GEMVs that stream the bf16 weights: memory bound, "
                f"best {best:.2f}x of a {sim.ceiling:.1f}x ceiling at bf16; FP8 halves the bytes"
            )
            write_json(proposal, {"precision": sim.pivot, "precision_why": why})
            self._wrote(bound, proposal)

    def _dossier(
        self, target_id: str, writable: list[Path], bound: SessionBinding | None = None
    ) -> None:
        """A dossier session: ``research.md`` from the target's spec, if it may write it."""
        path = self.run.target(target_id) / "research.md"
        if path.resolve() not in {p.resolve() for p in writable}:
            return
        sim = sim_target(read_json(self.run.target(target_id) / "spec.json", {}) or {})
        self._write(
            bound,
            path,
            f"# Dossier: `{target_id}`\n\n## Findings\n"
            f"* {sim.cls}: fuse the module into one kernel (simulated, no lookup)\n\n"
            f"## Ideas\n1. `{target_id}_fused`: one launch per call\n",
        )

    async def _plan(self, cwd: Path) -> dict[str, Any]:
        profile = read_json(cwd / "profile" / "profile.json", {}) or {}
        present = {c["cls"] for c in profile.get("classes", [])}
        taken = {
            (read_json(self.run.target(t) / "spec.json", {}) or {}).get("module_class")
            for t in self.run.target_ids()
        }
        targets = [
            _target_spec(sim) for sim in TARGETS if sim.cls in present and sim.cls not in taken
        ]
        await self._think(120)
        return {
            "analysis": "Re-profile: attention, MLP and norms are fast now; RoPE is next.",
            "targets": targets,
            "transforms": [
                {
                    "id": "graph_prefill",
                    "idea": "capture prefill in a CUDA graph per prompt bucket",
                    "why": "prefill is launch bound now",
                }
            ],
        }

    # -------------------------------------------------------- worker

    def worker(self, run: RunDir, command: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--kernel", action="append", default=[])
        parser.add_argument("--transform", action="append", default=[])
        parser.add_argument("--out-dir", type=Path)
        parser.add_argument("--target")
        parser.add_argument("--b-kernel", action="append", default=[])
        parser.add_argument("--b-transform", action="append", default=[])
        parser.add_argument("--rounds", type=int, default=8)
        ns, _ = parser.parse_known_args(list(args))
        if command == "e2e":
            return self._e2e(ns.kernel, ns.transform)
        if command == "e2e_ab":
            a, b = (ns.kernel, ns.transform), (ns.b_kernel, ns.b_transform)
            return self._e2e_ab(a, b, ns.rounds)
        if command == "analyze" and ns.out_dir:
            return self._reprofile(ns.out_dir, ns.kernel, ns.transform)
        if command == "capture":
            return self._capture(ns.target)
        return {"status": "error", "error": f"dry run: worker {command} {' '.join(args)}"}

    def _model(
        self, kernels: list[str], transforms: list[str]
    ) -> tuple[float, dict[str, float], dict[str, float], str | None]:
        """End-to-end ms, kernel speedups, transform factor per family, failure reason."""
        speedups: dict[str, float] = {}
        for item in kernels:
            target_id, _, path = item.partition("=")
            rec = snapshot_record(self.run, target_id, path) or {}
            speedups[target_id] = float(rec.get("speedup") or 1.0)
        families: dict[str, float] = {}
        records = read_jsonl(self.run.results_file())
        for path in transforms:
            name = Path(path).name
            rec = next((r for r in records if Path(r["transforms"][0]).name == name), {})
            family = FAMILIES.get(ledger.snapshot_stem(path), "graph")
            families[family] = max(families.get(family, 1.0), float(rec.get("speedup") or 1.0))
        reason = next(
            (
                why
                for (fam, tid), why in INCOMPATIBLE.items()
                if fam in families and tid in speedups
            ),
            None,
        )
        ms = BASELINE_MS
        for target_id, s in speedups.items():
            spec = read_json(self.run.target(target_id) / "spec.json", {}) or {}
            ms -= self.ref_ms(spec.get("module_class")) * (1 - 1 / s) * KERNEL_EFFICIENCY
        if families:
            factors = sorted(families.values(), reverse=True)
            factor = factors[0]
            for f in factors[1:]:
                factor *= 1 + 0.15 * (f - 1)
            if speedups:  # the kernels already removed part of the launch overhead
                factor = 1 + (factor - 1) * 0.8
            ms /= factor
        return ms, speedups, families, reason

    def _e2e(self, kernels: list[str], transforms: list[str]) -> dict[str, Any]:
        ms, _, _, reason = self._model(kernels, transforms)
        key = "+".join(sorted(ledger.item_label(i) for i in [*kernels, *transforms]))
        rng = _rng(self.seed, "e2e", key, len(ledger.rows(self.run)))
        self._hold(rng.uniform(65, 95))
        result = _e2e_result(ms * rng.uniform(0.997, 1.003), gated=self.gated)
        if reason:
            result.update(passed=False, reason=reason, metrics={"token_match": 0.32})
        return result

    def _e2e_ab(
        self, a: tuple[list[str], list[str]], b: tuple[list[str], list[str]], rounds: int
    ) -> dict[str, Any]:
        """A paired A/B (``worker e2e_ab``): both states share each round's GPU drift."""
        a_ms = self._model(*a)[0]
        b_ms, _, _, reason = self._model(*b)
        key = "+".join(sorted(ledger.item_label(i) for i in [*b[0], *b[1]]))
        rng = _rng(self.seed, "e2e_ab", key, len(ledger.rows(self.run)))
        self._hold(rng.uniform(150, 210))
        times: tuple[list[float], list[float]] = ([], [])
        for _ in range(rounds):
            drift = rng.uniform(0.99, 1.01)
            times[0].append(round(a_ms * drift * rng.uniform(0.998, 1.002), 3))
            times[1].append(round(b_ms * drift * rng.uniform(0.998, 1.002), 3))
        result = _e2e_result(statistics.median(times[1]), gated=self.gated)
        result.update(times_ms=times[1], ab={"mode": "paired", "a_ms": times[0], "b_ms": times[1]})
        if reason:
            result.update(passed=False, reason=reason, metrics={"token_match": 0.32})
        return result

    def _reprofile(
        self, out_dir: Path, kernels: list[str], transforms: list[str]
    ) -> dict[str, Any]:
        ms, speedups, families, reason = self._model(kernels, transforms)
        if reason:
            return {"status": "error", "error": f"the optimised model fails: {reason}"}
        self._hold(300)
        replaced = {
            (read_json(self.run.target(t) / "spec.json", {}) or {}).get("module_class")
            for t in speedups
        }
        factor = max(families.values(), default=1.0)
        scale: dict[str, float] = {cls: 0.0 for cls in replaced if cls}
        scale["*"] = 1 / factor
        for cls in CONTAINERS:
            scale[cls] = ms / BASELINE_MS
        profile = _profile(scale, busy=min(0.95, GPU_BUSY * factor))
        out = RunDir(out_dir)
        baseline = {
            "workload": "llm: prompt 512 tokens, 128 new tokens, batch 1 (simulated, optimised)",
            "median_ms": round(ms, 3),
            "times_ms": [round(ms * 0.998, 3), round(ms, 3), round(ms * 1.003, 3)],
            "deterministic": True,
        }
        write_json(out.baseline_json, baseline)
        write_json(out.profile_dir / "profile.json", profile)
        (out.profile_dir / "summary.md").write_text(_summary(profile, ms))
        table = read_json(self.run.profile_dir / "ceilings.json", None)
        if isinstance(table, dict) and table.get("rows"):  # every row as fast as the run now
            write_json(out.profile_dir / "ceilings.json", _scaled(table, ms / BASELINE_MS))
        return baseline

    def _capture(self, target_id: str) -> dict[str, Any]:
        from kernel_agent.kernels.compare import EXACT_TIER, tier_for

        spec = read_json(self.run.target(target_id) / "spec.json", {}) or {}
        info = _capture_info(sim_target(spec))
        tier = tier_for(self.orch.cfg.quality, spec.get("precision"))
        if tier != EXACT_TIER:  # as worker capture: the tier and precision of the target
            info.update(tier=tier, precision=spec["precision"])
        _write_target(self.run, {**spec, "capture": info})
        self._hold(45)
        return info


def _scaled(table: dict[str, Any], factor: float) -> dict[str, Any]:
    """A ceilings table with every row's time × ``factor`` (the floors are the work's: kept)."""
    rows = [
        {**r, "now_ms": round(float(r["now_ms"]) * factor, 3)} if r.get("now_ms") else r
        for r in table.get("rows") or []
    ]
    return {**table, "rows": rows}


# ------------------------------------------------------------------ results


def kernel_result(
    outcome: float | str, ref_ms: float, instances: int, sim: SimTarget
) -> dict[str, Any]:
    """An ``evaluate_candidate`` result (``ref_ms``: the target's time per model run)."""
    if isinstance(outcome, str):
        result: dict[str, Any] = {"status": outcome, "correct": False}
        if outcome == "incorrect":
            result["cases"] = [
                {"signature": "a0[1, 1, 1024]:bfloat16", "ok": False, "max_abs_err": 0.31}
            ]
        else:
            result["error"] = f"{outcome}: simulated failure"
        return result
    s = outcome
    ref_w = ref_ms / max(instances, 1)  # per instance and run, over the captured cases
    cases = []
    for sig, count, frac, case_s in (
        ("a0[1, 1, 1024]:bfloat16", 127, 0.9, s),
        ("a0[1, 512, 1024]:bfloat16", 1, 0.1, s),
    ):
        ref = ref_w * frac / count
        cases.append(
            {
                "signature": sig,
                "calls_per_run": count,
                "ok": True,
                "max_abs_err": 0.0078,
                "ref_ms": round(ref, 6),
                "new_ms": round(ref / case_s, 6),
                "speedup": round(case_s, 3),
                "timing_spread": 0.006,
            }
        )
    result = {
        "status": "ok",
        "correct": True,
        "cases": cases,
        "speedup": round(s, 4),
        "est_saved_ms_per_run": round(ref_ms * (1 - 1 / s), 3),
        "ref_ms_weighted": round(ref_w, 5),
        "new_ms_weighted": round(ref_w / s, 5),
        "eval_seconds": 41.0,
    }
    if sim.sol_at:  # the evaluator's speed-of-light estimate (issue #9)
        result["pct_of_sol"] = round(100 * s / sim.sol_at, 1)
        result["bound"] = "memory"
    return result


def _e2e_result(ms: float, *, gated: bool = False) -> dict[str, Any]:
    """A passing simulated ``e2e`` record; ``gated``: of a run whose quality mode has the
    perceptual gate (near-lossless, relaxed), whose records carry ``metrics.perceptual``."""
    gate = {"perceptual": {"passed": True, "reason": "", "simulated": True}} if gated else {}
    return {
        "status": "ok",
        "passed": True,
        "reason": None,
        "metrics": {"token_match": 1.0, "logits_cosine": 0.9998, **gate},
        "median_ms": round(ms, 3),
        "times_ms": [round(ms * 0.997, 3), round(ms, 3), round(ms * 1.004, 3)],
        "baseline_ms": BASELINE_MS,
        "speedup": round(BASELINE_MS / ms, 4),
        "peak_mem_gb": 2.1,
        "patches": {},
    }


def _idea_of(hypothesis: str) -> str:
    """The simulated engineer's ``idea_id`` of a hypothesis: its first three words."""
    return ledger.idea_slug(" ".join(hypothesis.split()[:3]))


def _base(sim: SimTarget, idea: str, default: str) -> str:
    return next((h for h in sim.hypotheses if _idea_of(h) == idea), default)


def _plan_directions(system: str) -> tuple[int, list[tuple[str, str]]]:
    """``(exp, [(idea_id, hypothesis)])`` of the research plan in a slice digest: the
    ranked directions of a plan written after ``exp`` (as :func:`_plan_md` writes them)."""
    plan = system.split("## Research plan", 1)[1] if "## Research plan" in system else ""
    after = re.search(r"after exp (\d+)", plan)
    ranked = plan.split("## Ranked directions", 1)[-1].split("\n## ", 1)[0] if plan else ""
    found = re.findall(r"^\d+\. `([a-z0-9_-]+)`: (.+)$", ranked, re.M)
    return (int(after[1]) if after else 0), found


def _next_idea(
    sim: SimTarget, rows: list[dict[str, Any]], plan: tuple[int, list[tuple[str, str]]]
) -> tuple[str, str]:
    """``(idea_id, hypothesis)`` of the simulated engineer's next candidate.

    The research plan's directions not tried since it was written come first; then
    one fix of an idea whose attempt failed (a bug is not evidence against the
    idea); then the target's ideas in turn, later passes as variants."""
    after, directions = plan
    for idea, hypothesis in directions:
        if not any(r.get("idea") == idea and (r["exp"] or 0) > after for r in rows):
            return idea, hypothesis
    last = rows[-1] if rows else None
    if last and last["status"] in ledger.FAILURES and last.get("idea"):
        before = rows[-2] if len(rows) > 1 else None
        if before is None or before.get("idea") != last["idea"]:
            base = _base(sim, last["idea"], str(last["hypothesis"]))
            return last["idea"], f"fix of exp {last['exp']} ({last['status']}): {base}"
    k = len(rows)
    hypothesis = sim.hypotheses[k % len(sim.hypotheses)]
    if k >= len(sim.hypotheses):
        hypothesis += f" (variant {k // len(sim.hypotheses) + 1})"
    return _idea_of(hypothesis), hypothesis


def _plan_md(target_id: str, sim: SimTarget, rows: list[dict[str, Any]]) -> str:
    """The simulated research agent's ``plan.md``: the pathology checklist applied to the
    ledger, untried and buggy ideas ranked first, ideas measured slow on the do-not-try list."""
    stats = ledger.ideas(rows)
    tried = {s["idea"] for s in stats}
    keeps = [r for r in rows if r["status"] == ledger.KEEP]
    best = max(keeps, key=lambda r: r["speedup"] or 0.0, default=None)
    streak = len(rows) - (rows.index(keeps[-1]) + 1 if keeps else 0)
    recent = rows[-5:]
    failed = sum(r["status"] in ledger.FAILURES for r in recent)
    fresh = [(_idea_of(h), h) for h in sim.hypotheses if _idea_of(h) not in tried]
    buggy = [s for s in stats if s["verdict"] == "buggy"]
    slow = [s for s in stats if s["verdict"] == "slow"]
    repeated = [f"`{s['idea']}` ×{s['tries']}" for s in stats if s["tries"] >= 3]
    ranked = (
        fresh[:3]
        + [(s["idea"], _base(sim, s["idea"], s["last_hypothesis"])) for s in buggy][
            : max(0, 3 - len(fresh))
        ]
    )
    if not ranked:  # every idea tried and none buggy: the best design, further
        idea = (best or {}).get("idea") or _idea_of(sim.hypotheses[0])
        ranked = [(idea, f"{_base(sim, idea, idea)}, tuned for the dominant decode case")]
    slow = [s for s in slow if s["idea"] not in {idea for idea, _ in ranked}]
    checks = [
        f"repetition loop ({', '.join(repeated)})" if repeated else "",
        f"correctness wall ({failed} of the last {len(recent)} failed)" if failed >= 3 else "",
        f"missing fundamentals ({len(fresh)} planned ideas never tried)" if fresh else "",
    ]
    lines = [
        f"# Plan: `{target_id}` after exp {rows[-1]['exp'] if rows else 0} (simulated)",
        "",
        "## Diagnosis",
        f"{streak} evaluations without a new best; best "
        + (f"{best['speedup']:.3f}x (exp {best['exp']})" if best else "none correct")
        + ". Checklist: "
        + ("; ".join(c for c in checks if c) or "local minimum of one design")
        + ".",
        "",
        "## Strategy",
        "**pivot**: the ideas never tried, ahead of more variants of the current design."
        if fresh
        else "**targeted fixes**: fix the buggy ideas, then tune the best design.",
        "",
        "## Ranked directions",
        *(f"{i}. `{idea}`: {text}" for i, (idea, text) in enumerate(ranked, 1)),
        "",
        "## Retry (failed, not refuted)",
        *(
            f"* `{s['idea']}`: {', '.join(s['statuses'])} in exp " + ", ".join(map(str, s["exps"]))
            for s in buggy
        ),
        "",
        "## Do not try",
        *(
            f"* `{s['idea']}`: measured correct and not faster (exp "
            + ", ".join(map(str, s["exps"]))
            + f"; best {s['best']:.3f}x)"
            for s in slow
        ),
        "",
        "## Notes for the engineer",
        f"Build on `history/{best['snapshot']}`." if best else "Start from the reference.",
    ]
    return "\n".join(lines) + "\n"


def _note(path: Path, row: dict[str, Any], ideas: tuple[str, ...] | list[str]) -> None:
    """Append the evaluation to NOTES.md and rewrite its ``## Open ideas`` section."""
    text = path.read_text() if path.exists() else ""
    log = text.split("\n## Open ideas", 1)[0].rstrip()
    speedup = "" if row["speedup"] is None else f" {row['speedup']:.3f}x"
    log += f"\n- exp {row['exp']}: {row['hypothesis']} → {row['status']}{speedup}"
    todo = "\n".join(f"- {idea}" for idea in list(ideas)[:3]) or "- (none)"
    path.write_text(f"{log.strip()}\n\n## Open ideas\n{todo}\n")
