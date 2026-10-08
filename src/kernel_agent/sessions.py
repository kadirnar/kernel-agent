"""Per-agent observability (docs/MULTIAGENT.md §3.11, issue #184): what each agent session is
doing, where its time goes, and what the GPU does.

**States.** Every agent session (``Orchestrator._agent``) has a :class:`Tracker` on the run's
:class:`Observer`. A session is in one state at a time, the most specific of what it does:

* ``starting``: its Claude Code process starts (until its init message);
* ``thinking``: a model call of its own is in flight (no tool runs);
* ``helper``: a subagent it delegated to works (the Agent tool);
* ``tool``: one of its own tools runs (Bash, Read, Write, ...: its own runs);
* ``evaluating``: one of its evaluation tools runs off the GPU (build, checks, records);
* ``queued``: one of its GPU jobs waits for the GPU (detail: the job's class);
* ``on_gpu``: one of its GPU jobs holds the GPU (detail: the job's kind);
* ``limited``: it stopped at a usage limit and waits for the reset;
* ``idle``: its turn ended and it stays open (a background task), or between two runs.

Tool spans come from the PreToolUse / PostToolUse(Failure) hooks that ``runner.run_agent``
installs (:meth:`Tracker.hooks`); a tool's result in the message stream closes a span whose
hook never came (a denied call). The GPU states come from the GPU job queue's events
(``gpuqueue.listen``): a session's jobs carry its label.

**Time split.** Each state's time counts in one part (:data:`PART`): model, GPU held, waiting
for the GPU, evaluation off the GPU, own runs, idle. That is the split docs/MULTIAGENT-DATA.md
measured from transcripts, from now on measured in every run.

**Records** (rate-limited where they go to shared logs):

* ``sessions.jsonl``: every state change of every session (the first record has its agent,
  role and arm; the last, ``ended``, its status, split, evaluations and USD);
* ``events.jsonl``: ``session_state`` when a session starts and ends and, in between, at most
  one per :data:`EVENT_S` per session; ``gpu_job`` for a GPU job that waited at least
  :data:`GPU_EVENT_S` (``gpu_queue.jsonl`` has every job);
* ``improve.json`` → ``sessions`` (the running ones) and ``gpu`` (holders, queue by class,
  busy share and waits over the last hour), at most every :data:`SAVE_S` (:meth:`Observer.attach`);
* ``costs.json`` → ``time``: a session's split in seconds.

**Views**: :func:`agents` (``kernel-agent status``: the Agents table) and :func:`gpu_lines` (its
GPU block), :func:`lanes` (``kernel-agent watch``: swimlanes and a GPU strip), :func:`spans`
and :func:`gpu_holds` (``improve.png``), :func:`report_lines` (report.md "Concurrency").
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import math
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    HookJSONOutput,
    HookMatcher,
    ResultMessage,
    SystemMessage,
    ToolResultBlock,
    UserMessage,
)

from kernel_agent import gpuqueue, interrupt, ledger, roles
from kernel_agent.roles import MCP_PREFIX
from kernel_agent.workspace import RunDir, append_jsonl, read_json

STARTING = "starting"
THINKING = "thinking"
HELPER = "helper"
TOOL = "tool"
EVALUATING = "evaluating"
QUEUED = "queued"
ON_GPU = "on_gpu"
LIMITED = "limited"
IDLE = "idle"
ENDED = "ended"
#: the part of the time split each state's time counts in
PART = {
    STARTING: "idle",
    THINKING: "model",
    HELPER: "model",
    TOOL: "tools",
    EVALUATING: "eval",
    QUEUED: "queued",
    ON_GPU: "gpu",
    LIMITED: "idle",
    IDLE: "idle",
}
PARTS = ("model", "gpu", "queued", "eval", "tools", "idle")  # the order the views show
WORK = ("model", "eval", "tools")  # a session's work off the GPU (governor.py: Z)
TITLES = {
    "model": "model",
    "gpu": "GPU held",
    "queued": "waiting for the GPU",
    "eval": "evaluation off the GPU",
    "tools": "own runs",
    "idle": "idle",
}
#: kernel-agent's tools whose call is a GPU job of the session (its evaluations, checks)
GPU_TOOLS = frozenset(
    {
        "evaluate_candidate",
        "evaluate_candidates",
        "sweep_candidate",
        "evaluate_e2e",
        "evaluate_e2e_batch",
        "check_harness",
        "verify_rewrite",
        "run_on_gpu",
        "submit_evaluation",
        "evaluation_result",
    }
)
#: ... and those whose call is an evaluation (a ledger row; not with ``mode="quick"``)
EVAL_TOOLS = frozenset(
    {
        "evaluate_candidate",
        "evaluate_candidates",
        "sweep_candidate",
        "evaluate_e2e",
        "evaluate_e2e_batch",
        "submit_evaluation",
    }
)
#: the tool that waits for a submitted evaluation (--async-evals): while it runs, the GPU
#: states of the session's detached jobs are its own (before, the session works meanwhile)
COLLECT_TOOL = "evaluation_result"
HELPER_TOOLS = frozenset({"Agent", "Task"})  # a subagent's work: model time
FILE = "sessions.jsonl"  # in the run directory
EVENT_S = 600.0  # a session's session_state events in events.jsonl: at most one per this long
GPU_EVENT_S = 60.0  # a GPU job that waited this long is a gpu_job event in events.jsonl
SAVE_S = 2.0  # improve.json's sessions / gpu: at most this often
HOUR_S = 3600.0
LANE_COLUMNS = 1600  # watch: a session's states coarsened to this many steps of the span shown

_current: ContextVar[Tracker | None] = ContextVar("kernel_agent_session_tracker", default=None)


def current() -> Tracker | None:
    """The tracker of the agent session this context runs (None: not in one)."""
    return _current.get()


def arm_of(agent: str, target: str | None = None) -> str | None:
    """The arm (target, ``systems``, ``native``) a session of ``agent`` works on (None: none,
    e.g. the planner)."""
    if target:
        return target
    head, _, rest = agent.partition("-")
    if head in ("systems", "native") and not rest:
        return head
    if head == "kernel" and rest:
        return rest.rpartition("-w")[0] if rest.rpartition("-w")[2].isdigit() else rest
    return rest if head in ("research", "dossier", "refactor") and rest else None


def describe(state: str, detail: str = "") -> str:
    """A state as ``kernel-agent status`` says it."""
    if state == TOOL:
        return f"running {detail}" if detail else "running a tool"
    if state == EVALUATING:
        return f"{detail} (off the GPU)" if detail else "evaluating (off the GPU)"
    if state == QUEUED:
        return f"waiting for the GPU ({detail})" if detail else "waiting for the GPU"
    if state == ON_GPU:
        return f"on the GPU ({detail})" if detail else "on the GPU"
    if state == HELPER:
        return f"helper ({detail})" if detail and detail not in HELPER_TOOLS else "helper"
    return {LIMITED: "usage limit", ENDED: "ended"}.get(state, state)


@contextlib.contextmanager
def tool(name: str, tool_input: dict[str, Any] | None = None) -> Iterator[None]:
    """A tool call of the session this context runs, as its hooks would report it (the
    simulated sessions of a dry run, ``dryrun.py``); nothing outside a session."""
    tracker = current()
    if tracker is None:
        yield
        return
    key = tracker.next_id()
    tracker.tool_started(key, name, tool_input)
    try:
        yield
    finally:
        tracker.tool_ended(key)


def thinking() -> None:
    """The session this context runs works on its first turn (a simulated session's start)."""
    if (tracker := current()) is not None:
        tracker.turn(THINKING)


# ------------------------------------------------------------------ one session


class Tracker:
    """The state and time split of one agent session (:meth:`Observer.open`). Thread-safe: its
    hooks and messages arrive on the event loop's thread, its GPU jobs' events on the threads
    that run them."""

    def __init__(
        self,
        observer: Observer,
        label: str,
        *,
        agent: str,
        role: str,
        arm: str | None,
        reserved: float | None,
        now: float,
    ) -> None:
        self.observer = observer
        self.label, self.agent, self.role, self.arm = label, agent, role, arm
        self.reserved = reserved  # USD the coordinator reserved for it
        self.started = self.since = now
        self.state, self.detail = STARTING, ""
        self.split = dict.fromkeys(PARTS, 0.0)
        self.evaluations = 0
        self.usd: float | None = None
        self.status: str | None = None  # how it ended (None: open)
        self.event_at = now  # its last session_state event
        self.unsent = False  # it changed state since that event
        self._turn = STARTING  # the state outside its tools and GPU jobs
        self._tools: dict[str, tuple[str, str, bool]] = {}  # id -> (state, name, an evaluation)
        self._jobs: dict[Any, tuple[str, str]] = {}  # GPU job id -> (QUEUED / ON_GPU, detail)
        self._detached: set[Any] = set()  # of those, its submitted evaluations' (--async-evals)
        self._lock = threading.Lock()
        self._ids = itertools.count(1)

    # -------------------------------------------------------- inputs

    def next_id(self) -> str:
        return f"sim-{next(self._ids)}"

    def hooks(self, hooks: dict[Any, list[HookMatcher]]) -> None:
        """Add its PreToolUse / PostToolUse / PostToolUseFailure hooks (every tool) to
        ``hooks`` (``runner.run_agent``)."""

        async def pre(data: Any, tool_use_id: str | None, context: Any) -> HookJSONOutput:
            data = data if isinstance(data, dict) else {}
            key = str(tool_use_id or data.get("tool_use_id") or "")
            self.tool_started(key, str(data.get("tool_name") or ""), data.get("tool_input"))
            return {}

        async def post(data: Any, tool_use_id: str | None, context: Any) -> HookJSONOutput:
            data = data if isinstance(data, dict) else {}
            self.tool_ended(str(tool_use_id or data.get("tool_use_id") or ""))
            return {}

        for event, fn in (("PreToolUse", pre), ("PostToolUse", post), ("PostToolUseFailure", post)):
            hooks.setdefault(event, []).append(HookMatcher(matcher=None, hooks=[fn]))

    def see(self, message: Any) -> None:
        """A message of its stream (``runner.run_agent``): its turns, and the tool results
        that close a tool span whose hook never came (a call a guard denied)."""
        init = isinstance(message, SystemMessage) and message.subtype == "init"
        if init or (isinstance(message, AssistantMessage) and message.parent_tool_use_id is None):
            self.turn(THINKING)  # a model call of its own: its first, or the next of a turn
        elif isinstance(message, UserMessage) and isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    self.tool_ended(block.tool_use_id)
        elif isinstance(message, ResultMessage):
            self.turn(IDLE)  # the turn ended; a background task may give it another

    def turn(self, state: str) -> None:
        """Its state outside tools and GPU jobs: starting, thinking, idle or limited."""
        with self._lock:
            if self._turn == state:
                return
            self._turn = state
        self._update()

    def tool_started(self, key: str, name: str, tool_input: Any = None) -> None:
        short = name.removeprefix(MCP_PREFIX)
        state = EVALUATING if short in GPU_TOOLS else HELPER if short in HELPER_TOOLS else TOOL
        given = tool_input if isinstance(tool_input, dict) else {}
        counts = short in EVAL_TOOLS and given.get("mode") != "quick"  # a ledger row
        if state == HELPER and given.get("subagent_type"):
            short = str(given["subagent_type"])
        with self._lock:
            self._tools[key] = (state, short, counts)
        self._update()

    def tool_ended(self, key: str) -> None:
        with self._lock:
            entry = self._tools.pop(key, None)
            if entry is None:
                return
            if entry[2]:
                self.evaluations += 1
        self._update(counted=entry[2])

    def gpu(self, event: dict[str, Any]) -> None:
        """A ``gpu_job`` event of one of its jobs (``gpuqueue``): queued, start, done, withdrawn."""
        state, job = event.get("state"), event.get("id")
        with self._lock:
            if state == "queued":
                self._jobs[job] = (QUEUED, str(event.get("class") or ""))
            elif state == "start":
                self._jobs[job] = (ON_GPU, str(event.get("kind") or ""))
            elif self._jobs.pop(job, None) is None:
                return
            if event.get("detached"):  # a submitted evaluation: its own only when collected
                self._detached.add(job)
            if state not in ("queued", "start"):
                self._detached.discard(job)
        ts = event.get("ts")
        self._update(now=float(ts) if isinstance(ts, int | float) else None)

    # -------------------------------------------------------- state

    def _resolve(self) -> tuple[str, str]:
        """The most specific of what it does (under the lock). A submitted evaluation's GPU
        job (``detached``) is what the session does only while it waits for its result."""
        collecting = any(name == COLLECT_TOOL for _, name, _ in self._tools.values())
        for want in (ON_GPU, QUEUED):
            for job, (state, detail) in self._jobs.items():
                if state == want and (collecting or job not in self._detached):
                    return state, detail
        for want in (EVALUATING, TOOL, HELPER):
            for state, name, _ in self._tools.values():
                if state == want:
                    return state, name
        return self._turn, ""

    def _update(self, *, now: float | None = None, counted: bool = False) -> None:
        with self._lock:
            if self.status is not None:
                return
            now = max(self.observer.now() if now is None else now, self.since)
            state, detail = self._resolve()
            if (state, detail) == (self.state, self.detail) and not counted:
                return
            self.split[PART.get(self.state, "idle")] += now - self.since
            self.state, self.detail, self.since = state, detail, now
            record: dict[str, Any] = {"ts": round(now, 3), "label": self.label, "state": state}
            if detail:
                record["detail"] = detail
            if counted:
                record["evaluations"] = self.evaluations
        self.observer._changed(self, record)

    @contextlib.asynccontextmanager
    async def running(self, result: Any = None) -> AsyncIterator[Tracker]:
        """``async with`` it around one run of the session (``Orchestrator._agent``; again when
        it is resumed after a usage limit): ``runner.run_agent`` finds it (:func:`current`)
        and installs its hooks. After the run the session is idle (``limited`` while
        ``result`` stopped at a usage limit). A run that raised (not its timeout) ends it."""
        token = _current.set(self)
        self.turn(STARTING)
        failed: BaseException | None = None
        try:
            yield self
        except TimeoutError:  # its time limit: the session ends normally (Orchestrator._agent)
            raise
        except BaseException as exc:
            failed = exc
            raise
        finally:
            _current.reset(token)
            with self._lock:
                self._tools.clear()  # its process ended (its GPU jobs end with their events)
            if failed is not None:
                interrupted = not isinstance(failed, Exception) or interrupt.requested()
                self.close(status="interrupted" if interrupted else "failed")
            else:
                self.turn(LIMITED if getattr(result, "usage_limit", None) else IDLE)

    def close(self, result: Any = None, *, status: str | None = None) -> dict[str, float]:
        """End the session (once; ``result``: its ``AgentResult``, for its USD and how it
        ended); returns its time split (seconds per part)."""
        if status is None:
            timed_out, error = getattr(result, "timed_out", False), getattr(result, "is_error", 0)
            status = "timed_out" if timed_out else "error" if error else "done"
        with self._lock:
            if self.status is not None:
                return self.time()
            now = max(self.observer.now(), self.since)
            self.split[PART.get(self.state, "idle")] += now - self.since
            self.state, self.detail, self.since, self.status = ENDED, "", now, status
            self._tools.clear()
            self._jobs.clear()
            usd = getattr(result, "cost_usd", None)
            self.usd = float(usd) if isinstance(usd, int | float) else None
            record: dict[str, Any] = {
                "ts": round(now, 3),
                "label": self.label,
                "state": ENDED,
                "status": status,
                "seconds": round(now - self.started, 1),
                "split": self.time(),
                "evaluations": self.evaluations,
                **({"usd": round(self.usd, 4)} if self.usd is not None else {}),
            }
        self.observer._closed(self, record)
        return self.time()

    def time(self) -> dict[str, float]:
        """Its time split so far (seconds per part, up to its last state change)."""
        return {part: round(seconds, 1) for part, seconds in self.split.items()}

    def snapshot(self, now: float) -> dict[str, Any]:
        """Its entry in ``improve.json`` → ``sessions``."""
        with self._lock:
            split = dict(self.split)
            if self.status is None:
                split[PART.get(self.state, "idle")] += max(now - self.since, 0.0)
            return {
                "label": self.label,
                "agent": self.agent,
                "role": self.role,
                **({"arm": self.arm} if self.arm else {}),
                "state": self.state,
                **({"detail": self.detail} if self.detail else {}),
                "since": round(self.since, 3),
                "started": round(self.started, 3),
                "evaluations": self.evaluations,
                **({"reserved": round(self.reserved, 2)} if self.reserved else {}),
                "split": {part: round(seconds, 1) for part, seconds in split.items()},
            }


# ------------------------------------------------------------------ the run's sessions


class Observer:
    """The agent sessions and the GPU of one run, live (one per orchestrator): the trackers
    of the open sessions (:meth:`open`), the GPU job queue's events while sessions run or the
    improve loop is attached, and the records (module docstring). ``reserved``: the USD the
    coordinator reserved for a session label."""

    def __init__(
        self, run: RunDir, *, reserved: Callable[[str], float | None] | None = None
    ) -> None:
        self.run = run
        self.reserved = reserved
        self.save: Callable[[dict[str, Any]], None] | None = None  # improve.json (attach)
        self.trackers: dict[str, Tracker] = {}  # the open sessions by label
        self.jobs: dict[Any, dict[str, Any]] = {}  # GPU jobs that hold or wait: id -> event
        self.holds: deque[tuple[float, float]] = deque()  # (start, end) of the last hour's holds
        self.waits: deque[tuple[float, float]] = deque()  # (ts, wait_s) of the last hour's starts
        self.loop: asyncio.AbstractEventLoop | None = None
        self.since: float | None = None  # when it started following the GPU queue
        self._lock = threading.RLock()
        self._unlisten: Callable[[], None] | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._saved = -math.inf  # the loop's time of the last save
        self._dirty = False
        self._recovered = False
        self._worked = 0.0  # the work off the GPU of the ended sessions of GPU roles (WORK)

    def now(self) -> float:
        return ledger.clock()  # the run's clock (a dry run's simulated one)

    # -------------------------------------------------------- sessions

    def open(self, label: str, *, agent: str, role: str, arm: str | None = None) -> Tracker:
        """A new session ``label`` (of ``agent``, a session of ``role``): its tracker, recorded."""
        self._start()
        now = self.now()
        reserved = self.reserved(label) if self.reserved is not None else None
        tracker = Tracker(
            self,
            label,
            agent=agent,
            role=role,
            arm=arm_of(agent, arm),
            reserved=reserved or None,
            now=now,
        )
        with self._lock:
            self.trackers[label] = tracker
        where = {"arm": tracker.arm} if tracker.arm else {}
        self._write(
            {
                "ts": round(now, 3),
                "label": label,
                "state": STARTING,
                "agent": agent,
                "role": role,
                **where,
                **({"reserved": round(tracker.reserved, 2)} if tracker.reserved else {}),
            }
        )
        ledger.event(
            self.run,
            "session_state",
            when=now,
            label=label,
            agent=agent,
            role=role,
            state=STARTING,
            **where,
        )
        self._touch()
        return tracker

    def _changed(self, tracker: Tracker, record: dict[str, Any]) -> None:
        """A session changed state: its record, and its event once :data:`EVENT_S` passed."""
        self._write(record)
        now = float(record["ts"])
        with self._lock:
            due = _event_due(tracker, now)
            if due:
                tracker.event_at, tracker.unsent = now, False
            else:
                tracker.unsent = True
        if due:
            self._state_event(tracker, now)
        self._touch()

    def _state_event(self, tracker: Tracker, now: float) -> None:
        with tracker._lock:
            state, detail, since = tracker.state, tracker.detail, tracker.since
        if state == ENDED:
            return
        ledger.event(
            self.run,
            "session_state",
            when=now,
            label=tracker.label,
            state=state,
            **({"detail": detail} if detail else {}),
            since=round(since, 3),
        )

    def _closed(self, tracker: Tracker, record: dict[str, Any]) -> None:
        self._write(record)
        with self._lock:
            if self.trackers.get(tracker.label) is tracker:
                del self.trackers[tracker.label]
            if roles.get(tracker.role).needs_gpu:
                self._worked += sum(float(record["split"].get(part) or 0.0) for part in WORK)
        fields = {k: v for k, v in record.items() if k not in ("ts", "label", "state")}
        ledger.event(
            self.run, "session_state", when=record["ts"], label=tracker.label, state=ENDED, **fields
        )
        self._touch(force=True)
        self._stop_if_idle()

    def _write(self, record: dict[str, Any]) -> None:
        with contextlib.suppress(OSError):  # a record: it never fails the session
            append_jsonl(self.run.root / FILE, record)

    def work(self) -> float:
        """Seconds the sessions of the roles that need the GPU (``RoleSpec.needs_gpu``) have
        worked off the GPU so far (:data:`WORK`: model, evaluations' own steps, own runs), the
        ended ones and the open ones (``governor.py``: Z, the work per evaluation)."""
        now = self.now()
        with self._lock:
            trackers, worked = list(self.trackers.values()), self._worked
        for tracker in trackers:
            if roles.get(tracker.role).needs_gpu:
                split = tracker.snapshot(now)["split"]
                worked += sum(float(split.get(part) or 0.0) for part in WORK)
        return worked

    # -------------------------------------------------------- the improve loop

    def attach(self, save: Callable[[dict[str, Any]], None]) -> None:
        """From now on ``save`` (the improve loop's: ``improve.json``) gets the live
        :meth:`snapshot` within :data:`SAVE_S` of every change; the GPU queue is followed
        meanwhile (background jobs too)."""
        self.save = save
        self._start()
        self._touch(force=True)

    def detach(self) -> None:
        """The improve loop ended: sessions still open (Ctrl-C during a usage-limit wait) end
        as interrupted, the last snapshot is saved, and the GPU queue is no longer followed."""
        for tracker in list(self.trackers.values()):
            tracker.close(status="interrupted")
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self.save is not None and self._dirty:
            self._flush()
        self.save = None
        self._stop_if_idle()

    def snapshot(self) -> dict[str, Any]:
        """``improve.json``'s ``sessions`` (the open ones) and ``gpu`` (the queue)."""
        now = self.now()
        with self._lock:
            trackers, jobs = list(self.trackers.values()), list(self.jobs.values())
            self._prune(now)
            holds, waits = list(self.holds), sorted(w for _, w in self.waits)
            since = self.since if self.since is not None else now
        span = min(HOUR_S, max(now - since, 0.0))
        busy = sum(min(b, now) - max(a, now - HOUR_S) for a, b in holds if b > now - HOUR_S)
        held = [e for e in jobs if e.get("state") == "start"]
        busy += sum(max(now - max(float(e["ts"]), now - HOUR_S), 0.0) for e in held)
        queue: dict[str, dict[str, Any]] = {}
        for e in jobs:
            if e.get("state") == "queued":
                entry = queue.setdefault(str(e.get("class")), {"jobs": 0, "max_wait_s": 0.0})
                entry["jobs"] += 1
                waited = round(max(now - float(e["ts"]), 0.0), 1)
                entry["max_wait_s"] = max(entry["max_wait_s"], waited)
        gpu = {
            "holders": [_job(e, now) for e in held],
            "waiting": [_job(e, now) for e in jobs if e.get("state") == "queued"],
            "queue": queue,
            "busy_1h": round(min(busy / span, 1.0), 3) if span > 0 else None,
            "jobs_1h": len(waits),
            **(
                {"wait_p50_s": round(_quantile(waits, 0.5), 1)}
                | {"wait_p95_s": round(_quantile(waits, 0.95), 1)}
                if waits
                else {}
            ),
        }
        return {"sessions": [t.snapshot(now) for t in trackers], "gpu": gpu}

    # -------------------------------------------------------- the GPU queue

    def _start(self) -> None:
        """Before the first session or the improve loop: close what a process that did not end
        cleanly left open (:func:`recover`), know the event loop, follow the GPU queue."""
        if not self._recovered:
            self._recovered = True
            with contextlib.suppress(Exception):  # records only
                recover(self.run)
        with contextlib.suppress(RuntimeError):
            self.loop = asyncio.get_running_loop()
        with self._lock:
            if self._unlisten is None:
                self._unlisten = gpuqueue.listen(self._gpu)
                self.since = self.now()

    def _stop_if_idle(self) -> None:
        with self._lock:
            if self._unlisten is not None and self.save is None and not self.trackers:
                self._unlisten()
                self._unlisten = None

    def _gpu(self, event: dict[str, Any]) -> None:
        """A ``gpu_job`` event (``gpuqueue.listen``, in the thread that has it)."""
        state, job = event.get("state"), event.get("id")
        ts = event.get("ts")
        now = float(ts) if isinstance(ts, int | float) else self.now()
        waited = float(event.get("wait_s") or 0.0)
        with self._lock:
            if state in ("queued", "start"):
                self.jobs[job] = dict(event)
            else:
                self.jobs.pop(job, None)
            if state == "start":
                self.waits.append((now, waited))
            elif state == "done":
                self.holds.append((now - float(event.get("hold_s") or 0.0), now))
            self._prune(now)
            session = event.get("session")
            tracker = self.trackers.get(str(session)) if session else None
        if tracker is not None:
            tracker.gpu(event)
        if state == "start" and waited >= GPU_EVENT_S:  # contention: the events feed says it
            keep = ("id", "kind", "class", "session", "target", "wait_s")
            fields = {k: event[k] for k in keep if event.get(k) is not None}
            record = {"ts": round(now, 3), "time": ledger.stamp(now), "event": "gpu_job"}
            with contextlib.suppress(OSError):  # (ledger.event takes no field named kind)
                append_jsonl(self.run.events, {**record, "state": "waited", **fields})
        self._touch()

    def _prune(self, now: float) -> None:
        while self.holds and self.holds[0][1] < now - HOUR_S:
            self.holds.popleft()
        while self.waits and self.waits[0][0] < now - HOUR_S:
            self.waits.popleft()

    # -------------------------------------------------------- saving (rate-limited)

    def _touch(self, force: bool = False) -> None:
        """Something changed: save within :data:`SAVE_S` (at once with ``force``) and write
        the sessions' due events, on the event loop's thread."""
        self._dirty = True
        if self.save is None and not any(t.unsent for t in list(self.trackers.values())):
            return
        loop = self.loop
        if loop is None or loop.is_closed():
            self._arm(force)
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self._arm(force)
        else:
            with contextlib.suppress(RuntimeError):  # the loop closed meanwhile
                loop.call_soon_threadsafe(self._arm, force)

    def _next_event_in(self) -> float | None:
        """Seconds until a session's unsent state is due as an event (None: none is)."""
        now = self.now()
        due = [t.event_at + EVENT_S - now for t in list(self.trackers.values()) if t.unsent]
        return max(min(due), 0.0) if due else None

    def _arm(self, force: bool = False) -> None:
        loop = self.loop
        if loop is None or loop.is_closed():
            self._flush()
            return
        waits = []
        if self._dirty and self.save is not None:
            waits.append(0.0 if force else max(self._saved + SAVE_S - loop.time(), 0.0))
        if (due := self._next_event_in()) is not None:
            waits.append(due)
        if not waits:
            return
        wait = min(waits)
        if wait <= 0.0:
            self._flush()
            return
        when = loop.time() + wait
        if self._timer is not None:
            if self._timer.when() <= when:
                return  # one fires sooner
            self._timer.cancel()
        self._timer = loop.call_later(wait, self._fire)

    def _fire(self) -> None:
        self._timer = None
        self._flush()

    def _flush(self) -> None:
        """Write the sessions' due events and save the snapshot (the loop's thread)."""
        now = self.now()
        for tracker in list(self.trackers.values()):
            with self._lock:
                due = tracker.unsent and _event_due(tracker, now)
                if due:
                    tracker.event_at, tracker.unsent = now, False
            if due:
                self._state_event(tracker, now)
        if self._dirty and self.save is not None:
            self._dirty = False
            if self.loop is not None and not self.loop.is_closed():
                self._saved = self.loop.time()
            with contextlib.suppress(Exception):  # a live view: it never fails the loop
                self.save(self.snapshot())
        loop = self.loop
        if (
            loop is not None
            and not loop.is_closed()
            and any(t.unsent for t in list(self.trackers.values()))
        ):
            self._arm()  # a timer for the next session event that falls due


def _event_due(tracker: Tracker, now: float) -> bool:
    """Whether ``tracker``'s next session_state event may be written at ``now`` (the same
    comparison :meth:`Observer._next_event_in` waits for)."""
    return tracker.event_at + EVENT_S <= now


def _job(event: dict[str, Any], now: float) -> dict[str, Any]:
    keep = ("id", "kind", "class", "session", "target")
    out = {k: event[k] for k in keep if event.get(k) is not None}
    return {**out, "since": event.get("ts"), "for_s": round(max(now - float(event["ts"]), 0.0), 1)}


def _quantile(values: list[float], q: float) -> float:
    """The ``q`` quantile of sorted ``values`` (linear between the closest ranks)."""
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


# ------------------------------------------------------------------ reading the records


@dataclass
class Session:
    """One session as its records tell it (:func:`replay`)."""

    label: str
    agent: str = ""
    role: str = ""
    arm: str | None = None
    start: float = 0.0
    end: float | None = None  # None: open
    status: str | None = None
    state: str = STARTING
    detail: str = ""
    since: float = 0.0
    evaluations: int = 0
    usd: float | None = None
    reserved: float | None = None
    # (start, end or None while it lasts, state, detail)
    segments: list[tuple[float, float | None, str, str]] = field(default_factory=list)

    def split(self, now: float | None = None) -> dict[str, float]:
        """Seconds per part of :data:`PARTS` (an open session's up to ``now``)."""
        out = dict.fromkeys(PARTS, 0.0)
        for a, b, state, _ in self.segments:
            end = b if b is not None else (now if now is not None else a)
            out[PART.get(state, "idle")] += max(end - a, 0.0)
        return out

    def seconds(self, now: float | None = None) -> float:
        end = self.end if self.end is not None else (now if now is not None else self.since)
        return max(end - self.start, 0.0)


def _records(path: Path) -> list[dict[str, Any]]:
    """The JSON lines of ``path`` ([] when missing); a line being written is left out."""
    try:
        text = path.read_text()
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        with contextlib.suppress(ValueError):
            record = json.loads(line)
            if isinstance(record, dict) and "ts" in record:
                out.append(record)
    return out


def read(run: RunDir) -> list[dict[str, Any]]:
    """The records of ``sessions.jsonl``."""
    return _records(run.root / FILE)


def read_gpu(run: RunDir) -> list[dict[str, Any]]:
    """The ``gpu_job`` events of ``gpu_queue.jsonl``."""
    return _records(run.root / gpuqueue.FILE)


def replay(records: Iterable[dict[str, Any]]) -> list[Session]:
    """The sessions the records of ``sessions.jsonl`` tell, in the order they started."""
    out: list[Session] = []
    live: dict[str, Session] = {}
    ordered = sorted(
        (r for r in records if isinstance(r.get("ts"), int | float) and r.get("label")),
        key=lambda r: float(r["ts"]),
    )
    for r in ordered:
        label, ts, state = str(r["label"]), float(r["ts"]), str(r.get("state") or "")
        s = live.get(label)
        if "role" in r:  # a new session (one still open under its label was cut short)
            if s is not None:
                _end(s, ts, "interrupted")
            s = Session(
                label,
                agent=str(r.get("agent") or label),
                role=str(r.get("role") or ""),
                arm=r.get("arm"),
                start=ts,
                since=ts,
                reserved=r.get("reserved"),
            )
            out.append(s)
            live[label] = s
        if s is None:
            continue  # a session whose first record is missing
        if "evaluations" in r:
            s.evaluations = int(r.get("evaluations") or 0)
        if state == ENDED:
            _end(s, ts, str(r.get("status") or "done"))
            usd = r.get("usd")
            s.usd = float(usd) if isinstance(usd, int | float) else None
            del live[label]
            continue
        if s.segments:
            a, _, last, detail = s.segments[-1]
            s.segments[-1] = (a, ts, last, detail)
        s.segments.append((ts, None, state, str(r.get("detail") or "")))
        s.state, s.detail, s.since = state, str(r.get("detail") or ""), ts
    return out


def _end(s: Session, ts: float, status: str) -> None:
    if s.segments:
        a, _, state, detail = s.segments[-1]
        s.segments[-1] = (a, ts, state, detail)
    s.end, s.status, s.state, s.since = ts, status, ENDED, ts


def recover(run: RunDir) -> list[str]:
    """Close the sessions ``sessions.jsonl`` leaves open (a process that did not end cleanly)
    as interrupted at their last record; returns their labels."""
    closed = []
    path = run.root / FILE
    with contextlib.suppress(OSError):  # a last line the crash cut short stays on its own
        if path.stat().st_size and not path.read_bytes().endswith(b"\n"):
            with path.open("a") as fh:
                fh.write("\n")
    for s in replay(read(run)):
        if s.end is not None:
            continue
        split = {part: round(v, 1) for part, v in s.split(s.since).items()}
        append_jsonl(
            run.root / FILE,
            {
                "ts": round(s.since, 3),
                "label": s.label,
                "state": ENDED,
                "status": "interrupted",
                "seconds": round(s.since - s.start, 1),
                "split": split,
                "evaluations": s.evaluations,
            },
        )
        closed.append(s.label)
    return closed


@dataclass(frozen=True)
class Hold:
    """One acquisition of the GPU (``gpu_queue.jsonl``): from ``start`` to ``end`` (None: it
    still holds it), after waiting ``wait`` seconds; ``session`` None: background work."""

    start: float
    end: float | None
    kind: str
    job_class: str
    session: str | None
    wait: float


def gpu_holds(events: Iterable[dict[str, Any]]) -> list[Hold]:
    """The GPU's holds from the ``gpu_job`` events, in the order they started."""
    started: dict[Any, dict[str, Any]] = {}
    out: list[Hold] = []
    for e in events:
        state, job = e.get("state"), e.get("id")
        if state == "start":
            started[job] = e
        elif state == "done" and job in started:
            s = started.pop(job)
            out.append(_hold(s, float(e["ts"])))
    out += [_hold(s, None) for s in started.values()]
    return sorted(out, key=lambda h: h.start)


def _hold(start: dict[str, Any], end: float | None) -> Hold:
    session = start.get("session")
    return Hold(
        start=float(start["ts"]),
        end=end,
        kind=str(start.get("kind") or ""),
        job_class=str(start.get("class") or ""),
        session=str(session) if session else None,
        wait=float(start.get("wait_s") or 0.0),
    )


def spans(run: RunDir) -> dict[str, tuple[float, float | None]]:
    """``label`` → (start, end) of every session of the run (the last one of a label)."""
    return {s.label: (s.start, s.end) for s in replay(read(run))}


# ------------------------------------------------------------------ status


def agents(run: RunDir) -> tuple[list[Session], list[Session]]:
    """(the open sessions, the ended ones) of the run."""
    sessions = replay(read(run))
    return [s for s in sessions if s.end is None], [s for s in sessions if s.end is not None]


def split_text(split: dict[str, float]) -> str:
    """``model 46%, GPU held 11%, ...`` (the parts with any time)."""
    total = sum(split.values())
    if total <= 0:
        return ""
    return ", ".join(
        f"{TITLES[p]} {split.get(p, 0.0) / total:.0%}" for p in PARTS if split.get(p, 0.0) > 0
    )


def total_split(sessions: Iterable[Session], now: float | None = None) -> dict[str, float]:
    out = dict.fromkeys(PARTS, 0.0)
    for s in sessions:
        for part, seconds in s.split(now).items():
            out[part] += seconds
    return out


def duration(seconds: float) -> str:
    """``42 s``, ``3.5 min``, ``2.1 h``."""
    if seconds < 90:
        return f"{seconds:.0f} s"
    return f"{seconds / 60:.1f} min" if seconds < 5400 else f"{seconds / 3600:.1f} h"


def gpu_lines(run: RunDir, width: int = 120) -> list[str]:
    """``kernel-agent status``'s GPU block beyond ``gpuqueue.status_lines``: the jobs waiting
    now by class, and the GPU's busy share and waits over the last hour of its queue (up to
    now while a job holds or waits for the GPU, else up to its last record)."""
    events = read_gpu(run)
    holds = gpu_holds(events)
    if not holds:
        return []
    last = max(max(h.end or h.start for h in holds), float(events[-1]["ts"]))
    now = max(time.time(), last)
    lines = []
    waiting: dict[Any, dict[str, Any]] = {}
    for e in events:
        if e.get("state") == "queued":
            waiting[e.get("id")] = e
        elif e.get("state") in ("start", "withdrawn"):
            waiting.pop(e.get("id"), None)
    if waiting:
        by: dict[str, list[float]] = {}
        for e in waiting.values():
            by.setdefault(str(e.get("class")), []).append(max(now - float(e["ts"]), 0.0))
        parts = [f"{c} {len(w)} (longest {duration(max(w))})" for c, w in by.items()]
        lines.append(("  queue by class: " + ", ".join(parts))[:width])
    end = now if waiting or any(h.end is None for h in holds) else last
    lo = end - HOUR_S
    busy = sum(min(h.end or end, end) - max(h.start, lo) for h in holds if (h.end or end) > lo)
    span = min(HOUR_S, max(end - holds[0].start, 1e-9))
    waits = sorted(h.wait for h in holds if h.start >= lo)
    text = f"  last hour of the queue: GPU busy {min(busy / span, 1.0):.0%}"
    if waits:
        p50, p95 = _quantile(waits, 0.5), _quantile(waits, 0.95)
        text += f", waits p50 {duration(p50)}, p95 {duration(p95)} ({len(waits)} jobs)"
    lines.append(text[:width])
    return lines


# ------------------------------------------------------------------ watch


def lanes(
    records: Iterable[dict[str, Any]],
    gpu_events: Iterable[dict[str, Any]],
    *,
    now: float | None = None,
    columns: int = LANE_COLUMNS,
) -> dict[str, Any]:
    """The swimlanes of ``kernel-agent watch`` ({} before the first session or GPU job).

    Every session gets a lane, sessions that ran at the same time different ones (as few lanes
    as that needs: one per concurrent session), with its states as segments (seconds after
    its start; ``end`` None while it lasts) coarsened to ``columns`` steps of the span shown;
    ``gpu``: the GPU's holds (seconds after ``start``) of the agents' jobs and of background
    work, adjacent ones of a kind merged at that resolution; ``split``: every session's time
    split so far; ``busy``: the GPU's busy share over the span."""
    sessions = replay(records)
    holds = gpu_holds(gpu_events)
    if not sessions and not holds:
        return {}
    start = min([s.start for s in sessions] + [h.start for h in holds])
    ends = [s.end if s.end is not None else s.since for s in sessions]
    ends += [h.end if h.end is not None else h.start for h in holds]
    end = max([*ends, now if now is not None else start])
    step = max(end - start, 1.0) / max(columns, 1)
    free: list[float] = []  # each lane's end so far
    out = []
    for s in sessions:
        stop = s.end if s.end is not None else math.inf
        lane = next((i for i, e in enumerate(free) if e <= s.start + 1e-6), len(free))
        if lane == len(free):
            free.append(stop)
        else:
            free[lane] = stop
        segments = _coarse([(a, b, state) for a, b, state, _ in s.segments], step)
        out.append(
            {
                "label": s.label,
                "agent": s.agent,
                "role": s.role,
                "arm": s.arm,
                "lane": lane,
                "start": round(s.start, 3),
                "end": round(s.end, 3) if s.end is not None else None,
                "state": s.state,
                "detail": s.detail,
                "since": round(s.since, 3),
                "status": s.status,
                "evaluations": s.evaluations,
                "usd": s.usd,
                "split": {p: round(v, 1) for p, v in s.split(now).items()},
                "segments": [
                    [round(a - s.start, 1), None if b is None else round(b - s.start, 1), state]
                    for a, b, state in segments
                ],
            }
        )
    gpu = _gpu_blocks(holds, start, step)
    busy = sum(min(h.end if h.end is not None else end, end) - h.start for h in holds)
    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "lanes": len(free),
        "sessions": out,
        "gpu": gpu,
        "split": {p: round(v, 1) for p, v in total_split(sessions, now).items()},
        "busy": round(min(busy / (end - start), 1.0), 3) if end > start else None,
    }


def _coarse(
    segments: list[tuple[float, float | None, str]], step: float
) -> list[tuple[float, float | None, str]]:
    """``segments`` in blocks of at least ``step`` seconds: a block shorter than that takes
    the segments after it until it is not, and shows the state it spent the most time in;
    adjacent blocks of one state are one."""
    blocks: list[list[Any]] = []  # [start, end, state, {state: seconds}]
    for a, b, state in segments:
        weight = (b - a) if b is not None else step  # an open segment: at least a step
        last = blocks[-1] if blocks else None
        if last is not None and last[1] is not None and last[1] - last[0] < step:
            last[1] = b
            last[3][state] = last[3].get(state, 0.0) + weight
            last[2] = max(last[3], key=last[3].__getitem__)
        elif last is not None and last[1] is not None and last[2] == state:
            last[1] = b
            last[3][state] = last[3].get(state, 0.0) + weight
        else:
            blocks.append([a, b, state, {state: weight}])
    out: list[tuple[float, float | None, str]] = []
    for a, b, state, _ in blocks:  # merging can give two neighbours the same state
        if out and out[-1][2] == state:
            out[-1] = (out[-1][0], b, state)
        else:
            out.append((a, b, state))
    return out


def _gpu_blocks(holds: list[Hold], start: float, step: float) -> list[list[Any]]:
    """The holds as ``[start, end, group, kind, session]`` (seconds after ``start``; group
    ``agent`` or ``background``), a hold that starts within ``step`` of the previous one of
    its group and kind merged into it."""
    out: list[list[Any]] = []
    last: dict[tuple[str, str], list[Any]] = {}
    for h in holds:
        group = "agent" if h.session else "background"
        a = round(h.start - start, 1)
        b = None if h.end is None else round(h.end - start, 1)
        prev = last.get((group, h.kind))
        if prev is not None and prev[1] is not None and a - prev[1] < step:
            prev[1] = b if b is None or b > prev[1] else prev[1]
            if prev[4] != h.session:
                prev[4] = None
            continue
        block = [a, b, group, h.kind, h.session]
        out.append(block)
        last[(group, h.kind)] = block
    return out


# ------------------------------------------------------------------ report


def _window(run: RunDir, sessions: list[Session], holds: list[Hold]) -> list[tuple[float, float]]:
    """The spans the improve loop ran (its phase spans), else the sessions' and GPU's span."""
    stamps = [s.end or s.since for s in sessions] + [h.end or h.start for h in holds]
    last = max(stamps, default=0.0)
    out = [
        (a, b if b is not None else last)
        for phase, a, b in ledger.phase_spans(run)
        if phase == "improve"
    ]
    out = [(a, b) for a, b in out if b > a]
    if out:
        return out
    first = min([s.start for s in sessions] + [h.start for h in holds], default=last)
    return [(first, last)] if last > first else []


def _overlap(a: float, b: float, windows: list[tuple[float, float]]) -> float:
    return sum(max(min(b, y) - max(a, x), 0.0) for x, y in windows)


def report_lines(run: RunDir) -> list[str]:
    """``## Concurrency`` of report.md: the agent sessions' measured time split per role (from
    their states, ``sessions.jsonl``), the sessions at once, and the GPU's use and waits
    (``gpu_queue.jsonl``); [] for a run without these records."""
    sessions = [s for s in replay(read(run)) if s.end is not None]
    holds = [h for h in gpu_holds(read_gpu(run)) if h.end is not None]
    if not sessions and not holds:
        return []
    windows = _window(run, sessions, holds)
    wall = sum(b - a for a, b in windows)
    lines = [
        "",
        "## Concurrency",
        "",
        "Measured from the agent sessions' states (`sessions.jsonl`: thinking, tools, "
        "evaluations, GPU queue) and the GPU job queue (`gpu_queue.jsonl`), as the time split "
        "of docs/MULTIAGENT-DATA.md.",
        "",
    ]
    state = read_json(run.root / "improve.json", {}) or {}
    agents_n = (state.get("config") or {}).get("agents")
    hours = sum(s.seconds() for s in sessions) / 3600
    if wall > 0:
        coordinator = state.get("coordinator") or {}
        peak = f", at most {coordinator['peak']}" if coordinator.get("peak") else ""
        lines.append(
            f"* {len(sessions)} agent sessions, {hours:.1f} session-hours in "
            f"{wall / 3600:.1f} h of the loop: {hours * 3600 / wall:.1f} sessions at once on "
            f"average{peak}" + (f" (--agents {agents_n})" if agents_n else "")
        )
    if holds and wall > 0:
        agent = sum(_overlap(h.start, h.end or h.start, windows) for h in holds if h.session)
        background = sum(
            _overlap(h.start, h.end or h.start, windows) for h in holds if not h.session
        )
        lines.append(
            f"* GPU busy {(agent + background) / wall:.0%} of that time: the agents' jobs "
            f"{agent / wall:.0%}, background work (integration, captures, re-profiles) "
            f"{background / wall:.0%}; {len(holds)} jobs"
        )
        waits = sorted(h.wait for h in holds if h.session)
        if waits:
            waited = sum(waits)
            lines.append(
                f"* the agents' GPU jobs waited {duration(waited)} in all: p50 "
                f"{duration(_quantile(waits, 0.5))}, p95 {duration(_quantile(waits, 0.95))}, max "
                f"{duration(waits[-1])} ({sum(w > 0 for w in waits)} of {len(waits)} waited)"
            )
    evaluations = sum(s.evaluations for s in sessions)
    if wall > 0 and evaluations:
        lines.append(
            f"* {evaluations} agent evaluations: {evaluations / (wall / 3600):.1f} per hour"
        )
    if not sessions:
        return [*lines, ""]
    heads = [TITLES[p] for p in PARTS]
    lines += [
        "",
        "| role | sessions | hours | " + " | ".join(heads) + " | evaluations | per session-hour |",
        "|---|---:|---:|" + "---:|" * len(PARTS) + "---:|---:|",
    ]
    ordered = sorted(
        {s.role for s in sessions}, key=lambda r: -sum(x.seconds() for x in sessions if x.role == r)
    )
    for role in [*ordered, None]:
        mine = [s for s in sessions if role is None or s.role == role]
        split = total_split(mine)
        total = sum(split.values())
        h = sum(s.seconds() for s in mine) / 3600
        n = sum(s.evaluations for s in mine)
        cells = [f"{split[p] / total:.0%}" if total > 0 else "—" for p in PARTS]
        name = "**all**" if role is None else role or "?"
        lines.append(
            f"| {name} | {len(mine)} | {h:.2f} | "
            + " | ".join(cells)
            + f" | {n} | {n / h if h > 0 else 0.0:.1f} |"
        )
    lines += [
        "",
        "Model: a model call of the session's own is in flight (a helper subagent's work "
        "too); GPU held / waiting for the GPU: one of its GPU jobs holds or waits for the GPU; "
        "evaluation off the GPU: an evaluation tool builds, checks or records; own runs: its "
        "Bash, file and search tools; idle: starting, a usage-limit wait, or open after its "
        "turn ended.",
        "",
    ]
    return lines
