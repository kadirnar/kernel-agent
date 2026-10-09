"""Concurrent agent sessions: ``kernel-agent improve --agents N`` (docs/MULTIAGENT.md §3.2,
§3.7, §3.8 and §3.12.2; issue #183).

``Improver._loop`` hands the loop to :class:`Coordinator` when ``--agents`` is above 1; with
``--agents 1`` it stays the sequential loop. The coordinator keeps up to N sessions running
(slices, research sessions, dossiers) as tasks of one ``asyncio.TaskGroup`` and decides
again whenever something changes: a session ends, the background re-integration ends, the
rate gate opens, a role's first session streams, a breaker's pause ends (and every
:data:`TICK_S` at the latest).

* **Slots** go to arms by the scheduler's scores with virtual pulls (``scheduler.assign``):
  a running session counts as if it had made its evaluations and found nothing, so the
  slots spread over the arms that pay. An arm gets its research session (or dossier) first
  when one is due, as in the sequential loop; it runs one session at a time, one per island
  with ``--islands`` (``Arm.max_sessions``; the island UCB picks which, ``workers.py``), and
  none while its research session runs; a role runs at most its
  ``RoleSpec.max_concurrent`` sessions (``roles.REGISTRY``; ``--role-max`` overrides). The
  native arm is not held while kernel arms improve: it takes a slot no other arm can use.
* **GPU-free roles first on a saturated GPU**: while a new evaluation would wait more than
  :data:`GPU_SATURATED_S` for the GPU (``gpuqueue.expected_wait``), a due session of a role
  that needs no GPU (``RoleSpec.needs_gpu``: research, dossier) takes the next slot before
  another engineer, who would only lengthen the queue.
* **Staggered starts** (prompt cache, §3.12.2): every session of a role has the same system
  prompt (``prompts.stable_prefix``, #181), so a role's first session starts alone and
  writes the cache; the role's others start once it has streamed its first message (or
  :data:`STAGGER_S` later) and read it instead of each writing one.
* **USD**: a session reserves its role's expected cost (``Budget.expected_usd``). One starts
  only while what is left minus the reservations covers it (or nothing runs), and its own
  cap is what is left minus the others' reservations (``Budget.agent_config``).
* **Rate gate** (:class:`RateGate`): a usage limit is the account's. The first session that
  stops at one closes the gate until the limit resets; no session starts meanwhile, and the
  sessions stopped at it wait on the gate (one wait, counted once) and are resumed when it
  opens. Every session's rate-limit events reach it (``runner.listening``).
* **Re-integration** in the background, at most one at a time, every ``--integrate-every``
  kept results; the sessions keep working meanwhile.
* **Round barrier**: a new round (re-profile, re-plan) starts only when no session and no
  re-integration runs.
* **Failure isolation**: a session that raises is recorded as failed and never cancels the
  others; Ctrl-C cancels all of them and each records itself as interrupted. Circuit
  breakers: an arm whose last :data:`ARM_FAILS` slices of the round failed retires for the
  round (``failing``), a role whose last :data:`ROLE_FAILS` sessions failed pauses for
  :data:`ROLE_PAUSE_S`, and ``MAX_FAILED_SLICES`` failed slices in a row (in the order they
  end) of more than one arm, or twice that many of one arm, stop the loop (the SDK or the
  login is broken).
* **Drain**: once a budget is spent (or ``--max-slices`` slices started, or the breakers stop
  the loop) nothing new starts and the loop waits for the running sessions; the final
  integration then has the GPU alone.
* **Ownership**: every engineer session may write only in its own directory
  (``Orchestrator.ownership``, ``runner.write_guard``). With ``--overlap avoid`` no two
  sessions work on overlapping modules at once; with ``warn`` (default) a session's digest
  names the concurrent sessions that do.
* **Board** (``board.py``, issue #187): a slice's modules are posted as its ``claim`` when it
  starts and its ``release`` when it ends; a ``claim`` its agent posts (``post_note``: a
  systems agent before a transform that replaces a module) joins its claims, which
  ``--overlap`` then weighs.
* **Governor** (``--agents auto``, ``governor.py``, issue #191): the slots are the governor's
  k, decided again at every wake-up from the GPU's knee, each usage window's projected
  utilization (every session's rate-limit events), the host's memory and the USD left; a
  session of a role that needs the GPU starts only below the knee, one of a model with a
  window of its own only below that window's k. Lowering k stops no session.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from claude_agent_sdk import AssistantMessage, RateLimitEvent

from kernel_agent import board, governor, gpuqueue, interrupt, ledger, roles, scheduler
from kernel_agent.agent import runner
from kernel_agent.improve import MAX_FAILED_SLICES, _ts, log, slices_chart
from kernel_agent.native import engine as native_engine
from kernel_agent.profiling import ceilings
from kernel_agent.scheduler import KERNEL, NATIVE, Arm, pick
from kernel_agent.workspace import read_json

if TYPE_CHECKING:
    from kernel_agent.improve import Improver

TICK_S = 300.0  # the loop decides again at least this often
GPU_SATURATED_S = 120.0  # a new evaluation would wait longer for the GPU: GPU-free roles first
STAGGER_S = 30.0  # a role's other sessions start at the latest this long after its first
ARM_FAILS = 3  # failed slices of an arm in a row (this round): retired for the round
ROLE_FAILS = 3  # failed sessions of a role in a row: the role pauses
ROLE_PAUSE_S = 1800.0
OVERLAP = ("allow", "warn", "avoid")  # --overlap
SLICE, RESEARCH, DOSSIER = "slice", "research", "dossier"

#: The modules a session works on: (module class or ``*`` for the whole model, qualname)
Claim = tuple[str, str | None]


def overlaps(a: list[Claim], b: list[Claim]) -> bool:
    """Whether two sessions' claims share a module: the same class (``*``: any), on the same
    instance or one of them on every instance."""
    return any(
        (x[0] == "*" or y[0] == "*" or x[0] == y[0]) and (not x[1] or not y[1] or x[1] == y[1])
        for x in a
        for y in b
    )


# ------------------------------------------------------------------ the rate gate


class RateGate:
    """The one wait for an account-wide usage limit that every session shares (§3.7).

    ``Orchestrator._wait_for_limit`` of the first session stopped at a limit closes it until
    the limit resets (:meth:`close`); every session stopped at it waits on it (:meth:`wait`)
    and is resumed when it opens; ``Budget.waiting`` starts no session while it is closed.
    Each closing is one wait: ``waits`` closings in a row without a session that ended
    without a limit in between (:meth:`progress`) back off as one session's waits do
    (``auth.wait_seconds``, ``auth.MAX_LIMIT_WAITS``). ``clock`` / ``sleep``: the
    orchestrator's (a dry run's simulated ones); ``on_change``: called with ``closed`` /
    ``open`` (the coordinator records it and decides again)."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        on_change: Callable[[str], None] | None = None,
    ) -> None:
        self.clock = clock
        self.sleep = sleep
        self.on_change = on_change
        self.until: float | None = None  # when it opens again (None: it is open)
        self.why = ""
        self.closings = 0  # times it closed
        self.waits = 0  # closings since a session last ended without a usage limit
        self.info: dict[str, Any] = {}  # the newest rate-limit event's status, utilization, ...
        self._open = asyncio.Event()
        self._open.set()
        self._timer: asyncio.Task[None] | None = None

    def closed(self) -> str | None:
        """Why no session may start now (None: the gate is open)."""
        if self._open.is_set():
            return None
        at = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.until or self.clock()))
        return f"usage limit: no agent session starts until {at} ({self.why[:120]})"

    def close(self, seconds: float, why: str) -> bool:
        """Close the gate for ``seconds`` (a later reset extends a closed one); returns
        whether it was open (a new wait)."""
        until = self.clock() + max(seconds, 0.0)
        opened = self._open.is_set()
        if not opened and self.until is not None and until <= self.until + 1.0:  # the same reset
            return False
        if opened:
            self.closings += 1
            self.waits += 1
            self.why = why
        self.until = until
        self._open.clear()
        if self._timer is not None:
            self._timer.cancel()
        self._timer = asyncio.get_running_loop().create_task(self._reopen(until - self.clock()))
        if self.on_change is not None:
            self.on_change("closed" if opened else "extended")
        return opened

    async def _reopen(self, seconds: float) -> None:
        await self.sleep(seconds)
        self._timer, self.until = None, None
        self._open.set()
        if self.on_change is not None:
            self.on_change("open")

    async def wait(self) -> None:
        """Until the gate is open."""
        await self._open.wait()

    def progress(self) -> None:
        """A session ended without a usage limit: the next closing is a first wait again."""
        self.waits = 0

    def see(self, info: Any) -> None:
        """A session's rate-limit event (``RateLimitEvent.rate_limit_info``): its status,
        window and utilization, logged when they change (an ``allowed_warning``)."""
        keys = ("status", "rate_limit_type", "utilization", "resets_at")
        new = {k: getattr(info, k, None) for k in keys}
        if (new["status"], new["rate_limit_type"]) != (
            self.info.get("status"),
            self.info.get("rate_limit_type"),
        ) and new["status"] not in (None, "allowed"):
            used = new["utilization"]
            share = f" at {float(used):.0%}" if isinstance(used, int | float) else ""
            log(f"rate limit: {new['status']} ({new['rate_limit_type'] or 'usage'} window{share})")
        self.info = new

    def stop(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None


# ------------------------------------------------------------------ jobs


@dataclass(eq=False)
class Job:
    """A session the coordinator runs: a slice, research session or dossier of ``arm``."""

    kind: str  # SLICE, RESEARCH or DOSSIER
    arm: Arm
    role: str  # kernel, systems, native, research, dossier
    rec: dict[str, Any]  # its record in improve.json
    arms: list[Arm]  # as the decision that started it saw them
    why: str = ""  # a research session's reason
    evaluations: int = 0  # it is expected to make (its virtual pull)
    claims: list[Claim] = field(default_factory=list)
    claimed: bool = False  # its claims are on the board (board.py): released when it ends
    usd: float = 0.0  # reserved
    note: str = ""  # appended to its digest
    started: float = 0.0  # the loop's time
    streamed: bool = False  # its first message arrived
    task: asyncio.Task[None] | None = None

    @property
    def label(self) -> str:
        return str(self.rec["label"])

    @property
    def agent(self) -> str:
        """Its agent name: a slice's record's (an island's, ``--islands``), else the role's."""
        if self.kind == SLICE:
            return str(self.rec.get("agent") or self.arm.agent)
        return f"{self.role}-{self.arm.id}"

    def running(self) -> scheduler.Running:
        return scheduler.Running(self.arm.id, self.role, self.agent, self.evaluations)


# ------------------------------------------------------------------ the coordinator


class Coordinator:
    """The improve loop with up to ``--agents`` sessions at once (module docstring)."""

    def __init__(self, improver: Improver) -> None:
        self.imp = improver
        self.orch = improver.orch
        self.icfg = improver.icfg
        self.slots = max(int(self.icfg.agents), 1)
        self.jobs: dict[str, Job] = {}  # running, by label
        self.integration: asyncio.Task[None] | None = None  # the background re-integration
        self.integrate_at = 0  # kept results a failed re-integration waits for
        self.started = 0  # slices started (--max-slices)
        # the slices in a row that failed, in the order they ended: (arm, error)
        self.failures: list[tuple[str, str]] = []
        self.role_fails: dict[str, int] = {}
        self.paused: dict[str, float] = {}  # role -> the loop's time it may start again
        self.warm: set[str] = set()  # roles whose first session streamed (staggered starts)
        self.draining: str | None = None
        self.peak = 0  # most sessions at once
        self.gate = RateGate()
        self.governor: governor.Governor | None = None  # --agents auto (run)
        self.wake: asyncio.Event | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.tg: asyncio.TaskGroup | None = None
        self._claims: dict[str, list[Claim]] = {}  # arm -> its claims (one decision)

    # -------------------------------------------------------- main

    async def run(self) -> str:
        """Loop until a budget is spent or every arm has stopped; returns why it stopped."""
        budget = self.orch.budget
        self.loop = asyncio.get_running_loop()
        self.wake = asyncio.Event()
        self.gate = RateGate(clock=self.orch.clock, sleep=self.orch.sleep, on_change=self._gated)
        budget.gate = self.gate
        self.orch.ownership = True  # every engineer session writes only in its own directory
        unlisten: Callable[[], None] | None = None
        if self.icfg.governor:  # --agents auto: the governor decides how many (governor.py)
            self.governor = governor.Governor(
                self.slots,
                clock=self.orch.clock,
                time_left=budget.agent_seconds_left,
                work=self.orch.observer.work,
                memory=None if self.orch.simulated else governor.host_memory,
                on_change=self._governed,
            )
            unlisten = gpuqueue.listen(self.governor.gpu)
            log(f"--agents auto: up to {self.slots} agent sessions at once, as the governor says")
        else:
            log(f"--agents {self.slots}: up to {self.slots} agent sessions at once")
        posts = board.active(self.imp.run)
        if posts is not None:  # an agent's claim joins its session's claims
            posts.listen(self._posted)
        try:
            async with asyncio.TaskGroup() as tg:
                self.tg = tg
                return await self._loop()
        except BaseExceptionGroup as group:  # a bug of the coordinator itself: its first error
            first: BaseException = group
            while isinstance(first, BaseExceptionGroup):
                first = first.exceptions[0]
            raise first from group
        finally:
            if posts is not None:
                posts.unlisten(self._posted)
            if unlisten is not None:
                unlisten()
            self.gate.stop()
            budget.gate = None
            self.orch.ownership = False
            self._abandoned()
            self.imp.state["coordinator"] = {
                "agents": self.slots,
                "peak": self.peak,
                "rate_limit_waits": self.gate.closings,
                **({"governor": self._governor_summary()} if self.governor else {}),
            }
            self.imp.save()

    async def _loop(self) -> str:
        imp = self.imp
        while True:
            interrupt.check()
            imp._keep_time()
            stop = self._stop_reason()
            started = 0
            if stop is None:
                await self.orch.seed_library(imp.targets())  # library priors before slices
                await self.orch.scout_libraries(imp.targets())  # the library bar (#227)
                started = self._fill()
                self._integrate_when_due()
            elif self.jobs and self.draining != stop:
                self.draining = stop
                log(f"draining ({stop}): no new session; {len(self.jobs)} still running")
            if not self.jobs and self.integration is None:
                if stop is not None:
                    return stop
                if not started:
                    idle = await self._idle()
                    if isinstance(idle, str):
                        return idle
                    if idle:  # a new round started: decide at once
                        continue
            await self._sleep()

    def _governed(self, decision: dict[str, Any]) -> None:
        """The governor changed k or one of its terms: a ``governor`` event, the live terms in
        ``improve.json``, a log line."""
        ledger.event(self.imp.run, governor.EVENT, **decision)
        self.imp.state["governor"] = {k: v for k, v in decision.items() if k != "why"}
        self.imp.save()
        log(f"governor: k = {decision['k']} ({decision['why']})")

    def _governor_summary(self) -> dict[str, Any]:
        """``improve.json`` → ``coordinator`` → ``governor``: k's range, mean and changes."""
        assert self.governor is not None
        gov = self.governor
        return {
            "low": gov.low,
            "high": gov.high,
            "mean": round(gov.mean_k(), 2),
            "changes": gov.changes,
        }

    def _slots(self) -> int:
        """The sessions that may run now: ``--agents N``, or the governor's k (``auto``)."""
        if self.governor is None:
            return self.slots
        free = self.orch.budget.free_usd()
        k_usd = None
        if free is not None:  # the running sessions and as many more as the USD left covers
            k_usd = len(self.jobs) + math.floor(max(free, 0.0) / max(self._usd(KERNEL), 0.01))
        return self.governor.update(k_usd)

    def _model(self, role: str) -> str:
        return roles.model_for(role, self.orch.cfg)

    def _gated(self, state: str) -> None:
        """The rate gate closed or opened: an event (``rate_gate``), and decide again."""
        until = {"until": round(self.gate.until, 3)} if self.gate.until else {}
        ledger.event(self.imp.run, "rate_gate", state=state, **until)
        self.poke()

    def poke(self) -> None:
        """Decide again soon (from any thread)."""
        if self.loop is None or self.wake is None or self.loop.is_closed():
            return
        with contextlib.suppress(RuntimeError):  # the loop closed meanwhile
            self.loop.call_soon_threadsafe(self.wake.set)

    async def _sleep(self) -> None:
        assert self.wake is not None
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(TICK_S):
                await self.wake.wait()
        self.wake.clear()

    def _stop_reason(self) -> str | None:
        """Why no new session starts any more (the loop drains): a budget is spent (not the
        rate gate or the reservations: they pass), ``--max-slices`` slices started, or the
        global circuit breaker."""
        budget = self.orch.budget
        if (why := budget.exhausted()) is not None and why != budget.waiting():
            return why
        if (cap := self.icfg.max_slices) is not None and self.started >= cap:
            return f"--max-slices {cap} reached"
        # failures of one arm alone are its breaker's (_breakers), unless they go on
        n, arms = len(self.failures), {arm for arm, _ in self.failures}
        if n >= MAX_FAILED_SLICES and (len(arms) > 1 or n >= 2 * MAX_FAILED_SLICES):
            last = self.failures[-1][1]
            return f"{len(self.failures)} agent sessions in a row failed (last: {last})"
        return None

    async def _idle(self) -> str | bool:
        """Nothing runs and nothing could start: True when a new round started, False to wait
        (the rate gate, a paused role), else why the loop stops (as the sequential loop says
        it)."""
        imp = self.imp
        if self.orch.budget.waiting() is not None:
            return False  # the gate's opening wakes the loop
        arms = self._breakers(imp._pickable(relax=True))
        if pick(arms) is None:
            if await imp._next_round(arms):
                return True
            arms = self._breakers(imp._pickable(relax=True))
            stops = "; ".join(f"{a.id}: {a.stop}" for a in arms)
            return f"every arm has stopped ({stops})" + (
                f"; no new round: {imp.no_round}" if imp.no_round else ""
            )
        arm, short = imp._in_time(arms)
        if arm is None:
            return short
        assert self.loop is not None
        if any(until > self.loop.time() for until in self.paused.values()):
            return False  # a role's pause ends with a wake-up
        return f"no session could start (best arm {arm.id}: --role-max or --overlap)"

    # -------------------------------------------------------- slots

    def _fill(self) -> int:
        """Start sessions in the free slots (module docstring); returns how many started."""
        free = self._slots() - len(self.jobs)
        budget = self.orch.budget
        if free <= 0 or budget.waiting() is not None:
            return 0
        imp = self.imp
        arms = self._breakers(imp._pickable(relax=True))
        self._claims = {}
        saturated = gpuqueue.expected_wait(gpuqueue.EVAL) >= GPU_SATURATED_S
        started = 0
        while started < free:
            running = [job.running() for job in self.jobs.values()]
            options: dict[str, Any] = {
                "role_max": scheduler.role_caps(self.icfg.role_max),
                "time_left": budget.agent_seconds_left(),
                "evaluations": self._evaluations(),
            }
            picked = []
            if saturated:  # a research session or dossier before another engineer
                picked = scheduler.assign(
                    arms, 1, running, imp.policy, ok=self._gpu_free_ok, **options
                )
            picked = picked or scheduler.assign(
                arms, 1, running, imp.policy, ok=self._ok, **options
            )
            if not picked:
                break
            arm = next(a for a in arms if a.id == picked[0].id)  # as built, not pulled
            kind, role, why = self._next(arm)
            if self._staggered(role):
                break  # the role's first session has not streamed yet: this slot waits for it
            self._start(arm, kind, role, why, arms)
            started += 1
        return started

    def _next(self, arm: Arm) -> tuple[str, str, str]:
        """(kind, role, why) of the next session of ``arm``: its research session when one
        is due, its dossier before its first slice, else a slice (as the sequential loop)."""
        if (why := self.imp.research_due(arm)) is not None:
            return RESEARCH, "research", why
        if self.imp.dossier_due(arm):
            return DOSSIER, "dossier", ""
        return SLICE, arm.kind, ""

    def _ok(self, arm: Arm) -> bool:
        """The coordinator's conditions for the next session of ``arm`` now: its role is not
        paused by its breaker, the governor allows one more of its model and role
        (``--agents auto``), its expected USD is covered (or nothing runs), and with
        ``--overlap avoid`` no running session works on the same modules."""
        assert self.loop is not None
        kind, role, _ = self._next(arm)
        if self.paused.get(role, 0.0) > self.loop.time():
            return False
        if self.governor is not None and self.governor.refuses(
            self._model(role), roles.get(role).needs_gpu
        ):
            return False
        free = self.orch.budget.free_usd()
        if free is not None and self.jobs and free < self._usd(role):
            return False
        if self.icfg.overlap == "avoid" and kind == SLICE:
            mine = self._claims_of(arm)
            others = [job for job in self.jobs.values() if job.arm.id != arm.id]  # not its islands
            return not any(overlaps(mine, job.claims) for job in others)
        return True

    def _gpu_free_ok(self, arm: Arm) -> bool:
        """The next session of ``arm`` is of a role that needs no GPU (``RoleSpec.needs_gpu``:
        research, dossier) and may start (:meth:`_ok`)."""
        return not roles.get(self._next(arm)[1]).needs_gpu and self._ok(arm)

    def _staggered(self, role: str) -> bool:
        """Whether a session of ``role`` waits for the role's first one to stream."""
        if not self.icfg.stagger or role in self.warm:
            return False
        return any(job.role == role for job in self.jobs.values())

    def _evaluations(self) -> dict[str, int]:
        """A session's expected evaluations by arm kind (its virtual pull)."""
        native = self.orch.cfg.native_evaluations or self.icfg.slice
        return {KERNEL: self.icfg.slice, scheduler.SYSTEMS: self.icfg.slice, NATIVE: native}

    def _usd(self, role: str) -> float:
        """USD a session of ``role`` reserves: its expected cost, at most ``--budget``."""
        expected = self.orch.budget.expected_usd(role)
        cap = self.orch.cfg.budget_usd_per_agent
        return min(expected, cap) if cap is not None else expected

    def _start(self, arm: Arm, kind: str, role: str, why: str, arms: list[Arm]) -> None:
        assert self.loop is not None and self.tg is not None
        imp = self.imp
        if kind == SLICE:
            rec = imp._open_slice(arm, arms)
            evaluations = self._evaluations().get(arm.kind, self.icfg.slice)
            self.started += 1
        elif kind == RESEARCH:
            rec, evaluations = imp._open_research(arm, why), 0
        else:
            rec, evaluations = imp._open_dossier(arm), 0
        job = Job(kind, arm, role, rec, arms, why=why, evaluations=evaluations)
        job.claims = self._claims_of(arm) if kind == SLICE else []
        job.claimed = board.claim(self.imp.run, job.label, role, arm.id, job.claims)
        job.note = self._note(job)
        job.usd = self._usd(role)
        job.started = self.loop.time()
        self.orch.budget.reserve_usd(job.label, job.usd)
        if self.governor is not None:
            self.governor.started(job.label, self._model(role), roles.get(role).needs_gpu)
        if self.icfg.stagger and not any(j.role == role for j in self.jobs.values()):
            self.loop.call_later(STAGGER_S, self._warmed, job)  # the first of its role now
        self.jobs[job.label] = job
        self.peak = max(self.peak, len(self.jobs))
        job.task = self.tg.create_task(self._run(job), name=job.label)

    def _warmed(self, job: Job) -> None:
        """The first session of a role streamed (or :data:`STAGGER_S` passed): the role's
        other sessions may start."""
        if job.label in self.jobs and job.role not in self.warm:
            self.warm.add(job.role)
            self.poke()

    def _heard(self, job: Job, message: Any) -> None:
        """A message of ``job``'s session (``runner.listening``)."""
        if isinstance(message, AssistantMessage) and not job.streamed:
            job.streamed = True
            self._warmed(job)
        elif isinstance(message, RateLimitEvent):
            self.gate.see(message.rate_limit_info)
            if self.governor is not None:  # every window's utilization (--agents auto)
                self.governor.see(message.rate_limit_info)

    # -------------------------------------------------------- sessions

    async def _run(self, job: Job) -> None:
        """One session and its record; an error is the session's alone (never the group's)."""
        imp = self.imp
        failed, error = False, ""
        what = {
            SLICE: f"slice {job.rec.get('n')} ({job.arm.id})",
            RESEARCH: f"research session of {job.arm.id}",
            DOSSIER: f"dossier of {job.arm.id}",
        }[job.kind]
        try:
            with runner.listening(functools.partial(self._heard, job)), imp._doing(what):
                if job.kind == SLICE:
                    rec = await imp._run_slice(job.rec, job.arm, job.arms, job.note)
                elif job.kind == RESEARCH:
                    rec = await imp._research(job.arm, job.why, job.rec)
                else:
                    rec = await imp._dossier(job.arm, job.rec)
            failed = rec.get("status") == "failed"
            error = str(rec.get("error") or "")
        except Exception as exc:  # not a session's failure (those are records): kept local
            failed, error = True, repr(exc)[:500]
            log(f"{job.label}: {error}")
            imp.activities.pop(asyncio.current_task(), None)
            self._failed_record(job, error)
        finally:
            self.jobs.pop(job.label, None)
            self.orch.budget.release_usd(job.label)
            if self.governor is not None:
                self.governor.ended(job.label)
            if job.claimed or job.claims:  # its modules are free again (board.py)
                board.release(self.imp.run, job.label, job.arm.id, job.claims)
            if not any(j.role == job.role for j in self.jobs.values()):
                self.warm.discard(job.role)  # the next batch of the role staggers again
            self.poke()
        self._ended(job, failed, error)

    def _failed_record(self, job: Job, error: str) -> None:
        """Close the record of a session that raised past its own handling as failed."""
        rec = job.rec
        if rec.get("status") != "running":
            return
        rec["error"] = error
        if job.kind == SLICE:
            self.imp._close(rec, "failed")
        elif job.kind == RESEARCH:
            self.imp._close_research(rec, "failed", plan=False)
        else:
            rec.update(status="failed", ended=_ts())
            self.imp.save()

    def _ended(self, job: Job, failed: bool, error: str) -> None:
        """The breakers after a session ended (Ctrl-C ends none: it cancels them all)."""
        assert self.loop is not None
        if job.kind == SLICE:
            self.failures = [*self.failures, (job.arm.id, error)] if failed else []
        fails = self.role_fails[job.role] = self.role_fails.get(job.role, 0) + 1 if failed else 0
        if fails >= ROLE_FAILS:
            self.role_fails[job.role] = 0
            self.paused[job.role] = self.loop.time() + ROLE_PAUSE_S
            self.loop.call_later(ROLE_PAUSE_S, self.poke)
            log(
                f"breaker: the last {fails} {job.role} sessions failed; no {job.role} session "
                f"for {ROLE_PAUSE_S / 60:.0f} min"
            )
        if self.imp.live_charts:
            slices_chart(self.imp.run)

    def _abandoned(self) -> None:
        """Close the records of sessions that never ran (cancelled before their first step:
        Ctrl-C right after they started) as interrupted."""
        for job in list(self.jobs.values()):
            if job.rec.get("status") == "running":
                if job.kind == SLICE:
                    self.imp._close(job.rec, "interrupted")
                elif job.kind == RESEARCH:
                    self.imp._close_research(job.rec, "interrupted", plan=False)
                else:
                    job.rec.update(status="interrupted", ended=_ts())
                    self.imp.save()
            self.orch.budget.release_usd(job.label)
            if job.claimed or job.claims:
                board.release(self.imp.run, job.label, job.arm.id, job.claims)
        self.jobs.clear()

    # -------------------------------------------------------- breakers and claims

    def _breakers(self, arms: list[Arm]) -> list[Arm]:
        """``arms`` with the live ones whose last :data:`ARM_FAILS` slices of this round
        failed retired for the round (``failing``)."""
        rnd = self.imp.round
        slices = self.imp.state["slices"]
        for arm in arms:
            if arm.stop is not None:
                continue
            mine = [
                s
                for s in slices
                if s["arm"] == arm.id
                and int(s.get("round") or 1) == rnd
                and s.get("status") != "running"
            ]
            last = mine[-ARM_FAILS:]
            if len(last) == ARM_FAILS and all(s.get("status") == "failed" for s in last):
                arm.stop = f"failing: its last {ARM_FAILS} sessions failed (circuit breaker)"
        return arms

    def _claims_of(self, arm: Arm) -> list[Claim]:
        """The modules a slice of ``arm`` works on (§3.6): a kernel target's class and
        qualname; the native arm's current stage's class, else the whole model (``*``);
        none for the systems agent (model-level transforms: the integration resolves them)."""
        if arm.id in self._claims:
            return self._claims[arm.id]
        run = self.imp.run
        claims: list[Claim] = []
        if arm.kind == KERNEL:
            spec = read_json(run.target(arm.id) / "spec.json", {}) or {}
            cls = str(spec.get("module_class") or arm.module_class or arm.id)
            claims = [(cls, spec.get("qualname") or None)]
        elif arm.kind == NATIVE:
            claims = [("*", None)]
            with contextlib.suppress(Exception):  # a claim only: never stops a decision
                columns = ceilings.columns(self.orch.allowed_precisions())
                stage = native_engine.status(run, ledger.rows(run), columns).stage
                if stage is not None and stage.scope == "stage" and stage.module_class:
                    claims = [(stage.module_class, None)]
        self._claims[arm.id] = claims
        return claims

    def _posted(self, entry: dict[str, Any]) -> None:
        """A board entry (``board.Board.listen``, from the thread that posted it): a
        ``claim`` an agent posted adds its module class to its session's claims, so
        ``--overlap`` weighs the modules the agent said it will change (§3.6)."""
        tags = entry.get("tags") or {}
        author, cls = str(entry.get("author")), tags.get("module_class")
        if entry.get("kind") != board.CLAIM or author == board.COORDINATOR or not cls:
            return
        for job in list(self.jobs.values()):
            if author in job.rec.get("sessions", [job.label]) and (cls, None) not in job.claims:
                job.claims = [*job.claims, (str(cls), None)]
                self.poke()

    def _note(self, job: Job) -> str:
        """The digest section of a slice that starts beside other sessions: who runs and
        (``--overlap warn``) who works on the same modules ("" when it runs alone)."""
        if job.kind != SLICE or not self.jobs:
            return ""
        lines = ["", "", f"## Sessions running beside this one (--agents {self.slots})"]
        for other in self.jobs.values():
            on = f"`{other.arm.id}`" + (
                f" ({other.arm.module_class})" if other.arm.module_class else ""
            )
            lines.append(f"* `{other.label}`: {other.role} on {on}")
        lines.append(
            "* Each session writes only in its own directory, and the evaluations take the "
            "GPU in turns (your session's clock stops while yours waits)."
        )
        siblings = [
            o.label for o in self.jobs.values() if o.arm.id == job.arm.id and o.kind == SLICE
        ]
        if siblings:  # the other islands of its target (--islands)
            lines.append(
                f"* {', '.join(f'`{s}`' for s in siblings)}: other islands of your target, each "
                "in its own direction: the integration takes the target's fastest kernel, "
                "whichever island made it."
            )
        others = [o for o in self.jobs.values() if o.arm.id != job.arm.id]
        same = [o.label for o in others if overlaps(job.claims, o.claims)]
        if same and self.icfg.overlap == "warn":
            job.rec["overlap"] = same
            lines.append(
                f"* {', '.join(f'`{s}`' for s in same)} works on modules you work on too: the "
                "integration measures both results and keeps what is fastest together (its "
                "replace steps); build on their kept results in the ledger, not against them."
            )
        return "\n".join(lines)

    # -------------------------------------------------------- re-integration

    def _integrate_when_due(self) -> None:
        """Start the background re-integration after ``--integrate-every`` kept results
        (one at a time; not when the agents' time left is shorter than it: the final
        integration takes them)."""
        every = self.icfg.integrate_every
        if every <= 0 or self.integration is not None or self.tg is None:
            return
        kept = self.imp.keeps_since_integration()
        if kept < max(every, self.integrate_at):
            return
        left = self.orch.budget.agent_seconds_left()
        if left is not None:
            steps, each = self.imp._integration_estimate()
            if steps * each > left:
                return
        why = f"{kept} kept results since the last integration"
        self.integration = self.tg.create_task(self._integrate(why, kept), name="integration")

    async def _integrate(self, why: str, kept: int) -> None:
        try:
            await self.imp.reintegrate(why, background=True)
            self.integrate_at = 0
        except Exception as exc:  # the sessions go on; the next try after more kept results
            log(f"re-integration failed: {exc!r}")
            self.imp.activities.pop(asyncio.current_task(), None)
            self.integrate_at = kept + self.icfg.integrate_every
        finally:
            self.integration = None
            self.poke()
