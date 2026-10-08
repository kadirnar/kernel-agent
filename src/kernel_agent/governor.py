"""``kernel-agent improve --agents auto``: how many agent sessions run at once
(docs/MULTIAGENT.md §3.7, issue #191).

The coordinator (``coordinator.py``) asks the :class:`Governor` how many sessions may run
(:meth:`Governor.update`, at every wake-up) and whether one of a given model and role may
start now (:meth:`Governor.refuses`). **Its objective is the run's wall-clock speed**: as
many sessions as make the optimisation progress faster. It never saves money (USD bounds k
only when ``--max-usd`` is given), and a usage window bounds k only so it does not run out
before it resets: a spent window pauses every session until then, which is slower. It keeps

* every session: ``k = min(cap, k_mem, k_usd, k_rate of the account's windows)``;
* the sessions of the roles that need the GPU (``RoleSpec.needs_gpu``): at most ``k_gpu``;
* the sessions of a model family with a window of its own (``seven_day_opus``,
  ``seven_day_sonnet``): at most that window's ``k_rate``.

**k_gpu**, the GPU's knee ``1 + Z/S`` rounded up, plus one (:data:`KNEE_PLUS`,
docs/MULTIAGENT.md §3.7 and §4: the GPU stays busy while a session thinks longer than the
average; past that, sessions only wait): ``Z`` is the time the sessions of the GPU roles work
off the GPU per evaluation (model, their own runs, an evaluation's own steps:
``sessions.Observer.work``), ``S`` the GPU time per evaluation (every job that held the GPU:
evaluations, sweeps, the re-integration's A/B steps, captures: ``gpuqueue.listen``). The
evaluations cancel out, so the knee is ``1 + work / held``. Both start from the runs studied for
#174 (:data:`PRIOR_Z_S`, :data:`PRIOR_S_S`, the re-integration's heavy end: knee 2, k_gpu 3;
docs/MULTIAGENT-DATA.md put the knee of our runs at 3-4) weighted as :data:`PRIOR_EVALS`
evaluations, so the first sessions do not swing it; it moves once the knee is
:data:`KNEE_MARGIN` past a whole number.

**k_rate**, per usage window (a ``RateLimitEvent``: its window and, in its raw data,
``unifiedWindows``: ``five_hour``, ``seven_day``, ``seven_day_opus``, ``seven_day_sonnet``), AIMD
on the utilization projected for when the window resets or the run's agent time ends, whichever
is first: ``p(k) = u + n(k) · s · H``. ``s`` is the window's utilization per session-second: what
it rose by over the session-seconds it counted between its events, the last hour or so weighing
most (:func:`memory_s`; the events' two-decimal rounding averages out), on top of a prior of
:data:`PRIOR_SESSION_H` session-hours at the measured rate (one session uses about 10 points of
the 5-hour window and 2.3 of the 7-day window per hour, docs/MULTIAGENT-DATA.md §7). ``n(k)``,
the sessions it will count with k allowed, is k times the share of its allowed slots that ran a
session over about the last hour (from a prior of 1), and at least the sessions running now.
k_rate starts at the cap. Once ``p(k)`` passes :data:`SHRINK_ABOVE` (100 %: the window would
run out before it resets) it goes down just enough, to the largest k with ``p`` under
:data:`TARGET` (95 %, the rest a margin for the estimates), at least one less, and at least
half when ``p(k)`` is past :data:`HALVE_ABOVE` or on a ``rejected`` event: the multiplicative
decrease, before the window is spent rather than after. After :data:`QUIET_S` without a change
or an ``allowed_warning`` it goes up by one while ``p(k + 1)`` stays under :data:`TARGET` (the
additive increase into headroom; ``s`` and the occupancy are measured at k, and on the safe
side for k + 1, whose sessions wait longer for the GPU). Between the two thresholds k holds,
so the window fills to about 95-100 % by its reset and never stalls the run if the estimates
hold. A window counts the sessions whose
model it meters (:func:`windows_of`): every model's sessions use the 5-hour and 7-day windows,
Opus sessions the Opus window too, so a spent Opus window holds the Opus roles while a Sonnet
dossier may still start. Without rate-limit events (an API key) there is no window. k never
goes below 1: when one session alone would spend a window, the rate gate still waits for it.

**k_mem**, the host's memory (docs/MULTIAGENT.md §6.4): each session's Claude Code CLI and its
builds (:data:`SESSION_GB`) beside one evaluator (:data:`EVALUATOR_GB`) and a reserve, out of
what is available plus what kernel-agent's processes hold already (``/proc``, :func:`host_memory`).

**k_usd**: the running sessions and as many more as what is left of ``--max-usd`` after their
reservations covers (the coordinator's; it still checks each start's own reservation).

Lowering k never stops a session: none starts until fewer run. With nothing running every term
is at least 1, so the first session always starts. Every change is a ``governor`` event
(events.jsonl) that says why; ``improve.json`` → ``governor`` has the live terms
(``kernel-agent status``: :func:`status_line`).
"""

from __future__ import annotations

import contextlib
import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernel_agent import interrupt

AUTO = "auto"
AUTO_MAX = 6  # --agents auto: at most this many sessions (docs/MULTIAGENT.md §3.10)
# k_gpu: the knee 1 + Z/S from these priors (the runs studied for #174: about 5 min of work
# per evaluation; per evaluation a kernel evaluation of 9-70 s and 100-350 GPU-seconds of the
# re-integration's A/B steps: the heavy end, so a run starts at 3 and measures its way up),
# weighted as PRIOR_EVALS evaluations
PRIOR_Z_S = 300.0
PRIOR_S_S = 300.0
PRIOR_EVALS = 8.0
KNEE_MARGIN = 0.25  # k_gpu moves once the knee is this far past a whole number
KNEE_PLUS = 1  # sessions past the knee (§3.7): the GPU stays busy while one thinks longer
# k_rate: the usage windows (name -> length in seconds, utilization one session uses per hour)
FIVE_HOUR, SEVEN_DAY = "five_hour", "seven_day"
SEVEN_DAY_OPUS, SEVEN_DAY_SONNET = "seven_day_opus", "seven_day_sonnet"
WINDOWS: dict[str, tuple[float, float]] = {
    FIVE_HOUR: (5 * 3600.0, 0.10),  # MULTIAGENT-DATA.md §7: +10.3 points per session-hour
    SEVEN_DAY: (7 * 86400.0, 0.023),  # +2.34
    SEVEN_DAY_OPUS: (7 * 86400.0, 0.023),  # not measured: as the account's 7-day window
    SEVEN_DAY_SONNET: (7 * 86400.0, 0.023),
}
ACCOUNT = (FIVE_HOUR, SEVEN_DAY)  # every model's sessions use them: they bound k itself
PRIOR_SESSION_H = 0.5  # the prior rate weighs as this many session-hours of measurement
MEMORY_H = {FIVE_HOUR: 1.0}  # a window's rate decays with this time constant (h; others: 4)
OCCUPANCY_S = 3600.0  # a window's occupancy decays with this time constant (about an hour)
PRIOR_OCCUPANCY_S = 900.0  # ... from a prior of full occupancy weighing this many seconds
# a window may be this full when it resets or the run's agent time ends: just under spent (a
# spent window pauses every session until it resets), the rest a margin for the estimates
TARGET = 0.95  # one session more while the projection with it stays under this
SHRINK_ABOVE = 1.0  # fewer once the projection says the window runs out before its reset
HALVE_ABOVE = 1.25  # ... at least half as many past this
QUIET_S = 900.0  # one session more at most this often (and not this soon after a warning)
NEW_WINDOW_S = 60.0  # a reset time this much later than the known one: a new window
# k_mem: GB of host memory
SESSION_GB = 3.0  # a session: its Claude Code CLI (~0.5) and its builds (nvcc, load_inline)
EVALUATOR_GB = 6.0  # one evaluator or end-to-end worker (the model and captures on the host)
RESERVE_GB = 2.0  # the desktop
MEMORY_S = 30.0  # the host's memory is read at most this often
EVENT = "governor"  # events.jsonl


def memory_s(window: str) -> float:
    """Seconds a window's measured rate remembers (its time constant): an hour for the 5-hour
    window, four for the 7-day ones, whose events move less per hour."""
    return MEMORY_H.get(window, 4.0) * 3600


def windows_of(model: str) -> tuple[str, ...]:
    """The usage windows a session of ``model`` uses: the account's 5-hour and 7-day windows,
    and its family's 7-day window (Opus, Sonnet)."""
    name = model.lower()
    family = (SEVEN_DAY_OPUS,) if "opus" in name else ()
    family = (SEVEN_DAY_SONNET,) if "sonnet" in name else family
    return (*ACCOUNT, *family)


def parse_agents(value: int | str) -> tuple[int, bool]:
    """``--agents``: (the most sessions at once, whether the governor decides below that):
    ``N`` -> (N, False); ``auto`` -> (:data:`AUTO_MAX`, True); ``auto:N`` -> (N, True)."""
    if isinstance(value, int):
        return value, False
    head, _, cap = str(value).strip().lower().partition(":")
    if head != AUTO:
        return int(value), False
    return (int(cap) if cap else AUTO_MAX), True


def host_memory() -> tuple[float, float] | None:
    """(GB the host has available, GB kernel-agent's processes hold: the resident memory of
    this process's descendants, i.e. the Claude Code CLIs, their builds and the evaluator);
    None without ``/proc``."""
    try:
        text = Path("/proc/meminfo").read_text()
    except OSError:
        return None
    available = None
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            available = float(line.split()[1]) / 2**20  # kB -> GB
    if available is None:
        return None
    page = os.sysconf("SC_PAGE_SIZE")
    ours = 0
    for pid, _ in interrupt.descendants():
        with contextlib.suppress(OSError, ValueError, IndexError):
            ours += int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * page
    return available, ours / 2**30


@dataclass
class Window:
    """One usage window as the rate-limit events tell it, and the sessions it allows (``k``).

    Its rate per session-second is what it rose by (``rose``) over the session-seconds it
    counted (``counted``) between its events, both decaying with :func:`memory_s` (the
    newest hours of the run count, and its two-decimal rounding averages out), on top of the
    prior (:data:`PRIOR_SESSION_H` session-hours at ``prior``). The sessions it will count
    with k allowed are k times its occupancy (the share of the allowed slots that ran a session
    over about the last hour, :data:`OCCUPANCY_S`: slots stay empty between sessions, at round
    barriers, when no arm is eligible or the GPU is at its knee), and at least those running
    now. Both are measured at the k it allows, so k + 1, whose sessions wait longer for the GPU
    and use none of the window meanwhile, is projected on the safe side."""

    name: str
    k: int
    prior: float  # utilization per session-second before anything is measured
    utilization: float | None = None
    resets_at: float | None = None  # unix time
    status: str = "allowed"
    changed: float = -math.inf  # the clock of its last change of k or warning
    projected: float | None = None  # p(k) at the last update
    rose: float = 0.0
    counted: float = 0.0
    last: tuple[float, float, float] | None = None  # its newest event: clock, u, session-s
    running: int = 0  # sessions it counts now
    occupied: float = 0.0  # session-seconds it counted, decaying (OCCUPANCY_S)
    allowed: float = 0.0  # ... and the sessions it allowed, the same way

    @property
    def occupancy(self) -> float:
        """The share of the sessions it allowed that ran (a prior of full occupancy)."""
        prior = PRIOR_OCCUPANCY_S
        return min((self.occupied + prior) / (self.allowed + prior), 1.0)

    def sessions(self, k: int) -> float:
        """The sessions it will count with ``k`` allowed (class docstring)."""
        return max(k * self.occupancy, min(self.running, k))

    @property
    def per_session_s(self) -> float:
        """Utilization one session uses per second (measured over the prior)."""
        weight = PRIOR_SESSION_H * 3600
        return (self.prior * weight + self.rose) / (weight + self.counted)

    def measure(self, now: float, used: float, counted: float, fresh: bool) -> None:
        """An event: it was ``used`` at ``now``, when the sessions it meters had run
        ``counted`` session-seconds in all (``fresh``: a new window, no rise to count)."""
        if self.last is not None and not fresh:
            then, was, before = self.last
            decay = math.exp(-max(now - then, 0.0) / memory_s(self.name))
            self.rose = self.rose * decay + max(used - was, 0.0)
            self.counted = self.counted * decay + max(counted - before, 0.0)
        self.last = (now, used, counted)

    def renew(self) -> None:
        """A new window (its reset passed): what it measured stays, it starts again."""
        self.last = None
        self.status = "allowed"

    def horizon(self, now: float, left: float | None) -> float:
        """Seconds until it resets or the run's agent time ends, whichever is first."""
        length = WINDOWS.get(self.name, (5 * 3600.0, 0.0))[0]
        until = self.resets_at - now if self.resets_at is not None else length
        return max(min(until, left) if left is not None else until, 0.0)

    def project(self, k: int, horizon: float) -> float:
        """Its utilization when it resets (or the run's agent time ends) with ``k`` sessions
        allowed."""
        return float(self.utilization or 0.0) + self.sessions(k) * self.per_session_s * horizon

    def fit(self, horizon: float, cap: int) -> int:
        """The most sessions that keep it under :data:`TARGET` (at least 1, at most ``cap``)."""
        return max([k for k in range(1, cap + 1) if self.project(k, horizon) <= TARGET] or [1])

    def to_dict(self, now: float) -> dict[str, Any]:
        out: dict[str, Any] = {
            "k": self.k,
            "utilization": None if self.utilization is None else round(self.utilization, 3),
            "per_session_h": round(self.per_session_s * 3600, 4),
            "measured_h": round(self.counted / 3600, 2),
            "occupancy": round(self.occupancy, 2),
            "status": self.status,
        }
        if self.resets_at is not None:
            out["resets_at"] = round(self.resets_at, 1)
            out["resets_in_h"] = round(max(self.resets_at - now, 0.0) / 3600, 2)
        if self.projected is not None:
            out["projected"] = round(self.projected, 3)
        return out


class Governor:
    """``--agents auto``: the sessions the coordinator keeps running (module docstring).

    ``cap``: ``--agents auto:N`` (default :data:`AUTO_MAX`); ``clock``: unix time (a dry run's
    simulated one); ``time_left``: seconds the run still allows agents (None: no limit);
    ``work``: seconds the sessions of the GPU roles have worked off the GPU so far
    (``sessions.Observer.work``); ``memory``: :func:`host_memory` (None: not measured, a dry
    run); ``on_change``: called with each decision that changed a term (the coordinator's
    ``governor`` event). :meth:`gpu` takes the GPU queue's events from any thread."""

    def __init__(
        self,
        cap: int = AUTO_MAX,
        *,
        clock: Callable[[], float] = time.time,
        time_left: Callable[[], float | None] = lambda: None,
        work: Callable[[], float] = lambda: 0.0,
        memory: Callable[[], tuple[float, float] | None] | None = host_memory,
        on_change: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.cap = max(int(cap), 1)
        self.clock = clock
        self.time_left = time_left
        self.work = work
        self.memory = memory
        self.on_change = on_change
        self._lock = threading.Lock()
        self.windows: dict[str, Window] = {}
        self.sessions: dict[str, tuple[tuple[str, ...], bool]] = {}  # label -> windows, GPU
        self.held = 0.0  # GPU seconds held since the start
        self.worked0 = work()  # the sessions' work before the start
        self.k_gpu = min(max(math.ceil(self.knee()) + KNEE_PLUS, 1), self.cap)
        self.k_mem: int | None = None
        self.memory_gb: tuple[float, float] | None = None
        self.k_usd: int | None = None
        self.k = min(self.cap, self.k_gpu)
        self.changes = 0
        self.low = self.high = self.k  # the lowest and highest k so far
        self._used: dict[str, float] = {}  # session-seconds each window counted so far
        self._reasons: list[str] = []  # why terms changed since the last decision reported
        self._reported: dict[str, Any] | None = None  # the terms of that decision
        self._at = self._start = self._k_at = clock()
        self._k_area = 0.0  # the integral of k over time (mean_k)
        self._memory_at = -math.inf

    # -------------------------------------------------------- inputs

    def gpu(self, event: dict[str, Any]) -> None:
        """A ``gpu_job`` event (``gpuqueue.listen``, any thread): a job's GPU hold is ``S``."""
        if event.get("state") == "done":
            with self._lock:
                self.held += float(event.get("hold_s") or 0.0)

    def started(self, label: str, model: str, needs_gpu: bool) -> None:
        """A session of ``model`` started (its role needs the GPU or not)."""
        self._advance()
        self.sessions[label] = (windows_of(model), needs_gpu)

    def ended(self, label: str) -> None:
        self._advance()
        self.sessions.pop(label, None)

    def see(self, info: Any) -> None:
        """A rate-limit event's ``rate_limit_info``: its status, window, utilization and reset
        time, and every window's utilization and reset (``raw["unifiedWindows"]``)."""
        raw = getattr(info, "raw", None)
        found: dict[str, dict[str, Any]] = {}
        unified = raw.get("unifiedWindows") if isinstance(raw, dict) else None
        for name, entry in (unified if isinstance(unified, dict) else {}).items():
            if isinstance(entry, dict):
                found[str(name)] = {"u": entry.get("utilization"), "at": entry.get("resetsAt")}
        kind, status = getattr(info, "rate_limit_type", None), getattr(info, "status", None)
        if kind:
            entry = found.setdefault(str(kind), {})
            if (used := getattr(info, "utilization", None)) is not None:
                entry["u"] = used
            if (at := getattr(info, "resets_at", None)) is not None:
                entry["at"] = at
        self._advance()
        for name, entry in found.items():
            if name in WINDOWS:
                self._observe(
                    name, entry.get("u"), entry.get("at"), status if name == kind else None
                )

    def _observe(self, name: str, used: Any, resets: Any, status: str | None) -> None:
        now = self.clock()
        if name not in self.windows:
            prior = WINDOWS[name][1] / 3600
            self.windows[name] = Window(name, k=self.cap, prior=prior)
        win = self.windows[name]
        at = float(resets) if isinstance(resets, int | float) else win.resets_at
        fresh = at is not None and win.resets_at is not None and at > win.resets_at + NEW_WINDOW_S
        if isinstance(used, int | float):
            used = max(float(used), 0.0)
            if win.utilization is not None and used < win.utilization - 0.005:
                fresh = True  # it reset meanwhile
        if fresh:
            win.renew()
        win.resets_at = at
        if isinstance(used, int | float):
            win.measure(now, used, self._used.get(name, 0.0), fresh)
            win.utilization = used
        if status == "rejected":  # spent: halve (the rate gate waits for the reset)
            win.utilization = max(float(win.utilization or 0.0), 1.0)
            if win.status != "rejected":
                self._set(win, max(1, win.k // 2), f"{name} window rejected", now)
        elif status == "allowed_warning" and win.status != "allowed_warning":
            win.changed = now  # no session more for a while after a warning
        if status is not None:
            win.status = status

    def _advance(self, now: float | None = None) -> None:
        """Count the session-seconds since the last call toward each window, and each
        window's occupancy (the sessions it counted against those it allowed)."""
        now = self.clock() if now is None else now
        span = max(now - self._at, 0.0)
        self._at = now
        for names, _ in self.sessions.values():
            for name in names:
                self._used[name] = self._used.get(name, 0.0) + span
        decay = math.exp(-span / OCCUPANCY_S)
        for name, win in self.windows.items():
            win.running = sum(name in names for names, _ in self.sessions.values())
            win.occupied = win.occupied * decay + win.running * span
            win.allowed = win.allowed * decay + min(self.k, win.k) * span

    # -------------------------------------------------------- decisions

    def knee(self) -> float:
        """The GPU's knee ``1 + Z/S`` (module docstring)."""
        with self._lock:
            held = self.held
        worked = max(self.work() - self.worked0, 0.0)
        return 1.0 + (PRIOR_EVALS * PRIOR_Z_S + worked) / (PRIOR_EVALS * PRIOR_S_S + held)

    def update(self, k_usd: int | None = None) -> int:
        """Decide again: k_gpu, each window's k_rate, k_mem and ``k_usd`` (the coordinator's);
        returns k, the sessions that may run in all."""
        now = self.clock()
        self._advance(now)
        knee = self.knee()
        up, down = math.ceil(knee - KNEE_MARGIN), math.ceil(knee + KNEE_MARGIN)
        up, down = up + KNEE_PLUS, down + KNEE_PLUS
        k_gpu = up if up > self.k_gpu else down if down < self.k_gpu else self.k_gpu
        k_gpu = min(max(k_gpu, 1), self.cap)
        if k_gpu != self.k_gpu:
            self._reasons.append(f"GPU knee 1 + Z/S = {knee:.1f}: k_gpu {self.k_gpu} -> {k_gpu}")
            self.k_gpu = k_gpu
        left = self.time_left()
        for win in self.windows.values():
            if win.resets_at is not None and now >= win.resets_at:  # it reset: a fresh window
                win.renew()
                win.utilization, win.resets_at = 0.0, None
            if win.utilization is None:
                continue
            horizon = win.horizon(now, left)
            first = win.projected is None  # down from the cap: no quiet period after it
            win.projected = projected = win.project(win.k, horizon)
            ends = left is not None and win.horizon(now, None) > left
            when = "the run's end" if ends else "its reset"
            if projected > SHRINK_ABOVE and win.k > 1:  # it would run out: back off, just enough
                k = min(win.k - 1, win.fit(horizon, self.cap))
                if projected > HALVE_ABOVE:  # well past it: at least halve
                    k = min(k, win.k // 2)
                why = f"{win.name} window {projected:.0%} at {when}"
                self._set(win, max(k, 1), why, win.changed if first else now)
            elif (
                win.k < self.cap
                and now - win.changed >= QUIET_S
                and (more := win.project(win.k + 1, horizon)) <= TARGET
            ):
                reason = f"{win.name} window {more:.0%} at {when} with one more"
                self._set(win, win.k + 1, reason, now)
        if self.memory is not None and now - self._memory_at >= MEMORY_S:
            self._memory_at = now
            with contextlib.suppress(Exception):  # a probe: never fails a decision
                self.memory_gb = self.memory()
            if self.memory_gb is not None:
                self._memory(*self.memory_gb)
        self.k_usd = None if k_usd is None else max(int(k_usd), 1)
        self._k_area += self.k * max(now - self._k_at, 0.0)
        self._k_at = now
        terms = {
            "cap": self.cap,
            "k_mem": self.k_mem or self.cap,
            "k_usd": self.k_usd or self.cap,
            **{name: w.k for name, w in self.windows.items() if name in ACCOUNT},
        }
        self.k = min(terms.values())
        self.low, self.high = min(self.low, self.k), max(self.high, self.k)
        if (decision := self._terms()) != self._reported:
            was = self._reported or {}
            if was and self.k != was.get("k"):  # what bounds it now
                bound = [name for name, k in terms.items() if k == self.k]
                self._reasons.append(f"k {was.get('k')} -> {self.k} ({', '.join(bound)})")
            self.changes += 1 if was else 0
            why = "; ".join(self._reasons) or "start"
            self._reported, self._reasons = decision, []
            if self.on_change is not None:
                self.on_change({**self.snapshot(), "why": why})
        return self.k

    def _memory(self, available: float, ours: float) -> None:
        """k_mem from the host's memory: the GB other programs do not hold (available plus
        kernel-agent's own) less the reserve and one evaluator, in sessions; up only with
        half a session to spare (no flapping)."""
        room = available + ours - RESERVE_GB - EVALUATOR_GB
        k = min(max(math.floor(room / SESSION_GB), 1), self.cap)
        old = self.k_mem
        if old is None or k < old or room >= (old + 1.5) * SESSION_GB:
            if old is not None and k != old:
                self._reasons.append(f"{room:.0f} GB of host memory: k_mem {old} -> {k}")
            self.k_mem = k

    def _set(self, win: Window, k: int, why: str, now: float) -> None:
        if k != win.k:
            self._reasons.append(f"{why}: k {win.k} -> {k}")
            win.k, win.changed = k, now

    def _terms(self) -> dict[str, Any]:
        """What a decision is (a change is a ``governor`` event; the USD term counts through k
        only: it moves with every session that ends)."""
        return {
            "k": self.k,
            "k_gpu": self.k_gpu,
            "k_mem": self.k_mem,
            "windows": {name: w.k for name, w in self.windows.items()},
        }

    def mean_k(self) -> float:
        """k averaged over the time since the start (up to the last update)."""
        span = self._k_at - self._start
        return self._k_area / span if span > 0 else float(self.k)

    def refuses(self, model: str, needs_gpu: bool) -> str | None:
        """Why a session of ``model`` (of a role that needs the GPU or not) may not start now
        besides k (None: it may): the GPU sessions are at the knee, or its model's own window
        allows no more of its sessions."""
        running = list(self.sessions.values())
        if needs_gpu and sum(gpu for _, gpu in running) >= self.k_gpu:
            return f"GPU knee: {self.k_gpu} sessions that use the GPU run"
        for name in windows_of(model):
            win = self.windows.get(name)
            if win is None or name in ACCOUNT:
                continue
            if sum(name in names for names, _ in running) >= win.k:
                return f"{name} window: {win.k} of its sessions run"
        return None

    # -------------------------------------------------------- views

    def snapshot(self) -> dict[str, Any]:
        """``improve.json`` → ``governor``: k and its terms now."""
        now = self.clock()
        with self._lock:
            held = self.held
        worked = max(self.work() - self.worked0, 0.0)
        out: dict[str, Any] = {
            "k": self.k,
            "cap": self.cap,
            "k_gpu": self.k_gpu,
            "knee": round(self.knee(), 2),
            "worked_s": round(worked, 1),
            "held_s": round(held, 1),
            "windows": {name: w.to_dict(now) for name, w in sorted(self.windows.items())},
            "running": len(self.sessions),
            "changes": self.changes,
            "low": self.low,
            "high": self.high,
        }
        if self.k_mem is not None:
            out["k_mem"] = self.k_mem
        if self.memory_gb is not None:
            available, ours = self.memory_gb
            out["memory_gb"] = {"available": round(available, 1), "ours": round(ours, 1)}
        if self.k_usd is not None:
            out["k_usd"] = self.k_usd
        return out


def status_line(state: dict[str, Any] | None) -> str | None:
    """``kernel-agent status``: the governor's k and its terms (``improve.json`` →
    ``governor``; None: no governor)."""
    if not state:
        return None
    parts = [f"GPU knee {state.get('k_gpu')} (1 + Z/S = {state.get('knee')})"]
    for name, w in (state.get("windows") or {}).items():
        used = w.get("utilization")
        text = f"{name} {w.get('k')}"
        if used is not None:
            text += f" ({float(used):.0%} used"
            if w.get("projected") is not None:
                text += f", {float(w['projected']):.0%} projected"
            text += ")"
        parts.append(text)
    if state.get("k_mem") is not None:
        parts.append(f"memory {state['k_mem']}")
    if state.get("k_usd") is not None:
        parts.append(f"$ {state['k_usd']}")
    return f"governor: k = {state.get('k')} of {state.get('cap')}: " + ", ".join(parts)
