"""A simulated clock for tests whose verdicts come from measured time (#211).

Put in place of a module's ``time`` (``monkeypatch.setattr(base, "time", Clock(...))``), it
makes what that module measures exact, however busy the machine is: a loaded machine
(several agents running the suite at once) slows real runs and sleeps by tens of ms.
"""

from __future__ import annotations

import time as _time
from typing import Any


class Clock:
    """Simulated seconds. Every reading (``perf_counter``, ``monotonic``, ``time``) returns
    the current time and then moves it ``tick`` seconds on, so a run timed by two readings
    takes ``tick``; ``sleep`` moves it on at once. Everything else is the ``time`` module's."""

    def __init__(self, tick: float = 0.0, now: float = 1000.0) -> None:
        self.tick = tick
        self.now = now

    def perf_counter(self) -> float:
        now = self.now
        self.now += self.tick
        return now

    monotonic = perf_counter
    time = perf_counter

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def __getattr__(self, name: str) -> Any:
        return getattr(_time, name)
