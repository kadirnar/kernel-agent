"""``kernel-agent improve --agents auto`` (``governor.py``, issue #191): the GPU's knee, AIMD
on each usage window's projected utilization, the host-memory and USD terms, the model
windows; and in a virtual-time dry run with simulated usage windows (``dryrun.SimUsage``) the
governor keeps the window under its target where fixed k runs it dry.

No GPU and no Claude.
"""

import asyncio
import itertools
from types import SimpleNamespace

import pytest

from kernel_agent import charts, cli, dryrun, governor, ledger, orchestrator, status
from kernel_agent.config import HAIKU_MODEL, OptimizeConfig
from kernel_agent.governor import (
    FIVE_HOUR,
    QUIET_S,
    SEVEN_DAY,
    SEVEN_DAY_OPUS,
    TARGET,
    Governor,
)
from kernel_agent.improve import ImproveConfig, Improver
from kernel_agent.workspace import read_json

OPUS, SONNET = "claude-opus-5-5", "claude-sonnet-5-5"
H = 3600.0


class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def info(status="allowed", kind=FIVE_HOUR, used=None, resets=None, **windows):
    """A ``RateLimitInfo`` as Claude Code's events carry it: every window's utilization and
    reset in ``raw["unifiedWindows"]`` (``windows``: name=(utilization, resets_at))."""
    unified = {name: {"utilization": u, "resetsAt": at} for name, (u, at) in windows.items()}
    return SimpleNamespace(
        status=status,
        rate_limit_type=kind,
        utilization=used,
        resets_at=resets,
        raw={"status": status, "unifiedWindows": unified},
    )


def gov(clock, cap=6, *, left=None, work=None, memory=None, changes=None):
    return Governor(
        cap,
        clock=clock,
        time_left=lambda: left,
        work=work or (lambda: 0.0),
        memory=memory,
        on_change=changes.append if changes is not None else None,
    )


# ------------------------------------------------------------------ k_gpu


def test_k_gpu_follows_the_measured_knee():
    clock, worked = Clock(), [0.0]
    g = gov(clock, work=lambda: worked[0])
    z0, s0 = governor.PRIOR_EVALS * governor.PRIOR_Z_S, governor.PRIOR_EVALS * governor.PRIOR_S_S
    assert g.knee() == pytest.approx(2.0)
    assert g.update() == 6 and g.k_gpu == 3  # the priors: knee 2, plus one (k: GPU-free too)
    # agents that think long per GPU second: the knee moves up
    worked[0] = 40 * 600.0  # 40 evaluations of 10 min work each, 60 s of GPU each
    for _ in range(40):
        g.gpu({"state": "done", "hold_s": 60.0})
    g.gpu({"state": "start", "hold_s": 999.0})  # only finished holds count
    knee = g.knee()
    assert knee == pytest.approx(1 + (z0 + 24000) / (s0 + 2400)) and knee > 5
    g.update()
    assert g.k_gpu == 6  # rounded up, plus one: at most the cap
    # then the re-integration's long A/B steps: S grows, the knee comes down
    for _ in range(40):
        g.gpu({"state": "done", "hold_s": 800.0})
    g.update()
    assert g.knee() < 2.0 and g.k_gpu == 3
    # hysteresis: a knee just past a whole number does not move it
    g.k_gpu = 4
    g.held = (g.work() + z0) / (3.1 - 1) - s0  # knee 3.1: 5 by the rule, but within 0.25
    assert g.knee() == pytest.approx(3.1)
    g.update()
    assert g.k_gpu == 4


def test_the_knee_holds_gpu_roles_not_gpu_free_ones():
    clock = Clock()
    g = gov(clock)
    for i in range(3):
        assert g.refuses(OPUS, True) is None
        g.started(f"kernel-{i}", OPUS, True)
    assert "GPU knee" in (g.refuses(OPUS, True) or "")
    assert g.refuses(SONNET, False) is None  # a dossier needs no GPU
    g.ended("kernel-0")
    assert g.refuses(OPUS, True) is None


# ------------------------------------------------------------------ k_rate: AIMD


def test_backs_off_before_the_window_is_spent():
    clock, changes = Clock(), []
    g = gov(clock, changes=changes)
    reset = clock.t + 4 * H
    g.see(info(**{FIVE_HOUR: (0.1, reset)}))
    assert g.update() == 2  # 10 % used, 4 h to the reset, 10 points per session-hour
    win = g.windows[FIVE_HOUR]
    assert win.projected is not None and win.projected > 1.0  # 6 sessions would run it out
    assert win.project(win.k, 4 * H) <= TARGET < win.project(win.k + 1, 4 * H)
    event = changes[-1]
    assert event["k"] == 2 and "five_hour window" in event["why"] and "-> 2" in event["why"]
    assert event["windows"][FIVE_HOUR]["k"] == 2 and event["windows"][FIVE_HOUR]["utilization"]


def test_decrease_is_multiplicative_when_the_window_would_run_out():
    clock = Clock()
    g = gov(clock)
    g.see(info(**{FIVE_HOUR: (0.0, clock.t + 5 * H)}))
    g.update()
    win = g.windows[FIVE_HOUR]
    win.k, win.projected = 4, 0.5  # (as if it had grown)
    win.utilization = 0.55  # then usage jumped: 4 sessions would need 2.6
    g.update()
    assert win.k == 1  # at least halved, down to what fits
    win.k, win.utilization = 4, 0.3  # just over the target with 4: one less, not half
    win.rose, win.counted = 0.0, 1e9  # (a measured rate of ~0: only the prior is left)
    g.update()
    assert win.k >= 2


def test_grows_by_one_after_a_quiet_period_with_headroom():
    clock = Clock()
    g = gov(clock, left=2 * H)  # the run ends in 2 h: the horizon
    g.started("kernel-a", OPUS, True)
    g.started("kernel-b", OPUS, True)
    reset = clock.t + 4 * H
    g.see(info(**{FIVE_HOUR: (0.5, reset)}))
    g.update()
    win = g.windows[FIVE_HOUR]
    assert win.k == 2  # 0.5 + 2 x 0.1 x 2 h = 0.9 at the prior rate
    steps = []
    for _ in range(18):  # 3 h in 10 min steps; the sessions use less than the prior says
        clock.t += 600
        g.see(info(**{FIVE_HOUR: (0.5, reset)}))
        g.update()
        steps.append((clock.t, win.k))
    grew = [(t, k) for (t, k), (_, was) in zip(steps, [(0, 2), *steps], strict=False) if k != was]
    assert len(grew) >= 2 and all(k == i + 3 for i, (_, k) in enumerate(grew))  # one at a time
    assert all(b - a >= QUIET_S for (a, _), (b, _) in itertools.pairwise(grew))
    assert win.per_session_s < win.prior  # what it measured


def test_a_warning_holds_growth_and_a_rejection_halves():
    clock = Clock()
    g = gov(clock, left=0.5 * H)  # half an hour left: room for every session
    reset = clock.t + 4 * H
    g.see(info(**{FIVE_HOUR: (0.2, reset)}))
    g.update()
    win = g.windows[FIVE_HOUR]
    assert win.k == 6
    win.k = 3  # (as if it had been lower)
    g.see(info("allowed_warning", FIVE_HOUR, 0.2, reset))
    clock.t += 60
    g.update()
    assert win.k == 3 and win.status == "allowed_warning"  # no session more after a warning
    clock.t += QUIET_S
    g.update()
    assert win.k == 4
    g.see(info("rejected", FIVE_HOUR, 1.0, reset))
    assert win.k == 2 and win.status == "rejected" and win.utilization == 1.0


def test_a_new_window_starts_fresh_and_keeps_its_rate():
    clock = Clock()
    g = gov(clock)
    g.started("kernel-a", OPUS, True)
    g.started("kernel-b", OPUS, True)
    reset = clock.t + 3 * H
    g.see(info(**{FIVE_HOUR: (0.10, reset)}))
    for step in range(1, 13):  # 2 sessions for 1 h: 0.15 per session-hour, two decimals
        clock.t += 300
        g.see(info(**{FIVE_HOUR: (round(0.10 + 0.3 * step / 12, 2), reset)}))
    win = g.windows[FIVE_HOUR]
    rate = win.per_session_s * H
    assert 0.12 < rate < 0.15  # measured (0.15), over the prior (0.10, half a session-hour)
    clock.t += 3 * H
    g.update()  # its reset passed: 0 % used, until the next event
    assert win.utilization == 0.0 and win.resets_at is None
    assert win.per_session_s * H == pytest.approx(rate, rel=0.05)  # what it learned stays


def test_projection_counts_the_sessions_that_run():
    clock = Clock()
    g = gov(clock)
    g.see(info(**{FIVE_HOUR: (0.1, clock.t + 4 * H)}))
    g.update()
    win = g.windows[FIVE_HOUR]
    full = win.project(2, 4 * H)
    win.occupied, win.allowed = 1800.0, 2 * 3600.0  # half the allowed slots ran a session
    assert win.occupancy < 0.5 and win.project(2, 4 * H) < full
    win.running = 2  # ... but two run now: at least those
    assert win.project(2, 4 * H) == pytest.approx(full)


# ------------------------------------------------------------------ model windows


def test_a_models_own_window_holds_its_sessions_only():
    clock = Clock()
    g = gov(clock)
    assert governor.windows_of(OPUS) == (FIVE_HOUR, SEVEN_DAY, SEVEN_DAY_OPUS)
    assert governor.windows_of(SONNET)[-1] == governor.SEVEN_DAY_SONNET
    assert governor.windows_of(HAIKU_MODEL) == (FIVE_HOUR, SEVEN_DAY)
    reset = clock.t + 100 * H
    g.see(info(**{FIVE_HOUR: (0.1, clock.t + 5 * H), SEVEN_DAY_OPUS: (0.86, reset)}))
    k = g.update()
    assert g.windows[SEVEN_DAY_OPUS].k == 1 and k >= 1  # the account's windows bound k
    g.started("kernel-a", OPUS, True)
    assert "seven_day_opus" in (g.refuses(OPUS, True) or "")
    assert g.refuses(SONNET, False) is None  # a Sonnet dossier still starts


def test_unknown_windows_and_api_keys_have_no_rate_term():
    clock = Clock()
    g = gov(clock)
    g.see(info("allowed", "overage", 0.5, clock.t + H))
    assert g.windows == {} and g.update() == g.cap  # no window: k is the cap's


# ------------------------------------------------------------------ memory, USD, start


def test_host_memory_and_usd_terms():
    clock, changes = Clock(), []
    mem = [(14.0, 1.0)]  # 14 GB available + 1 GB ours: 15 - 2 - 6 = 7 GB for 2 sessions
    g = gov(clock, memory=lambda: mem[0], changes=changes)
    assert g.update() == 2 and g.k_mem == 2
    mem[0] = (16.0, 1.0)  # 9 GB: 3 sessions, but not half a session to spare: no change
    clock.t += governor.MEMORY_S
    assert g.update() == 2
    mem[0] = (20.0, 2.0)  # 14 GB: 4 sessions
    clock.t += governor.MEMORY_S
    assert g.update() == 4 and "k_mem 2 -> 4" in changes[-1]["why"]
    assert g.update(k_usd=1) == 1  # what is left of --max-usd covers one session
    assert "k_usd" in changes[-1]["why"]
    assert governor.host_memory() is not None  # this host's /proc


def test_the_first_session_always_starts():
    clock = Clock()
    g = gov(clock, memory=lambda: (0.5, 0.0))
    g.see(info("rejected", FIVE_HOUR, 1.0, clock.t + H))
    assert g.update(k_usd=0) == 1  # every term is at least 1
    assert g.refuses(OPUS, True) is None  # nothing runs


def test_parse_agents_and_the_cli(monkeypatch, tmp_path):
    assert governor.parse_agents(3) == (3, False)
    assert governor.parse_agents("auto") == (governor.AUTO_MAX, True)
    assert governor.parse_agents("auto:4") == (4, True)
    seen = []
    monkeypatch.setattr(cli, "improve", None, raising=False)

    def fake(ns):
        seen.append(ns)
        return 0

    monkeypatch.setattr(cli, "cmd_improve", fake)
    assert cli.main(["improve", "org/m", "--agents", "auto:4", "--async-evals"]) == 0
    assert governor.parse_agents(seen[0].agents) == (4, True) and seen[0].async_evals
    for bad in ("auto:0", "fast", "0"):
        with pytest.raises(SystemExit):
            cli.main(["improve", "org/m", "--agents", bad])


def test_status_line():
    assert governor.status_line(None) is None
    line = governor.status_line(
        {
            "k": 2,
            "cap": 6,
            "k_gpu": 3,
            "knee": 3.2,
            "windows": {FIVE_HOUR: {"k": 2, "utilization": 0.42, "projected": 0.88}},
            "k_mem": 5,
        }
    )
    assert line == (
        "governor: k = 2 of 6: GPU knee 3 (1 + Z/S = 3.2), five_hour 2 (42% used, 88% projected), "
        "memory 5"
    )


# ------------------------------------------------------------------ the dry run


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(orchestrator.toolchain, "setup", dryrun.SimToolchain)
    monkeypatch.setattr(charts, "available", lambda: False)


def dry_run(tmp_path, agents, *, usage, governed=False, async_evals=False, hours=6.0, **cfg):
    config = OptimizeConfig(
        model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path, dossier=False, max_hours=hours, **cfg
    )
    orch = orchestrator.Orchestrator(dryrun.create_run(config), config)
    world = dryrun.World(orch, virtual=True)
    world.usage = usage
    icfg = ImproveConfig(agents=agents, governor=governed, async_evals=async_evals)
    improver = Improver(orch, icfg, require_capture=False, live_charts=False)

    async def main():
        with world.driving():
            return await improver.improve()

    with world.installed():
        reason = asyncio.run(main())
    return orch, world, reason


def heavy():
    """The measured p90 rate (17.6 points of the 5-hour window per session-hour)."""
    return dryrun.SimUsage.subscription(0.1, 0.3, five_hour_reset_h=4.0, per_hour=(0.21, 0.028))


def test_auto_keeps_the_window_where_fixed_k_runs_it_dry(tmp_path):
    orch, world, _ = dry_run(tmp_path / "fixed", 3, usage=heavy())
    assert world.usage.limited and world.usage.peak[FIVE_HOUR] >= 1.0  # spent: sessions stop
    gate = [e for e in ledger.events(orch.run) if e["event"] == "rate_gate"]
    assert [g["state"] for g in gate][:2] == ["closed", "open"]

    orch, world, reason = dry_run(tmp_path / "auto", 6, usage=heavy(), governed=True)
    assert not world.usage.limited and world.usage.peak[FIVE_HOUR] < 1.0
    assert reason.startswith(("time budget spent", "time left"))
    events = [e for e in ledger.events(orch.run) if e["event"] == governor.EVENT]
    assert events[0]["why"] == "start" and len(events) >= 3
    assert any("five_hour window" in e["why"] for e in events)
    state = read_json(orch.run.root / "improve.json")
    done = state["coordinator"]["governor"]
    assert done["low"] >= 1 and done["high"] <= 6 and 1.0 <= done["mean"] <= 3.0
    assert state["governor"]["windows"][FIVE_HOUR]["measured_h"] > 1.0
    assert not [e for e in ledger.events(orch.run) if e["event"] == "rate_gate"]
    assert "* governor: k between" in orch.run.report.read_text()
    assert "--agents auto (up to 6)" in orch.run.report.read_text()


def test_auto_never_exceeds_the_usd_reservations(tmp_path, monkeypatch):
    from kernel_agent.budget import Budget

    seen = []
    reserve = Budget.reserve_usd

    def checked(self, label, usd):
        reserve(self, label, usd)
        if len(self.reservations) > 1:  # (the first one starts whatever is left)
            assert sum(self.reservations.values()) <= self.max_usd - self.spent_usd() + 1e-9
        seen.append(len(self.reservations))

    monkeypatch.setattr(Budget, "reserve_usd", checked)
    usage = dryrun.SimUsage.subscription(0.0, 0.0)
    orch, _, reason = dry_run(tmp_path, 6, usage=usage, governed=True, max_usd=8.0, hours=None)
    assert "USD budget" in reason and max(seen) >= 2  # sessions at once, within the USD left
    state = read_json(orch.run.root / "improve.json")
    assert state["coordinator"]["governor"]["low"] >= 1


def test_status_shows_the_governor(tmp_path):
    usage = dryrun.SimUsage.subscription(0.3, 0.3)
    orch, _, _ = dry_run(tmp_path, 6, usage=usage, governed=True, hours=2.0)
    state = read_json(orch.run.root / "improve.json")
    assert governor.status_line(state["governor"]).startswith("governor: k = ")
    text = status.render(orch.run)
    assert "quality:" in text  # (no session left open: the Agents table has no live rows)
