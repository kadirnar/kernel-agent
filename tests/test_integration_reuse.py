"""Re-integrations reuse measurements by content, and ``--integrate-every 0`` (issue #93).

In runs/openbmb--VoxCPM2/20261006-004718 the second integration measured six transforms
alone again (exp 28-33, 22.6 min) whose snapshots were byte-identical to the ones the first
integration had measured (exp 11-16): the systems agent had evaluated them again in a
combination, which snapshots every file under a new name, and the reuse cache matched by
name. CPU only: a simulated run (``dryrun.create_run``), records written through the
evaluation tools' own code and a fake measurement layer (the e2e worker), or the improve
dry run.
"""

import argparse
import asyncio
import statistics
from pathlib import Path

import pytest

from kernel_agent import charts, dryrun, orchestrator
from kernel_agent.agent.tools import record_candidate, record_e2e_result, snapshot
from kernel_agent.config import OptimizeConfig
from kernel_agent.improve import ImproveConfig, Improver
from kernel_agent.integrate import reuse
from kernel_agent.kernels import evaluate
from kernel_agent.workspace import read_json, write_json

BASE = dryrun.BASELINE_MS
#: End-to-end speedup of each item alone, by what it applies (a transform's first line, a
#: kernel's target); a set's speedup is their product.
ALONE = {"cfm": 1.8, "attn": 1.5, "lm v1": 1.25, "lm v2": 1.3, "enc": 1.06}
WIGGLE = (1.0004, 0.9996, 1.0002, 0.9998, 1.0001, 0.9999, 1.0003, 0.9997)


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(orchestrator.toolchain, "setup", dryrun.SimToolchain)
    monkeypatch.setattr(charts, "available", lambda: False)


def make(tmp_path):
    config = OptimizeConfig(model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path)
    return orchestrator.Orchestrator(dryrun.create_run(config), config)


def label(item: str) -> str:
    if not item.startswith("/"):
        return item.partition("=")[0]
    return Path(item).read_text().splitlines()[0].removeprefix("# ")


def parse(args) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    for flag in ("--kernel", "--transform", "--b-kernel", "--b-transform"):
        parser.add_argument(flag, action="append", default=[])
    parser.add_argument("--rounds", type=int, default=8)
    ns, _ = parser.parse_known_args(list(args))
    return ns


def ms(items: list[str]) -> float:
    speedup = 1.0
    for item in items:
        speedup *= ALONE[label(item)]
    return BASE / speedup


def fake_worker(calls: list[tuple[list[str], list[str]]]):
    """The e2e worker: records (A, B) items of every measurement, by label."""

    def worker(run, command, *args):
        assert command == "e2e_ab"
        ns = parse(args)
        a, b = [*ns.kernel, *ns.transform], [*ns.b_kernel, *ns.b_transform]
        calls.append(([label(i) for i in a], [label(i) for i in b]))
        times = [round(ms(a) * w, 3) for w in WIGGLE], [round(ms(b) * w, 3) for w in WIGGLE]
        result = dryrun._e2e_result(statistics.median(times[1]))
        result.update(times_ms=times[1], ab={"mode": "paired", "a_ms": times[0], "b_ms": times[1]})
        return result

    return worker


def write(run, name: str) -> Path:
    path = run.transforms_dir / f"{name.split()[0]}.py"
    path.write_text(f"# {name}\n\ndef apply(workload):\n    pass\n")
    return path


def e2e(run, files: list[Path], kernels: list[str] | None = None, faster: float = 1.0) -> dict:
    """A passing ``evaluate_e2e`` record of ``files`` (+ ``kernels``): every file is
    snapshotted anew, as the tool does (``faster``: a luckier timing)."""
    snaps = [snapshot(run, f) for f in files]
    result = dryrun._e2e_result(ms([*map(str, files), *(kernels or [])]) / faster)
    record, _ = record_e2e_result(run, result, snaps, kernels or [], hypothesis="")
    return record


def kernel(run, target: str = "attn", speedup: float = 2.1) -> str:
    src = run.target(target) / "candidates" / "v1.py"
    src.write_text(f"# {target}\n")
    snap = snapshot(run, src, target)
    result = {"status": "ok", "correct": True, "speedup": speedup, "cases": []}
    record_candidate(run, target, src, snap, result, hypothesis="v1")
    return f"{target}={run.target(target) / 'history' / snap.name}"


def first_integration(tmp_path):
    """cfm, lm v1 and enc evaluated alone, the attn kernel; integrated once."""
    orch = make(tmp_path)
    run = orch.run
    files = {name: write(run, name) for name in ("cfm", "lm v1", "enc")}
    for f in files.values():
        e2e(run, [f])
    kernel(run)
    calls: list[tuple[list[str], list[str]]] = []
    orch.worker = fake_worker(calls)
    asyncio.run(orch.integrate())
    return orch, files, calls


def integrated(run) -> dict:
    return read_json(run.root / "integration.json")


def test_unchanged_content_under_new_snapshot_names_measures_nothing_again(tmp_path):
    orch, files, calls = first_integration(tmp_path)
    run = orch.run
    first = integrated(run)
    assert len(calls) == len(first["history"]) == 7  # 4 alone, then cfm + attn + lm v1 + enc
    assert first["reuse"] == {"reused": 0, "measured": 7}
    assert all(h["reuse_key"] for h in first["history"])

    # the systems agent evaluates the same files together, on top of the kernel: each is
    # snapshotted again under a new name, and that faster record now holds every idea's item
    attn = f"attn={orch._kernel_winners()[0][1]}"
    exp = e2e(run, list(files.values()), [attn])["exp"]
    calls.clear()
    asyncio.run(orch.integrate(reuse=True))
    again = integrated(run)
    alone = [h["items"][0] for h in again["history"] if not h["ab"]["a_items"]]
    before = [h["items"][0] for h in first["history"] if not h["ab"]["a_items"]]
    renamed = [i for i in alone if i.startswith("/")]
    assert len(renamed) == 3 and not set(renamed) & set(before)  # new names, same bytes
    # only the combination the agent measured is new: the seed against the best single item
    assert calls == [(["cfm"], ["attn", "cfm", "lm v1", "enc"])]  # kernels first
    assert again["composite"]["exp"] == exp and again["composite"]["seeded"]
    assert again["reuse"] == {"reused": 4, "measured": 1}
    assert [label(a["item"]) for a in again["accepted"]] == ["cfm", "lm v1", "enc", "attn"]

    calls.clear()  # and once more: everything from the cache
    asyncio.run(orch.integrate(reuse=True))
    assert calls == []
    assert integrated(run)["reuse"] == {"reused": 5, "measured": 0}
    assert integrated(run)["history"] == again["history"]


def test_a_changed_file_measures_only_its_item_again(tmp_path):
    orch, _, calls = first_integration(tmp_path)
    run = orch.run
    lm = write(run, "lm v2")  # the same idea, other content: faster alone
    e2e(run, [lm])
    calls.clear()
    asyncio.run(orch.integrate(reuse=True))
    data = integrated(run)

    alone = [b for a, b in calls if not a]
    assert alone == [["lm v2"]]  # cfm, attn and enc alone come from the cache
    # every other measurement is a step whose A or B holds the changed file
    assert calls and all("lm v2" in [*a, *b] for a, b in calls[1:])
    steps = [{label(i) for i in [*h["ab"]["a_items"], *h["items"]]} for h in data["history"]]
    assert len(calls) == sum("lm v2" in s for s in steps)
    assert data["reuse"] == {"reused": len(steps) - len(calls), "measured": len(calls)}
    assert [label(a["item"]) for a in data["accepted"]] == ["cfm", "attn", "lm v2", "enc"]
    swap = data["history"][-1]  # lm v1, which the last integration accepted, loses
    assert swap["kind"] == "swap" and label(swap["new"]) == "lm v1"
    assert not swap["ab"]["accepted"]


@pytest.mark.parametrize("change", ["schema", "rounds", "baseline"])
def test_a_new_evaluator_schema_or_baseline_measures_everything_again(
    tmp_path, monkeypatch, change
):
    orch, _, calls = first_integration(tmp_path)
    if change == "schema":  # e.g. timing at full clocks (#81): earlier timings are stale
        monkeypatch.setattr(evaluate, "EVALUATOR_SCHEMA", evaluate.EVALUATOR_SCHEMA + 1)
    elif change == "rounds":
        orch.cfg.ab_rounds = 16
    else:  # the baseline the A/B steps ran against (a new analyze)
        monkeypatch.setattr(orch.truth, "baseline_ms", lambda: BASE * 1.01)
    calls.clear()
    asyncio.run(orch.integrate(reuse=True))
    data = integrated(orch.run)
    assert len(calls) == len(data["history"]) == 7
    assert data["reuse"] == {"reused": 0, "measured": 7}


def test_irreversible_items_are_remembered_by_content(tmp_path):
    """An item that cannot be undone in-process is measured in separate processes from the
    start in the next integration, also under a new snapshot name."""
    orch, files, _ = first_integration(tmp_path)
    run = orch.run
    calls: list[str] = []

    def worker(run_dir, command, *args):
        calls.append(command)
        ns = parse(args)
        if command == "e2e_ab":
            enc = [t for t in [*ns.transform, *ns.b_transform] if label(t) == "enc"]
            if enc:  # what the worker reports: the items as given
                return {"status": "irreversible", "passed": False, "irreversible": enc}
            return fake_worker([])(run_dir, command, *args)
        return dryrun._e2e_result(ms([*ns.kernel, *ns.transform]))

    enc = next(h["items"][0] for h in integrated(run)["history"] if label(h["items"][0]) == "enc")
    orch.worker = worker
    orch.cfg.ab_rounds = 4  # every step is measured again
    asyncio.run(orch.integrate(reuse=True))
    assert integrated(run)["irreversible"] == [enc]
    e2e(run, [files["enc"]], faster=1.01)  # the same file, a new snapshot of it, now enc's best
    calls.clear()
    asyncio.run(orch.integrate(reuse=True))
    data = integrated(run)
    new = next(h["items"][0] for h in data["history"] if label(h["items"][0]) == "enc")
    assert new != enc and new in data["irreversible"]  # (the old one is a version to try)
    assert data["reuse"] == {"reused": 7, "measured": 0} and calls == []
    orch.cfg.ab_rounds = 8  # measured again: never in-process for enc
    asyncio.run(orch.integrate(reuse=True))
    with_enc = [h for h in integrated(run)["history"] if "enc" in map(label, h["items"])]
    assert with_enc and all(h["ab"]["mode"] == "separate" for h in with_enc)
    assert calls.count("e2e_ab") == 7 - len(with_enc)


# ------------------------------------------------------------------ content keys


def test_the_content_key_covers_what_an_item_loads(tmp_path):
    orch = make(tmp_path)
    run = orch.run
    history = run.history_dir()
    history.mkdir(parents=True, exist_ok=True)
    (history / "helpers.py").write_text("SCALE = 2\n")
    (history / "fused.cu").write_text("__global__ void k() {}\n")
    body = (
        "import helpers\nimport torch\nfrom kernel_agent.kernels import quant\n"
        "SRC = 'fused.cu'\n\ndef apply(workload):\n    pass\n"
    )
    one, two = history / "001_t_aaaa.py", history / "007_t_aaaa.py"
    one.write_text(body)
    two.write_text(body)
    files = reuse.loaded_files(one, run.root)
    assert set(files) == {
        "",
        str((history / "helpers.py").relative_to(run.root)),
        str((history / "fused.cu").relative_to(run.root)),
        "kernel_agent/kernels/__init__.py",
        "kernel_agent/kernels/quant.py",
    }
    key = reuse.item_key(run, "transform", str(one))
    assert key and key == reuse.item_key(run, "transform", str(two))  # by content, not name
    assert key != reuse.item_key(run, "kernel", f"attn={one}")
    (history / "helpers.py").write_text("SCALE = 3\n")  # an imported helper changed
    assert reuse.item_key(run, "transform", str(one)) != key
    key = reuse.item_key(run, "transform", str(one))
    (history / "fused.cu").write_text("__global__ void k2() {}\n")  # a named source changed
    assert reuse.item_key(run, "transform", str(one)) != key
    assert reuse.item_key(run, "transform", str(history / "missing.py")) is None

    attn = kernel(run)
    key = reuse.item_key(run, "kernel", attn)
    spec_file = run.target("attn") / "spec.json"
    spec = read_json(spec_file)
    write_json(spec_file, {**spec, "notes": "anything else"})
    assert reuse.item_key(run, "kernel", attn) == key
    write_json(spec_file, {**spec, "qualname_regex": r"layers\.0\."})  # applied elsewhere
    assert reuse.item_key(run, "kernel", attn) != key

    keys = reuse.Keys(run, {"evaluator_schema": 2})
    t = ("transform", str(two))
    assert keys.step([], [t]) != keys.step([t], [t])
    assert keys.step([], [t, ("kernel", attn)]) != keys.step([], [("kernel", attn), t])
    assert keys.step([], [("transform", str(history / "missing.py"))]) is None
    assert keys.arg(attn) == keys.item(("kernel", attn))


# ------------------------------------------------------------------ the improve loop


def loop(tmp_path, **icfg):
    config = OptimizeConfig(model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path)
    orch = orchestrator.Orchestrator(dryrun.create_run(config), config)
    world = dryrun.World(orch)
    improver = Improver(orch, ImproveConfig(**icfg), require_capture=False, live_charts=False)
    with world.installed():
        asyncio.run(improver.improve())
    return orch, read_json(orch.run.root / "improve.json")


def test_integrate_every_0_integrates_only_at_the_end(tmp_path):
    _, state = loop(tmp_path / "final", max_slices=4, integrate_every=0)
    assert [i["why"] for i in state["integrations"]] == ["final integration"]
    assert state["config"]["integrate_every"] == 0
    _, state = loop(tmp_path / "every", max_slices=4, integrate_every=1)
    assert len(state["integrations"]) > 1


def test_re_integrations_of_the_improve_loop_report_what_they_reused(tmp_path):
    orch, state = loop(tmp_path, max_slices=6, integrate_every=1)
    done = state["integrations"]
    assert len(done) >= 3 and done[0]["reused"] == 0
    assert sum(i["reused"] for i in done[1:]) > 0  # the unchanged items alone, at least
    data = integrated(orch.run)
    assert data["reuse"]["reused"] == done[-1]["reused"]
    assert sum(data["reuse"].values()) == len(data["history"])


def test_a_timeout_is_measured_again_not_reused(tmp_path):
    orch = make(tmp_path)
    run = orch.run
    for name in ("cfm", "enc"):
        e2e(run, [write(run, name)])
    calls: list[tuple[list[str], list[str]]] = []
    ok = fake_worker(calls)

    def flaky(run, command, *args):  # enc alone times out once (another process held the GPU)
        result = ok(run, command, *args)
        if calls[-1] == ([], ["enc"]) and len(calls) <= 2:
            return {"status": "timeout", "passed": False, "reason": "worker timed out"}
        return result

    orch.worker = flaky
    asyncio.run(orch.integrate())
    first = [h for h in integrated(run)["history"] if [label(i) for i in h["items"]] == ["enc"]]
    assert first and first[0]["status"] == "timeout"

    calls.clear()
    orch.worker = ok
    asyncio.run(orch.integrate(reuse=True))
    assert ([], ["enc"]) in calls  # measured again; cfm alone comes from the cache
    assert ([], ["cfm"]) not in calls
