"""The run's blackboard (``board.py``, issue #187): ``board.jsonl`` under concurrent writers,
cursors, subscriptions, caps, dedup, the piggyback on evaluation results, the ``post_note``
/ ``read_board`` tools, kernel-agent's own posts and what the digests, the research
evidence, the librarian and the report read of it.

CPU only, no Claude and no GPU: evaluations are faked; the improve loop is ``dryrun``'s.
"""

import asyncio
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from kernel_agent import board, charts, dryrun, improve, ledger, library, orchestrator, research
from kernel_agent.agent import prompts
from kernel_agent.agent import tools as tools_mod
from kernel_agent.agent.runner import AgentResult
from kernel_agent.agent.tools import SessionBinding, record_candidate, snapshot
from kernel_agent.budget import Budget
from kernel_agent.config import OptimizeConfig
from kernel_agent.scheduler import KERNEL, NATIVE, SYSTEMS, Arm, Policy
from kernel_agent.workspace import RunDir, write_json

SRC = Path(board.__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    """No real toolchain probing and no charts."""
    monkeypatch.setattr(orchestrator.toolchain, "setup", dryrun.SimToolchain)
    monkeypatch.setattr(charts, "available", lambda: False)


def make_run(tmp_path: Path) -> RunDir:
    """A run with kernel targets ``attn`` and ``mlp`` (``mlp`` in FP8 weights), ``norm``
    sharing ``attn``'s module class (another instance group)."""
    run = RunDir.create(tmp_path, "org/m")
    specs = {
        "attn": {"module_class": "Qwen3Attention"},
        "mlp": {"module_class": "Qwen3MLP", "precision": "fp8_weights"},
        "norm": {"module_class": "Qwen3Attention"},
    }
    for tid, spec in specs.items():
        write_json(run.target(tid) / "spec.json", {"id": tid, "backends": ["triton"], **spec})
    return run


@pytest.fixture
def run(tmp_path):
    out = make_run(tmp_path)
    board.open_board(out)
    yield out
    board.close(out)


def reader(run: RunDir, label: str, role: str, arm: str | None = None) -> board.Reader:
    return board.Reader.of(run, label, role, arm)


# ------------------------------------------------------------------ the store


def test_concurrent_writers_never_share_an_id_or_tear_a_line(tmp_path):
    """Threads posting through one board and through two boards of the same file (each its
    own open file description: the ``flock`` excludes them as it excludes processes)."""
    path = tmp_path / board.FILE
    a, b = board.Board(path), board.Board(path)

    def write(found: board.Board, who: str) -> None:
        for i in range(40):
            found.post(board.COORDINATOR, board.INSIGHT, f"{who} {i} " + "x" * 300)

    threads = [threading.Thread(target=write, args=(x, f"t{i}")) for i, x in enumerate([a, b] * 3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = path.read_text().splitlines()
    entries = [json.loads(line) for line in lines]
    assert len(entries) == 240 and [e["id"] for e in entries] == list(range(1, 241))
    assert len({e["text"] for e in entries}) == 240
    assert [e["id"] for e in a.entries()] == [e["id"] for e in b.entries()] == list(range(1, 241))


def test_another_process_appends_beside_this_one(tmp_path):
    path = tmp_path / board.FILE
    here = board.Board(path)
    code = (
        "import sys\nfrom kernel_agent import board\n"
        "b = board.Board(__import__('pathlib').Path(sys.argv[1]))\n"
        "for i in range(60):\n    b.post('coordinator', 'insight', f'child {i}')\n"
    )
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    child = subprocess.Popen([sys.executable, "-c", code, str(path)], env=env)
    for i in range(60):
        here.post(board.COORDINATOR, board.INSIGHT, f"parent {i}")
    assert child.wait(timeout=120) == 0
    entries = here.entries()
    assert [e["id"] for e in entries] == list(range(1, 121))
    assert sum(e["text"].startswith("child") for e in entries) == 60


def test_a_torn_last_line_is_skipped_and_the_ids_go_on(tmp_path):
    path = tmp_path / board.FILE
    first = board.Board(path).post(board.COORDINATOR, board.INSIGHT, "one")
    with path.open("a") as fh:
        fh.write('{"id": 2, "kind": "insi')  # a writer that died mid-line
    found = board.Board(path)
    assert [e["id"] for e in found.entries()] == [first["id"]]  # the partial line waits
    second = found.post(board.COORDINATOR, board.INSIGHT, "two")
    assert second["id"] == 2 and [e["text"] for e in board.Board(path).entries()] == ["one", "two"]


def test_dedup(run):
    found = board.active(run)
    a = reader(run, "kernel-attn#1", "kernel", "attn")
    note = {"kind": "trap", "target": "attn", "text": "Split-K GEMV loses at M=1: 0.8x"}
    assert found.note(run, a, note)["posted"] == 1
    again = {**note, "text": "split-k   GEMV loses at m=1:  0.8x"}  # case and spaces only
    with pytest.raises(board.Refused, match="already on the board as #1"):
        found.note(run, reader(run, "kernel-attn#2", "kernel", "attn"), again)
    assert found.note(run, a, {**note, "target": "mlp"})["posted"] == 2  # another target
    assert found.note(run, a, {**note, "kind": "insight"})["posted"] == 3  # another kind
    one = found.post(board.COORDINATOR, board.ROUND, "round 2", tags={"n": 2})
    assert found.post(board.COORDINATOR, board.ROUND, "round 2", tags={"n": 2}) == one
    assert len(found.entries()) == 4


# ------------------------------------------------------------------ caps


def test_caps(run, monkeypatch):
    found = board.active(run)
    a = reader(run, "kernel-attn#1", "kernel", "attn")
    long = found.note(run, a, {"kind": "insight", "text": "y" * 5000, "refs": ["r"] * 20})
    entry = found.entries()[-1]
    assert long["cut"] and len(entry["text"]) == board.TEXT_CHARS
    assert len(entry["refs"]) == board.REFS
    for i in range(board.POSTS_PER_SESSION - 1):
        found.note(run, a, {"kind": "insight", "text": f"fact {i}"})
    with pytest.raises(board.Refused, match="the most a session may"):
        found.note(run, a, {"kind": "insight", "text": "one more"})
    b = reader(run, "kernel-mlp#2", "kernel", "mlp")  # the cap is per session
    assert found.note(run, b, {"kind": "insight", "text": "one more"})["notes_left"] == 5
    monkeypatch.setattr(board, "MAX_ENTRIES", len(found.entries()))
    with pytest.raises(board.Refused, match="full"):
        found.note(run, b, {"kind": "insight", "text": "and another"})
    found.post(board.COORDINATOR, board.ROUND, "kernel-agent's own posts go on")


def test_note_checks(run):
    found = board.active(run)
    a = reader(run, "kernel-attn#1", "kernel", "attn")
    bad = [
        ({"kind": "progress", "text": "x"}, "kind is one of"),
        ({"kind": "insight", "text": "  "}, "text is required"),
        ({"kind": "insight", "text": "x", "target": "nope"}, "unknown target"),
        ({"kind": "insight", "text": "x", "to": "everyone"}, "to is"),
        ({"kind": "insight", "text": "x", "reply_to": 99}, "no entry #99"),
        ({"kind": "claim", "text": "I will change things"}, "a claim names"),
        ({"kind": "winner", "text": "mine is best", "refs": ["exp:3"]}, "kept result"),
    ]
    for args, why in bad:
        with pytest.raises(board.Refused, match=why):
            found.note(run, a, args)
    with pytest.raises(board.Refused, match="the board is for"):
        found.note(run, reader(run, "planner", "planner"), {"kind": "insight", "text": "x"})
    q = found.note(run, a, {"kind": "question", "text": "contiguous KV?", "to": "systems"})
    entry = found.entries()[-1]
    assert entry["to"] == "arm:systems" and entry["tags"]["arm"] == "attn"
    found.note(
        run,
        reader(run, "systems#2", "systems", "systems"),
        {"kind": "insight", "text": "yes, static cache", "reply_to": q["posted"]},
    )
    assert found.entries()[-1]["reply_to"] == q["posted"]
    claim = found.note(run, a, {"kind": "claim", "text": "fusing", "target": "Qwen3Attention"})
    tags = found.entries()[-1]["tags"]
    assert claim["posted"] and tags["module_class"] == "Qwen3Attention"
    assert tags["targets"] == ["attn", "norm"]  # every target of that class


def test_a_winner_note_needs_a_kept_result(run):
    found = board.active(run)
    src = run.target("attn") / "candidates" / "v1.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("def build(r):\n    return r\n")
    snap = snapshot(run, src, "attn")
    _, row = record_candidate(
        run,
        "attn",
        src,
        snap,
        {"status": "ok", "correct": True, "speedup": 1.4},
        hypothesis="fused qk",
    )
    assert row["status"] == ledger.KEEP
    a = reader(run, "kernel-attn#1", "kernel", "attn")
    out = found.note(
        run,
        a,
        {
            "kind": "winner",
            "text": "fused q/k norm wins: 1 launch",
            "refs": [f"history/{snap.name}"],
        },
    )
    entry = found.entries()[-1]
    assert out["posted"] == entry["id"] and entry["tags"]["exp"] == row["exp"]
    assert entry["tags"]["targets"] == ["attn"] and f"exp:{row['exp']}" in entry["refs"]


# ------------------------------------------------------------------ subscriptions


def test_subscriptions_by_role(run):
    found = board.active(run)
    post = found.post
    general = post("kernel-mlp#1", board.INSIGHT, "bf16x8 loads pay on this GPU")
    about_attn = post(
        "kernel-norm#2",
        board.TRAP,
        "warp-per-row loses",
        tags={"targets": ["norm"], "module_class": "Qwen3Attention"},
    )
    fp8 = post(
        "kernel-mlp#1",
        board.INSIGHT,
        "fp8 dot needs k%32",
        tags={"targets": ["mlp"], "module_class": "Qwen3MLP", "precision": "fp8_weights"},
    )
    winner = post(
        board.COORDINATOR,
        board.WINNER,
        "`mlp`: new best",
        tags={"targets": ["mlp"], "module_class": "Qwen3MLP", "session": "kernel-mlp#1"},
    )
    claim = post(
        board.COORDINATOR,
        board.CLAIM,
        "kernel-mlp#3 works on Qwen3MLP",
        tags={
            "targets": ["mlp"],
            "module_class": "Qwen3MLP",
            "arm": "mlp",
            "session": "kernel-mlp#3",
        },
    )
    release = post(
        board.COORDINATOR,
        board.RELEASE,
        "kernel-mlp#3 ended",
        tags={"targets": ["mlp"], "arm": "mlp", "session": "kernel-mlp#3"},
    )
    to_native = post("systems#4", board.QUESTION, "need a fused norm?", to="arm:native")
    integ = post(board.COORDINATOR, board.INTEGRATION, "integration 1: 1.5x")

    def ids(label, role, arm=None):
        r = reader(run, label, role, arm)
        return {e["id"] for e in found.entries() if board.relevant(e, r)}

    attn = ids("kernel-attn#5", "kernel", "attn")
    assert attn == {general["id"], about_attn["id"], integ["id"]}  # same class, general, all
    mlp = ids("kernel-mlp#6", "kernel", "mlp")
    assert mlp == {general["id"], fp8["id"], winner["id"], claim["id"], integ["id"]}
    assert ids("kernel-mlp#1", "kernel", "mlp") == {claim["id"], integ["id"]}  # not its own
    systems = ids("systems#7", SYSTEMS, SYSTEMS)
    assert systems == {general["id"], winner["id"], claim["id"], integ["id"]}
    native = ids("native#8", NATIVE, NATIVE)
    assert native == {
        general["id"],
        about_attn["id"],
        fp8["id"],
        winner["id"],
        to_native["id"],
        integ["id"],
    }
    assert release["id"] not in attn | mlp | systems | native  # it only closes a claim
    assert board.open_claims(found.entries()) == [
        e for e in found.entries() if e["id"] not in (claim["id"], release["id"])
    ]


# ------------------------------------------------------------------ cursors and piggyback


def test_cursors_and_piggyback(run):
    found = board.active(run)
    sys_reader = reader(run, "systems#1", SYSTEMS, SYSTEMS)
    found.post(board.COORDINATOR, board.WINNER, "`attn`: new best 1.2x", tags={"targets": ["attn"]})
    assert found.section(sys_reader)[1].startswith("## Board")  # sets its cursor
    assert found.piggyback(sys_reader) == {}  # what its digest showed is not news
    found.join("systems#1")  # its tools are built: a cursor its digest set stays
    a = reader(run, "kernel-attn#2", "kernel", "attn")
    found.note(
        run, a, {"kind": "insight", "text": "fusing RoPE into q/k norm: one launch\ndetails: ..."}
    )
    news = found.piggyback(sys_reader)["board"]
    assert news["new"] == 1 and news["items"] == [
        "#2 insight (kernel-attn#2): fusing RoPE into q/k norm: one launch"
    ]
    assert news["read"] == "read_board(since=1) for the full text"
    assert found.piggyback(sys_reader) == {}  # once
    assert found.piggyback(a) == {}  # never its own
    late = reader(run, "native#3", NATIVE, NATIVE)
    found.join(late.label)  # a session without a digest section: from its start on
    assert found.piggyback(late) == {}


def test_piggyback_size(run):
    found = board.active(run)
    for i in range(30):
        found.post(
            board.COORDINATOR,
            board.WINNER,
            f"`attn`: new best {i} " + "z" * 900 + "\nsecond line",
            tags={"targets": ["attn"]},
        )
    news = found.piggyback(reader(run, "systems#1", SYSTEMS, SYSTEMS))  # no cursor: none
    assert news == {}
    found.cursors["systems#1"] = 0
    news = found.piggyback(reader(run, "systems#1", SYSTEMS, SYSTEMS))["board"]
    assert news["new"] == 30 and len(news["items"]) == board.PIGGYBACK_ITEMS
    assert all(len(i) <= board.LINE_CHARS and "second line" not in i for i in news["items"])
    assert news["items"][-1].startswith("#30 winner")  # the newest
    assert len(json.dumps(news)) < 1200


def test_read_board(run):
    found = board.active(run)
    for i in range(80):
        found.post(
            board.COORDINATOR,
            board.WINNER,
            f"winner {i} " + "w" * 1400,
            tags={"targets": ["mlp" if i % 2 else "attn"]},
        )
    s = reader(run, "systems#1", SYSTEMS, SYSTEMS)
    out = found.read(s, {"limit": 100})
    assert out["matching"] == 80 and 1 <= out["shown"] <= board.READ_MAX
    assert len(json.dumps(out["entries"])) <= board.READ_CHARS
    assert out["entries"][-1]["id"] == 80 and out["entries"][-1]["text"].endswith("w")
    assert found.cursors["systems#1"] == 80  # the plain view: read
    attn = found.read(s, {"target": "attn", "since": 70, "kinds": "winner"})
    assert [e["id"] for e in attn["entries"]] == [71, 73, 75, 77, 79]
    with pytest.raises(board.Refused, match="unknown kinds"):
        found.read(s, {"kinds": ["chatter"]})


def test_query_by_reader_id_time_kind_and_target(run, monkeypatch):
    found = board.active(run)
    for i, (t, kind, target) in enumerate(
        [(1000.0, board.TRAP, "attn"), (2000.0, board.INSIGHT, "mlp"), (3000.0, board.TRAP, "mlp")]
    ):
        monkeypatch.setattr(ledger, "clock", lambda t=t: t)
        found.post(f"kernel-{target}#{i}", kind, f"note {i}", tags={"targets": [target]})
    assert [e["id"] for e in found.query(after=2000.0)] == [2, 3]
    assert [e["id"] for e in found.query(since=1, kinds=[board.TRAP])] == [3]
    assert [e["id"] for e in found.query(target="mlp")] == [2, 3]
    assert [e["id"] for e in found.query(reader(run, "kernel-attn#9", "kernel", "attn"))] == [1]
    monkeypatch.setattr(ledger, "clock", lambda: 3100.0)
    s = reader(run, "native#9", NATIVE, NATIVE)
    found.cursors[s.label] = 0
    assert [e["id"] for e in found.read(s, {"minutes": 10})["entries"]] == [3]
    assert found.cursors[s.label] == 0  # a window is not everything new: not read
    assert [e["id"] for e in found.read(s, {})["entries"]] == [1, 2, 3]
    assert found.cursors[s.label] == 3


# ------------------------------------------------------------------ the tools


def test_tools_post_read_and_piggyback_on_evaluation_results(run, monkeypatch):
    """A kernel session's new best reaches the systems session's next ``evaluate_e2e``
    result; the board's tools exist only for the roles that have the board."""
    tdir = run.target("attn")
    (tdir / "candidates").mkdir(parents=True)
    (tdir / "capture.pt").write_bytes(b"")
    (tdir / "candidates" / "v1.py").write_text("def build(r): ...\n")
    monkeypatch.setattr(
        tools_mod,
        "run_evaluation",
        lambda *a, **k: {"status": "ok", "correct": True, "speedup": 1.5},
    )
    monkeypatch.setattr(
        tools_mod,
        "call_worker",
        lambda *a, **k: {"passed": True, "speedup": 1.0, "median_ms": 10.0, "baseline_ms": 10.0},
    )
    monkeypatch.setattr(tools_mod, "create_sdk_mcp_server", lambda n, version, tools: tools)
    monkeypatch.setattr(tools_mod, "refresh", lambda *a: None)

    def server(label, role, **kw):
        bound = SessionBinding(label=label, role=role, **kw)
        return {t.name: t for t in tools_mod.build_server(run, Budget(run), binding=bound)}

    def call(tools, name, **args):
        out = asyncio.run(tools[name].handler(args))
        return json.loads(out["content"][0]["text"])

    kernel = server("kernel-attn#1", "kernel", target_id="attn")
    systems = server("systems#2", "systems")
    assert {"post_note", "read_board"} <= set(kernel) and "post_note" in systems
    assert "post_note" not in server("planner", "planner")

    out = call(
        kernel,
        "evaluate_candidate",
        target_id="attn",
        candidate="candidates/v1.py",
        hypothesis="fused qk norm",
    )
    assert out["ledger"]["status"] == ledger.KEEP and "board" not in out  # its own winner
    posted = call(kernel, "post_note", kind="trap", text="tl.dot on fp32 acc spills", target="attn")
    assert posted["status"] == "posted" and posted["notes_left"] == board.POSTS_PER_SESSION - 1
    assert call(kernel, "post_note", kind="nonsense", text="x")["status"] == "refused"
    e2e = call(systems, "evaluate_e2e", transforms=[], hypothesis="baseline")
    items = e2e["board"]["items"]
    assert e2e["board"]["new"] == 1 and items[0].startswith("#1 winner [attn] (coordinator)")
    assert "1.500x module speedup" in items[0]  # not the trap: about another target
    full = call(systems, "read_board", since=0, all=True)
    assert [e["kind"] for e in full["entries"]] == ["winner", "trap"]
    assert "evaluate_e2e kernels" in full["entries"][0]["text"]  # the winner says how to use it
    board.close(run)
    assert "post_note" not in server("kernel-attn#3", "kernel", target_id="attn")  # no board
    assert "board" not in call(systems, "evaluate_e2e", transforms=[], hypothesis="again")


# ------------------------------------------------------------------ kernel-agent's posts


def test_kernel_agent_posts_are_no_ops_without_a_board(tmp_path):
    run = make_run(tmp_path)
    src = run.target("attn") / "candidates" / "v1.py"
    src.parent.mkdir(parents=True)
    src.write_text("def build(r):\n    return r\n")
    record_candidate(
        run,
        "attn",
        src,
        snapshot(run, src, "attn"),
        {"status": "ok", "correct": True, "speedup": 1.4},
        hypothesis="h",
    )
    board.integration(run, {"n": 1, "speedup": 1.2, "accepted": ["attn"]})
    board.round_started(run, 2, 10.0, [], ["attn"])
    assert not board.claim(run, "kernel-attn#1", "kernel", "attn", [("Qwen3Attention", None)])
    assert not (run.root / board.FILE).exists() and board.load(run) == []
    assert board.report_lines(run) == [] and board.history_lines(run, "attn") == []


def test_claims_release_and_winners_built_on(run):
    assert board.claim(run, "kernel-attn#1", "kernel", "attn", [("Qwen3Attention", "layers.0")])
    assert not board.claim(run, "systems#2", "systems", "systems", [])  # nothing to claim
    board.release(run, "kernel-attn#1", "attn", [("Qwen3Attention", "layers.0")])
    entries = board.load(run)
    assert [e["kind"] for e in entries] == ["claim", "release"]
    assert "works on `Qwen3Attention` (`layers.0`)" in entries[0]["text"]
    src = run.target("attn") / "candidates" / "v1.py"
    src.parent.mkdir(parents=True)
    src.write_text("def build(r):\n    return r\n")
    snap = snapshot(run, src, "attn")
    record_candidate(
        run,
        "attn",
        src,
        snap,
        {"status": "ok", "correct": True, "speedup": 1.4},
        hypothesis="h",
        session="kernel-attn#1",
    )
    src.write_text("def build(r):\n    return r  # v2\n")
    snap2 = snapshot(run, src, "attn")
    record_candidate(
        run,
        "attn",
        src,
        snap2,
        {"status": "ok", "correct": True, "speedup": 1.3},
        hypothesis="h2",
        parent=f"history/{snap.name}",
        session="kernel-attn#3",
    )
    line = board.report_lines(run)[0]
    assert "1 of 1 posted winners built on by another session" in line
    assert "kernel-agent's 3 (claim 1, release 1, winner 1)" in line


# ------------------------------------------------------------------ what reads the board


def test_digest_research_evidence_and_librarian_read_the_board(run):
    found = board.active(run)
    found.post(
        "kernel-norm#1",
        board.TRAP,
        "warp-per-row RMSNorm loses at hidden 1024",
        tags={"targets": ["norm"], "module_class": "Qwen3Attention"},
    )
    found.post(
        "kernel-mlp#2",
        board.INSIGHT,
        "FP8 weights pay for M=1 GEMVs: 1.7x",
        tags={"targets": ["mlp"], "module_class": "Qwen3MLP"},
    )
    found.post(
        board.COORDINATOR,
        board.CLAIM,
        "kernel-mlp#3 works on Qwen3MLP",
        tags={"targets": ["mlp"], "arm": "mlp", "session": "kernel-mlp#3"},
    )
    arm = Arm("attn", KERNEL, 10.0, module_class="Qwen3Attention")
    digest = improve.kernel_digest(run, arm, 3, 4, Policy(), label="kernel-attn#4")
    assert "## Board" in digest and "#1 trap [norm] (kernel-norm#1)" in digest
    assert "FP8 weights" not in digest  # another module class
    assert found.cursors["kernel-attn#4"] == 3
    assert "## Board" not in improve.kernel_digest(run, arm, 3, 4, Policy())  # no label: none
    systems = improve.systems_digest(
        run, Arm(SYSTEMS, SYSTEMS, 10.0), 4, 4, Policy(), label="systems#5"
    )
    assert "#3 claim [mlp]" in systems  # open claims for the systems agent
    evidence = research.evidence(run, "attn", "plateau")
    assert "## Board: what the sessions concluded" in evidence and "warp-per-row" in evidence
    assert "claim" not in evidence.split("## Board", 1)[1]
    prompt = library.librarian_prompt(run, {"triton": "backend"})
    assert "# Board: the sessions' own conclusions" in prompt and "FP8 weights pay" in prompt


def test_board_on_with_one_agent(tmp_path):
    """``--board on`` with the sequential loop: the later sessions' digests carry the earlier
    ones' winners, and the board closes with the loop."""
    config = OptimizeConfig(model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path, dossier=False)
    orch = orchestrator.Orchestrator(dryrun.create_run(config), config)
    world = dryrun.World(orch)
    icfg = improve.ImproveConfig(max_slices=6, board="on")
    imp = improve.Improver(orch, icfg, require_capture=False, live_charts=False)
    with world.installed():
        asyncio.run(imp.improve())
    entries = board.load(orch.run)
    assert board.active(orch.run) is None and entries
    assert all(e["kind"] != board.CLAIM for e in entries)  # claims: concurrent sessions only
    winners = [e["id"] for e in entries if e["kind"] == board.WINNER]
    assert any(f"#{w} winner" in s["prompt"] for s in world.sessions for w in winners)
    assert all("# Board (post_note" in s["system"] for s in world.sessions[1:])


def test_board_note_in_the_stable_prefix():
    kernel = prompts.board_note("kernel-attn")
    assert kernel == prompts.board_note("kernel-mlp-w2")  # one cached prefix per role
    assert "# Board (post_note / read_board)" in kernel and "never scored" in kernel
    assert f"at most {board.POSTS_PER_SESSION} notes" in kernel
    assert "live" in prompts.board_note("native") and "claim" in prompts.board_note("systems")
    assert prompts.board_note("planner") == prompts.board_note("dossier-attn") == ""


def test_agent_sessions_get_the_board_tools_and_prompt(tmp_path):
    config = OptimizeConfig(model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path, dossier=False)
    orch = orchestrator.Orchestrator(dryrun.create_run(config), config)
    seen: list[dict] = []

    async def fake(name, **kwargs):
        seen.append(kwargs)
        return AgentResult(name=name)

    orch.agent_runner = fake

    def session(name, label, **binding):
        asyncio.run(
            orch._agent(
                name,
                label,
                prompt="p",
                system_append="s",
                cwd=tmp_path,
                binding=SessionBinding(**binding),
            )
        )
        return seen[-1]

    plain = session("kernel-attn", "kernel-attn#1", role="kernel", target_id="attn")
    assert not any(t.endswith("post_note") for t in plain["mcp_tools"])
    assert "# Board" not in plain["system_append"]
    with board.opened(orch.run):
        kernel = session("kernel-attn", "kernel-attn#2", role="kernel", target_id="attn")
        planner = session("planner", "planner")
    assert {"mcp__ka__post_note", "mcp__ka__read_board"} <= set(kernel["mcp_tools"])
    assert "# Board (post_note / read_board)" in kernel["system_append"]
    assert "# Board" not in planner["system_append"]
