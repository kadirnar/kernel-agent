import json
import shutil
import threading
import urllib.error
import urllib.request

import pytest
from synthetic_run import BASELINE_MS, COSTS, GREEDY, SCRIPTS, SINGLES, TRANSFORMS, make_run

from kernel_agent import charts, ledger, watch
from kernel_agent.cli import main
from kernel_agent.workspace import RunDir, write_json

N_ROWS = sum(len(s) for s in SCRIPTS.values()) + len(TRANSFORMS) + len(SINGLES) + len(GREEDY)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return make_run(tmp_path_factory.mktemp("runs"))


@pytest.fixture
def partial(run, tmp_path):
    """A copy of the synthetic run cut after its first 20 rows / 30 events (a run in progress)."""
    copy = RunDir(tmp_path / "run")
    shutil.copytree(run.root, copy.root)
    rows = run.ledger.read_text().splitlines(keepends=True)
    events = run.events.read_text().splitlines(keepends=True)
    copy.ledger.write_text("".join(rows[:21]))
    copy.events.write_text("".join(events[:30]))
    (copy.root / "integration.json").unlink()
    return copy, rows[21:], events[30:]


def test_state(run):
    state = watch.state(run)
    s, rows = state["summary"], state["rows"]
    expected = ledger.summary(run)
    assert len(rows) == N_ROWS == expected["evaluations"]
    assert [r["exp"] for r in rows] == list(range(1, N_ROWS + 1))
    assert rows[0]["ts"] == ledger.epoch(rows[0]["time"])
    assert rows[0]["file"] == f"targets/attn/history/{rows[0]['snapshot']}"
    assert (run.root / rows[0]["file"]).is_file()
    transform = next(r for r in rows if r["backend"] == "transform")
    assert transform["file"].startswith("transforms/history/")
    assert all(r["file"] is None for r in rows if r["backend"] == "integrate")

    header = s["run"]
    assert header["repo_id"] == "Qwen/Qwen3-0.6B" and header["modality"] == "llm"
    assert header["gpu"] == "NVIDIA GeForce RTX 5070 Ti" and header["parallel"] == 2
    assert header["phase"] == "report" and not header["running"]
    assert header["active_agents"] == [] and header["elapsed_s"] == pytest.approx(151.2 * 60)
    assert s["baseline"] == {
        "eager_ms": BASELINE_MS,
        "compiled_ms": 1104.6,
        "times_ms": [1529.8, 1532.4, 1536.1],
    }
    assert s["projected_ms"] == expected["projected_ms"]
    assert s["measured"]["ms"] == 889.0 and s["measured"]["label"] == "integrated"
    assert s["counts"]["keeps"] == expected["keeps"]
    assert s["counts"]["failures"] == expected["failures"]
    assert s["counts"]["e2e"] == expected["e2e_evals"]
    assert [t["id"] for t in s["targets"]] == ["attn", "mlp", "rmsnorm", "rope"]
    for mine, theirs in zip(s["targets"], expected["targets"], strict=True):
        assert mine["best_speedup"] == theirs["best_speedup"]
        assert mine["est_saved_ms"] == theirs["est_saved_ms"]
        assert (mine["evals"], mine["keeps"], mine["failures"]) == (
            theirs["evals"],
            theirs["keeps"],
            theirs["failures"],
        )
    assert [t["color"] for t in s["targets"]] == [0, 1, 2, 3]  # charts.target_shares order
    assert s["costs"]["total_usd"] == pytest.approx(sum(c[0] for c in COSTS.values()))
    burn = s["costs"]["burn"]
    assert len(burn) == len(COSTS) and burn[-1]["usd"] == pytest.approx(s["costs"]["total_usd"])
    assert [b["ts"] for b in burn] == sorted(b["ts"] for b in burn)
    steps = s["integration"]["steps"]
    assert steps == charts.integration_steps(
        json.loads((run.root / "integration.json").read_text())
    )
    assert steps[0]["ms"] == BASELINE_MS and steps[-1]["ms"] == 889.0
    assert [p["phase"] for p in s["phases"]][:3] == ["analyze", "plan", "capture"]

    assert state["events"] == ledger.events(run)[-watch.EVENTS_TAIL :]
    logs = state["logs"]
    assert {entry["agent"] for entry in logs} == set(COSTS)
    assert {entry["kind"] for entry in logs} == {"tool", "text", "result"}
    assert any(e["text"].startswith("evaluate_candidate candidate=candidates/") for e in logs)
    assert "ts" not in logs[0]  # snapshot lines carry no (made-up) time
    json.dumps(state, allow_nan=False)  # valid JSON for the browser


def test_agent_entries():
    tool = {"type": "AssistantMessage", "data": {"content": [
        {"id": "t", "name": "mcp__ka__evaluate_candidate", "input": {"candidate": "c/v1.py"}},
        {"text": "  two\nlines "},
    ]}}  # fmt: skip
    assert watch.agent_entries("kernel-rope", json.dumps(tool)) == [
        {"agent": "kernel-rope", "kind": "tool", "text": "evaluate_candidate candidate=c/v1.py"},
        {"agent": "kernel-rope", "kind": "text", "text": "two lines"},
    ]
    done = {"type": "ResultMessage", "data": {"num_turns": 7, "total_cost_usd": 1.5}}
    assert watch.agent_entries("x", json.dumps(done))[0]["text"] == "finished, 7 turns, $1.50"
    user = {"type": "UserMessage", "data": {"content": [{"text": "tool output"}]}}
    assert watch.agent_entries("x", json.dumps(user)) == []
    assert watch.agent_entries("x", '{"type": "AssistantMessage", "data": {"cont') == []


def test_tail(tmp_path):
    path = tmp_path / "log.jsonl"
    tail = watch.Tail(path)
    assert tail.read() == ([], False)
    path.write_text("a\nb")
    assert tail.read() == (["a"], False)
    assert tail.read() == ([], False)  # "b" is not finished yet
    with path.open("a") as fh:
        fh.write("c\nd\n")
    assert tail.read() == (["bc", "d"], False)
    path.write_text("x\n")  # rewritten shorter: start over
    assert tail.read() == (["x"], True)
    path.write_text("first line\nsecond\nthird\n")
    from_end = watch.Tail(path, from_end=10)
    assert from_end.read() == (["third"], False)  # skips the partial "...second"


def test_poll_deltas(partial):
    run, rest_rows, rest_events = partial
    watcher = watch.Watcher(run)
    state = watcher.snapshot()
    assert len(state["rows"]) == 20
    assert state["summary"]["integration"] is None
    assert watcher.poll() is None

    with run.ledger.open("a") as fh:  # a half-written row is not sent
        fh.write(rest_rows[0][:30])
    assert watcher.poll() is None
    with run.ledger.open("a") as fh:
        fh.write(rest_rows[0][30:] + rest_rows[1])
    kind, delta = watcher.poll()
    assert kind == "delta" and set(delta) == {"rows", "summary"}
    assert [r["exp"] for r in delta["rows"]] == [21, 22]
    assert delta["summary"]["counts"]["evaluations"] == 22

    with run.events.open("a") as fh:
        fh.writelines(rest_events[:3])
    kind, delta = watcher.poll()
    assert set(delta) == {"events", "summary"} and len(delta["events"]) == 3

    costs = json.loads((run.root / "costs.json").read_text())
    costs["kernel-attn"]["usd"] += 10.0
    write_json(run.root / "costs.json", costs)
    kind, delta = watcher.poll()
    assert set(delta) == {"summary"}
    assert delta["summary"]["costs"]["total_usd"] == pytest.approx(
        sum(c[0] for c in COSTS.values()) + 10.0
    )

    with (run.root / "logs" / "agent-kernel-new.jsonl").open("a") as fh:
        fh.write(json.dumps({"type": "AssistantMessage", "data": {"content": [{"text": "hi"}]}}))
        fh.write("\n")
    kind, delta = watcher.poll()
    assert delta["logs"][0]["agent"] == "kernel-new" and "ts" in delta["logs"][0]

    message = watch.sse(kind, delta)
    assert message.startswith(b"event: delta\ndata: ") and message.endswith(b"\n\n")
    assert message.count(b"\n") == 3
    assert json.loads(message.split(b"data: ", 1)[1]) == json.loads(json.dumps(delta))

    run.ledger.write_text("".join(run.ledger.read_text().splitlines(keepends=True)[:6]))
    kind, state = watcher.poll()  # the ledger was rewritten: a fresh snapshot
    assert kind == "state" and len(state["rows"]) == 5


def test_live_run_summary(partial):
    run, _, _ = partial
    events = run.events.read_text().splitlines(keepends=True)
    cut = next(i for i, line in enumerate(events) if '"agent": "kernel-attn"' in line)
    run.events.write_text("".join(events[: cut + 1]))
    with run.events.open("a") as fh:
        fh.write('{"ts": 1, "event": "evalu')  # being written
    s = watch.state(run)["summary"]["run"]
    assert s["running"] and s["phase"] == "kernels"
    assert [a["agent"] for a in s["active_agents"]] == ["kernel-attn"]


def test_tolerates_odd_files(partial):
    run, _, _ = partial
    run.run_json.write_text('["not", "an", "object"]')
    (run.root / "costs.json").write_text('{"planner": {"usd": 0.4')  # being written
    (run.root / "integration.json").write_text('{"baseline_ms": 1000, "history": 3}')
    (run.profile_dir / "profile.json").write_text('{"classes": 7}')
    state = watch.state(run)
    s = state["summary"]
    assert len(state["rows"]) == 20 and s["run"]["repo_id"] == run.root.name
    assert s["costs"]["total_usd"] == 0 and s["integration"] is None
    assert all(t["share"] is None for t in s["targets"])
    json.dumps(state, allow_nan=False)


def test_viewable(partial, tmp_path):
    root = partial[0].root
    notes = root / "targets" / "attn" / "NOTES.md"
    notes.write_text("# notes\n")
    assert watch.viewable(root, "targets/attn/NOTES.md") == notes.resolve()
    assert watch.viewable(root, "targets/./attn/NOTES.md") == notes.resolve()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret\n")
    (root / "link.txt").symlink_to(outside)
    (root / "blob.py").write_bytes(b"\x00\x01binary")
    (root / "big.txt").write_text("x" * 100)
    bad = {
        "../secret.txt": 403,
        "targets/../../secret.txt": 403,
        "../" * 12 + "etc/passwd": 403,
        str(outside): 403,
        "/etc/passwd": 403,
        "link.txt": 403,  # a symlink that leaves the run
        "": 400,
        "a\x00b.txt": 400,
        "..\\secret.txt": 400,
        "missing.md": 404,
        "targets": 404,  # a directory
        "targets/attn/capture.pt": 404,
    }
    (root / "targets" / "attn" / "capture.pt").write_bytes(b"\x00")
    bad["targets/attn/capture.pt"] = 415  # exists, not a text file
    for rel, status in bad.items():
        with pytest.raises(watch.FileError) as err:
            watch.viewable(root, rel)
        assert err.value.status == status, rel
    with pytest.raises(watch.FileError) as err:
        watch.read_text(watch.viewable(root, "blob.py"))
    assert err.value.status == 415
    assert watch.read_text(root / "big.txt", limit=10) == ("x" * 10, 100, True)
    listed = {f["path"] for f in watch.list_files(root)}
    assert {"results.tsv", "events.jsonl", "targets/attn/NOTES.md"} <= listed
    assert "link.txt" not in listed and "targets/attn/capture.pt" not in listed


def test_page_escapes_untrusted_text(partial):
    run, _, _ = partial
    evil = "</script><script>alert(1)</script> & <b>"
    ledger.record_e2e(
        run,
        {"status": "ok", "passed": True, "speedup": 1.0, "median_ms": 1.0, "baseline_ms": 1.0},
        backend="transform",
        snapshot="x",
        hypothesis=evil,
    )
    page = watch.page(run, "NONCE123").decode()
    assert evil not in page and "<script>alert" not in page
    assert page.count('<script nonce="NONCE123">') == 1
    assert "__STATE__" not in page and "__NONCE__" not in page
    blob = page.split('id="initial-state" type="application/json">', 1)[1].split("</script>")[0]
    assert json.loads(blob)["rows"][-1]["hypothesis"] == evil
    script = page.split('<script nonce="NONCE123">', 1)[1]
    assert ".innerHTML" not in script and "insertAdjacentHTML" not in script


def test_page_colours_match_charts():
    page = watch.PAGE.read_text().lower()
    for colour in (
        charts.KEEP_COLOR,
        charts.DISCARD_COLOR,
        charts.FAIL_COLOR,
        charts.PROJECTED_COLOR,
        *charts.TARGET_COLORS,
    ):
        assert colour.lower() in page


def _get(url, host=None):
    request = urllib.request.Request(url, headers={"Host": host} if host else {})
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, dict(response.headers), response.read()


def _sse_message(response):
    event, data = None, None
    while True:
        line = response.readline().decode()
        assert line, "stream closed"
        if line == "\n" and event:
            return event, data
        if line.startswith("event: "):
            event = line[7:].strip()
        elif line.startswith("data: "):
            data = json.loads(line[6:])


def test_server(partial):
    run, rest_rows, _ = partial
    server = watch.WatchServer(run, "127.0.0.1", 0, interval=0.05)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = server.url
    assert url.startswith("http://127.0.0.1:") and not url.endswith(":0/")
    try:
        status, headers, body = _get(url)
        assert status == 200 and headers["Content-Type"].startswith("text/html")
        nonce = headers["Content-Security-Policy"].split("'nonce-")[1].split("'")[0]
        assert f'<script nonce="{nonce}">'.encode() in body and b"Qwen/Qwen3-0.6B" in body

        status, headers, body = _get(url + "api/state")
        assert headers["Content-Type"] == "application/json"
        assert len(json.loads(body)["rows"]) == 20

        files = json.loads(_get(url + "api/files")[2])
        assert "results.tsv" in {f["path"] for f in files}
        status, headers, body = _get(url + "file?path=results.tsv")
        assert body.decode().startswith("exp\ttime\t") and headers["X-Truncated"] == "0"
        assert headers["Content-Type"].startswith("text/plain")
        for path, code in (("../../../../etc/passwd", 403), ("%2e%2e/x.md", 403), ("nope", 404)):
            with pytest.raises(urllib.error.HTTPError) as err:
                _get(url + f"file?path={path}")
            assert err.value.code == code
        with pytest.raises(urllib.error.HTTPError) as err:  # DNS rebinding
            _get(url + "api/state", host="attacker.example:8765")
        assert err.value.code == 403
        assert _get(url + "api/state", host=f"localhost:{server.server_address[1]}")[0] == 200

        with urllib.request.urlopen(url + "events", timeout=10) as stream:
            assert stream.headers["Content-Type"].startswith("text/event-stream")
            event, data = _sse_message(stream)
            assert event == "state" and len(data["rows"]) == 20
            with run.ledger.open("a") as fh:
                fh.writelines(rest_rows[:2])
            event, data = _sse_message(stream)
            assert event == "delta" and [r["exp"] for r in data["rows"]] == [21, 22]
    finally:
        server.shutdown()
        server.server_close()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_cli_watch_needs_a_run_dir(tmp_path):
    with pytest.raises(SystemExit, match="not a run directory"):
        main(["watch", str(tmp_path)])


def test_page_draws_progress_as_lines():
    """The live page shows the improvement as lines (the best so far, a thin line through every
    result, failures as ticks), never as scattered points; legends show line samples."""
    page = watch.PAGE.read_text()
    script = page.split('<script nonce="__NONCE__">', 1)[1]
    for gone in ("diamond(", '"cost-dot"', "dot(xs(", 'key("keep"', 'key("dkeep"', 'key("fail"'):
        assert gone not in script, gone
    for line in ('class: "each-line"', 'key("each"', 'key("failtick"', "failTick("):
        assert line in script, line
    assert "svg .each-line" in page and "svg .fail-tick" in page
