"""program.md: parsing, role routing, run-dir copy, mid-run edits, provenance (CPU only)."""

import asyncio
import hashlib

import pytest

from kernel_agent import cli, orchestrator, program
from kernel_agent.agent.runner import AgentResult
from kernel_agent.config import OptimizeConfig
from kernel_agent.hub import Modality, ModelCard
from kernel_agent.program import Program, parse, role_of
from kernel_agent.workspace import RunDir, read_json, read_jsonl, write_json

CARD = {"repo_id": "org/m", "modality": "llm", "architectures": ["X"], "params": 1}

TEXT = """# My program
Intro for humans: never sent.

## all
Measure in absolute ms.

## planner
Pick few targets.

## Kernel
One hypothesis per evaluation.
### Subheading stays
```python
## not a heading inside a fence
```

## kernel, systems
Shared rule.
<!-- a note for humans only -->

## notes for me
ignored text

<!--
## systems
commented-out section
-->
"""


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ------------------------------------------------------------ parsing / routing


def test_parse_sections():
    sections, warnings = parse(TEXT)
    assert set(sections) == {"all", "planner", "kernel", "systems"}
    assert sections["all"] == "Measure in absolute ms."
    assert sections["kernel"].startswith("One hypothesis per evaluation.\n### Subheading stays")
    assert "## not a heading inside a fence" in sections["kernel"]
    assert sections["kernel"].endswith("```\n\nShared rule.")  # repeated section: joined
    assert sections["systems"] == "Shared rule."  # the commented-out section is gone
    assert "Intro" not in str(sections) and "ignored text" not in str(sections)
    assert "note for humans" not in str(sections)
    assert len(warnings) == 1 and "`## notes for me`" in warnings[0]
    assert "its text is ignored" in warnings[0]

    _, warnings = parse("## kernel, research\nx\n##\ny\n")
    assert "unknown name research" in warnings[0] and "only kernel receive it" in warnings[0]
    assert "no name" in warnings[1]
    assert parse("") == ({}, [])
    assert parse("no sections at all\n### deeper only\n") == ({}, [])


def test_role_routing():
    assert role_of("kernel-rms_norm") == "kernel"
    assert role_of("kernel-a-b") == "kernel"
    assert [role_of(n) for n in ("planner", "systems", "harness")] == [
        "planner",
        "systems",
        "harness",
    ]
    assert role_of("research") is None
    sections, _ = parse(TEXT)
    prog = Program(sections, "f" * 64)
    kernel = prog.prompt_note("kernel-t1")
    assert kernel.startswith("\n\n# Program (program.md")
    assert kernel.endswith(
        "## all\nMeasure in absolute ms.\n\n## kernel\nOne hypothesis per evaluation."
        "\n### Subheading stays\n```python\n## not a heading inside a fence\n```\n\nShared rule."
    )
    planner = prog.prompt_note("planner")
    assert "Pick few targets" in planner and "One hypothesis" not in planner
    all_only = "## all\nMeasure in absolute ms."
    assert prog.for_role("harness") == all_only  # no harness section: `all` only
    assert prog.for_role(None) == all_only  # agents without a known role get `all`
    # missing sections / missing file: nothing is appended
    assert Program(*parse("## kernel\nonly kernel\n")[:1]).prompt_note("systems") == ""
    assert Program().prompt_note("kernel-t1") == ""


def test_template_covers_every_role():
    prog = Program.load(program.TEMPLATE)
    assert prog.warnings == () and prog.sha256 == sha(program.TEMPLATE.read_text())
    assert set(prog.sections) == set(program.SECTIONS)
    for agent in ("planner", "kernel-x", "systems", "harness"):
        note = prog.prompt_note(agent)
        assert "one hypothesis per evaluation" in note  # from `## all`
        assert "Edit this file" not in note  # the preamble is for humans
    assert "Amdahl" in prog.prompt_note("planner") and "Amdahl" not in prog.prompt_note("systems")


def test_missing_file(tmp_path):
    assert Program.load(tmp_path / "nope.md") == Program()


# ------------------------------------------------------------ run directory


def make_run(tmp_path, data=None):
    run = RunDir.create(tmp_path, "org/m")
    write_json(run.run_json, data or {"card": CARD, "phases": {}})
    return run


def test_install(tmp_path):
    run = make_run(tmp_path)
    dst = program.install(run)
    assert dst.read_text() == program.TEMPLATE.read_text()
    rec = run.load()["program"]
    assert rec["source"] == program.DEFAULT_SOURCE and rec["versions"] == []
    assert rec["sha256"] == sha(dst.read_text())

    dst.write_text("## all\nedited by a human\n")
    program.install(run)  # resume without --program keeps the edited copy
    assert dst.read_text() == "## all\nedited by a human\n"
    dst.unlink()
    program.install(run)  # ... and a deliberately deleted one
    assert not dst.exists()

    custom = tmp_path / "mine.md"
    custom.write_text("## kernel\nmine\n")
    program.install(run, str(custom))  # an explicit --program replaces the copy
    assert dst.read_text() == "## kernel\nmine\n"
    assert run.load()["program"]["source"] == str(custom.resolve())
    with pytest.raises(SystemExit, match="program file not found"):
        program.install(run, tmp_path / "missing.md")

    old = make_run(tmp_path / "old")  # a run from before program.md gets the template
    assert program.install(old).exists()


def test_for_agent_records_versions(tmp_path):
    run = make_run(tmp_path)
    dst = program.install(run)
    logs: list[str] = []
    first = program.for_agent(run, "planner", logs.append)
    assert first.sha256 == run.load()["program"]["sha256"] and logs == []
    assert program.for_agent(run, "kernel-a", logs.append) == first  # unchanged: one version
    assert (run.root / "logs" / f"program-{first.sha256[:12]}.md").read_text() == dst.read_text()

    dst.write_text("## kernel\nnew rule\n## bogus\nx\n")  # a human edits it mid-run
    second = program.for_agent(run, "kernel-b", logs.append)
    assert "new rule" in second.prompt_note("kernel-b")
    assert second.sha256 == sha(dst.read_text())
    assert (run.root / "logs" / f"program-{second.sha256[:12]}.md").exists()
    versions = run.load()["program"]["versions"]
    assert [(v["agent"], v["sha256"]) for v in versions] == [
        ("planner", first.sha256),
        ("kernel-b", second.sha256),
    ]
    assert "changed" in logs[0] and "`## bogus`" in logs[1]

    dst.unlink()
    assert program.for_agent(run, "systems", logs.append).sha256 is None
    assert "missing" in logs[-1] and run.load()["program"]["versions"][-1]["sha256"] is None


# ------------------------------------------------------------ orchestrator


class FakeToolchain:
    env: dict[str, str] = {}
    backends = {"triton": True}
    gpu: str | None = None

    def summary(self) -> str:
        return "GPU: none (test)"

    def to_dict(self) -> dict:
        return {}


def test_agents_see_mid_run_edits(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator.toolchain, "setup", FakeToolchain)
    run = make_run(tmp_path)
    cfg = OptimizeConfig(model_ref="org/m", runs_dir=tmp_path, backends=["triton"])
    phases = {p: {"done": True} for p in ("analyze", "plan", "capture")}
    write_json(run.run_json, {"card": CARD, "config": cfg.to_dict(), "phases": phases})
    for tid in ("t1", "t2"):
        spec = {"id": tid, "module_class": "M", "backends": ["triton"], "why": "w", "approach": "a"}
        write_json(run.target(tid) / "spec.json", spec)
    custom = tmp_path / "custom.md"
    custom.write_text("## all\nrule one\n## planner\nplanner only\n")
    orch = orchestrator.Orchestrator.resume(run.root, {"program": str(custom)})
    assert orch.cfg.program == str(custom)
    seen: dict[str, str] = {}

    async def fake_run_agent(name, *, system_append, result=None, **kwargs):
        seen[name] = system_append
        if name == "kernel-t1":  # the human edits program.md while kernel-t1 runs
            (run.root / "program.md").write_text("## kernel\nrule two\n")
        return result or AgentResult(name=name)

    monkeypatch.setattr(orchestrator, "run_agent", fake_run_agent)
    asyncio.run(orch.kernels())
    assert "rule one" in seen["kernel-t1"] and "planner only" not in seen["kernel-t1"]
    assert "rule two" in seen["kernel-t2"] and "rule one" not in seen["kernel-t2"]
    costs = read_json(run.root / "costs.json")
    assert costs["kernel-t1"]["program_sha256"] == sha(custom.read_text())
    assert costs["kernel-t2"]["program_sha256"] == sha("## kernel\nrule two\n")
    rec = run.load()["program"]
    assert rec["source"] == str(custom.resolve()) and len(rec["versions"]) == 2
    starts = [e for e in read_jsonl(run.events) if e["event"] == "agent_start"]
    assert [(e["agent"], e["program_sha256"]) for e in starts] == [
        ("kernel-t1", costs["kernel-t1"]["program_sha256"]),
        ("kernel-t2", costs["kernel-t2"]["program_sha256"]),
    ]


def test_create_copies_program(tmp_path, monkeypatch):
    class GpuToolchain(FakeToolchain):
        gpu = "fake"

    monkeypatch.setattr(orchestrator.toolchain, "setup", GpuToolchain)
    resolved: list[str] = []

    def fake_resolve(ref, **kwargs):
        resolved.append(ref)
        return ModelCard(repo_id="org/m", revision=None, modality=Modality.LLM)

    monkeypatch.setattr(orchestrator.hub, "resolve", fake_resolve)
    with pytest.raises(SystemExit, match="program file not found"):
        orchestrator.Orchestrator.create(
            OptimizeConfig(model_ref="org/m", runs_dir=tmp_path, program="missing.md")
        )
    assert resolved == []  # fails before touching the Hub or creating a run dir

    orch = orchestrator.Orchestrator.create(OptimizeConfig(model_ref="org/m", runs_dir=tmp_path))
    assert (orch.run.root / "program.md").read_text() == program.TEMPLATE.read_text()
    custom = tmp_path / "p.md"
    custom.write_text("## all\nx\n")
    cfg = OptimizeConfig(model_ref="org/m", runs_dir=tmp_path / "b", program=str(custom))
    orch = orchestrator.Orchestrator.create(cfg)
    assert (orch.run.root / "program.md").read_text() == "## all\nx\n"
    data = orch.run.load()
    assert data["config"]["program"] == str(custom)
    assert data["program"]["sha256"] == sha("## all\nx\n")


# ------------------------------------------------------------ CLI


def test_cli_program_flag(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(cli, "cmd_optimize", lambda ns: seen.append(cli._config(ns)) or 0)
    assert cli.main(["optimize", "org/m", "--program", "p.md"]) == 0
    assert cli.main(["optimize", "org/m"]) == 0
    assert seen[0].program == "p.md" and seen[1].program is None
    assert OptimizeConfig.from_dict(seen[0].to_dict()).program == "p.md"

    class Resumed(Exception):
        pass

    def fake_resume(cls, root, overrides=None):
        raise Resumed(overrides)

    monkeypatch.setattr(orchestrator.Orchestrator, "resume", classmethod(fake_resume))
    with pytest.raises(Resumed) as exc:
        cli.main(["resume", str(tmp_path), "--program", "q.md"])
    assert exc.value.args[0] == {"program": "q.md"}


def test_cli_program_init(tmp_path, capsys):
    target = tmp_path / "prog.md"
    assert cli.main(["program", "init", str(target)]) == 0
    assert target.read_text() == program.TEMPLATE.read_text()
    assert f"--program {target}" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="exists"):
        cli.main(["program", "init", str(target)])
    target.write_text("old")
    assert cli.main(["program", "init", str(target), "--force"]) == 0
    assert target.read_text() == program.TEMPLATE.read_text()
    assert cli.main(["program", "init", str(tmp_path)]) == 0  # a directory gets program.md
    assert (tmp_path / "program.md").exists()
