import json

from kernel_agent.agent import prompts
from kernel_agent.agent.tools import _snapshot, best_for_target, compact
from kernel_agent.workspace import RunDir, append_jsonl

CARD = {
    "repo_id": "org/m",
    "modality": "llm",
    "architectures": ["LlamaForCausalLM"],
    "params": 1,
    "readme_excerpt": "hi",
    "files": ["config.json"],
}


def test_knowledge_and_examples_exist():
    for guide in set(prompts.BACKEND_GUIDES.values()) | {"playbook.md"}:
        assert len(prompts.knowledge(guide)) > 500
    for backend in prompts.BACKEND_GUIDES:
        assert (prompts.EXAMPLES_DIR / f"{backend}_rmsnorm.py").exists()


def test_prompts_render():
    target = {"id": "rms", "module_class": "LlamaRMSNorm", "why": "w", "approach": "a"}
    text = prompts.engineer_prompt(
        target,
        {"qualname": "model.norm", "cases": [{"signature": "a0[1, 1, 8]", "count": 3}]},
        ["cuda", "triton"],
        "/usr/bin/python",
        "GPU: x",
        8,
        {"instances": 2, "calls": 6, "inclusive_ms": 1.0},
    )
    assert "LlamaRMSNorm" in text and "load_inline" in text and "Triton backend" in text
    plan = prompts.planner_prompt(CARD, {"median_ms": 1.0}, "# Profile", ["triton"], 3, "py", "tc")
    assert "up to 3" in plan
    assert "harness.py" in prompts.harness_prompt(CARD, "boom", "py", "tc")
    assert "transforms/" in prompts.systems_prompt(CARD, {"median_ms": 2.0}, "", [], "py", "tc", 4)
    json.dumps(prompts.PLAN_SCHEMA)


def test_snapshot_and_best(tmp_path):
    run = RunDir.create(tmp_path, "org/m")
    tdir = run.target("rms")
    (tdir / "candidates").mkdir(parents=True)
    src = tdir / "candidates" / "triton_v1.py"
    src.write_text("def build(r): return r\n")
    snap1 = _snapshot(src, tdir / "history")
    src.write_text("def build(r): return None\n")
    snap2 = _snapshot(src, tdir / "history")
    assert snap1.name.startswith("001_triton_v1_") and snap2.name.startswith("002_")
    assert snap1.read_text() != snap2.read_text()
    append_jsonl(tdir / "results.jsonl", {"correct": True, "speedup": 1.5, "snapshot": "a"})
    append_jsonl(tdir / "results.jsonl", {"correct": False, "speedup": 9.0, "snapshot": "b"})
    append_jsonl(tdir / "results.jsonl", {"correct": True, "speedup": 2.0, "snapshot": "c"})
    assert best_for_target(run, "rms")["snapshot"] == "c"
    out = compact({"status": "ok", "cases": [{"signature": "s", "ok": True, "speedup": 2.0}], "junk": 1})
    assert "junk" not in out and out["cases"][0]["speedup"] == 2.0
