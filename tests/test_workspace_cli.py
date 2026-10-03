import pytest

from kernel_agent.cli import _options, main
from kernel_agent.config import OptimizeConfig
from kernel_agent.workspace import RunDir, append_jsonl, read_jsonl, slug


def test_rundir_layout(tmp_path):
    run = RunDir.create(tmp_path, "org/model")
    assert run.root.parent.name == slug("org/model") == "org--model"
    assert RunDir.latest(tmp_path, "org/model") == run
    assert RunDir.latest(tmp_path, "other/model") is None
    append_jsonl(run.target("t1") / "results.jsonl", {"a": 1})
    append_jsonl(run.target("t1") / "results.jsonl", {"a": 2})
    assert [r["a"] for r in read_jsonl(run.target("t1") / "results.jsonl")] == [1, 2]
    assert run.target_ids() == []  # no spec.json yet


def test_options_parsing():
    opts = _options(
        ["prompt_len=1024", "min_cosine=0.995", "audio=/x.wav", "cpu_offload=true", "g=none"]
    )
    assert opts == {
        "prompt_len": 1024,
        "min_cosine": 0.995,
        "audio": "/x.wav",
        "cpu_offload": True,
        "g": None,
    }
    with pytest.raises(SystemExit):
        _options(["novalue"])


def test_config_roundtrip():
    cfg = OptimizeConfig(model_ref="a/b", hf_token="secret", backends=["triton"])
    data = cfg.to_dict()
    assert "hf_token" not in data
    assert OptimizeConfig.from_dict({**data, "unknown": 1}).backends == ["triton"]


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "optimize" in capsys.readouterr().out
