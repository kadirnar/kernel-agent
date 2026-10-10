import pytest

from kernel_agent.hub import (
    Modality,
    ModelCard,
    classify,
    config_architectures,
    detect_family,
    parse_model_ref,
)


@pytest.mark.parametrize(
    ("ref", "repo", "rev"),
    [
        ("https://huggingface.co/Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B", None),
        ("huggingface.co/openai/whisper-large-v3/tree/main", "openai/whisper-large-v3", "main"),
        (
            "https://hf.co/hexgrad/Kokoro-82M/blob/abc123/config.json",
            "hexgrad/Kokoro-82M",
            "abc123",
        ),
        ("black-forest-labs/FLUX.1-schnell", "black-forest-labs/FLUX.1-schnell", None),
        ("org/name@v1.0", "org/name", "v1.0"),
    ],
)
def test_parse_model_ref(ref, repo, rev):
    assert parse_model_ref(ref) == (repo, rev)


@pytest.mark.parametrize(
    "ref", ["https://huggingface.co/datasets/foo/bar", "not a model", "https://github.com/a/b"]
)
def test_parse_model_ref_rejects(ref):
    with pytest.raises(ValueError):
        parse_model_ref(ref)


def test_classify():
    assert classify("text-generation", "transformers", [], [], []) is Modality.LLM
    assert classify(None, None, [], [], ["model_index.json"]) is Modality.DIFFUSION
    assert classify(None, "diffusers", [], [], []) is Modality.DIFFUSION
    assert classify(None, None, ["WhisperForConditionalGeneration"], [], []) is Modality.STT
    assert classify(None, None, ["VitsModel"], [], []) is Modality.TTS
    assert classify(None, None, ["LlamaForCausalLM"], [], []) is Modality.LLM
    assert classify(None, None, ["Mystery"], ["text-to-speech"], []) is Modality.TTS
    assert classify(None, None, ["Mystery"], [], []) is Modality.UNKNOWN


def test_config_architectures():
    assert config_architectures({"architecture": "voxcpm2", "patch_size": 4}) == ["voxcpm2"]
    assert config_architectures({"architectures": ["LlamaForCausalLM"]}) == ["LlamaForCausalLM"]
    assert config_architectures({"_class_name": "FluxPipeline"}) == ["FluxPipeline"]
    assert config_architectures({"architecture": {"not": "a name"}}) == []
    assert config_architectures({}) == []


def test_detect_family():
    assert detect_family(["voxcpm2"]) == "voxcpm"
    assert detect_family(["VoxCPM"]) == "voxcpm"
    assert detect_family([], library="voxcpm") == "voxcpm"
    assert detect_family(["LlamaForCausalLM"], "llama", "transformers") is None
    assert detect_family([]) is None


def test_model_card_family_roundtrip():
    card = ModelCard("openbmb/VoxCPM2", None, Modality.TTS, family="voxcpm")
    assert ModelCard.from_dict(card.to_dict()).family == "voxcpm"
    old = card.to_dict()
    old.pop("family")  # run.json written before the field existed
    assert ModelCard.from_dict(old).family is None


def test_resolve_counts_the_model_at_the_root(monkeypatch, tmp_path):
    """A fine-tuned VoxCPM repo: the model at the root, LoRA adapters and a helper model in
    subfolders. The size is the root's weights; the Hub's parameter count (every file) is
    left out."""
    import json
    from types import SimpleNamespace

    import huggingface_hub

    from kernel_agent import hub

    gb = 1024**3
    sib = [
        ("config.json", 4000),
        ("model.safetensors", 4 * gb),
        ("audiovae.pth", gb // 3),
        *((f"adapters/{d}/lora_weights.safetensors", gb // 7) for d in ("saudi", "najdi")),
        ("adapters/saudi/lora_config.json", 500),
    ]
    info = SimpleNamespace(
        siblings=[SimpleNamespace(rfilename=n, size=z) for n, z in sib],
        safetensors=SimpleNamespace(total=2_700_000_000),
        pipeline_tag="text-to-speech",
        library_name="voxcpm",
        tags=["text-to-speech"],
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"architecture": "voxcpm2"}))
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", lambda self, *a, **k: info)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(config))
    card = hub.resolve("org/voxcpm2-dialects")
    assert card.size_gb == 4.0 and card.params is None and card.family == "voxcpm"

    info.siblings = info.siblings[:3]  # only the root model: as before
    card = hub.resolve("openbmb/VoxCPM2")
    assert card.size_gb == 4.0 and card.params == 2_700_000_000
