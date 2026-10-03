import pytest

from kernel_agent.hub import Modality, classify, parse_model_ref


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
