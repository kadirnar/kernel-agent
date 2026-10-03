"""Resolve a Hugging Face URL into a model description and a modality."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Modality(StrEnum):
    LLM = "llm"
    STT = "stt"
    TTS = "tts"
    DIFFUSION = "diffusion"
    UNKNOWN = "unknown"


_PIPELINE_TAGS: dict[str, Modality] = {
    "text-generation": Modality.LLM,
    "text2text-generation": Modality.LLM,
    "image-text-to-text": Modality.LLM,
    "automatic-speech-recognition": Modality.STT,
    "text-to-speech": Modality.TTS,
    "text-to-audio": Modality.TTS,
    "text-to-image": Modality.DIFFUSION,
    "image-to-image": Modality.DIFFUSION,
    "text-to-video": Modality.DIFFUSION,
    "image-to-video": Modality.DIFFUSION,
    "unconditional-image-generation": Modality.DIFFUSION,
}

_ARCH_HINTS: list[tuple[str, Modality]] = [
    (r"ForCausalLM$|ForConditionalGeneration$|LMHeadModel$", Modality.LLM),
    (r"Whisper|Wav2Vec2|Hubert|SpeechSeq2Seq|ForCTC$|Parakeet|Moonshine", Modality.STT),
    (r"Vits|SpeechT5|Bark|Kokoro|Parler|TTS|Musicgen|Dia|Csm|Orpheus|Vocos", Modality.TTS),
]

_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?(?:huggingface\.co|hf\.co)/"
    r"(?!datasets/|spaces/)(?P<repo>[\w.\-]+/[\w.\-]+)"
    r"(?:/(?:tree|blob|resolve)/(?P<rev>[^/?#]+))?"
)


def parse_model_ref(ref: str) -> tuple[str, str | None]:
    """Return ``(repo_id, revision)`` for a URL (``https://huggingface.co/org/name``)
    or a bare repo id (``org/name``, optionally ``org/name@rev``)."""
    ref = ref.strip()
    match = _URL_RE.match(ref)
    if match:
        return match.group("repo"), match.group("rev")
    if re.fullmatch(r"[\w.\-]+/[\w.\-]+(@[^\s]+)?", ref):
        repo, _, rev = ref.partition("@")
        return repo, rev or None
    raise ValueError(f"not a Hugging Face model URL or repo id: {ref!r}")


@dataclass
class ModelCard:
    repo_id: str
    revision: str | None
    modality: Modality
    pipeline_tag: str | None = None
    library: str | None = None
    architectures: list[str] = field(default_factory=list)
    model_type: str | None = None
    tags: list[str] = field(default_factory=list)
    params: int | None = None
    size_gb: float | None = None
    files: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    readme_excerpt: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["modality"] = self.modality.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelCard:
        data = dict(data)
        data["modality"] = Modality(data["modality"])
        return cls(**data)


def classify(
    pipeline_tag: str | None,
    library: str | None,
    architectures: list[str],
    tags: list[str],
    files: list[str],
) -> Modality:
    if "model_index.json" in files or library == "diffusers":
        return Modality.DIFFUSION
    if pipeline_tag in _PIPELINE_TAGS:
        return _PIPELINE_TAGS[pipeline_tag]
    for tag in tags:
        if tag in _PIPELINE_TAGS:
            return _PIPELINE_TAGS[tag]
    # Audio architectures are checked before the generic causal-LM pattern
    # because many TTS/STT models are also "...ForConditionalGeneration".
    for pattern, modality in reversed(_ARCH_HINTS):
        if any(re.search(pattern, arch) for arch in architectures):
            return modality
    return Modality.UNKNOWN


def resolve(ref: str, *, token: str | None = None, modality: str | None = None) -> ModelCard:
    """Fetch metadata (no weights) for ``ref`` from the Hub."""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    repo_id, revision = parse_model_ref(ref)
    api = HfApi(token=token)
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    siblings = info.siblings or []
    files = [s.rfilename for s in siblings]
    # Repos often ship the same weights twice (.safetensors + .bin); count one format.
    weight_ext = ".safetensors" if any(f.endswith(".safetensors") for f in files) else ".bin"
    size = sum((s.size or 0) for s in siblings if s.rfilename.endswith(weight_ext))

    def _json(name: str) -> dict[str, Any]:
        if name not in files:
            return {}
        try:
            path = hf_hub_download(repo_id, name, revision=revision, token=token)
        except EntryNotFoundError:
            return {}
        with open(path) as fh:
            loaded = json.load(fh)
        return loaded if isinstance(loaded, dict) else {}

    config = _json("config.json") or _json("model_index.json")
    architectures = list(config.get("architectures") or [])
    if not architectures and "_class_name" in config:
        architectures = [config["_class_name"]]

    readme = ""
    if "README.md" in files:
        try:
            with open(hf_hub_download(repo_id, "README.md", revision=revision, token=token)) as fh:
                readme = fh.read()[:6000]
        except EntryNotFoundError:
            pass

    params = None
    if info.safetensors is not None:
        params = int(info.safetensors.total)

    detected = classify(
        info.pipeline_tag, info.library_name, architectures, list(info.tags or []), files
    )
    return ModelCard(
        repo_id=repo_id,
        revision=revision,
        modality=Modality(modality) if modality else detected,
        pipeline_tag=info.pipeline_tag,
        library=info.library_name,
        architectures=architectures,
        model_type=config.get("model_type"),
        tags=list(info.tags or [])[:40],
        params=params,
        size_gb=round(size / 1024**3, 3) if size else None,
        files=files[:200],
        config={
            k: v
            for k, v in config.items()
            if not isinstance(v, dict | list) or k in {"architectures", "_class_name"}
        },
        readme_excerpt=readme,
    )
