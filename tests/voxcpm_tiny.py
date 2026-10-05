"""A tiny, randomly initialised VoxCPM2 (voxcpm's own classes, CPU, float32) for CPU
tests of code that drives VoxCPM2's internals (the batched throughput workload): the same
modules and code paths as the real model, a few hundred kilobytes of weights."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from typing import Any

import torch

HEAD_DIM = 8
VOCAB = 128  # VoxCPM2 special tokens: 101..104


def available() -> bool:
    return importlib.util.find_spec("voxcpm") is not None


class CharTokenizer:
    """``text -> ids``: one token per character (VoxCPM's tokenizer interface)."""

    def __call__(self, text: str) -> list[int]:
        return [5 + ord(c) % 95 for c in text]


def _lm(hidden: int, layers: int, heads: int, kv_heads: int, vocab: int = VOCAB) -> dict[str, Any]:
    half = [1.0] * (HEAD_DIM // 2)
    return {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_size": hidden,
        "intermediate_size": 2 * hidden,
        "max_position_embeddings": 512,
        "num_attention_heads": heads,
        "num_hidden_layers": layers,
        "num_key_value_heads": kv_heads,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000,
        "kv_channels": HEAD_DIM,
        "rope_scaling": {
            "type": "longrope",
            "long_factor": half,
            "short_factor": half,
            "original_max_position_embeddings": 512,
        },
        "vocab_size": vocab,
        "use_mup": False,
        "scale_emb": 12,
        "dim_model_base": 256,
        "scale_depth": 1.4,
    }


def tiny_voxcpm2(seed: int = 0) -> Any:
    """VoxCPM2Model with 2 base-LM layers (32 wide, 4 heads, 2 KV heads), 1 residual layer,
    1-layer LocEnc and LocDiT, patch size 2, 4-dim latents and an AudioVAE that decodes 128
    samples per patch (16 kHz) and encodes 1280 samples (80 ms) into one."""
    from voxcpm.model.voxcpm2 import VoxCPM2Model, VoxCPMConfig
    from voxcpm.modules.audiovae import AudioVAEConfigV2, AudioVAEV2

    config = VoxCPMConfig.model_validate(
        {
            "lm_config": _lm(32, 2, 4, 2),
            "patch_size": 2,
            "feat_dim": 4,
            "residual_lm_num_layers": 1,
            "residual_lm_no_rope": True,
            "scalar_quantization_latent_dim": 16,
            "scalar_quantization_scale": 9,
            "encoder_config": {
                "hidden_dim": 16,
                "ffn_dim": 32,
                "num_heads": 2,
                "num_layers": 1,
                "kv_channels": HEAD_DIM,
            },
            "dit_config": {
                "hidden_dim": 16,
                "ffn_dim": 32,
                "num_heads": 2,
                "num_layers": 1,
                "kv_channels": HEAD_DIM,
                "cfm_config": {"inference_cfg_rate": 2.0},
            },
            "max_length": 256,
            "device": "cpu",
            "dtype": "float32",
        }
    )
    vae = AudioVAEConfigV2(
        encoder_dim=4,
        encoder_rates=[8, 8, 10],  # VoxCPM2's hop (640): a reference voice of a few patches
        latent_dim=4,
        decoder_dim=32,
        decoder_rates=[4, 4, 4],
        sample_rate=16000,
        out_sample_rate=16000,
        sr_bin_boundaries=None,
    )
    torch.manual_seed(seed)
    model = VoxCPM2Model(config, SimpleNamespace(vocab={}), AudioVAEV2(vae), device="cpu")
    model.text_tokenizer = CharTokenizer()
    return model.eval()
