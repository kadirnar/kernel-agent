"""Causal / seq2seq language models via ``transformers``."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from kernel_agent.hub import Modality
from kernel_agent.workloads.base import Comparison, Workload, compare_tokens

PROMPT = (
    "The history of computing is a story of abstraction layers. Engineers first wired "
    "logic gates by hand, then wrote machine code, then assembly, then compilers that "
    "translated high level languages into efficient instructions. Today, GPU kernels are "
    "written in domain specific languages that hide the hardware details while still "
    "exposing tiling, memory hierarchy and parallelism to the programmer. "
)


class LLMWorkload(Workload):
    """Greedy generation: a ``prompt_len``-token prefill followed by
    ``new_tokens`` decode steps (both phases are profiled)."""

    modality = Modality.LLM
    defaults = {
        "prompt_len": 512,
        "new_tokens": 64,
        "batch_size": 1,
        "min_prefix": 16,
        "min_cosine": 0.99,
        "attn_implementation": "sdpa",
    }

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs: dict[str, Any] = {
            "revision": self.spec.revision,
            "trust_remote_code": self.spec.trust_remote_code,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec.repo_id, **kwargs)
        model: Any = AutoModelForCausalLM.from_pretrained(
            self.spec.repo_id,
            dtype=self.dtype,
            attn_implementation=self.options["attn_implementation"],
            **kwargs,
        )
        self.model = model.to(self.device).eval()

    def roots(self) -> dict[str, nn.Module]:
        return {"model": self.model}

    def reference_optimizations(self) -> str | None:
        """Static KV cache; ``generate`` then compiles the decode step itself
        (transformers' automatic ``torch.compile(mode="reduce-overhead")`` for
        compileable caches on CUDA; ``generation_config.compile_config`` tunes it)."""
        if not getattr(self.model, "_can_compile_fullgraph", False):
            return None  # the architecture does not declare torch.compile support
        self.model.generation_config.cache_implementation = "static"
        return (
            "static KV cache (cache_implementation='static'); transformers then "
            "torch.compiles the decode step (mode='reduce-overhead')"
        )

    def make_inputs(self) -> dict[str, torch.Tensor]:
        ids = self.tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt").input_ids[0]
        n = int(self.options["prompt_len"])
        reps = n // max(ids.numel(), 1) + 1
        ids = ids.repeat(reps)[:n]
        batch = ids.unsqueeze(0).repeat(int(self.options["batch_size"]), 1).to(self.device)
        return {"input_ids": batch, "attention_mask": torch.ones_like(batch)}

    def run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        n = int(self.options["new_tokens"])
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        out = self.model.generate(
            **inputs,
            max_new_tokens=n,
            min_new_tokens=n,
            do_sample=False,
            num_beams=1,
            pad_token_id=pad,
            return_dict_in_generate=True,
            output_scores=True,
        )
        prompt_len = inputs["input_ids"].shape[1]
        return {
            "tokens": out.sequences[:, prompt_len:].cpu(),
            "first_logits": out.scores[0].float().cpu(),
        }

    def compare(self, reference: Any, candidate: Any) -> Comparison:
        return compare_tokens(
            reference["tokens"],
            candidate["tokens"],
            reference.get("first_logits"),
            candidate.get("first_logits"),
            min_prefix=int(self.options["min_prefix"]),
            min_cosine=float(self.options["min_cosine"]),
        )
