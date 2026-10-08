"""Causal / seq2seq language models via ``transformers``.

The prompt is ``prompt_len`` tokens of long, non-repeating text (inputs version 2, #170):
the first tokens of a built-in ``context`` (``essay``: :data:`~.texts.ESSAY`, then the
story; ``story``: the held-out one) or of ``-o prompt=...`` (repeated when shorter),
rotated by ``prompt_shift`` tokens and ended by a ``request`` where one is set (the diverse
input set, :data:`~.texts.REQUESTS`). A run recorded before version 2 keeps its prompt: one
paragraph repeated up to ``prompt_len`` (:data:`PROMPT`), which a greedy model keeps
repeating, so prompt-lookup speculative decoding looked 40-67x faster on it.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from kernel_agent.hub import Modality
from kernel_agent.workloads import texts
from kernel_agent.workloads.base import Comparison, Workload, compare_tokens

#: Built-in contexts of inputs version 2 (``-o context=``): the prompt is cut from them.
CONTEXTS = {
    "essay": f"{texts.ESSAY}\n\n{texts.STORY}",
    "story": f"{texts.STORY}\n\n{texts.ESSAY}",
}
#: The diverse input set: the context of input ``i`` starts ``i`` x this many tokens later.
DIVERSE_STRIDE = 61
#: Inputs version 1: the prompt of runs recorded before #170, this paragraph repeated.
PROMPT = (
    "The history of computing is a story of abstraction layers. Engineers first wired "
    "logic gates by hand, then wrote machine code, then assembly, then compilers that "
    "translated high level languages into efficient instructions. Today, GPU kernels are "
    "written in domain specific languages that hide the hardware details while still "
    "exposing tiling, memory hierarchy and parallelism to the programmer. "
)
#: Inputs version 1: the held-out prompt (``holdout_options``), repeated too.
HOLDOUT_PROMPT = (
    "A lighthouse keeper on a remote island kept a careful log of every storm, every "
    "passing ship and every change in the colour of the sea. Years later, sailors read "
    "the notebooks to learn which currents were safe and which harbours offered shelter "
    "when the weather turned without warning. "
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
        # tokens may diverge earlier where the baseline's greedy choice was a near-tie: a
        # top-1 minus top-2 logit margin at most this (calibration: README)
        "near_tie": 0.5,
        "attn_implementation": "sdpa",
    }
    #: ``--quality near-lossless`` with the teacher-forced gate (:meth:`perceptual_quality`):
    #: the free-running greedy tokens are informational (FP8 weights change the first token
    #: on some natural prompts) and the first-step logits keep a sanity floor (calibration:
    #: README).
    near_lossless_options = {"min_prefix": 0, "min_cosine": 0.98}
    #: ``--quality relaxed`` (#175): twice the first-step logits budget (1 - 0.98 -> 1 - 0.96);
    #: FP8 scales x1.05 reach 0.98, int4 weights, scales x1.2, RMSNorm eps 1e-2 and a dropped
    #: KV head <= 0.86 (README).
    relaxed_options = {"min_cosine": 0.96}

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

    def _prompt_ids(self, prompt: str) -> torch.Tensor:
        ids: torch.Tensor = self.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0]
        return ids

    @property
    def legacy_inputs(self) -> bool:
        """The run was recorded before inputs version 2: the repeated paragraph."""
        return self.spec.inputs_version < 2

    def holdout_options(self, variant: int = 1) -> dict[str, Any] | None:
        """Another prompt at the same length (the story); variants ``>= 2`` rotate it by
        1..L-1 tokens."""
        held: dict[str, Any] = {"prompt": HOLDOUT_PROMPT}
        if not self.legacy_inputs:  # the story, also when -o prompt= sets the main prompt
            held = {"prompt": None, "context": "story"}
        text = HOLDOUT_PROMPT if self.legacy_inputs else CONTEXTS["story"]
        length = max(self._prompt_ids(text).numel(), 2)
        shift = 0 if variant <= 1 else 1 + (variant - 2) % (length - 1)
        return {**held, "prompt_shift": shift}

    def diverse_inputs(self) -> dict[str, dict[str, Any]]:
        """The 12 requests of :data:`~.texts.REQUESTS` (news, dialogue, code, recipe, poem,
        question, math, a CSV table, German, French, Spanish, Chinese), each at the end of
        a ``prompt_len``-token prompt whose context starts :data:`DIVERSE_STRIDE` tokens
        later than the previous one's: the shapes of the main input, other content. None
        for a run recorded before inputs version 2."""
        if self.legacy_inputs:
            return {}
        return {
            label: {"request": text, "prompt_shift": DIVERSE_STRIDE * i}
            for i, (label, text) in enumerate(texts.REQUESTS.items())
        }

    def variants(self) -> list[dict[str, Any]]:
        """A short prompt and a longer batch-2 prompt, a few decode steps each."""
        return [
            {"prompt_len": 37, "new_tokens": 4},
            {"prompt_len": 301, "batch_size": 2, "new_tokens": 4},
        ]

    def _context(self) -> str:
        """The text the prompt is cut from: ``prompt``, else the built-in ``context``."""
        if self.options.get("prompt"):
            return str(self.options["prompt"])
        if self.legacy_inputs:
            return PROMPT
        name = str(self.options.get("context") or "essay")
        if name not in CONTEXTS:
            raise ValueError(f"unknown context {name!r}; choose one of: {', '.join(CONTEXTS)}")
        return CONTEXTS[name]

    def make_inputs(self) -> dict[str, torch.Tensor]:
        n = int(self.options["prompt_len"])
        ids = self._prompt_ids(self._context())
        request = None if self.legacy_inputs else self.options.get("request")
        tail = self._prompt_ids(f"\n\n{request}")[-n:] if request else ids[:0]
        head = _window(ids, n - tail.numel(), int(self.options.get("prompt_shift") or 0))
        ids = torch.cat([head, tail])
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
        result = {
            "tokens": out.sequences[:, prompt_len:].cpu(),
            "first_logits": out.scores[0].float().cpu(),
        }
        if not self.legacy_inputs:  # top-1 minus top-2 logit of every step: the near-ties
            top2 = torch.stack(out.scores, 1).float().topk(2, dim=-1).values
            result["margins"] = (top2[..., 0] - top2[..., 1]).cpu()
        return result

    def compare(self, reference: Any, candidate: Any) -> Comparison:
        return compare_tokens(
            reference["tokens"],
            candidate["tokens"],
            reference.get("first_logits"),
            candidate.get("first_logits"),
            min_prefix=int(self.options["min_prefix"]),
            min_cosine=float(self.options["min_cosine"]),
            ref_margins=reference.get("margins"),
            near_tie=float(self.options["near_tie"]),
        )

    def perceptual_samples(self) -> list[dict[str, Any]]:
        """``--quality near-lossless``: the main prompt, the held-out one and the diverse
        set (14 natural prompts at the main input's shapes), judged teacher forced
        (:meth:`perceptual_quality`). None for a run recorded before inputs version 2."""
        if self.legacy_inputs:
            return []
        held = self.holdout_options(1) or {}
        samples = [{"sample": "main"}, {"sample": "held-out", **held}]
        return samples + [{"sample": k, **v} for k, v in self.diverse_inputs().items()]

    def perceptual_quality(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Teacher-forced scores (:func:`~.perceptual.score_llm`): one forward of the
        (patched) model over the prompt and a continuation. Against eager's continuation
        (``reference``): KL and top-1 agreement per position; on its own: the
        log-likelihood. Numerics that differ by design (FP8 weights) make free-running
        greedy text diverge within a few tokens on natural prompts although every
        distribution stays close; teacher forcing measures that closeness."""
        from kernel_agent.workloads.perceptual import score_llm

        items = []
        for s in samples:
            with self.with_options(s["options"]):
                prompt = self.make_inputs()["input_ids"][0]
            tokens = torch.as_tensor(s["output"]["tokens"])
            items.append(
                {
                    "prompt": prompt,
                    "tokens": tokens[0] if tokens.dim() > 1 else tokens,
                    "reference": s.get("reference"),
                }
            )
        with torch.inference_mode():
            return score_llm(self.model, items)

    def compare_perceptual(
        self, reference: list[dict[str, Any]], candidate: list[dict[str, Any]]
    ) -> Comparison:
        from kernel_agent.workloads import perceptual as p

        opt = self.options  # thresholds: -o max_kl=... (calibration: README)
        return p.compare_llm(
            reference,
            candidate,
            max_kl=float(opt.get("max_kl", p.LLM_MAX_KL)),
            max_kl_worst=float(opt.get("max_kl_worst", p.LLM_MAX_KL_WORST)),
            min_top1=float(opt.get("min_top1", p.LLM_MIN_TOP1)),
            max_nll_increase=float(opt.get("max_nll_increase", p.LLM_MAX_NLL_INCREASE)),
        )


def _window(ids: torch.Tensor, n: int, shift: int) -> torch.Tensor:
    """``n`` tokens of ``ids`` read cyclically from token ``shift``: no repetition while
    ``n`` is at most ``len(ids)``."""
    length = max(ids.numel(), 1)
    shift %= length
    return ids.repeat((n + shift) // length + 1)[shift : shift + n]
