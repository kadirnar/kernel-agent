"""Exact greedy prompt-lookup decoding with the model's own eager forward and a DynamicCache,
reporting its counters with Workload.report_stats (evidence for #170)."""

from __future__ import annotations

from typing import Any

import torch

K = 10  # drafted tokens per verification
NGRAMS = (3, 2)


def _draft(hist: list[int]) -> list[int] | None:
    for n in NGRAMS:
        if len(hist) <= n:
            continue
        tail = hist[-n:]
        for s in range(len(hist) - n - 1, -1, -1):  # most recent earlier occurrence
            if hist[s : s + n] == tail:
                cont = hist[s + n : s + n + K]
                if cont:
                    return cont
    return None


def apply(workload: Any) -> None:
    from transformers import DynamicCache

    model = workload.model
    original = workload.run
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])

    def greedy(logits: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        logits[..., eos] = float("-inf")  # min_new_tokens = new_tokens: EOS is masked
        return logits

    def run(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ids = inputs["input_ids"]
        if ids.shape[0] != 1:
            return original(inputs)
        n = int(workload.options["new_tokens"])
        cache = DynamicCache(config=model.config)
        first = greedy(model(input_ids=ids, past_key_values=cache, use_cache=True).logits[0, -1])
        tok = int(first.argmax())
        hist, gen = [*ids[0].tolist(), tok], [tok]
        workload.report_stats(steps=1, tokens=1)
        while len(gen) < n:
            draft = _draft(hist)
            draft = draft[: n - len(gen) - 1] if draft else None
            if not draft:
                x = torch.tensor([[tok]], device=ids.device)
                lg = greedy(model(input_ids=x, past_key_values=cache, use_cache=True).logits[0, -1])
                tok = int(lg.argmax())
                gen.append(tok)
                hist.append(tok)
                workload.report_stats(steps=1, tokens=1)
                continue
            x = torch.tensor([[tok, *draft]], device=ids.device)
            g = greedy(model(input_ids=x, past_key_values=cache, use_cache=True).logits[0])
            g = g.argmax(-1).tolist()
            a = 0
            while a < len(draft) and draft[a] == g[a]:
                a += 1
            new = g[: a + 1][: n - len(gen)]
            gen += new
            hist += new
            tok = new[-1]
            cache.crop(len(hist) - 1)  # K/V up to the last accepted token; `tok` comes next
            workload.report_stats(steps=1, verifies=1, drafted=len(draft), accepted=a, tokens=len(new))
        return {"tokens": torch.tensor([gen]), "first_logits": first[None].cpu()}

    workload.run = run
