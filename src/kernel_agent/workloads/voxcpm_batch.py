"""VoxCPM2 throughput: a batch of different requests decoded together.

``-o metric=throughput`` (``-o batch_size=N``, default 8) optimises seconds of audio
generated per wall second, the objective for many concurrent users. VoxCPM2's decode at
batch 1 is bound by kernel launches and weight reads, not by math, so decoding N requests
in one batch launches and reads the weights once per step for all of them (eager, RTX 5070
Ti: 1.9 s of audio per second at N = 1, 27 at N = 16, 66 at N = 128). VoxCPM's own
inference is batch 1 (``setup_cache(1, ...)``,
the stop flag of request 0 only); this workload is a faithful batched version of
``VoxCPM2Model._generate`` / ``_inference`` (zero-shot, or in a reference voice) for N
*different* texts:

* request 0 says ``text``, requests 1.. the sentences of :data:`REQUEST_TEXTS` (pairs of
  them past the end of the list). Request *b* draws its LocDiT noise from its own generator
  seeded ``seed + b``, the noise VoxCPM's batch-1 ``generate`` draws after
  ``torch.manual_seed(seed + b)``: the batched run intercepts the ``torch.randn`` call of
  ``feat_decoder.forward`` (:meth:`VoxCPMBatchWorkload.request_noise`).
* The prompts are right-padded to the longest one and prefilled once for the batch through
  the model's own ``forward``s (base LM, residual LM, LocEnc over the prompt): causal
  attention keeps every real position from seeing the padding, the text / audio masks zero
  the padded embeddings, and each request continues from its own last prompt position.
* A static KV cache with batch N per LM, sized to the longest prompt plus the generated
  patches. Every request decodes at its own position (its prompt length + the step):
  VoxCPM's ``forward_step`` takes one position for the whole batch, so the decode step is
  :meth:`VoxCPMBatchWorkload.lm_step`, ``forward_step`` with per-request positions (RoPE,
  cache write and attention mask per request) calling the layers' own norms, projections
  and MLPs.
* The LocDiT runs CFG at batch 2N, the LocEnc and the stop head at batch N, the AudioVAE
  decodes ``vae_batch`` requests per call (its fp32 activations are the memory limit).
  Stop flags are per request: in the fixed-length benchmark
  (``min_len = max_len = patches``) every request generates exactly ``patches`` patches;
  with the stop head live (natural-length run, perceptual samples) each request stops on
  its own and the loop ends when all have.

``-o metric=ttfa`` (with ``-o batch_size=N``) times a *burst*: N different requests
submitted at once and streamed together, as VoxCPM2's own streaming path streams one
(``generate_streaming``: every patch's newest latent through the stateful
``audio_vae.streaming_decode()``, one chunk per patch on the host). Each patch is decoded
for the whole batch in one ``decode_chunk`` call and marked once
(:meth:`~kernel_agent.workloads.base.Workload.mark_chunk`): the first mark is every
request's time to first audio (prefill, the first patch, its decode), the next ones give
the steady state (a patch's latency for all N streams against its 160 ms of audio) and
the playback stall. Inside :meth:`~kernel_agent.workloads.base.Workload.metric_window` the
loop stops at the first chunk. The quality checks judge the streamed audio, so an
AudioVAE replacement that breaks the stateful decode (its chunks decoded without the
previous chunk's causal-convolution state) fails the audio decoded from the reference
latents.

The metric's value (:mod:`kernel_agent.objective`) is the wall-clock time per second of
generated audio, ``1000 / throughput`` ms (``batch_size x patches`` patches of 160 ms in
the benchmark): lower stays better everywhere and every speedup is the throughput ratio.
``metric_detail`` reports the throughput (audio seconds per wall second) and the latency
of each request (a batch finishes together).

Quality is judged per request: teacher forcing on the batch's latents (one wrong request
fails the batch), the held-out input, the natural-length run (every request must stop at
its baseline patch). ``analyze`` also checks the batching itself (:meth:`self_check`):
every request against VoxCPM's own batch-1 ``generate`` with that text and seed, teacher
forced (bf16 batching changes GEMM shapes, nothing else). Each request's worst
teacher-forced step is excused when it is an outlier (``outlier_steps``, see ``defaults``).

Kernels and transforms written for batch 1 meet batch-N shapes here. The decode step does
not call ``MiniCPMDecoderLayer.forward_step`` / ``MiniCPMAttention.forward_step``,
``MiniCPMModel.forward_step`` or ``model._inference`` (their replacements are not used);
the prefill, LocEnc, LocDiT, AudioVAE, MLP, norm and projection modules run at batch N
(2N in the LocDiT): a replacement that cannot handle that must fall back, and per-request
teacher forcing rejects one that gets a request wrong.
"""

from __future__ import annotations

import contextlib
import functools
import math
from collections.abc import Callable, Iterator
from typing import Any

import torch
import torch.nn.functional as F

from kernel_agent import objective
from kernel_agent.workloads.base import Comparison
from kernel_agent.workloads.voxcpm import NATURAL_MIN_PATCHES, SHORT_TEXT, VoxCPMWorkload

#: Requests 1.. of a batch (request 0 says ``text``): different sentences and lengths.
REQUEST_TEXTS = (
    "The library opens at nine and closes late on Thursdays.",
    "Could you send me the slides from yesterday's meeting before lunch?",
    "Rain is expected in the afternoon, so take an umbrella with you.",
    "Our flight was delayed by two hours because of a storm over the mountains.",
    "Thank you for calling, please hold while we connect you to an agent.",
    "The recipe needs three eggs, a cup of flour and a pinch of salt.",
    "He finally fixed the old bicycle and rode it all the way to the coast.",
    "Turn left at the second traffic light and the station is on your right.",
    "Children learn languages quickly when they hear them every day.",
    "The museum's new exhibition shows paintings from the early twentieth century.",
    "Please confirm your appointment by replying to this message.",
    "A small cafe on the corner serves the best coffee in town.",
    "Scientists observed a bright comet passing close to the sun last week.",
    "The meeting has been moved to Friday at three in the conference room.",
    "Remember to water the plants while I am away on holiday.",
    "Good sound design makes a film feel real without anyone noticing it.",
)
#: The KV cache length of a batch (longest prompt + patches) is rounded up to this.
CACHE_ALIGN = 64
#: Failing requests named in a reason (the rest are counted).
SHOWN = 3


def request_texts(first: str, n: int) -> list[str]:
    """The ``n`` texts of a batch: ``first``, then :data:`REQUEST_TEXTS`, then pairs of
    them (every text differs)."""
    if n < 1:
        raise ValueError(f"batch_size must be >= 1, got {n}")
    pool = REQUEST_TEXTS
    texts = [first]
    for i in range(n - 1):
        k = i // len(pool)  # 0: one sentence; k: it and the sentence k further on
        sentence = pool[i % len(pool)]
        texts.append(sentence if k == 0 else f"{sentence} {pool[(i + k) % len(pool)]}")
    return texts


def per_request(results: list[Comparison], key: str) -> Comparison:
    """One verdict over per-request comparisons: every request must pass. The metrics are
    the first failing request's (else the one with the lowest ``key``), plus ``requests``,
    ``worst_request`` and ``failed_requests``."""
    failed = [b for b, c in enumerate(results) if not c.passed]
    worst = (
        failed[0]
        if failed
        else min(range(len(results)), key=lambda b: float(results[b].metrics.get(key, 1.0)))
    )
    metrics: dict[str, float | int | str] = {
        "requests": len(results),
        "worst_request": worst,
        **results[worst].metrics,
    }
    if failed:
        metrics["failed_requests"] = ",".join(map(str, failed))
    reasons = [f"request {b}: {results[b].reason}" for b in failed[:SHOWN]]
    if len(failed) > SHOWN:
        reasons.append(f"and {len(failed) - SHOWN} more of {len(results)} requests")
    return Comparison(not failed, metrics, "; ".join(reasons))


class VoxCPMBatchWorkload(VoxCPMWorkload):
    """``batch_size`` different requests, decoded as one batch (``metric=throughput``), or
    streamed as one burst (``metric=ttfa``)."""

    metrics = (objective.THROUGHPUT, objective.TTFA)
    # outlier_steps: the worst teacher-forced steps of each request excused from the step
    # thresholds (reported). Calibrated on VoxCPM2 (60 patches, batches of 4 / 8 / 16):
    # some trajectories have an ill-conditioned LocDiT step that any bf16-level change
    # flips (request 2 of the default batch, step 43: cosine 0.33 under every nn.Linear x
    # (1 ± 2^-8), 0.19 batched vs batch 1). Excusing one, correct changes (that
    # perturbation, the batching itself) keep mean >= 0.994 and min >= 0.75 per request;
    # broken RMSNorm (eps 1e-2) and request 0's noise for every request reach mean <= 0.84
    # on every request, a decode step at request 0's position <= 0.93 (5-12 steps below
    # 0.7) on each request with another prompt length (one at 0.985, still below 0.99).
    defaults = {
        **VoxCPMWorkload.defaults,
        "metric": objective.THROUGHPUT,
        "batch_size": 8,
        "vae_batch": 16,  # requests per AudioVAE decode call
        "outlier_steps": 1,
    }
    teacher_forcing_note = (
        "VoxCPM batch teacher forcing wraps `model.feat_decoder.forward` at run time and "
        "judges every request of the batch: candidates must call the current "
        "`model.feat_decoder` from Python once per patch for the whole batch and draw its "
        "noise with one `torch.randn((batch, ...))` call (the batched run serves row b from "
        "request b's generator). The batched loop is the workload's own "
        "(`kernel_agent/workloads/voxcpm_batch.py`): it never calls `model._inference`, "
        "`MiniCPMModel.forward_step` or the layers' `forward_step` (replacing them has no "
        "effect); the decode step is `workload.lm_step(lm, cache, hidden, positions)` (one "
        "position per request), which a transform may replace, and it calls each layer's "
        "norms, `self_attn.q_proj/k_proj/v_proj/o_proj` and `mlp` at batch N."
    )
    _static_kv = False  # the KV caches' addresses are static (a compiled `lm_step`)
    _kv: tuple[tuple[int, int], tuple[Any, ...]] | None = None  # (batch, slots), caches

    def load(self) -> None:
        super().load()
        if not hasattr(self.model, "fusion_concat_proj"):
            raise ValueError(
                "batch_size / metric=throughput needs VoxCPM2: the batched loop mirrors "
                "VoxCPM2Model._inference (VoxCPM 1.x has another residual path)"
            )
        if self.options["compile"]:  # `-o compile=true`: model.optimize() ran in load
            self._compile_lm_step()

    def reference_optimizations(self) -> str | None:
        """VoxCPM's ``model.optimize()`` (its compiled LocEnc and LocDiT estimator serve
        batch N too; its compiled ``forward_step``s are not called by the batched loop) and
        the batched decode step, :meth:`lm_step`, compiled the same way."""
        described = super().reference_optimizations()
        if described is None:
            return None
        self._compile_lm_step()
        return (
            f"{described}; the batched decode step (`lm_step`, a position per request) "
            "compiled the same way"
        )

    def _compile_lm_step(self) -> None:
        """``torch.compile(mode="reduce-overhead", fullgraph=True)`` of :meth:`lm_step`, as
        ``model.optimize()`` compiles ``forward_step`` (the caches at static addresses)."""
        self._static_kv = True
        self._kv = None
        self.lm_step = torch.compile(  # type: ignore[method-assign]
            self.lm_step, mode="reduce-overhead", fullgraph=True
        )

    @property
    def batch_size(self) -> int:
        return int(self.options["batch_size"])

    def make_inputs(self) -> list[str]:  # type: ignore[override]
        return request_texts(str(self.options["text"]), self.batch_size)

    def variants(self) -> list[dict[str, Any]]:
        """Another prefill length, and another batch size, few patches each."""
        few = min(int(self.options["patches"]), 8)
        return [
            {"text": SHORT_TEXT, "patches": few},
            {"batch_size": max(1, self.batch_size // 2), "patches": few},
        ]

    def output_seconds(self) -> float:
        """The audio of a fixed-length run: ``batch_size x patches`` patches (fixed by the
        options, whatever a candidate reports); with the stop head live, what was marked."""
        if self.options.get("min_patches") is not None:
            return super().output_seconds()
        samples = self.batch_size * int(self.options["patches"]) * self._patch_samples()
        return samples / self.sampling_rate

    def _patch_samples(self) -> int:
        """Audio samples the AudioVAE decodes from one latent patch."""
        return int(self.model.patch_size) * int(self.model._decode_chunk_size)

    # ------------------------------------------------------------------ the batched loop

    def run(self, inputs: list[str]) -> dict[str, Any]:  # type: ignore[override]
        latents: list[torch.Tensor] = []

        def record(pred: torch.Tensor) -> torch.Tensor:
            latents.append(pred.detach().clone())
            return pred

        with self._decoder_hook(record):
            audio, steps, margins = self._generate_batch(list(inputs))
        out = {
            # [batch, samples]: request b's audio, zero past steps[b] patches
            "audio": audio,
            # [patches, batch, feat_dim, patch_size]; empty if the decoder was bypassed
            "latents": torch.stack(latents).float().cpu() if latents else torch.empty(0),
            "steps": steps,  # [batch] patches generated per request
            "sampling_rate": self.sampling_rate,
        }
        if margins is not None:
            out["stop_margins"] = margins  # [steps, batch], natural length only
        return out

    def _generate_batch(
        self, texts: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """``VoxCPM2Model._generate`` + ``_inference`` for a batch of texts (zero-shot, or in
        the ``reference_wav`` voice): ``(audio [batch, samples], patches per request,
        stop margins [steps, batch] when the stop head is live)``."""
        model = self.model
        n = int(self.options["patches"])
        min_patches = self.options.get("min_patches")
        min_len = n if min_patches is None else int(min_patches)
        seed = int(self.options["seed"])
        device = torch.device(model.device)
        batch = len(texts)

        prompts = self._prompts(texts)
        lengths = [int(p[0].shape[0]) for p in prompts]
        width = max(lengths)
        text = torch.zeros(batch, width, dtype=torch.long)
        text_mask = torch.zeros(batch, width, dtype=torch.int32)
        feat_mask = torch.zeros(batch, width, dtype=torch.int32)
        feat = torch.zeros(batch, width, model.patch_size, model.audio_vae.latent_dim)
        for b, (tokens, feats, t_mask, a_mask) in enumerate(prompts):  # right padding
            text[b, : lengths[b]] = tokens
            feat[b, : lengths[b]] = feats
            text_mask[b, : lengths[b]] = t_mask
            feat_mask[b, : lengths[b]] = a_mask
        text, text_mask, feat_mask = text.to(device), text_mask.to(device), feat_mask.to(device)
        feat = feat.to(device).to(model._dtype())
        rows = torch.arange(batch, device=device)
        last = torch.tensor(lengths, device=device) - 1

        torch.manual_seed(seed)  # anything not intercepted: as deterministic as VoxCPM's run
        generators = [torch.Generator(device=device).manual_seed(seed + b) for b in range(batch)]
        base_cache, residual_cache = self._caches(batch, width + n)

        # prefill (VoxCPM2Model._inference before its loop, at batch N)
        prefill_encoder = getattr(model, "_feat_encoder_raw", model.feat_encoder)
        feat_embed = model.enc_to_lm_proj(prefill_encoder(feat))
        lm_config = model.config.lm_config
        scale_emb = lm_config.scale_emb if lm_config.use_mup else 1.0
        text_embed = model.base_lm.embed_tokens(text) * scale_emb
        combined = text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed
        prefix_feat_cond = feat[rows, last]  # [b, p, d]

        enc_outputs, kv = model.base_lm(inputs_embeds=combined, is_causal=True)
        base_cache.fill_caches(kv)
        audio_part, text_part = feat_mask.unsqueeze(-1), text_mask.unsqueeze(-1)
        enc_outputs = model.fsq_layer(enc_outputs) * audio_part + enc_outputs * text_part
        lm_hidden = enc_outputs[rows, last]
        residual_inputs = model.fusion_concat_proj(
            torch.cat((enc_outputs, audio_part * feat_embed), dim=-1)
        )
        residual_outputs, residual_kv = model.residual_lm(
            inputs_embeds=residual_inputs, is_causal=True
        )
        residual_cache.fill_caches(residual_kv)
        residual_hidden = residual_outputs[rows, last]

        # decode loop (VoxCPM2Model._inference, per-request positions and stop flags)
        positions = last + 1  # request b's first decode step sits after its prompt
        steps = [n] * batch
        finished = torch.zeros(batch, dtype=torch.bool)
        margins: list[torch.Tensor] = []
        pred_feat_seq = []
        flags = self.async_flags()
        chunks: list[torch.Tensor] = []  # metric=ttfa: the streamed audio, a patch per chunk
        stream = contextlib.ExitStack()
        decoder = None
        if self.metric == objective.TTFA:
            decoder = stream.enter_context(model.audio_vae.streaming_decode())
        with stream:
            for i in range(n):
                # every request's own stop flag (it depends on lm_hidden only), sent to the host
                # now and read once the LocDiT and LocEnc are queued: one host sync per patch, as
                # VoxCPM's loop, but on the flags' copy, not on the whole patch (bit-identical)
                logits = model.stop_head(model.stop_actn(model.stop_proj(lm_hidden)))
                ticket = flags.send(logits.argmax(dim=-1))
                dit_hidden = torch.cat(
                    (model.lm_to_dit_proj(lm_hidden), model.res_to_dit_proj(residual_hidden)),
                    dim=-1,
                )
                with self.request_noise(generators):
                    pred_feat = model.feat_decoder(
                        mu=dit_hidden,
                        patch_size=model.patch_size,
                        cond=prefix_feat_cond.transpose(1, 2).contiguous(),
                        n_timesteps=int(self.options["timesteps"]),
                        cfg_value=float(self.options["cfg"]),
                    ).transpose(1, 2)  # [b, p, d]
                curr_embed = model.enc_to_lm_proj(model.feat_encoder(pred_feat.unsqueeze(1)))
                pred_feat_seq.append(pred_feat.unsqueeze(1))
                prefix_feat_cond = pred_feat
                if decoder is not None:  # metric=ttfa: VoxCPM2's streaming path, batched
                    # the newest patch's latent [b, d, p] through the stateful decode: one
                    # chunk per request, on the host, marked once for the burst
                    wav = decoder.decode_chunk(pred_feat.transpose(1, 2).to(torch.float32))
                    chunks.append(wav.squeeze(1).float().cpu())
                    self.mark_chunk(audio_ms=1000.0 * chunks[-1].shape[-1] / self.sampling_rate)
                    if self.in_window:  # the time to first audio is known: stop here
                        steps = [i + 1] * batch
                        break

                stop = flags.read(ticket)
                if min_len < n:
                    margins.append(logits.detach().float())
                if i > min_len:
                    for b in (stop.eq(1) & ~finished).nonzero().flatten().tolist():
                        steps[b] = i + 1
                    finished |= stop.eq(1)
                    if bool(finished.all()):
                        break

                position = positions + i
                lm_hidden = model.fsq_layer(
                    self.lm_step(model.base_lm, base_cache, curr_embed[:, 0, :], position)
                )
                residual_input = model.fusion_concat_proj(
                    torch.cat((lm_hidden, curr_embed[:, 0, :]), dim=-1)
                )
                residual_hidden = self.lm_step(
                    model.residual_lm, residual_cache, residual_input, position
                )

        if chunks:  # streamed: request b's audio, zero past its own end
            audio = torch.cat(chunks, dim=-1)
            per_patch = self._patch_samples()
            for b, length in enumerate(steps):
                audio[b, length * per_patch :] = 0
        else:
            seq = torch.cat(pred_feat_seq, dim=1)  # [b, t, p, d]
            latent = seq.permute(0, 3, 1, 2).reshape(batch, seq.shape[-1], -1)  # [b, d, t * p]
            audio = self._decode(latent, steps)
        stop_margins = None
        if margins:
            stacked = torch.stack(margins).cpu()  # [steps, batch, 2]
            stop_margins = stacked[..., 1] - stacked[..., 0]
        return audio, torch.tensor(steps), stop_margins

    def _prompts(
        self, texts: list[str]
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """``(tokens, feats, text_mask, audio_mask)`` per text, as ``_generate`` builds them
        (zero-shot, or behind the ``reference_wav`` prefix, encoded once for the batch)."""
        model = self.model
        prefix = None
        if wav := self._reference_wav().get("reference_wav_path"):
            ref_feat = model._encode_wav(wav, padding_mode="right")
            prefix = model._make_ref_prefix(ref_feat, torch.device("cpu"))
        prompts = []
        for text in texts:
            tokens: torch.Tensor = torch.LongTensor(model.text_tokenizer(text))
            tokens = torch.cat([tokens, torch.tensor([model.audio_start_token], dtype=torch.int32)])
            length = tokens.shape[0]
            feats = torch.zeros((length, model.patch_size, model.audio_vae.latent_dim))
            t_mask = torch.ones(length, dtype=torch.int32)
            a_mask = torch.zeros(length, dtype=torch.int32)
            if prefix is not None:
                r_tokens, r_feats, r_t_mask, r_a_mask = prefix
                tokens, feats = torch.cat([r_tokens, tokens]), torch.cat([r_feats, feats])
                t_mask, a_mask = torch.cat([r_t_mask, t_mask]), torch.cat([r_a_mask, a_mask])
            prompts.append((tokens, feats, t_mask, a_mask))
        return prompts

    def _caches(self, batch: int, length: int) -> tuple[Any, Any]:
        """Static KV caches with batch ``batch`` for both LMs (kept for the next run of the
        same shape; ``fill_caches`` zeroes them as VoxCPM does)."""
        from voxcpm.modules.minicpm4.cache import StaticKVCache

        length = -(-length // CACHE_ALIGN) * CACHE_ALIGN
        key = (batch, length)
        kept = self._kv
        if kept is None or kept[0] != key:
            caches = []
            for lm in (self.model.base_lm, self.model.residual_lm):
                _, layers, _, heads, _, dim = lm.kv_cache.kv_cache.shape
                cache = lm.kv_cache.kv_cache
                caches.append(
                    StaticKVCache(layers, heads, dim, batch, cache.device, cache.dtype, length)
                )
            if self._static_kv:  # CUDA graphs of a compiled lm_step update them in place
                for c in caches:
                    torch._dynamo.mark_static_address(c.kv_cache)
            kept = self._kv = (key, tuple(caches))
        return kept[1]

    def lm_step(
        self, lm: Any, cache: Any, hidden: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """One decode step of ``lm`` (a ``MiniCPMModel``) for the batch: VoxCPM's
        ``forward_step``, request *b* at position ``positions[b]`` (RoPE, KV-cache slot and
        the attention mask over ``[0, positions[b]]``). The layers' norms, projections and
        MLPs are called as modules. A model-level transform may replace this method."""
        batch = hidden.shape[0]
        rope = None
        if lm.rope_emb is not None:
            cos, sin = lm.rope_emb(positions)  # [b, head_dim]
            rope = (cos[:, None, None, :], sin[:, None, None, :])
        slots = cache.kv_cache.shape[-2]
        mask = torch.arange(slots, device=hidden.device)[None, :] <= positions[:, None]
        mask = mask.view(batch, 1, 1, slots)
        rows = torch.arange(batch, device=hidden.device)
        for i, layer in enumerate(lm.layers):  # MiniCPMDecoderLayer.forward_step
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            hidden = _attention_step(
                layer.self_attn, hidden, rope, rows, positions, mask, cache.get_layer_cache(i)
            )
            scale = layer.scale_depth / math.sqrt(layer.num_hidden_layers)
            hidden = residual + (hidden * scale if layer.use_mup else hidden)
            residual = hidden
            hidden = layer.mlp(layer.post_attention_layernorm(hidden))
            hidden = residual + (hidden * scale if layer.use_mup else hidden)
        out: torch.Tensor = lm.norm(hidden)
        return out

    @contextlib.contextmanager
    def request_noise(self, generators: list[torch.Generator]) -> Iterator[None]:
        """Within the block, a ``torch.randn`` call for a ``[batch, ...]`` tensor (no
        ``generator`` given) draws row *b* from ``generators[b]`` as VoxCPM's batch-1 run draws
        it, so every request keeps its own noise whatever the batch. The block must make at
        least one such call (``feat_decoder.forward`` draws its noise there)."""
        original = torch.randn
        drawn = 0

        @functools.wraps(original)
        def randn(*size: Any, generator: Any = None, **kwargs: Any) -> torch.Tensor:
            nonlocal drawn
            shape = tuple(size[0]) if len(size) == 1 and not isinstance(size[0], int) else size
            if generator is not None or "out" in kwargs or not shape or shape[0] != len(generators):
                return original(*size, generator=generator, **kwargs)
            drawn += 1
            rows = [original((1, *shape[1:]), generator=g, **kwargs) for g in generators]
            return rows[0] if len(rows) == 1 else torch.cat(rows)

        torch.randn = randn
        try:
            yield
        finally:
            torch.randn = original
        if not drawn:
            raise RuntimeError(
                "the batched run draws every request's LocDiT noise from its own generator, "
                "but `model.feat_decoder` drew none with `torch.randn((batch, ...))`: a "
                "candidate must keep drawing the noise with torch.randn in feat_decoder.forward"
            )

    def _decode(self, latent: torch.Tensor, steps: list[int]) -> torch.Tensor:
        """The AudioVAE decode of every request, batched: up to ``vae_batch`` requests of
        one length per call (the fp32 decoder's activations grow with the batch: 16
        requests of 60 patches peak at ~5 GB); marks each request's audio
        (:meth:`mark_chunk`). Returns ``[batch, samples]``, zero past a request's end."""
        per_patch = self._patch_samples()
        patch = int(self.model.patch_size)
        size = max(1, int(self.options["vae_batch"]))
        audio = torch.zeros(len(steps), max(steps) * per_patch)
        for length in sorted(set(steps)):
            same = [b for b, s in enumerate(steps) if s == length]
            for group in (same[i : i + size] for i in range(0, len(same), size)):
                z = latent if len(group) == len(steps) else latent[group]
                wav = self.model.audio_vae.decode(z[..., : length * patch].to(torch.float32))
                wav = wav.squeeze(1).float().cpu()
                audio[group, : wav.shape[-1]] = wav
                for _ in group:
                    self.mark_chunk(audio_ms=1000.0 * wav.shape[-1] / self.sampling_rate)
        return audio

    # ------------------------------------------------------------------ quality

    def run_teacher_forced(  # type: ignore[override]
        self, inputs: list[str], reference: Any
    ) -> dict[str, Any]:
        ref = reference.get("latents") if isinstance(reference, dict) else None
        if ref is None or ref.numel() == 0:
            raise ValueError(
                "the baseline output has no recorded latents; re-run `analyze` to record them"
            )
        preds: list[torch.Tensor] = []

        def force(pred: torch.Tensor) -> torch.Tensor:
            step = len(preds)
            preds.append(pred.detach().float().cpu())
            if step >= ref.shape[0]:
                return pred  # longer than the reference; reported as a step-count mismatch
            if tuple(ref.shape[1:]) != tuple(pred.shape):
                raise ValueError(
                    f"the reference holds patches {tuple(ref.shape[1:])}, the batch predicts "
                    f"{tuple(pred.shape)} (another batch size?)"
                )
            return ref[step].to(device=pred.device, dtype=pred.dtype)

        with self._decoder_hook(force):
            out = self.run(inputs)  # through `self.run`: transforms wrapping it stay active
        if not preds:
            raise RuntimeError(
                "teacher forcing saw no `model.feat_decoder` call: the candidate bypasses the "
                "Python decoder call, so it cannot be validated"
            )
        return {"latents": torch.stack(preds), "audio": out["audio"], "steps": out["steps"]}

    def _batch_mismatch(self, reference: Any, candidate: Any, key: str) -> Comparison | None:
        ref, new = reference[key], candidate[key]
        if ref.dim() == new.dim() and ref.shape[:2] == new.shape[:2]:
            return None
        return Comparison(
            False,
            {"shape": str(list(new.shape)), "reference_shape": str(list(ref.shape))},
            f"{key} {list(new.shape)} != the reference's {list(ref.shape)}",
        )

    def compare(self, reference: Any, candidate: Any) -> Comparison:
        """Free running, per request (informational: the trajectory is chaotic)."""
        if bad := self._batch_mismatch(reference, candidate, "audio"):
            return bad
        single = super().compare
        return per_request(
            [
                single({"audio": reference["audio"][b]}, {"audio": candidate["audio"][b]})
                for b in range(reference["audio"].shape[0])
            ],
            "spectral_cosine",
        )

    def compare_teacher_forced(self, reference: Any, candidate: Any) -> Comparison:
        """VoxCPM's teacher-forced comparison for every request of the batch, each with its
        ``outlier_steps`` worst steps excused (``excused_steps`` in the metrics)."""
        ref, new = reference["latents"], candidate["latents"]
        if ref.dim() != 4 or new.dim() != 4 or ref.shape[1] != new.shape[1]:
            return Comparison(
                False,
                {"shape": str(list(new.shape)), "reference_shape": str(list(ref.shape))},
                f"teacher-forced latents {list(new.shape)} do not hold the reference's "
                f"{ref.shape[1] if ref.dim() > 1 else '?'} requests",
            )
        results, excused = [], []
        for b in range(ref.shape[1]):
            r, n = ref[:, b : b + 1], new[:, b : b + 1]
            keep = list(range(r.shape[0]))
            if r.shape[0] == n.shape[0]:  # else compare_steps reports the step counts
                cos = F.cosine_similarity(r.float().flatten(1), n.float().flatten(1), dim=1)
                worst = cos.argsort()[: int(self.options["outlier_steps"])].tolist()
                floor = float(self.options["min_step_cosine"])
                outliers = sorted(i for i in worst if cos[i] < floor)
                excused += [f"request {b} step {i} ({float(cos[i]):.3f})" for i in outliers]
                keep = [i for i in keep if i not in outliers]
                r, n = r[keep], n[keep]
            cmp = super().compare_teacher_forced(
                {"latents": r, "audio": reference["audio"][b]},
                {"latents": n, "audio": candidate["audio"][b]},
            )
            if isinstance(step := cmp.metrics.get("worst_step"), int) and step < len(keep):
                cmp.metrics["worst_step"] = keep[step]  # a step of the whole trajectory
            results.append(cmp)
        cmp = per_request(results, "mean_step_cosine")
        if excused:
            cmp.metrics["excused_steps"] = ", ".join(excused)
        return cmp

    def self_check(self, inputs: Any, reference: Any) -> dict[str, Any]:
        """The batching itself: request *b* of the batched loop, teacher forced on VoxCPM's
        own batch-1 ``generate`` of its text with seed ``seed + b``, must pass the
        teacher-forced comparison (every request)."""
        texts = list(inputs)
        seed = int(self.options["seed"])
        latents, audio = [], []
        for b, text in enumerate(texts):
            with self.with_options({"seed": seed + b}):
                single = VoxCPMWorkload.run(self, text)  # VoxCPM's own batch-1 generate
            latents.append(single["latents"])
            audio.append(single["audio"])
        batch1 = {"latents": torch.cat(latents, dim=1), "audio": torch.stack(audio)}
        forced = self.run_teacher_forced(texts, batch1)
        cmp = self.compare_teacher_forced(batch1, forced)
        return {
            "passed": cmp.passed,
            "reason": cmp.reason,
            "check": f"each of {len(texts)} requests vs VoxCPM's batch-1 generate, teacher forced",
            **cmp.metrics,
        }

    # ------------------------------------------------------------------ stop condition

    def natural_length_run(self, reference: Any = None) -> dict[str, Any]:
        """The batch with the stop head live: request 0 says ``natural_text``, the others
        their usual texts (``min_len`` 2, ``max_len`` ``natural_max_patches``), teacher
        forced on ``reference`` when given. ``steps`` is the longest request;
        ``request_steps`` and the baseline's ``request_stop_margins`` are per request."""
        max_patches = int(self.options["natural_max_patches"])
        options = {
            "text": self.options["natural_text"],
            "patches": max_patches,
            "min_patches": NATURAL_MIN_PATCHES,
        }
        with self.with_options(options):
            inputs = self.make_inputs()
            if reference is None:
                out = self.run(inputs)
            else:
                out = self.run_teacher_forced(inputs, reference)
        steps = [int(s) for s in out["steps"]]
        result: dict[str, Any] = {
            "steps": max(steps),
            "min_steps": NATURAL_MIN_PATCHES,
            "max_steps": max_patches,
            "request_steps": steps,
            "latents": out["latents"],
            "audio": out["audio"],
        }
        margins = out.get("stop_margins")
        if reference is None and margins is not None:
            result["request_stop_margins"] = [
                [float(m) for m in margins[: steps[b], b]] for b in range(len(steps))
            ]
        return result

    def compare_natural_length(self, reference: Any, candidate: Any) -> Comparison:
        """Every request must stop at its baseline patch (:func:`compare_stop` per request)."""
        ref_steps, new_steps = reference["request_steps"], candidate["request_steps"]
        if len(ref_steps) != len(new_steps):
            return Comparison(
                False,
                {"requests": len(new_steps), "reference_requests": len(ref_steps)},
                f"{len(new_steps)} requests != the reference's {len(ref_steps)}",
            )
        margins = reference.get("request_stop_margins") or [None] * len(ref_steps)
        check: Callable[[dict[str, Any], dict[str, Any]], Comparison] = (
            super().compare_natural_length
        )
        results = [
            check(
                {
                    "steps": r,
                    "min_steps": reference.get("min_steps", NATURAL_MIN_PATCHES),
                    "max_steps": reference.get("max_steps"),
                    "stop_margins": m,
                },
                {"steps": c},
            )
            for r, c, m in zip(ref_steps, new_steps, margins, strict=True)
        ]
        cmp = per_request(results, "steps")
        cmp.metrics.update(steps=max(new_steps), reference_steps=max(ref_steps))
        return cmp

    # ------------------------------------------------------------------ perceptual gate

    def perceptual_quality(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Each perceptual sample ran as a batch (the sample's text is request 0, the
        throughput run's code path at its batch size); request 0's audio is scored."""
        firsts = []
        for s in samples:
            out = s["output"]
            length = int(out["steps"][0]) * self._patch_samples()
            first = {"audio": out["audio"][0, :length], "sampling_rate": out["sampling_rate"]}
            firsts.append({**s, "output": first})
        return super().perceptual_quality(firsts)


def _attention_step(
    attn: Any,
    hidden: torch.Tensor,
    rope: tuple[torch.Tensor, torch.Tensor] | None,
    rows: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor,
    kv_cache: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """``MiniCPMAttention.forward_step`` with a position per request."""
    from voxcpm.modules.minicpm4.model import apply_rotary_pos_emb

    batch = hidden.shape[0]
    heads, kv_heads, dim = attn.num_heads, attn.num_key_value_heads, attn.head_dim
    query = attn.q_proj(hidden).view(batch, 1, heads, dim).transpose(1, 2)
    key = attn.k_proj(hidden).view(batch, 1, kv_heads, dim).transpose(1, 2)
    value = attn.v_proj(hidden).view(batch, 1, kv_heads, dim).transpose(1, 2)
    if rope is not None:
        query, key = apply_rotary_pos_emb(query, key, *rope)
    key_cache, value_cache = kv_cache
    key_cache[rows, :, positions] = key[:, :, 0]
    value_cache[rows, :, positions] = value[:, :, 0]
    out = F.scaled_dot_product_attention(
        query.contiguous(), key_cache, value_cache, attn_mask=mask, enable_gqa=True
    )
    out = out.transpose(1, 2).contiguous().reshape(batch, heads * dim)
    result: torch.Tensor = attn.o_proj(out)
    return result
