"""VoxCPM2's batched loop as a :class:`~kernel_agent.workloads.serving.SlotModel` (issue
#149): the generic :func:`~kernel_agent.workloads.serving.serve` on a real model, for the
tests (the tiny CPU VoxCPM2 of ``voxcpm_tiny``). The pieces are those of
``VoxCPMBatchWorkload._generate_batch``: a prefill of the admitted requests written into
their slots' rows of the static KV caches, the batched LocDiT / LocEnc step with every
slot's own noise generator, and the batched decode step (``lm_step``) at every slot's own
position. A request's post stage is its AudioVAE decode."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


class VoxCPMSlots:
    """``slots`` batch rows of ``workload`` (a loaded ``VoxCPMBatchWorkload``) serving
    ``texts`` (request *m* draws its LocDiT noise from a generator seeded ``seed + m``, as
    VoxCPM's batch-1 ``generate`` with that seed); ``patches``: the most steps of a
    request."""

    def __init__(self, workload: Any, texts: list[str], slots: int, patches: int) -> None:
        self.wl, self.model, self.slots = workload, workload.model, slots
        model = self.model
        self.device = torch.device(model.device)
        self.seed = int(workload.options["seed"])
        self.prompts = workload._prompts(texts)
        longest = max(int(p[0].shape[0]) for p in self.prompts)
        self.caches = workload._caches(slots, longest + patches)
        for cache in self.caches:
            cache.kv_cache.zero_()
        self.limit = self.caches[0].kv_cache.shape[-2] - 1  # idle slots stop advancing there
        hidden = model.config.lm_config.hidden_size
        dtype = model._dtype()
        self.lm_hidden = torch.zeros(slots, hidden, device=self.device, dtype=dtype)
        self.residual_hidden = torch.zeros_like(self.lm_hidden)
        shape = (slots, model.patch_size, model.audio_vae.latent_dim)
        self.prefix = torch.zeros(shape, device=self.device, dtype=dtype)
        self.positions = torch.zeros(slots, dtype=torch.long, device=self.device)
        spare = torch.Generator(device=self.device).manual_seed(self.seed - 1)  # idle slots
        self.generators = [spare] * slots
        self.patches: list[torch.Tensor] = []  # every step's [slots, p, d]
        self.curr: torch.Tensor | None = None

    def admit(self, slots: list[int], requests: list[int]) -> None:
        """The prefill of ``_generate_batch`` for these requests, into these slots."""
        model, device = self.model, self.device
        prompts = [self.prompts[m] for m in requests]
        lengths = [int(p[0].shape[0]) for p in prompts]
        width, k = max(lengths), len(prompts)
        text = torch.zeros(k, width, dtype=torch.long)
        text_mask = torch.zeros(k, width, dtype=torch.int32)
        feat_mask = torch.zeros(k, width, dtype=torch.int32)
        feat = torch.zeros(k, width, model.patch_size, model.audio_vae.latent_dim)
        for b, (tokens, feats, t_mask, a_mask) in enumerate(prompts):
            text[b, : lengths[b]], feat[b, : lengths[b]] = tokens, feats
            text_mask[b, : lengths[b]], feat_mask[b, : lengths[b]] = t_mask, a_mask
        text, text_mask, feat_mask = text.to(device), text_mask.to(device), feat_mask.to(device)
        feat = feat.to(device).to(model._dtype())
        rows = torch.arange(k, device=device)
        last = torch.tensor(lengths, device=device) - 1

        encoder = getattr(model, "_feat_encoder_raw", model.feat_encoder)
        feat_embed = model.enc_to_lm_proj(encoder(feat))
        lm_config = model.config.lm_config
        scale_emb = lm_config.scale_emb if lm_config.use_mup else 1.0
        text_embed = model.base_lm.embed_tokens(text) * scale_emb
        combined = text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed
        enc_outputs, kv = model.base_lm(inputs_embeds=combined, is_causal=True)
        audio_part, text_part = feat_mask.unsqueeze(-1), text_mask.unsqueeze(-1)
        enc_outputs = model.fsq_layer(enc_outputs) * audio_part + enc_outputs * text_part
        residual_inputs = model.fusion_concat_proj(
            torch.cat((enc_outputs, audio_part * feat_embed), dim=-1)
        )
        residual_outputs, residual_kv = model.residual_lm(
            inputs_embeds=residual_inputs, is_causal=True
        )

        index = torch.tensor(slots, device=device)
        for cache, layers in zip(self.caches, (kv, residual_kv), strict=True):
            cache.kv_cache[:, :, index] = 0  # the slot's previous request: gone
            for i, (key, value) in enumerate(layers):
                cache.kv_cache[0, i, index, :, :width] = key
                cache.kv_cache[1, i, index, :, :width] = value
        # out of place: the previous tensors are recorded steps of other requests
        self.lm_hidden = self.lm_hidden.index_copy(0, index, enc_outputs[rows, last])
        self.residual_hidden = self.residual_hidden.index_copy(
            0, index, residual_outputs[rows, last]
        )
        self.prefix = self.prefix.index_copy(0, index, feat[rows, last])
        self.positions = self.positions.index_copy(0, index, last + 1)
        for s, m in zip(slots, requests, strict=True):
            self.generators[s] = torch.Generator(device=self.device).manual_seed(self.seed + m)

    def step(self, send: Callable[[torch.Tensor], int]) -> None:
        model = self.model
        logits = model.stop_head(model.stop_actn(model.stop_proj(self.lm_hidden)))
        send(logits.argmax(dim=-1))
        dit_hidden = torch.cat(
            (model.lm_to_dit_proj(self.lm_hidden), model.res_to_dit_proj(self.residual_hidden)),
            dim=-1,
        )
        with self.wl.request_noise(self.generators):
            pred = model.feat_decoder(
                mu=dit_hidden,
                patch_size=model.patch_size,
                cond=self.prefix.transpose(1, 2).contiguous(),
                n_timesteps=int(self.wl.options["timesteps"]),
                cfg_value=float(self.wl.options["cfg"]),
            ).transpose(1, 2)
        self.curr = model.enc_to_lm_proj(model.feat_encoder(pred.unsqueeze(1)))
        self.patches.append(pred)
        self.prefix = pred

    def advance(self) -> None:
        model, curr = self.model, self.curr
        assert curr is not None
        base, residual = self.caches
        lm_hidden = model.fsq_layer(
            self.wl.lm_step(model.base_lm, base, curr[:, 0, :], self.positions)
        )
        residual_input = model.fusion_concat_proj(torch.cat((lm_hidden, curr[:, 0, :]), dim=-1))
        self.residual_hidden = self.wl.lm_step(
            model.residual_lm, residual, residual_input, self.positions
        )
        self.lm_hidden = lm_hidden
        self.positions = (self.positions + 1).clamp(max=self.limit)

    def finish(
        self, slots: list[int], requests: list[int], steps: int
    ) -> Callable[[], torch.Tensor]:
        seq = torch.stack(self.patches[len(self.patches) - steps :])[:, slots]  # [t, k, p, d]
        vae = self.model.audio_vae

        def decode() -> torch.Tensor:
            latent = seq.permute(1, 3, 0, 2).reshape(len(slots), seq.shape[-1], -1)
            wav: torch.Tensor = vae.decode(latent.to(torch.float32))
            return wav.squeeze(1).float()

        return decode
