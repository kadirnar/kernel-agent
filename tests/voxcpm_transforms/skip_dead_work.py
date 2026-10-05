"""Decode loop without dead work (on top of async_stop_loop's sync-free loop).

The reference `VoxCPM2Model._inference` computes three things whose results it
never uses:
* after the last generated patch (i == max_len - 1) the loop still runs the
  LocEnc on that patch, enc_to_lm_proj, the base LM step (28 layers, ~2.6 GB
  of weights), FSQ, fusion_concat_proj and the residual LM step; the loop then
  ends and only `pred_feat_seq` is used;
* the stop head (stop_proj 2048x2048 + SiLU + head + argmax + D2H) every
  patch, although the flag is only consulted when `i > min_len`;
* in the prefill, the LocEnc over the prompt audio features when `feat_mask`
  is all zero (zero-shot): its output only enters as `feat_mask * feat_embed`,
  i.e. exact zeros.  Checked at run time (one host read next to the
  reference's own `feat_mask[0, -1].item()`); otherwise it runs as before.
Everything else is async_stop_loop's loop: same calls, same order, same math;
`model.feat_decoder` is still called from Python once per patch.
"""


import types

import torch
from einops import rearrange
from tqdm import tqdm


def _pos(value: int, device) -> torch.Tensor:
    return torch.tensor([value]).pin_memory().to(device, non_blocking=True)


@torch.inference_mode()
def _inference(
    self,
    text,
    text_mask,
    feat,
    feat_mask,
    min_len=2,
    max_len=2000,
    inference_timesteps=10,
    cfg_value=2.0,
    streaming=False,
    streaming_prefix_len=4,
):
    B, T, P, D = feat.shape

    if bool(feat_mask.any().item()):
        prefill_encoder = getattr(self, "_feat_encoder_raw", self.feat_encoder)
        feat_embed = prefill_encoder(feat)  # [b, t, h_feat]
        feat_embed = self.enc_to_lm_proj(feat_embed)
    else:
        # feat_mask is all zero: feat_mask * feat_embed is exactly zero.
        feat_embed = feat.new_zeros(B, T, self.enc_to_lm_proj.out_features)

    if self.config.lm_config.use_mup:
        scale_emb = self.config.lm_config.scale_emb
    else:
        scale_emb = 1.0

    text_embed = self.base_lm.embed_tokens(text) * scale_emb
    combined_embed = text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed

    prefix_feat_cond = feat[:, -1, ...]  # b, p, d
    pred_feat_seq = []  # b, t, p, d
    curr_embed = None

    has_continuation_audio = feat_mask[0, -1].item() == 1
    context_len = 0
    if has_continuation_audio:
        audio_indices = feat_mask.squeeze(0).nonzero(as_tuple=True)[0]
        context_len = min(streaming_prefix_len - 1, len(audio_indices))
        last_audio_indices = audio_indices[-context_len:]
        pred_feat_seq = list(feat[:, last_audio_indices, :, :].split(1, dim=1))
    else:
        pred_feat_seq = []

    enc_outputs, kv_cache_tuple = self.base_lm(
        inputs_embeds=combined_embed,
        is_causal=True,
    )
    self.base_lm.kv_cache.fill_caches(kv_cache_tuple)

    enc_outputs = self.fsq_layer(enc_outputs) * feat_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
    lm_hidden = enc_outputs[:, -1, :]

    residual_enc_inputs = self.fusion_concat_proj(
        torch.cat((enc_outputs, feat_mask.unsqueeze(-1) * feat_embed), dim=-1)
    )
    residual_enc_outputs, residual_kv_cache_tuple = self.residual_lm(
        inputs_embeds=residual_enc_inputs,
        is_causal=True,
    )
    self.residual_lm.kv_cache.fill_caches(residual_kv_cache_tuple)
    residual_hidden = residual_enc_outputs[:, -1, :]

    stop_host = torch.empty((), dtype=torch.int64).pin_memory()
    stop_ready = torch.cuda.Event()

    for i in tqdm(range(max_len)):
        last = i == max_len - 1
        # Stop flag of this iteration (depends only on lm_hidden): issued now,
        # read on the host after the DiT/LocEnc work is queued.  Only needed
        # when the break condition can consult it.
        need_stop = i > min_len and not last
        if need_stop:
            stop_dev = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden))).argmax(dim=-1)[0]
            stop_host.copy_(stop_dev, non_blocking=True)
            stop_ready.record()

        dit_hidden_1 = self.lm_to_dit_proj(lm_hidden)  # [b, h_dit]
        dit_hidden_2 = self.res_to_dit_proj(residual_hidden)  # [b, h_dit]
        dit_hidden = torch.cat((dit_hidden_1, dit_hidden_2), dim=-1)

        pred_feat = self.feat_decoder(
            mu=dit_hidden,
            patch_size=self.patch_size,
            cond=prefix_feat_cond.transpose(1, 2).contiguous(),
            n_timesteps=inference_timesteps,
            cfg_value=cfg_value,
        ).transpose(
            1, 2
        )  # [b, p, d]

        if not last:  # curr_embed only feeds the next LM steps
            curr_embed = self.feat_encoder(pred_feat.unsqueeze(1))  # b, 1, c
            curr_embed = self.enc_to_lm_proj(curr_embed)

        pred_feat_seq.append(pred_feat.unsqueeze(1))  # b, 1, p, d
        prefix_feat_cond = pred_feat

        if streaming:
            feat_pred = rearrange(pred_feat.unsqueeze(1), "b t p d -> b d (t p)", b=B, p=self.patch_size)

            yield feat_pred, pred_feat_seq, context_len

            if len(pred_feat_seq) > streaming_prefix_len:
                pred_feat_seq = pred_feat_seq[-streaming_prefix_len:]

        if last:
            break  # the loop ends here: the LM steps below would be dead
        if need_stop:
            stop_ready.synchronize()
            if stop_host.item() == 1:
                break

        lm_hidden = self.base_lm.forward_step(
            curr_embed[:, 0, :], _pos(self.base_lm.kv_cache.step(), curr_embed.device)
        ).clone()

        lm_hidden = self.fsq_layer(lm_hidden)
        curr_residual_input = self.fusion_concat_proj(torch.cat((lm_hidden, curr_embed[:, 0, :]), dim=-1))
        residual_hidden = self.residual_lm.forward_step(
            curr_residual_input, _pos(self.residual_lm.kv_cache.step(), curr_embed.device)
        ).clone()

    if not streaming:
        pred_feat_seq = torch.cat(pred_feat_seq, dim=1)  # b, t, p, d
        feat_pred = rearrange(pred_feat_seq, "b t p d -> b d (t p)", b=B, p=self.patch_size)
        generated_feat = pred_feat_seq[:, context_len:, :, :].squeeze(0).cpu()
        yield feat_pred, generated_feat, context_len


def apply(workload) -> None:
    model = workload.model
    if not torch.cuda.is_available() or not next(model.parameters()).is_cuda:
        return
    model._inference = types.MethodType(_inference, model)
