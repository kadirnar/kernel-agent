"""Decode loop without per-patch host syncs that drain the GPU queue.

The reference `VoxCPM2Model._inference` loop blocks the host three times per
generated patch:
* `stop_head(...).argmax(...).cpu().item()` right after the LocEnc call waits
  for the whole DiT solver + LocEnc of the patch to finish;
* `torch.tensor([kv_cache.step()], device=cuda)` (base and residual LM) is a
  synchronous pageable H2D copy, i.e. a stream sync, each.
So the GPU queue is empty every patch, and the next launches (incl. the
~1.3 ms CPU-side cudaGraphLaunch of the 1.5k-node solver graph) sit on the
critical path: ~1-2 ms GPU idle per patch.

Here the loop is the same sequence of model calls with the same math, but:
* the stop logits are computed from `lm_hidden` (already final at the top of
  the iteration) *before* the DiT call and copied to pinned host memory with
  an event; the host checks the flag after it has queued the DiT + LocEnc
  work, so the wait only covers work issued before the DiT, and the break
  decision (same iteration, same condition) happens before the next LM step;
* position ids go through pinned memory with non-blocking copies.
`model.feat_decoder` is still called from Python once per patch (randn
inside), and the prefill / output handling is unchanged.
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

    prefill_encoder = getattr(self, "_feat_encoder_raw", self.feat_encoder)
    feat_embed = prefill_encoder(feat)  # [b, t, h_feat]
    feat_embed = self.enc_to_lm_proj(feat_embed)

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
        # Stop flag of this iteration (depends only on lm_hidden): issued now,
        # read on the host after the DiT/LocEnc work is queued.
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

        curr_embed = self.feat_encoder(pred_feat.unsqueeze(1))  # b, 1, c
        curr_embed = self.enc_to_lm_proj(curr_embed)

        pred_feat_seq.append(pred_feat.unsqueeze(1))  # b, 1, p, d
        prefix_feat_cond = pred_feat

        if streaming:
            feat_pred = rearrange(pred_feat.unsqueeze(1), "b t p d -> b d (t p)", b=B, p=self.patch_size)

            yield feat_pred, pred_feat_seq, context_len

            if len(pred_feat_seq) > streaming_prefix_len:
                pred_feat_seq = pred_feat_seq[-streaming_prefix_len:]

        stop_ready.synchronize()
        stop_flag = stop_host.item()
        if i > min_len and stop_flag == 1:
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
