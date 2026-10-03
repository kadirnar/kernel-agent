"""Diffusion pipelines (image / video) via ``diffusers``."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from kernel_agent.hub import Modality
from kernel_agent.workloads.base import Comparison, Workload, cosine, psnr

PROMPT = "a photograph of a red fox sitting in fresh snow, golden hour, highly detailed"


class DiffusionWorkload(Workload):
    modality = Modality.DIFFUSION
    defaults = {
        "prompt": PROMPT,
        "steps": 12,
        "height": 512,
        "width": 512,
        "seed": 0,
        "guidance_scale": None,
        "cpu_offload": False,
        "min_psnr": 25.0,
    }

    def load(self) -> None:
        from diffusers import DiffusionPipeline

        self.pipe = DiffusionPipeline.from_pretrained(
            self.spec.repo_id,
            revision=self.spec.revision,
            torch_dtype=self.dtype,
            trust_remote_code=self.spec.trust_remote_code,
        )
        if self.options["cpu_offload"]:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    def roots(self) -> dict[str, nn.Module]:
        return {
            name: comp for name, comp in self.pipe.components.items() if isinstance(comp, nn.Module)
        }

    def make_inputs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "prompt": self.options["prompt"],
            "num_inference_steps": int(self.options["steps"]),
            "height": int(self.options["height"]),
            "width": int(self.options["width"]),
        }
        if self.options["guidance_scale"] is not None:
            kwargs["guidance_scale"] = float(self.options["guidance_scale"])
        return kwargs

    def run(self, inputs: dict[str, Any]) -> dict[str, torch.Tensor]:
        generator = torch.Generator(device="cpu").manual_seed(int(self.options["seed"]))
        out = self.pipe(**inputs, generator=generator, output_type="pt")
        images = out.frames if hasattr(out, "frames") else out.images
        return {"images": torch.as_tensor(images).float().cpu()}

    def compare(self, reference: Any, candidate: Any) -> Comparison:
        ref, new = reference["images"], candidate["images"]
        if ref.shape != new.shape:
            return Comparison(False, {}, f"shape {tuple(new.shape)} != {tuple(ref.shape)}")
        value = psnr(ref, new, data_range=1.0)
        metrics: dict[str, float | int | str] = {
            "psnr_db": round(value, 2),
            "cosine": round(cosine(ref, new), 5),
        }
        minimum = float(self.options["min_psnr"])
        if value < minimum:
            return Comparison(False, metrics, f"PSNR {value:.1f} dB < {minimum} dB")
        return Comparison(True, metrics)
