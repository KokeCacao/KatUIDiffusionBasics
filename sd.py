import torch
import warnings

from tqdm import tqdm
from torch import Tensor
from typing import TypedDict, Optional, Literal, Dict, Any
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from backend.loader.decorator import KatzukiNode
from backend.nodes.builtin import BaseNode
from nodes.KatUIDiffusionBasics.util import should_update
from nodes.KatUIDiffusionBasics.basic import SchedulerLoader

from kokikit.diffusion import predict_noise_sd


class SDModelLoader(BaseNode):

    @KatzukiNode(node_type="diffusion.sd.model_loader")
    def __init__(self) -> None:
        pass

    class ReturnDict(TypedDict):
        text_encoder: CLIPTextModel
        tokenizer: CLIPTokenizer
        unet: UNet2DConditionModel
        vae: AutoencoderKL

    def execute(
            self,
            sd_path: Literal[
                "runwayml/stable-diffusion-v1-5",
                "stabilityai/stable-diffusion-2-1-base",
                "stabilityai/stable-diffusion-2-1",
            ] = "runwayml/stable-diffusion-v1-5",
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cuda"),
    ) -> ReturnDict:
        tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder='tokenizer', use_fast=False)
        text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
            sd_path,
            subfolder='text_encoder',
            device_map={'': 0},
        ) # type: ignore
        unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet") # type: ignore
        vae: AutoencoderKL = AutoencoderKL.from_pretrained(sd_path, subfolder="vae") # type: ignore

        text_encoder = text_encoder.to(device=device, dtype=dtype) # type: ignore
        unet = unet.to(device=device, dtype=dtype)
        vae = vae.to(device=device, dtype=dtype)

        return self.ReturnDict(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            vae=vae,
        )


class RunSD(BaseNode):

    @KatzukiNode(node_type="diffusion.sd.run_sd")
    def __init__(self) -> None:
        pass

    def execute(
        self,
        prompt_embeds_pos: Tensor,
        unet: UNet2DConditionModel,
        latents_original: Tensor,
        prompt_embeds_neg: Optional[Tensor] = None,
        scheduler: DDIMScheduler = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
        cfg: float = 7.5,
        vae: Optional[AutoencoderKL] = None,
        diffusion_steps: int = 50,
        cfg_rescale: float = 0.5,
        controlnet_condition: Dict[str, Any] = {},
    ) -> Tensor:
        if prompt_embeds_neg is None:
            prompt_embeds_neg = torch.zeros_like(prompt_embeds_pos)

        if scheduler is None:
            scheduler = SchedulerLoader().execute()

        device = latents_original.device
        scheduler.set_timesteps(diffusion_steps, device=device)

        with torch.no_grad():
            pbar = tqdm(scheduler.timesteps - 1)
            latents_noised = latents_original
            for step, time in enumerate(pbar):
                self.check_execution_state_change()

                latents_noised = latents_original
                noise_pred, noise_pred_x0 = predict_noise_sd(
                    unet_sd=unet,
                    latents_noised=latents_noised,
                    text_embeddings_conditional=prompt_embeds_pos,
                    text_embeddings_unconditional=prompt_embeds_neg,
                    cfg=cfg,
                    lora_scale=0.0,
                    t=time,
                    scheduler=scheduler,
                    reconstruction_loss=False,
                    cfg_rescale=cfg_rescale,
                    **controlnet_condition,
                )
                latents_noised = scheduler.step(noise_pred, time, latents_noised).prev_sample # type: ignore
                latents_original = latents_noised

                if vae is not None and (step % 10 == 0 or step == len(pbar) - 1):
                    _ = latents_noised if noise_pred_x0 is None else noise_pred_x0 # use x0 instead of actual image if available
                    image_batch = vae.decode(1 / vae.config['scaling_factor'] * _.clone().detach()).sample # [B, C, H, W] # type: ignore
                    image_batch = (image_batch / 2 + 0.5).clamp(0, 1) * 255
                    image_batch = image_batch.permute(0, 2, 3, 1).detach().cpu() # [B, C, H, W] -> [B, H, W, C]
                    self.set_output("latents_noised", image_batch)
                    self.set_output("progress", int(100 * step / len(pbar)))
                    self.send_update()
        return latents_noised
