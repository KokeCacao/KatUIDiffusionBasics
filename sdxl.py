import torch

from tqdm import tqdm
from torch import Tensor
from typing import Any, TypedDict, Optional, Union, List, Tuple, Dict, Literal
from diffusers.models.controlnet import ControlNetModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers import AutoencoderKL
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from backend.loader.decorator import KatzukiNode
from backend.nodes.builtin import BaseNode
from nodes.KatUIDiffusionBasics.util import should_update
from nodes.KatUIDiffusionBasics.basic import LatentImage, SchedulerLoader

from kokikit.diffusion import predict_noise_sdxl, predict_noise_sdxl_turbo


class SDXLModelLoader(BaseNode):

    @KatzukiNode(node_type="diffusion.sdxl.model_loader")
    def __init__(self) -> None:
        pass

    class ReturnDict(TypedDict):
        text_encoder_one: CLIPTextModel
        text_encoder_two: CLIPTextModelWithProjection
        tokenizer_one: CLIPTokenizer
        tokenizer_two: CLIPTokenizer
        unet: UNet2DConditionModel
        vae: AutoencoderKL

    def execute(self, sdxl_path: str = "stabilityai/sdxl-turbo", dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cuda")) -> ReturnDict:
        tokenizer_one: CLIPTokenizer = CLIPTokenizer.from_pretrained(sdxl_path, subfolder="tokenizer", use_fast=False)
        tokenizer_two: CLIPTokenizer = CLIPTokenizer.from_pretrained(sdxl_path, subfolder='tokenizer_2', use_fast=False)
        text_encoder_one: CLIPTextModel = CLIPTextModel.from_pretrained(
            sdxl_path,
            subfolder="text_encoder",
            device_map={'': 0} # BUG: https://github.com/tloen/alpaca-lora/issues/368 (NotImplementedError: Cannot copy out of meta tensor; no data!)
            # BUG: solution: https://huggingface.co/docs/transformers/main_classes/model
            ,
        ) # type: ignore
        text_encoder_two: CLIPTextModelWithProjection = CLIPTextModelWithProjection.from_pretrained(
            sdxl_path,
            subfolder='text_encoder_2',
            device_map={'': 0},
        ) # type: ignore
        unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(sdxl_path, subfolder="unet") # type: ignore
        vae: AutoencoderKL = AutoencoderKL.from_pretrained(sdxl_path, subfolder="vae") # type: ignore

        text_encoder_one = text_encoder_one.to(device=device, dtype=dtype) # type: ignore
        text_encoder_two = text_encoder_two.to(device=device, dtype=dtype) # type: ignore
        unet = unet.to(device=device, dtype=dtype)
        vae = vae.to(device=device, dtype=dtype)

        return self.ReturnDict(
            text_encoder_one=text_encoder_one,
            text_encoder_two=text_encoder_two,
            tokenizer_one=tokenizer_one,
            tokenizer_two=tokenizer_two,
            unet=unet,
            vae=vae,
        )


class SDXLTextEncode(BaseNode):

    @KatzukiNode(node_type="diffusion.sdxl.text_encode")
    def __init__(self) -> None:
        pass

    class ReturnDict(TypedDict):
        prompt_embeds: Union[List[Tensor], Tensor]
        pooled_prompt_embeds: Union[List[Tensor], Tensor]
        micro_embeds: Union[List[Tensor], Tensor]

    def execute(
            self,
            prompts_one: Union[List[str], str],
            original_size: Union[Tuple[int, int], List[Tuple[int, int]]],
            text_encoder_one: CLIPTextModel,
            text_encoder_two: CLIPTextModelWithProjection,
            tokenizer_one: CLIPTokenizer,
            tokenizer_two: CLIPTokenizer,
            prompts_two: Optional[Union[List[str], str]] = None,
            crops_coords_top_left: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None,
            target_size: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None,
            device: torch.device = torch.device("cuda"),
    ) -> ReturnDict:
        if prompts_two is None:
            prompts_two = prompts_one
        if target_size is None:
            target_size = original_size
        if crops_coords_top_left is None:
            if isinstance(original_size, list):
                crops_coords_top_left = [(0, 0) for _ in original_size]
            else:
                crops_coords_top_left = (0, 0)

        if isinstance(prompts_one, str):
            assert isinstance(prompts_two, str)
            assert isinstance(original_size, tuple)
            assert isinstance(crops_coords_top_left, tuple)
            assert isinstance(target_size, tuple)
            original_size = [original_size]
            crops_coords_top_left = [crops_coords_top_left]
            target_size = [target_size]
            batch_size = 1
        elif isinstance(prompts_one, list):
            assert isinstance(prompts_two, list)
            assert isinstance(original_size, list)
            assert isinstance(crops_coords_top_left, list)
            assert isinstance(target_size, list)
            batch_size = len(prompts_one)
        else:
            raise ValueError(f"Prompt type {type(prompts_one)} is not supported")

        prompts = [prompts_one, prompts_two]
        text_encoders = [text_encoder_one, text_encoder_two]
        tokenizers = [tokenizer_one, tokenizer_two]
        prompt_embeds_list = []
        pooled_prompt_embeds = None

        # obtain pooled_prompt_embeds and prompt_embeds
        for prompt, text_encoder, tokenizer in zip(prompts, text_encoders, tokenizers):
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1:-1])
                print(f"The following part of your input was truncated because CLIP can only handle sequences up to"
                      f" {tokenizer.model_max_length} tokens: {removed_text}")

            prompt_embeds = text_encoder(
                text_input_ids.to(device),
                output_hidden_states=True, # We need hidden_states for SDXL
            )

            pooled_prompt_embeds = (prompt_embeds[0]).to(device)
            prompt_embeds = (prompt_embeds.hidden_states[-2]).to(device)
            # BUG: why above two output is always on CPU?

            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        dtype = prompt_embeds.dtype

        # convert to list if input is list
        assert pooled_prompt_embeds is not None
        if isinstance(prompts_one, list):
            prompt_embeds = [prompt_embeds[i] for i in range(batch_size)]
            pooled_prompt_embeds = [pooled_prompt_embeds[i] for i in range(batch_size)]

        # create micro_embeds
        micro_embeds = []
        for i in range(batch_size):
            micro_embeds.append(torch.cat(
                [
                    torch.tensor(original_size[i], dtype=dtype, device=device),
                    torch.tensor(crops_coords_top_left[i], dtype=dtype, device=device),
                    torch.tensor(target_size[i], dtype=dtype, device=device),
                ],
                dim=-1,
            ))
        # convert to single tensor if input is single string
        if isinstance(prompts_one, str):
            micro_embeds = micro_embeds[0]

        return self.ReturnDict(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            micro_embeds=micro_embeds,
        )


class RunSDXL(BaseNode):

    @KatzukiNode(node_type="diffusion.sdxl.run_sdxl")
    def __init__(self) -> None:
        pass

    def execute(
        self,
        prompt_embeds_pos: Tensor,
        pooled_prompt_embeds_pos: Tensor,
        micro_embeds_pos: Tensor,
        unet: UNet2DConditionModel,
        latents_original: Tensor,
        prompt_embeds_neg: Optional[Tensor] = None,
        pooled_prompt_embeds_neg: Optional[Tensor] = None,
        micro_embeds_neg: Optional[Tensor] = None,
        scheduler: DDIMScheduler = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
        cfg: float = 7.5,
        vae: Optional[AutoencoderKL] = None,
        diffusion_steps: int = 50,
        cfg_rescale: float = 0.5,
        controlnet_condition: Dict[str, Any] = {},
    ) -> Tensor:
        if prompt_embeds_neg is None:
            prompt_embeds_neg = torch.zeros_like(prompt_embeds_pos)
        if pooled_prompt_embeds_neg is None:
            pooled_prompt_embeds_neg = torch.zeros_like(pooled_prompt_embeds_pos)
        if micro_embeds_neg is None:
            micro_embeds_neg = micro_embeds_pos

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
                noise_pred, noise_pred_x0 = predict_noise_sdxl(
                    unet_sdxl=unet,
                    latents_noised=latents_noised,
                    text_embeddings_conditional=prompt_embeds_pos,
                    text_embeddings_unconditional=prompt_embeds_neg,
                    text_embeddings_conditional_pooled=pooled_prompt_embeds_pos,
                    text_embeddings_unconditional_pooled=pooled_prompt_embeds_neg,
                    text_embeddings_conditional_micro=micro_embeds_pos,
                    text_embeddings_unconditional_micro=micro_embeds_neg,
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


class RunSDXLTurbo(BaseNode):

    @KatzukiNode(node_type="diffusion.sdxl.run_sdxl_turbo")
    def __init__(self) -> None:
        pass

    def execute(
        self,
        prompt_embeds_pos: Tensor,
        pooled_prompt_embeds_pos: Tensor,
        micro_embeds_pos: Tensor,
        unet: UNet2DConditionModel,
        latents_original: Tensor,
        scheduler: DDIMScheduler = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
        vae: Optional[AutoencoderKL] = None,
        diffusion_steps: int = 4,
        controlnet_condition: Dict[str, Any] = {},
    ) -> Tensor:
        if scheduler is None:
            scheduler = SchedulerLoader().execute()

        device = latents_original.device
        scheduler.set_timesteps(diffusion_steps, device=device)

        with torch.no_grad():
            pbar = tqdm(scheduler.timesteps)
            # TODO: so to fix 1 step inference, we multiply by init_noise_sigma and remove -1 for timesteps, not sure if it should be this way for other schedulers
            latents_noised = latents_original * scheduler.init_noise_sigma
            for step, time in enumerate(pbar):
                self.check_execution_state_change()

                latents_noised = latents_original
                noise_pred, noise_pred_x0 = predict_noise_sdxl_turbo(
                    unet_sdxl=unet,
                    latents_noised=latents_noised, # type: ignore
                    text_embeddings_unconditional=prompt_embeds_pos,
                    text_embeddings_unconditional_pooled=pooled_prompt_embeds_pos,
                    text_embeddings_unconditional_micro=micro_embeds_pos,
                    lora_scale=0.0,
                    t=time,
                    scheduler=scheduler,
                    reconstruction_loss=False,
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
