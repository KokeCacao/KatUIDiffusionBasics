import os
import torch
import numpy as np
import base64
import urllib.request
import urllib.error

from torch import Tensor
from typing import TypedDict, Optional, Literal, Tuple, Union, List, Callable
from PIL import Image, ImageOps
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
from diffusers import AutoencoderKL
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from backend.loader.decorator import KatzukiNode
from backend.nodes.builtin import BaseNode
from backend.constant import WORKFLOWS_PATH
from transformers import CLIPTokenizer, CLIPTextModel
from io import BytesIO

from kokikit.optimizers import Adan as AdaN
from kokikit.nerf import trunc_exp

from nodes.KatUIDiffusionBasics.image import ImageResize


class VAELoader(BaseNode):

    @KatzukiNode(node_type="diffusion.basic.vae_loader")
    def __init__(self) -> None:
        pass

    def execute(
            self,
            path: str = "stabilityai/stable-diffusion-2-1-base",
            subfolder: str = "vae",
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cuda"),
    ) -> AutoencoderKL:
        vae: AutoencoderKL = AutoencoderKL.from_pretrained(path, subfolder=subfolder, torch_dtype=dtype) # type: ignore
        vae = vae.to(device=device)
        return vae


class UNetLoader(BaseNode):

    @KatzukiNode(node_type="diffusion.basic.unet_loader")
    def __init__(self) -> None:
        pass

    def execute(
            self,
            path: str = "stabilityai/stable-diffusion-2-1-base",
            subfolder: str = "unet",
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cuda"),
    ) -> UNet2DConditionModel:
        unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(path, subfolder=subfolder, torch_dtype=dtype) # type: ignore
        unet = unet.to(device=device)
        return unet


class SchedulerLoader(BaseNode):

    @KatzukiNode(node_type="diffusion.basic.scheduler_loader")
    def __init__(self) -> None:
        pass

    def execute(
        self,
        scheduler_type: Literal["ddim", "pndm", "euler"] = "ddim",
        path: Literal["kxic/zero123-xl", "stabilityai/stable-diffusion-2-1-base", "runwayml/stable-diffusion-v1-5", "stabilityai/sdxl-turbo"] = "kxic/zero123-xl",
        dtype: torch.dtype = torch.float32,
    ) -> SchedulerMixin:
        if path == "kxic/zero123-xl":
            scheduler_type = "ddim"
        elif path == "stabilityai/stable-diffusion-2-1-base" or path == "runwayml/stable-diffusion-v1-5":
            scheduler_type = "pndm"
        elif path == "stabilityai/sdxl-turbo":
            scheduler_type = "euler"
        else:
            raise ValueError(f"Path {path} is not supported")

        if scheduler_type == "ddim":
            scheduler: DDIMScheduler = DDIMScheduler.from_pretrained(path, subfolder="scheduler", torch_dtype=dtype) # type: ignore
        elif scheduler_type == "pndm":
            scheduler: PNDMScheduler = PNDMScheduler.from_pretrained(path, subfolder="scheduler", torch_dtype=dtype) # type: ignore
        elif scheduler_type == "euler":
            scheduler: EulerAncestralDiscreteScheduler = EulerAncestralDiscreteScheduler.from_pretrained(path, subfolder="scheduler", torch_dtype=dtype)
        else:
            raise ValueError(f"Scheduler type {scheduler_type} is not supported")

        scheduler.betas = scheduler.betas.to(dtype=dtype)
        scheduler.alphas = scheduler.alphas.to(dtype=dtype)
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(dtype=dtype)

        return scheduler


class ImageLoader(BaseNode):

    @KatzukiNode(node_type="diffusion.basic.image_loader")
    def __init__(self) -> None:
        pass

    class ReturnDict(TypedDict):
        image: Tensor
        shape: torch.Size

    @staticmethod
    def load_remote_image(url: str) -> Optional[Image.Image]:
        try:
            req = urllib.request.Request(url, data=None, headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'})
            return Image.open(urllib.request.urlopen(req))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            else:
                raise e

    def execute(
            self,
            image_path: str,
            resize_hw: Optional[Tuple[int, ...]] = None, # TODO: change it to Tuple[int, int] when typing but is fixed
            resize_mode: Literal['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area', 'nearest-exact'] = 'bicubic',
            resize_align_corners: bool = True,
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cuda"),
    ) -> ReturnDict:

        header, separator, base64_data = image_path.partition(';base64,')
        if header.startswith("data:image/"):
            # base64 image
            binary_data = base64.b64decode(base64_data)
            image_data = BytesIO(binary_data)
            image = Image.open(image_data)
        elif "http" in image_path:
            # http or https image
            image: Optional[Image.Image] = self.load_remote_image(image_path)
        else:
            # local path
            image: Optional[Image.Image] = Image.open(image_path)

        assert image is not None, f"Failed to load image from {image_path}"

        image = ImageOps.exif_transpose(image)
        if image.mode == "RGBA":
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])
        elif image.mode == "RGB":
            rgb_image = image
        else:
            raise ValueError(f"Image mode {image.mode} is not supported")

        rgb_image = torch.from_numpy(np.array(rgb_image)).permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]
        rgb_image = rgb_image.to(dtype=dtype, device=device) / 127.5 - 1.0 # [0, 255] -> [-1, 1]
        assert torch.max(rgb_image) <= 1.0 and torch.min(rgb_image) >= -1.0, f"Image {image_path} is not in range [-1, 1]"

        if resize_hw is not None:
            rgb_image = ImageResize().execute(
                image=rgb_image,
                height=resize_hw[0],
                width=resize_hw[1],
                resize_mode=resize_mode,
                resize_align_corners=resize_align_corners,
            )

        return self.ReturnDict(
            image=rgb_image, # [1, 3, H, W]
            shape=rgb_image.shape, # [1, 3, H, W]
        )


class LatentImage(BaseNode):

    @KatzukiNode(node_type="diffusion.basic.latent_image")
    def __init__(self) -> None:
        pass

    def execute(
            self,
            random_seed: Optional[int] = None,
            image_height: int = 512,
            image_width: int = 512,
            batch_size: int = 1,
            channel: int = 4,
            vae_scale: int = 8,
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cuda"),
    ) -> Tensor:
        if random_seed is None:
            return torch.randn(batch_size, channel, image_height // vae_scale, image_width // vae_scale, dtype=dtype, device=device)
        torch.manual_seed(random_seed)
        return torch.randn(batch_size, channel, image_height // vae_scale, image_width // vae_scale, dtype=dtype, device=device)


class VAEDecode(BaseNode):

    @KatzukiNode(node_type="diffusion.basic.vae_decode")
    def __init__(self) -> None:
        pass

    def execute(self, latents: Tensor, vae: AutoencoderKL, to_image: bool = False) -> Tensor:
        image_batch = vae.decode(1 / vae.config['scaling_factor'] * latents.clone().detach()).sample # [B, C, H, W] # type: ignore
        image_batch = image_batch.clamp(-1, 1)

        if to_image:
            image_batch = (image_batch / 2 + 0.5).clamp(0, 1) * 255
            image_batch = image_batch.permute(0, 2, 3, 1).detach().cpu() # [B, C, H, W] -> [B, H, W, C]
        return image_batch


class TextEncode(BaseNode):

    @KatzukiNode(
        node_type="diffusion.basic.text_encode",
        input_description={
            "prompt": "The prompt to encode. Default to negative prompt.",
            "tokenizer": "The tokenizer model to use.",
            "text_encoder": "The text encoder model to use.",
            "tokenizer_path": "If tokenizer is None, then we use this path to load the tokenizer.",
            "tokenizer_subfolder": "The subfolder of the tokenizer within tokenizer_path.",
            "text_encoder_path": "If text_encoder is None, then we use this path to load the text encoder.",
            "text_encoder_subfolder": "The subfolder of the text encoder within text_encoder_path.",
            "device": "Device object. Default to cuda."
        },
        output_description={
            "text_embeds": "The text embeddings. If prompt is a string, then it is a tensor of shape [77, 768]. If prompt is a list of string, then it is a list of tensors of shape [77, 768].",
        },
    )
    def __init__(self) -> None:
        pass

    class ReturnDict(TypedDict):
        text_embeds: Union[List[Tensor], Tensor]

    def execute(
            self,
            prompts: Union[List[str], str] = "ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions",
            tokenizer: Optional[CLIPTokenizer] = None,
            text_encoder: Optional[CLIPTextModel] = None,
            tokenizer_path: str = "openai/clip-vit-large-patch14",
            tokenizer_subfolder: Optional[str] = None,
            text_encoder_path: str = "openai/clip-vit-large-patch14",
            text_encoder_subfolder: Optional[str] = None,
            device: torch.device = torch.device("cuda"),
    ) -> ReturnDict:
        if isinstance(prompts, str):
            return_type = "tensor"
            batch_size = 1
            prompts = [prompts]
        elif isinstance(prompts, list):
            return_type = "list"
            batch_size = len(prompts)
        else:
            raise ValueError(f"Prompt type {type(prompts)} is not supported")

        # load tokenizer
        if tokenizer is None:
            if tokenizer_subfolder is not None:
                tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, subfolder=tokenizer_subfolder)
            else:
                tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)

        # load text encoder
        if text_encoder is None:
            if text_encoder_subfolder is not None:
                text_encoder = CLIPTextModel.from_pretrained(
                    text_encoder_path,
                    subfolder=text_encoder_subfolder,
                    device_map={'': 0} # BUG: https://github.com/tloen/alpaca-lora/issues/368 (NotImplementedError: Cannot copy out of meta tensor; no data!)
                    # BUG: solution: https://huggingface.co/docs/transformers/main_classes/model
                    ,
                ) # type: ignore
            else:
                text_encoder = CLIPTextModel.from_pretrained(text_encoder_path, device_map={'': 0}) # type: ignore

        # put text encoder to device
        text_encoder = text_encoder.to(device=device) # type: ignore

        text_inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompts, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1:-1])
            print(f"The following part of your input was truncated because CLIP can only handle sequences up to"
                  f" {tokenizer.model_max_length} tokens: {removed_text}")

        prompt_embeds = text_encoder(text_input_ids.to(device),)
        prompt_embeds = prompt_embeds[0]

        if return_type == "tensor":
            return self.ReturnDict(text_embeds=prompt_embeds[0])
        elif return_type == "list":
            return self.ReturnDict(text_embeds=[prompt_embeds[i] for i in range(batch_size)])
        else:
            raise ValueError(f"Return type {return_type} is not supported")


class DType(BaseNode):

    @KatzukiNode(node_type="diffusion.basic.dtype")
    def __init__(self) -> None:
        pass

    def execute(self, dtype: Literal['float16', 'float32'] = 'float32') -> torch.dtype:
        return torch.float16 if dtype == 'float16' else torch.float32


class Device(BaseNode):

    @KatzukiNode(node_type="diffusion.basic.device")
    def __init__(self) -> None:
        pass

    def execute(self, device: Literal['cuda', 'cpu'] = 'cuda') -> torch.device:
        return torch.device('cuda') if device == 'cuda' else torch.device('cpu')


class Parameter(BaseNode):

    @KatzukiNode(node_type="diffusion.basic.parameter")
    def __init__(self) -> None:
        pass

    def execute(
            self,
            value: List[float] = [1.0],
            requires_grad: bool = True,
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cuda"),
    ) -> torch.nn.Parameter:
        return torch.nn.Parameter(
            torch.tensor(value, dtype=dtype, device=device),
            requires_grad=requires_grad,
        )


class Adam(BaseNode):

    @KatzukiNode(node_type="diffusion.basic.optimizer.adam")
    def __init__(self) -> None:
        pass

    def execute(
            self,
            params: List[torch.nn.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, ...] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.0,
            amsgrad: bool = False,
    ) -> torch.optim.Adam:
        return torch.optim.Adam(
            params=params,
            lr=lr,
            betas=betas, # type: ignore # TODO
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )


class AdamW(BaseNode):

    @KatzukiNode(node_type="diffusion.basic.optimizer.adamw")
    def __init__(self) -> None:
        pass

    def execute(
            self,
            params: List[torch.nn.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, ...] = (0.9, 0.99),
            eps: float = 1e-15,
            weight_decay: float = 0.01,
            amsgrad: bool = False,
    ) -> torch.optim.AdamW:
        return torch.optim.AdamW(
            params=params,
            lr=lr,
            betas=betas, # type: ignore # TODO
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )


class SGD(BaseNode):

    @KatzukiNode(node_type="diffusion.basic.optimizer.sgd")
    def __init__(self) -> None:
        pass

    def execute(
        self,
        params: List[torch.nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> torch.optim.SGD:
        return torch.optim.SGD(
            params=params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )


class Adan(BaseNode):

    @KatzukiNode(node_type="diffusion.basic.optimizer.adan")
    def __init__(self) -> None:
        pass

    def execute(
            self,
            params: List[torch.nn.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, ...] = (0.98, 0.92, 0.99),
            eps: float = 1e-8,
            weight_decay: float = 0.0,
            max_grad_norm: float = 0.0,
            no_prox: bool = False,
            foreach: bool = True,
    ) -> AdaN:
        return AdaN(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            no_prox=no_prox,
            foreach=foreach,
        )


class SoftMax(BaseNode):

    @KatzukiNode(node_type="diffusion.basic.activation.softmax")
    def __init__(self) -> None:
        pass

    def execute(self, dim: Optional[int] = None) -> torch.nn.Softmax:
        return torch.nn.Softmax(dim=dim)


class Tanh(BaseNode):

    @KatzukiNode(node_type="diffusion.basic.activation.tanh")
    def __init__(self) -> None:
        pass

    def execute(self, dim: Optional[int] = None) -> torch.nn.Tanh:
        return torch.nn.Tanh(dim=dim)


class Sigmoid(BaseNode):

    @KatzukiNode(node_type="diffusion.basic.activation.sigmoid")
    def __init__(self) -> None:
        pass

    def execute(self) -> torch.nn.Sigmoid:
        return torch.nn.Sigmoid()


class SoftPlus(BaseNode):

    @KatzukiNode(node_type="diffusion.basic.activation.softplus")
    def __init__(self) -> None:
        pass

    def execute(self, beta: int = 1, threshold: int = 20) -> torch.nn.Softplus:
        return torch.nn.Softplus(beta=beta, threshold=threshold)


class ReLU(BaseNode):

    @KatzukiNode(node_type="diffusion.basic.activation.relu")
    def __init__(self) -> None:
        pass

    def execute(self, inplace: bool = False) -> torch.nn.ReLU:
        return torch.nn.ReLU(inplace=inplace)


class Exponential(BaseNode):

    @KatzukiNode(node_type="diffusion.basic.activation.exponential")
    def __init__(self) -> None:
        pass

    def execute(self) -> Callable:
        return trunc_exp


class SaveStateDict(BaseNode):

    @KatzukiNode(node_type="diffusion.basic.save_state_dict")
    def __init__(self) -> None:
        pass

    class ReturnDict(TypedDict):
        path: str

    def execute(self, module: torch.nn.Module, path: str = str(WORKFLOWS_PATH / "state_dict.pth")) -> ReturnDict:
        state_dict = module.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.isdir(path):
            path = os.path.join(path, f"{module.__class__.__name__}.pth")
        torch.save(state_dict, path)
        return self.ReturnDict(path=path)


class LoadStateDict(BaseNode):

    @KatzukiNode(node_type="diffusion.basic.load_state_dict")
    def __init__(self) -> None:
        pass

    def execute(self, module: torch.nn.Module, path: str = str(WORKFLOWS_PATH / "state_dict.pth")) -> torch.nn.Module:
        state_dict = torch.load(path)
        module.load_state_dict(state_dict)
        return module
