import torch
import numpy as np

from torch import Tensor
from typing import TypedDict, Optional, Literal, Dict, Any
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers import AutoencoderKL
from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from backend.loader.decorator import KatzukiNode
from backend.nodes.builtin import BaseNode


class SVDModelLoader(BaseNode):

    @KatzukiNode(node_type="diffusion.svd.model_loader")
    def __init__(self) -> None:
        pass

    class ReturnDict(TypedDict):
        image_encoder: CLIPVisionModelWithProjection
        feature_extractor: CLIPImageProcessor
        unet: UNetSpatioTemporalConditionModel
        vae: AutoencoderKLTemporalDecoder

    def execute(
            self,
            sd_path: Literal[
                "stabilityai/stable-video-diffusion-img2vid-xt",
            ] = "stabilityai/stable-video-diffusion-img2vid-xt",
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cuda"),
    ) -> ReturnDict:
        image_encoder: CLIPVisionModelWithProjection = CLIPVisionModelWithProjection.from_pretrained(
            sd_path,
            subfolder='image_encoder',
            device_map={'': 0},
            variant="fp16" if dtype == torch.float16 else None,
        ) # type: ignore
        feature_extractor: CLIPImageProcessor = CLIPImageProcessor.from_pretrained(
            sd_path,
            subfolder='feature_extractor',
            device_map={'': 0},
            variant="fp16" if dtype == torch.float16 else None,
        ) # type: ignore
        unet: UNetSpatioTemporalConditionModel = UNetSpatioTemporalConditionModel.from_pretrained(sd_path, subfolder="unet", variant="fp16" if dtype == torch.float16 else None) # type: ignore
        vae: AutoencoderKLTemporalDecoder = AutoencoderKLTemporalDecoder.from_pretrained(sd_path, subfolder="vae", variant="fp16" if dtype == torch.float16 else None) # type: ignore

        image_encoder = image_encoder.to(device=device, dtype=dtype) # type: ignore
        feature_extractor = feature_extractor.to(device=device, dtype=dtype) # type: ignore
        unet = unet.to(device=device, dtype=dtype)
        vae = vae.to(device=device, dtype=dtype)

        return self.ReturnDict(
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            unet=unet,
            vae=vae,
        )


class SVDImageEncode(BaseNode):

    @KatzukiNode(node_type="diffusion.svd.image_encode")
    def __init__(self) -> None:
        pass

    class ReturnDict(TypedDict):
        image_embeddings: Tensor

    # NOTE: static methods here are direct borrow from huggingface implementation

    @staticmethod
    def _compute_padding(kernel_size):
        """Compute padding tuple."""
        # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
        # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
        if len(kernel_size) < 2:
            raise AssertionError(kernel_size)
        computed = [k - 1 for k in kernel_size]

        # for even kernels we need to do asymmetric padding :(
        out_padding = 2 * len(kernel_size) * [0]

        for i in range(len(kernel_size)):
            computed_tmp = computed[-(i + 1)]

            pad_front = computed_tmp // 2
            pad_rear = computed_tmp - pad_front

            out_padding[2 * i + 0] = pad_front
            out_padding[2 * i + 1] = pad_rear

        return out_padding

    @staticmethod
    def _gaussian(window_size: int, sigma):
        if isinstance(sigma, float):
            sigma = torch.tensor([[sigma]])

        batch_size = sigma.shape[0]

        x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

        if window_size % 2 == 0:
            x = x + 0.5

        gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

        return gauss / gauss.sum(-1, keepdim=True)

    @staticmethod
    def _filter2d(input, kernel):
        # prepare kernel
        b, c, h, w = input.shape
        tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

        tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

        height, width = tmp_kernel.shape[-2:]

        padding_shape: list[int] = SVDImageEncode._compute_padding([height, width])
        input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

        # kernel and input tensor reshape to align element-wise or batch-wise params
        tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
        input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

        # convolve the tensor with the kernel.
        output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

        out = output.view(b, c, h, w)
        return out

    @staticmethod
    def _gaussian_blur2d(input: Tensor, kernel_size, sigma):
        if isinstance(sigma, tuple):
            sigma = torch.tensor([sigma], dtype=input.dtype)
        else:
            sigma = sigma.to(dtype=input.dtype)

        ky, kx = int(kernel_size[0]), int(kernel_size[1])
        bs = sigma.shape[0]
        kernel_x = SVDImageEncode._gaussian(kx, sigma[:, 1].view(bs, 1))
        kernel_y = SVDImageEncode._gaussian(ky, sigma[:, 0].view(bs, 1))
        out_x = SVDImageEncode._filter2d(input, kernel_x[..., None, :])
        out = SVDImageEncode._filter2d(out_x, kernel_y[..., None])

        return out

    @staticmethod
    def _resize_with_antialiasing(input: Tensor, size, interpolation="bicubic", align_corners=True):
        h, w = input.shape[-2:]
        factors = (h / size[0], w / size[1])

        # First, we have to determine sigma
        # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
        sigmas = (
            max((factors[0] - 1.0) / 2.0, 0.001),
            max((factors[1] - 1.0) / 2.0, 0.001),
        )

        # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
        # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
        # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
        ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

        # Make sure it is odd
        if (ks[0] % 2) == 0:
            ks = ks[0] + 1, ks[1]

        if (ks[1] % 2) == 0:
            ks = ks[0], ks[1] + 1

        input = SVDImageEncode._gaussian_blur2d(input, ks, sigmas)

        output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
        return output

    def execute(
        self,
        image: Tensor,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection,
    ) -> ReturnDict:
        with torch.no_grad():
            image = SVDImageEncode._resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0 # [-1, 1] -> [0, 1]

            # Normalize the image with for CLIP input
            image = feature_extractor(
                images=image,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values

            image_embeddings = image_encoder(image).image_embeds

            return self.ReturnDict(
                image_embeddings=image_embeddings, # [1, 1024]
            )


class RunSVDQuick(BaseNode):

    @KatzukiNode(node_type="diffusion.svd.run_svd_quick")
    def __init__(self) -> None:
        pass

    class ReturnDict(TypedDict):
        frames: np.ndarray

    def execute(
        self,
        image: Tensor, # should be resized to 1024, 576?
        generator: Optional[torch.Generator] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 8,
    ) -> ReturnDict:
        image = (image + 1.0) / 2.0
        # here image must be in [0, 1]
        
        # from diffusers.utils import load_image
        # # Load the conditioning image
        # image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
        
        # convert image to PIL
        image = image.squeeze() # [C, H, W]
        image = image.permute(1, 2, 0).detach().cpu().numpy() # [H, W, C]
        image = (image * 255).astype(np.uint8)
        from PIL import Image
        image = Image.fromarray(image)
        
        image = image.resize((width, height))

        from diffusers import StableVideoDiffusionPipeline
        pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        pipe.enable_model_cpu_offload()

        # Load the conditioning image
        generator = torch.manual_seed(42)
        video_frames: np.ndarray = pipe(image, decode_chunk_size=8, generator=generator, height=height, width=width,
                                        num_inference_steps=num_inference_steps).frames[0] # type: ignore

        video_frames = [np.array(frame) for frame in video_frames]
        video_frames = np.array(video_frames)
        
        # print the dimension of video frames
        print(video_frames.shape)
        
        # turn numpy to tensor
        video_frames = torch.from_numpy(video_frames)
        return self.ReturnDict(frames=video_frames,)
