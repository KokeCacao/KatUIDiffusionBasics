import torch

from torch import Tensor
from typing import Literal, TypedDict, Optional, Dict, Any, Union, List, Tuple
from diffusers.models.controlnet import ControlNetModel, ControlNetOutput
from backend.loader.decorator import KatzukiNode
from backend.nodes.builtin import BaseNode


class ControlNetCondition(BaseNode):

    @KatzukiNode(node_type="diffusion.sdxl.control_net_condition")
    def __init__(self) -> None:
        pass

    class ReturnDict(TypedDict):
        controlnet_condition: Dict[str, Any]

    def execute(
        self,
        controlnet: Optional[Union[ControlNetModel, List[ControlNetModel]]],
        image: Optional[Union[Tensor, List[Tensor]]],
        conditioning_scale: Optional[Union[float, List[float]]] = 1.0,
        guess_mode: Optional[Union[bool, List[bool]]] = False,
        condition_side_control: bool = True,
        uncondition_side_control: bool = False,
    ) -> ReturnDict:
        return self.ReturnDict(controlnet_condition={
            "controlnet": controlnet,
            "image": image,
            "conditioning_scale": conditioning_scale,
            "guess_mode": guess_mode,
            "condition_side_control": condition_side_control,
            "uncondition_side_control": uncondition_side_control,
        })


class ControlNetModelLoader(BaseNode):

    @KatzukiNode(node_type="diffusion.controlnet.control_net_model_loader")
    def __init__(self) -> None:
        pass

    class ReturnDict(TypedDict):
        unet: ControlNetModel

    def execute(self, controlnet_path: Literal[
        "lllyasviel/control_v11p_sd15_openpose",
        "lllyasviel/control_v11p_sd15_inpaint",
        "lllyasviel/control_v11e_sd15_ip2p",
        "lllyasviel/control_v11f1e_sd15_tile",
        "lllyasviel/control_v11e_sd15_shuffle",
        "lllyasviel/control_v11p_sd15_softedge",
        "lllyasviel/control_v11p_sd15_scribble",
        "lllyasviel/control_v11p_sd15s2_lineart_anime",
        "lllyasviel/control_v11p_sd15_lineart",
        "lllyasviel/control_v11p_sd15_seg",
        "lllyasviel/control_v11p_sd15_normalbae",
        "lllyasviel/control_v11f1p_sd15_depth",
        "lllyasviel/control_v11p_sd15_mlsd",
        "lllyasviel/control_v11p_sd15_canny",
    ] = "lllyasviel/control_v11e_sd15_ip2p", dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cuda")) -> ReturnDict:
        controlnet: ControlNetModel = ControlNetModel.from_pretrained(controlnet_path) # type: ignore

        # monkey patch since lllyasviel's controlnet takes in image in [0, 1] range
        # however, stable diffusion takes in image in [-1, 1] range and [-1, 1] is Katzuki's convention for all tensor images
        original_forward = controlnet.forward

        def _forward(self, *args, **kwargs) -> Union[ControlNetOutput, Tuple[Tuple[torch.FloatTensor, ...], torch.FloatTensor]]:
            image = kwargs["controlnet_cond"]
            image = image * 0.5 + 0.5
            kwargs["controlnet_cond"] = image
            return original_forward(*args, **kwargs)

        setattr(controlnet, 'forward', _forward.__get__(controlnet, ControlNetModel))

        controlnet = controlnet.to(device=device, dtype=dtype)
        return self.ReturnDict(unet=controlnet)


class ControlNetXLModelLoader(BaseNode):

    @KatzukiNode(node_type="diffusion.controlnet.control_net_xl_model_loader")
    def __init__(self) -> None:
        pass

    class ReturnDict(TypedDict):
        unet: ControlNetModel

    def execute(
            self,
            controlnet_path: Literal[
                "diffusers/controlnet-canny-sdxl-1.0",
                "diffusers/controlnet-depth-sdxl-1.0",
                "diffusers/controlnet-zoe-depth-sdxl-1.0",
                "diffusers/controlnet-canny-sdxl-1.0-small",
                "diffusers/controlnet-depth-sdxl-1.0-small",
            ] = "diffusers/controlnet-depth-sdxl-1.0-small",
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cuda"),
    ) -> ReturnDict:
        controlnet: ControlNetModel = ControlNetModel.from_pretrained(controlnet_path) # type: ignore

        # monkey patch since lllyasviel's controlnet takes in image in [0, 1] range
        # however, stable diffusion takes in image in [-1, 1] range and [-1, 1] is Katzuki's convention for all tensor images
        original_forward = controlnet.forward

        def _forward(self, *args, **kwargs) -> Union[ControlNetOutput, Tuple[Tuple[torch.FloatTensor, ...], torch.FloatTensor]]:
            image = kwargs["controlnet_cond"]
            image = image * 0.5 + 0.5
            kwargs["controlnet_cond"] = image
            return original_forward(*args, **kwargs)

        setattr(controlnet, 'forward', _forward.__get__(controlnet, ControlNetModel))

        controlnet = controlnet.to(device=device, dtype=dtype)
        return self.ReturnDict(unet=controlnet)
