import torch

from typing import Literal
from torch import Tensor
from backend.loader.decorator import KatzukiNode
from backend.nodes.builtin import BaseNode


class ImageResize(BaseNode):

    @KatzukiNode(node_type="diffusion.image.resize")
    def __init__(self) -> None:
        pass

    def execute(
        self,
        image: Tensor, # [B, C, H, W]
        height: int,
        width: int,
        resize_mode: Literal['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area', 'nearest-exact'] = 'bicubic',
        resize_align_corners: bool = True,
    ) -> Tensor:

        if resize_mode in ['linear', 'bilinear', 'bicubic', 'trilinear']:
            rgb_image = torch.nn.functional.interpolate(image, size=(height, width), mode=resize_mode, align_corners=resize_align_corners)
        else:
            rgb_image = torch.nn.functional.interpolate(image, size=(height, width), mode=resize_mode)
        rgb_image = rgb_image.clamp(-1, 1) # BUG: for some reason bicubic interpolation will cause values to be out of range [-1, 1]

        return torch.nn.functional.interpolate(image, size=(height, width), mode="bilinear", align_corners=False)


class ToGrayScale(BaseNode):

    @KatzukiNode(node_type="diffusion.image.to_grayscale")
    def __init__(self) -> None:
        pass

    def execute(
            self,
            image: Tensor # [B, C, H, W]
    ) -> Tensor:
        return image.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)


class EdgeDetection(BaseNode):

    @KatzukiNode(node_type="diffusion.image.edge_detection")
    def __init__(self) -> None:
        pass

    @staticmethod
    def sobel_edge_detection(image_tensor: Tensor) -> Tensor:
        """
        Apply Sobel edge detection to an image tensor of shape [B, C, H, W] with values in range [-1, 1].
        :param image_tensor: A PyTorch Tensor of shape [B, C, H, W]
        :return: Edge-detected image tensor
        """

        # Define the Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(1, image_tensor.size(1), 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(1, image_tensor.size(1), 1, 1)

        # Check if CUDA is available and move tensors to GPU if it is
        sobel_x = sobel_x.cuda()
        sobel_y = sobel_y.cuda()
        image_tensor = image_tensor.cuda()

        # Apply the Sobel filters
        edge_x = torch.nn.functional.conv2d(image_tensor, sobel_x, padding=1)
        edge_y = torch.nn.functional.conv2d(image_tensor, sobel_y, padding=1)

        # Calculate the magnitude of the gradients
        edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)

        # Normalize the output image to be in the range [-1, 1]
        edge_magnitude = edge_magnitude / edge_magnitude.max() * 2 - 1

        return edge_magnitude

    def execute(
            self,
            image: Tensor # [B, C, H, W]
    ) -> Tensor:
        return self.sobel_edge_detection(image).repeat(1, 3, 1, 1)
