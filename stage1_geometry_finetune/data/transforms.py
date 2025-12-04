import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import (
    resize,
    to_pil_image,
    InterpolationMode,
)
from copy import deepcopy
from typing import Tuple


class ResizeLongestSide:
    """
    Utility that mirrors the behaviour used in SAM/SAM2 preprocessing.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        target_size = self.get_preprocess_shape(
            image.shape[0], image.shape[1], self.target_length
        )
        return np.array(resize(to_pil_image(image), target_size))

    def apply_masks(self, mask, original_size):
        assert mask.shape[1:] == original_size, "image and gt_mask size must match"
        target_size = self.get_preprocess_shape(
            mask.shape[1], mask.shape[2], self.target_length
        )
        return np.array(
            resize(torch.from_numpy(mask), target_size, InterpolationMode.NEAREST)
        )

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(old_h, old_w, self.target_length)
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        target_size = self.get_preprocess_shape(
            image.shape[2], image.shape[3], self.target_length
        )
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_masks_torch(self, mask, original_size):
        assert mask.shape[1:] == original_size, "image and gt_mask size must match"
        target_size = self.get_preprocess_shape(
            mask.shape[1], mask.shape[2], self.target_length
        )
        return F.interpolate(mask[:, None], target_size, mode="nearest").squeeze(1)

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(old_h, old_w, self.target_length)
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


def xywh2xyxy(xywh: torch.Tensor) -> torch.Tensor:
    top_left = xywh[:, :2]
    bottom_right = xywh[:, :2] + xywh[:, 2:]
    return torch.cat((top_left, bottom_right), dim=1)


def xyxy2xywh(xyxy: torch.Tensor) -> torch.Tensor:
    top_left = xyxy[:, :2]
    width = xyxy[:, 2] - xyxy[:, 0]
    height = xyxy[:, 3] - xyxy[:, 1]
    return torch.cat((top_left, width[:, None], height[:, None]), dim=1)


