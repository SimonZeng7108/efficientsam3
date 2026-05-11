"""构造 EfficientSAM3 原生 `BatchedDatapoint`。

SAM3 的训练入口不是普通 `(B,3,H,W)`，而是包含文本、几何 prompt、目标框、
mask 和 metadata 的 `BatchedDatapoint`。本文件把我们自己的少样本样本结构
转换成 SAM3 原生 dataclass，避免训练代码里堆满 shape 处理。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from sam3.model.data_misc import (
    BatchedDatapoint,
    BatchedFindTarget,
    BatchedInferenceMetadata,
    FindStage,
)


BoxCxCyWh = tuple[float, float, float, float]


@dataclass(frozen=True)
class FindBatchSample:
    """构造一个 find query 所需的最小样本。"""

    image: torch.Tensor
    image_id: str
    original_size: tuple[int, int]
    target_boxes: Sequence[BoxCxCyWh]
    target_masks: Sequence[torch.Tensor]
    prompt_boxes: Sequence[BoxCxCyWh]
    text: str


def build_batched_datapoint(
    samples: Sequence[FindBatchSample],
    device: torch.device,
) -> BatchedDatapoint:
    """把多个样本合成一个 SAM3 原生 batch。"""

    if not samples:
        raise ValueError("samples 不能为空")
    image_shapes = {tuple(sample.image.shape) for sample in samples}
    if len(image_shapes) != 1:
        raise ValueError("同一个 batch 内所有图片必须有相同的 CHW shape")

    img_batch = torch.stack([sample.image.to(device=device) for sample in samples], dim=0)
    text_batch = _unique_texts(samples)
    find_input = _build_find_stage(samples, text_batch, device)
    find_target = _build_find_target(samples, device)
    metadata = _build_metadata(samples, device)
    return BatchedDatapoint(
        img_batch=img_batch,
        find_text_batch=text_batch,
        find_inputs=[find_input],
        find_targets=[find_target],
        find_metadatas=[metadata],
        raw_images=None,
    )


def _unique_texts(samples: Sequence[FindBatchSample]) -> list[str]:
    """SAM3 文本 batch 去重，FindStage 通过 text_ids 引用。"""

    texts: list[str] = []
    for sample in samples:
        if sample.text not in texts:
            texts.append(sample.text)
    return texts


def _build_find_stage(
    samples: Sequence[FindBatchSample],
    text_batch: Sequence[str],
    device: torch.device,
) -> FindStage:
    """构造 FindStage。

    重点 shape：
    - input_boxes: (N_prompt_boxes, B_query, 4)
    - input_boxes_mask: (B_query, N_prompt_boxes)，True 表示 padding
    - input_boxes_label: (N_prompt_boxes, B_query)
    """

    batch_size = len(samples)
    max_prompts = max(len(sample.prompt_boxes) for sample in samples)
    input_boxes = torch.zeros((max_prompts, batch_size, 4), dtype=torch.float32, device=device)
    input_boxes_mask = torch.ones((batch_size, max_prompts), dtype=torch.bool, device=device)
    input_boxes_label = torch.zeros((max_prompts, batch_size), dtype=torch.long, device=device)
    object_ids: list[list[int]] = []

    for batch_index, sample in enumerate(samples):
        object_ids.append(list(range(len(sample.target_boxes))))
        for prompt_index, box in enumerate(sample.prompt_boxes):
            input_boxes[prompt_index, batch_index] = torch.tensor(box, dtype=torch.float32, device=device)
            input_boxes_mask[batch_index, prompt_index] = False
            input_boxes_label[prompt_index, batch_index] = 1

    return FindStage(
        img_ids=torch.arange(batch_size, dtype=torch.long, device=device),
        text_ids=torch.tensor([text_batch.index(sample.text) for sample in samples], dtype=torch.long, device=device),
        input_boxes=input_boxes,
        input_boxes_mask=input_boxes_mask,
        input_boxes_label=input_boxes_label,
        input_points=torch.empty((0, batch_size, 257), dtype=torch.float32, device=device),
        input_points_mask=torch.empty((batch_size, 0), dtype=torch.bool, device=device),
        object_ids=object_ids,
    )


def _build_find_target(samples: Sequence[FindBatchSample], device: torch.device) -> BatchedFindTarget:
    """构造 BatchedFindTarget。

    `boxes` 是 packed 表示，`boxes_padded` 是 matcher 需要的 padded 表示；
    两者都使用归一化 cxcywh。
    """

    num_boxes = torch.tensor([len(sample.target_boxes) for sample in samples], dtype=torch.long, device=device)
    max_boxes = int(num_boxes.max().item()) if len(num_boxes) else 0
    packed_boxes = [
        torch.tensor(box, dtype=torch.float32, device=device)
        for sample in samples
        for box in sample.target_boxes
    ]
    boxes = torch.stack(packed_boxes, dim=0) if packed_boxes else torch.zeros((0, 4), dtype=torch.float32, device=device)
    boxes_padded = torch.zeros((len(samples), max_boxes, 4), dtype=torch.float32, device=device)
    object_ids_padded = torch.full((len(samples), max_boxes), -1, dtype=torch.long, device=device)
    object_ids: list[torch.Tensor] = []
    for batch_index, sample in enumerate(samples):
        for object_index, box in enumerate(sample.target_boxes):
            boxes_padded[batch_index, object_index] = torch.tensor(box, dtype=torch.float32, device=device)
            object_ids_padded[batch_index, object_index] = object_index
            object_ids.append(torch.tensor(object_index, dtype=torch.long, device=device))

    packed_masks = [
        mask.to(device=device, dtype=torch.bool)
        for sample in samples
        for mask in sample.target_masks
    ]
    segments = torch.stack(packed_masks, dim=0) if packed_masks else torch.zeros((0, *samples[0].image.shape[-2:]), dtype=torch.bool, device=device)
    is_valid_segment = torch.ones((len(packed_masks),), dtype=torch.bool, device=device)
    packed_object_ids = torch.stack(object_ids, dim=0) if object_ids else torch.zeros((0,), dtype=torch.long, device=device)
    return BatchedFindTarget(
        num_boxes=num_boxes,
        boxes=boxes,
        boxes_padded=boxes_padded,
        repeated_boxes=torch.zeros((0, 4), dtype=torch.float32, device=device),
        segments=segments,
        semantic_segments=None,
        is_valid_segment=is_valid_segment,
        is_exhaustive=torch.ones((len(samples),), dtype=torch.bool, device=device),
        object_ids=packed_object_ids,
        object_ids_padded=object_ids_padded,
    )


def _build_metadata(samples: Sequence[FindBatchSample], device: torch.device) -> BatchedInferenceMetadata:
    """构造后处理所需 metadata。"""

    original_size = torch.tensor([sample.original_size for sample in samples], dtype=torch.long, device=device)
    batch_size = len(samples)
    image_ids = torch.arange(batch_size, dtype=torch.long, device=device)
    return BatchedInferenceMetadata(
        coco_image_id=image_ids,
        original_image_id=image_ids,
        original_category_id=torch.ones((batch_size,), dtype=torch.int, device=device),
        original_size=original_size,
        object_id=torch.zeros((batch_size,), dtype=torch.long, device=device),
        frame_index=torch.zeros((batch_size,), dtype=torch.long, device=device),
        is_conditioning_only=[False for _ in samples],
    )
