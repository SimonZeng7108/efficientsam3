"""EfficientSAM3 原生训练 batch 构造工具。

本模块负责把项目自己的 `Annotation` 转成 SAM3 训练链路需要的张量。
这里不引入候选框概念；每个训练样本只有图片和该图片上的真值目标。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from ..utils.torch import require_torch
from .masks import annotation_to_mask
from .models import HBB, Annotation, TrainingSample, normalize_annotation


@dataclass(frozen=True)
class NativeImageBatch:
    """送入 EfficientSAM3 图像 backbone 的最小 batch。

    `images` 是 torch.Tensor，形状为 `(B, 3, resolution, resolution)`；
    这里类型写成 `Any`，是为了让无 torch 环境仍能导入本模块。
    """

    image_ids: list[str]
    images: Any
    original_sizes: list[tuple[int, int]]


@dataclass(frozen=True)
class NativeSam3Batch:
    """一次 SAM3 原生训练所需的输入。

    `find_stage` 只保存图片索引和空文本索引；真实目标信息在
    `find_target` 中，后续交给 SAM3 原生 matcher/loss 使用。
    """

    image_batch: NativeImageBatch
    find_stage: Any
    find_target: Any
    annotations_by_image: dict[str, list[Annotation]]


class Sam3BatchBuilder:
    """SAM3 原生训练 batch 构造门面。"""

    def build_training_batch(
        self,
        annotations: Sequence[Annotation],
        image_map: Mapping[str, str | Path],
        *,
        resolution: int = 1008,
        device: str = "cuda",
        include_masks: bool = False,
    ) -> NativeSam3Batch:
        return build_sam3_training_batch(
            annotations,
            image_map,
            resolution=resolution,
            device=device,
            include_masks=include_masks,
        )

    def build_training_batch_from_samples(
        self,
        samples: Sequence[TrainingSample],
        image_map: Mapping[str, str | Path],
        *,
        resolution: int = 1008,
        device: str = "cuda",
        include_masks: bool = False,
    ) -> NativeSam3Batch:
        return build_sam3_training_batch_from_samples(
            samples,
            image_map,
            resolution=resolution,
            device=device,
            include_masks=include_masks,
        )

    def load_image_batch(
        self,
        image_ids: Sequence[str],
        image_map: Mapping[str, str | Path],
        *,
        resolution: int,
        device: str,
    ) -> NativeImageBatch:
        return load_image_batch(
            image_ids,
            image_map,
            resolution=resolution,
            device=device,
        )


def group_annotations_by_image(
    annotations: Sequence[Annotation],
) -> "OrderedDict[str, list[Annotation]]":
    """按 image_id 分组，并保留图片第一次出现的顺序。"""
    grouped: "OrderedDict[str, list[Annotation]]" = OrderedDict()
    for annotation in annotations:
        grouped.setdefault(annotation.image_id, []).append(annotation)
    return grouped


def group_training_samples_by_image(
    samples: Sequence[TrainingSample],
) -> "OrderedDict[str, list[Annotation]]":
    """按图片分组训练样本；负样本保留 image_id，但目标列表为空。"""
    grouped: "OrderedDict[str, list[Annotation]]" = OrderedDict()
    for sample in samples:
        grouped.setdefault(sample.image_id, []).extend(sample.annotations)
    return grouped


def hbb_to_cxcywh_norm(hbb: HBB, *, width: int, height: int) -> tuple[float, float, float, float]:
    """把像素 HBB 转成 SAM3 使用的归一化 `cx, cy, w, h`。"""
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    x1 = _clamp(hbb.x1, 0.0, float(width))
    y1 = _clamp(hbb.y1, 0.0, float(height))
    x2 = _clamp(hbb.x2, 0.0, float(width))
    y2 = _clamp(hbb.y2, 0.0, float(height))
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    box_w = max(0.0, right - left)
    box_h = max(0.0, bottom - top)
    cx = left + box_w / 2
    cy = top + box_h / 2
    return (
        _round_float(cx / width),
        _round_float(cy / height),
        _round_float(box_w / width),
        _round_float(box_h / height),
    )


def annotation_to_target_box(
    annotation: Annotation,
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """把任意支持的标注转成 SAM3 target box。

    第一版训练用 SAM3 原生 box loss，因此 polygon/OBB 会先派生 HBB。
    后续 mask loss 和 OBB 后处理可以继续使用原始 polygon 信息。
    """
    normalized = normalize_annotation(annotation)
    if normalized.hbb is None:
        raise ValueError(f"annotation has no usable box: {annotation.object_id}")
    return hbb_to_cxcywh_norm(normalized.hbb, width=width, height=height)


def load_image_batch(
    image_ids: Sequence[str],
    image_map: Mapping[str, str | Path],
    *,
    resolution: int,
    device: str,
) -> NativeImageBatch:
    """读取并归一化图片，变成 EfficientSAM3 backbone 输入。

    预处理保持和 `Sam3Processor` 一致：resize 到正方形、缩放到 `[0,1]`、
    再用 mean/std 为 0.5 的方式归一化到约 `[-1,1]`。
    """
    torch = require_torch()
    tensors = []
    original_sizes: list[tuple[int, int]] = []
    for image_id in image_ids:
        if image_id not in image_map:
            raise KeyError(f"image_id not found in image_map: {image_id}")
        tensor, size = _load_one_image_tensor(
            image_map[image_id],
            resolution=resolution,
            torch=torch,
        )
        tensors.append(tensor)
        original_sizes.append(size)
    images = torch.stack(tensors, dim=0).to(device)
    return NativeImageBatch(
        image_ids=[str(image_id) for image_id in image_ids],
        images=images,
        original_sizes=original_sizes,
    )


def build_sam3_training_batch(
    annotations: Sequence[Annotation],
    image_map: Mapping[str, str | Path],
    *,
    resolution: int = 1008,
    device: str = "cuda",
    include_masks: bool = False,
) -> NativeSam3Batch:
    """构造一次少样本训练 batch。

    该函数只使用图片和真值，不需要 proposal。为了让少样本迭代足够简单，
    当前实现把同一批标注按图片分组后一次送入 SAM3；如果显存不足，上层
    训练循环会按图片切小 batch。
    """
    if not annotations:
        raise ValueError("annotations must not be empty")
    samples = [
        TrainingSample(
            image_id=image_id,
            label=image_annotations[0].label,
            annotations=list(image_annotations),
        )
        for image_id, image_annotations in group_annotations_by_image(annotations).items()
    ]
    return build_sam3_training_batch_from_samples(
        samples,
        image_map,
        resolution=resolution,
        device=device,
        include_masks=include_masks,
    )


def build_sam3_training_batch_from_samples(
    samples: Sequence[TrainingSample],
    image_map: Mapping[str, str | Path],
    *,
    resolution: int = 1008,
    device: str = "cuda",
    include_masks: bool = False,
) -> NativeSam3Batch:
    """构造支持正样本和 no-object 负样本的 SAM3 原生训练 batch。"""
    if not samples:
        raise ValueError("training samples must not be empty")
    torch = require_torch()
    from sam3.model.data_misc import BatchedFindTarget

    grouped = group_training_samples_by_image(samples)
    image_ids = list(grouped)
    image_batch = load_image_batch(
        image_ids,
        image_map,
        resolution=resolution,
        device=device,
    )
    find_stage = _build_find_stage(
        batch_size=len(image_ids),
        torch=torch,
        device=device,
    )
    find_target = _build_find_target(
        grouped,
        image_batch.original_sizes,
        target_cls=BatchedFindTarget,
        torch=torch,
        device=device,
        include_masks=include_masks,
        resolution=resolution,
    )
    return NativeSam3Batch(
        image_batch=image_batch,
        find_stage=find_stage,
        find_target=find_target,
        annotations_by_image=dict(grouped),
    )


def _build_find_stage(*, batch_size: int, torch: Any, device: str) -> Any:
    """构造 SAM3 find stage。

    这里文本 id 全部为 0，必须配合 `NativeEfficientSAM3FewShotModel`
    调用；该 wrapper 会注入空 language feature，避免真的调用文本编码器。
    """
    from sam3.model.data_misc import FindStage

    return FindStage(
        img_ids=torch.arange(batch_size, device=device, dtype=torch.long),
        text_ids=torch.zeros(batch_size, device=device, dtype=torch.long),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )


def _build_find_target(
    grouped: Mapping[str, list[Annotation]],
    original_sizes: Sequence[tuple[int, int]],
    *,
    target_cls: Any,
    torch: Any,
    device: str,
    include_masks: bool = False,
    resolution: int | None = None,
) -> Any:
    """构造 SAM3 `BatchedFindTarget`。

    `boxes` 是 packed 表达，形状为 `(sum_targets, 4)`；
    `boxes_padded` 是 `(B, max_targets_per_image, 4)`，供
    `BinaryHungarianMatcherV2` 高效匹配使用。
    """
    if include_masks and resolution is None:
        raise ValueError("resolution must be provided when include_masks=True")

    per_image_boxes: list[list[tuple[float, float, float, float]]] = []
    flat_masks: list[np.ndarray] = []
    for annotations, (width, height) in zip(grouped.values(), original_sizes):
        boxes = []
        for annotation in annotations:
            boxes.append(annotation_to_target_box(annotation, width=width, height=height))
            if include_masks:
                # mask 与 box 使用相同遍历顺序，保证 packed target 中一一对应。
                flat_masks.append(
                    annotation_to_mask(
                        annotation,
                        width=width,
                        height=height,
                        resolution=int(resolution),
                    )
                )
        per_image_boxes.append(boxes)

    num_boxes = torch.tensor(
        [len(boxes) for boxes in per_image_boxes],
        dtype=torch.long,
        device=device,
    )
    flat_boxes = [box for boxes in per_image_boxes for box in boxes]
    if flat_boxes:
        boxes_tensor = torch.tensor(flat_boxes, dtype=torch.float32, device=device)
    else:
        boxes_tensor = torch.empty((0, 4), dtype=torch.float32, device=device)
    max_boxes = max((len(boxes) for boxes in per_image_boxes), default=0)
    boxes_padded = torch.zeros(
        (len(per_image_boxes), max_boxes, 4),
        dtype=torch.float32,
        device=device,
    )
    object_ids_padded = torch.full(
        (len(per_image_boxes), max_boxes),
        -1,
        dtype=torch.long,
        device=device,
    )
    next_object_id = 0
    object_ids: list[int] = []
    for batch_index, boxes in enumerate(per_image_boxes):
        for box_index, box in enumerate(boxes):
            boxes_padded[batch_index, box_index] = torch.tensor(
                box,
                dtype=torch.float32,
                device=device,
            )
            object_ids_padded[batch_index, box_index] = next_object_id
            object_ids.append(next_object_id)
            next_object_id += 1

    if include_masks:
        if flat_masks:
            segments = torch.as_tensor(
                np.stack(flat_masks, axis=0),
                dtype=torch.bool,
                device=device,
            )
        else:
            segments = torch.empty(
                (0, int(resolution), int(resolution)),
                dtype=torch.bool,
                device=device,
            )
        is_valid_segment = torch.ones(
            (len(flat_masks),),
            dtype=torch.bool,
            device=device,
        )
    else:
        segments = None
        is_valid_segment = None

    return target_cls(
        num_boxes=num_boxes,
        boxes=boxes_tensor,
        boxes_padded=boxes_padded,
        repeated_boxes=boxes_tensor,
        segments=segments,
        semantic_segments=None,
        is_valid_segment=is_valid_segment,
        is_exhaustive=torch.ones(len(per_image_boxes), dtype=torch.bool, device=device),
        object_ids=torch.tensor(object_ids, dtype=torch.long, device=device),
        object_ids_padded=object_ids_padded,
    )


def _load_one_image_tensor(path: str | Path, *, resolution: int, torch: Any) -> tuple[Any, tuple[int, int]]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        resized = rgb.resize((resolution, resolution))
        array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor, (width, height)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _round_float(value: float) -> float:
    """减少浮点尾差，让 JSON 和测试输出更稳定。"""
    return round(float(value), 10)
