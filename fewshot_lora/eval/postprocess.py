"""预测后处理：分数过滤、mask/box 转 OBB、单类别 NMS。

EfficientSAM3 输出的是归一化水平框和可选 mask。评估需要 OBB，所以这里优先
从 mask 拟合 OBB；如果没有 mask，则把水平框当作 angle=0 的 OBB 兜底。

本模块刻意贴近 `sam3.eval.postprocessors.PostProcessImage` 的职责边界：
原生后处理负责 score、box、mask 的尺度转换；本任务额外需要 OBB IoU，因此
这里只保留“SAM3 预测数组 -> OBB 实例 -> rotated NMS”这层任务特定薄逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import OrientedBox, mask_to_obb, rotated_iou
from .metrics import ImagePrediction, PredictionInstance


@dataclass(frozen=True)
class PredictionArrays:
    """从模型输出转成 numpy 后的轻量容器。"""

    scores: np.ndarray
    boxes_cxcywh: np.ndarray
    masks: np.ndarray | None = None


def postprocess_predictions(
    image_id: str,
    arrays: PredictionArrays,
    original_size: tuple[int, int],
    score_threshold: float,
    nms_iou_threshold: float,
    mask_threshold: float = 0.5,
) -> ImagePrediction:
    """把原始预测数组转换成 `ImagePrediction`。

    `original_size` 使用 `(height, width)`，和 PyTorch/图像张量习惯保持一致。
    """

    height, width = original_size
    instances: list[PredictionInstance] = []
    for index, raw_score in enumerate(arrays.scores):
        score = round(float(raw_score), 6)
        if score < score_threshold:
            continue
        obb = _obb_from_prediction(
            box_cxcywh=arrays.boxes_cxcywh[index],
            mask=None if arrays.masks is None else arrays.masks[index],
            width=width,
            height=height,
            mask_threshold=mask_threshold,
        )
        instances.append(PredictionInstance(obb=obb, score=score))

    kept = _rotated_nms(instances, nms_iou_threshold)
    return ImagePrediction(image_id=image_id, instances=kept)


def _obb_from_prediction(
    box_cxcywh: np.ndarray,
    mask: np.ndarray | None,
    width: int,
    height: int,
    mask_threshold: float,
) -> OrientedBox:
    """从单个预测得到评估用 OBB。"""

    if mask is not None:
        # mask 优先：分割头能给出旋转目标的真实形状，比水平框更适合 OBB IoU。
        bool_mask = mask if mask.dtype == np.bool_ else mask > mask_threshold
        mask_obb = mask_to_obb(bool_mask)
        if mask_obb is not None:
            return mask_obb

    # 兜底：模型只有水平框时，按 angle=0 的 OBB 参与评估。
    cx, cy, box_w, box_h = [float(value) for value in box_cxcywh]
    return OrientedBox(
        center=(cx * width, cy * height),
        size=(box_w * width, box_h * height),
        angle_degrees=0.0,
    )


def _rotated_nms(
    instances: list[PredictionInstance],
    iou_threshold: float,
) -> list[PredictionInstance]:
    """单类别旋转框 NMS。

    这里按 score 从高到低保留预测，若和已保留预测 IoU 太高就删除。
    """

    kept: list[PredictionInstance] = []
    for instance in sorted(instances, key=lambda item: item.score, reverse=True):
        if all(rotated_iou(instance.obb, kept_item.obb) <= iou_threshold for kept_item in kept):
            kept.append(instance)
    return kept
