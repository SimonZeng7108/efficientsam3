"""EfficientSAM3 原生输出到项目 Prediction 的后处理。"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any, Sequence

from ..data.models import HBB, OBB, Prediction


@dataclass(frozen=True)
class NativePredictionRecord:
    """SAM3 原生输出后的一条轻量预测记录。"""

    image_id: str
    prediction_id: str
    label: str
    score: float
    hbb: HBB
    obb: OBB | None = None
    mask_path: str | None = None


class NativePredictor:
    """原生输出后处理门面。"""

    def outputs_to_predictions(
        self,
        outputs: dict[str, Any],
        *,
        image_ids: Sequence[str],
        original_sizes: Sequence[tuple[int, int]],
        label: str,
        score_threshold: float,
    ) -> list[Prediction]:
        return native_outputs_to_predictions(
            outputs,
            image_ids=image_ids,
            original_sizes=original_sizes,
            label=label,
            score_threshold=score_threshold,
        )


def tensor_box_to_hbb(
    box: Sequence[float],
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """把 SAM3 归一化 `cxcywh` 转成原图像素 `x1,y1,x2,y2`。"""
    cx, cy, box_w, box_h = [float(value) for value in box]
    x1 = (cx - box_w / 2) * width
    y1 = (cy - box_h / 2) * height
    x2 = (cx + box_w / 2) * width
    y2 = (cy + box_h / 2) * height
    return (
        round(max(0.0, min(float(width), x1)), 6),
        round(max(0.0, min(float(height), y1)), 6),
        round(max(0.0, min(float(width), x2)), 6),
        round(max(0.0, min(float(height), y2)), 6),
    )


def record_to_prediction(record: NativePredictionRecord) -> Prediction:
    """把原生预测记录转成项目公共 `Prediction`。"""
    # 这里的 OBB 只是 HBB 的 angle=0 兼容字段，不代表已经具备真实旋转框预测能力。
    obb = record.obb or hbb_to_zero_angle_obb(record.hbb)
    return Prediction(
        image_id=record.image_id,
        prediction_id=record.prediction_id,
        label=record.label,
        score=record.score,
        hbb=record.hbb,
        obb=obb,
        mask_path=record.mask_path,
    )


def hbb_to_zero_angle_obb(hbb: HBB) -> OBB:
    """把水平框转成 angle=0 的 OBB 基线。

    这是为了统一输出结构；真实 OBB 仍需后续由 mask/polygon 拟合或增加旋转框分支。
    """
    width = max(0.0, float(hbb.x2) - float(hbb.x1))
    height = max(0.0, float(hbb.y2) - float(hbb.y1))
    return OBB(
        cx=float(hbb.x1) + width / 2,
        cy=float(hbb.y1) + height / 2,
        w=width,
        h=height,
        angle=0.0,
    )


def native_outputs_to_predictions(
    outputs: dict[str, Any],
    *,
    image_ids: Sequence[str],
    original_sizes: Sequence[tuple[int, int]],
    label: str,
    score_threshold: float,
) -> list[Prediction]:
    """把 SAM3 原生 batch 输出转成预测列表。

    分数使用 `pred_logits.sigmoid()`；如果 decoder 输出了 presence score，
    则与 presence score 相乘，保持和 `Sam3Processor` 后处理一致。
    """
    boxes = _to_nested_list(outputs["pred_boxes"])
    scores = _sigmoid_nested(outputs["pred_logits"])
    if "presence_logit_dec" in outputs:
        presence_scores = _sigmoid_flat(outputs["presence_logit_dec"])
        for batch_index, batch_scores in enumerate(scores):
            scores[batch_index] = [
                float(score) * float(presence_scores[batch_index])
                for score in batch_scores
            ]

    predictions: list[Prediction] = []
    for batch_index, image_id in enumerate(image_ids):
        width, height = original_sizes[batch_index]
        for query_index, score in enumerate(scores[batch_index]):
            if score < score_threshold:
                continue
            x1, y1, x2, y2 = tensor_box_to_hbb(
                boxes[batch_index][query_index],
                width=width,
                height=height,
            )
            record = NativePredictionRecord(
                image_id=image_id,
                prediction_id=f"{image_id}:{query_index:04d}",
                label=label,
                score=float(score),
                hbb=HBB(x1, y1, x2, y2),
            )
            predictions.append(record_to_prediction(record))
    return predictions


def _to_nested_list(value: Any) -> list:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().tolist()
    return value


def _sigmoid_nested(value: Any) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach().float().sigmoid().squeeze(-1).cpu().tolist()
    return [[_sigmoid_scalar(item) for item in row] for row in value]


def _sigmoid_flat(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().float().sigmoid().reshape(-1).cpu().tolist()
    return [_sigmoid_scalar(item) for item in value]


def _sigmoid_scalar(value: Any) -> float:
    """兼容 tensor.tolist() 后的 float 或 `[float]`。"""
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"expected singleton logit list, got {value}")
        value = value[0]
    number = float(value)
    if number >= 0:
        z = exp(-number)
        return 1 / (1 + z)
    z = exp(number)
    return z / (1 + z)
