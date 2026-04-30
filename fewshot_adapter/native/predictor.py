"""EfficientSAM3 原生输出到项目 Prediction 的后处理。"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any, Sequence

import numpy as np
from PIL import Image

from ..data.models import HBB, OBB, Point, Prediction
from ..geometry import polygon_to_obb

_BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
_NEAREST = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST


@dataclass(frozen=True)
class NativePredictionRecord:
    """SAM3 原生输出后的一条轻量预测记录。"""

    image_id: str
    prediction_id: str
    label: str
    score: float
    hbb: HBB
    obb: OBB | None = None
    polygon: list[Point] | None = None
    mask_path: str | None = None


@dataclass(frozen=True)
class MaskShape:
    """由预测 mask 派生出的几何形状。"""

    polygon: list[Point]
    obb: OBB


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
    # 有 mask 时 record.obb 来自 mask 后处理；无 mask 时才补 angle=0 兼容字段。
    obb = record.obb or hbb_to_zero_angle_obb(record.hbb)
    return Prediction(
        image_id=record.image_id,
        prediction_id=record.prediction_id,
        label=record.label,
        score=record.score,
        hbb=record.hbb,
        obb=obb,
        polygon=record.polygon,
        mask_path=record.mask_path,
    )


def hbb_to_zero_angle_obb(hbb: HBB) -> OBB:
    """把水平框转成 angle=0 的 OBB 基线。

    这是为了统一输出结构；真实旋转框优先由 `pred_masks` 后处理得到。
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
    masks = outputs.get("pred_masks")
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
            mask_shape = _prediction_mask_shape(
                masks,
                batch_index=batch_index,
                query_index=query_index,
                width=width,
                height=height,
            )
            record = NativePredictionRecord(
                image_id=image_id,
                prediction_id=f"{image_id}:{query_index:04d}",
                label=label,
                score=float(score),
                hbb=HBB(x1, y1, x2, y2),
                obb=None if mask_shape is None else mask_shape.obb,
                polygon=None if mask_shape is None else mask_shape.polygon,
            )
            predictions.append(record_to_prediction(record))
    return predictions


def mask_to_obb(mask: Any) -> OBB | None:
    """把已经二值化到原图尺寸的 mask 拟合为 OBB。"""
    shape = _binary_mask_to_shape(np.asarray(mask, dtype=bool))
    return None if shape is None else shape.obb


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


def _prediction_mask_shape(
    masks: Any,
    *,
    batch_index: int,
    query_index: int,
    width: int,
    height: int,
) -> MaskShape | None:
    """取出某个 query 的 mask，并转换成原图坐标下的 polygon/OBB。"""
    if masks is None:
        return None
    mask_array = _select_query_mask(masks, batch_index=batch_index, query_index=query_index)
    binary_mask = _mask_array_to_binary(mask_array, width=width, height=height)
    return _binary_mask_to_shape(binary_mask)


def _select_query_mask(masks: Any, *, batch_index: int, query_index: int) -> np.ndarray:
    """兼容 torch tensor 和轻量单测中的 Python list。"""
    if hasattr(masks, "detach"):
        selected = masks[batch_index, query_index]
        selected = selected.detach().float().cpu().numpy()
    else:
        selected = masks[batch_index][query_index]
    return np.asarray(selected)


def _mask_array_to_binary(mask_array: np.ndarray, *, width: int, height: int) -> np.ndarray:
    """把 SAM3 mask logits/probability/binary mask 统一成原图尺寸 bool mask。"""
    squeezed = np.squeeze(mask_array)
    if squeezed.ndim != 2:
        raise ValueError(f"expected 2D mask for one query, got shape {mask_array.shape}")

    if squeezed.dtype == np.bool_:
        if squeezed.shape != (height, width):
            return _resize_binary_mask(squeezed, width=width, height=height)
        return squeezed.astype(bool)

    values = np.asarray(squeezed, dtype=np.float32)
    values = np.nan_to_num(values, nan=-60.0, posinf=60.0, neginf=-60.0)
    if values.size == 0:
        return np.zeros((height, width), dtype=bool)
    if float(values.min()) < 0.0 or float(values.max()) > 1.0:
        values = _sigmoid_array(values)
    if values.shape != (height, width):
        values = _resize_float_mask(values, width=width, height=height)
    return values > 0.5


def _resize_binary_mask(mask: np.ndarray, *, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = image.resize((width, height), resample=_NEAREST)
    return np.asarray(resized, dtype=np.uint8) > 127


def _resize_float_mask(mask: np.ndarray, *, width: int, height: int) -> np.ndarray:
    clipped = np.clip(mask, 0.0, 1.0)
    image = Image.fromarray((clipped * 255.0).astype(np.uint8))
    resized = image.resize((width, height), resample=_BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _sigmoid_array(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _binary_mask_to_shape(mask: np.ndarray) -> MaskShape | None:
    mask = _largest_connected_component(mask)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    points: list[Point] = []
    for x, y in zip(xs.tolist(), ys.tolist()):
        left = float(x)
        top = float(y)
        right = left + 1.0
        bottom = top + 1.0
        # 用像素四角而不是中心点，可以让拟合出的 OBB 覆盖完整前景区域。
        points.extend(
            [
                (left, top),
                (right, top),
                (right, bottom),
                (left, bottom),
            ]
        )
    hull = _convex_hull(points)
    if len(hull) < 3:
        return None
    return MaskShape(polygon=hull, obb=polygon_to_obb(hull))


def _convex_hull(points: Sequence[Point]) -> list[Point]:
    """单调链凸包；输入是 mask 前景像素角点，输出按轮廓顺序排列。"""
    unique_points = sorted({(float(x), float(y)) for x, y in points})
    if len(unique_points) <= 1:
        return list(unique_points)

    lower: list[Point] = []
    for point in unique_points:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[Point] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def _cross(origin: Point, left: Point, right: Point) -> float:
    return (
        (left[0] - origin[0]) * (right[1] - origin[1])
        - (left[1] - origin[1]) * (right[0] - origin[0])
    )


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """只保留最大 8 邻域连通域，降低 mask 噪点对 OBB 的影响。"""
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return binary

    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    best_component: list[tuple[int, int]] = []
    for start_y, start_x in zip(*np.nonzero(binary)):
        if visited[start_y, start_x]:
            continue
        component = _collect_component(
            binary,
            visited,
            start_y=int(start_y),
            start_x=int(start_x),
            width=width,
            height=height,
        )
        if len(component) > len(best_component):
            best_component = component

    output = np.zeros_like(binary, dtype=bool)
    for y, x in best_component:
        output[y, x] = True
    return output


def _collect_component(
    binary: np.ndarray,
    visited: np.ndarray,
    *,
    start_y: int,
    start_x: int,
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    stack = [(start_y, start_x)]
    visited[start_y, start_x] = True
    component: list[tuple[int, int]] = []
    while stack:
        y, x = stack.pop()
        component.append((y, x))
        for next_y in range(max(0, y - 1), min(height, y + 2)):
            for next_x in range(max(0, x - 1), min(width, x + 2)):
                if visited[next_y, next_x] or not binary[next_y, next_x]:
                    continue
                visited[next_y, next_x] = True
                stack.append((next_y, next_x))
    return component
