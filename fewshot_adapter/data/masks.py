"""标注到粗 mask target 的转换工具。"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .models import Annotation, Point, normalize_annotation


def annotation_to_mask(
    annotation: Annotation,
    *,
    width: int,
    height: int,
    resolution: int,
) -> np.ndarray:
    """把 HBB/OBB/Polygon 标注栅格化为 SAM3 mask loss 使用的粗 mask。

    DataTrain 里通常只有框或四点多边形，没有像素级精细 mask。这里的目标不是
    生成完美分割真值，而是给 SAM3 mask head 一个“目标大致区域”的监督信号；
    后续再由预测 mask 拟合真实 OBB。
    """
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    if resolution <= 0:
        raise ValueError("mask resolution must be positive")

    normalized = normalize_annotation(annotation)
    if normalized.polygon is None or len(normalized.polygon) < 3:
        raise ValueError(f"annotation has no usable polygon for mask target: {annotation.object_id}")

    scaled_polygon = [
        _scale_point(point, width=width, height=height, resolution=resolution)
        for point in normalized.polygon
    ]
    mask = Image.new("1", (resolution, resolution), 0)
    ImageDraw.Draw(mask).polygon(scaled_polygon, fill=1)
    return np.asarray(mask, dtype=np.bool_)


def _scale_point(
    point: Point,
    *,
    width: int,
    height: int,
    resolution: int,
) -> tuple[float, float]:
    x, y = point
    return (
        float(x) / float(width) * float(resolution),
        float(y) / float(height) * float(resolution),
    )
