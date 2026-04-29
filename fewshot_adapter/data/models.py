"""少样本闭环中使用的标注与预测基础数据结构。

这里统一管理 HBB、OBB、polygon 等几何格式。后续无论数据集原始标注
来自水平框、旋转框还是多边形，都先转成这些结构再进入训练/评估流程。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import cos, radians, sin
from typing import Literal, Sequence

Point = tuple[float, float]
SourceType = Literal["hbb", "obb", "polygon", "mask"]


@dataclass(frozen=True)
class HBB:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class OBB:
    cx: float
    cy: float
    w: float
    h: float
    angle: float


@dataclass(frozen=True)
class Annotation:
    image_id: str
    object_id: str
    label: str
    source_type: SourceType
    hbb: HBB | None = None
    obb: OBB | None = None
    polygon: list[Point] | None = None
    mask_path: str | None = None


@dataclass(frozen=True)
class Prediction:
    image_id: str
    prediction_id: str
    label: str
    score: float
    hbb: HBB | None = None
    obb: OBB | None = None
    polygon: list[Point] | None = None
    mask_path: str | None = None


def _clean_number(value: float) -> float | int:
    rounded = round(value)
    if abs(value - rounded) < 1e-9:
        return int(rounded)
    return value


def hbb_to_polygon(hbb: HBB) -> list[Point]:
    return [
        (hbb.x1, hbb.y1),
        (hbb.x2, hbb.y1),
        (hbb.x2, hbb.y2),
        (hbb.x1, hbb.y2),
    ]


def obb_to_polygon(obb: OBB) -> list[Point]:
    half_w = obb.w / 2
    half_h = obb.h / 2
    theta = radians(obb.angle)
    cos_t = cos(theta)
    sin_t = sin(theta)
    corners = [
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h),
    ]
    points: list[Point] = []
    for x, y in corners:
        rx = obb.cx + x * cos_t - y * sin_t
        ry = obb.cy + x * sin_t + y * cos_t
        points.append((_clean_number(rx), _clean_number(ry)))
    return points


def polygon_to_hbb(polygon: Sequence[Point]) -> HBB:
    if not polygon:
        raise ValueError("polygon must contain at least one point")
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return HBB(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))


def normalize_annotation(annotation: Annotation) -> Annotation:
    """补齐标注的派生几何字段，方便后续匹配和训练使用。"""
    polygon = annotation.polygon
    if polygon is None:
        if annotation.obb is not None:
            polygon = obb_to_polygon(annotation.obb)
        elif annotation.hbb is not None:
            polygon = hbb_to_polygon(annotation.hbb)

    hbb = annotation.hbb
    if hbb is None and polygon is not None:
        hbb = polygon_to_hbb(polygon)

    return replace(annotation, polygon=polygon, hbb=hbb)
