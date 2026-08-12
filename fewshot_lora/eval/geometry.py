"""几何转换与 OBB IoU 工具。

训练和评估使用的几何格式不完全一样：
- EfficientSAM3 原生训练目标使用归一化 `cx, cy, w, h` 水平框。
- mask loss 使用 polygon 栅格化得到的二值 mask。
- 成功/失败评估使用 OBB IoU，尽量反映旋转目标的定位质量。

因此这里集中处理 polygon、AABB、mask、OBB 之间的转换，避免坐标格式
散落在训练和评估代码里造成混乱。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import cv2
import numpy as np


Point = tuple[float, float]
AABB = tuple[float, float, float, float]


@dataclass(frozen=True)
class Polygon4:
    """四点多边形，直接对应 DetectTrainData 的 `R:4` 标注。"""

    points: tuple[Point, Point, Point, Point]

    def __post_init__(self) -> None:
        if len(self.points) != 4:
            raise ValueError("Polygon4 必须包含且只包含四个点")


@dataclass(frozen=True)
class OrientedBox:
    """评估用旋转框。

    `center` 是像素坐标，`size` 是宽高，`angle_degrees` 是 OpenCV 旋转角度。
    """

    center: Point
    size: tuple[float, float]
    angle_degrees: float

    def area(self) -> float:
        return max(self.size[0], 0.0) * max(self.size[1], 0.0)


def polygon_to_aabb(polygon: Polygon4) -> AABB:
    """把 OBB polygon 转成外接水平框 xyxy。"""

    xs = [point[0] for point in polygon.points]
    ys = [point[1] for point in polygon.points]
    return min(xs), min(ys), max(xs), max(ys)


def aabb_to_normalized_cxcywh(aabb: AABB, width: int, height: int) -> tuple[float, float, float, float]:
    """把像素 xyxy 水平框转为 EfficientSAM3 需要的归一化 cxcywh。"""

    if width <= 0 or height <= 0:
        raise ValueError("图片宽高必须为正数")
    x0, y0, x1, y1 = aabb
    return (
        ((x0 + x1) * 0.5) / width,
        ((y0 + y1) * 0.5) / height,
        (x1 - x0) / width,
        (y1 - y0) / height,
    )


def polygon_to_mask(polygon: Polygon4, width: int, height: int) -> np.ndarray:
    """把四点 polygon 栅格化为 bool mask。

    OpenCV 填充多边形需要整数像素点；这里先裁剪到图像范围，再四舍五入。
    """

    if width <= 0 or height <= 0:
        raise ValueError("图片宽高必须为正数")
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.array(polygon.points, dtype=np.float32)
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    return mask.astype(bool)


def polygon_to_obb(polygon: Polygon4) -> OrientedBox:
    """用 OpenCV 最小外接旋转矩形从四点 polygon 得到 OBB。"""

    return _cv_rect_to_obb(cv2.minAreaRect(np.array(polygon.points, dtype=np.float32)))


def mask_to_obb(mask: np.ndarray) -> OrientedBox | None:
    """从预测 mask 拟合 OBB。

    实现步骤：
    1. 先取最大连通域，过滤掉零散噪点。
    2. 对最大连通域前景像素拟合 `cv2.minAreaRect`，得到真正的旋转框。
    3. 如果拟合结果是水平/垂直矩形，使用像素中心语义的水平框兜底，
       保持轴对齐 mask 的宽高解释稳定。
    """

    bool_mask = np.asarray(mask).astype(bool)
    if bool_mask.ndim != 2 or not bool_mask.any():
        return None
    component = _largest_component(bool_mask)
    rotated = _component_to_rotated_obb(component)
    if rotated is not None and not _is_axis_aligned_angle(rotated.angle_degrees):
        return rotated
    return _component_to_axis_aligned_obb(component)


def _component_to_axis_aligned_obb(component: np.ndarray) -> OrientedBox:
    """按前景像素中心解释返回水平框。

    例如 x=4..9 的 6 个像素，中心是 (4+9)/2=6.5，宽度是 6。
    """

    ys, xs = np.nonzero(component)
    x0 = float(xs.min())
    x_max = float(xs.max())
    y0 = float(ys.min())
    y_max = float(ys.max())
    return OrientedBox(
        center=((x0 + x_max) * 0.5, (y0 + y_max) * 0.5),
        size=(x_max - x0 + 1.0, y_max - y0 + 1.0),
        angle_degrees=0.0,
    )


def _component_to_rotated_obb(component: np.ndarray) -> OrientedBox | None:
    """用最大连通域的前景像素中心拟合旋转 OBB。"""

    ys, xs = np.nonzero(component)
    if len(xs) < 3:
        return None
    points = np.column_stack([xs, ys]).astype(np.float32)
    hull = cv2.convexHull(points)
    return _cv_rect_to_obb(cv2.minAreaRect(hull))


def _is_axis_aligned_angle(angle_degrees: float) -> bool:
    """判断 OpenCV 角度是否近似水平或垂直。"""

    angle = abs(float(angle_degrees)) % 90.0
    return angle < 1.0 or abs(angle - 90.0) < 1.0


def rotated_iou(a: OrientedBox, b: OrientedBox) -> float:
    """计算两个 OBB 的 IoU。"""

    if a.area() <= 0 or b.area() <= 0:
        return 0.0
    intersection_type, points = cv2.rotatedRectangleIntersection(_obb_to_cv_rect(a), _obb_to_cv_rect(b))
    if intersection_type == cv2.INTERSECT_NONE or points is None:
        inter_area = 0.0
    elif intersection_type == cv2.INTERSECT_FULL:
        inter_area = min(a.area(), b.area())
    else:
        inter_area = float(cv2.contourArea(points))
    union = a.area() + b.area() - inter_area
    if union <= 0:
        return 0.0
    iou = inter_area / union
    if math.isclose(iou, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        return 1.0
    return float(iou)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """返回二值 mask 中面积最大的连通域。"""

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest_label


def _obb_to_cv_rect(box: OrientedBox):
    """转换为 OpenCV rotatedRectangleIntersection 接受的 rect 格式。"""

    return (tuple(map(float, box.center)), tuple(map(float, box.size)), float(box.angle_degrees))


def _cv_rect_to_obb(rect) -> OrientedBox:
    """把 OpenCV minAreaRect 返回值转成项目内部 OBB。"""

    (cx, cy), (w, h), angle = rect
    return OrientedBox(center=(float(cx), float(cy)), size=(float(w), float(h)), angle_degrees=float(angle))
