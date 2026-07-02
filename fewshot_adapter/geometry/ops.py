"""Polygon/OBB 几何计算。"""

from __future__ import annotations

from math import atan2, cos, degrees, inf, sin
from typing import Sequence

from ..data.models import OBB, Point, obb_to_polygon


class GeometryOps:
    """几何工具门面，便于其他智能体快速定位能力边界。"""

    @staticmethod
    def polygon_area(polygon: Sequence[Point]) -> float:
        return polygon_area(polygon)

    @staticmethod
    def polygon_iou(left: Sequence[Point], right: Sequence[Point]) -> float:
        return polygon_iou(left, right)

    @staticmethod
    def obb_iou(left: OBB, right: OBB) -> float:
        return obb_iou(left, right)

    @staticmethod
    def polygon_to_obb(polygon: Sequence[Point]) -> OBB:
        return polygon_to_obb(polygon)


def polygon_area(polygon: Sequence[Point]) -> float:
    if len(polygon) < 3:
        return 0.0
    total = 0.0
    for idx, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(idx + 1) % len(polygon)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2


def polygon_iou(left: Sequence[Point], right: Sequence[Point]) -> float:
    left_area = polygon_area(left)
    right_area = polygon_area(right)
    if left_area <= 0 or right_area <= 0:
        return 0.0
    intersection_polygon = _clip_polygon(list(left), list(right))
    intersection = polygon_area(intersection_polygon)
    union = left_area + right_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def obb_iou(left: OBB, right: OBB) -> float:
    return polygon_iou(obb_to_polygon(left), obb_to_polygon(right))


def polygon_to_obb(polygon: Sequence[Point]) -> OBB:
    if len(polygon) < 3:
        raise ValueError("polygon must contain at least three points")

    best_area = inf
    best: OBB | None = None
    points = list(polygon)
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        edge_angle = atan2(y2 - y1, x2 - x1)
        cos_t = cos(-edge_angle)
        sin_t = sin(-edge_angle)
        rotated = [
            (x * cos_t - y * sin_t, x * sin_t + y * cos_t)
            for x, y in points
        ]
        xs = [point[0] for point in rotated]
        ys = [point[1] for point in rotated]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y
        area = width * height
        if area < best_area:
            center_x_rot = (min_x + max_x) / 2
            center_y_rot = (min_y + max_y) / 2
            cos_back = cos(edge_angle)
            sin_back = sin(edge_angle)
            cx = center_x_rot * cos_back - center_y_rot * sin_back
            cy = center_x_rot * sin_back + center_y_rot * cos_back
            best_area = area
            best = OBB(cx=cx, cy=cy, w=width, h=height, angle=_normalize_angle(degrees(edge_angle)))

    if best is None:
        raise ValueError("could not fit OBB to polygon")
    return best


def _clip_polygon(subject: list[Point], clip: list[Point]) -> list[Point]:
    output = subject
    if not output:
        return []
    clip_ccw = _signed_area(clip) >= 0
    for idx, edge_start in enumerate(clip):
        edge_end = clip[(idx + 1) % len(clip)]
        input_points = output
        output = []
        if not input_points:
            break
        previous = input_points[-1]
        for current in input_points:
            current_inside = _inside(current, edge_start, edge_end, clip_ccw)
            previous_inside = _inside(previous, edge_start, edge_end, clip_ccw)
            if current_inside:
                if not previous_inside:
                    output.append(_line_intersection(previous, current, edge_start, edge_end))
                output.append(current)
            elif previous_inside:
                output.append(_line_intersection(previous, current, edge_start, edge_end))
            previous = current
    return output


def _inside(point: Point, edge_start: Point, edge_end: Point, clip_ccw: bool) -> bool:
    cross = (
        (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1])
        - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])
    )
    return cross >= -1e-9 if clip_ccw else cross <= 1e-9


def _line_intersection(p1: Point, p2: Point, q1: Point, q2: Point) -> Point:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-12:
        return p2
    px = (
        (x1 * y2 - y1 * x2) * (x3 - x4)
        - (x1 - x2) * (x3 * y4 - y3 * x4)
    ) / denominator
    py = (
        (x1 * y2 - y1 * x2) * (y3 - y4)
        - (y1 - y2) * (x3 * y4 - y3 * x4)
    ) / denominator
    return (px, py)


def _signed_area(polygon: Sequence[Point]) -> float:
    total = 0.0
    for idx, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(idx + 1) % len(polygon)]
        total += x1 * y2 - x2 * y1
    return total / 2


def _normalize_angle(angle: float) -> float:
    while angle >= 90:
        angle -= 180
    while angle < -90:
        angle += 180
    if abs(angle) < 1e-9:
        return 0.0
    return angle
