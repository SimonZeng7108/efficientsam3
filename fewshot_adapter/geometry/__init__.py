"""几何层：HBB、OBB、polygon 的转换与 IoU 计算。"""

from .ops import GeometryOps, obb_iou, polygon_area, polygon_iou, polygon_to_obb

__all__ = [
    "GeometryOps",
    "obb_iou",
    "polygon_area",
    "polygon_iou",
    "polygon_to_obb",
]
