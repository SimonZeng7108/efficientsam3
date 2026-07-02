"""测试 polygon/OBB 几何计算。"""

from pytest import approx

from fewshot_adapter.data.models import OBB
from fewshot_adapter.geometry.ops import (
    obb_iou,
    polygon_area,
    polygon_iou,
    polygon_to_obb,
)


def test_polygon_area_returns_area_for_rectangle():
    polygon = [(0, 0), (4, 0), (4, 3), (0, 3)]

    assert polygon_area(polygon) == 12


def test_polygon_iou_returns_one_for_identical_polygons():
    polygon = [(0, 0), (4, 0), (4, 3), (0, 3)]

    assert polygon_iou(polygon, polygon) == 1.0


def test_polygon_iou_computes_partial_overlap():
    left = [(0, 0), (4, 0), (4, 4), (0, 4)]
    right = [(2, 0), (6, 0), (6, 4), (2, 4)]

    assert polygon_iou(left, right) == approx(1 / 3)


def test_obb_iou_matches_axis_aligned_overlap():
    left = OBB(cx=2, cy=2, w=4, h=4, angle=0)
    right = OBB(cx=4, cy=2, w=4, h=4, angle=0)

    assert obb_iou(left, right) == approx(1 / 3)


def test_polygon_to_obb_recovers_axis_aligned_rectangle():
    polygon = [(10, 20), (30, 20), (30, 50), (10, 50)]

    obb = polygon_to_obb(polygon)

    assert obb.cx == approx(20)
    assert obb.cy == approx(35)
    assert obb.w == approx(20)
    assert obb.h == approx(30)
    assert obb.angle == approx(0)
