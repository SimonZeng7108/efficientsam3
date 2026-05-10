import numpy as np
import cv2

from fewshot_lora.geometry import (
    OrientedBox,
    Polygon4,
    aabb_to_normalized_cxcywh,
    mask_to_obb,
    polygon_to_aabb,
    polygon_to_mask,
    rotated_iou,
)


def test_polygon_to_aabb_and_normalized_cxcywh():
    polygon = Polygon4(((10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0)))

    aabb = polygon_to_aabb(polygon)
    cxcywh = aabb_to_normalized_cxcywh(aabb, width=100, height=200)

    assert aabb == (10.0, 20.0, 30.0, 40.0)
    assert cxcywh == (0.2, 0.15, 0.2, 0.1)


def test_polygon_to_mask_rasterizes_inside_region():
    polygon = Polygon4(((1.0, 1.0), (4.0, 1.0), (4.0, 4.0), (1.0, 4.0)))

    mask = polygon_to_mask(polygon, width=6, height=6)

    assert mask.dtype == np.bool_
    assert mask.shape == (6, 6)
    assert mask[2, 2]
    assert not mask[0, 0]


def test_rotated_iou_is_one_for_identical_boxes_and_less_for_shifted_boxes():
    box = OrientedBox(center=(20.0, 20.0), size=(10.0, 8.0), angle_degrees=30.0)
    shifted = OrientedBox(center=(40.0, 20.0), size=(10.0, 8.0), angle_degrees=30.0)

    assert rotated_iou(box, box) == 1.0
    assert rotated_iou(box, shifted) == 0.0


def test_mask_to_obb_returns_axis_aligned_box_for_rectangular_component():
    mask = np.zeros((12, 16), dtype=bool)
    mask[3:8, 4:10] = True

    obb = mask_to_obb(mask)

    assert obb is not None
    assert obb.center == (6.5, 5.0)
    assert obb.size == (6.0, 5.0)


def test_mask_to_obb_fits_rotated_box_from_largest_component():
    mask = np.zeros((100, 100), dtype=np.uint8)
    expected = OrientedBox(center=(50.0, 50.0), size=(60.0, 20.0), angle_degrees=30.0)
    points = cv2.boxPoints(((50.0, 50.0), (60.0, 20.0), 30.0)).astype(np.int32)
    cv2.fillPoly(mask, [points], 1)

    obb = mask_to_obb(mask.astype(bool))

    assert obb is not None
    assert abs(obb.angle_degrees) > 1.0
    assert rotated_iou(obb, expected) > 0.8
