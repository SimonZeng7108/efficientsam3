"""测试标注格式统一和几何字段派生。"""

from fewshot_adapter.data.models import (
    HBB,
    OBB,
    Annotation,
    hbb_to_polygon,
    normalize_annotation,
    obb_to_polygon,
    polygon_to_hbb,
)


def test_hbb_to_polygon_returns_four_corners_clockwise():
    hbb = HBB(x1=10, y1=20, x2=30, y2=50)

    assert hbb_to_polygon(hbb) == [(10, 20), (30, 20), (30, 50), (10, 50)]


def test_polygon_to_hbb_wraps_all_points():
    polygon = [(5, 9), (20, 3), (17, 30), (1, 12)]

    assert polygon_to_hbb(polygon) == HBB(x1=1, y1=3, x2=20, y2=30)


def test_obb_to_polygon_preserves_axis_aligned_box_when_angle_zero():
    obb = OBB(cx=20, cy=30, w=20, h=10, angle=0)

    assert obb_to_polygon(obb) == [(10, 25), (30, 25), (30, 35), (10, 35)]


def test_normalize_annotation_derives_missing_polygon_and_hbb_from_obb():
    annotation = Annotation(
        image_id="img_1",
        object_id="gt_1",
        label="target",
        source_type="obb",
        obb=OBB(cx=20, cy=30, w=20, h=10, angle=0),
    )

    normalized = normalize_annotation(annotation)

    assert normalized.polygon == [(10, 25), (30, 25), (30, 35), (10, 35)]
    assert normalized.hbb == HBB(x1=10, y1=25, x2=30, y2=35)
    assert normalized.obb == annotation.obb
