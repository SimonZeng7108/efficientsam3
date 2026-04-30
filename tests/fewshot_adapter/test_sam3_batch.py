"""测试 SAM3 原生训练 batch 的纯数据转换部分。"""

from fewshot_adapter.data.models import Annotation, HBB, TrainingSample
from fewshot_adapter.data.sam3_batch import (
    annotation_to_target_box,
    group_annotations_by_image,
    group_training_samples_by_image,
    hbb_to_cxcywh_norm,
)


def test_group_annotations_by_image_keeps_first_seen_order():
    """按图片分组时保留首次出现顺序，便于闭环日志可读。"""
    annotations = [
        Annotation("b.jpg", "b1", "target", "hbb", hbb=HBB(0, 0, 10, 10)),
        Annotation("a.jpg", "a1", "target", "hbb", hbb=HBB(5, 5, 15, 25)),
        Annotation("b.jpg", "b2", "target", "hbb", hbb=HBB(20, 20, 30, 30)),
    ]

    grouped = group_annotations_by_image(annotations)

    assert list(grouped) == ["b.jpg", "a.jpg"]
    assert [item.object_id for item in grouped["b.jpg"]] == ["b1", "b2"]


def test_hbb_to_cxcywh_norm_converts_pixel_box_to_sam3_format():
    """SAM3 matcher 使用归一化 cxcywh，输入 HBB 是原图像素坐标。"""
    assert hbb_to_cxcywh_norm(HBB(10, 20, 30, 60), width=100, height=200) == (
        0.2,
        0.2,
        0.2,
        0.2,
    )


def test_annotation_to_target_box_uses_polygon_derived_hbb():
    """Polygon 标注先派生 HBB，第一版训练用 box loss 对齐 SAM3 原生输出。"""
    annotation = Annotation(
        image_id="poly.jpg",
        object_id="poly_1",
        label="ship",
        source_type="polygon",
        polygon=[(2, 4), (10, 4), (10, 14), (2, 14)],
    )

    assert annotation_to_target_box(annotation, width=20, height=20) == (
        0.3,
        0.45,
        0.4,
        0.5,
    )


def test_group_training_samples_by_image_keeps_negative_sample_with_empty_targets():
    """负样本图片进入 batch 时应保留 image_id，但目标列表为空。"""
    positive = TrainingSample(
        image_id="target.jpg",
        label="target",
        annotations=[Annotation("target.jpg", "gt_1", "target", "hbb", hbb=HBB(0, 0, 1, 1))],
    )
    negative = TrainingSample(
        image_id="background.jpg",
        label="target",
        annotations=[],
        sample_type="negative",
    )

    grouped = group_training_samples_by_image([positive, negative])

    assert list(grouped) == ["target.jpg", "background.jpg"]
    assert grouped["target.jpg"][0].object_id == "gt_1"
    assert grouped["background.jpg"] == []
