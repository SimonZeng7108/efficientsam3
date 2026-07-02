"""测试 SAM3 原生训练 batch 的纯数据转换部分。"""

from collections import OrderedDict

import numpy as np

from fewshot_adapter.data.models import Annotation, HBB, TrainingSample
from fewshot_adapter.data.masks import annotation_to_mask
from fewshot_adapter.data.sam3_batch import (
    _build_find_target,
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


def test_annotation_to_mask_rasterizes_polygon_at_training_resolution():
    """粗 mask target 使用原图标注点按训练分辨率缩放后栅格化。"""
    annotation = Annotation(
        image_id="poly.jpg",
        object_id="poly_1",
        label="ship",
        source_type="polygon",
        polygon=[(0, 0), (8, 0), (8, 8), (0, 8)],
    )

    mask = annotation_to_mask(annotation, width=16, height=16, resolution=4)

    assert mask.shape == (4, 4)
    assert mask.dtype == np.bool_
    assert bool(mask[0, 0]) is True
    assert bool(mask[1, 1]) is True
    assert bool(mask[3, 3]) is False


def test_build_find_target_can_include_coarse_masks_aligned_with_boxes():
    """启用 mask loss 时，SAM3 target 需要 packed masks 与 packed boxes 一一对齐。"""
    grouped = OrderedDict(
        [
            (
                "target.jpg",
                [
                    Annotation(
                        image_id="target.jpg",
                        object_id="gt_1",
                        label="target",
                        source_type="polygon",
                        polygon=[(0, 0), (8, 0), (8, 8), (0, 8)],
                    )
                ],
            ),
            ("background.jpg", []),
        ]
    )

    target = _build_find_target(
        grouped,
        [(16, 16), (16, 16)],
        target_cls=_FakeBatchedFindTarget,
        torch=_FakeTorch(),
        device="cpu",
        include_masks=True,
        resolution=4,
    )

    assert target.num_boxes.tolist() == [1, 0]
    assert target.boxes.shape == (1, 4)
    assert target.segments.shape == (1, 4, 4)
    assert target.is_valid_segment.tolist() == [True]
    assert bool(target.segments[0, 0, 0]) is True
    assert bool(target.segments[0, 3, 3]) is False


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


class _FakeBatchedFindTarget:
    """只保存构造参数，避免单测依赖真实 SAM3/PyTorch。"""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeTorch:
    """覆盖 `_build_find_target` 用到的少量 torch API。"""

    long = np.int64
    float32 = np.float32
    bool = np.bool_

    def tensor(self, data, *, dtype=None, device=None):
        return np.asarray(data, dtype=dtype)

    def as_tensor(self, data, *, dtype=None, device=None):
        return np.asarray(data, dtype=dtype)

    def empty(self, shape, *, dtype=None, device=None):
        return np.empty(shape, dtype=dtype)

    def zeros(self, shape, *, dtype=None, device=None):
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape, *, dtype=None, device=None):
        return np.ones(shape, dtype=dtype)

    def full(self, shape, fill_value, *, dtype=None, device=None):
        return np.full(shape, fill_value, dtype=dtype)
