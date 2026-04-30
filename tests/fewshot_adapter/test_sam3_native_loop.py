"""测试 EfficientSAM3 原生闭环的训练集更新辅助逻辑。"""

from fewshot_adapter.data.models import Annotation, HBB
from fewshot_adapter.evaluation.matching import ErrorItem
from fewshot_adapter.native.trainer import _compute_round_metrics, _resolve_label, add_selected_image_truth


def test_add_selected_image_truth_adds_gt_for_false_positive_image():
    """误检图片没有 ground_truth_ids 时，也要把该图真值加入下一轮。"""
    current = [
        Annotation("seed.jpg", "seed_1", "target", "hbb", hbb=HBB(0, 0, 10, 10))
    ]
    full_gt = current + [
        Annotation("hard.jpg", "hard_1", "target", "hbb", hbb=HBB(20, 20, 30, 30)),
        Annotation("hard.jpg", "hard_other", "other", "hbb", hbb=HBB(1, 1, 2, 2)),
    ]
    selected = ErrorItem(
        image_id="hard.jpg",
        error_type="false_positive",
        risk_score=0.9,
        reason="prediction has no matching ground truth",
        ground_truth_ids=[],
        prediction_ids=["pred_1"],
        selected_for_next_round=True,
    )

    next_train = add_selected_image_truth(
        current,
        full_gt,
        selected_image_id=selected.image_id,
        label="target",
    )

    assert [annotation.object_id for annotation in next_train] == ["seed_1", "hard_1"]


def test_resolve_label_requires_explicit_label_for_multiclass_data():
    """多类别数据不能静默取第一类，避免 GPU 验证时训练错目标。"""
    annotations = [
        Annotation("a.jpg", "a_1", "car", "hbb", hbb=HBB(0, 0, 1, 1)),
        Annotation("b.jpg", "b_1", "ship", "hbb", hbb=HBB(0, 0, 1, 1)),
    ]

    try:
        _resolve_label(None, annotations)
    except ValueError as exc:
        assert "--label" in str(exc)
    else:
        raise AssertionError("expected multi-class data to require an explicit label")


def test_compute_round_metrics_filters_to_target_label():
    """每轮 summary 指标只评估当前目标类别，避免其他类别污染 recall。"""
    metrics = _compute_round_metrics(
        full_ground_truth=[
            Annotation("target.jpg", "target_1", "target", "hbb", hbb=HBB(0, 0, 10, 10)),
            Annotation("other.jpg", "other_1", "other", "hbb", hbb=HBB(0, 0, 10, 10)),
        ],
        predictions=[],
        target_label="target",
        iou_threshold=0.5,
        iou_mode="hbb",
    )

    assert metrics["ground_truth_count"] == 1
    assert metrics["prediction_count"] == 0
    assert metrics["fn"] == 1
