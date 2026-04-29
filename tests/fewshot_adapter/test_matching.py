"""测试预测和真值匹配、错误队列生成、下一样本选择。"""

from fewshot_adapter.data.models import HBB, OBB, Annotation, Prediction
from fewshot_adapter.evaluation.matching import (
    build_error_queue,
    box_iou,
    greedy_match_predictions,
    select_next_training_sample,
)


def _gt(object_id: str, hbb: HBB, image_id: str = "img_1") -> Annotation:
    return Annotation(
        image_id=image_id,
        object_id=object_id,
        label="target",
        source_type="hbb",
        hbb=hbb,
    )


def _pred(prediction_id: str, hbb: HBB, score: float = 0.9, image_id: str = "img_1") -> Prediction:
    return Prediction(
        image_id=image_id,
        prediction_id=prediction_id,
        label="target",
        score=score,
        hbb=hbb,
    )


def _gt_obb(object_id: str, obb: OBB, image_id: str = "img_1") -> Annotation:
    return Annotation(
        image_id=image_id,
        object_id=object_id,
        label="target",
        source_type="obb",
        obb=obb,
    )


def _pred_obb(prediction_id: str, obb: OBB, score: float = 0.9, image_id: str = "img_1") -> Prediction:
    return Prediction(
        image_id=image_id,
        prediction_id=prediction_id,
        label="target",
        score=score,
        obb=obb,
    )


def test_box_iou_returns_one_for_identical_boxes():
    box = HBB(0, 0, 10, 10)

    assert box_iou(box, box) == 1.0


def test_greedy_match_predictions_matches_high_iou_prediction():
    matches = greedy_match_predictions(
        [_gt("gt_1", HBB(0, 0, 10, 10))],
        [_pred("pred_1", HBB(1, 1, 9, 9))],
        iou_threshold=0.5,
    )

    assert [(match.ground_truth.object_id, match.prediction.prediction_id) for match in matches] == [
        ("gt_1", "pred_1")
    ]


def test_greedy_match_predictions_never_matches_across_images():
    """相同 label 和 box 出现在不同图片时，不能互相匹配。"""
    matches = greedy_match_predictions(
        [_gt("gt_1", HBB(0, 0, 10, 10), image_id="left.jpg")],
        [_pred("pred_1", HBB(0, 0, 10, 10), image_id="right.jpg")],
        iou_threshold=0.5,
    )

    assert matches == []


def test_build_error_queue_reports_false_negative_when_prediction_missing():
    errors = build_error_queue(
        [_gt("gt_1", HBB(0, 0, 10, 10))],
        [],
        iou_threshold=0.5,
    )

    assert len(errors) == 1
    assert errors[0].error_type == "false_negative"
    assert errors[0].image_id == "img_1"
    assert errors[0].ground_truth_ids == ["gt_1"]


def test_build_error_queue_reports_false_positive_for_unmatched_prediction():
    errors = build_error_queue(
        [],
        [_pred("pred_1", HBB(0, 0, 10, 10), score=0.95)],
        iou_threshold=0.5,
    )

    assert len(errors) == 1
    assert errors[0].error_type == "false_positive"
    assert errors[0].prediction_ids == ["pred_1"]
    assert errors[0].risk_score == 0.95


def test_build_error_queue_reports_localization_error_for_low_iou_overlap():
    errors = build_error_queue(
        [_gt("gt_1", HBB(0, 0, 10, 10))],
        [_pred("pred_1", HBB(6, 0, 16, 10), score=0.8)],
        iou_threshold=0.5,
        localization_error_threshold=0.1,
    )

    assert len(errors) == 1
    assert errors[0].error_type == "localization_error"
    assert errors[0].ground_truth_ids == ["gt_1"]
    assert errors[0].prediction_ids == ["pred_1"]


def test_select_next_training_sample_prioritizes_false_negative_then_localization_then_false_positive():
    errors = build_error_queue(
        [
            _gt("gt_1", HBB(0, 0, 10, 10), image_id="img_fn"),
            _gt("gt_2", HBB(0, 0, 10, 10), image_id="img_loc"),
        ],
        [
            _pred("pred_loc", HBB(6, 0, 16, 10), score=0.8, image_id="img_loc"),
            _pred("pred_fp", HBB(0, 0, 10, 10), score=0.99, image_id="img_fp"),
        ],
        iou_threshold=0.5,
        localization_error_threshold=0.1,
    )

    selected = select_next_training_sample(errors)

    assert selected is not None
    assert selected.image_id == "img_fn"
    assert selected.error_type == "false_negative"
    assert selected.selected_for_next_round is True


def test_build_error_queue_can_match_using_obb_iou():
    errors = build_error_queue(
        [_gt_obb("gt_1", OBB(cx=0, cy=0, w=4, h=2, angle=45))],
        [_pred_obb("pred_1", OBB(cx=0, cy=0, w=4, h=2, angle=45))],
        iou_threshold=0.5,
        iou_mode="obb",
    )

    assert errors == []
