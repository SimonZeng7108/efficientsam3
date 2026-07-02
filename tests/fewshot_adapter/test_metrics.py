"""测试检测评估指标计算。"""

from pytest import approx

from fewshot_adapter.data.models import HBB, Annotation, Prediction
from fewshot_adapter.evaluation.metrics import compute_detection_metrics


def _gt(object_id: str, hbb: HBB, image_id: str = "img_1", label: str = "target") -> Annotation:
    return Annotation(
        image_id=image_id,
        object_id=object_id,
        label=label,
        source_type="hbb",
        hbb=hbb,
    )


def _pred(prediction_id: str, hbb: HBB, image_id: str = "img_1", label: str = "target") -> Prediction:
    return Prediction(
        image_id=image_id,
        prediction_id=prediction_id,
        label=label,
        score=0.9,
        hbb=hbb,
    )


def test_compute_detection_metrics_counts_tp_fp_fn_and_rates():
    """指标应基于同一套 IoU 匹配逻辑，便于和错误队列对齐。"""
    metrics = compute_detection_metrics(
        ground_truths=[
            _gt("gt_tp", HBB(0, 0, 10, 10)),
            _gt("gt_fn", HBB(20, 20, 30, 30)),
        ],
        predictions=[
            _pred("pred_tp", HBB(0, 0, 10, 10)),
            _pred("pred_fp", HBB(40, 40, 50, 50)),
        ],
        iou_threshold=0.5,
    )

    assert metrics.true_positive == 1
    assert metrics.false_positive == 1
    assert metrics.false_negative == 1
    assert metrics.precision == approx(0.5)
    assert metrics.recall == approx(0.5)
    assert metrics.f1 == approx(0.5)
    assert metrics.miou == approx(1.0)


def test_compute_detection_metrics_to_dict_uses_json_friendly_keys():
    metrics = compute_detection_metrics(
        ground_truths=[_gt("gt_1", HBB(0, 0, 10, 10))],
        predictions=[],
        iou_threshold=0.5,
    )

    assert metrics.to_dict() == {
        "ground_truth_count": 1,
        "prediction_count": 0,
        "tp": 0,
        "fp": 0,
        "fn": 1,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "miou": 0.0,
    }


def test_compute_detection_metrics_can_filter_one_label():
    """多类别数据评估单个目标时，不应把其他类别算成 FN/FP。"""
    metrics = compute_detection_metrics(
        ground_truths=[
            _gt("gt_target", HBB(0, 0, 10, 10), label="target"),
            _gt("gt_other", HBB(20, 20, 30, 30), label="other"),
        ],
        predictions=[
            _pred("pred_target", HBB(0, 0, 10, 10), label="target"),
            _pred("pred_other", HBB(20, 20, 30, 30), label="other"),
        ],
        label="target",
        iou_threshold=0.5,
    )

    assert metrics.ground_truth_count == 1
    assert metrics.prediction_count == 1
    assert metrics.true_positive == 1
    assert metrics.false_positive == 0
    assert metrics.false_negative == 0
