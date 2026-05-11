from fewshot_lora.eval.errors import ErrorType, build_error_queue
from fewshot_lora.eval.geometry import OrientedBox
from fewshot_lora.eval.metrics import ImageGroundTruth, ImagePrediction, PredictionInstance, evaluate_image


def _box(x: float) -> OrientedBox:
    return OrientedBox(center=(x, 10.0), size=(8.0, 8.0), angle_degrees=0.0)


def test_evaluate_image_counts_true_positive_false_positive_and_false_negative():
    result = evaluate_image(
        ground_truth=ImageGroundTruth(image_id="img", boxes=[_box(10.0), _box(40.0)]),
        prediction=ImagePrediction(
            image_id="img",
            instances=[
                PredictionInstance(obb=_box(10.0), score=0.9),
                PredictionInstance(obb=_box(80.0), score=0.8),
            ],
        ),
        iou_threshold=0.5,
        localization_iou_threshold=0.1,
    )

    assert result.true_positive_count == 1
    assert result.false_positive_count == 1
    assert result.false_negative_count == 1
    assert result.localization_error_count == 0
    assert result.precision == 0.5
    assert result.recall == 0.5


def test_evaluate_image_promotes_overlap_below_match_threshold_to_localization_error():
    result = evaluate_image(
        ground_truth=ImageGroundTruth(image_id="img", boxes=[_box(10.0)]),
        prediction=ImagePrediction(
            image_id="img",
            instances=[PredictionInstance(obb=_box(15.0), score=0.9)],
        ),
        iou_threshold=0.7,
        localization_iou_threshold=0.1,
    )

    assert result.true_positive_count == 0
    assert result.false_positive_count == 0
    assert result.false_negative_count == 0
    assert result.localization_error_count == 1


def test_build_error_queue_orders_failure_types_and_selects_image_once():
    false_positive = evaluate_image(
        ground_truth=ImageGroundTruth(image_id="fp", boxes=[]),
        prediction=ImagePrediction("fp", [PredictionInstance(_box(10.0), score=0.7)]),
        iou_threshold=0.5,
    )
    false_negative = evaluate_image(
        ground_truth=ImageGroundTruth(image_id="fn", boxes=[_box(10.0)]),
        prediction=ImagePrediction("fn", []),
        iou_threshold=0.5,
    )

    queue = build_error_queue([false_positive, false_negative])

    assert [item.image_id for item in queue] == ["fn", "fp"]
    assert [item.error_type for item in queue] == [ErrorType.FALSE_NEGATIVE, ErrorType.FALSE_POSITIVE]
