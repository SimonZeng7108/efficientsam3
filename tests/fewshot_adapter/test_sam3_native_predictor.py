"""测试 SAM3 原生输出后处理的纯几何部分。"""

from fewshot_adapter.data.models import HBB
from fewshot_adapter.native.predictor import (
    NativePredictionRecord,
    native_outputs_to_predictions,
    record_to_prediction,
    tensor_box_to_hbb,
)


def test_tensor_box_to_hbb_converts_cxcywh_to_pixels():
    """SAM3 输出归一化 cxcywh，产品预测 JSON 使用原图像素 HBB。"""
    assert tensor_box_to_hbb([0.5, 0.5, 0.25, 0.5], width=200, height=100) == (
        75.0,
        25.0,
        125.0,
        75.0,
    )


def test_record_to_prediction_preserves_native_score_and_label():
    """后处理记录转成公共 Prediction，供现有错误筛选模块复用。"""
    record = NativePredictionRecord(
        image_id="img.jpg",
        prediction_id="img.jpg:0003",
        label="target",
        score=0.75,
        hbb=HBB(1, 2, 3, 4),
    )

    prediction = record_to_prediction(record)

    assert prediction.image_id == "img.jpg"
    assert prediction.prediction_id == "img.jpg:0003"
    assert prediction.label == "target"
    assert prediction.score == 0.75
    assert prediction.hbb == HBB(1, 2, 3, 4)
    assert prediction.obb is not None
    assert prediction.obb.angle == 0.0


def test_native_outputs_to_predictions_handles_batched_logits_lists():
    """无 torch 单测中也要兼容 SAM3 的 `[B, Q, 1]` logits 形状。"""
    outputs = {
        "pred_boxes": [[[0.5, 0.5, 0.5, 0.5], [0.1, 0.1, 0.1, 0.1]]],
        "pred_logits": [[[4.0], [-4.0]]],
    }

    predictions = native_outputs_to_predictions(
        outputs,
        image_ids=["img.jpg"],
        original_sizes=[(100, 100)],
        label="target",
        score_threshold=0.5,
    )

    assert len(predictions) == 1
    assert predictions[0].prediction_id == "img.jpg:0000"
    assert predictions[0].hbb == HBB(25.0, 25.0, 75.0, 75.0)


def test_native_outputs_to_predictions_fits_obb_from_predicted_mask():
    """存在 pred_masks 时，产品 OBB 应来自 mask 形状，而不是 HBB 的 angle=0 占位框。"""
    mask = [
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1],
    ]
    outputs = {
        "pred_boxes": [[[0.5, 0.5, 1.0, 1.0]]],
        "pred_logits": [[[4.0]]],
        "pred_masks": [[mask]],
    }

    predictions = native_outputs_to_predictions(
        outputs,
        image_ids=["img.jpg"],
        original_sizes=[(6, 6)],
        label="target",
        score_threshold=0.5,
    )

    assert len(predictions) == 1
    assert predictions[0].obb is not None
    assert abs(predictions[0].obb.angle) > 1.0
