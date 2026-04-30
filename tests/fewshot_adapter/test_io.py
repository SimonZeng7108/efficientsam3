"""测试 JSON 标注、预测和错误队列读写。"""

import json

from fewshot_adapter.data.models import HBB, OBB, Annotation, Prediction
from fewshot_adapter.data.json_io import (
    load_annotations,
    load_predictions,
    save_annotations,
    save_error_queue,
    save_predictions,
)
from fewshot_adapter.evaluation.matching import ErrorItem


def test_load_annotations_reads_hbb_obb_and_polygon(tmp_path):
    path = tmp_path / "annotations.json"
    path.write_text(
        json.dumps(
            [
                {
                    "image_id": "img_1",
                    "object_id": "gt_hbb",
                    "label": "target",
                    "source_type": "hbb",
                    "hbb": [1, 2, 3, 4],
                },
                {
                    "image_id": "img_2",
                    "object_id": "gt_obb",
                    "label": "target",
                    "source_type": "obb",
                    "obb": {"cx": 5, "cy": 6, "w": 7, "h": 8, "angle": 9},
                },
                {
                    "image_id": "img_3",
                    "object_id": "gt_poly",
                    "label": "target",
                    "source_type": "polygon",
                    "polygon": [[0, 0], [2, 0], [2, 2], [0, 2]],
                },
            ]
        ),
        encoding="utf-8",
    )

    annotations = load_annotations(path)

    assert annotations[0].hbb == HBB(1, 2, 3, 4)
    assert annotations[1].obb == OBB(5, 6, 7, 8, 9)
    assert annotations[2].polygon == [(0, 0), (2, 0), (2, 2), (0, 2)]


def test_load_predictions_reads_score_and_geometry(tmp_path):
    path = tmp_path / "predictions.json"
    path.write_text(
        json.dumps(
            [
                {
                    "image_id": "img_1",
                    "prediction_id": "pred_1",
                    "label": "target",
                    "score": 0.87,
                    "hbb": [1, 2, 3, 4],
                }
            ]
        ),
        encoding="utf-8",
    )

    predictions = load_predictions(path)

    assert predictions[0].score == 0.87
    assert predictions[0].hbb == HBB(1, 2, 3, 4)


def test_save_error_queue_writes_selected_flag(tmp_path):
    path = tmp_path / "errors.json"
    errors = [
        ErrorItem(
            image_id="img_1",
            error_type="false_negative",
            risk_score=1.0,
            reason="missing",
            ground_truth_ids=["gt_1"],
            prediction_ids=[],
            selected_for_next_round=True,
        )
    ]

    save_error_queue(path, errors)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved[0]["image_id"] == "img_1"
    assert saved[0]["selected_for_next_round"] is True


def test_save_annotations_round_trips_hbb(tmp_path):
    path = tmp_path / "train.json"
    annotation = Annotation(
        image_id="img_1",
        object_id="gt_1",
        label="target",
        source_type="hbb",
        hbb=HBB(1, 2, 3, 4),
    )

    save_annotations(path, [annotation])

    loaded = load_annotations(path)
    assert loaded == [annotation]


def test_save_predictions_round_trips_candidate_geometry(tmp_path):
    path = tmp_path / "predictions.json"
    prediction = Prediction(
        image_id="img_1",
        prediction_id="cand_1",
        label="target",
        score=0.92,
        hbb=HBB(1, 2, 3, 4),
    )

    save_predictions(path, [prediction])

    loaded = load_predictions(path)
    assert loaded == [prediction]
