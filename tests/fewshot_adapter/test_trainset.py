"""测试根据选中错误样本更新训练集。"""

from fewshot_adapter.data.models import HBB, Annotation
from fewshot_adapter.evaluation.matching import ErrorItem
from fewshot_adapter.data.sampling import add_selected_errors_to_train_set


def _annotation(object_id: str, image_id: str) -> Annotation:
    return Annotation(
        image_id=image_id,
        object_id=object_id,
        label="target",
        source_type="hbb",
        hbb=HBB(0, 0, 10, 10),
    )


def test_add_selected_errors_to_train_set_adds_selected_ground_truth_once():
    train_set = [_annotation("gt_seed", "img_seed")]
    all_ground_truths = [
        _annotation("gt_seed", "img_seed"),
        _annotation("gt_next", "img_next"),
    ]
    errors = [
        ErrorItem(
            image_id="img_next",
            error_type="false_negative",
            risk_score=1.0,
            reason="missing",
            ground_truth_ids=["gt_next"],
            prediction_ids=[],
            selected_for_next_round=True,
        )
    ]

    next_train_set = add_selected_errors_to_train_set(train_set, all_ground_truths, errors)

    assert [item.object_id for item in next_train_set] == ["gt_seed", "gt_next"]


def test_add_selected_errors_to_train_set_ignores_duplicate_ground_truth():
    train_set = [_annotation("gt_next", "img_next")]
    all_ground_truths = [_annotation("gt_next", "img_next")]
    errors = [
        ErrorItem(
            image_id="img_next",
            error_type="false_negative",
            risk_score=1.0,
            reason="missing",
            ground_truth_ids=["gt_next"],
            prediction_ids=[],
            selected_for_next_round=True,
        )
    ]

    next_train_set = add_selected_errors_to_train_set(train_set, all_ground_truths, errors)

    assert [item.object_id for item in next_train_set] == ["gt_next"]
