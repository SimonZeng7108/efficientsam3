"""测试根据选中错误样本更新训练集。"""

from fewshot_adapter.data.models import HBB, Annotation, TrainingSample
from fewshot_adapter.evaluation.matching import ErrorItem
from fewshot_adapter.data.sampling import (
    add_selected_errors_to_train_set,
    add_selected_errors_to_training_samples,
)


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


def test_add_selected_errors_to_training_samples_adds_background_false_positive_as_negative():
    """纯背景误检应作为 no-object hard negative 进入下一轮训练。"""
    train_samples = [
        TrainingSample(
            image_id="seed.jpg",
            label="target",
            annotations=[_annotation("gt_seed", "seed.jpg")],
        )
    ]
    errors = [
        ErrorItem(
            image_id="background.jpg",
            error_type="false_positive",
            risk_score=0.95,
            reason="prediction has no matching ground truth",
            ground_truth_ids=[],
            prediction_ids=["background.jpg:0000"],
            selected_for_next_round=True,
        )
    ]

    next_samples = add_selected_errors_to_training_samples(
        train_samples,
        all_ground_truths=[_annotation("gt_seed", "seed.jpg")],
        errors=errors,
        label="target",
    )

    assert [(sample.image_id, sample.sample_type) for sample in next_samples] == [
        ("seed.jpg", "positive"),
        ("background.jpg", "negative"),
    ]
    assert next_samples[-1].annotations == []


def test_add_selected_errors_to_training_samples_uses_truth_for_labeled_false_positive():
    """有目标图片上的误检不能当整图负样本，应补入该图同类真值。"""
    train_samples = [
        TrainingSample(
            image_id="seed.jpg",
            label="target",
            annotations=[_annotation("gt_seed", "seed.jpg")],
        )
    ]
    hard_gt = _annotation("gt_hard", "hard.jpg")
    errors = [
        ErrorItem(
            image_id="hard.jpg",
            error_type="false_positive",
            risk_score=0.8,
            reason="duplicate prediction",
            ground_truth_ids=[],
            prediction_ids=["hard.jpg:0001"],
            selected_for_next_round=True,
        )
    ]

    next_samples = add_selected_errors_to_training_samples(
        train_samples,
        all_ground_truths=[_annotation("gt_seed", "seed.jpg"), hard_gt],
        errors=errors,
        label="target",
    )

    assert next_samples[-1].sample_type == "positive"
    assert next_samples[-1].image_id == "hard.jpg"
    assert [annotation.object_id for annotation in next_samples[-1].annotations] == ["gt_hard"]
