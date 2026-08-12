"""测试图片级训练样本，包括 no-object 负样本。"""

import pytest

from fewshot_adapter.data.json_io import load_training_samples, save_training_samples
from fewshot_adapter.data.models import HBB, Annotation, TrainingSample


def test_save_training_samples_round_trips_positive_and_negative(tmp_path):
    """训练集 JSON 要能同时保存正样本和纯背景负样本。"""
    positive = TrainingSample(
        image_id="target.jpg",
        label="obj",
        annotations=[
            Annotation("target.jpg", "target_1", "obj", "hbb", hbb=HBB(1, 2, 3, 4))
        ],
    )
    negative = TrainingSample(
        image_id="background.jpg",
        label="obj",
        annotations=[],
        sample_type="negative",
        reason="selected false_positive hard negative",
    )
    path = tmp_path / "train_samples.json"

    save_training_samples(path, [positive, negative])

    loaded = load_training_samples(path)
    assert loaded[0].sample_type == "positive"
    assert loaded[0].annotations[0].object_id == "target_1"
    assert loaded[1].sample_type == "negative"
    assert loaded[1].image_id == "background.jpg"
    assert loaded[1].reason == "selected false_positive hard negative"


def test_training_sample_rejects_unknown_sample_type():
    """训练样本类型只允许正样本或 no-object 负样本，避免脏 JSON 静默进入训练。"""
    with pytest.raises(ValueError, match="sample_type"):
        TrainingSample(
            image_id="bad.jpg",
            label="obj",
            annotations=[],
            sample_type="unknown",  # type: ignore[arg-type]
        )
