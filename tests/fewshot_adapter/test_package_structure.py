"""测试分包后的高层类接口。"""

import json

from fewshot_adapter.data.datatrain import DataTrainDataset
from fewshot_adapter.data.sampling import InitialTrainSelector, TrainSetUpdater
from fewshot_adapter.evaluation.matching import DetectionMatcher, ErrorSelector
from fewshot_adapter.geometry.ops import GeometryOps
from fewshot_adapter.native.loss import NativeLossFactory
from fewshot_adapter.native.predictor import NativePredictor
from fewshot_adapter.visualization import RoundVisualizationOutputs


def test_datatrain_dataset_saves_native_training_inputs(tmp_path):
    """DataTrainDataset 负责从原始文本生成原生训练所需 JSON。"""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "img_001.jpg").write_bytes(b"fake-image")
    datatrain = tmp_path / "DataTrain.txt"
    datatrain.write_text('img_001.jpg 1 R 10 20 30 40 "car"\n', encoding="utf-8")
    output_dir = tmp_path / "json"

    dataset = DataTrainDataset.from_file(datatrain, image_dir=image_dir)
    dataset.save_json(output_dir)

    full_gt = json.loads((output_dir / "full_gt.json").read_text(encoding="utf-8"))
    image_map = json.loads((output_dir / "image_map.json").read_text(encoding="utf-8"))
    assert full_gt[0]["object_id"] == "img_001_0001"
    assert image_map["img_001.jpg"].endswith("img_001.jpg")


def test_high_level_classes_are_importable():
    """后续智能体可以先看这些类名理解系统边界。"""
    assert InitialTrainSelector is not None
    assert TrainSetUpdater is not None
    assert DetectionMatcher is not None
    assert ErrorSelector is not None
    assert GeometryOps is not None
    assert NativeLossFactory is not None
    assert NativePredictor is not None
    assert RoundVisualizationOutputs is not None
