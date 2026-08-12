from pathlib import Path

import numpy as np
from PIL import Image

from fewshot_lora.data.dataset import load_detect_dataset
from fewshot_lora.eval.geometry import OrientedBox
from fewshot_lora.eval.postprocess import PredictionArrays, postprocess_predictions


def test_load_detect_dataset_resolves_images_and_converts_instances(tmp_path: Path):
    Image.new("RGB", (20, 10), color=(0, 0, 0)).save(tmp_path / "Part.JPG.BMP")
    (tmp_path / "DetectTrainData.txt").write_text(
        'Version 1.0.0\npart.jpg.bmp:1 R:4 2 2 8 2 8 6 2 6 "Sample"',
        encoding="utf-8",
    )

    dataset = load_detect_dataset(tmp_path)

    assert dataset.issues == []
    assert len(dataset.images) == 1
    image = dataset.images[0]
    assert image.path.name == "Part.JPG.BMP"
    assert image.width == 20
    assert image.height == 10
    assert image.instances[0].box_cxcywh == (0.25, 0.4, 0.3, 0.4)
    assert image.instances[0].mask.shape == (10, 20)


def test_postprocess_predictions_filters_scores_uses_masks_and_applies_nms():
    masks = np.zeros((3, 12, 12), dtype=bool)
    masks[0, 2:7, 2:7] = True
    masks[1, 2:7, 2:7] = True
    masks[2, 8:10, 8:10] = True
    arrays = PredictionArrays(
        scores=np.array([0.9, 0.8, 0.2], dtype=np.float32),
        boxes_cxcywh=np.array(
            [
                [0.4, 0.4, 0.4, 0.4],
                [0.4, 0.4, 0.4, 0.4],
                [0.8, 0.8, 0.1, 0.1],
            ],
            dtype=np.float32,
        ),
        masks=masks,
    )

    prediction = postprocess_predictions(
        image_id="img",
        arrays=arrays,
        original_size=(12, 12),
        score_threshold=0.5,
        nms_iou_threshold=0.5,
    )

    assert len(prediction.instances) == 1
    assert prediction.instances[0].score == 0.9
    assert prediction.instances[0].obb == OrientedBox(
        center=(4.0, 4.0),
        size=(5.0, 5.0),
        angle_degrees=0.0,
    )
