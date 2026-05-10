from pathlib import Path

import numpy as np

from fewshot_lora.config import FewShotLoRAConfig
from fewshot_lora.dataset import PreparedImage, PreparedInstance
from fewshot_lora.geometry import OrientedBox, Polygon4
from fewshot_lora.inference import _dataset_text_prompt


def _instance(label: str) -> PreparedInstance:
    polygon = Polygon4(((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)))
    return PreparedInstance(
        polygon=polygon,
        label=label,
        aabb=(0.0, 0.0, 4.0, 4.0),
        box_cxcywh=(0.5, 0.5, 1.0, 1.0),
        mask=np.ones((4, 4), dtype=bool),
        obb=OrientedBox(center=(2.0, 2.0), size=(4.0, 4.0), angle_degrees=0.0),
    )


def test_dataset_text_prompt_uses_single_category_label_for_negative_images(tmp_path: Path):
    config = FewShotLoRAConfig(output_dir=tmp_path)
    negative = PreparedImage("negative.bmp", tmp_path / "negative.bmp", 10, 10, [])
    positive = PreparedImage("positive.bmp", tmp_path / "positive.bmp", 10, 10, [_instance("Sample")])

    assert _dataset_text_prompt([negative, positive], config) == "Sample"
