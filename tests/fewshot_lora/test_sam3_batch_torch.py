import pytest

torch = pytest.importorskip("torch")

from fewshot_lora.sam3_batch import FindBatchSample, build_batched_datapoint


def test_build_batched_datapoint_shapes():
    image = torch.zeros((3, 8, 8), dtype=torch.float32)
    mask = torch.zeros((8, 8), dtype=torch.bool)
    mask[2:5, 2:5] = True
    sample = FindBatchSample(
        image=image,
        image_id="img",
        original_size=(8, 8),
        target_boxes=[(0.5, 0.5, 0.25, 0.25)],
        target_masks=[mask],
        prompt_boxes=[(0.5, 0.5, 0.25, 0.25)],
        text="Sample",
    )

    batch = build_batched_datapoint([sample], device=torch.device("cpu"))

    assert batch.img_batch.shape == (1, 3, 8, 8)
    assert batch.find_inputs[0].input_boxes.shape == (1, 1, 4)
    assert batch.find_inputs[0].input_boxes_mask.shape == (1, 1)
    assert batch.find_targets[0].boxes.shape == (1, 4)
    assert batch.find_targets[0].boxes_padded.shape == (1, 1, 4)
    assert batch.find_targets[0].segments.shape == (1, 8, 8)
