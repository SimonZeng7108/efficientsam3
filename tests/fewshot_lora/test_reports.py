import json
from pathlib import Path

from fewshot_lora.config import FewShotLoRAConfig
from fewshot_lora.data.datatrain import DataIssue
from fewshot_lora.runtime.loop import DatasetRunSummary, RoundSummary
from fewshot_lora.runtime.reports import write_dataset_summary


def test_write_dataset_summary_serializes_rounds_and_data_issues(tmp_path: Path):
    summary = DatasetRunSummary(
        dataset_name="demo",
        success=False,
        rounds=(
            RoundSummary(
                round_index=0,
                train_image_ids=("seed.bmp",),
                adapter_path=tmp_path / "adapter.pt",
                train_image_count=1,
                train_instance_count=2,
                train_seconds=3.5,
                train_steps=4,
                precision=0.5,
                recall=1.0,
                f1=2 / 3,
                mean_obb_iou=0.8,
                false_positive_count=1,
                false_negative_count=0,
                localization_error_count=0,
                next_image_id="next.bmp",
            ),
        ),
    )
    issues = [DataIssue("count_mismatch", "seed.bmp", 2, "bad count")]

    out_path = write_dataset_summary(tmp_path / "summary.json", FewShotLoRAConfig(output_dir=tmp_path), summary, issues)

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["dataset_name"] == "demo"
    assert payload["success"] is False
    assert payload["rounds"][0]["train_instance_count"] == 2
    assert payload["rounds"][0]["next_image_id"] == "next.bmp"
    assert payload["data_issues"][0]["kind"] == "count_mismatch"
