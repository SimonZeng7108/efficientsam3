from pathlib import Path

from fewshot_lora.config import FewShotLoRAConfig
from fewshot_lora.eval.metrics import ImageEval
from fewshot_lora.runtime.loop import TrainRoundOutput, run_dataset_loop


def test_config_creates_dataset_output_dir(tmp_path: Path):
    config = FewShotLoRAConfig(output_dir=tmp_path / "runs")

    assert config.dataset_output_dir(Path("C:/data/My Dataset")).name == "My_Dataset"


def test_run_dataset_loop_adds_initial_image_then_failure_image(tmp_path: Path):
    config = FewShotLoRAConfig(output_dir=tmp_path, max_rounds=3)
    calls: list[list[str]] = []

    def train_fn(train_image_ids, round_index, adapter_path):
        calls.append(list(train_image_ids))
        return adapter_path

    evals = [
        [
            ImageEval(
                image_id="bad",
                matches=(),
                localization_errors=(),
                false_positive_indices=(),
                false_negative_indices=(0,),
            )
        ],
        [
            ImageEval(
                image_id="bad",
                matches=(),
                localization_errors=(),
                false_positive_indices=(),
                false_negative_indices=(),
            )
        ],
    ]

    def evaluate_fn(round_index, adapter_path):
        return evals[round_index]

    summary = run_dataset_loop(
        dataset_name="demo",
        image_ids=["seed", "bad"],
        config=config,
        train_round=train_fn,
        evaluate_round=evaluate_fn,
        initial_selector=lambda ids: ids[0],
    )

    assert calls == [["seed"], ["seed", "bad"]]
    assert summary.success
    assert summary.rounds[-1].precision == 1.0
    assert summary.rounds[0].next_image_id == "bad"


def test_run_dataset_loop_records_training_stats_when_train_round_returns_output(tmp_path: Path):
    config = FewShotLoRAConfig(output_dir=tmp_path, max_rounds=1)

    def train_fn(train_image_ids, round_index, adapter_path):
        return TrainRoundOutput(
            adapter_path=adapter_path,
            train_image_count=len(train_image_ids),
            train_instance_count=3,
            train_seconds=1.25,
            train_steps=7,
        )

    summary = run_dataset_loop(
        dataset_name="demo",
        image_ids=["seed"],
        config=config,
        train_round=train_fn,
        evaluate_round=lambda round_index, adapter_path: [
            ImageEval(
                image_id="seed",
                matches=(),
                localization_errors=(),
                false_positive_indices=(),
                false_negative_indices=(),
            )
        ],
    )

    assert summary.rounds[0].train_image_count == 1
    assert summary.rounds[0].train_instance_count == 3
    assert summary.rounds[0].train_seconds == 1.25
    assert summary.rounds[0].train_steps == 7
