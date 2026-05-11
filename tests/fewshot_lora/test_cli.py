from types import ModuleType, SimpleNamespace
import sys

from fewshot_lora.cli import main, parse_args


def test_parse_args_keeps_smoke_annotation_filename_configurable():
    args = parse_args(
        [
            "--dataset-list",
            "datasets.txt",
            "--output-dir",
            "runs",
            "--annotation-filename",
            "DetectTrainData_sample5.txt",
            "--max-rounds",
            "2",
        ]
    )

    assert args.dataset_list == "datasets.txt"
    assert args.output_dir == "runs"
    assert args.annotation_filename == "DetectTrainData_sample5.txt"
    assert args.max_rounds == 2


def test_main_prints_chinese_summary(monkeypatch, capsys):
    fake_runner = ModuleType("fewshot_lora.runtime.runner")

    def fake_run_from_dataset_list(dataset_list_path, config):
        return SimpleNamespace(
            dataset_summaries=[
                SimpleNamespace(dataset_name="demo", success=True, rounds=[object()])
            ]
        )

    fake_runner.run_from_dataset_list = fake_run_from_dataset_list
    monkeypatch.setitem(sys.modules, "fewshot_lora.runtime.runner", fake_runner)

    exit_code = main(["--dataset-list", "datasets.txt", "--output-dir", "runs"])

    assert exit_code == 0
    assert "数据集 demo：成功，轮数=1" in capsys.readouterr().out
