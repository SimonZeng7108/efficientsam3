"""测试 DataTrain.txt 转 JSON 的命令行工具。"""

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def test_convert_datatrain_cli_writes_required_json_files(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (64, 64), (0, 0, 0)).save(image_dir / "img_001.jpg")
    datatrain = tmp_path / "DataTrain.txt"
    datatrain.write_text('img_001.jpg 1 R 10 20 30 40 "car"\n', encoding="utf-8")
    output_dir = tmp_path / "json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fewshot_adapter.convert_datatrain",
            "--datatrain",
            str(datatrain),
            "--image-dir",
            str(image_dir),
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    full_gt = json.loads((output_dir / "full_gt.json").read_text(encoding="utf-8"))
    image_map = json.loads((output_dir / "image_map.json").read_text(encoding="utf-8"))
    assert full_gt[0]["object_id"] == "img_001_0001"
    assert image_map["img_001.jpg"].endswith("img_001.jpg")
    assert not (output_dir / "candidates.json").exists()
    assert "Wrote full_gt.json and image_map.json" in result.stdout


def test_convert_datatrain_cli_supports_real_colon_format_and_no_object_images(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (64, 64), (0, 0, 0)).save(image_dir / "valid.bmp.bmp")
    Image.new("RGB", (64, 64), (0, 0, 0)).save(image_dir / "empty.jpg.bmp")
    datatrain = tmp_path / "DataTrain.txt"
    datatrain.write_text(
        "\n".join(
            [
                "Version 1.0.0",
                'valid.bmp.bmp:1 P:4 10 10 20 10 20 20 10 20 "obj"',
                'empty.jpg.bmp:1 R:4 1 1 1 1 1 1 1 1 "obj"',
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fewshot_adapter.convert_datatrain",
            "--datatrain",
            str(datatrain),
            "--image-dir",
            str(image_dir),
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    full_gt = json.loads((output_dir / "full_gt.json").read_text(encoding="utf-8"))
    image_map = json.loads((output_dir / "image_map.json").read_text(encoding="utf-8"))
    assert [item["image_id"] for item in full_gt] == ["valid.bmp.bmp"]
    assert full_gt[0]["polygon"] == [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]
    assert set(image_map) == {"valid.bmp.bmp", "empty.jpg.bmp"}


def test_convert_datatrain_cli_accepts_yaml_config(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (64, 64), (0, 0, 0)).save(image_dir / "img_001.jpg")
    datatrain = tmp_path / "DataTrain.txt"
    datatrain.write_text('img_001.jpg:1 P:4 10 10 20 10 20 20 10 20 "obj"\n', encoding="utf-8")
    output_dir = tmp_path / "json"
    config = tmp_path / "fewshot.yaml"
    config.write_text(
        "\n".join(
            [
                "DATA:",
                f"  DATATRAIN: {datatrain.as_posix()}",
                f"  IMAGE_DIR: {image_dir.as_posix()}",
                f"  OUTPUT_DIR: {output_dir.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fewshot_adapter.convert_datatrain",
            "--config",
            str(config),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert (output_dir / "full_gt.json").exists()
    assert (output_dir / "image_map.json").exists()


def test_convert_datatrain_cli_batch_writes_one_folder_per_dataset(tmp_path):
    """批量模式遍历每个子数据集，并把输出写到 dataset_name 子目录。"""
    batch_root = tmp_path / "fewshot_test_20260429"
    first = batch_root / "12356_工件定位"
    second = batch_root / "24q4_jindianmilk"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    Image.new("RGB", (64, 64), (0, 0, 0)).save(first / "a.jpg.bmp")
    Image.new("RGB", (64, 64), (0, 0, 0)).save(second / "b.jpg.bmp")
    (first / "DetectTrainData.txt").write_text(
        "\n".join(
            [
                "Version 1.0.0",
                'a.jpg.bmp:1 R:4 10 10 20 10 20 20 10 20 "Sample"',
            ]
        ),
        encoding="utf-8",
    )
    (second / "DetectTrainData.txt").write_text(
        "\n".join(
            [
                "Version 1.0.0",
                'b.jpg.bmp:1 R:4 5 5 15 5 15 15 5 15 "obj"',
            ]
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "dataset_json"
    config = tmp_path / "fewshot.yaml"
    config.write_text(
        "\n".join(
            [
                "DATA:",
                f"  OUTPUT_DIR: {output_root.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fewshot_adapter.convert_datatrain",
            "--batch",
            "--batch-root",
            str(batch_root),
            "--config",
            str(config),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    first_gt = json.loads((output_root / "12356_工件定位" / "full_gt.json").read_text(encoding="utf-8"))
    second_map = json.loads((output_root / "24q4_jindianmilk" / "image_map.json").read_text(encoding="utf-8"))
    assert first_gt[0]["label"] == "Sample"
    assert second_map["b.jpg.bmp"].endswith("b.jpg.bmp")
    assert "Found 2 datasets to process." in result.stdout
    assert "SUCCESS: 12356_工件定位" in result.stdout
