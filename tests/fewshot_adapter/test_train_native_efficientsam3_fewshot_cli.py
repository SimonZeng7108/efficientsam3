"""测试 EfficientSAM3 原生少样本训练 CLI 的轻量失败路径。"""

import json
import subprocess
import sys
from pathlib import Path


def test_train_native_cli_reports_clear_error_without_torch(tmp_path):
    """当前轻量环境没装 torch 时，CLI 仍应完成参数解析并给出明确提示。"""
    full_gt = tmp_path / "full_gt.json"
    image_map = tmp_path / "image_map.json"
    output_root = tmp_path / "runs"
    full_gt.write_text(
        json.dumps(
            [
                {
                    "image_id": "img.png",
                    "object_id": "gt_1",
                    "label": "target",
                    "source_type": "hbb",
                    "hbb": [0, 0, 10, 10],
                }
            ]
        ),
        encoding="utf-8",
    )
    image_map.write_text(json.dumps({"img.png": str(tmp_path / "img.png")}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fewshot_adapter.train_native_efficientsam3_fewshot",
            "--full-ground-truth",
            str(full_gt),
            "--image-map",
            str(image_map),
            "--checkpoint",
            "efficient_sam3_efficientvit_s.pt",
            "--output-root",
            str(output_root),
            "--max-rounds",
            "1",
            "--steps-per-round",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "PyTorch is required" in result.stderr


def test_train_native_cli_accepts_yaml_config_and_cli_overrides_before_torch(tmp_path):
    """配置文件负责默认参数，命令行仍可临时覆盖实验参数。"""
    full_gt = tmp_path / "full_gt.json"
    image_map = tmp_path / "image_map.json"
    output_root = tmp_path / "runs"
    config = tmp_path / "fewshot.yaml"
    full_gt.write_text(
        json.dumps(
            [
                {
                    "image_id": "img.png",
                    "object_id": "gt_1",
                    "label": "obj",
                    "source_type": "hbb",
                    "hbb": [0, 0, 10, 10],
                }
            ]
        ),
        encoding="utf-8",
    )
    image_map.write_text(json.dumps({"img.png": str(tmp_path / "img.png")}), encoding="utf-8")
    config.write_text(
        "\n".join(
            [
                "DATA:",
                f"  FULL_GROUND_TRUTH: {full_gt.as_posix()}",
                f"  IMAGE_MAP: {image_map.as_posix()}",
                "MODEL:",
                "  CHECKPOINT: efficient_sam3_efficientvit_s.pt",
                "TRAIN:",
                f"  OUTPUT_ROOT: {output_root.as_posix()}",
                "  STEPS_PER_ROUND: 80",
                "EVAL:",
                "  LABEL: obj",
                "  SCORE_THRESHOLD: 0.3",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fewshot_adapter.train_native_efficientsam3_fewshot",
            "--config",
            str(config),
            "--steps-per-round",
            "2",
            "--score-threshold",
            "0.2",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    resolved_config = (output_root / "resolved_config.yaml").read_text(encoding="utf-8")
    assert result.returncode != 0
    assert "PyTorch is required" in result.stderr
    assert "STEPS_PER_ROUND: 2" in resolved_config
    assert "SCORE_THRESHOLD: 0.2" in resolved_config
