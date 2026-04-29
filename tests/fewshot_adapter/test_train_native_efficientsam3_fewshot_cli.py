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
