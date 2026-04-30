"""把 DataTrain.txt 数据集转换为少样本训练 JSON。"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import apply_config_overrides, load_fewshot_config
from ..data.datatrain import DataTrainDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DataTrain.txt to few-shot JSON files.")
    parser.add_argument("--config", help="少样本 YAML 配置路径。")
    parser.add_argument("--datatrain", help="DataTrain.txt 路径；优先覆盖 DATA.DATATRAIN。")
    parser.add_argument("--image-dir", help="图片所在目录；优先覆盖 DATA.IMAGE_DIR。")
    parser.add_argument("--output-dir", help="输出 JSON 目录；优先覆盖 DATA.OUTPUT_DIR。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_fewshot_config(args.config)
    config = apply_config_overrides(
        config,
        {
            "DATA": {
                "DATATRAIN": args.datatrain,
                "IMAGE_DIR": args.image_dir,
                "OUTPUT_DIR": args.output_dir,
            }
        },
    )
    if not config.data.datatrain:
        raise ValueError("DataTrain path is required; pass --datatrain or DATA.DATATRAIN")
    if not config.data.image_dir:
        raise ValueError("image directory is required; pass --image-dir or DATA.IMAGE_DIR")
    dataset = DataTrainDataset.from_file(
        config.data.datatrain,
        image_dir=config.data.image_dir,
    )
    dataset.save_json(Path(config.data.output_dir))
    print("Wrote full_gt.json and image_map.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
