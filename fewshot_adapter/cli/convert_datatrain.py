"""把 DataTrain.txt 数据集转换为少样本训练 JSON。"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..data.datatrain import DataTrainDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DataTrain.txt to few-shot JSON files.")
    parser.add_argument("--datatrain", required=True, help="DataTrain.txt 路径。")
    parser.add_argument("--image-dir", required=True, help="图片所在目录。")
    parser.add_argument("--output-dir", required=True, help="输出 JSON 目录。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = DataTrainDataset.from_file(args.datatrain, image_dir=args.image_dir)
    dataset.save_json(Path(args.output_dir))
    print("Wrote full_gt.json and image_map.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
