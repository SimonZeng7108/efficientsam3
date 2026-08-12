"""把 DataTrain.txt 数据集转换为少样本训练 JSON。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ..config import apply_config_overrides, load_fewshot_config
from ..config.fewshot import FewShotExperimentConfig
from ..data.datatrain import DataTrainDataset


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DataTrain.txt to few-shot JSON files.")
    parser.add_argument("--config", help="少样本 YAML 配置路径。")
    parser.add_argument("--datatrain", help="DataTrain.txt 路径；优先覆盖 DATA.DATATRAIN。")
    parser.add_argument("--image-dir", help="图片所在目录；优先覆盖 DATA.IMAGE_DIR。")
    parser.add_argument("--output-dir", help="输出 JSON 目录；优先覆盖 DATA.OUTPUT_DIR。")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="批量处理：遍历 batch root 下每个子目录中的 DetectTrainData.txt。",
    )
    parser.add_argument(
        "--batch-root",
        help="批量处理根目录，例如 /home/data/public/datasets/fewshot_test_20260429。",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
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
    if args.batch:
        return _convert_batch(args=args, config=config)
    return _convert_one_from_config(config)


def _convert_one_from_config(config: FewShotExperimentConfig) -> int:
    """按最终配置转换单个 DataTrain 数据集。"""
    if not config.data.datatrain:
        raise ValueError("DataTrain path is required; pass --datatrain or DATA.DATATRAIN")
    if not config.data.image_dir:
        raise ValueError("image directory is required; pass --image-dir or DATA.IMAGE_DIR")
    _convert_one(
        datatrain=Path(config.data.datatrain),
        image_dir=Path(config.data.image_dir),
        output_dir=Path(config.data.output_dir),
    )
    print("Wrote full_gt.json and image_map.json")
    return 0


def _convert_batch(*, args: argparse.Namespace, config: FewShotExperimentConfig) -> int:
    """批量转换 batch root 下的多个子数据集。

    每个子目录形如：
    `dataset_name/DetectTrainData.txt + images`，输出到：
    `OUTPUT_DIR/dataset_name/full_gt.json` 和 `OUTPUT_DIR/dataset_name/image_map.json`。
    """
    root = _resolve_batch_root(args=args, config=config)
    if not root.is_dir():
        raise FileNotFoundError(f"batch root directory does not exist: {root}")

    datatrain_files = sorted(root.glob("*/DetectTrainData.txt"))
    if not datatrain_files:
        raise FileNotFoundError(f"No DetectTrainData.txt found under {root}")

    output_root = Path(config.data.output_dir)
    print(f"Found {len(datatrain_files)} datasets to process.")
    failures: list[tuple[str, Exception]] = []
    for datatrain in datatrain_files:
        dataset_dir = datatrain.parent
        dataset_name = dataset_dir.name
        output_dir = output_root / dataset_name
        print()
        print("=" * 60)
        print(f"Processing: {dataset_name}")
        print(f"  DataTrain: {datatrain}")
        print(f"  ImageDir:  {dataset_dir}")
        print(f"  OutputDir: {output_dir}")
        print("=" * 60)
        try:
            _convert_one(
                datatrain=datatrain,
                image_dir=dataset_dir,
                output_dir=output_dir,
            )
        except Exception as exc:  # noqa: BLE001 - 批量模式需要继续处理后续数据集。
            failures.append((dataset_name, exc))
            print(f"FAILED: {dataset_name}: {exc}")
            continue
        print("Wrote full_gt.json and image_map.json")
        print(f"SUCCESS: {dataset_name}")

    print()
    print("=" * 60)
    print(f"Batch processing complete. Processed {len(datatrain_files)} datasets.")
    if failures:
        print(f"Failed datasets: {len(failures)}")
        for dataset_name, exc in failures:
            print(f"  - {dataset_name}: {exc}")
        return 1
    return 0


def _convert_one(*, datatrain: Path, image_dir: Path, output_dir: Path) -> None:
    dataset = DataTrainDataset.from_file(
        datatrain,
        image_dir=image_dir,
    )
    dataset.save_json(output_dir)


def _resolve_batch_root(*, args: argparse.Namespace, config: FewShotExperimentConfig) -> Path:
    if args.batch_root:
        return Path(args.batch_root)
    if config.data.datatrain:
        return Path(config.data.datatrain).parent.parent
    if config.data.image_dir:
        return Path(config.data.image_dir).parent
    raise ValueError("batch mode requires --batch-root, --datatrain, or DATA.DATATRAIN")


if __name__ == "__main__":
    raise SystemExit(main())
