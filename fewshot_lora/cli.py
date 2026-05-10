"""命令行入口：批量运行少样本 LoRA 交互式同类目标查找实验。

这个文件尽量保持“轻量导入”：只解析参数和组装配置；真正需要 torch/SAM3
的 runner 在 `main()` 内部才导入。这样在没有 torch 的本地机器上，也能测试
CLI 参数解析和中文输出格式。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import EvaluationConfig, FewShotLoRAConfig, LoRAConfig, ModelConfig, TrainingConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。

    说明：这里的 help 文案全部使用中文，便于在服务器上直接 `--help` 查看。
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-list", required=True, help="数据集列表 txt，每行一个子数据集目录。")
    parser.add_argument("--output-dir", required=True, help="实验输出目录。")
    parser.add_argument("--annotation-filename", default="DetectTrainData.txt", help="标注文件名，可设为 DetectTrainData_sample5.txt 做快速测试。")
    parser.add_argument("--checkpoint-path", default=None, help="EfficientSAM3 checkpoint 路径。")
    parser.add_argument("--device", default="cuda", help="运行设备，例如 cuda 或 cpu。")
    parser.add_argument("--max-rounds", type=int, default=5, help="每个子数据集最大交互轮数。")
    parser.add_argument("--max-steps-per-round", type=int, default=80, help="每轮最多训练 step 数。")
    parser.add_argument("--max-seconds-per-round", type=float, default=60.0, help="每轮最多训练秒数。")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="预测分数阈值。")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="OBB IoU 成功匹配阈值。")
    parser.add_argument("--localization-iou-threshold", type=float, default=0.1, help="定位错误判定的最低重叠阈值。")
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5, help="单类别 OBB NMS 阈值。")
    parser.add_argument("--text-prompt", default="object", help="无 label 时使用的兜底文本 prompt。")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank。")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha 缩放参数。")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> FewShotLoRAConfig:
    """把 argparse 结果转换成核心 dataclass 配置。"""

    checkpoint_path = None if args.checkpoint_path is None else Path(args.checkpoint_path)
    return FewShotLoRAConfig(
        output_dir=Path(args.output_dir),
        max_rounds=args.max_rounds,
        annotation_filename=args.annotation_filename,
        device=args.device,
        model=ModelConfig(checkpoint_path=checkpoint_path),
        lora=LoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha),
        training=TrainingConfig(
            max_steps_per_round=args.max_steps_per_round,
            max_seconds_per_round=args.max_seconds_per_round,
        ),
        evaluation=EvaluationConfig(
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
            localization_iou_threshold=args.localization_iou_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
            text_prompt=args.text_prompt,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI 主流程：运行批量实验并打印中文摘要。"""

    args = parse_args(argv)
    config = config_from_args(args)
    from .runner import run_from_dataset_list

    summary = run_from_dataset_list(Path(args.dataset_list), config)
    for dataset_summary in summary.dataset_summaries:
        status = "成功" if dataset_summary.success else "未收敛"
        print(f"数据集 {dataset_summary.dataset_name}：{status}，轮数={len(dataset_summary.rounds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
