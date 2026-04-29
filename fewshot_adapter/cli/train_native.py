"""EfficientSAM3 原生少样本 Adapter 训练 CLI。"""

from __future__ import annotations

import argparse
import json
import sys

from ..native.adapter import NativeAdapterConfig
from ..native.trainer import NativeFewShotLoopConfig, run_native_fewshot_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train native EfficientSAM3 few-shot adapter without proposal candidates."
    )
    parser.add_argument("--full-ground-truth", required=True, help="全量真值标注 JSON。")
    parser.add_argument("--image-map", required=True, help="image_id 到图片路径的 JSON 映射。")
    parser.add_argument("--checkpoint", required=True, help="efficient_sam3_efficientvit_s.pt 路径。")
    parser.add_argument("--output-root", required=True, help="训练输出目录。")
    parser.add_argument("--label", help="目标类别；不传则使用全量真值第一类。")
    parser.add_argument("--device", default="cuda", help="训练设备。")
    parser.add_argument("--resolution", type=int, default=1008, help="EfficientSAM3 输入分辨率。")
    parser.add_argument("--seed", type=int, default=0, help="初始样本选择随机种子。")
    parser.add_argument("--max-rounds", type=int, default=10, help="最大自动闭合轮数。")
    parser.add_argument("--steps-per-round", type=int, default=80, help="每轮 adapter 训练步数。")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="adapter 学习率。")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="adapter weight decay。")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="推理置信度阈值。")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="判定匹配成功的 IoU 阈值。")
    parser.add_argument(
        "--localization-error-threshold",
        type=float,
        default=0.1,
        help="定位错误 IoU 下限。",
    )
    parser.add_argument("--iou-mode", choices=["hbb", "obb", "polygon"], default="hbb")
    parser.add_argument("--num-prompt-tokens", type=int, default=8, help="任务 prompt token 数。")
    parser.add_argument(
        "--train-bbox-embed",
        action="store_true",
        help="除 prompt/dot_prod_scoring 外，额外开放 decoder bbox_embed。",
    )
    parser.add_argument(
        "--train-decoder-cross-attention",
        action="store_true",
        help="额外开放 decoder cross attention 小范围参数；显存和过拟合风险更高。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    loop_config = NativeFewShotLoopConfig(
        checkpoint=args.checkpoint,
        output_root=args.output_root,
        label=args.label,
        device=args.device,
        resolution=args.resolution,
        seed=args.seed,
        max_rounds=args.max_rounds,
        steps_per_round=args.steps_per_round,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        localization_error_threshold=args.localization_error_threshold,
        iou_mode=args.iou_mode,
    )
    adapter_config = NativeAdapterConfig(
        num_prompt_tokens=args.num_prompt_tokens,
        train_bbox_embed=args.train_bbox_embed,
        train_decoder_cross_attention=args.train_decoder_cross_attention,
    )
    try:
        summary = run_native_fewshot_loop(
            full_ground_truth_path=args.full_ground_truth,
            image_map_path=args.image_map,
            config=loop_config,
            adapter_config=adapter_config,
        )
    except ModuleNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
