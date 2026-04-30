"""EfficientSAM3 原生少样本 Adapter 训练 CLI。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..config import (
    apply_config_overrides,
    build_adapter_config,
    build_loop_config,
    build_loss_config,
    load_fewshot_config,
    save_fewshot_config,
)
from ..native.trainer import run_native_fewshot_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train native EfficientSAM3 few-shot adapter without proposal candidates."
    )
    parser.add_argument("--config", help="少样本 YAML 配置路径。")
    parser.add_argument("--full-ground-truth", help="全量真值标注 JSON；优先覆盖 DATA.FULL_GROUND_TRUTH。")
    parser.add_argument("--image-map", help="image_id 到图片路径的 JSON 映射；优先覆盖 DATA.IMAGE_MAP。")
    parser.add_argument("--checkpoint", help="EfficientSAM3 checkpoint；优先覆盖 MODEL.CHECKPOINT。")
    parser.add_argument("--output-root", help="训练输出目录；优先覆盖 TRAIN.OUTPUT_ROOT。")
    parser.add_argument("--label", help="目标类别；优先覆盖 EVAL.LABEL。")
    parser.add_argument("--device", help="训练设备；优先覆盖 MODEL.DEVICE。")
    parser.add_argument("--resolution", type=int, help="EfficientSAM3 输入分辨率；优先覆盖 DATA.IMG_SIZE。")
    parser.add_argument("--backbone-type", help="EfficientSAM3 backbone 类型；优先覆盖 MODEL.BACKBONE_TYPE。")
    parser.add_argument("--model-name", help="EfficientSAM3 backbone 变体；优先覆盖 MODEL.MODEL_NAME。")
    parser.add_argument(
        "--enable-segmentation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否构建 SAM3 segmentation head；优先覆盖 MODEL.ENABLE_SEGMENTATION。",
    )
    parser.add_argument("--seed", type=int, help="初始样本选择随机种子。")
    parser.add_argument("--max-rounds", type=int, help="最大自动闭合轮数。")
    parser.add_argument("--steps-per-round", type=int, help="每轮 adapter 训练步数。")
    parser.add_argument("--learning-rate", type=float, help="adapter 学习率。")
    parser.add_argument("--weight-decay", type=float, help="adapter weight decay。")
    parser.add_argument("--score-threshold", type=float, help="推理置信度阈值。")
    parser.add_argument("--iou-threshold", type=float, help="判定匹配成功的 IoU 阈值。")
    parser.add_argument(
        "--localization-error-threshold",
        type=float,
        help="定位错误 IoU 下限。",
    )
    parser.add_argument("--iou-mode", choices=["hbb", "obb", "polygon"])
    parser.add_argument("--num-prompt-tokens", type=int, help="任务 prompt token 数。")
    parser.add_argument("--prompt-adapter-dim", type=int, help="prompt adapter bottleneck 维度。")
    parser.add_argument(
        "--train-dot-prod-scoring",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否训练 SAM3 dot_prod_scoring；优先覆盖 ADAPTER.TRAIN_DOT_PROD_SCORING。",
    )
    parser.add_argument(
        "--train-bbox-embed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="除 prompt/dot_prod_scoring 外，额外开放 decoder bbox_embed。",
    )
    parser.add_argument(
        "--train-decoder-cross-attention",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="额外开放 decoder cross attention 小范围参数；显存和过拟合风险更高。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_fewshot_config(args.config)
    config = apply_config_overrides(
        config,
        {
            "DATA": {
                "FULL_GROUND_TRUTH": args.full_ground_truth,
                "IMAGE_MAP": args.image_map,
                "IMG_SIZE": args.resolution,
            },
            "MODEL": {
                "CHECKPOINT": args.checkpoint,
                "DEVICE": args.device,
                "BACKBONE_TYPE": args.backbone_type,
                "MODEL_NAME": args.model_name,
                "ENABLE_SEGMENTATION": args.enable_segmentation,
            },
            "ADAPTER": {
                "NUM_PROMPT_TOKENS": args.num_prompt_tokens,
                "PROMPT_ADAPTER_DIM": args.prompt_adapter_dim,
                "TRAIN_DOT_PROD_SCORING": args.train_dot_prod_scoring,
                "TRAIN_BBOX_EMBED": args.train_bbox_embed,
                "TRAIN_DECODER_CROSS_ATTENTION": args.train_decoder_cross_attention,
            },
            "TRAIN": {
                "OUTPUT_ROOT": args.output_root,
                "SEED": args.seed,
                "MAX_ROUNDS": args.max_rounds,
                "STEPS_PER_ROUND": args.steps_per_round,
                "LEARNING_RATE": args.learning_rate,
                "WEIGHT_DECAY": args.weight_decay,
            },
            "EVAL": {
                "LABEL": args.label,
                "SCORE_THRESHOLD": args.score_threshold,
                "IOU_THRESHOLD": args.iou_threshold,
                "LOCALIZATION_ERROR_THRESHOLD": args.localization_error_threshold,
                "IOU_MODE": args.iou_mode,
            },
        },
    )
    loop_config = build_loop_config(config)
    adapter_config = build_adapter_config(config)
    loss_config = build_loss_config(config)
    save_fewshot_config(Path(loop_config.output_root) / "resolved_config.yaml", config)
    try:
        summary = run_native_fewshot_loop(
            full_ground_truth_path=config.data.full_ground_truth,
            image_map_path=config.data.image_map,
            config=loop_config,
            adapter_config=adapter_config,
            loss_config=loss_config,
        )
    except ModuleNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
