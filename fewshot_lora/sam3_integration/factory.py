"""构建可训练 EfficientSAM3 LoRA 模型的工厂函数。

把模型构建集中在这里有两个好处：
1. `runtime.runner` 只负责“什么时候训练/评估”，不关心 SAM3 构建细节。
2. 服务器调参时可以只改这个文件，替换 checkpoint、LoRA target 或 loss 组合。
"""

from __future__ import annotations

from ..config import FewShotLoRAConfig


def build_trainable_model(config: FewShotLoRAConfig):
    """构建 EfficientSAM3 image model 并只开放 LoRA 参数训练。"""

    # 延迟导入 SAM3 builder，避免没有 torch/SAM3 依赖时无法导入 CLI 和纯逻辑模块。
    from sam3.model_builder import build_efficientsam3_image_model

    from .losses import build_sam3_find_loss
    from .lora import apply_efficientvit_lora

    model = build_efficientsam3_image_model(
        backbone_type=config.model.backbone_type,
        model_name=config.model.model_name,
        enable_segmentation=config.model.enable_segmentation,
        eval_mode=False,
        checkpoint_path=None if config.model.checkpoint_path is None else str(config.model.checkpoint_path),
        load_from_HF=False,
        device=config.device,
        text_encoder_type=config.model.text_encoder_type,
    )
    lora_report = apply_efficientvit_lora(
        model,
        target_suffixes=config.lora.target_suffixes,
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
    )
    loss_fn = build_sam3_find_loss(
        enable_mask_loss=config.training.enable_mask_loss,
        normalization=config.training.loss_normalization,
    )
    return model, loss_fn, lora_report


# 兼容旧的内部调用名；新代码应导入公开的 `build_trainable_model`。
_build_trainable_model = build_trainable_model
