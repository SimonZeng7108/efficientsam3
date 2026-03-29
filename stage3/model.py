import logging
import types

import torch.nn as nn

from sam3.model_builder import build_efficientsam3_image_model


LOGGER = logging.getLogger(__name__)


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad


def _freeze_semantic_seg_head(model: nn.Module) -> None:
    """No grad / no DDP reduction for semantic head when semantic loss is off."""
    seg = getattr(model, "segmentation_head", None)
    if seg is not None and hasattr(seg, "semantic_seg_head"):
        _set_requires_grad(seg.semantic_seg_head, False)


def _freeze_non_encoder_parameters(
    model: nn.Module,
    train_vision_encoder: bool,
    train_text_encoder: bool,
) -> None:
    _set_requires_grad(model, False)

    if train_vision_encoder:
        _set_requires_grad(model.backbone.vision_backbone.trunk, True)
    if train_text_encoder:
        _set_requires_grad(model.backbone.language_backbone, True)


def _summarize_parameters(model: nn.Module) -> None:
    total_params = 0
    trainable_params = 0
    vision_params = 0
    text_params = 0

    for name, parameter in model.named_parameters():
        numel = parameter.numel()
        total_params += numel
        if parameter.requires_grad:
            trainable_params += numel
            if name.startswith("backbone.vision_backbone.trunk."):
                vision_params += numel
            elif name.startswith("backbone.language_backbone."):
                text_params += numel

    pct = 100.0 * trainable_params / max(total_params, 1)
    LOGGER.info(
        "Stage3 trainable params: %.2fM / %.2fM (%.2f%%), vision=%.2fM, text=%.2fM",
        trainable_params / 1e6,
        total_params / 1e6,
        pct,
        vision_params / 1e6,
        text_params / 1e6,
    )


def _set_stage3_module_modes(model: nn.Module, mode: bool) -> None:
    if not getattr(model, "_stage3_keep_frozen_modules_eval", True):
        model._stage3_original_train(mode)
        return

    model._stage3_original_train(False)
    model.training = mode
    if not mode:
        return

    if getattr(model, "_stage3_train_vision_encoder", True):
        model.backbone.vision_backbone.trunk.train()
    if getattr(model, "_stage3_train_text_encoder", True):
        model.backbone.language_backbone.train()


def _attach_stage3_training_behavior(
    model: nn.Module,
    train_vision_encoder: bool,
    train_text_encoder: bool,
    keep_frozen_modules_eval: bool,
) -> None:
    model._stage3_original_train = model.train
    model._stage3_train_vision_encoder = train_vision_encoder
    model._stage3_train_text_encoder = train_text_encoder
    model._stage3_keep_frozen_modules_eval = keep_frozen_modules_eval

    def _stage3_train(self: nn.Module, mode: bool = True) -> nn.Module:
        _set_stage3_module_modes(self, mode)
        return self

    model.train = types.MethodType(_stage3_train, model)


def build_stage3_model(
    bpe_path=None,
    device=None,
    eval_mode=False,
    checkpoint_path=None,
    load_from_HF=False,
    enable_segmentation=True,
    enable_inst_interactivity=False,
    backbone_type="tinyvit",
    model_name="11m",
    text_encoder_type="MobileCLIP-S1",
    text_encoder_context_length=16,
    text_encoder_pos_embed_table_size=16,
    interpolate_pos_embed=False,
    train_vision_encoder=True,
    train_text_encoder=True,
    freeze_non_encoder_parameters=True,
    keep_frozen_modules_eval=True,
    log_parameter_summary=True,
    freeze_semantic_seg_head=False,
):
    model = build_efficientsam3_image_model(
        bpe_path=bpe_path,
        device=device,
        eval_mode=eval_mode,
        checkpoint_path=checkpoint_path,
        load_from_HF=load_from_HF,
        enable_segmentation=enable_segmentation,
        enable_inst_interactivity=enable_inst_interactivity,
        backbone_type=backbone_type,
        model_name=model_name,
        text_encoder_type=text_encoder_type,
        text_encoder_context_length=text_encoder_context_length,
        text_encoder_pos_embed_table_size=text_encoder_pos_embed_table_size,
        interpolate_pos_embed=interpolate_pos_embed,
    )

    if freeze_semantic_seg_head:
        _freeze_semantic_seg_head(model)

    if freeze_non_encoder_parameters:
        _freeze_non_encoder_parameters(
            model,
            train_vision_encoder=train_vision_encoder,
            train_text_encoder=train_text_encoder,
        )
        _attach_stage3_training_behavior(
            model,
            train_vision_encoder=train_vision_encoder,
            train_text_encoder=train_text_encoder,
            keep_frozen_modules_eval=keep_frozen_modules_eval,
        )

    if log_parameter_summary:
        _summarize_parameters(model)

    model.train(not eval_mode)
    return model
