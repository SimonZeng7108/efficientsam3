"""EfficientSAM3 原生少样本训练层。"""

from .adapter import (
    NativeAdapterConfig,
    NativeEfficientSAM3FewShotModel,
    build_native_fewshot_model,
    freeze_for_fewshot,
    save_native_adapter,
    should_train_parameter,
)
from .loss import NativeLossConfig, NativeLossFactory, build_native_loss
from .predictor import (
    NativePredictionRecord,
    NativePredictor,
    hbb_to_zero_angle_obb,
    native_outputs_to_predictions,
    record_to_prediction,
    tensor_box_to_hbb,
)
from .trainer import (
    NativeFewShotLoopConfig,
    NativeFewShotTrainer,
    add_selected_image_truth,
    add_selected_training_sample,
    run_native_fewshot_loop,
)

__all__ = [
    "NativeAdapterConfig",
    "NativeEfficientSAM3FewShotModel",
    "NativeFewShotLoopConfig",
    "NativeFewShotTrainer",
    "NativeLossConfig",
    "NativeLossFactory",
    "NativePredictionRecord",
    "NativePredictor",
    "add_selected_image_truth",
    "add_selected_training_sample",
    "build_native_fewshot_model",
    "build_native_loss",
    "freeze_for_fewshot",
    "hbb_to_zero_angle_obb",
    "native_outputs_to_predictions",
    "record_to_prediction",
    "run_native_fewshot_loop",
    "save_native_adapter",
    "should_train_parameter",
    "tensor_box_to_hbb",
]
