"""EfficientSAM3 原生少样本自动闭环训练。

验证阶段没有交互界面，因此这里会自动选择错误样本并把真值加入下一轮。
训练本身使用 SAM3 原生输出和原生 loss，不再依赖 proposal_candidates.json。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ..data.json_io import load_annotations, save_error_queue, save_predictions, save_training_samples
from ..data.models import Annotation, Prediction, TrainingSample
from ..data.sam3_batch import build_sam3_training_batch, build_sam3_training_batch_from_samples, load_image_batch
from ..data.sampling import (
    add_selected_errors_to_training_samples,
    annotations_to_training_samples,
    create_initial_training_samples,
)
from ..evaluation.matching import build_error_queue, select_next_training_sample
from ..evaluation.metrics import compute_detection_metrics
from ..utils.torch import require_torch
from ..visualization.round_outputs import render_round_visualizations
from .adapter import NativeAdapterConfig, build_native_fewshot_model, save_native_adapter
from .loss import NativeLossConfig, build_native_loss
from .predictor import native_outputs_to_predictions

LogFn = Callable[[str], None]
_DEFAULT_LOG_EVERY = 10
_LOSS_LOG_KEYS = (
    "core_loss",
    "loss_ce",
    "loss_bbox",
    "loss_giou",
    "presence_loss",
)


@dataclass(frozen=True)
class NativeFewShotLoopConfig:
    """原生少样本闭环配置。"""

    checkpoint: str
    output_root: str
    label: str | None = None
    device: str = "cuda"
    resolution: int = 1008
    backbone_type: str = "efficientvit"
    model_name: str = "b0"
    enable_segmentation: bool = False
    seed: int = 0
    max_rounds: int = 10
    steps_per_round: int = 80
    learning_rate: float = 8e-5
    weight_decay: float = 0.1
    score_threshold: float = 0.3
    iou_threshold: float = 0.5
    localization_error_threshold: float = 0.1
    iou_mode: str = "hbb"


class NativeFewShotTrainer:
    """原生 EfficientSAM3 少样本闭环训练器。"""

    def __init__(
        self,
        config: NativeFewShotLoopConfig,
        *,
        adapter_config: NativeAdapterConfig | None = None,
        loss_config: NativeLossConfig | None = None,
    ):
        self.config = config
        self.adapter_config = adapter_config
        self.loss_config = loss_config

    def run(self, *, full_ground_truth_path: str | Path, image_map_path: str | Path) -> dict[str, Any]:
        return run_native_fewshot_loop(
            full_ground_truth_path=full_ground_truth_path,
            image_map_path=image_map_path,
            config=self.config,
            adapter_config=self.adapter_config,
            loss_config=self.loss_config,
        )


def run_native_fewshot_loop(
    *,
    full_ground_truth_path: str | Path,
    image_map_path: str | Path,
    config: NativeFewShotLoopConfig,
    adapter_config: NativeAdapterConfig | None = None,
    loss_config: NativeLossConfig | None = None,
    log_fn: LogFn | None = print,
) -> dict[str, Any]:
    """执行完整 EfficientSAM3 原生少样本闭环。"""
    resolved_loss_config = loss_config or NativeLossConfig()
    _validate_mask_training_config(config=config, loss_config=resolved_loss_config)
    torch = require_torch()
    from sam3.model.model_misc import SAM3Output

    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    full_ground_truth = load_annotations(full_ground_truth_path)
    image_map = _read_json_dict(image_map_path)
    target_label = _resolve_label(config.label, full_ground_truth)

    current_train = create_initial_training_samples(
        full_ground_truth,
        label=target_label,
        seed=config.seed,
    )
    save_training_samples(output_root / "train_round_0.json", current_train)

    wrapper = build_native_fewshot_model(
        checkpoint_path=config.checkpoint,
        config=adapter_config or NativeAdapterConfig(),
        device=config.device,
        resolution=config.resolution,
        backbone_type=config.backbone_type,
        model_name=config.model_name,
        enable_segmentation=config.enable_segmentation,
    )
    loss_fn = build_native_loss(resolved_loss_config)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in wrapper.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    _emit_logs(log_fn, _format_trainable_module_logs(wrapper))

    round_summaries = []
    for round_index in range(config.max_rounds):
        round_dir = output_root / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        _emit_log(
            log_fn,
            _format_round_start_log(
                round_index=round_index,
                max_rounds=config.max_rounds,
                samples=current_train,
                learning_rates=_optimizer_learning_rates(optimizer),
            ),
        )

        train_history = train_native_adapter_one_round(
            wrapper=wrapper,
            optimizer=optimizer,
            loss_fn=loss_fn,
            training_samples=current_train,
            image_map=image_map,
            steps=config.steps_per_round,
            resolution=config.resolution,
            device=config.device,
            sam3_output_cls=SAM3Output,
            round_index=round_index,
            log_every=_DEFAULT_LOG_EVERY,
            log_fn=log_fn,
            include_masks=resolved_loss_config.use_masks,
        )
        if train_history:
            _emit_log(
                log_fn,
                _format_round_loss_log(
                    round_index=round_index,
                    learning_rates=_optimizer_learning_rates(optimizer),
                    losses=train_history[-1],
                ),
            )
        adapter_path = round_dir / "adapter.pt"
        save_native_adapter(adapter_path, wrapper)

        predictions = predict_all_images(
            wrapper=wrapper,
            image_map=image_map,
            label=target_label,
            resolution=config.resolution,
            device=config.device,
            score_threshold=config.score_threshold,
        )
        predictions_path = round_dir / "predictions.json"
        errors_path = round_dir / "errors.json"
        next_train_path = round_dir / "next_train.json"
        save_predictions(predictions_path, predictions)
        target_ground_truth = _filter_ground_truth_by_label(full_ground_truth, target_label)
        errors = build_error_queue(
            ground_truths=target_ground_truth,
            predictions=predictions,
            iou_threshold=config.iou_threshold,
            localization_error_threshold=config.localization_error_threshold,
            iou_mode=config.iou_mode,
        )
        metrics = _compute_round_metrics(
            full_ground_truth=full_ground_truth,
            predictions=predictions,
            target_label=target_label,
            iou_threshold=config.iou_threshold,
            iou_mode=config.iou_mode,
        )
        _emit_log(
            log_fn,
            _format_eval_log(
                round_index=round_index,
                prediction_count=len(predictions),
                error_count=len(errors),
                metrics=metrics,
            ),
        )
        selected = select_next_training_sample(errors)
        if selected is not None:
            errors = [_mark_selected(error, selected) for error in errors]
        save_error_queue(errors_path, errors)
        if selected is not None:
            next_train = add_selected_training_sample(
                current_train,
                full_ground_truth,
                selected=selected,
                label=target_label,
            )
        else:
            next_train = list(current_train)
        _emit_log(
            log_fn,
            _format_selected_sample_log(
                selected=selected,
                next_train=next_train,
                label=target_label,
            ),
        )
        save_training_samples(next_train_path, next_train)
        visual_outputs = _render_round_visual_outputs(
            round_dir=round_dir,
            image_map=image_map,
            current_train=current_train,
            full_ground_truth=full_ground_truth,
            predictions=predictions,
            errors=errors,
        )

        summary = {
            "round": round_index,
            "train_count": _count_positive_annotations(current_train),
            "train_image_count": len(current_train),
            "negative_train_count": _count_negative_samples(current_train),
            "prediction_count": len(predictions),
            "error_count": len(errors),
            "metrics": metrics,
            "selected_image_id": None if selected is None else selected.image_id,
            "adapter": str(adapter_path),
            "predictions": str(predictions_path),
            "errors": str(errors_path),
            "next_train": str(next_train_path),
            **visual_outputs,
            "last_loss": train_history[-1] if train_history else None,
        }
        _write_json(round_dir / "summary.json", summary)
        round_summaries.append(summary)
        if not errors or _training_sample_signature(next_train) == _training_sample_signature(current_train):
            break
        current_train = next_train

    final_summary = {
        "mode": "native_efficientsam3_fewshot",
        "config": asdict(config),
        "adapter_config": asdict(adapter_config or NativeAdapterConfig()),
        "loss_config": asdict(resolved_loss_config),
        "rounds": round_summaries,
    }
    _write_json(output_root / "summary.json", final_summary)
    return final_summary


def train_native_adapter_one_round(
    *,
    wrapper: Any,
    optimizer: Any,
    loss_fn: Any,
    annotations: list[Annotation] | None = None,
    training_samples: list[TrainingSample] | None = None,
    image_map: dict[str, str],
    steps: int,
    resolution: int,
    device: str,
    sam3_output_cls: Any,
    round_index: int | None = None,
    log_every: int | None = None,
    log_fn: LogFn | None = None,
    include_masks: bool = False,
) -> list[dict[str, float]]:
    """训练当前 round 的 task prompt / adapter。"""
    if steps <= 0:
        return []
    torch = require_torch()
    if training_samples is None:
        if annotations is None:
            raise ValueError("annotations or training_samples must be provided")
        training_samples = annotations_to_training_samples(annotations)
    sample_items = list(enumerate(training_samples))
    if not sample_items:
        raise ValueError("current train set is empty")
    history: list[dict[str, float]] = []
    # 少样本训练集通常很小，按图片缓存 batch 可避免每个 step 重复 Image.open/resize/to(device)。
    batch_by_index = {
        sample_index: build_sam3_training_batch_from_samples(
            [sample],
            image_map,
            resolution=resolution,
            device=device,
            include_masks=include_masks,
        )
        for sample_index, sample in sample_items
    }
    wrapper.train()
    for step in range(steps):
        sample_index, _ = sample_items[step % len(sample_items)]
        batch = batch_by_index[sample_index]
        optimizer.zero_grad(set_to_none=True)
        out = wrapper.forward_batch(batch)
        sam3_output = sam3_output_cls([[out]])
        target_dict = wrapper.model.back_convert(batch.find_target)
        loss_dict = loss_fn(sam3_output, [target_dict])
        loss = loss_dict["core_loss"]
        if not torch.isfinite(loss).all():
            raise ValueError(f"non-finite training loss at step {step}: {float(loss.detach().cpu())}")
        loss.backward()
        optimizer.step()
        step_losses = {key: float(value.detach().cpu()) for key, value in loss_dict.items()}
        history.append(step_losses)
        if round_index is not None and _should_log_step(step, steps, log_every):
            _emit_log(
                log_fn,
                _format_loss_log(
                    round_index=round_index,
                    step=step,
                    steps=steps,
                    learning_rates=_optimizer_learning_rates(optimizer),
                    losses=step_losses,
                ),
            )
    return history


def predict_all_images(
    *,
    wrapper: Any,
    image_map: dict[str, str],
    label: str,
    resolution: int,
    device: str,
    score_threshold: float,
) -> list:
    """对全量图片执行 EfficientSAM3 原生推理。"""
    torch = require_torch()
    predictions = []
    wrapper.eval()
    image_ids = list(image_map)
    with torch.inference_mode():
        for image_id in image_ids:
            image_batch = load_image_batch(
                [image_id],
                image_map,
                resolution=resolution,
                device=device,
            )
            find_stage = _build_inference_find_stage(
                batch_size=1,
                torch=torch,
                device=device,
            )
            outputs = wrapper.forward_image_batch(
                images=image_batch.images,
                find_stage=find_stage,
                find_target=None,
            )
            predictions.extend(
                native_outputs_to_predictions(
                    outputs,
                    image_ids=image_batch.image_ids,
                    original_sizes=image_batch.original_sizes,
                    label=label,
                    score_threshold=score_threshold,
                )
            )
    return predictions


def add_selected_image_truth(
    train_set: list[Annotation],
    all_ground_truths: list[Annotation],
    *,
    selected_image_id: str,
    label: str,
) -> list[Annotation]:
    """兼容旧 annotation 训练集格式的真值追加函数。

    新的原生闭环请使用 `add_selected_training_sample`，它支持把纯背景误检
    转成 no-object 负样本；这里保留给旧测试或外部调用，不再承载新训练逻辑。
    """
    existing_ids = {annotation.object_id for annotation in train_set}
    next_train = list(train_set)
    for annotation in all_ground_truths:
        if annotation.image_id != selected_image_id or annotation.label != label:
            continue
        if annotation.object_id in existing_ids:
            continue
        next_train.append(annotation)
        existing_ids.add(annotation.object_id)
    return next_train


def add_selected_training_sample(
    train_samples: list[TrainingSample],
    all_ground_truths: list[Annotation],
    *,
    selected: Any,
    label: str,
) -> list[TrainingSample]:
    """把选中的错误样本加入图片级训练集。

    纯背景误检会生成 `negative` 样本；有同类真值的图片会生成或扩展正样本。
    """
    return add_selected_errors_to_training_samples(
        train_samples,
        all_ground_truths,
        [selected],
        label=label,
    )


def _compute_round_metrics(
    *,
    full_ground_truth: list[Annotation],
    predictions: list[Prediction],
    target_label: str,
    iou_threshold: float,
    iou_mode: str,
) -> dict[str, float | int]:
    """计算每轮 summary 中展示的检测指标。"""
    return compute_detection_metrics(
        ground_truths=full_ground_truth,
        predictions=predictions,
        label=target_label,
        iou_threshold=iou_threshold,
        iou_mode=iou_mode,
    ).to_dict()


def _render_round_visual_outputs(
    *,
    round_dir: str | Path,
    image_map: dict[str, str],
    current_train: list[TrainingSample] | list[Annotation],
    full_ground_truth: list[Annotation],
    predictions: list[Prediction],
    errors: list[Any],
) -> dict[str, str]:
    """渲染当前轮图片并返回可写入 summary 的目录路径。"""
    training_samples = (
        current_train
        if not current_train or isinstance(current_train[0], TrainingSample)
        else annotations_to_training_samples(current_train)
    )
    outputs = render_round_visualizations(
        round_dir=round_dir,
        image_map=image_map,
        train_annotations=[],
        training_samples=training_samples,
        full_ground_truth=full_ground_truth,
        predictions=predictions,
        errors=errors,
    )
    return outputs.to_summary_dict()


def _validate_mask_training_config(
    *,
    config: NativeFewShotLoopConfig,
    loss_config: NativeLossConfig,
) -> None:
    """在训练入口处校验 mask loss 所需的 SAM3 结构是否启用。"""
    if loss_config.use_masks and not config.enable_segmentation:
        raise ValueError(
            "LOSS.USE_MASKS=true requires MODEL.ENABLE_SEGMENTATION=true, "
            "because SAM3 only emits pred_masks when the segmentation head is built."
        )


def _format_trainable_module_logs(wrapper: Any) -> list[str]:
    """生成可微调模块清单日志。"""
    groups: dict[str, dict[str, Any]] = {}
    total_tensors = 0
    total_params = 0
    for name, parameter in wrapper.named_parameters():
        if not getattr(parameter, "requires_grad", False):
            continue
        module_name = _trainable_module_group(name)
        count = int(parameter.numel()) if hasattr(parameter, "numel") else 0
        group = groups.setdefault(module_name, {"tensors": 0, "params": 0, "names": []})
        group["tensors"] += 1
        group["params"] += count
        group["names"].append(name)
        total_tensors += 1
        total_params += count

    lines = [f"[fewshot] 本次可微调模块：{total_tensors} 个参数张量，共 {total_params} 个参数"]
    for module_name, group in groups.items():
        sample_names = ", ".join(group["names"][:3])
        if len(group["names"]) > 3:
            sample_names += ", ..."
        lines.append(
            f"[fewshot]   - {module_name}: {group['tensors']} 个张量，"
            f"{group['params']} 个参数 ({sample_names})"
        )
    if not groups:
        lines.append("[fewshot]   - 未发现 requires_grad=True 的参数，请检查冻结策略。")
    return lines


def _format_round_start_log(
    *,
    round_index: int,
    max_rounds: int,
    samples: Sequence[TrainingSample],
    learning_rates: Sequence[float],
) -> str:
    """生成每轮开始日志。"""
    return (
        f"[fewshot] 开始 round={round_index + 1}/{max_rounds} | "
        f"train_images={len(samples)} | "
        f"positive_targets={_count_positive_annotations(list(samples))} | "
        f"negative_images={_count_negative_samples(list(samples))} | "
        f"lr={_format_learning_rates(learning_rates)}"
    )


def _format_loss_log(
    *,
    round_index: int,
    step: int,
    steps: int,
    learning_rates: Sequence[float],
    losses: dict[str, float],
) -> str:
    """生成训练 step loss 日志。"""
    return (
        f"[fewshot] train round={round_index + 1} "
        f"step={step + 1}/{steps} | "
        f"lr={_format_learning_rates(learning_rates)} | "
        f"{_format_loss_items(losses)}"
    )


def _format_round_loss_log(
    *,
    round_index: int,
    learning_rates: Sequence[float],
    losses: dict[str, float],
) -> str:
    """生成每轮训练结束 loss 汇总日志。"""
    return (
        f"[fewshot] round={round_index + 1} 训练完成 | "
        f"lr={_format_learning_rates(learning_rates)} | "
        f"{_format_loss_items(losses)}"
    )


def _format_eval_log(
    *,
    round_index: int,
    prediction_count: int,
    error_count: int,
    metrics: dict[str, Any],
) -> str:
    """生成每轮推理评估日志。"""
    return (
        f"[fewshot] eval round={round_index + 1} | "
        f"pred={prediction_count} | err={error_count} | "
        f"P={_format_metric(metrics.get('precision', 0.0))} | "
        f"R={_format_metric(metrics.get('recall', 0.0))} | "
        f"F1={_format_metric(metrics.get('f1', 0.0))} | "
        f"mIoU={_format_metric(metrics.get('miou', 0.0))}"
    )


def _format_selected_sample_log(
    *,
    selected: Any | None,
    next_train: Sequence[TrainingSample],
    label: str,
) -> str:
    """生成下一轮自动选样日志。"""
    if selected is None:
        return "[fewshot] 下一轮选样：无新错误样本。"
    selected_samples = [
        sample
        for sample in next_train
        if sample.image_id == selected.image_id and sample.label == label
    ]
    if any(sample.sample_type == "negative" for sample in selected_samples):
        sample_kind = "no-object 负样本"
    elif selected_samples:
        sample_kind = "正样本"
    else:
        sample_kind = "未加入训练集"
    return (
        f"[fewshot] 下一轮选样：{sample_kind} | image={selected.image_id} | "
        f"type={selected.error_type} | risk={_format_metric(selected.risk_score)} | "
        f"reason={selected.reason}"
    )


def _count_positive_annotations(samples: list[TrainingSample]) -> int:
    return sum(len(sample.annotations) for sample in samples)


def _count_negative_samples(samples: list[TrainingSample]) -> int:
    return sum(1 for sample in samples if sample.sample_type == "negative")


def _trainable_module_group(parameter_name: str) -> str:
    normalized = parameter_name.removeprefix("module.")
    if normalized.startswith("task_prompt_tokens"):
        return "task_prompt_tokens"
    if normalized.startswith("prompt_adapter"):
        return "prompt_adapter"
    if ".dot_prod_scoring." in f".{normalized}":
        return "dot_prod_scoring"
    if ".bbox_embed" in normalized:
        return "transformer.decoder.bbox_embed"
    if ".cross_attn." in normalized or ".ca_text." in normalized:
        return "transformer.decoder.cross_attention"
    return normalized.rsplit(".", 1)[0]


def _format_loss_items(losses: dict[str, float]) -> str:
    ordered_keys = [key for key in _LOSS_LOG_KEYS if key in losses]
    ordered_keys.extend(sorted(key for key in losses if key not in set(ordered_keys)))
    return " | ".join(f"{key}={_format_loss_value(losses[key])}" for key in ordered_keys)


def _format_loss_value(value: Any) -> str:
    return f"{float(value):.4f}"


def _format_metric(value: Any) -> str:
    return f"{float(value):.4f}"


def _format_learning_rates(learning_rates: Sequence[float]) -> str:
    if not learning_rates:
        return "unknown"
    return ",".join(f"{float(lr):.6g}" for lr in learning_rates)


def _optimizer_learning_rates(optimizer: Any) -> list[float]:
    rates: list[float] = []
    for group in getattr(optimizer, "param_groups", []):
        if "lr" not in group:
            continue
        lr = float(group["lr"])
        if lr not in rates:
            rates.append(lr)
    return rates


def _should_log_step(step: int, steps: int, log_every: int | None) -> bool:
    interval = max(1, int(log_every or _DEFAULT_LOG_EVERY))
    return step == 0 or step + 1 == steps or (step + 1) % interval == 0


def _emit_logs(log_fn: LogFn | None, messages: Sequence[str]) -> None:
    for message in messages:
        _emit_log(log_fn, message)


def _emit_log(log_fn: LogFn | None, message: str) -> None:
    if log_fn is not None:
        log_fn(message)


def _training_sample_signature(samples: list[TrainingSample]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    positive_ids = tuple(
        sorted(
            annotation.object_id
            for sample in samples
            for annotation in sample.annotations
        )
    )
    negative_ids = tuple(
        sorted(
            f"{sample.label}:{sample.image_id}"
            for sample in samples
            if sample.sample_type == "negative"
        )
    )
    return positive_ids, negative_ids


def _filter_ground_truth_by_label(
    ground_truths: list[Annotation],
    label: str,
) -> list[Annotation]:
    """闭环当前只训练一个目标类，错误队列也应只评估该类。"""
    return [ground_truth for ground_truth in ground_truths if ground_truth.label == label]


def _build_inference_find_stage(*, batch_size: int, torch: Any, device: str) -> Any:
    from sam3.model.data_misc import FindStage

    return FindStage(
        img_ids=torch.arange(batch_size, device=device, dtype=torch.long),
        text_ids=torch.zeros(batch_size, device=device, dtype=torch.long),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )


def _resolve_label(label: str | None, annotations: list[Annotation]) -> str:
    if label:
        return label
    if not annotations:
        raise ValueError("full ground truth is empty; cannot infer label")
    labels = sorted({annotation.label for annotation in annotations})
    if len(labels) > 1:
        raise ValueError(
            "multiple labels found in full ground truth; pass --label explicitly. "
            f"available labels: {', '.join(labels)}"
        )
    return labels[0]


def _mark_selected(error: Any, selected: Any) -> Any:
    same_error = (
        error.image_id == selected.image_id
        and error.error_type == selected.error_type
        and error.ground_truth_ids == selected.ground_truth_ids
        and error.prediction_ids == selected.prediction_ids
    )
    return selected if same_error else error


def _read_json_dict(path: str | Path) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return {str(key): str(value) for key, value in payload.items()}


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
