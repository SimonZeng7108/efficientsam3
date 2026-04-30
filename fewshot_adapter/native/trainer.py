"""EfficientSAM3 原生少样本自动闭环训练。

验证阶段没有交互界面，因此这里会自动选择错误样本并把真值加入下一轮。
训练本身使用 SAM3 原生输出和原生 loss，不再依赖 proposal_candidates.json。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..data.json_io import load_annotations, save_annotations, save_error_queue, save_predictions
from ..data.models import Annotation, Prediction
from ..data.sam3_batch import build_sam3_training_batch, group_annotations_by_image, load_image_batch
from ..data.sampling import add_selected_errors_to_train_set, create_initial_train_set
from ..evaluation.matching import build_error_queue, select_next_training_sample
from ..evaluation.metrics import compute_detection_metrics
from ..utils.torch import require_torch
from .adapter import NativeAdapterConfig, build_native_fewshot_model, save_native_adapter
from .loss import NativeLossConfig, build_native_loss
from .predictor import native_outputs_to_predictions


@dataclass(frozen=True)
class NativeFewShotLoopConfig:
    """原生少样本闭环配置。"""

    checkpoint: str
    output_root: str
    label: str | None = None
    device: str = "cuda"
    resolution: int = 1008
    seed: int = 0
    max_rounds: int = 10
    steps_per_round: int = 80
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    score_threshold: float = 0.5
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
) -> dict[str, Any]:
    """执行完整 EfficientSAM3 原生少样本闭环。"""
    torch = require_torch()
    from sam3.model.model_misc import SAM3Output

    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    full_ground_truth = load_annotations(full_ground_truth_path)
    image_map = _read_json_dict(image_map_path)
    target_label = _resolve_label(config.label, full_ground_truth)

    current_train = create_initial_train_set(
        full_ground_truth,
        label=target_label,
        seed=config.seed,
    )
    save_annotations(output_root / "train_round_0.json", current_train)

    wrapper = build_native_fewshot_model(
        checkpoint_path=config.checkpoint,
        config=adapter_config or NativeAdapterConfig(),
        device=config.device,
        resolution=config.resolution,
    )
    loss_fn = build_native_loss(loss_config)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in wrapper.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    round_summaries = []
    for round_index in range(config.max_rounds):
        round_dir = output_root / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        train_history = train_native_adapter_one_round(
            wrapper=wrapper,
            optimizer=optimizer,
            loss_fn=loss_fn,
            annotations=current_train,
            image_map=image_map,
            steps=config.steps_per_round,
            resolution=config.resolution,
            device=config.device,
            sam3_output_cls=SAM3Output,
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
        selected = select_next_training_sample(errors)
        if selected is not None:
            errors = [_mark_selected(error, selected) for error in errors]
        save_error_queue(errors_path, errors)
        next_train = add_selected_errors_to_train_set(
            current_train,
            full_ground_truth,
            errors,
        )
        if selected is not None:
            next_train = add_selected_image_truth(
                next_train,
                full_ground_truth,
                selected_image_id=selected.image_id,
                label=target_label,
            )
        save_annotations(next_train_path, next_train)

        summary = {
            "round": round_index,
            "train_count": len(current_train),
            "prediction_count": len(predictions),
            "error_count": len(errors),
            "metrics": metrics,
            "selected_image_id": None if selected is None else selected.image_id,
            "adapter": str(adapter_path),
            "predictions": str(predictions_path),
            "errors": str(errors_path),
            "next_train": str(next_train_path),
            "last_loss": train_history[-1] if train_history else None,
        }
        _write_json(round_dir / "summary.json", summary)
        round_summaries.append(summary)
        if not errors or len(next_train) == len(current_train):
            break
        current_train = next_train

    final_summary = {
        "mode": "native_efficientsam3_fewshot",
        "config": asdict(config),
        "adapter_config": asdict(adapter_config or NativeAdapterConfig()),
        "loss_config": asdict(loss_config or NativeLossConfig()),
        "rounds": round_summaries,
    }
    _write_json(output_root / "summary.json", final_summary)
    return final_summary


def train_native_adapter_one_round(
    *,
    wrapper: Any,
    optimizer: Any,
    loss_fn: Any,
    annotations: list[Annotation],
    image_map: dict[str, str],
    steps: int,
    resolution: int,
    device: str,
    sam3_output_cls: Any,
) -> list[dict[str, float]]:
    """训练当前 round 的 task prompt / adapter。"""
    if steps <= 0:
        return []
    grouped_items = list(group_annotations_by_image(annotations).items())
    if not grouped_items:
        raise ValueError("current train set is empty")
    history: list[dict[str, float]] = []
    # 少样本训练集通常很小，按图片缓存 batch 可避免每个 step 重复 Image.open/resize/to(device)。
    batch_by_image_id = {
        image_id: build_sam3_training_batch(
            image_annotations,
            image_map,
            resolution=resolution,
            device=device,
        )
        for image_id, image_annotations in grouped_items
    }
    wrapper.train()
    for step in range(steps):
        image_id, _ = grouped_items[step % len(grouped_items)]
        batch = batch_by_image_id[image_id]
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
        history.append({key: float(value.detach().cpu()) for key, value in loss_dict.items()})
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
    """把被选中错误图片上的同类真值加入训练集。

    漏检/定位错误通常已经通过 `ground_truth_ids` 加入；误检图片没有
    ground_truth_ids 时，这个兜底逻辑可以把该图已有真值也喂给下一轮。
    如果图片确实没有目标，则当前版本不会构造 no-object 样本。
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
