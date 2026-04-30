"""每轮少样本闭环的图片可视化输出。

本模块只负责调试图片落盘，不参与训练和评估计算。这样 GPU 验证时可以直接
打开 `round_xx/` 目录查看本轮输入、错误样本和全量检测结果。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from PIL import Image, ImageDraw

from ..data.models import HBB, Annotation, Prediction, normalize_annotation, polygon_to_hbb
from ..evaluation.matching import ErrorItem

_GT_COLOR = (0, 180, 60)
_PRED_COLOR = (220, 40, 35)
_TEXT_COLOR = (255, 210, 0)
_TEXT_BG = (0, 0, 0)
_LINE_WIDTH = 3


@dataclass(frozen=True)
class RoundVisualizationOutputs:
    """一轮训练产生的三个可视化目录。"""

    train_inputs_dir: Path
    errors_dir: Path
    predictions_dir: Path

    def to_summary_dict(self) -> dict[str, str]:
        """写入 round summary，方便其他智能体直接定位图片输出。"""
        return {
            "train_inputs": str(self.train_inputs_dir),
            "errors_vis": str(self.errors_dir),
            "predictions_vis": str(self.predictions_dir),
        }


def render_round_visualizations(
    *,
    round_dir: str | Path,
    image_map: Mapping[str, str],
    train_annotations: list[Annotation],
    full_ground_truth: list[Annotation],
    predictions: list[Prediction],
    errors: list[ErrorItem],
) -> RoundVisualizationOutputs:
    """输出当前轮的训练输入图、错误复查图和全量检测结果图。"""
    root = Path(round_dir)
    outputs = RoundVisualizationOutputs(
        train_inputs_dir=root / "train_inputs",
        errors_dir=root / "errors_vis",
        predictions_dir=root / "predictions_vis",
    )
    _ensure_dirs(outputs)

    train_by_image = _group_annotations(train_annotations)
    full_gt_by_id = {annotation.object_id: normalize_annotation(annotation) for annotation in full_ground_truth}
    full_gt_by_image = _group_annotations(full_ground_truth)
    predictions_by_id = {prediction.prediction_id: prediction for prediction in predictions}
    predictions_by_image = _group_predictions(predictions)

    _render_train_inputs(outputs.train_inputs_dir, image_map, train_by_image)
    _render_prediction_results(outputs.predictions_dir, image_map, predictions_by_image)
    _render_error_results(
        outputs.errors_dir,
        image_map,
        errors,
        full_gt_by_id,
        full_gt_by_image,
        predictions_by_id,
        predictions_by_image,
    )
    return outputs


def _ensure_dirs(outputs: RoundVisualizationOutputs) -> None:
    for directory in (outputs.train_inputs_dir, outputs.errors_dir, outputs.predictions_dir):
        directory.mkdir(parents=True, exist_ok=True)


def _render_train_inputs(
    output_dir: Path,
    image_map: Mapping[str, str],
    train_by_image: Mapping[str, list[Annotation]],
) -> None:
    for image_id, annotations in train_by_image.items():
        image = _open_rgb(image_map, image_id)
        draw = ImageDraw.Draw(image)
        for annotation in annotations:
            _draw_annotation(draw, normalize_annotation(annotation), prefix="GT")
        _save_jpeg(image, output_dir / _output_name(image_id, "gt"))


def _render_prediction_results(
    output_dir: Path,
    image_map: Mapping[str, str],
    predictions_by_image: Mapping[str, list[Prediction]],
) -> None:
    for image_id in image_map:
        image = _open_rgb(image_map, image_id)
        draw = ImageDraw.Draw(image)
        for prediction in predictions_by_image.get(image_id, []):
            _draw_prediction(draw, prediction)
        _save_jpeg(image, output_dir / _output_name(image_id, "pred"))


def _render_error_results(
    output_dir: Path,
    image_map: Mapping[str, str],
    errors: list[ErrorItem],
    full_gt_by_id: Mapping[str, Annotation],
    full_gt_by_image: Mapping[str, list[Annotation]],
    predictions_by_id: Mapping[str, Prediction],
    predictions_by_image: Mapping[str, list[Prediction]],
) -> None:
    errors_by_image = _group_errors(errors)
    for image_id, image_errors in errors_by_image.items():
        image = _open_rgb(image_map, image_id)
        draw = ImageDraw.Draw(image)
        related_gt_ids = {
            ground_truth_id
            for error in image_errors
            for ground_truth_id in error.ground_truth_ids
        }
        related_prediction_ids = {
            prediction_id
            for error in image_errors
            for prediction_id in error.prediction_ids
        }

        # 如果错误项没有具体 GT/预测 ID，就画该图所有同类信息，方便人工复查上下文。
        annotations = [
            full_gt_by_id[ground_truth_id]
            for ground_truth_id in related_gt_ids
            if ground_truth_id in full_gt_by_id
        ] or full_gt_by_image.get(image_id, [])
        predictions = [
            predictions_by_id[prediction_id]
            for prediction_id in related_prediction_ids
            if prediction_id in predictions_by_id
        ] or predictions_by_image.get(image_id, [])

        for annotation in annotations:
            _draw_annotation(draw, normalize_annotation(annotation), prefix="GT")
        for prediction in predictions:
            _draw_prediction(draw, prediction)
        _draw_error_labels(draw, image_errors)
        _save_jpeg(image, output_dir / _output_name(image_id, "error"))


def _draw_annotation(draw: ImageDraw.ImageDraw, annotation: Annotation, *, prefix: str) -> None:
    label = f"{prefix} {annotation.label}"
    if annotation.polygon:
        points = [(float(x), float(y)) for x, y in annotation.polygon]
        draw.line(points + [points[0]], fill=_GT_COLOR, width=_LINE_WIDTH)
        _draw_text(draw, points[0], label)
        return
    if annotation.hbb is not None:
        _draw_hbb(draw, annotation.hbb, _GT_COLOR)
        _draw_text(draw, (annotation.hbb.x1, annotation.hbb.y1), label)


def _draw_prediction(draw: ImageDraw.ImageDraw, prediction: Prediction) -> None:
    hbb = _prediction_hbb(prediction)
    if hbb is None:
        return
    _draw_hbb(draw, hbb, _PRED_COLOR)
    _draw_text(
        draw,
        (hbb.x1, hbb.y1),
        f"PRED {prediction.label} {prediction.score:.2f}",
    )


def _draw_error_labels(draw: ImageDraw.ImageDraw, errors: list[ErrorItem]) -> None:
    for index, error in enumerate(errors):
        marker = "*" if error.selected_for_next_round else "-"
        _draw_text(
            draw,
            (4, 4 + index * 14),
            f"{marker} {error.error_type} risk={error.risk_score:.2f}",
        )


def _draw_hbb(draw: ImageDraw.ImageDraw, hbb: HBB, color: tuple[int, int, int]) -> None:
    draw.rectangle(
        [float(hbb.x1), float(hbb.y1), float(hbb.x2), float(hbb.y2)],
        outline=color,
        width=_LINE_WIDTH,
    )


def _draw_text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str) -> None:
    x, y = float(xy[0]), max(0.0, float(xy[1]) - 12)
    text_box = draw.textbbox((x, y), text)
    draw.rectangle(text_box, fill=_TEXT_BG)
    draw.text((x, y), text, fill=_TEXT_COLOR)


def _prediction_hbb(prediction: Prediction) -> HBB | None:
    if prediction.hbb is not None:
        return prediction.hbb
    if prediction.polygon is not None:
        return polygon_to_hbb(prediction.polygon)
    return None


def _open_rgb(image_map: Mapping[str, str], image_id: str) -> Image.Image:
    if image_id not in image_map:
        raise KeyError(f"image_id not found in image_map: {image_id}")
    return Image.open(image_map[image_id]).convert("RGB")


def _save_jpeg(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=92)


def _output_name(image_id: str, suffix: str) -> str:
    safe_name = "".join(char if char.isalnum() or char in "._-" else "_" for char in image_id)
    return f"{safe_name}_{suffix}.jpg"


def _group_annotations(annotations: Iterable[Annotation]) -> dict[str, list[Annotation]]:
    grouped: dict[str, list[Annotation]] = {}
    for annotation in annotations:
        grouped.setdefault(annotation.image_id, []).append(annotation)
    return grouped


def _group_predictions(predictions: Iterable[Prediction]) -> dict[str, list[Prediction]]:
    grouped: dict[str, list[Prediction]] = {}
    for prediction in predictions:
        grouped.setdefault(prediction.image_id, []).append(prediction)
    return grouped


def _group_errors(errors: Iterable[ErrorItem]) -> dict[str, list[ErrorItem]]:
    grouped: dict[str, list[ErrorItem]] = {}
    for error in errors:
        grouped.setdefault(error.image_id, []).append(error)
    return grouped
