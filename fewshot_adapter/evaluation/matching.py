"""预测结果与真值的匹配、错误判定和下一轮样本选择。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from ..data.models import HBB, Annotation, Prediction, normalize_annotation, polygon_to_hbb

ErrorType = Literal[
    "false_negative",
    "false_positive",
    "localization_error",
    "low_confidence_true_positive",
]
IouMode = Literal["hbb", "obb", "polygon"]


@dataclass(frozen=True)
class Match:
    ground_truth: Annotation
    prediction: Prediction
    iou: float


@dataclass(frozen=True)
class ErrorItem:
    image_id: str
    error_type: ErrorType
    risk_score: float
    reason: str
    ground_truth_ids: list[str]
    prediction_ids: list[str]
    selected_for_next_round: bool = False


class DetectionMatcher:
    """真值与预测的匹配/错误判定门面。"""

    def match(
        self,
        ground_truths: list[Annotation],
        predictions: list[Prediction],
        *,
        iou_threshold: float = 0.5,
        iou_mode: IouMode = "hbb",
    ) -> list[Match]:
        return greedy_match_predictions(
            ground_truths,
            predictions,
            iou_threshold=iou_threshold,
            iou_mode=iou_mode,
        )

    def build_error_queue(
        self,
        ground_truths: list[Annotation],
        predictions: list[Prediction],
        *,
        iou_threshold: float = 0.5,
        localization_error_threshold: float = 0.1,
        iou_mode: IouMode = "hbb",
    ) -> list[ErrorItem]:
        return build_error_queue(
            ground_truths,
            predictions,
            iou_threshold=iou_threshold,
            localization_error_threshold=localization_error_threshold,
            iou_mode=iou_mode,
        )


class ErrorSelector:
    """从错误队列中选择下一轮训练样本。"""

    def select(self, errors: list[ErrorItem]) -> ErrorItem | None:
        return select_next_training_sample(errors)


def box_iou(left: HBB, right: HBB) -> float:
    inter_x1 = max(left.x1, right.x1)
    inter_y1 = max(left.y1, right.y1)
    inter_x2 = min(left.x2, right.x2)
    inter_y2 = min(left.y2, right.y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    left_area = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
    right_area = max(0.0, right.x2 - right.x1) * max(0.0, right.y2 - right.y1)
    union = left_area + right_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _prediction_hbb(prediction: Prediction) -> HBB | None:
    if prediction.hbb is not None:
        return prediction.hbb
    if prediction.polygon is not None:
        return polygon_to_hbb(prediction.polygon)
    return None


def greedy_match_predictions(
    ground_truths: list[Annotation],
    predictions: list[Prediction],
    iou_threshold: float = 0.5,
    iou_mode: IouMode = "hbb",
) -> list[Match]:
    candidates: list[tuple[float, int, int]] = []
    normalized_gts = [normalize_annotation(gt) for gt in ground_truths]
    predictions_by_key = _group_predictions_by_image_label(predictions)
    for gt_index, gt in enumerate(normalized_gts):
        if gt.hbb is None:
            continue
        for pred_index, prediction in predictions_by_key.get((gt.image_id, gt.label), []):
            iou = _instance_iou(gt, prediction, iou_mode)
            if iou >= iou_threshold:
                candidates.append((iou, gt_index, pred_index))

    matches: list[Match] = []
    used_gts: set[int] = set()
    used_predictions: set[int] = set()
    for iou, gt_index, pred_index in sorted(candidates, reverse=True):
        if gt_index in used_gts or pred_index in used_predictions:
            continue
        used_gts.add(gt_index)
        used_predictions.add(pred_index)
        matches.append(Match(normalized_gts[gt_index], predictions[pred_index], iou))
    return matches


def build_error_queue(
    ground_truths: list[Annotation],
    predictions: list[Prediction],
    iou_threshold: float = 0.5,
    localization_error_threshold: float = 0.1,
    iou_mode: IouMode = "hbb",
) -> list[ErrorItem]:
    """根据真值和预测生成错误队列。"""
    normalized_gts = [normalize_annotation(gt) for gt in ground_truths]
    predictions_by_key = _group_predictions_by_image_label(predictions)
    matches = greedy_match_predictions(normalized_gts, predictions, iou_threshold, iou_mode)
    matched_gt_ids = {match.ground_truth.object_id for match in matches}
    matched_prediction_ids = {match.prediction.prediction_id for match in matches}

    errors: list[ErrorItem] = []
    for gt in normalized_gts:
        if gt.object_id in matched_gt_ids:
            continue
        best_prediction, best_iou = _best_unmatched_prediction_for_gt(
            gt,
            predictions_by_key.get((gt.image_id, gt.label), []),
            matched_prediction_ids,
            iou_mode,
        )
        if best_prediction is not None and best_iou >= localization_error_threshold:
            errors.append(
                ErrorItem(
                    image_id=gt.image_id,
                    error_type="localization_error",
                    risk_score=1.0 - best_iou,
                    reason="prediction overlaps ground truth but is below the match IoU threshold",
                    ground_truth_ids=[gt.object_id],
                    prediction_ids=[best_prediction.prediction_id],
                )
            )
            matched_prediction_ids.add(best_prediction.prediction_id)
        else:
            errors.append(
                ErrorItem(
                    image_id=gt.image_id,
                    error_type="false_negative",
                    risk_score=1.0,
                    reason="ground truth object has no matching prediction",
                    ground_truth_ids=[gt.object_id],
                    prediction_ids=[],
                )
            )

    for prediction in predictions:
        if prediction.prediction_id in matched_prediction_ids:
            continue
        errors.append(
            ErrorItem(
                image_id=prediction.image_id,
                error_type="false_positive",
                risk_score=prediction.score,
                reason="prediction has no matching ground truth",
                ground_truth_ids=[],
                prediction_ids=[prediction.prediction_id],
            )
        )

    return sorted(errors, key=_error_sort_key)


def _best_unmatched_prediction_for_gt(
    gt: Annotation,
    predictions: list[tuple[int, Prediction]],
    matched_prediction_ids: set[str],
    iou_mode: IouMode,
) -> tuple[Prediction | None, float]:
    best_prediction = None
    best_iou = 0.0
    for _, prediction in predictions:
        if prediction.prediction_id in matched_prediction_ids:
            continue
        iou = _instance_iou(gt, prediction, iou_mode)
        if iou > best_iou:
            best_iou = iou
            best_prediction = prediction
    return best_prediction, best_iou


def _group_predictions_by_image_label(
    predictions: list[Prediction],
) -> dict[tuple[str, str], list[tuple[int, Prediction]]]:
    grouped: dict[tuple[str, str], list[tuple[int, Prediction]]] = {}
    for pred_index, prediction in enumerate(predictions):
        grouped.setdefault((prediction.image_id, prediction.label), []).append((pred_index, prediction))
    return grouped


def _instance_iou(gt: Annotation, prediction: Prediction, iou_mode: IouMode) -> float:
    if iou_mode == "obb" and gt.obb is not None and prediction.obb is not None:
        from ..geometry.ops import obb_iou

        return obb_iou(gt.obb, prediction.obb)
    if iou_mode == "polygon" and gt.polygon is not None and prediction.polygon is not None:
        from ..geometry.ops import polygon_iou

        return polygon_iou(gt.polygon, prediction.polygon)
    if gt.hbb is None:
        return 0.0
    pred_hbb = _prediction_hbb(prediction)
    if pred_hbb is None:
        return 0.0
    return box_iou(gt.hbb, pred_hbb)


def select_next_training_sample(errors: list[ErrorItem]) -> ErrorItem | None:
    """按优先级选择下一轮加入训练的一条错误样本。"""
    if not errors:
        return None
    selected = min(errors, key=_error_sort_key)
    return replace(selected, selected_for_next_round=True)


def _error_sort_key(error: ErrorItem) -> tuple[int, float, str]:
    priority = {
        "false_negative": 0,
        "localization_error": 1,
        "false_positive": 2,
        "low_confidence_true_positive": 3,
    }[error.error_type]
    return (priority, -error.risk_score, error.image_id)
