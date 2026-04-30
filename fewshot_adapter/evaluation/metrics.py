"""检测结果指标计算。

本模块只负责把现有匹配结果汇总为便于观察的数字指标。训练闭环依然使用
`matching.py` 里的同一套 IoU 匹配逻辑，避免“错误队列”和“评估指标”口径不一致。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..data.models import Annotation, Prediction
from .matching import IouMode, greedy_match_predictions


@dataclass(frozen=True)
class DetectionMetrics:
    """单类别检测评估结果。

    `miou` 是所有 TP 匹配 IoU 的平均值；没有 TP 时记为 0，便于 JSON 汇总。
    """

    ground_truth_count: int
    prediction_count: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    miou: float

    def to_dict(self) -> dict[str, float | int]:
        """转换成 summary.json 使用的稳定字段名。"""
        return {
            "ground_truth_count": self.ground_truth_count,
            "prediction_count": self.prediction_count,
            "tp": self.true_positive,
            "fp": self.false_positive,
            "fn": self.false_negative,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "miou": self.miou,
        }


def compute_detection_metrics(
    *,
    ground_truths: list[Annotation],
    predictions: list[Prediction],
    label: str | None = None,
    iou_threshold: float = 0.5,
    iou_mode: IouMode = "hbb",
) -> DetectionMetrics:
    """计算 precision、recall、F1 和 mIoU。

    如果传入 `label`，会先过滤真值和预测，保证少样本单类别验证时只评估当前目标类。
    """
    target_ground_truths = _filter_annotations_by_label(ground_truths, label)
    target_predictions = _filter_predictions_by_label(predictions, label)
    matches = greedy_match_predictions(
        target_ground_truths,
        target_predictions,
        iou_threshold=iou_threshold,
        iou_mode=iou_mode,
    )

    true_positive = len(matches)
    false_positive = len(target_predictions) - true_positive
    false_negative = len(target_ground_truths) - true_positive
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    miou = _safe_divide(sum(match.iou for match in matches), true_positive)

    return DetectionMetrics(
        ground_truth_count=len(target_ground_truths),
        prediction_count=len(target_predictions),
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
        miou=miou,
    )


def _filter_annotations_by_label(annotations: list[Annotation], label: str | None) -> list[Annotation]:
    if label is None:
        return list(annotations)
    return [annotation for annotation in annotations if annotation.label == label]


def _filter_predictions_by_label(predictions: list[Prediction], label: str | None) -> list[Prediction]:
    if label is None:
        return list(predictions)
    return [prediction for prediction in predictions if prediction.label == label]


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
