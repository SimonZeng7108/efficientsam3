"""单类别多实例 OBB 匹配与指标计算。

本任务没有多类别分类问题：每个子数据集只有一个目标类别。因此评估只关心：
- GT 是否都有预测匹配。
- 是否有多余预测。
- 预测与 GT 的 OBB IoU 是否达标。
"""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import OrientedBox, rotated_iou


@dataclass(frozen=True)
class ImageGroundTruth:
    """一张图片的 GT OBB 列表。"""

    image_id: str
    boxes: list[OrientedBox]


@dataclass(frozen=True)
class PredictionInstance:
    """一个预测实例，包含 OBB 和模型分数。"""

    obb: OrientedBox
    score: float


@dataclass(frozen=True)
class ImagePrediction:
    """一张图片的预测结果。"""

    image_id: str
    instances: list[PredictionInstance]


@dataclass(frozen=True)
class MatchPair:
    """GT 和预测之间的一对匹配关系。"""

    gt_index: int
    pred_index: int
    iou: float


@dataclass(frozen=True)
class ImageEval:
    """一张图片的评估结果和派生指标。"""

    image_id: str
    matches: tuple[MatchPair, ...]
    localization_errors: tuple[MatchPair, ...]
    false_positive_indices: tuple[int, ...]
    false_negative_indices: tuple[int, ...]

    @property
    def true_positive_count(self) -> int:
        return len(self.matches)

    @property
    def false_positive_count(self) -> int:
        return len(self.false_positive_indices)

    @property
    def false_negative_count(self) -> int:
        return len(self.false_negative_indices)

    @property
    def localization_error_count(self) -> int:
        return len(self.localization_errors)

    @property
    def precision(self) -> float:
        """precision = TP / (TP + FP + 定位错误)。"""

        denominator = self.true_positive_count + self.false_positive_count + self.localization_error_count
        return 1.0 if denominator == 0 else self.true_positive_count / denominator

    @property
    def recall(self) -> float:
        """recall = TP / (TP + FN + 定位错误)。"""

        denominator = self.true_positive_count + self.false_negative_count + self.localization_error_count
        return 1.0 if denominator == 0 else self.true_positive_count / denominator

    @property
    def f1(self) -> float:
        precision = self.precision
        recall = self.recall
        return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    @property
    def mean_obb_iou(self) -> float:
        if not self.matches:
            return 0.0
        return sum(match.iou for match in self.matches) / len(self.matches)


def evaluate_image(
    ground_truth: ImageGroundTruth,
    prediction: ImagePrediction,
    iou_threshold: float,
    localization_iou_threshold: float = 0.1,
) -> ImageEval:
    """对单张图片做贪心 OBB IoU 匹配。

    匹配分两轮：
    1. IoU 达到 `iou_threshold` 的算 true positive。
    2. 没达成功阈值但超过 `localization_iou_threshold` 的算定位错误。

    这样可以区分“完全漏检”和“看到了但框不准”，便于下一轮选样。
    """

    if ground_truth.image_id != prediction.image_id:
        raise ValueError("GT 和预测的 image_id 必须一致")

    candidates: list[tuple[float, int, int]] = []
    for gt_idx, gt_box in enumerate(ground_truth.boxes):
        for pred_idx, pred in enumerate(prediction.instances):
            candidates.append((rotated_iou(gt_box, pred.obb), gt_idx, pred_idx))
    candidates.sort(reverse=True, key=lambda item: item[0])

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[MatchPair] = []
    localization_errors: list[MatchPair] = []

    # 第一轮：只接受达到正式成功阈值的匹配。
    for iou, gt_idx, pred_idx in candidates:
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        if iou >= iou_threshold:
            matches.append(MatchPair(gt_idx, pred_idx, iou))
            used_gt.add(gt_idx)
            used_pred.add(pred_idx)

    # 第二轮：剩下的候选中，若仍有明显重叠，则记为定位错误而不是 FP+FN。
    for iou, gt_idx, pred_idx in candidates:
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        if iou >= localization_iou_threshold:
            localization_errors.append(MatchPair(gt_idx, pred_idx, iou))
            used_gt.add(gt_idx)
            used_pred.add(pred_idx)

    false_negatives = tuple(idx for idx in range(len(ground_truth.boxes)) if idx not in used_gt)
    false_positives = tuple(idx for idx in range(len(prediction.instances)) if idx not in used_pred)
    return ImageEval(
        image_id=ground_truth.image_id,
        matches=tuple(matches),
        localization_errors=tuple(localization_errors),
        false_positive_indices=false_positives,
        false_negative_indices=false_negatives,
    )
