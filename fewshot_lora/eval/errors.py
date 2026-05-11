"""构造交互式修正的错误队列。

每一轮评估后，系统需要从失败样本里挑一张让“用户修正”。离线实验中我们
用 GT 模拟用户，因此这里输出的是下一轮应该加入训练集的图片优先级。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .metrics import ImageEval


class ErrorType(str, Enum):
    """错误类型，顺序越靠前越应该优先修正。"""

    FALSE_NEGATIVE = "false_negative"
    LOCALIZATION_ERROR = "localization_error"
    FALSE_POSITIVE = "false_positive"
    LOW_CONFIDENCE_TRUE_POSITIVE = "low_confidence_true_positive"


@dataclass(frozen=True)
class ErrorCandidate:
    """错误队列中的一个候选图片。"""

    image_id: str
    error_type: ErrorType
    count: int
    priority: int


_PRIORITY = {
    # 漏检通常最影响 recall，因此优先级最高。
    ErrorType.FALSE_NEGATIVE: 0,
    # 定位错误说明模型看到了目标，但 box/mask 还不准。
    ErrorType.LOCALIZATION_ERROR: 1,
    # 误检影响 precision，排在漏检和定位错误之后。
    ErrorType.FALSE_POSITIVE: 2,
    ErrorType.LOW_CONFIDENCE_TRUE_POSITIVE: 3,
}


def build_error_queue(evals: list[ImageEval]) -> list[ErrorCandidate]:
    """根据每张图的评估结果生成有序错误队列。"""

    candidates: list[ErrorCandidate] = []
    for image_eval in evals:
        if image_eval.false_negative_count:
            candidates.append(_candidate(image_eval, ErrorType.FALSE_NEGATIVE, image_eval.false_negative_count))
        elif image_eval.localization_error_count:
            candidates.append(_candidate(image_eval, ErrorType.LOCALIZATION_ERROR, image_eval.localization_error_count))
        elif image_eval.false_positive_count:
            candidates.append(_candidate(image_eval, ErrorType.FALSE_POSITIVE, image_eval.false_positive_count))
    candidates.sort(key=lambda item: (item.priority, item.image_id))
    return candidates


def _candidate(image_eval: ImageEval, error_type: ErrorType, count: int) -> ErrorCandidate:
    """创建一个错误候选对象。"""

    return ErrorCandidate(
        image_id=image_eval.image_id,
        error_type=error_type,
        count=count,
        priority=_PRIORITY[error_type],
    )
