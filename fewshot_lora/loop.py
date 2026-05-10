"""少样本交互闭环调度。

这个文件不直接依赖 torch，也不直接加载模型。训练和评估通过回调注入，
这样可以用 fake trainer/evaluator 单测闭环策略，同时在服务器上替换成真实
EfficientSAM3 LoRA 训练与评估。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .config import FewShotLoRAConfig
from .errors import build_error_queue
from .metrics import ImageEval


@dataclass(frozen=True)
class TrainRoundOutput:
    """单轮训练输出统计。"""

    adapter_path: Path
    train_image_count: int
    train_instance_count: int
    train_seconds: float
    train_steps: int


TrainRoundFn = Callable[[Sequence[str], int, Path], Path | TrainRoundOutput]
EvaluateRoundFn = Callable[[int, Path], list[ImageEval]]
InitialSelector = Callable[[Sequence[str]], str]


@dataclass(frozen=True)
class RoundSummary:
    """一轮交互的完整摘要，会写入 summary.json。"""

    round_index: int
    train_image_ids: tuple[str, ...]
    adapter_path: Path
    train_image_count: int
    train_instance_count: int
    train_seconds: float
    train_steps: int
    precision: float
    recall: float
    f1: float
    mean_obb_iou: float
    false_positive_count: int
    false_negative_count: int
    localization_error_count: int
    next_image_id: str | None


@dataclass(frozen=True)
class DatasetRunSummary:
    """单个子数据集的闭环运行结果。"""

    dataset_name: str
    success: bool
    rounds: tuple[RoundSummary, ...]


def run_dataset_loop(
    dataset_name: str,
    image_ids: Sequence[str],
    config: FewShotLoRAConfig,
    train_round: TrainRoundFn,
    evaluate_round: EvaluateRoundFn,
    initial_selector: InitialSelector = lambda ids: ids[0],
) -> DatasetRunSummary:
    """运行一个子数据集的交互式训练-评估-选样闭环。

    流程是：
    1. 选择第 0 轮初始正样本图。
    2. 用累计训练图训练/继续训练 LoRA。
    3. 对全量图片评估。
    4. 成功则停止；失败则从错误队列选择下一张图加入训练集。
    """

    if not image_ids:
        raise ValueError("image_ids 不能为空")

    train_image_ids: list[str] = [initial_selector(image_ids)]
    round_summaries: list[RoundSummary] = []
    success = False

    for round_index in range(config.max_rounds):
        adapter_path = config.output_dir / dataset_name / f"round_{round_index:02d}" / "adapter.pt"
        train_output = _normalize_train_output(
            train_round(tuple(train_image_ids), round_index, adapter_path),
            fallback_train_image_ids=train_image_ids,
        )
        adapter_path = train_output.adapter_path
        evals = evaluate_round(round_index, adapter_path)
        aggregate = _aggregate(evals)
        error_queue = build_error_queue(evals)
        next_image_id = _next_unseen_error_image(error_queue, train_image_ids)

        round_summaries.append(
            RoundSummary(
                round_index=round_index,
                train_image_ids=tuple(train_image_ids),
                adapter_path=adapter_path,
                train_image_count=train_output.train_image_count,
                train_instance_count=train_output.train_instance_count,
                train_seconds=train_output.train_seconds,
                train_steps=train_output.train_steps,
                precision=aggregate.precision,
                recall=aggregate.recall,
                f1=aggregate.f1,
                mean_obb_iou=aggregate.mean_obb_iou,
                false_positive_count=aggregate.false_positive_count,
                false_negative_count=aggregate.false_negative_count,
                localization_error_count=aggregate.localization_error_count,
                next_image_id=next_image_id,
            )
        )

        if _is_success(evals):
            success = True
            break
        if next_image_id is None:
            break
        train_image_ids.append(next_image_id)

    return DatasetRunSummary(
        dataset_name=dataset_name,
        success=success,
        rounds=tuple(round_summaries),
    )


def _normalize_train_output(
    output: Path | TrainRoundOutput,
    fallback_train_image_ids: Sequence[str],
) -> TrainRoundOutput:
    """兼容只返回 adapter path 的轻量回调。"""

    if isinstance(output, TrainRoundOutput):
        return output
    # 单元测试或外部轻量调用可只返回 adapter path；真实 runner 会返回完整训练统计。
    return TrainRoundOutput(
        adapter_path=output,
        train_image_count=len(fallback_train_image_ids),
        train_instance_count=0,
        train_seconds=0.0,
        train_steps=0,
    )


def _is_success(evals: list[ImageEval]) -> bool:
    """判断当前子数据集是否已达到 100% precision 和 recall。"""

    return all(
        image_eval.precision == 1.0
        and image_eval.recall == 1.0
        and image_eval.false_positive_count == 0
        and image_eval.false_negative_count == 0
        and image_eval.localization_error_count == 0
        for image_eval in evals
    )


def _next_unseen_error_image(error_queue, train_image_ids: list[str]) -> str | None:
    """优先选择尚未进入训练集的失败图。

    如果所有失败图都已经训练过，仍返回队首图片，便于继续强化最难样本。
    """

    seen = set(train_image_ids)
    for item in error_queue:
        if item.image_id not in seen:
            return item.image_id
    return error_queue[0].image_id if error_queue else None


def _aggregate(evals: list[ImageEval]) -> RoundSummary:
    """把所有图片级指标汇总成轮级指标。"""

    tp = sum(item.true_positive_count for item in evals)
    fp = sum(item.false_positive_count for item in evals)
    fn = sum(item.false_negative_count for item in evals)
    loc = sum(item.localization_error_count for item in evals)
    matched_ious = [match.iou for item in evals for match in item.matches]
    precision_den = tp + fp + loc
    recall_den = tp + fn + loc
    precision = 1.0 if precision_den == 0 else tp / precision_den
    recall = 1.0 if recall_den == 0 else tp / recall_den
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    mean_iou = 0.0 if not matched_ious else sum(matched_ious) / len(matched_ious)
    return _Aggregate(
        precision=precision,
        recall=recall,
        f1=f1,
        mean_obb_iou=mean_iou,
        false_positive_count=fp,
        false_negative_count=fn,
        localization_error_count=loc,
    )


@dataclass(frozen=True)
class _Aggregate:
    """内部轮级聚合指标，避免复用 RoundSummary 时填入无关字段。"""

    precision: float
    recall: float
    f1: float
    mean_obb_iou: float
    false_positive_count: int
    false_negative_count: int
    localization_error_count: int
