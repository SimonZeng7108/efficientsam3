"""训练集初始化与增量更新逻辑。

验证阶段每轮自动选择少量错误样本加入训练。本模块只处理“选哪些真值”，
不直接碰模型、优化器或图像张量。
"""

from __future__ import annotations

import random
from collections import defaultdict

from ..evaluation.matching import ErrorItem
from .models import Annotation


class InitialTrainSelector:
    """第 0 轮少样本训练集选择器。"""

    def select(
        self,
        annotations: list[Annotation],
        *,
        label: str | None = None,
        seed: int | None = None,
    ) -> list[Annotation]:
        return create_initial_train_set(annotations, label=label, seed=seed)


class TrainSetUpdater:
    """根据错误队列更新下一轮训练集。"""

    def add_selected_errors(
        self,
        train_set: list[Annotation],
        all_ground_truths: list[Annotation],
        errors: list[ErrorItem],
    ) -> list[Annotation]:
        return add_selected_errors_to_train_set(train_set, all_ground_truths, errors)


def create_initial_train_set(
    annotations: list[Annotation],
    *,
    label: str | None = None,
    seed: int | None = None,
) -> list[Annotation]:
    """随机选择一张含目标图片，并返回该图片上的目标标注。"""
    grouped: dict[str, list[Annotation]] = defaultdict(list)
    for annotation in annotations:
        if label is not None and annotation.label != label:
            continue
        grouped[annotation.image_id].append(annotation)

    if not grouped:
        raise ValueError("no annotations match the requested label")

    # 先排序再随机，保证相同 seed 在不同机器上选择一致。
    image_ids = sorted(grouped)
    selected_image_id = random.Random(seed).choice(image_ids)
    return grouped[selected_image_id]


def add_selected_errors_to_train_set(
    train_set: list[Annotation],
    all_ground_truths: list[Annotation],
    errors: list[ErrorItem],
) -> list[Annotation]:
    existing_ids = {annotation.object_id for annotation in train_set}
    gt_by_id = {annotation.object_id: annotation for annotation in all_ground_truths}
    next_train_set = list(train_set)

    for error in errors:
        if not error.selected_for_next_round:
            continue
        for ground_truth_id in error.ground_truth_ids:
            if ground_truth_id in existing_ids:
                continue
            ground_truth = gt_by_id.get(ground_truth_id)
            if ground_truth is None:
                continue
            next_train_set.append(ground_truth)
            existing_ids.add(ground_truth_id)

    return next_train_set
