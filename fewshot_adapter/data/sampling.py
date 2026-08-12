"""训练集初始化与增量更新逻辑。

验证阶段每轮自动选择少量错误样本加入训练。本模块只处理“选哪些真值”，
不直接碰模型、优化器或图像张量。
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import replace

from ..evaluation.matching import ErrorItem
from .models import Annotation, TrainingSample


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

    def select_samples(
        self,
        annotations: list[Annotation],
        *,
        label: str | None = None,
        seed: int | None = None,
    ) -> list[TrainingSample]:
        return create_initial_training_samples(annotations, label=label, seed=seed)


class TrainSetUpdater:
    """根据错误队列更新下一轮训练集。"""

    def add_selected_errors(
        self,
        train_set: list[Annotation],
        all_ground_truths: list[Annotation],
        errors: list[ErrorItem],
    ) -> list[Annotation]:
        return add_selected_errors_to_train_set(train_set, all_ground_truths, errors)

    def add_selected_errors_to_samples(
        self,
        train_samples: list[TrainingSample],
        all_ground_truths: list[Annotation],
        errors: list[ErrorItem],
        *,
        label: str,
    ) -> list[TrainingSample]:
        return add_selected_errors_to_training_samples(
            train_samples,
            all_ground_truths,
            errors,
            label=label,
        )


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


def create_initial_training_samples(
    annotations: list[Annotation],
    *,
    label: str | None = None,
    seed: int | None = None,
) -> list[TrainingSample]:
    """随机选择一张含目标图片，并返回图片级正样本。"""
    selected_annotations = create_initial_train_set(annotations, label=label, seed=seed)
    return annotations_to_training_samples(selected_annotations)


def annotations_to_training_samples(
    annotations: list[Annotation],
) -> list[TrainingSample]:
    """把目标级标注按图片聚合为正样本。"""
    grouped: dict[tuple[str, str], list[Annotation]] = defaultdict(list)
    for annotation in annotations:
        grouped[(annotation.image_id, annotation.label)].append(annotation)
    return [
        TrainingSample(image_id=image_id, label=label, annotations=list(grouped_annotations))
        for (image_id, label), grouped_annotations in grouped.items()
    ]


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


def add_selected_errors_to_training_samples(
    train_samples: list[TrainingSample],
    all_ground_truths: list[Annotation],
    errors: list[ErrorItem],
    *,
    label: str,
) -> list[TrainingSample]:
    """根据选中错误更新图片级训练样本，支持 no-object 负样本。"""
    next_samples = list(train_samples)
    gt_by_id = {annotation.object_id: annotation for annotation in all_ground_truths}
    gt_by_image: dict[str, list[Annotation]] = defaultdict(list)
    for annotation in all_ground_truths:
        if annotation.label == label:
            gt_by_image[annotation.image_id].append(annotation)

    for error in errors:
        if not error.selected_for_next_round:
            continue
        selected_truths = [
            gt_by_id[ground_truth_id]
            for ground_truth_id in error.ground_truth_ids
            if ground_truth_id in gt_by_id and gt_by_id[ground_truth_id].label == label
        ]
        if selected_truths:
            next_samples = _add_positive_annotations(next_samples, selected_truths, label=label)
            continue

        image_truths = gt_by_image.get(error.image_id, [])
        if image_truths:
            next_samples = _add_positive_annotations(next_samples, image_truths, label=label)
            continue

        if error.error_type == "false_positive":
            next_samples = _add_negative_sample(
                next_samples,
                image_id=error.image_id,
                label=label,
                reason=f"{error.error_type}: {error.reason}",
            )

    return next_samples


def _add_positive_annotations(
    samples: list[TrainingSample],
    annotations: list[Annotation],
    *,
    label: str,
) -> list[TrainingSample]:
    existing_ids = {
        annotation.object_id
        for sample in samples
        for annotation in sample.annotations
    }
    new_by_image: dict[str, list[Annotation]] = defaultdict(list)
    for annotation in annotations:
        if annotation.object_id not in existing_ids and annotation.label == label:
            new_by_image[annotation.image_id].append(annotation)
            existing_ids.add(annotation.object_id)
    if not new_by_image:
        return samples

    updated = list(samples)
    for image_id, new_annotations in new_by_image.items():
        matched_index = _find_positive_sample_index(updated, image_id=image_id, label=label)
        if matched_index is None:
            updated.append(
                TrainingSample(
                    image_id=image_id,
                    label=label,
                    annotations=list(new_annotations),
                )
            )
            continue
        sample = updated[matched_index]
        updated[matched_index] = replace(
            sample,
            annotations=[*sample.annotations, *new_annotations],
        )
    return updated


def _add_negative_sample(
    samples: list[TrainingSample],
    *,
    image_id: str,
    label: str,
    reason: str,
) -> list[TrainingSample]:
    for sample in samples:
        if sample.image_id == image_id and sample.label == label:
            return samples
    return [
        *samples,
        TrainingSample(
            image_id=image_id,
            label=label,
            annotations=[],
            sample_type="negative",
            reason=reason,
        ),
    ]


def _find_positive_sample_index(
    samples: list[TrainingSample],
    *,
    image_id: str,
    label: str,
) -> int | None:
    for index, sample in enumerate(samples):
        if sample.image_id == image_id and sample.label == label and sample.sample_type == "positive":
            return index
    return None
