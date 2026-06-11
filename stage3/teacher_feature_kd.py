import logging
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from sam3.model_builder import build_sam3_image_model
from sam3.train.loss.loss_fns import CORE_LOSS_KEY


LOGGER = logging.getLogger(__name__)


def _as_feature_list(features) -> list[torch.Tensor]:
    if features is None:
        return []
    if torch.is_tensor(features):
        return [features]
    if isinstance(features, (list, tuple)):
        return [feature for feature in features if torch.is_tensor(feature)]
    return []


class TeacherFeatureDistillation(nn.Module):
    """Distill frozen SAM3 image features while keeping supervised Stage 3 losses."""

    def __init__(
        self,
        teacher_checkpoint_path: str,
        feature_key: str = "backbone_fpn",
        weight: float = 0.05,
        loss_type: str = "cosine",
        levels: Iterable[int] | None = None,
        detach_student: bool = False,
        teacher_bpe_path: str | None = None,
    ):
        super().__init__()
        if feature_key not in {"backbone_fpn", "trunk_features"}:
            raise ValueError(f"Unsupported feature_key={feature_key}")
        if loss_type not in {"cosine", "normalized_l2"}:
            raise ValueError(f"Unsupported loss_type={loss_type}")

        self.teacher_checkpoint_path = teacher_checkpoint_path
        self.feature_key = feature_key
        self.weight = float(weight)
        self.loss_type = loss_type
        self.levels = list(levels) if levels is not None else None
        self.detach_student = detach_student
        self.teacher_bpe_path = teacher_bpe_path
        self._teacher = None

    @staticmethod
    def _zero_like_output(outputs) -> torch.Tensor:
        for value in outputs.values():
            if torch.is_tensor(value):
                return value.sum() * 0.0
            if isinstance(value, dict):
                try:
                    return TeacherFeatureDistillation._zero_like_output(value)
                except ValueError:
                    pass
        raise ValueError("TeacherFeatureDistillation could not find a tensor output")

    def _get_teacher(self, device: torch.device):
        if self._teacher is None:
            LOGGER.info(
                "Building frozen SAM3 teacher for %s KD from %s",
                self.feature_key,
                self.teacher_checkpoint_path,
            )
            teacher = build_sam3_image_model(
                bpe_path=self.teacher_bpe_path,
                checkpoint_path=self.teacher_checkpoint_path,
                enable_segmentation=True,
                device=str(device),
                eval_mode=True,
            )
            teacher.requires_grad_(False)
            object.__setattr__(self, "_teacher", teacher)
        return self._teacher

    def _select_levels(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.levels is None:
            return features
        return [features[index] for index in self.levels]

    def _feature_loss(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        if student.shape[-2:] != teacher.shape[-2:]:
            teacher = F.interpolate(
                teacher,
                size=student.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if student.shape[1] != teacher.shape[1]:
            raise ValueError(
                f"KD channel mismatch for {self.feature_key}: "
                f"student={tuple(student.shape)} teacher={tuple(teacher.shape)}"
            )

        student = student.float()
        teacher = teacher.float()
        if self.detach_student:
            student = student.detach()

        if self.loss_type == "cosine":
            student = F.normalize(student.flatten(2), dim=1)
            teacher = F.normalize(teacher.flatten(2), dim=1)
            return (1.0 - (student * teacher).sum(dim=1)).mean()

        student = F.normalize(student, dim=1)
        teacher = F.normalize(teacher, dim=1)
        return F.mse_loss(student, teacher)

    def forward(self, outputs, targets, indices, num_boxes, is_aux=False):
        if is_aux or self.weight == 0:
            zero = self._zero_like_output(outputs)
            return {CORE_LOSS_KEY: zero, f"loss_teacher_{self.feature_key}": zero}

        backbone_out = outputs.get("prev_encoder_out", {}).get("backbone_out", {})
        student_features = self._select_levels(_as_feature_list(backbone_out.get(self.feature_key)))
        if not student_features:
            zero = self._zero_like_output(outputs)
            return {CORE_LOSS_KEY: zero, f"loss_teacher_{self.feature_key}": zero}

        images = backbone_out["img_batch_all_stages"]
        teacher = self._get_teacher(images.device)
        with torch.no_grad():
            teacher_out = teacher.backbone.forward_image(images)
        teacher_features = self._select_levels(_as_feature_list(teacher_out.get(self.feature_key)))

        if len(student_features) != len(teacher_features):
            count = min(len(student_features), len(teacher_features))
            student_features = student_features[-count:]
            teacher_features = teacher_features[-count:]

        losses = [
            self._feature_loss(student, teacher)
            for student, teacher in zip(student_features, teacher_features)
        ]
        loss = torch.stack(losses).mean()
        weighted = loss * self.weight
        return {
            CORE_LOSS_KEY: weighted,
            f"loss_teacher_{self.feature_key}": weighted.detach(),
        }
