"""Stage 3 Loss Functions.

Single-prompt supervision: each training sample is
(image, 1 caption, 1 box, K points, 1 GT mask). The decoder produces Q mask
hypotheses; we pick the best-IoU query per sample for mask supervision and
teach ``pred_logits`` to concentrate its score on that query slot so that
inference-time ``argmax`` selection is well-calibrated.

Losses::

    L = lambda_emb   * MSE(student_embed, teacher_embed)         # distillation
      + lambda_bce   * BCE(sigma(best_mask), gt_mask)            # pixel level
      + lambda_dice  * Dice(sigma(best_mask), gt_mask)           # region level
      + lambda_cls   * BCE(pred_logits, one_hot(best_query))     # score head

The best-IoU query is selected in a ``torch.no_grad()`` block so that the
selection is treated as a target, not a decision variable.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from stage1_geometry_finetune.losses import (
    dice_loss,
    masked_mse_loss,
    sigmoid_ce_loss,
    sigmoid_focal_loss,
)


def _compute_best_query(
    pred_masks: torch.Tensor,
    gt_mask: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Return (B,) index of the query whose mask has highest IoU with the GT."""
    pred_bin = (pred_masks > threshold).float()
    gt = gt_mask.float()  # (B, 1, Hm, Wm)
    if valid_mask is not None:
        mv = valid_mask.float()
        pred_bin = pred_bin * mv
        gt = gt * mv
    intersection = (pred_bin * gt).sum(dim=(-2, -1))  # (B, Q)
    union = pred_bin.sum(dim=(-2, -1)) + gt.sum(dim=(-2, -1)) - intersection
    iou = intersection / (union + 1e-6)
    return iou.argmax(dim=-1)


class Stage3HybridLoss(nn.Module):
    """Hybrid loss for Stage 3 single-prompt supervised fine-tuning."""

    def __init__(
        self,
        embedding_weight: float = 0.0015,
        mask_bce_weight: float = 1.0,
        mask_dice_weight: float = 1.0,
        mask_focal_weight: float = 0.0,
        classification_weight: float = 0.1,
    ):
        super().__init__()
        self.embedding_weight = embedding_weight
        self.mask_bce_weight = mask_bce_weight
        self.mask_dice_weight = mask_dice_weight
        self.mask_focal_weight = mask_focal_weight
        self.classification_weight = classification_weight

    def forward(
        self,
        student_embedding: Optional[torch.Tensor] = None,
        teacher_embedding: Optional[torch.Tensor] = None,
        embed_valid_mask: Optional[torch.Tensor] = None,
        embed_sample_mask: Optional[torch.Tensor] = None,
        pred_masks: Optional[torch.Tensor] = None,
        pred_logits: Optional[torch.Tensor] = None,
        gt_mask: Optional[torch.Tensor] = None,
        mask_valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """Compute hybrid GT-supervised + embedding-distillation loss.

        Args:
            student_embedding: (B, C, He, We) student trunk output (fp32 or amp).
            teacher_embedding: (B, C, He, We) pre-computed teacher trunk output.
            embed_valid_mask:  (B, 1, He, We) valid-pixel mask for distillation.
            embed_sample_mask: (B,) bool; True if teacher_embedding is real.
            pred_masks:        (B, Q, Hm, Wm) mask LOGITS from the decoder.
            pred_logits:       (B, Q, 1) or (B, Q) score logits for each query.
            gt_mask:           (B, 1, Hm, Wm) binary GT mask (0/1).
            mask_valid_mask:   (B, 1, Hm, Wm) valid-pixel mask for mask loss.

        Returns:
            (losses_dict, best_idx) where ``best_idx`` is (B,) long tensor of the
            selected query per sample, or None if no masks were supplied.
        """
        device = _pick_device(student_embedding, pred_masks, gt_mask)
        losses: Dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=device)

        if (
            self.embedding_weight > 0
            and student_embedding is not None
            and teacher_embedding is not None
        ):
            if embed_sample_mask is not None and not embed_sample_mask.any():
                emb_loss = torch.zeros((), device=device)
            else:
                if embed_sample_mask is not None:
                    se = student_embedding[embed_sample_mask]
                    te = teacher_embedding[embed_sample_mask]
                    vm = (
                        embed_valid_mask[embed_sample_mask]
                        if embed_valid_mask is not None
                        else None
                    )
                else:
                    se = student_embedding
                    te = teacher_embedding
                    vm = embed_valid_mask
                emb_loss = masked_mse_loss(se, te.to(se.dtype), vm)
            losses["embed_mse"] = emb_loss
            total = total + self.embedding_weight * emb_loss

        best_idx: Optional[torch.Tensor] = None
        if pred_masks is not None and gt_mask is not None:
            B, Q = pred_masks.shape[:2]

            with torch.no_grad():
                gt_expand = gt_mask.expand(-1, Q, -1, -1)
                gt_expand = gt_expand.contiguous()
                mv_expand = (
                    mask_valid_mask.expand(-1, Q, -1, -1).contiguous()
                    if mask_valid_mask is not None
                    else None
                )
                best_idx = _compute_best_query(
                    pred_masks.detach(), gt_expand, mv_expand
                )

            batch_idx = torch.arange(B, device=device)
            best_mask = pred_masks[batch_idx, best_idx].unsqueeze(1)  # (B, 1, Hm, Wm)
            gt_for_loss = gt_mask.float()
            mv = mask_valid_mask

            if self.mask_bce_weight > 0:
                bce = sigmoid_ce_loss(
                    best_mask, gt_for_loss, valid_mask=mv, target_is_logit=False
                )
                losses["mask_bce"] = bce
                total = total + self.mask_bce_weight * bce

            if self.mask_dice_weight > 0:
                d_loss = dice_loss(
                    best_mask, gt_for_loss, valid_mask=mv, target_is_logit=False
                )
                losses["mask_dice"] = d_loss
                total = total + self.mask_dice_weight * d_loss

            if self.mask_focal_weight > 0:
                focal = sigmoid_focal_loss(best_mask, gt_for_loss, valid_mask=mv)
                losses["mask_focal"] = focal
                total = total + self.mask_focal_weight * focal

            if self.classification_weight > 0 and pred_logits is not None:
                logits = pred_logits
                if logits.ndim == 3 and logits.shape[-1] == 1:
                    logits = logits.squeeze(-1)
                target = torch.zeros_like(logits)
                target[batch_idx, best_idx] = 1.0
                cls_loss = F.binary_cross_entropy_with_logits(
                    logits.float(), target, reduction="mean"
                )
                losses["cls_bce"] = cls_loss
                total = total + self.classification_weight * cls_loss

        losses["total_loss"] = total
        return losses, best_idx


def _pick_device(*tensors) -> torch.device:
    for t in tensors:
        if isinstance(t, torch.Tensor):
            return t.device
    return torch.device("cpu")


def compute_miou(
    pred_mask_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    threshold: float = 0.0,
) -> float:
    """Compute mean IoU for a batch of (B, 1, H, W) best-query masks vs GT."""
    pred = (pred_mask_logits > threshold).float()
    gt = gt_mask.float()
    if valid_mask is not None:
        mv = valid_mask.float()
        pred = pred * mv
        gt = gt * mv

    intersection = (pred * gt).sum(dim=(-2, -1))
    union = pred.sum(dim=(-2, -1)) + gt.sum(dim=(-2, -1)) - intersection
    iou = intersection / (union + 1e-6)

    valid = gt.sum(dim=(-2, -1)) > 0
    if valid.any():
        return iou[valid].mean().item()
    return 0.0
