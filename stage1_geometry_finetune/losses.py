"""
Stage 2 Loss Functions for Geometry Fine-tuning.

Implements:
- Embedding loss (MSE with valid masking for padding)
- Mask BCE loss (Binary Cross-Entropy)
- Mask Dice loss
- Mask Focal loss (optional)
- IoU prediction loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Focal loss for imbalanced classification.
    
    Args:
        inputs: Predictions (logits)
        targets: Ground truth (probabilities or binary)
        valid_mask: Optional mask for valid regions
        alpha: Weighting factor
        gamma: Focusing parameter
        reduction: 'mean', 'sum', or 'none'
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    
    if valid_mask is not None:
        loss = loss * valid_mask
        if reduction == "mean":
            return loss.sum() / (valid_mask.sum() + 1e-8)
        elif reduction == "sum":
            return loss.sum()
    else:
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
    
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    target_is_logit: bool = False,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Binary Cross-Entropy loss.
    
    Args:
        inputs: Predictions (logits)
        targets: Ground truth (logits if target_is_logit else probabilities)
        valid_mask: Optional mask for valid regions
        target_is_logit: Whether targets are logits (will be sigmoidified)
        reduction: 'mean', 'sum', or 'none'
    """
    if target_is_logit:
        targets = torch.sigmoid(targets)
    
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    
    if valid_mask is not None:
        loss = loss * valid_mask
        if reduction == "mean":
            return loss.sum() / (valid_mask.sum() + 1e-8)
        elif reduction == "sum":
            return loss.sum()
    else:
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
    
    return loss


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    target_is_logit: bool = False,
    smooth: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Dice loss for mask prediction.
    
    Args:
        inputs: Predictions (logits)
        targets: Ground truth (logits if target_is_logit else probabilities)
        valid_mask: Optional mask for valid regions
        target_is_logit: Whether targets are logits (will be sigmoidified)
        smooth: Smoothing factor to avoid division by zero
        reduction: 'mean', 'sum', or 'none'
    """
    inputs = torch.sigmoid(inputs)
    if target_is_logit:
        targets = torch.sigmoid(targets)
    
    if valid_mask is not None:
        inputs = inputs * valid_mask
        targets = targets * valid_mask
    
    # Flatten spatial dimensions
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    
    intersection = (inputs * targets).sum(dim=1)
    union = inputs.sum(dim=1) + targets.sum(dim=1)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    loss = 1.0 - dice
    
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    
    return loss


def masked_mse_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    MSE loss with optional masking for valid regions (handles padding).
    
    Args:
        inputs: Predictions
        targets: Ground truth
        valid_mask: Optional mask for valid regions
        reduction: 'mean', 'sum', or 'none'
    """
    loss = F.mse_loss(inputs, targets, reduction="none")
    
    if valid_mask is not None:
        loss = loss * valid_mask
        if reduction == "mean":
            return loss.sum() / (valid_mask.sum() + 1e-8)
        elif reduction == "sum":
            return loss.sum()
    else:
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
    
    return loss


def masked_cosine_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Cosine similarity loss (1 - cosine_similarity).
    
    Args:
        inputs: Predictions (B, C, H, W)
        targets: Ground truth (B, C, H, W)
        valid_mask: Optional mask for valid regions (B, 1, H, W)
        reduction: 'mean', 'sum', or 'none'
    """
    # Normalize along channel dimension
    inputs_norm = F.normalize(inputs, dim=1)
    targets_norm = F.normalize(targets, dim=1)
    
    # Cosine similarity per spatial location
    cos_sim = (inputs_norm * targets_norm).sum(dim=1, keepdim=True)  # (B, 1, H, W)
    loss = 1.0 - cos_sim
    
    if valid_mask is not None:
        loss = loss * valid_mask
        if reduction == "mean":
            return loss.sum() / (valid_mask.sum() + 1e-8)
        elif reduction == "sum":
            return loss.sum()
    else:
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
    
    return loss


class GeometryFinetuningLoss(nn.Module):
    """
    Combined loss for Stage 2 Geometry Fine-tuning.
    
    Loss = α × embedding_loss + β × (bce_loss + dice_loss) + γ × iou_loss
    """
    
    def __init__(
        self,
        embedding_weight: float = 1.0,
        mask_bce_weight: float = 1.0,
        mask_dice_weight: float = 1.0,
        mask_focal_weight: float = 0.0,
        iou_weight: float = 0.5,
        temperature: float = 1.0,
        use_cosine_embedding: bool = False,
    ):
        super().__init__()
        self.embedding_weight = embedding_weight
        self.mask_bce_weight = mask_bce_weight
        self.mask_dice_weight = mask_dice_weight
        self.mask_focal_weight = mask_focal_weight
        self.iou_weight = iou_weight
        self.temperature = temperature
        self.use_cosine_embedding = use_cosine_embedding
    
    def forward(
        self,
        student_embedding: torch.Tensor,
        teacher_embedding: torch.Tensor,
        student_masks: Optional[torch.Tensor] = None,
        teacher_masks: Optional[torch.Tensor] = None,
        student_iou: Optional[torch.Tensor] = None,
        teacher_iou: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        mask_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            student_embedding: Student trunk embeddings (B, C, H, W)
            teacher_embedding: Teacher trunk embeddings (B, C, H, W)
            student_masks: Student mask predictions (B, N, H, W)
            teacher_masks: Teacher mask predictions (B, N, H, W)
            student_iou: Student IoU predictions (B, N)
            teacher_iou: Teacher IoU predictions (B, N)
            valid_mask: Valid mask for embeddings (handles padding)
            mask_valid_mask: Valid mask for mask predictions
            
        Returns:
            Dict with 'total_loss' and individual loss components
        """
        losses = {}
        total_loss = 0.0
        
        # Embedding loss
        if self.embedding_weight > 0:
            if self.use_cosine_embedding:
                embed_loss = masked_cosine_loss(
                    student_embedding, teacher_embedding, valid_mask
                )
                losses['embed_cosine'] = embed_loss
            else:
                embed_loss = masked_mse_loss(
                    student_embedding, teacher_embedding, valid_mask
                )
                losses['embed_mse'] = embed_loss
            total_loss = total_loss + self.embedding_weight * embed_loss
        
        # Mask losses
        if student_masks is not None and teacher_masks is not None:
            # Apply temperature scaling
            student_masks_scaled = student_masks / self.temperature
            teacher_masks_scaled = teacher_masks / self.temperature
            
            # BCE loss
            if self.mask_bce_weight > 0:
                bce_loss = sigmoid_ce_loss(
                    student_masks_scaled,
                    teacher_masks_scaled,
                    mask_valid_mask,
                    target_is_logit=True,
                )
                losses['mask_bce'] = bce_loss
                total_loss = total_loss + self.mask_bce_weight * bce_loss
            
            # Dice loss
            if self.mask_dice_weight > 0:
                dice = dice_loss(
                    student_masks_scaled,
                    teacher_masks_scaled,
                    mask_valid_mask,
                    target_is_logit=True,
                )
                losses['mask_dice'] = dice
                total_loss = total_loss + self.mask_dice_weight * dice
            
            # Focal loss
            if self.mask_focal_weight > 0:
                focal = sigmoid_focal_loss(
                    student_masks_scaled,
                    torch.sigmoid(teacher_masks_scaled),
                    mask_valid_mask,
                )
                losses['mask_focal'] = focal
                total_loss = total_loss + self.mask_focal_weight * focal
        
        # IoU prediction loss
        if student_iou is not None and teacher_iou is not None and self.iou_weight > 0:
            iou_loss = F.mse_loss(student_iou, teacher_iou)
            losses['iou_mse'] = iou_loss
            total_loss = total_loss + self.iou_weight * iou_loss
        
        losses['total_loss'] = total_loss
        return losses


def create_valid_mask(
    batch_size: int,
    embed_size: int,
    img_sizes: torch.Tensor,
    img_size: int = 1024,
    device: torch.device = "cuda",
) -> torch.Tensor:
    """
    Create valid mask for embeddings to handle padding.
    
    Args:
        batch_size: Batch size
        embed_size: Spatial size of embeddings (e.g., 64)
        img_sizes: Original image sizes before padding (B, 2) or list of tuples
        img_size: Padded image size (e.g., 1024)
        device: Device for tensor
        
    Returns:
        Valid mask (B, 1, embed_size, embed_size)
    """
    # Build mask in padded image space (img_size x img_size) then downsample.
    # This matches the Stage 1 approach and avoids rounding artifacts.
    valid = torch.zeros(batch_size, 1, img_size, img_size, device=device)

    for i in range(batch_size):
        h, w = img_sizes[i] if isinstance(img_sizes, torch.Tensor) else img_sizes[i]
        h_i = int(h)
        w_i = int(w)
        valid[i, :, :h_i, :w_i] = 1.0

    valid = F.interpolate(valid, size=(embed_size, embed_size), mode="bilinear", align_corners=False)
    return (valid > 0.5).float()
