"""
Stage 2 Model: Geometry Fine-tuning Model for EfficientSAM3

This module provides the model architecture for Stage 2 fine-tuning.

ARCHITECTURE OVERVIEW:
======================
The key insight is that SAM3's architecture has prompt-dependent processing:
- GeometryEncoder pools features FROM the backbone output to encode box/point prompts
- This uses grid_sample (points) and roi_align (boxes) on FPN features
- Therefore prompt embeddings differ based on backbone quality

TRAINING STRATEGY:
==================
We use DUAL-PATH distillation:

1. EMBEDDING LOSS (Loss 1):
   - Student trunk → student_embedding
   - Teacher trunk → teacher_embedding (loaded from saved files)
   - MSE(student_embedding, teacher_embedding)

2. MASK LOSS (Loss 2):
   - student_embedding → Frozen SAM3 (FPN, GeomEnc, Transformer, SegHead) → student_mask
   - teacher_embedding → Frozen SAM3 (same frozen components) → teacher_mask  
   - BCE + Dice(student_mask, teacher_mask)

Both embeddings go through the SAME frozen SAM3 components, so the only
difference in mask outputs is due to the embedding quality difference.

This is the correct approach for prompt-conditioned fine-tuning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
from dataclasses import dataclass

# Import student backbones
from sam3.backbones.repvit import repvit_m0_9, repvit_m1_1, repvit_m2_3
from sam3.backbones.tiny_vit import tiny_vit_5m_224, tiny_vit_11m_224, tiny_vit_21m_224
from sam3.backbones.efficientvit import (
    efficientvit_backbone_b0,
    efficientvit_backbone_b1,
    efficientvit_backbone_b2,
)


def _make_divisible(v, divisor, min_value=None):
    """Make value divisible by divisor (from MobileNetV2)."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class RepViTAdapter(nn.Module):
    """Adapter for RepViT backbone to extract final features."""
    def __init__(self, model, out_channels):
        super().__init__()
        self.model = model
        self.out_channels = out_channels

    def forward(self, x):
        for layer in self.model.features:
            x = layer(x)
        return x


class TinyViTAdapter(nn.Module):
    """Adapter for TinyViT backbone."""
    def __init__(self, model, img_size):
        super().__init__()
        self.model = model
        self.model.head = nn.Identity()
        self.final_hw = self._compute_resolution(img_size)
        self.out_channels = self.model.norm_head.normalized_shape[0]
        self.model.norm_head = nn.Identity()

    def forward(self, x):
        x = self.model.patch_embed(x)
        x = self.model.layers[0](x)
        for i in range(1, len(self.model.layers)):
            x = self.model.layers[i](x)
        B, N, C = x.shape
        H, W = self.final_hw
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x

    def _compute_resolution(self, img_size):
        H, W = self.model.patches_resolution
        for _ in range(self.model.num_layers - 1):
            H = (H - 1) // 2 + 1
            W = (W - 1) // 2 + 1
        return (H, W)


class EfficientViTAdapter(nn.Module):
    """Adapter for EfficientViT backbone."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.out_channels = self.model.width_list[-1]

    def forward(self, x):
        out = self.model(x)
        return out["stage_final"]


def build_student_backbone(backbone_name: str, img_size: int = 1024):
    """Build a student backbone and return (backbone_adapter, out_channels)."""
    backbone_name = backbone_name.lower()
    
    if backbone_name.startswith('repvit'):
        fn = {
            'repvit_m0_9': repvit_m0_9,
            'repvit_m1_1': repvit_m1_1,
            'repvit_m2_3': repvit_m2_3,
        }[backbone_name]
        model = fn(pretrained=False, num_classes=0, distillation=False)
        out_channels = _make_divisible(model.cfgs[-1][2], 8)
        return RepViTAdapter(model, out_channels), out_channels
    
    if backbone_name.startswith('tiny_vit'):
        fn = {
            'tiny_vit_5m': tiny_vit_5m_224,
            'tiny_vit_11m': tiny_vit_11m_224,
            'tiny_vit_21m': tiny_vit_21m_224,
        }[backbone_name]
        model = fn(pretrained=False, img_size=img_size)
        adapter = TinyViTAdapter(model, img_size)
        return adapter, adapter.out_channels
    
    if backbone_name.startswith('efficientvit'):
        fn = {
            'efficientvit_b0': efficientvit_backbone_b0,
            'efficientvit_b1': efficientvit_backbone_b1,
            'efficientvit_b2': efficientvit_backbone_b2,
        }[backbone_name]
        model = fn()
        adapter = EfficientViTAdapter(model)
        return adapter, adapter.out_channels
    
    raise ValueError(f"Unknown backbone: {backbone_name}")


class StudentTrunk(nn.Module):
    """
    Student trunk that outputs features matching SAM3's Hiera trunk output format.
    
    Input: images (B, 3, H, W)
    Output: features (B, embed_dim, H/16, W/16) - matches Hiera trunk output
    """
    
    def __init__(self, backbone_name: str, embed_dim: int = 1024, 
                 embed_size: int = 72, img_size: int = 1008):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.embed_size = embed_size
        self.img_size = img_size
        
        # Build backbone
        self.backbone, in_channels = build_student_backbone(backbone_name, img_size)
        
        # Projection head to match teacher trunk dimension
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through student trunk.
        
        Args:
            x: Input images (B, 3, H, W)
            
        Returns:
            Features (B, embed_dim, embed_size, embed_size)
        """
        # Get backbone features via adapter
        feats = self.backbone(x)
        
        # Project to embed_dim
        feats = self.head(feats)
        
        # Resize to embed_size if needed
        if feats.shape[-1] != self.embed_size or feats.shape[-2] != self.embed_size:
            feats = F.interpolate(
                feats,
                size=(self.embed_size, self.embed_size),
                mode="bilinear",
                align_corners=False,
            )
        
        return feats


class GeometryFinetuneModel(nn.Module):
    """
    Stage 2 Geometry Fine-tuning Model with DUAL-PATH Distillation.
    
    Architecture:
    - Student trunk (trainable): Lightweight backbone producing trunk embeddings
    - Frozen SAM3 components: SimpleFPN, GeometryEncoder, Transformer, SegmentationHead
    
    Training:
    - Forward student trunk → get student embeddings
    - Load saved teacher trunk embeddings  
    - Run BOTH through frozen SAM3 components
    - Compute: embedding_loss + mask_loss
    """
    
    def __init__(
        self,
        student_backbone_name: str,
        sam3_checkpoint_path: str,
        embed_dim: int = 1024,
        embed_size: int = 72,
        img_size: int = 1008,
        freeze_fpn: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.embed_size = embed_size
        self.img_size = img_size
        self.freeze_fpn = freeze_fpn
        
        # Build student trunk (trainable)
        self.student_trunk = StudentTrunk(
            backbone_name=student_backbone_name,
            embed_dim=embed_dim,
            embed_size=embed_size,
            img_size=img_size,
        )
        
        # Load SAM3 for frozen components
        if sam3_checkpoint_path is None:
            raise ValueError("sam3_checkpoint_path is required for Stage 2 mask distillation")
            
        print(f"Loading SAM3 from {sam3_checkpoint_path}...")
        from sam3.model_builder import build_sam3_image_model
        
        self.sam3 = build_sam3_image_model(
            checkpoint_path=sam3_checkpoint_path,
            load_from_HF=False if sam3_checkpoint_path else True,
            eval_mode=True,
            device="cpu",  # Load on CPU first
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
            enable_text_encoder=False,  # No text encoder needed
        )
        
        # Freeze all SAM3 components
        for param in self.sam3.parameters():
            param.requires_grad = False
        self.sam3.eval()
        
        # Extract frozen components from SAM3's vision backbone
        # These are the SimpleFPN convs that process trunk output
        self.frozen_convs = self.sam3.backbone.vision_backbone.convs
        self.frozen_position_encoding = self.sam3.backbone.vision_backbone.position_encoding
        self.scale_factors = self.sam3.backbone.vision_backbone.scale_factors
        
        # Reference to frozen SAM3 components for mask prediction
        self.geometry_encoder = self.sam3.geometry_encoder
        self.transformer = self.sam3.transformer
        self.segmentation_head = self.sam3.segmentation_head
        
        print(f"GeometryFinetuneModel initialized:")
        print(f"  - Student trunk: {student_backbone_name}")
        print(f"  - Embed dim: {embed_dim}, Embed size: {embed_size}")
        print(f"  - Frozen FPN: {freeze_fpn}")
        print(f"  - Mode: DUAL-PATH (embedding + mask distillation)")
        
    def forward_student(self, images: torch.Tensor) -> torch.Tensor:
        """Get student trunk embeddings."""
        return self.student_trunk(images)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass - returns student embeddings."""
        return self.forward_student(images)
    
    def apply_fpn(self, trunk_output: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Apply SimpleFPN neck to trunk output.
        
        Args:
            trunk_output: Trunk features (B, C, H, W)
            
        Returns:
            Tuple of (fpn_features, fpn_pos_encodings)
            Each is a list of 4 tensors for 4 scale levels
        """
        fpn_out = []
        fpn_pos = []
        
        x = trunk_output
        for conv in self.frozen_convs:
            fpn_x = conv(x)
            fpn_pos_out = self.frozen_position_encoding(fpn_x).to(fpn_x.dtype)
            fpn_out.append(fpn_x)
            fpn_pos.append(fpn_pos_out)
        
        return fpn_out, fpn_pos
    
    def forward_mask_prediction(
        self,
        trunk_output: torch.Tensor,
        boxes: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward through frozen SAM3 components to get mask predictions.
        
        This runs: trunk_output → SimpleFPN → GeometryEncoder → Transformer → SegmentationHead
        
        Args:
            trunk_output: Trunk embeddings (B, embed_dim, H, W)
            boxes: Box prompts (B, N, 4) in cxcywh format, normalized to [0,1]
            points: Point prompts (B, N, 2) normalized to [0,1]
            point_labels: Point labels (B, N) - 1 for foreground, 0 for background
            
        Returns:
            Dict with 'pred_masks', 'pred_logits', 'pred_boxes' etc.
        """
        batch_size = trunk_output.shape[0]
        device = trunk_output.device
        
        # Step 1: Apply FPN to get multi-scale features
        fpn_out, fpn_pos = self.apply_fpn(trunk_output)
        
        # SAM3 uses num_feature_levels=1, so take only last level for encoder
        num_feature_levels = self.sam3.num_feature_levels
        fpn_out_for_enc = fpn_out[-num_feature_levels:]
        fpn_pos_for_enc = fpn_pos[-num_feature_levels:]
        
        vis_feat_sizes = [x.shape[-2:] for x in fpn_out_for_enc]  # (H, W) shapes
        
        # Step 2: Build geometric prompt in SAM3's format
        from sam3.model.geometry_encoders import Prompt
        
        geometric_prompt = self._build_geometric_prompt(
            boxes=boxes,
            points=points,
            point_labels=point_labels,
            batch_size=batch_size,
            device=device,
        )
        
        # Step 3: Prepare image features in SAM3's format (seq-first)
        # SAM3 expects: [H*W, B, C] format
        img_feats = [x.flatten(2).permute(2, 0, 1) for x in fpn_out_for_enc]
        img_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in fpn_pos_for_enc]
        
        # Also prepare all FPN levels for geometry encoder (it may use all)
        all_img_feats = [x.flatten(2).permute(2, 0, 1) for x in fpn_out]
        all_img_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in fpn_pos]
        all_vis_feat_sizes = [x.shape[-2:] for x in fpn_out]
        
        # Step 4: Encode geometry prompts
        geo_feats, geo_masks = self.geometry_encoder(
            geo_prompt=geometric_prompt,
            img_feats=all_img_feats,
            img_sizes=all_vis_feat_sizes,
            img_pos_embeds=all_img_pos_embeds,
        )
        
        # No text features, so prompt = geometry only
        prompt = geo_feats
        prompt_mask = geo_masks
        
        # Step 5: Run transformer encoder (only uses last num_feature_levels)
        prompt_pos_embed = torch.zeros_like(prompt)
        memory = self.transformer.encoder(
            src=[f.clone() for f in img_feats],  # Clone to avoid in-place modification
            src_key_padding_mask=None,
            src_pos=[f.clone() for f in img_pos_embeds],
            prompt=prompt,
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes,
        )
        
        encoder_out = {
            "encoder_hidden_states": memory["memory"],
            "pos_embed": memory["pos_embed"],
            "padding_mask": memory["padding_mask"],
            "level_start_index": memory["level_start_index"],
            "spatial_shapes": memory["spatial_shapes"],
            "valid_ratios": memory["valid_ratios"],
            "vis_feat_sizes": vis_feat_sizes,
        }
        
        # Step 6: Run transformer decoder
        query_embed = self.transformer.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
        
        hs, reference_boxes, dec_presence_out, dec_presence_feats = self.transformer.decoder(
            tgt=tgt,
            memory=memory["memory"],
            memory_key_padding_mask=memory["padding_mask"],
            pos=memory["pos_embed"],
            reference_boxes=None,
            level_start_index=encoder_out["level_start_index"],
            spatial_shapes=encoder_out["spatial_shapes"],
            valid_ratios=encoder_out["valid_ratios"],
            tgt_mask=None,
            memory_text=prompt,
            text_attention_mask=prompt_mask,
            apply_dac=False,
        )
        
        hs = hs.transpose(1, 2)  # seq-first to batch-first
        reference_boxes = reference_boxes.transpose(1, 2)
        
        # Step 7: Score prediction
        if hasattr(self.sam3, 'dot_prod_scoring') and self.sam3.use_dot_prod_scoring:
            outputs_class = self.sam3.dot_prod_scoring(hs, prompt, prompt_mask)
        else:
            outputs_class = self.sam3.class_embed(hs)
        
        # Step 8: Run segmentation head
        img_ids = torch.arange(batch_size, device=device)
        
        # Create backbone_out dict for segmentation head
        backbone_out = {
            "backbone_fpn": fpn_out,
            "vision_pos_enc": fpn_pos,
        }
        
        seg_outputs = self.segmentation_head(
            backbone_feats=fpn_out,
            obj_queries=hs,
            image_ids=img_ids,
            encoder_hidden_states=memory["memory"],
            act_ckpt_enable=False,
            prompt=prompt,
            prompt_mask=prompt_mask,
        )
        
        # Compile output
        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": reference_boxes[-1],
        }
        for k, v in seg_outputs.items():
            out[k] = v
        
        return out
    
    def _build_geometric_prompt(
        self,
        boxes: Optional[torch.Tensor],
        points: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ):
        """Build SAM3's Prompt object from boxes and points."""
        from sam3.model.geometry_encoders import Prompt
        
        # Convert boxes from (B, N, 4) to (N, B, 4) format
        if boxes is not None and boxes.numel() > 0:
            boxes_t = boxes.transpose(0, 1)  # (N, B, 4)
            n_boxes = boxes_t.shape[0]
            box_mask = torch.zeros(batch_size, n_boxes, device=device, dtype=torch.bool)
            box_labels = torch.ones(n_boxes, batch_size, device=device, dtype=torch.long)
        else:
            boxes_t = torch.zeros(0, batch_size, 4, device=device)
            box_mask = torch.zeros(batch_size, 0, device=device, dtype=torch.bool)
            box_labels = torch.zeros(0, batch_size, device=device, dtype=torch.long)
        
        # Convert points from (B, N, 2) to (N, B, 2) format  
        if points is not None and points.numel() > 0:
            points_t = points.transpose(0, 1)  # (N, B, 2)
            n_points = points_t.shape[0]
            point_mask = torch.zeros(batch_size, n_points, device=device, dtype=torch.bool)
            if point_labels is not None:
                labels_t = point_labels.transpose(0, 1)  # (N, B)
            else:
                labels_t = torch.ones(n_points, batch_size, device=device, dtype=torch.long)
        else:
            points_t = torch.zeros(0, batch_size, 2, device=device)
            point_mask = torch.zeros(batch_size, 0, device=device, dtype=torch.bool)
            labels_t = torch.zeros(0, batch_size, device=device, dtype=torch.long)
        
        geometric_prompt = Prompt(
            box_embeddings=boxes_t,
            box_mask=box_mask,
            box_labels=box_labels,
            point_embeddings=points_t,
            point_mask=point_mask,
            point_labels=labels_t,
        )
        
        return geometric_prompt
    
    def train(self, mode: bool = True):
        """Set training mode - only student trunk is trainable."""
        super().train(mode)
        # Keep SAM3 in eval mode always
        self.sam3.eval()
        return self


def load_stage1_weights(model: GeometryFinetuneModel, stage1_checkpoint: str, logger=None):
    """
    Load Stage 1 pretrained weights into the student trunk.
    
    Args:
        model: GeometryFinetuneModel instance
        stage1_checkpoint: Path to Stage 1 checkpoint (merged with SAM3)
        logger: Optional logger
    """
    if logger:
        logger.info(f"Loading Stage 1 weights from: {stage1_checkpoint}")
    else:
        print(f"Loading Stage 1 weights from: {stage1_checkpoint}")
    
    checkpoint = torch.load(stage1_checkpoint, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # The merged Stage 1 checkpoint has keys like:
    # detector.backbone.vision_backbone.trunk.model.backbone.model.features...
    # We need to map these to our student_trunk.backbone.model.features...
    
    # Prefix to strip from merged checkpoint
    merged_prefix = "detector.backbone.vision_backbone.trunk.model."
    
    new_state_dict = {}
    loaded_count = 0
    
    for k, v in state_dict.items():
        if k.startswith(merged_prefix):
            # Strip the merged prefix and add student_trunk prefix
            suffix = k[len(merged_prefix):]
            new_key = f'student_trunk.{suffix}'
            new_state_dict[new_key] = v
            loaded_count += 1
    
    if loaded_count == 0:
        # Try alternate format (raw Stage 1 training checkpoint)
        for k, v in state_dict.items():
            if k.startswith('backbone.') or k.startswith('head.'):
                new_state_dict[f'student_trunk.{k}'] = v
                loaded_count += 1
    
    # Load with strict=False since we might have extra keys
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    
    if logger:
        logger.info(f"  Loaded {loaded_count} keys into student_trunk")
        logger.info(f"  Missing keys: {len(missing)}")
        if len(missing) > 0:
            logger.info(f"    First few missing: {missing[:3]}")
        logger.info(f"  Unexpected keys: {len(unexpected)}")
    else:
        print(f"  Loaded {loaded_count} keys into student_trunk")
        print(f"  Missing keys: {len(missing)}")
        if len(missing) > 0:
            print(f"    First few missing: {missing[:3]}")
        print(f"  Unexpected keys: {len(unexpected)}")
    
    return model
