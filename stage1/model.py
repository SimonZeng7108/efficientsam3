from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3.model_builder import build_sam3_image_model
from sam3.backbones.repvit import (
    _make_divisible,
    repvit_m0_9,
    repvit_m1_1,
    repvit_m2_3,
)
from sam3.backbones.tiny_vit import (
    tiny_vit_5m_224,
    tiny_vit_11m_224,
    tiny_vit_21m_224,
)
from sam3.backbones.efficientvit import (
    efficientvit_backbone_b0,
    efficientvit_backbone_b1,
    efficientvit_backbone_b2,
)


def build_student_model(config):
    backbone_name = config.MODEL.BACKBONE.lower()
    backbone, out_channels = _build_backbone(backbone_name, config.DATA.IMG_SIZE)
    return StudentEncoder(
        backbone=backbone,
        in_channels=out_channels,
        embed_dim=config.DISTILL.EMBED_DIM,
        embed_size=config.DISTILL.EMBED_SIZE,
        img_size=config.DATA.IMG_SIZE,
    )


def build_teacher_model(config):
    checkpoint = config.MODEL.RESUME if config.MODEL.RESUME else None
    teacher = SAM3TeacherEncoder(
        checkpoint_path=checkpoint,
        embed_size=config.DISTILL.EMBED_SIZE,
    )
    teacher.img_size = config.DATA.IMG_SIZE
    return teacher


class StudentEncoder(nn.Module):
    def __init__(self, backbone, in_channels, embed_dim, embed_size, img_size):
        super().__init__()
        self.backbone = backbone
        self.embed_size = embed_size
        self.img_size = img_size
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.head(feats)
        if feats.shape[-1] != self.embed_size or feats.shape[-2] != self.embed_size:
            feats = F.interpolate(
                feats,
                size=(self.embed_size, self.embed_size),
                mode="bilinear",
                align_corners=False,
            )
        return feats


class SAM3TeacherEncoder(nn.Module):
    def __init__(self, checkpoint_path=None, embed_size=64):
        super().__init__()
        self.embed_size = embed_size
        self.sam3 = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=False if checkpoint_path else True,
            eval_mode=True,
            device="cpu",
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
            enable_text_encoder=False,
        )
        for param in self.sam3.parameters():
            param.requires_grad = False
        self.sam3.eval()
        self.img_size = 1008

    def forward(self, x):
        # Distill the raw backbone features (1024 channels) to allow
        # the student to be a drop-in replacement for the backbone.
        # Access the ViT trunk directly through the vision_backbone
        backbone_out = self.sam3.backbone.vision_backbone.trunk(x)
        # ViT returns a list of features, we want the last one
        feats = backbone_out[-1]
        
        # Interpolate if needed (though usually trunk output is already at the target resolution)
        if feats.shape[-1] != self.embed_size or feats.shape[-2] != self.embed_size:
            feats = F.interpolate(
                feats,
                size=(self.embed_size, self.embed_size),
                mode="bilinear",
                align_corners=False,
            )
        return feats


class RepViTAdapter(nn.Module):
    def __init__(self, model, out_channels):
        super().__init__()
        self.model = model
        self.out_channels = out_channels

    def forward(self, x):
        for layer in self.model.features:
            x = layer(x)
        return x


class TinyViTAdapter(nn.Module):
    def __init__(self, model, img_size):
        super().__init__()
        self.model = model
        self.model.head = nn.Identity()
        self.final_hw = self._compute_resolution(img_size)
        self.out_channels = self.model.norm_head.normalized_shape[0]
        # Remove norm_head to avoid DDP unused parameter error
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
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.out_channels = self.model.width_list[-1]

    def forward(self, x):
        out = self.model(x)
        return out["stage_final"]


class EfficientSAM3VisionBackbone(nn.Module):
    def __init__(self, student_encoder, position_encoding):
        super().__init__()
        self.student_encoder = student_encoder
        self.position_encoding = position_encoding

    def forward(self, x):
        feats = self.student_encoder(x)
        sam3_out = [feats]
        sam3_pos = [self.position_encoding(feats).to(feats.dtype)]
        sam2_out = None
        sam2_pos = None
        return sam3_out, sam3_pos, sam2_out, sam2_pos


def build_efficient_sam3(config, checkpoint_path=None):
    student_encoder = build_student_model(config)

    # Build base SAM3 model structure
    sam3 = build_sam3_image_model(
        checkpoint_path=None,
        load_from_HF=False,
        eval_mode=True,
        device="cpu",
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    )

    # Replace vision backbone
    original_pos_enc = sam3.backbone.vision_backbone.position_encoding
    vision_backbone = EfficientSAM3VisionBackbone(student_encoder, original_pos_enc)
    sam3.backbone.vision_backbone = vision_backbone
    
    # Disable scalping since student encoder only returns one feature map
    sam3.backbone.scalp = 0
    
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
            
        # No remapping needed if checkpoint was converted with correct prefixes
        missing, unexpected = sam3.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint with {len(missing)} missing and {len(unexpected)} unexpected keys")
        
    return sam3
def _build_backbone(name, img_size):
    if name.startswith("repvit"):
        fn = {
            "repvit_m0_9": repvit_m0_9,
            "repvit_m1_1": repvit_m1_1,
            "repvit_m2_3": repvit_m2_3,
        }[name]
        model = fn(pretrained=False, num_classes=0, distillation=False)
        out_channels = _make_divisible(model.cfgs[-1][2], 8)
        return RepViTAdapter(model, out_channels), out_channels

    if name.startswith("tiny_vit"):
        fn = {
            "tiny_vit_5m": tiny_vit_5m_224,
            "tiny_vit_11m": tiny_vit_11m_224,
            "tiny_vit_21m": tiny_vit_21m_224,
        }[name]
        model = fn(pretrained=False, img_size=img_size)
        adapter = TinyViTAdapter(model, img_size)
        return adapter, adapter.out_channels

    if name.startswith("efficientvit"):
        fn = {
            "efficientvit_b0": efficientvit_backbone_b0,
            "efficientvit_b1": efficientvit_backbone_b1,
            "efficientvit_b2": efficientvit_backbone_b2,
        }[name]
        model = fn()
        adapter = EfficientViTAdapter(model)
        return adapter, adapter.out_channels

    raise ValueError(f"Unsupported backbone {name}")


