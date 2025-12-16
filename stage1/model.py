from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

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
from sam3.backbones.mobile_clip import MobileCLIPTextTransformer
from sam3.model.tokenizer_ve import SimpleTokenizer
from sam3.model.text_encoder_student import TextStudentEncoder, TextStudentEncoderWithTeacherEmbed


def build_image_student_model(config):
    backbone_name = config.MODEL.BACKBONE.lower()
    backbone, out_channels = _build_backbone(backbone_name, config.DATA.IMG_SIZE)
    return ImageStudentEncoder(
        backbone=backbone,
        in_channels=out_channels,
        embed_dim=config.DISTILL.EMBED_DIM,
        embed_size=config.DISTILL.EMBED_SIZE,
        img_size=config.DATA.IMG_SIZE,
    )


def build_text_student_model(config, logger=None):
    backbone = config.MODEL.BACKBONE
    
    # Check if we should use teacher embeddings (Option 3)
    use_teacher_embed = getattr(config.MODEL, 'USE_TEACHER_EMBED', False)
    teacher_checkpoint = getattr(config.MODEL, 'TEACHER_CHECKPOINT', None)
    
    # Default config values
    cfg = {
        "context_length": 77, # MobileCLIP default
        "vocab_size": 49408,
        "dim": 512,
        "ffn_multiplier_per_layer": 4.0,
        "n_heads_per_layer": 8,
        "n_transformer_layers": 12,
        "norm_layer": "layer_norm_fp32",
        "causal_masking": False,
        "model_name": "base",
        "embed_dropout": 0.0,
        "no_scale_embedding": False,
        "no_pos_embedding": False,
    }

    if backbone == "MobileCLIP-S0":
        cfg.update({
            "dim": 512,
            "n_transformer_layers": 4,  # Original MobileCLIP-S0 design
            "n_heads_per_layer": 8,
            "model_name": "mct",  # RepMixer architecture
            "ffn_multiplier_per_layer": 4.0,  # Original FFN multiplier
        })
    elif backbone in ["MobileCLIP-S1", "MobileCLIP2-S0", "MobileCLIP2-S2"]:
        cfg.update({
            "dim": 512,
            "n_transformer_layers": 12,
            "n_heads_per_layer": 8,
            "model_name": "base",
        })
    elif backbone == "MobileCLIP-B":
        cfg.update({
            "dim": 512,
            "n_transformer_layers": 12,
            "n_heads_per_layer": 8,
            "model_name": "base",
            "causal_masking": True,
        })
    elif backbone in ["MobileCLIP2-S3", "MobileCLIP2-S4", "MobileCLIP2-L"]:
        cfg.update({
            "dim": 768,
            "n_transformer_layers": 12,
            "n_heads_per_layer": 12,
            "model_name": "base", 
        })
        # Note: MobileCLIP2 might use "custom_text" or similar, but "base" with larger dim should work if architecture is standard transformer.
        # The subagent said "custom_text: true, no_causal_mask: true".
        # If "custom_text" implies standard transformer, then "base" is fine.
    else:
        # Default fallback
        pass

    # Choose model class based on whether we use teacher embeddings
    if use_teacher_embed:
        if logger:
            logger.info("Using teacher embeddings (frozen) with student transformer")
        
        if not teacher_checkpoint:
            raise ValueError(
                "MODEL.USE_TEACHER_EMBED is True but MODEL.TEACHER_CHECKPOINT is not set. "
                "Please provide the path to the SAM3 checkpoint."
            )
        
        model = TextStudentEncoderWithTeacherEmbed(
            cfg=cfg,
            context_length=32,  # Match teacher input length
            output_dim=config.DISTILL.EMBED_DIM,
            teacher_embed_dim=1024,  # SAM3 teacher uses 1024-dim embeddings
        )
        
        # Load and freeze teacher embeddings
        model.load_teacher_embeddings(teacher_checkpoint, logger=logger)
        
        if logger:
            # Count trainable vs frozen parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            logger.info(f"Model parameters: total={total_params:,}, trainable={trainable_params:,}, frozen={frozen_params:,}")
        
        return model

    # Standard approach: full student replacement
    model = TextStudentEncoder(
        cfg=cfg,
        context_length=32, # Match teacher input length
        output_dim=config.DISTILL.EMBED_DIM,
    )
    
    # Load pretrained weights if specified
    if hasattr(config.MODEL, 'PRETRAINED') and config.MODEL.PRETRAINED:
        pretrained_path = config.MODEL.PRETRAINED
        if logger:
            logger.info(f"Loading pretrained text encoder from: {pretrained_path}")
        
        try:
            pretrained_state = torch.load(pretrained_path, map_location='cpu')
            
            # Handle full MobileCLIP checkpoint (contains image + text encoders)
            if 'text_encoder.embedding_layer.weight' in pretrained_state:
                if logger:
                    logger.info("Detected full MobileCLIP checkpoint, extracting text encoder...")
                # Extract text encoder weights
                text_encoder_state = {}
                for key, value in pretrained_state.items():
                    if key.startswith('text_encoder.'):
                        new_key = key.replace('text_encoder.', 'encoder.')
                        text_encoder_state[new_key] = value
                pretrained_state = text_encoder_state
            
            # Load weights with flexible matching
            missing_keys, unexpected_keys = model.load_state_dict(pretrained_state, strict=False)
            
            if logger:
                logger.info(f"Loaded pretrained weights:")
                logger.info(f"  Missing keys: {len(missing_keys)}")
                if len(missing_keys) > 0 and len(missing_keys) <= 10:
                    for key in missing_keys:
                        logger.info(f"    - {key}")
                logger.info(f"  Unexpected keys: {len(unexpected_keys)}")
                if len(unexpected_keys) > 0 and len(unexpected_keys) <= 10:
                    for key in unexpected_keys:
                        logger.info(f"    - {key}")
                
                # The projector layer should be missing (we train it from scratch)
                expected_missing = ['projector.weight', 'projector.bias']
                actual_missing_important = [k for k in missing_keys if k not in expected_missing]
                if len(actual_missing_important) > 0:
                    logger.warning(f"  Important missing keys (not projector): {actual_missing_important}")
                else:
                    logger.info(f"  ✓ All important keys loaded successfully")
        
        except Exception as e:
            if logger:
                logger.error(f"Failed to load pretrained weights: {e}")
                logger.warning("Continuing with random initialization...")
            else:
                print(f"Warning: Failed to load pretrained weights: {e}")
    
    return model


def build_image_teacher_model(config):
    checkpoint = config.MODEL.RESUME if config.MODEL.RESUME else None
    teacher = SAM3ImageTeacherEncoder(
        checkpoint_path=checkpoint,
        embed_size=config.DISTILL.EMBED_SIZE,
    )
    teacher.img_size = config.DATA.IMG_SIZE
    return teacher


def build_text_teacher_model(config):
    checkpoint = config.MODEL.RESUME if config.MODEL.RESUME else None
    teacher = SAM3TextTeacherEncoder(
        checkpoint_path=checkpoint,
    )
    return teacher


class ImageStudentEncoder(nn.Module):
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


class SAM3ImageTeacherEncoder(nn.Module):
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


class SAM3TextTeacherEncoder(nn.Module):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        self.sam3 = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=False if checkpoint_path else True,
            eval_mode=True,
            device="cpu",
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
            enable_text_encoder=True,
            enable_vision_encoder=False,
        )
        for param in self.sam3.parameters():
            param.requires_grad = False
        self.sam3.eval()

    def forward(self, text, device):
        # text is a list of strings
        text_attention_mask, text_memory, text_embeds = self.sam3.backbone.language_backbone(
            text, input_boxes=None, device=device
        )
        # text_memory is [Seq, Batch, 256]
        return text_memory


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
    student_encoder = build_image_student_model(config)

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


