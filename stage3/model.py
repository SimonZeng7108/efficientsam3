"""Stage 3 Model: End-to-End Fine-Tuning for EfficientSAM3.

This wraps the REAL student EfficientSAM3 model (student vision trunk + student
MobileCLIP-S0 text encoder + SAM3 FPN / transformer / segmentation head) so that
training and inference use the exact same network topology. Stage 1 merged
checkpoints are loaded natively via ``build_efficientsam3_image_model``, which
already knows how to strip ``detector.`` / ``student_trunk.`` prefixes and
populate every module (not just the trunk).

Trainable scopes (controlled by ``trainable_scope``):
    * ``trunk_only``    - only ``backbone.vision_backbone.trunk`` is trained
    * ``trunk_fpn``     - ``trunk_only`` plus ``backbone.vision_backbone.convs``
                          (SimpleFPN projection towers)
    * ``trunk_seghead`` - ``trunk_only`` plus ``segmentation_head``

The text encoder (MobileCLIP-S0) is always frozen. The transformer encoder /
decoder and the DETR query head are always frozen, because they are already
well pretrained by SAM3 and fine-tuning them on our small SA-1B-1% subset
would very likely regress open-vocabulary recall.

Single-prompt supervision
-------------------------
Stage 3 adopts a **one-prompt-per-sample** topology:

    sample = (image, 1 caption, 1 box, K points, 1 GT mask)

This makes text / geometry / mask prompts mutually consistent (they all refer
to the *same* target instance), removes the need for a Hungarian matcher and
matches how SAM3 is used at inference time. The 200 DETR queries still run, and
we select the best-IoU query per sample for mask supervision; a lightweight
presence loss teaches ``pred_logits.argmax`` to land on that query slot.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from sam3.model.data_misc import FindStage
from sam3.model.geometry_encoders import Prompt


TRAINABLE_SCOPES = ("trunk_only", "trunk_fpn", "trunk_seghead")


class Stage3FinetuneModel(nn.Module):
    """End-to-end trainable wrapper around the student EfficientSAM3 model.

    The wrapped model is the same object returned by
    ``sam3.model_builder.build_efficientsam3_image_model`` - i.e. it contains:
        * student vision backbone (``backbone.vision_backbone``)
        * student MobileCLIP-S0 text encoder (``backbone.language_backbone``)
        * SAM3 transformer, dot-product scoring head, segmentation head
    """

    def __init__(
        self,
        backbone_type: str,
        model_name: str,
        stage1_checkpoint_path: Optional[str] = None,
        text_encoder_type: str = "MobileCLIP-S0",
        text_encoder_context_length: int = 77,
        trainable_scope: str = "trunk_only",
        img_size: int = 1008,
    ):
        super().__init__()
        assert trainable_scope in TRAINABLE_SCOPES, (
            f"trainable_scope must be one of {TRAINABLE_SCOPES}, "
            f"got '{trainable_scope}'"
        )
        self.trainable_scope = trainable_scope
        self.img_size = img_size

        from sam3.model_builder import build_efficientsam3_image_model

        print(
            f"[Stage3] Building EfficientSAM3 student: "
            f"backbone={backbone_type}/{model_name}, text={text_encoder_type}"
        )
        if stage1_checkpoint_path:
            print(f"[Stage3] Loading Stage 1 merged checkpoint: {stage1_checkpoint_path}")

        self.model = build_efficientsam3_image_model(
            backbone_type=backbone_type,
            model_name=model_name,
            checkpoint_path=stage1_checkpoint_path,
            load_from_HF=False,
            eval_mode=False,
            device="cpu",
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
            text_encoder_type=text_encoder_type,
            text_encoder_context_length=text_encoder_context_length,
        )

        for p in self.model.parameters():
            p.requires_grad = False

        self._apply_trainable_scope()

        self._trunk_features: Optional[torch.Tensor] = None
        self._register_trunk_hook()

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(
            f"[Stage3] Stage3FinetuneModel [{trainable_scope}]: "
            f"{n_train/1e6:.2f}M trainable / {n_total/1e6:.2f}M total"
        )

    def _apply_trainable_scope(self):
        vb = self.model.backbone.vision_backbone
        for p in vb.trunk.parameters():
            p.requires_grad = True
        if self.trainable_scope == "trunk_fpn":
            for p in vb.convs.parameters():
                p.requires_grad = True
        elif self.trainable_scope == "trunk_seghead":
            if self.model.segmentation_head is not None:
                for p in self.model.segmentation_head.parameters():
                    p.requires_grad = True

    def _register_trunk_hook(self):
        """Capture the student trunk output (pre-FPN) for embedding distillation.

        ``ImageStudentEncoder`` produces a ``(B, 1024, 72, 72)`` feature map
        that matches the saved SAM3 teacher embeddings exactly.
        """
        trunk_wrapper = self.model.backbone.vision_backbone.trunk
        student_encoder = trunk_wrapper.model

        def _hook(_module, _inputs, output):
            self._trunk_features = output

        student_encoder.register_forward_hook(_hook)

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.backbone.language_backbone.eval()
        self.model.transformer.eval()
        if hasattr(self.model, "dot_prod_scoring") and self.model.dot_prod_scoring is not None:
            self.model.dot_prod_scoring.eval()
        if self.trainable_scope != "trunk_seghead" and self.model.segmentation_head is not None:
            self.model.segmentation_head.eval()
        if self.trainable_scope == "trunk_only":
            self.model.backbone.vision_backbone.convs.eval()
        return self

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def forward_text(
        self,
        captions: List[str],
        device: torch.device,
        cast_dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode text prompts with the frozen student MobileCLIP-S0 in fp32.

        Runs the text encoder outside the outer autocast context so that the
        pretrained weights see their original dtype, then casts the returned
        language features to ``cast_dtype`` (typically fp16) so they can be
        concatenated with geometry features inside the autocast region.
        """
        disable_ctx = (
            torch.cuda.amp.autocast(enabled=False)
            if torch.cuda.is_available()
            else nullcontext()
        )
        with disable_ctx:
            out = self.model.backbone.forward_text(captions, device=device)
        if cast_dtype is not None:
            if out.get("language_features") is not None:
                out["language_features"] = out["language_features"].to(cast_dtype)
        return out

    def forward_grounding(
        self,
        images: torch.Tensor,
        captions: List[str],
        boxes_cxcywh_norm: torch.Tensor,
        points_norm: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """End-to-end forward for single-prompt supervision.

        Args:
            images:            (B, 3, H, W) normalized and padded.
            captions:          list of length B.
            boxes_cxcywh_norm: (B, 4) normalized cxcywh box, one per image.
            points_norm:       (B, K, 2) normalized xy points per image, or None.

        Returns:
            dict with keys ``pred_masks`` (B, Q, Hm, Wm),
            ``pred_logits`` (B, Q, 1), ``pred_boxes`` (B, Q, 4) and
            ``trunk_features`` (B, 1024, 72, 72).
        """
        device = images.device
        B = images.shape[0]

        self._trunk_features = None
        backbone_out = {"img_batch_all_stages": images}
        backbone_out.update(self.model.backbone.forward_image(images))

        cast_dtype = (
            torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else None
        )
        text_out = self.forward_text(captions, device=device, cast_dtype=cast_dtype)
        backbone_out.update(text_out)

        img_ids = torch.arange(B, device=device, dtype=torch.long)
        text_ids = torch.arange(B, device=device, dtype=torch.long)

        boxes_t = boxes_cxcywh_norm.unsqueeze(0)  # (1, B, 4)
        box_mask = torch.zeros(B, 1, device=device, dtype=torch.bool)
        box_labels = torch.ones(1, B, device=device, dtype=torch.long)

        if points_norm is not None and points_norm.numel() > 0:
            K = points_norm.shape[1]
            points_t = points_norm.transpose(0, 1).contiguous()  # (K, B, 2)
            point_mask = torch.zeros(B, K, device=device, dtype=torch.bool)
            point_labels = torch.ones(K, B, device=device, dtype=torch.long)
        else:
            points_t = torch.zeros(0, B, 2, device=device)
            point_mask = torch.zeros(B, 0, device=device, dtype=torch.bool)
            point_labels = torch.zeros(0, B, device=device, dtype=torch.long)

        geo_prompt = Prompt(
            box_embeddings=boxes_t,
            box_mask=box_mask,
            box_labels=box_labels,
            point_embeddings=points_t,
            point_mask=point_mask,
            point_labels=point_labels,
        )

        find_input = FindStage(
            img_ids=img_ids,
            text_ids=text_ids,
            input_boxes=boxes_t,
            input_boxes_mask=box_mask,
            input_boxes_label=box_labels,
            input_points=points_t,
            input_points_mask=point_mask,
        )

        prompt, prompt_mask, backbone_out = self.model._encode_prompt(
            backbone_out, find_input, geo_prompt
        )

        backbone_out, encoder_out, _ = self.model._run_encoder(
            backbone_out, find_input, prompt, prompt_mask
        )

        out: Dict[str, torch.Tensor] = {
            "encoder_hidden_states": encoder_out["encoder_hidden_states"],
        }

        out, hs = self.model._run_decoder(
            memory=out["encoder_hidden_states"],
            pos_embed=encoder_out["pos_embed"],
            src_mask=encoder_out["padding_mask"],
            out=out,
            prompt=prompt,
            prompt_mask=prompt_mask,
            encoder_out=encoder_out,
        )

        self.model._run_segmentation_heads(
            out=out,
            backbone_out=backbone_out,
            img_ids=find_input.img_ids,
            vis_feat_sizes=encoder_out["vis_feat_sizes"],
            encoder_hidden_states=out["encoder_hidden_states"],
            prompt=prompt,
            prompt_mask=prompt_mask,
            hs=hs,
        )

        out["trunk_features"] = self._trunk_features
        return out


def load_stage1_merged_into_stage3(
    stage3_model: Stage3FinetuneModel,
    stage1_checkpoint_path: str,
    logger=None,
):
    """Secondary loader: re-apply a Stage 1 merged checkpoint after construction.

    Normally the checkpoint is loaded inside ``build_efficientsam3_image_model``
    at ``__init__`` time. This helper is only used when resuming onto a fresh
    model instance (e.g. from an ablation script that rebuilds the model).
    """
    from sam3.model_builder import _load_checkpoint

    if logger is not None:
        logger.info(f"[Stage3] Re-loading Stage 1 merged checkpoint: {stage1_checkpoint_path}")
    _load_checkpoint(stage3_model.model, stage1_checkpoint_path)
    return stage3_model
