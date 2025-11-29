# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import Optional, Tuple

from sam3.model.sam3_tracker_base import Sam3TrackerBase
from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor
from sam3.model.hybrid_memory import HybridMemoryModule
from sam3.model.efficient_attention import EfficientRoPEAttention
from sam3.model.decoder import TransformerDecoderLayerv2, TransformerEncoderCrossAttention
from sam3.model.model_misc import TransformerWrapper

class EfficientSam3Tracker(Sam3TrackerPredictor):
    """
    EfficientSAM3 Tracker that uses Hybrid Memory Module (EdgeTAM) and Efficient Attention (EfficientTAM).
    """
    def __init__(
        self,
        hybrid_memory_dim=256,
        num_global_latents=64,
        num_spatial_latents=192,
        window_size=8,
        **kwargs
    ):
        # We need to intercept the creation of maskmem_backbone and transformer
        # But Sam3TrackerPredictor calls super().__init__ which expects them.
        # So we should create them here and pass them to super().__init__
        
        # 1. Create Hybrid Memory Module
        maskmem_backbone = HybridMemoryModule(
            dim=hybrid_memory_dim,
            num_global_latents=num_global_latents,
            num_spatial_latents=num_spatial_latents,
            window_size=window_size
        )
        
        # 2. Create Efficient Transformer
        # We need to replace the standard RoPEAttention with one that handles the compressed memory
        # The memory (K, V) will be [B, N_latents, C]
        # The query (Q) will be [B, H*W, C] (current frame features)
        
        # Self Attention (for current frame) - Standard RoPE
        self_attention = EfficientRoPEAttention(
            embedding_dim=256,
            num_heads=1,
            downsample_rate=1,
            dropout=0.1,
            rope_theta=10000.0,
            feat_sizes=[72, 72], # Assuming 1008/14 = 72
            use_fa3=False,
            use_rope_real=False,
        )

        # Cross Attention (Query=Frame, Key/Value=Memory)
        # Memory is 1D latents, so we disable 2D RoPE for K/V or handle it appropriately.
        # EfficientRoPEAttention has rope_k_repeat.
        # If K is 1D, we might not want RoPE on it, or we assume it has position info.
        # EdgeTAM adds position info to latents.
        # So we probably don't need RoPE on K.
        
        cross_attention = EfficientRoPEAttention(
            embedding_dim=256,
            num_heads=1,
            downsample_rate=1,
            dropout=0.1,
            kv_in_dim=hybrid_memory_dim, # Should match memory dim
            rope_theta=10000.0,
            feat_sizes=[72, 72], # For Q
            rope_k_repeat=False, # Don't repeat RoPE for K if it's not spatial?
            # We need to ensure RoPE is NOT applied to K if it's latents.
            # EfficientRoPEAttention implementation needs to support this.
            # If feat_sizes is provided, it tries to apply RoPE.
            # We might need to modify EfficientRoPEAttention to skip RoPE for K if it's not spatial.
            use_fa3=False,
            use_rope_real=False,
        )

        # Encoder layer
        encoder_layer = TransformerDecoderLayerv2(
            cross_attention_first=False,
            activation="relu",
            dim_feedforward=2048,
            dropout=0.1,
            pos_enc_at_attn=False,
            pre_norm=True,
            self_attention=self_attention,
            d_model=256,
            pos_enc_at_cross_attn_keys=True, # We might want to disable this if latents have their own PE
            pos_enc_at_cross_attn_queries=False,
            cross_attention=cross_attention,
        )

        # Encoder
        encoder = TransformerEncoderCrossAttention(
            remove_cross_attention_layers=[],
            batch_first=True,
            d_model=256,
            frozen=False,
            pos_enc_at_input=True,
            layer=encoder_layer,
            num_layers=4,
            use_act_checkpoint=False,
        )

        # Transformer wrapper
        transformer = TransformerWrapper(
            encoder=encoder,
            decoder=None,
            d_model=256,
        )
        
        # Update kwargs with our modules
        kwargs['maskmem_backbone'] = maskmem_backbone
        kwargs['transformer'] = transformer
        
        super().__init__(**kwargs)

    def forward(self, *args, **kwargs):
        # We might need to override forward if the input/output structure changes
        # But Sam3TrackerPredictor uses Sam3TrackerBase.forward which relies on the components we replaced.
        return super().forward(*args, **kwargs)
