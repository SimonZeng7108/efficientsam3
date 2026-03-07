# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import os
import torch
import torch.nn as nn
from sam3.backbones.mobile_clip import MobileCLIPTextTransformer
from sam3.model.tokenizer_ve import SimpleTokenizer

class TextStudentEncoder(nn.Module):
    def __init__(self, cfg, context_length, output_dim, bpe_path=None):
        super().__init__()
        self.context_length = context_length
        
        if bpe_path is None:
            bpe_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "assets", "bpe_simple_vocab_16e6.txt.gz"
            )
        
        self.tokenizer = SimpleTokenizer(bpe_path=bpe_path)
        
        # MobileCLIP Transformer (Student)
        # Includes its own embedding layer (49408 x dim)
        self.encoder = MobileCLIPTextTransformer(
            cfg=cfg,
            projection_dim=cfg["dim"]
        )
        
        # Post-Projection (Student Dim -> SAM3 d_model)
        self.projector = nn.Linear(cfg["dim"], output_dim)

    def set_context_length(self, context_length: int):
        """Resize positional embeddings to a new context length.

        Call this after checkpoint loading to truncate from the default ctx=77.
        """
        self.context_length = context_length
        if hasattr(self.encoder, "resize_pos_embed"):
            self.encoder.resize_pos_embed(context_length)

    def forward(self, text, input_boxes=None, device=None):
        # 1. Tokenize
        tokenized = self.tokenizer(text, context_length=self.context_length).to(device)
        
        # 2. Get input embeddings via student embedding layer
        input_embeds = self.encoder.forward_embedding(tokenized)  # [B, Seq, Dim]

        # 3. MobileCLIP Transformer (pass embeddings to skip redundant lookup)
        text_memory = self.encoder(
            input_embeds, return_all_tokens=True, input_is_embeddings=True
        )  # [B, Seq, Dim]

        # 4. Post-project to SAM3 d_model
        text_memory = self.projector(text_memory)  # [B, Seq, OutputDim]

        # 5. Attention mask: False=valid token, True=padding
        text_attention_mask = (tokenized != 0).bool().ne(1)

        return text_attention_mask, text_memory.transpose(0, 1), input_embeds.transpose(0, 1)
