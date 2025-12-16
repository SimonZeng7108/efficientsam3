# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import os
import torch
import torch.nn as nn
from sam3.backbones.mobile_clip import MobileCLIPTextTransformer
from sam3.model.tokenizer_ve import SimpleTokenizer


class TextStudentEncoder(nn.Module):
    """Text encoder that replaces teacher embeddings with student embeddings.
    
    This is the standard distillation approach where the entire text encoder
    (embeddings + transformer) is trained from scratch or from pretrained MobileCLIP.
    """
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
        
        # Post-Projection (Student Dim -> Output Dim)
        self.projector = nn.Linear(cfg["dim"], output_dim)

    def forward(self, text, input_boxes=None, device=None):
        # 1. Tokenize
        tokenized = self.tokenizer(text, context_length=self.context_length).to(device)
        
        # 2. Get input embeddings
        # We use the student's embedding layer
        input_embeds = self.encoder.forward_embedding(tokenized) # [Batch, Seq, Dim]
        
        # 3. MobileCLIP Transformer
        # Pass embeddings directly
        text_memory = self.encoder(input_embeds, return_all_tokens=True, input_is_embeddings=True) 
        
        # 4. Post-Project
        text_memory = self.projector(text_memory) # [Batch, Seq, OutputDim]
        
        # 5. Prepare output tuple compatible with SAM3VLBackbone
        # text_attention_mask: [Batch, Seq] (True for padding, False for valid - inverted logic)
        # But wait, VETextEncoder returns (tokenized != 0).bool().ne(1)
        # (tokenized != 0) is True for valid. .ne(1) makes it False for valid.
        # So False is valid, True is padding.
        text_attention_mask = (tokenized != 0).bool().ne(1)
        
        # Return tuple: (mask, memory, embeds)
        # memory and embeds are [Seq, Batch, Dim]
        return text_attention_mask, text_memory.transpose(0, 1), input_embeds.transpose(0, 1)


class TextStudentEncoderWithTeacherEmbed(nn.Module):
    """Text encoder that keeps teacher embeddings frozen and only trains the transformer.
    
    This approach is recommended when:
    - You have limited training data for distillation
    - You want faster convergence
    - You want to ensure the student understands the same vocabulary as the teacher
    
    The teacher's token embeddings (49408 x 1024) are frozen and projected down to
    the student dimension (512/768), then processed by the MobileCLIP transformer.
    
    Architecture:
        Token IDs → Teacher Embedding (frozen, 49408x1024) → Embed Proj (1024→student_dim)
                  → Student Transformer → Output Projector → [B, Seq, 256]
    """
    def __init__(self, cfg, context_length, output_dim, teacher_embed_dim=1024, bpe_path=None):
        super().__init__()
        self.context_length = context_length
        self.teacher_embed_dim = teacher_embed_dim
        self.student_dim = cfg["dim"]
        
        if bpe_path is None:
            bpe_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "assets", "bpe_simple_vocab_16e6.txt.gz"
            )
        
        self.tokenizer = SimpleTokenizer(bpe_path=bpe_path)
        
        # === TEACHER EMBEDDINGS (FROZEN) ===
        # These will be loaded from the teacher checkpoint
        self.token_embedding = nn.Embedding(49408, teacher_embed_dim)
        # Use context_length (32) for positional embedding, matching what we use during training
        self.positional_embedding = nn.Parameter(torch.empty(context_length, teacher_embed_dim))
        nn.init.normal_(self.positional_embedding, std=0.01)  # Will be overwritten by teacher
        
        # === EMBEDDING PROJECTION (TRAINABLE) ===
        # Project teacher embedding dim (1024) to student dim (512/768)
        self.embed_proj = nn.Linear(teacher_embed_dim, self.student_dim)
        
        # === STUDENT TRANSFORMER (TRAINABLE) ===
        # MobileCLIP transformer without its own embeddings
        self.encoder = MobileCLIPTextTransformer(
            cfg=cfg,
            projection_dim=cfg["dim"],
            skip_embeddings=True,  # Skip MobileCLIP's embedding layer
        )
        
        # === OUTPUT PROJECTION (TRAINABLE) ===
        self.projector = nn.Linear(self.student_dim, output_dim)
        
        # Track whether teacher embeddings have been loaded
        self._teacher_embeddings_loaded = False

    def load_teacher_embeddings(self, checkpoint_path, logger=None):
        """Load only the embedding layers from teacher checkpoint and freeze them."""
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in ckpt:
            ckpt = ckpt['model']
        
        loaded_token = False
        loaded_pos = False
        
        # Find embedding weights in checkpoint
        # Teacher path: detector.backbone.language_backbone.encoder.token_embedding.weight
        for key, value in ckpt.items():
            if 'language_backbone.encoder.token_embedding.weight' in key:
                self.token_embedding.weight.data = value
                loaded_token = True
                if logger:
                    logger.info(f"Loaded teacher token embeddings from: {key}")
            if 'language_backbone.encoder.positional_embedding' in key:
                self.positional_embedding.data = value
                loaded_pos = True
                if logger:
                    logger.info(f"Loaded teacher positional embeddings from: {key}")
        
        if not loaded_token or not loaded_pos:
            # Try alternative paths
            for key, value in ckpt.items():
                if not loaded_token and 'token_embedding.weight' in key and value.shape == (49408, 1024):
                    self.token_embedding.weight.data = value
                    loaded_token = True
                    if logger:
                        logger.info(f"Loaded teacher token embeddings from: {key}")
                if not loaded_pos and 'positional_embedding' in key and value.shape[1] == 1024:
                    self.positional_embedding.data = value
                    loaded_pos = True
                    if logger:
                        logger.info(f"Loaded teacher positional embeddings from: {key}")
        
        if loaded_token and loaded_pos:
            self._teacher_embeddings_loaded = True
            # Freeze the teacher embeddings
            self.token_embedding.weight.requires_grad = False
            self.positional_embedding.requires_grad = False
            if logger:
                logger.info("✓ Teacher embeddings loaded and frozen successfully")
        else:
            raise ValueError(
                f"Could not find teacher embeddings in checkpoint. "
                f"token_embedding loaded: {loaded_token}, positional_embedding loaded: {loaded_pos}"
            )

    def freeze_teacher_embeddings(self):
        """Ensure teacher embeddings are frozen (call after loading)."""
        self.token_embedding.weight.requires_grad = False
        self.positional_embedding.requires_grad = False

    def forward(self, text, input_boxes=None, device=None):
        # 1. Tokenize (same tokenizer as teacher)
        tokenized = self.tokenizer(text, context_length=self.context_length).to(device)
        seq_len = tokenized.shape[1]
        
        # 2. Teacher embeddings (frozen)
        token_emb = self.token_embedding(tokenized)  # [B, Seq, 1024]
        token_emb = token_emb + self.positional_embedding[:seq_len]  # [B, Seq, 1024]
        
        # 3. Project to student dimension (trainable)
        projected_emb = self.embed_proj(token_emb)  # [B, Seq, student_dim]
        
        # 4. Student transformer (trainable, skip internal embeddings)
        text_memory = self.encoder(projected_emb, return_all_tokens=True, input_is_embeddings=True)
        
        # 5. Output projection (trainable)
        text_memory = self.projector(text_memory)  # [B, Seq, 256]
        
        # 6. Return format compatible with SAM3
        text_attention_mask = (tokenized != 0).bool().ne(1)
        
        # Return tuple: (mask, memory, embeds)
        # memory and embeds are [Seq, Batch, Dim]
        return text_attention_mask, text_memory.transpose(0, 1), projected_emb.transpose(0, 1)
