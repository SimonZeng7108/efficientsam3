# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial
from typing import Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from sam3.sam.rope import (
    apply_rotary_enc,
    compute_axial_cis,
)

class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(kv_in_dim if kv_in_dim is not None else embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(kv_in_dim if kv_in_dim is not None else embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        self.dropout_p = dropout

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0
        )
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class EfficientRoPEAttention(Attention):
    """Attention with rotary position encoding and EfficientTAM pooling."""

    def __init__(
        self,
        *args,
        rope_theta=10000.0,
        # whether to repeat q rope to match k length
        # this is needed for cross-attention to memories
        rope_k_repeat=False,
        feat_sizes=(32, 32),  # [w, h] for stride 16 feats at 512 resolution
        **kwargs,
    ):
        # Pop arguments that are not supported by the base Attention class
        kwargs.pop('use_fa3', None)
        kwargs.pop('use_rope_real', None)
        
        super().__init__(*args, **kwargs)

        self.compute_cis = partial(
            compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta
        )
        # Initial freqs_cis, will be updated dynamically if needed
        freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.freqs_cis = freqs_cis
        self.rope_k_repeat = rope_k_repeat

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0
    ) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Apply rotary position encoding
        # Assuming k is [B, H, N, C]
        # We need to infer spatial dimensions w, h from N
        # This logic assumes square feature maps or provided feat_sizes
        # For memory attention, N can be large (T*H*W)
        
        # Note: In EfficientTAM, they handle the reshaping carefully.
        # Here we assume the standard SAM2/3 usage where we might need to update cis.
        
        # For simplicity in this implementation, we assume the caller handles 
        # the complex reshaping if N is not square, or we rely on the fact that
        # compute_axial_cis handles 1D sequences if needed or we just use the cached one.
        # However, EfficientTAM's pooling relies on reshaping to 2D.
        
        # Let's check if we need to update freqs_cis
        # q shape: [B, Heads, Nq, D]
        # k shape: [B, Heads, Nk, D]
        
        # Logic from EfficientTAM to handle RoPE
        w = h = int(math.sqrt(q.shape[-2])) # Approximation for square queries
        
        self.freqs_cis = self.freqs_cis.to(q.device)
        # If q is not square or size changed, update cis
        if self.freqs_cis.shape[0] != q.shape[-2]:
             # Fallback or update. For now, let's assume we can update if it looks square
             if w*h == q.shape[-2]:
                 self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope
        
        # Apply RoPE
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )

        dropout_p = self.dropout_p if self.training else 0.0

        # EfficientTAM Pooling Logic
        if self.rope_k_repeat:
            fs, bs, ns, ds = k.shape
            nq = q.shape[-2]
            
            # If memory is small enough, just do normal attention
            if num_k_rope <= nq:
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
            else:
                # Apply pooling to reduce K and V size
                s_kernel_size = 2
                intw, inth = int(w), int(h)
                
                # Reshape K to [Batch*Heads, T, H, W, C] roughly
                # But here k is [Batch, Heads, N, C]
                # We reshape to [Batch*Heads, -1, H, W] to pool spatially
                
                # Reshape for pooling: [B*Heads, T, H, W] -> Pool -> Flatten
                # Note: The EfficientTAM code reshapes based on 'nq' which is the spatial size of one frame usually
                
                k_landmarks = k[:, :, :num_k_rope, :].reshape(fs, -1, nq, ds)
                k_landmarks = k_landmarks.transpose(-2, -1).reshape(fs, -1, intw, inth)
                k_landmarks = F.avg_pool2d(
                    k_landmarks, s_kernel_size, stride=s_kernel_size
                )
                k_landmarks = k_landmarks.reshape(
                    fs, -1, ds, nq // (s_kernel_size * s_kernel_size)
                ).transpose(-2, -1)
                
                # Add log(kernel_size) to keys as per EfficientTAM (likely for scale adjustment)
                k_landmarks = torch.cat(
                    [
                        k_landmarks.reshape(fs, bs, -1, ds)
                        + 2 * math.log(s_kernel_size),
                        k[:, :, num_k_rope:, :],
                    ],
                    dim=-2,
                )

                v_landmarks = v[:, :, :num_k_rope, :].reshape(fs, -1, nq, ds)
                v_landmarks = v_landmarks.transpose(-2, -1).reshape(fs, -1, intw, inth)
                v_landmarks = F.avg_pool2d(
                    v_landmarks, s_kernel_size, stride=s_kernel_size
                )
                v_landmarks = v_landmarks.reshape(
                    fs, -1, ds, nq // (s_kernel_size * s_kernel_size)
                ).transpose(-2, -1)
                v_landmarks = torch.cat(
                    [
                        v_landmarks.reshape(fs, bs, -1, ds),
                        v[:, :, num_k_rope:, :],
                    ],
                    dim=-2,
                )
                
                out = F.scaled_dot_product_attention(
                    q, k_landmarks, v_landmarks, dropout_p=dropout_p
                )
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out
