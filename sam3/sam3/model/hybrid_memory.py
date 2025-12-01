# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory import SimpleMaskDownSampler
from .position_encoding import PositionEmbeddingSine

class PerceiverAttention(nn.Module):
    def __init__(
        self, *, dim, dim_head=64, heads=8, dropout_p=0.05, concat_kv_latents=True
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_x = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.dropout_p = dropout_p
        self.concat_kv_latents = concat_kv_latents
        
        # Use EfficientRoPEAttention for the core attention mechanism
        # We wrap it to handle the specific q/k/v projection logic of Perceiver
        # But EfficientRoPEAttention expects q, k, v inputs.
        # So we will use its internal logic or just use F.scaled_dot_product_attention
        # if we don't need the pooling here.
        # WAIT: The whole point is to use pooling here if X is large.
        # But EfficientRoPEAttention does its own projections.
        # Let's use a modified version or call the functional part if possible.
        # Actually, let's just implement the pooling logic directly here for simplicity
        # and consistency with EdgeTAM's structure, but adding the pooling.
        
    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, latents, x, pos=None):
        latents = self.norm_latents(latents)
        x = self.norm_x(x)

        q = self.to_q(latents)

        if self.concat_kv_latents:
            kv_input = torch.cat((x, latents), dim=-2)
        else:
            kv_input = x
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = self._separate_heads(q, self.heads)
        k = self._separate_heads(k, self.heads)
        v = self._separate_heads(v, self.heads)

        if pos is not None:
            assert not self.concat_kv_latents
            pos = self._separate_heads(pos, self.heads)
            k, v = k + pos, v + pos

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        out = self._recombine_heads(out)
        return self.to_out(out)

def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )

class GlobalPerceiver(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_latents=64,
        dim_head=64,
        heads=8,
        depth=2,
        dropout_p=0.05,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            dropout_p=dropout_p,
                        ),
                        FeedForward(dim),
                    ]
                )
            )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, pos=None):
        b = x.shape[0]
        latents = self.latents.repeat(b, 1, 1)

        for attn, ff in self.layers:
            latents = attn(latents, x, pos) + latents
            latents = ff(latents) + latents
            
        return self.norm(latents)


class SpatialPerceiver(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_latents=192,
        dim_head=64,
        heads=8,
        depth=2,
        dropout_p=0.05,
        window_size=8,
    ):
        super().__init__()
        self.window_size = window_size
        self.dim = dim
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(max(1, num_latents), dim))
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PerceiverAttention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            dropout_p=dropout_p,
                            concat_kv_latents=False,
                        ),
                        FeedForward(dim),
                    ]
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def _get_latents(self, B, num_windows, device):
        latents_per_window = max(1, self.num_latents // max(1, num_windows))
        base = self.latents
        if base.size(0) < latents_per_window:
            repeat = math.ceil(latents_per_window / base.size(0))
            base = base.repeat(repeat, 1)
        window_latents = base[:latents_per_window].to(device)
        window_latents = window_latents.unsqueeze(0).expand(num_windows, -1, -1)
        window_latents = window_latents.unsqueeze(0).expand(B, -1, -1, -1)
        return window_latents.contiguous().view(
            B * num_windows, latents_per_window, self.dim
        )

    def forward(self, x, pos=None):
        if x.dim() == 3:
            B, L, C = x.shape
            H = W = int(math.sqrt(L))
            x = x.view(B, H, W, C)
        elif x.dim() == 4 and x.shape[1] == self.dim:
            x = x.permute(0, 2, 3, 1)

        B, H, W, C = x.shape
        ws = self.window_size
        H_pad = math.ceil(H / ws) * ws
        W_pad = math.ceil(W / ws) * ws
        if H_pad != H or W_pad != W:
            pad_h = H_pad - H
            pad_w = W_pad - W
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H, W = H_pad, W_pad

        num_windows_h = H // ws
        num_windows_w = W // ws
        num_windows = num_windows_h * num_windows_w
        windows = (
            x.view(B, num_windows_h, ws, num_windows_w, ws, C)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(B * num_windows, ws * ws, C)
        )

        latents = self._get_latents(B, num_windows, x.device)

        for attn, ff in self.layers:
            latents = attn(latents, windows, pos=None) + latents
            latents = ff(latents) + latents

        latents = latents.view(B, num_windows, -1, C)
        latents = latents.view(B, -1, C)
        return self.norm(latents)

class HybridMemoryModule(nn.Module):
    def __init__(
        self,
        dim=256,
        in_dim=256,
        num_global_latents=64,
        num_spatial_latents=192,
        window_size=8,
        memory_grid: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.num_global_latents = num_global_latents
        self.num_spatial_latents = num_spatial_latents
        self.window_size = window_size

        total_latents = max(1, num_global_latents + num_spatial_latents)
        if memory_grid is None:
            mem_h = window_size
            mem_w = math.ceil(total_latents / mem_h)
        else:
            mem_h, mem_w = memory_grid
        self.mem_height = mem_h
        self.mem_width = mem_w
        self.total_latents = self.mem_height * self.mem_width

        self.mask_downsampler = SimpleMaskDownSampler(
            embed_dim=dim,
            kernel_size=3,
            stride=2,
            padding=1,
            total_stride=16,
            interpol_size=[1152, 1152],
        )
        self.pix_proj = nn.Conv2d(in_dim, dim, kernel_size=1)

        self.global_perceiver = GlobalPerceiver(dim=dim, num_latents=num_global_latents)
        self.spatial_perceiver = SpatialPerceiver(
            dim=dim,
            num_latents=num_spatial_latents,
            window_size=window_size,
        )

        self.pos_encoding = PositionEmbeddingSine(
            num_pos_feats=dim,
            normalize=True,
            scale=None,
            temperature=10000,
            precompute_resolution=max(self.mem_height, self.mem_width) * window_size,
        )

    def _flatten_features(self, x):
        if x.dim() == 4:
            return x.flatten(2).transpose(1, 2)
        return x

    def _project_memory(self, tokens):
        B, N, C = tokens.shape
        if N < self.total_latents:
            pad = self.total_latents - N
            tokens = torch.cat(
                [tokens, tokens.new_zeros(B, pad, C)], dim=1
            )
        tokens = tokens[:, : self.total_latents, :]
        mem = tokens.transpose(1, 2).contiguous().view(
            B, C, self.mem_height, self.mem_width
        )
        return mem

    def forward(self, image, pix_feat, mask):
        del image  # Unused placeholder to match SimpleMaskEncoder signature

        x = self.pix_proj(pix_feat)
        if mask is not None:
            mask_feat = self.mask_downsampler(mask.float())
            x = x + mask_feat

        x_flat = self._flatten_features(x)
        global_latents = self.global_perceiver(x_flat)
        spatial_latents = self.spatial_perceiver(x, pos=None)
        tokens = torch.cat([global_latents, spatial_latents], dim=1)
        mem = self._project_memory(tokens)
        pos = self.pos_encoding(mem).to(mem.dtype)

        return {"vision_features": mem, "vision_pos_enc": [pos]}
