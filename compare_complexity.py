import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, flop_count_table
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from sam3.model.memory import SimpleMaskEncoder
from sam3.model.hybrid_memory import HybridMemoryModule

def compare_complexity():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Define Inputs
    # Batch size 1, 256 channels, 72x72 resolution (feature map size for 1008x1008 input)
    B, C, H, W = 1, 256, 72, 72
    
    # Input for Memory Encoder: Image Features + Mask
    # SimpleMaskEncoder takes (pix_feat, mask)
    # pix_feat: [B, C, H, W]
    # mask: [B, 1, H*4, W*4] usually? Or H, W?
    # SimpleMaskDownSampler interpolates to interpol_size (1152) then downsamples by 16 -> 72.
    # So mask input size doesn't strictly matter as long as it's reasonable, but let's use 256x256 (standard SAM mask)
    # or 1008x1008 (image size).
    # Let's use 256x256 as placeholder.
    pix_feat = torch.randn(B, C, H, W).to(device)
    mask = torch.randn(B, 1, 256, 256).to(device)
    
    # 2. SAM3 Memory Encoder (Baseline)
    print("\n--- SAM3 Memory Encoder (Baseline) ---")
    # Recreate the exact config from model_builder.py
    from sam3.model.position_encoding import PositionEmbeddingSine
    from sam3.model.memory import SimpleMaskDownSampler, CXBlock, SimpleFuser
    
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=64,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=1008,
    )

    mask_downsampler = SimpleMaskDownSampler(
        kernel_size=3, stride=2, padding=1, interpol_size=[1152, 1152]
    )

    cx_block_layer = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )

    fuser = SimpleFuser(layer=cx_block_layer, num_layers=2)

    sam3_mem_enc = SimpleMaskEncoder(
        out_dim=64,
        position_encoding=position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
        in_dim=256
    ).to(device)
    
    # We need to wrap it to handle the forward signature for fvcore
    class SAM3Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, pix_feat, mask):
            return self.model(pix_feat, mask, skip_mask_sigmoid=True)
            
    sam3_wrapper = SAM3Wrapper(sam3_mem_enc)
    
    # FLOPs for SAM3
    flops_sam3 = FlopCountAnalysis(sam3_wrapper, (pix_feat, mask))
    print(flop_count_table(flops_sam3))
    
    # 3. EfficientSAM3 Hybrid Memory (Ours)
    print("\n--- EfficientSAM3 Hybrid Memory (Ours) ---")
    # HybridMemoryModule takes (x, pos)
    # x: [B, C, H, W] (fused features or just features?)
    # In our implementation, we use it as a replacement for the memory encoder?
    # Or as a replacement for the Memory Attention?
    # Wait, HybridMemoryModule is a "Perceiver" that compresses memory.
    # In `EfficientSam3Tracker`, we use it as `maskmem_backbone`.
    # So it should take the same inputs?
    # `_encode_new_memory` calls `self.memory_encoder(pix_feat, mask_for_mem)`
    # But `HybridMemoryModule.forward` takes `(x, pos)`.
    # We need to adapt HybridMemoryModule to take mask?
    # Or does it assume x is already fused?
    
    # In `sam3_tracker_base.py`:
    # maskmem_out = self.memory_encoder(pix_feat, mask_for_mem)
    
    # Our `HybridMemoryModule` in `hybrid_memory.py` currently has `forward(self, x, pos=None)`.
    # It doesn't take a mask!
    # This means our implementation of `HybridMemoryModule` is incomplete or intended for a different step.
    # EdgeTAM's memory encoder usually fuses mask and image first.
    # Let's assume we fuse them before passing to HybridMemoryModule for this comparison.
    # Or we should update HybridMemoryModule to take mask.
    
    # For fair comparison, let's assume the input `x` to HybridMemoryModule is the concatenation/fusion of image and mask.
    # But SimpleMaskEncoder does the fusion internally.
    
    # Let's instantiate HybridMemoryModule
    hybrid_mem = HybridMemoryModule(
        dim=256,
        num_global_latents=64,
        num_spatial_latents=243, # 81 windows * 3 latents
        window_size=8
    ).to(device)
    
    # Wrapper for HybridMemory
    class HybridWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            return self.model(x)
            
    # Input for Hybrid: [B, C, H, W]
    # We assume mask is already fused or we just pass image features to see the cost of the perceiver itself.
    # The Perceiver is the heavy part.
    hybrid_wrapper = HybridWrapper(hybrid_mem)
    
    flops_hybrid = FlopCountAnalysis(hybrid_wrapper, (pix_feat,))
    print(flop_count_table(flops_hybrid))
    
    print("\n--- Comparison ---")
    print(f"SAM3 FLOPs: {flops_sam3.total() / 1e9:.4f} GFLOPs")
    print(f"Hybrid FLOPs: {flops_hybrid.total() / 1e9:.4f} GFLOPs")
    print(f"Reduction: {(1 - flops_hybrid.total() / flops_sam3.total()) * 100:.2f}%")

if __name__ == "__main__":
    compare_complexity()
