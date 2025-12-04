"""
Convert Stage 1 Geometry Finetuned weights by merging with SAM3 text encoder.

This script takes a finetuned checkpoint (which may lack the text encoder if 
ENABLE_TEXT_ENCODER was False during training) and merges it with the text 
encoder from the full SAM3 checkpoint, producing a complete EfficientSAM3 model.

Usage:
    python stage1_geometry_finetune/convert_finetuned_weights.py \
        --finetuned-ckpt output/stage1_geometry_finetune/es_rv_m/ckpt_epoch_9.pth \
        --sam3-ckpt sam3_checkpoints/sam3.pt \
        --output output/efficient_sam3_repvit_m_finetuned.pt
"""

import argparse
import os
from pathlib import Path

import torch


def _torch_load(path, map_location="cpu", **kwargs):
    try:
        return torch.load(path, map_location=map_location, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, map_location=map_location, **kwargs)


def _load_state_dict(path: str):
    """Load state dict from checkpoint file."""
    obj = _torch_load(path, map_location="cpu")
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    if isinstance(obj, dict):
        for key in ("model", "state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        tensors_only = all(isinstance(v, torch.Tensor) for v in obj.values())
        if tensors_only:
            return obj
    raise ValueError(f"Unable to extract a state_dict from {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge Stage 1 Geometry Finetuned weights with SAM3 text encoder",
        add_help=True,
    )
    parser.add_argument(
        "--finetuned-ckpt",
        type=str,
        required=True,
        help="Path to the finetuned checkpoint (e.g., ckpt_epoch_9.pth).",
    )
    parser.add_argument(
        "--sam3-ckpt",
        type=str,
        default="sam3_checkpoints/sam3.pt",
        help="Full SAM3 checkpoint that provides the text encoder.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Destination checkpoint. Defaults to <finetuned>_merged.pt",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it exists.",
    )
    parser.add_argument(
        "--prepend-prefix",
        type=str,
        default="detector.",
        help="Prefix to prepend to finetuned checkpoint keys if not present (e.g. 'detector.').",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine output path
    finetuned_path = Path(args.finetuned_ckpt)
    if args.output is None:
        if finetuned_path.suffix:
            base = finetuned_path.with_suffix("")
        else:
            base = finetuned_path
        args.output = str(base) + "_merged.pt"
    
    # Check if output exists
    if os.path.exists(args.output) and not args.force:
        print(f"Output file already exists: {args.output}")
        print("Use --force to overwrite.")
        return
    
    print(f"Loading finetuned checkpoint from: {args.finetuned_ckpt}")
    finetuned_sd = _load_state_dict(args.finetuned_ckpt)
    
    print(f"Loading SAM3 checkpoint from: {args.sam3_ckpt}")
    sam3_sd = _load_state_dict(args.sam3_ckpt)
    
    # Prepare merged dictionary
    merged = {}
    
    # Process finetuned weights
    prefix = args.prepend_prefix
    renamed_count = 0
    
    for key, value in finetuned_sd.items():
        if not key.startswith(prefix):
            new_key = prefix + key
            merged[new_key] = value
            renamed_count += 1
        else:
            merged[key] = value
            
    if renamed_count > 0:
        print(f"✓ Prepended '{prefix}' to {renamed_count} keys from finetuned checkpoint.")
    else:
        print(f"✓ Finetuned keys already have prefix '{prefix}' (or prefix is empty).")

    # Text encoder prefix in SAM3 checkpoint
    # Based on inspection, it seems to be 'detector.backbone.language_backbone.'
    text_encoder_prefix = "detector.backbone.language_backbone."
    
    # Check if we already have text encoder (after renaming)
    has_text_encoder = any(k.startswith(text_encoder_prefix) for k in merged.keys())
    
    text_encoder_copied = 0
    if has_text_encoder:
        print("⚠️  Finetuned checkpoint already contains text encoder weights!")
        print("    This checkpoint may have been trained with ENABLE_TEXT_ENCODER: True")
        print("    No merging needed.")
    else:
        print("✓ Finetuned checkpoint does not have text encoder (expected).")
        print("  Merging text encoder from SAM3...")
        
        for key, value in sam3_sd.items():
            if key.startswith(text_encoder_prefix):
                merged[key] = value
                text_encoder_copied += 1
    
    # Save merged checkpoint
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({"model": merged}, args.output)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Summary:")
    print(f"  Finetuned parameters: {len(finetuned_sd)}")
    print(f"  Renamed parameters: {renamed_count}")
    print(f"  Text encoder parameters added: {text_encoder_copied}")
    print(f"  Total merged parameters: {len(merged)}")
    print(f"  Output saved to: {args.output}")
    print(f"{'='*70}")
    
    if not has_text_encoder and text_encoder_copied > 0:
        print("\n✓ Success! Your finetuned model now includes the text encoder.")
        print("  This checkpoint is ready to use without further conversion.")
    elif has_text_encoder:
        print("\n✓ Checkpoint copied (already complete).")
    else:
        print("\n⚠️  Warning: No text encoder found in SAM3 checkpoint!")
        print("    Please verify your SAM3 checkpoint path.")


if __name__ == "__main__":
    main()
