"""
Replace image encoder weights in pretrained merged model with geometry-finetuned weights.

This ensures 100% structure match since we start from the working pretrained model
and only swap out the image encoder portion.

Usage:
    python stage1_geometry_finetune/convert_geometry_finetune.py \
        --finetune-ckpt output_geometry_finetune/es_rv_m/ckpt_epoch_29.pth \
        --pretrained output/efficient_sam3_repvit_m.pt \
        --output output/efficient_sam3_repvit_m_geometry.pt
"""

import argparse
import os
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replace image encoder in pretrained model with finetuned weights"
    )
    parser.add_argument(
        "--finetune-ckpt",
        type=str,
        required=True,
        help="Path to geometry finetune checkpoint (e.g., ckpt_epoch_29.pth)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        required=True,
        help="Path to pretrained merged model (e.g., efficient_sam3_repvit_m.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path. Defaults to <pretrained>_geometry.pt",
    )
    parser.add_argument(
        "--student-prefix",
        type=str,
        default="student_trunk.",
        help="Prefix to strip from finetune checkpoint keys",
    )
    parser.add_argument(
        "--target-prefix",
        type=str,
        default="detector.backbone.vision_backbone.trunk.model.",
        help="Prefix in pretrained model for image encoder weights",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Default output path
    if args.output is None:
        pretrained_path = Path(args.pretrained)
        args.output = str(pretrained_path.with_suffix("")) + "_geometry.pt"

    print(f"Loading finetune checkpoint: {args.finetune_ckpt}")
    finetune_ckpt = torch.load(args.finetune_ckpt, map_location="cpu", weights_only=False)
    if "model" in finetune_ckpt:
        finetune_sd = finetune_ckpt["model"]
    else:
        finetune_sd = finetune_ckpt

    print(f"Loading pretrained model: {args.pretrained}")
    pretrained = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    if "model" in pretrained:
        pretrained_sd = pretrained["model"]
        wrap_in_model = True
    else:
        pretrained_sd = pretrained
        wrap_in_model = False

    # Extract student_trunk keys and build replacement mapping
    student_prefix = args.student_prefix
    target_prefix = args.target_prefix
    
    replacements = {}
    for key, value in finetune_sd.items():
        if key.startswith(student_prefix):
            # Strip student prefix, add target prefix
            stripped_key = key[len(student_prefix):]
            new_key = f"{target_prefix}{stripped_key}"
            replacements[new_key] = value

    print(f"\nStudent encoder keys extracted: {len(replacements)}")

    # Verify all replacement keys exist in pretrained model
    missing_keys = []
    matched_keys = []
    for key in replacements:
        if key in pretrained_sd:
            matched_keys.append(key)
        else:
            missing_keys.append(key)

    print(f"Keys matched in pretrained: {len(matched_keys)}")
    if missing_keys:
        print(f"WARNING: {len(missing_keys)} keys not found in pretrained model!")
        for k in missing_keys[:5]:
            print(f"  Missing: {k}")

    # Check for shape mismatches
    shape_mismatches = []
    for key in matched_keys:
        if replacements[key].shape != pretrained_sd[key].shape:
            shape_mismatches.append((key, pretrained_sd[key].shape, replacements[key].shape))

    if shape_mismatches:
        print(f"\nWARNING: {len(shape_mismatches)} shape mismatches!")
        for key, old_shape, new_shape in shape_mismatches[:5]:
            print(f"  {key}: {old_shape} -> {new_shape}")

    # Replace weights
    replaced_count = 0
    for key, value in replacements.items():
        if key in pretrained_sd:
            pretrained_sd[key] = value
            replaced_count += 1

    print(f"\nReplaced {replaced_count} weights in pretrained model")

    # Verify the replacement by comparing a few weights
    print("\n=== Verification ===")
    sample_keys = list(replacements.keys())[:3]
    all_match = True
    for key in sample_keys:
        if key in pretrained_sd:
            match = torch.equal(pretrained_sd[key], replacements[key])
            status = "✓" if match else "✗"
            print(f"  {status} {key.split('.')[-2]}.{key.split('.')[-1]}")
            if not match:
                all_match = False

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if wrap_in_model:
        torch.save({"model": pretrained_sd}, args.output)
    else:
        torch.save(pretrained_sd, args.output)

    print(f"\n{'='*60}")
    print(f"Output saved to: {args.output}")
    print(f"Total keys in output: {len(pretrained_sd)}")
    print(f"Image encoder weights replaced: {replaced_count}")
    if all_match and replaced_count == len(replacements):
        print("✓ All weights successfully replaced!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
