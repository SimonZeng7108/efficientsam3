import argparse
import os
from pathlib import Path
import torch


def _strip_prefix(key: str, prefix: str) -> str:
    return key[len(prefix):] if key.startswith(prefix) else key


def _normalize_image_student_key(key: str) -> str:
    key = _strip_prefix(key, "module.")
    key = _strip_prefix(key, "student_trunk.")

    # If someone passes an already-merged checkpoint, collapse to local student key.
    key = _strip_prefix(key, "detector.backbone.vision_backbone.trunk.model.")
    key = _strip_prefix(key, "detector.backbone.vision_backbone.trunk.")
    key = _strip_prefix(key, "backbone.vision_backbone.trunk.model.")
    key = _strip_prefix(key, "backbone.vision_backbone.trunk.")
    return key


def _normalize_text_student_key(key: str) -> str:
    key = _strip_prefix(key, "module.")

    # If someone passes an already-merged checkpoint, collapse to local student key.
    key = _strip_prefix(key, "detector.backbone.language_backbone.")
    key = _strip_prefix(key, "backbone.language_backbone.")
    return key

def _torch_load(path, map_location="cpu", **kwargs):
    try:
        return torch.load(path, map_location=map_location, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, map_location=map_location, **kwargs)

def _load_state_dict(path: str):
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
        "Combine Stage-1 student image and text encoders with a full SAM3 checkpoint",
        add_help=True,
    )
    parser.add_argument(
        "--image-student-ckpt",
        type=str,
        required=True,
        help="Path to the Stage-1 image student checkpoint.",
    )
    parser.add_argument(
        "--text-student-ckpt",
        type=str,
        required=True,
        help="Path to the Stage-1 text student checkpoint.",
    )
    parser.add_argument(
        "--sam3-ckpt",
        type=str,
        default="sam3_checkpoints/sam3.pt",
        help="Full SAM3 checkpoint that provides prompt encoder, decoder, etc.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Destination checkpoint. If not provided, constructed from model names.",
    )
    parser.add_argument(
        "--image-model-name",
        type=str,
        default="efficientvit_b0",
        help="Name of the image student model (used for filename).",
    )
    parser.add_argument(
        "--text-model-name",
        type=str,
        default="mobileclip_s0",
        help="Name of the text student model (used for filename).",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    if args.output is None:
        args.output = f"output/efficient_sam3_{args.image_model_name}_{args.text_model_name}.pth"
    
    print(f"Loading Image Student: {args.image_student_ckpt}")
    image_sd = _load_state_dict(args.image_student_ckpt)
    
    print(f"Loading Text Student: {args.text_student_ckpt}")
    text_sd = _load_state_dict(args.text_student_ckpt)
    
    print(f"Loading SAM3 Teacher: {args.sam3_ckpt}")
    teacher_sd = _load_state_dict(args.sam3_ckpt)

    merged = {}
    
    # Prefixes
    image_target_prefix = "detector.backbone.vision_backbone.trunk.model."
    text_target_prefix = "detector.backbone.language_backbone."

    # Teacher keys to replace completely
    image_replace_prefix = "detector.backbone.vision_backbone.trunk."
    text_replace_prefix = "detector.backbone.language_backbone."
    
    # 1. Add Image Student Weights
    print(f"Merging Image Student weights (target prefix: {image_target_prefix})...")
    for key, value in image_sd.items():
        key = _normalize_image_student_key(key)
        merged_key = f"{image_target_prefix}{key}"
        merged[merged_key] = value

    # 2. Add Text Student Weights
    print(f"Merging Text Student weights (target prefix: {text_target_prefix})...")
    for key, value in text_sd.items():
        key = _normalize_text_student_key(key)
        merged_key = f"{text_target_prefix}{key}"
        merged[merged_key] = value

    # 3. Add Teacher Weights (skipping those replaced)
    print("Merging Teacher weights...")
    skipped = 0
    replaced = 0
    appended = 0
    
    for key, value in teacher_sd.items():
        # Check if this key belongs to image or text backbone
        if key.startswith(image_replace_prefix):
            replaced += 1
            continue
        if key.startswith(text_replace_prefix):
            replaced += 1
            continue
            
        # If not replaced, keep it
        merged[key] = value
        appended += 1

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({"model": merged}, args.output)
    print(f"Combined checkpoint saved to: {args.output}")
    print(f"Teacher params: Replaced={replaced}, Skipped={skipped}, Appended={appended}")

if __name__ == "__main__":
    main()
