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
    # The student checkpoint has keys like "backbone.model.input_stem..."
    # The model expects keys like "backbone.vision_backbone.trunk.model.backbone.model.input_stem..."
    # So we need to prepend "detector.backbone.vision_backbone.trunk.model."
    image_prefix = "detector.backbone.vision_backbone.trunk.model."
    text_prefix = "detector.backbone.language_backbone."
    
    # 1. Add Image Student Weights
    print(f"Merging Image Student weights (prefix: {image_prefix})...")
    for key, value in image_sd.items():
        merged_key = f"{image_prefix}{key}"
        merged[merged_key] = value

    # 2. Add Text Student Weights
    print(f"Merging Text Student weights (prefix: {text_prefix})...")
    for key, value in text_sd.items():
        merged_key = f"{text_prefix}{key}"
        merged[merged_key] = value

    # 3. Add Teacher Weights (skipping those replaced)
    print("Merging Teacher weights...")
    skipped = 0
    replaced = 0
    appended = 0
    
    for key, value in teacher_sd.items():
        # Check if this key belongs to image or text backbone
        if key.startswith(image_prefix):
            replaced += 1
            continue
        if key.startswith(text_prefix):
            replaced += 1
            continue
            
        # If not replaced, keep it
        merged[key] = value
        appended += 1

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({"model": merged}, args.output)
    print(f"Combined checkpoint saved to: {args.output}")
    print(f"Teacher params: Replaced={replaced}, Appended={appended}")

if __name__ == "__main__":
    main()
