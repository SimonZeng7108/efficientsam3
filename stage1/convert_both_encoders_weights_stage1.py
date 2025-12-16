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


def detect_teacher_embed_model(student_sd):
    """Detect if the student model uses teacher embeddings (TextStudentEncoderWithTeacherEmbed)."""
    # Check for the embed_proj layer which only exists in TextStudentEncoderWithTeacherEmbed
    return any('embed_proj' in key for key in student_sd.keys())


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
    parser.add_argument(
        "--use-teacher-embed",
        action="store_true",
        help=(
            "If set, indicates the text student uses teacher embeddings (TextStudentEncoderWithTeacherEmbed). "
            "This will preserve teacher embedding weights in the merged checkpoint. "
            "Auto-detected if not specified."
        ),
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

    # Auto-detect if text student uses teacher embeddings
    uses_teacher_embed = args.use_teacher_embed or detect_teacher_embed_model(text_sd)
    if uses_teacher_embed:
        print("Detected: Text student uses teacher embeddings (TextStudentEncoderWithTeacherEmbed)")
        print("  - Teacher token_embedding and positional_embedding will be preserved")
        print("  - Student embed_proj, encoder, and projector will be merged")

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

    # 2. Handle Text Student Weights
    print(f"Merging Text Student weights (prefix: {text_prefix})...")
    
    # If using teacher embeddings (Option 3), copy teacher embedding weights to MobileCLIP's expected locations
    # SAM3 teacher has: encoder.token_embedding.weight [vocab, 1024], encoder.positional_embedding [ctx, 1024]
    # MobileCLIP expects: encoder.embedding_layer.weight [vocab, dim], encoder.positional_embedding.pos_embed.pos_embed [1, 1, ctx, dim]
    if uses_teacher_embed:
        for key, value in teacher_sd.items():
            if 'language_backbone.encoder.token_embedding.weight' in key:
                # Map to MobileCLIP's embedding_layer
                new_key = f"{text_prefix}encoder.embedding_layer.weight"
                merged[new_key] = value
                print(f"  Copied teacher: {key} -> {new_key} (shape: {value.shape})")
            elif 'language_backbone.encoder.positional_embedding' in key:
                # Map to MobileCLIP's positional_embedding structure
                # SAM3 has shape [context_length, embed_dim], MobileCLIP expects [1, 1, context_length, embed_dim]
                pos_embed = value
                if pos_embed.dim() == 2:
                    pos_embed = pos_embed.unsqueeze(0).unsqueeze(0)  # [ctx, dim] -> [1, 1, ctx, dim]
                new_key = f"{text_prefix}encoder.positional_embedding.pos_embed.pos_embed"
                merged[new_key] = pos_embed
                print(f"  Copied teacher: {key} -> {new_key} (reshaped: {value.shape} -> {pos_embed.shape})")
    
    # Add text student weights
    for key, value in text_sd.items():
        # Skip teacher embedding weights from student checkpoint if present (Option 3)
        if uses_teacher_embed and key in ['token_embedding.weight', 'positional_embedding']:
            print(f"  Skipping student embedding (using teacher's): {key}")
            continue
        
        # Skip embed_proj for teacher-embed models (it's only used during training, not inference)
        if uses_teacher_embed and key.startswith('embed_proj'):
            print(f"  Skipping training-only layer: {key}")
            continue
        
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
        
        # Skip if already in merged (e.g., from student weights)
        if key in merged:
            skipped += 1
            continue
            
        # If not replaced, keep it
        merged[key] = value
        appended += 1

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({"model": merged}, args.output)
    print(f"\nCombined checkpoint saved to: {args.output}")
    print(f"Image student params copied: {len(image_sd)}")
    print(f"Text student params copied: {len(text_sd)}")
    if uses_teacher_embed:
        print(f"Teacher embeddings preserved: token_embedding, positional_embedding")
    print(f"Teacher params: Replaced={replaced}, Skipped={skipped}, Appended={appended}")

if __name__ == "__main__":
    main()
