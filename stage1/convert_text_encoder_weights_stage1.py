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
        "Combine Stage-1 text student encoder with a full SAM3 checkpoint",
        add_help=True,
    )
    parser.add_argument(
        "--student-ckpt",
        type=str,
        required=True,
        help="Path to the Stage-1 text student checkpoint (expects 'model' key).",
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
        help="Destination checkpoint. Defaults to <student>_sam3.pth",
    )
    parser.add_argument(
        "--target-prefix",
        type=str,
        default="detector.backbone.language_backbone.",
        help="Prefix to prepend to every student weight before merging.",
    )
    parser.add_argument(
        "--replace-prefix",
        type=str,
        default=None,
        help=(
            "Teacher weights that share this prefix will be dropped so the "
            "student encoder fully replaces them. Defaults to --target-prefix."
        ),
    )
    parser.add_argument(
        "--skip-teacher-prefix",
        type=str,
        action="append",
        default=[],
        help="Additional teacher prefixes to skip when copying the SAM3 weights.",
    )
    parser.add_argument(
        "--use-teacher-embed",
        action="store_true",
        help=(
            "If set, indicates the student uses teacher embeddings (TextStudentEncoderWithTeacherEmbed). "
            "This will preserve teacher embedding weights in the merged checkpoint. "
            "Auto-detected if not specified."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    student_path = Path(args.student_ckpt)
    if args.output is None:
        if student_path.suffix:
            base = student_path.with_suffix("")
        else:
            base = student_path
        args.output = str(base) + "_sam3.pth"

    student_sd = _load_state_dict(args.student_ckpt)
    teacher_sd = _load_state_dict(args.sam3_ckpt)

    # Auto-detect if student uses teacher embeddings
    uses_teacher_embed = args.use_teacher_embed or detect_teacher_embed_model(student_sd)
    if uses_teacher_embed:
        print("Detected: Student uses teacher embeddings (TextStudentEncoderWithTeacherEmbed)")
        print("  - Teacher token_embedding and positional_embedding will be preserved")
        print("  - Student embed_proj, encoder, and projector will be merged")

    prefix = args.target_prefix.strip(".")
    prefix = f"{prefix}." if prefix else ""
    replace_prefix = (
        args.replace_prefix.strip(".")
        if args.replace_prefix is not None
        else args.target_prefix.strip(".")
    )
    replace_prefix = f"{replace_prefix}." if replace_prefix else ""
    skip_prefixes = [
        p.strip(".") + "." for p in args.skip_teacher_prefix if p is not None
    ]

    merged = {}
    
    # If using teacher embeddings, we need special handling
    if uses_teacher_embed:
        # First, copy teacher embedding weights that the student expects
        teacher_embed_keys = [
            "detector.backbone.language_backbone.token_embedding.weight",
            "detector.backbone.language_backbone.positional_embedding",
        ]
        
        # Look for teacher embeddings in various possible locations
        for key, value in teacher_sd.items():
            if 'language_backbone.encoder.token_embedding.weight' in key:
                new_key = f"{prefix}token_embedding.weight"
                merged[new_key] = value
                print(f"  Copied teacher: {key} -> {new_key}")
            elif 'language_backbone.encoder.positional_embedding' in key:
                new_key = f"{prefix}positional_embedding"
                merged[new_key] = value
                print(f"  Copied teacher: {key} -> {new_key}")
    
    # Copy student weights
    for key, value in student_sd.items():
        # Skip teacher embedding weights from student checkpoint if present
        # (they should come from the teacher checkpoint above)
        if uses_teacher_embed and key in ['token_embedding.weight', 'positional_embedding']:
            print(f"  Skipping student embedding (using teacher's): {key}")
            continue
        
        # Student keys might be "encoder.token_embedding.weight"
        # We prepend prefix
        merged_key = f"{prefix}{key}" if prefix else key
        merged[merged_key] = value

    skipped = 0
    replaced = 0
    appended = 0
    for key, value in teacher_sd.items():
        if replace_prefix and key.startswith(replace_prefix):
            replaced += 1
            continue
        if any(key.startswith(p) for p in skip_prefixes):
            skipped += 1
            continue
        if key in merged:
            # Prefer the student copy
            skipped += 1
            continue
        merged[key] = value
        appended += 1

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({"model": merged}, args.output)
    print(f"\nStudent parameters copied: {len(student_sd)} -> prefix '{prefix}'")
    if uses_teacher_embed:
        print(f"Teacher embeddings preserved: token_embedding, positional_embedding")
    print(
        f"Teacher params skipped={skipped}, replaced={replaced}, appended={appended}"
    )
    print(f"Combined checkpoint saved to: {args.output}")


if __name__ == "__main__":
    main()
