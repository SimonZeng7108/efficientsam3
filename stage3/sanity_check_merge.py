"""
Stage 3 — Sanity check for the merge + model-build pipeline.

This script:
1. Inspects stage 1 checkpoint key structures (image + text students).
2. Runs the merge and inspects the resulting key structure.
3. Builds the EfficientSAM3 model and loads the merged checkpoint.
4. Optionally compares encoder outputs between teacher SAM3 and student.

Usage (CPU-only, no GPU required):
    PYTHONPATH=sam3:. python stage3/sanity_check_merge.py
"""

import argparse
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sam3"))
sys.path.insert(0, ROOT)


def _torch_load(path, **kw):
    kw.setdefault("map_location", "cpu")
    try:
        return torch.load(path, weights_only=False, **kw)
    except TypeError:
        return torch.load(path, **kw)


def _extract_sd(obj):
    if isinstance(obj, dict):
        for k in ("model", "state_dict"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise ValueError("Cannot extract state_dict")


def _prefix_tree(keys, depth=3):
    """Group keys by their first *depth* dotted components."""
    tree = defaultdict(int)
    for k in keys:
        parts = k.split(".")
        prefix = ".".join(parts[:depth])
        tree[prefix] += 1
    return dict(sorted(tree.items()))


# ---------------------------------------------------------------------------
# Step 1: Inspect stage 1 checkpoints
# ---------------------------------------------------------------------------
def inspect_checkpoint(path, label):
    print(f"\n{'='*80}")
    print(f"  {label}: {path}")
    print(f"{'='*80}")
    raw = _torch_load(path)
    if isinstance(raw, dict):
        top_keys = list(raw.keys())
        print(f"  Top-level keys: {top_keys}")

    sd = _extract_sd(raw)
    print(f"  State dict entries: {len(sd)}")
    total_params = sum(v.numel() for v in sd.values())
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"\n  Key prefix tree (depth=3):")
    for prefix, count in _prefix_tree(sd.keys(), depth=3).items():
        print(f"    {prefix:60s}  ({count} keys)")
    print(f"\n  First 20 keys:")
    for i, k in enumerate(sorted(sd.keys())):
        if i >= 20:
            print(f"    ... ({len(sd) - 20} more)")
            break
        print(f"    {k:70s}  shape={list(sd[k].shape)}")
    return sd


# ---------------------------------------------------------------------------
# Step 2: Run merge and inspect
# ---------------------------------------------------------------------------
def run_merge(image_ckpt, text_ckpt, sam3_ckpt, output_path):
    print(f"\n{'='*80}")
    print(f"  Running merge → {output_path}")
    print(f"{'='*80}")

    from stage1.convert_both_encoders_weights_stage1 import (
        _load_state_dict,
        _normalize_image_student_key,
        _normalize_text_student_key,
    )

    image_sd = _load_state_dict(image_ckpt)
    text_sd = _load_state_dict(text_ckpt)
    teacher_sd = _load_state_dict(sam3_ckpt)

    image_target_prefix = "detector.backbone.vision_backbone.trunk.model."
    text_target_prefix = "detector.backbone.language_backbone."
    image_replace_prefix = "detector.backbone.vision_backbone.trunk."
    text_replace_prefix = "detector.backbone.language_backbone."

    merged = {}

    # Image student
    for key, value in image_sd.items():
        nk = _normalize_image_student_key(key)
        merged[f"{image_target_prefix}{nk}"] = value

    # Text student
    for key, value in text_sd.items():
        nk = _normalize_text_student_key(key)
        merged[f"{text_target_prefix}{nk}"] = value

    # Teacher (skip replaced backbone parts)
    replaced = 0
    kept = 0
    for key, value in teacher_sd.items():
        if key.startswith(image_replace_prefix) or key.startswith(text_replace_prefix):
            replaced += 1
            continue
        merged[key] = value
        kept += 1

    print(f"  Image student keys merged: {len(image_sd)}")
    print(f"  Text student keys merged:  {len(text_sd)}")
    print(f"  Teacher keys replaced:     {replaced}")
    print(f"  Teacher keys kept:         {kept}")
    print(f"  Total merged keys:         {len(merged)}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save({"model": merged}, output_path)
    print(f"  Saved to {output_path}")
    return merged


# ---------------------------------------------------------------------------
# Step 3: Build model and load merged checkpoint
# ---------------------------------------------------------------------------
def build_and_load(merged_path, bpe_path):
    print(f"\n{'='*80}")
    print(f"  Building EfficientSAM3 model and loading merged checkpoint")
    print(f"{'='*80}")

    from sam3.model_builder import build_efficientsam3_image_model

    model = build_efficientsam3_image_model(
        bpe_path=bpe_path,
        device="cpu",
        eval_mode=True,
        checkpoint_path=merged_path,
        load_from_HF=False,
        enable_segmentation=False,
        enable_inst_interactivity=False,
        backbone_type="tinyvit",
        model_name="11m",
        text_encoder_type="MobileCLIP-S1",
        text_encoder_context_length=16,
        text_encoder_pos_embed_table_size=16,
        interpolate_pos_embed=False,
    )

    model_sd = model.state_dict()
    print(f"  Model state dict entries: {len(model_sd)}")
    total = sum(v.numel() for v in model_sd.values())
    print(f"  Total parameters: {total:,} ({total/1e6:.2f}M)")

    # Check for parameters that were NOT loaded from the merged checkpoint
    merged_sd = _extract_sd(_torch_load(merged_path))

    # The model loads with _load_checkpoint which strips "detector." prefix
    model_keys = set(model_sd.keys())
    loaded_keys = set()
    for mk in merged_sd.keys():
        nk = mk
        if nk.startswith("detector."):
            nk = nk[len("detector."):]
        if "student_trunk." in nk:
            nk = nk.replace("student_trunk.", "")
        if nk in model_keys:
            loaded_keys.add(nk)

    unloaded = model_keys - loaded_keys
    print(f"\n  Model keys matched from merged ckpt: {len(loaded_keys)}")
    print(f"  Model keys NOT in merged ckpt:       {len(unloaded)}")
    if unloaded:
        print(f"  Unloaded keys (first 30):")
        for k in sorted(unloaded)[:30]:
            print(f"    {k}")

    return model


# ---------------------------------------------------------------------------
# Step 4: Compare encoder outputs
# ---------------------------------------------------------------------------
def compare_encoder_outputs(merged_model, sam3_ckpt_path, bpe_path):
    print(f"\n{'='*80}")
    print(f"  Comparing encoder outputs: Student vs Teacher")
    print(f"{'='*80}")

    from sam3.model_builder import build_sam3_image_model

    print("  Building SAM3 teacher model (this may take a moment)...")
    teacher = build_sam3_image_model(
        bpe_path=bpe_path,
        device="cpu",
        eval_mode=True,
        checkpoint_path=sam3_ckpt_path,
        load_from_HF=False,
        enable_segmentation=False,
        enable_inst_interactivity=False,
    )

    merged_model.eval()
    teacher.eval()

    dummy_img = torch.randn(1, 3, 1008, 1008)
    test_texts = ["a dog", "a cat sitting on a chair"]

    with torch.no_grad():
        # Vision encoder
        print("\n  --- Vision Encoder ---")
        student_vis = merged_model.backbone.forward_image(dummy_img)
        teacher_vis = teacher.backbone.forward_image(dummy_img)

        s_fpn = student_vis[0]
        t_fpn = teacher_vis[0]
        print(f"  Student FPN levels: {len(s_fpn)}")
        print(f"  Teacher FPN levels: {len(t_fpn)}")
        for i, (sf, tf) in enumerate(zip(s_fpn, t_fpn)):
            print(f"    Level {i}: student={list(sf.shape)}, teacher={list(tf.shape)}")
            if sf.shape == tf.shape:
                diff = (sf - tf).abs().mean().item()
                cos = F.cosine_similarity(sf.flatten(), tf.flatten(), dim=0).item()
                print(f"             diff={diff:.6f}, cosine={cos:.6f}")

        # Text encoder
        print("\n  --- Text Encoder ---")
        student_txt = merged_model.backbone.forward_text(test_texts, device="cpu")
        teacher_txt = teacher.backbone.forward_text(test_texts, device="cpu")

        s_mask, s_mem, s_emb = student_txt
        t_mask, t_mem, t_emb = teacher_txt
        print(f"  Student text memory: {list(s_mem.shape)}")
        print(f"  Teacher text memory: {list(t_mem.shape)}")
        print(f"  Student text mask:   {list(s_mask.shape)}")
        print(f"  Teacher text mask:   {list(t_mask.shape)}")

        # Compare overlapping sequence positions
        min_seq = min(s_mem.shape[0], t_mem.shape[0])
        s_sub = s_mem[:min_seq]
        t_sub = t_mem[:min_seq]
        diff = (s_sub - t_sub).abs().mean().item()
        cos = F.cosine_similarity(
            s_sub.reshape(-1), t_sub.reshape(-1), dim=0
        ).item()
        print(f"  Text memory diff (first {min_seq} tokens): {diff:.6f}")
        print(f"  Text memory cosine (first {min_seq} tokens): {cos:.6f}")
        print(f"  NOTE: Student uses distilled MobileCLIP-S1, teacher uses full SAM3 text encoder.")
        print(f"        Non-trivial differences are expected. Cosine > 0.3 indicates useful alignment.")


def parse_args():
    parser = argparse.ArgumentParser("Stage 3 Merge Sanity Check")
    parser.add_argument(
        "--image-ckpt",
        default="output/stage1_image_2p/es_tv_m/ckpt_epoch_49.pth",
    )
    parser.add_argument(
        "--text-ckpt",
        default="output/stage1_text/mobileclip_s1_5dataset_ctx16_fixed/ckpt_epoch_79.pth",
    )
    parser.add_argument(
        "--sam3-ckpt",
        default="sam3_checkpoints/sam3.pt",
    )
    parser.add_argument(
        "--output",
        default="output/efficient_sam3_stage3_es_tv_m_mobileclip_s1_ctx16.pth",
    )
    parser.add_argument(
        "--bpe-path",
        default="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
    )
    parser.add_argument(
        "--skip-compare",
        action="store_true",
        help="Skip the teacher-vs-student comparison (saves memory).",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip merge step and use existing merged checkpoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Step 1: Inspect stage 1 checkpoints
    image_sd = inspect_checkpoint(args.image_ckpt, "Stage 1 Image Student (TinyViT-11M)")
    text_sd = inspect_checkpoint(args.text_ckpt, "Stage 1 Text Student (MobileCLIP-S1)")
    inspect_checkpoint(args.sam3_ckpt, "SAM3 Teacher")

    # Step 2: Run merge
    if args.skip_merge and os.path.exists(args.output):
        print(f"\n  Skipping merge, using existing: {args.output}")
        merged_sd = _extract_sd(_torch_load(args.output))
    else:
        merged_sd = run_merge(
            args.image_ckpt, args.text_ckpt, args.sam3_ckpt, args.output
        )

    # Step 3: Build model and load
    model = build_and_load(args.output, args.bpe_path)

    # Step 4: Compare encoder outputs
    if not args.skip_compare:
        compare_encoder_outputs(model, args.sam3_ckpt, args.bpe_path)
    else:
        print("\n  Skipping teacher-vs-student comparison.")

    print(f"\n{'='*80}")
    print(f"  Sanity check complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
