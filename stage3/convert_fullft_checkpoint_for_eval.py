#!/usr/bin/env python3
"""
Convert a full-finetune Stage 3 trainer checkpoint into an evaluation checkpoint.

Full-FT checkpoints save the entire model state dict (no skip_saving_parameters).
The eval builder (build_efficientsam3_image_model) expects keys with a 'detector.'
prefix.  This script adds that prefix so the checkpoint loads correctly.

Usage:
    python stage3/convert_fullft_checkpoint_for_eval.py \
        --trainer-ckpt  output/stage3/.../checkpoints/checkpoint.pt \
        --output        output/stage3_seg_finetuned_eval.pth
"""
import argparse
import torch


def convert(trainer_ckpt_path: str, output_path: str):
    print(f"Loading trainer checkpoint: {trainer_ckpt_path}")
    ckpt = torch.load(trainer_ckpt_path, map_location="cpu", weights_only=True)

    if "model" in ckpt and isinstance(ckpt["model"], dict):
        model_state = ckpt["model"]
        epoch = ckpt.get("epoch", "?")
        steps = ckpt.get("steps", {})
    else:
        model_state = ckpt
        epoch = "?"
        steps = {}

    print(f"  Epoch={epoch}, steps={steps}")
    print(f"  Model keys: {len(model_state)}")

    eval_state = {}
    for k, v in model_state.items():
        if k.startswith("detector."):
            eval_state[k] = v
        else:
            eval_state[f"detector.{k}"] = v

    print(f"  Eval checkpoint keys: {len(eval_state)}")

    sample_keys = list(eval_state.keys())[:5]
    for k in sample_keys:
        print(f"    {k}: {eval_state[k].shape}")

    output = {"model": eval_state, "meta": {"epoch": epoch, "source": trainer_ckpt_path}}
    print(f"Saving eval checkpoint to: {output_path}")
    torch.save(output, output_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert full-FT Stage 3 checkpoint to eval format"
    )
    parser.add_argument(
        "--trainer-ckpt", required=True,
        help="Path to trainer checkpoint (checkpoint.pt or checkpoint_N.pt)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for the eval checkpoint (.pth)"
    )
    args = parser.parse_args()
    convert(args.trainer_ckpt, args.output)
