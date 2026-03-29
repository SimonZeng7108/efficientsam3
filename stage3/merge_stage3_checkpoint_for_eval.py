#!/usr/bin/env python3
"""
Merge a stage3 trainer checkpoint with the initial merged checkpoint to produce
a self-contained evaluation checkpoint.

Stage3 training only saves the encoder weights (backbone.language_backbone.* and
backbone.vision_backbone.*).  The decoder, geometry_encoder, convs, and
dot_prod_scoring weights are frozen and not saved in the trainer checkpoint.  This
script grafts the stage3-trained encoder weights onto the full initial merged
checkpoint so the result can be passed directly to build_efficientsam3_image_model.

Usage:
    python stage3/merge_stage3_checkpoint_for_eval.py \
        --stage3-ckpt  output/stage3/.../checkpoints/checkpoint.pt \
        --base-ckpt    output/efficient_sam3_stage3_es_tv_m_mobileclip_s1_ctx16.pth \
        --output       output/stage3_eval_epoch3.pth
"""
import argparse
import torch


def merge(stage3_ckpt_path: str, base_ckpt_path: str, output_path: str):
    print(f"Loading base checkpoint: {base_ckpt_path}")
    base = torch.load(base_ckpt_path, map_location="cpu", weights_only=True)
    # Base checkpoint may be {'model': state_dict, 'meta': {...}} or a flat state_dict.
    if not isinstance(base, dict):
        raise ValueError("Expected base checkpoint to be a dict.")
    if "model" in base and isinstance(base["model"], dict):
        base_state = base["model"]
        base_meta = base.get("meta", {})
    else:
        base_state = base
        base_meta = {}

    print(f"Loading stage3 checkpoint: {stage3_ckpt_path}")
    s3 = torch.load(stage3_ckpt_path, map_location="cpu", weights_only=True)
    if "model" in s3 and isinstance(s3["model"], dict):
        s3_model = s3["model"]
        epoch = s3.get("epoch", "?")
        steps = s3.get("steps", {})
    else:
        s3_model = s3
        epoch = "?"
        steps = {}

    print(f"  Stage3 checkpoint epoch={epoch}, steps={steps}")
    print(f"  Stage3 model keys: {len(s3_model)}")
    print(f"  Base state_dict keys: {len(base_state)}")

    # Stage3 keys are in the form  backbone.language_backbone.*
    #                          and  backbone.vision_backbone.*
    # Base keys are              detector.backbone.language_backbone.*
    #                        and detector.backbone.vision_backbone.*
    merged_state = dict(base_state)
    overridden = 0
    not_found = []
    for k, v in s3_model.items():
        detector_k = f"detector.{k}"
        if detector_k in merged_state:
            merged_state[detector_k] = v
            overridden += 1
        else:
            not_found.append(k)

    print(f"Overrode {overridden} keys from stage3 checkpoint.")
    if not_found:
        print(f"WARNING: {len(not_found)} stage3 keys had no match in base checkpoint:")
        for k in not_found[:10]:
            print(f"  {k}")
        if len(not_found) > 10:
            print(f"  ... and {len(not_found) - 10} more")

    output = {"model": merged_state, "meta": base_meta}
    print(f"Saving eval checkpoint to: {output_path}")
    torch.save(output, output_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge stage3 encoder ckpt into base merged ckpt for eval")
    parser.add_argument("--stage3-ckpt", required=True, help="Path to stage3 trainer checkpoint (checkpoint.pt)")
    parser.add_argument("--base-ckpt", required=True, help="Path to initial merged checkpoint (.pth)")
    parser.add_argument("--output", required=True, help="Output path for the eval checkpoint (.pth)")
    args = parser.parse_args()
    merge(args.stage3_ckpt, args.base_ckpt, args.output)
