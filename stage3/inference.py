"""Stage 3 Inference Wrapper.

Loads either a Stage 1 merged checkpoint OR a Stage 3 fine-tuned checkpoint
and runs text + geometry prompted segmentation through the same EfficientSAM3
model that was used at training time.

Usage:
    python stage3/inference.py \
        --checkpoint output_stage3/es_tv_m/ckpt_epoch_latest.pth \
        --backbone tinyvit_11m \
        --image test.jpg \
        --text "the red car" \
        --box 100,200,300,400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


BACKBONE_MAP = {
    "repvit_m0_9": ("repvit", "m0.9"),
    "repvit_m1_1": ("repvit", "m1.1"),
    "repvit_m2_3": ("repvit", "m2.3"),
    "tinyvit_5m": ("tinyvit", "5m"),
    "tinyvit_11m": ("tinyvit", "11m"),
    "tinyvit_21m": ("tinyvit", "21m"),
    "efficientvit_b0": ("efficientvit", "b0"),
    "efficientvit_b1": ("efficientvit", "b1"),
    "efficientvit_b2": ("efficientvit", "b2"),
}


def _strip_wrapper_prefix(state_dict):
    """Strip the ``model.`` prefix added by the Stage 3 training wrapper.

    A Stage 3 checkpoint's state dict keys look like ``model.backbone....``
    because ``Stage3FinetuneModel`` holds the EfficientSAM3 as ``self.model``.
    We strip that prefix so the keys match ``build_efficientsam3_image_model``'s
    output directly.
    """
    out = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("model."):
            new_k = new_k[len("model."):]
        out[new_k] = v
    return out


def _load_stage3_into_model(model, checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    else:
        state = ckpt
    state = _strip_wrapper_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"[inference] Loaded Stage 3 checkpoint: "
        f"{len(missing)} missing / {len(unexpected)} unexpected keys"
    )
    if missing:
        print(f"  first missing: {missing[:5]}")
    if unexpected:
        print(f"  first unexpected: {unexpected[:5]}")


def main():
    parser = argparse.ArgumentParser("Stage 3 Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--box", type=str, default=None, help="x1,y1,x2,y2")
    parser.add_argument("--point", type=str, default=None, help="x,y")
    parser.add_argument("--output", type=str, default="stage3_output.png")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backbone", type=str, default="tinyvit_11m",
                        choices=list(BACKBONE_MAP.keys()))
    parser.add_argument("--text-encoder", type=str, default="MobileCLIP-S0")
    parser.add_argument("--stage1-ckpt", action="store_true",
                        help="If set, load checkpoint directly via the model "
                             "builder (expects Stage 1 merged format).")
    args = parser.parse_args()

    backbone_type, model_name = BACKBONE_MAP[args.backbone]

    from sam3.model_builder import build_efficientsam3_image_model

    model = build_efficientsam3_image_model(
        backbone_type=backbone_type,
        model_name=model_name,
        checkpoint_path=args.checkpoint if args.stage1_ckpt else None,
        eval_mode=True,
        device=args.device,
        enable_segmentation=True,
        text_encoder_type=args.text_encoder,
    )
    if not args.stage1_ckpt:
        _load_stage3_into_model(model, args.checkpoint)
        model.to(args.device).eval()

    from sam3.model.sam3_image_processor import Sam3Processor
    processor = Sam3Processor(model)

    img = Image.open(args.image).convert("RGB")
    processor.set_image(img)

    if args.text:
        processor.set_text_prompt([args.text])

    if args.box:
        coords = list(map(float, args.box.split(",")))
        processor.set_box_prompt(np.array([coords]))

    if args.point:
        coords = list(map(float, args.point.split(",")))
        processor.set_point_prompt(
            np.array([[coords]]),
            np.array([[1]]),
        )

    results = processor.get_outputs()

    if "pred_masks" in results:
        masks = results["pred_masks"]
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        best_mask = (masks[0, 0] > 0).astype(np.uint8) * 255
        out_img = Image.fromarray(best_mask)
        out_img.save(args.output)
        print(f"Saved mask to {args.output}")
    else:
        print("No masks in output. Available keys:", list(results.keys()))


if __name__ == "__main__":
    main()
