#!/usr/bin/env python3

from __future__ import annotations

import argparse

from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import COLORS, plot_bbox, plot_mask


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EfficientSAM3 image inference with a text prompt and print mask count + scores."
    )
    parser.add_argument(
        "--checkpoint",
        default="/output/efficient_sam3_tinyvit_21m_mobileclip_s1.pth",
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--image",
        default="/sam3/assets/images/test_image.jpg",
        help="Path to input image",
    )
    parser.add_argument(
        "--output",
        default="sam3_vis.png",
        help="Path to save visualization image (PNG)",
    )
    parser.add_argument("--prompt", default="person", help="Text prompt")
    parser.add_argument(
        "--backbone-type",
        default="tinyvit",
        choices=["tinyvit", "efficientvit"],
        help="Backbone family",
    )
    parser.add_argument(
        "--model-name",
        default="21m",
        help='Model size/name (e.g. "11m", "21m", "b2")',
    )
    parser.add_argument(
        "--text-encoder-type",
        default="MobileCLIP-S1",
        help='Text encoder type (e.g. "MobileCLIP-S1")',
    )

    args = parser.parse_args()

    model = build_efficientsam3_image_model(
        checkpoint_path=args.checkpoint,
        backbone_type=args.backbone_type,
        model_name=args.model_name,
        text_encoder_type=args.text_encoder_type,
    )

    processor = Sam3Processor(model)

    image = Image.open(args.image).convert("RGB")

    inference_state = processor.set_image(image)
    inference_state = processor.set_text_prompt(args.prompt, inference_state)

    masks = inference_state["masks"]
    scores = inference_state["scores"]
    boxes = inference_state["boxes"]
    # masks: Bool tensor with shape [N, 1, H, W]; scores: tensor with shape [N]
    num_masks = int(masks.shape[0])
    print(num_masks, scores.detach().cpu())

    # Visualization (matches the notebook approach, but saves to disk instead of plt.show())
    fig = plt.figure(figsize=(12, 8))
    plt.imshow(image)
    w, h = image.size
    for i in range(num_masks):
        color = COLORS[i % len(COLORS)]
        plot_mask(masks[i].squeeze(0).cpu(), color=color)
        prob = scores[i].item()
        plot_bbox(
            h,
            w,
            boxes[i].cpu(),
            text=f"(id={i}, prob={prob:.2f})",
            box_format="XYXY",
            color=color,
            relative_coords=False,
        )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    print(f"saved visualization to: {args.output}")


if __name__ == "__main__":
    main()
