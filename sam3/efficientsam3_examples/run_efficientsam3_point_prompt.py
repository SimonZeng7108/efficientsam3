#!/usr/bin/env python3

from __future__ import annotations

import argparse

import numpy as np
from PIL import Image

from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal EfficientSAM3 point-prompt demo (predict_inst)."
    )
    parser.add_argument(
        "--checkpoint",
        default="/output/efficient_sam3_efficientvit_m.pt",
        help="Path to EfficientSAM3 checkpoint (.pt/.pth)",
    )
    parser.add_argument(
        "--image",
        default="/sam3/assets/images/test_image.jpg",
        help="Path to input image",
    )
    parser.add_argument(
        "--x",
        type=float,
        default=None,
        help="Point x in pixels (default: image center)",
    )
    parser.add_argument(
        "--y",
        type=float,
        default=None,
        help="Point y in pixels (default: image center)",
    )

    args = parser.parse_args()

    # Map output naming convention: efficientvit_s/m/l -> b0/b1/b2
    model = build_efficientsam3_image_model(
        checkpoint_path=args.checkpoint,
        backbone_type="efficientvit",
        model_name="b1",
        enable_inst_interactivity=True,
    )

    image = Image.open(args.image).convert("RGB")
    w, h = image.size

    x = float(args.x) if args.x is not None else (w / 2.0)
    y = float(args.y) if args.y is not None else (h / 2.0)

    point_coords = np.array([[x, y]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)  # 1=positive point

    processor = Sam3Processor(model)
    inference_state = processor.set_image(image)

    masks, scores, _ = model.predict_inst(
        inference_state,
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    print(len(masks), scores)


if __name__ == "__main__":
    main()
