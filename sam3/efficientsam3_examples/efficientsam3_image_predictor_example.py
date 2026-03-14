#!/usr/bin/env python3
"""
EfficientSAM3 Image Predictor Example Script
converted from efficientsam3_image_predictor_example.ipynb

"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse

# Ensure sam3 is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
# This file lives at: <project_root>/sam3/efficientsam3_examples/efficientsam3_image_predictor_example.py
# `project_root` is the parent of the `sam3/` python package folder.
sam3_pkg_dir = os.path.abspath(os.path.join(current_dir, ".."))  # .../<project_root>/sam3
project_root = os.path.abspath(os.path.join(sam3_pkg_dir, ".."))  # .../<project_root>
if project_root not in sys.path:
    sys.path.append(project_root)

import sam3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results
from sam3.model.box_ops import box_xywh_to_cxcywh

def parse_args():
    parser = argparse.ArgumentParser(description="EfficientSAM3 Image Predictor Example")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Confidence threshold for mask presence (default: 0.1)"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str,
        default=None,
        help="Path to the checkpoint file"
    )
    parser.add_argument(
        "--backbone-type",
        type=str,
        default="MobileCLIP-S0",
        choices=["MobileCLIP-S0", "MobileCLIP-S1", "MobileCLIP2-L"],
        help="LiteText backbone type (default: MobileCLIP-S0)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=16,
        help="Token context length to use at inference (default: 16)",
    )
    parser.add_argument(
        "--pos-embed-table-size",
        type=int,
        default=None,
        help="Positional embedding table size. Defaults to --context-length for fixed/slice inference.",
    )
    parser.add_argument(
        "--interpolate-pos-embed",
        action="store_true",
        default=False,
        help="Optional legacy mode: interpolate the positional table at inference instead of slicing.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    print("Setting up device...")
    from sam3.device import get_device
    device = get_device()

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 1. Setup Paths
    # We resolve relative to sam3 package location
    # In the notebook: f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    # sam3_root is defined as parent of sam3.__file__
    
    # Assets live under <project_root>/sam3/assets
    bpe_path = os.path.join(project_root, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
    image_path = os.path.join(project_root, "sam3", "assets", "dog_person.jpeg")
    
    # Checkpoint path
    # Default to the fixed ctx16 LiteText checkpoint.
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(
            project_root, "output", "ablation_merged", "efficient_sam3_text_s0_ctx16_fixed.pt"
        )

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        # print cwd for debugging
        print(f"CWD: {os.getcwd()}")
        return
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    # 2. Load Model
    print(
        "Loading SAM3 (ViT vision) + LiteText student text encoder "
        f"({args.backbone_type}, ctx={args.context_length}, interp={args.interpolate_pos_embed})..."
    )
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=checkpoint_path,
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
        text_encoder_type=args.backbone_type,
        text_encoder_context_length=args.context_length,
        text_encoder_pos_embed_table_size=args.pos_embed_table_size,
        interpolate_pos_embed=args.interpolate_pos_embed,
        device=device,
    )
    model.to(device)
    model.eval()

    # 3. Load Image & Initialize Processor
    print(f"Loading image: {image_path}")
    
    image_pil = Image.open(image_path).convert("RGB")
    width, height = image_pil.size
    print(f"Image Size: {width}x{height}")
    
    from sam3.device import get_autocast_device_type, get_autocast_dtype
    dtype_context = torch.autocast(get_autocast_device_type(device), dtype=get_autocast_dtype(device)) if device.type in ("cuda", "mps") else torch.no_grad()
    
    with dtype_context:
        print(f"Using confidence threshold: {args.threshold}")
        processor = Sam3Processor(model, confidence_threshold=args.threshold) 
        inference_state = processor.set_image(image_pil)

        # Output directory
        output_dir = os.path.join(project_root, "output")
        os.makedirs(output_dir, exist_ok=True)

        # ---------------------------------------------------------
        # Text Prompt Examples: "dog" and "person"
        # ---------------------------------------------------------
        print("\n--- Text Prompt 'dog' ---")
        processor.reset_all_prompts(inference_state)
        inference_state = processor.set_text_prompt(state=inference_state, prompt="dog")
        
        # Visualize
        output_path_dog = os.path.join(output_dir, "dog_person_example_dog.png")
        plot_results(image_pil, inference_state)
        plt.axis('off')
        plt.savefig(output_path_dog, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved result to {output_path_dog}")

        print("\n--- Text Prompt 'person' ---")
        processor.reset_all_prompts(inference_state)
        inference_state = processor.set_text_prompt(state=inference_state, prompt="person")
        
        # Visualize
        output_path_person = os.path.join(output_dir, "dog_person_example_person.png")
        plot_results(image_pil, inference_state)
        plt.axis('off')
        plt.savefig(output_path_person, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved result to {output_path_person}")



if __name__ == "__main__":
    main()
