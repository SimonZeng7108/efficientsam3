#!/usr/bin/env python3
"""
Simple inference script for SAM3-LiteText.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
from PIL import Image

workspace_root = os.path.dirname(os.path.abspath(__file__))
sam3_repo_root = os.path.dirname(workspace_root)
sys.path.insert(0, sam3_repo_root)

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results


def run_inference():
    # --- Configuration ---
    checkpoint_path = "output/sam3_litetext/efficient_sam3_image_encoder_mobileclip_s0_ctx16.pt"

    backbone_type = "MobileCLIP-S0"  # "MobileCLIP-S0", "MobileCLIP-S1", "MobileCLIP2-L"
    target_context_length = 16       # 16, 32, or 77

    image_path = "sam3/assets/dog_person.jpeg"
    bpe_path = "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    prompt = "dog"
    output_path = "litetext_result_dog.png"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ---------------------

    print(f"Using device: {device}")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        # return

    # 1. Build SAM3-LiteText model in a single call
    print(f"Building SAM3-LiteText model ({backbone_type}, ctx={target_context_length})...")
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=checkpoint_path,
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
        text_encoder_type=backbone_type,
        text_encoder_context_length=target_context_length,
        device=device,
    )

    # 2. Run Inference
    print(f"Running inference with prompt: '{prompt}'")
    processor = Sam3Processor(model, device=device, confidence_threshold=0.4)

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    image_pil = Image.open(image_path).convert("RGB")

    dtype_context = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.no_grad()
    with dtype_context:
        state = processor.set_image(image_pil)
        state = processor.set_text_prompt(prompt, state)

    # 3. Visualize
    if "scores" in state:
        print(f"Detections found: {len(state['scores'])}")
    else:
        print("No detections found.")

    plot_results(image_pil, state)
    plt.suptitle(f"{backbone_type} (ctx={target_context_length}) | Prompt: '{prompt}'", fontsize=12)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Result saved to {output_path}")


if __name__ == "__main__":
    run_inference()
