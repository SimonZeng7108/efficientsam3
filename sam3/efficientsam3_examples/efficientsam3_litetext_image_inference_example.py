#!/usr/bin/env python3
"""Simple SAM3.1 LiteText image inference example."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from sam3p1_demo_utils import run_text_prompt_demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM3.1 LiteText text-prompt image inference")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/sam3p1_litetext/efficient_sam3p1_litetext_mobileclip_s0_ctx16.pt",
        help="Checkpoint path (relative to repo root or absolute)",
    )
    parser.add_argument("--image", default="sam3/assets/dog_person.jpeg")
    parser.add_argument("--prompt", default="dog")
    parser.add_argument("--output", default="vis/litetext_image_inference.png")
    parser.add_argument("--device", default=None, help="cuda, cpu, or leave unset for auto")
    parser.add_argument("--confidence-threshold", type=float, default=0.4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_text_prompt_demo(
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        prompt=args.prompt,
        output_path=args.output,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )
    print(f"saved visualization to: {result['output']}")
    print(f"masks={result['num_masks']} scores={result['scores']}")


if __name__ == "__main__":
    main()
