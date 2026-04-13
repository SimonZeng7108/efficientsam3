#!/usr/bin/env python3
"""Utilities for SAM3.1 / EfficientSAM3.1 image demos.

This module centralizes checkpoint-name parsing and model construction so all
examples use the same SAM3.1-compatible loading path.
"""

from __future__ import annotations

import json
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from PIL import Image


EXAMPLES_DIR = Path(__file__).resolve().parent
SAM3_PKG_PARENT = EXAMPLES_DIR.parent
if str(SAM3_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(SAM3_PKG_PARENT))

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from sam3.visualization_utils import plot_results


BACKBONE_MODEL_MAP = {
    "repvit": {"s": "m0.9", "m": "m1.1", "l": "m2.3"},
    "tinyvit": {"s": "5m", "m": "11m", "l": "21m"},
    "efficientvit": {"s": "b0", "m": "b1", "l": "b2"},
}

TEXT_ENCODER_MAP = {
    "s0": "MobileCLIP-S0",
    "s1": "MobileCLIP-S1",
    "l": "MobileCLIP2-L",
}


def repo_root_from_examples() -> Path:
    """Return repository root when called from sam3/efficientsam3_examples."""
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(path_like: str | Path) -> Path:
    """Resolve a path relative to repository root when needed."""
    p = Path(path_like)
    if p.is_absolute():
        return p
    return repo_root_from_examples() / p


def _infer_stage1_sam3p1_args(name: str) -> dict[str, Any] | None:
    pat = re.compile(
        r"^efficient_sam3p1_(repvit|tinyvit|efficientvit)_([sml])_mobileclip_(s0|s1|l)_ctx(\d+)\.(pt|pth)$"
    )
    match = pat.match(name)
    if not match:
        return None

    backbone = match.group(1)
    size = match.group(2)
    text_variant = match.group(3)
    ctx = int(match.group(4))

    return {
        "variant": "stage1_sam3p1",
        "backbone_type": backbone,
        "model_name": BACKBONE_MODEL_MAP[backbone][size],
        "text_encoder_type": TEXT_ENCODER_MAP[text_variant],
        "text_encoder_context_length": ctx,
        "text_encoder_pos_embed_table_size": ctx,
        "interpolate_pos_embed": False,
    }


def _infer_litetext_sam3p1_args(name: str) -> dict[str, Any] | None:
    pat = re.compile(
        r"^efficient_sam3p1_litetext_mobileclip_(s0|s1|l)_ctx(\d+)\.(pt|pth)$"
    )
    match = pat.match(name)
    if not match:
        return None

    text_variant = match.group(1)
    ctx = int(match.group(2))
    return {
        "variant": "sam3p1_litetext",
        "backbone_type": "sam3",
        "model_name": None,
        "text_encoder_type": TEXT_ENCODER_MAP[text_variant],
        "text_encoder_context_length": ctx,
        "text_encoder_pos_embed_table_size": ctx,
        "interpolate_pos_embed": False,
    }


def infer_model_args_from_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    """Infer builder args from a SAM3.1 checkpoint filename."""
    ckpt = Path(checkpoint_path)
    parsed = _infer_stage1_sam3p1_args(ckpt.name)
    if parsed is None:
        parsed = _infer_litetext_sam3p1_args(ckpt.name)
    if parsed is None:
        raise ValueError(
            "Unsupported checkpoint naming pattern for "
            f"{ckpt.name}. Expected stage1_sam3p1 or sam3p1_litetext format."
        )
    return parsed


def build_processor_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | None = None,
    confidence_threshold: float = 0.4,
    enable_inst_interactivity: bool = False,
) -> tuple[Any, Sam3Processor, dict[str, Any]]:
    """Build model + processor from a checkpoint path with inferred args."""
    ckpt = resolve_repo_path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    inferred = infer_model_args_from_checkpoint(ckpt)
    repo_root = repo_root_from_examples()
    bpe_path = repo_root / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    builder_args: dict[str, Any] = {
        "bpe_path": str(bpe_path),
        "checkpoint_path": str(ckpt),
        "load_from_HF": False,
        "enable_segmentation": True,
        "enable_inst_interactivity": enable_inst_interactivity,
        "compile": False,
        "device": device,
        "text_encoder_type": inferred["text_encoder_type"],
        "text_encoder_context_length": inferred["text_encoder_context_length"],
        "text_encoder_pos_embed_table_size": inferred["text_encoder_pos_embed_table_size"],
        "interpolate_pos_embed": inferred["interpolate_pos_embed"],
    }

    if inferred["variant"] == "stage1_sam3p1":
        builder_args["backbone_type"] = inferred["backbone_type"]
        builder_args["model_name"] = inferred["model_name"]

    model = build_sam3_image_model(**builder_args)
    model = model.float()
    processor = Sam3Processor(
        model, device=device, confidence_threshold=confidence_threshold
    )
    return model, processor, inferred


def save_state_visualization(
    image_pil: Image.Image,
    state: dict[str, Any],
    output_path: str | Path,
    *,
    title: str | None = None,
) -> None:
    """Render and save a segmentation overlay from Sam3Processor inference state."""
    out = resolve_repo_path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 7))
    plot_results(image_pil, state)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.savefig(out, bbox_inches="tight", dpi=160)
    plt.close(fig)


def run_text_prompt_demo(
    checkpoint_path: str | Path,
    image_path: str | Path,
    prompt: str,
    output_path: str | Path,
    *,
    device: str | None = None,
    confidence_threshold: float = 0.4,
) -> dict[str, Any]:
    """Run text-prompt segmentation demo and save visualization."""
    _, processor, inferred = build_processor_from_checkpoint(
        checkpoint_path,
        device=device,
        confidence_threshold=confidence_threshold,
        enable_inst_interactivity=False,
    )
    image_file = resolve_repo_path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Image not found: {image_file}")

    image_pil = Image.open(image_file).convert("RGB")
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if str(processor.device).startswith("cuda")
        else nullcontext()
    )
    with autocast_ctx:
        state = processor.set_image(image_pil)
        state = processor.set_text_prompt(prompt=prompt, state=state)

    save_state_visualization(
        image_pil,
        state,
        output_path,
        title=f"{Path(checkpoint_path).name} | prompt='{prompt.strip()}'",
    )

    scores = state.get("scores")
    score_list = []
    if scores is not None:
        if torch.is_tensor(scores):
            score_list = [float(v) for v in scores.detach().cpu().tolist()]
        else:
            score_list = [float(v) for v in scores]

    return {
        "checkpoint": str(resolve_repo_path(checkpoint_path)),
        "image": str(image_file),
        "prompt": prompt,
        "output": str(resolve_repo_path(output_path)),
        "variant": inferred["variant"],
        "backbone_type": inferred.get("backbone_type"),
        "model_name": inferred.get("model_name"),
        "text_encoder_type": inferred["text_encoder_type"],
        "context_length": inferred["text_encoder_context_length"],
        "num_masks": len(score_list),
        "scores": score_list,
    }


def collect_checkpoint_paths(folder_path: str | Path) -> list[Path]:
    """Collect .pt/.pth checkpoint files from a folder."""
    folder = resolve_repo_path(folder_path)
    if not folder.exists():
        return []
    return [p for p in sorted(folder.iterdir()) if p.suffix in {".pt", ".pth"}]


def write_json(data: Any, output_path: str | Path) -> None:
    """Write JSON with stable formatting."""
    out = resolve_repo_path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
