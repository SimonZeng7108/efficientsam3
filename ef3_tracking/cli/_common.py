"""Shared helpers for the two CLI scripts."""

from __future__ import annotations

import argparse

from ..config import EdgeConfig


def add_edge_config_args(parser: argparse.ArgumentParser) -> None:
    """Add the common edge-config flags to ``parser``."""
    group = parser.add_argument_group("edge config")
    group.add_argument(
        "--preset",
        choices=["orin-agx", "orin-nx", "cpu"],
        default="orin-agx",
        help="Pick a hardware preset (default: orin-agx).",
    )
    group.add_argument("--checkpoint", default=None, help="Path to the SAM3 video checkpoint.")
    group.add_argument(
        "--load-from-hf",
        action="store_true",
        help="Download the base video model from HuggingFace if --checkpoint is not given.",
    )
    group.add_argument("--bpe-path", default=None, help="Optional override for the BPE vocab file.")
    group.add_argument(
        "--backbone-type",
        default=None,
        choices=["efficientvit", "tinyvit", "vit"],
        help="Override the backbone family.",
    )
    group.add_argument(
        "--model-name", default=None, help='Model size, e.g. "b0", "b1", "21m".'
    )
    group.add_argument(
        "--text-encoder-type",
        default=None,
        help='Text encoder variant, e.g. "MobileCLIP-S0" / "MobileCLIP-S1".',
    )
    group.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Token context length for the text encoder (used by LiteText variants).",
    )
    group.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16"],
        default=None,
        help="Override the inference precision.",
    )
    group.add_argument(
        "--frame-stride",
        type=int,
        default=None,
        help="Subsample the source video; reduce GPU load on the edge.",
    )
    group.add_argument(
        "--max-resolution",
        type=int,
        default=None,
        help="Maximum input resolution (longest side).",
    )


def edge_config_from_args(args: argparse.Namespace) -> EdgeConfig:
    presets = {
        "orin-agx": EdgeConfig.for_orin_agx,
        "orin-nx": EdgeConfig.for_orin_nx,
        "cpu": EdgeConfig.for_cpu,
    }
    cfg = presets[args.preset]()
    if args.checkpoint is not None:
        cfg.checkpoint_path = args.checkpoint
    if args.load_from_hf:
        cfg.load_from_hf = True
    if args.bpe_path is not None:
        cfg.bpe_path = args.bpe_path
    if args.backbone_type is not None:
        cfg.backbone_type = args.backbone_type
    if args.model_name is not None:
        cfg.model_name = args.model_name
    if args.text_encoder_type is not None:
        cfg.text_encoder_type = args.text_encoder_type
    if args.context_length is not None:
        cfg.text_encoder_context_length = args.context_length
    if args.precision is not None:
        cfg.precision = args.precision
    if args.frame_stride is not None:
        cfg.frame_stride = args.frame_stride
    if args.max_resolution is not None:
        cfg.max_resolution = args.max_resolution
    return cfg
