"""Run text-prompt tracking from the command line.

Example
-------

::

    python -m ef3_tracking.cli.track_text \\
        --video sample.mp4 \\
        --output tracked.mp4 \\
        --prompt "the red car" \\
        --preset orin-agx
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from ._common import add_edge_config_args, edge_config_from_args
from ..backends import build_edge_backend
from ..pipeline import TrackingPipeline
from ..text_tracker import TextTracker
from ..video_io import VideoReader, VideoWriter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EfficientSAM3 text-prompt video tracker (edge devices)."
    )
    parser.add_argument("--video", required=True, help="Path to .mp4 or frame directory.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output annotated video path (.mp4) or directory for PNG frames.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help='Natural-language description, e.g. "the red car".',
    )
    parser.add_argument(
        "--seed-frame",
        type=int,
        default=0,
        help="Frame index to ground the prompt on (default: 0).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Mask overlay transparency (default: 0.5).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Skip writing the annotated video and just print per-frame summaries.",
    )
    add_edge_config_args(parser)
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.prompt.strip():
        print("ERROR: --prompt cannot be empty.", file=sys.stderr)
        return 2

    config = edge_config_from_args(args)
    if not config.text_encoder_type:
        print(
            "ERROR: the active edge config has no text encoder; pass --text-encoder-type.",
            file=sys.stderr,
        )
        return 2

    reader = VideoReader(args.video, frame_stride=config.frame_stride)

    backend = build_edge_backend(config)
    tracker = TextTracker(backend)

    writer = None
    if not args.no_write:
        writer = VideoWriter(
            args.output,
            width=reader.width,
            height=reader.height,
            fps=reader.metadata.fps,
        )

    pipeline = TrackingPipeline(tracker, reader, writer, alpha=args.alpha)

    with pipeline.session():
        seed_dets = pipeline.seed_with_text(args.prompt, frame_idx=args.seed_frame)
        print(f"seed detections for '{args.prompt}': {len(seed_dets)}")
        result = pipeline.run()

    print(f"tracked {result.num_objects()} object(s) across {result.num_frames()} frame(s)")
    if writer is not None:
        print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
