"""Run manual-selection tracking from the command line.

Two ways to specify the selection:

    1. ``--box x1,y1,x2,y2`` -- pixel coords in the seed frame.
    2. ``--point x,y[,label]`` (repeatable) -- positive/negative clicks.

You can mix them. At least one positive point or a box is required.
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from ._common import add_edge_config_args, edge_config_from_args
from ..backends import build_edge_backend
from ..manual_tracker import ManualTracker
from ..pipeline import TrackingPipeline
from ..prompts import BoxPrompt, ManualSelection, PointPrompt
from ..video_io import VideoReader, VideoWriter


def _parse_box(value: str) -> BoxPrompt:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "expected --box as 'x1,y1,x2,y2'"
        )
    x1, y1, x2, y2 = (float(p) for p in parts)
    return BoxPrompt(x1=x1, y1=y1, x2=x2, y2=y2)


def _parse_point(value: str) -> PointPrompt:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(
            "expected --point as 'x,y' or 'x,y,label'"
        )
    x = float(parts[0])
    y = float(parts[1])
    label = int(parts[2]) if len(parts) == 3 else 1
    return PointPrompt(x=x, y=y, label=label)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EfficientSAM3 manual-selection video tracker (edge devices)."
    )
    parser.add_argument("--video", required=True, help="Path to .mp4 or frame directory.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output annotated video path (.mp4) or directory for PNG frames.",
    )
    parser.add_argument(
        "--box",
        type=_parse_box,
        default=None,
        help="Bounding box on the seed frame as 'x1,y1,x2,y2' in pixels.",
    )
    parser.add_argument(
        "--point",
        type=_parse_point,
        action="append",
        default=[],
        help="Point on the seed frame as 'x,y' or 'x,y,label' (label=1 positive, 0 negative). Repeatable.",
    )
    parser.add_argument(
        "--seed-frame",
        type=int,
        default=0,
        help="Frame index to anchor the selection on (default: 0).",
    )
    parser.add_argument(
        "--obj-id",
        type=int,
        default=0,
        help="Object id assigned to this selection (default: 0).",
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

    if args.box is None and not any(p.label == 1 for p in args.point):
        print(
            "ERROR: provide --box and/or at least one positive --point.",
            file=sys.stderr,
        )
        return 2

    config = edge_config_from_args(args)

    reader = VideoReader(args.video, frame_stride=config.frame_stride)

    selection = ManualSelection(
        points=list(args.point),
        box=args.box,
        obj_id=args.obj_id,
    )

    backend = build_edge_backend(config)
    tracker = ManualTracker(backend, label=f"obj{args.obj_id}")

    writer = None
    if not args.no_write:
        writer = VideoWriter(args.output, width=reader.width, height=reader.height, fps=reader.metadata.fps)

    pipeline = TrackingPipeline(tracker, reader, writer, alpha=args.alpha)

    with pipeline.session():
        seed_dets = pipeline.seed_with_selections([selection], frame_idx=args.seed_frame)
        print(f"seed detections: {len(seed_dets)}")
        result = pipeline.run()

    print(f"tracked {result.num_objects()} object(s) across {result.num_frames()} frame(s)")
    if writer is not None:
        print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
