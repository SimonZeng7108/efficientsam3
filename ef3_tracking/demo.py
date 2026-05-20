"""Offline mock demo -- exercises the whole pipeline without any model weights.

Useful for verifying that the package is installed correctly on an edge device
before downloading the real checkpoints. Run with::

    python -m ef3_tracking.demo --out /tmp/demo_out

It synthesises a short clip, runs both the manual and the text tracker against
the :class:`MockBackend`, and writes annotated MP4s to ``--out``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .backends import MockBackend
from .manual_tracker import ManualTracker, make_box_selection
from .pipeline import TrackingPipeline
from .text_tracker import TextTracker
from .video_io import VideoReader, VideoWriter


def _synthesize_clip(out_dir: Path, num_frames: int = 60, w: int = 320, h: int = 240) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / "synthetic.mp4"
    writer = cv2.VideoWriter(
        str(src), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h)
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open mp4 for writing: {src}")
    try:
        for i in range(num_frames):
            frame = np.full((h, w, 3), 30, dtype=np.uint8)
            cx = 40 + i * 3
            cy = h // 2
            r = 25
            cv2.circle(frame, (cx, cy), r, (200, 80, 80), -1)
            cv2.putText(
                frame,
                f"frame {i:03d}",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()
    return src


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="./demo_out", help="Output directory.")
    parser.add_argument("--num-frames", type=int, default=60)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    args = parser.parse_args()

    out_root = Path(args.out)
    src = _synthesize_clip(out_root, args.num_frames, args.width, args.height)
    print(f"synthetic source: {src}")

    text_out = out_root / "tracked_text.mp4"
    manual_out = out_root / "tracked_manual.mp4"

    # ----- Text tracker -----
    reader = VideoReader(src)
    backend = MockBackend(num_frames=reader.metadata.num_frames, width=reader.width, height=reader.height)
    writer = VideoWriter(text_out, width=reader.width, height=reader.height, fps=reader.metadata.fps)
    tracker = TextTracker(backend)
    pipeline = TrackingPipeline(tracker, reader, writer)
    with pipeline.session():
        seed = pipeline.seed_with_text("the red blob")
        print(f"[text] seed detections: {len(seed)}")
        result = pipeline.run()
    print(f"[text] wrote {text_out} ({result.num_frames()} frames)")

    # ----- Manual tracker -----
    reader = VideoReader(src)
    backend = MockBackend(num_frames=reader.metadata.num_frames, width=reader.width, height=reader.height)
    writer = VideoWriter(manual_out, width=reader.width, height=reader.height, fps=reader.metadata.fps)
    tracker = ManualTracker(backend, label="blob")
    pipeline = TrackingPipeline(tracker, reader, writer)
    with pipeline.session():
        seed = pipeline.seed_with_selections(
            [make_box_selection(20, args.height // 2 - 30, 80, args.height // 2 + 30, obj_id=0)]
        )
        print(f"[manual] seed detections: {len(seed)}")
        result = pipeline.run()
    print(f"[manual] wrote {manual_out} ({result.num_frames()} frames)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
