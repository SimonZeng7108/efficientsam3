"""End-to-end video-tracking pipeline.

`TrackingPipeline` glues together a video source, a tracker, and a video writer
so the typical script reduces to four lines:

    pipeline = TrackingPipeline(tracker, reader, writer)
    pipeline.seed_with_text("the red car")
    result = pipeline.run()

Both manual and text trackers reuse this pipeline -- only the seeding step
differs.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, List, Optional

import numpy as np

from .manual_tracker import ManualTracker
from .prompts import ManualSelection, TextPrompt
from .text_tracker import TextTracker
from .tracker import BaseTracker, TrackedObject, TrackingResult
from .video_io import VideoReader, VideoWriter
from .visualization import annotate_frame


class TrackingPipeline:
    """Wire a video source, a tracker, and an optional writer together."""

    def __init__(
        self,
        tracker: BaseTracker,
        reader: VideoReader,
        writer: Optional[VideoWriter] = None,
        *,
        alpha: float = 0.5,
    ) -> None:
        self._tracker = tracker
        self._reader = reader
        self._writer = writer
        self._alpha = alpha
        self._seeded = False
        self._seed_detections: List[TrackedObject] = []

    @property
    def tracker(self) -> BaseTracker:
        return self._tracker

    @property
    def reader(self) -> VideoReader:
        return self._reader

    @property
    def seed_detections(self) -> List[TrackedObject]:
        return list(self._seed_detections)

    @contextmanager
    def session(self) -> Iterator[BaseTracker]:
        self._tracker.open(str(self._reader.path))
        try:
            yield self._tracker
        finally:
            self._tracker.close()
            if self._writer is not None:
                self._writer.close()

    def seed_with_text(self, prompt: str | TextPrompt, frame_idx: int = 0) -> List[TrackedObject]:
        if not isinstance(self._tracker, TextTracker):
            raise TypeError(
                f"seed_with_text requires a TextTracker, got {type(self._tracker).__name__}"
            )
        if self._tracker.session_id is None:
            self._tracker.open(str(self._reader.path))
        self._seed_detections = self._tracker.set_prompt(prompt, frame_idx=frame_idx)
        self._seeded = True
        return self._seed_detections

    def seed_with_selections(
        self,
        selections: List[ManualSelection],
        frame_idx: int = 0,
    ) -> List[TrackedObject]:
        if not isinstance(self._tracker, ManualTracker):
            raise TypeError(
                f"seed_with_selections requires a ManualTracker, got {type(self._tracker).__name__}"
            )
        if self._tracker.session_id is None:
            self._tracker.open(str(self._reader.path))
        self._tracker.set_frame_size(self._reader.width, self._reader.height)

        all_dets: List[TrackedObject] = []
        for sel in selections:
            all_dets.extend(self._tracker.add_selection(sel, frame_idx=frame_idx))
        self._seed_detections = all_dets
        self._seeded = True
        return all_dets

    def run(self) -> TrackingResult:
        """Propagate through every frame; write annotated output if a writer was passed."""
        if not self._seeded:
            raise RuntimeError(
                "Pipeline has not been seeded. Call seed_with_text() or seed_with_selections() first."
            )

        result = TrackingResult(
            width=self._reader.width,
            height=self._reader.height,
            fps=self._reader.metadata.fps,
        )

        per_frame_objects = {}
        for fidx, objs in self._tracker.propagate():
            result.add(fidx, objs)
            per_frame_objects[fidx] = objs

        if self._writer is not None:
            self._write_video(per_frame_objects)
        return result

    def _write_video(self, per_frame_objects) -> None:
        for fidx, frame in self._reader:
            objs = per_frame_objects.get(fidx, [])
            annotated = annotate_frame(frame, objs, alpha=self._alpha)
            self._writer.write_frame(annotated, frame_index=fidx)
        self._writer.close()


def annotate_only(
    reader: VideoReader,
    result: TrackingResult,
    writer: VideoWriter,
    *,
    alpha: float = 0.5,
) -> None:
    """Re-render an annotated video from an already-computed `TrackingResult`."""
    for fidx, frame in reader:
        objs = result.frames.get(fidx, [])
        annotated = annotate_frame(frame, objs, alpha=alpha)
        writer.write_frame(annotated, frame_index=fidx)
    writer.close()
