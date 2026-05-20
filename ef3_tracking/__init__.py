"""EfficientSAM3 edge tracking package.

Tracks objects across a video on edge devices such as NVIDIA Jetson Orin AGX,
exposing two cleanly separated modes:

    * ``ManualTracker`` -- the user selects the object on frame 0 by clicking
      points and/or drawing a bounding box.
    * ``TextTracker``   -- the user passes a natural-language prompt and the
      ViT-based text encoder grounds it on every frame.

The package is intentionally split into small modules with no cross-imports of
the heavy ML stack at top level, so unit tests can run with mocked backends.
"""

from .config import EdgeConfig
from .prompts import BoxPrompt, PointPrompt, TextPrompt
from .tracker import TrackedObject, TrackingResult
from .manual_tracker import ManualTracker
from .text_tracker import TextTracker
from .pipeline import TrackingPipeline
from .video_io import VideoReader, VideoWriter

__all__ = [
    "EdgeConfig",
    "BoxPrompt",
    "PointPrompt",
    "TextPrompt",
    "TrackedObject",
    "TrackingResult",
    "ManualTracker",
    "TextTracker",
    "TrackingPipeline",
    "VideoReader",
    "VideoWriter",
]

__version__ = "0.1.0"
