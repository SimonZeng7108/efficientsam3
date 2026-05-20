"""Shared test fixtures: synthetic videos and a mock backend."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest
from PIL import Image

# Make the package importable when running pytest from any directory.
_PKG_ROOT = Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from ef3_tracking.backends.mock import MockBackend


@pytest.fixture
def synthetic_frames() -> List[np.ndarray]:
    """8 RGB frames, 64x80, with a moving blob so tests can assert on it."""
    frames = []
    for i in range(8):
        frame = np.full((64, 80, 3), 30, dtype=np.uint8)
        cx, cy, r = 20 + i, 30, 6
        ys, xs = np.ogrid[:64, :80]
        blob = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
        frame[blob] = [200, 80, 80]
        frames.append(frame)
    return frames


@pytest.fixture
def jpeg_video_dir(tmp_path: Path, synthetic_frames: List[np.ndarray]) -> Path:
    """A directory of JPEG frames suitable for ``VideoReader``."""
    target = tmp_path / "frames"
    target.mkdir()
    for i, frame in enumerate(synthetic_frames):
        Image.fromarray(frame).save(target / f"{i:05d}.jpg")
    return target


@pytest.fixture
def mp4_video(tmp_path: Path, synthetic_frames: List[np.ndarray]) -> Path:
    """An MP4 file containing the synthetic frames."""
    import cv2

    path = tmp_path / "video.mp4"
    h, w = synthetic_frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    assert writer.isOpened(), "cv2 VideoWriter failed to open"
    try:
        for frame in synthetic_frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return path


@pytest.fixture
def mock_backend() -> MockBackend:
    return MockBackend(num_frames=8, width=80, height=64, drift_per_frame=1)
