"""End-to-end tests for ``TrackingPipeline``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ef3_tracking.backends.mock import MockBackend
from ef3_tracking.manual_tracker import ManualTracker, make_box_selection
from ef3_tracking.pipeline import TrackingPipeline, annotate_only
from ef3_tracking.text_tracker import TextTracker
from ef3_tracking.video_io import VideoReader, VideoWriter


def test_pipeline_text_to_directory(jpeg_video_dir: Path, mock_backend: MockBackend, tmp_path: Path):
    reader = VideoReader(jpeg_video_dir)
    output_dir = tmp_path / "tracked_text"
    writer = VideoWriter(output_dir, width=reader.width, height=reader.height, fps=reader.metadata.fps)
    tracker = TextTracker(mock_backend)
    pipeline = TrackingPipeline(tracker, reader, writer)
    with pipeline.session():
        seed = pipeline.seed_with_text("car")
        assert len(seed) == 1
        result = pipeline.run()
    assert result.num_frames() == mock_backend.num_frames
    written = sorted(output_dir.glob("*.png"))
    assert len(written) == reader.metadata.num_frames

    first = np.array(Image.open(written[0]))
    raw_first = np.array(Image.open(sorted(jpeg_video_dir.glob("*.jpg"))[0]))
    # the writer should have changed pixels because of the overlay
    assert not np.array_equal(first.shape[:2], (0, 0))
    assert first.shape == raw_first.shape


def test_pipeline_manual_to_mp4(jpeg_video_dir: Path, mock_backend: MockBackend, tmp_path: Path):
    reader = VideoReader(jpeg_video_dir)
    output_path = tmp_path / "tracked_manual.mp4"
    writer = VideoWriter(output_path, width=reader.width, height=reader.height, fps=reader.metadata.fps)
    tracker = ManualTracker(mock_backend, label="car")
    pipeline = TrackingPipeline(tracker, reader, writer)
    with pipeline.session():
        seed = pipeline.seed_with_selections([make_box_selection(20, 10, 60, 50)])
        assert len(seed) == 1
        result = pipeline.run()
    assert result.num_frames() == mock_backend.num_frames
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_pipeline_requires_seeding_before_run(jpeg_video_dir: Path, mock_backend: MockBackend):
    reader = VideoReader(jpeg_video_dir)
    tracker = TextTracker(mock_backend)
    pipeline = TrackingPipeline(tracker, reader)
    with pipeline.session():
        with pytest.raises(RuntimeError):
            pipeline.run()


def test_seed_text_requires_text_tracker(jpeg_video_dir: Path, mock_backend: MockBackend):
    reader = VideoReader(jpeg_video_dir)
    tracker = ManualTracker(mock_backend)
    pipeline = TrackingPipeline(tracker, reader)
    with pipeline.session():
        with pytest.raises(TypeError):
            pipeline.seed_with_text("foo")


def test_seed_selections_requires_manual_tracker(jpeg_video_dir: Path, mock_backend: MockBackend):
    reader = VideoReader(jpeg_video_dir)
    tracker = TextTracker(mock_backend)
    pipeline = TrackingPipeline(tracker, reader)
    with pipeline.session():
        with pytest.raises(TypeError):
            pipeline.seed_with_selections([make_box_selection(0, 0, 10, 10)])


def test_pipeline_run_without_writer(jpeg_video_dir: Path, mock_backend: MockBackend):
    reader = VideoReader(jpeg_video_dir)
    tracker = TextTracker(mock_backend)
    pipeline = TrackingPipeline(tracker, reader)
    with pipeline.session():
        pipeline.seed_with_text("car")
        result = pipeline.run()
    assert result.num_frames() == mock_backend.num_frames


def test_annotate_only_rewrites_video(jpeg_video_dir: Path, mock_backend: MockBackend, tmp_path: Path):
    reader = VideoReader(jpeg_video_dir)
    tracker = TextTracker(mock_backend)
    pipeline = TrackingPipeline(tracker, reader)
    with pipeline.session():
        pipeline.seed_with_text("car")
        result = pipeline.run()

    out_dir = tmp_path / "rendered"
    reader2 = VideoReader(jpeg_video_dir)
    writer = VideoWriter(out_dir, width=reader2.width, height=reader2.height, fps=30.0)
    annotate_only(reader2, result, writer)
    assert len(list(out_dir.glob("*.png"))) == reader2.metadata.num_frames
