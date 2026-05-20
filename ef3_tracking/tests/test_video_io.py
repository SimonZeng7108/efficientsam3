"""Tests for ``VideoReader`` / ``VideoWriter``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ef3_tracking.video_io import VideoReader, VideoWriter


def test_video_reader_reads_jpeg_dir(jpeg_video_dir: Path):
    reader = VideoReader(jpeg_video_dir)
    assert reader.metadata.num_frames == 8
    assert reader.width == 80
    assert reader.height == 64

    frames = list(reader)
    assert len(frames) == 8
    fidx, frame = frames[0]
    assert fidx == 0
    assert frame.shape == (64, 80, 3)
    assert frame.dtype == np.uint8


def test_video_reader_respects_frame_stride(jpeg_video_dir: Path):
    reader = VideoReader(jpeg_video_dir, frame_stride=2)
    indices = [i for i, _ in reader]
    assert indices == [0, 2, 4, 6]
    assert len(reader) == 4


def test_video_reader_random_access(jpeg_video_dir: Path):
    reader = VideoReader(jpeg_video_dir)
    frame = reader.read_frame(3)
    assert frame.shape == (64, 80, 3)
    with pytest.raises(IndexError):
        reader.read_frame(100)


def test_video_reader_missing_path_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        VideoReader(tmp_path / "does_not_exist")


def test_video_reader_empty_dir_raises(tmp_path: Path):
    target = tmp_path / "empty"
    target.mkdir()
    with pytest.raises(FileNotFoundError):
        VideoReader(target)


def test_video_reader_unknown_extension_raises(tmp_path: Path):
    p = tmp_path / "weird.txt"
    p.write_text("not a video")
    with pytest.raises(ValueError):
        VideoReader(p)


def test_video_reader_handles_mp4(mp4_video: Path):
    reader = VideoReader(mp4_video)
    assert reader.width == 80
    assert reader.height == 64
    frames = list(reader)
    assert len(frames) >= 1
    assert frames[0][1].shape == (64, 80, 3)


def test_video_writer_directory_mode(tmp_path: Path):
    out = tmp_path / "out_dir"
    frame = np.zeros((30, 40, 3), dtype=np.uint8)
    with VideoWriter(out, width=40, height=30, fps=30.0) as writer:
        path0 = writer.write_frame(frame, frame_index=0)
        path1 = writer.write_frame(frame, frame_index=1)
    assert path0.exists() and path1.exists()
    assert path0.parent == out


def test_video_writer_mp4_mode(tmp_path: Path):
    out = tmp_path / "out.mp4"
    frame = np.zeros((30, 40, 3), dtype=np.uint8)
    with VideoWriter(out, width=40, height=30, fps=30.0) as writer:
        for i in range(4):
            writer.write_frame(frame, frame_index=i)
    assert out.exists()
    assert out.stat().st_size > 0


def test_video_writer_resizes_mismatched_frame(tmp_path: Path):
    out = tmp_path / "resized"
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    with VideoWriter(out, width=40, height=30, fps=30.0) as writer:
        path = writer.write_frame(frame, frame_index=0)
    assert path.exists()


def test_video_reader_invalid_stride_raises(jpeg_video_dir: Path):
    with pytest.raises(ValueError):
        VideoReader(jpeg_video_dir, frame_stride=0)
