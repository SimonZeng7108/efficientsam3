"""Video I/O helpers.

`VideoReader` accepts either an MP4 file or a directory of JPEG frames, so the
trackers work with both formats the SAM3 example notebooks use.

`VideoWriter` overlays masks/boxes onto frames as they arrive and writes either
an MP4 or a directory of PNGs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np


_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class VideoMetadata:
    width: int
    height: int
    fps: float
    num_frames: int


class VideoReader:
    """Iterate RGB frames from an mp4 / a JPEG frame directory.

    Importing ``cv2`` is deferred until actually needed -- callers that only
    poke the metadata don't have to install OpenCV.
    """

    def __init__(self, path: str | os.PathLike, frame_stride: int = 1) -> None:
        if frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
        self.path = Path(path)
        self.frame_stride = frame_stride

        if not self.path.exists():
            raise FileNotFoundError(f"video source not found: {self.path}")

        self._is_dir = self.path.is_dir()
        if self._is_dir:
            self._frame_paths = self._collect_frame_paths(self.path)
            if not self._frame_paths:
                raise FileNotFoundError(
                    f"no JPEG/PNG frames in {self.path}"
                )
            self._metadata = self._read_dir_metadata()
        else:
            if self.path.suffix.lower() not in _VIDEO_EXTS:
                raise ValueError(
                    f"unsupported video extension: {self.path.suffix}"
                )
            self._metadata = self._read_mp4_metadata()

    @staticmethod
    def _collect_frame_paths(dir_path: Path) -> List[Path]:
        files = [
            p for p in dir_path.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        ]

        def sort_key(p: Path):
            stem = p.stem
            try:
                return (0, int(stem))
            except ValueError:
                return (1, stem)

        return sorted(files, key=sort_key)

    def _read_dir_metadata(self) -> VideoMetadata:
        # peek at the first frame to learn the resolution
        first = self._read_one_image(self._frame_paths[0])
        h, w = first.shape[:2]
        return VideoMetadata(width=w, height=h, fps=30.0, num_frames=len(self._frame_paths))

    def _read_mp4_metadata(self) -> VideoMetadata:
        import cv2

        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {self.path}")
        try:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        return VideoMetadata(width=w, height=h, fps=fps, num_frames=n)

    @staticmethod
    def _read_one_image(path: Path) -> np.ndarray:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"could not read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @property
    def metadata(self) -> VideoMetadata:
        return self._metadata

    @property
    def width(self) -> int:
        return self._metadata.width

    @property
    def height(self) -> int:
        return self._metadata.height

    def __len__(self) -> int:
        n = self._metadata.num_frames
        return (n + self.frame_stride - 1) // self.frame_stride

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        if self._is_dir:
            for i, p in enumerate(self._frame_paths):
                if i % self.frame_stride != 0:
                    continue
                yield i, self._read_one_image(p)
        else:
            import cv2

            cap = cv2.VideoCapture(str(self.path))
            if not cap.isOpened():
                raise RuntimeError(f"could not open video: {self.path}")
            try:
                idx = 0
                while True:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break
                    if idx % self.frame_stride == 0:
                        yield idx, np.ascontiguousarray(
                            frame_bgr[:, :, ::-1]  # BGR -> RGB
                        )
                    idx += 1
            finally:
                cap.release()

    def read_frame(self, index: int) -> np.ndarray:
        """Random-access read of a single frame (RGB)."""
        if index < 0 or index >= self._metadata.num_frames:
            raise IndexError(
                f"frame index {index} out of range [0, {self._metadata.num_frames})"
            )
        if self._is_dir:
            return self._read_one_image(self._frame_paths[index])
        import cv2

        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {self.path}")
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame_bgr = cap.read()
            if not ret:
                raise RuntimeError(f"could not read frame {index} from {self.path}")
            return np.ascontiguousarray(frame_bgr[:, :, ::-1])
        finally:
            cap.release()


class VideoWriter:
    """Write annotated frames to an MP4 or to a directory of PNGs."""

    def __init__(
        self,
        output_path: str | os.PathLike,
        width: int,
        height: int,
        fps: float = 30.0,
    ) -> None:
        self.output_path = Path(output_path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self._mp4_writer = None
        self._is_dir = self.output_path.suffix.lower() not in _VIDEO_EXTS

        if self._is_dir:
            self.output_path.mkdir(parents=True, exist_ok=True)
        else:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _open_mp4_lazily(self) -> None:
        if self._mp4_writer is not None:
            return
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._mp4_writer = cv2.VideoWriter(
            str(self.output_path), fourcc, self.fps, (self.width, self.height)
        )
        if not self._mp4_writer.isOpened():
            raise RuntimeError(f"could not open mp4 for writing: {self.output_path}")

    def write_frame(self, frame_rgb: np.ndarray, frame_index: Optional[int] = None) -> Path:
        if frame_rgb.dtype != np.uint8:
            frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
        if frame_rgb.shape[:2] != (self.height, self.width):
            import cv2

            frame_rgb = cv2.resize(frame_rgb, (self.width, self.height))

        if self._is_dir:
            import cv2

            name = f"frame_{frame_index:06d}.png" if frame_index is not None else "frame.png"
            path = self.output_path / name
            cv2.imwrite(str(path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            return path

        import cv2

        self._open_mp4_lazily()
        self._mp4_writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        return self.output_path

    def close(self) -> None:
        if self._mp4_writer is not None:
            self._mp4_writer.release()
            self._mp4_writer = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
