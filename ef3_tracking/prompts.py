"""Prompt types used by the trackers.

Two semantic ideas live here:

    * Geometric prompts (clicks + boxes) for the manual tracker.
    * Text prompts for the ViT-text-encoder-driven tracker.

The dataclasses are deliberately small and frame-agnostic; the trackers attach
them to a specific frame when calling the backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class PointPrompt:
    """A single point on a frame, in absolute pixel coordinates.

    ``label`` follows the SAM convention: ``1`` is a positive (inside-object)
    click, ``0`` is a negative (background) click.
    """

    x: float
    y: float
    label: int = 1

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {self.label}")
        if self.x < 0 or self.y < 0:
            raise ValueError(f"point coords must be non-negative, got ({self.x}, {self.y})")


@dataclass(frozen=True)
class BoxPrompt:
    """An axis-aligned bounding box in absolute pixel coords (xyxy)."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(
                f"box must have x2>x1 and y2>y1, got ({self.x1},{self.y1},{self.x2},{self.y2})"
            )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def to_xywh(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.width, self.height)

    def to_normalized_xywh(self, image_width: int, image_height: int) -> Tuple[float, float, float, float]:
        """Return ``[x, y, w, h]`` normalized to ``[0, 1]`` -- the format the SAM3
        video predictor expects for ``bounding_boxes``."""
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image_width and image_height must be positive")
        return (
            self.x1 / image_width,
            self.y1 / image_height,
            self.width / image_width,
            self.height / image_height,
        )


@dataclass
class ManualSelection:
    """User selection on the seed frame: any mix of points and an optional box.

    At least one positive point OR a box must be present, otherwise the tracker
    has nothing to lock onto.
    """

    points: List[PointPrompt] = field(default_factory=list)
    box: BoxPrompt | None = None
    obj_id: int = 0

    def __post_init__(self) -> None:
        has_positive = any(p.label == 1 for p in self.points)
        if not has_positive and self.box is None:
            raise ValueError(
                "ManualSelection needs at least one positive point or a box"
            )

    def point_coords(self) -> List[List[float]]:
        return [[p.x, p.y] for p in self.points]

    def point_labels(self) -> List[int]:
        return [p.label for p in self.points]


@dataclass(frozen=True)
class TextPrompt:
    """A natural-language prompt for grounding.

    Empty / whitespace-only strings are rejected -- the underlying tokenizer
    would crash on them.
    """

    text: str

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("text prompt must be non-empty")

    @property
    def normalized(self) -> str:
        return self.text.strip().lower()


def validate_points_in_frame(
    points: Sequence[PointPrompt], width: int, height: int
) -> None:
    """Raise if any point falls outside the frame."""
    for p in points:
        if p.x >= width or p.y >= height:
            raise ValueError(
                f"point ({p.x}, {p.y}) is outside frame of size {width}x{height}"
            )


def validate_box_in_frame(box: BoxPrompt, width: int, height: int) -> None:
    """Raise if the box leaves the frame."""
    if box.x1 < 0 or box.y1 < 0 or box.x2 > width or box.y2 > height:
        raise ValueError(
            f"box ({box.x1},{box.y1},{box.x2},{box.y2}) leaves frame {width}x{height}"
        )
