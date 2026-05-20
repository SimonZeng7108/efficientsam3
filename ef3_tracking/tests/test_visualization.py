"""Tests for the visualization helpers."""

from __future__ import annotations

import numpy as np
import pytest

from ef3_tracking.tracker import TrackedObject
from ef3_tracking.visualization import (
    annotate_frame,
    composite_legend,
    draw_box,
    draw_point,
    overlay_mask,
    palette_color,
)


def _blank_frame(h: int = 64, w: int = 80) -> np.ndarray:
    return np.full((h, w, 3), 50, dtype=np.uint8)


def test_palette_color_is_deterministic_and_cycles():
    assert palette_color(0) == palette_color(0)
    assert palette_color(0) == palette_color(10)
    # different ids in the first cycle give different colors
    assert palette_color(0) != palette_color(1)


def test_overlay_mask_changes_only_masked_pixels():
    frame = _blank_frame()
    mask = np.zeros(frame.shape[:2], dtype=bool)
    mask[10:20, 30:50] = True
    out = overlay_mask(frame, mask, color=(255, 0, 0), alpha=0.5)
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)

    unchanged = ~mask
    np.testing.assert_array_equal(out[unchanged], frame[unchanged])


def test_overlay_mask_rejects_bad_alpha():
    frame = _blank_frame()
    mask = np.zeros(frame.shape[:2], dtype=bool)
    mask[0, 0] = True
    with pytest.raises(ValueError):
        overlay_mask(frame, mask, color=(255, 0, 0), alpha=1.5)


def test_overlay_mask_rejects_size_mismatch():
    frame = _blank_frame()
    mask = np.zeros((10, 10), dtype=bool)
    with pytest.raises(ValueError):
        overlay_mask(frame, mask, color=(255, 0, 0))


def test_draw_box_draws_visible_pixels():
    frame = _blank_frame()
    out = draw_box(frame, (5, 5, 30, 25), color=(255, 0, 0), thickness=2)
    # there should be SOME red pixels on the rectangle border
    red = (out[..., 0] > 100) & (out[..., 1] < 100) & (out[..., 2] < 100)
    assert red.any()


def test_draw_box_with_label_does_not_crash():
    frame = _blank_frame()
    out = draw_box(frame, (5, 5, 30, 25), color=(255, 0, 0), label="car 0.9")
    assert out.shape == frame.shape


def test_draw_point_marks_center():
    frame = _blank_frame()
    out = draw_point(frame, (10, 12), color=(0, 255, 0), radius=3)
    assert not np.array_equal(out[12, 10], frame[12, 10])


def test_annotate_frame_with_mask_and_implicit_box():
    frame = _blank_frame()
    mask = np.zeros(frame.shape[:2], dtype=bool)
    mask[10:20, 30:50] = True
    obj = TrackedObject(obj_id=0, mask=mask, score=0.9, label="car")
    out = annotate_frame(frame, [obj])
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)


def test_annotate_frame_skips_empty_masks():
    frame = _blank_frame()
    empty_mask = np.zeros(frame.shape[:2], dtype=bool)
    obj = TrackedObject(obj_id=0, mask=empty_mask, box_xyxy=None)
    out = annotate_frame(frame, [obj])
    # should be identical -- nothing to draw
    np.testing.assert_array_equal(out, frame)


def test_composite_legend_runs():
    frame = _blank_frame()
    out = composite_legend(frame, {0: "car", 1: "person"})
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)
