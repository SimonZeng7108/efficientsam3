"""Unit tests for the prompt dataclasses and validators."""

from __future__ import annotations

import pytest

from ef3_tracking.prompts import (
    BoxPrompt,
    ManualSelection,
    PointPrompt,
    TextPrompt,
    validate_box_in_frame,
    validate_points_in_frame,
)


def test_point_default_label_is_positive():
    p = PointPrompt(10, 20)
    assert p.label == 1


def test_point_rejects_unknown_label():
    with pytest.raises(ValueError):
        PointPrompt(10, 20, label=2)


def test_point_rejects_negative_coords():
    with pytest.raises(ValueError):
        PointPrompt(-1, 5)


def test_box_xywh_conversion():
    b = BoxPrompt(10, 20, 30, 40)
    assert b.to_xywh() == (10, 20, 20, 20)


def test_box_normalized_xywh():
    b = BoxPrompt(20, 40, 60, 80)
    norm = b.to_normalized_xywh(image_width=100, image_height=200)
    assert norm == pytest.approx((0.2, 0.2, 0.4, 0.2))


def test_box_normalized_xywh_rejects_bad_image_size():
    b = BoxPrompt(20, 40, 60, 80)
    with pytest.raises(ValueError):
        b.to_normalized_xywh(0, 100)


def test_box_rejects_inverted_coords():
    with pytest.raises(ValueError):
        BoxPrompt(50, 50, 10, 10)


def test_text_prompt_rejects_empty():
    with pytest.raises(ValueError):
        TextPrompt(text="")
    with pytest.raises(ValueError):
        TextPrompt(text="   ")


def test_text_prompt_normalized_is_lowercase_stripped():
    assert TextPrompt(text="  The Red CAR ").normalized == "the red car"


def test_manual_selection_requires_positive_point_or_box():
    with pytest.raises(ValueError):
        ManualSelection(points=[PointPrompt(10, 10, label=0)])

    # With box -> ok
    ManualSelection(box=BoxPrompt(0, 0, 10, 10))

    # With at least one positive point -> ok
    ManualSelection(points=[PointPrompt(5, 5, label=1)])


def test_manual_selection_point_helpers_match_inputs():
    sel = ManualSelection(
        points=[PointPrompt(1, 2, label=1), PointPrompt(3, 4, label=0)],
        obj_id=7,
    )
    assert sel.point_coords() == [[1, 2], [3, 4]]
    assert sel.point_labels() == [1, 0]
    assert sel.obj_id == 7


def test_validate_points_in_frame_rejects_oob():
    points = [PointPrompt(10, 10), PointPrompt(50, 100)]
    with pytest.raises(ValueError):
        validate_points_in_frame(points, width=40, height=200)


def test_validate_box_in_frame_rejects_oob():
    box = BoxPrompt(0, 0, 100, 50)
    with pytest.raises(ValueError):
        validate_box_in_frame(box, width=80, height=60)
    validate_box_in_frame(box, width=120, height=60)  # ok
