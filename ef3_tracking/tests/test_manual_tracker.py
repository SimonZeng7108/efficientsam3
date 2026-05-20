"""Behavioural tests for ``ManualTracker``."""

from __future__ import annotations

import pytest

from ef3_tracking.backends.mock import MockBackend
from ef3_tracking.manual_tracker import (
    ManualTracker,
    make_box_selection,
    make_point_selection,
)
from ef3_tracking.prompts import BoxPrompt, ManualSelection, PointPrompt


def test_open_close_lifecycle(mock_backend: MockBackend):
    tracker = ManualTracker(mock_backend)
    assert tracker.session_id is None
    tracker.open("dummy_source")
    assert tracker.session_id is not None
    tracker.close()
    assert tracker.session_id is None
    types = [name for name, _ in mock_backend.calls]
    assert "start_session" in types
    assert "close_session" in types


def test_double_open_raises(mock_backend: MockBackend):
    tracker = ManualTracker(mock_backend)
    tracker.open("dummy")
    with pytest.raises(RuntimeError):
        tracker.open("again")
    tracker.close()


def test_propagate_without_session_raises(mock_backend: MockBackend):
    tracker = ManualTracker(mock_backend)
    with pytest.raises(RuntimeError):
        next(iter(tracker.propagate()))


def test_add_box_selection_round_trip(mock_backend: MockBackend):
    tracker = ManualTracker(mock_backend)
    tracker.open("src")
    tracker.set_frame_size(80, 64)
    sel = make_box_selection(20, 10, 60, 50, obj_id=0)
    objs = tracker.add_selection(sel, frame_idx=0)
    assert len(objs) == 1
    assert objs[0].obj_id == 0
    assert objs[0].mask is not None
    assert objs[0].mask.shape == (64, 80)
    tracker.close()


def test_box_selection_without_frame_size_raises(mock_backend: MockBackend):
    tracker = ManualTracker(mock_backend)
    tracker.open("src")
    sel = make_box_selection(20, 10, 60, 50)
    with pytest.raises(RuntimeError):
        tracker.add_selection(sel)
    tracker.close()


def test_point_selection_round_trip(mock_backend: MockBackend):
    tracker = ManualTracker(mock_backend)
    tracker.open("src")
    tracker.set_frame_size(80, 64)
    sel = make_point_selection([(40, 32, 1), (10, 5, 0)], obj_id=3)
    objs = tracker.add_selection(sel, frame_idx=0)
    assert len(objs) == 1
    assert objs[0].obj_id == 3


def test_invalid_points_outside_frame_raises(mock_backend: MockBackend):
    tracker = ManualTracker(mock_backend)
    tracker.open("src")
    tracker.set_frame_size(80, 64)
    bad_sel = ManualSelection(points=[PointPrompt(200, 200, label=1)], obj_id=0)
    with pytest.raises(ValueError):
        tracker.add_selection(bad_sel)
    tracker.close()


def test_propagation_iterates_all_frames(mock_backend: MockBackend):
    tracker = ManualTracker(mock_backend)
    tracker.open("src")
    tracker.set_frame_size(80, 64)
    tracker.add_selection(make_box_selection(20, 10, 60, 50))
    seen = list(tracker.propagate())
    assert len(seen) == mock_backend.num_frames
    indices = [fidx for fidx, _ in seen]
    assert indices == list(range(mock_backend.num_frames))


def test_collect_returns_full_tracking_result(mock_backend: MockBackend):
    tracker = ManualTracker(mock_backend)
    tracker.open("src")
    tracker.set_frame_size(80, 64)
    tracker.add_selection(make_box_selection(20, 10, 60, 50))
    result = tracker.collect(width=80, height=64)
    assert result.num_frames() == mock_backend.num_frames
    assert result.num_objects() == 1


def test_tracker_records_selections(mock_backend: MockBackend):
    tracker = ManualTracker(mock_backend)
    tracker.open("src")
    tracker.set_frame_size(80, 64)
    sel1 = make_box_selection(0, 0, 30, 30, obj_id=0)
    sel2 = make_point_selection([(50, 30, 1)], obj_id=1)
    tracker.add_selection(sel1)
    tracker.add_selection(sel2)
    assert len(tracker.selections) == 2
    assert tracker.selections[0].obj_id == 0
    assert tracker.selections[1].obj_id == 1


def test_multi_object_tracking_propagates_both(mock_backend: MockBackend):
    tracker = ManualTracker(mock_backend)
    tracker.open("src")
    tracker.set_frame_size(80, 64)
    tracker.add_selection(make_box_selection(0, 0, 30, 30, obj_id=0))
    tracker.add_selection(make_point_selection([(60, 40, 1)], obj_id=1))
    result = tracker.collect(width=80, height=64)
    assert result.num_objects() == 2
