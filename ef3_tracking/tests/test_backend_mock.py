"""Tests for the mock backend itself."""

from __future__ import annotations

import pytest

from ef3_tracking.backends.mock import MockBackend


def test_start_session_creates_unique_ids():
    backend = MockBackend()
    a = backend.start_session("src1")
    b = backend.start_session("src2")
    assert a != b


def test_text_prompt_then_propagate_yields_object_each_frame():
    backend = MockBackend(num_frames=5, width=100, height=80)
    sid = backend.start_session("src")
    seed = backend.add_text_prompt(sid, frame_idx=0, text="car")
    assert seed["outputs"]
    frames = list(backend.propagate(sid))
    assert len(frames) == 5
    for f in frames:
        assert 0 in f["outputs"]
        assert f["outputs"][0]["mask"].any()


def test_geometric_prompt_box_overrides_centroid():
    backend = MockBackend(num_frames=3, width=100, height=100)
    sid = backend.start_session("src")
    backend.add_geometric_prompt(
        sid, frame_idx=0, obj_id=0,
        bounding_boxes=[[0.1, 0.1, 0.2, 0.2]],
        bounding_box_labels=[1],
    )
    frames = list(backend.propagate(sid))
    first_mask = frames[0]["outputs"][0]["mask"]
    ys, xs = first_mask.nonzero()
    cx = xs.mean()
    cy = ys.mean()
    # centroid should be near (0.2*100, 0.2*100) = (20, 20)
    assert 10 < cx < 30
    assert 10 < cy < 30


def test_geometric_prompt_with_points_uses_positive_centroid():
    backend = MockBackend(num_frames=2, width=100, height=100)
    sid = backend.start_session("src")
    backend.add_geometric_prompt(
        sid, frame_idx=0, obj_id=0,
        points=[[80, 70], [20, 20]],
        point_labels=[1, 0],
    )
    frames = list(backend.propagate(sid))
    mask = frames[0]["outputs"][0]["mask"]
    ys, xs = mask.nonzero()
    cx, cy = xs.mean(), ys.mean()
    assert cx > 50
    assert cy > 50


def test_propagate_unknown_session_raises():
    backend = MockBackend()
    with pytest.raises(RuntimeError):
        list(backend.propagate("nonexistent"))


def test_close_session_removes_state():
    backend = MockBackend()
    sid = backend.start_session("src")
    backend.add_text_prompt(sid, 0, "x")
    backend.close_session(sid)
    with pytest.raises(RuntimeError):
        list(backend.propagate(sid))
