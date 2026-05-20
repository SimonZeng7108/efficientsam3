"""Behavioural tests for ``TextTracker``."""

from __future__ import annotations

import pytest

from ef3_tracking.backends.mock import MockBackend
from ef3_tracking.text_tracker import TextTracker


def test_set_prompt_invokes_backend(mock_backend: MockBackend):
    tracker = TextTracker(mock_backend)
    tracker.open("src")
    objs = tracker.set_prompt("the red car")
    assert tracker.prompt is not None
    assert tracker.prompt.normalized == "the red car"
    assert len(objs) == 1
    assert objs[0].label == "the red car"
    types = [name for name, _ in mock_backend.calls]
    assert "add_text_prompt" in types


def test_set_prompt_without_session_raises(mock_backend: MockBackend):
    tracker = TextTracker(mock_backend)
    with pytest.raises(RuntimeError):
        tracker.set_prompt("anything")


def test_empty_prompt_raises(mock_backend: MockBackend):
    tracker = TextTracker(mock_backend)
    tracker.open("src")
    with pytest.raises(ValueError):
        tracker.set_prompt("   ")


def test_text_propagation_through_video(mock_backend: MockBackend):
    tracker = TextTracker(mock_backend)
    with tracker as t:
        t.open("src")
        seed = t.set_prompt("car")
        assert len(seed) == 1
        result = t.collect(width=80, height=64)
        assert result.num_frames() == mock_backend.num_frames
        assert result.num_objects() == 1


def test_object_drifts_across_frames(mock_backend: MockBackend):
    tracker = TextTracker(mock_backend)
    tracker.open("src")
    tracker.set_prompt("car")
    track = tracker.collect(width=80, height=64).object_track(0)
    centroids = []
    for fidx, obj in track:
        ys, xs = (obj.mask).nonzero()
        centroids.append((fidx, xs.mean()))
    # x should be monotonically non-decreasing because MockBackend drifts right
    xs = [c[1] for c in centroids]
    assert xs == sorted(xs)
    assert xs[-1] > xs[0]
    tracker.close()


def test_label_propagates_into_results(mock_backend: MockBackend):
    tracker = TextTracker(mock_backend)
    tracker.open("src")
    tracker.set_prompt("a person on a bike")
    result = tracker.collect(width=80, height=64)
    for objs in result.frames.values():
        for obj in objs:
            assert obj.label == "a person on a bike"
    tracker.close()
