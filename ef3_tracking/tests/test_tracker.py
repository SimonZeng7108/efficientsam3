"""Tests for the base tracker abstractions and the backend output parser."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ef3_tracking.tracker import (
    TrackedObject,
    TrackingResult,
    parse_backend_outputs,
)


def test_tracked_object_is_present_only_with_nonempty_mask():
    empty = TrackedObject(obj_id=0, mask=np.zeros((10, 10), dtype=bool))
    nonempty = TrackedObject(obj_id=0, mask=np.ones((10, 10), dtype=bool))
    none_mask = TrackedObject(obj_id=0)
    assert not empty.is_present
    assert nonempty.is_present
    assert not none_mask.is_present


def test_tracking_result_aggregates_frames_and_objects():
    r = TrackingResult(width=80, height=64)
    r.add(0, [TrackedObject(obj_id=0), TrackedObject(obj_id=1)])
    r.add(1, [TrackedObject(obj_id=0)])
    assert r.num_frames() == 2
    assert r.num_objects() == 2


def test_object_track_returns_chronological_pairs():
    r = TrackingResult()
    r.add(2, [TrackedObject(obj_id=0)])
    r.add(0, [TrackedObject(obj_id=0)])
    r.add(1, [TrackedObject(obj_id=0)])
    track = r.object_track(0)
    assert [f for f, _ in track] == [0, 1, 2]


def test_parse_backend_outputs_handles_torch_tensors():
    mask = torch.zeros(1, 10, 10, dtype=torch.bool)
    mask[0, 2:5, 3:6] = True
    outputs = {0: {"mask": mask, "score": 0.7}}
    parsed = parse_backend_outputs(outputs)
    assert len(parsed) == 1
    obj = parsed[0]
    assert obj.obj_id == 0
    assert obj.score == pytest.approx(0.7)
    assert obj.mask.shape == (10, 10)
    assert obj.box_xyxy == (3, 2, 5, 4)


def test_parse_backend_outputs_keeps_explicit_box():
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0] = True  # would yield (0,0,0,0)
    outputs = {1: {"mask": mask, "score": 0.5, "box_xyxy": (3, 4, 7, 9)}}
    parsed = parse_backend_outputs(outputs)
    assert parsed[0].box_xyxy == (3, 4, 7, 9)


def test_parse_backend_outputs_handles_empty():
    assert parse_backend_outputs({}) == []
    assert parse_backend_outputs({0: "not a dict"}) == []  # tolerated


def test_parse_backend_outputs_attaches_label():
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    outputs = {0: {"mask": mask, "score": 0.4}}
    parsed = parse_backend_outputs(outputs, label="car")
    assert parsed[0].label == "car"
