"""Unit tests for ``EdgeConfig``."""

from __future__ import annotations

import pytest

from ef3_tracking.config import EdgeConfig


def test_default_config_is_valid():
    cfg = EdgeConfig()
    assert cfg.precision == "fp16"
    assert cfg.frame_stride == 1
    assert cfg.max_num_objects == 8


def test_orin_agx_preset_picks_efficientvit_and_fp16():
    cfg = EdgeConfig.for_orin_agx()
    assert cfg.backbone_type == "efficientvit"
    assert cfg.precision == "fp16"
    assert cfg.text_encoder_type == "MobileCLIP-S0"


def test_orin_nx_preset_skips_more_frames_than_agx():
    nx = EdgeConfig.for_orin_nx()
    agx = EdgeConfig.for_orin_agx()
    assert nx.frame_stride >= agx.frame_stride
    assert nx.max_num_objects <= agx.max_num_objects


def test_cpu_preset_forces_cpu_device():
    cfg = EdgeConfig.for_cpu()
    assert cfg.device == "cpu"
    assert cfg.precision == "fp32"


@pytest.mark.parametrize("bad", [0, -1, -10])
def test_invalid_frame_stride_raises(bad):
    with pytest.raises(ValueError):
        EdgeConfig(frame_stride=bad)


@pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0])
def test_invalid_confidence_threshold_raises(bad):
    with pytest.raises(ValueError):
        EdgeConfig(confidence_threshold=bad)


def test_max_resolution_too_small_raises():
    with pytest.raises(ValueError):
        EdgeConfig(max_resolution=32)


def test_max_num_objects_must_be_positive():
    with pytest.raises(ValueError):
        EdgeConfig(max_num_objects=0)
