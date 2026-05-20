"""Smoke tests for the CLI argument parsers.

We don't run the full CLI end-to-end (that would require the real SAM3 stack);
we only verify that the parsers accept the expected flags and that
``edge_config_from_args`` produces a consistent ``EdgeConfig``.
"""

from __future__ import annotations

import pytest

from ef3_tracking.cli._common import add_edge_config_args, edge_config_from_args
from ef3_tracking.cli.track_manual import build_parser as build_manual_parser
from ef3_tracking.cli.track_manual import _parse_box, _parse_point
from ef3_tracking.cli.track_text import build_parser as build_text_parser


def test_parse_box_happy_path():
    box = _parse_box("10,20,30,40")
    assert (box.x1, box.y1, box.x2, box.y2) == (10, 20, 30, 40)


@pytest.mark.parametrize("bad", ["", "10,20", "10,20,30", "a,b,c,d"])
def test_parse_box_rejects_bad(bad):
    import argparse

    with pytest.raises((argparse.ArgumentTypeError, ValueError)):
        _parse_box(bad)


def test_parse_point_with_and_without_label():
    p1 = _parse_point("10,20")
    assert (p1.x, p1.y, p1.label) == (10, 20, 1)
    p2 = _parse_point("10,20,0")
    assert p2.label == 0


def test_manual_parser_accepts_box_only():
    parser = build_manual_parser()
    args = parser.parse_args(["--video", "v", "--output", "o", "--box", "10,10,40,40"])
    assert args.box is not None
    assert args.point == []


def test_manual_parser_accepts_multiple_points():
    parser = build_manual_parser()
    args = parser.parse_args(
        [
            "--video", "v",
            "--output", "o",
            "--point", "10,10",
            "--point", "30,40,0",
        ]
    )
    assert len(args.point) == 2
    assert args.point[1].label == 0


def test_text_parser_requires_prompt():
    parser = build_text_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--video", "v", "--output", "o"])


def test_text_parser_basic():
    parser = build_text_parser()
    args = parser.parse_args(["--video", "v", "--output", "o", "--prompt", "car"])
    assert args.prompt == "car"


def test_edge_config_from_args_orin_agx_default():
    import argparse

    parser = argparse.ArgumentParser()
    add_edge_config_args(parser)
    args = parser.parse_args([])
    cfg = edge_config_from_args(args)
    assert cfg.precision == "fp16"
    assert cfg.backbone_type == "efficientvit"


def test_edge_config_from_args_overrides():
    import argparse

    parser = argparse.ArgumentParser()
    add_edge_config_args(parser)
    args = parser.parse_args(
        [
            "--preset", "cpu",
            "--precision", "bf16",
            "--backbone-type", "tinyvit",
            "--model-name", "21m",
            "--frame-stride", "3",
            "--max-resolution", "640",
        ]
    )
    cfg = edge_config_from_args(args)
    assert cfg.device == "cpu"
    assert cfg.precision == "bf16"
    assert cfg.backbone_type == "tinyvit"
    assert cfg.model_name == "21m"
    assert cfg.frame_stride == 3
    assert cfg.max_resolution == 640
