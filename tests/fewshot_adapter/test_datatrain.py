"""测试 DataTrain.txt 数据集格式转换。"""

import pytest

from fewshot_adapter.data.models import HBB
from fewshot_adapter.data.datatrain import (
    build_image_map,
    parse_datatrain_line,
)


def test_parse_datatrain_line_reads_rectangle_and_polygon_objects():
    line = 'img_001.jpg 2 R 10 20 30 40 "car" ; P 1 2 5 2 5 6 1 6 "ship"'

    annotations = parse_datatrain_line(line)

    assert annotations[0].image_id == "img_001.jpg"
    assert annotations[0].object_id == "img_001_0001"
    assert annotations[0].label == "car"
    assert annotations[0].source_type == "hbb"
    assert annotations[0].hbb == HBB(10, 20, 30, 40)
    assert annotations[1].label == "ship"
    assert annotations[1].source_type == "polygon"
    assert annotations[1].polygon == [(1, 2), (5, 2), (5, 6), (1, 6)]


def test_build_image_map_maps_image_id_to_image_dir(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "img_001.jpg").write_bytes(b"fake-image")

    image_map = build_image_map(["img_001.jpg"], image_dir)

    assert image_map == {"img_001.jpg": str(tmp_path / "images" / "img_001.jpg")}


def test_build_image_map_reports_missing_images_early(tmp_path):
    """DataTrain 转换阶段应提前暴露图片名不匹配，而不是等训练时才失败。"""
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="image file not found"):
        build_image_map(["missing.jpg"], image_dir)
