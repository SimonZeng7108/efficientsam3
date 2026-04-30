"""测试 DataTrain.txt 数据集格式转换。"""

import pytest

from fewshot_adapter.data.models import HBB
from fewshot_adapter.data.datatrain import (
    DataTrainDataset,
    build_image_map,
    load_datatrain,
    load_datatrain_image_ids,
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


def test_parse_datatrain_line_reads_colon_polygon_format():
    """支持真实数据中的 `图片名:数量 P:4 x y ... "label"` 格式。"""
    line = '90008204_c1s1_00000.bmp.bmp:1 P:4 214 539 938 347 960 430 235 621 "obj"'

    annotations = parse_datatrain_line(line)

    assert len(annotations) == 1
    assert annotations[0].image_id == "90008204_c1s1_00000.bmp.bmp"
    assert annotations[0].object_id == "90008204_c1s1_00000.bmp_0001"
    assert annotations[0].label == "obj"
    assert annotations[0].source_type == "polygon"
    assert annotations[0].polygon == [(214, 539), (938, 347), (960, 430), (235, 621)]


def test_parse_datatrain_line_reads_colon_rotated_rectangle_as_polygon():
    """真实 R:4 标注给的是四个点，需要保留为 polygon，而不是压成水平框。"""
    line = '1_20241216111158.jpg.bmp:1 R:4 577 518 518 518 518 458 577 458 "obj"'

    annotations = parse_datatrain_line(line)

    assert annotations[0].source_type == "polygon"
    assert annotations[0].polygon == [(577, 518), (518, 518), (518, 458), (577, 458)]


def test_parse_datatrain_line_keeps_plain_jpg_filename():
    """图片名不限制后缀，只有 .jpg 的文件也要按原名保留。"""
    line = 'plain_image.jpg:1 P:4 10 10 20 10 20 20 10 20 "obj"'

    annotations = parse_datatrain_line(line)

    assert annotations[0].image_id == "plain_image.jpg"
    assert annotations[0].object_id == "plain_image_0001"


def test_parse_datatrain_line_skips_degenerate_placeholder_object():
    """`1 1 1 1 1 1 1 1` 是无目标占位，不应写入 full_gt。"""
    line = '90008205_c1s1_00088.bmp.bmp:1 P:4 1 1 1 1 1 1 1 1 "obj"'

    assert parse_datatrain_line(line) == []


def test_load_datatrain_skips_version_header(tmp_path):
    datatrain = tmp_path / "DataTrain.txt"
    datatrain.write_text(
        "\n".join(
            [
                "Version 1.0.0",
                '90008204_c1s1_00000.bmp.bmp:1 P:4 214 539 938 347 960 430 235 621 "obj"',
            ]
        ),
        encoding="utf-8",
    )

    annotations = load_datatrain(datatrain)

    assert len(annotations) == 1
    assert annotations[0].image_id == "90008204_c1s1_00000.bmp.bmp"


def test_datatrain_dataset_keeps_no_object_images_in_image_map(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "valid.bmp.bmp").write_bytes(b"fake-image")
    (image_dir / "empty.jpg.bmp").write_bytes(b"fake-image")
    datatrain = tmp_path / "DataTrain.txt"
    datatrain.write_text(
        "\n".join(
            [
                "Version 1.0.0",
                'valid.bmp.bmp:1 P:4 10 10 20 10 20 20 10 20 "obj"',
                'empty.jpg.bmp:1 R:4 1 1 1 1 1 1 1 1 "obj"',
            ]
        ),
        encoding="utf-8",
    )

    dataset = DataTrainDataset.from_file(datatrain, image_dir=image_dir)

    assert [annotation.image_id for annotation in dataset.annotations] == ["valid.bmp.bmp"]
    assert set(dataset.image_map) == {"valid.bmp.bmp", "empty.jpg.bmp"}


def test_load_datatrain_image_ids_keeps_placeholder_only_images(tmp_path):
    """全部图片都要进入 image_map，即使这一行只有无目标占位。"""
    datatrain = tmp_path / "DataTrain.txt"
    datatrain.write_text(
        "\n".join(
            [
                "Version 1.0.0",
                'valid.bmp.bmp:1 P:4 10 10 20 10 20 20 10 20 "obj"',
                'empty.jpg.bmp:1 R:4 1 1 1 1 1 1 1 1 "obj"',
            ]
        ),
        encoding="utf-8",
    )

    assert load_datatrain_image_ids(datatrain) == ["valid.bmp.bmp", "empty.jpg.bmp"]


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
