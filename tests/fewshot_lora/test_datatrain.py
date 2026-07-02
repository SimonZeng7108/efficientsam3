from pathlib import Path

from fewshot_lora.data.datatrain import parse_detect_train_data, read_dataset_list


def test_parse_detect_train_data_skips_version_and_parses_multi_instance_obb(tmp_path: Path):
    datatrain = tmp_path / "DetectTrainData.txt"
    datatrain.write_text(
        "\n".join(
            [
                "Version 1.0.0",
                '20230922101406.jpg.bmp:2 R:4 604 423 504 362 671 86 772 148 "Sample" '
                'R:4 10 20 30 20 30 40 10 40 "Sample"',
                'empty.bmp:0',
            ]
        ),
        encoding="utf-8",
    )

    dataset = parse_detect_train_data(datatrain)

    assert [record.image_name for record in dataset.records] == [
        "20230922101406.jpg.bmp",
        "empty.bmp",
    ]
    assert dataset.records[0].declared_count == 2
    assert len(dataset.records[0].instances) == 2
    assert dataset.records[0].instances[0].polygon.points[0] == (604.0, 423.0)
    assert dataset.records[0].instances[0].label == "Sample"
    assert dataset.records[1].instances == []
    assert dataset.issues == []


def test_parse_detect_train_data_reports_count_mismatch_without_dropping_instances(tmp_path: Path):
    datatrain = tmp_path / "DetectTrainData.txt"
    datatrain.write_text(
        'part.bmp:3 R:4 1 2 11 2 11 12 1 12 "obj"',
        encoding="utf-8",
    )

    dataset = parse_detect_train_data(datatrain)

    assert len(dataset.records) == 1
    assert len(dataset.records[0].instances) == 1
    assert len(dataset.issues) == 1
    assert dataset.issues[0].kind == "count_mismatch"
    assert "声明 3 个实例，但实际解析到 1 个" in dataset.issues[0].message


def test_read_dataset_list_ignores_blank_lines_and_comments(tmp_path: Path):
    list_file = tmp_path / "datasets.txt"
    list_file.write_text(
        "\n".join(
            [
                "",
                "# ignored",
                "relative_ds",
                str(tmp_path / "absolute_ds"),
            ]
        ),
        encoding="utf-8",
    )

    entries = read_dataset_list(list_file)

    assert entries == [tmp_path / "relative_ds", tmp_path / "absolute_ds"]
