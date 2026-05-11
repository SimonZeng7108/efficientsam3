"""解析 DetectTrainData 标注文件。

本项目的数据来自一批工业少样本子数据集，每个子数据集都有自己的
`DetectTrainData.txt`。这个文件负责把文本标注解析成统一的数据结构，
后续模块再把四点 OBB 转成 AABB、mask 和评估用 OBB。

特别注意：
- 这里不做图片路径解析，只处理文本内容；图片大小写容错在 `images.py`。
- 这里不假设 label 有语义，只保留原始字符串；评估按“单类别”处理。
- 声明实例数和实际 `R:4` 数量不一致时不会丢数据，而是写入 `issues`。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from ..eval.geometry import Polygon4


# 每个实例形如：R:4 x1 y1 x2 y2 x3 y3 x4 y4 "label"。
# 正则只负责识别完整实例；若一行里有额外未知片段，会通过 count_mismatch 暴露。
_INSTANCE_RE = re.compile(
    r'R:4\s+'
    r'([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+'
    r'([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+'
    r'([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+'
    r'([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+'
    r'"([^"]*)"'
)


@dataclass(frozen=True)
class DataIssue:
    """数据检查问题。

    这些问题通常不立刻中断实验，而是写入 summary.json，方便回头排查数据质量。
    """

    kind: str
    image_name: str | None
    line_number: int
    message: str


@dataclass(frozen=True)
class OBBInstance:
    """单个 OBB 标注实例，保留四点 polygon 和原始 label。"""

    polygon: Polygon4
    label: str


@dataclass(frozen=True)
class ImageAnnotationRecord:
    """标注文件中的一行图片记录。"""

    image_name: str
    declared_count: int
    instances: list[OBBInstance]
    line_number: int


@dataclass(frozen=True)
class DetectTrainDataset:
    """完整 DetectTrainData 解析结果。"""

    source_path: Path
    records: list[ImageAnnotationRecord]
    issues: list[DataIssue]


def read_dataset_list(path: Path) -> list[Path]:
    """读取批量数据集列表。

    每个非空、非注释行都是一个子数据集目录。相对路径按 txt 文件所在目录解析，
    这样把 list 文件和数据放在同一目录树下时更容易迁移到 Linux 服务器。
    """
    root = path.parent
    entries: list[Path] = []
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        dataset_path = Path(line)
        entries.append(dataset_path if dataset_path.is_absolute() else root / dataset_path)
    return entries


def parse_detect_train_data(path: Path) -> DetectTrainDataset:
    """解析 DetectTrainData 文件。

    解析策略故意偏宽松：能解析出的实例会尽量保留，格式问题放入 `issues`。
    这样一个子数据集里少量脏行不会让整批实验直接停掉。
    """

    records: list[ImageAnnotationRecord] = []
    issues: list[DataIssue] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        # DetectTrainData 第一行常见 `Version 1.0.0`，不是标注内容。
        if not line or line.startswith("Version"):
            continue

        if ":" not in line:
            issues.append(
                DataIssue("malformed_line", None, line_number, "缺少 ':' 分隔符")
            )
            continue

        image_name, rest = line.split(":", 1)
        image_name = image_name.strip()
        count_match = re.match(r"\s*(\d+)\b(.*)$", rest)
        if count_match is None:
            issues.append(
                DataIssue(
                    "malformed_count",
                    image_name,
                    line_number,
                    "图片名后缺少声明实例数量",
                )
            )
            continue

        declared_count = int(count_match.group(1))
        annotation_text = count_match.group(2)
        instances = _parse_instances(annotation_text)
        if declared_count != len(instances):
            issues.append(
                DataIssue(
                    "count_mismatch",
                    image_name,
                    line_number,
                    f"{image_name} 声明 {declared_count} 个实例，但实际解析到 {len(instances)} 个",
                )
            )

        records.append(
            ImageAnnotationRecord(
                image_name=image_name,
                declared_count=declared_count,
                instances=instances,
                line_number=line_number,
            )
        )

    return DetectTrainDataset(source_path=path, records=records, issues=issues)


def _parse_instances(text: str) -> list[OBBInstance]:
    """从一行标注剩余文本里提取所有 `R:4` 实例。"""

    instances: list[OBBInstance] = []
    for match in _INSTANCE_RE.finditer(text):
        coords = [float(match.group(i)) for i in range(1, 9)]
        points = tuple((coords[i], coords[i + 1]) for i in range(0, 8, 2))
        instances.append(OBBInstance(polygon=Polygon4(points), label=match.group(9)))
    return instances
