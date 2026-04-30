"""DataTrain.txt 数据集格式转换工具。

用户当前数据集只有图片目录和一个 `DataTrain.txt`。本模块把这种文本格式转换
成少样本闭环统一使用的标注对象，并提供 `DataTrainDataset` 作为高层入口。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .json_io import save_annotations
from .models import HBB, Annotation

_OBJECT_PATTERN = re.compile(r'\s*([PRpr])(?::(\d+))?\s+([^"]+?)\s+"([^"]+)"\s*;?')
_COLON_HEADER_PATTERN = re.compile(r"^\s*(.+?):(\d+)\s*(.*)$")


@dataclass(frozen=True)
class _DataTrainParseResult:
    annotations: list[Annotation]
    image_ids: list[str]


@dataclass(frozen=True)
class _ObjectSegment:
    shape_type: str
    point_count: int | None
    coords: list[float]
    label: str


@dataclass(frozen=True)
class DataTrainDataset:
    """DataTrain 数据集的内存表示。

    `annotations` 是全量真值，`image_map` 是 image_id 到图片路径的映射。
    验证阶段保存为 `full_gt.json` 和 `image_map.json` 后即可进入原生训练闭环。
    """

    annotations: list[Annotation]
    image_map: dict[str, str]

    @classmethod
    def from_file(cls, path: str | Path, *, image_dir: str | Path) -> "DataTrainDataset":
        result = _load_datatrain_result(path)
        return cls(
            annotations=result.annotations,
            image_map=build_image_map(result.image_ids, image_dir),
        )

    def save_json(self, output_dir: str | Path) -> None:
        """保存原生少样本闭环需要的两个输入文件。"""
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        save_annotations(output_root / "full_gt.json", self.annotations)
        save_image_map(output_root / "image_map.json", self.image_map)


def load_datatrain(path: str | Path) -> list[Annotation]:
    """读取整个 DataTrain.txt，返回统一 Annotation 列表。"""
    return _load_datatrain_result(path).annotations


def load_datatrain_image_ids(path: str | Path) -> list[str]:
    """读取 DataTrain.txt 中出现过的全部图片名，包括无目标占位图片。"""
    return _load_datatrain_result(path).image_ids


def _load_datatrain_result(path: str | Path) -> _DataTrainParseResult:
    """读取整个 DataTrain.txt，同时返回有效标注和全部图片 ID。"""
    annotations: list[Annotation] = []
    image_ids: list[str] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if _should_skip_line(stripped):
            continue
        try:
            image_id, _, _ = _split_line_header(stripped)
            image_ids.append(image_id)
            annotations.extend(parse_datatrain_line(stripped))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return _DataTrainParseResult(annotations=annotations, image_ids=image_ids)


def parse_datatrain_line(line: str) -> list[Annotation]:
    """解析 DataTrain.txt 的一行。"""
    image_name, expected_count, object_text = _split_line_header(line)
    image_id = _clean_image_id(image_name)
    object_segments = _parse_object_segments(object_text)
    if len(object_segments) != expected_count:
        raise ValueError(
            f"object count mismatch for {image_id}: expected {expected_count}, got {len(object_segments)}"
        )

    annotations: list[Annotation] = []
    for index, segment in enumerate(object_segments, 1):
        object_id = f"{_safe_stem(image_id)}_{index:04d}"
        annotation = _annotation_from_object(
            image_id,
            object_id,
            segment.shape_type,
            segment.coords,
            segment.label,
            point_count=segment.point_count,
        )
        if annotation is not None:
            annotations.append(annotation)
    return annotations


def build_image_map(image_ids: list[str], image_dir: str | Path) -> dict[str, str]:
    """根据图片目录生成 image_id 到图片路径的映射。"""
    root = Path(image_dir)
    image_map: dict[str, str] = {}
    missing: list[str] = []
    for image_id in sorted(set(image_ids)):
        image_path = root / image_id
        if not image_path.is_file():
            missing.append(str(image_path))
            continue
        image_map[image_id] = str(image_path)
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... and {len(missing) - 5} more"
        raise FileNotFoundError(f"image file not found: {preview}{suffix}")
    return image_map


def save_image_map(path: str | Path, image_map: dict[str, str]) -> None:
    """保存 image_map.json。"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(image_map, ensure_ascii=False, indent=2), encoding="utf-8")


def _split_line_header(line: str) -> tuple[str, int, str]:
    colon_match = _COLON_HEADER_PATTERN.match(line)
    if colon_match is not None:
        image_name, object_count, object_text = colon_match.groups()
        return _clean_image_id(image_name), int(object_count), object_text

    parts = line.split(maxsplit=2)
    if len(parts) < 2:
        raise ValueError("line must start with image_name and object_count")
    image_name = _clean_image_id(parts[0])
    try:
        object_count = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"invalid object count: {parts[1]}") from exc
    object_text = parts[2] if len(parts) > 2 else ""
    return image_name, object_count, object_text


def _parse_object_segments(object_text: str) -> list[_ObjectSegment]:
    segments: list[_ObjectSegment] = []
    cursor = 0
    for match in _OBJECT_PATTERN.finditer(object_text):
        gap = object_text[cursor:match.start()]
        if gap.strip(" ;"):
            raise ValueError(f"invalid object segment: {gap.strip()}")
        shape_type, point_count_text, coord_text, label = match.groups()
        segments.append(
            _ObjectSegment(
                shape_type=shape_type,
                point_count=None if point_count_text is None else int(point_count_text),
                coords=_parse_numbers(coord_text),
                label=label,
            )
        )
        cursor = match.end()

    tail = object_text[cursor:]
    if tail.strip(" ;"):
        raise ValueError(f"invalid object segment: {tail.strip()}")
    return segments


def _annotation_from_object(
    image_id: str,
    object_id: str,
    shape_type: str,
    coords: list[float],
    label: str,
    *,
    point_count: int | None = None,
) -> Annotation | None:
    if point_count is not None:
        if point_count < 3:
            raise ValueError(f"{shape_type}: point count must be at least 3")
        if len(coords) != point_count * 2:
            raise ValueError(f"{shape_type}:{point_count} object must have {point_count * 2} coordinates")
        points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
        if _is_degenerate_polygon(points):
            return None
        return Annotation(
            image_id=image_id,
            object_id=object_id,
            label=label,
            source_type="polygon",
            polygon=points,
        )

    if shape_type.upper() == "R":
        if len(coords) != 4:
            raise ValueError("R rectangle object must have 4 coordinates")
        x1, y1, x2, y2 = coords
        if x1 == x2 or y1 == y2:
            return None
        return Annotation(
            image_id=image_id,
            object_id=object_id,
            label=label,
            source_type="hbb",
            hbb=HBB(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)),
        )

    if len(coords) < 6 or len(coords) % 2 != 0:
        raise ValueError("P polygon object must have an even number of at least 6 coordinates")
    points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    if _is_degenerate_polygon(points):
        return None
    return Annotation(
        image_id=image_id,
        object_id=object_id,
        label=label,
        source_type="polygon",
        polygon=points,
    )


def _parse_numbers(coord_text: str) -> list[float]:
    try:
        return [float(token) for token in coord_text.split()]
    except ValueError as exc:
        raise ValueError(f"invalid coordinate list: {coord_text}") from exc


def _clean_image_id(value: str) -> str:
    return value.strip().strip("{}")


def _safe_stem(image_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(image_id).stem)


def _should_skip_line(stripped: str) -> bool:
    return not stripped or stripped.startswith("#") or stripped.lower().startswith("version ")


def _is_degenerate_polygon(points: list[tuple[float, float]]) -> bool:
    """判断四点占位框或零面积 polygon 是否应视为无目标。"""
    if len(set(points)) < 3:
        return True
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) <= 1e-9
