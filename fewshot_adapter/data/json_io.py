"""JSON 输入输出工具。

验证阶段的各个脚本通过 JSON 串起来：标注、预测、错误队列、训练集都用
明确的 JSON 文件落盘，便于调试、复现实验和后续替换成数据库。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..evaluation.matching import ErrorItem
from .models import HBB, OBB, Annotation, Prediction


class AnnotationJsonIO:
    """标注、预测和错误队列 JSON 的高层门面。"""

    @staticmethod
    def load_annotations(path: str | Path) -> list[Annotation]:
        return load_annotations(path)

    @staticmethod
    def load_predictions(path: str | Path) -> list[Prediction]:
        return load_predictions(path)

    @staticmethod
    def load_error_queue(path: str | Path) -> list[ErrorItem]:
        return load_error_queue(path)

    @staticmethod
    def save_annotations(path: str | Path, annotations: list[Annotation]) -> None:
        save_annotations(path, annotations)

    @staticmethod
    def save_predictions(path: str | Path, predictions: list[Prediction]) -> None:
        save_predictions(path, predictions)

    @staticmethod
    def save_error_queue(path: str | Path, errors: list[ErrorItem]) -> None:
        save_error_queue(path, errors)


def load_annotations(path: str | Path) -> list[Annotation]:
    """读取标注 JSON。"""
    raw_items = _read_json_list(path)
    return [_annotation_from_dict(item) for item in raw_items]


def load_predictions(path: str | Path) -> list[Prediction]:
    """读取预测 JSON。"""
    raw_items = _read_json_list(path)
    return [_prediction_from_dict(item) for item in raw_items]


def save_annotations(path: str | Path, annotations: list[Annotation]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [_annotation_to_dict(annotation) for annotation in annotations]
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_predictions(path: str | Path, predictions: list[Prediction]) -> None:
    """保存预测 JSON，供后续错误筛选和人工复查使用。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [_prediction_to_dict(prediction) for prediction in predictions]
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_error_queue(path: str | Path, errors: list[ErrorItem]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(error) for error in errors]
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_error_queue(path: str | Path) -> list[ErrorItem]:
    raw_items = _read_json_list(path)
    return [_error_item_from_dict(item) for item in raw_items]


def _annotation_to_dict(annotation: Annotation) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "image_id": annotation.image_id,
        "object_id": annotation.object_id,
        "label": annotation.label,
        "source_type": annotation.source_type,
    }
    if annotation.hbb is not None:
        payload["hbb"] = [
            annotation.hbb.x1,
            annotation.hbb.y1,
            annotation.hbb.x2,
            annotation.hbb.y2,
        ]
    if annotation.obb is not None:
        payload["obb"] = {
            "cx": annotation.obb.cx,
            "cy": annotation.obb.cy,
            "w": annotation.obb.w,
            "h": annotation.obb.h,
            "angle": annotation.obb.angle,
        }
    if annotation.polygon is not None:
        payload["polygon"] = [[x, y] for x, y in annotation.polygon]
    if annotation.mask_path is not None:
        payload["mask_path"] = annotation.mask_path
    return payload


def _prediction_to_dict(prediction: Prediction) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "image_id": prediction.image_id,
        "prediction_id": prediction.prediction_id,
        "label": prediction.label,
        "score": prediction.score,
    }
    if prediction.hbb is not None:
        payload["hbb"] = [
            prediction.hbb.x1,
            prediction.hbb.y1,
            prediction.hbb.x2,
            prediction.hbb.y2,
        ]
    if prediction.obb is not None:
        payload["obb"] = {
            "cx": prediction.obb.cx,
            "cy": prediction.obb.cy,
            "w": prediction.obb.w,
            "h": prediction.obb.h,
            "angle": prediction.obb.angle,
        }
    if prediction.polygon is not None:
        payload["polygon"] = [[x, y] for x, y in prediction.polygon]
    if prediction.mask_path is not None:
        payload["mask_path"] = prediction.mask_path
    return payload


def _read_json_list(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return payload


def _annotation_from_dict(item: dict[str, Any]) -> Annotation:
    return Annotation(
        image_id=str(item["image_id"]),
        object_id=str(item["object_id"]),
        label=str(item["label"]),
        source_type=item["source_type"],
        hbb=_parse_hbb(item.get("hbb")),
        obb=_parse_obb(item.get("obb")),
        polygon=_parse_polygon(item.get("polygon")),
        mask_path=item.get("mask_path"),
    )


def _prediction_from_dict(item: dict[str, Any]) -> Prediction:
    return Prediction(
        image_id=str(item["image_id"]),
        prediction_id=str(item["prediction_id"]),
        label=str(item["label"]),
        score=float(item["score"]),
        hbb=_parse_hbb(item.get("hbb")),
        obb=_parse_obb(item.get("obb")),
        polygon=_parse_polygon(item.get("polygon")),
        mask_path=item.get("mask_path"),
    )


def _error_item_from_dict(item: dict[str, Any]) -> ErrorItem:
    return ErrorItem(
        image_id=str(item["image_id"]),
        error_type=item["error_type"],
        risk_score=float(item["risk_score"]),
        reason=str(item["reason"]),
        ground_truth_ids=[str(value) for value in item.get("ground_truth_ids", [])],
        prediction_ids=[str(value) for value in item.get("prediction_ids", [])],
        selected_for_next_round=bool(item.get("selected_for_next_round", False)),
    )


def _parse_hbb(value: Any) -> HBB | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("hbb must be [x1, y1, x2, y2]")
    return HBB(float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def _parse_obb(value: Any) -> OBB | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("obb must be an object with cx, cy, w, h, angle")
    return OBB(
        cx=float(value["cx"]),
        cy=float(value["cy"]),
        w=float(value["w"]),
        h=float(value["h"]),
        angle=float(value["angle"]),
    )


def _parse_polygon(value: Any) -> list[tuple[float, float]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("polygon must be a list of [x, y] points")
    return [(float(point[0]), float(point[1])) for point in value]
