from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from stage3.data_engine.annotations import (
    GROUPED_SCHEMA_VERSION,
    area_to_fraction,
    bbox_xywh_to_normalized_xywh,
    bbox_xywh_to_xyxy,
    disambiguate_duplicate_labels,
    is_generic_label,
    normalize_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw image text-mask pair annotations into grouped Stage 3 manifests."
    )
    parser.add_argument(
        "--raw-jsonl",
        default="data/sa1b_stage3_pseudo/raw/train.jsonl",
        type=str,
    )
    parser.add_argument(
        "--output-dir",
        default="data/sa1b_stage3_pseudo/grouped",
        type=str,
    )
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--min-confidence", default=0.35, type=float)
    parser.add_argument(
        "--group-strategy",
        default="disambiguate",
        choices=["disambiguate", "merge"],
        help="How to handle duplicate labels within one image.",
    )
    parser.add_argument("--images-per-shard", default=1000, type=int)
    return parser.parse_args()


def _iter_raw_records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r") as fopen:
        for line in fopen:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _is_accepted(record: Dict[str, Any], min_confidence: float) -> bool:
    label = normalize_label(record.get("normalized_label") or record.get("label"))
    if record.get("rejected", False):
        return False
    if record.get("ambiguous", False):
        return False
    if is_generic_label(label):
        return False
    if float(record.get("confidence", 0.0)) < min_confidence:
        return False
    return bool(label)


def _record_bbox_xywh(record: Dict[str, Any]) -> List[float]:
    return [float(v) for v in record["bbox_xywh"]]


def _record_bbox_xyxy(record: Dict[str, Any]) -> List[float]:
    if "bbox_xyxy" in record and record["bbox_xyxy"]:
        return [float(v) for v in record["bbox_xyxy"]]
    return bbox_xywh_to_xyxy(_record_bbox_xywh(record))


def _annotation_from_record(
    record: Dict[str, Any], object_id: int, source_name: str
) -> Dict[str, Any]:
    width = int(record["width"])
    height = int(record["height"])
    bbox_xywh = _record_bbox_xywh(record)
    area_px = float(record.get("area", 0.0))
    return {
        "id": object_id,
        "image_id": 0,
        "object_id": object_id,
        # Keep both original-style normalized boxes and explicit pixel-space boxes.
        "bbox": bbox_xywh_to_normalized_xywh(bbox_xywh, width=width, height=height),
        "bbox_xywh": bbox_xywh,
        "bbox_xyxy": _record_bbox_xyxy(record),
        "area": area_px,
        "area_frac": float(
            record.get("area_frac", area_to_fraction(area_px, width=width, height=height))
        ),
        "segmentation": record["segmentation"],
        "is_crowd": 0,
        "source": source_name,
        "source_mask_id": str(record["mask_id"]),
        "source_mask_index": int(record.get("mask_index", -1)),
        "pseudo_confidence": float(record.get("confidence", 0.0)),
    }


def _query_from_group(
    query_id: int,
    query_text: str,
    output_object_ids: List[int],
    prompt_bboxes_xywh: List[List[float]],
    width: int,
    height: int,
    confidence: float,
    source_mask_ids: List[str],
) -> Dict[str, Any]:
    prompt_boxes_xywh = [[float(v) for v in bbox] for bbox in prompt_bboxes_xywh]
    prompt_boxes_xyxy = [bbox_xywh_to_xyxy(bbox) for bbox in prompt_boxes_xywh]
    return {
        "id": query_id,
        "original_cat_id": -1,
        "object_ids_output": output_object_ids,
        "query_text": query_text,
        "query_processing_order": 0,
        "ptr_x_query_id": None,
        "ptr_y_query_id": None,
        "image_id": 0,
        # Original SAM3 JSON loaders expect normalized xywh here.
        "input_box": [
            bbox_xywh_to_normalized_xywh(bbox, width=width, height=height)
            for bbox in prompt_boxes_xywh
        ],
        # Explicit pixel-space prompt boxes are easier for new Stage 3 adapters.
        "input_box_xywh": prompt_boxes_xywh,
        "input_box_xyxy": prompt_boxes_xyxy,
        "input_box_label": [1 for _ in prompt_boxes_xywh],
        "input_points": None,
        # Pseudo labels are filtered and incomplete, so they should not be treated as exhaustive.
        "is_exhaustive": False,
        "is_pixel_exhaustive": False,
        "confidence": float(confidence),
        "source_mask_ids": source_mask_ids,
    }


def _build_image_row(records: List[Dict[str, Any]], group_strategy: str) -> Dict[str, Any]:
    exemplar = records[0]
    source_name = "sa1b_pseudo"
    annotations: List[Dict[str, Any]] = []
    accepted_records: List[Tuple[Dict[str, Any], int]] = []
    for record in records:
        if not _is_accepted(record, min_confidence=args.min_confidence):
            continue
        object_id = len(accepted_records)
        annotations.append(
            _annotation_from_record(record, object_id=object_id, source_name=source_name)
        )
        accepted_records.append((record, object_id))

    if not accepted_records:
        return {}

    queries: List[Dict[str, Any]] = []
    if group_strategy == "merge":
        grouped: Dict[str, List[tuple[Dict[str, Any], int]]] = defaultdict(list)
        for record, object_id in accepted_records:
            grouped[normalize_label(record["normalized_label"])].append((record, object_id))
        for query_id, (label, members) in enumerate(
            sorted(grouped.items(), key=lambda item: item[0])
        ):
            queries.append(
                _query_from_group(
                    query_id=query_id,
                    query_text=label,
                    output_object_ids=[object_id for _, object_id in members],
                    prompt_bboxes_xywh=[_record_bbox_xywh(member) for member, _ in members],
                    width=int(exemplar["width"]),
                    height=int(exemplar["height"]),
                    confidence=min(float(member.get("confidence", 0.0)) for member, _ in members),
                    source_mask_ids=[str(member["mask_id"]) for member, _ in members],
                )
            )
    else:
        grouped = defaultdict(list)
        for record, object_id in accepted_records:
            grouped[normalize_label(record["normalized_label"])].append((record, object_id))
        used_labels: set[str] = set()
        query_id = 0
        for label, members in sorted(grouped.items(), key=lambda item: item[0]):
            if len(members) == 1:
                record, object_id = members[0]
                query_text = label
                used_labels.add(query_text)
                queries.append(
                    _query_from_group(
                        query_id=query_id,
                        query_text=query_text,
                        output_object_ids=[object_id],
                        prompt_bboxes_xywh=[_record_bbox_xywh(record)],
                        width=int(exemplar["width"]),
                        height=int(exemplar["height"]),
                        confidence=float(record.get("confidence", 0.0)),
                        source_mask_ids=[str(record["mask_id"])],
                    )
                )
                query_id += 1
                continue
            for record, object_id in members:
                query_text = disambiguate_duplicate_labels(
                    label=label,
                    bbox_xywh=record["bbox_xywh"],
                    width=int(record["width"]),
                    height=int(record["height"]),
                    used_labels=used_labels,
                )
                used_labels.add(query_text)
                queries.append(
                    _query_from_group(
                        query_id=query_id,
                        query_text=query_text,
                        output_object_ids=[object_id],
                        prompt_bboxes_xywh=[_record_bbox_xywh(record)],
                        width=int(exemplar["width"]),
                        height=int(exemplar["height"]),
                        confidence=float(record.get("confidence", 0.0)),
                        source_mask_ids=[str(record["mask_id"])],
                    )
                )
                query_id += 1

    if not queries:
        return {}

    return {
        "schema_version": GROUPED_SCHEMA_VERSION,
        "source": source_name,
        "image_id": exemplar["image_id"],
        "image_path": exemplar["image_path"],
        "width": int(exemplar["width"]),
        "height": int(exemplar["height"]),
        "file_name": Path(str(exemplar["image_path"])).name,
        "annotations": annotations,
        "queries": queries,
        "stats": {
            "num_raw_masks": len(records),
            "num_training_annotations": len(annotations),
            "num_training_queries": len(queries),
            "group_strategy": group_strategy,
        },
    }


def main() -> None:
    global args
    args = parse_args()
    raw_jsonl = Path(args.raw_jsonl)
    output_dir = Path(args.output_dir)
    split_output_dir = output_dir / args.split
    split_output_dir.mkdir(parents=True, exist_ok=True)

    grouped_by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in _iter_raw_records(raw_jsonl):
        grouped_by_image[str(record["image_id"])].append(record)

    shard_index = 0
    lines_in_shard = 0
    current_path = split_output_dir / f"{args.split}-{shard_index:05d}.jsonl"
    current_file = current_path.open("w")
    emitted_images = 0

    try:
        for image_id in sorted(grouped_by_image.keys()):
            image_row = _build_image_row(
                records=grouped_by_image[image_id],
                group_strategy=args.group_strategy,
            )
            if not image_row:
                continue
            if lines_in_shard >= args.images_per_shard:
                current_file.close()
                shard_index += 1
                lines_in_shard = 0
                current_path = split_output_dir / f"{args.split}-{shard_index:05d}.jsonl"
                current_file = current_path.open("w")
            current_file.write(json.dumps(image_row) + "\n")
            lines_in_shard += 1
            emitted_images += 1
    finally:
        current_file.close()

    print(
        json.dumps(
            {
                "raw_jsonl": str(raw_jsonl),
                "output_dir": str(split_output_dir),
                "group_strategy": args.group_strategy,
                "emitted_images": emitted_images,
                "num_shards": shard_index + 1,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
