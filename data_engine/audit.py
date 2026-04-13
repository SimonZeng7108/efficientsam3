from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

try:
    from data_engine.annotations import visualize_annotation_example
except ModuleNotFoundError:
    from annotations import visualize_annotation_example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit image text-mask pair data engine outputs."
    )
    parser.add_argument(
        "--raw-jsonl",
        default="data/sa1b_stage3_pseudo/raw/train.jsonl",
        type=str,
    )
    parser.add_argument(
        "--grouped-glob",
        default="data/sa1b_stage3_pseudo/grouped/train/train-*.jsonl",
        type=str,
    )
    parser.add_argument("--render-dir", default=None, type=str)
    parser.add_argument("--sample-images", default=200, type=int)
    parser.add_argument("--seed", default=123, type=int)
    return parser.parse_args()


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r") as fopen:
        for line in fopen:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _collect_raw_stats(raw_jsonl: Path) -> Dict[str, Any]:
    label_lengths: List[int] = []
    rejected = 0
    generic = 0
    duplicate_per_image = 0
    total = 0
    image_to_labels: Dict[str, List[str]] = defaultdict(list)

    for record in _iter_jsonl(raw_jsonl):
        total += 1
        label = str(record.get("normalized_label", "") or "")
        if record.get("rejected", False):
            rejected += 1
        if label in {"", "object", "thing", "unknown", "part"}:
            generic += 1
        if label:
            label_lengths.append(len(label.split()))
            image_to_labels[str(record["image_id"])].append(label)

    for labels in image_to_labels.values():
        counts = Counter(labels)
        duplicate_per_image += sum(1 for count in counts.values() if count > 1)

    return {
        "raw_total_masks": total,
        "raw_rejected_masks": rejected,
        "raw_rejected_rate": 0.0 if total == 0 else rejected / total,
        "raw_generic_rate": 0.0 if total == 0 else generic / total,
        "raw_avg_label_words": 0.0 if not label_lengths else float(np.mean(label_lengths)),
        "raw_duplicate_label_groups": duplicate_per_image,
    }


def _collect_grouped_stats(
    grouped_paths: List[Path],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    query_lengths: List[int] = []
    total_queries = 0
    total_annotations = 0
    for path in grouped_paths:
        for row in _iter_jsonl(path):
            rows.append(row)
            total_queries += len(row.get("queries", []))
            total_annotations += len(row.get("annotations", []))
            for query in row.get("queries", []):
                query_lengths.append(len(str(query.get("query_text", "")).split()))
    return (
        {
            "grouped_images": len(rows),
            "grouped_queries": total_queries,
            "grouped_annotations": total_annotations,
            "grouped_avg_query_words": 0.0 if not query_lengths else float(np.mean(query_lengths)),
        },
        rows,
    )


def _query_prompt_boxes_xywh(row: Dict[str, Any], query: Dict[str, Any]) -> List[List[float]]:
    boxes = query.get("input_box_xywh")
    if boxes is None:
        boxes = query.get("input_box")
        if boxes is None:
            return []
        if boxes and isinstance(boxes[0], (int, float)):
            boxes = [boxes]
        if boxes and max(float(v) for box in boxes for v in box) <= 1.5:
            width = max(int(row.get("width", 1)), 1)
            height = max(int(row.get("height", 1)), 1)
            boxes = [
                [
                    float(box[0]) * width,
                    float(box[1]) * height,
                    float(box[2]) * width,
                    float(box[3]) * height,
                ]
                for box in boxes
            ]
    elif boxes and isinstance(boxes[0], (int, float)):
        boxes = [boxes]

    return [[float(v) for v in box] for box in (boxes or [])]


def _render_samples(
    rows: List[Dict[str, Any]], render_dir: Path, sample_images: int, seed: int
) -> None:
    render_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    chosen_rows = list(rows)
    rng.shuffle(chosen_rows)
    chosen_rows = chosen_rows[:sample_images]
    for row in chosen_rows:
        image_path = Path(row["image_path"])
        if not image_path.exists():
            continue
        ann_by_id = {
            int(annotation.get("object_id", annotation.get("id", -1))): annotation
            for annotation in row.get("annotations", [])
        }
        render_annotations: List[Dict[str, Any]] = []
        for query in row.get("queries", []):
            object_ids = [int(v) for v in query.get("object_ids_output", [])]
            query_area = None
            if len(object_ids) == 1 and object_ids[0] in ann_by_id:
                query_area = ann_by_id[object_ids[0]].get("area")
            prompt_boxes_xywh = _query_prompt_boxes_xywh(row, query)
            prompt_boxes_xyxy = query.get("input_box_xyxy") or []
            if prompt_boxes_xyxy and isinstance(prompt_boxes_xyxy[0], (int, float)):
                prompt_boxes_xyxy = [prompt_boxes_xyxy]
            for prompt_index, prompt_box in enumerate(prompt_boxes_xywh):
                render_annotations.append(
                    {
                        "bbox_xywh": prompt_box,
                        "query_id": int(query.get("id", len(render_annotations))),
                        "query_text": query["query_text"],
                        "object_ids_output": object_ids,
                        "source_mask_ids": query.get("source_mask_ids"),
                        "area": query_area,
                        "crop_box_xyxy": prompt_boxes_xyxy[prompt_index]
                        if prompt_index < len(prompt_boxes_xyxy)
                        else None,
                    }
                )
        visualize_annotation_example(
            image_path=str(image_path),
            annotations=render_annotations,
            output_path=str(render_dir / f"{row['image_id']}.jpg"),
        )


def main() -> None:
    args = parse_args()
    raw_jsonl = Path(args.raw_jsonl)
    grouped_paths = sorted(Path().glob(args.grouped_glob))
    summary = _collect_raw_stats(raw_jsonl)
    grouped_summary, rows = _collect_grouped_stats(grouped_paths)
    summary.update(grouped_summary)

    if args.render_dir:
        _render_samples(
            rows=rows,
            render_dir=Path(args.render_dir),
            sample_images=args.sample_images,
            seed=args.seed,
        )
        summary["render_dir"] = args.render_dir
        summary["rendered_samples"] = min(args.sample_images, len(rows))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
