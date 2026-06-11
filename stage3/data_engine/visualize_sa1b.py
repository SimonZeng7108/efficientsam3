from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from pycocotools import mask as mask_utils

from stage3.data_engine.annotations import visualize_annotation_example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize SA-1B annotation examples with grouped annotation/query context."
    )
    parser.add_argument("--sa1b-root", default="data/SA-1B-2p", type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--num-examples", default=10, type=int)
    parser.add_argument("--raw-jsonl", default=None, type=str)
    parser.add_argument(
        "--output-dir",
        default="output/stage3/data_engine_sa1b_examples",
        type=str,
    )
    return parser.parse_args()


def _load_raw_record_by_mask_id(raw_jsonl: str | None) -> dict[str, Dict[str, Any]]:
    if raw_jsonl is None:
        return {}
    path = Path(raw_jsonl)
    if not path.exists():
        return {}
    mapping: dict[str, Dict[str, Any]] = {}
    with path.open("r") as fopen:
        for line in fopen:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            mapping[str(record["mask_id"])] = record
    return mapping


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r") as fopen:
        for line in fopen:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _decode_mask(segmentation: dict) -> object:
    if isinstance(segmentation.get("counts"), list):
        rle = mask_utils.frPyObjects(segmentation, segmentation["size"][0], segmentation["size"][1])
    else:
        rle = segmentation
    mask = mask_utils.decode(rle)
    if getattr(mask, "ndim", 0) == 3:
        mask = mask[..., 0]
    return mask.astype(bool)


def _render_from_raw_jsonl(raw_jsonl: Path, output_dir: Path, num_examples: int) -> int:
    rendered = 0
    for record in _iter_jsonl(raw_jsonl):
        if rendered >= num_examples:
            break
        image_path = Path(record["image_path"])
        if not image_path.exists():
            continue
        visualize_annotation_example(
            image_path=str(image_path),
            annotations=[
                {
                    "bbox_xywh": [float(v) for v in record["bbox_xywh"]],
                    "annotation_text": str(
                        record.get("label")
                        or record.get("normalized_label")
                        or "annotation pending"
                    ),
                    "mask_id": str(record["mask_id"]),
                    "area": record.get("area"),
                    "crop_box_xyxy": record.get("crop_box_xyxy"),
                    "mask": _decode_mask(record["segmentation"]),
                }
            ],
            output_path=str(output_dir / f"{image_path.stem}_{record['mask_id']}.jpg"),
            max_annotations=1,
            box_width=10,
        )
        rendered += 1
    return rendered


def main() -> None:
    args = parse_args()
    sa1b_root = Path(args.sa1b_root)
    ann_dir = sa1b_root / "annotations" / args.split
    img_dir = sa1b_root / "images" / args.split
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_record_by_mask_id = _load_raw_record_by_mask_id(args.raw_jsonl)

    if args.raw_jsonl is not None:
        raw_jsonl_path = Path(args.raw_jsonl)
        if raw_jsonl_path.exists():
            rendered = _render_from_raw_jsonl(
                raw_jsonl=raw_jsonl_path,
                output_dir=output_dir,
                num_examples=args.num_examples,
            )
            print(
                json.dumps(
                    {
                        "output_dir": str(output_dir),
                        "rendered_examples": rendered,
                        "split": args.split,
                        "source": str(raw_jsonl_path),
                    },
                    indent=2,
                )
            )
            return

    ann_files = sorted(ann_dir.glob("*.json"))
    rendered = 0
    for ann_file in ann_files:
        if rendered >= args.num_examples:
            break
        with ann_file.open("r") as fopen:
            payload = json.load(fopen)
        image_info = payload["image"]
        image_path = img_dir / image_info["file_name"]
        if not image_path.exists():
            continue
        for ann in payload.get("annotations", []):
            if rendered >= args.num_examples:
                break
            mask_id = str(ann["id"])
            raw_record = raw_record_by_mask_id.get(mask_id, {})
            annotation_text = str(
                raw_record.get("label")
                or raw_record.get("normalized_label")
                or "annotation pending"
            )
            visualize_annotation_example(
                image_path=str(image_path),
                annotations=[
                    {
                        "bbox_xywh": [float(v) for v in ann["bbox"]],
                        "annotation_text": annotation_text,
                        "mask_id": mask_id,
                        "area": raw_record.get("area", ann.get("area")),
                        "crop_box_xyxy": raw_record.get("crop_box_xyxy"),
                        "mask": _decode_mask(ann["segmentation"]),
                    }
                ],
                output_path=str(
                    output_dir / f"{Path(image_info['file_name']).stem}_{mask_id}.jpg"
                ),
                max_annotations=1,
                box_width=10,
            )
            rendered += 1

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "rendered_examples": rendered,
                "split": args.split,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
