from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample random SA-1B masks that pass strict quality thresholds and "
            "write a visualization-ready raw JSONL."
        )
    )
    parser.add_argument("--sa1b-root", default="data/sa-1b-1p_reorg", type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--num-examples", default=10, type=int)
    parser.add_argument("--min-area", default=5000.0, type=float)
    parser.add_argument("--min-predicted-iou", default=0.98, type=float)
    parser.add_argument("--min-stability-score", default=0.98, type=float)
    parser.add_argument("--seed", default=20260409, type=int)
    parser.add_argument(
        "--output-jsonl",
        default=(
            "output/stage3/sa1b_1p_reorg_threshold_examples_098/"
            "train_threshold_examples.jsonl"
        ),
        type=str,
    )
    return parser.parse_args()


def _make_record(
    image_info: Dict[str, Any],
    image_path: Path,
    ann: Dict[str, Any],
    width: int,
    height: int,
) -> Dict[str, Any]:
    area = float(ann.get("area", 0.0))
    predicted_iou = float(ann.get("predicted_iou", 0.0))
    stability_score = float(ann.get("stability_score", 0.0))
    x, y, w, h = [float(v) for v in ann["bbox"]]
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(width, int((x + w) + 0.999999))
    y1 = min(height, int((y + h) + 0.999999))
    # Threshold sampler emits quality examples, not model-generated text labels.
    label = ""
    return {
        "image_id": str(image_info["image_id"]),
        "image_path": str(image_path),
        "mask_id": str(ann["id"]),
        "bbox_xywh": [x, y, w, h],
        "area": area,
        "predicted_iou": predicted_iou,
        "stability_score": stability_score,
        "crop_box_xyxy": [float(x0), float(y0), float(x1), float(y1)],
        "segmentation": ann["segmentation"],
        "label": label,
        "normalized_label": label,
        "label_source": "threshold_sampler_no_text",
        "rejected": False,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.sa1b_root)
    ann_dir = root / "annotations" / args.split
    img_dir = root / "images" / args.split

    ann_files = sorted(ann_dir.glob("*.json"))
    rng = random.Random(args.seed)
    rng.shuffle(ann_files)

    selected: List[Dict[str, Any]] = []
    for ann_path in ann_files:
        with ann_path.open("r") as fopen:
            payload = json.load(fopen)

        image_info = payload.get("image", {})
        file_name = image_info.get("file_name")
        if not file_name:
            continue

        image_path = img_dir / file_name
        if not image_path.exists():
            continue

        width = int(image_info.get("width", 0))
        height = int(image_info.get("height", 0))

        annotations = list(payload.get("annotations", []))
        rng.shuffle(annotations)

        for ann in annotations:
            area = float(ann.get("area", 0.0))
            predicted_iou = float(ann.get("predicted_iou", 0.0))
            stability_score = float(ann.get("stability_score", 0.0))

            if area <= args.min_area:
                continue
            if predicted_iou <= args.min_predicted_iou:
                continue
            if stability_score <= args.min_stability_score:
                continue

            selected.append(
                _make_record(
                    image_info=image_info,
                    image_path=image_path,
                    ann=ann,
                    width=width,
                    height=height,
                )
            )
            if len(selected) >= args.num_examples:
                break

        if len(selected) >= args.num_examples:
            break

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w") as fopen:
        for record in selected:
            fopen.write(json.dumps(record) + "\n")

    print(
        json.dumps(
            {
                "output_jsonl": str(output_jsonl),
                "selected": len(selected),
                "seed": args.seed,
                "min_area_gt": args.min_area,
                "min_predicted_iou_gt": args.min_predicted_iou,
                "min_stability_score_gt": args.min_stability_score,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
