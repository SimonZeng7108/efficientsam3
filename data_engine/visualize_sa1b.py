from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from pycocotools import mask as mask_utils

try:
    from stage3.data_engine.annotations import (
        MIN_SCREENING_AREA,
        MIN_SCREENING_PREDICTED_IOU,
        MIN_SCREENING_STABILITY_SCORE,
        visualize_annotation_example,
    )
except ModuleNotFoundError:
    # Allow running this script without importing the full stage3 package tree.
    annotations_path = Path(__file__).resolve().with_name("annotations.py")
    spec = importlib.util.spec_from_file_location(
        "stage3_data_engine_annotations",
        annotations_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load annotations module from {annotations_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    MIN_SCREENING_AREA = module.MIN_SCREENING_AREA
    MIN_SCREENING_PREDICTED_IOU = module.MIN_SCREENING_PREDICTED_IOU
    MIN_SCREENING_STABILITY_SCORE = module.MIN_SCREENING_STABILITY_SCORE
    visualize_annotation_example = module.visualize_annotation_example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize SA-1B annotation examples with grouped annotation/query context."
    )
    parser.add_argument("--sa1b-root", default="data/sa-1b-1p_reorg", type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--num-examples", default=10, type=int)
    parser.add_argument("--raw-jsonl", default=None, type=str)
    parser.add_argument(
        "--annotation-source",
        default="auto",
        choices=["auto", "enhanced", "text", "base"],
        help=(
            "Annotation file source when --raw-jsonl is not provided. "
            "auto: prefer *_enhanced.json, then *_text.json, else base *.json"
        ),
    )
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="Include rejected/ambiguous rows when visualizing from --raw-jsonl.",
    )
    parser.add_argument(
        "--min-area",
        default=MIN_SCREENING_AREA,
        type=float,
        help=(
            "Skip records whose area (px) is at or below this threshold. "
            f"Values below {MIN_SCREENING_AREA:.0f} are clamped."
        ),
    )
    parser.add_argument(
        "--min-predicted-iou",
        default=MIN_SCREENING_PREDICTED_IOU,
        type=float,
        help=(
            "Skip records whose predicted_iou is at or below this threshold. "
            f"Values below {MIN_SCREENING_PREDICTED_IOU:.2f} are clamped."
        ),
    )
    parser.add_argument(
        "--min-stability-score",
        default=MIN_SCREENING_STABILITY_SCORE,
        type=float,
        help=(
            "Skip records whose stability_score is at or below this threshold. "
            f"Values below {MIN_SCREENING_STABILITY_SCORE:.2f} are clamped."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="output/stage3/data_engine_sa1b_examples",
        type=str,
    )
    parser.add_argument(
        "--pre_processed",
        action="store_true",
        help=(
            "Visualize from prebuilt *_enhanced.json files directly. "
            "No raw-jsonl label resolution is used in this mode."
        ),
    )
    parser.add_argument(
        "--require-label",
        action="store_true",
        help=(
            "When visualizing from --raw-jsonl, skip records whose resolved label "
            "would be 'annotation pending'."
        ),
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


def _iter_latest_raw_records(path: Path) -> Iterable[Dict[str, Any]]:
    latest_by_key: Dict[str, tuple[int, Dict[str, Any]]] = {}
    for line_index, record in enumerate(_iter_jsonl(path), start=1):
        image_id = str(record.get("image_id") or "").strip()
        mask_id = str(record.get("mask_id") or "").strip()
        if not mask_id:
            continue
        key = f"{image_id}:{mask_id}" if image_id else mask_id
        latest_by_key[key] = (line_index, record)

    for _, record in sorted(latest_by_key.values(), key=lambda item: item[0]):
        yield record


def _decode_mask(segmentation: dict) -> object:
    if isinstance(segmentation.get("counts"), list):
        rle = mask_utils.frPyObjects(segmentation, segmentation["size"][0], segmentation["size"][1])
    else:
        rle = segmentation
    mask = mask_utils.decode(rle)
    if getattr(mask, "ndim", 0) == 3:
        mask = mask[..., 0]
    return mask.astype(bool)


def _mask_bbox_xyxy_from_segmentation(segmentation: Dict[str, Any]) -> List[float]:
    if not isinstance(segmentation, dict):
        return []
    if "counts" not in segmentation or "size" not in segmentation:
        return []

    size = segmentation.get("size", [])
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        return []

    seg_h, seg_w = int(size[0]), int(size[1])
    counts = segmentation.get("counts")
    if isinstance(counts, list):
        rle = mask_utils.frPyObjects(segmentation, seg_h, seg_w)
        if isinstance(rle, list):
            if not rle:
                return []
            rle = mask_utils.merge(rle)
    else:
        rle = segmentation

    bbox = mask_utils.toBbox(rle)
    if hasattr(bbox, "tolist"):
        bbox = bbox.tolist()
    if isinstance(bbox, list) and bbox and isinstance(bbox[0], list):
        bbox = bbox[0]
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return []

    x, y, w, h = [float(v) for v in bbox]
    if w <= 0.0 or h <= 0.0:
        return []
    return [x, y, x + w, y + h]


def _xywh_to_xyxy(bbox_xywh: Iterable[float]) -> List[float]:
    x, y, w, h = [float(v) for v in bbox_xywh]
    return [x, y, x + w, y + h]


def _annotation_bbox_xywh(annotation: Dict[str, Any]) -> List[float]:
    if annotation.get("bbox_xywh"):
        return [float(v) for v in annotation["bbox_xywh"]]
    if annotation.get("bbox"):
        return [float(v) for v in annotation["bbox"]]
    return []


def _annotation_crop_box_xyxy(
    annotation: Dict[str, Any],
    raw_record: Dict[str, Any],
) -> List[float]:
    if raw_record.get("crop_box_xyxy"):
        return [float(v) for v in raw_record["crop_box_xyxy"]]
    if annotation.get("crop_box_xyxy"):
        return [float(v) for v in annotation["crop_box_xyxy"]]
    if annotation.get("crop_box_xywh"):
        return _xywh_to_xyxy(annotation["crop_box_xywh"])
    if annotation.get("crop_box"):
        return _xywh_to_xyxy(annotation["crop_box"])
    return []


def _annotation_point_coords(
    annotation: Dict[str, Any],
    raw_record: Dict[str, Any],
) -> List[List[float]]:
    points = annotation.get("point_coords")
    if points is None:
        points = raw_record.get("point_coords")
    if points is None:
        return []

    if (
        isinstance(points, (list, tuple))
        and len(points) == 2
        and all(isinstance(value, (int, float)) for value in points)
    ):
        return [[float(points[0]), float(points[1])]]

    normalized: List[List[float]] = []
    if isinstance(points, (list, tuple)):
        for point in points:
            if (
                isinstance(point, (list, tuple))
                and len(point) >= 2
                and isinstance(point[0], (int, float))
                and isinstance(point[1], (int, float))
            ):
                normalized.append([float(point[0]), float(point[1])])
    return normalized


def _annotation_mask_center_xy(
    annotation: Dict[str, Any],
    raw_record: Dict[str, Any],
) -> List[float]:
    center = annotation.get("mask_center_xy")
    if center is None:
        center = raw_record.get("mask_center_xy")
    if (
        isinstance(center, (list, tuple))
        and len(center) >= 2
        and isinstance(center[0], (int, float))
        and isinstance(center[1], (int, float))
    ):
        return [float(center[0]), float(center[1])]
    return []


def _annotation_mask_sample_points_xy(
    annotation: Dict[str, Any],
    raw_record: Dict[str, Any],
) -> List[List[float]]:
    points = annotation.get("mask_sample_points_xy")
    if points is None:
        points = raw_record.get("mask_sample_points_xy")
    if points is None:
        return []

    normalized: List[List[float]] = []
    if isinstance(points, (list, tuple)):
        for point in points:
            if (
                isinstance(point, (list, tuple))
                and len(point) >= 2
                and isinstance(point[0], (int, float))
                and isinstance(point[1], (int, float))
            ):
                normalized.append([float(point[0]), float(point[1])])
    return normalized


def _annotation_mask_bbox_xyxy(
    annotation: Dict[str, Any],
    raw_record: Dict[str, Any],
) -> List[float]:
    if raw_record.get("mask_bbox_xyxy"):
        return [float(v) for v in raw_record["mask_bbox_xyxy"]]
    if annotation.get("mask_bbox_xyxy"):
        return [float(v) for v in annotation["mask_bbox_xyxy"]]
    if raw_record.get("mask_bbox_xywh"):
        return _xywh_to_xyxy(raw_record["mask_bbox_xywh"])
    if annotation.get("mask_bbox_xywh"):
        return _xywh_to_xyxy(annotation["mask_bbox_xywh"])

    segmentation = annotation.get("segmentation")
    if segmentation is None:
        segmentation = raw_record.get("segmentation", {})
    return _mask_bbox_xyxy_from_segmentation(segmentation)


def _annotation_text(
    annotation: Dict[str, Any],
    raw_record: Dict[str, Any],
    text_annotation: Dict[str, Any] | None = None,
) -> str:
    """Resolve the best available text label from all sources.

    Priority (highest to lowest):
    1. VLM-generated label from _text.json (written by generate.py, highest quality)
    2. VLM-generated label direct from the raw JSONL record
    3. Label embedded in the annotation dict (e.g. when reading _text.json directly)
    Placeholder / stub labels are filtered at each level.
    """
    text_annotation = text_annotation or {}
    return _choose_text_label(
        text_annotation.get("label_10"),
        # _text.json labels come from the VLM acceptance pass and are highest quality
        text_annotation.get("label"),
        text_annotation.get("normalized_label"),
        raw_record.get("label_10"),
        # raw JSONL record label (from VLM inference, may be stub)
        raw_record.get("label"),
        raw_record.get("normalized_label"),
        annotation.get("label_10"),
        # label embedded in base/text annotation struct
        annotation.get("label"),
        annotation.get("text_label"),
        annotation.get("prompt_label"),
        annotation.get("normalized_label"),
    )


def _annotation_label_triplet(
    annotation: Dict[str, Any],
    raw_record: Dict[str, Any],
    text_annotation: Dict[str, Any] | None = None,
) -> tuple[str, str, str]:
    text_annotation = text_annotation or {}
    label_10 = _annotation_text(
        annotation=annotation,
        raw_record=raw_record,
        text_annotation=text_annotation,
    )
    label_5 = _choose_text_label(
        text_annotation.get("label_5"),
        raw_record.get("label_5"),
        annotation.get("label_5"),
    )
    label_2 = _choose_text_label(
        text_annotation.get("label_2"),
        raw_record.get("label_2"),
        annotation.get("label_2"),
    )
    return label_10, label_5, label_2


def _is_raw_record_visualizable(
    record: Dict[str, Any],
    include_rejected: bool,
) -> bool:
    if not include_rejected:
        if bool(record.get("rejected", False)):
            return False
        if bool(record.get("ambiguous", False)):
            return False
    return True


def _effective_gate_thresholds(
    min_area: float,
    min_predicted_iou: float,
    min_stability_score: float,
) -> tuple[float, float, float]:
    return (
        max(float(min_area), MIN_SCREENING_AREA),
        max(float(min_predicted_iou), MIN_SCREENING_PREDICTED_IOU),
        max(float(min_stability_score), MIN_SCREENING_STABILITY_SCORE),
    )


def _passes_quality_gates(
    area: Any,
    predicted_iou: Any,
    stability_score: Any,
    min_area: float,
    min_predicted_iou: float,
    min_stability_score: float,
) -> bool:
    if min_area > 0.0 and float(area or 0.0) <= min_area:
        return False
    if min_predicted_iou > 0.0 and float(predicted_iou or 0.0) <= min_predicted_iou:
        return False
    if min_stability_score > 0.0 and float(stability_score or 0.0) <= min_stability_score:
        return False
    return True


def _annotation_files(ann_dir: Path, source: str) -> List[Path]:
    enhanced_files = sorted(ann_dir.glob("*_enhanced.json"))
    text_files = sorted(ann_dir.glob("*_text.json"))
    base_files = sorted(
        path
        for path in ann_dir.glob("*.json")
        if not path.name.endswith("_text.json")
        and not path.name.endswith("_enhanced.json")
    )
    if source == "enhanced":
        return enhanced_files
    if source == "text":
        return text_files
    if source == "base":
        return base_files
    if enhanced_files:
        return enhanced_files
    if text_files:
        return text_files
    return base_files


def _is_placeholder_label(text: Any) -> bool:
    """Return True for known programmatically-generated stub/placeholder labels."""
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    # Labels produced by the stub VLM backend (_run_stub_vlm_response)
    if normalized.startswith("sa1b crop instance"):
        return True
    # Legacy quality-sampler placeholder strings
    if normalized.startswith("mask with iou"):
        return True
    return False


def _choose_text_label(*candidates: Any) -> str:
    placeholder_fallback = ""
    for candidate in candidates:
        text = str(candidate or "").strip()
        if not text:
            continue
        if _is_placeholder_label(text):
            if not placeholder_fallback:
                placeholder_fallback = text
            continue
        return text
    if placeholder_fallback:
        return placeholder_fallback
    return "annotation pending"


def _raw_record_annotation_file(record: Dict[str, Any], ann_dir: Path) -> Path | None:
    # Strategy 1: reconstruct from image_path stem + ann_dir (prefer current root)
    image_path_value = record.get("image_path")
    if image_path_value:
        image_stem = Path(str(image_path_value)).stem
        candidate = ann_dir / f"{image_stem}.json"
        if candidate.exists():
            return candidate

    # Strategy 2: reconstruct from image_id + ann_dir (prefer current root)
    image_id = str(record.get("image_id") or "").strip()
    if image_id:
        candidate = ann_dir / f"sa_{image_id}.json"
        if candidate.exists():
            return candidate

    # Strategy 3: use annotation_path from the record directly (fallback)
    annotation_path_value = record.get("annotation_path")
    if annotation_path_value:
        direct = Path(str(annotation_path_value))
        if direct.is_absolute() and direct.exists():
            return direct
        relative = Path.cwd() / direct
        if relative.exists():
            return relative

    return None


def _raw_record_image_file(record: Dict[str, Any], img_dir: Path) -> Path | None:
    image_path_value = record.get("image_path")
    if image_path_value:
        image_name = Path(str(image_path_value)).name
        candidate = img_dir / image_name
        if candidate.exists():
            return candidate

    image_id = str(record.get("image_id") or "").strip()
    if image_id:
        candidate = img_dir / f"sa_{image_id}.jpg"
        if candidate.exists():
            return candidate

    if image_path_value:
        direct = Path(str(image_path_value))
        if direct.is_absolute() and direct.exists():
            return direct
        relative = Path.cwd() / direct
        if relative.exists():
            return relative

    return None


def _raw_record_annotation_fallback(
    record: Dict[str, Any],
    ann_dir: Path,
    cache: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    annotation_file = _raw_record_annotation_file(record=record, ann_dir=ann_dir)
    if annotation_file is None:
        return {}

    cache_key = str(annotation_file)
    annotation_by_id = cache.get(cache_key)
    if annotation_by_id is None:
        with annotation_file.open("r") as fopen:
            payload = json.load(fopen)
        annotation_by_id = {
            str(ann.get("id")): ann for ann in payload.get("annotations", [])
        }
        cache[cache_key] = annotation_by_id

    return annotation_by_id.get(str(record.get("mask_id")), {})


def _raw_record_text_annotation_fallback(
    record: Dict[str, Any],
    ann_dir: Path,
    cache: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    annotation_file = _raw_record_annotation_file(record=record, ann_dir=ann_dir)
    if annotation_file is None:
        return {}

    # Prefer enhanced labels first, then legacy _text labels.
    label_file_candidates = [
        annotation_file.with_name(f"{annotation_file.stem}_enhanced.json"),
        annotation_file.with_name(f"{annotation_file.stem}_text.json"),
    ]
    label_file = next((path for path in label_file_candidates if path.exists()), None)
    if label_file is None:
        return {}

    cache_key = str(label_file)
    annotation_by_id = cache.get(cache_key)
    if annotation_by_id is None:
        with label_file.open("r") as fopen:
            payload = json.load(fopen)
        annotation_by_id = {
            str(ann.get("id")): ann for ann in payload.get("annotations", [])
        }
        cache[cache_key] = annotation_by_id

    return annotation_by_id.get(str(record.get("mask_id")), {})


def _render_from_raw_jsonl(
    raw_jsonl: Path,
    ann_dir: Path,
    img_dir: Path,
    output_dir: Path,
    num_examples: int,
    include_rejected: bool,
    min_area: float = 0.0,
    min_predicted_iou: float = 0.0,
    min_stability_score: float = 0.0,
    require_label: bool = False,
) -> int:
    annotation_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
    text_annotation_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
    rendered = 0
    for record in _iter_latest_raw_records(raw_jsonl):
        if rendered >= num_examples:
            break
        if not _is_raw_record_visualizable(
            record,
            include_rejected=include_rejected,
        ):
            continue

        # Always load the original base JSON annotation as the authoritative metadata
        # source. The JSONL record may have been generated with different flags or
        # with the stub backend, so the base JSON is the ground truth.
        base_ann = _raw_record_annotation_fallback(
            record=record,
            ann_dir=ann_dir,
            cache=annotation_cache,
        )
        # Load the _text.json annotation for VLM-generated labels (highest quality)
        text_ann = _raw_record_text_annotation_fallback(
            record=record,
            ann_dir=ann_dir,
            cache=text_annotation_cache,
        )

        image_path = _raw_record_image_file(record=record, img_dir=img_dir)
        if image_path is None:
            continue

        # Use base JSON bbox as authoritative; fall back to JSONL record
        bbox_xywh = _annotation_bbox_xywh(base_ann)
        if not bbox_xywh:
            bbox_xywh = [float(v) for v in record.get("bbox_xywh", [])]
        if not bbox_xywh:
            continue

        # Use base JSON segmentation as authoritative; fall back to JSONL record
        segmentation = base_ann.get("segmentation") or record.get("segmentation")
        if segmentation is None:
            continue

        # Resolve each metadata field: base JSON > JSONL record
        area = base_ann.get("area") if base_ann.get("area") is not None else record.get("area")
        predicted_iou = (
            base_ann.get("predicted_iou")
            if base_ann.get("predicted_iou") is not None
            else record.get("predicted_iou")
        )
        stability_score = (
            base_ann.get("stability_score")
            if base_ann.get("stability_score") is not None
            else record.get("stability_score")
        )
        if not _passes_quality_gates(
            area=area,
            predicted_iou=predicted_iou,
            stability_score=stability_score,
            min_area=min_area,
            min_predicted_iou=min_predicted_iou,
            min_stability_score=min_stability_score,
        ):
            continue
        mask_bbox_xyxy = _annotation_mask_bbox_xyxy(
            annotation=base_ann,
            raw_record=record,
        )
        # For Qwen crop preview, always prefer tight mask bounds.
        crop_box_xyxy = mask_bbox_xyxy or _annotation_crop_box_xyxy(
            annotation=base_ann,
            raw_record=record,
        )

        label_10, label_5, label_2 = _annotation_label_triplet(
            annotation=base_ann,
            raw_record=record,
            text_annotation=text_ann,
        )
        if require_label and (
            label_10 == "annotation pending"
            or _is_placeholder_label(label_10)
        ):
            continue

        visualize_annotation_example(
            image_path=str(image_path),
            annotations=[
                {
                    "bbox_xywh": bbox_xywh,
                    # Use "label" key so _annotation_structure_text reads it directly
                    "label": label_10,
                    "label_10": label_10,
                    "label_5": label_5,
                    "label_2": label_2,
                    "mask_id": str(record["mask_id"]),
                    "image_id": str(record.get("image_id") or ""),
                    "area": area,
                    "predicted_iou": predicted_iou,
                    "stability_score": stability_score,
                    "point_coords": _annotation_point_coords(
                        annotation=base_ann,
                        raw_record=record,
                    ),
                    "mask_center_xy": _annotation_mask_center_xy(
                        annotation=base_ann,
                        raw_record=record,
                    ),
                    "mask_sample_points_xy": _annotation_mask_sample_points_xy(
                        annotation=base_ann,
                        raw_record=record,
                    ),
                    "mask_bbox_xyxy": mask_bbox_xyxy,
                    "crop_box_xyxy": crop_box_xyxy,
                    "mask": _decode_mask(segmentation),
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
    requested_min_area = float(args.min_area)
    requested_min_predicted_iou = float(args.min_predicted_iou)
    requested_min_stability_score = float(args.min_stability_score)
    (
        args.min_area,
        args.min_predicted_iou,
        args.min_stability_score,
    ) = _effective_gate_thresholds(
        min_area=requested_min_area,
        min_predicted_iou=requested_min_predicted_iou,
        min_stability_score=requested_min_stability_score,
    )
    if (
        args.min_area != requested_min_area
        or args.min_predicted_iou != requested_min_predicted_iou
        or args.min_stability_score != requested_min_stability_score
    ):
        print(
            (
                "Clamped screening thresholds to hard minima: "
                f"min_area={args.min_area:.0f}, "
                f"min_predicted_iou={args.min_predicted_iou:.2f}, "
                f"min_stability_score={args.min_stability_score:.2f}"
            ),
            file=sys.stderr,
        )
    sa1b_root = Path(args.sa1b_root)
    ann_dir = sa1b_root / "annotations" / args.split
    img_dir = sa1b_root / "images" / args.split
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_jsonl_candidate = Path(args.raw_jsonl) if args.raw_jsonl else None
    if raw_jsonl_candidate is None:
        default_raw_jsonl = Path("data/sa1b_stage3_pseudo/raw") / f"{args.split}.jsonl"
        if default_raw_jsonl.exists():
            raw_jsonl_candidate = default_raw_jsonl

    raw_record_by_mask_id = _load_raw_record_by_mask_id(
        str(raw_jsonl_candidate) if raw_jsonl_candidate is not None else None
    )

    if args.pre_processed:
        ann_files = sorted(ann_dir.glob("*_enhanced.json"))
        rendered = 0
        for ann_file in ann_files:
            if rendered >= args.num_examples:
                break
            with ann_file.open("r") as fopen:
                payload = json.load(fopen)

            image_info = payload.get("image", {})
            file_name = image_info.get("file_name")
            if not file_name:
                continue
            image_path = img_dir / file_name
            if not image_path.exists():
                continue

            for ann in payload.get("annotations", []):
                if rendered >= args.num_examples:
                    break

                label_10, label_5, label_2 = _annotation_label_triplet(
                    annotation=ann,
                    raw_record={},
                )
                if args.require_label and (
                    label_10 == "annotation pending"
                    or _is_placeholder_label(label_10)
                ):
                    continue

                bbox_xywh = _annotation_bbox_xywh(ann)
                if not bbox_xywh:
                    continue
                if "segmentation" not in ann:
                    continue

                if not _passes_quality_gates(
                    area=ann.get("area"),
                    predicted_iou=ann.get("predicted_iou"),
                    stability_score=ann.get("stability_score"),
                    min_area=args.min_area,
                    min_predicted_iou=args.min_predicted_iou,
                    min_stability_score=args.min_stability_score,
                ):
                    continue

                raw_record: Dict[str, Any] = {}
                mask_bbox_xyxy = _annotation_mask_bbox_xyxy(
                    annotation=ann,
                    raw_record=raw_record,
                )
                crop_box_xyxy = mask_bbox_xyxy or _annotation_crop_box_xyxy(
                    annotation=ann,
                    raw_record=raw_record,
                )
                mask_id = str(ann.get("id", ""))
                if not mask_id:
                    continue

                visualize_annotation_example(
                    image_path=str(image_path),
                    annotations=[
                        {
                            "bbox_xywh": bbox_xywh,
                            "label": label_10,
                            "label_10": label_10,
                            "label_5": label_5,
                            "label_2": label_2,
                            "mask_id": mask_id,
                            "image_id": str(image_info.get("image_id") or ""),
                            "area": ann.get("area"),
                            "predicted_iou": ann.get("predicted_iou"),
                            "stability_score": ann.get("stability_score"),
                            "point_coords": _annotation_point_coords(
                                annotation=ann,
                                raw_record=raw_record,
                            ),
                            "mask_center_xy": _annotation_mask_center_xy(
                                annotation=ann,
                                raw_record=raw_record,
                            ),
                            "mask_sample_points_xy": _annotation_mask_sample_points_xy(
                                annotation=ann,
                                raw_record=raw_record,
                            ),
                            "mask_bbox_xyxy": mask_bbox_xyxy,
                            "crop_box_xyxy": crop_box_xyxy,
                            "mask": _decode_mask(ann["segmentation"]),
                        }
                    ],
                    output_path=str(output_dir / f"{Path(file_name).stem}_{mask_id}.jpg"),
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
                    "source": "pre_processed_enhanced_json",
                    "annotation_files_considered": len(ann_files),
                    "min_area_threshold": args.min_area,
                    "min_predicted_iou_threshold": args.min_predicted_iou,
                    "min_stability_score_threshold": args.min_stability_score,
                },
                indent=2,
            )
        )
        return

    if raw_jsonl_candidate is not None:
        raw_jsonl_path = raw_jsonl_candidate
        if raw_jsonl_path.exists():
            rendered = _render_from_raw_jsonl(
                raw_jsonl=raw_jsonl_path,
                ann_dir=ann_dir,
                img_dir=img_dir,
                output_dir=output_dir,
                num_examples=args.num_examples,
                include_rejected=args.include_rejected,
                min_area=args.min_area,
                min_predicted_iou=args.min_predicted_iou,
                min_stability_score=args.min_stability_score,
                require_label=args.require_label,
            )
            print(
                json.dumps(
                    {
                        "output_dir": str(output_dir),
                        "rendered_examples": rendered,
                        "split": args.split,
                        "source": str(raw_jsonl_path),
                        "include_rejected": args.include_rejected,
                        "min_area_threshold": args.min_area,
                        "min_predicted_iou_threshold": args.min_predicted_iou,
                        "min_stability_score_threshold": args.min_stability_score,
                    },
                    indent=2,
                )
            )
            return

    ann_files = _annotation_files(ann_dir=ann_dir, source=args.annotation_source)
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
            label_10, label_5, label_2 = _annotation_label_triplet(
                annotation=ann,
                raw_record=raw_record,
            )
            if args.require_label and (
                label_10 == "annotation pending"
                or _is_placeholder_label(label_10)
            ):
                continue
            bbox_xywh = _annotation_bbox_xywh(ann)
            if not bbox_xywh:
                continue
            if "segmentation" not in ann:
                continue

            area_value = ann.get("area", raw_record.get("area"))
            predicted_iou_value = ann.get(
                "predicted_iou",
                raw_record.get("predicted_iou"),
            )
            stability_score_value = ann.get(
                "stability_score",
                raw_record.get("stability_score"),
            )
            if not _passes_quality_gates(
                area=area_value,
                predicted_iou=predicted_iou_value,
                stability_score=stability_score_value,
                min_area=args.min_area,
                min_predicted_iou=args.min_predicted_iou,
                min_stability_score=args.min_stability_score,
            ):
                continue

            mask_bbox_xyxy = _annotation_mask_bbox_xyxy(
                annotation=ann,
                raw_record=raw_record,
            )
            # Keep preview consistent with tight mask crop.
            crop_box_xyxy = mask_bbox_xyxy or _annotation_crop_box_xyxy(
                annotation=ann,
                raw_record=raw_record,
            )
            visualize_annotation_example(
                image_path=str(image_path),
                annotations=[
                    {
                        "bbox_xywh": bbox_xywh,
                        "label": label_10,
                        "label_10": label_10,
                        "label_5": label_5,
                        "label_2": label_2,
                        "mask_id": mask_id,
                        "image_id": str(image_info.get("image_id") or ""),
                        "area": area_value,
                        "predicted_iou": predicted_iou_value,
                        "stability_score": stability_score_value,
                        "point_coords": _annotation_point_coords(
                            annotation=ann,
                            raw_record=raw_record,
                        ),
                        "mask_center_xy": _annotation_mask_center_xy(
                            annotation=ann,
                            raw_record=raw_record,
                        ),
                        "mask_sample_points_xy": _annotation_mask_sample_points_xy(
                            annotation=ann,
                            raw_record=raw_record,
                        ),
                        "mask_bbox_xyxy": mask_bbox_xyxy,
                        "crop_box_xyxy": crop_box_xyxy,
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
                "annotation_source": args.annotation_source,
                "annotation_files_considered": len(ann_files),
                "min_area_threshold": args.min_area,
                "min_predicted_iou_threshold": args.min_predicted_iou,
                "min_stability_score_threshold": args.min_stability_score,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
