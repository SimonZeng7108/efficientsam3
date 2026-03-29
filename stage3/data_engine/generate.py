from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

from sam3.agent.client_llm import send_generate_request
from stage3.data_engine.annotations import (
    DEFAULT_MODEL_NAME,
    PROMPT_VERSION,
    RAW_SCHEMA_VERSION,
    area_to_fraction,
    bbox_xywh_to_normalized_xywh,
    bbox_xywh_to_xyxy,
    build_qwen_labeling_messages,
    is_generic_label,
    parse_model_json_response,
    phrase_word_count,
)

_LOCAL_MODEL = None
_LOCAL_PROCESSOR = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate raw image text-mask pair annotations from SA-1B with a VLM."
    )
    parser.add_argument(
        "--sa1b-root",
        default="data/SA-1B-2p",
        type=str,
        help="Root of the SA-1B subset with images/{split} and annotations/{split}.",
    )
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument(
        "--output-root",
        default="data/sa1b_stage3_pseudo",
        type=str,
        help="Base output directory for raw labels and prompt renders.",
    )
    parser.add_argument(
        "--server-url",
        default=None,
        type=str,
        help="OpenAI-compatible VLM endpoint.",
    )
    parser.add_argument(
        "--inference-backend",
        default="local_transformers",
        choices=["local_transformers", "openai_api", "stub"],
        help=(
            "local_transformers: HF Qwen3-VL; openai_api: --server-url; "
            "stub: deterministic JSON (no model — for pipeline / CI smoke tests)."
        ),
    )
    parser.add_argument("--api-key", default=None, type=str)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, type=str)
    parser.add_argument("--max-tokens", default=256, type=int)
    parser.add_argument(
        "--device-map",
        default="auto",
        type=str,
        help="Transformers device_map for local inference.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for local transformers inference.",
    )
    parser.add_argument("--start-index", default=0, type=int)
    parser.add_argument("--limit-images", default=None, type=int)
    parser.add_argument("--max-masks-per-image", default=None, type=int)
    parser.add_argument("--min-area", default=3000.0, type=float)
    parser.add_argument("--min-stability-score", default=0.95, type=float)
    parser.add_argument("--min-predicted-iou", default=0.88, type=float)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate records even if a mask id already exists in the raw jsonl.",
    )
    return parser.parse_args()


def _resolve_torch_dtype(dtype_name: str):
    import torch

    if dtype_name == "auto":
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    return getattr(torch, dtype_name)


def _ensure_local_model(model_name: str, device_map: str, dtype_name: str):
    global _LOCAL_MODEL, _LOCAL_PROCESSOR
    if _LOCAL_MODEL is not None and _LOCAL_PROCESSOR is not None:
        return _LOCAL_MODEL, _LOCAL_PROCESSOR

    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
    except ImportError as error:
        raise RuntimeError(
            "Local transformers inference requires `transformers` with Qwen3-VL support."
        ) from error

    torch_dtype = _resolve_torch_dtype(dtype_name)
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": True,
    }
    if device_map != "cpu":
        model_kwargs["attn_implementation"] = "sdpa"
    _LOCAL_MODEL = AutoModelForImageTextToText.from_pretrained(
        model_name,
        **model_kwargs,
    )
    _LOCAL_PROCESSOR = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    return _LOCAL_MODEL, _LOCAL_PROCESSOR


def _run_stub_vlm_response(image_id: str, mask_id: str) -> str:
    """Return a valid model-style JSON string without calling a VLM."""
    tail = mask_id[-8:] if len(mask_id) >= 8 else mask_id
    label = f"sa1b crop instance {tail}"
    payload = {
        "label": label,
        "confidence": 0.82,
        "ambiguous": False,
        "reject_reason": None,
    }
    return json.dumps(payload, ensure_ascii=False)


def _run_local_transformers_request(
    messages: List[Dict[str, Any]],
    model_name: str,
    max_tokens: int,
    device_map: str,
    dtype_name: str,
) -> Optional[str]:
    normalized_messages: List[Dict[str, Any]] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            normalized_content = [{"type": "text", "text": content}]
        else:
            normalized_content = content
        normalized_messages.append(
            {
                "role": message["role"],
                "content": normalized_content,
            }
        )
    model, processor = _ensure_local_model(
        model_name=model_name,
        device_map=device_map,
        dtype_name=dtype_name,
    )
    try:
        inputs = processor.apply_chat_template(
            normalized_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not output_text:
            return None
        return output_text[0]
    except Exception as error:
        print(f"Local transformers request failed: {error}")
        return None


def _annotation_files(root: Path, split: str) -> List[Path]:
    ann_dir = root / "annotations" / split
    return sorted(ann_dir.glob("*.json"))


def _image_path(root: Path, split: str, image_info: Dict[str, Any]) -> Path:
    return root / "images" / split / image_info["file_name"]


def _clamp_bbox_xyxy(
    bbox_xywh: Iterable[float],
    width: int,
    height: int,
    pad_ratio: float = 0.0,
    min_pad: int = 0,
) -> Tuple[int, int, int, int]:
    x, y, w, h = [float(v) for v in bbox_xywh]
    pad_x = max(min_pad, int(round(w * pad_ratio)))
    pad_y = max(min_pad, int(round(h * pad_ratio)))
    x0 = max(0, int(np.floor(x - pad_x)))
    y0 = max(0, int(np.floor(y - pad_y)))
    x1 = min(width, int(np.ceil(x + w + pad_x)))
    y1 = min(height, int(np.ceil(y + h + pad_y)))
    return x0, y0, x1, y1


def _crop_xyxy_to_xywh(crop_xyxy: Tuple[int, int, int, int]) -> List[float]:
    x0, y0, x1, y1 = crop_xyxy
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def _crop_image(
    image: Image.Image, crop_xyxy: Tuple[int, int, int, int]
) -> Image.Image:
    x0, y0, x1, y1 = crop_xyxy
    return image.crop((x0, y0, x1, y1)).convert("RGB")


def _load_existing_mask_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    existing = set()
    with path.open("r") as fopen:
        for line in fopen:
            line = line.strip()
            if not line:
                continue
            try:
                existing.add(str(json.loads(line)["mask_id"]))
            except (KeyError, json.JSONDecodeError):
                continue
    return existing


def _save_prompt_bundle(
    prompt_root: Path,
    image_id: str,
    mask_id: str,
    crop: Image.Image,
) -> Path:
    image_dir = prompt_root / str(image_id)
    image_dir.mkdir(parents=True, exist_ok=True)
    crop_path = image_dir / f"{mask_id}_crop.jpg"
    crop.save(crop_path, quality=95)
    return crop_path


def _build_record(
    image_info: Dict[str, Any],
    image_path: Path,
    ann_path: Path,
    ann: Dict[str, Any],
    crop_box_xyxy: Optional[Tuple[int, int, int, int]],
    mask_index: int,
    raw_response: str,
    label: str,
    confidence: float,
    ambiguous: bool,
    reject_reason: str,
    rejected: bool,
) -> Dict[str, Any]:
    bbox_xywh = [float(v) for v in ann["bbox"]]
    width = int(image_info["width"])
    height = int(image_info["height"])
    area_px = float(ann.get("area", 0.0))
    crop_box_xywh = (
        _crop_xyxy_to_xywh(crop_box_xyxy) if crop_box_xyxy is not None else []
    )
    return {
        "schema_version": RAW_SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "prompt_mode": "crop_only",
        "model_name": args.model,
        "image_id": str(image_info["image_id"]),
        "image_path": str(image_path),
        "annotation_path": str(ann_path),
        "mask_id": str(ann["id"]),
        "mask_index": mask_index,
        "width": width,
        "height": height,
        "bbox_xywh": bbox_xywh,
        "bbox_xyxy": bbox_xywh_to_xyxy(bbox_xywh),
        "bbox_norm_xywh": bbox_xywh_to_normalized_xywh(
            bbox_xywh,
            width=width,
            height=height,
        ),
        "crop_box_xywh": crop_box_xywh,
        "crop_box_xyxy": [float(v) for v in crop_box_xyxy]
        if crop_box_xyxy is not None
        else [],
        "point_coords": ann.get("point_coords"),
        "area": area_px,
        "area_frac": area_to_fraction(area_px, width=width, height=height),
        "predicted_iou": float(ann.get("predicted_iou", 0.0)),
        "stability_score": float(ann.get("stability_score", 0.0)),
        "segmentation": ann["segmentation"],
        "label": label,
        "normalized_label": label,
        "confidence": confidence,
        "ambiguous": ambiguous,
        "rejected": rejected,
        "reject_reason": reject_reason,
        "raw_response": raw_response,
    }


def _reject_record(reason: str) -> Tuple[str, float, bool, str, bool]:
    return "", 0.0, True, reason, True


def main() -> None:
    global args
    args = parse_args()
    sa1b_root = Path(args.sa1b_root)
    output_root = Path(args.output_root)
    raw_dir = output_root / "raw"
    prompt_dir = output_root / "prompt_renders" / args.split
    raw_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    raw_jsonl = raw_dir / f"{args.split}.jsonl"

    existing_mask_ids = set() if args.overwrite else _load_existing_mask_ids(raw_jsonl)
    ann_files = _annotation_files(sa1b_root, args.split)
    ann_files = ann_files[args.start_index :]
    if args.limit_images is not None:
        ann_files = ann_files[: args.limit_images]

    processed = 0
    skipped = 0
    filtered_min_area = 0
    with raw_jsonl.open("a") as fout:
        for ann_path in ann_files:
            with ann_path.open("r") as fopen:
                ann_data = json.load(fopen)
            image_info = ann_data["image"]
            image_path = _image_path(sa1b_root, args.split, image_info)
            if not image_path.exists():
                print(f"Missing image for annotation: {ann_path}", file=sys.stderr)
                continue

            image = Image.open(image_path).convert("RGB")
            annotations = ann_data.get("annotations", [])
            if args.max_masks_per_image is not None:
                annotations = annotations[: args.max_masks_per_image]

            for mask_index, ann in enumerate(annotations):
                mask_id = str(ann["id"])
                if mask_id in existing_mask_ids:
                    skipped += 1
                    continue
                if float(ann.get("area", 0.0)) < args.min_area:
                    filtered_min_area += 1
                    continue
                crop_xyxy = _clamp_bbox_xyxy(
                    ann["bbox"],
                    width=int(image_info["width"]),
                    height=int(image_info["height"]),
                )
                if float(ann.get("stability_score", 0.0)) < args.min_stability_score:
                    label, confidence, ambiguous, reject_reason, rejected = _reject_record("min_stability_score")
                    raw_response = ""
                elif float(ann.get("predicted_iou", 0.0)) < args.min_predicted_iou:
                    label, confidence, ambiguous, reject_reason, rejected = _reject_record("min_predicted_iou")
                    raw_response = ""
                else:
                    crop = _crop_image(image, crop_xyxy=crop_xyxy)
                    crop_path = _save_prompt_bundle(
                        prompt_root=prompt_dir,
                        image_id=str(image_info["image_id"]),
                        mask_id=mask_id,
                        crop=crop,
                    )
                    messages = build_qwen_labeling_messages(
                        crop_image_path=str(crop_path),
                    )
                    if args.inference_backend == "openai_api":
                        if not args.server_url:
                            raise RuntimeError(
                                "--server-url is required when --inference-backend=openai_api"
                            )
                        raw_response = send_generate_request(
                            messages=messages,
                            server_url=args.server_url,
                            model=args.model,
                            api_key=args.api_key,
                            max_tokens=args.max_tokens,
                        )
                    elif args.inference_backend == "stub":
                        raw_response = _run_stub_vlm_response(
                            image_id=str(image_info["image_id"]),
                            mask_id=mask_id,
                        )
                    else:
                        raw_response = _run_local_transformers_request(
                            messages=messages,
                            model_name=args.model,
                            max_tokens=args.max_tokens,
                            device_map=args.device_map,
                            dtype_name=args.torch_dtype,
                        )
                    if raw_response is None:
                        label, confidence, ambiguous, reject_reason, rejected = _reject_record(
                            "empty_response"
                        )
                        raw_response = ""
                    else:
                        try:
                            label, confidence, ambiguous, reject_reason = parse_model_json_response(
                                raw_response
                            )
                            rejected = (
                                ambiguous
                                or not label
                                or is_generic_label(label)
                                or phrase_word_count(label) > 10
                            )
                            if rejected and not reject_reason:
                                if not label:
                                    reject_reason = "empty_label"
                                elif is_generic_label(label):
                                    reject_reason = "generic_label"
                                else:
                                    reject_reason = "ambiguous_or_long"
                        except Exception as error:
                            label, confidence, ambiguous, reject_reason, rejected = _reject_record(
                                f"parse_error:{error}"
                            )
                record = _build_record(
                    image_info=image_info,
                    image_path=image_path,
                    ann_path=ann_path,
                    ann=ann,
                    crop_box_xyxy=crop_xyxy,
                    mask_index=mask_index,
                    raw_response=raw_response,
                    label=label,
                    confidence=confidence,
                    ambiguous=ambiguous,
                    reject_reason=reject_reason,
                    rejected=rejected,
                )
                fout.write(json.dumps(record) + "\n")
                fout.flush()
                processed += 1
                existing_mask_ids.add(mask_id)
                if processed % 100 == 0:
                    print(
                        (
                            f"processed={processed} skipped_existing={skipped} "
                            f"filtered_min_area={filtered_min_area} split={args.split}"
                        ),
                        file=sys.stderr,
                    )

    print(
        json.dumps(
            {
                "raw_jsonl": str(raw_jsonl),
                "processed": processed,
                "skipped_existing": skipped,
                "filtered_min_area": filtered_min_area,
                "min_area_threshold": args.min_area,
                "split": args.split,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
