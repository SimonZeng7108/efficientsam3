from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    from stage3.data_engine.annotations import (
        DEFAULT_MODEL_NAME,
        MIN_SCREENING_AREA,
        MIN_SCREENING_PREDICTED_IOU,
        MIN_SCREENING_STABILITY_SCORE,
        PROMPT_VERSION,
        RAW_SCHEMA_VERSION,
        area_to_fraction,
        bbox_xywh_to_normalized_xywh,
        bbox_xywh_to_xyxy,
        build_qwen_labeling_messages,
        is_generic_label,
        normalize_label,
        phrase_word_count,
    )
except ModuleNotFoundError:
    # Allow running data_engine scripts without importing the full stage3 package tree.
    annotations_path = Path(__file__).resolve().with_name("annotations.py")
    spec = importlib.util.spec_from_file_location(
        "stage3_data_engine_annotations",
        annotations_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load annotations module from {annotations_path}")
    annotations_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = annotations_module
    spec.loader.exec_module(annotations_module)

    DEFAULT_MODEL_NAME = annotations_module.DEFAULT_MODEL_NAME
    MIN_SCREENING_AREA = annotations_module.MIN_SCREENING_AREA
    MIN_SCREENING_PREDICTED_IOU = annotations_module.MIN_SCREENING_PREDICTED_IOU
    MIN_SCREENING_STABILITY_SCORE = annotations_module.MIN_SCREENING_STABILITY_SCORE
    PROMPT_VERSION = annotations_module.PROMPT_VERSION
    RAW_SCHEMA_VERSION = annotations_module.RAW_SCHEMA_VERSION
    area_to_fraction = annotations_module.area_to_fraction
    bbox_xywh_to_normalized_xywh = annotations_module.bbox_xywh_to_normalized_xywh
    bbox_xywh_to_xyxy = annotations_module.bbox_xywh_to_xyxy
    build_qwen_labeling_messages = annotations_module.build_qwen_labeling_messages
    is_generic_label = annotations_module.is_generic_label
    normalize_label = annotations_module.normalize_label
    phrase_word_count = annotations_module.phrase_word_count

_LOCAL_MODEL = None
_LOCAL_PROCESSOR = None
_TEXT_REWRITE_MODEL = None
_TEXT_REWRITE_TOKENIZER = None
_TEXT_REWRITE_MODEL_NAME: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate raw image text-mask pair annotations from SA-1B with a VLM."
    )
    parser.add_argument(
        "--sa1b-root",
        default="data/sa-1b-1p_reorg",
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
        "--inference-backend",
        default="local_transformers",
        choices=["local_transformers", "stub"],
        help=(
            "local_transformers: HF Qwen3-VL; "
            "stub: deterministic JSON (no model — for pipeline / CI smoke tests)."
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, type=str)
    parser.add_argument(
        "--text-rewrite-model",
        default="Qwen/Qwen3.5-2B",
        type=str,
        help="Text-only model used to rewrite accepted labels into shorter prompt variants.",
    )
    parser.add_argument("--max-tokens", default=256, type=int)
    parser.add_argument(
        "--rewrite-max-tokens",
        default=96,
        type=int,
        help="Maximum output tokens for the text rewrite model.",
    )
    parser.add_argument(
        "--disable-label-rewrite",
        action="store_true",
        help="Skip the text-only label rewrite pass.",
    )
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
    parser.add_argument(
        "--min-area",
        default=MIN_SCREENING_AREA,
        type=float,
        help=(
            "Minimum mask area (px) for screening. "
            f"Values below {MIN_SCREENING_AREA:.0f} are clamped."
        ),
    )
    parser.add_argument(
        "--min-stability-score",
        default=MIN_SCREENING_STABILITY_SCORE,
        type=float,
        help=(
            "Minimum stability_score for screening. "
            f"Values below {MIN_SCREENING_STABILITY_SCORE:.2f} are clamped."
        ),
    )
    parser.add_argument(
        "--min-predicted-iou",
        default=MIN_SCREENING_PREDICTED_IOU,
        type=float,
        help=(
            "Minimum predicted_iou for screening. "
            f"Values below {MIN_SCREENING_PREDICTED_IOU:.2f} are clamped."
        ),
    )
    parser.add_argument(
        "--crop-box-source",
        default="mask",
        choices=["mask", "bbox"],
        help=(
            "Crop source for VLM input: 'mask' uses a tight box from segmentation RLE, "
            "'bbox' uses the annotation bbox."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate records even if a mask id already exists in the raw jsonl.",
    )
    parser.add_argument(
        "--num-mask-sample-points",
        default=10,
        type=int,
        help=(
            "Number of random points sampled from inside each mask and stored in raw records. "
            "These points complement the original SA-1B prompt point."
        ),
    )
    parser.add_argument(
        "--write-enhanced-annotations",
        dest="write_enhanced_annotations",
        action="store_true",
        help=(
            "Write per-image _enhanced.json files next to source annotations. "
            "Each output copies the source JSON and injects gated pseudo labels + sampled "
            "mask points into each accepted mask annotation."
        ),
    )
    parser.add_argument(
        "--no-write-enhanced-annotations",
        dest="write_enhanced_annotations",
        action="store_false",
        help="Disable writing per-image _enhanced.json outputs.",
    )
    # Backward-compatible aliases.
    parser.add_argument(
        "--write-text-annotations",
        dest="write_enhanced_annotations",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-write-text-annotations",
        dest="write_enhanced_annotations",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(write_enhanced_annotations=True)
    return parser.parse_args()


def _resolve_torch_dtype(dtype_name: str):
    import torch

    if dtype_name == "auto":
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    return getattr(torch, dtype_name)


def _effective_gate_thresholds(
    min_area: float,
    min_predicted_iou: float,
    min_stability_score: float,
) -> Tuple[float, float, float]:
    return (
        max(float(min_area), MIN_SCREENING_AREA),
        max(float(min_predicted_iou), MIN_SCREENING_PREDICTED_IOU),
        max(float(min_stability_score), MIN_SCREENING_STABILITY_SCORE),
    )


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
    """Return a valid model-style JSON string without calling a VLM.

    Records generated with the stub backend are marked with a pseudo-label of the
    form ``"sa1b crop instance <id>"`` which the visualizer treats as a placeholder
    (no real text label was produced).  These records are intentionally accepted
    (not rejected) so the full pipeline can be smoke-tested end-to-end.
    """
    tail = mask_id[-8:] if len(mask_id) >= 8 else mask_id
    label = f"sa1b crop instance {tail}"
    payload = {
        "label": label,
        "confidence": 0.82,
        "ambiguous": False,
        "reject_reason": None,
        "label_source": "stub_backend",
    }
    return json.dumps(payload, ensure_ascii=False)


def _ensure_local_text_rewrite_model(
    model_name: str,
    device_map: str,
    dtype_name: str,
):
    global _TEXT_REWRITE_MODEL, _TEXT_REWRITE_TOKENIZER, _TEXT_REWRITE_MODEL_NAME
    if (
        _TEXT_REWRITE_MODEL is not None
        and _TEXT_REWRITE_TOKENIZER is not None
        and _TEXT_REWRITE_MODEL_NAME == model_name
    ):
        return _TEXT_REWRITE_MODEL, _TEXT_REWRITE_TOKENIZER

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Text rewrite requires `transformers` with AutoModelForCausalLM support."
        ) from error

    torch_dtype = _resolve_torch_dtype(dtype_name)
    _TEXT_REWRITE_MODEL = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    _TEXT_REWRITE_TOKENIZER = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    _TEXT_REWRITE_MODEL_NAME = model_name
    return _TEXT_REWRITE_MODEL, _TEXT_REWRITE_TOKENIZER


def _build_text_rewrite_messages(
    label: str,
    max_words: int,
    exact_words: Optional[int] = None,
) -> List[Dict[str, str]]:
    limit_instruction = (
        f"Use exactly {exact_words} words."
        if exact_words is not None
        else f"Use at most {max_words} words."
    )
    system_prompt = (
        "You are an expert linguist that shortens object descriptions into concise, grammatically correct noun phrases. "
        "Return only the shortened noun phrase. Do not include punctuation, quotes, or explanations."
    )
    user_prompt = (
        f"Original label: {label}\n\n"
        "Shorten the label into a grammatically correct noun phrase describing the main object. "
        "Keep the most important modifiers and the core noun. "
        f"{limit_instruction}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _extract_plain_label_line(text: Optional[str]) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    if raw.startswith("```"):
        raw = raw.strip("`")
        if "\n" in raw:
            raw = raw.split("\n", 1)[1]

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return ""
    first = lines[0].strip("-•* \t\"'")
    if ":" in first:
        left, right = first.split(":", 1)
        if len(left.split()) <= 3 and right.strip():
            first = right.strip()
    return first


def _parse_vl_label_response(
    raw_text: Optional[str],
) -> Tuple[str, float, bool, str, bool]:
    label_10 = normalize_label(_extract_plain_label_line(raw_text), max_words=10)
    if not label_10:
        return _reject_record("empty_label")

    rejected = is_generic_label(label_10) or phrase_word_count(label_10) > 10
    reject_reason = ""
    if rejected:
        if is_generic_label(label_10):
            reject_reason = "generic_label"
        else:
            reject_reason = "ambiguous_or_long"
    return label_10, 1.0, False, reject_reason, rejected


def _run_local_text_rewrite_request(
    label: str,
    model_name: str,
    max_tokens: int,
    device_map: str,
    dtype_name: str,
    max_words: int,
    exact_words: Optional[int] = None,
) -> Optional[str]:
    model, tokenizer = _ensure_local_text_rewrite_model(
        model_name=model_name,
        device_map=device_map,
        dtype_name=dtype_name,
    )
    messages = _build_text_rewrite_messages(
        label=label,
        max_words=max_words,
        exact_words=exact_words,
    )
    try:
        if hasattr(tokenizer, "apply_chat_template"):
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = (
                f"System: {messages[0]['content']}\n"
                f"User: {messages[1]['content']}\n"
                "Assistant:"
            )
        inputs = tokenizer([str(prompt_text)], return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            return None
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        output_text = tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not output_text:
            return None
        return output_text[0]
    except Exception as error:
        print(f"Local text rewrite request failed: {error}")
        return None


def _parse_text_rewrite_phrase(
    raw_text: Optional[str],
    max_words: int,
    exact_words: Optional[int] = None,
) -> str:
    if not raw_text:
        raise ValueError("Empty response from text model")

    phrase = normalize_label(_extract_plain_label_line(raw_text), max_words=max_words)
    if not phrase:
        raise ValueError(f"Could not parse valid phrase. Raw: {raw_text}")

    words = phrase.split()
    if exact_words is not None:
        phrase = " ".join(words[:exact_words])

    return phrase


def _run_local_transformers_request(
    messages: List[Dict[str, Any]],
    model_name: str,
    max_tokens: int,
    device_map: str,
    dtype_name: str,
) -> Optional[str]:
    # Normalize chat messages to the multimodal format expected by Qwen processors.
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
        prompt_text = processor.apply_chat_template(
            normalized_messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        if isinstance(prompt_text, list):
            prompt_text = prompt_text[0]

        image_inputs: List[Image.Image] = []
        for message in normalized_messages:
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "image":
                    continue
                image_path = item.get("image")
                if image_path is None:
                    continue
                try:
                    image_inputs.append(Image.open(str(image_path)).convert("RGB"))
                except Exception as error:
                    print(f"Failed to load crop image for local transformers: {error}")

        processor_kwargs: Dict[str, Any] = {
            "text": [str(prompt_text)],
            "return_tensors": "pt",
            "padding": True,
        }
        if image_inputs:
            processor_kwargs["images"] = image_inputs

        inputs = processor(**processor_kwargs)
        inputs = inputs.to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            return None
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(input_ids, generated_ids)
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
    return sorted(
        path
        for path in ann_dir.glob("*.json")
        if not path.name.endswith("_text.json")
        and not path.name.endswith("_enhanced.json")
    )


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


def _mask_bbox_xyxy_from_segmentation(
    segmentation: Dict[str, Any],
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(segmentation, dict):
        return None
    if "counts" not in segmentation or "size" not in segmentation:
        return None

    try:
        from pycocotools import mask as mask_utils
    except ImportError as error:
        raise RuntimeError(
            "--crop-box-source=mask requires pycocotools. Install it or use --crop-box-source=bbox."
        ) from error

    size = segmentation.get("size", [])
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        return None

    seg_h, seg_w = int(size[0]), int(size[1])
    counts = segmentation.get("counts")
    if isinstance(counts, list):
        rle = mask_utils.frPyObjects(segmentation, seg_h, seg_w)
        if isinstance(rle, list):
            if not rle:
                return None
            rle = mask_utils.merge(rle)
    else:
        rle = segmentation

    bbox = mask_utils.toBbox(rle)
    if hasattr(bbox, "tolist"):
        bbox = bbox.tolist()
    if isinstance(bbox, list) and bbox and isinstance(bbox[0], list):
        bbox = bbox[0]
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    x, y, w, h = [float(v) for v in bbox]
    if w <= 0.0 or h <= 0.0:
        return None

    x0 = max(0, int(np.floor(x)))
    y0 = max(0, int(np.floor(y)))
    x1 = min(width, int(np.ceil(x + w)))
    y1 = min(height, int(np.ceil(y + h)))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _stable_seed(*values: Any) -> int:
    token = "|".join(str(value) for value in values)
    seed = 2166136261
    for byte in token.encode("utf-8"):
        seed ^= byte
        seed = (seed * 16777619) & 0xFFFFFFFF
    return seed


def _decode_mask_from_segmentation(
    segmentation: Dict[str, Any],
    width: int,
    height: int,
) -> Optional[np.ndarray]:
    if not isinstance(segmentation, dict):
        return None
    if "counts" not in segmentation or "size" not in segmentation:
        return None

    try:
        from pycocotools import mask as mask_utils
    except ImportError as error:
        raise RuntimeError(
            "Mask geometry extraction requires pycocotools. Install it to export mask center/sample points."
        ) from error

    size = segmentation.get("size", [])
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        return None

    seg_h, seg_w = int(size[0]), int(size[1])
    counts = segmentation.get("counts")
    if isinstance(counts, list):
        rle = mask_utils.frPyObjects(segmentation, seg_h, seg_w)
        if isinstance(rle, list):
            if not rle:
                return None
            rle = mask_utils.merge(rle)
    else:
        rle = segmentation

    mask = mask_utils.decode(rle)
    if getattr(mask, "ndim", 0) == 3:
        mask = mask[..., 0]
    if mask.shape != (seg_h, seg_w):
        return None

    mask_bool = np.asarray(mask).astype(bool)
    if not mask_bool.any():
        return None
    if seg_w != width or seg_h != height:
        return None
    return mask_bool


def _mask_center_and_sample_points(
    segmentation: Dict[str, Any],
    width: int,
    height: int,
    num_sample_points: int,
    seed: int,
) -> Tuple[List[float], List[List[float]]]:
    mask = _decode_mask_from_segmentation(
        segmentation=segmentation,
        width=width,
        height=height,
    )
    if mask is None:
        return [], []

    ys, xs = np.where(mask)
    if ys.size == 0:
        return [], []

    center_xy = [float(xs.mean()), float(ys.mean())]
    sample_count = min(max(int(num_sample_points), 0), int(ys.size))
    if sample_count <= 0:
        return center_xy, []

    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(ys.size, size=sample_count, replace=False)
    sampled_points_xy = [
        [float(xs[idx]), float(ys[idx])] for idx in sample_indices.tolist()
    ]
    return center_xy, sampled_points_xy


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
    crop_source: str,
    mask_index: int,
    raw_response: str,
    label: str,
    confidence: float,
    ambiguous: bool,
    reject_reason: str,
    rejected: bool,
    label_5: str,
    label_2: str,
    num_mask_sample_points: int,
    inference_backend: str,
) -> Dict[str, Any]:
    bbox_xywh = [float(v) for v in ann["bbox"]]
    width = int(image_info["width"])
    height = int(image_info["height"])
    area_px = float(ann.get("area", 0.0))
    mask_center_xy, mask_sample_points_xy = _mask_center_and_sample_points(
        segmentation=ann.get("segmentation", {}),
        width=width,
        height=height,
        num_sample_points=num_mask_sample_points,
        seed=_stable_seed(image_info.get("image_id"), ann.get("id")),
    )
    crop_box_xywh = (
        _crop_xyxy_to_xywh(crop_box_xyxy) if crop_box_xyxy is not None else []
    )
    return {
        "schema_version": RAW_SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "prompt_mode": "crop_only",
        "model_name": args.model,
        "inference_backend": inference_backend,
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
        "crop_source": crop_source,
        "crop_box_xywh": crop_box_xywh,
        "crop_box_xyxy": [float(v) for v in crop_box_xyxy]
        if crop_box_xyxy is not None
        else [],
        "point_coords": ann.get("point_coords"),
        "mask_center_xy": mask_center_xy,
        "mask_sample_points_xy": mask_sample_points_xy,
        "area": area_px,
        "area_frac": area_to_fraction(area_px, width=width, height=height),
        "predicted_iou": float(ann.get("predicted_iou", 0.0)),
        "stability_score": float(ann.get("stability_score", 0.0)),
        "segmentation": ann["segmentation"],
        "label": label,
        "label_10": label,
        "normalized_label": label,
        "label_5": str(label_5 or "").strip(),
        "label_2": str(label_2 or "").strip(),
        "confidence": confidence,
        "ambiguous": ambiguous,
        "rejected": rejected,
        "reject_reason": reject_reason,
        "raw_response": raw_response,
    }


def _reject_record(reason: str) -> Tuple[str, float, bool, str, bool]:
    return "", 0.0, True, reason, True


def _canonical_annotation_path(path_value: Any) -> Optional[Path]:
    if path_value is None:
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def _load_latest_records_index(
    raw_jsonl: Path,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if not raw_jsonl.exists():
        return {}

    records_index: Dict[str, Dict[str, Dict[str, Any]]] = {}
    with raw_jsonl.open("r") as fopen:
        for line in fopen:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            annotation_path = _canonical_annotation_path(record.get("annotation_path"))
            mask_id = str(record.get("mask_id", "")).strip()
            if annotation_path is None or not mask_id:
                continue
            ann_key = str(annotation_path)
            records_index.setdefault(ann_key, {})[mask_id] = record
    return records_index


def _load_latest_records_by_mask_id(raw_jsonl: Path) -> Dict[str, Dict[str, Any]]:
    if not raw_jsonl.exists():
        return {}

    records_by_mask_id: Dict[str, Dict[str, Any]] = {}
    with raw_jsonl.open("r") as fopen:
        for line in fopen:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            mask_id = str(record.get("mask_id", "")).strip()
            if not mask_id:
                continue
            # Keep the last-seen record per mask id so reruns use freshest labels.
            records_by_mask_id[mask_id] = record
    return records_by_mask_id


def _is_placeholder_label(label: str) -> bool:
    normalized = str(label or "").strip().lower()
    if not normalized:
        return False
    if normalized.startswith("sa1b crop instance"):
        return True
    if normalized.startswith("mask with iou"):
        return True
    return False


def _is_record_accepted_for_text_export(
    record: Dict[str, Any],
    min_area: float,
    min_stability_score: float,
    min_predicted_iou: float,
) -> bool:
    label = str(
        record.get("label_10")
        or record.get("normalized_label")
        or record.get("label")
        or ""
    ).strip()
    if not label:
        return False
    if bool(record.get("rejected", False)):
        return False
    if bool(record.get("ambiguous", False)):
        return False
    if float(record.get("area", 0.0)) <= float(min_area):
        return False
    if float(record.get("stability_score", 0.0)) <= float(min_stability_score):
        return False
    if float(record.get("predicted_iou", 0.0)) <= float(min_predicted_iou):
        return False
    if is_generic_label(label):
        return False
    if _is_placeholder_label(label):
        return False
    return True


def _build_enhanced_annotation(
    source_annotation: Dict[str, Any],
    record: Dict[str, Any],
) -> Dict[str, Any]:
    output_annotation = dict(source_annotation)
    label_10 = str(
        record.get("label_10")
        or record.get("normalized_label")
        or record.get("label")
        or ""
    ).strip()
    label_5 = str(record.get("label_5") or "").strip()
    label_2 = str(record.get("label_2") or "").strip()
    output_annotation["label_10"] = label_10
    if label_5:
        output_annotation["label_5"] = label_5
    if label_2:
        output_annotation["label_2"] = label_2
    output_annotation["mask_sample_points_xy"] = [
        [float(point[0]), float(point[1])]
        for point in record.get("mask_sample_points_xy", [])
        if isinstance(point, (list, tuple)) and len(point) >= 2
    ]
    return output_annotation


def _has_valid_prompt_points(points: Any) -> bool:
    if not isinstance(points, list) or not points:
        return False
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            float(point[0])
            float(point[1])
        except (TypeError, ValueError):
            continue
        return True
    return False


def _write_enhanced_annotation_file(
    annotation_path: Path,
    source_data: Dict[str, Any],
    accepted_records_by_mask_id: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[Path], int]:
    output_path = annotation_path.with_name(f"{annotation_path.stem}_enhanced.json")
    output_data = dict(source_data)

    enhanced_annotations: List[Dict[str, Any]] = []
    for annotation in source_data.get("annotations", []):
        mask_id = str(annotation.get("id", "")).strip()
        record = accepted_records_by_mask_id.get(mask_id)
        if record is None:
            continue

        label = str(
            record.get("label_10")
            or record.get("normalized_label")
            or record.get("label")
            or ""
        ).strip()
        if not label:
            continue

        prompt_points = record.get("point_coords", annotation.get("point_coords", []))
        if not _has_valid_prompt_points(prompt_points):
            continue

        enhanced_annotations.append(
            _build_enhanced_annotation(
                source_annotation=annotation,
                record=record,
            )
        )

    output_data["annotations"] = enhanced_annotations
    with output_path.open("w") as fout:
        json.dump(output_data, fout)
    return output_path, len(enhanced_annotations)


def main() -> None:
    global args

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
    output_root = Path(args.output_root)
    raw_dir = output_root / "raw"
    prompt_dir = output_root / "prompt_renders" / args.split
    raw_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    raw_jsonl = raw_dir / f"{args.split}.jsonl"

    if args.inference_backend == "stub":
        print(
            "WARNING: --inference-backend=stub generates placeholder labels "
            "('sa1b crop instance ...'). Use local_transformers for real Qwen labels.",
            file=sys.stderr,
        )

    existing_mask_ids = set() if args.overwrite else _load_existing_mask_ids(raw_jsonl)
    existing_records_index = (
        _load_latest_records_index(raw_jsonl)
        if (not args.overwrite and args.write_enhanced_annotations)
        else {}
    )
    existing_records_by_mask_id = (
        _load_latest_records_by_mask_id(raw_jsonl)
        if (not args.overwrite and args.write_enhanced_annotations)
        else {}
    )
    ann_files = _annotation_files(sa1b_root, args.split)
    ann_files = ann_files[args.start_index :]
    if args.limit_images is not None:
        ann_files = ann_files[: args.limit_images]

    processed = 0
    skipped = 0
    filtered_min_area = 0
    crop_from_mask = 0
    crop_from_bbox = 0
    crop_mask_fallback_to_bbox = 0
    enhanced_json_files_written = 0
    enhanced_json_files_skipped_empty = 0
    enhanced_masks_kept = 0
    enhanced_masks_from_existing = 0
    label_rewrite_requests = 0
    label_rewrite_failures = 0
    with raw_jsonl.open("a") as fout:
        for ann_path in ann_files:
            with ann_path.open("r") as fopen:
                ann_data = json.load(fopen)
            image_info = ann_data["image"]
            ann_key = str(ann_path.resolve())
            existing_records_for_ann = existing_records_index.get(ann_key, {})
            accepted_records_for_ann: Dict[str, Dict[str, Any]] = {}
            image_path = _image_path(sa1b_root, args.split, image_info)
            if not image_path.exists():
                print(f"Missing image for annotation: {ann_path}", file=sys.stderr)
                continue

            image = Image.open(image_path).convert("RGB")
            annotations = ann_data.get("annotations", [])
            valid_annotations = [
                ann for ann in annotations
                if float(ann.get("area", 0.0)) > args.min_area and
                   float(ann.get("predicted_iou", 0.0)) >= args.min_predicted_iou and
                   float(ann.get("stability_score", 0.0)) >= args.min_stability_score
            ]
            if args.max_masks_per_image is not None:
                annotations_to_process = valid_annotations[: args.max_masks_per_image]
            else:
                annotations_to_process = annotations

            for mask_index, ann in enumerate(annotations_to_process):
                mask_id = str(ann["id"])
                if mask_id in existing_mask_ids:
                    skipped += 1
                    if args.write_enhanced_annotations:
                        existing_record = existing_records_for_ann.get(mask_id)
                        if existing_record is None:
                            candidate = existing_records_by_mask_id.get(mask_id)
                            if candidate is not None and str(
                                candidate.get("image_id", "")
                            ) == str(image_info["image_id"]):
                                existing_record = candidate
                        if (
                            existing_record is not None
                            and _is_record_accepted_for_text_export(
                                existing_record,
                                min_area=args.min_area,
                                min_stability_score=args.min_stability_score,
                                min_predicted_iou=args.min_predicted_iou,
                            )
                        ):
                            accepted_records_for_ann[mask_id] = existing_record
                            enhanced_masks_from_existing += 1
                    continue
                if float(ann.get("area", 0.0)) <= args.min_area:
                    filtered_min_area += 1
                    continue
                width = int(image_info["width"])
                height = int(image_info["height"])
                crop_source = "bbox"
                if args.crop_box_source == "mask":
                    mask_crop_xyxy = _mask_bbox_xyxy_from_segmentation(
                        segmentation=ann.get("segmentation", {}),
                        width=width,
                        height=height,
                    )
                    if mask_crop_xyxy is not None:
                        crop_xyxy = mask_crop_xyxy
                        crop_source = "mask"
                        crop_from_mask += 1
                    else:
                        crop_xyxy = _clamp_bbox_xyxy(
                            ann["bbox"],
                            width=width,
                            height=height,
                        )
                        crop_from_bbox += 1
                        crop_mask_fallback_to_bbox += 1
                else:
                    crop_xyxy = _clamp_bbox_xyxy(
                        ann["bbox"],
                        width=width,
                        height=height,
                    )
                    crop_from_bbox += 1
                if float(ann.get("stability_score", 0.0)) <= args.min_stability_score:
                    label, confidence, ambiguous, reject_reason, rejected = _reject_record("min_stability_score")
                    raw_response = ""
                elif float(ann.get("predicted_iou", 0.0)) <= args.min_predicted_iou:
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
                    if args.inference_backend == "stub":
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
                        label, confidence, ambiguous, reject_reason, rejected = _parse_vl_label_response(
                            raw_response
                        )

                label_5 = ""
                label_2 = ""
                if label and not rejected:
                    if args.disable_label_rewrite:
                         raise ValueError("disable_label_rewrite is set, but fallback is disabled. Cannot proceed without text rewrite model.")

                    if args.inference_backend != "local_transformers":
                         raise ValueError("inference_backend must be local_transformers for text rewrite.")

                    label_rewrite_requests += 1
                    rewrite_5_raw_response = _run_local_text_rewrite_request(
                        label=label,
                        model_name=args.text_rewrite_model,
                        max_tokens=args.rewrite_max_tokens,
                        device_map=args.device_map,
                        dtype_name=args.torch_dtype,
                        max_words=5,
                    )
                    if rewrite_5_raw_response is None:
                        raise RuntimeError("rewrite_5_raw_response is None")
                    label_5 = _parse_text_rewrite_phrase(
                        raw_text=rewrite_5_raw_response,
                        max_words=5,
                    )

                    label_rewrite_requests += 1
                    rewrite_2_raw_response = _run_local_text_rewrite_request(
                        label=label,
                        model_name=args.text_rewrite_model,
                        max_tokens=args.rewrite_max_tokens,
                        device_map=args.device_map,
                        dtype_name=args.torch_dtype,
                        max_words=2,
                        exact_words=2,
                    )
                    if rewrite_2_raw_response is None:
                        raise RuntimeError("rewrite_2_raw_response is None")
                    label_2 = _parse_text_rewrite_phrase(
                        raw_text=rewrite_2_raw_response,
                        max_words=2,
                        exact_words=2,
                    )

                record = _build_record(
                    image_info=image_info,
                    image_path=image_path,
                    ann_path=ann_path,
                    ann=ann,
                    crop_box_xyxy=crop_xyxy,
                    crop_source=crop_source,
                    mask_index=mask_index,
                    raw_response=raw_response,
                    label=label,
                    confidence=confidence,
                    ambiguous=ambiguous,
                    reject_reason=reject_reason,
                    rejected=rejected,
                    label_5=label_5,
                    label_2=label_2,
                    num_mask_sample_points=args.num_mask_sample_points,
                    inference_backend=args.inference_backend,
                )
                fout.write(json.dumps(record) + "\n")
                fout.flush()
                if args.write_enhanced_annotations and _is_record_accepted_for_text_export(
                    record,
                    min_area=args.min_area,
                    min_stability_score=args.min_stability_score,
                    min_predicted_iou=args.min_predicted_iou,
                ):
                    accepted_records_for_ann[mask_id] = record
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

            if args.write_enhanced_annotations:
                written_path, kept_count = _write_enhanced_annotation_file(
                    annotation_path=ann_path,
                    source_data=ann_data,
                    accepted_records_by_mask_id=accepted_records_for_ann,
                )
                if written_path is None:
                    enhanced_json_files_skipped_empty += 1
                else:
                    enhanced_json_files_written += 1
                enhanced_masks_kept += kept_count

    print(
        json.dumps(
            {
                "raw_jsonl": str(raw_jsonl),
                "processed": processed,
                "skipped_existing": skipped,
                "filtered_min_area": filtered_min_area,
                "crop_box_source_requested": args.crop_box_source,
                "crop_box_used_mask": crop_from_mask,
                "crop_box_used_bbox": crop_from_bbox,
                "crop_box_mask_fallback_to_bbox": crop_mask_fallback_to_bbox,
                "write_enhanced_annotations": args.write_enhanced_annotations,
                "enhanced_json_files_written": enhanced_json_files_written,
                "enhanced_json_files_skipped_empty": enhanced_json_files_skipped_empty,
                "enhanced_masks_kept": enhanced_masks_kept,
                "enhanced_masks_from_existing": enhanced_masks_from_existing,
                "text_rewrite_model": None
                if args.disable_label_rewrite
                else args.text_rewrite_model,
                "label_rewrite_requests": label_rewrite_requests,
                "label_rewrite_failures": label_rewrite_failures,
                "min_area_threshold": args.min_area,
                "min_predicted_iou_threshold": args.min_predicted_iou,
                "min_stability_score_threshold": args.min_stability_score,
                "hard_min_area": MIN_SCREENING_AREA,
                "hard_min_predicted_iou": MIN_SCREENING_PREDICTED_IOU,
                "hard_min_stability_score": MIN_SCREENING_STABILITY_SCORE,
                "split": args.split,
                "inference_backend": args.inference_backend,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
