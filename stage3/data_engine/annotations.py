"""
Shared helpers for the Stage 3 image text-mask pair data engine.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


RAW_SCHEMA_VERSION = "sa1b_stage3_raw_v3"
GROUPED_SCHEMA_VERSION = "sa1b_stage3_grouped_v2"
PROMPT_VERSION = "sa1b_qwen_crop_np_v2"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MAX_LABEL_WORDS = 10

GENERIC_LABELS = {
    "",
    "unknown",
    "unclear",
    "not sure",
    "object",
    "objects",
    "item",
    "items",
    "thing",
    "things",
    "stuff",
    "entity",
    "entities",
    "part",
    "parts",
    "region",
    "regions",
    "area",
    "areas",
    "background",
    "foreground",
}

STOPWORD_PREFIXES = {"a", "an", "the"}
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
WHITESPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s/-]+")


@dataclass(frozen=True)
class RawMaskLabelRecord:
    image_id: str
    image_path: str
    annotation_path: str
    mask_id: str
    mask_index: int
    width: int
    height: int
    bbox_xywh: List[float]
    bbox_xyxy: List[float]
    bbox_norm_xywh: List[float]
    area: float
    area_frac: float
    segmentation: Any
    label: str
    normalized_label: str
    confidence: float
    ambiguous: bool
    rejected: bool
    reject_reason: str
    raw_response: str


def _default_box_colors() -> List[Tuple[int, int, int]]:
    return [
        (0, 255, 200),
        (255, 210, 0),
        (255, 99, 132),
        (54, 162, 235),
        (153, 102, 255),
        (255, 159, 64),
        (75, 192, 192),
        (201, 203, 207),
    ]


def bbox_xywh_to_xyxy(bbox_xywh: Iterable[float]) -> List[float]:
    x, y, w, h = [float(v) for v in bbox_xywh]
    return [x, y, x + w, y + h]


def bbox_xywh_to_normalized_xywh(
    bbox_xywh: Iterable[float], width: int, height: int
) -> List[float]:
    x, y, w, h = [float(v) for v in bbox_xywh]
    width = max(int(width), 1)
    height = max(int(height), 1)
    return [x / width, y / height, w / width, h / height]


def area_to_fraction(area_px: float, width: int, height: int) -> float:
    denom = float(max(int(width), 1) * max(int(height), 1))
    return float(area_px) / denom


def _load_overlay_font(
    image_size: Tuple[int, int], scale: float = 1.0
) -> ImageFont.ImageFont:
    width, height = image_size
    base_font_size = max(56, min(280, int(min(width, height) * 0.12)))
    font_size = max(12, int(round(((base_font_size / 25.0) * 3.0) * scale)))
    font_candidates = [
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]
    for font_path in font_candidates:
        if not Path(font_path).is_absolute() or Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size=font_size)
            except OSError:
                continue
    return ImageFont.load_default()


def _annotation_banner_text(index: int, annotation: Dict[str, Any]) -> str:
    label = str(
        annotation.get("label")
        or annotation.get("query_text")
        or annotation.get("annotation_text")
        or ""
    ).strip()
    if not label:
        label = "annotation pending"
    area = annotation.get("area")
    if area is None:
        return f"{index + 1}. {label}"
    try:
        area_value = float(area)
    except (TypeError, ValueError):
        return f"{index + 1}. {label}"
    return f"{index + 1}. {label}\nmask area: {area_value:,.0f} px"


def build_qwen_labeling_messages(
    crop_image_path: str,
) -> List[Dict[str, Any]]:
    system_prompt = (
        "You are an expert generic labler for mask annotations. "
        "Return JSON only with ONE key: 'label'. "
        "Describe the main visible object or object part within the tight boundaries of this cropped image. "
        "The cropped image you see corresponds to the most top, bottom, left and right points of the target object. "
        "Make sure the words used are common object noun phrases whenever possible. "
        "Use at most 10 words. Avoid full sentences, adjectives without nouns, and vague words "
        "like object, thing, item, stuff, region, area, or unknown unless nothing better is possible. "
        "If the crop is too ambiguous, too small, or unreadable, still try to give your best noun phrase guess."
    )
    user_prompt = (
        "Describe what is visible in this cropped image in as much detail as possible, "
        "but keep the final label to 10 words or fewer as a valid JSON with key 'label'. "
        "Focus on the main visible object or object part. "
        "Return JSON only."
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image", "image": crop_image_path},
            ],
        },
    ]


def _wrap_text_lines(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int
) -> List[str]:
    wrapped: List[str] = []
    for paragraph in str(text).split("\n"):
        words = paragraph.split()
        if not words:
            wrapped.append("")
            continue
        current_line = words[0]
        for word in words[1:]:
            candidate = f"{current_line} {word}"
            left, top, right, bottom = draw.textbbox((0, 0), candidate, font=font)
            if (right - left) <= max_width:
                current_line = candidate
            else:
                wrapped.append(current_line)
                current_line = word
        wrapped.append(current_line)
    return wrapped or [""]


def _format_numeric(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number - round(number)) < 1e-4:
        return str(int(round(number)))
    return f"{number:.1f}"


def _format_numeric_list(values: Iterable[Any]) -> str:
    return "[" + ", ".join(_format_numeric(value) for value in values) + "]"


def _resolve_crop_box_xyxy(
    annotation: Dict[str, Any], image_size: Tuple[int, int]
) -> Optional[Tuple[int, int, int, int]]:
    width, height = image_size
    if annotation.get("crop_box_xyxy"):
        values = [float(v) for v in annotation["crop_box_xyxy"]]
    elif annotation.get("crop_box_xywh"):
        values = bbox_xywh_to_xyxy(annotation["crop_box_xywh"])
    elif annotation.get("bbox_xywh"):
        x, y, w, h = [float(v) for v in annotation["bbox_xywh"]]
        pad_x = max(16, int(round(w * 0.15)))
        pad_y = max(16, int(round(h * 0.15)))
        values = [x - pad_x, y - pad_y, x + w + pad_x, y + h + pad_y]
    else:
        return None

    x0 = max(0, min(width, int(round(values[0]))))
    y0 = max(0, min(height, int(round(values[1]))))
    x1 = max(0, min(width, int(round(values[2]))))
    y1 = max(0, min(height, int(round(values[3]))))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _build_crop_preview(
    source_image: Image.Image,
    annotation: Dict[str, Any],
    color: Tuple[int, int, int],
) -> Optional[Image.Image]:
    crop_box = _resolve_crop_box_xyxy(annotation, source_image.size)
    if crop_box is None:
        return None
    x0, y0, x1, y1 = crop_box
    preview = source_image.crop((x0, y0, x1, y1)).convert("RGBA")
    mask = annotation.get("mask")
    if mask is not None:
        mask_array = np.asarray(mask).astype(bool)
        if mask_array.shape == (source_image.size[1], source_image.size[0]):
            mask_crop = mask_array[y0:y1, x0:x1]
            overlay = Image.new("RGBA", preview.size, (0, 0, 0, 0))
            overlay_pixels = overlay.load()
            mask_y, mask_x = np.where(mask_crop)
            for yy, xx in zip(mask_y.tolist(), mask_x.tolist()):
                overlay_pixels[xx, yy] = (*color, 110)
            preview = Image.alpha_composite(preview, overlay)
    preview_draw = ImageDraw.Draw(preview)
    preview_draw.rectangle(
        (0, 0, preview.size[0] - 1, preview.size[1] - 1),
        outline=color,
        width=max(4, int(round(min(preview.size) * 0.02))),
    )
    return preview.convert("RGB")


def _annotation_structure_text(index: int, annotation: Dict[str, Any]) -> str:
    label = str(
        annotation.get("label")
        or annotation.get("query_text")
        or annotation.get("annotation_text")
        or ""
    ).strip()
    if not label:
        label = "annotation pending"

    entry_kind = "query" if (
        "query_id" in annotation or "object_ids_output" in annotation
    ) else "annotation"
    entry_id = annotation.get(
        "query_id",
        annotation.get("object_id", annotation.get("mask_id", index)),
    )
    
    # Prettier layout
    lines = [
        f"◆ {entry_kind.upper()} ID: {entry_id}",
        f"  Text: \"{label}\""
    ]

    extra_props = []
    if annotation.get("object_ids_output") is not None:
        extra_props.append(f"Targets: {_format_numeric_list(annotation['object_ids_output'])}")
    if annotation.get("mask_id") is not None:
        extra_props.append(f"Mask: {annotation['mask_id']}")
    if annotation.get("area") is not None:
        extra_props.append(f"Area: {_format_numeric(annotation['area'])} px")
        
    if extra_props:
        lines.append(f"  Info: {' | '.join(extra_props)}")

    if annotation.get("bbox_xywh") is not None:
        lines.append(f"  BBox (xywh): {_format_numeric_list(annotation['bbox_xywh'])}")
    
    crop_box = _resolve_crop_box_xyxy(annotation, image_size=(10**9, 10**9))
    if crop_box is not None:
        lines.append(f"  Crop (xyxy): {_format_numeric_list(crop_box)}")
        
    return "\n".join(lines)


def visualize_annotation_example(
    image_path: str,
    annotations: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    max_annotations: Optional[int] = None,
    box_width: int = 10,
) -> Image.Image:
    source_image = Image.open(image_path).convert("RGB")
    overview = source_image.convert("RGBA")
    overlay = Image.new("RGBA", overview.size, (0, 0, 0, 0))
    items = annotations if max_annotations is None else annotations[:max_annotations]
    colors = _default_box_colors()
    badge_font = _load_overlay_font(source_image.size, scale=0.85)
    header_font = _load_overlay_font(source_image.size, scale=0.9)
    body_font = _load_overlay_font(source_image.size, scale=0.72)
    draw = ImageDraw.Draw(overview)

    for idx, ann in enumerate(items):
        color = colors[idx % len(colors)]
        mask = ann.get("mask")
        if mask is not None:
            mask_array = np.asarray(mask).astype(bool)
            if mask_array.shape == (overview.size[1], overview.size[0]):
                overlay_pixels = overlay.load()
                mask_y, mask_x = np.where(mask_array)
                for yy, xx in zip(mask_y.tolist(), mask_x.tolist()):
                    overlay_pixels[xx, yy] = (*color, 105)

    overview = Image.alpha_composite(overview, overlay)
    draw = ImageDraw.Draw(overview)
    badge_padding = max(8, int(getattr(badge_font, "size", 18) * 0.35))
    for idx, ann in enumerate(items):
        color = colors[idx % len(colors)]
        x, y, w, h = [float(v) for v in ann["bbox_xywh"]]
        draw.rectangle((x, y, x + w, y + h), outline=color, width=box_width)
        badge_text = str(idx + 1)
        left, top, right, bottom = draw.textbbox((0, 0), badge_text, font=badge_font)
        badge = (
            x,
            max(0, y - (bottom - top) - (2 * badge_padding)),
            x + (right - left) + (2 * badge_padding),
            max(0, y - (bottom - top) - (2 * badge_padding))
            + (bottom - top)
            + (2 * badge_padding),
        )
        draw.rounded_rectangle(badge, radius=max(8, badge_padding), fill=color)
        draw.text(
            (badge[0] + badge_padding, badge[1] + badge_padding),
            badge_text,
            fill=(0, 0, 0),
            font=badge_font,
        )

    overview_rgb = overview.convert("RGB")
    preview_image = (
        _build_crop_preview(source_image, items[0], colors[0])
        if len(items) == 1
        else None
    )

    panel_width = max(420, int(round(source_image.size[0] * 0.42)))
    gap = max(24, int(round(source_image.size[0] * 0.025)))
    panel_margin = max(18, int(getattr(body_font, "size", 16) * 0.8))
    card_padding = max(14, int(getattr(body_font, "size", 16) * 0.6))
    card_gap = max(14, int(getattr(body_font, "size", 16) * 0.55))
    color_bar_width = max(10, int(getattr(body_font, "size", 16) * 0.35))
    measure_draw = ImageDraw.Draw(Image.new("RGB", (panel_width, max(1, overview_rgb.size[1]))))
    line_left, line_top, line_right, line_bottom = measure_draw.textbbox(
        (0, 0), "Ag", font=body_font
    )
    line_height = line_bottom - line_top

    header_lines = _wrap_text_lines(
        measure_draw,
        "Annotation / query structure",
        header_font,
        panel_width - (2 * panel_margin),
    )
    header_height = len(header_lines) * line_height + max(0, len(header_lines) - 1) * 6

    preview_height = 0
    preview_title_lines: List[str] = []
    if preview_image is not None:
        preview_title_lines = _wrap_text_lines(
            measure_draw,
            "Qwen crop input",
            body_font,
            panel_width - (2 * panel_margin),
        )
        max_preview_width = panel_width - (2 * panel_margin)
        max_preview_height = max(180, int(round(source_image.size[1] * 0.32)))
        scale = min(
            max_preview_width / max(preview_image.size[0], 1),
            max_preview_height / max(preview_image.size[1], 1),
            1.0,
        )
        if scale < 1.0:
            resample_lanczos = (
                Image.Resampling.LANCZOS
                if hasattr(Image, "Resampling")
                else Image.LANCZOS
            )
            preview_image = preview_image.resize(
                (
                    max(1, int(round(preview_image.size[0] * scale))),
                    max(1, int(round(preview_image.size[1] * scale))),
                ),
                resample_lanczos,
            )
        preview_height = (
            (len(preview_title_lines) * line_height)
            + max(0, len(preview_title_lines) - 1) * 4
            + card_gap
            + preview_image.size[1]
            + card_gap
        )

    cards: List[Dict[str, Any]] = []
    card_text_width = panel_width - (2 * panel_margin) - color_bar_width - (2 * card_padding)
    for idx, ann in enumerate(items):
        wrapped_lines = _wrap_text_lines(
            measure_draw,
            _annotation_structure_text(idx, ann),
            body_font,
            card_text_width,
        )
        card_height = (
            (2 * card_padding)
            + (len(wrapped_lines) * line_height)
            + max(0, len(wrapped_lines) - 1) * 4
        )
        cards.append(
            {
                "lines": wrapped_lines,
                "height": card_height,
                "color": colors[idx % len(colors)],
            }
        )

    panel_height = (
        panel_margin
        + header_height
        + card_gap
        + preview_height
        + sum(card["height"] for card in cards)
        + max(0, len(cards) - 1) * card_gap
        + panel_margin
    )
    canvas_height = max(overview_rgb.size[1], panel_height)
    canvas_width = overview_rgb.size[0] + gap + panel_width
    canvas = Image.new("RGB", (canvas_width, canvas_height), (15, 18, 24))
    canvas.paste(overview_rgb, (0, 0))

    panel_x = overview_rgb.size[0] + gap
    canvas_draw = ImageDraw.Draw(canvas)
    canvas_draw.rounded_rectangle(
        (panel_x, 0, canvas_width - 1, canvas_height - 1),
        radius=max(18, panel_margin),
        fill=(20, 24, 32),
        outline=(46, 57, 74),
        width=2,
    )

    current_y = panel_margin
    for line in header_lines:
        canvas_draw.text((panel_x + panel_margin, current_y), line, fill=(255, 255, 255), font=header_font)
        current_y += line_height + 6
    current_y += max(0, card_gap - 6)

    if preview_image is not None:
        for line in preview_title_lines:
            canvas_draw.text(
                (panel_x + panel_margin, current_y),
                line,
                fill=(196, 208, 255),
                font=body_font,
            )
            current_y += line_height + 4
        preview_x = panel_x + panel_margin
        canvas.paste(preview_image, (preview_x, current_y))
        canvas_draw.rectangle(
            (
                preview_x,
                current_y,
                preview_x + preview_image.size[0] - 1,
                current_y + preview_image.size[1] - 1,
            ),
            outline=(82, 105, 140),
            width=2,
        )
        current_y += preview_image.size[1] + card_gap

    card_width = panel_width - (2 * panel_margin)
    for card in cards:
        x0 = panel_x + panel_margin
        y0 = current_y
        x1 = x0 + card_width
        y1 = y0 + card["height"]
        canvas_draw.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=max(12, int(card_padding * 0.9)),
            fill=(28, 34, 44),
            outline=(60, 74, 92),
            width=2,
        )
        canvas_draw.rectangle((x0, y0, x0 + color_bar_width, y1), fill=card["color"])
        text_x = x0 + color_bar_width + card_padding
        text_y = y0 + card_padding
        for line in card["lines"]:
            canvas_draw.text((text_x, text_y), line, fill=(240, 244, 250), font=body_font)
            text_y += line_height + 4
        current_y = y1 + card_gap

    if output_path is not None:
        canvas.save(output_path, quality=95)
    return canvas


def extract_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if "\n" in stripped:
            stripped = stripped.split("\n", 1)[1]
    match = JSON_RE.search(stripped)
    candidate = match.group(0) if match else stripped
    return json.loads(candidate)


def normalize_label(text: Optional[str], max_words: int = MAX_LABEL_WORDS) -> str:
    if text is None:
        return ""
    text = text.strip().lower()
    text = text.replace("_", " ")
    text = text.replace("\n", " ")
    text = NON_ALNUM_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip(" /-")
    if not text:
        return ""
    words = text.split()
    while words and words[0] in STOPWORD_PREFIXES:
        words = words[1:]
    if not words:
        return ""
    if len(words) > max_words:
        words = words[:max_words]
    return " ".join(words)


def is_generic_label(text: str) -> bool:
    normalized = normalize_label(text)
    return normalized in GENERIC_LABELS


def phrase_word_count(text: str) -> int:
    normalized = normalize_label(text)
    if not normalized:
        return 0
    return len(normalized.split())


def choose_prompt_bbox(bboxes_xywh: Iterable[Iterable[float]]) -> List[float]:
    best_bbox: Optional[List[float]] = None
    best_area = -1.0
    for bbox in bboxes_xywh:
        current = [float(v) for v in bbox]
        area = current[2] * current[3]
        if area > best_area:
            best_bbox = current
            best_area = area
    if best_bbox is None:
        raise ValueError("Cannot choose a prompt bbox from an empty bbox list.")
    return best_bbox


def spatial_prefix_for_bbox(bbox_xywh: Iterable[float], width: int, height: int) -> str:
    x, y, w, h = [float(v) for v in bbox_xywh]
    center_x = x + (w / 2.0)
    center_y = y + (h / 2.0)
    horizontal = "left" if center_x < (width / 2.0) else "right"
    vertical = "upper" if center_y < (height / 2.0) else "lower"
    return f"{vertical} {horizontal}"


def disambiguate_duplicate_labels(
    label: str,
    bbox_xywh: Iterable[float],
    width: int,
    height: int,
    used_labels: set[str],
    max_words: int = MAX_LABEL_WORDS,
) -> str:
    normalized = normalize_label(label, max_words=max_words)
    if normalized not in used_labels:
        return normalized

    prefixed = normalize_label(
        f"{spatial_prefix_for_bbox(bbox_xywh, width=width, height=height)} {normalized}",
        max_words=max_words,
    )
    if prefixed not in used_labels:
        return prefixed

    for suffix in range(2, 100):
        candidate = normalize_label(f"{prefixed} {suffix}", max_words=max_words)
        if candidate not in used_labels:
            return candidate
    return prefixed


def parse_model_json_response(raw_text: str) -> Tuple[str, float, bool, str]:
    parsed = extract_json_object(raw_text)
    label = normalize_label(parsed.get("label"))
    confidence = float(parsed.get("confidence", 1.0))
    ambiguous = bool(parsed.get("ambiguous", False))
    reject_reason = str(parsed.get("reject_reason", "") or "").strip()
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 1.0
    confidence = max(0.0, min(1.0, confidence))
    return label, confidence, ambiguous, reject_reason
