"""Drawing helpers: overlay masks and bounding boxes on RGB frames.

Pure numpy + OpenCV; no matplotlib so the headless edge case is easy.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .tracker import TrackedObject

_DEFAULT_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (255, 56, 56),
    (56, 255, 56),
    (56, 105, 255),
    (255, 178, 29),
    (207, 56, 255),
    (29, 226, 255),
    (255, 255, 56),
    (255, 56, 178),
    (124, 252, 0),
    (255, 105, 180),
)


def palette_color(obj_id: int) -> Tuple[int, int, int]:
    """Return a deterministic RGB color for an object id."""
    return _DEFAULT_PALETTE[obj_id % len(_DEFAULT_PALETTE)]


def overlay_mask(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.5,
) -> np.ndarray:
    """Alpha-blend ``color`` into ``frame_rgb`` wherever ``mask`` is truthy.

    The input frame is never mutated.
    """
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise ValueError(f"frame must be HxWx3, got shape {frame_rgb.shape}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0,1], got {alpha}")

    mask_bool = np.asarray(mask).astype(bool)
    if mask_bool.shape != frame_rgb.shape[:2]:
        raise ValueError(
            f"mask shape {mask_bool.shape} does not match frame {frame_rgb.shape[:2]}"
        )

    out = frame_rgb.astype(np.float32, copy=True)
    color_arr = np.asarray(color, dtype=np.float32)
    out[mask_bool] = (1.0 - alpha) * out[mask_bool] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def _mask_to_xyxy(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Tight axis-aligned bbox of a binary mask, or ``None`` if empty."""
    mask_bool = np.asarray(mask).astype(bool)
    if not mask_bool.any():
        return None
    ys, xs = np.where(mask_bool)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def draw_box(
    frame_rgb: np.ndarray,
    xyxy: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    label: Optional[str] = None,
) -> np.ndarray:
    """Draw a rectangle (and optional label) onto a copy of the frame."""
    import cv2

    x1, y1, x2, y2 = (int(v) for v in xyxy)
    out = frame_rgb.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        y_text = max(0, y1 - 4)
        cv2.rectangle(out, (x1, y_text - th - 4), (x1 + tw + 4, y_text + baseline), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 2, y_text - 2),
            font,
            font_scale,
            (0, 0, 0),
            text_thickness,
            lineType=cv2.LINE_AA,
        )
    return out


def draw_point(
    frame_rgb: np.ndarray,
    xy: Tuple[float, float],
    color: Tuple[int, int, int],
    radius: int = 5,
) -> np.ndarray:
    import cv2

    out = frame_rgb.copy()
    cv2.circle(out, (int(xy[0]), int(xy[1])), radius, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(out, (int(xy[0]), int(xy[1])), radius, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    return out


def annotate_frame(
    frame_rgb: np.ndarray,
    objects: Iterable[TrackedObject],
    *,
    alpha: float = 0.5,
    draw_boxes: bool = True,
    draw_labels: bool = True,
    label_prefix: str = "obj",
) -> np.ndarray:
    """Draw all tracked objects on ``frame_rgb`` and return the new frame."""
    out = frame_rgb
    for obj in objects:
        color = palette_color(obj.obj_id)
        if obj.mask is not None:
            out = overlay_mask(out, obj.mask, color, alpha=alpha)
        if draw_boxes:
            xyxy = obj.box_xyxy
            if xyxy is None and obj.mask is not None:
                xyxy = _mask_to_xyxy(obj.mask)
            if xyxy is not None:
                label = None
                if draw_labels:
                    parts = [f"{label_prefix} {obj.obj_id}"]
                    if obj.score is not None:
                        parts.append(f"{obj.score:.2f}")
                    if obj.label:
                        parts.append(obj.label)
                    label = " ".join(parts)
                out = draw_box(out, xyxy, color, label=label)
    return out


def composite_legend(
    frame_rgb: np.ndarray,
    objects_by_id: Mapping[int, str],
    origin: Tuple[int, int] = (10, 10),
) -> np.ndarray:
    """Draw a small legend in the corner mapping ``obj_id -> label``."""
    import cv2

    out = frame_rgb.copy()
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_h = 18
    for i, (obj_id, label) in enumerate(objects_by_id.items()):
        color = palette_color(obj_id)
        cv2.rectangle(out, (x, y + i * line_h), (x + 12, y + i * line_h + 12), color, -1)
        cv2.putText(
            out,
            f"{label}",
            (x + 18, y + i * line_h + 11),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA,
        )
    return out
