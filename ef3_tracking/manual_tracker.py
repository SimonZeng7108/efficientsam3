"""Manual-selection tracker.

The user picks the object to track on frame 0 with any mix of:

    * positive clicks (label=1) -- "this is the object"
    * negative clicks (label=0) -- "this is background"
    * a bounding box drawn around the object

Internally we forward this as a single ``add_prompt`` call to the SAM3 video
predictor using the ``points`` + ``bounding_boxes`` keyword family. From there
the tracker propagates the masklet through the whole video, no text encoder
required -- this is the lightest mode and the one we recommend on Orin AGX
when no language model is needed.
"""

from __future__ import annotations

from typing import List, Optional

from .prompts import (
    BoxPrompt,
    ManualSelection,
    PointPrompt,
    validate_box_in_frame,
    validate_points_in_frame,
)
from .tracker import BaseTracker, BackendProtocol, TrackedObject, parse_backend_outputs


class ManualTracker(BaseTracker):
    """Track an object selected by hand on the seed frame."""

    def __init__(self, backend: BackendProtocol, *, label: Optional[str] = None) -> None:
        super().__init__(backend, label=label)
        self._selections: List[ManualSelection] = []
        self._frame_size: Optional[tuple[int, int]] = None  # (width, height)

    def set_frame_size(self, width: int, height: int) -> None:
        """Tell the tracker the resolution of the source so prompts can be validated."""
        if width <= 0 or height <= 0:
            raise ValueError(f"frame size must be positive, got {width}x{height}")
        self._frame_size = (int(width), int(height))

    def add_selection(
        self,
        selection: ManualSelection,
        frame_idx: int = 0,
    ) -> List[TrackedObject]:
        """Register a manual selection and return what the model finds for it.

        Multiple selections (distinct ``obj_id``s) can be added before
        propagation -- handy when the user wants to track several objects from
        a single seed frame.
        """
        sid = self._require_session()
        self._validate(selection)

        boxes = None
        box_labels = None
        if selection.box is not None and self._frame_size is None:
            raise RuntimeError(
                "set_frame_size(width, height) must be called before adding a box selection"
            )
        if selection.box is not None:
            w, h = self._frame_size
            boxes = [list(selection.box.to_normalized_xywh(w, h))]
            box_labels = [1]

        resp = self._backend.add_geometric_prompt(
            session_id=sid,
            frame_idx=frame_idx,
            obj_id=selection.obj_id,
            points=selection.point_coords() or None,
            point_labels=selection.point_labels() or None,
            bounding_boxes=boxes,
            bounding_box_labels=box_labels,
        )
        self._selections.append(selection)
        return parse_backend_outputs(resp.get("outputs", {}), label=self._label)

    def _validate(self, selection: ManualSelection) -> None:
        if self._frame_size is not None:
            w, h = self._frame_size
            validate_points_in_frame(selection.points, w, h)
            if selection.box is not None:
                validate_box_in_frame(selection.box, w, h)

    @property
    def selections(self) -> List[ManualSelection]:
        return list(self._selections)


def make_box_selection(
    x1: float, y1: float, x2: float, y2: float, obj_id: int = 0
) -> ManualSelection:
    return ManualSelection(box=BoxPrompt(x1=x1, y1=y1, x2=x2, y2=y2), obj_id=obj_id)


def make_point_selection(
    points: List[tuple[float, float, int]] | List[PointPrompt],
    obj_id: int = 0,
) -> ManualSelection:
    """Build a ManualSelection from a list of points.

    The ``points`` arg accepts either ``PointPrompt`` instances or raw
    ``(x, y, label)`` tuples.
    """
    parsed: List[PointPrompt] = []
    for p in points:
        if isinstance(p, PointPrompt):
            parsed.append(p)
        else:
            x, y, lab = p
            parsed.append(PointPrompt(x=float(x), y=float(y), label=int(lab)))
    return ManualSelection(points=parsed, obj_id=obj_id)
