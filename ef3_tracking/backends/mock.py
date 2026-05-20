"""In-memory fake backend used by the unit tests and the offline demo.

It implements the same surface as ``Sam3EdgeBackend`` but synthesises masks
deterministically so the pipeline can be exercised without GPUs, model
weights, or even the SAM3 import.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from ..tracker import BackendProtocol


@dataclass
class _Session:
    session_id: str
    source: str
    num_frames: int = 8
    width: int = 320
    height: int = 240
    text_prompt: Optional[str] = None
    objects: Dict[int, Dict[str, Any]] = field(default_factory=dict)


class MockBackend(BackendProtocol):
    """Tiny synthetic backend.

    On ``add_text_prompt`` it materialises one object whose mask is a circle in
    the middle of the frame; on ``add_geometric_prompt`` it materialises a mask
    around the supplied points / box. ``propagate`` then drifts the mask one
    pixel to the right per frame, which is enough motion for tests to verify
    bbox extraction, mask propagation, and writer output.
    """

    def __init__(
        self,
        *,
        num_frames: int = 8,
        width: int = 320,
        height: int = 240,
        drift_per_frame: int = 1,
    ) -> None:
        self.num_frames = num_frames
        self.width = width
        self.height = height
        self.drift_per_frame = drift_per_frame
        self._sessions: Dict[str, _Session] = {}
        self.calls: List[tuple[str, Dict[str, Any]]] = []

    def start_session(self, source: str) -> str:
        sid = str(uuid.uuid4())
        self._sessions[sid] = _Session(
            session_id=sid,
            source=source,
            num_frames=self.num_frames,
            width=self.width,
            height=self.height,
        )
        self.calls.append(("start_session", {"source": source, "session_id": sid}))
        return sid

    def add_text_prompt(self, session_id: str, frame_idx: int, text: str) -> Dict[str, Any]:
        session = self._get(session_id)
        session.text_prompt = text
        obj_id = 0
        # Place a single round detection in the centre of the frame.
        cx, cy, r = session.width // 2, session.height // 2, min(session.width, session.height) // 6
        session.objects = {obj_id: {"cx": cx, "cy": cy, "r": r, "score": 0.95}}
        mask = _circle_mask(session.height, session.width, cx, cy, r)
        self.calls.append(("add_text_prompt", {"session_id": session_id, "frame_idx": frame_idx, "text": text}))
        return {
            "frame_index": frame_idx,
            "outputs": {obj_id: {"mask": mask, "score": 0.95}},
        }

    def add_geometric_prompt(
        self,
        session_id: str,
        frame_idx: int,
        obj_id: int,
        points: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        bounding_boxes: Optional[List[List[float]]] = None,
        bounding_box_labels: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        session = self._get(session_id)
        if bounding_boxes:
            # bounding_boxes are normalized xywh
            bx, by, bw, bh = bounding_boxes[0]
            cx = int((bx + bw / 2) * session.width)
            cy = int((by + bh / 2) * session.height)
            r = int(max(bw * session.width, bh * session.height) / 2)
        elif points:
            # Use the centroid of positive points
            pos = [p for p, lab in zip(points, point_labels or [1] * len(points)) if lab == 1]
            if not pos:
                pos = points
            cx = int(sum(p[0] for p in pos) / len(pos))
            cy = int(sum(p[1] for p in pos) / len(pos))
            r = max(20, min(session.width, session.height) // 8)
        else:
            cx, cy, r = session.width // 2, session.height // 2, 30

        session.objects[obj_id] = {"cx": cx, "cy": cy, "r": r, "score": 0.99}
        mask = _circle_mask(session.height, session.width, cx, cy, r)
        self.calls.append(
            (
                "add_geometric_prompt",
                {
                    "session_id": session_id,
                    "frame_idx": frame_idx,
                    "obj_id": obj_id,
                    "points": points,
                    "point_labels": point_labels,
                    "bounding_boxes": bounding_boxes,
                    "bounding_box_labels": bounding_box_labels,
                },
            )
        )
        return {
            "frame_index": frame_idx,
            "outputs": {obj_id: {"mask": mask, "score": 0.99}},
        }

    def propagate(self, session_id: str) -> Iterator[Dict[str, Any]]:
        session = self._get(session_id)
        self.calls.append(("propagate", {"session_id": session_id}))
        for f in range(session.num_frames):
            outputs: Dict[int, Dict[str, Any]] = {}
            for obj_id, obj in session.objects.items():
                cx = min(session.width - 1, obj["cx"] + f * self.drift_per_frame)
                mask = _circle_mask(session.height, session.width, cx, obj["cy"], obj["r"])
                outputs[obj_id] = {"mask": mask, "score": obj["score"]}
            yield {"frame_index": f, "outputs": outputs}

    def close_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self.calls.append(("close_session", {"session_id": session_id}))

    def _get(self, session_id: str) -> _Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise RuntimeError(f"unknown session: {session_id}")
        return session


def _circle_mask(height: int, width: int, cx: int, cy: int, r: int) -> np.ndarray:
    ys, xs = np.ogrid[:height, :width]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
    return mask.astype(bool)
