"""Base tracker abstraction and result dataclasses.

The base class hides the SAM3 video predictor behind a small surface area:

    start_session(source) -> session
    add_prompt(...)       -> per-prompt action
    propagate()           -> iterator of (frame_idx, [TrackedObject])
    close_session()

Concrete subclasses live in ``manual_tracker.py`` and ``text_tracker.py``. The
underlying *backend* is injected, which lets the tests use a tiny fake instead
of importing the heavy ML stack.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np


@dataclass
class TrackedObject:
    """One object's tracked state on a single frame."""

    obj_id: int
    mask: Optional[np.ndarray] = None
    score: Optional[float] = None
    box_xyxy: Optional[Tuple[int, int, int, int]] = None
    label: Optional[str] = None

    @property
    def is_present(self) -> bool:
        """True if the object has a non-empty mask on this frame."""
        return self.mask is not None and bool(self.mask.any())


@dataclass
class TrackingResult:
    """Aggregated per-frame tracking output for a whole video."""

    frames: Dict[int, List[TrackedObject]] = field(default_factory=dict)
    width: int = 0
    height: int = 0
    fps: float = 30.0

    def add(self, frame_idx: int, objects: List[TrackedObject]) -> None:
        self.frames[frame_idx] = objects

    def num_objects(self) -> int:
        ids: set[int] = set()
        for objs in self.frames.values():
            for o in objs:
                ids.add(o.obj_id)
        return len(ids)

    def num_frames(self) -> int:
        return len(self.frames)

    def object_track(self, obj_id: int) -> List[Tuple[int, TrackedObject]]:
        """Return ``(frame_idx, TrackedObject)`` pairs ordered by frame."""
        result = []
        for fidx in sorted(self.frames.keys()):
            for obj in self.frames[fidx]:
                if obj.obj_id == obj_id:
                    result.append((fidx, obj))
                    break
        return result


class BackendProtocol(abc.ABC):
    """Adapter between the trackers and the SAM3 video predictor.

    Real implementations wrap ``Sam3VideoPredictorMultiGPU``. Tests use a
    lightweight fake that returns simple shapes -- this is the seam that keeps
    the heavy ML imports out of the unit-test path.
    """

    @abc.abstractmethod
    def start_session(self, source: str) -> str: ...

    @abc.abstractmethod
    def add_text_prompt(
        self, session_id: str, frame_idx: int, text: str
    ) -> Dict[str, Any]: ...

    @abc.abstractmethod
    def add_geometric_prompt(
        self,
        session_id: str,
        frame_idx: int,
        obj_id: int,
        points: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        bounding_boxes: Optional[List[List[float]]] = None,
        bounding_box_labels: Optional[List[int]] = None,
    ) -> Dict[str, Any]: ...

    @abc.abstractmethod
    def propagate(self, session_id: str) -> Iterator[Dict[str, Any]]: ...

    @abc.abstractmethod
    def close_session(self, session_id: str) -> None: ...


def _mask_xyxy(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    mask_bool = np.asarray(mask).astype(bool)
    if not mask_bool.any():
        return None
    ys, xs = np.where(mask_bool)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def parse_backend_outputs(
    outputs: Dict[int, Dict[str, Any]] | Dict[Any, Any],
    label: Optional[str] = None,
) -> List[TrackedObject]:
    """Convert the backend's per-frame output dict into a list of TrackedObjects.

    The SAM3 video predictor yields entries that look like::

        {obj_id: {"mask": np.ndarray | torch.Tensor, "score": float, ...}}

    This helper is tolerant: missing fields turn into ``None``, tensors get
    pulled to numpy, and the bbox is derived from the mask when absent.
    """
    parsed: List[TrackedObject] = []
    if not outputs:
        return parsed
    for obj_id, payload in outputs.items():
        if not isinstance(payload, dict):
            continue
        mask = payload.get("mask")
        if mask is not None:
            mask = _to_numpy(mask)
            if mask.ndim > 2:
                mask = np.squeeze(mask)
        score = payload.get("score")
        if score is not None:
            score = float(score)
        box = payload.get("box_xyxy") or payload.get("box")
        if box is None and mask is not None:
            box = _mask_xyxy(mask)
        parsed.append(
            TrackedObject(
                obj_id=int(obj_id),
                mask=mask,
                score=score,
                box_xyxy=tuple(int(v) for v in box) if box is not None else None,
                label=label,
            )
        )
    return parsed


def _to_numpy(x: Any) -> np.ndarray:
    """Convert tensors / lists / arrays to a plain numpy array."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)


class BaseTracker(abc.ABC):
    """Common machinery shared by the two tracker variants."""

    def __init__(self, backend: BackendProtocol, *, label: Optional[str] = None) -> None:
        self._backend = backend
        self._label = label
        self._session_id: Optional[str] = None

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    def open(self, source: str) -> str:
        if self._session_id is not None:
            raise RuntimeError("session already open; call close() first")
        self._session_id = self._backend.start_session(source)
        return self._session_id

    def close(self) -> None:
        if self._session_id is not None:
            try:
                self._backend.close_session(self._session_id)
            finally:
                self._session_id = None

    def _require_session(self) -> str:
        if self._session_id is None:
            raise RuntimeError("no open session; call open() first")
        return self._session_id

    def propagate(self) -> Iterator[Tuple[int, List[TrackedObject]]]:
        """Yield ``(frame_idx, [TrackedObject])`` for every propagated frame."""
        sid = self._require_session()
        for resp in self._backend.propagate(sid):
            fidx = int(resp["frame_index"])
            yield fidx, parse_backend_outputs(resp.get("outputs", {}), label=self._label)

    def collect(self, *, width: int, height: int, fps: float = 30.0) -> TrackingResult:
        """Drain `propagate()` into a `TrackingResult`."""
        result = TrackingResult(width=width, height=height, fps=fps)
        for fidx, objs in self.propagate():
            result.add(fidx, objs)
        return result

    def __enter__(self) -> "BaseTracker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
