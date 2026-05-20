"""Real SAM3 / EfficientSAM3 backend for the trackers.

This module is the only place that imports the SAM3 stack -- everything else
in ``ef3_tracking`` works against the ``BackendProtocol``. Importing the SAM3
package happens lazily inside ``build()`` so users on edge devices without the
full dependency set can still use the package's testing layer.

Edge-device knobs honored from ``EdgeConfig``:

    * precision     -> torch dtype + autocast
    * gpu_ids       -> ``gpus_to_use`` passed to the SAM3 predictor
    * device="cpu"  -> forces CPU-only execution
    * max_resolution -> downscaled video frames (consumer side)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Optional

from ..config import EdgeConfig
from ..tracker import BackendProtocol


class Sam3EdgeBackend(BackendProtocol):
    """Thin adapter that forwards to ``Sam3VideoPredictorMultiGPU``."""

    def __init__(self, predictor: Any) -> None:
        self._predictor = predictor

    @property
    def predictor(self) -> Any:
        return self._predictor

    def start_session(self, source: str) -> str:
        response = self._predictor.handle_request(
            {"type": "start_session", "resource_path": source}
        )
        return response["session_id"]

    def add_text_prompt(self, session_id: str, frame_idx: int, text: str) -> Dict[str, Any]:
        return self._predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": frame_idx,
                "text": text,
            }
        )

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
        request: Dict[str, Any] = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": frame_idx,
            "obj_id": obj_id,
        }
        if points is not None:
            request["points"] = points
            request["point_labels"] = point_labels
        if bounding_boxes is not None:
            request["bounding_boxes"] = bounding_boxes
            request["bounding_box_labels"] = bounding_box_labels
        return self._predictor.handle_request(request)

    def propagate(self, session_id: str) -> Iterator[Dict[str, Any]]:
        yield from self._predictor.handle_stream_request(
            {"type": "propagate_in_video", "session_id": session_id}
        )

    def close_session(self, session_id: str) -> None:
        try:
            self._predictor.handle_request(
                {"type": "close_session", "session_id": session_id}
            )
        except Exception:
            # Closing is best-effort; the predictor may already be torn down.
            pass


def build_edge_backend(config: EdgeConfig) -> Sam3EdgeBackend:
    """Construct a real SAM3 video predictor configured for an edge device.

    Heavy imports happen inside this function so ``ef3_tracking`` can be used
    on machines without the full SAM3 environment.
    """
    import torch

    _apply_precision_flags(config.precision)

    from sam3.model_builder import build_efficientsam3_video_predictor

    gpus = config.gpu_ids
    if gpus is None and torch.cuda.is_available() and config.device != "cpu":
        gpus = [torch.cuda.current_device()]

    if config.bpe_path is None:
        bpe_path = _default_bpe_path()
    else:
        bpe_path = config.bpe_path

    predictor_kwargs: Dict[str, Any] = {
        "checkpoint_path": config.checkpoint_path,
        "load_from_HF": config.load_from_hf,
        "bpe_path": bpe_path,
        "backbone_type": config.backbone_type,
        "model_name": config.model_name,
        "strict_state_dict_loading": False,
        "gpus_to_use": gpus,
    }
    if config.text_encoder_type:
        predictor_kwargs["text_encoder_type"] = config.text_encoder_type
        predictor_kwargs["text_encoder_context_length"] = config.text_encoder_context_length
    predictor_kwargs.update(config.extra_model_kwargs)

    predictor = build_efficientsam3_video_predictor(**predictor_kwargs)
    return Sam3EdgeBackend(predictor)


def _apply_precision_flags(precision: str) -> None:
    """Switch on TF32 / matmul knobs that benefit Orin and Ampere-class GPUs."""
    try:
        import torch
    except ImportError:
        return
    if precision in ("fp16", "bf16") and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True


def _default_bpe_path() -> Optional[str]:
    """Best-effort lookup of the BPE vocab bundled with the SAM3 repo."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "..", "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz"),
        os.path.join(here, "..", "..", "sam3", "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz"),
    ]
    for path in candidates:
        path = os.path.normpath(path)
        if os.path.isfile(path):
            return path
    return None
