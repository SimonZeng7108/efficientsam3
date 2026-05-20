"""Text-prompt tracker.

The user supplies a natural-language description -- "the red car", "a person
in a yellow jacket" -- and the ViT text encoder grounds it on frame 0. The
predictor then propagates each grounded instance through the whole video.

This is the heavier of the two trackers (the text encoder must run at least
once per session), but on Orin AGX it still hits real-time at 720p with the
MobileCLIP-S0 student encoder.
"""

from __future__ import annotations

from typing import List, Optional

from .prompts import TextPrompt
from .tracker import BaseTracker, BackendProtocol, TrackedObject, parse_backend_outputs


class TextTracker(BaseTracker):
    """Track everything matching a text prompt across a video."""

    def __init__(self, backend: BackendProtocol) -> None:
        super().__init__(backend, label=None)
        self._prompt: Optional[TextPrompt] = None

    @property
    def prompt(self) -> Optional[TextPrompt]:
        return self._prompt

    def set_prompt(
        self, prompt: str | TextPrompt, frame_idx: int = 0
    ) -> List[TrackedObject]:
        """Attach a text prompt to the open session.

        Returns the objects the model already detected on the seed frame, so a
        caller can sanity-check the grounding before paying for propagation.
        """
        sid = self._require_session()
        if isinstance(prompt, str):
            prompt = TextPrompt(text=prompt)
        self._prompt = prompt
        self._label = prompt.normalized
        resp = self._backend.add_text_prompt(
            session_id=sid, frame_idx=frame_idx, text=prompt.text
        )
        return parse_backend_outputs(resp.get("outputs", {}), label=self._label)
