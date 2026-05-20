"""Edge-device runtime configuration.

`EdgeConfig` collects all the knobs that change how the underlying SAM3 video
predictor runs on a constrained device like the Jetson Orin AGX. Defaults are
picked to fit Orin AGX 32GB while still tracking 1080p video at a usable rate.

The config is a plain dataclass: no I/O, no torch imports, no model loading.
That keeps it cheap to construct and trivial to unit-test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

Precision = Literal["fp32", "fp16", "bf16"]
BackboneType = Literal["efficientvit", "tinyvit", "vit"]


@dataclass
class EdgeConfig:
    """Runtime configuration for edge-device tracking."""

    backbone_type: BackboneType = "efficientvit"
    model_name: str = "b0"
    text_encoder_type: Optional[str] = "MobileCLIP-S0"
    text_encoder_context_length: int = 16

    precision: Precision = "fp16"
    device: Optional[str] = None
    gpu_ids: Optional[List[int]] = None

    frame_stride: int = 1
    max_resolution: Optional[int] = None
    max_num_objects: int = 8

    checkpoint_path: Optional[str] = None
    bpe_path: Optional[str] = None
    load_from_hf: bool = False

    confidence_threshold: float = 0.4

    extra_model_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {self.frame_stride}")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}"
            )
        if self.max_resolution is not None and self.max_resolution < 64:
            raise ValueError(
                f"max_resolution must be >= 64 if set, got {self.max_resolution}"
            )
        if self.max_num_objects < 1:
            raise ValueError(
                f"max_num_objects must be >= 1, got {self.max_num_objects}"
            )

    @classmethod
    def for_orin_agx(cls) -> "EdgeConfig":
        """Recommended preset for NVIDIA Jetson Orin AGX (32GB)."""
        return cls(
            backbone_type="efficientvit",
            model_name="b0",
            text_encoder_type="MobileCLIP-S0",
            text_encoder_context_length=16,
            precision="fp16",
            frame_stride=1,
            max_resolution=720,
            max_num_objects=8,
        )

    @classmethod
    def for_orin_nx(cls) -> "EdgeConfig":
        """Tighter preset for NVIDIA Jetson Orin NX (8GB / 16GB)."""
        return cls(
            backbone_type="efficientvit",
            model_name="b0",
            text_encoder_type="MobileCLIP-S0",
            text_encoder_context_length=16,
            precision="fp16",
            frame_stride=2,
            max_resolution=512,
            max_num_objects=4,
        )

    @classmethod
    def for_cpu(cls) -> "EdgeConfig":
        """CPU fallback. Slow, but useful for debugging and CI."""
        return cls(
            backbone_type="efficientvit",
            model_name="b0",
            text_encoder_type="MobileCLIP-S0",
            text_encoder_context_length=16,
            precision="fp32",
            device="cpu",
            frame_stride=4,
            max_resolution=384,
            max_num_objects=2,
        )
