"""
Transform to inject geometry-only queries into the Stage 3 training pipeline.

By adding queries with ``query_text="geometric"`` and a single target object,
we force the model to rely on the geometry encoder (box prompt → cross-attention
with FPN image features) rather than text for detection.  The frozen geometry
encoder's cross-attention provides gradient signal back to the trainable image
encoder, keeping the FPN feature space compatible with SAM3's geometry pathway.

These queries are consumed by ``RandomGeometricInputsAPI`` which samples a
(noised) box from the object's GT mask and writes it into ``query.input_bbox``.
"""

import random
from typing import Optional

import torch

from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    InferenceMetadata,
)


class AddGeometricQueries:
    """Append geometry-only queries for randomly selected objects.

    For each image in the datapoint, with probability *geo_prob*, up to
    *max_geo_queries* objects that carry a decoded mask are selected and
    turned into additional ``FindQuery`` entries whose ``query_text`` is set
    to *geometric_query_str*.  A downstream ``RandomGeometricInputsAPI``
    transform is expected to fill in the actual box prompt.

    Parameters
    ----------
    geo_prob : float
        Per-image probability of adding *any* geometry queries.
    max_geo_queries : int
        Maximum number of geometry-only queries to add per image.
    min_mask_area : float
        Minimum mask area (in pixels) for an object to be eligible.
    geometric_query_str : str
        The magic string that ``RandomGeometricInputsAPI`` looks for.
    """

    def __init__(
        self,
        geo_prob: float = 0.5,
        max_geo_queries: int = 8,
        min_mask_area: float = 64.0,
        geometric_query_str: str = "geometric",
    ):
        self.geo_prob = geo_prob
        self.max_geo_queries = max_geo_queries
        self.min_mask_area = min_mask_area
        self.geometric_query_str = geometric_query_str

    def _is_valid_object(self, obj) -> bool:
        seg = obj.segment
        if seg is None or obj.is_crowd:
            return False
        if isinstance(seg, torch.Tensor):
            return seg.sum().item() >= self.min_mask_area
        return True

    def __call__(self, datapoint: Datapoint, **kwargs) -> Datapoint:
        if random.random() > self.geo_prob:
            return datapoint

        image = datapoint.images[0]
        candidates = [
            (idx, obj)
            for idx, obj in enumerate(image.objects)
            if self._is_valid_object(obj)
        ]
        if not candidates:
            return datapoint

        n_geo = min(len(candidates), self.max_geo_queries)
        selected = random.sample(candidates, n_geo)

        for obj_idx, obj in selected:
            geo_query = FindQueryLoaded(
                query_text=self.geometric_query_str,
                image_id=0,
                object_ids_output=[obj_idx],
                is_exhaustive=True,
                query_processing_order=0,
                input_bbox=None,
                input_bbox_label=torch.ones(1, dtype=torch.long),
                input_points=None,
                semantic_target=None,
                is_pixel_exhaustive=True,
                inference_metadata=InferenceMetadata(
                    coco_image_id=-1,
                    original_image_id=-1,
                    original_category_id=-1,
                    original_size=image.size,
                    object_id=getattr(obj, "object_id", -1),
                    frame_index=0,
                ),
            )
            datapoint.find_queries.append(geo_query)

        return datapoint
