"""
Stage 3 image text-mask pair data engine utilities.
"""

from stage3.data_engine.annotations import (
    DEFAULT_MODEL_NAME,
    GROUPED_SCHEMA_VERSION,
    MAX_LABEL_WORDS,
    PROMPT_VERSION,
    RAW_SCHEMA_VERSION,
    RawMaskLabelRecord,
    area_to_fraction,
    bbox_xywh_to_normalized_xywh,
    bbox_xywh_to_xyxy,
    build_qwen_labeling_messages,
    choose_prompt_bbox,
    disambiguate_duplicate_labels,
    extract_json_object,
    is_generic_label,
    normalize_label,
    parse_model_json_response,
    phrase_word_count,
    visualize_annotation_example,
)

__all__ = [
    "DEFAULT_MODEL_NAME",
    "GROUPED_SCHEMA_VERSION",
    "MAX_LABEL_WORDS",
    "PROMPT_VERSION",
    "RAW_SCHEMA_VERSION",
    "RawMaskLabelRecord",
    "area_to_fraction",
    "bbox_xywh_to_normalized_xywh",
    "bbox_xywh_to_xyxy",
    "build_qwen_labeling_messages",
    "choose_prompt_bbox",
    "disambiguate_duplicate_labels",
    "extract_json_object",
    "is_generic_label",
    "normalize_label",
    "parse_model_json_response",
    "phrase_word_count",
    "visualize_annotation_example",
]
